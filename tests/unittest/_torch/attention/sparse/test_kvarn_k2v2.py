# SPDX-License-Identifier: Apache-2.0
"""KVarN k2v2 dense MLA latent coverage.

These tests stay CPU-only so they can run on protected B200 hosts without
allocating GPU memory. They exercise the same pack/layout/dequant contracts the
SMC-SD decode path depends on when ``mla_latent_kv_dtype='kvarn_k2v2'``.
"""

from types import SimpleNamespace

import torch

from tensorrt_llm._torch.attention_backend.sparse.dsa import DSATrtllmAttention
from tensorrt_llm._torch.attention_backend.sparse.kvarn_backend import (
    KVarNLatentPool,
    parse_kvarn_dtype,
)
from tensorrt_llm._torch.attention_backend.sparse.kvarn_core import (
    _pack_lowbit,
    _unpack_lowbit,
)
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().reshape(1, -1), b.float().reshape(1, -1)
    ).item()


def test_lowbit_pack_unpack_k2_odd_smc_rows():
    # Odd row count mirrors SMC target verify batches such as M=25; packing must
    # depend only on the latent channel axis, not on an 8-aligned batch/M shape.
    q = (torch.arange(25 * 512, dtype=torch.uint8).reshape(25, 512) % 4)

    packed = _pack_lowbit(q, bits=2)
    assert packed.shape == (25, 128)
    assert packed.stride(-1) == 1

    unpacked = _unpack_lowbit(packed, bits=2, orig_last=512)
    assert torch.equal(unpacked, q)


def test_kvarn_k2v2_config_is_dense_mla_only():
    cfg = parse_kvarn_dtype(
        "kvarn_k2v2", kv_lora_rank=512, qk_rope_head_dim=64, iters=2
    )

    assert cfg.name == "kvarn_k2v2"
    assert cfg.ckv_bits == 2
    assert cfg.pe_bits == 2
    assert cfg.bits_per_elem(group=64) < 2.4

    sparse_cfg = DeepSeekSparseAttentionConfig(
        indexer_k_dtype="fp8",
        mla_latent_kv_dtype="kvarn_k2v2",
        mla_latent_kv_amortize=True,
    )
    assert sparse_cfg.indexer_k_dtype == "fp8"
    assert sparse_cfg.mla_latent_kv_dtype == "kvarn_k2v2"
    assert sparse_cfg.mla_latent_kv_amortize is True


def test_kvarn_latent_pool_k2v2_roundtrip_cpu():
    torch.manual_seed(20260606)
    group = 64
    cfg = parse_kvarn_dtype(
        "kvarn_k2v2", kv_lora_rank=512, qk_rope_head_dim=64, iters=2
    )
    pool = KVarNLatentPool(num_blocks=4, group=group, cfg=cfg, device=torch.device("cpu"))

    # Smooth-ish latent data gives a deterministic fidelity floor without making
    # this CPU test spend time on large random outliers.
    ckv = torch.randn(group, cfg.kv_lora_rank, dtype=torch.float16) * 0.35
    k_pe = torch.randn(group, cfg.qk_rope_head_dim, dtype=torch.float16) * 0.20

    pool.store_block(2, ckv, k_pe)
    assert bool(pool.valid[2])
    assert int(pool.commit_gen[2]) == 1
    assert pool.bytes_per_block == cfg.packed_bytes(group)

    ckv_rt, kpe_rt = pool.load_block(2)
    assert ckv_rt.shape == ckv.shape
    assert kpe_rt.shape == k_pe.shape
    # 2-bit KVarN is lossy; this catches layout/scale/pack regressions while
    # allowing the expected quantization error.
    assert _cosine(ckv_rt, ckv) > 0.80
    assert _cosine(kpe_rt, k_pe) > 0.80

    ckv_b, kpe_b = pool.load_blocks(torch.tensor([2], dtype=torch.long))
    assert torch.equal(ckv_b[0], ckv_rt)
    assert torch.equal(kpe_b[0], kpe_rt)


class _FakeNonLocalKVarNManager:
    kvarn_enabled = True
    tokens_per_block = 64
    kvarn_cfg = SimpleNamespace(sink_tokens=0)

    def __init__(self):
        self.pool_queries = 0

    def get_kvarn_latent_pool(self, layer_idx):
        self.pool_queries += 1
        return None

    def get_buffers(self, *args, **kwargs):
        raise AssertionError("non-local KVarN path must not read dense buffers")


class _FakeKVarNMetadata:
    def __init__(self, mgr):
        self.kv_cache_manager = mgr


def test_kvarn_commit_and_restore_noop_for_non_local_layers(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    mgr = _FakeNonLocalKVarNManager()
    metadata = _FakeKVarNMetadata(mgr)
    attn = DSATrtllmAttention.__new__(DSATrtllmAttention)
    attn.layer_idx = 17

    DSATrtllmAttention.kvarn_commit_full_blocks(attn, metadata, is_generation=False)
    DSATrtllmAttention.kvarn_restore_for_decode(attn, metadata)

    assert mgr.pool_queries == 2


class _FakeLocalKVarNPool:
    def __init__(self):
        self.num_blocks = 4
        self.valid = torch.zeros(4, dtype=torch.bool)
        self.valid[1] = True
        self.commit_gen = torch.zeros(4, dtype=torch.int64)
        self.commit_gen[1] = 3
        self.ckv = torch.arange(64 * 2, dtype=torch.float16).reshape(1, 64, 2)
        self.kpe = torch.arange(64, dtype=torch.float16).reshape(1, 64, 1)
        self.loaded_ids = None

    def load_blocks(self, block_ids):
        self.loaded_ids = block_ids.clone()
        assert block_ids.tolist() == [1]
        return self.ckv, self.kpe


class _FakeLocalKVarNManager:
    kvarn_enabled = True
    tokens_per_block = 64
    kvarn_amortize_restore = False
    kvarn_cfg = SimpleNamespace(kv_lora_rank=2)

    def __init__(self):
        self.pool = _FakeLocalKVarNPool()
        self.buf = torch.zeros(4, 1, 64, 1, 3, dtype=torch.float16)

    def get_kvarn_latent_pool(self, layer_idx):
        assert layer_idx == 2
        return self.pool

    def get_buffers(self, layer_idx, kv_layout="NHD"):
        assert layer_idx == 2
        assert kv_layout == "NHD"
        return self.buf


class _FakeLocalKVarNMetadata:
    num_contexts = 0
    num_generations = 1
    kv_lens_runtime = torch.tensor([64], dtype=torch.int32)
    block_table = torch.tensor([[1, -1]], dtype=torch.int32)

    def __init__(self, mgr):
        self.kv_cache_manager = mgr


def test_kvarn_restore_writes_local_layer_main_pool(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    mgr = _FakeLocalKVarNManager()
    metadata = _FakeLocalKVarNMetadata(mgr)
    attn = DSATrtllmAttention.__new__(DSATrtllmAttention)
    attn.layer_idx = 2
    attn._kvarn_restored_gen = None

    DSATrtllmAttention.kvarn_restore_for_decode(attn, metadata)

    assert torch.equal(mgr.pool.loaded_ids, torch.tensor([1], dtype=torch.long))
    assert torch.equal(mgr.buf[1, 0, :, 0, :2], mgr.pool.ckv[0])
    assert torch.equal(mgr.buf[1, 0, :, 0, 2:], mgr.pool.kpe[0])

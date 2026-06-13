# SPDX-License-Identifier: Apache-2.0
"""KVarN k2v2 dense MLA latent coverage.

These tests stay CPU-only so they can run on protected B200 hosts without
allocating GPU memory. They exercise the same pack/layout/dequant contracts the
SMC-SD decode path depends on when ``mla_latent_kv_dtype='kvarn_k2v2'``.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.dsa import DSATrtllmAttention
from tensorrt_llm._torch.attention_backend.sparse.kvarn_backend import (
    KVARN_BDR_HISPARSE_LAYOUT,
    KVARN_LEGACY_SIDEPOOL_LAYOUT,
    KVarNBDRSourcePool,
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


def test_kvarn_k2v2_hisparse_bdr_layout_is_not_legacy_sidepool():
    cfg = parse_kvarn_dtype(
        "kvarn_k2v2", kv_lora_rank=512, qk_rope_head_dim=64, iters=2
    )

    layout = cfg.hisparse_bdr_layout(group=64)

    assert layout.name == KVARN_BDR_HISPARSE_LAYOUT
    assert layout.ckv_bits == 2
    assert layout.requested_pe_bits == 2
    assert layout.pe_storage_bits == 8
    assert layout.num_subblocks == 4
    assert layout.ckv_packed_bytes_per_token == 128
    assert layout.ckv_scale_zp_bytes_per_token == 16
    assert layout.pe_payload_bytes_per_token == 64
    assert layout.packed_bytes_per_block == 64 * (128 + 16 + 64)
    assert layout.field_offsets == {
        "ckv_q": (0, 8192),
        "ckv_scale_zp": (8192, 9216),
        "pe_byte": (9216, 13312),
    }
    assert layout.packed_bytes_per_block != cfg.packed_bytes(64)


def test_kvarn_bdr_source_pool_fragments_and_recycle_cpu():
    cfg = parse_kvarn_dtype(
        "kvarn_k2v2", kv_lora_rank=512, qk_rope_head_dim=64, iters=2
    )
    layout = cfg.hisparse_bdr_layout(group=64)
    pool = KVarNBDRSourcePool(num_blocks=4, layout=layout,
                              device=torch.device("cpu"))

    assert pool.storage_layout_name == KVARN_BDR_HISPARSE_LAYOUT
    assert pool.bytes_per_block == 13312
    dst_ptrs, dst_sizes = pool.record_destination_fragments([2])
    assert dst_ptrs.tolist() == [pool.store.data_ptr() +
                                 2 * pool.bytes_per_block]
    assert dst_sizes.tolist() == [pool.bytes_per_block]
    with pytest.raises(RuntimeError, match="uncommitted"):
        pool.packed_source_fragments([2])

    record = torch.arange(pool.bytes_per_block,
                          dtype=torch.int64).remainder(256).to(torch.uint8)
    pool.commit_record_bytes(2, record)
    assert bool(pool.valid[2])
    assert int(pool.commit_gen[2]) == 1
    src_ptrs, src_sizes = pool.packed_source_fragments([2])
    assert src_ptrs.tolist() == dst_ptrs.tolist()
    assert src_sizes.tolist() == dst_sizes.tolist()

    pool.invalidate_blocks([2])
    assert not bool(pool.valid[2])
    assert not bool(pool.valid_host[2])
    with pytest.raises(RuntimeError, match="uncommitted"):
        pool.packed_source_fragments([2])


def test_kvarn_latent_pool_k2v2_roundtrip_cpu():
    torch.manual_seed(20260606)
    group = 64
    cfg = parse_kvarn_dtype(
        "kvarn_k2v2", kv_lora_rank=512, qk_rope_head_dim=64, iters=2
    )
    pool = KVarNLatentPool(num_blocks=4, group=group, cfg=cfg, device=torch.device("cpu"))
    assert pool.storage_layout_name == KVARN_LEGACY_SIDEPOOL_LAYOUT

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

    src_ptrs, src_sizes = pool.packed_source_fragments([2])
    assert src_ptrs.tolist() == [pool.store.data_ptr() +
                                 2 * pool.bytes_per_block]
    assert src_sizes.tolist() == [pool.bytes_per_block]
    with pytest.raises(RuntimeError, match="uncommitted"):
        pool.packed_source_fragments([1])


def test_hisparse_direct_to_host_rejects_legacy_kvarn_source_layout():
    from tensorrt_llm._torch.attention_backend.sparse.dsa import DSACacheManager

    mgr = DSACacheManager.__new__(DSACacheManager)
    mgr.kvarn_latent_pool_per_layer = [object()]
    mgr.kvarn_hisparse_source_layout = KVARN_LEGACY_SIDEPOOL_LAYOUT

    with pytest.raises(NotImplementedError, match=KVARN_BDR_HISPARSE_LAYOUT):
        DSACacheManager.kvarn_packed_source_fragments(mgr, [0], [0])


def test_hisparse_direct_to_host_uses_bdr_source_pool():
    from tensorrt_llm._torch.attention_backend.sparse.dsa import DSACacheManager

    cfg = parse_kvarn_dtype(
        "kvarn_k2v2", kv_lora_rank=512, qk_rope_head_dim=64, iters=2
    )
    pool = KVarNBDRSourcePool(num_blocks=4,
                              layout=cfg.hisparse_bdr_layout(group=64),
                              device=torch.device("cpu"))
    pool.commit_record_bytes(1, torch.ones(pool.bytes_per_block,
                                           dtype=torch.uint8))
    mgr = DSACacheManager.__new__(DSACacheManager)
    mgr.kvarn_latent_pool_per_layer = [object()]
    mgr.kvarn_hisparse_bdr_pool_per_layer = [pool]
    mgr.kvarn_hisparse_source_layout = KVARN_BDR_HISPARSE_LAYOUT
    mgr.layer_offsets = {5: 0}

    ptrs, sizes = DSACacheManager.kvarn_packed_source_fragments(
        mgr, [5], [1])

    assert ptrs.tolist() == [pool.store.data_ptr() + pool.bytes_per_block]
    assert sizes.tolist() == [pool.bytes_per_block]


def test_hisparse_bdr_writer_destination_and_commit_hooks():
    from tensorrt_llm._torch.attention_backend.sparse.dsa import DSACacheManager

    cfg = parse_kvarn_dtype(
        "kvarn_k2v2", kv_lora_rank=512, qk_rope_head_dim=64, iters=2
    )
    layout = cfg.hisparse_bdr_layout(group=64)
    pools = [
        KVarNBDRSourcePool(num_blocks=4, layout=layout,
                           device=torch.device("cpu"))
        for _ in range(2)
    ]
    mgr = DSACacheManager.__new__(DSACacheManager)
    mgr.kvarn_latent_pool_per_layer = [object()]
    mgr.kvarn_hisparse_bdr_pool_per_layer = pools
    mgr.kvarn_hisparse_source_layout = KVARN_BDR_HISPARSE_LAYOUT
    mgr.layer_offsets = {5: 0, 6: 1}

    ptrs, sizes = DSACacheManager.kvarn_bdr_record_destination_fragments(
        mgr, [5, 6], [1, 3])

    expected_ptrs = [
        pools[0].store.data_ptr() + pools[0].bytes_per_block,
        pools[0].store.data_ptr() + 3 * pools[0].bytes_per_block,
        pools[1].store.data_ptr() + pools[1].bytes_per_block,
        pools[1].store.data_ptr() + 3 * pools[1].bytes_per_block,
    ]
    assert ptrs.tolist() == expected_ptrs
    assert sizes.tolist() == [pools[0].bytes_per_block] * 4
    with pytest.raises(RuntimeError, match="uncommitted"):
        DSACacheManager.kvarn_packed_source_fragments(mgr, [5], [1])

    DSACacheManager.mark_kvarn_bdr_records_committed(mgr, [5, 6], [1, 3])

    src_ptrs, src_sizes = DSACacheManager.kvarn_packed_source_fragments(
        mgr, [5, 6], [1, 3])
    assert src_ptrs.tolist() == expected_ptrs
    assert src_sizes.tolist() == [pools[0].bytes_per_block] * 4


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
        self.restored_marked = None

    def load_blocks(self, block_ids):
        self.loaded_ids = block_ids.clone()
        assert block_ids.tolist() == [1]
        return self.ckv, self.kpe

    def mark_restored_host(self, block_ids):
        # Real pools mirror the restore epoch on host for the pre-replay
        # delta walk; record the call so the test can assert it fired.
        self.restored_marked = [int(b) for b in block_ids]


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


class _FakeCommitPool:
    def __init__(self, valid=False):
        self.valid = torch.zeros(4, dtype=torch.bool)
        self.valid[1] = bool(valid)
        self.store_calls = []

    def store_block(self, block_id, ckv, k_pe):
        self.store_calls.append(
            (int(block_id), tuple(ckv.shape), tuple(k_pe.shape)))
        self.valid[int(block_id)] = True


class _FakeBDRCommitPool:
    def __init__(self, valid=False):
        self.valid_host = np.zeros(4, dtype=bool)
        self.valid_host[1] = bool(valid)


class _FakeDualKVarNManager:
    kvarn_enabled = True
    tokens_per_block = 64
    kvarn_amortize_restore = False
    kvarn_hisparse_source_layout = KVARN_BDR_HISPARSE_LAYOUT
    kvarn_cfg = SimpleNamespace(sink_tokens=0, kv_lora_rank=2)

    def __init__(self, *, legacy_valid=False, bdr_valid=False):
        self.pool = _FakeCommitPool(valid=legacy_valid)
        self.bdr_pool = _FakeBDRCommitPool(valid=bdr_valid)
        self.buf = torch.zeros(4, 1, 64, 1, 3, dtype=torch.float16)
        self.bdr_store_calls = []

    def get_kvarn_latent_pool(self, layer_idx):
        assert layer_idx == 2
        return self.pool

    def get_kvarn_hisparse_bdr_pool(self, layer_idx):
        assert layer_idx == 2
        return self.bdr_pool

    def get_buffers(self, layer_idx, kv_layout="NHD"):
        assert layer_idx == 2
        assert kv_layout == "NHD"
        return self.buf

    def kvarn_store_block(self, layer_idx, block_id, ckv, k_pe):
        assert layer_idx == 2
        self.pool.store_block(block_id, ckv, k_pe)

    def kvarn_store_hisparse_bdr_block(self, layer_idx, block_id, latent_block):
        assert layer_idx == 2
        self.bdr_store_calls.append((int(block_id), tuple(latent_block.shape)))
        self.bdr_pool.valid_host[int(block_id)] = True


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
    assert mgr.pool.restored_marked == [1]


def test_kvarn_commit_full_blocks_populates_bdr_pool_when_legacy_already_valid():
    metadata = _FakeLocalKVarNMetadata(
        _FakeDualKVarNManager(legacy_valid=True, bdr_valid=False))
    attn = DSATrtllmAttention.__new__(DSATrtllmAttention)
    attn.layer_idx = 2

    DSATrtllmAttention.kvarn_commit_full_blocks(
        attn, metadata, is_generation=False)

    mgr = metadata.kv_cache_manager
    assert mgr.pool.store_calls == []
    assert mgr.bdr_store_calls == [(1, (64, 3))]
    assert bool(mgr.bdr_pool.valid_host[1])


def test_kvarn_commit_full_blocks_writes_legacy_and_bdr_records():
    metadata = _FakeLocalKVarNMetadata(
        _FakeDualKVarNManager(legacy_valid=False, bdr_valid=False))
    attn = DSATrtllmAttention.__new__(DSATrtllmAttention)
    attn.layer_idx = 2

    DSATrtllmAttention.kvarn_commit_full_blocks(
        attn, metadata, is_generation=False)

    mgr = metadata.kv_cache_manager
    assert mgr.pool.store_calls == [(1, (64, 2), (64, 1))]
    assert mgr.bdr_store_calls == [(1, (64, 3))]


def _smooth_latent(group: int, cfg, seed: int):
    torch.manual_seed(seed)
    ckv = torch.randn(group, cfg.kv_lora_rank, dtype=torch.float16) * 0.35
    k_pe = torch.randn(group, cfg.qk_rope_head_dim, dtype=torch.float16) * 0.20
    return ckv, k_pe


def test_kvarn_latent_pool_free_reuse_commit_restore_cpu():
    """Block-id recycle contract: free -> reuse -> commit -> restore.

    ``invalidate_blocks`` (the DSACacheManager free/rewind hook) must clear
    ``valid`` + host mirror and reset the restore epoch so that (1) the
    commit walk's idempotence check (keyed on ``valid``) re-commits the new
    owner -- ``commit_gen`` bumps then, the documented recycle contract --
    and (2) no restore path (full-scan keep mask ``pool.valid[cand]``, the
    amortized epoch compare, or the host delta filter) can write the dying
    owner's record over the new owner's fresh fp16 latent.
    """
    group = 64
    cfg = parse_kvarn_dtype(
        "kvarn_k2v2", kv_lora_rank=512, qk_rope_head_dim=64, iters=2
    )
    pool = KVarNLatentPool(
        num_blocks=4, group=group, cfg=cfg, device=torch.device("cpu")
    )
    old_ckv, old_kpe = _smooth_latent(group, cfg, seed=20260610)

    # Owner A commits block 2 and the decode walk restores it.
    pool.store_block(2, old_ckv, old_kpe)
    assert pool.stale_committed_host([2]) == [2]
    pool.mark_restored_host([2])
    assert pool.stale_committed_host([2]) == []
    old_rt, _ = pool.load_block(2)

    # Block 2 returns to the allocator and is recycled to owner B.
    pool.invalidate_blocks([2])
    assert not bool(pool.valid[2])  # full-scan keep mask skips it
    assert not bool(pool.valid_host[2])  # commit walk re-commits it
    assert int(pool.restored_gen_host[2]) == -1
    assert int(pool.commit_gen[2]) == 1  # monotonic: never reset
    assert int(pool.commit_gen_host[2]) == 1
    assert pool.stale_committed_host([2]) == []  # delta walk: no stale fire
    # Idempotent on double-free and a no-op on empty / never-committed ids.
    pool.invalidate_blocks([2])
    pool.invalidate_blocks([])
    pool.invalidate_blocks([0, 3])
    assert not pool.valid_host.any()

    # Owner B fills the recycled id and the commit walk re-commits it: the
    # content epoch advances past every recorded restore epoch.
    new_ckv, new_kpe = _smooth_latent(group, cfg, seed=20260611)
    pool.store_block(2, new_ckv, new_kpe)
    assert bool(pool.valid[2]) and bool(pool.valid_host[2])
    assert int(pool.commit_gen[2]) == 2
    assert int(pool.commit_gen_host[2]) == 2
    assert pool.stale_committed_host([2]) == [2]
    pool.mark_restored_host([2])
    assert pool.stale_committed_host([2]) == []

    # The restore dequantizes B's record; A's content is gone.
    new_rt, _ = pool.load_block(2)
    assert _cosine(new_rt, new_ckv) > 0.80
    assert _cosine(new_rt, old_rt) < 0.5
    assert _cosine(new_rt, old_ckv) < 0.5


def test_dsa_cache_manager_invalidate_fans_out_all_layer_pools():
    from tensorrt_llm._torch.attention_backend.sparse.dsa import DSACacheManager

    group = 64
    cfg = parse_kvarn_dtype(
        "kvarn_k2v2", kv_lora_rank=512, qk_rope_head_dim=64, iters=2
    )
    pools = [
        KVarNLatentPool(
            num_blocks=4, group=group, cfg=cfg, device=torch.device("cpu")
        )
        for _ in range(2)
    ]
    ckv, k_pe = _smooth_latent(group, cfg, seed=20260612)
    for p in pools:
        p.store_block(1, ckv, k_pe)
        p.store_block(3, ckv, k_pe)

    mgr = DSACacheManager.__new__(DSACacheManager)
    mgr.kvarn_latent_pool_per_layer = pools
    mgr.blocks_in_secondary_pool = 0
    mgr.num_blocks = 4

    # Out-of-range ids (stale tables / padding) are filtered, never raised.
    DSACacheManager._kvarn_invalidate_block_ids(mgr, [1, -1, 99])
    for p in pools:
        assert not bool(p.valid[1]) and not bool(p.valid_host[1])
        assert bool(p.valid[3]) and bool(p.valid_host[3])  # untouched

    # Host-offload posture: block ids are not pool slots; must not touch.
    mgr.blocks_in_secondary_pool = 8
    DSACacheManager._kvarn_invalidate_block_ids(mgr, [3])
    for p in pools:
        assert bool(p.valid[3]) and bool(p.valid_host[3])

# SPDX-License-Identifier: Apache-2.0
"""KVarN k2v2 dense MLA latent coverage.

These tests stay CPU-only so they can run on protected B200 hosts without
allocating GPU memory. They exercise the same pack/layout/dequant contracts the
SMC-SD decode path depends on when ``mla_latent_kv_dtype='kvarn_k2v2'``.
"""

import torch

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

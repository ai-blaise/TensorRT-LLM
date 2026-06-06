# SPDX-License-Identifier: Apache-2.0
"""KVarN GQA packed-record correctness coverage."""

import pytest


_TORCH = pytest.importorskip("torch")

from tensorrt_llm._torch.attention_backend.kvarn_gqa_attention import (  # noqa: E402
    _KVarNGQASidePool,
)
from tensorrt_llm._torch.attention_backend.kvarn_gqa import (  # noqa: E402
    KVarNGQAConfig,
    KVarNGQAPackedPool,
    _pack_bits_flat,
    _unpack_bits_flat,
    dequantize_gqa_tile,
    parse_kvarn_gqa_dtype,
    quantize_gqa_tile,
)


def test_kvarn_gqa_k2v2_layout_matches_huawei_record_contract():
    cfg = parse_kvarn_gqa_dtype("kvarn_k2v2_g128")

    assert cfg.k_packed_bytes == 4096
    assert cfg.v_packed_bytes == 4096
    assert cfg.k_scale_bytes == 768
    assert cfg.v_scale_bytes == 768
    assert cfg.tile_bytes == 9728
    assert cfg.tile_bytes_aligned == 9728
    assert cfg.bytes_per_token_slot == 76
    assert cfg.v_zp_offset + cfg.group * 2 == cfg.tile_bytes


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_lowbit_bitstream_pack_round_trips(bits):
    torch = _TORCH
    values = torch.arange(0, 128 * 128, dtype=torch.int32).reshape(1, 128, 128)
    values = values % (1 << bits)
    packed_bytes = (values.numel() * bits + 7) // 8

    packed = _pack_bits_flat(values, bits, packed_bytes)
    restored = _unpack_bits_flat(packed, bits, values.shape)

    assert torch.equal(restored.to(torch.int32), values)


def test_kvarn_gqa_tile_quantize_dequantize_shapes_and_finiteness():
    torch = _TORCH
    torch.manual_seed(7)
    cfg = KVarNGQAConfig(sinkhorn_iters=2)
    k = torch.randn(cfg.group, 2, cfg.head_dim, dtype=torch.float16)
    v = torch.randn_like(k)

    records = quantize_gqa_tile(k, v, cfg)
    k_restored, v_restored = dequantize_gqa_tile(records, cfg)

    assert records.shape == (2, cfg.tile_bytes_aligned)
    assert records.dtype == torch.uint8
    assert k_restored.shape == k.shape
    assert v_restored.shape == v.shape
    assert torch.isfinite(k_restored).all()
    assert torch.isfinite(v_restored).all()
    assert torch.nn.functional.cosine_similarity(
        k.float().flatten(), k_restored.flatten(), dim=0).item() > 0.55
    assert torch.nn.functional.cosine_similarity(
        v.float().flatten(), v_restored.flatten(), dim=0).item() > 0.55


def test_kvarn_gqa_packed_pool_and_transfer_view():
    torch = _TORCH
    cfg = KVarNGQAConfig(sinkhorn_iters=1)
    pool = KVarNGQAPackedPool(num_layers=2, num_blocks=3, num_kv_heads=1, cfg=cfg)
    k = torch.randn(cfg.group, 1, cfg.head_dim, dtype=torch.float16)
    v = torch.randn_like(k)

    pool.store_block(1, 2, k, v)
    k_restored, v_restored = pool.load_block(1, 2)
    view = pool.transfer_view(1, torch.tensor([2]))

    assert pool.valid[1, 2]
    assert pool.commit_gen[1, 2].item() == 1
    assert k_restored.shape == k.shape
    assert v_restored.shape == v.shape
    assert view["records"].shape == (1, 1, cfg.tile_bytes_aligned)
    assert view["valid"].shape == (1,)
    assert view["tile_bytes"].item() == cfg.tile_bytes_aligned


def test_kvarn_gqa_side_pool_is_fixed_capacity_and_snapshot_ready():
    torch = _TORCH
    cfg = KVarNGQAConfig(sinkhorn_iters=1)
    pool = _KVarNGQASidePool(
        cfg,
        num_layers=2,
        max_batch_size=2,
        max_blocks_per_seq=3,
        num_kv_heads=1,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )
    slot = pool.slot_for_request(17)
    k_tok = torch.ones((1, cfg.head_dim), dtype=torch.float16)
    v_tok = torch.full_like(k_tok, 2)

    pool.put_sink(1, slot, k_tok, v_tok, 0)
    sink_k, sink_v = pool.sink_tensors(1, slot, 1)
    assert sink_k.shape == (1, 1, cfg.head_dim)
    assert torch.equal(sink_k[0], k_tok)
    assert torch.equal(sink_v[0], v_tok)

    for offset in range(cfg.group):
        pool.put_tail(1, slot, cfg.group, offset, k_tok, v_tok)
    committed_id = id(pool.committed)
    pool.mark_committed(1, slot, 17, cfg.group)
    assert id(pool.committed) == committed_id
    assert pool.is_committed(1, slot, cfg.group)

    snapshot = pool.transfer_snapshot(1, 17)
    assert snapshot["sink_k"].shape == (cfg.sink_tokens, 1, cfg.head_dim)
    assert snapshot["tail_k"].shape == (cfg.group, 1, cfg.head_dim)
    assert snapshot["committed"].shape == (3,)
    assert snapshot["commit_gen"].shape == (3,)

    with pytest.raises(RuntimeError, match="exceeds graph-safe side-pool"):
        pool.mark_committed(1, slot, 17, cfg.group * 3)


def test_kvarn_gqa_side_pool_fails_closed_for_missing_sink_and_tail():
    torch = _TORCH
    cfg = KVarNGQAConfig()
    pool = _KVarNGQASidePool(
        cfg,
        num_layers=1,
        max_batch_size=1,
        max_blocks_per_seq=2,
        num_kv_heads=1,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )
    slot = pool.slot_for_request(9)
    tok = torch.zeros((1, cfg.head_dim), dtype=torch.float16)

    with pytest.raises(RuntimeError, match="sink state incomplete"):
        pool.sink_tensors(0, slot, 1)

    pool.put_tail(0, slot, cfg.group, 0, tok, tok)
    with pytest.raises(RuntimeError, match="tail advanced"):
        pool.put_tail(0, slot, cfg.group * 2, 0, tok, tok)

    with pytest.raises(RuntimeError, match="side pool exhausted"):
        pool.slot_for_request(10)

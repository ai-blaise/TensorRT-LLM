# SPDX-License-Identifier: Apache-2.0
"""KVarN GQA packed-record correctness coverage."""

import pytest


_TORCH = pytest.importorskip("torch")

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

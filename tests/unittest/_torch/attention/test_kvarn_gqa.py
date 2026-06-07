# SPDX-License-Identifier: Apache-2.0
"""KVarN GQA packed-record correctness coverage."""

import pytest


_TORCH = pytest.importorskip("torch")

from tensorrt_llm._torch.attention_backend import kvarn_gqa_attention as _gqa_attention  # noqa: E402
from tensorrt_llm._torch.attention_backend.kvarn_gqa_attention import (  # noqa: E402
    KVarNGQAAttention,
    _KVarNGQASidePool,
    _write_record_to_page,
)
from tensorrt_llm._torch.attention_backend.kvarn_gqa import (  # noqa: E402
    KVarNGQAConfig,
    KVarNGQAPackedPool,
    _pack_bits_flat,
    _unpack_bits_flat,
    dequantize_gqa_tile,
    dequantize_gqa_tiles,
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



def test_kvarn_gqa_side_pool_release_request_clears_abort_reuse_state():
    torch = _TORCH
    cfg = KVarNGQAConfig()
    pool = _KVarNGQASidePool(
        cfg,
        num_layers=2,
        max_batch_size=1,
        max_blocks_per_seq=2,
        num_kv_heads=1,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )
    slot = pool.slot_for_request(41)
    tok = torch.ones((1, cfg.head_dim), dtype=torch.float16)
    pool.put_sink(0, slot, tok, tok, 0)
    pool.put_tail(1, slot, cfg.group, 0, tok, tok)
    pool.mark_committed(0, slot, 41, 0)

    assert int(pool.sink_len[0, slot].item()) == 1
    assert bool(pool.tail_filled[1, slot, 0].item())
    assert pool.is_committed(0, slot, 0)
    assert pool.request_block_to_slot_block[(0, 41, 0)] == 0

    pool.release_request(41)

    assert 41 not in pool.request_to_slot
    assert slot not in pool.slot_to_request
    assert int(pool.sink_len[:, slot].sum().item()) == 0
    assert not bool(pool.tail_filled[:, slot].any().item())
    assert torch.equal(pool.tail_block_start[:, slot], torch.full((2,), -1, dtype=torch.int64))
    assert not bool(pool.committed[:, slot].any().item())
    assert int(pool.commit_gen[:, slot].sum().item()) == 0
    assert not pool.request_block_to_slot_block

    reused = pool.slot_for_request(42)
    assert reused == slot
    with pytest.raises(RuntimeError, match="sink state incomplete"):
        pool.sink_tensors(0, reused, 1)


def test_kvarn_gqa_batched_dequant_matches_single_tile_reference():
    torch = _TORCH
    torch.manual_seed(20260607)
    cfg = KVarNGQAConfig(sinkhorn_iters=1)
    records = []
    for i in range(2):
        k = (torch.randn(cfg.group, 1, cfg.head_dim, dtype=torch.float16) * 0.25 + i)
        v = (torch.randn_like(k) * 0.25 - i)
        records.append(quantize_gqa_tile(k, v, cfg))
    records = torch.stack(records, dim=0)

    k_batch, v_batch = dequantize_gqa_tiles(records, cfg)

    assert k_batch.shape == (2, cfg.group, 1, cfg.head_dim)
    assert v_batch.shape == (2, cfg.group, 1, cfg.head_dim)
    for i in range(2):
        k_single, v_single = dequantize_gqa_tile(records[i], cfg)
        assert torch.equal(k_batch[i], k_single)
        assert torch.equal(v_batch[i], v_single)


def test_kvarn_gqa_bdr_restore_scales_with_churn_not_working_set(monkeypatch):
    torch = _TORCH
    torch.manual_seed(20260607)
    cfg = KVarNGQAConfig(sinkhorn_iters=1)
    state = _KVarNGQASidePool(
        cfg,
        num_layers=1,
        max_batch_size=1,
        max_blocks_per_seq=5,
        num_kv_heads=1,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )
    kv_pages = torch.zeros((8, 1, cfg.group, 1, cfg.bytes_per_token_slot),
                           dtype=torch.uint8)
    state.ensure_bdr_pool(kv_pages.shape[0])
    assert state.restored_gen.dtype == torch.int64
    assert state.physical_commit_gen.dtype == torch.int64

    slot = state.slot_for_request(7)
    block_ids = [0, 2, 4, 6, 7]
    for logical_block, physical_block in ((1, 2), (2, 4), (3, 6)):
        k = torch.randn(cfg.group, 1, cfg.head_dim, dtype=torch.float16) * 0.2
        v = torch.randn_like(k) * 0.2
        record = quantize_gqa_tile(k + logical_block, v - logical_block, cfg)
        _write_record_to_page(kv_pages[physical_block, 0], record, cfg)
        state.mark_committed(0, slot, 7, logical_block * cfg.group,
                             physical_block_id=physical_block)

    real_dequant = _gqa_attention.dequantize_gqa_tiles
    calls = []

    def counted_dequant(records, cfg):
        calls.append(int(records.shape[0]))
        return real_dequant(records, cfg)

    monkeypatch.setattr(_gqa_attention, "dequantize_gqa_tiles", counted_dequant)

    logical, physical = state.restore_committed_blocks_amortized(
        0, slot, block_ids, seq_len=4 * cfg.group, kv_pages=kv_pages,
        amortize=True)

    assert logical.tolist() == [1, 2, 3]
    assert physical.tolist() == [2, 4, 6]
    assert calls == [3]
    assert torch.equal(state.restored_gen[0, physical],
                       state.physical_commit_gen[0, physical])

    calls.clear()
    state.restore_committed_blocks_amortized(
        0, slot, block_ids, seq_len=4 * cfg.group, kv_pages=kv_pages,
        amortize=True)
    assert calls == []

    k = torch.randn(cfg.group, 1, cfg.head_dim, dtype=torch.float16) * 0.2
    v = torch.randn_like(k) * 0.2
    record = quantize_gqa_tile(k, v, cfg)
    _write_record_to_page(kv_pages[4, 0], record, cfg)
    state.mark_committed(0, slot, 7, 2 * cfg.group, physical_block_id=4)

    calls.clear()
    state.restore_committed_blocks_amortized(
        0, slot, block_ids, seq_len=4 * cfg.group, kv_pages=kv_pages,
        amortize=True)
    assert calls == [1]
    assert int(state.restored_gen[0, 4].item()) == int(state.physical_commit_gen[0, 4].item())

    calls.clear()
    state.restore_committed_blocks_amortized(
        0, slot, block_ids, seq_len=4 * cfg.group, kv_pages=kv_pages,
        amortize=False)
    assert calls == [3]

    state.release_request(7)
    assert not bool(state.physical_valid[0, 2].item())
    assert not bool(state.physical_valid[0, 4].item())
    assert not bool(state.physical_valid[0, 6].item())
    assert int(state.restored_gen[0, torch.tensor([2, 4, 6])].sum().item()) == 0


def test_kvarn_gqa_side_pool_transfer_meta_slots_and_fragments():
    torch = _TORCH
    from types import SimpleNamespace

    from tensorrt_llm._torch.disaggregation.native.transfer import RecvReqInfo, Sender

    cfg = KVarNGQAConfig(sinkhorn_iters=1)
    src_pool = _KVarNGQASidePool(
        cfg,
        num_layers=2,
        max_batch_size=3,
        max_blocks_per_seq=4,
        num_kv_heads=1,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )
    dst_pool = _KVarNGQASidePool(
        cfg,
        num_layers=2,
        max_batch_size=3,
        max_blocks_per_seq=4,
        num_kv_heads=1,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )
    src_slot = src_pool.slot_for_request(123)
    dst_slot = dst_pool.slot_for_request(123)
    assert src_slot == 0
    assert dst_slot == 0

    src_meta = src_pool.transfer_meta(device_id=0)
    dst_meta = dst_pool.transfer_meta(device_id=0)

    assert src_meta.ptrs.shape == src_meta.item_sizes.shape
    assert src_meta.max_slots == 3
    assert any(name.endswith("sink_k") for name in src_meta.names)
    assert any(name.endswith("tail_block_start") for name in src_meta.names)
    assert any(name.endswith("commit_gen") for name in src_meta.names)

    sender_self = SimpleNamespace(
        _kvarn_gqa_side_pool=src_pool,
        _registrar=SimpleNamespace(
            self_rank_info=SimpleNamespace(kvarn_gqa_side_meta=src_meta)
        ),
    )
    peer_ri = SimpleNamespace(kvarn_gqa_side_meta=dst_meta)
    req_info = RecvReqInfo(
        sender_req_id=11,
        instance_name="dst",
        instance_rank=0,
        block_ids_per_layer_groups=[],
        unique_rid=123,
        kvarn_gqa_side_slot=dst_slot,
    )
    task = SimpleNamespace(_unique_rid=123, _slice=SimpleNamespace(is_last_slice=True))

    src_ptrs, dst_ptrs, sizes = Sender._collect_kvarn_gqa_side_frags(
        sender_self, peer_ri, req_info, task
    )

    assert torch.equal(torch.from_numpy(sizes), torch.from_numpy(src_meta.item_sizes))
    assert torch.equal(torch.from_numpy(src_ptrs), torch.from_numpy(src_meta.ptrs))
    assert torch.equal(torch.from_numpy(dst_ptrs), torch.from_numpy(dst_meta.ptrs))

    # Non-final slices carry packed pages only; final slice carries side state.
    task._slice.is_last_slice = False
    assert Sender._collect_kvarn_gqa_side_frags(sender_self, peer_ri, req_info, task) is None

    # Missing side metadata must fail closed rather than transferring incomplete KVarN state.
    task._slice.is_last_slice = True
    with pytest.raises(RuntimeError, match="side-state metadata"):
        Sender._collect_kvarn_gqa_side_frags(
            sender_self, SimpleNamespace(kvarn_gqa_side_meta=None), req_info, task
        )


def test_kvarn_gqa_packed_decode_guards_causal_multitoken_and_scaling():
    from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask

    attn = KVarNGQAAttention(layer_idx=0, num_heads=2, num_kv_heads=1,
                             head_dim=128)
    assert attn._packed_decode_supported_or_raise(
        attention_mask=PredefinedAttentionMask.CAUSAL,
        q_len=1,
        attention_window_size=None)
    assert attn._packed_decode_supported_or_raise(
        attention_mask=PredefinedAttentionMask.FULL,
        q_len=25,
        attention_window_size=None)

    with pytest.raises(NotImplementedError, match="causal multi-token prefill"):
        attn._packed_decode_supported_or_raise(
            attention_mask=PredefinedAttentionMask.CAUSAL,
            q_len=5,
            attention_window_size=None)
    with pytest.raises(NotImplementedError, match="sliding-window"):
        attn._packed_decode_supported_or_raise(
            attention_mask=PredefinedAttentionMask.CAUSAL,
            q_len=1,
            attention_window_size=1024)

    scaled = KVarNGQAAttention(layer_idx=0, num_heads=2, num_kv_heads=1,
                               head_dim=128, q_scaling=0.5)
    with pytest.raises(NotImplementedError, match="q_scaling"):
        scaled._packed_decode_supported_or_raise(
            attention_mask=PredefinedAttentionMask.CAUSAL,
            q_len=1,
            attention_window_size=None)


def test_kvarn_gqa_packed_decode_block_list_uses_committed_physical_blocks():
    torch = _TORCH
    cfg = KVarNGQAConfig()
    attn = KVarNGQAAttention(layer_idx=0, num_heads=2, num_kv_heads=1,
                             head_dim=128)
    state = _KVarNGQASidePool(
        cfg,
        num_layers=1,
        max_batch_size=1,
        max_blocks_per_seq=4,
        num_kv_heads=1,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )
    slot = state.slot_for_request(91)
    # Logical block 0 is the fp16 sink. Packed decode should pass only full
    # committed post-sink physical blocks to the CUDA op.
    state.mark_committed(0, slot, 91, cfg.group, physical_block_id=5)
    state.mark_committed(0, slot, 91, cfg.group * 2, physical_block_id=7)

    packed = attn._packed_decode_full_blocks(
        state, slot, [0, 5, 7, 9], seq_len=3 * cfg.group)

    assert packed.dtype == torch.long
    assert packed.tolist() == [5, 7]

    with pytest.raises(NotImplementedError, match="uncommitted full block"):
        attn._packed_decode_full_blocks(
            state, slot, [0, 5, 7, 9], seq_len=4 * cfg.group)

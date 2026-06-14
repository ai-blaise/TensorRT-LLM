#!/usr/bin/env python3
"""Non-disruptive CUDA smoke for the OP-TRT HiSparse planner/copy chain.

This proof intentionally uses the production native torch ops and tensor
contracts. It does not allocate model weights, launch serving, or use a Python
TopK/request-table fallback. The host tier is a tiny pinned uint8 KVarN-shaped
record pool; the test verifies native planning, mapped pinned-host copy,
post-copy metadata commit, and hot-index construction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch


REQUIRED_OPS = (
    "trtllm::hisparse_topk_to_block_positions",
    "trtllm::hisparse_classify_resident_blocks",
    "trtllm::hisparse_resolve_blocks_to_host_slots",
    "trtllm::hisparse_plan_hot_slots",
    "trtllm::hisparse_compact_miss_schedule",
    "trtllm::hisparse_submit_packed_kvarn_copy_schedule",
    "trtllm::hisparse_commit_hot_slots",
    "trtllm::hisparse_build_hot_indices",
)


def _cuda_op_ready(name: str) -> bool:
    try:
        return bool(torch._C._dispatch_has_kernel_for_dispatch_key(name, "CUDA"))
    except Exception:
        return False


def _require_ops() -> None:
    missing = [name for name in REQUIRED_OPS if not _cuda_op_ready(name)]
    if missing:
        raise RuntimeError("missing CUDA HiSparse op(s): " + ", ".join(missing))


def _assert_all_zero(name: str, tensor: torch.Tensor, rows: Iterable[int]) -> None:
    values = tensor.detach().cpu().tolist()
    bad = [(row, values[row]) for row in rows if values[row] != 0]
    if bad:
        raise AssertionError(f"{name} expected status 0 for rows {list(rows)}, got {bad}")


def _assert_nonzero(name: str, tensor: torch.Tensor, rows: Iterable[int]) -> None:
    values = tensor.detach().cpu().tolist()
    bad = [(row, values[row]) for row in rows if values[row] == 0]
    if bad:
        raise AssertionError(f"{name} expected nonzero status for rows {list(rows)}, got {bad}")


def _pattern(layer: int, slot: int, record_bytes: int) -> torch.Tensor:
    return torch.tensor(
        [(layer * 53 + slot * 17 + byte) % 256 for byte in range(record_bytes)],
        dtype=torch.uint8,
    )


def _fill_host(host_packed: torch.Tensor) -> None:
    layers, slots, record_bytes = host_packed.shape
    for layer in range(layers):
        for slot in range(slots):
            host_packed[layer, slot, :record_bytes].copy_(
                _pattern(layer, slot, record_bytes)
            )


def _block_to_index(blocks: torch.Tensor, counts: torch.Tensor, row: int) -> dict[int, int]:
    blocks_cpu = blocks.detach().cpu()
    count = int(counts.detach().cpu()[row].item())
    return {int(blocks_cpu[row, i].item()): i for i in range(count)}


def _assert_hot_copy(
    host_packed: torch.Tensor,
    hot_packed: torch.Tensor,
    compact_host_slots: torch.Tensor,
    compact_hot_slots: torch.Tensor,
    copy_count: torch.Tensor,
    *,
    layer_idx: int,
    record_bytes: int,
) -> None:
    torch.cuda.synchronize()
    count = int(copy_count.detach().cpu()[0].item())
    host_slots = compact_host_slots.detach().cpu().tolist()[:count]
    hot_slots = compact_hot_slots.detach().cpu().tolist()[:count]
    hot_cpu = hot_packed.detach().cpu()
    for src_slot, dst_slot in zip(host_slots, hot_slots):
        expected = host_packed[layer_idx, src_slot, :record_bytes]
        actual = hot_cpu[layer_idx, dst_slot, :record_bytes]
        if not torch.equal(actual, expected):
            raise AssertionError(
                f"hot copy mismatch for host_slot={src_slot}, hot_slot={dst_slot}"
            )


def _assert_hot_metadata(
    hot_host_slot: torch.Tensor,
    hot_commit_gen: torch.Tensor,
    host_slots: torch.Tensor,
    commit_gens: torch.Tensor,
    planned_hot_slots: torch.Tensor,
    resident_flags: torch.Tensor,
    blocks: torch.Tensor,
    counts: torch.Tensor,
    *,
    layer_idx: int,
    rows: Iterable[int],
) -> None:
    hot_host_cpu = hot_host_slot.detach().cpu()
    hot_gen_cpu = hot_commit_gen.detach().cpu()
    host_cpu = host_slots.detach().cpu()
    gen_cpu = commit_gens.detach().cpu()
    planned_cpu = planned_hot_slots.detach().cpu()
    flags_cpu = resident_flags.detach().cpu()
    counts_cpu = counts.detach().cpu()
    blocks_cpu = blocks.detach().cpu()
    for row in rows:
        for i in range(int(counts_cpu[row].item())):
            if int(flags_cpu[row, i].item()) != 0:
                continue
            hot_slot = int(planned_cpu[row, i].item())
            expected_host = int(host_cpu[row, i].item())
            expected_gen = int(gen_cpu[row, i].item())
            actual_host = int(hot_host_cpu[layer_idx, hot_slot].item())
            actual_gen = int(hot_gen_cpu[layer_idx, hot_slot].item())
            if actual_host != expected_host or actual_gen != expected_gen:
                block = int(blocks_cpu[row, i].item())
                raise AssertionError(
                    "hot metadata mismatch "
                    f"row={row} block={block} hot_slot={hot_slot}: "
                    f"host/gen {actual_host}/{actual_gen}, "
                    f"expected {expected_host}/{expected_gen}"
                )


def _assert_hot_indices(
    topk: torch.Tensor,
    hot_indices: torch.Tensor,
    blocks: torch.Tensor,
    counts: torch.Tensor,
    planned_hot_slots: torch.Tensor,
    resident_flags: torch.Tensor,
    build_status: torch.Tensor,
    *,
    layer_idx: int,
    rows_ok: Iterable[int],
    rows_bad: Iterable[int],
    tokens_per_block: int,
    stride_factor: int,
) -> None:
    _assert_all_zero("build_status", build_status, rows_ok)
    _assert_nonzero("build_status", build_status, rows_bad)
    topk_cpu = topk.detach().cpu()
    hot_indices_cpu = hot_indices.detach().cpu()
    planned_cpu = planned_hot_slots.detach().cpu()
    flags_cpu = resident_flags.detach().cpu()
    for row in rows_ok:
        block_index = _block_to_index(blocks, counts, row)
        for col, token_value in enumerate(topk_cpu[row].tolist()):
            token = int(token_value)
            actual = int(hot_indices_cpu[row, col].item())
            if token < 0:
                if actual != -1:
                    raise AssertionError(f"padding index row={row} col={col} emitted {actual}")
                continue
            block_pos = token // tokens_per_block
            token_offset = token % tokens_per_block
            block_slot = block_index[block_pos]
            flag = int(flags_cpu[row, block_slot].item())
            if flag != 0:
                expected = -1
            else:
                hot_slot = int(planned_cpu[row, block_slot].item())
                expected = hot_slot * stride_factor + layer_idx * tokens_per_block + token_offset
            if actual != expected:
                raise AssertionError(
                    f"hot index mismatch row={row} col={col} token={token}: "
                    f"got {actual}, expected {expected}"
                )


def _main_chain(device: torch.device, record_bytes: int) -> None:
    tokens_per_block = 64
    max_blocks_per_row = 4
    num_layers = 2
    layer_idx = 1
    host_capacity = 16
    hot_capacity = 8
    stride_factor = num_layers * tokens_per_block

    topk = torch.tensor(
        [
            [0, 1, 64, 65, 128, 130, -1, -1],
            [64, 65, 192, 193, 194, -1, -1, -1],
            [64, 66, -1, -1, -1, -1, -1, -1],
        ],
        dtype=torch.int32,
        device=device,
    )
    blocks, counts, overflow = torch.ops.trtllm.hisparse_topk_to_block_positions(
        topk, tokens_per_block, max_blocks_per_row
    )
    torch.cuda.synchronize()
    if overflow.detach().cpu().tolist() != [0, 0, 0]:
        raise AssertionError(f"unexpected overflow flags: {overflow.detach().cpu().tolist()}")
    if counts.detach().cpu().tolist() != [3, 2, 1]:
        raise AssertionError(f"unexpected block counts: {counts.detach().cpu().tolist()}")

    row_kv_lens = torch.tensor([160, 300, 128], dtype=torch.int64, device=device)
    tail_block_pos = torch.tensor([2, 4, -1], dtype=torch.int32, device=device)
    tail_valid = torch.tensor([True, True, False], dtype=torch.bool, device=device)
    resident_flags, resident_status = torch.ops.trtllm.hisparse_classify_resident_blocks(
        blocks, counts, row_kv_lens, tail_block_pos, tail_valid, tokens_per_block, 1
    )
    _assert_all_zero("resident_status", resident_status, [0, 1, 2])

    block_index0 = _block_to_index(blocks, counts, 0)
    flags0 = resident_flags.detach().cpu()[0].tolist()
    if flags0[block_index0[0]] != 1 or flags0[block_index0[1]] != 0 or flags0[block_index0[2]] != 2:
        raise AssertionError(f"unexpected row0 resident flags: blocks={block_index0}, flags={flags0}")

    row_request_ids = torch.tensor([1001, 1002, 1003], dtype=torch.int64, device=device)
    request_ids = torch.tensor([1001, 1002, 1003], dtype=torch.int64, device=device)
    request_block_host_slots = torch.full((3, 8), -1, dtype=torch.int64, device=device)
    request_block_commit_gen = torch.full((3, 8), -1, dtype=torch.int64, device=device)
    request_admitted = torch.tensor([True, True, False], dtype=torch.bool, device=device)
    request_block_host_slots[0, 1] = 5
    request_block_commit_gen[0, 1] = 101
    request_block_host_slots[1, 1] = 6
    request_block_commit_gen[1, 1] = 201
    request_block_host_slots[1, 3] = 7
    request_block_commit_gen[1, 3] = 203
    request_block_host_slots[2, 1] = 8
    request_block_commit_gen[2, 1] = 301

    host_slots, commit_gens, _block_status, resolve_status = (
        torch.ops.trtllm.hisparse_resolve_blocks_to_host_slots(
            row_request_ids,
            blocks,
            counts,
            resident_flags,
            resident_status,
            request_ids,
            request_block_host_slots,
            request_block_commit_gen,
            request_admitted,
        )
    )
    _assert_all_zero("resolve_status", resolve_status, [0, 1])
    _assert_nonzero("resolve_status", resolve_status, [2])

    hot_host_slot = torch.full((num_layers, hot_capacity), -1, dtype=torch.int64, device=device)
    hot_commit_gen = torch.full((num_layers, hot_capacity), -1, dtype=torch.int64, device=device)
    hot_lru_tick = torch.zeros((num_layers, hot_capacity), dtype=torch.int64, device=device)
    # Stale hot metadata for row1/block1: host slot matches, commit generation
    # does not. The first planner pass must treat this as a miss and repair it
    # through copy + post-copy metadata commit rather than reusing the slot.
    hot_host_slot[layer_idx, 0] = 6
    hot_commit_gen[layer_idx, 0] = 999
    hot_lru_tick[layer_idx, 0] = 1
    host_packed = torch.empty(
        (num_layers, host_capacity, record_bytes), dtype=torch.uint8, pin_memory=True
    )
    hot_packed = torch.zeros(
        (num_layers, hot_capacity, record_bytes), dtype=torch.uint8, device=device
    )
    _fill_host(host_packed)

    planned_hot_slots, planned_lru_tick, miss_host_slots, miss_hot_slots, miss_counts, _hit_flags, plan_status = (
        torch.ops.trtllm.hisparse_plan_hot_slots(
            host_slots,
            commit_gens,
            counts,
            resident_flags,
            resolve_status,
            hot_host_slot,
            hot_commit_gen,
            hot_lru_tick,
            layer_idx,
            10,
        )
    )
    _assert_all_zero("plan_status", plan_status, [0, 1])
    _assert_nonzero("plan_status", plan_status, [2])
    if miss_counts.detach().cpu().tolist()[:3] != [1, 2, 0]:
        raise AssertionError(f"unexpected first-pass miss counts: {miss_counts.detach().cpu().tolist()}")

    compact_host, compact_hot, compact_rows, copy_count, compact_status = (
        torch.ops.trtllm.hisparse_compact_miss_schedule(
            miss_host_slots, miss_hot_slots, miss_counts, plan_status
        )
    )
    _assert_all_zero("compact_status", compact_status, [0, 1])
    _assert_nonzero("compact_status", compact_status, [2])
    if int(copy_count.detach().cpu()[0].item()) != 3:
        raise AssertionError(f"expected 3 first-pass copies, got {copy_count.detach().cpu().tolist()}")

    copy_status = torch.ops.trtllm.hisparse_submit_packed_kvarn_copy_schedule(
        host_packed,
        hot_packed,
        compact_host,
        compact_hot,
        compact_rows,
        copy_count,
        compact_status,
        layer_idx,
        record_bytes,
    )
    _assert_all_zero("copy_status", copy_status, [0, 1])
    _assert_nonzero("copy_status", copy_status, [2])
    _assert_hot_copy(
        host_packed,
        hot_packed,
        compact_host,
        compact_hot,
        copy_count,
        layer_idx=layer_idx,
        record_bytes=record_bytes,
    )

    commit_status = torch.ops.trtllm.hisparse_commit_hot_slots(
        host_slots,
        commit_gens,
        planned_hot_slots,
        planned_lru_tick,
        counts,
        copy_status,
        hot_host_slot,
        hot_commit_gen,
        hot_lru_tick,
        resident_flags,
        layer_idx,
    )
    _assert_all_zero("commit_status", commit_status, [0, 1])
    _assert_nonzero("commit_status", commit_status, [2])
    _assert_hot_metadata(
        hot_host_slot,
        hot_commit_gen,
        host_slots,
        commit_gens,
        planned_hot_slots,
        resident_flags,
        blocks,
        counts,
        layer_idx=layer_idx,
        rows=[0, 1],
    )

    hot_indices, build_status = torch.ops.trtllm.hisparse_build_hot_indices(
        topk,
        blocks,
        planned_hot_slots,
        counts,
        commit_status,
        resident_flags,
        hot_capacity,
        tokens_per_block,
        stride_factor,
        layer_idx,
    )
    _assert_hot_indices(
        topk,
        hot_indices,
        blocks,
        counts,
        planned_hot_slots,
        resident_flags,
        build_status,
        layer_idx=layer_idx,
        rows_ok=[0, 1],
        rows_bad=[2],
        tokens_per_block=tokens_per_block,
        stride_factor=stride_factor,
    )

    # Re-run the planner after metadata commit. The same committed blocks should
    # now be hits, producing no compact copy schedule for admitted rows.
    planned_hot_slots2, planned_lru_tick2, miss_host_slots2, miss_hot_slots2, miss_counts2, hit_flags2, plan_status2 = (
        torch.ops.trtllm.hisparse_plan_hot_slots(
            host_slots,
            commit_gens,
            counts,
            resident_flags,
            resolve_status,
            hot_host_slot,
            hot_commit_gen,
            hot_lru_tick,
            layer_idx,
            1000,
        )
    )
    _assert_all_zero("plan_status2", plan_status2, [0, 1])
    _assert_nonzero("plan_status2", plan_status2, [2])
    if miss_counts2.detach().cpu().tolist()[:3] != [0, 0, 0]:
        raise AssertionError(f"expected second-pass hits, got misses {miss_counts2.detach().cpu().tolist()}")
    hit_cpu = hit_flags2.detach().cpu()
    for row in (0, 1):
        for i in range(int(counts.detach().cpu()[row].item())):
            if int(resident_flags.detach().cpu()[row, i].item()) == 0 and int(hit_cpu[row, i].item()) != 1:
                raise AssertionError(f"expected hit flag for row={row} block_index={i}")

    compact_host2, compact_hot2, compact_rows2, copy_count2, compact_status2 = (
        torch.ops.trtllm.hisparse_compact_miss_schedule(
            miss_host_slots2, miss_hot_slots2, miss_counts2, plan_status2
        )
    )
    if int(copy_count2.detach().cpu()[0].item()) != 0:
        raise AssertionError(f"expected zero second-pass copies, got {copy_count2.detach().cpu().tolist()}")
    copy_status2 = torch.ops.trtllm.hisparse_submit_packed_kvarn_copy_schedule(
        host_packed,
        hot_packed,
        compact_host2,
        compact_hot2,
        compact_rows2,
        copy_count2,
        compact_status2,
        layer_idx,
        record_bytes,
    )
    commit_status2 = torch.ops.trtllm.hisparse_commit_hot_slots(
        host_slots,
        commit_gens,
        planned_hot_slots2,
        planned_lru_tick2,
        counts,
        copy_status2,
        hot_host_slot,
        hot_commit_gen,
        hot_lru_tick,
        resident_flags,
        layer_idx,
    )
    hot_indices2, build_status2 = torch.ops.trtllm.hisparse_build_hot_indices(
        topk,
        blocks,
        planned_hot_slots2,
        counts,
        commit_status2,
        resident_flags,
        hot_capacity,
        tokens_per_block,
        stride_factor,
        layer_idx,
    )
    _assert_hot_indices(
        topk,
        hot_indices2,
        blocks,
        counts,
        planned_hot_slots2,
        resident_flags,
        build_status2,
        layer_idx=layer_idx,
        rows_ok=[0, 1],
        rows_bad=[2],
        tokens_per_block=tokens_per_block,
        stride_factor=stride_factor,
    )


def _negative_cases(device: torch.device) -> None:
    overflow_topk = torch.tensor([[0, 64, 128]], dtype=torch.int32, device=device)
    _blocks, counts, overflow = torch.ops.trtllm.hisparse_topk_to_block_positions(
        overflow_topk, 64, 2
    )
    torch.cuda.synchronize()
    if overflow.detach().cpu().tolist() != [1] or counts.detach().cpu().tolist() != [3]:
        raise AssertionError(
            "overflow row did not fail closed through count=max_blocks+1: "
            f"overflow={overflow.detach().cpu().tolist()} counts={counts.detach().cpu().tolist()}"
        )

    blocks = torch.tensor([[1]], dtype=torch.int32, device=device)
    block_counts = torch.tensor([1], dtype=torch.int32, device=device)
    resident_flags = torch.zeros((1, 1), dtype=torch.uint8, device=device)
    resident_status = torch.zeros((1,), dtype=torch.uint8, device=device)
    row_request_ids = torch.tensor([2001], dtype=torch.int64, device=device)
    request_ids = torch.tensor([2001], dtype=torch.int64, device=device)
    request_block_host_slots = torch.full((1, 4), -1, dtype=torch.int64, device=device)
    request_block_commit_gen = torch.full((1, 4), -1, dtype=torch.int64, device=device)
    request_admitted = torch.tensor([True], dtype=torch.bool, device=device)
    request_block_host_slots[0, 1] = 3
    host_slots, commit_gens, _block_status, resolve_status = (
        torch.ops.trtllm.hisparse_resolve_blocks_to_host_slots(
            row_request_ids,
            blocks,
            block_counts,
            resident_flags,
            resident_status,
            request_ids,
            request_block_host_slots,
            request_block_commit_gen,
            request_admitted,
        )
    )
    _assert_nonzero("uncommitted resolve_status", resolve_status, [0])
    if host_slots.detach().cpu().tolist() != [[-1]] or commit_gens.detach().cpu().tolist() != [[-1]]:
        raise AssertionError("uncommitted block should not expose host slot or commit generation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, help="Optional libth_hisparse_smoke.so path to load")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--record-bytes", type=int, default=32)
    args = parser.parse_args()

    if args.library:
        torch.ops.load_library(str(args.library))
    _require_ops()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for native HiSparse planner/copy smoke")
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    _main_chain(device, args.record_bytes)
    _negative_cases(device)
    torch.cuda.synchronize()
    print("native hisparse planner/copy smoke passed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Graph-capture PROBE for the overlapped native HiSparse swap-in copy.

PROVES the whole reason for keeping the swap-in copy on the native-op route: the
overlapped path (coordinator copy stream + fork/join events + the schedule kernel
launched on the copy stream) is CUDA-graph-capture-safe. It captures the swap-in
into a cudaGraph, replays it N times, and asserts:
  (1) no capture / instantiation / replay error, AND
  (2) the hot-buffer bytes produced by the captured+replayed graph are
      byte-IDENTICAL to the eager (non-captured) overlapped path AND to the
      serial in-stream path.

The schedule (compact_host/hot_slots, copy_count) is produced ONCE on device by
the real planner chain (topk -> ... -> compact_miss_schedule), exactly as the live
forward does; the captured region is just the submit op driven by that device
schedule, forked onto the coordinator copy stream and joined back. No host sync /
no d2h / no CPU gather is on the captured path.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


REQUIRED_OPS = (
    "trtllm::hisparse_topk_to_block_positions",
    "trtllm::hisparse_classify_resident_blocks",
    "trtllm::hisparse_resolve_blocks_to_host_slots",
    "trtllm::hisparse_plan_hot_slots",
    "trtllm::hisparse_compact_miss_schedule",
    "trtllm::hisparse_submit_packed_kvarn_copy_schedule",
)


def _cuda_op_ready(name: str) -> bool:
    try:
        return bool(torch._C._dispatch_has_kernel_for_dispatch_key(name, "CUDA"))
    except Exception:
        return False


def _require_ops() -> None:
    missing = [n for n in REQUIRED_OPS if not _cuda_op_ready(n)]
    if missing:
        raise RuntimeError("missing CUDA HiSparse op(s): " + ", ".join(missing))


def _overlap_args_supported() -> bool:
    op = torch.ops.trtllm.hisparse_submit_packed_kvarn_copy_schedule
    names = {a.name for a in op.default._schema.arguments}
    return "overlap_copy_stream" in names and "copy_stream_handle" in names


def _pattern(layer: int, slot: int, record_bytes: int, device) -> torch.Tensor:
    return torch.tensor(
        [(layer * 53 + slot * 17 + byte) % 256 for byte in range(record_bytes)],
        dtype=torch.uint8, device=device)


def _build_schedule(device, record_bytes):
    """Run the real planner chain to get a device-resident miss schedule + a pinned
    host_packed filled with a known pattern. Returns the tensors the submit needs."""
    tokens_per_block = 64
    max_blocks_per_row = 4
    num_layers = 2
    layer_idx = 1
    host_capacity = 16
    hot_capacity = 8

    topk = torch.tensor(
        [
            [0, 1, 64, 65, 128, 130, -1, -1],
            [64, 65, 192, 193, 194, -1, -1, -1],
            [64, 66, -1, -1, -1, -1, -1, -1],
        ],
        dtype=torch.int32, device=device)
    blocks, counts, _ = torch.ops.trtllm.hisparse_topk_to_block_positions(
        topk, tokens_per_block, max_blocks_per_row)
    row_kv_lens = torch.tensor([160, 300, 128], dtype=torch.int64, device=device)
    tail_block_pos = torch.tensor([2, 4, -1], dtype=torch.int32, device=device)
    tail_valid = torch.tensor([True, True, False], dtype=torch.bool, device=device)
    resident_flags, resident_status = torch.ops.trtllm.hisparse_classify_resident_blocks(
        blocks, counts, row_kv_lens, tail_block_pos, tail_valid, tokens_per_block, 1)

    row_request_ids = torch.tensor([1001, 1002, 1003], dtype=torch.int64, device=device)
    request_ids = torch.tensor([1001, 1002, 1003], dtype=torch.int64, device=device)
    req_host = torch.full((3, 8), -1, dtype=torch.int64, device=device)
    req_gen = torch.full((3, 8), -1, dtype=torch.int64, device=device)
    request_admitted = torch.tensor([True, True, False], dtype=torch.bool, device=device)
    req_host[0, 1] = 5; req_gen[0, 1] = 101
    req_host[1, 1] = 6; req_gen[1, 1] = 201
    req_host[1, 3] = 7; req_gen[1, 3] = 203
    req_host[2, 1] = 8; req_gen[2, 1] = 301
    host_slots, commit_gens, _b, resolve_status = (
        torch.ops.trtllm.hisparse_resolve_blocks_to_host_slots(
            row_request_ids, blocks, counts, resident_flags, resident_status,
            request_ids, req_host, req_gen, request_admitted))

    hot_host_slot = torch.full((num_layers, hot_capacity), -1, dtype=torch.int64, device=device)
    hot_commit_gen = torch.full((num_layers, hot_capacity), -1, dtype=torch.int64, device=device)
    hot_lru_tick = torch.zeros((num_layers, hot_capacity), dtype=torch.int64, device=device)
    hot_host_slot[layer_idx, 0] = 6
    hot_commit_gen[layer_idx, 0] = 999
    hot_lru_tick[layer_idx, 0] = 1

    host_packed = torch.empty((num_layers, host_capacity, record_bytes),
                              dtype=torch.uint8, pin_memory=True)
    for layer in range(num_layers):
        for slot in range(host_capacity):
            host_packed[layer, slot, :record_bytes].copy_(
                _pattern(layer, slot, record_bytes, "cpu"))

    planned_hot_slots, planned_lru_tick, miss_host, miss_hot, miss_counts, _hf, plan_status = (
        torch.ops.trtllm.hisparse_plan_hot_slots(
            host_slots, commit_gens, counts, resident_flags, resolve_status,
            hot_host_slot, hot_commit_gen, hot_lru_tick, layer_idx, 10))
    compact_host, compact_hot, compact_rows, copy_count, compact_status = (
        torch.ops.trtllm.hisparse_compact_miss_schedule(
            miss_host, miss_hot, miss_counts, plan_status))
    torch.cuda.synchronize()
    n_copies = int(copy_count.cpu()[0].item())
    return dict(host_packed=host_packed, compact_host=compact_host,
                compact_hot=compact_hot, compact_rows=compact_rows,
                copy_count=copy_count, compact_status=compact_status,
                layer_idx=layer_idx, record_bytes=record_bytes,
                hot_shape=(num_layers, hot_capacity, record_bytes),
                n_copies=n_copies)


def _submit(host_packed, hot_packed, s, *, overlap, copy_stream_handle, layer_idx, record_bytes):
    if overlap:
        return torch.ops.trtllm.hisparse_submit_packed_kvarn_copy_schedule(
            host_packed, hot_packed, s["compact_host"], s["compact_hot"],
            s["compact_rows"], s["copy_count"], s["compact_status"], layer_idx,
            record_bytes, True, int(copy_stream_handle))
    return torch.ops.trtllm.hisparse_submit_packed_kvarn_copy_schedule(
        host_packed, hot_packed, s["compact_host"], s["compact_hot"],
        s["compact_rows"], s["copy_count"], s["compact_status"], layer_idx,
        record_bytes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--record-bytes", type=int, default=512)
    parser.add_argument("--replays", type=int, default=8)
    args = parser.parse_args()
    if args.library:
        torch.ops.load_library(str(args.library))
    _require_ops()
    if not _overlap_args_supported():
        raise RuntimeError(
            "built op does not advertise overlap args (overlap_copy_stream / "
            "copy_stream_handle) -- rebuild the smoke .so from this worktree.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    s = _build_schedule(device, args.record_bytes)
    layer_idx, rb, hot_shape = s["layer_idx"], s["record_bytes"], s["hot_shape"]
    print(f"schedule built: {s['n_copies']} device miss copies, record_bytes={rb}")

    # --- reference 1: serial in-stream submit ---
    hot_serial = torch.zeros(hot_shape, dtype=torch.uint8, device=device)
    st_serial = _submit(s["host_packed"], hot_serial, s, overlap=False,
                        copy_stream_handle=0, layer_idx=layer_idx, record_bytes=rb)
    torch.cuda.synchronize()
    ref_serial = hot_serial.clone()

    # --- reference 2: eager overlapped submit (copy stream + fork/join) ---
    copy_stream = torch.cuda.Stream(device=device)
    fork_event = torch.cuda.Event()
    done_event = torch.cuda.Event()
    hot_eager = torch.zeros(hot_shape, dtype=torch.uint8, device=device)
    main_stream = torch.cuda.current_stream(device=device)
    fork_event.record(main_stream)
    copy_stream.wait_event(fork_event)
    st_eager = _submit(s["host_packed"], hot_eager, s, overlap=True,
                       copy_stream_handle=copy_stream.cuda_stream,
                       layer_idx=layer_idx, record_bytes=rb)
    done_event.record(copy_stream)
    main_stream.wait_event(done_event)
    torch.cuda.synchronize()
    ref_eager = hot_eager.clone()

    serial_vs_eager = bool(torch.equal(ref_serial, ref_eager))
    status_match = bool(torch.equal(st_serial.cpu(), st_eager.cpu()))
    print(f"serial == eager-overlap bytes: {serial_vs_eager}")
    print(f"serial copy_status == eager copy_status: {status_match}")

    # --- capture the overlapped submit into a CUDA graph, replay N times ---
    # Capture stream + a separate copy stream used inside the captured region.
    hot_graph = torch.zeros(hot_shape, dtype=torch.uint8, device=device)
    cap_copy_stream = torch.cuda.Stream(device=device)
    cap_fork = torch.cuda.Event()
    cap_done = torch.cuda.Event()
    graph = torch.cuda.CUDAGraph()

    # Warm-up the capture stream side allocator (PyTorch recommends a warm iter on
    # a side stream before capture); run one eager overlapped submit into a scratch.
    scratch = torch.zeros(hot_shape, dtype=torch.uint8, device=device)
    warm_stream = torch.cuda.Stream(device=device)
    warm_stream.wait_stream(torch.cuda.current_stream(device=device))
    with torch.cuda.stream(warm_stream):
        _submit(s["host_packed"], scratch, s, overlap=True,
                copy_stream_handle=cap_copy_stream.cuda_stream,
                layer_idx=layer_idx, record_bytes=rb)
    torch.cuda.current_stream(device=device).wait_stream(warm_stream)
    torch.cuda.synchronize()

    capture_err = None
    try:
        with torch.cuda.graph(graph):
            cap_main = torch.cuda.current_stream(device=device)
            cap_fork.record(cap_main)
            cap_copy_stream.wait_event(cap_fork)
            graph_status = _submit(
                s["host_packed"], hot_graph, s, overlap=True,
                copy_stream_handle=cap_copy_stream.cuda_stream,
                layer_idx=layer_idx, record_bytes=rb)
            cap_done.record(cap_copy_stream)
            cap_main.wait_event(cap_done)
    except Exception as exc:  # noqa: BLE001
        capture_err = exc

    if capture_err is not None:
        print(f"CAPTURE ERROR: {type(capture_err).__name__}: {capture_err}")
        print("VERDICT graph-capture: FAIL")
        raise SystemExit(1)

    replay_err = None
    try:
        for _ in range(args.replays):
            hot_graph.zero_()
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        replay_err = exc

    if replay_err is not None:
        print(f"REPLAY ERROR: {type(replay_err).__name__}: {replay_err}")
        print("VERDICT graph-capture: FAIL")
        raise SystemExit(1)

    ref_graph = hot_graph.clone()
    graph_vs_serial = bool(torch.equal(ref_graph, ref_serial))
    graph_status_match = bool(torch.equal(graph_status.cpu(), st_serial.cpu()))
    print(f"replays completed: {args.replays} (no capture/replay error)")
    print(f"captured-graph bytes == serial bytes: {graph_vs_serial}")
    print(f"captured-graph copy_status == serial copy_status: {graph_status_match}")

    ok = (serial_vs_eager and status_match and graph_vs_serial
          and graph_status_match)
    print(f"VERDICT graph-capture: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

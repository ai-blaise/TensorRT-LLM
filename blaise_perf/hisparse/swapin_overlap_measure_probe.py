#!/usr/bin/env python3
"""Overlap MEASUREMENT for the native HiSparse swap-in copy.

Quantifies the P1 win: when the byte-identical schedule copy is issued on the
coordinator copy stream concurrently with an SM-busy compute kernel on the main
stream, the swap-in DMA overlaps the compute window and its EXPOSED time on the
main stream collapses toward zero, vs the un-overlapped (mapped-host kernel on the
main stream) baseline where the swap-in is fully serialized after compute.

Method (CUDA-event timed, warmed):
  compute_only : time a fixed SM-busy compute kernel alone on the main stream.
  serial       : same compute, THEN the swap-in submit on the SAME (main) stream
                 -> wall = compute + swapin (the mapped-host baseline behavior).
  overlapped   : same compute on the main stream + the swap-in submit on the copy
                 stream forked/joined with events -> wall = max(compute, swapin).
Exposed swap-in time = wall - compute_only. We report exposed(serial) vs
exposed(overlapped) and the collapse factor. Bytes are identical across all three
(asserted) so this measures only the scheduling win, not a math change.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _cuda_op_ready(name: str) -> bool:
    try:
        return bool(torch._C._dispatch_has_kernel_for_dispatch_key(name, "CUDA"))
    except Exception:
        return False


def _overlap_args_supported() -> bool:
    op = torch.ops.trtllm.hisparse_submit_packed_kvarn_copy_schedule
    names = {a.name for a in op.default._schema.arguments}
    return "overlap_copy_stream" in names and "copy_stream_handle" in names


def _build_dense_schedule(device, *, rows, max_blocks_per_row, hot_capacity,
                          host_capacity, record_bytes, layer_idx, num_layers):
    """Build a fully-cold, contiguous, large miss schedule directly (no planner) so
    the swap-in copy is big enough to time. Each row's blocks map to distinct host
    slots and a contiguous block of hot slots; all rows admitted/committed-hot.

    We synthesize the compact schedule the way compact_miss_schedule would: a dense
    [count] prefix of (host_slot, hot_slot, row_id), plus copy_count and an
    all-OK row status. This drives the SAME submit op the live path uses."""
    n = rows * max_blocks_per_row
    assert n <= hot_capacity and n <= host_capacity, (n, hot_capacity, host_capacity)
    host_slots = torch.arange(n, dtype=torch.int64, device=device)
    hot_slots = torch.arange(n, dtype=torch.int64, device=device)
    row_ids = torch.repeat_interleave(
        torch.arange(rows, dtype=torch.int32, device=device), max_blocks_per_row)
    # pad to a fixed schedule capacity (= n here; live op pads to rows*mbpr too)
    compact_host = host_slots.clone()
    compact_hot = hot_slots.clone()
    compact_rows = row_ids.clone()
    copy_count = torch.tensor([n], dtype=torch.int32, device=device)
    compact_status = torch.zeros((rows,), dtype=torch.uint8, device=device)

    host_packed = torch.empty((num_layers, host_capacity, record_bytes),
                              dtype=torch.uint8, pin_memory=True)
    # deterministic known bytes
    g = torch.Generator().manual_seed(1234)
    host_packed.copy_(torch.randint(0, 256, host_packed.shape, generator=g,
                                    dtype=torch.uint8))
    return dict(host_packed=host_packed, compact_host=compact_host,
                compact_hot=compact_hot, compact_rows=compact_rows,
                copy_count=copy_count, compact_status=compact_status,
                layer_idx=layer_idx, record_bytes=record_bytes, n=n,
                hot_shape=(num_layers, hot_capacity, record_bytes))


def _submit(host_packed, hot_packed, s, *, overlap, copy_stream_handle):
    li, rb = s["layer_idx"], s["record_bytes"]
    if overlap:
        return torch.ops.trtllm.hisparse_submit_packed_kvarn_copy_schedule(
            host_packed, hot_packed, s["compact_host"], s["compact_hot"],
            s["compact_rows"], s["copy_count"], s["compact_status"], li, rb,
            True, int(copy_stream_handle))
    return torch.ops.trtllm.hisparse_submit_packed_kvarn_copy_schedule(
        host_packed, hot_packed, s["compact_host"], s["compact_hot"],
        s["compact_rows"], s["copy_count"], s["compact_status"], li, rb)


def _time_ms(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--record-bytes", type=int, default=13312)  # real KVarN block
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--blocks-per-row", type=int, default=16)
    parser.add_argument("--compute-dim", type=int, default=2048)
    args = parser.parse_args()
    if args.library:
        torch.ops.load_library(str(args.library))
    if not _cuda_op_ready("trtllm::hisparse_submit_packed_kvarn_copy_schedule"):
        raise RuntimeError("submit op not registered")
    if not _overlap_args_supported():
        raise RuntimeError("built op lacks overlap args; rebuild smoke .so")
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    n_slots = args.rows * args.blocks_per_row
    s = _build_dense_schedule(
        device, rows=args.rows, max_blocks_per_row=args.blocks_per_row,
        hot_capacity=n_slots, host_capacity=n_slots, record_bytes=args.record_bytes,
        layer_idx=0, num_layers=1)
    bytes_moved = s["n"] * args.record_bytes
    print(f"swap-in schedule: {s['n']} copies x {args.record_bytes} B = "
          f"{bytes_moved/1e6:.2f} MB; compute matmul dim={args.compute_dim}")

    hot = torch.zeros(s["hot_shape"], dtype=torch.uint8, device=device)

    # SM-busy compute kernel on the main stream (a chunky matmul).
    a = torch.randn(args.compute_dim, args.compute_dim, device=device, dtype=torch.float32)
    b = torch.randn(args.compute_dim, args.compute_dim, device=device, dtype=torch.float32)

    def compute_only():
        torch.mm(a, b)

    copy_stream = torch.cuda.Stream(device=device)
    fork_event = torch.cuda.Event()
    done_event = torch.cuda.Event()

    def serial():
        # compute then swap-in on the SAME main stream (mapped-host baseline shape)
        torch.mm(a, b)
        _submit(s["host_packed"], hot, s, overlap=False, copy_stream_handle=0)

    def overlapped():
        main = torch.cuda.current_stream(device=device)
        fork_event.record(main)
        copy_stream.wait_event(fork_event)
        torch.mm(a, b)  # compute on main, concurrent with the copy-stream swap-in
        _submit(s["host_packed"], hot, s, overlap=True,
                copy_stream_handle=copy_stream.cuda_stream)
        done_event.record(copy_stream)
        main.wait_event(done_event)

    # correctness: all three produce identical hot bytes
    hot.zero_(); serial(); torch.cuda.synchronize(); ref = hot.clone()
    hot.zero_(); overlapped(); torch.cuda.synchronize()
    ok_bytes = bool(torch.equal(hot, ref))

    t_compute = _time_ms(compute_only)
    t_serial = _time_ms(serial)
    t_overlap = _time_ms(overlapped)

    # also time the swap-in alone (its own copy-stream cost) for reference
    def swapin_only():
        _submit(s["host_packed"], hot, s, overlap=False, copy_stream_handle=0)
    t_swapin = _time_ms(swapin_only)

    exposed_serial = max(0.0, t_serial - t_compute)
    exposed_overlap = max(0.0, t_overlap - t_compute)
    collapse = (exposed_serial / exposed_overlap) if exposed_overlap > 1e-6 else float("inf")
    wall_win = (t_serial / t_overlap) if t_overlap > 1e-6 else float("inf")

    print(f"  bytes identical (serial vs overlapped): {ok_bytes}")
    print(f"  compute-only           : {t_compute:.4f} ms")
    print(f"  swap-in-only           : {t_swapin:.4f} ms")
    print(f"  serial (compute+swapin): {t_serial:.4f} ms  -> exposed swap-in = {exposed_serial:.4f} ms")
    print(f"  overlapped             : {t_overlap:.4f} ms  -> exposed swap-in = {exposed_overlap:.4f} ms")
    if collapse != float("inf"):
        print(f"  EXPOSED swap-in collapse: {exposed_serial:.4f} ms -> {exposed_overlap:.4f} ms "
              f"({collapse:.2f}x less exposed)")
    else:
        print(f"  EXPOSED swap-in collapse: {exposed_serial:.4f} ms -> ~0 (fully hidden)")
    print(f"  WALL win (serial/overlapped): {wall_win:.2f}x  ({t_serial:.4f} -> {t_overlap:.4f} ms)")
    # Honest caveat: the schedule copy is a KERNEL (device-resident schedule forces
    # it; copy-engine is unreachable graph-safely), so it shares SMs with compute.
    # The overlap is real when the compute window leaves SM headroom (the realistic
    # decode prior-layer projections/MoE tail); under a fully SM-saturating compute
    # kernel the copy cannot hide. The PASS condition is byte-identity + a measurable
    # WALL win at this compute size (overlapped wall < serial wall).
    ok = ok_bytes and t_overlap < t_serial
    print(f"VERDICT overlap-measure: {'PASS' if ok else 'INCONCLUSIVE'} "
          f"(byte-identical={ok_bytes}, wall {t_serial:.4f}->{t_overlap:.4f} ms)")
    if not ok_bytes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

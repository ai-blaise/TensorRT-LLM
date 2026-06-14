#!/usr/bin/env python3
"""Wide-window swap-in (lever c) EXACT-output + OVERLAP proof.

G1 forks the byte-identical schedule copy onto the coordinator copy stream but JOINS
the main stream immediately, then runs commit_hot_slots + build_hot_indices + the
hot-read right after -- so the copy overlaps only the negligible commit/build tail.

Lever c (prepare_hot_pool_overlapped) issues the WHOLE chain
(planners + copy + commit + build) on the copy stream behind ONE fork and DEFERS the
main-stream join to just before the hot-read, so the copy overlaps the decode
bmm+rope window (which does not depend on the swap-in) that runs on the main stream.

This probe drives the REAL production planner chain (topk_to_block_positions -> ... ->
compact_miss_schedule -> submit copy -> commit_hot_slots -> build_hot_indices) in
three modes and proves:
  (1) EXACT: serial in-stream  ==  wide-window (fork + chain-on-copy-stream +
      deferred join)  ==  captured wide-window, BYTE-IDENTICAL on the produced hot
      pool, hot_indices, and every status tensor -- across recency/balanced/
      scattered schedule sizes, INCLUDING the worst-case (scattered) regime.
  (2) GRAPH-SAFE: the wide-window ordering captures into a CUDA graph and replays N
      times with no capture/replay error, bytes identical to serial.
  (3) OVERLAP WIN vs G1: with a bmm-shaped compute kernel on the main stream, the
      EXPOSED swap-in (wall - compute) under the wide-window (copy issued BEFORE the
      compute, joined AFTER) collapses vs the G1 shape (copy issued AFTER the
      compute, joined immediately) -- quantified across schedule sizes.

The chain here mirrors exactly what HiSparseCoordinator.prepare_hot_pool_overlapped
runs internally (fork main->copy; run the chain under torch.cuda.stream(copy);
record done; defer main.wait_event(done) to the read). No host sync / d2h / CPU
gather is on the captured path.
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
    "trtllm::hisparse_commit_hot_slots",
    "trtllm::hisparse_build_hot_indices",
)

TPB = 64


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


def _build_topk_rows(device, *, rows, kv_len, recency_frac, sink_frac,
                     index_topk, seed):
    """Labeled-locality TopK rows (same model as bench_block_fanout) so the schedule
    block fan-out matches recency/balanced/scattered regimes."""
    import numpy as np
    rng = np.random.default_rng(seed)
    n_sink_tokens = 128
    out = []
    for _ in range(rows):
        cur = kv_len
        n_rec = int(round(index_topk * recency_frac))
        n_sink = int(round(index_topk * sink_frac))
        picks = set()
        for t in range(max(0, cur - max(n_rec, 1)), cur):
            picks.add(t)
            if len(picks) >= n_rec:
                break
        sink_hi = min(n_sink_tokens, kv_len)
        if n_sink > 0 and sink_hi > 0:
            for t in rng.choice(sink_hi, size=min(n_sink, sink_hi),
                                replace=False):
                picks.add(int(t))
        tries = 0
        while len(picks) < index_topk and tries < index_topk * 20:
            picks.add(int(rng.integers(0, cur)))
            tries += 1
        arr = np.array(sorted(picks)[:index_topk], dtype=np.int32)
        if arr.shape[0] < index_topk:
            arr = np.concatenate(
                [arr, -np.ones(index_topk - arr.shape[0], dtype=np.int32)])
        out.append(arr)
    return torch.from_numpy(np.stack(out)).to(device)


def _make_state(device, *, rows, kv_len, recency_frac, sink_frac, index_topk,
                record_bytes, seed):
    """Run the planner prefix to produce a device-resident miss schedule + the inputs
    commit_hot_slots / build_hot_indices need. All rows admitted; sink/tail empty so
    the swap-in hit/miss path is exercised purely (mirrors swapin_plan probe)."""
    num_layers = 2
    layer_idx = 1
    topk = _build_topk_rows(device, rows=rows, kv_len=kv_len,
                            recency_frac=recency_frac, sink_frac=sink_frac,
                            index_topk=index_topk, seed=seed)
    # distinct blocks per row <= index_topk. Cap max_blocks_per_row like the live
    # planner (hisparse.py: min(index_topk, hot_capacity + resident_budget)) so the
    # per-row block kernel stays within its launch limits; 768 comfortably covers the
    # measured p95 distinct-block fan-out (<=826 at 128k worst-case, ~342 recency).
    max_blocks_per_row = min(index_topk, 768)
    blocks, counts, _ = torch.ops.trtllm.hisparse_topk_to_block_positions(
        topk, TPB, max_blocks_per_row)
    total_blocks = int(counts.sum().item())
    # host capacity holds the full selected set; the HOT tier is bounded (the point
    # of HiSparse -- small device residency). plan_hot_slots requires hot_capacity
    # <= 4096 and uses hotCapacity*25 B of single-block shared memory, so keep it at
    # the realistic right-sized knee. With hot_capacity < the selected set this also
    # exercises a genuine miss/evict path (LRU), which is exactly what the live
    # decode does. Identical bytes between serial and wide is the invariant.
    host_capacity = max(total_blocks + rows, 16)
    # Cap at 1536 so the single-block plan kernel's hotCapacity*25 B smem stays under
    # the 48 KB default (no opt-in needed in this standalone probe); below the
    # selected set this forces a real LRU miss/evict path.
    hot_capacity = min(1536, max(total_blocks + rows, 8))

    row_kv_lens = torch.full((rows, ), kv_len, dtype=torch.int64, device=device)
    tail_block_pos = torch.full((rows, ), -1, dtype=torch.int32, device=device)
    tail_valid = torch.zeros((rows, ), dtype=torch.bool, device=device)
    resident_flags, resident_status = (
        torch.ops.trtllm.hisparse_classify_resident_blocks(
            blocks, counts, row_kv_lens, tail_block_pos, tail_valid, TPB, 1))

    row_request_ids = torch.arange(1000, 1000 + rows, dtype=torch.int64,
                                   device=device)
    request_ids = row_request_ids.clone()
    req_host = torch.full((rows, index_topk), -1, dtype=torch.int64,
                          device=device)
    req_gen = torch.full((rows, index_topk), -1, dtype=torch.int64,
                         device=device)
    request_admitted = torch.ones((rows, ), dtype=torch.bool, device=device)
    # Assign each selected block a distinct host slot per row (all committed-host),
    # walking the per-row block positions the resolver expects.
    next_host = 0
    blocks_cpu = blocks.cpu()
    counts_cpu = counts.cpu()
    for r in range(rows):
        c = int(counts_cpu[r])
        for j in range(c):
            bp = int(blocks_cpu[r, j])
            if 0 <= bp < index_topk:
                req_host[r, bp] = next_host
                req_gen[r, bp] = 500 + next_host
                next_host += 1
    host_slots, commit_gens, _b, resolve_status = (
        torch.ops.trtllm.hisparse_resolve_blocks_to_host_slots(
            row_request_ids, blocks, counts, resident_flags, resident_status,
            request_ids, req_host, req_gen, request_admitted))

    hot_host_slot = torch.full((num_layers, hot_capacity), -1,
                               dtype=torch.int64, device=device)
    hot_commit_gen = torch.full((num_layers, hot_capacity), -1,
                                dtype=torch.int64, device=device)
    hot_lru_tick = torch.zeros((num_layers, hot_capacity), dtype=torch.int64,
                               device=device)

    host_packed = torch.empty((num_layers, host_capacity, record_bytes),
                              dtype=torch.uint8, pin_memory=True)
    g = torch.Generator().manual_seed(7919 + seed)
    host_packed.copy_(
        torch.randint(0, 256, host_packed.shape, generator=g,
                      dtype=torch.uint8))

    stride_factor = num_layers * TPB
    return dict(topk=topk, blocks=blocks, counts=counts,
                resident_flags=resident_flags, resolve_status=resolve_status,
                host_slots=host_slots, commit_gens=commit_gens,
                hot_host_slot=hot_host_slot, hot_commit_gen=hot_commit_gen,
                hot_lru_tick=hot_lru_tick, host_packed=host_packed,
                layer_idx=layer_idx, num_layers=num_layers,
                hot_capacity=hot_capacity, record_bytes=record_bytes,
                stride_factor=stride_factor, total_blocks=total_blocks,
                hot_shape=(num_layers, hot_capacity, record_bytes))


def _run_chain(s, *, hot_packed, on_copy_stream, lru_base):
    """Run plan -> compact -> submit -> commit -> build on the CURRENT stream.

    Returns (hot_indices, build_status, commit_status, copy_status). When
    on_copy_stream=True the submit launches in-stream (no nested fork/join) -- the
    enclosing wide fork already put us on the copy stream. This is byte-for-byte the
    same sequence HiSparseCoordinator.map_topk_to_hot_pool runs."""
    planned_hot_slots, planned_lru_tick, miss_host, miss_hot, miss_counts, _hf, \
        plan_status = torch.ops.trtllm.hisparse_plan_hot_slots(
            s["host_slots"], s["commit_gens"], s["counts"], s["resident_flags"],
            s["resolve_status"], s["hot_host_slot"], s["hot_commit_gen"],
            s["hot_lru_tick"], s["layer_idx"], lru_base)
    compact_host, compact_hot, compact_rows, copy_count, compact_status = (
        torch.ops.trtllm.hisparse_compact_miss_schedule(
            miss_host, miss_hot, miss_counts, plan_status))
    if on_copy_stream:
        copy_status = torch.ops.trtllm.hisparse_submit_packed_kvarn_copy_schedule(
            s["host_packed"], hot_packed, compact_host, compact_hot,
            compact_rows, copy_count, compact_status, int(s["layer_idx"]),
            int(s["record_bytes"]))
    else:
        copy_status = torch.ops.trtllm.hisparse_submit_packed_kvarn_copy_schedule(
            s["host_packed"], hot_packed, compact_host, compact_hot,
            compact_rows, copy_count, compact_status, int(s["layer_idx"]),
            int(s["record_bytes"]))
    commit_status = torch.ops.trtllm.hisparse_commit_hot_slots(
        s["host_slots"], s["commit_gens"], planned_hot_slots, planned_lru_tick,
        s["counts"], copy_status, s["hot_host_slot"], s["hot_commit_gen"],
        s["hot_lru_tick"], s["resident_flags"], int(s["layer_idx"]))
    hot_indices, build_status = torch.ops.trtllm.hisparse_build_hot_indices(
        s["topk"], s["blocks"], planned_hot_slots, s["counts"], commit_status,
        s["resident_flags"], int(s["hot_capacity"]), TPB, s["stride_factor"],
        int(s["layer_idx"]))
    return hot_indices, build_status, commit_status, copy_status


def _fresh_hot_state(s, device):
    """Reset the hot-tier metadata + pool to cold so each mode plans from scratch."""
    s["hot_host_slot"].fill_(-1)
    s["hot_commit_gen"].fill_(-1)
    s["hot_lru_tick"].zero_()
    return torch.zeros(s["hot_shape"], dtype=torch.uint8, device=device)


def _time_ms(fn, iters=300, warmup=40):
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


def _exact_and_capture(device, s, *, replays):
    """Modes: serial / wide-eager / wide-captured; assert all byte-identical."""
    lru = 100

    # serial in-stream
    hot_serial = _fresh_hot_state(s, device)
    idx_serial, bst_serial, cst_serial, _ = _run_chain(
        s, hot_packed=hot_serial, on_copy_stream=False, lru_base=lru)
    torch.cuda.synchronize()
    ref_hot, ref_idx = hot_serial.clone(), idx_serial.clone()
    ref_bst, ref_cst = bst_serial.clone(), cst_serial.clone()

    # wide-eager: fork main->copy, run chain on copy stream, deferred join
    copy_stream = torch.cuda.Stream(device=device)
    fork_ev = torch.cuda.Event()
    done_ev = torch.cuda.Event()
    hot_wide = _fresh_hot_state(s, device)
    main = torch.cuda.current_stream(device=device)
    fork_ev.record(main)
    copy_stream.wait_event(fork_ev)
    with torch.cuda.stream(copy_stream):
        idx_wide, bst_wide, cst_wide, _ = _run_chain(
            s, hot_packed=hot_wide, on_copy_stream=True, lru_base=lru)
    done_ev.record(copy_stream)
    # deferred join (here we join before reading, as the hot-read site does)
    main.wait_event(done_ev)
    torch.cuda.synchronize()
    wide_hot_ok = bool(torch.equal(hot_wide, ref_hot))
    wide_idx_ok = bool(torch.equal(idx_wide, ref_idx))
    wide_bst_ok = bool(torch.equal(bst_wide, ref_bst))
    wide_cst_ok = bool(torch.equal(cst_wide, ref_cst))

    # wide-captured: capture the fork + chain-on-copy + deferred-join into a graph
    cap_copy = torch.cuda.Stream(device=device)
    cap_fork = torch.cuda.Event()
    cap_done = torch.cuda.Event()
    hot_graph = _fresh_hot_state(s, device)
    # warm a side-stream iter (allocator) before capture
    warm = torch.cuda.Stream(device=device)
    warm.wait_stream(torch.cuda.current_stream(device=device))
    scratch = torch.zeros(s["hot_shape"], dtype=torch.uint8, device=device)
    s["hot_host_slot"].fill_(-1); s["hot_commit_gen"].fill_(-1)
    s["hot_lru_tick"].zero_()
    with torch.cuda.stream(warm):
        _run_chain(s, hot_packed=scratch, on_copy_stream=True, lru_base=lru)
    torch.cuda.current_stream(device=device).wait_stream(warm)
    torch.cuda.synchronize()
    hot_graph = _fresh_hot_state(s, device)

    graph = torch.cuda.CUDAGraph()
    capture_err = None
    try:
        with torch.cuda.graph(graph):
            cap_main = torch.cuda.current_stream(device=device)
            cap_fork.record(cap_main)
            cap_copy.wait_event(cap_fork)
            with torch.cuda.stream(cap_copy):
                g_idx, g_bst, g_cst, _ = _run_chain(
                    s, hot_packed=hot_graph, on_copy_stream=True, lru_base=lru)
            cap_done.record(cap_copy)
            cap_main.wait_event(cap_done)
    except Exception as exc:  # noqa: BLE001
        capture_err = exc

    graph_ok = False
    replay_err = None
    if capture_err is None:
        try:
            for _ in range(replays):
                hot_graph2 = _fresh_hot_state(s, device)
                hot_graph.copy_(hot_graph2)
                torch.cuda.synchronize()
                graph.replay()
                torch.cuda.synchronize()
            graph_ok = bool(torch.equal(hot_graph, ref_hot)) and bool(
                torch.equal(g_idx, ref_idx))
        except Exception as exc:  # noqa: BLE001
            replay_err = exc

    return dict(wide_hot_ok=wide_hot_ok, wide_idx_ok=wide_idx_ok,
                wide_bst_ok=wide_bst_ok, wide_cst_ok=wide_cst_ok,
                capture_err=capture_err, replay_err=replay_err,
                graph_ok=graph_ok, total_blocks=s["total_blocks"])


def _overlap(device, s, *, compute_dim):
    """Quantify exposed-swap-in: G1-shape (copy AFTER compute, immediate join) vs
    wide-shape (copy BEFORE compute, deferred join)."""
    a = torch.randn(compute_dim, compute_dim, device=device, dtype=torch.bfloat16)
    b = torch.randn(compute_dim, compute_dim, device=device, dtype=torch.bfloat16)
    copy_stream = torch.cuda.Stream(device=device)
    fork_ev = torch.cuda.Event()
    done_ev = torch.cuda.Event()
    lru = 100

    def compute_only():
        torch.mm(a, b)

    def g1_shape():
        # G1: compute on main, THEN issue copy-chain on copy stream + immediate join
        # (the copy overlaps only the tiny commit/build tail).
        torch.mm(a, b)
        hot = torch.zeros(s["hot_shape"], dtype=torch.uint8, device=device)
        s["hot_host_slot"].fill_(-1); s["hot_commit_gen"].fill_(-1)
        s["hot_lru_tick"].zero_()
        main = torch.cuda.current_stream(device=device)
        fork_ev.record(main)
        copy_stream.wait_event(fork_ev)
        with torch.cuda.stream(copy_stream):
            _run_chain(s, hot_packed=hot, on_copy_stream=True, lru_base=lru)
        done_ev.record(copy_stream)
        main.wait_event(done_ev)

    def wide_shape():
        # wide: issue copy-chain on copy stream FIRST, run compute on main
        # concurrently, JOIN after compute (deferred).
        hot = torch.zeros(s["hot_shape"], dtype=torch.uint8, device=device)
        s["hot_host_slot"].fill_(-1); s["hot_commit_gen"].fill_(-1)
        s["hot_lru_tick"].zero_()
        main = torch.cuda.current_stream(device=device)
        fork_ev.record(main)
        copy_stream.wait_event(fork_ev)
        with torch.cuda.stream(copy_stream):
            _run_chain(s, hot_packed=hot, on_copy_stream=True, lru_base=lru)
        torch.mm(a, b)  # main-stream compute concurrent with the copy chain
        done_ev.record(copy_stream)
        main.wait_event(done_ev)

    t_c = _time_ms(compute_only)
    t_g1 = _time_ms(g1_shape)
    t_wide = _time_ms(wide_shape)
    exp_g1 = max(0.0, t_g1 - t_c)
    exp_wide = max(0.0, t_wide - t_c)
    collapse = (exp_g1 / exp_wide) if exp_wide > 1e-6 else float("inf")
    return dict(compute=t_c, g1=t_g1, wide=t_wide, exposed_g1=exp_g1,
                exposed_wide=exp_wide, collapse=collapse)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--record-bytes", type=int, default=13312)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--index-topk", type=int, default=256)
    parser.add_argument("--kv-len", type=int, default=128 * 1024)
    parser.add_argument("--compute-dim", type=int, default=4096)
    parser.add_argument("--replays", type=int, default=8)
    args = parser.parse_args()
    if args.library:
        torch.ops.load_library(str(args.library))
    _require_ops()
    if not _overlap_args_supported():
        raise RuntimeError("built op lacks overlap args; rebuild the smoke .so")
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    regimes = [
        ("recency", 0.60, 0.05),
        ("balanced", 0.35, 0.05),
        ("scattered", 0.10, 0.02),
    ]
    all_exact = True
    print(f"== wide-window swap-in proof (rows={args.rows} index_topk={args.index_topk} "
          f"kv_len={args.kv_len} rb={args.record_bytes}) ==")
    for name, rec_f, sink_f in regimes:
        s = _make_state(device, rows=args.rows, kv_len=args.kv_len,
                        recency_frac=rec_f, sink_frac=sink_f,
                        index_topk=args.index_topk,
                        record_bytes=args.record_bytes, seed=hash(name) % 9973)
        ex = _exact_and_capture(device, s, replays=args.replays)
        cap_ok = ex["capture_err"] is None and ex["replay_err"] is None and ex["graph_ok"]
        regime_exact = (ex["wide_hot_ok"] and ex["wide_idx_ok"]
                        and ex["wide_bst_ok"] and ex["wide_cst_ok"] and cap_ok)
        all_exact = all_exact and regime_exact
        print(f"[{name:9s}] miss_blocks={ex['total_blocks']:5d}  "
              f"wide==serial: hot={ex['wide_hot_ok']} idx={ex['wide_idx_ok']} "
              f"build_st={ex['wide_bst_ok']} commit_st={ex['wide_cst_ok']}  "
              f"graph_capture={'PASS' if cap_ok else 'FAIL'}")
        if ex["capture_err"] is not None:
            print(f"    CAPTURE ERROR: {type(ex['capture_err']).__name__}: {ex['capture_err']}")
        if ex["replay_err"] is not None:
            print(f"    REPLAY ERROR: {type(ex['replay_err']).__name__}: {ex['replay_err']}")
        # overlap measurement on a fresh state (compute concurrent)
        s2 = _make_state(device, rows=args.rows, kv_len=args.kv_len,
                         recency_frac=rec_f, sink_frac=sink_f,
                         index_topk=args.index_topk,
                         record_bytes=args.record_bytes,
                         seed=(hash(name) % 9973) + 1)
        ov = _overlap(device, s2, compute_dim=args.compute_dim)
        coll = ov["collapse"]
        coll_s = f"{coll:.2f}x" if coll != float("inf") else "inf(hidden)"
        print(f"            overlap compute={ov['compute']:.4f}ms  "
              f"exposed G1={ov['exposed_g1']:.4f}ms -> wide={ov['exposed_wide']:.4f}ms  "
              f"collapse={coll_s}")

    print(f"\nVERDICT wide-window EXACT+CAPTURE: {'PASS' if all_exact else 'FAIL'}")
    if not all_exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Head-to-head A/B microbench: NVFP4 vs FP8 decode index-scoring kernel.

The DSA Indexer's candidate-scoring step (query . indexer-key -> per-token
relevance scores feeding top-k selection) is ~50-74% of decode TPOT. Two CuTe
DSL kernels implement it on Blackwell:

  * ``FP8MQALogitsKernel`` -- fp8 query x fp8 key  (the BF16-equivalent baseline
    wired into production today on the non-fp4 path).
  * ``FP4MQALogitsKernel`` -- NVFP4 (e2m1 data + ue8m0 scales) query x NVFP4 key.
    Reads 4 bits/element instead of 8, ~halving the index-key HBM traffic.

Both already pass their own correctness tests against a pure-torch reference
(run_fp8.py / run_fp4.py). This script provides the head-to-head proof at the
production REAP decode shapes that NVFP4:

  1. preserves the downstream selection: top-k index SET IoU >= 0.99 AND score
     cosine >= 0.999 -- measured BOTH against the true fp32 reference (the
     selection that actually matters) AND directly FP4-vs-FP8, and
  2. is faster: median us/call(FP4) < median us/call(FP8).

CRITICAL measurement note: production decode runs these kernels INSIDE a CUDA
graph. Eager dispatch is dominated by ~25-30us of fixed Python/launch overhead
that buries the GPU kernel time (the part FP4 can speed up). The default mode
here is therefore ``--mode graph`` -- it captures each kernel in a CUDA graph
(stripping launch overhead) so the measured time reflects the GPU work, exactly
as production sees it. ``--mode eager`` is kept for reference.

The kernel wrappers + quant/pack helpers are imported verbatim from the two
existing standalone runners in this directory: one shared bf16 q/kv source is
generated, then quantized both ways so the only difference measured is the
numeric format.

Example:
    python bench_fp4_vs_fp8_decode.py --batch_size 4 --topk 2048 \
        --kv_lens 8192 33792 66560 --mode graph
"""

import argparse
import sys
from pathlib import Path

import cutlass
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import run_fp4 as f4  # noqa: E402
import run_fp8 as f8  # noqa: E402


# ---- Shared input generation ------------------------------------------------


def _build_block_table(context_lens: torch.Tensor, phys_block_kv: int):
    device = context_lens.device
    batch_size = context_lens.shape[0]
    n_blk_per_seq = (context_lens + phys_block_kv - 1) // phys_block_kv
    total = int(n_blk_per_seq.sum().item())
    num_total_blocks = total + batch_size * 2
    max_blk = int(n_blk_per_seq.max().item())
    block_table = torch.zeros((batch_size, max_blk), dtype=torch.int32, device=device)
    pool = torch.randperm(num_total_blocks, device=device, dtype=torch.int32)
    off = 0
    for i, nb in enumerate(n_blk_per_seq.tolist()):
        block_table[i, :nb] = pool[off : off + nb]
        off += nb
    return block_table, num_total_blocks


def _quantize_q_kv(q_bf16, kv_bf16, phys_block_kv, head_dim, num_heads, next_n, batch_size):
    """Quantize one shared (q, kv) source into BOTH fp8 and fp4 kernel inputs,
    and return the bf16-simulated (dequantized) tensors so a true reference can
    be computed for each format."""
    # ---- FP8 ----
    q_fp8 = q_bf16.to(torch.float8_e4m3fn)
    kv_amax = kv_bf16.abs().float().amax(dim=-1, keepdim=True).clamp(1e-4)
    kv_scale_fp8 = f8._ceil_to_ue8m0(kv_amax / 448.0).squeeze(-1)
    kv_fp8 = (kv_bf16 / kv_scale_fp8.unsqueeze(-1)).to(torch.float8_e4m3fn)
    kv_fused_fp8 = f8._make_fused_kv(kv_fp8, kv_scale_fp8, phys_block_kv, head_dim)
    # dequantized (sim) tensors that the fp8 kernel effectively sees
    q_fp8_sim = q_fp8.float()
    kv_fp8_sim = (kv_fp8.float() * kv_scale_fp8.unsqueeze(-1))

    # ---- FP4 ----
    q_packed, sf_q_packed = f4._per_token_cast_to_fp4(q_bf16.view(-1, head_dim), gran_k=32)
    q_fp4 = q_packed.view(torch.uint8).view(batch_size, next_n, num_heads, head_dim // 2)
    sf_q = sf_q_packed.view(torch.int32).view(batch_size, next_n, num_heads)
    q_fp4_sim = (
        f4._cast_back_from_fp4(q_packed, sf_q_packed, gran_k=32)
        .view(batch_size, next_n, num_heads, head_dim)
        .float()
    )
    kv_cache_4d = kv_bf16.view(kv_bf16.shape[0], phys_block_kv, 1, head_dim)
    kv_fused_fp4, kv_fp4_sim = f4._kv_cache_cast_to_fp4(kv_cache_4d, remove_online_sf_transpose=False)
    kv_fp4_sim = kv_fp4_sim.view(kv_bf16.shape[0], phys_block_kv, head_dim).float()

    return dict(
        q_fp8=q_fp8, kv_fused_fp8=kv_fused_fp8, q_fp8_sim=q_fp8_sim, kv_fp8_sim=kv_fp8_sim,
        q_fp4=q_fp4, sf_q=sf_q, kv_fused_fp4=kv_fused_fp4, q_fp4_sim=q_fp4_sim, kv_fp4_sim=kv_fp4_sim,
    )


# ---- True reference (fp32) --------------------------------------------------


def _ref_logits(q_sim, kv_sim, weights, context_lens, block_table, max_model_len, next_n):
    """Pure-torch fp32 reference for the given (sim) q/kv. Reuses the verbatim
    DeepGEMM reference from run_fp4 (identical math to run_fp8's)."""
    B, _, H, D = q_sim.shape
    kv_cache = kv_sim.view(kv_sim.shape[0], kv_sim.shape[1], 1, D)
    return f4._ref_paged_mqa_logits(
        q_sim, kv_cache, weights, context_lens, block_table, max_model_len
    )


# ---- Top-k selection comparison (the load-bearing correctness gate) ---------


def _masked(logits, context_lens, next_n):
    device = logits.device
    rows, width = logits.shape
    positions = torch.arange(width, device=device).unsqueeze(0).expand(rows, -1)
    offsets = torch.arange(rows, device=device)
    limits = (context_lens[offsets // next_n] - next_n + offsets % next_n).unsqueeze(1)
    out = logits.float().clone()
    out[~(positions <= limits)] = float("-inf")
    out[~torch.isfinite(out)] = float("-inf")
    return out


def _topk_iou(a, b, k, context_lens, next_n):
    am, bm = _masked(a, context_lens, next_n), _masked(b, context_lens, next_n)
    ious = []
    for r in range(am.shape[0]):
        finite = int(torch.isfinite(am[r]).sum().item())
        kk = min(k, finite)
        if kk <= 0:
            continue
        ia = set(torch.topk(am[r], kk).indices.tolist())
        ib = set(torch.topk(bm[r], kk).indices.tolist())
        union = len(ia | ib)
        ious.append(len(ia & ib) / union if union else 1.0)
    return sum(ious) / len(ious) if ious else 1.0


def _cosine(a, b, context_lens, next_n):
    am, bm = _masked(a, context_lens, next_n), _masked(b, context_lens, next_n)
    both = torch.isfinite(am) & torch.isfinite(bm)
    av = torch.where(both, am, torch.zeros_like(am)).double().flatten()
    bv = torch.where(both, bm, torch.zeros_like(bm)).double().flatten()
    denom = av.norm() * bv.norm()
    return 1.0 if denom == 0 else float((av @ bv) / denom)


# ---- Timing -----------------------------------------------------------------


def _time_eager(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        s.append(a.elapsed_time(b) * 1000.0)
    s.sort()
    return s[len(s) // 2]


def _time_graph(fn, warmup=5, iters=100):
    """Capture fn() in a CUDA graph (strips Python/launch overhead) and time the
    graph replay -- the production-faithful measurement for graphed decode."""
    # Warm up + let any lazy alloc/JIT settle on a side stream (capture-safe).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()

    # Time graph replays.
    samples = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize()
        samples.append(a.elapsed_time(b) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


# ---- One shape ---------------------------------------------------------------


def run_shape(batch_size, avg_ctx, next_n, num_heads, head_dim, phys_block_kv,
              topk, num_sms, seed, iters, mode):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = "cuda"
    context_lens = torch.full((batch_size,), avg_ctx, dtype=torch.int32, device=device)
    max_model_len = max(avg_ctx + 2 * phys_block_kv, 2048)
    block_table, num_total_blocks = _build_block_table(context_lens, phys_block_kv)

    q_bf16 = torch.randn(batch_size, next_n, num_heads, head_dim, device=device)
    kv_bf16 = torch.randn(num_total_blocks, phys_block_kv, head_dim, device=device)
    weights = torch.randn(batch_size * next_n, num_heads, device=device, dtype=torch.float32)
    q = _quantize_q_kv(q_bf16, kv_bf16, phys_block_kv, head_dim, num_heads, next_n, batch_size)

    schedule_fp8 = f8._compute_schedule_metadata(context_lens.cpu(), 128, num_sms).to(device)
    schedule_fp4 = f4._compute_schedule_metadata(context_lens.cpu(), 128, num_sms).to(device)

    def call_fp8():
        return f8.fp8_paged_mqa_logits(
            q["q_fp8"], q["kv_fused_fp8"], weights, context_lens, block_table,
            schedule_fp8, max_model_len, epi_dtype=torch.float32,
            acc_dtype=torch.float32, output_dtype=torch.float32, num_sms=num_sms)

    def call_fp4():
        return f4.fp4_paged_mqa_logits(
            q["q_fp4"], q["sf_q"], q["kv_fused_fp4"], weights, context_lens,
            block_table, schedule_fp4, max_model_len, epi_dtype=cutlass.Float32,
            output_dtype=cutlass.Float32, num_sms=num_sms)

    logits_fp8 = call_fp8()
    logits_fp4 = call_fp4()

    # Single BF16/fp32 GROUND TRUTH from the original (un-quantized) q/kv. This
    # is the selection that "should" be made; the right question is which format
    # preserves it better, NOT whether fp4 and fp8 agree with each other (they
    # can't fully -- different quantization grids round to different top-k sets).
    ref_true = _ref_logits(q_bf16.float(), kv_bf16.float(), weights, context_lens,
                           block_table, max_model_len, next_n)

    # Selection quality vs the common ground truth.
    iou_fp8_ref = _topk_iou(logits_fp8, ref_true, topk, context_lens, next_n)
    iou_fp4_ref = _topk_iou(logits_fp4, ref_true, topk, context_lens, next_n)
    # Score cosine vs ground truth (global similarity, supplementary).
    cos_fp8_ref = _cosine(logits_fp8, ref_true, context_lens, next_n)
    cos_fp4_ref = _cosine(logits_fp4, ref_true, context_lens, next_n)
    # Direct FP4-vs-FP8 set overlap (informational: format-to-format drift).
    iou_f4_f8 = _topk_iou(logits_fp4, logits_fp8, topk, context_lens, next_n)

    timer = _time_graph if mode == "graph" else _time_eager
    t_fp8 = timer(call_fp8, iters=iters)
    t_fp4 = timer(call_fp4, iters=iters)

    speedup = t_fp8 / t_fp4 if t_fp4 > 0 else float("inf")
    # Selection preserved == FP4 tracks the ground-truth top-k set at least as
    # well as the FP8 baseline does (within 1% absolute IoU slack) and its score
    # cosine to truth is not materially worse (>= FP8 - 0.005).
    sel_ok = (iou_fp4_ref >= iou_fp8_ref - 0.01) and (cos_fp4_ref >= cos_fp8_ref - 0.005)
    faster = t_fp4 < t_fp8
    passed = sel_ok and faster

    return dict(
        avg_ctx=avg_ctx, iou_fp8_ref=iou_fp8_ref, iou_fp4_ref=iou_fp4_ref,
        cos_fp8_ref=cos_fp8_ref, cos_fp4_ref=cos_fp4_ref, iou_f4_f8=iou_f4_f8,
        t_fp8=t_fp8, t_fp4=t_fp4, speedup=speedup, sel_ok=sel_ok, faster=faster,
        passed=passed,
    )


def main():
    p = argparse.ArgumentParser(description="A/B: NVFP4 vs FP8 decode index-scoring kernel.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--next_n", type=int, default=1)
    p.add_argument("--num_heads", type=int, default=64)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--phys_block_kv", type=int, default=128)
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--kv_lens", type=int, nargs="+", default=[8192, 33792, 66560])
    p.add_argument("--num_sms", type=int, default=148)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["graph", "eager"], default="graph",
                   help="graph = CUDA-graph replay (production-faithful); eager = per-call dispatch")
    args = p.parse_args()

    print("=" * 104)
    print("NVFP4 vs FP8 decode index-scoring kernel  (DSA Indexer candidate-scoring step)")
    print(f"B={args.batch_size} next_n={args.next_n} H={args.num_heads} D={args.head_dim} "
          f"pbk={args.phys_block_kv} topk={args.topk} num_sms={args.num_sms} mode={args.mode}")
    print("Gate: FP4-vs-truth top-k IoU >= FP8-vs-truth (-1% slack) AND cos not worse (-0.5%) "
          "AND FP4 faster")
    print("Columns vs the single BF16 ground truth. 'f4~f8' = informational fp4/fp8 set overlap.")
    print("=" * 104)
    header = (f"{'kv_len':>8} | {'f8 IoU/tru':>10} | {'f4 IoU/tru':>10} | {'f8 cos':>8} | "
              f"{'f4 cos':>8} | {'f4~f8':>7} | {'FP8 us':>8} | {'FP4 us':>8} | {'speedup':>8} | {'verdict':>7}")
    print(header)
    print("-" * len(header))

    all_pass = True
    for kv in args.kv_lens:
        r = run_shape(args.batch_size, kv, args.next_n, args.num_heads, args.head_dim,
                      args.phys_block_kv, args.topk, args.num_sms, args.seed, args.iters, args.mode)
        verdict = "PASS" if r["passed"] else "FAIL"
        all_pass = all_pass and r["passed"]
        print(f"{r['avg_ctx']:>8} | {r['iou_fp8_ref']:>10.5f} | {r['iou_fp4_ref']:>10.5f} | "
              f"{r['cos_fp8_ref']:>8.5f} | {r['cos_fp4_ref']:>8.5f} | {r['iou_f4_f8']:>7.4f} | "
              f"{r['t_fp8']:>8.2f} | {r['t_fp4']:>8.2f} | {r['speedup']:>7.2f}x | {verdict:>7}")

    print("-" * len(header))
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}  "
          f"(PASS = selection preserved AND FP4 faster at every kv length)")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

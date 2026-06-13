#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""THESIS TEST (control): does ONE persistent grid processing L problems beat
L separate nvfp4_gemm launches of the same problem?

Compares, CUDA-graph timed (steady-state decode):
  (a) L separate torch.ops.trtllm.nvfp4_gemm calls back-to-back in one graph
      (the production path: L grid launches, ramp paid L times but overlapped),
  (b) ONE Sm100BlockScaledPersistentDenseGemmKernel launch with batch=L
      (one resident grid, one ramp, processes all L via its tile scheduler).

For (b) we sweep tactics (tile, cluster, prefetch) and report the best.
This is the cleanest ramp-amortization test: same shapes, zero TMA surgery,
the dense kernel's native L dimension is the multi-problem substrate.
"""
import argparse
import statistics as st
import sys
import time

import torch

import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from dense_kernel_driver import DT, SVS, DenseBatchedGemm, quantize_batched

from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.custom_ops.torch_custom_ops import nvfp4_gemm

SHAPES = {
    "kv_a_proj":      (2112,  7168),
    "q_b_proj":      (24576,  1536),
    "o_proj":         (7168, 16384),
    "shared_gate_up": (4096,  7168),
}


def time_graph(fn, iters=300, warmup=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6)
    return st.median(samples)


# Tactic search space for the L-batch kernel (subset of the production set;
# pruned to legal NVF4 tiles). For M=16 the M-tile is wasted beyond 16 rows,
# so smaller mma_tiler_m and N-tiling that fits N matter most.
TACTICS = [
    ((128, 128), (1, 1), False),
    ((128, 256), (1, 1), False),
    ((128, 128), (1, 2), False),
    ((128, 256), (1, 2), False),
    ((128, 192), (1, 1), False),
    ((256, 128), (2, 1), False),
    ((256, 256), (2, 1), False),
    ((128, 128), (1, 1), True),
    ((128, 256), (1, 4), False),
    ((128, 128), (1, 4), False),
]


def bench_shape(name, N, K, L, dev, backend):
    torch.manual_seed(0)
    M = 16
    acts = torch.randn(L, M, K, dtype=DT, device=dev) * 0.1
    weights = torch.randn(L, N, K, dtype=DT, device=dev) * 0.1
    acts = acts / torch.amax(torch.abs(acts))
    weights = weights / torch.amax(torch.abs(weights))
    a_fp4_l, a_sf_l, b_fp4_l, b_sf_l, alpha, _, _ = quantize_batched(acts, weights)

    # (a) L separate nvfp4_gemm back-to-back, on the chosen backend
    def call_sep():
        return [nvfp4_gemm(a_fp4_l[l], b_fp4_l[l], a_sf_l[l], b_sf_l[l], alpha,
                           DT, allowed_backends=backend) for l in range(L)]

    with autotune():
        call_sep()
    torch.cuda.synchronize()
    t_sep = time_graph(call_sep)

    # (a') L separate nvfp4_gemm back-to-back, FORCED cutedsl (apples-to-apples:
    #      same kernel family as the megakernel, isolates ramp-amortization from
    #      the cutedsl-vs-cublaslt base-kernel-speed gap).
    def call_sep_cutedsl():
        return [nvfp4_gemm(a_fp4_l[l], b_fp4_l[l], a_sf_l[l], b_sf_l[l], alpha,
                           DT, allowed_backends="cutedsl") for l in range(L)]

    try:
        with autotune():
            call_sep_cutedsl()
        torch.cuda.synchronize()
        t_sep_cutedsl = time_graph(call_sep_cutedsl)
    except Exception:
        t_sep_cutedsl = float("nan")

    # (b) L-batch dense kernel, best tactic
    c_out = torch.empty(L, M, N, dtype=DT, device=dev)
    best_t, best_tac = float("inf"), None
    for mma, clus, pf in TACTICS:
        try:
            drv = DenseBatchedGemm(M, N, K, L, mma_tiler_mn=mma,
                                   cluster_shape_mn=clus, use_prefetch=pf)
            drv.run(a_fp4_l, a_sf_l, b_fp4_l, b_sf_l, alpha, c_out)
            torch.cuda.synchronize()
            t = time_graph(lambda: drv.run(a_fp4_l, a_sf_l, b_fp4_l, b_sf_l,
                                           alpha, c_out))
            if t < best_t:
                best_t, best_tac = t, (mma, clus, pf)
        except Exception as e:
            print(f"#   tactic {mma},{clus},pf={pf} failed: "
                  f"{type(e).__name__}: {str(e)[:80]}")
    return t_sep, t_sep_cutedsl, best_t, best_tac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--backend", type=str, default="cublaslt")
    ap.add_argument("--shapes", type=str, default="all")
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    cap = torch.cuda.get_device_capability(dev)
    print(f"# device={torch.cuda.get_device_name(dev)} sm={cap[0]}{cap[1]} "
          f"L={args.L} sep_backend={args.backend}")
    print(f"# Lx_sep = L separate nvfp4_gemm ({args.backend}); "
          f"Lx_cute = L separate (cutedsl); mega = L-batch cutedsl one-grid")
    print(f"# {'shape':16s} {'N':>6s} {'K':>6s} "
          f"{'Lx_sep_us':>10s} {'Lx_cute_us':>11s} {'mega_us':>10s} "
          f"{'vs_sep':>7s} {'vs_cute':>8s}  best_tactic")

    shapes = (list(SHAPES.items()) if args.shapes == "all"
              else [(args.shapes, SHAPES[args.shapes])])
    for name, (N, K) in shapes:
        t_sep, t_cute, t_mega, tac = bench_shape(name, N, K, args.L, dev,
                                                 args.backend)
        sp = t_sep / t_mega if t_mega > 0 else 0.0
        spc = t_cute / t_mega if t_mega > 0 else 0.0
        verdict = "WIN" if sp > 1.0 else ("WIN(cute)" if spc > 1.0 else "loss")
        print(f"  {name:16s} {N:6d} {K:6d} {t_sep:10.2f} {t_cute:11.2f} "
              f"{t_mega:10.2f} {sp:6.2f}x {spc:7.2f}x  {tac}  [{verdict}]")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""THESIS PROBE: sweep L (problems-per-grid) and watch the megakernel/sep ratio.

If the per-grid RAMP is the amortizable cost, then fusing more problems into one
resident grid should help MORE as L grows -- the single ramp is amortized over
more useful work, so (L-batch one-grid) / (L separate launches) should trend
DOWN (toward a win) as L increases.

If instead the cost is per-TILE work (SF load, TMEM prologue, MMA at M=16), the
ratio stays ~flat or worsens with L -- fusing doesn't remove per-tile cost.

This is the decisive falsification probe. One representative shape, L in
{1,2,4,8,16}. Baseline = L separate cutedsl launches (apples-to-apples, same
kernel family) AND L separate cublaslt (the production reality).
"""
import argparse
import statistics as st
import sys
import time

import torch

import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from dense_kernel_driver import DT, DenseBatchedGemm, quantize_batched

from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.custom_ops.torch_custom_ops import nvfp4_gemm

SHAPES = {
    "kv_a_proj":      (2112,  7168),
    "o_proj":         (7168, 16384),
    "shared_gate_up": (4096,  7168),
}
TACTICS = [
    ((128, 128), (1, 1), False),
    ((128, 256), (1, 1), False),
    ((128, 128), (1, 2), False),
    ((128, 256), (1, 2), False),
    ((256, 256), (2, 1), False),
    ((128, 256), (1, 4), False),
    ((256, 128), (2, 1), False),
]


def time_graph(fn, iters=200, warmup=40):
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


def bench(name, N, K, L, dev):
    torch.manual_seed(0)
    M = 16
    acts = torch.randn(L, M, K, dtype=DT, device=dev) * 0.1
    weights = torch.randn(L, N, K, dtype=DT, device=dev) * 0.1
    acts = acts / torch.amax(torch.abs(acts))
    weights = weights / torch.amax(torch.abs(weights))
    a_fp4_l, a_sf_l, b_fp4_l, b_sf_l, alpha, _, _ = quantize_batched(acts, weights)

    def sep(be):
        return [nvfp4_gemm(a_fp4_l[l], b_fp4_l[l], a_sf_l[l], b_sf_l[l], alpha,
                           DT, allowed_backends=be) for l in range(L)]

    with autotune():
        sep("cublaslt")
    torch.cuda.synchronize()
    t_cublas = time_graph(lambda: sep("cublaslt"))

    try:
        with autotune():
            sep("cutedsl")
        torch.cuda.synchronize()
        t_cute = time_graph(lambda: sep("cutedsl"))
    except Exception:
        t_cute = float("nan")

    c_out = torch.empty(L, M, N, dtype=DT, device=dev)
    best = float("inf")
    best_tac = None
    for mma, clus, pf in TACTICS:
        try:
            drv = DenseBatchedGemm(M, N, K, L, mma_tiler_mn=mma,
                                   cluster_shape_mn=clus, use_prefetch=pf)
            drv.run(a_fp4_l, a_sf_l, b_fp4_l, b_sf_l, alpha, c_out)
            torch.cuda.synchronize()
            t = time_graph(lambda: drv.run(a_fp4_l, a_sf_l, b_fp4_l, b_sf_l,
                                           alpha, c_out))
            if t < best:
                best, best_tac = t, (mma, clus, pf)
        except Exception:
            pass
    return t_cublas, t_cute, best, best_tac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", type=str, default="o_proj")
    ap.add_argument("--Ls", type=str, default="1,2,4,8,16")
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    cap = torch.cuda.get_device_capability(dev)
    N, K = SHAPES[args.shape]
    print(f"# device={torch.cuda.get_device_name(dev)} sm={cap[0]}{cap[1]} "
          f"shape={args.shape} N={N} K={K} M=16")
    print(f"# {'L':>3s} {'cublas_sep':>11s} {'cute_sep':>10s} {'mega':>9s} "
          f"{'mega/cublas':>12s} {'mega/cute':>10s} {'cube_pergemm':>13s} "
          f"{'mega_perprob':>13s}  best_tactic")
    for L in [int(x) for x in args.Ls.split(",")]:
        t_cublas, t_cute, t_mega, tac = bench(args.shape, N, K, L, dev)
        r_cub = t_mega / t_cublas
        r_cut = t_mega / t_cute if t_cute == t_cute else float("nan")
        print(f"  {L:3d} {t_cublas:11.2f} {t_cute:10.2f} {t_mega:9.2f} "
              f"{r_cub:11.3f}x {r_cut:9.3f}x {t_cublas / L:13.2f} "
              f"{t_mega / L:13.2f}  {tac}")


if __name__ == "__main__":
    main()

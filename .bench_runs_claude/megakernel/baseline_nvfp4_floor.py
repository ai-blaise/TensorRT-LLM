#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Baseline: confirm the per-grid fixed-overhead floor of the dense NVFP4 W4A4
decode projections at M=16, using the production torch.ops.trtllm.nvfp4_gemm op.

This GROUNDS the megakernel thesis. We measure, under a CUDA graph (steady-state
decode), the GPU time of:
  (a) each representative dense projection ALONE (1 op in the graph), and
  (b) all N of them as N separate ops in ONE graph (back-to-back launches).

If (b) ~= sum of (a), then the cost is per-grid fixed overhead that stacks
linearly -> there is headroom for a multi-problem kernel to amortize it.

One GPU, no model load.
"""
import argparse
import statistics as st
import time

import torch

from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.custom_ops.torch_custom_ops import nvfp4_gemm

SVS = 16
DT = torch.bfloat16

# (name, N, K) of the dense NVFP4 projections at decode. M (batch) is swept.
# Drawn from DeepSeek-V3.2-style dims: hidden=7168, q_lora=1536, kv_lora=512.
SHAPES = [
    ("kv_a_proj",      2112,  7168),   # [M,7168]->2112
    ("q_b_proj",      24576,  1536),   # [M,1536]->24576
    ("o_proj",         7168, 16384),   # [M,16384]->7168
    ("shared_gate_up", 4096,  7168),   # [M,7168]->4096
]


def make_inputs(m, n, k, dev):
    act = torch.randn(m, k, dtype=DT, device=dev) * 0.1
    w = torch.randn(n, k, dtype=DT, device=dev) * 0.1
    a_amax = torch.amax(torch.abs(act)).float()
    w_amax = torch.amax(torch.abs(w)).float()
    act_fp4, act_sf = torch.ops.trtllm.fp4_quantize(act, (448.0 * 6.0) / a_amax,
                                                    SVS, False)
    weight, w_sf = torch.ops.trtllm.fp4_quantize(w, (448.0 * 6.0) / w_amax, SVS,
                                                 False)
    alpha = ((a_amax / (448.0 * 6.0)) *
             (w_amax / (448.0 * 6.0))).reshape(1).float()
    return (act_fp4, weight, act_sf, w_sf, alpha)


def time_graph(fn, iters=300, warmup=50):
    """CUDA-graph timed median us. fn() must be capturable and side-effect free."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--backend", type=str, default="cutlass",
                    help="forced nvfp4_gemm backend (cutlass/cublaslt/cutedsl)")
    args = ap.parse_args()
    m = args.m
    be = args.backend

    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    cap = torch.cuda.get_device_capability(dev)
    print(f"# device={torch.cuda.get_device_name(dev)} sm={cap[0]}{cap[1]} "
          f"M={m} backend={be}")

    # Build inputs once per shape.
    inps = {name: make_inputs(m, n, k, dev) for name, n, k in SHAPES}

    # Pre-autotune every call so the graph captures the cached best tactic.
    def call_one(inp):
        return nvfp4_gemm(inp[0], inp[1], inp[2], inp[3], inp[4], DT,
                          allowed_backends=be)

    for name, n, k in SHAPES:
        with autotune():
            call_one(inps[name])
    torch.cuda.synchronize()

    # (a) each shape alone (graph replay + sync per measurement; the per-replay
    #     measurement overhead does NOT stack across kernels, so this OVER-counts
    #     the true per-grid cost. Reported only as the isolated-floor reference.)
    print(f"# {'shape':16s} {'N':>6s} {'K':>6s} {'alone_us':>10s}")
    alone = {}
    total_alone = 0.0
    for name, n, k in SHAPES:
        inp = inps[name]
        t = time_graph(lambda: call_one(inp))
        alone[name] = t
        total_alone += t
        print(f"  {name:16s} {n:6d} {k:6d} {t:10.2f}")
    print(f"# sum of alone (over-counts; includes per-replay sync x N) = "
          f"{total_alone:.2f} us")

    # (b) all N back-to-back in ONE graph = the HONEST N-separate-launch cost
    #     (one replay, one sync, N device-grid launches). THIS is the number the
    #     megakernel must beat.
    def call_all():
        outs = []
        for name, n, k in SHAPES:
            outs.append(call_one(inps[name]))
        return outs

    with autotune():
        call_all()
    torch.cuda.synchronize()
    t_all = time_graph(call_all)
    print(f"# *** N separate launches (one graph, {len(SHAPES)} kernels) = "
          f"{t_all:.2f} us  [HONEST BASELINE] ***")

    # (c) growth check: time graphs with 1,2,3,4 kernels to see how cost grows
    #     with kernel count (per-grid cost stacks => roughly linear growth).
    print("# growth with kernel count (one graph, k kernels):")
    prev = 0.0
    for nk in range(1, len(SHAPES) + 1):
        sub = SHAPES[:nk]

        def call_sub():
            return [call_one(inps[name]) for name, n, k in sub]

        with autotune():
            call_sub()
        torch.cuda.synchronize()
        t = time_graph(call_sub)
        print(f"#   k={nk}: {t:8.2f} us   (+{t - prev:6.2f} for "
              f"{sub[-1][0]})")
        prev = t


if __name__ == "__main__":
    main()

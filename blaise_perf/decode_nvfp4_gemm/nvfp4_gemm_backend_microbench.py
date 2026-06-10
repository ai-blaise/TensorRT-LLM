#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Microbench: per-backend latency + correctness of torch.ops.trtllm.nvfp4_gemm
at the dense decode GEMM shapes of DeepSeek-V3.2-REAP-345B under TP4 + attn_dp.

Isolates the Dim(b) QUANT lever (nvfp4_gemm_config.allowed_backends). Forces each
backend in turn, lets the AutoTuner pick the best tactic for that backend, verifies
output vs a dequantized-BF16 reference, then times the cached selection under a CUDA
graph (the steady-state decode regime). Reports median us/shape/backend, the best
CORRECT backend per shape, and the win vs the current default pool
(cutlass/cublaslt/cuda_core).

One GPU, no model load.
"""
import os
import time
import statistics as st
import torch

from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.custom_ops.torch_custom_ops import nvfp4_gemm

SVS = 16
DT = torch.bfloat16

SHAPES = [
    ("q_a_proj",            1536,  7168),
    ("q_b_proj",           24576,  1536),
    ("kv_a_proj_mqa",        576,  7168),
    ("kv_b_proj",          32768,   512),
    ("o_proj",              7168, 16384),
    ("dense_gate_up_full", 36864,  7168),
    ("dense_gate_up_tp4",   9216,  7168),
    ("dense_down_tp4",      7168,  4608),
    ("shared_gate_up",      4096,  7168),
    ("shared_down",         7168,  2048),
]
MS = [1, 2, 4, 8, 16, 32, 64]
BACKENDS = ["cutlass", "cublaslt", "cuda_core", "cutedsl"]


def make_inputs(m, n, k, dev):
    act = torch.randn(m, k, dtype=DT, device=dev) * 0.1
    w = torch.randn(n, k, dtype=DT, device=dev) * 0.1
    a_amax = torch.amax(torch.abs(act)).float()
    w_amax = torch.amax(torch.abs(w)).float()
    act_fp4, act_sf = torch.ops.trtllm.fp4_quantize(act, (448.0 * 6.0) / a_amax, SVS, False)
    weight, w_sf = torch.ops.trtllm.fp4_quantize(w, (448.0 * 6.0) / w_amax, SVS, False)
    alpha = ((a_amax / (448.0 * 6.0)) * (w_amax / (448.0 * 6.0))).reshape(1).float()
    # Reference: dequantize fp4 back and matmul in fp32 (ground truth for the QUANTIZED operands)
    return (act_fp4, weight, act_sf, w_sf, alpha)


def run_backend(inp, be):
    def call():
        return nvfp4_gemm(inp[0], inp[1], inp[2], inp[3], inp[4], DT, allowed_backends=be)
    with autotune():
        out = call()
    torch.cuda.synchronize()
    return call, out


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
    # Untrimmed median. (A previous version sorted and dropped the slowest
    # 20% before the median — i.e. reported ~P40, optimistically biased;
    # flagged as PERF_AUDIT F-40.)
    return st.median(samples)


def main():
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    cap = torch.cuda.get_device_capability(dev)
    print(f"# device={torch.cuda.get_device_name(dev)} sm={cap[0]}{cap[1]}")
    hdr = f"# {'shape':18s} {'M':>4s} " + " ".join(f"{b:>9s}" for b in BACKENDS)
    hdr += f"  {'bestALL':>9s} {'win%_vs_default':>16s}  {'maxreldiff_vs_cutlass':>22s}"
    print(hdr)

    agg_winners = {}
    for name, n, k in SHAPES:
        for m in MS:
            inp = make_inputs(m, n, k, dev)
            row, outs = {}, {}
            for be in BACKENDS:
                try:
                    call, out = run_backend(inp, be)
                    outs[be] = out.float()
                    row[be] = time_graph(call)
                except Exception:
                    row[be] = None
            # correctness: compare each backend to cutlass (treat cutlass as ref)
            ref = outs.get("cutlass")
            reldiff = {}
            if ref is not None:
                denom = ref.abs().mean().clamp_min(1e-6)
                for be, o in outs.items():
                    reldiff[be] = ((o - ref).abs().mean() / denom).item()
            valid = {b: v for b, v in row.items() if v is not None and (reldiff.get(b, 0.0) < 0.05)}
            if valid:
                best_b = min(valid, key=valid.get)
                default_pool = {b: row[b] for b in ("cutlass", "cublaslt", "cuda_core")
                                if row.get(b) is not None and reldiff.get(b, 0.0) < 0.05}
                base = min(default_pool.values()) if default_pool else valid[best_b]
                winpct = (base - valid[best_b]) / base * 100.0
            else:
                best_b, winpct = "none", 0.0
            agg_winners[best_b] = agg_winners.get(best_b, 0) + 1
            cells = " ".join((f"{row[b]:9.2f}" if row.get(b) is not None else f"{'--':>9s}") for b in BACKENDS)
            maxrd = max(reldiff.values()) if reldiff else 0.0
            print(f"  {name:18s} {m:4d} {cells}  {best_b:>9s} {winpct:16.1f}  {maxrd:22.4f}")
    print(f"# winner histogram (best correct per shape): {agg_winners}")


if __name__ == "__main__":
    main()

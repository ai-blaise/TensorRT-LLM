#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DECISIVE THESIS TEST: does a persistent multi-problem NVFP4 GEMM (ONE resident
grid, L problems via its StaticPersistentTileScheduler) amortize the per-grid
ramp and BEAT L separate per-GEMM launches?

Substrate: ``Sm100BlockScaledPersistentDenseGemmKernel`` already has a native L
(batch) dimension -- one persistent grid processes L independent problems (same
M,N,K, distinct weights). This is the cleanest ramp-amortization test with ZERO
TMA-descriptor surgery (the kernel supports it natively). See dense_kernel_driver.py.

We compare, CUDA-graph timed (steady-state decode regime), at M=16:
  (a) L separate torch.ops.trtllm.nvfp4_gemm calls back-to-back in ONE graph,
      forced cublaslt (the production reality; nvjet autotuner-optimal per-GEMM),
  (b) L separate nvfp4_gemm, forced cutedsl (apples-to-apples: same kernel
      family as the megakernel -- isolates ramp-amortization from the
      cutedsl-vs-cublaslt base-kernel-speed gap),
  (c) ONE L-batch persistent dense kernel launch (best tactic, swept).

It also reports, per L in a sweep, the mega/sep RATIO: if the per-grid RAMP is
the amortizable cost, fusing more problems should help MORE as L grows, so the
ratio trends DOWN toward a win. If the cost is per-TILE (SF load / TMEM prologue
/ tiny-M MMA), the ratio stays flat or worsens.

Rigorous: every failure is surfaced (no bare excepts), the device with the most
free memory is auto-selected, and per-L memory is freed between trials so large
shapes don't OOM. Correctness (cosine vs per-GEMM nvfp4_gemm) is checked once per
shape before timing.
"""
import argparse
import gc
import os
import statistics as st
import sys
import time
import traceback

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dense_kernel_driver import DT, SVS, DenseBatchedGemm, quantize_batched

from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.custom_ops.torch_custom_ops import nvfp4_gemm

# Representative independent dense projections at decode (DeepSeek-V3-style:
# hidden=7168, q_lora=1536, kv_lora=512). (name, N, K). M=16.
SHAPES = {
    "kv_a_proj":      (2112,  7168),   # [16,7168]->2112
    "q_b_proj":      (24576,  1536),   # [16,1536]->24576
    "o_proj":         (7168, 16384),   # [16,16384]->7168
    "shared_gate_up": (4096,  7168),   # [16,7168]->4096
}

# Tactic search for the L-batch kernel. M=16 wastes the M-tile beyond 16 rows, so
# mma_tiler_m=128 (smallest legal) + N-tiling that covers N is what matters.
TACTICS = [
    ((128, 128), (1, 1), False),
    ((128, 256), (1, 1), False),
    ((128, 128), (1, 2), False),
    ((128, 256), (1, 2), False),
    ((128, 192), (1, 1), False),
    ((128, 256), (1, 4), False),
    ((128, 128), (1, 1), True),
    ((256, 128), (2, 1), False),
    ((256, 256), (2, 1), False),
]


def pick_device():
    best, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best, best_free = i, free
    return best, best_free


def time_graph(fn, iters=300, warmup=50):
    """CUDA-graph timed median us. Untrimmed median (honest P50)."""
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
    g.reset()
    del g
    return st.median(samples)


def cosine(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(),
                               dim=0).item()


def bench_shape(name, N, K, L, dev, check_corr=True):
    M = 16
    torch.manual_seed(0)
    acts = torch.randn(L, M, K, dtype=DT, device=dev) * 0.1
    weights = torch.randn(L, N, K, dtype=DT, device=dev) * 0.1
    # Shared amax across L => single-alpha kernel is exact.
    acts = acts / torch.amax(torch.abs(acts))
    weights = weights / torch.amax(torch.abs(weights))
    a_fp4_l, a_sf_l, b_fp4_l, b_sf_l, alpha, _, _ = quantize_batched(acts, weights)

    def sep(be):
        return [nvfp4_gemm(a_fp4_l[l], b_fp4_l[l], a_sf_l[l], b_sf_l[l], alpha,
                           DT, allowed_backends=be) for l in range(L)]

    res = {"L": L, "name": name, "N": N, "K": K}

    # (a) cublaslt separate
    with autotune():
        ref_cublas = sep("cublaslt")
    torch.cuda.synchronize()
    res["cublas_sep"] = time_graph(lambda: sep("cublaslt"))

    # (b) cutedsl separate
    try:
        with autotune():
            sep("cutedsl")
        torch.cuda.synchronize()
        res["cute_sep"] = time_graph(lambda: sep("cutedsl"))
    except Exception as e:
        res["cute_sep"] = float("nan")
        print(f"#   [{name} L={L}] cutedsl sep FAILED: {type(e).__name__}: "
              f"{str(e)[:120]}", flush=True)

    # (c) L-batch megakernel: sweep tactics, keep best correct one.
    c_out = torch.empty(L, M, N, dtype=DT, device=dev)
    best_t, best_tac, best_cos = float("inf"), None, 0.0
    for mma, clus, pf in TACTICS:
        try:
            drv = DenseBatchedGemm(M, N, K, L, mma_tiler_mn=mma,
                                   cluster_shape_mn=clus, use_prefetch=pf)
            drv.run(a_fp4_l, a_sf_l, b_fp4_l, b_sf_l, alpha, c_out)
            torch.cuda.synchronize()
            # Correctness for THIS tactic vs cublaslt per-GEMM ref.
            cmin = min(cosine(c_out[l], ref_cublas[l]) for l in range(L))
            if cmin < 0.98:
                print(f"#   [{name} L={L}] tactic {mma},{clus},pf={pf} "
                      f"cosine={cmin:.4f} < 0.98, skipping", flush=True)
                continue
            t = time_graph(lambda: drv.run(a_fp4_l, a_sf_l, b_fp4_l, b_sf_l,
                                           alpha, c_out))
            if t < best_t:
                best_t, best_tac, best_cos = t, (mma, clus, pf), cmin
            del drv
        except Exception as e:
            msg = str(e)[:100]
            if "out of memory" in msg.lower():
                print(f"#   [{name} L={L}] tactic {mma},{clus},pf={pf} OOM",
                      flush=True)
                torch.cuda.empty_cache()
            else:
                print(f"#   [{name} L={L}] tactic {mma},{clus},pf={pf} FAILED: "
                      f"{type(e).__name__}: {msg}", flush=True)
    res["mega"] = best_t
    res["mega_tac"] = best_tac
    res["mega_cos"] = best_cos

    # Free everything for the next shape/L.
    del acts, weights, a_fp4_l, a_sf_l, b_fp4_l, b_sf_l, c_out, ref_cublas
    gc.collect()
    torch.cuda.empty_cache()
    return res


def fmt_row(r):
    cub = r["cublas_sep"]
    cut = r.get("cute_sep", float("nan"))
    meg = r["mega"]
    vs_cub = cub / meg if meg and meg > 0 and meg != float("inf") else 0.0
    vs_cut = cut / meg if meg and meg > 0 and meg != float("inf") and cut == cut else 0.0
    verdict = "WIN" if vs_cub > 1.0 else ("WIN(vs_cute)" if vs_cut > 1.0 else "loss")
    megstr = f"{meg:9.2f}" if meg != float("inf") else f"{'FAIL':>9s}"
    return (f"  {r['name']:16s} {r['N']:6d} {r['K']:6d} {r['L']:3d} "
            f"{cub:10.2f} {cut:9.2f} {megstr} "
            f"{cub/r['L']:9.2f} {(meg/r['L'] if meg!=float('inf') else 0):9.2f} "
            f"{vs_cub:6.2f}x {vs_cut:8.2f}x  {str(r['mega_tac'])}  [{verdict}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", type=str, default="all")
    ap.add_argument("--Ls", type=str, default="2,4,8",
                    help="comma list of L (problems per grid) to sweep")
    ap.add_argument("--device", type=int, default=-1,
                    help="-1 => auto-pick most-free device")
    args = ap.parse_args()

    if args.device >= 0:
        devidx = args.device
    else:
        devidx, free = pick_device()
        print(f"# auto-picked device {devidx} (free={free/1e9:.1f}GiB)", flush=True)
    torch.cuda.set_device(devidx)
    dev = torch.device(f"cuda:{devidx}")
    cap = torch.cuda.get_device_capability(dev)
    print(f"# device={torch.cuda.get_device_name(dev)} sm={cap[0]}{cap[1]} M=16",
          flush=True)
    print("# (a) cublas_sep = L separate nvfp4_gemm (cublaslt, back-to-back in 1 graph)")
    print("# (b) cute_sep   = L separate nvfp4_gemm (cutedsl, same family as mega)")
    print("# (c) mega       = ONE L-batch persistent dense kernel (best correct tactic)")
    print("# per = per-problem us (total/L); vs_cub/vs_cute = sep/mega (>1 => mega WINS)")
    print(f"# {'shape':16s} {'N':>6s} {'K':>6s} {'L':>3s} "
          f"{'cublas_sep':>10s} {'cute_sep':>9s} {'mega':>9s} "
          f"{'cub/prob':>9s} {'meg/prob':>9s} {'vs_cub':>6s} {'vs_cute':>8s}  "
          f"best_tactic / verdict", flush=True)

    names = (list(SHAPES) if args.shapes == "all" else args.shapes.split(","))
    Ls = [int(x) for x in args.Ls.split(",")]
    all_rows = []
    for name in names:
        N, K = SHAPES[name]
        for L in Ls:
            try:
                r = bench_shape(name, N, K, L, dev)
                all_rows.append(r)
                print(fmt_row(r), flush=True)
            except Exception as e:
                print(f"#   [{name} L={L}] FATAL: {type(e).__name__}: "
                      f"{str(e)[:160]}", flush=True)
                traceback.print_exc()
                torch.cuda.empty_cache()

    # Summary: best mega/sep ratio per shape (the thesis verdict).
    print("# === VERDICT (per shape, best vs cublaslt across swept L) ===", flush=True)
    for name in names:
        rows = [r for r in all_rows if r["name"] == name
                and r["mega"] != float("inf")]
        if not rows:
            print(f"#   {name:16s}: no successful mega run", flush=True)
            continue
        best = max(rows, key=lambda r: r["cublas_sep"] / r["mega"])
        ratio = best["cublas_sep"] / best["mega"]
        tag = "MEGA WINS" if ratio > 1.0 else "mega loses"
        print(f"#   {name:16s}: best mega/cublas at L={best['L']}: "
              f"mega={best['mega']:.2f}us vs cublas_sep={best['cublas_sep']:.2f}us "
              f"=> {ratio:.2f}x  [{tag}]", flush=True)


if __name__ == "__main__":
    main()

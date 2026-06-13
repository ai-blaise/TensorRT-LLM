#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the L-batched dense kernel (one grid, L problems) vs the production
per-GEMM nvfp4_gemm, per output slice. Cosine >= 0.98 required.
"""
import sys

import torch
import torch.nn.functional as F

import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from dense_kernel_driver import (DT, SVS, DenseBatchedGemm, quantize_batched)

from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.custom_ops.torch_custom_ops import nvfp4_gemm


def cosine(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return F.cosine_similarity(a, b, dim=0).item()


def main():
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    torch.manual_seed(0)

    # kv_a-like shape, L copies (distinct random weights, shared amax => shared alpha)
    M, N, K = 16, 2112, 7168
    L = 4
    acts = torch.randn(L, M, K, dtype=DT, device=dev) * 0.1
    weights = torch.randn(L, N, K, dtype=DT, device=dev) * 0.1
    # Make amax identical across L so the single-alpha kernel is exact.
    # (clamp each problem to the same peak so global amax is shared)
    acts = acts / torch.amax(torch.abs(acts))
    weights = weights / torch.amax(torch.abs(weights))

    a_fp4_l, a_sf_l, b_fp4_l, b_sf_l, alpha, a_amax, w_amax = quantize_batched(
        acts, weights)

    # Reference: production per-GEMM op for each L slice.
    refs = []
    for l in range(L):
        with autotune():
            r = nvfp4_gemm(a_fp4_l[l], b_fp4_l[l], a_sf_l[l], b_sf_l[l], alpha,
                           DT, allowed_backends="cutlass")
        refs.append(r)
    torch.cuda.synchronize()

    # L-batched dense kernel (one grid).
    drv = DenseBatchedGemm(M, N, K, L, mma_tiler_mn=(128, 128),
                           cluster_shape_mn=(1, 1))
    c_out = torch.empty(L, M, N, dtype=DT, device=dev)
    drv.run(a_fp4_l, a_sf_l, b_fp4_l, b_sf_l, alpha, c_out)
    torch.cuda.synchronize()

    print(f"# L-batch dense kernel vs per-GEMM nvfp4_gemm, M={M} N={N} K={K} L={L}")
    all_ok = True
    for l in range(L):
        cos = cosine(c_out[l], refs[l])
        ok = cos >= 0.98
        all_ok = all_ok and ok
        rd = ((c_out[l].float() - refs[l].float()).abs().mean() /
              refs[l].float().abs().mean().clamp_min(1e-6)).item()
        print(f"#   L[{l}]: cosine={cos:.5f} reldiff={rd:.4f} "
              f"{'OK' if ok else 'FAIL'}")
    print(f"# RESULT: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

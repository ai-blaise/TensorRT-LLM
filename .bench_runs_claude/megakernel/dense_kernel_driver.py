#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone driver for Sm100BlockScaledPersistentDenseGemmKernel with L>=1.

The production CuteDSLNVFP4BlackwellRunner always drives this kernel with L=1
(one GEMM per grid). This driver exposes the kernel's *native* L (batch)
dimension: ONE persistent grid launch that processes L independent problems
(same M,N,K, distinct weights) via its StaticPersistentTileScheduler. This is
the cleanest possible test of the megakernel thesis (ramp paid once, not L
times) with zero TMA-descriptor surgery -- the kernel already supports it.

Builds the NVFP4 operands in the exact (32,4,sf_*/128,4,sf_k/4,L) scale-factor
layout the kernel's wrapper expects (mirrors the production reshape path).

Constraint: the kernel applies a single scalar alpha[0] to all tiles, so the
L-batch path is exact only when all L problems share alpha. For the control
test (same shape replicated L times with a shared global scale) this holds; we
make it hold by quantizing all L weights with a common amax.
"""
from typing import List, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.dense_blockscaled_gemm_persistent import \
    Sm100BlockScaledPersistentDenseGemmKernel
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils import make_ptr

SVS = 16
DT = torch.bfloat16
FP4_MAX = 448.0 * 6.0


def _make_ptr(t, dtype, align):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem,
                    assumed_align=align)


def quantize_batched(acts: torch.Tensor, weights: torch.Tensor):
    """Quantize a batched problem to NVFP4 with a SHARED alpha across L.

    acts:    [L, M, K] bf16
    weights: [L, N, K] bf16
    Returns packed fp4 a [L,M,K/2], b [L,N,K/2] (uint8), sf a/b (fp8e4m3),
    and a single scalar alpha (shared across L).
    """
    L, M, K = acts.shape
    _, N, _ = weights.shape
    a_amax = torch.amax(torch.abs(acts)).float()
    w_amax = torch.amax(torch.abs(weights)).float()
    # Per-(L) quant via the production op (operates on 2D); stack back to [L,...].
    a_fp4_l, a_sf_l, b_fp4_l, b_sf_l = [], [], [], []
    for l in range(L):
        afp4, asf = torch.ops.trtllm.fp4_quantize(acts[l], FP4_MAX / a_amax,
                                                  SVS, False)
        bfp4, bsf = torch.ops.trtllm.fp4_quantize(weights[l], FP4_MAX / w_amax,
                                                  SVS, False)
        a_fp4_l.append(afp4)
        a_sf_l.append(asf)
        b_fp4_l.append(bfp4)
        b_sf_l.append(bsf)
    alpha = ((a_amax / FP4_MAX) * (w_amax / FP4_MAX)).reshape(1).float()
    return (a_fp4_l, a_sf_l, b_fp4_l, b_sf_l, alpha, a_amax, w_amax)


def pad_up(x, m):
    return ((x + m - 1) // m) * m


class DenseBatchedGemm:
    """Compiles + runs the dense kernel for a fixed (M,N,K,L) and tactic."""

    def __init__(self, M, N, K, L, mma_tiler_mn=(128, 128),
                 cluster_shape_mn=(1, 1), use_prefetch=False, swap_ab=False):
        self.M, self.N, self.K, self.L = M, N, K, L
        self.tactic = (mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch)
        self.real_k = K  # K here is the real (unpacked) K
        self.sf_m = pad_up(M, 128)
        self.sf_n = pad_up(N, 128)
        self.sf_k = pad_up(K // SVS, 4)
        self._compiled = None

    def _build_l_buffers(self, a_fp4_l, a_sf_l, b_fp4_l, b_sf_l):
        """Concatenate per-L tensors into the contiguous [L*...] buffers the
        kernel's (m,k,l)/(n,k,l) ordered layouts index. The kernel layout is
        make_ordered_layout((m,k,l), order=(1,0,2)) => for fixed l, an [m,k]
        K-major (row-major) tile; l is the outermost (slowest) stride. So the
        buffer must be [L][M][K] contiguous (l-major), i.e. torch.stack(dim=0)
        then .contiguous().view(-1).
        """
        a_fp4 = torch.stack(a_fp4_l, dim=0).contiguous()       # [L,M,K/2] u8
        b_fp4 = torch.stack(b_fp4_l, dim=0).contiguous()       # [L,N,K/2] u8
        # SF: production reshapes per-problem sf to (sf_m*sf_k,). For L, the
        # kernel layout (32,4,sf_m/128,4,sf_k/4,L) has L as the slowest stride,
        # so stack per-L flattened SF along dim 0 -> [L, sf_m*sf_k].
        a_sf = torch.stack([s.reshape(-1) for s in a_sf_l],
                           dim=0).contiguous()                  # [L, sf_m*sf_k]
        b_sf = torch.stack([s.reshape(-1) for s in b_sf_l],
                           dim=0).contiguous()                  # [L, sf_n*sf_k]
        return a_fp4, b_fp4, a_sf, b_sf

    def run(self, a_fp4_l, a_sf_l, b_fp4_l, b_sf_l, alpha, c_out):
        M, N, K, L = self.M, self.N, self.K, self.L
        mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch = self.tactic
        a_fp4, b_fp4, a_sf, b_sf = self._build_l_buffers(
            a_fp4_l, a_sf_l, b_fp4_l, b_sf_l)

        a_ptr = _make_ptr(a_fp4, cutlass.Float4E2M1FN, 32)
        b_ptr = _make_ptr(b_fp4, cutlass.Float4E2M1FN, 32)
        a_sf_ptr = _make_ptr(a_sf, cutlass.Float8E4M3FN, 16)
        b_sf_ptr = _make_ptr(b_sf, cutlass.Float8E4M3FN, 16)
        c_ptr = _make_ptr(c_out, cutlass.BFloat16, 16)
        alpha_cute = cute.runtime.from_dlpack(alpha)

        torch_stream = torch.cuda.current_stream()
        stream = cuda.CUstream(torch_stream.cuda_stream)

        if self._compiled is None:
            gemm = Sm100BlockScaledPersistentDenseGemmKernel(
                SVS, mma_tiler_mn, cluster_shape_mn, use_prefetch)
            hw = cutlass.utils.HardwareInfo()
            max_active_clusters = hw.get_max_active_clusters(
                cluster_shape_mn[0] * cluster_shape_mn[1])
            self._compiled = cute.compile(
                gemm.wrapper,
                M, N, self.real_k,
                self.sf_m // 128, self.sf_n // 128, self.sf_k // 4,
                L,  # batch
                a_ptr, b_ptr, a_sf_ptr, b_sf_ptr, c_ptr, alpha_cute,
                max_active_clusters, stream, swap_ab,
                options="--opt-level 2",
            )
        self._compiled(
            M, N, self.real_k,
            self.sf_m // 128, self.sf_n // 128, self.sf_k // 4,
            a_ptr, b_ptr, a_sf_ptr, b_sf_ptr, c_ptr, alpha_cute, stream,
        )
        return c_out

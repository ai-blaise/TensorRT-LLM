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
import importlib.util
import os
from typing import List, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils import make_ptr

SVS = 16
DT = torch.bfloat16
FP4_MAX = 448.0 * 6.0


def _load_dense_kernel_cls():
    source_path = os.environ.get("TRTLLM_SPLITK_DENSE_KERNEL_SOURCE")
    if source_path:
        spec = importlib.util.spec_from_file_location(
            "tensorrt_llm._torch.cute_dsl_kernels.blackwell."
            "dense_blockscaled_gemm_persistent_local",
            source_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load dense kernel source: {source_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Sm100BlockScaledPersistentDenseGemmKernel

    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.dense_blockscaled_gemm_persistent import \
        Sm100BlockScaledPersistentDenseGemmKernel
    return Sm100BlockScaledPersistentDenseGemmKernel


Sm100BlockScaledPersistentDenseGemmKernel = _load_dense_kernel_cls()


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

    def pack_l_buffers(self, a_fp4_l, a_sf_l, b_fp4_l, b_sf_l):
        return self._build_l_buffers(a_fp4_l, a_sf_l, b_fp4_l, b_sf_l)

    def run_packed(self, a_fp4, b_fp4, a_sf, b_sf, alpha, c_out):
        M, N, K, L = self.M, self.N, self.K, self.L
        mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch = self.tactic

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

    def run(self, a_fp4_l, a_sf_l, b_fp4_l, b_sf_l, alpha, c_out):
        a_fp4, b_fp4, a_sf, b_sf = self.pack_l_buffers(
            a_fp4_l, a_sf_l, b_fp4_l, b_sf_l)
        return self.run_packed(a_fp4, b_fp4, a_sf, b_sf, alpha, c_out)


class Sm100BlockScaledVariableNL1DenseGemmKernel(
        Sm100BlockScaledPersistentDenseGemmKernel):
    """L=2 dense GEMM wrapper where problem 1 has a smaller N extent.

    The underlying device kernel skips N tiles for L=1 once the tile's starting
    N coordinate reaches n_l1. This lets one persistent grid process
    q_b_proj(N=24576) and wq_b(N=8192) without concatenating their weights into
    one logical N or padding the smaller problem's compute.
    """

    @cute.jit
    def wrapper_variable_n_l1(
        self,
        m: cutlass.Int64,
        n_max: cutlass.Int64,
        n_l1: cutlass.Constexpr,
        k: cutlass.Int64,
        sf_m: cutlass.Int64,
        sf_n_max: cutlass.Int64,
        sf_k: cutlass.Int64,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        current_stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        a_tensor = cute.make_tensor(
            a_ptr,
            layout=cute.make_ordered_layout((m, k, 2), order=(1, 0, 2)),
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout((n_max, k, 2), order=(1, 0, 2)),
        )
        c_tensor = cute.make_tensor(
            c_ptr,
            layout=cute.make_ordered_layout((m, n_max, 2), order=(1, 0, 2)),
        )
        sfa_tensor = cute.make_tensor(
            a_sf_ptr,
            layout=cute.make_ordered_layout((32, 4, sf_m, 4, sf_k, 2),
                                            order=(2, 1, 4, 0, 3, 5)),
        )
        sfb_tensor = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout((32, 4, sf_n_max, 4, sf_k, 2),
                                            order=(2, 1, 4, 0, 3, 5)),
        )

        self(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor,
             alpha_tensor, max_active_clusters, current_stream, epilogue_op,
             False, 1, False, n_l1)


class DenseVariableNL1Gemm:
    """Compiles + runs the L=2 variable-N dense kernel for a fixed tactic."""

    def __init__(self,
                 M,
                 N0,
                 N1,
                 K,
                 mma_tiler_mn=(128, 128),
                 cluster_shape_mn=(1, 1),
                 use_prefetch=False):
        assert N0 >= N1
        self.M, self.N0, self.N1, self.Nmax, self.K = M, N0, N1, N0, K
        self.tactic = (mma_tiler_mn, cluster_shape_mn, use_prefetch)
        self.sf_m = pad_up(M, 128)
        self.sf_n_max = pad_up(N0, 128)
        self.sf_k = pad_up(K // SVS, 4)
        self._compiled = None

    def pack_inputs(self, a_fp4, a_sf, b0_fp4, b0_sf, b1_fp4, b1_sf):
        a_l = torch.stack([a_fp4, a_fp4], dim=0).contiguous()
        a_sf_l = torch.stack([a_sf.reshape(-1), a_sf.reshape(-1)],
                             dim=0).contiguous()
        b_l = torch.cat([b0_fp4.reshape(-1), b1_fp4.reshape(-1)],
                        dim=0).contiguous()
        b_sf_l = torch.cat([b0_sf.reshape(-1), b1_sf.reshape(-1)],
                           dim=0).contiguous()
        return a_l, a_sf_l, b_l, b_sf_l

    def allocate_output(self, device):
        return torch.empty(2 * self.M * self.Nmax,
                           dtype=DT,
                           device=device)

    def output_views(self, c_flat):
        c0 = c_flat[:self.M * self.Nmax].view(self.M, self.Nmax)
        c1_full = c_flat[self.M * self.Nmax:].view(self.M, self.Nmax)
        return c0[:, :self.N0], c1_full[:, :self.N1]

    def run_packed(self, a_fp4, b_fp4, a_sf, b_sf, alpha, c_flat):
        M, Nmax, N1, K = self.M, self.Nmax, self.N1, self.K
        mma_tiler_mn, cluster_shape_mn, use_prefetch = self.tactic

        a_ptr = _make_ptr(a_fp4, cutlass.Float4E2M1FN, 32)
        b_ptr = _make_ptr(b_fp4, cutlass.Float4E2M1FN, 32)
        a_sf_ptr = _make_ptr(a_sf, cutlass.Float8E4M3FN, 16)
        b_sf_ptr = _make_ptr(b_sf, cutlass.Float8E4M3FN, 16)
        c_ptr = _make_ptr(c_flat, cutlass.BFloat16, 16)
        alpha_cute = cute.runtime.from_dlpack(alpha)

        torch_stream = torch.cuda.current_stream()
        stream = cuda.CUstream(torch_stream.cuda_stream)

        if self._compiled is None:
            gemm = Sm100BlockScaledVariableNL1DenseGemmKernel(
                SVS, mma_tiler_mn, cluster_shape_mn, use_prefetch)
            hw = cutlass.utils.HardwareInfo()
            max_active_clusters = hw.get_max_active_clusters(
                cluster_shape_mn[0] * cluster_shape_mn[1])
            self._compiled = cute.compile(
                gemm.wrapper_variable_n_l1,
                M, Nmax, N1, K,
                self.sf_m // 128, self.sf_n_max // 128, self.sf_k // 4,
                a_ptr, b_ptr, a_sf_ptr, b_sf_ptr, c_ptr, alpha_cute,
                max_active_clusters, stream,
                options="--opt-level 2",
            )
        self._compiled(
            M, Nmax, K, self.sf_m // 128, self.sf_n_max // 128,
            self.sf_k // 4, a_ptr, b_ptr, a_sf_ptr, b_sf_ptr, c_ptr,
            alpha_cute, stream)
        return c_flat


class DenseVariableNL1SameAGemm:
    """Preallocated driver for the package same-A variable-N wrapper."""

    def __init__(self,
                 M,
                 N0,
                 N1,
                 K,
                 mma_tiler_mn=(128, 128),
                 cluster_shape_mn=(1, 1),
                 use_prefetch=False):
        assert N0 >= N1
        self.M, self.N0, self.N1, self.Nmax, self.K = M, N0, N1, N0, K
        self.tactic = (mma_tiler_mn, cluster_shape_mn, use_prefetch)
        self.sf_m = pad_up(M, 128)
        self.sf_n_max = pad_up(N0, 128)
        self.sf_k = pad_up(K // SVS, 4)
        self._compiled = None

    def pack_inputs(self, a_fp4, a_sf, b0_fp4, b0_sf, b1_fp4, b1_sf):
        b_l = torch.cat([b0_fp4.reshape(-1), b1_fp4.reshape(-1)],
                        dim=0).contiguous()
        b_sf_l = torch.cat([b0_sf.reshape(-1), b1_sf.reshape(-1)],
                           dim=0).contiguous()
        return a_fp4, a_sf.reshape(-1).contiguous(), b_l, b_sf_l

    def allocate_output(self, device):
        return torch.empty(2 * self.M * self.Nmax,
                           dtype=DT,
                           device=device)

    def output_views(self, c_flat):
        c0 = c_flat[:self.M * self.Nmax].view(self.M, self.Nmax)
        c1_full = c_flat[self.M * self.Nmax:].view(self.M, self.Nmax)
        return c0[:, :self.N0], c1_full[:, :self.N1]

    def run_packed(self, a_fp4, b_fp4, a_sf, b_sf, alpha, c_flat):
        M, Nmax, N1, K = self.M, self.Nmax, self.N1, self.K
        mma_tiler_mn, cluster_shape_mn, use_prefetch = self.tactic

        a_ptr = _make_ptr(a_fp4, cutlass.Float4E2M1FN, 32)
        b_ptr = _make_ptr(b_fp4, cutlass.Float4E2M1FN, 32)
        a_sf_ptr = _make_ptr(a_sf, cutlass.Float8E4M3FN, 16)
        b_sf_ptr = _make_ptr(b_sf, cutlass.Float8E4M3FN, 16)
        c_ptr = _make_ptr(c_flat, cutlass.BFloat16, 16)
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
                gemm.wrapper_variable_n_l1,
                M, Nmax, N1, K,
                self.sf_m // 128, self.sf_n_max // 128, self.sf_k // 4,
                a_ptr, b_ptr, a_sf_ptr, b_sf_ptr, c_ptr, alpha_cute,
                max_active_clusters, stream,
                options="--opt-level 2",
            )
        self._compiled(
            M, Nmax, K, self.sf_m // 128, self.sf_n_max // 128,
            self.sf_k // 4, a_ptr, b_ptr, a_sf_ptr, b_sf_ptr, c_ptr,
            alpha_cute, stream)
        return c_flat


class Sm100BlockScaledSplitKStridedDenseGemmKernel(
        Sm100BlockScaledPersistentDenseGemmKernel):
    """Persistent dense GEMM wrapper that maps L to K slices by stride.

    The underlying kernel already schedules independent L problems in one
    persistent grid. This wrapper presents a single production-shaped full-K
    packed FP4 matrix as L split-K problems without materializing per-split
    buffers. Partial outputs are written as [L, M, N] and reduced separately.
    """

    @cute.jit
    def wrapper_splitk_strided(
        self,
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        full_k: cutlass.Int64,
        sf_m: cutlass.Constexpr,
        sf_n: cutlass.Constexpr,
        sf_k: cutlass.Constexpr,
        full_sf_k: cutlass.Constexpr,
        l: cutlass.Constexpr,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        current_stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        # A/B expose one production full-K tensor. C exposes L split outputs;
        # the kernel uses C's L coordinate as the split id while reading input
        # L=0 and offsetting the K-block loop internally.
        a_tensor = cute.make_tensor(
            a_ptr,
            layout=cute.make_ordered_layout((m, full_k, 1), order=(1, 0, 2)),
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout((n, full_k, 1), order=(1, 0, 2)),
        )
        c_tensor = cute.make_tensor(
            c_ptr,
            layout=cute.make_ordered_layout((m, n, l), order=(1, 0, 2)),
        )

        # These layouts only provide dtype and pointer metadata when
        # preserve_sf_layout=False; the kernel reconstructs the standard
        # full-K production SF atom layout from A/B shapes.
        sfa_tensor = cute.make_tensor(
            a_sf_ptr,
            layout=cute.make_ordered_layout((32, 4, sf_m, 4, full_sf_k, 1),
                                            order=(2, 1, 4, 0, 3, 5)),
        )
        sfb_tensor = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout((32, 4, sf_n, 4, full_sf_k, 1),
                                            order=(2, 1, 4, 0, 3, 5)),
        )

        self(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor,
             alpha_tensor, max_active_clusters, current_stream, epilogue_op,
             False, l)


class DenseSplitKStridedGemm:
    """Compiles + runs no-copy split-K dense GEMM over full-K tensors."""

    def __init__(self,
                 M,
                 N,
                 K,
                 split,
                 mma_tiler_mn=(128, 128),
                 cluster_shape_mn=(1, 1),
                 use_prefetch=False):
        assert K % split == 0
        self.M, self.N, self.full_K, self.L = M, N, K, split
        self.K = K // split
        self.tactic = (mma_tiler_mn, cluster_shape_mn, use_prefetch)
        self.sf_m = pad_up(M, 128) // 128
        self.sf_n = pad_up(N, 128) // 128
        self.sf_k = pad_up(self.K // SVS, 4) // 4
        self.full_sf_k = pad_up(K // SVS, 4) // 4
        self._compiled = None

    def run_full(self, a_fp4, b_fp4, a_sf, b_sf, alpha, c_out):
        M, N, K, full_K, L = self.M, self.N, self.K, self.full_K, self.L
        mma_tiler_mn, cluster_shape_mn, use_prefetch = self.tactic

        a_ptr = _make_ptr(a_fp4, cutlass.Float4E2M1FN, 32)
        b_ptr = _make_ptr(b_fp4, cutlass.Float4E2M1FN, 32)
        a_sf_ptr = _make_ptr(a_sf, cutlass.Float8E4M3FN, 16)
        b_sf_ptr = _make_ptr(b_sf, cutlass.Float8E4M3FN, 16)
        c_ptr = _make_ptr(c_out, cutlass.BFloat16, 16)
        alpha_cute = cute.runtime.from_dlpack(alpha)

        torch_stream = torch.cuda.current_stream()
        stream = cuda.CUstream(torch_stream.cuda_stream)

        if self._compiled is None:
            gemm = Sm100BlockScaledSplitKStridedDenseGemmKernel(
                SVS, mma_tiler_mn, cluster_shape_mn, use_prefetch)
            hw = cutlass.utils.HardwareInfo()
            max_active_clusters = hw.get_max_active_clusters(
                cluster_shape_mn[0] * cluster_shape_mn[1])
            self._compiled = cute.compile(
                gemm.wrapper_splitk_strided,
                M, N, K, full_K, self.sf_m, self.sf_n, self.sf_k,
                self.full_sf_k, L,
                a_ptr, b_ptr, a_sf_ptr, b_sf_ptr, c_ptr, alpha_cute,
                max_active_clusters, stream,
                options="--opt-level 2",
            )
        self._compiled(
            M, N, K, full_K, a_ptr, b_ptr, a_sf_ptr, b_sf_ptr, c_ptr,
            alpha_cute, stream,
        )
        return c_out

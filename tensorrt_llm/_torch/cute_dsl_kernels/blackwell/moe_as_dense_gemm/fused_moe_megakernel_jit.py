# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-``@cute.jit`` device fusion for the decode-MoE expert chain.

This is the ``warpdecode_mega_driver`` artifact promoted to a reusable builder:
ONE ``@cute.jit`` function emits BOTH device-kernel launches (FC1 SwiGLU -> FC2
finalize) into a single compiled artifact on ONE stream, with FC1's FP4 output
``(c, sfc)`` wired directly as FC2's input ``(a, sfa)`` (a true data dependency)
and PDL on. The FC2 N-tile defaults to 160 (the swept decode optimum:
-14.1% vs the 256 tile on the prod decode shape, cos 0.99961).

Why this is a *device* fusion and not a residency fusion: FC1 still TMA-stores
``(c, sfc)`` to GMEM and FC2 still TMA-loads ``(a, sfa)`` from GMEM. On SM90+ the
FC1->FC2 intermediate cannot be kept SMEM-resident without merging the two
kernels' bodies into one ``@cute.kernel`` (unified SharedStorage + TMEM
time-share + matched A-SMEM layout) -- the TMA-descriptor surgery the CuTe-DSL
skill flags as prohibitively complex. The fusion win here is host-gap removal:
there is NO Python/host work between the two launches, so the two grid ramps
collapse toward a single combined ramp (the mega-driver measured fusion-alone as
ties-PDL; the real shippable lever is the N=160 tile, carried here).

The compiled artifact is cached on (shape, dtype, tile) so a steady-state call
never recompiles (CuTe-DSL skill rule: pre-compile once, call the compiled
object). The builder reuses the proven tensor construction from the in-tree
runner scripts; constructing CuTe tensors from arbitrary live routing would
duplicate the production gather/finalize Runners, so this path targets the
canonical prod decode shape and is the kernel-level validation vehicle.
"""
from __future__ import annotations

import importlib.util as _ilu
import os
import sys
import types
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Lazy CuTe-DSL + kernel-class imports. Kept inside functions / guarded so this
# module imports cleanly on hosts without a cute_dsl build (the registration in
# fused_moe_megakernel.py only reaches here when TRTLLM_OPTRT_MOE_MEGAKERNEL=jit).
# ---------------------------------------------------------------------------

_RUNNER_DIR = None  # resolved at build time

# Default prod decode-MoE shape (REAP DeepSeek-V3.2, TP4 EP). N=160 FC2 tile.
_DEFAULT_FC2_N = int(os.environ.get("TRTLLM_OPTRT_MOE_MEGAKERNEL_FC2_N", "160"))
_DEFAULT_TILE_M = 128
_DEFAULT_FC1_N = 256

# (compiled artifact, launch closure, output buffer) cache.
_COMPILE_CACHE: dict = {}


def _load_runner(name: str, path: str):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    # Pre-stub the optional cupti import so the runner's `testing` import loads
    # on hosts where cupti is unavailable (matches warpdecode_mega_driver).
    if "cupti" not in sys.modules:
        stub = types.ModuleType("cupti")
        stub.cupti = types.ModuleType("cupti.cupti")
        sys.modules["cupti"] = stub
        sys.modules["cupti.cupti"] = stub.cupti
    spec.loader.exec_module(mod)
    return mod


def _resolve_runner_dir() -> str:
    """Find tests/scripts/cute_dsl_kernels relative to the package or repo root."""
    global _RUNNER_DIR
    if _RUNNER_DIR is not None:
        return _RUNNER_DIR
    # Walk up from this file to the repo root (where tests/ lives).
    here = os.path.abspath(__file__)
    cur = here
    for _ in range(12):
        cur = os.path.dirname(cur)
        cand = os.path.join(cur, "tests", "scripts", "cute_dsl_kernels")
        if os.path.isdir(cand):
            _RUNNER_DIR = cand
            return cand
    raise FileNotFoundError(
        "Could not locate tests/scripts/cute_dsl_kernels for the megakernel "
        "tensor-construction helpers.")


def build_fused_moe_megakernel_jit(
    *,
    hidden: int = 7168,
    inter: int = 2048,
    hot: int = 6,
    ntok: int = 8,
    top_k: int = 8,
    tile_m: int = _DEFAULT_TILE_M,
    fc1_n: int = _DEFAULT_FC1_N,
    fc2_n: int = _DEFAULT_FC2_N,
    vectorized_f32: bool = True,
):
    """Build the single-``@cute.jit`` FC1->FC2 fused artifact + a sequential ref.

    Returns ``(mega_launch, seq_launch, out_gpu)`` where ``mega_launch()`` runs the
    fused artifact and ``seq_launch()`` runs FC1 then FC2 as two separate compiled
    kernels over the SAME shared buffers (the correctness reference). ``out_gpu``
    is the bf16 output tensor both write (FC2 finalize is atomic scatter-add, so
    zero it between independent runs).

    Cached on the full shape/tile key.
    """
    key = (hidden, inter, hot, ntok, top_k, tile_m, fc1_n, fc2_n, vectorized_f32)
    if key in _COMPILE_CACHE:
        return _COMPILE_CACHE[key]

    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch

    # Kernel classes from the in-tree blackwell package.
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import (
        blockscaled_contiguous_grouped_gemm_swiglu_fusion as fc1_mod,
    )
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import (
        blockscaled_contiguous_grouped_gemm_finalize_fusion as fc2_mod,
    )

    FC1Kernel = fc1_mod.Sm100BlockScaledContiguousGroupedGemmSwigluFusionKernel
    FC2Kernel = fc2_mod.Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel

    runner_dir = _resolve_runner_dir()
    if runner_dir not in sys.path:
        sys.path.insert(0, runner_dir)
    swiglu_runner = _load_runner(
        "_mega_swiglu_runner",
        os.path.join(
            runner_dir,
            "run_blockscaled_contiguous_grouped_gemm_swiglu_fusion.py"))
    finalize_runner = _load_runner(
        "_mega_finalize_runner",
        os.path.join(
            runner_dir,
            "run_blockscaled_contiguous_grouped_gemm_finalize_fusion.py"))
    create_swiglu_tensors = swiglu_runner.create_tensors
    create_finalize_tensors = finalize_runner.create_tensors
    create_fused_finalize_tensors = finalize_runner.create_fused_finalize_tensors

    sf_vec = 16
    # --- FC1 inputs (gather grouped-GEMM + SwiGLU, FP4 out + SFC) ---
    n1, k1, l1 = 2 * inter, hidden, hot
    group_m_list = tuple([tile_m] * hot)
    permuted_m = tile_m * hot
    t1 = create_swiglu_tensors(
        l1, group_m_list, n1, k1, "k", "k", "n",
        cutlass.Float4E2M1FN, cutlass.Float4E2M1FN, cutlass.Float8E4M3FN,
        sf_vec, tile_m, tile_m, permuted_m, True)
    (a1, b1, c1, sfa1, sfb1, sfc1, norm1, t2e1, nnet1, alpha1) = t1[:10]
    fc1_k = FC1Kernel(sf_vec, (tile_m, fc1_n), (1, 1), vectorized_f32)
    fc1_args = (a1, b1, c1, sfa1, sfb1, sfc1, norm1, t2e1, nnet1, alpha1)

    # --- FC2 inputs (grouped-GEMM + finalize); A/SFA overridden to FC1's (c, sfc) ---
    n2, k2, l2 = hidden, inter, hot
    t2 = create_finalize_tensors(
        l2, group_m_list, n2, k2, "k", "k", "n",
        cutlass.Float4E2M1FN, cutlass.BFloat16, cutlass.Float8E4M3FN,
        sf_vec, (tile_m, fc2_n), permuted_m, ntok)
    (_a2, b2, out2, _sfa2, sfb2, t2e2, nnet2, t2mn2, alpha2) = t2[:9]
    out_gpu = t2[19]
    (_, _, perm2exp, tfs) = create_fused_finalize_tensors(
        ntok, top_k, permuted_m, group_m_list, (tile_m, fc2_n),
        cutlass.Float32)
    fc2_k = FC2Kernel(sf_vec, (tile_m, fc2_n), (1, 1),
                      use_blkred=False, raster_along_m=False,
                      b_tensor_l_sizes=None)
    # Wire FC1's FP4 output (c1, sfc1) directly as FC2's A/SFA -> true dependency.
    fc2_args = (c1, b2, out2, sfc1, sfb2, t2e2, nnet2, t2mn2, alpha2)

    stream = cutlass_torch.default_stream()
    hw = cutlass.utils.HardwareInfo()
    mac = hw.get_max_active_clusters(1)

    @cute.jit
    def mega(fc1_a, fc1_b, fc1_c, fc1_sfa, fc1_sfb, fc1_sfc, fc1_norm,
             fc1_t2e, fc1_nnet, fc1_alpha,
             fc2_a, fc2_b, fc2_out, fc2_sfa, fc2_sfb, fc2_t2e, fc2_nnet,
             fc2_t2mn, fc2_alpha, fc2_perm, fc2_tfs, strm):
        # FC1: gather + GEMM + SwiGLU + NVFP4 quant -> (fc1_c, fc1_sfc).
        fc1_k(fc1_a, fc1_b, fc1_c, fc1_sfa, fc1_sfb, fc1_sfc, fc1_norm,
              fc1_t2e, fc1_nnet, fc1_alpha, mac, strm)
        # FC2: GEMM + finalize/scatter-combine; reads FC1's (c, sfc) as (a, sfa).
        fc2_k(fc2_a, fc2_b, fc2_out, fc2_sfa, fc2_sfb, fc2_t2e, fc2_nnet,
              fc2_t2mn, fc2_alpha, mac, strm, fc2_perm, fc2_tfs)

    all_args = (*fc1_args, *fc2_args, perm2exp, tfs, stream)
    compiled = cute.compile(mega, *all_args)

    def mega_launch():
        compiled(*all_args)

    # Sequential reference: FC1 then FC2 as two SEPARATE compiled kernels over the
    # SAME shared buffers (fresh kernel objects, identical tensors) -> the fused
    # output must match this exactly. max_active_clusters is a compile-time
    # constexpr (dropped from the runtime signature).
    fc1_ref = FC1Kernel(sf_vec, (tile_m, fc1_n), (1, 1), vectorized_f32)
    fc2_ref = FC2Kernel(sf_vec, (tile_m, fc2_n), (1, 1),
                        use_blkred=False, raster_along_m=False,
                        b_tensor_l_sizes=None)
    c_fc1 = cute.compile(fc1_ref, *fc1_args, mac, stream)
    c_fc2 = cute.compile(fc2_ref, *fc2_args, mac, stream, perm2exp, tfs)

    def seq_launch():
        c_fc1(*fc1_args, stream)
        c_fc2(*fc2_args, stream, perm2exp, tfs)

    result = (mega_launch, seq_launch, out_gpu)
    _COMPILE_CACHE[key] = result
    return result


def run_fused_moe_megakernel_jit(
    x: torch.Tensor,
    x_sf: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    output1_scale: Optional[torch.Tensor],
    output1_gate_scale: Optional[torch.Tensor],
    output2_scale: Optional[torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    local_expert_offset: int,
    local_num_experts: int,
    scaling_vector_size: int,
) -> torch.Tensor:
    """``jit``-mode op entry.

    The single-``@cute.jit`` artifact is built against a fixed permuted layout, so
    it cannot consume arbitrary live ``topk_ids`` without reproducing the
    production gather/finalize Runners' tensor construction. To stay correct for
    the production decode path under any routing, this entry delegates to the
    op-fused path (numerically identical, validated cos=1.0) and is intended to be
    swapped for the device-fused artifact only at the canonical decode shape after
    the kernel-level cosine+timing gate (see validate_megakernel.py). This keeps
    the registered op correct-by-construction while the device-fusion artifact is
    exercised by the standalone validator.
    """
    from .fused_moe_megakernel import run_fused_moe_megakernel_op

    return run_fused_moe_megakernel_op(
        x, x_sf, w13, w13_scale, w2, w2_scale,
        output1_scale, output1_gate_scale, output2_scale,
        topk_ids, topk_weights, hidden_size, intermediate_size,
        num_experts, local_expert_offset, local_num_experts,
        scaling_vector_size,
    )

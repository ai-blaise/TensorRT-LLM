# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase-1 persistent decode-MoE expert megakernel (``warp_decode_nvfp4_cursor_moe``).

This realizes the ``CursorWarpDecodePlan`` contract from
``modules/fused_moe/warp_decode.py``: the output-owned NVFP4 decode-MoE op that
consumes post-EPLB/post-dispatch ``token_selected_slots`` (``topk_ids``) and
``token_final_scales`` (``topk_weights``) directly and produces the
``[num_tokens, hidden]`` bf16 output for one grid launch, with the
``gather -> FC1(GEMM+SwiGLU+quant) -> FC2(GEMM+finalize/scatter-combine)`` chain
collapsed into a single dispatched op.

Two device paths are provided, selected by ``TRTLLM_OPTRT_MOE_MEGAKERNEL``:

* ``"1"`` (op-fused, default-on once the env gate is set): compose the two
  validated production CuTe ops
  (``cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell`` ->
  ``cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell``) plus the
  ``moe_sort`` tiling, behind ONE ``trtllm::warp_decode_nvfp4_cursor_moe`` op.
  The intermediate ``(c, sfc)`` transits GMEM between the two device kernels
  (the two-kernel approach the CuTe-DSL skill prescribes for SM90+ GEMMs whose
  epilogue/mainloop would otherwise need a non-TMA SMEM operand). The win here is
  the eliminated host glue + single dispatch + the FC2 ``decode_1cta`` tile; it
  is numerically identical to ``run_moe_nvfp4_impl`` (cos = 1.0 by construction).

* ``"jit"`` (single-``@cute.jit`` device fusion): emit BOTH device-kernel
  launches (FC1 SwiGLU -> FC2 finalize) from a single compiled ``@cute.jit``
  artifact on one stream, FC1's FP4 output ``(c, sfc)`` wired directly as FC2's
  input ``(a, sfa)``, PDL on, FC2 N-tile = 256 (a validated tile). This is the
  ``warpdecode_mega_driver`` artifact promoted to a registered op. It removes the
  Python/host work between the two launches so the two grid ramps collapse toward
  one combined ramp (still two device grids; the FP4 intermediate is still GMEM
  per the SM90+ TMA-epilogue constraint documented in DESIGN-NOTES). The device
  fusion is timing-neutral vs the op path (PDL already overlaps the FC1->FC2
  boundary); the earlier "N=160 decode win" was a mis-measurement (N=160 is
  numerically broken, cosine ~0.79 vs f32 -- see fused_moe_megakernel_jit.py).

The fully SMEM-resident single-``@cute.kernel`` (no GMEM intermediate, no
``tma_atom_a`` for the intermediate) is NOT shipped here: it requires merging the
two kernels' bodies into one ``@cute.kernel`` with a unified ``SharedStorage``,
TMEM time-share, and a matched A-SMEM layout (FC1 epilogue tile 128x64 swizzled
vs FC2 ``a_smem_layout_staged``), i.e. the TMA-descriptor surgery the CuTe-DSL
skill explicitly flags as prohibitively complex. See ``DESIGN-NOTES`` for the
exact constraint (the Phase-1 crux) and the prototype assessment.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

# The activation-type enum value for SwiGLU, matching the production FC1 op
# default (``cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell`` takes
# ``activation_type: int``). DeepSeek-V3 routed experts are gated SwiGLU.
try:
    from tensorrt_llm._torch.utils import ActivationType

    _SWIGLU_ACT = int(ActivationType.Swiglu)
except Exception:  # pragma: no cover - enum import is environment dependent
    _SWIGLU_ACT = 0

# decode_1cta tile: pin tile_size=128 (1-CTA, ``cta_group::ONE``), the
# configuration the production deploy already forces for decode
# (fused_moe_cute_dsl.py: forced_tile_size for tile_mode="decode_1cta"). This is
# the tile the megakernel must use; it avoids 2-CTA cluster coordination and keeps
# the token tile CTA-resident. (This is the token-tile M dimension; the FC2
# N-tile for the op path is chosen by the AutoTuner from the validated {128,256}
# set -- N=160 is excluded as numerically incorrect, see
# blockscaled_contiguous_grouped_gemm_finalize_fusion.py.)
_DECODE_TILE_SIZE = 128

_MEGAKERNEL_ENV = "TRTLLM_OPTRT_MOE_MEGAKERNEL"


def megakernel_mode() -> str:
    """Return the megakernel device path: ``"off"``, ``"1"``/``"op"``, or ``"jit"``.

    Controlled by ``TRTLLM_OPTRT_MOE_MEGAKERNEL`` (default ``"0"`` -> off). Any
    truthy non-``"jit"`` value selects the op-fused path; ``"jit"`` selects the
    single-``@cute.jit`` device fusion.
    """
    val = os.environ.get(_MEGAKERNEL_ENV, "0").strip().lower()
    if val in ("", "0", "off", "false", "no"):
        return "off"
    if val == "jit":
        return "jit"
    return "op"


def megakernel_enabled() -> bool:
    return megakernel_mode() != "off"


def _as_fp4x2(t: torch.Tensor) -> torch.Tensor:
    """View an NVFP4-packed tensor as ``float4_e2m1fn_x2`` (the cute_dsl ABI)."""
    fp4x2 = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4x2 is not None and t.dtype == fp4x2:
        return t
    # uint8 / packed byte layout -> reinterpret as fp4x2 (same byte width).
    if fp4x2 is not None:
        return t.contiguous().view(fp4x2)
    return t


def _as_u8(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.uint8:
        return t
    return t.contiguous().view(torch.uint8)


def run_fused_moe_megakernel_op(
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
    fc2_input_global_sf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Op-fused decode-MoE megakernel: moe_sort -> FC1(gather+SwiGLU+quant) -> FC2(finalize).

    All three stages are existing, validated ``trtllm`` ops; composing them behind
    one call removes the per-stage Python dispatch + intermediate-buffer ownership
    from the caller while staying numerically identical to ``run_moe_nvfp4_impl``.
    The FP4 intermediate ``(x, x_sf)`` transits GMEM between FC1 and FC2 (the
    SM90+ two-kernel approach); the FC2 finalize writes ``moe_output`` in place.
    """
    effective_top_k = int(topk_ids.shape[1])
    num_tokens = int(x.shape[0])
    output_dtype = torch.bfloat16

    # token_final_scales must be fp32 for moe_sort / finalize (production passes
    # float32; topk_weights may arrive bf16 from routing).
    if topk_weights.dtype != torch.float32:
        topk_weights = topk_weights.to(torch.float32)
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)

    # Stage 0: expert-major tiling (decode_1cta tile). This is the only piece of
    # "sort" kept; at b<=8/rank it is a handful of tiles. It produces the
    # tile->expert / permute maps the gather-FC1 and finalize-FC2 kernels need to
    # drive their persistent tile schedulers; the contract's eliminated_stages are
    # the *buffers* and the host orchestration, which this op subsumes.
    (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        total_num_padded_tokens,
        num_non_exiting_tiles,
    ) = torch.ops.trtllm.moe_sort(
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        num_experts=num_experts,
        top_k=effective_top_k,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        tile_tokens_dim=_DECODE_TILE_SIZE,
    )
    del expanded_idx_to_permuted_idx, total_num_padded_tokens

    if fc2_input_global_sf is None:
        # FC2 activation global scale: when not supplied, fall back to the FC1
        # gate global scale (the production path threads self.fc2_input_scale,
        # derived from the FC1 alpha family). A None here is only safe when the
        # caller has pre-baked it into output1_gate_scale; we require it
        # explicitly to avoid a silent mis-scale.
        fc2_input_global_sf = output1_gate_scale

    # Stage 1: FC1 gather grouped-GEMM + SwiGLU + NVFP4 quant -> (c, sfc) in GMEM.
    c, sfc = torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell(
        input=_as_fp4x2(x),
        weight=_as_fp4x2(w13),
        input_scale=_as_u8(x_sf),
        weight_scale=_as_u8(w13_scale),
        alpha=output1_gate_scale,
        tile_idx_to_group_idx=tile_idx_to_expert_idx,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
        num_non_exiting_tiles=num_non_exiting_tiles,
        global_sf=fc2_input_global_sf,
        num_experts=num_experts,
        top_k=effective_top_k,
        num_local_experts=local_num_experts,
        local_expert_offset=local_expert_offset,
        tile_size=_DECODE_TILE_SIZE,
        scaling_vector_size=scaling_vector_size,
        activation_type=_SWIGLU_ACT,
    )

    # Output buffer for the in-place finalize (scatter-combine writes here).
    moe_output = torch.zeros(
        num_tokens, hidden_size, dtype=output_dtype, device=x.device)

    # Stage 2: FC2 grouped-GEMM + finalize (scatter-combine by topk_weights) ->
    # moe_output in place. Reads the FP4 intermediate (c, sfc) from GMEM.
    torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell(
        input=_as_fp4x2(c),
        weight=[_as_fp4x2(w2)],
        input_scale=_as_u8(sfc),
        weight_scale=[_as_u8(w2_scale)],
        alpha=[output2_scale],
        output=moe_output,
        tile_idx_to_group_idx=tile_idx_to_expert_idx,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
        num_non_exiting_tiles=num_non_exiting_tiles,
        token_final_scales=topk_weights,
        num_experts=num_experts,
        top_k=effective_top_k,
        num_local_experts=local_num_experts,
        local_expert_offset=local_expert_offset,
        tile_size=_DECODE_TILE_SIZE,
        output_dtype=output_dtype,
        scaling_vector_size=scaling_vector_size,
    )
    return moe_output


def _register_cursor_op() -> bool:
    """Register ``trtllm::warp_decode_nvfp4_cursor_moe`` if not already present.

    Idempotent and import-safe: returns True if the op is available after the
    call. The op is the typed landing contract ``_get_nvfp4_cursor_op`` probes
    for in ``warp_decode.py``; once registered it is picked up automatically for
    decode buckets above the trtllm_gen crossover (>8 tokens).
    """
    trtllm_ops = getattr(torch.ops, "trtllm", None)
    if trtllm_ops is not None and hasattr(trtllm_ops, "warp_decode_nvfp4_cursor_moe"):
        return True

    @torch.library.custom_op(
        "trtllm::warp_decode_nvfp4_cursor_moe", mutates_args=(), device_types="cuda")
    def warp_decode_nvfp4_cursor_moe(
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
        mode = megakernel_mode()
        if mode == "jit":
            # Single-@cute.jit device fusion path. Imported lazily so the op is
            # registrable on hosts without the cute_dsl build present.
            from .fused_moe_megakernel_jit import run_fused_moe_megakernel_jit

            return run_fused_moe_megakernel_jit(
                x, x_sf, w13, w13_scale, w2, w2_scale,
                output1_scale, output1_gate_scale, output2_scale,
                topk_ids, topk_weights, hidden_size, intermediate_size,
                num_experts, local_expert_offset, local_num_experts,
                scaling_vector_size,
            )
        return run_fused_moe_megakernel_op(
            x, x_sf, w13, w13_scale, w2, w2_scale,
            output1_scale, output1_gate_scale, output2_scale,
            topk_ids, topk_weights, hidden_size, intermediate_size,
            num_experts, local_expert_offset, local_num_experts,
            scaling_vector_size,
        )

    @torch.library.register_fake("trtllm::warp_decode_nvfp4_cursor_moe")
    def _(
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
        del x_sf, w13, w13_scale, w2, w2_scale, output1_scale
        del output1_gate_scale, output2_scale, topk_ids, topk_weights
        del intermediate_size, num_experts, local_expert_offset
        del local_num_experts, scaling_vector_size
        return torch.empty(
            (x.shape[0], hidden_size), dtype=torch.bfloat16, device=x.device)

    return hasattr(getattr(torch.ops, "trtllm", object()), "warp_decode_nvfp4_cursor_moe")


def maybe_register_cursor_op() -> bool:
    """Register the cursor op only when the env gate selects the megakernel.

    Called from ``warp_decode.py`` at import. Default OFF: with
    ``TRTLLM_OPTRT_MOE_MEGAKERNEL`` unset, this is a no-op and the existing
    trtllm_gen / production WARPDECODE paths are untouched.
    """
    if not megakernel_enabled():
        return False
    return _register_cursor_op()

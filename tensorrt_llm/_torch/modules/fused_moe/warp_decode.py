# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-gated WarpDecode MoE fast path (legacy trtllm_gen overlay).

Relationship to the canonical WarpDecode backend
-------------------------------------------------
The CANONICAL WarpDecode path is the output-owned NVFP4 decode realized by
``CuteDslFusedMoE`` and selected with ``moe_backend="WARPDECODE"`` (see
``create_moe.get_moe_cls`` and ``WARPDECODE.md``). That path drives the
``cute_dsl`` gather-grouped-GEMM + SwiGLU / grouped-GEMM-finalize ops under the
AutoTuner and is controlled by ``WarpDecodeConfig.tile_mode``.

This module is a SECONDARY, opt-in overlay that runs a trtllm_gen
``FP4BlockScaleMoERunner`` at the scheduler dispatch point (after routing,
optional EPLB routing, and quantization metadata are materialized) when
``MoeConfig.warp_decode.enabled`` is set on top of another backend. It exists
for the decode-only crossover where the trtllm_gen runner is competitive; the
native scheduler remains responsible for TP/EP, attention-DP, CUDA-graph, and
disaggregated-serving control flow. By default it lets the runner pick its
tactic automatically (``tactic=[-1, -1]``); the frozen ``_NVFP4_TARGET_*``
tables are an explicitly-selected override only and imply no measured speedup.
Prefer the ``WARPDECODE`` backend; keep this overlay disabled unless you have a
specific reason to use the trtllm_gen fast path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

from ...custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
from ...utils import ActType_TrtllmGen

if TYPE_CHECKING:
    from .configurable_moe import ConfigurableMoE


def _maybe_register_megakernel_cursor_op() -> None:
    """Optionally register the Phase-1 persistent decode-MoE megakernel op.

    Env-gated by ``TRTLLM_OPTRT_MOE_MEGAKERNEL`` (default off): when set, registers
    ``trtllm::warp_decode_nvfp4_cursor_moe`` -- the typed landing contract that
    ``_get_nvfp4_cursor_op`` below probes for. This is additive and side-effect
    free when the gate is unset (the function returns immediately) and import-safe
    on hosts without a cute_dsl build (registration failures are swallowed so the
    overlay falls back to the existing trtllm_gen / native paths). See
    ``cute_dsl_kernels/blackwell/moe_as_dense_gemm/fused_moe_megakernel.py``.
    """
    if os.environ.get("TRTLLM_OPTRT_MOE_MEGAKERNEL", "0").strip().lower() in (
            "", "0", "off", "false", "no"):
        return
    try:
        from ...cute_dsl_kernels.blackwell.moe_as_dense_gemm.fused_moe_megakernel import (
            maybe_register_cursor_op,
        )

        if maybe_register_cursor_op():
            logger.info_once(
                "WarpDecode megakernel: registered "
                "trtllm::warp_decode_nvfp4_cursor_moe "
                "(TRTLLM_OPTRT_MOE_MEGAKERNEL set).",
                key="warp_decode_megakernel_registered")
    except Exception as exc:  # noqa: BLE001 - registration must never break import
        logger.warning_once(
            "WarpDecode megakernel registration skipped (%s); falling back to "
            "the existing NVFP4 overlay paths." % type(exc).__name__,
            key="warp_decode_megakernel_register_failed")


_maybe_register_megakernel_cursor_op()


class WarpDecodeStatus(str, Enum):
    DISABLED = "disabled"
    FALLBACK = "fallback"
    NOT_APPLICABLE = "not_applicable"
    SELECTED = "selected"


_BF16_SUPPORTED_BACKENDS = {"CutlassFusedMoE", "DenseGEMMFusedMoE", "TRTLLMGenFusedMoE"}
_NVFP4_SUPPORTED_BACKENDS = {"TRTLLMGenFusedMoE", "CuteDslFusedMoE"}
_NVFP4_TARGET_HIDDEN_SIZE = 7168
_NVFP4_TARGET_INTERMEDIATE_SIZE = 2048
_NVFP4_TARGET_NUM_EXPERTS = 128
_NVFP4_TARGET_TOP_K = 8
_NVFP4_TARGET_SCALING_VECTOR_SIZE = 16
_NVFP4_SUPPORTED_INPUT_DTYPES = {torch.uint8, getattr(torch, "float4_e2m1fn_x2", torch.uint8)}
_NVFP4_TRTLLM_GEN_CROSSOVER_TOKENS = 8
_NVFP4_CURSOR_GRAPH_BUCKETS = (1, 2, 4, 8, 16, 32)
_NVFP4_CURSOR_WARPS_PER_CTA = 8
_TRTLLM_GEN_DEEPSEEK_V3_ROUTING = 2
_TRTLLM_GEN_SWIGLU = 0
# Per-bucket [tile_tokens_dim, tactic] for the trtllm_gen FP4BlockScaleMoERunner,
# RE-TUNED for the production TP/dense-E decode regime (DP2/TP4: 128 experts
# present, intermediate_size sharded to I=2048/4=512, topk=8, H=7168, REAP-128).
# Selected by warpdecode_tactic_retune.py: enumerate get_valid_configs() per bucket
# at the TP4 shape, graph + time each, pick the fastest whose output is numerically
# identical to the AutoTuner-picked tactic (cos=1.0 vs autotuned, validated against
# run_moe_reference_fp4). These REPLACE the prior EP-rank-tuned values, which ran
# 1.08-1.28x slower at this regime (bs1 18.47us vs old 20.52us; bs32 26.71us vs
# old 31.32us). They are graph-safe: a fixed tactic means no host AutoTuner call
# inside the captured decode region. Used when TRTLLM_WARP_DECODE_FIXED_TACTIC=1;
# the default overlay path still passes tactic=[-1,-1] (AutoTuner), which lands on
# the same tactic after warmup. TP8 (I=256) alt-topology table is in
# tests/scripts/cute_dsl_kernels/RESULTS_trtllm_gen_fp4.md.
_NVFP4_TARGET_TACTICS = {
    1: [8, 81],
    2: [8, 81],
    4: [8, 81],
    8: [8, 70],
    16: [16, 52],
    32: [32, 52],
}



@dataclass(frozen=True)
class CursorWarpDecodePlan:
    """Static launch and workspace contract for the output-owned NVFP4 path.

    The current selected NVFP4 implementation below remains the guarded
    TRTLLMGen crossover for <=8 decode tokens.  This plan describes the next
    production kernel contract so graph bucket sizing, scratch reuse, and route
    metadata ownership are testable before the CuTe/CZS kernels are landed.
    """

    requested_tokens: int
    bucket_tokens: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    scaling_vector_size: int
    warps_per_cta: int

    @property
    def exact_expanded_rows(self) -> int:
        return self.bucket_tokens * self.top_k

    @property
    def gate_up_warps(self) -> int:
        return self.exact_expanded_rows * self.intermediate_size

    @property
    def down_warps(self) -> int:
        return self.bucket_tokens * self.hidden_size

    @property
    def route_slots_shape(self) -> Tuple[int, int]:
        return (self.bucket_tokens, self.top_k)

    @property
    def route_scales_shape(self) -> Tuple[int, int]:
        return (self.bucket_tokens, self.top_k)

    @property
    def intermediate_shape(self) -> Tuple[int, int, int]:
        return (self.bucket_tokens, self.top_k, self.intermediate_size)

    @property
    def output_shape(self) -> Tuple[int, int]:
        return (self.bucket_tokens, self.hidden_size)

    @property
    def activation_scale_shape(self) -> Tuple[int, int]:
        return (
            self.bucket_tokens,
            self.hidden_size // self.scaling_vector_size,
        )

    @property
    def eliminated_stages(self) -> Tuple[str, ...]:
        return (
            "expert_major_batches",
            "expert_padding",
            "moe_sort",
            "scatter_combine",
            "activation_gather_buffer",
            "per_expert_output_buffer",
        )


def _cursor_bucket_for_num_tokens(num_tokens: int) -> int:
    if num_tokens <= 0:
        raise ValueError("WarpDecode cursor bucket requires a positive token count.")
    for bucket in _NVFP4_CURSOR_GRAPH_BUCKETS:
        if num_tokens <= bucket:
            return bucket
    raise ValueError(
        f"WarpDecode cursor bucket does not cover {num_tokens} tokens; "
        f"supported buckets are {_NVFP4_CURSOR_GRAPH_BUCKETS}.")


def get_cursor_warp_decode_plan(num_tokens: int) -> CursorWarpDecodePlan:
    """Return the graph-stable output-owned NVFP4 kernel contract.

    Route values are still produced every decode step, but the route tensor
    addresses and scratch buffers should be bucket-stable.  Kernels consume
    post-EPLB/post-dispatch ``token_selected_slots`` and ``token_final_scales``
    in this exact shape; they do not build expert-major padded rows.
    """

    bucket_tokens = _cursor_bucket_for_num_tokens(num_tokens)
    return CursorWarpDecodePlan(
        requested_tokens=num_tokens,
        bucket_tokens=bucket_tokens,
        top_k=_NVFP4_TARGET_TOP_K,
        hidden_size=_NVFP4_TARGET_HIDDEN_SIZE,
        intermediate_size=_NVFP4_TARGET_INTERMEDIATE_SIZE,
        num_experts=_NVFP4_TARGET_NUM_EXPERTS,
        scaling_vector_size=_NVFP4_TARGET_SCALING_VECTOR_SIZE,
        warps_per_cta=_NVFP4_CURSOR_WARPS_PER_CTA,
    )


def _config(moe: "ConfigurableMoE"):
    model_config = getattr(moe, "model_config", None)
    return getattr(model_config, "warp_decode_config", None)


def _policy(config) -> str:
    return getattr(config, "policy", "auto")


def _record(moe: "ConfigurableMoE", status: WarpDecodeStatus, reason: str) -> None:
    moe.warp_decode_last_status = status.value
    moe.warp_decode_last_reason = reason


def _log_key(reason: str) -> str:
    safe_reason = "".join(c if c.isalnum() or c == "_" else "_" for c in reason)
    return f"warp_decode_fallback_{safe_reason}"


def _guard_failure(moe: "ConfigurableMoE", config, reason: str) -> None:
    _record(moe, WarpDecodeStatus.FALLBACK, reason)
    fallback_allowed = getattr(config, "allow_parallelism_fallback", True)
    if _policy(config) == "force" or (
            _policy(config) != "fallback_only" and not fallback_allowed):
        raise NotImplementedError(
            f"WarpDecode FALLBACK forbidden by policy; runtime guard failed (reason={reason}).")
    if _policy(config) != "fallback_only":
        logger.warning_once(
            f"WarpDecode FALLBACK to native MoE backend (reason={reason}).",
            key=_log_key(reason))


def _top_k(moe: "ConfigurableMoE", token_selected_experts: torch.Tensor) -> int:
    if token_selected_experts is not None:
        return int(token_selected_experts.shape[-1])
    routing_method = getattr(moe, "routing_method", None)
    return int(
        getattr(routing_method, "experts_per_token", getattr(routing_method, "top_k", 0)))


def _is_pure_decode(moe: "ConfigurableMoE") -> bool:
    return bool(getattr(moe, "warp_decode_is_decode_only", False))


def _num_tokens(x: torch.Tensor) -> int:
    return int(x.shape[0])


def _is_cuda_graph(moe: "ConfigurableMoE") -> bool:
    return bool(getattr(moe, "warp_decode_is_cuda_graph", False))


def _is_deepep_ll_layout(moe: "ConfigurableMoE") -> bool:
    """Whether the active comm strategy hands the MoE an expert-major recv.

    DeepEPLowLatency._modify_output_to_adapt_fused_moe flattens the padded
    [num_local_experts, ep*token_limit, H] recv into top-1 rows with a
    num_slots sentinel on masked rows. The overlay's trtllm_gen runner
    requires token-major top-k routing, so it can never serve this layout;
    the canonical WARPDECODE CuteDslFusedMoE backend consumes it natively
    (moe_sort drops the sentinel rows without inflating tiles).
    """
    comm = getattr(moe, "comm", None)
    if comm is None:
        return False
    from .communication.deep_ep_low_latency import DeepEPLowLatency

    return isinstance(comm, DeepEPLowLatency)


def _tensor_data(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if isinstance(tensor, torch.nn.Parameter):
        return tensor.data
    return tensor


def _nvfp4_weight_bytes(weight: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Return FP4 weights in the byte-packed layout consumed by trtllm_gen."""
    if weight is None:
        return None
    fp4x2_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if weight.dtype == torch.uint8 or (
            fp4x2_dtype is not None and weight.dtype == fp4x2_dtype):
        return weight
    return weight.contiguous().view(torch.uint8)


def _nvfp4_scale_fp8(scale: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Return NVFP4 block scales in the FP8 layout consumed by trtllm_gen."""
    if scale is None:
        return None
    if scale.dtype == torch.float8_e4m3fn:
        return scale
    return scale.contiguous().view(torch.float8_e4m3fn)


def _is_standard_bf16_weight_layout(moe: "ConfigurableMoE") -> bool:
    quant_method = getattr(moe.backend, "quant_method", None)
    if getattr(quant_method, "use_shuffled_weight", False):
        return False
    if getattr(moe.backend, "_trtllm_gen_layout_transform_pending", False):
        return False
    return True


def _get_bf16_op():
    trtllm_ops = getattr(torch.ops, "trtllm", None)
    if trtllm_ops is not None and hasattr(trtllm_ops, "warp_decode_bf16_moe_packed"):
        return trtllm_ops.warp_decode_bf16_moe_packed

    try:
        from sglang.srt.layers.moe.warp_decode.kernels import (  # type: ignore[import-not-found]
            warp_decode_moe_packed,
        )
    except (AttributeError, ImportError):
        return None
    return warp_decode_moe_packed


def _required_nvfp4_ops_available() -> bool:
    trtllm_ops = getattr(torch.ops, "trtllm", None)
    return trtllm_ops is not None and callable(FP4BlockScaleMoERunner)


_TRTLLM_GEN_AUTO_TACTIC = [-1, -1]


def _shape_has_fixed_tactic(num_tokens: int) -> bool:
    """Whether ``_NVFP4_TARGET_TACTICS`` covers this decode token count.

    The retuned table is keyed by the cursor graph buckets (1,2,4,8,16,32).
    Token counts above the largest bucket (e.g. padded decode batches 48/64)
    have no validated entry, so we must not pin a tactic for them.
    """
    return 0 < num_tokens <= _NVFP4_CURSOR_GRAPH_BUCKETS[-1]


def _use_fixed_overlay_tactic() -> bool:
    """Whether the overlay should pin a hand-enumerated [tile, tactic] pair.

    Default is True (optimal-on): the frozen ``_NVFP4_TARGET_TACTICS`` table is
    the warpdecode_tactic_retune.py output for the production TP/dense-E decode
    regime, validated cos=1.0 against the AutoTuner-picked tactic, and is
    graph-safe (a fixed tactic means no host AutoTuner call inside the captured
    decode region, removing warmup-order variance). The AutoTuner converges to
    the same tactic after warmup, so pinning is numerically identical while
    avoiding the in-graph host call. Set ``TRTLLM_WARP_DECODE_FIXED_TACTIC=0``
    to opt back out to the pure ``tactic=[-1,-1]`` AutoTuner path. Shapes the
    table does not cover always fall back to AutoTuner (see
    ``_nvfp4_overlay_tactic``), so this default never errors on uncovered
    (e.g. padded 48/64) buckets.
    """
    return os.environ.get("TRTLLM_WARP_DECODE_FIXED_TACTIC", "1") == "1"


def _nvfp4_target_tactic(num_tokens: int) -> List[int]:
    bucket_tokens = _cursor_bucket_for_num_tokens(num_tokens)
    return _NVFP4_TARGET_TACTICS[bucket_tokens]


def _nvfp4_overlay_tactic(num_tokens: int) -> List[int]:
    """Tactic for the overlay runner: pinned retuned tactic by default for
    covered decode buckets, AutoTuner otherwise.

    The fixed table is used only when (a) it is not explicitly disabled via
    TRTLLM_WARP_DECODE_FIXED_TACTIC=0 and (b) the token count is covered by the
    retuned buckets. Uncovered shapes (padded 48/64 decode, or any prefill-ish
    count above the largest bucket) fall back to tactic=[-1,-1] so the overlay
    never raises on an unmapped bucket -- the AutoTuner picks the same family
    of configs there.
    """
    if _use_fixed_overlay_tactic() and _shape_has_fixed_tactic(num_tokens):
        return _nvfp4_target_tactic(num_tokens)
    return list(_TRTLLM_GEN_AUTO_TACTIC)


def _run_nvfp4_explicit_tactic(
    x: torch.Tensor,
    x_sf: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    output1_scale: torch.Tensor,
    output1_gate_scale: torch.Tensor,
    output2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    local_expert_offset: int,
    local_num_experts: int,
    scaling_vector_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    w13 = _nvfp4_weight_bytes(w13)
    w2 = _nvfp4_weight_bytes(w2)
    w13_scale = _nvfp4_scale_fp8(w13_scale)
    w2_scale = _nvfp4_scale_fp8(w2_scale)
    if w13 is None or w2 is None or w13_scale is None or w2_scale is None:
        raise RuntimeError("NVFP4 tensors disappeared after guard check.")
    packed_hidden_size = int(w13.shape[-1])
    padded_hidden_size = packed_hidden_size * 2
    if x.shape[-1] < packed_hidden_size:
        x = _pad_nvfp4_last_dim(x, packed_hidden_size)
    expected_scale_cols = padded_hidden_size // scaling_vector_size
    if x_sf.shape[-1] < expected_scale_cols:
        x_sf = _pad_nvfp4_last_dim(x_sf, expected_scale_cols)
    if out is not None:
        # M1 one-sided a2a (NVLinkOneSided combine-into-workspace): write the MoE
        # result straight into the comm workspace payload tensor instead of a fresh
        # buffer, so the downstream payload_in_workspace=True combine is zero-copy.
        # `out` is the 2D workspace view [ep_size*max_tokens, hidden] from
        # get_combine_payload_tensor_in_workspace(); it is contiguous with
        # numel == x.shape[0]*hidden, so this is a no-copy reshape.
        output = out.view(x.shape[0], hidden_size)
    else:
        output = torch.empty((x.shape[0], hidden_size), dtype=torch.bfloat16, device=x.device)
    # Enable PDL for the decode bucket. The direct C++ runner call avoids the
    # registered custom-op dispatcher and Python TunableRunner wrapper while
    # preserving the same trtllm_gen kernels. Tactic selection defaults to auto
    # (_nvfp4_overlay_tactic) so this stays consistent with the canonical
    # WARPDECODE backend's autotune-default policy.
    os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
    runner = _nvfp4_torch_runner()
    result = runner.run_moe(
        None,
        None,
        x,
        x_sf.flatten().view(torch.float8_e4m3fn),
        w13,
        w13_scale,
        None,
        None,
        None,
        None,
        w2,
        w2_scale,
        None,
        output1_scale,
        output1_gate_scale,
        output2_scale,
        num_experts,
        topk_ids.shape[1],
        8,
        4,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        None,
        _TRTLLM_GEN_DEEPSEEK_V3_ROUTING,
        True,
        _nvfp4_overlay_tactic(int(x.shape[0])),
        topk_weights.to(torch.bfloat16),
        topk_ids,
        output,
    )
    # The kernel writes in-place into `output`. When that is the M1 comm
    # workspace view, return the exact workspace buffer so the downstream
    # payload_in_workspace=True combine sees the matching data_ptr.
    return output if out is not None else result[0]


@lru_cache(maxsize=None)
def _nvfp4_torch_runner():
    return torch.classes.trtllm.FP4BlockScaleMoERunner(_TRTLLM_GEN_SWIGLU)


def _pad_nvfp4_last_dim(tensor: torch.Tensor, target_cols: int) -> torch.Tensor:
    """Pad packed NVFP4 payloads/scales to the runner's padded hidden contract."""
    if tensor.shape[-1] == target_cols:
        return tensor.contiguous()
    if tensor.shape[-1] > target_cols:
        raise ValueError(
            f"NVFP4 tensor width {tensor.shape[-1]} exceeds target {target_cols}.")
    padded = tensor.new_zeros((*tensor.shape[:-1], target_cols))
    padded[..., :tensor.shape[-1]].copy_(tensor)
    return padded


@lru_cache(maxsize=None)
def _nvfp4_runner(
    num_experts: int,
    top_k: int,
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
) -> FP4BlockScaleMoERunner:
    return FP4BlockScaleMoERunner(
        num_experts,
        top_k,
        8,
        4,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        None,
        _TRTLLM_GEN_DEEPSEEK_V3_ROUTING,
        True,
        ActType_TrtllmGen.SwiGlu.value,
        tune_max_num_tokens=8192,
        use_dp=False,
    )


@torch.library.custom_op(
    "trtllm::warp_decode_nvfp4_moe", mutates_args=(), device_types="cuda")
def _warp_decode_nvfp4_moe(
    x: torch.Tensor,
    x_sf: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    output1_scale: torch.Tensor,
    output1_gate_scale: torch.Tensor,
    output2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    local_expert_offset: int,
    local_num_experts: int,
    scaling_vector_size: int,
) -> torch.Tensor:
    return _run_nvfp4_explicit_tactic(
        x,
        x_sf,
        w13,
        w13_scale,
        w2,
        w2_scale,
        output1_scale,
        output1_gate_scale,
        output2_scale,
        topk_ids,
        topk_weights,
        hidden_size,
        intermediate_size,
        num_experts,
        local_expert_offset,
        local_num_experts,
        scaling_vector_size,
    )


def _warp_decode_nvfp4_moe_autotuned_reference(
    x: torch.Tensor,
    x_sf: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    output1_scale: torch.Tensor,
    output1_gate_scale: torch.Tensor,
    output2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    local_expert_offset: int,
    local_num_experts: int,
    scaling_vector_size: int,
) -> torch.Tensor:
    w13 = _nvfp4_weight_bytes(w13)
    w2 = _nvfp4_weight_bytes(w2)
    w13_scale = _nvfp4_scale_fp8(w13_scale)
    w2_scale = _nvfp4_scale_fp8(w2_scale)
    if w13 is None or w2 is None or w13_scale is None or w2_scale is None:
        raise RuntimeError("NVFP4 tensors disappeared after guard check.")
    packed_hidden_size = int(w13.shape[-1])
    padded_hidden_size = packed_hidden_size * 2
    if x.shape[-1] < packed_hidden_size:
        x = _pad_nvfp4_last_dim(x, packed_hidden_size)
    expected_scale_cols = padded_hidden_size // scaling_vector_size
    if x_sf.shape[-1] < expected_scale_cols:
        x_sf = _pad_nvfp4_last_dim(x_sf, expected_scale_cols)
    output = torch.empty((x.shape[0], hidden_size), dtype=torch.bfloat16, device=x.device)
    outputs = torch.ops.trtllm.fp4_block_scale_moe_runner(
        None,
        None,
        x,
        x_sf.flatten().view(torch.float8_e4m3fn),
        w13,
        w13_scale,
        None,
        None,
        None,
        None,
        w2,
        w2_scale,
        None,
        output1_scale,
        output1_gate_scale,
        output2_scale,
        num_experts,
        topk_ids.shape[1],
        8,
        4,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        None,
        _TRTLLM_GEN_DEEPSEEK_V3_ROUTING,
        True,
        _TRTLLM_GEN_SWIGLU,
        topk_weights.to(torch.bfloat16),
        topk_ids,
        output,
        8192,
        False,
    )
    return outputs[0]


@torch.library.register_fake("trtllm::warp_decode_nvfp4_moe")
def _(
    x: torch.Tensor,
    x_sf: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    output1_scale: torch.Tensor,
    output1_gate_scale: torch.Tensor,
    output2_scale: torch.Tensor,
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
    return torch.empty((x.shape[0], hidden_size), dtype=torch.bfloat16, device=x.device)


def _get_nvfp4_op():
    if _required_nvfp4_ops_available():
        return _run_nvfp4_explicit_tactic
    return None



def _get_nvfp4_cursor_op():
    trtllm_ops = getattr(torch.ops, "trtllm", None)
    if trtllm_ops is not None and hasattr(trtllm_ops, "warp_decode_nvfp4_cursor_moe"):
        op = trtllm_ops.warp_decode_nvfp4_cursor_moe
        if callable(op):
            return op
    return None


def _has_nvfp4_cursor_op() -> bool:
    return _get_nvfp4_cursor_op() is not None


def _get_supported_nvfp4_cursor_op(num_tokens: int):
    try:
        get_cursor_warp_decode_plan(num_tokens)
    except ValueError:
        return None
    return _get_nvfp4_cursor_op()


def _backend_int(backend, name: str, default: int = 0) -> int:
    return int(getattr(backend, name, default))


def _is_nvfp4_target_model_shape(backend, x: torch.Tensor, x_sf: torch.Tensor) -> bool:
    hidden_size = _backend_int(backend, "hidden_size", _NVFP4_TARGET_HIDDEN_SIZE)
    intermediate_size = _backend_int(
        backend,
        "intermediate_size",
        _backend_int(backend, "intermediate_size_per_partition", _NVFP4_TARGET_INTERMEDIATE_SIZE),
    )
    num_experts = _backend_int(backend, "num_experts", _backend_int(backend, "num_slots", 0))
    scaling_vector_size = _backend_int(
        backend, "scaling_vector_size", _NVFP4_TARGET_SCALING_VECTOR_SIZE)
    return (
        x.dim() == 2
        and x.shape[1] * 2 == _NVFP4_TARGET_HIDDEN_SIZE
        and x_sf.dim() == 2
        and x_sf.shape[0] == x.shape[0]
        and x_sf.shape[1] == _NVFP4_TARGET_HIDDEN_SIZE // _NVFP4_TARGET_SCALING_VECTOR_SIZE
        and hidden_size == _NVFP4_TARGET_HIDDEN_SIZE
        and intermediate_size == _NVFP4_TARGET_INTERMEDIATE_SIZE
        and num_experts == _NVFP4_TARGET_NUM_EXPERTS
        and scaling_vector_size == _NVFP4_TARGET_SCALING_VECTOR_SIZE
    )


def _bf16_guard_failure(
    moe: "ConfigurableMoE",
    *,
    x: torch.Tensor,
    token_selected_experts: Optional[torch.Tensor],
    token_final_scales: Optional[torch.Tensor],
) -> Optional[str]:
    if moe.backend.__class__.__name__ not in _BF16_SUPPORTED_BACKENDS:
        return f"bf16_unsupported_backend_{moe.backend.__class__.__name__}"
    if getattr(moe, "comm", None) is not None:
        return "bf16_external_comm_not_supported"
    if getattr(moe, "enable_dwdp", False):
        return "bf16_dwdp_not_supported"
    if x.dtype != torch.bfloat16:
        return f"bf16_input_dtype_{x.dtype}"
    if token_selected_experts is None or token_final_scales is None:
        return "routing_not_materialized"
    if token_selected_experts.dtype != torch.int32:
        return f"topk_ids_dtype_{token_selected_experts.dtype}"
    if token_final_scales.dtype not in (torch.float32, torch.bfloat16):
        return f"topk_weights_dtype_{token_final_scales.dtype}"
    if _top_k(moe, token_selected_experts) != 8:
        return "top_k_not_8"

    w13 = _tensor_data(getattr(moe.backend, "w3_w1_weight", None))
    w2 = _tensor_data(getattr(moe.backend, "w2_weight", None))
    if w13 is None or w2 is None:
        return "bf16_missing_weights"
    if w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        return "bf16_weight_dtype_mismatch"
    if w13.dim() != 3 or w2.dim() != 3:
        return "bf16_weight_rank_mismatch"
    if w13.shape[0] != w2.shape[0]:
        return "bf16_weight_expert_mismatch"
    if w13.shape[-1] != x.shape[-1]:
        return "bf16_hidden_size_mismatch"
    if w13.shape[1] != 2 * w2.shape[2]:
        return "bf16_packed_gate_up_shape_mismatch"
    if w2.shape[1] != x.shape[-1]:
        return "bf16_down_hidden_size_mismatch"
    if not _is_standard_bf16_weight_layout(moe):
        return "bf16_nonstandard_weight_layout"
    if _get_bf16_op() is None:
        return "bf16_warp_decode_op_unavailable"
    return None


def _get_backend_tensor(moe: "ConfigurableMoE", *names: str) -> Optional[torch.Tensor]:
    getter = getattr(moe.backend, "_get_data_or_none", None)
    for name in names:
        tensor = getter(name) if getter is not None else None
        if tensor is None:
            tensor = getattr(moe.backend, name, None)
        tensor = _tensor_data(tensor)
        if tensor is not None:
            return tensor
    return None


def _nvfp4_guard_failure(
    moe: "ConfigurableMoE",
    *,
    x: torch.Tensor,
    token_selected_experts: Optional[torch.Tensor],
    token_final_scales: Optional[torch.Tensor],
    x_sf: Optional[torch.Tensor],
) -> Optional[str]:
    backend = moe.backend
    backend_name = backend.__class__.__name__
    if backend_name not in _NVFP4_SUPPORTED_BACKENDS:
        return f"nvfp4_unsupported_backend_{backend_name}"
    if getattr(moe, "enable_dwdp", False):
        return "nvfp4_dwdp_not_supported"
    if x_sf is None:
        return "nvfp4_missing_activation_scales"
    if token_selected_experts is None or token_final_scales is None:
        return "routing_not_materialized"
    if token_selected_experts.dtype != torch.int32:
        return f"topk_ids_dtype_{token_selected_experts.dtype}"
    if token_final_scales.dtype not in (torch.float32, torch.bfloat16):
        return f"topk_weights_dtype_{token_final_scales.dtype}"
    if _top_k(moe, token_selected_experts) != _NVFP4_TARGET_TOP_K:
        return "top_k_not_8"
    if _num_tokens(x) > _NVFP4_TRTLLM_GEN_CROSSOVER_TOKENS:
        if (
            _get_supported_nvfp4_cursor_op(_num_tokens(x)) is None
            and _get_nvfp4_op() is None
        ):
            return "nvfp4_cursor_warp_decode_op_unavailable"
    if x.dtype not in _NVFP4_SUPPORTED_INPUT_DTYPES:
        return f"nvfp4_input_dtype_{x.dtype}"
    if x_sf.dtype != torch.uint8:
        return f"nvfp4_activation_scale_dtype_{x_sf.dtype}"
    if x.dim() != 2 or x.shape[1] * 2 != _NVFP4_TARGET_HIDDEN_SIZE:
        return "nvfp4_input_shape_mismatch"
    if x_sf.dim() != 2 or x_sf.shape[0] != x.shape[0]:
        return "nvfp4_activation_scale_batch_mismatch"
    if x_sf.shape[1] != _NVFP4_TARGET_HIDDEN_SIZE // _NVFP4_TARGET_SCALING_VECTOR_SIZE:
        return "nvfp4_activation_scale_shape_mismatch"

    hidden_size = _backend_int(backend, "hidden_size", _NVFP4_TARGET_HIDDEN_SIZE)
    if hidden_size != _NVFP4_TARGET_HIDDEN_SIZE:
        return f"nvfp4_hidden_size_{hidden_size}"
    intermediate_size = _backend_int(
        backend,
        "intermediate_size",
        _backend_int(backend, "intermediate_size_per_partition", _NVFP4_TARGET_INTERMEDIATE_SIZE),
    )
    if intermediate_size != _NVFP4_TARGET_INTERMEDIATE_SIZE:
        return f"nvfp4_intermediate_size_{intermediate_size}"
    num_experts = _backend_int(backend, "num_experts", _backend_int(backend, "num_slots", 0))
    if num_experts != _NVFP4_TARGET_NUM_EXPERTS:
        return f"nvfp4_num_experts_{num_experts}"
    scaling_vector_size = _backend_int(
        backend, "scaling_vector_size", _NVFP4_TARGET_SCALING_VECTOR_SIZE)
    if scaling_vector_size != _NVFP4_TARGET_SCALING_VECTOR_SIZE:
        return "nvfp4_scaling_vector_size_not_16"

    w13 = _get_backend_tensor(moe, "w3_w1_weight")
    w13_scale = _get_backend_tensor(moe, "w3_w1_weight_scale", "w3_w1_weight_scaling_factor")
    w2 = _get_backend_tensor(moe, "w2_weight")
    w2_scale = _get_backend_tensor(moe, "w2_weight_scale", "w2_weight_scaling_factor")
    if w13 is None or w13_scale is None or w2 is None or w2_scale is None:
        return "nvfp4_missing_weights_or_scales"
    w13 = _nvfp4_weight_bytes(w13)
    w2 = _nvfp4_weight_bytes(w2)
    w13_scale = _nvfp4_scale_fp8(w13_scale)
    w2_scale = _nvfp4_scale_fp8(w2_scale)
    if w13 is None or w2 is None or w13_scale is None or w2_scale is None:
        return "nvfp4_missing_weights_or_scales"
    if w13.dim() != 3 or w2.dim() != 3:
        return "nvfp4_weight_rank_mismatch"
    if w13_scale.dim() != 3 or w2_scale.dim() != 3:
        return "nvfp4_weight_scale_rank_mismatch"
    packed_hidden_size = int(w13.shape[-1])
    padded_hidden_size = packed_hidden_size * 2
    if packed_hidden_size < int(x.shape[-1]):
        return "nvfp4_weight_hidden_smaller_than_input"
    if padded_hidden_size % scaling_vector_size != 0:
        return "nvfp4_padded_hidden_not_scale_aligned"
    if int(w13_scale.shape[-1]) != padded_hidden_size // scaling_vector_size:
        return "nvfp4_w13_scale_hidden_mismatch"
    if int(w2.shape[1]) != padded_hidden_size:
        return "nvfp4_w2_hidden_mismatch"
    if int(w2_scale.shape[1]) != padded_hidden_size:
        return "nvfp4_w2_scale_hidden_mismatch"
    if int(w2.shape[-1]) != intermediate_size // 2:
        return "nvfp4_w2_intermediate_mismatch"
    if int(w2_scale.shape[-1]) != intermediate_size // scaling_vector_size:
        return "nvfp4_w2_scale_intermediate_mismatch"
    if (
        _num_tokens(x) <= _NVFP4_TRTLLM_GEN_CROSSOVER_TOKENS
        and _get_nvfp4_op() is None
    ):
        return "nvfp4_warp_decode_op_unavailable"
    if (
        _num_tokens(x) > _NVFP4_TRTLLM_GEN_CROSSOVER_TOKENS
        and _get_nvfp4_cursor_op() is None
        and _get_nvfp4_op() is None
    ):
        return "nvfp4_cursor_warp_decode_op_unavailable"
    return None


def get_warp_decode_guard_failure(
    moe: "ConfigurableMoE",
    *,
    x: torch.Tensor,
    token_selected_experts: Optional[torch.Tensor],
    token_final_scales: Optional[torch.Tensor],
    x_sf: Optional[torch.Tensor],
    do_finalize: bool,
    all_rank_num_tokens: Optional[List[int]],
) -> Optional[str]:
    del all_rank_num_tokens

    config = _config(moe)
    if config is None or not getattr(config, "enabled", False):
        return "disabled"
    if _policy(config) == "fallback_only":
        return "policy_fallback_only"
    if not _is_pure_decode(moe):
        return "not_decode_only"
    if _is_deepep_ll_layout(moe):
        return "deepep_ll_expert_major_layout"
    if _is_cuda_graph(moe) and not getattr(moe, "has_nvfp4", False):
        return "cuda_graph_not_supported"
    if not do_finalize:
        return "finalize_disabled"
    if (not getattr(moe, "has_nvfp4", False)
            and getattr(moe, "layer_load_balancer", None) is not None):
        return "eplb_not_supported"
    if _num_tokens(x) == 0:
        return "empty_batch"
    if (
        not getattr(moe, "has_nvfp4", False)
        and _num_tokens(x) > getattr(config, "max_batch_size", 64)
    ):
        return "batch_too_large"

    if getattr(moe, "has_nvfp4", False):
        return _nvfp4_guard_failure(
            moe,
            x=x,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            x_sf=x_sf,
        )

    return _bf16_guard_failure(
        moe,
        x=x,
        token_selected_experts=token_selected_experts,
        token_final_scales=token_final_scales,
    )


def _run_bf16_warp_decode(
    moe: "ConfigurableMoE",
    *,
    x: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
) -> torch.Tensor:
    op = _get_bf16_op()
    if op is None:
        raise RuntimeError("BF16 WarpDecode op disappeared after guard check.")
    w13 = _tensor_data(moe.backend.w3_w1_weight)
    w2 = _tensor_data(moe.backend.w2_weight)
    intermediate_size = int(w2.shape[2])
    return op(
        hidden_states=x,
        w13=w13,
        w2=w2,
        topk_ids=token_selected_experts,
        topk_weights=token_final_scales,
        intermediate_size=intermediate_size,
        inplace=False,
    )


def _run_nvfp4_warp_decode(
    moe: "ConfigurableMoE",
    *,
    x: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    x_sf: torch.Tensor,
    moe_output: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, str]:
    output1_scale = _get_backend_tensor(moe, "fc31_scale_c")
    output1_gate_scale = _get_backend_tensor(moe, "fc31_alpha")
    output2_scale = _get_backend_tensor(moe, "fc2_alpha")
    if output1_scale is None:
        output1_scale = output1_gate_scale
    # Scalar FC2-input requant global scale for the megakernel cursor op. The
    # trtllm_gen backend carries it as a registered scalar parameter (fc31_scale_c
    # is derived from it); the cursor op needs it directly so it does NOT fall
    # back to the per-expert fc31_alpha (which trips the scalar-global_sf assert).
    fc2_input_scale = _get_backend_tensor(moe, "fc2_input_scale")
    local_num_experts = _backend_int(
        moe.backend, "expert_size_per_partition", _backend_int(moe.backend, "num_slots", 0))

    cursor_op = _get_supported_nvfp4_cursor_op(_num_tokens(x))
    use_cursor = _num_tokens(x) > _NVFP4_TRTLLM_GEN_CROSSOVER_TOKENS and cursor_op is not None
    op = cursor_op if use_cursor else _get_nvfp4_op()
    if op is None:
        raise RuntimeError("NVFP4 WarpDecode op disappeared after guard check.")
    op_args = (
        x,
        x_sf,
        _get_backend_tensor(moe, "w3_w1_weight"),
        _get_backend_tensor(moe, "w3_w1_weight_scale", "w3_w1_weight_scaling_factor"),
        _get_backend_tensor(moe, "w2_weight"),
        _get_backend_tensor(moe, "w2_weight_scale", "w2_weight_scaling_factor"),
        output1_scale,
        output1_gate_scale,
        output2_scale,
        token_selected_experts,
        token_final_scales,
        _NVFP4_TARGET_HIDDEN_SIZE,
        _NVFP4_TARGET_INTERMEDIATE_SIZE,
        _NVFP4_TARGET_NUM_EXPERTS,
        _backend_int(moe.backend, "slot_start", 0),
        local_num_experts,
        _NVFP4_TARGET_SCALING_VECTOR_SIZE,
    )
    if use_cursor:
        # Cursor megakernel op has a fixed torch-op schema (no out buffer). It
        # takes the scalar fc2_input_scale as a trailing arg (the FC2-input requant
        # global scale the trtllm_gen overlay otherwise omits). If M1 one-sided a2a
        # supplied a workspace payload tensor, land the result into it so the
        # downstream payload_in_workspace=True combine stays valid.
        output = op(*op_args, fc2_input_scale)
        if moe_output is not None:
            moe_output.view(output.shape).copy_(output)
            output = moe_output.view(output.shape)
        return output, "nvfp4_cursor_op"
    # Explicit-tactic path writes directly into the workspace payload (zero-copy)
    # when moe_output is provided (M1 NVLinkOneSided); else allocates its own.
    return op(*op_args, out=moe_output), "nvfp4_explicit_tactic_op"


def try_run_warp_decode(
    moe: "ConfigurableMoE",
    *,
    x: torch.Tensor,
    token_selected_experts: Optional[torch.Tensor],
    token_final_scales: Optional[torch.Tensor],
    x_sf: Optional[torch.Tensor],
    do_finalize: bool,
    all_rank_num_tokens: Optional[List[int]],
    moe_output: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    config = _config(moe)
    if config is None or not getattr(config, "enabled", False):
        _record(moe, WarpDecodeStatus.DISABLED, "disabled")
        return None

    reason = get_warp_decode_guard_failure(
        moe,
        x=x,
        token_selected_experts=token_selected_experts,
        token_final_scales=token_final_scales,
        x_sf=x_sf,
        do_finalize=do_finalize,
        all_rank_num_tokens=all_rank_num_tokens,
    )
    if reason is not None:
        if reason in ("not_decode_only", "deepep_ll_expert_major_layout"):
            _record(moe, WarpDecodeStatus.NOT_APPLICABLE, reason)
            return None
        _guard_failure(moe, config, reason)
        return None

    assert token_selected_experts is not None
    assert token_final_scales is not None
    if getattr(moe, "has_nvfp4", False):
        assert x_sf is not None
        output, selected_reason = _run_nvfp4_warp_decode(
            moe,
            x=x,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            x_sf=x_sf,
            moe_output=moe_output,
        )
        if selected_reason == "nvfp4_cursor_op":
            _record(moe, WarpDecodeStatus.SELECTED, selected_reason)
            logger.info_once("WarpDecode SELECTED (overlay): Cursor NVFP4 path.",
                             key="warp_decode_selected_nvfp4_cursor")
        elif selected_reason == "nvfp4_explicit_tactic_op":
            _record(moe, WarpDecodeStatus.SELECTED, selected_reason)
            logger.info_once("WarpDecode SELECTED (overlay): trtllm_gen NVFP4 path.",
                             key="warp_decode_selected_nvfp4_explicit_tactic")
        else:
            _record(moe, WarpDecodeStatus.SELECTED, selected_reason)
            logger.info_once("WarpDecode SELECTED (overlay): native NVFP4 path.",
                             key="warp_decode_selected_nvfp4")
    else:
        output = _run_bf16_warp_decode(
            moe,
            x=x,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
        )
        if moe_output is not None:
            # BF16 op allocates its own buffer; land it in the M1 comm workspace.
            moe_output.view(output.shape).copy_(output)
            output = moe_output.view(output.shape)
        _record(moe, WarpDecodeStatus.SELECTED, "bf16_op")
        logger.info_once("WarpDecode SELECTED (overlay): BF16 OP-compatible path.",
                         key="warp_decode_selected_bf16")
    return output

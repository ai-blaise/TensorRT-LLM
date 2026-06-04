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


class WarpDecodeStatus(str, Enum):
    DISABLED = "disabled"
    FALLBACK = "fallback"
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


def _tensor_data(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if isinstance(tensor, torch.nn.Parameter):
        return tensor.data
    return tensor


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


def _use_fixed_overlay_tactic() -> bool:
    """Whether the overlay should pin a hand-enumerated [tile, tactic] pair.

    Default is False: the overlay lets the trtllm_gen runner choose its config
    automatically (``tactic=[-1, -1]``), matching the autotune-default policy of
    the canonical WARPDECODE backend. Set ``TRTLLM_WARP_DECODE_FIXED_TACTIC=1``
    to opt into the frozen ``_NVFP4_TARGET_TACTICS`` table for the target shape.
    """
    return os.environ.get("TRTLLM_WARP_DECODE_FIXED_TACTIC", "0") == "1"


def _nvfp4_target_tactic(num_tokens: int) -> List[int]:
    bucket_tokens = _cursor_bucket_for_num_tokens(num_tokens)
    return _NVFP4_TARGET_TACTICS[bucket_tokens]


def _nvfp4_overlay_tactic(num_tokens: int) -> List[int]:
    """Tactic for the overlay runner: auto by default, fixed table only on opt-in."""
    if _use_fixed_overlay_tactic():
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
) -> torch.Tensor:
    del hidden_size, scaling_vector_size
    # Enable PDL for the decode bucket. The direct C++ runner call avoids the
    # registered custom-op dispatcher and Python TunableRunner wrapper while
    # preserving the same trtllm_gen kernels. Tactic selection defaults to auto
    # (_nvfp4_overlay_tactic) so this stays consistent with the canonical
    # WARPDECODE backend's autotune-default policy.
    os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
    runner = _nvfp4_torch_runner()
    return runner.run_moe(
        None,
        None,
        x,
        x_sf.flatten().view(torch.float8_e4m3fn),
        w13,
        w13_scale.view(torch.float8_e4m3fn),
        None,
        None,
        None,
        None,
        w2,
        w2_scale.view(torch.float8_e4m3fn),
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
        None,
    )[0]


@lru_cache(maxsize=None)
def _nvfp4_torch_runner():
    return torch.classes.trtllm.FP4BlockScaleMoERunner(_TRTLLM_GEN_SWIGLU)


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
    del hidden_size, scaling_vector_size
    outputs = torch.ops.trtllm.fp4_block_scale_moe_runner(
        None,
        None,
        x,
        x_sf.flatten().view(torch.float8_e4m3fn),
        w13,
        w13_scale.view(torch.float8_e4m3fn),
        None,
        None,
        None,
        None,
        w2,
        w2_scale.view(torch.float8_e4m3fn),
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
        None,
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
        try:
            get_cursor_warp_decode_plan(_num_tokens(x))
        except ValueError:
            return "nvfp4_batch_above_cursor_bucket"
        if not _has_nvfp4_cursor_op() and _get_nvfp4_op() is None:
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
    if w13.dim() != 3 or w2.dim() != 3:
        return "nvfp4_weight_rank_mismatch"
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
    if _is_cuda_graph(moe) and not getattr(moe, "has_nvfp4", False):
        return "cuda_graph_not_supported"
    if not do_finalize:
        return "finalize_disabled"
    if (not getattr(moe, "has_nvfp4", False)
            and getattr(moe, "layer_load_balancer", None) is not None):
        return "eplb_not_supported"
    if _num_tokens(x) == 0:
        return "empty_batch"
    if _num_tokens(x) > getattr(config, "max_batch_size", 64):
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
) -> Tuple[torch.Tensor, str]:
    output1_scale = _get_backend_tensor(moe, "fc31_scale_c")
    output1_gate_scale = _get_backend_tensor(moe, "fc31_alpha")
    output2_scale = _get_backend_tensor(moe, "fc2_alpha")
    if output1_scale is None:
        output1_scale = output1_gate_scale
    local_num_experts = _backend_int(
        moe.backend, "expert_size_per_partition", _backend_int(moe.backend, "num_slots", 0))

    cursor_op = _get_nvfp4_cursor_op()
    use_cursor = _num_tokens(x) > _NVFP4_TRTLLM_GEN_CROSSOVER_TOKENS and cursor_op is not None
    op = cursor_op if use_cursor else _get_nvfp4_op()
    if op is None:
        raise RuntimeError("NVFP4 WarpDecode op disappeared after guard check.")
    return op(
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
    ), "nvfp4_cursor_op" if use_cursor else "nvfp4_explicit_tactic_op"


def try_run_warp_decode(
    moe: "ConfigurableMoE",
    *,
    x: torch.Tensor,
    token_selected_experts: Optional[torch.Tensor],
    token_final_scales: Optional[torch.Tensor],
    x_sf: Optional[torch.Tensor],
    do_finalize: bool,
    all_rank_num_tokens: Optional[List[int]],
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
        _record(moe, WarpDecodeStatus.SELECTED, "bf16_op")
        logger.info_once("WarpDecode SELECTED (overlay): BF16 OP-compatible path.",
                         key="warp_decode_selected_bf16")
    return output

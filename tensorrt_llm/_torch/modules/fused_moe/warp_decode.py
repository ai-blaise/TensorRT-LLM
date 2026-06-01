# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-gated WarpDecode MoE fast path.

WarpDecode is an overlay on the existing MoE backend. It is intentionally
invoked at the scheduler dispatch point after routing, optional EPLB routing,
and quantization metadata have been materialized. The native scheduler remains
responsible for TP/EP, attention-DP, CUDA-graph, and disaggregated-serving
control flow.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, List, Optional

import torch

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from .configurable_moe import ConfigurableMoE


class WarpDecodeStatus(str, Enum):
    DISABLED = "disabled"
    FALLBACK = "fallback"
    SELECTED = "selected"


_BF16_SUPPORTED_BACKENDS = {"CutlassFusedMoE", "DenseGEMMFusedMoE", "TRTLLMGenFusedMoE"}
_NVFP4_SUPPORTED_BACKENDS = {"TRTLLMGenFusedMoE", "CuteDslFusedMoE"}


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
            f"WarpDecode runtime guard failed: {reason}.")
    if _policy(config) != "fallback_only":
        logger.warning_once(
            f"WarpDecode falling back to native MoE backend: {reason}.",
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
    if _is_cuda_graph(moe):
        return "cuda_graph_not_supported"
    if not do_finalize:
        return "finalize_disabled"
    if getattr(moe, "layer_load_balancer", None) is not None:
        return "eplb_not_supported"
    if _num_tokens(x) == 0:
        return "empty_batch"
    if _num_tokens(x) > getattr(config, "max_batch_size", 64):
        return "batch_too_large"

    if getattr(moe, "has_nvfp4", False):
        if moe.backend.__class__.__name__ not in _NVFP4_SUPPORTED_BACKENDS:
            return f"nvfp4_unsupported_backend_{moe.backend.__class__.__name__}"
        if x_sf is None:
            return "nvfp4_missing_activation_scales"
        if token_selected_experts is None or token_final_scales is None:
            return "routing_not_materialized"
        if _top_k(moe, token_selected_experts) != 8:
            return "top_k_not_8"
        return "nvfp4_warp_decode_kernel_missing"

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
    output = _run_bf16_warp_decode(
        moe,
        x=x,
        token_selected_experts=token_selected_experts,
        token_final_scales=token_final_scales,
    )
    _record(moe, WarpDecodeStatus.SELECTED, "bf16_op")
    logger.info_once("WarpDecode selected BF16 OP-compatible path.",
                     key="warp_decode_selected_bf16")
    return output

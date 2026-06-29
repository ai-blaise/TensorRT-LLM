#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native DeepSeek resident decode engine shim.

The production TileRT-style implementation belongs behind ``create_engine``.
The caller has already validated that the request is decode-only DeepSeek, built
TRT-LLM attention/KV metadata, and constructed a stable invocation contract.

This module intentionally creates a real resident engine object even before the
C++/CUDA body exists. The object owns per-shape scratch and exposes one narrow
native-call boundary. Until the named native op is registered, ``execute``
returns ``None`` so the caller falls back to the existing model path.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, List

from tensorrt_llm.logger import logger

_NATIVE_OP_ENV_NAME = "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_NATIVE_OP"
_VALIDATE_MANIFEST_ENV_NAME = (
    "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_VALIDATE_MANIFEST")
_STAGE_SCHEDULER_ENV_NAME = (
    "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_STAGE_SCHEDULER")
_WINDOW_STAGE_SCHEDULER_ENV_NAME = (
    "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_STAGE_SCHEDULER")
_WINDOW_STAGE_BODY_ENV_NAME = (
    "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_BODY")
_DSA_KV_DISPATCH_DENSE_NVFP4 = 0
_DSA_KV_DISPATCH_STANDARD_MLA = 1
_WINDOW_TRACE_ENV_NAME = "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_TRACE"
_WINDOW_ADMISSION_DEBUG_ENV_NAME = (
    "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_ADMISSION_DEBUG")
_NATIVE_LINEAR_BRIDGE_ENV_NAME = (
    "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_NATIVE_LINEAR_BRIDGE")
_ALLOW_SINGLE_STEP_ADP_ENV_NAME = (
    "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_ALLOW_SINGLE_STEP_ADP")
_SHARED_FP4OUT_SWIGLU_ENV_NAME = (
    "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_SHARED_FP4OUT_SWIGLU")
_WINDOW_CUDA_GRAPH_ENV_NAME = (
    "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_CUDA_GRAPH")
_WINDOW_CUDA_GRAPH_REQUIRE_REPLAY_ENV_NAME = (
    "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_CUDA_GRAPH_REQUIRE_REPLAY")
_WINDOW_CUDA_GRAPH_RUNTIME_STAGE_MAX_NUMEL = 1_000_000
_DEFAULT_NATIVE_OP = "trtllm.deepseek_resident_decode"
_SITE_ATTN_O_PROJ_WEIGHT = 15
_SITE_ATTN_GATE_PROJ_WEIGHT = 19
_SITE_ATTN_K_B_PROJ_TRANS = 23
_SITE_ATTN_K_B_PROJ_TRANS_SCALE = 24
_SITE_ATTN_K_B_PROJ_TRANS_DEQUANT = 25
_SITE_ATTN_V_B_PROJ = 26
_SITE_ATTN_V_B_PROJ_SCALE = 27
_SITE_ATTN_V_B_PROJ_DEQUANT = 28
_SITE_MOE_GATE_WEIGHT = 30
_SITE_MOE_GATE_E_SCORE_CORRECTION_BIAS = 31
_SITE_DENSE_MLP_GATE_UP_WEIGHT = 40
_SITE_DENSE_MLP_DOWN_WEIGHT = 44
_SITE_SHARED_EXPERT_GATE_UP_WEIGHT = 50
_SITE_SHARED_EXPERT_GATE_UP_WEIGHT_SCALE = 51
_SITE_SHARED_EXPERT_GATE_UP_WEIGHT_SCALE_2 = 52
_SITE_SHARED_EXPERT_GATE_UP_INPUT_SCALE = 53
_SITE_SHARED_EXPERT_DOWN_WEIGHT = 54
_SITE_SHARED_EXPERT_DOWN_WEIGHT_SCALE = 55
_SITE_SHARED_EXPERT_DOWN_WEIGHT_SCALE_2 = 56
_SITE_SHARED_EXPERT_DOWN_INPUT_SCALE = 57
_SITE_EXPERT_GATE_UP_WEIGHT = 60
_SITE_EXPERT_GATE_UP_WEIGHT_SCALE = 61
_SITE_EXPERT_GATE_UP_INPUT_SCALE = 62
_SITE_EXPERT_DOWN_WEIGHT = 63
_SITE_EXPERT_DOWN_WEIGHT_SCALE = 64
_SITE_EXPERT_DOWN_INPUT_SCALE = 65
_REQUEST_ID_PADDING_SENTINEL_UINT64 = (1 << 64) - 1
_REQUEST_ID_SIGNED_INT64_MAX = (1 << 63) - 1
_REQUEST_ID_PADDING_SENTINELS = frozenset(
    (0, -1, _REQUEST_ID_PADDING_SENTINEL_UINT64))


def _torchbind_int_list(values: Any) -> list[int]:
    result = [int(value) for value in values]
    try:
        import torch

        annotate = getattr(getattr(torch, "jit", None), "annotate", None)
        if callable(annotate):
            return annotate(List[int], result)
    except (ImportError, RuntimeError, TypeError, AttributeError):
        pass
    return result


def _torchbind_request_id_list(values: Any) -> list[int]:
    request_ids = []
    for value in values:
        request_id = int(value)
        if (request_id in _REQUEST_ID_PADDING_SENTINELS or request_id < 0
                or request_id > _REQUEST_ID_SIGNED_INT64_MAX):
            request_id = 0
        request_ids.append(request_id)
    return _torchbind_int_list(request_ids)


_SITE_ATTN_KV_A_PROJ_ALPHA = 66
_SITE_ATTN_KV_B_PROJ_ALPHA = 67
_SITE_ATTN_O_PROJ_ALPHA = 68
_SITE_ATTN_GATE_PROJ_ALPHA = 69
_SITE_DENSE_MLP_GATE_UP_ALPHA = 70
_SITE_DENSE_MLP_DOWN_ALPHA = 71
_SITE_SHARED_EXPERT_GATE_UP_ALPHA = 72
_SITE_SHARED_EXPERT_DOWN_ALPHA = 73
_SITE_EXPERT_GATE_UP_ALPHA = 74
_SITE_EXPERT_DOWN_ALPHA = 75
_SITE_ATTN_Q_A_LAYERNORM_WEIGHT = 76
_SITE_ATTN_KV_A_LAYERNORM_WEIGHT = 77
_SITE_ATTN_Q_B_PROJ_WEIGHT = 78
_SITE_ATTN_Q_B_PROJ_WEIGHT_SCALE = 79
_SITE_ATTN_Q_B_PROJ_WEIGHT_SCALE_2 = 80
_SITE_ATTN_Q_B_PROJ_INPUT_SCALE = 81
_SITE_ATTN_Q_B_PROJ_ALPHA = 82
_SITE_ATTN_INDEXER_WQ_B_WEIGHT = 83
_SITE_ATTN_INDEXER_WQ_B_WEIGHT_SCALE = 84
_SITE_ATTN_INDEXER_WQ_B_WEIGHT_SCALE_2 = 85
_SITE_ATTN_INDEXER_WQ_B_INPUT_SCALE = 86
_SITE_ATTN_INDEXER_WQ_B_ALPHA = 87
_SITE_ATTN_INDEXER_WK_WEIGHT = 88
_SITE_ATTN_INDEXER_WK_WEIGHT_SCALE = 89
_SITE_ATTN_INDEXER_WK_WEIGHT_SCALE_2 = 90
_SITE_ATTN_INDEXER_WK_INPUT_SCALE = 91
_SITE_ATTN_INDEXER_WK_ALPHA = 92
_SITE_ATTN_INDEXER_WEIGHTS_PROJ_WEIGHT = 93
_SITE_ATTN_INDEXER_WEIGHTS_PROJ_WEIGHT_SCALE = 94
_SITE_ATTN_INDEXER_WEIGHTS_PROJ_WEIGHT_SCALE_2 = 95
_SITE_ATTN_INDEXER_WEIGHTS_PROJ_INPUT_SCALE = 96
_SITE_ATTN_INDEXER_WEIGHTS_PROJ_ALPHA = 97
_SITE_ATTN_INDEXER_K_NORM_WEIGHT = 98
_SITE_ATTN_INDEXER_K_NORM_BIAS = 99
_SITE_ATTN_INDEXER_ROTARY_COS_SIN = 100
_SITE_EXPERT_GATE_UP_OUTPUT_SCALE = 101
_OPTIONAL_DSA_DISPATCH_METADATA_TENSORS = (
    "kv_lens_expanded_cuda",
    "block_table_expanded",
    "scheduler_metadata_buffer_expanded",
    "topk_indices_buffer",
    "heuristic_prev_topk",
    "heuristic_scratch_values",
)


def _exception_reason(prefix: str, exc: BaseException) -> str:
    message = str(exc).strip().splitlines()
    detail = message[0] if message else type(exc).__name__
    if len(detail) > 200:
        detail = detail[:200]
    return f"{prefix}:{type(exc).__name__}:{detail}"


def _window_trace_enabled() -> bool:
    return (_env_flag(_WINDOW_TRACE_ENV_NAME)
            or _env_flag(_WINDOW_ADMISSION_DEBUG_ENV_NAME))


def _tensor_summary(tensor: Any) -> dict[str, Any] | None:
    if tensor is None:
        return None
    is_cuda = getattr(tensor, "is_cuda", None)
    if callable(is_cuda):
        is_cuda = is_cuda()
    return {
        "shape": _tensor_shape(tensor),
        "dtype": str(getattr(tensor, "dtype", None)),
        "device": str(getattr(tensor, "device", None)),
        "is_cuda": bool(is_cuda) if is_cuda is not None else None,
    }


def _window_trace(event: str, payload: dict[str, Any] | None = None) -> None:
    if not _window_trace_enabled():
        return
    record = {"event": event}
    if payload:
        record.update(payload)
    logger.info(f"OPTRT_DEEPSEEK_RESIDENT_WINDOW_TRACE {record}")


@dataclass
class DeepSeekResidentShapeState:
    """Persistent state for one serving decode shape."""

    shape_key: tuple[Any, ...]
    batch_size: int
    hidden_size: int | None
    vocab_size: int | None
    device: Any
    dtype: Any
    scratch: dict[str, Any] = field(default_factory=dict)
    calls: int = 0


@dataclass(frozen=True)
class DeepSeekResidentTensorAsset:
    """One semantic tensor site in the resident DeepSeek model body."""

    name: str
    tensor: Any
    shape: tuple[int, ...]
    dtype: str
    device: str


@dataclass(frozen=True)
class DeepSeekResidentLayerAssets:
    """Tensor table slice for one decoder layer."""

    layer_idx: int
    layer_kind: str
    start: int
    stop: int
    tensor_names: tuple[str, ...]


@dataclass(frozen=True)
class DeepSeekResidentLayerTensorSite:
    """Semantic lookup entry for one layer tensor."""

    layer_idx: int
    site_id: int
    tensor_idx: int
    tensor_name: str


@dataclass(frozen=True)
class DeepSeekResidentModelAssets:
    """Stable resident tensor table consumed by the native decode body."""

    tensors: tuple[Any, ...]
    tensor_names: tuple[str, ...]
    tensor_specs: tuple[DeepSeekResidentTensorAsset, ...]
    layers: tuple[DeepSeekResidentLayerAssets, ...]
    layer_offsets: tuple[int, ...]
    layer_kinds: tuple[int, ...]
    layer_tensor_sites: tuple[DeepSeekResidentLayerTensorSite, ...]
    layer_site_offsets: tuple[int, ...]
    layer_site_ids: tuple[int, ...]
    layer_site_tensor_indices: tuple[int, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "num_tensors": len(self.tensors),
            "tensor_names": self.tensor_names,
            "layer_offsets": self.layer_offsets,
            "layer_kinds": self.layer_kinds,
            "layer_site_offsets": self.layer_site_offsets,
            "layer_site_ids": self.layer_site_ids,
            "layer_site_tensor_indices": self.layer_site_tensor_indices,
            "layers": [
                {
                    "layer_idx": layer.layer_idx,
                    "layer_kind": layer.layer_kind,
                    "start": layer.start,
                    "stop": layer.stop,
                    "num_tensors": layer.stop - layer.start,
                } for layer in self.layers
            ],
        }


@dataclass(frozen=True)
class DeepSeekResidentStepResult:
    """Result from the resident stage scheduler."""

    completed: bool
    reason: str
    stage: str
    layer_idx: int | None = None
    completed_layers: int = 0
    outputs: Any | None = None


@dataclass(frozen=True)
class DeepSeekResidentDsaWindowLayerPlan:
    """Static DSA dispatch contract for one layer in a native window."""

    layer_idx: int
    static_metadata_tensors: tuple[Any, ...]
    runtime_tensors: dict[str, Any]
    runtime_config: dict[str, int]
    runtime_scalars: dict[str, float]
    scratch_shapes: tuple[tuple[int, ...], ...]


@dataclass
class DeepSeekResidentWindowCudaGraphState:
    """CUDA graph state for one native resident decode-window contract."""

    key: tuple[Any, ...]
    graph: Any | None = None
    produced_tokens: Any | None = None
    static_initial_tokens: Any | None = None
    static_payload: dict[str, Any] | None = None
    warmed_direct: bool = False
    captures: int = 0
    replays: int = 0
    failures: int = 0
    last_reason: str = "not_run"


@dataclass
class DeepSeekResidentSchedulerState:
    """Native scheduler state owned across decode invocations."""

    manifest_validated: bool = False
    manifest_validation_reason: str = "not_requested"
    handle_created: bool = False
    handle_creation_reason: str = "not_attempted"
    calls: int = 0
    native_declines: int = 0
    last_execution_reason: str = "not_run"
    stage_scheduler_calls: int = 0
    stage_scheduler_declines: int = 0
    stage_scheduler_completions: int = 0
    last_stage_scheduler_reason: str = "not_run"
    embedding_stage_calls: int = 0
    input_rmsnorm_stage_calls: int = 0
    input_gated_norm_stage_calls: int = 0
    attention_dsa_proj_stage_calls: int = 0
    last_attention_dsa_proj_reason: str = "not_run"
    attention_dsa_indexer_native_wk_wp_stage_calls: int = 0
    last_attention_dsa_indexer_native_wk_wp_reason: str = "not_run"
    attention_dsa_native_dispatch_stage_calls: int = 0
    last_attention_dsa_native_dispatch_reason: str = "not_run"
    attention_dsa_attn_stage_calls: int = 0
    last_attention_dsa_attn_reason: str = "not_run"
    attention_output_tail_stage_calls: int = 0
    last_attention_output_tail_reason: str = "not_run"
    post_attention_rmsnorm_stage_calls: int = 0
    post_attention_gated_norm_stage_calls: int = 0
    moe_router_stage_calls: int = 0
    last_moe_router_reason: str = "not_run"
    dense_mlp_stage_calls: int = 0
    last_dense_mlp_reason: str = "not_run"
    post_ffn_rmsnorm_stage_calls: int = 0
    lm_head_logits_stage_calls: int = 0
    sampling_stage_calls: int = 0
    last_sampling_reason: str = "not_run"
    window_stage_calls: int = 0
    last_window_reason: str = "resident_window_native_not_implemented"
    window_prepare_stage_calls: int = 0
    last_window_prepare_reason: str = "not_run"
    window_sample_stage_calls: int = 0
    last_window_sample_reason: str = "not_run"
    window_scheduler_calls: int = 0
    window_scheduler_declines: int = 0
    window_scheduler_completions: int = 0
    last_window_scheduler_reason: str = "not_run"
    window_metadata_refresh_calls: int = 0
    last_window_metadata_refresh_reason: str = "not_run"
    window_metadata_native_device_refresh_calls: int = 0
    last_window_metadata_native_device_refresh_reason: str = "not_run"
    dsa_window_plan_builds: int = 0
    dsa_window_plan_layers: int = 0
    last_dsa_window_plan_reason: str = "not_run"
    window_cuda_graph_attempts: int = 0
    window_cuda_graph_captures: int = 0
    window_cuda_graph_replays: int = 0
    window_cuda_graph_failures: int = 0
    last_window_cuda_graph_reason: str = "not_run"


class DeepSeekResidentNativeEngine:
    """Resident engine wrapper for the native DeepSeek decode body."""

    def __init__(
        self,
        *,
        model: Any,
        contract: Any,
        dist: Any,
        native_op_name: str,
    ) -> None:
        self._model = model
        self._contract = contract
        self._dist = dist
        self._native_op_name = native_op_name
        self._assets = build_deepseek_resident_model_assets(model, contract)
        self._resident_tensors = list(self._assets.tensors)
        self._resident_layer_offsets = list(self._assets.layer_offsets)
        self._resident_layer_kinds = list(self._assets.layer_kinds)
        self._resident_layer_site_offsets = list(
            self._assets.layer_site_offsets)
        self._resident_layer_site_ids = list(self._assets.layer_site_ids)
        self._resident_layer_site_tensor_indices = list(
            self._assets.layer_site_tensor_indices)
        self._scheduler_state = DeepSeekResidentSchedulerState()
        self._shape_states: dict[tuple[Any, ...], DeepSeekResidentShapeState] = {}
        self._native_op = None
        self._native_op_checked = False
        self._native_handle = None
        self._native_handle_checked = False
        if _env_flag(_SHARED_FP4OUT_SWIGLU_ENV_NAME):
            self._ensure_cute_dsl_nvfp4_dense_gemm_swiglu_fp4out_op()

    def execute(
        self,
        *,
        request: Any,
        contract: Any,
        invocation: Any,
        inputs: dict[str, Any],
    ) -> Any | None:
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            self._scheduler_state.native_declines += 1
            self._scheduler_state.last_execution_reason = "missing_input_ids"
            return None

        if self._attention_dp_single_step_native_disabled():
            self._scheduler_state.native_declines += 1
            self._scheduler_state.last_execution_reason = (
                "resident_native_single_step_attention_dp_disabled")
            return None

        native_handle = self._get_or_create_native_handle()
        if _env_flag(_STAGE_SCHEDULER_ENV_NAME):
            state = self._get_shape_state(invocation, input_ids)
            self._build_dsa_window_plan(
                invocation=invocation,
                inputs=inputs,
                state=state,
            )
            result = self.run_decode_step_scheduler(
                input_ids=input_ids,
                invocation=invocation,
                inputs=inputs,
            )
            if result.completed:
                self._scheduler_state.last_execution_reason = result.reason
                return self._outputs_with_greedy_sample(
                    logits=result.outputs,
                    invocation=invocation,
                )
            self._scheduler_state.native_declines += 1
            self._scheduler_state.last_execution_reason = result.reason
            return None

        native_op = self._get_native_op()
        if native_op is None:
            self._scheduler_state.native_declines += 1
            self._scheduler_state.last_execution_reason = (
                "native_decode_op_not_ready")
            return None

        self._maybe_validate_manifest()
        state = self._get_shape_state(invocation, input_ids)
        self._build_dsa_window_plan(
            invocation=invocation,
            inputs=inputs,
            state=state,
        )
        state.calls += 1
        self._scheduler_state.calls += 1
        if native_handle is not None and callable(
                getattr(native_handle, "decode", None)):
            self._scheduler_state.last_execution_reason = (
                "resident_native_handle_decode_executed")
            logits = native_handle.decode(
                input_ids,
                state.scratch["hidden_states"],
                state.scratch["logits"],
                invocation.real_batch_size,
                invocation.padded_batch_size,
                invocation.input_tokens,
                _torchbind_int_list(invocation.request_ids),
                _torchbind_int_list(invocation.seq_lens),
                _torchbind_int_list(invocation.cached_tokens),
            )
            return self._outputs_with_greedy_sample(
                logits=logits,
                invocation=invocation,
            )
        self._scheduler_state.last_execution_reason = (
            "resident_native_op_decode_executed")
        logits = native_op(
            input_ids,
            state.scratch["hidden_states"],
            state.scratch["logits"],
            invocation.real_batch_size,
            invocation.padded_batch_size,
            invocation.input_tokens,
            _torchbind_int_list(invocation.request_ids),
            _torchbind_int_list(invocation.seq_lens),
            _torchbind_int_list(invocation.cached_tokens),
            self._resident_layer_offsets,
            self._resident_layer_kinds,
            self._resident_tensors,
        )
        return self._outputs_with_greedy_sample(
            logits=logits,
            invocation=invocation,
        )

    def _attention_dp_single_step_native_disabled(self) -> bool:
        if _env_flag(_ALLOW_SINGLE_STEP_ADP_ENV_NAME):
            return False
        mapping = getattr(self._dist, "mapping", None)
        return bool(getattr(mapping, "enable_attention_dp", False))

    def execution_state(self) -> dict[str, Any]:
        """Return the latest native-body execution status for executor gating."""

        return {
            "reason": self._scheduler_state.last_execution_reason,
            "stage_scheduler_reason":
            self._scheduler_state.last_stage_scheduler_reason,
            "stage_scheduler_calls":
            self._scheduler_state.stage_scheduler_calls,
            "stage_scheduler_completions":
            self._scheduler_state.stage_scheduler_completions,
            "stage_scheduler_declines":
            self._scheduler_state.stage_scheduler_declines,
            "native_decode_calls": self._scheduler_state.calls,
            "native_declines": self._scheduler_state.native_declines,
            "attention_dsa_proj_stage_calls":
            self._scheduler_state.attention_dsa_proj_stage_calls,
            "attention_dsa_proj_reason":
            self._scheduler_state.last_attention_dsa_proj_reason,
            "attention_dsa_indexer_native_wk_wp_stage_calls":
            self._scheduler_state.attention_dsa_indexer_native_wk_wp_stage_calls,
            "attention_dsa_indexer_native_wk_wp_reason":
            self._scheduler_state.last_attention_dsa_indexer_native_wk_wp_reason,
            "attention_dsa_native_dispatch_stage_calls":
            self._scheduler_state.attention_dsa_native_dispatch_stage_calls,
            "attention_dsa_native_dispatch_reason":
            self._scheduler_state.last_attention_dsa_native_dispatch_reason,
            "attention_dsa_attn_stage_calls":
            self._scheduler_state.attention_dsa_attn_stage_calls,
            "attention_dsa_attn_reason":
            self._scheduler_state.last_attention_dsa_attn_reason,
            "attention_output_tail_stage_calls":
            self._scheduler_state.attention_output_tail_stage_calls,
            "attention_output_tail_reason":
            self._scheduler_state.last_attention_output_tail_reason,
            "dense_mlp_stage_calls":
            self._scheduler_state.dense_mlp_stage_calls,
            "moe_router_stage_calls":
            self._scheduler_state.moe_router_stage_calls,
            "moe_router_reason":
            self._scheduler_state.last_moe_router_reason,
            "dense_mlp_reason":
            self._scheduler_state.last_dense_mlp_reason,
            "window_sample_stage_calls":
            self._scheduler_state.window_sample_stage_calls,
            "window_prepare_stage_calls":
            self._scheduler_state.window_prepare_stage_calls,
            "window_sample_reason":
            self._scheduler_state.last_window_sample_reason,
            "window_prepare_reason":
            self._scheduler_state.last_window_prepare_reason,
            "window_scheduler_reason":
            self._scheduler_state.last_window_scheduler_reason,
            "window_scheduler_calls":
            self._scheduler_state.window_scheduler_calls,
            "window_scheduler_completions":
            self._scheduler_state.window_scheduler_completions,
            "window_scheduler_declines":
            self._scheduler_state.window_scheduler_declines,
            "window_metadata_refresh_calls":
            self._scheduler_state.window_metadata_refresh_calls,
            "window_metadata_refresh_reason":
            self._scheduler_state.last_window_metadata_refresh_reason,
            "window_metadata_native_device_refresh_calls":
            self._scheduler_state.window_metadata_native_device_refresh_calls,
            "window_metadata_native_device_refresh_reason":
            self._scheduler_state.
            last_window_metadata_native_device_refresh_reason,
            "dsa_window_plan_builds":
            self._scheduler_state.dsa_window_plan_builds,
            "dsa_window_plan_layers":
            self._scheduler_state.dsa_window_plan_layers,
            "dsa_window_plan_reason":
            self._scheduler_state.last_dsa_window_plan_reason,
            "sample_backend_state": self.sample_backend_state(),
            "window_backend_state": self.window_backend_state(),
            "handle_created": self._scheduler_state.handle_created,
            "handle_creation_reason":
            self._scheduler_state.handle_creation_reason,
        }

    def window_backend_state(self) -> dict[str, Any]:
        """Return native multi-step window readiness for executor gating."""

        native_handle = self._get_or_create_native_handle()
        run_decode_window = (
            None if native_handle is None else
            getattr(native_handle, "run_decode_window", None))
        run_decode_window_with_plan = (
            None if native_handle is None else
            getattr(native_handle, "run_decode_window_with_dsa_plan", None))
        ready_fn = (
            None if native_handle is None else
            getattr(native_handle, "run_decode_window_ready", None))
        plan_ready_fn = (
            None if native_handle is None else
            getattr(native_handle, "run_decode_window_with_dsa_plan_ready",
                    None))
        native_window_ready = (
            callable(run_decode_window) and callable(ready_fn)
            and bool(ready_fn())) or (
                callable(run_decode_window_with_plan)
                and callable(plan_ready_fn) and bool(plan_ready_fn()))
        window_stage_body_ready = self._window_stage_body_ready(native_handle)
        window_stage_scheduler_forced = _env_flag(
            _WINDOW_STAGE_SCHEDULER_ENV_NAME)
        native_window_contract = self._native_window_contract_state(
            native_handle=native_handle,
            native_window_ready=native_window_ready,
            window_stage_body_ready=window_stage_body_ready,
            run_decode_window=run_decode_window,
            ready_fn=ready_fn,
            run_decode_window_with_plan=run_decode_window_with_plan,
            plan_ready_fn=plan_ready_fn,
        )
        if native_window_ready:
            reason = "resident_window_native_ready"
            ready = True
            backend = "deepseek_resident_window_native_v1"
        elif window_stage_body_ready:
            reason = "resident_window_stage_scheduler_ready"
            ready = True
            backend = "deepseek_resident_window_native_v1"
        elif callable(run_decode_window_with_plan) and callable(plan_ready_fn):
            reason = self._native_window_with_dsa_plan_not_ready_reason(
                native_handle)
            ready = False
            backend = "deepseek_resident_window_native_v1"
        else:
            default_not_ready_reason = (
                "resident_window_native_not_ready"
                if self._scheduler_state.last_window_reason
                == "resident_window_stage_scheduler_executed" else None)
            reason = self._native_window_not_ready_reason(
                native_handle, default_reason=default_not_ready_reason)
            ready = False
            backend = "deepseek_resident_window_native_v1"
        return {
            "backend": backend,
            "ready": ready,
            "reason": reason,
            "metadata": {
                "native": True,
                "native_window_ready":
                native_window_ready,
                "window_stage_body_ready":
                window_stage_body_ready,
                "window_stage_body_enabled":
                _window_stage_body_enabled(),
                "window_stage_scheduler_ready":
                window_stage_body_ready or window_stage_scheduler_forced,
                "window_stage_scheduler_debug_only":
                window_stage_scheduler_forced and not window_stage_body_ready,
                "window_stage_calls":
                self._scheduler_state.window_stage_calls,
                "window_sample_stage_calls":
                self._scheduler_state.window_sample_stage_calls,
                "window_prepare_stage_calls":
                self._scheduler_state.window_prepare_stage_calls,
                "window_scheduler_calls":
                self._scheduler_state.window_scheduler_calls,
                "window_scheduler_completions":
                self._scheduler_state.window_scheduler_completions,
                "window_scheduler_declines":
                self._scheduler_state.window_scheduler_declines,
                "last_window_reason":
                self._scheduler_state.last_window_reason,
                "last_window_scheduler_reason":
                self._scheduler_state.last_window_scheduler_reason,
                "last_window_prepare_reason":
                self._scheduler_state.last_window_prepare_reason,
                "last_window_sample_reason":
                self._scheduler_state.last_window_sample_reason,
                "window_metadata_refresh_calls":
                self._scheduler_state.window_metadata_refresh_calls,
                "window_metadata_native_device_refresh_calls":
                self._scheduler_state.
                window_metadata_native_device_refresh_calls,
                "last_window_metadata_native_device_refresh_reason":
                self._scheduler_state.
                last_window_metadata_native_device_refresh_reason,
                "dsa_window_plan_builds":
                self._scheduler_state.dsa_window_plan_builds,
                "dsa_window_plan_layers":
                self._scheduler_state.dsa_window_plan_layers,
                "last_dsa_window_plan_reason":
                self._scheduler_state.last_dsa_window_plan_reason,
                "window_cuda_graph_enabled":
                _env_flag(_WINDOW_CUDA_GRAPH_ENV_NAME),
                "window_cuda_graph_attempts":
                self._scheduler_state.window_cuda_graph_attempts,
                "window_cuda_graph_captures":
                self._scheduler_state.window_cuda_graph_captures,
                "window_cuda_graph_replays":
                self._scheduler_state.window_cuda_graph_replays,
                "window_cuda_graph_failures":
                self._scheduler_state.window_cuda_graph_failures,
                "last_window_cuda_graph_reason":
                self._scheduler_state.last_window_cuda_graph_reason,
                "stage_scheduler_calls":
                self._scheduler_state.stage_scheduler_calls,
                "stage_scheduler_completions":
                self._scheduler_state.stage_scheduler_completions,
                "native_window_contract":
                native_window_contract,
            },
        }

    def _native_window_contract_state(
        self,
        *,
        native_handle: Any,
        native_window_ready: bool,
        window_stage_body_ready: bool,
        run_decode_window: Any,
        ready_fn: Any,
        run_decode_window_with_plan: Any = None,
        plan_ready_fn: Any = None,
    ) -> dict[str, Any]:
        """Summarize the implementation pieces needed for a real native window."""

        layer_contracts = _as_tuple(getattr(self._contract, "layers", ()))
        dense_layers = sum(
            1 for layer in layer_contracts
            if getattr(layer, "layer_kind", None) == "dense")
        moe_layers = sum(
            1 for layer in layer_contracts
            if getattr(layer, "layer_kind", None) == "moe")
        dsa_dispatch_ready = self._native_handle_bool_method(
            native_handle, "run_layer_dsa_attention_dispatch_ready")
        dsa_window_plan_ready = (
            self._scheduler_state.last_dsa_window_plan_reason
            == "resident_dsa_window_plan_ready")
        native_moe_experts_ready = (
            moe_layers == 0 or callable(
                None if native_handle is None else getattr(
                    native_handle, "run_layer_moe_experts", None)))
        moe_expert_python_bridge_ready = (
            moe_layers == 0 or self._precomputed_moe_experts_available())
        native_moe_expert_assets_ready = (
            moe_layers == 0 or self._moe_expert_assets_not_ready_reason()
            is None)
        native_attention_metadata_refresh_ready = callable(
            None if native_handle is None else getattr(
                native_handle,
                "run_decode_window_attention_metadata_device_refresh",
                None,
            ))
        native_dsa_indexer_assets_ready = (
            self._dsa_indexer_assets_not_ready_reason() is None)
        component_ready = {
            "native_window_handle":
            native_handle is not None,
            "native_window_method":
            callable(run_decode_window) or callable(run_decode_window_with_plan),
            "native_window_ready_gate":
            callable(ready_fn) or callable(plan_ready_fn),
            "native_window_body":
            native_window_ready,
            "native_window_prepare_step":
            callable(
                None if native_handle is None else
                getattr(native_handle, "run_decode_window_prepare_step", None)),
            "native_window_sample_step":
            callable(
                None if native_handle is None else
                getattr(native_handle, "run_decode_window_sample_step", None)),
            "native_window_advance_state":
            callable(
                None if native_handle is None else
                getattr(native_handle, "run_decode_window_advance_state", None)),
            "native_dsa_attention_dispatch":
            bool(dsa_dispatch_ready),
            "native_dsa_indexer_assets":
            native_dsa_indexer_assets_ready,
            "native_moe_router":
            callable(
                None if native_handle is None else
                getattr(native_handle, "run_layer_moe_router", None)),
            "native_dense_mlp":
            callable(
                None if native_handle is None else
                getattr(native_handle, "run_layer_dense_mlp", None)),
            "native_moe_experts":
            native_moe_experts_ready,
            "native_moe_expert_assets":
            native_moe_expert_assets_ready,
            "moe_expert_python_bridge":
            moe_expert_python_bridge_ready,
            "native_attention_metadata_refresh":
            native_attention_metadata_refresh_ready,
            "native_attention_metadata_device_refresh":
            native_attention_metadata_refresh_ready,
            "native_dsa_window_plan":
            native_window_ready or dsa_window_plan_ready,
        }
        required_components = [
            "native_window_handle",
            "native_window_method",
            "native_window_ready_gate",
            "native_window_body",
            "native_window_prepare_step",
            "native_window_sample_step",
            "native_window_advance_state",
            "native_dsa_attention_dispatch",
            "native_dsa_indexer_assets",
            "native_attention_metadata_refresh",
            "native_dsa_window_plan",
        ]
        if dense_layers > 0:
            required_components.append("native_dense_mlp")
        if moe_layers > 0:
            required_components.extend((
                "native_moe_expert_assets",
                "native_moe_router",
                "native_moe_experts",
            ))
        if native_window_ready:
            missing_components: tuple[str, ...] = ()
        else:
            missing_components = tuple(
                component for component in required_components
                if not component_ready.get(component, False))
        return {
            "ready":
            native_window_ready and not missing_components,
            "first_missing_component":
            missing_components[0] if missing_components else None,
            "missing_components":
            missing_components,
            "required_components":
            tuple(required_components),
            "component_ready":
            component_ready,
            "component_reasons": {
                "native_dsa_attention_dispatch":
                None if dsa_dispatch_ready else
                self._native_dsa_attention_dispatch_not_ready_reason(
                    native_handle),
                "native_dsa_indexer_assets":
                self._dsa_indexer_assets_not_ready_reason(),
                "native_window_body":
                None if native_window_ready else
                self._native_window_not_ready_reason(native_handle),
                "native_dsa_window_plan":
                None if (native_window_ready or dsa_window_plan_ready) else
                self._scheduler_state.last_dsa_window_plan_reason,
                "native_moe_experts":
                None if native_moe_experts_ready else
                "resident_native_moe_experts_method_missing",
                "native_moe_expert_assets":
                self._moe_expert_assets_not_ready_reason(),
                "moe_expert_python_bridge":
                None if moe_expert_python_bridge_ready else
                self._precomputed_moe_experts_not_ready_reason(),
                "native_attention_metadata_refresh":
                None if native_attention_metadata_refresh_ready else
                "resident_window_metadata_native_device_refresh_missing",
            },
            "layer_counts": {
                "total": len(layer_contracts),
                "dense": dense_layers,
                "moe": moe_layers,
            },
            "asset_counts": {
                "resident_tensors": len(self._resident_tensors),
                "layer_tensor_sites": len(self._assets.layer_tensor_sites),
            },
        }

    def execute_window(
        self,
        *,
        request: Any,
        contract: Any,
        invocation: Any,
        window_contract: Any,
        inputs: dict[str, Any],
    ) -> Any | None:
        """Own a resident decode window.

        This is the native-window insertion point. It calls a handle-owned
        multi-step scheduler when that method exists and otherwise declines
        before mutating request-visible state.
        """

        self._scheduler_state.window_stage_calls += 1
        make_sample_state = getattr(request, "make_sample_state", None)
        if not callable(make_sample_state):
            self._scheduler_state.last_window_reason = (
                "resident_window_sample_state_factory_missing")
            return None
        initial_tokens = _sample_state_new_tokens(getattr(
            request, "sample_state", None))
        if initial_tokens is None:
            self._scheduler_state.last_window_reason = (
                "resident_window_initial_tokens_missing")
            return None
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            self._scheduler_state.last_window_reason = (
                "resident_window_input_ids_missing")
            return None
        position_ids = inputs.get("position_ids")
        if position_ids is None:
            self._scheduler_state.last_window_reason = (
                "resident_window_position_ids_missing")
            return None
        attn_metadata = inputs.get("attn_metadata")
        seq_lens_cuda = getattr(attn_metadata, "seq_lens_cuda", None)
        if seq_lens_cuda is None:
            self._scheduler_state.last_window_reason = (
                "resident_window_seq_lens_cuda_missing")
            return None
        kv_lens_cuda = getattr(attn_metadata, "kv_lens_cuda_runtime", None)
        if kv_lens_cuda is None:
            kv_lens_cuda = getattr(attn_metadata, "kv_lens_cuda", None)
        if kv_lens_cuda is None:
            self._scheduler_state.last_window_reason = (
                "resident_window_kv_lens_cuda_missing")
            return None
        window_stage_scheduler_forced = _env_flag(
            _WINDOW_STAGE_SCHEDULER_ENV_NAME)
        native_handle = self._get_or_create_native_handle()
        if window_stage_scheduler_forced:
            return self._execute_window_stage_scheduler(
                request=request,
                invocation=invocation,
                window_contract=window_contract,
                inputs=inputs,
                initial_tokens=initial_tokens,
                input_ids=input_ids,
                position_ids=position_ids,
                kv_lens_cuda=kv_lens_cuda,
                make_sample_state=make_sample_state,
            )
        state = self._get_shape_state(invocation, input_ids)
        dsa_window_plan = self._build_dsa_window_plan(
            invocation=invocation,
            inputs=inputs,
            state=state,
        )
        dsa_plan_window_available = (
            native_handle is not None
            and callable(
                getattr(native_handle, "run_decode_window_with_dsa_plan",
                        None))
            and callable(
                getattr(native_handle,
                        "run_decode_window_with_dsa_plan_ready", None)))
        plan_window = self._try_execute_native_window_with_dsa_plan(
            native_handle=native_handle,
            request=request,
            invocation=invocation,
            window_contract=window_contract,
            initial_tokens=initial_tokens,
            input_ids=input_ids,
            position_ids=position_ids,
            seq_lens_cuda=seq_lens_cuda,
            kv_lens_cuda=kv_lens_cuda,
            make_sample_state=make_sample_state,
            state=state,
            dsa_window_plan=dsa_window_plan,
        )
        if plan_window is not None:
            return plan_window
        if dsa_plan_window_available:
            return None
        run_decode_window = (
            None if native_handle is None else
            getattr(native_handle, "run_decode_window", None))
        if not callable(run_decode_window):
            if self._window_stage_body_ready(native_handle):
                return self._execute_window_stage_scheduler(
                    request=request,
                    invocation=invocation,
                    window_contract=window_contract,
                    inputs=inputs,
                    initial_tokens=initial_tokens,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    kv_lens_cuda=kv_lens_cuda,
                    make_sample_state=make_sample_state,
                )
            self._scheduler_state.last_window_reason = (
                self._native_window_not_ready_reason(
                    native_handle,
                    default_reason="resident_window_native_method_missing"))
            return None
        ready_fn = getattr(native_handle, "run_decode_window_ready", None)
        native_window_ready = callable(ready_fn) and bool(ready_fn())
        if not native_window_ready:
            if self._window_stage_body_ready(native_handle):
                return self._execute_window_stage_scheduler(
                    request=request,
                    invocation=invocation,
                    window_contract=window_contract,
                    inputs=inputs,
                    initial_tokens=initial_tokens,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    kv_lens_cuda=kv_lens_cuda,
                    make_sample_state=make_sample_state,
                )
            self._scheduler_state.last_window_reason = (
                self._native_window_not_ready_reason(
                    native_handle,
                    default_reason="resident_window_native_not_ready"))
            return None

        max_owned_steps = int(getattr(window_contract,
                                      "max_safe_owned_steps", 1))
        if max_owned_steps <= 1:
            self._scheduler_state.last_window_reason = (
                "resident_window_insufficient_owned_steps")
            return None
        window_tokens = self._ensure_window_new_tokens_scratch(
            state, max_owned_steps)
        if window_tokens is None:
            self._scheduler_state.last_window_reason = (
                "resident_window_token_scratch_missing")
            return None

        try:
            produced_tokens = run_decode_window(
                initial_tokens,
                window_tokens,
                input_ids,
                state.scratch["hidden_states"],
                state.scratch["logits"],
                position_ids,
                seq_lens_cuda,
                kv_lens_cuda,
                max_owned_steps,
                invocation.input_tokens,
                _torchbind_request_id_list(invocation.request_ids),
                _torchbind_int_list(invocation.seq_lens),
                _torchbind_int_list(invocation.cached_tokens),
            )
        except RuntimeError:
            self._scheduler_state.last_window_reason = (
                "resident_window_native_failed")
            return None
        if produced_tokens is None:
            self._scheduler_state.last_window_reason = (
                "resident_window_native_declined")
            return None

        sample_state = make_sample_state(
            source_sample_state=request.sample_state,
            new_tokens=produced_tokens,
            owned_steps=max_owned_steps,
            request_order=True,
        )
        self._scheduler_state.last_window_reason = (
            "resident_window_native_executed")
        return SimpleNamespace(
            sample_state=sample_state,
            owned_steps=max_owned_steps,
            requested_window_steps=int(
                getattr(window_contract, "requested_window_steps",
                        max_owned_steps)),
            break_reason="resident_window_native_executed",
            executed=True,
        )

    def _try_execute_native_window_with_dsa_plan(
        self,
        *,
        native_handle: Any,
        request: Any,
        invocation: Any,
        window_contract: Any,
        initial_tokens: Any,
        input_ids: Any,
        position_ids: Any,
        seq_lens_cuda: Any,
        kv_lens_cuda: Any,
        make_sample_state: Any,
        state: DeepSeekResidentShapeState,
        dsa_window_plan: tuple[DeepSeekResidentDsaWindowLayerPlan,
                               ...] | None,
    ) -> Any | None:
        run_decode_window = (
            None if native_handle is None else getattr(
                native_handle, "run_decode_window_with_dsa_plan", None))
        ready_fn = (
            None if native_handle is None else getattr(
                native_handle, "run_decode_window_with_dsa_plan_ready", None))
        if not callable(run_decode_window) or not callable(ready_fn):
            return None
        if not bool(ready_fn()):
            self._scheduler_state.last_window_reason = (
                self._native_window_with_dsa_plan_not_ready_reason(
                    native_handle))
            return None
        if dsa_window_plan is None:
            self._scheduler_state.last_window_reason = (
                self._scheduler_state.last_dsa_window_plan_reason)
            return None
        max_owned_steps = int(getattr(window_contract,
                                      "max_safe_owned_steps", 1))
        if max_owned_steps <= 1:
            self._scheduler_state.last_window_reason = (
                "resident_window_insufficient_owned_steps")
            return None
        window_tokens = self._ensure_window_new_tokens_scratch(
            state, max_owned_steps)
        if window_tokens is None:
            self._scheduler_state.last_window_reason = (
                "resident_window_token_scratch_missing")
            return None
        payload = self._dsa_window_plan_payload(
            state=state,
            dsa_window_plan=dsa_window_plan,
        )
        if payload is None:
            self._scheduler_state.last_window_reason = (
                "resident_window_native_plan_payload_failed:"
                f"{self._scheduler_state.last_dsa_window_plan_reason}")
            return None
        graph_tokens = self._try_execute_native_window_with_dsa_plan_cuda_graph(
            run_decode_window=run_decode_window,
            invocation=invocation,
            initial_tokens=initial_tokens,
            window_tokens=window_tokens,
            input_ids=input_ids,
            hidden_states=state.scratch["hidden_states"],
            logits=state.scratch["logits"],
            position_ids=position_ids,
            seq_lens_cuda=seq_lens_cuda,
            kv_lens_cuda=kv_lens_cuda,
            payload=payload,
            max_owned_steps=max_owned_steps,
            dsa_window_plan=dsa_window_plan,
            state=state,
        )
        if graph_tokens is not None:
            produced_tokens = graph_tokens
            window_reason = self._scheduler_state.last_window_reason
        else:
            graph_reason = self._scheduler_state.last_window_cuda_graph_reason
            graph_failed_after_entry = graph_reason.startswith(
                ("resident_window_cuda_graph_capture_failed",
                 "resident_window_cuda_graph_replay_failed"))
            if (_env_flag(_WINDOW_CUDA_GRAPH_ENV_NAME)
                    and (_env_flag(_WINDOW_CUDA_GRAPH_REQUIRE_REPLAY_ENV_NAME)
                         or graph_failed_after_entry)):
                return None
            try:
                produced_tokens = run_decode_window(
                    initial_tokens,
                    window_tokens,
                    input_ids,
                    state.scratch["hidden_states"],
                    state.scratch["logits"],
                    position_ids,
                    seq_lens_cuda,
                    kv_lens_cuda,
                    payload["layer_indices"],
                    payload["metadata_offsets"],
                    payload["metadata_tensors"],
                    payload["runtime_tensors"],
                    payload["runtime_config"],
                    payload["runtime_scalars"],
                    payload["scratch_offsets"],
                    payload["scratch_tensors"],
                    max_owned_steps,
                    invocation.input_tokens,
                    _torchbind_request_id_list(invocation.request_ids),
                    _torchbind_int_list(invocation.seq_lens),
                    _torchbind_int_list(invocation.cached_tokens),
                )
            except RuntimeError as exc:
                self._scheduler_state.last_window_reason = _exception_reason(
                    "resident_window_native_plan_failed", exc)
                return None
            window_reason = "resident_window_native_plan_executed"
        if produced_tokens is None:
            self._scheduler_state.last_window_reason = (
                "resident_window_native_plan_declined")
            return None
        sample_state = make_sample_state(
            source_sample_state=request.sample_state,
            new_tokens=produced_tokens,
            owned_steps=max_owned_steps,
            request_order=True,
        )
        self._scheduler_state.last_window_reason = window_reason
        return SimpleNamespace(
            sample_state=sample_state,
            owned_steps=max_owned_steps,
            requested_window_steps=int(
                getattr(window_contract, "requested_window_steps",
                        max_owned_steps)),
            break_reason=window_reason,
            executed=True,
        )

    def _try_execute_native_window_with_dsa_plan_cuda_graph(
        self,
        *,
        run_decode_window: Any,
        invocation: Any,
        initial_tokens: Any,
        window_tokens: Any,
        input_ids: Any,
        hidden_states: Any,
        logits: Any,
        position_ids: Any,
        seq_lens_cuda: Any,
        kv_lens_cuda: Any,
        payload: dict[str, Any],
        max_owned_steps: int,
        dsa_window_plan: tuple[DeepSeekResidentDsaWindowLayerPlan, ...],
        state: DeepSeekResidentShapeState,
    ) -> Any | None:
        if not _env_flag(_WINDOW_CUDA_GRAPH_ENV_NAME):
            return None
        self._scheduler_state.window_cuda_graph_attempts += 1
        if not bool(getattr(invocation, "cuda_graph_replay", False)):
            reason = "resident_window_cuda_graph_no_model_graph_plan"
            self._record_window_cuda_graph_reason(reason)
            return None
        torch = _import_torch()
        cuda = None if torch is None else getattr(torch, "cuda", None)
        graph_context = None if cuda is None else getattr(cuda, "graph", None)
        if (cuda is None or not callable(getattr(cuda, "CUDAGraph", None))
                or not callable(graph_context)):
            reason = "resident_window_cuda_graph_unavailable"
            self._record_window_cuda_graph_reason(reason)
            return None
        graph_key_parts = self._window_cuda_graph_key_parts(
            invocation=invocation,
            max_owned_steps=max_owned_steps,
            dsa_window_plan=dsa_window_plan,
            payload=payload,
            initial_tokens=initial_tokens,
            input_ids=input_ids,
            hidden_states=hidden_states,
            logits=logits,
            position_ids=position_ids,
            seq_lens_cuda=seq_lens_cuda,
            kv_lens_cuda=kv_lens_cuda,
            window_tokens=window_tokens,
        )
        if graph_key_parts is None:
            self._record_window_cuda_graph_reason(
                "resident_window_cuda_graph_key_failed")
            return None
        graph_key = tuple(value for _, value in graph_key_parts)

        graph_states = state.scratch.setdefault("window_cuda_graph_states", {})
        graph_state = graph_states.get(graph_key)
        if graph_state is None:
            self._trace_window_cuda_graph_key(
                state=state,
                key_parts=graph_key_parts,
                graph_state_count=len(graph_states),
            )
            graph_state = DeepSeekResidentWindowCudaGraphState(key=graph_key)
            graph_states[graph_key] = graph_state

        static_initial_tokens = self._ensure_window_cuda_graph_static_tensor(
            graph_state=graph_state,
            attr_name="static_initial_tokens",
            tensor=initial_tokens,
        )
        if static_initial_tokens is None:
            self._record_window_cuda_graph_reason(
                "resident_window_cuda_graph_static_initial_tokens_failed")
            return None
        copy_ = getattr(static_initial_tokens, "copy_", None)
        if not callable(copy_):
            self._record_window_cuda_graph_reason(
                "resident_window_cuda_graph_static_initial_tokens_not_copyable")
            return None
        copy_(initial_tokens)

        graph_payload = self._ensure_window_cuda_graph_static_payload(
            graph_state=graph_state,
            payload=payload,
        )
        if graph_payload is None:
            self._record_window_cuda_graph_reason(
                "resident_window_cuda_graph_static_payload_failed")
            return None

        if graph_state.graph is None and not graph_state.warmed_direct:
            graph_state.warmed_direct = True
            reason = "resident_window_cuda_graph_warmup_required"
            graph_state.last_reason = reason
            self._record_window_cuda_graph_reason(reason)
            return None
        if graph_state.graph is not None:
            try:
                graph_state.graph.replay()
            except RuntimeError as exc:
                graph_state.failures += 1
                self._scheduler_state.window_cuda_graph_failures += 1
                reason = _exception_reason(
                    "resident_window_cuda_graph_replay_failed", exc)
                graph_state.last_reason = reason
                self._record_window_cuda_graph_reason(reason)
                return None
            graph_state.replays += 1
            self._scheduler_state.window_cuda_graph_replays += 1
            reason = "resident_window_native_plan_cuda_graph_replayed"
            graph_state.last_reason = reason
            self._record_window_cuda_graph_reason(reason)
            self._scheduler_state.last_window_reason = reason
            return graph_state.produced_tokens

        try:
            graph = cuda.CUDAGraph()
            synchronize = getattr(cuda, "synchronize", None)
            if callable(synchronize):
                synchronize()
            with graph_context(graph):
                produced_tokens = run_decode_window(
                    static_initial_tokens,
                    window_tokens,
                    input_ids,
                    hidden_states,
                    logits,
                    position_ids,
                    seq_lens_cuda,
                    kv_lens_cuda,
                    graph_payload["layer_indices"],
                    graph_payload["metadata_offsets"],
                    graph_payload["metadata_tensors"],
                    graph_payload["runtime_tensors"],
                    graph_payload["runtime_config"],
                    graph_payload["runtime_scalars"],
                    graph_payload["scratch_offsets"],
                    graph_payload["scratch_tensors"],
                    max_owned_steps,
                    invocation.input_tokens,
                    _torchbind_request_id_list(invocation.request_ids),
                    _torchbind_int_list(invocation.seq_lens),
                    _torchbind_int_list(invocation.cached_tokens),
                )
        except RuntimeError as exc:
            graph_state.failures += 1
            self._scheduler_state.window_cuda_graph_failures += 1
            reason = _exception_reason(
                "resident_window_cuda_graph_capture_failed", exc)
            graph_state.last_reason = reason
            self._record_window_cuda_graph_reason(reason)
            return None
        graph_state.graph = graph
        graph_state.produced_tokens = produced_tokens
        graph_state.captures += 1
        self._scheduler_state.window_cuda_graph_captures += 1
        reason = "resident_window_native_plan_cuda_graph_captured"
        graph_state.last_reason = reason
        self._record_window_cuda_graph_reason(reason)
        self._scheduler_state.last_window_reason = reason
        return produced_tokens

    def _record_window_cuda_graph_reason(self, reason: str) -> None:
        self._scheduler_state.last_window_cuda_graph_reason = reason
        _window_trace("cuda_graph", {"reason": reason})

    def _ensure_window_cuda_graph_static_tensor(
        self,
        *,
        graph_state: DeepSeekResidentWindowCudaGraphState,
        attr_name: str,
        tensor: Any,
    ) -> Any | None:
        current = getattr(graph_state, attr_name)
        if current is not None and _tensor_layout_key(current) == _tensor_layout_key(
                tensor):
            return current
        empty_like = getattr(tensor, "new_empty", None)
        shape = getattr(tensor, "shape", None)
        if not callable(empty_like) or shape is None:
            return None
        try:
            static_tensor = empty_like(tuple(int(dim) for dim in shape))
        except (RuntimeError, TypeError, ValueError):
            return None
        setattr(graph_state, attr_name, static_tensor)
        return static_tensor

    def _ensure_window_cuda_graph_static_payload(
        self,
        *,
        graph_state: DeepSeekResidentWindowCudaGraphState,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        static_payload = graph_state.static_payload
        if static_payload is None:
            static_payload = {"runtime_tensors": []}
            graph_state.static_payload = static_payload

        runtime_tensors: list[dict[str, Any]] = []
        static_runtime_tensors = static_payload.setdefault(
            "runtime_tensors", [])
        source_runtime_tensors = payload["runtime_tensors"]
        if len(static_runtime_tensors) != len(source_runtime_tensors):
            static_runtime_tensors[:] = [{} for _ in source_runtime_tensors]
        for layer_idx, tensor_dict in enumerate(source_runtime_tensors):
            static_tensor_dict = static_runtime_tensors[layer_idx]
            runtime_tensor_dict = self._ensure_window_cuda_graph_static_tensor_dict(
                static_tensor_dict=static_tensor_dict,
                tensors=tensor_dict,
                copy_values=True,
            )
            if runtime_tensor_dict is None:
                return None
            runtime_tensors.append(runtime_tensor_dict)

        return {
            "layer_indices": payload["layer_indices"],
            "metadata_offsets": payload["metadata_offsets"],
            "metadata_tensors": payload["metadata_tensors"],
            "runtime_tensors": runtime_tensors,
            "runtime_config": payload["runtime_config"],
            "runtime_scalars": payload["runtime_scalars"],
            "scratch_offsets": payload["scratch_offsets"],
            "scratch_tensors": payload["scratch_tensors"],
        }

    def _ensure_window_cuda_graph_static_tensor_dict(
        self,
        *,
        static_tensor_dict: dict[str, Any],
        tensors: dict[str, Any],
        copy_values: bool,
    ) -> dict[str, Any] | None:
        stale_keys = set(static_tensor_dict) - {str(key) for key in tensors}
        for key in stale_keys:
            static_tensor_dict.pop(key, None)
        for key, tensor in tensors.items():
            name = str(key)
            if not _tensor_can_new_empty(tensor):
                static_tensor_dict[name] = tensor
                continue
            static_tensor = self._ensure_window_cuda_graph_static_payload_tensor(
                current=static_tensor_dict.get(name),
                tensor=tensor,
            )
            if static_tensor is None:
                return None
            if copy_values and not _copy_tensor_value(static_tensor, tensor):
                return None
            static_tensor_dict[name] = static_tensor
        return static_tensor_dict

    def _ensure_window_cuda_graph_static_payload_tensor(
        self,
        *,
        current: Any,
        tensor: Any,
    ) -> Any | None:
        if current is not None and _tensor_layout_key(current) == _tensor_layout_key(
                tensor):
            return current
        empty_like = getattr(tensor, "new_empty", None)
        shape = getattr(tensor, "shape", None)
        if not callable(empty_like) or shape is None:
            return None
        try:
            return empty_like(tuple(int(dim) for dim in shape))
        except (RuntimeError, TypeError, ValueError):
            return None

    def _window_cuda_graph_key(
        self,
        **kwargs: Any,
    ) -> tuple[Any, ...] | None:
        key_parts = self._window_cuda_graph_key_parts(**kwargs)
        if key_parts is None:
            return None
        return tuple(value for _, value in key_parts)

    def _window_cuda_graph_key_parts(
        self,
        *,
        invocation: Any,
        max_owned_steps: int,
        dsa_window_plan: tuple[DeepSeekResidentDsaWindowLayerPlan, ...],
        payload: dict[str, Any],
        initial_tokens: Any,
        input_ids: Any,
        hidden_states: Any,
        logits: Any,
        position_ids: Any,
        seq_lens_cuda: Any,
        kv_lens_cuda: Any,
        window_tokens: Any,
    ) -> tuple[Any, ...] | None:
        indexer_widths = _dsa_window_indexer_widths(
            cached_tokens=invocation.cached_tokens,
            input_tokens=invocation.input_tokens,
            owned_steps=max_owned_steps,
            dsa_window_plan=dsa_window_plan,
        )
        if indexer_widths is None:
            return None
        return (
            ("version", "dsa_plan_cuda_graph_v3"),
            ("stable_shape_key", invocation.stable_shape_key),
            ("max_owned_steps", int(max_owned_steps)),
            ("layer_indices",
             tuple(int(idx) for idx in payload["layer_indices"])),
            ("metadata_offsets",
             tuple(int(offset) for offset in payload["metadata_offsets"])),
            ("scratch_offsets",
             tuple(int(offset) for offset in payload["scratch_offsets"])),
            ("indexer_widths", indexer_widths),
            ("initial_tokens_layout", _tensor_layout_key(initial_tokens)),
            ("input_ids", _tensor_pointer_key(input_ids)),
            ("hidden_states", _tensor_pointer_key(hidden_states)),
            ("logits", _tensor_pointer_key(logits)),
            ("position_ids", _tensor_pointer_key(position_ids)),
            ("seq_lens_cuda", _tensor_pointer_key(seq_lens_cuda)),
            ("kv_lens_cuda", _tensor_pointer_key(kv_lens_cuda)),
            ("window_tokens", _tensor_pointer_key(window_tokens)),
            ("payload_tensors", _payload_tensor_graph_key(payload)),
        )

    def _trace_window_cuda_graph_key(
        self,
        *,
        state: DeepSeekResidentShapeState,
        key_parts: tuple[tuple[str, Any], ...],
        graph_state_count: int,
    ) -> None:
        if not _window_trace_enabled():
            return
        previous = state.scratch.get("last_window_cuda_graph_key_parts")
        if previous is None:
            changed = tuple(name for name, _ in key_parts)
        else:
            previous_by_name = dict(previous)
            changed = tuple(name for name, value in key_parts
                            if previous_by_name.get(name) != value)
        state.scratch["last_window_cuda_graph_key_parts"] = key_parts
        _window_trace(
            "cuda_graph_key", {
                "status":
                "new",
                "graph_state_count":
                int(graph_state_count),
                "changed":
                changed[:8],
                "changed_count":
                len(changed),
                "fingerprint":
                _fingerprint_value(tuple(value for _, value in key_parts)),
                "changed_fingerprints":
                tuple((name, _fingerprint_value(value))
                      for name, value in key_parts if name in changed[:8]),
            })

    def _native_window_with_dsa_plan_not_ready_reason(
        self,
        native_handle: Any,
        default_reason: str = "resident_window_native_plan_not_ready",
    ) -> str:
        reason_fn = (
            None if native_handle is None else getattr(
                native_handle,
                "run_decode_window_with_dsa_plan_not_ready_reason",
                None,
            ))
        if not callable(reason_fn):
            return default_reason
        try:
            reason = reason_fn()
        except (RuntimeError, TypeError):
            return default_reason
        return reason if isinstance(reason, str) and reason else default_reason

    def sample_backend_state(self) -> dict[str, Any]:
        """Return the resident sampler readiness exposed to the executor."""

        reason = self._scheduler_state.last_sampling_reason
        if reason not in (
                "resident_sampling_native_ready",
                "resident_sampling_native_executed",
        ):
            native_handle = self._get_or_create_native_handle()
            run_greedy_sample = (
                None if native_handle is None else
                getattr(native_handle, "run_greedy_sample", None))
            if callable(run_greedy_sample) and self._shape_states:
                reason = "resident_sampling_native_ready"

        ready = reason in (
            "resident_sampling_native_ready",
            "resident_sampling_native_executed",
        )
        return {
            "backend": "deepseek_resident_sampler_native_v1",
            "ready": ready,
            "reason": reason,
            "metadata": {
                "native": True,
                "sampling_stage_calls":
                self._scheduler_state.sampling_stage_calls,
                "last_sampling_reason":
                self._scheduler_state.last_sampling_reason,
            },
        }

    def run_decode_step_scheduler(
        self,
        *,
        input_ids: Any,
        invocation: Any,
        inputs: dict[str, Any] | None = None,
        attention_core_runner: Any | None = None,
        moe_experts_runner: Any | None = None,
        skip_input_embedding: bool = False,
    ) -> DeepSeekResidentStepResult:
        """Run one resident decode body through the currently owned stages.

        Optional runners are explicit holes for the production kernels that do
        not exist yet. Without them this method returns the first missing stage
        and leaves serving on the normal fallback path.
        """

        self._scheduler_state.stage_scheduler_calls += 1
        completed_layers = 0

        if not skip_input_embedding:
            embedded = self.run_input_embedding_stage(
                input_ids=input_ids,
                invocation=invocation,
            )
            if embedded is None:
                return self._stage_scheduler_result(
                    completed=False,
                    reason="missing_input_embedding_stage",
                    stage="input_embedding",
                    completed_layers=completed_layers,
                )

        state = self._shape_states.get(invocation.stable_shape_key)
        if state is None:
            return self._stage_scheduler_result(
                completed=False,
                reason="missing_shape_state",
                stage="shape_state",
                completed_layers=completed_layers,
            )
        state.scratch["current_residual_states"] = state.scratch[
            "hidden_states"]

        layer_contracts = _as_tuple(getattr(self._contract, "layers", ()))
        for layer_position, layer_contract in enumerate(layer_contracts):
            layer_idx = int(getattr(layer_contract, "layer_idx",
                                    completed_layers))
            is_last_layer = layer_position == len(layer_contracts) - 1
            state.scratch.pop("moe_post_ffn_finalized", None)
            if completed_layers == 0:
                normed = self.run_layer_input_rmsnorm_stage(
                    layer_idx=layer_idx,
                    invocation=invocation,
                    eps=self._rms_norm_eps(),
                )
                if normed is None:
                    return self._stage_scheduler_result(
                        completed=False,
                        reason=f"layer_{layer_idx}_input_rmsnorm_missing",
                        stage="input_rmsnorm",
                        layer_idx=layer_idx,
                        completed_layers=completed_layers,
                    )
            else:
                state.scratch["norm_hidden_states"] = state.scratch[
                    "hidden_states"]

            gated = self.run_layer_input_gated_norm_stage(
                layer_idx=layer_idx,
                invocation=invocation,
            )
            if gated is None:
                return self._stage_scheduler_result(
                    completed=False,
                    reason=f"layer_{layer_idx}_input_gated_norm_missing",
                    stage="input_gated_norm",
                    layer_idx=layer_idx,
                    completed_layers=completed_layers,
                )

            if attention_core_runner is None:
                attention_ready = self.run_layer_attention_core_stage(
                    layer_idx=layer_idx,
                    invocation=invocation,
                    inputs=inputs or {},
                )
            else:
                attention_ready = attention_core_runner(
                    engine=self,
                    state=state,
                    layer_idx=layer_idx,
                    invocation=invocation,
                )
            if attention_ready is None and attention_core_runner is None:
                return self._stage_scheduler_result(
                    completed=False,
                    reason=f"layer_{layer_idx}_attention_core_missing",
                    stage="attention_core",
                    layer_idx=layer_idx,
                    completed_layers=completed_layers,
                )
            if not attention_ready:
                return self._stage_scheduler_result(
                    completed=False,
                    reason=f"layer_{layer_idx}_attention_core_declined",
                    stage="attention_core",
                    layer_idx=layer_idx,
                    completed_layers=completed_layers,
                )

            attention_tail = self.run_layer_attention_output_tail_stage(
                layer_idx=layer_idx,
                invocation=invocation,
            )
            if attention_tail is None:
                return self._stage_scheduler_result(
                    completed=False,
                    reason=f"layer_{layer_idx}_attention_tail_missing",
                    stage="attention_tail",
                    layer_idx=layer_idx,
                    completed_layers=completed_layers,
                )

            post_attention_normed = self.run_layer_post_attention_rmsnorm_stage(
                layer_idx=layer_idx,
                invocation=invocation,
                eps=self._rms_norm_eps(),
            )
            if post_attention_normed is None:
                return self._stage_scheduler_result(
                    completed=False,
                    reason=f"layer_{layer_idx}_post_attention_rmsnorm_missing",
                    stage="post_attention_rmsnorm",
                    layer_idx=layer_idx,
                    completed_layers=completed_layers,
                )

            post_attention_gated = self.run_layer_post_attention_gated_norm_stage(
                layer_idx=layer_idx,
                invocation=invocation,
            )
            if post_attention_gated is None:
                return self._stage_scheduler_result(
                    completed=False,
                    reason=f"layer_{layer_idx}_post_attention_gate_missing",
                    stage="post_attention_gate",
                    layer_idx=layer_idx,
                    completed_layers=completed_layers,
                )

            layer_kind = getattr(layer_contract, "layer_kind", "unknown")
            if layer_kind == "dense":
                ffn_output = self.run_layer_dense_mlp_stage(
                    layer_idx=layer_idx,
                    invocation=invocation,
                )
                if ffn_output is None:
                    return self._stage_scheduler_result(
                        completed=False,
                        reason=f"layer_{layer_idx}_dense_mlp_missing",
                        stage="dense_mlp",
                        layer_idx=layer_idx,
                        completed_layers=completed_layers,
                    )
            elif layer_kind == "moe":
                router_output = self.run_layer_moe_router_stage(
                    layer_idx=layer_idx,
                    invocation=invocation,
                )
                if router_output is None:
                    return self._stage_scheduler_result(
                        completed=False,
                        reason=f"layer_{layer_idx}_moe_router_missing",
                        stage="moe_router",
                        layer_idx=layer_idx,
                        completed_layers=completed_layers,
                    )
                if moe_experts_runner is None:
                    moe_ready = self.run_layer_moe_experts_stage(
                        layer_idx=layer_idx,
                        invocation=invocation,
                        inputs=inputs or {},
                    )
                else:
                    moe_ready = moe_experts_runner(
                        engine=self,
                        state=state,
                        layer_idx=layer_idx,
                        invocation=invocation,
                        router_scratch=state.scratch["moe_router_states"]
                        [layer_idx],
                        output_scratch=state.scratch["dense_mlp_output_states"],
                    )
                if moe_ready is None and moe_experts_runner is None:
                    return self._stage_scheduler_result(
                        completed=False,
                        reason=f"layer_{layer_idx}_moe_experts_missing",
                        stage="moe_experts",
                        layer_idx=layer_idx,
                        completed_layers=completed_layers,
                    )
                if not moe_ready:
                    moe_reason = state.scratch.get("last_moe_experts_reason",
                                                   "unknown")
                    return self._stage_scheduler_result(
                        completed=False,
                        reason=(
                            f"layer_{layer_idx}_moe_experts_declined:"
                            f"{moe_reason}"),
                        stage="moe_experts",
                        layer_idx=layer_idx,
                        completed_layers=completed_layers,
                    )
            else:
                return self._stage_scheduler_result(
                    completed=False,
                    reason=f"layer_{layer_idx}_unsupported_layer_kind",
                    stage="ffn",
                    layer_idx=layer_idx,
                    completed_layers=completed_layers,
                )

            if not bool(state.scratch.pop("moe_post_ffn_finalized", False)):
                next_hidden = self.run_layer_post_ffn_rmsnorm_stage(
                    layer_idx=layer_idx,
                    invocation=invocation,
                    eps=self._rms_norm_eps(),
                )
                if next_hidden is None:
                    return self._stage_scheduler_result(
                        completed=False,
                        reason=f"layer_{layer_idx}_post_ffn_rmsnorm_missing",
                        stage="post_ffn_rmsnorm",
                        layer_idx=layer_idx,
                        completed_layers=completed_layers,
                    )
            if not is_last_layer:
                self._promote_next_layer_state(state)
            completed_layers += 1

        logits = self.run_lm_head_logits_stage(invocation=invocation)
        if logits is None:
            return self._stage_scheduler_result(
                completed=False,
                reason="lm_head_logits_missing",
                stage="lm_head_logits",
                completed_layers=completed_layers,
            )
        return self._stage_scheduler_result(
            completed=True,
            reason="resident_stage_scheduler_completed",
            stage="completed",
            completed_layers=completed_layers,
            outputs=logits,
        )

    def _execute_window_stage_scheduler(
        self,
        *,
        request: Any,
        invocation: Any,
        window_contract: Any,
        inputs: dict[str, Any],
        initial_tokens: Any,
        input_ids: Any,
        position_ids: Any,
        kv_lens_cuda: Any,
        make_sample_state: Any,
    ) -> Any | None:
        """Run a debug resident-owned multi-step window through stage methods."""

        self._scheduler_state.window_scheduler_calls += 1
        max_owned_steps = int(getattr(window_contract,
                                      "max_safe_owned_steps", 1))
        if max_owned_steps <= 1:
            return self._window_scheduler_result(
                completed=False,
                reason="resident_window_scheduler_insufficient_owned_steps",
            )
        state = self._get_shape_state(invocation, input_ids)
        if self._ensure_window_new_tokens_scratch(state,
                                                  max_owned_steps) is None:
            return self._window_scheduler_result(
                completed=False,
                reason="resident_window_scheduler_token_scratch_missing",
            )

        _window_trace(
            "scheduler_begin",
            {
                "max_owned_steps":
                max_owned_steps,
                "requested_window_steps":
                int(
                    getattr(window_contract, "requested_window_steps",
                            max_owned_steps)),
                "input_tokens":
                int(getattr(invocation, "input_tokens", 0) or 0),
                "real_batch_size":
                int(getattr(invocation, "real_batch_size", 0) or 0),
                "padded_batch_size":
                int(getattr(invocation, "padded_batch_size", 0) or 0),
                "initial_tokens":
                _tensor_summary(initial_tokens),
                "input_ids":
                _tensor_summary(input_ids),
                "position_ids":
                _tensor_summary(position_ids),
                "kv_lens_cuda":
                _tensor_summary(kv_lens_cuda),
            },
        )
        for output_step_idx in range(1, max_owned_steps):
            step_start_time = time.perf_counter()
            _window_trace("prepare_begin", {"output_step_idx": output_step_idx})
            prepared_hidden = self.run_decode_window_prepare_step_stage(
                invocation=invocation,
                initial_tokens=initial_tokens,
                input_ids=input_ids,
                position_ids=position_ids,
                kv_lens_cuda=kv_lens_cuda,
                owned_steps=max_owned_steps,
                output_step_idx=output_step_idx,
            )
            _window_trace(
                "prepare_end",
                {
                    "output_step_idx":
                    output_step_idx,
                    "elapsed_ms":
                    (time.perf_counter() - step_start_time) * 1000.0,
                    "prepared_hidden":
                    _tensor_summary(prepared_hidden),
                    "reason":
                    self._scheduler_state.last_window_prepare_reason,
                },
            )
            if prepared_hidden is None:
                reason = (
                    "resident_window_scheduler_prepare_failed:"
                    f"{self._scheduler_state.last_window_prepare_reason}")
                return self._window_scheduler_result(
                    completed=False,
                    reason=reason,
                )
            metadata_start_time = time.perf_counter()
            _window_trace("metadata_refresh_begin",
                          {"output_step_idx": output_step_idx})
            if not self._refresh_window_attention_metadata(inputs):
                _window_trace(
                    "metadata_refresh_end",
                    {
                        "output_step_idx":
                        output_step_idx,
                        "elapsed_ms":
                        (time.perf_counter() - metadata_start_time) * 1000.0,
                        "reason":
                        self._scheduler_state.last_window_metadata_refresh_reason,
                    },
                )
                return self._window_scheduler_result(
                    completed=False,
                    reason="resident_window_scheduler_metadata_refresh_failed",
                )
            _window_trace(
                "metadata_refresh_end",
                {
                    "output_step_idx":
                    output_step_idx,
                    "elapsed_ms":
                    (time.perf_counter() - metadata_start_time) * 1000.0,
                    "reason":
                    self._scheduler_state.last_window_metadata_refresh_reason,
                },
            )

            decode_start_time = time.perf_counter()
            _window_trace("decode_begin", {"output_step_idx": output_step_idx})
            step_result = self.run_decode_step_scheduler(
                input_ids=input_ids,
                invocation=invocation,
                inputs=inputs,
                skip_input_embedding=True,
            )
            _window_trace(
                "decode_end",
                {
                    "output_step_idx":
                    output_step_idx,
                    "elapsed_ms":
                    (time.perf_counter() - decode_start_time) * 1000.0,
                    "completed":
                    step_result.completed,
                    "reason":
                    step_result.reason,
                    "stage":
                    step_result.stage,
                    "layer_idx":
                    step_result.layer_idx,
                    "completed_layers":
                    step_result.completed_layers,
                },
            )
            if not step_result.completed:
                return self._window_scheduler_result(
                    completed=False,
                    reason=f"resident_window_scheduler_{step_result.reason}",
                )

            sample_start_time = time.perf_counter()
            _window_trace("sample_begin", {"output_step_idx": output_step_idx})
            sampled_tokens = self.run_decode_window_sample_step_stage(
                invocation=invocation,
                owned_steps=max_owned_steps,
                output_step_idx=output_step_idx,
            )
            _window_trace(
                "sample_end",
                {
                    "output_step_idx":
                    output_step_idx,
                    "elapsed_ms":
                    (time.perf_counter() - sample_start_time) * 1000.0,
                    "sampled_tokens":
                    _tensor_summary(sampled_tokens),
                    "reason":
                    self._scheduler_state.last_window_sample_reason,
                },
            )
            if sampled_tokens is None:
                return self._window_scheduler_result(
                    completed=False,
                    reason="resident_window_scheduler_sample_failed",
                )
            _window_trace(
                "step_end",
                {
                    "output_step_idx":
                    output_step_idx,
                    "elapsed_ms":
                    (time.perf_counter() - step_start_time) * 1000.0,
                },
            )

        window_tokens = state.scratch.get("window_new_tokens")
        sample_state = make_sample_state(
            source_sample_state=request.sample_state,
            new_tokens=window_tokens,
            owned_steps=max_owned_steps,
            request_order=True,
        )
        self._window_scheduler_result(
            completed=True,
            reason="resident_window_stage_scheduler_executed",
        )
        _window_trace(
            "scheduler_end",
            {
                "max_owned_steps": max_owned_steps,
                "window_tokens": _tensor_summary(window_tokens),
            },
        )
        return SimpleNamespace(
            sample_state=sample_state,
            owned_steps=max_owned_steps,
            requested_window_steps=int(
                getattr(window_contract, "requested_window_steps",
                        max_owned_steps)),
            break_reason="resident_window_stage_scheduler_executed",
            executed=True,
        )

    def run_layer_moe_experts_stage(
        self,
        *,
        layer_idx: int,
        invocation: Any,
        inputs: dict[str, Any],
    ) -> bool | None:
        """Bridge resident router state into the existing MoE expert backend."""

        state = self._shape_states.get(invocation.stable_shape_key)
        if state is None:
            return None
        router_scratch = state.scratch.get("moe_router_states", {}).get(
            layer_idx)
        if router_scratch is None:
            return None
        native_ready = self._try_run_native_moe_experts_stage(
            layer_idx=layer_idx,
            invocation=invocation,
            state=state,
            router_scratch=router_scratch,
        )
        if native_ready is not None:
            return native_ready
        layer = self._model_layer(layer_idx)
        moe = getattr(layer, "mlp", None)
        experts = getattr(moe, "experts", None)
        shared_experts = getattr(moe, "shared_experts", None)
        if experts is None or shared_experts is None:
            return None
        hidden_states = state.scratch["post_attention_gated_hidden_states"]
        router_logits = router_scratch["logits"]
        attn_metadata = inputs.get("attn_metadata")
        all_rank_num_tokens = getattr(attn_metadata, "all_rank_num_tokens",
                                      None)
        precomputed_ready = self._try_run_precomputed_moe_experts_stage(
            layer=layer,
            moe=moe,
            experts=experts,
            shared_experts=shared_experts,
            hidden_states=hidden_states,
            router_scratch=router_scratch,
            all_rank_num_tokens=all_rank_num_tokens,
            state=state,
        )
        if precomputed_ready is not None:
            return precomputed_ready
        if not self._moe_bridge_do_finalize(layer, moe, hidden_states):
            return self._run_deferred_moe_allreduce_bridge(
                layer=layer,
                moe=moe,
                experts=experts,
                shared_experts=shared_experts,
                hidden_states=hidden_states,
                router_logits=router_logits,
                all_rank_num_tokens=all_rank_num_tokens,
                state=state,
            )

        try:
            routed_output = experts(
                hidden_states,
                router_logits,
                do_finalize=True,
                output_dtype=getattr(hidden_states, "dtype", None),
                all_rank_num_tokens=all_rank_num_tokens,
                use_dp_padding=False,
                **_wide_ep_moe_kwargs(experts),
            )
            shared_output = shared_experts(hidden_states)
            shared_output_scale = getattr(moe, "shared_output_scale", None)
            if shared_output_scale is not None:
                shared_output *= shared_output_scale
            final_hidden_states = self._combine_moe_outputs(
                moe=moe,
                layer=layer,
                shared_output=shared_output,
                routed_output=routed_output,
            )
        except (AttributeError, RuntimeError, AssertionError, TypeError):
            state.scratch["last_moe_experts_reason"] = "moe_bridge_failed"
            return False

        output_scratch = state.scratch["dense_mlp_output_states"]
        if hasattr(output_scratch, "copy_"):
            output_scratch.copy_(final_hidden_states)
        else:
            state.scratch["dense_mlp_output_states"] = final_hidden_states
        state.scratch["last_moe_experts_reason"] = "moe_bridge_executed"
        return True

    def _try_run_native_moe_experts_stage(
        self,
        *,
        layer_idx: int,
        invocation: Any,
        state: DeepSeekResidentShapeState,
        router_scratch: dict[str, Any],
    ) -> bool | None:
        native_handle = self._get_or_create_native_handle()
        run_moe_experts = (
            None if native_handle is None else
            getattr(native_handle, "run_layer_moe_experts", None))
        if not callable(run_moe_experts):
            return None
        assets_ready = getattr(native_handle,
                               "run_layer_moe_expert_assets_ready", None)
        if callable(assets_ready) and not bool(assets_ready(layer_idx)):
            reason_fn = getattr(native_handle,
                                "run_layer_moe_expert_assets_not_ready_reason",
                                None)
            reason = (
                reason_fn(layer_idx) if callable(reason_fn) else
                "resident_native_moe_experts_assets_missing")
            state.scratch["last_moe_experts_reason"] = reason
            return None
        token_selected_experts = router_scratch.get("topk_indices")
        token_final_scales = router_scratch.get("topk_weights")
        if token_selected_experts is None or token_final_scales is None:
            state.scratch["last_moe_experts_reason"] = (
                "resident_native_moe_experts_router_scratch_missing")
            return False
        layer = self._model_layer(layer_idx)
        moe = None if layer is None else getattr(layer, "mlp", None)
        scaling_vector_size, allowed_backends, warpdecode_config = (
            self._moe_experts_native_linear_runtime(moe))
        shared_output_scale = getattr(moe, "shared_output_scale", None)
        if shared_output_scale is None:
            shared_output_scale = 1.0
        try:
            run_moe_experts(
                layer_idx,
                state.scratch["post_attention_gated_hidden_states"],
                token_selected_experts,
                token_final_scales,
                state.scratch["dense_mlp_output_states"],
                invocation.input_tokens,
                scaling_vector_size,
                str(allowed_backends),
                float(shared_output_scale),
                int(warpdecode_config["num_experts"]),
                int(warpdecode_config["local_expert_offset"]),
                int(warpdecode_config["local_num_experts"]),
                int(warpdecode_config["intermediate_size"]),
            )
        except (AttributeError, RuntimeError, AssertionError,
                TypeError) as exc:
            state.scratch["last_moe_experts_reason"] = _exception_reason(
                "resident_native_moe_experts_failed", exc)
            return None
        state.scratch["last_moe_experts_reason"] = (
            "resident_native_moe_experts_executed")
        return True

    def _moe_experts_native_linear_runtime(
        self,
        moe: Any,
    ) -> tuple[int, str, dict[str, int]]:
        candidates: list[Any] = []
        experts = None if moe is None else getattr(moe, "experts", None)
        backend = self._moe_experts_backend(moe)
        shared_experts = (
            None if moe is None else getattr(moe, "shared_experts", None))
        candidates.append(experts)
        candidates.append(backend)
        if shared_experts is not None:
            candidates.extend((
                getattr(shared_experts, "gate_up_proj", None),
                getattr(shared_experts, "down_proj", None),
            ))
        scaling_vector_size = 16
        allowed_backends = "cutlass,cublaslt,cuda_core"
        for module in candidates:
            if module is None:
                continue
            value = getattr(module, "scaling_vector_size", None)
            if value:
                try:
                    scaling_vector_size = int(value)
                    break
                except (TypeError, ValueError):
                    pass
        for module in candidates:
            if module is None:
                continue
            value = getattr(module, "nvfp4_allowed_backends_str", None)
            if value:
                allowed_backends = str(value)
                break
        return (scaling_vector_size, allowed_backends,
                self._moe_warpdecode_native_runtime_config(moe))

    @staticmethod
    def _safe_int_attr(module: Any, name: str, default: int = 0) -> int:
        value = None if module is None else getattr(module, name, None)
        try:
            return int(value if value is not None else default)
        except (TypeError, ValueError):
            return int(default)

    def _moe_experts_backend(self, moe: Any) -> Any | None:
        experts = None if moe is None else getattr(moe, "experts", None)
        backend = None if experts is None else getattr(experts, "backend", None)
        return backend if backend is not None else experts

    def _moe_warpdecode_native_runtime_config(self, moe: Any) -> dict[str, int]:
        backend = self._moe_experts_backend(moe)
        num_experts = self._safe_int_attr(
            backend, "num_experts",
            self._safe_int_attr(backend, "num_slots", 0))
        local_num_experts = self._safe_int_attr(
            backend, "expert_size_per_partition",
            self._safe_int_attr(backend, "num_slots", 0))
        intermediate_size = self._safe_int_attr(
            backend, "intermediate_size",
            self._safe_int_attr(backend, "intermediate_size_per_partition", 0))
        return {
            "num_experts": num_experts,
            "local_expert_offset": self._safe_int_attr(backend, "slot_start", 0),
            "local_num_experts": local_num_experts,
            "intermediate_size": intermediate_size,
        }

    def _try_run_precomputed_moe_experts_stage(
        self,
        *,
        layer: Any,
        moe: Any,
        experts: Any,
        shared_experts: Any,
        hidden_states: Any,
        router_scratch: dict[str, Any],
        all_rank_num_tokens: Any,
        state: DeepSeekResidentShapeState,
    ) -> bool | None:
        """Run production MoE using resident-router top-k when supported."""

        forward_precomputed_route = (
            getattr(experts, "forward_precomputed_route", None))
        if not callable(forward_precomputed_route):
            return None
        if not self._moe_bridge_do_finalize(layer, moe, hidden_states):
            return None
        token_selected_experts = router_scratch.get("topk_indices")
        token_final_scales = router_scratch.get("topk_weights")
        if token_selected_experts is None or token_final_scales is None:
            state.scratch["last_moe_experts_reason"] = (
                "moe_precomputed_route_missing")
            return False
        try:
            routed_output = forward_precomputed_route(
                hidden_states,
                token_selected_experts,
                token_final_scales,
                do_finalize=True,
                output_dtype=getattr(hidden_states, "dtype", None),
                all_rank_num_tokens=all_rank_num_tokens,
                use_dp_padding=False,
            )
            shared_output = shared_experts(hidden_states)
            shared_output_scale = getattr(moe, "shared_output_scale", None)
            if shared_output_scale is not None:
                shared_output *= shared_output_scale
            final_hidden_states = self._combine_moe_outputs(
                moe=moe,
                layer=layer,
                shared_output=shared_output,
                routed_output=routed_output,
            )
        except (AttributeError, RuntimeError, AssertionError, TypeError):
            state.scratch["last_moe_experts_reason"] = (
                "moe_precomputed_route_backend_failed")
            return False

        output_scratch = state.scratch["dense_mlp_output_states"]
        if hasattr(output_scratch, "copy_"):
            output_scratch.copy_(final_hidden_states)
        else:
            state.scratch["dense_mlp_output_states"] = final_hidden_states
        state.scratch["last_moe_experts_reason"] = (
            "moe_precomputed_route_backend_executed")
        return True

    def _refresh_window_attention_metadata(self, inputs: dict[str, Any]) -> bool:
        attn_metadata = inputs.get("attn_metadata")
        if attn_metadata is None:
            self._scheduler_state.last_window_metadata_refresh_reason = (
                "resident_window_metadata_missing")
            return False
        native_device_refreshed = (
            self._try_run_native_window_attention_metadata_device_refresh(
                attn_metadata))
        on_update_kv_lens = getattr(attn_metadata, "on_update_kv_lens", None)
        if not callable(on_update_kv_lens):
            self._scheduler_state.last_window_metadata_refresh_reason = (
                "resident_window_metadata_refresh_missing")
            return False
        try:
            on_update_kv_lens()
        except (RuntimeError, TypeError):
            self._scheduler_state.last_window_metadata_refresh_reason = (
                "resident_window_metadata_refresh_failed")
            return False
        self._scheduler_state.window_metadata_refresh_calls += 1
        self._scheduler_state.last_window_metadata_refresh_reason = (
            "resident_window_metadata_refreshed_with_native_device_update"
            if native_device_refreshed else
            "resident_window_metadata_refreshed")
        return True

    def _try_run_native_window_attention_metadata_device_refresh(
        self,
        attn_metadata: Any,
    ) -> bool:
        native_handle = self._get_or_create_native_handle()
        refresh = (
            None if native_handle is None else
            getattr(native_handle,
                    "run_decode_window_attention_metadata_device_refresh",
                    None))
        if not callable(refresh):
            self._scheduler_state.last_window_metadata_native_device_refresh_reason = (
                "resident_window_metadata_native_device_refresh_missing")
            return False
        kv_cache_manager = getattr(attn_metadata, "kv_cache_manager", None)
        if kv_cache_manager is None:
            self._scheduler_state.last_window_metadata_native_device_refresh_reason = (
                "resident_window_metadata_native_device_refresh_kv_manager_missing"
            )
            return False
        kv_lens_cuda = getattr(attn_metadata, "kv_lens_cuda_runtime", None)
        if kv_lens_cuda is None:
            kv_lens_cuda = getattr(attn_metadata, "kv_lens_cuda", None)
        required_tensors = {
            "seq_lens_cuda": getattr(attn_metadata, "seq_lens_cuda", None),
            "kv_lens_cuda": kv_lens_cuda,
            "req_idx_per_token": getattr(attn_metadata, "req_idx_per_token",
                                         None),
            "indexer_k_cache_block_offsets": getattr(
                attn_metadata, "indexer_k_cache_block_offsets", None),
            "slot_mapping_fp8": getattr(attn_metadata, "slot_mapping_fp8",
                                        None),
            "slot_mapping_scale": getattr(attn_metadata, "slot_mapping_scale",
                                          None),
            "gen_kv_indptr": getattr(attn_metadata, "gen_kv_indptr", None),
            "gen_cached_token_indptr": getattr(
                attn_metadata, "gen_cached_token_indptr", None),
            "kv_lens_cuda_2d": getattr(attn_metadata, "kv_lens_cuda_2d", None),
        }
        for name, tensor in required_tensors.items():
            if tensor is None:
                self._scheduler_state.last_window_metadata_native_device_refresh_reason = (
                    f"resident_window_metadata_native_device_refresh_{name}_missing"
                )
                return False
        try:
            num_tokens = int(getattr(attn_metadata, "num_tokens", 0) or 0)
            num_seqs = int(getattr(attn_metadata, "num_seqs", 0) or 0)
            num_contexts = int(getattr(attn_metadata, "num_contexts", 0) or 0)
            num_generations = int(
                getattr(attn_metadata, "num_generations", 0) or 0)
            index_head_dim = int(getattr(kv_cache_manager, "index_head_dim",
                                         0) or 0)
            tokens_per_block = int(
                getattr(kv_cache_manager, "tokens_per_block", 0) or 0)
            quant_block_size = int(
                getattr(kv_cache_manager, "quant_block_size", 0) or 0)
        except (TypeError, ValueError):
            self._scheduler_state.last_window_metadata_native_device_refresh_reason = (
                "resident_window_metadata_native_device_refresh_config_invalid"
            )
            return False
        if (num_tokens <= 0 or num_seqs <= 0 or index_head_dim <= 0
                or tokens_per_block <= 0 or quant_block_size <= 0):
            self._scheduler_state.last_window_metadata_native_device_refresh_reason = (
                "resident_window_metadata_native_device_refresh_config_invalid"
            )
            return False
        use_fp4 = bool(getattr(kv_cache_manager, "use_fp4", False))
        data_bytes_per_token = index_head_dim // 2 if use_fp4 else index_head_dim
        try:
            refresh(
                required_tensors["seq_lens_cuda"],
                required_tensors["kv_lens_cuda"],
                required_tensors["req_idx_per_token"],
                required_tensors["indexer_k_cache_block_offsets"],
                required_tensors["slot_mapping_fp8"],
                required_tensors["slot_mapping_scale"],
                required_tensors["gen_kv_indptr"],
                required_tensors["gen_cached_token_indptr"],
                required_tensors["kv_lens_cuda_2d"],
                num_tokens,
                num_seqs,
                num_contexts,
                num_generations,
                index_head_dim,
                tokens_per_block,
                quant_block_size,
                data_bytes_per_token,
            )
        except (AttributeError, RuntimeError, AssertionError,
                TypeError) as exc:
            self._scheduler_state.last_window_metadata_native_device_refresh_reason = (
                _exception_reason(
                    "resident_window_metadata_native_device_refresh_failed",
                    exc))
            return False
        self._scheduler_state.window_metadata_native_device_refresh_calls += 1
        self._scheduler_state.last_window_metadata_native_device_refresh_reason = (
            "resident_window_metadata_native_device_refresh_executed")
        return True

    def run_layer_attention_core_stage(
        self,
        *,
        layer_idx: int,
        invocation: Any,
        inputs: dict[str, Any],
    ) -> bool | None:
        """Run the existing DeepSeek DSA attention core into resident scratch.

        This is a bridge stage, not the final resident CUDA attention body. It
        deliberately calls the current TRT-LLM DSA core without the output gate
        and without ``o_proj`` so the resident handle remains the owner of the
        attention tail.
        """

        position_ids = inputs.get("position_ids")
        attn_metadata = inputs.get("attn_metadata")
        if position_ids is None or attn_metadata is None:
            return None
        layer = self._model_layer(layer_idx)
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is None:
            return None
        forward_impl_with_dsa = getattr(self_attn, "forward_impl_with_dsa",
                                        None)
        if not callable(forward_impl_with_dsa):
            return None
        state = self._shape_states.get(invocation.stable_shape_key)
        if state is None:
            return None
        tail_scratch = self._ensure_attention_output_tail_scratch(
            state, layer_idx)
        if tail_scratch is None:
            return None
        forward_dsa_proj = getattr(self_attn, "forward_dsa_proj", None)
        forward_dsa_attn = getattr(self_attn, "forward_dsa_attn", None)
        if callable(forward_dsa_proj) and callable(forward_dsa_attn):
            proj_outputs = self._try_run_native_dsa_attention_projection_stage(
                layer_idx=layer_idx,
                invocation=invocation,
                self_attn=self_attn,
                position_ids=position_ids,
                state=state,
            )
            if proj_outputs is None:
                try:
                    proj_outputs = forward_dsa_proj(
                        position_ids,
                        state.scratch["gated_hidden_states"],
                        attn_metadata,
                        kv_proj_input=None,
                    )
                except (AttributeError, RuntimeError, AssertionError,
                        TypeError):
                    self._scheduler_state.last_attention_dsa_proj_reason = (
                        "resident_attention_dsa_proj_failed")
                    return False
            if (not isinstance(proj_outputs, (list, tuple))
                    or len(proj_outputs) < 4):
                self._scheduler_state.last_attention_dsa_proj_reason = (
                    "resident_attention_dsa_proj_contract_failed")
                return False
            state.scratch.setdefault("attention_dsa_proj_outputs",
                                     {})[layer_idx] = tuple(proj_outputs)
            self._scheduler_state.attention_dsa_proj_stage_calls += 1
            self._scheduler_state.last_attention_dsa_proj_reason = (
                "resident_attention_dsa_proj_bridge_executed")

            native_dispatch_ready = (
                self._try_run_native_dsa_attention_dispatch_stage(
                    layer_idx=layer_idx,
                    invocation=invocation,
                    self_attn=self_attn,
                    attn_metadata=attn_metadata,
                    position_ids=position_ids,
                    proj_outputs=tuple(proj_outputs),
                    output=tail_scratch["attention_core_output"],
                ))
            if native_dispatch_ready is not None:
                return native_dispatch_ready

            try:
                forward_dsa_attn(
                    proj_outputs[0],
                    proj_outputs[1],
                    proj_outputs[2],
                    proj_outputs[3],
                    list(proj_outputs[4:]),
                    position_ids,
                    attn_metadata,
                    tail_scratch["attention_core_output"],
                )
            except (AttributeError, RuntimeError, AssertionError, TypeError):
                self._scheduler_state.last_attention_dsa_attn_reason = (
                    "resident_attention_dsa_attn_failed")
                return False
            self._scheduler_state.attention_dsa_attn_stage_calls += 1
            self._scheduler_state.last_attention_dsa_attn_reason = (
                "resident_attention_dsa_attn_bridge_executed")
            return True

        try:
            forward_impl_with_dsa(
                position_ids,
                state.scratch["gated_hidden_states"],
                attn_metadata,
                output=tail_scratch["attention_core_output"],
                kv_proj_input=None,
            )
        except (AttributeError, RuntimeError, AssertionError, TypeError):
            self._scheduler_state.last_attention_dsa_attn_reason = (
                "resident_attention_forward_impl_with_dsa_failed")
            return False
        self._scheduler_state.last_attention_dsa_attn_reason = (
            "resident_attention_forward_impl_with_dsa_bridge_executed")
        return True

    def _try_run_native_dsa_attention_projection_stage(
        self,
        *,
        layer_idx: int,
        invocation: Any,
        self_attn: Any,
        position_ids: Any,
        state: DeepSeekResidentShapeState,
    ) -> tuple[Any, ...] | None:
        native_handle = self._get_or_create_native_handle()
        run_projection = (
            None if native_handle is None else
            getattr(native_handle, "run_layer_dsa_attention_projection", None))
        if not callable(run_projection):
            return None
        scratch = self._ensure_dsa_projection_scratch(
            layer_idx=layer_idx,
            state=state,
            self_attn=self_attn,
        )
        if scratch is None:
            return None
        kv_a_proj = getattr(self_attn, "kv_a_proj_with_mqa", None)
        scaling_vector_size = int(
            getattr(kv_a_proj, "scaling_vector_size", 16) or 16)
        allowed_backends = getattr(kv_a_proj, "nvfp4_allowed_backends_str",
                                   None)
        if not allowed_backends:
            allowed_backends = "cublaslt"
        q_lora_rank = int(getattr(self_attn, "q_lora_rank", 0) or 0)
        kv_lora_rank = int(getattr(self_attn, "kv_lora_rank", 0) or 0)
        rope_dim = int(getattr(self_attn, "qk_rope_head_dim", 0) or 0)
        if q_lora_rank <= 0 or kv_lora_rank <= 0 or rope_dim <= 0:
            return None
        try:
            projection_outputs = run_projection(
                layer_idx,
                state.scratch["gated_hidden_states"],
                scratch["q"],
                scratch["q_lora"],
                scratch["compressed_kv"],
                scratch["k_pe"],
                scratch["latent_cache"],
                invocation.input_tokens,
                q_lora_rank,
                kv_lora_rank,
                rope_dim,
                self._rms_norm_eps(),
                scaling_vector_size,
                str(allowed_backends),
            )
        except (AttributeError, RuntimeError, AssertionError,
                TypeError) as exc:
            self._scheduler_state.last_attention_dsa_proj_reason = (
                _exception_reason(
                    "resident_attention_dsa_projection_native_failed", exc))
            return None
        if (not isinstance(projection_outputs, (list, tuple))
                or len(projection_outputs) < 5):
            self._scheduler_state.last_attention_dsa_proj_reason = (
                "resident_attention_dsa_projection_native_contract_failed")
            return None

        indexer_outputs = self._native_projection_indexer_outputs(
            layer_idx=layer_idx,
            state=state,
            input_tokens=invocation.input_tokens,
            self_attn=self_attn,
            position_ids=position_ids,
            hidden_states=state.scratch["gated_hidden_states"],
            q_lora=projection_outputs[4],
        )
        if indexer_outputs is None:
            return None
        self._scheduler_state.last_attention_dsa_proj_reason = (
            "resident_attention_dsa_projection_native_executed")
        return tuple(projection_outputs[:4]) + tuple(indexer_outputs)

    def _native_projection_indexer_outputs(
        self,
        *,
        layer_idx: int,
        state: DeepSeekResidentShapeState,
        input_tokens: int,
        self_attn: Any,
        position_ids: Any,
        hidden_states: Any,
        q_lora: Any,
    ) -> tuple[Any, ...] | None:
        mqa = getattr(self_attn, "mqa", None)
        indexer = None if mqa is None else getattr(mqa, "indexer", None)
        if indexer is None or bool(getattr(indexer, "skip_topk", False)):
            return None
        pre_indexer_proj = getattr(indexer, "pre_indexer_proj", None)
        if not callable(pre_indexer_proj):
            return None
        native_wk_wp = self._try_run_native_dsa_indexer_wk_weights_projection(
            layer_idx=layer_idx,
            state=state,
            indexer=indexer,
            hidden_states=hidden_states,
            input_tokens=input_tokens,
        )
        precomputed_indexer_k = None
        precomputed_weights = None
        if native_wk_wp is not None:
            precomputed_indexer_k, precomputed_weights = native_wk_wp
        try:
            outputs = pre_indexer_proj(
                q_lora,
                hidden_states,
                position_ids,
                precomputed_indexer_k=precomputed_indexer_k,
                precomputed_weights=precomputed_weights,
            )
        except (AttributeError, RuntimeError, AssertionError, TypeError):
            return None
        if not isinstance(outputs, (list, tuple)) or len(outputs) < 5:
            return None
        return tuple(outputs[:5])

    def _try_run_native_dsa_indexer_wk_weights_projection(
        self,
        *,
        layer_idx: int,
        state: DeepSeekResidentShapeState,
        indexer: Any,
        hidden_states: Any,
        input_tokens: int,
    ) -> tuple[Any, Any] | None:
        native_handle = self._get_or_create_native_handle()
        run_projection = (
            None if native_handle is None else getattr(
                native_handle,
                "run_layer_dsa_indexer_wk_weights_projection",
                None,
            ))
        if not callable(run_projection):
            self._scheduler_state.last_attention_dsa_indexer_native_wk_wp_reason = (
                "resident_attention_dsa_indexer_wk_wp_native_method_missing")
            return None
        scratch = self._ensure_dsa_indexer_wk_weights_scratch(
            layer_idx=layer_idx,
            state=state,
            indexer=indexer,
        )
        if scratch is None:
            return None
        wk = getattr(indexer, "wk", None)
        scaling_vector_size = int(getattr(wk, "scaling_vector_size", 16)
                                  or 16)
        allowed_backends = getattr(wk, "nvfp4_allowed_backends_str", None)
        if not allowed_backends:
            allowed_backends = "cublaslt"
        try:
            outputs = run_projection(
                layer_idx,
                hidden_states,
                scratch["indexer_k"],
                scratch["weights"],
                int(input_tokens),
                scaling_vector_size,
                str(allowed_backends),
            )
        except (AttributeError, RuntimeError, AssertionError,
                TypeError) as exc:
            self._scheduler_state.last_attention_dsa_indexer_native_wk_wp_reason = (
                _exception_reason(
                    "resident_attention_dsa_indexer_wk_wp_native_failed",
                    exc))
            return None
        if not isinstance(outputs, (list, tuple)) or len(outputs) < 2:
            self._scheduler_state.last_attention_dsa_indexer_native_wk_wp_reason = (
                "resident_attention_dsa_indexer_wk_wp_native_contract_failed")
            return None
        self._scheduler_state.attention_dsa_indexer_native_wk_wp_stage_calls += 1
        self._scheduler_state.last_attention_dsa_indexer_native_wk_wp_reason = (
            "resident_attention_dsa_indexer_wk_wp_native_executed")
        return outputs[0], outputs[1]

    def _ensure_dsa_indexer_wk_weights_scratch(
        self,
        *,
        layer_idx: int,
        state: DeepSeekResidentShapeState,
        indexer: Any,
    ) -> dict[str, Any] | None:
        head_dim = int(getattr(indexer, "head_dim", 0) or 0)
        n_heads = int(getattr(indexer, "n_heads", 0) or 0)
        if head_dim <= 0 or n_heads <= 0:
            self._scheduler_state.last_attention_dsa_indexer_native_wk_wp_reason = (
                "resident_attention_dsa_indexer_wk_wp_shape_invalid")
            return None
        scratch_states = state.scratch.setdefault(
            "dsa_indexer_wk_weights_states", {})
        expected_key = (state.batch_size, head_dim, n_heads)
        scratch = scratch_states.get(layer_idx)
        if scratch is not None and scratch.get("key") == expected_key:
            return scratch["tensors"]
        torch = _import_torch()
        if torch is None:
            self._scheduler_state.last_attention_dsa_indexer_native_wk_wp_reason = (
                "resident_attention_dsa_indexer_wk_wp_torch_missing")
            return None
        tensors = {
            "indexer_k":
            torch.empty(
                (state.batch_size, head_dim),
                device=state.device,
                dtype=_torch_float32_dtype(torch),
            ),
            "weights":
            torch.empty(
                (state.batch_size, n_heads),
                device=state.device,
                dtype=_torch_float32_dtype(torch),
            ),
        }
        scratch_states[layer_idx] = {
            "key": expected_key,
            "tensors": tensors,
        }
        return tensors

    def _try_run_native_dsa_attention_dispatch_stage(
        self,
        *,
        layer_idx: int,
        invocation: Any,
        self_attn: Any,
        attn_metadata: Any,
        position_ids: Any,
        proj_outputs: tuple[Any, ...],
        output: Any,
    ) -> bool | None:
        native_handle = self._get_or_create_native_handle()
        if native_handle is None:
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_native_handle_missing")
            return None
        run_dispatch = getattr(native_handle,
                               "run_layer_dsa_attention_dispatch", None)
        if not callable(run_dispatch):
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_native_method_missing")
            return None
        ready_fn = getattr(native_handle,
                           "run_layer_dsa_attention_dispatch_ready", None)
        if not callable(ready_fn) or not bool(ready_fn()):
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                self._native_dsa_attention_dispatch_not_ready_reason(
                    native_handle))
            return None
        mqa = getattr(self_attn, "mqa", None)
        kv_cache_decline_reason = self._dsa_native_kv_cache_decline_reason(
            mqa=mqa, attn_metadata=attn_metadata)
        if kv_cache_decline_reason is not None:
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                kv_cache_decline_reason)
            return None
        if not self._restore_kvarn_for_native_dsa_dispatch(
                mqa=mqa, attn_metadata=attn_metadata):
            return False
        seq_lens_cuda = getattr(attn_metadata, "seq_lens_cuda", None)
        if seq_lens_cuda is None:
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_seq_lens_cuda_missing")
            return False
        kv_lens_cuda = getattr(attn_metadata, "kv_lens_cuda_runtime", None)
        if kv_lens_cuda is None:
            kv_lens_cuda = getattr(attn_metadata, "kv_lens_cuda", None)
        if kv_lens_cuda is None:
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_kv_lens_cuda_missing")
            return False
        try:
            topk_indices = self._dsa_dispatch_topk_indices(
                self_attn=self_attn,
                attn_metadata=attn_metadata,
                proj_outputs=proj_outputs,
                input_tokens=invocation.input_tokens,
            )
            if topk_indices is None:
                return False
            dispatch_metadata_tensors = (
                self._dsa_dispatch_metadata_tensors(
                    layer_idx, attn_metadata, topk_indices=topk_indices))
            if dispatch_metadata_tensors is None:
                return False
            dispatch_scratch_tensors = self._ensure_dsa_dispatch_scratch(
                layer_idx=layer_idx,
                state=self._shape_states.get(invocation.stable_shape_key),
                attn_metadata=attn_metadata,
                proj_outputs=proj_outputs,
                output=output,
                input_tokens=invocation.input_tokens,
            )
            if dispatch_scratch_tensors is None:
                return False
            runtime_descriptor = self._dsa_dispatch_runtime_descriptor(
                layer_idx=layer_idx,
                self_attn=self_attn,
                attn_metadata=attn_metadata,
                topk_indices=topk_indices,
                input_tokens=invocation.input_tokens,
            )
            if runtime_descriptor is None:
                return False
            (dispatch_runtime_tensors, dispatch_runtime_config,
             dispatch_runtime_scalars) = runtime_descriptor
            run_dispatch(
                layer_idx,
                proj_outputs[0],
                proj_outputs[1],
                proj_outputs[2],
                proj_outputs[3],
                list(proj_outputs[4:]),
                position_ids,
                seq_lens_cuda,
                kv_lens_cuda,
                dispatch_metadata_tensors,
                dispatch_runtime_tensors,
                dispatch_runtime_config,
                dispatch_runtime_scalars,
                dispatch_scratch_tensors,
                output,
                invocation.input_tokens,
            )
        except (AttributeError, RuntimeError, AssertionError,
                TypeError) as exc:
            reason = _exception_reason(
                "resident_attention_dsa_dispatch_native_failed", exc)
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                reason)
            _window_trace(
                "dsa_attention_dispatch_native_failed", {
                    "layer_idx": layer_idx,
                    "reason": reason,
                    "input_tokens": invocation.input_tokens,
                })
            return False
        self._scheduler_state.attention_dsa_native_dispatch_stage_calls += 1
        self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
            "resident_attention_dsa_dispatch_native_executed")
        self._scheduler_state.attention_dsa_attn_stage_calls += 1
        self._scheduler_state.last_attention_dsa_attn_reason = (
            "resident_attention_dsa_attn_native_executed")
        return True

    def _build_dsa_window_plan(
        self,
        *,
        invocation: Any,
        inputs: dict[str, Any],
        state: DeepSeekResidentShapeState,
    ) -> tuple[DeepSeekResidentDsaWindowLayerPlan, ...] | None:
        attn_metadata = inputs.get("attn_metadata")
        if attn_metadata is None:
            return self._dsa_window_plan_decline(
                "resident_dsa_window_plan_attn_metadata_missing")
        native_handle = self._get_or_create_native_handle()
        if native_handle is None:
            return self._dsa_window_plan_decline(
                "resident_dsa_window_plan_native_handle_missing")
        run_dispatch = getattr(native_handle,
                               "run_layer_dsa_attention_dispatch", None)
        if not callable(run_dispatch):
            return self._dsa_window_plan_decline(
                "resident_dsa_window_plan_native_dispatch_method_missing")
        dispatch_ready = self._native_handle_bool_method(
            native_handle, "run_layer_dsa_attention_dispatch_ready")
        if not dispatch_ready:
            return self._dsa_window_plan_decline(
                self._native_dsa_attention_dispatch_not_ready_reason(
                    native_handle))

        layer_contracts = _as_tuple(getattr(self._contract, "layers", ()))
        if not layer_contracts:
            return self._dsa_window_plan_decline(
                "resident_dsa_window_plan_layers_missing")
        if not self._ensure_cute_dsl_fp4_paged_mqa_logits_op():
            return self._dsa_window_plan_decline(
                self._scheduler_state.last_dsa_window_plan_reason)

        plan: list[DeepSeekResidentDsaWindowLayerPlan] = []
        for layer_contract in layer_contracts:
            try:
                layer_idx = int(getattr(layer_contract, "layer_idx"))
            except (TypeError, ValueError):
                return self._dsa_window_plan_decline(
                    "resident_dsa_window_plan_layer_idx_invalid")
            layer = self._model_layer(layer_idx)
            self_attn = None if layer is None else getattr(layer, "self_attn",
                                                           None)
            mqa = None if self_attn is None else getattr(self_attn, "mqa", None)
            if mqa is None:
                return self._dsa_window_plan_decline(
                    f"resident_dsa_window_plan_layer_{layer_idx}_mqa_missing")
            kv_cache_decline_reason = (
                self._dsa_native_kv_cache_decline_reason(
                    mqa=mqa, attn_metadata=attn_metadata))
            if kv_cache_decline_reason is not None:
                return self._dsa_window_plan_decline(
                    kv_cache_decline_reason.replace(
                        "resident_attention_dsa_dispatch",
                        f"resident_dsa_window_plan_layer_{layer_idx}"))

            static_metadata_tensors = (
                self._dsa_dispatch_static_metadata_tensors(
                    layer_idx, attn_metadata))
            if static_metadata_tensors is None:
                return self._dsa_window_plan_decline(
                    self._last_dsa_dispatch_reason())
            runtime_descriptor = self._dsa_dispatch_runtime_descriptor(
                layer_idx=layer_idx,
                self_attn=self_attn,
                attn_metadata=attn_metadata,
                topk_indices=None,
                input_tokens=invocation.input_tokens,
            )
            if runtime_descriptor is None:
                return self._dsa_window_plan_decline(
                    self._last_dsa_dispatch_reason())
            scratch_shapes = self._dsa_window_plan_scratch_shapes(
                layer_idx=layer_idx,
                state=state,
                self_attn=self_attn,
                attn_metadata=attn_metadata,
                input_tokens=invocation.input_tokens,
            )
            if scratch_shapes is None:
                return self._dsa_window_plan_decline(
                    self._last_dsa_dispatch_reason())
            runtime_tensors, runtime_config, runtime_scalars = runtime_descriptor
            runtime_config = dict(runtime_config)
            runtime_scalars = dict(runtime_scalars)
            runtime_scalars["resident_rms_norm_eps"] = self._rms_norm_eps()
            runtime_config["resident_attention_sf_vec_size"] = (
                self._attention_native_sf_vec_size(self_attn))
            runtime_config["resident_mlp_sf_vec_size"] = (
                self._layer_mlp_native_sf_vec_size(layer))
            if getattr(layer_contract, "layer_kind", None) == "moe":
                router_config = self._moe_router_config(layer_idx)
                if router_config is None:
                    return self._dsa_window_plan_decline(
                        f"resident_dsa_window_plan_layer_{layer_idx}_moe_router_config_missing"
                    )
                runtime_config.update({
                    "resident_moe_top_k":
                    int(router_config["top_k"]),
                    "resident_moe_n_group":
                    int(router_config["n_group"]),
                    "resident_moe_topk_group":
                    int(router_config["topk_group"]),
                })
                runtime_scalars[
                    "resident_moe_routed_scaling_factor"] = float(
                        router_config["routed_scaling_factor"])
                moe = None if layer is None else getattr(layer, "mlp", None)
                warpdecode_config = self._moe_warpdecode_native_runtime_config(
                    moe)
                runtime_config.update({
                    "resident_moe_num_experts":
                    int(warpdecode_config["num_experts"]),
                    "resident_moe_local_expert_offset":
                    int(warpdecode_config["local_expert_offset"]),
                    "resident_moe_local_num_experts":
                    int(warpdecode_config["local_num_experts"]),
                    "resident_moe_intermediate_size":
                    int(warpdecode_config["intermediate_size"]),
                })
                shared_output_scale = getattr(moe, "shared_output_scale", None)
                runtime_scalars["resident_moe_shared_output_scale"] = (
                    1.0 if shared_output_scale is None else
                    float(shared_output_scale))
            plan.append(
                DeepSeekResidentDsaWindowLayerPlan(
                    layer_idx=layer_idx,
                    static_metadata_tensors=tuple(static_metadata_tensors),
                    runtime_tensors=runtime_tensors,
                    runtime_config=runtime_config,
                    runtime_scalars=runtime_scalars,
                    scratch_shapes=scratch_shapes,
                ))

        plan_tuple = tuple(plan)
        state.scratch["dsa_window_plan"] = plan_tuple
        self._scheduler_state.dsa_window_plan_builds += 1
        self._scheduler_state.dsa_window_plan_layers = len(plan_tuple)
        self._scheduler_state.last_dsa_window_plan_reason = (
            "resident_dsa_window_plan_ready")
        _window_trace("dsa_window_plan_ready", {
            "layers": len(plan_tuple),
        })
        return plan_tuple

    def _dsa_window_plan_decline(
        self,
        reason: str,
    ) -> None:
        self._scheduler_state.dsa_window_plan_layers = 0
        self._scheduler_state.last_dsa_window_plan_reason = reason
        _window_trace("dsa_window_plan_decline", {
            "reason": reason,
        })
        return None

    def _ensure_cute_dsl_fp4_paged_mqa_logits_op(self) -> bool:
        """Ensure the Python custom op schema exists before C++ dispatch."""

        torch = _import_torch()
        trtllm_ops = None if torch is None else getattr(
            getattr(torch, "ops", None), "trtllm", None)
        if hasattr(trtllm_ops, "cute_dsl_fp4_paged_mqa_logits"):
            return True
        try:
            from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops  # noqa: F401
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
            self._scheduler_state.last_dsa_window_plan_reason = (
                "resident_dsa_window_plan_cute_dsl_fp4_paged_mqa_logits_missing")
            return False
        torch = _import_torch()
        trtllm_ops = None if torch is None else getattr(
            getattr(torch, "ops", None), "trtllm", None)
        if not hasattr(trtllm_ops, "cute_dsl_fp4_paged_mqa_logits"):
            self._scheduler_state.last_dsa_window_plan_reason = (
                "resident_dsa_window_plan_cute_dsl_fp4_paged_mqa_logits_missing")
            return False
        return True

    def _ensure_cute_dsl_nvfp4_dense_gemm_swiglu_fp4out_op(self) -> bool:
        """Ensure the optional fused shared-expert op is registered."""

        op_name = "cute_dsl_nvfp4_dense_gemm_swiglu_fp4out_blackwell"
        torch = _import_torch()
        trtllm_ops = None if torch is None else getattr(
            getattr(torch, "ops", None), "trtllm", None)
        if hasattr(trtllm_ops, op_name):
            return True
        try:
            from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops  # noqa: F401
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError
                ) as exc:
            logger.warning(
                "OPTRT_DEEPSEEK_RESIDENT_SHARED_FP4OUT_SWIGLU unavailable: "
                f"{type(exc).__name__}: {exc}")
            return False
        torch = _import_torch()
        trtllm_ops = None if torch is None else getattr(
            getattr(torch, "ops", None), "trtllm", None)
        if not hasattr(trtllm_ops, op_name):
            logger.warning(
                "OPTRT_DEEPSEEK_RESIDENT_SHARED_FP4OUT_SWIGLU unavailable: "
                "custom op schema missing")
            return False
        logger.info("OPTRT_DEEPSEEK_RESIDENT_SHARED_FP4OUT_SWIGLU registered")
        return True

    def _dsa_window_plan_payload(
        self,
        *,
        state: DeepSeekResidentShapeState,
        dsa_window_plan: tuple[DeepSeekResidentDsaWindowLayerPlan, ...],
    ) -> dict[str, Any] | None:
        """Flatten per-layer DSA plan data for the native window ABI."""

        layer_indices: list[int] = []
        metadata_offsets: list[int] = [0]
        metadata_tensors: list[Any] = []
        runtime_tensors: list[dict[str, Any]] = []
        runtime_config: list[dict[str, int]] = []
        runtime_scalars: list[dict[str, float]] = []
        scratch_offsets: list[int] = [0]
        scratch_tensors: list[Any] = []

        for layer_plan in dsa_window_plan:
            layer_indices.append(int(layer_plan.layer_idx))
            metadata_tensors.extend(layer_plan.static_metadata_tensors)
            metadata_offsets.append(len(metadata_tensors))
            runtime_tensors.append(dict(layer_plan.runtime_tensors))
            runtime_config.append(dict(layer_plan.runtime_config))
            runtime_scalars.append(dict(layer_plan.runtime_scalars))
            layer_scratch = self._ensure_dsa_window_plan_scratch_tensors(
                state=state,
                layer_plan=layer_plan,
            )
            if layer_scratch is None:
                return None
            scratch_tensors.extend(layer_scratch)
            scratch_offsets.append(len(scratch_tensors))

        return {
            "layer_indices": layer_indices,
            "metadata_offsets": metadata_offsets,
            "metadata_tensors": metadata_tensors,
            "runtime_tensors": runtime_tensors,
            "runtime_config": runtime_config,
            "runtime_scalars": runtime_scalars,
            "scratch_offsets": scratch_offsets,
            "scratch_tensors": scratch_tensors,
        }

    def _attention_native_sf_vec_size(self, self_attn: Any) -> int:
        kv_a_proj = getattr(self_attn, "kv_a_proj_with_mqa", None)
        value = getattr(kv_a_proj, "scaling_vector_size", None)
        try:
            return int(value or 16)
        except (TypeError, ValueError):
            return 16

    def _layer_mlp_native_sf_vec_size(self, layer: Any) -> int:
        mlp = None if layer is None else getattr(layer, "mlp", None)
        candidates = [
            getattr(mlp, "gate_up_proj", None),
            getattr(mlp, "down_proj", None),
        ]
        experts = getattr(mlp, "experts", None)
        backend = self._moe_experts_backend(mlp)
        shared_experts = getattr(mlp, "shared_experts", None)
        candidates.extend((
            experts,
            backend,
            getattr(shared_experts, "gate_up_proj", None),
            getattr(shared_experts, "down_proj", None),
        ))
        for module in candidates:
            value = getattr(module, "scaling_vector_size", None)
            if value:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
        return 16

    def _ensure_dsa_window_plan_scratch_tensors(
        self,
        *,
        state: DeepSeekResidentShapeState,
        layer_plan: DeepSeekResidentDsaWindowLayerPlan,
    ) -> list[Any] | None:
        torch = _import_torch()
        if torch is None:
            self._scheduler_state.last_dsa_window_plan_reason = (
                "resident_dsa_window_plan_torch_missing")
            return None
        scratch_states = state.scratch.setdefault(
            "dsa_window_plan_scratch_states", {})
        scratch_shapes = tuple(tuple(int(dim) for dim in shape)
                               for shape in layer_plan.scratch_shapes)
        expected_key = (
            int(layer_plan.layer_idx),
            scratch_shapes,
            str(state.device),
            str(state.dtype),
        )
        cached = scratch_states.get(int(layer_plan.layer_idx))
        if cached is not None and cached.get("key") == expected_key:
            return cached["tensors"]

        kv_dispatch_mode = int(
            layer_plan.runtime_config.get("kv_dispatch_mode",
                                          _DSA_KV_DISPATCH_DENSE_NVFP4))
        indexer_scratch_start = (
            8 if kv_dispatch_mode == _DSA_KV_DISPATCH_STANDARD_MLA else 5)
        tensors: list[Any] = []
        for idx, shape in enumerate(scratch_shapes):
            if idx in (0, 1):
                dtype = state.dtype
            elif idx in (2, 3):
                dtype = _torch_int32_dtype(torch)
            elif idx == 4:
                dtype = _torch_uint32_dtype(torch)
            elif (kv_dispatch_mode == _DSA_KV_DISPATCH_STANDARD_MLA
                  and idx in (5, 6)):
                dtype = _torch_float32_dtype(torch)
            elif kv_dispatch_mode == _DSA_KV_DISPATCH_STANDARD_MLA and idx == 7:
                dtype = _torch_uint8_dtype(torch)
            elif idx in (indexer_scratch_start, indexer_scratch_start + 1):
                dtype = _torch_uint8_dtype(torch)
            elif idx in (indexer_scratch_start + 2, indexer_scratch_start + 4,
                         indexer_scratch_start + 5,
                         indexer_scratch_start + 6):
                dtype = _torch_int32_dtype(torch)
            elif idx == indexer_scratch_start + 3:
                dtype = _torch_float32_dtype(torch)
            else:
                dtype = state.dtype
            try:
                tensors.append(
                    torch.empty(shape, device=state.device, dtype=dtype))
            except (AttributeError, RuntimeError, TypeError, ValueError):
                self._scheduler_state.last_dsa_window_plan_reason = (
                    "resident_dsa_window_plan_scratch_alloc_failed")
                return None

        scratch_states[int(layer_plan.layer_idx)] = {
            "key": expected_key,
            "tensors": tensors,
        }
        return tensors

    def _dsa_native_kv_cache_decline_reason(
        self,
        *,
        mqa: Any,
        attn_metadata: Any,
    ) -> str | None:
        if bool(getattr(mqa, "has_fp4_kv_cache", False)):
            return None
        quant_mode = getattr(mqa, "quant_mode", None)
        has_fp4_kv_cache = getattr(quant_mode, "has_fp4_kv_cache", None)
        if callable(has_fp4_kv_cache) and bool(has_fp4_kv_cache()):
            return None

        kv_cache_manager = getattr(attn_metadata, "kv_cache_manager", None)
        if kv_cache_manager is None:
            return "resident_attention_dsa_dispatch_non_nvfp4_kv_cache"
        dtype = getattr(kv_cache_manager, "dtype", None)
        dtype_name = getattr(dtype, "name", None)
        if dtype_name == "NVFP4" or str(dtype).endswith("NVFP4"):
            return None
        if (bool(getattr(kv_cache_manager, "kvarn_enabled", False))
                or getattr(kv_cache_manager, "kvarn_cfg", None) is not None):
            restore = getattr(mqa, "kvarn_restore_for_decode", None)
            if callable(restore):
                return None
            return "resident_attention_dsa_dispatch_kvarn_restore_missing"
        return "resident_attention_dsa_dispatch_non_nvfp4_kv_cache"

    def _restore_kvarn_for_native_dsa_dispatch(
        self,
        *,
        mqa: Any,
        attn_metadata: Any,
    ) -> bool:
        kv_cache_manager = getattr(attn_metadata, "kv_cache_manager", None)
        if not (bool(getattr(kv_cache_manager, "kvarn_enabled", False))
                or getattr(kv_cache_manager, "kvarn_cfg", None) is not None):
            return True
        restore = getattr(mqa, "kvarn_restore_for_decode", None)
        if not callable(restore):
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_kvarn_restore_missing")
            return False
        try:
            restore(attn_metadata)
        except (AttributeError, RuntimeError, AssertionError, TypeError):
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_kvarn_restore_failed")
            return False
        return True

    def _last_dsa_dispatch_reason(self) -> str:
        return self._scheduler_state.last_attention_dsa_native_dispatch_reason

    def _dsa_dispatch_topk_indices(
        self,
        *,
        self_attn: Any,
        attn_metadata: Any,
        proj_outputs: tuple[Any, ...],
        input_tokens: int,
    ) -> Any | None:
        if len(proj_outputs) < 9:
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_indexer_outputs_missing")
            return None
        mqa = getattr(self_attn, "mqa", None)
        indexer = None if mqa is None else getattr(mqa, "indexer", None)
        sparse_attn_indexer = (
            None if indexer is None else
            getattr(indexer, "sparse_attn_indexer", None))
        if not callable(sparse_attn_indexer):
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_indexer_missing")
            return None
        num_tokens = int(getattr(attn_metadata, "num_tokens", input_tokens))
        if num_tokens <= 0 or num_tokens > int(input_tokens):
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_num_tokens_invalid")
            return None
        try:
            q = _slice_first_dim(proj_outputs[0], num_tokens)
            q_fp8 = _slice_first_dim(proj_outputs[4], num_tokens)
            k_fp8 = _slice_first_dim(proj_outputs[5], num_tokens)
            k_scale = _slice_first_dim(proj_outputs[6], num_tokens)
            weights = _slice_first_dim(proj_outputs[7], num_tokens)
            q_scale = _slice_first_dim(proj_outputs[8], num_tokens)
            topk_indices = sparse_attn_indexer(
                attn_metadata,
                q,
                q_fp8,
                k_fp8,
                k_scale,
                weights,
                q_scale=q_scale,
            )
        except (AttributeError, RuntimeError, AssertionError, TypeError,
                IndexError):
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_topk_failed")
            return None
        if topk_indices is None:
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_topk_missing")
            return None
        return topk_indices

    def _dsa_dispatch_metadata_tensors(
        self,
        layer_idx: int,
        attn_metadata: Any,
        *,
        topk_indices: Any,
    ) -> list[Any] | None:
        if topk_indices is None:
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_topk_indices_missing")
            return None
        static_tensors = self._dsa_dispatch_static_metadata_tensors(
            layer_idx, attn_metadata)
        if static_tensors is None:
            return None
        return [topk_indices, *static_tensors]

    def _dsa_dispatch_static_metadata_tensors(
        self,
        layer_idx: int,
        attn_metadata: Any,
    ) -> list[Any] | None:
        tensors: list[Any] = []

        def append_required(name: str, value: Any) -> bool:
            if value is None:
                self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                    f"resident_attention_dsa_dispatch_{name}_missing")
                return False
            tensors.append(value)
            return True

        block_table = getattr(attn_metadata, "block_table", None)
        if not append_required("block_table", block_table):
            return None
        kv_cache_manager = getattr(attn_metadata, "kv_cache_manager", None)
        get_indexer_k_cache = (
            None if kv_cache_manager is None else
            getattr(kv_cache_manager, "get_indexer_k_cache_buffers", None))
        if not callable(get_indexer_k_cache):
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_indexer_k_cache_missing")
            return None
        try:
            indexer_k_cache = get_indexer_k_cache(layer_idx)
        except (AttributeError, IndexError, KeyError, RuntimeError, TypeError):
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_indexer_k_cache_missing")
            return None
        if not append_required("indexer_k_cache", indexer_k_cache):
            return None
        indexer_k_cache_block_offsets = getattr(
            attn_metadata, "indexer_k_cache_block_offsets", None)
        if not append_required("indexer_k_cache_block_offsets",
                               indexer_k_cache_block_offsets):
            return None
        scheduler_metadata_buffer = getattr(attn_metadata,
                                            "scheduler_metadata_buffer", None)
        if not append_required("scheduler_metadata_buffer",
                               scheduler_metadata_buffer):
            return None
        slot_mapping_fp8 = getattr(attn_metadata, "slot_mapping_fp8", None)
        if not append_required("slot_mapping_fp8", slot_mapping_fp8):
            return None
        slot_mapping_scale = getattr(attn_metadata, "slot_mapping_scale", None)
        if not append_required("slot_mapping_scale", slot_mapping_scale):
            return None
        gen_kv_indptr = getattr(attn_metadata, "gen_kv_indptr", None)
        if not append_required("gen_kv_indptr", gen_kv_indptr):
            return None
        gen_cached_token_indptr = getattr(attn_metadata,
                                          "gen_cached_token_indptr", None)
        if not append_required("gen_cached_token_indptr",
                               gen_cached_token_indptr):
            return None
        kv_lens_cuda_2d = getattr(attn_metadata, "kv_lens_cuda_2d", None)
        if not append_required("kv_lens_cuda_2d", kv_lens_cuda_2d):
            return None

        for name in _OPTIONAL_DSA_DISPATCH_METADATA_TENSORS:
            value = getattr(attn_metadata, name, None)
            if value is not None:
                tensors.append(value)
        return tensors

    def _dsa_window_plan_scratch_shapes(
        self,
        *,
        layer_idx: int,
        state: DeepSeekResidentShapeState,
        self_attn: Any,
        attn_metadata: Any,
        input_tokens: int,
    ) -> tuple[tuple[int, ...], ...] | None:
        k_b_shape = self._layer_site_shape(layer_idx,
                                           _SITE_ATTN_K_B_PROJ_TRANS)
        v_b_shape = self._layer_site_shape(layer_idx, _SITE_ATTN_V_B_PROJ)
        if (not k_b_shape or len(k_b_shape) != 3 or not v_b_shape
                or len(v_b_shape) != 3):
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_projection_assets_missing")
            return None
        mqa = getattr(self_attn, "mqa", None)
        num_heads = int(k_b_shape[0])
        kv_lora_rank = int(k_b_shape[1])
        rope_dim = int(
            getattr(mqa, "qk_rope_head_dim",
                    getattr(self_attn, "qk_rope_head_dim", 0)) or 0)
        v_proj_heads = int(v_b_shape[0])
        v_proj_rank = int(v_b_shape[2])
        if (num_heads <= 0 or kv_lora_rank <= 0 or rope_dim <= 0
                or v_proj_heads != num_heads or v_proj_rank != kv_lora_rank):
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_projection_shape_invalid")
            return None
        num_seqs = int(getattr(attn_metadata, "num_seqs", input_tokens) or 0)
        if num_seqs <= 0:
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_num_seqs_invalid")
            return None
        shapes = [
            (state.batch_size, num_heads, kv_lora_rank + rope_dim),
            (state.batch_size, num_heads, kv_lora_rank),
            (num_seqs + 1, ),
            (num_seqs + 1, ),
            (1, ),
        ]
        kv_cache_manager = getattr(attn_metadata, "kv_cache_manager", None)
        dtype = getattr(kv_cache_manager, "dtype", None)
        dtype_name = getattr(dtype, "name", None) or str(dtype)
        if dtype_name == "FP8" or str(dtype_name).lower().endswith("fp8"):
            shapes.extend((
                (2, ),
                (1, ),
                (state.batch_size, num_heads, kv_lora_rank + rope_dim),
            ))
        mqa = getattr(self_attn, "mqa", None)
        indexer = None if mqa is None else getattr(mqa, "indexer", None)
        indexer_head_dim = int(
            getattr(kv_cache_manager, "index_head_dim", 0)
            or getattr(indexer, "head_dim", 0)
            or getattr(mqa, "qk_nope_head_dim", 0) or 0)
        try:
            num_sparse_topk = int(
                getattr(attn_metadata, "num_sparse_topk", 0)
                or getattr(indexer, "index_topk", 0) or 0)
        except (TypeError, ValueError):
            num_sparse_topk = 0
        if indexer_head_dim <= 0 or indexer_head_dim % 2 != 0:
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_indexer_head_dim_invalid")
            return None
        if num_sparse_topk <= 0:
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_num_sparse_topk_missing")
            return None
        indexer_num_heads = int(getattr(indexer, "n_heads", 0) or 0)
        if indexer_num_heads <= 0:
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_indexer_num_heads_invalid")
            return None
        shapes.extend((
            (state.batch_size, indexer_num_heads, indexer_head_dim // 2),
            (state.batch_size, indexer_head_dim // 2),
            (state.batch_size, 1),
            (state.batch_size, indexer_num_heads),
            (state.batch_size, indexer_num_heads, 1),
            (state.batch_size, num_sparse_topk),
            (state.batch_size, ),
        ))
        return tuple(shapes)

    def _ensure_dsa_dispatch_scratch(
        self,
        *,
        layer_idx: int,
        state: DeepSeekResidentShapeState | None,
        attn_metadata: Any,
        proj_outputs: tuple[Any, ...],
        output: Any,
        input_tokens: int,
    ) -> list[Any] | None:
        if state is None:
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_state_missing")
            return None
        if len(proj_outputs) < 4:
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_proj_outputs_missing")
            return None
        q = proj_outputs[0]
        k_pe = proj_outputs[2]
        q_shape = _tensor_shape(q)
        k_pe_shape = _tensor_shape(k_pe)
        output_shape = _tensor_shape(output)
        if len(q_shape) != 2 or len(k_pe_shape) != 2:
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_proj_shape_invalid")
            return None
        if len(output_shape) != 2:
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_output_shape_invalid")
            return None
        k_b_shape = self._layer_site_shape(layer_idx,
                                           _SITE_ATTN_K_B_PROJ_TRANS)
        v_b_shape = self._layer_site_shape(layer_idx, _SITE_ATTN_V_B_PROJ)
        if (not k_b_shape or len(k_b_shape) != 3 or not v_b_shape
                or len(v_b_shape) != 3):
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_projection_assets_missing")
            return None
        input_tokens = int(input_tokens)
        if q_shape[0] < input_tokens or k_pe_shape[0] < input_tokens:
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_proj_batch_too_small")
            return None
        num_heads = int(k_b_shape[0])
        kv_lora_rank = int(k_b_shape[1])
        rope_dim = int(k_pe_shape[1])
        v_proj_heads = int(v_b_shape[0])
        v_proj_rank = int(v_b_shape[2])
        if (num_heads <= 0 or kv_lora_rank <= 0 or rope_dim <= 0
                or v_proj_heads != num_heads or v_proj_rank != kv_lora_rank):
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_projection_shape_invalid")
            return None
        num_seqs = int(getattr(attn_metadata, "num_seqs", input_tokens))
        if num_seqs <= 0:
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_num_seqs_invalid")
            return None
        kv_cache_manager = getattr(attn_metadata, "kv_cache_manager", None)
        dtype = getattr(kv_cache_manager, "dtype", None)
        dtype_name = getattr(dtype, "name", None) or str(dtype)
        needs_fp8_mla_scratch = (
            dtype_name == "FP8" or str(dtype_name).lower().endswith("fp8"))
        scratch_states = state.scratch.setdefault("dsa_dispatch_scratch_states",
                                                 {})
        expected_key = (state.batch_size, num_seqs, num_heads, kv_lora_rank,
                        rope_dim, needs_fp8_mla_scratch)
        scratch = scratch_states.get(layer_idx)
        if scratch is not None and scratch.get("key") == expected_key:
            return scratch["tensors"]
        torch = _import_torch()
        if torch is None:
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_torch_missing")
            return None
        device = state.device
        dtype = getattr(q, "dtype", state.dtype)
        fused_q = torch.empty(
            (state.batch_size, num_heads, kv_lora_rank + rope_dim),
            device=device,
            dtype=dtype,
        )
        latent_output = torch.empty(
            (state.batch_size, num_heads, kv_lora_rank),
            device=device,
            dtype=dtype,
        )
        cu_q_seqlens = torch.empty(
            (num_seqs + 1, ),
            device=device,
            dtype=_torch_int32_dtype(torch),
        )
        cu_kv_seqlens = torch.empty(
            (num_seqs + 1, ),
            device=device,
            dtype=_torch_int32_dtype(torch),
        )
        fmha_scheduler_counter = torch.empty(
            (1, ),
            device=device,
            dtype=_torch_uint32_dtype(torch),
        )
        tensors = [
            fused_q,
            latent_output,
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
        ]
        if needs_fp8_mla_scratch:
            tensors.extend((
                torch.empty(
                    (2, ),
                    device=device,
                    dtype=_torch_float32_dtype(torch),
                ),
                torch.empty(
                    (1, ),
                    device=device,
                    dtype=_torch_float32_dtype(torch),
                ),
                torch.empty(
                    (state.batch_size, num_heads, kv_lora_rank + rope_dim),
                    device=device,
                    dtype=_torch_uint8_dtype(torch),
                ),
            ))
        scratch_states[layer_idx] = {
            "key": expected_key,
            "tensors": tensors,
        }
        return tensors

    def _ensure_dsa_projection_scratch(
        self,
        *,
        layer_idx: int,
        state: DeepSeekResidentShapeState,
        self_attn: Any,
    ) -> dict[str, Any] | None:
        q_lora_rank = int(getattr(self_attn, "q_lora_rank", 0) or 0)
        kv_lora_rank = int(getattr(self_attn, "kv_lora_rank", 0) or 0)
        rope_dim = int(getattr(self_attn, "qk_rope_head_dim", 0) or 0)
        q_width = self._q_projection_output_size(layer_idx, self_attn)
        if (q_lora_rank <= 0 or kv_lora_rank <= 0 or rope_dim <= 0
                or q_width is None or q_width <= 0):
            self._scheduler_state.last_attention_dsa_proj_reason = (
                "resident_attention_dsa_projection_shape_invalid")
            return None
        scratch_states = state.scratch.setdefault("dsa_projection_states", {})
        expected_key = (
            state.batch_size,
            q_width,
            q_lora_rank,
            kv_lora_rank,
            rope_dim,
        )
        scratch = scratch_states.get(layer_idx)
        if scratch is not None and scratch.get("key") == expected_key:
            return scratch["tensors"]
        torch = _import_torch()
        if torch is None:
            self._scheduler_state.last_attention_dsa_proj_reason = (
                "resident_attention_dsa_projection_torch_missing")
            return None
        tensors = {
            "q":
            torch.empty((state.batch_size, q_width),
                        device=state.device,
                        dtype=state.dtype),
            "q_lora":
            torch.empty((state.batch_size, q_lora_rank),
                        device=state.device,
                        dtype=state.dtype),
            "compressed_kv":
            torch.empty((state.batch_size, kv_lora_rank),
                        device=state.device,
                        dtype=state.dtype),
            "k_pe":
            torch.empty((state.batch_size, rope_dim),
                        device=state.device,
                        dtype=state.dtype),
            "latent_cache":
            torch.empty((state.batch_size, kv_lora_rank + rope_dim),
                        device=state.device,
                        dtype=state.dtype),
        }
        scratch_states[layer_idx] = {
            "key": expected_key,
            "tensors": tensors,
        }
        return tensors

    def _q_projection_output_size(
        self,
        layer_idx: int,
        self_attn: Any,
    ) -> int | None:
        q_b_shape = self._layer_site_shape(layer_idx,
                                           _SITE_ATTN_Q_B_PROJ_WEIGHT)
        if q_b_shape and len(q_b_shape) == 2:
            return int(q_b_shape[0])
        num_heads = int(
            getattr(self_attn, "num_heads_tp_cp",
                    getattr(self_attn, "num_heads_tp", 0)) or 0)
        qk_nope = int(getattr(self_attn, "qk_nope_head_dim", 0) or 0)
        qk_rope = int(getattr(self_attn, "qk_rope_head_dim", 0) or 0)
        if num_heads > 0 and qk_nope > 0 and qk_rope > 0:
            return num_heads * (qk_nope + qk_rope)
        return None

    def _dsa_dispatch_runtime_descriptor(
        self,
        *,
        layer_idx: int | None = None,
        self_attn: Any,
        attn_metadata: Any,
        topk_indices: Any,
        input_tokens: int,
    ) -> tuple[dict[str, Any], dict[str, int], dict[str, float]] | None:
        mqa = getattr(self_attn, "mqa", None)
        kv_cache_manager = getattr(attn_metadata, "kv_cache_manager", None)
        if mqa is None or kv_cache_manager is None:
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_runtime_owner_missing")
            return None
        max_seq_len = int(getattr(attn_metadata, "max_seq_len", 0) or 0)
        ensure_rope_table_size = getattr(mqa, "_ensure_rope_table_size", None)
        if callable(ensure_rope_table_size) and max_seq_len > 0:
            try:
                ensure_rope_table_size(max_seq_len)
            except (AttributeError, RuntimeError, AssertionError, TypeError):
                self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                    "resident_attention_dsa_dispatch_rope_table_failed")
                return None

        runtime_tensors: dict[str, Any] = {}

        def add_required_tensor(name: str, value: Any) -> bool:
            if value is None:
                self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                    f"resident_attention_dsa_dispatch_{name}_missing")
                return False
            runtime_tensors[name] = value
            return True

        def add_optional_tensor(name: str, value: Any) -> None:
            if value is not None:
                runtime_tensors[name] = value

        if not add_required_tensor("rotary_cos_sin",
                                   getattr(mqa, "rotary_cos_sin", None)):
            return None
        topk_indices_pool = getattr(attn_metadata, "topk_indices_pool", None)
        add_optional_tensor("topk_indices_pool", topk_indices_pool)
        if not add_required_tensor("dsa_req_idx_per_token",
                                   getattr(attn_metadata, "req_idx_per_token",
                                           None)):
            return None
        if not add_required_tensor("kv_cache_block_offsets",
                                   getattr(attn_metadata,
                                           "kv_cache_block_offsets", None)):
            return None
        if not add_required_tensor("host_kv_cache_pool_pointers",
                                   getattr(attn_metadata,
                                           "host_kv_cache_pool_pointers",
                                           None)):
            return None
        if not add_required_tensor("host_kv_cache_pool_mapping",
                                   getattr(attn_metadata,
                                           "host_kv_cache_pool_mapping",
                                           None)):
            return None
        if not add_required_tensor("kv_lens_runtime",
                                   getattr(attn_metadata, "kv_lens_runtime",
                                           None)):
            return None
        if not add_required_tensor("prompt_lens_cpu_runtime",
                                   getattr(attn_metadata,
                                           "prompt_lens_cpu_runtime", None)):
            return None
        get_primary_pool = getattr(kv_cache_manager, "get_unique_primary_pool",
                                   None)
        if not callable(get_primary_pool):
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_dense_kv_pool_missing")
            return None
        try:
            dense_kv_pool = get_primary_pool()
        except (AttributeError, RuntimeError, AssertionError, TypeError):
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_dense_kv_pool_missing")
            return None
        if not add_required_tensor("dense_kv_pool", dense_kv_pool):
            return None

        dtype = getattr(kv_cache_manager, "dtype", None)
        dtype_name = getattr(dtype, "name", None) or str(dtype)
        kv_dispatch_mode = _DSA_KV_DISPATCH_STANDARD_MLA
        if dtype_name == "NVFP4" or str(dtype).endswith("NVFP4"):
            kv_dispatch_mode = _DSA_KV_DISPATCH_DENSE_NVFP4
        if bool(getattr(mqa, "has_fp4_kv_cache", False)):
            kv_dispatch_mode = _DSA_KV_DISPATCH_DENSE_NVFP4
        quant_mode = getattr(mqa, "quant_mode", None)
        has_fp4_kv_cache = getattr(quant_mode, "has_fp4_kv_cache", None)
        if callable(has_fp4_kv_cache) and bool(has_fp4_kv_cache()):
            kv_dispatch_mode = _DSA_KV_DISPATCH_DENSE_NVFP4

        if kv_dispatch_mode == _DSA_KV_DISPATCH_DENSE_NVFP4:
            get_scale_pool = getattr(kv_cache_manager,
                                     "get_dense_block_scale_pool", None)
            if not callable(get_scale_pool):
                self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                    "resident_attention_dsa_dispatch_dense_kv_scale_pool_method_missing")
                return None
            try:
                dense_kv_scale_pool = get_scale_pool()
            except AssertionError:
                self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                    "resident_attention_dsa_dispatch_dense_kv_scale_pool_unsupported_dtype:"
                    f"{dtype_name}")
                return None
            except (AttributeError, RuntimeError, TypeError):
                self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                    "resident_attention_dsa_dispatch_dense_kv_scale_pool_failed")
                return None
            if not add_required_tensor("dense_kv_scale_pool",
                                       dense_kv_scale_pool):
                return None
        else:
            for name, value in (
                ("attention_workspace",
                 getattr(attn_metadata, "effective_workspace", None)),
                ("host_total_kv_lens",
                 getattr(attn_metadata, "host_total_kv_lens", None)),
                ("prompt_lens_cuda_runtime",
                 getattr(attn_metadata, "prompt_lens_cuda_runtime", None)),
                ("host_request_types_runtime",
                 getattr(attn_metadata, "host_request_types_runtime", None)),
            ):
                if not add_required_tensor(name, value):
                    return None
            add_optional_tensor("cache_indirection",
                                getattr(attn_metadata, "cache_indirection",
                                        None))
            add_optional_tensor(
                "kv_scale_orig_quant",
                getattr(mqa, "kv_scale_orig_quant",
                        getattr(self_attn, "kv_scale_orig_quant", None)))
            add_optional_tensor(
                "kv_scale_quant_orig",
                getattr(mqa, "kv_scale_quant_orig",
                        getattr(self_attn, "kv_scale_quant_orig", None)))
            add_optional_tensor(
                "flash_mla_tile_scheduler_metadata",
                getattr(attn_metadata, "flash_mla_tile_scheduler_metadata",
                        None))
            add_optional_tensor("flash_mla_num_splits",
                                getattr(attn_metadata,
                                        "flash_mla_num_splits", None))

        add_optional_tensor("block_ids_per_seq",
                            getattr(attn_metadata, "block_ids_per_seq", None))
        add_optional_tensor("helix_position_offsets",
                            getattr(attn_metadata, "helix_position_offsets",
                                    None))
        add_optional_tensor("helix_is_inactive_rank",
                            getattr(attn_metadata, "helix_is_inactive_rank",
                                    None))
        add_optional_tensor(
            "sparse_mla_tile_scheduler_metadata",
            getattr(attn_metadata, "sparse_mla_tile_scheduler_metadata", None))
        add_optional_tensor("sparse_mla_num_splits",
                            getattr(attn_metadata, "sparse_mla_num_splits",
                                    None))

        tokens_per_block = int(
            getattr(kv_cache_manager, "tokens_per_block", 0) or 0)
        indexer = getattr(mqa, "indexer", None)
        indexer_head_dim = int(
            getattr(kv_cache_manager, "index_head_dim", 0)
            or getattr(indexer, "head_dim", 0)
            or getattr(mqa, "qk_nope_head_dim", 0) or 0)
        indexer_num_heads = int(getattr(indexer, "n_heads", 0) or 0)
        indexer_quant_block_size = int(
            getattr(kv_cache_manager, "quant_block_size", 0)
            or indexer_head_dim or 0)
        indexer_use_fp4 = bool(getattr(kv_cache_manager, "use_fp4", False))
        indexer_data_bytes_per_token = (
            indexer_head_dim // 2 if indexer_use_fp4 else indexer_head_dim)
        sparse_attention_config = getattr(mqa, "sparse_attention_config", None)
        indexer_mode = getattr(sparse_attention_config, "indexer_mode", "")
        enable_nvfp4_hisa = bool(
            getattr(sparse_attention_config, "enable_nvfp4_hisa", False))
        resident_indexer_hisa_enabled = bool(
            indexer_use_fp4 and indexer_mode == "indexcache-hisa"
            and enable_nvfp4_hisa)
        num_seqs = int(getattr(attn_metadata, "num_seqs", input_tokens) or 0)
        get_local_layer_idx = getattr(mqa, "get_local_layer_idx", None)
        try:
            local_layer_idx = (
                int(get_local_layer_idx(attn_metadata))
                if callable(get_local_layer_idx) else
                int(getattr(self_attn, "layer_idx", 0) or 0))
        except (AttributeError, RuntimeError, AssertionError, TypeError,
                ValueError):
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_local_layer_idx_failed")
            return None
        if resident_indexer_hisa_enabled:
            get_hisa_page_reps = getattr(
                kv_cache_manager, "get_indexer_hisa_page_rep_buffers", None)
            if callable(get_hisa_page_reps):
                try:
                    page_reps, page_counts = get_hisa_page_reps(
                        int(layer_idx if layer_idx is not None else getattr(
                            self_attn, "layer_idx", local_layer_idx)))
                except (AttributeError, IndexError, KeyError, RuntimeError,
                        TypeError, ValueError):
                    page_reps = None
                    page_counts = None
                add_optional_tensor("indexer_hisa_page_reps", page_reps)
                add_optional_tensor("indexer_hisa_page_counts", page_counts)
        get_indices_block_size = getattr(sparse_attention_config,
                                         "get_indices_block_size", None)
        try:
            sparse_attn_indices_block_size = (
                int(get_indices_block_size())
                if callable(get_indices_block_size) else 1)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            sparse_attn_indices_block_size = 1
        try:
            num_sparse_topk = int(
                getattr(attn_metadata, "num_sparse_topk", 0) or 0)
        except (TypeError, ValueError):
            num_sparse_topk = 0
        topk_shape = _tensor_shape(topk_indices)
        if num_sparse_topk <= 0 and len(topk_shape) >= 2:
            num_sparse_topk = int(topk_shape[1])
        if (kv_dispatch_mode == _DSA_KV_DISPATCH_STANDARD_MLA
                and num_sparse_topk <= 0):
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_num_sparse_topk_missing")
            return None

        def int_config_attr(owner: Any, name: str, default: int) -> int:
            value = getattr(owner, name, default)
            enum_value = getattr(value, "value", value)
            try:
                return int(enum_value)
            except (TypeError, ValueError):
                return int(default)

        def float_config_attr(owner: Any, name: str, default: float) -> float:
            value = getattr(owner, name, default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        config = {
            "num_contexts":
            int(getattr(attn_metadata, "num_contexts", 0) or 0),
            "num_ctx_tokens":
            int(getattr(attn_metadata, "num_ctx_tokens", 0) or 0),
            "num_generations":
            int(getattr(attn_metadata, "num_generations", num_seqs) or 0),
            "num_seqs":
            num_seqs,
            "max_seq_len":
            max_seq_len,
            "beam_width":
            int(getattr(attn_metadata, "beam_width", 1) or 1),
            "tokens_per_block":
            tokens_per_block,
            "local_layer_idx":
            local_layer_idx,
            "kv_dispatch_mode":
            kv_dispatch_mode,
            "max_num_requests":
            int(getattr(attn_metadata, "max_num_requests", num_seqs)
                or num_seqs),
            "max_context_length":
            int(getattr(attn_metadata, "max_context_length", max_seq_len)
                or max_seq_len),
            "attention_window_size":
            int(getattr(attn_metadata, "max_seq_len", max_seq_len)
                or max_seq_len),
            "num_sparse_topk":
            num_sparse_topk,
            "sparse_attn_indices_block_size":
            sparse_attn_indices_block_size,
            "mask_type":
            int_config_attr(mqa, "mask_type",
                            int_config_attr(self_attn, "mask_type", 0)),
            "position_embedding_type":
            int_config_attr(
                mqa, "position_embedding_type",
                int_config_attr(self_attn, "position_embedding_type", 0)),
            "rope_dim":
            int_config_attr(mqa, "rope_dim",
                            int_config_attr(self_attn, "rope_dim", 0)),
            "rope_scale_type":
            int_config_attr(
                mqa, "rope_scale_type",
                int_config_attr(self_attn, "rope_scale_type", 0)),
            "rope_max_positions":
            int_config_attr(
                mqa, "rope_max_positions",
                int_config_attr(self_attn, "rope_max_positions", 0)),
            "rope_original_max_positions":
            int_config_attr(
                mqa, "rope_original_max_positions",
                int_config_attr(self_attn, "rope_original_max_positions",
                                0)),
            "attention_chunk_size":
            int_config_attr(
                mqa, "attention_chunk_size",
                int_config_attr(self_attn, "attention_chunk_size", 0)),
            "use_paged_context_fmha":
            int(bool(getattr(attn_metadata, "use_paged_context_fmha",
                             False))),
            "predicted_tokens_per_seq":
            int(getattr(mqa, "predicted_tokens_per_seq", 1) or 1),
            "num_heads":
            int(getattr(mqa, "num_heads",
                        getattr(self_attn, "num_heads", 0)) or 0),
            "num_kv_heads":
            int(getattr(mqa, "num_kv_heads",
                        getattr(self_attn, "num_kv_heads", 1)) or 1),
            "head_dim":
            int(getattr(mqa, "head_dim",
                        getattr(self_attn, "head_dim", 0)) or 0),
            "quant_mode":
            int(getattr(mqa, "quant_mode",
                        getattr(self_attn, "quant_mode", 0)) or 0),
            "q_lora_rank":
            int(getattr(mqa, "q_lora_rank",
                        getattr(self_attn, "q_lora_rank", 0)) or 0),
            "kv_lora_rank":
            int(getattr(mqa, "kv_lora_rank",
                        getattr(self_attn, "kv_lora_rank", 0)) or 0),
            "qk_nope_head_dim":
            int(getattr(mqa, "qk_nope_head_dim",
                        getattr(self_attn, "qk_nope_head_dim", 0)) or 0),
            "qk_rope_head_dim":
            int(getattr(mqa, "qk_rope_head_dim",
                        getattr(self_attn, "qk_rope_head_dim", 0)) or 0),
            "v_head_dim":
            int(getattr(self_attn, "v_head_dim",
                        getattr(mqa, "v_head_dim", 0)) or 0),
            "rope_append":
            int(bool(getattr(mqa, "rope_append",
                             getattr(self_attn, "rope_append", False)))),
            "resident_indexer_head_dim":
            indexer_head_dim,
            "resident_indexer_num_heads":
            indexer_num_heads,
            "resident_indexer_rope_dim":
            int(getattr(indexer, "rope_dim", getattr(mqa, "qk_rope_head_dim",
                                                      0)) or 0),
            "resident_indexer_quant_block_size":
            indexer_quant_block_size,
            "resident_indexer_data_bytes_per_token":
            indexer_data_bytes_per_token,
            "resident_indexer_skip_topk":
            int(bool(getattr(indexer, "skip_topk", False))),
            "resident_indexer_step_freq":
            max(1, int(getattr(indexer, "index_topk_step_freq", 1) or 1)),
            "resident_indexer_step_recency_patch":
            int(bool(getattr(indexer, "_xstep_recency_patch", False))),
            "resident_indexer_hisa_enabled":
            int(resident_indexer_hisa_enabled),
            "resident_indexer_hisa_block_size":
            int(getattr(sparse_attention_config, "hisa_block_size", 128)
                or 128),
            "resident_indexer_hisa_block_topk":
            int(getattr(sparse_attention_config, "hisa_block_topk", 64)
                or 64),
            "resident_indexer_hisa_min_seq_len":
            int(getattr(sparse_attention_config, "hisa_min_seq_len", 0)
                or 0),
        }
        if any(value <= 0 for key, value in config.items() if key not in (
                "num_contexts", "num_ctx_tokens", "local_layer_idx",
                "kv_dispatch_mode", "rope_append", "quant_mode", "mask_type",
                "position_embedding_type", "rope_dim", "rope_scale_type",
                "rope_max_positions", "rope_original_max_positions",
                "attention_chunk_size", "use_paged_context_fmha",
                "num_sparse_topk", "resident_indexer_skip_topk",
                "resident_indexer_step_recency_patch",
                "resident_indexer_hisa_enabled",
                "resident_indexer_hisa_min_seq_len")):
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_runtime_config_invalid")
            return None
        scalars = {
            "q_scaling":
            float(getattr(mqa, "q_scaling",
                          getattr(self_attn, "q_scaling", 1.0))),
            "softmax_scale":
            float(getattr(self_attn, "softmax_scale",
                          getattr(mqa, "softmax_scale", 1.0))),
            "rope_base":
            float_config_attr(mqa, "rope_base",
                              float_config_attr(self_attn, "rope_base",
                                                10000.0)),
            "rope_scale":
            float_config_attr(mqa, "rope_scale",
                              float_config_attr(self_attn, "rope_scale",
                                                1.0)),
            "rope_short_m_scale":
            float_config_attr(
                mqa, "rope_short_m_scale",
                float_config_attr(self_attn, "rope_short_m_scale", 1.0)),
            "rope_long_m_scale":
            float_config_attr(
                mqa, "rope_long_m_scale",
                float_config_attr(self_attn, "rope_long_m_scale", 1.0)),
            "resident_indexer_weight_scale_factor":
            float_config_attr(getattr(mqa, "indexer", None),
                              "weight_scale_factor",
                              float(getattr(self_attn, "softmax_scale", 1.0))
                              *
                              (float(config["resident_indexer_num_heads"])**
                               -0.5)),
            "resident_indexer_hisa_compression_ratio":
            float_config_attr(sparse_attention_config,
                              "hisa_compression_ratio", 4.0),
        }
        if any(value <= 0.0 for key, value in scalars.items()
               if key != "resident_indexer_hisa_compression_ratio"):
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_runtime_scalars_invalid")
            return None
        if scalars["resident_indexer_hisa_compression_ratio"] < 0.0:
            self._scheduler_state.last_attention_dsa_native_dispatch_reason = (
                "resident_attention_dsa_dispatch_runtime_scalars_invalid")
            return None
        return runtime_tensors, config, scalars

    def _native_dsa_attention_dispatch_not_ready_reason(
        self,
        native_handle: Any,
    ) -> str:
        reason_fn = getattr(
            native_handle, "run_layer_dsa_attention_dispatch_not_ready_reason",
            None)
        if not callable(reason_fn):
            return "resident_attention_dsa_dispatch_native_not_ready"
        try:
            reason = reason_fn()
        except (RuntimeError, TypeError):
            return "resident_attention_dsa_dispatch_native_not_ready"
        return reason if isinstance(reason, str) and reason else (
            "resident_attention_dsa_dispatch_native_not_ready")

    def _native_handle_bool_method(
        self,
        native_handle: Any,
        method_name: str,
    ) -> bool | None:
        method = (
            None if native_handle is None else getattr(
                native_handle, method_name, None))
        if not callable(method):
            return None
        try:
            return bool(method())
        except (RuntimeError, TypeError):
            return None

    def run_input_embedding_stage(
        self,
        *,
        input_ids: Any,
        invocation: Any,
    ) -> Any | None:
        native_handle = self._get_or_create_native_handle()
        if native_handle is None:
            return None
        run_input_embedding = getattr(native_handle, "run_input_embedding",
                                      None)
        if not callable(run_input_embedding):
            return None
        state = self._get_shape_state(invocation, input_ids)
        self._scheduler_state.embedding_stage_calls += 1
        return run_input_embedding(
            input_ids,
            state.scratch["hidden_states"],
            invocation.input_tokens,
        )

    def run_layer_input_rmsnorm_stage(
        self,
        *,
        layer_idx: int,
        invocation: Any,
        eps: float,
        use_gemma: bool = False,
    ) -> Any | None:
        native_handle = self._get_or_create_native_handle()
        if native_handle is None:
            return None
        run_input_rmsnorm = getattr(native_handle,
                                    "run_layer_input_rmsnorm", None)
        if not callable(run_input_rmsnorm):
            return None
        state = self._shape_states.get(invocation.stable_shape_key)
        if state is None:
            return None
        self._scheduler_state.input_rmsnorm_stage_calls += 1
        bridged = self._try_run_input_rmsnorm_python_bridge(
            layer_idx=layer_idx,
            state=state,
            input_tokens=invocation.input_tokens,
        )
        if bridged is not None:
            return bridged
        return run_input_rmsnorm(
            layer_idx,
            state.scratch["hidden_states"],
            state.scratch["norm_hidden_states"],
            invocation.input_tokens,
            eps,
            use_gemma,
        )

    def run_layer_input_gated_norm_stage(
        self,
        *,
        layer_idx: int,
        invocation: Any,
    ) -> Any | None:
        native_handle = self._get_or_create_native_handle()
        if native_handle is None:
            return None
        run_input_gated_norm = getattr(native_handle,
                                       "run_layer_input_gated_norm", None)
        if not callable(run_input_gated_norm):
            return None
        state = self._shape_states.get(invocation.stable_shape_key)
        if state is None:
            return None
        self._scheduler_state.input_gated_norm_stage_calls += 1
        bridged = self._try_run_gated_norm_python_bridge(
            layer_idx=layer_idx,
            state=state,
            input_name="norm_hidden_states",
            output_name="gated_hidden_states",
            down_name="input_gated_norm_down",
            up_name="input_gated_norm_up",
            input_tokens=invocation.input_tokens,
        )
        if bridged is not None:
            return bridged
        return run_input_gated_norm(
            layer_idx,
            state.scratch["norm_hidden_states"],
            state.scratch["gated_hidden_states"],
            invocation.input_tokens,
        )

    def run_layer_attention_output_tail_stage(
        self,
        *,
        layer_idx: int,
        invocation: Any,
    ) -> Any | None:
        native_handle = self._get_or_create_native_handle()
        if native_handle is None:
            return None
        run_attention_tail = getattr(native_handle,
                                     "run_layer_attention_output_tail", None)
        if not callable(run_attention_tail):
            return None
        state = self._shape_states.get(invocation.stable_shape_key)
        if state is None:
            return None
        tail_scratch = self._ensure_attention_output_tail_scratch(
            state, layer_idx)
        if tail_scratch is None:
            return None
        self._scheduler_state.attention_output_tail_stage_calls += 1
        layer = self._model_layer(layer_idx)
        self_attn = getattr(layer, "self_attn", None)
        if self._attention_output_tail_needs_python_bridge(layer_idx,
                                                           self_attn):
            bridged = self._run_layer_attention_output_tail_python_bridge(
                layer_idx=layer_idx,
                layer=layer,
                self_attn=self_attn,
                state=state,
                tail_scratch=tail_scratch,
                input_tokens=invocation.input_tokens,
            )
            if bridged is not None:
                self._scheduler_state.last_attention_output_tail_reason = (
                    state.scratch.pop(
                        "last_attention_output_tail_bridge_reason",
                        "resident_attention_output_tail_python_bridge_executed"
                    ))
                return bridged
            self._scheduler_state.last_attention_output_tail_reason = (
                "resident_attention_output_tail_python_bridge_failed")
            return None
        return run_attention_tail(
            layer_idx,
            tail_scratch["attention_core_output"],
            state.scratch["gated_hidden_states"],
            tail_scratch["attention_gate"],
            state.scratch["attention_hidden_states"],
            invocation.input_tokens,
        )

    def _try_run_native_linear_bridge(
        self,
        *,
        native_handle: Any,
        module: Any,
        input_tensor: Any,
        input_tokens: int,
        include_bias: bool = True,
    ) -> Any | None:
        if not _env_flag(_NATIVE_LINEAR_BRIDGE_ENV_NAME):
            return None
        run_linear = getattr(native_handle, "run_nvfp4_linear", None)
        if not callable(run_linear):
            return None
        if module is None or not bool(getattr(module, "has_nvfp4", False)):
            return None
        if getattr(module, "pre_quant_scale", None) is not None:
            return None
        if bool(getattr(module, "force_dynamic_quantization", False)):
            return None
        weight = getattr(module, "weight", None)
        weight_scale = getattr(module, "weight_scale", None)
        input_scale = getattr(module, "input_scale", None)
        alpha = getattr(module, "alpha", None)
        if (weight is None or weight_scale is None or input_scale is None
                or alpha is None):
            return None
        scaling_vector_size = int(
            getattr(module, "scaling_vector_size", 16) or 16)
        allowed_backends = getattr(module, "nvfp4_allowed_backends_str", None)
        if not allowed_backends:
            allowed_backends = "cutlass,cublaslt,cuda_core"
        try:
            output = run_linear(
                input_tensor,
                weight,
                weight_scale,
                input_scale,
                alpha,
                int(input_tokens),
                scaling_vector_size,
                str(allowed_backends),
            )
            out_features = int(getattr(module, "out_features", 0) or 0)
            shape = getattr(output, "shape", ())
            if out_features > 0 and len(shape) > 0 and int(
                    shape[-1]) > out_features:
                output = output[..., :out_features].contiguous()
            if include_bias:
                bias = getattr(module, "bias", None)
                if bias is not None:
                    output = output + bias
            return output
        except (AttributeError, RuntimeError, AssertionError, TypeError,
                ValueError):
            return None

    def _apply_linear_allreduce(
        self,
        *,
        module: Any,
        output: Any,
        all_reduce_params: Any | None,
    ) -> Any | None:
        if not bool(getattr(module, "reduce_output", False)):
            return output
        all_reduce = getattr(module, "all_reduce", None)
        if all_reduce is None or not callable(all_reduce):
            return None
        try:
            return all_reduce(output, all_reduce_params=all_reduce_params)
        except (AttributeError, RuntimeError, AssertionError, TypeError):
            return None

    def _attention_output_tail_needs_python_bridge(
        self,
        layer_idx: int,
        self_attn: Any,
    ) -> bool:
        o_proj = getattr(self_attn, "o_proj", None)
        logical_in_features = getattr(o_proj, "in_features", None)
        o_proj_shape = self._layer_site_shape(layer_idx,
                                              _SITE_ATTN_O_PROJ_WEIGHT)
        if (logical_in_features is not None and o_proj_shape
                and len(o_proj_shape) == 2
                and int(o_proj_shape[1]) != int(logical_in_features)):
            return True
        return bool(getattr(o_proj, "has_nvfp4", False))

    def _run_layer_attention_output_tail_python_bridge(
        self,
        *,
        layer_idx: int,
        layer: Any,
        self_attn: Any,
        state: DeepSeekResidentShapeState,
        tail_scratch: dict[str, Any],
        input_tokens: int,
    ) -> Any | None:
        torch = _import_torch()
        if torch is None:
            return None
        o_proj = getattr(self_attn, "o_proj", None)
        if o_proj is None or not callable(o_proj):
            return None
        input_tokens = int(input_tokens)
        if input_tokens <= 0:
            return None
        try:
            attention_output = tail_scratch["attention_core_output"].narrow(
                0, 0, input_tokens)
            attention_input = state.scratch["gated_hidden_states"].narrow(
                0, 0, input_tokens)
            native_handle = self._get_or_create_native_handle()
            gate_proj = getattr(self_attn, "gate_proj", None)
            o_proj = getattr(self_attn, "o_proj", None)
            if native_handle is not None:
                native_attention_output = attention_output
                if gate_proj is not None:
                    gate = self._try_run_native_linear_bridge(
                        native_handle=native_handle,
                        module=gate_proj,
                        input_tensor=attention_input,
                        input_tokens=input_tokens,
                    )
                    if gate is not None:
                        native_attention_output = (
                            native_attention_output * torch.sigmoid(gate))
                    else:
                        native_attention_output = None
                if native_attention_output is not None:
                    tp_rank = int(getattr(o_proj, "tp_rank", 0) or 0)
                    include_bias = tp_rank == 0
                    projected = self._try_run_native_linear_bridge(
                        native_handle=native_handle,
                        module=o_proj,
                        input_tensor=native_attention_output,
                        input_tokens=input_tokens,
                        include_bias=include_bias,
                    )
                    if projected is not None:
                        projected = self._apply_linear_allreduce(
                            module=o_proj,
                            output=projected,
                            all_reduce_params=self._attention_allreduce_params(
                                layer),
                        )
                        if projected is not None:
                            output = state.scratch["attention_hidden_states"]
                            output.narrow(0, 0,
                                          input_tokens).copy_(projected)
                            state.scratch[
                                "last_attention_output_tail_bridge_reason"] = (
                                    "resident_attention_output_tail_native_linear_bridge_executed"
                                )
                            return output
            if gate_proj is not None:
                gate = gate_proj(attention_input)
                attention_output = attention_output * torch.sigmoid(gate)
            projected = o_proj(
                attention_output,
                all_reduce_params=self._attention_allreduce_params(layer),
                layer_idx=layer_idx,
            )
            output = state.scratch["attention_hidden_states"]
            output.narrow(0, 0, input_tokens).copy_(projected)
            return output
        except (AttributeError, RuntimeError, AssertionError, TypeError):
            return None

    def run_layer_post_attention_rmsnorm_stage(
        self,
        *,
        layer_idx: int,
        invocation: Any,
        eps: float,
        use_gemma: bool = False,
    ) -> Any | None:
        native_handle = self._get_or_create_native_handle()
        if native_handle is None:
            return None
        run_post_attention_rmsnorm = getattr(
            native_handle, "run_layer_post_attention_rmsnorm", None)
        if not callable(run_post_attention_rmsnorm):
            return None
        state = self._shape_states.get(invocation.stable_shape_key)
        if state is None:
            return None
        self._scheduler_state.post_attention_rmsnorm_stage_calls += 1
        residual_input_scratch = state.scratch.get(
            "current_residual_states", state.scratch["hidden_states"])
        bridged = self._try_run_residual_rmsnorm_python_bridge(
            layer_idx=layer_idx,
            norm_name="post_attention_layernorm",
            input_tensor=state.scratch["attention_hidden_states"],
            residual_tensor=residual_input_scratch,
            output_tensor=state.scratch["post_attention_norm_hidden_states"],
            residual_output_tensor=state.scratch["post_attention_residual_states"],
            input_tokens=invocation.input_tokens,
        )
        if bridged is not None:
            return bridged
        return run_post_attention_rmsnorm(
            layer_idx,
            state.scratch["attention_hidden_states"],
            residual_input_scratch,
            state.scratch["post_attention_norm_hidden_states"],
            state.scratch["post_attention_residual_states"],
            invocation.input_tokens,
            eps,
            use_gemma,
        )

    def run_layer_post_attention_gated_norm_stage(
        self,
        *,
        layer_idx: int,
        invocation: Any,
    ) -> Any | None:
        native_handle = self._get_or_create_native_handle()
        if native_handle is None:
            return None
        run_post_attention_gated_norm = getattr(
            native_handle, "run_layer_post_attention_gated_norm", None)
        if not callable(run_post_attention_gated_norm):
            return None
        state = self._shape_states.get(invocation.stable_shape_key)
        if state is None:
            return None
        self._scheduler_state.post_attention_gated_norm_stage_calls += 1
        bridged = self._try_run_gated_norm_python_bridge(
            layer_idx=layer_idx,
            state=state,
            input_name="post_attention_norm_hidden_states",
            output_name="post_attention_gated_hidden_states",
            down_name="post_attention_gated_norm_down",
            up_name="post_attention_gated_norm_up",
            input_tokens=invocation.input_tokens,
        )
        if bridged is not None:
            return bridged
        return run_post_attention_gated_norm(
            layer_idx,
            state.scratch["post_attention_norm_hidden_states"],
            state.scratch["post_attention_gated_hidden_states"],
            invocation.input_tokens,
        )

    def _try_run_input_rmsnorm_python_bridge(
        self,
        *,
        layer_idx: int,
        state: DeepSeekResidentShapeState,
        input_tokens: int,
    ) -> Any | None:
        layer = self._model_layer(layer_idx)
        norm = getattr(layer, "input_layernorm", None)
        if norm is None or not callable(norm):
            return None
        input_tokens = int(input_tokens)
        if input_tokens <= 0:
            return None
        try:
            hidden_states = state.scratch["hidden_states"].narrow(
                0, 0, input_tokens)
            normed = norm(hidden_states)
            output = state.scratch["norm_hidden_states"]
            output.narrow(0, 0, input_tokens).copy_(normed)
            return output
        except (AttributeError, RuntimeError, AssertionError, TypeError):
            return None

    def _try_run_residual_rmsnorm_python_bridge(
        self,
        *,
        layer_idx: int,
        norm_name: str,
        input_tensor: Any,
        residual_tensor: Any,
        output_tensor: Any,
        residual_output_tensor: Any,
        input_tokens: int,
    ) -> Any | None:
        layer = self._model_layer(layer_idx)
        norm = getattr(layer, norm_name, None)
        if norm is None or not callable(norm):
            return None
        input_tokens = int(input_tokens)
        if input_tokens <= 0:
            return None
        try:
            input_prefix = input_tensor.narrow(0, 0, input_tokens)
            residual_prefix = residual_tensor.narrow(0, 0, input_tokens)
            normed, residual_out = norm(input_prefix, residual_prefix)
            output_tensor.narrow(0, 0, input_tokens).copy_(normed)
            residual_output_tensor.narrow(0, 0,
                                          input_tokens).copy_(residual_out)
            return output_tensor
        except (AttributeError, RuntimeError, AssertionError, TypeError,
                ValueError):
            return None

    def _try_run_gated_norm_python_bridge(
        self,
        *,
        layer_idx: int,
        state: DeepSeekResidentShapeState,
        input_name: str,
        output_name: str,
        down_name: str,
        up_name: str,
        input_tokens: int,
    ) -> Any | None:
        layer = self._model_layer(layer_idx)
        apply_gated_norm = getattr(layer, "_maybe_apply_gated_norm", None)
        gate_down = getattr(layer, down_name, None)
        gate_up = getattr(layer, up_name, None)
        if not callable(apply_gated_norm):
            return None
        input_tokens = int(input_tokens)
        if input_tokens <= 0:
            return None
        try:
            hidden_states = state.scratch[input_name].narrow(
                0, 0, input_tokens)
            gated = apply_gated_norm(hidden_states, gate_down, gate_up)
            output = state.scratch[output_name]
            output.narrow(0, 0, input_tokens).copy_(gated)
            return output
        except (AttributeError, RuntimeError, AssertionError, TypeError,
                ValueError):
            return None

    def run_layer_dense_mlp_stage(
        self,
        *,
        layer_idx: int,
        invocation: Any,
    ) -> Any | None:
        native_handle = self._get_or_create_native_handle()
        if native_handle is None:
            return None
        run_dense_mlp = getattr(native_handle, "run_layer_dense_mlp", None)
        if not callable(run_dense_mlp):
            return None
        state = self._shape_states.get(invocation.stable_shape_key)
        if state is None:
            return None
        self._scheduler_state.dense_mlp_stage_calls += 1
        layer = self._model_layer(layer_idx)
        mlp = getattr(layer, "mlp", None)
        if self._dense_mlp_needs_python_bridge(layer_idx, mlp):
            bridged = self._run_layer_dense_mlp_python_bridge(
                layer_idx=layer_idx,
                layer=layer,
                mlp=mlp,
                state=state,
                input_tokens=invocation.input_tokens,
            )
            if bridged is not None:
                self._scheduler_state.last_dense_mlp_reason = (
                    "resident_dense_mlp_python_bridge_executed")
                return bridged
            self._scheduler_state.last_dense_mlp_reason = (
                "resident_dense_mlp_python_bridge_failed")
            return None
        dense_mlp_scratch = self._ensure_dense_mlp_scratch(state, layer_idx)
        if dense_mlp_scratch is None:
            return None
        self._scheduler_state.last_dense_mlp_reason = (
            "resident_dense_mlp_native_executed")
        return run_dense_mlp(
            layer_idx,
            state.scratch["post_attention_gated_hidden_states"],
            dense_mlp_scratch,
            state.scratch["dense_mlp_output_states"],
            invocation.input_tokens,
        )

    def _dense_mlp_needs_python_bridge(
        self,
        layer_idx: int,
        mlp: Any,
    ) -> bool:
        gate_up_proj = getattr(mlp, "gate_up_proj", None)
        logical_in_features = getattr(gate_up_proj, "in_features", None)
        gate_up_shape = self._layer_site_shape(layer_idx,
                                               _SITE_DENSE_MLP_GATE_UP_WEIGHT)
        if (logical_in_features is not None and gate_up_shape
                and len(gate_up_shape) == 2
                and int(gate_up_shape[1]) != int(logical_in_features)):
            return True
        return bool(getattr(gate_up_proj, "has_nvfp4", False))

    def _run_layer_dense_mlp_python_bridge(
        self,
        *,
        layer_idx: int,
        layer: Any,
        mlp: Any,
        state: DeepSeekResidentShapeState,
        input_tokens: int,
    ) -> Any | None:
        if mlp is None or not callable(mlp):
            return None
        input_tokens = int(input_tokens)
        if input_tokens <= 0:
            return None
        try:
            hidden_states = state.scratch[
                "post_attention_gated_hidden_states"].narrow(
                    0, 0, input_tokens)
            final_hidden_states = mlp(
                hidden_states,
                final_all_reduce_params=self._dense_mlp_final_allreduce_params(
                    layer),
            )
            output = state.scratch["dense_mlp_output_states"]
            output.narrow(0, 0, input_tokens).copy_(final_hidden_states)
            return output
        except (AttributeError, RuntimeError, AssertionError, TypeError):
            return None

    def run_layer_moe_router_stage(
        self,
        *,
        layer_idx: int,
        invocation: Any,
    ) -> Any | None:
        native_handle = self._get_or_create_native_handle()
        if native_handle is None:
            return None
        run_moe_router = getattr(native_handle, "run_layer_moe_router", None)
        if not callable(run_moe_router):
            return None
        state = self._shape_states.get(invocation.stable_shape_key)
        if state is None:
            return None
        moe_router_scratch = self._ensure_moe_router_scratch(state, layer_idx)
        if moe_router_scratch is None:
            return None
        router_config = self._moe_router_config(layer_idx)
        if router_config is None:
            return None
        self._scheduler_state.moe_router_stage_calls += 1
        bridged = self._try_run_moe_router_python_bridge(
            layer_idx=layer_idx,
            state=state,
            router_scratch=moe_router_scratch,
            input_tokens=invocation.input_tokens,
        )
        if bridged is not None:
            return bridged
        return run_moe_router(
            layer_idx,
            state.scratch["post_attention_gated_hidden_states"],
            moe_router_scratch["logits"],
            moe_router_scratch["scores"],
            moe_router_scratch["topk_indices"],
            moe_router_scratch["topk_weights"],
            invocation.input_tokens,
            router_config["top_k"],
            router_config["n_group"],
            router_config["topk_group"],
            router_config["routed_scaling_factor"],
        )

    def _try_run_moe_router_python_bridge(
        self,
        *,
        layer_idx: int,
        state: DeepSeekResidentShapeState,
        router_scratch: dict[str, Any],
        input_tokens: int,
    ) -> Any | None:
        """Use the production DeepSeek gate/router kernels for resident routing."""

        layer = self._model_layer(layer_idx)
        mlp = getattr(layer, "mlp", None)
        gate = getattr(mlp, "gate", None)
        gate_apply = getattr(gate, "apply", None)
        if gate is None or not callable(gate) or not callable(gate_apply):
            return None
        input_tokens = int(input_tokens)
        if input_tokens <= 0:
            return None
        try:
            hidden_states = state.scratch[
                "post_attention_gated_hidden_states"].narrow(
                    0, 0, input_tokens)
            router_logits = gate(hidden_states)
            token_final_scales, token_selected_experts = gate_apply(
                router_logits)

            logits_scratch = router_scratch["logits"].narrow(
                0, 0, input_tokens)
            topk_indices_scratch = router_scratch["topk_indices"].narrow(
                0, 0, input_tokens)
            topk_weights_scratch = router_scratch["topk_weights"].narrow(
                0, 0, input_tokens)
            logits_scratch.copy_(router_logits.to(logits_scratch.dtype))
            topk_indices_scratch.copy_(
                token_selected_experts.to(topk_indices_scratch.dtype))
            topk_weights_scratch.copy_(
                token_final_scales.to(topk_weights_scratch.dtype))
            self._scheduler_state.last_moe_router_reason = (
                "resident_moe_router_python_bridge_executed")
            return topk_weights_scratch
        except (AttributeError, RuntimeError, AssertionError, TypeError):
            self._scheduler_state.last_moe_router_reason = (
                "resident_moe_router_python_bridge_failed")
            return None

    def run_layer_post_ffn_rmsnorm_stage(
        self,
        *,
        layer_idx: int,
        invocation: Any,
        eps: float,
        use_gemma: bool = False,
    ) -> Any | None:
        native_handle = self._get_or_create_native_handle()
        if native_handle is None:
            return None
        run_post_ffn_rmsnorm = getattr(native_handle,
                                       "run_layer_post_ffn_rmsnorm", None)
        if not callable(run_post_ffn_rmsnorm):
            return None
        state = self._shape_states.get(invocation.stable_shape_key)
        if state is None:
            return None
        self._scheduler_state.post_ffn_rmsnorm_stage_calls += 1
        bridged = self._try_run_residual_rmsnorm_python_bridge(
            layer_idx=layer_idx,
            norm_name="next_layer_layernorm",
            input_tensor=state.scratch["dense_mlp_output_states"],
            residual_tensor=state.scratch["post_attention_residual_states"],
            output_tensor=state.scratch["next_layer_hidden_states"],
            residual_output_tensor=state.scratch["next_layer_residual_states"],
            input_tokens=invocation.input_tokens,
        )
        if bridged is not None:
            return bridged
        return run_post_ffn_rmsnorm(
            layer_idx,
            state.scratch["dense_mlp_output_states"],
            state.scratch["post_attention_residual_states"],
            state.scratch["next_layer_hidden_states"],
            state.scratch["next_layer_residual_states"],
            invocation.input_tokens,
            eps,
            use_gemma,
        )

    def run_lm_head_logits_stage(
        self,
        *,
        invocation: Any,
    ) -> Any | None:
        native_handle = self._get_or_create_native_handle()
        if native_handle is None:
            return None
        run_lm_head_logits = getattr(native_handle, "run_lm_head_logits",
                                     None)
        if not callable(run_lm_head_logits):
            return None
        state = self._shape_states.get(invocation.stable_shape_key)
        if state is None:
            return None
        self._scheduler_state.lm_head_logits_stage_calls += 1
        return run_lm_head_logits(
            state.scratch["next_layer_hidden_states"],
            state.scratch["logits"],
            invocation.input_tokens,
        )

    def run_greedy_sample_stage(
        self,
        *,
        invocation: Any,
    ) -> Any | None:
        native_handle = self._get_or_create_native_handle()
        if native_handle is None:
            self._scheduler_state.last_sampling_reason = (
                "resident_sampling_native_handle_missing")
            return None
        run_greedy_sample = getattr(native_handle, "run_greedy_sample", None)
        if not callable(run_greedy_sample):
            self._scheduler_state.last_sampling_reason = (
                "resident_sampling_native_method_missing")
            return None
        state = self._shape_states.get(invocation.stable_shape_key)
        if state is None:
            self._scheduler_state.last_sampling_reason = (
                "resident_sampling_native_shape_state_missing")
            return None
        try:
            sampled_tokens = run_greedy_sample(
                state.scratch["logits"],
                state.scratch["new_tokens"],
                invocation.input_tokens,
            )
        except RuntimeError:
            self._scheduler_state.last_sampling_reason = (
                "resident_sampling_native_failed")
            return None
        self._scheduler_state.sampling_stage_calls += 1
        self._scheduler_state.last_sampling_reason = (
            "resident_sampling_native_executed")
        return sampled_tokens

    def _window_stage_body_ready(
        self,
        native_handle: Any,
    ) -> bool:
        return self._window_stage_body_not_ready_reason(native_handle) is None

    def _window_stage_body_not_ready_reason(
        self,
        native_handle: Any,
    ) -> str | None:
        if not _window_stage_body_enabled():
            return "resident_window_stage_body_disabled"
        if native_handle is None:
            return "resident_window_stage_body_native_handle_missing"

        required_methods = (
            "run_decode_window_prepare_step",
            "run_decode_window_sample_step",
            "run_decode_window_advance_state",
            "run_layer_input_rmsnorm",
            "run_layer_input_gated_norm",
            "run_layer_attention_output_tail",
            "run_layer_post_attention_rmsnorm",
            "run_layer_post_attention_gated_norm",
            "run_layer_post_ffn_rmsnorm",
            "run_lm_head_logits",
            "run_decode_window_attention_metadata_device_refresh",
        )
        for method_name in required_methods:
            method = getattr(native_handle, method_name, None)
            if not callable(method):
                return f"resident_window_stage_body_{method_name}_missing"

        layer_contracts = _as_tuple(getattr(self._contract, "layers", ()))
        if not layer_contracts:
            return "resident_window_stage_body_layers_missing"
        if not self._native_handle_bool_method(
                native_handle, "run_layer_dsa_attention_dispatch_ready"):
            return self._native_dsa_attention_dispatch_not_ready_reason(
                native_handle)

        for layer_contract in layer_contracts:
            layer_kind = getattr(layer_contract, "layer_kind", None)
            if layer_kind == "dense":
                dense_mlp = getattr(native_handle, "run_layer_dense_mlp",
                                    None)
                if not callable(dense_mlp):
                    return "resident_window_stage_body_dense_mlp_missing"
                continue
            if layer_kind == "moe":
                moe_router = getattr(native_handle, "run_layer_moe_router",
                                     None)
                if not callable(moe_router):
                    return "resident_window_stage_body_moe_router_missing"
                native_moe_experts = getattr(native_handle,
                                             "run_layer_moe_experts", None)
                if callable(native_moe_experts):
                    moe_assets_reason = self._moe_expert_assets_not_ready_reason(
                    )
                    if moe_assets_reason is not None:
                        return moe_assets_reason
                    continue
                if self._precomputed_moe_experts_available():
                    continue
                return "resident_window_stage_body_moe_experts_missing"
                continue
            return "resident_window_stage_body_unsupported_layer_kind"
        return None

    def _window_body_not_ready_reason(
        self,
        native_handle: Any,
    ) -> str:
        native_reason = self._native_window_not_ready_reason(native_handle)
        stage_reason = self._window_stage_body_not_ready_reason(native_handle)
        if stage_reason is None:
            return native_reason
        return f"{native_reason}:{stage_reason}"

    def _native_window_not_ready_reason(
        self,
        native_handle: Any,
        default_reason: str | None = None,
    ) -> str:
        """Return the handle-owned native window blocker when available."""

        scheduler_reason = self._scheduler_state.last_window_reason
        current_reason = default_reason or scheduler_reason
        if default_reason is None and scheduler_reason not in {
                "resident_window_native_not_implemented",
                "resident_window_native_not_ready",
                "resident_window_native_method_missing",
        } and not scheduler_reason.startswith("resident_window_native_missing_"):
            return scheduler_reason
        reason_fn = (
            None if native_handle is None else
            getattr(native_handle, "run_decode_window_not_ready_reason", None))
        if not callable(reason_fn):
            return current_reason
        try:
            reason = reason_fn()
        except (RuntimeError, TypeError):
            return current_reason
        return reason if isinstance(reason, str) and reason else current_reason

    def run_decode_window_sample_step_stage(
        self,
        *,
        invocation: Any,
        owned_steps: int,
        output_step_idx: int,
    ) -> Any | None:
        """Sample logits into the resident window token scratch."""

        native_handle = self._get_or_create_native_handle()
        if native_handle is None:
            self._scheduler_state.last_window_sample_reason = (
                "resident_window_sample_native_handle_missing")
            return None
        sample_step = getattr(native_handle, "run_decode_window_sample_step",
                              None)
        if not callable(sample_step):
            self._scheduler_state.last_window_sample_reason = (
                "resident_window_sample_native_method_missing")
            return None
        state = self._shape_states.get(invocation.stable_shape_key)
        if state is None:
            self._scheduler_state.last_window_sample_reason = (
                "resident_window_sample_shape_state_missing")
            return None
        window_tokens = self._ensure_window_new_tokens_scratch(
            state, owned_steps)
        if window_tokens is None:
            self._scheduler_state.last_window_sample_reason = (
                "resident_window_sample_token_scratch_missing")
            return None
        _window_trace(
            "native_sample_call_begin",
            {
                "output_step_idx": output_step_idx,
                "owned_steps": owned_steps,
                "logits": _tensor_summary(state.scratch.get("logits")),
                "window_tokens": _tensor_summary(window_tokens),
                "input_tokens": int(getattr(invocation, "input_tokens", 0)
                                    or 0),
            },
        )
        start_time = time.perf_counter()
        try:
            sampled_tokens = sample_step(
                state.scratch["logits"],
                window_tokens,
                int(output_step_idx),
                invocation.input_tokens,
            )
        except RuntimeError as exc:
            self._scheduler_state.last_window_sample_reason = (
                _exception_reason("resident_window_sample_native_failed", exc))
            _window_trace(
                "native_sample_call_failed",
                {
                    "output_step_idx":
                    output_step_idx,
                    "elapsed_ms":
                    (time.perf_counter() - start_time) * 1000.0,
                    "reason":
                    self._scheduler_state.last_window_sample_reason,
                },
            )
            return None
        self._scheduler_state.window_sample_stage_calls += 1
        self._scheduler_state.last_window_sample_reason = (
            "resident_window_sample_native_executed")
        _window_trace(
            "native_sample_call_end",
            {
                "output_step_idx":
                output_step_idx,
                "elapsed_ms":
                (time.perf_counter() - start_time) * 1000.0,
                "sampled_tokens":
                _tensor_summary(sampled_tokens),
            },
        )
        return sampled_tokens

    def run_decode_window_prepare_step_stage(
        self,
        *,
        invocation: Any,
        initial_tokens: Any,
        input_ids: Any,
        position_ids: Any,
        kv_lens_cuda: Any,
        owned_steps: int,
        output_step_idx: int,
    ) -> Any | None:
        """Feed a window token into the next resident model step."""

        native_handle = self._get_or_create_native_handle()
        if native_handle is None:
            self._scheduler_state.last_window_prepare_reason = (
                "resident_window_prepare_native_handle_missing")
            return None
        prepare_step = getattr(native_handle, "run_decode_window_prepare_step",
                               None)
        if not callable(prepare_step):
            self._scheduler_state.last_window_prepare_reason = (
                "resident_window_prepare_native_method_missing")
            return None
        state = self._get_shape_state(invocation, input_ids)
        window_tokens = self._ensure_window_new_tokens_scratch(
            state, owned_steps)
        if window_tokens is None:
            self._scheduler_state.last_window_prepare_reason = (
                "resident_window_prepare_token_scratch_missing")
            return None
        _window_trace(
            "native_prepare_call_begin",
            {
                "output_step_idx": output_step_idx,
                "owned_steps": owned_steps,
                "initial_tokens": _tensor_summary(initial_tokens),
                "window_tokens": _tensor_summary(window_tokens),
                "input_ids": _tensor_summary(input_ids),
                "hidden_states": _tensor_summary(
                    state.scratch.get("hidden_states")),
                "position_ids": _tensor_summary(position_ids),
                "kv_lens_cuda": _tensor_summary(kv_lens_cuda),
                "input_tokens": int(getattr(invocation, "input_tokens", 0)
                                    or 0),
            },
        )
        start_time = time.perf_counter()
        try:
            prepared_hidden = prepare_step(
                initial_tokens,
                window_tokens,
                input_ids,
                state.scratch["hidden_states"],
                position_ids,
                kv_lens_cuda,
                int(output_step_idx),
                invocation.input_tokens,
            )
        except RuntimeError as exc:
            self._scheduler_state.last_window_prepare_reason = (
                _exception_reason("resident_window_prepare_native_failed", exc))
            _window_trace(
                "native_prepare_call_failed",
                {
                    "output_step_idx":
                    output_step_idx,
                    "elapsed_ms":
                    (time.perf_counter() - start_time) * 1000.0,
                    "reason":
                    self._scheduler_state.last_window_prepare_reason,
                },
            )
            return None
        self._scheduler_state.window_prepare_stage_calls += 1
        self._scheduler_state.last_window_prepare_reason = (
            "resident_window_prepare_native_executed")
        _window_trace(
            "native_prepare_call_end",
            {
                "output_step_idx":
                output_step_idx,
                "elapsed_ms":
                (time.perf_counter() - start_time) * 1000.0,
                "prepared_hidden":
                _tensor_summary(prepared_hidden),
            },
        )
        return prepared_hidden

    def _outputs_with_greedy_sample(
        self,
        *,
        logits: Any,
        invocation: Any,
    ) -> dict[str, Any]:
        outputs = {"logits": logits}
        sampled_tokens = self.run_greedy_sample_stage(invocation=invocation)
        if sampled_tokens is not None:
            outputs["resident_sample_device_new_tokens"] = sampled_tokens
        return outputs

    def _get_shape_state(
        self,
        invocation: Any,
        input_ids: Any,
    ) -> DeepSeekResidentShapeState:
        shape_key = invocation.stable_shape_key
        state = self._shape_states.get(shape_key)
        if state is not None:
            return state

        dtype = _model_activation_dtype(self._model, input_ids)
        device = getattr(input_ids, "device", None)
        state = DeepSeekResidentShapeState(
            shape_key=shape_key,
            batch_size=invocation.padded_batch_size,
            hidden_size=getattr(self._contract, "hidden_size", None),
            vocab_size=getattr(self._contract, "vocab_size", None),
            device=device,
            dtype=dtype,
        )
        state.scratch.update(_allocate_scratch(state, input_ids))
        self._shape_states[shape_key] = state
        return state

    def _ensure_window_new_tokens_scratch(
        self,
        state: DeepSeekResidentShapeState,
        owned_steps: int,
    ) -> Any | None:
        window_new_tokens = state.scratch.get("window_new_tokens")
        shape = getattr(window_new_tokens, "shape", None)
        if (window_new_tokens is not None and shape is not None
                and len(shape) == 3 and int(shape[0]) >= owned_steps
                and int(shape[1]) == state.batch_size and int(shape[2]) == 1):
            return window_new_tokens
        torch = _import_torch()
        if torch is None:
            return None
        window_new_tokens = torch.empty(
            (owned_steps, state.batch_size, 1),
            device=state.device,
            dtype=_torch_int32_dtype(torch),
        )
        state.scratch["window_new_tokens"] = window_new_tokens
        return window_new_tokens

    def _ensure_attention_output_tail_scratch(
        self,
        state: DeepSeekResidentShapeState,
        layer_idx: int,
    ) -> dict[str, Any] | None:
        attention_output_tail_states = state.scratch.setdefault(
            "attention_output_tail_states", {})
        if layer_idx in attention_output_tail_states:
            return attention_output_tail_states[layer_idx]
        torch = _import_torch()
        if torch is None:
            return None
        attention_output_size = self._attention_output_size(layer_idx)
        if attention_output_size is None:
            return None
        scratch = {
            "attention_core_output":
            torch.empty((state.batch_size, attention_output_size),
                        device=state.device,
                        dtype=state.dtype),
            "attention_gate":
            torch.empty((state.batch_size, attention_output_size),
                        device=state.device,
                        dtype=state.dtype),
        }
        attention_output_tail_states[layer_idx] = scratch
        return scratch

    def _attention_output_size(self, layer_idx: int) -> int | None:
        gate_shape = self._layer_site_shape(layer_idx,
                                            _SITE_ATTN_GATE_PROJ_WEIGHT)
        o_proj_shape = self._layer_site_shape(layer_idx,
                                              _SITE_ATTN_O_PROJ_WEIGHT)
        if gate_shape and len(gate_shape) == 2:
            return int(gate_shape[0])
        if o_proj_shape and len(o_proj_shape) == 2:
            return int(o_proj_shape[1])
        return None

    def _ensure_dense_mlp_scratch(
        self,
        state: DeepSeekResidentShapeState,
        layer_idx: int,
    ) -> Any | None:
        dense_mlp_intermediate_states = state.scratch.setdefault(
            "dense_mlp_intermediate_states", {})
        if layer_idx in dense_mlp_intermediate_states:
            return dense_mlp_intermediate_states[layer_idx]
        torch = _import_torch()
        if torch is None:
            return None
        intermediate_size = self._dense_mlp_intermediate_size(layer_idx)
        if intermediate_size is None:
            return None
        dense_mlp_intermediate_states[layer_idx] = torch.empty(
            (state.batch_size, intermediate_size),
            device=state.device,
            dtype=state.dtype,
        )
        return dense_mlp_intermediate_states[layer_idx]

    def _dense_mlp_intermediate_size(self, layer_idx: int) -> int | None:
        gate_up_shape = self._layer_site_shape(layer_idx,
                                               _SITE_DENSE_MLP_GATE_UP_WEIGHT)
        down_shape = self._layer_site_shape(layer_idx, _SITE_DENSE_MLP_DOWN_WEIGHT)
        if gate_up_shape and len(gate_up_shape) == 2 and gate_up_shape[0] % 2 == 0:
            return int(gate_up_shape[0] // 2)
        if down_shape and len(down_shape) == 2:
            return int(down_shape[1])
        return None

    def _ensure_moe_router_scratch(
        self,
        state: DeepSeekResidentShapeState,
        layer_idx: int,
    ) -> dict[str, Any] | None:
        moe_router_states = state.scratch.setdefault("moe_router_states", {})
        if layer_idx in moe_router_states:
            return moe_router_states[layer_idx]
        torch = _import_torch()
        if torch is None:
            return None
        gate_shape = self._layer_site_shape(layer_idx, _SITE_MOE_GATE_WEIGHT)
        if gate_shape is None or len(gate_shape) != 2:
            return None
        router_config = self._moe_router_config(layer_idx)
        if router_config is None:
            return None
        num_experts = int(gate_shape[0])
        top_k = int(router_config["top_k"])
        scratch = {
            "logits":
            torch.empty((state.batch_size, num_experts),
                        device=state.device,
                        dtype=_torch_float32_dtype(torch)),
            "scores":
            torch.empty((state.batch_size, num_experts),
                        device=state.device,
                        dtype=_torch_float32_dtype(torch)),
            "topk_indices":
            torch.empty((state.batch_size, top_k),
                        device=state.device,
                        dtype=_torch_int32_dtype(torch)),
            "topk_weights":
            torch.empty((state.batch_size, top_k),
                        device=state.device,
                        dtype=_torch_float32_dtype(torch)),
        }
        moe_router_states[layer_idx] = scratch
        return scratch

    def _moe_router_config(self, layer_idx: int) -> dict[str, Any] | None:
        layer_contract = self._layer_contract(layer_idx)
        if layer_contract is None:
            return None
        if getattr(layer_contract, "layer_kind", None) != "moe":
            return None
        top_k = getattr(layer_contract, "top_k", None) or getattr(
            self._contract, "num_experts_per_tok", None)
        if top_k is None:
            return None
        n_group = getattr(self._contract, "n_group", None) or 1
        topk_group = getattr(self._contract, "topk_group", None) or n_group
        routed_scaling_factor = getattr(self._contract,
                                        "routed_scaling_factor", None)
        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0
        return {
            "top_k": int(top_k),
            "n_group": int(n_group),
            "topk_group": int(topk_group),
            "routed_scaling_factor": float(routed_scaling_factor),
        }

    def _layer_contract(self, layer_idx: int) -> Any | None:
        for layer_contract in _as_tuple(getattr(self._contract, "layers", ())):
            if int(getattr(layer_contract, "layer_idx", -1)) == layer_idx:
                return layer_contract
        return None

    def _model_layer(self, layer_idx: int) -> Any | None:
        model_body = getattr(self._model, "model", None)
        layers = _as_tuple(getattr(model_body, "layers", ()))
        if layer_idx < 0 or layer_idx >= len(layers):
            return None
        return layers[layer_idx]

    def _precomputed_moe_experts_available(self) -> bool:
        for layer_contract in _as_tuple(getattr(self._contract, "layers", ())):
            if getattr(layer_contract, "layer_kind", None) != "moe":
                continue
            layer_idx = int(getattr(layer_contract, "layer_idx", -1))
            layer = self._model_layer(layer_idx)
            moe = None if layer is None else getattr(layer, "mlp", None)
            experts = None if moe is None else getattr(moe, "experts", None)
            shared_experts = (
                None if moe is None else getattr(moe, "shared_experts", None))
            forward_precomputed_route = (
                None if experts is None else
                getattr(experts, "forward_precomputed_route", None))
            if shared_experts is None or not callable(
                    forward_precomputed_route):
                return False
        return True

    def _dsa_indexer_assets_not_ready_reason(self) -> str | None:
        for layer_contract in _as_tuple(getattr(self._contract, "layers", ())):
            layer_idx = int(getattr(layer_contract, "layer_idx", -1))
            layer = self._model_layer(layer_idx)
            self_attn = None if layer is None else getattr(layer, "self_attn",
                                                           None)
            mqa = None if self_attn is None else getattr(self_attn, "mqa", None)
            indexer = None if mqa is None else getattr(mqa, "indexer", None)
            if indexer is None:
                return f"layer_{layer_idx}_dsa_indexer_missing"
            if bool(getattr(indexer, "skip_topk", False)):
                continue
            required_sites = (
                (_SITE_ATTN_INDEXER_WQ_B_WEIGHT, "wq_b.weight"),
                (_SITE_ATTN_INDEXER_WK_WEIGHT, "wk.weight"),
                (_SITE_ATTN_INDEXER_WEIGHTS_PROJ_WEIGHT,
                 "weights_proj.weight"),
                (_SITE_ATTN_INDEXER_K_NORM_WEIGHT, "k_norm.weight"),
                (_SITE_ATTN_INDEXER_ROTARY_COS_SIN,
                 "rotary_emb.rotary_cos_sin"),
            )
            for site_id, name in required_sites:
                if self._layer_site_shape(layer_idx, site_id) is None:
                    return f"layer_{layer_idx}_dsa_indexer_{name}_missing"
            quantized_modules = (
                ("wq_b", getattr(indexer, "wq_b", None),
                 _SITE_ATTN_INDEXER_WQ_B_WEIGHT_SCALE,
                 _SITE_ATTN_INDEXER_WQ_B_INPUT_SCALE,
                 _SITE_ATTN_INDEXER_WQ_B_ALPHA),
                ("wk", getattr(indexer, "wk", None),
                 _SITE_ATTN_INDEXER_WK_WEIGHT_SCALE,
                 _SITE_ATTN_INDEXER_WK_INPUT_SCALE,
                 _SITE_ATTN_INDEXER_WK_ALPHA),
                ("weights_proj", getattr(indexer, "weights_proj", None),
                 _SITE_ATTN_INDEXER_WEIGHTS_PROJ_WEIGHT_SCALE,
                 _SITE_ATTN_INDEXER_WEIGHTS_PROJ_INPUT_SCALE,
                 _SITE_ATTN_INDEXER_WEIGHTS_PROJ_ALPHA),
            )
            for name, module, scale_site, input_scale_site, alpha_site in (
                    quantized_modules):
                if not bool(getattr(module, "has_nvfp4", False)):
                    continue
                for site_id, suffix in (
                    (scale_site, "weight_scale"),
                    (input_scale_site, "input_scale"),
                    (alpha_site, "alpha"),
                ):
                    if self._layer_site_shape(layer_idx, site_id) is None:
                        return (
                            f"layer_{layer_idx}_dsa_indexer_{name}_{suffix}_missing"
                        )
        return None

    def _moe_expert_assets_not_ready_reason(self) -> str | None:
        for layer_contract in _as_tuple(getattr(self._contract, "layers", ())):
            if getattr(layer_contract, "layer_kind", None) != "moe":
                continue
            layer_idx = int(getattr(layer_contract, "layer_idx", -1))
            required_sites = (
                (_SITE_SHARED_EXPERT_GATE_UP_WEIGHT,
                 "shared_experts.gate_up_proj.weight"),
                (_SITE_SHARED_EXPERT_DOWN_WEIGHT,
                 "shared_experts.down_proj.weight"),
                (_SITE_EXPERT_GATE_UP_WEIGHT, "experts.gate_up_proj_weight"),
                (_SITE_EXPERT_DOWN_WEIGHT, "experts.down_proj_weight"),
            )
            for site_id, name in required_sites:
                if self._layer_site_shape(layer_idx, site_id) is None:
                    return f"layer_{layer_idx}_moe_{name}_missing"
        return None

    def _precomputed_moe_experts_not_ready_reason(self) -> str:
        for layer_contract in _as_tuple(getattr(self._contract, "layers", ())):
            if getattr(layer_contract, "layer_kind", None) != "moe":
                continue
            layer_idx = int(getattr(layer_contract, "layer_idx", -1))
            layer = self._model_layer(layer_idx)
            if layer is None:
                return f"layer_{layer_idx}_moe_layer_missing"
            moe = getattr(layer, "mlp", None)
            if moe is None:
                return f"layer_{layer_idx}_moe_module_missing"
            if getattr(moe, "shared_experts", None) is None:
                return f"layer_{layer_idx}_moe_shared_experts_missing"
            experts = getattr(moe, "experts", None)
            forward_precomputed_route = (
                None if experts is None else
                getattr(experts, "forward_precomputed_route", None))
            if not callable(forward_precomputed_route):
                return f"layer_{layer_idx}_moe_precomputed_route_missing"
        return "moe_precomputed_route_missing"

    def _rms_norm_eps(self) -> float:
        return float(getattr(self._contract, "rms_norm_eps", None) or 1e-6)

    def _promote_next_layer_state(
        self,
        state: DeepSeekResidentShapeState,
    ) -> None:
        state.scratch["hidden_states"], state.scratch[
            "next_layer_hidden_states"] = (
                state.scratch["next_layer_hidden_states"],
                state.scratch["hidden_states"],
            )
        state.scratch["current_residual_states"] = state.scratch[
            "next_layer_residual_states"]

    def _moe_bridge_do_finalize(
        self,
        layer: Any,
        moe: Any,
        hidden_states: Any,
    ) -> bool:
        mapping = getattr(moe, "mapping", None)
        is_multi_node = getattr(mapping, "is_multi_node", None)
        if callable(is_multi_node) and is_multi_node():
            return True
        fusion_config = getattr(layer, "fusion_config", None)
        if not bool(getattr(fusion_config, "POST_MOE_FUSION", False)):
            return True
        model_config = getattr(layer, "model_config", None)
        moe_backend = getattr(model_config, "moe_backend", None)
        experts = getattr(moe, "experts", None)
        if moe_backend != "TRTLLM":
            return True
        if not bool(getattr(experts, "has_nvfp4", False)):
            return True
        is_p2p_supported = bool(getattr(layer, "is_p2p_supported", False))
        if not is_p2p_supported:
            return True
        max_token = getattr(getattr(layer, "moe_allreduce", None),
                            "max_token", None)
        if max_token is None:
            return True
        hidden_shape = getattr(hidden_states, "shape", ())
        if len(hidden_shape) == 0 or int(hidden_shape[0]) > int(max_token):
            return True
        return False

    def _run_deferred_moe_allreduce_bridge(
        self,
        *,
        layer: Any,
        moe: Any,
        experts: Any,
        shared_experts: Any,
        hidden_states: Any,
        router_logits: Any,
        all_rank_num_tokens: Any,
        state: DeepSeekResidentShapeState,
    ) -> bool:
        if bool(getattr(moe, "use_dp", False)):
            state.scratch["last_moe_experts_reason"] = (
                "post_moe_fusion_deferred_dp_unsupported")
            return False
        moe_allreduce = getattr(layer, "moe_allreduce", None)
        if moe_allreduce is None or not callable(moe_allreduce):
            state.scratch["last_moe_experts_reason"] = (
                "post_moe_fusion_allreduce_missing")
            return False
        next_layer_layernorm = getattr(layer, "next_layer_layernorm", None)
        norm_weight = getattr(next_layer_layernorm, "weight", None)
        eps = getattr(next_layer_layernorm, "variance_epsilon", None)
        if norm_weight is None or eps is None:
            state.scratch["last_moe_experts_reason"] = (
                "post_moe_fusion_next_layer_norm_missing")
            return False
        moe_all_reduce_params_cls = _resolve_moe_all_reduce_params_class()
        if moe_all_reduce_params_cls is None:
            state.scratch["last_moe_experts_reason"] = (
                "post_moe_fusion_params_missing")
            return False

        try:
            routed_output = experts(
                hidden_states,
                router_logits,
                do_finalize=False,
                output_dtype=getattr(hidden_states, "dtype", None),
                all_rank_num_tokens=all_rank_num_tokens,
                use_dp_padding=False,
                **_wide_ep_moe_kwargs(experts),
            )
            shared_output = shared_experts(hidden_states)
            shared_output_scale = getattr(moe, "shared_output_scale", None)
            if shared_output_scale is not None:
                shared_output *= shared_output_scale
            if (not isinstance(routed_output, (list, tuple))
                    or len(routed_output) != 3):
                state.scratch["last_moe_experts_reason"] = (
                    "post_moe_fusion_deferred_output_contract_failed")
                return False
            fc2_output, expert_scale_factor, expanded_idx_to_permuted_idx = (
                routed_output)
            all_reduce_params = moe_all_reduce_params_cls(
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                expert_scale_factor=expert_scale_factor,
                shared_expert_output=shared_output,
                residual=state.scratch["post_attention_residual_states"],
                norm_weight=norm_weight,
                eps=eps,
                is_cutlass_min_latency=False,
            )
            allreduce_output = moe_allreduce(
                fc2_output, all_reduce_params=all_reduce_params)
            if (not isinstance(allreduce_output, (list, tuple))
                    or len(allreduce_output) != 2):
                state.scratch["last_moe_experts_reason"] = (
                    "post_moe_fusion_allreduce_output_contract_failed")
                return False
        except (AttributeError, RuntimeError, AssertionError, TypeError):
            state.scratch["last_moe_experts_reason"] = (
                "post_moe_fusion_deferred_bridge_failed")
            return False

        next_hidden_states, next_residual_states = allreduce_output
        state.scratch["next_layer_hidden_states"] = next_hidden_states
        state.scratch["next_layer_residual_states"] = next_residual_states
        state.scratch["moe_post_ffn_finalized"] = True
        state.scratch["last_moe_experts_reason"] = (
            "post_moe_fusion_deferred_allreduce_executed")
        return True

    def _combine_moe_outputs(
        self,
        *,
        moe: Any,
        layer: Any,
        shared_output: Any,
        routed_output: Any,
    ) -> Any:
        final_hidden_states = self._add_tensors(shared_output, routed_output)
        mapping = getattr(moe, "mapping", None)
        tp_size = int(getattr(mapping, "tp_size", 1) or 1)
        if bool(getattr(moe, "use_dp", False)) or tp_size <= 1:
            return final_hidden_states
        allreduce = getattr(moe, "allreduce", None)
        if allreduce is None:
            return final_hidden_states
        return allreduce(final_hidden_states,
                         all_reduce_params=self._moe_final_allreduce_params(
                             layer, tp_size))

    def _add_tensors(self, left: Any, right: Any) -> Any:
        if hasattr(left, "add_"):
            return left.add_(right)
        return left + right

    def _moe_final_allreduce_params(
        self,
        layer: Any,
        tp_size: int,
    ) -> Any | None:
        all_reduce_params_cls = _resolve_all_reduce_params_class()
        if all_reduce_params_cls is None:
            return None
        fusion_config = getattr(layer, "fusion_config", None)
        enable_allreduce = not (bool(
            getattr(fusion_config, "POST_MOE_FUSION", False)) or tp_size == 1)
        return all_reduce_params_cls(enable_allreduce=enable_allreduce)

    def _dense_mlp_final_allreduce_params(self, layer: Any) -> Any | None:
        all_reduce_params_cls = _resolve_all_reduce_params_class()
        if all_reduce_params_cls is None:
            return None
        fusion_config = getattr(layer, "fusion_config", None)
        mlp_tp_size = int(getattr(layer, "mlp_tp_size", 1) or 1)
        enable_allreduce = not (bool(
            getattr(fusion_config, "POST_MLP_FUSION", False))
                                or mlp_tp_size == 1)
        return all_reduce_params_cls(enable_allreduce=enable_allreduce)

    def _attention_allreduce_params(self, layer: Any) -> Any | None:
        all_reduce_params_cls = _resolve_all_reduce_params_class()
        if all_reduce_params_cls is None:
            return None
        return all_reduce_params_cls(
            enable_allreduce=not bool(
                getattr(layer, "disable_attn_allreduce", False)))

    def _stage_scheduler_result(
        self,
        *,
        completed: bool,
        reason: str,
        stage: str,
        layer_idx: int | None = None,
        completed_layers: int = 0,
        outputs: Any | None = None,
    ) -> DeepSeekResidentStepResult:
        if completed:
            self._scheduler_state.stage_scheduler_completions += 1
        else:
            self._scheduler_state.stage_scheduler_declines += 1
        self._scheduler_state.last_stage_scheduler_reason = reason
        self._scheduler_state.last_execution_reason = reason
        return DeepSeekResidentStepResult(
            completed=completed,
            reason=reason,
            stage=stage,
            layer_idx=layer_idx,
            completed_layers=completed_layers,
            outputs=outputs,
        )

    def _window_scheduler_result(
        self,
        *,
        completed: bool,
        reason: str,
    ) -> None:
        if completed:
            self._scheduler_state.window_scheduler_completions += 1
        else:
            self._scheduler_state.window_scheduler_declines += 1
        self._scheduler_state.last_window_scheduler_reason = reason
        self._scheduler_state.last_window_reason = reason

    def _layer_site_shape(
        self,
        layer_idx: int,
        site_id: int,
    ) -> tuple[int, ...] | None:
        for site in self._assets.layer_tensor_sites:
            if site.layer_idx != layer_idx or site.site_id != site_id:
                continue
            return self._assets.tensor_specs[site.tensor_idx].shape
        return None

    def _get_native_op(self) -> Any | None:
        if self._native_op_checked:
            return self._native_op
        self._native_op_checked = True
        if not _native_op_ready(self._native_op_name):
            return None
        self._native_op = _resolve_native_op(self._native_op_name)
        return self._native_op

    def _get_or_create_native_handle(self) -> Any | None:
        if self._native_handle_checked:
            return self._native_handle
        self._native_handle_checked = True
        handle_cls = _resolve_native_class(
            "trtllm.DeepseekResidentDecodeHandle")
        if handle_cls is None:
            self._scheduler_state.handle_creation_reason = (
                "missing_handle_class")
            return None
        try:
            self._native_handle = handle_cls(
                self._resident_layer_offsets,
                self._resident_layer_kinds,
                self._resident_layer_site_offsets,
                self._resident_layer_site_ids,
                self._resident_layer_site_tensor_indices,
                self._resident_tensors,
            )
        except RuntimeError as exc:
            self._scheduler_state.handle_creation_reason = (
                f"create_failed:{type(exc).__name__}")
            return None
        self._scheduler_state.handle_created = True
        self._scheduler_state.handle_creation_reason = "created"
        self._scheduler_state.manifest_validated = True
        self._scheduler_state.manifest_validation_reason = (
            "validated_by_handle")
        return self._native_handle

    def _maybe_validate_manifest(self) -> None:
        if self._scheduler_state.manifest_validated:
            return
        if os.environ.get(_VALIDATE_MANIFEST_ENV_NAME, "0") != "1":
            return
        prepare_op = _resolve_native_op(
            f"{self._native_op_name}_prepare")
        if prepare_op is None:
            self._scheduler_state.manifest_validation_reason = (
                "missing_prepare_op")
            return
        try:
            valid = bool(
                prepare_op(
                    self._resident_layer_offsets,
                    self._resident_layer_kinds,
                    self._resident_tensors,
                ))
        except RuntimeError as exc:
            self._scheduler_state.manifest_validation_reason = (
                f"prepare_failed:{type(exc).__name__}")
            return
        self._scheduler_state.manifest_validated = valid
        self._scheduler_state.manifest_validation_reason = (
            "validated" if valid else "prepare_returned_false")


def _allocate_scratch(
    state: DeepSeekResidentShapeState,
    input_ids: Any,
) -> dict[str, Any]:
    torch = _import_torch()
    if torch is None:
        return {}
    device = state.device if state.device is not None else getattr(
        input_ids, "device", None)
    dtype = state.dtype
    hidden_size = state.hidden_size or 1
    vocab_size = state.vocab_size or 1
    return {
        "hidden_states":
        torch.empty((state.batch_size, hidden_size),
                    device=device,
                    dtype=dtype),
        "norm_hidden_states":
        torch.empty((state.batch_size, hidden_size),
                    device=device,
                    dtype=dtype),
        "gated_hidden_states":
        torch.empty((state.batch_size, hidden_size),
                    device=device,
                    dtype=dtype),
        "attention_hidden_states":
        torch.empty((state.batch_size, hidden_size),
                    device=device,
                    dtype=dtype),
        "post_attention_norm_hidden_states":
        torch.empty((state.batch_size, hidden_size),
                    device=device,
                    dtype=dtype),
        "post_attention_gated_hidden_states":
        torch.empty((state.batch_size, hidden_size),
                    device=device,
                    dtype=dtype),
        "post_attention_residual_states":
        torch.empty((state.batch_size, hidden_size),
                    device=device,
                    dtype=dtype),
        "dense_mlp_output_states":
        torch.empty((state.batch_size, hidden_size),
                    device=device,
                    dtype=dtype),
        "next_layer_hidden_states":
        torch.empty((state.batch_size, hidden_size),
                    device=device,
                    dtype=dtype),
        "next_layer_residual_states":
        torch.empty((state.batch_size, hidden_size),
                    device=device,
                    dtype=dtype),
        "logits":
        torch.empty((state.batch_size, vocab_size),
                    device=device,
                    dtype=dtype),
        "new_tokens":
        torch.empty((1, state.batch_size, 1),
                    device=device,
                    dtype=_torch_int32_dtype(torch)),
    }


def _model_activation_dtype(model: Any, input_ids: Any) -> Any:
    try:
        for parameter in model.parameters():
            return parameter.dtype
    except (AttributeError, TypeError):
        pass
    return getattr(input_ids, "dtype", None)


def _sample_state_new_tokens(sample_state: Any) -> Any | None:
    device_state = getattr(sample_state, "device", None)
    device_tokens = (
        None if device_state is None else
        getattr(device_state, "new_tokens", None))
    if device_tokens is not None:
        return device_tokens
    host_state = getattr(sample_state, "host", None)
    return (
        None if host_state is None else getattr(host_state, "new_tokens", None))


def _slice_first_dim(tensor: Any, length: int) -> Any:
    try:
        return tensor[:length, ...]
    except (TypeError, AttributeError, IndexError):
        shape = getattr(tensor, "shape", None)
        if shape is not None and len(shape) >= 1 and int(shape[0]) == int(length):
            return tensor
        raise


def build_deepseek_resident_model_assets(
    model: Any,
    contract: Any,
) -> DeepSeekResidentModelAssets:
    """Build the stable tensor table for a resident DeepSeek decode body."""

    builder = _DeepSeekResidentAssetBuilder()
    model_body = getattr(model, "model", None)

    builder.add_module("model.embed_tokens", getattr(model_body, "embed_tokens", None))
    builder.add_module("model.norm", getattr(model_body, "norm", None))
    builder.add_module("lm_head", getattr(model, "lm_head", None))

    layer_assets: list[DeepSeekResidentLayerAssets] = []
    layer_tensor_sites: list[DeepSeekResidentLayerTensorSite] = []
    layer_site_offsets: list[int] = []
    layers = _as_tuple(getattr(model_body, "layers", ()))
    for layer_contract in _as_tuple(getattr(contract, "layers", ())):
        layer_idx = int(getattr(layer_contract, "layer_idx", len(layer_assets)))
        if layer_idx < 0 or layer_idx >= len(layers):
            continue
        layer = layers[layer_idx]
        start = len(builder.assets)
        layer_prefix = f"model.layers.{layer_idx}"
        builder.add_module(f"{layer_prefix}.input_layernorm",
                           getattr(layer, "input_layernorm", None))
        builder.add_module(f"{layer_prefix}.post_attention_layernorm",
                           getattr(layer, "post_attention_layernorm", None))
        builder.add_module(f"{layer_prefix}.next_layer_layernorm",
                           getattr(layer, "next_layer_layernorm", None))
        builder.add_module(f"{layer_prefix}.input_gated_norm_down",
                           getattr(layer, "input_gated_norm_down", None))
        builder.add_module(f"{layer_prefix}.input_gated_norm_up",
                           getattr(layer, "input_gated_norm_up", None))
        builder.add_module(f"{layer_prefix}.post_attention_gated_norm_down",
                           getattr(layer, "post_attention_gated_norm_down", None))
        builder.add_module(f"{layer_prefix}.post_attention_gated_norm_up",
                           getattr(layer, "post_attention_gated_norm_up", None))
        self_attn = getattr(layer, "self_attn", None)
        builder.add_module(f"{layer_prefix}.self_attn", self_attn)
        builder.add_tensor(
            f"{layer_prefix}.self_attn.mqa.indexer.rotary_emb.rotary_cos_sin",
            _nested_attr(self_attn,
                         ("mqa", "indexer", "rotary_emb",
                          "rotary_cos_sin")),
        )
        builder.add_module(f"{layer_prefix}.mlp", getattr(layer, "mlp", None))
        stop = len(builder.assets)
        layer_site_offsets.append(len(layer_tensor_sites))
        layer_tensor_sites.extend(
            _layer_tensor_sites(
                layer_idx=layer_idx,
                layer_prefix=layer_prefix,
                tensor_specs=builder.assets,
                start=start,
                stop=stop,
            ))
        layer_assets.append(
            DeepSeekResidentLayerAssets(
                layer_idx=layer_idx,
                layer_kind=str(getattr(layer_contract, "layer_kind", "unknown")),
                start=start,
                stop=stop,
                tensor_names=tuple(asset.name for asset in builder.assets[start:stop]),
            ))

    return DeepSeekResidentModelAssets(
        tensors=tuple(asset.tensor for asset in builder.assets),
        tensor_names=tuple(asset.name for asset in builder.assets),
        tensor_specs=tuple(builder.assets),
        layers=tuple(layer_assets),
        layer_offsets=_layer_offsets(layer_assets, len(builder.assets)),
        layer_kinds=tuple(
            _layer_kind_id(layer.layer_kind) for layer in layer_assets),
        layer_tensor_sites=tuple(layer_tensor_sites),
        layer_site_offsets=_layer_site_offsets(layer_site_offsets,
                                               len(layer_tensor_sites)),
        layer_site_ids=tuple(site.site_id for site in layer_tensor_sites),
        layer_site_tensor_indices=tuple(site.tensor_idx
                                        for site in layer_tensor_sites),
    )


class _DeepSeekResidentAssetBuilder:

    def __init__(self) -> None:
        self.assets: list[DeepSeekResidentTensorAsset] = []
        self._asset_names: set[str] = set()

    def add_module(self, prefix: str, module: Any) -> None:
        if module is None:
            return
        found_tensor = False
        for name, tensor in _iter_module_tensors(module):
            found_tensor = True
            self.add_tensor(f"{prefix}.{name}", tensor)
        if not found_tensor and _is_tensor_like(module):
            self.add_tensor(prefix, module)

    def add_tensor(self, name: str, tensor: Any) -> None:
        if name in self._asset_names or not _is_tensor_like(tensor):
            return
        self._asset_names.add(name)
        self.assets.append(
            DeepSeekResidentTensorAsset(
                name=name,
                tensor=tensor,
                shape=_tensor_shape(tensor),
                dtype=str(getattr(tensor, "dtype", "")),
                device=str(getattr(tensor, "device", "")),
            ))


def _iter_module_tensors(module: Any) -> tuple[tuple[str, Any], ...]:
    tensors: list[tuple[str, Any]] = []
    seen_names: set[str] = set()
    for iterator_name in ("named_parameters", "named_buffers"):
        iterator = getattr(module, iterator_name, None)
        if not callable(iterator):
            continue
        try:
            items = iterator(recurse=True)
        except TypeError:
            items = iterator()
        except RuntimeError:
            continue
        for name, tensor in items:
            if name in seen_names or not _is_tensor_like(tensor):
                continue
            seen_names.add(name)
            tensors.append((name, tensor))

    for attr_name in _FALLBACK_TENSOR_ATTRS:
        if attr_name in seen_names:
            continue
        tensor = getattr(module, attr_name, None)
        if not _is_tensor_like(tensor):
            continue
        seen_names.add(attr_name)
        tensors.append((attr_name, tensor))

    for child_name, child_module in _iter_fallback_child_modules(module):
        for name, tensor in _iter_module_tensors(child_module):
            qualified_name = f"{child_name}.{name}"
            if qualified_name in seen_names:
                continue
            seen_names.add(qualified_name)
            tensors.append((qualified_name, tensor))
    return tuple(tensors)


def _nested_attr(root: Any, path: tuple[str, ...]) -> Any | None:
    value = root
    for name in path:
        if value is None:
            return None
        value = getattr(value, name, None)
    return value


_FALLBACK_TENSOR_ATTRS = (
    "weight",
    "bias",
    "weight_scale",
    "weight_scale_2",
    "input_scale",
    "inv_input_scale",
    "e_score_correction_bias",
    "k_b_proj_trans",
    "k_b_proj_trans_scale",
    "k_b_proj_trans_dequant",
    "v_b_proj",
    "v_b_proj_scale",
    "v_b_proj_dequant",
    "gate_up_proj_weight",
    "gate_up_proj_weight_scale",
    "gate_up_proj_input_scale",
    "down_proj_weight",
    "down_proj_weight_scale",
    "down_proj_input_scale",
    "w3_w1_weight",
    "w3_w1_weight_scale",
    "w3_w1_weight_scaling_factor",
    "w2_weight",
    "w2_weight_scale",
    "w2_weight_scaling_factor",
    "fc31_input_scale",
    "fc31_scale_c",
    "fc2_input_scale",
    "fc31_alpha",
    "fc2_alpha",
    "rotary_cos_sin",
)


def _layer_offsets(
    layers: list[DeepSeekResidentLayerAssets],
    num_tensors: int,
) -> tuple[int, ...]:
    if not layers:
        return (num_tensors, )
    return tuple(layer.start for layer in layers) + (layers[-1].stop, )


def _layer_kind_id(layer_kind: str) -> int:
    if layer_kind == "dense":
        return 0
    if layer_kind == "moe":
        return 1
    return -1


def _layer_tensor_sites(
    *,
    layer_idx: int,
    layer_prefix: str,
    tensor_specs: list[DeepSeekResidentTensorAsset],
    start: int,
    stop: int,
) -> tuple[DeepSeekResidentLayerTensorSite, ...]:
    sites: list[DeepSeekResidentLayerTensorSite] = []
    prefix = f"{layer_prefix}."
    for tensor_idx in range(start, stop):
        tensor_name = tensor_specs[tensor_idx].name
        if not tensor_name.startswith(prefix):
            continue
        local_name = tensor_name[len(prefix):]
        site_id = _LAYER_TENSOR_SITE_IDS.get(local_name)
        if site_id is None:
            continue
        sites.append(
            DeepSeekResidentLayerTensorSite(
                layer_idx=layer_idx,
                site_id=site_id,
                tensor_idx=tensor_idx,
                tensor_name=tensor_name,
            ))
    return _dedupe_layer_tensor_sites(sites)


def _dedupe_layer_tensor_sites(
    sites: list[DeepSeekResidentLayerTensorSite],
) -> tuple[DeepSeekResidentLayerTensorSite, ...]:
    selected: dict[int, tuple[int, DeepSeekResidentLayerTensorSite]] = {}
    for site in sites:
        rank = _layer_tensor_site_preference_rank(site.site_id,
                                                 site.tensor_name)
        current = selected.get(site.site_id)
        if current is None or rank < current[0]:
            selected[site.site_id] = (rank, site)
    return tuple(site for _, site in sorted(
        selected.values(), key=lambda item: item[1].tensor_idx))


def _layer_tensor_site_preference_rank(site_id: int, tensor_name: str) -> int:
    preferred_suffixes = _LAYER_TENSOR_SITE_PREFERRED_SUFFIXES.get(site_id)
    if preferred_suffixes is None:
        return 0
    for rank, suffix in enumerate(preferred_suffixes):
        if tensor_name.endswith(suffix):
            return rank
    return len(preferred_suffixes)


def _layer_site_offsets(
    layer_site_starts: list[int],
    num_sites: int,
) -> tuple[int, ...]:
    if not layer_site_starts:
        return (num_sites, )
    return tuple(layer_site_starts) + (num_sites, )


_LAYER_TENSOR_SITE_IDS = {
    "input_layernorm.weight": 0,
    "post_attention_layernorm.weight": 1,
    "next_layer_layernorm.weight": 2,
    "input_gated_norm_down.weight": 3,
    "input_gated_norm_up.weight": 4,
    "post_attention_gated_norm_down.weight": 5,
    "post_attention_gated_norm_up.weight": 6,
    "self_attn.kv_a_proj_with_mqa.weight": 7,
    "self_attn.kv_a_proj_with_mqa.weight_scale": 8,
    "self_attn.kv_a_proj_with_mqa.weight_scale_2": 9,
    "self_attn.kv_a_proj_with_mqa.input_scale": 10,
    "self_attn.kv_a_proj_with_mqa.inv_input_scale": 11,
    "self_attn.kv_b_proj.weight": 12,
    "self_attn.kv_b_proj.weight_scale": 13,
    "self_attn.kv_b_proj.weight_scale_2": 14,
    "self_attn.o_proj.weight": 15,
    "self_attn.o_proj.weight_scale": 16,
    "self_attn.o_proj.weight_scale_2": 17,
    "self_attn.o_proj.input_scale": 18,
    "self_attn.gate_proj.weight": 19,
    "self_attn.gate_proj.weight_scale": 20,
    "self_attn.gate_proj.weight_scale_2": 21,
    "self_attn.gate_proj.input_scale": 22,
    "self_attn.k_b_proj_trans": _SITE_ATTN_K_B_PROJ_TRANS,
    "self_attn.k_b_proj_trans_scale": _SITE_ATTN_K_B_PROJ_TRANS_SCALE,
    "self_attn.k_b_proj_trans_dequant": _SITE_ATTN_K_B_PROJ_TRANS_DEQUANT,
    "self_attn.v_b_proj": _SITE_ATTN_V_B_PROJ,
    "self_attn.v_b_proj_scale": _SITE_ATTN_V_B_PROJ_SCALE,
    "self_attn.v_b_proj_dequant": _SITE_ATTN_V_B_PROJ_DEQUANT,
    "self_attn.kv_a_proj_with_mqa.alpha": _SITE_ATTN_KV_A_PROJ_ALPHA,
    "self_attn.kv_b_proj.alpha": _SITE_ATTN_KV_B_PROJ_ALPHA,
    "self_attn.o_proj.alpha": _SITE_ATTN_O_PROJ_ALPHA,
    "self_attn.gate_proj.alpha": _SITE_ATTN_GATE_PROJ_ALPHA,
    "self_attn.q_a_layernorm.weight": _SITE_ATTN_Q_A_LAYERNORM_WEIGHT,
    "self_attn.kv_a_layernorm.weight": _SITE_ATTN_KV_A_LAYERNORM_WEIGHT,
    "self_attn.q_b_proj.weight": _SITE_ATTN_Q_B_PROJ_WEIGHT,
    "self_attn.q_b_proj.weight_scale": _SITE_ATTN_Q_B_PROJ_WEIGHT_SCALE,
    "self_attn.q_b_proj.weight_scale_2": _SITE_ATTN_Q_B_PROJ_WEIGHT_SCALE_2,
    "self_attn.q_b_proj.input_scale": _SITE_ATTN_Q_B_PROJ_INPUT_SCALE,
    "self_attn.q_b_proj.alpha": _SITE_ATTN_Q_B_PROJ_ALPHA,
    "self_attn.mqa.indexer.wq_b.weight": _SITE_ATTN_INDEXER_WQ_B_WEIGHT,
    "self_attn.mqa.indexer.wq_b.weight_scale":
    _SITE_ATTN_INDEXER_WQ_B_WEIGHT_SCALE,
    "self_attn.mqa.indexer.wq_b.weight_scale_2":
    _SITE_ATTN_INDEXER_WQ_B_WEIGHT_SCALE_2,
    "self_attn.mqa.indexer.wq_b.input_scale":
    _SITE_ATTN_INDEXER_WQ_B_INPUT_SCALE,
    "self_attn.mqa.indexer.wq_b.alpha": _SITE_ATTN_INDEXER_WQ_B_ALPHA,
    "self_attn.mqa.indexer.wk.weight": _SITE_ATTN_INDEXER_WK_WEIGHT,
    "self_attn.mqa.indexer.wk.weight_scale":
    _SITE_ATTN_INDEXER_WK_WEIGHT_SCALE,
    "self_attn.mqa.indexer.wk.weight_scale_2":
    _SITE_ATTN_INDEXER_WK_WEIGHT_SCALE_2,
    "self_attn.mqa.indexer.wk.input_scale":
    _SITE_ATTN_INDEXER_WK_INPUT_SCALE,
    "self_attn.mqa.indexer.wk.alpha": _SITE_ATTN_INDEXER_WK_ALPHA,
    "self_attn.mqa.indexer.weights_proj.weight":
    _SITE_ATTN_INDEXER_WEIGHTS_PROJ_WEIGHT,
    "self_attn.mqa.indexer.weights_proj.weight_scale":
    _SITE_ATTN_INDEXER_WEIGHTS_PROJ_WEIGHT_SCALE,
    "self_attn.mqa.indexer.weights_proj.weight_scale_2":
    _SITE_ATTN_INDEXER_WEIGHTS_PROJ_WEIGHT_SCALE_2,
    "self_attn.mqa.indexer.weights_proj.input_scale":
    _SITE_ATTN_INDEXER_WEIGHTS_PROJ_INPUT_SCALE,
    "self_attn.mqa.indexer.weights_proj.alpha":
    _SITE_ATTN_INDEXER_WEIGHTS_PROJ_ALPHA,
    "self_attn.mqa.indexer.k_norm.weight":
    _SITE_ATTN_INDEXER_K_NORM_WEIGHT,
    "self_attn.mqa.indexer.k_norm.bias": _SITE_ATTN_INDEXER_K_NORM_BIAS,
    "self_attn.mqa.indexer.rotary_emb.rotary_cos_sin":
    _SITE_ATTN_INDEXER_ROTARY_COS_SIN,
    "mlp.gate.weight": 30,
    "mlp.gate.e_score_correction_bias": 31,
    "mlp.gate_up_proj.weight": 40,
    "mlp.gate_up_proj.weight_scale": 41,
    "mlp.gate_up_proj.weight_scale_2": 42,
    "mlp.gate_up_proj.input_scale": 43,
    "mlp.down_proj.weight": 44,
    "mlp.down_proj.weight_scale": 45,
    "mlp.down_proj.weight_scale_2": 46,
    "mlp.down_proj.input_scale": 47,
    "mlp.gate_up_proj.alpha": _SITE_DENSE_MLP_GATE_UP_ALPHA,
    "mlp.down_proj.alpha": _SITE_DENSE_MLP_DOWN_ALPHA,
    "mlp.shared_experts.gate_up_proj.weight":
    _SITE_SHARED_EXPERT_GATE_UP_WEIGHT,
    "mlp.shared_experts.gate_up_proj.weight_scale":
    _SITE_SHARED_EXPERT_GATE_UP_WEIGHT_SCALE,
    "mlp.shared_experts.gate_up_proj.weight_scale_2":
    _SITE_SHARED_EXPERT_GATE_UP_WEIGHT_SCALE_2,
    "mlp.shared_experts.gate_up_proj.input_scale":
    _SITE_SHARED_EXPERT_GATE_UP_INPUT_SCALE,
    "mlp.shared_experts.down_proj.weight": _SITE_SHARED_EXPERT_DOWN_WEIGHT,
    "mlp.shared_experts.down_proj.weight_scale":
    _SITE_SHARED_EXPERT_DOWN_WEIGHT_SCALE,
    "mlp.shared_experts.down_proj.weight_scale_2":
    _SITE_SHARED_EXPERT_DOWN_WEIGHT_SCALE_2,
    "mlp.shared_experts.down_proj.input_scale":
    _SITE_SHARED_EXPERT_DOWN_INPUT_SCALE,
    "mlp.shared_experts.gate_up_proj.alpha": _SITE_SHARED_EXPERT_GATE_UP_ALPHA,
    "mlp.shared_experts.down_proj.alpha": _SITE_SHARED_EXPERT_DOWN_ALPHA,
    "mlp.experts.gate_up_proj_weight": _SITE_EXPERT_GATE_UP_WEIGHT,
    "mlp.experts.gate_up_proj_weight_scale":
    _SITE_EXPERT_GATE_UP_WEIGHT_SCALE,
    "mlp.experts.gate_up_proj_input_scale":
    _SITE_EXPERT_GATE_UP_INPUT_SCALE,
    "mlp.experts.down_proj_weight": _SITE_EXPERT_DOWN_WEIGHT,
    "mlp.experts.down_proj_weight_scale": _SITE_EXPERT_DOWN_WEIGHT_SCALE,
    "mlp.experts.down_proj_input_scale": _SITE_EXPERT_DOWN_INPUT_SCALE,
    "mlp.experts.gate_up_proj_alpha": _SITE_EXPERT_GATE_UP_ALPHA,
    "mlp.experts.down_proj_alpha": _SITE_EXPERT_DOWN_ALPHA,
    "mlp.experts.w3_w1_weight": _SITE_EXPERT_GATE_UP_WEIGHT,
    "mlp.experts.w3_w1_weight_scale": _SITE_EXPERT_GATE_UP_WEIGHT_SCALE,
    "mlp.experts.w3_w1_weight_scaling_factor":
    _SITE_EXPERT_GATE_UP_WEIGHT_SCALE,
    "mlp.experts.fc31_input_scale": _SITE_EXPERT_GATE_UP_INPUT_SCALE,
    "mlp.experts.fc31_scale_c": _SITE_EXPERT_GATE_UP_OUTPUT_SCALE,
    "mlp.experts.fc31_alpha": _SITE_EXPERT_GATE_UP_ALPHA,
    "mlp.experts.w2_weight": _SITE_EXPERT_DOWN_WEIGHT,
    "mlp.experts.w2_weight_scale": _SITE_EXPERT_DOWN_WEIGHT_SCALE,
    "mlp.experts.w2_weight_scaling_factor": _SITE_EXPERT_DOWN_WEIGHT_SCALE,
    "mlp.experts.fc2_input_scale": _SITE_EXPERT_DOWN_INPUT_SCALE,
    "mlp.experts.fc2_alpha": _SITE_EXPERT_DOWN_ALPHA,
    "mlp.experts.backend.w3_w1_weight": _SITE_EXPERT_GATE_UP_WEIGHT,
    "mlp.experts.backend.w3_w1_weight_scale":
    _SITE_EXPERT_GATE_UP_WEIGHT_SCALE,
    "mlp.experts.backend.w3_w1_weight_scaling_factor":
    _SITE_EXPERT_GATE_UP_WEIGHT_SCALE,
    "mlp.experts.backend.fc31_input_scale":
    _SITE_EXPERT_GATE_UP_INPUT_SCALE,
    "mlp.experts.backend.fc31_scale_c":
    _SITE_EXPERT_GATE_UP_OUTPUT_SCALE,
    "mlp.experts.backend.fc31_alpha": _SITE_EXPERT_GATE_UP_ALPHA,
    "mlp.experts.backend.w2_weight": _SITE_EXPERT_DOWN_WEIGHT,
    "mlp.experts.backend.w2_weight_scale": _SITE_EXPERT_DOWN_WEIGHT_SCALE,
    "mlp.experts.backend.w2_weight_scaling_factor":
    _SITE_EXPERT_DOWN_WEIGHT_SCALE,
    "mlp.experts.backend.fc2_input_scale": _SITE_EXPERT_DOWN_INPUT_SCALE,
    "mlp.experts.backend.fc2_alpha": _SITE_EXPERT_DOWN_ALPHA,
}

_LAYER_TENSOR_SITE_PREFERRED_SUFFIXES = {
    _SITE_EXPERT_GATE_UP_WEIGHT: (
        "mlp.experts.backend.w3_w1_weight",
        "mlp.experts.w3_w1_weight",
        "mlp.experts.gate_up_proj_weight",
    ),
    _SITE_EXPERT_GATE_UP_WEIGHT_SCALE: (
        "mlp.experts.backend.w3_w1_weight_scale",
        "mlp.experts.backend.w3_w1_weight_scaling_factor",
        "mlp.experts.w3_w1_weight_scale",
        "mlp.experts.w3_w1_weight_scaling_factor",
        "mlp.experts.gate_up_proj_weight_scale",
    ),
    _SITE_EXPERT_GATE_UP_INPUT_SCALE: (
        "mlp.experts.backend.fc31_input_scale",
        "mlp.experts.fc31_input_scale",
        "mlp.experts.gate_up_proj_input_scale",
    ),
    _SITE_EXPERT_GATE_UP_OUTPUT_SCALE: (
        "mlp.experts.backend.fc31_scale_c",
        "mlp.experts.fc31_scale_c",
    ),
    _SITE_EXPERT_GATE_UP_ALPHA: (
        "mlp.experts.backend.fc31_alpha",
        "mlp.experts.fc31_alpha",
        "mlp.experts.gate_up_proj_alpha",
    ),
    _SITE_EXPERT_DOWN_WEIGHT: (
        "mlp.experts.backend.w2_weight",
        "mlp.experts.w2_weight",
        "mlp.experts.down_proj_weight",
    ),
    _SITE_EXPERT_DOWN_WEIGHT_SCALE: (
        "mlp.experts.backend.w2_weight_scale",
        "mlp.experts.backend.w2_weight_scaling_factor",
        "mlp.experts.w2_weight_scale",
        "mlp.experts.w2_weight_scaling_factor",
        "mlp.experts.down_proj_weight_scale",
    ),
    _SITE_EXPERT_DOWN_INPUT_SCALE: (
        "mlp.experts.backend.fc2_input_scale",
        "mlp.experts.fc2_input_scale",
        "mlp.experts.down_proj_input_scale",
    ),
    _SITE_EXPERT_DOWN_ALPHA: (
        "mlp.experts.backend.fc2_alpha",
        "mlp.experts.fc2_alpha",
        "mlp.experts.down_proj_alpha",
    ),
}


def _is_tensor_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _iter_fallback_child_modules(module: Any) -> tuple[tuple[str, Any], ...]:
    try:
        items = vars(module).items()
    except TypeError:
        return ()

    children: list[tuple[str, Any]] = []
    for name, value in items:
        if name.startswith("_") or value is None or _is_tensor_like(value):
            continue
        if isinstance(value, (bool, int, float, str, bytes)):
            continue
        children.append((name, value))
    return tuple(children)


def _tensor_shape(tensor: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in getattr(tensor, "shape", ()))
    except (TypeError, ValueError):
        return ()


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    try:
        return tuple(value)
    except TypeError:
        return ()


def _resolve_native_op(native_op_name: str) -> Any | None:
    torch = _import_torch()
    if torch is None:
        return None
    namespace, op_name = _split_native_op_name(native_op_name)
    if namespace is None:
        return None
    try:
        op_namespace = getattr(torch.ops, namespace)
        op = getattr(op_namespace, op_name)
    except (AttributeError, RuntimeError):
        return None
    return op if callable(op) else None


def _resolve_native_class(native_class_name: str) -> Any | None:
    torch = _import_torch()
    if torch is None:
        return None
    namespace, class_name = _split_native_op_name(native_class_name)
    if namespace is None:
        return None
    try:
        class_namespace = getattr(torch.classes, namespace)
        class_obj = getattr(class_namespace, class_name)
    except (AttributeError, RuntimeError):
        return None
    return class_obj if callable(class_obj) else None


def _resolve_all_reduce_params_class() -> Any | None:
    try:
        from tensorrt_llm._torch.distributed import AllReduceParams
    except ImportError:
        return None
    return AllReduceParams


def _resolve_moe_all_reduce_params_class() -> Any | None:
    try:
        from tensorrt_llm._torch.distributed import MoEAllReduceParams
    except ImportError:
        return None
    return MoEAllReduceParams


def _wide_ep_moe_kwargs(experts: Any) -> dict[str, Any]:
    try:
        from tensorrt_llm._torch.modules.fused_moe.fused_moe_wide_ep import (
            WideEPMoE)
    except (AttributeError, ImportError, RuntimeError):
        WideEPMoE = None
    if WideEPMoE is not None and isinstance(experts, WideEPMoE):
        return {"alltoall_result_do_sum": False}
    if experts.__class__.__name__ == "WideEPMoE":
        return {"alltoall_result_do_sum": False}
    return {}


def _native_op_ready(native_op_name: str) -> bool:
    namespace, op_name = _split_native_op_name(native_op_name)
    if namespace is None:
        return False
    ready_op = _resolve_native_op(f"{namespace}.{op_name}_ready")
    if ready_op is None:
        return True
    try:
        return bool(ready_op())
    except RuntimeError:
        return False


def _split_native_op_name(native_op_name: str) -> tuple[str | None, str]:
    parts = native_op_name.split(".", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None, native_op_name
    return parts[0], parts[1]


def _import_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _tensor_stride(tensor: Any) -> tuple[int, ...]:
    stride = getattr(tensor, "stride", None)
    if not callable(stride):
        return ()
    try:
        return tuple(int(dim) for dim in stride())
    except (RuntimeError, TypeError, ValueError):
        return ()


def _tensor_data_ptr(tensor: Any) -> int | None:
    data_ptr = getattr(tensor, "data_ptr", None)
    if not callable(data_ptr):
        return None
    try:
        return int(data_ptr())
    except (RuntimeError, TypeError, ValueError):
        return None


def _tensor_layout_key(tensor: Any) -> tuple[Any, ...]:
    return (
        _tensor_shape(tensor),
        _tensor_stride(tensor),
        str(getattr(tensor, "dtype", None)),
        str(getattr(tensor, "device", None)),
    )


def _tensor_numel(tensor: Any) -> int:
    shape = _tensor_shape(tensor)
    if not shape:
        return 1
    numel = 1
    for dim in shape:
        numel *= max(int(dim), 0)
    return numel


def _tensor_pointer_key(tensor: Any) -> tuple[Any, ...]:
    return (*_tensor_layout_key(tensor), _tensor_data_ptr(tensor))


def _dict_value_key(values: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted((str(key), value) for key, value in values.items()))


def _runtime_tensor_dict_key(
    runtime_tensors: dict[str, Any],
) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    return tuple(
        sorted((str(key), _tensor_pointer_key(tensor))
               for key, tensor in runtime_tensors.items()))


def _runtime_tensor_dict_graph_key(
    runtime_tensors: dict[str, Any],
) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    return tuple(
        sorted((str(key),
                _tensor_layout_key(tensor) if _tensor_can_new_empty(tensor) else
                _tensor_pointer_key(tensor))
               for key, tensor in runtime_tensors.items()))


def _payload_tensor_pointer_key(payload: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(_tensor_pointer_key(tensor)
              for tensor in payload["metadata_tensors"]),
        tuple(
            _runtime_tensor_dict_key(runtime_tensors)
            for runtime_tensors in payload["runtime_tensors"]),
        tuple(_dict_value_key(config)
              for config in payload["runtime_config"]),
        tuple(_dict_value_key(scalars)
              for scalars in payload["runtime_scalars"]),
        tuple(_tensor_pointer_key(tensor)
              for tensor in payload["scratch_tensors"]),
    )


def _payload_tensor_graph_key(payload: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(_tensor_pointer_key(tensor)
              for tensor in payload["metadata_tensors"]),
        tuple(
            _runtime_tensor_dict_graph_key(runtime_tensors)
            for runtime_tensors in payload["runtime_tensors"]),
        tuple(_dict_value_key(config)
              for config in payload["runtime_config"]),
        tuple(_dict_value_key(scalars)
              for scalars in payload["runtime_scalars"]),
        tuple(_tensor_pointer_key(tensor)
              for tensor in payload["scratch_tensors"]),
    )


def _copy_tensor_value(dst: Any, src: Any) -> bool:
    copy_ = getattr(dst, "copy_", None)
    if not callable(copy_):
        return False
    try:
        copy_(src)
    except (RuntimeError, TypeError, ValueError):
        return False
    return True


def _tensor_can_new_empty(tensor: Any) -> bool:
    return (callable(getattr(tensor, "new_empty", None))
            and getattr(tensor, "shape", None) is not None
            and _tensor_numel(tensor)
            <= _WINDOW_CUDA_GRAPH_RUNTIME_STAGE_MAX_NUMEL)


def _fingerprint_value(value: Any) -> str:
    text = repr(value)
    if len(text) > 4096:
        text = text[:4096]
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _next_power_of_two_clamped(value: int, hard_cap: int) -> int:
    if value <= 0 or hard_cap <= 0:
        return hard_cap
    if value > hard_cap // 2:
        return hard_cap
    bucket = 1
    while bucket < value and bucket < hard_cap:
        bucket <<= 1
    return min(bucket, hard_cap)


def _max_live_window_kv_len(
    cached_tokens: tuple[int, ...],
    input_tokens: int,
    owned_steps: int,
) -> int:
    active_tokens = min(len(cached_tokens), max(int(input_tokens), 0))
    max_live_tokens = 0
    for idx in range(active_tokens):
        max_live_tokens = max(max_live_tokens, int(cached_tokens[idx]))
    return max_live_tokens + max(int(owned_steps), 1)


def _dsa_window_indexer_widths(
    *,
    cached_tokens: tuple[int, ...],
    input_tokens: int,
    owned_steps: int,
    dsa_window_plan: tuple[DeepSeekResidentDsaWindowLayerPlan, ...],
) -> tuple[tuple[int, int], ...] | None:
    max_live_kv_len = _max_live_window_kv_len(
        cached_tokens=tuple(int(token) for token in cached_tokens),
        input_tokens=int(input_tokens),
        owned_steps=int(owned_steps),
    )
    widths: list[tuple[int, int]] = []
    for layer_plan in dsa_window_plan:
        try:
            hard_cap = int(layer_plan.runtime_config.get("max_seq_len", 0))
            layer_idx = int(layer_plan.layer_idx)
        except (TypeError, ValueError):
            return None
        widths.append(
            (layer_idx, _next_power_of_two_clamped(max_live_kv_len, hard_cap)))
    return tuple(widths)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes",
                                                         "on")


def _env_flag_default(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _window_stage_body_enabled() -> bool:
    return _env_flag_default(_WINDOW_STAGE_BODY_ENV_NAME, True)


def _torch_float32_dtype(torch: Any) -> Any:
    return getattr(torch, "float32", "torch.float32")


def _torch_int32_dtype(torch: Any) -> Any:
    return getattr(torch, "int32", "torch.int32")


def _torch_uint32_dtype(torch: Any) -> Any:
    return getattr(torch, "uint32", _torch_int32_dtype(torch))


def _torch_uint8_dtype(torch: Any) -> Any:
    return getattr(torch, "uint8", "torch.uint8")


def create_engine(
    *,
    model: Any,
    contract: Any,
    dist: Any,
) -> DeepSeekResidentNativeEngine:
    """Return the resident native engine wrapper."""

    return DeepSeekResidentNativeEngine(
        model=model,
        contract=contract,
        dist=dist,
        native_op_name=os.environ.get(_NATIVE_OP_ENV_NAME,
                                      _DEFAULT_NATIVE_OP).strip(),
    )

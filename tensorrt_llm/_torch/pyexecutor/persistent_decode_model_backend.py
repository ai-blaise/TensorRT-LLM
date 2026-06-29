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

"""Model-engine handoff for a TileRT-style resident decode body.

The executor-level persistent decode loop can own multiple decode iterations,
but the current bridge still calls back into ``ModelEngine.forward`` for every
token. A real TileRT-style path needs a lower boundary: after TRT-LLM has built
the production scheduled-request, KV, and attention metadata contracts, but
before the DeepSeek layer loop is replayed as the normal per-token graph.

This module defines that lower boundary. The only backend registered today is a
default-off DeepSeek resident stub that records why it declined execution. It is
intentionally not a performance feature; it is the insertion point for the
model-specific resident body.
"""

from __future__ import annotations

import importlib
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch

from tensorrt_llm.logger import logger

_BACKEND_ENV_NAME = "TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND"
_REPORT_EVERY_ENV_NAME = "TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND_EVERY"
_RANKS_ENV_NAME = "TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND_RANKS"

_DISABLED_BACKEND_NAMES = ("", "0", "disabled", "none")
_DEEPSEEK_RESIDENT_V0_BACKEND_NAME = "deepseek_resident_v0"
_DEEPSEEK_RESIDENT_PYTHON_V1_BACKEND_NAME = "deepseek_resident_python_v1"
_DEEPSEEK_RESIDENT_NATIVE_V1_BACKEND_NAME = "deepseek_resident_native_v1"
_NATIVE_MODULE_ENV_NAME = "TRTLLM_OPTRT_PERSISTENT_DECODE_NATIVE_MODULE"
_DEFAULT_NATIVE_MODULE = (
    "tensorrt_llm._torch.pyexecutor.deepseek_resident_native")
_NATIVE_MIN_LOCAL_BATCH_ENV_NAME = (
    "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MIN_LOCAL_BATCH")
_DEFAULT_REPORT_EVERY = 128
_RESIDENT_ALLOW_STREAMING_ENV_NAME = (
    "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_ALLOW_STREAMING")
_RESIDENT_TERMINAL_WINDOW_ENV_NAME = (
    "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_TERMINAL_WINDOW")
_REQUEST_ID_PADDING_SENTINEL_UINT64 = (1 << 64) - 1
_REQUEST_ID_PADDING_SENTINELS = frozenset(
    (0, -1, _REQUEST_ID_PADDING_SENTINEL_UINT64))
_REQUEST_ID_MISMATCH_PREVIEW_LIMIT = 8


@dataclass(frozen=True)
class PersistentDecodeModelForwardRequest:
    """Inputs available at the model-engine resident handoff."""

    plan: Any
    real_requests: Any
    padded_requests: Any
    inputs: dict[str, Any]
    gather_ids: Any
    attn_metadata: Any
    spec_metadata: Any
    kv_cache_manager: Any
    draft_kv_cache_manager: Any
    resource_manager: Any
    model: Any
    cuda_graph_key: Any
    can_run_graph: bool
    gather_context_logits: bool
    preprocess_inputs: Callable[[dict[str, Any]], dict[str, Any]]
    model_forward: Callable[..., Any]
    without_logits: bool


def _request_id_preview(request_ids: tuple[int, ...]) -> str:
    preview = ",".join(str(request_id)
                       for request_id in request_ids[:_REQUEST_ID_MISMATCH_PREVIEW_LIMIT])
    if len(request_ids) > _REQUEST_ID_MISMATCH_PREVIEW_LIMIT:
        preview = f"{preview},..."
    return f"[{preview};n={len(request_ids)}]"


def _is_padding_request_id(request_id: int) -> bool:
    return request_id in _REQUEST_ID_PADDING_SENTINELS


def _request_ids_mismatch_reason(
    invocation_request_ids: tuple[int, ...],
    invocation_real_request_ids: tuple[int, ...],
    request_ids: tuple[int, ...],
    all_request_ids: tuple[int, ...],
) -> str:
    return (
        "request_ids_mismatch:"
        f"invocation={_request_id_preview(invocation_request_ids)},"
        f"invocation_real={_request_id_preview(invocation_real_request_ids)},"
        f"request={_request_id_preview(request_ids)},"
        f"all={_request_id_preview(all_request_ids)}")


def _window_metadata_lane_indices(
    invocation_request_ids: tuple[int, ...],
    has_real_requests: bool,
) -> tuple[int, ...]:
    if not has_real_requests:
        return tuple(range(len(invocation_request_ids)))
    return tuple(
        idx for idx, request_id in enumerate(invocation_request_ids)
        if not _is_padding_request_id(request_id))


def _project_window_seq_lens(
    invocation_seq_lens: tuple[int, ...],
    lane_indices: tuple[int, ...],
    request_count: int,
) -> tuple[int, ...]:
    seq_lens = tuple(
        invocation_seq_lens[idx] if idx < len(invocation_seq_lens) else 1
        for idx in lane_indices[:request_count])
    if len(seq_lens) < request_count:
        seq_lens = seq_lens + (1, ) * (request_count - len(seq_lens))
    return seq_lens


def _project_window_cached_tokens(
    invocation_cached_tokens: tuple[int, ...],
    lane_indices: tuple[int, ...],
    request_count: int,
) -> tuple[int, ...] | None:
    cached_tokens = tuple(
        invocation_cached_tokens[idx]
        for idx in lane_indices[:request_count]
        if idx < len(invocation_cached_tokens))
    if len(cached_tokens) == request_count:
        return cached_tokens
    if len(invocation_cached_tokens) >= request_count:
        return invocation_cached_tokens[:request_count]
    return None


@dataclass(frozen=True)
class PersistentDecodeModelForwardResult:
    """Outputs from a backend-owned model forward."""

    outputs: Any
    backend: str
    reason: str
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class PersistentDecodeModelWindowRequest:
    """Inputs for a model-backend-owned resident decode window."""

    forward_request: PersistentDecodeModelForwardRequest
    sample_state: Any
    requested_window_steps: int
    make_sample_state: Callable[..., Any] | None = None


@dataclass(frozen=True)
class PersistentDecodeModelWindowResult:
    """Result from a model-backend-owned resident decode window."""

    sample_state: Any
    owned_steps: int
    requested_window_steps: int
    break_reason: str
    executed: bool = True


@dataclass(frozen=True)
class PersistentDecodeModelWindowSequenceContract:
    """Per-request state that must remain stable inside a resident window."""

    request_id: int
    seq_slot: int
    max_new_tokens: int
    decoding_iter: int
    remaining_decode_steps: int
    beam_width: int
    streaming: bool
    return_log_probs: bool
    return_generation_logits: bool
    stop_words_present: bool
    end_id: int | None
    draft_token_count: int
    state: str


@dataclass(frozen=True)
class PersistentDecodeModelWindowContract:
    """Stable-batch contract consumed by a native multi-step window."""

    requested_window_steps: int
    already_sampled_steps: int
    requested_additional_steps: int
    max_safe_owned_steps: int
    request_ids: tuple[int, ...]
    seq_slots: tuple[int, ...]
    sequences: tuple[PersistentDecodeModelWindowSequenceContract, ...]
    input_ids_spec: "PersistentDecodeTensorSpec"
    position_ids_spec: "PersistentDecodeTensorSpec | None"
    attention_metadata_type: str
    kv_cache_manager_type: str
    cached_tokens: tuple[int, ...]
    seq_lens: tuple[int, ...]
    stable_shape_key: tuple[Any, ...]

    @property
    def ready(self) -> bool:
        return self.max_safe_owned_steps > 1

    def summary(self) -> dict[str, Any]:
        return {
            "requested_window_steps": self.requested_window_steps,
            "already_sampled_steps": self.already_sampled_steps,
            "requested_additional_steps": self.requested_additional_steps,
            "max_safe_owned_steps": self.max_safe_owned_steps,
            "request_ids": self.request_ids,
            "seq_slots": self.seq_slots,
            "cached_tokens": self.cached_tokens,
            "seq_lens": self.seq_lens,
            "attention_metadata_type": self.attention_metadata_type,
            "kv_cache_manager_type": self.kv_cache_manager_type,
            "input_ids_spec": self.input_ids_spec.__dict__,
            "position_ids_spec": None if self.position_ids_spec is None else
            self.position_ids_spec.__dict__,
            "stable_shape_key": self.stable_shape_key,
        }


@dataclass(frozen=True)
class PersistentDecodeTensorSpec:
    """Shape contract for a tensor crossing the resident-body boundary."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    device: str


@dataclass(frozen=True)
class DeepSeekResidentInvocationContract:
    """Per-forward invocation contract for a native resident DeepSeek body."""

    real_batch_size: int
    padded_batch_size: int
    request_ids: tuple[int, ...]
    seq_lens: tuple[int, ...]
    cached_tokens: tuple[int, ...]
    input_tokens: int
    cuda_graph_replay: bool
    cuda_graph_padding: bool
    attention_metadata_type: str
    kv_cache_manager_type: str
    input_tensor_specs: tuple[PersistentDecodeTensorSpec, ...]
    gather_ids_spec: PersistentDecodeTensorSpec | None

    @property
    def stable_shape_key(self) -> tuple[Any, ...]:
        return (
            self.padded_batch_size,
            self.input_tokens,
            self.cuda_graph_padding,
            self.attention_metadata_type,
            self.kv_cache_manager_type,
            tuple((spec.name, spec.shape, spec.dtype)
                  for spec in self.input_tensor_specs),
            None if self.gather_ids_spec is None else (
                self.gather_ids_spec.shape,
                self.gather_ids_spec.dtype,
            ),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "real_batch_size": self.real_batch_size,
            "padded_batch_size": self.padded_batch_size,
            "request_ids": self.request_ids,
            "seq_lens": self.seq_lens,
            "cached_tokens": self.cached_tokens,
            "input_tokens": self.input_tokens,
            "cuda_graph_replay": self.cuda_graph_replay,
            "cuda_graph_padding": self.cuda_graph_padding,
            "attention_metadata_type": self.attention_metadata_type,
            "kv_cache_manager_type": self.kv_cache_manager_type,
            "input_tensor_specs": [
                spec.__dict__ for spec in self.input_tensor_specs
            ],
            "gather_ids_spec": None if self.gather_ids_spec is None else
            self.gather_ids_spec.__dict__,
            "stable_shape_key": self.stable_shape_key,
        }


@dataclass(frozen=True)
class DeepSeekResidentLayerContract:
    """Layer-level work that a resident DeepSeek body must own."""

    layer_idx: int
    layer_kind: str
    attention_type: str
    mlp_type: str
    has_input_gated_norm: bool
    has_post_attention_gated_norm: bool
    input_gate_rank: int | None
    post_attention_gate_rank: int | None
    has_attention_output_gate: bool
    has_kv_a_projection: bool
    has_next_layer_layernorm: bool
    disable_attn_allreduce: bool
    pre_mlp_or_moe_fusion: bool
    post_mlp_or_moe_fusion: bool
    num_experts: int | None
    top_k: int | None


@dataclass(frozen=True)
class DeepSeekResidentBodyContract:
    """Model-level contract for the TileRT-style resident body."""

    model_type: str
    architectures: tuple[str, ...]
    num_layers: int
    hidden_size: int | None
    vocab_size: int | None
    rms_norm_eps: float | None
    gated_norm: bool
    attention_output_gate: bool
    n_routed_experts: int | None
    num_experts_per_tok: int | None
    n_group: int | None
    topk_group: int | None
    routed_scaling_factor: float | None
    first_k_dense_replace: int | None
    moe_layer_freq: int | None
    layer_kind_hist: dict[str, int]
    layers: tuple[DeepSeekResidentLayerContract, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "architectures": self.architectures,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "rms_norm_eps": self.rms_norm_eps,
            "gated_norm": self.gated_norm,
            "attention_output_gate": self.attention_output_gate,
            "n_routed_experts": self.n_routed_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "n_group": self.n_group,
            "topk_group": self.topk_group,
            "routed_scaling_factor": self.routed_scaling_factor,
            "first_k_dense_replace": self.first_k_dense_replace,
            "moe_layer_freq": self.moe_layer_freq,
            "layer_kind_hist": dict(sorted(self.layer_kind_hist.items())),
            "attention_types": sorted(
                {layer.attention_type for layer in self.layers}),
            "mlp_types": sorted({layer.mlp_type for layer in self.layers}),
            "attention_output_gate_layers": sum(
                layer.has_attention_output_gate for layer in self.layers),
            "input_gated_norm_layers": sum(
                layer.has_input_gated_norm for layer in self.layers),
            "post_attention_gated_norm_layers": sum(
                layer.has_post_attention_gated_norm
                for layer in self.layers),
        }


class PersistentDecodeModelBackend(Protocol):
    """Backend contract for replacing the DeepSeek model forward body."""

    name: str

    def try_execute(
        self,
        request: PersistentDecodeModelForwardRequest,
    ) -> PersistentDecodeModelForwardResult | None:
        """Return a backend-owned result, or ``None`` to fall back."""

    def window_backend_state(self) -> dict[str, Any]:
        """Return resident decode-window readiness."""

    def try_execute_window(
        self,
        request: PersistentDecodeModelWindowRequest,
    ) -> PersistentDecodeModelWindowResult | None:
        """Return a backend-owned decode-window result, or ``None``."""


class DeepSeekResidentNativeEngine(Protocol):
    """Native engine object loaded by ``deepseek_resident_native_v1``."""

    def execute(
        self,
        *,
        request: PersistentDecodeModelForwardRequest,
        contract: DeepSeekResidentBodyContract,
        invocation: DeepSeekResidentInvocationContract,
        inputs: dict[str, Any],
    ) -> Any:
        """Run one resident decode model-body invocation."""

    def window_backend_state(self) -> dict[str, Any]:
        """Return native decode-window readiness."""

    def execute_window(
        self,
        *,
        request: PersistentDecodeModelWindowRequest,
        contract: DeepSeekResidentBodyContract,
        invocation: DeepSeekResidentInvocationContract,
        window_contract: PersistentDecodeModelWindowContract,
        inputs: dict[str, Any],
    ) -> Any:
        """Own a resident decode window."""


class DeepSeekResidentModelBackendV0:
    """Default-off resident model backend placeholder.

    This class deliberately declines execution. Keeping the rejection logic in
    the same object that will later own the native body prevents the next step
    from growing another profiler-only path.
    """

    name = _DEEPSEEK_RESIDENT_V0_BACKEND_NAME

    def __init__(self, dist: Any) -> None:
        self._dist = dist
        self._rank = getattr(dist, "rank", None)
        self._tp_rank = getattr(dist, "tp_rank", None)
        self._report_every = max(
            0, _env_int(_REPORT_EVERY_ENV_NAME, _DEFAULT_REPORT_EVERY))
        self._attempts = 0
        self._reasons: Counter[str] = Counter()
        self._last_reason: str | None = None
        self._batch_hist: Counter[int] = Counter()
        self._graph_hist: Counter[bool] = Counter()
        self._contract_model_id: int | None = None
        self._contract: DeepSeekResidentBodyContract | None = None
        self._contract_rejection: str | None = None

    def try_execute(
        self,
        request: PersistentDecodeModelForwardRequest,
    ) -> PersistentDecodeModelForwardResult | None:
        reason = self._reject_reason(request)
        self._record(request, reason)
        return None

    def window_backend_state(self) -> dict[str, Any]:
        return {
            "backend": None,
            "ready": False,
            "reason": "resident_window_model_backend_not_native",
            "metadata": {
                "model_backend": self.name,
            },
        }

    def try_execute_window(
        self,
        request: PersistentDecodeModelWindowRequest,
    ) -> PersistentDecodeModelWindowResult | None:
        self._record(request.forward_request,
                     "resident_window_model_backend_not_native")
        return None

    @property
    def last_reason(self) -> str | None:
        return self._last_reason

    def _reject_reason(
        self,
        request: PersistentDecodeModelForwardRequest,
    ) -> str:
        plan = request.plan
        if plan is None:
            return "missing_persistent_decode_plan"
        if not bool(getattr(plan, "eligible", False)):
            return str(getattr(plan, "reason", "ineligible_plan"))
        if request.spec_metadata is not None:
            return "spec_metadata_present"
        if request.inputs.get("attn_metadata") is None:
            return "missing_attention_metadata_input"
        if request.inputs.get("input_ids") is None:
            return "missing_input_ids"
        contract, contract_rejection = self._get_contract(request.model)
        if contract is None:
            return f"resident_model_contract_{contract_rejection}"
        return "resident_model_body_contract_ready_not_implemented"

    def _record(
        self,
        request: PersistentDecodeModelForwardRequest,
        reason: str,
    ) -> None:
        self._attempts += 1
        self._last_reason = reason
        self._reasons[reason] += 1
        batch_size = int(getattr(request.plan, "real_generation_requests", 0)
                         or 0)
        self._batch_hist[batch_size] += 1
        self._graph_hist[bool(request.can_run_graph)] += 1

        if self._report_every <= 0 or self._attempts % self._report_every != 0:
            return
        summary = {
            "rank": self._rank,
            "tp_rank": self._tp_rank,
            "backend": self.name,
            "attempts": self._attempts,
            "reasons": dict(self._reasons),
            "batch_hist": dict(sorted(self._batch_hist.items())),
            "cuda_graph_replay": dict(self._graph_hist),
        }
        if self._contract is not None:
            summary["contract"] = self._contract.summary()
        if self._contract_rejection is not None:
            summary["contract_rejection"] = self._contract_rejection
        window_state = self._window_backend_state_summary()
        if window_state is not None:
            summary["window_backend_state"] = window_state
        logger.info(f"OPTRT_PERSISTENT_DECODE_MODEL_BACKEND {summary}")
        self._reset()

    def _window_backend_state_summary(self) -> dict[str, Any] | None:
        state_fn = getattr(self, "window_backend_state", None)
        if not callable(state_fn):
            return None
        try:
            state = state_fn()
        except (RuntimeError, TypeError) as exc:
            return {
                "backend": None,
                "ready": False,
                "reason": f"state_failed:{type(exc).__name__}",
            }
        if not isinstance(state, dict):
            return {
                "backend": None,
                "ready": False,
                "reason": "state_invalid",
            }
        summary: dict[str, Any] = {
            "backend": state.get("backend"),
            "ready": bool(state.get("ready", False)),
            "reason": state.get("reason"),
        }
        metadata = state.get("metadata")
        if not isinstance(metadata, dict):
            return summary
        for key in (
                "dsa_window_plan_builds",
                "dsa_window_plan_layers",
                "last_dsa_window_plan_reason",
        ):
            if key in metadata:
                summary[key] = metadata.get(key)
        native_contract = metadata.get("native_window_contract")
        if isinstance(native_contract, dict):
            summary["native_window_contract"] = {
                "ready":
                bool(native_contract.get("ready", False)),
                "first_missing_component":
                native_contract.get("first_missing_component"),
                "missing_components":
                tuple(native_contract.get("missing_components", ())),
                "layer_counts":
                native_contract.get("layer_counts"),
                "component_ready":
                native_contract.get("component_ready"),
                "component_reasons":
                native_contract.get("component_reasons"),
            }
        return summary

    def _reset(self) -> None:
        self._attempts = 0
        self._reasons.clear()
        self._batch_hist.clear()
        self._graph_hist.clear()

    def _get_contract(
        self,
        model: Any,
    ) -> tuple[DeepSeekResidentBodyContract | None, str | None]:
        model_id = id(model)
        if self._contract_model_id == model_id:
            return self._contract, self._contract_rejection

        self._contract_model_id = model_id
        self._contract, self._contract_rejection = (
            build_deepseek_resident_body_contract(model))
        return self._contract, self._contract_rejection


class DeepSeekResidentModelBackendPythonV1(DeepSeekResidentModelBackendV0):
    """First executable resident model-body boundary.

    This backend intentionally still uses the existing DeepSeek module graph for
    the actual math. The important difference from ``_forward_step`` is the
    ownership boundary: once the production inputs and attention metadata are
    prepared, this backend owns preprocessing, model-body execution, logits
    wrapping, and optional gather. The next implementation can replace
    ``model_forward`` with a native resident DeepSeek layer loop without adding
    another executor-level bridge.
    """

    name = _DEEPSEEK_RESIDENT_PYTHON_V1_BACKEND_NAME
    _CONTRACT_READY_REASON = "resident_model_body_contract_ready_not_implemented"

    def try_execute(
        self,
        request: PersistentDecodeModelForwardRequest,
    ) -> PersistentDecodeModelForwardResult | None:
        reason = self._reject_reason(request)
        if reason != self._CONTRACT_READY_REASON:
            self._record(request, reason)
            return None

        outputs = self._execute_python_body(request)
        self._record(request, "resident_model_body_python_v1_executed")
        return PersistentDecodeModelForwardResult(
            outputs=outputs,
            backend=self.name,
            reason="resident_model_body_python_v1_executed",
        )

    def _execute_python_body(
        self,
        request: PersistentDecodeModelForwardRequest,
    ) -> dict[str, Any] | Any:
        inputs = request.preprocess_inputs(request.inputs)
        gather_ids = request.gather_ids
        spec_metadata = inputs.get("spec_metadata", None)
        if spec_metadata is not None:
            gather_ids = spec_metadata.gather_ids

        outputs = request.model_forward(
            **inputs,
            return_context_logits=(
                gather_ids is not None or request.gather_context_logits),
        )

        if request.without_logits:
            return outputs

        if isinstance(outputs, dict):
            logits = outputs.get("logits", None)
            if logits is None:
                return outputs
        else:
            logits = outputs
            outputs = {"logits": logits}

        if gather_ids is not None:
            outputs["logits"] = logits[gather_ids]
        return outputs


class DeepSeekResidentModelBackendNativeV1(DeepSeekResidentModelBackendV0):
    """Native resident DeepSeek model-body handoff.

    This backend is the first production-shaped ABI for the real TileRT-style
    implementation. It does not use the public TileRT batch-one slot. Instead it
    validates the same TRT-LLM decode-only batches as the Python bridge, builds a
    stable shape/request descriptor, and calls an optional native engine module.
    If the module is absent or declines the shape, execution falls back to the
    existing model path.
    """

    name = _DEEPSEEK_RESIDENT_NATIVE_V1_BACKEND_NAME
    _CONTRACT_READY_REASON = "resident_model_body_contract_ready_not_implemented"

    def __init__(self, dist: Any) -> None:
        super().__init__(dist)
        self._native_module_name = os.environ.get(
            _NATIVE_MODULE_ENV_NAME, _DEFAULT_NATIVE_MODULE).strip()
        self._native_engine_model_id: int | None = None
        self._native_engine: DeepSeekResidentNativeEngine | None = None
        self._native_engine_rejection: str | None = None
        self._native_shape_hist: Counter[tuple[Any, ...]] = Counter()
        self._min_local_batch = max(
            1, _env_int(_NATIVE_MIN_LOCAL_BATCH_ENV_NAME, 1))

    def try_execute(
        self,
        request: PersistentDecodeModelForwardRequest,
    ) -> PersistentDecodeModelForwardResult | None:
        reason = self._reject_reason(request)
        if reason != self._CONTRACT_READY_REASON:
            self._record(request, reason)
            return None

        invocation, invocation_rejection = (
            build_deepseek_resident_invocation_contract(request))
        if invocation is None:
            self._record(
                request,
                f"resident_native_invocation_{invocation_rejection}",
            )
            return None
        if not self._local_batch_admitted(invocation):
            self._record(
                request,
                self._local_batch_decline_reason(invocation),
            )
            return None

        contract, contract_rejection = self._get_contract(request.model)
        if contract is None:
            self._record(
                request,
                f"resident_model_contract_{contract_rejection}",
            )
            return None

        native_engine, native_rejection = self._get_native_engine(
            request.model, contract)
        if native_engine is None:
            self._record(request, f"resident_native_engine_{native_rejection}")
            return None

        inputs = request.preprocess_inputs(request.inputs)
        outputs = native_engine.execute(
            request=request,
            contract=contract,
            invocation=invocation,
            inputs=inputs,
        )
        if outputs is None:
            native_execution_state = self._native_execution_state(
                native_engine)
            execution_reason = str(
                native_execution_state.get("reason",
                                           "resident_native_engine_declined"))
            attention_reason = str(
                native_execution_state.get(
                    "attention_dsa_native_dispatch_reason", "not_run"))
            if attention_reason != "not_run":
                execution_reason = f"{execution_reason}:{attention_reason}"
            self._record(
                request,
                f"resident_native_engine_declined:{execution_reason}",
            )
            return None

        native_execution_state = self._native_execution_state(native_engine)
        execution_reason = str(
            native_execution_state.get("reason",
                                       "resident_native_engine_executed"))
        self._native_shape_hist[invocation.stable_shape_key] += 1
        self._record(request, execution_reason)
        return PersistentDecodeModelForwardResult(
            outputs=outputs,
            backend=self.name,
            reason=execution_reason,
            metadata={"native_execution_state": native_execution_state},
        )

    def window_backend_state(self) -> dict[str, Any]:
        native_engine = self._native_engine
        if native_engine is None:
            return {
                "backend": "deepseek_resident_window_native_v1",
                "ready": False,
                "reason": "resident_window_native_engine_not_created",
                "metadata": {
                    "model_backend": self.name,
                    "native_engine_rejection": self._native_engine_rejection,
                },
            }
        state_fn = getattr(native_engine, "window_backend_state", None)
        if not callable(state_fn):
            return {
                "backend": "deepseek_resident_window_native_v1",
                "ready": False,
                "reason": "resident_window_native_state_missing",
                "metadata": {
                    "model_backend": self.name,
                },
            }
        try:
            state = state_fn()
        except (RuntimeError, TypeError) as exc:
            return {
                "backend": "deepseek_resident_window_native_v1",
                "ready": False,
                "reason":
                f"resident_window_native_state_failed:{type(exc).__name__}",
                "metadata": {
                    "model_backend": self.name,
                },
            }
        if isinstance(state, dict):
            return dict(state)
        return {
            "backend": "deepseek_resident_window_native_v1",
            "ready": False,
            "reason": "resident_window_native_state_invalid",
            "metadata": {
                "model_backend": self.name,
            },
        }

    def try_execute_window(
        self,
        request: PersistentDecodeModelWindowRequest,
    ) -> PersistentDecodeModelWindowResult | None:
        forward_request = request.forward_request
        reason = self._reject_reason(forward_request)
        if reason != self._CONTRACT_READY_REASON:
            self._record(forward_request, reason)
            return self._decline_window(request, reason)

        invocation, invocation_rejection = (
            build_deepseek_resident_invocation_contract(forward_request))
        if invocation is None:
            reason = (
                f"resident_native_window_invocation_{invocation_rejection}")
            self._record(forward_request, reason)
            return self._decline_window(request, reason)
        if not self._window_local_batch_admitted(invocation):
            reason = self._window_local_batch_decline_reason(invocation)
            self._record(forward_request, reason)
            return self._decline_window(request, reason)

        window_contract, window_rejection = (
            build_persistent_decode_model_window_contract(request, invocation))
        if window_contract is None:
            reason = f"resident_native_window_contract_{window_rejection}"
            self._record(forward_request, reason)
            return self._decline_window(request, reason)

        contract, contract_rejection = self._get_contract(forward_request.model)
        if contract is None:
            reason = f"resident_model_contract_{contract_rejection}"
            self._record(forward_request, reason)
            return self._decline_window(request, reason)

        native_engine, native_rejection = self._get_native_engine(
            forward_request.model, contract)
        if native_engine is None:
            reason = f"resident_native_window_engine_{native_rejection}"
            self._record(forward_request, reason)
            return self._decline_window(request, reason)

        execute_window = getattr(native_engine, "execute_window", None)
        if not callable(execute_window):
            reason = "resident_native_window_execute_missing"
            self._record(forward_request, reason)
            return self._decline_window(request, reason)

        with torch.inference_mode():
            inputs = forward_request.preprocess_inputs(forward_request.inputs)
            result = execute_window(
                request=request,
                contract=contract,
                invocation=invocation,
                window_contract=window_contract,
                inputs=inputs,
            )
        if result is None:
            reason = self._native_window_decline_reason(native_engine)
            self._record(forward_request, reason)
            return PersistentDecodeModelWindowResult(
                sample_state=request.sample_state,
                owned_steps=1,
                requested_window_steps=request.requested_window_steps,
                break_reason=reason,
                executed=False,
            )
        if isinstance(result, PersistentDecodeModelWindowResult):
            self._record(forward_request, result.break_reason)
            return result
        if all(
                hasattr(result, attr) for attr in (
                    "sample_state",
                    "owned_steps",
                    "requested_window_steps",
                    "break_reason",
                )):
            window_result = PersistentDecodeModelWindowResult(
                sample_state=result.sample_state,
                owned_steps=int(result.owned_steps),
                requested_window_steps=int(result.requested_window_steps),
                break_reason=str(result.break_reason),
                executed=bool(getattr(result, "executed", True)),
            )
            self._record(forward_request, window_result.break_reason)
            return window_result
        self._record(forward_request, "resident_native_window_result_invalid")
        return self._decline_window(request,
                                    "resident_native_window_result_invalid")

    def _decline_window(
        self,
        request: PersistentDecodeModelWindowRequest,
        reason: str,
    ) -> PersistentDecodeModelWindowResult:
        return PersistentDecodeModelWindowResult(
            sample_state=request.sample_state,
            owned_steps=1,
            requested_window_steps=request.requested_window_steps,
            break_reason=reason,
            executed=False,
        )

    def _native_window_decline_reason(self, native_engine: Any) -> str:
        state_fn = getattr(native_engine, "window_backend_state", None)
        if not callable(state_fn):
            return "resident_native_window_declined"
        try:
            state = state_fn()
        except (RuntimeError, TypeError) as exc:
            return f"resident_native_window_declined:{type(exc).__name__}"
        if not isinstance(state, dict):
            return "resident_native_window_declined"
        metadata = state.get("metadata")
        if isinstance(metadata, dict):
            reason = metadata.get("last_window_reason")
            if isinstance(reason, str) and reason:
                return f"resident_native_window_declined:{reason}"
        reason = state.get("reason")
        if isinstance(reason, str) and reason:
            return f"resident_native_window_declined:{reason}"
        return "resident_native_window_declined"

    def _native_execution_state(self, native_engine: Any) -> dict[str, Any]:
        execution_state = getattr(native_engine, "execution_state", None)
        if not callable(execution_state):
            return {"reason": "resident_native_engine_executed"}
        try:
            state = execution_state()
        except (RuntimeError, TypeError):
            return {"reason": "resident_native_engine_executed"}
        if not isinstance(state, dict):
            return {"reason": "resident_native_engine_executed"}
        return state

    def _get_native_engine(
        self,
        model: Any,
        contract: DeepSeekResidentBodyContract,
    ) -> tuple[DeepSeekResidentNativeEngine | None, str | None]:
        model_id = id(model)
        if self._native_engine_model_id == model_id:
            return self._native_engine, self._native_engine_rejection

        self._native_engine_model_id = model_id
        self._native_engine = None
        self._native_engine_rejection = None

        try:
            module = importlib.import_module(self._native_module_name)
        except ImportError as exc:
            self._native_engine_rejection = (
                f"module_import_failed:{type(exc).__name__}")
            return None, self._native_engine_rejection

        factory = getattr(module, "create_engine", None)
        if factory is None:
            self._native_engine_rejection = "missing_create_engine"
            return None, self._native_engine_rejection

        try:
            native_engine = factory(
                model=model,
                contract=contract,
                dist=self._dist,
            )
        except (TypeError, RuntimeError) as exc:
            self._native_engine_rejection = (
                f"create_engine_failed:{type(exc).__name__}")
            return None, self._native_engine_rejection

        if native_engine is None:
            self._native_engine_rejection = "create_engine_returned_none"
            return None, self._native_engine_rejection
        if not callable(getattr(native_engine, "execute", None)):
            self._native_engine_rejection = "engine_missing_execute"
            return None, self._native_engine_rejection

        self._native_engine = native_engine
        return self._native_engine, None

    def _local_batch_admitted(
        self,
        invocation: DeepSeekResidentInvocationContract,
    ) -> bool:
        return invocation.real_batch_size >= self._min_local_batch

    def _local_batch_decline_reason(
        self,
        invocation: DeepSeekResidentInvocationContract,
    ) -> str:
        return (
            "resident_native_min_local_batch_not_met:"
            f"{invocation.real_batch_size}<{self._min_local_batch}")

    def _window_local_batch_admitted(
        self,
        invocation: DeepSeekResidentInvocationContract,
    ) -> bool:
        if self._local_batch_admitted(invocation):
            return True
        return (self._min_local_batch <= 1 and invocation.real_batch_size == 0
                and invocation.padded_batch_size > 0)

    def _window_local_batch_decline_reason(
        self,
        invocation: DeepSeekResidentInvocationContract,
    ) -> str:
        return self._local_batch_decline_reason(invocation)


def build_deepseek_resident_body_contract(
    model: Any,
) -> tuple[DeepSeekResidentBodyContract | None, str | None]:
    """Inspect the loaded DeepSeek model and build the resident-body contract."""

    config = _pretrained_config_from_model(model)
    if config is None:
        return None, "missing_pretrained_config"

    model_body = getattr(model, "model", None)
    layers_obj = getattr(model_body, "layers", None)
    if layers_obj is None:
        return None, "missing_layers"

    try:
        layers = tuple(layers_obj)
    except TypeError:
        return None, "layers_not_iterable"

    num_layers = _as_optional_int(getattr(config, "num_hidden_layers", None))
    if num_layers is None or num_layers <= 0:
        return None, "invalid_num_hidden_layers"
    if len(layers) < num_layers:
        return None, "layer_count_short"

    layer_contracts: list[DeepSeekResidentLayerContract] = []
    layer_kind_hist: Counter[str] = Counter()
    gated_norm = bool(getattr(config, "gated_norm", False))
    attention_output_gate = bool(getattr(config, "attention_output_gate",
                                         False))

    for layer_idx, layer in enumerate(layers[:num_layers]):
        layer_contract, rejection = _build_layer_contract(
            layer_idx=layer_idx,
            layer=layer,
            gated_norm=gated_norm,
            attention_output_gate=attention_output_gate,
        )
        if layer_contract is None:
            return None, rejection
        layer_contracts.append(layer_contract)
        layer_kind_hist[layer_contract.layer_kind] += 1

    return DeepSeekResidentBodyContract(
        model_type=str(getattr(config, "model_type", "")),
        architectures=_tuple_of_str(getattr(config, "architectures", ())),
        num_layers=num_layers,
        hidden_size=_as_optional_int(getattr(config, "hidden_size", None)),
        vocab_size=_as_optional_int(getattr(config, "vocab_size", None)),
        rms_norm_eps=_as_optional_float(getattr(config, "rms_norm_eps", None)),
        gated_norm=gated_norm,
        attention_output_gate=attention_output_gate,
        n_routed_experts=_as_optional_int(
            getattr(config, "n_routed_experts", None)),
        num_experts_per_tok=_as_optional_int(
            getattr(config, "num_experts_per_tok", None)),
        n_group=_as_optional_int(getattr(config, "n_group", None)),
        topk_group=_as_optional_int(getattr(config, "topk_group", None)),
        routed_scaling_factor=_as_optional_float(
            getattr(config, "routed_scaling_factor", None)),
        first_k_dense_replace=_as_optional_int(
            getattr(config, "first_k_dense_replace", None)),
        moe_layer_freq=_as_optional_int(getattr(config, "moe_layer_freq",
                                                None)),
        layer_kind_hist=dict(layer_kind_hist),
        layers=tuple(layer_contracts),
    ), None


def build_deepseek_resident_invocation_contract(
    request: PersistentDecodeModelForwardRequest,
) -> tuple[DeepSeekResidentInvocationContract | None, str | None]:
    """Build the per-call ABI contract consumed by a native resident body."""

    plan = request.plan
    if plan is None:
        return None, "missing_plan"

    input_ids_spec = _tensor_spec("input_ids", request.inputs.get("input_ids"))
    if input_ids_spec is None:
        return None, "missing_input_ids_tensor_spec"

    input_tensor_specs = _input_tensor_specs(request.inputs)
    if not input_tensor_specs:
        return None, "missing_input_tensor_specs"

    real_batch_size = _as_optional_int(
        getattr(plan, "real_generation_requests", None))
    if real_batch_size is None or real_batch_size < 0:
        return None, "invalid_real_batch_size"

    padded_batch_size = _as_optional_int(
        getattr(plan, "padded_generation_requests", None))
    if padded_batch_size is None or padded_batch_size < real_batch_size:
        padded_batch_size = real_batch_size
    if padded_batch_size <= 0:
        return None, "invalid_padded_batch_size"

    request_ids = _tuple_of_ints(
        getattr(plan, "request_ids", None),
        fallback=getattr(request.attn_metadata, "request_ids", None),
    )
    seq_lens = _tuple_of_ints(
        getattr(plan, "seq_lens", None),
        fallback=getattr(request.attn_metadata, "seq_lens", None),
    )
    target_seq_lens = len(request_ids) if request_ids else real_batch_size
    if len(seq_lens) < target_seq_lens:
        seq_lens = seq_lens + (1, ) * (target_seq_lens - len(seq_lens))
    cached_tokens = _tuple_of_ints(
        getattr(plan, "cached_tokens", None),
        fallback=_cached_tokens_from_attention_metadata(request.attn_metadata),
    )

    return DeepSeekResidentInvocationContract(
        real_batch_size=real_batch_size,
        padded_batch_size=padded_batch_size,
        request_ids=request_ids,
        seq_lens=seq_lens,
        cached_tokens=cached_tokens,
        input_tokens=_as_optional_int(getattr(plan, "input_tokens", None))
        or _first_dim(input_ids_spec),
        cuda_graph_replay=bool(getattr(plan, "cuda_graph_replay",
                                       request.can_run_graph)),
        cuda_graph_padding=bool(getattr(plan, "cuda_graph_padding", False)),
        attention_metadata_type=str(
            getattr(
                plan,
                "attention_metadata_type",
                type(request.attn_metadata).__name__,
            )),
        kv_cache_manager_type=str(
            getattr(
                plan,
                "kv_cache_manager_type",
                type(request.kv_cache_manager).__name__,
            )),
        input_tensor_specs=input_tensor_specs,
        gather_ids_spec=_tensor_spec("gather_ids", request.gather_ids),
    ), None


def build_persistent_decode_model_window_contract(
    request: PersistentDecodeModelWindowRequest,
    invocation: DeepSeekResidentInvocationContract,
) -> tuple[PersistentDecodeModelWindowContract | None, str | None]:
    """Build the stable multi-step decode-window contract.

    The first native resident window intentionally supports only the serving
    subset that can stay resident across several decode steps. More general
    request features keep using the normal executor path until their host and
    device state transitions are explicitly represented here.
    """

    requested_window_steps = _as_optional_int(request.requested_window_steps)
    if requested_window_steps is None or requested_window_steps <= 1:
        return None, "invalid_requested_window_steps"

    sample_state = request.sample_state
    sample_requests = getattr(sample_state, "requests", None)
    if sample_requests is None:
        return None, "missing_sample_requests"
    try:
        requests = tuple(sample_requests)
    except TypeError:
        return None, "sample_requests_not_iterable"
    if not requests:
        return None, "empty_sample_requests"

    sequences: list[PersistentDecodeModelWindowSequenceContract] = []
    real_request_ids: list[int] = []
    min_remaining_decode_steps: int | None = None
    allow_streaming = _resident_window_allow_streaming()
    for sequence_idx, sequence_request in enumerate(requests):
        is_dummy_request = _is_dummy_request_like(sequence_request)
        request_id = _as_optional_int(
            getattr(sequence_request, "py_request_id", None))
        if request_id is None:
            return None, f"request_{sequence_idx}_missing_request_id"

        seq_slot = _as_optional_int(
            getattr(sequence_request, "py_seq_slot", None))
        if seq_slot is None:
            return None, f"request_{request_id}_missing_seq_slot"

        max_new_tokens = _as_optional_int(
            getattr(sequence_request, "py_max_new_tokens", None))
        if max_new_tokens is None:
            return None, f"request_{request_id}_missing_max_new_tokens"

        decoding_iter = _as_optional_int(
            getattr(sequence_request, "py_decoding_iter", None))
        if decoding_iter is None:
            return None, f"request_{request_id}_missing_decoding_iter"

        remaining_decode_steps = max_new_tokens - decoding_iter
        if is_dummy_request and remaining_decode_steps <= 1:
            remaining_decode_steps = requested_window_steps + 1
            max_new_tokens = decoding_iter + remaining_decode_steps
        elif remaining_decode_steps <= 1:
            return None, f"request_{request_id}_insufficient_remaining_decode_steps"

        beam_width = _as_optional_int(
            getattr(sequence_request, "py_beam_width", None))
        if beam_width is None and is_dummy_request:
            beam_width = 1
        elif beam_width is None:
            return None, f"request_{request_id}_missing_beam_width"
        if not is_dummy_request and beam_width != 1:
            return None, f"request_{request_id}_beam_width_gt_one"

        streaming = bool(getattr(sequence_request, "streaming", False))
        if not is_dummy_request and streaming and not allow_streaming:
            return None, f"request_{request_id}_streaming"

        return_log_probs = bool(
            getattr(sequence_request, "py_return_log_probs", False))
        if not is_dummy_request and return_log_probs:
            return None, f"request_{request_id}_return_log_probs"

        return_generation_logits = bool(
            getattr(sequence_request, "py_return_generation_logits", False))
        if not is_dummy_request and return_generation_logits:
            return None, f"request_{request_id}_return_generation_logits"

        stop_words_present = bool(
            getattr(sequence_request, "py_stop_words_list", None))
        if not is_dummy_request and stop_words_present:
            return None, f"request_{request_id}_stop_words"

        end_id = getattr(sequence_request, "py_end_id", None)
        end_id_value = _as_optional_int(end_id)
        if is_dummy_request and end_id_value is None:
            end_id_value = -1
        elif not is_dummy_request and end_id_value != -1:
            return None, f"request_{request_id}_end_id"

        draft_token_count = _draft_token_count(sequence_request)
        if not is_dummy_request and draft_token_count:
            return None, f"request_{request_id}_draft_tokens"

        sequences.append(
            PersistentDecodeModelWindowSequenceContract(
                request_id=request_id,
                seq_slot=seq_slot,
                max_new_tokens=max_new_tokens,
                decoding_iter=decoding_iter,
                remaining_decode_steps=remaining_decode_steps,
                beam_width=beam_width,
                streaming=streaming,
                return_log_probs=return_log_probs,
                return_generation_logits=return_generation_logits,
                stop_words_present=stop_words_present,
                end_id=end_id_value,
                draft_token_count=draft_token_count,
                state=_state_name(sequence_request),
            ))
        if not is_dummy_request:
            real_request_ids.append(request_id)
            if min_remaining_decode_steps is None:
                min_remaining_decode_steps = remaining_decode_steps
            else:
                min_remaining_decode_steps = min(
                    min_remaining_decode_steps,
                    remaining_decode_steps,
                )

    all_request_ids = tuple(sequence.request_id for sequence in sequences)
    request_ids = tuple(real_request_ids) if real_request_ids else all_request_ids
    invocation_real_request_ids = tuple(
        request_id for request_id in invocation.request_ids
        if not _is_padding_request_id(request_id))
    if (invocation.request_ids and invocation.request_ids not in (
            request_ids, all_request_ids)
            and invocation_real_request_ids != request_ids):
        return None, _request_ids_mismatch_reason(
            invocation.request_ids,
            invocation_real_request_ids,
            request_ids,
            all_request_ids,
        )

    lane_indices = _window_metadata_lane_indices(
        invocation.request_ids,
        has_real_requests=bool(real_request_ids),
    )
    seq_lens = _project_window_seq_lens(
        invocation.seq_lens,
        lane_indices,
        len(request_ids),
    )
    cached_tokens = _project_window_cached_tokens(
        invocation.cached_tokens,
        lane_indices,
        len(request_ids),
    )
    if cached_tokens is None:
        return None, "cached_tokens_missing_for_window_requests"

    seq_slots = tuple(sequence.seq_slot for sequence in sequences)
    if len(set(seq_slots)) != len(seq_slots):
        return None, "duplicate_seq_slots"

    forward_request = request.forward_request
    input_ids_spec = _tensor_spec("input_ids",
                                  forward_request.inputs.get("input_ids"))
    if input_ids_spec is None:
        return None, "missing_input_ids_tensor_spec"
    position_ids_spec = _tensor_spec("position_ids",
                                     forward_request.inputs.get("position_ids"))

    terminal_reserve_steps = 0 if _resident_window_terminal_enabled() else 1
    max_safe_owned_steps = min(
        requested_window_steps,
        max(1,
            int(min_remaining_decode_steps
                if min_remaining_decode_steps is not None else
                requested_window_steps + 1) - terminal_reserve_steps),
    )
    if max_safe_owned_steps <= 1:
        return None, "insufficient_safe_owned_steps"

    return PersistentDecodeModelWindowContract(
        requested_window_steps=requested_window_steps,
        already_sampled_steps=1,
        requested_additional_steps=requested_window_steps - 1,
        max_safe_owned_steps=max_safe_owned_steps,
        request_ids=request_ids,
        seq_slots=seq_slots,
        sequences=tuple(sequences),
        input_ids_spec=input_ids_spec,
        position_ids_spec=position_ids_spec,
        attention_metadata_type=invocation.attention_metadata_type,
        kv_cache_manager_type=invocation.kv_cache_manager_type,
        cached_tokens=cached_tokens,
        seq_lens=seq_lens,
        stable_shape_key=invocation.stable_shape_key,
    ), None


def _build_layer_contract(
    *,
    layer_idx: int,
    layer: Any,
    gated_norm: bool,
    attention_output_gate: bool,
) -> tuple[DeepSeekResidentLayerContract | None, str | None]:
    attention = getattr(layer, "self_attn", None)
    if attention is None:
        return None, f"layer_{layer_idx}_missing_self_attn"
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return None, f"layer_{layer_idx}_missing_mlp"

    input_layernorm = getattr(layer, "input_layernorm", None)
    post_attention_layernorm = getattr(layer, "post_attention_layernorm", None)
    next_layer_layernorm = getattr(layer, "next_layer_layernorm", None)
    if input_layernorm is None:
        return None, f"layer_{layer_idx}_missing_input_layernorm"
    if post_attention_layernorm is None:
        return None, f"layer_{layer_idx}_missing_post_attention_layernorm"
    if next_layer_layernorm is None:
        return None, f"layer_{layer_idx}_missing_next_layer_layernorm"

    input_gate_down = getattr(layer, "input_gated_norm_down", None)
    input_gate_up = getattr(layer, "input_gated_norm_up", None)
    post_gate_down = getattr(layer, "post_attention_gated_norm_down", None)
    post_gate_up = getattr(layer, "post_attention_gated_norm_up", None)
    if gated_norm and (input_gate_down is None or input_gate_up is None):
        return None, f"layer_{layer_idx}_missing_input_gated_norm"
    if gated_norm and (post_gate_down is None or post_gate_up is None):
        return None, f"layer_{layer_idx}_missing_post_attention_gated_norm"

    has_attention_output_gate = getattr(attention, "gate_proj", None) is not None
    if attention_output_gate and not has_attention_output_gate:
        return None, f"layer_{layer_idx}_missing_attention_output_gate"

    has_kv_a_projection = getattr(attention, "kv_a_proj_with_mqa", None) is not None
    if not has_kv_a_projection:
        return None, f"layer_{layer_idx}_missing_kv_a_projection"

    layer_kind = _layer_kind(mlp)
    fusion_config = getattr(layer, "fusion_config", None)
    return DeepSeekResidentLayerContract(
        layer_idx=layer_idx,
        layer_kind=layer_kind,
        attention_type=type(attention).__name__,
        mlp_type=type(mlp).__name__,
        has_input_gated_norm=input_gate_down is not None
        and input_gate_up is not None,
        has_post_attention_gated_norm=post_gate_down is not None
        and post_gate_up is not None,
        input_gate_rank=_linear_out_features(input_gate_down),
        post_attention_gate_rank=_linear_out_features(post_gate_down),
        has_attention_output_gate=has_attention_output_gate,
        has_kv_a_projection=has_kv_a_projection,
        has_next_layer_layernorm=next_layer_layernorm is not None,
        disable_attn_allreduce=bool(
            getattr(layer, "disable_attn_allreduce", False)),
        pre_mlp_or_moe_fusion=bool(
            getattr(fusion_config, "PRE_MLP_FUSION", False)
            or getattr(fusion_config, "PRE_MOE_FUSION", False)),
        post_mlp_or_moe_fusion=bool(
            getattr(fusion_config, "POST_MLP_FUSION", False)
            or getattr(fusion_config, "POST_MOE_FUSION", False)),
        num_experts=_as_optional_int(getattr(layer, "num_experts", None))
        if layer_kind == "moe" else None,
        top_k=_as_optional_int(getattr(layer, "top_k", None))
        if layer_kind == "moe" else None,
    ), None


def _pretrained_config_from_model(model: Any) -> Any:
    model_config = getattr(model, "model_config", None)
    config = getattr(model_config, "pretrained_config", None)
    return config if config is not None else getattr(model, "config", None)


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _tuple_of_str(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value, )
    try:
        return tuple(str(item) for item in value)
    except TypeError:
        return ()


def _tuple_of_ints(value: Any, *, fallback: Any = None) -> tuple[int, ...]:
    values = _tensor_to_list(value)
    if not values and fallback is not None:
        values = _tensor_to_list(fallback)
    result: list[int] = []
    for item in values:
        int_value = _as_optional_int(item)
        if int_value is not None:
            result.append(int_value)
    return tuple(result)


def _tensor_to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
    except RuntimeError:
        return []
    if isinstance(value, int):
        return [value]
    try:
        return list(value)
    except TypeError:
        return []


def _cached_tokens_from_attention_metadata(attn_metadata: Any) -> Any:
    kv_cache_params = getattr(attn_metadata, "kv_cache_params", None)
    return getattr(kv_cache_params, "num_cached_tokens_per_seq", None)


def _input_tensor_specs(
    inputs: dict[str, Any],
) -> tuple[PersistentDecodeTensorSpec, ...]:
    specs: list[PersistentDecodeTensorSpec] = []
    for name, value in sorted(inputs.items()):
        spec = _tensor_spec(name, value)
        if spec is not None:
            specs.append(spec)
    return tuple(specs)


def _tensor_spec(
    name: str,
    value: Any,
) -> PersistentDecodeTensorSpec | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        shape_tuple = tuple(int(dim) for dim in shape)
    except (TypeError, ValueError):
        return None
    return PersistentDecodeTensorSpec(
        name=name,
        shape=shape_tuple,
        dtype=str(getattr(value, "dtype", "")),
        device=str(getattr(value, "device", "")),
    )


def _first_dim(spec: PersistentDecodeTensorSpec) -> int:
    if not spec.shape:
        return 1
    return spec.shape[0]


def _is_dummy_request_like(request: Any) -> bool:
    return bool(
        getattr(request, "is_attention_dp_dummy", False)
        or getattr(request, "is_cuda_graph_dummy", False)
        or getattr(request, "is_dummy_request", False))


def _draft_token_count(request: Any) -> int:
    draft_tokens = getattr(request, "py_draft_tokens", None)
    if draft_tokens is None:
        context_phase_params = getattr(request, "context_phase_params", None)
        draft_tokens = getattr(context_phase_params, "draft_tokens", None)
    if draft_tokens is None:
        return 0
    try:
        return len(draft_tokens)
    except TypeError:
        return 0


def _state_name(request: Any) -> str:
    state = getattr(request, "state", None)
    name = getattr(state, "name", None)
    if name is not None:
        return str(name)
    return str(state)


def _linear_out_features(linear: Any) -> int | None:
    weight = getattr(linear, "weight", None)
    shape = getattr(weight, "shape", None)
    if shape is None:
        return None
    try:
        return int(shape[0])
    except (TypeError, ValueError, IndexError):
        return None


def _layer_kind(mlp: Any) -> str:
    mlp_type = type(mlp).__name__.lower()
    if "moe" in mlp_type:
        return "moe"
    return "dense"


def create_persistent_decode_model_backend(
        dist: Any) -> PersistentDecodeModelBackend | None:
    backend_name = os.environ.get(_BACKEND_ENV_NAME, "disabled").strip()
    if backend_name.lower() in _DISABLED_BACKEND_NAMES:
        return None
    if not _rank_enabled(getattr(dist, "rank", None)):
        return None
    if backend_name == _DEEPSEEK_RESIDENT_V0_BACKEND_NAME:
        return DeepSeekResidentModelBackendV0(dist)
    if backend_name == _DEEPSEEK_RESIDENT_PYTHON_V1_BACKEND_NAME:
        return DeepSeekResidentModelBackendPythonV1(dist)
    if backend_name == _DEEPSEEK_RESIDENT_NATIVE_V1_BACKEND_NAME:
        return DeepSeekResidentModelBackendNativeV1(dist)
    raise ValueError(
        f"Unsupported {_BACKEND_ENV_NAME}={backend_name!r}; supported "
        f"backends: {_DEEPSEEK_RESIDENT_V0_BACKEND_NAME!r}, "
        f"{_DEEPSEEK_RESIDENT_PYTHON_V1_BACKEND_NAME!r}, "
        f"{_DEEPSEEK_RESIDENT_NATIVE_V1_BACKEND_NAME!r}")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _rank_enabled(rank: int | None) -> bool:
    raw = os.environ.get(_RANKS_ENV_NAME, "0").strip()
    if raw in ("*", "all", "ALL"):
        return True
    enabled = {item.strip() for item in raw.split(",") if item.strip()}
    return str(rank) in enabled


def _resident_window_allow_streaming() -> bool:
    return os.environ.get(_RESIDENT_ALLOW_STREAMING_ENV_NAME, "0") == "1"


def _resident_window_terminal_enabled() -> bool:
    return os.environ.get(_RESIDENT_TERMINAL_WINDOW_ENV_NAME, "0") == "1"

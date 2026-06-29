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

"""Executor-level handoff for a resident decode engine.

The model-forward planner observes a single TRT-LLM forward call. A TileRT-style
engine needs a higher-level boundary: it must own the model step, sampling, token
feedback, request-state update, and token egress for several decode iterations.
This module is the default-off executor-level takeover point and the backend
boundary where a model-specific resident DeepSeek path can replace the current
Python loop.
"""

from __future__ import annotations

import os
from concurrent import futures
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from tensorrt_llm.logger import logger

_ENABLE_ENV_NAME = "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE"
_DEBUG_ENV_NAME = "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_DEBUG"
_TRACE_ENV_NAME = "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_TRACE"
_REPORT_EVERY_ENV_NAME = "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_EVERY"
_RANKS_ENV_NAME = "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RANKS"
_WINDOW_STEPS_ENV_NAME = "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_STEPS"
_ADP_WINDOW_ENV_NAME = "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_ADP_WINDOW"
_BACKEND_ENV_NAME = "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_BACKEND"
_RESIDENT_FLEX_STEPS_ENV_NAME = (
    "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_FLEX_STEPS")
_RESIDENT_TERMINAL_WINDOW_ENV_NAME = (
    "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_TERMINAL_WINDOW")
_ALLOW_SAMPLING_BRIDGE_ENV_NAME = (
    "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_ALLOW_SAMPLING_BRIDGE")

_DEFAULT_REPORT_EVERY = 128
_DEFAULT_WINDOW_STEPS = 1
_PYTHON_RESIDENT_BACKEND_NAME = "python_resident"
_DEEPSEEK_GRAPH_RESIDENT_BACKEND_NAME = "deepseek_graph_resident"
_DEEPSEEK_NATIVE_RESIDENT_BACKEND_NAME = "deepseek_native_resident"
_DEEPSEEK_NATIVE_MODEL_BACKEND_NAME = "deepseek_resident_native_v1"
_DEEPSEEK_NATIVE_SAMPLE_BACKEND_NAME = "deepseek_resident_sampler_native_v1"
_DEEPSEEK_NATIVE_WINDOW_BACKEND_NAME = "deepseek_resident_window_native_v1"
_MODEL_ENGINE_CUDA_GRAPH_BACKEND_NAME = "model_engine_cuda_graph_replay_v1"
_PYEXECUTOR_SAMPLE_BRIDGE_BACKEND_NAME = "pyexecutor_sampling_bridge_v1"
_PYEXECUTOR_WINDOW_LOOP_BACKEND_NAME = "pyexecutor_window_loop_v1"
_NATIVE_MODEL_BODY_READY_REASONS = frozenset({
    "resident_stage_scheduler_completed",
    "resident_native_handle_decode_executed",
    "resident_native_op_decode_executed",
})
_NATIVE_SAMPLE_READY_REASONS = frozenset({
    "resident_sampling_native_ready",
    "resident_sampling_native_executed",
})
_PYEXECUTOR_SAMPLE_BRIDGE_READY_REASONS = frozenset({
    "resident_sampling_bridge_ready",
    "resident_sampling_bridge_executed",
})
_NATIVE_WINDOW_READY_REASONS = frozenset({
    "resident_window_native_ready",
    "resident_window_native_executed",
    "resident_window_stage_scheduler_ready",
    "resident_window_stage_scheduler_executed",
})
_MODEL_ENGINE_CUDA_GRAPH_READY_REASONS = frozenset({
    "cuda_graph_replay_ready",
    "cuda_graph_replay_executed",
})
_PYEXECUTOR_WINDOW_LOOP_READY_REASONS = frozenset({
    "resident_window_loop_bridge_ready",
    "resident_window_loop_bridge_executed",
})


@dataclass(frozen=True)
class PersistentDecodeEnginePlan:
    eligible: bool
    reason: str
    initial_takeover_contract_ready: bool
    initial_takeover_contract_reason: str
    batch_size: int
    request_ids: tuple[int, ...]
    cached_tokens: tuple[int, ...]
    has_previous_sample_state: bool
    has_device_token_feedback: bool
    previous_sample_state_type: str
    previous_device_new_tokens_shape: tuple[int, ...]
    previous_host_new_tokens_shape: tuple[int, ...]
    sampler_event_present: bool
    streaming_requests: int
    return_log_probs_requests: int
    max_beam_width: int
    real_generation_requests: int
    generation_to_complete_requests: int
    attention_dp_enabled: bool
    attention_dp_dummy_requests: int
    request_state_hist: tuple[tuple[str, int], ...]
    has_draft_batch: bool
    use_previous_draft_tokens: bool
    guided_decoder_present: bool
    enable_spec_decode: bool
    requested_window_steps: int
    stable_window_steps: int
    min_decoding_iter: int
    min_remaining_decode_steps: int
    window_contract_ready: bool
    window_contract_reason: str
    takeover_enabled: bool
    adp_window_enabled: bool


@dataclass(frozen=True)
class PersistentDecodeEngineResult:
    """Future result type once the resident engine owns a decode window."""

    batch_outputs: Any
    sample_state: Any
    owned_steps: int = 1


ExecutePersistentDecodeStep = Callable[[], PersistentDecodeEngineResult]


@dataclass(frozen=True)
class PersistentDecodeWindowResult:
    """Result from a backend-owned decode window."""

    scheduled_batch: Any
    sample_state: Any
    owned_steps: int
    requested_window_steps: int
    break_reason: str
    executed: bool


@dataclass(frozen=True)
class PersistentDecodeSampleStepResult:
    """Result from a resident sampling/token-feedback step."""

    sample_state: Any
    backend: str
    reason: str
    executed: bool = True
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class PersistentDecodeWindowCallbacks:
    """Executor services needed by the resident decode backend.

    The first implementation delegates to existing PyExecutor primitives. A
    TileRT-style backend should replace the forward/sample/token-feedback
    callbacks with a model-specific resident body while preserving the same
    request/KV/cache ownership contract.
    """

    initial_admitted: Callable[[Any, int], tuple[bool, str]]
    continue_admitted: Callable[[Any, tuple[int, ...], int], tuple[bool, str]]
    request_ids: Callable[[Any], tuple[int, ...]]
    can_defer_host_updates: Callable[[Any], bool]
    defer_update_requests: Callable[[Any], None]
    sample_has_finished_requests: Callable[[Any], tuple[bool, str]]
    update_requests: Callable[[Any], None]
    prepare_resources: Callable[[Any], None]
    forward_step: Callable[[Any, Any], Any]
    sample_async: Callable[[Any, Any], Any]
    update_request_states: Callable[[Any], None]
    increment_iter_counter: Callable[[], None]
    materialize_deferred_samples: Callable[[list[Any]], None]
    stage_start: Callable[[], int]
    stage_end: Callable[[str, int], None]
    record_timing: Callable[[int, int, str], None]
    log_execution: Callable[[dict[str, Any]], None]
    resident_sample_step: (
        Callable[[Any, Any], PersistentDecodeSampleStepResult | None] | None
    ) = None
    capture_deferred_sample: Callable[[Any], Any] | None = None
    materialize_deferred_captures: Callable[[list[Any]], None] | None = None
    can_accumulate_token_egress: Callable[[Any], bool] | None = None
    enqueue_deferred_samples: Callable[[list[Any]], None] | None = None
    model_body_backend_state: Callable[[], dict[str, Any]] | None = None
    model_graph_backend_state: Callable[[], dict[str, Any]] | None = None
    sample_backend_state: Callable[[], dict[str, Any]] | None = None
    window_backend_state: Callable[[], dict[str, Any]] | None = None
    resident_window_step: (
        Callable[[Any, Any, int], PersistentDecodeWindowResult | None] | None
    ) = None
    async_token_egress: bool = False


class PersistentDecodeBackend(Protocol):
    """Backend contract for a resident decode execution model."""

    name: str

    def execute_decode_window(
        self,
        *,
        scheduled_batch: Any,
        sample_state: Any,
        requested_window_steps: int,
        configured_window_steps: int,
        flex_steps: bool,
        callbacks: PersistentDecodeWindowCallbacks,
    ) -> PersistentDecodeWindowResult:
        """Own a decode window and return the final executor-visible state."""


class PersistentDecodeTokenEgress:
    """Window-local token egress for sampled states.

    In synchronous mode this captures sampled tokens at the window boundary. In
    async mode it starts that capture as soon as a sample state is available and
    applies the captured tokens at the same boundary. The worker side must only
    read immutable request identity/slot metadata and host sample tensors; the
    main executor still owns request mutation.
    """

    def __init__(
        self,
        materialize: Callable[[list[Any]], None],
        *,
        capture: Callable[[Any], Any] | None = None,
        materialize_captures: Callable[[list[Any]], None] | None = None,
        async_enabled: bool = False,
        stage_start: Callable[[], int] | None = None,
        stage_end: Callable[[str, int], None] | None = None,
    ) -> None:
        self._materialize = materialize
        self._capture = capture
        self._materialize_captures = materialize_captures
        self._stage_start = stage_start
        self._stage_end = stage_end
        self._async_enabled = (
            async_enabled and capture is not None
            and materialize_captures is not None)
        self._sample_states: list[Any] = []
        self._capture_futures: list[futures.Future[Any]] = []
        self._executor: futures.ThreadPoolExecutor | None = None
        if self._async_enabled:
            self._executor = futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="optrt-token-egress",
            )

    @property
    def pending_count(self) -> int:
        return len(self._sample_states) + len(self._capture_futures)

    def enqueue_sample_state(self, sample_state: Any) -> None:
        if self._async_enabled:
            assert self._executor is not None
            assert self._capture is not None
            self._capture_futures.append(
                self._executor.submit(self._capture, sample_state))
            return
        self._sample_states.append(sample_state)

    def flush(self) -> int:
        pending_count = self.pending_count
        if pending_count == 0:
            return 0

        if self._capture_futures:
            assert self._materialize_captures is not None
            captures = self._record_stage(
                "resident_token_egress_wait",
                lambda: [
                    future.result() for future in self._capture_futures
                ],
            )
            self._record_stage(
                "resident_token_egress_apply",
                lambda: self._materialize_captures(captures),
            )
            self._capture_futures.clear()
        elif self._capture is not None and self._materialize_captures is not None:
            captures = self._record_stage(
                "resident_token_egress_capture",
                lambda: [
                    self._capture(sample_state)
                    for sample_state in self._sample_states
                ],
            )
            self._record_stage(
                "resident_token_egress_apply",
                lambda: self._materialize_captures(captures),
            )
        else:
            self._record_stage(
                "resident_token_egress_legacy_materialize",
                lambda: self._materialize(self._sample_states),
            )

        self._sample_states.clear()
        return pending_count

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _record_stage(self, stage: str, fn: Callable[[], Any]) -> Any:
        start = 0 if self._stage_start is None else self._stage_start()
        try:
            return fn()
        finally:
            if self._stage_end is not None:
                self._stage_end(stage, start)


class PythonResidentDecodeBackend:
    """Bridge backend that keeps the current PyExecutor resident semantics."""

    name = _PYTHON_RESIDENT_BACKEND_NAME

    def execute_decode_window(
        self,
        *,
        scheduled_batch: Any,
        sample_state: Any,
        requested_window_steps: int,
        configured_window_steps: int,
        flex_steps: bool,
        callbacks: PersistentDecodeWindowCallbacks,
    ) -> PersistentDecodeWindowResult:
        cohort_request_ids = callbacks.request_ids(scheduled_batch)

        def reject(
            reason: str,
            *,
            defer_host_updates: bool | None = None,
            native_window: bool | None = None,
        ) -> PersistentDecodeWindowResult:
            callbacks.log_execution({
                "owned_steps": 1,
                "requested_window_steps": requested_window_steps,
                "configured_window_steps": configured_window_steps,
                "admission_reason": reason,
                "break_reason": reason,
                "defer_host_updates": defer_host_updates,
                "accumulate_token_egress": None,
                "flex_steps": flex_steps,
                "cohort_request_ids": cohort_request_ids,
                "backend": self.name,
                "sample_backend": _callback_state_backend(
                    callbacks.sample_backend_state),
                "window_backend": _callback_state_backend(
                    callbacks.window_backend_state),
                "native_window": native_window,
                "async_token_egress": callbacks.async_token_egress,
                "executed": False,
            })
            return PersistentDecodeWindowResult(
                scheduled_batch=scheduled_batch,
                sample_state=sample_state,
                owned_steps=1,
                requested_window_steps=requested_window_steps,
                break_reason=reason,
                executed=False,
            )

        admitted, admission_reason = callbacks.initial_admitted(
            scheduled_batch, requested_window_steps)
        if not admitted:
            return reject(admission_reason)
        backend_admitted, backend_reason = self._resident_backend_admitted(
            callbacks, requested_window_steps)
        if not backend_admitted:
            return reject(backend_reason)

        break_reason = "resident_completed_requested_window"
        owned_steps = 1
        current_batch = scheduled_batch
        current_sample_state = sample_state
        defer_host_updates = callbacks.can_defer_host_updates(current_batch)
        window_admitted, window_reason = self._window_backend_admitted(
            callbacks, requested_window_steps)
        if not window_admitted:
            return reject(window_reason, defer_host_updates=defer_host_updates)
        host_update_admitted, host_update_reason = (
            self._host_update_mode_admitted(defer_host_updates))
        if not host_update_admitted:
            return reject(host_update_reason,
                          defer_host_updates=defer_host_updates)

        resident_window_result = self._resident_window_result(
            scheduled_batch=scheduled_batch,
            sample_state=sample_state,
            requested_window_steps=requested_window_steps,
            configured_window_steps=configured_window_steps,
            flex_steps=flex_steps,
            callbacks=callbacks,
        )
        if resident_window_result is not None:
            if resident_window_result.executed:
                callbacks.record_timing(
                    resident_window_result.owned_steps,
                    resident_window_result.requested_window_steps,
                    resident_window_result.break_reason,
                )
                callbacks.log_execution({
                    "owned_steps": resident_window_result.owned_steps,
                    "requested_window_steps":
                    resident_window_result.requested_window_steps,
                    "configured_window_steps": configured_window_steps,
                    "admission_reason": admission_reason,
                    "break_reason": resident_window_result.break_reason,
                    "defer_host_updates": defer_host_updates,
                    "accumulate_token_egress": None,
                    "flex_steps": flex_steps,
                    "cohort_request_ids": cohort_request_ids,
                    "backend": self.name,
                    "sample_backend": _callback_state_backend(
                        callbacks.sample_backend_state),
                    "window_backend": _callback_state_backend(
                        callbacks.window_backend_state),
                    "native_window": True,
                    "async_token_egress": callbacks.async_token_egress,
                })
            else:
                callbacks.log_execution({
                    "owned_steps": resident_window_result.owned_steps,
                    "requested_window_steps":
                    resident_window_result.requested_window_steps,
                    "configured_window_steps": configured_window_steps,
                    "admission_reason": admission_reason,
                    "break_reason": resident_window_result.break_reason,
                    "defer_host_updates": defer_host_updates,
                    "accumulate_token_egress": None,
                    "flex_steps": flex_steps,
                    "cohort_request_ids": cohort_request_ids,
                    "backend": self.name,
                    "sample_backend": _callback_state_backend(
                        callbacks.sample_backend_state),
                    "window_backend": _callback_state_backend(
                        callbacks.window_backend_state),
                    "native_window": True,
                    "async_token_egress": callbacks.async_token_egress,
                    "executed": False,
                })
            return resident_window_result

        window_start = callbacks.stage_start()
        accumulate_token_egress = (
            defer_host_updates
            and callbacks.can_accumulate_token_egress is not None
            and callbacks.enqueue_deferred_samples is not None
            and callbacks.can_accumulate_token_egress(current_batch))
        deferred_sample_states: list[Any] = []
        token_egress = PersistentDecodeTokenEgress(
            callbacks.materialize_deferred_samples,
            capture=callbacks.capture_deferred_sample,
            materialize_captures=callbacks.materialize_deferred_captures,
            async_enabled=callbacks.async_token_egress,
            stage_start=callbacks.stage_start,
            stage_end=callbacks.stage_end,
        )

        try:
            while owned_steps < requested_window_steps:
                if defer_host_updates:
                    stage_start = callbacks.stage_start()
                    callbacks.defer_update_requests(current_sample_state)
                    if accumulate_token_egress:
                        deferred_sample_states.append(current_sample_state)
                    else:
                        token_egress.enqueue_sample_state(current_sample_state)
                    callbacks.stage_end("resident_defer_update_requests",
                                        stage_start)
                else:
                    stage_start = callbacks.stage_start()
                    sample_finished, finish_reason = (
                        callbacks.sample_has_finished_requests(
                            current_sample_state))
                    callbacks.stage_end("resident_sample_finish_check",
                                        stage_start)
                    if sample_finished:
                        break_reason = f"resident_{finish_reason}"
                        break

                    stage_start = callbacks.stage_start()
                    callbacks.update_requests(current_sample_state)
                    callbacks.stage_end("resident_update_requests",
                                        stage_start)

                stage_start = callbacks.stage_start()
                continue_admitted, continue_reason = callbacks.continue_admitted(
                    current_batch, cohort_request_ids, requested_window_steps)
                callbacks.stage_end("resident_continue_admission",
                                    stage_start)
                if not continue_admitted:
                    break_reason = f"resident_{continue_reason}"
                    break

                stage_start = callbacks.stage_start()
                callbacks.prepare_resources(current_batch)
                callbacks.stage_end("resident_prepare_resources", stage_start)

                stage_start = callbacks.stage_start()
                next_outputs = callbacks.forward_step(
                    current_batch, current_sample_state.device)
                callbacks.stage_end("resident_forward_step", stage_start)
                if next_outputs is None:
                    break_reason = "resident_forward_failed"
                    break
                forward_admitted, forward_reason = (
                    self._forward_step_admitted(callbacks))
                if not forward_admitted:
                    break_reason = forward_reason
                    break

                stage_start = callbacks.stage_start()
                next_sample_state, sample_reason = self._sample_step(
                    callbacks, current_batch, next_outputs)
                callbacks.stage_end(self._sample_stage_name(callbacks),
                                    stage_start)
                if next_sample_state is None:
                    break_reason = sample_reason
                    break

                stage_start = callbacks.stage_start()
                callbacks.update_request_states(current_batch)
                callbacks.stage_end("resident_update_request_states",
                                    stage_start)

                current_sample_state = next_sample_state
                owned_steps += 1
                callbacks.increment_iter_counter()

            if deferred_sample_states:
                assert callbacks.enqueue_deferred_samples is not None
                stage_start = callbacks.stage_start()
                callbacks.enqueue_deferred_samples(deferred_sample_states)
                callbacks.stage_end("resident_enqueue_deferred_egress",
                                    stage_start)
            if token_egress.pending_count:
                stage_start = callbacks.stage_start()
                token_egress.flush()
                callbacks.stage_end("resident_materialize_deferred",
                                    stage_start)
        finally:
            token_egress.close()

        callbacks.stage_end("resident_total_window", window_start)
        callbacks.record_timing(owned_steps, requested_window_steps,
                                break_reason)
        callbacks.log_execution({
            "owned_steps": owned_steps,
            "requested_window_steps": requested_window_steps,
            "configured_window_steps": configured_window_steps,
            "admission_reason": admission_reason,
            "break_reason": break_reason,
            "defer_host_updates": defer_host_updates,
            "accumulate_token_egress": accumulate_token_egress,
            "flex_steps": flex_steps,
            "cohort_request_ids": cohort_request_ids,
            "backend": self.name,
            "sample_backend": _callback_state_backend(
                callbacks.sample_backend_state),
            "window_backend": _callback_state_backend(
                callbacks.window_backend_state),
            "async_token_egress": callbacks.async_token_egress,
        })

        return PersistentDecodeWindowResult(
            scheduled_batch=current_batch,
            sample_state=current_sample_state,
            owned_steps=owned_steps,
            requested_window_steps=requested_window_steps,
            break_reason=break_reason,
            executed=True,
        )

    def _resident_backend_admitted(
        self,
        callbacks: PersistentDecodeWindowCallbacks,
        requested_window_steps: int,
    ) -> tuple[bool, str]:
        return True, "ready"

    def _forward_step_admitted(
        self,
        callbacks: PersistentDecodeWindowCallbacks,
    ) -> tuple[bool, str]:
        return True, "ready"

    def _sample_step(
        self,
        callbacks: PersistentDecodeWindowCallbacks,
        batch: Any,
        outputs: Any,
    ) -> tuple[Any | None, str]:
        if callbacks.resident_sample_step is None:
            sample_state = callbacks.sample_async(batch, outputs)
            if sample_state is None:
                return None, "resident_sample_failed"
            return sample_state, "resident_sample_async_executed"

        sample_result = callbacks.resident_sample_step(batch, outputs)
        if sample_result is None:
            return None, "resident_sample_step_failed"
        admitted, reason = self._sample_step_admitted(callbacks, sample_result)
        if not admitted:
            return None, reason
        if sample_result.sample_state is None:
            return None, sample_result.reason
        return sample_result.sample_state, sample_result.reason

    def _sample_step_admitted(
        self,
        callbacks: PersistentDecodeWindowCallbacks,
        sample_result: PersistentDecodeSampleStepResult,
    ) -> tuple[bool, str]:
        return True, "ready"

    def _host_update_mode_admitted(
        self,
        defer_host_updates: bool,
    ) -> tuple[bool, str]:
        return True, "ready"

    def _window_backend_admitted(
        self,
        callbacks: PersistentDecodeWindowCallbacks,
        requested_window_steps: int,
    ) -> tuple[bool, str]:
        return True, "ready"

    def _resident_window_result(
        self,
        *,
        scheduled_batch: Any,
        sample_state: Any,
        requested_window_steps: int,
        configured_window_steps: int,
        flex_steps: bool,
        callbacks: PersistentDecodeWindowCallbacks,
    ) -> PersistentDecodeWindowResult | None:
        return None

    def _sample_stage_name(
        self,
        callbacks: PersistentDecodeWindowCallbacks,
    ) -> str:
        if callbacks.resident_sample_step is not None:
            return "resident_sample_step"
        return "resident_sample_async"


class DeepSeekGraphResidentDecodeBackend(PythonResidentDecodeBackend):
    """Resident window backend that preserves production CUDA graph replay.

    This is the serving baseline for the TileRT-style execution boundary: the
    executor owns a multi-step decode window, while model-body execution still
    goes through ``ModelEngine.forward`` and its CUDA graph capture/replay path.
    It deliberately rejects the experimental native model/window backends so
    that a host-dispatched C++ stage loop cannot be mistaken for the production
    graph path.
    """

    name = _DEEPSEEK_GRAPH_RESIDENT_BACKEND_NAME

    def _resident_backend_admitted(
        self,
        callbacks: PersistentDecodeWindowCallbacks,
        requested_window_steps: int,
    ) -> tuple[bool, str]:
        model_body_ready, model_body_reason = (
            self._model_body_backend_not_configured(callbacks))
        if not model_body_ready:
            return False, model_body_reason
        graph_ready, graph_reason = self._model_graph_backend_ready(callbacks)
        if not graph_ready:
            return False, graph_reason
        return self._window_backend_admitted(callbacks, requested_window_steps)

    def _forward_step_admitted(
        self,
        callbacks: PersistentDecodeWindowCallbacks,
    ) -> tuple[bool, str]:
        model_body_ready, model_body_reason = (
            self._model_body_backend_not_configured(callbacks))
        if not model_body_ready:
            return False, model_body_reason
        return self._model_graph_backend_ready(callbacks)

    def _window_backend_admitted(
        self,
        callbacks: PersistentDecodeWindowCallbacks,
        requested_window_steps: int,
    ) -> tuple[bool, str]:
        if requested_window_steps <= 1:
            return True, "ready"
        return self._window_loop_backend_ready(callbacks)

    def _model_body_backend_not_configured(
        self,
        callbacks: PersistentDecodeWindowCallbacks,
    ) -> tuple[bool, str]:
        if callbacks.model_body_backend_state is None:
            return False, "resident_graph_missing_model_body_backend_state"
        state = callbacks.model_body_backend_state()
        backend = state.get("backend")
        reason = str(state.get("reason", "unknown"))
        configured = bool(state.get("configured", False))
        attempted = bool(state.get("attempted", False))
        executed = bool(state.get("executed", False))
        if configured or attempted or executed or backend is not None:
            return (
                False,
                f"resident_graph_model_body_backend_active:"
                f"{backend}:{reason}",
            )
        return True, "ready"

    def _model_graph_backend_ready(
        self,
        callbacks: PersistentDecodeWindowCallbacks,
    ) -> tuple[bool, str]:
        if callbacks.model_graph_backend_state is None:
            return False, "resident_graph_missing_model_graph_backend_state"
        state = callbacks.model_graph_backend_state()
        backend = state.get("backend")
        reason = str(state.get("reason", "unknown"))
        ready = bool(state.get("ready", False))
        if not ready:
            return (
                False,
                f"resident_graph_model_backend_not_ready:{backend}:{reason}",
            )
        if (backend == _MODEL_ENGINE_CUDA_GRAPH_BACKEND_NAME
                and reason in _MODEL_ENGINE_CUDA_GRAPH_READY_REASONS):
            return True, "ready"
        return (
            False,
            f"resident_graph_model_backend_not_cuda_graph:"
            f"{backend}:{reason}",
        )

    def _window_loop_backend_ready(
        self,
        callbacks: PersistentDecodeWindowCallbacks,
    ) -> tuple[bool, str]:
        if callbacks.window_backend_state is None:
            return False, "resident_graph_missing_window_backend_state"
        state = callbacks.window_backend_state()
        backend = state.get("backend")
        reason = str(state.get("reason", "unknown"))
        ready = bool(state.get("ready", False))
        if not ready:
            return (
                False,
                f"resident_graph_window_backend_not_ready:"
                f"{backend}:{reason}",
            )
        if (backend == _PYEXECUTOR_WINDOW_LOOP_BACKEND_NAME
                and reason in _PYEXECUTOR_WINDOW_LOOP_READY_REASONS):
            return True, "ready"
        return (
            False,
            f"resident_graph_window_backend_not_loop:{backend}:{reason}",
        )


class DeepSeekNativeResidentDecodeBackend(PythonResidentDecodeBackend):
    """Strict executor window backend for the native DeepSeek resident body."""

    name = _DEEPSEEK_NATIVE_RESIDENT_BACKEND_NAME

    def _resident_backend_admitted(
        self,
        callbacks: PersistentDecodeWindowCallbacks,
        requested_window_steps: int,
    ) -> tuple[bool, str]:
        if requested_window_steps > 1:
            if callbacks.resident_window_step is not None:
                return True, "ready"
            return False, "resident_missing_native_window_step"
        model_body_ready, reason = self._native_model_body_ready(callbacks)
        if not model_body_ready:
            return False, reason
        return self._native_sample_backend_ready(callbacks)

    def _forward_step_admitted(
        self,
        callbacks: PersistentDecodeWindowCallbacks,
    ) -> tuple[bool, str]:
        return self._native_model_body_ready(callbacks)

    def _host_update_mode_admitted(
        self,
        defer_host_updates: bool,
    ) -> tuple[bool, str]:
        if defer_host_updates:
            return True, "ready"
        return False, "resident_native_sampling_requires_deferred_host_updates"

    def _window_backend_admitted(
        self,
        callbacks: PersistentDecodeWindowCallbacks,
        requested_window_steps: int,
    ) -> tuple[bool, str]:
        if requested_window_steps <= 1:
            return True, "ready"
        if callbacks.resident_window_step is not None:
            return True, "ready"
        return False, "resident_missing_native_window_step"

    def _resident_window_result(
        self,
        *,
        scheduled_batch: Any,
        sample_state: Any,
        requested_window_steps: int,
        configured_window_steps: int,
        flex_steps: bool,
        callbacks: PersistentDecodeWindowCallbacks,
    ) -> PersistentDecodeWindowResult | None:
        if callbacks.resident_window_step is None:
            return PersistentDecodeWindowResult(
                scheduled_batch=scheduled_batch,
                sample_state=sample_state,
                owned_steps=1,
                requested_window_steps=requested_window_steps,
                break_reason="resident_missing_native_window_step",
                executed=False,
            )
        stage_start = callbacks.stage_start()
        result = callbacks.resident_window_step(
            scheduled_batch,
            sample_state,
            requested_window_steps,
        )
        callbacks.stage_end("resident_window_step", stage_start)
        if result is not None:
            return result
        return PersistentDecodeWindowResult(
            scheduled_batch=scheduled_batch,
            sample_state=sample_state,
            owned_steps=1,
            requested_window_steps=requested_window_steps,
            break_reason="resident_native_window_step_failed",
            executed=False,
        )

    def _native_window_backend_ready(
        self,
        callbacks: PersistentDecodeWindowCallbacks,
    ) -> tuple[bool, str]:
        if callbacks.window_backend_state is None:
            return False, "resident_missing_window_backend_state"
        state = callbacks.window_backend_state()
        backend = state.get("backend")
        reason = str(state.get("reason", "unknown"))
        ready = bool(state.get("ready", False))
        if not ready:
            return (
                False,
                f"resident_window_backend_not_ready:{backend}:{reason}",
            )
        if (backend == _DEEPSEEK_NATIVE_WINDOW_BACKEND_NAME
                and reason in _NATIVE_WINDOW_READY_REASONS):
            return True, "ready"
        return (
            False,
            f"resident_window_backend_not_native:{backend}:{reason}",
        )

    def _sample_step_admitted(
        self,
        callbacks: PersistentDecodeWindowCallbacks,
        sample_result: PersistentDecodeSampleStepResult,
    ) -> tuple[bool, str]:
        if not sample_result.executed:
            return (
                False,
                f"resident_sampling_backend_not_executed:"
                f"{sample_result.reason}",
            )
        return self._sample_backend_identity_ready(
            backend=sample_result.backend,
            reason=sample_result.reason,
            ready=True,
        )

    def _native_model_body_ready(
        self,
        callbacks: PersistentDecodeWindowCallbacks,
    ) -> tuple[bool, str]:
        if callbacks.model_body_backend_state is None:
            return False, "resident_missing_model_body_backend_state"
        state = callbacks.model_body_backend_state()
        backend = state.get("backend")
        reason = str(state.get("reason", "unknown"))
        if backend != _DEEPSEEK_NATIVE_MODEL_BACKEND_NAME:
            return (
                False,
                f"resident_model_body_backend_not_native:{backend}:{reason}",
            )
        if not bool(state.get("executed", False)):
            return (
                False,
                f"resident_model_body_backend_not_executed:{reason}",
            )
        native_reason = _native_execution_reason(state)
        if native_reason not in _NATIVE_MODEL_BODY_READY_REASONS:
            return (
                False,
                f"resident_model_body_incomplete:{native_reason}",
            )
        return True, "ready"

    def _native_sample_backend_ready(
        self,
        callbacks: PersistentDecodeWindowCallbacks,
    ) -> tuple[bool, str]:
        if callbacks.sample_backend_state is None:
            return False, "resident_missing_sampling_backend_state"
        state = callbacks.sample_backend_state()
        backend = state.get("backend")
        reason = str(state.get("reason", "unknown"))
        ready = bool(state.get("ready", False))
        return self._sample_backend_identity_ready(
            backend=backend,
            reason=reason,
            ready=ready,
        )

    def _sample_backend_identity_ready(
        self,
        *,
        backend: Any,
        reason: str,
        ready: bool,
    ) -> tuple[bool, str]:
        if not ready:
            return (
                False,
                f"resident_sampling_backend_not_ready:{backend}:{reason}",
            )
        if (backend == _DEEPSEEK_NATIVE_SAMPLE_BACKEND_NAME
                and reason in _NATIVE_SAMPLE_READY_REASONS):
            return True, "ready"
        if (backend == _PYEXECUTOR_SAMPLE_BRIDGE_BACKEND_NAME
                and _sampling_bridge_allowed()
                and reason in _PYEXECUTOR_SAMPLE_BRIDGE_READY_REASONS):
            return True, "resident_sampling_bridge_allowed"
        return (
            False,
            f"resident_sampling_backend_not_native:{backend}:{reason}",
        )


def create_persistent_decode_backend() -> PersistentDecodeBackend:
    backend_name = os.environ.get(_BACKEND_ENV_NAME,
                                  _PYTHON_RESIDENT_BACKEND_NAME)
    if backend_name == _PYTHON_RESIDENT_BACKEND_NAME:
        return PythonResidentDecodeBackend()
    if backend_name == _DEEPSEEK_GRAPH_RESIDENT_BACKEND_NAME:
        return DeepSeekGraphResidentDecodeBackend()
    if backend_name == _DEEPSEEK_NATIVE_RESIDENT_BACKEND_NAME:
        return DeepSeekNativeResidentDecodeBackend()
    raise ValueError(
        f"Unsupported {_BACKEND_ENV_NAME}={backend_name!r}; supported "
        f"backends: {_PYTHON_RESIDENT_BACKEND_NAME!r}, "
        f"{_DEEPSEEK_GRAPH_RESIDENT_BACKEND_NAME!r}, "
        f"{_DEEPSEEK_NATIVE_RESIDENT_BACKEND_NAME!r}")


def _native_execution_reason(state: dict[str, Any]) -> str:
    metadata = state.get("metadata")
    if isinstance(metadata, dict):
        native_execution_state = metadata.get("native_execution_state")
        if isinstance(native_execution_state, dict):
            reason = native_execution_state.get("reason")
            if reason is not None:
                return str(reason)
    return str(state.get("reason", "unknown"))


def _callback_state_backend(
    callback: Callable[[], dict[str, Any]] | None,
) -> str | None:
    if callback is None:
        return None
    state = callback()
    backend = state.get("backend")
    return None if backend is None else str(backend)


def _sampling_bridge_allowed() -> bool:
    return os.environ.get(_ALLOW_SAMPLING_BRIDGE_ENV_NAME, "0") == "1"


def persistent_decode_engine_enabled() -> bool:
    return os.environ.get(_ENABLE_ENV_NAME, "0") == "1"


def persistent_decode_engine_debug_enabled() -> bool:
    return os.environ.get(_DEBUG_ENV_NAME, "0") == "1"


def persistent_decode_engine_trace_enabled() -> bool:
    return os.environ.get(_TRACE_ENV_NAME, "0") == "1"


def persistent_decode_engine_adp_window_enabled() -> bool:
    return os.environ.get(_ADP_WINDOW_ENV_NAME, "0") == "1"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _rank_enabled(rank: int | None) -> bool:
    raw = os.environ.get(_RANKS_ENV_NAME, "0").strip()
    if raw in ("*", "all", "ALL"):
        return True
    enabled = {item.strip() for item in raw.split(",") if item.strip()}
    return str(rank) in enabled


def _request_ids(requests: list[Any]) -> tuple[int, ...]:
    return tuple(int(getattr(request, "py_request_id", -1)) for request in requests)


def _cached_tokens(requests: list[Any]) -> tuple[int, ...]:
    return tuple(int(getattr(request, "cached_tokens", 0)) for request in requests)


def _has_draft_tokens(requests: list[Any]) -> bool:
    for request in requests:
        draft_tokens = getattr(request, "py_draft_tokens", None)
        if draft_tokens is None:
            draft_tokens = getattr(request, "draft_tokens", None)
        if draft_tokens is None:
            continue
        try:
            if len(draft_tokens) > 0:
                return True
        except TypeError:
            return True
    return False


def _tensor_shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return ()
    try:
        return tuple(int(dim) for dim in shape)
    except (TypeError, ValueError):
        return ()


def _sample_state_new_tokens_shape(sample_state: Any,
                                   tensor_group: str) -> tuple[int, ...]:
    tensors = getattr(sample_state, tensor_group, None)
    return _tensor_shape(getattr(tensors, "new_tokens", None))


def _count_truthy_attr(requests: list[Any], attr: str) -> int:
    return sum(1 for request in requests if bool(getattr(request, attr, False)))


def _max_beam_width(requests: list[Any]) -> int:
    if not requests:
        return 0
    return max(int(getattr(request, "py_beam_width", 1)) for request in requests)


def _request_state_name(request: Any) -> str:
    state = getattr(request, "state", None)
    name = getattr(state, "name", None)
    if name is not None:
        return str(name)
    return str(state)


def _state_hist(requests: list[Any]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(Counter(_request_state_name(request)
                                for request in requests).items()))


def _count_generation_to_complete(requests: list[Any]) -> int:
    return sum(
        1 for request in requests
        if _request_state_name(request).endswith("GENERATION_TO_COMPLETE"))


def _is_dummy_request(request: Any) -> bool:
    return bool(
        getattr(request, "is_attention_dp_dummy", False)
        or getattr(request, "is_cuda_graph_dummy", False)
        or getattr(request, "is_dummy_request", False))


def _real_requests(requests: list[Any]) -> list[Any]:
    return [request for request in requests if not _is_dummy_request(request)]


def _count_dummy_requests(requests: list[Any]) -> int:
    return sum(1 for request in requests if _is_dummy_request(request))


def _min_int_attr(requests: list[Any], attr: str, default: int = 0) -> int:
    values: list[int] = []
    for request in requests:
        value = getattr(request, attr, None)
        if isinstance(value, int):
            values.append(value)
    return min(values) if values else default


def _remaining_decode_steps(request: Any) -> int | None:
    max_new_tokens = getattr(request, "py_max_new_tokens", None)
    decoding_iter = getattr(request, "py_decoding_iter", None)
    if not isinstance(max_new_tokens, int) or not isinstance(decoding_iter, int):
        return None
    return max(0, max_new_tokens - decoding_iter)


def _min_remaining_decode_steps(requests: list[Any]) -> int:
    values = [
        remaining for request in requests
        if (remaining := _remaining_decode_steps(request)) is not None
    ]
    return min(values) if values else 0


class PersistentDecodeEngine:
    """Disabled-by-default executor handoff for TileRT-style decode."""

    def __init__(self, dist: Any) -> None:
        self._rank = getattr(dist, "rank", None)
        self._tp_rank = getattr(dist, "tp_rank", None)
        self._debug_enabled = persistent_decode_engine_debug_enabled() and _rank_enabled(
            self._rank)
        self._trace_enabled = persistent_decode_engine_trace_enabled() and _rank_enabled(
            self._rank)
        self._takeover_enabled = persistent_decode_engine_enabled()
        self._adp_window_enabled = persistent_decode_engine_adp_window_enabled(
        )
        self._report_every = _env_int(_REPORT_EVERY_ENV_NAME,
                                      _DEFAULT_REPORT_EVERY)
        self._requested_window_steps = max(
            1, _env_int(_WINDOW_STEPS_ENV_NAME, _DEFAULT_WINDOW_STEPS))
        self._plans_seen = 0
        self._eligible_seen = 0
        self._contract_ready_seen = 0
        self._window_contract_ready_seen = 0
        self._takeovers_seen = 0
        self._takeover_fallbacks_seen = 0
        self._reason_hist: Counter[str] = Counter()
        self._contract_reason_hist: Counter[str] = Counter()
        self._window_contract_reason_hist: Counter[str] = Counter()
        self._batch_hist: Counter[int] = Counter()
        self._stable_window_steps = 0
        self._last_request_ids: tuple[int, ...] = ()
        self._last_cached_tokens: tuple[int, ...] = ()
        self._max_stable_window_steps = 0
        self._last_plan: PersistentDecodeEnginePlan | None = None

    @property
    def takeover_enabled(self) -> bool:
        return self._takeover_enabled

    @property
    def requested_window_steps(self) -> int:
        return self._requested_window_steps

    @property
    def adp_window_enabled(self) -> bool:
        return self._adp_window_enabled

    @property
    def last_plan(self) -> PersistentDecodeEnginePlan | None:
        return self._last_plan

    def try_execute(
        self,
        *,
        scheduled_batch: Any,
        previous_sample_state: Any,
        current_sample_state_device: Any,
        has_draft_batch: bool,
        use_previous_draft_tokens: bool,
        guided_decoder_present: bool,
        enable_spec_decode: bool,
        attention_dp_enabled: bool = False,
        execute_step: ExecutePersistentDecodeStep | None = None,
    ) -> PersistentDecodeEngineResult | None:
        """Inspect the executor-level handoff and optionally execute it.

        The first serving-side takeover still delegates the actual forward and
        sample work to the executor callback. That is intentionally weaker than
        the target TileRT-style resident kernel loop, but it moves the control
        boundary to the object that will replace the split PyExecutor path.
        """

        if not self._debug_enabled and not self._takeover_enabled:
            return None

        plan = self._build_plan(
            scheduled_batch=scheduled_batch,
            previous_sample_state=previous_sample_state,
            current_sample_state_device=current_sample_state_device,
            has_draft_batch=has_draft_batch,
            use_previous_draft_tokens=use_previous_draft_tokens,
            guided_decoder_present=guided_decoder_present,
            enable_spec_decode=enable_spec_decode,
            attention_dp_enabled=attention_dp_enabled,
        )
        self._record_plan(plan)
        self._last_plan = plan

        if not self._takeover_enabled:
            self._maybe_report()
            return None
        if not plan.initial_takeover_contract_ready:
            self._takeover_fallbacks_seen += 1
            self._maybe_report()
            return None
        if execute_step is None:
            self._takeover_fallbacks_seen += 1
            logger.warning(
                "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE is enabled and the "
                "takeover contract is ready, but no executor callback was "
                "provided; falling back to the standard PyExecutor decode step."
            )
            self._maybe_report()
            return None

        self._takeovers_seen += 1
        if self._trace_enabled:
            logger.info(
                "OPTRT_PERSISTENT_DECODE_ENGINE_TAKEOVER "
                f"{asdict(plan)}")
        self._maybe_report()
        return execute_step()

    def _build_plan(
        self,
        *,
        scheduled_batch: Any,
        previous_sample_state: Any,
        current_sample_state_device: Any,
        has_draft_batch: bool,
        use_previous_draft_tokens: bool,
        guided_decoder_present: bool,
        enable_spec_decode: bool,
        attention_dp_enabled: bool = False,
    ) -> PersistentDecodeEnginePlan:
        generation_requests = list(getattr(scheduled_batch, "generation_requests", ()))
        context_requests = list(getattr(scheduled_batch, "context_requests", ()))
        real_generation_requests = _real_requests(generation_requests)
        request_ids = _request_ids(generation_requests)
        cached_tokens = _cached_tokens(generation_requests)
        previous_device_new_tokens_shape = _sample_state_new_tokens_shape(
            previous_sample_state, "device")
        previous_host_new_tokens_shape = _sample_state_new_tokens_shape(
            previous_sample_state, "host")
        return_log_probs_requests = _count_truthy_attr(generation_requests,
                                                       "py_return_log_probs")
        max_beam_width = _max_beam_width(generation_requests)
        generation_to_complete_requests = _count_generation_to_complete(
            real_generation_requests)
        attention_dp_dummy_requests = _count_dummy_requests(generation_requests)
        request_state_hist = _state_hist(generation_requests)
        min_decoding_iter = _min_int_attr(real_generation_requests,
                                          "py_decoding_iter")
        min_remaining_decode_steps = _min_remaining_decode_steps(
            real_generation_requests)

        if request_ids and request_ids == self._last_request_ids:
            stable_window_steps = self._stable_window_steps + 1
        else:
            stable_window_steps = 1 if request_ids else 0

        reason = "eligible_executor_decode_window"
        eligible = True
        if context_requests:
            eligible = False
            reason = "context_requests_present"
        elif not generation_requests:
            eligible = False
            reason = "no_generation_requests"
        elif previous_sample_state is None:
            eligible = False
            reason = "missing_previous_sample_state"
        elif current_sample_state_device is None:
            eligible = False
            reason = "missing_device_token_feedback"
        elif has_draft_batch or use_previous_draft_tokens or _has_draft_tokens(
                generation_requests):
            eligible = False
            reason = "draft_tokens_present"
        elif guided_decoder_present:
            eligible = False
            reason = "guided_decoder_present"
        elif enable_spec_decode:
            eligible = False
            reason = "spec_decode_enabled"

        initial_takeover_contract_ready = eligible
        initial_takeover_contract_reason = "ready"
        if not eligible:
            initial_takeover_contract_ready = False
            initial_takeover_contract_reason = reason
        elif max_beam_width != 1:
            initial_takeover_contract_ready = False
            initial_takeover_contract_reason = "beam_width_gt_one"
        elif return_log_probs_requests:
            initial_takeover_contract_ready = False
            initial_takeover_contract_reason = "return_log_probs_present"
        elif not previous_device_new_tokens_shape:
            initial_takeover_contract_ready = False
            initial_takeover_contract_reason = "missing_previous_device_new_tokens"
        elif generation_to_complete_requests:
            initial_takeover_contract_ready = False
            initial_takeover_contract_reason = "generation_to_complete_request"

        window_contract_ready = initial_takeover_contract_ready
        window_contract_reason = initial_takeover_contract_reason
        if window_contract_ready and self._requested_window_steps <= 1:
            window_contract_reason = "single_step_window"
        elif (window_contract_ready and attention_dp_enabled
              and not self._adp_window_enabled):
            window_contract_ready = False
            window_contract_reason = "attention_dp_enabled"
        elif (window_contract_ready and attention_dp_dummy_requests
              and not self._adp_window_enabled):
            window_contract_ready = False
            window_contract_reason = "attention_dp_dummy_request"
        elif (window_contract_ready and real_generation_requests
              and min_decoding_iter < 2):
            window_contract_ready = False
            window_contract_reason = "before_first_token_boundary"
        elif window_contract_ready and real_generation_requests:
            terminal_window_enabled = _env_bool(
                _RESIDENT_TERMINAL_WINDOW_ENV_NAME)
            if terminal_window_enabled:
                required_remaining_decode_steps = 0
            elif _env_bool(_RESIDENT_FLEX_STEPS_ENV_NAME):
                required_remaining_decode_steps = 2
            else:
                required_remaining_decode_steps = self._requested_window_steps
            if min_remaining_decode_steps <= required_remaining_decode_steps:
                window_contract_ready = False
                window_contract_reason = "insufficient_remaining_decode_steps"
        if (window_contract_ready and real_generation_requests
                and not _env_bool(_RESIDENT_TERMINAL_WINDOW_ENV_NAME)
                and min_remaining_decode_steps <= 2):
            window_contract_ready = False
            window_contract_reason = "insufficient_remaining_decode_steps"
        elif window_contract_ready and not real_generation_requests:
            window_contract_reason = "adp_dummy_lane_ready"

        return PersistentDecodeEnginePlan(
            eligible=eligible,
            reason=reason,
            initial_takeover_contract_ready=initial_takeover_contract_ready,
            initial_takeover_contract_reason=initial_takeover_contract_reason,
            batch_size=len(generation_requests),
            request_ids=request_ids,
            cached_tokens=cached_tokens,
            has_previous_sample_state=previous_sample_state is not None,
            has_device_token_feedback=current_sample_state_device is not None,
            previous_sample_state_type=type(previous_sample_state).__name__,
            previous_device_new_tokens_shape=previous_device_new_tokens_shape,
            previous_host_new_tokens_shape=previous_host_new_tokens_shape,
            sampler_event_present=getattr(previous_sample_state,
                                          "sampler_event", None) is not None,
            streaming_requests=_count_truthy_attr(generation_requests,
                                                  "streaming"),
            return_log_probs_requests=return_log_probs_requests,
            max_beam_width=max_beam_width,
            real_generation_requests=len(real_generation_requests),
            generation_to_complete_requests=generation_to_complete_requests,
            attention_dp_enabled=attention_dp_enabled,
            attention_dp_dummy_requests=attention_dp_dummy_requests,
            request_state_hist=request_state_hist,
            has_draft_batch=has_draft_batch,
            use_previous_draft_tokens=use_previous_draft_tokens,
            guided_decoder_present=guided_decoder_present,
            enable_spec_decode=enable_spec_decode,
            requested_window_steps=self._requested_window_steps,
            stable_window_steps=stable_window_steps,
            min_decoding_iter=min_decoding_iter,
            min_remaining_decode_steps=min_remaining_decode_steps,
            window_contract_ready=window_contract_ready,
            window_contract_reason=window_contract_reason,
            takeover_enabled=self._takeover_enabled,
            adp_window_enabled=self._adp_window_enabled,
        )

    def _record_plan(self, plan: PersistentDecodeEnginePlan) -> None:
        self._plans_seen += 1
        if plan.eligible:
            self._eligible_seen += 1
        if plan.initial_takeover_contract_ready:
            self._contract_ready_seen += 1
        if plan.window_contract_ready:
            self._window_contract_ready_seen += 1
        self._reason_hist[plan.reason] += 1
        self._contract_reason_hist[plan.initial_takeover_contract_reason] += 1
        self._window_contract_reason_hist[plan.window_contract_reason] += 1
        self._batch_hist[plan.batch_size] += 1
        self._stable_window_steps = plan.stable_window_steps
        self._last_request_ids = plan.request_ids
        self._last_cached_tokens = plan.cached_tokens
        self._max_stable_window_steps = max(self._max_stable_window_steps,
                                            plan.stable_window_steps)

        if self._trace_enabled:
            logger.info(f"OPTRT_PERSISTENT_DECODE_ENGINE_PLAN {asdict(plan)}")

    def _maybe_report(self) -> None:
        if not (self._debug_enabled or self._trace_enabled):
            return
        if self._report_every <= 0:
            return
        if self._plans_seen % self._report_every == 0:
            self.report()

    def report(self) -> None:
        if self._plans_seen == 0:
            return
        summary = {
            "rank": self._rank,
            "tp_rank": self._tp_rank,
            "plans": self._plans_seen,
            "eligible": self._eligible_seen,
            "contract_ready": self._contract_ready_seen,
            "window_contract_ready": self._window_contract_ready_seen,
            "takeovers": self._takeovers_seen,
            "takeover_fallbacks": self._takeover_fallbacks_seen,
            "reasons": dict(sorted(self._reason_hist.items())),
            "contract_reasons": dict(
                sorted(self._contract_reason_hist.items())),
            "window_contract_reasons": dict(
                sorted(self._window_contract_reason_hist.items())),
            "batch_hist": dict(sorted(self._batch_hist.items())),
            "requested_window_steps": self._requested_window_steps,
            "adp_window_enabled": self._adp_window_enabled,
            "max_stable_window_steps": self._max_stable_window_steps,
            "active_window_steps": self._stable_window_steps,
            "takeover_enabled": self._takeover_enabled,
        }
        logger.info(f"OPTRT_PERSISTENT_DECODE_ENGINE {summary}")
        self._plans_seen = 0
        self._eligible_seen = 0
        self._contract_ready_seen = 0
        self._window_contract_ready_seen = 0
        self._takeovers_seen = 0
        self._takeover_fallbacks_seen = 0
        self._reason_hist.clear()
        self._contract_reason_hist.clear()
        self._window_contract_reason_hist.clear()
        self._batch_hist.clear()
        self._max_stable_window_steps = self._stable_window_steps

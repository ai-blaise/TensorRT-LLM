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

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.persistent_decode_engine import (
    DeepSeekGraphResidentDecodeBackend,
    DeepSeekNativeResidentDecodeBackend,
    PersistentDecodeEngine,
    PersistentDecodeEngineResult,
    PersistentDecodeSampleStepResult,
    PersistentDecodeTokenEgress,
    PersistentDecodeWindowCallbacks,
    PersistentDecodeWindowResult,
    PythonResidentDecodeBackend,
    create_persistent_decode_backend,
)


class _FakeTensor:

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


def _request(
    request_id: int,
    *,
    beam_width: int = 1,
    return_log_probs: bool = False,
    streaming: bool = False,
    state: str = "GENERATION_IN_PROGRESS",
    decoding_iter: int = 3,
    max_new_tokens: int = 512,
    attention_dp_dummy: bool = False,
) -> SimpleNamespace:

    return SimpleNamespace(
        py_request_id=request_id,
        cached_tokens=2055,
        draft_tokens=None,
        py_beam_width=beam_width,
        py_return_log_probs=return_log_probs,
        streaming=streaming,
        state=state,
        py_decoding_iter=decoding_iter,
        py_max_new_tokens=max_new_tokens,
        is_attention_dp_dummy=attention_dp_dummy,
        is_cuda_graph_dummy=False,
        is_dummy_request=False,
    )


def _scheduled_batch(requests: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(
        generation_requests=requests,
        context_requests=[],
    )


def _sample_state() -> SimpleNamespace:
    return SimpleNamespace(
        device=SimpleNamespace(new_tokens=_FakeTensor((1, 2, 1))),
        host=SimpleNamespace(new_tokens=_FakeTensor((1, 2, 1))),
        sampler_event=object(),
    )


def _plan(
    engine: PersistentDecodeEngine,
    requests: list[SimpleNamespace],
    *,
    previous_sample_state: SimpleNamespace | None = None,
    attention_dp_enabled: bool = False,
):
    previous_sample_state = previous_sample_state or _sample_state()
    return engine._build_plan(
        scheduled_batch=_scheduled_batch(requests),
        previous_sample_state=previous_sample_state,
        current_sample_state_device=previous_sample_state.device,
        has_draft_batch=False,
        use_previous_draft_tokens=False,
        guided_decoder_present=False,
        enable_spec_decode=False,
        attention_dp_enabled=attention_dp_enabled,
    )


def test_initial_takeover_contract_ready_for_simple_decode(monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_DEBUG", "1")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RANKS", "0")
    engine = PersistentDecodeEngine(SimpleNamespace(rank=0, tp_rank=0))

    plan = _plan(engine, [_request(11), _request(12, streaming=True)])

    assert plan.eligible
    assert plan.initial_takeover_contract_ready
    assert plan.initial_takeover_contract_reason == "ready"
    assert plan.previous_device_new_tokens_shape == (1, 2, 1)
    assert plan.sampler_event_present
    assert plan.streaming_requests == 1
    assert plan.real_generation_requests == 2
    assert plan.requested_window_steps == 1
    assert plan.window_contract_ready
    assert plan.window_contract_reason == "single_step_window"


def test_initial_takeover_contract_blocks_logprobs(monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_DEBUG", "1")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RANKS", "0")
    engine = PersistentDecodeEngine(SimpleNamespace(rank=0, tp_rank=0))

    plan = _plan(engine, [_request(11, return_log_probs=True)])

    assert plan.eligible
    assert not plan.initial_takeover_contract_ready
    assert plan.initial_takeover_contract_reason == "return_log_probs_present"
    assert not plan.window_contract_ready
    assert plan.window_contract_reason == "return_log_probs_present"


def test_initial_takeover_contract_stops_before_completion(monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_DEBUG", "1")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RANKS", "0")
    engine = PersistentDecodeEngine(SimpleNamespace(rank=0, tp_rank=0))

    plan = _plan(engine, [_request(11, state="GENERATION_TO_COMPLETE")])

    assert plan.eligible
    assert not plan.initial_takeover_contract_ready
    assert plan.initial_takeover_contract_reason == "generation_to_complete_request"
    assert plan.generation_to_complete_requests == 1
    assert not plan.window_contract_ready
    assert plan.window_contract_reason == "generation_to_complete_request"


def test_window_contract_blocks_attention_dp(monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_DEBUG", "1")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RANKS", "0")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_STEPS",
                       "4")
    engine = PersistentDecodeEngine(SimpleNamespace(rank=0, tp_rank=0))

    plan = _plan(
        engine,
        [_request(11), _request(12)],
        attention_dp_enabled=True,
    )

    assert plan.initial_takeover_contract_ready
    assert not plan.window_contract_ready
    assert plan.window_contract_reason == "attention_dp_enabled"
    assert plan.requested_window_steps == 4


def test_window_contract_blocks_attention_dp_dummy(monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_DEBUG", "1")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RANKS", "0")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_STEPS",
                       "4")
    engine = PersistentDecodeEngine(SimpleNamespace(rank=0, tp_rank=0))

    plan = _plan(engine, [_request(0, attention_dp_dummy=True)])

    assert plan.initial_takeover_contract_ready
    assert not plan.window_contract_ready
    assert plan.window_contract_reason == "attention_dp_dummy_request"
    assert plan.attention_dp_dummy_requests == 1
    assert plan.real_generation_requests == 0


def test_window_contract_allows_adp_dummy_lane_when_enabled(monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_DEBUG", "1")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RANKS", "0")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_STEPS",
                       "4")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_ADP_WINDOW",
                       "1")
    engine = PersistentDecodeEngine(SimpleNamespace(rank=0, tp_rank=0))

    plan = _plan(
        engine,
        [_request(0, attention_dp_dummy=True)],
        attention_dp_enabled=True,
    )

    assert plan.initial_takeover_contract_ready
    assert plan.window_contract_ready
    assert plan.window_contract_reason == "adp_dummy_lane_ready"
    assert plan.attention_dp_dummy_requests == 1
    assert plan.real_generation_requests == 0
    assert plan.adp_window_enabled


def test_window_contract_waits_until_after_first_token(monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_DEBUG", "1")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RANKS", "0")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_STEPS",
                       "4")
    engine = PersistentDecodeEngine(SimpleNamespace(rank=0, tp_rank=0))

    plan = _plan(engine, [_request(11, decoding_iter=1)])

    assert plan.initial_takeover_contract_ready
    assert not plan.window_contract_ready
    assert plan.window_contract_reason == "before_first_token_boundary"


def test_window_contract_requires_remaining_tokens(monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_DEBUG", "1")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RANKS", "0")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_STEPS",
                       "4")
    engine = PersistentDecodeEngine(SimpleNamespace(rank=0, tp_rank=0))

    plan = _plan(engine, [_request(11, decoding_iter=509, max_new_tokens=512)])

    assert plan.initial_takeover_contract_ready
    assert not plan.window_contract_ready
    assert plan.window_contract_reason == "insufficient_remaining_decode_steps"


def test_window_contract_allows_terminal_resident_window(monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_DEBUG", "1")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RANKS", "0")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_STEPS",
                       "128")
    monkeypatch.setenv(
        "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_TERMINAL_WINDOW",
        "1")
    engine = PersistentDecodeEngine(SimpleNamespace(rank=0, tp_rank=0))

    plan = _plan(engine, [_request(11, decoding_iter=30, max_new_tokens=32)])

    assert plan.initial_takeover_contract_ready
    assert plan.window_contract_ready
    assert plan.window_contract_reason == "ready"
    assert plan.min_remaining_decode_steps == 2


def test_window_contract_allows_shorter_flex_window(monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_DEBUG", "1")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RANKS", "0")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_STEPS",
                       "128")
    monkeypatch.setenv(
        "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_FLEX_STEPS", "1")
    engine = PersistentDecodeEngine(SimpleNamespace(rank=0, tp_rank=0))

    plan = _plan(engine, [_request(11, decoding_iter=3, max_new_tokens=32)])

    assert plan.initial_takeover_contract_ready
    assert plan.window_contract_ready
    assert plan.window_contract_reason == "ready"
    assert plan.min_remaining_decode_steps == 29


def test_window_contract_ready_without_attention_dp(monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_DEBUG", "1")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RANKS", "0")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_STEPS",
                       "4")
    engine = PersistentDecodeEngine(SimpleNamespace(rank=0, tp_rank=0))

    plan = _plan(engine, [_request(11, decoding_iter=8, max_new_tokens=512)])

    assert plan.initial_takeover_contract_ready
    assert plan.window_contract_ready
    assert plan.window_contract_reason == "ready"


def test_takeover_executes_callback_when_enabled_and_ready(monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE", "1")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RANKS", "0")
    engine = PersistentDecodeEngine(SimpleNamespace(rank=0, tp_rank=0))
    expected = PersistentDecodeEngineResult(
        batch_outputs={"logits": object()},
        sample_state=_sample_state(),
    )
    calls = 0

    def execute_step() -> PersistentDecodeEngineResult:
        nonlocal calls
        calls += 1
        return expected

    result = engine.try_execute(
        scheduled_batch=_scheduled_batch([_request(11)]),
        previous_sample_state=_sample_state(),
        current_sample_state_device=_sample_state().device,
        has_draft_batch=False,
        use_previous_draft_tokens=False,
        guided_decoder_present=False,
        enable_spec_decode=False,
        attention_dp_enabled=False,
        execute_step=execute_step,
    )

    assert result is expected
    assert calls == 1
    assert engine._takeovers_seen == 1


def test_takeover_does_not_execute_callback_when_contract_blocks(monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE", "1")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RANKS", "0")
    engine = PersistentDecodeEngine(SimpleNamespace(rank=0, tp_rank=0))
    calls = 0

    def execute_step() -> PersistentDecodeEngineResult:
        nonlocal calls
        calls += 1
        return PersistentDecodeEngineResult(
            batch_outputs={"logits": object()},
            sample_state=_sample_state(),
        )

    result = engine.try_execute(
        scheduled_batch=_scheduled_batch([_request(11, return_log_probs=True)]),
        previous_sample_state=_sample_state(),
        current_sample_state_device=_sample_state().device,
        has_draft_batch=False,
        use_previous_draft_tokens=False,
        guided_decoder_present=False,
        enable_spec_decode=False,
        attention_dp_enabled=False,
        execute_step=execute_step,
    )

    assert result is None
    assert calls == 0
    assert engine._takeover_fallbacks_seen == 1


def test_debug_only_mode_does_not_execute_callback(monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_DEBUG", "1")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RANKS", "0")
    engine = PersistentDecodeEngine(SimpleNamespace(rank=0, tp_rank=0))
    calls = 0

    def execute_step() -> PersistentDecodeEngineResult:
        nonlocal calls
        calls += 1
        return PersistentDecodeEngineResult(
            batch_outputs={"logits": object()},
            sample_state=_sample_state(),
        )

    result = engine.try_execute(
        scheduled_batch=_scheduled_batch([_request(11)]),
        previous_sample_state=_sample_state(),
        current_sample_state_device=_sample_state().device,
        has_draft_batch=False,
        use_previous_draft_tokens=False,
        guided_decoder_present=False,
        enable_spec_decode=False,
        attention_dp_enabled=False,
        execute_step=execute_step,
    )

    assert result is None
    assert calls == 0


def test_python_resident_backend_owns_requested_window_steps():
    backend = PythonResidentDecodeBackend()
    counts = {
        "defer_update": 0,
        "forward": 0,
        "sample": 0,
        "update_state": 0,
        "increment": 0,
        "materialize": 0,
        "timing": 0,
        "log": 0,
    }

    def forward_step(batch, new_tensors_device):
        counts["forward"] += 1
        return {"device": new_tensors_device, "batch": batch}

    def sample_async(batch, outputs):
        counts["sample"] += 1
        return SimpleNamespace(device=f"device_{counts['sample']}")

    def materialize(sample_states):
        counts["materialize"] += len(sample_states)

    callbacks = PersistentDecodeWindowCallbacks(
        initial_admitted=lambda batch, steps: (True, "ready"),
        continue_admitted=lambda batch, request_ids, steps: (True, "ready"),
        request_ids=lambda batch: (11, 12),
        can_defer_host_updates=lambda batch: True,
        defer_update_requests=lambda sample_state: counts.__setitem__(
            "defer_update", counts["defer_update"] + 1),
        sample_has_finished_requests=lambda sample_state:
        (False, "not_finished"),
        update_requests=lambda sample_state: None,
        prepare_resources=lambda batch: None,
        forward_step=forward_step,
        sample_async=sample_async,
        update_request_states=lambda batch: counts.__setitem__(
            "update_state", counts["update_state"] + 1),
        increment_iter_counter=lambda: counts.__setitem__(
            "increment", counts["increment"] + 1),
        materialize_deferred_samples=materialize,
        stage_start=lambda: 0,
        stage_end=lambda stage, start: None,
        record_timing=lambda owned, requested, reason: counts.__setitem__(
            "timing", counts["timing"] + 1),
        log_execution=lambda summary: counts.__setitem__(
            "log", counts["log"] + 1),
    )

    result = backend.execute_decode_window(
        scheduled_batch=SimpleNamespace(name="batch"),
        sample_state=SimpleNamespace(device="device_0"),
        requested_window_steps=4,
        configured_window_steps=4,
        flex_steps=False,
        callbacks=callbacks,
    )

    assert result.executed
    assert result.owned_steps == 4
    assert result.break_reason == "resident_completed_requested_window"
    assert result.sample_state.device == "device_3"
    assert counts["defer_update"] == 3
    assert counts["forward"] == 3
    assert counts["sample"] == 3
    assert counts["update_state"] == 3
    assert counts["increment"] == 3
    assert counts["materialize"] == 3
    assert counts["timing"] == 1
    assert counts["log"] == 1


def test_python_resident_backend_can_accumulate_token_egress():
    backend = PythonResidentDecodeBackend()
    counts = {
        "materialize": 0,
        "enqueue": 0,
    }
    enqueued_samples = []

    def sample_async(batch, outputs):
        return SimpleNamespace(device=f"device_{len(enqueued_samples)}")

    callbacks = PersistentDecodeWindowCallbacks(
        initial_admitted=lambda batch, steps: (True, "ready"),
        continue_admitted=lambda batch, request_ids, steps: (True, "ready"),
        request_ids=lambda batch: (11, 12),
        can_defer_host_updates=lambda batch: True,
        defer_update_requests=lambda sample_state: None,
        sample_has_finished_requests=lambda sample_state:
        (False, "not_finished"),
        update_requests=lambda sample_state: None,
        prepare_resources=lambda batch: None,
        forward_step=lambda batch, new_tensors_device: {"logits": object()},
        sample_async=sample_async,
        update_request_states=lambda batch: None,
        increment_iter_counter=lambda: None,
        materialize_deferred_samples=lambda sample_states:
        counts.__setitem__("materialize", len(sample_states)),
        stage_start=lambda: 0,
        stage_end=lambda stage, start: None,
        record_timing=lambda owned, requested, reason: None,
        log_execution=lambda summary: None,
        can_accumulate_token_egress=lambda batch: True,
        enqueue_deferred_samples=lambda sample_states:
        (enqueued_samples.extend(sample_states),
         counts.__setitem__("enqueue", len(sample_states))),
    )

    result = backend.execute_decode_window(
        scheduled_batch=SimpleNamespace(name="batch"),
        sample_state=SimpleNamespace(device="device_0"),
        requested_window_steps=4,
        configured_window_steps=4,
        flex_steps=False,
        callbacks=callbacks,
    )

    assert result.executed
    assert result.owned_steps == 4
    assert counts["materialize"] == 0
    assert counts["enqueue"] == 3
    assert len(enqueued_samples) == 3


def test_python_resident_backend_materializes_when_accumulation_blocks():
    backend = PythonResidentDecodeBackend()
    materialized_samples = []
    enqueued_samples = []

    callbacks = PersistentDecodeWindowCallbacks(
        initial_admitted=lambda batch, steps: (True, "ready"),
        continue_admitted=lambda batch, request_ids, steps: (True, "ready"),
        request_ids=lambda batch: (11, 12),
        can_defer_host_updates=lambda batch: True,
        defer_update_requests=lambda sample_state: None,
        sample_has_finished_requests=lambda sample_state:
        (False, "not_finished"),
        update_requests=lambda sample_state: None,
        prepare_resources=lambda batch: None,
        forward_step=lambda batch, new_tensors_device: {"logits": object()},
        sample_async=lambda batch, outputs: SimpleNamespace(device="next"),
        update_request_states=lambda batch: None,
        increment_iter_counter=lambda: None,
        materialize_deferred_samples=materialized_samples.extend,
        stage_start=lambda: 0,
        stage_end=lambda stage, start: None,
        record_timing=lambda owned, requested, reason: None,
        log_execution=lambda summary: None,
        can_accumulate_token_egress=lambda batch: False,
        enqueue_deferred_samples=enqueued_samples.extend,
    )

    result = backend.execute_decode_window(
        scheduled_batch=SimpleNamespace(name="batch"),
        sample_state=SimpleNamespace(device="device_0"),
        requested_window_steps=4,
        configured_window_steps=4,
        flex_steps=False,
        callbacks=callbacks,
    )

    assert result.executed
    assert len(materialized_samples) == 3
    assert enqueued_samples == []


def test_python_resident_backend_reports_unexecuted_when_admission_blocks():
    backend = PythonResidentDecodeBackend()
    callback_calls = 0

    def unexpected_callback(*args):
        nonlocal callback_calls
        callback_calls += 1

    callbacks = PersistentDecodeWindowCallbacks(
        initial_admitted=lambda batch, steps:
        (False, "target_concurrency_mismatch"),
        continue_admitted=lambda batch, request_ids, steps: (True, "ready"),
        request_ids=lambda batch: (11, ),
        can_defer_host_updates=lambda batch: True,
        defer_update_requests=unexpected_callback,
        sample_has_finished_requests=lambda sample_state:
        (False, "not_finished"),
        update_requests=unexpected_callback,
        prepare_resources=unexpected_callback,
        forward_step=lambda batch, new_tensors_device: {"logits": object()},
        sample_async=lambda batch, outputs: SimpleNamespace(device="next"),
        update_request_states=unexpected_callback,
        increment_iter_counter=lambda: unexpected_callback(),
        materialize_deferred_samples=unexpected_callback,
        stage_start=lambda: 0,
        stage_end=lambda stage, start: None,
        record_timing=lambda owned, requested, reason: None,
        log_execution=lambda summary: None,
    )

    result = backend.execute_decode_window(
        scheduled_batch=SimpleNamespace(name="batch"),
        sample_state=SimpleNamespace(device="device_0"),
        requested_window_steps=16,
        configured_window_steps=16,
        flex_steps=False,
        callbacks=callbacks,
    )

    assert not result.executed
    assert result.owned_steps == 1
    assert result.break_reason == "target_concurrency_mismatch"
    assert callback_calls == 0


def test_deepseek_native_resident_backend_blocks_python_model_body():
    backend = DeepSeekNativeResidentDecodeBackend()
    forward_calls = 0

    def forward_step(batch, new_tensors_device):
        nonlocal forward_calls
        forward_calls += 1
        return {"logits": object()}

    callbacks = PersistentDecodeWindowCallbacks(
        initial_admitted=lambda batch, steps: (True, "ready"),
        continue_admitted=lambda batch, request_ids, steps: (True, "ready"),
        request_ids=lambda batch: (11, 12),
        can_defer_host_updates=lambda batch: True,
        defer_update_requests=lambda sample_state: None,
        sample_has_finished_requests=lambda sample_state:
        (False, "not_finished"),
        update_requests=lambda sample_state: None,
        prepare_resources=lambda batch: None,
        forward_step=forward_step,
        sample_async=lambda batch, outputs: SimpleNamespace(device="next"),
        update_request_states=lambda batch: None,
        increment_iter_counter=lambda: None,
        materialize_deferred_samples=lambda sample_states: None,
        stage_start=lambda: 0,
        stage_end=lambda stage, start: None,
        record_timing=lambda owned, requested, reason: None,
        log_execution=lambda summary: None,
        model_body_backend_state=lambda: {
            "backend": "deepseek_resident_python_v1",
            "executed": True,
            "reason": "resident_model_body_python_v1_executed",
        },
    )

    result = backend.execute_decode_window(
        scheduled_batch=SimpleNamespace(name="batch"),
        sample_state=SimpleNamespace(device="device_0"),
        requested_window_steps=4,
        configured_window_steps=4,
        flex_steps=False,
        callbacks=callbacks,
    )

    assert not result.executed
    assert result.break_reason.startswith(
        "resident_model_body_backend_not_native")
    assert forward_calls == 0


def test_deepseek_native_resident_backend_blocks_missing_sample_backend():
    backend = DeepSeekNativeResidentDecodeBackend()
    forward_calls = 0

    def forward_step(batch, new_tensors_device):
        nonlocal forward_calls
        forward_calls += 1
        return {"logits": object()}

    callbacks = PersistentDecodeWindowCallbacks(
        initial_admitted=lambda batch, steps: (True, "ready"),
        continue_admitted=lambda batch, request_ids, steps: (True, "ready"),
        request_ids=lambda batch: (11, 12),
        can_defer_host_updates=lambda batch: True,
        defer_update_requests=lambda sample_state: None,
        sample_has_finished_requests=lambda sample_state:
        (False, "not_finished"),
        update_requests=lambda sample_state: None,
        prepare_resources=lambda batch: None,
        forward_step=forward_step,
        sample_async=lambda batch, outputs: SimpleNamespace(device="next"),
        update_request_states=lambda batch: None,
        increment_iter_counter=lambda: None,
        materialize_deferred_samples=lambda sample_states: None,
        stage_start=lambda: 0,
        stage_end=lambda stage, start: None,
        record_timing=lambda owned, requested, reason: None,
        log_execution=lambda summary: None,
        model_body_backend_state=lambda: {
            "backend": "deepseek_resident_native_v1",
            "executed": True,
            "reason": "resident_stage_scheduler_completed",
            "metadata": {
                "native_execution_state": {
                    "reason": "resident_stage_scheduler_completed",
                    "stage_scheduler_reason": "resident_stage_scheduler_completed",
                },
            },
        },
    )

    result = backend.execute_decode_window(
        scheduled_batch=SimpleNamespace(name="batch"),
        sample_state=SimpleNamespace(device="device_0"),
        requested_window_steps=3,
        configured_window_steps=3,
        flex_steps=False,
        callbacks=callbacks,
    )

    assert not result.executed
    assert result.break_reason == "resident_missing_sampling_backend_state"
    assert forward_calls == 0


def test_deepseek_native_resident_backend_blocks_pyexecutor_sample_bridge():
    backend = DeepSeekNativeResidentDecodeBackend()
    forward_calls = 0
    sample_calls = 0

    def forward_step(batch, new_tensors_device):
        nonlocal forward_calls
        forward_calls += 1
        return {"logits": object()}

    def resident_sample_step(batch, outputs):
        nonlocal sample_calls
        sample_calls += 1
        return PersistentDecodeSampleStepResult(
            sample_state=SimpleNamespace(device="next"),
            backend="pyexecutor_sampling_bridge_v1",
            reason="resident_sampling_bridge_executed",
        )

    callbacks = PersistentDecodeWindowCallbacks(
        initial_admitted=lambda batch, steps: (True, "ready"),
        continue_admitted=lambda batch, request_ids, steps: (True, "ready"),
        request_ids=lambda batch: (11, 12),
        can_defer_host_updates=lambda batch: True,
        defer_update_requests=lambda sample_state: None,
        sample_has_finished_requests=lambda sample_state:
        (False, "not_finished"),
        update_requests=lambda sample_state: None,
        prepare_resources=lambda batch: None,
        forward_step=forward_step,
        sample_async=lambda batch, outputs: SimpleNamespace(device="generic"),
        update_request_states=lambda batch: None,
        increment_iter_counter=lambda: None,
        materialize_deferred_samples=lambda sample_states: None,
        stage_start=lambda: 0,
        stage_end=lambda stage, start: None,
        record_timing=lambda owned, requested, reason: None,
        log_execution=lambda summary: None,
        resident_sample_step=resident_sample_step,
        model_body_backend_state=lambda: {
            "backend": "deepseek_resident_native_v1",
            "executed": True,
            "reason": "resident_stage_scheduler_completed",
            "metadata": {
                "native_execution_state": {
                    "reason": "resident_stage_scheduler_completed",
                    "stage_scheduler_reason": "resident_stage_scheduler_completed",
                },
            },
        },
        sample_backend_state=lambda: {
            "backend": "pyexecutor_sampling_bridge_v1",
            "ready": True,
            "reason": "resident_sampling_bridge_ready",
        },
    )

    result = backend.execute_decode_window(
        scheduled_batch=SimpleNamespace(name="batch"),
        sample_state=SimpleNamespace(device="device_0"),
        requested_window_steps=3,
        configured_window_steps=3,
        flex_steps=False,
        callbacks=callbacks,
    )

    assert not result.executed
    assert result.break_reason == (
        "resident_sampling_backend_not_native:"
        "pyexecutor_sampling_bridge_v1:resident_sampling_bridge_ready")
    assert forward_calls == 0
    assert sample_calls == 0


def test_deepseek_native_resident_backend_requires_deferred_host_updates():
    backend = DeepSeekNativeResidentDecodeBackend()
    forward_calls = 0

    def forward_step(batch, new_tensors_device):
        nonlocal forward_calls
        forward_calls += 1
        return {"logits": object()}

    callbacks = PersistentDecodeWindowCallbacks(
        initial_admitted=lambda batch, steps: (True, "ready"),
        continue_admitted=lambda batch, request_ids, steps: (True, "ready"),
        request_ids=lambda batch: (11, 12),
        can_defer_host_updates=lambda batch: False,
        defer_update_requests=lambda sample_state: None,
        sample_has_finished_requests=lambda sample_state:
        (False, "not_finished"),
        update_requests=lambda sample_state: None,
        prepare_resources=lambda batch: None,
        forward_step=forward_step,
        sample_async=lambda batch, outputs: SimpleNamespace(device="generic"),
        update_request_states=lambda batch: None,
        increment_iter_counter=lambda: None,
        materialize_deferred_samples=lambda sample_states: None,
        stage_start=lambda: 0,
        stage_end=lambda stage, start: None,
        record_timing=lambda owned, requested, reason: None,
        log_execution=lambda summary: None,
        resident_sample_step=lambda batch, outputs:
        PersistentDecodeSampleStepResult(
            sample_state=SimpleNamespace(device="next"),
            backend="deepseek_resident_sampler_native_v1",
            reason="resident_sampling_native_executed",
        ),
        model_body_backend_state=lambda: {
            "backend": "deepseek_resident_native_v1",
            "executed": True,
            "reason": "resident_stage_scheduler_completed",
            "metadata": {
                "native_execution_state": {
                    "reason": "resident_stage_scheduler_completed",
                    "stage_scheduler_reason": "resident_stage_scheduler_completed",
                },
            },
        },
        sample_backend_state=lambda: {
            "backend": "deepseek_resident_sampler_native_v1",
            "ready": True,
            "reason": "resident_sampling_native_ready",
        },
        window_backend_state=lambda: {
            "backend": "deepseek_resident_window_native_v1",
            "ready": True,
            "reason": "resident_window_stage_scheduler_ready",
        },
    )

    result = backend.execute_decode_window(
        scheduled_batch=SimpleNamespace(name="batch"),
        sample_state=SimpleNamespace(device="device_0"),
        requested_window_steps=3,
        configured_window_steps=3,
        flex_steps=False,
        callbacks=callbacks,
    )

    assert not result.executed
    assert result.break_reason == (
        "resident_native_sampling_requires_deferred_host_updates")
    assert forward_calls == 0


def test_deepseek_native_resident_backend_blocks_missing_window_backend():
    backend = DeepSeekNativeResidentDecodeBackend()
    forward_calls = 0

    def forward_step(batch, new_tensors_device):
        nonlocal forward_calls
        forward_calls += 1
        return {"logits": object()}

    callbacks = PersistentDecodeWindowCallbacks(
        initial_admitted=lambda batch, steps: (True, "ready"),
        continue_admitted=lambda batch, request_ids, steps: (True, "ready"),
        request_ids=lambda batch: (11, 12),
        can_defer_host_updates=lambda batch: True,
        defer_update_requests=lambda sample_state: None,
        sample_has_finished_requests=lambda sample_state:
        (False, "not_finished"),
        update_requests=lambda sample_state: None,
        prepare_resources=lambda batch: None,
        forward_step=forward_step,
        sample_async=lambda batch, outputs: SimpleNamespace(device="generic"),
        update_request_states=lambda batch: None,
        increment_iter_counter=lambda: None,
        materialize_deferred_samples=lambda sample_states: None,
        stage_start=lambda: 0,
        stage_end=lambda stage, start: None,
        record_timing=lambda owned, requested, reason: None,
        log_execution=lambda summary: None,
        resident_sample_step=lambda batch, outputs:
        PersistentDecodeSampleStepResult(
            sample_state=SimpleNamespace(device="next"),
            backend="deepseek_resident_sampler_native_v1",
            reason="resident_sampling_native_executed",
        ),
        model_body_backend_state=lambda: {
            "backend": "deepseek_resident_native_v1",
            "executed": True,
            "reason": "resident_stage_scheduler_completed",
            "metadata": {
                "native_execution_state": {
                    "reason": "resident_stage_scheduler_completed",
                    "stage_scheduler_reason": "resident_stage_scheduler_completed",
                },
            },
        },
        sample_backend_state=lambda: {
            "backend": "deepseek_resident_sampler_native_v1",
            "ready": True,
            "reason": "resident_sampling_native_ready",
        },
    )

    result = backend.execute_decode_window(
        scheduled_batch=SimpleNamespace(name="batch"),
        sample_state=SimpleNamespace(device="device_0"),
        requested_window_steps=3,
        configured_window_steps=3,
        flex_steps=False,
        callbacks=callbacks,
    )

    assert not result.executed
    assert result.break_reason == "resident_missing_window_backend_state"
    assert forward_calls == 0


def test_deepseek_native_resident_backend_blocks_unready_native_window_backend():
    backend = DeepSeekNativeResidentDecodeBackend()
    forward_calls = 0
    native_window_calls = 0

    def forward_step(batch, new_tensors_device):
        nonlocal forward_calls
        forward_calls += 1
        return {"logits": object()}

    def resident_window_step(batch, sample_state, requested_window_steps):
        nonlocal native_window_calls
        native_window_calls += 1
        return PersistentDecodeWindowResult(
            scheduled_batch=batch,
            sample_state=sample_state,
            owned_steps=1,
            requested_window_steps=requested_window_steps,
            break_reason="resident_native_window_step_failed",
            executed=False,
        )

    callbacks = PersistentDecodeWindowCallbacks(
        initial_admitted=lambda batch, steps: (True, "ready"),
        continue_admitted=lambda batch, request_ids, steps: (True, "ready"),
        request_ids=lambda batch: (11, 12),
        can_defer_host_updates=lambda batch: True,
        defer_update_requests=lambda sample_state: None,
        sample_has_finished_requests=lambda sample_state:
        (False, "not_finished"),
        update_requests=lambda sample_state: None,
        prepare_resources=lambda batch: None,
        forward_step=forward_step,
        sample_async=lambda batch, outputs: SimpleNamespace(device="generic"),
        update_request_states=lambda batch: None,
        increment_iter_counter=lambda: None,
        materialize_deferred_samples=lambda sample_states: None,
        stage_start=lambda: 0,
        stage_end=lambda stage, start: None,
        record_timing=lambda owned, requested, reason: None,
        log_execution=lambda summary: None,
        resident_sample_step=lambda batch, outputs:
        PersistentDecodeSampleStepResult(
            sample_state=SimpleNamespace(device="next"),
            backend="deepseek_resident_sampler_native_v1",
            reason="resident_sampling_native_executed",
        ),
        model_body_backend_state=lambda: {
            "backend": "deepseek_resident_native_v1",
            "executed": True,
            "reason": "resident_stage_scheduler_completed",
            "metadata": {
                "native_execution_state": {
                    "reason": "resident_stage_scheduler_completed",
                    "stage_scheduler_reason": "resident_stage_scheduler_completed",
                },
            },
        },
        sample_backend_state=lambda: {
            "backend": "deepseek_resident_sampler_native_v1",
            "ready": True,
            "reason": "resident_sampling_native_ready",
        },
        window_backend_state=lambda: {
            "backend": "deepseek_resident_window_native_v1",
            "ready": False,
            "reason": "resident_window_native_not_implemented",
        },
        resident_window_step=resident_window_step,
    )

    result = backend.execute_decode_window(
        scheduled_batch=SimpleNamespace(name="batch"),
        sample_state=SimpleNamespace(device="device_0"),
        requested_window_steps=3,
        configured_window_steps=3,
        flex_steps=False,
        callbacks=callbacks,
    )

    assert not result.executed
    assert result.break_reason == "resident_native_window_step_failed"
    assert forward_calls == 0
    assert native_window_calls == 1


def test_deepseek_native_resident_backend_propagates_native_window_blocker():
    backend = DeepSeekNativeResidentDecodeBackend()
    forward_calls = 0
    native_window_calls = 0

    def forward_step(batch, new_tensors_device):
        nonlocal forward_calls
        forward_calls += 1
        return {"logits": object()}

    def resident_window_step(batch, sample_state, requested_window_steps):
        nonlocal native_window_calls
        native_window_calls += 1
        return PersistentDecodeWindowResult(
            scheduled_batch=batch,
            sample_state=sample_state,
            owned_steps=1,
            requested_window_steps=requested_window_steps,
            break_reason=(
                "resident_native_window_declined:"
                "resident_window_native_missing_dsa_attention_dispatch"),
            executed=False,
        )

    callbacks = PersistentDecodeWindowCallbacks(
        initial_admitted=lambda batch, steps: (True, "ready"),
        continue_admitted=lambda batch, request_ids, steps: (True, "ready"),
        request_ids=lambda batch: (11, 12),
        can_defer_host_updates=lambda batch: True,
        defer_update_requests=lambda sample_state: None,
        sample_has_finished_requests=lambda sample_state:
        (False, "not_finished"),
        update_requests=lambda sample_state: None,
        prepare_resources=lambda batch: None,
        forward_step=forward_step,
        sample_async=lambda batch, outputs: SimpleNamespace(device="generic"),
        update_request_states=lambda batch: None,
        increment_iter_counter=lambda: None,
        materialize_deferred_samples=lambda sample_states: None,
        stage_start=lambda: 0,
        stage_end=lambda stage, start: None,
        record_timing=lambda owned, requested, reason: None,
        log_execution=lambda summary: None,
        resident_sample_step=lambda batch, outputs:
        PersistentDecodeSampleStepResult(
            sample_state=SimpleNamespace(device="next"),
            backend="deepseek_resident_sampler_native_v1",
            reason="resident_sampling_native_executed",
        ),
        model_body_backend_state=lambda: {
            "backend": "deepseek_resident_native_v1",
            "executed": True,
            "reason": "resident_stage_scheduler_completed",
            "metadata": {
                "native_execution_state": {
                    "reason": "resident_stage_scheduler_completed",
                    "stage_scheduler_reason": "resident_stage_scheduler_completed",
                },
            },
        },
        sample_backend_state=lambda: {
            "backend": "deepseek_resident_sampler_native_v1",
            "ready": True,
            "reason": "resident_sampling_native_ready",
        },
        window_backend_state=lambda: {
            "backend": "deepseek_resident_window_native_v1",
            "ready": False,
            "reason": "resident_window_native_missing_dsa_attention_dispatch",
        },
        resident_window_step=resident_window_step,
    )

    result = backend.execute_decode_window(
        scheduled_batch=SimpleNamespace(name="batch"),
        sample_state=SimpleNamespace(device="device_0"),
        requested_window_steps=3,
        configured_window_steps=3,
        flex_steps=False,
        callbacks=callbacks,
    )

    assert not result.executed
    assert result.break_reason == (
        "resident_native_window_declined:"
        "resident_window_native_missing_dsa_attention_dispatch")
    assert forward_calls == 0
    assert native_window_calls == 1


def test_deepseek_native_resident_backend_blocks_python_window_loop():
    backend = DeepSeekNativeResidentDecodeBackend()

    callbacks = PersistentDecodeWindowCallbacks(
        initial_admitted=lambda batch, steps: (True, "ready"),
        continue_admitted=lambda batch, request_ids, steps: (True, "ready"),
        request_ids=lambda batch: (11, 12),
        can_defer_host_updates=lambda batch: True,
        defer_update_requests=lambda sample_state: None,
        sample_has_finished_requests=lambda sample_state:
        (False, "not_finished"),
        update_requests=lambda sample_state: None,
        prepare_resources=lambda batch: None,
        forward_step=lambda batch, new_tensors_device: {"logits": object()},
        sample_async=lambda batch, outputs: SimpleNamespace(device="generic"),
        update_request_states=lambda batch: None,
        increment_iter_counter=lambda: None,
        materialize_deferred_samples=lambda sample_states: None,
        stage_start=lambda: 0,
        stage_end=lambda stage, start: None,
        record_timing=lambda owned, requested, reason: None,
        log_execution=lambda summary: None,
        resident_sample_step=lambda batch, outputs:
        PersistentDecodeSampleStepResult(
            sample_state=SimpleNamespace(device="next"),
            backend="deepseek_resident_sampler_native_v1",
            reason="resident_sampling_native_executed",
        ),
        model_body_backend_state=lambda: {
            "backend": "deepseek_resident_native_v1",
            "executed": True,
            "reason": "resident_stage_scheduler_completed",
            "metadata": {
                "native_execution_state": {
                    "reason": "resident_stage_scheduler_completed",
                    "stage_scheduler_reason": "resident_stage_scheduler_completed",
                },
            },
        },
        sample_backend_state=lambda: {
            "backend": "deepseek_resident_sampler_native_v1",
            "ready": True,
            "reason": "resident_sampling_native_ready",
        },
        window_backend_state=lambda: {
            "backend": "pyexecutor_window_loop_v1",
            "ready": True,
            "reason": "resident_window_loop_bridge_ready",
        },
    )

    result = backend.execute_decode_window(
        scheduled_batch=SimpleNamespace(name="batch"),
        sample_state=SimpleNamespace(device="device_0"),
        requested_window_steps=3,
        configured_window_steps=3,
        flex_steps=False,
        callbacks=callbacks,
    )

    assert not result.executed
    assert result.break_reason == "resident_missing_native_window_step"


def test_deepseek_native_resident_backend_requires_native_window_step():
    backend = DeepSeekNativeResidentDecodeBackend()
    forward_calls = 0
    sample_calls = 0

    def forward_step(batch, new_tensors_device):
        nonlocal forward_calls
        forward_calls += 1
        return {"logits": object()}

    def resident_sample_step(batch, outputs):
        nonlocal sample_calls
        sample_calls += 1
        return PersistentDecodeSampleStepResult(
            sample_state=SimpleNamespace(device="next"),
            backend="deepseek_resident_sampler_native_v1",
            reason="resident_sampling_native_executed",
        )

    callbacks = PersistentDecodeWindowCallbacks(
        initial_admitted=lambda batch, steps: (True, "ready"),
        continue_admitted=lambda batch, request_ids, steps: (True, "ready"),
        request_ids=lambda batch: (11, 12),
        can_defer_host_updates=lambda batch: True,
        defer_update_requests=lambda sample_state: None,
        sample_has_finished_requests=lambda sample_state:
        (False, "not_finished"),
        update_requests=lambda sample_state: None,
        prepare_resources=lambda batch: None,
        forward_step=forward_step,
        sample_async=lambda batch, outputs: SimpleNamespace(device="generic"),
        update_request_states=lambda batch: None,
        increment_iter_counter=lambda: None,
        materialize_deferred_samples=lambda sample_states: None,
        stage_start=lambda: 0,
        stage_end=lambda stage, start: None,
        record_timing=lambda owned, requested, reason: None,
        log_execution=lambda summary: None,
        resident_sample_step=resident_sample_step,
        model_body_backend_state=lambda: {
            "backend": "deepseek_resident_native_v1",
            "executed": True,
            "reason": "resident_stage_scheduler_completed",
            "metadata": {
                "native_execution_state": {
                    "reason": "resident_stage_scheduler_completed",
                    "stage_scheduler_reason": "resident_stage_scheduler_completed",
                },
            },
        },
        sample_backend_state=lambda: {
            "backend": "deepseek_resident_sampler_native_v1",
            "ready": True,
            "reason": "resident_sampling_native_ready",
        },
        window_backend_state=lambda: {
            "backend": "deepseek_resident_window_native_v1",
            "ready": True,
            "reason": "resident_window_native_ready",
        },
    )

    result = backend.execute_decode_window(
        scheduled_batch=SimpleNamespace(name="batch"),
        sample_state=SimpleNamespace(device="device_0"),
        requested_window_steps=3,
        configured_window_steps=3,
        flex_steps=False,
        callbacks=callbacks,
    )

    assert not result.executed
    assert result.break_reason == "resident_missing_native_window_step"
    assert forward_calls == 0
    assert sample_calls == 0


def test_deepseek_native_resident_backend_rejects_python_window_loop_override(
        monkeypatch):
    monkeypatch.setenv(
        "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_ALLOW_PYTHON_WINDOW_LOOP", "1")
    backend = DeepSeekNativeResidentDecodeBackend()
    counts = {
        "forward": 0,
        "sample": 0,
        "native_window": 0,
    }

    callbacks = PersistentDecodeWindowCallbacks(
        initial_admitted=lambda batch, steps: (True, "ready"),
        continue_admitted=lambda batch, request_ids, steps: (True, "ready"),
        request_ids=lambda batch: (11, 12),
        can_defer_host_updates=lambda batch: True,
        defer_update_requests=lambda sample_state: None,
        sample_has_finished_requests=lambda sample_state:
        (False, "not_finished"),
        update_requests=lambda sample_state: None,
        prepare_resources=lambda batch: None,
        forward_step=lambda batch, new_tensors_device:
        (counts.__setitem__("forward", counts["forward"] + 1) or {
            "logits": object()
        }),
        sample_async=lambda batch, outputs: SimpleNamespace(device="generic"),
        update_request_states=lambda batch: None,
        increment_iter_counter=lambda: None,
        materialize_deferred_samples=lambda sample_states: None,
        stage_start=lambda: 0,
        stage_end=lambda stage, start: None,
        record_timing=lambda owned, requested, reason: None,
        log_execution=lambda summary: None,
        resident_sample_step=lambda batch, outputs:
        (counts.__setitem__("sample", counts["sample"] + 1)
         or PersistentDecodeSampleStepResult(
             sample_state=SimpleNamespace(device="next"),
             backend="deepseek_resident_sampler_native_v1",
             reason="resident_sampling_native_executed",
         )),
        model_body_backend_state=lambda: {
            "backend": "deepseek_resident_native_v1",
            "executed": True,
            "reason": "resident_stage_scheduler_completed",
            "metadata": {
                "native_execution_state": {
                    "reason": "resident_stage_scheduler_completed",
                    "stage_scheduler_reason": "resident_stage_scheduler_completed",
                },
            },
        },
        sample_backend_state=lambda: {
            "backend": "deepseek_resident_sampler_native_v1",
            "ready": True,
            "reason": "resident_sampling_native_ready",
        },
        window_backend_state=lambda: {
            "backend": "pyexecutor_window_loop_v1",
            "ready": True,
            "reason": "resident_window_loop_bridge_ready",
        },
    )

    result = backend.execute_decode_window(
        scheduled_batch=SimpleNamespace(name="batch"),
        sample_state=SimpleNamespace(device="device_0"),
        requested_window_steps=3,
        configured_window_steps=3,
        flex_steps=False,
        callbacks=callbacks,
    )

    assert not result.executed
    assert result.owned_steps == 1
    assert result.break_reason == "resident_missing_native_window_step"
    assert counts == {
        "forward": 0,
        "sample": 0,
        "native_window": 0,
    }


def test_deepseek_native_resident_backend_window_bypasses_incomplete_step_body():
    backend = DeepSeekNativeResidentDecodeBackend()
    counts = {
        "forward": 0,
        "native_window": 0,
    }

    def resident_window_step(batch, sample_state, requested_window_steps):
        counts["native_window"] += 1
        return PersistentDecodeWindowResult(
            scheduled_batch=batch,
            sample_state=SimpleNamespace(device="native_window"),
            owned_steps=requested_window_steps,
            requested_window_steps=requested_window_steps,
            break_reason="resident_window_native_plan_executed",
            executed=True,
        )

    callbacks = PersistentDecodeWindowCallbacks(
        initial_admitted=lambda batch, steps: (True, "ready"),
        continue_admitted=lambda batch, request_ids, steps: (True, "ready"),
        request_ids=lambda batch: (11, 12),
        can_defer_host_updates=lambda batch: True,
        defer_update_requests=lambda sample_state: None,
        sample_has_finished_requests=lambda sample_state:
        (False, "not_finished"),
        update_requests=lambda sample_state: None,
        prepare_resources=lambda batch: None,
        forward_step=lambda batch, new_tensors_device:
        (counts.__setitem__("forward", counts["forward"] + 1) or {
            "logits": object()
        }),
        sample_async=lambda batch, outputs: SimpleNamespace(device="generic"),
        update_request_states=lambda batch: None,
        increment_iter_counter=lambda: None,
        materialize_deferred_samples=lambda sample_states: None,
        stage_start=lambda: 0,
        stage_end=lambda stage, start: None,
        record_timing=lambda owned, requested, reason: None,
        log_execution=lambda summary: None,
        model_body_backend_state=lambda: {
            "backend": "deepseek_resident_native_v1",
            "executed": False,
            "reason": "resident_model_body_incomplete",
            "metadata": {
                "native_execution_state": {
                    "reason": "layer_1_moe_experts_missing",
                },
            },
        },
        window_backend_state=lambda: {
            "backend": "deepseek_resident_window_native_v1",
            "ready": True,
            "reason": "resident_window_native_ready",
        },
        resident_window_step=resident_window_step,
    )

    result = backend.execute_decode_window(
        scheduled_batch=SimpleNamespace(name="batch"),
        sample_state=SimpleNamespace(device="device_0"),
        requested_window_steps=3,
        configured_window_steps=3,
        flex_steps=False,
        callbacks=callbacks,
    )

    assert result.executed
    assert result.owned_steps == 3
    assert result.break_reason == "resident_window_native_plan_executed"
    assert result.sample_state.device == "native_window"
    assert counts == {
        "forward": 0,
        "native_window": 1,
    }


def test_deepseek_native_resident_backend_owns_after_native_body_and_sample():
    backend = DeepSeekNativeResidentDecodeBackend()
    counts = {
        "forward": 0,
        "sample": 0,
        "increment": 0,
        "native_window": 0,
    }

    def forward_step(batch, new_tensors_device):
        counts["forward"] += 1
        return {"device": new_tensors_device, "batch": batch}

    def sample_async(batch, outputs):
        counts["sample"] += 1
        return SimpleNamespace(device=f"device_{counts['sample']}")

    def resident_sample_step(batch, outputs):
        sample_state = sample_async(batch, outputs)
        return PersistentDecodeSampleStepResult(
            sample_state=sample_state,
            backend="deepseek_resident_sampler_native_v1",
            reason="resident_sampling_native_executed",
        )

    def resident_window_step(batch, sample_state, requested_window_steps):
        counts["native_window"] += 1
        return PersistentDecodeWindowResult(
            scheduled_batch=batch,
            sample_state=SimpleNamespace(device="native_window"),
            owned_steps=requested_window_steps,
            requested_window_steps=requested_window_steps,
            break_reason="resident_native_window_executed",
            executed=True,
        )

    callbacks = PersistentDecodeWindowCallbacks(
        initial_admitted=lambda batch, steps: (True, "ready"),
        continue_admitted=lambda batch, request_ids, steps: (True, "ready"),
        request_ids=lambda batch: (11, 12),
        can_defer_host_updates=lambda batch: True,
        defer_update_requests=lambda sample_state: None,
        sample_has_finished_requests=lambda sample_state:
        (False, "not_finished"),
        update_requests=lambda sample_state: None,
        prepare_resources=lambda batch: None,
        forward_step=forward_step,
        sample_async=sample_async,
        update_request_states=lambda batch: None,
        increment_iter_counter=lambda: counts.__setitem__(
            "increment", counts["increment"] + 1),
        materialize_deferred_samples=lambda sample_states: None,
        stage_start=lambda: 0,
        stage_end=lambda stage, start: None,
        record_timing=lambda owned, requested, reason: None,
        log_execution=lambda summary: None,
        resident_sample_step=resident_sample_step,
        model_body_backend_state=lambda: {
            "backend": "deepseek_resident_native_v1",
            "executed": True,
            "reason": "resident_stage_scheduler_completed",
            "metadata": {
                "native_execution_state": {
                    "reason": "resident_stage_scheduler_completed",
                    "stage_scheduler_reason": "resident_stage_scheduler_completed",
                },
            },
        },
        sample_backend_state=lambda: {
            "backend": "deepseek_resident_sampler_native_v1",
            "ready": True,
            "reason": "resident_sampling_native_ready",
        },
        window_backend_state=lambda: {
            "backend": "deepseek_resident_window_native_v1",
            "ready": True,
            "reason": "resident_window_native_ready",
        },
        resident_window_step=resident_window_step,
    )

    result = backend.execute_decode_window(
        scheduled_batch=SimpleNamespace(name="batch"),
        sample_state=SimpleNamespace(device="device_0"),
        requested_window_steps=3,
        configured_window_steps=3,
        flex_steps=False,
        callbacks=callbacks,
    )

    assert result.executed
    assert result.owned_steps == 3
    assert result.break_reason == "resident_native_window_executed"
    assert result.sample_state.device == "native_window"
    assert counts == {
        "forward": 0,
        "sample": 0,
        "increment": 0,
        "native_window": 1,
    }


def test_deepseek_native_resident_backend_allows_sample_bridge_with_override(
        monkeypatch):
    monkeypatch.setenv(
        "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_ALLOW_SAMPLING_BRIDGE", "1")
    backend = DeepSeekNativeResidentDecodeBackend()
    counts = {
        "forward": 0,
        "sample": 0,
        "native_window": 0,
    }

    def forward_step(batch, new_tensors_device):
        counts["forward"] += 1
        return {"device": new_tensors_device, "batch": batch}

    def resident_sample_step(batch, outputs):
        counts["sample"] += 1
        return PersistentDecodeSampleStepResult(
            sample_state=SimpleNamespace(device=f"device_{counts['sample']}"),
            backend="pyexecutor_sampling_bridge_v1",
            reason="resident_sampling_bridge_executed",
            metadata={
                "native": False,
            },
        )

    def resident_window_step(batch, sample_state, requested_window_steps):
        counts["native_window"] += 1
        return PersistentDecodeWindowResult(
            scheduled_batch=batch,
            sample_state=SimpleNamespace(device="native_window"),
            owned_steps=requested_window_steps,
            requested_window_steps=requested_window_steps,
            break_reason="resident_native_window_executed",
            executed=True,
        )

    callbacks = PersistentDecodeWindowCallbacks(
        initial_admitted=lambda batch, steps: (True, "ready"),
        continue_admitted=lambda batch, request_ids, steps: (True, "ready"),
        request_ids=lambda batch: (11, 12),
        can_defer_host_updates=lambda batch: True,
        defer_update_requests=lambda sample_state: None,
        sample_has_finished_requests=lambda sample_state:
        (False, "not_finished"),
        update_requests=lambda sample_state: None,
        prepare_resources=lambda batch: None,
        forward_step=forward_step,
        sample_async=lambda batch, outputs: SimpleNamespace(device="generic"),
        update_request_states=lambda batch: None,
        increment_iter_counter=lambda: None,
        materialize_deferred_samples=lambda sample_states: None,
        stage_start=lambda: 0,
        stage_end=lambda stage, start: None,
        record_timing=lambda owned, requested, reason: None,
        log_execution=lambda summary: None,
        resident_sample_step=resident_sample_step,
        model_body_backend_state=lambda: {
            "backend": "deepseek_resident_native_v1",
            "executed": True,
            "reason": "resident_stage_scheduler_completed",
            "metadata": {
                "native_execution_state": {
                    "reason": "resident_stage_scheduler_completed",
                    "stage_scheduler_reason": "resident_stage_scheduler_completed",
                },
            },
        },
        sample_backend_state=lambda: {
            "backend": "pyexecutor_sampling_bridge_v1",
            "ready": True,
            "reason": "resident_sampling_bridge_ready",
        },
        window_backend_state=lambda: {
            "backend": "deepseek_resident_window_native_v1",
            "ready": True,
            "reason": "resident_window_native_ready",
        },
        resident_window_step=resident_window_step,
    )

    result = backend.execute_decode_window(
        scheduled_batch=SimpleNamespace(name="batch"),
        sample_state=SimpleNamespace(device="device_0"),
        requested_window_steps=3,
        configured_window_steps=3,
        flex_steps=False,
        callbacks=callbacks,
    )

    assert result.executed
    assert result.owned_steps == 3
    assert result.break_reason == "resident_native_window_executed"
    assert counts == {
        "forward": 0,
        "sample": 0,
        "native_window": 1,
    }


def test_deepseek_native_resident_backend_blocks_incomplete_native_body():
    backend = DeepSeekNativeResidentDecodeBackend()
    forward_calls = 0

    def forward_step(batch, new_tensors_device):
        nonlocal forward_calls
        forward_calls += 1
        return {"logits": object()}

    callbacks = PersistentDecodeWindowCallbacks(
        initial_admitted=lambda batch, steps: (True, "ready"),
        continue_admitted=lambda batch, request_ids, steps: (True, "ready"),
        request_ids=lambda batch: (11, 12),
        can_defer_host_updates=lambda batch: True,
        defer_update_requests=lambda sample_state: None,
        sample_has_finished_requests=lambda sample_state:
        (False, "not_finished"),
        update_requests=lambda sample_state: None,
        prepare_resources=lambda batch: None,
        forward_step=forward_step,
        sample_async=lambda batch, outputs: SimpleNamespace(device="next"),
        update_request_states=lambda batch: None,
        increment_iter_counter=lambda: None,
        materialize_deferred_samples=lambda sample_states: None,
        stage_start=lambda: 0,
        stage_end=lambda stage, start: None,
        record_timing=lambda owned, requested, reason: None,
        log_execution=lambda summary: None,
        model_body_backend_state=lambda: {
            "backend": "deepseek_resident_native_v1",
            "executed": True,
            "reason": "resident_native_engine_executed",
            "metadata": {
                "native_execution_state": {
                    "reason": "layer_1_moe_experts_missing",
                    "stage_scheduler_reason": "layer_1_moe_experts_missing",
                },
            },
        },
    )

    result = backend.execute_decode_window(
        scheduled_batch=SimpleNamespace(name="batch"),
        sample_state=SimpleNamespace(device="device_0"),
        requested_window_steps=3,
        configured_window_steps=3,
        flex_steps=False,
        callbacks=callbacks,
    )

    assert not result.executed
    assert result.break_reason == (
        "resident_model_body_incomplete:layer_1_moe_experts_missing")
    assert forward_calls == 0


def _graph_resident_callbacks(
    counts: dict[str, int],
    *,
    model_body_state: dict[str, object] | None = None,
    graph_state: dict[str, object] | None = None,
    window_state: dict[str, object] | None = None,
) -> PersistentDecodeWindowCallbacks:
    model_body_state = model_body_state or {
        "configured": False,
        "attempted": False,
        "executed": False,
        "backend": None,
        "reason": "not_attempted",
    }
    graph_state = graph_state or {
        "backend": "model_engine_cuda_graph_replay_v1",
        "ready": True,
        "reason": "cuda_graph_replay_ready",
    }
    window_state = window_state or {
        "backend": "pyexecutor_window_loop_v1",
        "ready": True,
        "reason": "resident_window_loop_bridge_ready",
    }

    def forward_step(batch, new_tensors_device):
        counts["forward"] += 1
        return {"device": new_tensors_device, "batch": batch}

    def sample_async(batch, outputs):
        counts["sample"] += 1
        return SimpleNamespace(device=f"device_{counts['sample']}")

    return PersistentDecodeWindowCallbacks(
        initial_admitted=lambda batch, steps: (True, "ready"),
        continue_admitted=lambda batch, request_ids, steps: (True, "ready"),
        request_ids=lambda batch: (11, 12),
        can_defer_host_updates=lambda batch: True,
        defer_update_requests=lambda sample_state:
        counts.__setitem__("defer", counts["defer"] + 1),
        sample_has_finished_requests=lambda sample_state:
        (False, "not_finished"),
        update_requests=lambda sample_state: None,
        prepare_resources=lambda batch: None,
        forward_step=forward_step,
        sample_async=sample_async,
        update_request_states=lambda batch: None,
        increment_iter_counter=lambda: counts.__setitem__(
            "increment", counts["increment"] + 1),
        materialize_deferred_samples=lambda sample_states:
        counts.__setitem__("materialize", len(sample_states)),
        stage_start=lambda: 0,
        stage_end=lambda stage, start: None,
        record_timing=lambda owned, requested, reason: None,
        log_execution=lambda summary:
        counts.__setitem__("logged", counts["logged"] + 1),
        model_body_backend_state=lambda: model_body_state,
        model_graph_backend_state=lambda: graph_state,
        window_backend_state=lambda: window_state,
    )


def test_deepseek_graph_resident_backend_owns_cuda_graph_window():
    backend = DeepSeekGraphResidentDecodeBackend()
    counts = {
        "defer": 0,
        "forward": 0,
        "sample": 0,
        "increment": 0,
        "materialize": 0,
        "logged": 0,
    }

    result = backend.execute_decode_window(
        scheduled_batch=SimpleNamespace(name="batch"),
        sample_state=SimpleNamespace(device="device_0"),
        requested_window_steps=3,
        configured_window_steps=3,
        flex_steps=False,
        callbacks=_graph_resident_callbacks(counts),
    )

    assert result.executed
    assert result.owned_steps == 3
    assert result.break_reason == "resident_completed_requested_window"
    assert result.sample_state.device == "device_2"
    assert counts == {
        "defer": 2,
        "forward": 2,
        "sample": 2,
        "increment": 2,
        "materialize": 2,
        "logged": 1,
    }


def test_deepseek_graph_resident_backend_blocks_native_window_backend():
    backend = DeepSeekGraphResidentDecodeBackend()
    counts = {
        "defer": 0,
        "forward": 0,
        "sample": 0,
        "increment": 0,
        "materialize": 0,
        "logged": 0,
    }

    result = backend.execute_decode_window(
        scheduled_batch=SimpleNamespace(name="batch"),
        sample_state=SimpleNamespace(device="device_0"),
        requested_window_steps=3,
        configured_window_steps=3,
        flex_steps=False,
        callbacks=_graph_resident_callbacks(
            counts,
            window_state={
                "backend": "deepseek_resident_window_native_v1",
                "ready": True,
                "reason": "resident_window_native_ready",
            },
        ),
    )

    assert not result.executed
    assert result.break_reason == (
        "resident_graph_window_backend_not_loop:"
        "deepseek_resident_window_native_v1:resident_window_native_ready")
    assert counts["forward"] == 0
    assert counts["sample"] == 0


def test_deepseek_graph_resident_backend_blocks_native_model_body():
    backend = DeepSeekGraphResidentDecodeBackend()
    counts = {
        "defer": 0,
        "forward": 0,
        "sample": 0,
        "increment": 0,
        "materialize": 0,
        "logged": 0,
    }

    result = backend.execute_decode_window(
        scheduled_batch=SimpleNamespace(name="batch"),
        sample_state=SimpleNamespace(device="device_0"),
        requested_window_steps=3,
        configured_window_steps=3,
        flex_steps=False,
        callbacks=_graph_resident_callbacks(
            counts,
            model_body_state={
                "configured": True,
                "attempted": True,
                "executed": True,
                "backend": "deepseek_resident_native_v1",
                "reason": "resident_stage_scheduler_completed",
            },
        ),
    )

    assert not result.executed
    assert result.break_reason == (
        "resident_graph_model_body_backend_active:"
        "deepseek_resident_native_v1:resident_stage_scheduler_completed")
    assert counts["forward"] == 0
    assert counts["sample"] == 0


def test_persistent_decode_backend_factory_loads_native_resident_backend(
        monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_BACKEND",
                       "deepseek_native_resident")

    assert isinstance(create_persistent_decode_backend(),
                      DeepSeekNativeResidentDecodeBackend)


def test_persistent_decode_backend_factory_loads_graph_resident_backend(
        monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_BACKEND",
                       "deepseek_graph_resident")

    assert isinstance(create_persistent_decode_backend(),
                      DeepSeekGraphResidentDecodeBackend)


def test_persistent_decode_backend_factory_rejects_unknown_backend(
        monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_BACKEND",
                       "deepseek_tilert")

    with pytest.raises(ValueError, match="Unsupported"):
        create_persistent_decode_backend()


def test_token_egress_async_captures_before_materialize():
    materialized_samples = []
    materialized_captures = []

    egress = PersistentDecodeTokenEgress(
        materialized_samples.extend,
        capture=lambda sample_state: f"captured_{sample_state}",
        materialize_captures=materialized_captures.extend,
        async_enabled=True,
    )

    egress.enqueue_sample_state("sample_a")
    egress.enqueue_sample_state("sample_b")

    assert egress.pending_count == 2
    assert egress.flush() == 2
    egress.close()

    assert materialized_samples == []
    assert materialized_captures == ["captured_sample_a", "captured_sample_b"]
    assert egress.pending_count == 0

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
from unittest.mock import Mock

import torch

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests


def _request(
    request_id: int,
    *,
    attention_dp_dummy: bool = False,
    max_new_tokens: int = 64,
    decoding_iter: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        py_request_id=request_id,
        py_max_new_tokens=max_new_tokens,
        py_decoding_iter=decoding_iter,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        is_attention_dp_dummy=attention_dp_dummy,
        is_cuda_graph_dummy=False,
        is_dummy_request=False,
        is_finished=False,
        is_disagg_generation_init_state=False,
        is_disagg_generation_transmission_in_progress=False,
        is_disagg_generation_transmission_complete=False,
        py_beam_width=1,
        py_return_log_probs=False,
        py_return_generation_logits=False,
        streaming=False,
        py_stop_words_list=[],
        py_draft_tokens=None,
        py_disaggregated_params=None,
    )


def _executor(*, enable_attention_dp: bool = True, is_kv_manager_v2: bool = False) -> PyExecutor:
    executor = object.__new__(PyExecutor)
    executor.enable_attention_dp = enable_attention_dp
    executor._is_kv_manager_v2 = is_kv_manager_v2
    executor._optrt_resident_terminal_window = False
    executor._optrt_resident_cohort_execute = False
    executor.guided_decoder = None
    executor.model_engine = SimpleNamespace(enable_spec_decode=False)
    executor.use_spec_decode = False
    executor.drafter = None
    executor.kv_connector_manager = None
    executor._optrt_resident_allow_streaming = True
    executor._optrt_resident_disagg_bootstrap_steps = 0
    executor.stream_interval = 50
    executor._optrt_persistent_resident_backend = SimpleNamespace(
        name="deepseek_graph_resident")
    executor._optrt_defer_attention_dp_dummy_cleanup = False
    executor.active_requests = []
    executor.inflight_req_ids = Mock()
    executor.resource_manager = Mock()
    return executor


class _ResidentWindowRequest:

    def __init__(self) -> None:
        self.py_request_id = 11
        self.py_seq_slot = 0
        self.py_orig_prompt_len = 2
        self.py_decoding_iter = 3
        self.py_max_new_tokens = 128
        self.py_num_accepted_draft_tokens = -1
        self.py_rewind_len = -1
        self.state = LlmRequestState.GENERATION_IN_PROGRESS
        self.streaming = True
        self.is_attention_dp_dummy = False
        self.is_cuda_graph_dummy = False
        self.is_dummy_request = False
        self._prompt_tokens = [101, 102]
        self._generated_tokens = [7]
        self._finished_reason = None
        self.set_generated_tokens_calls = 0

    def get_tokens(self, beam: int) -> list[int]:
        assert beam == 0
        return self._prompt_tokens + self._generated_tokens

    def get_num_tokens(self, beam: int) -> int:
        return len(self.get_tokens(beam))

    def set_generated_tokens(self, tokens: list[list[int]]) -> None:
        self.set_generated_tokens_calls += 1
        self._generated_tokens = list(tokens[0])

    def add_new_token(self, token: int, beam: int) -> None:
        raise AssertionError("native window placeholder updates must be bulked")

    def set_finished_reason(self, reason, beam: int) -> None:
        assert beam == 0
        self._finished_reason = reason


def test_resident_prepare_resources_skips_dummy_only_adp_rank() -> None:
    executor = _executor()
    batch = ScheduledRequests()
    batch.generation_requests = [_request(0, attention_dp_dummy=True)]

    executor._optrt_resident_prepare_resources(batch)

    executor.resource_manager.prepare_resources.assert_not_called()


def test_resident_prepare_resources_filters_dummy_adp_lanes() -> None:
    executor = _executor()
    real_request = _request(11)
    dummy_request = _request(0, attention_dp_dummy=True)
    batch = ScheduledRequests()
    batch.generation_requests = [real_request, dummy_request]

    executor._optrt_resident_prepare_resources(batch)

    executor.resource_manager.prepare_resources.assert_called_once()
    filtered_batch = executor.resource_manager.prepare_resources.call_args.args[0]
    assert filtered_batch is not batch
    assert filtered_batch.generation_requests == [real_request]
    assert batch.generation_requests == [real_request, dummy_request]


def test_resident_prepare_resources_keeps_normal_batch() -> None:
    executor = _executor()
    batch = ScheduledRequests()
    batch.generation_requests = [_request(11)]

    executor._optrt_resident_prepare_resources(batch)

    executor.resource_manager.prepare_resources.assert_called_once_with(batch)


def test_resident_prepare_resources_skips_kv_manager_v2() -> None:
    executor = _executor(is_kv_manager_v2=True)
    batch = ScheduledRequests()
    batch.generation_requests = [_request(11)]

    executor._optrt_resident_prepare_resources(batch)

    executor.resource_manager.prepare_resources.assert_not_called()


def test_resident_cohort_collective_allows_empty_adp_rank_for_native_window(
) -> None:
    executor = _executor()
    executor._optrt_persistent_resident_backend = SimpleNamespace(
        name="deepseek_native_resident")
    executor._optrt_resident_target_requests = (1, )
    executor.dist = Mock()
    executor.dist.tp_allreduce.side_effect = [0, 1, 0]

    admitted, any_blocked, total_real, min_has_real, target_match = (
        executor._optrt_resident_cohort_collective_admitted(
            None, 0, require_target_total=True))

    assert admitted
    assert any_blocked == 0
    assert total_real == 1
    assert min_has_real == 0
    assert target_match


def test_resident_cohort_collective_blocks_empty_adp_rank_for_graph_window(
) -> None:
    executor = _executor()
    executor._optrt_resident_target_requests = (1, )
    executor.dist = Mock()
    executor.dist.tp_allreduce.side_effect = [0, 1, 0]

    admitted, _, _, min_has_real, _ = (
        executor._optrt_resident_cohort_collective_admitted(
            None, 0, require_target_total=True))

    assert not admitted
    assert min_has_real == 0


def test_resident_cohort_admission_ignores_unscheduled_disagg_transfer() -> None:
    executor = _executor()
    ready_request = _request(11, decoding_iter=3)
    transfer_request = _request(12, decoding_iter=0)
    transfer_request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    transfer_request.is_disagg_generation_transmission_in_progress = True
    executor.active_requests = [ready_request, transfer_request]

    batch = ScheduledRequests()
    batch.generation_requests = [ready_request]

    reason, local_real_requests, summary = executor._optrt_resident_cohort_local_state(
        batch,
        requested_window_steps=16,
        require_full_window=True,
    )

    assert reason is None
    assert local_real_requests == 1
    assert summary["active_disagg_generation_transfer_requests"] == 1
    assert summary["active_blocking_reason"] == "disagg_generation_transfer_active"


def test_resident_cohort_admission_blocks_scheduled_disagg_transfer() -> None:
    executor = _executor()
    transfer_request = _request(12, decoding_iter=3)
    transfer_request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    transfer_request.is_disagg_generation_transmission_in_progress = True
    executor.active_requests = [transfer_request]

    batch = ScheduledRequests()
    batch.generation_requests = [transfer_request]

    reason, local_real_requests, summary = executor._optrt_resident_cohort_local_state(
        batch,
        requested_window_steps=16,
        require_full_window=True,
    )

    assert reason == "disagg_generation_transfer_active"
    assert local_real_requests == 1
    assert summary["active_disagg_generation_transfer_requests"] == 1


def test_resident_cohort_admission_keeps_non_native_first_token_boundary(
) -> None:
    executor = _executor()
    request = _request(11, decoding_iter=1)
    executor.active_requests = [request]

    batch = ScheduledRequests()
    batch.generation_requests = [request]

    reason, local_real_requests, _ = executor._optrt_resident_cohort_local_state(
        batch,
        requested_window_steps=16,
        require_full_window=True,
    )

    assert reason == "before_first_token_boundary"
    assert local_real_requests == 1


def test_resident_cohort_admission_allows_native_first_token_boundary(
) -> None:
    executor = _executor()
    executor._optrt_persistent_resident_backend = SimpleNamespace(
        name="deepseek_native_resident")
    request = _request(11, decoding_iter=1)
    executor.active_requests = [request]

    batch = ScheduledRequests()
    batch.generation_requests = [request]

    reason, local_real_requests, _ = executor._optrt_resident_cohort_local_state(
        batch,
        requested_window_steps=16,
        require_full_window=True,
    )

    assert reason is None
    assert local_real_requests == 1


def test_resident_cohort_admission_blocks_native_before_first_token() -> None:
    executor = _executor()
    executor._optrt_persistent_resident_backend = SimpleNamespace(
        name="deepseek_native_resident")
    request = _request(11, decoding_iter=0)
    executor.active_requests = [request]

    batch = ScheduledRequests()
    batch.generation_requests = [request]

    reason, local_real_requests, _ = executor._optrt_resident_cohort_local_state(
        batch,
        requested_window_steps=16,
        require_full_window=True,
    )

    assert reason == "before_first_token_boundary"
    assert local_real_requests == 1


def test_update_request_states_defers_attention_dp_dummy_cleanup() -> None:
    executor = _executor()
    dummy_request = _request(0, attention_dp_dummy=True)
    executor.active_requests = [dummy_request]
    executor._optrt_defer_attention_dp_dummy_cleanup = True
    executor._terminate_request = Mock()

    executor._update_request_states_tp(ScheduledRequests())

    assert executor.active_requests == [dummy_request]
    executor._terminate_request.assert_not_called()
    executor.inflight_req_ids.erase.assert_not_called()


def test_cleanup_attention_dp_dummy_requests_frees_deferred_dummy() -> None:
    executor = _executor()
    dummy_request = _request(0, attention_dp_dummy=True)
    executor.active_requests = [dummy_request]
    executor._terminate_request = Mock()

    executor._optrt_cleanup_attention_dp_dummy_requests()

    assert executor.active_requests == []
    assert dummy_request.state == LlmRequestState.GENERATION_COMPLETE
    executor.inflight_req_ids.erase.assert_called_once_with(0)
    executor._terminate_request.assert_called_once_with(dummy_request)


def test_resident_cohort_admission_reports_peer_first_token_boundary() -> None:
    executor = _executor()
    executor.dist = Mock()
    executor._optrt_persistent_window_admission_debug = False
    executor._optrt_resident_target_requests = (16, )
    executor._optrt_resident_cohort_local_state = Mock(
        return_value=(None, 4, {}))
    executor._optrt_resident_cohort_collective_admitted = Mock(
        return_value=(False, 1, 16, 1, True))
    executor.dist.tp_allreduce.return_value = 1

    admitted, reason = executor._optrt_resident_cohort_admitted(
        ScheduledRequests(), 128)

    assert not admitted
    assert reason == "before_first_token_boundary"
    executor.dist.tp_allreduce.assert_called_once()


def test_resident_cohort_admission_reports_peer_terminal_tail() -> None:
    executor = _executor()
    executor.dist = Mock()
    executor._optrt_persistent_window_admission_debug = False
    executor._optrt_resident_target_requests = (16, )
    executor._optrt_resident_cohort_local_state = Mock(
        return_value=(None, 4, {}))
    executor._optrt_resident_cohort_collective_admitted = Mock(
        return_value=(False, 1, 16, 1, True))
    executor.dist.tp_allreduce.side_effect = [0, 1]

    admitted, reason = executor._optrt_resident_cohort_admitted(
        ScheduledRequests(), 128)

    assert not admitted
    assert reason == "insufficient_remaining_decode_steps"


def test_resident_require_native_window_allows_disagg_bootstrap() -> None:
    executor = _executor()
    executor._optrt_resident_cohort_execute = True
    executor._optrt_resident_require_native_window = True
    executor._optrt_resident_flex_steps = True
    executor._optrt_resident_async_token_egress = False
    executor._optrt_resident_effective_window_steps = Mock(return_value=128)
    executor._optrt_persistent_decode_engine = SimpleNamespace(
        last_plan=SimpleNamespace(requested_window_steps=128),
        requested_window_steps=128,
    )
    executor._optrt_persistent_resident_backend = Mock()
    executor._optrt_persistent_resident_backend.execute_decode_window.return_value = (
        SimpleNamespace(
            executed=False,
            break_reason="disagg_generation_transfer_active",
        ))
    executor.model_engine = SimpleNamespace(
        optrt_persistent_decode_model_backend_state=Mock())

    batch = ScheduledRequests()
    sample_state = SimpleNamespace()

    updated_batch, updated_sample_state, owned_steps, admitted = (
        executor._optrt_try_execute_resident_decode_cohort(
            batch,
            sample_state,
        ))

    assert updated_batch is batch
    assert updated_sample_state is sample_state
    assert owned_steps == 1
    assert admitted


def test_resident_require_native_window_allows_terminal_tail() -> None:
    executor = _executor()
    executor._optrt_resident_cohort_execute = True
    executor._optrt_resident_require_native_window = True
    executor._optrt_resident_flex_steps = True
    executor._optrt_resident_async_token_egress = False
    executor._optrt_resident_effective_window_steps = Mock(return_value=128)
    executor._optrt_persistent_decode_engine = SimpleNamespace(
        last_plan=SimpleNamespace(requested_window_steps=128),
        requested_window_steps=128,
    )
    executor._optrt_persistent_resident_backend = Mock()
    executor._optrt_persistent_resident_backend.execute_decode_window.return_value = (
        SimpleNamespace(
            executed=False,
            break_reason="insufficient_remaining_decode_steps",
        ))
    executor.model_engine = SimpleNamespace(
        optrt_persistent_decode_model_backend_state=Mock())

    batch = ScheduledRequests()
    sample_state = SimpleNamespace()

    updated_batch, updated_sample_state, owned_steps, admitted = (
        executor._optrt_try_execute_resident_decode_cohort(
            batch,
            sample_state,
        ))

    assert updated_batch is batch
    assert updated_sample_state is sample_state
    assert owned_steps == 1
    assert admitted


def test_resident_effective_window_steps_ignores_dummy_only_adp_rank_for_min() -> None:
    executor = _executor()
    executor._optrt_resident_flex_steps = True
    executor.dist = Mock()
    sentinel_remaining_decode_steps = 1 << 30

    def allreduce(value, *, op):
        if value == sentinel_remaining_decode_steps:
            return 34
        if value == 0:
            return 1
        raise AssertionError(f"unexpected allreduce value {value}")

    executor.dist.tp_allreduce.side_effect = allreduce
    batch = ScheduledRequests()
    batch.generation_requests = [_request(0, attention_dp_dummy=True)]

    assert executor._optrt_resident_effective_window_steps(batch, 128) == 32

    allreduce_values = [
        call.args[0] for call in executor.dist.tp_allreduce.call_args_list
    ]
    assert allreduce_values == [sentinel_remaining_decode_steps, 0]


def test_resident_effective_window_steps_empty_adp_cohort_stays_single_step() -> None:
    executor = _executor()
    executor._optrt_resident_flex_steps = True
    executor.dist = Mock()
    executor.dist.tp_allreduce.side_effect = [1 << 30, 0]
    batch = ScheduledRequests()
    batch.generation_requests = [_request(0, attention_dp_dummy=True)]

    assert executor._optrt_resident_effective_window_steps(batch, 128) == 1


def test_resident_effective_window_steps_can_own_terminal_tail() -> None:
    executor = _executor(enable_attention_dp=False)
    executor._optrt_resident_flex_steps = True
    executor._optrt_resident_terminal_window = True
    batch = ScheduledRequests()
    batch.generation_requests = [
        _request(11, max_new_tokens=32, decoding_iter=3)
    ]

    assert executor._optrt_resident_effective_window_steps(batch, 128) == 29


def test_resident_one_step_effective_window_does_not_fallback() -> None:
    executor = _executor(enable_attention_dp=False)
    executor._optrt_resident_cohort_execute = True
    executor._optrt_resident_flex_steps = True
    executor._optrt_resident_terminal_window = True
    executor._optrt_persistent_decode_engine = SimpleNamespace(
        last_plan=None,
        requested_window_steps=16,
    )
    batch = ScheduledRequests()
    batch.generation_requests = [
        _request(11, max_new_tokens=4, decoding_iter=3)
    ]
    sample_state = SimpleNamespace()

    next_batch, next_sample_state, owned_steps, resident_admitted = (
        executor._optrt_try_execute_resident_decode_cohort(
            batch,
            sample_state,
        ))

    assert next_batch is batch
    assert next_sample_state is sample_state
    assert owned_steps == 1
    assert resident_admitted


def test_resident_window_placeholders_are_bulk_materialized() -> None:
    executor = _executor(enable_attention_dp=False)
    request = _ResidentWindowRequest()
    placeholder_sample_state = SimpleNamespace(requests=[request])

    executor._optrt_resident_defer_window_update_requests(
        placeholder_sample_state,
        placeholder_steps=2,
    )

    assert request._generated_tokens == [7, 0, 0]
    assert request.py_decoding_iter == 5
    assert request.py_num_accepted_draft_tokens == 0
    assert request.py_rewind_len == 0
    assert request.set_generated_tokens_calls == 1

    window_tokens = torch.tensor([[[41]], [[42]], [[43]]], dtype=torch.int64)
    native_sample_state = SimpleNamespace(
        requests=[request],
        optrt_resident_request_order=True,
        optrt_resident_window_host_new_tokens=window_tokens,
    )

    executor._optrt_resident_update_window_requests(
        native_sample_state,
        [request],
        [0],
        owned_steps=3,
    )

    assert request._generated_tokens == [7, 41, 42, 43]
    assert request.py_decoding_iter == 6


def test_resident_window_bulk_update_finishes_terminal_length() -> None:
    executor = _executor(enable_attention_dp=False)
    request = _ResidentWindowRequest()
    request.py_decoding_iter = 126
    request.py_max_new_tokens = 128
    request._generated_tokens = [7] * 126
    placeholder_sample_state = SimpleNamespace(requests=[request])

    executor._optrt_resident_defer_window_update_requests(
        placeholder_sample_state,
        placeholder_steps=1,
    )

    window_tokens = torch.tensor([[[41]], [[42]]], dtype=torch.int64)
    native_sample_state = SimpleNamespace(
        requests=[request],
        optrt_resident_request_order=True,
        optrt_resident_window_host_new_tokens=window_tokens,
    )

    executor._optrt_resident_update_window_requests(
        native_sample_state,
        [request],
        [0],
        owned_steps=2,
    )

    assert request.py_decoding_iter == 128
    assert request.state == LlmRequestState.GENERATION_COMPLETE
    assert request._generated_tokens[-2:] == [41, 42]


def test_resident_window_seq_slot_expansion_slices_capacity_prefix() -> None:
    executor = _executor(enable_attention_dp=False)
    executor._optrt_resident_native_token_stats = {
        "resident_native_window_seq_slot_expansions": 0
    }
    window_tokens = torch.arange(121 * 2, dtype=torch.int64).reshape(121, 2, 1)
    source_sample_state = SimpleNamespace(
        requests=[
            SimpleNamespace(py_seq_slot=1),
            SimpleNamespace(py_seq_slot=3),
        ],
        device=SimpleNamespace(new_tokens=torch.zeros((1, 4, 1),
                                                      dtype=torch.int64)),
    )

    expanded, request_order = (
        executor._optrt_expand_resident_window_tokens_to_seq_slots(
            source_sample_state=source_sample_state,
            window_tokens=window_tokens,
            owned_steps=119,
        ))

    assert not request_order
    assert expanded.shape == (119, 4, 1)
    assert torch.equal(expanded[:, 1, :], window_tokens[:119, 0, :])
    assert torch.equal(expanded[:, 3, :], window_tokens[:119, 1, :])
    assert torch.equal(expanded[:, 0, :], torch.zeros((119, 1), dtype=torch.int64))
    assert torch.equal(expanded[:, 2, :], torch.zeros((119, 1), dtype=torch.int64))
    assert executor._optrt_resident_native_token_stats[
        "resident_native_window_seq_slot_expansions"] == 1


def test_resident_window_placeholder_marks_crossed_stream_boundary() -> None:
    executor = _executor(enable_attention_dp=False)
    request = _ResidentWindowRequest()
    request.py_decoding_iter = 49
    placeholder_sample_state = SimpleNamespace(requests=[request])

    executor._optrt_resident_defer_window_update_requests(
        placeholder_sample_state,
        placeholder_steps=16,
    )

    assert request.py_decoding_iter == 65
    assert request.optrt_resident_pending_stream_emit


def test_resident_window_update_marks_crossed_stream_boundary() -> None:
    executor = _executor(enable_attention_dp=False)
    request = _ResidentWindowRequest()
    request.py_decoding_iter = 49
    request._generated_tokens = [7] * 49
    native_sample_state = SimpleNamespace(
        requests=[request],
        optrt_resident_request_order=True,
        optrt_resident_window_host_new_tokens=torch.tensor(
            [[[41]]], dtype=torch.int64),
    )

    executor._optrt_resident_update_window_requests(
        native_sample_state,
        [request],
        [0],
        owned_steps=1,
    )

    assert request.py_decoding_iter == 50
    assert request.optrt_resident_pending_stream_emit

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

from tensorrt_llm._torch.pyexecutor.persistent_decode_window import (
    PersistentDecodeWindowTracker,
)


def _dist() -> SimpleNamespace:
    return SimpleNamespace(rank=0, tp_rank=0)


def _plan(
    *,
    request_ids: tuple[int, ...] = (11, 12),
    cached_tokens: tuple[int, ...] = (2055, 2055),
    eligible: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        eligible=eligible,
        reason="eligible_decode_only_deepseek_batch",
        cuda_graph_replay=True,
        cuda_graph_padding=False,
        request_ids=request_ids,
        cached_tokens=cached_tokens,
        real_generation_requests=len(request_ids),
        metadata_tokens=len(request_ids),
    )


def test_disagg_cached_tokens_do_not_break_stable_window(monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_DEBUG", "1")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_RANKS", "0")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_TARGET", "3")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_EVERY", "1000")

    tracker = PersistentDecodeWindowTracker(_dist())

    tracker.record(_plan())
    tracker.record(_plan())
    tracker.record(_plan())

    assert tracker._open is not None
    assert tracker._open.steps == 3
    assert tracker._target_hits == 1
    assert tracker._break_reasons == {}


def test_request_change_breaks_stable_window(monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_DEBUG", "1")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_RANKS", "0")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_EVERY", "1000")

    tracker = PersistentDecodeWindowTracker(_dist())

    tracker.record(_plan(request_ids=(11, 12)))
    tracker.record(_plan(request_ids=(11, 13)))

    assert tracker._open is not None
    assert tracker._open.request_ids == (11, 13)
    assert tracker._open.steps == 1
    assert tracker._closed_windows == 1
    assert tracker._break_reasons == {"request_ids_changed": 1}

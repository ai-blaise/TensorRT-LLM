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

"""Stable-window tracker for a future resident decode engine.

The TileRT-style serving path needs to own several decode steps while request
slots, KV ownership, graph shape, and token feedback remain resident. This
default-off probe records whether the current production scheduler presents
those windows naturally before we replace the per-step layer graph replay.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from typing import Any

from tensorrt_llm.logger import logger

_ENV_NAME = "TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_DEBUG"
_REPORT_EVERY_ENV_NAME = "TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_EVERY"
_TARGET_STEPS_ENV_NAME = "TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_TARGET"
_RANKS_ENV_NAME = "TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_RANKS"

_DEFAULT_REPORT_EVERY = 256
_DEFAULT_TARGET_STEPS = 16


def persistent_decode_window_enabled() -> bool:
    return os.environ.get(_ENV_NAME, "0") == "1"


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


@dataclass
class _OpenWindow:
    request_ids: tuple[int, ...]
    start_cached_tokens: tuple[int, ...]
    last_cached_tokens: tuple[int, ...]
    steps: int = 1
    target_hit_recorded: bool = False

    @property
    def batch_size(self) -> int:
        return len(self.request_ids)

    def extend(self, cached_tokens: tuple[int, ...]) -> None:
        self.last_cached_tokens = cached_tokens
        self.steps += 1

    def summary(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "batch_size": self.batch_size,
            "request_ids": self.request_ids,
            "start_cached_tokens": self.start_cached_tokens,
            "last_cached_tokens": self.last_cached_tokens,
        }


class PersistentDecodeWindowTracker:
    """Tracks stable scalar-decode windows in the current serving loop."""

    def __init__(self, dist: Any) -> None:
        self._rank = getattr(dist, "rank", None)
        self._tp_rank = getattr(dist, "tp_rank", None)
        self._enabled = persistent_decode_window_enabled() and _rank_enabled(
            self._rank)
        self._report_every = max(
            1, _env_int(_REPORT_EVERY_ENV_NAME, _DEFAULT_REPORT_EVERY))
        self._target_steps = max(
            1, _env_int(_TARGET_STEPS_ENV_NAME, _DEFAULT_TARGET_STEPS))
        self._steps_seen = 0
        self._closed_windows = 0
        self._window_hist: Counter[int] = Counter()
        self._break_reasons: Counter[str] = Counter()
        self._target_hits = 0
        self._max_window_steps = 0
        self._open: _OpenWindow | None = None

    def record(self, plan: Any) -> None:
        if not self._enabled:
            return
        self._steps_seen += 1

        reason = self._plan_break_reason(plan)
        if reason is not None:
            self._close(reason)
        else:
            self._record_eligible(plan)

        if self._steps_seen % self._report_every == 0:
            self.report()

    def report(self) -> None:
        if not self._enabled or self._steps_seen == 0:
            return
        active_window = self._open.summary() if self._open is not None else None
        summary = {
            "rank": self._rank,
            "tp_rank": self._tp_rank,
            "steps_seen": self._steps_seen,
            "target_steps": self._target_steps,
            "target_hits": self._target_hits,
            "max_window_steps": max(
                self._max_window_steps,
                self._open.steps if self._open is not None else 0),
            "closed_windows": self._closed_windows,
            "window_len_hist": dict(sorted(self._window_hist.items())),
            "break_reasons": dict(sorted(self._break_reasons.items())),
            "active_window": active_window,
            "active_window_reaches_target": (
                self._open is not None
                and self._open.steps >= self._target_steps),
        }
        logger.info(f"OPTRT_PERSISTENT_DECODE_WINDOW {summary}")
        self._reset_interval()

    def _plan_break_reason(self, plan: Any) -> str | None:
        if plan is None:
            return "missing_plan"
        if not bool(getattr(plan, "eligible", False)):
            return str(getattr(plan, "reason", "ineligible_plan"))
        if not bool(getattr(plan, "cuda_graph_replay", False)):
            return "not_cuda_graph_replay"
        if bool(getattr(plan, "cuda_graph_padding", False)):
            return "cuda_graph_padding"
        request_ids = tuple(getattr(plan, "request_ids", ()))
        cached_tokens = tuple(getattr(plan, "cached_tokens", ()))
        if not request_ids:
            return "missing_request_ids"
        if not cached_tokens:
            return "missing_cached_tokens"
        if len(request_ids) != int(getattr(plan, "real_generation_requests", 0)):
            return "request_id_count_mismatch"
        if len(cached_tokens) != len(request_ids):
            return "cached_token_count_mismatch"
        if int(getattr(plan, "metadata_tokens", 0)) != len(request_ids):
            return "metadata_token_count_mismatch"
        return None

    def _record_eligible(self, plan: Any) -> None:
        request_ids = tuple(getattr(plan, "request_ids", ()))
        cached_tokens = tuple(getattr(plan, "cached_tokens", ()))
        if self._open is None:
            self._open = _OpenWindow(
                request_ids=request_ids,
                start_cached_tokens=cached_tokens,
                last_cached_tokens=cached_tokens,
            )
            self._maybe_count_target_hit()
            return
        if self._open.request_ids != request_ids:
            self._close("request_ids_changed")
            self._open = _OpenWindow(
                request_ids=request_ids,
                start_cached_tokens=cached_tokens,
                last_cached_tokens=cached_tokens,
            )
            self._maybe_count_target_hit()
            return
        self._open.extend(cached_tokens)
        self._maybe_count_target_hit()

    def _maybe_count_target_hit(self) -> None:
        if self._open is None:
            return
        if self._open.target_hit_recorded:
            return
        if self._open.steps < self._target_steps:
            return
        self._target_hits += 1
        self._open.target_hit_recorded = True

    def _close(self, reason: str) -> None:
        self._break_reasons[reason] += 1
        if self._open is None:
            return
        steps = self._open.steps
        self._window_hist[steps] += 1
        self._closed_windows += 1
        self._max_window_steps = max(self._max_window_steps, steps)
        if steps >= self._target_steps and not self._open.target_hit_recorded:
            self._target_hits += 1
        self._open = None

    def _reset_interval(self) -> None:
        open_window = self._open
        self._steps_seen = 0
        self._closed_windows = 0
        self._window_hist.clear()
        self._break_reasons.clear()
        self._target_hits = 0
        self._max_window_steps = open_window.steps if open_window else 0

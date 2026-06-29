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

"""Timing probe for the experimental persistent decode handoff.

The profiler is default-off and only records batches that the persistent decode
planner marked eligible. It is meant to answer where the current production
path spends time before replacing that path with a TileRT-style resident decode
engine.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch

from tensorrt_llm.logger import logger

_PROFILE_ENV_NAME = "TRTLLM_OPTRT_PERSISTENT_DECODE_TIMING_DEBUG"
_REPORT_EVERY_ENV_NAME = "TRTLLM_OPTRT_PERSISTENT_DECODE_TIMING_EVERY"
_CUDA_EVENTS_ENV_NAME = "TRTLLM_OPTRT_PERSISTENT_DECODE_TIMING_CUDA_EVENTS"
_RANKS_ENV_NAME = "TRTLLM_OPTRT_PERSISTENT_DECODE_TIMING_RANKS"

_DEFAULT_REPORT_EVERY = 256
_NS_PER_US = 1_000.0
_NS_PER_S = 1_000_000_000.0

_WALL_STAGES = (
    ("metadata_setup", "forward_enter", "after_metadata_setup"),
    ("cuda_graph_lookup", "after_metadata_setup", "after_cuda_graph_lookup"),
    ("prepare_inputs", "after_cuda_graph_lookup", "after_prepare_inputs"),
    ("plan_build", "after_prepare_inputs", "after_plan"),
    ("execute_enqueue", "before_execute", "after_execute"),
    ("forward_callback", "after_execute", "after_forward_callback"),
    ("logit_postprocess", "after_forward_callback", "after_logit_postprocessors"),
    ("total_forward_path", "forward_enter", "after_logit_postprocessors"),
)


def persistent_decode_profile_enabled() -> bool:
    return os.environ.get(_PROFILE_ENV_NAME, "0") == "1"


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


def _tok_s_from_ns(ns: float) -> float:
    if ns <= 0.0:
        return 0.0
    return _NS_PER_S / ns


@dataclass
class _StageAccum:
    count: int = 0
    total_ns: int = 0
    max_ns: int = 0

    def add(self, ns: int) -> None:
        self.count += 1
        self.total_ns += ns
        self.max_ns = max(self.max_ns, ns)

    def summary(self) -> dict[str, float | int]:
        avg_ns = self.total_ns / self.count if self.count else 0.0
        return {
            "count": self.count,
            "avg_us": avg_ns / _NS_PER_US,
            "max_us": self.max_ns / _NS_PER_US,
            "tok_s_user_est": _tok_s_from_ns(avg_ns),
        }


class PersistentDecodeStepTiming:
    """Per-forward timing state returned by `PersistentDecodeProfiler`."""

    def __init__(self, profiler: "PersistentDecodeProfiler") -> None:
        self._profiler = profiler
        self._marks: dict[str, int] = {}
        self._plan: Any = None
        self._cuda_start: torch.cuda.Event | None = None
        self._cuda_end: torch.cuda.Event | None = None

    @property
    def plan(self) -> Any:
        return self._plan

    @property
    def marks(self) -> dict[str, int]:
        return self._marks

    @property
    def cuda_event_pair(
            self) -> tuple[torch.cuda.Event, torch.cuda.Event] | None:
        if self._cuda_start is None or self._cuda_end is None:
            return None
        return self._cuda_start, self._cuda_end

    def mark(self, name: str) -> None:
        self._marks[name] = time.perf_counter_ns()

    def set_plan(self, plan: Any) -> None:
        self._plan = plan

    def begin_execute_cuda(self) -> None:
        if not self._profiler.cuda_events_enabled:
            return
        self._cuda_start = torch.cuda.Event(enable_timing=True)
        self._cuda_end = torch.cuda.Event(enable_timing=True)
        self._cuda_start.record(torch.cuda.current_stream())

    def end_execute_cuda(self) -> None:
        if self._cuda_end is None:
            return
        self._cuda_end.record(torch.cuda.current_stream())


class PersistentDecodeProfiler:
    """Aggregate profiler for eligible persistent-decode batches."""

    def __init__(self, dist: Any) -> None:
        self._rank = getattr(dist, "rank", None)
        self._tp_rank = getattr(dist, "tp_rank", None)
        self._enabled = persistent_decode_profile_enabled() and _rank_enabled(
            self._rank)
        self._report_every = max(
            1, _env_int(_REPORT_EVERY_ENV_NAME, _DEFAULT_REPORT_EVERY))
        self.cuda_events_enabled = (
            self._enabled
            and os.environ.get(_CUDA_EVENTS_ENV_NAME, "1") == "1"
            and torch.cuda.is_available())
        self._wall_accums: dict[str, _StageAccum] = {}
        self._batch_hist: Counter[int] = Counter()
        self._reason_hist: Counter[str] = Counter()
        self._graph_hist: Counter[bool] = Counter()
        self._padding_hist: Counter[bool] = Counter()
        self._cuda_event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event,
                                           int]] = []
        self._cuda_execute_ms: list[float] = []
        self._steps = 0
        self._tokens = 0

    def start_step(self) -> PersistentDecodeStepTiming | None:
        if not self._enabled:
            return None
        step = PersistentDecodeStepTiming(self)
        step.mark("forward_enter")
        return step

    def record(self, step: PersistentDecodeStepTiming | None) -> None:
        if step is None:
            return
        plan = step.plan
        if plan is None or not bool(getattr(plan, "eligible", False)):
            return

        batch_size = int(getattr(plan, "real_generation_requests", 0))
        self._steps += 1
        self._tokens += batch_size
        self._batch_hist[batch_size] += 1
        self._reason_hist[str(getattr(plan, "reason", ""))] += 1
        self._graph_hist[bool(getattr(plan, "cuda_graph_replay", False))] += 1
        self._padding_hist[bool(getattr(plan, "cuda_graph_padding",
                                        False))] += 1

        marks = step.marks
        for stage, start, end in _WALL_STAGES:
            start_ns = marks.get(start)
            end_ns = marks.get(end)
            if start_ns is None or end_ns is None:
                continue
            self._wall_accums.setdefault(stage,
                                         _StageAccum()).add(end_ns - start_ns)

        pair = step.cuda_event_pair
        if pair is not None:
            self._cuda_event_pairs.append((pair[0], pair[1], batch_size))

        if self._steps % self._report_every == 0:
            self.report()

    def report(self) -> None:
        if self._steps == 0:
            return
        self._drain_cuda_events()
        summary = {
            "rank": self._rank,
            "tp_rank": self._tp_rank,
            "steps": self._steps,
            "tokens": self._tokens,
            "batch_hist": dict(sorted(self._batch_hist.items())),
            "reasons": dict(self._reason_hist),
            "cuda_graph_replay": dict(self._graph_hist),
            "cuda_graph_padding": dict(self._padding_hist),
            "wall": {
                key: value.summary()
                for key, value in sorted(self._wall_accums.items())
            },
            "cuda_execute": self._cuda_summary(),
        }
        logger.info(f"OPTRT_PERSISTENT_DECODE_TIMING {summary}")
        self._reset_interval()

    def _drain_cuda_events(self) -> None:
        if not self._cuda_event_pairs:
            return
        pairs = self._cuda_event_pairs
        self._cuda_event_pairs = []
        try:
            pairs[-1][1].synchronize()
        except RuntimeError as exc:
            logger.warning(
                "OPTRT_PERSISTENT_DECODE_TIMING disabled CUDA event timing: "
                f"{exc}")
            self.cuda_events_enabled = False
            return
        for start, end, _batch_size in pairs:
            try:
                self._cuda_execute_ms.append(float(start.elapsed_time(end)))
            except RuntimeError:
                continue

    def _cuda_summary(self) -> dict[str, float | int]:
        if not self._cuda_execute_ms:
            return {"count": 0}
        total = sum(self._cuda_execute_ms)
        avg_ms = total / len(self._cuda_execute_ms)
        return {
            "count": len(self._cuda_execute_ms),
            "avg_ms": avg_ms,
            "max_ms": max(self._cuda_execute_ms),
            "tok_s_user_est": 1000.0 / avg_ms if avg_ms > 0.0 else 0.0,
        }

    def _reset_interval(self) -> None:
        self._wall_accums.clear()
        self._batch_hist.clear()
        self._reason_hist.clear()
        self._graph_hist.clear()
        self._padding_hist.clear()
        self._cuda_execute_ms.clear()
        self._steps = 0
        self._tokens = 0

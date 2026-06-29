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

"""CUDA graph compatible DeepSeek stage timing.

This probe is intentionally default-off. When enabled before CUDA graph warmup,
the stage events are captured into the replay graph, so reports describe the
last replayed decode step rather than only the Python capture pass.
"""

from __future__ import annotations

import os
from collections import defaultdict

import torch

from tensorrt_llm.logger import logger

_ENABLE_ENV_NAME = "TRTLLM_OPTRT_DEEPSEEK_STAGE_TIMING_DEBUG"
_REPORT_EVERY_ENV_NAME = "TRTLLM_OPTRT_DEEPSEEK_STAGE_TIMING_EVERY"
_RANKS_ENV_NAME = "TRTLLM_OPTRT_DEEPSEEK_STAGE_TIMING_RANKS"

_DEFAULT_REPORT_EVERY = 128
_TOP_ENTRY_COUNT = 12
_LAYER_STAGES = (
    "input_norm_gate",
    "attention",
    "post_norm_gate",
    "ffn",
    "post_ffn",
)

_TIMERS: list["DeepseekStageTimer"] = []
_STEPS_SINCE_REPORT = 0


def deepseek_stage_profile_enabled() -> bool:
    return os.environ.get(_ENABLE_ENV_NAME, "0") == "1"


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


def _new_event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True, external=True)


class DeepseekStageTimer:
    """CUDA-event stage timings for one DeepSeek component."""

    def __init__(self,
                 *,
                 rank: int | None,
                 layer_idx: int | None,
                 layer_kind: str,
                 component: str = "layer",
                 stages: tuple[str, ...] = _LAYER_STAGES) -> None:
        self.rank = rank
        self.layer_idx = -1 if layer_idx is None else layer_idx
        self.layer_kind = layer_kind
        self.component = component
        self.enabled = (deepseek_stage_profile_enabled()
                        and _rank_enabled(rank) and torch.cuda.is_available())
        self._events: dict[str, tuple[torch.cuda.Event,
                                      torch.cuda.Event]] = {}
        if not self.enabled:
            return
        self._events = {
            stage: (_new_event(), _new_event())
            for stage in stages
        }
        _TIMERS.append(self)

    def start(self, stage: str) -> None:
        if not self.enabled:
            return
        events = self._events.get(stage)
        if events is None:
            return
        start, _ = events
        start.record(torch.cuda.current_stream())

    def end(self, stage: str) -> None:
        if not self.enabled:
            return
        events = self._events.get(stage)
        if events is None:
            return
        _, end = events
        end.record(torch.cuda.current_stream())

    def elapsed_ms_by_stage(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        timings = {}
        for stage, (start, end) in self._events.items():
            try:
                timings[stage] = float(start.elapsed_time(end))
            except (RuntimeError, ValueError):
                continue
        return timings


def maybe_report_deepseek_stage_profile() -> None:
    """Report last-step stage timing after an eligible decode replay."""
    global _STEPS_SINCE_REPORT
    if not deepseek_stage_profile_enabled() or not _TIMERS:
        return
    _STEPS_SINCE_REPORT += 1
    report_every = max(1, _env_int(_REPORT_EVERY_ENV_NAME,
                                   _DEFAULT_REPORT_EVERY))
    if _STEPS_SINCE_REPORT % report_every != 0:
        return

    torch.cuda.synchronize()

    by_stage_ms: defaultdict[str, float] = defaultdict(float)
    by_kind_stage_ms: defaultdict[str, float] = defaultdict(float)
    by_kind_count: defaultdict[str, int] = defaultdict(int)
    by_component_stage_ms: defaultdict[str, float] = defaultdict(float)
    by_component_kind_stage_ms: defaultdict[str, float] = defaultdict(float)
    by_component_count: defaultdict[str, int] = defaultdict(int)
    top_entries = []
    active_timers = 0

    for timer in _TIMERS:
        stage_ms = timer.elapsed_ms_by_stage()
        if not stage_ms:
            continue
        active_timers += 1
        by_component_count[timer.component] += 1
        if timer.component == "layer":
            by_kind_count[timer.layer_kind] += 1
        for stage, elapsed_ms in stage_ms.items():
            component_stage = f"{timer.component}.{stage}"
            component_kind_stage = (
                f"{timer.component}.{timer.layer_kind}.{stage}")
            by_component_stage_ms[component_stage] += elapsed_ms
            by_component_kind_stage_ms[component_kind_stage] += elapsed_ms
            if timer.component == "layer":
                by_stage_ms[stage] += elapsed_ms
                by_kind_stage_ms[f"{timer.layer_kind}.{stage}"] += elapsed_ms
            top_entries.append({
                "component": timer.component,
                "layer": timer.layer_idx,
                "kind": timer.layer_kind,
                "stage": stage,
                "ms": elapsed_ms,
            })

    top_entries.sort(key=lambda item: item["ms"], reverse=True)
    total_ms = sum(by_stage_ms.values())
    summary = {
        "steps": _STEPS_SINCE_REPORT,
        "active_layers": by_component_count.get("layer", 0),
        "active_timers": active_timers,
        "layer_kind_count": dict(sorted(by_kind_count.items())),
        "component_count": dict(sorted(by_component_count.items())),
        "last_step_total_stage_ms": total_ms,
        "last_step_stage_ms": dict(sorted(by_stage_ms.items())),
        "last_step_kind_stage_ms": dict(sorted(by_kind_stage_ms.items())),
        "last_step_component_stage_ms":
        dict(sorted(by_component_stage_ms.items())),
        "last_step_component_kind_stage_ms":
        dict(sorted(by_component_kind_stage_ms.items())),
        "top": top_entries[:_TOP_ENTRY_COUNT],
    }
    logger.info(f"OPTRT_DEEPSEEK_STAGE_TIMING {summary}")
    _STEPS_SINCE_REPORT = 0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Probe whether route-locality WarpDecode tactics are production-usable.

This benchmark intentionally separates the no-overhead upper bound from a
hot-path dynamic selector. A route-sensitive tactic is only useful in
production if the route-locality signal is already available from scheduling or
can be computed without erasing the latency win.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from warpdecode_tactic_sweep_bucket import CURRENT_TACTICS, _run_explicit, _runner, _time_call
from warpdecode_target_harness import HIDDEN, _make_inputs

ROUTE_TACTICS = {
    16: {
        "single_expert": [32, 18],
        "slot8": [16, 48],
        "paired": CURRENT_TACTICS[16],
        "round_robin": CURRENT_TACTICS[16],
    },
    32: {
        "single_expert": [32, 23],
        "slot8": [32, 52],
        "paired": CURRENT_TACTICS[32],
        "round_robin": CURRENT_TACTICS[32],
    },
}


def _finite_scales(data: dict[str, torch.Tensor]) -> None:
    data["x_sf"] = torch.ones_like(data["x_sf"])
    data["w13_sf"] = torch.ones_like(data["w13_sf"])
    data["w2_sf"] = torch.ones_like(data["w2_sf"])


def _select_by_active_experts(tokens: int, active_experts: int) -> list[int]:
    if tokens == 32:
        if active_experts <= 1:
            return [32, 23]
        if active_experts <= 8:
            return [32, 52]
        return CURRENT_TACTICS[32]
    if tokens == 16:
        if active_experts <= 1:
            return [32, 18]
        if active_experts <= 8:
            return [16, 48]
        return CURRENT_TACTICS[16]
    return CURRENT_TACTICS[tokens]


def _active_expert_count_sync(ids: torch.Tensor) -> int:
    # Deliberately honest: turning dynamic GPU routes into a Python tactic
    # requires materialization unless scheduler metadata already provides it.
    return int(torch.unique(ids).numel())


def _cosine_and_max_abs(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    if tuple(out.shape) != tuple(ref.shape):
        raise RuntimeError(f"bad output shape {tuple(out.shape)} != {tuple(ref.shape)}")
    if not torch.isfinite(out.float()).all():
        raise RuntimeError("non-finite output")
    cos = torch.nn.functional.cosine_similarity(
        out.float().reshape(1, -1), ref.float().reshape(1, -1), dim=-1
    ).item()
    max_abs = (out.float() - ref.float()).abs().max().item()
    if not math.isfinite(cos) or cos < 0.999:
        raise RuntimeError(f"cosine {cos} below tolerance; max_abs={max_abs}")
    return cos, max_abs


def _timed_row(tokens: int, pattern: str, warmup: int, iters: int, repeats: int) -> dict[str, object]:
    torch.manual_seed(620300 + tokens * 17 + len(pattern))
    data = _make_inputs(tokens, pattern, torch.device("cuda"))
    _finite_scales(data)
    runner = _runner()
    current_tactic = CURRENT_TACTICS[tokens]
    best_tactic = ROUTE_TACTICS[tokens][pattern]
    active_precomputed = _active_expert_count_sync(data["ids"])
    selected_tactic = _select_by_active_experts(tokens, active_precomputed)

    ref = _run_explicit(runner, data, current_tactic)
    if tuple(ref.shape) != (tokens, HIDDEN):
        raise RuntimeError(f"bad reference shape {tuple(ref.shape)}")

    best = _run_explicit(runner, data, best_tactic)
    dyn = _run_explicit(runner, data, selected_tactic)
    best_cos, best_abs = _cosine_and_max_abs(best, ref)
    dyn_cos, dyn_abs = _cosine_and_max_abs(dyn, ref)

    current_min, current_med = _time_call(
        lambda: _run_explicit(runner, data, current_tactic), warmup, iters, repeats
    )
    best_min, best_med = _time_call(
        lambda: _run_explicit(runner, data, best_tactic), warmup, iters, repeats
    )

    def dynamic_call() -> torch.Tensor:
        active = _active_expert_count_sync(data["ids"])
        return _run_explicit(runner, data, _select_by_active_experts(tokens, active))

    dynamic_min, dynamic_med = _time_call(dynamic_call, warmup, iters, repeats)
    count_min, count_med = _time_call(
        lambda: _active_expert_count_sync(data["ids"]), warmup, iters, repeats
    )

    return {
        "tokens": tokens,
        "route_pattern": pattern,
        "active_experts_precomputed": active_precomputed,
        "current_tactic": current_tactic,
        "route_best_tactic": best_tactic,
        "dynamic_selected_tactic": selected_tactic,
        "best_cosine_vs_current": best_cos,
        "best_max_abs_vs_current": best_abs,
        "dynamic_cosine_vs_current": dyn_cos,
        "dynamic_max_abs_vs_current": dyn_abs,
        "current_min_ms": current_min,
        "current_median_ms": current_med,
        "route_best_min_ms": best_min,
        "route_best_median_ms": best_med,
        "dynamic_selector_min_ms": dynamic_min,
        "dynamic_selector_median_ms": dynamic_med,
        "active_count_only_min_ms": count_min,
        "active_count_only_median_ms": count_med,
        "route_best_speedup_vs_current_min": current_min / best_min,
        "dynamic_speedup_vs_current_min": current_min / dynamic_min,
        "selector_overhead_vs_best_min_ms": dynamic_min - best_min,
        "route_best_worth_promoting_upper_bound": best_min < current_min * 0.99,
        "dynamic_worth_promoting_hot_path": dynamic_min < current_min * 0.99,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="16,32")
    parser.add_argument("--patterns", default="single_expert,slot8,paired,round_robin")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for tokens in [int(x) for x in args.tokens.split(",") if x.strip()]:
        if tokens not in ROUTE_TACTICS:
            raise ValueError(f"unsupported token bucket {tokens}")
        for pattern in [x for x in args.patterns.split(",") if x.strip()]:
            row = _timed_row(tokens, pattern, args.warmup, args.iters, args.repeats)
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    payload = {
        "purpose": "Validate route-locality tactic selection with and without hot-path dynamic route counting.",
        "decision_rule": "Promote only if selector has no hidden synchronization/overhead and passes correctness plus >1% latency win.",
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

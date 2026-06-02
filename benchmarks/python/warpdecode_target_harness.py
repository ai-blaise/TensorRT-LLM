# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Target-shape WarpDecode harness for B200 NVFP4 DeepSeek decode."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import (
    ActType_TrtllmGen,
    FP4BlockScaleMoERunner,
)

HIDDEN = 7168
INTERMEDIATE = 2048
NUM_EXPERTS = 128
LOCAL_EXPERTS = 16
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
SCALE_VEC = 16
DEEPSEEK_V3_ROUTING = 2
BUCKETS = (1, 2, 4, 8, 16, 32)
TARGET_TACTICS = {
    1: [8, 26],
    2: [8, 75],
    4: [8, 53],
    8: [8, 53],
    16: [8, 53],
    32: [16, 52],
}


def _time_call(fn, warmup: int, iters: int, repeats: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        values.append(start.elapsed_time(end) / iters)
    return min(values), statistics.median(values)


def _target_tactic(tokens: int) -> list[int]:
    return TARGET_TACTICS[tokens]


def _make_inputs(tokens: int, pattern: str, device: torch.device) -> dict[str, torch.Tensor]:
    data: dict[str, torch.Tensor] = {
        "x": torch.randint(0, 256, (tokens, HIDDEN // 2), device=device, dtype=torch.uint8),
        "x_sf": torch.randint(
            0, 256, (tokens, HIDDEN // SCALE_VEC), device=device, dtype=torch.uint8
        )
        .view(torch.float8_e4m3fn)
        .flatten(),
        "w13": torch.randint(
            0,
            256,
            (LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2),
            device=device,
            dtype=torch.uint8,
        ),
        "w13_sf": torch.randint(
            0,
            256,
            (LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN // SCALE_VEC),
            device=device,
            dtype=torch.uint8,
        ).view(torch.float8_e4m3fn),
        "w2": torch.randint(
            0,
            256,
            (LOCAL_EXPERTS, HIDDEN, INTERMEDIATE // 2),
            device=device,
            dtype=torch.uint8,
        ),
        "w2_sf": torch.randint(
            0,
            256,
            (LOCAL_EXPERTS, HIDDEN, INTERMEDIATE // SCALE_VEC),
            device=device,
            dtype=torch.uint8,
        ).view(torch.float8_e4m3fn),
        "o1": torch.ones((LOCAL_EXPERTS,), device=device, dtype=torch.float32),
        "og": torch.ones((LOCAL_EXPERTS,), device=device, dtype=torch.float32),
        "o2": torch.ones((LOCAL_EXPERTS,), device=device, dtype=torch.float32),
        "weights": torch.full((tokens, TOP_K), 1.0 / TOP_K, device=device, dtype=torch.bfloat16),
    }
    if pattern == "single_expert":
        ids = torch.zeros((tokens, TOP_K), device=device, dtype=torch.int32)
    elif pattern == "slot8":
        ids = torch.arange(TOP_K, device=device, dtype=torch.int32).view(1, TOP_K)
        ids = ids.expand(tokens, TOP_K).contiguous()
    elif pattern == "paired":
        token = torch.arange(tokens, device=device, dtype=torch.int32).view(tokens, 1)
        route = torch.arange(TOP_K, device=device, dtype=torch.int32).view(1, TOP_K)
        ids = (token * 2 + route // 2) % LOCAL_EXPERTS
    else:
        ids = torch.arange(tokens * TOP_K, device=device, dtype=torch.int32).reshape(tokens, TOP_K)
        ids = ids % LOCAL_EXPERTS
    data["ids"] = ids.contiguous()
    return data


def _native(data: dict[str, torch.Tensor], do_finalize: bool):
    return torch.ops.trtllm.fp4_block_scale_moe_runner(
        None,
        None,
        data["x"],
        data["x_sf"],
        data["w13"],
        data["w13_sf"],
        None,
        None,
        None,
        None,
        data["w2"],
        data["w2_sf"],
        None,
        data["o1"],
        data["og"],
        data["o2"],
        NUM_EXPERTS,
        TOP_K,
        N_GROUP,
        TOPK_GROUP,
        INTERMEDIATE,
        0,
        LOCAL_EXPERTS,
        None,
        DEEPSEEK_V3_ROUTING,
        do_finalize,
        ActType_TrtllmGen.SwiGlu.value,
        data["weights"],
        data["ids"],
        None,
        8192,
        False,
    )


def _explicit(data: dict[str, torch.Tensor], tokens: int):
    runner = FP4BlockScaleMoERunner(
        NUM_EXPERTS,
        TOP_K,
        N_GROUP,
        TOPK_GROUP,
        INTERMEDIATE,
        0,
        LOCAL_EXPERTS,
        None,
        DEEPSEEK_V3_ROUTING,
        True,
        ActType_TrtllmGen.SwiGlu.value,
        tune_max_num_tokens=8192,
        use_dp=False,
    )
    return runner.forward(
        [
            None,
            None,
            data["x"],
            data["x_sf"],
            data["w13"],
            data["w13_sf"],
            None,
            None,
            None,
            None,
            data["w2"],
            data["w2_sf"],
            None,
            data["o1"],
            data["og"],
            data["o2"],
            data["weights"],
            data["ids"],
        ],
        tactic=_target_tactic(tokens),
    )[0]


def _parse_tokens(raw: str) -> list[int]:
    return [int(item) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="1,2,4,8,16,32")
    parser.add_argument("--route-pattern", default="round_robin")
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--include-no-finalize", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    rows = []
    device = torch.device("cuda")
    for tokens in _parse_tokens(args.tokens):
        if tokens not in BUCKETS:
            raise ValueError(f"unsupported token bucket: {tokens}")
        data = _make_inputs(tokens, args.route_pattern, device)
        native_min, native_med = _time_call(
            lambda: _native(data, True)[0], args.warmup, args.iters, args.repeats
        )
        explicit_min = explicit_med = None
        explicit_min, explicit_med = _time_call(
            lambda: _explicit(data, tokens), args.warmup, args.iters, args.repeats
        )
        no_fin_min = no_fin_med = None
        if args.include_no_finalize:
            no_fin_min, no_fin_med = _time_call(
                lambda: _native(data, False), args.warmup, args.iters, args.repeats
            )
        row = {
            "tokens": tokens,
            "native_min_ms": native_min,
            "native_median_ms": native_med,
            "explicit_min_ms": explicit_min,
            "explicit_median_ms": explicit_med,
            "no_finalize_min_ms": no_fin_min,
            "no_finalize_median_ms": no_fin_med,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

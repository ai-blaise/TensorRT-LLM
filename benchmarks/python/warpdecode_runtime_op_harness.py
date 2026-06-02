
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-op WarpDecode harness for B200 NVFP4 DeepSeek decode."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch


def _load_source_warp_decode() -> None:
    source_root = Path(os.environ.get("TENSORRT_LLM_SOURCE_ROOT", "/workspace/TensorRT-LLM"))
    module_path = source_root / "tensorrt_llm/_torch/modules/fused_moe/warp_decode.py"
    name = "tensorrt_llm._torch.modules.fused_moe.warp_decode"
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


_load_source_warp_decode()
from warpdecode_target_harness import (
    HIDDEN,
    INTERMEDIATE,
    LOCAL_EXPERTS,
    NUM_EXPERTS,
    SCALE_VEC,
    _explicit,
    _make_inputs,
    _parse_tokens,
    _time_call,
)


def _runtime_op(data: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.ops.trtllm.warp_decode_nvfp4_moe(
        data["x"],
        data["x_sf"],
        data["w13"],
        data["w13_sf"],
        data["w2"],
        data["w2_sf"],
        data["o1"],
        data["og"],
        data["o2"],
        data["ids"],
        data["weights"],
        HIDDEN,
        INTERMEDIATE,
        NUM_EXPERTS,
        0,
        LOCAL_EXPERTS,
        SCALE_VEC,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="1,2,4,8,16,32")
    parser.add_argument("--route-pattern", default="round_robin")
    parser.add_argument("--warmup", type=int, default=24)
    parser.add_argument("--iters", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    os.environ.pop("TRTLLM_ENABLE_PDL", None)
    device = torch.device("cuda")
    rows = []
    for tokens in _parse_tokens(args.tokens):
        data = _make_inputs(tokens, args.route_pattern, device)
        for name in ("x_sf", "w13_sf", "w2_sf"):
            data[name].fill_(1.0)
        runtime = _runtime_op(data)
        explicit = _explicit(data, tokens)
        torch.cuda.synchronize()
        diff = (runtime.float() - explicit.float()).abs()
        cos = torch.nn.functional.cosine_similarity(
            runtime.float().flatten(), explicit.float().flatten(), dim=0
        )
        runtime_min, runtime_med = _time_call(
            lambda: _runtime_op(data), args.warmup, args.iters, args.repeats
        )
        row = {
            "tokens": tokens,
            "route_pattern": args.route_pattern,
            "runtime_op_min_ms": runtime_min,
            "runtime_op_median_ms": runtime_med,
            "max_abs_vs_explicit": float(diff.max()),
            "cos_vs_explicit": float(cos),
            "pdl_env_after_runtime_op": os.environ.get("TRTLLM_ENABLE_PDL"),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

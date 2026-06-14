#!/usr/bin/env python3
"""Serving-layout import smoke for OP-TRT HiSparse native ops.

This proof intentionally goes through the normal TensorRT-LLM package import
path. Importing ``tensorrt_llm`` must load ``tensorrt_llm/libs/libth_common.so``;
the planner/copy smoke then runs without passing an explicit ``--library``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _loaded_th_common_paths() -> list[Path]:
    paths: list[Path] = []
    for raw in torch.classes.loaded_libraries:
        path = Path(str(raw))
        if path.name in ("libth_common.so", "th_common.dll"):
            paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--record-bytes", type=int, default=32)
    parser.add_argument(
        "--expect-site-packages",
        type=Path,
        default=Path("/opt/dynamo/venv/lib/python3.12/site-packages"),
        help="Expected site-packages root for the imported tensorrt_llm package.",
    )
    args = parser.parse_args()

    import tensorrt_llm  # noqa: F401

    package_root = Path(tensorrt_llm.__path__[0]).resolve()
    expected_root = (args.expect_site_packages / "tensorrt_llm").resolve()
    if package_root != expected_root:
        raise RuntimeError(
            f"expected tensorrt_llm from {expected_root}, imported {package_root}"
        )

    expected_th_common = (package_root / "libs" / "libth_common.so").resolve()
    loaded = [path.resolve() for path in _loaded_th_common_paths()]
    if expected_th_common not in loaded:
        raise RuntimeError(
            "normal package import did not load the serving-layout "
            f"libth_common.so. expected={expected_th_common} loaded={loaded}"
        )

    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    import native_planner_copy_smoke
    import sparse_mla_kvarn_hot_smoke

    sys.argv = [
        str(script_dir / "native_planner_copy_smoke.py"),
        "--device",
        args.device,
        "--record-bytes",
        str(args.record_bytes),
    ]
    native_planner_copy_smoke.main()
    sys.argv = [
        str(script_dir / "sparse_mla_kvarn_hot_smoke.py"),
        "--device",
        args.device,
    ]
    sparse_mla_kvarn_hot_smoke.main()
    print(f"serving import smoke used package={package_root}")
    print(f"serving import smoke used libth_common={expected_th_common}")


if __name__ == "__main__":
    main()

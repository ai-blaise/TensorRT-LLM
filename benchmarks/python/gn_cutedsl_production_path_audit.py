#!/usr/bin/env python3
"""Audit gn-kernels CuTeDSL kernels against the op-trt production path.

This script is deliberately conservative: kernels marked outside the current
production path are inventoried but not benchmarked.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


GN_COMMIT = "94cdf76a25a8762c8daf989ec54a28a8607d872a"
DEFAULT_GN_ROOT = Path("/home/spencer/work/gn-kernels-94cdf76-cutedsl")


@dataclass(frozen=True)
class KernelMapping:
    gn_file: str
    kernel: str
    gn_contract: str
    closest_op_trt_surface: str
    production_path: bool
    benchmark: bool
    decision: str


MAPPINGS = (
    KernelMapping(
        gn_file="gn_kernels/cutedsl/sm100_mm_bf16.py",
        kernel="MatmulSm100 / mm(A, B)",
        gn_contract="B200 SM100 dense BF16 GEMM, A[M,K], B[K,N], C[M,N] BF16, fixed BN=256 cta_group=2 wrapper.",
        closest_op_trt_surface=(
            "trtllm::cute_dsl_bf16_gemm_blackwell / "
            "tensorrt_llm/_torch/cute_dsl_kernels/blackwell/dense_gemm_persistent.py"
        ),
        production_path=False,
        benchmark=False,
        decision=(
            "Out of scope for the live stack: BF16 CuteDSL GEMM is behind "
            "use_cute_dsl_bf16_gemm, while the current production path is "
            "WarpDecode NVFP4 MoE plus DSA/HISA/KVarN attention."
        ),
    ),
    KernelMapping(
        gn_file="gn_kernels/cutedsl/sm100_mm_mxfp8.py",
        kernel="MatmulMXFP8Sm100 / mm(A, B, SFA, SFB)",
        gn_contract=(
            "B200 SM100 dense MXFP8 GEMM, Float8E4M3FN A/B, E8M0 scale factors in "
            "NVIDIA MMA layout, BF16 output."
        ),
        closest_op_trt_surface=(
            "trtllm::cute_dsl_fp8_gemm_blackwell / "
            "dense_blockscaled_gemm_persistent.py"
        ),
        production_path=False,
        benchmark=False,
        decision=(
            "Out of scope for the live stack: generic dense MXFP8/blockscaled GEMM "
            "does not map to KVarN/DSA paged MQA logits or WarpDecode grouped MoE."
        ),
    ),
    KernelMapping(
        gn_file="gn_kernels/cutedsl/sm100_mm_nvfp4.py",
        kernel="MatmulNVFP4Sm100 / mm(A, B, SFA, SFB)",
        gn_contract=(
            "B200 SM100 dense NVFP4 GEMM, packed Float4E2M1 A/B, E4M3/E8M0-style "
            "scale factors in MMA layout, BF16 output."
        ),
        closest_op_trt_surface=(
            "Related but not equivalent to WarpDecode "
            "trtllm::cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell and "
            "trtllm::cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell; closest "
            "plain surface is trtllm::cute_dsl_nvfp4_gemm_blackwell."
        ),
        production_path=False,
        benchmark=False,
        decision=(
            "Out of scope for promotion: live WarpDecode requires gather/grouped "
            "expert routing, tile_idx metadata, activation/quantization fusion, "
            "finalize/combine, EP composability, and fail-closed no-fallback behavior. "
            "The gn kernel is plain dense GEMM and is not an equivalent."
        ),
    ),
    KernelMapping(
        gn_file="gn_kernels/cutedsl/sm80_mm_bf16.py",
        kernel="MatmulSm80 / mm(A, B)",
        gn_contract="SM80 BF16 dense GEMM using cp.async and warp MMA.",
        closest_op_trt_surface="None for the B200 SM100 live path.",
        production_path=False,
        benchmark=False,
        decision="Out of scope: production target is B200/SM100, not SM80.",
    ),
)


def run(cmd: list[str], cwd: Path | None = None, check: bool = False) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=check)
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(cmd, 127, "", str(exc))


def git_rev(path: Path) -> str | None:
    if not path.is_dir():
        return None
    proc = run(["git", "rev-parse", "HEAD"], cwd=path)
    return proc.stdout.strip() if proc.returncode == 0 else None


def nvidia_smi() -> dict[str, Any]:
    query = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    pmon = run(["nvidia-smi", "pmon", "-c", "1"])
    gpus: list[dict[str, Any]] = []
    if query.returncode == 0:
        for line in query.stdout.splitlines():
            idx, name, used, total, util = [part.strip() for part in line.split(",")]
            gpus.append(
                {
                    "index": int(idx),
                    "name": name,
                    "memory_used_mib": int(used),
                    "memory_total_mib": int(total),
                    "utilization_gpu_pct": int(util),
                }
            )
    active_pmon = [
        line for line in pmon.stdout.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return {
        "available": query.returncode == 0,
        "gpus": gpus,
        "pmon_active_rows": active_pmon,
        "occupied": any(gpu["memory_used_mib"] > 0 for gpu in gpus) or bool(active_pmon),
    }


def build_report(gn_root: Path) -> dict[str, Any]:
    missing = [
        mapping.gn_file for mapping in MAPPINGS
        if not (gn_root / mapping.gn_file).is_file()
    ]
    return {
        "gn_root": str(gn_root),
        "expected_gn_commit": GN_COMMIT,
        "actual_gn_commit": git_rev(gn_root),
        "missing_gn_files": missing,
        "gpu_occupancy": nvidia_smi(),
        "mappings": [asdict(mapping) for mapping in MAPPINGS],
        "production_path_candidates": [
            asdict(mapping) for mapping in MAPPINGS if mapping.production_path
        ],
        "benchmark_policy": (
            "Benchmark only production_path=true mappings. Current audit has no "
            "production-path equivalent candidates, so GPU benchmarks are skipped."
        ),
        "integration_decision": (
            "No gn kernel is promoted. No measured performance win exists on an "
            "equivalent production-path surface."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gn-root", type=Path, default=DEFAULT_GN_ROOT)
    parser.add_argument("--json", type=Path, help="Optional JSON report path.")
    parser.add_argument(
        "--allow-gpu-bench",
        action="store_true",
        help="Reserved for future production-path candidates; no effect while none are mapped.",
    )
    args = parser.parse_args()

    report = build_report(args.gn_root)
    if report["actual_gn_commit"] != GN_COMMIT:
        report["status"] = "failed"
        report["failure"] = "gn-kernels checkout is not pinned to the requested commit"
    elif report["missing_gn_files"]:
        report["status"] = "failed"
        report["failure"] = "one or more expected gn_kernels/cutedsl files are missing"
    else:
        report["status"] = "ok"

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json:
        args.json.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

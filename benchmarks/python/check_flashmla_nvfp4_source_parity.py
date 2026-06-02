#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Check that op-trt's native NVFP4 sparse MLA copy matches FlashMLA."""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import sys


CORE_FILES = (
    "defines.h",
    "params.h",
    "utils.h",
    "sm100/decode/head64_nvfp4/config.h",
    "sm100/decode/head64_nvfp4/kernel.cuh",
    "sm100/decode/head64_nvfp4/kernel.h",
    "sm100/decode/head64_nvfp4/instantiations/v32.cu",
    "sm100/helpers.h",
    "smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu",
    "smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h",
    "smxx/decode/combine/combine.h",
)

KERUTILS_FILES = (
    "common/common.h",
    "device/common.h",
    "device/device.cuh",
    "device/sm100/gemm.cuh",
    "device/sm100/helpers.cuh",
    "device/sm100/intrinsics.cuh",
    "device/sm100/tma_cta_group2_nosplit.cuh",
    "device/sm80/helpers.cuh",
    "device/sm80/intrinsics.cuh",
    "device/sm90/helpers.cuh",
    "device/sm90/intrinsics.cuh",
    "host/host.h",
    "kerutils.cuh",
    "supplemental/torch_tensors.h",
)

COMBINE_FILE = "smxx/decode/combine/combine.cu"


def read(path: Path) -> str:
    try:
        return path.read_text()
    except FileNotFoundError as exc:
        raise SystemExit(f"missing file: {path}") from exc


def diff_text(ref_path: Path, op_path: Path, ref: str, op: str) -> str:
    return "".join(
        difflib.unified_diff(
            ref.splitlines(keepends=True),
            op.splitlines(keepends=True),
            fromfile=str(ref_path),
            tofile=str(op_path),
        )
    )


def normalize_allowed_combine_delta(text: str) -> str:
    extra_buckets = """        } else if (NUM_SPLITS <= 256) {                    \\
            constexpr static int NAME = 256;               \\
            return __VA_ARGS__();                          \\
        } else if (NUM_SPLITS <= 512) {                    \\
            constexpr static int NAME = 512;               \\
            return __VA_ARGS__();                          \\
        } else if (NUM_SPLITS <= 1024) {                   \\
            constexpr static int NAME = 1024;              \\
            return __VA_ARGS__();                          \\
"""
    text = text.replace(extra_buckets, "")
    text = text.replace("        // Use cudaLaunchKernelEx to enable PDL (Programmatic Dependent Launch)\n", "")
    text = text.replace("            smem_size,\n", "            0,\n")
    return text


def check_exact_pair(ref_root: Path, op_root: Path, ref_rel: str, op_rel: str) -> list[str]:
    ref_path = ref_root / ref_rel
    op_path = op_root / op_rel
    ref = read(ref_path)
    op = read(op_path)
    if ref == op:
        return []
    return [f"{op_rel} differs from FlashMLA {ref_rel}:\n{diff_text(ref_path, op_path, ref, op)}"]


def check_combine(ref_root: Path, op_root: Path) -> list[str]:
    ref_path = ref_root / COMBINE_FILE
    op_path = op_root / COMBINE_FILE
    ref = normalize_allowed_combine_delta(read(ref_path))
    op = normalize_allowed_combine_delta(read(op_path))
    if op == ref:
        return []
    return [
        f"{COMBINE_FILE} has unapproved drift beyond the op-trt split/launch adapter:\n"
        + diff_text(ref_path, op_path, ref, op)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flashmla-csrc", required=True, type=Path)
    parser.add_argument("--optrt-nvfp4-sparse", required=True, type=Path)
    args = parser.parse_args()

    failures: list[str] = []
    strict_pair_count = 0
    for rel in CORE_FILES:
        failures.extend(check_exact_pair(args.flashmla_csrc, args.optrt_nvfp4_sparse, rel, rel))
        strict_pair_count += 1
    for rel in KERUTILS_FILES:
        failures.extend(
            check_exact_pair(
                args.flashmla_csrc,
                args.optrt_nvfp4_sparse,
                f"kerutils/include/kerutils/{rel}",
                f"kerutils/{rel}",
            )
        )
        strict_pair_count += 1
    failures.extend(check_combine(args.flashmla_csrc, args.optrt_nvfp4_sparse))

    if failures:
        print("FlashMLA NVFP4 source parity failed", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("FlashMLA NVFP4 source parity passed")
    print(f"strict_exact_files={strict_pair_count}")
    print("allowed_combine_delta=dispatch buckets <=256/512/1024 plus dynamic shared-memory launch size")
    print("intentionally_omitted=FlashMLA api/cutlass vendor tree/head128/head64 BF16/prefill/sm90/model1/q_prequant, because this import targets only sparse MLA NVFP4 decode through op-trt wrappers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

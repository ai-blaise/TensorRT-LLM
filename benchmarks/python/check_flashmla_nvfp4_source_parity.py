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


CONFIG_FILE = "sm100/decode/head64_nvfp4/config.h"
KERNEL_FILE = "sm100/decode/head64_nvfp4/kernel.cuh"


def normalize_allowed_config_delta(text: str) -> str:
    text = text.replace(
        "// V32 full-NVFP4 token layout: 576 packed FP4 score dims (288 B),\n"
        "// 36 E4M3 scales (one per 16 dims), then 12 B padding for 16 B alignment.\n"
        "// PV consumes only the first D_V=512 dequantized dims.\n",
        "// V32 full-NVFP4 score cache layout for op-trt: 576 packed FP4 score dims\n"
        "// (288 B/token) in the KV data pool. E4M3 block scales are stored in the\n"
        "// separate KV scale pool at 36 B/token, matching TensorRT-LLM NVFP4 cache\n"
        "// storage and avoiding request-time repacking. PV consumes the first D_V=512\n"
        "// dequantized dims.\n",
    )
    text = text.replace(
        "static constexpr int NVFP4_TOKEN_BYTES = MODEL_TYPE == ModelType::V32 ? 336 : (D_NOPE/2)+2*D_ROPE+NUM_SCALES_EACH_TOKEN;",
        "static constexpr int NVFP4_TOKEN_BYTES = MODEL_TYPE == ModelType::V32 ? NVFP4_SCORE_BYTES : (D_NOPE/2)+2*D_ROPE+NUM_SCALES_EACH_TOKEN;",
    )
    return text


def normalize_allowed_kernel_delta(text: str) -> str:
    text = text.replace(
        """            uint8_t* k_scales_ptr =\n                MODEL_TYPE == ModelType::V32 ?\n                (uint8_t*)params.kv + NVFP4_SCORE_BYTES :\n                (uint8_t*)params.kv + params.page_block_size*((D_NOPE/2)+2*D_ROPE);\n            uint8_t* extra_k_scales_ptr =\n                MODEL_TYPE == ModelType::V32 ?\n                (uint8_t*)params.extra_kv + NVFP4_SCORE_BYTES :\n                (uint8_t*)params.extra_kv + params.extra_page_block_size*((D_NOPE/2)+2*D_ROPE);\n""",
        """            uint8_t* k_scales_ptr =\n                MODEL_TYPE == ModelType::V32 ?\n                params.kv_scales :\n                (uint8_t*)params.kv + params.page_block_size*((D_NOPE/2)+2*D_ROPE);\n            uint8_t* extra_k_scales_ptr =\n                MODEL_TYPE == ModelType::V32 ?\n                nullptr :\n                (uint8_t*)params.extra_kv + params.extra_page_block_size*((D_NOPE/2)+2*D_ROPE);\n""",
    )
    text = text.replace(
        """                    int64_t cur_k_block_stride = IS_EXTRA_BLOCK ? params.stride_extra_kv_block : params.stride_kv_block;\n                    [[maybe_unused]] int cur_k_row_stride = IS_EXTRA_BLOCK ? params.stride_extra_kv_row : params.stride_kv_row;\n                    uint8_t* cur_k_scales_ptr = IS_EXTRA_BLOCK ? extra_k_scales_ptr : k_scales_ptr;\n""",
        """                    int64_t cur_k_block_stride = IS_EXTRA_BLOCK ? params.stride_extra_kv_block : params.stride_kv_block;\n                    [[maybe_unused]] int cur_k_row_stride = IS_EXTRA_BLOCK ? params.stride_extra_kv_row : params.stride_kv_row;\n                    int64_t cur_scale_block_stride =\n                        (MODEL_TYPE == ModelType::V32 && !IS_EXTRA_BLOCK) ? params.stride_kv_scales_block : cur_k_block_stride;\n                    int cur_scale_row_stride =\n                        (MODEL_TYPE == ModelType::V32 && !IS_EXTRA_BLOCK) ? params.stride_kv_scales_row : cur_k_row_stride;\n                    uint8_t* cur_k_scales_ptr = IS_EXTRA_BLOCK ? extra_k_scales_ptr : k_scales_ptr;\n""",
    )
    text = text.replace(
        """                        if constexpr (MODEL_TYPE == ModelType::V32) {\n                            offset = is_token_valid ? block_idx*cur_k_block_stride + idx_in_block*cur_k_row_stride : 0;\n                        } else {\n""",
        """                        if constexpr (MODEL_TYPE == ModelType::V32) {\n                            offset = is_token_valid ? block_idx*cur_scale_block_stride + idx_in_block*cur_scale_row_stride : 0;\n                        } else {\n""",
    )
    text = text.replace(
        """                            const uint4* src_vec = reinterpret_cast<const uint4*>(src);\n                            uint4 v0 = __ldg(src_vec + 0);\n                            uint4 v1 = __ldg(src_vec + 1);\n                            dst_words[0] = v0.x;\n                            dst_words[1] = v0.y;\n                            dst_words[2] = v0.z;\n                            dst_words[3] = v0.w;\n                            dst_words[4] = v1.x;\n                            dst_words[5] = v1.y;\n                            dst_words[6] = v1.z;\n                            dst_words[7] = v1.w;\n                            if constexpr (NUM_SCALES_EACH_TOKEN > 32) {\n                                dst_words[8] = __ldg(reinterpret_cast<const uint32_t*>(src + 32));\n                            }\n""",
        """                            const uint32_t* src_words = reinterpret_cast<const uint32_t*>(src);\n                            CUTE_UNROLL\n                            for (int s = 0; s < NUM_SCALES_EACH_TOKEN / 4; ++s) {\n                                dst_words[s] = __ldg(src_words + s);\n                            }\n""",
    )
    text = text.replace(
        """    if constexpr (MODEL_TYPE == ModelType::MODEL1) {\n        constexpr int BYTES_PER_TOKEN = NVFP4_TOKEN_BYTES;\n        KU_ASSERT(params.stride_kv_row == BYTES_PER_TOKEN, "Each page block in KV cache must be contiguous for head64 sparse fp8 decoding attention in MODEL1");  // Each block must be contiguous\n    }\n""",
        """    constexpr int BYTES_PER_TOKEN = NVFP4_TOKEN_BYTES;\n    KU_ASSERT(params.stride_kv_row == BYTES_PER_TOKEN, "Each page block in NVFP4 KV cache must be contiguous for head64 sparse decoding attention");\n    if constexpr (MODEL_TYPE == ModelType::V32) {\n        KU_ASSERT(params.kv_scales != nullptr, "V3.2 NVFP4 decode requires separate KV scale pool");\n        KU_ASSERT(params.stride_kv_scales_row == NUM_SCALES_EACH_TOKEN, "V3.2 NVFP4 scales must be contiguous per token");\n    }\n""",
    )
    return text


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


def check_normalized_file(ref_root: Path, op_root: Path, rel: str, normalizer, label: str) -> list[str]:
    ref_path = ref_root / rel
    op_path = op_root / rel
    ref = normalizer(read(ref_path))
    op = normalizer(read(op_path))
    if op == ref:
        return []
    return [f"{rel} has unapproved drift beyond {label}:\n" + diff_text(ref_path, op_path, ref, op)]


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
    failures.extend(check_normalized_file(args.flashmla_csrc, args.optrt_nvfp4_sparse, CONFIG_FILE, normalize_allowed_config_delta, "the op-trt split scale-pool config adapter"))
    failures.extend(check_normalized_file(args.flashmla_csrc, args.optrt_nvfp4_sparse, KERNEL_FILE, normalize_allowed_kernel_delta, "the op-trt split scale-pool kernel adapter"))
    failures.extend(check_combine(args.flashmla_csrc, args.optrt_nvfp4_sparse))

    if failures:
        print("FlashMLA NVFP4 source parity failed", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("FlashMLA NVFP4 source parity passed")
    print(f"strict_exact_files={strict_pair_count}")
    print("allowed_config_kernel_delta=split op-trt data/scale pools instead of FlashMLA inline 336B row; scalar 32-bit scale loads because 36B split-scale rows are not 16B-aligned")
    print("allowed_combine_delta=dispatch buckets <=256/512/1024; dynamic shared-memory launch size normalized to FlashMLA latest zero-smem launch")
    print("intentionally_omitted=FlashMLA api/cutlass vendor tree/head128/head64 BF16/prefill/sm90/model1/q_prequant, because this import targets only sparse MLA NVFP4 decode through op-trt wrappers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

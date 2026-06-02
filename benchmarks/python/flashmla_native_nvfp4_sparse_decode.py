#!/usr/bin/env python3
"""Benchmark native op-trt NVFP4 sparse MLA decode against FlashMLA.

This harness targets DeepSeek-V3.2 MLA decode on B200:
  q:         [B, 1, 128, 576] bf16
  kv:        [num_pages, 64, 1, 288] uint8 packed e2m1
  kv_scales: [num_pages, 64, 1, 36] uint8 E4M3/UE4M3 scale bytes
  indices:   [B, 1, TopK] int32 token ids

It intentionally measures the preallocated scheduler-metadata path because that
is the production/comparable FlashMLA baseline.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch

H_Q = 128
H_KV = 1
D_QK = 576
D_V = 512
PAGE_BLOCK_SIZE = 64
PACKED_BYTES = 288
SCALE_BYTES = 36
SCALE_BLOCK = 16
DEVICE = "cuda"

E2M1_VALUES = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def load_native_extension(repo_root: Path, build_dir: Path):
    from torch.utils.cpp_extension import load

    cutlass_root = build_dir / "_deps" / "cutlass-src"
    if not (cutlass_root / "include").exists():
        raise FileNotFoundError(
            f"CUTLASS headers not found under {cutlass_root}; run the focused CMake configure first"
        )

    flashmla_root = repo_root / "cpp" / "tensorrt_llm" / "kernels" / "flashMLA"
    nvfp4_root = flashmla_root / "nvfp4_sparse"
    sources = [
        repo_root / "benchmarks" / "python" / "flashmla_native_nvfp4_sparse_decode_ext.cpp",
        flashmla_root / "sparse_mla_decode_nvfp4.cu",
        nvfp4_root / "sm100" / "decode" / "head64_nvfp4" / "instantiations" / "v32.cu",
        nvfp4_root / "smxx" / "decode" / "get_decoding_sched_meta" / "get_decoding_sched_meta.cu",
        nvfp4_root / "smxx" / "decode" / "combine" / "combine.cu",
    ]
    include_dirs = [
        repo_root / "cpp",
        nvfp4_root,
        nvfp4_root / "kerutils" / "include",
        nvfp4_root / "sm100" / "decode" / "head64_nvfp4",
        nvfp4_root / "smxx" / "decode" / "get_decoding_sched_meta",
        nvfp4_root / "smxx" / "decode" / "combine",
        cutlass_root / "include",
        cutlass_root / "tools" / "util" / "include",
    ]
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0")
    return load(
        name="optrt_flashmla_native_nvfp4_bench",
        sources=[str(x) for x in sources],
        extra_include_paths=[str(x) for x in include_dirs],
        extra_cflags=["-O3", "-std=c++20", "-DNDEBUG", "-Wno-deprecated-declarations"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "-DNDEBUG",
            "-D_USE_MATH_DEFINES",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "--ptxas-options=-v,--register-usage-level=10,--warn-on-spills,--warn-on-local-memory-usage",
            "-gencode=arch=compute_100f,code=sm_100f",
        ],
        build_directory=str(repo_root / ".torch_ext" / "flashmla_native_nvfp4"),
        verbose=True,
        with_cuda=True,
    )


def load_flashmla(flashmla_root: Path | None):
    if flashmla_root is not None:
        sys.path.insert(0, str(flashmla_root))
    return importlib.import_module("flash_mla.cuda")


def dequant_fp4_packed(packed_bytes: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=packed_bytes.device)
    low = packed_bytes & 0xF
    high = (packed_bytes >> 4) & 0xF
    out = torch.empty(
        *packed_bytes.shape[:-1],
        packed_bytes.shape[-1] * 2,
        dtype=torch.float32,
        device=packed_bytes.device,
    )
    out[..., 0::2] = table[low.long()]
    out[..., 1::2] = table[high.long()]
    return out


def reference_attention(q: torch.Tensor, kv: torch.Tensor, kv_scales: torch.Tensor, indices: torch.Tensor, sm_scale: float):
    b, s_q, h_q, _ = q.shape
    out = torch.zeros(b, s_q, h_q, D_V, dtype=torch.float32, device=q.device)
    for batch_i in range(b):
        for query_i in range(s_q):
            tok = indices[batch_i, query_i]
            block_idx = tok // PAGE_BLOCK_SIZE
            in_block = tok % PAGE_BLOCK_SIZE
            score_bytes = kv[block_idx, in_block, 0, :]
            scale_bytes = kv_scales[block_idx, in_block, 0, :]
            score = dequant_fp4_packed(score_bytes)
            scales = scale_bytes.view(torch.float8_e4m3fn).float()
            k = score * scales.repeat_interleave(SCALE_BLOCK, dim=-1)
            logits = q[batch_i, query_i].float() @ k.transpose(0, 1) * sm_scale
            attn = torch.softmax(logits, dim=-1)
            out[batch_i, query_i] = attn @ k[:, :D_V]
    return out.bfloat16()


def make_inputs(batch_size: int, topk: int, num_blocks: int | None = None, seed: int = 0):
    torch.manual_seed(seed)
    if num_blocks is None:
        num_blocks = max((batch_size * topk + PAGE_BLOCK_SIZE - 1) // PAGE_BLOCK_SIZE * 2, 64)
    q = torch.randn(batch_size, 1, H_Q, D_QK, dtype=torch.bfloat16, device=DEVICE) * 0.01
    indices = torch.randint(0, num_blocks * PAGE_BLOCK_SIZE, (batch_size, 1, topk), dtype=torch.int32, device=DEVICE)
    topk_length = torch.full((batch_size,), topk, dtype=torch.int32, device=DEVICE)
    kv = torch.randint(0, 256, (num_blocks, PAGE_BLOCK_SIZE, H_KV, PACKED_BYTES), dtype=torch.uint8, device=DEVICE)
    kv_scales = torch.randint(0x30, 0x48, (num_blocks, PAGE_BLOCK_SIZE, H_KV, SCALE_BYTES), dtype=torch.uint8, device=DEVICE)
    sm_scale = 1.0 / (D_QK ** 0.5)
    return q, kv, kv_scales, indices, topk_length, sm_scale


def time_cuda(fn: Callable[[], object], warmup: int, iters: int) -> tuple[float, float, list[float]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 / iters)
    values.sort()
    return values[0], statistics.median(values), values


def native_call(native_ext, q, kv, kv_scales, indices, topk_length, sm_scale, metadata=None, splits=None):
    return native_ext.sparse_mla_decode_nvfp4(
        q, kv, kv_scales, indices, topk_length, metadata, splits, sm_scale
    )


def flashmla_call(fc, q, kv, kv_scales, indices, topk_length, sm_scale, metadata=None, splits=None):
    return fc.sparse_decode_fwd_nvfp4(
        q, kv, kv_scales, indices, topk_length, None, metadata, splits, D_V, sm_scale
    )


def tensor_compare(a: torch.Tensor, b: torch.Tensor):
    diff = (a.float() - b.float()).abs()
    rms = diff.square().mean().sqrt()
    cos = torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0)
    return {
        "max_abs": float(diff.max().item()),
        "rms": float(rms.item()),
        "cosine": float(cos.item()),
    }


def run_correctness(native_ext, fc, args):
    rows = []
    for topk in [int(x) for x in args.correctness_topks.split(",") if x]:
        q, kv, kv_scales, indices, topk_length, sm_scale = make_inputs(args.correctness_batch, topk, seed=123 + topk)
        native_out, native_lse, native_meta, native_splits, _, _ = native_call(
            native_ext, q, kv, kv_scales, indices, topk_length, sm_scale
        )
        ref = reference_attention(q, kv, kv_scales, indices, sm_scale)
        torch.cuda.synchronize()
        row = {
            "kind": "correctness",
            "batch_size": args.correctness_batch,
            "topk": topk,
            "native_vs_reference": tensor_compare(native_out, ref),
            "metadata_shape": list(native_meta.shape),
            "splits_shape": list(native_splits.shape),
            "finite": bool(torch.isfinite(native_out).all().item()) and bool(torch.isfinite(native_lse).all().item()),
        }
        if fc is not None:
            flash_out, _, _, _ = flashmla_call(fc, q, kv, kv_scales, indices, topk_length, sm_scale)
            row["native_vs_flashmla"] = tensor_compare(native_out, flash_out)
        row["passed"] = (
            row["finite"]
            and row["native_vs_reference"]["max_abs"] < 0.5
            and row["native_vs_reference"]["rms"] < 0.065
            and row["native_vs_reference"]["cosine"] > 0.994
            and (
                fc is None
                or (
                    row["native_vs_flashmla"]["max_abs"] < 0.5
                    and row["native_vs_flashmla"]["rms"] < 0.065
                    and row["native_vs_flashmla"]["cosine"] > 0.994
                )
            )
        )
        print(json.dumps(row, sort_keys=True))
        rows.append(row)
        if not row["passed"]:
            raise AssertionError(f"correctness failed for topk={topk}: {row}")
    return rows


def run_bench(native_ext, fc, args):
    rows = []
    for batch_size in [int(x) for x in args.batch_sizes.split(",") if x]:
        iters = min(args.iters, 50) if batch_size >= 128 else args.iters
        q, kv, kv_scales, indices, topk_length, sm_scale = make_inputs(batch_size, args.topk, seed=1000 + batch_size)
        native_out, native_lse, native_meta, native_splits, _, _ = native_call(
            native_ext, q, kv, kv_scales, indices, topk_length, sm_scale
        )
        torch.cuda.synchronize()
        row = {
            "kind": "benchmark",
            "batch_size": batch_size,
            "topk": args.topk,
            "iters": iters,
            "native_finite": bool(torch.isfinite(native_out).all().item()) and bool(torch.isfinite(native_lse).all().item()),
            "native_metadata_shape": list(native_meta.shape),
            "native_splits_last": int(native_splits[-1].item()),
        }
        if fc is not None:
            flash_out, _, flash_meta, flash_splits = flashmla_call(fc, q, kv, kv_scales, indices, topk_length, sm_scale)
            torch.cuda.synchronize()
            cmp_row = tensor_compare(native_out, flash_out)
            if not (cmp_row["max_abs"] < 0.5 and cmp_row["rms"] < 0.065 and cmp_row["cosine"] > 0.994):
                raise AssertionError(f"native output does not match FlashMLA at batch={batch_size}: {cmp_row}")
            flash_min, flash_med, flash_vals = time_cuda(
                lambda: flashmla_call(fc, q, kv, kv_scales, indices, topk_length, sm_scale, flash_meta, flash_splits),
                args.warmup,
                iters,
            )
            row.update({
                "flashmla_prealloc_min_us": flash_min,
                "flashmla_prealloc_median_us": flash_med,
                "flashmla_values_us": flash_vals,
                "compare": cmp_row,
            })
        native_min, native_med, native_vals = time_cuda(
            lambda: native_call(native_ext, q, kv, kv_scales, indices, topk_length, sm_scale, native_meta, native_splits),
            args.warmup,
            iters,
        )
        row.update({
            "native_prealloc_min_us": native_min,
            "native_prealloc_median_us": native_med,
            "native_values_us": native_vals,
        })
        if fc is not None:
            row["native_vs_flashmla_speedup_median"] = row["flashmla_prealloc_median_us"] / native_med
            row["accepted_vs_flashmla"] = native_med <= row["flashmla_prealloc_median_us"]
        print(json.dumps(row, sort_keys=True))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=Path("cpp/build_native_nvfp4"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--flashmla-root", type=Path, default=None)
    parser.add_argument("--batch-sizes", default="32,64,128")
    parser.add_argument("--topk", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--correctness-batch", type=int, default=1)
    parser.add_argument("--correctness-topks", default="64,1024")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--native-only", action="store_true", help="Skip FlashMLA Python extension import and run native finite/reference gates only.")
    args = parser.parse_args()

    native_ext = load_native_extension(args.repo_root.resolve(), args.build_dir.resolve())
    fc = None if args.native_only else load_flashmla(args.flashmla_root)
    print(json.dumps({"kind": "setup", "native_extension": str(native_ext), "flashmla_root": str(args.flashmla_root) if args.flashmla_root else None, "native_only": args.native_only}, sort_keys=True))
    rows = []
    rows.extend(run_correctness(native_ext, fc, args))
    rows.extend(run_bench(native_ext, fc, args))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

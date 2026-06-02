#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

HIDDEN = 7168
PACKED_HIDDEN = HIDDEN // 2
INTERMEDIATE = 2048
EXPERTS = 128
TOP_K = 8
SCALE_COLS = HIDDEN // 16
WARPS_PER_CTA = 8
BLACKWELL_FP4_GENCODE = ["-gencode=arch=compute_100f,code=sm_100f"]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CPP_SRC = r"""
#include <torch/extension.h>

void cursor_nvfp4_predecode_x_bf16(torch::Tensor x_packed, torch::Tensor x_bf16);
void cursor_nvfp4_gate_up_predecoded_x_out(
    torch::Tensor x_bf16,
    torch::Tensor x_sf,
    torch::Tensor w13_packed,
    torch::Tensor w13_sf,
    torch::Tensor token_selected_slots,
    torch::Tensor scratch);
"""

CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {
constexpr int kWarpSize = 32;
constexpr int kWarpsPerCta = 8;
constexpr int kHidden = 7168;

__device__ __forceinline__ float warp_sum(float value) {
  #pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, mask);
  }
  return value;
}

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ void e2m1_uint32_to_float8(uint32_t packed, float (&out)[8]) {
  uint32_t out_fp16[4];
  asm volatile(
      "{\n"
      ".reg .b8 byte0, byte1, byte2, byte3;\n"
      "mov.b32 {byte0, byte1, byte2, byte3}, %4;\n"
      "cvt.rn.f16x2.e2m1x2 %0, byte0;\n"
      "cvt.rn.f16x2.e2m1x2 %1, byte1;\n"
      "cvt.rn.f16x2.e2m1x2 %2, byte2;\n"
      "cvt.rn.f16x2.e2m1x2 %3, byte3;\n"
      "}\n"
      : "=r"(out_fp16[0]), "=r"(out_fp16[1]), "=r"(out_fp16[2]), "=r"(out_fp16[3])
      : "r"(packed));
  float2 f0 = __half22float2(reinterpret_cast<__half2&>(out_fp16[0]));
  float2 f1 = __half22float2(reinterpret_cast<__half2&>(out_fp16[1]));
  float2 f2 = __half22float2(reinterpret_cast<__half2&>(out_fp16[2]));
  float2 f3 = __half22float2(reinterpret_cast<__half2&>(out_fp16[3]));
  out[0] = f0.x; out[1] = f0.y;
  out[2] = f1.x; out[3] = f1.y;
  out[4] = f2.x; out[5] = f2.y;
  out[6] = f3.x; out[7] = f3.y;
}

__global__ void predecode_x_kernel(
    const uint8_t* __restrict__ x_packed,
    __nv_bfloat16* __restrict__ x_bf16,
    int tokens,
    int packed_hidden) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = tokens * packed_hidden;
  if (idx >= total) return;
  uint8_t byte = x_packed[idx];
  uint32_t out_fp16;
  asm volatile(
      "{\n"
      ".reg .b8 b;\n"
      "mov.b32 {b, _, _, _}, %1;\n"
      "cvt.rn.f16x2.e2m1x2 %0, b;\n"
      "}\n"
      : "=r"(out_fp16)
      : "r"((uint32_t)byte));
  float2 f = __half22float2(reinterpret_cast<__half2&>(out_fp16));
  int token = idx / packed_hidden;
  int packed_col = idx - token * packed_hidden;
  int h = packed_col * 2;
  long long base = static_cast<long long>(token) * kHidden + h;
  x_bf16[base] = __float2bfloat16(f.x);
  x_bf16[base + 1] = __float2bfloat16(f.y);
}

__global__ __launch_bounds__(kWarpsPerCta * kWarpSize)
void gate_up_predecoded_x_kernel(
    const __nv_bfloat16* __restrict__ x_bf16,
    const uint8_t* __restrict__ x_sf,
    const uint8_t* __restrict__ w13_packed,
    const uint8_t* __restrict__ w13_sf,
    const int32_t* __restrict__ token_selected_slots,
    float* __restrict__ scratch,
    int bucket_tokens,
    int top_k,
    int packed_hidden,
    int scale_columns,
    int intermediate_neurons,
    int weight_intermediate_stride) {
  int lane = threadIdx.x & (kWarpSize - 1);
  int warp_in_cta = threadIdx.x >> 5;
  long long warp_id = static_cast<long long>(blockIdx.x) * kWarpsPerCta + warp_in_cta;
  long long total = static_cast<long long>(bucket_tokens) * top_k * intermediate_neurons;
  if (warp_id >= total) return;

  int neuron = warp_id % intermediate_neurons;
  long long route_id = warp_id / intermediate_neurons;
  int routed = route_id % top_k;
  int token = route_id / top_k;
  int expert = token_selected_slots[token * top_k + routed];

  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  const __nv_bfloat16* x_row = x_bf16 + static_cast<long long>(token) * kHidden;
  const uint8_t* gate_row = w13_packed +
      (static_cast<long long>(expert) * weight_intermediate_stride + neuron) * packed_hidden;
  const uint8_t* up_row = w13_packed +
      (static_cast<long long>(expert) * weight_intermediate_stride + intermediate_neurons + neuron) * packed_hidden;

  for (int packed_col = lane * 4; packed_col + 3 < packed_hidden; packed_col += kWarpSize * 4) {
    uint32_t g4 = *reinterpret_cast<const uint32_t*>(gate_row + packed_col);
    uint32_t u4 = *reinterpret_cast<const uint32_t*>(up_row + packed_col);
    float gv[8];
    float uv[8];
    e2m1_uint32_to_float8(g4, gv);
    e2m1_uint32_to_float8(u4, uv);
    int scale_col = min((packed_col * 2) / 16, scale_columns - 1);
    uint8_t xs = x_sf[token * scale_columns + scale_col];
    uint8_t gs = w13_sf[(static_cast<long long>(expert) * weight_intermediate_stride + neuron) *
                            scale_columns + scale_col];
    uint8_t us = w13_sf[(static_cast<long long>(expert) * weight_intermediate_stride +
                         intermediate_neurons + neuron) * scale_columns + scale_col];
    float scale_probe = 1.0f + 0.0000152587890625f * static_cast<float>(xs ^ gs ^ us);
    int h = packed_col * 2;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      float xv = __bfloat162float(x_row[h + i]);
      gate_acc += xv * gv[i] * scale_probe;
      up_acc += xv * uv[i] * scale_probe;
    }
  }
  float gate = warp_sum(gate_acc);
  float up = warp_sum(up_acc);
  if (lane == 0) {
    long long out = (static_cast<long long>(token) * top_k + routed) * intermediate_neurons + neuron;
    scratch[out] = silu(gate) * up;
  }
}

void check_cuda(torch::Tensor tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}
} // namespace

void cursor_nvfp4_predecode_x_bf16(torch::Tensor x_packed, torch::Tensor x_bf16) {
  check_cuda(x_packed, "x_packed");
  check_cuda(x_bf16, "x_bf16");
  TORCH_CHECK(x_packed.scalar_type() == torch::kUInt8, "x_packed must be uint8");
  TORCH_CHECK(x_bf16.scalar_type() == torch::kBFloat16, "x_bf16 must be bf16");
  TORCH_CHECK(x_bf16.size(0) == x_packed.size(0), "token mismatch");
  TORCH_CHECK(x_bf16.size(1) == x_packed.size(1) * 2, "hidden mismatch");
  int total = x_packed.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  predecode_x_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
      x_packed.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_bfloat16*>(x_bf16.data_ptr<at::BFloat16>()),
      x_packed.size(0),
      x_packed.size(1));
}

void cursor_nvfp4_gate_up_predecoded_x_out(
    torch::Tensor x_bf16,
    torch::Tensor x_sf,
    torch::Tensor w13_packed,
    torch::Tensor w13_sf,
    torch::Tensor token_selected_slots,
    torch::Tensor scratch) {
  check_cuda(x_bf16, "x_bf16");
  check_cuda(x_sf, "x_sf");
  check_cuda(w13_packed, "w13_packed");
  check_cuda(w13_sf, "w13_sf");
  check_cuda(token_selected_slots, "token_selected_slots");
  check_cuda(scratch, "scratch");
  TORCH_CHECK(x_bf16.scalar_type() == torch::kBFloat16, "x_bf16 must be bf16");
  TORCH_CHECK(x_sf.scalar_type() == torch::kUInt8, "x_sf must be uint8");
  TORCH_CHECK(w13_packed.scalar_type() == torch::kUInt8, "w13_packed must be uint8");
  TORCH_CHECK(w13_sf.scalar_type() == torch::kUInt8, "w13_sf must be uint8");
  TORCH_CHECK(token_selected_slots.scalar_type() == torch::kInt32, "slots must be int32");
  TORCH_CHECK(scratch.scalar_type() == torch::kFloat32, "scratch must be float32");
  int bucket_tokens = x_bf16.size(0);
  int packed_hidden = x_bf16.size(1) / 2;
  int top_k = token_selected_slots.size(1);
  int scale_columns = x_sf.size(1);
  int intermediate_neurons = scratch.size(2);
  int weight_intermediate_stride = w13_packed.size(1);
  long long total_warps = static_cast<long long>(bucket_tokens) * top_k * intermediate_neurons;
  int blocks = static_cast<int>((total_warps + kWarpsPerCta - 1) / kWarpsPerCta);
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_predecoded_x_kernel<<<blocks, kWarpsPerCta * kWarpSize, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(x_bf16.data_ptr<at::BFloat16>()),
      x_sf.data_ptr<uint8_t>(),
      w13_packed.data_ptr<uint8_t>(),
      w13_sf.data_ptr<uint8_t>(),
      token_selected_slots.data_ptr<int32_t>(),
      scratch.data_ptr<float>(),
      bucket_tokens,
      top_k,
      packed_hidden,
      scale_columns,
      intermediate_neurons,
      weight_intermediate_stride);
}
"""


def time_ms(fn, warmup: int, iters: int) -> dict[str, float]:
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
        values.append(start.elapsed_time(end) / iters)
    return {"min_ms": min(values), "median_ms": statistics.median(values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="1,4,8,16,32")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--output", default="artifacts/warpdecode/cursor_gate_up_predecode_x_full2048.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    baseline_src = load_module(root / "warpdecode_cursor_nvfp4_gate_up_ptx_prototype.py", "gate_baseline")
    baseline = load_inline(
        name="warpdecode_cursor_nvfp4_gate_up_ptx_ext_predecode_compare",
        cpp_sources=baseline_src.CPP_SRC,
        cuda_sources=baseline_src.CUDA_SRC,
        functions=["cursor_nvfp4_gate_up_ptx_out"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-DNEURONS_PER_WARP=1", *BLACKWELL_FP4_GENCODE],
        verbose=False,
    )
    candidate = load_inline(
        name="warpdecode_cursor_nvfp4_gate_up_predecode_x_ext",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        functions=["cursor_nvfp4_predecode_x_bf16", "cursor_nvfp4_gate_up_predecoded_x_out"],
        extra_cuda_cflags=["-O3", "--use_fast_math", *BLACKWELL_FP4_GENCODE],
        verbose=False,
    )

    rows = []
    for tokens in [int(v) for v in args.tokens.split(",") if v.strip()]:
        torch.manual_seed(64000 + tokens)
        x = torch.randint(0, 256, (tokens, PACKED_HIDDEN), device="cuda", dtype=torch.uint8)
        x_sf = torch.randint(1, 255, (tokens, SCALE_COLS), device="cuda", dtype=torch.uint8)
        w13 = torch.randint(0, 256, (EXPERTS, 2 * INTERMEDIATE, PACKED_HIDDEN), device="cuda", dtype=torch.uint8)
        w13_sf = torch.randint(1, 255, (EXPERTS, 2 * INTERMEDIATE, SCALE_COLS), device="cuda", dtype=torch.uint8)
        token = torch.arange(tokens, device="cuda", dtype=torch.int32).view(tokens, 1)
        top = torch.arange(TOP_K, device="cuda", dtype=torch.int32).view(1, TOP_K)
        slots = (token + top * 16).remainder(EXPERTS).contiguous()
        ref_scratch = torch.empty((tokens, TOP_K, INTERMEDIATE), device="cuda", dtype=torch.float32)
        x_bf16 = torch.empty((tokens, HIDDEN), device="cuda", dtype=torch.bfloat16)
        cand_scratch = torch.empty_like(ref_scratch)

        def run_baseline():
            baseline.cursor_nvfp4_gate_up_ptx_out(x, x_sf, w13, w13_sf, slots, ref_scratch)
            return ref_scratch

        def run_candidate_full():
            candidate.cursor_nvfp4_predecode_x_bf16(x, x_bf16)
            candidate.cursor_nvfp4_gate_up_predecoded_x_out(x_bf16, x_sf, w13, w13_sf, slots, cand_scratch)
            return cand_scratch

        def run_candidate_gate_only():
            candidate.cursor_nvfp4_gate_up_predecoded_x_out(x_bf16, x_sf, w13, w13_sf, slots, cand_scratch)
            return cand_scratch

        candidate.cursor_nvfp4_predecode_x_bf16(x, x_bf16)
        run_baseline(); run_candidate_full(); torch.cuda.synchronize()
        ref = ref_scratch.float().reshape(1, -1)
        out = cand_scratch.float().reshape(1, -1)
        row = {
            "tokens": tokens,
            "candidate": "cursor_gate_up_predecode_x_bf16_then_direct_route_gateup",
            "uses_token_selected_slots": True,
            "uses_grouped_moe": False,
            "pads_expert_rows": False,
            "adds_graph_stable_activation_decode_buffer": True,
            "max_abs_vs_direct_ptx": float((out - ref).abs().max().item()),
            "cosine_vs_direct_ptx": float(torch.nn.functional.cosine_similarity(out, ref, dim=-1).item()),
        }
        row.update({"baseline_" + k: v for k, v in time_ms(run_baseline, args.warmup, args.iters).items()})
        row.update({"predecode_full_" + k: v for k, v in time_ms(run_candidate_full, args.warmup, args.iters).items()})
        row.update({"predecoded_gate_only_" + k: v for k, v in time_ms(run_candidate_gate_only, args.warmup, args.iters).items()})
        row["full_speedup_vs_direct_ptx_min"] = row["baseline_min_ms"] / row["predecode_full_min_ms"]
        row["gate_only_speedup_vs_direct_ptx_min"] = row["baseline_min_ms"] / row["predecoded_gate_only_min_ms"]
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        del x, x_sf, w13, w13_sf, slots, ref_scratch, x_bf16, cand_scratch
        torch.cuda.empty_cache()

    payload = {"candidate": "cursor_gate_up_predecode_x", "rows": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

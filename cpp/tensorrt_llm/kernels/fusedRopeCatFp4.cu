/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fusedRopeCatFp4.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cuda_bf16.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

// Constants (fixed for DSV3.2 FP4 indexer: head_dim=128, per-block-32 quant).
// Quantization stages mirror fusedCatFp4.cu exactly (the non-RoPE sibling); the
// only addition here is the in-register neox RoPE rotation of the pe slice
// before the amax reduction, fusing away the standalone flashinfer RoPE kernel
// and the q_pe/k_pe write-back + reload it would otherwise force.
constexpr int HEAD_DIM = 128;
constexpr int WARP_SIZE = 32;
constexpr int ELEMS_PER_THREAD = 4;
constexpr int ROWS_PER_BLOCK = 8;
constexpr float INV_FP4_E2M1_MAX = 1.0f / 6.0f;
constexpr float MIN_AMAX = 1.0e-12f;

union BF16x4
{
    int2 vec;
    __nv_bfloat162 bf16x2[2];
};

/// FP4 E2M1 quantize a single scaled value. Bit-identical to fusedCatFp4.cu.
__device__ __forceinline__ uint32_t quantizeFp4E2M1(float scaled)
{
    float ax = fminf(fabsf(scaled), 6.0f);
    uint32_t idx = static_cast<uint32_t>(
        (ax > 0.25f) + (ax > 0.75f) + (ax > 1.25f) + (ax > 1.75f) + (ax > 2.5f) + (ax > 3.5f) + (ax > 5.0f));
    uint32_t code = idx & 0x7u;
    uint32_t sign = (scaled < 0.0f && idx != 0u) ? 1u : 0u;
    return code | (sign << 3);
}

/// Fused neox-RoPE(pe) + cat(pe, nope) + per-block-32 FP4 E2M1 quantize.
///
/// Grid: (ceil(M / ROWS_PER_BLOCK),)   Block: (WARP_SIZE * ROWS_PER_BLOCK,) = 256.
/// One warp handles one 128-element output row.
///
/// neox RoPE on the pe slice (rope_dim = pe_dim, applied over the full pe span):
///   half = pe_dim / 2;  x1 = pe[i] (i in [0,half)),  x2 = pe[i + half]
///   out[i]        = x1*cos[i] - x2*sin[i]
///   out[i + half] = x2*cos[i] + x1*sin[i]
/// cos_sin layout matches flashinfer apply_rope_with_cos_sin_cache_inplace:
///   cos_sin[pos] is [pe_dim] with cos = first half, sin = second half.
/// The rotated result is rounded to BF16 (the prod path materializes BF16 q_pe
/// between RoPE and fused_cat_fp4), then quantized exactly as fusedCatFp4.
/// nope passes through unrotated. Output packing identical to fusedCatFp4.
__global__ __launch_bounds__(WARP_SIZE* ROWS_PER_BLOCK) void fusedRopeCatFp4Kernel(int8_t* __restrict__ packed_out,
    int32_t* __restrict__ scale_out, __nv_bfloat16 const* __restrict__ pe, __nv_bfloat16 const* __restrict__ nope,
    float const* __restrict__ cos_sin, int32_t const* __restrict__ pos, int32_t M, int32_t pe_dim, int32_t nope_dim,
    int32_t pe_row_stride, int32_t nope_row_stride, int32_t cos_sin_stride)
{
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * ROWS_PER_BLOCK + warp_in_block;

    if (row >= M)
    {
        return;
    }

    int const rope_half = pe_dim >> 1;
    int const pair_lane_off = rope_half / ELEMS_PER_THREAD;
    int const base = lane * ELEMS_PER_THREAD;

    float v0, v1, v2, v3;

    bool from_pe = (base < pe_dim);
    if (from_pe)
    {
        // ---- Load this lane's 4 pe elems, shuffle the neox-paired lane's 4, rotate ----
        __nv_bfloat16 const* pe_row = pe + static_cast<int64_t>(row) * pe_row_stride;
        BF16x4 self_l;
        self_l.vec = *reinterpret_cast<int2 const*>(pe_row + base);
        float2 s0 = __bfloat1622float2(self_l.bf16x2[0]);
        float2 s1 = __bfloat1622float2(self_l.bf16x2[1]);
        float a0 = s0.x, a1 = s0.y, a2 = s1.x, a3 = s1.y;

        bool lower = (base < rope_half); // lanes [0, pe_dim/8) hold x1, next hold x2
        int pair_lane = lower ? (lane + pair_lane_off) : (lane - pair_lane_off);

        // Only the pe lanes [0, pe_dim/ELEMS_PER_THREAD) take this branch (since
        // base < pe_dim iff lane < pe_dim/ELEMS_PER_THREAD) and are converged.
        // The nope lanes are in the else branch and MUST be excluded from the
        // shuffle mask: a full-warp 0xFFFFFFFF mask with the nope lanes absent
        // returns undefined values and corrupts the per-block amax below.
        unsigned const pe_mask = (1u << (pe_dim / ELEMS_PER_THREAD)) - 1u;
        float p0 = __shfl_sync(pe_mask, a0, pair_lane);
        float p1 = __shfl_sync(pe_mask, a1, pair_lane);
        float p2 = __shfl_sync(pe_mask, a2, pair_lane);
        float p3 = __shfl_sync(pe_mask, a3, pair_lane);

        // cos/sin index = position within rope_half (0..rope_half-1).
        float const* cs = cos_sin + static_cast<int64_t>(pos[row]) * cos_sin_stride;
        int ci = lower ? base : (base - rope_half);
        float c0 = cs[ci + 0], c1 = cs[ci + 1], c2 = cs[ci + 2], c3 = cs[ci + 3];
        float n0 = cs[rope_half + ci + 0], n1 = cs[rope_half + ci + 1], n2 = cs[rope_half + ci + 2],
              n3 = cs[rope_half + ci + 3];

        // lower: a = x1 (self), p = x2 (paired) -> out = x1*cos - x2*sin
        // upper: a = x2 (self), p = x1 (paired) -> out = x2*cos + x1*sin
        float sgn = lower ? -1.0f : 1.0f;
        v0 = a0 * c0 + sgn * p0 * n0;
        v1 = a1 * c1 + sgn * p1 * n1;
        v2 = a2 * c2 + sgn * p2 * n2;
        v3 = a3 * c3 + sgn * p3 * n3;

        // Round once to BF16 to mirror the prod path (RoPE writes BF16 q_pe,
        // fused_cat_fp4 re-reads it), so the FP4 codes match bit-for-bit.
        v0 = __bfloat162float(__float2bfloat16(v0));
        v1 = __bfloat162float(__float2bfloat16(v1));
        v2 = __bfloat162float(__float2bfloat16(v2));
        v3 = __bfloat162float(__float2bfloat16(v3));
    }
    else
    {
        __nv_bfloat16 const* nope_row = nope + static_cast<int64_t>(row) * nope_row_stride;
        int col = base - pe_dim;
        BF16x4 loaded;
        loaded.vec = *reinterpret_cast<int2 const*>(nope_row + col);
        float2 f0 = __bfloat1622float2(loaded.bf16x2[0]);
        float2 f1 = __bfloat1622float2(loaded.bf16x2[1]);
        v0 = f0.x;
        v1 = f0.y;
        v2 = f1.x;
        v3 = f1.y;
    }

    // ---- Per-block-32 amax (group of 8 lanes); all 32 lanes reconverged ----
    float local_max = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    float amax = local_max;
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 4));
    amax = fmaxf(amax, MIN_AMAX);

    // ---- UE8M0 scale via IEEE 754 bit manipulation ----
    float ratio = amax * INV_FP4_E2M1_MAX;
    uint32_t bits = __float_as_uint(ratio);
    uint32_t exp_bits = bits & 0x7F800000u;
    if ((bits & 0x007FFFFFu) != 0u)
    {
        exp_bits += 0x00800000u;
    }
    float scale = __uint_as_float(exp_bits);

    // ---- FP4 E2M1 quantize (IEEE div for bit-exact DeepGEMM parity) ----
    uint32_t c0 = quantizeFp4E2M1(v0 / scale);
    uint32_t c1 = quantizeFp4E2M1(v1 / scale);
    uint32_t c2 = quantizeFp4E2M1(v2 / scale);
    uint32_t c3 = quantizeFp4E2M1(v3 / scale);

    uint8_t byte0 = static_cast<uint8_t>(c0 | (c1 << 4));
    uint8_t byte1 = static_cast<uint8_t>(c2 | (c3 << 4));
    int base_out = row * (HEAD_DIM / 2) + lane * 2;
    packed_out[base_out + 0] = static_cast<int8_t>(byte0);
    packed_out[base_out + 1] = static_cast<int8_t>(byte1);

    // ---- Pack the 4 UE8M0 exponent bytes into one int32 (little-endian) ----
    uint32_t my_exp = (__float_as_uint(scale) >> 23) & 0xFFu;
    uint32_t b0 = __shfl_sync(0xFFFFFFFFu, my_exp, 0);
    uint32_t b1 = __shfl_sync(0xFFFFFFFFu, my_exp, 8);
    uint32_t b2 = __shfl_sync(0xFFFFFFFFu, my_exp, 16);
    uint32_t b3 = __shfl_sync(0xFFFFFFFFu, my_exp, 24);
    if (lane == 0)
    {
        uint32_t packed_scale = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        scale_out[row] = static_cast<int32_t>(packed_scale);
    }
}

} // anonymous namespace

void invokeFusedRopeCatFp4(int8_t* packed_out, int32_t* scale_out, __nv_bfloat16 const* pe, __nv_bfloat16 const* nope,
    float const* cos_sin, int32_t const* pos, int32_t M, int32_t pe_dim, int32_t nope_dim, int32_t head_dim,
    int32_t pe_row_stride, int32_t nope_row_stride, int32_t cos_sin_stride, cudaStream_t stream)
{
    if (M == 0)
    {
        return;
    }

    TLLM_CHECK_WITH_INFO(head_dim == HEAD_DIM, "fusedRopeCatFp4: head_dim must be 128, got %d", head_dim);
    TLLM_CHECK_WITH_INFO(pe_dim + nope_dim == head_dim, "fusedRopeCatFp4: pe_dim (%d) + nope_dim (%d) != head_dim (%d)",
        pe_dim, nope_dim, head_dim);
    TLLM_CHECK_WITH_INFO((pe_dim & 1) == 0, "fusedRopeCatFp4: pe_dim (%d) must be even for neox RoPE", pe_dim);
    TLLM_CHECK_WITH_INFO(pe_dim % ELEMS_PER_THREAD == 0,
        "fusedRopeCatFp4: pe_dim (%d) must be a multiple of %d for vectorized access", pe_dim, ELEMS_PER_THREAD);
    TLLM_CHECK_WITH_INFO((pe_dim / 2) % ELEMS_PER_THREAD == 0,
        "fusedRopeCatFp4: rope_half (%d) must be a multiple of %d so neox pairs map to whole lanes", pe_dim / 2,
        ELEMS_PER_THREAD);
    TLLM_CHECK_WITH_INFO(
        pe_row_stride >= pe_dim, "fusedRopeCatFp4: pe_row_stride (%d) must be >= pe_dim (%d)", pe_row_stride, pe_dim);
    TLLM_CHECK_WITH_INFO(nope_row_stride >= nope_dim, "fusedRopeCatFp4: nope_row_stride (%d) must be >= nope_dim (%d)",
        nope_row_stride, nope_dim);
    TLLM_CHECK_WITH_INFO(pe_row_stride % ELEMS_PER_THREAD == 0,
        "fusedRopeCatFp4: pe_row_stride (%d) must be a multiple of %d for aligned vectorized access", pe_row_stride,
        ELEMS_PER_THREAD);
    TLLM_CHECK_WITH_INFO(nope_row_stride % ELEMS_PER_THREAD == 0,
        "fusedRopeCatFp4: nope_row_stride (%d) must be a multiple of %d for aligned vectorized access", nope_row_stride,
        ELEMS_PER_THREAD);

    int num_blocks = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    dim3 grid(num_blocks);
    dim3 block(WARP_SIZE * ROWS_PER_BLOCK);

    fusedRopeCatFp4Kernel<<<grid, block, 0, stream>>>(packed_out, scale_out, pe, nope, cos_sin, pos, M, pe_dim,
        nope_dim, pe_row_stride, nope_row_stride, cos_sin_stride);

    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels

TRTLLM_NAMESPACE_END

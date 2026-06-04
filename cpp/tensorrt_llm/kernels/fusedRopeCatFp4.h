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

#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

/// Fused neox-RoPE(pe) + concat(pe, nope) + FP4 E2M1 per-block-32 quantization
/// for the DSA indexer Q/K projection.
///
/// Equivalent to: flashinfer apply_rope_with_cos_sin_cache_inplace on the pe
/// slice, followed by fusedCatFp4 on (rotated_pe, nope) — but as a single
/// kernel, removing the standalone RoPE launch and the BF16 q_pe/k_pe
/// write-back + reload between the two ops. Output is bit-identical to that
/// two-step path (the rotated value is rounded to BF16 before quantization,
/// matching prod which materializes BF16 q_pe in between).
///
/// neox RoPE (rope_dim == pe_dim, applied over the whole pe span):
///   half = pe_dim / 2;  x1 = pe[i], x2 = pe[i + half]  (i in [0, half))
///   out[i]        = x1*cos[i] - x2*sin[i]
///   out[i + half] = x2*cos[i] + x1*sin[i]
/// cos_sin layout matches flashinfer: cos_sin[pos] is [pe_dim] with cos in the
/// first half and sin in the second half.
///
/// @param packed_out      [M, head_dim/2] int8, two E2M1 codes per byte.
/// @param scale_out       [M, 1] int32, four UE8M0 exponents packed little-endian.
/// @param pe              [M, pe_dim] BF16, the rotary slice (gets rotated).
/// @param nope            [M, nope_dim] BF16, the non-rotary slice (pass-through).
/// @param cos_sin         [max_pos, pe_dim] float32; cos first half, sin second half.
/// @param pos             [M] int32, per-row position index into cos_sin.
/// @param M               Number of rows.
/// @param pe_dim          Rotary dim (== rope_dim). Even, multiple of 4, and
///                        pe_dim/2 a multiple of 4.
/// @param nope_dim        Non-rotary dim. pe_dim + nope_dim == head_dim == 128.
/// @param head_dim        Total head dim (must be 128).
/// @param pe_row_stride   Row stride (elements) of pe.
/// @param nope_row_stride Row stride (elements) of nope.
/// @param cos_sin_stride  Row stride (elements) of cos_sin (== pe_dim for contiguous).
/// @param stream          CUDA stream.
void invokeFusedRopeCatFp4(int8_t* packed_out, int32_t* scale_out, __nv_bfloat16 const* pe, __nv_bfloat16 const* nope,
    float const* cos_sin, int32_t const* pos, int32_t M, int32_t pe_dim, int32_t nope_dim, int32_t head_dim,
    int32_t pe_row_stride, int32_t nope_row_stride, int32_t cos_sin_stride, cudaStream_t stream = 0);

} // namespace kernels

TRTLLM_NAMESPACE_END

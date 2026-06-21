/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/fusedActivationQuant.h"
#include "tensorrt_llm/kernels/quantization.cuh"
#include "tensorrt_llm/kernels/quantization.h"

#include <cstdint>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

constexpr int kEltsPerThread = 8;

bool useTiledSmallM()
{
    static bool const enabled = []()
    {
        char const* const value = std::getenv("TRTLLM_FUSED_SIGMOID_MUL_QUANT_TILED_SMALL_M");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

__device__ __forceinline__ float relu2_f32(float x)
{
    float r = fmaxf(0.0f, x);
    return r * r;
}

template <typename T>
__device__ __forceinline__ typename TypeConverter<T>::Type make_packed_pair(float x0, float x1)
{
    if constexpr (std::is_same_v<T, half>)
    {
        return __floats2half2_rn(x0, x1);
    }
    else
    {
        return __floats2bfloat162_rn(x0, x1);
    }
}

template <typename T>
__device__ __forceinline__ typename TypeConverter<T>::Type gated_sigmoid_mul_pair(T x0, T gate0, T x1, T gate1)
{
    float s0 = __expf(-static_cast<float>(gate0));
    float s1 = __expf(-static_cast<float>(gate1));
    s0 = 1.0f / (1.0f + s0);
    s1 = 1.0f / (1.0f + s1);
    auto sPacked = make_packed_pair<T>(s0, s1);

    float2 sRounded;
    if constexpr (std::is_same_v<T, half>)
    {
        sRounded = __half22float2(sPacked);
    }
    else
    {
        sRounded = __bfloat1622float2(sPacked);
    }

    float y0 = static_cast<float>(x0) * sRounded.x;
    float y1 = static_cast<float>(x1) * sRounded.y;
    return make_packed_pair<T>(y0, y1);
}

// Fused relu2 + NVFP4 quantization kernel.
//
// To match the unfused path (PyTorch relu2 -> cvt_warp_fp16_to_fp4), relu2 is
// computed in f32 then rounded back to native precision (bf16/fp16) before
// quantization. Absmax and scale-factor math follow cvt_warp_fp16_to_fp4 exactly.
// Column padding to a multiple of (4 * kSfVecSize) matches quantize_with_block_size
// for the swizzled SF layout.
template <typename T>
__global__ void fusedRelu2QuantizeKernel(T const* __restrict__ input, float const* __restrict__ sfScale,
    uint32_t* __restrict__ outputFp4, uint32_t* __restrict__ outputSf, int m, int n)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    constexpr int kSfVecSize = 16;
    constexpr int kNumThreadsPerSf = kSfVecSize / kEltsPerThread;
    constexpr int kPackedPerThread = kEltsPerThread / 2;

    using PackedType = std::conditional_t<std::is_same_v<T, half>, __half2, __nv_bfloat162>;

    float const SFScaleVal = sfScale[0];
    int const numColThreads = n / kEltsPerThread;
    int const numColVecs = n / kSfVecSize;
    int const numColThreadsPadded = ((n + 4 * kSfVecSize - 1) / (4 * kSfVecSize)) * (4 * kSfVecSize) / kEltsPerThread;
    int const rowIdx = blockIdx.x;

    if (rowIdx >= m)
        return;

    for (int colIdx = threadIdx.x; colIdx < numColThreadsPadded; colIdx += blockDim.x)
    {
        bool const isValidCol = colIdx < numColThreads;
        PackedType packedVals[kPackedPerThread];

        if (isValidCol)
        {
            int const inputOffset = rowIdx * n + colIdx * kEltsPerThread;
#pragma unroll
            for (int i = 0; i < kPackedPerThread; i++)
            {
                float f0 = relu2_f32(static_cast<float>(input[inputOffset + i * 2]));
                float f1 = relu2_f32(static_cast<float>(input[inputOffset + i * 2 + 1]));
                if constexpr (std::is_same_v<T, half>)
                {
                    packedVals[i] = __floats2half2_rn(f0, f1);
                }
                else
                {
                    packedVals[i] = __floats2bfloat162_rn(f0, f1);
                }
            }
        }
        else
        {
#pragma unroll
            for (int i = 0; i < kPackedPerThread; i++)
            {
                if constexpr (std::is_same_v<T, half>)
                {
                    packedVals[i] = __float2half2_rn(0.0f);
                }
                else
                {
                    packedVals[i] = __float2bfloat162_rn(0.0f);
                }
            }
        }

        // Absmax in native precision, then reduce across the SF group (2 threads).
        auto localMax = cuda_abs(packedVals[0]);
#pragma unroll
        for (int i = 1; i < kPackedPerThread; i++)
        {
            localMax = cuda_max(localMax, cuda_abs(packedVals[i]));
        }
        localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
        float vecMax = float(cuda_max(localMax.x, localMax.y));

        // Scale-factor computation (identical to cvt_warp_fp16_to_fp4).
        float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
        __nv_fp8_e4m3 fp8SF = __nv_fp8_e4m3(SFValue);
        uint8_t fp8SFVal = fp8SF.__x;
        SFValue = static_cast<float>(fp8SF);

        float outputScale
            = vecMax != 0.0f ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal)) : 0.0f;

        if (colIdx % kNumThreadsPerSf == 0)
        {
            auto sfOutPtr = cvt_quant_get_sf_out_offset<uint32_t, kNumThreadsPerSf>(std::nullopt, rowIdx, colIdx,
                std::optional<int>(m), numColVecs, outputSf, QuantizationSFLayout::SWIZZLED);
            if (sfOutPtr != nullptr)
            {
                *sfOutPtr = fp8SFVal;
            }
        }

        if (isValidCol)
        {
            float2 fp2Vals[kPackedPerThread];
#pragma unroll
            for (int i = 0; i < kPackedPerThread; i++)
            {
                if constexpr (std::is_same_v<T, half>)
                {
                    fp2Vals[i] = __half22float2(packedVals[i]);
                }
                else
                {
                    fp2Vals[i] = __bfloat1622float2(packedVals[i]);
                }
                fp2Vals[i].x *= outputScale;
                fp2Vals[i].y *= outputScale;
            }

            outputFp4[rowIdx * numColThreads + colIdx] = fp32_vec_to_e2m1(fp2Vals);
        }
    }
#else
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("FP4 quantization requires SM100 (Blackwell) or later!\n");
    }
#endif
}

// Fused sigmoid-mul + NVFP4 quantization kernel.
//
// This matches the production two-kernel path:
//   fused_sigmoid_mul(input, gate) -> fp4_quantize(..., swizzled_layout=True)
//
// The activation rounding points mirror fused_sigmoid_mul: sigmoid is rounded
// to native precision before the product, then the product is rounded to native
// precision before the FP4 pack. FP4 packing itself is delegated to
// cvt_warp_fp16_to_fp4 so E2M1 conversion and SF math stay identical to the
// generic quantizer.
template <typename T>
__global__ void fusedSigmoidMulQuantizeKernel(T const* __restrict__ input, T const* __restrict__ gate,
    float const* __restrict__ sfScale, uint32_t* __restrict__ outputFp4, uint32_t* __restrict__ outputSf, int m, int n)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    constexpr int kSfVecSize = 16;
    constexpr int kEltsPerPackedThread = CVT_ELTS_PER_THREAD;
    constexpr int kPackedPerThread = kEltsPerPackedThread / 2;
    constexpr int kNumThreadsPerSf = kSfVecSize / kEltsPerPackedThread;

    static_assert(kEltsPerPackedThread == kEltsPerThread);

    float const SFScaleVal = sfScale[0];
    int const numColThreads = n / kEltsPerPackedThread;
    int const numColVecs = n / kSfVecSize;
    int const numColThreadsPadded
        = ((n + 4 * kSfVecSize - 1) / (4 * kSfVecSize)) * (4 * kSfVecSize) / kEltsPerPackedThread;
    int const rowIdx = blockIdx.x;
    bool const isValidRow = rowIdx < m;

    for (int colIdx = threadIdx.x; colIdx < numColThreadsPadded; colIdx += blockDim.x)
    {
        bool const isValidCol = colIdx < numColThreads;
        PackedVec<T> vec;

        if (isValidRow && isValidCol)
        {
            int const offset = rowIdx * n + colIdx * kEltsPerPackedThread;
#pragma unroll
            for (int i = 0; i < kPackedPerThread; i++)
            {
                vec.elts[i] = gated_sigmoid_mul_pair(
                    input[offset + i * 2], gate[offset + i * 2], input[offset + i * 2 + 1], gate[offset + i * 2 + 1]);
            }
        }
        else
        {
#pragma unroll
            for (int i = 0; i < kPackedPerThread; i++)
            {
                vec.elts[i] = make_packed_pair<T>(0.0f, 0.0f);
            }
        }

        auto sfOutPtr = cvt_quant_get_sf_out_offset<uint32_t, kNumThreadsPerSf>(
            std::nullopt, rowIdx, colIdx, std::optional<int>(m), numColVecs, outputSf, QuantizationSFLayout::SWIZZLED);
        uint32_t e2m1Vec = cvt_warp_fp16_to_fp4<T, kSfVecSize, false>(vec, SFScaleVal, sfOutPtr);

        if (isValidRow && isValidCol)
        {
            outputFp4[rowIdx * numColThreads + colIdx] = e2m1Vec;
        }
    }
#else
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("FP4 quantization requires SM100 (Blackwell) or later!\n");
    }
#endif
}

template <typename T>
__global__ void fusedSigmoidMulQuantizeTiledKernel(T const* __restrict__ input, T const* __restrict__ gate,
    float const* __restrict__ sfScale, uint32_t* __restrict__ outputFp4, uint32_t* __restrict__ outputSf, int m, int n)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    constexpr int kSfVecSize = 16;
    constexpr int kEltsPerPackedThread = CVT_ELTS_PER_THREAD;
    constexpr int kPackedPerThread = kEltsPerPackedThread / 2;
    constexpr int kNumThreadsPerSf = kSfVecSize / kEltsPerPackedThread;

    static_assert(kEltsPerPackedThread == kEltsPerThread);

    float const SFScaleVal = sfScale[0];
    int const numColThreads = n / kEltsPerPackedThread;
    int const numColVecs = n / kSfVecSize;
    int const numColThreadsPadded
        = ((n + 4 * kSfVecSize - 1) / (4 * kSfVecSize)) * (4 * kSfVecSize) / kEltsPerPackedThread;
    int const rowIdx = blockIdx.x;
    int const colIdx = blockIdx.y * blockDim.x + threadIdx.x;

    if (rowIdx >= m || colIdx >= numColThreadsPadded)
    {
        return;
    }

    bool const isValidCol = colIdx < numColThreads;
    PackedVec<T> vec;

    if (isValidCol)
    {
        int const offset = rowIdx * n + colIdx * kEltsPerPackedThread;
#pragma unroll
        for (int i = 0; i < kPackedPerThread; i++)
        {
            vec.elts[i] = gated_sigmoid_mul_pair(
                input[offset + i * 2], gate[offset + i * 2], input[offset + i * 2 + 1], gate[offset + i * 2 + 1]);
        }
    }
    else
    {
#pragma unroll
        for (int i = 0; i < kPackedPerThread; i++)
        {
            vec.elts[i] = make_packed_pair<T>(0.0f, 0.0f);
        }
    }

    auto sfOutPtr = cvt_quant_get_sf_out_offset<uint32_t, kNumThreadsPerSf>(
        std::nullopt, rowIdx, colIdx, std::optional<int>(m), numColVecs, outputSf, QuantizationSFLayout::SWIZZLED);
    uint32_t e2m1Vec = cvt_warp_fp16_to_fp4<T, kSfVecSize, false>(vec, SFScaleVal, sfOutPtr);

    if (isValidCol)
    {
        outputFp4[rowIdx * numColThreads + colIdx] = e2m1Vec;
    }
#else
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("FP4 quantization requires SM100 (Blackwell) or later!\n");
    }
#endif
}

template <typename T>
void invokeFusedRelu2Quantize(T const* input, float const* sfScale, std::uint8_t* outputFp4, std::uint8_t* outputSf,
    int m, int n, int sfVecSize, cudaStream_t stream)
{
    constexpr int kSfVecSize = 16;
    int const numColThreadsPadded = ((n + 4 * kSfVecSize - 1) / (4 * kSfVecSize)) * (4 * kSfVecSize) / kEltsPerThread;
    int threadsPerBlock = min(512, numColThreadsPadded);
    threadsPerBlock = max(32, ((threadsPerBlock + 31) / 32) * 32);

    fusedRelu2QuantizeKernel<T><<<m, threadsPerBlock, 0, stream>>>(
        input, sfScale, reinterpret_cast<uint32_t*>(outputFp4), reinterpret_cast<uint32_t*>(outputSf), m, n);
}

template <typename T>
void invokeFusedSigmoidMulQuantize(T const* input, T const* gate, float const* sfScale, std::uint8_t* outputFp4,
    std::uint8_t* outputSf, int m, int n, int sfVecSize, cudaStream_t stream)
{
    constexpr int kSfVecSize = 16;
    int const numColThreadsPadded = ((n + 4 * kSfVecSize - 1) / (4 * kSfVecSize)) * (4 * kSfVecSize) / kEltsPerThread;
    int threadsPerBlock = min(512, numColThreadsPadded);
    threadsPerBlock = max(32, ((threadsPerBlock + 31) / 32) * 32);
    int const numPaddedRowsForSf = PadUpFn(m, 128);

    if (m < 128 && useTiledSmallM())
    {
        int const numColTiles = (numColThreadsPadded + threadsPerBlock - 1) / threadsPerBlock;
        dim3 const grid(m, numColTiles);
        fusedSigmoidMulQuantizeTiledKernel<T><<<grid, threadsPerBlock, 0, stream>>>(
            input, gate, sfScale, reinterpret_cast<uint32_t*>(outputFp4), reinterpret_cast<uint32_t*>(outputSf), m, n);
    }
    else
    {
        fusedSigmoidMulQuantizeKernel<T><<<numPaddedRowsForSf, threadsPerBlock, 0, stream>>>(
            input, gate, sfScale, reinterpret_cast<uint32_t*>(outputFp4), reinterpret_cast<uint32_t*>(outputSf), m, n);
    }
}

template void invokeFusedRelu2Quantize<half>(
    half const*, float const*, std::uint8_t*, std::uint8_t*, int, int, int, cudaStream_t);

#ifdef ENABLE_BF16
template void invokeFusedRelu2Quantize<__nv_bfloat16>(
    __nv_bfloat16 const*, float const*, std::uint8_t*, std::uint8_t*, int, int, int, cudaStream_t);
#endif

template void invokeFusedSigmoidMulQuantize<half>(
    half const*, half const*, float const*, std::uint8_t*, std::uint8_t*, int, int, int, cudaStream_t);

#ifdef ENABLE_BF16
template void invokeFusedSigmoidMulQuantize<__nv_bfloat16>(__nv_bfloat16 const*, __nv_bfloat16 const*, float const*,
    std::uint8_t*, std::uint8_t*, int, int, int, cudaStream_t);
#endif

} // namespace kernels

TRTLLM_NAMESPACE_END

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

#include "DeepseekResidentNorm.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

template <typename T>
__device__ __forceinline__ float residentNormToFloat(T value);

template <>
__device__ __forceinline__ float residentNormToFloat<half>(half value)
{
    return __half2float(value);
}

template <>
__device__ __forceinline__ float residentNormToFloat<__nv_bfloat16>(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T residentNormFromFloat(float value);

template <>
__device__ __forceinline__ half residentNormFromFloat<half>(float value)
{
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 residentNormFromFloat<__nv_bfloat16>(float value)
{
    return __float2bfloat16(value);
}

template <int32_t kThreads>
__device__ __forceinline__ float residentNormBlockSum(float value)
{
    __shared__ float shared[kThreads];
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    shared[tid] = value;
    __syncthreads();

    for (int32_t stride = kThreads / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

template <typename T, int32_t kThreads>
__global__ void deepseekResidentRmsNormKernel(T* __restrict__ output, int64_t outputStride0,
    T const* __restrict__ input, int64_t inputStride0, T const* __restrict__ weight, int64_t weightStride0,
    int32_t hiddenSize, float eps)
{
    int32_t const tokenIdx = static_cast<int32_t>(blockIdx.x);
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    T const* inputRow = input + static_cast<int64_t>(tokenIdx) * inputStride0;
    T* outputRow = output + static_cast<int64_t>(tokenIdx) * outputStride0;

    float localSum = 0.0F;
    for (int32_t hiddenIdx = tid; hiddenIdx < hiddenSize; hiddenIdx += kThreads)
    {
        float const value = residentNormToFloat(inputRow[hiddenIdx]);
        localSum += value * value;
    }
    float const sum = residentNormBlockSum<kThreads>(localSum);
    float const invRms = rsqrtf(sum / static_cast<float>(hiddenSize) + eps);

    for (int32_t hiddenIdx = tid; hiddenIdx < hiddenSize; hiddenIdx += kThreads)
    {
        float const value = residentNormToFloat(inputRow[hiddenIdx]);
        float const gamma = residentNormToFloat(weight[static_cast<int64_t>(hiddenIdx) * weightStride0]);
        outputRow[hiddenIdx] = residentNormFromFloat<T>(value * invRms * gamma);
    }
}

template <typename T, int32_t kThreads>
__global__ void deepseekResidentDsaKvASplitNormPackKernel(T const* __restrict__ kvA, int64_t kvAStride0,
    T* __restrict__ qLora, int64_t qLoraStride0, T* __restrict__ compressedKv, int64_t compressedKvStride0,
    T* __restrict__ kPe, int64_t kPeStride0, T* __restrict__ latentCache, int64_t latentCacheStride0,
    T const* __restrict__ qWeight, int64_t qWeightStride0, T const* __restrict__ kvWeight, int64_t kvWeightStride0,
    int32_t qLoraRank, int32_t kvLoraRank, int32_t ropeDim, float eps)
{
    int32_t const tokenIdx = static_cast<int32_t>(blockIdx.x);
    int32_t const tid = static_cast<int32_t>(threadIdx.x);

    T const* kvARow = kvA + static_cast<int64_t>(tokenIdx) * kvAStride0;
    T* qLoraRow = qLora + static_cast<int64_t>(tokenIdx) * qLoraStride0;
    T* compressedKvRow = compressedKv + static_cast<int64_t>(tokenIdx) * compressedKvStride0;
    T* kPeRow = kPe + static_cast<int64_t>(tokenIdx) * kPeStride0;
    T* latentRow = latentCache + static_cast<int64_t>(tokenIdx) * latentCacheStride0;

    int32_t const kvOffset = qLoraRank;
    int32_t const ropeOffset = qLoraRank + kvLoraRank;

    float qLocalSum = 0.0F;
    for (int32_t hiddenIdx = tid; hiddenIdx < qLoraRank; hiddenIdx += kThreads)
    {
        float const value = residentNormToFloat(kvARow[hiddenIdx]);
        qLocalSum += value * value;
    }
    float const qSum = residentNormBlockSum<kThreads>(qLocalSum);
    float const qInvRms = rsqrtf(qSum / static_cast<float>(qLoraRank) + eps);

    float kvLocalSum = 0.0F;
    for (int32_t hiddenIdx = tid; hiddenIdx < kvLoraRank; hiddenIdx += kThreads)
    {
        float const value = residentNormToFloat(kvARow[kvOffset + hiddenIdx]);
        kvLocalSum += value * value;
    }
    float const kvSum = residentNormBlockSum<kThreads>(kvLocalSum);
    float const kvInvRms = rsqrtf(kvSum / static_cast<float>(kvLoraRank) + eps);

    for (int32_t hiddenIdx = tid; hiddenIdx < qLoraRank; hiddenIdx += kThreads)
    {
        float const value = residentNormToFloat(kvARow[hiddenIdx]);
        float const gamma = residentNormToFloat(qWeight[static_cast<int64_t>(hiddenIdx) * qWeightStride0]);
        qLoraRow[hiddenIdx] = residentNormFromFloat<T>(value * qInvRms * gamma);
    }

    for (int32_t hiddenIdx = tid; hiddenIdx < kvLoraRank; hiddenIdx += kThreads)
    {
        float const value = residentNormToFloat(kvARow[kvOffset + hiddenIdx]);
        float const gamma = residentNormToFloat(kvWeight[static_cast<int64_t>(hiddenIdx) * kvWeightStride0]);
        T const normalized = residentNormFromFloat<T>(value * kvInvRms * gamma);
        compressedKvRow[hiddenIdx] = normalized;
        latentRow[hiddenIdx] = normalized;
    }

    for (int32_t hiddenIdx = tid; hiddenIdx < ropeDim; hiddenIdx += kThreads)
    {
        T const value = kvARow[ropeOffset + hiddenIdx];
        kPeRow[hiddenIdx] = value;
        latentRow[kvLoraRank + hiddenIdx] = value;
    }
}

template <typename T, int32_t kThreads>
__global__ void deepseekResidentAddRmsNormKernel(T* __restrict__ normOutput, int64_t normOutputStride0,
    T* __restrict__ residualOutput, int64_t residualOutputStride0, T const* __restrict__ input, int64_t inputStride0,
    T const* __restrict__ residual, int64_t residualStride0, T const* __restrict__ weight, int64_t weightStride0,
    int32_t hiddenSize, float eps)
{
    int32_t const tokenIdx = static_cast<int32_t>(blockIdx.x);
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    T const* inputRow = input + static_cast<int64_t>(tokenIdx) * inputStride0;
    T const* residualRow = residual + static_cast<int64_t>(tokenIdx) * residualStride0;
    T* residualOutputRow = residualOutput + static_cast<int64_t>(tokenIdx) * residualOutputStride0;
    T* normOutputRow = normOutput + static_cast<int64_t>(tokenIdx) * normOutputStride0;

    float localSum = 0.0F;
    for (int32_t hiddenIdx = tid; hiddenIdx < hiddenSize; hiddenIdx += kThreads)
    {
        float const value = residentNormToFloat(inputRow[hiddenIdx]) + residentNormToFloat(residualRow[hiddenIdx]);
        localSum += value * value;
    }
    float const sum = residentNormBlockSum<kThreads>(localSum);
    float const invRms = rsqrtf(sum / static_cast<float>(hiddenSize) + eps);

    for (int32_t hiddenIdx = tid; hiddenIdx < hiddenSize; hiddenIdx += kThreads)
    {
        float const value = residentNormToFloat(inputRow[hiddenIdx]) + residentNormToFloat(residualRow[hiddenIdx]);
        float const gamma = residentNormToFloat(weight[static_cast<int64_t>(hiddenIdx) * weightStride0]);
        residualOutputRow[hiddenIdx] = residentNormFromFloat<T>(value);
        normOutputRow[hiddenIdx] = residentNormFromFloat<T>(value * invRms * gamma);
    }
}

template <typename T, int32_t kThreads, int32_t kMaxRank>
__global__ void deepseekResidentAddRmsNormLowRankGateKernel(T* __restrict__ gatedOutput,
    int64_t gatedOutputStride0, T* __restrict__ residualOutput, int64_t residualOutputStride0,
    T const* __restrict__ input, int64_t inputStride0, T const* __restrict__ residual, int64_t residualStride0,
    T const* __restrict__ normWeight, int64_t normWeightStride0, T const* __restrict__ downWeight,
    int64_t downWeightStride0, int64_t downWeightStride1, T const* __restrict__ upWeight, int64_t upWeightStride0,
    int64_t upWeightStride1, int32_t hiddenSize, int32_t rank, float eps, bool useGemma)
{
    extern __shared__ __align__(sizeof(float)) unsigned char sharedBytes[];
    T* normShared = reinterpret_cast<T*>(sharedBytes);
    size_t const gateOffset = ((static_cast<size_t>(hiddenSize) * sizeof(T) + sizeof(float) - 1U) / sizeof(float))
        * sizeof(float);
    float* gate = reinterpret_cast<float*>(sharedBytes + gateOffset);

    int32_t const tokenIdx = static_cast<int32_t>(blockIdx.x);
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    T const* inputRow = input + static_cast<int64_t>(tokenIdx) * inputStride0;
    T const* residualRow = residual + static_cast<int64_t>(tokenIdx) * residualStride0;
    T* residualOutputRow = residualOutput + static_cast<int64_t>(tokenIdx) * residualOutputStride0;
    T* gatedOutputRow = gatedOutput + static_cast<int64_t>(tokenIdx) * gatedOutputStride0;

    float localSum = 0.0F;
    for (int32_t hiddenIdx = tid; hiddenIdx < hiddenSize; hiddenIdx += kThreads)
    {
        float const value = residentNormToFloat(inputRow[hiddenIdx]) + residentNormToFloat(residualRow[hiddenIdx]);
        localSum += value * value;
    }
    float const sum = residentNormBlockSum<kThreads>(localSum);
    float const invRms = rsqrtf(sum / static_cast<float>(hiddenSize) + eps);

    for (int32_t hiddenIdx = tid; hiddenIdx < hiddenSize; hiddenIdx += kThreads)
    {
        float const value = residentNormToFloat(inputRow[hiddenIdx]) + residentNormToFloat(residualRow[hiddenIdx]);
        float gamma = residentNormToFloat(normWeight[static_cast<int64_t>(hiddenIdx) * normWeightStride0]);
        if (useGemma)
        {
            gamma += 1.0F;
        }
        residualOutputRow[hiddenIdx] = residentNormFromFloat<T>(value);
        normShared[hiddenIdx] = residentNormFromFloat<T>(value * invRms * gamma);
    }
    __syncthreads();

    for (int32_t rankIdx = 0; rankIdx < rank; ++rankIdx)
    {
        float localDot = 0.0F;
        T const* downWeightRow = downWeight + static_cast<int64_t>(rankIdx) * downWeightStride0;
        for (int32_t hiddenIdx = tid; hiddenIdx < hiddenSize; hiddenIdx += kThreads)
        {
            float const inputValue = residentNormToFloat(normShared[hiddenIdx]);
            float const weightValue
                = residentNormToFloat(downWeightRow[static_cast<int64_t>(hiddenIdx) * downWeightStride1]);
            localDot += inputValue * weightValue;
        }
        float const dot = residentNormBlockSum<kThreads>(localDot);
        if (tid == 0)
        {
            float const silu = dot / (1.0F + __expf(-dot));
            gate[rankIdx] = residentNormToFloat(residentNormFromFloat<T>(silu));
        }
        __syncthreads();
    }

    for (int32_t hiddenIdx = tid; hiddenIdx < hiddenSize; hiddenIdx += kThreads)
    {
        float dot = 0.0F;
        T const* upWeightRow = upWeight + static_cast<int64_t>(hiddenIdx) * upWeightStride0;
        for (int32_t rankIdx = 0; rankIdx < rank; ++rankIdx)
        {
            float const weightValue = residentNormToFloat(upWeightRow[static_cast<int64_t>(rankIdx) * upWeightStride1]);
            dot += gate[rankIdx] * weightValue;
        }
        float const roundedDot = residentNormToFloat(residentNormFromFloat<T>(dot));
        float const sigmoid = 1.0F / (1.0F + __expf(-roundedDot));
        float const roundedSigmoid = residentNormToFloat(residentNormFromFloat<T>(sigmoid));
        float const inputValue = residentNormToFloat(normShared[hiddenIdx]);
        gatedOutputRow[hiddenIdx] = residentNormFromFloat<T>(inputValue * roundedSigmoid);
    }
}

template <typename T>
__global__ void deepseekResidentSigmoidMulKernel(T* __restrict__ output, int64_t outputStride0,
    T const* __restrict__ input, int64_t inputStride0, float const* __restrict__ gate, int64_t gateStride0,
    int32_t inputTokens, int32_t hiddenSize)
{
    int64_t const totalElements = static_cast<int64_t>(inputTokens) * static_cast<int64_t>(hiddenSize);
    int64_t const linearIdx
        = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    if (linearIdx >= totalElements)
    {
        return;
    }

    int32_t const tokenIdx = static_cast<int32_t>(linearIdx / hiddenSize);
    int32_t const hiddenIdx = static_cast<int32_t>(linearIdx - static_cast<int64_t>(tokenIdx) * hiddenSize);
    float const inputValue = residentNormToFloat(input[static_cast<int64_t>(tokenIdx) * inputStride0 + hiddenIdx]);
    float const gateValue = gate[static_cast<int64_t>(tokenIdx) * gateStride0 + hiddenIdx];
    float const sigmoid = 1.0F / (1.0F + __expf(-gateValue));
    output[static_cast<int64_t>(tokenIdx) * outputStride0 + hiddenIdx] = residentNormFromFloat<T>(inputValue * sigmoid);
}

template <typename T>
__global__ void deepseekResidentSwiGluFloatToOutputKernel(T* __restrict__ output, int64_t outputStride0,
    float const* __restrict__ gate, int64_t gateStride0, float const* __restrict__ up, int64_t upStride0,
    int32_t inputTokens, int32_t hiddenSize)
{
    int64_t const totalElements = static_cast<int64_t>(inputTokens) * static_cast<int64_t>(hiddenSize);
    int64_t const linearIdx
        = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    if (linearIdx >= totalElements)
    {
        return;
    }

    int32_t const tokenIdx = static_cast<int32_t>(linearIdx / hiddenSize);
    int32_t const hiddenIdx = static_cast<int32_t>(linearIdx - static_cast<int64_t>(tokenIdx) * hiddenSize);
    float const gateValue = gate[static_cast<int64_t>(tokenIdx) * gateStride0 + hiddenIdx];
    float const upValue = up[static_cast<int64_t>(tokenIdx) * upStride0 + hiddenIdx];
    float const sigmoid = 1.0F / (1.0F + __expf(-gateValue));
    output[static_cast<int64_t>(tokenIdx) * outputStride0 + hiddenIdx]
        = residentNormFromFloat<T>((gateValue * sigmoid) * upValue);
}

constexpr int32_t kMaxLowRankGateRank = 64;

template <typename T, int32_t kThreads, int32_t kMaxRank>
__global__ void deepseekResidentLowRankGateKernel(T* __restrict__ output, int64_t outputStride0,
    T const* __restrict__ input, int64_t inputStride0, T const* __restrict__ downWeight, int64_t downWeightStride0,
    int64_t downWeightStride1, T const* __restrict__ upWeight, int64_t upWeightStride0, int64_t upWeightStride1,
    int32_t hiddenSize, int32_t rank)
{
    __shared__ float gate[kMaxRank];
    int32_t const tokenIdx = static_cast<int32_t>(blockIdx.x);
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    T const* inputRow = input + static_cast<int64_t>(tokenIdx) * inputStride0;
    T* outputRow = output + static_cast<int64_t>(tokenIdx) * outputStride0;

    for (int32_t rankIdx = 0; rankIdx < rank; ++rankIdx)
    {
        float localSum = 0.0F;
        T const* downWeightRow = downWeight + static_cast<int64_t>(rankIdx) * downWeightStride0;
        for (int32_t hiddenIdx = tid; hiddenIdx < hiddenSize; hiddenIdx += kThreads)
        {
            float const inputValue = residentNormToFloat(inputRow[hiddenIdx]);
            float const weightValue
                = residentNormToFloat(downWeightRow[static_cast<int64_t>(hiddenIdx) * downWeightStride1]);
            localSum += inputValue * weightValue;
        }
        float const sum = residentNormBlockSum<kThreads>(localSum);
        if (tid == 0)
        {
            float const silu = sum / (1.0F + __expf(-sum));
            gate[rankIdx] = residentNormToFloat(residentNormFromFloat<T>(silu));
        }
        __syncthreads();
    }

    for (int32_t hiddenIdx = tid; hiddenIdx < hiddenSize; hiddenIdx += kThreads)
    {
        float dot = 0.0F;
        T const* upWeightRow = upWeight + static_cast<int64_t>(hiddenIdx) * upWeightStride0;
        for (int32_t rankIdx = 0; rankIdx < rank; ++rankIdx)
        {
            float const weightValue = residentNormToFloat(upWeightRow[static_cast<int64_t>(rankIdx) * upWeightStride1]);
            dot += gate[rankIdx] * weightValue;
        }
        float const roundedDot = residentNormToFloat(residentNormFromFloat<T>(dot));
        float const sigmoid = 1.0F / (1.0F + __expf(-roundedDot));
        float const roundedSigmoid = residentNormToFloat(residentNormFromFloat<T>(sigmoid));
        float const inputValue = residentNormToFloat(inputRow[hiddenIdx]);
        outputRow[hiddenIdx] = residentNormFromFloat<T>(inputValue * roundedSigmoid);
    }
}

template <typename T>
__global__ void deepseekResidentAddScaledFloatToOutputKernel(T* __restrict__ output, int64_t outputStride0,
    float const* __restrict__ addend, int64_t addendStride0, int32_t inputTokens, int32_t hiddenSize, float addendScale)
{
    int64_t const totalElements = static_cast<int64_t>(inputTokens) * static_cast<int64_t>(hiddenSize);
    int64_t const linearIdx
        = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    if (linearIdx >= totalElements)
    {
        return;
    }

    int32_t const tokenIdx = static_cast<int32_t>(linearIdx / hiddenSize);
    int32_t const hiddenIdx = static_cast<int32_t>(linearIdx - static_cast<int64_t>(tokenIdx) * hiddenSize);
    T* outputRow = output + static_cast<int64_t>(tokenIdx) * outputStride0;
    float const outputValue = residentNormToFloat(outputRow[hiddenIdx]);
    float const addendValue = addend[static_cast<int64_t>(tokenIdx) * addendStride0 + hiddenIdx];
    outputRow[hiddenIdx] = residentNormFromFloat<T>(outputValue + addendValue * addendScale);
}

template <typename T>
void launchDeepseekResidentRmsNorm(void* output, int64_t outputStride0, void const* input, int64_t inputStride0,
    void const* weight, int64_t weightStride0, int32_t inputTokens, int32_t hiddenSize, float eps, cudaStream_t stream)
{
    constexpr int32_t kThreads = 256;
    deepseekResidentRmsNormKernel<T, kThreads><<<inputTokens, kThreads, 0, stream>>>(reinterpret_cast<T*>(output),
        outputStride0, reinterpret_cast<T const*>(input), inputStride0, reinterpret_cast<T const*>(weight),
        weightStride0, hiddenSize, eps);
}

template <typename T>
void launchDeepseekResidentDsaKvASplitNormPack(void const* kvA, int64_t kvAStride0, void* qLora, int64_t qLoraStride0,
    void* compressedKv, int64_t compressedKvStride0, void* kPe, int64_t kPeStride0, void* latentCache,
    int64_t latentCacheStride0, void const* qWeight, int64_t qWeightStride0, void const* kvWeight,
    int64_t kvWeightStride0, int32_t inputTokens, int32_t qLoraRank, int32_t kvLoraRank, int32_t ropeDim, float eps,
    cudaStream_t stream)
{
    constexpr int32_t kThreads = 256;
    deepseekResidentDsaKvASplitNormPackKernel<T, kThreads>
        <<<inputTokens, kThreads, 0, stream>>>(reinterpret_cast<T const*>(kvA), kvAStride0, reinterpret_cast<T*>(qLora),
            qLoraStride0, reinterpret_cast<T*>(compressedKv), compressedKvStride0, reinterpret_cast<T*>(kPe),
            kPeStride0, reinterpret_cast<T*>(latentCache), latentCacheStride0, reinterpret_cast<T const*>(qWeight),
            qWeightStride0, reinterpret_cast<T const*>(kvWeight), kvWeightStride0, qLoraRank, kvLoraRank, ropeDim, eps);
}

template <typename T>
void launchDeepseekResidentAddRmsNorm(void* normOutput, int64_t normOutputStride0, void* residualOutput,
    int64_t residualOutputStride0, void const* input, int64_t inputStride0, void const* residual,
    int64_t residualStride0, void const* weight, int64_t weightStride0, int32_t inputTokens, int32_t hiddenSize,
    float eps, cudaStream_t stream)
{
    constexpr int32_t kThreads = 256;
    deepseekResidentAddRmsNormKernel<T, kThreads><<<inputTokens, kThreads, 0, stream>>>(
        reinterpret_cast<T*>(normOutput), normOutputStride0, reinterpret_cast<T*>(residualOutput),
        residualOutputStride0, reinterpret_cast<T const*>(input), inputStride0, reinterpret_cast<T const*>(residual),
        residualStride0, reinterpret_cast<T const*>(weight), weightStride0, hiddenSize, eps);
}

template <typename T>
void launchDeepseekResidentAddRmsNormLowRankGate(void* gatedOutput, int64_t gatedOutputStride0,
    void* residualOutput, int64_t residualOutputStride0, void const* input, int64_t inputStride0,
    void const* residual, int64_t residualStride0, void const* normWeight, int64_t normWeightStride0,
    void const* downWeight, int64_t downWeightStride0, int64_t downWeightStride1, void const* upWeight,
    int64_t upWeightStride0, int64_t upWeightStride1, int32_t inputTokens, int32_t hiddenSize, int32_t rank,
    float eps, bool useGemma, cudaStream_t stream)
{
    constexpr int32_t kThreads = 256;
    size_t const normSharedBytes = static_cast<size_t>(hiddenSize) * sizeof(T);
    size_t const gateOffset = ((normSharedBytes + sizeof(float) - 1U) / sizeof(float)) * sizeof(float);
    size_t const sharedBytes = gateOffset + static_cast<size_t>(rank) * sizeof(float);
    deepseekResidentAddRmsNormLowRankGateKernel<T, kThreads, kMaxLowRankGateRank>
        <<<inputTokens, kThreads, sharedBytes, stream>>>(reinterpret_cast<T*>(gatedOutput), gatedOutputStride0,
            reinterpret_cast<T*>(residualOutput), residualOutputStride0, reinterpret_cast<T const*>(input),
            inputStride0, reinterpret_cast<T const*>(residual), residualStride0,
            reinterpret_cast<T const*>(normWeight), normWeightStride0, reinterpret_cast<T const*>(downWeight),
            downWeightStride0, downWeightStride1, reinterpret_cast<T const*>(upWeight), upWeightStride0,
            upWeightStride1, hiddenSize, rank, eps, useGemma);
}

template <typename T>
void launchDeepseekResidentSigmoidMul(void* output, int64_t outputStride0, void const* input, int64_t inputStride0,
    float const* gate, int64_t gateStride0, int32_t inputTokens, int32_t hiddenSize, cudaStream_t stream)
{
    constexpr int32_t kThreads = 256;
    int64_t const totalElements = static_cast<int64_t>(inputTokens) * static_cast<int64_t>(hiddenSize);
    int32_t const blocks = static_cast<int32_t>((totalElements + kThreads - 1) / kThreads);
    deepseekResidentSigmoidMulKernel<T><<<blocks, kThreads, 0, stream>>>(reinterpret_cast<T*>(output), outputStride0,
        reinterpret_cast<T const*>(input), inputStride0, gate, gateStride0, inputTokens, hiddenSize);
}

template <typename T>
void launchDeepseekResidentSwiGluFloatToOutput(void* output, int64_t outputStride0, float const* gate,
    int64_t gateStride0, float const* up, int64_t upStride0, int32_t inputTokens, int32_t hiddenSize,
    cudaStream_t stream)
{
    constexpr int32_t kThreads = 256;
    int64_t const totalElements = static_cast<int64_t>(inputTokens) * static_cast<int64_t>(hiddenSize);
    int32_t const blocks = static_cast<int32_t>((totalElements + kThreads - 1) / kThreads);
    deepseekResidentSwiGluFloatToOutputKernel<T><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<T*>(output), outputStride0, gate, gateStride0, up, upStride0, inputTokens, hiddenSize);
}

template <typename T>
void launchDeepseekResidentLowRankGate(void* output, int64_t outputStride0, void const* input, int64_t inputStride0,
    void const* downWeight, int64_t downWeightStride0, int64_t downWeightStride1, void const* upWeight,
    int64_t upWeightStride0, int64_t upWeightStride1, int32_t inputTokens, int32_t hiddenSize, int32_t rank,
    cudaStream_t stream)
{
    constexpr int32_t kThreads = 256;
    deepseekResidentLowRankGateKernel<T, kThreads, kMaxLowRankGateRank><<<inputTokens, kThreads, 0, stream>>>(
        reinterpret_cast<T*>(output), outputStride0, reinterpret_cast<T const*>(input), inputStride0,
        reinterpret_cast<T const*>(downWeight), downWeightStride0, downWeightStride1,
        reinterpret_cast<T const*>(upWeight), upWeightStride0, upWeightStride1, hiddenSize, rank);
}

template <typename T>
void launchDeepseekResidentAddScaledFloatToOutput(void* output, int64_t outputStride0, float const* addend,
    int64_t addendStride0, int32_t inputTokens, int32_t hiddenSize, float addendScale, cudaStream_t stream)
{
    constexpr int32_t kThreads = 256;
    int64_t const totalElements = static_cast<int64_t>(inputTokens) * static_cast<int64_t>(hiddenSize);
    int32_t const blocks = static_cast<int32_t>((totalElements + kThreads - 1) / kThreads);
    deepseekResidentAddScaledFloatToOutputKernel<T><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<T*>(output), outputStride0, addend, addendStride0, inputTokens, hiddenSize, addendScale);
}

} // anonymous namespace

void invokeDeepseekResidentRmsNorm(void* output, int64_t outputStride0, void const* input, int64_t inputStride0,
    void const* weight, int64_t weightStride0, int32_t inputTokens, int32_t hiddenSize, float eps,
    DeepseekResidentNormDtype dtype, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(output != nullptr, "output must not be null");
    TLLM_CHECK_WITH_INFO(input != nullptr, "input must not be null");
    TLLM_CHECK_WITH_INFO(weight != nullptr, "weight must not be null");
    TLLM_CHECK_WITH_INFO(inputTokens > 0, "inputTokens must be positive");
    TLLM_CHECK_WITH_INFO(hiddenSize > 0, "hiddenSize must be positive");
    TLLM_CHECK_WITH_INFO(outputStride0 >= hiddenSize, "output stride is smaller than hidden size");
    TLLM_CHECK_WITH_INFO(inputStride0 >= hiddenSize, "input stride is smaller than hidden size");
    TLLM_CHECK_WITH_INFO(weightStride0 > 0, "weight stride must be positive");

    if (dtype == DeepseekResidentNormDtype::kFloat16)
    {
        launchDeepseekResidentRmsNorm<half>(
            output, outputStride0, input, inputStride0, weight, weightStride0, inputTokens, hiddenSize, eps, stream);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(dtype == DeepseekResidentNormDtype::kBfloat16, "unsupported resident RMSNorm dtype");
        launchDeepseekResidentRmsNorm<__nv_bfloat16>(
            output, outputStride0, input, inputStride0, weight, weightStride0, inputTokens, hiddenSize, eps, stream);
    }
    TLLM_CUDA_CHECK(cudaGetLastError());
}

void invokeDeepseekResidentDsaKvASplitNormPack(void const* kvA, int64_t kvAStride0, void* qLora, int64_t qLoraStride0,
    void* compressedKv, int64_t compressedKvStride0, void* kPe, int64_t kPeStride0, void* latentCache,
    int64_t latentCacheStride0, void const* qWeight, int64_t qWeightStride0, void const* kvWeight,
    int64_t kvWeightStride0, int32_t inputTokens, int32_t qLoraRank, int32_t kvLoraRank, int32_t ropeDim, float eps,
    DeepseekResidentNormDtype dtype, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(kvA != nullptr, "kvA must not be null");
    TLLM_CHECK_WITH_INFO(qLora != nullptr, "qLora must not be null");
    TLLM_CHECK_WITH_INFO(compressedKv != nullptr, "compressedKv must not be null");
    TLLM_CHECK_WITH_INFO(kPe != nullptr, "kPe must not be null");
    TLLM_CHECK_WITH_INFO(latentCache != nullptr, "latentCache must not be null");
    TLLM_CHECK_WITH_INFO(qWeight != nullptr, "qWeight must not be null");
    TLLM_CHECK_WITH_INFO(kvWeight != nullptr, "kvWeight must not be null");
    TLLM_CHECK_WITH_INFO(inputTokens > 0, "inputTokens must be positive");
    TLLM_CHECK_WITH_INFO(qLoraRank > 0, "qLoraRank must be positive");
    TLLM_CHECK_WITH_INFO(kvLoraRank > 0, "kvLoraRank must be positive");
    TLLM_CHECK_WITH_INFO(ropeDim > 0, "ropeDim must be positive");
    TLLM_CHECK_WITH_INFO(kvAStride0 >= qLoraRank + kvLoraRank + ropeDim, "kvA stride is smaller than packed width");
    TLLM_CHECK_WITH_INFO(qLoraStride0 >= qLoraRank, "qLora stride is smaller than q_lora_rank");
    TLLM_CHECK_WITH_INFO(compressedKvStride0 >= kvLoraRank, "compressedKv stride is smaller than kv_lora_rank");
    TLLM_CHECK_WITH_INFO(kPeStride0 >= ropeDim, "kPe stride is smaller than rope dim");
    TLLM_CHECK_WITH_INFO(latentCacheStride0 >= kvLoraRank + ropeDim, "latentCache stride is smaller than latent width");
    TLLM_CHECK_WITH_INFO(qWeightStride0 > 0, "qWeight stride must be positive");
    TLLM_CHECK_WITH_INFO(kvWeightStride0 > 0, "kvWeight stride must be positive");

    if (dtype == DeepseekResidentNormDtype::kFloat16)
    {
        launchDeepseekResidentDsaKvASplitNormPack<half>(kvA, kvAStride0, qLora, qLoraStride0, compressedKv,
            compressedKvStride0, kPe, kPeStride0, latentCache, latentCacheStride0, qWeight, qWeightStride0, kvWeight,
            kvWeightStride0, inputTokens, qLoraRank, kvLoraRank, ropeDim, eps, stream);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(
            dtype == DeepseekResidentNormDtype::kBfloat16, "unsupported resident DSA KV-A split/norm/pack dtype");
        launchDeepseekResidentDsaKvASplitNormPack<__nv_bfloat16>(kvA, kvAStride0, qLora, qLoraStride0, compressedKv,
            compressedKvStride0, kPe, kPeStride0, latentCache, latentCacheStride0, qWeight, qWeightStride0, kvWeight,
            kvWeightStride0, inputTokens, qLoraRank, kvLoraRank, ropeDim, eps, stream);
    }
    TLLM_CUDA_CHECK(cudaGetLastError());
}

void invokeDeepseekResidentAddRmsNorm(void* normOutput, int64_t normOutputStride0, void* residualOutput,
    int64_t residualOutputStride0, void const* input, int64_t inputStride0, void const* residual,
    int64_t residualStride0, void const* weight, int64_t weightStride0, int32_t inputTokens, int32_t hiddenSize,
    float eps, DeepseekResidentNormDtype dtype, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(normOutput != nullptr, "normOutput must not be null");
    TLLM_CHECK_WITH_INFO(residualOutput != nullptr, "residualOutput must not be null");
    TLLM_CHECK_WITH_INFO(input != nullptr, "input must not be null");
    TLLM_CHECK_WITH_INFO(residual != nullptr, "residual must not be null");
    TLLM_CHECK_WITH_INFO(weight != nullptr, "weight must not be null");
    TLLM_CHECK_WITH_INFO(inputTokens > 0, "inputTokens must be positive");
    TLLM_CHECK_WITH_INFO(hiddenSize > 0, "hiddenSize must be positive");
    TLLM_CHECK_WITH_INFO(normOutputStride0 >= hiddenSize, "norm output stride is smaller than hidden size");
    TLLM_CHECK_WITH_INFO(residualOutputStride0 >= hiddenSize, "residual output stride is smaller than hidden size");
    TLLM_CHECK_WITH_INFO(inputStride0 >= hiddenSize, "input stride is smaller than hidden size");
    TLLM_CHECK_WITH_INFO(residualStride0 >= hiddenSize, "residual stride is smaller than hidden size");
    TLLM_CHECK_WITH_INFO(weightStride0 > 0, "weight stride must be positive");

    if (dtype == DeepseekResidentNormDtype::kFloat16)
    {
        launchDeepseekResidentAddRmsNorm<half>(normOutput, normOutputStride0, residualOutput, residualOutputStride0,
            input, inputStride0, residual, residualStride0, weight, weightStride0, inputTokens, hiddenSize, eps,
            stream);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(dtype == DeepseekResidentNormDtype::kBfloat16, "unsupported resident add RMSNorm dtype");
        launchDeepseekResidentAddRmsNorm<__nv_bfloat16>(normOutput, normOutputStride0, residualOutput,
            residualOutputStride0, input, inputStride0, residual, residualStride0, weight, weightStride0, inputTokens,
            hiddenSize, eps, stream);
    }
    TLLM_CUDA_CHECK(cudaGetLastError());
}

void invokeDeepseekResidentAddRmsNormLowRankGate(void* gatedOutput, int64_t gatedOutputStride0, void* residualOutput,
    int64_t residualOutputStride0, void const* input, int64_t inputStride0, void const* residual,
    int64_t residualStride0, void const* normWeight, int64_t normWeightStride0, void const* downWeight,
    int64_t downWeightStride0, int64_t downWeightStride1, void const* upWeight, int64_t upWeightStride0,
    int64_t upWeightStride1, int32_t inputTokens, int32_t hiddenSize, int32_t rank, float eps, bool useGemma,
    DeepseekResidentNormDtype dtype, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(gatedOutput != nullptr, "gatedOutput must not be null");
    TLLM_CHECK_WITH_INFO(residualOutput != nullptr, "residualOutput must not be null");
    TLLM_CHECK_WITH_INFO(input != nullptr, "input must not be null");
    TLLM_CHECK_WITH_INFO(residual != nullptr, "residual must not be null");
    TLLM_CHECK_WITH_INFO(normWeight != nullptr, "normWeight must not be null");
    TLLM_CHECK_WITH_INFO(downWeight != nullptr, "downWeight must not be null");
    TLLM_CHECK_WITH_INFO(upWeight != nullptr, "upWeight must not be null");
    TLLM_CHECK_WITH_INFO(inputTokens > 0, "inputTokens must be positive");
    TLLM_CHECK_WITH_INFO(hiddenSize > 0, "hiddenSize must be positive");
    TLLM_CHECK_WITH_INFO(rank > 0 && rank <= kMaxLowRankGateRank, "low-rank gate rank is unsupported");
    TLLM_CHECK_WITH_INFO(gatedOutputStride0 >= hiddenSize, "gated output stride is smaller than hidden size");
    TLLM_CHECK_WITH_INFO(residualOutputStride0 >= hiddenSize, "residual output stride is smaller than hidden size");
    TLLM_CHECK_WITH_INFO(inputStride0 >= hiddenSize, "input stride is smaller than hidden size");
    TLLM_CHECK_WITH_INFO(residualStride0 >= hiddenSize, "residual stride is smaller than hidden size");
    TLLM_CHECK_WITH_INFO(normWeightStride0 > 0, "norm weight stride must be positive");
    TLLM_CHECK_WITH_INFO(downWeightStride0 >= hiddenSize, "down-weight row stride is smaller than hidden size");
    TLLM_CHECK_WITH_INFO(downWeightStride1 > 0, "down-weight column stride must be positive");
    TLLM_CHECK_WITH_INFO(upWeightStride0 >= rank, "up-weight row stride is smaller than rank");
    TLLM_CHECK_WITH_INFO(upWeightStride1 > 0, "up-weight column stride must be positive");

    if (dtype == DeepseekResidentNormDtype::kFloat16)
    {
        launchDeepseekResidentAddRmsNormLowRankGate<half>(gatedOutput, gatedOutputStride0, residualOutput,
            residualOutputStride0, input, inputStride0, residual, residualStride0, normWeight, normWeightStride0,
            downWeight, downWeightStride0, downWeightStride1, upWeight, upWeightStride0, upWeightStride1, inputTokens,
            hiddenSize, rank, eps, useGemma, stream);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(
            dtype == DeepseekResidentNormDtype::kBfloat16, "unsupported resident add RMSNorm low-rank gate dtype");
        launchDeepseekResidentAddRmsNormLowRankGate<__nv_bfloat16>(gatedOutput, gatedOutputStride0, residualOutput,
            residualOutputStride0, input, inputStride0, residual, residualStride0, normWeight, normWeightStride0,
            downWeight, downWeightStride0, downWeightStride1, upWeight, upWeightStride0, upWeightStride1, inputTokens,
            hiddenSize, rank, eps, useGemma, stream);
    }
    TLLM_CUDA_CHECK(cudaGetLastError());
}

void invokeDeepseekResidentSigmoidMul(void* output, int64_t outputStride0, void const* input, int64_t inputStride0,
    float const* gate, int64_t gateStride0, int32_t inputTokens, int32_t hiddenSize, DeepseekResidentNormDtype dtype,
    cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(output != nullptr, "output must not be null");
    TLLM_CHECK_WITH_INFO(input != nullptr, "input must not be null");
    TLLM_CHECK_WITH_INFO(gate != nullptr, "gate must not be null");
    TLLM_CHECK_WITH_INFO(inputTokens > 0, "inputTokens must be positive");
    TLLM_CHECK_WITH_INFO(hiddenSize > 0, "hiddenSize must be positive");
    TLLM_CHECK_WITH_INFO(outputStride0 >= hiddenSize, "output stride is smaller than hidden size");
    TLLM_CHECK_WITH_INFO(inputStride0 >= hiddenSize, "input stride is smaller than hidden size");
    TLLM_CHECK_WITH_INFO(gateStride0 >= hiddenSize, "gate stride is smaller than hidden size");

    if (dtype == DeepseekResidentNormDtype::kFloat16)
    {
        launchDeepseekResidentSigmoidMul<half>(
            output, outputStride0, input, inputStride0, gate, gateStride0, inputTokens, hiddenSize, stream);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(dtype == DeepseekResidentNormDtype::kBfloat16, "unsupported resident sigmoid-mul dtype");
        launchDeepseekResidentSigmoidMul<__nv_bfloat16>(
            output, outputStride0, input, inputStride0, gate, gateStride0, inputTokens, hiddenSize, stream);
    }
    TLLM_CUDA_CHECK(cudaGetLastError());
}

void invokeDeepseekResidentSwiGluFloatToOutput(void* output, int64_t outputStride0, float const* gate,
    int64_t gateStride0, float const* up, int64_t upStride0, int32_t inputTokens, int32_t hiddenSize,
    DeepseekResidentNormDtype dtype, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(output != nullptr, "output must not be null");
    TLLM_CHECK_WITH_INFO(gate != nullptr, "gate must not be null");
    TLLM_CHECK_WITH_INFO(up != nullptr, "up must not be null");
    TLLM_CHECK_WITH_INFO(inputTokens > 0, "inputTokens must be positive");
    TLLM_CHECK_WITH_INFO(hiddenSize > 0, "hiddenSize must be positive");
    TLLM_CHECK_WITH_INFO(outputStride0 >= hiddenSize, "output stride is smaller than hidden size");
    TLLM_CHECK_WITH_INFO(gateStride0 >= hiddenSize, "gate stride is smaller than hidden size");
    TLLM_CHECK_WITH_INFO(upStride0 >= hiddenSize, "up stride is smaller than hidden size");

    if (dtype == DeepseekResidentNormDtype::kFloat16)
    {
        launchDeepseekResidentSwiGluFloatToOutput<half>(
            output, outputStride0, gate, gateStride0, up, upStride0, inputTokens, hiddenSize, stream);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(dtype == DeepseekResidentNormDtype::kBfloat16, "unsupported resident SwiGLU dtype");
        launchDeepseekResidentSwiGluFloatToOutput<__nv_bfloat16>(
            output, outputStride0, gate, gateStride0, up, upStride0, inputTokens, hiddenSize, stream);
    }
    TLLM_CUDA_CHECK(cudaGetLastError());
}

void invokeDeepseekResidentLowRankGate(void* output, int64_t outputStride0, void const* input, int64_t inputStride0,
    void const* downWeight, int64_t downWeightStride0, int64_t downWeightStride1, void const* upWeight,
    int64_t upWeightStride0, int64_t upWeightStride1, int32_t inputTokens, int32_t hiddenSize, int32_t rank,
    DeepseekResidentNormDtype dtype, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(output != nullptr, "output must not be null");
    TLLM_CHECK_WITH_INFO(input != nullptr, "input must not be null");
    TLLM_CHECK_WITH_INFO(downWeight != nullptr, "downWeight must not be null");
    TLLM_CHECK_WITH_INFO(upWeight != nullptr, "upWeight must not be null");
    TLLM_CHECK_WITH_INFO(inputTokens > 0, "inputTokens must be positive");
    TLLM_CHECK_WITH_INFO(hiddenSize > 0, "hiddenSize must be positive");
    TLLM_CHECK_WITH_INFO(rank > 0 && rank <= kMaxLowRankGateRank, "low-rank gate rank is unsupported");
    TLLM_CHECK_WITH_INFO(outputStride0 >= hiddenSize, "output stride is smaller than hidden size");
    TLLM_CHECK_WITH_INFO(inputStride0 >= hiddenSize, "input stride is smaller than hidden size");
    TLLM_CHECK_WITH_INFO(downWeightStride0 >= hiddenSize, "down-weight row stride is smaller than hidden size");
    TLLM_CHECK_WITH_INFO(downWeightStride1 > 0, "down-weight column stride must be positive");
    TLLM_CHECK_WITH_INFO(upWeightStride0 >= rank, "up-weight row stride is smaller than rank");
    TLLM_CHECK_WITH_INFO(upWeightStride1 > 0, "up-weight column stride must be positive");

    if (dtype == DeepseekResidentNormDtype::kFloat16)
    {
        launchDeepseekResidentLowRankGate<half>(output, outputStride0, input, inputStride0, downWeight,
            downWeightStride0, downWeightStride1, upWeight, upWeightStride0, upWeightStride1, inputTokens, hiddenSize,
            rank, stream);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(dtype == DeepseekResidentNormDtype::kBfloat16, "unsupported resident low-rank gate dtype");
        launchDeepseekResidentLowRankGate<__nv_bfloat16>(output, outputStride0, input, inputStride0, downWeight,
            downWeightStride0, downWeightStride1, upWeight, upWeightStride0, upWeightStride1, inputTokens, hiddenSize,
            rank, stream);
    }
    TLLM_CUDA_CHECK(cudaGetLastError());
}

void invokeDeepseekResidentAddScaledFloatToOutput(void* output, int64_t outputStride0, float const* addend,
    int64_t addendStride0, int32_t inputTokens, int32_t hiddenSize, float addendScale, DeepseekResidentNormDtype dtype,
    cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(output != nullptr, "output must not be null");
    TLLM_CHECK_WITH_INFO(addend != nullptr, "addend must not be null");
    TLLM_CHECK_WITH_INFO(inputTokens > 0, "inputTokens must be positive");
    TLLM_CHECK_WITH_INFO(hiddenSize > 0, "hiddenSize must be positive");
    TLLM_CHECK_WITH_INFO(outputStride0 >= hiddenSize, "output stride is smaller than hidden size");
    TLLM_CHECK_WITH_INFO(addendStride0 >= hiddenSize, "addend stride is smaller than hidden size");

    if (dtype == DeepseekResidentNormDtype::kFloat16)
    {
        launchDeepseekResidentAddScaledFloatToOutput<half>(
            output, outputStride0, addend, addendStride0, inputTokens, hiddenSize, addendScale, stream);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(
            dtype == DeepseekResidentNormDtype::kBfloat16, "unsupported resident add-scaled-float output dtype");
        launchDeepseekResidentAddScaledFloatToOutput<__nv_bfloat16>(
            output, outputStride0, addend, addendStride0, inputTokens, hiddenSize, addendScale, stream);
    }
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels

TRTLLM_NAMESPACE_END

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

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_runtime_api.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

enum class DeepseekResidentNormDtype : int32_t
{
    kFloat16 = 0,
    kBfloat16 = 1,
};

void invokeDeepseekResidentRmsNorm(void* output, int64_t outputStride0, void const* input, int64_t inputStride0,
    void const* weight, int64_t weightStride0, int32_t inputTokens, int32_t hiddenSize, float eps,
    DeepseekResidentNormDtype dtype, cudaStream_t stream);

void invokeDeepseekResidentDsaKvASplitNormPack(void const* kvA, int64_t kvAStride0, void* qLora, int64_t qLoraStride0,
    void* compressedKv, int64_t compressedKvStride0, void* kPe, int64_t kPeStride0, void* latentCache,
    int64_t latentCacheStride0, void const* qWeight, int64_t qWeightStride0, void const* kvWeight,
    int64_t kvWeightStride0, int32_t inputTokens, int32_t qLoraRank, int32_t kvLoraRank, int32_t ropeDim, float eps,
    DeepseekResidentNormDtype dtype, cudaStream_t stream);

void invokeDeepseekResidentAddRmsNorm(void* normOutput, int64_t normOutputStride0, void* residualOutput,
    int64_t residualOutputStride0, void const* input, int64_t inputStride0, void const* residual,
    int64_t residualStride0, void const* weight, int64_t weightStride0, int32_t inputTokens, int32_t hiddenSize,
    float eps, DeepseekResidentNormDtype dtype, cudaStream_t stream);

void invokeDeepseekResidentAddRmsNormLowRankGate(void* gatedOutput, int64_t gatedOutputStride0, void* residualOutput,
    int64_t residualOutputStride0, void const* input, int64_t inputStride0, void const* residual,
    int64_t residualStride0, void const* normWeight, int64_t normWeightStride0, void const* downWeight,
    int64_t downWeightStride0, int64_t downWeightStride1, void const* upWeight, int64_t upWeightStride0,
    int64_t upWeightStride1, int32_t inputTokens, int32_t hiddenSize, int32_t rank, float eps, bool useGemma,
    DeepseekResidentNormDtype dtype, cudaStream_t stream);

void invokeDeepseekResidentSigmoidMul(void* output, int64_t outputStride0, void const* input, int64_t inputStride0,
    float const* gate, int64_t gateStride0, int32_t inputTokens, int32_t hiddenSize, DeepseekResidentNormDtype dtype,
    cudaStream_t stream);

void invokeDeepseekResidentSwiGluFloatToOutput(void* output, int64_t outputStride0, float const* gate,
    int64_t gateStride0, float const* up, int64_t upStride0, int32_t inputTokens, int32_t hiddenSize,
    DeepseekResidentNormDtype dtype, cudaStream_t stream);

void invokeDeepseekResidentLowRankGate(void* output, int64_t outputStride0, void const* input, int64_t inputStride0,
    void const* downWeight, int64_t downWeightStride0, int64_t downWeightStride1, void const* upWeight,
    int64_t upWeightStride0, int64_t upWeightStride1, int32_t inputTokens, int32_t hiddenSize, int32_t rank,
    DeepseekResidentNormDtype dtype, cudaStream_t stream);

void invokeDeepseekResidentAddScaledFloatToOutput(void* output, int64_t outputStride0, float const* addend,
    int64_t addendStride0, int32_t inputTokens, int32_t hiddenSize, float addendScale, DeepseekResidentNormDtype dtype,
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END

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

#include "IndexerHisaNvfp4.h"
#include "tensorrt_llm/common/assert.h"

#include <algorithm>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

constexpr int kIndexerHeadDim = 128;
constexpr int kNVFP4ValueBytes = kIndexerHeadDim / 2;
constexpr int kScaleBytes = 4;
constexpr int kHisaBlockSize = 128;

__device__ __forceinline__ float decodeE2M1(uint32_t code, float scale)
{
    constexpr float kValues[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    float value = kValues[code & 0x7U] * scale;
    return (code & 0x8U) != 0U ? -value : value;
}

__device__ __forceinline__ float loadNvfp4Value(uint8_t const* valuePtr, uint32_t scaleWord, int dim)
{
    uint32_t packed = valuePtr[dim >> 1];
    uint32_t code = (dim & 1) != 0 ? (packed >> 4) : (packed & 0xFU);
    uint32_t scaleExp = (scaleWord >> ((dim >> 5) * 8)) & 0xFFU;
    float scale = __uint_as_float(scaleExp << 23);
    return decodeE2M1(code, scale);
}

__global__ void indexerHisaMeanPoolNvfp4Kernel(uint8_t const* __restrict__ kCache,
    int32_t const* __restrict__ blockTable, int32_t const* __restrict__ kvLens, float* __restrict__ reps,
    int32_t batchSize, int32_t maxBlocks, int32_t pageTableStride, int32_t cacheDim1, int64_t cacheStride0,
    int64_t cacheStride1, int64_t cacheStride2, int64_t cacheStride3)
{
    int dim = static_cast<int>(threadIdx.x);
    if (dim >= kIndexerHeadDim)
    {
        return;
    }

    int totalTasks = batchSize * maxBlocks;
    for (int task = static_cast<int>(blockIdx.x); task < totalTasks; task += static_cast<int>(gridDim.x))
    {
        int batch = task / maxBlocks;
        int hisaBlock = task - batch * maxBlocks;
        int seqLen = kvLens[batch];
        int tokenStart = hisaBlock * kHisaBlockSize;
        int tokenCount = tokenStart < seqLen ? min(kHisaBlockSize, seqLen - tokenStart) : 0;
        float sum = 0.0F;

        for (int i = 0; i < tokenCount; ++i)
        {
            int token = tokenStart + i;
            int logicalPage = token / cacheDim1;
            int pageOffset = token - logicalPage * cacheDim1;
            int32_t physicalPage = blockTable[batch * pageTableStride + logicalPage];
            if (physicalPage < 0)
            {
                continue;
            }

            int64_t tokenBase = static_cast<int64_t>(physicalPage) * cacheStride0
                + static_cast<int64_t>(pageOffset) * cacheStride1;
            uint8_t const* tokenPtr = kCache + tokenBase;
            uint8_t const* valuePtr = tokenPtr;
            uint32_t scaleWord = *reinterpret_cast<uint32_t const*>(tokenPtr + kNVFP4ValueBytes * cacheStride3);
            sum += loadNvfp4Value(valuePtr, scaleWord, dim);
        }

        int64_t outIdx = (static_cast<int64_t>(batch) * maxBlocks + hisaBlock) * kIndexerHeadDim + dim;
        reps[outIdx] = tokenCount == 0 ? 0.0F : sum / static_cast<float>(tokenCount);
    }
}

} // namespace

void invokeIndexerHisaMeanPoolNvfp4(uint8_t const* kCache, int32_t const* blockTable, int32_t const* kvLens,
    float* reps, int32_t batchSize, int32_t maxBlocks, int32_t pageTableStride, int32_t cacheDim0, int32_t cacheDim1,
    int32_t cacheDim2, int32_t cacheDim3, int64_t cacheStride0, int64_t cacheStride1, int64_t cacheStride2,
    int64_t cacheStride3, cudaStream_t stream)
{
    if (batchSize == 0 || maxBlocks == 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(cacheDim1 > 0, "indexer_hisa_mean_pool_nvfp4 requires a positive cache page size");
    TLLM_CHECK_WITH_INFO(cacheDim2 == 1, "indexer_hisa_mean_pool_nvfp4 requires cache dim 2 == 1");
    TLLM_CHECK_WITH_INFO(cacheDim3 == kNVFP4ValueBytes + kScaleBytes,
        "indexer_hisa_mean_pool_nvfp4 requires per-token stride 68 for NVFP4 indexer cache");
    TLLM_CHECK_WITH_INFO(cacheDim0 > 0, "indexer_hisa_mean_pool_nvfp4 requires a non-empty cache");
    TLLM_CHECK_WITH_INFO(cacheStride3 == 1, "indexer_hisa_mean_pool_nvfp4 requires byte-contiguous token payloads");

    dim3 block(kIndexerHeadDim);
    int totalTasks = batchSize * maxBlocks;
    int activeBlocksPerSm = 0;
    int smCount = 0;
    int device = 0;
    TLLM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSm, indexerHisaMeanPoolNvfp4Kernel, block.x, 0));
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
    int residentBlocks = smCount * activeBlocksPerSm;
    int launchBlocks = residentBlocks > 0 ? std::min(totalTasks, residentBlocks) : totalTasks;
    indexerHisaMeanPoolNvfp4Kernel<<<launchBlocks, block, 0, stream>>>(kCache, blockTable, kvLens, reps, batchSize,
        maxBlocks, pageTableStride, cacheDim1, cacheStride0, cacheStride1, cacheStride2, cacheStride3);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels

TRTLLM_NAMESPACE_END

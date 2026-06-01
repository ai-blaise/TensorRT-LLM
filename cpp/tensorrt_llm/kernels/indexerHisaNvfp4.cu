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
#include <cfloat>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

constexpr int kIndexerHeadDim = 128;
constexpr int kNVFP4ValueBytes = kIndexerHeadDim / 2;
constexpr int kScaleBytes = 4;
constexpr int kWarpSize = 32;
constexpr int kFp4ElemsPerLane = 4;
constexpr int kRowsPerQuantBlock = 8;
constexpr int kFp4QuantBlockSize = 32;
constexpr int kHisaBlockSize = 128;
constexpr float kInvE2M1Max = 1.0F / 6.0F;
constexpr float kMinAmax = 1.0e-12F;

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

__device__ __forceinline__ uint32_t quantizeE2M1(float scaled)
{
    float ax = fminf(fabsf(scaled), 6.0F);
    uint32_t idx = static_cast<uint32_t>((ax > 0.25F) + (ax > 0.75F) + (ax > 1.25F) + (ax > 1.75F)
        + (ax > 2.5F) + (ax > 3.5F) + (ax > 5.0F));
    uint32_t code = idx & 0x7U;
    uint32_t sign = (scaled < 0.0F && idx != 0U) ? 1U : 0U;
    return code | (sign << 3);
}


__device__ __forceinline__ int32_t flatIndexToPage(
    int64_t flatIdx, int32_t cacheDim1, int32_t cacheDim2, int32_t cacheDim3)
{
    return static_cast<int32_t>(flatIdx / (static_cast<int64_t>(cacheDim1) * cacheDim2 * cacheDim3));
}

__device__ __forceinline__ int32_t flatIndexToPageOffset(
    int64_t flatIdx, int32_t cacheDim1, int32_t cacheDim2, int32_t cacheDim3)
{
    return static_cast<int32_t>((flatIdx / (static_cast<int64_t>(cacheDim2) * cacheDim3)) % cacheDim1);
}

__global__ void indexerHisaResetPageCountsKernel(
    int64_t const* __restrict__ slotMappingFp8, int32_t* __restrict__ pageCounts, int32_t numTokens, int32_t cacheDim0,
    int32_t cacheDim1, int32_t cacheDim2, int32_t cacheDim3)
{
    int token = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (token >= numTokens)
    {
        return;
    }
    int64_t flatIdx = slotMappingFp8[token];
    if (flatIdx < 0)
    {
        return;
    }
    int32_t page = flatIndexToPage(flatIdx, cacheDim1, cacheDim2, cacheDim3);
    if (page < 0 || page >= cacheDim0)
    {
        return;
    }
    if (flatIndexToPageOffset(flatIdx, cacheDim1, cacheDim2, cacheDim3) == 0)
    {
        pageCounts[page] = 0;
    }
}

__global__ void indexerHisaUpdatePageCountsKernel(
    int64_t const* __restrict__ slotMappingFp8, int32_t* __restrict__ pageCounts, int32_t numTokens, int32_t cacheDim0,
    int32_t cacheDim1, int32_t cacheDim2, int32_t cacheDim3)
{
    int token = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (token >= numTokens)
    {
        return;
    }
    int64_t flatIdx = slotMappingFp8[token];
    if (flatIdx < 0)
    {
        return;
    }
    int32_t page = flatIndexToPage(flatIdx, cacheDim1, cacheDim2, cacheDim3);
    if (page < 0 || page >= cacheDim0)
    {
        return;
    }
    int32_t count = flatIndexToPageOffset(flatIdx, cacheDim1, cacheDim2, cacheDim3) + 1;
    atomicMax(pageCounts + page, count);
}

__global__ void indexerHisaRecomputePageRepsNvfp4Kernel(uint8_t const* __restrict__ kCache,
    float* __restrict__ pageReps, int32_t const* __restrict__ pageCounts, int64_t const* __restrict__ slotMappingFp8,
    int32_t numTokens, int32_t cacheDim0, int32_t cacheDim1, int32_t cacheDim2, int32_t cacheDim3,
    int64_t cacheStride0, int64_t cacheStride1, int64_t cacheStride2, int64_t cacheStride3)
{
    int dim = static_cast<int>(threadIdx.x);
    if (dim >= kIndexerHeadDim)
    {
        return;
    }
    int token = static_cast<int>(blockIdx.x);
    if (token >= numTokens)
    {
        return;
    }
    int64_t flatIdx = slotMappingFp8[token];
    if (flatIdx < 0)
    {
        return;
    }
    int32_t page = flatIndexToPage(flatIdx, cacheDim1, cacheDim2, cacheDim3);
    if (page < 0 || page >= cacheDim0)
    {
        return;
    }
    int32_t pageOffset = flatIndexToPageOffset(flatIdx, cacheDim1, cacheDim2, cacheDim3);
    int32_t count = min(max(pageCounts[page], 0), cacheDim1);
    if (pageOffset + 1 != count)
    {
        return;
    }
    float sum = 0.0F;
    for (int offset = 0; offset < count; ++offset)
    {
        int64_t tokenBase = static_cast<int64_t>(page) * cacheStride0 + static_cast<int64_t>(offset) * cacheStride1;
        uint8_t const* tokenPtr = kCache + tokenBase;
        uint8_t const* valuePtr = tokenPtr;
        uint32_t scaleWord = *reinterpret_cast<uint32_t const*>(tokenPtr + kNVFP4ValueBytes * cacheStride3);
        sum += loadNvfp4Value(valuePtr, scaleWord, dim);
    }
    pageReps[static_cast<int64_t>(page) * kIndexerHeadDim + dim] = count == 0 ? 0.0F : sum / static_cast<float>(count);
}

__global__ void indexerHisaBlockRepsFromPagesKernel(float const* __restrict__ pageReps,
    int32_t const* __restrict__ pageCounts, int32_t const* __restrict__ blockTable,
    int32_t const* __restrict__ kvLens, float* __restrict__ reps, int32_t batchSize, int32_t maxBlocks,
    int32_t pageTableStride, int32_t numPages, int32_t pageSize, int32_t pagesPerHisaBlock)
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
        int maxLogicalPages = (seqLen + pageSize - 1) / pageSize;
        int firstLogicalPage = hisaBlock * pagesPerHisaBlock;
        float weightedSum = 0.0F;
        int countSum = 0;

        for (int pageOffset = 0; pageOffset < pagesPerHisaBlock; ++pageOffset)
        {
            int logicalPage = firstLogicalPage + pageOffset;
            if (logicalPage >= maxLogicalPages || logicalPage >= pageTableStride)
            {
                continue;
            }

            int physicalPage = blockTable[batch * pageTableStride + logicalPage];
            if (physicalPage < 0 || physicalPage >= numPages)
            {
                continue;
            }

            int remaining = seqLen - logicalPage * pageSize;
            int count = min(max(pageCounts[physicalPage], 0), pageSize);
            count = min(count, max(remaining, 0));
            weightedSum += pageReps[static_cast<int64_t>(physicalPage) * kIndexerHeadDim + dim]
                * static_cast<float>(count);
            countSum += count;
        }

        int64_t outIdx = (static_cast<int64_t>(batch) * maxBlocks + hisaBlock) * kIndexerHeadDim + dim;
        reps[outIdx] = countSum == 0 ? 0.0F : weightedSum / static_cast<float>(countSum);
    }
}



__global__ __launch_bounds__(kWarpSize * kRowsPerQuantBlock) void indexerHisaQuantizeBlockRepsNvfp4Kernel(
    float const* __restrict__ blockReps, int8_t* __restrict__ packed, int32_t* __restrict__ scales, int32_t totalRows)
{
    int warpInBlock = static_cast<int>(threadIdx.x) / kWarpSize;
    int lane = static_cast<int>(threadIdx.x) % kWarpSize;
    int row = static_cast<int>(blockIdx.x) * kRowsPerQuantBlock + warpInBlock;
    if (row >= totalRows)
    {
        return;
    }

    int baseDim = lane * kFp4ElemsPerLane;
    float const* rowPtr = blockReps + static_cast<int64_t>(row) * kIndexerHeadDim;
    float v0 = rowPtr[baseDim + 0];
    float v1 = rowPtr[baseDim + 1];
    float v2 = rowPtr[baseDim + 2];
    float v3 = rowPtr[baseDim + 3];

    float localMax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    float amax = localMax;
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFU, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFU, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFU, amax, 4));
    amax = fmaxf(amax, kMinAmax);

    float ratio = amax * kInvE2M1Max;
    uint32_t bits = __float_as_uint(ratio);
    uint32_t expBits = bits & 0x7F800000U;
    if ((bits & 0x007FFFFFU) != 0U)
    {
        expBits += 0x00800000U;
    }
    float scale = __uint_as_float(expBits);

    uint32_t c0 = quantizeE2M1(v0 / scale);
    uint32_t c1 = quantizeE2M1(v1 / scale);
    uint32_t c2 = quantizeE2M1(v2 / scale);
    uint32_t c3 = quantizeE2M1(v3 / scale);
    uint8_t byte0 = static_cast<uint8_t>(c0 | (c1 << 4));
    uint8_t byte1 = static_cast<uint8_t>(c2 | (c3 << 4));
    int packedBase = row * kNVFP4ValueBytes + lane * 2;
    packed[packedBase + 0] = static_cast<int8_t>(byte0);
    packed[packedBase + 1] = static_cast<int8_t>(byte1);

    uint32_t exp = (__float_as_uint(scale) >> 23) & 0xFFU;
    uint32_t b0 = __shfl_sync(0xFFFFFFFFU, exp, 0);
    uint32_t b1 = __shfl_sync(0xFFFFFFFFU, exp, kFp4QuantBlockSize / kFp4ElemsPerLane);
    uint32_t b2 = __shfl_sync(0xFFFFFFFFU, exp, 2 * kFp4QuantBlockSize / kFp4ElemsPerLane);
    uint32_t b3 = __shfl_sync(0xFFFFFFFFU, exp, 3 * kFp4QuantBlockSize / kFp4ElemsPerLane);
    if (lane == 0)
    {
        scales[row] = static_cast<int32_t>(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
    }
}


__global__ __launch_bounds__(kWarpSize * kRowsPerQuantBlock) void indexerHisaQuantizedBlockRepsFromPagesKernel(
    float const* __restrict__ pageReps, int32_t const* __restrict__ pageCounts,
    int32_t const* __restrict__ blockTable, int32_t const* __restrict__ kvLens, int8_t* __restrict__ packed,
    int32_t* __restrict__ scales, int32_t batchSize, int32_t maxBlocks, int32_t pageTableStride, int32_t numPages,
    int32_t pageSize, int32_t pagesPerHisaBlock)
{
    int warpInBlock = static_cast<int>(threadIdx.x) / kWarpSize;
    int lane = static_cast<int>(threadIdx.x) % kWarpSize;
    int row = static_cast<int>(blockIdx.x) * kRowsPerQuantBlock + warpInBlock;
    int totalRows = batchSize * maxBlocks;
    if (row >= totalRows)
    {
        return;
    }

    int batch = row / maxBlocks;
    int hisaBlock = row - batch * maxBlocks;
    int seqLen = kvLens[batch];
    int maxLogicalPages = (seqLen + pageSize - 1) / pageSize;
    int firstLogicalPage = hisaBlock * pagesPerHisaBlock;

    auto loadAverage = [&](int dim) {
        float weightedSum = 0.0F;
        int countSum = 0;
        for (int pageOffset = 0; pageOffset < pagesPerHisaBlock; ++pageOffset)
        {
            int logicalPage = firstLogicalPage + pageOffset;
            if (logicalPage >= maxLogicalPages || logicalPage >= pageTableStride)
            {
                continue;
            }
            int physicalPage = blockTable[batch * pageTableStride + logicalPage];
            if (physicalPage < 0 || physicalPage >= numPages)
            {
                continue;
            }
            int remaining = seqLen - logicalPage * pageSize;
            int count = min(max(pageCounts[physicalPage], 0), pageSize);
            count = min(count, max(remaining, 0));
            weightedSum += pageReps[static_cast<int64_t>(physicalPage) * kIndexerHeadDim + dim]
                * static_cast<float>(count);
            countSum += count;
        }
        return countSum == 0 ? 0.0F : weightedSum / static_cast<float>(countSum);
    };

    int baseDim = lane * kFp4ElemsPerLane;
    float v0 = loadAverage(baseDim + 0);
    float v1 = loadAverage(baseDim + 1);
    float v2 = loadAverage(baseDim + 2);
    float v3 = loadAverage(baseDim + 3);

    float localMax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    float amax = localMax;
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFU, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFU, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFU, amax, 4));
    amax = fmaxf(amax, kMinAmax);

    float ratio = amax * kInvE2M1Max;
    uint32_t bits = __float_as_uint(ratio);
    uint32_t expBits = bits & 0x7F800000U;
    if ((bits & 0x007FFFFFU) != 0U)
    {
        expBits += 0x00800000U;
    }
    float scale = __uint_as_float(expBits);

    uint32_t c0 = quantizeE2M1(v0 / scale);
    uint32_t c1 = quantizeE2M1(v1 / scale);
    uint32_t c2 = quantizeE2M1(v2 / scale);
    uint32_t c3 = quantizeE2M1(v3 / scale);
    int packedBase = row * kNVFP4ValueBytes + lane * 2;
    packed[packedBase + 0] = static_cast<int8_t>(static_cast<uint8_t>(c0 | (c1 << 4)));
    packed[packedBase + 1] = static_cast<int8_t>(static_cast<uint8_t>(c2 | (c3 << 4)));

    uint32_t exp = (__float_as_uint(scale) >> 23) & 0xFFU;
    uint32_t b0 = __shfl_sync(0xFFFFFFFFU, exp, 0);
    uint32_t b1 = __shfl_sync(0xFFFFFFFFU, exp, kFp4QuantBlockSize / kFp4ElemsPerLane);
    uint32_t b2 = __shfl_sync(0xFFFFFFFFU, exp, 2 * kFp4QuantBlockSize / kFp4ElemsPerLane);
    uint32_t b3 = __shfl_sync(0xFFFFFFFFU, exp, 3 * kFp4QuantBlockSize / kFp4ElemsPerLane);
    if (lane == 0)
    {
        scales[row] = static_cast<int32_t>(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
    }
}


__global__ void indexerHisaBlockScoresNvfp4Kernel(uint8_t const* __restrict__ qValues,
    int32_t const* __restrict__ qScales, float const* __restrict__ weights, float const* __restrict__ blockReps,
    int32_t const* __restrict__ prefixLens, float* __restrict__ blockScores, int32_t numRows, int32_t numHeads,
    int32_t maxBlocks, int32_t nextN, int32_t blockSize, int64_t qStride0, int64_t qStride1, int64_t qStride2)
{
    int row = static_cast<int>(blockIdx.x);
    if (row >= numRows)
    {
        return;
    }
    int batch = row / nextN;
    int blockCount = (prefixLens[row] + blockSize - 1) / blockSize;
    blockCount = min(max(blockCount, 0), maxBlocks);

    for (int blockId = static_cast<int>(threadIdx.x); blockId < maxBlocks; blockId += static_cast<int>(blockDim.x))
    {
        float score = -FLT_MAX;
        if (blockId < blockCount)
        {
            score = 0.0F;
            float const* rep = blockReps + (static_cast<int64_t>(batch) * maxBlocks + blockId) * kIndexerHeadDim;
            for (int head = 0; head < numHeads; ++head)
            {
                uint8_t const* q = qValues + static_cast<int64_t>(row) * qStride0
                    + static_cast<int64_t>(head) * qStride1;
                uint32_t scaleWord = static_cast<uint32_t>(qScales[row * numHeads + head]);
                float dot = 0.0F;
#pragma unroll 4
                for (int dim = 0; dim < kIndexerHeadDim; ++dim)
                {
                    dot += loadNvfp4Value(q, scaleWord, dim) * rep[dim];
                }
                score += max(dot, 0.0F) * weights[row * numHeads + head];
            }
        }
        blockScores[row * maxBlocks + blockId] = score;
    }
}


__global__ void indexerHisaCandidatePagesKernel(int32_t const* __restrict__ topBlocks,
    int32_t const* __restrict__ blockTable, int32_t* __restrict__ candidatePageTable, int32_t numRows,
    int32_t blockTopK, int32_t pageTableStride, int32_t nextN, int32_t pagesPerHisaBlock)
{
    int totalPages = blockTopK * pagesPerHisaBlock;
    int row = static_cast<int>(blockIdx.x);
    if (row >= numRows)
    {
        return;
    }
    int batch = row / nextN;
    for (int col = static_cast<int>(threadIdx.x); col < totalPages; col += static_cast<int>(blockDim.x))
    {
        int blockSlot = col / pagesPerHisaBlock;
        int pageOffset = col - blockSlot * pagesPerHisaBlock;
        int topBlock = topBlocks[row * blockTopK + blockSlot];
        int logicalPage = topBlock * pagesPerHisaBlock + pageOffset;
        logicalPage = min(max(logicalPage, 0), pageTableStride - 1);
        candidatePageTable[row * totalPages + col] = blockTable[batch * pageTableStride + logicalPage];
    }
}

__global__ void indexerHisaMaskScoresKernel(float* __restrict__ candidateScores,
    int32_t const* __restrict__ topBlocks, int32_t const* __restrict__ prefixLens, int32_t numRows, int32_t blockTopK,
    int32_t candidateLen, int32_t blockSize)
{
    int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int total = numRows * candidateLen;
    if (idx >= total)
    {
        return;
    }
    int row = idx / candidateLen;
    int col = idx - row * candidateLen;
    int blockSlot = col / blockSize;
    if (blockSlot >= blockTopK)
    {
        candidateScores[idx] = -FLT_MAX;
        return;
    }
    int offset = col - blockSlot * blockSize;
    int topBlock = topBlocks[row * blockTopK + blockSlot];
    int token = topBlock * blockSize + offset;
    if (token >= prefixLens[row])
    {
        candidateScores[idx] = -FLT_MAX;
    }
}

__global__ void indexerHisaRemapSelectedKernel(int32_t const* __restrict__ selected,
    int32_t const* __restrict__ topBlocks, int32_t const* __restrict__ prefixLens, int32_t* __restrict__ topkIndices,
    int32_t numRows, int32_t selectedTopK, int32_t indexTopK, int32_t blockTopK, int32_t blockSize)
{
    int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int total = numRows * indexTopK;
    if (idx >= total)
    {
        return;
    }
    int row = idx / indexTopK;
    int col = idx - row * indexTopK;
    int value = -1;
    if (col < selectedTopK)
    {
        int selectedOffset = selected[row * selectedTopK + col];
        int blockSlot = selectedOffset / blockSize;
        int offset = selectedOffset - blockSlot * blockSize;
        if (blockSlot >= 0 && blockSlot < blockTopK)
        {
            int token = topBlocks[row * blockTopK + blockSlot] * blockSize + offset;
            if (token < prefixLens[row])
            {
                value = token;
            }
        }
    }
    topkIndices[idx] = value;
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


void invokeIndexerHisaUpdatePageRepsNvfp4(uint8_t const* kCache, float* pageReps, int32_t* pageCounts,
    int64_t const* slotMappingFp8, int32_t numTokens, int32_t cacheDim0, int32_t cacheDim1, int32_t cacheDim2,
    int32_t cacheDim3, int64_t cacheStride0, int64_t cacheStride1, int64_t cacheStride2, int64_t cacheStride3,
    cudaStream_t stream)
{
    if (numTokens == 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(cacheDim1 > 0, "indexer_hisa_update_page_reps_nvfp4 requires a positive cache page size");
    TLLM_CHECK_WITH_INFO(cacheDim2 == 1, "indexer_hisa_update_page_reps_nvfp4 requires cache dim 2 == 1");
    TLLM_CHECK_WITH_INFO(cacheDim3 == kNVFP4ValueBytes + kScaleBytes,
        "indexer_hisa_update_page_reps_nvfp4 requires per-token stride 68 for NVFP4 indexer cache");
    TLLM_CHECK_WITH_INFO(cacheDim0 > 0, "indexer_hisa_update_page_reps_nvfp4 requires a non-empty cache");
    TLLM_CHECK_WITH_INFO(cacheStride3 == 1, "indexer_hisa_update_page_reps_nvfp4 requires byte-contiguous payloads");

    constexpr int kThreads = 256;
    dim3 resetBlocks((numTokens + kThreads - 1) / kThreads);
    indexerHisaResetPageCountsKernel<<<resetBlocks, kThreads, 0, stream>>>(
        slotMappingFp8, pageCounts, numTokens, cacheDim0, cacheDim1, cacheDim2, cacheDim3);
    TLLM_CUDA_CHECK(cudaGetLastError());
    indexerHisaUpdatePageCountsKernel<<<resetBlocks, kThreads, 0, stream>>>(
        slotMappingFp8, pageCounts, numTokens, cacheDim0, cacheDim1, cacheDim2, cacheDim3);
    TLLM_CUDA_CHECK(cudaGetLastError());
    indexerHisaRecomputePageRepsNvfp4Kernel<<<numTokens, kIndexerHeadDim, 0, stream>>>(kCache, pageReps, pageCounts,
        slotMappingFp8, numTokens, cacheDim0, cacheDim1, cacheDim2, cacheDim3, cacheStride0, cacheStride1,
        cacheStride2, cacheStride3);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

void invokeIndexerHisaBlockRepsFromPagesNvfp4(float const* pageReps, int32_t const* pageCounts,
    int32_t const* blockTable, int32_t const* kvLens, float* reps, int32_t batchSize, int32_t maxBlocks,
    int32_t pageTableStride, int32_t numPages, int32_t pageSize, cudaStream_t stream)
{
    if (batchSize == 0 || maxBlocks == 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(pageSize > 0, "indexer_hisa_block_reps_from_pages_nvfp4 requires a positive page size");
    TLLM_CHECK_WITH_INFO(kHisaBlockSize % pageSize == 0,
        "indexer_hisa_block_reps_from_pages_nvfp4 requires HISA block size to be divisible by page size");
    TLLM_CHECK_WITH_INFO(numPages > 0, "indexer_hisa_block_reps_from_pages_nvfp4 requires non-empty page reps");
    TLLM_CHECK_WITH_INFO(pageTableStride > 0,
        "indexer_hisa_block_reps_from_pages_nvfp4 requires a positive page-table stride");

    constexpr int kThreads = kIndexerHeadDim;
    int pagesPerHisaBlock = kHisaBlockSize / pageSize;
    int tasks = batchSize * maxBlocks;
    int blocks = min(tasks, 4096);
    indexerHisaBlockRepsFromPagesKernel<<<blocks, kThreads, 0, stream>>>(pageReps, pageCounts, blockTable, kvLens,
        reps, batchSize, maxBlocks, pageTableStride, numPages, pageSize, pagesPerHisaBlock);
    TLLM_CUDA_CHECK(cudaGetLastError());
}



void invokeIndexerHisaQuantizeBlockRepsNvfp4(
    float const* blockReps, int8_t* packed, int32_t* scales, int32_t totalRows, cudaStream_t stream)
{
    if (totalRows == 0)
    {
        return;
    }
    constexpr int kThreads = kWarpSize * kRowsPerQuantBlock;
    int blocks = (totalRows + kRowsPerQuantBlock - 1) / kRowsPerQuantBlock;
    indexerHisaQuantizeBlockRepsNvfp4Kernel<<<blocks, kThreads, 0, stream>>>(blockReps, packed, scales, totalRows);
    TLLM_CUDA_CHECK(cudaGetLastError());
}


void invokeIndexerHisaQuantizedBlockRepsFromPagesNvfp4(float const* pageReps, int32_t const* pageCounts,
    int32_t const* blockTable, int32_t const* kvLens, int8_t* packed, int32_t* scales, int32_t batchSize,
    int32_t maxBlocks, int32_t pageTableStride, int32_t numPages, int32_t pageSize, cudaStream_t stream)
{
    if (batchSize == 0 || maxBlocks == 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(pageSize > 0, "indexer_hisa_quantized_block_reps_from_pages_nvfp4 requires a positive page size");
    TLLM_CHECK_WITH_INFO(kHisaBlockSize % pageSize == 0,
        "indexer_hisa_quantized_block_reps_from_pages_nvfp4 requires HISA block size to be divisible by page size");
    TLLM_CHECK_WITH_INFO(numPages > 0, "indexer_hisa_quantized_block_reps_from_pages_nvfp4 requires non-empty page reps");
    TLLM_CHECK_WITH_INFO(pageTableStride > 0,
        "indexer_hisa_quantized_block_reps_from_pages_nvfp4 requires a positive page-table stride");

    constexpr int kThreads = kWarpSize * kRowsPerQuantBlock;
    int totalRows = batchSize * maxBlocks;
    int blocks = (totalRows + kRowsPerQuantBlock - 1) / kRowsPerQuantBlock;
    int pagesPerHisaBlock = kHisaBlockSize / pageSize;
    indexerHisaQuantizedBlockRepsFromPagesKernel<<<blocks, kThreads, 0, stream>>>(pageReps, pageCounts, blockTable,
        kvLens, packed, scales, batchSize, maxBlocks, pageTableStride, numPages, pageSize, pagesPerHisaBlock);
    TLLM_CUDA_CHECK(cudaGetLastError());
}


void invokeIndexerHisaBlockScoresNvfp4(uint8_t const* qValues, int32_t const* qScales, float const* weights,
    float const* blockReps, int32_t const* prefixLens, float* blockScores, int32_t numRows, int32_t numHeads,
    int32_t maxBlocks, int32_t nextN, int32_t blockSize, int64_t qStride0, int64_t qStride1, int64_t qStride2,
    cudaStream_t stream)
{
    if (numRows == 0 || maxBlocks == 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(numHeads > 0, "indexer_hisa_block_scores_nvfp4 requires at least one head");
    TLLM_CHECK_WITH_INFO(nextN > 0, "indexer_hisa_block_scores_nvfp4 requires next_n > 0");
    TLLM_CHECK_WITH_INFO(blockSize > 0, "indexer_hisa_block_scores_nvfp4 requires positive block size");
    TLLM_CHECK_WITH_INFO(maxBlocks <= 4096, "indexer_hisa_block_scores_nvfp4 supports at most 4096 HISA blocks");
    TLLM_CHECK_WITH_INFO(qStride2 == 1, "indexer_hisa_block_scores_nvfp4 requires contiguous packed FP4 query bytes");

    constexpr int kThreads = 256;
    indexerHisaBlockScoresNvfp4Kernel<<<numRows, kThreads, 0, stream>>>(qValues, qScales, weights, blockReps,
        prefixLens, blockScores, numRows, numHeads, maxBlocks, nextN, blockSize, qStride0, qStride1, qStride2);
    TLLM_CUDA_CHECK(cudaGetLastError());
}


void invokeIndexerHisaCandidatePages(int32_t const* topBlocks, int32_t const* blockTable,
    int32_t* candidatePageTable, int32_t numRows, int32_t blockTopK, int32_t pageTableStride, int32_t nextN,
    int32_t pagesPerHisaBlock, cudaStream_t stream)
{
    if (numRows == 0 || blockTopK == 0 || pagesPerHisaBlock == 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(pageTableStride > 0, "indexer_hisa_candidate_pages requires a positive page-table stride");
    TLLM_CHECK_WITH_INFO(nextN > 0, "indexer_hisa_candidate_pages requires next_n > 0");
    constexpr int kThreads = 256;
    indexerHisaCandidatePagesKernel<<<numRows, kThreads, 0, stream>>>(
        topBlocks, blockTable, candidatePageTable, numRows, blockTopK, pageTableStride, nextN, pagesPerHisaBlock);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

void invokeIndexerHisaMaskScores(float* candidateScores, int32_t const* topBlocks, int32_t const* prefixLens,
    int32_t numRows, int32_t blockTopK, int32_t candidateLen, int32_t blockSize, cudaStream_t stream)
{
    if (numRows == 0 || candidateLen == 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(blockTopK > 0, "indexer_hisa_mask_scores requires block_topk > 0");
    TLLM_CHECK_WITH_INFO(blockSize > 0, "indexer_hisa_mask_scores requires block_size > 0");
    constexpr int kThreads = 256;
    int total = numRows * candidateLen;
    int blocks = (total + kThreads - 1) / kThreads;
    indexerHisaMaskScoresKernel<<<blocks, kThreads, 0, stream>>>(
        candidateScores, topBlocks, prefixLens, numRows, blockTopK, candidateLen, blockSize);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

void invokeIndexerHisaRemapSelected(int32_t const* selected, int32_t const* topBlocks, int32_t const* prefixLens,
    int32_t* topkIndices, int32_t numRows, int32_t selectedTopK, int32_t indexTopK, int32_t blockTopK, int32_t blockSize,
    cudaStream_t stream)
{
    if (numRows == 0 || indexTopK == 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(selectedTopK >= 0, "indexer_hisa_remap_selected requires selected_topk >= 0");
    TLLM_CHECK_WITH_INFO(blockTopK > 0, "indexer_hisa_remap_selected requires block_topk > 0");
    TLLM_CHECK_WITH_INFO(blockSize > 0, "indexer_hisa_remap_selected requires block_size > 0");
    constexpr int kThreads = 256;
    int total = numRows * indexTopK;
    int blocks = (total + kThreads - 1) / kThreads;
    indexerHisaRemapSelectedKernel<<<blocks, kThreads, 0, stream>>>(
        selected, topBlocks, prefixLens, topkIndices, numRows, selectedTopK, indexTopK, blockTopK, blockSize);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

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

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

#include "DeepseekResidentAttentionMetadata.h"
#include "tensorrt_llm/common/assert.h"

#include <algorithm>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

__device__ __forceinline__ int64_t clampedBlockIndex(
    int64_t globalPosition, int32_t tokensPerBlock, int32_t maxBlocksPerSeq)
{
    int64_t blockIdx = globalPosition / tokensPerBlock;
    blockIdx = blockIdx < 0 ? 0 : blockIdx;
    int64_t const maxBlockIdx = static_cast<int64_t>(maxBlocksPerSeq) - 1;
    return blockIdx > maxBlockIdx ? maxBlockIdx : blockIdx;
}

__global__ void deepseekResidentAttentionMetadataRefreshKernel(int32_t const* __restrict__ seqLens,
    int64_t seqLensStride, int32_t const* __restrict__ kvLens, int64_t kvLensStride,
    int32_t const* __restrict__ reqIdxPerToken, int64_t reqIdxPerTokenStride,
    int32_t const* __restrict__ indexerKCacheBlockOffsets, int64_t blockOffsetsStride0, int64_t blockOffsetsStride1,
    int64_t* __restrict__ slotMappingFp8, int64_t slotMappingFp8Stride, int64_t* __restrict__ slotMappingScale,
    int64_t slotMappingScaleStride, int64_t* __restrict__ genKvIndptr, int64_t genKvIndptrStride,
    int64_t* __restrict__ genCachedTokenIndptr, int64_t genCachedTokenIndptrStride, int32_t* __restrict__ kvLens2d,
    int64_t kvLens2dStride0, int64_t kvLens2dStride1, int32_t numTokens, int32_t numSeqs, int32_t numContexts,
    int32_t numGenerations, int32_t maxBlocksPerSeq, int32_t nextNCap, int32_t headDim, int32_t tokensPerBlock,
    int32_t quantBlockSize, int32_t dataBytesPerToken)
{
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    int64_t const scaleSize = static_cast<int64_t>(headDim / quantBlockSize) * 4;
    int64_t const blockStride = static_cast<int64_t>(tokensPerBlock) * (dataBytesPerToken + scaleSize);
    int64_t const scaleBaseOffset = static_cast<int64_t>(tokensPerBlock) * dataBytesPerToken;

    for (int32_t tokenIdx = tid; tokenIdx < numTokens; tokenIdx += static_cast<int32_t>(blockDim.x))
    {
        int32_t reqIdx = reqIdxPerToken[tokenIdx * reqIdxPerTokenStride];
        reqIdx = reqIdx < 0 ? 0 : reqIdx;
        reqIdx = reqIdx >= numSeqs ? numSeqs - 1 : reqIdx;

        int64_t seqStart = 0;
        for (int32_t seqIdx = 0; seqIdx < reqIdx; ++seqIdx)
        {
            seqStart += seqLens[seqIdx * seqLensStride];
        }

        int64_t const seqLen = seqLens[reqIdx * seqLensStride];
        int64_t const kvLen = kvLens[reqIdx * kvLensStride];
        int64_t const tokenOffset = static_cast<int64_t>(tokenIdx) - seqStart;
        int64_t const globalPosition = kvLen - seqLen + tokenOffset;
        int64_t const blockIdx = clampedBlockIndex(globalPosition, tokensPerBlock, maxBlocksPerSeq);
        int64_t posInBlock = globalPosition % tokensPerBlock;
        posInBlock = posInBlock < 0 ? posInBlock + tokensPerBlock : posInBlock;
        int64_t const blockId
            = indexerKCacheBlockOffsets[reqIdx * blockOffsetsStride0 + blockIdx * blockOffsetsStride1];

        slotMappingFp8[tokenIdx * slotMappingFp8Stride] = blockId * blockStride + posInBlock * dataBytesPerToken;
        slotMappingScale[tokenIdx * slotMappingScaleStride]
            = blockId * blockStride + scaleBaseOffset + posInBlock * scaleSize;
    }

    if (numGenerations <= 0)
    {
        return;
    }

    if (tid == 0)
    {
        int64_t kvPrefix = 0;
        int64_t cachedPrefix = 0;
        genKvIndptr[0] = 0;
        genCachedTokenIndptr[0] = 0;
        for (int32_t generationIdx = 0; generationIdx < numGenerations; ++generationIdx)
        {
            int32_t const seqIdx = numContexts + generationIdx;
            int64_t const seqLen = seqLens[seqIdx * seqLensStride];
            int64_t const kvLen = kvLens[seqIdx * kvLensStride];
            kvPrefix += kvLen;
            cachedPrefix += kvLen - seqLen;
            genKvIndptr[(generationIdx + 1) * genKvIndptrStride] = kvPrefix;
            genCachedTokenIndptr[(generationIdx + 1) * genCachedTokenIndptrStride] = cachedPrefix;
        }
    }

    for (int32_t idx = tid; idx < numGenerations * nextNCap; idx += static_cast<int32_t>(blockDim.x))
    {
        int32_t const generationIdx = idx / nextNCap;
        int32_t const nextIdx = idx - generationIdx * nextNCap;
        int32_t const seqIdx = numContexts + generationIdx;
        kvLens2d[generationIdx * kvLens2dStride0 + nextIdx * kvLens2dStride1] = kvLens[seqIdx * kvLensStride];
    }
}

__global__ void deepseekResidentDenseTopkDecodeKernel(int32_t const* __restrict__ kvLens, int64_t kvLensStride,
    int32_t* __restrict__ topkIndices, int64_t topkStride0, int64_t topkStride1, int32_t numRows, int32_t indexTopk)
{
    int32_t const row = static_cast<int32_t>(blockIdx.x);
    int32_t const col = static_cast<int32_t>(blockIdx.y) * static_cast<int32_t>(blockDim.x)
        + static_cast<int32_t>(threadIdx.x);
    if (row >= numRows || col >= indexTopk)
    {
        return;
    }

    int32_t const kvLen = kvLens[row * kvLensStride];
    topkIndices[row * topkStride0 + col * topkStride1] = col < kvLen ? col : -1;
}

} // anonymous namespace

void invokeDeepseekResidentAttentionMetadataRefresh(int32_t const* seqLens, int64_t seqLensStride,
    int32_t const* kvLens, int64_t kvLensStride, int32_t const* reqIdxPerToken, int64_t reqIdxPerTokenStride,
    int32_t const* indexerKCacheBlockOffsets, int64_t blockOffsetsStride0, int64_t blockOffsetsStride1,
    int64_t* slotMappingFp8, int64_t slotMappingFp8Stride, int64_t* slotMappingScale, int64_t slotMappingScaleStride,
    int64_t* genKvIndptr, int64_t genKvIndptrStride, int64_t* genCachedTokenIndptr, int64_t genCachedTokenIndptrStride,
    int32_t* kvLens2d, int64_t kvLens2dStride0, int64_t kvLens2dStride1, int32_t numTokens, int32_t numSeqs,
    int32_t numContexts, int32_t numGenerations, int32_t maxBlocksPerSeq, int32_t nextNCap, int32_t headDim,
    int32_t tokensPerBlock, int32_t quantBlockSize, int32_t dataBytesPerToken, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(seqLens != nullptr, "seqLens must not be null");
    TLLM_CHECK_WITH_INFO(kvLens != nullptr, "kvLens must not be null");
    TLLM_CHECK_WITH_INFO(reqIdxPerToken != nullptr, "reqIdxPerToken must not be null");
    TLLM_CHECK_WITH_INFO(indexerKCacheBlockOffsets != nullptr, "indexerKCacheBlockOffsets must not be null");
    TLLM_CHECK_WITH_INFO(slotMappingFp8 != nullptr, "slotMappingFp8 must not be null");
    TLLM_CHECK_WITH_INFO(slotMappingScale != nullptr, "slotMappingScale must not be null");
    TLLM_CHECK_WITH_INFO(numTokens > 0, "numTokens must be positive");
    TLLM_CHECK_WITH_INFO(numSeqs > 0, "numSeqs must be positive");
    TLLM_CHECK_WITH_INFO(numContexts >= 0, "numContexts must be non-negative");
    TLLM_CHECK_WITH_INFO(numGenerations >= 0, "numGenerations must be non-negative");
    TLLM_CHECK_WITH_INFO(numContexts + numGenerations <= numSeqs, "context/generation counts exceed numSeqs");
    TLLM_CHECK_WITH_INFO(maxBlocksPerSeq > 0, "maxBlocksPerSeq must be positive");
    TLLM_CHECK_WITH_INFO(headDim > 0, "headDim must be positive");
    TLLM_CHECK_WITH_INFO(tokensPerBlock > 0, "tokensPerBlock must be positive");
    TLLM_CHECK_WITH_INFO(quantBlockSize > 0, "quantBlockSize must be positive");
    TLLM_CHECK_WITH_INFO(dataBytesPerToken > 0, "dataBytesPerToken must be positive");
    TLLM_CHECK_WITH_INFO(headDim % quantBlockSize == 0, "headDim must be divisible by quantBlockSize");
    if (numGenerations > 0)
    {
        TLLM_CHECK_WITH_INFO(genKvIndptr != nullptr, "genKvIndptr must not be null when generations are present");
        TLLM_CHECK_WITH_INFO(
            genCachedTokenIndptr != nullptr, "genCachedTokenIndptr must not be null when generations are present");
        TLLM_CHECK_WITH_INFO(kvLens2d != nullptr, "kvLens2d must not be null when generations are present");
        TLLM_CHECK_WITH_INFO(nextNCap > 0, "nextNCap must be positive when generations are present");
    }

    constexpr int32_t kThreads = 256;
    deepseekResidentAttentionMetadataRefreshKernel<<<1, kThreads, 0, stream>>>(seqLens, seqLensStride, kvLens,
        kvLensStride, reqIdxPerToken, reqIdxPerTokenStride, indexerKCacheBlockOffsets, blockOffsetsStride0,
        blockOffsetsStride1, slotMappingFp8, slotMappingFp8Stride, slotMappingScale, slotMappingScaleStride,
        genKvIndptr, genKvIndptrStride, genCachedTokenIndptr, genCachedTokenIndptrStride, kvLens2d, kvLens2dStride0,
        kvLens2dStride1, numTokens, numSeqs, numContexts, numGenerations, maxBlocksPerSeq, nextNCap, headDim,
        tokensPerBlock, quantBlockSize, dataBytesPerToken);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

void invokeDeepseekResidentDenseTopkDecode(int32_t const* kvLens, int64_t kvLensStride, int32_t* topkIndices,
    int64_t topkStride0, int64_t topkStride1, int32_t numRows, int32_t indexTopk, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(kvLens != nullptr, "kvLens must not be null");
    TLLM_CHECK_WITH_INFO(topkIndices != nullptr, "topkIndices must not be null");
    TLLM_CHECK_WITH_INFO(kvLensStride > 0, "kvLensStride must be positive");
    TLLM_CHECK_WITH_INFO(topkStride0 > 0, "topkStride0 must be positive");
    TLLM_CHECK_WITH_INFO(topkStride1 > 0, "topkStride1 must be positive");
    TLLM_CHECK_WITH_INFO(numRows > 0, "numRows must be positive");
    TLLM_CHECK_WITH_INFO(indexTopk > 0, "indexTopk must be positive");

    constexpr int32_t kThreads = 256;
    dim3 const grid(numRows, (indexTopk + kThreads - 1) / kThreads);
    deepseekResidentDenseTopkDecodeKernel<<<grid, kThreads, 0, stream>>>(
        kvLens, kvLensStride, topkIndices, topkStride0, topkStride1, numRows, indexTopk);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels

TRTLLM_NAMESPACE_END

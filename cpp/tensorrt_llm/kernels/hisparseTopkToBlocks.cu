/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#include "hisparseTopkToBlocks.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <algorithm>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

__device__ __forceinline__ uint32_t hisparseHash32(uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

__global__ void hisparseTopkToBlockPositionsKernel(int32_t const* __restrict__ topkIndices,
    int32_t* __restrict__ blockPositions, int32_t* __restrict__ blockCounts, uint8_t* __restrict__ overflowFlags,
    int32_t numRows, int32_t indexTopK, int32_t tokensPerBlock, int32_t maxBlocksPerRow, int32_t hashCapacity)
{
    int32_t const row = blockIdx.x;
    if (row >= numRows)
    {
        return;
    }

    extern __shared__ int32_t smem[];
    int32_t* hashKeys = smem;
    int32_t* selectedBlocks = hashKeys + hashCapacity;
    int32_t* selectedCount = selectedBlocks + maxBlocksPerRow;
    int32_t* overflow = selectedCount + 1;

    for (int32_t i = threadIdx.x; i < hashCapacity; i += blockDim.x)
    {
        hashKeys[i] = -1;
    }
    for (int32_t i = threadIdx.x; i < maxBlocksPerRow; i += blockDim.x)
    {
        selectedBlocks[i] = -1;
    }
    if (threadIdx.x == 0)
    {
        *selectedCount = 0;
        *overflow = 0;
    }
    __syncthreads();

    int32_t const* rowTopk = topkIndices + static_cast<int64_t>(row) * indexTopK;
    for (int32_t col = threadIdx.x; col < indexTopK; col += blockDim.x)
    {
        int32_t const token = rowTopk[col];
        if (token < 0)
        {
            continue;
        }
        int32_t const blockPos = token / tokensPerBlock;
        uint32_t slot = hisparseHash32(static_cast<uint32_t>(blockPos)) & static_cast<uint32_t>(hashCapacity - 1);
        bool inserted = false;
        for (int32_t probe = 0; probe < hashCapacity; ++probe)
        {
            int32_t const old = atomicCAS(hashKeys + slot, -1, blockPos);
            if (old == -1)
            {
                inserted = true;
                break;
            }
            if (old == blockPos)
            {
                break;
            }
            slot = (slot + 1) & static_cast<uint32_t>(hashCapacity - 1);
        }
        if (!inserted)
        {
            continue;
        }
        int32_t const dst = atomicAdd(selectedCount, 1);
        if (dst < maxBlocksPerRow)
        {
            selectedBlocks[dst] = blockPos;
        }
        else
        {
            atomicExch(overflow, 1);
        }
    }
    __syncthreads();

    int32_t const count = *selectedCount;
    int32_t const clippedCount = count < maxBlocksPerRow ? count : maxBlocksPerRow;
    int32_t* rowBlocks = blockPositions + static_cast<int64_t>(row) * maxBlocksPerRow;
    for (int32_t i = threadIdx.x; i < maxBlocksPerRow; i += blockDim.x)
    {
        rowBlocks[i] = i < clippedCount ? selectedBlocks[i] : -1;
    }
    if (threadIdx.x == 0)
    {
        blockCounts[row] = clippedCount;
        overflowFlags[row] = static_cast<uint8_t>((*overflow != 0) || (count > maxBlocksPerRow));
    }
}

} // namespace

void invokeHisparseTopkToBlockPositions(int32_t const* topkIndices, int32_t* blockPositions, int32_t* blockCounts,
    uint8_t* overflowFlags, int32_t numRows, int32_t indexTopK, int32_t tokensPerBlock, int32_t maxBlocksPerRow,
    int32_t hashCapacity, cudaStream_t stream)
{
    if (numRows <= 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(indexTopK > 0, "hisparse_topk_to_block_positions requires index_topk > 0");
    TLLM_CHECK_WITH_INFO(tokensPerBlock > 0, "hisparse_topk_to_block_positions requires tokens_per_block > 0");
    TLLM_CHECK_WITH_INFO(maxBlocksPerRow > 0, "hisparse_topk_to_block_positions requires max_blocks_per_row > 0");
    TLLM_CHECK_WITH_INFO((hashCapacity & (hashCapacity - 1)) == 0,
        "hisparse_topk_to_block_positions hash_capacity must be a power of two");
    TLLM_CHECK_WITH_INFO(hashCapacity >= indexTopK * 2,
        "hisparse_topk_to_block_positions hash_capacity must be at least 2 * index_topk");

    constexpr int32_t kThreads = 256;
    size_t const smemBytes = static_cast<size_t>(hashCapacity + maxBlocksPerRow + 2) * sizeof(int32_t);
    hisparseTopkToBlockPositionsKernel<<<numRows, kThreads, smemBytes, stream>>>(topkIndices, blockPositions,
        blockCounts, overflowFlags, numRows, indexTopK, tokensPerBlock, maxBlocksPerRow, hashCapacity);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels

TRTLLM_NAMESPACE_END

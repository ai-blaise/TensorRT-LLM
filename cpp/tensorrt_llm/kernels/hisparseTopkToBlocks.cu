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

enum HiSparseResolveStatus : uint8_t
{
    kResolveOk = 0,
    kResolveMissingRequest = 1,
    kResolveNotAdmitted = 2,
    kResolveBlockOutOfRange = 3,
    kResolveUncommittedBlock = 4,
    kResolveBadBlockCount = 5,
};

enum HiSparsePlanStatus : uint8_t
{
    kPlanOk = 0,
    kPlanUpstreamInvalid = 1,
    kPlanBadBlockCount = 2,
    kPlanInvalidResolvedBlock = 3,
    kPlanInsufficientHotSlots = 4,
};

enum HiSparseCommitStatus : uint8_t
{
    kCommitOk = 0,
    kCommitUpstreamInvalid = 1,
    kCommitBadBlockCount = 2,
    kCommitInvalidPlan = 3,
    kCommitHotSlotOutOfRange = 4,
};

enum HiSparseCompactStatus : uint8_t
{
    kCompactOk = 0,
    kCompactUpstreamInvalid = 1,
    kCompactBadMissCount = 2,
    kCompactInvalidMissSlot = 3,
};

enum HiSparseBuildHotIndexStatus : uint8_t
{
    kBuildHotIndexOk = 0,
    kBuildHotIndexUpstreamInvalid = 1,
    kBuildHotIndexBadBlockCount = 2,
    kBuildHotIndexBlockMissing = 3,
    kBuildHotIndexInvalidHotSlot = 4,
    kBuildHotIndexOverflow = 5,
};

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
    uint8_t const didOverflow = static_cast<uint8_t>((*overflow != 0) || (count > maxBlocksPerRow));
    int32_t* rowBlocks = blockPositions + static_cast<int64_t>(row) * maxBlocksPerRow;
    for (int32_t i = threadIdx.x; i < maxBlocksPerRow; i += blockDim.x)
    {
        rowBlocks[i] = i < clippedCount ? selectedBlocks[i] : -1;
    }
    if (threadIdx.x == 0)
    {
        // Propagate overflow through the existing native status path. The
        // resolver treats counts wider than max_blocks_per_row as invalid, so
        // an enabled HiSparse path cannot silently serve a clipped hot set.
        blockCounts[row] = didOverflow ? maxBlocksPerRow + 1 : clippedCount;
        overflowFlags[row] = didOverflow;
    }
}

__global__ void hisparseResolveBlocksToHostSlotsKernel(int64_t const* __restrict__ rowRequestIds,
    int32_t const* __restrict__ blockPositions, int32_t const* __restrict__ blockCounts,
    int64_t const* __restrict__ requestIds, int64_t const* __restrict__ requestBlockHostSlots,
    int64_t const* __restrict__ requestBlockCommitGen, bool const* __restrict__ requestAdmitted,
    int64_t* __restrict__ hostSlots, int64_t* __restrict__ commitGens, uint8_t* __restrict__ blockStatus,
    uint8_t* __restrict__ rowStatus, int32_t numRows, int32_t maxBlocksPerRow, int32_t requestSlotCapacity,
    int32_t maxBlocksPerRequest)
{
    int32_t const row = blockIdx.x;
    if (row >= numRows)
    {
        return;
    }

    __shared__ int32_t tableSlot;
    __shared__ int32_t rowCode;

    int32_t const rowOffset = row * maxBlocksPerRow;
    if (threadIdx.x == 0)
    {
        int32_t const count = blockCounts[row];
        tableSlot = -1;
        rowCode = kResolveOk;
        if (count < 0 || count > maxBlocksPerRow)
        {
            rowCode = kResolveBadBlockCount;
        }
        else
        {
            int64_t const reqId = rowRequestIds[row];
            for (int32_t slot = 0; slot < requestSlotCapacity; ++slot)
            {
                if (requestIds[slot] == reqId)
                {
                    tableSlot = slot;
                    break;
                }
            }
            if (tableSlot < 0)
            {
                rowCode = kResolveMissingRequest;
            }
            else if (!requestAdmitted[tableSlot])
            {
                rowCode = kResolveNotAdmitted;
            }
        }
    }
    __syncthreads();

    int32_t const count = rowCode == kResolveBadBlockCount ? maxBlocksPerRow : blockCounts[row];
    for (int32_t i = threadIdx.x; i < maxBlocksPerRow; i += blockDim.x)
    {
        int64_t resolvedHostSlot = -1;
        int64_t resolvedCommitGen = -1;
        int32_t code = kResolveOk;
        if (i < count)
        {
            code = rowCode;
            if (code == kResolveOk)
            {
                int32_t const blockPos = blockPositions[rowOffset + i];
                if (blockPos < 0 || blockPos >= maxBlocksPerRequest)
                {
                    code = kResolveBlockOutOfRange;
                    atomicCAS(&rowCode, kResolveOk, code);
                }
                else
                {
                    int64_t const tableOffset = static_cast<int64_t>(tableSlot) * maxBlocksPerRequest + blockPos;
                    resolvedHostSlot = requestBlockHostSlots[tableOffset];
                    resolvedCommitGen = requestBlockCommitGen[tableOffset];
                    if (resolvedHostSlot < 0 || resolvedCommitGen < 0)
                    {
                        resolvedHostSlot = -1;
                        resolvedCommitGen = -1;
                        code = kResolveUncommittedBlock;
                        atomicCAS(&rowCode, kResolveOk, code);
                    }
                }
            }
        }
        hostSlots[rowOffset + i] = resolvedHostSlot;
        commitGens[rowOffset + i] = resolvedCommitGen;
        blockStatus[rowOffset + i] = static_cast<uint8_t>(code);
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        rowStatus[row] = static_cast<uint8_t>(rowCode);
    }
}

__global__ void hisparsePlanHotSlotsKernel(int64_t const* __restrict__ hostSlots,
    int64_t const* __restrict__ commitGens, int32_t const* __restrict__ blockCounts,
    uint8_t const* __restrict__ resolveRowStatus, int64_t const* __restrict__ hotHostSlot,
    int64_t const* __restrict__ hotCommitGen, int64_t const* __restrict__ hotLruTick,
    int64_t* __restrict__ plannedHotSlots, int64_t* __restrict__ plannedLruTick,
    int64_t* __restrict__ missHostSlots, int64_t* __restrict__ missHotSlots, int32_t* __restrict__ missCounts,
    uint8_t* __restrict__ hitFlags, uint8_t* __restrict__ rowStatus, int32_t numRows, int32_t maxBlocksPerRow,
    int32_t numLayers, int32_t hotCapacity, int32_t layerIdx, int64_t lruTickBase)
{
    if (blockIdx.x != 0 || layerIdx < 0 || layerIdx >= numLayers)
    {
        return;
    }

    extern __shared__ int64_t smem64[];
    int64_t* plannedHost = smem64;
    int64_t* plannedCommit = plannedHost + hotCapacity;
    int64_t* plannedLru = plannedCommit + hotCapacity;
    uint8_t* protectedSlots = reinterpret_cast<uint8_t*>(plannedLru + hotCapacity);

    int64_t const layerOffset = static_cast<int64_t>(layerIdx) * hotCapacity;
    for (int32_t slot = threadIdx.x; slot < hotCapacity; slot += blockDim.x)
    {
        plannedHost[slot] = hotHostSlot[layerOffset + slot];
        plannedCommit[slot] = hotCommitGen[layerOffset + slot];
        plannedLru[slot] = hotLruTick[layerOffset + slot];
        protectedSlots[slot] = 0;
    }
    for (int32_t index = threadIdx.x; index < numRows * maxBlocksPerRow; index += blockDim.x)
    {
        plannedHotSlots[index] = -1;
        plannedLruTick[index] = -1;
        missHostSlots[index] = -1;
        missHotSlots[index] = -1;
        hitFlags[index] = 0;
    }
    for (int32_t row = threadIdx.x; row < numRows; row += blockDim.x)
    {
        missCounts[row] = 0;
        rowStatus[row] = kPlanOk;
    }
    __syncthreads();

    if (threadIdx.x != 0)
    {
        return;
    }

    int64_t nextLru = lruTickBase;
    for (int32_t row = 0; row < numRows; ++row)
    {
        int64_t const rowOffset = static_cast<int64_t>(row) * maxBlocksPerRow;
        uint8_t const upstreamStatus = resolveRowStatus[row];
        if (upstreamStatus != kResolveOk)
        {
            rowStatus[row] = kPlanUpstreamInvalid;
            continue;
        }

        int32_t const count = blockCounts[row];
        if (count < 0 || count > maxBlocksPerRow)
        {
            rowStatus[row] = kPlanBadBlockCount;
            continue;
        }
        if (count > hotCapacity)
        {
            rowStatus[row] = kPlanInsufficientHotSlots;
            continue;
        }

        bool validResolvedBlocks = true;
        for (int32_t i = 0; i < count; ++i)
        {
            if (hostSlots[rowOffset + i] < 0 || commitGens[rowOffset + i] < 0)
            {
                validResolvedBlocks = false;
                break;
            }
        }
        if (!validResolvedBlocks)
        {
            rowStatus[row] = kPlanInvalidResolvedBlock;
            continue;
        }

        int32_t requiredMisses = 0;
        for (int32_t i = 0; i < count; ++i)
        {
            int64_t const hostSlot = hostSlots[rowOffset + i];
            int64_t const commitGen = commitGens[rowOffset + i];
            bool hit = false;
            for (int32_t slot = 0; slot < hotCapacity; ++slot)
            {
                if (plannedHost[slot] == hostSlot && plannedCommit[slot] == commitGen)
                {
                    hit = true;
                    break;
                }
            }
            if (!hit)
            {
                ++requiredMisses;
            }
        }
        int32_t availableVictims = 0;
        for (int32_t slot = 0; slot < hotCapacity; ++slot)
        {
            if (protectedSlots[slot] == 0)
            {
                ++availableVictims;
            }
        }
        if (requiredMisses > availableVictims)
        {
            rowStatus[row] = kPlanInsufficientHotSlots;
            continue;
        }

        int32_t missCount = 0;
        for (int32_t i = 0; i < count; ++i)
        {
            int64_t const hostSlot = hostSlots[rowOffset + i];
            int64_t const commitGen = commitGens[rowOffset + i];
            int32_t selectedHotSlot = -1;
            bool hit = false;

            for (int32_t slot = 0; slot < hotCapacity; ++slot)
            {
                if (plannedHost[slot] == hostSlot && plannedCommit[slot] == commitGen)
                {
                    selectedHotSlot = slot;
                    hit = true;
                    break;
                }
            }

            if (selectedHotSlot < 0)
            {
                for (int32_t slot = 0; slot < hotCapacity; ++slot)
                {
                    if (protectedSlots[slot] == 0 && plannedHost[slot] < 0)
                    {
                        selectedHotSlot = slot;
                        break;
                    }
                }
            }
            if (selectedHotSlot < 0)
            {
                int64_t bestTick = 0x7fffffffffffffffLL;
                for (int32_t slot = 0; slot < hotCapacity; ++slot)
                {
                    if (protectedSlots[slot] != 0)
                    {
                        continue;
                    }
                    if (plannedLru[slot] < bestTick)
                    {
                        bestTick = plannedLru[slot];
                        selectedHotSlot = slot;
                    }
                }
            }
            if (selectedHotSlot < 0)
            {
                rowStatus[row] = kPlanInsufficientHotSlots;
                missCount = 0;
                for (int32_t clear = 0; clear < count; ++clear)
                {
                    plannedHotSlots[rowOffset + clear] = -1;
                    plannedLruTick[rowOffset + clear] = -1;
                    hitFlags[rowOffset + clear] = 0;
                }
                break;
            }

            ++nextLru;
            plannedHotSlots[rowOffset + i] = selectedHotSlot;
            plannedLruTick[rowOffset + i] = nextLru;
            hitFlags[rowOffset + i] = static_cast<uint8_t>(hit);
            protectedSlots[selectedHotSlot] = 1;
            plannedHost[selectedHotSlot] = hostSlot;
            plannedCommit[selectedHotSlot] = commitGen;
            plannedLru[selectedHotSlot] = nextLru;
            if (!hit)
            {
                missHostSlots[rowOffset + missCount] = hostSlot;
                missHotSlots[rowOffset + missCount] = selectedHotSlot;
                ++missCount;
            }
        }
        if (rowStatus[row] == kPlanOk)
        {
            missCounts[row] = missCount;
        }
    }
}

__global__ void hisparseCompactMissScheduleKernel(int64_t const* __restrict__ missHostSlots,
    int64_t const* __restrict__ missHotSlots, int32_t const* __restrict__ missCounts,
    uint8_t const* __restrict__ planRowStatus, int64_t* __restrict__ compactHostSlots,
    int64_t* __restrict__ compactHotSlots, int32_t* __restrict__ compactRowIds, int32_t* __restrict__ copyCount,
    uint8_t* __restrict__ rowStatus, int32_t numRows, int32_t maxBlocksPerRow)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    int32_t dst = 0;
    for (int32_t row = 0; row < numRows; ++row)
    {
        uint8_t const upstreamStatus = planRowStatus[row];
        if (upstreamStatus != kPlanOk)
        {
            rowStatus[row] = kCompactUpstreamInvalid;
            continue;
        }

        int32_t const count = missCounts[row];
        if (count < 0 || count > maxBlocksPerRow)
        {
            rowStatus[row] = kCompactBadMissCount;
            continue;
        }

        int64_t const rowOffset = static_cast<int64_t>(row) * maxBlocksPerRow;
        bool valid = true;
        for (int32_t i = 0; i < count; ++i)
        {
            if (missHostSlots[rowOffset + i] < 0 || missHotSlots[rowOffset + i] < 0)
            {
                valid = false;
                break;
            }
        }
        if (!valid)
        {
            rowStatus[row] = kCompactInvalidMissSlot;
            continue;
        }

        for (int32_t i = 0; i < count; ++i)
        {
            compactHostSlots[dst] = missHostSlots[rowOffset + i];
            compactHotSlots[dst] = missHotSlots[rowOffset + i];
            compactRowIds[dst] = row;
            ++dst;
        }
        rowStatus[row] = kCompactOk;
    }
    copyCount[0] = dst;
}

__global__ void hisparseCommitHotSlotsKernel(int64_t const* __restrict__ hostSlots,
    int64_t const* __restrict__ commitGens, int64_t const* __restrict__ plannedHotSlots,
    int64_t const* __restrict__ plannedLruTick, int32_t const* __restrict__ blockCounts,
    uint8_t const* __restrict__ planRowStatus, int64_t* __restrict__ hotHostSlot,
    int64_t* __restrict__ hotCommitGen, int64_t* __restrict__ hotLruTick, uint8_t* __restrict__ rowStatus,
    int32_t numRows, int32_t maxBlocksPerRow, int32_t numLayers, int32_t hotCapacity, int32_t layerIdx)
{
    if (blockIdx.x != 0 || threadIdx.x != 0 || layerIdx < 0 || layerIdx >= numLayers)
    {
        return;
    }

    int64_t const layerOffset = static_cast<int64_t>(layerIdx) * hotCapacity;
    for (int32_t row = 0; row < numRows; ++row)
    {
        uint8_t const upstreamStatus = planRowStatus[row];
        if (upstreamStatus != kPlanOk)
        {
            rowStatus[row] = kCommitUpstreamInvalid;
            continue;
        }
        int32_t const count = blockCounts[row];
        if (count < 0 || count > maxBlocksPerRow)
        {
            rowStatus[row] = kCommitBadBlockCount;
            continue;
        }

        bool valid = true;
        int64_t const rowOffset = static_cast<int64_t>(row) * maxBlocksPerRow;
        for (int32_t i = 0; i < count; ++i)
        {
            int64_t const hotSlot = plannedHotSlots[rowOffset + i];
            if (hotSlot < 0 || hotSlot >= hotCapacity)
            {
                rowStatus[row] = kCommitHotSlotOutOfRange;
                valid = false;
                break;
            }
            if (hostSlots[rowOffset + i] < 0 || commitGens[rowOffset + i] < 0 || plannedLruTick[rowOffset + i] < 0)
            {
                rowStatus[row] = kCommitInvalidPlan;
                valid = false;
                break;
            }
        }
        if (!valid)
        {
            continue;
        }

        for (int32_t i = 0; i < count; ++i)
        {
            int64_t const hotSlot = plannedHotSlots[rowOffset + i];
            int64_t const dst = layerOffset + hotSlot;
            hotHostSlot[dst] = hostSlots[rowOffset + i];
            hotCommitGen[dst] = commitGens[rowOffset + i];
            hotLruTick[dst] = plannedLruTick[rowOffset + i];
        }
        rowStatus[row] = kCommitOk;
    }
}

__global__ void hisparseBuildHotIndicesKernel(int32_t const* __restrict__ topkIndices,
    int32_t const* __restrict__ blockPositions, int64_t const* __restrict__ plannedHotSlots,
    int32_t const* __restrict__ blockCounts, uint8_t const* __restrict__ commitRowStatus,
    int32_t* __restrict__ hotIndices, uint8_t* __restrict__ rowStatus, int32_t numRows, int32_t indexTopK,
    int32_t maxBlocksPerRow, int32_t hotCapacity, int32_t tokensPerBlock, int32_t strideFactor, int32_t layerIdx)
{
    int32_t const row = blockIdx.x;
    if (row >= numRows)
    {
        return;
    }

    __shared__ int32_t rowCode;
    if (threadIdx.x == 0)
    {
        rowCode = commitRowStatus[row] == kCommitOk ? kBuildHotIndexOk : kBuildHotIndexUpstreamInvalid;
        int32_t const count = blockCounts[row];
        if (rowCode == kBuildHotIndexOk && (count < 0 || count > maxBlocksPerRow))
        {
            rowCode = kBuildHotIndexBadBlockCount;
        }
    }
    __syncthreads();

    int64_t const topkRowOffset = static_cast<int64_t>(row) * indexTopK;
    int64_t const blockRowOffset = static_cast<int64_t>(row) * maxBlocksPerRow;
    int32_t const count = blockCounts[row];
    for (int32_t col = threadIdx.x; col < indexTopK; col += blockDim.x)
    {
        int32_t out = -1;
        int32_t const token = topkIndices[topkRowOffset + col];
        if (token >= 0 && rowCode == kBuildHotIndexOk)
        {
            int32_t const blockPos = token / tokensPerBlock;
            int32_t const tokenOffset = token % tokensPerBlock;
            int64_t hotSlot = -1;
            bool foundBlock = false;
            for (int32_t i = 0; i < count; ++i)
            {
                if (blockPositions[blockRowOffset + i] == blockPos)
                {
                    hotSlot = plannedHotSlots[blockRowOffset + i];
                    foundBlock = true;
                    break;
                }
            }
            if (!foundBlock)
            {
                atomicCAS(&rowCode, kBuildHotIndexOk, kBuildHotIndexBlockMissing);
            }
            else if (hotSlot < 0 || hotSlot >= hotCapacity)
            {
                atomicCAS(&rowCode, kBuildHotIndexOk, kBuildHotIndexInvalidHotSlot);
            }
            else
            {
                int64_t const global = hotSlot * static_cast<int64_t>(strideFactor)
                    + static_cast<int64_t>(layerIdx) * tokensPerBlock + tokenOffset;
                if (global < 0 || global > 2147483647LL)
                {
                    atomicCAS(&rowCode, kBuildHotIndexOk, kBuildHotIndexOverflow);
                }
                else
                {
                    out = static_cast<int32_t>(global);
                }
            }
        }
        hotIndices[topkRowOffset + col] = out;
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        rowStatus[row] = static_cast<uint8_t>(rowCode);
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

void invokeHisparseResolveBlocksToHostSlots(int64_t const* rowRequestIds, int32_t const* blockPositions,
    int32_t const* blockCounts, int64_t const* requestIds, int64_t const* requestBlockHostSlots,
    int64_t const* requestBlockCommitGen, bool const* requestAdmitted, int64_t* hostSlots, int64_t* commitGens,
    uint8_t* blockStatus, uint8_t* rowStatus, int32_t numRows, int32_t maxBlocksPerRow,
    int32_t requestSlotCapacity, int32_t maxBlocksPerRequest, cudaStream_t stream)
{
    if (numRows <= 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(maxBlocksPerRow > 0, "hisparse_resolve_blocks_to_host_slots requires max_blocks_per_row > 0");
    TLLM_CHECK_WITH_INFO(
        requestSlotCapacity > 0, "hisparse_resolve_blocks_to_host_slots requires request_slot_capacity > 0");
    TLLM_CHECK_WITH_INFO(
        maxBlocksPerRequest > 0, "hisparse_resolve_blocks_to_host_slots requires max_blocks_per_request > 0");

    constexpr int32_t kThreads = 128;
    hisparseResolveBlocksToHostSlotsKernel<<<numRows, kThreads, 0, stream>>>(rowRequestIds, blockPositions,
        blockCounts, requestIds, requestBlockHostSlots, requestBlockCommitGen, requestAdmitted, hostSlots, commitGens,
        blockStatus, rowStatus, numRows, maxBlocksPerRow, requestSlotCapacity, maxBlocksPerRequest);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

void invokeHisparsePlanHotSlots(int64_t const* hostSlots, int64_t const* commitGens, int32_t const* blockCounts,
    uint8_t const* resolveRowStatus, int64_t const* hotHostSlot, int64_t const* hotCommitGen,
    int64_t const* hotLruTick, int64_t* plannedHotSlots, int64_t* plannedLruTick, int64_t* missHostSlots,
    int64_t* missHotSlots, int32_t* missCounts, uint8_t* hitFlags, uint8_t* rowStatus, int32_t numRows,
    int32_t maxBlocksPerRow, int32_t numLayers, int32_t hotCapacity, int32_t layerIdx, int64_t lruTickBase,
    cudaStream_t stream)
{
    if (numRows <= 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(maxBlocksPerRow > 0, "hisparse_plan_hot_slots requires max_blocks_per_row > 0");
    TLLM_CHECK_WITH_INFO(numLayers > 0, "hisparse_plan_hot_slots requires num_layers > 0");
    TLLM_CHECK_WITH_INFO(hotCapacity > 0, "hisparse_plan_hot_slots requires hot_capacity > 0");
    TLLM_CHECK_WITH_INFO(layerIdx >= 0 && layerIdx < numLayers, "hisparse_plan_hot_slots layer_idx out of range");
    TLLM_CHECK_WITH_INFO(hotCapacity <= 4096, "hisparse_plan_hot_slots supports hot_capacity <= 4096");

    constexpr int32_t kThreads = 128;
    size_t const smemBytes = static_cast<size_t>(hotCapacity) * (3 * sizeof(int64_t) + sizeof(uint8_t));
    hisparsePlanHotSlotsKernel<<<1, kThreads, smemBytes, stream>>>(hostSlots, commitGens, blockCounts,
        resolveRowStatus, hotHostSlot, hotCommitGen, hotLruTick, plannedHotSlots, plannedLruTick, missHostSlots,
        missHotSlots, missCounts, hitFlags, rowStatus, numRows, maxBlocksPerRow, numLayers, hotCapacity, layerIdx,
        lruTickBase);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

void invokeHisparseCompactMissSchedule(int64_t const* missHostSlots, int64_t const* missHotSlots,
    int32_t const* missCounts, uint8_t const* planRowStatus, int64_t* compactHostSlots, int64_t* compactHotSlots,
    int32_t* compactRowIds, int32_t* copyCount, uint8_t* rowStatus, int32_t numRows, int32_t maxBlocksPerRow,
    cudaStream_t stream)
{
    if (numRows <= 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(maxBlocksPerRow > 0, "hisparse_compact_miss_schedule requires max_blocks_per_row > 0");

    hisparseCompactMissScheduleKernel<<<1, 1, 0, stream>>>(missHostSlots, missHotSlots, missCounts, planRowStatus,
        compactHostSlots, compactHotSlots, compactRowIds, copyCount, rowStatus, numRows, maxBlocksPerRow);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

void invokeHisparseCommitHotSlots(int64_t const* hostSlots, int64_t const* commitGens, int64_t const* plannedHotSlots,
    int64_t const* plannedLruTick, int32_t const* blockCounts, uint8_t const* planRowStatus, int64_t* hotHostSlot,
    int64_t* hotCommitGen, int64_t* hotLruTick, uint8_t* rowStatus, int32_t numRows, int32_t maxBlocksPerRow,
    int32_t numLayers, int32_t hotCapacity, int32_t layerIdx, cudaStream_t stream)
{
    if (numRows <= 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(maxBlocksPerRow > 0, "hisparse_commit_hot_slots requires max_blocks_per_row > 0");
    TLLM_CHECK_WITH_INFO(numLayers > 0, "hisparse_commit_hot_slots requires num_layers > 0");
    TLLM_CHECK_WITH_INFO(hotCapacity > 0, "hisparse_commit_hot_slots requires hot_capacity > 0");
    TLLM_CHECK_WITH_INFO(layerIdx >= 0 && layerIdx < numLayers, "hisparse_commit_hot_slots layer_idx out of range");

    hisparseCommitHotSlotsKernel<<<1, 1, 0, stream>>>(hostSlots, commitGens, plannedHotSlots, plannedLruTick,
        blockCounts, planRowStatus, hotHostSlot, hotCommitGen, hotLruTick, rowStatus, numRows, maxBlocksPerRow,
        numLayers, hotCapacity, layerIdx);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

void invokeHisparseBuildHotIndices(int32_t const* topkIndices, int32_t const* blockPositions,
    int64_t const* plannedHotSlots, int32_t const* blockCounts, uint8_t const* commitRowStatus,
    int32_t* hotIndices, uint8_t* rowStatus, int32_t numRows, int32_t indexTopK, int32_t maxBlocksPerRow,
    int32_t hotCapacity, int32_t tokensPerBlock, int32_t strideFactor, int32_t layerIdx, cudaStream_t stream)
{
    if (numRows <= 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(indexTopK > 0, "hisparse_build_hot_indices requires index_topk > 0");
    TLLM_CHECK_WITH_INFO(maxBlocksPerRow > 0, "hisparse_build_hot_indices requires max_blocks_per_row > 0");
    TLLM_CHECK_WITH_INFO(hotCapacity > 0, "hisparse_build_hot_indices requires hot_capacity > 0");
    TLLM_CHECK_WITH_INFO(tokensPerBlock > 0, "hisparse_build_hot_indices requires tokens_per_block > 0");
    TLLM_CHECK_WITH_INFO(strideFactor > 0, "hisparse_build_hot_indices requires stride_factor > 0");
    TLLM_CHECK_WITH_INFO(layerIdx >= 0, "hisparse_build_hot_indices requires layer_idx >= 0");

    constexpr int32_t kThreads = 256;
    hisparseBuildHotIndicesKernel<<<numRows, kThreads, 0, stream>>>(topkIndices, blockPositions, plannedHotSlots,
        blockCounts, commitRowStatus, hotIndices, rowStatus, numRows, indexTopK, maxBlocksPerRow, hotCapacity,
        tokensPerBlock, strideFactor, layerIdx);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels

TRTLLM_NAMESPACE_END

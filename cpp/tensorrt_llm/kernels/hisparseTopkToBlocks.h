/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

void invokeHisparseTopkToBlockPositions(int32_t const* topkIndices, int32_t* blockPositions, int32_t* blockCounts,
    uint8_t* overflowFlags, int32_t numRows, int32_t indexTopK, int32_t tokensPerBlock, int32_t maxBlocksPerRow,
    int32_t hashCapacity, cudaStream_t stream);

void invokeHisparseResolveBlocksToHostSlots(int64_t const* rowRequestIds, int32_t const* blockPositions,
    int32_t const* blockCounts, int64_t const* requestIds, int64_t const* requestBlockHostSlots,
    int64_t const* requestBlockCommitGen, bool const* requestAdmitted, int64_t* hostSlots, int64_t* commitGens,
    uint8_t* blockStatus, uint8_t* rowStatus, int32_t numRows, int32_t maxBlocksPerRow,
    int32_t requestSlotCapacity, int32_t maxBlocksPerRequest, cudaStream_t stream);

void invokeHisparsePlanHotSlots(int64_t const* hostSlots, int64_t const* commitGens, int32_t const* blockCounts,
    uint8_t const* resolveRowStatus, int64_t const* hotHostSlot, int64_t const* hotCommitGen,
    int64_t const* hotLruTick, int64_t* plannedHotSlots, int64_t* plannedLruTick, int64_t* missHostSlots,
    int64_t* missHotSlots, int32_t* missCounts, uint8_t* hitFlags, uint8_t* rowStatus, int32_t numRows,
    int32_t maxBlocksPerRow, int32_t numLayers, int32_t hotCapacity, int32_t layerIdx, int64_t lruTickBase,
    cudaStream_t stream);

void invokeHisparseCompactMissSchedule(int64_t const* missHostSlots, int64_t const* missHotSlots,
    int32_t const* missCounts, uint8_t const* planRowStatus, int64_t* compactHostSlots, int64_t* compactHotSlots,
    int32_t* compactRowIds, int32_t* copyCount, uint8_t* rowStatus, int32_t numRows, int32_t maxBlocksPerRow,
    cudaStream_t stream);

void invokeHisparseCommitHotSlots(int64_t const* hostSlots, int64_t const* commitGens, int64_t const* plannedHotSlots,
    int64_t const* plannedLruTick, int32_t const* blockCounts, uint8_t const* planRowStatus, int64_t* hotHostSlot,
    int64_t* hotCommitGen, int64_t* hotLruTick, uint8_t* rowStatus, int32_t numRows, int32_t maxBlocksPerRow,
    int32_t numLayers, int32_t hotCapacity, int32_t layerIdx, cudaStream_t stream);

void invokeHisparseBuildHotIndices(int32_t const* topkIndices, int32_t const* blockPositions,
    int64_t const* plannedHotSlots, int32_t const* blockCounts, uint8_t const* commitRowStatus,
    int32_t* hotIndices, uint8_t* rowStatus, int32_t numRows, int32_t indexTopK, int32_t maxBlocksPerRow,
    int32_t hotCapacity, int32_t tokensPerBlock, int32_t strideFactor, int32_t layerIdx, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END

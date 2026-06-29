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
#include "tensorrt_llm/common/cudaUtils.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

void invokeIndexerHisaMeanPoolNvfp4(uint8_t const* kCache, int32_t const* blockTable, int32_t const* kvLens,
    float* reps, int32_t batchSize, int32_t maxBlocks, int32_t pageTableStride, int32_t cacheDim0, int32_t cacheDim1,
    int32_t cacheDim2, int32_t cacheDim3, int64_t cacheStride0, int64_t cacheStride1, int64_t cacheStride2,
    int64_t cacheStride3, cudaStream_t stream = 0);

void invokeIndexerHisaUpdatePageRepsNvfp4(uint8_t const* kCache, float* pageReps, int32_t* pageCounts,
    int64_t const* slotMappingFp8, int32_t numTokens, int32_t cacheDim0, int32_t cacheDim1, int32_t cacheDim2,
    int32_t cacheDim3, int64_t cacheStride0, int64_t cacheStride1, int64_t cacheStride2, int64_t cacheStride3,
    cudaStream_t stream = 0);

void invokeIndexerHisaBlockRepsFromPagesNvfp4(float const* pageReps, int32_t const* pageCounts,
    int32_t const* blockTable, int32_t const* kvLens, float* reps, int32_t batchSize, int32_t maxBlocks,
    int32_t pageTableStride, int32_t numPages, int32_t pageSize, cudaStream_t stream = 0);

void invokeIndexerHisaQuantizeBlockRepsNvfp4(
    float const* blockReps, int8_t* packed, int32_t* scales, int32_t totalRows, cudaStream_t stream = 0);

void invokeIndexerHisaQuantizedBlockRepsFromPagesNvfp4(float const* pageReps, int32_t const* pageCounts,
    int32_t const* blockTable, int32_t const* kvLens, int8_t* packed, int32_t* scales, int32_t batchSize,
    int32_t maxBlocks, int32_t pageTableStride, int32_t numPages, int32_t pageSize, cudaStream_t stream = 0);

void invokeIndexerHisaBlockScoresNvfp4(uint8_t const* qValues, int32_t const* qScales, float const* weights,
    float const* blockReps, int32_t const* prefixLens, float* blockScores, int32_t numRows, int32_t numHeads,
    int32_t maxBlocks, int32_t nextN, int32_t blockSize, int64_t qStride0, int64_t qStride1, int64_t qStride2,
    cudaStream_t stream = 0);

void invokeIndexerHisaCandidatePages(int32_t const* topBlocks, int32_t const* blockTable, int32_t* candidatePageTable,
    int32_t numRows, int32_t blockTopK, int32_t pageTableStride, int32_t nextN, int32_t pagesPerHisaBlock,
    cudaStream_t stream = 0);

void invokeIndexerHisaBlockCounts(
    int32_t const* prefixLens, int32_t* blockCounts, int32_t numRows, int32_t blockSize, cudaStream_t stream = 0);

// scoreStride0 is candidateScores' row stride in elements. The DeepGEMM paged
// MQA logits output is row-padded (stride0 > candidateLen), so flat indexing
// would drift the masked positions for every row > 0.
void invokeIndexerHisaMaskScores(float* candidateScores, int32_t const* topBlocks, int32_t const* prefixLens,
    int32_t numRows, int32_t blockTopK, int32_t candidateLen, int32_t blockSize, int64_t scoreStride0,
    cudaStream_t stream = 0);

void invokeIndexerHisaRemapSelected(int32_t const* selected, int32_t const* topBlocks, int32_t const* prefixLens,
    int32_t* topkIndices, int32_t numRows, int32_t selectedTopK, int32_t indexTopK, int32_t blockTopK,
    int32_t blockSize, cudaStream_t stream = 0);

} // namespace kernels

TRTLLM_NAMESPACE_END

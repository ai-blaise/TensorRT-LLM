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

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

void invokeDeepseekResidentAttentionMetadataRefresh(int32_t const* seqLens, int64_t seqLensStride,
    int32_t const* kvLens, int64_t kvLensStride, int32_t const* reqIdxPerToken, int64_t reqIdxPerTokenStride,
    int32_t const* indexerKCacheBlockOffsets, int64_t blockOffsetsStride0, int64_t blockOffsetsStride1,
    int64_t* slotMappingFp8, int64_t slotMappingFp8Stride, int64_t* slotMappingScale, int64_t slotMappingScaleStride,
    int64_t* genKvIndptr, int64_t genKvIndptrStride, int64_t* genCachedTokenIndptr, int64_t genCachedTokenIndptrStride,
    int32_t* kvLens2d, int64_t kvLens2dStride0, int64_t kvLens2dStride1, int32_t numTokens, int32_t numSeqs,
    int32_t numContexts, int32_t numGenerations, int32_t maxBlocksPerSeq, int32_t nextNCap, int32_t headDim,
    int32_t tokensPerBlock, int32_t quantBlockSize, int32_t dataBytesPerToken, cudaStream_t stream = 0);

void invokeDeepseekResidentDenseTopkDecode(int32_t const* kvLens, int64_t kvLensStride, int32_t* topkIndices,
    int64_t topkStride0, int64_t topkStride1, int32_t numRows, int32_t indexTopk, cudaStream_t stream = 0);

} // namespace kernels

TRTLLM_NAMESPACE_END

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
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

struct KVarNGqaK2V2G128Layout
{
    static constexpr int kGroupSize = 128;
    static constexpr int kHeadDim = 128;
    static constexpr int kBitsPerElement = 2;
    static constexpr int kBytesPerPackedVector = kHeadDim * kBitsPerElement / 8;
    static constexpr int kBytesPerTokenSlot = 76;
    static constexpr int kTileBytes = kBytesPerTokenSlot * kGroupSize;
};

bool kvarnGqaBackendReady();

void invokeKvarnGqaStoreK2V2G128(void const* k, void const* v, std::uint8_t* packedRecords,
    std::int64_t const* blockIds, int layerIdx, int numBlocks, int numKvHeads, int headDim, int groupSize, bool useBf16,
    bool pageLayout, std::int64_t strideBlock, std::int64_t strideToken, std::int64_t strideHead, std::int64_t strideByte,
    cudaStream_t stream = 0);

void invokeKvarnGqaDecodeK2V2G128(void const* q, std::uint8_t const* packedRecords,
    std::int64_t const* blockIds, void const* sinkK, void const* sinkV, void const* tailK, void const* tailV,
    std::int32_t const* seqLens, void* output, int numQueries, int numBlocks, int numHeads, int numKvHeads,
    int headDim, int groupSize, bool useBf16, int seqLensCount, bool pageLayout, std::int64_t strideBlock,
    std::int64_t strideToken, std::int64_t strideHead, std::int64_t strideByte, cudaStream_t stream = 0);

} // namespace kernels

TRTLLM_NAMESPACE_END

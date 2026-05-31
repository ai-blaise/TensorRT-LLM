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

} // namespace kernels

TRTLLM_NAMESPACE_END

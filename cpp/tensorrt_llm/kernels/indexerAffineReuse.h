/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Affine-reuse of a previously computed global TopK index tensor across a
// TopK-reuse ("S") indexer layer. convertReqIndexToGlobal writes
//   out = base * strideFactor + (tok % blockSize) + layerId * blockSize
// for valid entries and -1 for invalid ones. Within an FSSS reuse group the
// owning ("F") layer's local TopK is reused unchanged, so `base`,
// `tok % blockSize` and the validity are identical across the group; only the
// `layerId * blockSize` term changes. Hence the S-layer global indices equal
// the cached F-layer global indices plus a per-layer-pair constant
//   delta = (layerId_S - layerId_F) * blockSize,
// with -1 entries preserved. This single elementwise pass replaces the full
// gather on each reuse layer.
void invokeIndexerAffineReuse(
    int32_t const* gFglobal, int32_t* out, int64_t numElems, int32_t delta, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END

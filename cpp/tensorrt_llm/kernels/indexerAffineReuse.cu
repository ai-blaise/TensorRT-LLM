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

#include "tensorrt_llm/kernels/indexerAffineReuse.h"

namespace tensorrt_llm
{
namespace kernels
{

namespace
{
__global__ void indexerAffineReuseKernel(
    int32_t const* __restrict__ gFglobal, int32_t* __restrict__ out, int64_t numElems, int32_t delta)
{
    int64_t const i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= numElems)
    {
        return;
    }
    int32_t const v = gFglobal[i];
    out[i] = (v < 0) ? -1 : v + delta;
}
} // namespace

void invokeIndexerAffineReuse(
    int32_t const* gFglobal, int32_t* out, int64_t numElems, int32_t delta, cudaStream_t stream)
{
    if (numElems <= 0)
    {
        return;
    }
    constexpr int kThreads = 256;
    int64_t const blocks = (numElems + kThreads - 1) / kThreads;
    indexerAffineReuseKernel<<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
        gFglobal, out, numElems, delta);
}

} // namespace kernels
} // namespace tensorrt_llm

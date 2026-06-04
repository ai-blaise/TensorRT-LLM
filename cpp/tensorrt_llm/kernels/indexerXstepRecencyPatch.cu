/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/IndexerXstepRecencyPatch.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

// One block per cached Top-K row. Threads in the block stride over the trailing
// `maxDelta` columns. Only the first `delta` of those columns are overwritten
// (with descending absolute positions); the rest keep their cached value, so
// the kernel only touches columns it changes (coalesced int32 writes over a
// contiguous trailing slice).
__global__ void indexerXstepRecencyPatchKernel(int32_t* __restrict__ cachedTopK, int32_t const* __restrict__ refreshEnd,
    int32_t const* __restrict__ curKvLens, int const numRows, int const indexTopK, int const nextN, int const maxDelta)
{
    int const row = blockIdx.x;
    if (row >= numRows)
    {
        return;
    }

    int const batch = row / nextN;
    int const offset = row - batch * nextN;
    int const refresh = refreshEnd[row];
    // curEnd is the absolute end position for this row at the current step.
    int const curEnd = curKvLens[batch] - nextN + offset + 1;
    // delta = clamp(curEnd - refresh, 0, maxDelta).
    int delta = curEnd - refresh;
    delta = delta < 0 ? 0 : (delta > maxDelta ? maxDelta : delta);
    if (delta <= 0)
    {
        return; // nothing appended since refresh; leave the row untouched.
    }

    // Trailing block starts at column (indexTopK - maxDelta); column-offset c in
    // [0, delta) gets absolute position (refresh + delta - 1 - c). Columns with
    // c >= delta are left as the cached value (matches torch.where(valid, ...)).
    int const base = indexTopK - maxDelta;
    int32_t* rowPtr = cachedTopK + static_cast<int64_t>(row) * indexTopK + base;
    int const top = refresh + delta - 1;
    for (int c = threadIdx.x; c < delta; c += blockDim.x)
    {
        rowPtr[c] = top - c;
    }
}

} // namespace

void invokeIndexerXstepRecencyPatch(int32_t* cachedTopK, int32_t const* refreshEnd, int32_t const* curKvLens,
    int const numRows, int const indexTopK, int const nextN, int const maxDelta, cudaStream_t const stream)
{
    if (numRows <= 0 || maxDelta <= 0)
    {
        return;
    }
    // One block per row; cap threads at the patch width rounded up to a warp.
    int threads = maxDelta < 32 ? 32 : (maxDelta > 256 ? 256 : ((maxDelta + 31) / 32) * 32);
    dim3 grid(numRows);
    dim3 block(threads);
    indexerXstepRecencyPatchKernel<<<grid, block, 0, stream>>>(
        cachedTopK, refreshEnd, curKvLens, numRows, indexTopK, nextN, maxDelta);
}

} // namespace kernels

TRTLLM_NAMESPACE_END

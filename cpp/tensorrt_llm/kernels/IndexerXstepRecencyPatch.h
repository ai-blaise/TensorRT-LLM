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

#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

/// Cross-step recency patch for the DSA indexer cross-step Top-K reuse.
///
/// Fuses the ~20 dependent launch-bound PyTorch tensor ops of
/// ``Indexer._xstep_reuse_decode``'s recency-patch branch into a single
/// kernel launch (one block per cached Top-K row). On a reuse decode step the
/// cached ``[numRows, indexTopK]`` int32 Top-K buffer is patched in place so
/// the KV positions appended since the last refresh are selectable, without
/// recomputing the logits MQA + Top-K.
///
/// Per row ``r`` (``r`` indexes ``cachedTopK`` rows == num_gen_tokens):
///   batch  = r / nextN
///   offset = r % nextN
///   curEnd = curKvLens[batch] - nextN + offset + 1
///   delta  = clamp(curEnd - refreshEnd[r], 0, maxDelta)
/// For each trailing column ``c`` in ``[0, maxDelta)`` (absolute column
/// ``indexTopK - maxDelta + c``): write ``refreshEnd[r] + delta - 1 - c`` when
/// ``c < delta``; otherwise leave the cached value untouched. This reproduces
/// the PyTorch reference column placement, descending ordering, and clamp
/// semantics exactly.
///
/// @param cachedTopK   [numRows, indexTopK] int32, patched in place.
/// @param refreshEnd   [numRows] int32, per-row refresh-time end position.
/// @param curKvLens    [numBatches] int32, per-batch current KV length
///                     (gen slice of kv_lens); numBatches == numRows / nextN.
/// @param numRows      cached Top-K rows (num_gen_tokens).
/// @param indexTopK    cached Top-K width (column count of cachedTopK).
/// @param nextN        speculative width (rows per batch element).
/// @param maxDelta     number of trailing columns eligible for patching.
void invokeIndexerXstepRecencyPatch(int32_t* cachedTopK, int32_t const* refreshEnd, int32_t const* curKvLens,
    int const numRows, int const indexTopK, int const nextN, int const maxDelta, cudaStream_t const stream = 0);

} // namespace kernels

TRTLLM_NAMESPACE_END

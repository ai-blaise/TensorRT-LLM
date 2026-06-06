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

#include "tensorrt_llm/kernels/kvarnGqaKernels.h"

#include "tensorrt_llm/common/assert.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

bool kvarnGqaBackendReady()
{
    // This flips to true only after the B200 store/decode kernels, packed-record
    // transfer, sparse reads, CUDA graph lifecycle, and benchmark gates are in place.
    return false;
}

void invokeKvarnGqaStoreK2V2G128(void const*, void const*, std::uint8_t*, std::int64_t const*, int, int, int, int,
    int, cudaStream_t)
{
    TLLM_CHECK_WITH_INFO(false,
        "kvarn_gqa_store k2v2_g128 is registered as a production integration boundary, but the fused B200 "
        "store kernel is not implemented or validated yet");
}

void invokeKvarnGqaDecodeK2V2G128(void const*, std::uint8_t const*, std::int64_t const*, void const*, void const*,
    void const*, void const*, std::int32_t const*, void*, int, int, int, int, int, int, cudaStream_t)
{
    TLLM_CHECK_WITH_INFO(false,
        "kvarn_gqa_decode k2v2_g128 is registered as a production integration boundary, but the fused B200 "
        "decode kernel is not implemented or validated yet");
}

} // namespace kernels

TRTLLM_NAMESPACE_END

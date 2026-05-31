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

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include "tensorrt_llm/kernels/IndexerHisaNvfp4.h"

namespace th = torch;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

th::Tensor indexer_hisa_mean_pool_nvfp4(th::Tensor const& kCache, th::Tensor const& blockTable,
    th::Tensor const& kvLens, int64_t maxBlocks)
{
    TORCH_CHECK(kCache.is_cuda() && blockTable.is_cuda() && kvLens.is_cuda(),
        "k_cache, block_table, and kv_lens must be CUDA tensors");
    TORCH_CHECK(kCache.get_device() == blockTable.get_device() && kCache.get_device() == kvLens.get_device(),
        "k_cache, block_table, and kv_lens must be on the same device");
    TORCH_CHECK(kCache.scalar_type() == torch::kUInt8, "k_cache must be uint8");
    TORCH_CHECK(blockTable.scalar_type() == torch::kInt32, "block_table must be int32");
    TORCH_CHECK(kvLens.scalar_type() == torch::kInt32, "kv_lens must be int32");
    TORCH_CHECK(kCache.dim() == 4,
        "k_cache must be [num_blocks, block_size, 1, per_token_size], got %d dimensions",
        static_cast<int>(kCache.dim()));
    TORCH_CHECK(blockTable.dim() == 2, "block_table must be a 2D tensor");
    TORCH_CHECK(kvLens.dim() == 1, "kv_lens must be a 1D tensor");
    TORCH_CHECK(blockTable.is_contiguous(), "block_table must be contiguous");
    TORCH_CHECK(kvLens.is_contiguous(), "kv_lens must be contiguous");
    TORCH_CHECK(maxBlocks >= 0, "max_blocks must be non-negative");

    int32_t batchSize = static_cast<int32_t>(blockTable.size(0));
    TORCH_CHECK(kvLens.size(0) >= batchSize, "kv_lens must contain at least one length per block_table row");
    auto addressableBlocks = (blockTable.size(1) * kCache.size(1) + 127) / 128;
    TORCH_CHECK(maxBlocks <= addressableBlocks,
        "max_blocks exceeds the number of logical blocks addressable by block_table");

    auto reps = th::empty({batchSize, maxBlocks, 128},
        th::TensorOptions().dtype(torch::kFloat32).device(kCache.device()));
    if (batchSize == 0 || maxBlocks == 0)
    {
        return reps;
    }

    auto stream = at::cuda::getCurrentCUDAStream(kCache.get_device());
    tk::invokeIndexerHisaMeanPoolNvfp4(kCache.data_ptr<uint8_t>(), blockTable.data_ptr<int32_t>(),
        kvLens.data_ptr<int32_t>(), reps.data_ptr<float>(), batchSize, static_cast<int32_t>(maxBlocks),
        static_cast<int32_t>(blockTable.stride(0)), static_cast<int32_t>(kCache.size(0)),
        static_cast<int32_t>(kCache.size(1)), static_cast<int32_t>(kCache.size(2)), static_cast<int32_t>(kCache.size(3)),
        static_cast<int64_t>(kCache.stride(0)), static_cast<int64_t>(kCache.stride(1)),
        static_cast<int64_t>(kCache.stride(2)), static_cast<int64_t>(kCache.stride(3)), stream);
    return reps;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("indexer_hisa_mean_pool_nvfp4(Tensor k_cache, Tensor block_table, Tensor kv_lens, int max_blocks) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("indexer_hisa_mean_pool_nvfp4", &tensorrt_llm::torch_ext::indexer_hisa_mean_pool_nvfp4);
}

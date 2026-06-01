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

#include <limits>

namespace th = torch;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{


void indexer_hisa_update_page_reps_nvfp4(th::Tensor const& kCache, th::Tensor& pageReps, th::Tensor& pageCounts,
    th::Tensor const& slotMappingFp8, int64_t numTokens)
{
    TORCH_CHECK(kCache.is_cuda() && pageReps.is_cuda() && pageCounts.is_cuda() && slotMappingFp8.is_cuda(),
        "k_cache, page_reps, page_counts, and slot_mapping_fp8 must be CUDA tensors");
    TORCH_CHECK(kCache.get_device() == pageReps.get_device() && kCache.get_device() == pageCounts.get_device()
            && kCache.get_device() == slotMappingFp8.get_device(),
        "all tensors must be on the same device");
    TORCH_CHECK(kCache.scalar_type() == torch::kUInt8, "k_cache must be uint8");
    TORCH_CHECK(pageReps.scalar_type() == torch::kFloat32, "page_reps must be float32");
    TORCH_CHECK(pageCounts.scalar_type() == torch::kInt32, "page_counts must be int32");
    TORCH_CHECK(slotMappingFp8.scalar_type() == torch::kInt64, "slot_mapping_fp8 must be int64");
    TORCH_CHECK(kCache.dim() == 4, "k_cache must be [num_blocks, block_size, 1, per_token_size]");
    TORCH_CHECK(pageReps.dim() == 2 && pageReps.size(1) == 128, "page_reps must be [num_blocks, 128]");
    TORCH_CHECK(pageCounts.dim() == 1, "page_counts must be [num_blocks]");
    TORCH_CHECK(pageReps.size(0) == kCache.size(0), "page_reps page count must match k_cache");
    TORCH_CHECK(pageCounts.size(0) == kCache.size(0), "page_counts page count must match k_cache");
    TORCH_CHECK(slotMappingFp8.dim() == 1, "slot_mapping_fp8 must be 1D");
    TORCH_CHECK(slotMappingFp8.is_contiguous(), "slot_mapping_fp8 must be contiguous");
    TORCH_CHECK(numTokens >= 0 && numTokens <= slotMappingFp8.size(0), "invalid num_tokens");

    auto stream = at::cuda::getCurrentCUDAStream(kCache.get_device());
    tk::invokeIndexerHisaUpdatePageRepsNvfp4(kCache.data_ptr<uint8_t>(), pageReps.data_ptr<float>(),
        pageCounts.data_ptr<int32_t>(), slotMappingFp8.data_ptr<int64_t>(), static_cast<int32_t>(numTokens),
        static_cast<int32_t>(kCache.size(0)), static_cast<int32_t>(kCache.size(1)), static_cast<int32_t>(kCache.size(2)),
        static_cast<int32_t>(kCache.size(3)), static_cast<int64_t>(kCache.stride(0)),
        static_cast<int64_t>(kCache.stride(1)), static_cast<int64_t>(kCache.stride(2)),
        static_cast<int64_t>(kCache.stride(3)), stream);
}

th::Tensor indexer_hisa_block_reps_from_pages_nvfp4(th::Tensor const& pageReps, th::Tensor const& pageCounts,
    th::Tensor const& blockTable, th::Tensor const& kvLens, int64_t maxBlocks, int64_t pageSize)
{
    TORCH_CHECK(pageReps.is_cuda() && pageCounts.is_cuda() && blockTable.is_cuda() && kvLens.is_cuda(),
        "page_reps, page_counts, block_table, and kv_lens must be CUDA tensors");
    TORCH_CHECK(pageReps.get_device() == pageCounts.get_device() && pageReps.get_device() == blockTable.get_device()
            && pageReps.get_device() == kvLens.get_device(),
        "all tensors must be on the same device");
    TORCH_CHECK(pageReps.scalar_type() == torch::kFloat32, "page_reps must be float32");
    TORCH_CHECK(pageCounts.scalar_type() == torch::kInt32, "page_counts must be int32");
    TORCH_CHECK(blockTable.scalar_type() == torch::kInt32, "block_table must be int32");
    TORCH_CHECK(kvLens.scalar_type() == torch::kInt32, "kv_lens must be int32");
    TORCH_CHECK(pageReps.dim() == 2 && pageReps.size(1) == 128, "page_reps must be [num_pages, 128]");
    TORCH_CHECK(pageCounts.dim() == 1, "page_counts must be [num_pages]");
    TORCH_CHECK(pageCounts.size(0) == pageReps.size(0), "page_counts page count must match page_reps");
    TORCH_CHECK(blockTable.dim() == 2, "block_table must be a 2D tensor");
    TORCH_CHECK(kvLens.dim() == 1, "kv_lens must be a 1D tensor");
    TORCH_CHECK(blockTable.is_contiguous(), "block_table must be contiguous");
    TORCH_CHECK(kvLens.is_contiguous(), "kv_lens must be contiguous");
    TORCH_CHECK(maxBlocks >= 0, "max_blocks must be non-negative");
    TORCH_CHECK(pageSize > 0, "page_size must be positive");

    int32_t batchSize = static_cast<int32_t>(blockTable.size(0));
    TORCH_CHECK(kvLens.size(0) >= batchSize, "kv_lens must contain at least one length per block_table row");
    auto addressableBlocks = (blockTable.size(1) * pageSize + 127) / 128;
    TORCH_CHECK(maxBlocks <= addressableBlocks,
        "max_blocks exceeds the number of logical blocks addressable by block_table");

    auto reps = th::empty({batchSize, maxBlocks, 128},
        th::TensorOptions().dtype(torch::kFloat32).device(pageReps.device()));
    auto stream = at::cuda::getCurrentCUDAStream(pageReps.get_device());
    tk::invokeIndexerHisaBlockRepsFromPagesNvfp4(pageReps.data_ptr<float>(), pageCounts.data_ptr<int32_t>(),
        blockTable.data_ptr<int32_t>(), kvLens.data_ptr<int32_t>(), reps.data_ptr<float>(), batchSize,
        static_cast<int32_t>(maxBlocks), static_cast<int32_t>(blockTable.stride(0)),
        static_cast<int32_t>(pageReps.size(0)), static_cast<int32_t>(pageSize), stream);
    return reps;
}


std::tuple<th::Tensor, th::Tensor> indexer_hisa_quantize_block_reps_nvfp4(th::Tensor const& blockReps)
{
    TORCH_CHECK(blockReps.is_cuda(), "block_reps must be a CUDA tensor");
    TORCH_CHECK(blockReps.scalar_type() == torch::kFloat32, "block_reps must be float32");
    TORCH_CHECK(blockReps.dim() == 3 && blockReps.size(2) == 128, "block_reps must be [batch, max_blocks, 128]");
    TORCH_CHECK(blockReps.is_contiguous(), "block_reps must be contiguous");

    int64_t batchSize = blockReps.size(0);
    int64_t maxBlocks = blockReps.size(1);
    int64_t totalRows = batchSize * maxBlocks;
    TORCH_CHECK(totalRows <= std::numeric_limits<int32_t>::max(), "too many block representative rows");

    auto packed = th::empty({batchSize, maxBlocks, 64},
        th::TensorOptions().dtype(torch::kInt8).device(blockReps.device()));
    auto scales = th::empty({batchSize, maxBlocks},
        th::TensorOptions().dtype(torch::kInt32).device(blockReps.device()));
    if (totalRows == 0)
    {
        return {packed, scales};
    }

    auto stream = at::cuda::getCurrentCUDAStream(blockReps.get_device());
    tk::invokeIndexerHisaQuantizeBlockRepsNvfp4(blockReps.data_ptr<float>(), packed.data_ptr<int8_t>(),
        scales.data_ptr<int32_t>(), static_cast<int32_t>(totalRows), stream);
    return {packed, scales};
}


std::tuple<th::Tensor, th::Tensor> indexer_hisa_quantized_block_reps_from_pages_nvfp4(th::Tensor const& pageReps,
    th::Tensor const& pageCounts, th::Tensor const& blockTable, th::Tensor const& kvLens, int64_t maxBlocks,
    int64_t pageSize)
{
    TORCH_CHECK(pageReps.is_cuda() && pageCounts.is_cuda() && blockTable.is_cuda() && kvLens.is_cuda(),
        "page_reps, page_counts, block_table, and kv_lens must be CUDA tensors");
    TORCH_CHECK(pageReps.get_device() == pageCounts.get_device() && pageReps.get_device() == blockTable.get_device()
            && pageReps.get_device() == kvLens.get_device(),
        "all tensors must be on the same device");
    TORCH_CHECK(pageReps.scalar_type() == torch::kFloat32, "page_reps must be float32");
    TORCH_CHECK(pageCounts.scalar_type() == torch::kInt32, "page_counts must be int32");
    TORCH_CHECK(blockTable.scalar_type() == torch::kInt32, "block_table must be int32");
    TORCH_CHECK(kvLens.scalar_type() == torch::kInt32, "kv_lens must be int32");
    TORCH_CHECK(pageReps.dim() == 2 && pageReps.size(1) == 128, "page_reps must be [num_pages, 128]");
    TORCH_CHECK(pageCounts.dim() == 1, "page_counts must be [num_pages]");
    TORCH_CHECK(pageCounts.size(0) == pageReps.size(0), "page_counts page count must match page_reps");
    TORCH_CHECK(blockTable.dim() == 2, "block_table must be a 2D tensor");
    TORCH_CHECK(kvLens.dim() == 1, "kv_lens must be a 1D tensor");
    TORCH_CHECK(blockTable.is_contiguous(), "block_table must be contiguous");
    TORCH_CHECK(kvLens.is_contiguous(), "kv_lens must be contiguous");
    TORCH_CHECK(maxBlocks >= 0, "max_blocks must be non-negative");
    TORCH_CHECK(pageSize > 0, "page_size must be positive");

    int32_t batchSize = static_cast<int32_t>(blockTable.size(0));
    TORCH_CHECK(kvLens.size(0) >= batchSize, "kv_lens must contain at least one length per block_table row");
    auto addressableBlocks = (blockTable.size(1) * pageSize + 127) / 128;
    TORCH_CHECK(maxBlocks <= addressableBlocks,
        "max_blocks exceeds the number of logical blocks addressable by block_table");

    auto packed = th::empty({batchSize, maxBlocks, 64},
        th::TensorOptions().dtype(torch::kInt8).device(pageReps.device()));
    auto scales = th::empty({batchSize, maxBlocks},
        th::TensorOptions().dtype(torch::kInt32).device(pageReps.device()));
    if (batchSize == 0 || maxBlocks == 0)
    {
        return {packed, scales};
    }

    auto stream = at::cuda::getCurrentCUDAStream(pageReps.get_device());
    tk::invokeIndexerHisaQuantizedBlockRepsFromPagesNvfp4(pageReps.data_ptr<float>(), pageCounts.data_ptr<int32_t>(),
        blockTable.data_ptr<int32_t>(), kvLens.data_ptr<int32_t>(), packed.data_ptr<int8_t>(),
        scales.data_ptr<int32_t>(), batchSize, static_cast<int32_t>(maxBlocks),
        static_cast<int32_t>(blockTable.stride(0)), static_cast<int32_t>(pageReps.size(0)),
        static_cast<int32_t>(pageSize), stream);
    return {packed, scales};
}


th::Tensor indexer_hisa_block_scores_nvfp4(th::Tensor const& qValues, th::Tensor const& qScales,
    th::Tensor const& weights, th::Tensor const& blockReps, th::Tensor const& prefixLens, int64_t blockTopK,
    int64_t nextN, int64_t blockSize)
{
    TORCH_CHECK(qValues.is_cuda() && qScales.is_cuda() && weights.is_cuda() && blockReps.is_cuda()
            && prefixLens.is_cuda(),
        "q_values, q_scales, weights, block_reps, and prefix_lens must be CUDA tensors");
    TORCH_CHECK(qValues.get_device() == qScales.get_device() && qValues.get_device() == weights.get_device()
            && qValues.get_device() == blockReps.get_device() && qValues.get_device() == prefixLens.get_device(),
        "all tensors must be on the same device");
    TORCH_CHECK(qValues.scalar_type() == torch::kUInt8, "q_values must be uint8");
    TORCH_CHECK(qScales.scalar_type() == torch::kInt32, "q_scales must be int32");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "weights must be float32");
    TORCH_CHECK(blockReps.scalar_type() == torch::kFloat32, "block_reps must be float32");
    TORCH_CHECK(prefixLens.scalar_type() == torch::kInt32, "prefix_lens must be int32");
    TORCH_CHECK(qValues.dim() == 3 && qValues.size(2) == 64, "q_values must be [num_rows, num_heads, 64]");
    TORCH_CHECK(qScales.dim() == 2, "q_scales must be [num_rows, num_heads]");
    TORCH_CHECK(weights.dim() == 2, "weights must be [num_rows, num_heads]");
    TORCH_CHECK(blockReps.dim() == 3 && blockReps.size(2) == 128, "block_reps must be [batch, max_blocks, 128]");
    TORCH_CHECK(prefixLens.dim() == 1, "prefix_lens must be 1D");
    TORCH_CHECK(qScales.size(0) == qValues.size(0) && qScales.size(1) == qValues.size(1),
        "q_scales shape must match q_values rows and heads");
    TORCH_CHECK(weights.size(0) == qValues.size(0) && weights.size(1) == qValues.size(1),
        "weights shape must match q_values rows and heads");
    TORCH_CHECK(prefixLens.size(0) == qValues.size(0), "prefix_lens must have one entry per query row");
    TORCH_CHECK(nextN > 0, "next_n must be positive");
    TORCH_CHECK((qValues.size(0) + nextN - 1) / nextN <= blockReps.size(0),
        "block_reps must contain one row per batch element");
    TORCH_CHECK(blockTopK > 0 && blockTopK <= blockReps.size(1), "invalid block_topk");

    auto blockScores = th::empty({qValues.size(0), blockReps.size(1)},
        th::TensorOptions().dtype(torch::kFloat32).device(qValues.device()));
    auto stream = at::cuda::getCurrentCUDAStream(qValues.get_device());
    tk::invokeIndexerHisaBlockScoresNvfp4(qValues.data_ptr<uint8_t>(), qScales.data_ptr<int32_t>(),
        weights.data_ptr<float>(), blockReps.data_ptr<float>(), prefixLens.data_ptr<int32_t>(),
        blockScores.data_ptr<float>(), static_cast<int32_t>(qValues.size(0)), static_cast<int32_t>(qValues.size(1)),
        static_cast<int32_t>(blockReps.size(1)), static_cast<int32_t>(nextN), static_cast<int32_t>(blockSize),
        static_cast<int64_t>(qValues.stride(0)), static_cast<int64_t>(qValues.stride(1)),
        static_cast<int64_t>(qValues.stride(2)), stream);
    return blockScores;
}


th::Tensor indexer_hisa_candidate_pages(th::Tensor const& topBlocks, th::Tensor const& blockTable, int64_t nextN,
    int64_t pagesPerHisaBlock)
{
    TORCH_CHECK(topBlocks.is_cuda() && blockTable.is_cuda(), "top_blocks and block_table must be CUDA tensors");
    TORCH_CHECK(topBlocks.get_device() == blockTable.get_device(), "top_blocks and block_table must be on same device");
    TORCH_CHECK(topBlocks.scalar_type() == torch::kInt32, "top_blocks must be int32");
    TORCH_CHECK(blockTable.scalar_type() == torch::kInt32, "block_table must be int32");
    TORCH_CHECK(topBlocks.dim() == 2, "top_blocks must be [num_rows, block_topk]");
    TORCH_CHECK(blockTable.dim() == 2, "block_table must be [batch, pages]");
    TORCH_CHECK(blockTable.is_contiguous(), "block_table must be contiguous");
    TORCH_CHECK(nextN > 0, "next_n must be positive");
    TORCH_CHECK(pagesPerHisaBlock > 0, "pages_per_hisa_block must be positive");
    TORCH_CHECK((topBlocks.size(0) + nextN - 1) / nextN <= blockTable.size(0),
        "block_table must contain one row per batch element");

    auto out = th::empty({topBlocks.size(0), topBlocks.size(1) * pagesPerHisaBlock},
        th::TensorOptions().dtype(torch::kInt32).device(topBlocks.device()));
    auto stream = at::cuda::getCurrentCUDAStream(topBlocks.get_device());
    tk::invokeIndexerHisaCandidatePages(topBlocks.data_ptr<int32_t>(), blockTable.data_ptr<int32_t>(),
        out.data_ptr<int32_t>(), static_cast<int32_t>(topBlocks.size(0)), static_cast<int32_t>(topBlocks.size(1)),
        static_cast<int32_t>(blockTable.stride(0)), static_cast<int32_t>(nextN),
        static_cast<int32_t>(pagesPerHisaBlock), stream);
    return out;
}

void indexer_hisa_mask_scores(th::Tensor& candidateScores, th::Tensor const& topBlocks, th::Tensor const& prefixLens,
    int64_t blockSize)
{
    TORCH_CHECK(candidateScores.is_cuda() && topBlocks.is_cuda() && prefixLens.is_cuda(),
        "candidate_scores, top_blocks, and prefix_lens must be CUDA tensors");
    TORCH_CHECK(candidateScores.get_device() == topBlocks.get_device()
            && candidateScores.get_device() == prefixLens.get_device(),
        "all tensors must be on the same device");
    TORCH_CHECK(candidateScores.scalar_type() == torch::kFloat32, "candidate_scores must be float32");
    TORCH_CHECK(topBlocks.scalar_type() == torch::kInt32, "top_blocks must be int32");
    TORCH_CHECK(prefixLens.scalar_type() == torch::kInt32, "prefix_lens must be int32");
    TORCH_CHECK(candidateScores.dim() == 2, "candidate_scores must be [num_rows, candidate_len]");
    TORCH_CHECK(topBlocks.dim() == 2, "top_blocks must be [num_rows, block_topk]");
    TORCH_CHECK(prefixLens.dim() == 1, "prefix_lens must be 1D");
    TORCH_CHECK(candidateScores.size(0) == topBlocks.size(0) && candidateScores.size(0) == prefixLens.size(0),
        "candidate_scores, top_blocks, and prefix_lens row counts must match");
    TORCH_CHECK(blockSize > 0, "block_size must be positive");

    auto stream = at::cuda::getCurrentCUDAStream(candidateScores.get_device());
    tk::invokeIndexerHisaMaskScores(candidateScores.data_ptr<float>(), topBlocks.data_ptr<int32_t>(),
        prefixLens.data_ptr<int32_t>(), static_cast<int32_t>(candidateScores.size(0)),
        static_cast<int32_t>(topBlocks.size(1)), static_cast<int32_t>(candidateScores.size(1)),
        static_cast<int32_t>(blockSize), stream);
}

th::Tensor indexer_hisa_remap_selected(th::Tensor const& selected, th::Tensor const& topBlocks,
    th::Tensor const& prefixLens, int64_t blockSize, int64_t indexTopK)
{
    TORCH_CHECK(selected.is_cuda() && topBlocks.is_cuda() && prefixLens.is_cuda(),
        "selected, top_blocks, and prefix_lens must be CUDA tensors");
    TORCH_CHECK(selected.get_device() == topBlocks.get_device() && selected.get_device() == prefixLens.get_device(),
        "all tensors must be on the same device");
    TORCH_CHECK(selected.scalar_type() == torch::kInt32, "selected must be int32");
    TORCH_CHECK(topBlocks.scalar_type() == torch::kInt32, "top_blocks must be int32");
    TORCH_CHECK(prefixLens.scalar_type() == torch::kInt32, "prefix_lens must be int32");
    TORCH_CHECK(selected.dim() == 2, "selected must be [num_rows, selected_topk]");
    TORCH_CHECK(topBlocks.dim() == 2, "top_blocks must be [num_rows, block_topk]");
    TORCH_CHECK(prefixLens.dim() == 1, "prefix_lens must be 1D");
    TORCH_CHECK(selected.size(0) == topBlocks.size(0) && selected.size(0) == prefixLens.size(0),
        "selected, top_blocks, and prefix_lens row counts must match");
    TORCH_CHECK(blockSize > 0, "block_size must be positive");
    TORCH_CHECK(indexTopK > 0, "index_topk must be positive");

    auto out = th::empty({selected.size(0), indexTopK},
        th::TensorOptions().dtype(torch::kInt32).device(selected.device()));
    auto stream = at::cuda::getCurrentCUDAStream(selected.get_device());
    tk::invokeIndexerHisaRemapSelected(selected.data_ptr<int32_t>(), topBlocks.data_ptr<int32_t>(),
        prefixLens.data_ptr<int32_t>(), out.data_ptr<int32_t>(), static_cast<int32_t>(selected.size(0)),
        static_cast<int32_t>(selected.size(1)), static_cast<int32_t>(indexTopK),
        static_cast<int32_t>(topBlocks.size(1)), static_cast<int32_t>(blockSize), stream);
    return out;
}

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
    m.def("indexer_hisa_update_page_reps_nvfp4(Tensor k_cache, Tensor(a!) page_reps, Tensor(b!) page_counts, Tensor slot_mapping_fp8, int num_tokens) -> ()");
    m.def("indexer_hisa_block_reps_from_pages_nvfp4(Tensor page_reps, Tensor page_counts, Tensor block_table, Tensor kv_lens, int max_blocks, int page_size) -> Tensor");
    m.def("indexer_hisa_quantize_block_reps_nvfp4(Tensor block_reps) -> (Tensor, Tensor)");
    m.def("indexer_hisa_quantized_block_reps_from_pages_nvfp4(Tensor page_reps, Tensor page_counts, Tensor block_table, Tensor kv_lens, int max_blocks, int page_size) -> (Tensor, Tensor)");
    m.def("indexer_hisa_block_scores_nvfp4(Tensor q_values, Tensor q_scales, Tensor weights, Tensor block_reps, Tensor prefix_lens, int block_topk, int next_n, int block_size) -> Tensor");
    m.def("indexer_hisa_candidate_pages(Tensor top_blocks, Tensor block_table, int next_n, int pages_per_hisa_block) -> Tensor");
    m.def("indexer_hisa_mask_scores(Tensor(a!) candidate_scores, Tensor top_blocks, Tensor prefix_lens, int block_size) -> ()");
    m.def("indexer_hisa_remap_selected(Tensor selected, Tensor top_blocks, Tensor prefix_lens, int block_size, int index_topk) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("indexer_hisa_mean_pool_nvfp4", &tensorrt_llm::torch_ext::indexer_hisa_mean_pool_nvfp4);
    m.impl("indexer_hisa_update_page_reps_nvfp4", &tensorrt_llm::torch_ext::indexer_hisa_update_page_reps_nvfp4);
    m.impl("indexer_hisa_block_reps_from_pages_nvfp4", &tensorrt_llm::torch_ext::indexer_hisa_block_reps_from_pages_nvfp4);
    m.impl("indexer_hisa_quantize_block_reps_nvfp4", &tensorrt_llm::torch_ext::indexer_hisa_quantize_block_reps_nvfp4);
    m.impl("indexer_hisa_quantized_block_reps_from_pages_nvfp4", &tensorrt_llm::torch_ext::indexer_hisa_quantized_block_reps_from_pages_nvfp4);
    m.impl("indexer_hisa_block_scores_nvfp4", &tensorrt_llm::torch_ext::indexer_hisa_block_scores_nvfp4);
    m.impl("indexer_hisa_candidate_pages", &tensorrt_llm::torch_ext::indexer_hisa_candidate_pages);
    m.impl("indexer_hisa_mask_scores", &tensorrt_llm::torch_ext::indexer_hisa_mask_scores);
    m.impl("indexer_hisa_remap_selected", &tensorrt_llm::torch_ext::indexer_hisa_remap_selected);
}

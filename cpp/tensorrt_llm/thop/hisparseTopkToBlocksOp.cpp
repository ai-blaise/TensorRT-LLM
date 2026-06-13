/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/hisparseTopkToBlocks.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <tuple>

namespace th = torch;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace
{

int64_t nextPow2(int64_t value)
{
    int64_t out = 1;
    while (out < value)
    {
        out <<= 1;
    }
    return out;
}

} // namespace

std::tuple<th::Tensor, th::Tensor, th::Tensor> hisparseTopkToBlockPositions(
    th::Tensor const& topkIndices, int64_t tokensPerBlock, int64_t maxBlocksPerRow)
{
    TORCH_CHECK(topkIndices.is_cuda(), "topk_indices must be a CUDA tensor");
    TORCH_CHECK(topkIndices.scalar_type() == torch::kInt32, "topk_indices must be int32");
    TORCH_CHECK(topkIndices.dim() == 2, "topk_indices must have shape [rows, index_topk]");
    TORCH_CHECK(tokensPerBlock > 0 && tokensPerBlock <= std::numeric_limits<int32_t>::max(),
        "tokens_per_block must be positive int32-sized, got ", tokensPerBlock);
    TORCH_CHECK(maxBlocksPerRow > 0 && maxBlocksPerRow <= std::numeric_limits<int32_t>::max(),
        "max_blocks_per_row must be positive int32-sized, got ", maxBlocksPerRow);

    auto topk = topkIndices.contiguous();
    int64_t const rows = topk.size(0);
    int64_t const indexTopK = topk.size(1);
    TORCH_CHECK(indexTopK > 0 && indexTopK <= 2048,
        "hisparse_topk_to_block_positions supports 1 <= index_topk <= 2048, got ", indexTopK);
    TORCH_CHECK(maxBlocksPerRow <= indexTopK,
        "max_blocks_per_row cannot exceed index_topk; got max_blocks_per_row=", maxBlocksPerRow,
        ", index_topk=", indexTopK);

    int64_t const hashCapacity = nextPow2(indexTopK * 2);
    TORCH_CHECK(hashCapacity <= 8192,
        "hisparse_topk_to_block_positions hash table too large for shared-memory planner: ", hashCapacity);

    c10::cuda::CUDAGuard guard(topk.device());
    auto blocks = th::empty({rows, maxBlocksPerRow}, topk.options());
    auto counts = th::empty({rows}, topk.options());
    auto overflow = th::empty({rows}, topk.options().dtype(torch::kUInt8));

    tk::invokeHisparseTopkToBlockPositions(topk.data_ptr<int32_t>(), blocks.data_ptr<int32_t>(),
        counts.data_ptr<int32_t>(), overflow.data_ptr<uint8_t>(), static_cast<int32_t>(rows),
        static_cast<int32_t>(indexTopK), static_cast<int32_t>(tokensPerBlock), static_cast<int32_t>(maxBlocksPerRow),
        static_cast<int32_t>(hashCapacity), at::cuda::getCurrentCUDAStream(topk.get_device()).stream());
    return {blocks, counts, overflow};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "hisparse_topk_to_block_positions(Tensor topk_indices, int tokens_per_block, int max_blocks_per_row) "
        "-> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("hisparse_topk_to_block_positions", &tensorrt_llm::torch_ext::hisparseTopkToBlockPositions);
}

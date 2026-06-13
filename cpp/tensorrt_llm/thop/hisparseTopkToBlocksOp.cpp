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

std::tuple<th::Tensor, th::Tensor, th::Tensor, th::Tensor> hisparseResolveBlocksToHostSlots(
    th::Tensor const& rowRequestIds, th::Tensor const& blockPositions, th::Tensor const& blockCounts,
    th::Tensor const& requestIds, th::Tensor const& requestBlockHostSlots,
    th::Tensor const& requestBlockCommitGen, th::Tensor const& requestAdmitted)
{
    TORCH_CHECK(rowRequestIds.is_cuda(), "row_request_ids must be a CUDA tensor");
    TORCH_CHECK(blockPositions.is_cuda(), "block_positions must be a CUDA tensor");
    TORCH_CHECK(blockCounts.is_cuda(), "block_counts must be a CUDA tensor");
    TORCH_CHECK(requestIds.is_cuda(), "request_ids must be a CUDA tensor");
    TORCH_CHECK(requestBlockHostSlots.is_cuda(), "request_block_host_slots must be a CUDA tensor");
    TORCH_CHECK(requestBlockCommitGen.is_cuda(), "request_block_commit_gen must be a CUDA tensor");
    TORCH_CHECK(requestAdmitted.is_cuda(), "request_admitted must be a CUDA tensor");
    TORCH_CHECK(rowRequestIds.scalar_type() == torch::kInt64, "row_request_ids must be int64");
    TORCH_CHECK(blockPositions.scalar_type() == torch::kInt32, "block_positions must be int32");
    TORCH_CHECK(blockCounts.scalar_type() == torch::kInt32, "block_counts must be int32");
    TORCH_CHECK(requestIds.scalar_type() == torch::kInt64, "request_ids must be int64");
    TORCH_CHECK(requestBlockHostSlots.scalar_type() == torch::kInt64, "request_block_host_slots must be int64");
    TORCH_CHECK(requestBlockCommitGen.scalar_type() == torch::kInt64, "request_block_commit_gen must be int64");
    TORCH_CHECK(requestAdmitted.scalar_type() == torch::kBool, "request_admitted must be bool");
    TORCH_CHECK(rowRequestIds.dim() == 1, "row_request_ids must have shape [rows]");
    TORCH_CHECK(blockPositions.dim() == 2, "block_positions must have shape [rows, max_blocks_per_row]");
    TORCH_CHECK(blockCounts.dim() == 1, "block_counts must have shape [rows]");
    TORCH_CHECK(requestIds.dim() == 1, "request_ids must have shape [request_slots]");
    TORCH_CHECK(requestBlockHostSlots.dim() == 2,
        "request_block_host_slots must have shape [request_slots, max_blocks_per_request]");
    TORCH_CHECK(requestBlockCommitGen.dim() == 2,
        "request_block_commit_gen must have shape [request_slots, max_blocks_per_request]");
    TORCH_CHECK(requestAdmitted.dim() == 1, "request_admitted must have shape [request_slots]");

    int64_t const rows = blockPositions.size(0);
    int64_t const maxBlocksPerRow = blockPositions.size(1);
    int64_t const requestSlots = requestIds.size(0);
    int64_t const maxBlocksPerRequest = requestBlockHostSlots.size(1);
    TORCH_CHECK(rowRequestIds.size(0) == rows, "row_request_ids rows mismatch: got ", rowRequestIds.size(0),
        ", expected ", rows);
    TORCH_CHECK(blockCounts.size(0) == rows, "block_counts rows mismatch: got ", blockCounts.size(0),
        ", expected ", rows);
    TORCH_CHECK(requestBlockHostSlots.size(0) == requestSlots,
        "request_block_host_slots slot dimension mismatch: got ", requestBlockHostSlots.size(0),
        ", expected ", requestSlots);
    TORCH_CHECK(requestBlockCommitGen.size(0) == requestSlots,
        "request_block_commit_gen slot dimension mismatch: got ", requestBlockCommitGen.size(0),
        ", expected ", requestSlots);
    TORCH_CHECK(requestBlockCommitGen.size(1) == maxBlocksPerRequest,
        "request_block_commit_gen width mismatch: got ", requestBlockCommitGen.size(1),
        ", expected ", maxBlocksPerRequest);
    TORCH_CHECK(requestAdmitted.size(0) == requestSlots, "request_admitted slot dimension mismatch: got ",
        requestAdmitted.size(0), ", expected ", requestSlots);
    TORCH_CHECK(maxBlocksPerRow > 0 && maxBlocksPerRow <= std::numeric_limits<int32_t>::max(),
        "max_blocks_per_row must be positive int32-sized, got ", maxBlocksPerRow);
    TORCH_CHECK(requestSlots > 0 && requestSlots <= std::numeric_limits<int32_t>::max(),
        "request slot capacity must be positive int32-sized, got ", requestSlots);
    TORCH_CHECK(maxBlocksPerRequest > 0 && maxBlocksPerRequest <= std::numeric_limits<int32_t>::max(),
        "max_blocks_per_request must be positive int32-sized, got ", maxBlocksPerRequest);

    c10::cuda::CUDAGuard guard(blockPositions.device());
    TORCH_CHECK(rowRequestIds.get_device() == blockPositions.get_device(),
        "row_request_ids must be on the same CUDA device as block_positions");
    TORCH_CHECK(blockCounts.get_device() == blockPositions.get_device(),
        "block_counts must be on the same CUDA device as block_positions");
    TORCH_CHECK(requestIds.get_device() == blockPositions.get_device(),
        "request_ids must be on the same CUDA device as block_positions");
    TORCH_CHECK(requestBlockHostSlots.get_device() == blockPositions.get_device(),
        "request_block_host_slots must be on the same CUDA device as block_positions");
    TORCH_CHECK(requestBlockCommitGen.get_device() == blockPositions.get_device(),
        "request_block_commit_gen must be on the same CUDA device as block_positions");
    TORCH_CHECK(requestAdmitted.get_device() == blockPositions.get_device(),
        "request_admitted must be on the same CUDA device as block_positions");

    auto rowIds = rowRequestIds.contiguous();
    auto blocks = blockPositions.contiguous();
    auto counts = blockCounts.contiguous();
    auto tableIds = requestIds.contiguous();
    auto tableHostSlots = requestBlockHostSlots.contiguous();
    auto tableCommitGen = requestBlockCommitGen.contiguous();
    auto tableAdmitted = requestAdmitted.contiguous();
    auto hostSlots = th::empty({rows, maxBlocksPerRow}, tableHostSlots.options());
    auto commitGens = th::empty({rows, maxBlocksPerRow}, tableCommitGen.options());
    auto blockStatus = th::empty({rows, maxBlocksPerRow}, blocks.options().dtype(torch::kUInt8));
    auto rowStatus = th::empty({rows}, blocks.options().dtype(torch::kUInt8));

    tk::invokeHisparseResolveBlocksToHostSlots(rowIds.data_ptr<int64_t>(), blocks.data_ptr<int32_t>(),
        counts.data_ptr<int32_t>(), tableIds.data_ptr<int64_t>(), tableHostSlots.data_ptr<int64_t>(),
        tableCommitGen.data_ptr<int64_t>(), tableAdmitted.data_ptr<bool>(), hostSlots.data_ptr<int64_t>(),
        commitGens.data_ptr<int64_t>(), blockStatus.data_ptr<uint8_t>(), rowStatus.data_ptr<uint8_t>(),
        static_cast<int32_t>(rows), static_cast<int32_t>(maxBlocksPerRow), static_cast<int32_t>(requestSlots),
        static_cast<int32_t>(maxBlocksPerRequest), at::cuda::getCurrentCUDAStream(blocks.get_device()).stream());
    return {hostSlots, commitGens, blockStatus, rowStatus};
}

std::tuple<th::Tensor, th::Tensor, th::Tensor, th::Tensor, th::Tensor, th::Tensor, th::Tensor> hisparsePlanHotSlots(
    th::Tensor const& hostSlots, th::Tensor const& commitGens, th::Tensor const& blockCounts,
    th::Tensor const& resolveRowStatus, th::Tensor const& hotHostSlot, th::Tensor const& hotCommitGen,
    th::Tensor const& hotLruTick, int64_t layerIdx, int64_t lruTickBase)
{
    TORCH_CHECK(hostSlots.is_cuda(), "host_slots must be a CUDA tensor");
    TORCH_CHECK(commitGens.is_cuda(), "commit_gens must be a CUDA tensor");
    TORCH_CHECK(blockCounts.is_cuda(), "block_counts must be a CUDA tensor");
    TORCH_CHECK(resolveRowStatus.is_cuda(), "resolve_row_status must be a CUDA tensor");
    TORCH_CHECK(hotHostSlot.is_cuda(), "hot_host_slot must be a CUDA tensor");
    TORCH_CHECK(hotCommitGen.is_cuda(), "hot_commit_gen must be a CUDA tensor");
    TORCH_CHECK(hotLruTick.is_cuda(), "hot_lru_tick must be a CUDA tensor");
    TORCH_CHECK(hostSlots.scalar_type() == torch::kInt64, "host_slots must be int64");
    TORCH_CHECK(commitGens.scalar_type() == torch::kInt64, "commit_gens must be int64");
    TORCH_CHECK(blockCounts.scalar_type() == torch::kInt32, "block_counts must be int32");
    TORCH_CHECK(resolveRowStatus.scalar_type() == torch::kUInt8, "resolve_row_status must be uint8");
    TORCH_CHECK(hotHostSlot.scalar_type() == torch::kInt64, "hot_host_slot must be int64");
    TORCH_CHECK(hotCommitGen.scalar_type() == torch::kInt64, "hot_commit_gen must be int64");
    TORCH_CHECK(hotLruTick.scalar_type() == torch::kInt64, "hot_lru_tick must be int64");
    TORCH_CHECK(hostSlots.dim() == 2, "host_slots must have shape [rows, max_blocks_per_row]");
    TORCH_CHECK(commitGens.dim() == 2, "commit_gens must have shape [rows, max_blocks_per_row]");
    TORCH_CHECK(blockCounts.dim() == 1, "block_counts must have shape [rows]");
    TORCH_CHECK(resolveRowStatus.dim() == 1, "resolve_row_status must have shape [rows]");
    TORCH_CHECK(hotHostSlot.dim() == 2, "hot_host_slot must have shape [num_layers, hot_capacity]");
    TORCH_CHECK(hotCommitGen.dim() == 2, "hot_commit_gen must have shape [num_layers, hot_capacity]");
    TORCH_CHECK(hotLruTick.dim() == 2, "hot_lru_tick must have shape [num_layers, hot_capacity]");

    int64_t const rows = hostSlots.size(0);
    int64_t const maxBlocksPerRow = hostSlots.size(1);
    int64_t const numLayers = hotHostSlot.size(0);
    int64_t const hotCapacity = hotHostSlot.size(1);
    TORCH_CHECK(commitGens.size(0) == rows && commitGens.size(1) == maxBlocksPerRow,
        "commit_gens shape must match host_slots");
    TORCH_CHECK(blockCounts.size(0) == rows, "block_counts rows mismatch: got ", blockCounts.size(0),
        ", expected ", rows);
    TORCH_CHECK(resolveRowStatus.size(0) == rows, "resolve_row_status rows mismatch: got ",
        resolveRowStatus.size(0), ", expected ", rows);
    TORCH_CHECK(hotCommitGen.size(0) == numLayers && hotCommitGen.size(1) == hotCapacity,
        "hot_commit_gen shape must match hot_host_slot");
    TORCH_CHECK(hotLruTick.size(0) == numLayers && hotLruTick.size(1) == hotCapacity,
        "hot_lru_tick shape must match hot_host_slot");
    TORCH_CHECK(layerIdx >= 0 && layerIdx < numLayers, "layer_idx out of range: ", layerIdx,
        " for num_layers=", numLayers);
    TORCH_CHECK(maxBlocksPerRow > 0 && maxBlocksPerRow <= std::numeric_limits<int32_t>::max(),
        "max_blocks_per_row must be positive int32-sized, got ", maxBlocksPerRow);
    TORCH_CHECK(numLayers > 0 && numLayers <= std::numeric_limits<int32_t>::max(),
        "num_layers must be positive int32-sized, got ", numLayers);
    TORCH_CHECK(hotCapacity > 0 && hotCapacity <= 4096,
        "hot_capacity must be in [1, 4096], got ", hotCapacity);

    c10::cuda::CUDAGuard guard(hostSlots.device());
    TORCH_CHECK(commitGens.get_device() == hostSlots.get_device(),
        "commit_gens must be on the same CUDA device as host_slots");
    TORCH_CHECK(blockCounts.get_device() == hostSlots.get_device(),
        "block_counts must be on the same CUDA device as host_slots");
    TORCH_CHECK(resolveRowStatus.get_device() == hostSlots.get_device(),
        "resolve_row_status must be on the same CUDA device as host_slots");
    TORCH_CHECK(hotHostSlot.get_device() == hostSlots.get_device(),
        "hot_host_slot must be on the same CUDA device as host_slots");
    TORCH_CHECK(hotCommitGen.get_device() == hostSlots.get_device(),
        "hot_commit_gen must be on the same CUDA device as host_slots");
    TORCH_CHECK(hotLruTick.get_device() == hostSlots.get_device(),
        "hot_lru_tick must be on the same CUDA device as host_slots");

    auto host = hostSlots.contiguous();
    auto gen = commitGens.contiguous();
    auto counts = blockCounts.contiguous();
    auto resolveStatus = resolveRowStatus.contiguous();
    auto hotHost = hotHostSlot.contiguous();
    auto hotGen = hotCommitGen.contiguous();
    auto hotLru = hotLruTick.contiguous();
    auto plannedHotSlots = th::empty_like(host);
    auto plannedLruTick = th::empty_like(host);
    auto missHostSlots = th::empty_like(host);
    auto missHotSlots = th::empty_like(host);
    auto missCounts = th::empty({rows}, counts.options());
    auto hitFlags = th::empty({rows, maxBlocksPerRow}, host.options().dtype(torch::kUInt8));
    auto rowStatus = th::empty({rows}, host.options().dtype(torch::kUInt8));

    tk::invokeHisparsePlanHotSlots(host.data_ptr<int64_t>(), gen.data_ptr<int64_t>(), counts.data_ptr<int32_t>(),
        resolveStatus.data_ptr<uint8_t>(), hotHost.data_ptr<int64_t>(), hotGen.data_ptr<int64_t>(),
        hotLru.data_ptr<int64_t>(), plannedHotSlots.data_ptr<int64_t>(), plannedLruTick.data_ptr<int64_t>(),
        missHostSlots.data_ptr<int64_t>(), missHotSlots.data_ptr<int64_t>(), missCounts.data_ptr<int32_t>(),
        hitFlags.data_ptr<uint8_t>(), rowStatus.data_ptr<uint8_t>(), static_cast<int32_t>(rows),
        static_cast<int32_t>(maxBlocksPerRow), static_cast<int32_t>(numLayers), static_cast<int32_t>(hotCapacity),
        static_cast<int32_t>(layerIdx), lruTickBase, at::cuda::getCurrentCUDAStream(host.get_device()).stream());
    return {plannedHotSlots, plannedLruTick, missHostSlots, missHotSlots, missCounts, hitFlags, rowStatus};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "hisparse_topk_to_block_positions(Tensor topk_indices, int tokens_per_block, int max_blocks_per_row) "
        "-> (Tensor, Tensor, Tensor)");
    m.def(
        "hisparse_resolve_blocks_to_host_slots(Tensor row_request_ids, Tensor block_positions, Tensor block_counts, "
        "Tensor request_ids, Tensor request_block_host_slots, Tensor request_block_commit_gen, "
        "Tensor request_admitted) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "hisparse_plan_hot_slots(Tensor host_slots, Tensor commit_gens, Tensor block_counts, "
        "Tensor resolve_row_status, Tensor hot_host_slot, Tensor hot_commit_gen, Tensor hot_lru_tick, "
        "int layer_idx, int lru_tick_base) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("hisparse_topk_to_block_positions", &tensorrt_llm::torch_ext::hisparseTopkToBlockPositions);
    m.impl("hisparse_resolve_blocks_to_host_slots", &tensorrt_llm::torch_ext::hisparseResolveBlocksToHostSlots);
    m.impl("hisparse_plan_hot_slots", &tensorrt_llm::torch_ext::hisparsePlanHotSlots);
}

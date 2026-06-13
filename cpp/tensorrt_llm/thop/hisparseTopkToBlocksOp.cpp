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

std::tuple<th::Tensor, th::Tensor> hisparseClassifyResidentBlocks(th::Tensor const& blockPositions,
    th::Tensor const& blockCounts, th::Tensor const& rowKvLens, th::Tensor const& tailBlockPos,
    th::Tensor const& tailValid, int64_t tokensPerBlock, int64_t sinkBlocks)
{
    TORCH_CHECK(blockPositions.is_cuda(), "block_positions must be a CUDA tensor");
    TORCH_CHECK(blockCounts.is_cuda(), "block_counts must be a CUDA tensor");
    TORCH_CHECK(rowKvLens.is_cuda(), "row_kv_lens must be a CUDA tensor");
    TORCH_CHECK(tailBlockPos.is_cuda(), "tail_block_pos must be a CUDA tensor");
    TORCH_CHECK(tailValid.is_cuda(), "tail_valid must be a CUDA tensor");
    TORCH_CHECK(blockPositions.scalar_type() == torch::kInt32, "block_positions must be int32");
    TORCH_CHECK(blockCounts.scalar_type() == torch::kInt32, "block_counts must be int32");
    TORCH_CHECK(rowKvLens.scalar_type() == torch::kInt64, "row_kv_lens must be int64");
    TORCH_CHECK(tailBlockPos.scalar_type() == torch::kInt32, "tail_block_pos must be int32");
    TORCH_CHECK(tailValid.scalar_type() == torch::kBool, "tail_valid must be bool");
    TORCH_CHECK(blockPositions.dim() == 2, "block_positions must have shape [rows, max_blocks_per_row]");
    TORCH_CHECK(blockCounts.dim() == 1, "block_counts must have shape [rows]");
    TORCH_CHECK(rowKvLens.dim() == 1, "row_kv_lens must have shape [rows]");
    TORCH_CHECK(tailBlockPos.dim() == 1, "tail_block_pos must have shape [rows]");
    TORCH_CHECK(tailValid.dim() == 1, "tail_valid must have shape [rows]");

    int64_t const rows = blockPositions.size(0);
    int64_t const maxBlocksPerRow = blockPositions.size(1);
    TORCH_CHECK(rows <= std::numeric_limits<int32_t>::max(),
        "rows must fit int32 for hisparse_classify_resident_blocks, got ", rows);
    TORCH_CHECK(maxBlocksPerRow > 0 && maxBlocksPerRow <= std::numeric_limits<int32_t>::max(),
        "max_blocks_per_row must be positive int32-sized, got ", maxBlocksPerRow);
    TORCH_CHECK(blockCounts.size(0) == rows, "block_counts rows mismatch");
    TORCH_CHECK(rowKvLens.size(0) == rows, "row_kv_lens rows mismatch");
    TORCH_CHECK(tailBlockPos.size(0) == rows, "tail_block_pos rows mismatch");
    TORCH_CHECK(tailValid.size(0) == rows, "tail_valid rows mismatch");
    TORCH_CHECK(tokensPerBlock > 0 && tokensPerBlock <= std::numeric_limits<int32_t>::max(),
        "tokens_per_block must be positive int32-sized, got ", tokensPerBlock);
    TORCH_CHECK(sinkBlocks >= 0 && sinkBlocks <= std::numeric_limits<int32_t>::max(),
        "sink_blocks must be non-negative int32-sized, got ", sinkBlocks);

    c10::cuda::CUDAGuard guard(blockPositions.device());
    TORCH_CHECK(blockCounts.get_device() == blockPositions.get_device(),
        "block_counts must be on the same CUDA device as block_positions");
    TORCH_CHECK(rowKvLens.get_device() == blockPositions.get_device(),
        "row_kv_lens must be on the same CUDA device as block_positions");
    TORCH_CHECK(tailBlockPos.get_device() == blockPositions.get_device(),
        "tail_block_pos must be on the same CUDA device as block_positions");
    TORCH_CHECK(tailValid.get_device() == blockPositions.get_device(),
        "tail_valid must be on the same CUDA device as block_positions");

    auto blocks = blockPositions.contiguous();
    auto counts = blockCounts.contiguous();
    auto kvLens = rowKvLens.contiguous();
    auto tailPos = tailBlockPos.contiguous();
    auto tail = tailValid.contiguous();
    auto residentBlockFlags = th::empty({rows, maxBlocksPerRow}, blocks.options().dtype(torch::kUInt8));
    auto rowStatus = th::empty({rows}, blocks.options().dtype(torch::kUInt8));

    tk::invokeHisparseClassifyResidentBlocks(blocks.data_ptr<int32_t>(), counts.data_ptr<int32_t>(),
        kvLens.data_ptr<int64_t>(), tailPos.data_ptr<int32_t>(), tail.data_ptr<bool>(),
        residentBlockFlags.data_ptr<uint8_t>(), rowStatus.data_ptr<uint8_t>(), static_cast<int32_t>(rows),
        static_cast<int32_t>(maxBlocksPerRow), static_cast<int32_t>(tokensPerBlock), static_cast<int32_t>(sinkBlocks),
        at::cuda::getCurrentCUDAStream(blocks.get_device()).stream());
    return {residentBlockFlags, rowStatus};
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

std::tuple<th::Tensor, th::Tensor, th::Tensor, th::Tensor, th::Tensor> hisparseCompactMissSchedule(
    th::Tensor const& missHostSlots, th::Tensor const& missHotSlots, th::Tensor const& missCounts,
    th::Tensor const& planRowStatus)
{
    TORCH_CHECK(missHostSlots.is_cuda(), "miss_host_slots must be a CUDA tensor");
    TORCH_CHECK(missHotSlots.is_cuda(), "miss_hot_slots must be a CUDA tensor");
    TORCH_CHECK(missCounts.is_cuda(), "miss_counts must be a CUDA tensor");
    TORCH_CHECK(planRowStatus.is_cuda(), "plan_row_status must be a CUDA tensor");
    TORCH_CHECK(missHostSlots.scalar_type() == torch::kInt64, "miss_host_slots must be int64");
    TORCH_CHECK(missHotSlots.scalar_type() == torch::kInt64, "miss_hot_slots must be int64");
    TORCH_CHECK(missCounts.scalar_type() == torch::kInt32, "miss_counts must be int32");
    TORCH_CHECK(planRowStatus.scalar_type() == torch::kUInt8, "plan_row_status must be uint8");
    TORCH_CHECK(missHostSlots.dim() == 2, "miss_host_slots must have shape [rows, max_blocks_per_row]");
    TORCH_CHECK(missHotSlots.dim() == 2, "miss_hot_slots must have shape [rows, max_blocks_per_row]");
    TORCH_CHECK(missCounts.dim() == 1, "miss_counts must have shape [rows]");
    TORCH_CHECK(planRowStatus.dim() == 1, "plan_row_status must have shape [rows]");

    int64_t const rows = missHostSlots.size(0);
    int64_t const maxBlocksPerRow = missHostSlots.size(1);
    TORCH_CHECK(missHotSlots.sizes() == missHostSlots.sizes(), "miss_hot_slots shape must match miss_host_slots");
    TORCH_CHECK(missCounts.size(0) == rows, "miss_counts rows mismatch: got ", missCounts.size(0),
        ", expected ", rows);
    TORCH_CHECK(planRowStatus.size(0) == rows, "plan_row_status rows mismatch: got ", planRowStatus.size(0),
        ", expected ", rows);
    TORCH_CHECK(maxBlocksPerRow > 0 && maxBlocksPerRow <= std::numeric_limits<int32_t>::max(),
        "max_blocks_per_row must be positive int32-sized, got ", maxBlocksPerRow);
    TORCH_CHECK(rows >= 0 && rows <= std::numeric_limits<int32_t>::max(), "rows must be int32-sized, got ", rows);

    c10::cuda::CUDAGuard guard(missHostSlots.device());
    TORCH_CHECK(missHotSlots.get_device() == missHostSlots.get_device(),
        "miss_hot_slots must be on the same CUDA device as miss_host_slots");
    TORCH_CHECK(missCounts.get_device() == missHostSlots.get_device(),
        "miss_counts must be on the same CUDA device as miss_host_slots");
    TORCH_CHECK(planRowStatus.get_device() == missHostSlots.get_device(),
        "plan_row_status must be on the same CUDA device as miss_host_slots");

    auto missHost = missHostSlots.contiguous();
    auto missHot = missHotSlots.contiguous();
    auto counts = missCounts.contiguous();
    auto planStatus = planRowStatus.contiguous();
    auto compactHost = th::empty({rows * maxBlocksPerRow}, missHost.options());
    auto compactHot = th::empty({rows * maxBlocksPerRow}, missHot.options());
    auto compactRows = th::empty({rows * maxBlocksPerRow}, counts.options());
    auto copyCount = th::zeros({1}, counts.options());
    auto rowStatus = th::empty({rows}, missHost.options().dtype(torch::kUInt8));

    tk::invokeHisparseCompactMissSchedule(missHost.data_ptr<int64_t>(), missHot.data_ptr<int64_t>(),
        counts.data_ptr<int32_t>(), planStatus.data_ptr<uint8_t>(), compactHost.data_ptr<int64_t>(),
        compactHot.data_ptr<int64_t>(), compactRows.data_ptr<int32_t>(), copyCount.data_ptr<int32_t>(),
        rowStatus.data_ptr<uint8_t>(), static_cast<int32_t>(rows), static_cast<int32_t>(maxBlocksPerRow),
        at::cuda::getCurrentCUDAStream(missHost.get_device()).stream());
    return {compactHost, compactHot, compactRows, copyCount, rowStatus};
}

th::Tensor hisparseCommitHotSlots(th::Tensor const& hostSlots, th::Tensor const& commitGens,
    th::Tensor const& plannedHotSlots, th::Tensor const& plannedLruTick, th::Tensor const& blockCounts,
    th::Tensor const& planRowStatus, th::Tensor const& hotHostSlot, th::Tensor const& hotCommitGen,
    th::Tensor const& hotLruTick, int64_t layerIdx)
{
    TORCH_CHECK(hostSlots.is_cuda(), "host_slots must be a CUDA tensor");
    TORCH_CHECK(commitGens.is_cuda(), "commit_gens must be a CUDA tensor");
    TORCH_CHECK(plannedHotSlots.is_cuda(), "planned_hot_slots must be a CUDA tensor");
    TORCH_CHECK(plannedLruTick.is_cuda(), "planned_lru_tick must be a CUDA tensor");
    TORCH_CHECK(blockCounts.is_cuda(), "block_counts must be a CUDA tensor");
    TORCH_CHECK(planRowStatus.is_cuda(), "plan_row_status must be a CUDA tensor");
    TORCH_CHECK(hotHostSlot.is_cuda(), "hot_host_slot must be a CUDA tensor");
    TORCH_CHECK(hotCommitGen.is_cuda(), "hot_commit_gen must be a CUDA tensor");
    TORCH_CHECK(hotLruTick.is_cuda(), "hot_lru_tick must be a CUDA tensor");
    TORCH_CHECK(hostSlots.scalar_type() == torch::kInt64, "host_slots must be int64");
    TORCH_CHECK(commitGens.scalar_type() == torch::kInt64, "commit_gens must be int64");
    TORCH_CHECK(plannedHotSlots.scalar_type() == torch::kInt64, "planned_hot_slots must be int64");
    TORCH_CHECK(plannedLruTick.scalar_type() == torch::kInt64, "planned_lru_tick must be int64");
    TORCH_CHECK(blockCounts.scalar_type() == torch::kInt32, "block_counts must be int32");
    TORCH_CHECK(planRowStatus.scalar_type() == torch::kUInt8, "plan_row_status must be uint8");
    TORCH_CHECK(hotHostSlot.scalar_type() == torch::kInt64, "hot_host_slot must be int64");
    TORCH_CHECK(hotCommitGen.scalar_type() == torch::kInt64, "hot_commit_gen must be int64");
    TORCH_CHECK(hotLruTick.scalar_type() == torch::kInt64, "hot_lru_tick must be int64");
    TORCH_CHECK(hostSlots.dim() == 2, "host_slots must have shape [rows, max_blocks_per_row]");
    TORCH_CHECK(commitGens.dim() == 2, "commit_gens must have shape [rows, max_blocks_per_row]");
    TORCH_CHECK(plannedHotSlots.dim() == 2, "planned_hot_slots must have shape [rows, max_blocks_per_row]");
    TORCH_CHECK(plannedLruTick.dim() == 2, "planned_lru_tick must have shape [rows, max_blocks_per_row]");
    TORCH_CHECK(blockCounts.dim() == 1, "block_counts must have shape [rows]");
    TORCH_CHECK(planRowStatus.dim() == 1, "plan_row_status must have shape [rows]");
    TORCH_CHECK(hotHostSlot.dim() == 2, "hot_host_slot must have shape [num_layers, hot_capacity]");
    TORCH_CHECK(hotCommitGen.dim() == 2, "hot_commit_gen must have shape [num_layers, hot_capacity]");
    TORCH_CHECK(hotLruTick.dim() == 2, "hot_lru_tick must have shape [num_layers, hot_capacity]");
    TORCH_CHECK(hotHostSlot.is_contiguous(), "hot_host_slot must be contiguous for in-place metadata commit");
    TORCH_CHECK(hotCommitGen.is_contiguous(), "hot_commit_gen must be contiguous for in-place metadata commit");
    TORCH_CHECK(hotLruTick.is_contiguous(), "hot_lru_tick must be contiguous for in-place metadata commit");

    int64_t const rows = hostSlots.size(0);
    int64_t const maxBlocksPerRow = hostSlots.size(1);
    int64_t const numLayers = hotHostSlot.size(0);
    int64_t const hotCapacity = hotHostSlot.size(1);
    TORCH_CHECK(commitGens.sizes() == hostSlots.sizes(), "commit_gens shape must match host_slots");
    TORCH_CHECK(plannedHotSlots.sizes() == hostSlots.sizes(), "planned_hot_slots shape must match host_slots");
    TORCH_CHECK(plannedLruTick.sizes() == hostSlots.sizes(), "planned_lru_tick shape must match host_slots");
    TORCH_CHECK(blockCounts.size(0) == rows, "block_counts rows mismatch: got ", blockCounts.size(0),
        ", expected ", rows);
    TORCH_CHECK(planRowStatus.size(0) == rows, "plan_row_status rows mismatch: got ", planRowStatus.size(0),
        ", expected ", rows);
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
    TORCH_CHECK(hotCapacity > 0 && hotCapacity <= std::numeric_limits<int32_t>::max(),
        "hot_capacity must be positive int32-sized, got ", hotCapacity);

    c10::cuda::CUDAGuard guard(hostSlots.device());
    TORCH_CHECK(commitGens.get_device() == hostSlots.get_device(),
        "commit_gens must be on the same CUDA device as host_slots");
    TORCH_CHECK(plannedHotSlots.get_device() == hostSlots.get_device(),
        "planned_hot_slots must be on the same CUDA device as host_slots");
    TORCH_CHECK(plannedLruTick.get_device() == hostSlots.get_device(),
        "planned_lru_tick must be on the same CUDA device as host_slots");
    TORCH_CHECK(blockCounts.get_device() == hostSlots.get_device(),
        "block_counts must be on the same CUDA device as host_slots");
    TORCH_CHECK(planRowStatus.get_device() == hostSlots.get_device(),
        "plan_row_status must be on the same CUDA device as host_slots");
    TORCH_CHECK(hotHostSlot.get_device() == hostSlots.get_device(),
        "hot_host_slot must be on the same CUDA device as host_slots");
    TORCH_CHECK(hotCommitGen.get_device() == hostSlots.get_device(),
        "hot_commit_gen must be on the same CUDA device as host_slots");
    TORCH_CHECK(hotLruTick.get_device() == hostSlots.get_device(),
        "hot_lru_tick must be on the same CUDA device as host_slots");

    auto host = hostSlots.contiguous();
    auto gen = commitGens.contiguous();
    auto plannedHot = plannedHotSlots.contiguous();
    auto plannedLru = plannedLruTick.contiguous();
    auto counts = blockCounts.contiguous();
    auto planStatus = planRowStatus.contiguous();
    auto rowStatus = th::empty({rows}, host.options().dtype(torch::kUInt8));

    tk::invokeHisparseCommitHotSlots(host.data_ptr<int64_t>(), gen.data_ptr<int64_t>(),
        plannedHot.data_ptr<int64_t>(), plannedLru.data_ptr<int64_t>(), counts.data_ptr<int32_t>(),
        planStatus.data_ptr<uint8_t>(), hotHostSlot.data_ptr<int64_t>(), hotCommitGen.data_ptr<int64_t>(),
        hotLruTick.data_ptr<int64_t>(), rowStatus.data_ptr<uint8_t>(), static_cast<int32_t>(rows),
        static_cast<int32_t>(maxBlocksPerRow), static_cast<int32_t>(numLayers), static_cast<int32_t>(hotCapacity),
        static_cast<int32_t>(layerIdx), at::cuda::getCurrentCUDAStream(host.get_device()).stream());
    return rowStatus;
}

std::tuple<th::Tensor, th::Tensor> hisparseBuildHotIndices(th::Tensor const& topkIndices,
    th::Tensor const& blockPositions, th::Tensor const& plannedHotSlots, th::Tensor const& blockCounts,
    th::Tensor const& commitRowStatus, int64_t hotCapacity, int64_t tokensPerBlock, int64_t strideFactor,
    int64_t layerIdx)
{
    TORCH_CHECK(topkIndices.is_cuda(), "topk_indices must be a CUDA tensor");
    TORCH_CHECK(blockPositions.is_cuda(), "block_positions must be a CUDA tensor");
    TORCH_CHECK(plannedHotSlots.is_cuda(), "planned_hot_slots must be a CUDA tensor");
    TORCH_CHECK(blockCounts.is_cuda(), "block_counts must be a CUDA tensor");
    TORCH_CHECK(commitRowStatus.is_cuda(), "commit_row_status must be a CUDA tensor");
    TORCH_CHECK(topkIndices.scalar_type() == torch::kInt32, "topk_indices must be int32");
    TORCH_CHECK(blockPositions.scalar_type() == torch::kInt32, "block_positions must be int32");
    TORCH_CHECK(plannedHotSlots.scalar_type() == torch::kInt64, "planned_hot_slots must be int64");
    TORCH_CHECK(blockCounts.scalar_type() == torch::kInt32, "block_counts must be int32");
    TORCH_CHECK(commitRowStatus.scalar_type() == torch::kUInt8, "commit_row_status must be uint8");
    TORCH_CHECK(topkIndices.dim() == 2, "topk_indices must have shape [rows, index_topk]");
    TORCH_CHECK(blockPositions.dim() == 2, "block_positions must have shape [rows, max_blocks_per_row]");
    TORCH_CHECK(plannedHotSlots.dim() == 2, "planned_hot_slots must have shape [rows, max_blocks_per_row]");
    TORCH_CHECK(blockCounts.dim() == 1, "block_counts must have shape [rows]");
    TORCH_CHECK(commitRowStatus.dim() == 1, "commit_row_status must have shape [rows]");

    int64_t const rows = topkIndices.size(0);
    int64_t const indexTopK = topkIndices.size(1);
    int64_t const maxBlocksPerRow = blockPositions.size(1);
    TORCH_CHECK(blockPositions.size(0) == rows, "block_positions rows mismatch: got ", blockPositions.size(0),
        ", expected ", rows);
    TORCH_CHECK(plannedHotSlots.size(0) == rows && plannedHotSlots.size(1) == maxBlocksPerRow,
        "planned_hot_slots shape must match block_positions");
    TORCH_CHECK(blockCounts.size(0) == rows, "block_counts rows mismatch: got ", blockCounts.size(0),
        ", expected ", rows);
    TORCH_CHECK(commitRowStatus.size(0) == rows, "commit_row_status rows mismatch: got ", commitRowStatus.size(0),
        ", expected ", rows);
    TORCH_CHECK(indexTopK > 0 && indexTopK <= std::numeric_limits<int32_t>::max(),
        "index_topk must be positive int32-sized, got ", indexTopK);
    TORCH_CHECK(maxBlocksPerRow > 0 && maxBlocksPerRow <= std::numeric_limits<int32_t>::max(),
        "max_blocks_per_row must be positive int32-sized, got ", maxBlocksPerRow);
    TORCH_CHECK(hotCapacity > 0 && hotCapacity <= std::numeric_limits<int32_t>::max(),
        "hot_capacity must be positive int32-sized, got ", hotCapacity);
    TORCH_CHECK(tokensPerBlock > 0 && tokensPerBlock <= std::numeric_limits<int32_t>::max(),
        "tokens_per_block must be positive int32-sized, got ", tokensPerBlock);
    TORCH_CHECK(strideFactor > 0 && strideFactor <= std::numeric_limits<int32_t>::max(),
        "stride_factor must be positive int32-sized, got ", strideFactor);
    TORCH_CHECK(layerIdx >= 0 && layerIdx <= std::numeric_limits<int32_t>::max(),
        "layer_idx must be non-negative int32-sized, got ", layerIdx);

    c10::cuda::CUDAGuard guard(topkIndices.device());
    TORCH_CHECK(blockPositions.get_device() == topkIndices.get_device(),
        "block_positions must be on the same CUDA device as topk_indices");
    TORCH_CHECK(plannedHotSlots.get_device() == topkIndices.get_device(),
        "planned_hot_slots must be on the same CUDA device as topk_indices");
    TORCH_CHECK(blockCounts.get_device() == topkIndices.get_device(),
        "block_counts must be on the same CUDA device as topk_indices");
    TORCH_CHECK(commitRowStatus.get_device() == topkIndices.get_device(),
        "commit_row_status must be on the same CUDA device as topk_indices");

    auto topk = topkIndices.contiguous();
    auto blocks = blockPositions.contiguous();
    auto hotSlots = plannedHotSlots.contiguous();
    auto counts = blockCounts.contiguous();
    auto commitStatus = commitRowStatus.contiguous();
    auto hotIndices = th::empty_like(topk);
    auto rowStatus = th::empty({rows}, topk.options().dtype(torch::kUInt8));

    tk::invokeHisparseBuildHotIndices(topk.data_ptr<int32_t>(), blocks.data_ptr<int32_t>(),
        hotSlots.data_ptr<int64_t>(), counts.data_ptr<int32_t>(), commitStatus.data_ptr<uint8_t>(),
        hotIndices.data_ptr<int32_t>(), rowStatus.data_ptr<uint8_t>(), static_cast<int32_t>(rows),
        static_cast<int32_t>(indexTopK), static_cast<int32_t>(maxBlocksPerRow), static_cast<int32_t>(hotCapacity),
        static_cast<int32_t>(tokensPerBlock), static_cast<int32_t>(strideFactor), static_cast<int32_t>(layerIdx),
        at::cuda::getCurrentCUDAStream(topk.get_device()).stream());
    return {hotIndices, rowStatus};
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
        "hisparse_classify_resident_blocks(Tensor block_positions, Tensor block_counts, Tensor row_kv_lens, "
        "Tensor tail_block_pos, Tensor tail_valid, int tokens_per_block, int sink_blocks) -> (Tensor, Tensor)");
    m.def(
        "hisparse_plan_hot_slots(Tensor host_slots, Tensor commit_gens, Tensor block_counts, "
        "Tensor resolve_row_status, Tensor hot_host_slot, Tensor hot_commit_gen, Tensor hot_lru_tick, "
        "int layer_idx, int lru_tick_base) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "hisparse_compact_miss_schedule(Tensor miss_host_slots, Tensor miss_hot_slots, Tensor miss_counts, "
        "Tensor plan_row_status) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "hisparse_commit_hot_slots(Tensor host_slots, Tensor commit_gens, Tensor planned_hot_slots, "
        "Tensor planned_lru_tick, Tensor block_counts, Tensor plan_row_status, Tensor(a!) hot_host_slot, "
        "Tensor(b!) hot_commit_gen, Tensor(c!) hot_lru_tick, int layer_idx) -> Tensor");
    m.def(
        "hisparse_build_hot_indices(Tensor topk_indices, Tensor block_positions, Tensor planned_hot_slots, "
        "Tensor block_counts, Tensor commit_row_status, int hot_capacity, int tokens_per_block, int stride_factor, "
        "int layer_idx) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("hisparse_topk_to_block_positions", &tensorrt_llm::torch_ext::hisparseTopkToBlockPositions);
    m.impl("hisparse_resolve_blocks_to_host_slots", &tensorrt_llm::torch_ext::hisparseResolveBlocksToHostSlots);
    m.impl("hisparse_classify_resident_blocks", &tensorrt_llm::torch_ext::hisparseClassifyResidentBlocks);
    m.impl("hisparse_plan_hot_slots", &tensorrt_llm::torch_ext::hisparsePlanHotSlots);
    m.impl("hisparse_compact_miss_schedule", &tensorrt_llm::torch_ext::hisparseCompactMissSchedule);
    m.impl("hisparse_commit_hot_slots", &tensorrt_llm::torch_ext::hisparseCommitHotSlots);
    m.impl("hisparse_build_hot_indices", &tensorrt_llm::torch_ext::hisparseBuildHotIndices);
}

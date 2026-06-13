/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#include "tensorrt_llm/kernels/flashMLA/sparse_mla_decode_kvarn_hot.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <limits>
#include <optional>
#include <tuple>

namespace th = torch;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace
{
constexpr int64_t kHeadQ = 128;
constexpr int64_t kDqk = 576;
constexpr int64_t kDv = 512;
constexpr int64_t kTokensPerBlock = 64;
constexpr int64_t kKvarnBits = 2;
constexpr int64_t kKvLoraRank = 512;
constexpr int64_t kQkRopeHeadDim = 64;

int32_t checkedInt32(int64_t value, char const* name)
{
    TORCH_CHECK(value >= std::numeric_limits<int32_t>::min() && value <= std::numeric_limits<int32_t>::max(),
        name, " does not fit int32: ", value);
    return static_cast<int32_t>(value);
}

int64_t expectedBdrBytesPerBlock(int64_t tokensPerBlock, int64_t kvLoraRank, int64_t qkRopeHeadDim, int64_t kvarnBits)
{
    TORCH_CHECK(kvarnBits == kKvarnBits, "production sparse MLA KVarN-hot decode requires kvarn_bits=2");
    TORCH_CHECK(kvLoraRank == kKvLoraRank, "production sparse MLA KVarN-hot decode requires kv_lora_rank=512");
    TORCH_CHECK(qkRopeHeadDim == kQkRopeHeadDim,
        "production sparse MLA KVarN-hot decode requires qk_rope_head_dim=64");
    TORCH_CHECK(tokensPerBlock == kTokensPerBlock,
        "production sparse MLA KVarN-hot decode requires tokens_per_block=64");
    TORCH_CHECK(kvLoraRank % 128 == 0, "production sparse MLA KVarN-hot decode requires 128-wide BDR subblocks");

    auto const numSubblocks = kvLoraRank / 128;
    auto const ckvBytes = tokensPerBlock * kvLoraRank * kvarnBits / 8;
    auto const scaleZpBytes = tokensPerBlock * 2 * numSubblocks * 2;
    auto const peBytes = tokensPerBlock * qkRopeHeadDim;
    return ckvBytes + scaleZpBytes + peBytes;
}

void checkCudaTensor(at::Tensor const& tensor, char const* name)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

void checkSameDevice(at::Tensor const& ref, at::Tensor const& tensor, char const* name)
{
    checkCudaTensor(tensor, name);
    TORCH_CHECK(tensor.get_device() == ref.get_device(), name, " must be on the same CUDA device as q");
}

void checkOptionalSameDevice(at::Tensor const& ref, std::optional<at::Tensor> const& tensor, char const* name)
{
    if (tensor.has_value())
    {
        checkSameDevice(ref, *tensor, name);
    }
}

void checkOptionalContiguous(std::optional<at::Tensor> const& tensor, char const* name)
{
    if (tensor.has_value())
    {
        TORCH_CHECK(tensor->is_contiguous(), name, " must be contiguous");
    }
}

bool anyResidentTensor(std::optional<at::Tensor> const& residentKvLens, std::optional<at::Tensor> const& residentReqIdx,
    std::optional<at::Tensor> const& residentRequestIds, std::optional<at::Tensor> const& residentKvPool,
    std::optional<at::Tensor> const& residentBlockTable, std::optional<at::Tensor> const& residentTailBlockPos,
    std::optional<at::Tensor> const& residentTailTokenCount, std::optional<at::Tensor> const& residentTailValid,
    std::optional<at::Tensor> const& requestTopkIndices)
{
    return residentKvLens.has_value() || residentReqIdx.has_value() || residentRequestIds.has_value()
        || residentKvPool.has_value() || residentBlockTable.has_value() || residentTailBlockPos.has_value()
        || residentTailTokenCount.has_value() || residentTailValid.has_value() || requestTopkIndices.has_value();
}

void checkResidentTokenAbi(at::Tensor const& q, at::Tensor const& indices, int64_t rows, int64_t tokensPerBlock,
    int64_t residentSinkTokens, int64_t residentSinkBlocks, std::optional<at::Tensor> const& residentKvLens,
    std::optional<at::Tensor> const& residentReqIdx, std::optional<at::Tensor> const& residentRequestIds,
    std::optional<at::Tensor> const& residentKvPool, std::optional<at::Tensor> const& residentBlockTable,
    std::optional<at::Tensor> const& residentTailBlockPos, std::optional<at::Tensor> const& residentTailTokenCount,
    std::optional<at::Tensor> const& residentTailValid, std::optional<at::Tensor> const& requestTopkIndices)
{
    auto const present = anyResidentTensor(residentKvLens, residentReqIdx, residentRequestIds, residentKvPool,
        residentBlockTable, residentTailBlockPos, residentTailTokenCount, residentTailValid, requestTopkIndices);
    if (!present)
    {
        TORCH_CHECK(residentSinkTokens == 0 && residentSinkBlocks == 0,
            "resident sink counts require the complete explicit_sink_tail_v1 tensor ABI");
        return;
    }
    TORCH_CHECK(residentKvLens.has_value() && residentReqIdx.has_value() && residentRequestIds.has_value()
            && residentKvPool.has_value() && residentBlockTable.has_value() && residentTailBlockPos.has_value()
            && residentTailTokenCount.has_value() && residentTailValid.has_value() && requestTopkIndices.has_value(),
        "explicit_sink_tail_v1 requires resident_kv_lens, resident_req_idx, resident_request_ids, resident_kv_pool, "
        "resident_block_table, resident_tail_block_pos, resident_tail_token_count, resident_tail_valid, "
        "and request_topk_indices");
    checkSameDevice(q, *requestTopkIndices, "request_topk_indices");
    checkSameDevice(q, *residentKvLens, "resident_kv_lens");
    checkSameDevice(q, *residentReqIdx, "resident_req_idx");
    checkSameDevice(q, *residentRequestIds, "resident_request_ids");
    checkSameDevice(q, *residentKvPool, "resident_kv_pool");
    checkSameDevice(q, *residentBlockTable, "resident_block_table");
    checkSameDevice(q, *residentTailBlockPos, "resident_tail_block_pos");
    checkSameDevice(q, *residentTailTokenCount, "resident_tail_token_count");
    checkSameDevice(q, *residentTailValid, "resident_tail_valid");
    TORCH_CHECK(requestTopkIndices->scalar_type() == at::ScalarType::Int, "request_topk_indices must be int32");
    TORCH_CHECK(residentKvLens->scalar_type() == at::ScalarType::Long, "resident_kv_lens must be int64");
    TORCH_CHECK(residentReqIdx->scalar_type() == at::ScalarType::Long, "resident_req_idx must be int64");
    TORCH_CHECK(residentRequestIds->scalar_type() == at::ScalarType::Long, "resident_request_ids must be int64");
    TORCH_CHECK(residentKvPool->scalar_type() == at::ScalarType::BFloat16
            || residentKvPool->scalar_type() == at::ScalarType::Half,
        "resident_kv_pool must be bf16 or fp16 normal decode KV");
    TORCH_CHECK(residentBlockTable->scalar_type() == at::ScalarType::Int, "resident_block_table must be int32");
    TORCH_CHECK(residentTailBlockPos->scalar_type() == at::ScalarType::Int, "resident_tail_block_pos must be int32");
    TORCH_CHECK(
        residentTailTokenCount->scalar_type() == at::ScalarType::Int, "resident_tail_token_count must be int32");
    TORCH_CHECK(residentTailValid->scalar_type() == at::ScalarType::Bool, "resident_tail_valid must be bool");
    TORCH_CHECK(requestTopkIndices->dim() == 3 && requestTopkIndices->sizes() == indices.sizes(),
        "request_topk_indices must have the same [batch, s_q, topk] shape as hot indices");
    TORCH_CHECK(residentKvLens->dim() == 1 && residentKvLens->size(0) == rows,
        "resident_kv_lens must have shape [batch * s_q]");
    TORCH_CHECK(residentReqIdx->dim() == 1 && residentReqIdx->size(0) == rows,
        "resident_req_idx must have shape [batch * s_q]");
    TORCH_CHECK(residentRequestIds->dim() == 1 && residentRequestIds->size(0) == rows,
        "resident_request_ids must have shape [batch * s_q]");
    TORCH_CHECK(residentKvPool->dim() == 3 && residentKvPool->size(1) == 1 && residentKvPool->size(2) == kDqk,
        "resident_kv_pool must have shape [global_tokens, 1, 576]");
    TORCH_CHECK(residentBlockTable->dim() == 2, "resident_block_table must have shape [seqs, blocks]");
    TORCH_CHECK(residentTailBlockPos->dim() == 1 && residentTailBlockPos->size(0) == rows,
        "resident_tail_block_pos must have shape [batch * s_q]");
    TORCH_CHECK(residentTailTokenCount->dim() == 1 && residentTailTokenCount->size(0) == rows,
        "resident_tail_token_count must have shape [batch * s_q]");
    TORCH_CHECK(residentTailValid->dim() == 1 && residentTailValid->size(0) == rows,
        "resident_tail_valid must have shape [batch * s_q]");
    TORCH_CHECK(residentSinkTokens >= 0 && residentSinkBlocks >= 0, "resident sink counts must be non-negative");
    TORCH_CHECK(residentSinkBlocks == residentSinkTokens / tokensPerBlock,
        "resident_sink_blocks must equal resident_sink_tokens / tokens_per_block for explicit_sink_tail_v1");
}

void checkHotPackedStrides(at::Tensor const& hotPacked, int64_t expectedBytes)
{
    TORCH_CHECK(hotPacked.stride(2) == 1, "hot_packed records must be byte-contiguous");
    TORCH_CHECK(hotPacked.stride(1) >= expectedBytes,
        "hot_packed slot stride must be at least the production BDR record bytes=", expectedBytes,
        " to prevent overlapping hot records; got ", hotPacked.stride(1));
    TORCH_CHECK(hotPacked.stride(0) >= hotPacked.size(1) * hotPacked.stride(1),
        "hot_packed layer stride must cover all hot slots to prevent overlapping layers; got stride(0)=",
        hotPacked.stride(0), ", required at least ", hotPacked.size(1) * hotPacked.stride(1));
}

} // namespace

std::tuple<th::Tensor, th::Tensor, th::Tensor, th::Tensor> sparse_mla_decode_kvarn_hot(th::Tensor const& q,
    th::Tensor const& hotPacked, th::Tensor const& indices, th::Tensor const& rowStatus,
    std::optional<th::Tensor> const& topkLength, std::optional<th::Tensor> const& attnSink, int64_t layerIdx,
    int64_t tokensPerBlock, int64_t strideFactor, int64_t kvarnBits, int64_t kvLoraRank, int64_t qkRopeHeadDim,
    double smScale, std::optional<th::Tensor> const& residentKvLens, std::optional<th::Tensor> const& residentReqIdx,
    std::optional<th::Tensor> const& residentRequestIds, std::optional<th::Tensor> const& residentKvPool,
    std::optional<th::Tensor> const& residentBlockTable, std::optional<th::Tensor> const& residentTailBlockPos,
    std::optional<th::Tensor> const& residentTailTokenCount, std::optional<th::Tensor> const& residentTailValid,
    int64_t residentSinkTokens, int64_t residentSinkBlocks, std::optional<th::Tensor> const& requestTopkIndices)
{
    checkCudaTensor(q, "q");
    checkSameDevice(q, hotPacked, "hot_packed");
    checkSameDevice(q, indices, "indices");
    checkSameDevice(q, rowStatus, "row_status");
    checkOptionalSameDevice(q, topkLength, "topk_length");
    checkOptionalSameDevice(q, attnSink, "attn_sink");

    TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16, "q must be bf16");
    TORCH_CHECK(hotPacked.scalar_type() == at::ScalarType::Byte, "hot_packed must be uint8 production BDR KVarN records");
    TORCH_CHECK(indices.scalar_type() == at::ScalarType::Int, "indices must be int32");
    TORCH_CHECK(rowStatus.scalar_type() == at::ScalarType::Byte, "row_status must be uint8");
    if (topkLength.has_value())
    {
        TORCH_CHECK(topkLength->scalar_type() == at::ScalarType::Int, "topk_length must be int32");
    }
    if (attnSink.has_value())
    {
        TORCH_CHECK(attnSink->scalar_type() == at::ScalarType::Float, "attn_sink must be fp32");
    }

    TORCH_CHECK(q.dim() == 4, "q must have shape [batch, s_q, h_q, d_qk]");
    TORCH_CHECK(hotPacked.dim() == 3, "hot_packed must have shape [num_layers, hot_capacity, packed_bytes]");
    TORCH_CHECK(indices.dim() == 3, "indices must have shape [batch, s_q, topk]");
    TORCH_CHECK(rowStatus.dim() == 1, "row_status must have shape [batch * s_q]");

    auto const b = q.size(0);
    auto const sQ = q.size(1);
    auto const hQ = q.size(2);
    auto const dQk = q.size(3);
    auto const numLayers = hotPacked.size(0);
    auto const hotCapacity = hotPacked.size(1);
    auto const topK = indices.size(2);
    TORCH_CHECK(b > 0 && sQ > 0 && topK > 0, "batch, s_q, and topk must be positive");
    TORCH_CHECK(hQ == kHeadQ && dQk == kDqk, "q must use production V3.2 shape [*,*,128,576]");
    TORCH_CHECK(tokensPerBlock == kTokensPerBlock, "tokens_per_block must be 64");
    TORCH_CHECK(kvarnBits == 2, "kvarn_bits must be 2 for production kvarn_k2v2");
    TORCH_CHECK(kvLoraRank == kKvLoraRank && qkRopeHeadDim == kQkRopeHeadDim,
        "production dense MLA KVarN-hot decode requires kv_lora_rank=512 and qk_rope_head_dim=64");
    TORCH_CHECK(layerIdx >= 0 && layerIdx < numLayers, "layer_idx out of range");
    TORCH_CHECK(hotCapacity > 0, "hot_packed hot capacity must be positive");
    auto const expectedBytes = expectedBdrBytesPerBlock(tokensPerBlock, kvLoraRank, qkRopeHeadDim, kvarnBits);
    TORCH_CHECK(hotPacked.size(2) >= expectedBytes,
        "hot_packed record bytes are smaller than production BDR layout: got ", hotPacked.size(2),
        ", expected at least ", expectedBytes);
    TORCH_CHECK(strideFactor >= numLayers * tokensPerBlock,
        "stride_factor must cover all layer token ranges: got ", strideFactor,
        ", expected at least ", numLayers * tokensPerBlock);
    TORCH_CHECK(indices.size(0) == b && indices.size(1) == sQ, "indices batch/s_q dimensions must match q");
    TORCH_CHECK(rowStatus.size(0) == b * sQ, "row_status must have one entry per [batch, s_q] row");
    if (topkLength.has_value())
    {
        TORCH_CHECK(topkLength->dim() == 1 && (topkLength->size(0) == b || topkLength->size(0) == b * sQ),
            "topk_length must be [batch] or [batch * s_q]");
    }
    if (attnSink.has_value())
    {
        TORCH_CHECK(attnSink->dim() == 1 && attnSink->size(0) == hQ, "attn_sink must be [h_q]");
    }
    checkResidentTokenAbi(q, indices, b * sQ, tokensPerBlock, residentSinkTokens, residentSinkBlocks, residentKvLens,
        residentReqIdx, residentRequestIds, residentKvPool, residentBlockTable, residentTailBlockPos, residentTailTokenCount,
        residentTailValid, requestTopkIndices);

    TORCH_CHECK(q.stride(3) == 1, "q last dimension must be contiguous");
    checkHotPackedStrides(hotPacked, expectedBytes);
    TORCH_CHECK(indices.stride(2) == 1, "indices last dimension must be contiguous");
    TORCH_CHECK(rowStatus.is_contiguous(), "row_status must be contiguous");
    checkOptionalContiguous(topkLength, "topk_length");
    checkOptionalContiguous(attnSink, "attn_sink");
    checkOptionalContiguous(requestTopkIndices, "request_topk_indices");
    checkOptionalContiguous(residentKvLens, "resident_kv_lens");
    checkOptionalContiguous(residentReqIdx, "resident_req_idx");
    checkOptionalContiguous(residentRequestIds, "resident_request_ids");
    checkOptionalContiguous(residentKvPool, "resident_kv_pool");
    checkOptionalContiguous(residentBlockTable, "resident_block_table");
    checkOptionalContiguous(residentTailBlockPos, "resident_tail_block_pos");
    checkOptionalContiguous(residentTailTokenCount, "resident_tail_token_count");
    checkOptionalContiguous(residentTailValid, "resident_tail_valid");

    c10::cuda::CUDAGuard deviceGuard(q.device());
    auto out = th::empty({b, sQ, hQ, kDv}, q.options());
    auto lse = th::empty({b, sQ, hQ}, q.options().dtype(at::ScalarType::Float));
    auto metadata = th::empty({0, 0}, q.options().dtype(at::ScalarType::Int));
    auto splits = th::empty({0}, q.options().dtype(at::ScalarType::Int));

    tk::SparseMlaDecodeKvarnHotParams params{};
    params.q = q.data_ptr();
    params.hotPacked = reinterpret_cast<uint8_t*>(hotPacked.data_ptr());
    params.indices = reinterpret_cast<int32_t*>(indices.data_ptr());
    params.rowStatus = reinterpret_cast<uint8_t*>(rowStatus.data_ptr());
    params.requestTopkIndices
        = requestTopkIndices.has_value() ? reinterpret_cast<int32_t*>(requestTopkIndices->data_ptr()) : nullptr;
    params.topkLength = topkLength.has_value() ? reinterpret_cast<int32_t*>(topkLength->data_ptr()) : nullptr;
    params.residentKvLens
        = residentKvLens.has_value() ? reinterpret_cast<int64_t*>(residentKvLens->data_ptr()) : nullptr;
    params.residentReqIdx
        = residentReqIdx.has_value() ? reinterpret_cast<int64_t*>(residentReqIdx->data_ptr()) : nullptr;
    params.residentRequestIds
        = residentRequestIds.has_value() ? reinterpret_cast<int64_t*>(residentRequestIds->data_ptr()) : nullptr;
    params.residentKvPool = residentKvPool.has_value() ? residentKvPool->data_ptr() : nullptr;
    params.residentBlockTable
        = residentBlockTable.has_value() ? reinterpret_cast<int32_t*>(residentBlockTable->data_ptr()) : nullptr;
    params.residentTailBlockPos = residentTailBlockPos.has_value()
        ? reinterpret_cast<int32_t*>(residentTailBlockPos->data_ptr())
        : nullptr;
    params.residentTailTokenCount = residentTailTokenCount.has_value()
        ? reinterpret_cast<int32_t*>(residentTailTokenCount->data_ptr())
        : nullptr;
    params.residentTailValid
        = residentTailValid.has_value() ? reinterpret_cast<bool*>(residentTailValid->data_ptr()) : nullptr;
    params.attnSink = attnSink.has_value() ? reinterpret_cast<float*>(attnSink->data_ptr()) : nullptr;
    params.lse = reinterpret_cast<float*>(lse.data_ptr());
    params.out = out.data_ptr();
    params.b = checkedInt32(b, "batch");
    params.sQ = checkedInt32(sQ, "s_q");
    params.hQ = checkedInt32(hQ, "h_q");
    params.dQk = checkedInt32(dQk, "d_qk");
    params.dV = kDv;
    params.numLayers = checkedInt32(numLayers, "num_layers");
    params.hotCapacity = checkedInt32(hotCapacity, "hot_capacity");
    params.topK = checkedInt32(topK, "topk");
    params.topkLengthSize = topkLength.has_value() ? checkedInt32(topkLength->size(0), "topk_length.size(0)") : 0;
    params.residentRows = residentKvLens.has_value() ? checkedInt32(residentKvLens->size(0), "resident_rows") : 0;
    params.residentSinkTokens = checkedInt32(residentSinkTokens, "resident_sink_tokens");
    params.residentSinkBlocks = checkedInt32(residentSinkBlocks, "resident_sink_blocks");
    params.layerIdx = checkedInt32(layerIdx, "layer_idx");
    params.tokensPerBlock = checkedInt32(tokensPerBlock, "tokens_per_block");
    params.strideFactor = checkedInt32(strideFactor, "stride_factor");
    params.kvarnBits = checkedInt32(kvarnBits, "kvarn_bits");
    params.kvLoraRank = checkedInt32(kvLoraRank, "kv_lora_rank");
    params.qkRopeHeadDim = checkedInt32(qkRopeHeadDim, "qk_rope_head_dim");
    params.smScale = static_cast<float>(smScale);
    params.strideQB = q.stride(0);
    params.strideQSQ = q.stride(1);
    params.strideQHQ = q.stride(2);
    params.strideHotLayer = hotPacked.stride(0);
    params.strideHotSlot = hotPacked.stride(1);
    params.strideHotRecord = hotPacked.size(2);
    params.strideIndicesB = indices.stride(0);
    params.strideIndicesSQ = indices.stride(1);
    params.strideRequestTopkB = requestTopkIndices.has_value() ? requestTopkIndices->stride(0) : 0;
    params.strideRequestTopkSQ = requestTopkIndices.has_value() ? requestTopkIndices->stride(1) : 0;
    params.strideResidentKvPoolToken = residentKvPool.has_value() ? residentKvPool->stride(0) : 0;
    params.strideResidentKvPoolHead = residentKvPool.has_value() ? residentKvPool->stride(1) : 0;
    params.strideResidentBlockTableB = residentBlockTable.has_value() ? residentBlockTable->stride(0) : 0;
    params.strideResidentBlockTableBlock = residentBlockTable.has_value() ? residentBlockTable->stride(1) : 0;
    params.strideLseB = lse.stride(0);
    params.strideLseSQ = lse.stride(1);
    params.strideOB = out.stride(0);
    params.strideOSQ = out.stride(1);
    params.strideOHQ = out.stride(2);

    tk::invokeSparseMlaDecodeKvarnHot(params, at::cuda::getCurrentCUDAStream(q.get_device()));
    return {out, lse.transpose(1, 2), metadata, splits};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "sparse_mla_decode_kvarn_hot(Tensor q, Tensor hot_packed, Tensor indices, Tensor row_status, "
        "Tensor? topk_length=None, Tensor? attn_sink=None, int layer_idx=0, int tokens_per_block=64, "
        "int stride_factor=64, int kvarn_bits=2, int kv_lora_rank=512, int qk_rope_head_dim=64, "
        "float sm_scale=1., Tensor? resident_kv_lens=None, Tensor? resident_req_idx=None, "
        "Tensor? resident_request_ids=None, Tensor? resident_kv_pool=None, Tensor? resident_block_table=None, "
        "Tensor? resident_tail_block_pos=None, Tensor? resident_tail_token_count=None, "
        "Tensor? resident_tail_valid=None, int resident_sink_tokens=0, int resident_sink_blocks=0, "
        "Tensor? request_topk_indices=None) "
        "-> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("sparse_mla_decode_kvarn_hot", &tensorrt_llm::torch_ext::sparse_mla_decode_kvarn_hot);
}

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
constexpr int64_t kKvLoraRank = 512;
constexpr int64_t kQkRopeHeadDim = 64;

int32_t checkedInt32(int64_t value, char const* name)
{
    TORCH_CHECK(value >= std::numeric_limits<int32_t>::min() && value <= std::numeric_limits<int32_t>::max(),
        name, " does not fit int32: ", value);
    return static_cast<int32_t>(value);
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

} // namespace

std::tuple<th::Tensor, th::Tensor, th::Tensor, th::Tensor> sparse_mla_decode_kvarn_hot(th::Tensor const& q,
    th::Tensor const& hotPacked, th::Tensor const& indices, th::Tensor const& rowStatus,
    std::optional<th::Tensor> const& topkLength, std::optional<th::Tensor> const& attnSink, int64_t layerIdx,
    int64_t tokensPerBlock, int64_t strideFactor, int64_t kvarnBits, int64_t kvLoraRank, int64_t qkRopeHeadDim,
    double smScale)
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

    TORCH_CHECK(q.stride(3) == 1, "q last dimension must be contiguous");
    TORCH_CHECK(hotPacked.stride(2) == 1, "hot_packed records must be byte-contiguous");
    TORCH_CHECK(indices.stride(2) == 1, "indices last dimension must be contiguous");
    TORCH_CHECK(rowStatus.is_contiguous(), "row_status must be contiguous");
    checkOptionalContiguous(topkLength, "topk_length");
    checkOptionalContiguous(attnSink, "attn_sink");

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
    params.topkLength = topkLength.has_value() ? reinterpret_cast<int32_t*>(topkLength->data_ptr()) : nullptr;
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
        "float sm_scale=1.) -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("sparse_mla_decode_kvarn_hot", &tensorrt_llm::torch_ext::sparse_mla_decode_kvarn_hot);
}

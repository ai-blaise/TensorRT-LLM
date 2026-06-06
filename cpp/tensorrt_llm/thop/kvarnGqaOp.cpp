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
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace th = torch;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

void check_cuda_contiguous(th::Tensor const& tensor, char const* name)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}


struct PackedRecordStrides
{
    bool pageLayout;
    int64_t strideBlock;
    int64_t strideToken;
    int64_t strideHead;
    int64_t strideByte;
};

PackedRecordStrides get_packed_record_strides(th::Tensor const& packedRecords, int64_t numKvHeads, char const* opName)
{
    if (packedRecords.dim() == 3)
    {
        TORCH_CHECK(packedRecords.size(1) >= numKvHeads, opName, " packed_records [blocks, kv_heads, tile_bytes] has too few kv heads");
        TORCH_CHECK(packedRecords.size(2) >= tk::KVarNGqaK2V2G128Layout::kTileBytes,
            opName, " packed_records [blocks, kv_heads, tile_bytes] tile dim is too small");
        return PackedRecordStrides{false, packedRecords.stride(0), 0, packedRecords.stride(1), packedRecords.stride(2)};
    }
    if (packedRecords.dim() == 5)
    {
        TORCH_CHECK(packedRecords.size(1) >= 1, opName, " packed_records page layout needs a K plane at dim=1");
        TORCH_CHECK(packedRecords.size(2) == tk::KVarNGqaK2V2G128Layout::kGroupSize,
            opName, " packed_records page layout requires tokens_per_block=128");
        TORCH_CHECK(packedRecords.size(3) >= numKvHeads, opName, " packed_records page layout has too few kv heads");
        TORCH_CHECK(packedRecords.size(4) == tk::KVarNGqaK2V2G128Layout::kBytesPerTokenSlot,
            opName, " packed_records page layout requires bytes_per_token_slot=76");
        return PackedRecordStrides{true, packedRecords.stride(0), packedRecords.stride(2), packedRecords.stride(3),
            packedRecords.stride(4)};
    }
    TORCH_CHECK(false, opName,
        " packed_records must be [blocks, kv_heads, 9728] record layout or [blocks, planes, 128, kv_heads, 76] page layout");
}

void check_group_shape(int64_t headDim, int64_t groupSize)
{
    TORCH_CHECK(headDim == tk::KVarNGqaK2V2G128Layout::kHeadDim,
        "kvarn_gqa k2v2_g128 requires head_dim=128, got ", headDim);
    TORCH_CHECK(groupSize == tk::KVarNGqaK2V2G128Layout::kGroupSize,
        "kvarn_gqa k2v2_g128 requires group_size=128, got ", groupSize);
}


int64_t side_tokens(th::Tensor const& tensor, int64_t numQueries, int64_t numKvHeads, int64_t headDim, char const* name)
{
    if (tensor.numel() == 0)
    {
        return 0;
    }
    TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Half || tensor.scalar_type() == at::ScalarType::BFloat16,
        name, " must be fp16/bf16 when present");
    if (tensor.dim() == 3)
    {
        TORCH_CHECK(tensor.size(1) == numKvHeads && tensor.size(2) == headDim,
            name, " must be [tokens, num_kv_heads, head_dim]");
        return tensor.size(0);
    }
    if (tensor.dim() == 4)
    {
        TORCH_CHECK(tensor.size(0) == 1 || tensor.size(0) == numQueries,
            name, " batch dim must be 1 or num_queries");
        TORCH_CHECK(tensor.size(2) == numKvHeads && tensor.size(3) == headDim,
            name, " must be [batch, tokens, num_kv_heads, head_dim]");
        return tensor.size(1);
    }
    TORCH_CHECK(false, name, " must be empty, [tokens, num_kv_heads, head_dim], or [batch, tokens, num_kv_heads, head_dim]");
}

int64_t side_batch(th::Tensor const& tensor)
{
    return tensor.dim() == 4 ? tensor.size(0) : 1;
}

} // namespace

bool kvarn_gqa_backend_ready()
{
    return tk::kvarnGqaBackendReady();
}

void kvarn_gqa_store(th::Tensor const& k, th::Tensor const& v, th::Tensor const& packedRecords,
    th::Tensor const& blockIds, int64_t layerIdx, int64_t headDim, int64_t groupSize)
{
    check_cuda_contiguous(k, "k");
    check_cuda_contiguous(v, "v");
    check_cuda_contiguous(packedRecords, "packed_records");
    check_cuda_contiguous(blockIds, "block_ids");
    TORCH_CHECK(k.scalar_type() == at::ScalarType::Half || k.scalar_type() == at::ScalarType::BFloat16,
        "kvarn_gqa_store supports fp16/bf16 K input only");
    TORCH_CHECK(v.scalar_type() == k.scalar_type(), "kvarn_gqa_store K and V dtypes must match");
    TORCH_CHECK(packedRecords.scalar_type() == at::ScalarType::Byte, "packed_records must be uint8");
    TORCH_CHECK(blockIds.scalar_type() == at::ScalarType::Long, "block_ids must be int64");
    TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "k and v must be [num_blocks, group, num_kv_heads, head_dim]");
    TORCH_CHECK(k.sizes() == v.sizes(), "k and v shapes must match");
    check_group_shape(headDim, groupSize);
    TORCH_CHECK(k.size(1) == groupSize && k.size(3) == headDim,
        "kvarn_gqa_store input shape must match group_size/head_dim");
    TORCH_CHECK(blockIds.dim() == 1 && blockIds.size(0) == k.size(0),
        "block_ids must be [num_blocks] and match K/V blocks");
    auto strides = get_packed_record_strides(packedRecords, k.size(2), "kvarn_gqa_store");

    auto stream = at::cuda::getCurrentCUDAStream(k.get_device());
    tk::invokeKvarnGqaStoreK2V2G128(k.data_ptr(), v.data_ptr(), packedRecords.data_ptr<std::uint8_t>(),
        blockIds.data_ptr<std::int64_t>(), static_cast<int>(layerIdx), static_cast<int>(k.size(0)),
        static_cast<int>(k.size(2)), static_cast<int>(headDim), static_cast<int>(groupSize),
        k.scalar_type() == at::ScalarType::BFloat16, strides.pageLayout, strides.strideBlock, strides.strideToken,
        strides.strideHead, strides.strideByte, stream);
}

th::Tensor kvarn_gqa_decode(th::Tensor const& q, th::Tensor const& packedRecords, th::Tensor const& blockIds,
    th::Tensor const& sinkK, th::Tensor const& sinkV, th::Tensor const& tailK, th::Tensor const& tailV,
    th::Tensor const& seqLens, int64_t numHeads, int64_t numKvHeads, int64_t headDim, int64_t groupSize)
{
    check_cuda_contiguous(q, "q");
    check_cuda_contiguous(packedRecords, "packed_records");
    check_cuda_contiguous(blockIds, "block_ids");
    check_cuda_contiguous(sinkK, "sink_k");
    check_cuda_contiguous(sinkV, "sink_v");
    check_cuda_contiguous(tailK, "tail_k");
    check_cuda_contiguous(tailV, "tail_v");
    check_cuda_contiguous(seqLens, "seq_lens");
    TORCH_CHECK(q.scalar_type() == at::ScalarType::Half || q.scalar_type() == at::ScalarType::BFloat16,
        "kvarn_gqa_decode supports fp16/bf16 Q input only");
    TORCH_CHECK(packedRecords.scalar_type() == at::ScalarType::Byte, "packed_records must be uint8");
    TORCH_CHECK(blockIds.scalar_type() == at::ScalarType::Long, "block_ids must be int64");
    TORCH_CHECK(seqLens.scalar_type() == at::ScalarType::Int, "seq_lens must be int32");
    TORCH_CHECK(q.dim() == 3, "q must be [num_queries, num_heads, head_dim]");
    check_group_shape(headDim, groupSize);
    TORCH_CHECK(q.size(1) == numHeads && q.size(2) == headDim,
        "q shape must match num_heads/head_dim");
    TORCH_CHECK(numHeads % numKvHeads == 0, "num_heads must be divisible by num_kv_heads");
    TORCH_CHECK(seqLens.dim() == 1 && (seqLens.size(0) == 1 || seqLens.size(0) == q.size(0)),
        "seq_lens must be [1] or [num_queries]");
    TORCH_CHECK(sinkK.scalar_type() == sinkV.scalar_type(), "sink_k/sink_v dtypes must match");
    TORCH_CHECK(tailK.scalar_type() == tailV.scalar_type(), "tail_k/tail_v dtypes must match");
    TORCH_CHECK(sinkK.numel() == sinkV.numel(), "sink_k/sink_v must have matching element counts");
    TORCH_CHECK(tailK.numel() == tailV.numel(), "tail_k/tail_v must have matching element counts");
    auto sinkTokens = side_tokens(sinkK, q.size(0), numKvHeads, headDim, "sink_k");
    auto sinkVTokens = side_tokens(sinkV, q.size(0), numKvHeads, headDim, "sink_v");
    auto tailTokens = side_tokens(tailK, q.size(0), numKvHeads, headDim, "tail_k");
    auto tailVTokens = side_tokens(tailV, q.size(0), numKvHeads, headDim, "tail_v");
    TORCH_CHECK(sinkTokens == sinkVTokens, "sink_k/sink_v token counts must match");
    TORCH_CHECK(tailTokens == tailVTokens, "tail_k/tail_v token counts must match");
    auto strides = get_packed_record_strides(packedRecords, numKvHeads, "kvarn_gqa_decode");

    auto output = th::empty_like(q);
    auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
    tk::invokeKvarnGqaDecodeK2V2G128(q.data_ptr(), packedRecords.data_ptr<std::uint8_t>(),
        blockIds.data_ptr<std::int64_t>(), sinkK.data_ptr(), sinkV.data_ptr(), tailK.data_ptr(), tailV.data_ptr(),
        seqLens.data_ptr<std::int32_t>(), output.data_ptr(), static_cast<int>(q.size(0)), static_cast<int>(blockIds.size(0)),
        static_cast<int>(numHeads), static_cast<int>(numKvHeads), static_cast<int>(headDim), static_cast<int>(groupSize),
        q.scalar_type() == at::ScalarType::BFloat16, static_cast<int>(seqLens.size(0)), static_cast<int>(sinkTokens),
        static_cast<int>(side_batch(sinkK)), static_cast<int>(tailTokens), static_cast<int>(side_batch(tailK)),
        strides.pageLayout, strides.strideBlock, strides.strideToken, strides.strideHead, strides.strideByte, stream);
    return output;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("kvarn_gqa_backend_ready() -> bool");
    m.def(
        "kvarn_gqa_store(Tensor k, Tensor v, Tensor packed_records, Tensor block_ids, int layer_idx, int head_dim, "
        "int group_size) -> ()");
    m.def(
        "kvarn_gqa_decode(Tensor q, Tensor packed_records, Tensor block_ids, Tensor sink_k, Tensor sink_v, "
        "Tensor tail_k, Tensor tail_v, Tensor seq_lens, int num_heads, int num_kv_heads, int head_dim, "
        "int group_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
    m.impl("kvarn_gqa_backend_ready", &tensorrt_llm::torch_ext::kvarn_gqa_backend_ready);
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("kvarn_gqa_store", &tensorrt_llm::torch_ext::kvarn_gqa_store);
    m.impl("kvarn_gqa_decode", &tensorrt_llm::torch_ext::kvarn_gqa_decode);
}

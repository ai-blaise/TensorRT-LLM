/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#include "tensorrt_llm/kernels/hisparseKvarnHotRead.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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

int64_t expectedBdrBytesPerBlock(int64_t tokensPerBlock, int64_t kvLoraRank, int64_t qkRopeHeadDim, int64_t kvarnBits)
{
    TORCH_CHECK(kvarnBits == 2 || kvarnBits == 4, "kvarn_bits must be 2 or 4");
    TORCH_CHECK(kvLoraRank == 512, "production HiSparse KVarN-hot reader requires kv_lora_rank=512");
    TORCH_CHECK(qkRopeHeadDim == 64, "production HiSparse KVarN-hot reader requires qk_rope_head_dim=64");
    TORCH_CHECK(tokensPerBlock == 64, "production HiSparse KVarN-hot reader requires tokens_per_block=64");
    auto const numSubblocks = kvLoraRank / 128;
    auto const ckvBytes = tokensPerBlock * kvLoraRank * kvarnBits / 8;
    auto const scaleZpBytes = tokensPerBlock * 2 * numSubblocks * static_cast<int64_t>(sizeof(half));
    auto const peBytes = tokensPerBlock * qkRopeHeadDim;
    return ckvBytes + scaleZpBytes + peBytes;
}

void checkCudaTensor(th::Tensor const& tensor, char const* name)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

} // namespace

std::tuple<th::Tensor, th::Tensor> hisparseReadKvarnHotBdr(th::Tensor const& hotPacked,
    th::Tensor const& hotIndices, th::Tensor const& topkLength, th::Tensor const& rowStatus, int64_t layerIdx,
    int64_t tokensPerBlock, int64_t kvarnBits, int64_t kvLoraRank, int64_t qkRopeHeadDim)
{
    checkCudaTensor(hotPacked, "hot_packed");
    checkCudaTensor(hotIndices, "hot_indices");
    checkCudaTensor(topkLength, "topk_length");
    checkCudaTensor(rowStatus, "row_status");
    TORCH_CHECK(hotPacked.scalar_type() == torch::kUInt8,
        "hot_packed must be uint8 production BDR KVarN records");
    TORCH_CHECK(hotIndices.scalar_type() == torch::kInt32, "hot_indices must be int32");
    TORCH_CHECK(topkLength.scalar_type() == torch::kInt32, "topk_length must be int32");
    TORCH_CHECK(rowStatus.scalar_type() == torch::kUInt8, "row_status must be uint8");
    TORCH_CHECK(hotPacked.dim() == 3, "hot_packed must have shape [num_layers, hot_capacity, packed_bytes]");
    TORCH_CHECK(hotIndices.dim() == 2, "hot_indices must have shape [rows, index_topk]");
    TORCH_CHECK(topkLength.dim() == 1, "topk_length must have shape [rows]");
    TORCH_CHECK(rowStatus.dim() == 1, "row_status must have shape [rows]");
    TORCH_CHECK(hotPacked.stride(2) == 1, "hot_packed records must be byte-contiguous");
    TORCH_CHECK(hotIndices.is_contiguous(), "hot_indices must be contiguous");
    TORCH_CHECK(topkLength.is_contiguous(), "topk_length must be contiguous");
    TORCH_CHECK(rowStatus.is_contiguous(), "row_status must be contiguous");

    int64_t const rows = hotIndices.size(0);
    int64_t const indexTopK = hotIndices.size(1);
    int64_t const numLayers = hotPacked.size(0);
    int64_t const hotCapacity = hotPacked.size(1);
    int64_t const latentDim = kvLoraRank + qkRopeHeadDim;
    auto const expectedBytes = expectedBdrBytesPerBlock(tokensPerBlock, kvLoraRank, qkRopeHeadDim, kvarnBits);
    TORCH_CHECK(hotPacked.size(2) >= expectedBytes,
        "hot_packed record bytes are smaller than production BDR layout: got ", hotPacked.size(2),
        ", expected at least ", expectedBytes);
    TORCH_CHECK(rows >= 0 && rows <= std::numeric_limits<int32_t>::max(),
        "rows must fit int32, got ", rows);
    TORCH_CHECK(indexTopK > 0 && indexTopK <= std::numeric_limits<int32_t>::max(),
        "index_topk must be positive int32-sized, got ", indexTopK);
    TORCH_CHECK(numLayers > 0 && numLayers <= std::numeric_limits<int32_t>::max(),
        "num_layers must be positive int32-sized, got ", numLayers);
    TORCH_CHECK(hotCapacity > 0 && hotCapacity <= std::numeric_limits<int32_t>::max(),
        "hot_capacity must be positive int32-sized, got ", hotCapacity);
    TORCH_CHECK(layerIdx >= 0 && layerIdx < numLayers,
        "layer_idx out of range: ", layerIdx, " for num_layers=", numLayers);
    TORCH_CHECK(topkLength.size(0) == rows, "topk_length rows mismatch");
    TORCH_CHECK(rowStatus.size(0) == rows, "row_status rows mismatch");
    TORCH_CHECK(latentDim == 576, "production HiSparse KVarN-hot reader requires latent_dim=576");

    c10::cuda::CUDAGuard guard(hotPacked.device());
    int32_t const device = hotPacked.get_device();
    TORCH_CHECK(hotIndices.get_device() == device, "hot_indices must be on the same device as hot_packed");
    TORCH_CHECK(topkLength.get_device() == device, "topk_length must be on the same device as hot_packed");
    TORCH_CHECK(rowStatus.get_device() == device, "row_status must be on the same device as hot_packed");

    auto latentOut = th::empty({rows, indexTopK, latentDim},
        hotIndices.options().dtype(torch::kBFloat16));
    auto outputStatus = th::empty({rows}, rowStatus.options());

    tk::invokeHisparseReadKvarnHotBdr(hotPacked.data_ptr<uint8_t>(), hotIndices.data_ptr<int32_t>(),
        topkLength.data_ptr<int32_t>(), rowStatus.data_ptr<uint8_t>(),
        static_cast<__nv_bfloat16*>(latentOut.data_ptr()), outputStatus.data_ptr<uint8_t>(),
        static_cast<int32_t>(rows), static_cast<int32_t>(indexTopK), static_cast<int32_t>(numLayers),
        static_cast<int32_t>(hotCapacity), hotPacked.stride(0), hotPacked.stride(1), hotPacked.size(2),
        static_cast<int32_t>(layerIdx), static_cast<int32_t>(tokensPerBlock), static_cast<int32_t>(kvarnBits),
        static_cast<int32_t>(kvLoraRank), static_cast<int32_t>(qkRopeHeadDim),
        at::cuda::getCurrentCUDAStream(device).stream());

    auto const kernelErr = cudaGetLastError();
    TORCH_CHECK(kernelErr == cudaSuccess, "hisparse_read_kvarn_hot_bdr kernel launch failed: ",
        cudaGetErrorString(kernelErr));
    return {latentOut, outputStatus};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "hisparse_read_kvarn_hot_bdr(Tensor hot_packed, Tensor hot_indices, Tensor topk_length, Tensor row_status, "
        "int layer_idx, int tokens_per_block, int kvarn_bits, int kv_lora_rank, int qk_rope_head_dim) -> "
        "(Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("hisparse_read_kvarn_hot_bdr", &tensorrt_llm::torch_ext::hisparseReadKvarnHotBdr);
}

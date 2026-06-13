/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#include "tensorrt_llm/kernels/mlaKernels.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace th = torch;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace
{

void checkCudaTensor(th::Tensor const& tensor, char const* name)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

int64_t expectedBdrBytesPerBlock(int64_t tokensPerBlock, int64_t kvLoraRank, int64_t qkRopeHeadDim, int64_t ckvBits)
{
    TORCH_CHECK(ckvBits == 2 || ckvBits == 4, "ckv_bits must be 2 or 4");
    TORCH_CHECK(kvLoraRank % 128 == 0, "kv_lora_rank must be divisible by 128");
    auto const numSubblocks = kvLoraRank / 128;
    auto const ckvBytes = tokensPerBlock * kvLoraRank * ckvBits / 8;
    auto const scaleZpBytes = tokensPerBlock * 2 * numSubblocks * static_cast<int64_t>(sizeof(half));
    auto const peBytes = tokensPerBlock * qkRopeHeadDim;
    return ckvBytes + scaleZpBytes + peBytes;
}

template <typename T>
void mlaBdrWriteKvarnRecordTyped(th::Tensor const& latentBlock, th::Tensor const& bdrRecords, int64_t blockId,
    int64_t ckvBits, int64_t kvLoraRank, int64_t qkRopeHeadDim)
{
    auto stream = at::cuda::getCurrentCUDAStream(latentBlock.get_device());
    tk::invokeMLABdrWriteKvarnRecord<T>(static_cast<T const*>(latentBlock.data_ptr()), latentBlock.stride(0),
        latentBlock.stride(1), bdrRecords.data_ptr<std::uint8_t>(), bdrRecords.stride(0), static_cast<int>(blockId),
        static_cast<int>(latentBlock.size(0)), static_cast<int>(kvLoraRank), static_cast<int>(qkRopeHeadDim),
        static_cast<int>(ckvBits), stream);
}

} // namespace

void mlaBdrWriteKvarnRecord(th::Tensor const& latentBlock, th::Tensor const& bdrRecords, int64_t blockId,
    int64_t ckvBits, int64_t kvLoraRank, int64_t qkRopeHeadDim)
{
    checkCudaTensor(latentBlock, "latent_block");
    checkCudaTensor(bdrRecords, "bdr_records");
    TORCH_CHECK(latentBlock.device() == bdrRecords.device(), "latent_block and bdr_records must be on the same device");
    TORCH_CHECK(latentBlock.scalar_type() == at::ScalarType::Half
            || latentBlock.scalar_type() == at::ScalarType::BFloat16
            || latentBlock.scalar_type() == at::ScalarType::Float,
        "latent_block must be fp16, bf16, or fp32");
    TORCH_CHECK(bdrRecords.scalar_type() == at::ScalarType::Byte, "bdr_records must be uint8");
    TORCH_CHECK(latentBlock.dim() == 2, "latent_block must have shape [tokens_per_block, latent_dim]");
    TORCH_CHECK(bdrRecords.dim() == 2, "bdr_records must have shape [num_blocks, packed_bytes_per_block]");
    TORCH_CHECK(latentBlock.stride(1) == 1, "latent_block last dimension must be contiguous");
    TORCH_CHECK(bdrRecords.stride(1) == 1, "bdr_records last dimension must be byte-contiguous");
    TORCH_CHECK(blockId >= 0 && blockId < bdrRecords.size(0), "block_id out of range for bdr_records");
    TORCH_CHECK(latentBlock.size(1) >= kvLoraRank + qkRopeHeadDim,
        "latent_block latent_dim is smaller than kv_lora_rank + qk_rope_head_dim");
    auto const expectedBytes = expectedBdrBytesPerBlock(latentBlock.size(0), kvLoraRank, qkRopeHeadDim, ckvBits);
    TORCH_CHECK(bdrRecords.size(1) >= expectedBytes,
        "bdr_records packed dimension is too small for the production BDR layout");

    c10::cuda::CUDAGuard guard(latentBlock.device());
    if (latentBlock.scalar_type() == at::ScalarType::Half)
    {
        mlaBdrWriteKvarnRecordTyped<half>(latentBlock, bdrRecords, blockId, ckvBits, kvLoraRank, qkRopeHeadDim);
    }
    else if (latentBlock.scalar_type() == at::ScalarType::BFloat16)
    {
        mlaBdrWriteKvarnRecordTyped<__nv_bfloat16>(
            latentBlock, bdrRecords, blockId, ckvBits, kvLoraRank, qkRopeHeadDim);
    }
    else
    {
        mlaBdrWriteKvarnRecordTyped<float>(latentBlock, bdrRecords, blockId, ckvBits, kvLoraRank, qkRopeHeadDim);
    }
    auto const kernelErr = cudaGetLastError();
    TORCH_CHECK(kernelErr == cudaSuccess, "mla_bdr_write_kvarn_record kernel launch failed: ",
        cudaGetErrorString(kernelErr));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "mla_bdr_write_kvarn_record(Tensor latent_block, Tensor bdr_records, int block_id, int ckv_bits, "
        "int kv_lora_rank, int qk_rope_head_dim) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("mla_bdr_write_kvarn_record", &tensorrt_llm::torch_ext::mlaBdrWriteKvarnRecord);
}

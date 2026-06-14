/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#include "tensorrt_llm/kernels/hisparseKvarnHotRead.h"
#include "tensorrt_llm/kernels/hisparseKvarnBdrRead.cuh"

#include <stdexcept>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

void check(bool condition, char const* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void checkCuda(cudaError_t status, char const* message)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(message);
    }
}

__global__ void hisparseReadKvarnHotBdrKernel(uint8_t const* __restrict__ hotPacked,
    int32_t const* __restrict__ hotIndices, int32_t const* __restrict__ topkLength,
    uint8_t const* __restrict__ inputRowStatus, __nv_bfloat16* __restrict__ latentOut,
    uint8_t* __restrict__ outputRowStatus, int32_t numRows, int32_t indexTopK, int32_t numLayers,
    int32_t hotCapacity, int64_t hotLayerStride, int64_t hotSlotStride, int64_t hotRecordStride, int32_t layerIdx,
    int32_t tokensPerBlock, int32_t kvLoraRank, int32_t qkRopeHeadDim)
{
    int32_t const row = static_cast<int32_t>(blockIdx.x);
    if (row >= numRows)
    {
        return;
    }

    HiSparseKvarnK2v2BdrLayout const layout
        = makeHisparseKvarnK2v2BdrLayout(tokensPerBlock, kvLoraRank, qkRopeHeadDim);
    int32_t const latentDim = layout.latentDim;
    int32_t const rowTopK = topkLength[row];
    int32_t const strideFactor = numLayers * tokensPerBlock;

    __shared__ int32_t rowCode;
    if (threadIdx.x == 0)
    {
        rowCode = inputRowStatus[row] == 0 ? kHotReadOk : kHotReadUpstreamInvalid;
        if (rowCode == kHotReadOk && (rowTopK < 0 || rowTopK > indexTopK))
        {
            rowCode = kHotReadBadTopKLength;
        }
        if (rowCode == kHotReadOk && hotRecordStride < hisparseKvarnK2v2BdrRecordBytes(
                layout.tokensPerBlock, layout.kvLoraRank, layout.qkRopeHeadDim))
        {
            rowCode = kHotReadInvalidIndex;
        }
    }
    __syncthreads();

    int64_t const rowOutOffset = static_cast<int64_t>(row) * indexTopK * latentDim;
    int64_t const rowIndexOffset = static_cast<int64_t>(row) * indexTopK;
    int64_t const total = static_cast<int64_t>(indexTopK) * latentDim;
    for (int64_t linear = threadIdx.x; linear < total; linear += blockDim.x)
    {
        int32_t const col = static_cast<int32_t>(linear / latentDim);
        int32_t const dim = static_cast<int32_t>(linear - static_cast<int64_t>(col) * latentDim);
        __nv_bfloat16 out = __float2bfloat16_rn(0.0F);

        if (rowCode == kHotReadOk && col < rowTopK)
        {
            int32_t const hotIndex = hotIndices[rowIndexOffset + col];
            HiSparseKvarnHotAddress const address
                = decodeHisparseKvarnHotIndex(hotIndex, strideFactor, layerIdx, hotCapacity, tokensPerBlock);
            if (address.status != kHotReadOk)
            {
                atomicCAS(&rowCode, kHotReadOk, static_cast<int32_t>(address.status));
            }
            else
            {
                uint8_t const* record = hotPacked + static_cast<int64_t>(layerIdx) * hotLayerStride
                    + static_cast<int64_t>(address.hotSlot) * hotSlotStride;
                out = readHisparseKvarnK2v2BdrLatentValue(record, layout, address.tokenOffset, dim);
            }
        }
        latentOut[rowOutOffset + linear] = out;
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        outputRowStatus[row] = static_cast<uint8_t>(rowCode);
    }
}

} // namespace

void invokeHisparseReadKvarnHotBdr(uint8_t const* hotPacked, int32_t const* hotIndices,
    int32_t const* topkLength, uint8_t const* inputRowStatus, __nv_bfloat16* latentOut, uint8_t* outputRowStatus,
    int32_t numRows, int32_t indexTopK, int32_t numLayers, int32_t hotCapacity, int64_t hotLayerStride,
    int64_t hotSlotStride, int64_t hotRecordStride, int32_t layerIdx, int32_t tokensPerBlock, int32_t kvarnBits,
    int32_t kvLoraRank, int32_t qkRopeHeadDim, cudaStream_t stream)
{
    if (numRows <= 0)
    {
        return;
    }
    check(indexTopK > 0, "hisparse_read_kvarn_hot_bdr requires index_topk > 0");
    check(numLayers > 0, "hisparse_read_kvarn_hot_bdr requires num_layers > 0");
    check(hotCapacity > 0, "hisparse_read_kvarn_hot_bdr requires hot_capacity > 0");
    check(layerIdx >= 0 && layerIdx < numLayers, "hisparse_read_kvarn_hot_bdr layer_idx out of range");
    check(tokensPerBlock == 64, "hisparse_read_kvarn_hot_bdr production path requires tpb=64");
    check(kvLoraRank == 512, "hisparse_read_kvarn_hot_bdr production path requires kv_lora_rank=512");
    check(qkRopeHeadDim == 64, "hisparse_read_kvarn_hot_bdr production path requires qk_rope_head_dim=64");
    check(kvarnBits == 2, "hisparse_read_kvarn_hot_bdr production path requires kvarn_bits=2");
    check(hisparseKvarnK2v2BdrLayoutIsProduction(tokensPerBlock, kvLoraRank, qkRopeHeadDim),
        "hisparse_read_kvarn_hot_bdr requires the production KVarN BDR layout");

    constexpr int32_t kThreads = 256;
    hisparseReadKvarnHotBdrKernel<<<numRows, kThreads, 0, stream>>>(hotPacked, hotIndices, topkLength, inputRowStatus,
        latentOut, outputRowStatus, numRows, indexTopK, numLayers, hotCapacity, hotLayerStride, hotSlotStride,
        hotRecordStride, layerIdx, tokensPerBlock, kvLoraRank, qkRopeHeadDim);
    checkCuda(cudaGetLastError(), "hisparse_read_kvarn_hot_bdr kernel launch failed");
}

} // namespace kernels

TRTLLM_NAMESPACE_END

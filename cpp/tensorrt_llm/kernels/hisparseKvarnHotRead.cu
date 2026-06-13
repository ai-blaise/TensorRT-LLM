/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#include "tensorrt_llm/kernels/hisparseKvarnHotRead.h"

#include "tensorrt_llm/common/cudaUtils.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

enum HiSparseKvarnHotReadStatus : uint8_t
{
    kHotReadOk = 0,
    kHotReadUpstreamInvalid = 1,
    kHotReadBadTopKLength = 2,
    kHotReadInvalidIndex = 3,
    kHotReadHotSlotOutOfRange = 4,
    kHotReadLayerMismatch = 5,
    kHotReadTokenOffsetOutOfRange = 6,
};

template <int BITS>
__device__ __forceinline__ float readLowBitKvarnValue(
    uint8_t const* tokenPacked, __half const* tokenScaleZp, int dim)
{
    static_assert(BITS == 2 || BITS == 4, "HiSparse KVarN hot reader supports 2-bit or 4-bit C-KV");
    constexpr int kValuesPerByte = 8 / BITS;
    constexpr int kMask = (1 << BITS) - 1;
    int const byteIdx = dim / kValuesPerByte;
    int const shift = (dim % kValuesPerByte) * BITS;
    int const q = (tokenPacked[byteIdx] >> shift) & kMask;
    int const subblock = dim / 128;
    float const scale = __half2float(tokenScaleZp[subblock]);
    float const zp = __half2float(tokenScaleZp[4 + subblock]);
    return static_cast<float>(q) * scale + zp;
}

__device__ __forceinline__ float readFp8E4m3Byte(uint8_t byte)
{
    __nv_fp8_e4m3 value;
    value.__x = byte;
    return static_cast<float>(value);
}

template <int BITS>
__global__ void hisparseReadKvarnHotBdrKernel(uint8_t const* __restrict__ hotPacked,
    int32_t const* __restrict__ hotIndices, int32_t const* __restrict__ topkLength,
    uint8_t const* __restrict__ inputRowStatus, __nv_bfloat16* __restrict__ latentOut,
    uint8_t* __restrict__ outputRowStatus, int32_t numRows, int32_t indexTopK, int32_t numLayers,
    int32_t hotCapacity, int64_t hotLayerStride, int64_t hotSlotStride, int64_t hotRecordStride, int32_t layerIdx,
    int32_t tokensPerBlock, int32_t kvLoraRank, int32_t qkRopeHeadDim)
{
    static_assert(BITS == 2 || BITS == 4, "HiSparse KVarN hot reader supports 2-bit or 4-bit C-KV");
    int32_t const row = static_cast<int32_t>(blockIdx.x);
    if (row >= numRows)
    {
        return;
    }

    constexpr int32_t kBdrOrder = 128;
    constexpr int32_t kNumSubblocks = 512 / kBdrOrder;
    int32_t const latentDim = kvLoraRank + qkRopeHeadDim;
    int32_t const rowTopK = topkLength[row];
    int32_t const strideFactor = numLayers * tokensPerBlock;
    int64_t const ckvBytesPerToken = static_cast<int64_t>(kvLoraRank) * BITS / 8;
    int64_t const ckvBytesPerBlock = static_cast<int64_t>(tokensPerBlock) * ckvBytesPerToken;
    int64_t const scaleZpBytesPerToken = static_cast<int64_t>(2 * kNumSubblocks * sizeof(__half));
    int64_t const scaleZpBytesPerBlock = static_cast<int64_t>(tokensPerBlock) * scaleZpBytesPerToken;
    int64_t const peBytesPerBlock = static_cast<int64_t>(tokensPerBlock) * qkRopeHeadDim;

    __shared__ int32_t rowCode;
    if (threadIdx.x == 0)
    {
        rowCode = inputRowStatus[row] == 0 ? kHotReadOk : kHotReadUpstreamInvalid;
        if (rowCode == kHotReadOk && (rowTopK < 0 || rowTopK > indexTopK))
        {
            rowCode = kHotReadBadTopKLength;
        }
        if (rowCode == kHotReadOk
            && hotRecordStride < ckvBytesPerBlock + scaleZpBytesPerBlock + peBytesPerBlock)
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
            if (hotIndex < 0)
            {
                atomicCAS(&rowCode, kHotReadOk, kHotReadInvalidIndex);
            }
            else
            {
                int32_t const hotSlot = hotIndex / strideFactor;
                int32_t const rem = hotIndex - hotSlot * strideFactor;
                int32_t const encodedLayer = rem / tokensPerBlock;
                int32_t const tokenOffset = rem - encodedLayer * tokensPerBlock;
                if (hotSlot < 0 || hotSlot >= hotCapacity)
                {
                    atomicCAS(&rowCode, kHotReadOk, kHotReadHotSlotOutOfRange);
                }
                else if (encodedLayer != layerIdx)
                {
                    atomicCAS(&rowCode, kHotReadOk, kHotReadLayerMismatch);
                }
                else if (tokenOffset < 0 || tokenOffset >= tokensPerBlock)
                {
                    atomicCAS(&rowCode, kHotReadOk, kHotReadTokenOffsetOutOfRange);
                }
                else
                {
                    uint8_t const* record = hotPacked + static_cast<int64_t>(layerIdx) * hotLayerStride
                        + static_cast<int64_t>(hotSlot) * hotSlotStride;
                    uint8_t const* ckvData = record;
                    uint8_t const* ckvScaleZpBytes = ckvData + ckvBytesPerBlock;
                    uint8_t const* peBytes = ckvScaleZpBytes + scaleZpBytesPerBlock;
                    if (dim < kvLoraRank)
                    {
                        uint8_t const* tokenPacked = ckvData + static_cast<int64_t>(tokenOffset) * ckvBytesPerToken;
                        auto const* tokenScaleZp = reinterpret_cast<__half const*>(
                            ckvScaleZpBytes + static_cast<int64_t>(tokenOffset) * scaleZpBytesPerToken);
                        out = __float2bfloat16_rn(readLowBitKvarnValue<BITS>(tokenPacked, tokenScaleZp, dim));
                    }
                    else
                    {
                        int32_t const peDim = dim - kvLoraRank;
                        uint8_t const byte = peBytes[static_cast<int64_t>(tokenOffset) * qkRopeHeadDim + peDim];
                        out = __float2bfloat16_rn(readFp8E4m3Byte(byte));
                    }
                }
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
    TLLM_CHECK_WITH_INFO(indexTopK > 0, "hisparse_read_kvarn_hot_bdr requires index_topk > 0");
    TLLM_CHECK_WITH_INFO(numLayers > 0, "hisparse_read_kvarn_hot_bdr requires num_layers > 0");
    TLLM_CHECK_WITH_INFO(hotCapacity > 0, "hisparse_read_kvarn_hot_bdr requires hot_capacity > 0");
    TLLM_CHECK_WITH_INFO(layerIdx >= 0 && layerIdx < numLayers, "hisparse_read_kvarn_hot_bdr layer_idx out of range");
    TLLM_CHECK_WITH_INFO(tokensPerBlock == 64, "hisparse_read_kvarn_hot_bdr production path requires tpb=64");
    TLLM_CHECK_WITH_INFO(kvLoraRank == 512, "hisparse_read_kvarn_hot_bdr production path requires kv_lora_rank=512");
    TLLM_CHECK_WITH_INFO(qkRopeHeadDim == 64,
        "hisparse_read_kvarn_hot_bdr production path requires qk_rope_head_dim=64");
    TLLM_CHECK_WITH_INFO(kvarnBits == 2 || kvarnBits == 4,
        "hisparse_read_kvarn_hot_bdr supports kvarn_bits=2 or 4");

    constexpr int32_t kThreads = 256;
    if (kvarnBits == 2)
    {
        hisparseReadKvarnHotBdrKernel<2><<<numRows, kThreads, 0, stream>>>(hotPacked, hotIndices, topkLength,
            inputRowStatus, latentOut, outputRowStatus, numRows, indexTopK, numLayers, hotCapacity, hotLayerStride,
            hotSlotStride, hotRecordStride, layerIdx, tokensPerBlock, kvLoraRank, qkRopeHeadDim);
    }
    else
    {
        hisparseReadKvarnHotBdrKernel<4><<<numRows, kThreads, 0, stream>>>(hotPacked, hotIndices, topkLength,
            inputRowStatus, latentOut, outputRowStatus, numRows, indexTopK, numLayers, hotCapacity, hotLayerStride,
            hotSlotStride, hotRecordStride, layerIdx, tokensPerBlock, kvLoraRank, qkRopeHeadDim);
    }
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels

TRTLLM_NAMESPACE_END

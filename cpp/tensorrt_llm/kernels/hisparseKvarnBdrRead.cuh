/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
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

struct HiSparseKvarnK2v2BdrLayout
{
    int32_t tokensPerBlock;
    int32_t kvLoraRank;
    int32_t qkRopeHeadDim;
    int64_t ckvBytesPerToken;
    int64_t ckvBytesPerBlock;
    int64_t scaleZpBytesPerToken;
    int64_t scaleZpBytesPerBlock;
    int64_t peBytesPerBlock;
    int32_t latentDim;
};

struct HiSparseKvarnHotAddress
{
    int32_t hotSlot;
    int32_t tokenOffset;
    HiSparseKvarnHotReadStatus status;
};

__host__ __device__ __forceinline__ int64_t hisparseKvarnK2v2BdrRecordBytes(
    int32_t tokensPerBlock, int32_t kvLoraRank, int32_t qkRopeHeadDim)
{
    constexpr int32_t kBdrOrder = 128;
    int32_t const numSubblocks = kvLoraRank / kBdrOrder;
    int64_t const ckvBytesPerToken = static_cast<int64_t>(kvLoraRank) * 2 / 8;
    int64_t const scaleZpBytesPerToken = static_cast<int64_t>(2 * numSubblocks * sizeof(__half));
    int64_t const peBytesPerToken = qkRopeHeadDim;
    return static_cast<int64_t>(tokensPerBlock) * (ckvBytesPerToken + scaleZpBytesPerToken + peBytesPerToken);
}

__host__ __device__ __forceinline__ HiSparseKvarnK2v2BdrLayout makeHisparseKvarnK2v2BdrLayout(
    int32_t tokensPerBlock, int32_t kvLoraRank, int32_t qkRopeHeadDim)
{
    constexpr int32_t kBdrOrder = 128;
    int32_t const numSubblocks = kvLoraRank / kBdrOrder;
    HiSparseKvarnK2v2BdrLayout layout;
    layout.tokensPerBlock = tokensPerBlock;
    layout.kvLoraRank = kvLoraRank;
    layout.qkRopeHeadDim = qkRopeHeadDim;
    layout.ckvBytesPerToken = static_cast<int64_t>(kvLoraRank) * 2 / 8;
    layout.ckvBytesPerBlock = static_cast<int64_t>(tokensPerBlock) * layout.ckvBytesPerToken;
    layout.scaleZpBytesPerToken = static_cast<int64_t>(2 * numSubblocks * sizeof(__half));
    layout.scaleZpBytesPerBlock = static_cast<int64_t>(tokensPerBlock) * layout.scaleZpBytesPerToken;
    layout.peBytesPerBlock = static_cast<int64_t>(tokensPerBlock) * qkRopeHeadDim;
    layout.latentDim = kvLoraRank + qkRopeHeadDim;
    return layout;
}

__host__ __device__ __forceinline__ bool hisparseKvarnK2v2BdrLayoutIsProduction(
    int32_t tokensPerBlock, int32_t kvLoraRank, int32_t qkRopeHeadDim)
{
    return tokensPerBlock == 64 && kvLoraRank == 512 && qkRopeHeadDim == 64;
}

__device__ __forceinline__ HiSparseKvarnHotAddress decodeHisparseKvarnHotIndex(
    int32_t hotIndex, int32_t strideFactor, int32_t layerIdx, int32_t hotCapacity, int32_t tokensPerBlock)
{
    HiSparseKvarnHotAddress address;
    address.hotSlot = -1;
    address.tokenOffset = -1;
    address.status = kHotReadOk;

    if (hotIndex < 0)
    {
        address.status = kHotReadInvalidIndex;
        return address;
    }

    int32_t const hotSlot = hotIndex / strideFactor;
    int32_t const rem = hotIndex - hotSlot * strideFactor;
    int32_t const encodedLayer = rem / tokensPerBlock;
    int32_t const tokenOffset = rem - encodedLayer * tokensPerBlock;
    if (hotSlot < 0 || hotSlot >= hotCapacity)
    {
        address.status = kHotReadHotSlotOutOfRange;
    }
    else if (encodedLayer != layerIdx)
    {
        address.status = kHotReadLayerMismatch;
    }
    else if (tokenOffset < 0 || tokenOffset >= tokensPerBlock)
    {
        address.status = kHotReadTokenOffsetOutOfRange;
    }
    else
    {
        address.hotSlot = hotSlot;
        address.tokenOffset = tokenOffset;
    }
    return address;
}

__device__ __forceinline__ float readHisparseKvarnK2v2PackedCkvValue(
    uint8_t const* tokenPacked, __half const* tokenScaleZp, int dim)
{
    constexpr int32_t kBits = 2;
    constexpr int32_t kValuesPerByte = 8 / kBits;
    constexpr int32_t kMask = (1 << kBits) - 1;
    int32_t const byteIdx = dim / kValuesPerByte;
    int32_t const shift = (dim % kValuesPerByte) * kBits;
    int32_t const q = (tokenPacked[byteIdx] >> shift) & kMask;
    int32_t const subblock = dim / 128;
    float const scale = __half2float(tokenScaleZp[subblock]);
    float const zp = __half2float(tokenScaleZp[4 + subblock]);
    return static_cast<float>(q) * scale + zp;
}

__device__ __forceinline__ float readHisparseFp8E4m3Byte(uint8_t byte)
{
    __nv_fp8_e4m3 value;
    value.__x = byte;
    return static_cast<float>(value);
}

__device__ __forceinline__ __nv_bfloat16 readHisparseKvarnK2v2BdrLatentValue(
    uint8_t const* record, HiSparseKvarnK2v2BdrLayout const& layout, int32_t tokenOffset, int32_t dim)
{
    uint8_t const* ckvData = record;
    uint8_t const* ckvScaleZpBytes = ckvData + layout.ckvBytesPerBlock;
    uint8_t const* peBytes = ckvScaleZpBytes + layout.scaleZpBytesPerBlock;
    if (dim < layout.kvLoraRank)
    {
        uint8_t const* tokenPacked = ckvData + static_cast<int64_t>(tokenOffset) * layout.ckvBytesPerToken;
        auto const* tokenScaleZp = reinterpret_cast<__half const*>(
            ckvScaleZpBytes + static_cast<int64_t>(tokenOffset) * layout.scaleZpBytesPerToken);
        return __float2bfloat16_rn(readHisparseKvarnK2v2PackedCkvValue(tokenPacked, tokenScaleZp, dim));
    }

    int32_t const peDim = dim - layout.kvLoraRank;
    uint8_t const byte = peBytes[static_cast<int64_t>(tokenOffset) * layout.qkRopeHeadDim + peDim];
    return __float2bfloat16_rn(readHisparseFp8E4m3Byte(byte));
}

} // namespace kernels

TRTLLM_NAMESPACE_END

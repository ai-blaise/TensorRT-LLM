/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/kernels/mlaKernels.h"

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
using tensorrt_llm::common::cuda_cast;

namespace
{

static constexpr float kInvSqrtHadamard128 = 0.088388347648318f;

template <int ELTS>
inline __device__ void recordBdrFwhtSubblockWarp(float (&reg)[ELTS], int laneInBlk, unsigned mask)
{
#pragma unroll
    for (int len = 1; len < ELTS; len <<= 1)
    {
#pragma unroll
        for (int i = 0; i < ELTS; ++i)
        {
            int partner = i ^ len;
            if (i < partner)
            {
                float u = reg[i], v = reg[partner];
                reg[i] = u + v;
                reg[partner] = u - v;
            }
        }
    }

    constexpr int kLanes = 128 / ELTS;
#pragma unroll
    for (int span = 1; span < kLanes; span <<= 1)
    {
        bool low = ((laneInBlk & span) == 0);
#pragma unroll
        for (int i = 0; i < ELTS; ++i)
        {
            float other = __shfl_xor_sync(mask, reg[i], span);
            reg[i] = low ? (reg[i] + other) : (other - reg[i]);
        }
    }

#pragma unroll
    for (int i = 0; i < ELTS; ++i)
    {
        reg[i] *= kInvSqrtHadamard128;
    }
}

template <int ELTS, int BITS>
inline __device__ void recordBdrPackLowBitVec(uint8_t* packed, float const (&reg)[ELTS], float scale, float zp)
{
    static_assert(BITS == 2 || BITS == 4, "KVarN BDR supports 2-bit or 4-bit packing");
    static_assert((ELTS * BITS) % 8 == 0, "ELTS must pack to whole bytes");
    constexpr int kValsPerByte = 8 / BITS;
    constexpr int kQMax = (1 << BITS) - 1;
    float inv = 1.0f / scale;
#pragma unroll
    for (int byteIdx = 0; byteIdx < (ELTS * BITS) / 8; ++byteIdx)
    {
        uint8_t out = 0;
#pragma unroll
        for (int j = 0; j < kValsPerByte; ++j)
        {
            int const elt = byteIdx * kValsPerByte + j;
            int q = __float2int_rn((reg[elt] - zp) * inv);
            q = q < 0 ? 0 : (q > kQMax ? kQMax : q);
            out |= static_cast<uint8_t>(q << (j * BITS));
        }
        packed[byteIdx] = out;
    }
}

template <int ELTS>
inline __device__ void recordBdrSubblockMinMax(
    float const (&reg)[ELTS], unsigned mask, float& outLo, float& outHi)
{
    float lo = reg[0], hi = reg[0];
#pragma unroll
    for (int i = 1; i < ELTS; ++i)
    {
        lo = fminf(lo, reg[i]);
        hi = fmaxf(hi, reg[i]);
    }
    constexpr int kLanes = 128 / ELTS;
#pragma unroll
    for (int span = kLanes / 2; span >= 1; span >>= 1)
    {
        lo = fminf(lo, __shfl_xor_sync(mask, lo, span));
        hi = fmaxf(hi, __shfl_xor_sync(mask, hi, span));
    }
    outLo = lo;
    outHi = hi;
}

inline __device__ void recordBdrWriteHalfUnaligned(uint8_t* dst, __half value)
{
    uint16_t const raw = __half_as_ushort(value);
    dst[0] = static_cast<uint8_t>(raw & 0xFFu);
    dst[1] = static_cast<uint8_t>((raw >> 8) & 0xFFu);
}

template <typename T, int DCKV, int HORDER, int BITS>
__global__ void mlaBdrWriteKvarnRecordKernel(T const* __restrict__ latentBlock,
    int64_t latentTokenStride, int64_t latentDimStride, uint8_t* __restrict__ bdrRecords,
    int64_t bdrRecordStride, int blockId, int tokensPerBlock, int qkRopeHeadDim)
{
    static_assert(BITS == 2 || BITS == 4, "KVarN BDR supports 2-bit or 4-bit packing");
    constexpr int kNSub = DCKV / HORDER;
    constexpr int kVecPerSub = HORDER / 8;
    constexpr int kBytesPerVec = 8 * BITS / 8;
    constexpr int kCkvBytesPerToken = DCKV * BITS / 8;

    int const tok = blockIdx.x;
    if (tok >= tokensPerBlock)
    {
        return;
    }

    uint8_t* record = bdrRecords + static_cast<int64_t>(blockId) * bdrRecordStride;
    uint8_t* ckvData = record;
    uint8_t* ckvScaleZpBytes = ckvData + static_cast<int64_t>(tokensPerBlock) * kCkvBytesPerToken;
    uint8_t* peBytes = ckvScaleZpBytes + static_cast<int64_t>(tokensPerBlock) * (2 * kNSub * sizeof(__half));
    int const lane = threadIdx.x;
    int const sub = lane / kVecPerSub;
    int const laneInBlk = lane % kVecPerSub;
    unsigned const mask = 0xFFFFu << ((sub % 2) * 16);

    float reg[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        int const c = lane * 8 + i;
        reg[i] = cuda_cast<float>(
            latentBlock[static_cast<int64_t>(tok) * latentTokenStride + static_cast<int64_t>(c) * latentDimStride]);
    }
    recordBdrFwhtSubblockWarp<8>(reg, laneInBlk, mask);
    float lo, hi;
    recordBdrSubblockMinMax<8>(reg, mask, lo, hi);
    constexpr int kQMax = (1 << BITS) - 1;
    float const sc = fmaxf((hi - lo) / static_cast<float>(kQMax), 1e-10f);
    __half const hsc = __float2half(sc), hzp = __float2half(lo);
    uint8_t* tokData = ckvData + static_cast<int64_t>(tok) * kCkvBytesPerToken;
    recordBdrPackLowBitVec<8, BITS>(tokData + lane * kBytesPerVec, reg, __half2float(hsc), __half2float(hzp));
    if (laneInBlk == 0)
    {
        uint8_t* tokScaleBytes = ckvScaleZpBytes + static_cast<int64_t>(tok) * (2 * kNSub * sizeof(__half));
        recordBdrWriteHalfUnaligned(tokScaleBytes + static_cast<int64_t>(sub) * sizeof(__half), hsc);
        recordBdrWriteHalfUnaligned(tokScaleBytes + static_cast<int64_t>(kNSub + sub) * sizeof(__half), hzp);
    }

    if (lane < qkRopeHeadDim)
    {
        float const pe = cuda_cast<float>(latentBlock[static_cast<int64_t>(tok) * latentTokenStride
            + static_cast<int64_t>(DCKV + lane) * latentDimStride]);
        __nv_fp8_e4m3 const pe8 = cuda_cast<__nv_fp8_e4m3>(pe);
        peBytes[static_cast<int64_t>(tok) * qkRopeHeadDim + lane] = pe8.__x;
    }
}

} // namespace

template <typename T>
void invokeMLABdrWriteKvarnRecord(T const* latentBlock, int64_t latentTokenStride, int64_t latentDimStride,
    uint8_t* bdrRecords, int64_t bdrRecordStride, int blockId, int tokensPerBlock, int kvLoraRank,
    int qkRopeHeadDim, int bits, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(kvLoraRank == 512, "KVarN BDR record writer currently supports DCKV=512.");
    TLLM_CHECK_WITH_INFO(qkRopeHeadDim > 0 && qkRopeHeadDim <= 64,
        "KVarN BDR record writer currently supports 1 <= qk_rope_head_dim <= 64.");
    TLLM_CHECK_WITH_INFO(bits == 2 || bits == 4, "KVarN BDR record writer supports bits=2 or bits=4, got %d.", bits);
    TLLM_CHECK_WITH_INFO(tokensPerBlock > 0, "KVarN BDR record writer requires tokens_per_block > 0.");
    TLLM_CHECK_WITH_INFO(blockId >= 0, "KVarN BDR record writer requires a non-negative block id.");
    constexpr int kVecs = 512 / 8;
    if (bits == 2)
    {
        mlaBdrWriteKvarnRecordKernel<T, 512, 128, 2><<<tokensPerBlock, kVecs, 0, stream>>>(
            latentBlock, latentTokenStride, latentDimStride, bdrRecords, bdrRecordStride, blockId, tokensPerBlock,
            qkRopeHeadDim);
    }
    else
    {
        mlaBdrWriteKvarnRecordKernel<T, 512, 128, 4><<<tokensPerBlock, kVecs, 0, stream>>>(
            latentBlock, latentTokenStride, latentDimStride, bdrRecords, bdrRecordStride, blockId, tokensPerBlock,
            qkRopeHeadDim);
    }
}

template void invokeMLABdrWriteKvarnRecord<float>(float const* latentBlock, int64_t latentTokenStride,
    int64_t latentDimStride, uint8_t* bdrRecords, int64_t bdrRecordStride, int blockId, int tokensPerBlock,
    int kvLoraRank, int qkRopeHeadDim, int bits, cudaStream_t stream);
template void invokeMLABdrWriteKvarnRecord<half>(half const* latentBlock, int64_t latentTokenStride,
    int64_t latentDimStride, uint8_t* bdrRecords, int64_t bdrRecordStride, int blockId, int tokensPerBlock,
    int kvLoraRank, int qkRopeHeadDim, int bits, cudaStream_t stream);
template void invokeMLABdrWriteKvarnRecord<__nv_bfloat16>(__nv_bfloat16 const* latentBlock,
    int64_t latentTokenStride, int64_t latentDimStride, uint8_t* bdrRecords, int64_t bdrRecordStride, int blockId,
    int tokensPerBlock, int kvLoraRank, int qkRopeHeadDim, int bits, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END

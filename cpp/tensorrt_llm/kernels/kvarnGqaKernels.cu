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

#include "tensorrt_llm/common/assert.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

using Layout = KVarNGqaK2V2G128Layout;

static constexpr int kKPackedOffset = 0;
static constexpr int kKSRowAbsOffset = 4096;
static constexpr int kKZpAbsOffset = 4352;
static constexpr int kKSColOffset = 4608;
static constexpr int kVPackedOffset = 4864;
static constexpr int kVSColOffset = 8960;
static constexpr int kVSRowAbsOffset = 9216;
static constexpr int kVZpAbsOffset = 9472;
static constexpr float kHadamardScale = 0.08838834764831845f; // 1 / sqrt(128)

struct PackedRecordView
{
    std::uint8_t const* ptr;
    bool pageLayout;
    std::int64_t strideBlock;
    std::int64_t strideToken;
    std::int64_t strideHead;
    std::int64_t strideByte;
};

__device__ __forceinline__ int hadamardSign(int row, int col)
{
    return (__popc(static_cast<unsigned>(row & col)) & 1) ? -1 : 1;
}

__device__ __forceinline__ std::uint8_t recordByte(PackedRecordView view, std::int64_t blockId, int kvHead, int byteIdx)
{
    if (view.pageLayout)
    {
        int tokenSlot = byteIdx / Layout::kBytesPerTokenSlot;
        int byteInSlot = byteIdx - tokenSlot * Layout::kBytesPerTokenSlot;
        return view.ptr[blockId * view.strideBlock + tokenSlot * view.strideToken + kvHead * view.strideHead
            + byteInSlot * view.strideByte];
    }
    return view.ptr[blockId * view.strideBlock + kvHead * view.strideHead + byteIdx * view.strideByte];
}

__device__ __forceinline__ float readPackedFp16(PackedRecordView view, std::int64_t blockId, int kvHead, int byteOffset)
{
    std::uint16_t lo = recordByte(view, blockId, kvHead, byteOffset);
    std::uint16_t hi = recordByte(view, blockId, kvHead, byteOffset + 1);
    union
    {
        std::uint16_t u;
        __half h;
    } cvt;
    cvt.u = static_cast<std::uint16_t>(lo | (hi << 8));
    return __half2float(cvt.h);
}

__device__ __forceinline__ int readPacked2(PackedRecordView view, std::int64_t blockId, int kvHead, int baseOffset, int valueIdx)
{
    int bit = valueIdx * 2;
    int byteIdx = baseOffset + (bit >> 3);
    int shift = bit & 7;
    return (recordByte(view, blockId, kvHead, byteIdx) >> shift) & 0x3;
}

__device__ __forceinline__ float dequantKRot(PackedRecordView view, std::int64_t blockId, int kvHead, int token, int dim)
{
    int q = readPacked2(view, blockId, kvHead, kKPackedOffset, dim * Layout::kGroupSize + token);
    float sRowAbs = readPackedFp16(view, blockId, kvHead, kKSRowAbsOffset + dim * 2);
    float zpAbs = readPackedFp16(view, blockId, kvHead, kKZpAbsOffset + dim * 2);
    float sCol = readPackedFp16(view, blockId, kvHead, kKSColOffset + token * 2);
    return (static_cast<float>(q) * sRowAbs + zpAbs) * sCol;
}

__device__ __forceinline__ float dequantVRot(PackedRecordView view, std::int64_t blockId, int kvHead, int token, int dim)
{
    int q = readPacked2(view, blockId, kvHead, kVPackedOffset, token * Layout::kHeadDim + dim);
    float sCol = readPackedFp16(view, blockId, kvHead, kVSColOffset + dim * 2);
    float sRowAbs = readPackedFp16(view, blockId, kvHead, kVSRowAbsOffset + token * 2);
    float zpAbs = readPackedFp16(view, blockId, kvHead, kVZpAbsOffset + token * 2);
    return (static_cast<float>(q) * sRowAbs + zpAbs) * sCol;
}

template <typename T>
__device__ __forceinline__ float loadScalar(T const* ptr)
{
    return static_cast<float>(*ptr);
}

template <>
__device__ __forceinline__ float loadScalar<__half>(__half const* ptr)
{
    return __half2float(*ptr);
}

template <>
__device__ __forceinline__ float loadScalar<__nv_bfloat16>(__nv_bfloat16 const* ptr)
{
    return __bfloat162float(*ptr);
}

template <typename T>
__device__ __forceinline__ void storeScalar(T* ptr, float value)
{
    *ptr = static_cast<T>(value);
}

template <>
__device__ __forceinline__ void storeScalar<__half>(__half* ptr, float value)
{
    *ptr = __float2half_rn(value);
}

template <>
__device__ __forceinline__ void storeScalar<__nv_bfloat16>(__nv_bfloat16* ptr, float value)
{
    *ptr = __float2bfloat16(value);
}

template <typename T>
__global__ void kvarnGqaDecodeReferenceKernel(T const* q, PackedRecordView records, std::int64_t const* blockIds,
    std::int32_t const* seqLens, T* output, int numQueries, int numBlocks, int numHeads, int numKvHeads,
    int seqLensCount)
{
    int query = blockIdx.x;
    int head = blockIdx.y;
    if (query >= numQueries || head >= numHeads || threadIdx.x != 0)
    {
        return;
    }

    int groups = numHeads / numKvHeads;
    int kvHead = head / groups;
    int seqLen = seqLensCount == 1 ? seqLens[0] : seqLens[query];
    int cappedSeqLen = seqLen > 0 ? seqLen : 0;
    int maxPackedTokens = numBlocks * Layout::kGroupSize;
    int totalTokens = cappedSeqLen < maxPackedTokens ? cappedSeqLen : maxPackedTokens;
    T const* qBase = q + (static_cast<std::int64_t>(query) * numHeads + head) * Layout::kHeadDim;
    T* outBase = output + (static_cast<std::int64_t>(query) * numHeads + head) * Layout::kHeadDim;

    if (totalTokens <= 0)
    {
        for (int d = 0; d < Layout::kHeadDim; ++d)
        {
            storeScalar(outBase + d, 0.0f);
        }
        return;
    }

    float qRot[Layout::kHeadDim];
    for (int d = 0; d < Layout::kHeadDim; ++d)
    {
        float acc = 0.0f;
        for (int j = 0; j < Layout::kHeadDim; ++j)
        {
            acc += loadScalar(qBase + j) * static_cast<float>(hadamardSign(j, d));
        }
        qRot[d] = acc * kHadamardScale;
    }

    float maxLogit = -FLT_MAX;
    for (int linearToken = 0; linearToken < totalTokens; ++linearToken)
    {
        int block = linearToken / Layout::kGroupSize;
        int token = linearToken - block * Layout::kGroupSize;
        std::int64_t blockId = blockIds[block];
        float dot = 0.0f;
        for (int d = 0; d < Layout::kHeadDim; ++d)
        {
            dot += qRot[d] * dequantKRot(records, blockId, kvHead, token, d);
        }
        float logit = dot * kHadamardScale;
        maxLogit = fmaxf(maxLogit, logit);
    }

    float denom = 0.0f;
    float accRot[Layout::kHeadDim];
    for (int d = 0; d < Layout::kHeadDim; ++d)
    {
        accRot[d] = 0.0f;
    }

    for (int linearToken = 0; linearToken < totalTokens; ++linearToken)
    {
        int block = linearToken / Layout::kGroupSize;
        int token = linearToken - block * Layout::kGroupSize;
        std::int64_t blockId = blockIds[block];
        float dot = 0.0f;
        for (int d = 0; d < Layout::kHeadDim; ++d)
        {
            dot += qRot[d] * dequantKRot(records, blockId, kvHead, token, d);
        }
        float weight = expf(dot * kHadamardScale - maxLogit);
        denom += weight;
        for (int d = 0; d < Layout::kHeadDim; ++d)
        {
            accRot[d] += weight * dequantVRot(records, blockId, kvHead, token, d);
        }
    }

    float invDenom = denom > 0.0f ? 1.0f / denom : 0.0f;
    for (int j = 0; j < Layout::kHeadDim; ++j)
    {
        float out = 0.0f;
        for (int d = 0; d < Layout::kHeadDim; ++d)
        {
            out += accRot[d] * invDenom * static_cast<float>(hadamardSign(j, d));
        }
        storeScalar(outBase + j, out * kHadamardScale);
    }
}

} // namespace

bool kvarnGqaBackendReady()
{
    // The decode kernel below is an experimental correctness path only: it is
    // serial per query/head and ignores fp16 sink/tail side state. Readiness must
    // stay false until store, side-state transfer, sparse reads, CUDA graph
    // lifecycle, and B200 performance gates pass.
    return false;
}

void invokeKvarnGqaStoreK2V2G128(void const*, void const*, std::uint8_t*, std::int64_t const*, int, int, int, int,
    int, bool, bool, std::int64_t, std::int64_t, std::int64_t, std::int64_t, cudaStream_t)
{
    TLLM_CHECK_WITH_INFO(false,
        "kvarn_gqa_store k2v2_g128 is registered as a production integration boundary, but the fused B200 "
        "store kernel is not implemented or validated yet");
}

void invokeKvarnGqaDecodeK2V2G128(void const* q, std::uint8_t const* packedRecords,
    std::int64_t const* blockIds, void const*, void const*, void const*, void const*, std::int32_t const* seqLens,
    void* output, int numQueries, int numBlocks, int numHeads, int numKvHeads, int headDim, int groupSize, bool useBf16,
    int seqLensCount, bool pageLayout, std::int64_t strideBlock, std::int64_t strideToken, std::int64_t strideHead,
    std::int64_t strideByte, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(headDim == Layout::kHeadDim && groupSize == Layout::kGroupSize,
        "kvarn_gqa_decode currently supports only k2v2_g128");
    TLLM_CHECK_WITH_INFO(numKvHeads > 0 && numHeads % numKvHeads == 0,
        "kvarn_gqa_decode requires num_heads divisible by num_kv_heads");
    TLLM_CHECK_WITH_INFO(numBlocks >= 0 && numQueries >= 0, "kvarn_gqa_decode got negative sizes");
    PackedRecordView view{packedRecords, pageLayout, strideBlock, strideToken, strideHead, strideByte};
    dim3 grid(numQueries, numHeads);
    if (useBf16)
    {
        kvarnGqaDecodeReferenceKernel<<<grid, 1, 0, stream>>>(static_cast<__nv_bfloat16 const*>(q), view, blockIds,
            seqLens, static_cast<__nv_bfloat16*>(output), numQueries, numBlocks, numHeads, numKvHeads, seqLensCount);
    }
    else
    {
        kvarnGqaDecodeReferenceKernel<<<grid, 1, 0, stream>>>(static_cast<__half const*>(q), view, blockIds, seqLens,
            static_cast<__half*>(output), numQueries, numBlocks, numHeads, numKvHeads, seqLensCount);
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END

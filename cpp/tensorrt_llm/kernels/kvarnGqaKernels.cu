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


struct PackedRecordWriteView
{
    std::uint8_t* ptr;
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


__device__ __forceinline__ std::uint8_t recordByte(PackedRecordWriteView view, std::int64_t blockId, int kvHead, int byteIdx)
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

__device__ __forceinline__ void writeRecordByte(
    PackedRecordWriteView view, std::int64_t blockId, int kvHead, int byteIdx, std::uint8_t value)
{
    if (view.pageLayout)
    {
        int tokenSlot = byteIdx / Layout::kBytesPerTokenSlot;
        int byteInSlot = byteIdx - tokenSlot * Layout::kBytesPerTokenSlot;
        view.ptr[blockId * view.strideBlock + tokenSlot * view.strideToken + kvHead * view.strideHead
            + byteInSlot * view.strideByte] = value;
        return;
    }
    view.ptr[blockId * view.strideBlock + kvHead * view.strideHead + byteIdx * view.strideByte] = value;
}

__device__ __forceinline__ void writePackedFp16(
    PackedRecordWriteView view, std::int64_t blockId, int kvHead, int byteOffset, float value)
{
    union
    {
        std::uint16_t u;
        __half h;
    } cvt;
    cvt.h = __float2half_rn(value);
    writeRecordByte(view, blockId, kvHead, byteOffset, static_cast<std::uint8_t>(cvt.u & 0xff));
    writeRecordByte(view, blockId, kvHead, byteOffset + 1, static_cast<std::uint8_t>((cvt.u >> 8) & 0xff));
}

__device__ __forceinline__ float clampf(float x, float lo, float hi)
{
    return fminf(fmaxf(x, lo), hi);
}

__device__ float tileStd(float const* tile, float const* logCol, float const* logRow, bool byColumn, int idx)
{
    float sum = 0.0f;
    float sumSq = 0.0f;
    for (int i = 0; i < Layout::kGroupSize; ++i)
    {
        int r = byColumn ? i : idx;
        int c = byColumn ? idx : i;
        float x = tile[r * Layout::kHeadDim + c] / expf(logRow[r] + logCol[c]);
        sum += x;
        sumSq += x * x;
    }
    float n = static_cast<float>(Layout::kGroupSize);
    float var = (sumSq - (sum * sum / n)) / (n - 1.0f);
    return sqrtf(fmaxf(var, 0.0f));
}

__device__ float tileImbalance(float const* tile, float const* logCol, float const* logRow)
{
    float minCol = FLT_MAX;
    float maxCol = 0.0f;
    float minRow = FLT_MAX;
    float maxRow = 0.0f;
    for (int i = 0; i < Layout::kGroupSize; ++i)
    {
        float sc = tileStd(tile, logCol, logRow, true, i);
        float sr = tileStd(tile, logCol, logRow, false, i);
        minCol = fminf(minCol, sc);
        maxCol = fmaxf(maxCol, sc);
        minRow = fminf(minRow, sr);
        maxRow = fmaxf(maxRow, sr);
    }
    return maxCol / fmaxf(minCol, 1e-8f) + maxRow / fmaxf(minRow, 1e-8f);
}

template <bool IsKey, typename T>
__device__ void quantizeAndWriteTile(T const* src, PackedRecordWriteView view, std::int64_t blockId, int inputBlock, int kvHead,
    int numKvHeads)
{
    float tile[Layout::kGroupSize * Layout::kHeadDim];
    float logCol[Layout::kHeadDim];
    float logRow[Layout::kGroupSize];
    float bestCol[Layout::kHeadDim];
    float bestRow[Layout::kGroupSize];

    for (int i = 0; i < Layout::kHeadDim; ++i)
    {
        logCol[i] = 0.0f;
        bestCol[i] = 1.0f;
    }
    for (int i = 0; i < Layout::kGroupSize; ++i)
    {
        logRow[i] = 0.0f;
        bestRow[i] = 1.0f;
    }

    for (int r = 0; r < Layout::kGroupSize; ++r)
    {
        for (int c = 0; c < Layout::kHeadDim; ++c)
        {
            int token = IsKey ? c : r;
            int rotDim = IsKey ? r : c;
            float acc = 0.0f;
            for (int j = 0; j < Layout::kHeadDim; ++j)
            {
                std::int64_t srcIdx
                    = ((static_cast<std::int64_t>(inputBlock) * Layout::kGroupSize + token) * numKvHeads + kvHead)
                    * Layout::kHeadDim + j;
                acc += loadScalar(src + srcIdx) * static_cast<float>(hadamardSign(j, rotDim));
            }
            tile[r * Layout::kHeadDim + c] = acc * kHadamardScale;
        }
    }

    float bestImbalance = tileImbalance(tile, logCol, logRow);
    for (int iter = 0; iter < 16; ++iter)
    {
        for (int c = 0; c < Layout::kHeadDim; ++c)
        {
            float std = clampf(tileStd(tile, logCol, logRow, true, c), 1e-3f, 1e3f);
            logCol[c] = clampf(logCol[c] + logf(std), -0.3f, 10.0f);
        }
        for (int r = 0; r < Layout::kGroupSize; ++r)
        {
            float std = clampf(tileStd(tile, logCol, logRow, false, r), 1e-3f, 1e3f);
            logRow[r] = clampf(logRow[r] + logf(std), -0.3f, 10.0f);
        }
        float imb = tileImbalance(tile, logCol, logRow);
        if (imb <= bestImbalance)
        {
            bestImbalance = imb;
            for (int c = 0; c < Layout::kHeadDim; ++c)
            {
                bestCol[c] = expf(logCol[c]);
            }
            for (int r = 0; r < Layout::kGroupSize; ++r)
            {
                bestRow[r] = expf(logRow[r]);
            }
        }
    }

    int packedOffset = IsKey ? kKPackedOffset : kVPackedOffset;
    int sRowOffset = IsKey ? kKSRowAbsOffset : kVSRowAbsOffset;
    int zpOffset = IsKey ? kKZpAbsOffset : kVZpAbsOffset;
    int sColOffset = IsKey ? kKSColOffset : kVSColOffset;
    for (int i = 0; i < 4096; ++i)
    {
        writeRecordByte(view, blockId, kvHead, packedOffset + i, 0);
    }

    for (int r = 0; r < Layout::kGroupSize; ++r)
    {
        float lo = FLT_MAX;
        float hi = -FLT_MAX;
        for (int c = 0; c < Layout::kHeadDim; ++c)
        {
            float balanced = tile[r * Layout::kHeadDim + c] / bestRow[r] / bestCol[c];
            lo = fminf(lo, balanced);
            hi = fmaxf(hi, balanced);
        }
        float scale = fmaxf((hi - lo) / 3.0f, 1e-10f);
        writePackedFp16(view, blockId, kvHead, sRowOffset + r * 2, bestRow[r] * scale);
        writePackedFp16(view, blockId, kvHead, zpOffset + r * 2, bestRow[r] * lo);
        for (int c = 0; c < Layout::kHeadDim; ++c)
        {
            float balanced = tile[r * Layout::kHeadDim + c] / bestRow[r] / bestCol[c];
            int q = static_cast<int>(floorf((balanced - lo) / scale + 0.5f));
            q = q < 0 ? 0 : (q > 3 ? 3 : q);
            int valueIdx = r * Layout::kHeadDim + c;
            int bit = valueIdx * 2;
            int byteIdx = packedOffset + (bit >> 3);
            int shift = bit & 7;
            std::uint8_t old = recordByte(view, blockId, kvHead, byteIdx);
            writeRecordByte(view, blockId, kvHead, byteIdx, old | static_cast<std::uint8_t>(q << shift));
        }
    }
    for (int c = 0; c < Layout::kHeadDim; ++c)
    {
        writePackedFp16(view, blockId, kvHead, sColOffset + c * 2, bestCol[c]);
    }
}

template <typename T>
__global__ void kvarnGqaStoreReferenceKernel(T const* k, T const* v, PackedRecordWriteView records,
    std::int64_t const* blockIds, int numBlocks, int numKvHeads)
{
    int inputBlock = blockIdx.x;
    int kvHead = blockIdx.y;
    if (inputBlock >= numBlocks || kvHead >= numKvHeads || threadIdx.x != 0)
    {
        return;
    }
    std::int64_t blockId = blockIds[inputBlock];
    quantizeAndWriteTile<true>(k, records, blockId, inputBlock, kvHead, numKvHeads);
    quantizeAndWriteTile<false>(v, records, blockId, inputBlock, kvHead, numKvHeads);
}

template <typename T>
__device__ __forceinline__ float loadSideScalar(
    T const* side, int query, int batch, int token, int kvHead, int dim, int tokens, int numKvHeads)
{
    int sideBatch = batch == 1 ? 0 : query;
    std::int64_t idx = ((static_cast<std::int64_t>(sideBatch) * tokens + token) * numKvHeads + kvHead)
        * Layout::kHeadDim + dim;
    return loadScalar(side + idx);
}

template <typename T>
__device__ float loadSideRotated(
    T const* side, int query, int batch, int token, int kvHead, int rotDim, int tokens, int numKvHeads)
{
    float acc = 0.0f;
    for (int j = 0; j < Layout::kHeadDim; ++j)
    {
        acc += loadSideScalar(side, query, batch, token, kvHead, j, tokens, numKvHeads)
            * static_cast<float>(hadamardSign(j, rotDim));
    }
    return acc * kHadamardScale;
}

template <typename T>
__device__ float loadKRotatedForLogicalToken(T const* sinkK, T const* tailK, PackedRecordView records,
    std::int64_t const* blockIds, int query, int kvHead, int dim, int logicalToken, int sinkCount, int packedCount,
    int sinkTokens, int sinkBatch, int tailTokens, int tailBatch, int numKvHeads)
{
    if (logicalToken < sinkCount)
    {
        return loadSideRotated(sinkK, query, sinkBatch, logicalToken, kvHead, dim, sinkTokens, numKvHeads);
    }
    int afterSink = logicalToken - sinkCount;
    if (afterSink < packedCount)
    {
        int block = afterSink / Layout::kGroupSize;
        int token = afterSink - block * Layout::kGroupSize;
        return dequantKRot(records, blockIds[block], kvHead, token, dim);
    }
    int tailToken = afterSink - packedCount;
    return loadSideRotated(tailK, query, tailBatch, tailToken, kvHead, dim, tailTokens, numKvHeads);
}

template <typename T>
__device__ float loadVRotatedForLogicalToken(T const* sinkV, T const* tailV, PackedRecordView records,
    std::int64_t const* blockIds, int query, int kvHead, int dim, int logicalToken, int sinkCount, int packedCount,
    int sinkTokens, int sinkBatch, int tailTokens, int tailBatch, int numKvHeads)
{
    if (logicalToken < sinkCount)
    {
        return loadSideRotated(sinkV, query, sinkBatch, logicalToken, kvHead, dim, sinkTokens, numKvHeads);
    }
    int afterSink = logicalToken - sinkCount;
    if (afterSink < packedCount)
    {
        int block = afterSink / Layout::kGroupSize;
        int token = afterSink - block * Layout::kGroupSize;
        return dequantVRot(records, blockIds[block], kvHead, token, dim);
    }
    int tailToken = afterSink - packedCount;
    return loadSideRotated(tailV, query, tailBatch, tailToken, kvHead, dim, tailTokens, numKvHeads);
}

template <typename T>
__global__ void kvarnGqaDequantAmortizedKernel(PackedRecordView records, std::int64_t const* blockIds,
    T* readableK, T* readableV, int numChurnBlocks, int numPhysicalBlocks, int numKvHeads)
{
    int churnIdx = blockIdx.x;
    int kvHead = blockIdx.y;
    if (churnIdx >= numChurnBlocks || kvHead >= numKvHeads)
    {
        return;
    }
    std::int64_t blockId = blockIds[churnIdx];
    if (blockId < 0 || blockId >= numPhysicalBlocks)
    {
        return;
    }

    int total = Layout::kGroupSize * Layout::kHeadDim;
    for (int linear = threadIdx.x; linear < total; linear += blockDim.x)
    {
        int token = linear / Layout::kHeadDim;
        int dim = linear - token * Layout::kHeadDim;
        float kOut = 0.0f;
        float vOut = 0.0f;
        for (int rotDim = 0; rotDim < Layout::kHeadDim; ++rotDim)
        {
            float sign = static_cast<float>(hadamardSign(dim, rotDim));
            kOut += dequantKRot(records, blockId, kvHead, token, rotDim) * sign;
            vOut += dequantVRot(records, blockId, kvHead, token, rotDim) * sign;
        }
        std::int64_t outIdx = ((blockId * Layout::kGroupSize + token) * numKvHeads + kvHead) * Layout::kHeadDim + dim;
        storeScalar(readableK + outIdx, kOut * kHadamardScale);
        storeScalar(readableV + outIdx, vOut * kHadamardScale);
    }
}

template <typename T>
__global__ void kvarnGqaDecodeReferenceKernel(T const* q, PackedRecordView records, std::int64_t const* blockIds,
    T const* sinkK, T const* sinkV, T const* tailK, T const* tailV, std::int32_t const* seqLens, T* output,
    int numQueries, int numBlocks, int numHeads, int numKvHeads, int seqLensCount, int sinkTokens, int sinkBatch,
    int tailTokens, int tailBatch)
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
    int sinkCount = sinkTokens > 0 ? (cappedSeqLen < sinkTokens ? cappedSeqLen : sinkTokens) : 0;
    int remainingAfterSink = cappedSeqLen - sinkCount;
    int maxPackedTokens = numBlocks * Layout::kGroupSize;
    int packedCount = remainingAfterSink < maxPackedTokens ? remainingAfterSink : maxPackedTokens;
    int remainingAfterPacked = remainingAfterSink - packedCount;
    int tailCount = tailTokens > 0 ? (remainingAfterPacked < tailTokens ? remainingAfterPacked : tailTokens) : 0;
    int totalTokens = sinkCount + packedCount + tailCount;
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
        float dot = 0.0f;
        for (int d = 0; d < Layout::kHeadDim; ++d)
        {
            dot += qRot[d] * loadKRotatedForLogicalToken(sinkK, tailK, records, blockIds, query, kvHead, d,
                linearToken, sinkCount, packedCount, sinkTokens, sinkBatch, tailTokens, tailBatch, numKvHeads);
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
        float dot = 0.0f;
        for (int d = 0; d < Layout::kHeadDim; ++d)
        {
            dot += qRot[d] * loadKRotatedForLogicalToken(sinkK, tailK, records, blockIds, query, kvHead, d,
                linearToken, sinkCount, packedCount, sinkTokens, sinkBatch, tailTokens, tailBatch, numKvHeads);
        }
        float weight = expf(dot * kHadamardScale - maxLogit);
        denom += weight;
        for (int d = 0; d < Layout::kHeadDim; ++d)
        {
            accRot[d] += weight * loadVRotatedForLogicalToken(sinkV, tailV, records, blockIds, query, kvHead, d,
                linearToken, sinkCount, packedCount, sinkTokens, sinkBatch, tailTokens, tailBatch, numKvHeads);
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


template <typename T, int THREADS, int MAX_TOKENS>
__global__ void kvarnGqaDecodeSmallKernel(T const* q, PackedRecordView records, std::int64_t const* blockIds,
    T const* sinkK, T const* sinkV, T const* tailK, T const* tailV, std::int32_t const* seqLens, T* output,
    int numQueries, int numBlocks, int numHeads, int numKvHeads, int seqLensCount, int sinkTokens, int sinkBatch,
    int tailTokens, int tailBatch)
{
    int query = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;
    if (query >= numQueries || head >= numHeads)
    {
        return;
    }

    __shared__ float qRot[Layout::kHeadDim];
    __shared__ float accRot[Layout::kHeadDim];
    __shared__ float logits[MAX_TOKENS];
    __shared__ float red[THREADS];

    int groups = numHeads / numKvHeads;
    int kvHead = head / groups;
    int seqLen = seqLensCount == 1 ? seqLens[0] : seqLens[query];
    int cappedSeqLen = seqLen > 0 ? seqLen : 0;
    int sinkCount = sinkTokens > 0 ? (cappedSeqLen < sinkTokens ? cappedSeqLen : sinkTokens) : 0;
    int remainingAfterSink = cappedSeqLen - sinkCount;
    int maxPackedTokens = numBlocks * Layout::kGroupSize;
    int packedCount = remainingAfterSink < maxPackedTokens ? remainingAfterSink : maxPackedTokens;
    int remainingAfterPacked = remainingAfterSink - packedCount;
    int tailCount = tailTokens > 0 ? (remainingAfterPacked < tailTokens ? remainingAfterPacked : tailTokens) : 0;
    int totalTokens = sinkCount + packedCount + tailCount;
    T const* qBase = q + (static_cast<std::int64_t>(query) * numHeads + head) * Layout::kHeadDim;
    T* outBase = output + (static_cast<std::int64_t>(query) * numHeads + head) * Layout::kHeadDim;

    for (int d = tid; d < Layout::kHeadDim; d += THREADS)
    {
        float acc = 0.0f;
        for (int j = 0; j < Layout::kHeadDim; ++j)
        {
            acc += loadScalar(qBase + j) * static_cast<float>(hadamardSign(j, d));
        }
        qRot[d] = acc * kHadamardScale;
        accRot[d] = 0.0f;
    }
    __syncthreads();

    if (totalTokens <= 0)
    {
        for (int d = tid; d < Layout::kHeadDim; d += THREADS)
        {
            storeScalar(outBase + d, 0.0f);
        }
        return;
    }
    if (totalTokens > MAX_TOKENS)
    {
        return;
    }

    float localMax = -FLT_MAX;
    for (int linearToken = tid; linearToken < totalTokens; linearToken += THREADS)
    {
        float dot = 0.0f;
        for (int d = 0; d < Layout::kHeadDim; ++d)
        {
            dot += qRot[d] * loadKRotatedForLogicalToken(sinkK, tailK, records, blockIds, query, kvHead, d,
                linearToken, sinkCount, packedCount, sinkTokens, sinkBatch, tailTokens, tailBatch, numKvHeads);
        }
        float logit = dot * kHadamardScale;
        logits[linearToken] = logit;
        localMax = fmaxf(localMax, logit);
    }
    red[tid] = localMax;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            red[tid] = fmaxf(red[tid], red[tid + stride]);
        }
        __syncthreads();
    }
    float maxLogit = red[0];

    float localDenom = 0.0f;
    for (int linearToken = tid; linearToken < totalTokens; linearToken += THREADS)
    {
        float weight = expf(logits[linearToken] - maxLogit);
        logits[linearToken] = weight;
        localDenom += weight;
    }
    red[tid] = localDenom;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            red[tid] += red[tid + stride];
        }
        __syncthreads();
    }
    float invDenom = red[0] > 0.0f ? 1.0f / red[0] : 0.0f;

    for (int d = tid; d < Layout::kHeadDim; d += THREADS)
    {
        float acc = 0.0f;
        for (int linearToken = 0; linearToken < totalTokens; ++linearToken)
        {
            acc += logits[linearToken] * loadVRotatedForLogicalToken(sinkV, tailV, records, blockIds, query, kvHead, d,
                linearToken, sinkCount, packedCount, sinkTokens, sinkBatch, tailTokens, tailBatch, numKvHeads);
        }
        accRot[d] = acc;
    }
    __syncthreads();

    for (int j = tid; j < Layout::kHeadDim; j += THREADS)
    {
        float out = 0.0f;
        for (int d = 0; d < Layout::kHeadDim; ++d)
        {
            out += accRot[d] * invDenom * static_cast<float>(hadamardSign(j, d));
        }
        storeScalar(outBase + j, out * kHadamardScale);
    }
}

template <typename T, int THREADS>
__global__ void kvarnGqaDecodeParallelKernel(T const* q, PackedRecordView records, std::int64_t const* blockIds,
    T const* sinkK, T const* sinkV, T const* tailK, T const* tailV, std::int32_t const* seqLens, T* output,
    int numQueries, int numBlocks, int numHeads, int numKvHeads, int seqLensCount, int sinkTokens, int sinkBatch,
    int tailTokens, int tailBatch)
{
    int query = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;
    if (query >= numQueries || head >= numHeads)
    {
        return;
    }

    __shared__ float qRot[Layout::kHeadDim];
    __shared__ float accRot[Layout::kHeadDim];
    __shared__ float red[THREADS];

    int groups = numHeads / numKvHeads;
    int kvHead = head / groups;
    int seqLen = seqLensCount == 1 ? seqLens[0] : seqLens[query];
    int cappedSeqLen = seqLen > 0 ? seqLen : 0;
    int sinkCount = sinkTokens > 0 ? (cappedSeqLen < sinkTokens ? cappedSeqLen : sinkTokens) : 0;
    int remainingAfterSink = cappedSeqLen - sinkCount;
    int maxPackedTokens = numBlocks * Layout::kGroupSize;
    int packedCount = remainingAfterSink < maxPackedTokens ? remainingAfterSink : maxPackedTokens;
    int remainingAfterPacked = remainingAfterSink - packedCount;
    int tailCount = tailTokens > 0 ? (remainingAfterPacked < tailTokens ? remainingAfterPacked : tailTokens) : 0;
    int totalTokens = sinkCount + packedCount + tailCount;
    T const* qBase = q + (static_cast<std::int64_t>(query) * numHeads + head) * Layout::kHeadDim;
    T* outBase = output + (static_cast<std::int64_t>(query) * numHeads + head) * Layout::kHeadDim;

    for (int d = tid; d < Layout::kHeadDim; d += THREADS)
    {
        float acc = 0.0f;
        for (int j = 0; j < Layout::kHeadDim; ++j)
        {
            acc += loadScalar(qBase + j) * static_cast<float>(hadamardSign(j, d));
        }
        qRot[d] = acc * kHadamardScale;
        accRot[d] = 0.0f;
    }
    __syncthreads();

    if (totalTokens <= 0)
    {
        for (int d = tid; d < Layout::kHeadDim; d += THREADS)
        {
            storeScalar(outBase + d, 0.0f);
        }
        return;
    }

    float localMax = -FLT_MAX;
    for (int linearToken = tid; linearToken < totalTokens; linearToken += THREADS)
    {
        float dot = 0.0f;
        for (int d = 0; d < Layout::kHeadDim; ++d)
        {
            dot += qRot[d] * loadKRotatedForLogicalToken(sinkK, tailK, records, blockIds, query, kvHead, d,
                linearToken, sinkCount, packedCount, sinkTokens, sinkBatch, tailTokens, tailBatch, numKvHeads);
        }
        localMax = fmaxf(localMax, dot * kHadamardScale);
    }
    red[tid] = localMax;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            red[tid] = fmaxf(red[tid], red[tid + stride]);
        }
        __syncthreads();
    }
    float maxLogit = red[0];

    float localDenom = 0.0f;
    for (int linearToken = tid; linearToken < totalTokens; linearToken += THREADS)
    {
        float dot = 0.0f;
        for (int d = 0; d < Layout::kHeadDim; ++d)
        {
            dot += qRot[d] * loadKRotatedForLogicalToken(sinkK, tailK, records, blockIds, query, kvHead, d,
                linearToken, sinkCount, packedCount, sinkTokens, sinkBatch, tailTokens, tailBatch, numKvHeads);
        }
        float weight = expf(dot * kHadamardScale - maxLogit);
        localDenom += weight;
        for (int d = 0; d < Layout::kHeadDim; ++d)
        {
            float vRot = loadVRotatedForLogicalToken(sinkV, tailV, records, blockIds, query, kvHead, d,
                linearToken, sinkCount, packedCount, sinkTokens, sinkBatch, tailTokens, tailBatch, numKvHeads);
            atomicAdd(&accRot[d], weight * vRot);
        }
    }
    red[tid] = localDenom;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            red[tid] += red[tid + stride];
        }
        __syncthreads();
    }
    float invDenom = red[0] > 0.0f ? 1.0f / red[0] : 0.0f;

    for (int j = tid; j < Layout::kHeadDim; j += THREADS)
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
    // The store kernel and BDR helpers are still experimental correctness paths.
    // Decode now uses a block-parallel packed read/dequant/scoring kernel, but readiness must stay false
    // until disaggregated side-state transfer, sparse reads, CUDA graph lifecycle,
    // runtime correctness, and B200 performance gates pass.
    return false;
}

void invokeKvarnGqaStoreK2V2G128(void const* k, void const* v, std::uint8_t* packedRecords,
    std::int64_t const* blockIds, int, int numBlocks, int numKvHeads, int headDim, int groupSize, bool useBf16,
    bool pageLayout, std::int64_t strideBlock, std::int64_t strideToken, std::int64_t strideHead,
    std::int64_t strideByte, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(headDim == Layout::kHeadDim && groupSize == Layout::kGroupSize,
        "kvarn_gqa_store currently supports only k2v2_g128");
    TLLM_CHECK_WITH_INFO(numBlocks >= 0 && numKvHeads > 0, "kvarn_gqa_store got invalid sizes");
    PackedRecordWriteView view{packedRecords, pageLayout, strideBlock, strideToken, strideHead, strideByte};
    dim3 grid(numBlocks, numKvHeads);
    if (useBf16)
    {
        kvarnGqaStoreReferenceKernel<<<grid, 1, 0, stream>>>(static_cast<__nv_bfloat16 const*>(k),
            static_cast<__nv_bfloat16 const*>(v), view, blockIds, numBlocks, numKvHeads);
    }
    else
    {
        kvarnGqaStoreReferenceKernel<<<grid, 1, 0, stream>>>(static_cast<__half const*>(k),
            static_cast<__half const*>(v), view, blockIds, numBlocks, numKvHeads);
    }
}

void invokeKvarnGqaDequantAmortizedK2V2G128(std::uint8_t const* packedRecords, std::int64_t const* blockIds,
    void* readableK, void* readableV, int numChurnBlocks, int numPhysicalBlocks, int numKvHeads, int headDim,
    int groupSize, bool useBf16, bool pageLayout, std::int64_t strideBlock, std::int64_t strideToken,
    std::int64_t strideHead, std::int64_t strideByte, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(headDim == Layout::kHeadDim && groupSize == Layout::kGroupSize,
        "kvarn_gqa_dequant_amortized currently supports only k2v2_g128");
    TLLM_CHECK_WITH_INFO(numChurnBlocks >= 0 && numPhysicalBlocks >= 0 && numKvHeads > 0,
        "kvarn_gqa_dequant_amortized got invalid sizes");
    if (numChurnBlocks == 0)
    {
        return;
    }
    PackedRecordView view{packedRecords, pageLayout, strideBlock, strideToken, strideHead, strideByte};
    dim3 grid(numChurnBlocks, numKvHeads);
    constexpr int kThreads = 256;
    if (useBf16)
    {
        kvarnGqaDequantAmortizedKernel<<<grid, kThreads, 0, stream>>>(view, blockIds,
            static_cast<__nv_bfloat16*>(readableK), static_cast<__nv_bfloat16*>(readableV),
            numChurnBlocks, numPhysicalBlocks, numKvHeads);
    }
    else
    {
        kvarnGqaDequantAmortizedKernel<<<grid, kThreads, 0, stream>>>(view, blockIds,
            static_cast<__half*>(readableK), static_cast<__half*>(readableV), numChurnBlocks, numPhysicalBlocks,
            numKvHeads);
    }
}

void invokeKvarnGqaDecodeK2V2G128(void const* q, std::uint8_t const* packedRecords,
    std::int64_t const* blockIds, void const* sinkK, void const* sinkV, void const* tailK, void const* tailV,
    std::int32_t const* seqLens, void* output, int numQueries, int numBlocks, int numHeads, int numKvHeads, int headDim,
    int groupSize, bool useBf16, int seqLensCount, int sinkTokens, int sinkBatch, int tailTokens, int tailBatch,
    bool pageLayout, std::int64_t strideBlock, std::int64_t strideToken, std::int64_t strideHead, std::int64_t strideByte,
    cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(headDim == Layout::kHeadDim && groupSize == Layout::kGroupSize,
        "kvarn_gqa_decode currently supports only k2v2_g128");
    TLLM_CHECK_WITH_INFO(numKvHeads > 0 && numHeads % numKvHeads == 0,
        "kvarn_gqa_decode requires num_heads divisible by num_kv_heads");
    TLLM_CHECK_WITH_INFO(numBlocks >= 0 && numQueries >= 0, "kvarn_gqa_decode got negative sizes");
    PackedRecordView view{packedRecords, pageLayout, strideBlock, strideToken, strideHead, strideByte};
    dim3 grid(numQueries, numHeads);
    bool useSmallDecode = (sinkTokens + numBlocks * Layout::kGroupSize + tailTokens) <= 256;
    if (useBf16)
    {
        if (useSmallDecode)
        {
            kvarnGqaDecodeSmallKernel<__nv_bfloat16, 256, 256><<<grid, 256, 0, stream>>>(
                static_cast<__nv_bfloat16 const*>(q), view, blockIds, static_cast<__nv_bfloat16 const*>(sinkK),
                static_cast<__nv_bfloat16 const*>(sinkV), static_cast<__nv_bfloat16 const*>(tailK),
                static_cast<__nv_bfloat16 const*>(tailV), seqLens, static_cast<__nv_bfloat16*>(output), numQueries,
                numBlocks, numHeads, numKvHeads, seqLensCount, sinkTokens, sinkBatch, tailTokens, tailBatch);
        }
        else
        {
            kvarnGqaDecodeParallelKernel<__nv_bfloat16, 256><<<grid, 256, 0, stream>>>(
                static_cast<__nv_bfloat16 const*>(q), view, blockIds, static_cast<__nv_bfloat16 const*>(sinkK),
                static_cast<__nv_bfloat16 const*>(sinkV), static_cast<__nv_bfloat16 const*>(tailK),
                static_cast<__nv_bfloat16 const*>(tailV), seqLens, static_cast<__nv_bfloat16*>(output), numQueries,
                numBlocks, numHeads, numKvHeads, seqLensCount, sinkTokens, sinkBatch, tailTokens, tailBatch);
        }
    }
    else
    {
        if (useSmallDecode)
        {
            kvarnGqaDecodeSmallKernel<__half, 256, 256><<<grid, 256, 0, stream>>>(static_cast<__half const*>(q),
                view, blockIds, static_cast<__half const*>(sinkK), static_cast<__half const*>(sinkV),
                static_cast<__half const*>(tailK), static_cast<__half const*>(tailV), seqLens, static_cast<__half*>(output),
                numQueries, numBlocks, numHeads, numKvHeads, seqLensCount, sinkTokens, sinkBatch, tailTokens, tailBatch);
        }
        else
        {
            kvarnGqaDecodeParallelKernel<__half, 256><<<grid, 256, 0, stream>>>(static_cast<__half const*>(q),
                view, blockIds, static_cast<__half const*>(sinkK), static_cast<__half const*>(sinkV),
                static_cast<__half const*>(tailK), static_cast<__half const*>(tailV), seqLens, static_cast<__half*>(output),
                numQueries, numBlocks, numHeads, numKvHeads, seqLensCount, sinkTokens, sinkBatch, tailTokens, tailBatch);
        }
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END

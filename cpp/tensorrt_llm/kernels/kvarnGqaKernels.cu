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

void checkKvarnGqaCuda(cudaError_t err, char const* what)
{
    TLLM_CHECK_WITH_INFO(err == cudaSuccess, "%s failed: %s", what, cudaGetErrorString(err));
}

void checkKvarnGqaLaunch(char const* what)
{
    checkKvarnGqaCuda(cudaGetLastError(), what);
}

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

// In-place fast Walsh-Hadamard transform of a single 128-element vector held in
// shared memory, parallelized over the block's threads. Natural (Sylvester)
// order: result[d] = sum_j vec[j] * (-1)^popcount(j & d), i.e. exactly H @ vec
// where H is the symmetric Hadamard matrix used by hadamardSign. log2(128)=7
// butterfly stages of 64 add/sub pairs replace the O(N^2) matrix-sum. The caller
// applies kHadamardScale (1/sqrt(128)) afterwards if a normalized transform is
// wanted. Must be entered with all threads converged; issues a __syncthreads()
// between stages, so all threads in the block must call it.
__device__ __forceinline__ void fwht128Shared(float* vec)
{
    constexpr int kN = Layout::kHeadDim;       // 128
    constexpr int kPairs = kN / 2;             // 64
    int tid = threadIdx.x;
    __syncthreads();
    for (int half = 1; half < kN; half <<= 1)
    {
        for (int p = tid; p < kPairs; p += blockDim.x)
        {
            int group = p / half;
            int k = p - group * half;
            int i = group * (half << 1) + k;
            int j = i + half;
            float u = vec[i];
            float w = vec[j];
            vec[i] = u + w;
            vec[j] = u - w;
        }
        __syncthreads();
    }
}

// Single-thread in-place FWHT of a 128-element local/register array. Same
// natural-order convention as fwht128Shared / hadamardSign.
__device__ __forceinline__ void fwht128Local(float* vec)
{
    constexpr int kN = Layout::kHeadDim;       // 128
    for (int half = 1; half < kN; half <<= 1)
    {
        for (int i = 0; i < kN; i += (half << 1))
        {
            for (int k = 0; k < half; ++k)
            {
                float u = vec[i + k];
                float w = vec[i + k + half];
                vec[i + k] = u + w;
                vec[i + k + half] = u - w;
            }
        }
    }
}

// Batched in-place FWHT of `numRows` contiguous 128-element rows in shared
// memory (rows[row * 128 + d]). All rows transform simultaneously; threads stride
// over the (row, pair) space and a __syncthreads() separates the 7 butterfly
// stages, so every block thread must call this. Replaces the O(numRows * N^2)
// per-row matrix-sum with O(numRows * N log N).
__device__ __forceinline__ void fwht128SharedBatched(float* rows, int numRows)
{
    constexpr int kN = Layout::kHeadDim;       // 128
    constexpr int kPairs = kN / 2;             // 64
    int tid = threadIdx.x;
    int totalPairs = numRows * kPairs;
    __syncthreads();
    for (int half = 1; half < kN; half <<= 1)
    {
        for (int p = tid; p < totalPairs; p += blockDim.x)
        {
            int row = p / kPairs;
            int pp = p - row * kPairs;
            int group = pp / half;
            int k = pp - group * half;
            int base = row * kN + group * (half << 1) + k;
            float u = rows[base];
            float w = rows[base + half];
            rows[base] = u + w;
            rows[base + half] = u - w;
        }
        __syncthreads();
    }
}

// In-place transpose of a 128x128 row-major shared-memory matrix. Threads stride
// over the strict upper triangle and swap (i,j) with (j,i). Caller must sync
// before relying on the result (a trailing __syncthreads() is issued).
__device__ __forceinline__ void transpose128Shared(float* mat)
{
    constexpr int kN = Layout::kHeadDim;       // 128
    for (int idx = threadIdx.x; idx < kN * kN; idx += blockDim.x)
    {
        int i = idx / kN;
        int j = idx - i * kN;
        if (i < j)
        {
            float tmp = mat[i * kN + j];
            mat[i * kN + j] = mat[j * kN + i];
            mat[j * kN + i] = tmp;
        }
    }
    __syncthreads();
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

// Cooperatively stage one 4096-byte packed code plane (K or V) for a block into
// shared memory so the score/value loops unpack codes from smem instead of issuing
// strided/page-addressed global byte loads.
__device__ __forceinline__ void stageCodePlane(
    PackedRecordView view, std::int64_t blockId, int kvHead, int baseOffset, std::uint8_t* dst)
{
    constexpr int kPlaneBytes = Layout::kHeadDim * Layout::kGroupSize / 4;  // 4096
    for (int i = threadIdx.x; i < kPlaneBytes; i += blockDim.x)
    {
        dst[i] = recordByte(view, blockId, kvHead, baseOffset + i);
    }
}

__device__ __forceinline__ int readPacked2Smem(std::uint8_t const* plane, int valueIdx)
{
    int bit = valueIdx * 2;
    return (plane[bit >> 3] >> (bit & 7)) & 0x3;
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

__device__ float tileStdScaled(float const* tile, float const* invCol, float const* invRow, bool byColumn, int idx)
{
    float sum = 0.0f;
    float sumSq = 0.0f;
    for (int i = 0; i < Layout::kGroupSize; ++i)
    {
        int r = byColumn ? i : idx;
        int c = byColumn ? idx : i;
        float x = tile[r * Layout::kHeadDim + c] * invRow[r] * invCol[c];
        sum += x;
        sumSq += x * x;
    }
    float n = static_cast<float>(Layout::kGroupSize);
    float var = (sumSq - (sum * sum / n)) / (n - 1.0f);
    return sqrtf(fmaxf(var, 0.0f));
}

template <bool IsKey, typename T>
__device__ void quantizeAndWriteTileParallel(T const* src, PackedRecordWriteView view, std::int64_t blockId,
    int inputBlock, int kvHead, int numKvHeads, float* smem)
{
    float* tile = smem;
    float* logCol = tile + Layout::kGroupSize * Layout::kHeadDim;
    float* logRow = logCol + Layout::kHeadDim;
    float* bestCol = logRow + Layout::kGroupSize;
    float* bestRow = bestCol + Layout::kHeadDim;
    float* tmpCol = bestRow + Layout::kGroupSize;
    float* tmpRow = tmpCol + Layout::kHeadDim;
    float* invCol = tmpRow + Layout::kGroupSize;
    float* invRow = invCol + Layout::kHeadDim;
    float* scalar = invRow + Layout::kGroupSize;
    int tid = threadIdx.x;

    for (int i = tid; i < Layout::kHeadDim; i += blockDim.x)
    {
        logCol[i] = 0.0f;
        bestCol[i] = 1.0f;
    }
    for (int i = tid; i < Layout::kGroupSize; i += blockDim.x)
    {
        logRow[i] = 0.0f;
        bestRow[i] = 1.0f;
    }
    __syncthreads();

    // Stage src into tile in token-major [token][j] order, then rotate every token
    // row with a batched in-place FWHT (O(N log N) per row vs the O(N^2) matrix-sum).
    // Key needs the rotated tile in [rotDim][token] order, so transpose after.
    int total = Layout::kGroupSize * Layout::kHeadDim;
    for (int linear = tid; linear < total; linear += blockDim.x)
    {
        int token = linear / Layout::kHeadDim;
        int j = linear - token * Layout::kHeadDim;
        std::int64_t srcIdx
            = ((static_cast<std::int64_t>(inputBlock) * Layout::kGroupSize + token) * numKvHeads + kvHead)
            * Layout::kHeadDim + j;
        tile[linear] = loadScalar(src + srcIdx);
    }
    fwht128SharedBatched(tile, Layout::kGroupSize);
    for (int linear = tid; linear < total; linear += blockDim.x)
    {
        tile[linear] *= kHadamardScale;
    }
    __syncthreads();
    if (IsKey)
    {
        transpose128Shared(tile);
    }

    if (tid < Layout::kHeadDim)
    {
        tmpCol[tid] = tileStd(tile, logCol, logRow, true, tid);
        tmpRow[tid] = tileStd(tile, logCol, logRow, false, tid);
    }
    __syncthreads();
    if (tid == 0)
    {
        float minCol = FLT_MAX;
        float maxCol = 0.0f;
        float minRow = FLT_MAX;
        float maxRow = 0.0f;
        for (int i = 0; i < Layout::kGroupSize; ++i)
        {
            minCol = fminf(minCol, tmpCol[i]);
            maxCol = fmaxf(maxCol, tmpCol[i]);
            minRow = fminf(minRow, tmpRow[i]);
            maxRow = fmaxf(maxRow, tmpRow[i]);
        }
        scalar[0] = maxCol / fmaxf(minCol, 1e-8f) + maxRow / fmaxf(minRow, 1e-8f);
    }
    __syncthreads();

    for (int iter = 0; iter < 16; ++iter)
    {
        if (tid < Layout::kHeadDim)
        {
            invCol[tid] = expf(-logCol[tid]);
            invRow[tid] = expf(-logRow[tid]);
        }
        __syncthreads();
        if (tid < Layout::kHeadDim)
        {
            float std = clampf(tileStdScaled(tile, invCol, invRow, true, tid), 1e-3f, 1e3f);
            logCol[tid] = clampf(logCol[tid] + logf(std), -0.3f, 10.0f);
        }
        __syncthreads();
        if (tid < Layout::kHeadDim)
        {
            invCol[tid] = expf(-logCol[tid]);
        }
        __syncthreads();
        if (tid < Layout::kGroupSize)
        {
            float std = clampf(tileStdScaled(tile, invCol, invRow, false, tid), 1e-3f, 1e3f);
            logRow[tid] = clampf(logRow[tid] + logf(std), -0.3f, 10.0f);
        }
        __syncthreads();
        if (tid < Layout::kHeadDim)
        {
            invCol[tid] = expf(-logCol[tid]);
            invRow[tid] = expf(-logRow[tid]);
        }
        __syncthreads();
        if (tid < Layout::kHeadDim)
        {
            tmpCol[tid] = tileStdScaled(tile, invCol, invRow, true, tid);
            tmpRow[tid] = tileStdScaled(tile, invCol, invRow, false, tid);
        }
        __syncthreads();
        if (tid == 0)
        {
            float minCol = FLT_MAX;
            float maxCol = 0.0f;
            float minRow = FLT_MAX;
            float maxRow = 0.0f;
            for (int i = 0; i < Layout::kGroupSize; ++i)
            {
                minCol = fminf(minCol, tmpCol[i]);
                maxCol = fmaxf(maxCol, tmpCol[i]);
                minRow = fminf(minRow, tmpRow[i]);
                maxRow = fmaxf(maxRow, tmpRow[i]);
            }
            float imb = maxCol / fmaxf(minCol, 1e-8f) + maxRow / fmaxf(minRow, 1e-8f);
            if (imb <= scalar[0])
            {
                scalar[0] = imb;
                for (int i = 0; i < Layout::kHeadDim; ++i)
                {
                    bestCol[i] = expf(logCol[i]);
                    bestRow[i] = expf(logRow[i]);
                }
            }
        }
        __syncthreads();
    }

    int packedOffset = IsKey ? kKPackedOffset : kVPackedOffset;
    int sRowOffset = IsKey ? kKSRowAbsOffset : kVSRowAbsOffset;
    int zpOffset = IsKey ? kKZpAbsOffset : kVZpAbsOffset;
    int sColOffset = IsKey ? kKSColOffset : kVSColOffset;
    for (int r = tid; r < Layout::kGroupSize; r += blockDim.x)
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
        for (int byteCol = 0; byteCol < Layout::kHeadDim / 4; ++byteCol)
        {
            std::uint8_t packed = 0;
            for (int lane = 0; lane < 4; ++lane)
            {
                int c = byteCol * 4 + lane;
                float balanced = tile[r * Layout::kHeadDim + c] / bestRow[r] / bestCol[c];
                int q = static_cast<int>(floorf((balanced - lo) / scale + 0.5f));
                q = q < 0 ? 0 : (q > 3 ? 3 : q);
                packed |= static_cast<std::uint8_t>(q << (lane * 2));
            }
            int byteIdx = packedOffset + r * (Layout::kHeadDim / 4) + byteCol;
            writeRecordByte(view, blockId, kvHead, byteIdx, packed);
        }
    }
    for (int c = tid; c < Layout::kHeadDim; c += blockDim.x)
    {
        writePackedFp16(view, blockId, kvHead, sColOffset + c * 2, bestCol[c]);
    }
    __syncthreads();
}

template <typename T>
__global__ void kvarnGqaStoreParallelKernel(T const* k, T const* v, PackedRecordWriteView records,
    std::int64_t const* blockIds, int numBlocks, int numKvHeads)
{
    int inputBlock = blockIdx.x;
    int kvHead = blockIdx.y;
    if (inputBlock >= numBlocks || kvHead >= numKvHeads)
    {
        return;
    }
    extern __shared__ float smem[];
    std::int64_t blockId = blockIds[inputBlock];
    quantizeAndWriteTileParallel<true>(k, records, blockId, inputBlock, kvHead, numKvHeads, smem);
    __syncthreads();
    quantizeAndWriteTileParallel<false>(v, records, blockId, inputBlock, kvHead, numKvHeads, smem);
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

// Factored K score for one logical token: returns sum_d qRot[d]*Krot[d,token].
// For packed tokens the per-token sCol scalar is read once (outside the dim loop)
// instead of once per dim as the elementwise dequantKRot path did.
template <typename T>
__device__ float scoreKLogicalTokenFactored(float const* qRot, T const* sinkK, T const* tailK,
    PackedRecordView records, std::int64_t const* blockIds, int query, int kvHead, int logicalToken, int sinkCount,
    int packedCount, int sinkTokens, int sinkBatch, int tailTokens, int tailBatch, int numKvHeads)
{
    int afterSink = logicalToken - sinkCount;
    if (logicalToken >= sinkCount && afterSink < packedCount)
    {
        int block = afterSink / Layout::kGroupSize;
        int token = afterSink - block * Layout::kGroupSize;
        std::int64_t blockId = blockIds[block];
        float sCol = readPackedFp16(records, blockId, kvHead, kKSColOffset + token * 2);
        float dot = 0.0f;
        for (int d = 0; d < Layout::kHeadDim; ++d)
        {
            int code = readPacked2(records, blockId, kvHead, kKPackedOffset, d * Layout::kGroupSize + token);
            float sRowAbs = readPackedFp16(records, blockId, kvHead, kKSRowAbsOffset + d * 2);
            float zpAbs = readPackedFp16(records, blockId, kvHead, kKZpAbsOffset + d * 2);
            dot += qRot[d] * (static_cast<float>(code) * sRowAbs + zpAbs);
        }
        return dot * sCol;
    }
    // sink/tail tokens: elementwise rotated path.
    float dot = 0.0f;
    for (int d = 0; d < Layout::kHeadDim; ++d)
    {
        dot += qRot[d]
            * loadKRotatedForLogicalToken(sinkK, tailK, records, blockIds, query, kvHead, d, logicalToken, sinkCount,
                packedCount, sinkTokens, sinkBatch, tailTokens, tailBatch, numKvHeads);
    }
    return dot;
}

template <bool IsKey, typename T>
__device__ void dequantReadableTileFromShared(PackedRecordView records, std::int64_t blockId, int kvHead,
    T* readable, int numKvHeads, float* rotTile)
{
    int total = Layout::kGroupSize * Layout::kHeadDim;
    int tid = threadIdx.x;
    for (int linear = tid; linear < total; linear += blockDim.x)
    {
        int token = linear / Layout::kHeadDim;
        int dim = linear - token * Layout::kHeadDim;
        rotTile[linear] = IsKey ? dequantKRot(records, blockId, kvHead, token, dim)
                                : dequantVRot(records, blockId, kvHead, token, dim);
    }
    __syncthreads();

    // Inverse-rotate each token row via batched in-place FWHT instead of the
    // O(N^2) per-element matrix-sum. rotTile is token-major and the output is too,
    // so no transpose is needed: out[token][dim] = FWHT(rotTile[token,:])[dim].
    fwht128SharedBatched(rotTile, Layout::kGroupSize);
    for (int linear = tid; linear < total; linear += blockDim.x)
    {
        int token = linear / Layout::kHeadDim;
        int dim = linear - token * Layout::kHeadDim;
        std::int64_t outIdx = ((blockId * Layout::kGroupSize + token) * numKvHeads + kvHead) * Layout::kHeadDim + dim;
        storeScalar(readable + outIdx, rotTile[linear] * kHadamardScale);
    }
    __syncthreads();
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

    extern __shared__ float rotTile[];
    dequantReadableTileFromShared<true>(records, blockId, kvHead, readableK, numKvHeads, rotTile);
    dequantReadableTileFromShared<false>(records, blockId, kvHead, readableV, numKvHeads, rotTile);
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

// Sparse top-k decode that stages every resident block's K and V code planes plus
// their per-dim scale vectors into shared memory once per CTA, so the scattered
// top-k score/value gather reads 2-bit codes and fp16 scales from smem instead of
// issuing strided/page-addressed global byte loads. Per-token K-sCol / V-sRowAbs /
// V-zpAbs are pre-resolved into smem keyed by the top-k slot. Sink/tail tokens (if
// any) fall back to the elementwise global rotated path. Selected only when the
// staged planes fit the dynamic smem budget.
template <typename T, int THREADS, int MAX_TOPK, int MAX_BLOCKS>
__global__ void kvarnGqaDecodeSparseTopkStagedKernel(T const* q, PackedRecordView records,
    std::int64_t const* blockIds, T const* sinkK, T const* sinkV, T const* tailK, T const* tailV,
    std::int32_t const* seqLens, std::int64_t const* sparseIndices, T* output, int numQueries, int numBlocks,
    int numHeads, int numKvHeads, int seqLensCount, int sinkTokens, int sinkBatch, int tailTokens, int tailBatch,
    int sparseTopk, std::int64_t sparseStrideKv, std::int64_t sparseStrideQuery, std::int64_t sparseStrideTopk)
{
    constexpr int HD = Layout::kHeadDim;
    constexpr int GS = Layout::kGroupSize;
    constexpr int kPlaneBytes = HD * GS / 4;   // 4096
    int query = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;
    if (query >= numQueries || head >= numHeads)
    {
        return;
    }

    extern __shared__ std::uint8_t sparseSmem[];
    // Layout (byte cursor): K planes | V planes | K sRowAbs | K zpAbs | V sCol (all per block).
    std::uint8_t* kPlanes = sparseSmem;
    std::uint8_t* vPlanes = kPlanes + numBlocks * kPlaneBytes;
    float* kSRowAbs = reinterpret_cast<float*>(vPlanes + numBlocks * kPlaneBytes);  // [block][HD]
    float* kZpAbs = kSRowAbs + numBlocks * HD;                                       // [block][HD]
    float* vSCol = kZpAbs + numBlocks * HD;                                          // [block][HD]

    __shared__ float qRot[HD];
    __shared__ float accRot[HD];
    __shared__ float logits[MAX_TOPK];
    __shared__ int logicalTokens[MAX_TOPK];
    __shared__ int tokBlock[MAX_TOPK];     // packed-token block index, or -1 for sink/tail/invalid
    __shared__ int tokInBlk[MAX_TOPK];     // token position within its block
    __shared__ float tokKSCol[MAX_TOPK];   // sCol_K[token] for packed tokens
    __shared__ float tokVSRow[MAX_TOPK];   // sRowAbs_V[token]
    __shared__ float tokVZp[MAX_TOPK];     // zpAbs_V[token]
    __shared__ unsigned char blockTouched[MAX_BLOCKS];  // 1 if any top-k token lands in block
    __shared__ float red[THREADS];

    int groups = numHeads / numKvHeads;
    int kvHead = head / groups;
    int seqLen = seqLensCount == 1 ? seqLens[0] : seqLens[query];
    int cappedSeqLen = seqLen > 0 ? seqLen : 0;
    int sinkCount = sinkTokens > 0 ? (cappedSeqLen < sinkTokens ? cappedSeqLen : sinkTokens) : 0;
    int remainingAfterSink = cappedSeqLen - sinkCount;
    int maxPackedTokens = numBlocks * GS;
    int packedCount = remainingAfterSink < maxPackedTokens ? remainingAfterSink : maxPackedTokens;
    int remainingAfterPacked = remainingAfterSink - packedCount;
    int tailCount = tailTokens > 0 ? (remainingAfterPacked < tailTokens ? remainingAfterPacked : tailTokens) : 0;
    int totalTokens = sinkCount + packedCount + tailCount;
    int numResident = (packedCount + GS - 1) / GS;  // blocks actually holding packed tokens
    T const* qBase = q + (static_cast<std::int64_t>(query) * numHeads + head) * HD;
    T* outBase = output + (static_cast<std::int64_t>(query) * numHeads + head) * HD;

    // Rotate q via in-place FWHT.
    for (int d = tid; d < HD; d += THREADS)
    {
        qRot[d] = loadScalar(qBase + d);
        accRot[d] = 0.0f;
    }
    for (int b = tid; b < numResident; b += THREADS)
    {
        blockTouched[b] = 0;
    }
    __syncthreads();
    // Resolve top-k slots and pre-read their per-token scales.
    for (int i = tid; i < sparseTopk; i += THREADS)
    {
        std::int64_t offset = static_cast<std::int64_t>(kvHead) * sparseStrideKv
            + static_cast<std::int64_t>(query) * sparseStrideQuery + static_cast<std::int64_t>(i) * sparseStrideTopk;
        std::int64_t token = sparseIndices[offset];
        int lt = (token >= 0 && token < totalTokens) ? static_cast<int>(token) : -1;
        logicalTokens[i] = lt;
        logits[i] = -FLT_MAX;
        int b = -1;
        int tib = 0;
        if (lt >= 0)
        {
            int afterSink = lt - sinkCount;
            if (lt >= sinkCount && afterSink < packedCount)
            {
                b = afterSink / GS;
                tib = afterSink - b * GS;
                std::int64_t bid = blockIds[b];
                tokKSCol[i] = readPackedFp16(records, bid, kvHead, kKSColOffset + tib * 2);
                tokVSRow[i] = readPackedFp16(records, bid, kvHead, kVSRowAbsOffset + tib * 2);
                tokVZp[i] = readPackedFp16(records, bid, kvHead, kVZpAbsOffset + tib * 2);
                blockTouched[b] = 1;  // benign race: all writers store the same value
            }
        }
        tokBlock[i] = b;
        tokInBlk[i] = tib;
    }
    __syncthreads();
    // Stage K/V code planes and K(sRowAbs,zpAbs) + V(sCol) scale vectors, but only
    // for blocks that at least one selected top-k token lands in. Clustered top-k
    // indices touch few blocks, so this skips most of the 8 KB/block plane traffic.
    for (int b = 0; b < numResident; ++b)
    {
        if (!blockTouched[b])
        {
            continue;
        }
        std::int64_t bid = blockIds[b];
        stageCodePlane(records, bid, kvHead, kKPackedOffset, kPlanes + b * kPlaneBytes);
        stageCodePlane(records, bid, kvHead, kVPackedOffset, vPlanes + b * kPlaneBytes);
        for (int d = tid; d < HD; d += THREADS)
        {
            kSRowAbs[b * HD + d] = readPackedFp16(records, bid, kvHead, kKSRowAbsOffset + d * 2);
            kZpAbs[b * HD + d] = readPackedFp16(records, bid, kvHead, kKZpAbsOffset + d * 2);
            vSCol[b * HD + d] = readPackedFp16(records, bid, kvHead, kVSColOffset + d * 2);
        }
    }
    fwht128Shared(qRot);
    for (int d = tid; d < HD; d += THREADS)
    {
        qRot[d] *= kHadamardScale;
    }
    __syncthreads();

    if (totalTokens <= 0 || sparseTopk <= 0)
    {
        for (int d = tid; d < HD; d += THREADS)
        {
            storeScalar(outBase + d, 0.0f);
        }
        return;
    }

    // ---- Score pass: K codes + scales from smem for packed tokens. ----
    float localMax = -FLT_MAX;
    for (int i = tid; i < sparseTopk; i += THREADS)
    {
        int lt = logicalTokens[i];
        if (lt < 0)
        {
            continue;
        }
        int b = tokBlock[i];
        float logit;
        if (b >= 0)
        {
            std::uint8_t const* plane = kPlanes + b * kPlaneBytes;
            float const* sRow = kSRowAbs + b * HD;
            float const* zp = kZpAbs + b * HD;
            int tib = tokInBlk[i];
            float dot = 0.0f;
            for (int d = 0; d < HD; ++d)
            {
                int code = readPacked2Smem(plane, d * GS + tib);
                dot += qRot[d] * (static_cast<float>(code) * sRow[d] + zp[d]);
            }
            logit = dot * tokKSCol[i] * kHadamardScale;
        }
        else
        {
            // sink/tail token: elementwise global rotated path.
            float dot = scoreKLogicalTokenFactored(qRot, sinkK, tailK, records, blockIds, query, kvHead, lt,
                sinkCount, packedCount, sinkTokens, sinkBatch, tailTokens, tailBatch, numKvHeads);
            logit = dot * kHadamardScale;
        }
        logits[i] = logit;
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
    for (int i = tid; i < sparseTopk; i += THREADS)
    {
        if (logicalTokens[i] < 0)
        {
            continue;
        }
        float weight = expf(logits[i] - maxLogit);
        logits[i] = weight;
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
    __syncthreads();

    // ---- Value pass: each thread owns rotated dims; V codes + scales from smem. ----
    for (int d = tid; d < HD; d += THREADS)
    {
        float acc = 0.0f;
        for (int i = 0; i < sparseTopk; ++i)
        {
            int lt = logicalTokens[i];
            if (lt < 0)
            {
                continue;
            }
            int b = tokBlock[i];
            if (b >= 0)
            {
                std::uint8_t const* plane = vPlanes + b * kPlaneBytes;
                int tib = tokInBlk[i];
                int code = readPacked2Smem(plane, tib * HD + d);
                acc += logits[i] * (static_cast<float>(code) * tokVSRow[i] + tokVZp[i]) * vSCol[b * HD + d];
            }
            else
            {
                acc += logits[i] * loadVRotatedForLogicalToken(sinkV, tailV, records, blockIds, query, kvHead, d,
                    lt, sinkCount, packedCount, sinkTokens, sinkBatch, tailTokens, tailBatch, numKvHeads);
            }
        }
        accRot[d] = acc;
    }
    __syncthreads();

    // Inverse-rotate the accumulator via in-place FWHT.
    fwht128Shared(accRot);
    for (int j = tid; j < HD; j += THREADS)
    {
        storeScalar(outBase + j, accRot[j] * invDenom * kHadamardScale);
    }
}

template <typename T, int THREADS, int MAX_TOPK>
__global__ void kvarnGqaDecodeSparseTopkKernel(T const* q, PackedRecordView records,
    std::int64_t const* blockIds, T const* sinkK, T const* sinkV, T const* tailK, T const* tailV,
    std::int32_t const* seqLens, std::int64_t const* sparseIndices, T* output, int numQueries, int numBlocks,
    int numHeads, int numKvHeads, int seqLensCount, int sinkTokens, int sinkBatch, int tailTokens, int tailBatch,
    int sparseTopk, std::int64_t sparseStrideKv, std::int64_t sparseStrideQuery, std::int64_t sparseStrideTopk)
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
    __shared__ float logits[MAX_TOPK];
    __shared__ int logicalTokens[MAX_TOPK];
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

    // Rotate q via in-place FWHT (O(N log N)) instead of the O(N^2) matrix-sum.
    for (int d = tid; d < Layout::kHeadDim; d += THREADS)
    {
        qRot[d] = loadScalar(qBase + d);
        accRot[d] = 0.0f;
    }
    for (int i = tid; i < sparseTopk; i += THREADS)
    {
        std::int64_t offset = static_cast<std::int64_t>(kvHead) * sparseStrideKv
            + static_cast<std::int64_t>(query) * sparseStrideQuery + static_cast<std::int64_t>(i) * sparseStrideTopk;
        std::int64_t token = sparseIndices[offset];
        logicalTokens[i] = token >= 0 && token < totalTokens ? static_cast<int>(token) : -1;
        logits[i] = -FLT_MAX;
    }
    fwht128Shared(qRot);
    for (int d = tid; d < Layout::kHeadDim; d += THREADS)
    {
        qRot[d] *= kHadamardScale;
    }
    __syncthreads();

    if (totalTokens <= 0 || sparseTopk <= 0)
    {
        for (int d = tid; d < Layout::kHeadDim; d += THREADS)
        {
            storeScalar(outBase + d, 0.0f);
        }
        return;
    }

    float localMax = -FLT_MAX;
    for (int i = tid; i < sparseTopk; i += THREADS)
    {
        int linearToken = logicalTokens[i];
        if (linearToken < 0)
        {
            continue;
        }
        float dot = scoreKLogicalTokenFactored(qRot, sinkK, tailK, records, blockIds, query, kvHead, linearToken,
            sinkCount, packedCount, sinkTokens, sinkBatch, tailTokens, tailBatch, numKvHeads);
        float logit = dot * kHadamardScale;
        logits[i] = logit;
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
    for (int i = tid; i < sparseTopk; i += THREADS)
    {
        if (logicalTokens[i] < 0)
        {
            continue;
        }
        float weight = expf(logits[i] - maxLogit);
        logits[i] = weight;
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
        for (int i = 0; i < sparseTopk; ++i)
        {
            int linearToken = logicalTokens[i];
            if (linearToken < 0)
            {
                continue;
            }
            acc += logits[i] * loadVRotatedForLogicalToken(sinkV, tailV, records, blockIds, query, kvHead, d,
                linearToken, sinkCount, packedCount, sinkTokens, sinkBatch, tailTokens, tailBatch, numKvHeads);
        }
        accRot[d] = acc;
    }
    __syncthreads();

    // Inverse-rotate the accumulator via in-place FWHT.
    fwht128Shared(accRot);
    for (int j = tid; j < Layout::kHeadDim; j += THREADS)
    {
        storeScalar(outBase + j, accRot[j] * invDenom * kHadamardScale);
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

// Dynamic-shared-memory dense decode for arbitrary token counts. Each block owns
// one (query, head). The per-token logit/weight is cached in dynamic shared
// memory, so the value-accumulation pass reuses it instead of re-dequantizing K,
// and the value accumulation is parallelized over head dims (no atomics).
template <typename T, int THREADS>
__global__ void kvarnGqaDecodeDynKernel(T const* q, PackedRecordView records, std::int64_t const* blockIds,
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

    extern __shared__ float dynSmem[];
    float* weights = dynSmem;                  // [totalTokens] cached logits then weights
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

    // Pass 1: dequant K once per (token, dim), score, cache the logit, track block max.
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
        weights[linearToken] = logit;
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
    __syncthreads();

    // Convert cached logits to weights and reduce the denominator (reuse cached
    // logits; no second K dequant).
    float localDenom = 0.0f;
    for (int linearToken = tid; linearToken < totalTokens; linearToken += THREADS)
    {
        float weight = expf(weights[linearToken] - maxLogit);
        weights[linearToken] = weight;
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

    // Pass 2: value accumulation parallel over head dims; each thread owns a set of
    // rotated dims and dequants V once per (token, dim). No atomics.
    for (int d = tid; d < Layout::kHeadDim; d += THREADS)
    {
        float acc = 0.0f;
        for (int linearToken = 0; linearToken < totalTokens; ++linearToken)
        {
            acc += weights[linearToken] * loadVRotatedForLogicalToken(sinkV, tailV, records, blockIds, query, kvHead, d,
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

// Factored pure-packed dense decode (no sink/tail). Algebraically identical to the
// generic dyn kernel but hoists the per-dim / per-token fp16 scale unpacking out of
// the hot per-element loops:
//   logit(t)  = sCol_K[t] * ( sum_d (qRot[d]*sRowAbs_K[d]) * code_K(d,t) + sum_d qRot[d]*zpAbs_K[d] )
//   accRot(d) = sCol_V[d] * ( sum_t (w[t]*sRowAbs_V[t]) * code_V(t,d) + sum_t w[t]*zpAbs_V[t] )
// so each packed sub-block reads its fp16 scales once and the inner loops touch only
// 2-bit codes and precomputed shared-memory floats.
template <typename T, int THREADS>
__global__ void kvarnGqaDecodePackedFactoredKernel(T const* q, PackedRecordView records,
    std::int64_t const* blockIds, std::int32_t const* seqLens, T* output, int numQueries, int numBlocks, int numHeads,
    int numKvHeads, int seqLensCount)
{
    int query = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;
    if (query >= numQueries || head >= numHeads)
    {
        return;
    }

    constexpr int HD = Layout::kHeadDim;
    constexpr int GS = Layout::kGroupSize;
    extern __shared__ float dynSmem[];
    float* weights = dynSmem;                  // [totalTokens]
    std::uint8_t* codeSmem = reinterpret_cast<std::uint8_t*>(weights + ((numBlocks * GS + 3) & ~3));
    __shared__ float qRot[HD];
    __shared__ float accRot[HD];
    __shared__ float qScaled[HD];              // qRot[d]*sRowAbs_K[block][d]
    __shared__ float sColCache[GS];            // per-block K sCol[token] or V sCol[dim]
    __shared__ float wScaled[GS];              // weight[t]*sRowAbs_V[block][t]
    __shared__ float red[THREADS];

    int groups = numHeads / numKvHeads;
    int kvHead = head / groups;
    int seqLen = seqLensCount == 1 ? seqLens[0] : seqLens[query];
    int cappedSeqLen = seqLen > 0 ? seqLen : 0;
    int maxPackedTokens = numBlocks * GS;
    int totalTokens = cappedSeqLen < maxPackedTokens ? cappedSeqLen : maxPackedTokens;
    int numFullBlocks = totalTokens / GS;
    int tailInBlock = totalTokens - numFullBlocks * GS;  // partial trailing packed block
    T const* qBase = q + (static_cast<std::int64_t>(query) * numHeads + head) * HD;
    T* outBase = output + (static_cast<std::int64_t>(query) * numHeads + head) * HD;

    // Rotate q via in-place FWHT (O(N log N)) instead of the O(N^2) Hadamard
    // matrix-sum; result is bit-identical to H @ q up to fp summation order.
    for (int d = tid; d < HD; d += THREADS)
    {
        qRot[d] = loadScalar(qBase + d);
        accRot[d] = 0.0f;
    }
    fwht128Shared(qRot);
    for (int d = tid; d < HD; d += THREADS)
    {
        qRot[d] *= kHadamardScale;
    }
    __syncthreads();

    if (totalTokens <= 0)
    {
        for (int d = tid; d < HD; d += THREADS)
        {
            storeScalar(outBase + d, 0.0f);
        }
        return;
    }

    // ---- Score pass: per packed sub-block, factor K scales, then score tokens. ----
    float localMax = -FLT_MAX;
    int scoreBlocks = numFullBlocks + (tailInBlock > 0 ? 1 : 0);
    for (int b = 0; b < scoreBlocks; ++b)
    {
        std::int64_t blockId = blockIds[b];
        int tokensInBlock = (b < numFullBlocks) ? GS : tailInBlock;
        // Precompute qScaled[d] = qRot[d]*sRowAbs_K[d] and per-thread partial of
        // qZpConst = sum_d qRot[d]*zpAbs_K[d]; also cache sCol_K[token].
        float partialZp = 0.0f;
        for (int d = tid; d < HD; d += THREADS)
        {
            float sRowAbs = readPackedFp16(records, blockId, kvHead, kKSRowAbsOffset + d * 2);
            float zpAbs = readPackedFp16(records, blockId, kvHead, kKZpAbsOffset + d * 2);
            qScaled[d] = qRot[d] * sRowAbs;
            partialZp += qRot[d] * zpAbs;
        }
        for (int t = tid; t < tokensInBlock; t += THREADS)
        {
            sColCache[t] = readPackedFp16(records, blockId, kvHead, kKSColOffset + t * 2);
        }
        red[tid] = partialZp;
        __syncthreads();
        for (int stride = THREADS / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride)
            {
                red[tid] += red[tid + stride];
            }
            __syncthreads();
        }
        float qZpConst = red[0];
        // Stage K code plane to smem (overlaps with the zp reduction sync above).
        stageCodePlane(records, blockId, kvHead, kKPackedOffset, codeSmem);
        __syncthreads();

        for (int t = tid; t < tokensInBlock; t += THREADS)
        {
            float dot = 0.0f;
            for (int d = 0; d < HD; ++d)
            {
                int code = readPacked2Smem(codeSmem, d * GS + t);
                dot += qScaled[d] * static_cast<float>(code);
            }
            float logit = (dot + qZpConst) * sColCache[t] * kHadamardScale;
            int globalT = b * GS + t;
            weights[globalT] = logit;
            localMax = fmaxf(localMax, logit);
        }
        __syncthreads();
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
    __syncthreads();

    float localDenom = 0.0f;
    for (int t = tid; t < totalTokens; t += THREADS)
    {
        float w = expf(weights[t] - maxLogit);
        weights[t] = w;
        localDenom += w;
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
    __syncthreads();

    // ---- Value pass: per packed sub-block, factor V scales, accumulate over dims. ----
    for (int b = 0; b < scoreBlocks; ++b)
    {
        std::int64_t blockId = blockIds[b];
        int tokensInBlock = (b < numFullBlocks) ? GS : tailInBlock;
        // Precompute wScaled[t] = weight[t]*sRowAbs_V[t]; per-thread partial of
        // wZpConst = sum_t weight[t]*zpAbs_V[t]; cache sCol_V[dim].
        float partialZp = 0.0f;
        for (int t = tid; t < tokensInBlock; t += THREADS)
        {
            int globalT = b * GS + t;
            float w = weights[globalT];
            float sRowAbs = readPackedFp16(records, blockId, kvHead, kVSRowAbsOffset + t * 2);
            float zpAbs = readPackedFp16(records, blockId, kvHead, kVZpAbsOffset + t * 2);
            wScaled[t] = w * sRowAbs;
            partialZp += w * zpAbs;
        }
        for (int d = tid; d < HD; d += THREADS)
        {
            sColCache[d] = readPackedFp16(records, blockId, kvHead, kVSColOffset + d * 2);
        }
        red[tid] = partialZp;
        __syncthreads();
        for (int stride = THREADS / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride)
            {
                red[tid] += red[tid + stride];
            }
            __syncthreads();
        }
        float wZpConst = red[0];
        // Stage V code plane to smem (overlaps with the zp reduction sync above).
        stageCodePlane(records, blockId, kvHead, kVPackedOffset, codeSmem);
        __syncthreads();

        for (int d = tid; d < HD; d += THREADS)
        {
            float acc = 0.0f;
            for (int t = 0; t < tokensInBlock; ++t)
            {
                int code = readPacked2Smem(codeSmem, t * HD + d);
                acc += wScaled[t] * static_cast<float>(code);
            }
            accRot[d] += (acc + wZpConst) * sColCache[d];
        }
        __syncthreads();
    }

    // Inverse-rotate the accumulator via in-place FWHT (H is symmetric and its own
    // inverse up to the 1/N normalization carried by kHadamardScale on both q and
    // out). out[j] = invDenom * kHadamardScale * (H @ accRot)[j].
    fwht128Shared(accRot);
    for (int j = tid; j < HD; j += THREADS)
    {
        storeScalar(outBase + j, accRot[j] * invDenom * kHadamardScale);
    }
}

// Flash-decoding split-K phase 1 for the pure-packed path. grid =
// (numQueries, numHeads, numSplits); each CTA owns a contiguous slice of packed
// blocks and emits a partial softmax state (running max m, denom l, and the
// rotated weighted-V accumulator accRot[HD], pre-inverse-FWHT and pre-division)
// to `partials`. The combine kernel merges the splits. This exists purely to
// raise CTA occupancy for low query counts, where the single-CTA-per-(query,head)
// factored kernel leaves almost all SMs idle. partials layout per (query, head,
// split): [m, l, accRot[0..HD-1]] = (HD + 2) floats.
template <typename T, int THREADS>
__global__ void kvarnGqaDecodeSplitKernel(T const* q, PackedRecordView records, std::int64_t const* blockIds,
    std::int32_t const* seqLens, float* partials, int numQueries, int numBlocks, int numHeads, int numKvHeads,
    int seqLensCount, int numSplits, int blocksPerSplit)
{
    constexpr int HD = Layout::kHeadDim;
    constexpr int GS = Layout::kGroupSize;
    int query = blockIdx.x;
    int head = blockIdx.y;
    int splitIdx = blockIdx.z;
    int tid = threadIdx.x;
    if (query >= numQueries || head >= numHeads || splitIdx >= numSplits)
    {
        return;
    }

    extern __shared__ float dynSmem[];
    float* weights = dynSmem;                  // [blocksPerSplit * GS]
    std::uint8_t* codeSmem = reinterpret_cast<std::uint8_t*>(weights + ((blocksPerSplit * GS + 3) & ~3));
    __shared__ float qRot[HD];
    __shared__ float accRot[HD];
    __shared__ float qScaled[HD];
    __shared__ float sColCache[GS];
    __shared__ float wScaled[GS];
    __shared__ float red[THREADS];

    int groups = numHeads / numKvHeads;
    int kvHead = head / groups;
    int seqLen = seqLensCount == 1 ? seqLens[0] : seqLens[query];
    int cappedSeqLen = seqLen > 0 ? seqLen : 0;
    int maxPackedTokens = numBlocks * GS;
    int totalTokens = cappedSeqLen < maxPackedTokens ? cappedSeqLen : maxPackedTokens;
    int numScoreBlocks = (totalTokens + GS - 1) / GS;     // blocks with >=1 token
    int blockBegin = splitIdx * blocksPerSplit;
    int blockEnd = blockBegin + blocksPerSplit;
    if (blockEnd > numScoreBlocks)
    {
        blockEnd = numScoreBlocks;
    }
    float* part = partials + (static_cast<std::int64_t>(query) * numHeads + head) * numSplits * (HD + 2)
        + static_cast<std::int64_t>(splitIdx) * (HD + 2);

    for (int d = tid; d < HD; d += THREADS)
    {
        qRot[d] = loadScalar(q + (static_cast<std::int64_t>(query) * numHeads + head) * HD + d);
        accRot[d] = 0.0f;
    }
    fwht128Shared(qRot);
    for (int d = tid; d < HD; d += THREADS)
    {
        qRot[d] *= kHadamardScale;
    }
    __syncthreads();

    // Empty split (no blocks assigned): emit neutral partial (m=-inf, l=0, acc=0).
    if (blockBegin >= blockEnd || totalTokens <= 0)
    {
        for (int d = tid; d < HD; d += THREADS)
        {
            part[2 + d] = 0.0f;
        }
        if (tid == 0)
        {
            part[0] = -FLT_MAX;
            part[1] = 0.0f;
        }
        return;
    }

    // ---- Score pass over the split's blocks. weights indexed split-local. ----
    float localMax = -FLT_MAX;
    for (int b = blockBegin; b < blockEnd; ++b)
    {
        std::int64_t blockId = blockIds[b];
        int blockTokens = totalTokens - b * GS;
        int tokensInBlock = blockTokens < GS ? blockTokens : GS;
        float partialZp = 0.0f;
        for (int d = tid; d < HD; d += THREADS)
        {
            float sRowAbs = readPackedFp16(records, blockId, kvHead, kKSRowAbsOffset + d * 2);
            float zpAbs = readPackedFp16(records, blockId, kvHead, kKZpAbsOffset + d * 2);
            qScaled[d] = qRot[d] * sRowAbs;
            partialZp += qRot[d] * zpAbs;
        }
        for (int t = tid; t < tokensInBlock; t += THREADS)
        {
            sColCache[t] = readPackedFp16(records, blockId, kvHead, kKSColOffset + t * 2);
        }
        red[tid] = partialZp;
        __syncthreads();
        for (int stride = THREADS / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride)
            {
                red[tid] += red[tid + stride];
            }
            __syncthreads();
        }
        float qZpConst = red[0];
        stageCodePlane(records, blockId, kvHead, kKPackedOffset, codeSmem);
        __syncthreads();

        for (int t = tid; t < tokensInBlock; t += THREADS)
        {
            float dot = 0.0f;
            for (int d = 0; d < HD; ++d)
            {
                int code = readPacked2Smem(codeSmem, d * GS + t);
                dot += qScaled[d] * static_cast<float>(code);
            }
            float logit = (dot + qZpConst) * sColCache[t] * kHadamardScale;
            weights[(b - blockBegin) * GS + t] = logit;
            localMax = fmaxf(localMax, logit);
        }
        __syncthreads();
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
    __syncthreads();

    int splitTokens = (blockEnd - blockBegin - 1) * GS;
    {
        int lastBlockTokens = totalTokens - (blockEnd - 1) * GS;
        splitTokens += lastBlockTokens < GS ? lastBlockTokens : GS;
    }
    float localDenom = 0.0f;
    for (int t = tid; t < splitTokens; t += THREADS)
    {
        float w = expf(weights[t] - maxLogit);
        weights[t] = w;
        localDenom += w;
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
    float splitDenom = red[0];
    __syncthreads();

    // ---- Value pass over the split's blocks (no division; combine does it). ----
    for (int b = blockBegin; b < blockEnd; ++b)
    {
        std::int64_t blockId = blockIds[b];
        int blockTokens = totalTokens - b * GS;
        int tokensInBlock = blockTokens < GS ? blockTokens : GS;
        float partialZp = 0.0f;
        for (int t = tid; t < tokensInBlock; t += THREADS)
        {
            float w = weights[(b - blockBegin) * GS + t];
            float sRowAbs = readPackedFp16(records, blockId, kvHead, kVSRowAbsOffset + t * 2);
            float zpAbs = readPackedFp16(records, blockId, kvHead, kVZpAbsOffset + t * 2);
            wScaled[t] = w * sRowAbs;
            partialZp += w * zpAbs;
        }
        for (int d = tid; d < HD; d += THREADS)
        {
            sColCache[d] = readPackedFp16(records, blockId, kvHead, kVSColOffset + d * 2);
        }
        red[tid] = partialZp;
        __syncthreads();
        for (int stride = THREADS / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride)
            {
                red[tid] += red[tid + stride];
            }
            __syncthreads();
        }
        float wZpConst = red[0];
        stageCodePlane(records, blockId, kvHead, kVPackedOffset, codeSmem);
        __syncthreads();

        for (int d = tid; d < HD; d += THREADS)
        {
            float acc = 0.0f;
            for (int t = 0; t < tokensInBlock; ++t)
            {
                int code = readPacked2Smem(codeSmem, t * HD + d);
                acc += wScaled[t] * static_cast<float>(code);
            }
            accRot[d] += (acc + wZpConst) * sColCache[d];
        }
        __syncthreads();
    }

    // Emit the split's partial state (rotated accumulator, pre-inverse-FWHT).
    for (int d = tid; d < HD; d += THREADS)
    {
        part[2 + d] = accRot[d];
    }
    if (tid == 0)
    {
        part[0] = maxLogit;
        part[1] = splitDenom;
    }
}

// Flash-decoding combine (phase 2). grid = (numQueries, numHeads). Merges the
// numSplits partial softmax states with the standard online rescale, applies the
// final inverse FWHT, and writes the output.
template <typename T, int THREADS>
__global__ void kvarnGqaDecodeCombineKernel(
    float const* partials, T* output, int numQueries, int numHeads, int numSplits)
{
    constexpr int HD = Layout::kHeadDim;
    int query = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;
    if (query >= numQueries || head >= numHeads)
    {
        return;
    }

    __shared__ float accRot[HD];
    __shared__ float red[THREADS];
    __shared__ float globalMax;
    __shared__ float globalDenom;
    float const* base
        = partials + (static_cast<std::int64_t>(query) * numHeads + head) * numSplits * (HD + 2);

    // Global max over splits.
    float m = -FLT_MAX;
    for (int s = tid; s < numSplits; s += THREADS)
    {
        m = fmaxf(m, base[s * (HD + 2)]);
    }
    red[tid] = m;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            red[tid] = fmaxf(red[tid], red[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        globalMax = red[0];
    }
    for (int d = tid; d < HD; d += THREADS)
    {
        accRot[d] = 0.0f;
    }
    __syncthreads();

    // Rescaled denom and accumulator merge.
    float ld = 0.0f;
    for (int s = 0; s < numSplits; ++s)
    {
        float ms = base[s * (HD + 2)];
        float ls = base[s * (HD + 2) + 1];
        if (ls <= 0.0f)
        {
            continue;
        }
        float scale = expf(ms - globalMax);
        for (int d = tid; d < HD; d += THREADS)
        {
            accRot[d] += scale * base[s * (HD + 2) + 2 + d];
        }
        if (tid == 0)
        {
            ld += scale * ls;
        }
    }
    if (tid == 0)
    {
        globalDenom = ld;
    }
    __syncthreads();

    float invDenom = globalDenom > 0.0f ? 1.0f / globalDenom : 0.0f;
    fwht128Shared(accRot);
    T* outBase = output + (static_cast<std::int64_t>(query) * numHeads + head) * HD;
    for (int j = tid; j < HD; j += THREADS)
    {
        storeScalar(outBase + j, accRot[j] * invDenom * kHadamardScale);
    }
}

} // namespace

bool kvarnGqaBackendReady()
{
    // The store/decode kernels are block-parallel packed KVarN paths, while BDR helpers remain guarded.
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
    constexpr int kThreads = 256;
    constexpr std::size_t kSharedFloats = Layout::kGroupSize * Layout::kHeadDim + 8 * Layout::kHeadDim + 1;
    constexpr std::size_t kSharedBytes = kSharedFloats * sizeof(float);
    if (useBf16)
    {
        checkKvarnGqaCuda(cudaFuncSetAttribute(kvarnGqaStoreParallelKernel<__nv_bfloat16>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSharedBytes)),
            "kvarn_gqa_store bf16 dynamic smem attribute");
        kvarnGqaStoreParallelKernel<__nv_bfloat16><<<grid, kThreads, kSharedBytes, stream>>>(
            static_cast<__nv_bfloat16 const*>(k), static_cast<__nv_bfloat16 const*>(v), view, blockIds, numBlocks,
            numKvHeads);
    }
    else
    {
        checkKvarnGqaCuda(cudaFuncSetAttribute(kvarnGqaStoreParallelKernel<__half>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSharedBytes)),
            "kvarn_gqa_store fp16 dynamic smem attribute");
        kvarnGqaStoreParallelKernel<__half><<<grid, kThreads, kSharedBytes, stream>>>(static_cast<__half const*>(k),
            static_cast<__half const*>(v), view, blockIds, numBlocks, numKvHeads);
    }
    checkKvarnGqaLaunch("kvarn_gqa kernel launch");
}

void invokeKvarnGqaDecodeSparseK2V2G128(void const* q, std::uint8_t const* packedRecords,
    std::int64_t const* blockIds, void const* sinkK, void const* sinkV, void const* tailK, void const* tailV,
    std::int32_t const* seqLens, std::int64_t const* sparseIndices, void* output, int numQueries, int numBlocks,
    int numHeads, int numKvHeads, int headDim, int groupSize, bool useBf16, int seqLensCount, int sinkTokens,
    int sinkBatch, int tailTokens, int tailBatch, int sparseTopk, std::int64_t sparseStrideKv,
    std::int64_t sparseStrideQuery, std::int64_t sparseStrideTopk, bool pageLayout, std::int64_t strideBlock,
    std::int64_t strideToken, std::int64_t strideHead, std::int64_t strideByte, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(headDim == Layout::kHeadDim && groupSize == Layout::kGroupSize,
        "kvarn_gqa_decode_sparse currently supports only k2v2_g128");
    TLLM_CHECK_WITH_INFO(numKvHeads > 0 && numHeads % numKvHeads == 0,
        "kvarn_gqa_decode_sparse requires num_heads divisible by num_kv_heads");
    TLLM_CHECK_WITH_INFO(numBlocks >= 0 && numQueries >= 0 && sparseTopk >= 0,
        "kvarn_gqa_decode_sparse got invalid sizes");
    TLLM_CHECK_WITH_INFO(sparseTopk <= 256,
        "kvarn_gqa_decode_sparse currently supports sparse top-k <= 256");
    PackedRecordView view{packedRecords, pageLayout, strideBlock, strideToken, strideHead, strideByte};
    dim3 grid(numQueries, numHeads);
    constexpr int kThreads = 256;
    // Staged variant pulls all resident blocks' code planes + scale vectors into
    // smem once. Per-block footprint: 2 code planes (2*4096) + K(sRowAbs,zpAbs) + V
    // sCol (3*128 floats) = 9728 bytes. Opt in when it fits the SM100 cap.
    constexpr int kMaxStageBlocks = 16;
    constexpr std::size_t kPerBlockStageBytes = 2u * 4096u + 3u * Layout::kHeadDim * sizeof(float);
    std::size_t stageSmemBytes = static_cast<std::size_t>(numBlocks) * kPerBlockStageBytes;
    constexpr std::size_t kMaxStageSmem = 200u * 1024u;
    bool useStaged = numBlocks > 0 && numBlocks <= kMaxStageBlocks && stageSmemBytes <= kMaxStageSmem;
    if (useBf16)
    {
        if (useStaged)
        {
            checkKvarnGqaCuda(
                cudaFuncSetAttribute(kvarnGqaDecodeSparseTopkStagedKernel<__nv_bfloat16, kThreads, 256, kMaxStageBlocks>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(stageSmemBytes)),
                "kvarn_gqa_decode_sparse bf16 staged smem attribute");
            kvarnGqaDecodeSparseTopkStagedKernel<__nv_bfloat16, kThreads, 256, kMaxStageBlocks>
                <<<grid, kThreads, stageSmemBytes, stream>>>(static_cast<__nv_bfloat16 const*>(q), view, blockIds,
                    static_cast<__nv_bfloat16 const*>(sinkK), static_cast<__nv_bfloat16 const*>(sinkV),
                    static_cast<__nv_bfloat16 const*>(tailK), static_cast<__nv_bfloat16 const*>(tailV), seqLens,
                    sparseIndices, static_cast<__nv_bfloat16*>(output), numQueries, numBlocks, numHeads, numKvHeads,
                    seqLensCount, sinkTokens, sinkBatch, tailTokens, tailBatch, sparseTopk, sparseStrideKv,
                    sparseStrideQuery, sparseStrideTopk);
        }
        else
        {
            kvarnGqaDecodeSparseTopkKernel<__nv_bfloat16, 256, 256><<<grid, kThreads, 0, stream>>>(
                static_cast<__nv_bfloat16 const*>(q), view, blockIds, static_cast<__nv_bfloat16 const*>(sinkK),
                static_cast<__nv_bfloat16 const*>(sinkV), static_cast<__nv_bfloat16 const*>(tailK),
                static_cast<__nv_bfloat16 const*>(tailV), seqLens, sparseIndices, static_cast<__nv_bfloat16*>(output),
                numQueries, numBlocks, numHeads, numKvHeads, seqLensCount, sinkTokens, sinkBatch, tailTokens, tailBatch,
                sparseTopk, sparseStrideKv, sparseStrideQuery, sparseStrideTopk);
        }
    }
    else
    {
        if (useStaged)
        {
            checkKvarnGqaCuda(
                cudaFuncSetAttribute(kvarnGqaDecodeSparseTopkStagedKernel<__half, kThreads, 256, kMaxStageBlocks>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(stageSmemBytes)),
                "kvarn_gqa_decode_sparse fp16 staged smem attribute");
            kvarnGqaDecodeSparseTopkStagedKernel<__half, kThreads, 256, kMaxStageBlocks>
                <<<grid, kThreads, stageSmemBytes, stream>>>(static_cast<__half const*>(q), view, blockIds,
                    static_cast<__half const*>(sinkK), static_cast<__half const*>(sinkV),
                    static_cast<__half const*>(tailK), static_cast<__half const*>(tailV), seqLens, sparseIndices,
                    static_cast<__half*>(output), numQueries, numBlocks, numHeads, numKvHeads, seqLensCount, sinkTokens,
                    sinkBatch, tailTokens, tailBatch, sparseTopk, sparseStrideKv, sparseStrideQuery, sparseStrideTopk);
        }
        else
        {
            kvarnGqaDecodeSparseTopkKernel<__half, 256, 256><<<grid, kThreads, 0, stream>>>(static_cast<__half const*>(q),
                view, blockIds, static_cast<__half const*>(sinkK), static_cast<__half const*>(sinkV),
                static_cast<__half const*>(tailK), static_cast<__half const*>(tailV), seqLens, sparseIndices,
                static_cast<__half*>(output), numQueries, numBlocks, numHeads, numKvHeads, seqLensCount, sinkTokens,
                sinkBatch, tailTokens, tailBatch, sparseTopk, sparseStrideKv, sparseStrideQuery, sparseStrideTopk);
        }
    }
    checkKvarnGqaLaunch("kvarn_gqa kernel launch");
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
    constexpr std::size_t kSharedBytes = Layout::kGroupSize * Layout::kHeadDim * sizeof(float);
    if (useBf16)
    {
        checkKvarnGqaCuda(cudaFuncSetAttribute(kvarnGqaDequantAmortizedKernel<__nv_bfloat16>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSharedBytes)),
            "kvarn_gqa_dequant_amortized bf16 dynamic smem attribute");
        kvarnGqaDequantAmortizedKernel<<<grid, kThreads, kSharedBytes, stream>>>(view, blockIds,
            static_cast<__nv_bfloat16*>(readableK), static_cast<__nv_bfloat16*>(readableV),
            numChurnBlocks, numPhysicalBlocks, numKvHeads);
    }
    else
    {
        checkKvarnGqaCuda(cudaFuncSetAttribute(kvarnGqaDequantAmortizedKernel<__half>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSharedBytes)),
            "kvarn_gqa_dequant_amortized fp16 dynamic smem attribute");
        kvarnGqaDequantAmortizedKernel<<<grid, kThreads, kSharedBytes, stream>>>(view, blockIds,
            static_cast<__half*>(readableK), static_cast<__half*>(readableV), numChurnBlocks, numPhysicalBlocks,
            numKvHeads);
    }
    checkKvarnGqaLaunch("kvarn_gqa kernel launch");
}

namespace
{
// Lazily-grown persistent device workspace for flash-decoding split-K partials.
// A stable allocation (same address across a CUDA-graph capture/replay) keeps
// graph replay bit-identical; it only ever grows, never shrinks or frees mid-run.
float* acquireSplitWorkspace(std::size_t floatsNeeded, cudaStream_t stream)
{
    static float* ptr = nullptr;
    static std::size_t capacity = 0;
    if (floatsNeeded > capacity)
    {
        if (ptr != nullptr)
        {
            checkKvarnGqaCuda(cudaStreamSynchronize(stream), "kvarn_gqa split workspace sync");
            checkKvarnGqaCuda(cudaFree(ptr), "kvarn_gqa split workspace free");
        }
        checkKvarnGqaCuda(cudaMalloc(&ptr, floatsNeeded * sizeof(float)), "kvarn_gqa split workspace alloc");
        capacity = floatsNeeded;
    }
    return ptr;
}

int splitDecodeSmCount()
{
    static int sm = 0;
    if (sm == 0)
    {
        int dev = 0;
        checkKvarnGqaCuda(cudaGetDevice(&dev), "kvarn_gqa get device");
        checkKvarnGqaCuda(
            cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev), "kvarn_gqa sm count");
    }
    return sm;
}
} // namespace

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
    constexpr int kThreads = 256;
    int maxTotalTokens = sinkTokens + numBlocks * Layout::kGroupSize + tailTokens;
    bool useSmallDecode = maxTotalTokens <= 256;
    // Dynamic-shared-memory decode caches per-token weights so the value pass does
    // not re-dequant K and accumulates without atomics. Opt in up to the SM100 cap.
    std::size_t dynSmemBytes = static_cast<std::size_t>(maxTotalTokens) * sizeof(float);
    constexpr std::size_t kMaxDynSmem = 200u * 1024u;
    bool useDynDecode = !useSmallDecode && dynSmemBytes <= kMaxDynSmem;
    // Pure-packed reads (no fp16 sink/tail side state) use the factored kernel that
    // hoists scale unpacking out of the hot loops and stages one 4096-byte code plane
    // in smem (weights region is padded to a 4-float boundary). It handles any token
    // count incl. partial trailing blocks, so it also serves the <=256-token case and
    // takes priority over the (slower, unfactored) small kernel.
    constexpr std::size_t kCodePlaneBytes = 4096;
    std::size_t factoredSmemBytes
        = static_cast<std::size_t>((maxTotalTokens + 3) & ~3) * sizeof(float) + kCodePlaneBytes;
    bool usePackedFactored
        = sinkTokens == 0 && tailTokens == 0 && maxTotalTokens > 0 && factoredSmemBytes <= kMaxDynSmem;
    // Flash-decoding split-K: when the factored grid (numQueries*numHeads CTAs)
    // leaves the SMs underutilized, split the packed-block range across multiple
    // CTAs per (query, head) and merge partial softmax states. Pure-packed only.
    int gridCTAs = numQueries * numHeads;
    int numSplits = 1;
    int blocksPerSplit = numBlocks;
    bool useSplit = false;
    if (usePackedFactored && numBlocks >= 2)
    {
        int targetCTAs = 2 * splitDecodeSmCount();
        int want = (targetCTAs + gridCTAs - 1) / gridCTAs;   // splits to ~fill SMs
        if (want < 1)
        {
            want = 1;
        }
        if (want > numBlocks)
        {
            want = numBlocks;
        }
        if (want >= 2)
        {
            blocksPerSplit = (numBlocks + want - 1) / want;
            numSplits = (numBlocks + blocksPerSplit - 1) / blocksPerSplit;
            // Split phase-1 smem only needs this split's blocks of weights + 1 code plane.
            std::size_t splitSmemBytes
                = static_cast<std::size_t>((blocksPerSplit * Layout::kGroupSize + 3) & ~3) * sizeof(float)
                + kCodePlaneBytes;
            useSplit = numSplits >= 2 && splitSmemBytes <= kMaxDynSmem;
            if (useSplit)
            {
                factoredSmemBytes = splitSmemBytes;  // reuse var for the split attr set below
            }
        }
    }
    // Small kernel now only handles the <=256 case that still carries fp16 sink/tail state.
    useSmallDecode = useSmallDecode && !usePackedFactored;
    if (useSplit)
    {
        std::size_t wsFloats = static_cast<std::size_t>(numQueries) * numHeads * numSplits
            * (Layout::kHeadDim + 2);
        float* partials = acquireSplitWorkspace(wsFloats, stream);
        dim3 splitGrid(numQueries, numHeads, numSplits);
        dim3 combineGrid(numQueries, numHeads);
        constexpr int kCombineThreads = 256;
        if (useBf16)
        {
            checkKvarnGqaCuda(cudaFuncSetAttribute(kvarnGqaDecodeSplitKernel<__nv_bfloat16, kThreads>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(factoredSmemBytes)),
                "kvarn_gqa_decode bf16 split smem attribute");
            kvarnGqaDecodeSplitKernel<__nv_bfloat16, kThreads><<<splitGrid, kThreads, factoredSmemBytes, stream>>>(
                static_cast<__nv_bfloat16 const*>(q), view, blockIds, seqLens, partials, numQueries, numBlocks,
                numHeads, numKvHeads, seqLensCount, numSplits, blocksPerSplit);
            kvarnGqaDecodeCombineKernel<__nv_bfloat16, kCombineThreads><<<combineGrid, kCombineThreads, 0, stream>>>(
                partials, static_cast<__nv_bfloat16*>(output), numQueries, numHeads, numSplits);
        }
        else
        {
            checkKvarnGqaCuda(cudaFuncSetAttribute(kvarnGqaDecodeSplitKernel<__half, kThreads>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(factoredSmemBytes)),
                "kvarn_gqa_decode fp16 split smem attribute");
            kvarnGqaDecodeSplitKernel<__half, kThreads><<<splitGrid, kThreads, factoredSmemBytes, stream>>>(
                static_cast<__half const*>(q), view, blockIds, seqLens, partials, numQueries, numBlocks, numHeads,
                numKvHeads, seqLensCount, numSplits, blocksPerSplit);
            kvarnGqaDecodeCombineKernel<__half, kCombineThreads><<<combineGrid, kCombineThreads, 0, stream>>>(
                partials, static_cast<__half*>(output), numQueries, numHeads, numSplits);
        }
        checkKvarnGqaLaunch("kvarn_gqa kernel launch");
        return;
    }
    if (useBf16)
    {
        if (useSmallDecode)
        {
            kvarnGqaDecodeSmallKernel<__nv_bfloat16, 256, 256><<<grid, kThreads, 0, stream>>>(
                static_cast<__nv_bfloat16 const*>(q), view, blockIds, static_cast<__nv_bfloat16 const*>(sinkK),
                static_cast<__nv_bfloat16 const*>(sinkV), static_cast<__nv_bfloat16 const*>(tailK),
                static_cast<__nv_bfloat16 const*>(tailV), seqLens, static_cast<__nv_bfloat16*>(output), numQueries,
                numBlocks, numHeads, numKvHeads, seqLensCount, sinkTokens, sinkBatch, tailTokens, tailBatch);
        }
        else if (usePackedFactored)
        {
            checkKvarnGqaCuda(cudaFuncSetAttribute(kvarnGqaDecodePackedFactoredKernel<__nv_bfloat16, kThreads>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(factoredSmemBytes)),
                "kvarn_gqa_decode bf16 factored smem attribute");
            kvarnGqaDecodePackedFactoredKernel<__nv_bfloat16, kThreads><<<grid, kThreads, factoredSmemBytes, stream>>>(
                static_cast<__nv_bfloat16 const*>(q), view, blockIds, seqLens, static_cast<__nv_bfloat16*>(output),
                numQueries, numBlocks, numHeads, numKvHeads, seqLensCount);
        }
        else if (useDynDecode)
        {
            checkKvarnGqaCuda(cudaFuncSetAttribute(kvarnGqaDecodeDynKernel<__nv_bfloat16, kThreads>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(dynSmemBytes)),
                "kvarn_gqa_decode bf16 dyn smem attribute");
            kvarnGqaDecodeDynKernel<__nv_bfloat16, kThreads><<<grid, kThreads, dynSmemBytes, stream>>>(
                static_cast<__nv_bfloat16 const*>(q), view, blockIds, static_cast<__nv_bfloat16 const*>(sinkK),
                static_cast<__nv_bfloat16 const*>(sinkV), static_cast<__nv_bfloat16 const*>(tailK),
                static_cast<__nv_bfloat16 const*>(tailV), seqLens, static_cast<__nv_bfloat16*>(output), numQueries,
                numBlocks, numHeads, numKvHeads, seqLensCount, sinkTokens, sinkBatch, tailTokens, tailBatch);
        }
        else
        {
            kvarnGqaDecodeParallelKernel<__nv_bfloat16, 256><<<grid, kThreads, 0, stream>>>(
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
            kvarnGqaDecodeSmallKernel<__half, 256, 256><<<grid, kThreads, 0, stream>>>(static_cast<__half const*>(q),
                view, blockIds, static_cast<__half const*>(sinkK), static_cast<__half const*>(sinkV),
                static_cast<__half const*>(tailK), static_cast<__half const*>(tailV), seqLens, static_cast<__half*>(output),
                numQueries, numBlocks, numHeads, numKvHeads, seqLensCount, sinkTokens, sinkBatch, tailTokens, tailBatch);
        }
        else if (usePackedFactored)
        {
            checkKvarnGqaCuda(cudaFuncSetAttribute(kvarnGqaDecodePackedFactoredKernel<__half, kThreads>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(factoredSmemBytes)),
                "kvarn_gqa_decode fp16 factored smem attribute");
            kvarnGqaDecodePackedFactoredKernel<__half, kThreads><<<grid, kThreads, factoredSmemBytes, stream>>>(
                static_cast<__half const*>(q), view, blockIds, seqLens, static_cast<__half*>(output),
                numQueries, numBlocks, numHeads, numKvHeads, seqLensCount);
        }
        else if (useDynDecode)
        {
            checkKvarnGqaCuda(cudaFuncSetAttribute(kvarnGqaDecodeDynKernel<__half, kThreads>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(dynSmemBytes)),
                "kvarn_gqa_decode fp16 dyn smem attribute");
            kvarnGqaDecodeDynKernel<__half, kThreads><<<grid, kThreads, dynSmemBytes, stream>>>(
                static_cast<__half const*>(q), view, blockIds, static_cast<__half const*>(sinkK),
                static_cast<__half const*>(sinkV), static_cast<__half const*>(tailK), static_cast<__half const*>(tailV),
                seqLens, static_cast<__half*>(output), numQueries, numBlocks, numHeads, numKvHeads, seqLensCount,
                sinkTokens, sinkBatch, tailTokens, tailBatch);
        }
        else
        {
            kvarnGqaDecodeParallelKernel<__half, 256><<<grid, kThreads, 0, stream>>>(static_cast<__half const*>(q),
                view, blockIds, static_cast<__half const*>(sinkK), static_cast<__half const*>(sinkV),
                static_cast<__half const*>(tailK), static_cast<__half const*>(tailV), seqLens, static_cast<__half*>(output),
                numQueries, numBlocks, numHeads, numKvHeads, seqLensCount, sinkTokens, sinkBatch, tailTokens, tailBatch);
        }
    }
    checkKvarnGqaLaunch("kvarn_gqa kernel launch");
}

} // namespace kernels

TRTLLM_NAMESPACE_END

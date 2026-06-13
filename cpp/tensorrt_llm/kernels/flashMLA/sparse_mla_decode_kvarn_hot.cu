/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#include "tensorrt_llm/kernels/flashMLA/sparse_mla_decode_kvarn_hot.h"

#include "tensorrt_llm/kernels/hisparseKvarnBdrRead.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{
constexpr int32_t kHeadQ = 128;
constexpr int32_t kDqk = 576;
constexpr int32_t kDv = 512;
constexpr int32_t kTokensPerBlock = 64;
constexpr int32_t kKvLoraRank = 512;
constexpr int32_t kQkRopeHeadDim = 64;
constexpr int32_t kThreads = 256;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr int32_t kResidentKvPoolBf16 = 0;
constexpr int32_t kResidentKvPoolFp16 = 1;

// Head-grouped flash decode: grid is (rows, headGroups, splits). Each block owns
// kHeadsPerBlock query heads of one row and a contiguous slice of that row's selected
// tokens (split-K). It streams its token slice in tiles of kTileTokens, dequantizes
// each tile's K/V latent ONCE into a shared bf16 tile (shared across all heads in the
// block, removing the per-head redundant inverse-Hadamard dequant), and runs
// online-softmax flash attention per head. With numSplits==1 it finalizes directly to
// out/lse; with numSplits>1 it writes per-split partial flash state to scratch and the
// combine kernel reduces.
constexpr int32_t kHeadsPerBlock = 16;
constexpr int32_t kHeadGroups = kHeadQ / kHeadsPerBlock;
constexpr int32_t kTileTokens = 32;
constexpr int32_t kThreadsPerHead = kThreads / kHeadsPerBlock;
constexpr int32_t kMaxSplits = 16;

__device__ __forceinline__ float bf16ToFloat(void const* ptr, int64_t offset)
{
    auto const* q = reinterpret_cast<__nv_bfloat16 const*>(ptr);
    return __bfloat162float(q[offset]);
}

__device__ __forceinline__ void writeBf16(void* ptr, int64_t offset, float value)
{
    auto* out = reinterpret_cast<__nv_bfloat16*>(ptr);
    out[offset] = __float2bfloat16_rn(value);
}

// Inverse-Hadamard-128 of one C-KV dim from a per-token base cache. Bit-identical
// to readHisparseKvarnK2v2PackedCkvOriginalValue: same base[j], ascending-j
// accumulation order, same signs, same final scale.
__device__ __forceinline__ float ckvHadamardFromCache(float const* base, int32_t dim)
{
    constexpr int32_t kBdrOrder = 128;
    constexpr float kInvSqrtHadamard128 = 0.088388347648318f;
    int32_t const subblockBase = (dim / kBdrOrder) * kBdrOrder;
    int32_t const localDim = dim - subblockBase;
    float acc = 0.0F;
#pragma unroll 1
    for (int32_t j = 0; j < kBdrOrder; ++j)
    {
        float const value = base[subblockBase + j];
        int32_t const parity = __popc(static_cast<unsigned>(localDim & j)) & 1;
        acc += parity ? -value : value;
    }
    return acc * kInvSqrtHadamard128;
}

__device__ __forceinline__ float readResidentLatentValue(
    SparseMlaDecodeKvarnHotParams const& params, int64_t globalToken, int32_t dim)
{
    int64_t const offset = globalToken * params.strideResidentKvPoolToken + dim;
    if (params.residentKvPoolDtype == kResidentKvPoolBf16)
    {
        auto const* pool = reinterpret_cast<__nv_bfloat16 const*>(params.residentKvPool);
        return __bfloat162float(pool[offset]);
    }
    if (params.residentKvPoolDtype == kResidentKvPoolFp16)
    {
        auto const* pool = reinterpret_cast<__half const*>(params.residentKvPool);
        return __half2float(pool[offset]);
    }
    return 0.0F;
}

struct HiSparseResidentTokenAddress
{
    uint8_t status;
    int64_t globalToken;
};

__device__ __forceinline__ uint8_t readRequestTopkToken(
    SparseMlaDecodeKvarnHotParams const& params, int32_t batch, int32_t s, int32_t k, int32_t& token)
{
    if (params.requestTopkIndices == nullptr)
    {
        return kHotReadInvalidIndex;
    }
    int64_t const requestIndexBase = static_cast<int64_t>(batch) * params.strideRequestTopkB
        + static_cast<int64_t>(s) * params.strideRequestTopkSQ;
    token = params.requestTopkIndices[requestIndexBase + k];
    return kHotReadOk;
}

__device__ __forceinline__ HiSparseResidentTokenAddress decodeResidentTokenAddress(
    SparseMlaDecodeKvarnHotParams const& params, int32_t row, int32_t requestToken)
{
    HiSparseResidentTokenAddress address{kHotReadOk, -1};
    if (params.residentKvLens == nullptr || params.residentReqIdx == nullptr || params.residentRequestIds == nullptr
        || params.residentKvPool == nullptr || params.residentBlockTable == nullptr
        || params.residentTailBlockPos == nullptr || params.residentTailTokenCount == nullptr
        || params.residentTailValid == nullptr)
    {
        address.status = kHotReadInvalidIndex;
        return address;
    }
    if (params.residentKvPoolDtype != kResidentKvPoolBf16 && params.residentKvPoolDtype != kResidentKvPoolFp16)
    {
        address.status = kHotReadInvalidIndex;
        return address;
    }
    if (row < 0 || row >= params.residentRows)
    {
        address.status = kHotReadInvalidIndex;
        return address;
    }
    if (params.residentRequestIds[row] < 0)
    {
        address.status = kHotReadInvalidIndex;
        return address;
    }

    int64_t const kvLen = params.residentKvLens[row];
    if (requestToken < 0 || static_cast<int64_t>(requestToken) >= kvLen)
    {
        address.status = kHotReadInvalidIndex;
        return address;
    }

    int32_t const blockPos = requestToken / params.tokensPerBlock;
    int32_t const tokenOffset = requestToken % params.tokensPerBlock;
    bool const isSink = blockPos < params.residentSinkBlocks;
    bool const isTail = params.residentTailValid[row] && blockPos == params.residentTailBlockPos[row]
        && tokenOffset < params.residentTailTokenCount[row];
    if (!isSink && !isTail)
    {
        address.status = kHotReadInvalidIndex;
        return address;
    }

    int64_t const reqIdx = params.residentReqIdx[row];
    if (reqIdx < 0 || reqIdx >= params.residentBlockTableRows || blockPos < 0
        || blockPos >= params.residentBlockTableBlocks)
    {
        address.status = kHotReadInvalidIndex;
        return address;
    }
    int64_t const tableOffset = reqIdx * params.strideResidentBlockTableB
        + static_cast<int64_t>(blockPos) * params.strideResidentBlockTableBlock;
    int32_t const blockId = params.residentBlockTable[tableOffset];
    if (blockId < 0)
    {
        address.status = kHotReadInvalidIndex;
        return address;
    }

    int64_t const globalToken = static_cast<int64_t>(blockId) * params.strideFactor
        + static_cast<int64_t>(params.layerIdx) * params.tokensPerBlock + tokenOffset;
    if (globalToken < 0 || globalToken >= params.residentKvPoolTokens)
    {
        address.status = kHotReadInvalidIndex;
        return address;
    }
    address.globalToken = globalToken;
    return address;
}

struct HiSparseSelectedToken
{
    uint8_t status;
    bool active;
    bool isHot;
    uint8_t const* record;
    int32_t tokenOffset;
    int64_t residentGlobalToken;
};

__device__ __forceinline__ HiSparseSelectedToken resolveSelectedToken(
    SparseMlaDecodeKvarnHotParams const& params, int32_t row, int32_t batch, int32_t s, int32_t k, int64_t indexBase)
{
    HiSparseSelectedToken r{kHotReadOk, false, false, nullptr, 0, -1};
    int32_t const hotIndex = params.indices[indexBase + k];
    if (hotIndex < 0)
    {
        int32_t requestToken = -1;
        uint8_t const tokenStatus = readRequestTopkToken(params, batch, s, k, requestToken);
        if (tokenStatus != kHotReadOk)
        {
            r.status = tokenStatus;
            return r;
        }
        if (requestToken < 0)
        {
            r.active = false;
            return r;
        }
        r.active = true;
        HiSparseResidentTokenAddress const addr = decodeResidentTokenAddress(params, row, requestToken);
        if (addr.status != kHotReadOk)
        {
            r.status = addr.status;
            return r;
        }
        r.isHot = false;
        r.residentGlobalToken = addr.globalToken;
        return r;
    }
    r.active = true;
    HiSparseKvarnHotAddress const addr = decodeHisparseKvarnHotIndex(
        hotIndex, params.strideFactor, params.layerIdx, params.hotCapacity, params.tokensPerBlock);
    if (addr.status != kHotReadOk)
    {
        r.status = addr.status;
        return r;
    }
    r.isHot = true;
    r.record = params.hotPacked + static_cast<int64_t>(params.layerIdx) * params.strideHotLayer
        + static_cast<int64_t>(addr.hotSlot) * params.strideHotSlot;
    r.tokenOffset = addr.tokenOffset;
    return r;
}

__global__ __launch_bounds__(kThreads) void sparseMlaDecodeKvarnHotKernel(SparseMlaDecodeKvarnHotParams params)
{
    int32_t const row = static_cast<int32_t>(blockIdx.x);
    int32_t const headGroup = static_cast<int32_t>(blockIdx.y);
    int32_t const splitIdx = static_cast<int32_t>(blockIdx.z);
    int32_t const totalRows = params.b * params.sQ;
    int32_t const numSplits = params.numSplits;
    if (row >= totalRows || headGroup >= kHeadGroups || splitIdx >= numSplits)
    {
        return;
    }
    int32_t const headBase = headGroup * kHeadsPerBlock;
    int32_t const tid = static_cast<int32_t>(threadIdx.x);

    int32_t const batch = row / params.sQ;
    int32_t const s = row - batch * params.sQ;
    int32_t const rowTopK = params.topkLength == nullptr ? params.topK
        : (params.topkLengthSize == params.b ? params.topkLength[batch] : params.topkLength[row]);
    HiSparseKvarnK2v2BdrLayout const layout = makeHisparseKvarnK2v2BdrLayout(
        params.tokensPerBlock, params.kvLoraRank, params.qkRopeHeadDim);

    int64_t const indexBase = static_cast<int64_t>(batch) * params.strideIndicesB
        + static_cast<int64_t>(s) * params.strideIndicesSQ;
    int64_t const qRowBase = static_cast<int64_t>(batch) * params.strideQB + static_cast<int64_t>(s) * params.strideQSQ;
    int64_t const outRowBase
        = static_cast<int64_t>(batch) * params.strideOB + static_cast<int64_t>(s) * params.strideOSQ;
    int64_t const lseRowBase
        = static_cast<int64_t>(batch) * params.strideLseB + static_cast<int64_t>(s) * params.strideLseSQ;

    bool const splitMode = numSplits > 1;

    // Token range for this split: contiguous tile-aligned slices of [0, rowTopK).
    int32_t kStart = 0;
    int32_t kEnd = rowTopK;
    if (splitMode)
    {
        int32_t const tilesTotal = (rowTopK + kTileTokens - 1) / kTileTokens;
        int32_t const tilesPerSplit = (tilesTotal + numSplits - 1) / numSplits;
        kStart = splitIdx * tilesPerSplit * kTileTokens;
        kEnd = min(rowTopK, kStart + tilesPerSplit * kTileTokens);
        if (kStart >= kEnd)
        {
            kStart = 0;
            kEnd = 0; // empty split: contributes -inf max / 0 denom
        }
    }

    extern __shared__ float smem[];
    __nv_bfloat16* kTile = reinterpret_cast<__nv_bfloat16*>(smem);
    float* baseCache = reinterpret_cast<float*>(kTile + static_cast<int64_t>(kTileTokens) * kDqk);
    float* acc = baseCache + kKvLoraRank;
    float* runMax = acc + static_cast<int64_t>(kHeadsPerBlock) * kDv;
    float* runDenom = runMax + kHeadsPerBlock;
    float* tileScore = runDenom + kHeadsPerBlock;
    __shared__ int32_t tileStatusAgg;
    __shared__ int32_t rowCode;

    if (tid == 0)
    {
        rowCode = params.rowStatus[row] == 0 ? kHotReadOk : kHotReadUpstreamInvalid;
        if (rowCode == kHotReadOk && (rowTopK < 0 || rowTopK > params.topK))
        {
            rowCode = kHotReadBadTopKLength;
        }
        tileStatusAgg = kHotReadOk;
    }
    for (int32_t i = tid; i < kHeadsPerBlock * kDv; i += kThreads)
    {
        acc[i] = 0.0F;
    }
    for (int32_t h = tid; h < kHeadsPerBlock; h += kThreads)
    {
        runMax[h] = kNegInf;
        runDenom[h] = 0.0F;
    }
    __syncthreads();

    // Per-thread Q register cache: each thread serves one head (headLocal, fixed by
    // tid) and owns Q dims {laneH, laneH+kThreadsPerHead, ...}. Loading Q once here
    // (vs re-reading from GMEM for every token of every tile) removes the dominant
    // redundant Q traffic; values are byte-identical.
    int32_t const headLocalTop = tid / kThreadsPerHead;
    int32_t const laneHTop = tid % kThreadsPerHead;
    int32_t const headTop = headBase + headLocalTop;
    constexpr int32_t kQRegPerThread = (kDqk + kThreadsPerHead - 1) / kThreadsPerHead;
    float qReg[kQRegPerThread];
    if (rowCode == kHotReadOk)
    {
        int64_t const qBaseTop = qRowBase + static_cast<int64_t>(headTop) * params.strideQHQ;
#pragma unroll
        for (int32_t r = 0; r < kQRegPerThread; ++r)
        {
            int32_t const d = laneHTop + r * kThreadsPerHead;
            qReg[r] = (d < kDqk) ? bf16ToFloat(params.q, qBaseTop + d) : 0.0F;
        }
        for (int32_t tileStart = kStart; tileStart < kEnd; tileStart += kTileTokens)
        {
            int32_t const tileLen = min(kTileTokens, kEnd - tileStart);

            // --- (1) dequant this tile's K/V latent ONCE into kTile ---
            for (int32_t tt = 0; tt < tileLen; ++tt)
            {
                int32_t const k = tileStart + tt;
                HiSparseSelectedToken st = resolveSelectedToken(params, row, batch, s, k, indexBase);
                if (st.status != kHotReadOk && tid == 0)
                {
                    atomicCAS(&tileStatusAgg, kHotReadOk, static_cast<int32_t>(st.status));
                }
                bool const buildHot = (st.status == kHotReadOk) && st.active && st.isHot;
                if (buildHot)
                {
                    uint8_t const* tokenPacked
                        = st.record + static_cast<int64_t>(st.tokenOffset) * layout.ckvBytesPerToken;
                    uint8_t const* tokenScaleZpBytes = st.record + layout.ckvBytesPerBlock
                        + static_cast<int64_t>(st.tokenOffset) * layout.scaleZpBytesPerToken;
                    for (int32_t d = tid; d < layout.kvLoraRank; d += kThreads)
                    {
                        baseCache[d] = readHisparseKvarnK2v2PackedCkvValue(tokenPacked, tokenScaleZpBytes, d);
                    }
                    __syncthreads();
                    uint8_t const* peBytes = st.record + layout.ckvBytesPerBlock + layout.scaleZpBytesPerBlock;
                    for (int32_t d = tid; d < kDqk; d += kThreads)
                    {
                        float val;
                        if (d < layout.kvLoraRank)
                        {
                            val = ckvHadamardFromCache(baseCache, d);
                        }
                        else
                        {
                            uint8_t const byte = peBytes[static_cast<int64_t>(st.tokenOffset) * layout.qkRopeHeadDim
                                + (d - layout.kvLoraRank)];
                            val = readHisparseFp8E4m3Byte(byte);
                        }
                        kTile[static_cast<int64_t>(tt) * kDqk + d] = __float2bfloat16_rn(val);
                    }
                    __syncthreads();
                }
                else if (st.status == kHotReadOk && st.active && !st.isHot)
                {
                    for (int32_t d = tid; d < kDqk; d += kThreads)
                    {
                        float const val = readResidentLatentValue(params, st.residentGlobalToken, d);
                        kTile[static_cast<int64_t>(tt) * kDqk + d] = __float2bfloat16_rn(val);
                    }
                    __syncthreads();
                }
                else
                {
                    for (int32_t d = tid; d < kDqk; d += kThreads)
                    {
                        kTile[static_cast<int64_t>(tt) * kDqk + d] = __float2bfloat16_rn(0.0F);
                    }
                    __syncthreads();
                }
            }

            // --- (2) per-head scores + online-softmax update ---
            int32_t const headLocal = headLocalTop;
            int32_t const laneH = laneHTop;

            for (int32_t tt = 0; tt < tileLen; ++tt)
            {
                int32_t const k = tileStart + tt;
                int32_t const hotIndex = params.indices[indexBase + k];
                bool active;
                if (hotIndex < 0)
                {
                    int32_t requestToken = -1;
                    uint8_t const ts = readRequestTopkToken(params, batch, s, k, requestToken);
                    active = (ts == kHotReadOk) && (requestToken >= 0);
                }
                else
                {
                    active = true;
                }
                float part = 0.0F;
                __nv_bfloat16 const* kt = kTile + static_cast<int64_t>(tt) * kDqk;
#pragma unroll
                for (int32_t r = 0; r < kQRegPerThread; ++r)
                {
                    int32_t const d = laneH + r * kThreadsPerHead;
                    if (d < kDqk)
                    {
                        part += qReg[r] * __bfloat162float(kt[d]);
                    }
                }
#pragma unroll
                for (int32_t off = kThreadsPerHead / 2; off > 0; off >>= 1)
                {
                    part += __shfl_down_sync(0xffffffffu, part, off, kThreadsPerHead);
                }
                if (laneH == 0)
                {
                    tileScore[headLocal * kTileTokens + tt] = active ? (part * params.smScale) : kNegInf;
                }
            }
            __syncthreads();

            float tileMax = kNegInf;
            for (int32_t tt = 0; tt < tileLen; ++tt)
            {
                tileMax = fmaxf(tileMax, tileScore[headLocal * kTileTokens + tt]);
            }
            float const prevMax = runMax[headLocal];
            float const prevDenom = runDenom[headLocal];
            float const newMax = fmaxf(prevMax, tileMax);
            float const correction = (prevMax == kNegInf) ? 0.0F : expf(prevMax - newMax);
            for (int32_t d = laneH; d < kDv; d += kThreadsPerHead)
            {
                float a = acc[headLocal * kDv + d] * correction;
                for (int32_t tt = 0; tt < tileLen; ++tt)
                {
                    float const sc = tileScore[headLocal * kTileTokens + tt];
                    if (sc == kNegInf)
                    {
                        continue;
                    }
                    float const w = expf(sc - newMax);
                    a += w * __bfloat162float(kTile[static_cast<int64_t>(tt) * kDqk + d]);
                }
                acc[headLocal * kDv + d] = a;
            }
            if (laneH == 0)
            {
                float tileDenom = 0.0F;
                for (int32_t tt = 0; tt < tileLen; ++tt)
                {
                    float const sc = tileScore[headLocal * kTileTokens + tt];
                    if (sc == kNegInf)
                    {
                        continue;
                    }
                    tileDenom += expf(sc - newMax);
                }
                runDenom[headLocal] = prevDenom * correction + tileDenom;
                runMax[headLocal] = newMax;
            }
            __syncthreads();
        }
    }

    if (tid == 0 && rowCode == kHotReadOk && tileStatusAgg != kHotReadOk)
    {
        rowCode = tileStatusAgg;
    }
    __syncthreads();

    int32_t const headLocalF = tid / kThreadsPerHead;
    int32_t const laneF = tid % kThreadsPerHead;
    int32_t const headF = headBase + headLocalF;

    // --- split-mode: write partial flash state to scratch; combine kernel finalizes ---
    if (splitMode)
    {
        int64_t const partBase
            = ((static_cast<int64_t>(row) * params.hQ + headF) * numSplits + splitIdx);
        // a failed row marks all its partials as empty (-inf/0) so combine yields zero.
        bool const failed = (rowCode != kHotReadOk);
        float const m = failed ? kNegInf : runMax[headLocalF];
        float const d = failed ? 0.0F : runDenom[headLocalF];
        float* pacc = params.partialAcc + partBase * kDv;
        for (int32_t dd = laneF; dd < kDv; dd += kThreadsPerHead)
        {
            pacc[dd] = failed ? 0.0F : acc[headLocalF * kDv + dd];
        }
        if (laneF == 0)
        {
            params.partialMax[partBase] = m;
            params.partialDenom[partBase] = d;
        }
        return;
    }

    // --- single-split: finalize directly ---
    int64_t const outBase = outRowBase + static_cast<int64_t>(headF) * params.strideOHQ;
    if (rowCode != kHotReadOk)
    {
        for (int32_t d = laneF; d < kDv; d += kThreadsPerHead)
        {
            writeBf16(params.out, outBase + d, 0.0F);
        }
        if (laneF == 0)
        {
            params.lse[lseRowBase + headF] = kNegInf;
        }
        return;
    }

    float const sinkVal = params.attnSink == nullptr ? kNegInf : params.attnSink[headF];
    float const mF = runMax[headLocalF];
    float finalMax = (sinkVal != kNegInf) ? fmaxf(mF, sinkVal) : mF;
    float denom = runDenom[headLocalF];
    if (finalMax != mF)
    {
        float const corr = (mF == kNegInf) ? 0.0F : expf(mF - finalMax);
        denom = denom * corr;
    }
    if (sinkVal != kNegInf)
    {
        denom += expf(sinkVal - finalMax);
    }
    float const accScale = (finalMax == mF) ? 1.0F : ((mF == kNegInf) ? 0.0F : expf(mF - finalMax));
    float const invDenom = denom > 0.0F ? 1.0F / denom : 0.0F;
    for (int32_t d = laneF; d < kDv; d += kThreadsPerHead)
    {
        float const a = acc[headLocalF * kDv + d] * accScale;
        writeBf16(params.out, outBase + d, a * invDenom);
    }
    if (laneF == 0)
    {
        params.lse[lseRowBase + headF] = (denom > 0.0F) ? (logf(denom) + finalMax) : kNegInf;
    }
}

// Combine partial flash states across splits into final out/lse, per (row, head).
// One block per (row, head); 128 threads cooperate over kDv dims.
__global__ __launch_bounds__(128) void sparseMlaDecodeKvarnHotCombineKernel(SparseMlaDecodeKvarnHotParams params)
{
    int32_t const row = static_cast<int32_t>(blockIdx.x);
    int32_t const head = static_cast<int32_t>(blockIdx.y);
    int32_t const totalRows = params.b * params.sQ;
    if (row >= totalRows || head >= params.hQ)
    {
        return;
    }
    int32_t const numSplits = params.numSplits;
    int32_t const tid = static_cast<int32_t>(threadIdx.x);

    int32_t const batch = row / params.sQ;
    int32_t const s = row - batch * params.sQ;
    int64_t const outBase = static_cast<int64_t>(batch) * params.strideOB
        + static_cast<int64_t>(s) * params.strideOSQ + static_cast<int64_t>(head) * params.strideOHQ;
    int64_t const lseOff = static_cast<int64_t>(batch) * params.strideLseB
        + static_cast<int64_t>(s) * params.strideLseSQ + head;

    int64_t const partRowHead = (static_cast<int64_t>(row) * params.hQ + head) * numSplits;

    // global max over splits + optional sink
    __shared__ float sMax;
    __shared__ float sDenom;
    float const sinkVal = params.attnSink == nullptr ? kNegInf : params.attnSink[head];
    if (tid == 0)
    {
        float gmax = sinkVal;
        for (int32_t sp = 0; sp < numSplits; ++sp)
        {
            gmax = fmaxf(gmax, params.partialMax[partRowHead + sp]);
        }
        float gden = (sinkVal != kNegInf && gmax != kNegInf) ? expf(sinkVal - gmax) : 0.0F;
        for (int32_t sp = 0; sp < numSplits; ++sp)
        {
            float const m = params.partialMax[partRowHead + sp];
            if (m == kNegInf)
            {
                continue;
            }
            gden += params.partialDenom[partRowHead + sp] * expf(m - gmax);
        }
        sMax = gmax;
        sDenom = gden;
    }
    __syncthreads();

    float const gmax = sMax;
    float const gden = sDenom;
    float const invDenom = gden > 0.0F ? 1.0F / gden : 0.0F;

    for (int32_t d = tid; d < kDv; d += 128)
    {
        float o = 0.0F;
        for (int32_t sp = 0; sp < numSplits; ++sp)
        {
            float const m = params.partialMax[partRowHead + sp];
            if (m == kNegInf)
            {
                continue;
            }
            float const scale = expf(m - gmax);
            o += params.partialAcc[(partRowHead + sp) * kDv + d] * scale;
        }
        writeBf16(params.out, outBase + d, o * invDenom);
    }
    if (tid == 0)
    {
        params.lse[lseOff] = (gden > 0.0F) ? (logf(gden) + gmax) : kNegInf;
    }
}

} // namespace

void invokeSparseMlaDecodeKvarnHot(SparseMlaDecodeKvarnHotParams const& paramsIn, cudaStream_t stream)
{
    SparseMlaDecodeKvarnHotParams params = paramsIn;
    if (params.hQ != kHeadQ || params.dQk != kDqk || params.dV != kDv || params.tokensPerBlock != kTokensPerBlock
        || params.kvLoraRank != kKvLoraRank || params.qkRopeHeadDim != kQkRopeHeadDim || params.kvarnBits != 2)
    {
        throw std::runtime_error(
            "sparse MLA KVarN-hot decode requires production V3.2 shape h_q=128,d_qk=576,d_v=512,tpb=64,kvarn_k2v2");
    }
    if (params.b <= 0 || params.sQ <= 0 || params.topK <= 0 || params.numLayers <= 0 || params.hotCapacity <= 0)
    {
        throw std::runtime_error("sparse MLA KVarN-hot decode requires positive batch, s_q, topk, layers, and hot capacity");
    }
    if (params.residentKvPool != nullptr && params.residentKvPoolDtype != kResidentKvPoolBf16
        && params.residentKvPoolDtype != kResidentKvPoolFp16)
    {
        throw std::runtime_error("sparse MLA KVarN-hot decode requires resident KV pool dtype bf16 or fp16");
    }
    if (params.topK > 2048)
    {
        throw std::runtime_error("sparse MLA KVarN-hot decode currently supports topk <= 2048");
    }

    int32_t const totalRows = params.b * params.sQ;
    // adaptive split-K: add token-range splits when (rows * headGroups) under-fills the
    // GPU, so few-row decode batches keep all SMs busy without redundant dequant.
    int32_t const baseBlocks = totalRows * kHeadGroups;
    int32_t const tilesTotal = (params.topK + kTileTokens - 1) / kTileTokens;
    constexpr int32_t kTargetBlocks = 304; // ~2x SM count on B200
    int32_t numSplits = (baseBlocks >= kTargetBlocks) ? 1 : ((kTargetBlocks + baseBlocks - 1) / baseBlocks);
    numSplits = min(numSplits, min(kMaxSplits, max(1, tilesTotal)));
    if (numSplits < 1)
    {
        numSplits = 1;
    }
    params.numSplits = numSplits;

    size_t const sharedBytes = static_cast<size_t>(kTileTokens) * kDqk * sizeof(__nv_bfloat16)
        + static_cast<size_t>(kKvLoraRank) * sizeof(float)
        + static_cast<size_t>(kHeadsPerBlock) * kDv * sizeof(float)
        + static_cast<size_t>(2 * kHeadsPerBlock) * sizeof(float)
        + static_cast<size_t>(kHeadsPerBlock) * kTileTokens * sizeof(float);

    static bool attrSet = false;
    if (!attrSet)
    {
        cudaFuncSetAttribute(
            sparseMlaDecodeKvarnHotKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sharedBytes));
        attrSet = true;
    }

    float* partialAcc = nullptr;
    float* partialMax = nullptr;
    float* partialDenom = nullptr;
    if (numSplits > 1)
    {
        size_t const nRowHeadSplit = static_cast<size_t>(totalRows) * params.hQ * numSplits;
        cudaMallocAsync(reinterpret_cast<void**>(&partialAcc), nRowHeadSplit * kDv * sizeof(float), stream);
        cudaMallocAsync(reinterpret_cast<void**>(&partialMax), nRowHeadSplit * sizeof(float), stream);
        cudaMallocAsync(reinterpret_cast<void**>(&partialDenom), nRowHeadSplit * sizeof(float), stream);
        params.partialAcc = partialAcc;
        params.partialMax = partialMax;
        params.partialDenom = partialDenom;
    }

    dim3 const grid(totalRows, kHeadGroups, numSplits);
    sparseMlaDecodeKvarnHotKernel<<<grid, kThreads, sharedBytes, stream>>>(params);
    auto err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        if (partialAcc != nullptr)
        {
            cudaFreeAsync(partialAcc, stream);
            cudaFreeAsync(partialMax, stream);
            cudaFreeAsync(partialDenom, stream);
        }
        throw std::runtime_error("sparse MLA KVarN-hot decode kernel launch failed");
    }

    if (numSplits > 1)
    {
        dim3 const cgrid(totalRows, params.hQ, 1);
        sparseMlaDecodeKvarnHotCombineKernel<<<cgrid, 128, 0, stream>>>(params);
        err = cudaGetLastError();
        cudaFreeAsync(partialAcc, stream);
        cudaFreeAsync(partialMax, stream);
        cudaFreeAsync(partialDenom, stream);
        if (err != cudaSuccess)
        {
            throw std::runtime_error("sparse MLA KVarN-hot decode combine kernel launch failed");
        }
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END

/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#include "tensorrt_llm/kernels/flashMLA/sparse_mla_decode_kvarn_hot.h"

#include "tensorrt_llm/kernels/hisparseKvarnBdrRead.cuh"
#include "tensorrt_llm/kernels/flashMLA/nvfp4_sparse/params.h"
#include "tensorrt_llm/kernels/flashMLA/nvfp4_sparse/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

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
constexpr int32_t kBlockSizeTopK = 64;
constexpr int32_t kFixedOverheadBlocks = 5;
constexpr int32_t kMaxNumSmParts = 4096;
constexpr int32_t kSingleBatchSchedulerOverhead = 16;
constexpr int32_t kShortBatchSchedulerOverhead = 15;
constexpr int32_t kMediumBatchSchedulerOverhead = 14;
constexpr int32_t kLongBatchSchedulerOverhead = 5;
constexpr float kLog2E = 1.4426950408889634F;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr int32_t kResidentKvPoolBf16 = 0;
constexpr int32_t kResidentKvPoolFp16 = 1;

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

__device__ __forceinline__ float blockReduceSum(float value, float* scratch)
{
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return scratch[0];
}

__global__ void sparseMlaDecodeKvarnHotKernel(SparseMlaDecodeKvarnHotParams params)
{
    int32_t const row = static_cast<int32_t>(blockIdx.x);
    int32_t const head = static_cast<int32_t>(blockIdx.y);
    int32_t const totalRows = params.b * params.sQ;
    if (row >= totalRows || head >= params.hQ)
    {
        return;
    }

    extern __shared__ float shared[];
    float* scores = shared;
    float* reduce = scores + params.topK;

    int32_t const batch = row / params.sQ;
    int32_t const s = row - batch * params.sQ;
    int32_t const rowTopK = params.topkLength == nullptr ? params.topK
        : (params.topkLengthSize == params.b ? params.topkLength[batch] : params.topkLength[row]);
    HiSparseKvarnK2v2BdrLayout const layout = makeHisparseKvarnK2v2BdrLayout(
        params.tokensPerBlock, params.kvLoraRank, params.qkRopeHeadDim);

    __shared__ int32_t rowCode;
    __shared__ float rowMax;
    __shared__ float rowDenom;
    if (threadIdx.x == 0)
    {
        rowCode = params.rowStatus[row] == 0 ? kHotReadOk : kHotReadUpstreamInvalid;
        if (rowCode == kHotReadOk && (rowTopK < 0 || rowTopK > params.topK))
        {
            rowCode = kHotReadBadTopKLength;
        }
        rowMax = kNegInf;
        rowDenom = 0.0F;
    }
    __syncthreads();

    int64_t const qBase = static_cast<int64_t>(batch) * params.strideQB
        + static_cast<int64_t>(s) * params.strideQSQ + static_cast<int64_t>(head) * params.strideQHQ;
    int64_t const indexBase = static_cast<int64_t>(batch) * params.strideIndicesB
        + static_cast<int64_t>(s) * params.strideIndicesSQ;

    for (int32_t k = 0; k < params.topK; ++k)
    {
        float scorePart = 0.0F;
        bool activeToken = false;
        if (rowCode == kHotReadOk && k < rowTopK)
        {
            int32_t const hotIndex = params.indices[indexBase + k];
            if (hotIndex < 0)
            {
                int32_t requestToken = -1;
                uint8_t const tokenStatus = readRequestTopkToken(params, batch, s, k, requestToken);
                if (tokenStatus != kHotReadOk)
                {
                    atomicCAS(&rowCode, kHotReadOk, static_cast<int32_t>(tokenStatus));
                }
                else if (requestToken >= 0)
                {
                    activeToken = true;
                    HiSparseResidentTokenAddress const residentAddress
                        = decodeResidentTokenAddress(params, row, requestToken);
                    if (residentAddress.status != kHotReadOk)
                    {
                        atomicCAS(&rowCode, kHotReadOk, static_cast<int32_t>(residentAddress.status));
                    }
                    else
                    {
                        for (int32_t dim = threadIdx.x; dim < params.dQk; dim += blockDim.x)
                        {
                            float const qVal = bf16ToFloat(params.q, qBase + dim);
                            float const kVal = readResidentLatentValue(params, residentAddress.globalToken, dim);
                            scorePart += qVal * kVal;
                        }
                    }
                }
            }
            else
            {
                activeToken = true;
                HiSparseKvarnHotAddress const address = decodeHisparseKvarnHotIndex(
                    hotIndex, params.strideFactor, params.layerIdx, params.hotCapacity, params.tokensPerBlock);
                if (address.status != kHotReadOk)
                {
                    atomicCAS(&rowCode, kHotReadOk, static_cast<int32_t>(address.status));
                }
                else
                {
                    uint8_t const* record = params.hotPacked
                        + static_cast<int64_t>(params.layerIdx) * params.strideHotLayer
                        + static_cast<int64_t>(address.hotSlot) * params.strideHotSlot;
                    for (int32_t dim = threadIdx.x; dim < params.dQk; dim += blockDim.x)
                    {
                        float const qVal = bf16ToFloat(params.q, qBase + dim);
                        float const kVal = __bfloat162float(
                            readHisparseKvarnK2v2BdrLatentValue(record, layout, address.tokenOffset, dim));
                        scorePart += qVal * kVal;
                    }
                }
            }
        }
        float const score = blockReduceSum(scorePart, reduce) * params.smScale;
        if (threadIdx.x == 0)
        {
            scores[k] = (rowCode == kHotReadOk && k < rowTopK && activeToken) ? score : kNegInf;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        if (rowCode == kHotReadOk)
        {
            float maxVal = params.attnSink == nullptr ? kNegInf : params.attnSink[head];
            for (int32_t k = 0; k < rowTopK; ++k)
            {
                maxVal = fmaxf(maxVal, scores[k]);
            }
            float denom = params.attnSink == nullptr ? 0.0F : expf(params.attnSink[head] - maxVal);
            for (int32_t k = 0; k < rowTopK; ++k)
            {
                if (scores[k] == kNegInf)
                {
                    scores[k] = 0.0F;
                    continue;
                }
                float const weight = expf(scores[k] - maxVal);
                scores[k] = weight;
                denom += weight;
            }
            float const invDenom = denom > 0.0F ? 1.0F / denom : 0.0F;
            for (int32_t k = 0; k < rowTopK; ++k)
            {
                scores[k] *= invDenom;
            }
            rowMax = maxVal;
            rowDenom = denom;
        }
    }
    __syncthreads();

    int64_t const outBase = static_cast<int64_t>(batch) * params.strideOB
        + static_cast<int64_t>(s) * params.strideOSQ + static_cast<int64_t>(head) * params.strideOHQ;
    if (rowCode != kHotReadOk)
    {
        for (int32_t dim = threadIdx.x; dim < params.dV; dim += blockDim.x)
        {
            writeBf16(params.out, outBase + dim, 0.0F);
        }
        if (threadIdx.x == 0)
        {
            params.lse[static_cast<int64_t>(batch) * params.strideLseB + static_cast<int64_t>(s) * params.strideLseSQ
                + head] = kNegInf;
        }
        return;
    }

    __shared__ int32_t valueCode;
    if (threadIdx.x == 0)
    {
        valueCode = kHotReadOk;
    }
    __syncthreads();

    for (int32_t k = threadIdx.x; k < rowTopK; k += blockDim.x)
    {
        int32_t const hotIndex = params.indices[indexBase + k];
        if (hotIndex < 0)
        {
            int32_t requestToken = -1;
            uint8_t const tokenStatus = readRequestTopkToken(params, batch, s, k, requestToken);
            if (tokenStatus != kHotReadOk)
            {
                atomicCAS(&valueCode, kHotReadOk, static_cast<int32_t>(tokenStatus));
            }
            else if (requestToken >= 0)
            {
                HiSparseResidentTokenAddress const residentAddress
                    = decodeResidentTokenAddress(params, row, requestToken);
                if (residentAddress.status != kHotReadOk)
                {
                    atomicCAS(&valueCode, kHotReadOk, static_cast<int32_t>(residentAddress.status));
                }
            }
        }
        else
        {
            HiSparseKvarnHotAddress const address = decodeHisparseKvarnHotIndex(
                hotIndex, params.strideFactor, params.layerIdx, params.hotCapacity, params.tokensPerBlock);
            if (address.status != kHotReadOk)
            {
                atomicCAS(&valueCode, kHotReadOk, static_cast<int32_t>(address.status));
            }
        }
    }
    __syncthreads();

    if (valueCode != kHotReadOk)
    {
        for (int32_t dim = threadIdx.x; dim < params.dV; dim += blockDim.x)
        {
            writeBf16(params.out, outBase + dim, 0.0F);
        }
        if (threadIdx.x == 0)
        {
            params.lse[static_cast<int64_t>(batch) * params.strideLseB + static_cast<int64_t>(s) * params.strideLseSQ
                + head] = kNegInf;
        }
        return;
    }

    for (int32_t dim = threadIdx.x; dim < params.dV; dim += blockDim.x)
    {
        float acc = 0.0F;
        for (int32_t k = 0; k < rowTopK; ++k)
        {
            if (scores[k] == kNegInf)
            {
                continue;
            }
            int32_t const hotIndex = params.indices[indexBase + k];
            float vVal = 0.0F;
            if (hotIndex < 0)
            {
                int32_t requestToken = -1;
                uint8_t const tokenStatus = readRequestTopkToken(params, batch, s, k, requestToken);
                if (tokenStatus == kHotReadOk && requestToken >= 0)
                {
                    HiSparseResidentTokenAddress const residentAddress
                        = decodeResidentTokenAddress(params, row, requestToken);
                    if (residentAddress.status == kHotReadOk)
                    {
                        vVal = readResidentLatentValue(params, residentAddress.globalToken, dim);
                    }
                }
            }
            else
            {
                HiSparseKvarnHotAddress const address = decodeHisparseKvarnHotIndex(
                    hotIndex, params.strideFactor, params.layerIdx, params.hotCapacity, params.tokensPerBlock);
                uint8_t const* record = params.hotPacked
                    + static_cast<int64_t>(params.layerIdx) * params.strideHotLayer
                    + static_cast<int64_t>(address.hotSlot) * params.strideHotSlot;
                vVal = __bfloat162float(readHisparseKvarnK2v2BdrLatentValue(record, layout, address.tokenOffset, dim));
            }
            acc += scores[k] * vVal;
        }
        writeBf16(params.out, outBase + dim, acc);
    }
    if (threadIdx.x == 0)
    {
        params.lse[static_cast<int64_t>(batch) * params.strideLseB + static_cast<int64_t>(s) * params.strideLseSQ
            + head] = logf(rowDenom) + rowMax;
    }

}

__device__ __forceinline__ int32_t ceilDivDevice(int32_t x, int32_t y)
{
    return (x + y - 1) / y;
}

__device__ __forceinline__ void writeFailClosedRowHead(SparseMlaDecodeKvarnHotParams const& params, int32_t row,
    int32_t head, int32_t code)
{
    if (params.rowHeadStatus != nullptr && code != kHotReadOk)
    {
        atomicCAS(params.rowHeadStatus + static_cast<int64_t>(row) * params.hQ + head, kHotReadOk, code);
    }
}

__device__ __forceinline__ int32_t getKvarnHotRowTopK(
    SparseMlaDecodeKvarnHotParams const& params, int32_t row, int32_t batch)
{
    return params.topkLength == nullptr ? params.topK
        : (params.topkLengthSize == params.b ? params.topkLength[batch] : params.topkLength[row]);
}

__device__ __forceinline__ void zeroSplitAccum(SparseMlaDecodeKvarnHotParams const& params, int32_t splitIdx,
    int32_t s, int32_t head)
{
    for (int32_t dim = threadIdx.x; dim < params.dV; dim += blockDim.x)
    {
        params.outAccum[static_cast<int64_t>(splitIdx) * params.strideOAccumSplit
            + static_cast<int64_t>(s) * params.strideOAccumSQ + static_cast<int64_t>(head) * params.strideOAccumHQ
            + dim] = 0.0F;
    }
    if (threadIdx.x == 0)
    {
        params.lseAccum[static_cast<int64_t>(splitIdx) * params.strideLseAccumSplit
            + static_cast<int64_t>(s) * params.strideLseAccumSQ + head] = kNegInf;
    }
}

__global__ void sparseMlaDecodeKvarnHotSplitProducerKernel(SparseMlaDecodeKvarnHotParams params)
{
    int32_t const s = static_cast<int32_t>(blockIdx.x);
    int32_t const partition = static_cast<int32_t>(blockIdx.y);
    int32_t const head = static_cast<int32_t>(blockIdx.z);
    if (s >= params.sQ || partition >= params.numSmParts || head >= params.hQ)
    {
        return;
    }

    auto const* metadata = reinterpret_cast<DecodingSchedMeta const*>(params.tileSchedulerMetadata);
    DecodingSchedMeta const sched = metadata[partition];
    if (sched.begin_req_idx >= params.b)
    {
        return;
    }

    extern __shared__ float shared[];
    float* scores = shared;
    float* reduce = scores + kBlockSizeTopK;
    HiSparseKvarnK2v2BdrLayout const layout = makeHisparseKvarnK2v2BdrLayout(
        params.tokensPerBlock, params.kvLoraRank, params.qkRopeHeadDim);

    for (int32_t batch = sched.begin_req_idx; batch <= sched.end_req_idx; ++batch)
    {
        int32_t const row = batch * params.sQ + s;
        int32_t const rowTopK = getKvarnHotRowTopK(params, row, batch);
        int32_t const totalTopkBlocks = max(ceilDivDevice(params.topK, kBlockSizeTopK), 1);
        int32_t const startBlock = batch == sched.begin_req_idx ? sched.begin_block_idx : 0;
        int32_t const endBlock = batch == sched.end_req_idx ? sched.end_block_idx : totalTopkBlocks;
        int32_t const splitIdx = batch == sched.begin_req_idx
            ? params.numSplits[batch] + sched.begin_split_idx
            : params.numSplits[batch];

        __shared__ int32_t rowCode;
        __shared__ float localMaxLog2;
        __shared__ float localDenom;
        if (threadIdx.x == 0)
        {
            rowCode = params.rowStatus[row] == 0 ? kHotReadOk : kHotReadUpstreamInvalid;
            if (rowCode == kHotReadOk && (rowTopK < 0 || rowTopK > params.topK))
            {
                rowCode = kHotReadBadTopKLength;
            }
            if (rowCode == kHotReadOk && (startBlock < 0 || endBlock < startBlock || endBlock - startBlock > 1))
            {
                rowCode = kHotReadInvalidIndex;
            }
            localMaxLog2 = kNegInf;
            localDenom = 0.0F;
        }
        __syncthreads();

        int32_t const startK = startBlock * kBlockSizeTopK;
        int32_t const endK = min(endBlock * kBlockSizeTopK, rowTopK);
        int32_t const localK = max(endK - startK, 0);
        int64_t const qBase = static_cast<int64_t>(batch) * params.strideQB
            + static_cast<int64_t>(s) * params.strideQSQ + static_cast<int64_t>(head) * params.strideQHQ;
        int64_t const indexBase = static_cast<int64_t>(batch) * params.strideIndicesB
            + static_cast<int64_t>(s) * params.strideIndicesSQ;

        for (int32_t local = 0; local < kBlockSizeTopK; ++local)
        {
            int32_t const k = startK + local;
            float scorePart = 0.0F;
            bool activeToken = false;
            if (rowCode == kHotReadOk && local < localK)
            {
                int32_t const hotIndex = params.indices[indexBase + k];
                if (hotIndex < 0)
                {
                    int32_t requestToken = -1;
                    uint8_t const tokenStatus = readRequestTopkToken(params, batch, s, k, requestToken);
                    if (tokenStatus != kHotReadOk)
                    {
                        atomicCAS(&rowCode, kHotReadOk, static_cast<int32_t>(tokenStatus));
                    }
                    else if (requestToken >= 0)
                    {
                        activeToken = true;
                        HiSparseResidentTokenAddress const residentAddress
                            = decodeResidentTokenAddress(params, row, requestToken);
                        if (residentAddress.status != kHotReadOk)
                        {
                            atomicCAS(&rowCode, kHotReadOk, static_cast<int32_t>(residentAddress.status));
                        }
                        else
                        {
                            for (int32_t dim = threadIdx.x; dim < params.dQk; dim += blockDim.x)
                            {
                                float const qVal = bf16ToFloat(params.q, qBase + dim);
                                float const kVal = readResidentLatentValue(params, residentAddress.globalToken, dim);
                                scorePart += qVal * kVal;
                            }
                        }
                    }
                }
                else
                {
                    activeToken = true;
                    HiSparseKvarnHotAddress const address = decodeHisparseKvarnHotIndex(
                        hotIndex, params.strideFactor, params.layerIdx, params.hotCapacity, params.tokensPerBlock);
                    if (address.status != kHotReadOk)
                    {
                        atomicCAS(&rowCode, kHotReadOk, static_cast<int32_t>(address.status));
                    }
                    else
                    {
                        uint8_t const* record = params.hotPacked
                            + static_cast<int64_t>(params.layerIdx) * params.strideHotLayer
                            + static_cast<int64_t>(address.hotSlot) * params.strideHotSlot;
                        for (int32_t dim = threadIdx.x; dim < params.dQk; dim += blockDim.x)
                        {
                            float const qVal = bf16ToFloat(params.q, qBase + dim);
                            float const kVal = __bfloat162float(
                                readHisparseKvarnK2v2BdrLatentValue(record, layout, address.tokenOffset, dim));
                            scorePart += qVal * kVal;
                        }
                    }
                }
            }
            float const score = blockReduceSum(scorePart, reduce) * params.smScale * kLog2E;
            if (threadIdx.x == 0)
            {
                scores[local] = (rowCode == kHotReadOk && local < localK && activeToken) ? score : kNegInf;
            }
            __syncthreads();
        }

        if (threadIdx.x == 0)
        {
            if (rowCode == kHotReadOk)
            {
                float maxVal = kNegInf;
                for (int32_t local = 0; local < localK; ++local)
                {
                    maxVal = fmaxf(maxVal, scores[local]);
                }
                float denom = 0.0F;
                if (maxVal != kNegInf)
                {
                    for (int32_t local = 0; local < localK; ++local)
                    {
                        if (scores[local] == kNegInf)
                        {
                            scores[local] = 0.0F;
                        }
                        else
                        {
                            float const weight = exp2f(scores[local] - maxVal);
                            scores[local] = weight;
                            denom += weight;
                        }
                    }
                    float const invDenom = denom > 0.0F ? 1.0F / denom : 0.0F;
                    for (int32_t local = 0; local < localK; ++local)
                    {
                        scores[local] *= invDenom;
                    }
                }
                localMaxLog2 = maxVal;
                localDenom = denom;
            }
            if (rowCode != kHotReadOk)
            {
                writeFailClosedRowHead(params, row, head, rowCode);
            }
        }
        __syncthreads();

        if (rowCode != kHotReadOk || localDenom == 0.0F || localMaxLog2 == kNegInf)
        {
            zeroSplitAccum(params, splitIdx, s, head);
            __syncthreads();
            continue;
        }

        for (int32_t dim = threadIdx.x; dim < params.dV; dim += blockDim.x)
        {
            float acc = 0.0F;
            for (int32_t local = 0; local < localK; ++local)
            {
                if (scores[local] == 0.0F)
                {
                    continue;
                }
                int32_t const k = startK + local;
                int32_t const hotIndex = params.indices[indexBase + k];
                float vVal = 0.0F;
                if (hotIndex < 0)
                {
                    int32_t requestToken = -1;
                    uint8_t const tokenStatus = readRequestTopkToken(params, batch, s, k, requestToken);
                    if (tokenStatus == kHotReadOk && requestToken >= 0)
                    {
                        HiSparseResidentTokenAddress const residentAddress
                            = decodeResidentTokenAddress(params, row, requestToken);
                        if (residentAddress.status == kHotReadOk)
                        {
                            vVal = readResidentLatentValue(params, residentAddress.globalToken, dim);
                        }
                    }
                }
                else
                {
                    HiSparseKvarnHotAddress const address = decodeHisparseKvarnHotIndex(
                        hotIndex, params.strideFactor, params.layerIdx, params.hotCapacity, params.tokensPerBlock);
                    uint8_t const* record = params.hotPacked
                        + static_cast<int64_t>(params.layerIdx) * params.strideHotLayer
                        + static_cast<int64_t>(address.hotSlot) * params.strideHotSlot;
                    vVal = __bfloat162float(readHisparseKvarnK2v2BdrLatentValue(record, layout, address.tokenOffset, dim));
                }
                acc += scores[local] * vVal;
            }
            params.outAccum[static_cast<int64_t>(splitIdx) * params.strideOAccumSplit
                + static_cast<int64_t>(s) * params.strideOAccumSQ + static_cast<int64_t>(head) * params.strideOAccumHQ
                + dim] = acc;
        }
        if (threadIdx.x == 0)
        {
            params.lseAccum[static_cast<int64_t>(splitIdx) * params.strideLseAccumSplit
                + static_cast<int64_t>(s) * params.strideLseAccumSQ + head]
                = log2f(localDenom) + localMaxLog2;
        }
        __syncthreads();
    }
}

__global__ void sparseMlaDecodeKvarnHotSplitCombineKernel(SparseMlaDecodeKvarnHotParams params)
{
    int32_t const row = static_cast<int32_t>(blockIdx.x);
    int32_t const head = static_cast<int32_t>(blockIdx.y);
    int32_t const totalRows = params.b * params.sQ;
    if (row >= totalRows || head >= params.hQ)
    {
        return;
    }
    int32_t const batch = row / params.sQ;
    int32_t const s = row - batch * params.sQ;
    int32_t const status = params.rowHeadStatus == nullptr ? kHotReadOk
        : params.rowHeadStatus[static_cast<int64_t>(row) * params.hQ + head];
    int64_t const outBase = static_cast<int64_t>(batch) * params.strideOB
        + static_cast<int64_t>(s) * params.strideOSQ + static_cast<int64_t>(head) * params.strideOHQ;
    int64_t const lseOffset = static_cast<int64_t>(batch) * params.strideLseB
        + static_cast<int64_t>(s) * params.strideLseSQ + head;
    if (status != kHotReadOk)
    {
        for (int32_t dim = threadIdx.x; dim < params.dV; dim += blockDim.x)
        {
            writeBf16(params.out, outBase + dim, 0.0F);
        }
        if (threadIdx.x == 0)
        {
            params.lse[lseOffset] = kNegInf;
        }
        return;
    }

    __shared__ float globalLseLog2;
    int32_t const startSplit = params.numSplits[batch];
    int32_t const endSplit = params.numSplits[batch + 1];
    int32_t const numSplits = endSplit - startSplit;
    if (threadIdx.x == 0)
    {
        float maxLse = kNegInf;
        for (int32_t split = 0; split < numSplits; ++split)
        {
            float const lse = params.lseAccum[static_cast<int64_t>(startSplit + split) * params.strideLseAccumSplit
                + static_cast<int64_t>(s) * params.strideLseAccumSQ + head];
            maxLse = fmaxf(maxLse, lse);
        }
        float const sinkLog2 = params.attnSink == nullptr ? kNegInf : params.attnSink[head] * kLog2E;
        maxLse = fmaxf(maxLse, sinkLog2);
        float denom = 0.0F;
        if (maxLse != kNegInf)
        {
            if (sinkLog2 != kNegInf)
            {
                denom += exp2f(sinkLog2 - maxLse);
            }
            for (int32_t split = 0; split < numSplits; ++split)
            {
                float const lse = params.lseAccum[static_cast<int64_t>(startSplit + split) * params.strideLseAccumSplit
                    + static_cast<int64_t>(s) * params.strideLseAccumSQ + head];
                denom += lse == kNegInf ? 0.0F : exp2f(lse - maxLse);
            }
        }
        globalLseLog2 = denom > 0.0F ? log2f(denom) + maxLse : kNegInf;
        params.lse[lseOffset] = globalLseLog2 == kNegInf ? kNegInf : globalLseLog2 / kLog2E;
    }
    __syncthreads();

    for (int32_t dim = threadIdx.x; dim < params.dV; dim += blockDim.x)
    {
        float acc = 0.0F;
        for (int32_t split = 0; split < numSplits; ++split)
        {
            if (globalLseLog2 == kNegInf)
            {
                continue;
            }
            float const lse = params.lseAccum[static_cast<int64_t>(startSplit + split) * params.strideLseAccumSplit
                + static_cast<int64_t>(s) * params.strideLseAccumSQ + head];
            float const splitScale = lse == kNegInf ? 0.0F : exp2f(lse - globalLseLog2);
            if (splitScale == 0.0F)
            {
                continue;
            }
            acc += splitScale
                * params.outAccum[static_cast<int64_t>(startSplit + split) * params.strideOAccumSplit
                    + static_cast<int64_t>(s) * params.strideOAccumSQ
                    + static_cast<int64_t>(head) * params.strideOAccumHQ + dim];
        }
        writeBf16(params.out, outBase + dim, acc);
    }
}

} // namespace

void invokeSparseMlaDecodeKvarnHot(SparseMlaDecodeKvarnHotParams const& params, cudaStream_t stream)
{
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
    if (params.residentKvPool != nullptr
        && params.residentKvPoolDtype != kResidentKvPoolBf16
        && params.residentKvPoolDtype != kResidentKvPoolFp16)
    {
        throw std::runtime_error("sparse MLA KVarN-hot decode requires resident KV pool dtype bf16 or fp16");
    }
    if (params.topK > 2048)
    {
        throw std::runtime_error("sparse MLA KVarN-hot decode currently supports topk <= 2048");
    }
    size_t const sharedBytes = static_cast<size_t>(params.topK + kThreads) * sizeof(float);
    dim3 const grid(params.b * params.sQ, params.hQ, 1);
    sparseMlaDecodeKvarnHotKernel<<<grid, kThreads, sharedBytes, stream>>>(params);
    auto const err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error("sparse MLA KVarN-hot decode kernel launch failed");
    }
}


namespace
{
int32_t ceilDivHost(int32_t x, int32_t y)
{
    return (x + y - 1) / y;
}

int32_t currentSmCountKvarnHotSplit()
{
    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess)
    {
        throw std::runtime_error("cudaGetDevice failed in sparse MLA KVarN-hot split decode");
    }
    cudaDeviceProp prop{};
    status = cudaGetDeviceProperties(&prop, device);
    if (status != cudaSuccess)
    {
        throw std::runtime_error("cudaGetDeviceProperties failed in sparse MLA KVarN-hot split decode");
    }
    if (prop.major < 10)
    {
        throw std::runtime_error("sparse MLA KVarN-hot split decode requires Blackwell/SM100 or newer");
    }
    return static_cast<int32_t>(prop.multiProcessorCount);
}
} // namespace

int32_t getSparseMlaDecodeKvarnHotMetadataWidth()
{
    return static_cast<int32_t>(sizeof(DecodingSchedMeta) / sizeof(int32_t));
}

int32_t getSparseMlaDecodeKvarnHotNumSmPartsForShape(int32_t b, int32_t sQ, int32_t topK)
{
    if (b <= 0 || sQ <= 0 || topK <= 0)
    {
        throw std::runtime_error("batch, s_q, and topk must be positive for sparse MLA KVarN-hot split decode");
    }
    int32_t const topkBlocks = ceilDivHost(topK, kBlockSizeTopK);
    int32_t const schedulerOverhead = b == 1 ? kSingleBatchSchedulerOverhead
        : b >= 64                    ? kLongBatchSchedulerOverhead
        : b >= 32                    ? kMediumBatchSchedulerOverhead
                                      : kShortBatchSchedulerOverhead;
    int32_t const oneBlockParts = b * (topkBlocks + schedulerOverhead);
    int32_t const smFloor = topK >= 1024 ? 1 : std::max(currentSmCountKvarnHotSplit() / sQ, 1);
    return std::min(std::max(smFloor, oneBlockParts), kMaxNumSmParts);
}

int32_t getSparseMlaDecodeKvarnHotTotalSplits(int32_t b, int32_t numSmParts)
{
    return b + numSmParts;
}

void invokeSparseMlaDecodeKvarnHotSplit(SparseMlaDecodeKvarnHotParams const& params, cudaStream_t stream)
{
    if (params.hQ != kHeadQ || params.dQk != kDqk || params.dV != kDv || params.tokensPerBlock != kTokensPerBlock
        || params.kvLoraRank != kKvLoraRank || params.qkRopeHeadDim != kQkRopeHeadDim || params.kvarnBits != 2)
    {
        throw std::runtime_error(
            "sparse MLA KVarN-hot split decode requires production V3.2 shape h_q=128,d_qk=576,d_v=512,tpb=64,kvarn_k2v2");
    }
    if (params.b <= 0 || params.sQ <= 0 || params.topK <= 0 || params.numLayers <= 0 || params.hotCapacity <= 0
        || params.numSmParts <= 0)
    {
        throw std::runtime_error(
            "sparse MLA KVarN-hot split decode requires positive batch, s_q, topk, layers, hot capacity, and num_sm_parts");
    }
    if (params.residentKvPool != nullptr
        && params.residentKvPoolDtype != kResidentKvPoolBf16
        && params.residentKvPoolDtype != kResidentKvPoolFp16)
    {
        throw std::runtime_error("sparse MLA KVarN-hot split decode requires resident KV pool dtype bf16 or fp16");
    }
    if (params.topK > 2048)
    {
        throw std::runtime_error("sparse MLA KVarN-hot split decode currently supports topk <= 2048");
    }
    if (params.tileSchedulerMetadata == nullptr || params.numSplits == nullptr || params.lseAccum == nullptr
        || params.outAccum == nullptr || params.rowHeadStatus == nullptr)
    {
        throw std::runtime_error("sparse MLA KVarN-hot split decode requires scheduler, accum, and row-head status buffers");
    }

    GetDecodeSchedMetaParams schedulerParams{params.b, params.sQ, kBlockSizeTopK, kFixedOverheadBlocks, params.topK,
        0, nullptr, nullptr, nullptr, reinterpret_cast<DecodingSchedMeta*>(params.tileSchedulerMetadata), params.numSplits,
        params.numSmParts, stream};
    smxx::decode::run_get_decoding_sched_meta_kernel(schedulerParams);

    auto const statusBytes = static_cast<size_t>(params.b) * static_cast<size_t>(params.sQ) * static_cast<size_t>(params.hQ) * sizeof(int32_t);
    auto st = cudaMemsetAsync(params.rowHeadStatus, 0, statusBytes, stream);
    if (st != cudaSuccess)
    {
        throw std::runtime_error("sparse MLA KVarN-hot split decode status memset failed");
    }

    size_t const sharedBytes = static_cast<size_t>(kBlockSizeTopK + kThreads) * sizeof(float);
    dim3 const producerGrid(params.sQ, params.numSmParts, params.hQ);
    sparseMlaDecodeKvarnHotSplitProducerKernel<<<producerGrid, kThreads, sharedBytes, stream>>>(params);
    st = cudaGetLastError();
    if (st != cudaSuccess)
    {
        throw std::runtime_error(std::string("sparse MLA KVarN-hot split producer launch failed: ") + cudaGetErrorString(st));
    }

    dim3 const combineGrid(params.b * params.sQ, params.hQ, 1);
    sparseMlaDecodeKvarnHotSplitCombineKernel<<<combineGrid, kThreads, 0, stream>>>(params);
    st = cudaGetLastError();
    if (st != cudaSuccess)
    {
        throw std::runtime_error(std::string("sparse MLA KVarN-hot split combine launch failed: ") + cudaGetErrorString(st));
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END

/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#include "tensorrt_llm/kernels/flashMLA/sparse_mla_decode_kvarn_hot.h"

#include "tensorrt_llm/kernels/hisparseKvarnBdrRead.cuh"

#include <cuda_bf16.h>
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
        if (rowCode == kHotReadOk && k < rowTopK)
        {
            int32_t const hotIndex = params.indices[indexBase + k];
            HiSparseKvarnHotAddress const address = decodeHisparseKvarnHotIndex(
                hotIndex, params.strideFactor, params.layerIdx, params.hotCapacity, params.tokensPerBlock);
            if (address.status != kHotReadOk)
            {
                if (threadIdx.x == 0)
                {
                    rowCode = static_cast<int32_t>(address.status);
                }
            }
            else
            {
                uint8_t const* record = params.hotPacked + static_cast<int64_t>(params.layerIdx) * params.strideHotLayer
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
        float const score = blockReduceSum(scorePart, reduce) * params.smScale;
        if (threadIdx.x == 0)
        {
            scores[k] = (rowCode == kHotReadOk && k < rowTopK) ? score : kNegInf;
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

    for (int32_t dim = threadIdx.x; dim < params.dV; dim += blockDim.x)
    {
        float acc = 0.0F;
        for (int32_t k = 0; k < rowTopK; ++k)
        {
            int32_t const hotIndex = params.indices[indexBase + k];
            HiSparseKvarnHotAddress const address = decodeHisparseKvarnHotIndex(
                hotIndex, params.strideFactor, params.layerIdx, params.hotCapacity, params.tokensPerBlock);
            uint8_t const* record = params.hotPacked + static_cast<int64_t>(params.layerIdx) * params.strideHotLayer
                + static_cast<int64_t>(address.hotSlot) * params.strideHotSlot;
            float const vVal = __bfloat162float(
                readHisparseKvarnK2v2BdrLatentValue(record, layout, address.tokenOffset, dim));
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

} // namespace kernels

TRTLLM_NAMESPACE_END

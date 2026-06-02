/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorrt_llm/kernels/flashMLA/sparse_mla_decode_nvfp4.h"

#include "combine.h"
#include "get_decoding_sched_meta.h"
#include "kernel.h"
#include "params.h"

#include <algorithm>
#include <cutlass/bfloat16.h>
#include <stdexcept>

namespace tensorrt_llm::kernels
{
namespace
{
constexpr float kLog2E = 1.4426950408889634F;
constexpr int32_t kHeadQ = 128;
constexpr int32_t kHeadSplit = 64;
constexpr int32_t kDqk = 576;
constexpr int32_t kDv = 512;
constexpr int32_t kPageBlockSize = 64;
constexpr int32_t kKvBytesPerToken = 288;
constexpr int32_t kKvScaleBytesPerToken = 36;
constexpr int32_t kFixedOverheadBlocks = 5;
constexpr int32_t kBlockSizeTopK = 64;
constexpr int32_t kMaxProducerBlocksPerSplit = 3;
constexpr int32_t kMaxNvfp4NumSmParts = 4096;
constexpr int32_t kShortBatchSchedulerOverhead = 15;
constexpr int32_t kLongBatchSchedulerOverhead = 8;

int32_t ceilDiv(int32_t x, int32_t y)
{
    return (x + y - 1) / y;
}

int32_t currentSmCount()
{
    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess)
    {
        throw std::runtime_error("cudaGetDevice failed in sparse MLA NVFP4 decode");
    }
    cudaDeviceProp prop{};
    status = cudaGetDeviceProperties(&prop, device);
    if (status != cudaSuccess)
    {
        throw std::runtime_error("cudaGetDeviceProperties failed in sparse MLA NVFP4 decode");
    }
    if (prop.major < 10)
    {
        throw std::runtime_error("sparse MLA NVFP4 decode requires Blackwell/SM100 or newer");
    }
    return static_cast<int32_t>(prop.multiProcessorCount);
}
} // namespace

int32_t getSparseMlaDecodeNvfp4MetadataWidth()
{
    return static_cast<int32_t>(sizeof(DecodingSchedMeta) / sizeof(int32_t));
}

int32_t getSparseMlaDecodeNvfp4BlockSizeTopK()
{
    return kBlockSizeTopK;
}

int32_t getSparseMlaDecodeNvfp4FixedOverheadBlocks()
{
    return kFixedOverheadBlocks;
}

int32_t getSparseMlaDecodeNvfp4NumSmParts(int32_t sQ)
{
    if (sQ <= 0)
    {
        throw std::runtime_error("s_q must be positive for sparse MLA NVFP4 decode");
    }
    return std::max(currentSmCount() / sQ, 1);
}

int32_t getSparseMlaDecodeNvfp4NumSmPartsForShape(int32_t b, int32_t sQ, int32_t topK)
{
    if (b <= 0 || sQ <= 0 || topK <= 0)
    {
        throw std::runtime_error("batch, s_q, and topk must be positive for sparse MLA NVFP4 decode");
    }

    int32_t const topkBlocks = ceilDiv(topK, kBlockSizeTopK);
    int32_t const workBlocks = b * (topkBlocks + kFixedOverheadBlocks);
    int32_t const boundedParts = ceilDiv(workBlocks, kMaxProducerBlocksPerSplit);

    // FlashMLA's API layer uses max(num_sms / s_q, 1). That can group
    // multiple 64-token top-k blocks into one producer split for V3.2 NVFP4,
    // which is non-finite for random mixed packed-FP4 payloads at B>=8 on the
    // current reference kernel. Keep the imported producer logic intact and
    // constrain scheduling in the op-trt adapter until the multi-block producer
    // path is fixed and revalidated. Empirical B200 sweeps for topk=1024 show
    // +15 is fastest/equivalent for B8/B16/B32, while +8 is fastest/equivalent
    // for B64/B128 because it reduces producer-part and combine pressure.
    int32_t const schedulerOverhead = b >= 64 ? kLongBatchSchedulerOverhead : kShortBatchSchedulerOverhead;
    int32_t const oneBlockParts = b * (topkBlocks + schedulerOverhead);
    int32_t const smFloor = getSparseMlaDecodeNvfp4NumSmParts(sQ);
    return std::min(std::max({smFloor, boundedParts, oneBlockParts}), kMaxNvfp4NumSmParts);
}

int32_t getSparseMlaDecodeNvfp4TotalSplits(int32_t b, int32_t numSmParts)
{
    return b + numSmParts;
}

int32_t getSparseMlaDecodeNvfp4MaxSplitsPerRequest(int32_t topK)
{
    if (topK <= 0)
    {
        throw std::runtime_error("topk must be positive for sparse MLA NVFP4 decode");
    }
    return std::max(ceilDiv(topK, kBlockSizeTopK), 1);
}

void invokeSparseMlaDecodeNvfp4(SparseMlaDecodeNvfp4Params const& params, cudaStream_t stream)
{
    if (params.hQ != kHeadQ || params.hKv != 1 || params.dQk != kDqk || params.dV != kDv)
    {
        throw std::runtime_error("sparse MLA NVFP4 decode currently supports only V3.2 shape h_q=128,h_kv=1,d_qk=576,d_v=512");
    }
    if (params.pageBlockSize != kPageBlockSize || params.strideKvRow != kKvBytesPerToken
        || params.strideKvScalesRow != kKvScaleBytesPerToken)
    {
        throw std::runtime_error("sparse MLA NVFP4 decode requires contiguous V3.2 NVFP4 token layout: kv row 288B, scale row 36B, page size 64");
    }

    SparseAttnDecodeParams decodeParams{
        params.b,
        params.sQ,
        params.hQ,
        params.hKv,
        params.dQk,
        params.dV,
        params.smScale,
        params.smScale * kLog2E,
        params.numBlocks,
        params.pageBlockSize,
        params.topK,
        ModelType::V32,

        reinterpret_cast<cutlass::bfloat16_t*>(params.q),
        reinterpret_cast<cutlass::bfloat16_t*>(params.kv),
        params.indices,
        params.topkLength,
        params.attnSink,
        params.lse,
        reinterpret_cast<cutlass::bfloat16_t*>(params.out),

        0,
        0,
        0,
        nullptr,
        nullptr,
        nullptr,

        params.kvScales,
        params.strideKvScalesBlock,
        params.strideKvScalesRow,

        params.strideQB,
        params.strideQSQ,
        params.strideQHQ,
        params.strideKvBlock,
        params.strideKvRow,
        params.strideIndicesB,
        params.strideIndicesSQ,
        params.strideLseB,
        params.strideLseSQ,
        params.strideOB,
        params.strideOSQ,
        params.strideOHQ,
        0,
        0,
        0,
        0,
        stream};

    if (params.computeSchedulerMetadata)
    {
        GetDecodeSchedMetaParams schedulerParams{
            params.b,
            params.sQ,
            kBlockSizeTopK,
            kFixedOverheadBlocks,
            params.topK,
            0,
            params.topkLength,
            nullptr,
            nullptr,
            reinterpret_cast<DecodingSchedMeta*>(params.tileSchedulerMetadata),
            params.numSplits,
            params.numSmParts,
            stream};
        smxx::decode::run_get_decoding_sched_meta_kernel(schedulerParams);
    }

    decodeParams.tile_scheduler_metadata_ptr = reinterpret_cast<DecodingSchedMeta*>(params.tileSchedulerMetadata);
    decodeParams.num_splits_ptr = params.numSplits;
    decodeParams.num_sm_parts = params.numSmParts;
    decodeParams.lse_accum = params.lseAccum;
    decodeParams.o_accum = params.outAccum;
    decodeParams.stride_lse_accum_split = params.strideLseAccumSplit;
    decodeParams.stride_lse_accum_s_q = params.strideLseAccumSQ;
    decodeParams.stride_o_accum_split = params.strideOAccumSplit;
    decodeParams.stride_o_accum_s_q = params.strideOAccumSQ;
    decodeParams.stride_o_accum_h_q = params.strideOAccumHQ;

    for (int startHeadIdx = 0; startHeadIdx < kHeadQ; startHeadIdx += kHeadSplit)
    {
        SparseAttnDecodeParams curParams = decodeParams;
        curParams.q += startHeadIdx * params.strideQHQ;
        if (curParams.attn_sink != nullptr)
        {
            curParams.attn_sink += startHeadIdx;
        }
        curParams.lse += startHeadIdx;
        curParams.out += startHeadIdx * params.strideOHQ;
        curParams.lse_accum += startHeadIdx;
        curParams.o_accum += startHeadIdx * params.strideOAccumHQ;
        curParams.h_q = kHeadSplit;
        sm100::decode::head64_nvfp4::run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::V32>(curParams);
    }

    CombineParams combineParams{
        params.b,
        params.sQ,
        params.hQ,
        params.dV,
        params.lse,
        params.out,
        params.strideLseB,
        params.strideLseSQ,
        params.strideOB,
        params.strideOSQ,
        params.strideOHQ,
        params.lseAccum,
        params.outAccum,
        params.strideLseAccumSplit,
        params.strideLseAccumSQ,
        params.strideOAccumSplit,
        params.strideOAccumSQ,
        params.strideOAccumHQ,
        reinterpret_cast<DecodingSchedMeta*>(params.tileSchedulerMetadata),
        params.numSplits,
        // B1/B4 use FlashMLA's original combine bucket, which is the measured
        // faster finite reference path. Larger batches use the per-request split
        // bound because the adapter deliberately expands producer SM parts to
        // keep V3.2 NVFP4 top-k blocks one-per-split.
        params.b <= 4 ? params.numSmParts : std::min(params.numSmParts, getSparseMlaDecodeNvfp4MaxSplitsPerRequest(params.topK)),
        params.attnSink,
        stream};
    smxx::decode::run_flash_mla_combine_kernel<cutlass::bfloat16_t>(combineParams);
}

} // namespace tensorrt_llm::kernels

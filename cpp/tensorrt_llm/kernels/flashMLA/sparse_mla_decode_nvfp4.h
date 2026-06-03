/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

#include "tensorrt_llm/common/config.h"

TRTLLM_NAMESPACE_BEGIN
namespace kernels
{

struct SparseMlaDecodeNvfp4Params
{
    void* q;
    uint8_t* kv;
    uint8_t* kvScales;
    int32_t* indices;
    int32_t* topkLength;
    float* attnSink;
    float* lse;
    void* out;
    int32_t* tileSchedulerMetadata;
    int32_t* numSplits;
    float* lseAccum;
    float* outAccum;

    int32_t b;
    int32_t sQ;
    int32_t hQ;
    int32_t hKv;
    int32_t dQk;
    int32_t dV;
    int32_t numBlocks;
    int32_t pageBlockSize;
    int32_t topK;
    float smScale;

    int32_t strideQB;
    int32_t strideQSQ;
    int32_t strideQHQ;
    int32_t strideKvBlock;
    int32_t strideKvRow;
    int32_t strideKvScalesBlock;
    int32_t strideKvScalesRow;
    int32_t strideIndicesB;
    int32_t strideIndicesSQ;
    int32_t strideLseB;
    int32_t strideLseSQ;
    int32_t strideOB;
    int32_t strideOSQ;
    int32_t strideOHQ;
    int32_t strideLseAccumSplit;
    int32_t strideLseAccumSQ;
    int32_t strideOAccumSplit;
    int32_t strideOAccumSQ;
    int32_t strideOAccumHQ;
    int32_t numSmParts;
    bool computeSchedulerMetadata;
};

int32_t getSparseMlaDecodeNvfp4MetadataWidth();
int32_t getSparseMlaDecodeNvfp4BlockSizeTopK();
int32_t getSparseMlaDecodeNvfp4FixedOverheadBlocks();
int32_t getSparseMlaDecodeNvfp4NumSmParts(int32_t sQ);
int32_t getSparseMlaDecodeNvfp4NumSmPartsForShape(int32_t b, int32_t sQ, int32_t topK);
int32_t getSparseMlaDecodeNvfp4TotalSplits(int32_t b, int32_t numSmParts);

void invokeSparseMlaDecodeNvfp4(SparseMlaDecodeNvfp4Params const& params, cudaStream_t stream);

} // namespace kernels
TRTLLM_NAMESPACE_END

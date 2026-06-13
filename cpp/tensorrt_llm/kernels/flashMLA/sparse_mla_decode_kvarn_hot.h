/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cuda_runtime_api.h>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

struct SparseMlaDecodeKvarnHotParams
{
    void* q;
    uint8_t* hotPacked;
    int32_t* indices;
    uint8_t* rowStatus;
    int32_t* topkLength;
    int64_t* residentKvLens;
    int64_t* residentReqIdx;
    int64_t* residentRequestIds;
    int32_t* residentBlockTable;
    int32_t* residentTailBlockPos;
    int32_t* residentTailTokenCount;
    bool* residentTailValid;
    float* attnSink;
    float* lse;
    void* out;

    int32_t b;
    int32_t sQ;
    int32_t hQ;
    int32_t dQk;
    int32_t dV;
    int32_t numLayers;
    int32_t hotCapacity;
    int32_t topK;
    int32_t topkLengthSize;
    int32_t residentRows;
    int32_t residentSinkTokens;
    int32_t residentSinkBlocks;
    int32_t layerIdx;
    int32_t tokensPerBlock;
    int32_t strideFactor;
    int32_t kvarnBits;
    int32_t kvLoraRank;
    int32_t qkRopeHeadDim;
    float smScale;

    int64_t strideQB;
    int64_t strideQSQ;
    int64_t strideQHQ;
    int64_t strideHotLayer;
    int64_t strideHotSlot;
    int64_t strideHotRecord;
    int64_t strideIndicesB;
    int64_t strideIndicesSQ;
    int64_t strideResidentBlockTableB;
    int64_t strideResidentBlockTableBlock;
    int64_t strideLseB;
    int64_t strideLseSQ;
    int64_t strideOB;
    int64_t strideOSQ;
    int64_t strideOHQ;
};

void invokeSparseMlaDecodeKvarnHot(SparseMlaDecodeKvarnHotParams const& params, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END

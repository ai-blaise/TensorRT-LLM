/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

void invokeHisparseReadKvarnHotBdr(uint8_t const* hotPacked, int32_t const* hotIndices,
    int32_t const* topkLength, uint8_t const* inputRowStatus, __nv_bfloat16* latentOut, uint8_t* outputRowStatus,
    int32_t numRows, int32_t indexTopK, int32_t numLayers, int32_t hotCapacity, int64_t hotLayerStride,
    int64_t hotSlotStride, int64_t hotRecordStride, int32_t layerIdx, int32_t tokensPerBlock, int32_t kvarnBits,
    int32_t kvLoraRank, int32_t qkRopeHeadDim, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END

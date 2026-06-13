/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

void invokeHisparseTopkToBlockPositions(int32_t const* topkIndices, int32_t* blockPositions, int32_t* blockCounts,
    uint8_t* overflowFlags, int32_t numRows, int32_t indexTopK, int32_t tokensPerBlock, int32_t maxBlocksPerRow,
    int32_t hashCapacity, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END

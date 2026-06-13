/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>

namespace th = torch;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace
{

void checkByteTensor(th::Tensor const& tensor, char const* name)
{
    TORCH_CHECK(tensor.scalar_type() == torch::kUInt8,
        name, " must be uint8 packed KVarN storage; got scalar_type=", tensor.scalar_type());
}

void checkLayeredPackedTensor(th::Tensor const& tensor, char const* name, int64_t packedBytesPerBlock)
{
    TORCH_CHECK(tensor.dim() == 3, name, " must have shape [num_layers, num_slots, packed_bytes]");
    TORCH_CHECK(tensor.size(2) >= packedBytesPerBlock,
        name, " last dimension must be at least packed_bytes_per_block=", packedBytesPerBlock,
        ", got ", tensor.size(2));
    TORCH_CHECK(tensor.stride(2) == 1, name, " records must be byte-contiguous in the last dimension");
}

void checkSlots(th::Tensor const& slots, char const* name)
{
    TORCH_CHECK(slots.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(slots.scalar_type() == torch::kInt64 || slots.scalar_type() == torch::kInt32,
        name, " must be int64 or int32");
    TORCH_CHECK(slots.dim() == 1, name, " must be a 1D tensor");
    TORCH_CHECK(slots.is_contiguous(), name, " must be contiguous");
}

int64_t readSlot(th::Tensor const& slots, int64_t i)
{
    if (slots.scalar_type() == torch::kInt64)
    {
        return slots.data_ptr<int64_t>()[i];
    }
    return static_cast<int64_t>(slots.data_ptr<int32_t>()[i]);
}

void checkedMemcpyAsync(void* dst, void const* src, size_t bytes, cudaStream_t stream)
{
    auto const err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
    TORCH_CHECK(err == cudaSuccess, "hisparse_swap_in_packed_kvarn cudaMemcpyAsync failed: ", cudaGetErrorString(err));
}

} // namespace

void hisparseSwapInPackedKvarn(th::Tensor const& hostPacked, th::Tensor const& hotPacked,
    th::Tensor const& hostSlots, th::Tensor const& hotSlots, int64_t layerIdx, int64_t packedBytesPerBlock)
{
    TORCH_CHECK(packedBytesPerBlock > 0
            && packedBytesPerBlock <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
        "packed_bytes_per_block must be positive int32-sized bytes, got ", packedBytesPerBlock);
    checkByteTensor(hostPacked, "host_packed");
    checkByteTensor(hotPacked, "hot_packed");
    TORCH_CHECK(hostPacked.device().is_cpu(), "host_packed must be a CPU host-pinned packed KVarN tensor");
    TORCH_CHECK(hotPacked.is_cuda(), "hot_packed must be a CUDA packed KVarN tensor");
    TORCH_CHECK(hostPacked.is_pinned(),
        "host_packed must be pinned CPU memory so host-to-hot copies are stream-ordered and non-blocking");
    checkLayeredPackedTensor(hostPacked, "host_packed", packedBytesPerBlock);
    checkLayeredPackedTensor(hotPacked, "hot_packed", packedBytesPerBlock);
    TORCH_CHECK(hostPacked.size(0) == hotPacked.size(0),
        "host_packed and hot_packed must have the same number of layers");
    TORCH_CHECK(layerIdx >= 0 && layerIdx < hostPacked.size(0),
        "layer_idx out of range: ", layerIdx, " for num_layers=", hostPacked.size(0));

    checkSlots(hostSlots, "host_slots");
    checkSlots(hotSlots, "hot_slots");
    TORCH_CHECK(hostSlots.size(0) == hotSlots.size(0),
        "host_slots and hot_slots must have the same length");

    int64_t const numCopies = hostSlots.size(0);
    if (numCopies == 0)
    {
        return;
    }

    c10::cuda::CUDAGuard guard(hotPacked.device());
    auto stream = at::cuda::getCurrentCUDAStream(hotPacked.get_device()).stream();
    auto const* hostBase = hostPacked.data_ptr<uint8_t>();
    auto* hotBase = hotPacked.data_ptr<uint8_t>();

    int64_t const hostLayerStride = hostPacked.stride(0);
    int64_t const hostSlotStride = hostPacked.stride(1);
    int64_t const hotLayerStride = hotPacked.stride(0);
    int64_t const hotSlotStride = hotPacked.stride(1);
    int64_t const hostCapacity = hostPacked.size(1);
    int64_t const hotCapacity = hotPacked.size(1);
    bool const compactSlotRows = hostSlotStride == packedBytesPerBlock && hotSlotStride == packedBytesPerBlock;

    for (int64_t i = 0; i < numCopies;)
    {
        int64_t const hostSlot = readSlot(hostSlots, i);
        int64_t const hotSlot = readSlot(hotSlots, i);
        TORCH_CHECK(hostSlot >= 0 && hostSlot < hostCapacity,
            "host slot out of range at copy ", i, ": ", hostSlot, " for capacity ", hostCapacity);
        TORCH_CHECK(hotSlot >= 0 && hotSlot < hotCapacity,
            "hot slot out of range at copy ", i, ": ", hotSlot, " for capacity ", hotCapacity);

        int64_t runBlocks = 1;
        if (compactSlotRows)
        {
            while (i + runBlocks < numCopies)
            {
                int64_t const nextHost = readSlot(hostSlots, i + runBlocks);
                int64_t const nextHot = readSlot(hotSlots, i + runBlocks);
                if (nextHost != hostSlot + runBlocks || nextHot != hotSlot + runBlocks)
                {
                    break;
                }
                TORCH_CHECK(nextHost >= 0 && nextHost < hostCapacity,
                    "host slot out of range at copy ", i + runBlocks, ": ", nextHost,
                    " for capacity ", hostCapacity);
                TORCH_CHECK(nextHot >= 0 && nextHot < hotCapacity,
                    "hot slot out of range at copy ", i + runBlocks, ": ", nextHot,
                    " for capacity ", hotCapacity);
                ++runBlocks;
            }
        }

        auto const* src = hostBase + layerIdx * hostLayerStride + hostSlot * hostSlotStride;
        auto* dst = hotBase + layerIdx * hotLayerStride + hotSlot * hotSlotStride;
        checkedMemcpyAsync(dst, src, static_cast<size_t>(runBlocks * packedBytesPerBlock), stream);
        i += runBlocks;
    }
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "hisparse_swap_in_packed_kvarn(Tensor host_packed, Tensor hot_packed, Tensor host_slots, Tensor hot_slots, "
        "int layer_idx, int packed_bytes_per_block) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("hisparse_swap_in_packed_kvarn", &tensorrt_llm::torch_ext::hisparseSwapInPackedKvarn);
}

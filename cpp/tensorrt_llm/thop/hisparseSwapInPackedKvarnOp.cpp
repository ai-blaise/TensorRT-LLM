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

enum HiSparseCopyStatus : uint8_t
{
    kCopyOk = 0,
    kCopyUpstreamInvalid = 1,
    kCopyBadCount = 2,
    kCopyInvalidRow = 3,
    kCopyHostSlotOutOfRange = 4,
    kCopyHotSlotOutOfRange = 5,
};

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

__global__ void hisparseSubmitPackedKvarnCopyScheduleKernel(uint8_t const* __restrict__ hostBase,
    uint8_t* __restrict__ hotBase, int64_t const* __restrict__ compactHostSlots,
    int64_t const* __restrict__ compactHotSlots, int32_t const* __restrict__ compactRowIds,
    int32_t const* __restrict__ copyCount, uint8_t* __restrict__ rowStatus, int32_t scheduleCapacity,
    int32_t numRows, int64_t hostLayerStride, int64_t hostSlotStride, int64_t hotLayerStride, int64_t hotSlotStride,
    int64_t hostCapacity, int64_t hotCapacity, int32_t layerIdx, int32_t packedBytesPerBlock)
{
    int32_t const count = copyCount[0];
    if (count < 0 || count > scheduleCapacity)
    {
        for (int32_t row = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x); row < numRows;
             row += static_cast<int32_t>(gridDim.x * blockDim.x))
        {
            rowStatus[row] = kCopyBadCount;
        }
        return;
    }

    int32_t const copyIdx = static_cast<int32_t>(blockIdx.x);
    if (copyIdx >= count)
    {
        return;
    }

    int32_t const row = compactRowIds[copyIdx];
    if (row < 0 || row >= numRows)
    {
        // A compact schedule with an impossible row id has no safe owner row to
        // mark. Fail the whole row-status vector so the commit stage cannot
        // publish hot metadata for copies that may not have happened.
        for (int32_t statusRow = threadIdx.x; statusRow < numRows; statusRow += blockDim.x)
        {
            rowStatus[statusRow] = kCopyInvalidRow;
        }
        return;
    }
    if (rowStatus[row] != kCopyOk)
    {
        return;
    }

    int64_t const hostSlot = compactHostSlots[copyIdx];
    int64_t const hotSlot = compactHotSlots[copyIdx];
    if (hostSlot < 0 || hostSlot >= hostCapacity)
    {
        rowStatus[row] = kCopyHostSlotOutOfRange;
        return;
    }
    if (hotSlot < 0 || hotSlot >= hotCapacity)
    {
        rowStatus[row] = kCopyHotSlotOutOfRange;
        return;
    }

    uint8_t const* src = hostBase + static_cast<int64_t>(layerIdx) * hostLayerStride + hostSlot * hostSlotStride;
    uint8_t* dst = hotBase + static_cast<int64_t>(layerIdx) * hotLayerStride + hotSlot * hotSlotStride;
    for (int32_t byte = threadIdx.x; byte < packedBytesPerBlock; byte += blockDim.x)
    {
        dst[byte] = src[byte];
    }
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

th::Tensor hisparseSubmitPackedKvarnCopySchedule(th::Tensor const& hostPacked, th::Tensor const& hotPacked,
    th::Tensor const& compactHostSlots, th::Tensor const& compactHotSlots, th::Tensor const& compactRowIds,
    th::Tensor const& copyCount, th::Tensor const& compactRowStatus, int64_t layerIdx, int64_t packedBytesPerBlock)
{
    TORCH_CHECK(packedBytesPerBlock > 0
            && packedBytesPerBlock <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
        "packed_bytes_per_block must be positive int32-sized bytes, got ", packedBytesPerBlock);
    checkByteTensor(hostPacked, "host_packed");
    checkByteTensor(hotPacked, "hot_packed");
    TORCH_CHECK(hostPacked.device().is_cpu(), "host_packed must be CPU mapped-pinned packed KVarN storage");
    TORCH_CHECK(hotPacked.is_cuda(), "hot_packed must be a CUDA packed KVarN tensor");
    TORCH_CHECK(hostPacked.is_pinned(),
        "host_packed must be pinned CPU memory and device-addressable for schedule-driven HiSparse copy");
    checkLayeredPackedTensor(hostPacked, "host_packed", packedBytesPerBlock);
    checkLayeredPackedTensor(hotPacked, "hot_packed", packedBytesPerBlock);
    TORCH_CHECK(hostPacked.size(0) == hotPacked.size(0),
        "host_packed and hot_packed must have the same number of layers");
    TORCH_CHECK(layerIdx >= 0 && layerIdx < hostPacked.size(0),
        "layer_idx out of range: ", layerIdx, " for num_layers=", hostPacked.size(0));
    TORCH_CHECK(compactHostSlots.is_cuda(), "compact_host_slots must be a CUDA tensor");
    TORCH_CHECK(compactHotSlots.is_cuda(), "compact_hot_slots must be a CUDA tensor");
    TORCH_CHECK(compactRowIds.is_cuda(), "compact_row_ids must be a CUDA tensor");
    TORCH_CHECK(copyCount.is_cuda(), "copy_count must be a CUDA tensor");
    TORCH_CHECK(compactRowStatus.is_cuda(), "compact_row_status must be a CUDA tensor");
    TORCH_CHECK(compactHostSlots.scalar_type() == torch::kInt64, "compact_host_slots must be int64");
    TORCH_CHECK(compactHotSlots.scalar_type() == torch::kInt64, "compact_hot_slots must be int64");
    TORCH_CHECK(compactRowIds.scalar_type() == torch::kInt32, "compact_row_ids must be int32");
    TORCH_CHECK(copyCount.scalar_type() == torch::kInt32, "copy_count must be int32");
    TORCH_CHECK(compactRowStatus.scalar_type() == torch::kUInt8, "compact_row_status must be uint8");
    TORCH_CHECK(compactHostSlots.dim() == 1, "compact_host_slots must have shape [schedule_capacity]");
    TORCH_CHECK(compactHotSlots.dim() == 1, "compact_hot_slots must have shape [schedule_capacity]");
    TORCH_CHECK(compactRowIds.dim() == 1, "compact_row_ids must have shape [schedule_capacity]");
    TORCH_CHECK(copyCount.dim() == 1 && copyCount.size(0) == 1, "copy_count must have shape [1]");
    TORCH_CHECK(compactRowStatus.dim() == 1, "compact_row_status must have shape [rows]");
    TORCH_CHECK(compactHotSlots.size(0) == compactHostSlots.size(0),
        "compact_hot_slots shape must match compact_host_slots");
    TORCH_CHECK(compactRowIds.size(0) == compactHostSlots.size(0),
        "compact_row_ids shape must match compact_host_slots");

    c10::cuda::CUDAGuard guard(hotPacked.device());
    int32_t const device = hotPacked.get_device();
    TORCH_CHECK(compactHostSlots.get_device() == device,
        "compact_host_slots must be on the same CUDA device as hot_packed");
    TORCH_CHECK(compactHotSlots.get_device() == device,
        "compact_hot_slots must be on the same CUDA device as hot_packed");
    TORCH_CHECK(compactRowIds.get_device() == device, "compact_row_ids must be on the same CUDA device as hot_packed");
    TORCH_CHECK(copyCount.get_device() == device, "copy_count must be on the same CUDA device as hot_packed");
    TORCH_CHECK(compactRowStatus.get_device() == device,
        "compact_row_status must be on the same CUDA device as hot_packed");

    int canUseHostPointer = 0;
    auto attrErr = cudaDeviceGetAttribute(&canUseHostPointer, cudaDevAttrCanUseHostPointerForRegisteredMem, device);
    TORCH_CHECK(attrErr == cudaSuccess,
        "cudaDeviceGetAttribute(cudaDevAttrCanUseHostPointerForRegisteredMem) failed: ", cudaGetErrorString(attrErr));
    TORCH_CHECK(canUseHostPointer != 0,
        "HiSparse schedule-driven copy requires a device that can access registered host memory directly");

    void* mappedHostPtr = nullptr;
    auto mapErr = cudaHostGetDevicePointer(&mappedHostPtr, hostPacked.data_ptr<uint8_t>(), 0);
    if (mapErr != cudaSuccess)
    {
        // On devices that report cudaDevAttrCanUseHostPointerForRegisteredMem,
        // the original registered host pointer is also device-addressable. Clear
        // the failed runtime status before launching the mapped-host copy kernel.
        cudaGetLastError();
        mappedHostPtr = hostPacked.data_ptr<uint8_t>();
    }

    auto hostSlots = compactHostSlots.contiguous();
    auto hotSlots = compactHotSlots.contiguous();
    auto rowIds = compactRowIds.contiguous();
    auto count = copyCount.contiguous();
    auto rowStatus = compactRowStatus.clone();

    int64_t const capacity = hostSlots.size(0);
    int64_t const rows = compactRowStatus.size(0);
    if (rows == 0)
    {
        return rowStatus;
    }
    TORCH_CHECK(capacity <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
        "schedule capacity must be int32-sized, got ", capacity);
    TORCH_CHECK(rows <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()), "rows must be int32-sized, got ",
        rows);

    constexpr int32_t kThreads = 256;
    int32_t const grid = static_cast<int32_t>(capacity > 0 ? capacity : 1);
    auto stream = at::cuda::getCurrentCUDAStream(device).stream();
    hisparseSubmitPackedKvarnCopyScheduleKernel<<<grid, kThreads, 0, stream>>>(
        static_cast<uint8_t const*>(mappedHostPtr), hotPacked.data_ptr<uint8_t>(), hostSlots.data_ptr<int64_t>(),
        hotSlots.data_ptr<int64_t>(), rowIds.data_ptr<int32_t>(), count.data_ptr<int32_t>(),
        rowStatus.data_ptr<uint8_t>(), static_cast<int32_t>(capacity), static_cast<int32_t>(rows),
        hostPacked.stride(0), hostPacked.stride(1), hotPacked.stride(0), hotPacked.stride(1), hostPacked.size(1),
        hotPacked.size(1), static_cast<int32_t>(layerIdx), static_cast<int32_t>(packedBytesPerBlock));
    auto const kernelErr = cudaGetLastError();
    TORCH_CHECK(kernelErr == cudaSuccess,
        "hisparse_submit_packed_kvarn_copy_schedule kernel launch failed: ", cudaGetErrorString(kernelErr));
    return rowStatus;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "hisparse_swap_in_packed_kvarn(Tensor host_packed, Tensor hot_packed, Tensor host_slots, Tensor hot_slots, "
        "int layer_idx, int packed_bytes_per_block) -> ()");
    m.def(
        "hisparse_submit_packed_kvarn_copy_schedule(Tensor host_packed, Tensor hot_packed, "
        "Tensor compact_host_slots, Tensor compact_hot_slots, Tensor compact_row_ids, Tensor copy_count, "
        "Tensor compact_row_status, int layer_idx, int packed_bytes_per_block) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("hisparse_swap_in_packed_kvarn", &tensorrt_llm::torch_ext::hisparseSwapInPackedKvarn);
    m.impl("hisparse_submit_packed_kvarn_copy_schedule",
        &tensorrt_llm::torch_ext::hisparseSubmitPackedKvarnCopySchedule);
}

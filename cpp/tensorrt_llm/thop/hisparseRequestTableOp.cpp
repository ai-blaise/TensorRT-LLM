/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#include "tensorrt_llm/common/opUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace th = torch;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace
{

void checkCpuTensor(th::Tensor const& tensor, char const* name)
{
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
}

void checkPinnedCpuTensor(th::Tensor const& tensor, char const* name)
{
    checkCpuTensor(tensor, name);
    TORCH_CHECK(tensor.is_pinned(), name, " must be pinned CPU memory for stream-ordered host-to-device publish");
}

void checkCudaTensor(th::Tensor const& tensor, char const* name)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

void checkInt64Tensor(th::Tensor const& tensor, char const* name)
{
    TORCH_CHECK(tensor.scalar_type() == torch::kInt64, name, " must be int64");
}

void checkBoolTensor(th::Tensor const& tensor, char const* name)
{
    TORCH_CHECK(tensor.scalar_type() == torch::kBool, name, " must be bool");
}

void checkSlots(th::Tensor const& slots)
{
    checkCpuTensor(slots, "table_slots");
    TORCH_CHECK(slots.scalar_type() == torch::kInt64 || slots.scalar_type() == torch::kInt32,
        "table_slots must be int64 or int32");
    TORCH_CHECK(slots.dim() == 1, "table_slots must be a 1D tensor");
    TORCH_CHECK(slots.is_contiguous(), "table_slots must be contiguous");
}

int64_t readSlot(th::Tensor const& slots, int64_t i)
{
    if (slots.scalar_type() == torch::kInt64)
    {
        return slots.data_ptr<int64_t>()[i];
    }
    return static_cast<int64_t>(slots.data_ptr<int32_t>()[i]);
}

void checkedH2DCopy(void* dst, void const* src, size_t bytes, cudaStream_t stream, char const* what)
{
    auto const err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
    TORCH_CHECK(err == cudaSuccess, "hisparse_publish_request_table_slots ", what,
        " cudaMemcpyAsync failed: ", cudaGetErrorString(err));
}

void checkRequestTableShapes(th::Tensor const& requestIdsHost, th::Tensor const& requestIdsDevice,
    th::Tensor const& requestBlockHostSlotsHost, th::Tensor const& requestBlockHostSlotsDevice,
    th::Tensor const& requestBlockCommitGenHost, th::Tensor const& requestBlockCommitGenDevice,
    th::Tensor const& requestAdmittedHost, th::Tensor const& requestAdmittedDevice)
{
    checkPinnedCpuTensor(requestIdsHost, "request_ids_host");
    checkPinnedCpuTensor(requestBlockHostSlotsHost, "request_block_host_slots_host");
    checkPinnedCpuTensor(requestBlockCommitGenHost, "request_block_commit_gen_host");
    checkPinnedCpuTensor(requestAdmittedHost, "request_admitted_host");
    checkCudaTensor(requestIdsDevice, "request_ids_device");
    checkCudaTensor(requestBlockHostSlotsDevice, "request_block_host_slots_device");
    checkCudaTensor(requestBlockCommitGenDevice, "request_block_commit_gen_device");
    checkCudaTensor(requestAdmittedDevice, "request_admitted_device");

    checkInt64Tensor(requestIdsHost, "request_ids_host");
    checkInt64Tensor(requestIdsDevice, "request_ids_device");
    checkInt64Tensor(requestBlockHostSlotsHost, "request_block_host_slots_host");
    checkInt64Tensor(requestBlockHostSlotsDevice, "request_block_host_slots_device");
    checkInt64Tensor(requestBlockCommitGenHost, "request_block_commit_gen_host");
    checkInt64Tensor(requestBlockCommitGenDevice, "request_block_commit_gen_device");
    checkBoolTensor(requestAdmittedHost, "request_admitted_host");
    checkBoolTensor(requestAdmittedDevice, "request_admitted_device");

    TORCH_CHECK(requestIdsHost.dim() == 1, "request_ids_host must have shape [request_slots]");
    TORCH_CHECK(requestIdsDevice.dim() == 1, "request_ids_device must have shape [request_slots]");
    TORCH_CHECK(requestAdmittedHost.dim() == 1, "request_admitted_host must have shape [request_slots]");
    TORCH_CHECK(requestAdmittedDevice.dim() == 1, "request_admitted_device must have shape [request_slots]");
    TORCH_CHECK(requestBlockHostSlotsHost.dim() == 2,
        "request_block_host_slots_host must have shape [request_slots, max_blocks]");
    TORCH_CHECK(requestBlockHostSlotsDevice.dim() == 2,
        "request_block_host_slots_device must have shape [request_slots, max_blocks]");
    TORCH_CHECK(requestBlockCommitGenHost.dim() == 2,
        "request_block_commit_gen_host must have shape [request_slots, max_blocks]");
    TORCH_CHECK(requestBlockCommitGenDevice.dim() == 2,
        "request_block_commit_gen_device must have shape [request_slots, max_blocks]");
    TORCH_CHECK(requestIdsHost.sizes() == requestIdsDevice.sizes(), "request_ids host/device shape mismatch");
    TORCH_CHECK(requestAdmittedHost.sizes() == requestAdmittedDevice.sizes(),
        "request_admitted host/device shape mismatch");
    TORCH_CHECK(requestBlockHostSlotsHost.sizes() == requestBlockHostSlotsDevice.sizes(),
        "request_block_host_slots host/device shape mismatch");
    TORCH_CHECK(requestBlockCommitGenHost.sizes() == requestBlockCommitGenDevice.sizes(),
        "request_block_commit_gen host/device shape mismatch");
    TORCH_CHECK(requestBlockHostSlotsHost.sizes() == requestBlockCommitGenHost.sizes(),
        "request block host-slot and commit-gen shapes must match");
    TORCH_CHECK(requestBlockHostSlotsHost.size(0) == requestIdsHost.size(0),
        "request block table row count must match request id capacity");
    TORCH_CHECK(requestBlockHostSlotsHost.stride(1) == 1 && requestBlockHostSlotsDevice.stride(1) == 1,
        "request_block_host_slots rows must be contiguous");
    TORCH_CHECK(requestBlockCommitGenHost.stride(1) == 1 && requestBlockCommitGenDevice.stride(1) == 1,
        "request_block_commit_gen rows must be contiguous");
    TORCH_CHECK(requestIdsDevice.device() == requestBlockHostSlotsDevice.device()
            && requestIdsDevice.device() == requestBlockCommitGenDevice.device()
            && requestIdsDevice.device() == requestAdmittedDevice.device(),
        "all device request-table mirrors must be on the same CUDA device");
}

} // namespace

void hisparsePublishRequestTableSlots(th::Tensor const& requestIdsHost, th::Tensor const& requestIdsDevice,
    th::Tensor const& requestBlockHostSlotsHost, th::Tensor const& requestBlockHostSlotsDevice,
    th::Tensor const& requestBlockCommitGenHost, th::Tensor const& requestBlockCommitGenDevice,
    th::Tensor const& requestAdmittedHost, th::Tensor const& requestAdmittedDevice, th::Tensor const& tableSlots,
    bool syncBlocks)
{
    checkRequestTableShapes(requestIdsHost, requestIdsDevice, requestBlockHostSlotsHost, requestBlockHostSlotsDevice,
        requestBlockCommitGenHost, requestBlockCommitGenDevice, requestAdmittedHost, requestAdmittedDevice);
    checkSlots(tableSlots);
    int64_t const numSlots = tableSlots.size(0);
    if (numSlots == 0)
    {
        return;
    }

    c10::cuda::CUDAGuard guard(requestIdsDevice.device());
    auto stream = at::cuda::getCurrentCUDAStream(requestIdsDevice.get_device()).stream();
    int64_t const requestCapacity = requestIdsHost.size(0);
    int64_t const maxBlocks = requestBlockHostSlotsHost.size(1);
    for (int64_t i = 0; i < numSlots; ++i)
    {
        int64_t const slot = readSlot(tableSlots, i);
        TORCH_CHECK(slot >= 0 && slot < requestCapacity,
            "table slot out of range at index ", i, ": ", slot, " for request capacity ", requestCapacity);
        checkedH2DCopy(requestIdsDevice.data_ptr<int64_t>() + slot, requestIdsHost.data_ptr<int64_t>() + slot,
            sizeof(int64_t), stream, "request_ids");
        checkedH2DCopy(requestAdmittedDevice.data_ptr<bool>() + slot, requestAdmittedHost.data_ptr<bool>() + slot,
            sizeof(bool), stream, "request_admitted");
        if (!syncBlocks)
        {
            continue;
        }
        checkedH2DCopy(requestBlockHostSlotsDevice.data_ptr<int64_t>() + slot * requestBlockHostSlotsDevice.stride(0),
            requestBlockHostSlotsHost.data_ptr<int64_t>() + slot * requestBlockHostSlotsHost.stride(0),
            static_cast<size_t>(maxBlocks * sizeof(int64_t)), stream, "request_block_host_slots");
        checkedH2DCopy(requestBlockCommitGenDevice.data_ptr<int64_t>() + slot * requestBlockCommitGenDevice.stride(0),
            requestBlockCommitGenHost.data_ptr<int64_t>() + slot * requestBlockCommitGenHost.stride(0),
            static_cast<size_t>(maxBlocks * sizeof(int64_t)), stream, "request_block_commit_gen");
    }
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "hisparse_publish_request_table_slots(Tensor request_ids_host, Tensor(a!) request_ids_device, "
        "Tensor request_block_host_slots_host, Tensor(b!) request_block_host_slots_device, "
        "Tensor request_block_commit_gen_host, Tensor(c!) request_block_commit_gen_device, "
        "Tensor request_admitted_host, Tensor(d!) request_admitted_device, Tensor table_slots, bool sync_blocks) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("hisparse_publish_request_table_slots", &tensorrt_llm::torch_ext::hisparsePublishRequestTableSlots);
}

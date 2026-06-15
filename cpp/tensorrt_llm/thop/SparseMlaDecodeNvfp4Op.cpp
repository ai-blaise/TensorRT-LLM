/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/flashMLA/sparse_mla_decode_nvfp4.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <limits>
#include <optional>
#include <tuple>

namespace th = torch;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN
namespace torch_ext
{
namespace
{
constexpr int64_t kHeadQ = 128;
constexpr int64_t kHeadKv = 1;
constexpr int64_t kDqk = 576;
constexpr int64_t kDv = 512;
constexpr int64_t kPageBlockSize = 64;
constexpr int64_t kKvBytesPerToken = 288;
constexpr int64_t kScaleBytesPerToken = 36;

int32_t checkedInt32(int64_t value, char const* name)
{
    TORCH_CHECK(value >= std::numeric_limits<int32_t>::min() && value <= std::numeric_limits<int32_t>::max(),
        name, " does not fit int32: ", value);
    return static_cast<int32_t>(value);
}

bool isByteStorage(at::Tensor const& tensor)
{
    auto dtype = tensor.scalar_type();
    return dtype == at::ScalarType::Byte || dtype == at::ScalarType::Char || dtype == at::ScalarType::Float8_e4m3fn;
}

void checkCudaTensor(at::Tensor const& tensor, char const* name)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

void checkSameDevice(at::Tensor const& ref, at::Tensor const& tensor, char const* name)
{
    checkCudaTensor(tensor, name);
    TORCH_CHECK(tensor.get_device() == ref.get_device(), name, " must be on the same CUDA device as q");
}

void checkOptionalSameDevice(at::Tensor const& ref, std::optional<at::Tensor> const& tensor, char const* name)
{
    if (tensor.has_value())
    {
        checkSameDevice(ref, *tensor, name);
    }
}

void checkOptionalDtype(std::optional<at::Tensor> const& tensor, at::ScalarType dtype, char const* name)
{
    if (tensor.has_value())
    {
        TORCH_CHECK(tensor->scalar_type() == dtype, name, " has wrong dtype");
    }
}

void checkOptionalContiguous(std::optional<at::Tensor> const& tensor, char const* name)
{
    if (tensor.has_value())
    {
        TORCH_CHECK(tensor->is_contiguous(), name, " must be contiguous");
    }
}
} // namespace

std::tuple<th::Tensor, th::Tensor, th::Tensor, th::Tensor> sparse_mla_decode_nvfp4(th::Tensor const& q,
    th::Tensor const& kv, th::Tensor const& kvScales, th::Tensor const& indices,
    std::optional<th::Tensor> const& topkLength, std::optional<th::Tensor> const& attnSink,
    std::optional<th::Tensor> const& tileSchedulerMetadata, std::optional<th::Tensor> const& numSplits, int64_t dV,
    double smScale)
{
    checkCudaTensor(q, "q");
    checkSameDevice(q, kv, "kv");
    checkSameDevice(q, kvScales, "kv_scales");
    checkSameDevice(q, indices, "indices");
    checkOptionalSameDevice(q, topkLength, "topk_length");
    checkOptionalSameDevice(q, attnSink, "attn_sink");
    checkOptionalSameDevice(q, tileSchedulerMetadata, "tile_scheduler_metadata");
    checkOptionalSameDevice(q, numSplits, "num_splits");

    TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16, "q must be bf16");
    TORCH_CHECK(isByteStorage(kv), "kv must be uint8/int8/fp8 storage carrying packed NVFP4 payload bytes");
    TORCH_CHECK(isByteStorage(kvScales), "kv_scales must be uint8/int8/fp8 storage carrying E4M3 scale bytes");
    TORCH_CHECK(indices.scalar_type() == at::ScalarType::Int, "indices must be int32");
    checkOptionalDtype(topkLength, at::ScalarType::Int, "topk_length");
    checkOptionalDtype(attnSink, at::ScalarType::Float, "attn_sink");
    checkOptionalDtype(tileSchedulerMetadata, at::ScalarType::Int, "tile_scheduler_metadata");
    checkOptionalDtype(numSplits, at::ScalarType::Int, "num_splits");

    TORCH_CHECK(q.dim() == 4, "q must have shape [batch, s_q, h_q, d_qk]");
    TORCH_CHECK(kv.dim() == 4, "kv must have shape [num_pages, page_size, h_kv, 288]");
    TORCH_CHECK(kvScales.dim() == 4, "kv_scales must have shape [num_pages, page_size, h_kv, 36]");
    TORCH_CHECK(indices.dim() == 3, "indices must have shape [batch, s_q, topk]");

    auto const b = q.size(0);
    auto const sQ = q.size(1);
    auto const hQ = q.size(2);
    auto const dQk = q.size(3);
    auto const numBlocks = kv.size(0);
    auto const pageBlockSize = kv.size(1);
    auto const hKv = kv.size(2);
    auto const topK = indices.size(2);

    TORCH_CHECK(b > 0 && sQ > 0 && topK > 0, "batch, s_q, and topk must be positive");
    TORCH_CHECK(hQ == kHeadQ && hKv == kHeadKv && dQk == kDqk && dV == kDv,
        "sparse_mla_decode_nvfp4 supports only target V3.2 shape: h_q=128, h_kv=1, d_qk=576, d_v=512");
    TORCH_CHECK(pageBlockSize == kPageBlockSize, "page size must be 64 for target V3.2 NVFP4 sparse MLA");
    TORCH_CHECK(kv.size(3) == kKvBytesPerToken, "kv last dim must be 288 packed NVFP4 bytes");
    TORCH_CHECK(kvScales.size(0) == numBlocks && kvScales.size(1) == pageBlockSize && kvScales.size(2) == hKv
            && kvScales.size(3) == kScaleBytesPerToken,
        "kv_scales must match kv pages with 36 E4M3 scale bytes per token");
    TORCH_CHECK(indices.size(0) == b && indices.size(1) == sQ, "indices batch/s_q dimensions must match q");
    if (topkLength.has_value())
    {
        TORCH_CHECK(topkLength->dim() == 1 && topkLength->size(0) == b, "topk_length must be [batch]");
    }
    if (attnSink.has_value())
    {
        TORCH_CHECK(attnSink->dim() == 1 && attnSink->size(0) == hQ, "attn_sink must be [h_q]");
    }

    TORCH_CHECK(q.stride(3) == 1, "q last dimension must be contiguous");
    TORCH_CHECK(kv.stride(3) == 1 && kv.stride(1) == kKvBytesPerToken,
        "kv tokens must be packed-contiguous with stride(1)=288");
    TORCH_CHECK(kvScales.stride(3) == 1 && kvScales.stride(1) == kScaleBytesPerToken,
        "kv_scales tokens must be packed-contiguous with stride(1)=36");
    TORCH_CHECK(indices.stride(2) == 1, "indices last dimension must be contiguous");
    checkOptionalContiguous(topkLength, "topk_length");
    checkOptionalContiguous(attnSink, "attn_sink");
    checkOptionalContiguous(tileSchedulerMetadata, "tile_scheduler_metadata");
    checkOptionalContiguous(numSplits, "num_splits");
    TORCH_CHECK(tileSchedulerMetadata.has_value() == numSplits.has_value(),
        "tile_scheduler_metadata and num_splits must be provided together");

    c10::cuda::CUDAGuard deviceGuard(q.device());
    auto out = th::empty({b, sQ, hQ, dV}, q.options());
    auto lse = th::empty({b, sQ, hQ}, q.options().dtype(at::ScalarType::Float));

    bool const computeSchedulerMetadata = !tileSchedulerMetadata.has_value();
    auto const numSmParts = computeSchedulerMetadata
        ? tk::getSparseMlaDecodeNvfp4NumSmPartsForShape(
            checkedInt32(b, "batch"), checkedInt32(sQ, "s_q"), checkedInt32(topK, "topk"))
        : checkedInt32(tileSchedulerMetadata->size(0), "tile_scheduler_metadata.size(0)");
    auto metadata = computeSchedulerMetadata
        ? th::empty({numSmParts, tk::getSparseMlaDecodeNvfp4MetadataWidth()}, q.options().dtype(at::ScalarType::Int))
        : *tileSchedulerMetadata;
    auto splits = computeSchedulerMetadata ? th::empty({b + 1}, q.options().dtype(at::ScalarType::Int)) : *numSplits;

    TORCH_CHECK(metadata.dim() == 2 && metadata.size(0) == numSmParts
            && metadata.size(1) == tk::getSparseMlaDecodeNvfp4MetadataWidth(),
        "tile_scheduler_metadata has wrong shape for this s_q/device");
    TORCH_CHECK(splits.dim() == 1 && splits.size(0) == b + 1, "num_splits must be [batch + 1]");

    auto const totalSplits = tk::getSparseMlaDecodeNvfp4TotalSplits(checkedInt32(b, "batch"), numSmParts);
    auto lseAccum = th::empty({totalSplits, sQ, hQ}, q.options().dtype(at::ScalarType::Float));
    auto outAccum = th::empty({totalSplits, sQ, hQ, dV}, q.options().dtype(at::ScalarType::Float));

    tk::SparseMlaDecodeNvfp4Params params{};
    params.q = q.data_ptr();
    params.kv = reinterpret_cast<uint8_t*>(kv.data_ptr());
    params.kvScales = reinterpret_cast<uint8_t*>(kvScales.data_ptr());
    params.indices = reinterpret_cast<int32_t*>(indices.data_ptr());
    params.topkLength = topkLength.has_value() ? reinterpret_cast<int32_t*>(topkLength->data_ptr()) : nullptr;
    params.attnSink = attnSink.has_value() ? reinterpret_cast<float*>(attnSink->data_ptr()) : nullptr;
    params.lse = reinterpret_cast<float*>(lse.data_ptr());
    params.out = out.data_ptr();
    params.tileSchedulerMetadata = reinterpret_cast<int32_t*>(metadata.data_ptr());
    params.numSplits = reinterpret_cast<int32_t*>(splits.data_ptr());
    params.lseAccum = reinterpret_cast<float*>(lseAccum.data_ptr());
    params.outAccum = reinterpret_cast<float*>(outAccum.data_ptr());
    params.b = checkedInt32(b, "batch");
    params.sQ = checkedInt32(sQ, "s_q");
    params.hQ = checkedInt32(hQ, "h_q");
    params.hKv = checkedInt32(hKv, "h_kv");
    params.dQk = checkedInt32(dQk, "d_qk");
    params.dV = checkedInt32(dV, "d_v");
    params.numBlocks = checkedInt32(numBlocks, "num_blocks");
    params.pageBlockSize = checkedInt32(pageBlockSize, "page_block_size");
    params.topK = checkedInt32(topK, "topk");
    params.smScale = static_cast<float>(smScale);
    params.strideQB = checkedInt32(q.stride(0), "q.stride(0)");
    params.strideQSQ = checkedInt32(q.stride(1), "q.stride(1)");
    params.strideQHQ = checkedInt32(q.stride(2), "q.stride(2)");
    params.strideKvBlock = checkedInt32(kv.stride(0), "kv.stride(0)");
    params.strideKvRow = checkedInt32(kv.stride(1), "kv.stride(1)");
    params.strideKvScalesBlock = checkedInt32(kvScales.stride(0), "kv_scales.stride(0)");
    params.strideKvScalesRow = checkedInt32(kvScales.stride(1), "kv_scales.stride(1)");
    params.strideIndicesB = checkedInt32(indices.stride(0), "indices.stride(0)");
    params.strideIndicesSQ = checkedInt32(indices.stride(1), "indices.stride(1)");
    params.strideLseB = checkedInt32(lse.stride(0), "lse.stride(0)");
    params.strideLseSQ = checkedInt32(lse.stride(1), "lse.stride(1)");
    params.strideOB = checkedInt32(out.stride(0), "out.stride(0)");
    params.strideOSQ = checkedInt32(out.stride(1), "out.stride(1)");
    params.strideOHQ = checkedInt32(out.stride(2), "out.stride(2)");
    params.strideLseAccumSplit = checkedInt32(lseAccum.stride(0), "lse_accum.stride(0)");
    params.strideLseAccumSQ = checkedInt32(lseAccum.stride(1), "lse_accum.stride(1)");
    params.strideOAccumSplit = checkedInt32(outAccum.stride(0), "out_accum.stride(0)");
    params.strideOAccumSQ = checkedInt32(outAccum.stride(1), "out_accum.stride(1)");
    params.strideOAccumHQ = checkedInt32(outAccum.stride(2), "out_accum.stride(2)");
    params.numSmParts = numSmParts;
    params.computeSchedulerMetadata = computeSchedulerMetadata;

    tk::invokeSparseMlaDecodeNvfp4(params, at::cuda::getCurrentCUDAStream(q.get_device()));
    return {out, lse.transpose(1, 2), metadata, splits};
}

// Fused variant: identical attention math to sparse_mla_decode_nvfp4, but the
// per-head v_b (W_UV) projection latent[d_v=512] @ W_UV[d_v,v_head_dim] is fused
// into the decode/combine epilogue. Returns the projected output [b, s_q, h_q,
// v_head_dim] directly, dropping the external bmm. Milestone scope: bf16 W_UV
// only (the caller keeps the external bmm for fp8 W_UV). The original op is left
// intact as the fallback.
std::tuple<th::Tensor, th::Tensor, th::Tensor, th::Tensor> sparse_mla_decode_nvfp4_vfuse(th::Tensor const& q,
    th::Tensor const& kv, th::Tensor const& kvScales, th::Tensor const& indices, th::Tensor const& vBProj,
    std::optional<th::Tensor> const& topkLength, std::optional<th::Tensor> const& attnSink,
    std::optional<th::Tensor> const& tileSchedulerMetadata, std::optional<th::Tensor> const& numSplits, int64_t dV,
    int64_t vHeadDim, double smScale)
{
    checkCudaTensor(q, "q");
    checkSameDevice(q, kv, "kv");
    checkSameDevice(q, kvScales, "kv_scales");
    checkSameDevice(q, indices, "indices");
    checkSameDevice(q, vBProj, "v_b_proj");
    checkOptionalSameDevice(q, topkLength, "topk_length");
    checkOptionalSameDevice(q, attnSink, "attn_sink");
    checkOptionalSameDevice(q, tileSchedulerMetadata, "tile_scheduler_metadata");
    checkOptionalSameDevice(q, numSplits, "num_splits");

    TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16, "q must be bf16");
    TORCH_CHECK(isByteStorage(kv), "kv must be uint8/int8/fp8 storage carrying packed NVFP4 payload bytes");
    TORCH_CHECK(isByteStorage(kvScales), "kv_scales must be uint8/int8/fp8 storage carrying E4M3 scale bytes");
    TORCH_CHECK(indices.scalar_type() == at::ScalarType::Int, "indices must be int32");
    TORCH_CHECK(vBProj.scalar_type() == at::ScalarType::BFloat16,
        "v_b_proj must be bf16 for the fused op (fp8 W_UV must use the external bmm)");
    checkOptionalDtype(topkLength, at::ScalarType::Int, "topk_length");
    checkOptionalDtype(attnSink, at::ScalarType::Float, "attn_sink");
    checkOptionalDtype(tileSchedulerMetadata, at::ScalarType::Int, "tile_scheduler_metadata");
    checkOptionalDtype(numSplits, at::ScalarType::Int, "num_splits");

    TORCH_CHECK(q.dim() == 4, "q must have shape [batch, s_q, h_q, d_qk]");
    TORCH_CHECK(kv.dim() == 4, "kv must have shape [num_pages, page_size, h_kv, 288]");
    TORCH_CHECK(kvScales.dim() == 4, "kv_scales must have shape [num_pages, page_size, h_kv, 36]");
    TORCH_CHECK(indices.dim() == 3, "indices must have shape [batch, s_q, topk]");
    TORCH_CHECK(vBProj.dim() == 3, "v_b_proj must have shape [h_q, v_head_dim, kv_lora_rank]");

    auto const b = q.size(0);
    auto const sQ = q.size(1);
    auto const hQ = q.size(2);
    auto const dQk = q.size(3);
    auto const numBlocks = kv.size(0);
    auto const pageBlockSize = kv.size(1);
    auto const hKv = kv.size(2);
    auto const topK = indices.size(2);

    TORCH_CHECK(b > 0 && sQ > 0 && topK > 0, "batch, s_q, and topk must be positive");
    TORCH_CHECK(hQ == kHeadQ && hKv == kHeadKv && dQk == kDqk && dV == kDv,
        "sparse_mla_decode_nvfp4_vfuse supports only target V3.2 shape: h_q=128, h_kv=1, d_qk=576, d_v=512");
    TORCH_CHECK(pageBlockSize == kPageBlockSize, "page size must be 64 for target V3.2 NVFP4 sparse MLA");
    TORCH_CHECK(kv.size(3) == kKvBytesPerToken, "kv last dim must be 288 packed NVFP4 bytes");
    TORCH_CHECK(kvScales.size(0) == numBlocks && kvScales.size(1) == pageBlockSize && kvScales.size(2) == hKv
            && kvScales.size(3) == kScaleBytesPerToken,
        "kv_scales must match kv pages with 36 E4M3 scale bytes per token");
    TORCH_CHECK(indices.size(0) == b && indices.size(1) == sQ, "indices batch/s_q dimensions must match q");
    TORCH_CHECK(vBProj.size(0) == hQ && vBProj.size(1) == vHeadDim && vBProj.size(2) == dV,
        "v_b_proj must have shape [h_q=128, v_head_dim, kv_lora_rank=512]");
    TORCH_CHECK(vHeadDim == 128, "fused v_b currently supports only v_head_dim=128");
    if (topkLength.has_value())
    {
        TORCH_CHECK(topkLength->dim() == 1 && topkLength->size(0) == b, "topk_length must be [batch]");
    }
    if (attnSink.has_value())
    {
        TORCH_CHECK(attnSink->dim() == 1 && attnSink->size(0) == hQ, "attn_sink must be [h_q]");
    }

    TORCH_CHECK(q.stride(3) == 1, "q last dimension must be contiguous");
    TORCH_CHECK(kv.stride(3) == 1 && kv.stride(1) == kKvBytesPerToken,
        "kv tokens must be packed-contiguous with stride(1)=288");
    TORCH_CHECK(kvScales.stride(3) == 1 && kvScales.stride(1) == kScaleBytesPerToken,
        "kv_scales tokens must be packed-contiguous with stride(1)=36");
    TORCH_CHECK(indices.stride(2) == 1, "indices last dimension must be contiguous");
    TORCH_CHECK(vBProj.stride(2) == 1, "v_b_proj last (kv_lora_rank) dimension must be contiguous");
    checkOptionalContiguous(topkLength, "topk_length");
    checkOptionalContiguous(attnSink, "attn_sink");
    checkOptionalContiguous(tileSchedulerMetadata, "tile_scheduler_metadata");
    checkOptionalContiguous(numSplits, "num_splits");
    TORCH_CHECK(tileSchedulerMetadata.has_value() == numSplits.has_value(),
        "tile_scheduler_metadata and num_splits must be provided together");

    c10::cuda::CUDAGuard deviceGuard(q.device());
    // Output is the projected v_head_dim-wide tensor (the external bmm is dropped).
    auto out = th::empty({b, sQ, hQ, vHeadDim}, q.options());
    auto lse = th::empty({b, sQ, hQ}, q.options().dtype(at::ScalarType::Float));

    bool const computeSchedulerMetadata = !tileSchedulerMetadata.has_value();
    auto const numSmParts = computeSchedulerMetadata
        ? tk::getSparseMlaDecodeNvfp4NumSmPartsForShape(
            checkedInt32(b, "batch"), checkedInt32(sQ, "s_q"), checkedInt32(topK, "topk"))
        : checkedInt32(tileSchedulerMetadata->size(0), "tile_scheduler_metadata.size(0)");
    auto metadata = computeSchedulerMetadata
        ? th::empty({numSmParts, tk::getSparseMlaDecodeNvfp4MetadataWidth()}, q.options().dtype(at::ScalarType::Int))
        : *tileSchedulerMetadata;
    auto splits = computeSchedulerMetadata ? th::empty({b + 1}, q.options().dtype(at::ScalarType::Int)) : *numSplits;

    TORCH_CHECK(metadata.dim() == 2 && metadata.size(0) == numSmParts
            && metadata.size(1) == tk::getSparseMlaDecodeNvfp4MetadataWidth(),
        "tile_scheduler_metadata has wrong shape for this s_q/device");
    TORCH_CHECK(splits.dim() == 1 && splits.size(0) == b + 1, "num_splits must be [batch + 1]");

    // The split path's o_accum / lse_accum stay at the latent width (d_v=512): the
    // decode kernel writes the partial latent, and the combine kernel does the
    // cross-split reduction and THEN the W_UV projection down to v_head_dim.
    auto const totalSplits = tk::getSparseMlaDecodeNvfp4TotalSplits(checkedInt32(b, "batch"), numSmParts);
    auto lseAccum = th::empty({totalSplits, sQ, hQ}, q.options().dtype(at::ScalarType::Float));
    auto outAccum = th::empty({totalSplits, sQ, hQ, dV}, q.options().dtype(at::ScalarType::Float));

    tk::SparseMlaDecodeNvfp4Params params{};
    params.q = q.data_ptr();
    params.kv = reinterpret_cast<uint8_t*>(kv.data_ptr());
    params.kvScales = reinterpret_cast<uint8_t*>(kvScales.data_ptr());
    params.indices = reinterpret_cast<int32_t*>(indices.data_ptr());
    params.topkLength = topkLength.has_value() ? reinterpret_cast<int32_t*>(topkLength->data_ptr()) : nullptr;
    params.attnSink = attnSink.has_value() ? reinterpret_cast<float*>(attnSink->data_ptr()) : nullptr;
    params.lse = reinterpret_cast<float*>(lse.data_ptr());
    params.out = out.data_ptr();
    params.tileSchedulerMetadata = reinterpret_cast<int32_t*>(metadata.data_ptr());
    params.numSplits = reinterpret_cast<int32_t*>(splits.data_ptr());
    params.lseAccum = reinterpret_cast<float*>(lseAccum.data_ptr());
    params.outAccum = reinterpret_cast<float*>(outAccum.data_ptr());
    params.vBProj = reinterpret_cast<cutlass::bfloat16_t*>(vBProj.data_ptr());
    params.vHeadDim = checkedInt32(vHeadDim, "v_head_dim");
    params.strideVbH = checkedInt32(vBProj.stride(0), "v_b_proj.stride(0)");
    params.strideVbVhd = checkedInt32(vBProj.stride(1), "v_b_proj.stride(1)");
    params.strideVbD = checkedInt32(vBProj.stride(2), "v_b_proj.stride(2)");
    params.b = checkedInt32(b, "batch");
    params.sQ = checkedInt32(sQ, "s_q");
    params.hQ = checkedInt32(hQ, "h_q");
    params.hKv = checkedInt32(hKv, "h_kv");
    params.dQk = checkedInt32(dQk, "d_qk");
    params.dV = checkedInt32(dV, "d_v");
    params.numBlocks = checkedInt32(numBlocks, "num_blocks");
    params.pageBlockSize = checkedInt32(pageBlockSize, "page_block_size");
    params.topK = checkedInt32(topK, "topk");
    params.smScale = static_cast<float>(smScale);
    params.strideQB = checkedInt32(q.stride(0), "q.stride(0)");
    params.strideQSQ = checkedInt32(q.stride(1), "q.stride(1)");
    params.strideQHQ = checkedInt32(q.stride(2), "q.stride(2)");
    params.strideKvBlock = checkedInt32(kv.stride(0), "kv.stride(0)");
    params.strideKvRow = checkedInt32(kv.stride(1), "kv.stride(1)");
    params.strideKvScalesBlock = checkedInt32(kvScales.stride(0), "kv_scales.stride(0)");
    params.strideKvScalesRow = checkedInt32(kvScales.stride(1), "kv_scales.stride(1)");
    params.strideIndicesB = checkedInt32(indices.stride(0), "indices.stride(0)");
    params.strideIndicesSQ = checkedInt32(indices.stride(1), "indices.stride(1)");
    params.strideLseB = checkedInt32(lse.stride(0), "lse.stride(0)");
    params.strideLseSQ = checkedInt32(lse.stride(1), "lse.stride(1)");
    params.strideOB = checkedInt32(out.stride(0), "out.stride(0)");
    params.strideOSQ = checkedInt32(out.stride(1), "out.stride(1)");
    params.strideOHQ = checkedInt32(out.stride(2), "out.stride(2)");
    params.strideLseAccumSplit = checkedInt32(lseAccum.stride(0), "lse_accum.stride(0)");
    params.strideLseAccumSQ = checkedInt32(lseAccum.stride(1), "lse_accum.stride(1)");
    params.strideOAccumSplit = checkedInt32(outAccum.stride(0), "out_accum.stride(0)");
    params.strideOAccumSQ = checkedInt32(outAccum.stride(1), "out_accum.stride(1)");
    params.strideOAccumHQ = checkedInt32(outAccum.stride(2), "out_accum.stride(2)");
    params.numSmParts = numSmParts;
    params.computeSchedulerMetadata = computeSchedulerMetadata;

    tk::invokeSparseMlaDecodeNvfp4(params, at::cuda::getCurrentCUDAStream(q.get_device()));
    return {out, lse.transpose(1, 2), metadata, splits};
}

} // namespace torch_ext
TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "sparse_mla_decode_nvfp4(Tensor q, Tensor kv, Tensor kv_scales, Tensor indices, Tensor? topk_length=None, "
        "Tensor? attn_sink=None, Tensor? tile_scheduler_metadata=None, Tensor? num_splits=None, int d_v=512, "
        "float sm_scale=1.) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "sparse_mla_decode_nvfp4_vfuse(Tensor q, Tensor kv, Tensor kv_scales, Tensor indices, Tensor v_b_proj, "
        "Tensor? topk_length=None, Tensor? attn_sink=None, Tensor? tile_scheduler_metadata=None, "
        "Tensor? num_splits=None, int d_v=512, int v_head_dim=128, float sm_scale=1.) -> (Tensor, Tensor, Tensor, "
        "Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("sparse_mla_decode_nvfp4", &tensorrt_llm::torch_ext::sparse_mla_decode_nvfp4);
    m.impl("sparse_mla_decode_nvfp4_vfuse", &tensorrt_llm::torch_ext::sparse_mla_decode_nvfp4_vfuse);
}

#include "tensorrt_llm/kernels/flashMLA/sparse_mla_decode_nvfp4.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdlib>
#include <limits>
#include <optional>
#include <tuple>

namespace tk = tensorrt_llm::kernels;

namespace {
constexpr int64_t kHeadQ = 128;
constexpr int64_t kHeadKv = 1;
constexpr int64_t kDqk = 576;
constexpr int64_t kDv = 512;
constexpr int64_t kPageBlockSize = 64;
constexpr int64_t kKvBytesPerToken = 288;
constexpr int64_t kScaleBytesPerToken = 36;

int32_t checkedInt32(int64_t value, char const* name) {
    TORCH_CHECK(value >= std::numeric_limits<int32_t>::min() && value <= std::numeric_limits<int32_t>::max(),
        name, " does not fit int32: ", value);
    return static_cast<int32_t>(value);
}

bool isByteStorage(at::Tensor const& tensor) {
    auto dtype = tensor.scalar_type();
    return dtype == at::ScalarType::Byte || dtype == at::ScalarType::Char || dtype == at::ScalarType::Float8_e4m3fn;
}

void checkCudaTensor(at::Tensor const& tensor, char const* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
}

void checkSameDevice(at::Tensor const& ref, at::Tensor const& tensor, char const* name) {
    checkCudaTensor(tensor, name);
    TORCH_CHECK(tensor.get_device() == ref.get_device(), name, " must be on q device");
}

void checkOptionalSameDevice(at::Tensor const& ref, std::optional<at::Tensor> const& tensor, char const* name) {
    if (tensor.has_value()) {
        checkSameDevice(ref, *tensor, name);
    }
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> sparse_mla_decode_nvfp4_bench(
    at::Tensor const& q,
    at::Tensor const& kv,
    at::Tensor const& kvScales,
    at::Tensor const& indices,
    std::optional<at::Tensor> const& topkLength,
    std::optional<at::Tensor> const& tileSchedulerMetadata,
    std::optional<at::Tensor> const& numSplits,
    double smScale) {
    checkCudaTensor(q, "q");
    checkSameDevice(q, kv, "kv");
    checkSameDevice(q, kvScales, "kv_scales");
    checkSameDevice(q, indices, "indices");
    checkOptionalSameDevice(q, topkLength, "topk_length");
    checkOptionalSameDevice(q, tileSchedulerMetadata, "tile_scheduler_metadata");
    checkOptionalSameDevice(q, numSplits, "num_splits");

    TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16, "q must be bf16");
    TORCH_CHECK(isByteStorage(kv), "kv must be byte/fp8 storage");
    TORCH_CHECK(isByteStorage(kvScales), "kv_scales must be byte/fp8 storage");
    TORCH_CHECK(indices.scalar_type() == at::ScalarType::Int, "indices must be int32");
    if (topkLength.has_value()) {
        TORCH_CHECK(topkLength->scalar_type() == at::ScalarType::Int, "topk_length must be int32");
        TORCH_CHECK(topkLength->is_contiguous(), "topk_length must be contiguous");
    }
    if (tileSchedulerMetadata.has_value()) {
        TORCH_CHECK(tileSchedulerMetadata->scalar_type() == at::ScalarType::Int, "metadata must be int32");
        TORCH_CHECK(tileSchedulerMetadata->is_contiguous(), "metadata must be contiguous");
    }
    if (numSplits.has_value()) {
        TORCH_CHECK(numSplits->scalar_type() == at::ScalarType::Int, "num_splits must be int32");
        TORCH_CHECK(numSplits->is_contiguous(), "num_splits must be contiguous");
    }
    TORCH_CHECK(tileSchedulerMetadata.has_value() == numSplits.has_value(),
        "metadata and num_splits must be provided together");

    TORCH_CHECK(q.dim() == 4, "q must be [B,SQ,HQ,DQK]");
    TORCH_CHECK(kv.dim() == 4, "kv must be [pages,page,Hkv,288]");
    TORCH_CHECK(kvScales.dim() == 4, "kv_scales must be [pages,page,Hkv,36]");
    TORCH_CHECK(indices.dim() == 3, "indices must be [B,SQ,TopK]");

    auto const b = q.size(0);
    auto const sQ = q.size(1);
    auto const hQ = q.size(2);
    auto const dQk = q.size(3);
    auto const numBlocks = kv.size(0);
    auto const pageBlockSize = kv.size(1);
    auto const hKv = kv.size(2);
    auto const topK = indices.size(2);

    TORCH_CHECK(hQ == kHeadQ && hKv == kHeadKv && dQk == kDqk,
        "target V3.2 shape required: h_q=128,h_kv=1,d_qk=576");
    TORCH_CHECK(pageBlockSize == kPageBlockSize, "page size must be 64");
    TORCH_CHECK(kv.size(3) == kKvBytesPerToken, "kv last dim must be 288");
    TORCH_CHECK(kvScales.size(0) == numBlocks && kvScales.size(1) == pageBlockSize && kvScales.size(2) == hKv
            && kvScales.size(3) == kScaleBytesPerToken,
        "kv_scales must match kv with last dim 36");
    TORCH_CHECK(indices.size(0) == b && indices.size(1) == sQ, "indices dims must match q");
    TORCH_CHECK(q.stride(3) == 1, "q last dim must be contiguous");
    TORCH_CHECK(kv.stride(3) == 1 && kv.stride(1) == kKvBytesPerToken, "kv token rows must be contiguous");
    TORCH_CHECK(kvScales.stride(3) == 1 && kvScales.stride(1) == kScaleBytesPerToken, "scale token rows must be contiguous");
    TORCH_CHECK(indices.stride(2) == 1, "indices last dim must be contiguous");

    c10::cuda::CUDAGuard deviceGuard(q.device());
    auto out = at::empty({b, sQ, hQ, kDv}, q.options());
    auto lse = at::empty({b, sQ, hQ}, q.options().dtype(at::ScalarType::Float));
    bool const computeMetadata = !tileSchedulerMetadata.has_value();
    auto numSmParts = computeMetadata
        ? tk::getSparseMlaDecodeNvfp4NumSmPartsForShape(
            checkedInt32(b, "batch"), checkedInt32(sQ, "s_q"), checkedInt32(topK, "topk"))
        : checkedInt32(tileSchedulerMetadata->size(0), "tile_scheduler_metadata.size(0)");
    if (computeMetadata) {
        if (char const* overrideSmParts = std::getenv("OPTRT_NVFP4_NUM_SM_PARTS")) {
            numSmParts = checkedInt32(std::strtol(overrideSmParts, nullptr, 10), "OPTRT_NVFP4_NUM_SM_PARTS");
        }
    }
    auto metadata = computeMetadata
        ? at::empty({numSmParts, tk::getSparseMlaDecodeNvfp4MetadataWidth()}, q.options().dtype(at::ScalarType::Int))
        : *tileSchedulerMetadata;
    auto splits = computeMetadata ? at::empty({b + 1}, q.options().dtype(at::ScalarType::Int)) : *numSplits;
    auto const totalSplits = tk::getSparseMlaDecodeNvfp4TotalSplits(checkedInt32(b, "batch"), numSmParts);
    auto lseAccum = at::empty({totalSplits, sQ, hQ}, q.options().dtype(at::ScalarType::Float));
    auto outAccum = at::empty({totalSplits, sQ, hQ, kDv}, q.options().dtype(at::ScalarType::Float));

    tk::SparseMlaDecodeNvfp4Params params{};
    params.q = q.data_ptr();
    params.kv = reinterpret_cast<uint8_t*>(kv.data_ptr());
    params.kvScales = reinterpret_cast<uint8_t*>(kvScales.data_ptr());
    params.indices = reinterpret_cast<int32_t*>(indices.data_ptr());
    params.topkLength = topkLength.has_value() ? reinterpret_cast<int32_t*>(topkLength->data_ptr()) : nullptr;
    params.attnSink = nullptr;
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
    params.dV = kDv;
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
    params.computeSchedulerMetadata = computeMetadata;

    tk::invokeSparseMlaDecodeNvfp4(params, at::cuda::getCurrentCUDAStream(q.get_device()));
    return {out, lse.transpose(1, 2), metadata, splits, lseAccum, outAccum};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_mla_decode_nvfp4", &sparse_mla_decode_nvfp4_bench,
        "Benchmark-only native op-trt sparse MLA NVFP4 decode wrapper");
}

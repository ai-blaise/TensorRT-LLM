// Standalone build shim for iterating on indexer_xstep_recency_patch without
// rebuilding libth_common.so. Registers the SAME trtllm op + dispatch as the
// production cpp/tensorrt_llm/thop/indexerXstepRecencyPatchOp.cpp, but includes
// only torch/ATen headers (no heavy trtllm headers) so it links against torch
// alone and loads via torch.ops.load_library. The production op file is the
// committed/CMake path; this shim exists purely for the microbench .so.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "tensorrt_llm/kernels/IndexerXstepRecencyPatch.h"

namespace tk = tensorrt_llm::kernels;

namespace
{

torch::Tensor indexer_xstep_recency_patch(torch::Tensor const& cached_topk, torch::Tensor const& refresh_end,
    torch::Tensor const& cur_kv_lens, int64_t next_n, int64_t max_delta)
{
    TORCH_CHECK(cached_topk.is_cuda() && refresh_end.is_cuda() && cur_kv_lens.is_cuda(),
        "cached_topk, refresh_end, and cur_kv_lens must be CUDA tensors");
    TORCH_CHECK(cached_topk.dim() == 2, "cached_topk must be 2D [numRows, indexTopK]");
    TORCH_CHECK(refresh_end.dim() == 1, "refresh_end must be 1D [numRows]");
    TORCH_CHECK(cur_kv_lens.dim() == 1, "cur_kv_lens must be 1D [numBatches]");
    TORCH_CHECK(cached_topk.scalar_type() == at::ScalarType::Int, "cached_topk must be int32");
    TORCH_CHECK(refresh_end.scalar_type() == at::ScalarType::Int, "refresh_end must be int32");
    TORCH_CHECK(cur_kv_lens.scalar_type() == at::ScalarType::Int, "cur_kv_lens must be int32");
    TORCH_CHECK(cached_topk.is_contiguous() && refresh_end.is_contiguous() && cur_kv_lens.is_contiguous(),
        "inputs must be contiguous");
    TORCH_CHECK(next_n > 0, "next_n must be > 0");

    auto const numRows = static_cast<int>(cached_topk.size(0));
    auto const indexTopK = static_cast<int>(cached_topk.size(1));
    TORCH_CHECK(refresh_end.size(0) == cached_topk.size(0), "refresh_end length must equal cached_topk.size(0)");
    TORCH_CHECK(numRows % next_n == 0, "cached_topk.size(0) must be divisible by next_n");
    TORCH_CHECK(max_delta >= 0 && max_delta <= indexTopK, "max_delta must be in [0, indexTopK]");

    auto stream = at::cuda::getCurrentCUDAStream(cached_topk.get_device());
    tk::invokeIndexerXstepRecencyPatch(cached_topk.data_ptr<int32_t>(), refresh_end.data_ptr<int32_t>(),
        cur_kv_lens.data_ptr<int32_t>(), numRows, indexTopK, static_cast<int>(next_n), static_cast<int>(max_delta),
        stream);
    return cached_topk;
}

} // namespace

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "indexer_xstep_recency_patch(Tensor cached_topk, Tensor refresh_end, Tensor cur_kv_lens, int next_n, "
        "int max_delta) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("indexer_xstep_recency_patch", &indexer_xstep_recency_patch);
}

/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include "tensorrt_llm/kernels/IndexerXstepRecencyPatch.h"

namespace th = torch;
namespace tl = tensorrt_llm;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// Fused cross-step recency patch for the DSA indexer cross-step Top-K reuse.
// Replaces the ~20 launch-bound PyTorch tensor ops of the recency-patch branch
// in Indexer._xstep_reuse_decode with a single kernel launch. Patches
// cached_topk in place and returns it (so the Python side can keep its existing
// `cached = op(...)` assignment shape).
th::Tensor indexer_xstep_recency_patch(th::Tensor const& cached_topk, th::Tensor const& refresh_end,
    th::Tensor const& cur_kv_lens, int64_t next_n, int64_t max_delta)
{
    TORCH_CHECK(cached_topk.is_cuda() && refresh_end.is_cuda() && cur_kv_lens.is_cuda(),
        "cached_topk, refresh_end, and cur_kv_lens must be CUDA tensors");
    TORCH_CHECK(cached_topk.get_device() == refresh_end.get_device()
            && cached_topk.get_device() == cur_kv_lens.get_device(),
        "cached_topk, refresh_end, and cur_kv_lens must be on the same device");

    TORCH_CHECK(cached_topk.dim() == 2, "cached_topk must be a 2D Tensor [numRows, indexTopK]");
    TORCH_CHECK(refresh_end.dim() == 1, "refresh_end must be a 1D Tensor [numRows]");
    TORCH_CHECK(cur_kv_lens.dim() == 1, "cur_kv_lens must be a 1D Tensor [numBatches]");

    TORCH_CHECK(cached_topk.scalar_type() == at::ScalarType::Int, "cached_topk must be int32");
    TORCH_CHECK(refresh_end.scalar_type() == at::ScalarType::Int, "refresh_end must be int32");
    TORCH_CHECK(cur_kv_lens.scalar_type() == at::ScalarType::Int, "cur_kv_lens must be int32");

    TORCH_CHECK(cached_topk.is_contiguous(), "cached_topk must be contiguous");
    TORCH_CHECK(refresh_end.is_contiguous(), "refresh_end must be contiguous");
    TORCH_CHECK(cur_kv_lens.is_contiguous(), "cur_kv_lens must be contiguous");

    TORCH_CHECK(next_n > 0, "next_n must be greater than 0");

    auto const numRows64 = cached_topk.size(0);
    auto const indexTopK64 = cached_topk.size(1);
    TORCH_CHECK(refresh_end.size(0) == numRows64, "refresh_end length must equal cached_topk.size(0)");
    TORCH_CHECK(numRows64 % next_n == 0, "cached_topk.size(0) must be divisible by next_n");
    TORCH_CHECK(
        cur_kv_lens.size(0) * next_n >= numRows64, "cur_kv_lens length * next_n must be >= cached_topk.size(0)");
    TORCH_CHECK(max_delta >= 0 && max_delta <= indexTopK64, "max_delta must be in [0, indexTopK]");

    int32_t num_rows = static_cast<int32_t>(numRows64);
    int32_t index_topk = static_cast<int32_t>(indexTopK64);

    auto stream = at::cuda::getCurrentCUDAStream(cached_topk.get_device());
    tk::invokeIndexerXstepRecencyPatch(cached_topk.data_ptr<int32_t>(), refresh_end.data_ptr<int32_t>(),
        cur_kv_lens.data_ptr<int32_t>(), num_rows, index_topk, static_cast<int32_t>(next_n),
        static_cast<int32_t>(max_delta), stream);

    return cached_topk;
}

} // end namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "indexer_xstep_recency_patch(Tensor cached_topk, Tensor refresh_end, Tensor cur_kv_lens, int next_n, "
        "int max_delta) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("indexer_xstep_recency_patch", &tensorrt_llm::torch_ext::indexer_xstep_recency_patch);
}

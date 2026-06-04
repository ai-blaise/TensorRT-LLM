/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/kernels/indexerAffineReuse.h"
#include "tensorrt_llm/runtime/torchUtils.h"

namespace th = torch;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// Reuse the global TopK indices of an FSSS owning ("F") indexer layer on a
// reuse ("S") layer by adding a per-layer-pair constant offset, preserving -1
// entries. See indexerAffineReuse.h for the derivation. `delta` is
// (layerId_S - layerId_F) * blockSize, a static per-layer value.
th::Tensor indexerAffineReuse(th::Tensor const& globalIndicesF, int64_t delta)
{
    TORCH_CHECK(globalIndicesF.is_cuda(), "global_indices_f must be a CUDA tensor");
    TORCH_CHECK(globalIndicesF.scalar_type() == th::kInt32, "global_indices_f must be int32");

    auto gF = globalIndicesF.contiguous();
    auto out = th::empty_like(gF);

    auto stream = at::cuda::getCurrentCUDAStream(gF.get_device()).stream();
    tk::invokeIndexerAffineReuse(
        gF.data_ptr<int32_t>(), out.data_ptr<int32_t>(), gF.numel(), static_cast<int32_t>(delta), stream);

    return out;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("indexer_affine_reuse(Tensor global_indices_f, int delta) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("indexer_affine_reuse", &tensorrt_llm::torch_ext::indexerAffineReuse);
}

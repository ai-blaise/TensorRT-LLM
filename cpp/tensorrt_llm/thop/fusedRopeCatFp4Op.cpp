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

#include "tensorrt_llm/kernels/fusedRopeCatFp4.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAGuard.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

std::tuple<at::Tensor, at::Tensor> fused_rope_cat_fp4(
    at::Tensor const& pe, at::Tensor const& nope, at::Tensor const& cos_sin, at::Tensor const& pos)
{
    CHECK_TH_CUDA(pe);
    CHECK_TH_CUDA(nope);
    CHECK_TH_CUDA(cos_sin);
    CHECK_TH_CUDA(pos);
    TORCH_CHECK(pe.device() == nope.device() && pe.device() == cos_sin.device() && pe.device() == pos.device(),
        "pe, nope, cos_sin, pos must be on the same CUDA device");
    c10::cuda::CUDAGuard deviceGuard{pe.device()};

    TORCH_CHECK(pe.scalar_type() == at::ScalarType::BFloat16, "pe must be BF16, got ", pe.scalar_type());
    TORCH_CHECK(nope.scalar_type() == at::ScalarType::BFloat16, "nope must be BF16, got ", nope.scalar_type());
    TORCH_CHECK(cos_sin.scalar_type() == at::ScalarType::Float, "cos_sin must be FP32, got ", cos_sin.scalar_type());
    TORCH_CHECK(pos.scalar_type() == at::ScalarType::Int, "pos must be int32, got ", pos.scalar_type());
    TORCH_CHECK(pe.dim() >= 2, "pe must be >= 2D, got ", pe.dim(), "D");
    TORCH_CHECK(nope.dim() >= 2, "nope must be >= 2D, got ", nope.dim(), "D");

    TORCH_CHECK(pe.stride(-1) == 1, "pe must have contiguous innermost dim (stride(-1)==1), got ", pe.stride(-1));
    TORCH_CHECK(nope.stride(-1) == 1, "nope must have contiguous innermost dim (stride(-1)==1), got ", nope.stride(-1));
    TORCH_CHECK(cos_sin.stride(-1) == 1, "cos_sin must have contiguous innermost dim, got ", cos_sin.stride(-1));

    // 8-byte vectorized BF16 loads (int2 reinterpret of 4x BF16).
    TORCH_CHECK(reinterpret_cast<uintptr_t>(pe.data_ptr()) % 8 == 0,
        "pe.data_ptr() must be 8-byte aligned for vectorized BF16 loads");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(nope.data_ptr()) % 8 == 0,
        "nope.data_ptr() must be 8-byte aligned for vectorized BF16 loads");

    auto const pe_dim = static_cast<int32_t>(pe.size(-1));
    auto const nope_dim = static_cast<int32_t>(nope.size(-1));
    auto const head_dim = pe_dim + nope_dim;
    TORCH_CHECK(head_dim == 128, "head_dim (pe_dim + nope_dim) must be 128, got ", head_dim);

    // cos_sin is [max_pos, pe_dim]: cos first half, sin second half.
    TORCH_CHECK(cos_sin.size(-1) == pe_dim, "cos_sin.size(-1) (", cos_sin.size(-1), ") must equal pe_dim (", pe_dim,
        ") — cos first half, sin second half");

    auto const pe_M = pe.numel() / pe_dim;
    auto const nope_M = nope.numel() / nope_dim;
    TORCH_CHECK(pe_M == nope_M, "pe and nope must have same number of rows. pe: ", pe_M, ", nope: ", nope_M);
    auto const M = static_cast<int32_t>(pe_M);
    TORCH_CHECK(pos.numel() == M, "pos must have M (", M, ") entries, got ", pos.numel());

    auto const pe_row_stride = static_cast<int32_t>(pe.stride(-2));
    auto const nope_row_stride = static_cast<int32_t>(nope.stride(-2));
    auto const cos_sin_stride = static_cast<int32_t>(cos_sin.stride(0));

    at::Tensor packed_out
        = at::detail::empty_cuda({M, head_dim / 2}, at::ScalarType::Char, pe.device(), /* stride */ std::nullopt);
    at::Tensor scale_out = at::detail::empty_cuda({M, 1}, at::ScalarType::Int, pe.device(), /* stride */ std::nullopt);

    auto stream = at::cuda::getCurrentCUDAStream(pe.get_device());

    tensorrt_llm::kernels::invokeFusedRopeCatFp4(reinterpret_cast<int8_t*>(packed_out.data_ptr()),
        reinterpret_cast<int32_t*>(scale_out.data_ptr()), reinterpret_cast<__nv_bfloat16 const*>(pe.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(nope.data_ptr()), cos_sin.data_ptr<float>(), pos.data_ptr<int32_t>(), M,
        pe_dim, nope_dim, head_dim, pe_row_stride, nope_row_stride, cos_sin_stride, stream);

    return {packed_out, scale_out};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("fused_rope_cat_fp4(Tensor pe, Tensor nope, Tensor cos_sin, Tensor pos) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_rope_cat_fp4", &tensorrt_llm::torch_ext::fused_rope_cat_fp4);
}

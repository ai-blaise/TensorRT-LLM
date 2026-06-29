/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/cublasScaledMM.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <string>

namespace th = torch;
namespace tl = tensorrt_llm;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{
template <int kBegin, int kEnd, int kNumExperts, int kHiddenDim>
struct LoopUnroller
{
    static void unroll(
        int num_tokens, float* output, __nv_bfloat16 const* input, __nv_bfloat16 const* weights, cudaStream_t stream)
    {
        if (num_tokens == kBegin)
        {
            tk::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, kBegin, kNumExperts, kHiddenDim>(
                output, input, weights, stream);
        }
        else
        {
            LoopUnroller<kBegin + 1, kEnd, kNumExperts, kHiddenDim>::unroll(num_tokens, output, input, weights, stream);
        }
    }
};

template <int kEnd, int kNumExperts, int kHiddenDim>
struct LoopUnroller<kEnd, kEnd, kNumExperts, kHiddenDim>
{
    static void unroll(
        int num_tokens, float* output, __nv_bfloat16 const* input, __nv_bfloat16 const* weights, cudaStream_t stream)
    {
        if (num_tokens == kEnd)
        {
            tk::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, kEnd, kNumExperts, kHiddenDim>(
                output, input, weights, stream);
        }
        else
        {
            throw std::invalid_argument("Invalid num_tokens, only supports 1 to 16");
        }
    }
};

template <int kNumExperts, int kHiddenDim>
void runRouterGemm(
    int num_tokens, float* output, __nv_bfloat16 const* input, __nv_bfloat16 const* weights, cudaStream_t stream)
{
    LoopUnroller<1, 16, kNumExperts, kHiddenDim>::unroll(num_tokens, output, input, weights, stream);
}

bool routerGemmDebugEnabled()
{
    char const* value = std::getenv("TRTLLM_OPTRT_DSV3_ROUTER_GEMM_DEBUG");
    if (value == nullptr)
    {
        return false;
    }
    std::string flag(value);
    return flag == "1" || flag == "true" || flag == "TRUE" || flag == "on" || flag == "ON" || flag == "yes"
        || flag == "YES";
}

void maybeLogRouterGemmDispatch(th::Tensor const& mat_a, th::Tensor const& mat_b, th::Tensor const& out,
    int num_tokens, int num_experts, int hidden_dim, bool shape_ok, char const* branch)
{
    static std::array<std::atomic<int>, 17> debugPrintsByTokenCount{};
    if (!routerGemmDebugEnabled())
    {
        return;
    }
    int const tokenBucket = std::clamp(num_tokens, 0, 16);
    int const printIdx = debugPrintsByTokenCount[static_cast<size_t>(tokenBucket)].fetch_add(
        1, std::memory_order_relaxed);
    if (printIdx >= 16)
    {
        return;
    }
    std::cout << "[dsv3_router_gemm_dispatch] idx=" << printIdx << " branch=" << branch
              << " num_tokens=" << num_tokens << " num_experts=" << num_experts << " hidden_dim=" << hidden_dim
              << " shape_ok=" << (shape_ok ? 1 : 0) << " mat_a_dtype=" << mat_a.scalar_type()
              << " mat_b_dtype=" << mat_b.scalar_type() << " out_dtype=" << out.scalar_type()
              << " mat_a_sizes=" << mat_a.sizes() << " mat_a_strides=" << mat_a.strides()
              << " mat_b_sizes=" << mat_b.sizes() << " mat_b_strides=" << mat_b.strides()
              << " out_sizes=" << out.sizes() << " out_strides=" << out.strides() << std::endl;
}
} // namespace

th::Tensor& dsv3_router_gemm_op_out(
    th::Tensor const& mat_a, th::Tensor const& mat_b, std::optional<at::Tensor> const& bias, th::Tensor& out)
{
    int const num_tokens = mat_a.sizes()[0];
    int const num_experts = mat_b.sizes()[1];
    int const hidden_dim = mat_a.sizes()[1];
    auto const out_dtype_ = out.scalar_type();
    auto const data_type = mat_a.scalar_type();
    constexpr int kBlaiseNumExperts = 128;
    constexpr int kDsv3NumExperts = 256;
    constexpr int kHiddenDim7168 = 7168; // DeepSeek-V3 / DeepSeek-V3.2
    constexpr int kHiddenDim6144 = 6144; // GLM-5
    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2);
    TORCH_CHECK(out.dim() == 2, "router GEMM output must be 2D");
    TORCH_CHECK(out.sizes()[0] == num_tokens && out.sizes()[1] == num_experts,
        "router GEMM output shape must match [num_tokens, num_experts]");
    TORCH_CHECK(mat_a.strides()[1] == 1 && out.strides()[1] == 1); // Row-major
    TORCH_CHECK(out.strides()[0] == num_experts, "router GEMM output must be row-major contiguous");
    TORCH_CHECK(mat_b.strides()[0] == 1);                          // Column-major
    TORCH_CHECK(!bias.has_value(), "bias is not support yet");
    auto stream = at::cuda::getCurrentCUDAStream(mat_a.get_device());
    bool const shape_ok = (num_tokens >= 1 && num_tokens <= 16
        && mat_b.sizes()[0] == hidden_dim && data_type == torch::kBFloat16 && out_dtype_ == torch::kFloat32);

    char const* branch = "fallback";
    if (shape_ok && num_experts == kBlaiseNumExperts && hidden_dim == kHiddenDim7168)
    {
        branch = "fast_128_7168";
        maybeLogRouterGemmDispatch(mat_a, mat_b, out, num_tokens, num_experts, hidden_dim, shape_ok, branch);
        runRouterGemm<kBlaiseNumExperts, kHiddenDim7168>(num_tokens, reinterpret_cast<float*>(out.mutable_data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(mat_b.data_ptr()), stream);
    }
    else if (shape_ok && num_experts == kDsv3NumExperts && hidden_dim == kHiddenDim7168)
    {
        branch = "fast_256_7168";
        maybeLogRouterGemmDispatch(mat_a, mat_b, out, num_tokens, num_experts, hidden_dim, shape_ok, branch);
        runRouterGemm<kDsv3NumExperts, kHiddenDim7168>(num_tokens, reinterpret_cast<float*>(out.mutable_data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(mat_b.data_ptr()), stream);
    }
    else if (shape_ok && num_experts == kDsv3NumExperts && hidden_dim == kHiddenDim6144)
    {
        branch = "fast_256_6144";
        maybeLogRouterGemmDispatch(mat_a, mat_b, out, num_tokens, num_experts, hidden_dim, shape_ok, branch);
        runRouterGemm<kDsv3NumExperts, kHiddenDim6144>(num_tokens, reinterpret_cast<float*>(out.mutable_data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(mat_b.data_ptr()), stream);
    }
    else // fallback to cublas, can be slow
    {
        maybeLogRouterGemmDispatch(mat_a, mat_b, out, num_tokens, num_experts, hidden_dim, shape_ok, branch);
        cublas_mm_out(mat_a, mat_b, bias, out);
    }

    return out;
}

th::Tensor dsv3_router_gemm_op(th::Tensor const& mat_a, th::Tensor const& mat_b, std::optional<at::Tensor> const& bias,
    std::optional<c10::ScalarType> const& out_dtype)
{
    auto const out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
    std::vector<int64_t> output_size = {mat_a.sizes()[0], mat_b.sizes()[1]};
    th::Tensor out = th::empty(output_size, mat_a.options().dtype(out_dtype_));
    return dsv3_router_gemm_op_out(mat_a, mat_b, bias, out);
}

} // end namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("dsv3_router_gemm_op(Tensor mat_a, Tensor mat_b, Tensor? bias, ScalarType? out_dtype) -> (Tensor out)");
    m.def("dsv3_router_gemm_op_out(Tensor mat_a, Tensor mat_b, Tensor? bias, Tensor(a!) out) -> (Tensor(a!) out)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("dsv3_router_gemm_op", &tensorrt_llm::torch_ext::dsv3_router_gemm_op);
    m.impl("dsv3_router_gemm_op_out", &tensorrt_llm::torch_ext::dsv3_router_gemm_op_out);
}

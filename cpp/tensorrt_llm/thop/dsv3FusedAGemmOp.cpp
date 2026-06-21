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

#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/cublasScaledMM.h"

#include <optional>

namespace th = torch;
namespace tl = tensorrt_llm;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace
{
int getDsv3GateTileM()
{
    auto const tileM = tensorrt_llm::common::getIntEnv("TRTLLM_OPTRT_MLA_GATE_DSV3_TILE_M").value_or(16);
    return (tileM == 32 || tileM == 64 || tileM == 128) ? tileM : 16;
}

int getDsv3GateTileN(int numTokens)
{
    auto const tileN = tensorrt_llm::common::getIntEnv("TRTLLM_OPTRT_MLA_GATE_DSV3_TILE_N").value_or(0);
    if (tileN == 8 || tileN == 16 || tileN == 32)
    {
        return tileN;
    }
    return numTokens <= 8 ? 8 : 16;
}

bool useDsv3GateSingleLaunch()
{
    static bool const enabled = tensorrt_llm::common::getBoolEnv("TRTLLM_OPTRT_MLA_GATE_DSV3_SINGLE_OUTPUT");
    return enabled;
}

template <int kTileM, int kTileN>
void invokeDsv3GateGemm(__nv_bfloat16* out, __nv_bfloat16 const* input, __nv_bfloat16 const* weight, int numTokens,
    int outputLd, cudaStream_t stream)
{
    constexpr int kHdIn = 7168;
    constexpr int kGateHalfOut = 8192;
    constexpr int kGateOut = 2 * kGateHalfOut;
    if (useDsv3GateSingleLaunch())
    {
        tk::dsv3MinLatencyKernels::invokeFusedAGemmStridedTiled<__nv_bfloat16, kHdIn, kGateOut, kTileM, kTileN>(
            out, input, weight, numTokens, outputLd, stream);
        return;
    }

    tk::dsv3MinLatencyKernels::invokeFusedAGemmStridedTiled<__nv_bfloat16, kHdIn, kGateHalfOut, kTileM, kTileN>(
        out, input, weight, numTokens, outputLd, stream);
    tk::dsv3MinLatencyKernels::invokeFusedAGemmStridedTiled<__nv_bfloat16, kHdIn, kGateHalfOut, kTileM, kTileN>(
        out + kGateHalfOut, input, weight + kGateHalfOut * kHdIn, numTokens, outputLd, stream);
}

template <int kTileN>
void invokeDsv3GateGemmForTileM(int tileM, __nv_bfloat16* out, __nv_bfloat16 const* input, __nv_bfloat16 const* weight,
    int numTokens, int outputLd, cudaStream_t stream)
{
    if (tileM == 64)
    {
        invokeDsv3GateGemm<64, kTileN>(out, input, weight, numTokens, outputLd, stream);
    }
    else if (tileM == 128)
    {
        invokeDsv3GateGemm<128, kTileN>(out, input, weight, numTokens, outputLd, stream);
    }
    else if (tileM == 32)
    {
        invokeDsv3GateGemm<32, kTileN>(out, input, weight, numTokens, outputLd, stream);
    }
    else
    {
        invokeDsv3GateGemm<16, kTileN>(out, input, weight, numTokens, outputLd, stream);
    }
}

} // namespace

th::Tensor dsv3_fused_a_gemm_op(th::Tensor const& mat_a, th::Tensor const& mat_b, std::optional<at::Tensor> const& bias,
    std::optional<c10::ScalarType> const& out_dtype)
{
    int const num_tokens = mat_a.sizes()[0];
    int const hd_in = mat_a.sizes()[1];
    int const hd_out = mat_b.sizes()[1];
    // auto const out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
    auto const out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
    auto const data_type = mat_a.scalar_type();
    constexpr int kHdIn = 7168;
    constexpr int kHdOut = 2112;
    std::vector<int64_t> output_size = {num_tokens, hd_out};
    th::Tensor out = th::empty(output_size, mat_a.options().dtype(out_dtype_));

    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2);
    TORCH_CHECK(mat_a.strides()[1] == 1 && out.strides()[1] == 1); // Row-major
    TORCH_CHECK(mat_b.strides()[0] == 1);                          // Column-major
    TORCH_CHECK(!bias.has_value(), "bias is not support yet");
    auto const sm = tensorrt_llm::common::getSMVersion();
    if (sm >= 90)
    {
        bool use_custom_kernel = false;
        if (num_tokens >= 1 && num_tokens <= 16 && hd_in == kHdIn && hd_out == kHdOut && data_type == torch::kBFloat16
            && out_dtype_ == torch::kBFloat16)
        {
            use_custom_kernel = true;
        }
        if (use_custom_kernel)
        {
            auto stream = at::cuda::getCurrentCUDAStream(mat_a.get_device());
            if (num_tokens <= 8)
            {
                tk::dsv3MinLatencyKernels::invokeFusedAGemm<__nv_bfloat16, kHdIn, kHdOut, 8>(
                    reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr()),
                    reinterpret_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
                    reinterpret_cast<__nv_bfloat16 const*>(mat_b.data_ptr()), num_tokens, stream);
            }
            else
            {
                tk::dsv3MinLatencyKernels::invokeFusedAGemm<__nv_bfloat16, kHdIn, kHdOut, 16>(
                    reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr()),
                    reinterpret_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
                    reinterpret_cast<__nv_bfloat16 const*>(mat_b.data_ptr()), num_tokens, stream);
            }
        }
        else
        {
            cublas_mm_out(mat_a, mat_b, bias, out);
        }
    }
    else
    {
        cublas_mm_out(mat_a, mat_b, bias, out);
    }
    return out;
}

th::Tensor dsv3_gate_gemm_op(th::Tensor const& mat_a, th::Tensor const& weight)
{
    int const num_tokens = mat_a.sizes()[0];
    int const hd_in = mat_a.sizes()[1];
    int const hd_out = weight.sizes()[0];
    auto const data_type = mat_a.scalar_type();
    constexpr int kHdIn = 7168;
    constexpr int kGateHalfOut = 8192;
    constexpr int kGateOut = 2 * kGateHalfOut;
    std::vector<int64_t> output_size = {num_tokens, hd_out};
    th::Tensor out = th::empty(output_size, mat_a.options());

    TORCH_CHECK(mat_a.dim() == 2 && weight.dim() == 2);
    TORCH_CHECK(mat_a.strides()[1] == 1 && out.strides()[1] == 1); // Row-major
    TORCH_CHECK(weight.strides()[1] == 1, "gate weight must be row-major contiguous");
    TORCH_CHECK(weight.sizes()[1] == hd_in, "gate weight/input K mismatch");

    auto const sm = tensorrt_llm::common::getSMVersion();
    bool const use_custom_kernel = (sm >= 90 && num_tokens >= 1 && num_tokens <= 64 && hd_in == kHdIn
        && hd_out == kGateOut && data_type == torch::kBFloat16 && weight.scalar_type() == torch::kBFloat16);
    if (use_custom_kernel)
    {
        auto stream = at::cuda::getCurrentCUDAStream(mat_a.get_device());
        auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr());
        auto const* input_ptr = reinterpret_cast<__nv_bfloat16 const*>(mat_a.data_ptr());
        auto const* weight_ptr = reinterpret_cast<__nv_bfloat16 const*>(weight.data_ptr());
        int const tileM = getDsv3GateTileM();
        int const tileN = getDsv3GateTileN(num_tokens);
        if (tileN == 32)
        {
            invokeDsv3GateGemmForTileM<32>(tileM, out_ptr, input_ptr, weight_ptr, num_tokens, hd_out, stream);
        }
        else if (tileN == 8)
        {
            invokeDsv3GateGemmForTileM<8>(tileM, out_ptr, input_ptr, weight_ptr, num_tokens, hd_out, stream);
        }
        else
        {
            invokeDsv3GateGemmForTileM<16>(tileM, out_ptr, input_ptr, weight_ptr, num_tokens, hd_out, stream);
        }
    }
    else
    {
        std::optional<at::Tensor> bias = std::nullopt;
        auto weight_t = weight.t();
        cublas_mm_out(mat_a, weight_t, bias, out);
    }
    return out;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("dsv3_fused_a_gemm_op(Tensor mat_a, Tensor mat_b, Tensor? bias, ScalarType? out_dtype) -> (Tensor out)");
    m.def("dsv3_gate_gemm_op(Tensor mat_a, Tensor weight) -> (Tensor out)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("dsv3_fused_a_gemm_op", &tensorrt_llm::torch_ext::dsv3_fused_a_gemm_op);
    m.impl("dsv3_gate_gemm_op", &tensorrt_llm::torch_ext::dsv3_gate_gemm_op);
}

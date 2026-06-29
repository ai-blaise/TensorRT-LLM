/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

// Registers the native resident DeepSeek decode body entrypoint.
//
// The Python resident backend owns the model/request/KV contract and only calls
// deepseek_resident_decode when deepseek_resident_decode_ready() returns true.
// Until the C++/CUDA body is implemented, the ready guard remains false.

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/DeepseekResidentAttentionMetadata.h"
#include "tensorrt_llm/kernels/DeepseekResidentNorm.h"
#include "tensorrt_llm/kernels/IndexerKCacheScatter.h"
#include "tensorrt_llm/kernels/IndexerTopK.h"
#include "tensorrt_llm/kernels/IndexerXstepRecencyPatch.h"
#include "tensorrt_llm/kernels/fusedRopeCatFp4.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.h"
#include "tensorrt_llm/thop/thUtils.h"

#if defined(TRTLLM_OPTRT_ENABLE_DEEP_GEMM_RESIDENT_INDEXER)
#include "apis/attention.hpp"
#include "jit/compiler.hpp"
#include "jit/include_parser.hpp"
#include "jit/kernel_runtime.hpp"
#endif

#include <ATen/core/Dict.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/copy.h>
#include <c10/core/SymInt.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace th = torch;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

void MLARopeGeneration(torch::Tensor fused_q, torch::Tensor q_pe, torch::Tensor latent_cache,
    std::optional<torch::Tensor> rotary_cos_sin, torch::Tensor cu_q_seqlens, torch::Tensor cu_kv_seqlens,
    torch::Tensor fmha_scheduler_counter, std::optional<torch::Tensor> mla_bmm1_scale,
    std::optional<torch::Tensor> mla_bmm2_scale, std::optional<torch::Tensor> quant_q_buffer,
    torch::Tensor sequence_length, torch::Tensor host_past_key_value_lengths, torch::Tensor host_context_lengths,
    int64_t num_contexts, std::optional<torch::Tensor> kv_cache_block_offsets,
    std::optional<torch::Tensor> host_kv_cache_pool_pointers, std::optional<torch::Tensor> host_kv_cache_pool_mapping,
    std::optional<torch::Tensor> kv_scale_orig_quant, std::optional<torch::Tensor> kv_scale_quant_orig,
    std::optional<torch::Tensor> out_scale, std::optional<torch::Tensor> block_ids_per_seq,
    std::vector<std::optional<torch::Tensor>> helix_tensor_params, int64_t predicted_tokens_per_seq, int64_t layer_idx,
    int64_t num_heads, int64_t num_kv_heads, int64_t head_size, int64_t tokens_per_block, int64_t attention_window_size,
    int64_t beam_width, int64_t quant_mode, double q_scaling, int64_t q_lora_rank, int64_t kv_lora_rank,
    int64_t qk_nope_head_dim, int64_t qk_rope_head_dim, int64_t v_head_dim, bool rope_append);

void attention(torch::Tensor q, std::optional<torch::Tensor> k, std::optional<torch::Tensor> v, torch::Tensor& output,
    std::optional<torch::Tensor> output_sf, std::optional<torch::Tensor> workspace_, torch::Tensor sequence_length,
    torch::Tensor host_past_key_value_lengths, torch::Tensor host_total_kv_lens, torch::Tensor context_lengths,
    torch::Tensor host_context_lengths, torch::Tensor host_request_types,
    std::optional<torch::Tensor> kv_cache_block_offsets, std::optional<torch::Tensor> host_kv_cache_pool_pointers,
    std::optional<torch::Tensor> host_kv_cache_pool_mapping, std::optional<torch::Tensor> cache_indirection,
    std::optional<torch::Tensor> kv_scale_orig_quant, std::optional<torch::Tensor> kv_scale_quant_orig,
    std::optional<torch::Tensor> out_scale, std::optional<torch::Tensor> rotary_inv_freq,
    std::optional<torch::Tensor> rotary_cos_sin, std::optional<torch::Tensor> latent_cache,
    std::optional<torch::Tensor> q_pe, std::optional<torch::Tensor> block_ids_per_seq,
    std::optional<torch::Tensor> attention_sinks, bool is_fused_qkv, bool update_kv_cache,
    int64_t predicted_tokens_per_seq, int64_t local_layer_idx, int64_t num_heads, int64_t num_kv_heads,
    int64_t head_size, std::optional<int64_t> tokens_per_block, int64_t max_num_requests, int64_t max_context_length,
    int64_t attention_window_size, int64_t beam_width, int64_t mask_type, int64_t quant_mode, double q_scaling,
    int64_t position_embedding_type, int64_t rope_dim, double rope_base, int64_t rope_scale_type, double rope_scale,
    double rope_short_m_scale, double rope_long_m_scale, int64_t rope_max_positions,
    int64_t rope_original_max_positions, bool use_paged_context_fmha, std::optional<int64_t> attention_input_type,
    bool is_mla_enable, std::optional<int64_t> chunked_prefill_buffer_batch_size, std::optional<int64_t> q_lora_rank,
    std::optional<int64_t> kv_lora_rank, std::optional<int64_t> qk_nope_head_dim,
    std::optional<int64_t> qk_rope_head_dim, std::optional<int64_t> v_head_dim, std::optional<bool> rope_append,
    std::optional<torch::Tensor> mrope_rotary_cos_sin, std::optional<torch::Tensor> mrope_position_deltas,
    std::optional<torch::Tensor> helix_position_offsets, std::optional<torch::Tensor> helix_is_inactive_rank,
    std::optional<int64_t> attention_chunk_size, std::optional<torch::Tensor> softmax_stats_tensor,
    bool is_spec_decoding_enabled, bool use_spec_decoding, bool is_spec_dec_tree,
    std::optional<torch::Tensor> spec_decoding_generation_lengths,
    std::optional<torch::Tensor> spec_decoding_position_offsets_for_cpp,
    std::optional<torch::Tensor> spec_decoding_packed_mask,
    std::optional<torch::Tensor> spec_decoding_bl_tree_mask_offset,
    std::optional<torch::Tensor> spec_decoding_bl_tree_mask,
    std::optional<torch::Tensor> spec_bl_tree_first_sparse_mask_offset_kv,
    std::optional<torch::Tensor> sparse_kv_indices, std::optional<torch::Tensor> sparse_kv_offsets,
    std::optional<torch::Tensor> sparse_attn_indices, std::optional<torch::Tensor> sparse_attn_offsets,
    int64_t sparse_attn_indices_block_size, std::optional<int64_t> num_sparse_topk,
    std::optional<torch::Tensor> sparse_mla_topk_lens,
    std::optional<double> skip_softmax_threshold_scale_factor_prefill,
    std::optional<double> skip_softmax_threshold_scale_factor_decode, std::optional<torch::Tensor> skip_softmax_stat,
    std::optional<torch::Tensor> cu_q_seqlens, std::optional<torch::Tensor> cu_kv_seqlens,
    std::optional<torch::Tensor> fmha_scheduler_counter, std::optional<torch::Tensor> mla_bmm1_scale,
    std::optional<torch::Tensor> mla_bmm2_scale, std::optional<torch::Tensor> quant_q_buffer,
    std::optional<torch::Tensor> flash_mla_tile_scheduler_metadata, std::optional<torch::Tensor> flash_mla_num_splits,
    int64_t sage_attn_num_elts_per_blk_q, int64_t sage_attn_num_elts_per_blk_k, int64_t sage_attn_num_elts_per_blk_v,
    bool sage_attn_qk_int8, int64_t num_contexts, int64_t num_ctx_tokens,
    std::optional<int64_t> compressed_kv_cache_pool_ptr);

std::tuple<th::Tensor, th::Tensor, th::Tensor, th::Tensor> sparse_mla_decode_nvfp4_vfuse(th::Tensor const& q,
    th::Tensor const& kv, th::Tensor const& kvScales, th::Tensor const& indices, th::Tensor const& vBProj,
    std::optional<th::Tensor> const& topkLength, std::optional<th::Tensor> const& attnSink,
    std::optional<th::Tensor> const& tileSchedulerMetadata, std::optional<th::Tensor> const& numSplits, int64_t dV,
    int64_t vHeadDim, double smScale);
std::tuple<th::Tensor, th::Tensor, th::Tensor, th::Tensor> sparse_mla_decode_nvfp4_vfuse_out(th::Tensor const& q,
    th::Tensor const& kv, th::Tensor const& kvScales, th::Tensor const& indices, th::Tensor const& vBProj,
    th::Tensor const& output, std::optional<th::Tensor> const& topkLength, std::optional<th::Tensor> const& attnSink,
    std::optional<th::Tensor> const& tileSchedulerMetadata, std::optional<th::Tensor> const& numSplits, int64_t dV,
    int64_t vHeadDim, double smScale);

th::Tensor convertReqIndexToGlobal(th::Tensor const& reqId, th::Tensor const& blockTable,
    th::Tensor const& tokenIndices, int64_t blockSize, int64_t numTopkTokens, int64_t strideFactor, int64_t layerId);

using DeepseekResidentMoeRunnerType = tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::MoE::Runner;

std::vector<torch::Tensor> run_fp4_block_scale_moe_runner(torch::optional<torch::Tensor> const& routing_logits,
    torch::optional<torch::Tensor> const& routing_bias, torch::Tensor const& hidden_states,
    torch::optional<torch::Tensor> const& hidden_states_scale, torch::Tensor const& gemm1_weights,
    torch::Tensor const& gemm1_weights_scale, std::optional<torch::Tensor> const& gemm1_bias,
    std::optional<torch::Tensor> const& gemm1_alpha, std::optional<torch::Tensor> const& gemm1_beta,
    std::optional<torch::Tensor> const& gemm1_clamp_limit, torch::Tensor const& gemm2_weights,
    torch::Tensor const& gemm2_weights_scale, std::optional<torch::Tensor> const& gemm2_bias,
    torch::Tensor const& output1_scales_scalar, torch::Tensor const& output1_scales_gate_scalar,
    torch::Tensor const& output2_scales_scalar, int64_t num_experts, int64_t top_k, std::optional<int64_t> n_group,
    std::optional<int64_t> topk_group, int64_t intermediate_size, int64_t local_expert_offset,
    int64_t local_num_experts, std::optional<double> routed_scaling_factor, int64_t tile_tokens_dim,
    int64_t routing_method_type, bool do_finalize, batchedGemm::trtllm::gen::Dtype dtype,
    DeepseekResidentMoeRunnerType& moe_runner, int64_t moeConfigIndex,
    torch::optional<torch::Tensor> const& topk_weights, torch::optional<torch::Tensor> const& topk_ids,
    torch::optional<torch::Tensor> const& out_tensor);

th::Tensor dsv3_router_gemm_op(th::Tensor const& mat_a, th::Tensor const& mat_b, std::optional<at::Tensor> const& bias,
    std::optional<c10::ScalarType> const& out_dtype);
th::Tensor& dsv3_router_gemm_op_out(
    th::Tensor const& mat_a, th::Tensor const& mat_b, std::optional<at::Tensor> const& bias, th::Tensor& out);

th::Tensor BlockScaleInterleaveReverse(th::Tensor const& blockScale);

at::Tensor& cuda_core_nvfp4_gemm_out(at::Tensor const& matA, at::Tensor const& matB, at::Tensor const& scaleA,
    at::Tensor const& scaleB, at::Tensor const& alpha, std::optional<at::Tensor> const& bias, at::Tensor& out);

namespace
{
std::atomic<int64_t> gResidentAttentionTailFp4OutGateVisits{0};

enum class ResidentAttentionTailFp4OutGateStat : size_t
{
    kAttempts,
    kDisabled,
    kInvalidInputTokens,
    kMissingGateWeightScale,
    kMissingGateInputScale,
    kMissingGateAlpha,
    kMissingOProjWeightScale,
    kMissingOProjInputScale,
    kMissingOProjAlpha,
    kShapeRejected,
    kContiguousDtypeRejected,
    kGateProjRejected,
    kFusedQuantRejected,
    kCount
};

constexpr size_t kResidentAttentionTailFp4OutGateStatCount
    = static_cast<size_t>(ResidentAttentionTailFp4OutGateStat::kCount);

std::array<std::atomic<int64_t>, kResidentAttentionTailFp4OutGateStatCount>
    gResidentAttentionTailFp4OutGateStats{};

int64_t residentAttentionTailFp4OutGateVisits()
{
    return gResidentAttentionTailFp4OutGateVisits.load(std::memory_order_relaxed);
}

int64_t residentAttentionTailFp4OutGateStat(ResidentAttentionTailFp4OutGateStat stat)
{
    return gResidentAttentionTailFp4OutGateStats.at(static_cast<size_t>(stat)).load(std::memory_order_relaxed);
}

void recordResidentAttentionTailFp4OutGateStat(ResidentAttentionTailFp4OutGateStat stat)
{
    static_cast<void>(
        gResidentAttentionTailFp4OutGateStats.at(static_cast<size_t>(stat)).fetch_add(1, std::memory_order_relaxed));
}

void recordResidentAttentionTailFp4OutGateVisit()
{
    static_cast<void>(gResidentAttentionTailFp4OutGateVisits.fetch_add(1, std::memory_order_relaxed));
}

bool envFlagEnabled(char const* name)
{
    char const* value = std::getenv(name);
    if (value == nullptr)
    {
        return false;
    }
    std::string flag(value);
    return flag == "1" || flag == "true" || flag == "TRUE" || flag == "on" || flag == "ON" || flag == "yes"
        || flag == "YES";
}

std::optional<bool> optionalEnvFlag(char const* name)
{
    char const* value = std::getenv(name);
    if (value == nullptr)
    {
        return std::nullopt;
    }
    std::string flag(value);
    if (flag == "1" || flag == "true" || flag == "TRUE" || flag == "on" || flag == "ON" || flag == "yes"
        || flag == "YES")
    {
        return true;
    }
    if (flag == "0" || flag == "false" || flag == "FALSE" || flag == "off" || flag == "OFF" || flag == "no"
        || flag == "NO")
    {
        return false;
    }
    return std::nullopt;
}

int64_t envIntOrDefault(char const* name, int64_t defaultValue)
{
    char const* value = std::getenv(name);
    if (value == nullptr)
    {
        return defaultValue;
    }
    try
    {
        return std::stoll(value);
    }
    catch (std::exception const&)
    {
        return defaultValue;
    }
}

int64_t distributedRank()
{
    return envIntOrDefault("RANK", envIntOrDefault("OMPI_COMM_WORLD_RANK", envIntOrDefault("LOCAL_RANK", 0)));
}

std::string envStringOrDefault(char const* name, std::string const& defaultValue)
{
    char const* value = std::getenv(name);
    if (value == nullptr || std::string(value).empty())
    {
        return defaultValue;
    }
    return std::string(value);
}

int64_t padUpToMultiple(int64_t value, int64_t multiple)
{
    TORCH_CHECK(value >= 0, "value to pad must be non-negative");
    TORCH_CHECK(multiple > 0, "padding multiple must be positive");
    return ((value + multiple - 1) / multiple) * multiple;
}

#if defined(TRTLLM_OPTRT_ENABLE_DEEP_GEMM_RESIDENT_INDEXER)
std::filesystem::path existingDeepGemmRootCandidate(std::filesystem::path const& candidate)
{
    if (!candidate.empty() && std::filesystem::exists(candidate / "include" / "deep_gemm"))
    {
        return candidate;
    }
    return {};
}

std::filesystem::path resolveDeepGemmRoot()
{
    if (char const* envRoot = std::getenv("TRTLLM_OPTRT_DEEP_GEMM_ROOT"))
    {
        if (auto const root = existingDeepGemmRootCandidate(envRoot); !root.empty())
        {
            return root;
        }
    }

    std::vector<std::filesystem::path> const candidates
        = {std::filesystem::current_path() / "tensorrt_llm" / "deep_gemm", "/workspace/tensorrt_llm/deep_gemm",
            "/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/deep_gemm",
            "/usr/local/lib/python3.12/dist-packages/tensorrt_llm/deep_gemm",
            "/usr/local/lib/python3.10/dist-packages/tensorrt_llm/deep_gemm",
            "/home/sjpat/TensorRT-LLM/tensorrt_llm/deep_gemm"};
    for (auto const& candidate : candidates)
    {
        if (auto const root = existingDeepGemmRootCandidate(candidate); !root.empty())
        {
            return root;
        }
    }

    TORCH_CHECK(false,
        "Unable to locate DeepGEMM runtime root. Set TRTLLM_OPTRT_DEEP_GEMM_ROOT to the tensorrt_llm/deep_gemm "
        "directory.");
    return {};
}

void ensureDeepGemmResidentIndexerRuntime()
{
    static std::once_flag initOnce;
    std::call_once(initOnce,
        []()
        {
            if (std::getenv("DG_USE_PYTORCH_CUBLASLT_HANDLE") == nullptr)
            {
                static_cast<void>(setenv("DG_USE_PYTORCH_CUBLASLT_HANDLE", "1", 0));
            }

            std::filesystem::path const deepGemmRoot = resolveDeepGemmRoot();
            std::string const cudaHome = envStringOrDefault("CUDA_HOME", "/usr/local/cuda");
            deep_gemm::Compiler::prepare_init(deepGemmRoot.string(), cudaHome);
            deep_gemm::KernelRuntime::prepare_init(cudaHome);
            deep_gemm::IncludeParser::prepare_init(deepGemmRoot.string());
        });
}
#endif

int64_t nextPowerOfTwoClamped(int64_t value, int64_t hardCap)
{
    if (value <= 0 || hardCap <= 0)
    {
        return hardCap;
    }
    if (value > hardCap / 2)
    {
        return hardCap;
    }
    int64_t bucket = 1;
    while (bucket < value && bucket < hardCap)
    {
        bucket <<= 1;
    }
    return std::min(bucket, hardCap);
}

int64_t ceilDivPositive(int64_t value, int64_t divisor)
{
    TORCH_CHECK(value >= 0, "ceilDivPositive value must be non-negative");
    TORCH_CHECK(divisor > 0, "ceilDivPositive divisor must be positive");
    return (value + divisor - 1) / divisor;
}

int64_t liveWindowIndexerMaxKvLen(th::List<int64_t> cachedTokens, int64_t inputTokens, int64_t ownedSteps)
{
    int64_t maxLiveTokens = 0;
    size_t const activeTokens = std::min(cachedTokens.size(), static_cast<size_t>(std::max<int64_t>(inputTokens, 0)));
    for (size_t idx = 0; idx < activeTokens; ++idx)
    {
        maxLiveTokens = std::max(maxLiveTokens, cachedTokens.get(idx));
    }
    // The window appends one token per owned decode step. Keep the score buffer
    // wide enough for every step in the owned window without using the static
    // model max length for short-context requests.
    return maxLiveTokens + std::max<int64_t>(ownedSteps, 1);
}

int64_t liveWindowIndexerLogitsWidth(
    th::List<int64_t> cachedTokens, int64_t inputTokens, int64_t ownedSteps, int64_t hardCap)
{
    int64_t const maxLiveTokens = liveWindowIndexerMaxKvLen(cachedTokens, inputTokens, ownedSteps);
    return nextPowerOfTwoClamped(maxLiveTokens, hardCap);
}

bool envRankEnabled(char const* name)
{
    char const* value = std::getenv(name);
    if (value == nullptr)
    {
        return true;
    }
    std::string const ranks(value);
    if (ranks.empty() || ranks == "all" || ranks == "ALL")
    {
        return true;
    }
    int64_t const rank = distributedRank();
    std::stringstream stream(ranks);
    std::string item;
    while (std::getline(stream, item, ','))
    {
        if (item == "all" || item == "ALL")
        {
            return true;
        }
        try
        {
            if (std::stoll(item) == rank)
            {
                return true;
            }
        }
        catch (std::exception const&)
        {
        }
    }
    return false;
}

using ResidentWindowTimingClock = std::chrono::steady_clock;

double residentWindowElapsedUs(
    ResidentWindowTimingClock::time_point const start, ResidentWindowTimingClock::time_point const stop)
{
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count()) / 1000.0;
}

bool residentWindowCppTimingEnabled()
{
    return envFlagEnabled("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_CPP_TIMING")
        && envRankEnabled("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_CPP_TIMING_RANKS");
}

bool residentWindowCudaEventTimingEnabled()
{
    return envFlagEnabled("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_CUDA_EVENT_TIMING")
        && envRankEnabled("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_CUDA_EVENT_TIMING_RANKS");
}

bool isStreamCapturing(cudaStream_t stream)
{
    cudaStreamCaptureStatus status{cudaStreamCaptureStatusNone};
    AT_CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    return status != cudaStreamCaptureStatusNone;
}

bool residentWindowScratchCacheEnabled()
{
    char const* value = std::getenv("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_SCRATCH_CACHE");
    if (value == nullptr)
    {
        return false;
    }
    std::string flag(value);
    return flag == "1" || flag == "true" || flag == "TRUE" || flag == "on" || flag == "ON" || flag == "yes"
        || flag == "YES";
}

int64_t residentIndexerStepFreq(int64_t configValue)
{
    int64_t const value
        = envIntOrDefault("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_INDEXER_STEP_FREQ", std::max<int64_t>(configValue, 1));
    return std::max<int64_t>(value, 1);
}

bool residentIndexerStepRecencyPatchEnabled(bool configValue)
{
    std::optional<bool> const envValue = optionalEnvFlag("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_INDEXER_STEP_RECENCY_PATCH");
    return envValue.value_or(configValue);
}

int64_t residentIndexerHisaMinSeqLen(int64_t configValue)
{
    return std::max<int64_t>(
        envIntOrDefault("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_HISA_MIN_SEQ_LEN", std::max<int64_t>(configValue, 0)), 0);
}

bool residentFusedQbWqBEnabled()
{
    return envFlagEnabled("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_FUSED_QB_WQB");
}

bool residentFusedKvAWkWpEnabled()
{
    return envFlagEnabled("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_FUSED_KVA_WKWP");
}

struct ResidentWindowTimingStats
{
    double contractUs{0.0};
    double planValidateUs{0.0};
    double dimensionScanUs{0.0};
    double scratchAllocUs{0.0};
    double prepareUs{0.0};
    double inputNormUs{0.0};
    double attentionProjectionUs{0.0};
    double metadataRefreshUs{0.0};
    double indexerProjectionUs{0.0};
    double indexerScatterUs{0.0};
    double indexerPackUs{0.0};
    double indexerLogitsTopkUs{0.0};
    double metadataListUs{0.0};
    double attentionDispatchUs{0.0};
    double attentionTailUs{0.0};
    double postAttentionNormUs{0.0};
    double denseMlpUs{0.0};
    double moeRouterUs{0.0};
    double moeExpertsUs{0.0};
    double moeSharedExpertUs{0.0};
    double moeRoutedExpertUs{0.0};
    double moeCombineUs{0.0};
    double postFfnNormUs{0.0};
    double lmHeadUs{0.0};
    double sampleUs{0.0};
    int64_t decodeSteps{0};
    int64_t layerVisits{0};
    int64_t denseLayerVisits{0};
    int64_t moeLayerVisits{0};
    int64_t indexerTopkComputeLayerVisits{0};
    int64_t indexerTopkReuseLayerVisits{0};
    int64_t indexerTopkXstepReuseLayerVisits{0};
    int64_t indexerTopkDenseLayerVisits{0};
    int64_t indexerTopkHisaLayerVisits{0};
    int64_t indexerLogitsWidth{0};
    int64_t indexerMaxSeqLen{0};
};

struct ResidentWindowCudaEventRecord
{
    cudaEvent_t start{};
    cudaEvent_t stop{};
    double* bucket{nullptr};
};

void finalizeResidentWindowCudaEventTiming(std::vector<ResidentWindowCudaEventRecord>& records)
{
    for (auto& record : records)
    {
        AT_CUDA_CHECK(cudaEventSynchronize(record.stop));
        float elapsedMs{0.0F};
        AT_CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, record.start, record.stop));
        if (record.bucket != nullptr)
        {
            *record.bucket += static_cast<double>(elapsedMs) * 1000.0;
        }
        AT_CUDA_CHECK(cudaEventDestroy(record.start));
        AT_CUDA_CHECK(cudaEventDestroy(record.stop));
    }
    records.clear();
}

struct ResidentWindowScopedTimer
{
    ResidentWindowScopedTimer(bool enabled, double& bucket)
        : mEnabled(enabled)
        , mBucket(bucket)
        , mStart(ResidentWindowTimingClock::now())
    {
    }

    ~ResidentWindowScopedTimer()
    {
        if (mEnabled)
        {
            mBucket += residentWindowElapsedUs(mStart, ResidentWindowTimingClock::now());
        }
    }

    bool const mEnabled;
    double& mBucket;
    ResidentWindowTimingClock::time_point const mStart;
};

struct ResidentWindowScopedCudaEventTimer
{
    ResidentWindowScopedCudaEventTimer(
        bool enabled, double& bucket, std::vector<ResidentWindowCudaEventRecord>& records, cudaStream_t stream)
        : mEnabled(enabled)
        , mBucket(bucket)
        , mRecords(records)
        , mStream(stream)
    {
        if (mEnabled)
        {
            AT_CUDA_CHECK(cudaEventCreate(&mStart));
            AT_CUDA_CHECK(cudaEventCreate(&mStop));
            AT_CUDA_CHECK(cudaEventRecord(mStart, mStream));
        }
    }

    ~ResidentWindowScopedCudaEventTimer()
    {
        if (mEnabled)
        {
            cudaError_t const err = cudaEventRecord(mStop, mStream);
            if (err == cudaSuccess)
            {
                mRecords.push_back(ResidentWindowCudaEventRecord{mStart, mStop, &mBucket});
            }
            else
            {
                static_cast<void>(cudaEventDestroy(mStart));
                static_cast<void>(cudaEventDestroy(mStop));
            }
        }
    }

    bool const mEnabled;
    double& mBucket;
    std::vector<ResidentWindowCudaEventRecord>& mRecords;
    cudaStream_t const mStream;
    cudaEvent_t mStart{};
    cudaEvent_t mStop{};
};

struct ResidentWindowScopedStageTimer
{
    ResidentWindowScopedStageTimer(bool hostEnabled, double& hostBucket, bool deviceEnabled, double& deviceBucket,
        std::vector<ResidentWindowCudaEventRecord>& records, cudaStream_t stream)
        : hostTimer(hostEnabled, hostBucket)
        , deviceTimer(deviceEnabled, deviceBucket, records, stream)
    {
    }

    ResidentWindowScopedTimer hostTimer;
    ResidentWindowScopedCudaEventTimer deviceTimer;
};

struct ResidentMoeExpertTiming
{
    double sharedExpertUs{0.0};
    double routedExpertUs{0.0};
    double combineUs{0.0};
};

constexpr int64_t kDsaKvDispatchDenseNvfp4 = 0;
constexpr int64_t kDsaKvDispatchStandardMla = 1;
constexpr int64_t kTrtllmGenDeepSeekV3Routing = 2;
constexpr int64_t kTrtllmGenSwiGlu = 0;

bool warpDecodeFixedTacticEnabled()
{
    char const* value = std::getenv("TRTLLM_WARP_DECODE_FIXED_TACTIC");
    if (value == nullptr)
    {
        return true;
    }
    std::string flag(value);
    return !(flag == "0" || flag == "false" || flag == "FALSE" || flag == "off" || flag == "OFF" || flag == "no"
        || flag == "NO");
}

bool residentFusedPostAttentionGateEnabled()
{
    return optionalEnvFlag("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_FUSED_POST_ATTENTION_GATE").value_or(true);
}

bool residentMoeRawRoutingEnabled()
{
    char const* value = std::getenv("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MOE_RAW_ROUTING");
    if (value == nullptr)
    {
        return false;
    }
    std::string flag(value);
    return flag == "1" || flag == "true" || flag == "TRUE" || flag == "on" || flag == "ON" || flag == "yes"
        || flag == "YES";
}

bool residentMoeRouterGemmEnabled()
{
    char const* value = std::getenv("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MOE_ROUTER_GEMM");
    if (value == nullptr)
    {
        return false;
    }
    std::string flag(value);
    return flag == "1" || flag == "true" || flag == "TRUE" || flag == "on" || flag == "ON" || flag == "yes"
        || flag == "YES";
}

std::optional<std::pair<int64_t, int64_t>> residentMoeTacticOverride()
{
    char const* value = std::getenv("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MOE_TACTIC");
    if (value == nullptr || value[0] == '\0')
    {
        return std::nullopt;
    }

    std::string spec(value);
    std::replace(spec.begin(), spec.end(), ':', ',');
    std::istringstream stream(spec);
    int64_t tileTokensDim{-1};
    int64_t configIndex{-1};
    char comma{'\0'};
    if ((stream >> tileTokensDim >> comma >> configIndex) && comma == ',')
    {
        return std::make_pair(tileTokensDim, configIndex);
    }

    TORCH_CHECK(false, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MOE_TACTIC must be '<tile_tokens_dim>,<config_index>'");
    return std::nullopt;
}

bool residentMoeTacticDebugEnabled()
{
    return optionalEnvFlag("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MOE_TACTIC_DEBUG").value_or(false);
}

int64_t nextPowerOfTwoInt64(int64_t value)
{
    if (value <= 1)
    {
        return 1;
    }
    int64_t result = 1;
    while (result < value)
    {
        result *= 2;
    }
    return result;
}
} // namespace

enum class DeepseekResidentLayerKind
{
    kDense = 0,
    kMoe = 1,
};

enum class DeepseekResidentTensorSlot
{
    kEmbedding = 0,
    kFinalNorm = 1,
    kLmHead = 2,
};

enum class DeepseekResidentLayerTensorSite
{
    kInputLayernorm = 0,
    kPostAttentionLayernorm = 1,
    kNextLayerLayernorm = 2,
    kInputGatedNormDown = 3,
    kInputGatedNormUp = 4,
    kPostAttentionGatedNormDown = 5,
    kPostAttentionGatedNormUp = 6,
    kAttentionKvAProjWeight = 7,
    kAttentionKvAProjWeightScale = 8,
    kAttentionKvAProjWeightScale2 = 9,
    kAttentionKvAProjInputScale = 10,
    kAttentionKvAProjInvInputScale = 11,
    kAttentionKvBProjWeight = 12,
    kAttentionKvBProjWeightScale = 13,
    kAttentionKvBProjWeightScale2 = 14,
    kAttentionOProjWeight = 15,
    kAttentionOProjWeightScale = 16,
    kAttentionOProjWeightScale2 = 17,
    kAttentionOProjInputScale = 18,
    kAttentionGateProjWeight = 19,
    kAttentionGateProjWeightScale = 20,
    kAttentionGateProjWeightScale2 = 21,
    kAttentionGateProjInputScale = 22,
    kAttentionKBProjTrans = 23,
    kAttentionKBProjTransScale = 24,
    kAttentionKBProjTransDequant = 25,
    kAttentionVBProj = 26,
    kAttentionVBProjScale = 27,
    kAttentionVBProjDequant = 28,
    kMoeGateWeight = 30,
    kMoeGateEScoreCorrectionBias = 31,
    kDenseMlpGateUpWeight = 40,
    kDenseMlpGateUpWeightScale = 41,
    kDenseMlpGateUpWeightScale2 = 42,
    kDenseMlpGateUpInputScale = 43,
    kDenseMlpDownWeight = 44,
    kDenseMlpDownWeightScale = 45,
    kDenseMlpDownWeightScale2 = 46,
    kDenseMlpDownInputScale = 47,
    kSharedExpertGateUpWeight = 50,
    kSharedExpertGateUpWeightScale = 51,
    kSharedExpertGateUpWeightScale2 = 52,
    kSharedExpertGateUpInputScale = 53,
    kSharedExpertDownWeight = 54,
    kSharedExpertDownWeightScale = 55,
    kSharedExpertDownWeightScale2 = 56,
    kSharedExpertDownInputScale = 57,
    kExpertGateUpWeight = 60,
    kExpertGateUpWeightScale = 61,
    kExpertGateUpInputScale = 62,
    kExpertDownWeight = 63,
    kExpertDownWeightScale = 64,
    kExpertDownInputScale = 65,
    kAttentionKvAProjAlpha = 66,
    kAttentionKvBProjAlpha = 67,
    kAttentionOProjAlpha = 68,
    kAttentionGateProjAlpha = 69,
    kDenseMlpGateUpAlpha = 70,
    kDenseMlpDownAlpha = 71,
    kSharedExpertGateUpAlpha = 72,
    kSharedExpertDownAlpha = 73,
    kExpertGateUpAlpha = 74,
    kExpertDownAlpha = 75,
    kAttentionQALayernormWeight = 76,
    kAttentionKvALayernormWeight = 77,
    kAttentionQBProjWeight = 78,
    kAttentionQBProjWeightScale = 79,
    kAttentionQBProjWeightScale2 = 80,
    kAttentionQBProjInputScale = 81,
    kAttentionQBProjAlpha = 82,
    kAttentionIndexerWQBWeight = 83,
    kAttentionIndexerWQBWeightScale = 84,
    kAttentionIndexerWQBWeightScale2 = 85,
    kAttentionIndexerWQBInputScale = 86,
    kAttentionIndexerWQBAlpha = 87,
    kAttentionIndexerWKWeight = 88,
    kAttentionIndexerWKWeightScale = 89,
    kAttentionIndexerWKWeightScale2 = 90,
    kAttentionIndexerWKInputScale = 91,
    kAttentionIndexerWKAlpha = 92,
    kAttentionIndexerWeightsProjWeight = 93,
    kAttentionIndexerWeightsProjWeightScale = 94,
    kAttentionIndexerWeightsProjWeightScale2 = 95,
    kAttentionIndexerWeightsProjInputScale = 96,
    kAttentionIndexerWeightsProjAlpha = 97,
    kAttentionIndexerKNormWeight = 98,
    kAttentionIndexerKNormBias = 99,
    kAttentionIndexerRotaryCosSin = 100,
    kExpertGateUpOutputScale = 101,
};

struct DeepseekResidentManifest
{
    int64_t nbLayers;
    int64_t nbTensors;
};

struct DeepseekResidentRequest
{
    int64_t realBatchSize;
    int64_t paddedBatchSize;
    int64_t inputTokens;
};

struct DeepseekResidentFusedQbWqBCache
{
    bool eligible{false};
    at::Tensor weight;
    at::Tensor weightScale;
    at::Tensor inputScale;
    at::Tensor alpha;
    int64_t qbOut{0};
    int64_t wqOut{0};
    double wqOutScale{1.0};
    at::ScalarType outDtype{at::ScalarType::BFloat16};
};

struct DeepseekResidentFusedKvAWkWpCache
{
    bool eligible{false};
    at::Tensor weight;
    at::Tensor weightScale;
    at::Tensor inputScale;
    at::Tensor alpha;
    int64_t kvOut{0};
    int64_t kvOutPadded{0};
    int64_t wkOut{0};
    int64_t wpOut{0};
    double wkOutScale{1.0};
    double wpOutScale{1.0};
    at::ScalarType outDtype{at::ScalarType::BFloat16};
};

struct DeepseekResidentDsaAttentionProjectionResult
{
    at::Tensor qScratch;
    at::Tensor compressedKvScratch;
    at::Tensor kPeScratch;
    at::Tensor latentCacheScratch;
    at::Tensor qLoraScratch;
    bool hasPrecomputedWqB{false};
    double precomputedWqBScale{1.0};
    bool hasPrecomputedIndexerKWeights{false};
};

struct DeepseekResidentCompiledDsaWindowLayer
{
    int64_t layerIdx{0};
    bool isDense{false};
    c10::Dict<std::string, at::Tensor> runtimeTensors;
    c10::Dict<std::string, int64_t> runtimeConfig;
    c10::Dict<std::string, double> runtimeScalars;
    c10::List<at::Tensor> metadataTensors;
    c10::List<at::Tensor> dispatchScratchTensors;
    at::Tensor fusedQScratch;
    at::Tensor latentOutputScratch;
    at::Tensor cuQSeqLensScratch;
    at::Tensor cuKvSeqLensScratch;
    at::Tensor fmhaSchedulerCounterScratch;
    std::optional<at::Tensor> mlaBmm1ScaleScratch;
    std::optional<at::Tensor> mlaBmm2ScaleScratch;
    std::optional<at::Tensor> quantQBufferScratch;
    at::Tensor kBProjTrans;
    at::Tensor vBProj;
    at::Tensor blockTable;
    at::Tensor indexerKCache;
    at::Tensor indexerKCacheBlockOffsets;
    at::Tensor schedulerMetadataBuffer;
    at::Tensor slotMappingFp8;
    at::Tensor slotMappingScale;
    at::Tensor genKvIndptr;
    at::Tensor genCachedTokenIndptr;
    at::Tensor kvLensCuda2d;
    at::Tensor reqIdxPerToken;
    std::optional<at::Tensor> indexerHisaPageReps;
    std::optional<at::Tensor> indexerHisaPageCounts;
    at::Tensor qFp4Scratch;
    at::Tensor kFp4Scratch;
    at::Tensor kScaleScratch;
    at::Tensor indexerWeightsScratch;
    at::Tensor qScaleScratch;
    at::Tensor topkIndices;
    at::Tensor xstepRefreshEnd;
    at::Tensor rotaryCosSin;
    std::optional<at::Tensor> topkIndicesPoolRuntime;
    at::Tensor kvCacheBlockOffsets;
    at::Tensor hostKvCachePoolPointers;
    at::Tensor hostKvCachePoolMapping;
    at::Tensor kvLensRuntime;
    at::Tensor promptLensCpuRuntime;
    at::Tensor denseKvPool;
    at::Tensor denseKvScalePool;
    at::Tensor denseKvPacked;
    at::Tensor denseKvScales;
    std::optional<at::Tensor> blockIdsPerSeq;
    std::optional<at::Tensor> helixPositionOffsets;
    std::optional<at::Tensor> helixIsInactiveRank;
    std::optional<at::Tensor> attentionWorkspace;
    std::optional<at::Tensor> hostTotalKvLens;
    std::optional<at::Tensor> promptLensCudaRuntime;
    std::optional<at::Tensor> hostRequestTypesRuntime;
    std::optional<at::Tensor> cacheIndirection;
    std::optional<at::Tensor> kvScaleOrigQuant;
    std::optional<at::Tensor> kvScaleQuantOrig;
    std::optional<at::Tensor> flashMlaTileSchedulerMetadata;
    std::optional<at::Tensor> flashMlaNumSplits;
    int64_t numSeqs{0};
    int64_t numContexts{0};
    int64_t numCtxTokens{0};
    int64_t numGenerations{0};
    int64_t numSparseTopk{0};
    int64_t numHeads{0};
    int64_t qLoraRank{0};
    int64_t kvLoraRank{0};
    int64_t qkNopeHeadDim{0};
    int64_t qkRopeHeadDim{0};
    int64_t vHeadDim{0};
    int64_t kvDispatchMode{0};
    int64_t maxSeqLen{0};
    int64_t beamWidth{1};
    int64_t localLayerIdx{0};
    int64_t predictedTokensPerSeq{1};
    int64_t numKvHeads{1};
    int64_t headDim{0};
    int64_t quantMode{0};
    int64_t ropeAppend{0};
    int64_t maxNumRequests{0};
    int64_t maxContextLength{0};
    int64_t attentionWindowSize{0};
    int64_t sparseAttnIndicesBlockSize{1};
    int64_t maskType{0};
    int64_t positionEmbeddingType{0};
    int64_t ropeDim{0};
    int64_t ropeScaleType{0};
    int64_t ropeMaxPositions{0};
    int64_t ropeOriginalMaxPositions{0};
    int64_t attentionChunkSize{0};
    int64_t usePagedContextFmha{0};
    int64_t tokensPerBlock{0};
    int64_t residentIndexerHeadDim{0};
    int64_t residentIndexerNumHeads{0};
    int64_t residentIndexerRopeDim{0};
    int64_t residentIndexerQuantBlockSize{0};
    int64_t residentIndexerDataBytesPerToken{0};
    bool reusePreviousIndexerTopk{false};
    int64_t residentIndexerStepFreq{1};
    bool residentIndexerStepRecencyPatch{false};
    bool residentIndexerHisaEnabled{false};
    int64_t residentIndexerHisaBlockSize{128};
    int64_t residentIndexerHisaBlockTopK{64};
    int64_t residentIndexerHisaMinSeqLen{0};
    double residentIndexerHisaCompressionRatio{4.0};
    int64_t attentionSfVecSize{16};
    int64_t mlpSfVecSize{16};
    double qScaling{1.0};
    double softmaxScale{1.0};
    double ropeBase{10000.0};
    double ropeScale{1.0};
    double ropeShortMScale{1.0};
    double ropeLongMScale{1.0};
    double rmsNormEps{1e-6};
    double residentIndexerWeightScaleFactor{1.0};
    int64_t qWidth{0};
    int64_t dispatchScratchCount{0};
    int64_t indexerScratchStart{0};
    int64_t denseIntermediateSize{0};
    int64_t moeTopK{0};
    int64_t moeNGroup{0};
    int64_t moeTopkGroup{0};
    double moeRoutedScalingFactor{1.0};
    double moeSharedOutputScale{1.0};
    int64_t moeNumExperts{0};
    int64_t moeLocalExpertOffset{0};
    int64_t moeLocalNumExperts{0};
    int64_t moeIntermediateSize{0};
    int64_t moeSharedIntermediateSize{0};
};

struct DeepseekResidentWindowScratchKey
{
    int64_t deviceIndex{-1};
    int64_t scalarType{-1};
    int64_t maxBatchSize{0};
    int64_t hiddenSize{0};
    int64_t maxQWidth{0};
    int64_t maxQLoraRank{0};
    int64_t maxKvLoraRank{0};
    int64_t maxRopeDim{0};
    int64_t maxKvAWidth{0};
    int64_t maxLatentWidth{0};
    int64_t maxAttentionCoreWidth{0};
    int64_t maxDenseIntermediateWidth{0};
    int64_t maxMoeSharedIntermediateWidth{0};
    int64_t maxMoeExperts{0};
    int64_t maxMoeTopK{0};
};

bool operator==(DeepseekResidentWindowScratchKey const& lhs, DeepseekResidentWindowScratchKey const& rhs)
{
    return lhs.deviceIndex == rhs.deviceIndex && lhs.scalarType == rhs.scalarType
        && lhs.maxBatchSize == rhs.maxBatchSize && lhs.hiddenSize == rhs.hiddenSize && lhs.maxQWidth == rhs.maxQWidth
        && lhs.maxQLoraRank == rhs.maxQLoraRank && lhs.maxKvLoraRank == rhs.maxKvLoraRank
        && lhs.maxRopeDim == rhs.maxRopeDim && lhs.maxKvAWidth == rhs.maxKvAWidth
        && lhs.maxLatentWidth == rhs.maxLatentWidth && lhs.maxAttentionCoreWidth == rhs.maxAttentionCoreWidth
        && lhs.maxDenseIntermediateWidth == rhs.maxDenseIntermediateWidth
        && lhs.maxMoeSharedIntermediateWidth == rhs.maxMoeSharedIntermediateWidth
        && lhs.maxMoeExperts == rhs.maxMoeExperts && lhs.maxMoeTopK == rhs.maxMoeTopK;
}

struct DeepseekResidentWindowScratch
{
    DeepseekResidentWindowScratchKey key;
    bool valid{false};
    at::Tensor normHiddenScratch;
    at::Tensor gatedHiddenScratch;
    at::Tensor attentionHiddenScratch;
    at::Tensor postAttentionNormScratch;
    at::Tensor postAttentionGatedScratch;
    at::Tensor postAttentionResidualScratch;
    at::Tensor denseMlpOutputScratch;
    at::Tensor nextLayerHiddenScratch;
    at::Tensor nextLayerResidualScratch;
    at::Tensor qScratchStorage;
    at::Tensor kvAScratchStorage;
    at::Tensor qLoraScratchStorage;
    at::Tensor compressedKvScratchStorage;
    at::Tensor kPeScratchStorage;
    at::Tensor latentCacheScratchStorage;
    at::Tensor attentionCoreOutputScratchStorage;
    at::Tensor attentionGateScratchStorage;
    at::Tensor attentionGateLogitsScratchStorage;
    at::Tensor denseMlpIntermediateScratchStorage;
    at::Tensor denseMlpGateUpScratchStorage;
    at::Tensor moeSharedIntermediateScratchStorage;
    at::Tensor moeSharedGateUpScratchStorage;
    at::Tensor moeSharedOutputScratchStorage;
    at::Tensor routerLogitsScratchStorage;
    at::Tensor routerScoresScratchStorage;
    at::Tensor routerTopkIndicesScratchStorage;
    at::Tensor routerTopkWeightsScratchStorage;
};

at::Tensor getRequiredTensor(c10::Dict<std::string, at::Tensor> const& tensors, std::string const& name)
{
    auto const iter = tensors.find(name);
    TORCH_CHECK(iter != tensors.end(), "missing DSA runtime tensor: ", name);
    return iter->value();
}

std::optional<at::Tensor> getOptionalTensor(c10::Dict<std::string, at::Tensor> const& tensors, std::string const& name)
{
    auto const iter = tensors.find(name);
    if (iter == tensors.end())
    {
        return std::nullopt;
    }
    return iter->value();
}

int64_t getRequiredInt(c10::Dict<std::string, int64_t> const& config, std::string const& name)
{
    auto const iter = config.find(name);
    TORCH_CHECK(iter != config.end(), "missing DSA runtime config: ", name);
    return iter->value();
}

double getRequiredDouble(c10::Dict<std::string, double> const& scalars, std::string const& name)
{
    auto const iter = scalars.find(name);
    TORCH_CHECK(iter != scalars.end(), "missing DSA runtime scalar: ", name);
    return iter->value();
}

int64_t getOptionalInt(c10::Dict<std::string, int64_t> const& config, std::string const& name, int64_t defaultValue)
{
    auto const iter = config.find(name);
    if (iter == config.end())
    {
        return defaultValue;
    }
    return iter->value();
}

double getOptionalDouble(c10::Dict<std::string, double> const& scalars, std::string const& name, double defaultValue)
{
    auto const iter = scalars.find(name);
    if (iter == scalars.end())
    {
        return defaultValue;
    }
    return iter->value();
}

void checkCudaRuntimeTensor(at::Tensor const& tensor, std::string const& name)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.numel() > 0, name, " must not be empty");
}

void checkRuntimeTensor(at::Tensor const& tensor, std::string const& name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.numel() > 0, name, " must not be empty");
}

std::tuple<at::Tensor, at::Tensor> callTrtllmFp4Quantize(
    at::Tensor const& input, at::Tensor const& inputScale, int64_t sfVecSize, bool isSfSwizzledLayout = true)
{
    TORCH_CHECK(input.is_cuda(), "NVFP4 input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "NVFP4 input must be 2D");
    TORCH_CHECK(sfVecSize > 0, "NVFP4 scaling vector size must be positive");
    if (inputScale.defined())
    {
        TORCH_CHECK(inputScale.is_cuda(), "NVFP4 input scale must be a CUDA tensor");
        TORCH_CHECK(inputScale.numel() > 0, "NVFP4 input scale must not be empty");
    }

    c10::Stack stack;
    stack.emplace_back(input);
    if (inputScale.defined())
    {
        stack.emplace_back(inputScale);
    }
    else
    {
        stack.emplace_back();
    }
    stack.emplace_back(sfVecSize);
    stack.emplace_back(false);
    stack.emplace_back(isSfSwizzledLayout);

    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("trtllm::fp4_quantize", "");
    op.callBoxed(&stack);
    if (stack.size() == 1 && stack.back().isTuple())
    {
        auto const& elements = stack.back().toTupleRef().elements();
        TORCH_CHECK(elements.size() == 2, "trtllm::fp4_quantize returned an invalid tuple");
        return {elements[0].toTensor(), elements[1].toTensor()};
    }
    TORCH_CHECK(stack.size() == 2, "trtllm::fp4_quantize returned an invalid boxed result");
    return {stack[0].toTensor(), stack[1].toTensor()};
}

std::optional<std::tuple<at::Tensor, at::Tensor>> tryCallFusedSigmoidMulQuantNvfp4Swizzled(
    at::Tensor const& input, at::Tensor const& gate, at::Tensor const& inputScale, int64_t sfVecSize)
{
    if (!input.defined() || !gate.defined() || !inputScale.defined() || !input.is_cuda() || !gate.is_cuda()
        || !inputScale.is_cuda() || input.dim() != 2 || gate.dim() != 2 || input.sizes() != gate.sizes()
        || input.scalar_type() != gate.scalar_type() || (input.scalar_type() != at::ScalarType::Half
            && input.scalar_type() != at::ScalarType::BFloat16)
        || !input.is_contiguous() || !gate.is_contiguous() || sfVecSize <= 0)
    {
        return std::nullopt;
    }

    try
    {
        c10::Stack stack;
        stack.emplace_back(input);
        stack.emplace_back(gate);
        stack.emplace_back(inputScale);
        stack.emplace_back(sfVecSize);

        static auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
            "trtllm::fused_sigmoid_mul_quant_nvfp4_swizzled", "");
        op.callBoxed(&stack);
        if (stack.size() == 1 && stack.back().isTuple())
        {
            auto const& elements = stack.back().toTupleRef().elements();
            TORCH_CHECK(
                elements.size() == 2, "trtllm::fused_sigmoid_mul_quant_nvfp4_swizzled returned an invalid tuple");
            return std::make_tuple(elements[0].toTensor(), elements[1].toTensor());
        }
        TORCH_CHECK(stack.size() == 2,
            "trtllm::fused_sigmoid_mul_quant_nvfp4_swizzled returned an invalid boxed result");
        return std::make_tuple(stack[0].toTensor(), stack[1].toTensor());
    }
    catch (c10::Error const&)
    {
        return std::nullopt;
    }
}

void invokeFusedRopeCatFp4Into(at::Tensor const& packedOut, at::Tensor const& scaleOut, at::Tensor const& pe,
    at::Tensor const& nope, at::Tensor const& cosSin, at::Tensor const& positionIds)
{
    TORCH_CHECK(packedOut.is_cuda(), "fused_rope_cat_fp4 packed output must be a CUDA tensor");
    TORCH_CHECK(scaleOut.is_cuda(), "fused_rope_cat_fp4 scale output must be a CUDA tensor");
    TORCH_CHECK(pe.is_cuda(), "fused_rope_cat_fp4 pe must be a CUDA tensor");
    TORCH_CHECK(nope.is_cuda(), "fused_rope_cat_fp4 nope must be a CUDA tensor");
    TORCH_CHECK(cosSin.is_cuda(), "fused_rope_cat_fp4 cos_sin must be a CUDA tensor");
    TORCH_CHECK(positionIds.is_cuda(), "fused_rope_cat_fp4 position ids must be a CUDA tensor");
    TORCH_CHECK(packedOut.device() == pe.device() && scaleOut.device() == pe.device() && nope.device() == pe.device()
            && cosSin.device() == pe.device() && positionIds.device() == pe.device(),
        "fused_rope_cat_fp4 tensors must be on the same CUDA device");
    TORCH_CHECK(packedOut.scalar_type() == at::ScalarType::Char || packedOut.scalar_type() == at::ScalarType::Byte,
        "fused_rope_cat_fp4 packed output must be int8 or uint8");
    TORCH_CHECK(scaleOut.scalar_type() == at::ScalarType::Int, "fused_rope_cat_fp4 scale output must be int32");
    TORCH_CHECK(pe.scalar_type() == at::ScalarType::BFloat16, "fused_rope_cat_fp4 pe must be BF16");
    TORCH_CHECK(nope.scalar_type() == at::ScalarType::BFloat16, "fused_rope_cat_fp4 nope must be BF16");
    TORCH_CHECK(cosSin.scalar_type() == at::ScalarType::Float, "fused_rope_cat_fp4 cos_sin must be FP32");
    TORCH_CHECK(positionIds.scalar_type() == at::ScalarType::Int, "fused_rope_cat_fp4 position ids must be int32");
    TORCH_CHECK(pe.dim() == 2 && nope.dim() == 2, "fused_rope_cat_fp4 pe/nope must be 2D views");
    TORCH_CHECK(cosSin.dim() == 2, "fused_rope_cat_fp4 cos_sin must be 2D");
    TORCH_CHECK(positionIds.dim() == 1, "fused_rope_cat_fp4 position ids must be 1D");
    TORCH_CHECK(pe.stride(1) == 1 && nope.stride(1) == 1 && cosSin.stride(1) == 1,
        "fused_rope_cat_fp4 inputs must have contiguous innermost dimensions");
    TORCH_CHECK(packedOut.is_contiguous(), "fused_rope_cat_fp4 packed output scratch must be contiguous");
    TORCH_CHECK(scaleOut.is_contiguous(), "fused_rope_cat_fp4 scale output scratch must be contiguous");

    int64_t const rows = pe.size(0);
    int64_t const peDim = pe.size(1);
    int64_t const nopeDim = nope.size(1);
    int64_t const headDim = peDim + nopeDim;
    TORCH_CHECK(rows == nope.size(0), "fused_rope_cat_fp4 pe/nope row counts must match");
    TORCH_CHECK(positionIds.numel() == rows, "fused_rope_cat_fp4 position ids length must match row count");
    TORCH_CHECK(headDim == 128, "fused_rope_cat_fp4 head_dim must be 128");
    TORCH_CHECK(
        packedOut.numel() >= rows * (headDim / 2), "fused_rope_cat_fp4 packed output scratch is smaller than required");
    TORCH_CHECK(scaleOut.numel() >= rows, "fused_rope_cat_fp4 scale output scratch is smaller than required");
    TORCH_CHECK(rows <= std::numeric_limits<int32_t>::max() && peDim <= std::numeric_limits<int32_t>::max()
            && nopeDim <= std::numeric_limits<int32_t>::max() && headDim <= std::numeric_limits<int32_t>::max()
            && pe.stride(0) <= std::numeric_limits<int32_t>::max()
            && nope.stride(0) <= std::numeric_limits<int32_t>::max()
            && cosSin.stride(0) <= std::numeric_limits<int32_t>::max(),
        "fused_rope_cat_fp4 dimensions exceed int32 range");

    auto stream = at::cuda::getCurrentCUDAStream(pe.get_device());
    tk::invokeFusedRopeCatFp4(reinterpret_cast<int8_t*>(packedOut.mutable_data_ptr()),
        scaleOut.mutable_data_ptr<int32_t>(), reinterpret_cast<__nv_bfloat16 const*>(pe.const_data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(nope.const_data_ptr()), cosSin.const_data_ptr<float>(),
        positionIds.const_data_ptr<int32_t>(), static_cast<int32_t>(rows), static_cast<int32_t>(peDim),
        static_cast<int32_t>(nopeDim), static_cast<int32_t>(headDim), static_cast<int32_t>(pe.stride(0)),
        static_cast<int32_t>(nope.stride(0)), static_cast<int32_t>(cosSin.stride(0)), stream);
}

at::Tensor callTrtllmCuteDslFp4PagedMqaLogits(at::Tensor const& q, at::Tensor const& qScale, at::Tensor const& kvFused,
    at::Tensor const& weights, at::Tensor const& contextLens, at::Tensor const& blockTable,
    at::Tensor const& scheduleMeta, int64_t maxContextLen)
{
    TORCH_CHECK(q.is_cuda(), "cute_dsl_fp4_paged_mqa_logits q must be a CUDA tensor");
    TORCH_CHECK(qScale.is_cuda(), "cute_dsl_fp4_paged_mqa_logits q scale must be a CUDA tensor");
    TORCH_CHECK(kvFused.is_cuda(), "cute_dsl_fp4_paged_mqa_logits KV cache must be a CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "cute_dsl_fp4_paged_mqa_logits weights must be a CUDA tensor");
    TORCH_CHECK(contextLens.is_cuda(), "cute_dsl_fp4_paged_mqa_logits context_lens must be a CUDA tensor");
    TORCH_CHECK(blockTable.is_cuda(), "cute_dsl_fp4_paged_mqa_logits block_table must be a CUDA tensor");
    TORCH_CHECK(scheduleMeta.is_cuda(), "cute_dsl_fp4_paged_mqa_logits schedule_meta must be a CUDA tensor");
    TORCH_CHECK(maxContextLen > 0, "cute_dsl_fp4_paged_mqa_logits max_context_len must be positive");

    c10::Stack stack;
    stack.emplace_back(q);
    stack.emplace_back(qScale);
    stack.emplace_back(kvFused);
    stack.emplace_back(weights);
    stack.emplace_back(contextLens);
    stack.emplace_back(blockTable);
    stack.emplace_back(scheduleMeta);
    stack.emplace_back(maxContextLen);
    stack.emplace_back(static_cast<int64_t>(1));
    stack.emplace_back(at::ScalarType::Float);
    stack.emplace_back(at::ScalarType::Float);
    stack.emplace_back(false);

    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("trtllm::cute_dsl_fp4_paged_mqa_logits", "");
    op.callBoxed(&stack);
    TORCH_CHECK(stack.size() == 1, "trtllm::cute_dsl_fp4_paged_mqa_logits returned an invalid boxed result");
    return stack.back().toTensor();
}

at::ScalarType residentDeepGemmIndexerLogitsDtype()
{
    std::string const dtype = envStringOrDefault("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_INDEXER_LOGITS_DTYPE", "bf16");
    if (dtype == "bf16" || dtype == "bfloat16")
    {
        return at::ScalarType::BFloat16;
    }
    TORCH_CHECK(dtype == "fp32" || dtype == "float32",
        "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_INDEXER_LOGITS_DTYPE must be bf16 or fp32");
    return at::ScalarType::Float;
}

at::Tensor callDeepGemmFp4PagedMqaLogits(at::Tensor const& q, at::Tensor const& qScale, at::Tensor const& kvFused,
    at::Tensor const& weights, at::Tensor const& contextLens, at::Tensor const& blockTable,
    at::Tensor const& scheduleMeta, int64_t maxContextLen, std::optional<at::ScalarType> logitsDtype = std::nullopt)
{
#if defined(TRTLLM_OPTRT_ENABLE_DEEP_GEMM_RESIDENT_INDEXER)
    TORCH_CHECK(q.is_cuda(), "DeepGEMM fp4_paged_mqa_logits q must be a CUDA tensor");
    TORCH_CHECK(qScale.is_cuda(), "DeepGEMM fp4_paged_mqa_logits q scale must be a CUDA tensor");
    TORCH_CHECK(kvFused.is_cuda(), "DeepGEMM fp4_paged_mqa_logits KV cache must be a CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "DeepGEMM fp4_paged_mqa_logits weights must be a CUDA tensor");
    TORCH_CHECK(contextLens.is_cuda(), "DeepGEMM fp4_paged_mqa_logits context_lens must be a CUDA tensor");
    TORCH_CHECK(blockTable.is_cuda(), "DeepGEMM fp4_paged_mqa_logits block_table must be a CUDA tensor");
    TORCH_CHECK(scheduleMeta.is_cuda(), "DeepGEMM fp4_paged_mqa_logits schedule_meta must be a CUDA tensor");
    TORCH_CHECK(maxContextLen > 0, "DeepGEMM fp4_paged_mqa_logits max_context_len must be positive");
    TORCH_CHECK(q.dim() == 4, "DeepGEMM fp4_paged_mqa_logits q must be [B, next_n, H, D/2]");
    TORCH_CHECK(qScale.dim() == 3, "DeepGEMM fp4_paged_mqa_logits q scale must be [B, next_n, H]");
    TORCH_CHECK(q.size(0) == qScale.size(0) && q.size(1) == qScale.size(1) && q.size(2) == qScale.size(2),
        "DeepGEMM fp4_paged_mqa_logits q and q scale shapes are inconsistent");
    TORCH_CHECK(qScale.scalar_type() == at::ScalarType::Int, "DeepGEMM fp4_paged_mqa_logits q scale must be int32");
    TORCH_CHECK(kvFused.scalar_type() == at::ScalarType::Byte, "DeepGEMM fp4_paged_mqa_logits KV cache must be uint8");
    TORCH_CHECK(weights.scalar_type() == at::ScalarType::Float, "DeepGEMM fp4_paged_mqa_logits weights must be fp32");
    TORCH_CHECK(
        blockTable.scalar_type() == at::ScalarType::Int, "DeepGEMM fp4_paged_mqa_logits block table must be int32");
    TORCH_CHECK(
        contextLens.scalar_type() == at::ScalarType::Int, "DeepGEMM fp4_paged_mqa_logits context_lens must be int32");

    ensureDeepGemmResidentIndexerRuntime();

    at::Tensor const qInt8 = q.scalar_type() == at::ScalarType::Char ? q : q.view(torch::kInt8);
    at::Tensor const contextLens2d
        = contextLens.dim() == 1 ? contextLens.reshape({contextLens.size(0), 1}) : contextLens;
    TORCH_CHECK(contextLens2d.dim() == 2, "DeepGEMM fp4_paged_mqa_logits context_lens must be 1D or 2D");
    TORCH_CHECK(contextLens2d.size(0) == qInt8.size(0) && contextLens2d.size(1) == qInt8.size(1),
        "DeepGEMM fp4_paged_mqa_logits context_lens shape must match q [B, next_n]");

    return deep_gemm::attention::fp8_fp4_paged_mqa_logits(std::make_tuple(qInt8, std::optional<torch::Tensor>(qScale)),
        kvFused, weights, contextLens2d.contiguous(), blockTable, scheduleMeta, static_cast<int>(maxContextLen),
        /*clean_logits=*/false, logitsDtype.value_or(residentDeepGemmIndexerLogitsDtype()));
#else
    static_cast<void>(q);
    static_cast<void>(qScale);
    static_cast<void>(kvFused);
    static_cast<void>(weights);
    static_cast<void>(contextLens);
    static_cast<void>(blockTable);
    static_cast<void>(scheduleMeta);
    static_cast<void>(maxContextLen);
    TORCH_CHECK(false, "DeepGEMM resident indexer support was not compiled into th_common");
    return {};
#endif
}

at::Tensor callDeepGemmPagedMqaLogitsMetadata(at::Tensor const& contextLens, int64_t blockKv, int64_t numSms)
{
#if defined(TRTLLM_OPTRT_ENABLE_DEEP_GEMM_RESIDENT_INDEXER)
    TORCH_CHECK(contextLens.is_cuda(), "DeepGEMM paged MQA metadata context_lens must be a CUDA tensor");
    TORCH_CHECK(
        contextLens.scalar_type() == at::ScalarType::Int, "DeepGEMM paged MQA metadata context_lens must be int32");
    TORCH_CHECK(contextLens.dim() == 2, "DeepGEMM paged MQA metadata context_lens must be 2D");
    TORCH_CHECK(contextLens.is_contiguous(), "DeepGEMM paged MQA metadata context_lens must be contiguous");
    TORCH_CHECK(blockKv > 0 && numSms > 0, "DeepGEMM paged MQA metadata dimensions must be positive");
    ensureDeepGemmResidentIndexerRuntime();
    return deep_gemm::attention::get_paged_mqa_logits_metadata(
        contextLens, static_cast<int>(blockKv), static_cast<int>(numSms));
#else
    static_cast<void>(contextLens);
    static_cast<void>(blockKv);
    static_cast<void>(numSms);
    TORCH_CHECK(false, "DeepGEMM resident indexer metadata support was not compiled into th_common");
    return {};
#endif
}

void callIndexerHisaUpdatePageRepsNvfp4(at::Tensor const& kCache, at::Tensor const& pageReps,
    at::Tensor const& pageCounts, at::Tensor const& slotMappingFp8, int64_t numTokens)
{
    c10::Stack stack;
    stack.emplace_back(kCache);
    stack.emplace_back(pageReps);
    stack.emplace_back(pageCounts);
    stack.emplace_back(slotMappingFp8);
    stack.emplace_back(numTokens);
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("trtllm::indexer_hisa_update_page_reps_nvfp4", "");
    op.callBoxed(&stack);
}

at::Tensor callIndexerHisaBlockRepsFromPagesNvfp4(at::Tensor const& pageReps, at::Tensor const& pageCounts,
    at::Tensor const& blockTable, at::Tensor const& kvLens, int64_t maxBlocks, int64_t pageSize)
{
    c10::Stack stack;
    stack.emplace_back(pageReps);
    stack.emplace_back(pageCounts);
    stack.emplace_back(blockTable);
    stack.emplace_back(kvLens);
    stack.emplace_back(maxBlocks);
    stack.emplace_back(pageSize);
    static auto op
        = c10::Dispatcher::singleton().findSchemaOrThrow("trtllm::indexer_hisa_block_reps_from_pages_nvfp4", "");
    op.callBoxed(&stack);
    TORCH_CHECK(stack.size() == 1, "trtllm::indexer_hisa_block_reps_from_pages_nvfp4 returned invalid boxed result");
    return stack.back().toTensor();
}

at::Tensor callIndexerHisaBlockCounts(at::Tensor const& prefixLens, int64_t blockSize)
{
    c10::Stack stack;
    stack.emplace_back(prefixLens);
    stack.emplace_back(blockSize);
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("trtllm::indexer_hisa_block_counts", "");
    op.callBoxed(&stack);
    TORCH_CHECK(stack.size() == 1, "trtllm::indexer_hisa_block_counts returned invalid boxed result");
    return stack.back().toTensor();
}

at::Tensor callIndexerHisaBlockScoresNvfp4(at::Tensor const& qValues, at::Tensor const& qScales,
    at::Tensor const& weights, at::Tensor const& blockReps, at::Tensor const& prefixLens, int64_t blockTopK,
    int64_t nextN, int64_t blockSize)
{
    c10::Stack stack;
    stack.emplace_back(qValues);
    stack.emplace_back(qScales);
    stack.emplace_back(weights);
    stack.emplace_back(blockReps);
    stack.emplace_back(prefixLens);
    stack.emplace_back(blockTopK);
    stack.emplace_back(nextN);
    stack.emplace_back(blockSize);
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("trtllm::indexer_hisa_block_scores_nvfp4", "");
    op.callBoxed(&stack);
    TORCH_CHECK(stack.size() == 1, "trtllm::indexer_hisa_block_scores_nvfp4 returned invalid boxed result");
    return stack.back().toTensor();
}

at::Tensor callIndexerHisaCandidatePages(
    at::Tensor const& topBlocks, at::Tensor const& blockTable, int64_t nextN, int64_t pagesPerHisaBlock)
{
    c10::Stack stack;
    stack.emplace_back(topBlocks);
    stack.emplace_back(blockTable);
    stack.emplace_back(nextN);
    stack.emplace_back(pagesPerHisaBlock);
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("trtllm::indexer_hisa_candidate_pages", "");
    op.callBoxed(&stack);
    TORCH_CHECK(stack.size() == 1, "trtllm::indexer_hisa_candidate_pages returned invalid boxed result");
    return stack.back().toTensor();
}

void callIndexerHisaMaskScores(
    at::Tensor const& candidateScores, at::Tensor const& topBlocks, at::Tensor const& prefixLens, int64_t blockSize)
{
    c10::Stack stack;
    stack.emplace_back(candidateScores);
    stack.emplace_back(topBlocks);
    stack.emplace_back(prefixLens);
    stack.emplace_back(blockSize);
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("trtllm::indexer_hisa_mask_scores", "");
    op.callBoxed(&stack);
}

at::Tensor callIndexerHisaRemapSelected(at::Tensor const& selected, at::Tensor const& topBlocks,
    at::Tensor const& prefixLens, int64_t blockSize, int64_t indexTopK)
{
    c10::Stack stack;
    stack.emplace_back(selected);
    stack.emplace_back(topBlocks);
    stack.emplace_back(prefixLens);
    stack.emplace_back(blockSize);
    stack.emplace_back(indexTopK);
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("trtllm::indexer_hisa_remap_selected", "");
    op.callBoxed(&stack);
    TORCH_CHECK(stack.size() == 1, "trtllm::indexer_hisa_remap_selected returned invalid boxed result");
    return stack.back().toTensor();
}

at::Tensor callTrtllmNvfp4Gemm(at::Tensor const& actFp4, at::Tensor const& weight, at::Tensor const& actSf,
    at::Tensor const& weightScale, at::Tensor const& alpha, at::ScalarType outputDtype, int64_t outputBufferKind,
    std::string const& allowedBackends)
{
    TORCH_CHECK(actFp4.is_cuda(), "NVFP4 activation tensor must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "NVFP4 weight tensor must be a CUDA tensor");
    TORCH_CHECK(actSf.is_cuda(), "NVFP4 activation scale tensor must be a CUDA tensor");
    TORCH_CHECK(weightScale.is_cuda(), "NVFP4 weight scale tensor must be a CUDA tensor");
    TORCH_CHECK(alpha.is_cuda(), "NVFP4 alpha tensor must be a CUDA tensor");
    TORCH_CHECK(!allowedBackends.empty(), "NVFP4 allowed backends must not be empty");

    c10::Stack stack;
    stack.emplace_back(actFp4);
    stack.emplace_back(weight);
    stack.emplace_back(actSf);
    stack.emplace_back(weightScale);
    stack.emplace_back(alpha);
    stack.emplace_back(outputDtype);
    stack.emplace_back(c10::SymInt(outputBufferKind));
    stack.emplace_back(allowedBackends);
    stack.emplace_back();

    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("trtllm::nvfp4_gemm", "");
    op.callBoxed(&stack);
    TORCH_CHECK(stack.size() == 1, "trtllm::nvfp4_gemm returned an invalid boxed result");
    return stack.back().toTensor();
}

std::optional<std::tuple<at::Tensor, at::Tensor>> tryCallCuteDslNvfp4DenseGemmSwiGluFp4Out(at::Tensor const& actFp4,
    at::Tensor const& weight, at::Tensor const& actSf, at::Tensor const& weightScale, at::Tensor const& alpha,
    at::Tensor const& globalSf)
{
    try
    {
        c10::Stack stack;
        stack.emplace_back(actFp4);
        stack.emplace_back(weight);
        stack.emplace_back(actSf);
        stack.emplace_back(weightScale);
        stack.emplace_back(alpha);
        stack.emplace_back(globalSf);
        stack.emplace_back(true);

        static auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
            "trtllm::cute_dsl_nvfp4_dense_gemm_swiglu_fp4out_blackwell", "");
        op.callBoxed(&stack);
        TORCH_CHECK(stack.size() == 1,
            "trtllm::cute_dsl_nvfp4_dense_gemm_swiglu_fp4out_blackwell returned an invalid boxed result");
        c10::intrusive_ptr<c10::ivalue::Tuple> result = stack.back().toTuple();
        TORCH_CHECK(result->elements().size() == 2,
            "trtllm::cute_dsl_nvfp4_dense_gemm_swiglu_fp4out_blackwell must return two tensors");
        return std::make_tuple(result->elements()[0].toTensor(), result->elements()[1].toTensor());
    }
    catch (c10::Error const&)
    {
        return std::nullopt;
    }
}

bool residentNvfp4CudaCoreOutEnabled()
{
    return envFlagEnabled("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_NVFP4_CUDA_CORE_OUT");
}

bool residentSharedExpertFp4OutSwiGluEnabled()
{
    return envFlagEnabled("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_SHARED_FP4OUT_SWIGLU");
}

bool residentAttentionTailFp4OutGateEnabled()
{
    return envFlagEnabled("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_ATTENTION_TAIL_FP4OUT_GATE");
}

int64_t residentSharedExpertFp4OutSwiGluMinBatch()
{
    return std::max<int64_t>(envIntOrDefault("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_SHARED_FP4OUT_SWIGLU_MIN_BATCH", 16), 1);
}

bool canRunCudaCoreNvfp4GemmOut(
    at::Tensor const& input, at::Tensor const& weight, at::Tensor const& output, int64_t inputTokens)
{
    constexpr int64_t kCudaCoreNvfp4MaxM = 8;
    if (!residentNvfp4CudaCoreOutEnabled() || inputTokens <= 0 || inputTokens > kCudaCoreNvfp4MaxM)
    {
        return false;
    }
    if (input.dim() != 2 || weight.dim() != 2 || output.dim() != 2)
    {
        return false;
    }
    if (!input.is_cuda() || !weight.is_cuda() || !output.is_cuda() || input.get_device() != output.get_device()
        || input.get_device() != weight.get_device())
    {
        return false;
    }
    if (input.size(0) < inputTokens || output.size(0) < inputTokens || output.size(1) != weight.size(0))
    {
        return false;
    }
    if (output.stride(1) != 1 || output.stride(0) != output.size(1))
    {
        return false;
    }
    return true;
}

void callCudaCoreNvfp4GemmOut(at::Tensor const& actFp4, at::Tensor const& weight, at::Tensor const& actSf,
    at::Tensor const& weightScale, at::Tensor const& alpha, at::Tensor const& output, int64_t inputTokens,
    char const* debugLabel)
{
    TORCH_CHECK(actFp4.dim() == 2, "CUDA-core NVFP4 GEMM activation must be 2D at ", debugLabel);
    TORCH_CHECK(weight.dim() == 2, "CUDA-core NVFP4 GEMM weight must be 2D at ", debugLabel);
    TORCH_CHECK(output.dim() == 2, "CUDA-core NVFP4 GEMM output must be 2D at ", debugLabel);
    TORCH_CHECK(actFp4.size(0) == inputTokens, "CUDA-core NVFP4 GEMM activation rows mismatch at ", debugLabel);
    TORCH_CHECK(
        output.size(0) >= inputTokens, "CUDA-core NVFP4 GEMM output batch is smaller than input at ", debugLabel);
    TORCH_CHECK(output.size(1) == weight.size(0), "CUDA-core NVFP4 GEMM output width mismatch at ", debugLabel);

    int64_t const scaleRows = padUpToMultiple(inputTokens, 128);
    TORCH_CHECK(
        actSf.numel() % scaleRows == 0, "CUDA-core NVFP4 GEMM activation scale cannot be reshaped at ", debugLabel);
    at::Tensor actSfUnswizzled = BlockScaleInterleaveReverse(actSf.reshape({scaleRows, -1}));
    at::Tensor outputPrefix = output.narrow(0, 0, inputTokens);
    static_cast<void>(
        cuda_core_nvfp4_gemm_out(actFp4, weight, actSfUnswizzled, weightScale, alpha, std::nullopt, outputPrefix));
}

at::Tensor rmsNorm2d(at::Tensor const& input, at::Tensor const& weight, int64_t inputTokens, double eps)
{
    TORCH_CHECK(input.is_cuda(), "RMSNorm input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "RMSNorm weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "RMSNorm input must be 2D");
    TORCH_CHECK(weight.dim() == 1, "RMSNorm weight must be 1D");
    TORCH_CHECK(inputTokens > 0, "RMSNorm input_tokens must be positive");
    TORCH_CHECK(input.size(0) >= inputTokens, "RMSNorm input batch is smaller than input_tokens");
    TORCH_CHECK(input.size(1) == weight.size(0), "RMSNorm weight size must match hidden dim");

    at::Tensor prefix = input.narrow(0, 0, inputTokens);
    at::Tensor inputFloat = prefix.to(at::ScalarType::Float);
    at::Tensor variance = inputFloat.pow(2).mean(-1, true);
    at::Tensor normalized = inputFloat * at::rsqrt(variance + eps);
    return normalized.to(input.scalar_type()) * weight;
}

std::optional<tk::DeepseekResidentNormDtype> residentNormDtype(at::ScalarType dtype)
{
    if (dtype == at::ScalarType::Half)
    {
        return tk::DeepseekResidentNormDtype::kFloat16;
    }
    if (dtype == at::ScalarType::BFloat16)
    {
        return tk::DeepseekResidentNormDtype::kBfloat16;
    }
    return std::nullopt;
}

bool tryRunResidentRmsNorm(at::Tensor const& input, at::Tensor const& output, at::Tensor const& weight,
    int64_t inputTokens, double eps, bool useGemma)
{
    if (useGemma || input.dim() != 2 || output.dim() != 2 || weight.dim() != 1 || inputTokens <= 0
        || input.size(0) < inputTokens || output.size(0) < inputTokens || input.size(1) != output.size(1)
        || weight.size(0) != input.size(1) || input.scalar_type() != output.scalar_type()
        || input.scalar_type() != weight.scalar_type() || input.stride(1) != 1 || output.stride(1) != 1
        || weight.stride(0) != 1)
    {
        return false;
    }
    auto const dtype = residentNormDtype(input.scalar_type());
    if (!dtype.has_value() || inputTokens > std::numeric_limits<int32_t>::max()
        || input.size(1) > std::numeric_limits<int32_t>::max())
    {
        return false;
    }

    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    tk::invokeDeepseekResidentRmsNorm(output.mutable_data_ptr(), output.stride(0), input.const_data_ptr(),
        input.stride(0), weight.const_data_ptr(), weight.stride(0), static_cast<int32_t>(inputTokens),
        static_cast<int32_t>(input.size(1)), static_cast<float>(eps), dtype.value(), stream);
    return true;
}

bool tryRunResidentDsaKvASplitNormPack(at::Tensor const& kvA, at::Tensor const& qLora, at::Tensor const& compressedKv,
    at::Tensor const& kPe, at::Tensor const& latentCache, at::Tensor const& qWeight, at::Tensor const& kvWeight,
    int64_t inputTokens, int64_t qLoraRank, int64_t kvLoraRank, int64_t ropeDim, double eps)
{
    if (!kvA.is_cuda() || !qLora.is_cuda() || !compressedKv.is_cuda() || !kPe.is_cuda() || !latentCache.is_cuda()
        || !qWeight.is_cuda() || !kvWeight.is_cuda() || kvA.dim() != 2 || qLora.dim() != 2 || compressedKv.dim() != 2
        || kPe.dim() != 2 || latentCache.dim() != 2 || qWeight.dim() != 1 || kvWeight.dim() != 1 || inputTokens <= 0
        || qLoraRank <= 0 || kvLoraRank <= 0 || ropeDim <= 0 || kvA.size(0) < inputTokens || qLora.size(0) < inputTokens
        || compressedKv.size(0) < inputTokens || kPe.size(0) < inputTokens || latentCache.size(0) < inputTokens
        || kvA.size(1) < qLoraRank + kvLoraRank + ropeDim || qLora.size(1) != qLoraRank
        || compressedKv.size(1) != kvLoraRank || kPe.size(1) != ropeDim || latentCache.size(1) != kvLoraRank + ropeDim
        || qWeight.size(0) != qLoraRank || kvWeight.size(0) != kvLoraRank || kvA.scalar_type() != qLora.scalar_type()
        || kvA.scalar_type() != compressedKv.scalar_type() || kvA.scalar_type() != kPe.scalar_type()
        || kvA.scalar_type() != latentCache.scalar_type() || kvA.scalar_type() != qWeight.scalar_type()
        || kvA.scalar_type() != kvWeight.scalar_type() || kvA.stride(1) != 1 || qLora.stride(1) != 1
        || compressedKv.stride(1) != 1 || kPe.stride(1) != 1 || latentCache.stride(1) != 1 || qWeight.stride(0) != 1
        || kvWeight.stride(0) != 1)
    {
        return false;
    }
    int32_t const device = kvA.get_device();
    if (qLora.get_device() != device || compressedKv.get_device() != device || kPe.get_device() != device
        || latentCache.get_device() != device || qWeight.get_device() != device || kvWeight.get_device() != device)
    {
        return false;
    }
    auto const dtype = residentNormDtype(kvA.scalar_type());
    if (!dtype.has_value() || inputTokens > std::numeric_limits<int32_t>::max()
        || qLoraRank > std::numeric_limits<int32_t>::max() || kvLoraRank > std::numeric_limits<int32_t>::max()
        || ropeDim > std::numeric_limits<int32_t>::max())
    {
        return false;
    }

    auto stream = at::cuda::getCurrentCUDAStream(device);
    tk::invokeDeepseekResidentDsaKvASplitNormPack(kvA.const_data_ptr(), kvA.stride(0), qLora.mutable_data_ptr(),
        qLora.stride(0), compressedKv.mutable_data_ptr(), compressedKv.stride(0), kPe.mutable_data_ptr(), kPe.stride(0),
        latentCache.mutable_data_ptr(), latentCache.stride(0), qWeight.const_data_ptr(), qWeight.stride(0),
        kvWeight.const_data_ptr(), kvWeight.stride(0), static_cast<int32_t>(inputTokens),
        static_cast<int32_t>(qLoraRank), static_cast<int32_t>(kvLoraRank), static_cast<int32_t>(ropeDim),
        static_cast<float>(eps), dtype.value(), stream);
    return true;
}

bool tryRunResidentAddRmsNorm(at::Tensor const& input, at::Tensor const& residual, at::Tensor const& normOutput,
    at::Tensor const& residualOutput, at::Tensor const& weight, int64_t inputTokens, double eps, bool useGemma)
{
    if (useGemma || input.dim() != 2 || residual.dim() != 2 || normOutput.dim() != 2 || residualOutput.dim() != 2
        || weight.dim() != 1 || inputTokens <= 0 || input.size(0) < inputTokens || residual.size(0) < inputTokens
        || normOutput.size(0) < inputTokens || residualOutput.size(0) < inputTokens || input.size(1) != residual.size(1)
        || input.size(1) != normOutput.size(1) || input.size(1) != residualOutput.size(1)
        || weight.size(0) != input.size(1) || input.scalar_type() != residual.scalar_type()
        || input.scalar_type() != normOutput.scalar_type() || input.scalar_type() != residualOutput.scalar_type()
        || input.scalar_type() != weight.scalar_type() || input.stride(1) != 1 || residual.stride(1) != 1
        || normOutput.stride(1) != 1 || residualOutput.stride(1) != 1 || weight.stride(0) != 1)
    {
        return false;
    }
    auto const dtype = residentNormDtype(input.scalar_type());
    if (!dtype.has_value() || inputTokens > std::numeric_limits<int32_t>::max()
        || input.size(1) > std::numeric_limits<int32_t>::max())
    {
        return false;
    }

    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    tk::invokeDeepseekResidentAddRmsNorm(normOutput.mutable_data_ptr(), normOutput.stride(0),
        residualOutput.mutable_data_ptr(), residualOutput.stride(0), input.const_data_ptr(), input.stride(0),
        residual.const_data_ptr(), residual.stride(0), weight.const_data_ptr(), weight.stride(0),
        static_cast<int32_t>(inputTokens), static_cast<int32_t>(input.size(1)), static_cast<float>(eps), dtype.value(),
        stream);
    return true;
}

bool tryRunResidentAddRmsNormLowRankGate(at::Tensor const& input, at::Tensor const& residual,
    at::Tensor const& gatedOutput, at::Tensor const& residualOutput, at::Tensor const& normWeight,
    at::Tensor const& downWeight, at::Tensor const& upWeight, int64_t inputTokens, double eps, bool useGemma)
{
    constexpr int64_t kMaxResidentLowRankGateRank = 64;
    if (!residentFusedPostAttentionGateEnabled())
    {
        return false;
    }
    if (input.dim() != 2 || residual.dim() != 2 || gatedOutput.dim() != 2 || residualOutput.dim() != 2
        || normWeight.dim() != 1 || downWeight.dim() != 2 || upWeight.dim() != 2 || inputTokens <= 0
        || input.size(0) < inputTokens || residual.size(0) < inputTokens || gatedOutput.size(0) < inputTokens
        || residualOutput.size(0) < inputTokens || input.size(1) != residual.size(1)
        || input.size(1) != gatedOutput.size(1) || input.size(1) != residualOutput.size(1)
        || normWeight.size(0) != input.size(1) || downWeight.size(1) != input.size(1)
        || upWeight.size(0) != input.size(1) || upWeight.size(1) != downWeight.size(0) || downWeight.size(0) <= 0
        || downWeight.size(0) > kMaxResidentLowRankGateRank || input.scalar_type() != residual.scalar_type()
        || input.scalar_type() != gatedOutput.scalar_type() || input.scalar_type() != residualOutput.scalar_type()
        || input.scalar_type() != normWeight.scalar_type() || input.scalar_type() != downWeight.scalar_type()
        || input.scalar_type() != upWeight.scalar_type() || input.stride(1) != 1 || residual.stride(1) != 1
        || gatedOutput.stride(1) != 1 || residualOutput.stride(1) != 1 || normWeight.stride(0) <= 0
        || downWeight.stride(1) != 1 || upWeight.stride(1) != 1)
    {
        return false;
    }
    if (input.get_device() != residual.get_device() || input.get_device() != gatedOutput.get_device()
        || input.get_device() != residualOutput.get_device() || input.get_device() != normWeight.get_device()
        || input.get_device() != downWeight.get_device() || input.get_device() != upWeight.get_device())
    {
        return false;
    }
    auto const dtype = residentNormDtype(input.scalar_type());
    if (!dtype.has_value() || inputTokens > std::numeric_limits<int32_t>::max()
        || input.size(1) > std::numeric_limits<int32_t>::max()
        || downWeight.size(0) > std::numeric_limits<int32_t>::max())
    {
        return false;
    }

    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    tk::invokeDeepseekResidentAddRmsNormLowRankGate(gatedOutput.mutable_data_ptr(), gatedOutput.stride(0),
        residualOutput.mutable_data_ptr(), residualOutput.stride(0), input.const_data_ptr(), input.stride(0),
        residual.const_data_ptr(), residual.stride(0), normWeight.const_data_ptr(), normWeight.stride(0),
        downWeight.const_data_ptr(), downWeight.stride(0), downWeight.stride(1), upWeight.const_data_ptr(),
        upWeight.stride(0), upWeight.stride(1), static_cast<int32_t>(inputTokens), static_cast<int32_t>(input.size(1)),
        static_cast<int32_t>(downWeight.size(0)), static_cast<float>(eps), useGemma, dtype.value(), stream);
    return true;
}

bool tryRunResidentSigmoidMul(
    at::Tensor const& input, at::Tensor const& gate, at::Tensor const& output, int64_t inputTokens)
{
    if (input.dim() != 2 || gate.dim() != 2 || output.dim() != 2 || inputTokens <= 0 || input.size(0) < inputTokens
        || gate.size(0) < inputTokens || output.size(0) < inputTokens || input.size(1) != gate.size(1)
        || input.size(1) != output.size(1) || input.scalar_type() != output.scalar_type()
        || gate.scalar_type() != at::ScalarType::Float || input.stride(1) != 1 || gate.stride(1) != 1
        || output.stride(1) != 1)
    {
        return false;
    }
    auto const dtype = residentNormDtype(input.scalar_type());
    if (!dtype.has_value() || inputTokens > std::numeric_limits<int32_t>::max()
        || input.size(1) > std::numeric_limits<int32_t>::max())
    {
        return false;
    }

    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    tk::invokeDeepseekResidentSigmoidMul(output.mutable_data_ptr(), output.stride(0), input.const_data_ptr(),
        input.stride(0), gate.const_data_ptr<float>(), gate.stride(0), static_cast<int32_t>(inputTokens),
        static_cast<int32_t>(input.size(1)), dtype.value(), stream);
    return true;
}

bool tryRunResidentSwiGluFloatToOutput(
    at::Tensor const& gate, at::Tensor const& up, at::Tensor const& output, int64_t inputTokens)
{
    if (gate.dim() != 2 || up.dim() != 2 || output.dim() != 2 || inputTokens <= 0 || gate.size(0) < inputTokens
        || up.size(0) < inputTokens || output.size(0) < inputTokens || gate.size(1) != up.size(1)
        || gate.size(1) != output.size(1) || gate.scalar_type() != at::ScalarType::Float
        || up.scalar_type() != at::ScalarType::Float || gate.stride(1) != 1 || up.stride(1) != 1
        || output.stride(1) != 1)
    {
        return false;
    }
    auto const dtype = residentNormDtype(output.scalar_type());
    if (!dtype.has_value() || inputTokens > std::numeric_limits<int32_t>::max()
        || output.size(1) > std::numeric_limits<int32_t>::max())
    {
        return false;
    }
    if (gate.get_device() != output.get_device() || up.get_device() != output.get_device())
    {
        return false;
    }

    auto stream = at::cuda::getCurrentCUDAStream(output.get_device());
    tk::invokeDeepseekResidentSwiGluFloatToOutput(output.mutable_data_ptr(), output.stride(0),
        gate.const_data_ptr<float>(), gate.stride(0), up.const_data_ptr<float>(), up.stride(0),
        static_cast<int32_t>(inputTokens), static_cast<int32_t>(output.size(1)), dtype.value(), stream);
    return true;
}

bool tryRunResidentLowRankGate(at::Tensor const& input, at::Tensor const& output, at::Tensor const& downWeight,
    at::Tensor const& upWeight, int64_t inputTokens)
{
    constexpr int64_t kMaxResidentLowRankGateRank = 64;
    if (!input.is_cuda() || !output.is_cuda() || !downWeight.is_cuda() || !upWeight.is_cuda() || input.dim() != 2
        || output.dim() != 2 || downWeight.dim() != 2 || upWeight.dim() != 2 || inputTokens <= 0
        || input.size(0) < inputTokens || output.size(0) < inputTokens || input.size(1) != output.size(1)
        || downWeight.size(1) != input.size(1) || upWeight.size(0) != input.size(1)
        || upWeight.size(1) != downWeight.size(0) || downWeight.size(0) <= 0
        || downWeight.size(0) > kMaxResidentLowRankGateRank || input.scalar_type() != output.scalar_type()
        || input.scalar_type() != downWeight.scalar_type() || input.scalar_type() != upWeight.scalar_type()
        || input.stride(1) != 1 || output.stride(1) != 1 || downWeight.stride(1) != 1 || upWeight.stride(1) != 1)
    {
        return false;
    }
    if (input.get_device() != output.get_device() || input.get_device() != downWeight.get_device()
        || input.get_device() != upWeight.get_device())
    {
        return false;
    }
    auto const dtype = residentNormDtype(input.scalar_type());
    if (!dtype.has_value() || inputTokens > std::numeric_limits<int32_t>::max()
        || input.size(1) > std::numeric_limits<int32_t>::max()
        || downWeight.size(0) > std::numeric_limits<int32_t>::max())
    {
        return false;
    }

    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    tk::invokeDeepseekResidentLowRankGate(output.mutable_data_ptr(), output.stride(0), input.const_data_ptr(),
        input.stride(0), downWeight.const_data_ptr(), downWeight.stride(0), downWeight.stride(1),
        upWeight.const_data_ptr(), upWeight.stride(0), upWeight.stride(1), static_cast<int32_t>(inputTokens),
        static_cast<int32_t>(input.size(1)), static_cast<int32_t>(downWeight.size(0)), dtype.value(), stream);
    return true;
}

bool tryRunResidentAddScaledFloatToOutput(
    at::Tensor const& addend, at::Tensor const& output, int64_t inputTokens, double addendScale)
{
    if (!envFlagEnabled("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MOE_FUSED_COMBINE"))
    {
        return false;
    }
    if (addend.dim() != 2 || output.dim() != 2 || inputTokens <= 0 || addend.size(0) < inputTokens
        || output.size(0) < inputTokens || addend.size(1) != output.size(1)
        || addend.scalar_type() != at::ScalarType::Float || addend.stride(1) != 1 || output.stride(1) != 1)
    {
        return false;
    }
    auto const dtype = residentNormDtype(output.scalar_type());
    if (!dtype.has_value() || inputTokens > std::numeric_limits<int32_t>::max()
        || output.size(1) > std::numeric_limits<int32_t>::max())
    {
        return false;
    }
    if (addend.get_device() != output.get_device())
    {
        return false;
    }

    auto stream = at::cuda::getCurrentCUDAStream(output.get_device());
    if (envFlagEnabled("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MOE_FUSED_COMBINE_DEVICE_SYNC"))
    {
        cudaError_t const syncErr = cudaDeviceSynchronize();
        TORCH_CHECK(
            syncErr == cudaSuccess, "resident fused MoE combine pre-sync failed: ", cudaGetErrorString(syncErr));
    }
    tk::invokeDeepseekResidentAddScaledFloatToOutput(output.mutable_data_ptr(), output.stride(0),
        addend.const_data_ptr<float>(), addend.stride(0), static_cast<int32_t>(inputTokens),
        static_cast<int32_t>(output.size(1)), static_cast<float>(addendScale), dtype.value(), stream);
    if (envFlagEnabled("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MOE_FUSED_COMBINE_DEVICE_SYNC"))
    {
        cudaError_t const syncErr = cudaDeviceSynchronize();
        TORCH_CHECK(
            syncErr == cudaSuccess, "resident fused MoE combine post-sync failed: ", cudaGetErrorString(syncErr));
    }
    return true;
}

at::Tensor layerNorm2d(at::Tensor const& input, at::Tensor const& weight, std::optional<at::Tensor> const& bias,
    int64_t inputTokens, double eps)
{
    TORCH_CHECK(input.is_cuda(), "LayerNorm input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "LayerNorm weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "LayerNorm input must be 2D");
    TORCH_CHECK(weight.dim() == 1, "LayerNorm weight must be 1D");
    TORCH_CHECK(inputTokens > 0, "LayerNorm input_tokens must be positive");
    TORCH_CHECK(input.size(0) >= inputTokens, "LayerNorm input batch is smaller than input_tokens");
    TORCH_CHECK(input.size(1) == weight.size(0), "LayerNorm weight size must match hidden dim");
    if (bias.has_value())
    {
        TORCH_CHECK(bias.value().is_cuda(), "LayerNorm bias must be a CUDA tensor");
        TORCH_CHECK(bias.value().dim() == 1, "LayerNorm bias must be 1D");
        TORCH_CHECK(bias.value().size(0) == weight.size(0), "LayerNorm bias size must match hidden dim");
    }

    at::Tensor inputFloat = input.narrow(0, 0, inputTokens).to(at::ScalarType::Float);
    at::Tensor mean = inputFloat.mean(-1, true);
    at::Tensor centered = inputFloat - mean;
    at::Tensor variance = centered.pow(2).mean(-1, true);
    at::Tensor output = centered * at::rsqrt(variance + eps);
    output = output * weight.to(at::ScalarType::Float);
    if (bias.has_value())
    {
        output = output + bias.value().to(at::ScalarType::Float);
    }
    return output.to(input.scalar_type());
}

DeepseekResidentManifest validateDeepseekResidentManifest(
    th::List<int64_t> residentLayerOffsets, th::List<int64_t> residentLayerKinds, c10::List<at::Tensor> residentTensors)
{
    TORCH_CHECK(residentLayerOffsets.size() == residentLayerKinds.size() + 1,
        "resident_layer_offsets must have one more entry than resident_layer_kinds");
    TORCH_CHECK(!residentLayerOffsets.empty(), "resident_layer_offsets must not be empty");
    TORCH_CHECK(!residentTensors.empty(), "resident_tensors must not be empty");

    int64_t previousOffset = residentLayerOffsets.get(0);
    TORCH_CHECK(previousOffset >= 0, "resident_layer_offsets must be non-negative");
    for (size_t idx = 1; idx < residentLayerOffsets.size(); ++idx)
    {
        int64_t const offset = residentLayerOffsets.get(idx);
        TORCH_CHECK(offset >= previousOffset, "resident_layer_offsets must be monotonically increasing");
        previousOffset = offset;
    }
    TORCH_CHECK(
        previousOffset <= static_cast<int64_t>(residentTensors.size()), "resident_layer_offsets exceed tensor count");

    for (size_t idx = 0; idx < residentLayerKinds.size(); ++idx)
    {
        int64_t const layerKind = residentLayerKinds.get(idx);
        TORCH_CHECK(layerKind == static_cast<int64_t>(DeepseekResidentLayerKind::kDense)
                || layerKind == static_cast<int64_t>(DeepseekResidentLayerKind::kMoe),
            "resident_layer_kinds entries must be dense(0) or moe(1)");
    }
    for (size_t idx = 0; idx < residentTensors.size(); ++idx)
    {
        auto const tensor = residentTensors.get(idx);
        TORCH_CHECK(tensor.is_cuda(), "resident_tensors must all be CUDA tensors");
    }

    return DeepseekResidentManifest{
        static_cast<int64_t>(residentLayerKinds.size()), static_cast<int64_t>(residentTensors.size())};
}

void validateDeepseekResidentLayerTensorSites(th::List<int64_t> residentLayerSiteOffsets,
    th::List<int64_t> residentLayerSiteIds, th::List<int64_t> residentLayerSiteTensorIndices, int64_t const nbLayers,
    int64_t const nbTensors)
{
    TORCH_CHECK(residentLayerSiteOffsets.size() == static_cast<size_t>(nbLayers + 1),
        "resident_layer_site_offsets must have one more entry than resident_layer_kinds");
    TORCH_CHECK(residentLayerSiteIds.size() == residentLayerSiteTensorIndices.size(),
        "resident_layer_site_ids and resident_layer_site_tensor_indices must have the same length");

    int64_t previousOffset = residentLayerSiteOffsets.get(0);
    TORCH_CHECK(previousOffset >= 0, "resident_layer_site_offsets must be non-negative");
    for (size_t idx = 1; idx < residentLayerSiteOffsets.size(); ++idx)
    {
        int64_t const offset = residentLayerSiteOffsets.get(idx);
        TORCH_CHECK(offset >= previousOffset, "resident_layer_site_offsets must be monotonically increasing");
        previousOffset = offset;
    }
    TORCH_CHECK(previousOffset <= static_cast<int64_t>(residentLayerSiteIds.size()),
        "resident_layer_site_offsets exceed site count");

    for (size_t idx = 0; idx < residentLayerSiteTensorIndices.size(); ++idx)
    {
        int64_t const tensorIdx = residentLayerSiteTensorIndices.get(idx);
        TORCH_CHECK(tensorIdx >= 0 && tensorIdx < nbTensors, "resident layer site tensor index is out of range");
    }
}

DeepseekResidentRequest validateDeepseekResidentRequest(at::Tensor const& inputIds,
    at::Tensor const& hiddenStatesScratch, at::Tensor const& logitsScratch, int64_t realBatchSize,
    int64_t paddedBatchSize, int64_t inputTokens, th::List<int64_t> requestIds, th::List<int64_t> seqLens,
    th::List<int64_t> cachedTokens)
{
    TORCH_CHECK(inputIds.is_cuda(), "input_ids must be a CUDA tensor");
    TORCH_CHECK(hiddenStatesScratch.is_cuda(), "hidden_states_scratch must be a CUDA tensor");
    TORCH_CHECK(logitsScratch.is_cuda(), "logits_scratch must be a CUDA tensor");
    TORCH_CHECK(realBatchSize > 0, "real_batch_size must be positive");
    TORCH_CHECK(paddedBatchSize >= realBatchSize, "padded_batch_size must be >= real_batch_size");
    TORCH_CHECK(inputTokens > 0, "input_tokens must be positive");
    TORCH_CHECK(
        static_cast<int64_t>(requestIds.size()) >= realBatchSize, "request_ids is shorter than real_batch_size");
    TORCH_CHECK(static_cast<int64_t>(seqLens.size()) >= realBatchSize, "seq_lens is shorter than real_batch_size");
    TORCH_CHECK(
        static_cast<int64_t>(cachedTokens.size()) >= realBatchSize, "cached_tokens is shorter than real_batch_size");
    return DeepseekResidentRequest{realBatchSize, paddedBatchSize, inputTokens};
}

bool deepseekResidentDecodeReady()
{
    return false;
}

class DeepseekResidentDecodeHandle : public torch::CustomClassHolder
{
public:
    DeepseekResidentDecodeHandle(th::List<int64_t> residentLayerOffsets, th::List<int64_t> residentLayerKinds,
        th::List<int64_t> residentLayerSiteOffsets, th::List<int64_t> residentLayerSiteIds,
        th::List<int64_t> residentLayerSiteTensorIndices, c10::List<at::Tensor> residentTensors)
    {
        auto const manifest
            = validateDeepseekResidentManifest(residentLayerOffsets, residentLayerKinds, residentTensors);
        validateDeepseekResidentLayerTensorSites(residentLayerSiteOffsets, residentLayerSiteIds,
            residentLayerSiteTensorIndices, manifest.nbLayers, manifest.nbTensors);
        mNbLayers = manifest.nbLayers;
        mNbTensors = manifest.nbTensors;
        for (size_t idx = 0; idx < residentLayerOffsets.size(); ++idx)
        {
            mLayerOffsets.push_back(residentLayerOffsets.get(idx));
        }
        for (size_t idx = 0; idx < residentLayerKinds.size(); ++idx)
        {
            mLayerKinds.push_back(residentLayerKinds.get(idx));
        }
        for (size_t idx = 0; idx < residentLayerSiteOffsets.size(); ++idx)
        {
            mLayerSiteOffsets.push_back(residentLayerSiteOffsets.get(idx));
        }
        for (size_t idx = 0; idx < residentLayerSiteIds.size(); ++idx)
        {
            mLayerSiteIds.push_back(residentLayerSiteIds.get(idx));
        }
        for (size_t idx = 0; idx < residentLayerSiteTensorIndices.size(); ++idx)
        {
            mLayerSiteTensorIndices.push_back(residentLayerSiteTensorIndices.get(idx));
        }
        for (size_t idx = 0; idx < residentTensors.size(); ++idx)
        {
            mResidentTensors.push_back(residentTensors.get(idx));
        }
    }

    int64_t getNbLayers() const
    {
        return mNbLayers;
    }

    int64_t getNbTensors() const
    {
        return mNbTensors;
    }

    bool hasLayerTensorSite(int64_t layerIdx, int64_t siteId) const
    {
        return findLayerTensorIndex(layerIdx, siteId).has_value();
    }

    bool runLayerDsaIndexerAssetsReady(int64_t layerIdx) const
    {
        return runLayerDsaIndexerAssetsNotReadyReason(layerIdx).empty();
    }

    std::string runLayerDsaIndexerAssetsNotReadyReason(int64_t layerIdx) const
    {
        if (layerIdx < 0 || layerIdx >= mNbLayers)
        {
            return "resident_dsa_indexer_layer_idx_out_of_range";
        }
        std::vector<std::pair<DeepseekResidentLayerTensorSite, char const*>> const requiredSites{
            {DeepseekResidentLayerTensorSite::kAttentionIndexerWQBWeight, "wq_b_weight"},
            {DeepseekResidentLayerTensorSite::kAttentionIndexerWKWeight, "wk_weight"},
            {DeepseekResidentLayerTensorSite::kAttentionIndexerWeightsProjWeight, "weights_proj_weight"},
            {DeepseekResidentLayerTensorSite::kAttentionIndexerKNormWeight, "k_norm_weight"},
            {DeepseekResidentLayerTensorSite::kAttentionIndexerRotaryCosSin, "rotary_cos_sin"},
        };
        for (auto const& site : requiredSites)
        {
            if (!hasLayerTensorSite(layerIdx, static_cast<int64_t>(site.first)))
            {
                return std::string("resident_dsa_indexer_") + site.second + "_missing";
            }
        }
        return "";
    }

    bool runLayerMoeExpertAssetsReady(int64_t layerIdx) const
    {
        return runLayerMoeExpertAssetsNotReadyReason(layerIdx).empty();
    }

    std::string runLayerMoeExpertAssetsNotReadyReason(int64_t layerIdx) const
    {
        if (layerIdx < 0 || layerIdx >= mNbLayers)
        {
            return "resident_moe_expert_layer_idx_out_of_range";
        }
        if (mLayerKinds.at(static_cast<size_t>(layerIdx)) != static_cast<int64_t>(DeepseekResidentLayerKind::kMoe))
        {
            return "";
        }
        std::vector<std::pair<DeepseekResidentLayerTensorSite, char const*>> const requiredSites{
            {DeepseekResidentLayerTensorSite::kSharedExpertGateUpWeight, "shared_gate_up_weight"},
            {DeepseekResidentLayerTensorSite::kSharedExpertDownWeight, "shared_down_weight"},
            {DeepseekResidentLayerTensorSite::kExpertGateUpWeight, "expert_gate_up_weight"},
            {DeepseekResidentLayerTensorSite::kExpertDownWeight, "expert_down_weight"},
        };
        for (auto const& site : requiredSites)
        {
            if (!hasLayerTensorSite(layerIdx, static_cast<int64_t>(site.first)))
            {
                return std::string("resident_moe_expert_") + site.second + "_missing";
            }
        }
        return "";
    }

    at::Tensor runNvfp4Linear(at::Tensor const& input, at::Tensor const& weight, at::Tensor const& weightScale,
        at::Tensor const& inputScale, at::Tensor const& alpha, int64_t inputTokens, int64_t sfVecSize,
        std::string const& allowedBackends) const
    {
        return runNvfp4LinearTyped(input, weight, weightScale, inputScale, alpha, inputTokens, sfVecSize,
            allowedBackends, input.scalar_type(), "run_nvfp4_linear");
    }

    std::tuple<at::Tensor, at::Tensor> runLayerDsaIndexerWkWeightsProjection(int64_t layerIdx,
        at::Tensor const& hiddenStatesScratch, at::Tensor const& indexerKScratch,
        at::Tensor const& indexerWeightsScratch, int64_t inputTokens, int64_t sfVecSize,
        std::string const& allowedBackends) const
    {
        TORCH_CHECK(layerIdx >= 0 && layerIdx < mNbLayers, "layer_idx is out of range");
        TORCH_CHECK(hiddenStatesScratch.is_cuda(), "hidden_states_scratch must be a CUDA tensor");
        TORCH_CHECK(indexerKScratch.is_cuda(), "indexer_k_scratch must be a CUDA tensor");
        TORCH_CHECK(indexerWeightsScratch.is_cuda(), "indexer_weights_scratch must be a CUDA tensor");
        TORCH_CHECK(inputTokens > 0, "input_tokens must be positive");
        TORCH_CHECK(sfVecSize > 0, "NVFP4 scaling vector size must be positive");
        TORCH_CHECK(hiddenStatesScratch.dim() == 2, "hidden_states_scratch must be 2D");
        TORCH_CHECK(indexerKScratch.dim() == 2, "indexer_k_scratch must be 2D");
        TORCH_CHECK(indexerWeightsScratch.dim() == 2, "indexer_weights_scratch must be 2D");
        TORCH_CHECK(hiddenStatesScratch.size(0) >= inputTokens, "hidden_states_scratch batch is smaller than input");
        TORCH_CHECK(indexerKScratch.size(0) >= inputTokens, "indexer_k_scratch batch is smaller than input");
        TORCH_CHECK(
            indexerWeightsScratch.size(0) >= inputTokens, "indexer_weights_scratch batch is smaller than input");
        TORCH_CHECK(indexerKScratch.scalar_type() == at::ScalarType::Float, "indexer_k_scratch must be float32");
        TORCH_CHECK(
            indexerWeightsScratch.scalar_type() == at::ScalarType::Float, "indexer_weights_scratch must be float32");

        at::Tensor const& wkWeight = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kAttentionIndexerWKWeight, "DSA indexer wk weight");
        at::Tensor const& weightsProjWeight = getLayerTensor(layerIdx,
            DeepseekResidentLayerTensorSite::kAttentionIndexerWeightsProjWeight, "DSA indexer weights_proj weight");
        at::Tensor indexerK = runLayerDsaIndexerLinearToFloat(layerIdx, hiddenStatesScratch, wkWeight,
            DeepseekResidentLayerTensorSite::kAttentionIndexerWKWeightScale,
            DeepseekResidentLayerTensorSite::kAttentionIndexerWKWeightScale2,
            DeepseekResidentLayerTensorSite::kAttentionIndexerWKInputScale,
            DeepseekResidentLayerTensorSite::kAttentionIndexerWKAlpha, inputTokens, sfVecSize, allowedBackends,
            "dsa_wk");
        at::Tensor indexerWeights = runLayerDsaIndexerLinearToFloat(layerIdx, hiddenStatesScratch, weightsProjWeight,
            DeepseekResidentLayerTensorSite::kAttentionIndexerWeightsProjWeightScale,
            DeepseekResidentLayerTensorSite::kAttentionIndexerWeightsProjWeightScale2,
            DeepseekResidentLayerTensorSite::kAttentionIndexerWeightsProjInputScale,
            DeepseekResidentLayerTensorSite::kAttentionIndexerWeightsProjAlpha, inputTokens, sfVecSize, allowedBackends,
            "dsa_weights_proj");

        TORCH_CHECK(indexerK.dim() == 2, "DSA indexer wk output must be 2D");
        TORCH_CHECK(indexerWeights.dim() == 2, "DSA indexer weights_proj output must be 2D");
        TORCH_CHECK(indexerK.size(0) >= inputTokens, "DSA indexer wk output batch is smaller than input");
        TORCH_CHECK(
            indexerWeights.size(0) >= inputTokens, "DSA indexer weights_proj output batch is smaller than input");
        TORCH_CHECK(
            indexerKScratch.size(1) <= indexerK.size(1), "indexer_k_scratch is wider than DSA indexer wk output");
        TORCH_CHECK(indexerWeightsScratch.size(1) <= indexerWeights.size(1),
            "indexer_weights_scratch is wider than DSA indexer weights_proj output");
        indexerKScratch.narrow(0, 0, inputTokens)
            .copy_(indexerK.narrow(0, 0, inputTokens).narrow(1, 0, indexerKScratch.size(1)));
        indexerWeightsScratch.narrow(0, 0, inputTokens)
            .copy_(indexerWeights.narrow(0, 0, inputTokens).narrow(1, 0, indexerWeightsScratch.size(1)));
        return {indexerKScratch, indexerWeightsScratch};
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> runLayerDsaIndexerFp4ProjectionImpl(
        int64_t layerIdx, at::Tensor const& qLoraScratch, at::Tensor const& hiddenStatesScratch,
        at::Tensor const& positionIds, at::Tensor const& qFp4Scratch, at::Tensor const& kFp4Scratch,
        at::Tensor const& kScaleScratch, at::Tensor const& indexerWeightsScratch, at::Tensor const& qScaleScratch,
        std::optional<at::Tensor> const& precomputedWqB, double precomputedWqBScale,
        std::optional<at::Tensor> const& precomputedIndexerK,
        std::optional<at::Tensor> const& precomputedIndexerWeights, int64_t inputTokens, int64_t nHeads,
        int64_t headDim, int64_t ropeDim, double eps, double weightScaleFactor, int64_t sfVecSize,
        std::string const& allowedBackends) const
    {
        TORCH_CHECK(layerIdx >= 0 && layerIdx < mNbLayers, "layer_idx is out of range");
        TORCH_CHECK(qLoraScratch.is_cuda(), "q_lora_scratch must be a CUDA tensor");
        TORCH_CHECK(hiddenStatesScratch.is_cuda(), "hidden_states_scratch must be a CUDA tensor");
        TORCH_CHECK(positionIds.is_cuda(), "position_ids must be a CUDA tensor");
        TORCH_CHECK(qFp4Scratch.is_cuda(), "q_fp4_scratch must be a CUDA tensor");
        TORCH_CHECK(kFp4Scratch.is_cuda(), "k_fp4_scratch must be a CUDA tensor");
        TORCH_CHECK(kScaleScratch.is_cuda(), "k_scale_scratch must be a CUDA tensor");
        TORCH_CHECK(indexerWeightsScratch.is_cuda(), "indexer_weights_scratch must be a CUDA tensor");
        TORCH_CHECK(qScaleScratch.is_cuda(), "q_scale_scratch must be a CUDA tensor");
        TORCH_CHECK(inputTokens > 0, "input_tokens must be positive");
        TORCH_CHECK(nHeads > 0, "n_heads must be positive");
        TORCH_CHECK(headDim == 128, "resident FP4 indexer projection requires head_dim == 128");
        TORCH_CHECK(ropeDim > 0 && ropeDim < headDim, "rope_dim must be in (0, head_dim)");
        TORCH_CHECK(sfVecSize > 0, "NVFP4 scaling vector size must be positive");
        TORCH_CHECK(qLoraScratch.dim() == 2, "q_lora_scratch must be 2D");
        TORCH_CHECK(hiddenStatesScratch.dim() == 2, "hidden_states_scratch must be 2D");
        TORCH_CHECK(qFp4Scratch.dim() == 3, "q_fp4_scratch must be 3D");
        TORCH_CHECK(kFp4Scratch.dim() == 2, "k_fp4_scratch must be 2D");
        TORCH_CHECK(kScaleScratch.dim() == 2, "k_scale_scratch must be 2D");
        TORCH_CHECK(indexerWeightsScratch.dim() == 2, "indexer_weights_scratch must be 2D");
        TORCH_CHECK(qScaleScratch.dim() == 3, "q_scale_scratch must be 3D");
        TORCH_CHECK(qLoraScratch.size(0) >= inputTokens, "q_lora_scratch batch is smaller than input");
        TORCH_CHECK(hiddenStatesScratch.size(0) >= inputTokens, "hidden_states_scratch batch is smaller than input");
        TORCH_CHECK(qFp4Scratch.size(0) >= inputTokens, "q_fp4_scratch batch is smaller than input");
        TORCH_CHECK(qFp4Scratch.size(1) == nHeads, "q_fp4_scratch head count must match n_heads");
        TORCH_CHECK(qFp4Scratch.size(2) == headDim / 2, "q_fp4_scratch packed dim must match head_dim / 2");
        TORCH_CHECK(kFp4Scratch.size(0) >= inputTokens, "k_fp4_scratch batch is smaller than input");
        TORCH_CHECK(kFp4Scratch.size(1) == headDim / 2, "k_fp4_scratch packed dim must match head_dim / 2");
        TORCH_CHECK(kScaleScratch.size(0) >= inputTokens, "k_scale_scratch batch is smaller than input");
        TORCH_CHECK(kScaleScratch.size(1) == 1, "k_scale_scratch must have one packed scale per token");
        TORCH_CHECK(
            indexerWeightsScratch.size(0) >= inputTokens, "indexer_weights_scratch batch is smaller than input");
        TORCH_CHECK(indexerWeightsScratch.size(1) == nHeads, "indexer_weights_scratch width must match n_heads");
        TORCH_CHECK(qScaleScratch.size(0) >= inputTokens, "q_scale_scratch batch is smaller than input");
        TORCH_CHECK(qScaleScratch.size(1) == nHeads, "q_scale_scratch head count must match n_heads");
        TORCH_CHECK(qScaleScratch.size(2) == 1, "q_scale_scratch must have one packed scale per token/head");
        TORCH_CHECK(
            qFp4Scratch.scalar_type() == at::ScalarType::Char || qFp4Scratch.scalar_type() == at::ScalarType::Byte,
            "q_fp4_scratch must be int8 or uint8");
        TORCH_CHECK(
            kFp4Scratch.scalar_type() == at::ScalarType::Char || kFp4Scratch.scalar_type() == at::ScalarType::Byte,
            "k_fp4_scratch must be int8 or uint8");
        TORCH_CHECK(kScaleScratch.scalar_type() == at::ScalarType::Int, "k_scale_scratch must be int32");
        TORCH_CHECK(
            indexerWeightsScratch.scalar_type() == at::ScalarType::Float, "indexer_weights_scratch must be float32");
        TORCH_CHECK(qScaleScratch.scalar_type() == at::ScalarType::Int, "q_scale_scratch must be int32");

        at::Tensor const& wqBWeight = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kAttentionIndexerWQBWeight, "DSA indexer wq_b weight");
        at::Tensor const& wkWeight = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kAttentionIndexerWKWeight, "DSA indexer wk weight");
        at::Tensor const& weightsProjWeight = getLayerTensor(layerIdx,
            DeepseekResidentLayerTensorSite::kAttentionIndexerWeightsProjWeight, "DSA indexer weights_proj weight");
        at::Tensor const& kNormWeight = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kAttentionIndexerKNormWeight, "DSA indexer k_norm weight");
        std::optional<at::Tensor> const kNormBias
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionIndexerKNormBias);
        at::Tensor const& rotaryCosSinRaw = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kAttentionIndexerRotaryCosSin, "DSA indexer rotary_cos_sin");

        at::Tensor qProjected;
        if (precomputedWqB.has_value())
        {
            TORCH_CHECK(precomputedWqB->is_cuda(), "precomputed DSA indexer wq_b output must be CUDA");
            TORCH_CHECK(precomputedWqB->dim() == 2, "precomputed DSA indexer wq_b output must be 2D");
            TORCH_CHECK(precomputedWqB->size(0) >= inputTokens,
                "precomputed DSA indexer wq_b output batch is smaller than input");
            TORCH_CHECK(precomputedWqB->size(1) >= nHeads * headDim,
                "precomputed DSA indexer wq_b output width is smaller than n_heads * head_dim");
            qProjected = precomputedWqB.value();
        }
        else
        {
            at::Tensor qIndexerInput = selectDsaIndexerWqBInput(layerIdx, qLoraScratch, wqBWeight, inputTokens);
            qProjected = runLayerDsaIndexerLinearToDtype(layerIdx, qIndexerInput, wqBWeight,
                DeepseekResidentLayerTensorSite::kAttentionIndexerWQBWeightScale,
                DeepseekResidentLayerTensorSite::kAttentionIndexerWQBWeightScale2,
                DeepseekResidentLayerTensorSite::kAttentionIndexerWQBInputScale,
                DeepseekResidentLayerTensorSite::kAttentionIndexerWQBAlpha, inputTokens, sfVecSize, allowedBackends,
                hiddenStatesScratch.scalar_type(), "dsa_wq_b");
        }
        at::Tensor indexerK;
        if (precomputedIndexerK.has_value())
        {
            TORCH_CHECK(precomputedIndexerK->is_cuda(), "precomputed DSA indexer wk output must be CUDA");
            TORCH_CHECK(precomputedIndexerK->dim() == 2, "precomputed DSA indexer wk output must be 2D");
            TORCH_CHECK(precomputedIndexerK->size(0) >= inputTokens,
                "precomputed DSA indexer wk output batch is smaller than input");
            TORCH_CHECK(precomputedIndexerK->size(1) >= headDim,
                "precomputed DSA indexer wk output width is smaller than head_dim");
            indexerK = precomputedIndexerK.value();
        }
        else
        {
            indexerK = runLayerDsaIndexerLinearToDtype(layerIdx, hiddenStatesScratch, wkWeight,
                DeepseekResidentLayerTensorSite::kAttentionIndexerWKWeightScale,
                DeepseekResidentLayerTensorSite::kAttentionIndexerWKWeightScale2,
                DeepseekResidentLayerTensorSite::kAttentionIndexerWKInputScale,
                DeepseekResidentLayerTensorSite::kAttentionIndexerWKAlpha, inputTokens, sfVecSize, allowedBackends,
                hiddenStatesScratch.scalar_type(), "dsa_wk");
        }
        TORCH_CHECK(qProjected.size(1) >= nHeads * headDim, "wq_b output width is smaller than n_heads * head_dim");
        TORCH_CHECK(indexerK.size(1) >= headDim, "wk output width is smaller than head_dim");
        at::Tensor indexerWeightsPrefix = indexerWeightsScratch.narrow(0, 0, inputTokens);
        double const combinedWeightScaleFactor = weightScaleFactor * precomputedWqBScale;
        if (precomputedIndexerWeights.has_value())
        {
            TORCH_CHECK(precomputedIndexerWeights->is_cuda(),
                "precomputed DSA indexer weights_proj output must be CUDA");
            TORCH_CHECK(precomputedIndexerWeights->dim() == 2,
                "precomputed DSA indexer weights_proj output must be 2D");
            TORCH_CHECK(precomputedIndexerWeights->size(0) >= inputTokens,
                "precomputed DSA indexer weights_proj output batch is smaller than input");
            TORCH_CHECK(precomputedIndexerWeights->size(1) >= nHeads,
                "precomputed DSA indexer weights_proj output width is smaller than n_heads");
            indexerWeightsPrefix.copy_(precomputedIndexerWeights->narrow(0, 0, inputTokens)
                    .narrow(1, 0, nHeads)
                    .to(at::ScalarType::Float));
        }
        else if (!tryRunLayerDsaIndexerLinearToOutput(layerIdx, hiddenStatesScratch, weightsProjWeight,
                DeepseekResidentLayerTensorSite::kAttentionIndexerWeightsProjWeightScale,
                DeepseekResidentLayerTensorSite::kAttentionIndexerWeightsProjInputScale,
                DeepseekResidentLayerTensorSite::kAttentionIndexerWeightsProjAlpha, indexerWeightsPrefix, inputTokens,
                sfVecSize, "dsa_weights_proj_out"))
        {
            at::Tensor indexerWeights = runLayerDsaIndexerLinearToFloat(layerIdx, hiddenStatesScratch,
                weightsProjWeight, DeepseekResidentLayerTensorSite::kAttentionIndexerWeightsProjWeightScale,
                DeepseekResidentLayerTensorSite::kAttentionIndexerWeightsProjWeightScale2,
                DeepseekResidentLayerTensorSite::kAttentionIndexerWeightsProjInputScale,
                DeepseekResidentLayerTensorSite::kAttentionIndexerWeightsProjAlpha, inputTokens, sfVecSize,
                allowedBackends, "dsa_weights_proj");
            TORCH_CHECK(indexerWeights.size(1) >= nHeads, "weights_proj output width is smaller than n_heads");
            indexerWeightsPrefix.copy_(indexerWeights.narrow(0, 0, inputTokens)
                    .narrow(1, 0, nHeads)
                    .to(at::ScalarType::Float));
        }
        if (combinedWeightScaleFactor != 1.0)
        {
            indexerWeightsPrefix.mul_(combinedWeightScaleFactor);
        }

        at::Tensor q = qProjected.narrow(0, 0, inputTokens)
                           .narrow(1, 0, nHeads * headDim)
                           .reshape({inputTokens, nHeads, headDim});
        at::Tensor indexerKForNorm = indexerK.narrow(0, 0, inputTokens).narrow(1, 0, headDim);
        at::Tensor kNorm = layerNorm2d(indexerKForNorm, kNormWeight, kNormBias, inputTokens, eps);
        at::Tensor rotaryCosSin = rotaryCosSinRaw.reshape({rotaryCosSinRaw.size(0), -1}).to(at::ScalarType::Float);
        TORCH_CHECK(rotaryCosSin.dim() == 2, "rotary_cos_sin must be 2D after flattening");
        TORCH_CHECK(rotaryCosSin.size(1) == ropeDim, "rotary_cos_sin width must match rope_dim");

        at::Tensor posFlat
            = positionIds.reshape({positionIds.numel()}).narrow(0, 0, inputTokens).to(at::ScalarType::Int).contiguous();
        at::Tensor posQ
            = posFlat.unsqueeze(1).expand({inputTokens, nHeads}).reshape({inputTokens * nHeads}).contiguous();

        at::Tensor qPe = q.narrow(2, 0, ropeDim).reshape({inputTokens * nHeads, ropeDim});
        at::Tensor qNope = q.narrow(2, ropeDim, headDim - ropeDim).reshape({inputTokens * nHeads, headDim - ropeDim});
        at::Tensor kPe = kNorm.narrow(1, 0, ropeDim);
        at::Tensor kNope = kNorm.narrow(1, ropeDim, headDim - ropeDim);
        at::Tensor rotaryCosSinContiguous = rotaryCosSin.is_contiguous() ? rotaryCosSin : rotaryCosSin.contiguous();

        invokeFusedRopeCatFp4Into(qFp4Scratch.narrow(0, 0, inputTokens), qScaleScratch.narrow(0, 0, inputTokens), qPe,
            qNope, rotaryCosSinContiguous, posQ);
        invokeFusedRopeCatFp4Into(kFp4Scratch.narrow(0, 0, inputTokens), kScaleScratch.narrow(0, 0, inputTokens), kPe,
            kNope, rotaryCosSinContiguous, posFlat);
        return {qFp4Scratch, kFp4Scratch, kScaleScratch, indexerWeightsScratch, qScaleScratch};
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> runLayerDsaIndexerFp4Projection(
        int64_t layerIdx, at::Tensor const& qLoraScratch, at::Tensor const& hiddenStatesScratch,
        at::Tensor const& positionIds, at::Tensor const& qFp4Scratch, at::Tensor const& kFp4Scratch,
        at::Tensor const& kScaleScratch, at::Tensor const& indexerWeightsScratch, at::Tensor const& qScaleScratch,
        int64_t inputTokens, int64_t nHeads, int64_t headDim, int64_t ropeDim, double eps, double weightScaleFactor,
        int64_t sfVecSize, std::string const& allowedBackends) const
    {
        return runLayerDsaIndexerFp4ProjectionImpl(layerIdx, qLoraScratch, hiddenStatesScratch, positionIds,
            qFp4Scratch, kFp4Scratch, kScaleScratch, indexerWeightsScratch, qScaleScratch, std::nullopt, 1.0,
            std::nullopt, std::nullopt, inputTokens, nHeads, headDim, ropeDim, eps, weightScaleFactor, sfVecSize,
            allowedBackends);
    }

    at::Tensor runNvfp4LinearTyped(at::Tensor const& input, at::Tensor const& weight, at::Tensor const& weightScale,
        at::Tensor const& inputScale, at::Tensor const& alpha, int64_t inputTokens, int64_t sfVecSize,
        std::string const& allowedBackends, at::ScalarType outputDtype, char const* debugLabel) const
    {
        TORCH_CHECK(input.is_cuda(), "NVFP4 linear input must be a CUDA tensor");
        TORCH_CHECK(weight.is_cuda(), "NVFP4 linear weight must be a CUDA tensor");
        TORCH_CHECK(weightScale.is_cuda(), "NVFP4 linear weight scale must be a CUDA tensor");
        TORCH_CHECK(inputScale.is_cuda(), "NVFP4 linear input scale must be a CUDA tensor");
        TORCH_CHECK(alpha.is_cuda(), "NVFP4 linear alpha must be a CUDA tensor");
        TORCH_CHECK(inputTokens > 0, "input_tokens must be positive");
        TORCH_CHECK(sfVecSize > 0, "NVFP4 linear scaling vector size must be positive");
        TORCH_CHECK(input.dim() == 2, "NVFP4 linear input must be 2D");
        TORCH_CHECK(input.size(0) >= inputTokens, "NVFP4 linear input batch is smaller than input_tokens");
        TORCH_CHECK(weight.dim() == 2, "NVFP4 linear weight must be 2D");
        TORCH_CHECK(weight.size(1) > 0, "NVFP4 linear weight input dim must be positive");
        TORCH_CHECK(input.size(1) * 2 == weight.size(1) || input.size(1) == weight.size(1) * 2
                || input.size(1) == weight.size(1),
            "NVFP4 linear input/weight dimensions are incompatible at ", debugLabel, ": input_shape=", input.sizes(),
            " weight_shape=", weight.sizes(), " input_tokens=", inputTokens, " sf_vec_size=", sfVecSize,
            " output_dtype=", outputDtype, " allowed_backends=", allowedBackends);

        at::Tensor inputPrefix = input.narrow(0, 0, inputTokens).contiguous();
        auto fp4 = callTrtllmFp4Quantize(inputPrefix, inputScale, sfVecSize);
        return callTrtllmNvfp4Gemm(
            std::get<0>(fp4), weight, std::get<1>(fp4), weightScale, alpha, outputDtype, 0, allowedBackends);
    }

    bool tryRunNvfp4LinearTypedToOutput(at::Tensor const& input, at::Tensor const& weight,
        at::Tensor const& weightScale, at::Tensor const& inputScale, at::Tensor const& alpha, at::Tensor const& output,
        int64_t inputTokens, int64_t sfVecSize, char const* debugLabel) const
    {
        if (!canRunCudaCoreNvfp4GemmOut(input, weight, output, inputTokens))
        {
            return false;
        }
        TORCH_CHECK(weightScale.is_cuda(), "NVFP4 linear output path weight scale must be a CUDA tensor");
        TORCH_CHECK(inputScale.is_cuda(), "NVFP4 linear output path input scale must be a CUDA tensor");
        TORCH_CHECK(alpha.is_cuda(), "NVFP4 linear output path alpha must be a CUDA tensor");
        TORCH_CHECK(sfVecSize > 0, "NVFP4 linear output path scaling vector size must be positive");
        TORCH_CHECK(input.size(1) * 2 == weight.size(1) || input.size(1) == weight.size(1) * 2
                || input.size(1) == weight.size(1),
            "NVFP4 linear output path input/weight dimensions are incompatible at ", debugLabel,
            ": input_shape=", input.sizes(), " weight_shape=", weight.sizes(), " input_tokens=", inputTokens,
            " sf_vec_size=", sfVecSize, " output_dtype=", output.scalar_type());

        at::Tensor inputPrefix = input.narrow(0, 0, inputTokens).contiguous();
        auto fp4 = callTrtllmFp4Quantize(inputPrefix, inputScale, sfVecSize);
        callCudaCoreNvfp4GemmOut(
            std::get<0>(fp4), weight, std::get<1>(fp4), weightScale, alpha, output, inputTokens, debugLabel);
        return true;
    }

    at::Tensor runLayerDsaIndexerLinearWithDtypes(int64_t layerIdx, at::Tensor const& input, at::Tensor const& weight,
        DeepseekResidentLayerTensorSite weightScaleSite, DeepseekResidentLayerTensorSite weightScale2Site,
        DeepseekResidentLayerTensorSite inputScaleSite, DeepseekResidentLayerTensorSite alphaSite, int64_t inputTokens,
        int64_t sfVecSize, std::string const& allowedBackends, at::ScalarType gemmOutputDtype,
        at::ScalarType finalDtype, char const* debugLabel) const
    {
        std::optional<at::Tensor> const weightScale = getOptionalLayerTensor(layerIdx, weightScaleSite);
        std::optional<at::Tensor> const weightScale2 = getOptionalLayerTensor(layerIdx, weightScale2Site);
        std::optional<at::Tensor> const inputScale = getOptionalLayerTensor(layerIdx, inputScaleSite);
        std::optional<at::Tensor> const alpha = getOptionalLayerTensor(layerIdx, alphaSite);
        if (weightScale.has_value() && inputScale.has_value() && alpha.has_value())
        {
            return runNvfp4LinearTyped(input, weight, weightScale.value(), inputScale.value(), alpha.value(),
                inputTokens, sfVecSize, allowedBackends, gemmOutputDtype, debugLabel)
                .to(finalDtype);
        }
        if (weightScale.has_value() && weightScale2.has_value())
        {
            at::Tensor inputPrefix = input.narrow(0, 0, inputTokens).contiguous();
            at::Tensor inputAmax = at::amax(at::abs(inputPrefix)).to(at::ScalarType::Float);
            at::Tensor globalMax = at::full({1}, 448.0 * 6.0, inputPrefix.options().dtype(at::ScalarType::Float));
            at::Tensor dynamicInputScale = globalMax / inputAmax;
            at::Tensor dynamicAlpha = (inputAmax / globalMax)
                * weightScale2.value().reshape({-1}).narrow(0, 0, 1).to(at::ScalarType::Float).contiguous();
            return runNvfp4LinearTyped(input, weight, weightScale.value(), dynamicInputScale, dynamicAlpha, inputTokens,
                sfVecSize, allowedBackends, gemmOutputDtype, debugLabel)
                .to(finalDtype);
        }
        at::Tensor inputPrefix = input.narrow(0, 0, inputTokens).to(at::ScalarType::Float);
        if (inputPrefix.size(1) != weight.size(1))
        {
            TORCH_CHECK(inputPrefix.size(1) >= weight.size(1),
                "dense DSA indexer input width is smaller than weight input dim");
            inputPrefix = inputPrefix.narrow(1, 0, weight.size(1));
        }
        return at::matmul(inputPrefix, weight.to(at::ScalarType::Float).t()).to(finalDtype);
    }

    at::Tensor runLayerDsaIndexerLinearToDtype(int64_t layerIdx, at::Tensor const& input, at::Tensor const& weight,
        DeepseekResidentLayerTensorSite weightScaleSite, DeepseekResidentLayerTensorSite weightScale2Site,
        DeepseekResidentLayerTensorSite inputScaleSite, DeepseekResidentLayerTensorSite alphaSite, int64_t inputTokens,
        int64_t sfVecSize, std::string const& allowedBackends, at::ScalarType outputDtype, char const* debugLabel) const
    {
        return runLayerDsaIndexerLinearWithDtypes(layerIdx, input, weight, weightScaleSite, weightScale2Site,
            inputScaleSite, alphaSite, inputTokens, sfVecSize, allowedBackends, outputDtype, outputDtype, debugLabel);
    }

    at::Tensor runLayerDsaIndexerLinearToFloat(int64_t layerIdx, at::Tensor const& input, at::Tensor const& weight,
        DeepseekResidentLayerTensorSite weightScaleSite, DeepseekResidentLayerTensorSite weightScale2Site,
        DeepseekResidentLayerTensorSite inputScaleSite, DeepseekResidentLayerTensorSite alphaSite, int64_t inputTokens,
        int64_t sfVecSize, std::string const& allowedBackends, char const* debugLabel) const
    {
        return runLayerDsaIndexerLinearWithDtypes(layerIdx, input, weight, weightScaleSite, weightScale2Site,
            inputScaleSite, alphaSite, inputTokens, sfVecSize, allowedBackends, input.scalar_type(),
            at::ScalarType::Float, debugLabel);
    }

    DeepseekResidentFusedQbWqBCache const* getFusedQbWqBCache(int64_t layerIdx) const
    {
        if (!residentFusedQbWqBEnabled())
        {
            return nullptr;
        }
        auto const iter = mFusedQbWqBCache.find(layerIdx);
        if (iter != mFusedQbWqBCache.end())
        {
            return iter->second.eligible ? &iter->second : nullptr;
        }
        if (isStreamCapturing(at::cuda::getCurrentCUDAStream().stream()))
        {
            return nullptr;
        }

        DeepseekResidentFusedQbWqBCache cache;
        auto reject = [&]()
        {
            auto inserted = mFusedQbWqBCache.emplace(layerIdx, std::move(cache));
            return inserted.first->second.eligible ? &inserted.first->second : nullptr;
        };

        std::optional<at::Tensor> const qBWeight
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionQBProjWeight);
        std::optional<at::Tensor> const qBWeightScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionQBProjWeightScale);
        std::optional<at::Tensor> const qBWeightScale2
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionQBProjWeightScale2);
        std::optional<at::Tensor> const qBInputScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionQBProjInputScale);
        std::optional<at::Tensor> const qBAlpha
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionQBProjAlpha);
        std::optional<at::Tensor> const wqBWeight
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionIndexerWQBWeight);
        std::optional<at::Tensor> const wqBWeightScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionIndexerWQBWeightScale);
        std::optional<at::Tensor> const wqBWeightScale2
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionIndexerWQBWeightScale2);
        if (!qBWeight.has_value() || !qBWeightScale.has_value() || !qBWeightScale2.has_value()
            || !qBInputScale.has_value() || !qBAlpha.has_value() || !wqBWeight.has_value()
            || !wqBWeightScale.has_value() || !wqBWeightScale2.has_value())
        {
            return reject();
        }
        if (qBWeight->scalar_type() != at::ScalarType::Byte || wqBWeight->scalar_type() != at::ScalarType::Byte)
        {
            return reject();
        }
        if (qBWeight->dim() != 2 || wqBWeight->dim() != 2 || qBWeightScale->dim() != 1
            || wqBWeightScale->dim() != 1 || qBWeight->size(1) != wqBWeight->size(1)
            || qBWeight->size(0) % 128 != 0)
        {
            return reject();
        }
        double const qBScale2 = qBWeightScale2->reshape({-1}).narrow(0, 0, 1).to(at::ScalarType::Double).item<double>();
        double const wqBScale2
            = wqBWeightScale2->reshape({-1}).narrow(0, 0, 1).to(at::ScalarType::Double).item<double>();
        if (!std::isfinite(qBScale2) || !std::isfinite(wqBScale2) || qBScale2 == 0.0)
        {
            return reject();
        }

        cache.weight = at::cat({qBWeight.value(), wqBWeight.value()}, 0).contiguous();
        cache.weightScale = at::cat({qBWeightScale.value(), wqBWeightScale.value()}, 0).contiguous();
        cache.inputScale = qBInputScale.value();
        cache.alpha = qBAlpha.value();
        cache.qbOut = qBWeight->size(0);
        cache.wqOut = wqBWeight->size(0);
        cache.wqOutScale = wqBScale2 / qBScale2;
        cache.outDtype = at::ScalarType::BFloat16;
        cache.eligible = true;
        auto inserted = mFusedQbWqBCache.emplace(layerIdx, std::move(cache));
        return &inserted.first->second;
    }

    DeepseekResidentFusedKvAWkWpCache const* getFusedKvAWkWpCache(
        int64_t layerIdx, int64_t kvOut) const
    {
        if (!residentFusedKvAWkWpEnabled())
        {
            return nullptr;
        }
        auto const iter = mFusedKvAWkWpCache.find(layerIdx);
        if (iter != mFusedKvAWkWpCache.end())
        {
            return iter->second.eligible ? &iter->second : nullptr;
        }
        if (isStreamCapturing(at::cuda::getCurrentCUDAStream().stream()))
        {
            return nullptr;
        }

        DeepseekResidentFusedKvAWkWpCache cache;
        auto reject = [&]()
        {
            auto inserted = mFusedKvAWkWpCache.emplace(layerIdx, std::move(cache));
            return inserted.first->second.eligible ? &inserted.first->second : nullptr;
        };

        std::optional<at::Tensor> const kvAWeight
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionKvAProjWeight);
        std::optional<at::Tensor> const kvAWeightScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionKvAProjWeightScale);
        std::optional<at::Tensor> const kvAWeightScale2
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionKvAProjWeightScale2);
        std::optional<at::Tensor> const kvAInputScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionKvAProjInputScale);
        std::optional<at::Tensor> const kvAAlpha
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionKvAProjAlpha);
        std::optional<at::Tensor> const wkWeight
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionIndexerWKWeight);
        std::optional<at::Tensor> const wkWeightScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionIndexerWKWeightScale);
        std::optional<at::Tensor> const wkWeightScale2
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionIndexerWKWeightScale2);
        std::optional<at::Tensor> const wpWeight
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionIndexerWeightsProjWeight);
        std::optional<at::Tensor> const wpWeightScale = getOptionalLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kAttentionIndexerWeightsProjWeightScale);
        std::optional<at::Tensor> const wpWeightScale2 = getOptionalLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kAttentionIndexerWeightsProjWeightScale2);
        if (!kvAWeight.has_value() || !kvAWeightScale.has_value() || !kvAWeightScale2.has_value()
            || !kvAInputScale.has_value() || !kvAAlpha.has_value() || !wkWeight.has_value()
            || !wkWeightScale.has_value() || !wkWeightScale2.has_value() || !wpWeight.has_value()
            || !wpWeightScale.has_value() || !wpWeightScale2.has_value())
        {
            return reject();
        }
        if (kvAWeight->scalar_type() != at::ScalarType::Byte || wkWeight->scalar_type() != at::ScalarType::Byte
            || wpWeight->scalar_type() != at::ScalarType::Byte)
        {
            return reject();
        }
        if (kvAWeight->dim() != 2 || wkWeight->dim() != 2 || wpWeight->dim() != 2 || kvAWeightScale->dim() != 1
            || wkWeightScale->dim() != 1 || wpWeightScale->dim() != 1 || kvAWeight->size(1) != wkWeight->size(1)
            || kvAWeight->size(1) != wpWeight->size(1) || kvAWeight->size(0) != kvOut || wkWeight->size(0) % 128 != 0)
        {
            return reject();
        }

        int64_t const kBlocks = (kvAWeight->size(1) * 2) / 16;
        int64_t const kBlocksPadded = ceilDivPositive(kBlocks, 4) * 4;
        int64_t const kvOutPadded = ceilDivPositive(kvOut, 128) * 128;
        int64_t const wkwpOut = wkWeight->size(0) + wpWeight->size(0);
        int64_t const wkwpOutPadded = ceilDivPositive(wkwpOut, 128) * 128;
        if (kvAWeightScale->numel() != kvOutPadded * kBlocksPadded
            || (wkWeightScale->numel() + wpWeightScale->numel()) != wkwpOutPadded * kBlocksPadded)
        {
            return reject();
        }

        double const kvScale2
            = kvAWeightScale2->reshape({-1}).narrow(0, 0, 1).to(at::ScalarType::Double).item<double>();
        double const wkScale2
            = wkWeightScale2->reshape({-1}).narrow(0, 0, 1).to(at::ScalarType::Double).item<double>();
        double const wpScale2
            = wpWeightScale2->reshape({-1}).narrow(0, 0, 1).to(at::ScalarType::Double).item<double>();
        if (!std::isfinite(kvScale2) || !std::isfinite(wkScale2) || !std::isfinite(wpScale2) || kvScale2 == 0.0)
        {
            return reject();
        }

        int64_t const padRows = kvOutPadded - kvOut;
        if (padRows > 0)
        {
            at::Tensor padWeight = at::zeros({padRows, kvAWeight->size(1)}, kvAWeight->options());
            cache.weight = at::cat({kvAWeight.value(), padWeight, wkWeight.value(), wpWeight.value()}, 0).contiguous();
        }
        else
        {
            cache.weight = at::cat({kvAWeight.value(), wkWeight.value(), wpWeight.value()}, 0).contiguous();
        }
        cache.weightScale
            = at::cat({kvAWeightScale.value(), wkWeightScale.value(), wpWeightScale.value()}, 0).contiguous();
        cache.inputScale = kvAInputScale.value();
        cache.alpha = kvAAlpha.value();
        cache.kvOut = kvOut;
        cache.kvOutPadded = kvOutPadded;
        cache.wkOut = wkWeight->size(0);
        cache.wpOut = wpWeight->size(0);
        cache.wkOutScale = wkScale2 / kvScale2;
        cache.wpOutScale = wpScale2 / kvScale2;
        cache.outDtype = at::ScalarType::BFloat16;
        cache.eligible = true;
        auto inserted = mFusedKvAWkWpCache.emplace(layerIdx, std::move(cache));
        return &inserted.first->second;
    }

    bool tryRunLayerDsaIndexerLinearToOutput(int64_t layerIdx, at::Tensor const& input, at::Tensor const& weight,
        DeepseekResidentLayerTensorSite weightScaleSite, DeepseekResidentLayerTensorSite inputScaleSite,
        DeepseekResidentLayerTensorSite alphaSite, at::Tensor const& output, int64_t inputTokens, int64_t sfVecSize,
        char const* debugLabel) const
    {
        std::optional<at::Tensor> const weightScale = getOptionalLayerTensor(layerIdx, weightScaleSite);
        std::optional<at::Tensor> const inputScale = getOptionalLayerTensor(layerIdx, inputScaleSite);
        std::optional<at::Tensor> const alpha = getOptionalLayerTensor(layerIdx, alphaSite);
        if (!weightScale.has_value() || !inputScale.has_value() || !alpha.has_value())
        {
            return false;
        }
        return tryRunNvfp4LinearTypedToOutput(input, weight, weightScale.value(), inputScale.value(), alpha.value(),
            output, inputTokens, sfVecSize, debugLabel);
    }

    at::Tensor selectDsaIndexerWqBInput(
        int64_t layerIdx, at::Tensor const& qLoraScratch, at::Tensor const& wqBWeight, int64_t inputTokens) const
    {
        TORCH_CHECK(qLoraScratch.dim() == 2, "q_lora_scratch must be 2D");
        TORCH_CHECK(qLoraScratch.size(0) >= inputTokens, "q_lora_scratch batch is smaller than input_tokens");
        TORCH_CHECK(wqBWeight.dim() == 2, "DSA indexer wq_b weight must be 2D");
        TORCH_CHECK(wqBWeight.size(1) > 0, "DSA indexer wq_b weight input dim must be positive");
        int64_t const qLoraScratchWidth = qLoraScratch.size(1);
        int64_t const weightInputDim = wqBWeight.size(1);
        TORCH_CHECK(qLoraScratchWidth > 0, "q_lora_scratch width must be positive");

        std::optional<at::Tensor> const weightScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionIndexerWQBWeightScale);
        std::optional<at::Tensor> const inputScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionIndexerWQBInputScale);
        std::optional<at::Tensor> const alpha
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionIndexerWQBAlpha);
        std::optional<at::Tensor> const weightScale2
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionIndexerWQBWeightScale2);
        bool const hasStaticNvfp4Scales = weightScale.has_value() && inputScale.has_value() && alpha.has_value();
        bool const hasDynamicNvfp4Scales = weightScale.has_value() && weightScale2.has_value();
        bool const hasNvfp4Scales = hasStaticNvfp4Scales || hasDynamicNvfp4Scales;
        int64_t selectedWidth = weightInputDim;
        if (hasNvfp4Scales)
        {
            std::array<int64_t, 3> const candidates{
                weightInputDim * 2,
                weightInputDim,
                weightInputDim % 2 == 0 ? weightInputDim / 2 : 0,
            };
            selectedWidth = 0;
            for (int64_t const candidate : candidates)
            {
                if (candidate > 0 && candidate <= qLoraScratchWidth)
                {
                    selectedWidth = candidate;
                    break;
                }
            }
            TORCH_CHECK(selectedWidth > 0,
                "q_lora_scratch width is incompatible with packed DSA indexer wq_b weight input dim");
        }
        TORCH_CHECK(
            selectedWidth <= qLoraScratchWidth, "q_lora_scratch width is smaller than DSA indexer wq_b input dim");
        return qLoraScratch.narrow(1, 0, selectedWidth);
    }

    at::Tensor runLinearMaybeNvfp4ToFloat(at::Tensor const& input, at::Tensor const& weight,
        std::optional<at::Tensor> const& weightScale, std::optional<at::Tensor> const& inputScale,
        std::optional<at::Tensor> const& alpha, int64_t inputTokens, int64_t sfVecSize,
        std::string const& allowedBackends, char const* debugLabel) const
    {
        if (weightScale.has_value() && inputScale.has_value() && alpha.has_value())
        {
            return runNvfp4LinearTyped(input, weight, weightScale.value(), inputScale.value(), alpha.value(),
                inputTokens, sfVecSize, allowedBackends, at::ScalarType::Float, debugLabel);
        }
        at::Tensor inputPrefix = input.narrow(0, 0, inputTokens).to(at::ScalarType::Float);
        return at::matmul(inputPrefix, weight.to(at::ScalarType::Float).t());
    }

    bool tryRunLinearMaybeNvfp4ToOutput(at::Tensor const& input, at::Tensor const& weight,
        std::optional<at::Tensor> const& weightScale, std::optional<at::Tensor> const& inputScale,
        std::optional<at::Tensor> const& alpha, at::Tensor const& output, int64_t inputTokens, int64_t sfVecSize,
        char const* debugLabel) const
    {
        if (!weightScale.has_value() || !inputScale.has_value() || !alpha.has_value())
        {
            return false;
        }
        return tryRunNvfp4LinearTypedToOutput(input, weight, weightScale.value(), inputScale.value(), alpha.value(),
            output, inputTokens, sfVecSize, debugLabel);
    }

    bool tryRunAttentionTailFp4OutGate(at::Tensor const& attentionCoreOutputScratch,
        at::Tensor const& attentionInputScratch, at::Tensor const& attentionGateScratch,
        at::Tensor const& attentionHiddenStatesScratch, at::Tensor const& gateWeight,
        std::optional<at::Tensor> const& gateWeightScale, std::optional<at::Tensor> const& gateInputScale,
        std::optional<at::Tensor> const& gateAlpha, at::Tensor const& oProjWeight,
        std::optional<at::Tensor> const& oProjWeightScale, std::optional<at::Tensor> const& oProjInputScale,
        std::optional<at::Tensor> const& oProjAlpha, int64_t inputTokens) const
    {
        constexpr int64_t kSfVecSize = 16;
        recordResidentAttentionTailFp4OutGateStat(ResidentAttentionTailFp4OutGateStat::kAttempts);
        if (!residentAttentionTailFp4OutGateEnabled())
        {
            recordResidentAttentionTailFp4OutGateStat(ResidentAttentionTailFp4OutGateStat::kDisabled);
            return false;
        }
        if (inputTokens <= 0)
        {
            recordResidentAttentionTailFp4OutGateStat(ResidentAttentionTailFp4OutGateStat::kInvalidInputTokens);
            return false;
        }
        bool const gateHasAnyNvfp4Scale
            = gateWeightScale.has_value() || gateInputScale.has_value() || gateAlpha.has_value();
        bool const gateHasNvfp4Scales
            = gateWeightScale.has_value() && gateInputScale.has_value() && gateAlpha.has_value();
        if (gateHasAnyNvfp4Scale && !gateWeightScale.has_value())
        {
            recordResidentAttentionTailFp4OutGateStat(ResidentAttentionTailFp4OutGateStat::kMissingGateWeightScale);
            return false;
        }
        if (gateHasAnyNvfp4Scale && !gateInputScale.has_value())
        {
            recordResidentAttentionTailFp4OutGateStat(ResidentAttentionTailFp4OutGateStat::kMissingGateInputScale);
            return false;
        }
        if (gateHasAnyNvfp4Scale && !gateAlpha.has_value())
        {
            recordResidentAttentionTailFp4OutGateStat(ResidentAttentionTailFp4OutGateStat::kMissingGateAlpha);
            return false;
        }
        if (!oProjWeightScale.has_value())
        {
            recordResidentAttentionTailFp4OutGateStat(ResidentAttentionTailFp4OutGateStat::kMissingOProjWeightScale);
            return false;
        }
        if (!oProjInputScale.has_value())
        {
            recordResidentAttentionTailFp4OutGateStat(ResidentAttentionTailFp4OutGateStat::kMissingOProjInputScale);
            return false;
        }
        if (!oProjAlpha.has_value())
        {
            recordResidentAttentionTailFp4OutGateStat(ResidentAttentionTailFp4OutGateStat::kMissingOProjAlpha);
            return false;
        }
        if (attentionCoreOutputScratch.dim() != 2 || attentionInputScratch.dim() != 2
            || attentionGateScratch.dim() != 2 || attentionHiddenStatesScratch.dim() != 2 || gateWeight.dim() != 2
            || oProjWeight.dim() != 2 || attentionCoreOutputScratch.size(0) < inputTokens
            || attentionInputScratch.size(0) < inputTokens || attentionGateScratch.size(0) < inputTokens
            || attentionHiddenStatesScratch.size(0) < inputTokens || gateWeight.size(0) < attentionGateScratch.size(1)
            || (!gateHasNvfp4Scales && gateWeight.size(1) != attentionInputScratch.size(1))
            || attentionGateScratch.size(1) != attentionCoreOutputScratch.size(1)
            || attentionHiddenStatesScratch.size(1) != oProjWeight.size(0)
            || attentionCoreOutputScratch.scalar_type() != attentionGateScratch.scalar_type())
        {
            recordResidentAttentionTailFp4OutGateStat(ResidentAttentionTailFp4OutGateStat::kShapeRejected);
            return false;
        }

        at::Tensor corePrefix = attentionCoreOutputScratch.narrow(0, 0, inputTokens);
        at::Tensor inputPrefix = attentionInputScratch.narrow(0, 0, inputTokens);
        at::Tensor gatePrefix = attentionGateScratch.narrow(0, 0, inputTokens);
        at::Tensor outputPrefix = attentionHiddenStatesScratch.narrow(0, 0, inputTokens);
        if (!corePrefix.is_contiguous() || !gatePrefix.is_contiguous()
            || (corePrefix.scalar_type() != at::ScalarType::Half
                && corePrefix.scalar_type() != at::ScalarType::BFloat16)
            || (!gateHasNvfp4Scales && gateWeight.scalar_type() != corePrefix.scalar_type())
            || oProjWeight.size(1) <= 0)
        {
            recordResidentAttentionTailFp4OutGateStat(ResidentAttentionTailFp4OutGateStat::kContiguousDtypeRejected);
            return false;
        }

        if (gateHasNvfp4Scales
            && !tryRunLinearMaybeNvfp4ToOutput(inputPrefix, gateWeight, gateWeightScale, gateInputScale, gateAlpha,
                gatePrefix, inputTokens, kSfVecSize, "attention_gate_proj_fp4out_gate"))
        {
            at::Tensor gateOutput = runNvfp4LinearTyped(inputPrefix, gateWeight, gateWeightScale.value(),
                gateInputScale.value(), gateAlpha.value(), inputTokens, kSfVecSize, "cutlass,cublaslt,cuda_core",
                gatePrefix.scalar_type(), "attention_gate_proj_fp4out_gate");
            TORCH_CHECK(gateOutput.dim() == 2, "attention fused gate_proj output must be 2D");
            TORCH_CHECK(gateOutput.size(0) >= inputTokens,
                "attention fused gate_proj output batch is smaller than input");
            TORCH_CHECK(gateOutput.size(1) >= attentionCoreOutputScratch.size(1),
                "attention fused gate_proj output dim must cover attention core output dim");
            gatePrefix.copy_(gateOutput.narrow(0, 0, inputTokens).narrow(1, 0, attentionCoreOutputScratch.size(1)));
        }
        else if (!gateHasNvfp4Scales)
        {
            at::Tensor gateOutput = at::matmul(inputPrefix, gateWeight.t());
            TORCH_CHECK(gateOutput.dim() == 2, "attention fused BF16 gate_proj output must be 2D");
            TORCH_CHECK(gateOutput.size(0) >= inputTokens,
                "attention fused BF16 gate_proj output batch is smaller than input");
            TORCH_CHECK(gateOutput.size(1) >= attentionCoreOutputScratch.size(1),
                "attention fused BF16 gate_proj output dim must cover attention core output dim");
            gatePrefix.copy_(gateOutput.narrow(0, 0, inputTokens).narrow(1, 0, attentionCoreOutputScratch.size(1)));
        }

        std::optional<std::tuple<at::Tensor, at::Tensor>> gatedFp4
            = tryCallFusedSigmoidMulQuantNvfp4Swizzled(corePrefix, gatePrefix, oProjInputScale.value(), kSfVecSize);
        if (!gatedFp4.has_value())
        {
            recordResidentAttentionTailFp4OutGateStat(ResidentAttentionTailFp4OutGateStat::kFusedQuantRejected);
            return false;
        }

        if (canRunCudaCoreNvfp4GemmOut(corePrefix, oProjWeight, outputPrefix, inputTokens))
        {
            callCudaCoreNvfp4GemmOut(std::get<0>(gatedFp4.value()), oProjWeight, std::get<1>(gatedFp4.value()),
                oProjWeightScale.value(), oProjAlpha.value(), outputPrefix, inputTokens,
                "attention_o_proj_fp4out_gate");
            recordResidentAttentionTailFp4OutGateVisit();
            return true;
        }

        at::Tensor output = callTrtllmNvfp4Gemm(std::get<0>(gatedFp4.value()), oProjWeight,
            std::get<1>(gatedFp4.value()), oProjWeightScale.value(), oProjAlpha.value(), outputPrefix.scalar_type(), 0,
            "cutlass,cublaslt,cuda_core");
        TORCH_CHECK(output.dim() == 2, "attention fused o_proj output must be 2D");
        TORCH_CHECK(output.size(0) >= inputTokens, "attention fused o_proj output batch is smaller than input");
        TORCH_CHECK(
            output.size(1) >= outputPrefix.size(1), "attention fused o_proj output dim is smaller than expected");
        outputPrefix.copy_(output.narrow(0, 0, inputTokens).narrow(1, 0, outputPrefix.size(1)));
        recordResidentAttentionTailFp4OutGateVisit();
        return true;
    }

    at::Tensor runSwiGluMlpToFloat(at::Tensor const& input, at::Tensor const& gateUpWeight,
        at::Tensor const& downWeight, std::optional<at::Tensor> const& gateUpWeightScale,
        std::optional<at::Tensor> const& gateUpInputScale, std::optional<at::Tensor> const& gateUpAlpha,
        std::optional<at::Tensor> const& downWeightScale, std::optional<at::Tensor> const& downInputScale,
        std::optional<at::Tensor> const& downAlpha, int64_t inputTokens, int64_t sfVecSize,
        std::string const& allowedBackends, char const* gateUpDebugLabel, char const* downDebugLabel) const
    {
        at::Tensor gateUp = runLinearMaybeNvfp4ToFloat(input, gateUpWeight, gateUpWeightScale, gateUpInputScale,
            gateUpAlpha, inputTokens, sfVecSize, allowedBackends, gateUpDebugLabel);
        TORCH_CHECK(gateUp.dim() == 2, "SwiGLU gate_up output must be 2D");
        TORCH_CHECK(gateUp.size(0) >= inputTokens, "SwiGLU gate_up output batch is smaller than input");
        TORCH_CHECK(gateUp.size(1) % 2 == 0, "SwiGLU gate_up output dim must be even");
        int64_t const intermediateSize = gateUp.size(1) / 2;
        at::Tensor gate = gateUp.narrow(1, 0, intermediateSize);
        at::Tensor up = gateUp.narrow(1, intermediateSize, intermediateSize);
        at::Tensor activated = (gate * at::sigmoid(gate)) * up;
        return runLinearMaybeNvfp4ToFloat(activated.to(input.scalar_type()), downWeight, downWeightScale,
            downInputScale, downAlpha, inputTokens, sfVecSize, allowedBackends, downDebugLabel);
    }

    at::Tensor runSwiGluMlpToScratch(at::Tensor const& input, at::Tensor const& gateUpWeight,
        at::Tensor const& downWeight, std::optional<at::Tensor> const& gateUpWeightScale,
        std::optional<at::Tensor> const& gateUpInputScale, std::optional<at::Tensor> const& gateUpAlpha,
        std::optional<at::Tensor> const& downWeightScale, std::optional<at::Tensor> const& downInputScale,
        std::optional<at::Tensor> const& downAlpha, at::Tensor const& gateUpScratch,
        at::Tensor const& intermediateScratch, at::Tensor const& outputScratch, int64_t inputTokens, int64_t sfVecSize,
        std::string const& allowedBackends, char const* gateUpDebugLabel, char const* downDebugLabel) const
    {
        TORCH_CHECK(input.is_cuda(), "SwiGLU MLP input must be a CUDA tensor");
        TORCH_CHECK(gateUpWeight.is_cuda(), "SwiGLU MLP gate_up weight must be a CUDA tensor");
        TORCH_CHECK(downWeight.is_cuda(), "SwiGLU MLP down weight must be a CUDA tensor");
        TORCH_CHECK(gateUpScratch.is_cuda(), "SwiGLU MLP gate_up scratch must be a CUDA tensor");
        TORCH_CHECK(intermediateScratch.is_cuda(), "SwiGLU MLP intermediate scratch must be a CUDA tensor");
        TORCH_CHECK(outputScratch.is_cuda(), "SwiGLU MLP output scratch must be a CUDA tensor");
        TORCH_CHECK(inputTokens > 0, "input_tokens must be positive");
        TORCH_CHECK(sfVecSize > 0, "NVFP4 scaling vector size must be positive");
        TORCH_CHECK(input.dim() == 2, "SwiGLU MLP input must be 2D");
        TORCH_CHECK(gateUpWeight.dim() == 2, "SwiGLU MLP gate_up weight must be 2D");
        TORCH_CHECK(downWeight.dim() == 2, "SwiGLU MLP down weight must be 2D");
        TORCH_CHECK(gateUpScratch.dim() == 2, "SwiGLU MLP gate_up scratch must be 2D");
        TORCH_CHECK(intermediateScratch.dim() == 2, "SwiGLU MLP intermediate scratch must be 2D");
        TORCH_CHECK(outputScratch.dim() == 2, "SwiGLU MLP output scratch must be 2D");
        TORCH_CHECK(input.size(0) >= inputTokens, "SwiGLU MLP input batch is smaller than input");
        TORCH_CHECK(gateUpScratch.size(0) >= inputTokens, "SwiGLU MLP gate_up scratch batch is smaller than input");
        TORCH_CHECK(
            intermediateScratch.size(0) >= inputTokens, "SwiGLU MLP intermediate scratch batch is smaller than input");
        TORCH_CHECK(outputScratch.size(0) >= inputTokens, "SwiGLU MLP output scratch batch is smaller than input");
        TORCH_CHECK(gateUpWeight.size(0) % 2 == 0, "SwiGLU MLP gate_up output dim must be even");
        int64_t const gateUpWidth = gateUpWeight.size(0);
        int64_t const intermediateSize = gateUpWidth / 2;
        int64_t const outputWidth = downWeight.size(0);
        TORCH_CHECK(gateUpScratch.size(1) >= gateUpWidth, "SwiGLU MLP gate_up scratch is too narrow");
        TORCH_CHECK(intermediateScratch.size(1) >= intermediateSize, "SwiGLU MLP intermediate scratch is too narrow");
        TORCH_CHECK(outputScratch.size(1) >= outputWidth, "SwiGLU MLP output scratch is too narrow");
        TORCH_CHECK(downWeight.size(1) == intermediateSize || downWeight.size(1) * 2 == intermediateSize
                || downWeight.size(1) == intermediateSize * 2,
            "SwiGLU MLP down weight input dim is incompatible");

        at::Tensor inputPrefix = input.narrow(0, 0, inputTokens);
        at::Tensor outputPrefix = outputScratch.narrow(0, 0, inputTokens).narrow(1, 0, outputWidth);
        if (residentSharedExpertFp4OutSwiGluEnabled() && inputTokens >= residentSharedExpertFp4OutSwiGluMinBatch()
            && gateUpWeightScale.has_value() && gateUpInputScale.has_value() && gateUpAlpha.has_value()
            && downWeightScale.has_value() && downInputScale.has_value() && downAlpha.has_value())
        {
            at::Tensor inputPrefixContiguous = inputPrefix.contiguous();
            auto const gateUpFp4 = callTrtllmFp4Quantize(inputPrefixContiguous, gateUpInputScale.value(), sfVecSize);
            std::optional<std::tuple<at::Tensor, at::Tensor>> fusedIntermediate
                = tryCallCuteDslNvfp4DenseGemmSwiGluFp4Out(std::get<0>(gateUpFp4), gateUpWeight, std::get<1>(gateUpFp4),
                    gateUpWeightScale.value(), gateUpAlpha.value(), downInputScale.value());
            if (fusedIntermediate.has_value())
            {
                at::Tensor downOutput = callTrtllmNvfp4Gemm(std::get<0>(fusedIntermediate.value()), downWeight,
                    std::get<1>(fusedIntermediate.value()), downWeightScale.value(), downAlpha.value(),
                    outputScratch.scalar_type(), 0, allowedBackends);
                TORCH_CHECK(downOutput.dim() == 2, "fused SwiGLU down output must be 2D");
                TORCH_CHECK(downOutput.size(0) >= inputTokens, "fused SwiGLU down output batch is smaller than input");
                TORCH_CHECK(downOutput.size(1) >= outputWidth, "fused SwiGLU down output dim is smaller than expected");
                outputPrefix.copy_(
                    downOutput.narrow(0, 0, inputTokens).narrow(1, 0, outputWidth).to(outputPrefix.scalar_type()));
                return outputScratch;
            }
        }

        at::Tensor gateUpPrefix = gateUpScratch.narrow(0, 0, inputTokens).narrow(1, 0, gateUpWidth);
        if (!tryRunLinearMaybeNvfp4ToOutput(inputPrefix, gateUpWeight, gateUpWeightScale, gateUpInputScale, gateUpAlpha,
                gateUpPrefix, inputTokens, sfVecSize, gateUpDebugLabel))
        {
            at::Tensor gateUp = runLinearMaybeNvfp4ToFloat(inputPrefix, gateUpWeight, gateUpWeightScale,
                gateUpInputScale, gateUpAlpha, inputTokens, sfVecSize, allowedBackends, gateUpDebugLabel);
            TORCH_CHECK(gateUp.dim() == 2, "SwiGLU gate_up output must be 2D");
            TORCH_CHECK(gateUp.size(0) >= inputTokens, "SwiGLU gate_up output batch is smaller than input");
            TORCH_CHECK(gateUp.size(1) >= gateUpWidth, "SwiGLU gate_up output dim is smaller than expected");
            gateUpPrefix.copy_(gateUp.narrow(0, 0, inputTokens).narrow(1, 0, gateUpWidth));
        }

        at::Tensor intermediatePrefix = intermediateScratch.narrow(0, 0, inputTokens).narrow(1, 0, intermediateSize);
        at::Tensor gate = gateUpPrefix.narrow(1, 0, intermediateSize);
        at::Tensor up = gateUpPrefix.narrow(1, intermediateSize, intermediateSize);
        if (!tryRunResidentSwiGluFloatToOutput(gate, up, intermediatePrefix, inputTokens))
        {
            at::Tensor activated = (gate * at::sigmoid(gate)) * up;
            intermediatePrefix.copy_(activated.to(intermediatePrefix.scalar_type()));
        }

        if (!tryRunLinearMaybeNvfp4ToOutput(intermediatePrefix, downWeight, downWeightScale, downInputScale, downAlpha,
                outputPrefix, inputTokens, sfVecSize, downDebugLabel))
        {
            at::Tensor downOutput = runLinearMaybeNvfp4ToFloat(intermediatePrefix, downWeight, downWeightScale,
                downInputScale, downAlpha, inputTokens, sfVecSize, allowedBackends, downDebugLabel);
            TORCH_CHECK(downOutput.dim() == 2, "SwiGLU down output must be 2D");
            TORCH_CHECK(downOutput.size(0) >= inputTokens, "SwiGLU down output batch is smaller than input");
            TORCH_CHECK(downOutput.size(1) >= outputWidth, "SwiGLU down output dim is smaller than expected");
            outputPrefix.copy_(
                downOutput.narrow(0, 0, inputTokens).narrow(1, 0, outputWidth).to(outputPrefix.scalar_type()));
        }
        return outputScratch;
    }

    static std::optional<at::Tensor> selectExpertTensor(std::optional<at::Tensor> const& tensor, int64_t expertIdx)
    {
        if (!tensor.has_value())
        {
            return std::nullopt;
        }
        at::Tensor const& value = tensor.value();
        if (value.dim() > 0 && value.size(0) > expertIdx)
        {
            return value.select(0, expertIdx);
        }
        return value;
    }

    at::Tensor runInputEmbedding(
        at::Tensor const& inputIds, at::Tensor const& hiddenStatesScratch, int64_t inputTokens) const
    {
        TORCH_CHECK(inputIds.is_cuda(), "input_ids must be a CUDA tensor");
        TORCH_CHECK(hiddenStatesScratch.is_cuda(), "hidden_states_scratch must be a CUDA tensor");
        TORCH_CHECK(inputTokens > 0, "input_tokens must be positive");
        TORCH_CHECK(
            mNbTensors > static_cast<int64_t>(DeepseekResidentTensorSlot::kEmbedding), "missing embedding tensor");
        at::Tensor const& embedding = mResidentTensors.at(static_cast<size_t>(DeepseekResidentTensorSlot::kEmbedding));
        TORCH_CHECK(embedding.dim() == 2, "embedding tensor must be 2D");
        TORCH_CHECK(hiddenStatesScratch.dim() == 2, "hidden_states_scratch must be 2D");
        TORCH_CHECK(hiddenStatesScratch.size(0) >= inputTokens, "hidden_states_scratch batch is smaller than input");
        TORCH_CHECK(
            hiddenStatesScratch.size(1) == embedding.size(1), "hidden_states_scratch hidden dim must match embedding");
        TORCH_CHECK(hiddenStatesScratch.scalar_type() == embedding.scalar_type(),
            "hidden_states_scratch dtype must match embedding dtype");

        at::Tensor flatInputIds = inputIds.reshape({inputIds.numel()});
        TORCH_CHECK(flatInputIds.numel() >= inputTokens, "input_ids is shorter than input_tokens");
        if (flatInputIds.scalar_type() != at::ScalarType::Long)
        {
            flatInputIds = flatInputIds.toType(at::ScalarType::Long);
        }
        at::Tensor tokenIds = flatInputIds.narrow(0, 0, inputTokens);
        at::Tensor embedded = embedding.index_select(0, tokenIds);
        at::Tensor hiddenPrefix = hiddenStatesScratch.narrow(0, 0, inputTokens);
        hiddenPrefix.copy_(embedded);
        return hiddenStatesScratch;
    }

    at::Tensor runLayerInputRmsNorm(int64_t layerIdx, at::Tensor const& hiddenStatesScratch,
        at::Tensor const& normHiddenStatesScratch, int64_t inputTokens, double eps, bool useGemma) const
    {
        TORCH_CHECK(layerIdx >= 0 && layerIdx < mNbLayers, "layer_idx is out of range");
        TORCH_CHECK(hiddenStatesScratch.is_cuda(), "hidden_states_scratch must be a CUDA tensor");
        TORCH_CHECK(normHiddenStatesScratch.is_cuda(), "norm_hidden_states_scratch must be a CUDA tensor");
        TORCH_CHECK(inputTokens > 0, "input_tokens must be positive");
        TORCH_CHECK(hiddenStatesScratch.dim() == 2, "hidden_states_scratch must be 2D");
        TORCH_CHECK(normHiddenStatesScratch.dim() == 2, "norm_hidden_states_scratch must be 2D");
        TORCH_CHECK(hiddenStatesScratch.size(0) >= inputTokens, "hidden_states_scratch batch is smaller than input");
        TORCH_CHECK(
            normHiddenStatesScratch.size(0) >= inputTokens, "norm_hidden_states_scratch batch is smaller than input");
        TORCH_CHECK(hiddenStatesScratch.size(1) == normHiddenStatesScratch.size(1),
            "norm_hidden_states_scratch hidden dim must match hidden_states_scratch");
        TORCH_CHECK(hiddenStatesScratch.scalar_type() == normHiddenStatesScratch.scalar_type(),
            "norm_hidden_states_scratch dtype must match hidden_states_scratch dtype");

        at::Tensor const& weight
            = getLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kInputLayernorm, "layer input RMSNorm weight");
        TORCH_CHECK(weight.dim() == 1, "layer input RMSNorm weight must be 1D");
        TORCH_CHECK(weight.size(0) == hiddenStatesScratch.size(1), "RMSNorm weight size must match hidden dim");
        TORCH_CHECK(weight.scalar_type() == hiddenStatesScratch.scalar_type(),
            "RMSNorm weight dtype must match hidden_states_scratch dtype");

        if (tryRunResidentRmsNorm(hiddenStatesScratch, normHiddenStatesScratch, weight, inputTokens, eps, useGemma))
        {
            return normHiddenStatesScratch;
        }

        at::Tensor inputPrefix = hiddenStatesScratch.narrow(0, 0, inputTokens);
        at::Tensor normPrefix = normHiddenStatesScratch.narrow(0, 0, inputTokens);
        at::Tensor inputFloat = inputPrefix.to(at::ScalarType::Float);
        at::Tensor variance = inputFloat.pow(2).mean(-1, true);
        at::Tensor normalized = inputFloat * at::rsqrt(variance + eps);
        at::Tensor effectiveWeight = useGemma ? weight + 1 : weight;
        at::Tensor output = normalized.to(hiddenStatesScratch.scalar_type()) * effectiveWeight;
        normPrefix.copy_(output);
        return normHiddenStatesScratch;
    }

    at::Tensor runLayerInputGatedNorm(int64_t layerIdx, at::Tensor const& normHiddenStatesScratch,
        at::Tensor const& gatedHiddenStatesScratch, int64_t inputTokens) const
    {
        TORCH_CHECK(layerIdx >= 0 && layerIdx < mNbLayers, "layer_idx is out of range");
        TORCH_CHECK(normHiddenStatesScratch.is_cuda(), "norm_hidden_states_scratch must be a CUDA tensor");
        TORCH_CHECK(gatedHiddenStatesScratch.is_cuda(), "gated_hidden_states_scratch must be a CUDA tensor");
        TORCH_CHECK(inputTokens > 0, "input_tokens must be positive");
        TORCH_CHECK(normHiddenStatesScratch.dim() == 2, "norm_hidden_states_scratch must be 2D");
        TORCH_CHECK(gatedHiddenStatesScratch.dim() == 2, "gated_hidden_states_scratch must be 2D");
        TORCH_CHECK(
            normHiddenStatesScratch.size(0) >= inputTokens, "norm_hidden_states_scratch batch is smaller than input");
        TORCH_CHECK(
            gatedHiddenStatesScratch.size(0) >= inputTokens, "gated_hidden_states_scratch batch is smaller than input");
        TORCH_CHECK(gatedHiddenStatesScratch.size(1) == normHiddenStatesScratch.size(1),
            "gated_hidden_states_scratch hidden dim must match norm_hidden_states_scratch");
        TORCH_CHECK(gatedHiddenStatesScratch.scalar_type() == normHiddenStatesScratch.scalar_type(),
            "gated_hidden_states_scratch dtype must match norm_hidden_states_scratch dtype");

        at::Tensor const& downWeight = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kInputGatedNormDown, "input gated norm down weight");
        at::Tensor const& upWeight = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kInputGatedNormUp, "input gated norm up weight");
        TORCH_CHECK(downWeight.dim() == 2, "input gated norm down weight must be 2D");
        TORCH_CHECK(upWeight.dim() == 2, "input gated norm up weight must be 2D");
        TORCH_CHECK(downWeight.size(1) == normHiddenStatesScratch.size(1),
            "input gated norm down hidden dim must match norm_hidden_states_scratch");
        TORCH_CHECK(upWeight.size(0) == normHiddenStatesScratch.size(1),
            "input gated norm up hidden dim must match norm_hidden_states_scratch");
        TORCH_CHECK(upWeight.size(1) == downWeight.size(0), "input gated norm rank dimensions must match");

        if (tryRunResidentLowRankGate(
                normHiddenStatesScratch, gatedHiddenStatesScratch, downWeight, upWeight, inputTokens))
        {
            return gatedHiddenStatesScratch;
        }

        at::Tensor inputPrefix = normHiddenStatesScratch.narrow(0, 0, inputTokens);
        at::Tensor gatedPrefix = gatedHiddenStatesScratch.narrow(0, 0, inputTokens);
        at::Tensor inputFloat = inputPrefix.to(at::ScalarType::Float);
        at::Tensor downFloat = downWeight.to(at::ScalarType::Float);
        at::Tensor gate = at::matmul(inputFloat, downFloat.t());
        gate = gate * at::sigmoid(gate);
        gate = at::matmul(gate.to(upWeight.scalar_type()), upWeight.t());
        gate = at::sigmoid(gate).to(inputPrefix.scalar_type());
        at::Tensor output = inputPrefix * gate;
        gatedPrefix.copy_(output);
        return gatedHiddenStatesScratch;
    }

    DeepseekResidentDsaAttentionProjectionResult runLayerDsaAttentionProjectionImpl(int64_t layerIdx,
        at::Tensor const& hiddenStatesScratch, at::Tensor const& qScratch, at::Tensor const& kvAScratch,
        at::Tensor const& qLoraScratch, at::Tensor const& compressedKvScratch, at::Tensor const& kPeScratch,
        at::Tensor const& latentCacheScratch, std::optional<at::Tensor> const& indexerWqBScratch,
        std::optional<at::Tensor> const& indexerKScratch,
        std::optional<at::Tensor> const& indexerWeightsScratch, int64_t inputTokens, int64_t qLoraRank,
        int64_t kvLoraRank, int64_t ropeDim, double eps, int64_t sfVecSize, std::string const& allowedBackends) const
    {
        TORCH_CHECK(layerIdx >= 0 && layerIdx < mNbLayers, "layer_idx is out of range");
        TORCH_CHECK(hiddenStatesScratch.is_cuda(), "hidden_states_scratch must be a CUDA tensor");
        TORCH_CHECK(qScratch.is_cuda(), "q_scratch must be a CUDA tensor");
        TORCH_CHECK(kvAScratch.is_cuda(), "kv_a_scratch must be a CUDA tensor");
        TORCH_CHECK(qLoraScratch.is_cuda(), "q_lora_scratch must be a CUDA tensor");
        TORCH_CHECK(compressedKvScratch.is_cuda(), "compressed_kv_scratch must be a CUDA tensor");
        TORCH_CHECK(kPeScratch.is_cuda(), "k_pe_scratch must be a CUDA tensor");
        TORCH_CHECK(latentCacheScratch.is_cuda(), "latent_cache_scratch must be a CUDA tensor");
        TORCH_CHECK(inputTokens > 0, "input_tokens must be positive");
        TORCH_CHECK(qLoraRank > 0 && kvLoraRank > 0 && ropeDim > 0, "MLA projection dimensions must be positive");
        TORCH_CHECK(hiddenStatesScratch.dim() == 2, "hidden_states_scratch must be 2D");
        TORCH_CHECK(qScratch.dim() == 2, "q_scratch must be 2D");
        TORCH_CHECK(kvAScratch.dim() == 2, "kv_a_scratch must be 2D");
        TORCH_CHECK(qLoraScratch.dim() == 2, "q_lora_scratch must be 2D");
        TORCH_CHECK(compressedKvScratch.dim() == 2, "compressed_kv_scratch must be 2D");
        TORCH_CHECK(kPeScratch.dim() == 2, "k_pe_scratch must be 2D");
        TORCH_CHECK(latentCacheScratch.dim() == 2, "latent_cache_scratch must be 2D");
        TORCH_CHECK(
            hiddenStatesScratch.size(0) >= inputTokens, "hidden_states_scratch batch is smaller than input_tokens");
        TORCH_CHECK(qScratch.size(0) >= inputTokens, "q_scratch batch is smaller than input_tokens");
        TORCH_CHECK(kvAScratch.size(0) >= inputTokens, "kv_a_scratch batch is smaller than input_tokens");
        TORCH_CHECK(qLoraScratch.size(0) >= inputTokens, "q_lora_scratch batch is smaller than input_tokens");
        TORCH_CHECK(
            compressedKvScratch.size(0) >= inputTokens, "compressed_kv_scratch batch is smaller than input_tokens");
        TORCH_CHECK(kPeScratch.size(0) >= inputTokens, "k_pe_scratch batch is smaller than input_tokens");
        TORCH_CHECK(
            latentCacheScratch.size(0) >= inputTokens, "latent_cache_scratch batch is smaller than input_tokens");
        TORCH_CHECK(qLoraScratch.size(1) == qLoraRank, "q_lora_scratch width must match q_lora_rank");
        TORCH_CHECK(kvAScratch.size(1) >= qLoraRank + kvLoraRank + ropeDim,
            "kv_a_scratch width must cover q_lora + kv_lora + rope dims");
        TORCH_CHECK(compressedKvScratch.size(1) == kvLoraRank, "compressed_kv_scratch width must match kv_lora_rank");
        TORCH_CHECK(kPeScratch.size(1) == ropeDim, "k_pe_scratch width must match rope dim");
        TORCH_CHECK(latentCacheScratch.size(1) == kvLoraRank + ropeDim,
            "latent_cache_scratch width must match kv_lora_rank + rope dim");

        at::Tensor const& kvAWeight = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kAttentionKvAProjWeight, "attention kv_a_proj weight");
        at::Tensor const& kvAWeightScale = getLayerTensor(layerIdx,
            DeepseekResidentLayerTensorSite::kAttentionKvAProjWeightScale, "attention kv_a_proj weight_scale");
        at::Tensor const& kvAInputScale = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kAttentionKvAProjInputScale, "attention kv_a_proj input_scale");
        at::Tensor const& kvAAlpha = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kAttentionKvAProjAlpha, "attention kv_a_proj alpha");
        at::Tensor const& qALayernormWeight = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kAttentionQALayernormWeight, "attention q_a_layernorm weight");
        at::Tensor const& kvALayernormWeight = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kAttentionKvALayernormWeight, "attention kv_a_layernorm weight");
        at::Tensor const& qBWeight = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kAttentionQBProjWeight, "attention q_b_proj weight");
        at::Tensor const& qBWeightScale = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kAttentionQBProjWeightScale, "attention q_b_proj weight_scale");
        at::Tensor const& qBInputScale = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kAttentionQBProjInputScale, "attention q_b_proj input_scale");
        at::Tensor const& qBAlpha = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kAttentionQBProjAlpha, "attention q_b_proj alpha");

        int64_t const kvAWidth = qLoraRank + kvLoraRank + ropeDim;
        at::Tensor kvAPrefix = kvAScratch.narrow(0, 0, inputTokens).narrow(1, 0, kvAWidth);
        bool hasPrecomputedIndexerKWeights = false;
        bool usedFusedKvAWkWp = false;
        at::Tensor fusedKvAWkWpOutput;
        DeepseekResidentFusedKvAWkWpCache const* fusedKvAWkWp = getFusedKvAWkWpCache(layerIdx, kvAWidth);
        if (fusedKvAWkWp != nullptr && indexerKScratch.has_value() && indexerWeightsScratch.has_value())
        {
            TORCH_CHECK(indexerKScratch->is_cuda(), "fused KVA/WK/WP indexer_k scratch must be CUDA");
            TORCH_CHECK(indexerWeightsScratch->is_cuda(),
                "fused KVA/WK/WP indexer_weights scratch must be CUDA");
            TORCH_CHECK(indexerKScratch->dim() == 2, "fused KVA/WK/WP indexer_k scratch must be 2D");
            TORCH_CHECK(indexerWeightsScratch->dim() == 2,
                "fused KVA/WK/WP indexer_weights scratch must be 2D");
            TORCH_CHECK(indexerKScratch->size(0) >= inputTokens,
                "fused KVA/WK/WP indexer_k scratch batch is smaller than input");
            TORCH_CHECK(indexerWeightsScratch->size(0) >= inputTokens,
                "fused KVA/WK/WP indexer_weights scratch batch is smaller than input");
            TORCH_CHECK(indexerKScratch->size(1) >= fusedKvAWkWp->wkOut,
                "fused KVA/WK/WP indexer_k scratch width is too small");
            TORCH_CHECK(indexerWeightsScratch->size(1) >= fusedKvAWkWp->wpOut,
                "fused KVA/WK/WP indexer_weights scratch width is too small");
            fusedKvAWkWpOutput = runNvfp4LinearTyped(hiddenStatesScratch, fusedKvAWkWp->weight,
                fusedKvAWkWp->weightScale, fusedKvAWkWp->inputScale, fusedKvAWkWp->alpha, inputTokens, sfVecSize,
                allowedBackends, fusedKvAWkWp->outDtype, "attention_kv_a_wk_wp_fused");
            TORCH_CHECK(fusedKvAWkWpOutput.dim() == 2, "fused KVA/WK/WP output must be 2D");
            TORCH_CHECK(fusedKvAWkWpOutput.size(0) >= inputTokens,
                "fused KVA/WK/WP output batch is smaller than input");
            TORCH_CHECK(fusedKvAWkWpOutput.size(1)
                    >= fusedKvAWkWp->kvOutPadded + fusedKvAWkWp->wkOut + fusedKvAWkWp->wpOut,
                "fused KVA/WK/WP output width is smaller than expected");
            kvAPrefix.copy_(fusedKvAWkWpOutput.narrow(0, 0, inputTokens).narrow(1, 0, kvAWidth));
            usedFusedKvAWkWp = true;
        }
        else if (!tryRunNvfp4LinearTypedToOutput(hiddenStatesScratch, kvAWeight, kvAWeightScale, kvAInputScale,
                     kvAAlpha, kvAPrefix, inputTokens, sfVecSize, "attention_kv_a_proj_out"))
        {
            at::Tensor kva = runNvfp4Linear(hiddenStatesScratch, kvAWeight, kvAWeightScale, kvAInputScale, kvAAlpha,
                inputTokens, sfVecSize, allowedBackends);
            TORCH_CHECK(kva.dim() == 2, "kv_a_proj output must be 2D");
            TORCH_CHECK(kva.size(0) >= inputTokens, "kv_a_proj output batch is smaller than input_tokens");
            TORCH_CHECK(kva.size(1) >= kvAWidth, "kv_a_proj output width is smaller than q_lora + kv_lora + rope dims");
            kvAPrefix.copy_(kva.narrow(0, 0, inputTokens).narrow(1, 0, kvAWidth));
        }

        if (!tryRunResidentDsaKvASplitNormPack(kvAPrefix, qLoraScratch, compressedKvScratch, kPeScratch,
                latentCacheScratch, qALayernormWeight, kvALayernormWeight, inputTokens, qLoraRank, kvLoraRank, ropeDim,
                eps))
        {
            at::Tensor qRaw = kvAPrefix.narrow(1, 0, qLoraRank);
            at::Tensor compressedRaw = kvAPrefix.narrow(1, qLoraRank, kvLoraRank);
            at::Tensor kPeRaw = kvAPrefix.narrow(1, qLoraRank + kvLoraRank, ropeDim);
            at::Tensor qLoraPrefix = qLoraScratch.narrow(0, 0, inputTokens);
            at::Tensor compressedPrefix = compressedKvScratch.narrow(0, 0, inputTokens);
            at::Tensor kPePrefix = kPeScratch.narrow(0, 0, inputTokens);
            at::Tensor latentPrefix = latentCacheScratch.narrow(0, 0, inputTokens);

            if (!tryRunResidentRmsNorm(qRaw, qLoraScratch, qALayernormWeight, inputTokens, eps, false))
            {
                qLoraPrefix.copy_(rmsNorm2d(qRaw, qALayernormWeight, inputTokens, eps));
            }
            if (!tryRunResidentRmsNorm(compressedRaw, compressedKvScratch, kvALayernormWeight, inputTokens, eps, false))
            {
                compressedPrefix.copy_(rmsNorm2d(compressedRaw, kvALayernormWeight, inputTokens, eps));
            }
            kPePrefix.copy_(kPeRaw.narrow(0, 0, inputTokens));
            latentPrefix.narrow(1, 0, kvLoraRank).copy_(compressedPrefix);
            latentPrefix.narrow(1, kvLoraRank, ropeDim).copy_(kPePrefix);
        }

        if (usedFusedKvAWkWp)
        {
            at::Tensor fusedPrefix = fusedKvAWkWpOutput.narrow(0, 0, inputTokens);
            at::Tensor wkOutput = fusedPrefix.narrow(1, fusedKvAWkWp->kvOutPadded, fusedKvAWkWp->wkOut);
            at::Tensor wpOutput
                = fusedPrefix.narrow(1, fusedKvAWkWp->kvOutPadded + fusedKvAWkWp->wkOut, fusedKvAWkWp->wpOut);
            at::Tensor wkForCopy = wkOutput;
            if (fusedKvAWkWp->wkOutScale != 1.0)
            {
                wkForCopy = wkOutput.to(at::ScalarType::Float).mul(fusedKvAWkWp->wkOutScale);
            }
            indexerKScratch->narrow(0, 0, inputTokens)
                .narrow(1, 0, fusedKvAWkWp->wkOut)
                .copy_(wkForCopy.to(indexerKScratch->scalar_type()));

            at::Tensor wpForCopy = wpOutput.to(at::ScalarType::Float);
            if (fusedKvAWkWp->wpOutScale != 1.0)
            {
                wpForCopy = wpForCopy.mul(fusedKvAWkWp->wpOutScale);
            }
            indexerWeightsScratch->narrow(0, 0, inputTokens)
                .narrow(1, 0, fusedKvAWkWp->wpOut)
                .copy_(wpForCopy);
            hasPrecomputedIndexerKWeights = true;
        }

        at::Tensor qOutput = qScratch.narrow(0, 0, inputTokens);
        bool hasPrecomputedWqB = false;
        double precomputedWqBScale = 1.0;
        DeepseekResidentFusedQbWqBCache const* fusedQbWqB = getFusedQbWqBCache(layerIdx);
        if (fusedQbWqB != nullptr && indexerWqBScratch.has_value())
        {
            at::Tensor indexerWqBPrefix = indexerWqBScratch->narrow(0, 0, inputTokens).narrow(1, 0, fusedQbWqB->wqOut);
            TORCH_CHECK(indexerWqBPrefix.is_cuda(), "fused q_b/wq_b scratch must be CUDA");
            TORCH_CHECK(indexerWqBPrefix.dim() == 2, "fused q_b/wq_b scratch must be 2D");
            TORCH_CHECK(indexerWqBPrefix.size(1) >= fusedQbWqB->wqOut, "fused q_b/wq_b scratch is too narrow");
            at::Tensor inputPrefix = qLoraScratch.narrow(0, 0, inputTokens).contiguous();
            auto fp4 = callTrtllmFp4Quantize(inputPrefix, fusedQbWqB->inputScale, sfVecSize);
            at::Tensor fusedOutput = callTrtllmNvfp4Gemm(std::get<0>(fp4), fusedQbWqB->weight,
                std::get<1>(fp4), fusedQbWqB->weightScale, fusedQbWqB->alpha, qOutput.scalar_type(), 0,
                allowedBackends);
            TORCH_CHECK(fusedOutput.dim() == 2, "fused q_b/wq_b output must be 2D");
            TORCH_CHECK(fusedOutput.size(0) >= inputTokens, "fused q_b/wq_b output batch is smaller than input");
            TORCH_CHECK(fusedOutput.size(1) >= fusedQbWqB->qbOut + fusedQbWqB->wqOut,
                "fused q_b/wq_b output width is smaller than expected");
            qOutput.copy_(fusedOutput.narrow(0, 0, inputTokens).narrow(1, 0, qScratch.size(1)));
            indexerWqBPrefix.copy_(
                fusedOutput.narrow(0, 0, inputTokens).narrow(1, fusedQbWqB->qbOut, fusedQbWqB->wqOut));
            hasPrecomputedWqB = true;
            precomputedWqBScale = fusedQbWqB->wqOutScale;
        }
        else if (!tryRunNvfp4LinearTypedToOutput(qLoraScratch, qBWeight, qBWeightScale, qBInputScale, qBAlpha, qOutput,
                     inputTokens, sfVecSize, "attention_q_b_proj_out"))
        {
            at::Tensor qProjected = runNvfp4Linear(
                qLoraScratch, qBWeight, qBWeightScale, qBInputScale, qBAlpha, inputTokens, sfVecSize, allowedBackends);
            TORCH_CHECK(qProjected.dim() == 2, "q_b_proj output must be 2D");
            TORCH_CHECK(qProjected.size(0) >= inputTokens, "q_b_proj output batch is smaller than input_tokens");
            TORCH_CHECK(qProjected.size(1) >= qScratch.size(1), "q_b_proj output width is smaller than q_scratch");
            qOutput.copy_(qProjected.narrow(0, 0, inputTokens).narrow(1, 0, qScratch.size(1)));
        }

        return DeepseekResidentDsaAttentionProjectionResult{qScratch, compressedKvScratch, kPeScratch,
            latentCacheScratch, qLoraScratch, hasPrecomputedWqB, precomputedWqBScale,
            hasPrecomputedIndexerKWeights};
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> runLayerDsaAttentionProjection(
        int64_t layerIdx, at::Tensor const& hiddenStatesScratch, at::Tensor const& qScratch,
        at::Tensor const& kvAScratch, at::Tensor const& qLoraScratch, at::Tensor const& compressedKvScratch,
        at::Tensor const& kPeScratch, at::Tensor const& latentCacheScratch, int64_t inputTokens, int64_t qLoraRank,
        int64_t kvLoraRank, int64_t ropeDim, double eps, int64_t sfVecSize, std::string const& allowedBackends) const
    {
        DeepseekResidentDsaAttentionProjectionResult result = runLayerDsaAttentionProjectionImpl(layerIdx,
            hiddenStatesScratch, qScratch, kvAScratch, qLoraScratch, compressedKvScratch, kPeScratch,
            latentCacheScratch, std::nullopt, std::nullopt, std::nullopt, inputTokens, qLoraRank, kvLoraRank, ropeDim,
            eps, sfVecSize, allowedBackends);
        return {result.qScratch, result.compressedKvScratch, result.kPeScratch, result.latentCacheScratch,
            result.qLoraScratch};
    }

    at::Tensor runLayerAttentionOutputTailWithGateLogits(int64_t layerIdx, at::Tensor const& attentionCoreOutputScratch,
        at::Tensor const& attentionInputScratch, at::Tensor const& attentionGateScratch,
        at::Tensor const& attentionGateLogitsScratch, at::Tensor const& attentionHiddenStatesScratch,
        int64_t inputTokens) const
    {
        TORCH_CHECK(layerIdx >= 0 && layerIdx < mNbLayers, "layer_idx is out of range");
        TORCH_CHECK(attentionCoreOutputScratch.is_cuda(), "attention_core_output_scratch must be a CUDA tensor");
        TORCH_CHECK(attentionInputScratch.is_cuda(), "attention_input_scratch must be a CUDA tensor");
        TORCH_CHECK(attentionGateScratch.is_cuda(), "attention_gate_scratch must be a CUDA tensor");
        TORCH_CHECK(attentionGateLogitsScratch.is_cuda(), "attention_gate_logits_scratch must be a CUDA tensor");
        TORCH_CHECK(attentionHiddenStatesScratch.is_cuda(), "attention_hidden_states_scratch must be a CUDA tensor");
        TORCH_CHECK(inputTokens > 0, "input_tokens must be positive");
        TORCH_CHECK(attentionCoreOutputScratch.dim() == 2, "attention_core_output_scratch must be 2D");
        TORCH_CHECK(attentionInputScratch.dim() == 2, "attention_input_scratch must be 2D");
        TORCH_CHECK(attentionGateScratch.dim() == 2, "attention_gate_scratch must be 2D");
        TORCH_CHECK(attentionGateLogitsScratch.dim() == 2, "attention_gate_logits_scratch must be 2D");
        TORCH_CHECK(attentionHiddenStatesScratch.dim() == 2, "attention_hidden_states_scratch must be 2D");
        TORCH_CHECK(attentionCoreOutputScratch.size(0) >= inputTokens,
            "attention_core_output_scratch batch is smaller than input");
        TORCH_CHECK(
            attentionInputScratch.size(0) >= inputTokens, "attention_input_scratch batch is smaller than input");
        TORCH_CHECK(attentionGateScratch.size(0) >= inputTokens, "attention_gate_scratch batch is smaller than input");
        TORCH_CHECK(attentionGateLogitsScratch.size(0) >= inputTokens,
            "attention_gate_logits_scratch batch is smaller than input");
        TORCH_CHECK(attentionHiddenStatesScratch.size(0) >= inputTokens,
            "attention_hidden_states_scratch batch is smaller than input");
        TORCH_CHECK(attentionGateScratch.size(1) == attentionCoreOutputScratch.size(1),
            "attention_gate_scratch dim must match attention_core_output_scratch");
        TORCH_CHECK(attentionGateLogitsScratch.size(1) == attentionCoreOutputScratch.size(1),
            "attention_gate_logits_scratch dim must match attention_core_output_scratch");
        TORCH_CHECK(attentionCoreOutputScratch.scalar_type() == attentionInputScratch.scalar_type(),
            "attention core output and attention input dtypes must match");
        TORCH_CHECK(attentionGateScratch.scalar_type() == attentionInputScratch.scalar_type(),
            "attention_gate_scratch dtype must match attention input dtype");
        TORCH_CHECK(attentionGateLogitsScratch.scalar_type() == at::ScalarType::Float,
            "attention_gate_logits_scratch must be float32");
        TORCH_CHECK(attentionHiddenStatesScratch.scalar_type() == attentionInputScratch.scalar_type(),
            "attention_hidden_states_scratch dtype must match attention input dtype");

        at::Tensor const& gateWeight = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kAttentionGateProjWeight, "attention gate_proj weight");
        at::Tensor const& oProjWeight = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kAttentionOProjWeight, "attention o_proj weight");
        TORCH_CHECK(gateWeight.dim() == 2, "attention gate_proj weight must be 2D");
        TORCH_CHECK(oProjWeight.dim() == 2, "attention o_proj weight must be 2D");

        at::Tensor corePrefix = attentionCoreOutputScratch.narrow(0, 0, inputTokens);
        at::Tensor inputPrefix = attentionInputScratch.narrow(0, 0, inputTokens);
        at::Tensor gatePrefix = attentionGateScratch.narrow(0, 0, inputTokens);
        at::Tensor outputPrefix = attentionHiddenStatesScratch.narrow(0, 0, inputTokens);
        at::Tensor gateLogitsPrefix = attentionGateLogitsScratch.narrow(0, 0, inputTokens);
        std::optional<at::Tensor> const gateWeightScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionGateProjWeightScale);
        std::optional<at::Tensor> const gateInputScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionGateProjInputScale);
        std::optional<at::Tensor> const gateAlpha
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionGateProjAlpha);
        std::optional<at::Tensor> const oProjWeightScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionOProjWeightScale);
        std::optional<at::Tensor> const oProjInputScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionOProjInputScale);
        std::optional<at::Tensor> const oProjAlpha
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionOProjAlpha);

        if (tryRunAttentionTailFp4OutGate(attentionCoreOutputScratch, attentionInputScratch, attentionGateScratch,
                attentionHiddenStatesScratch, gateWeight, gateWeightScale, gateInputScale, gateAlpha, oProjWeight,
                oProjWeightScale, oProjInputScale, oProjAlpha, inputTokens))
        {
            return attentionHiddenStatesScratch;
        }

        if (!tryRunLinearMaybeNvfp4ToOutput(inputPrefix, gateWeight, gateWeightScale, gateInputScale, gateAlpha,
                gateLogitsPrefix, inputTokens, /*sfVecSize=*/16, "attention_gate_proj_out"))
        {
            at::Tensor gate = runLinearMaybeNvfp4ToFloat(inputPrefix, gateWeight, gateWeightScale, gateInputScale,
                gateAlpha, inputTokens, /*sfVecSize=*/16, "cutlass,cublaslt,cuda_core", "attention_gate_proj");
            TORCH_CHECK(gate.dim() == 2, "attention gate_proj output must be 2D");
            TORCH_CHECK(gate.size(0) >= inputTokens, "attention gate_proj output batch is smaller than input");
            TORCH_CHECK(gate.size(1) >= attentionCoreOutputScratch.size(1),
                "attention gate_proj output dim must cover attention core output dim");
            gateLogitsPrefix.copy_(gate.narrow(0, 0, inputTokens).narrow(1, 0, attentionCoreOutputScratch.size(1)));
        }
        if (!tryRunResidentSigmoidMul(corePrefix, gateLogitsPrefix, attentionGateScratch, inputTokens))
        {
            at::Tensor gatedAttention = corePrefix.to(at::ScalarType::Float) * at::sigmoid(gateLogitsPrefix);
            gatePrefix.copy_(gatedAttention.to(inputPrefix.scalar_type()));
        }
        if (!tryRunLinearMaybeNvfp4ToOutput(attentionGateScratch, oProjWeight, oProjWeightScale, oProjInputScale,
                oProjAlpha, outputPrefix, inputTokens, /*sfVecSize=*/16, "attention_o_proj_out"))
        {
            at::Tensor output
                = runLinearMaybeNvfp4ToFloat(attentionGateScratch, oProjWeight, oProjWeightScale, oProjInputScale,
                    oProjAlpha, inputTokens, /*sfVecSize=*/16, "cutlass,cublaslt,cuda_core", "attention_o_proj");
            TORCH_CHECK(output.dim() == 2, "attention o_proj output must be 2D");
            TORCH_CHECK(output.size(0) >= inputTokens, "attention o_proj output batch is smaller than input");
            TORCH_CHECK(output.size(1) >= attentionHiddenStatesScratch.size(1),
                "attention o_proj output dim must cover hidden dim");
            outputPrefix.copy_(output.narrow(0, 0, inputTokens)
                    .narrow(1, 0, attentionHiddenStatesScratch.size(1))
                    .to(inputPrefix.scalar_type()));
        }
        return attentionHiddenStatesScratch;
    }

    at::Tensor runLayerAttentionOutputTail(int64_t layerIdx, at::Tensor const& attentionCoreOutputScratch,
        at::Tensor const& attentionInputScratch, at::Tensor const& attentionGateScratch,
        at::Tensor const& attentionHiddenStatesScratch, int64_t inputTokens) const
    {
        at::Tensor attentionGateLogitsScratch = at::empty({attentionGateScratch.size(0), attentionGateScratch.size(1)},
            attentionGateScratch.options().dtype(at::ScalarType::Float));
        return runLayerAttentionOutputTailWithGateLogits(layerIdx, attentionCoreOutputScratch, attentionInputScratch,
            attentionGateScratch, attentionGateLogitsScratch, attentionHiddenStatesScratch, inputTokens);
    }

    at::Tensor runLayerPostAttentionRmsNorm(int64_t layerIdx, at::Tensor const& attentionHiddenStatesScratch,
        at::Tensor const& residualInputScratch, at::Tensor const& postAttentionNormScratch,
        at::Tensor const& postAttentionResidualScratch, int64_t inputTokens, double eps, bool useGemma) const
    {
        TORCH_CHECK(layerIdx >= 0 && layerIdx < mNbLayers, "layer_idx is out of range");
        TORCH_CHECK(attentionHiddenStatesScratch.is_cuda(), "attention_hidden_states_scratch must be a CUDA tensor");
        TORCH_CHECK(residualInputScratch.is_cuda(), "residual_input_scratch must be a CUDA tensor");
        TORCH_CHECK(postAttentionNormScratch.is_cuda(), "post_attention_norm_scratch must be a CUDA tensor");
        TORCH_CHECK(postAttentionResidualScratch.is_cuda(), "post_attention_residual_scratch must be a CUDA tensor");
        TORCH_CHECK(inputTokens > 0, "input_tokens must be positive");
        TORCH_CHECK(attentionHiddenStatesScratch.dim() == 2, "attention_hidden_states_scratch must be 2D");
        TORCH_CHECK(residualInputScratch.dim() == 2, "residual_input_scratch must be 2D");
        TORCH_CHECK(postAttentionNormScratch.dim() == 2, "post_attention_norm_scratch must be 2D");
        TORCH_CHECK(postAttentionResidualScratch.dim() == 2, "post_attention_residual_scratch must be 2D");
        TORCH_CHECK(attentionHiddenStatesScratch.size(0) >= inputTokens,
            "attention_hidden_states_scratch batch is smaller than input");
        TORCH_CHECK(residualInputScratch.size(0) >= inputTokens, "residual_input_scratch batch is smaller than input");
        TORCH_CHECK(
            postAttentionNormScratch.size(0) >= inputTokens, "post_attention_norm_scratch batch is smaller than input");
        TORCH_CHECK(postAttentionResidualScratch.size(0) >= inputTokens,
            "post_attention_residual_scratch batch is smaller than input");
        TORCH_CHECK(attentionHiddenStatesScratch.size(1) == residualInputScratch.size(1),
            "attention hidden dim must match residual input hidden dim");
        TORCH_CHECK(postAttentionNormScratch.size(1) == attentionHiddenStatesScratch.size(1),
            "post_attention_norm_scratch hidden dim must match attention hidden dim");
        TORCH_CHECK(postAttentionResidualScratch.size(1) == attentionHiddenStatesScratch.size(1),
            "post_attention_residual_scratch hidden dim must match attention hidden dim");
        TORCH_CHECK(attentionHiddenStatesScratch.scalar_type() == residualInputScratch.scalar_type(),
            "attention and residual input dtypes must match");
        TORCH_CHECK(postAttentionNormScratch.scalar_type() == attentionHiddenStatesScratch.scalar_type(),
            "post_attention_norm_scratch dtype must match attention dtype");
        TORCH_CHECK(postAttentionResidualScratch.scalar_type() == attentionHiddenStatesScratch.scalar_type(),
            "post_attention_residual_scratch dtype must match attention dtype");

        at::Tensor const& weight = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kPostAttentionLayernorm, "post-attention RMSNorm weight");
        TORCH_CHECK(weight.dim() == 1, "post-attention RMSNorm weight must be 1D");
        TORCH_CHECK(
            weight.size(0) == attentionHiddenStatesScratch.size(1), "RMSNorm weight size must match hidden dim");
        TORCH_CHECK(weight.scalar_type() == attentionHiddenStatesScratch.scalar_type(),
            "RMSNorm weight dtype must match attention_hidden_states_scratch dtype");

        if (tryRunResidentAddRmsNorm(attentionHiddenStatesScratch, residualInputScratch, postAttentionNormScratch,
                postAttentionResidualScratch, weight, inputTokens, eps, useGemma))
        {
            return postAttentionNormScratch;
        }

        at::Tensor attentionPrefix = attentionHiddenStatesScratch.narrow(0, 0, inputTokens);
        at::Tensor residualInputPrefix = residualInputScratch.narrow(0, 0, inputTokens);
        at::Tensor normPrefix = postAttentionNormScratch.narrow(0, 0, inputTokens);
        at::Tensor residualOutPrefix = postAttentionResidualScratch.narrow(0, 0, inputTokens);
        at::Tensor residualFloat
            = attentionPrefix.to(at::ScalarType::Float) + residualInputPrefix.to(at::ScalarType::Float);
        residualOutPrefix.copy_(residualFloat.to(attentionHiddenStatesScratch.scalar_type()));
        at::Tensor variance = residualFloat.pow(2).mean(-1, true);
        at::Tensor normalized = residualFloat * at::rsqrt(variance + eps);
        at::Tensor effectiveWeight = useGemma ? weight + 1 : weight;
        at::Tensor output = normalized.to(attentionHiddenStatesScratch.scalar_type()) * effectiveWeight;
        normPrefix.copy_(output);
        return postAttentionNormScratch;
    }

    at::Tensor runLayerPostAttentionGatedNorm(int64_t layerIdx, at::Tensor const& postAttentionNormScratch,
        at::Tensor const& postAttentionGatedScratch, int64_t inputTokens) const
    {
        TORCH_CHECK(layerIdx >= 0 && layerIdx < mNbLayers, "layer_idx is out of range");
        TORCH_CHECK(postAttentionNormScratch.is_cuda(), "post_attention_norm_scratch must be a CUDA tensor");
        TORCH_CHECK(postAttentionGatedScratch.is_cuda(), "post_attention_gated_scratch must be a CUDA tensor");
        TORCH_CHECK(inputTokens > 0, "input_tokens must be positive");
        TORCH_CHECK(postAttentionNormScratch.dim() == 2, "post_attention_norm_scratch must be 2D");
        TORCH_CHECK(postAttentionGatedScratch.dim() == 2, "post_attention_gated_scratch must be 2D");
        TORCH_CHECK(
            postAttentionNormScratch.size(0) >= inputTokens, "post_attention_norm_scratch batch is smaller than input");
        TORCH_CHECK(postAttentionGatedScratch.size(0) >= inputTokens,
            "post_attention_gated_scratch batch is smaller than input");
        TORCH_CHECK(postAttentionGatedScratch.size(1) == postAttentionNormScratch.size(1),
            "post_attention_gated_scratch hidden dim must match post_attention_norm_scratch");
        TORCH_CHECK(postAttentionGatedScratch.scalar_type() == postAttentionNormScratch.scalar_type(),
            "post_attention_gated_scratch dtype must match post_attention_norm_scratch dtype");

        at::Tensor const& downWeight = getLayerTensor(layerIdx,
            DeepseekResidentLayerTensorSite::kPostAttentionGatedNormDown, "post-attention gated norm down weight");
        at::Tensor const& upWeight = getLayerTensor(layerIdx,
            DeepseekResidentLayerTensorSite::kPostAttentionGatedNormUp, "post-attention gated norm up weight");
        TORCH_CHECK(downWeight.dim() == 2, "post-attention gated norm down weight must be 2D");
        TORCH_CHECK(upWeight.dim() == 2, "post-attention gated norm up weight must be 2D");
        TORCH_CHECK(downWeight.size(1) == postAttentionNormScratch.size(1),
            "post-attention gated norm down hidden dim must match post_attention_norm_scratch");
        TORCH_CHECK(upWeight.size(0) == postAttentionNormScratch.size(1),
            "post-attention gated norm up hidden dim must match post_attention_norm_scratch");
        TORCH_CHECK(upWeight.size(1) == downWeight.size(0), "post-attention gated norm rank dimensions must match");

        if (tryRunResidentLowRankGate(
                postAttentionNormScratch, postAttentionGatedScratch, downWeight, upWeight, inputTokens))
        {
            return postAttentionGatedScratch;
        }

        at::Tensor inputPrefix = postAttentionNormScratch.narrow(0, 0, inputTokens);
        at::Tensor gatedPrefix = postAttentionGatedScratch.narrow(0, 0, inputTokens);
        at::Tensor inputFloat = inputPrefix.to(at::ScalarType::Float);
        at::Tensor downFloat = downWeight.to(at::ScalarType::Float);
        at::Tensor gate = at::matmul(inputFloat, downFloat.t());
        gate = gate * at::sigmoid(gate);
        gate = at::matmul(gate.to(upWeight.scalar_type()), upWeight.t());
        gate = at::sigmoid(gate).to(inputPrefix.scalar_type());
        at::Tensor output = inputPrefix * gate;
        gatedPrefix.copy_(output);
        return postAttentionGatedScratch;
    }

    bool tryRunLayerPostAttentionRmsNormLowRankGate(int64_t layerIdx, at::Tensor const& attentionHiddenStatesScratch,
        at::Tensor const& residualInputScratch, at::Tensor const& postAttentionGatedScratch,
        at::Tensor const& postAttentionResidualScratch, int64_t inputTokens, double eps, bool useGemma) const
    {
        TORCH_CHECK(layerIdx >= 0 && layerIdx < mNbLayers, "layer_idx is out of range");
        at::Tensor const& normWeight = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kPostAttentionLayernorm, "post-attention RMSNorm weight");
        at::Tensor const& downWeight = getLayerTensor(layerIdx,
            DeepseekResidentLayerTensorSite::kPostAttentionGatedNormDown, "post-attention gated norm down weight");
        at::Tensor const& upWeight = getLayerTensor(layerIdx,
            DeepseekResidentLayerTensorSite::kPostAttentionGatedNormUp, "post-attention gated norm up weight");
        return tryRunResidentAddRmsNormLowRankGate(attentionHiddenStatesScratch, residualInputScratch,
            postAttentionGatedScratch, postAttentionResidualScratch, normWeight, downWeight, upWeight, inputTokens, eps,
            useGemma);
    }

    at::Tensor runLayerMoeRouterLogits(int64_t layerIdx, at::Tensor const& postAttentionGatedScratch,
        at::Tensor const& routerLogitsScratch, int64_t inputTokens) const
    {
        TORCH_CHECK(layerIdx >= 0 && layerIdx < mNbLayers, "layer_idx is out of range");
        TORCH_CHECK(
            mLayerKinds.at(static_cast<size_t>(layerIdx)) == static_cast<int64_t>(DeepseekResidentLayerKind::kMoe),
            "run_layer_moe_router_logits requires a MoE layer");
        TORCH_CHECK(postAttentionGatedScratch.is_cuda(), "post_attention_gated_scratch must be a CUDA tensor");
        TORCH_CHECK(routerLogitsScratch.is_cuda(), "router_logits_scratch must be a CUDA tensor");
        TORCH_CHECK(inputTokens > 0, "input_tokens must be positive");
        TORCH_CHECK(postAttentionGatedScratch.dim() == 2, "post_attention_gated_scratch must be 2D");
        TORCH_CHECK(routerLogitsScratch.dim() == 2, "router_logits_scratch must be 2D");
        TORCH_CHECK(postAttentionGatedScratch.size(0) >= inputTokens,
            "post_attention_gated_scratch batch is smaller than input");
        TORCH_CHECK(routerLogitsScratch.size(0) >= inputTokens, "router_logits_scratch batch is smaller than input");
        TORCH_CHECK(
            routerLogitsScratch.scalar_type() == at::ScalarType::Float, "router_logits_scratch must be float32");

        at::Tensor const& gateWeight
            = getLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kMoeGateWeight, "MoE gate weight");
        TORCH_CHECK(gateWeight.dim() == 2, "MoE gate weight must be 2D");
        int64_t const numExperts = gateWeight.size(0);
        TORCH_CHECK(numExperts > 0, "MoE gate weight must have experts");
        TORCH_CHECK(gateWeight.size(1) == postAttentionGatedScratch.size(1),
            "MoE gate hidden dim must match post_attention_gated_scratch");
        TORCH_CHECK(routerLogitsScratch.size(1) == numExperts, "router_logits_scratch expert dim is wrong");

        at::Tensor inputPrefix = postAttentionGatedScratch.narrow(0, 0, inputTokens);
        at::Tensor logitsPrefix = routerLogitsScratch.narrow(0, 0, inputTokens);
        at::Tensor routerInput = inputPrefix.is_contiguous() ? inputPrefix : inputPrefix.contiguous();
        dsv3_router_gemm_op_out(routerInput, gateWeight.t(), std::nullopt, logitsPrefix);
        return routerLogitsScratch;
    }

    at::Tensor runLayerMoeRouter(int64_t layerIdx, at::Tensor const& postAttentionGatedScratch,
        at::Tensor const& routerLogitsScratch, at::Tensor const& routerScoresScratch,
        at::Tensor const& routerTopkIndicesScratch, at::Tensor const& routerTopkWeightsScratch, int64_t inputTokens,
        int64_t topK, int64_t nGroup, int64_t topkGroup, double routedScalingFactor) const
    {
        TORCH_CHECK(layerIdx >= 0 && layerIdx < mNbLayers, "layer_idx is out of range");
        TORCH_CHECK(
            mLayerKinds.at(static_cast<size_t>(layerIdx)) == static_cast<int64_t>(DeepseekResidentLayerKind::kMoe),
            "run_layer_moe_router requires a MoE layer");
        TORCH_CHECK(postAttentionGatedScratch.is_cuda(), "post_attention_gated_scratch must be a CUDA tensor");
        TORCH_CHECK(routerLogitsScratch.is_cuda(), "router_logits_scratch must be a CUDA tensor");
        TORCH_CHECK(routerScoresScratch.is_cuda(), "router_scores_scratch must be a CUDA tensor");
        TORCH_CHECK(routerTopkIndicesScratch.is_cuda(), "router_topk_indices_scratch must be a CUDA tensor");
        TORCH_CHECK(routerTopkWeightsScratch.is_cuda(), "router_topk_weights_scratch must be a CUDA tensor");
        TORCH_CHECK(inputTokens > 0, "input_tokens must be positive");
        TORCH_CHECK(topK > 0, "top_k must be positive");
        TORCH_CHECK(nGroup > 0, "n_group must be positive");
        TORCH_CHECK(topkGroup > 0, "topk_group must be positive");
        TORCH_CHECK(postAttentionGatedScratch.dim() == 2, "post_attention_gated_scratch must be 2D");
        TORCH_CHECK(routerLogitsScratch.dim() == 2, "router_logits_scratch must be 2D");
        TORCH_CHECK(routerScoresScratch.dim() == 2, "router_scores_scratch must be 2D");
        TORCH_CHECK(routerTopkIndicesScratch.dim() == 2, "router_topk_indices_scratch must be 2D");
        TORCH_CHECK(routerTopkWeightsScratch.dim() == 2, "router_topk_weights_scratch must be 2D");
        TORCH_CHECK(postAttentionGatedScratch.size(0) >= inputTokens,
            "post_attention_gated_scratch batch is smaller than input");
        TORCH_CHECK(routerLogitsScratch.size(0) >= inputTokens, "router_logits_scratch batch is smaller than input");
        TORCH_CHECK(routerScoresScratch.size(0) >= inputTokens, "router_scores_scratch batch is smaller than input");
        TORCH_CHECK(
            routerTopkIndicesScratch.size(0) >= inputTokens, "router_topk_indices_scratch batch is smaller than input");
        TORCH_CHECK(
            routerTopkWeightsScratch.size(0) >= inputTokens, "router_topk_weights_scratch batch is smaller than input");
        TORCH_CHECK(
            routerLogitsScratch.scalar_type() == at::ScalarType::Float, "router_logits_scratch must be float32");
        TORCH_CHECK(
            routerScoresScratch.scalar_type() == at::ScalarType::Float, "router_scores_scratch must be float32");
        TORCH_CHECK(
            routerTopkIndicesScratch.scalar_type() == at::ScalarType::Int, "router_topk_indices_scratch must be int32");
        TORCH_CHECK(routerTopkWeightsScratch.scalar_type() == at::ScalarType::Float,
            "router_topk_weights_scratch must be float32");

        at::Tensor const& gateWeight
            = getLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kMoeGateWeight, "MoE gate weight");
        at::Tensor const& routingBias = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kMoeGateEScoreCorrectionBias, "MoE e_score_correction_bias");
        TORCH_CHECK(gateWeight.dim() == 2, "MoE gate weight must be 2D");
        TORCH_CHECK(routingBias.dim() == 1, "MoE e_score_correction_bias must be 1D");
        int64_t const numExperts = gateWeight.size(0);
        TORCH_CHECK(numExperts > 0, "MoE gate weight must have experts");
        TORCH_CHECK(gateWeight.size(1) == postAttentionGatedScratch.size(1),
            "MoE gate hidden dim must match post_attention_gated_scratch");
        TORCH_CHECK(routingBias.size(0) == numExperts, "MoE routing bias size must match number of experts");
        TORCH_CHECK(topK <= numExperts, "top_k must be <= num_experts");
        TORCH_CHECK(numExperts % nGroup == 0, "num_experts must be divisible by n_group");
        TORCH_CHECK(routerLogitsScratch.size(1) == numExperts, "router_logits_scratch expert dim is wrong");
        TORCH_CHECK(routerScoresScratch.size(1) == numExperts, "router_scores_scratch expert dim is wrong");
        TORCH_CHECK(routerTopkIndicesScratch.size(1) == topK, "router_topk_indices_scratch top_k dim is wrong");
        TORCH_CHECK(routerTopkWeightsScratch.size(1) == topK, "router_topk_weights_scratch top_k dim is wrong");

        at::Tensor inputPrefix = postAttentionGatedScratch.narrow(0, 0, inputTokens);
        at::Tensor logitsPrefix = routerLogitsScratch.narrow(0, 0, inputTokens);
        at::Tensor scoresPrefix = routerScoresScratch.narrow(0, 0, inputTokens);
        at::Tensor topkIndicesPrefix = routerTopkIndicesScratch.narrow(0, 0, inputTokens);
        at::Tensor topkWeightsPrefix = routerTopkWeightsScratch.narrow(0, 0, inputTokens);

        at::Tensor logits = logitsPrefix;
        if (residentMoeRouterGemmEnabled())
        {
            at::Tensor routerInput = inputPrefix.is_contiguous() ? inputPrefix : inputPrefix.contiguous();
            dsv3_router_gemm_op_out(routerInput, gateWeight.t(), std::nullopt, logitsPrefix);
        }
        else
        {
            logits = at::matmul(inputPrefix.to(at::ScalarType::Float), gateWeight.to(at::ScalarType::Float).t());
            logitsPrefix.copy_(logits);
        }
        at::Tensor scores = at::sigmoid(logits);
        at::Tensor scoresWithBias = scores + routingBias.to(at::ScalarType::Float);
        scoresPrefix.copy_(scores);

        at::Tensor topkValues;
        at::Tensor topkIndices;
        if (nGroup == 1 && topkGroup == 1)
        {
            std::tie(topkValues, topkIndices)
                = at::topk(scoresWithBias, topK, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
            topkValues = scores.gather(/*dim=*/1, topkIndices).to(at::ScalarType::Float);
            at::Tensor topkValuesSum = topkValues.sum(/*dim=*/-1, /*keepdim=*/true) + 1e-20;
            topkValues = topkValues / topkValuesSum * routedScalingFactor;
        }
        else
        {
            int64_t const expertsPerGroup = numExperts / nGroup;
            int64_t const groupTopK = std::min<int64_t>(2, expertsPerGroup);
            int64_t const effectiveTopkGroup = std::min<int64_t>(topkGroup, nGroup);
            at::Tensor groupedScoresWithBias = scoresWithBias.reshape({inputTokens, nGroup, expertsPerGroup});
            at::Tensor groupTopkValues;
            at::Tensor groupTopkIndices;
            std::tie(groupTopkValues, groupTopkIndices)
                = at::topk(groupedScoresWithBias, groupTopK, /*dim=*/-1, /*largest=*/true, /*sorted=*/true);
            static_cast<void>(groupTopkIndices);
            at::Tensor groupScores = groupTopkValues.sum(/*dim=*/-1);
            at::Tensor selectedGroupValues;
            at::Tensor groupIdx;
            std::tie(selectedGroupValues, groupIdx)
                = at::topk(groupScores, effectiveTopkGroup, /*dim=*/-1, /*largest=*/true, /*sorted=*/true);
            static_cast<void>(selectedGroupValues);
            at::Tensor groupMask = at::zeros_like(groupScores);
            groupMask.scatter_(/*dim=*/1, groupIdx, at::ones(groupIdx.sizes(), groupMask.options()));
            at::Tensor scoreMask = groupMask.unsqueeze(-1)
                                       .expand({inputTokens, nGroup, expertsPerGroup})
                                       .reshape({inputTokens, numExperts});
            at::Tensor negativeInf = at::full_like(scoresWithBias, -std::numeric_limits<float>::infinity());
            at::Tensor maskedScoresWithBias
                = at::where(scoreMask.to(at::ScalarType::Bool), scoresWithBias, negativeInf);
            at::Tensor ignoredTopkValues;
            at::Tensor topkIdx;
            std::tie(ignoredTopkValues, topkIdx)
                = at::topk(maskedScoresWithBias, topK, /*dim=*/-1, /*largest=*/true, /*sorted=*/true);
            static_cast<void>(ignoredTopkValues);
            at::Tensor newMask = at::zeros_like(scores);
            newMask.scatter_(/*dim=*/1, topkIdx, at::ones(topkIdx.sizes(), scores.options()));
            at::Tensor selectedScores = scores * newMask;
            at::Tensor scoreSum = selectedScores.sum(/*dim=*/-1, /*keepdim=*/true) + 1e-20;
            selectedScores = selectedScores / scoreSum * routedScalingFactor;
            std::tie(topkValues, topkIndices)
                = at::topk(selectedScores, topK, /*dim=*/-1, /*largest=*/true, /*sorted=*/true);
        }

        topkIndicesPrefix.copy_(topkIndices.to(at::ScalarType::Int));
        topkWeightsPrefix.copy_(topkValues.to(at::ScalarType::Float));
        return routerTopkWeightsScratch;
    }

    static bool isWarpDecodeNvfp4ExpertLayout(at::Tensor const& inputPrefix, at::Tensor const& expertGateUpWeight,
        at::Tensor const& expertDownWeight, int64_t sfVecSize)
    {
        if (inputPrefix.dim() != 2 || expertGateUpWeight.dim() != 3 || expertDownWeight.dim() != 3)
        {
            return false;
        }
        int64_t const hiddenSize = inputPrefix.size(1);
        if (hiddenSize <= 0 || sfVecSize <= 0)
        {
            return false;
        }
        int64_t const packedHiddenSize = expertGateUpWeight.size(2) * sfVecSize;
        int64_t const packedIntermediateSize = expertDownWeight.size(2) * sfVecSize;
        if (packedIntermediateSize <= 0)
        {
            return false;
        }
        return expertGateUpWeight.size(0) == expertDownWeight.size(0) && packedHiddenSize == hiddenSize
            && expertDownWeight.size(1) == hiddenSize && expertGateUpWeight.size(1) == packedIntermediateSize * 2;
    }

    static at::Tensor nvfp4WeightAsByte(at::Tensor const& weight, char const* name)
    {
        TORCH_CHECK(weight.is_cuda(), name, " must be a CUDA tensor");
        TORCH_CHECK(weight.numel() > 0, name, " must not be empty");
        if (weight.scalar_type() == at::ScalarType::Byte)
        {
            return weight.contiguous();
        }
        return weight.contiguous().view(torch::kUInt8);
    }

    static at::Tensor nvfp4ScaleAsFp8(at::Tensor const& scale, char const* name)
    {
        TORCH_CHECK(scale.is_cuda(), name, " must be a CUDA tensor");
        TORCH_CHECK(scale.numel() > 0, name, " must not be empty");
        if (scale.scalar_type() == at::ScalarType::Float8_e4m3fn)
        {
            return scale.contiguous();
        }
        return scale.contiguous().view(torch::kFloat8_e4m3fn);
    }

    static at::Tensor requireFloatScale(std::optional<at::Tensor> const& scale, char const* name)
    {
        TORCH_CHECK(scale.has_value(), name, " is required for WARPDECODE NVFP4 MoE");
        TORCH_CHECK(scale.value().is_cuda(), name, " must be a CUDA tensor");
        TORCH_CHECK(scale.value().scalar_type() == at::ScalarType::Float, name, " must be float32");
        TORCH_CHECK(scale.value().dim() == 1, name, " must be 1D");
        TORCH_CHECK(scale.value().numel() > 0, name, " must not be empty");
        return scale.value().contiguous();
    }

    DeepseekResidentMoeRunnerType& fp4MoeRunner(int64_t tileTokensDim) const
    {
        auto iter = mFp4MoeRunners.find(tileTokensDim);
        if (iter == mFp4MoeRunners.end())
        {
            iter = mFp4MoeRunners
                       .emplace(tileTokensDim,
                           std::make_unique<DeepseekResidentMoeRunnerType>(batchedGemm::trtllm::gen::Dtype::E2m1,
                               batchedGemm::trtllm::gen::Dtype::E2m1, false, tileTokensDim,
                               static_cast<tensorrt_llm::kernels::ActType>(kTrtllmGenSwiGlu)))
                       .first;
        }
        return *iter->second;
    }

    std::pair<int64_t, int64_t> selectWarpDecodeNvfp4MoeTactic(int64_t inputTokens, int64_t hiddenSize, int64_t topK,
        int64_t globalNumExperts, int64_t localNumExperts, int64_t intermediateSize) const
    {
        std::array<int64_t, 6> const supportedTileN{8, 16, 32, 64, 128, 256};
        std::pair<int64_t, int64_t> defaultTactic{-1, -1};
        if (warpDecodeFixedTacticEnabled() && hiddenSize == 7168 && intermediateSize == 2048 && globalNumExperts == 128
            && localNumExperts == 32 && topK == 8)
        {
            if (inputTokens <= 1)
            {
                defaultTactic = {8, 81};
            }
            else if (inputTokens <= 2)
            {
                defaultTactic = {8, 81};
            }
            else if (inputTokens <= 4)
            {
                defaultTactic = {8, 81};
            }
            else if (inputTokens <= 8)
            {
                defaultTactic = {8, 70};
            }
            else if (inputTokens <= 16)
            {
                defaultTactic = {16, 52};
            }
            else if (inputTokens <= 32)
            {
                defaultTactic = {32, 52};
            }
        }

        if (defaultTactic.first < 0 || defaultTactic.second < 0)
        {
            int64_t const denom = std::max<int64_t>(localNumExperts, 1);
            int64_t const avgTokensPerExpert = std::max<int64_t>(1, (inputTokens * topK + denom - 1) / denom);
            int64_t const tileTokensDim
                = std::clamp(nextPowerOfTwoInt64(avgTokensPerExpert), supportedTileN.front(), supportedTileN.back());
            int64_t const configIndex
                = fp4MoeRunner(tileTokensDim)
                      .getDefaultValidConfigIndex(topK, hiddenSize, intermediateSize, localNumExperts, inputTokens);
            defaultTactic = {tileTokensDim, configIndex};
        }

        std::pair<int64_t, int64_t> selectedTactic = defaultTactic;
        auto const overrideTactic = residentMoeTacticOverride();
        bool const usingOverride = overrideTactic.has_value();
        if (overrideTactic)
        {
            int64_t const tileTokensDim = overrideTactic->first;
            int64_t const configIndex = overrideTactic->second;
            bool const supportedTile
                = std::find(supportedTileN.begin(), supportedTileN.end(), tileTokensDim) != supportedTileN.end();
            TORCH_CHECK(supportedTile, "resident MoE tactic override has unsupported tile_tokens_dim: ", tileTokensDim);
            auto const validConfigs
                = fp4MoeRunner(tileTokensDim)
                      .getValidConfigIndices(topK, hiddenSize, intermediateSize, localNumExperts, inputTokens);
            bool const validConfig
                = std::find(validConfigs.begin(), validConfigs.end(), configIndex) != validConfigs.end();
            TORCH_CHECK(validConfig,
                "resident MoE tactic override is invalid for shape: tile_tokens_dim=", tileTokensDim,
                " config_index=", configIndex, " input_tokens=", inputTokens, " hidden_size=", hiddenSize,
                " intermediate_size=", intermediateSize, " local_num_experts=", localNumExperts, " top_k=", topK);
            selectedTactic = *overrideTactic;
        }

        if (residentMoeTacticDebugEnabled())
        {
            std::ostringstream key;
            key << inputTokens << ':' << hiddenSize << ':' << intermediateSize << ':' << localNumExperts << ':'
                << selectedTactic.first << ':' << selectedTactic.second << ':' << usingOverride;
            static std::mutex sLoggedMutex;
            static std::unordered_map<std::string, bool> sLogged;
            bool shouldLog{false};
            {
                std::lock_guard<std::mutex> lock(sLoggedMutex);
                shouldLog = sLogged.emplace(key.str(), true).second;
            }
            if (shouldLog)
            {
                auto const validConfigs
                    = fp4MoeRunner(selectedTactic.first)
                          .getValidConfigIndices(topK, hiddenSize, intermediateSize, localNumExperts, inputTokens);
                std::cout << "[resident_moe_tactic] input_tokens=" << inputTokens << " hidden_size=" << hiddenSize
                          << " intermediate_size=" << intermediateSize << " top_k=" << topK
                          << " local_num_experts=" << localNumExperts << " default_tile=" << defaultTactic.first
                          << " default_config=" << defaultTactic.second << " selected_tile=" << selectedTactic.first
                          << " selected_config=" << selectedTactic.second << " override=" << (usingOverride ? "1" : "0")
                          << " valid_config_count_for_selected_tile=" << validConfigs.size() << std::endl;
            }
        }
        return selectedTactic;
    }

    void runWarpDecodeNvfp4MoeExperts(int64_t layerIdx, at::Tensor const& inputPrefix,
        at::Tensor const& routerTopkIndicesScratch, at::Tensor const& routerTopkWeightsScratch,
        at::Tensor const& outputPrefix, int64_t inputTokens, int64_t sfVecSize, int64_t globalNumExperts,
        int64_t localExpertOffset, int64_t localNumExperts, int64_t intermediateSize) const
    {
        TORCH_CHECK(outputPrefix.scalar_type() == at::ScalarType::BFloat16,
            "WARPDECODE NVFP4 MoE output scratch must be bfloat16");
        at::Tensor const& expertGateUpWeight
            = getLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kExpertGateUpWeight, "expert gate_up weight");
        at::Tensor const& expertDownWeight
            = getLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kExpertDownWeight, "expert down weight");
        std::optional<at::Tensor> const expertGateUpWeightScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kExpertGateUpWeightScale);
        std::optional<at::Tensor> const expertGateUpInputScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kExpertGateUpInputScale);
        std::optional<at::Tensor> const expertGateUpOutputScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kExpertGateUpOutputScale);
        std::optional<at::Tensor> const expertGateUpAlpha
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kExpertGateUpAlpha);
        std::optional<at::Tensor> const expertDownWeightScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kExpertDownWeightScale);
        std::optional<at::Tensor> const expertDownAlpha
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kExpertDownAlpha);

        TORCH_CHECK(expertGateUpWeightScale.has_value(), "WARPDECODE NVFP4 MoE requires expert gate_up weight scale");
        TORCH_CHECK(expertGateUpInputScale.has_value(), "WARPDECODE NVFP4 MoE requires expert gate_up input scale");
        TORCH_CHECK(expertGateUpAlpha.has_value(), "WARPDECODE NVFP4 MoE requires expert gate_up alpha");
        TORCH_CHECK(expertDownWeightScale.has_value(), "WARPDECODE NVFP4 MoE requires expert down weight scale");
        TORCH_CHECK(expertDownAlpha.has_value(), "WARPDECODE NVFP4 MoE requires expert down alpha");

        int64_t const resolvedLocalNumExperts = localNumExperts > 0 ? localNumExperts : expertGateUpWeight.size(0);
        int64_t const resolvedIntermediateSize
            = intermediateSize > 0 ? intermediateSize : expertDownWeight.size(2) * sfVecSize;
        int64_t const resolvedGlobalNumExperts = globalNumExperts > 0 ? globalNumExperts : moeNumExperts(layerIdx);
        int64_t const hiddenSize = inputPrefix.size(1);
        int64_t const topK = routerTopkIndicesScratch.size(1);
        TORCH_CHECK(resolvedLocalNumExperts == expertGateUpWeight.size(0),
            "WARPDECODE NVFP4 MoE local expert count must match gate_up weight");
        TORCH_CHECK(resolvedLocalNumExperts == expertDownWeight.size(0),
            "WARPDECODE NVFP4 MoE local expert count must match down weight");
        TORCH_CHECK(resolvedGlobalNumExperts >= resolvedLocalNumExperts,
            "WARPDECODE NVFP4 MoE global expert count must be >= local expert count");
        TORCH_CHECK(expertGateUpWeight.size(2) * sfVecSize == hiddenSize,
            "WARPDECODE NVFP4 MoE gate_up packed hidden dim must match input");
        TORCH_CHECK(expertDownWeight.size(1) == hiddenSize, "WARPDECODE NVFP4 MoE down hidden dim must match input");
        TORCH_CHECK(expertGateUpWeight.size(1) == resolvedIntermediateSize * 2,
            "WARPDECODE NVFP4 MoE gate_up weight dim must match intermediate size");
        TORCH_CHECK(expertDownWeight.size(2) * sfVecSize == resolvedIntermediateSize,
            "WARPDECODE NVFP4 MoE down weight dim must match packed intermediate size");

        at::Tensor inputForQuant = inputPrefix.narrow(0, 0, inputTokens);
        if (!inputForQuant.is_contiguous())
        {
            inputForQuant = inputForQuant.contiguous();
        }
        auto fp4Input = callTrtllmFp4Quantize(inputForQuant, expertGateUpInputScale.value(), sfVecSize, false);
        at::Tensor hiddenStates = std::get<0>(fp4Input);
        at::Tensor hiddenStatesScale = std::get<1>(fp4Input).reshape({-1}).view(torch::kFloat8_e4m3fn);
        at::Tensor topkIds = routerTopkIndicesScratch.narrow(0, 0, inputTokens);
        if (topkIds.scalar_type() != at::ScalarType::Int)
        {
            topkIds = topkIds.to(at::ScalarType::Int);
        }
        if (!topkIds.is_contiguous())
        {
            topkIds = topkIds.contiguous();
        }
        at::Tensor topkWeights
            = routerTopkWeightsScratch.narrow(0, 0, inputTokens).to(at::ScalarType::BFloat16).contiguous();
        at::Tensor expertGateUpWeightBytes = nvfp4WeightAsByte(expertGateUpWeight, "expert gate_up weight");
        at::Tensor expertDownWeightBytes = nvfp4WeightAsByte(expertDownWeight, "expert down weight");
        at::Tensor output1Scale = expertGateUpOutputScale.has_value()
            ? requireFloatScale(expertGateUpOutputScale, "expert gate_up output scale")
            : requireFloatScale(expertGateUpAlpha, "expert gate_up alpha");
        at::Tensor output1GateScale = requireFloatScale(expertGateUpAlpha, "expert gate_up alpha");
        at::Tensor output2Scale = requireFloatScale(expertDownAlpha, "expert down alpha");
        at::Tensor w13Scale = nvfp4ScaleAsFp8(expertGateUpWeightScale.value(), "expert gate_up weight scale");
        at::Tensor w2Scale = nvfp4ScaleAsFp8(expertDownWeightScale.value(), "expert down weight scale");

        auto const tactic = selectWarpDecodeNvfp4MoeTactic(
            inputTokens, hiddenSize, topK, resolvedGlobalNumExperts, resolvedLocalNumExperts, resolvedIntermediateSize);
        torch::optional<torch::Tensor> topkWeightsOpt(topkWeights);
        torch::optional<torch::Tensor> topkIdsOpt(topkIds);
        torch::optional<torch::Tensor> outputOpt(outputPrefix);
        auto& runner = fp4MoeRunner(tactic.first);
        static_cast<void>(run_fp4_block_scale_moe_runner(torch::nullopt, torch::nullopt, hiddenStates,
            hiddenStatesScale, expertGateUpWeightBytes, w13Scale, std::nullopt, std::nullopt, std::nullopt,
            std::nullopt, expertDownWeightBytes, w2Scale, std::nullopt, output1Scale, output1GateScale, output2Scale,
            resolvedGlobalNumExperts, topK, int64_t{8}, int64_t{4}, resolvedIntermediateSize, localExpertOffset,
            resolvedLocalNumExperts, std::nullopt, tactic.first, kTrtllmGenDeepSeekV3Routing, true,
            batchedGemm::trtllm::gen::Dtype::E2m1, runner, tactic.second, topkWeightsOpt, topkIdsOpt, outputOpt));
    }

    void runWarpDecodeNvfp4MoeExpertsFromRoutingLogits(int64_t layerIdx, at::Tensor const& inputPrefix,
        at::Tensor const& routerLogitsScratch, at::Tensor const& outputPrefix, int64_t inputTokens, int64_t topK,
        int64_t nGroup, int64_t topkGroup, double routedScalingFactor, int64_t sfVecSize, int64_t globalNumExperts,
        int64_t localExpertOffset, int64_t localNumExperts, int64_t intermediateSize) const
    {
        TORCH_CHECK(outputPrefix.scalar_type() == at::ScalarType::BFloat16,
            "WARPDECODE NVFP4 MoE output scratch must be bfloat16");
        TORCH_CHECK(routerLogitsScratch.is_cuda(), "router_logits_scratch must be a CUDA tensor");
        TORCH_CHECK(
            routerLogitsScratch.scalar_type() == at::ScalarType::Float, "router_logits_scratch must be float32");
        TORCH_CHECK(routerLogitsScratch.dim() == 2, "router_logits_scratch must be 2D");
        TORCH_CHECK(routerLogitsScratch.size(0) >= inputTokens, "router_logits_scratch batch is smaller than input");
        TORCH_CHECK(topK > 0, "raw-routing WARPDECODE NVFP4 MoE top_k must be positive");
        TORCH_CHECK(nGroup > 0, "raw-routing WARPDECODE NVFP4 MoE n_group must be positive");
        TORCH_CHECK(topkGroup > 0, "raw-routing WARPDECODE NVFP4 MoE topk_group must be positive");

        at::Tensor const& routingBias = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kMoeGateEScoreCorrectionBias, "MoE e_score_correction_bias");
        at::Tensor const& expertGateUpWeight
            = getLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kExpertGateUpWeight, "expert gate_up weight");
        at::Tensor const& expertDownWeight
            = getLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kExpertDownWeight, "expert down weight");
        std::optional<at::Tensor> const expertGateUpWeightScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kExpertGateUpWeightScale);
        std::optional<at::Tensor> const expertGateUpInputScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kExpertGateUpInputScale);
        std::optional<at::Tensor> const expertGateUpOutputScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kExpertGateUpOutputScale);
        std::optional<at::Tensor> const expertGateUpAlpha
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kExpertGateUpAlpha);
        std::optional<at::Tensor> const expertDownWeightScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kExpertDownWeightScale);
        std::optional<at::Tensor> const expertDownAlpha
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kExpertDownAlpha);

        TORCH_CHECK(routingBias.dim() == 1, "MoE e_score_correction_bias must be 1D");
        TORCH_CHECK(expertGateUpWeightScale.has_value(), "WARPDECODE NVFP4 MoE requires expert gate_up weight scale");
        TORCH_CHECK(expertGateUpInputScale.has_value(), "WARPDECODE NVFP4 MoE requires expert gate_up input scale");
        TORCH_CHECK(expertGateUpAlpha.has_value(), "WARPDECODE NVFP4 MoE requires expert gate_up alpha");
        TORCH_CHECK(expertDownWeightScale.has_value(), "WARPDECODE NVFP4 MoE requires expert down weight scale");
        TORCH_CHECK(expertDownAlpha.has_value(), "WARPDECODE NVFP4 MoE requires expert down alpha");

        int64_t const resolvedLocalNumExperts = localNumExperts > 0 ? localNumExperts : expertGateUpWeight.size(0);
        int64_t const resolvedIntermediateSize
            = intermediateSize > 0 ? intermediateSize : expertDownWeight.size(2) * sfVecSize;
        int64_t const resolvedGlobalNumExperts = globalNumExperts > 0 ? globalNumExperts : moeNumExperts(layerIdx);
        int64_t const hiddenSize = inputPrefix.size(1);
        TORCH_CHECK(routerLogitsScratch.size(1) == resolvedGlobalNumExperts,
            "router_logits_scratch expert dim must match global expert count");
        TORCH_CHECK(routingBias.size(0) == resolvedGlobalNumExperts,
            "MoE e_score_correction_bias size must match global expert count");
        TORCH_CHECK(resolvedLocalNumExperts == expertGateUpWeight.size(0),
            "WARPDECODE NVFP4 MoE local expert count must match gate_up weight");
        TORCH_CHECK(resolvedLocalNumExperts == expertDownWeight.size(0),
            "WARPDECODE NVFP4 MoE local expert count must match down weight");
        TORCH_CHECK(resolvedGlobalNumExperts >= resolvedLocalNumExperts,
            "WARPDECODE NVFP4 MoE global expert count must be >= local expert count");
        TORCH_CHECK(expertGateUpWeight.size(2) * sfVecSize == hiddenSize,
            "WARPDECODE NVFP4 MoE gate_up packed hidden dim must match input");
        TORCH_CHECK(expertDownWeight.size(1) == hiddenSize, "WARPDECODE NVFP4 MoE down hidden dim must match input");
        TORCH_CHECK(expertGateUpWeight.size(1) == resolvedIntermediateSize * 2,
            "WARPDECODE NVFP4 MoE gate_up weight dim must match intermediate size");
        TORCH_CHECK(expertDownWeight.size(2) * sfVecSize == resolvedIntermediateSize,
            "WARPDECODE NVFP4 MoE down weight dim must match packed intermediate size");

        at::Tensor inputForQuant = inputPrefix.narrow(0, 0, inputTokens);
        if (!inputForQuant.is_contiguous())
        {
            inputForQuant = inputForQuant.contiguous();
        }
        auto fp4Input = callTrtllmFp4Quantize(inputForQuant, expertGateUpInputScale.value(), sfVecSize, false);
        at::Tensor hiddenStates = std::get<0>(fp4Input);
        at::Tensor hiddenStatesScale = std::get<1>(fp4Input).reshape({-1}).view(torch::kFloat8_e4m3fn);
        at::Tensor routingLogits = routerLogitsScratch.narrow(0, 0, inputTokens);
        if (!routingLogits.is_contiguous())
        {
            routingLogits = routingLogits.contiguous();
        }
        at::Tensor routingBiasContiguous = routingBias.is_contiguous() ? routingBias : routingBias.contiguous();
        at::Tensor expertGateUpWeightBytes = nvfp4WeightAsByte(expertGateUpWeight, "expert gate_up weight");
        at::Tensor expertDownWeightBytes = nvfp4WeightAsByte(expertDownWeight, "expert down weight");
        at::Tensor output1Scale = expertGateUpOutputScale.has_value()
            ? requireFloatScale(expertGateUpOutputScale, "expert gate_up output scale")
            : requireFloatScale(expertGateUpAlpha, "expert gate_up alpha");
        at::Tensor output1GateScale = requireFloatScale(expertGateUpAlpha, "expert gate_up alpha");
        at::Tensor output2Scale = requireFloatScale(expertDownAlpha, "expert down alpha");
        at::Tensor w13Scale = nvfp4ScaleAsFp8(expertGateUpWeightScale.value(), "expert gate_up weight scale");
        at::Tensor w2Scale = nvfp4ScaleAsFp8(expertDownWeightScale.value(), "expert down weight scale");

        auto const tactic = selectWarpDecodeNvfp4MoeTactic(
            inputTokens, hiddenSize, topK, resolvedGlobalNumExperts, resolvedLocalNumExperts, resolvedIntermediateSize);
        torch::optional<torch::Tensor> routingLogitsOpt(routingLogits);
        torch::optional<torch::Tensor> routingBiasOpt(routingBiasContiguous);
        torch::optional<torch::Tensor> outputOpt(outputPrefix);
        auto& runner = fp4MoeRunner(tactic.first);
        static_cast<void>(run_fp4_block_scale_moe_runner(routingLogitsOpt, routingBiasOpt, hiddenStates,
            hiddenStatesScale, expertGateUpWeightBytes, w13Scale, std::nullopt, std::nullopt, std::nullopt,
            std::nullopt, expertDownWeightBytes, w2Scale, std::nullopt, output1Scale, output1GateScale, output2Scale,
            resolvedGlobalNumExperts, topK, nGroup, topkGroup, resolvedIntermediateSize, localExpertOffset,
            resolvedLocalNumExperts, std::optional<double>(routedScalingFactor), tactic.first,
            kTrtllmGenDeepSeekV3Routing, true, batchedGemm::trtllm::gen::Dtype::E2m1, runner, tactic.second,
            torch::nullopt, torch::nullopt, outputOpt));
    }

    at::Tensor runLayerMoeExpertsImpl(int64_t layerIdx, at::Tensor const& postAttentionGatedScratch,
        at::Tensor const& routerTopkIndicesScratch, at::Tensor const& routerTopkWeightsScratch,
        at::Tensor const& denseMlpOutputScratch, at::Tensor const& moeSharedIntermediateScratch,
        at::Tensor const& moeSharedGateUpScratch, at::Tensor const& moeSharedOutputScratch, int64_t inputTokens,
        int64_t sfVecSize, std::string const& allowedBackends, double sharedOutputScale, int64_t globalNumExperts,
        int64_t localExpertOffset, int64_t localNumExperts, int64_t intermediateSize) const
    {
        TORCH_CHECK(layerIdx >= 0 && layerIdx < mNbLayers, "layer_idx is out of range");
        TORCH_CHECK(
            mLayerKinds.at(static_cast<size_t>(layerIdx)) == static_cast<int64_t>(DeepseekResidentLayerKind::kMoe),
            "run_layer_moe_experts requires a MoE layer");
        TORCH_CHECK(runLayerMoeExpertAssetsReady(layerIdx), runLayerMoeExpertAssetsNotReadyReason(layerIdx));
        TORCH_CHECK(postAttentionGatedScratch.is_cuda(), "post_attention_gated_scratch must be a CUDA tensor");
        TORCH_CHECK(routerTopkIndicesScratch.is_cuda(), "router_topk_indices_scratch must be a CUDA tensor");
        TORCH_CHECK(routerTopkWeightsScratch.is_cuda(), "router_topk_weights_scratch must be a CUDA tensor");
        TORCH_CHECK(denseMlpOutputScratch.is_cuda(), "dense_mlp_output_scratch must be a CUDA tensor");
        TORCH_CHECK(moeSharedIntermediateScratch.is_cuda(), "moe_shared_intermediate_scratch must be a CUDA tensor");
        TORCH_CHECK(moeSharedGateUpScratch.is_cuda(), "moe_shared_gate_up_scratch must be a CUDA tensor");
        TORCH_CHECK(moeSharedOutputScratch.is_cuda(), "moe_shared_output_scratch must be a CUDA tensor");
        TORCH_CHECK(inputTokens > 0, "input_tokens must be positive");
        TORCH_CHECK(sfVecSize > 0, "NVFP4 scaling vector size must be positive");
        TORCH_CHECK(postAttentionGatedScratch.dim() == 2, "post_attention_gated_scratch must be 2D");
        TORCH_CHECK(routerTopkIndicesScratch.dim() == 2, "router_topk_indices_scratch must be 2D");
        TORCH_CHECK(routerTopkWeightsScratch.dim() == 2, "router_topk_weights_scratch must be 2D");
        TORCH_CHECK(denseMlpOutputScratch.dim() == 2, "dense_mlp_output_scratch must be 2D");
        TORCH_CHECK(moeSharedIntermediateScratch.dim() == 2, "moe_shared_intermediate_scratch must be 2D");
        TORCH_CHECK(moeSharedGateUpScratch.dim() == 2, "moe_shared_gate_up_scratch must be 2D");
        TORCH_CHECK(moeSharedOutputScratch.dim() == 2, "moe_shared_output_scratch must be 2D");
        TORCH_CHECK(postAttentionGatedScratch.size(0) >= inputTokens,
            "post_attention_gated_scratch batch is smaller than input");
        TORCH_CHECK(
            routerTopkIndicesScratch.size(0) >= inputTokens, "router_topk_indices_scratch batch is smaller than input");
        TORCH_CHECK(
            routerTopkWeightsScratch.size(0) >= inputTokens, "router_topk_weights_scratch batch is smaller than input");
        TORCH_CHECK(
            denseMlpOutputScratch.size(0) >= inputTokens, "dense_mlp_output_scratch batch is smaller than input");
        TORCH_CHECK(moeSharedIntermediateScratch.size(0) >= inputTokens,
            "moe_shared_intermediate_scratch batch is smaller than input");
        TORCH_CHECK(
            moeSharedGateUpScratch.size(0) >= inputTokens, "moe_shared_gate_up_scratch batch is smaller than input");
        TORCH_CHECK(
            moeSharedOutputScratch.size(0) >= inputTokens, "moe_shared_output_scratch batch is smaller than input");
        TORCH_CHECK(denseMlpOutputScratch.size(1) == postAttentionGatedScratch.size(1),
            "dense_mlp_output_scratch hidden dim must match post_attention_gated_scratch");
        TORCH_CHECK(moeSharedOutputScratch.size(1) >= postAttentionGatedScratch.size(1),
            "moe_shared_output_scratch hidden dim must cover post_attention_gated_scratch");
        TORCH_CHECK(routerTopkIndicesScratch.scalar_type() == at::ScalarType::Int
                || routerTopkIndicesScratch.scalar_type() == at::ScalarType::Long,
            "router_topk_indices_scratch must be int32 or int64");
        TORCH_CHECK(routerTopkWeightsScratch.scalar_type() == at::ScalarType::Float,
            "router_topk_weights_scratch must be float32");

        at::Tensor const& sharedGateUpWeight = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kSharedExpertGateUpWeight, "shared expert gate_up weight");
        at::Tensor const& sharedDownWeight = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kSharedExpertDownWeight, "shared expert down weight");
        at::Tensor const& expertGateUpWeight
            = getLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kExpertGateUpWeight, "expert gate_up weight");
        at::Tensor const& expertDownWeight
            = getLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kExpertDownWeight, "expert down weight");
        TORCH_CHECK(sharedGateUpWeight.dim() == 2, "shared expert gate_up weight must be 2D");
        TORCH_CHECK(sharedDownWeight.dim() == 2, "shared expert down weight must be 2D");
        TORCH_CHECK(expertGateUpWeight.dim() == 3, "expert gate_up weight must be 3D");
        TORCH_CHECK(expertDownWeight.dim() == 3, "expert down weight must be 3D");
        int64_t const numExperts = expertGateUpWeight.size(0);
        int64_t const topK = routerTopkIndicesScratch.size(1);
        TORCH_CHECK(numExperts > 0, "expert weights must contain at least one expert");
        TORCH_CHECK(topK > 0, "router top_k must be positive");

        at::Tensor inputPrefix = postAttentionGatedScratch.narrow(0, 0, inputTokens);
        at::Tensor outputPrefix = denseMlpOutputScratch.narrow(0, 0, inputTokens);
        at::Tensor sharedOutput = runSwiGluMlpToScratch(inputPrefix, sharedGateUpWeight, sharedDownWeight,
            getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kSharedExpertGateUpWeightScale),
            getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kSharedExpertGateUpInputScale),
            getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kSharedExpertGateUpAlpha),
            getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kSharedExpertDownWeightScale),
            getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kSharedExpertDownInputScale),
            getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kSharedExpertDownAlpha),
            moeSharedGateUpScratch, moeSharedIntermediateScratch, moeSharedOutputScratch, inputTokens, sfVecSize,
            allowedBackends, "moe_shared_gate_up", "moe_shared_down");

        if (isWarpDecodeNvfp4ExpertLayout(inputPrefix, expertGateUpWeight, expertDownWeight, sfVecSize))
        {
            runWarpDecodeNvfp4MoeExperts(layerIdx, inputPrefix, routerTopkIndicesScratch, routerTopkWeightsScratch,
                outputPrefix, inputTokens, sfVecSize, globalNumExperts, localExpertOffset, localNumExperts,
                intermediateSize);
            if (!tryRunResidentAddScaledFloatToOutput(sharedOutput, outputPrefix, inputTokens, sharedOutputScale))
            {
                at::Tensor scaledSharedOutput = sharedOutput;
                if (sharedOutputScale != 1.0)
                {
                    scaledSharedOutput = sharedOutput * sharedOutputScale;
                }
                outputPrefix.copy_((scaledSharedOutput + outputPrefix.to(scaledSharedOutput.scalar_type()))
                        .to(denseMlpOutputScratch.scalar_type()));
            }
            return denseMlpOutputScratch;
        }

        if (sharedOutputScale != 1.0)
        {
            sharedOutput = sharedOutput * sharedOutputScale;
        }
        at::Tensor routedOutput = at::zeros_like(sharedOutput);
        at::Tensor const topkIndices = routerTopkIndicesScratch.narrow(0, 0, inputTokens).to(at::ScalarType::Long);
        at::Tensor const topkWeights = routerTopkWeightsScratch.narrow(0, 0, inputTokens).to(at::ScalarType::Float);
        at::Tensor const flatExperts = topkIndices.reshape({inputTokens * topK});
        at::Tensor const flatWeights = topkWeights.reshape({inputTokens * topK});
        std::optional<at::Tensor> const expertGateUpWeightScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kExpertGateUpWeightScale);
        std::optional<at::Tensor> const expertGateUpInputScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kExpertGateUpInputScale);
        std::optional<at::Tensor> const expertGateUpAlpha
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kExpertGateUpAlpha);
        std::optional<at::Tensor> const expertDownWeightScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kExpertDownWeightScale);
        std::optional<at::Tensor> const expertDownInputScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kExpertDownInputScale);
        std::optional<at::Tensor> const expertDownAlpha
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kExpertDownAlpha);

        for (int64_t expertIdx = 0; expertIdx < numExperts; ++expertIdx)
        {
            at::Tensor positions = at::nonzero(flatExperts.eq(expertIdx)).reshape({-1});
            if (positions.numel() == 0)
            {
                continue;
            }
            at::Tensor tokenIndices = at::floor_divide(positions, topK);
            at::Tensor expertInput = inputPrefix.index_select(0, tokenIndices);
            at::Tensor expertOutput = runSwiGluMlpToFloat(expertInput, expertGateUpWeight.select(0, expertIdx),
                expertDownWeight.select(0, expertIdx), selectExpertTensor(expertGateUpWeightScale, expertIdx),
                selectExpertTensor(expertGateUpInputScale, expertIdx), selectExpertTensor(expertGateUpAlpha, expertIdx),
                selectExpertTensor(expertDownWeightScale, expertIdx),
                selectExpertTensor(expertDownInputScale, expertIdx), selectExpertTensor(expertDownAlpha, expertIdx),
                expertInput.size(0), sfVecSize, allowedBackends, "moe_routed_gate_up_fallback",
                "moe_routed_down_fallback");
            at::Tensor expertWeights = flatWeights.index_select(0, positions).unsqueeze(1);
            routedOutput.index_add_(0, tokenIndices, expertOutput * expertWeights);
        }

        outputPrefix.copy_((sharedOutput + routedOutput).to(denseMlpOutputScratch.scalar_type()));
        return denseMlpOutputScratch;
    }

    at::Tensor runLayerMoeExperts(int64_t layerIdx, at::Tensor const& postAttentionGatedScratch,
        at::Tensor const& routerTopkIndicesScratch, at::Tensor const& routerTopkWeightsScratch,
        at::Tensor const& denseMlpOutputScratch, at::Tensor const& moeSharedIntermediateScratch,
        at::Tensor const& moeSharedGateUpScratch, at::Tensor const& moeSharedOutputScratch, int64_t inputTokens,
        int64_t sfVecSize, std::string const& allowedBackends, double sharedOutputScale, int64_t globalNumExperts,
        int64_t localExpertOffset, int64_t localNumExperts, int64_t intermediateSize) const
    {
        return runLayerMoeExpertsImpl(layerIdx, postAttentionGatedScratch, routerTopkIndicesScratch,
            routerTopkWeightsScratch, denseMlpOutputScratch, moeSharedIntermediateScratch, moeSharedGateUpScratch,
            moeSharedOutputScratch, inputTokens, sfVecSize, allowedBackends, sharedOutputScale, globalNumExperts,
            localExpertOffset, localNumExperts, intermediateSize);
    }

    at::Tensor runLayerMoeExpertsFromRoutingLogitsImpl(int64_t layerIdx, at::Tensor const& postAttentionGatedScratch,
        at::Tensor const& routerLogitsScratch, at::Tensor const& denseMlpOutputScratch,
        at::Tensor const& moeSharedIntermediateScratch, at::Tensor const& moeSharedGateUpScratch,
        at::Tensor const& moeSharedOutputScratch, int64_t inputTokens, int64_t topK, int64_t nGroup, int64_t topkGroup,
        double routedScalingFactor, int64_t sfVecSize, std::string const& allowedBackends, double sharedOutputScale,
        int64_t globalNumExperts, int64_t localExpertOffset, int64_t localNumExperts, int64_t intermediateSize,
        ResidentMoeExpertTiming* timing = nullptr) const
    {
        TORCH_CHECK(layerIdx >= 0 && layerIdx < mNbLayers, "layer_idx is out of range");
        TORCH_CHECK(
            mLayerKinds.at(static_cast<size_t>(layerIdx)) == static_cast<int64_t>(DeepseekResidentLayerKind::kMoe),
            "run_layer_moe_experts_from_routing_logits requires a MoE layer");
        TORCH_CHECK(runLayerMoeExpertAssetsReady(layerIdx), runLayerMoeExpertAssetsNotReadyReason(layerIdx));
        TORCH_CHECK(postAttentionGatedScratch.is_cuda(), "post_attention_gated_scratch must be a CUDA tensor");
        TORCH_CHECK(routerLogitsScratch.is_cuda(), "router_logits_scratch must be a CUDA tensor");
        TORCH_CHECK(denseMlpOutputScratch.is_cuda(), "dense_mlp_output_scratch must be a CUDA tensor");
        TORCH_CHECK(moeSharedIntermediateScratch.is_cuda(), "moe_shared_intermediate_scratch must be a CUDA tensor");
        TORCH_CHECK(moeSharedGateUpScratch.is_cuda(), "moe_shared_gate_up_scratch must be a CUDA tensor");
        TORCH_CHECK(moeSharedOutputScratch.is_cuda(), "moe_shared_output_scratch must be a CUDA tensor");
        TORCH_CHECK(inputTokens > 0, "input_tokens must be positive");
        TORCH_CHECK(topK > 0, "top_k must be positive");
        TORCH_CHECK(nGroup > 0, "n_group must be positive");
        TORCH_CHECK(topkGroup > 0, "topk_group must be positive");
        TORCH_CHECK(sfVecSize > 0, "NVFP4 scaling vector size must be positive");
        TORCH_CHECK(postAttentionGatedScratch.dim() == 2, "post_attention_gated_scratch must be 2D");
        TORCH_CHECK(routerLogitsScratch.dim() == 2, "router_logits_scratch must be 2D");
        TORCH_CHECK(denseMlpOutputScratch.dim() == 2, "dense_mlp_output_scratch must be 2D");
        TORCH_CHECK(moeSharedIntermediateScratch.dim() == 2, "moe_shared_intermediate_scratch must be 2D");
        TORCH_CHECK(moeSharedGateUpScratch.dim() == 2, "moe_shared_gate_up_scratch must be 2D");
        TORCH_CHECK(moeSharedOutputScratch.dim() == 2, "moe_shared_output_scratch must be 2D");
        TORCH_CHECK(postAttentionGatedScratch.size(0) >= inputTokens,
            "post_attention_gated_scratch batch is smaller than input");
        TORCH_CHECK(routerLogitsScratch.size(0) >= inputTokens, "router_logits_scratch batch is smaller than input");
        TORCH_CHECK(
            denseMlpOutputScratch.size(0) >= inputTokens, "dense_mlp_output_scratch batch is smaller than input");
        TORCH_CHECK(moeSharedIntermediateScratch.size(0) >= inputTokens,
            "moe_shared_intermediate_scratch batch is smaller than input");
        TORCH_CHECK(
            moeSharedGateUpScratch.size(0) >= inputTokens, "moe_shared_gate_up_scratch batch is smaller than input");
        TORCH_CHECK(
            moeSharedOutputScratch.size(0) >= inputTokens, "moe_shared_output_scratch batch is smaller than input");
        TORCH_CHECK(denseMlpOutputScratch.size(1) == postAttentionGatedScratch.size(1),
            "dense_mlp_output_scratch hidden dim must match post_attention_gated_scratch");
        TORCH_CHECK(moeSharedOutputScratch.size(1) >= postAttentionGatedScratch.size(1),
            "moe_shared_output_scratch hidden dim must cover post_attention_gated_scratch");
        TORCH_CHECK(
            routerLogitsScratch.scalar_type() == at::ScalarType::Float, "router_logits_scratch must be float32");

        at::Tensor const& sharedGateUpWeight = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kSharedExpertGateUpWeight, "shared expert gate_up weight");
        at::Tensor const& sharedDownWeight = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kSharedExpertDownWeight, "shared expert down weight");
        at::Tensor const& expertGateUpWeight
            = getLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kExpertGateUpWeight, "expert gate_up weight");
        at::Tensor const& expertDownWeight
            = getLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kExpertDownWeight, "expert down weight");
        TORCH_CHECK(sharedGateUpWeight.dim() == 2, "shared expert gate_up weight must be 2D");
        TORCH_CHECK(sharedDownWeight.dim() == 2, "shared expert down weight must be 2D");
        TORCH_CHECK(expertGateUpWeight.dim() == 3, "expert gate_up weight must be 3D");
        TORCH_CHECK(expertDownWeight.dim() == 3, "expert down weight must be 3D");

        at::Tensor inputPrefix = postAttentionGatedScratch.narrow(0, 0, inputTokens);
        at::Tensor outputPrefix = denseMlpOutputScratch.narrow(0, 0, inputTokens);
        TORCH_CHECK(isWarpDecodeNvfp4ExpertLayout(inputPrefix, expertGateUpWeight, expertDownWeight, sfVecSize),
            "raw-routing MoE experts currently require WARPDECODE NVFP4 expert layout");

        at::Tensor sharedOutput;
        {
            double ignoredUs = 0.0;
            ResidentWindowScopedTimer const timer(
                timing != nullptr, timing != nullptr ? timing->sharedExpertUs : ignoredUs);
            sharedOutput = runSwiGluMlpToScratch(inputPrefix, sharedGateUpWeight, sharedDownWeight,
                getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kSharedExpertGateUpWeightScale),
                getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kSharedExpertGateUpInputScale),
                getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kSharedExpertGateUpAlpha),
                getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kSharedExpertDownWeightScale),
                getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kSharedExpertDownInputScale),
                getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kSharedExpertDownAlpha),
                moeSharedGateUpScratch, moeSharedIntermediateScratch, moeSharedOutputScratch, inputTokens, sfVecSize,
                allowedBackends, "moe_shared_gate_up", "moe_shared_down");
        }

        {
            double ignoredUs = 0.0;
            ResidentWindowScopedTimer const timer(
                timing != nullptr, timing != nullptr ? timing->routedExpertUs : ignoredUs);
            runWarpDecodeNvfp4MoeExpertsFromRoutingLogits(layerIdx, inputPrefix, routerLogitsScratch, outputPrefix,
                inputTokens, topK, nGroup, topkGroup, routedScalingFactor, sfVecSize, globalNumExperts,
                localExpertOffset, localNumExperts, intermediateSize);
        }
        {
            double ignoredUs = 0.0;
            ResidentWindowScopedTimer const timer(timing != nullptr, timing != nullptr ? timing->combineUs : ignoredUs);
            if (tryRunResidentAddScaledFloatToOutput(sharedOutput, outputPrefix, inputTokens, sharedOutputScale))
            {
                return denseMlpOutputScratch;
            }
            {
                at::Tensor scaledSharedOutput = sharedOutput;
                if (sharedOutputScale != 1.0)
                {
                    scaledSharedOutput = sharedOutput * sharedOutputScale;
                }
                outputPrefix.copy_((scaledSharedOutput + outputPrefix.to(scaledSharedOutput.scalar_type()))
                        .to(denseMlpOutputScratch.scalar_type()));
            }
        }
        return denseMlpOutputScratch;
    }

    at::Tensor runLayerMoeExpertsFromRoutingLogits(int64_t layerIdx, at::Tensor const& postAttentionGatedScratch,
        at::Tensor const& routerLogitsScratch, at::Tensor const& denseMlpOutputScratch,
        at::Tensor const& moeSharedIntermediateScratch, at::Tensor const& moeSharedGateUpScratch,
        at::Tensor const& moeSharedOutputScratch, int64_t inputTokens, int64_t topK, int64_t nGroup, int64_t topkGroup,
        double routedScalingFactor, int64_t sfVecSize, std::string const& allowedBackends, double sharedOutputScale,
        int64_t globalNumExperts, int64_t localExpertOffset, int64_t localNumExperts, int64_t intermediateSize) const
    {
        return runLayerMoeExpertsFromRoutingLogitsImpl(layerIdx, postAttentionGatedScratch, routerLogitsScratch,
            denseMlpOutputScratch, moeSharedIntermediateScratch, moeSharedGateUpScratch, moeSharedOutputScratch,
            inputTokens, topK, nGroup, topkGroup, routedScalingFactor, sfVecSize, allowedBackends, sharedOutputScale,
            globalNumExperts, localExpertOffset, localNumExperts, intermediateSize);
    }

    at::Tensor runLayerDenseMlpWithGateUpScratch(int64_t layerIdx, at::Tensor const& postAttentionGatedScratch,
        at::Tensor const& denseMlpIntermediateScratch, at::Tensor const& denseMlpGateUpScratch,
        at::Tensor const& denseMlpOutputScratch, int64_t inputTokens) const
    {
        TORCH_CHECK(layerIdx >= 0 && layerIdx < mNbLayers, "layer_idx is out of range");
        TORCH_CHECK(
            mLayerKinds.at(static_cast<size_t>(layerIdx)) == static_cast<int64_t>(DeepseekResidentLayerKind::kDense),
            "run_layer_dense_mlp requires a dense layer");
        TORCH_CHECK(postAttentionGatedScratch.is_cuda(), "post_attention_gated_scratch must be a CUDA tensor");
        TORCH_CHECK(denseMlpIntermediateScratch.is_cuda(), "dense_mlp_intermediate_scratch must be a CUDA tensor");
        TORCH_CHECK(denseMlpGateUpScratch.is_cuda(), "dense_mlp_gate_up_scratch must be a CUDA tensor");
        TORCH_CHECK(denseMlpOutputScratch.is_cuda(), "dense_mlp_output_scratch must be a CUDA tensor");
        TORCH_CHECK(inputTokens > 0, "input_tokens must be positive");
        TORCH_CHECK(postAttentionGatedScratch.dim() == 2, "post_attention_gated_scratch must be 2D");
        TORCH_CHECK(denseMlpIntermediateScratch.dim() == 2, "dense_mlp_intermediate_scratch must be 2D");
        TORCH_CHECK(denseMlpGateUpScratch.dim() == 2, "dense_mlp_gate_up_scratch must be 2D");
        TORCH_CHECK(denseMlpOutputScratch.dim() == 2, "dense_mlp_output_scratch must be 2D");
        TORCH_CHECK(postAttentionGatedScratch.size(0) >= inputTokens,
            "post_attention_gated_scratch batch is smaller than input");
        TORCH_CHECK(denseMlpIntermediateScratch.size(0) >= inputTokens,
            "dense_mlp_intermediate_scratch batch is smaller than input");
        TORCH_CHECK(
            denseMlpGateUpScratch.size(0) >= inputTokens, "dense_mlp_gate_up_scratch batch is smaller than input");
        TORCH_CHECK(
            denseMlpOutputScratch.size(0) >= inputTokens, "dense_mlp_output_scratch batch is smaller than input");
        TORCH_CHECK(denseMlpOutputScratch.size(1) == postAttentionGatedScratch.size(1),
            "dense_mlp_output_scratch hidden dim must match post_attention_gated_scratch");
        TORCH_CHECK(denseMlpIntermediateScratch.scalar_type() == postAttentionGatedScratch.scalar_type(),
            "dense_mlp_intermediate_scratch dtype must match post_attention_gated_scratch dtype");
        TORCH_CHECK(
            denseMlpGateUpScratch.scalar_type() == at::ScalarType::Float, "dense_mlp_gate_up_scratch must be float32");
        TORCH_CHECK(denseMlpOutputScratch.scalar_type() == postAttentionGatedScratch.scalar_type(),
            "dense_mlp_output_scratch dtype must match post_attention_gated_scratch dtype");

        at::Tensor const& gateUpWeight = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kDenseMlpGateUpWeight, "dense MLP gate_up_proj weight");
        at::Tensor const& downWeight = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kDenseMlpDownWeight, "dense MLP down_proj weight");
        TORCH_CHECK(gateUpWeight.dim() == 2, "dense MLP gate_up_proj weight must be 2D");
        TORCH_CHECK(downWeight.dim() == 2, "dense MLP down_proj weight must be 2D");

        at::Tensor inputPrefix = postAttentionGatedScratch.narrow(0, 0, inputTokens);
        at::Tensor intermediatePrefix = denseMlpIntermediateScratch.narrow(0, 0, inputTokens);
        at::Tensor outputPrefix = denseMlpOutputScratch.narrow(0, 0, inputTokens);
        std::optional<at::Tensor> const gateUpWeightScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kDenseMlpGateUpWeightScale);
        std::optional<at::Tensor> const gateUpInputScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kDenseMlpGateUpInputScale);
        std::optional<at::Tensor> const gateUpAlpha
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kDenseMlpGateUpAlpha);
        int64_t const gateUpWidth = gateUpWeight.size(0);
        TORCH_CHECK(gateUpWidth % 2 == 0, "dense MLP gate_up output dim must be even");
        TORCH_CHECK(
            denseMlpGateUpScratch.size(1) >= gateUpWidth, "dense_mlp_gate_up_scratch dim must cover gate_up output");
        at::Tensor gateUpPrefix = denseMlpGateUpScratch.narrow(0, 0, inputTokens).narrow(1, 0, gateUpWidth);
        if (!tryRunLinearMaybeNvfp4ToOutput(inputPrefix, gateUpWeight, gateUpWeightScale, gateUpInputScale, gateUpAlpha,
                gateUpPrefix, inputTokens, /*sfVecSize=*/16, "dense_mlp_gate_up_out"))
        {
            at::Tensor gateUp
                = runLinearMaybeNvfp4ToFloat(inputPrefix, gateUpWeight, gateUpWeightScale, gateUpInputScale,
                    gateUpAlpha, inputTokens, /*sfVecSize=*/16, "cutlass,cublaslt,cuda_core", "dense_mlp_gate_up");
            TORCH_CHECK(gateUp.dim() == 2, "dense MLP gate_up output must be 2D");
            TORCH_CHECK(gateUp.size(0) >= inputTokens, "dense MLP gate_up output batch is smaller than input");
            TORCH_CHECK(gateUp.size(1) >= gateUpWidth, "dense MLP gate_up output dim is smaller than expected");
            gateUpPrefix.copy_(gateUp.narrow(0, 0, inputTokens).narrow(1, 0, gateUpWidth));
        }
        int64_t const intermediateSize = gateUpWidth / 2;
        TORCH_CHECK(denseMlpIntermediateScratch.size(1) >= intermediateSize,
            "dense_mlp_intermediate_scratch dim must cover gate_up half dim");
        at::Tensor gate = gateUpPrefix.narrow(1, 0, intermediateSize);
        at::Tensor up = gateUpPrefix.narrow(1, intermediateSize, intermediateSize);
        at::Tensor intermediateActivePrefix = intermediatePrefix.narrow(1, 0, intermediateSize);
        if (!tryRunResidentSwiGluFloatToOutput(gate, up, intermediateActivePrefix, inputTokens))
        {
            at::Tensor activated = (gate * at::sigmoid(gate)) * up;
            intermediateActivePrefix.copy_(activated.to(inputPrefix.scalar_type()));
        }
        std::optional<at::Tensor> const downWeightScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kDenseMlpDownWeightScale);
        std::optional<at::Tensor> const downInputScale
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kDenseMlpDownInputScale);
        std::optional<at::Tensor> const downAlpha
            = getOptionalLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kDenseMlpDownAlpha);
        if (!tryRunLinearMaybeNvfp4ToOutput(intermediateActivePrefix, downWeight, downWeightScale, downInputScale,
                downAlpha, outputPrefix, inputTokens, /*sfVecSize=*/16, "dense_mlp_down_out"))
        {
            at::Tensor downOutput
                = runLinearMaybeNvfp4ToFloat(intermediateActivePrefix, downWeight, downWeightScale, downInputScale,
                    downAlpha, inputTokens, /*sfVecSize=*/16, "cutlass,cublaslt,cuda_core", "dense_mlp_down");
            TORCH_CHECK(downOutput.dim() == 2, "dense MLP down output must be 2D");
            TORCH_CHECK(downOutput.size(0) >= inputTokens, "dense MLP down output batch is smaller than input");
            TORCH_CHECK(
                downOutput.size(1) >= denseMlpOutputScratch.size(1), "dense MLP down output dim must cover hidden dim");
            outputPrefix.copy_(downOutput.narrow(0, 0, inputTokens)
                    .narrow(1, 0, denseMlpOutputScratch.size(1))
                    .to(inputPrefix.scalar_type()));
        }
        return denseMlpOutputScratch;
    }

    at::Tensor runLayerDenseMlp(int64_t layerIdx, at::Tensor const& postAttentionGatedScratch,
        at::Tensor const& denseMlpIntermediateScratch, at::Tensor const& denseMlpOutputScratch,
        int64_t inputTokens) const
    {
        at::Tensor denseMlpGateUpScratch
            = at::empty({denseMlpIntermediateScratch.size(0), denseMlpIntermediateScratch.size(1) * 2},
                denseMlpIntermediateScratch.options().dtype(at::ScalarType::Float));
        return runLayerDenseMlpWithGateUpScratch(layerIdx, postAttentionGatedScratch, denseMlpIntermediateScratch,
            denseMlpGateUpScratch, denseMlpOutputScratch, inputTokens);
    }

    at::Tensor runLayerPostFfnRmsNorm(int64_t layerIdx, at::Tensor const& denseMlpOutputScratch,
        at::Tensor const& residualInputScratch, at::Tensor const& nextLayerHiddenScratch,
        at::Tensor const& nextLayerResidualScratch, int64_t inputTokens, double eps, bool useGemma) const
    {
        TORCH_CHECK(layerIdx >= 0 && layerIdx < mNbLayers, "layer_idx is out of range");
        TORCH_CHECK(denseMlpOutputScratch.is_cuda(), "dense_mlp_output_scratch must be a CUDA tensor");
        TORCH_CHECK(residualInputScratch.is_cuda(), "residual_input_scratch must be a CUDA tensor");
        TORCH_CHECK(nextLayerHiddenScratch.is_cuda(), "next_layer_hidden_scratch must be a CUDA tensor");
        TORCH_CHECK(nextLayerResidualScratch.is_cuda(), "next_layer_residual_scratch must be a CUDA tensor");
        TORCH_CHECK(inputTokens > 0, "input_tokens must be positive");
        TORCH_CHECK(denseMlpOutputScratch.dim() == 2, "dense_mlp_output_scratch must be 2D");
        TORCH_CHECK(residualInputScratch.dim() == 2, "residual_input_scratch must be 2D");
        TORCH_CHECK(nextLayerHiddenScratch.dim() == 2, "next_layer_hidden_scratch must be 2D");
        TORCH_CHECK(nextLayerResidualScratch.dim() == 2, "next_layer_residual_scratch must be 2D");
        TORCH_CHECK(
            denseMlpOutputScratch.size(0) >= inputTokens, "dense_mlp_output_scratch batch is smaller than input");
        TORCH_CHECK(residualInputScratch.size(0) >= inputTokens, "residual_input_scratch batch is smaller than input");
        TORCH_CHECK(
            nextLayerHiddenScratch.size(0) >= inputTokens, "next_layer_hidden_scratch batch is smaller than input");
        TORCH_CHECK(
            nextLayerResidualScratch.size(0) >= inputTokens, "next_layer_residual_scratch batch is smaller than input");
        TORCH_CHECK(denseMlpOutputScratch.size(1) == residualInputScratch.size(1),
            "dense MLP output hidden dim must match residual input hidden dim");
        TORCH_CHECK(nextLayerHiddenScratch.size(1) == denseMlpOutputScratch.size(1),
            "next_layer_hidden_scratch hidden dim must match dense MLP output hidden dim");
        TORCH_CHECK(nextLayerResidualScratch.size(1) == denseMlpOutputScratch.size(1),
            "next_layer_residual_scratch hidden dim must match dense MLP output hidden dim");
        TORCH_CHECK(denseMlpOutputScratch.scalar_type() == residualInputScratch.scalar_type(),
            "dense MLP output and residual input dtypes must match");
        TORCH_CHECK(nextLayerHiddenScratch.scalar_type() == denseMlpOutputScratch.scalar_type(),
            "next_layer_hidden_scratch dtype must match dense MLP output dtype");
        TORCH_CHECK(nextLayerResidualScratch.scalar_type() == denseMlpOutputScratch.scalar_type(),
            "next_layer_residual_scratch dtype must match dense MLP output dtype");

        at::Tensor const& weight = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kNextLayerLayernorm, "next-layer RMSNorm weight");
        TORCH_CHECK(weight.dim() == 1, "next-layer RMSNorm weight must be 1D");
        TORCH_CHECK(weight.size(0) == denseMlpOutputScratch.size(1), "RMSNorm weight size must match hidden dim");
        TORCH_CHECK(weight.scalar_type() == denseMlpOutputScratch.scalar_type(),
            "RMSNorm weight dtype must match dense_mlp_output_scratch dtype");

        if (tryRunResidentAddRmsNorm(denseMlpOutputScratch, residualInputScratch, nextLayerHiddenScratch,
                nextLayerResidualScratch, weight, inputTokens, eps, useGemma))
        {
            return nextLayerHiddenScratch;
        }

        at::Tensor denseMlpPrefix = denseMlpOutputScratch.narrow(0, 0, inputTokens);
        at::Tensor residualInputPrefix = residualInputScratch.narrow(0, 0, inputTokens);
        at::Tensor hiddenPrefix = nextLayerHiddenScratch.narrow(0, 0, inputTokens);
        at::Tensor residualOutPrefix = nextLayerResidualScratch.narrow(0, 0, inputTokens);
        at::Tensor residualFloat
            = denseMlpPrefix.to(at::ScalarType::Float) + residualInputPrefix.to(at::ScalarType::Float);
        residualOutPrefix.copy_(residualFloat.to(denseMlpOutputScratch.scalar_type()));
        at::Tensor variance = residualFloat.pow(2).mean(-1, true);
        at::Tensor normalized = residualFloat * at::rsqrt(variance + eps);
        at::Tensor effectiveWeight = useGemma ? weight + 1 : weight;
        at::Tensor output = normalized.to(denseMlpOutputScratch.scalar_type()) * effectiveWeight;
        hiddenPrefix.copy_(output);
        return nextLayerHiddenScratch;
    }

    at::Tensor runLmHeadLogits(
        at::Tensor const& hiddenStatesScratch, at::Tensor const& logitsScratch, int64_t inputTokens) const
    {
        TORCH_CHECK(hiddenStatesScratch.is_cuda(), "hidden_states_scratch must be a CUDA tensor");
        TORCH_CHECK(logitsScratch.is_cuda(), "logits_scratch must be a CUDA tensor");
        TORCH_CHECK(inputTokens > 0, "input_tokens must be positive");
        TORCH_CHECK(hiddenStatesScratch.dim() == 2, "hidden_states_scratch must be 2D");
        TORCH_CHECK(logitsScratch.dim() == 2, "logits_scratch must be 2D");
        TORCH_CHECK(hiddenStatesScratch.size(0) >= inputTokens, "hidden_states_scratch batch is smaller than input");
        TORCH_CHECK(logitsScratch.size(0) >= inputTokens, "logits_scratch batch is smaller than input");
        TORCH_CHECK(mNbTensors > static_cast<int64_t>(DeepseekResidentTensorSlot::kLmHead), "missing LM head tensor");
        at::Tensor const& lmHead = mResidentTensors.at(static_cast<size_t>(DeepseekResidentTensorSlot::kLmHead));
        TORCH_CHECK(lmHead.dim() == 2, "LM head tensor must be 2D");
        TORCH_CHECK(lmHead.size(1) == hiddenStatesScratch.size(1), "LM head hidden dim must match hidden states");
        TORCH_CHECK(logitsScratch.size(1) == lmHead.size(0), "logits_scratch vocab dim must match LM head");
        TORCH_CHECK(hiddenStatesScratch.scalar_type() == lmHead.scalar_type(),
            "LM head dtype must match hidden_states_scratch dtype");

        at::Tensor hiddenPrefix = hiddenStatesScratch.narrow(0, 0, inputTokens);
        at::Tensor logitsPrefix = logitsScratch.narrow(0, 0, inputTokens);
        at::Tensor logits = at::matmul(hiddenPrefix.to(lmHead.scalar_type()), lmHead.t());
        logitsPrefix.copy_(logits.to(logitsScratch.scalar_type()));
        return logitsScratch;
    }

    at::Tensor runGreedySample(
        at::Tensor const& logitsScratch, at::Tensor const& newTokensScratch, int64_t inputTokens) const
    {
        TORCH_CHECK(logitsScratch.is_cuda(), "logits_scratch must be a CUDA tensor");
        TORCH_CHECK(newTokensScratch.is_cuda(), "new_tokens_scratch must be a CUDA tensor");
        TORCH_CHECK(inputTokens > 0, "input_tokens must be positive");
        TORCH_CHECK(logitsScratch.dim() == 2, "logits_scratch must be 2D");
        TORCH_CHECK(newTokensScratch.dim() == 3, "new_tokens_scratch must be 3D");
        TORCH_CHECK(logitsScratch.size(0) >= inputTokens, "logits_scratch batch is smaller than input");
        TORCH_CHECK(newTokensScratch.size(0) >= 1, "new_tokens_scratch must have at least one token step");
        TORCH_CHECK(newTokensScratch.size(1) >= inputTokens, "new_tokens_scratch batch is smaller than input");
        TORCH_CHECK(newTokensScratch.size(2) >= 1, "new_tokens_scratch must have at least one beam");
        TORCH_CHECK(newTokensScratch.scalar_type() == at::ScalarType::Int
                || newTokensScratch.scalar_type() == at::ScalarType::Long,
            "new_tokens_scratch must be int32 or int64");

        at::Tensor logitsPrefix = validTokenLogitsPrefix(logitsScratch, inputTokens);
        at::Tensor tokenIds = std::get<1>(logitsPrefix.max(/*dim=*/1, /*keepdim=*/false));
        at::Tensor tokenOut = newTokensScratch.select(0, 0).narrow(0, 0, inputTokens).select(1, 0);
        tokenOut.copy_(tokenIds.to(newTokensScratch.scalar_type()));
        return newTokensScratch;
    }

    at::Tensor runDecodeWindowAttentionMetadataDeviceRefresh(at::Tensor const& seqLensCuda,
        at::Tensor const& kvLensCuda, at::Tensor const& reqIdxPerToken, at::Tensor const& indexerKCacheBlockOffsets,
        at::Tensor const& slotMappingFp8, at::Tensor const& slotMappingScale, at::Tensor const& genKvIndptr,
        at::Tensor const& genCachedTokenIndptr, at::Tensor const& kvLensCuda2d, int64_t numTokens, int64_t numSeqs,
        int64_t numContexts, int64_t numGenerations, int64_t headDim, int64_t tokensPerBlock, int64_t quantBlockSize,
        int64_t dataBytesPerToken) const
    {
        TORCH_CHECK(seqLensCuda.is_cuda(), "seq_lens_cuda must be a CUDA tensor");
        TORCH_CHECK(kvLensCuda.is_cuda(), "kv_lens_cuda must be a CUDA tensor");
        TORCH_CHECK(reqIdxPerToken.is_cuda(), "req_idx_per_token must be a CUDA tensor");
        TORCH_CHECK(indexerKCacheBlockOffsets.is_cuda(), "indexer_k_cache_block_offsets must be a CUDA tensor");
        TORCH_CHECK(slotMappingFp8.is_cuda(), "slot_mapping_fp8 must be a CUDA tensor");
        TORCH_CHECK(slotMappingScale.is_cuda(), "slot_mapping_scale must be a CUDA tensor");
        TORCH_CHECK(numTokens > 0, "num_tokens must be positive");
        TORCH_CHECK(numSeqs > 0, "num_seqs must be positive");
        TORCH_CHECK(numContexts >= 0, "num_contexts must be non-negative");
        TORCH_CHECK(numGenerations >= 0, "num_generations must be non-negative");
        TORCH_CHECK(numContexts + numGenerations <= numSeqs, "context/generation counts exceed num_seqs");
        TORCH_CHECK(headDim > 0, "head_dim must be positive");
        TORCH_CHECK(tokensPerBlock > 0, "tokens_per_block must be positive");
        TORCH_CHECK(quantBlockSize > 0, "quant_block_size must be positive");
        TORCH_CHECK(dataBytesPerToken > 0, "data_bytes_per_token must be positive");
        TORCH_CHECK(seqLensCuda.dim() == 1, "seq_lens_cuda must be 1D");
        TORCH_CHECK(kvLensCuda.dim() == 1, "kv_lens_cuda must be 1D");
        TORCH_CHECK(reqIdxPerToken.dim() == 1, "req_idx_per_token must be 1D");
        TORCH_CHECK(indexerKCacheBlockOffsets.dim() == 2, "indexer_k_cache_block_offsets must be 2D");
        TORCH_CHECK(slotMappingFp8.dim() == 1, "slot_mapping_fp8 must be 1D");
        TORCH_CHECK(slotMappingScale.dim() == 1, "slot_mapping_scale must be 1D");
        TORCH_CHECK(seqLensCuda.numel() >= numSeqs, "seq_lens_cuda is shorter than num_seqs");
        TORCH_CHECK(kvLensCuda.numel() >= numSeqs, "kv_lens_cuda is shorter than num_seqs");
        TORCH_CHECK(reqIdxPerToken.numel() >= numTokens, "req_idx_per_token is shorter than num_tokens");
        TORCH_CHECK(
            indexerKCacheBlockOffsets.size(0) >= numSeqs, "indexer_k_cache_block_offsets has fewer rows than num_seqs");
        TORCH_CHECK(indexerKCacheBlockOffsets.size(1) > 0, "indexer_k_cache_block_offsets has no blocks");
        TORCH_CHECK(slotMappingFp8.numel() >= numTokens, "slot_mapping_fp8 is shorter than num_tokens");
        TORCH_CHECK(slotMappingScale.numel() >= numTokens, "slot_mapping_scale is shorter than num_tokens");
        TORCH_CHECK(slotMappingFp8.scalar_type() == at::ScalarType::Long, "slot_mapping_fp8 must be int64");
        TORCH_CHECK(slotMappingScale.scalar_type() == at::ScalarType::Long, "slot_mapping_scale must be int64");
        TORCH_CHECK(headDim % quantBlockSize == 0, "head_dim must be divisible by quant_block_size");

        int64_t const maxBlocks = indexerKCacheBlockOffsets.size(1);
        bool const nativeMetadataRefreshAvailable = seqLensCuda.scalar_type() == at::ScalarType::Int
            && kvLensCuda.scalar_type() == at::ScalarType::Int && reqIdxPerToken.scalar_type() == at::ScalarType::Int
            && indexerKCacheBlockOffsets.scalar_type() == at::ScalarType::Int
            && (numGenerations == 0
                || (genKvIndptr.is_cuda() && genCachedTokenIndptr.is_cuda() && kvLensCuda2d.is_cuda()
                    && genKvIndptr.dim() == 1 && genCachedTokenIndptr.dim() == 1 && kvLensCuda2d.dim() == 2
                    && genKvIndptr.scalar_type() == at::ScalarType::Long
                    && genCachedTokenIndptr.scalar_type() == at::ScalarType::Long
                    && kvLensCuda2d.scalar_type() == at::ScalarType::Int && genKvIndptr.numel() >= numGenerations + 1
                    && genCachedTokenIndptr.numel() >= numGenerations + 1 && kvLensCuda2d.size(0) >= numGenerations));

        if (nativeMetadataRefreshAvailable)
        {
            TORCH_CHECK(numTokens <= std::numeric_limits<int32_t>::max(), "num_tokens exceeds int32 range");
            TORCH_CHECK(numSeqs <= std::numeric_limits<int32_t>::max(), "num_seqs exceeds int32 range");
            TORCH_CHECK(numContexts <= std::numeric_limits<int32_t>::max(), "num_contexts exceeds int32 range");
            TORCH_CHECK(numGenerations <= std::numeric_limits<int32_t>::max(), "num_generations exceeds int32 range");
            TORCH_CHECK(maxBlocks <= std::numeric_limits<int32_t>::max(), "max block count exceeds int32 range");
            TORCH_CHECK(headDim <= std::numeric_limits<int32_t>::max(), "head_dim exceeds int32 range");
            TORCH_CHECK(tokensPerBlock <= std::numeric_limits<int32_t>::max(), "tokens_per_block exceeds int32 range");
            TORCH_CHECK(quantBlockSize <= std::numeric_limits<int32_t>::max(), "quant_block_size exceeds int32 range");
            TORCH_CHECK(
                dataBytesPerToken <= std::numeric_limits<int32_t>::max(), "data_bytes_per_token exceeds int32 range");

            int64_t const nextNCap = numGenerations > 0 ? kvLensCuda2d.size(1) : 0;
            TORCH_CHECK(nextNCap <= std::numeric_limits<int32_t>::max(), "next-n capacity exceeds int32 range");
            auto stream = at::cuda::getCurrentCUDAStream(seqLensCuda.get_device());
            tk::invokeDeepseekResidentAttentionMetadataRefresh(seqLensCuda.data_ptr<int32_t>(), seqLensCuda.stride(0),
                kvLensCuda.data_ptr<int32_t>(), kvLensCuda.stride(0), reqIdxPerToken.data_ptr<int32_t>(),
                reqIdxPerToken.stride(0), indexerKCacheBlockOffsets.data_ptr<int32_t>(),
                indexerKCacheBlockOffsets.stride(0), indexerKCacheBlockOffsets.stride(1),
                slotMappingFp8.data_ptr<int64_t>(), slotMappingFp8.stride(0), slotMappingScale.data_ptr<int64_t>(),
                slotMappingScale.stride(0), numGenerations > 0 ? genKvIndptr.data_ptr<int64_t>() : nullptr,
                numGenerations > 0 ? genKvIndptr.stride(0) : 1,
                numGenerations > 0 ? genCachedTokenIndptr.data_ptr<int64_t>() : nullptr,
                numGenerations > 0 ? genCachedTokenIndptr.stride(0) : 1,
                numGenerations > 0 ? kvLensCuda2d.data_ptr<int32_t>() : nullptr,
                numGenerations > 0 ? kvLensCuda2d.stride(0) : 1, numGenerations > 0 ? kvLensCuda2d.stride(1) : 1,
                static_cast<int32_t>(numTokens), static_cast<int32_t>(numSeqs), static_cast<int32_t>(numContexts),
                static_cast<int32_t>(numGenerations), static_cast<int32_t>(maxBlocks), static_cast<int32_t>(nextNCap),
                static_cast<int32_t>(headDim), static_cast<int32_t>(tokensPerBlock),
                static_cast<int32_t>(quantBlockSize), static_cast<int32_t>(dataBytesPerToken), stream);
            return slotMappingFp8;
        }

        at::Tensor seqLens = seqLensCuda.narrow(0, 0, numSeqs).to(at::ScalarType::Long);
        at::Tensor kvLens = kvLensCuda.narrow(0, 0, numSeqs).to(at::ScalarType::Long);
        at::Tensor startPositions = kvLens - seqLens;
        at::Tensor reqIndices = reqIdxPerToken.narrow(0, 0, numTokens).to(at::ScalarType::Long);
        at::Tensor seqStarts = at::cumsum(seqLens, /*dim=*/0, at::ScalarType::Long) - seqLens;
        at::Tensor tokenOffsets = at::arange(numTokens, seqLens.options()) - seqStarts.index_select(0, reqIndices);
        at::Tensor globalPositions = startPositions.index_select(0, reqIndices) + tokenOffsets;
        at::Tensor blockIndicesInSeq = at::floor_divide(globalPositions, tokensPerBlock);
        blockIndicesInSeq = at::clamp(blockIndicesInSeq, 0, maxBlocks - 1);
        at::Tensor posInBlocks = globalPositions.remainder(tokensPerBlock);
        at::Tensor flatBlockOffsets = indexerKCacheBlockOffsets.narrow(0, 0, numSeqs).reshape({-1});
        at::Tensor flatBlockIndices = reqIndices * maxBlocks + blockIndicesInSeq;
        at::Tensor blockIds = flatBlockOffsets.index_select(0, flatBlockIndices).to(at::ScalarType::Long);

        int64_t const scaleSize = headDim / quantBlockSize * 4;
        TORCH_CHECK(scaleSize > 0, "computed scale size must be positive");
        int64_t const blockStride = tokensPerBlock * (dataBytesPerToken + scaleSize);
        int64_t const scaleBaseOffset = tokensPerBlock * dataBytesPerToken;
        at::Tensor fp8Indices = blockIds * blockStride + posInBlocks * dataBytesPerToken;
        at::Tensor scaleIndices = blockIds * blockStride + scaleBaseOffset + posInBlocks * scaleSize;
        slotMappingFp8.narrow(0, 0, numTokens).copy_(fp8Indices);
        slotMappingScale.narrow(0, 0, numTokens).copy_(scaleIndices);

        if (numGenerations > 0)
        {
            TORCH_CHECK(genKvIndptr.is_cuda(), "gen_kv_indptr must be a CUDA tensor when generations are present");
            TORCH_CHECK(genCachedTokenIndptr.is_cuda(),
                "gen_cached_token_indptr must be a CUDA tensor when generations are present");
            TORCH_CHECK(kvLensCuda2d.is_cuda(), "kv_lens_cuda_2d must be a CUDA tensor when generations are present");
            TORCH_CHECK(genKvIndptr.dim() == 1, "gen_kv_indptr must be 1D");
            TORCH_CHECK(genCachedTokenIndptr.dim() == 1, "gen_cached_token_indptr must be 1D");
            TORCH_CHECK(kvLensCuda2d.dim() == 2, "kv_lens_cuda_2d must be 2D");
            TORCH_CHECK(genKvIndptr.numel() >= numGenerations + 1, "gen_kv_indptr is shorter than num_generations + 1");
            TORCH_CHECK(genCachedTokenIndptr.numel() >= numGenerations + 1,
                "gen_cached_token_indptr is shorter than num_generations + 1");
            TORCH_CHECK(kvLensCuda2d.size(0) >= numGenerations, "kv_lens_cuda_2d has too few rows");
            at::Tensor genKvLens = kvLensCuda.narrow(0, numContexts, numGenerations).to(at::ScalarType::Long);
            at::Tensor genSeqLens = seqLensCuda.narrow(0, numContexts, numGenerations).to(at::ScalarType::Long);
            at::Tensor genCachedLens = genKvLens - genSeqLens;
            genKvIndptr.narrow(0, 0, 1).zero_();
            genCachedTokenIndptr.narrow(0, 0, 1).zero_();
            genKvIndptr.narrow(0, 1, numGenerations).copy_(at::cumsum(genKvLens, /*dim=*/0, at::ScalarType::Long));
            genCachedTokenIndptr.narrow(0, 1, numGenerations)
                .copy_(at::cumsum(genCachedLens, /*dim=*/0, at::ScalarType::Long));
            int64_t const nextNCap = kvLensCuda2d.size(1);
            kvLensCuda2d.narrow(0, 0, numGenerations)
                .narrow(1, 0, nextNCap)
                .copy_(genKvLens.to(kvLensCuda2d.scalar_type()).unsqueeze(1).expand({numGenerations, nextNCap}));
        }
        return slotMappingFp8;
    }

    at::Tensor runIndexerKCacheScatter(at::Tensor const& kFp8, at::Tensor const& kScale, at::Tensor const& kCache,
        at::Tensor const& slotMappingFp8, at::Tensor const& slotMappingScale, int64_t inputTokens) const
    {
        TORCH_CHECK(kFp8.is_cuda(), "k_fp8 must be a CUDA tensor");
        TORCH_CHECK(kScale.is_cuda(), "k_scale must be a CUDA tensor");
        TORCH_CHECK(kCache.is_cuda(), "k_cache must be a CUDA tensor");
        TORCH_CHECK(slotMappingFp8.is_cuda(), "slot_mapping_fp8 must be a CUDA tensor");
        TORCH_CHECK(slotMappingScale.is_cuda(), "slot_mapping_scale must be a CUDA tensor");
        TORCH_CHECK(inputTokens > 0, "input_tokens must be positive");
        TORCH_CHECK(kFp8.dim() == 2, "k_fp8 must be 2D");
        TORCH_CHECK(kScale.dim() == 2, "k_scale must be 2D");
        TORCH_CHECK(kCache.dim() == 4, "k_cache must be 4D");
        TORCH_CHECK(slotMappingFp8.dim() == 1, "slot_mapping_fp8 must be 1D");
        TORCH_CHECK(slotMappingScale.dim() == 1, "slot_mapping_scale must be 1D");
        TORCH_CHECK(kFp8.size(0) >= inputTokens, "k_fp8 batch is smaller than input_tokens");
        TORCH_CHECK(kScale.size(0) >= inputTokens, "k_scale batch is smaller than input_tokens");
        TORCH_CHECK(slotMappingFp8.numel() >= inputTokens, "slot_mapping_fp8 is shorter than input_tokens");
        TORCH_CHECK(slotMappingScale.numel() >= inputTokens, "slot_mapping_scale is shorter than input_tokens");
        TORCH_CHECK(kFp8.element_size() == 1, "k_fp8 must have 1-byte elements");
        TORCH_CHECK(kScale.element_size() == 4, "k_scale must have 4-byte elements");
        TORCH_CHECK(kCache.scalar_type() == at::ScalarType::Byte, "k_cache must be uint8");
        TORCH_CHECK(slotMappingFp8.scalar_type() == at::ScalarType::Long, "slot_mapping_fp8 must be int64");
        TORCH_CHECK(slotMappingScale.scalar_type() == at::ScalarType::Long, "slot_mapping_scale must be int64");
        TORCH_CHECK(kFp8.is_contiguous(), "k_fp8 must be contiguous");
        TORCH_CHECK(kScale.is_contiguous(), "k_scale must be contiguous");
        TORCH_CHECK(slotMappingFp8.is_contiguous(), "slot_mapping_fp8 must be contiguous");
        TORCH_CHECK(slotMappingScale.is_contiguous(), "slot_mapping_scale must be contiguous");

        int32_t const headDim = static_cast<int32_t>(kFp8.size(1));
        int32_t const scaleSize = static_cast<int32_t>(kScale.size(1)) * static_cast<int32_t>(kScale.element_size());
        int32_t const cacheDim0 = static_cast<int32_t>(kCache.size(0));
        int32_t const cacheDim1 = static_cast<int32_t>(kCache.size(1));
        int32_t const cacheDim2 = static_cast<int32_t>(kCache.size(2));
        int32_t const cacheDim3 = static_cast<int32_t>(kCache.size(3));
        TORCH_CHECK(cacheDim2 == 1, "k_cache dimension 2 must be 1");
        TORCH_CHECK(headDim == 128 || headDim == 64, "k_fp8 head_dim must be 128 or 64");
        TORCH_CHECK(scaleSize == 4, "k_scale must encode 4 scale bytes per token");

        auto stream = at::cuda::getCurrentCUDAStream(kFp8.get_device());
        tk::invokeIndexerKCacheScatter(reinterpret_cast<uint8_t const*>(kFp8.data_ptr()),
            reinterpret_cast<uint8_t const*>(kScale.data_ptr()), kCache.data_ptr<uint8_t>(),
            slotMappingFp8.data_ptr<int64_t>(), slotMappingScale.data_ptr<int64_t>(), static_cast<int32_t>(inputTokens),
            headDim, scaleSize, cacheDim0, cacheDim1, cacheDim2, cacheDim3, static_cast<int64_t>(kCache.stride(0)),
            static_cast<int64_t>(kCache.stride(1)), static_cast<int64_t>(kCache.stride(2)),
            static_cast<int64_t>(kCache.stride(3)), stream);
        return kCache;
    }

    at::Tensor runIndexerDenseTopkDecode(at::Tensor const& kvLensCuda, at::Tensor const& topkIndicesScratch,
        int64_t inputTokens, int64_t indexTopk) const
    {
        TORCH_CHECK(kvLensCuda.is_cuda(), "kv_lens_cuda must be a CUDA tensor");
        TORCH_CHECK(topkIndicesScratch.is_cuda(), "topk_indices_scratch must be a CUDA tensor");
        TORCH_CHECK(inputTokens > 0, "input_tokens must be positive");
        TORCH_CHECK(indexTopk > 0, "index_topk must be positive");
        TORCH_CHECK(kvLensCuda.dim() == 1, "kv_lens_cuda must be 1D");
        TORCH_CHECK(topkIndicesScratch.dim() == 2, "topk_indices_scratch must be 2D");
        TORCH_CHECK(kvLensCuda.scalar_type() == at::ScalarType::Int, "kv_lens_cuda must be int32");
        TORCH_CHECK(topkIndicesScratch.scalar_type() == at::ScalarType::Int, "topk_indices_scratch must be int32");
        TORCH_CHECK(kvLensCuda.size(0) >= inputTokens, "kv_lens_cuda has fewer rows than input");
        TORCH_CHECK(topkIndicesScratch.size(0) >= inputTokens, "topk_indices_scratch has fewer rows than input");
        TORCH_CHECK(topkIndicesScratch.size(1) >= indexTopk, "topk_indices_scratch has fewer columns than index_topk");
        TORCH_CHECK(kvLensCuda.is_contiguous(), "kv_lens_cuda must be contiguous");
        TORCH_CHECK(topkIndicesScratch.is_contiguous(), "topk_indices_scratch must be contiguous");
        TORCH_CHECK(kvLensCuda.get_device() == topkIndicesScratch.get_device(),
            "kv_lens_cuda and topk_indices_scratch must be on the same device");

        auto stream = at::cuda::getCurrentCUDAStream(topkIndicesScratch.get_device());
        tk::invokeDeepseekResidentDenseTopkDecode(kvLensCuda.data_ptr<int32_t>(),
            static_cast<int64_t>(kvLensCuda.stride(0)), topkIndicesScratch.data_ptr<int32_t>(),
            static_cast<int64_t>(topkIndicesScratch.stride(0)), static_cast<int64_t>(topkIndicesScratch.stride(1)),
            static_cast<int32_t>(inputTokens), static_cast<int32_t>(indexTopk), stream);
        return topkIndicesScratch;
    }

    at::Tensor runIndexerTopkDecode(at::Tensor const& logits, at::Tensor const& kvLensCuda,
        at::Tensor const& topkIndicesScratch, int64_t nextN, int64_t indexTopk) const
    {
        TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
        TORCH_CHECK(kvLensCuda.is_cuda(), "kv_lens_cuda must be a CUDA tensor");
        TORCH_CHECK(topkIndicesScratch.is_cuda(), "topk_indices_scratch must be a CUDA tensor");
        TORCH_CHECK(nextN > 0, "next_n must be positive");
        TORCH_CHECK(indexTopk > 0, "index_topk must be positive");
        TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
        TORCH_CHECK(kvLensCuda.dim() == 1, "kv_lens_cuda must be 1D");
        TORCH_CHECK(topkIndicesScratch.dim() == 2, "topk_indices_scratch must be 2D");
        TORCH_CHECK(logits.scalar_type() == at::ScalarType::Float || logits.scalar_type() == at::ScalarType::BFloat16
                || logits.scalar_type() == at::ScalarType::Half,
            "logits dtype must be float32, bfloat16, or float16");
        TORCH_CHECK(kvLensCuda.scalar_type() == at::ScalarType::Int, "kv_lens_cuda must be int32");
        TORCH_CHECK(topkIndicesScratch.scalar_type() == at::ScalarType::Int, "topk_indices_scratch must be int32");
        TORCH_CHECK(logits.size(0) > 0, "logits must have at least one row");
        TORCH_CHECK(kvLensCuda.size(0) * nextN == logits.size(0),
            "kv_lens_cuda length multiplied by next_n must equal logits rows");
        TORCH_CHECK(topkIndicesScratch.size(0) >= logits.size(0), "topk_indices_scratch has fewer rows than logits");
        TORCH_CHECK(topkIndicesScratch.size(1) >= indexTopk, "topk_indices_scratch has fewer columns than index_topk");
        TORCH_CHECK(kvLensCuda.is_contiguous(), "kv_lens_cuda must be contiguous");
        TORCH_CHECK(topkIndicesScratch.is_contiguous(), "topk_indices_scratch must be contiguous");
        TORCH_CHECK(logits.stride(0) >= 0 && logits.stride(1) >= 0, "logits strides must be non-negative");

        int32_t const splitWorkThreshold = 200 * 1000;
        int32_t const numRows = static_cast<int32_t>(logits.size(0));
        int32_t const numColumns = static_cast<int32_t>(logits.size(1));
        int32_t const stride0 = static_cast<int32_t>(logits.stride(0));
        int32_t const stride1 = static_cast<int32_t>(logits.stride(1));
        auto stream = at::cuda::getCurrentCUDAStream(logits.get_device());

        if (logits.scalar_type() == at::ScalarType::Float)
        {
            at::Tensor auxIndices = at::empty({0}, logits.options().dtype(at::ScalarType::Int));
            at::Tensor auxLogits = at::empty({0}, logits.options().dtype(at::ScalarType::Float));
            constexpr int64_t multipleBlocksPerRowConfig = 10;
            if (numColumns >= splitWorkThreshold)
            {
                auxIndices = at::empty(
                    {numRows, multipleBlocksPerRowConfig, indexTopk}, logits.options().dtype(at::ScalarType::Int));
                auxLogits = at::empty(
                    {numRows, multipleBlocksPerRowConfig, indexTopk}, logits.options().dtype(at::ScalarType::Float));
            }
            tk::invokeIndexerTopKDecode(logits.data_ptr<float>(), kvLensCuda.data_ptr<int32_t>(),
                topkIndicesScratch.data_ptr<int32_t>(), auxLogits.data_ptr<float>(), auxIndices.data_ptr<int32_t>(),
                splitWorkThreshold, numRows, numColumns, stride0, stride1, static_cast<int32_t>(nextN),
                static_cast<int32_t>(indexTopk), nullptr, 0, 0, nullptr, stream);
        }
        else if (logits.scalar_type() == at::ScalarType::BFloat16)
        {
            tk::invokeIndexerTopKDecode(reinterpret_cast<__nv_bfloat16 const*>(logits.data_ptr()),
                kvLensCuda.data_ptr<int32_t>(), topkIndicesScratch.data_ptr<int32_t>(), splitWorkThreshold, numRows,
                numColumns, stride0, stride1, static_cast<int32_t>(nextN), static_cast<int32_t>(indexTopk), nullptr, 0,
                0, nullptr, stream);
        }
        else
        {
            tk::invokeIndexerTopKDecode(reinterpret_cast<__half const*>(logits.data_ptr()),
                kvLensCuda.data_ptr<int32_t>(), topkIndicesScratch.data_ptr<int32_t>(), splitWorkThreshold, numRows,
                numColumns, stride0, stride1, static_cast<int32_t>(nextN), static_cast<int32_t>(indexTopk), nullptr, 0,
                0, nullptr, stream);
        }
        return topkIndicesScratch;
    }

    at::Tensor runIndexerXstepRecencyPatch(at::Tensor const& cachedTopk, at::Tensor const& refreshEnd,
        at::Tensor const& curKvLens, int64_t nextN, int64_t maxDelta) const
    {
        TORCH_CHECK(cachedTopk.is_cuda(), "cached_topk must be a CUDA tensor");
        TORCH_CHECK(refreshEnd.is_cuda(), "refresh_end must be a CUDA tensor");
        TORCH_CHECK(curKvLens.is_cuda(), "cur_kv_lens must be a CUDA tensor");
        TORCH_CHECK(cachedTopk.dim() == 2, "cached_topk must be 2D");
        TORCH_CHECK(refreshEnd.dim() == 1, "refresh_end must be 1D");
        TORCH_CHECK(curKvLens.dim() == 1, "cur_kv_lens must be 1D");
        TORCH_CHECK(cachedTopk.scalar_type() == at::ScalarType::Int, "cached_topk must be int32");
        TORCH_CHECK(refreshEnd.scalar_type() == at::ScalarType::Int, "refresh_end must be int32");
        TORCH_CHECK(curKvLens.scalar_type() == at::ScalarType::Int, "cur_kv_lens must be int32");
        TORCH_CHECK(cachedTopk.is_contiguous(), "cached_topk must be contiguous");
        TORCH_CHECK(refreshEnd.is_contiguous(), "refresh_end must be contiguous");
        TORCH_CHECK(curKvLens.is_contiguous(), "cur_kv_lens must be contiguous");
        TORCH_CHECK(nextN > 0, "next_n must be positive");
        TORCH_CHECK(maxDelta >= 0, "max_delta must be non-negative");
        TORCH_CHECK(refreshEnd.size(0) >= cachedTopk.size(0), "refresh_end is shorter than cached_topk rows");
        TORCH_CHECK(curKvLens.size(0) * nextN >= cachedTopk.size(0),
            "cur_kv_lens length multiplied by next_n must cover cached_topk rows");
        TORCH_CHECK(maxDelta <= cachedTopk.size(1), "max_delta exceeds cached_topk width");

        auto stream = at::cuda::getCurrentCUDAStream(cachedTopk.get_device());
        tk::invokeIndexerXstepRecencyPatch(cachedTopk.data_ptr<int32_t>(), refreshEnd.data_ptr<int32_t>(),
            curKvLens.data_ptr<int32_t>(), static_cast<int32_t>(cachedTopk.size(0)),
            static_cast<int32_t>(cachedTopk.size(1)), static_cast<int32_t>(nextN), static_cast<int32_t>(maxDelta),
            stream);
        return cachedTopk;
    }

    bool runDecodeWindowReady() const
    {
        return false;
    }

    std::string runDecodeWindowNotReadyReason() const
    {
        if (mNbLayers <= 0)
        {
            return "resident_window_native_no_layers";
        }
        if (!runLayerDsaAttentionDispatchReady())
        {
            return "resident_window_native_missing_dsa_attention_dispatch";
        }
        return "resident_window_native_missing_window_execution_plan";
    }

    bool runLayerDsaAttentionDispatchReady() const
    {
        return envFlagEnabled("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_DSA_NATIVE_BODY");
    }

    std::string runLayerDsaAttentionDispatchNotReadyReason() const
    {
        if (!envFlagEnabled("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_DSA_NATIVE_BODY"))
        {
            return "resident_attention_dsa_dispatch_native_body_disabled";
        }
        return "resident_attention_dsa_dispatch_native_not_implemented";
    }

    at::Tensor runLayerDsaAttentionDispatch(int64_t layerIdx, at::Tensor const& q, at::Tensor const& compressedKv,
        at::Tensor const& kPe, at::Tensor const& latentCache, c10::List<at::Tensor> indexerIntermediates,
        at::Tensor const& positionIds, at::Tensor const& seqLensCuda, at::Tensor const& kvLensCuda,
        c10::List<at::Tensor> dsaDispatchMetadataTensors, c10::Dict<std::string, at::Tensor> dsaDispatchRuntimeTensors,
        c10::Dict<std::string, int64_t> dsaDispatchRuntimeConfig,
        c10::Dict<std::string, double> dsaDispatchRuntimeScalars, c10::List<at::Tensor> dsaDispatchScratchTensors,
        at::Tensor const& attentionCoreOutputScratch, int64_t inputTokens) const
    {
        TORCH_CHECK(layerIdx >= 0 && layerIdx < mNbLayers, "layer_idx is out of range");
        TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
        TORCH_CHECK(compressedKv.is_cuda(), "compressed_kv must be a CUDA tensor");
        TORCH_CHECK(kPe.is_cuda(), "k_pe must be a CUDA tensor");
        TORCH_CHECK(latentCache.is_cuda(), "latent_cache must be a CUDA tensor");
        TORCH_CHECK(positionIds.is_cuda(), "position_ids must be a CUDA tensor");
        TORCH_CHECK(seqLensCuda.is_cuda(), "seq_lens_cuda must be a CUDA tensor");
        TORCH_CHECK(kvLensCuda.is_cuda(), "kv_lens_cuda must be a CUDA tensor");
        TORCH_CHECK(attentionCoreOutputScratch.is_cuda(), "attention_core_output_scratch must be a CUDA tensor");
        TORCH_CHECK(inputTokens > 0, "input_tokens must be positive");
        TORCH_CHECK(q.dim() == 2, "q must be 2D");
        TORCH_CHECK(compressedKv.dim() == 2, "compressed_kv must be 2D");
        TORCH_CHECK(kPe.dim() == 2, "k_pe must be 2D");
        TORCH_CHECK(latentCache.dim() == 2, "latent_cache must be 2D");
        TORCH_CHECK(positionIds.dim() >= 1 && positionIds.dim() <= 3, "position_ids must be 1D, 2D, or 3D");
        TORCH_CHECK(seqLensCuda.dim() == 1, "seq_lens_cuda must be 1D");
        TORCH_CHECK(kvLensCuda.dim() == 1, "kv_lens_cuda must be 1D");
        TORCH_CHECK(attentionCoreOutputScratch.dim() == 2, "attention_core_output_scratch must be 2D");
        TORCH_CHECK(q.size(0) >= inputTokens, "q batch is smaller than input");
        TORCH_CHECK(compressedKv.size(0) >= inputTokens, "compressed_kv batch is smaller than input");
        TORCH_CHECK(kPe.size(0) >= inputTokens, "k_pe batch is smaller than input");
        TORCH_CHECK(latentCache.size(0) >= inputTokens, "latent_cache batch is smaller than input");
        TORCH_CHECK(positionIds.numel() >= inputTokens, "position_ids is shorter than input_tokens");
        TORCH_CHECK(seqLensCuda.numel() >= inputTokens, "seq_lens_cuda is shorter than input_tokens");
        TORCH_CHECK(kvLensCuda.numel() >= inputTokens, "kv_lens_cuda is shorter than input_tokens");
        TORCH_CHECK(attentionCoreOutputScratch.size(0) >= inputTokens,
            "attention_core_output_scratch batch is smaller than input");
        at::Tensor const& kBProjTrans = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kAttentionKBProjTrans, "attention k_b_proj_trans");
        at::Tensor const& vBProj
            = getLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kAttentionVBProj, "attention v_b_proj");
        TORCH_CHECK(kBProjTrans.is_cuda(), "attention k_b_proj_trans must be a CUDA tensor");
        TORCH_CHECK(vBProj.is_cuda(), "attention v_b_proj must be a CUDA tensor");
        TORCH_CHECK(kBProjTrans.dim() == 3, "attention k_b_proj_trans must be 3D");
        TORCH_CHECK(vBProj.dim() == 3, "attention v_b_proj must be 3D");
        TORCH_CHECK(kBProjTrans.size(0) > 0 && kBProjTrans.size(1) > 0 && kBProjTrans.size(2) > 0,
            "attention k_b_proj_trans must not be empty");
        TORCH_CHECK(
            vBProj.size(0) > 0 && vBProj.size(1) > 0 && vBProj.size(2) > 0, "attention v_b_proj must not be empty");
        TORCH_CHECK(vBProj.size(0) == kBProjTrans.size(0), "attention k_b/v_b projection head counts must match");
        TORCH_CHECK(vBProj.size(2) == kBProjTrans.size(1), "attention v_b kv_lora_rank must match k_b output rank");
        for (size_t idx = 0; idx < indexerIntermediates.size(); ++idx)
        {
            auto const tensor = indexerIntermediates.get(idx);
            TORCH_CHECK(tensor.is_cuda(), "indexer_intermediates must all be CUDA tensors");
            TORCH_CHECK(tensor.dim() >= 1, "indexer_intermediates entries must have at least one dimension");
            TORCH_CHECK(tensor.size(0) >= inputTokens, "indexer_intermediates batch is smaller than input");
        }
        TORCH_CHECK(!dsaDispatchMetadataTensors.empty(), "dsa_dispatch_metadata_tensors must not be empty");
        for (size_t idx = 0; idx < dsaDispatchMetadataTensors.size(); ++idx)
        {
            auto const tensor = dsaDispatchMetadataTensors.get(idx);
            TORCH_CHECK(tensor.is_cuda(), "dsa_dispatch_metadata_tensors must all be CUDA tensors");
            TORCH_CHECK(tensor.dim() >= 1, "dsa_dispatch_metadata_tensors entries must have at least one dimension");
            TORCH_CHECK(tensor.numel() > 0, "dsa_dispatch_metadata_tensors entries must not be empty");
        }
        at::Tensor const topkIndices = dsaDispatchMetadataTensors.get(0);
        TORCH_CHECK(topkIndices.scalar_type() == at::ScalarType::Int, "dsa dispatch topk_indices must be int32");
        TORCH_CHECK(topkIndices.dim() == 2, "dsa dispatch topk_indices must be 2D");
        TORCH_CHECK(topkIndices.size(0) >= inputTokens, "dsa dispatch topk_indices batch is smaller than input");
        TORCH_CHECK(topkIndices.size(1) > 0, "dsa dispatch topk_indices width must be positive");
        TORCH_CHECK(dsaDispatchScratchTensors.size() >= 5,
            "dsa_dispatch_scratch_tensors must contain fused_q, latent_output, cu_q_seqlens, cu_kv_seqlens, and "
            "fmha_scheduler_counter");
        for (size_t idx = 0; idx < dsaDispatchScratchTensors.size(); ++idx)
        {
            auto const tensor = dsaDispatchScratchTensors.get(idx);
            TORCH_CHECK(tensor.is_cuda(), "dsa_dispatch_scratch_tensors must all be CUDA tensors");
            TORCH_CHECK(tensor.numel() > 0, "dsa_dispatch_scratch_tensors entries must not be empty");
        }
        at::Tensor const fusedQScratch = dsaDispatchScratchTensors.get(0);
        at::Tensor const latentOutputScratch = dsaDispatchScratchTensors.get(1);
        at::Tensor const cuQSeqLensScratch = dsaDispatchScratchTensors.get(2);
        at::Tensor const cuKvSeqLensScratch = dsaDispatchScratchTensors.get(3);
        at::Tensor const fmhaSchedulerCounterScratch = dsaDispatchScratchTensors.get(4);
        TORCH_CHECK(fusedQScratch.scalar_type() == q.scalar_type(), "dsa fused_q scratch dtype must match q");
        TORCH_CHECK(
            latentOutputScratch.scalar_type() == q.scalar_type(), "dsa latent output scratch dtype must match q");
        TORCH_CHECK(fusedQScratch.dim() == 3, "dsa fused_q scratch must be 3D");
        TORCH_CHECK(latentOutputScratch.dim() == 3, "dsa latent output scratch must be 3D");
        TORCH_CHECK(fusedQScratch.size(0) >= inputTokens, "dsa fused_q scratch batch is smaller than input");
        TORCH_CHECK(
            latentOutputScratch.size(0) >= inputTokens, "dsa latent output scratch batch is smaller than input");
        TORCH_CHECK(fusedQScratch.size(1) == kBProjTrans.size(0), "dsa fused_q scratch head count must match k_b");
        TORCH_CHECK(
            latentOutputScratch.size(1) == kBProjTrans.size(0), "dsa latent output scratch head count must match k_b");
        TORCH_CHECK(
            fusedQScratch.size(2) == kBProjTrans.size(1) + kPe.size(1), "dsa fused_q scratch hidden width is invalid");
        TORCH_CHECK(
            latentOutputScratch.size(2) == kBProjTrans.size(1), "dsa latent output scratch width must be kv_lora_rank");
        TORCH_CHECK(cuQSeqLensScratch.scalar_type() == at::ScalarType::Int, "dsa cu_q_seqlens scratch must be int32");
        TORCH_CHECK(cuKvSeqLensScratch.scalar_type() == at::ScalarType::Int, "dsa cu_kv_seqlens scratch must be int32");
        TORCH_CHECK(cuQSeqLensScratch.dim() == 1, "dsa cu_q_seqlens scratch must be 1D");
        TORCH_CHECK(cuKvSeqLensScratch.dim() == 1, "dsa cu_kv_seqlens scratch must be 1D");
        TORCH_CHECK(cuQSeqLensScratch.numel() >= 2, "dsa cu_q_seqlens scratch must have at least two entries");
        TORCH_CHECK(cuKvSeqLensScratch.numel() == cuQSeqLensScratch.numel(),
            "dsa cu_kv_seqlens scratch must match cu_q_seqlens length");
        TORCH_CHECK(fmhaSchedulerCounterScratch.dim() == 1 && fmhaSchedulerCounterScratch.numel() >= 1,
            "dsa fmha_scheduler_counter scratch must be a non-empty 1D tensor");
        TORCH_CHECK(fmhaSchedulerCounterScratch.scalar_type() == at::ScalarType::UInt32
                || fmhaSchedulerCounterScratch.scalar_type() == at::ScalarType::Int,
            "dsa fmha_scheduler_counter scratch must be uint32 or int32");

        at::Tensor const rotaryCosSin = getRequiredTensor(dsaDispatchRuntimeTensors, "rotary_cos_sin");
        std::optional<at::Tensor> const topkIndicesPoolRuntime
            = getOptionalTensor(dsaDispatchRuntimeTensors, "topk_indices_pool");
        at::Tensor const kvCacheBlockOffsets = getRequiredTensor(dsaDispatchRuntimeTensors, "kv_cache_block_offsets");
        at::Tensor const hostKvCachePoolPointers
            = getRequiredTensor(dsaDispatchRuntimeTensors, "host_kv_cache_pool_pointers");
        at::Tensor const hostKvCachePoolMapping
            = getRequiredTensor(dsaDispatchRuntimeTensors, "host_kv_cache_pool_mapping");
        at::Tensor const kvLensRuntime = getRequiredTensor(dsaDispatchRuntimeTensors, "kv_lens_runtime");
        at::Tensor const promptLensCpuRuntime = getRequiredTensor(dsaDispatchRuntimeTensors, "prompt_lens_cpu_runtime");
        at::Tensor const denseKvPool = getRequiredTensor(dsaDispatchRuntimeTensors, "dense_kv_pool");
        checkCudaRuntimeTensor(rotaryCosSin, "rotary_cos_sin");
        checkCudaRuntimeTensor(kvCacheBlockOffsets, "kv_cache_block_offsets");
        checkRuntimeTensor(hostKvCachePoolPointers, "host_kv_cache_pool_pointers");
        checkRuntimeTensor(hostKvCachePoolMapping, "host_kv_cache_pool_mapping");
        checkRuntimeTensor(kvLensRuntime, "kv_lens_runtime");
        checkRuntimeTensor(promptLensCpuRuntime, "prompt_lens_cpu_runtime");
        checkCudaRuntimeTensor(denseKvPool, "dense_kv_pool");
        TORCH_CHECK(rotaryCosSin.dim() == 2, "rotary_cos_sin must be 2D");
        TORCH_CHECK(kvCacheBlockOffsets.dim() >= 1, "kv_cache_block_offsets must have at least one dimension");
        TORCH_CHECK(kvLensRuntime.dim() == 1, "kv_lens_runtime must be 1D");
        TORCH_CHECK(promptLensCpuRuntime.dim() == 1, "prompt_lens_cpu_runtime must be 1D");
        TORCH_CHECK(kvLensRuntime.numel() >= seqLensCuda.numel(), "kv_lens_runtime is shorter than seq_lens_cuda");
        TORCH_CHECK(promptLensCpuRuntime.numel() >= seqLensCuda.numel(),
            "prompt_lens_cpu_runtime is shorter than seq_lens_cuda");
        TORCH_CHECK(denseKvPool.dim() >= 2, "dense_kv_pool must expose block and layer dimensions");

        int64_t const numSeqs = getRequiredInt(dsaDispatchRuntimeConfig, "num_seqs");
        int64_t const numGenerations = getRequiredInt(dsaDispatchRuntimeConfig, "num_generations");
        int64_t const numContexts = getRequiredInt(dsaDispatchRuntimeConfig, "num_contexts");
        int64_t const numCtxTokens = getRequiredInt(dsaDispatchRuntimeConfig, "num_ctx_tokens");
        int64_t const maxSeqLen = getRequiredInt(dsaDispatchRuntimeConfig, "max_seq_len");
        int64_t const beamWidth = getRequiredInt(dsaDispatchRuntimeConfig, "beam_width");
        int64_t const tokensPerBlock = getRequiredInt(dsaDispatchRuntimeConfig, "tokens_per_block");
        int64_t const localLayerIdx = getRequiredInt(dsaDispatchRuntimeConfig, "local_layer_idx");
        int64_t const predictedTokensPerSeq = getRequiredInt(dsaDispatchRuntimeConfig, "predicted_tokens_per_seq");
        int64_t const numHeads = getRequiredInt(dsaDispatchRuntimeConfig, "num_heads");
        int64_t const numKvHeads = getRequiredInt(dsaDispatchRuntimeConfig, "num_kv_heads");
        int64_t const headDim = getRequiredInt(dsaDispatchRuntimeConfig, "head_dim");
        int64_t const qLoraRank = getRequiredInt(dsaDispatchRuntimeConfig, "q_lora_rank");
        int64_t const kvLoraRank = getRequiredInt(dsaDispatchRuntimeConfig, "kv_lora_rank");
        int64_t const qkNopeHeadDim = getRequiredInt(dsaDispatchRuntimeConfig, "qk_nope_head_dim");
        int64_t const qkRopeHeadDim = getRequiredInt(dsaDispatchRuntimeConfig, "qk_rope_head_dim");
        int64_t const vHeadDim = getRequiredInt(dsaDispatchRuntimeConfig, "v_head_dim");
        int64_t const quantMode = getRequiredInt(dsaDispatchRuntimeConfig, "quant_mode");
        int64_t const ropeAppend = getRequiredInt(dsaDispatchRuntimeConfig, "rope_append");
        int64_t const kvDispatchMode = getRequiredInt(dsaDispatchRuntimeConfig, "kv_dispatch_mode");
        int64_t const maxNumRequests = getRequiredInt(dsaDispatchRuntimeConfig, "max_num_requests");
        int64_t const maxContextLength = getRequiredInt(dsaDispatchRuntimeConfig, "max_context_length");
        int64_t const attentionWindowSize = getRequiredInt(dsaDispatchRuntimeConfig, "attention_window_size");
        int64_t const numSparseTopk = getRequiredInt(dsaDispatchRuntimeConfig, "num_sparse_topk");
        int64_t const sparseAttnIndicesBlockSize
            = getRequiredInt(dsaDispatchRuntimeConfig, "sparse_attn_indices_block_size");
        int64_t const maskType = getRequiredInt(dsaDispatchRuntimeConfig, "mask_type");
        int64_t const positionEmbeddingType = getRequiredInt(dsaDispatchRuntimeConfig, "position_embedding_type");
        int64_t const ropeDim = getRequiredInt(dsaDispatchRuntimeConfig, "rope_dim");
        int64_t const ropeScaleType = getRequiredInt(dsaDispatchRuntimeConfig, "rope_scale_type");
        int64_t const ropeMaxPositions = getRequiredInt(dsaDispatchRuntimeConfig, "rope_max_positions");
        int64_t const ropeOriginalMaxPositions
            = getRequiredInt(dsaDispatchRuntimeConfig, "rope_original_max_positions");
        int64_t const attentionChunkSize = getRequiredInt(dsaDispatchRuntimeConfig, "attention_chunk_size");
        int64_t const usePagedContextFmha = getRequiredInt(dsaDispatchRuntimeConfig, "use_paged_context_fmha");
        double const qScaling = getRequiredDouble(dsaDispatchRuntimeScalars, "q_scaling");
        double const softmaxScale = getRequiredDouble(dsaDispatchRuntimeScalars, "softmax_scale");
        double const ropeBase = getRequiredDouble(dsaDispatchRuntimeScalars, "rope_base");
        double const ropeScale = getRequiredDouble(dsaDispatchRuntimeScalars, "rope_scale");
        double const ropeShortMScale = getRequiredDouble(dsaDispatchRuntimeScalars, "rope_short_m_scale");
        double const ropeLongMScale = getRequiredDouble(dsaDispatchRuntimeScalars, "rope_long_m_scale");
        TORCH_CHECK(numSeqs > 0 && numGenerations > 0 && numContexts >= 0 && numContexts + numGenerations == numSeqs,
            "DSA runtime sequence counts are invalid");
        TORCH_CHECK(numCtxTokens >= 0 && numCtxTokens <= inputTokens, "DSA runtime num_ctx_tokens is invalid");
        TORCH_CHECK(localLayerIdx >= 0, "DSA runtime local_layer_idx must be non-negative");
        TORCH_CHECK(maxSeqLen > 0 && beamWidth > 0 && tokensPerBlock == 64 && predictedTokensPerSeq > 0,
            "DSA runtime max_seq_len/beam_width/tokens_per_block/predicted_tokens_per_seq are invalid");
        TORCH_CHECK(numHeads == kBProjTrans.size(0), "DSA runtime num_heads must match k_b projection heads");
        TORCH_CHECK(numKvHeads == 1, "DSA runtime num_kv_heads must be 1 for MLA");
        TORCH_CHECK(headDim == kvLoraRank + qkRopeHeadDim, "DSA runtime head_dim must equal kv_lora_rank + rope dim");
        TORCH_CHECK(qLoraRank > 0 && kvLoraRank == kBProjTrans.size(1), "DSA runtime q/kv LoRA ranks are invalid");
        TORCH_CHECK(qkNopeHeadDim == kBProjTrans.size(2), "DSA runtime qk_nope_head_dim must match k_b K dimension");
        TORCH_CHECK(qkRopeHeadDim == kPe.size(1), "DSA runtime rope dim must match k_pe width");
        TORCH_CHECK(vHeadDim == vBProj.size(1), "DSA runtime v_head_dim must match v_b projection");
        TORCH_CHECK(qScaling > 0.0 && softmaxScale > 0.0, "DSA runtime scalars must be positive");
        TORCH_CHECK(kvDispatchMode == kDsaKvDispatchDenseNvfp4 || kvDispatchMode == kDsaKvDispatchStandardMla,
            "DSA runtime kv_dispatch_mode is invalid");
        TORCH_CHECK(maxNumRequests > 0 && maxContextLength > 0 && attentionWindowSize > 0,
            "DSA runtime attention bounds are invalid");
        TORCH_CHECK(sparseAttnIndicesBlockSize > 0, "DSA runtime sparse attention block size must be positive");
        TORCH_CHECK(kvDispatchMode != kDsaKvDispatchStandardMla || numSparseTopk > 0,
            "standard MLA DSA runtime requires num_sparse_topk");
        TORCH_CHECK(
            denseKvPool.size(0) > 0 && denseKvPool.size(1) > 0, "dense_kv_pool must have block and layer dimensions");
        int64_t const topkPoolStrideFactor = denseKvPool.size(1) * tokensPerBlock;
        TORCH_CHECK(topkPoolStrideFactor > 0, "topk pool stride factor must be positive");

        at::Tensor topkIndicesPool;
        if (topkIndicesPoolRuntime.has_value())
        {
            topkIndicesPool = topkIndicesPoolRuntime.value();
        }
        else
        {
            at::Tensor const dsaReqIdxPerToken = getRequiredTensor(dsaDispatchRuntimeTensors, "dsa_req_idx_per_token");
            checkCudaRuntimeTensor(dsaReqIdxPerToken, "dsa_req_idx_per_token");
            TORCH_CHECK(dsaReqIdxPerToken.scalar_type() == at::ScalarType::Int, "dsa_req_idx_per_token must be int32");
            TORCH_CHECK(dsaReqIdxPerToken.dim() == 1, "dsa_req_idx_per_token must be 1D");
            TORCH_CHECK(dsaReqIdxPerToken.numel() >= numCtxTokens + inputTokens,
                "dsa_req_idx_per_token is shorter than the generation token window");
            TORCH_CHECK(dsaDispatchMetadataTensors.size() > 1,
                "dsa_dispatch_metadata_tensors must include block_table after topk_indices");
            at::Tensor const blockTable = dsaDispatchMetadataTensors.get(1);
            TORCH_CHECK(blockTable.scalar_type() == at::ScalarType::Int, "dsa dispatch block_table must be int32");
            TORCH_CHECK(blockTable.dim() == 2, "dsa dispatch block_table must be 2D");
            TORCH_CHECK(blockTable.size(0) >= numContexts + numGenerations,
                "dsa dispatch block_table is shorter than sequence count");
            at::Tensor reqIdxPrefix = dsaReqIdxPerToken.narrow(0, numCtxTokens, inputTokens);
            if (numContexts > 0)
            {
                reqIdxPrefix = reqIdxPrefix - numContexts;
            }
            at::Tensor const blockTableGen = blockTable.narrow(0, numContexts, numGenerations);
            at::Tensor const topkIndicesPrefix = topkIndices.narrow(0, 0, inputTokens);
            topkIndicesPool = convertReqIndexToGlobal(reqIdxPrefix, blockTableGen, topkIndicesPrefix, tokensPerBlock,
                topkIndices.size(1), topkPoolStrideFactor, localLayerIdx);
        }
        checkCudaRuntimeTensor(topkIndicesPool, "topk_indices_pool");
        TORCH_CHECK(topkIndicesPool.scalar_type() == at::ScalarType::Int, "topk_indices_pool must be int32");
        TORCH_CHECK(topkIndicesPool.dim() == 2, "topk_indices_pool must be 2D");
        TORCH_CHECK(topkIndicesPool.size(0) >= inputTokens, "topk_indices_pool batch is smaller than input");
        TORCH_CHECK(
            topkIndicesPool.size(1) == topkIndices.size(1), "topk_indices_pool width must match raw topk_indices");

        TORCH_CHECK(envFlagEnabled("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_DSA_NATIVE_BODY"),
            "DeepseekResidentDecodeHandle::runLayerDsaAttentionDispatch native body is disabled");
        TORCH_CHECK(
            q.scalar_type() == at::ScalarType::BFloat16, "resident DSA native body currently supports only bf16 q");
        TORCH_CHECK(kBProjTrans.scalar_type() == at::ScalarType::BFloat16,
            "resident DSA native body currently supports only bf16 k_b_proj_trans");
        TORCH_CHECK(vBProj.scalar_type() == at::ScalarType::BFloat16,
            "resident DSA native body currently supports only bf16 v_b_proj");
        TORCH_CHECK(numHeads == 128 && headDim == 576 && kvLoraRank == 512 && qkRopeHeadDim == 64 && vHeadDim == 128,
            "resident DSA native body currently supports only the DeepSeek V3.2 dense-NVFP4 production shape");
        TORCH_CHECK(inputTokens % numSeqs == 0,
            "resident DSA native body requires a uniform number of query tokens per sequence");
        int64_t const sQ = inputTokens / numSeqs;
        TORCH_CHECK(q.size(1) == numHeads * (qkNopeHeadDim + qkRopeHeadDim),
            "q width must match num_heads * (qk_nope_head_dim + qk_rope_head_dim)");
        TORCH_CHECK(attentionCoreOutputScratch.size(1) >= numHeads * vHeadDim,
            "attention_core_output_scratch width is too small for projected MLA output");

        at::Tensor qPrefix = q.narrow(0, 0, inputTokens);
        at::Tensor q3d = qPrefix.reshape({inputTokens, numHeads, qkNopeHeadDim + qkRopeHeadDim});
        at::Tensor qNope = q3d.narrow(2, 0, qkNopeHeadDim);
        at::Tensor qPe = q3d.narrow(2, qkNopeHeadDim, qkRopeHeadDim);
        at::Tensor fusedQ = fusedQScratch.narrow(0, 0, inputTokens);
        at::Tensor qNopeOut = fusedQ.narrow(2, 0, kvLoraRank).transpose(0, 1);
        at::bmm_out(qNopeOut, qNope.transpose(0, 1), kBProjTrans.transpose(1, 2));

        std::vector<std::optional<th::Tensor>> helixTensorParams{
            getOptionalTensor(dsaDispatchRuntimeTensors, "helix_position_offsets"),
            getOptionalTensor(dsaDispatchRuntimeTensors, "helix_is_inactive_rank")};
        std::optional<at::Tensor> mlaBmm1ScaleScratch;
        std::optional<at::Tensor> mlaBmm2ScaleScratch;
        std::optional<at::Tensor> quantQBufferScratch;
        if (kvDispatchMode == kDsaKvDispatchStandardMla)
        {
            TORCH_CHECK(dsaDispatchScratchTensors.size() >= 8,
                "standard MLA DSA dispatch requires FP8 MLA scale and quant-q scratch tensors");
            mlaBmm1ScaleScratch = dsaDispatchScratchTensors.get(5);
            mlaBmm2ScaleScratch = dsaDispatchScratchTensors.get(6);
            quantQBufferScratch = dsaDispatchScratchTensors.get(7);
            checkCudaRuntimeTensor(mlaBmm1ScaleScratch.value(), "mla_bmm1_scale_scratch");
            checkCudaRuntimeTensor(mlaBmm2ScaleScratch.value(), "mla_bmm2_scale_scratch");
            checkCudaRuntimeTensor(quantQBufferScratch.value(), "quant_q_buffer_scratch");
            TORCH_CHECK(
                mlaBmm1ScaleScratch->scalar_type() == at::ScalarType::Float, "mla_bmm1_scale_scratch must be float32");
            TORCH_CHECK(
                mlaBmm2ScaleScratch->scalar_type() == at::ScalarType::Float, "mla_bmm2_scale_scratch must be float32");
            TORCH_CHECK(
                quantQBufferScratch->scalar_type() == at::ScalarType::Byte, "quant_q_buffer_scratch must be uint8");
            TORCH_CHECK(mlaBmm1ScaleScratch->numel() >= 2, "mla_bmm1_scale_scratch is too small");
            TORCH_CHECK(mlaBmm2ScaleScratch->numel() >= 1, "mla_bmm2_scale_scratch is too small");
            TORCH_CHECK(quantQBufferScratch->dim() == 3, "quant_q_buffer_scratch must be 3D");
            TORCH_CHECK(quantQBufferScratch->size(0) >= inputTokens && quantQBufferScratch->size(1) == numHeads
                    && quantQBufferScratch->size(2) == headDim,
                "quant_q_buffer_scratch shape is invalid");
        }
        MLARopeGeneration(fusedQ, qPe, latentCache.narrow(0, 0, inputTokens), rotaryCosSin, cuQSeqLensScratch,
            cuKvSeqLensScratch, fmhaSchedulerCounterScratch, mlaBmm1ScaleScratch, mlaBmm2ScaleScratch,
            quantQBufferScratch, kvLensCuda, kvLensRuntime, promptLensCpuRuntime, numContexts, kvCacheBlockOffsets,
            hostKvCachePoolPointers, hostKvCachePoolMapping, std::nullopt, std::nullopt, std::nullopt,
            getOptionalTensor(dsaDispatchRuntimeTensors, "block_ids_per_seq"), helixTensorParams, predictedTokensPerSeq,
            localLayerIdx, numHeads, numKvHeads, headDim, tokensPerBlock, maxSeqLen, beamWidth, quantMode, qScaling,
            qLoraRank, kvLoraRank, qkNopeHeadDim, qkRopeHeadDim, vHeadDim, ropeAppend != 0);

        if (kvDispatchMode == kDsaKvDispatchStandardMla)
        {
            at::Tensor attentionWorkspace = getRequiredTensor(dsaDispatchRuntimeTensors, "attention_workspace");
            at::Tensor hostTotalKvLens = getRequiredTensor(dsaDispatchRuntimeTensors, "host_total_kv_lens");
            at::Tensor promptLensCudaRuntime = getRequiredTensor(dsaDispatchRuntimeTensors, "prompt_lens_cuda_runtime");
            at::Tensor hostRequestTypesRuntime
                = getRequiredTensor(dsaDispatchRuntimeTensors, "host_request_types_runtime");
            TORCH_CHECK(attentionWorkspace.defined(), "attention_workspace must be defined");
            TORCH_CHECK(attentionWorkspace.is_cuda(), "attention_workspace must be a CUDA tensor");
            checkRuntimeTensor(hostTotalKvLens, "host_total_kv_lens");
            checkCudaRuntimeTensor(promptLensCudaRuntime, "prompt_lens_cuda_runtime");
            checkRuntimeTensor(hostRequestTypesRuntime, "host_request_types_runtime");
            TORCH_CHECK(hostTotalKvLens.dim() == 1 && hostTotalKvLens.numel() >= 2,
                "host_total_kv_lens must be a CPU tensor with context and generation totals");
            TORCH_CHECK(promptLensCudaRuntime.dim() == 1 && promptLensCudaRuntime.numel() >= numSeqs,
                "prompt_lens_cuda_runtime is shorter than sequence count");
            TORCH_CHECK(hostRequestTypesRuntime.dim() == 1 && hostRequestTypesRuntime.numel() >= numSeqs,
                "host_request_types_runtime is shorter than sequence count");

            at::Tensor fusedQ2d = fusedQ.reshape({inputTokens, numHeads * headDim});
            at::Tensor latentOutput = latentOutputScratch.narrow(0, 0, inputTokens);
            at::Tensor latentOutput2d = latentOutput.reshape({inputTokens, numHeads * kvLoraRank});
            attention(fusedQ2d, std::nullopt, std::nullopt, latentOutput2d, std::nullopt, attentionWorkspace,
                kvLensCuda, kvLensRuntime, hostTotalKvLens, promptLensCudaRuntime, promptLensCpuRuntime,
                hostRequestTypesRuntime, kvCacheBlockOffsets, hostKvCachePoolPointers, hostKvCachePoolMapping,
                getOptionalTensor(dsaDispatchRuntimeTensors, "cache_indirection"),
                getOptionalTensor(dsaDispatchRuntimeTensors, "kv_scale_orig_quant"),
                getOptionalTensor(dsaDispatchRuntimeTensors, "kv_scale_quant_orig"), std::nullopt, std::nullopt,
                rotaryCosSin, latentCache.narrow(0, 0, inputTokens), qPe,
                getOptionalTensor(dsaDispatchRuntimeTensors, "block_ids_per_seq"), std::nullopt,
                /*is_fused_qkv=*/true, /*update_kv_cache=*/true, predictedTokensPerSeq, localLayerIdx, numHeads,
                numKvHeads, headDim, tokensPerBlock, maxNumRequests, maxContextLength, attentionWindowSize, beamWidth,
                maskType, quantMode, qScaling, positionEmbeddingType, ropeDim, ropeBase, ropeScaleType, ropeScale,
                ropeShortMScale, ropeLongMScale, ropeMaxPositions, ropeOriginalMaxPositions,
                static_cast<bool>(usePagedContextFmha), 2, /*is_mla_enable=*/true, std::nullopt, qLoraRank, kvLoraRank,
                qkNopeHeadDim, qkRopeHeadDim, vHeadDim, static_cast<bool>(ropeAppend), std::nullopt, std::nullopt,
                getOptionalTensor(dsaDispatchRuntimeTensors, "helix_position_offsets"),
                getOptionalTensor(dsaDispatchRuntimeTensors, "helix_is_inactive_rank"),
                attentionChunkSize > 0 ? std::optional<int64_t>(attentionChunkSize) : std::nullopt, std::nullopt,
                /*is_spec_decoding_enabled=*/false, /*use_spec_decoding=*/false, /*is_spec_dec_tree=*/false,
                std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                std::nullopt, topkIndicesPool.narrow(0, 0, inputTokens), std::nullopt, sparseAttnIndicesBlockSize,
                numSparseTopk, std::nullopt, std::nullopt, std::nullopt, std::nullopt, cuQSeqLensScratch,
                cuKvSeqLensScratch, fmhaSchedulerCounterScratch, mlaBmm1ScaleScratch, mlaBmm2ScaleScratch,
                quantQBufferScratch, getOptionalTensor(dsaDispatchRuntimeTensors, "flash_mla_tile_scheduler_metadata"),
                getOptionalTensor(dsaDispatchRuntimeTensors, "flash_mla_num_splits"), 0, 0, 0,
                /*sage_attn_qk_int8=*/false, numContexts, numCtxTokens, std::nullopt);

            at::Tensor output3d = attentionCoreOutputScratch.narrow(0, 0, inputTokens)
                                      .narrow(1, 0, numHeads * vHeadDim)
                                      .reshape({inputTokens, numHeads, vHeadDim});
            at::Tensor outputHeads = output3d.transpose(0, 1);
            at::bmm_out(outputHeads, latentOutput.transpose(0, 1), vBProj.transpose(1, 2));
            return attentionCoreOutputScratch;
        }

        at::Tensor const denseKvScalePool = getRequiredTensor(dsaDispatchRuntimeTensors, "dense_kv_scale_pool");
        checkCudaRuntimeTensor(denseKvScalePool, "dense_kv_scale_pool");
        TORCH_CHECK(denseKvScalePool.dim() >= 2, "dense_kv_scale_pool must expose block and layer dimensions");
        int64_t const denseKvPages = denseKvPool.size(0) * denseKvPool.size(1);
        TORCH_CHECK(denseKvPool.numel() == denseKvPages * tokensPerBlock * headDim / 2,
            "dense_kv_pool element count does not match dense NVFP4 dispatch shape");
        TORCH_CHECK(denseKvScalePool.numel() == denseKvPages * tokensPerBlock * headDim / 16,
            "dense_kv_scale_pool element count does not match dense NVFP4 dispatch shape");
        at::Tensor kvPacked = denseKvPool.reshape({denseKvPages, tokensPerBlock, 1, headDim / 2});
        at::Tensor kvScales = denseKvScalePool.reshape({denseKvPages, tokensPerBlock, 1, headDim / 16});
        at::Tensor sparseIndices
            = topkIndicesPool.narrow(0, 0, inputTokens).reshape({numSeqs, sQ, topkIndicesPool.size(1)}).contiguous();
        at::Tensor sparseQ = fusedQ.reshape({numSeqs, sQ, numHeads, headDim});
        at::Tensor output4d = attentionCoreOutputScratch.narrow(0, 0, inputTokens)
                                  .narrow(1, 0, numHeads * vHeadDim)
                                  .reshape({numSeqs, sQ, numHeads, vHeadDim});
        static_cast<void>(sparse_mla_decode_nvfp4_vfuse_out(sparseQ, kvPacked, kvScales, sparseIndices, vBProj,
            output4d, std::nullopt, std::nullopt, std::nullopt, std::nullopt, kvLoraRank, vHeadDim, softmaxScale));
        return attentionCoreOutputScratch;
    }

    at::Tensor runLayerDsaAttentionDispatchCompiled(DeepseekResidentCompiledDsaWindowLayer const& layerPlan,
        at::Tensor const& q, at::Tensor const& compressedKv, at::Tensor const& kPe, at::Tensor const& latentCache,
        at::Tensor const& positionIds, at::Tensor const& seqLensCuda, at::Tensor const& kvLensCuda,
        at::Tensor const& topkIndices, at::Tensor const& attentionCoreOutputScratch, int64_t inputTokens) const
    {
        static_cast<void>(compressedKv);
        static_cast<void>(seqLensCuda);

        int64_t const numSeqs = layerPlan.numSeqs;
        int64_t const numContexts = layerPlan.numContexts;
        int64_t const numGenerations = layerPlan.numGenerations;
        int64_t const numCtxTokens = layerPlan.numCtxTokens;
        int64_t const numHeads = layerPlan.numHeads;
        int64_t const numKvHeads = layerPlan.numKvHeads;
        int64_t const headDim = layerPlan.headDim;
        int64_t const qLoraRank = layerPlan.qLoraRank;
        int64_t const kvLoraRank = layerPlan.kvLoraRank;
        int64_t const qkNopeHeadDim = layerPlan.qkNopeHeadDim;
        int64_t const qkRopeHeadDim = layerPlan.qkRopeHeadDim;
        int64_t const vHeadDim = layerPlan.vHeadDim;
        int64_t const tokensPerBlock = layerPlan.tokensPerBlock;
        int64_t const sQ = inputTokens / numSeqs;

        at::Tensor topkIndicesPool;
        if (layerPlan.topkIndicesPoolRuntime.has_value())
        {
            topkIndicesPool = layerPlan.topkIndicesPoolRuntime.value();
        }
        else
        {
            int64_t const topkPoolStrideFactor = layerPlan.denseKvPool.size(1) * tokensPerBlock;
            at::Tensor reqIdxPrefix = layerPlan.reqIdxPerToken.narrow(0, numCtxTokens, inputTokens);
            if (numContexts > 0)
            {
                reqIdxPrefix = reqIdxPrefix - numContexts;
            }
            at::Tensor const blockTableGen = layerPlan.blockTable.narrow(0, numContexts, numGenerations);
            at::Tensor const topkIndicesPrefix = topkIndices.narrow(0, 0, inputTokens);
            topkIndicesPool = convertReqIndexToGlobal(reqIdxPrefix, blockTableGen, topkIndicesPrefix, tokensPerBlock,
                topkIndices.size(1), topkPoolStrideFactor, layerPlan.localLayerIdx);
        }

        at::Tensor qPrefix = q.narrow(0, 0, inputTokens);
        at::Tensor q3d = qPrefix.reshape({inputTokens, numHeads, qkNopeHeadDim + qkRopeHeadDim});
        at::Tensor qNope = q3d.narrow(2, 0, qkNopeHeadDim);
        at::Tensor qPe = q3d.narrow(2, qkNopeHeadDim, qkRopeHeadDim);
        at::Tensor fusedQ = layerPlan.fusedQScratch.narrow(0, 0, inputTokens);
        at::Tensor qNopeOut = fusedQ.narrow(2, 0, kvLoraRank).transpose(0, 1);
        at::bmm_out(qNopeOut, qNope.transpose(0, 1), layerPlan.kBProjTrans.transpose(1, 2));

        std::vector<std::optional<th::Tensor>> helixTensorParams{
            layerPlan.helixPositionOffsets, layerPlan.helixIsInactiveRank};
        MLARopeGeneration(fusedQ, qPe, latentCache.narrow(0, 0, inputTokens), layerPlan.rotaryCosSin,
            layerPlan.cuQSeqLensScratch, layerPlan.cuKvSeqLensScratch, layerPlan.fmhaSchedulerCounterScratch,
            layerPlan.mlaBmm1ScaleScratch, layerPlan.mlaBmm2ScaleScratch, layerPlan.quantQBufferScratch, kvLensCuda,
            layerPlan.kvLensRuntime, layerPlan.promptLensCpuRuntime, numContexts, layerPlan.kvCacheBlockOffsets,
            layerPlan.hostKvCachePoolPointers, layerPlan.hostKvCachePoolMapping, std::nullopt, std::nullopt,
            std::nullopt, layerPlan.blockIdsPerSeq, helixTensorParams, layerPlan.predictedTokensPerSeq,
            layerPlan.localLayerIdx, numHeads, numKvHeads, headDim, tokensPerBlock, layerPlan.maxSeqLen,
            layerPlan.beamWidth, layerPlan.quantMode, layerPlan.qScaling, qLoraRank, kvLoraRank, qkNopeHeadDim,
            qkRopeHeadDim, vHeadDim, layerPlan.ropeAppend != 0);

        if (layerPlan.kvDispatchMode == kDsaKvDispatchStandardMla)
        {
            at::Tensor fusedQ2d = fusedQ.reshape({inputTokens, numHeads * headDim});
            at::Tensor latentOutput = layerPlan.latentOutputScratch.narrow(0, 0, inputTokens);
            at::Tensor latentOutput2d = latentOutput.reshape({inputTokens, numHeads * kvLoraRank});
            attention(fusedQ2d, std::nullopt, std::nullopt, latentOutput2d, std::nullopt, layerPlan.attentionWorkspace,
                kvLensCuda, layerPlan.kvLensRuntime, layerPlan.hostTotalKvLens.value(),
                layerPlan.promptLensCudaRuntime.value(), layerPlan.promptLensCpuRuntime,
                layerPlan.hostRequestTypesRuntime.value(), layerPlan.kvCacheBlockOffsets,
                layerPlan.hostKvCachePoolPointers, layerPlan.hostKvCachePoolMapping, layerPlan.cacheIndirection,
                layerPlan.kvScaleOrigQuant, layerPlan.kvScaleQuantOrig, std::nullopt, std::nullopt,
                layerPlan.rotaryCosSin, latentCache.narrow(0, 0, inputTokens), qPe, layerPlan.blockIdsPerSeq,
                std::nullopt, /*is_fused_qkv=*/true, /*update_kv_cache=*/true, layerPlan.predictedTokensPerSeq,
                layerPlan.localLayerIdx, numHeads, numKvHeads, headDim, tokensPerBlock, layerPlan.maxNumRequests,
                layerPlan.maxContextLength, layerPlan.attentionWindowSize, layerPlan.beamWidth, layerPlan.maskType,
                layerPlan.quantMode, layerPlan.qScaling, layerPlan.positionEmbeddingType, layerPlan.ropeDim,
                layerPlan.ropeBase, layerPlan.ropeScaleType, layerPlan.ropeScale, layerPlan.ropeShortMScale,
                layerPlan.ropeLongMScale, layerPlan.ropeMaxPositions, layerPlan.ropeOriginalMaxPositions,
                static_cast<bool>(layerPlan.usePagedContextFmha), 2, /*is_mla_enable=*/true, std::nullopt, qLoraRank,
                kvLoraRank, qkNopeHeadDim, qkRopeHeadDim, vHeadDim, static_cast<bool>(layerPlan.ropeAppend),
                std::nullopt, std::nullopt, layerPlan.helixPositionOffsets, layerPlan.helixIsInactiveRank,
                layerPlan.attentionChunkSize > 0 ? std::optional<int64_t>(layerPlan.attentionChunkSize) : std::nullopt,
                std::nullopt, /*is_spec_decoding_enabled=*/false, /*use_spec_decoding=*/false,
                /*is_spec_dec_tree=*/false, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                std::nullopt, std::nullopt, std::nullopt, topkIndicesPool.narrow(0, 0, inputTokens), std::nullopt,
                layerPlan.sparseAttnIndicesBlockSize, layerPlan.numSparseTopk, std::nullopt, std::nullopt, std::nullopt,
                std::nullopt, layerPlan.cuQSeqLensScratch, layerPlan.cuKvSeqLensScratch,
                layerPlan.fmhaSchedulerCounterScratch, layerPlan.mlaBmm1ScaleScratch, layerPlan.mlaBmm2ScaleScratch,
                layerPlan.quantQBufferScratch, layerPlan.flashMlaTileSchedulerMetadata, layerPlan.flashMlaNumSplits, 0,
                0, 0, /*sage_attn_qk_int8=*/false, numContexts, numCtxTokens, std::nullopt);

            at::Tensor output3d = attentionCoreOutputScratch.narrow(0, 0, inputTokens)
                                      .narrow(1, 0, numHeads * vHeadDim)
                                      .reshape({inputTokens, numHeads, vHeadDim});
            at::Tensor outputHeads = output3d.transpose(0, 1);
            at::bmm_out(outputHeads, latentOutput.transpose(0, 1), layerPlan.vBProj.transpose(1, 2));
            return attentionCoreOutputScratch;
        }

        at::Tensor sparseIndices
            = topkIndicesPool.narrow(0, 0, inputTokens).reshape({numSeqs, sQ, topkIndicesPool.size(1)}).contiguous();
        at::Tensor sparseQ = fusedQ.reshape({numSeqs, sQ, numHeads, headDim});
        at::Tensor output4d = attentionCoreOutputScratch.narrow(0, 0, inputTokens)
                                  .narrow(1, 0, numHeads * vHeadDim)
                                  .reshape({numSeqs, sQ, numHeads, vHeadDim});
        static_cast<void>(sparse_mla_decode_nvfp4_vfuse_out(sparseQ, layerPlan.denseKvPacked, layerPlan.denseKvScales,
            sparseIndices, layerPlan.vBProj, output4d, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
            kvLoraRank, vHeadDim, layerPlan.softmaxScale));
        return attentionCoreOutputScratch;
    }

    at::Tensor runDecodeWindowAdvanceState(at::Tensor const& initialTokens, at::Tensor const& windowTokensScratch,
        at::Tensor const& inputIdsScratch, at::Tensor const& positionIds, at::Tensor const& kvLensCuda,
        int64_t outputStepIdx, int64_t inputTokens) const
    {
        validateDecodeWindowAdvanceContract(
            initialTokens, windowTokensScratch, inputIdsScratch, positionIds, kvLensCuda, outputStepIdx, inputTokens);

        at::Tensor inputPrefix = inputIdsScratch.reshape({inputIdsScratch.numel()}).narrow(0, 0, inputTokens);
        if (outputStepIdx == 1)
        {
            at::Tensor initialTokenIds = initialTokens.select(0, 0).narrow(0, 0, inputTokens).select(1, 0);
            at::Tensor initialOutput = windowTokensScratch.select(0, 0).narrow(0, 0, inputTokens).select(1, 0);
            initialOutput.copy_(initialTokenIds.to(windowTokensScratch.scalar_type()));
            inputPrefix.copy_(initialTokenIds.to(inputIdsScratch.scalar_type()));
        }
        else
        {
            at::Tensor sourceTokenIds
                = windowTokensScratch.select(0, outputStepIdx - 1).narrow(0, 0, inputTokens).select(1, 0);
            inputPrefix.copy_(sourceTokenIds.to(inputIdsScratch.scalar_type()));
        }

        at::Tensor positionPrefix = positionIds.reshape({positionIds.numel()}).narrow(0, 0, inputTokens);
        positionPrefix.add_(1);
        at::Tensor kvLensPrefix = kvLensCuda.narrow(0, 0, inputTokens);
        kvLensPrefix.add_(1);
        return inputIdsScratch;
    }

    at::Tensor runDecodeWindowSampleStep(at::Tensor const& logitsScratch, at::Tensor const& windowTokensScratch,
        int64_t outputStepIdx, int64_t inputTokens) const
    {
        validateDecodeWindowSampleStepContract(logitsScratch, windowTokensScratch, outputStepIdx, inputTokens);

        at::Tensor logitsPrefix = validTokenLogitsPrefix(logitsScratch, inputTokens);
        at::Tensor tokenIds = std::get<1>(logitsPrefix.max(/*dim=*/1, /*keepdim=*/false));
        at::Tensor tokenOut = windowTokensScratch.select(0, outputStepIdx).narrow(0, 0, inputTokens).select(1, 0);
        tokenOut.copy_(tokenIds.to(windowTokensScratch.scalar_type()));
        return windowTokensScratch;
    }

    at::Tensor runDecodeWindowPrepareStep(at::Tensor const& initialTokens, at::Tensor const& windowTokensScratch,
        at::Tensor const& inputIdsScratch, at::Tensor const& hiddenStatesScratch, at::Tensor const& positionIds,
        at::Tensor const& kvLensCuda, int64_t outputStepIdx, int64_t inputTokens) const
    {
        validateDecodeWindowPrepareStepContract(initialTokens, windowTokensScratch, inputIdsScratch,
            hiddenStatesScratch, positionIds, kvLensCuda, outputStepIdx, inputTokens);
        static_cast<void>(runDecodeWindowAdvanceState(
            initialTokens, windowTokensScratch, inputIdsScratch, positionIds, kvLensCuda, outputStepIdx, inputTokens));
        return runInputEmbedding(inputIdsScratch, hiddenStatesScratch, inputTokens);
    }

    bool runDecodeWindowWithDsaPlanReady() const
    {
        return envFlagEnabled("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_BODY") && mNbLayers > 0
            && runLayerDsaAttentionDispatchReady();
    }

    std::string runDecodeWindowWithDsaPlanNotReadyReason() const
    {
        if (mNbLayers <= 0)
        {
            return "resident_window_native_no_layers";
        }
        if (!envFlagEnabled("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_BODY"))
        {
            return "resident_window_native_body_disabled";
        }
        if (!runLayerDsaAttentionDispatchReady())
        {
            return "resident_window_native_missing_dsa_attention_dispatch";
        }
        return "resident_window_native_plan_ready";
    }

    at::Tensor runDecodeWindowWithDsaPlan(at::Tensor const& initialTokens, at::Tensor const& windowTokensScratch,
        at::Tensor const& inputIdsScratch, at::Tensor const& hiddenStatesScratch, at::Tensor const& logitsScratch,
        at::Tensor const& positionIds, at::Tensor const& seqLensCuda, at::Tensor const& kvLensCuda,
        th::List<int64_t> dsaLayerIndices, th::List<int64_t> dsaMetadataOffsets,
        c10::List<at::Tensor> dsaMetadataTensors, c10::List<c10::Dict<std::string, at::Tensor>> dsaRuntimeTensors,
        c10::List<c10::Dict<std::string, int64_t>> dsaRuntimeConfig,
        c10::List<c10::Dict<std::string, double>> dsaRuntimeScalars, th::List<int64_t> dsaScratchOffsets,
        c10::List<at::Tensor> dsaScratchTensors, int64_t ownedSteps, int64_t inputTokens, th::List<int64_t> requestIds,
        th::List<int64_t> seqLens, th::List<int64_t> cachedTokens) const
    {
        bool const timingEnabled = residentWindowCppTimingEnabled();
        cudaStream_t const timingStream = at::cuda::getCurrentCUDAStream(initialTokens.get_device()).stream();
        bool const cudaEventTimingEnabled = residentWindowCudaEventTimingEnabled() && !isStreamCapturing(timingStream);
        ResidentWindowTimingStats timing;
        ResidentWindowTimingStats deviceTiming;
        std::vector<ResidentWindowCudaEventRecord> cudaEventTimingRecords;
        int64_t const attentionTailFp4OutGateVisitsStart = residentAttentionTailFp4OutGateVisits();
        std::array<int64_t, kResidentAttentionTailFp4OutGateStatCount> attentionTailFp4OutGateStatsStart{};
        for (size_t i = 0; i < attentionTailFp4OutGateStatsStart.size(); ++i)
        {
            attentionTailFp4OutGateStatsStart[i]
                = residentAttentionTailFp4OutGateStat(static_cast<ResidentAttentionTailFp4OutGateStat>(i));
        }
        auto attentionTailFp4OutGateStatDelta
            = [&attentionTailFp4OutGateStatsStart](ResidentAttentionTailFp4OutGateStat stat) {
                  size_t const index = static_cast<size_t>(stat);
                  return residentAttentionTailFp4OutGateStat(stat) - attentionTailFp4OutGateStatsStart[index];
              };
        auto const totalStart = ResidentWindowTimingClock::now();
        {
            ResidentWindowScopedStageTimer const timer(timingEnabled, timing.contractUs, cudaEventTimingEnabled,
                deviceTiming.contractUs, cudaEventTimingRecords, timingStream);
            validateDecodeWindowTokenContract(initialTokens, windowTokensScratch, inputIdsScratch, hiddenStatesScratch,
                logitsScratch, positionIds, seqLensCuda, kvLensCuda, ownedSteps, inputTokens, requestIds, seqLens,
                cachedTokens);
        }
        int64_t const nbPlanLayers = static_cast<int64_t>(dsaLayerIndices.size());
        std::vector<DeepseekResidentCompiledDsaWindowLayer> dsaWindowLayers;
        dsaWindowLayers.reserve(static_cast<size_t>(std::max<int64_t>(nbPlanLayers, 0)));
        if (cudaEventTimingEnabled)
        {
            constexpr int64_t kApproxTimedStagesPerLayer = 16;
            constexpr int64_t kFixedTimedStages = 8;
            cudaEventTimingRecords.reserve(static_cast<size_t>(
                std::max<int64_t>(ownedSteps * nbPlanLayers * kApproxTimedStagesPerLayer + kFixedTimedStages, 0)));
        }
        {
            ResidentWindowScopedStageTimer const timer(timingEnabled, timing.planValidateUs, cudaEventTimingEnabled,
                deviceTiming.planValidateUs, cudaEventTimingRecords, timingStream);
            TORCH_CHECK(nbPlanLayers > 0, "DSA window plan must contain at least one layer");
            TORCH_CHECK(static_cast<int64_t>(dsaRuntimeTensors.size()) == nbPlanLayers,
                "DSA runtime tensor dict count must match layer count");
            TORCH_CHECK(static_cast<int64_t>(dsaRuntimeConfig.size()) == nbPlanLayers,
                "DSA runtime config dict count must match layer count");
            TORCH_CHECK(static_cast<int64_t>(dsaRuntimeScalars.size()) == nbPlanLayers,
                "DSA runtime scalar dict count must match layer count");
            TORCH_CHECK(static_cast<int64_t>(dsaMetadataOffsets.size()) == nbPlanLayers + 1,
                "DSA metadata offsets must have layer count + 1 entries");
            TORCH_CHECK(static_cast<int64_t>(dsaScratchOffsets.size()) == nbPlanLayers + 1,
                "DSA scratch offsets must have layer count + 1 entries");
            TORCH_CHECK(dsaMetadataOffsets.get(0) == 0, "DSA metadata offsets must start at zero");
            TORCH_CHECK(dsaScratchOffsets.get(0) == 0, "DSA scratch offsets must start at zero");
            TORCH_CHECK(dsaMetadataOffsets.get(nbPlanLayers) == static_cast<int64_t>(dsaMetadataTensors.size()),
                "DSA metadata final offset must match flattened tensor count");
            TORCH_CHECK(dsaScratchOffsets.get(nbPlanLayers) == static_cast<int64_t>(dsaScratchTensors.size()),
                "DSA scratch final offset must match flattened tensor count");

            for (int64_t planIdx = 0; planIdx < nbPlanLayers; ++planIdx)
            {
                DeepseekResidentCompiledDsaWindowLayer compiled;
                compiled.layerIdx = dsaLayerIndices.get(planIdx);
                TORCH_CHECK(compiled.layerIdx >= 0 && compiled.layerIdx < mNbLayers,
                    "DSA window plan layer index is out of range");
                int64_t const metadataStart = dsaMetadataOffsets.get(planIdx);
                int64_t const metadataStop = dsaMetadataOffsets.get(planIdx + 1);
                int64_t const scratchStart = dsaScratchOffsets.get(planIdx);
                int64_t const scratchStop = dsaScratchOffsets.get(planIdx + 1);
                TORCH_CHECK(metadataStart >= 0 && metadataStart <= metadataStop
                        && metadataStop <= static_cast<int64_t>(dsaMetadataTensors.size()),
                    "DSA metadata offsets are invalid");
                TORCH_CHECK(scratchStart >= 0 && scratchStart <= scratchStop
                        && scratchStop <= static_cast<int64_t>(dsaScratchTensors.size()),
                    "DSA scratch offsets are invalid");
                TORCH_CHECK(metadataStop > metadataStart, "DSA window plan metadata tensor span must not be empty");
                TORCH_CHECK(metadataStop - metadataStart >= 9, "DSA window plan metadata tensor span is too short");
                compiled.runtimeTensors = dsaRuntimeTensors.get(planIdx);
                compiled.runtimeConfig = dsaRuntimeConfig.get(planIdx);
                compiled.runtimeScalars = dsaRuntimeScalars.get(planIdx);
                TORCH_CHECK(!compiled.runtimeTensors.empty(), "DSA window plan runtime tensor dict must not be empty");
                TORCH_CHECK(!compiled.runtimeConfig.empty(), "DSA window plan runtime config dict must not be empty");
                TORCH_CHECK(!compiled.runtimeScalars.empty(), "DSA window plan runtime scalar dict must not be empty");

                compiled.kvDispatchMode = getRequiredInt(compiled.runtimeConfig, "kv_dispatch_mode");
                TORCH_CHECK(compiled.kvDispatchMode == kDsaKvDispatchDenseNvfp4
                        || compiled.kvDispatchMode == kDsaKvDispatchStandardMla,
                    "native DSA window body kv_dispatch_mode is invalid");
                compiled.dispatchScratchCount = compiled.kvDispatchMode == kDsaKvDispatchStandardMla ? 8 : 5;
                compiled.indexerScratchStart = compiled.dispatchScratchCount;
                TORCH_CHECK(scratchStop - scratchStart >= compiled.indexerScratchStart + 7,
                    "DSA window plan scratch span must contain dispatch and FP4 indexer scratch");

                for (int64_t tensorIdx = metadataStart; tensorIdx < metadataStop; ++tensorIdx)
                {
                    at::Tensor const tensor = dsaMetadataTensors.get(tensorIdx);
                    TORCH_CHECK(tensor.is_cuda(), "DSA window plan metadata tensors must be CUDA tensors");
                    TORCH_CHECK(tensor.numel() > 0, "DSA window plan metadata tensors must not be empty");
                }
                for (int64_t tensorIdx = scratchStart; tensorIdx < scratchStop; ++tensorIdx)
                {
                    at::Tensor const tensor = dsaScratchTensors.get(tensorIdx);
                    TORCH_CHECK(tensor.is_cuda(), "DSA window plan scratch tensors must be CUDA tensors");
                    TORCH_CHECK(tensor.numel() > 0, "DSA window plan scratch tensors must not be empty");
                }

                compiled.blockTable = dsaMetadataTensors.get(metadataStart);
                compiled.indexerKCache = dsaMetadataTensors.get(metadataStart + 1);
                compiled.indexerKCacheBlockOffsets = dsaMetadataTensors.get(metadataStart + 2);
                compiled.schedulerMetadataBuffer = dsaMetadataTensors.get(metadataStart + 3);
                compiled.slotMappingFp8 = dsaMetadataTensors.get(metadataStart + 4);
                compiled.slotMappingScale = dsaMetadataTensors.get(metadataStart + 5);
                compiled.genKvIndptr = dsaMetadataTensors.get(metadataStart + 6);
                compiled.genCachedTokenIndptr = dsaMetadataTensors.get(metadataStart + 7);
                compiled.kvLensCuda2d = dsaMetadataTensors.get(metadataStart + 8);
                compiled.reqIdxPerToken = getRequiredTensor(compiled.runtimeTensors, "dsa_req_idx_per_token");
                compiled.indexerHisaPageReps = getOptionalTensor(compiled.runtimeTensors, "indexer_hisa_page_reps");
                compiled.indexerHisaPageCounts = getOptionalTensor(compiled.runtimeTensors, "indexer_hisa_page_counts");

                compiled.qFp4Scratch = dsaScratchTensors.get(scratchStart + compiled.indexerScratchStart);
                compiled.kFp4Scratch = dsaScratchTensors.get(scratchStart + compiled.indexerScratchStart + 1);
                compiled.kScaleScratch = dsaScratchTensors.get(scratchStart + compiled.indexerScratchStart + 2);
                compiled.indexerWeightsScratch = dsaScratchTensors.get(scratchStart + compiled.indexerScratchStart + 3);
                compiled.qScaleScratch = dsaScratchTensors.get(scratchStart + compiled.indexerScratchStart + 4);
                compiled.topkIndices = dsaScratchTensors.get(scratchStart + compiled.indexerScratchStart + 5);
                compiled.xstepRefreshEnd = dsaScratchTensors.get(scratchStart + compiled.indexerScratchStart + 6);

                compiled.kBProjTrans = getLayerTensor(compiled.layerIdx,
                    DeepseekResidentLayerTensorSite::kAttentionKBProjTrans, "attention k_b_proj_trans");
                compiled.vBProj = getLayerTensor(
                    compiled.layerIdx, DeepseekResidentLayerTensorSite::kAttentionVBProj, "attention v_b_proj");
                compiled.rotaryCosSin = getRequiredTensor(compiled.runtimeTensors, "rotary_cos_sin");
                compiled.topkIndicesPoolRuntime = getOptionalTensor(compiled.runtimeTensors, "topk_indices_pool");
                compiled.kvCacheBlockOffsets = getRequiredTensor(compiled.runtimeTensors, "kv_cache_block_offsets");
                compiled.hostKvCachePoolPointers
                    = getRequiredTensor(compiled.runtimeTensors, "host_kv_cache_pool_pointers");
                compiled.hostKvCachePoolMapping
                    = getRequiredTensor(compiled.runtimeTensors, "host_kv_cache_pool_mapping");
                compiled.kvLensRuntime = getRequiredTensor(compiled.runtimeTensors, "kv_lens_runtime");
                compiled.promptLensCpuRuntime = getRequiredTensor(compiled.runtimeTensors, "prompt_lens_cpu_runtime");
                compiled.denseKvPool = getRequiredTensor(compiled.runtimeTensors, "dense_kv_pool");
                compiled.blockIdsPerSeq = getOptionalTensor(compiled.runtimeTensors, "block_ids_per_seq");
                compiled.helixPositionOffsets = getOptionalTensor(compiled.runtimeTensors, "helix_position_offsets");
                compiled.helixIsInactiveRank = getOptionalTensor(compiled.runtimeTensors, "helix_is_inactive_rank");

                compiled.metadataTensors.push_back(compiled.topkIndices);
                for (int64_t tensorIdx = metadataStart; tensorIdx < metadataStop; ++tensorIdx)
                {
                    compiled.metadataTensors.push_back(dsaMetadataTensors.get(tensorIdx));
                }
                for (int64_t tensorIdx = scratchStart; tensorIdx < scratchStart + compiled.dispatchScratchCount;
                    ++tensorIdx)
                {
                    compiled.dispatchScratchTensors.push_back(dsaScratchTensors.get(tensorIdx));
                }
                compiled.fusedQScratch = dsaScratchTensors.get(scratchStart);
                compiled.latentOutputScratch = dsaScratchTensors.get(scratchStart + 1);
                compiled.cuQSeqLensScratch = dsaScratchTensors.get(scratchStart + 2);
                compiled.cuKvSeqLensScratch = dsaScratchTensors.get(scratchStart + 3);
                compiled.fmhaSchedulerCounterScratch = dsaScratchTensors.get(scratchStart + 4);
                if (compiled.kvDispatchMode == kDsaKvDispatchStandardMla)
                {
                    compiled.mlaBmm1ScaleScratch = dsaScratchTensors.get(scratchStart + 5);
                    compiled.mlaBmm2ScaleScratch = dsaScratchTensors.get(scratchStart + 6);
                    compiled.quantQBufferScratch = dsaScratchTensors.get(scratchStart + 7);
                }

                compiled.numSeqs = getRequiredInt(compiled.runtimeConfig, "num_seqs");
                compiled.numContexts = getRequiredInt(compiled.runtimeConfig, "num_contexts");
                compiled.numCtxTokens = getRequiredInt(compiled.runtimeConfig, "num_ctx_tokens");
                compiled.numGenerations = getRequiredInt(compiled.runtimeConfig, "num_generations");
                compiled.numSparseTopk = getRequiredInt(compiled.runtimeConfig, "num_sparse_topk");
                compiled.numHeads = getRequiredInt(compiled.runtimeConfig, "num_heads");
                compiled.qLoraRank = getRequiredInt(compiled.runtimeConfig, "q_lora_rank");
                compiled.kvLoraRank = getRequiredInt(compiled.runtimeConfig, "kv_lora_rank");
                compiled.qkNopeHeadDim = getRequiredInt(compiled.runtimeConfig, "qk_nope_head_dim");
                compiled.qkRopeHeadDim = getRequiredInt(compiled.runtimeConfig, "qk_rope_head_dim");
                compiled.vHeadDim = getRequiredInt(compiled.runtimeConfig, "v_head_dim");
                compiled.maxSeqLen = getRequiredInt(compiled.runtimeConfig, "max_seq_len");
                compiled.beamWidth = getRequiredInt(compiled.runtimeConfig, "beam_width");
                compiled.localLayerIdx = getRequiredInt(compiled.runtimeConfig, "local_layer_idx");
                compiled.predictedTokensPerSeq = getRequiredInt(compiled.runtimeConfig, "predicted_tokens_per_seq");
                compiled.numKvHeads = getRequiredInt(compiled.runtimeConfig, "num_kv_heads");
                compiled.headDim = getRequiredInt(compiled.runtimeConfig, "head_dim");
                compiled.quantMode = getRequiredInt(compiled.runtimeConfig, "quant_mode");
                compiled.ropeAppend = getRequiredInt(compiled.runtimeConfig, "rope_append");
                compiled.maxNumRequests = getRequiredInt(compiled.runtimeConfig, "max_num_requests");
                compiled.maxContextLength = getRequiredInt(compiled.runtimeConfig, "max_context_length");
                compiled.attentionWindowSize = getRequiredInt(compiled.runtimeConfig, "attention_window_size");
                compiled.sparseAttnIndicesBlockSize
                    = getRequiredInt(compiled.runtimeConfig, "sparse_attn_indices_block_size");
                compiled.maskType = getRequiredInt(compiled.runtimeConfig, "mask_type");
                compiled.positionEmbeddingType = getRequiredInt(compiled.runtimeConfig, "position_embedding_type");
                compiled.ropeDim = getRequiredInt(compiled.runtimeConfig, "rope_dim");
                compiled.ropeScaleType = getRequiredInt(compiled.runtimeConfig, "rope_scale_type");
                compiled.ropeMaxPositions = getRequiredInt(compiled.runtimeConfig, "rope_max_positions");
                compiled.ropeOriginalMaxPositions
                    = getRequiredInt(compiled.runtimeConfig, "rope_original_max_positions");
                compiled.attentionChunkSize = getRequiredInt(compiled.runtimeConfig, "attention_chunk_size");
                compiled.usePagedContextFmha = getRequiredInt(compiled.runtimeConfig, "use_paged_context_fmha");
                compiled.tokensPerBlock = getRequiredInt(compiled.runtimeConfig, "tokens_per_block");
                compiled.residentIndexerHeadDim = getRequiredInt(compiled.runtimeConfig, "resident_indexer_head_dim");
                compiled.residentIndexerNumHeads = getRequiredInt(compiled.runtimeConfig, "resident_indexer_num_heads");
                compiled.residentIndexerRopeDim = getRequiredInt(compiled.runtimeConfig, "resident_indexer_rope_dim");
                compiled.residentIndexerQuantBlockSize
                    = getRequiredInt(compiled.runtimeConfig, "resident_indexer_quant_block_size");
                compiled.residentIndexerDataBytesPerToken
                    = getRequiredInt(compiled.runtimeConfig, "resident_indexer_data_bytes_per_token");
                compiled.reusePreviousIndexerTopk
                    = getOptionalInt(compiled.runtimeConfig, "resident_indexer_skip_topk", 0) != 0;
                compiled.residentIndexerStepFreq
                    = residentIndexerStepFreq(getOptionalInt(compiled.runtimeConfig, "resident_indexer_step_freq", 1));
                compiled.residentIndexerStepRecencyPatch = residentIndexerStepRecencyPatchEnabled(
                    getOptionalInt(compiled.runtimeConfig, "resident_indexer_step_recency_patch", 0) != 0);
                compiled.residentIndexerHisaEnabled
                    = getOptionalInt(compiled.runtimeConfig, "resident_indexer_hisa_enabled", 0) != 0;
                compiled.residentIndexerHisaBlockSize
                    = getOptionalInt(compiled.runtimeConfig, "resident_indexer_hisa_block_size", 128);
                compiled.residentIndexerHisaBlockTopK
                    = getOptionalInt(compiled.runtimeConfig, "resident_indexer_hisa_block_topk", 64);
                compiled.residentIndexerHisaMinSeqLen = residentIndexerHisaMinSeqLen(
                    getOptionalInt(compiled.runtimeConfig, "resident_indexer_hisa_min_seq_len", 0));
                compiled.attentionSfVecSize
                    = getOptionalInt(compiled.runtimeConfig, "resident_attention_sf_vec_size", 16);
                compiled.mlpSfVecSize = getOptionalInt(compiled.runtimeConfig, "resident_mlp_sf_vec_size", 16);
                compiled.qScaling = getRequiredDouble(compiled.runtimeScalars, "q_scaling");
                compiled.softmaxScale = getRequiredDouble(compiled.runtimeScalars, "softmax_scale");
                compiled.ropeBase = getRequiredDouble(compiled.runtimeScalars, "rope_base");
                compiled.ropeScale = getRequiredDouble(compiled.runtimeScalars, "rope_scale");
                compiled.ropeShortMScale = getRequiredDouble(compiled.runtimeScalars, "rope_short_m_scale");
                compiled.ropeLongMScale = getRequiredDouble(compiled.runtimeScalars, "rope_long_m_scale");
                compiled.rmsNormEps = getOptionalDouble(compiled.runtimeScalars, "resident_rms_norm_eps", 1e-6);
                compiled.residentIndexerWeightScaleFactor
                    = getRequiredDouble(compiled.runtimeScalars, "resident_indexer_weight_scale_factor");
                compiled.residentIndexerHisaCompressionRatio
                    = getOptionalDouble(compiled.runtimeScalars, "resident_indexer_hisa_compression_ratio", 4.0);
                TORCH_CHECK(compiled.kBProjTrans.is_cuda(), "attention k_b_proj_trans must be a CUDA tensor");
                TORCH_CHECK(compiled.vBProj.is_cuda(), "attention v_b_proj must be a CUDA tensor");
                TORCH_CHECK(compiled.kBProjTrans.dim() == 3, "attention k_b_proj_trans must be 3D");
                TORCH_CHECK(compiled.vBProj.dim() == 3, "attention v_b_proj must be 3D");
                TORCH_CHECK(compiled.vBProj.size(0) == compiled.kBProjTrans.size(0),
                    "attention k_b/v_b projection head counts must match");
                TORCH_CHECK(compiled.vBProj.size(2) == compiled.kBProjTrans.size(1),
                    "attention v_b kv_lora_rank must match k_b output rank");
                TORCH_CHECK(compiled.numHeads == compiled.kBProjTrans.size(0),
                    "DSA runtime num_heads must match k_b projection heads");
                TORCH_CHECK(compiled.numKvHeads == 1, "DSA runtime num_kv_heads must be 1 for MLA");
                TORCH_CHECK(compiled.headDim == compiled.kvLoraRank + compiled.qkRopeHeadDim,
                    "DSA runtime head_dim must equal kv_lora_rank + rope dim");
                TORCH_CHECK(compiled.qLoraRank > 0 && compiled.kvLoraRank == compiled.kBProjTrans.size(1),
                    "DSA runtime q/kv LoRA ranks are invalid");
                TORCH_CHECK(compiled.qkNopeHeadDim == compiled.kBProjTrans.size(2),
                    "DSA runtime qk_nope_head_dim must match k_b K dimension");
                TORCH_CHECK(
                    compiled.vHeadDim == compiled.vBProj.size(1), "DSA runtime v_head_dim must match v_b projection");
                TORCH_CHECK(compiled.rotaryCosSin.is_cuda(), "rotary_cos_sin must be a CUDA tensor");
                TORCH_CHECK(compiled.kvCacheBlockOffsets.is_cuda(), "kv_cache_block_offsets must be a CUDA tensor");
                TORCH_CHECK(compiled.denseKvPool.is_cuda(), "dense_kv_pool must be a CUDA tensor");
                TORCH_CHECK(compiled.kvDispatchMode == kDsaKvDispatchDenseNvfp4
                        || compiled.kvDispatchMode == kDsaKvDispatchStandardMla,
                    "DSA runtime kv_dispatch_mode is invalid");
                if (compiled.kvDispatchMode == kDsaKvDispatchDenseNvfp4)
                {
                    compiled.denseKvScalePool = getRequiredTensor(compiled.runtimeTensors, "dense_kv_scale_pool");
                    TORCH_CHECK(compiled.denseKvScalePool.is_cuda(), "dense_kv_scale_pool must be a CUDA tensor");
                    TORCH_CHECK(
                        compiled.denseKvPool.dim() >= 2, "dense_kv_pool must expose block and layer dimensions");
                    TORCH_CHECK(compiled.denseKvScalePool.dim() >= 2,
                        "dense_kv_scale_pool must expose block and layer dimensions");
                    int64_t const denseKvPages = compiled.denseKvPool.size(0) * compiled.denseKvPool.size(1);
                    TORCH_CHECK(
                        compiled.denseKvPool.numel() == denseKvPages * compiled.tokensPerBlock * compiled.headDim / 2,
                        "dense_kv_pool element count does not match dense NVFP4 dispatch shape");
                    TORCH_CHECK(compiled.denseKvScalePool.numel()
                            == denseKvPages * compiled.tokensPerBlock * compiled.headDim / 16,
                        "dense_kv_scale_pool element count does not match dense NVFP4 dispatch shape");
                    compiled.denseKvPacked = compiled.denseKvPool.reshape(
                        {denseKvPages, compiled.tokensPerBlock, 1, compiled.headDim / 2});
                    compiled.denseKvScales = compiled.denseKvScalePool.reshape(
                        {denseKvPages, compiled.tokensPerBlock, 1, compiled.headDim / 16});
                }
                else
                {
                    compiled.attentionWorkspace = getOptionalTensor(compiled.runtimeTensors, "attention_workspace");
                    compiled.hostTotalKvLens = getOptionalTensor(compiled.runtimeTensors, "host_total_kv_lens");
                    compiled.promptLensCudaRuntime
                        = getOptionalTensor(compiled.runtimeTensors, "prompt_lens_cuda_runtime");
                    compiled.hostRequestTypesRuntime
                        = getOptionalTensor(compiled.runtimeTensors, "host_request_types_runtime");
                    compiled.cacheIndirection = getOptionalTensor(compiled.runtimeTensors, "cache_indirection");
                    compiled.kvScaleOrigQuant = getOptionalTensor(compiled.runtimeTensors, "kv_scale_orig_quant");
                    compiled.kvScaleQuantOrig = getOptionalTensor(compiled.runtimeTensors, "kv_scale_quant_orig");
                    compiled.flashMlaTileSchedulerMetadata
                        = getOptionalTensor(compiled.runtimeTensors, "flash_mla_tile_scheduler_metadata");
                    compiled.flashMlaNumSplits = getOptionalTensor(compiled.runtimeTensors, "flash_mla_num_splits");
                    TORCH_CHECK(compiled.attentionWorkspace.has_value(), "attention_workspace must be defined");
                    TORCH_CHECK(compiled.hostTotalKvLens.has_value(), "host_total_kv_lens must be defined");
                    TORCH_CHECK(compiled.promptLensCudaRuntime.has_value(), "prompt_lens_cuda_runtime must be defined");
                    TORCH_CHECK(
                        compiled.hostRequestTypesRuntime.has_value(), "host_request_types_runtime must be defined");
                }
                at::Tensor const& qBWeight = getLayerTensor(compiled.layerIdx,
                    DeepseekResidentLayerTensorSite::kAttentionQBProjWeight, "attention q_b_proj weight");
                TORCH_CHECK(qBWeight.dim() == 2, "attention q_b_proj weight must be 2D");
                compiled.qWidth = qBWeight.size(0);
                compiled.isDense = mLayerKinds.at(static_cast<size_t>(compiled.layerIdx))
                    == static_cast<int64_t>(DeepseekResidentLayerKind::kDense);
                if (compiled.isDense)
                {
                    compiled.denseIntermediateSize = denseMlpIntermediateSize(compiled.layerIdx);
                }
                else
                {
                    compiled.moeTopK = getRequiredInt(compiled.runtimeConfig, "resident_moe_top_k");
                    compiled.moeNGroup = getRequiredInt(compiled.runtimeConfig, "resident_moe_n_group");
                    compiled.moeTopkGroup = getRequiredInt(compiled.runtimeConfig, "resident_moe_topk_group");
                    compiled.moeRoutedScalingFactor
                        = getRequiredDouble(compiled.runtimeScalars, "resident_moe_routed_scaling_factor");
                    compiled.moeSharedOutputScale
                        = getOptionalDouble(compiled.runtimeScalars, "resident_moe_shared_output_scale", 1.0);
                    compiled.moeNumExperts = moeNumExperts(compiled.layerIdx);
                    compiled.moeLocalExpertOffset
                        = getOptionalInt(compiled.runtimeConfig, "resident_moe_local_expert_offset", 0);
                    compiled.moeLocalNumExperts
                        = getOptionalInt(compiled.runtimeConfig, "resident_moe_local_num_experts", 0);
                    compiled.moeIntermediateSize
                        = getOptionalInt(compiled.runtimeConfig, "resident_moe_intermediate_size", 0);
                    at::Tensor const& sharedGateUpWeight = getLayerTensor(compiled.layerIdx,
                        DeepseekResidentLayerTensorSite::kSharedExpertGateUpWeight, "shared expert gate_up weight");
                    TORCH_CHECK(sharedGateUpWeight.dim() == 2, "shared expert gate_up weight must be 2D");
                    TORCH_CHECK(sharedGateUpWeight.size(0) % 2 == 0, "shared expert gate_up output dim must be even");
                    compiled.moeSharedIntermediateSize = sharedGateUpWeight.size(0) / 2;
                }
                dsaWindowLayers.push_back(std::move(compiled));
            }
        }

        TORCH_CHECK(runDecodeWindowWithDsaPlanReady(), runDecodeWindowWithDsaPlanNotReadyReason());
        TORCH_CHECK(nbPlanLayers == mNbLayers, "DSA window plan must cover every resident layer");

        bool const useDeepGemmIndexer = envFlagEnabled("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_INDEXER_DEEPGEMM");
        int64_t const maxBatchSize = hiddenStatesScratch.size(0);
        int64_t maxQWidth = 0;
        int64_t maxQLoraRank = 0;
        int64_t maxKvLoraRank = 0;
        int64_t maxRopeDim = 0;
        int64_t maxKvAWidth = 0;
        int64_t maxLatentWidth = 0;
        int64_t maxAttentionCoreWidth = 0;
        int64_t maxDenseIntermediateWidth = 0;
        int64_t maxMoeSharedIntermediateWidth = 0;
        int64_t maxMoeExperts = 0;
        int64_t maxMoeTopK = 0;
        {
            ResidentWindowScopedStageTimer const timer(timingEnabled, timing.dimensionScanUs, cudaEventTimingEnabled,
                deviceTiming.dimensionScanUs, cudaEventTimingRecords, timingStream);
            for (auto const& layerPlan : dsaWindowLayers)
            {
                maxQWidth = std::max(maxQWidth, layerPlan.qWidth);
                maxQLoraRank = std::max(maxQLoraRank, layerPlan.qLoraRank);
                maxKvLoraRank = std::max(maxKvLoraRank, layerPlan.kvLoraRank);
                maxRopeDim = std::max(maxRopeDim, layerPlan.qkRopeHeadDim);
                maxKvAWidth
                    = std::max(maxKvAWidth, layerPlan.qLoraRank + layerPlan.kvLoraRank + layerPlan.qkRopeHeadDim);
                maxLatentWidth = std::max(maxLatentWidth, layerPlan.kvLoraRank + layerPlan.qkRopeHeadDim);
                maxAttentionCoreWidth = std::max(maxAttentionCoreWidth, layerPlan.numHeads * layerPlan.vHeadDim);

                if (layerPlan.isDense)
                {
                    maxDenseIntermediateWidth = std::max(maxDenseIntermediateWidth, layerPlan.denseIntermediateSize);
                }
                else
                {
                    maxMoeSharedIntermediateWidth
                        = std::max(maxMoeSharedIntermediateWidth, layerPlan.moeSharedIntermediateSize);
                    maxMoeExperts = std::max(maxMoeExperts, layerPlan.moeNumExperts);
                    maxMoeTopK = std::max(maxMoeTopK, layerPlan.moeTopK);
                }
            }

            TORCH_CHECK(maxQWidth > 0 && maxQLoraRank > 0 && maxKvLoraRank > 0 && maxRopeDim > 0 && maxKvAWidth > 0
                    && maxLatentWidth > 0 && maxAttentionCoreWidth > 0,
                "native DSA window scratch dimensions must be positive");
        }

        at::Tensor normHiddenScratch;
        at::Tensor gatedHiddenScratch;
        at::Tensor attentionHiddenScratch;
        at::Tensor postAttentionNormScratch;
        at::Tensor postAttentionGatedScratch;
        at::Tensor postAttentionResidualScratch;
        at::Tensor denseMlpOutputScratch;
        at::Tensor nextLayerHiddenScratch;
        at::Tensor nextLayerResidualScratch;
        at::Tensor qScratchStorage;
        at::Tensor kvAScratchStorage;
        at::Tensor qLoraScratchStorage;
        at::Tensor compressedKvScratchStorage;
        at::Tensor kPeScratchStorage;
        at::Tensor latentCacheScratchStorage;
        at::Tensor attentionCoreOutputScratchStorage;
        at::Tensor attentionGateScratchStorage;
        at::Tensor attentionGateLogitsScratchStorage;
        at::Tensor denseMlpIntermediateScratchStorage;
        at::Tensor denseMlpGateUpScratchStorage;
        at::Tensor moeSharedIntermediateScratchStorage;
        at::Tensor moeSharedGateUpScratchStorage;
        at::Tensor moeSharedOutputScratchStorage;
        at::Tensor routerLogitsScratchStorage;
        at::Tensor routerScoresScratchStorage;
        at::Tensor routerTopkIndicesScratchStorage;
        at::Tensor routerTopkWeightsScratchStorage;
        std::unique_lock<std::mutex> scratchLock;
        {
            ResidentWindowScopedStageTimer const timer(timingEnabled, timing.scratchAllocUs, cudaEventTimingEnabled,
                deviceTiming.scratchAllocUs, cudaEventTimingRecords, timingStream);
            if (residentWindowScratchCacheEnabled())
            {
                scratchLock = std::unique_lock<std::mutex>(mWindowScratchMutex);
                DeepseekResidentWindowScratch& scratch
                    = getResidentWindowScratch(hiddenStatesScratch, maxBatchSize, maxQWidth, maxQLoraRank,
                        maxKvLoraRank, maxRopeDim, maxKvAWidth, maxLatentWidth, maxAttentionCoreWidth,
                        maxDenseIntermediateWidth, maxMoeSharedIntermediateWidth, maxMoeExperts, maxMoeTopK);
                normHiddenScratch = scratch.normHiddenScratch;
                gatedHiddenScratch = scratch.gatedHiddenScratch;
                attentionHiddenScratch = scratch.attentionHiddenScratch;
                postAttentionNormScratch = scratch.postAttentionNormScratch;
                postAttentionGatedScratch = scratch.postAttentionGatedScratch;
                postAttentionResidualScratch = scratch.postAttentionResidualScratch;
                denseMlpOutputScratch = scratch.denseMlpOutputScratch;
                nextLayerHiddenScratch = scratch.nextLayerHiddenScratch;
                nextLayerResidualScratch = scratch.nextLayerResidualScratch;
                qScratchStorage = scratch.qScratchStorage;
                kvAScratchStorage = scratch.kvAScratchStorage;
                qLoraScratchStorage = scratch.qLoraScratchStorage;
                compressedKvScratchStorage = scratch.compressedKvScratchStorage;
                kPeScratchStorage = scratch.kPeScratchStorage;
                latentCacheScratchStorage = scratch.latentCacheScratchStorage;
                attentionCoreOutputScratchStorage = scratch.attentionCoreOutputScratchStorage;
                attentionGateScratchStorage = scratch.attentionGateScratchStorage;
                attentionGateLogitsScratchStorage = scratch.attentionGateLogitsScratchStorage;
                denseMlpIntermediateScratchStorage = scratch.denseMlpIntermediateScratchStorage;
                denseMlpGateUpScratchStorage = scratch.denseMlpGateUpScratchStorage;
                moeSharedIntermediateScratchStorage = scratch.moeSharedIntermediateScratchStorage;
                moeSharedGateUpScratchStorage = scratch.moeSharedGateUpScratchStorage;
                moeSharedOutputScratchStorage = scratch.moeSharedOutputScratchStorage;
                routerLogitsScratchStorage = scratch.routerLogitsScratchStorage;
                routerScoresScratchStorage = scratch.routerScoresScratchStorage;
                routerTopkIndicesScratchStorage = scratch.routerTopkIndicesScratchStorage;
                routerTopkWeightsScratchStorage = scratch.routerTopkWeightsScratchStorage;
            }
            else
            {
                normHiddenScratch = at::empty_like(hiddenStatesScratch);
                gatedHiddenScratch = at::empty_like(hiddenStatesScratch);
                attentionHiddenScratch = at::empty_like(hiddenStatesScratch);
                postAttentionNormScratch = at::empty_like(hiddenStatesScratch);
                postAttentionGatedScratch = at::empty_like(hiddenStatesScratch);
                postAttentionResidualScratch = at::empty_like(hiddenStatesScratch);
                denseMlpOutputScratch = at::empty_like(hiddenStatesScratch);
                nextLayerHiddenScratch = at::empty_like(hiddenStatesScratch);
                nextLayerResidualScratch = at::empty_like(hiddenStatesScratch);
                qScratchStorage = at::empty({maxBatchSize, maxQWidth}, hiddenStatesScratch.options());
                kvAScratchStorage = at::empty({maxBatchSize, maxKvAWidth}, hiddenStatesScratch.options());
                qLoraScratchStorage = at::empty({maxBatchSize, maxQLoraRank}, hiddenStatesScratch.options());
                compressedKvScratchStorage = at::empty({maxBatchSize, maxKvLoraRank}, hiddenStatesScratch.options());
                kPeScratchStorage = at::empty({maxBatchSize, maxRopeDim}, hiddenStatesScratch.options());
                latentCacheScratchStorage = at::empty({maxBatchSize, maxLatentWidth}, hiddenStatesScratch.options());
                attentionCoreOutputScratchStorage
                    = at::empty({maxBatchSize, maxAttentionCoreWidth}, hiddenStatesScratch.options());
                attentionGateScratchStorage
                    = at::empty({maxBatchSize, maxAttentionCoreWidth}, hiddenStatesScratch.options());
                attentionGateLogitsScratchStorage = at::empty(
                    {maxBatchSize, maxAttentionCoreWidth}, hiddenStatesScratch.options().dtype(at::ScalarType::Float));
                if (maxDenseIntermediateWidth > 0)
                {
                    denseMlpIntermediateScratchStorage
                        = at::empty({maxBatchSize, maxDenseIntermediateWidth}, hiddenStatesScratch.options());
                    denseMlpGateUpScratchStorage = at::empty({maxBatchSize, maxDenseIntermediateWidth * 2},
                        hiddenStatesScratch.options().dtype(at::ScalarType::Float));
                }
                if (maxMoeSharedIntermediateWidth > 0)
                {
                    moeSharedIntermediateScratchStorage
                        = at::empty({maxBatchSize, maxMoeSharedIntermediateWidth}, hiddenStatesScratch.options());
                    moeSharedGateUpScratchStorage = at::empty({maxBatchSize, maxMoeSharedIntermediateWidth * 2},
                        hiddenStatesScratch.options().dtype(at::ScalarType::Float));
                    moeSharedOutputScratchStorage = at::empty({maxBatchSize, hiddenStatesScratch.size(1)},
                        hiddenStatesScratch.options().dtype(at::ScalarType::Float));
                }
                if (maxMoeExperts > 0)
                {
                    routerLogitsScratchStorage = at::empty(
                        {maxBatchSize, maxMoeExperts}, hiddenStatesScratch.options().dtype(at::ScalarType::Float));
                    routerScoresScratchStorage = at::empty(
                        {maxBatchSize, maxMoeExperts}, hiddenStatesScratch.options().dtype(at::ScalarType::Float));
                }
                if (maxMoeTopK > 0)
                {
                    routerTopkIndicesScratchStorage = at::empty(
                        {maxBatchSize, maxMoeTopK}, hiddenStatesScratch.options().dtype(at::ScalarType::Int));
                    routerTopkWeightsScratchStorage = at::empty(
                        {maxBatchSize, maxMoeTopK}, hiddenStatesScratch.options().dtype(at::ScalarType::Float));
                }
            }
        }

        for (int64_t outputStepIdx = 1; outputStepIdx < ownedSteps; ++outputStepIdx)
        {
            ++timing.decodeSteps;
            int64_t const windowDecodeStep = outputStepIdx - 1;
            {
                ResidentWindowScopedStageTimer const timer(timingEnabled, timing.prepareUs, cudaEventTimingEnabled,
                    deviceTiming.prepareUs, cudaEventTimingRecords, timingStream);
                static_cast<void>(runDecodeWindowPrepareStep(initialTokens, windowTokensScratch, inputIdsScratch,
                    hiddenStatesScratch, positionIds, kvLensCuda, outputStepIdx, inputTokens));
            }

            at::Tensor currentHiddenScratch = hiddenStatesScratch;
            at::Tensor currentResidualScratch = hiddenStatesScratch;
            at::Tensor cachedIndexerTopk;
            bool hasCachedIndexerTopk = false;
            for (int64_t planIdx = 0; planIdx < nbPlanLayers; ++planIdx)
            {
                ++timing.layerVisits;
                auto& layerPlan = dsaWindowLayers.at(static_cast<size_t>(planIdx));
                int64_t const layerIdx = layerPlan.layerIdx;
                int64_t const numSeqs = layerPlan.numSeqs;
                int64_t const numContexts = layerPlan.numContexts;
                int64_t const numCtxTokens = layerPlan.numCtxTokens;
                int64_t const numGenerations = layerPlan.numGenerations;
                int64_t const numSparseTopk = layerPlan.numSparseTopk;
                int64_t const numHeads = layerPlan.numHeads;
                int64_t const qLoraRank = layerPlan.qLoraRank;
                int64_t const kvLoraRank = layerPlan.kvLoraRank;
                int64_t const qkRopeHeadDim = layerPlan.qkRopeHeadDim;
                int64_t const vHeadDim = layerPlan.vHeadDim;
                int64_t const kvDispatchMode = layerPlan.kvDispatchMode;
                int64_t const maxSeqLen = layerPlan.maxSeqLen;
                int64_t const indexerLogitsWidth
                    = liveWindowIndexerLogitsWidth(cachedTokens, inputTokens, ownedSteps, maxSeqLen);
                int64_t const indexerMaxLiveKvLen = liveWindowIndexerMaxKvLen(cachedTokens, inputTokens, ownedSteps);
                int64_t const tokensPerBlock = layerPlan.tokensPerBlock;
                int64_t const residentIndexerHeadDim = layerPlan.residentIndexerHeadDim;
                int64_t const residentIndexerNumHeads = layerPlan.residentIndexerNumHeads;
                int64_t const residentIndexerRopeDim = layerPlan.residentIndexerRopeDim;
                int64_t const residentIndexerQuantBlockSize = layerPlan.residentIndexerQuantBlockSize;
                int64_t const residentIndexerDataBytesPerToken = layerPlan.residentIndexerDataBytesPerToken;
                bool const reusePreviousIndexerTopk = layerPlan.reusePreviousIndexerTopk;
                int64_t const residentIndexerStepFreq = layerPlan.residentIndexerStepFreq;
                bool const residentIndexerStepRecencyPatch = layerPlan.residentIndexerStepRecencyPatch;
                bool const residentIndexerHisaEnabled = layerPlan.residentIndexerHisaEnabled;
                int64_t const residentIndexerHisaBlockSize = layerPlan.residentIndexerHisaBlockSize;
                int64_t const residentIndexerHisaBlockTopK = layerPlan.residentIndexerHisaBlockTopK;
                int64_t const residentIndexerHisaMinSeqLen = layerPlan.residentIndexerHisaMinSeqLen;
                int64_t const attentionSfVecSize = layerPlan.attentionSfVecSize;
                int64_t const mlpSfVecSize = layerPlan.mlpSfVecSize;
                double const rmsNormEps = layerPlan.rmsNormEps;
                double const residentIndexerWeightScaleFactor = layerPlan.residentIndexerWeightScaleFactor;
                double const residentIndexerHisaCompressionRatio = layerPlan.residentIndexerHisaCompressionRatio;

                TORCH_CHECK(numContexts == 0 && numCtxTokens == 0,
                    "native DSA window body currently supports decode-only windows");
                TORCH_CHECK(numGenerations == numSeqs && inputTokens == numSeqs,
                    "native DSA window body currently supports one decode token per sequence");
                TORCH_CHECK(numSparseTopk > 0, "native DSA window body requires positive num_sparse_topk");
                TORCH_CHECK(kvDispatchMode == kDsaKvDispatchDenseNvfp4 || kvDispatchMode == kDsaKvDispatchStandardMla,
                    "native DSA window body kv_dispatch_mode is invalid");
                TORCH_CHECK(indexerLogitsWidth > 0 && indexerLogitsWidth <= maxSeqLen,
                    "native DSA window body indexer logits width is invalid");
                TORCH_CHECK(residentIndexerHeadDim == 128 && residentIndexerRopeDim == qkRopeHeadDim,
                    "native DSA window body currently supports only the DeepSeek FP4 indexer shape");
                TORCH_CHECK(
                    residentIndexerNumHeads > 0, "native DSA window body requires positive resident_indexer_num_heads");
                TORCH_CHECK(residentIndexerQuantBlockSize > 0 && residentIndexerDataBytesPerToken > 0,
                    "native DSA window body indexer cache layout is invalid");
                TORCH_CHECK(residentIndexerHisaBlockSize > 0 && residentIndexerHisaBlockTopK > 0,
                    "native DSA window body HISA config is invalid");
                TORCH_CHECK(residentIndexerHisaCompressionRatio >= 0.0,
                    "native DSA window body HISA compression ratio must be non-negative");
                timing.indexerLogitsWidth = std::max(timing.indexerLogitsWidth, indexerLogitsWidth);
                timing.indexerMaxSeqLen = std::max(timing.indexerMaxSeqLen, maxSeqLen);

                at::Tensor layerInputScratch;
                {
                    ResidentWindowScopedStageTimer const timer(timingEnabled, timing.inputNormUs,
                        cudaEventTimingEnabled, deviceTiming.inputNormUs, cudaEventTimingRecords, timingStream);
                    if (planIdx == 0)
                    {
                        static_cast<void>(runLayerInputRmsNorm(
                            layerIdx, currentHiddenScratch, normHiddenScratch, inputTokens, rmsNormEps, false));
                        layerInputScratch = normHiddenScratch;
                    }
                    else
                    {
                        layerInputScratch = currentHiddenScratch;
                    }
                    static_cast<void>(
                        runLayerInputGatedNorm(layerIdx, layerInputScratch, gatedHiddenScratch, inputTokens));
                }

                int64_t const qWidth = layerPlan.qWidth;
                at::Tensor qScratch = qScratchStorage.narrow(1, 0, qWidth);
                at::Tensor kvAScratch = kvAScratchStorage.narrow(1, 0, qLoraRank + kvLoraRank + qkRopeHeadDim);
                at::Tensor qLoraScratch = qLoraScratchStorage.narrow(1, 0, qLoraRank);
                at::Tensor compressedKvScratch = compressedKvScratchStorage.narrow(1, 0, kvLoraRank);
                at::Tensor kPeScratch = kPeScratchStorage.narrow(1, 0, qkRopeHeadDim);
                at::Tensor latentCacheScratch = latentCacheScratchStorage.narrow(1, 0, kvLoraRank + qkRopeHeadDim);
                at::Tensor qFp4Scratch = layerPlan.qFp4Scratch;
                at::Tensor kFp4Scratch = layerPlan.kFp4Scratch;
                at::Tensor kScaleScratch = layerPlan.kScaleScratch;
                at::Tensor indexerWeightsScratch = layerPlan.indexerWeightsScratch;
                at::Tensor qScaleScratch = layerPlan.qScaleScratch;
                at::Tensor topkIndices = layerPlan.topkIndices;
                at::Tensor xstepRefreshEnd = layerPlan.xstepRefreshEnd;
                std::optional<at::Tensor> precomputedWqBScratch;
                std::optional<at::Tensor> precomputedIndexerKScratch;
                std::optional<at::Tensor> precomputedIndexerWeightsScratch;
                if (!reusePreviousIndexerTopk && residentFusedQbWqBEnabled())
                {
                    at::Tensor fusedQFlat
                        = layerPlan.fusedQScratch.reshape({layerPlan.fusedQScratch.size(0), -1});
                    int64_t const precomputedWqBWidth = residentIndexerNumHeads * residentIndexerHeadDim;
                    if (fusedQFlat.size(1) >= precomputedWqBWidth)
                    {
                        precomputedWqBScratch = fusedQFlat.narrow(1, 0, precomputedWqBWidth);
                    }
                }
                if (!reusePreviousIndexerTopk && residentFusedKvAWkWpEnabled()
                    && kvAScratch.size(1) >= residentIndexerHeadDim)
                {
                    precomputedIndexerKScratch = kvAScratch.narrow(1, 0, residentIndexerHeadDim);
                    precomputedIndexerWeightsScratch = indexerWeightsScratch;
                }
                DeepseekResidentDsaAttentionProjectionResult attentionProjectionResult;
                {
                    ResidentWindowScopedStageTimer const timer(timingEnabled, timing.attentionProjectionUs,
                        cudaEventTimingEnabled, deviceTiming.attentionProjectionUs, cudaEventTimingRecords,
                        timingStream);
                    attentionProjectionResult = runLayerDsaAttentionProjectionImpl(layerIdx, gatedHiddenScratch,
                        qScratch, kvAScratch, qLoraScratch, compressedKvScratch, kPeScratch, latentCacheScratch,
                        precomputedWqBScratch, precomputedIndexerKScratch, precomputedIndexerWeightsScratch,
                        inputTokens, qLoraRank, kvLoraRank, qkRopeHeadDim, rmsNormEps, attentionSfVecSize,
                        "cutlass,cublaslt,cuda_core");
                }

                {
                    ResidentWindowScopedStageTimer const timer(timingEnabled, timing.metadataRefreshUs,
                        cudaEventTimingEnabled, deviceTiming.metadataRefreshUs, cudaEventTimingRecords, timingStream);
                    static_cast<void>(
                        runDecodeWindowAttentionMetadataDeviceRefresh(seqLensCuda, kvLensCuda, layerPlan.reqIdxPerToken,
                            layerPlan.indexerKCacheBlockOffsets, layerPlan.slotMappingFp8, layerPlan.slotMappingScale,
                            layerPlan.genKvIndptr, layerPlan.genCachedTokenIndptr, layerPlan.kvLensCuda2d, inputTokens,
                            numSeqs, numContexts, numGenerations, residentIndexerHeadDim, tokensPerBlock,
                            residentIndexerQuantBlockSize, residentIndexerDataBytesPerToken));
                }

                at::Tensor topkIndicesForDispatch = topkIndices;
                if (reusePreviousIndexerTopk)
                {
                    TORCH_CHECK(hasCachedIndexerTopk,
                        "native DSA window TopK reuse layer reached before an indexer producer layer");
                    TORCH_CHECK(cachedIndexerTopk.dim() == 2, "cached DSA indexer TopK must be 2D");
                    TORCH_CHECK(cachedIndexerTopk.size(0) >= inputTokens,
                        "cached DSA indexer TopK batch is smaller than input");
                    TORCH_CHECK(cachedIndexerTopk.size(1) == numSparseTopk,
                        "cached DSA indexer TopK width must match num_sparse_topk");
                    topkIndicesForDispatch = cachedIndexerTopk;
                    ++timing.indexerTopkReuseLayerVisits;
                }
                else
                {
                    bool const xstepReuseEnabled = residentIndexerStepFreq > 1 && numContexts == 0
                        && numGenerations == inputTokens && numGenerations == numSeqs;
                    bool const xstepReuseStep
                        = xstepReuseEnabled && ((windowDecodeStep % residentIndexerStepFreq) != 0);
                    at::Tensor contextLens = kvLensCuda.narrow(0, numContexts, numGenerations);
                    if (!contextLens.is_contiguous())
                    {
                        contextLens = contextLens.contiguous();
                    }
                    if (xstepReuseStep)
                    {
                        if (residentIndexerStepRecencyPatch)
                        {
                            int64_t const maxDelta = std::min<int64_t>(residentIndexerStepFreq - 1, numSparseTopk);
                            static_cast<void>(runIndexerXstepRecencyPatch(
                                topkIndices, xstepRefreshEnd, contextLens, /*nextN=*/1, maxDelta));
                        }
                        topkIndicesForDispatch = topkIndices;
                        cachedIndexerTopk = topkIndices;
                        hasCachedIndexerTopk = true;
                        ++timing.indexerTopkReuseLayerVisits;
                        ++timing.indexerTopkXstepReuseLayerVisits;
                    }
                    else
                    {
                        {
                            ResidentWindowScopedStageTimer const timer(timingEnabled, timing.indexerProjectionUs,
                                cudaEventTimingEnabled, deviceTiming.indexerProjectionUs, cudaEventTimingRecords,
                                timingStream);
                            std::optional<at::Tensor> precomputedWqB
                                = attentionProjectionResult.hasPrecomputedWqB ? precomputedWqBScratch : std::nullopt;
                            std::optional<at::Tensor> precomputedIndexerK
                                = attentionProjectionResult.hasPrecomputedIndexerKWeights
                                ? precomputedIndexerKScratch
                                : std::nullopt;
                            std::optional<at::Tensor> precomputedIndexerWeights
                                = attentionProjectionResult.hasPrecomputedIndexerKWeights
                                ? precomputedIndexerWeightsScratch
                                : std::nullopt;
                            static_cast<void>(runLayerDsaIndexerFp4ProjectionImpl(layerIdx, qLoraScratch,
                                gatedHiddenScratch, positionIds, qFp4Scratch, kFp4Scratch, kScaleScratch,
                                indexerWeightsScratch, qScaleScratch, precomputedWqB,
                                attentionProjectionResult.precomputedWqBScale, precomputedIndexerK,
                                precomputedIndexerWeights, inputTokens, residentIndexerNumHeads,
                                residentIndexerHeadDim, residentIndexerRopeDim, rmsNormEps,
                                residentIndexerWeightScaleFactor, attentionSfVecSize, "cutlass,cublaslt,cuda_core"));
                        }
                        {
                            ResidentWindowScopedStageTimer const timer(timingEnabled, timing.indexerScatterUs,
                                cudaEventTimingEnabled, deviceTiming.indexerScatterUs, cudaEventTimingRecords,
                                timingStream);
                            static_cast<void>(
                                runIndexerKCacheScatter(kFp4Scratch, kScaleScratch, layerPlan.indexerKCache,
                                    layerPlan.slotMappingFp8, layerPlan.slotMappingScale, inputTokens));
                            if (layerPlan.indexerHisaPageReps.has_value()
                                && layerPlan.indexerHisaPageCounts.has_value())
                            {
                                callIndexerHisaUpdatePageRepsNvfp4(layerPlan.indexerKCache,
                                    layerPlan.indexerHisaPageReps.value(), layerPlan.indexerHisaPageCounts.value(),
                                    layerPlan.slotMappingFp8, inputTokens);
                            }
                        }

                        at::Tensor qDecode;
                        at::Tensor qScaleDecode;
                        at::Tensor weightsDecode;
                        at::Tensor blockTableGen;
                        {
                            ResidentWindowScopedStageTimer const timer(timingEnabled, timing.indexerPackUs,
                                cudaEventTimingEnabled, deviceTiming.indexerPackUs, cudaEventTimingRecords,
                                timingStream);
                            qDecode
                                = qFp4Scratch.narrow(0, 0, inputTokens)
                                      .reshape({inputTokens, 1, residentIndexerNumHeads, residentIndexerHeadDim / 2})
                                      .contiguous();
                            if (!useDeepGemmIndexer && qDecode.scalar_type() != at::ScalarType::Byte)
                            {
                                qDecode = qDecode.to(at::ScalarType::Byte);
                            }
                            qScaleDecode = qScaleScratch.narrow(0, 0, inputTokens)
                                               .reshape({inputTokens, residentIndexerNumHeads})
                                               .reshape({inputTokens, 1, residentIndexerNumHeads})
                                               .contiguous();
                            weightsDecode = indexerWeightsScratch.narrow(0, 0, inputTokens).contiguous();
                            blockTableGen = layerPlan.blockTable.narrow(0, numContexts, numGenerations);
                        }
                        {
                            ResidentWindowScopedStageTimer const timer(timingEnabled, timing.indexerLogitsTopkUs,
                                cudaEventTimingEnabled, deviceTiming.indexerLogitsTopkUs, cudaEventTimingRecords,
                                timingStream);
                            bool usedDenseTopk = false;
                            bool usedNativeHisa = false;
                            if (indexerMaxLiveKvLen <= numSparseTopk)
                            {
                                static_cast<void>(
                                    runIndexerDenseTopkDecode(contextLens, topkIndices, inputTokens, numSparseTopk));
                                usedDenseTopk = true;
                                ++timing.indexerTopkDenseLayerVisits;
                            }
#if defined(TRTLLM_OPTRT_ENABLE_DEEP_GEMM_RESIDENT_INDEXER)
                            bool const nativeHisaCandidate = !usedDenseTopk && residentIndexerHisaEnabled
                                && layerPlan.indexerHisaPageReps.has_value()
                                && layerPlan.indexerHisaPageCounts.has_value() && residentIndexerHisaBlockSize == 128
                                && tokensPerBlock > 0 && (residentIndexerHisaBlockSize % tokensPerBlock) == 0
                                && indexerMaxLiveKvLen >= residentIndexerHisaMinSeqLen && residentIndexerHeadDim == 128
                                && residentIndexerDataBytesPerToken == 64;
                            if (nativeHisaCandidate)
                            {
                                int64_t const maxHisaBlocks
                                    = ceilDivPositive(indexerMaxLiveKvLen, residentIndexerHisaBlockSize);
                                int64_t const minHisaBlocks
                                    = ceilDivPositive(numSparseTopk, residentIndexerHisaBlockSize);
                                int64_t blockTopK = residentIndexerHisaBlockTopK;
                                if (residentIndexerHisaCompressionRatio > 0.0)
                                {
                                    blockTopK = static_cast<int64_t>(std::ceil(
                                        static_cast<double>(maxHisaBlocks) / residentIndexerHisaCompressionRatio));
                                }
                                blockTopK = std::min(std::max(blockTopK, minHisaBlocks), maxHisaBlocks);
                                int64_t const candidateLen = blockTopK * residentIndexerHisaBlockSize;
                                if (maxHisaBlocks > 0 && blockTopK > 0 && candidateLen >= numSparseTopk)
                                {
                                    at::Tensor blockTableGenContiguous
                                        = blockTableGen.is_contiguous() ? blockTableGen : blockTableGen.contiguous();
                                    at::Tensor blockCounts
                                        = callIndexerHisaBlockCounts(contextLens, residentIndexerHisaBlockSize);
                                    at::Tensor blockReps = callIndexerHisaBlockRepsFromPagesNvfp4(
                                        layerPlan.indexerHisaPageReps.value(), layerPlan.indexerHisaPageCounts.value(),
                                        blockTableGenContiguous, contextLens, maxHisaBlocks, tokensPerBlock);
                                    at::Tensor qValuesFlat = qDecode
                                                                 .reshape({inputTokens, residentIndexerNumHeads,
                                                                     residentIndexerHeadDim / 2})
                                                                 .contiguous();
                                    at::Tensor qScaleFlat
                                        = qScaleDecode.reshape({inputTokens, residentIndexerNumHeads}).contiguous();
                                    at::Tensor blockScores = callIndexerHisaBlockScoresNvfp4(qValuesFlat, qScaleFlat,
                                        weightsDecode, blockReps, contextLens, blockTopK, /*nextN=*/1,
                                        residentIndexerHisaBlockSize);
                                    at::Tensor topBlocks = at::empty({inputTokens, blockTopK}, topkIndices.options());
                                    static_cast<void>(runIndexerTopkDecode(
                                        blockScores, blockCounts, topBlocks, /*nextN=*/1, blockTopK));

                                    int64_t const pagesPerHisaBlock = residentIndexerHisaBlockSize / tokensPerBlock;
                                    at::Tensor candidatePageTable = callIndexerHisaCandidatePages(
                                        topBlocks, blockTableGenContiguous, /*nextN=*/1, pagesPerHisaBlock);
                                    at::Tensor candidateContextLens
                                        = at::empty({inputTokens, 1}, contextLens.options());
                                    candidateContextLens.fill_(candidateLen);
                                    int64_t const numSms
                                        = std::max<int64_t>(layerPlan.schedulerMetadataBuffer.size(0) - 1, 1);
                                    at::Tensor candidateSchedule = callDeepGemmPagedMqaLogitsMetadata(
                                        candidateContextLens, /*blockKv=*/64, numSms);
                                    at::Tensor candidateScores = callDeepGemmFp4PagedMqaLogits(qDecode, qScaleDecode,
                                        layerPlan.indexerKCache, weightsDecode, candidateContextLens,
                                        candidatePageTable, candidateSchedule, candidateLen, at::ScalarType::Float);
                                    callIndexerHisaMaskScores(
                                        candidateScores, topBlocks, contextLens, residentIndexerHisaBlockSize);
                                    at::Tensor selected
                                        = at::empty({inputTokens, numSparseTopk}, topkIndices.options());
                                    at::Tensor selectedLengths = at::empty({inputTokens}, contextLens.options());
                                    selectedLengths.fill_(candidateLen);
                                    static_cast<void>(runIndexerTopkDecode(
                                        candidateScores, selectedLengths, selected, /*nextN=*/1, numSparseTopk));
                                    at::Tensor remapped = callIndexerHisaRemapSelected(
                                        selected, topBlocks, contextLens, residentIndexerHisaBlockSize, numSparseTopk);
                                    topkIndices.narrow(0, 0, inputTokens).copy_(remapped);
                                    usedNativeHisa = true;
                                    ++timing.indexerTopkHisaLayerVisits;
                                }
                            }
#endif
                            if (!usedDenseTopk && !usedNativeHisa)
                            {
                                at::Tensor indexerLogits = useDeepGemmIndexer
                                    ? callDeepGemmFp4PagedMqaLogits(qDecode, qScaleDecode, layerPlan.indexerKCache,
                                          weightsDecode, contextLens, blockTableGen, layerPlan.schedulerMetadataBuffer,
                                          indexerLogitsWidth)
                                    : callTrtllmCuteDslFp4PagedMqaLogits(qDecode, qScaleDecode, layerPlan.indexerKCache,
                                          weightsDecode, contextLens, blockTableGen, layerPlan.schedulerMetadataBuffer,
                                          indexerLogitsWidth);
                                static_cast<void>(runIndexerTopkDecode(
                                    indexerLogits, contextLens, topkIndices, /*nextN=*/1, numSparseTopk));
                            }
                        }
                        if (xstepReuseEnabled && residentIndexerStepRecencyPatch)
                        {
                            xstepRefreshEnd.narrow(0, 0, numGenerations).copy_(contextLens);
                        }
                        cachedIndexerTopk = topkIndices;
                        hasCachedIndexerTopk = true;
                        ++timing.indexerTopkComputeLayerVisits;
                    }
                }
                at::Tensor attentionCoreOutputScratch
                    = attentionCoreOutputScratchStorage.narrow(1, 0, numHeads * vHeadDim);
                {
                    ResidentWindowScopedStageTimer const timer(timingEnabled, timing.attentionDispatchUs,
                        cudaEventTimingEnabled, deviceTiming.attentionDispatchUs, cudaEventTimingRecords, timingStream);
                    static_cast<void>(runLayerDsaAttentionDispatchCompiled(layerPlan, qScratch, compressedKvScratch,
                        kPeScratch, latentCacheScratch, positionIds, seqLensCuda, kvLensCuda, topkIndicesForDispatch,
                        attentionCoreOutputScratch, inputTokens));
                }

                at::Tensor attentionGateScratch = attentionGateScratchStorage.narrow(1, 0, numHeads * vHeadDim);
                at::Tensor attentionGateLogitsScratch
                    = attentionGateLogitsScratchStorage.narrow(1, 0, numHeads * vHeadDim);
                {
                    ResidentWindowScopedStageTimer const timer(timingEnabled, timing.attentionTailUs,
                        cudaEventTimingEnabled, deviceTiming.attentionTailUs, cudaEventTimingRecords, timingStream);
                    static_cast<void>(runLayerAttentionOutputTailWithGateLogits(layerIdx, attentionCoreOutputScratch,
                        gatedHiddenScratch, attentionGateScratch, attentionGateLogitsScratch, attentionHiddenScratch,
                        inputTokens));
                }
                {
                    ResidentWindowScopedStageTimer const timer(timingEnabled, timing.postAttentionNormUs,
                        cudaEventTimingEnabled, deviceTiming.postAttentionNormUs, cudaEventTimingRecords, timingStream);
                    if (!tryRunLayerPostAttentionRmsNormLowRankGate(layerIdx, attentionHiddenScratch,
                            currentResidualScratch, postAttentionGatedScratch, postAttentionResidualScratch,
                            inputTokens, rmsNormEps, false))
                    {
                        static_cast<void>(runLayerPostAttentionRmsNorm(layerIdx, attentionHiddenScratch,
                            currentResidualScratch, postAttentionNormScratch, postAttentionResidualScratch, inputTokens,
                            rmsNormEps, false));
                        static_cast<void>(runLayerPostAttentionGatedNorm(
                            layerIdx, postAttentionNormScratch, postAttentionGatedScratch, inputTokens));
                    }
                }

                if (layerPlan.isDense)
                {
                    ++timing.denseLayerVisits;
                    at::Tensor denseMlpIntermediateScratch
                        = denseMlpIntermediateScratchStorage.narrow(1, 0, layerPlan.denseIntermediateSize);
                    at::Tensor denseMlpGateUpScratch
                        = denseMlpGateUpScratchStorage.narrow(1, 0, layerPlan.denseIntermediateSize * 2);
                    {
                        ResidentWindowScopedStageTimer const timer(timingEnabled, timing.denseMlpUs,
                            cudaEventTimingEnabled, deviceTiming.denseMlpUs, cudaEventTimingRecords, timingStream);
                        static_cast<void>(runLayerDenseMlpWithGateUpScratch(layerIdx, postAttentionGatedScratch,
                            denseMlpIntermediateScratch, denseMlpGateUpScratch, denseMlpOutputScratch, inputTokens));
                    }
                }
                else
                {
                    ++timing.moeLayerVisits;
                    int64_t const topK = layerPlan.moeTopK;
                    int64_t const nGroup = layerPlan.moeNGroup;
                    int64_t const topkGroup = layerPlan.moeTopkGroup;
                    double const routedScalingFactor = layerPlan.moeRoutedScalingFactor;
                    double const sharedOutputScale = layerPlan.moeSharedOutputScale;
                    int64_t const numExperts = layerPlan.moeNumExperts;
                    int64_t const localExpertOffset = layerPlan.moeLocalExpertOffset;
                    int64_t const localNumExperts = layerPlan.moeLocalNumExperts;
                    int64_t const moeIntermediateSize = layerPlan.moeIntermediateSize;
                    at::Tensor routerLogitsScratch = routerLogitsScratchStorage.narrow(1, 0, numExperts);
                    at::Tensor routerScoresScratch = routerScoresScratchStorage.narrow(1, 0, numExperts);
                    at::Tensor routerTopkIndicesScratch = routerTopkIndicesScratchStorage.narrow(1, 0, topK);
                    at::Tensor routerTopkWeightsScratch = routerTopkWeightsScratchStorage.narrow(1, 0, topK);
                    at::Tensor moeSharedIntermediateScratch
                        = moeSharedIntermediateScratchStorage.narrow(1, 0, layerPlan.moeSharedIntermediateSize);
                    at::Tensor moeSharedGateUpScratch
                        = moeSharedGateUpScratchStorage.narrow(1, 0, layerPlan.moeSharedIntermediateSize * 2);
                    at::Tensor moeSharedOutputScratch
                        = moeSharedOutputScratchStorage.narrow(1, 0, postAttentionGatedScratch.size(1));
                    if (residentMoeRawRoutingEnabled())
                    {
                        {
                            ResidentWindowScopedStageTimer const timer(timingEnabled, timing.moeRouterUs,
                                cudaEventTimingEnabled, deviceTiming.moeRouterUs, cudaEventTimingRecords, timingStream);
                            static_cast<void>(runLayerMoeRouterLogits(
                                layerIdx, postAttentionGatedScratch, routerLogitsScratch, inputTokens));
                        }
                        {
                            ResidentMoeExpertTiming moeExpertTiming;
                            ResidentWindowScopedStageTimer const timer(timingEnabled, timing.moeExpertsUs,
                                cudaEventTimingEnabled, deviceTiming.moeExpertsUs, cudaEventTimingRecords,
                                timingStream);
                            static_cast<void>(runLayerMoeExpertsFromRoutingLogitsImpl(layerIdx,
                                postAttentionGatedScratch, routerLogitsScratch, denseMlpOutputScratch,
                                moeSharedIntermediateScratch, moeSharedGateUpScratch, moeSharedOutputScratch,
                                inputTokens, topK, nGroup, topkGroup, routedScalingFactor, mlpSfVecSize,
                                "cutlass,cublaslt,cuda_core", sharedOutputScale, numExperts, localExpertOffset,
                                localNumExperts, moeIntermediateSize, timingEnabled ? &moeExpertTiming : nullptr));
                            timing.moeSharedExpertUs += moeExpertTiming.sharedExpertUs;
                            timing.moeRoutedExpertUs += moeExpertTiming.routedExpertUs;
                            timing.moeCombineUs += moeExpertTiming.combineUs;
                        }
                    }
                    else
                    {
                        {
                            ResidentWindowScopedStageTimer const timer(timingEnabled, timing.moeRouterUs,
                                cudaEventTimingEnabled, deviceTiming.moeRouterUs, cudaEventTimingRecords, timingStream);
                            static_cast<void>(runLayerMoeRouter(layerIdx, postAttentionGatedScratch,
                                routerLogitsScratch, routerScoresScratch, routerTopkIndicesScratch,
                                routerTopkWeightsScratch, inputTokens, topK, nGroup, topkGroup, routedScalingFactor));
                        }
                        {
                            ResidentWindowScopedStageTimer const timer(timingEnabled, timing.moeExpertsUs,
                                cudaEventTimingEnabled, deviceTiming.moeExpertsUs, cudaEventTimingRecords,
                                timingStream);
                            static_cast<void>(runLayerMoeExpertsImpl(layerIdx, postAttentionGatedScratch,
                                routerTopkIndicesScratch, routerTopkWeightsScratch, denseMlpOutputScratch,
                                moeSharedIntermediateScratch, moeSharedGateUpScratch, moeSharedOutputScratch,
                                inputTokens, mlpSfVecSize, "cutlass,cublaslt,cuda_core", sharedOutputScale, numExperts,
                                localExpertOffset, localNumExperts, moeIntermediateSize));
                        }
                    }
                }

                {
                    ResidentWindowScopedStageTimer const timer(timingEnabled, timing.postFfnNormUs,
                        cudaEventTimingEnabled, deviceTiming.postFfnNormUs, cudaEventTimingRecords, timingStream);
                    static_cast<void>(
                        runLayerPostFfnRmsNorm(layerIdx, denseMlpOutputScratch, postAttentionResidualScratch,
                            nextLayerHiddenScratch, nextLayerResidualScratch, inputTokens, rmsNormEps, false));
                }
                currentHiddenScratch = nextLayerHiddenScratch;
                currentResidualScratch = nextLayerResidualScratch;
            }

            {
                ResidentWindowScopedStageTimer const timer(timingEnabled, timing.lmHeadUs, cudaEventTimingEnabled,
                    deviceTiming.lmHeadUs, cudaEventTimingRecords, timingStream);
                static_cast<void>(runLmHeadLogits(currentHiddenScratch, logitsScratch, inputTokens));
            }
            {
                ResidentWindowScopedStageTimer const timer(timingEnabled, timing.sampleUs, cudaEventTimingEnabled,
                    deviceTiming.sampleUs, cudaEventTimingRecords, timingStream);
                static_cast<void>(
                    runDecodeWindowSampleStep(logitsScratch, windowTokensScratch, outputStepIdx, inputTokens));
            }
        }
        double const totalUs = residentWindowElapsedUs(totalStart, ResidentWindowTimingClock::now());
        if (cudaEventTimingEnabled)
        {
            finalizeResidentWindowCudaEventTiming(cudaEventTimingRecords);
        }
        if (timingEnabled || cudaEventTimingEnabled)
        {
            std::cout << "OPTRT_DEEPSEEK_RESIDENT_WINDOW_CPP_TIMING {"
                      << "'rank': " << distributedRank() << ", 'owned_steps': " << ownedSteps
                      << ", 'decode_steps': " << timing.decodeSteps << ", 'input_tokens': " << inputTokens
                      << ", 'layers': " << nbPlanLayers << ", 'layer_visits': " << timing.layerVisits
                      << ", 'attention_tail_fp4out_gate_visits': "
                      << residentAttentionTailFp4OutGateVisits() - attentionTailFp4OutGateVisitsStart
                      << ", 'attention_tail_fp4out_gate_attempts': "
                      << attentionTailFp4OutGateStatDelta(ResidentAttentionTailFp4OutGateStat::kAttempts)
                      << ", 'attention_tail_fp4out_gate_disabled': "
                      << attentionTailFp4OutGateStatDelta(ResidentAttentionTailFp4OutGateStat::kDisabled)
                      << ", 'attention_tail_fp4out_gate_invalid_input_tokens': "
                      << attentionTailFp4OutGateStatDelta(ResidentAttentionTailFp4OutGateStat::kInvalidInputTokens)
                      << ", 'attention_tail_fp4out_gate_missing_gate_weight_scale': "
                      << attentionTailFp4OutGateStatDelta(
                             ResidentAttentionTailFp4OutGateStat::kMissingGateWeightScale)
                      << ", 'attention_tail_fp4out_gate_missing_gate_input_scale': "
                      << attentionTailFp4OutGateStatDelta(
                             ResidentAttentionTailFp4OutGateStat::kMissingGateInputScale)
                      << ", 'attention_tail_fp4out_gate_missing_gate_alpha': "
                      << attentionTailFp4OutGateStatDelta(ResidentAttentionTailFp4OutGateStat::kMissingGateAlpha)
                      << ", 'attention_tail_fp4out_gate_missing_o_proj_weight_scale': "
                      << attentionTailFp4OutGateStatDelta(
                             ResidentAttentionTailFp4OutGateStat::kMissingOProjWeightScale)
                      << ", 'attention_tail_fp4out_gate_missing_o_proj_input_scale': "
                      << attentionTailFp4OutGateStatDelta(
                             ResidentAttentionTailFp4OutGateStat::kMissingOProjInputScale)
                      << ", 'attention_tail_fp4out_gate_missing_o_proj_alpha': "
                      << attentionTailFp4OutGateStatDelta(ResidentAttentionTailFp4OutGateStat::kMissingOProjAlpha)
                      << ", 'attention_tail_fp4out_gate_shape_rejected': "
                      << attentionTailFp4OutGateStatDelta(ResidentAttentionTailFp4OutGateStat::kShapeRejected)
                      << ", 'attention_tail_fp4out_gate_contiguous_dtype_rejected': "
                      << attentionTailFp4OutGateStatDelta(
                             ResidentAttentionTailFp4OutGateStat::kContiguousDtypeRejected)
                      << ", 'attention_tail_fp4out_gate_gate_proj_rejected': "
                      << attentionTailFp4OutGateStatDelta(ResidentAttentionTailFp4OutGateStat::kGateProjRejected)
                      << ", 'attention_tail_fp4out_gate_fused_quant_rejected': "
                      << attentionTailFp4OutGateStatDelta(ResidentAttentionTailFp4OutGateStat::kFusedQuantRejected)
                      << ", 'dense_layer_visits': " << timing.denseLayerVisits
                      << ", 'moe_layer_visits': " << timing.moeLayerVisits << ", 'indexer_scorer': '"
                      << (useDeepGemmIndexer ? "deepgemm" : "cute_dsl") << "'"
                      << ", 'indexer_topk_compute_layer_visits': " << timing.indexerTopkComputeLayerVisits
                      << ", 'indexer_topk_reuse_layer_visits': " << timing.indexerTopkReuseLayerVisits
                      << ", 'indexer_topk_xstep_reuse_layer_visits': " << timing.indexerTopkXstepReuseLayerVisits
                      << ", 'indexer_topk_dense_layer_visits': " << timing.indexerTopkDenseLayerVisits
                      << ", 'indexer_topk_hisa_layer_visits': " << timing.indexerTopkHisaLayerVisits
                      << ", 'indexer_logits_width': " << timing.indexerLogitsWidth
                      << ", 'indexer_max_seq_len': " << timing.indexerMaxSeqLen << ", 'total_ms': " << totalUs / 1000.0
                      << ", 'contract_ms': " << timing.contractUs / 1000.0
                      << ", 'plan_validate_ms': " << timing.planValidateUs / 1000.0
                      << ", 'dimension_scan_ms': " << timing.dimensionScanUs / 1000.0
                      << ", 'scratch_alloc_ms': " << timing.scratchAllocUs / 1000.0
                      << ", 'prepare_ms': " << timing.prepareUs / 1000.0
                      << ", 'input_norm_ms': " << timing.inputNormUs / 1000.0
                      << ", 'attention_projection_ms': " << timing.attentionProjectionUs / 1000.0
                      << ", 'metadata_refresh_ms': " << timing.metadataRefreshUs / 1000.0
                      << ", 'indexer_projection_ms': " << timing.indexerProjectionUs / 1000.0
                      << ", 'indexer_scatter_ms': " << timing.indexerScatterUs / 1000.0
                      << ", 'indexer_pack_ms': " << timing.indexerPackUs / 1000.0
                      << ", 'indexer_logits_topk_ms': " << timing.indexerLogitsTopkUs / 1000.0
                      << ", 'metadata_list_ms': " << timing.metadataListUs / 1000.0
                      << ", 'attention_dispatch_ms': " << timing.attentionDispatchUs / 1000.0
                      << ", 'attention_tail_ms': " << timing.attentionTailUs / 1000.0
                      << ", 'post_attention_norm_ms': " << timing.postAttentionNormUs / 1000.0
                      << ", 'dense_mlp_ms': " << timing.denseMlpUs / 1000.0
                      << ", 'moe_router_ms': " << timing.moeRouterUs / 1000.0
                      << ", 'moe_experts_ms': " << timing.moeExpertsUs / 1000.0
                      << ", 'moe_shared_expert_ms': " << timing.moeSharedExpertUs / 1000.0
                      << ", 'moe_routed_expert_ms': " << timing.moeRoutedExpertUs / 1000.0
                      << ", 'moe_combine_ms': " << timing.moeCombineUs / 1000.0
                      << ", 'post_ffn_norm_ms': " << timing.postFfnNormUs / 1000.0
                      << ", 'lm_head_ms': " << timing.lmHeadUs / 1000.0 << ", 'sample_ms': " << timing.sampleUs / 1000.0
                      << ", 'cuda_event_timing': " << (cudaEventTimingEnabled ? 1 : 0)
                      << ", 'contract_device_ms': " << deviceTiming.contractUs / 1000.0
                      << ", 'plan_validate_device_ms': " << deviceTiming.planValidateUs / 1000.0
                      << ", 'dimension_scan_device_ms': " << deviceTiming.dimensionScanUs / 1000.0
                      << ", 'scratch_alloc_device_ms': " << deviceTiming.scratchAllocUs / 1000.0
                      << ", 'prepare_device_ms': " << deviceTiming.prepareUs / 1000.0
                      << ", 'input_norm_device_ms': " << deviceTiming.inputNormUs / 1000.0
                      << ", 'attention_projection_device_ms': " << deviceTiming.attentionProjectionUs / 1000.0
                      << ", 'metadata_refresh_device_ms': " << deviceTiming.metadataRefreshUs / 1000.0
                      << ", 'indexer_projection_device_ms': " << deviceTiming.indexerProjectionUs / 1000.0
                      << ", 'indexer_scatter_device_ms': " << deviceTiming.indexerScatterUs / 1000.0
                      << ", 'indexer_pack_device_ms': " << deviceTiming.indexerPackUs / 1000.0
                      << ", 'indexer_logits_topk_device_ms': " << deviceTiming.indexerLogitsTopkUs / 1000.0
                      << ", 'attention_dispatch_device_ms': " << deviceTiming.attentionDispatchUs / 1000.0
                      << ", 'attention_tail_device_ms': " << deviceTiming.attentionTailUs / 1000.0
                      << ", 'post_attention_norm_device_ms': " << deviceTiming.postAttentionNormUs / 1000.0
                      << ", 'dense_mlp_device_ms': " << deviceTiming.denseMlpUs / 1000.0
                      << ", 'moe_router_device_ms': " << deviceTiming.moeRouterUs / 1000.0
                      << ", 'moe_experts_device_ms': " << deviceTiming.moeExpertsUs / 1000.0
                      << ", 'post_ffn_norm_device_ms': " << deviceTiming.postFfnNormUs / 1000.0
                      << ", 'lm_head_device_ms': " << deviceTiming.lmHeadUs / 1000.0
                      << ", 'sample_device_ms': " << deviceTiming.sampleUs / 1000.0 << "}" << std::endl;
        }
        return windowTokensScratch;
    }

    at::Tensor runDecodeWindow(at::Tensor const& initialTokens, at::Tensor const& windowTokensScratch,
        at::Tensor const& inputIdsScratch, at::Tensor const& hiddenStatesScratch, at::Tensor const& logitsScratch,
        at::Tensor const& positionIds, at::Tensor const& seqLensCuda, at::Tensor const& kvLensCuda, int64_t ownedSteps,
        int64_t inputTokens, th::List<int64_t> requestIds, th::List<int64_t> seqLens,
        th::List<int64_t> cachedTokens) const
    {
        validateDecodeWindowTokenContract(initialTokens, windowTokensScratch, inputIdsScratch, hiddenStatesScratch,
            logitsScratch, positionIds, seqLensCuda, kvLensCuda, ownedSteps, inputTokens, requestIds, seqLens,
            cachedTokens);
        static_cast<void>(runDecodeWindowPrepareStep(initialTokens, windowTokensScratch, inputIdsScratch,
            hiddenStatesScratch, positionIds, kvLensCuda, 1, inputTokens));
        TORCH_CHECK(false,
            "DeepseekResidentDecodeHandle::runDecodeWindow model body is not implemented after initial prepare step");
        return windowTokensScratch;
    }

    at::Tensor decode(at::Tensor const& inputIds, at::Tensor const& hiddenStatesScratch,
        at::Tensor const& logitsScratch, int64_t realBatchSize, int64_t paddedBatchSize, int64_t inputTokens,
        th::List<int64_t> requestIds, th::List<int64_t> seqLens, th::List<int64_t> cachedTokens) const
    {
        static_cast<void>(validateDeepseekResidentRequest(inputIds, hiddenStatesScratch, logitsScratch, realBatchSize,
            paddedBatchSize, inputTokens, requestIds, seqLens, cachedTokens));
        static_cast<void>(runInputEmbedding(inputIds, hiddenStatesScratch, inputTokens));
        TORCH_CHECK(false, "DeepseekResidentDecodeHandle::decode CUDA body is not implemented");
        return logitsScratch;
    }

private:
    DeepseekResidentWindowScratch& getResidentWindowScratch(at::Tensor const& hiddenStatesScratch, int64_t maxBatchSize,
        int64_t maxQWidth, int64_t maxQLoraRank, int64_t maxKvLoraRank, int64_t maxRopeDim, int64_t maxKvAWidth,
        int64_t maxLatentWidth, int64_t maxAttentionCoreWidth, int64_t maxDenseIntermediateWidth,
        int64_t maxMoeSharedIntermediateWidth, int64_t maxMoeExperts, int64_t maxMoeTopK) const
    {
        TORCH_CHECK(hiddenStatesScratch.is_cuda(), "hidden_states_scratch must be a CUDA tensor");
        TORCH_CHECK(hiddenStatesScratch.dim() == 2, "hidden_states_scratch must be 2D");
        TORCH_CHECK(maxBatchSize > 0, "resident window scratch max_batch_size must be positive");
        TORCH_CHECK(hiddenStatesScratch.size(1) > 0, "resident window scratch hidden size must be positive");

        DeepseekResidentWindowScratchKey const key{
            hiddenStatesScratch.get_device(),
            static_cast<int64_t>(hiddenStatesScratch.scalar_type()),
            maxBatchSize,
            hiddenStatesScratch.size(1),
            maxQWidth,
            maxQLoraRank,
            maxKvLoraRank,
            maxRopeDim,
            maxKvAWidth,
            maxLatentWidth,
            maxAttentionCoreWidth,
            maxDenseIntermediateWidth,
            maxMoeSharedIntermediateWidth,
            maxMoeExperts,
            maxMoeTopK,
        };
        if (mWindowScratch.valid && mWindowScratch.key == key)
        {
            return mWindowScratch;
        }

        auto const baseOptions = hiddenStatesScratch.options();
        auto const floatOptions = baseOptions.dtype(at::ScalarType::Float);
        auto const intOptions = baseOptions.dtype(at::ScalarType::Int);
        int64_t const hiddenSize = hiddenStatesScratch.size(1);

        mWindowScratch.key = key;
        mWindowScratch.valid = true;
        mWindowScratch.normHiddenScratch = at::empty({maxBatchSize, hiddenSize}, baseOptions);
        mWindowScratch.gatedHiddenScratch = at::empty({maxBatchSize, hiddenSize}, baseOptions);
        mWindowScratch.attentionHiddenScratch = at::empty({maxBatchSize, hiddenSize}, baseOptions);
        mWindowScratch.postAttentionNormScratch = at::empty({maxBatchSize, hiddenSize}, baseOptions);
        mWindowScratch.postAttentionGatedScratch = at::empty({maxBatchSize, hiddenSize}, baseOptions);
        mWindowScratch.postAttentionResidualScratch = at::empty({maxBatchSize, hiddenSize}, baseOptions);
        mWindowScratch.denseMlpOutputScratch = at::empty({maxBatchSize, hiddenSize}, baseOptions);
        mWindowScratch.nextLayerHiddenScratch = at::empty({maxBatchSize, hiddenSize}, baseOptions);
        mWindowScratch.nextLayerResidualScratch = at::empty({maxBatchSize, hiddenSize}, baseOptions);
        mWindowScratch.qScratchStorage = at::empty({maxBatchSize, maxQWidth}, baseOptions);
        mWindowScratch.kvAScratchStorage = at::empty({maxBatchSize, maxKvAWidth}, baseOptions);
        mWindowScratch.qLoraScratchStorage = at::empty({maxBatchSize, maxQLoraRank}, baseOptions);
        mWindowScratch.compressedKvScratchStorage = at::empty({maxBatchSize, maxKvLoraRank}, baseOptions);
        mWindowScratch.kPeScratchStorage = at::empty({maxBatchSize, maxRopeDim}, baseOptions);
        mWindowScratch.latentCacheScratchStorage = at::empty({maxBatchSize, maxLatentWidth}, baseOptions);
        mWindowScratch.attentionCoreOutputScratchStorage
            = at::empty({maxBatchSize, maxAttentionCoreWidth}, baseOptions);
        mWindowScratch.attentionGateScratchStorage = at::empty({maxBatchSize, maxAttentionCoreWidth}, baseOptions);
        mWindowScratch.attentionGateLogitsScratchStorage
            = at::empty({maxBatchSize, maxAttentionCoreWidth}, floatOptions);
        if (maxDenseIntermediateWidth > 0)
        {
            mWindowScratch.denseMlpIntermediateScratchStorage
                = at::empty({maxBatchSize, maxDenseIntermediateWidth}, baseOptions);
            mWindowScratch.denseMlpGateUpScratchStorage
                = at::empty({maxBatchSize, maxDenseIntermediateWidth * 2}, floatOptions);
        }
        else
        {
            mWindowScratch.denseMlpIntermediateScratchStorage = at::Tensor();
            mWindowScratch.denseMlpGateUpScratchStorage = at::Tensor();
        }
        if (maxMoeSharedIntermediateWidth > 0)
        {
            mWindowScratch.moeSharedIntermediateScratchStorage
                = at::empty({maxBatchSize, maxMoeSharedIntermediateWidth}, baseOptions);
            mWindowScratch.moeSharedGateUpScratchStorage
                = at::empty({maxBatchSize, maxMoeSharedIntermediateWidth * 2}, floatOptions);
            mWindowScratch.moeSharedOutputScratchStorage = at::empty({maxBatchSize, hiddenSize}, floatOptions);
        }
        else
        {
            mWindowScratch.moeSharedIntermediateScratchStorage = at::Tensor();
            mWindowScratch.moeSharedGateUpScratchStorage = at::Tensor();
            mWindowScratch.moeSharedOutputScratchStorage = at::Tensor();
        }
        if (maxMoeExperts > 0)
        {
            mWindowScratch.routerLogitsScratchStorage = at::empty({maxBatchSize, maxMoeExperts}, floatOptions);
            mWindowScratch.routerScoresScratchStorage = at::empty({maxBatchSize, maxMoeExperts}, floatOptions);
        }
        else
        {
            mWindowScratch.routerLogitsScratchStorage = at::Tensor();
            mWindowScratch.routerScoresScratchStorage = at::Tensor();
        }
        if (maxMoeTopK > 0)
        {
            mWindowScratch.routerTopkIndicesScratchStorage = at::empty({maxBatchSize, maxMoeTopK}, intOptions);
            mWindowScratch.routerTopkWeightsScratchStorage = at::empty({maxBatchSize, maxMoeTopK}, floatOptions);
        }
        else
        {
            mWindowScratch.routerTopkIndicesScratchStorage = at::Tensor();
            mWindowScratch.routerTopkWeightsScratchStorage = at::Tensor();
        }
        return mWindowScratch;
    }

    at::Tensor makeDenseDecodeTopkIndices(at::Tensor const& kvLensCuda, int64_t inputTokens, int64_t indexTopk) const
    {
        TORCH_CHECK(kvLensCuda.is_cuda(), "kv_lens_cuda must be a CUDA tensor");
        TORCH_CHECK(kvLensCuda.dim() == 1, "kv_lens_cuda must be 1D");
        TORCH_CHECK(kvLensCuda.numel() >= inputTokens, "kv_lens_cuda is shorter than input_tokens");
        TORCH_CHECK(indexTopk > 0, "index_topk must be positive");

        at::Tensor positions = kvLensCuda.narrow(0, 0, inputTokens).to(at::ScalarType::Long) - 1;
        at::Tensor range = at::arange(indexTopk, positions.options()).reshape({1, indexTopk});
        at::Tensor expandedRange = range.expand({inputTokens, indexTopk});
        at::Tensor mask = range <= positions.reshape({inputTokens, 1});
        at::Tensor padding = at::full({inputTokens, indexTopk}, -1, positions.options());
        return at::where(mask, expandedRange, padding).to(at::ScalarType::Int);
    }

    int64_t denseMlpIntermediateSize(int64_t layerIdx) const
    {
        at::Tensor const& gateUpWeight = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kDenseMlpGateUpWeight, "dense MLP gate_up_proj weight");
        at::Tensor const& downWeight = getLayerTensor(
            layerIdx, DeepseekResidentLayerTensorSite::kDenseMlpDownWeight, "dense MLP down_proj weight");
        TORCH_CHECK(gateUpWeight.dim() == 2, "dense MLP gate_up_proj weight must be 2D");
        TORCH_CHECK(downWeight.dim() == 2, "dense MLP down_proj weight must be 2D");
        if (gateUpWeight.size(0) % 2 == 0)
        {
            return gateUpWeight.size(0) / 2;
        }
        TORCH_CHECK(downWeight.size(1) > 0, "dense MLP down_proj weight input dim must be positive");
        return downWeight.size(1);
    }

    int64_t moeNumExperts(int64_t layerIdx) const
    {
        at::Tensor const& gateWeight
            = getLayerTensor(layerIdx, DeepseekResidentLayerTensorSite::kMoeGateWeight, "MoE gate weight");
        TORCH_CHECK(gateWeight.dim() == 2, "MoE gate weight must be 2D");
        TORCH_CHECK(gateWeight.size(0) > 0, "MoE gate weight must have experts");
        return gateWeight.size(0);
    }

    at::Tensor validTokenLogitsPrefix(at::Tensor const& logitsScratch, int64_t inputTokens) const
    {
        TORCH_CHECK(
            mNbTensors > static_cast<int64_t>(DeepseekResidentTensorSlot::kEmbedding), "missing embedding tensor");
        at::Tensor const& embedding = mResidentTensors.at(static_cast<size_t>(DeepseekResidentTensorSlot::kEmbedding));
        TORCH_CHECK(embedding.dim() == 2, "embedding tensor must be 2D");
        int64_t const validVocabSize = std::min<int64_t>(logitsScratch.size(1), embedding.size(0));
        TORCH_CHECK(validVocabSize > 0, "valid sampling vocabulary must be non-empty");
        return logitsScratch.narrow(0, 0, inputTokens).narrow(1, 0, validVocabSize);
    }

    static void validateDecodeWindowTokenContract(at::Tensor const& initialTokens,
        at::Tensor const& windowTokensScratch, at::Tensor const& inputIdsScratch, at::Tensor const& hiddenStatesScratch,
        at::Tensor const& logitsScratch, at::Tensor const& positionIds, at::Tensor const& seqLensCuda,
        at::Tensor const& kvLensCuda, int64_t ownedSteps, int64_t inputTokens, th::List<int64_t> requestIds,
        th::List<int64_t> seqLens, th::List<int64_t> cachedTokens)
    {
        TORCH_CHECK(initialTokens.is_cuda(), "initial_tokens must be a CUDA tensor");
        TORCH_CHECK(windowTokensScratch.is_cuda(), "window_tokens_scratch must be a CUDA tensor");
        TORCH_CHECK(inputIdsScratch.is_cuda(), "input_ids_scratch must be a CUDA tensor");
        TORCH_CHECK(hiddenStatesScratch.is_cuda(), "hidden_states_scratch must be a CUDA tensor");
        TORCH_CHECK(logitsScratch.is_cuda(), "logits_scratch must be a CUDA tensor");
        TORCH_CHECK(positionIds.is_cuda(), "position_ids must be a CUDA tensor");
        TORCH_CHECK(seqLensCuda.is_cuda(), "seq_lens_cuda must be a CUDA tensor");
        TORCH_CHECK(kvLensCuda.is_cuda(), "kv_lens_cuda must be a CUDA tensor");
        TORCH_CHECK(ownedSteps > 1, "owned_steps must be greater than one");
        TORCH_CHECK(inputTokens > 0, "input_tokens must be positive");
        TORCH_CHECK(initialTokens.dim() == 3, "initial_tokens must be 3D");
        TORCH_CHECK(windowTokensScratch.dim() == 3, "window_tokens_scratch must be 3D");
        TORCH_CHECK(inputIdsScratch.dim() >= 1, "input_ids_scratch must have at least one dimension");
        TORCH_CHECK(hiddenStatesScratch.dim() == 2, "hidden_states_scratch must be 2D");
        TORCH_CHECK(logitsScratch.dim() == 2, "logits_scratch must be 2D");
        TORCH_CHECK(positionIds.dim() >= 1 && positionIds.dim() <= 3, "position_ids must be 1D, 2D, or 3D");
        TORCH_CHECK(seqLensCuda.dim() == 1, "seq_lens_cuda must be 1D");
        TORCH_CHECK(kvLensCuda.dim() == 1, "kv_lens_cuda must be 1D");
        TORCH_CHECK(initialTokens.size(0) >= 1, "initial_tokens must contain at least one sampled step");
        TORCH_CHECK(initialTokens.size(1) >= inputTokens, "initial_tokens batch is smaller than input");
        TORCH_CHECK(initialTokens.size(2) >= 1, "initial_tokens must have at least one beam");
        TORCH_CHECK(windowTokensScratch.size(0) >= ownedSteps, "window_tokens_scratch step dim is smaller than window");
        TORCH_CHECK(windowTokensScratch.size(1) >= inputTokens, "window_tokens_scratch batch is smaller than input");
        TORCH_CHECK(windowTokensScratch.size(2) >= 1, "window_tokens_scratch must have at least one beam");
        TORCH_CHECK(inputIdsScratch.numel() >= inputTokens, "input_ids_scratch is shorter than input_tokens");
        TORCH_CHECK(hiddenStatesScratch.size(0) >= inputTokens, "hidden_states_scratch batch is smaller than input");
        TORCH_CHECK(logitsScratch.size(0) >= inputTokens, "logits_scratch batch is smaller than input");
        TORCH_CHECK(positionIds.numel() >= inputTokens, "position_ids is shorter than input_tokens");
        TORCH_CHECK(seqLensCuda.numel() >= inputTokens, "seq_lens_cuda is shorter than input_tokens");
        TORCH_CHECK(kvLensCuda.numel() >= inputTokens, "kv_lens_cuda is shorter than input_tokens");
        TORCH_CHECK(
            initialTokens.scalar_type() == at::ScalarType::Int || initialTokens.scalar_type() == at::ScalarType::Long,
            "initial_tokens must be int32 or int64");
        TORCH_CHECK(windowTokensScratch.scalar_type() == at::ScalarType::Int
                || windowTokensScratch.scalar_type() == at::ScalarType::Long,
            "window_tokens_scratch must be int32 or int64");
        TORCH_CHECK(inputIdsScratch.scalar_type() == at::ScalarType::Int
                || inputIdsScratch.scalar_type() == at::ScalarType::Long,
            "input_ids_scratch must be int32 or int64");
        TORCH_CHECK(static_cast<int64_t>(requestIds.size()) >= inputTokens, "request_ids is shorter than input_tokens");
        TORCH_CHECK(static_cast<int64_t>(seqLens.size()) >= inputTokens, "seq_lens is shorter than input_tokens");
        TORCH_CHECK(
            static_cast<int64_t>(cachedTokens.size()) >= inputTokens, "cached_tokens is shorter than input_tokens");
    }

    static void validateDecodeWindowAdvanceContract(at::Tensor const& initialTokens,
        at::Tensor const& windowTokensScratch, at::Tensor const& inputIdsScratch, at::Tensor const& positionIds,
        at::Tensor const& kvLensCuda, int64_t outputStepIdx, int64_t inputTokens)
    {
        TORCH_CHECK(initialTokens.is_cuda(), "initial_tokens must be a CUDA tensor");
        TORCH_CHECK(windowTokensScratch.is_cuda(), "window_tokens_scratch must be a CUDA tensor");
        TORCH_CHECK(inputIdsScratch.is_cuda(), "input_ids_scratch must be a CUDA tensor");
        TORCH_CHECK(positionIds.is_cuda(), "position_ids must be a CUDA tensor");
        TORCH_CHECK(kvLensCuda.is_cuda(), "kv_lens_cuda must be a CUDA tensor");
        TORCH_CHECK(outputStepIdx > 0, "output_step_idx must be greater than zero");
        TORCH_CHECK(inputTokens > 0, "input_tokens must be positive");
        TORCH_CHECK(initialTokens.dim() == 3, "initial_tokens must be 3D");
        TORCH_CHECK(windowTokensScratch.dim() == 3, "window_tokens_scratch must be 3D");
        TORCH_CHECK(inputIdsScratch.numel() >= inputTokens, "input_ids_scratch is shorter than input_tokens");
        TORCH_CHECK(positionIds.numel() >= inputTokens, "position_ids is shorter than input_tokens");
        TORCH_CHECK(kvLensCuda.dim() == 1, "kv_lens_cuda must be 1D");
        TORCH_CHECK(kvLensCuda.numel() >= inputTokens, "kv_lens_cuda is shorter than input_tokens");
        TORCH_CHECK(windowTokensScratch.size(0) > outputStepIdx - 1,
            "window_tokens_scratch does not contain the requested source step");
        TORCH_CHECK(windowTokensScratch.size(1) >= inputTokens, "window_tokens_scratch batch is smaller than input");
        TORCH_CHECK(windowTokensScratch.size(2) >= 1, "window_tokens_scratch must have at least one beam");
        TORCH_CHECK(initialTokens.size(0) >= 1, "initial_tokens must contain at least one sampled step");
        TORCH_CHECK(initialTokens.size(1) >= inputTokens, "initial_tokens batch is smaller than input");
        TORCH_CHECK(initialTokens.size(2) >= 1, "initial_tokens must have at least one beam");
        TORCH_CHECK(
            initialTokens.scalar_type() == at::ScalarType::Int || initialTokens.scalar_type() == at::ScalarType::Long,
            "initial_tokens must be int32 or int64");
        TORCH_CHECK(windowTokensScratch.scalar_type() == at::ScalarType::Int
                || windowTokensScratch.scalar_type() == at::ScalarType::Long,
            "window_tokens_scratch must be int32 or int64");
        TORCH_CHECK(inputIdsScratch.scalar_type() == at::ScalarType::Int
                || inputIdsScratch.scalar_type() == at::ScalarType::Long,
            "input_ids_scratch must be int32 or int64");
    }

    static void validateDecodeWindowSampleStepContract(at::Tensor const& logitsScratch,
        at::Tensor const& windowTokensScratch, int64_t outputStepIdx, int64_t inputTokens)
    {
        TORCH_CHECK(logitsScratch.is_cuda(), "logits_scratch must be a CUDA tensor");
        TORCH_CHECK(windowTokensScratch.is_cuda(), "window_tokens_scratch must be a CUDA tensor");
        TORCH_CHECK(outputStepIdx > 0, "output_step_idx must be greater than zero");
        TORCH_CHECK(inputTokens > 0, "input_tokens must be positive");
        TORCH_CHECK(logitsScratch.dim() == 2, "logits_scratch must be 2D");
        TORCH_CHECK(windowTokensScratch.dim() == 3, "window_tokens_scratch must be 3D");
        TORCH_CHECK(logitsScratch.size(0) >= inputTokens, "logits_scratch batch is smaller than input");
        TORCH_CHECK(windowTokensScratch.size(0) > outputStepIdx,
            "window_tokens_scratch does not contain the requested output step");
        TORCH_CHECK(windowTokensScratch.size(1) >= inputTokens, "window_tokens_scratch batch is smaller than input");
        TORCH_CHECK(windowTokensScratch.size(2) >= 1, "window_tokens_scratch must have at least one beam");
        TORCH_CHECK(windowTokensScratch.scalar_type() == at::ScalarType::Int
                || windowTokensScratch.scalar_type() == at::ScalarType::Long,
            "window_tokens_scratch must be int32 or int64");
    }

    static void validateDecodeWindowPrepareStepContract(at::Tensor const& initialTokens,
        at::Tensor const& windowTokensScratch, at::Tensor const& inputIdsScratch, at::Tensor const& hiddenStatesScratch,
        at::Tensor const& positionIds, at::Tensor const& kvLensCuda, int64_t outputStepIdx, int64_t inputTokens)
    {
        validateDecodeWindowAdvanceContract(
            initialTokens, windowTokensScratch, inputIdsScratch, positionIds, kvLensCuda, outputStepIdx, inputTokens);
        TORCH_CHECK(hiddenStatesScratch.is_cuda(), "hidden_states_scratch must be a CUDA tensor");
        TORCH_CHECK(hiddenStatesScratch.dim() == 2, "hidden_states_scratch must be 2D");
        TORCH_CHECK(hiddenStatesScratch.size(0) >= inputTokens, "hidden_states_scratch batch is smaller than input");
    }

    at::Tensor const& getLayerTensor(
        int64_t layerIdx, DeepseekResidentLayerTensorSite site, char const* missingName) const
    {
        std::optional<size_t> const tensorIdx = findLayerTensorIndex(layerIdx, static_cast<int64_t>(site));
        TORCH_CHECK(tensorIdx.has_value(), "missing ", missingName);
        return mResidentTensors.at(tensorIdx.value());
    }

    std::optional<at::Tensor> getOptionalLayerTensor(int64_t layerIdx, DeepseekResidentLayerTensorSite site) const
    {
        std::optional<size_t> const tensorIdx = findLayerTensorIndex(layerIdx, static_cast<int64_t>(site));
        if (!tensorIdx.has_value())
        {
            return std::nullopt;
        }
        return mResidentTensors.at(tensorIdx.value());
    }

    std::optional<size_t> findLayerTensorIndex(int64_t layerIdx, int64_t siteId) const
    {
        if (layerIdx < 0 || layerIdx >= mNbLayers)
        {
            return std::nullopt;
        }
        int64_t const siteStart = mLayerSiteOffsets.at(static_cast<size_t>(layerIdx));
        int64_t const siteStop = mLayerSiteOffsets.at(static_cast<size_t>(layerIdx + 1));
        for (int64_t siteIdx = siteStart; siteIdx < siteStop; ++siteIdx)
        {
            if (mLayerSiteIds.at(static_cast<size_t>(siteIdx)) != siteId)
            {
                continue;
            }
            int64_t const tensorIdx = mLayerSiteTensorIndices.at(static_cast<size_t>(siteIdx));
            TORCH_CHECK(tensorIdx >= 0 && tensorIdx < mNbTensors, "resident layer site tensor index is out of range");
            return static_cast<size_t>(tensorIdx);
        }
        return std::nullopt;
    }

    int64_t mNbLayers{0};
    int64_t mNbTensors{0};
    std::vector<int64_t> mLayerOffsets;
    std::vector<int64_t> mLayerKinds;
    std::vector<int64_t> mLayerSiteOffsets;
    std::vector<int64_t> mLayerSiteIds;
    std::vector<int64_t> mLayerSiteTensorIndices;
    std::vector<at::Tensor> mResidentTensors;
    mutable std::unordered_map<int64_t, DeepseekResidentFusedQbWqBCache> mFusedQbWqBCache;
    mutable std::unordered_map<int64_t, DeepseekResidentFusedKvAWkWpCache> mFusedKvAWkWpCache;
    mutable std::unordered_map<int64_t, std::unique_ptr<DeepseekResidentMoeRunnerType>> mFp4MoeRunners;
    mutable std::mutex mWindowScratchMutex;
    mutable DeepseekResidentWindowScratch mWindowScratch;
};

bool deepseekResidentDecodePrepare(
    th::List<int64_t> residentLayerOffsets, th::List<int64_t> residentLayerKinds, c10::List<at::Tensor> residentTensors)
{
    static_cast<void>(validateDeepseekResidentManifest(residentLayerOffsets, residentLayerKinds, residentTensors));
    return true;
}

at::Tensor deepseekResidentDecode(at::Tensor const& inputIds, at::Tensor const& hiddenStatesScratch,
    at::Tensor const& logitsScratch, int64_t realBatchSize, int64_t paddedBatchSize, int64_t inputTokens,
    th::List<int64_t> requestIds, th::List<int64_t> seqLens, th::List<int64_t> cachedTokens,
    th::List<int64_t> residentLayerOffsets, th::List<int64_t> residentLayerKinds, c10::List<at::Tensor> residentTensors)
{
    static_cast<void>(validateDeepseekResidentRequest(inputIds, hiddenStatesScratch, logitsScratch, realBatchSize,
        paddedBatchSize, inputTokens, requestIds, seqLens, cachedTokens));
    static_cast<void>(validateDeepseekResidentManifest(residentLayerOffsets, residentLayerKinds, residentTensors));
    TORCH_CHECK(false, "deepseek_resident_decode CUDA body is not implemented");
    return logitsScratch;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.class_<tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle>("DeepseekResidentDecodeHandle")
        .def(torch::init<th::List<int64_t>, th::List<int64_t>, th::List<int64_t>, th::List<int64_t>, th::List<int64_t>,
            c10::List<at::Tensor>>())
        .def("get_nb_layers", &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::getNbLayers)
        .def("get_nb_tensors", &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::getNbTensors)
        .def("has_layer_tensor_site", &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::hasLayerTensorSite)
        .def("run_layer_dsa_indexer_assets_ready",
            &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runLayerDsaIndexerAssetsReady)
        .def("run_layer_dsa_indexer_assets_not_ready_reason",
            &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runLayerDsaIndexerAssetsNotReadyReason)
        .def("run_layer_moe_expert_assets_ready",
            &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runLayerMoeExpertAssetsReady)
        .def("run_layer_moe_expert_assets_not_ready_reason",
            &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runLayerMoeExpertAssetsNotReadyReason)
        .def("run_nvfp4_linear", &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runNvfp4Linear)
        .def("run_layer_dsa_indexer_wk_weights_projection",
            &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runLayerDsaIndexerWkWeightsProjection)
        .def("run_layer_dsa_indexer_fp4_projection",
            &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runLayerDsaIndexerFp4Projection)
        .def("run_input_embedding", &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runInputEmbedding)
        .def("run_layer_input_rmsnorm", &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runLayerInputRmsNorm)
        .def("run_layer_input_gated_norm",
            &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runLayerInputGatedNorm)
        .def("run_layer_dsa_attention_projection",
            &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runLayerDsaAttentionProjection)
        .def("run_layer_attention_output_tail",
            &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runLayerAttentionOutputTail)
        .def("run_layer_post_attention_rmsnorm",
            &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runLayerPostAttentionRmsNorm)
        .def("run_layer_post_attention_gated_norm",
            &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runLayerPostAttentionGatedNorm)
        .def("run_layer_moe_router", &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runLayerMoeRouter)
        .def("run_layer_moe_experts", &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runLayerMoeExperts)
        .def("run_layer_dense_mlp", &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runLayerDenseMlp)
        .def("run_layer_post_ffn_rmsnorm",
            &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runLayerPostFfnRmsNorm)
        .def("run_lm_head_logits", &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runLmHeadLogits)
        .def("run_greedy_sample", &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runGreedySample)
        .def("run_decode_window_attention_metadata_device_refresh",
            &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runDecodeWindowAttentionMetadataDeviceRefresh)
        .def("run_indexer_k_cache_scatter",
            &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runIndexerKCacheScatter)
        .def("run_indexer_topk_decode", &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runIndexerTopkDecode)
        .def("run_indexer_xstep_recency_patch",
            &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runIndexerXstepRecencyPatch)
        .def("run_decode_window_ready", &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runDecodeWindowReady)
        .def("run_decode_window_not_ready_reason",
            &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runDecodeWindowNotReadyReason)
        .def("run_layer_dsa_attention_dispatch_ready",
            &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runLayerDsaAttentionDispatchReady)
        .def("run_layer_dsa_attention_dispatch_not_ready_reason",
            &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runLayerDsaAttentionDispatchNotReadyReason)
        .def("run_layer_dsa_attention_dispatch",
            &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runLayerDsaAttentionDispatch)
        .def("run_decode_window_advance_state",
            &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runDecodeWindowAdvanceState)
        .def("run_decode_window_sample_step",
            &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runDecodeWindowSampleStep)
        .def("run_decode_window_prepare_step",
            &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runDecodeWindowPrepareStep)
        .def("run_decode_window_with_dsa_plan_ready",
            &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runDecodeWindowWithDsaPlanReady)
        .def("run_decode_window_with_dsa_plan_not_ready_reason",
            &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runDecodeWindowWithDsaPlanNotReadyReason)
        .def("run_decode_window_with_dsa_plan",
            &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runDecodeWindowWithDsaPlan)
        .def("run_decode_window", &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::runDecodeWindow)
        .def("decode", &tensorrt_llm::torch_ext::DeepseekResidentDecodeHandle::decode);
    m.def("deepseek_resident_decode_ready() -> bool");
    m.def(
        "deepseek_resident_decode_prepare(int[] resident_layer_offsets, int[] resident_layer_kinds, "
        "Tensor[] resident_tensors) -> bool");
    m.def(
        "deepseek_resident_decode(Tensor input_ids, Tensor(a!) hidden_states_scratch, Tensor(b!) logits_scratch, "
        "int real_batch_size, int padded_batch_size, int input_tokens, int[] request_ids, int[] seq_lens, "
        "int[] cached_tokens, int[] resident_layer_offsets, int[] resident_layer_kinds, "
        "Tensor[] resident_tensors) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
    m.impl("deepseek_resident_decode_ready", &tensorrt_llm::torch_ext::deepseekResidentDecodeReady);
    m.impl("deepseek_resident_decode_prepare", &tensorrt_llm::torch_ext::deepseekResidentDecodePrepare);
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("deepseek_resident_decode", &tensorrt_llm::torch_ext::deepseekResidentDecode);
}

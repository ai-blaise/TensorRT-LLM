/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/mathUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttentionUtils.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/mlaKernels.h"
#include "tensorrt_llm/kernels/quantization.cuh"
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

using namespace tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// A stateful callback functor that maintains the running sum between consecutive scans.
struct BlockPrefixCallbackOp
{
    // Running prefix
    int mRunningTotal;

    // Constructor
    __device__ BlockPrefixCallbackOp(int runningTotal)
        : mRunningTotal(runningTotal)
    {
    }

    // Thread-0 is responsible for returning a value for seeding the block-wide scan.
    __device__ int operator()(int blockAggregate)
    {
        int oldPrefix = mRunningTotal;
        mRunningTotal += blockAggregate;
        return oldPrefix;
    }
};

template <typename T>
struct VecType
{
    using Type = T;
    using GPTJEltType = T;
};

template <>
struct VecType<float>
{
    using Type = float4;
    using GPTJEltType = float2;
};

template <>
struct VecType<half>
{
    using Type = uint4;
    using GPTJEltType = uint32_t;
};

template <>
struct VecType<__nv_bfloat16>
{
    using Type = mmha::bf16_8_t;
    using GPTJEltType = __nv_bfloat162;
};

struct __align__(16) fp8_16_t
{
    __nv_fp8x4_e4m3 x;
    __nv_fp8x4_e4m3 y;
    __nv_fp8x4_e4m3 z;
    __nv_fp8x4_e4m3 w;
};

template <>
struct VecType<__nv_fp8_e4m3>
{
    using Type = fp8_16_t;
    using GPTJEltType = __nv_fp8x2_e4m3;
};

template <typename T>
struct loadPagedKVKernelTraits
{
    static constexpr int kLoraSize = 512;
    static constexpr int kRopeSize = 64;
    static constexpr int kHeadSize = kLoraSize + kRopeSize;
    using VecT = typename VecType<T>::Type;
    static constexpr int kBytesPerElem = sizeof(T);
    static constexpr int kBytesPerLoad = 16;
    static constexpr int kElemPerLoad = kBytesPerLoad / kBytesPerElem;
    static_assert((kHeadSize * kBytesPerElem) % kBytesPerLoad == 0,
        "kHeadSize * kBytesPerElem must be multiple of kBytesPerLoad (16Bytes)");
    static constexpr int kVecPerHead = (kHeadSize * kBytesPerElem) / kBytesPerLoad;
    static constexpr int kThreadPerHead = kVecPerHead; // for each head, we use kThreadPerHead threads to fetch all the
                                                       // kv cache data, each thread read kv cache only once.
    static constexpr int kTokenPerBlock
        = std::is_same_v<T, float> ? 4 : 8; // for each block, we fetch 4 tokens for fp32, 8 tokens for other types.
    static constexpr int kBlockSize = kThreadPerHead * kTokenPerBlock;
    static constexpr int kKVThreadPerHead = (kLoraSize * kBytesPerElem) / kBytesPerLoad;
};

template <typename SrcType, int NUM>
inline __device__ void quantCopy(
    __nv_fp8_e4m3* dst_global_ptr, SrcType const* src_fragment_ptr, float const scale_val = 1.f)
{
    using DstVecType = typename std::conditional<sizeof(SrcType) == 2, float2, float>::type;
    using SrcType2 =
        typename std::conditional<sizeof(SrcType) == 2, typename TypeConverter<SrcType>::Type, float2>::type;
    static constexpr int COPY_SIZE = sizeof(DstVecType);
    static constexpr int TOTAL_COPY_SIZE = NUM * sizeof(__nv_fp8_e4m3);
    static constexpr int LOOP_NUM = TOTAL_COPY_SIZE / COPY_SIZE;
    static_assert(TOTAL_COPY_SIZE % COPY_SIZE == 0);
    static constexpr int CVT_NUM = COPY_SIZE / sizeof(__nv_fp8_e4m3) / 2;
    static_assert(COPY_SIZE % (sizeof(__nv_fp8_e4m3) * 2) == 0);
    DstVecType fragment;
    int offset = 0;
#pragma unroll
    for (int i = 0; i < LOOP_NUM; ++i)
    {
#pragma unroll
        for (int j = 0; j < CVT_NUM; ++j)
        {
            float2 val2 = cuda_cast<float2>(reinterpret_cast<SrcType2 const*>(src_fragment_ptr)[j + offset]);
            val2.x *= scale_val;
            val2.y *= scale_val;
            reinterpret_cast<__nv_fp8x2_e4m3*>(&fragment)[j] = __nv_fp8x2_e4m3(val2);
        }
        reinterpret_cast<DstVecType*>(dst_global_ptr)[i] = fragment;
        offset += CVT_NUM;
    }
}

// NVFP4 KV write for the dense MLA latent. Each thread holds ELTS_PER_VEC
// elements; for bf16 ELTS_PER_VEC == 8, which matches the warp-cooperative
// quantizer cvt_warp_fp16_to_fp4 (8 elements/thread, two adjacent lanes reduce
// the amax over a 16-element scale block). The 576-wide latent is laid out as
// 72 contiguous 8-element vecs (latent vecs 0..63, rope vecs 64..71); vec v of
// token t writes 4 packed E2M1 bytes at byte offset t*kvBytesPerToken + v*4 in
// the data pool and, for even v, one E4M3 scale byte at offset
// t*scaleBytesPerToken + v/2 in the parallel block-scale pool. SFScaleVal == 1
// so the stored E4M3 scale is exactly block_amax/6 and the decode kernel's
// (E2M1 code) * (E4M3 scale) recovers the original value.
//
// kDataBytesPerToken / kScaleBytesPerToken are the per-token byte strides of
// the FP4 data pool (576/2 = 288) and the block-scale pool (576/16 = 36).
template <typename T, int ELTS_PER_VEC, int K_DATA_BYTES_PER_TOKEN, int K_SCALE_BYTES_PER_TOKEN>
inline __device__ void quantCopyNvfp4(
    uint8_t* kDataBlock, uint8_t* kScaleBlock, int localTokenIdx, int vecIdx, T const* srcFragment)
{
    // cvt_warp_fp16_to_fp4 packs CVT_ELTS_PER_THREAD (8) elements per thread and
    // reduces the per-16-element amax across two adjacent lanes. NVFP4 MLA KV is
    // only ever instantiated with a 16-bit T (bf16/fp16 -> ELTS_PER_VEC == 8);
    // guard with `if constexpr` so the float instantiation (ELTS_PER_VEC == 4),
    // which is never reached at runtime, still compiles.
    if constexpr (ELTS_PER_VEC == CVT_ELTS_PER_THREAD)
    {
        constexpr int kSfVecSize = 16;
        // Two adjacent threads (vecIdx 2k, 2k+1) share one E4M3 scale block; only
        // the even thread writes the scale byte (matching cvt_warp_fp16_to_fp4).
        uint8_t* sfOut = (vecIdx % 2 == 0)
            ? kScaleBlock + static_cast<size_t>(localTokenIdx) * K_SCALE_BYTES_PER_TOKEN + vecIdx / 2
            : nullptr;
        PackedVec<T> packed = *reinterpret_cast<PackedVec<T> const*>(srcFragment);
        uint32_t const e2m1
            = cvt_warp_fp16_to_fp4<T, kSfVecSize, /*UE8M0_SF=*/false>(packed, /*SFScaleVal=*/1.0f, sfOut);
        // 8 E2M1 codes = 4 bytes at this vec's slot in the token's data region.
        uint32_t* dataDst = reinterpret_cast<uint32_t*>(
            kDataBlock + static_cast<size_t>(localTokenIdx) * K_DATA_BYTES_PER_TOKEN);
        dataDst[vecIdx] = e2m1;
    }
}

template <typename DstType, int NUM>
inline __device__ void dequantCopy(
    DstType* dst_global_ptr, __nv_fp8_e4m3 const* src_fragment_ptr, float const scale_val = 1.f)
{
    using DstVecType = typename VecType<DstType>::Type;
    using DstType2 =
        typename std::conditional<sizeof(DstType) == 2, typename TypeConverter<DstType>::Type, float2>::type;
    static constexpr int COPY_SIZE = sizeof(DstVecType);
    static constexpr int TOTAL_COPY_SIZE = NUM * sizeof(DstType);
    static constexpr int LOOP_NUM = TOTAL_COPY_SIZE / COPY_SIZE;
    static_assert(TOTAL_COPY_SIZE % COPY_SIZE == 0);
    static constexpr int CVT_NUM = COPY_SIZE / sizeof(DstType) / 2;
    static_assert(COPY_SIZE % (sizeof(DstType) * 2) == 0);
    DstVecType fragment;
    int offset = 0;
#pragma unroll
    for (int i = 0; i < LOOP_NUM; ++i)
    {
#pragma unroll
        for (int j = 0; j < CVT_NUM; ++j)
        {
            float2 val2 = cuda_cast<float2>(reinterpret_cast<__nv_fp8x2_e4m3 const*>(src_fragment_ptr)[j + offset]);
            val2.x *= scale_val;
            val2.y *= scale_val;
            reinterpret_cast<DstType2*>(&fragment)[j] = cuda_cast<DstType2>(val2);
        }
        reinterpret_cast<DstVecType*>(dst_global_ptr)[i] = fragment;
        offset += CVT_NUM;
    }
}

// ============================ KVarN / BDR ============================
// Block-diagonal-Hadamard + per-token low-bit dense MLA latent KV (KvCacheDataType::KVARN).
//
// Design (validated: kvarn_inkernel/bdr_inkernel_bench.cu, bdr_vs_kvarn_verdict.log):
//   * order-128 block-diagonal Hadamard on the 512-d compressed_kv (4 sub-blocks)
//     and a single 64-wide Hadamard on k_pe. The rotation is the only decorrelator
//     (no Sinkhorn s_col): rotation alone recovers >=0.993 cos at INT4, matching the
//     full Sinkhorn variant (SAW-INT4 arXiv:2604.19157 Table 2/3, confirmed here).
//   * per-token asymmetric INT4 RTN (scale, zp); per-head scale array isolates
//     outlier head-groups (longctx verdict: cos 0.92->0.96 @128K).
//   * READ stores ckv in the ROTATED frame: dequant-on-read is unpack+(q*scale+zp),
//     NO inverse Hadamard, NO per-channel gather. The matching H is folded into
//     k_b_proj_trans (Q-correction): (q@(W_UK H)) @ (H k)^T == (q@W_UK)@k^T. This is
//     why the read is ~fp8-cost (16.4us vs 11.3us @ N=32) not the 64-71us a Sinkhorn
//     inverse-rotate read costs.

// In-warp block-diagonal FWHT-128 over one token's data held across consecutive
// lanes (each lane owns ELTS channels). Intra-lane butterfly for the low stages,
// __shfl_xor_sync for the high (cross-lane) stages; 128/ELTS lanes per sub-block.
template <typename T, int ELTS>
inline __device__ void bdRotate128InWarp(float (&reg)[ELTS], int laneInBlock, unsigned mask)
{
#pragma unroll
    for (int len = 1; len < ELTS; len <<= 1)
    {
#pragma unroll
        for (int i = 0; i < ELTS; ++i)
        {
            int partner = i ^ len;
            if (i < partner)
            {
                float u = reg[i], v = reg[partner];
                reg[i] = u + v;
                reg[partner] = u - v;
            }
        }
    }
    int kLanesPerBlock = 128 / ELTS;
#pragma unroll
    for (int span = 1; span < 32; span <<= 1)
    {
        if (span >= kLanesPerBlock)
            break;
        bool low = ((laneInBlock & span) == 0);
#pragma unroll
        for (int i = 0; i < ELTS; ++i)
        {
            float other = __shfl_xor_sync(mask, reg[i], span);
            reg[i] = low ? (reg[i] + other) : (other - reg[i]);
        }
    }
    constexpr float kInvSqrt128 = 0.088388347648318f;
#pragma unroll
    for (int i = 0; i < ELTS; ++i)
        reg[i] *= kInvSqrt128;
}

// KVarN dequant-on-read: unpack ELTS low-bit codes from the vec slot and apply
// the per-token/sub-block (scale, zp). Output is the ROTATED-frame value
// (Q-side fold un-rotates downstream). Drop-in alongside dequantCopy.
template <typename DstType, int ELTS, int BITS>
inline __device__ void dequantCopyKVarN(
    DstType* dst_global_ptr, uint8_t const* packed, float scale, float zp)
{
    static_assert(BITS == 2 || BITS == 4, "KVarN BDR supports 2-bit or 4-bit packing");
    static_assert((ELTS * BITS) % 8 == 0, "ELTS must pack to whole bytes");
    constexpr int kValsPerByte = 8 / BITS;
    constexpr int kMask = (1 << BITS) - 1;
    using DstVecType = typename VecType<DstType>::Type;
    DstVecType frag;
    DstType* fragElts = reinterpret_cast<DstType*>(&frag);
#pragma unroll
    for (int i = 0; i < ELTS; ++i)
    {
        uint8_t const b = packed[i / kValsPerByte];
        int const shift = (i % kValsPerByte) * BITS;
        int const q = (b >> shift) & kMask;
        fragElts[i] = cuda_cast<DstType>(static_cast<float>(q) * scale + zp);
    }
    *reinterpret_cast<DstVecType*>(dst_global_ptr) = frag;
}
// ---------------------------------------------------------------------------
// KVarN/BDR production in-kernel write: warp-cooperative block-diagonal Hadamard
// (order HORDER=128) + per-(token,sub-block) asymmetric INT4 RTN, fused at the
// generation-kernel KV write site. Per the 128K accuracy verdict
// (kvarn_inkernel/results/bdr_longctx_4k_128k.log) the scale is PER SUB-BLOCK
// (4 scales/zps on the 512-d ckv), not per-token: a single per-token scale
// collapses to ~0.866 cos at 128K, per-sub-block holds ~0.992.
//
// Threading match (DSV3 gen kernel, fp16): a token's 512-d latent = K_VECS_PER_HEAD
// = 64 vecs over 64 lanes (2 warps), ELTS_PER_VEC=8 ch/lane. One HORDER=128
// sub-block = 16 lanes * 8 ch -> fits inside ONE warp half. So both the
// sub-block Hadamard and its min/max scale reduction are pure intra-warp
// __shfl_xor over a 16-lane group: no shared memory, no cross-warp sync.
//
// kInvSqrtHadamard = 1/sqrt(128).
static constexpr float kInvSqrtHadamard128 = 0.088388347648318f;

// Warp-cooperative FWHT over a 16-lane group (128 ch) where each lane holds
// ELTS contiguous channels in reg[]. Intra-lane butterfly (stages < ELTS) then
// cross-lane __shfl_xor (stages ELTS..64). laneInBlk in [0,16). Result is the
// rotated sub-block, normalized by 1/sqrt(128).
template <int ELTS>
inline __device__ void bdrFwhtSubblockWarp(float (&reg)[ELTS], int laneInBlk, unsigned mask)
{
    // intra-lane stages.
#pragma unroll
    for (int len = 1; len < ELTS; len <<= 1)
    {
#pragma unroll
        for (int i = 0; i < ELTS; ++i)
        {
            int partner = i ^ len;
            if (i < partner)
            {
                float u = reg[i], v = reg[partner];
                reg[i] = u + v;
                reg[partner] = u - v;
            }
        }
    }
    // cross-lane stages within the 16-lane (128/ELTS) sub-block.
    constexpr int kLanes = 128 / ELTS;
#pragma unroll
    for (int span = 1; span < kLanes; span <<= 1)
    {
        bool low = ((laneInBlk & span) == 0);
#pragma unroll
        for (int i = 0; i < ELTS; ++i)
        {
            float other = __shfl_xor_sync(mask, reg[i], span);
            reg[i] = low ? (reg[i] + other) : (other - reg[i]);
        }
    }
#pragma unroll
    for (int i = 0; i < ELTS; ++i)
        reg[i] *= kInvSqrtHadamard128;
}

// KVarN write of one ELTS-wide vec of a 128-ch sub-block: the caller has the
// post-rotate reg[] (via bdrFwhtSubblockWarp) and the sub-block (scale,zp) from
// a 16-lane min/max reduction. Packs ELTS values low-first into whole bytes.
template <int ELTS, int BITS>
inline __device__ void bdrPackLowBitVec(uint8_t* packed, float const (&reg)[ELTS], float scale, float zp)
{
    static_assert(BITS == 2 || BITS == 4, "KVarN BDR supports 2-bit or 4-bit packing");
    static_assert((ELTS * BITS) % 8 == 0, "ELTS must pack to whole bytes");
    constexpr int kValsPerByte = 8 / BITS;
    constexpr int kQMax = (1 << BITS) - 1;
    float inv = 1.0f / scale;
#pragma unroll
    for (int byteIdx = 0; byteIdx < (ELTS * BITS) / 8; ++byteIdx)
    {
        uint8_t out = 0;
#pragma unroll
        for (int j = 0; j < kValsPerByte; ++j)
        {
            int const elt = byteIdx * kValsPerByte + j;
            int q = __float2int_rn((reg[elt] - zp) * inv);
            q = q < 0 ? 0 : (q > kQMax ? kQMax : q);
            out |= static_cast<uint8_t>(q << (j * BITS));
        }
        packed[byteIdx] = out;
    }
}

// 16-lane (128-ch sub-block) min/max reduction over each lane's ELTS values.
template <int ELTS>
inline __device__ void bdrSubblockMinMax(
    float const (&reg)[ELTS], int /*laneInBlk*/, unsigned mask, float& outLo, float& outHi)
{
    float lo = reg[0], hi = reg[0];
#pragma unroll
    for (int i = 1; i < ELTS; ++i)
    {
        lo = fminf(lo, reg[i]);
        hi = fmaxf(hi, reg[i]);
    }
    constexpr int kLanes = 128 / ELTS;
#pragma unroll
    for (int span = kLanes / 2; span >= 1; span >>= 1)
    {
        lo = fminf(lo, __shfl_xor_sync(mask, lo, span));
        hi = fmaxf(hi, __shfl_xor_sync(mask, hi, span));
    }
    outLo = lo;
    outHi = hi;
}
// =====================================================================

template <typename T, int BLOCK_SIZE, int K_DIM, int ROPE_DIM, typename KVCacheBuffer>
__global__ void applyMLARopeAndAssignQKVKernelOptContext(T* q_ptr, T* q_pe, T* k_ptr, T const* fuse_buf,
    KVCacheBuffer kv_cache, int q_pe_ld, int q_pe_stride, float2 const* cos_sin_cache, size_t head_num, int head_size,
    int c_k, int* cu_q_seqlens, int32_t const* kv_cache_lengths, uint32_t max_input_seq_len, KvCacheDataType cache_type,
    float const* quant_scale_kv, int32_t const* helix_position_offsets, bool absorption_mode)
{

    // Constants.
    using VecT = typename VecType<T>::Type;
    using GPTJEltT = typename VecType<T>::GPTJEltType;
    constexpr auto HEAD_SIZE = ROPE_DIM;
    constexpr auto K_HEAD_SIZE = K_DIM;
    constexpr auto BYTES_PER_ELT = sizeof(T);
    constexpr auto BYTES_PER_LOAD = 16;
    constexpr auto ELTS_PER_VEC = BYTES_PER_LOAD / BYTES_PER_ELT;
    static_assert((HEAD_SIZE * BYTES_PER_ELT) % BYTES_PER_LOAD == 0, "Head size needs to be multiple of 16 bytes.");
    constexpr auto VECS_PER_HEAD = HEAD_SIZE * BYTES_PER_ELT / BYTES_PER_LOAD;
    constexpr auto K_VECS_PER_HEAD = K_HEAD_SIZE * BYTES_PER_ELT / BYTES_PER_LOAD;
    static_assert(BLOCK_SIZE % VECS_PER_HEAD == 0, "Kernel block should be able to handle entire heads.");
    constexpr auto TOKENS_PER_BLOCK = BLOCK_SIZE / VECS_PER_HEAD;
    constexpr auto K_TOKENS_PER_BLOCK = BLOCK_SIZE / K_VECS_PER_HEAD;
    constexpr auto TOTAL_VECS_PER_HEAD = VECS_PER_HEAD + K_VECS_PER_HEAD;

    // Block/Head idx.
    size_t const batch_idx = blockIdx.y;
    size_t const head_idx = blockIdx.z;

    // The nope head_size for q.
    // Use the latent_space head size in the absorption mode.
    int nope_head_size_q = absorption_mode ? c_k : head_size;

    if (head_idx < head_num)
    {
        size_t const head_dim_vec_idx = (threadIdx.x % VECS_PER_HEAD);
        size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;

        size_t const seq_len_loop_end
            = size_t((max_input_seq_len + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK;
        float quant_scale_kv_val = quant_scale_kv ? quant_scale_kv[0] : 1.f;

        // Mainloop.
        for (int local_token_idx = (threadIdx.x / VECS_PER_HEAD) + blockIdx.x * TOKENS_PER_BLOCK;
             local_token_idx < seq_len_loop_end; local_token_idx += TOKENS_PER_BLOCK * gridDim.x)
        {

            int const global_token_offset = cu_q_seqlens[batch_idx];
            int const cache_seq_len = kv_cache_lengths[batch_idx];

            // Derive cached offset and current input length
            int const current_seq_len = cu_q_seqlens[batch_idx + 1] - global_token_offset;
            int const cached_offset = cache_seq_len - current_seq_len;

            int token_idx_in_kv_cache = local_token_idx + cached_offset;
            // Check against BOTH total cache length (valid slot) AND input length (valid read)
            bool const valid_token = (token_idx_in_kv_cache < cache_seq_len) && (local_token_idx < current_seq_len);

            // Limit the token_idx to cache seq length (we need all threads in this block to be involved).
            token_idx_in_kv_cache = std::min(token_idx_in_kv_cache, cache_seq_len - 1);
            int const safe_local_token_idx = std::min(local_token_idx, current_seq_len - 1);
            int const global_token_idx = safe_local_token_idx + global_token_offset;

            auto const position_id
                = helix_position_offsets ? helix_position_offsets[global_token_idx] : token_idx_in_kv_cache;
            float2 const* rotary_coef_cache_buffer
                = cos_sin_cache + static_cast<size_t>(ROPE_DIM) * position_id + (head_dim_idx / 2);

            VecT q, k;
            auto const src_k_global_offset = static_cast<size_t>(global_token_idx) * (c_k + ROPE_DIM) + c_k;
            auto src_q_global_offset = static_cast<size_t>(global_token_idx) * head_num * (head_size + ROPE_DIM)
                + (head_size + ROPE_DIM) * head_idx + head_size;
            // In the absorption mode, we load pe from q_pe instead of q_ptr.
            T* q_pe_input = q_ptr;
            if (absorption_mode)
            {
                q_pe_input = q_pe;
                src_q_global_offset = static_cast<size_t>(global_token_idx) * q_pe_stride + q_pe_ld * head_idx;
            }

            q = *reinterpret_cast<VecT const*>(&q_pe_input[src_q_global_offset + head_dim_idx]);
            k = *reinterpret_cast<VecT const*>(&fuse_buf[src_k_global_offset + head_dim_idx]);

            // Pack two elements into one for gptj rotary embedding.
#pragma unroll
            for (int elt_id = 0; elt_id < ELTS_PER_VEC / 2; elt_id++)
            {
                GPTJEltT& q_ = reinterpret_cast<GPTJEltT*>(&q)[elt_id];
                GPTJEltT& k_ = reinterpret_cast<GPTJEltT*>(&k)[elt_id];

                float2 rotary_coef_cache = rotary_coef_cache_buffer[elt_id];
                mmha::apply_rotary_embedding_gptj(q_, k_, rotary_coef_cache);
            }
            // do sync
            __syncwarp();
            if (valid_token)
            {
                if (head_idx == 0)
                {
                    auto kDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_idx_in_kv_cache));
                    auto inBlockIdx = kv_cache.getKVLocalIdx(
                        token_idx_in_kv_cache, 0, TOTAL_VECS_PER_HEAD, K_VECS_PER_HEAD + head_dim_vec_idx);
                    if (cache_type == KvCacheDataType::FP8)
                    {

                        quantCopy<T, ELTS_PER_VEC>(reinterpret_cast<__nv_fp8_e4m3*>(kDst) + inBlockIdx * ELTS_PER_VEC,
                            reinterpret_cast<T const*>(&k), quant_scale_kv_val);
                    }
                    else
                        reinterpret_cast<VecT*>(kDst)[inBlockIdx] = k;
                }
                auto const dst_q_idx = static_cast<size_t>(global_token_idx) * head_num * (nope_head_size_q + ROPE_DIM)
                    + head_idx * (nope_head_size_q + ROPE_DIM) + nope_head_size_q + head_dim_idx;
                auto const dst_k_idx = static_cast<size_t>(global_token_idx) * head_num * (head_size + ROPE_DIM)
                    + head_idx * (head_size + ROPE_DIM) + head_size + head_dim_idx;
                reinterpret_cast<VecT*>(q_ptr)[dst_q_idx / ELTS_PER_VEC] = q;
                // Only write to k_pe to k_buf in the non-absorption mode.
                if (!absorption_mode)
                {
                    reinterpret_cast<VecT*>(k_ptr)[dst_k_idx / ELTS_PER_VEC] = k;
                }
            }
        }
    }
    else
    {
        int block_dim = gridDim.z - head_num;
        int block_id = head_idx - head_num;
        size_t const head_dim_vec_idx = (threadIdx.x % K_VECS_PER_HEAD);
        size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;

        size_t const seq_len_loop_end
            = size_t((max_input_seq_len + K_TOKENS_PER_BLOCK - 1) / K_TOKENS_PER_BLOCK) * K_TOKENS_PER_BLOCK;
        float quant_scale_kv_val = quant_scale_kv ? quant_scale_kv[0] : 1.f;

        // Mainloop.
        for (int local_token_idx = (threadIdx.x / K_VECS_PER_HEAD) + gridDim.x * K_TOKENS_PER_BLOCK * block_id
                 + blockIdx.x * K_TOKENS_PER_BLOCK;
             local_token_idx < seq_len_loop_end; local_token_idx += block_dim * K_TOKENS_PER_BLOCK * gridDim.x)
        {

            int const global_token_offset = cu_q_seqlens[batch_idx];
            int const cache_seq_len = kv_cache_lengths[batch_idx];

            // Derive cached offset and current input length (same as first loop)
            int const current_seq_len = cu_q_seqlens[batch_idx + 1] - global_token_offset;
            int const cached_offset = cache_seq_len - current_seq_len;

            int token_idx_in_kv_cache = local_token_idx + cached_offset;
            // Check against BOTH total cache length (valid slot) AND input length (valid read)
            bool const valid_token = (token_idx_in_kv_cache < cache_seq_len) && (local_token_idx < current_seq_len);

            // Limit the token_idx to cache seq length (we need all threads in this block to be involved).
            token_idx_in_kv_cache = std::min(token_idx_in_kv_cache, cache_seq_len - 1);
            int const safe_local_token_idx = std::min(local_token_idx, current_seq_len - 1);
            int const global_token_idx = safe_local_token_idx + global_token_offset;

            if (valid_token)
            {
                auto const src_k_global_offset = static_cast<size_t>(global_token_idx) * (c_k + ROPE_DIM);

                auto kDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_idx_in_kv_cache));
                auto inBlockIdx
                    = kv_cache.getKVLocalIdx(token_idx_in_kv_cache, 0, TOTAL_VECS_PER_HEAD, head_dim_vec_idx);
                if (cache_type == KvCacheDataType::FP8)
                {

                    quantCopy<T, ELTS_PER_VEC>(reinterpret_cast<__nv_fp8_e4m3*>(kDst) + inBlockIdx * ELTS_PER_VEC,
                        fuse_buf + src_k_global_offset + head_dim_idx, quant_scale_kv_val);
                }
                else
                    reinterpret_cast<VecT*>(kDst)[inBlockIdx]
                        = *reinterpret_cast<VecT const*>(&fuse_buf[src_k_global_offset + head_dim_idx]);
            }
        }
    }
}

template <typename T, int BLOCK_SIZE, int K_DIM, int ROPE_DIM, typename KVCacheBuffer>
__global__ void applyMLARopeAndAssignQKVKernelGeneration(T* qkv_output, T* q_pe, T const* fuse_buf, void* quant_q,
    KVCacheBuffer kv_cache, KVBlockArray kv_scale_cache, float2 const* cos_sin_cache, size_t head_num, int c_k,
    int total_s_len, int seq_len, int* seqQOffset, uint32_t* fmha_tile_counter, int32_t const* kv_cache_lengths,
    int* seqKVOffsets, int q_pe_ld, int q_pe_stride, KvCacheDataType cache_type, float* bmm1_scale, float* bmm2_scale,
    float const* quant_scale_o, float const* quant_scale_q, float const* quant_scale_kv, float const* dequant_scale_q,
    float const* dequant_scale_kv, float host_bmm1_scale, int32_t const* helix_position_offsets,
    bool const* helix_is_inactive_rank)
{
    // Constants.
    using VecT = typename VecType<T>::Type;
    using GPTJEltT = typename VecType<T>::GPTJEltType;
    constexpr auto HEAD_SIZE = ROPE_DIM;
    constexpr auto K_HEAD_SIZE = K_DIM;
    constexpr auto BYTES_PER_ELT = sizeof(T);
    constexpr auto BYTES_PER_LOAD = 16;
    constexpr auto ELTS_PER_VEC = BYTES_PER_LOAD / BYTES_PER_ELT;
    static_assert((HEAD_SIZE * BYTES_PER_ELT) % BYTES_PER_LOAD == 0, "Head size needs to be multiple of 16 bytes.");
    constexpr auto VECS_PER_HEAD = HEAD_SIZE * BYTES_PER_ELT / BYTES_PER_LOAD;
    constexpr auto K_VECS_PER_HEAD = K_HEAD_SIZE * BYTES_PER_ELT / BYTES_PER_LOAD;
    static_assert(BLOCK_SIZE % VECS_PER_HEAD == 0, "Kernel block should be able to handle entire heads.");
    constexpr auto TOKENS_PER_BLOCK = BLOCK_SIZE / VECS_PER_HEAD;
    constexpr auto K_TOKENS_PER_BLOCK = BLOCK_SIZE / K_VECS_PER_HEAD;
    constexpr auto TOTAL_VEC_PER_HEAD = VECS_PER_HEAD + K_VECS_PER_HEAD;
    // NVFP4 dense KV byte strides per token: 576 elems -> 288 packed E2M1 data
    // bytes + 36 E4M3 block-scale bytes (one scale per 16 elems).
    constexpr auto NVFP4_DATA_BYTES_PER_TOKEN = (K_DIM + ROPE_DIM) / 2;
    constexpr auto NVFP4_SCALE_BYTES_PER_TOKEN = (K_DIM + ROPE_DIM) / 16;

    // Block/Head idx.
    size_t const head_idx = blockIdx.y;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
    {
        fmha_tile_counter[0] = 0;
        seqQOffset[0] = 0;

        // Calculate bmm scale for FP8 MLA
        if (cache_type == KvCacheDataType::FP8)
        {
            float dequant_scale_q_val = dequant_scale_q ? dequant_scale_q[0] : 1.f;
            float dequant_scale_kv_val = dequant_scale_kv ? dequant_scale_kv[0] : 1.f;
            float quant_scale_o_val = quant_scale_o ? quant_scale_o[0] : 1.f;
            if (bmm1_scale)
            {
                // The scale prepared for log2 optimization.
                constexpr float kLog2e = 1.4426950408889634074f;
                // The scale after fmha bmm1.
                float bmm1_scale_val = dequant_scale_q_val * dequant_scale_kv_val * host_bmm1_scale;
                bmm1_scale[0] = bmm1_scale_val;
                bmm1_scale[1] = bmm1_scale_val * kLog2e;
            }
            if (bmm2_scale)
            {
                // The scale after fmha bmm2.
                bmm2_scale[0] = quant_scale_o_val * dequant_scale_kv_val;
            }
        }
    }

    if (head_idx <= head_num)
    {
        size_t const head_dim_vec_idx = (threadIdx.x % VECS_PER_HEAD);
        size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;

        int const seq_len_loop_end = size_t((total_s_len + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK;
        float const quant_scale_q_val = quant_scale_q ? quant_scale_q[0] : 1.0f;
        float const quant_scale_kv_val = quant_scale_kv ? quant_scale_kv[0] : 1.0f;

        // Mainloop.
        for (int global_token_idx = (threadIdx.x / VECS_PER_HEAD) + blockIdx.x * TOKENS_PER_BLOCK;
             global_token_idx < seq_len_loop_end; global_token_idx += TOKENS_PER_BLOCK * gridDim.x)
        {
            auto batch_idx = global_token_idx / seq_len;
            auto local_token_idx = global_token_idx % seq_len;
            bool const valid_token = global_token_idx < total_s_len;
            VecT data;

            if (valid_token)
            {

                auto const position_id
                    = (helix_position_offsets != nullptr ? helix_position_offsets[global_token_idx]
                                                         : kv_cache_lengths[batch_idx] - seq_len + local_token_idx);
                float2 const* rotary_coef_cache_buffer
                    = cos_sin_cache + static_cast<size_t>(ROPE_DIM) * position_id + (head_dim_idx / 2);

                if (head_idx == head_num)
                {
                    auto const src_k_global_offset = static_cast<size_t>(global_token_idx) * (c_k + ROPE_DIM) + c_k;

                    data = *reinterpret_cast<VecT const*>(&fuse_buf[src_k_global_offset + head_dim_idx]);
                }
                else
                {
                    auto const src_q_global_offset
                        = static_cast<size_t>(global_token_idx) * q_pe_stride + q_pe_ld * head_idx;

                    data = *reinterpret_cast<VecT const*>(&q_pe[src_q_global_offset + head_dim_idx]);
                }

                // Pack two elements into one for gptj rotary embedding.
#pragma unroll
                for (int elt_id = 0; elt_id < ELTS_PER_VEC / 2; elt_id++)
                {
                    GPTJEltT& data_ = reinterpret_cast<GPTJEltT*>(&data)[elt_id];

                    float2 rotary_coef_cache = rotary_coef_cache_buffer[elt_id];
                    data_ = mmha::rotary_embedding_transform(data_, rotary_coef_cache);
                }
            }

            __syncwarp();

            if (valid_token)
            {
                if (head_idx == head_num)
                {
                    // If helix parallelism is being used, only write to KV cache if current rank is active.
                    if (helix_is_inactive_rank == nullptr || !helix_is_inactive_rank[batch_idx])
                    {
                        auto const token_kv_idx = kv_cache_lengths[batch_idx] - seq_len + local_token_idx;

                        {
                            auto kDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_kv_idx));
                            auto inBlockIdx = kv_cache.getKVLocalIdx(
                                token_kv_idx, 0, TOTAL_VEC_PER_HEAD, K_VECS_PER_HEAD + head_dim_vec_idx);
                            if (cache_type == KvCacheDataType::NVFP4)
                            {
                                // rope part occupies the trailing vecs of the
                                // 576-wide latent: vecIdx = K_VECS_PER_HEAD + ...
                                quantCopyNvfp4<T, ELTS_PER_VEC, NVFP4_DATA_BYTES_PER_TOKEN,
                                    NVFP4_SCALE_BYTES_PER_TOKEN>(reinterpret_cast<uint8_t*>(kDst),
                                    reinterpret_cast<uint8_t*>(kv_scale_cache.getKBlockPtr(batch_idx, token_kv_idx)),
                                    kv_scale_cache.getLocalIdx(token_kv_idx), K_VECS_PER_HEAD + head_dim_vec_idx,
                                    reinterpret_cast<T const*>(&data));
                            }
                            else if (cache_type == KvCacheDataType::FP8)
                            {

                                quantCopy<T, ELTS_PER_VEC>(
                                    reinterpret_cast<__nv_fp8_e4m3*>(kDst) + inBlockIdx * ELTS_PER_VEC,
                                    reinterpret_cast<T const*>(&data), quant_scale_kv_val);
                            }
                            else
                                reinterpret_cast<VecT*>(kDst)[inBlockIdx] = data;
                        }
                    }
                }
                else
                {
                    auto const dst_q_idx = static_cast<size_t>(global_token_idx) * head_num * (c_k + ROPE_DIM)
                        + head_idx * (c_k + ROPE_DIM) + c_k + head_dim_idx;
                    if (cache_type == KvCacheDataType::FP8)
                    {
                        quantCopy<T, ELTS_PER_VEC>(reinterpret_cast<__nv_fp8_e4m3*>(quant_q) + dst_q_idx,
                            reinterpret_cast<T const*>(&data), quant_scale_q_val);
                    }
                    else
                        reinterpret_cast<VecT*>(qkv_output)[dst_q_idx / ELTS_PER_VEC] = data;
                }
            }
        }
    }
    else if (head_idx <= head_num + 8)
    {
        int block_dim = gridDim.y - head_num - 1;
        int block_id = head_idx - head_num - 1;
        size_t const head_dim_vec_idx = (threadIdx.x % K_VECS_PER_HEAD);
        size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;

        size_t const seq_len_loop_end
            = size_t((total_s_len + K_TOKENS_PER_BLOCK - 1) / K_TOKENS_PER_BLOCK) * K_TOKENS_PER_BLOCK;
        float quant_scale_kv_val = quant_scale_kv ? quant_scale_kv[0] : 1.0f;

        // Mainloop.
        for (int global_token_idx = (threadIdx.x / K_VECS_PER_HEAD) + gridDim.x * K_TOKENS_PER_BLOCK * block_id
                 + blockIdx.x * K_TOKENS_PER_BLOCK;
             global_token_idx < seq_len_loop_end; global_token_idx += block_dim * K_TOKENS_PER_BLOCK * gridDim.x)
        {
            auto batch_idx = global_token_idx / seq_len;
            auto local_token_idx = global_token_idx % seq_len;
            bool valid_token = global_token_idx < total_s_len;

            if (valid_token)
            {
                if (head_dim_vec_idx == 0)
                {
                    seqQOffset[batch_idx + 1] = head_num * seq_len * (batch_idx + 1);
                }

                // If helix parallelism is being used, only write to KV cache if current rank is active.
                if (helix_is_inactive_rank == nullptr || !helix_is_inactive_rank[batch_idx])
                {
                    auto const token_kv_idx = kv_cache_lengths[batch_idx] - seq_len + local_token_idx;
                    auto const src_kv_global_offset = static_cast<size_t>(global_token_idx) * (c_k + ROPE_DIM);

                    {
                        auto kDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_kv_idx));
                        auto inBlockIdx = kv_cache.getKVLocalIdx(token_kv_idx, 0, TOTAL_VEC_PER_HEAD, head_dim_vec_idx);

                        if (cache_type == KvCacheDataType::NVFP4)
                        {
                            // latent (nope) part occupies the leading vecs of
                            // the 576-wide latent: vecIdx = head_dim_vec_idx.
                            quantCopyNvfp4<T, ELTS_PER_VEC, NVFP4_DATA_BYTES_PER_TOKEN, NVFP4_SCALE_BYTES_PER_TOKEN>(
                                reinterpret_cast<uint8_t*>(kDst),
                                reinterpret_cast<uint8_t*>(kv_scale_cache.getKBlockPtr(batch_idx, token_kv_idx)),
                                kv_scale_cache.getLocalIdx(token_kv_idx), head_dim_vec_idx,
                                &fuse_buf[src_kv_global_offset + head_dim_idx]);
                        }
                        else if (cache_type == KvCacheDataType::FP8)
                        {
                            quantCopy<T, ELTS_PER_VEC>(
                                reinterpret_cast<__nv_fp8_e4m3*>(kDst) + inBlockIdx * ELTS_PER_VEC,
                                fuse_buf + src_kv_global_offset + head_dim_idx, quant_scale_kv_val);
                        }
                        else
                            reinterpret_cast<VecT*>(kDst)[inBlockIdx]
                                = *reinterpret_cast<VecT const*>(&fuse_buf[src_kv_global_offset + head_dim_idx]);
                    }
                }
            }
        }
    }
    else
    {
        if (cache_type == KvCacheDataType::FP8)
        {
            int block_dim = gridDim.y - head_num - 1 - 8;
            int block_id = head_idx - head_num - 1 - 8;
            size_t const head_dim_vec_idx = (threadIdx.x % K_VECS_PER_HEAD);
            size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;
            size_t const head_num_idx = (block_id % head_num) * (K_HEAD_SIZE + HEAD_SIZE);

            size_t const seq_len_loop_end
                = size_t((total_s_len + K_TOKENS_PER_BLOCK - 1) / K_TOKENS_PER_BLOCK) * K_TOKENS_PER_BLOCK;
            float quant_scale_q_val = quant_scale_q ? quant_scale_q[0] : 1.0f;

            // Mainloop.
            for (int global_token_idx = (threadIdx.x / K_VECS_PER_HEAD)
                     + (block_id / head_num) * gridDim.x * K_TOKENS_PER_BLOCK + blockIdx.x * K_TOKENS_PER_BLOCK;
                 global_token_idx < seq_len_loop_end;
                 global_token_idx += (block_dim / head_num) * gridDim.x * K_TOKENS_PER_BLOCK)
            {
                if (global_token_idx < total_s_len)
                {
                    size_t const load_idx
                        = global_token_idx * head_num * (K_HEAD_SIZE + HEAD_SIZE) + head_num_idx + head_dim_idx;
                    quantCopy<T, ELTS_PER_VEC>(
                        reinterpret_cast<__nv_fp8_e4m3*>(quant_q) + load_idx, qkv_output + load_idx, quant_scale_q_val);
                }
            }
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif

    // The implementation of the parallel scan in the thread block (see CUB for details).
    using BlockScan = cub::BlockScan<int, BLOCK_SIZE>;

    // Allocate storage in shared memory to do the scan.
    __shared__ typename BlockScan::TempStorage tempKVStorage;
    BlockPrefixCallbackOp prefixKVOp(0);

    if (blockIdx.x == 0 && blockIdx.y == 0)
    {
        int const batchSizeBound = total_s_len / seq_len;
        for (int batchOffset = 0; batchOffset <= batchSizeBound; batchOffset += BLOCK_SIZE)
        {
            // The index of the batch.
            int batchIdx = batchOffset + threadIdx.x;
            int seqKVLength = 0;
            if (batchIdx < batchSizeBound)
            {
                seqKVLength = kv_cache_lengths[batchIdx];
            }
            int seqKVOffset;
            BlockScan(tempKVStorage).ExclusiveSum(seqKVLength, seqKVOffset, prefixKVOp);
            if (batchIdx <= batchSizeBound)
            {
                seqKVOffsets[batchIdx] = seqKVOffset;
            }
        }
    }
}

template <typename T, typename TCache>
__global__ void loadPagedKVCacheForMLAKernel(T* compressed_kv_ptr, T* k_pe_ptr,
    tensorrt_llm::kernels::KVBlockArray const kv_cache, int64_t const* cu_ctx_cached_kv_lens, int max_input_seq_len,
    float const* kv_scale_quant_orig_ptr, __half const* kvarn_scale_pool_ptr = nullptr, int const kvarn_bits = 4)
{
    static_assert(std::is_same_v<T, TCache> || std::is_same_v<TCache, __nv_fp8_e4m3>,
        "TCache must be either the same type as T or __nv_fp8_e4m3");
    // KVarN/BDR: ckv stored as INT4 in the fp8 byte-storage path (2 nibbles/byte) +
    // per-(token,sub-block) {scale,zp} in kvarn_scale_pool_ptr. dequant-on-read is
    // unpack+(q*scale+zp) -> ROTATED-frame fp16 (Q-side fold un-rotates downstream).
    using KT = typename tensorrt_llm::kernels::loadPagedKVKernelTraits<TCache>;
    constexpr int kKvarnHOrder = 128;
    constexpr int kKvarnNSub = KT::kLoraSize / kKvarnHOrder;          // 512/128 = 4
    constexpr int kKvarnScaleStride = 2 * kKvarnNSub;                 // scale[nsub]|zp[nsub]

    int const batch_idx = static_cast<int>(blockIdx.y);
    float const kv_scale_quant_orig = kv_scale_quant_orig_ptr ? kv_scale_quant_orig_ptr[0] : 1.0f;

    size_t const head_dim_vec_idx = (threadIdx.x % KT::kVecPerHead);
    size_t const head_dim_idx = head_dim_vec_idx * KT::kElemPerLoad;
    bool const is_valid_kv = head_dim_vec_idx < KT::kKVThreadPerHead;

    size_t const seq_len_loop_end
        = (max_input_seq_len + KT::kTokenPerBlock - 1) / KT::kTokenPerBlock * KT::kTokenPerBlock;

    int64_t const global_token_offset = cu_ctx_cached_kv_lens[batch_idx];
    int64_t const cache_kv_len = cu_ctx_cached_kv_lens[batch_idx + 1] - cu_ctx_cached_kv_lens[batch_idx];

    for (int local_token_idx = (threadIdx.x / KT::kThreadPerHead) + blockIdx.x * KT::kTokenPerBlock;
         local_token_idx < seq_len_loop_end; local_token_idx += KT::kTokenPerBlock * gridDim.x)
    {
        int token_idx_in_kv_cache = local_token_idx;
        bool const valid_token = token_idx_in_kv_cache < cache_kv_len;

        if (valid_token)
        {
            auto* kvSrc = reinterpret_cast<TCache*>(kv_cache.getKBlockPtr(batch_idx, token_idx_in_kv_cache));
            // head_idx === 0
            auto kvBlockIdx
                = kv_cache.getKVLocalIdx(token_idx_in_kv_cache, 0, KT::kVecPerHead, static_cast<int>(head_dim_vec_idx));

            auto src_data = reinterpret_cast<typename KT::VecT*>(kvSrc)[kvBlockIdx];

            int const global_token_idx = local_token_idx + global_token_offset;

            if (is_valid_kv)
            {
                // compressed_kv {total_token, lora_size}
                int const dstIdx = global_token_idx * KT::kLoraSize + head_dim_idx;

                // copy back to compressed_kv
                if constexpr (std::is_same_v<TCache, T>)
                {
                    *reinterpret_cast<typename KT::VecT*>(compressed_kv_ptr + dstIdx) = src_data;
                }
                else if constexpr (std::is_same_v<TCache, __nv_fp8_e4m3>)
                {
                    if (kvarn_scale_pool_ptr != nullptr)
                    {
                        // KVarN: which 128-wide sub-block this vec falls in.
                        int const sub = head_dim_idx / kKvarnHOrder;
                        __half const* sc = kvarn_scale_pool_ptr
                            + static_cast<size_t>(global_token_idx) * kKvarnScaleStride;
                        float const scale = __half2float(sc[sub]);
                        float const zp = __half2float(sc[kKvarnNSub + sub]);
                        // src_data holds KT::kElemPerLoad low-bit codes, packed low-first.
                        if (kvarn_bits == 2)
                        {
                            dequantCopyKVarN<T, KT::kElemPerLoad, 2>(compressed_kv_ptr + dstIdx,
                                reinterpret_cast<uint8_t const*>(&src_data), scale, zp);
                        }
                        else
                        {
                            dequantCopyKVarN<T, KT::kElemPerLoad, 4>(compressed_kv_ptr + dstIdx,
                                reinterpret_cast<uint8_t const*>(&src_data), scale, zp);
                        }
                    }
                    else
                    {
                        dequantCopy<T, KT::kElemPerLoad>(compressed_kv_ptr + dstIdx,
                            reinterpret_cast<__nv_fp8_e4m3 const*>(&src_data), kv_scale_quant_orig);
                    }
                }
            }
            else
            {
                // k_pe {total_token, rope_size}
                int const dstIdx = global_token_idx * KT::kRopeSize + (head_dim_idx - KT::kLoraSize);

                // copy back to k_pe
                if constexpr (std::is_same_v<TCache, T>)
                {
                    *reinterpret_cast<typename KT::VecT*>(k_pe_ptr + dstIdx) = src_data;
                }
                else if constexpr (std::is_same_v<TCache, __nv_fp8_e4m3>)
                {
                    dequantCopy<T, KT::kElemPerLoad>(
                        k_pe_ptr + dstIdx, reinterpret_cast<__nv_fp8_e4m3 const*>(&src_data), kv_scale_quant_orig);
                }
            }
        }
    }
}

// q {total_uncached_tokens, h, d_nope + d_rope}
// latent_cache {total_uncached_tokens, d_k + d_rope}
template <typename T, typename TCache, int BLOCK_SIZE, int K_DIM, int ROPE_DIM>
__global__ void applyMLARopeAppendPagedKVAssignQKernel(KVBlockArray kv_cache, KVBlockArray kv_scale_cache, T* q_ptr,
    T* latent_cache_ptr, int64_t const* cu_ctx_cached_kv_lens, int64_t const* cu_seq_lens,
    int const max_input_uncached_seq_len, float2 const* cos_sin_cache, size_t head_num, int nope_size,
    float const* kv_scale_orig_quant_ptr, KvCacheDataType cache_type)
{
    static_assert(std::is_same_v<T, TCache> || std::is_same_v<TCache, __nv_fp8_e4m3>,
        "TCache must be either the same type as T or __nv_fp8_e4m3 (NVFP4 reuses the fp8 byte-storage path)");
    // Constants.
    using VecT = typename VecType<T>::Type;
    using GPTJEltT = typename VecType<T>::GPTJEltType;
    constexpr auto HEAD_SIZE = ROPE_DIM;
    constexpr auto K_HEAD_SIZE = K_DIM;
    constexpr auto BYTES_PER_ELT = sizeof(T);
    constexpr auto BYTES_PER_LOAD = 16;
    constexpr auto ELTS_PER_VEC = BYTES_PER_LOAD / BYTES_PER_ELT;
    static_assert((HEAD_SIZE * BYTES_PER_ELT) % BYTES_PER_LOAD == 0, "Head size needs to be multiple of 16 bytes.");
    constexpr auto VECS_PER_HEAD = HEAD_SIZE * BYTES_PER_ELT / BYTES_PER_LOAD;
    constexpr auto K_VECS_PER_HEAD = K_HEAD_SIZE * BYTES_PER_ELT / BYTES_PER_LOAD;
    static_assert(BLOCK_SIZE % VECS_PER_HEAD == 0, "Kernel block should be able to handle entire heads.");
    constexpr auto TOKENS_PER_BLOCK = BLOCK_SIZE / VECS_PER_HEAD;
    constexpr auto K_TOKENS_PER_BLOCK = BLOCK_SIZE / K_VECS_PER_HEAD;
    constexpr auto TOTAL_VECS_PER_HEAD = VECS_PER_HEAD + K_VECS_PER_HEAD;
    // NVFP4 dense KV byte strides per token (see generation kernel).
    constexpr auto NVFP4_DATA_BYTES_PER_TOKEN = (K_DIM + ROPE_DIM) / 2;
    constexpr auto NVFP4_SCALE_BYTES_PER_TOKEN = (K_DIM + ROPE_DIM) / 16;

    // Block/Head idx.
    size_t const batch_idx = blockIdx.y;
    size_t const head_idx = blockIdx.z;

    int64_t const global_token_offset = cu_seq_lens[batch_idx] - cu_ctx_cached_kv_lens[batch_idx];
    int64_t const cached_kv_len = cu_ctx_cached_kv_lens[batch_idx + 1] - cu_ctx_cached_kv_lens[batch_idx];
    int64_t const uncached_kv_len = cu_seq_lens[batch_idx + 1] - cu_seq_lens[batch_idx] - cached_kv_len;

    if (head_idx <= head_num)
    {
        size_t const head_dim_vec_idx = (threadIdx.x % VECS_PER_HEAD);
        size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;

        size_t const seq_len_loop_end
            = size_t((max_input_uncached_seq_len + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK;
        float quant_scale_kv_val = kv_scale_orig_quant_ptr ? kv_scale_orig_quant_ptr[0] : 1.f;

        // Mainloop.
        for (int local_token_idx = (threadIdx.x / VECS_PER_HEAD) + blockIdx.x * TOKENS_PER_BLOCK;
             local_token_idx < seq_len_loop_end; local_token_idx += TOKENS_PER_BLOCK * gridDim.x)
        {

            int token_idx_in_kv_cache = local_token_idx + cached_kv_len;
            bool valid_token = local_token_idx < uncached_kv_len;
            int const global_token_idx = local_token_idx + global_token_offset;
            VecT data;

            if (valid_token)
            {
                auto const position_id = token_idx_in_kv_cache;
                float2 const* rotary_coef_cache_buffer
                    = cos_sin_cache + static_cast<size_t>(ROPE_DIM) * position_id + (head_dim_idx / 2);

                if (head_idx == head_num)
                {
                    auto const src_k_global_offset = static_cast<size_t>(global_token_idx) * (K_DIM + ROPE_DIM) + K_DIM;
                    data = *reinterpret_cast<VecT const*>(&latent_cache_ptr[src_k_global_offset + head_dim_idx]);
                }
                else
                {
                    auto const src_q_global_offset
                        = static_cast<size_t>(global_token_idx) * head_num * (nope_size + ROPE_DIM)
                        + (nope_size + ROPE_DIM) * head_idx + nope_size;
                    data = *reinterpret_cast<VecT const*>(&q_ptr[src_q_global_offset + head_dim_idx]);
                }

                // Pack two elements into one for gptj rotary embedding.
#pragma unroll
                for (int elt_id = 0; elt_id < ELTS_PER_VEC / 2; elt_id++)
                {
                    GPTJEltT& data_ = reinterpret_cast<GPTJEltT*>(&data)[elt_id];

                    float2 rotary_coef_cache = rotary_coef_cache_buffer[elt_id];
                    data_ = mmha::rotary_embedding_transform(data_, rotary_coef_cache);
                }
            }
            // do sync
            __syncwarp();
            if (valid_token)
            {
                if (head_idx == head_num)
                {
                    auto kDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_idx_in_kv_cache));
                    auto inBlockIdx = kv_cache.getKVLocalIdx(
                        token_idx_in_kv_cache, 0, TOTAL_VECS_PER_HEAD, K_VECS_PER_HEAD + head_dim_vec_idx);
                    if constexpr (std::is_same_v<TCache, __nv_fp8_e4m3>)
                    {
                        if (cache_type == KvCacheDataType::NVFP4)
                        {
                            // rope part: trailing vecs of the 576-wide latent.
                            quantCopyNvfp4<T, ELTS_PER_VEC, NVFP4_DATA_BYTES_PER_TOKEN, NVFP4_SCALE_BYTES_PER_TOKEN>(
                                reinterpret_cast<uint8_t*>(kDst),
                                reinterpret_cast<uint8_t*>(
                                    kv_scale_cache.getKBlockPtr(batch_idx, token_idx_in_kv_cache)),
                                kv_cache.getLocalIdx(token_idx_in_kv_cache), K_VECS_PER_HEAD + head_dim_vec_idx,
                                reinterpret_cast<T const*>(&data));
                        }
                        else
                        {
                            quantCopy<T, ELTS_PER_VEC>(
                                reinterpret_cast<__nv_fp8_e4m3*>(kDst) + inBlockIdx * ELTS_PER_VEC,
                                reinterpret_cast<T const*>(&data), quant_scale_kv_val);
                        }
                    }
                    else
                    {
                        reinterpret_cast<VecT*>(kDst)[inBlockIdx] = data;
                    }
                    // copy to latent_cache (for chunked prefill, it will not load kv cache for uncached k_pe)
                    // we only need to copy original value.
                    auto const src_k_global_offset = static_cast<size_t>(global_token_idx) * (K_DIM + ROPE_DIM) + K_DIM;
                    *reinterpret_cast<VecT*>(&latent_cache_ptr[src_k_global_offset + head_dim_idx]) = data;
                }
                else
                {
                    auto const dst_q_idx = static_cast<size_t>(global_token_idx) * head_num * (nope_size + ROPE_DIM)
                        + head_idx * (nope_size + ROPE_DIM) + nope_size + head_dim_idx;
                    reinterpret_cast<VecT*>(q_ptr)[dst_q_idx / ELTS_PER_VEC] = data;
                }
            }
        }
    }
    else
    {
        int block_dim = gridDim.z - head_num - 1;
        int block_id = head_idx - head_num - 1;
        size_t const head_dim_vec_idx = (threadIdx.x % K_VECS_PER_HEAD);
        size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;

        size_t const seq_len_loop_end
            = size_t((max_input_uncached_seq_len + K_TOKENS_PER_BLOCK - 1) / K_TOKENS_PER_BLOCK) * K_TOKENS_PER_BLOCK;
        float quant_scale_kv_val = kv_scale_orig_quant_ptr ? kv_scale_orig_quant_ptr[0] : 1.f;

        // Mainloop.
        for (int local_token_idx = (threadIdx.x / K_VECS_PER_HEAD) + gridDim.x * K_TOKENS_PER_BLOCK * block_id
                 + blockIdx.x * K_TOKENS_PER_BLOCK;
             local_token_idx < seq_len_loop_end; local_token_idx += block_dim * K_TOKENS_PER_BLOCK * gridDim.x)
        {

            int token_idx_in_kv_cache = local_token_idx + cached_kv_len;
            bool valid_token = local_token_idx < uncached_kv_len;
            int const global_token_idx = local_token_idx + global_token_offset;

            if (valid_token)
            {
                auto const src_k_global_offset = static_cast<size_t>(global_token_idx) * (K_DIM + ROPE_DIM);

                auto kDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_idx_in_kv_cache));
                auto inBlockIdx
                    = kv_cache.getKVLocalIdx(token_idx_in_kv_cache, 0, TOTAL_VECS_PER_HEAD, head_dim_vec_idx);
                if constexpr (std::is_same_v<TCache, __nv_fp8_e4m3>)
                {
                    if (cache_type == KvCacheDataType::NVFP4)
                    {
                        // latent (nope) part: leading vecs of the 576-wide latent.
                        quantCopyNvfp4<T, ELTS_PER_VEC, NVFP4_DATA_BYTES_PER_TOKEN, NVFP4_SCALE_BYTES_PER_TOKEN>(
                            reinterpret_cast<uint8_t*>(kDst),
                            reinterpret_cast<uint8_t*>(kv_scale_cache.getKBlockPtr(batch_idx, token_idx_in_kv_cache)),
                            kv_cache.getLocalIdx(token_idx_in_kv_cache), head_dim_vec_idx,
                            &latent_cache_ptr[src_k_global_offset + head_dim_idx]);
                    }
                    else
                    {
                        quantCopy<T, ELTS_PER_VEC>(reinterpret_cast<__nv_fp8_e4m3*>(kDst) + inBlockIdx * ELTS_PER_VEC,
                            latent_cache_ptr + src_k_global_offset + head_dim_idx, quant_scale_kv_val);
                    }
                }
                else
                {
                    reinterpret_cast<VecT*>(kDst)[inBlockIdx]
                        = *reinterpret_cast<VecT const*>(&latent_cache_ptr[src_k_global_offset + head_dim_idx]);
                }
            }
        }
    }
}

template <typename T, int BLOCK_SIZE, int QK_NOPE_HEAD_DIM, int QK_ROPE_HEAD_DIM, int V_HEAD_DIM, bool ABSORPTION_MODE>
__global__ void quantizeCopyInputToFp8Kernel(T const* q_buf, __nv_fp8_e4m3* quant_q_buf, T const* k_buf,
    __nv_fp8_e4m3* quant_k_buf, T const* v_buf, __nv_fp8_e4m3* quant_v_buf, int total_q_len, int total_kv_len,
    float const* quant_scale_qkv_ptr, float* bmm1_scale, float* bmm2_scale, float const* quant_scale_o,
    float const* dequant_scale_q, float const* dequant_scale_kv, float host_bmm1_scale)
{
    // Constants.
    using VecT = typename VecType<T>::Type;
    constexpr auto BYTES_PER_ELT = sizeof(T);
    constexpr auto BYTES_PER_LOAD = 16;
    constexpr auto ELTS_PER_VEC = BYTES_PER_LOAD / BYTES_PER_ELT;
    constexpr auto QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM;
    static_assert(
        (QK_HEAD_DIM * BYTES_PER_ELT) % BYTES_PER_LOAD == 0, "QK head size needs to be multiple of 16 bytes.");
    static_assert((V_HEAD_DIM * BYTES_PER_ELT) % BYTES_PER_LOAD == 0, "V head size needs to be multiple of 16 bytes.");
    constexpr auto QK_VECS_PER_HEAD = QK_HEAD_DIM * BYTES_PER_ELT / BYTES_PER_LOAD;
    constexpr auto V_VECS_PER_HEAD = V_HEAD_DIM * BYTES_PER_ELT / BYTES_PER_LOAD;
    static_assert(BLOCK_SIZE % QK_VECS_PER_HEAD == 0, "Kernel block should be able to handle entire heads.");
    static_assert(ABSORPTION_MODE || (BLOCK_SIZE % V_VECS_PER_HEAD) == 0,
        "Kernel block should be able to handle entire heads in non-absorption mode.");
    constexpr auto QK_TOKENS_PER_BLOCK = BLOCK_SIZE / QK_VECS_PER_HEAD;
    constexpr auto V_TOKENS_PER_BLOCK = BLOCK_SIZE / V_VECS_PER_HEAD;

    size_t const head_idx = blockIdx.z;
    size_t const head_num = gridDim.z;

    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
    {
        // Calculate bmm scale for FP8 MLA
        float dequant_scale_q_val = dequant_scale_q ? dequant_scale_q[0] : 1.f;
        float dequant_scale_kv_val = dequant_scale_kv ? dequant_scale_kv[0] : 1.f;
        float quant_scale_o_val = quant_scale_o ? quant_scale_o[0] : 1.f;
        if (bmm1_scale)
        {
            // The scale prepared for log2 optimization.
            constexpr float kLog2e = 1.4426950408889634074f;
            // The scale after fmha bmm1.
            float bmm1_scale_val = dequant_scale_q_val * dequant_scale_kv_val * host_bmm1_scale;
            bmm1_scale[0] = bmm1_scale_val;
            bmm1_scale[1] = bmm1_scale_val * kLog2e;
        }
        if (bmm2_scale)
        {
            // The scale after fmha bmm2.
            bmm2_scale[0] = quant_scale_o_val * dequant_scale_kv_val;
        }
    }

    size_t const qk_head_dim_vec_idx = (threadIdx.x % QK_VECS_PER_HEAD);
    size_t const v_head_dim_vec_idx = (threadIdx.x % V_VECS_PER_HEAD);
    size_t const qk_head_dim_idx = qk_head_dim_vec_idx * ELTS_PER_VEC;
    size_t const v_head_dim_idx = v_head_dim_vec_idx * ELTS_PER_VEC;

    size_t const q_len_loop_end
        = size_t((total_q_len + QK_TOKENS_PER_BLOCK - 1) / QK_TOKENS_PER_BLOCK) * QK_TOKENS_PER_BLOCK;
    size_t const k_len_loop_end
        = size_t((total_kv_len + QK_TOKENS_PER_BLOCK - 1) / QK_TOKENS_PER_BLOCK) * QK_TOKENS_PER_BLOCK;
    size_t const v_len_loop_end
        = size_t((total_kv_len + V_TOKENS_PER_BLOCK - 1) / V_TOKENS_PER_BLOCK) * V_TOKENS_PER_BLOCK;
    float quant_scale_qkv_val = quant_scale_qkv_ptr ? quant_scale_qkv_ptr[0] : 1.f;

    // Quantize Q, both src and dst are contiguous
    for (int q_token_idx = (threadIdx.x / QK_VECS_PER_HEAD) + blockIdx.x * QK_TOKENS_PER_BLOCK;
         q_token_idx < q_len_loop_end; q_token_idx += QK_TOKENS_PER_BLOCK * gridDim.x)
    {
        if (q_token_idx < total_q_len)
        {
            auto const src_q_idx
                = static_cast<size_t>(q_token_idx) * QK_HEAD_DIM * head_num + head_idx * QK_HEAD_DIM + qk_head_dim_idx;
            auto const dst_q_idx = src_q_idx;
            quantCopy<T, ELTS_PER_VEC>(quant_q_buf + dst_q_idx, &q_buf[src_q_idx], quant_scale_qkv_val);
        }
    }

    // Only quantize K and V in non-absorption mode.
    if constexpr (!ABSORPTION_MODE)
    {
        // Quantize K, both src and dst are contiguous
        for (int k_token_idx = (threadIdx.x / QK_VECS_PER_HEAD) + blockIdx.x * QK_TOKENS_PER_BLOCK;
             k_token_idx < k_len_loop_end; k_token_idx += QK_TOKENS_PER_BLOCK * gridDim.x)
        {
            if (k_token_idx < total_kv_len)
            {
                auto const src_k_idx = static_cast<size_t>(k_token_idx) * QK_HEAD_DIM * head_num
                    + head_idx * QK_HEAD_DIM + qk_head_dim_idx;
                auto const dst_k_idx = src_k_idx;
                quantCopy<T, ELTS_PER_VEC>(quant_k_buf + dst_k_idx, &k_buf[src_k_idx], quant_scale_qkv_val);
            }
        }
        // Quantize V, dst V is contiguous, but src V is not contiguous, so we need to calculate the stride
        size_t const src_v_token_stride = (QK_NOPE_HEAD_DIM + V_HEAD_DIM) * head_num;
        for (int v_token_idx = (threadIdx.x / V_VECS_PER_HEAD) + blockIdx.x * V_TOKENS_PER_BLOCK;
             v_token_idx < v_len_loop_end; v_token_idx += V_TOKENS_PER_BLOCK * gridDim.x)
        {
            if (v_token_idx < total_kv_len)
            {
                auto const src_v_idx
                    = static_cast<size_t>(v_token_idx) * src_v_token_stride + head_idx * V_HEAD_DIM + v_head_dim_idx;
                auto const dst_v_idx
                    = static_cast<size_t>(v_token_idx) * V_HEAD_DIM * head_num + head_idx * V_HEAD_DIM + v_head_dim_idx;
                quantCopy<T, ELTS_PER_VEC>(quant_v_buf + dst_v_idx, &v_buf[src_v_idx], quant_scale_qkv_val);
            }
        }
    }
}

template <typename T, typename KVCacheBuffer>
void invokeMLARopeContext(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream)
{
    dim3 grid(int(tensorrt_llm::common::divUp(params.max_input_seq_len, 32)), params.batch_size, params.head_num + 8);
    auto head_size = params.meta.qk_nope_head_dim;
    if (params.meta.rope_append)
    {
        applyMLARopeAndAssignQKVKernelOptContext<T, 256, 512, 64, KVCacheBuffer><<<grid, 256, 0, stream>>>(params.q_buf,
            params.q_pe, params.k_buf, params.latent_cache, kv_cache_buffer, params.q_pe_ld, params.q_pe_stride,
            params.cos_sin_cache, params.head_num, head_size, params.meta.kv_lora_rank, params.cu_q_seqlens,
            params.cache_seq_lens, params.max_input_seq_len, params.cache_type, params.quant_scale_kv,
            params.helix_position_offsets, params.absorption_mode);
    }
    else
    {
        applyMLARopeAndAssignQKVKernelOptContext<T, 256, 448, 64, KVCacheBuffer><<<grid, 256, 0, stream>>>(params.q_buf,
            params.q_pe, params.k_buf, params.latent_cache, kv_cache_buffer, params.q_pe_ld, params.q_pe_stride,
            params.cos_sin_cache, params.head_num, head_size, params.meta.kv_lora_rank, params.cu_q_seqlens,
            params.cache_seq_lens, params.max_input_seq_len, params.cache_type, params.quant_scale_kv,
            params.helix_position_offsets, params.absorption_mode);
    }
}

template <typename T>
void invokeMLAContextFp8Quantize(MlaParams<T>& params, int total_kv_len, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(params.cache_type == KvCacheDataType::FP8, "MLA Context: cache_type must be FP8");
    TLLM_CHECK_WITH_INFO(params.q_buf != nullptr, "MLA Context: q_buf must be non-null");
    TLLM_CHECK_WITH_INFO(params.absorption_mode || params.k_buf != nullptr,
        "MLA Context: k_buf must be non-null in non-absorption mode");
    TLLM_CHECK_WITH_INFO(params.absorption_mode || params.v_buf != nullptr,
        "MLA Context: v_buf must be non-null in non-absorption mode");
    TLLM_CHECK_WITH_INFO(params.quant_q_buf != nullptr, "MLA Context: quant_q_buf must be non-null");
    TLLM_CHECK_WITH_INFO(params.absorption_mode || params.quant_k_buf != nullptr,
        "MLA Context: quant_k_buf must be non-null in non-absorption mode");
    TLLM_CHECK_WITH_INFO(params.absorption_mode || params.quant_v_buf != nullptr,
        "MLA Context: quant_v_buf must be non-null in non-absorption mode");

    TLLM_LOG_DEBUG("MLA RoPE Context: Quantizing separate qkv to FP8");

    if (params.acc_q_len > 0)
    {
        // The Q tensor has layout of [num_tokens, head_num, 576] in the absorption mode.
        // Convert Q to FP8 in absorption mode.
        if (params.absorption_mode)
        {

            if (params.meta.rope_append)
            {
                constexpr int threads_per_block = 288;
                constexpr int num_tokens_per_block = threads_per_block * 16 / 576 * sizeof(T);
                dim3 grid(int(tensorrt_llm::common::divUp(total_kv_len, num_tokens_per_block)), 1, params.head_num);

                TLLM_LOG_DEBUG(
                    "Launching quantizeCopyInputToFp8Kernel with grid_size: (%d, %d, %d), threads_per_block: %d, "
                    "total_kv_len: %d, acc_q_len: %d, absorption_mode: %d",
                    grid.x, grid.y, grid.z, threads_per_block, total_kv_len, params.acc_q_len, params.absorption_mode);

                quantizeCopyInputToFp8Kernel<T, threads_per_block, 512, 64, 512, true>
                    <<<grid, threads_per_block, 0, stream>>>(params.q_buf,
                        static_cast<__nv_fp8_e4m3*>(params.quant_q_buf), params.k_buf,
                        static_cast<__nv_fp8_e4m3*>(params.quant_k_buf), params.v_buf,
                        static_cast<__nv_fp8_e4m3*>(params.quant_v_buf), params.acc_q_len, total_kv_len,
                        params.quant_scale_qkv, params.bmm1_scale, params.bmm2_scale, params.quant_scale_o,
                        params.dequant_scale_q, params.dequant_scale_kv, params.host_bmm1_scale);
            }
            else
            {
                constexpr int threads_per_block = 256;
                constexpr int num_tokens_per_block = threads_per_block * 16 / 512 * sizeof(T);
                dim3 grid(int(tensorrt_llm::common::divUp(total_kv_len, num_tokens_per_block)), 1, params.head_num);

                TLLM_LOG_DEBUG(
                    "Launching quantizeCopyInputToFp8Kernel with grid_size: (%d, %d, %d), threads_per_block: %d, "
                    "total_kv_len: %d, acc_q_len: %d, absorption_mode: %d",
                    grid.x, grid.y, grid.z, threads_per_block, total_kv_len, params.acc_q_len, params.absorption_mode);

                quantizeCopyInputToFp8Kernel<T, threads_per_block, 448, 64, 512, true>
                    <<<grid, threads_per_block, 0, stream>>>(params.q_buf,
                        static_cast<__nv_fp8_e4m3*>(params.quant_q_buf), params.k_buf,
                        static_cast<__nv_fp8_e4m3*>(params.quant_k_buf), params.v_buf,
                        static_cast<__nv_fp8_e4m3*>(params.quant_v_buf), params.acc_q_len, total_kv_len,
                        params.quant_scale_qkv, params.bmm1_scale, params.bmm2_scale, params.quant_scale_o,
                        params.dequant_scale_q, params.dequant_scale_kv, params.host_bmm1_scale);
            }
        }
        else
        {
            // The Q or K tensor has layout of [num_tokens, head_num, 192] in the non-absorption mode.
            // The V tensor has layout of [num_tokens, head_num, 128] in the non-absorption mode.
            // Convert Q, K, V to FP8 in non-absorption mode.

            constexpr int threads_per_block = 384;
            constexpr int num_tokens_per_block = threads_per_block * 16 / 192 * sizeof(T);
            dim3 grid(int(tensorrt_llm::common::divUp(total_kv_len, num_tokens_per_block)), 1, params.head_num);

            TLLM_LOG_DEBUG(
                "Launching quantizeCopyInputToFp8Kernel with grid_size: (%d, %d, %d), threads_per_block: %d, "
                "total_kv_len: %d, acc_q_len: %d, absorption_mode: %d",
                grid.x, grid.y, grid.z, threads_per_block, total_kv_len, params.acc_q_len, params.absorption_mode);

            quantizeCopyInputToFp8Kernel<T, threads_per_block, 128, 64, 128, false>
                <<<grid, threads_per_block, 0, stream>>>(params.q_buf, static_cast<__nv_fp8_e4m3*>(params.quant_q_buf),
                    params.k_buf, static_cast<__nv_fp8_e4m3*>(params.quant_k_buf), params.v_buf,
                    static_cast<__nv_fp8_e4m3*>(params.quant_v_buf), params.acc_q_len, total_kv_len,
                    params.quant_scale_qkv, params.bmm1_scale, params.bmm2_scale, params.quant_scale_o,
                    params.dequant_scale_q, params.dequant_scale_kv, params.host_bmm1_scale);
        }
    }
    else
    {
        TLLM_LOG_WARNING("MLA RoPE Context: acc_q_len is 0, skipping quantization.");
    }
}

template <typename T, typename KVCacheBuffer>
void invokeMLARopeGeneration(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream)
{
    dim3 grid(int(tensorrt_llm::common::divUp(params.acc_q_len, 32)), params.head_num + 1 + 8);
    if (params.cache_type == KvCacheDataType::FP8)
        grid.y += params.head_num * 8;
    TLLM_CHECK_WITH_INFO(params.acc_q_len % params.batch_size == 0,
        "MLA can only support input sequences with the same sequence length.");
    auto seq_len = params.acc_q_len / params.batch_size;

    auto* kernel_instance = &applyMLARopeAndAssignQKVKernelGeneration<T, 256, 512, 64, KVCacheBuffer>;
    if (!params.meta.rope_append)
    {
        kernel_instance = &applyMLARopeAndAssignQKVKernelGeneration<T, 256, 448, 64, KVCacheBuffer>;
    }
    cudaLaunchConfig_t config;
    config.gridDim = grid;
    config.blockDim = 256;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, kernel_instance, params.q_buf, params.q_pe, params.latent_cache, params.quant_q_buf,
        kv_cache_buffer, params.kv_scale_cache, params.cos_sin_cache, params.head_num, params.meta.kv_lora_rank,
        params.acc_q_len, seq_len, params.seqQOffset, params.fmha_tile_counter, params.cache_seq_lens,
        params.cu_kv_seqlens, params.q_pe_ld, params.q_pe_stride, params.cache_type, params.bmm1_scale,
        params.bmm2_scale, params.quant_scale_o, params.quant_scale_q, params.quant_scale_kv, params.dequant_scale_q,
        params.dequant_scale_kv, params.host_bmm1_scale, params.helix_position_offsets, params.helix_is_inactive_rank);
}

template <typename T, typename TCache>
void invokeMLALoadPagedKV(T* compressed_kv_ptr, T* k_pe_ptr, KVBlockArray& kv_cache, int const num_contexts,
    int64_t const* cu_ctx_cached_kv_lens, int const max_input_seq_len, int const lora_size, int const rope_size,
    float const* kv_scale_quant_orig_ptr, cudaStream_t stream, void const* kvarn_scale_pool_ptr, int const kvarn_bits)
{
    TLLM_CHECK_WITH_INFO(kvarn_bits == 2 || kvarn_bits == 4,
        "KVarN BDR paged MLA read supports bits=2 or bits=4, got %d.", kvarn_bits);
    using KT = typename tensorrt_llm::kernels::loadPagedKVKernelTraits<TCache>;
    // {seq_len / token_per_block, batch_size, head_num}
    TLLM_CHECK_WITH_INFO(lora_size == KT::kLoraSize, "lora_size should be equal to %d", KT::kLoraSize);
    TLLM_CHECK_WITH_INFO(rope_size == KT::kRopeSize, "rope_size should be equal to %d", KT::kRopeSize);
    TLLM_CHECK_WITH_INFO(lora_size + rope_size == KT::kHeadSize, "head dim should be equal to %d", KT::kHeadSize);
    dim3 grid(static_cast<int>(tensorrt_llm::common::divUp(max_input_seq_len, KT::kTokenPerBlock)), num_contexts, 1);
    // KVarN: non-null scale pool routes the ckv read through dequantCopyKVarN
    // (INT4 unpack + per-(token,sub-block) affine -> rotated-frame fp16).
    loadPagedKVCacheForMLAKernel<T, TCache><<<grid, KT::kBlockSize, 0, stream>>>(
        compressed_kv_ptr, k_pe_ptr, kv_cache, cu_ctx_cached_kv_lens, max_input_seq_len, kv_scale_quant_orig_ptr,
        reinterpret_cast<__half const*>(kvarn_scale_pool_ptr));
}

// =============================== KVarN write ===============================
// Fused block-diagonal-Hadamard (order 128) + per-(token,sub-block) INT4 RTN of
// the post-RoPE dense MLA latent ckv. Warp-cooperative: one warp-half (16 lanes)
// owns one 128-d sub-block; rotation + min/max scale are pure __shfl (no smem,
// no cross-warp). Launch right after the RoPE kernel = zero host round-trip and
// no fp16 staging pool. INT4 is validated by kvarn_inkernel; INT2 is the same
// packed low-bit path with qmax=3 for kvarn_k2v2 dense MLA storage.
//   ckv_in : [num_tokens, DCKV] post-RoPE fp16 latent (rotated frame written out)
//   data   : low-bit packed cache, NSUB*(DCKV/NSUB*BITS/8) bytes/token, by token
//   scale  : per-(token,sub-block) {scale[NSUB], zp[NSUB]} __half, stride 2*NSUB
template <typename T, int DCKV, int HORDER, int BITS>
__global__ void mlaBdrQuantizeLatentKernel(
    T const* __restrict__ ckv_in, uint8_t* __restrict__ data, __half* __restrict__ scale, int num_tokens)
{
    static_assert(BITS == 2 || BITS == 4, "KVarN BDR supports 2-bit or 4-bit packing");
    constexpr int kNSub = DCKV / HORDER;          // 4
    constexpr int kVecPerSub = HORDER / 8;        // 16 lanes (8 ch/lane)
    constexpr int kVecs = DCKV / 8;               // 64 lanes / token
    constexpr int kBytesPerTok = kNSub * (HORDER * BITS / 8);
    int const tok = blockIdx.x;
    if (tok >= num_tokens)
        return;
    int const lane = threadIdx.x;                 // 0..63
    int const sub = lane / kVecPerSub;            // 0..3
    int const laneInBlk = lane % kVecPerSub;      // 0..15
    unsigned const mask = 0xFFFFu << ((sub % 2) * 16); // 16-lane sub-block group
    float reg[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        reg[i] = static_cast<float>(ckv_in[static_cast<size_t>(tok) * DCKV + lane * 8 + i]);
    bdrFwhtSubblockWarp<8>(reg, laneInBlk, mask);
    float lo, hi;
    bdrSubblockMinMax<8>(reg, laneInBlk, mask, lo, hi);
    constexpr int kQMax = (1 << BITS) - 1;
    float const sc = fmaxf((hi - lo) / static_cast<float>(kQMax), 1e-10f);
    __half const hsc = __float2half(sc), hzp = __float2half(lo);
    uint8_t* tokData = data + static_cast<size_t>(tok) * kBytesPerTok;
    constexpr int kBytesPerVec = 8 * BITS / 8;
    bdrPackLowBitVec<8, BITS>(tokData + lane * kBytesPerVec, reg, __half2float(hsc), __half2float(hzp));
    if (laneInBlk == 0)
    {
        __half* tokScale = scale + static_cast<size_t>(tok) * (2 * kNSub);
        tokScale[sub] = hsc;
        tokScale[kNSub + sub] = hzp;
    }
}

template <typename T>
void invokeMLABdrQuantizeLatent(
    T const* ckv_in, uint8_t* data, void* scale_pool, int num_tokens, int dckv, int bits, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(dckv == 512, "KVarN BDR latent quantize currently supports DCKV=512.");
    TLLM_CHECK_WITH_INFO(bits == 2 || bits == 4, "KVarN BDR latent quantize supports bits=2 or bits=4, got %d.", bits);
    constexpr int kVecs = 512 / 8; // 64 lanes/token
    if (bits == 2)
    {
        mlaBdrQuantizeLatentKernel<T, 512, 128, 2><<<num_tokens, kVecs, 0, stream>>>(
            ckv_in, data, reinterpret_cast<__half*>(scale_pool), num_tokens);
    }
    else
    {
        mlaBdrQuantizeLatentKernel<T, 512, 128, 4><<<num_tokens, kVecs, 0, stream>>>(
            ckv_in, data, reinterpret_cast<__half*>(scale_pool), num_tokens);
    }
}
// ===========================================================================

template <typename T, typename TCache>
void invokeMLARopeAppendPagedKVAssignQ(KVBlockArray& kv_cache, KVBlockArray& kv_scale_cache, T* q_ptr,
    T* latent_cache_ptr, int const num_requests, int64_t const* cu_ctx_cached_kv_lens, int64_t const* cu_seq_lens,
    int const max_input_uncached_seq_len, float2 const* cos_sin_cache, size_t head_num, int nope_size, int rope_size,
    int lora_size, float const* kv_scale_orig_quant_ptr, KvCacheDataType cache_type, cudaStream_t stream)
{
    dim3 grid(int(tensorrt_llm::common::divUp(max_input_uncached_seq_len, 32)), num_requests, head_num + 1 + 8);
    TLLM_CHECK_WITH_INFO(lora_size == 512 || lora_size == 448, "lora_size should be equal to %d or %d", 512, 448);
    TLLM_CHECK_WITH_INFO(rope_size == 64, "rope_size should be equal to %d", 64);
    if (lora_size == 512)
    {
        applyMLARopeAppendPagedKVAssignQKernel<T, TCache, 256, 512, 64><<<grid, 256, 0, stream>>>(kv_cache,
            kv_scale_cache, q_ptr, latent_cache_ptr, cu_ctx_cached_kv_lens, cu_seq_lens, max_input_uncached_seq_len,
            cos_sin_cache, head_num, nope_size, kv_scale_orig_quant_ptr, cache_type);
    }
    else
    {
        applyMLARopeAppendPagedKVAssignQKernel<T, TCache, 256, 448, 64><<<grid, 256, 0, stream>>>(kv_cache,
            kv_scale_cache, q_ptr, latent_cache_ptr, cu_ctx_cached_kv_lens, cu_seq_lens, max_input_uncached_seq_len,
            cos_sin_cache, head_num, nope_size, kv_scale_orig_quant_ptr, cache_type);
    }
}

#define INSTANTIATE_MLA_ROPE(T, KVCacheBuffer)                                                                         \
    template void invokeMLARopeContext(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream);      \
    template void invokeMLARopeGeneration(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream);

INSTANTIATE_MLA_ROPE(float, KVBlockArray);
INSTANTIATE_MLA_ROPE(half, KVBlockArray);
INSTANTIATE_MLA_ROPE(float, KVLinearBuffer);
INSTANTIATE_MLA_ROPE(half, KVLinearBuffer);
INSTANTIATE_MLA_ROPE(__nv_bfloat16, KVBlockArray);
INSTANTIATE_MLA_ROPE(__nv_bfloat16, KVLinearBuffer);

#define INSTANTIATE_MLA_QUANTIZE(T)                                                                                    \
    template void invokeMLAContextFp8Quantize<T>(MlaParams<T> & params, int total_kv_len, cudaStream_t stream);

INSTANTIATE_MLA_QUANTIZE(float);
INSTANTIATE_MLA_QUANTIZE(half);
INSTANTIATE_MLA_QUANTIZE(__nv_bfloat16);

#define INSTANTIATE_RW_KVCACHE_MLA(T, TCache)                                                                          \
    template void invokeMLALoadPagedKV<T, TCache>(T * compressed_kv_ptr, T * k_pe_ptr, KVBlockArray & kv_cache,        \
        int const num_contexts, int64_t const* cu_ctx_cached_kv_lens, int const max_input_seq_len,                     \
        int const lora_size, int const rope_size, float const* kv_scale_quant_orig_ptr, cudaStream_t stream,           \
        void const* kvarn_scale_pool_ptr, int const kvarn_bits);                                                       \
    template void invokeMLARopeAppendPagedKVAssignQ<T, TCache>(KVBlockArray & kv_cache,                               \
        KVBlockArray & kv_scale_cache, T * q_ptr, T * latent_cache_ptr, int const num_requests,                       \
        int64_t const* cu_ctx_cached_kv_lens, int64_t const* cu_seq_lens, int const max_input_uncached_seq_len,        \
        float2 const* cos_sin_cache, size_t head_num, int nope_size, int rope_size, int lora_size,                     \
        float const* kv_scale_orig_quant_ptr, KvCacheDataType cache_type, cudaStream_t stream);

INSTANTIATE_RW_KVCACHE_MLA(float, float);
INSTANTIATE_RW_KVCACHE_MLA(float, __nv_fp8_e4m3);
INSTANTIATE_RW_KVCACHE_MLA(half, half);
INSTANTIATE_RW_KVCACHE_MLA(half, __nv_fp8_e4m3);
INSTANTIATE_RW_KVCACHE_MLA(__nv_bfloat16, __nv_bfloat16);
INSTANTIATE_RW_KVCACHE_MLA(__nv_bfloat16, __nv_fp8_e4m3);

#define INSTANTIATE_MLA_BDR_QUANTIZE(T)                                                                                \
    template void invokeMLABdrQuantizeLatent<T>(                                                                        \
        T const* ckv_in, uint8_t* data, void* scale_pool, int num_tokens, int dckv, int bits, cudaStream_t stream);
INSTANTIATE_MLA_BDR_QUANTIZE(float);
INSTANTIATE_MLA_BDR_QUANTIZE(half);
INSTANTIATE_MLA_BDR_QUANTIZE(__nv_bfloat16);

// In-place MLA RoPE: apply RoPE to the last rope_dim elements of each [nope_dim + rope_dim] head.
// Uses 16-byte vectorized load/store (VecType) and mmha::rotary_embedding_transform for the
// interleaved path. Each thread handles ELTS_PER_VEC elements (8 bf16 = 4 rotation pairs).
// Grid: (num_tokens, ceil(num_heads / HPB)), Block: (VECS_PER_ROPE, HPB)
// cos_sin_cache layout: [max_positions, 2, half_rope] float (cos block then sin block)
template <typename T, bool IS_INVERSE, bool IS_NEOX, int HEADS_PER_BLOCK>
__global__ void mlaRoPEInplaceKernel(T* __restrict__ data, int32_t const* __restrict__ position_ids,
    float const* __restrict__ cos_sin_cache, int num_heads, int nope_dim, int rope_dim)
{
    using VecT = typename VecType<T>::Type;
    using GPTJEltT = typename VecType<T>::GPTJEltType;
    constexpr int BYTES_PER_ELT = sizeof(T);
    constexpr int BYTES_PER_LOAD = 16;
    constexpr int ELTS_PER_VEC = BYTES_PER_LOAD / BYTES_PER_ELT;

    int const tid = threadIdx.x;
    int const half_rope = rope_dim / 2;
    // Neox: each thread handles one VecT from each half → half_rope elements per half
    // Interleaved: each thread handles one VecT of interleaved pairs → rope_dim elements
    int const vecs_per_rope
        = IS_NEOX ? (half_rope * BYTES_PER_ELT / BYTES_PER_LOAD) : (rope_dim * BYTES_PER_ELT / BYTES_PER_LOAD);
    int const head_idx = blockIdx.y * HEADS_PER_BLOCK + threadIdx.y;
    if (head_idx >= num_heads || tid >= vecs_per_rope)
        return;

    int const head_size = nope_dim + rope_dim;
    T* head_ptr = data + (static_cast<int64_t>(blockIdx.x) * num_heads + head_idx) * head_size;

    int const pos = position_ids[blockIdx.x];
    int const elem_offset = tid * ELTS_PER_VEC;
    // cos at [pos, 0, ...], sin at [pos, 1, ...]
    float const* cos_ptr = cos_sin_cache + pos * 2 * half_rope + elem_offset;
    float const* sin_ptr = cos_ptr + half_rope;

    if constexpr (IS_NEOX)
    {
        // Neox: first half = x1[0..half), second half = x2[0..half) — two separate 16-byte loads
        VecT v1 = *reinterpret_cast<VecT const*>(&head_ptr[nope_dim + elem_offset]);
        VecT v2 = *reinterpret_cast<VecT const*>(&head_ptr[nope_dim + half_rope + elem_offset]);

        // Each GPTJEltT holds 2 consecutive elements from the same half.
        // For neox, we rotate (v1[j], v2[j]) independently for each element j.
#pragma unroll
        for (int i = 0; i < ELTS_PER_VEC / 2; i++)
        {
            GPTJEltT& e1 = reinterpret_cast<GPTJEltT*>(&v1)[i];
            GPTJEltT& e2 = reinterpret_cast<GPTJEltT*>(&v2)[i];

            // Construct (x1, x2) pairs and rotate — 2 pairs per GPTJElt
            float2 coef0{cos_ptr[i * 2], IS_INVERSE ? -sin_ptr[i * 2] : sin_ptr[i * 2]};
            float2 coef1{cos_ptr[i * 2 + 1], IS_INVERSE ? -sin_ptr[i * 2 + 1] : sin_ptr[i * 2 + 1]};

            float2 p1 = mmha::rotary_embedding_transform(float2{static_cast<float>(reinterpret_cast<T*>(&e1)[0]),
                                                             static_cast<float>(reinterpret_cast<T*>(&e2)[0])},
                coef0);
            float2 p2 = mmha::rotary_embedding_transform(float2{static_cast<float>(reinterpret_cast<T*>(&e1)[1]),
                                                             static_cast<float>(reinterpret_cast<T*>(&e2)[1])},
                coef1);

            reinterpret_cast<T*>(&e1)[0] = static_cast<T>(p1.x);
            reinterpret_cast<T*>(&e1)[1] = static_cast<T>(p2.x);
            reinterpret_cast<T*>(&e2)[0] = static_cast<T>(p1.y);
            reinterpret_cast<T*>(&e2)[1] = static_cast<T>(p2.y);
        }

        *reinterpret_cast<VecT*>(&head_ptr[nope_dim + elem_offset]) = v1;
        *reinterpret_cast<VecT*>(&head_ptr[nope_dim + half_rope + elem_offset]) = v2;
    }
    else
    {
        // Interleaved: (x1, x2) adjacent pairs — matches GPTJ layout, single 16-byte load
        VecT v = *reinterpret_cast<VecT const*>(&head_ptr[nope_dim + elem_offset]);

        // For interleaved, cos_ptr/sin_ptr index by pair (half the element count)
        float const* cos_pair = cos_sin_cache + pos * 2 * half_rope + (elem_offset / 2);
        float const* sin_pair = cos_pair + half_rope;

#pragma unroll
        for (int i = 0; i < ELTS_PER_VEC / 2; i++)
        {
            GPTJEltT& elt = reinterpret_cast<GPTJEltT*>(&v)[i];
            float2 coef{cos_pair[i], IS_INVERSE ? -sin_pair[i] : sin_pair[i]};
            elt = mmha::rotary_embedding_transform(elt, coef);
        }

        *reinterpret_cast<VecT*>(&head_ptr[nope_dim + elem_offset]) = v;
    }
}

template <typename T>
void invokeMLARoPEInplace(T* data, int32_t const* position_ids, float const* cos_sin_cache, int num_tokens,
    int num_heads, int nope_dim, int rope_dim, bool inverse, bool is_neox, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(rope_dim % 4 == 0, "rope_dim must be divisible by 4");
    constexpr int BYTES_PER_LOAD = 16;
    int const elt_size = static_cast<int>(sizeof(T));

    auto launch = [&](auto inverse_tag, auto neox_tag)
    {
        constexpr bool INV = decltype(inverse_tag)::value;
        constexpr bool NEOX = decltype(neox_tag)::value;
        // Neox loads from two halves → threads = half_rope elements / ELTS_PER_VEC
        // Interleaved loads contiguous → threads = rope_dim elements / ELTS_PER_VEC
        int const active_elts = NEOX ? (rope_dim / 2) : rope_dim;
        int const vecs_per_rope = active_elts * elt_size / BYTES_PER_LOAD;

        constexpr int kMaxBlockSize = 256;
        constexpr int kMaxHeadsPerBlock = 16;
        int const hpb = std::max(1, std::min({kMaxBlockSize / vecs_per_rope, num_heads, kMaxHeadsPerBlock}));
        dim3 grid(num_tokens, (num_heads + hpb - 1) / hpb);

        if (hpb <= 4)
        {
            mlaRoPEInplaceKernel<T, INV, NEOX, 4><<<grid, dim3(vecs_per_rope, 4), 0, stream>>>(
                data, position_ids, cos_sin_cache, num_heads, nope_dim, rope_dim);
        }
        else if (hpb <= 8)
        {
            mlaRoPEInplaceKernel<T, INV, NEOX, 8><<<grid, dim3(vecs_per_rope, 8), 0, stream>>>(
                data, position_ids, cos_sin_cache, num_heads, nope_dim, rope_dim);
        }
        else
        {
            mlaRoPEInplaceKernel<T, INV, NEOX, 16><<<grid, dim3(vecs_per_rope, 16), 0, stream>>>(
                data, position_ids, cos_sin_cache, num_heads, nope_dim, rope_dim);
        }
    };

    if (inverse && is_neox)
        launch(std::true_type{}, std::true_type{});
    else if (inverse && !is_neox)
        launch(std::true_type{}, std::false_type{});
    else if (!inverse && is_neox)
        launch(std::false_type{}, std::true_type{});
    else
        launch(std::false_type{}, std::false_type{});
}

#define INSTANTIATE_MLA_ROPE_INPLACE(T)                                                                                \
    template void invokeMLARoPEInplace<T>(T * data, int32_t const* position_ids, float const* cos_sin_cache,           \
        int num_tokens, int num_heads, int nope_dim, int rope_dim, bool inverse, bool is_neox, cudaStream_t stream);

INSTANTIATE_MLA_ROPE_INPLACE(__nv_bfloat16);
INSTANTIATE_MLA_ROPE_INPLACE(half);

} // namespace kernels

TRTLLM_NAMESPACE_END

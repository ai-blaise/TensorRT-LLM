/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

// HiSparse hot-read sparse-MLA decode (opt5): tcgen05/UMMA + TMEM rewrite.
//
// The two GEMMs (score Q.K^T and value P.V) run on 5th-gen tensor cores
// (SM100 tcgen05/UMMA, WS 1-CTA bf16->f32 atoms) with BOTH accumulators in TMEM:
//   - O[64,512] accumulator at TMEM cols [0,256)  (two N=256 tiles, 128 phys cols each)
//   - S[64,N_tile] accumulator at TMEM cols [256, 256+N_tile/2)
// Moving the 64KB acc[heads][512] fp32 V-accumulator out of SMEM into TMEM frees the
// dual-lock that pinned M19 to 2 blocks/SM, and UMMA does the GEMMs at TC throughput.
//
// The KVarN-BDR 2-bit dequant (FWHT + 2-bit unpack + fp8 PE) is reused verbatim from
// the scalar M19 kernel; only its write target changes to fill the canonical SW128
// bf16 SMEM operand tiles (sK[N_tile,576] for score, sV[N_tile,512] for value).
//
// Block = 64 heads of one row (grid (rows, 2 headGroups, splits)), 128 threads.
// Online (flash-decoding) softmax between the GEMMs reads S from TMEM, does
// row-max/exp/sum (the 64 columns of a head split across thread t and t^64), writes
// the bf16 weights P to SMEM, rescales the O TMEM accumulator on max growth, and feeds
// P back as the value-GEMM A operand. Verified vs the TRUE dense BDR-dequant reference.

#include "tensorrt_llm/kernels/flashMLA/sparse_mla_decode_kvarn_hot.h"

#include "tensorrt_llm/kernels/hisparseKvarnBdrRead.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <limits>
#include <stdexcept>

// The cute/cutlass/kerutils TYPES (SMEM layouts, TMEM/UMMA wrappers, barriers) must be
// visible in BOTH the host and device compilation passes -- the host pass emits the
// kernel-registration stub that references them. Only the inline-asm tcgen05 INSTRUCTIONS
// inside the kernel body are device-only (and the kerutils headers self-guard those).
// Include unconditionally; gate only the kernel body on __CUDA_ARCH__.
#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>

#include "kerutils/common/common.h"
#include "kerutils/device/common.h"
#include "kerutils/device/sm100/gemm.cuh"
#include "kerutils/device/sm100/helpers.cuh"
#include "kerutils/device/sm100/intrinsics.cuh"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
#define HISPARSE_UMMA_ENABLED 1
#endif

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{
constexpr int32_t kHeadQ = 128;
constexpr int32_t kDqk = 576;
constexpr int32_t kDv = 512;
constexpr int32_t kTokensPerBlock = 64;
constexpr int32_t kKvLoraRank = 512;
constexpr int32_t kQkRopeHeadDim = 64;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr int32_t kResidentKvPoolBf16 = 0;
constexpr int32_t kResidentKvPoolFp16 = 1;

// UMMA flash-decode tiling: 64 heads/block (M=64 = the WS-M64 UMMA M-mode), 128 threads
// (4 warps), token tile N_tile=64 (the score N-mode and the value K-mode). The SMEM
// operand tiles + Q at N_tile=64 sum to ~221KB (< 228KB B200 opt-in SMEM); larger tiles
// overflow. The dequant is warp-per-token (4 warps -> tokens tt = warpId, +4, ...).
constexpr int32_t kHeadsPerBlock = 64;
constexpr int32_t kHeadGroups = kHeadQ / kHeadsPerBlock; // 2
constexpr int32_t kTileTokens = 64;
constexpr int32_t kThreads = 128;
constexpr int32_t kMaxSplits = 16;
constexpr int32_t kValueNTile = 256; // value-GEMM N per atom (kDv/kValueNTile = 2 atoms)

// TMEM column layout (512 cols total). A WS-M64 [64,N] accumulator occupies N/2 phys cols.
constexpr int32_t kTmemO0 = 0;                    // O tile0 [64,256] -> cols [0,128)
constexpr int32_t kTmemO1 = 128;                  // O tile1 [64,256] -> cols [128,256)
constexpr int32_t kTmemS = 256;                   // S [64,N_tile] -> cols [256, 256+N_tile/2)

__device__ __forceinline__ void writeBf16(void* ptr, int64_t offset, float value)
{
    auto* out = reinterpret_cast<__nv_bfloat16*>(ptr);
    out[offset] = __float2bfloat16_rn(value);
}

// Warp-shuffle Fast Walsh-Hadamard Transform over a 16-lane group (one 128-channel
// sub-block) where lane l holds the ELTS contiguous channels [l*ELTS, l*ELTS+ELTS).
// Identical primitive to the M19 scalar kernel; bit-exact vs the natural Hadamard.
template <int ELTS>
__device__ __forceinline__ void fwhtSubblockWarp(float (&reg)[ELTS], int laneInBlk, unsigned mask)
{
    constexpr float kInvSqrtHadamard128 = 0.088388347648318f;
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

__device__ __forceinline__ float readResidentLatentValue(
    SparseMlaDecodeKvarnHotParams const& params, int64_t globalToken, int32_t dim)
{
    int64_t const offset = globalToken * params.strideResidentKvPoolToken + dim;
    if (params.residentKvPoolDtype == kResidentKvPoolBf16)
    {
        auto const* pool = reinterpret_cast<__nv_bfloat16 const*>(params.residentKvPool);
        return __bfloat162float(pool[offset]);
    }
    if (params.residentKvPoolDtype == kResidentKvPoolFp16)
    {
        auto const* pool = reinterpret_cast<__half const*>(params.residentKvPool);
        return __half2float(pool[offset]);
    }
    return 0.0F;
}

struct HiSparseResidentTokenAddress
{
    uint8_t status;
    int64_t globalToken;
};

__device__ __forceinline__ uint8_t readRequestTopkToken(
    SparseMlaDecodeKvarnHotParams const& params, int32_t batch, int32_t s, int32_t k, int32_t& token)
{
    if (params.requestTopkIndices == nullptr)
    {
        return kHotReadInvalidIndex;
    }
    int64_t const requestIndexBase = static_cast<int64_t>(batch) * params.strideRequestTopkB
        + static_cast<int64_t>(s) * params.strideRequestTopkSQ;
    token = params.requestTopkIndices[requestIndexBase + k];
    return kHotReadOk;
}

__device__ __forceinline__ HiSparseResidentTokenAddress decodeResidentTokenAddress(
    SparseMlaDecodeKvarnHotParams const& params, int32_t row, int32_t requestToken)
{
    HiSparseResidentTokenAddress address{kHotReadOk, -1};
    if (params.residentKvLens == nullptr || params.residentReqIdx == nullptr || params.residentRequestIds == nullptr
        || params.residentKvPool == nullptr || params.residentBlockTable == nullptr
        || params.residentTailBlockPos == nullptr || params.residentTailTokenCount == nullptr
        || params.residentTailValid == nullptr)
    {
        address.status = kHotReadInvalidIndex;
        return address;
    }
    if (params.residentKvPoolDtype != kResidentKvPoolBf16 && params.residentKvPoolDtype != kResidentKvPoolFp16)
    {
        address.status = kHotReadInvalidIndex;
        return address;
    }
    if (row < 0 || row >= params.residentRows)
    {
        address.status = kHotReadInvalidIndex;
        return address;
    }
    if (params.residentRequestIds[row] < 0)
    {
        address.status = kHotReadInvalidIndex;
        return address;
    }

    int64_t const kvLen = params.residentKvLens[row];
    if (requestToken < 0 || static_cast<int64_t>(requestToken) >= kvLen)
    {
        address.status = kHotReadInvalidIndex;
        return address;
    }

    int32_t const blockPos = requestToken / params.tokensPerBlock;
    int32_t const tokenOffset = requestToken % params.tokensPerBlock;
    bool const isSink = blockPos < params.residentSinkBlocks;
    bool const isTail = params.residentTailValid[row] && blockPos == params.residentTailBlockPos[row]
        && tokenOffset < params.residentTailTokenCount[row];
    if (!isSink && !isTail)
    {
        address.status = kHotReadInvalidIndex;
        return address;
    }

    int64_t const reqIdx = params.residentReqIdx[row];
    if (reqIdx < 0 || reqIdx >= params.residentBlockTableRows || blockPos < 0
        || blockPos >= params.residentBlockTableBlocks)
    {
        address.status = kHotReadInvalidIndex;
        return address;
    }
    int64_t const tableOffset = reqIdx * params.strideResidentBlockTableB
        + static_cast<int64_t>(blockPos) * params.strideResidentBlockTableBlock;
    int32_t const blockId = params.residentBlockTable[tableOffset];
    if (blockId < 0)
    {
        address.status = kHotReadInvalidIndex;
        return address;
    }

    int64_t const globalToken = static_cast<int64_t>(blockId) * params.strideFactor
        + static_cast<int64_t>(params.layerIdx) * params.tokensPerBlock + tokenOffset;
    if (globalToken < 0 || globalToken >= params.residentKvPoolTokens)
    {
        address.status = kHotReadInvalidIndex;
        return address;
    }
    address.globalToken = globalToken;
    return address;
}

struct HiSparseSelectedToken
{
    uint8_t status;
    bool active;
    bool isHot;
    uint8_t const* record;
    int32_t tokenOffset;
    int64_t residentGlobalToken;
};

__device__ __forceinline__ HiSparseSelectedToken resolveSelectedToken(
    SparseMlaDecodeKvarnHotParams const& params, int32_t row, int32_t batch, int32_t s, int32_t k, int64_t indexBase)
{
    HiSparseSelectedToken r{kHotReadOk, false, false, nullptr, 0, -1};
    int32_t const hotIndex = params.indices[indexBase + k];
    if (hotIndex < 0)
    {
        int32_t requestToken = -1;
        uint8_t const tokenStatus = readRequestTopkToken(params, batch, s, k, requestToken);
        if (tokenStatus != kHotReadOk)
        {
            r.status = tokenStatus;
            return r;
        }
        if (requestToken < 0)
        {
            r.active = false;
            return r;
        }
        r.active = true;
        HiSparseResidentTokenAddress const addr = decodeResidentTokenAddress(params, row, requestToken);
        if (addr.status != kHotReadOk)
        {
            r.status = addr.status;
            return r;
        }
        r.isHot = false;
        r.residentGlobalToken = addr.globalToken;
        return r;
    }
    r.active = true;
    HiSparseKvarnHotAddress const addr = decodeHisparseKvarnHotIndex(
        hotIndex, params.strideFactor, params.layerIdx, params.hotCapacity, params.tokensPerBlock);
    if (addr.status != kHotReadOk)
    {
        r.status = addr.status;
        return r;
    }
    r.isHot = true;
    r.record = params.hotPacked + static_cast<int64_t>(params.layerIdx) * params.strideHotLayer
        + static_cast<int64_t>(addr.hotSlot) * params.strideHotSlot;
    r.tokenOffset = addr.tokenOffset;
    return r;
}

using kerutils::bf16;
namespace cg = cute;

// Canonical SW128 K-major SMEM operand layouts.
using SmemLayoutQ = decltype(ku::make_umma_canonical_k_major_layout<kHeadsPerBlock, kDqk, 128, bf16>());
using SmemLayoutK = decltype(ku::make_umma_canonical_k_major_layout<kTileTokens, kDqk, 128, bf16>());
using SmemLayoutP = decltype(ku::make_umma_canonical_k_major_layout<kHeadsPerBlock, kTileTokens, 128, bf16>());
// V (value-GEMM B operand, MN-major [dim, token]) is derived from the FIRST 512 dims of
// the SAME 576-wide SW128 K store via transposed composition -- NO separate sV tile
// (saves kTileTokens*512 bf16 of SMEM; verified cos=1.0 vs a standalone V store).
using SmemLayoutVb = decltype(cg::composition(
    SmemLayoutK{}, cg::make_layout(cg::Shape<cg::Int<kDv>, cg::Int<kTileTokens>>{}, cg::Stride<cg::Int<kTileTokens>, cg::_1>{})));

using ScoreMMA = decltype(cg::make_tiled_mma(
    cg::SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, kHeadsPerBlock, kTileTokens, cg::UMMA::Major::K, cg::UMMA::Major::K>{}));
using ValueMMA = decltype(cg::make_tiled_mma(
    cg::SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, kHeadsPerBlock, kValueNTile, cg::UMMA::Major::K, cg::UMMA::Major::MN>{}));

__global__ __launch_bounds__(kThreads) void sparseMlaDecodeKvarnHotKernel(SparseMlaDecodeKvarnHotParams params)
{
#if !defined(HISPARSE_UMMA_ENABLED)
    return;
#else
    int32_t const row = static_cast<int32_t>(blockIdx.x);
    int32_t const headGroup = static_cast<int32_t>(blockIdx.y);
    int32_t const splitIdx = static_cast<int32_t>(blockIdx.z);
    int32_t const totalRows = params.b * params.sQ;
    int32_t const numSplits = params.numSplits;
    if (row >= totalRows || headGroup >= kHeadGroups || splitIdx >= numSplits)
    {
        return;
    }
    int32_t const headBase = headGroup * kHeadsPerBlock;
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    int32_t const warp = tid >> 5;

    int32_t const batch = row / params.sQ;
    int32_t const s = row - batch * params.sQ;
    int32_t const rowTopK = params.topkLength == nullptr ? params.topK
        : (params.topkLengthSize == params.b ? params.topkLength[batch] : params.topkLength[row]);
    HiSparseKvarnK2v2BdrLayout const layout = makeHisparseKvarnK2v2BdrLayout(
        params.tokensPerBlock, params.kvLoraRank, params.qkRopeHeadDim);

    int64_t const indexBase = static_cast<int64_t>(batch) * params.strideIndicesB
        + static_cast<int64_t>(s) * params.strideIndicesSQ;
    int64_t const qRowBase = static_cast<int64_t>(batch) * params.strideQB + static_cast<int64_t>(s) * params.strideQSQ;
    int64_t const outRowBase
        = static_cast<int64_t>(batch) * params.strideOB + static_cast<int64_t>(s) * params.strideOSQ;
    int64_t const lseRowBase
        = static_cast<int64_t>(batch) * params.strideLseB + static_cast<int64_t>(s) * params.strideLseSQ;

    bool const splitMode = numSplits > 1;

    int32_t kStart = 0;
    int32_t kEnd = rowTopK;
    if (splitMode)
    {
        int32_t const tilesTotal = (rowTopK + kTileTokens - 1) / kTileTokens;
        int32_t const tilesPerSplit = (tilesTotal + numSplits - 1) / numSplits;
        kStart = splitIdx * tilesPerSplit * kTileTokens;
        kEnd = min(rowTopK, kStart + tilesPerSplit * kTileTokens);
        if (kStart >= kEnd)
        {
            kStart = 0;
            kEnd = 0;
        }
    }

    // SMEM plan: SW128 operand tiles + per-head flash scalars + scratch + barriers.
    extern __shared__ char smemRaw[];
    struct SmemPlan
    {
        cg::array_aligned<bf16, cg::cosize_v<SmemLayoutQ>> q;
        cg::array_aligned<bf16, cg::cosize_v<SmemLayoutK>> k;
        cg::array_aligned<bf16, cg::cosize_v<SmemLayoutP>> p;
        float runMax[kHeadsPerBlock];
        float runDenom[kHeadsPerBlock];
        float rowExch[kThreads]; // peer-half exchange for max/denom (thread t <-> t^64)
        uint8_t tileActive[kTileTokens];
        cg::array_aligned<uint32_t, 1> tmemBase;
        cutlass::arch::ClusterTransactionBarrier barScore;
        cutlass::arch::ClusterTransactionBarrier barValue;
        int32_t rowCode;
        int32_t tileStatusAgg;
    };
    SmemPlan& sm = *reinterpret_cast<SmemPlan*>(smemRaw);

    cg::Tensor sQ = cg::make_tensor(cg::make_smem_ptr(sm.q.data()), SmemLayoutQ{});
    cg::Tensor sK = cg::make_tensor(cg::make_smem_ptr(sm.k.data()), SmemLayoutK{});
    cg::Tensor sP = cg::make_tensor(cg::make_smem_ptr(sm.p.data()), SmemLayoutP{});
    // V operand is a transposed view of the first 512 dims of sK (no separate sV tile).
    cg::Tensor sVb = cg::make_tensor(cg::make_smem_ptr(sm.k.data()), SmemLayoutVb{});

    if (tid == 0)
    {
        sm.rowCode = params.rowStatus[row] == 0 ? kHotReadOk : kHotReadUpstreamInvalid;
        if (sm.rowCode == kHotReadOk && (rowTopK < 0 || rowTopK > params.topK))
        {
            sm.rowCode = kHotReadBadTopKLength;
        }
        sm.tileStatusAgg = kHotReadOk;
    }
    for (int32_t h = tid; h < kHeadsPerBlock; h += kThreads)
    {
        sm.runMax[h] = kNegInf;
        sm.runDenom[h] = 0.0F;
    }

    // TMEM allocation (one elected warp) + barrier init.
    if (warp == 0)
    {
        if (cg::elect_one_sync())
        {
            sm.barScore.init(1);
            sm.barValue.init(1);
            cutlass::arch::fence_barrier_init();
        }
        cg::TMEM::Allocator1Sm().allocate(512, sm.tmemBase.data());
        cg::TMEM::Allocator1Sm().release_allocation_lock();
    }
    __syncthreads();
    uint32_t const tmemBase = sm.tmemBase.data()[0];
    int32_t const rowCode = sm.rowCode;
#ifdef HISPARSE_DBG
    bool const dbg = (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0);
    if (dbg) printf("[K] alloc done tmemBase=%u rowCode=%d kStart=%d kEnd=%d numSplits=%d\n", tmemBase, rowCode, kStart, kEnd, numSplits);
#endif

    // TMEM accumulator fragments.
    ScoreMMA scoreMma;
    ValueMMA valueMma;
    cg::Tensor tS = cg::partition_fragment_C(scoreMma, cg::Shape<cg::Int<kHeadsPerBlock>, cg::Int<kTileTokens>>{});
    tS.data().get() = tmemBase + kTmemS;
    cg::Tensor tO0 = cg::partition_fragment_C(valueMma, cg::Shape<cg::Int<kHeadsPerBlock>, cg::Int<kValueNTile>>{});
    tO0.data().get() = tmemBase + kTmemO0;
    cg::Tensor tO1 = cg::partition_fragment_C(valueMma, cg::Shape<cg::Int<kHeadsPerBlock>, cg::Int<kValueNTile>>{});
    tO1.data().get() = tmemBase + kTmemO1;

    // Per-thread softmax ownership: thread t owns head h = tid % 64 and the column half
    // colHalf = tid / 64 (0 -> S cols [0,32), 1 -> [32,64)) of that head's score row.
    int32_t const myHead = tid % kHeadsPerBlock;       // 0..63
    int32_t const colHalf = tid / kHeadsPerBlock;       // 0 or 1
    constexpr int32_t kHalfCols = kTileTokens / 2;       // 32 (== N_tile/2)
    int32_t const peer = tid ^ kHeadsPerBlock;           // thread holding the other half of myHead

    if (rowCode == kHotReadOk)
    {
        // Load Q[64,576] into sQ (SW128) once. Each thread streams a strided slice.
        int64_t const qHeadBase = qRowBase + static_cast<int64_t>(headBase) * params.strideQHQ;
        __nv_bfloat16 const* qbase = reinterpret_cast<__nv_bfloat16 const*>(params.q);
        for (int32_t i = tid; i < kHeadsPerBlock * kDqk; i += kThreads)
        {
            int32_t const hh = i / kDqk;
            int32_t const dd = i - hh * kDqk;
            sQ(hh, dd) = bf16(__bfloat162float(qbase[qHeadBase + static_cast<int64_t>(hh) * params.strideQHQ + dd]));
        }
        __syncthreads();

        bool firstTile = true;
        bool scorePhase = false; // ClusterTransactionBarrier wait-phase parity (flips per reuse)
        bool valuePhase = false;
        for (int32_t tileStart = kStart; tileStart < kEnd; tileStart += kTileTokens)
        {
            int32_t const tileLen = min(kTileTokens, kEnd - tileStart);

            // --- (1) dequant N_tile tokens -> sK (576, SW128) + sV (512, SW128) ---
            // Warp-per-token (4 warps): tokens tt = warpId, +4, ... Reused FWHT/unpack/PE
            // from M19; only the store target changed to the cute SW128 operand tiles.
            {
                int32_t const lane = tid & 31;
                int32_t const warpId = warp;
                int32_t const laneInBlk = lane & 7;
                int32_t const subblock = lane >> 3;
                int32_t const subBase = subblock * 128;
                unsigned const subMask = 0xFFu << (subblock * 8);
                for (int32_t tt = warpId; tt < kTileTokens; tt += (kThreads / 32))
                {
                    bool const inRange = tt < tileLen;
                    int32_t const k = tileStart + tt;
                    HiSparseSelectedToken st{kHotReadInvalidIndex, false, false, nullptr, 0, -1};
                    if (inRange)
                    {
                        st = resolveSelectedToken(params, row, batch, s, k, indexBase);
                        if (st.status != kHotReadOk && lane == 0)
                        {
                            atomicCAS(&sm.tileStatusAgg, kHotReadOk, static_cast<int32_t>(st.status));
                        }
                        if (lane == 0)
                        {
                            sm.tileActive[tt] = (st.status == kHotReadOk && st.active) ? 1 : 0;
                        }
                    }
                    bool const buildHot = inRange && (st.status == kHotReadOk) && st.active && st.isHot;
                    if (buildHot)
                    {
                        uint8_t const* tokenPacked
                            = st.record + static_cast<int64_t>(st.tokenOffset) * layout.ckvBytesPerToken;
                        uint8_t const* tokenScaleZpBytes = st.record + layout.ckvBytesPerBlock
                            + static_cast<int64_t>(st.tokenOffset) * layout.scaleZpBytesPerToken;
                        uint8_t const* peBytes = st.record + layout.ckvBytesPerBlock + layout.scaleZpBytesPerBlock;
                        {
                            float const scale = __half2float(readHisparseHalfUnaligned(
                                tokenScaleZpBytes + static_cast<int64_t>(subblock) * sizeof(__half)));
                            float const zp = __half2float(readHisparseHalfUnaligned(
                                tokenScaleZpBytes + static_cast<int64_t>(4 + subblock) * sizeof(__half)));
                            float reg[16];
                            int32_t const dim0 = subBase + laneInBlk * 16;
                            uint8_t const* pk = tokenPacked + (dim0 >> 2);
#pragma unroll
                            for (int32_t b = 0; b < 4; ++b)
                            {
                                uint32_t const byte = pk[b];
#pragma unroll
                                for (int32_t j = 0; j < 4; ++j)
                                {
                                    int32_t const q = (byte >> (j * 2)) & 0x3;
                                    reg[b * 4 + j] = static_cast<float>(q) * scale + zp;
                                }
                            }
                            fwhtSubblockWarp<16>(reg, laneInBlk, subMask);
#pragma unroll
                            for (int32_t i = 0; i < 16; ++i)
                            {
                                sK(tt, dim0 + i) = bf16(reg[i]); // C-KV (dim<512); V reads it transposed
                            }
                        }
#pragma unroll
                        for (int32_t r = 0; r < (kQkRopeHeadDim + 31) / 32; ++r)
                        {
                            int32_t const peDim = lane + r * 32;
                            if (peDim < layout.qkRopeHeadDim)
                            {
                                uint8_t const byte
                                    = peBytes[static_cast<int64_t>(st.tokenOffset) * layout.qkRopeHeadDim + peDim];
                                sK(tt, layout.kvLoraRank + peDim) = bf16(readHisparseFp8E4m3Byte(byte));
                            }
                        }
                    }
                    else if (inRange && st.status == kHotReadOk && st.active && !st.isHot)
                    {
                        for (int32_t d = lane; d < kDqk; d += 32)
                            sK(tt, d) = bf16(readResidentLatentValue(params, st.residentGlobalToken, d));
                    }
                    else
                    {
                        // masked or out-of-range token: zero K (score uses it, masked later by
                        // tileActive; V reads the same zeros transposed -> 0 contribution).
                        for (int32_t d = lane; d < kDqk; d += 32)
                            sK(tt, d) = bf16(0.0F);
                    }
                }
            }
            __syncthreads();

            // --- (2) SCORE UMMA: S[64,N_tile] = Q . K^T ---
            if (warp == 0 && cg::elect_one_sync())
            {
                ku::tcgen05_after_thread_sync();
                ku::utcmma_ss(scoreMma, sQ, sK, tS, /*clear_accum=*/true);
                ku::umma_arrive_noelect(sm.barScore);
            }
            sm.barScore.wait(scorePhase);
            scorePhase = !scorePhase;
            ku::tcgen05_after_thread_sync();

            // --- (3) softmax: read S, row-max/exp/sum over the head's N_tile cols ---
            float sc[kHalfCols];
            ku::tmem_ld_32dp32bNx<kHalfCols>(tmemBase + kTmemS, sc);
            cutlass::arch::fence_view_async_tmem_load();
            // mask inactive tokens, scale by smScale, partial max over my 32 cols.
            float partMax = kNegInf;
#pragma unroll
            for (int32_t j = 0; j < kHalfCols; ++j)
            {
                int32_t const tok = colHalf * kHalfCols + j;
                float v = (tok < tileLen && sm.tileActive[tok]) ? (sc[j] * params.smScale) : kNegInf;
                sc[j] = v;
                partMax = fmaxf(partMax, v);
            }
            // exchange partial max with peer-half -> full tile max for this head.
            sm.rowExch[tid] = partMax;
            __syncthreads();
            float const tileMax = fmaxf(partMax, sm.rowExch[peer]);
            float const prevMax = sm.runMax[myHead];
            float const prevDenom = sm.runDenom[myHead];
            float const newMax = fmaxf(prevMax, tileMax);
            float const correction = (prevMax == kNegInf) ? 0.0F : __expf(prevMax - newMax);
            // weights for my 32 cols -> sP; partial denom.
            float partDenom = 0.0F;
#pragma unroll
            for (int32_t j = 0; j < kHalfCols; ++j)
            {
                int32_t const tok = colHalf * kHalfCols + j;
                float const w = (sc[j] == kNegInf) ? 0.0F : __expf(sc[j] - newMax);
                partDenom += w;
                sP(myHead, tok) = bf16(w);
            }
            sm.rowExch[tid] = partDenom;
            __syncthreads();
            float const tileDenom = partDenom + sm.rowExch[peer];

            // --- rescale O accumulator in TMEM by the per-row correction (max growth) ---
            // Every thread rescales ITS OWN 128 cols of each O tile (WS-M64 readout map:
            // thread t -> row t%64, cols (t/64)*128 + [0,128) of each tile). The branch is
            // gated ONLY on !firstTile (UNIFORM across the block) -- never on the per-thread
            // `correction`, because tmem_ld/tmem_st are warp-collective and would hang on
            // partial-warp participation when correction differs per head within a warp.
            // Multiplying by correction==1.0F (no growth for that row) is a harmless no-op.
            if (!firstTile)
            {
                constexpr int32_t kOHalf = kValueNTile / 2; // 128
                float o0[kOHalf], o1[kOHalf];
                ku::tmem_ld_32dp32bNx<kOHalf>(tmemBase + kTmemO0, o0);
                ku::tmem_ld_32dp32bNx<kOHalf>(tmemBase + kTmemO1, o1);
                cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
                for (int32_t j = 0; j < kOHalf; ++j)
                {
                    o0[j] *= correction;
                    o1[j] *= correction;
                }
                ku::tcgen05_before_thread_sync();
                ku::tmem_st_32dp32bNx<kOHalf>(tmemBase + kTmemO0, o0);
                ku::tmem_st_32dp32bNx<kOHalf>(tmemBase + kTmemO1, o1);
                cutlass::arch::fence_view_async_tmem_store();
            }
            if (colHalf == 0)
            {
                sm.runMax[myHead] = newMax;
                sm.runDenom[myHead] = prevDenom * correction + tileDenom;
            }
            __syncthreads();

            // --- (4) VALUE UMMA: O += P . V  (accumulate; clear on first tile) ---
            if (warp == 0 && cg::elect_one_sync())
            {
                ku::tcgen05_after_thread_sync();
                cg::Tensor sVbLo = cg::local_tile(sVb, cg::Shape<cg::Int<kValueNTile>, cg::Int<kTileTokens>>{}, cg::make_coord(cg::_0{}, cg::_0{}));
                cg::Tensor sVbHi = cg::local_tile(sVb, cg::Shape<cg::Int<kValueNTile>, cg::Int<kTileTokens>>{}, cg::make_coord(cg::_1{}, cg::_0{}));
                ku::utcmma_ss(valueMma, sP, sVbLo, tO0, firstTile);
                ku::utcmma_ss(valueMma, sP, sVbHi, tO1, firstTile);
                ku::umma_arrive_noelect(sm.barValue);
            }
            sm.barValue.wait(valuePhase);
            valuePhase = !valuePhase;
            ku::tcgen05_after_thread_sync();
            __syncthreads();
            firstTile = false;
        }
    }

    if (tid == 0 && rowCode == kHotReadOk && sm.tileStatusAgg != kHotReadOk)
    {
        sm.rowCode = sm.tileStatusAgg;
    }
    __syncthreads();
    int32_t const finalRowCode = sm.rowCode;

    // Epilogue: read O[64,512] from TMEM, scale, write out + lse.
    // Each thread: row = tid%64, half = tid/64 (which 256 cols of the 512 output).
    int32_t const eHead = tid % kHeadsPerBlock;
    int32_t const eHalf = tid / kHeadsPerBlock; // 0 -> out cols via O0/O1 col [0,128); 1 -> [128,256)
    int32_t const headF = headBase + eHead;
    int64_t const outBase = outRowBase + static_cast<int64_t>(headF) * params.strideOHQ;

    if (splitMode)
    {
        int64_t const partBase = ((static_cast<int64_t>(row) * params.hQ + headF) * numSplits + splitIdx);
        bool const failed = (finalRowCode != kHotReadOk);
        float const m = failed ? kNegInf : sm.runMax[eHead];
        float const d = failed ? 0.0F : sm.runDenom[eHead];
        float* pacc = params.partialAcc + partBase * kDv;
        constexpr int32_t kOHalf = kValueNTile / 2; // 128
        float o0[kOHalf], o1[kOHalf];
        if (!failed)
        {
            ku::tmem_ld_32dp32bNx<kOHalf>(tmemBase + kTmemO0, o0);
            ku::tmem_ld_32dp32bNx<kOHalf>(tmemBase + kTmemO1, o1);
            cutlass::arch::fence_view_async_tmem_load();
        }
#pragma unroll
        for (int32_t j = 0; j < kOHalf; ++j)
        {
            int32_t const c0 = eHalf * kOHalf + j;        // 0..255  (tile0 -> O cols 0..255)
            int32_t const c1 = 256 + eHalf * kOHalf + j;  // 256..511 (tile1)
            pacc[c0] = failed ? 0.0F : o0[j];
            pacc[c1] = failed ? 0.0F : o1[j];
        }
        if (tid < kHeadsPerBlock)
        {
            params.partialMax[partBase] = m;
            params.partialDenom[partBase] = d;
        }
        __syncthreads();
        if (warp == 0)
            cg::TMEM::Allocator1Sm().free(0, 512);
        return;
    }

    if (finalRowCode != kHotReadOk)
    {
        // failed-row path: no TMEM was read (mainloop skipped), but warp 0 still frees the
        // allocation; sync so all warps reach here together.
        for (int32_t d = tid; d < kHeadsPerBlock * kDv; d += kThreads)
        {
            int32_t const hh = d / kDv;
            int32_t const dd = d - hh * kDv;
            writeBf16(params.out, outRowBase + static_cast<int64_t>(headBase + hh) * params.strideOHQ + dd, 0.0F);
        }
        for (int32_t h = tid; h < kHeadsPerBlock; h += kThreads)
            params.lse[lseRowBase + headBase + h] = kNegInf;
        __syncthreads();
        if (warp == 0)
            cg::TMEM::Allocator1Sm().free(0, 512);
        return;
    }

    float const sinkVal = params.attnSink == nullptr ? kNegInf : params.attnSink[headF];
    float const mF = sm.runMax[eHead];
    float finalMax = (sinkVal != kNegInf) ? fmaxf(mF, sinkVal) : mF;
    float denom = sm.runDenom[eHead];
    if (finalMax != mF)
    {
        float const corr = (mF == kNegInf) ? 0.0F : __expf(mF - finalMax);
        denom = denom * corr;
    }
    if (sinkVal != kNegInf)
        denom += __expf(sinkVal - finalMax);
    float const accScale = (finalMax == mF) ? 1.0F : ((mF == kNegInf) ? 0.0F : __expf(mF - finalMax));
    float const invDenom = denom > 0.0F ? 1.0F / denom : 0.0F;
    float const oScale = accScale * invDenom;

    constexpr int32_t kOHalf = kValueNTile / 2; // 128
    float o0[kOHalf], o1[kOHalf];
    ku::tmem_ld_32dp32bNx<kOHalf>(tmemBase + kTmemO0, o0);
    ku::tmem_ld_32dp32bNx<kOHalf>(tmemBase + kTmemO1, o1);
    cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
    for (int32_t j = 0; j < kOHalf; ++j)
    {
        int32_t const c0 = eHalf * kOHalf + j;
        int32_t const c1 = 256 + eHalf * kOHalf + j;
        writeBf16(params.out, outBase + c0, o0[j] * oScale);
        writeBf16(params.out, outBase + c1, o1[j] * oScale);
    }
    if (tid < kHeadsPerBlock)
        params.lse[lseRowBase + headF] = (denom > 0.0F) ? (logf(denom) + finalMax) : kNegInf;

    // All warps must finish their TMEM reads before warp 0 deallocates TMEM.
    __syncthreads();
    if (warp == 0)
        cg::TMEM::Allocator1Sm().free(0, 512);
#endif
}

// Combine partial flash states across splits into final out/lse, per (row, head).
__global__ __launch_bounds__(128) void sparseMlaDecodeKvarnHotCombineKernel(SparseMlaDecodeKvarnHotParams params)
{
    int32_t const row = static_cast<int32_t>(blockIdx.x);
    int32_t const head = static_cast<int32_t>(blockIdx.y);
    int32_t const totalRows = params.b * params.sQ;
    if (row >= totalRows || head >= params.hQ)
    {
        return;
    }
    int32_t const numSplits = params.numSplits;
    int32_t const tid = static_cast<int32_t>(threadIdx.x);

    int32_t const batch = row / params.sQ;
    int32_t const s = row - batch * params.sQ;
    int64_t const outBase = static_cast<int64_t>(batch) * params.strideOB
        + static_cast<int64_t>(s) * params.strideOSQ + static_cast<int64_t>(head) * params.strideOHQ;
    int64_t const lseOff = static_cast<int64_t>(batch) * params.strideLseB
        + static_cast<int64_t>(s) * params.strideLseSQ + head;

    int64_t const partRowHead = (static_cast<int64_t>(row) * params.hQ + head) * numSplits;

    __shared__ float sMax;
    __shared__ float sDenom;
    float const sinkVal = params.attnSink == nullptr ? kNegInf : params.attnSink[head];
    if (tid == 0)
    {
        float gmax = sinkVal;
        for (int32_t sp = 0; sp < numSplits; ++sp)
            gmax = fmaxf(gmax, params.partialMax[partRowHead + sp]);
        float gden = (sinkVal != kNegInf && gmax != kNegInf) ? expf(sinkVal - gmax) : 0.0F;
        for (int32_t sp = 0; sp < numSplits; ++sp)
        {
            float const m = params.partialMax[partRowHead + sp];
            if (m == kNegInf)
                continue;
            gden += params.partialDenom[partRowHead + sp] * expf(m - gmax);
        }
        sMax = gmax;
        sDenom = gden;
    }
    __syncthreads();

    float const gmax = sMax;
    float const gden = sDenom;
    float const invDenom = gden > 0.0F ? 1.0F / gden : 0.0F;

    for (int32_t d = tid; d < kDv; d += 128)
    {
        float o = 0.0F;
        for (int32_t sp = 0; sp < numSplits; ++sp)
        {
            float const m = params.partialMax[partRowHead + sp];
            if (m == kNegInf)
                continue;
            float const scale = expf(m - gmax);
            o += params.partialAcc[(partRowHead + sp) * kDv + d] * scale;
        }
        writeBf16(params.out, outBase + d, o * invDenom);
    }
    if (tid == 0)
        params.lse[lseOff] = (gden > 0.0F) ? (logf(gden) + gmax) : kNegInf;
}

} // namespace

void invokeSparseMlaDecodeKvarnHot(SparseMlaDecodeKvarnHotParams const& paramsIn, cudaStream_t stream)
{
    SparseMlaDecodeKvarnHotParams params = paramsIn;
    if (params.hQ != kHeadQ || params.dQk != kDqk || params.dV != kDv || params.tokensPerBlock != kTokensPerBlock
        || params.kvLoraRank != kKvLoraRank || params.qkRopeHeadDim != kQkRopeHeadDim || params.kvarnBits != 2)
    {
        throw std::runtime_error(
            "sparse MLA KVarN-hot decode requires production V3.2 shape h_q=128,d_qk=576,d_v=512,tpb=64,kvarn_k2v2");
    }
    if (params.b <= 0 || params.sQ <= 0 || params.topK <= 0 || params.numLayers <= 0 || params.hotCapacity <= 0)
    {
        throw std::runtime_error("sparse MLA KVarN-hot decode requires positive batch, s_q, topk, layers, and hot capacity");
    }
    if (params.residentKvPool != nullptr && params.residentKvPoolDtype != kResidentKvPoolBf16
        && params.residentKvPoolDtype != kResidentKvPoolFp16)
    {
        throw std::runtime_error("sparse MLA KVarN-hot decode requires resident KV pool dtype bf16 or fp16");
    }
    if (params.topK > 2048)
    {
        throw std::runtime_error("sparse MLA KVarN-hot decode currently supports topk <= 2048");
    }

    int32_t const totalRows = params.b * params.sQ;
    int32_t const baseBlocks = totalRows * kHeadGroups;
    int32_t const tilesTotal = (params.topK + kTileTokens - 1) / kTileTokens;
    // Split-K factor. This kernel is TMEM-bound to 1 EFFECTIVE block/SM: it allocates all
    // 512 TMEM columns and a B200 SM has exactly 512, so a second SMEM-resident CTA cannot
    // run concurrently (verified: a single SM with two 512-col-TMEM CTAs doubles kernel
    // walltime). Therefore the optimum is to fill the SMs to ~ONE wave (baseBlocks*numSplits
    // <= #SMs) and no more -- extra splits beyond one wave only multiply per-CTA overhead
    // (TMEM alloc/free, barrier init, Q-reload) under serialized execution. Measured B16:
    // numSplits 10 (320 CTAs) = 0.347ms vs numSplits 4 (128 CTAs, 1 wave) = 0.217ms.
    // HISPARSE_TARGET_BLOCKS overrides the SM target for tuning.
    static int32_t const smCount = []() {
        char const* e = std::getenv("HISPARSE_TARGET_BLOCKS");
        if (e != nullptr) { int32_t v = atoi(e); if (v > 0) return v; }
        int dev = 0; cudaGetDevice(&dev);
        int sm = 148; cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);
        return static_cast<int32_t>(sm);
    }();
    // numSplits = #SMs / baseBlocks, floored to stay within a single wave (>=1).
    int32_t numSplits = max(1, smCount / max(1, baseBlocks));
    numSplits = min(numSplits, min(kMaxSplits, max(1, tilesTotal)));
    if (numSplits < 1)
        numSplits = 1;
    params.numSplits = numSplits;

    // Dynamic SMEM for the kernel's SmemPlan. Computed from the same constants the device
    // SmemPlan uses (SW128 operand tiles are exact multiples of their elem counts) plus a
    // generous slack for the scalars/barriers/alignment. Over-provisioning is safe.
    size_t const sharedBytes = (static_cast<size_t>(kHeadsPerBlock) * kDqk      // q
                                   + static_cast<size_t>(kTileTokens) * kDqk    // k (V shares it)
                                   + static_cast<size_t>(kHeadsPerBlock) * kTileTokens) // p
            * sizeof(__nv_bfloat16)
        + (static_cast<size_t>(2 * kHeadsPerBlock) + kThreads) * sizeof(float)
        + static_cast<size_t>(kTileTokens) * sizeof(uint8_t)
        + 4096; // tmemBase + 2 barriers + rowCode/tileStatusAgg + alignment slack

    static bool attrSet = false;
    if (!attrSet)
    {
        cudaFuncSetAttribute(
            sparseMlaDecodeKvarnHotKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sharedBytes));
        attrSet = true;
    }

    float* partialAcc = nullptr;
    float* partialMax = nullptr;
    float* partialDenom = nullptr;
    if (numSplits > 1)
    {
        size_t const nRowHeadSplit = static_cast<size_t>(totalRows) * params.hQ * numSplits;
        cudaMallocAsync(reinterpret_cast<void**>(&partialAcc), nRowHeadSplit * kDv * sizeof(float), stream);
        cudaMallocAsync(reinterpret_cast<void**>(&partialMax), nRowHeadSplit * sizeof(float), stream);
        cudaMallocAsync(reinterpret_cast<void**>(&partialDenom), nRowHeadSplit * sizeof(float), stream);
        params.partialAcc = partialAcc;
        params.partialMax = partialMax;
        params.partialDenom = partialDenom;
    }

    dim3 const grid(totalRows, kHeadGroups, numSplits);
    sparseMlaDecodeKvarnHotKernel<<<grid, kThreads, sharedBytes, stream>>>(params);
    auto err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        if (partialAcc != nullptr)
        {
            cudaFreeAsync(partialAcc, stream);
            cudaFreeAsync(partialMax, stream);
            cudaFreeAsync(partialDenom, stream);
        }
        throw std::runtime_error("sparse MLA KVarN-hot decode kernel launch failed");
    }

    if (numSplits > 1)
    {
        dim3 const cgrid(totalRows, params.hQ, 1);
        sparseMlaDecodeKvarnHotCombineKernel<<<cgrid, 128, 0, stream>>>(params);
        err = cudaGetLastError();
        cudaFreeAsync(partialAcc, stream);
        cudaFreeAsync(partialMax, stream);
        cudaFreeAsync(partialDenom, stream);
        if (err != cudaSuccess)
        {
            throw std::runtime_error("sparse MLA KVarN-hot decode combine kernel launch failed");
        }
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END

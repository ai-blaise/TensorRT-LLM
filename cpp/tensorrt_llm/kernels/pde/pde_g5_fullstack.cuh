// PDE G5 — FULL-LAYER-STACK persistent decode megakernel (standalone, sm_100).
//
// The integration capstone of the Persistent Decode Engine: N synthetic decoder
// layers run under ONE persistent cooperative launch, with CROSS-LAYER WEIGHT
// PREFETCH. It composes the validated PDE primitives:
//   - G0 substrate : persistent cooperative grid + grid-wide region barrier.
//   - G1 regions   : a weight-heavy GEMM region as the per-layer compute body.
//   - G4 het-worker: COPY ∥ COMPUTE overlap, lifted from KV-staging to WEIGHT
//                    staging — producer warps stream layer L+1's weights while
//                    MMA warps compute layer L (the UNBLOCKED analog of the
//                    dependency-blocked KV prefetch: weights are STATIC, their
//                    GMEM addresses are known a step ahead).
//   - G9 persistence: device-resident multi-UNIT execution under one launch with
//                    state handed unit->unit on-chip, lifted from cross-STEP to
//                    cross-LAYER.
//
// WHY G5 IS NOT GEMM-FUSION-FOR-ITS-OWN-SAKE.
// G2 already proved that fusing compute GEMMs into a megakernel purely to delete
// launches is NET-NEGATIVE at decode scale: the per-launch host cost (~a few us)
// is dwarfed by the grid-barrier + occupancy penalties of a monolith. So the
// launch-elimination (N launches -> 1) is the SMALL lever here (G2 measured the
// launch-elim alone at ~5%). The DISTINCT, unblocked G5 value is the CROSS-LAYER
// WEIGHT PREFETCH: decode is weight-BANDWIDTH-bound (batch M is tiny, so every
// weight byte is read ~once per layer and the layer time is essentially the time
// to STREAM the weights from HBM). In a persistent grid the DMA/producer warps
// can prefetch layer L+1's weight tiles GMEM->staged-buffer DURING layer L's
// compute, hiding the dominant weight-load latency behind the MMA of the prior
// layer. THAT is the win we isolate (prefetch ON vs OFF).
//
// DECODE-REALISTIC SHAPES. hidden H = 7168 (DeepSeek-V3.2 model dim). Each
// synthetic layer = an "attention output projection" GEMM (Kp=512 -> H) followed
// by an "MoE expert FC1" GEMM (H -> Nm). The two GEMMs' WEIGHT volume per layer
// (Kp*H + H*Nm bf16) is the BW-bound quantity; the decode batch M is small
// (8/16/32) so the layer is weight-streaming-bound, not math-bound — exactly the
// regime where weight prefetch pays.
//
// CORRECTNESS (HARD, never a self-compare). The N-layer megakernel output ==
// the per-layer-launch baseline == an INDEPENDENT CPU reference, bit-exact /
// cos >= 0.999999, across N (4/16/61). The per-layer activation recurrence is
// deterministic (fixed warp-shuffle reduction order in BOTH the megakernel and
// the baseline so they match in float too; the CPU does the same math in f64 and
// is compared by cosine). Activations hand off layer->layer through a device
// buffer kept L2-resident; bytes are checked at the final layer.
//
// Standalone nvcc -arch=sm_100. Does NOT touch the decode/model path; the
// ABI-frozen files (SparseMlaDecodeKvarnHotOp.cpp / hisparseKvarnBdrRead.cuh)
// are untouched.
#pragma once

#include "pde_substrate.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>

namespace pde {
namespace g5 {

namespace cg = cooperative_groups;
namespace wmma = nvcuda::wmma;

// ---------------------------------------------------------------------------
// Geometry. Decode-realistic: H=7168 model hidden. A "layer" is two weight-heavy
// GEMMs chained:
//   GEMM-1 ("attn out-proj"):  Y1[M, H ]  = X [M, Kp] * W1[Kp, H ]
//   GEMM-2 ("MoE FC1")      :  Y2[M, Nm]  = Y1[M, H ] * W2[H , Nm]
// then the layer's OUTPUT activation X_next[M, Kp] is a fixed deterministic
// reduction of Y2 back down to Kp so the next layer consumes a Kp-wide input
// (a residual-like contraction; keeps the inter-layer hand-off small + the
// recurrence well-defined across N layers). The dominant BYTE traffic per layer
// is the WEIGHTS: (Kp*H + H*Nm) bf16. With M tiny this is the weight-BW regime.
// ---------------------------------------------------------------------------
constexpr int kH = 7168;     // model hidden
constexpr int kKp = 512;     // attn v-path / kv_lora_rank (GEMM-1 K, layer I/O width)
constexpr int kNm = 2048;    // MoE FC1 intermediate (GEMM-2 N)

// wmma atom + tiling (mirrors G1 so the GEMM math is the validated path).
constexpr int kWmmaM = 16, kWmmaN = 16, kWmmaK = 16;
constexpr int kTileM = 16;                       // decode batch band (M<=32 -> <=2 bands)
constexpr int kNumMmaWarps = 4;
constexpr int kTileN = kWmmaN * kNumMmaWarps;    // 64
constexpr int kTileK = 32;
constexpr int kNumDmaWarps = 1;                  // warp0 = weight DMA producer
constexpr int kNumWarps = kNumMmaWarps + kNumDmaWarps;  // 5
constexpr int kBlockThreads = kNumWarps * 32;    // 160

using elem_t = __nv_bfloat16;
__device__ __forceinline__ float to_f(elem_t x) { return __bfloat162float(x); }
__host__ inline elem_t to_e_host(float x) { return __nv_bfloat16(x); }

// SMEM staging for the GEMM (single-buffer tile, padded LD to cut bank conflicts).
constexpr int kLdA = kTileK + 8;
constexpr int kLdW = kTileN + 8;
constexpr int kSmemAelems = kTileM * kLdA;
constexpr int kSmemWelems = kTileK * kLdW;

__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* gmem_ptr) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n");
}

// ---------------------------------------------------------------------------
// Per-layer weights. Each layer L owns W1[L] (Kp x H) and W2[L] (H x Nm),
// row-major, contiguous in one big device array indexed by layer. The
// "stationary"/baseline reads them straight from GMEM. The "prefetch" path also
// keeps a STAGED double-buffer (hot-weight ring) into which producer warps stream
// the NEXT layer's weights while the current layer computes; the compute then
// reads from the staged buffer instead of GMEM.
//
//   w1[L] base = W1_all + (size_t)L * Kp * H
//   w2[L] base = W2_all + (size_t)L * H  * Nm
// ---------------------------------------------------------------------------
struct LayerWeights {
  const elem_t* W1_all;   // [N_layers, Kp, H]
  const elem_t* W2_all;   // [N_layers, H, Nm]
  int n_layers;
};
__device__ __forceinline__ const elem_t* w1_of(const LayerWeights& w, int L) {
  return w.W1_all + (size_t)L * kKp * kH;
}
__device__ __forceinline__ const elem_t* w2_of(const LayerWeights& w, int L) {
  return w.W2_all + (size_t)L * kH * kNm;
}
constexpr size_t kW1Elems = (size_t)kKp * kH;     // per-layer W1 element count
constexpr size_t kW2Elems = (size_t)kH * kNm;     // per-layer W2 element count
constexpr size_t kWLayerElems = kW1Elems + kW2Elems;  // total per-layer weight elems
constexpr size_t kWLayerBytes = kWLayerElems * sizeof(elem_t);

// ---------------------------------------------------------------------------
// GEMM region descriptor + the validated one-tile wmma body (mirrors G1).
// A = [M,K] row-major, W = [K,N] row-major (K-major), C = [M,N] row-major.
// Cbf = bf16 copy of C (the next region/layer's activation). The WEIGHT pointer
// `W` is the only thing that differs between the GMEM-direct and staged-buffer
// reads; the math is byte-identical so all paths agree.
// ---------------------------------------------------------------------------
struct GemmRegion {
  const elem_t* A;
  const elem_t* W;
  float* C;
  elem_t* Cbf;
  int M, K, N;
};

__device__ __forceinline__ void stage_tile(const elem_t* A, int M, int K,
                                            const elem_t* W, int N, int m0,
                                            int k0, int n0, elem_t* sA,
                                            elem_t* sW, int lane) {
  for (int v = lane; v < (kTileM * kTileK) / 8; v += 32) {
    int row = (v * 8) / kTileK;
    int col = (v * 8) % kTileK;
    if ((m0 + row) < M && (k0 + col) + 7 < K) {
      cp_async_16(sA + row * kLdA + col, A + (size_t)(m0 + row) * K + (k0 + col));
    } else {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        bool ok = ((m0 + row) < M) && ((k0 + col + e) < K);
        sA[row * kLdA + col + e] =
            ok ? A[(size_t)(m0 + row) * K + (k0 + col + e)] : __float2bfloat16(0.0f);
      }
    }
  }
  for (int v = lane; v < (kTileK * kTileN) / 8; v += 32) {
    int row = (v * 8) / kTileN;
    int col = (v * 8) % kTileN;
    if ((k0 + row) < K && (n0 + col) + 7 < N) {
      cp_async_16(sW + row * kLdW + col, W + (size_t)(k0 + row) * N + (n0 + col));
    } else {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        bool ok = ((k0 + row) < K) && ((n0 + col + e) < N);
        sW[row * kLdW + col + e] =
            ok ? W[(size_t)(k0 + row) * N + (n0 + col + e)] : __float2bfloat16(0.0f);
      }
    }
  }
}

// One N-tile of C[M,N] = A[M,K]*W[K,N]. warp0 stages each K-tile; warps 1..4
// wmma their 16-wide N sub-tile; __syncthreads handoff. ALL CTA threads run every
// __syncthreads (uniform control flow). Identical to the G1-validated body.
__device__ __forceinline__ void gemm_one_tile(const GemmRegion& g, int n0,
                                              elem_t* sA, elem_t* sW) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const bool is_mma = (warp >= 1 && warp <= kNumMmaWarps);
  const int k_steps = (g.K + kTileK - 1) / kTileK;
  __shared__ float s_epi[kNumMmaWarps][kWmmaM * kWmmaN];

  for (int m0 = 0; m0 < g.M; m0 += kTileM) {
    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc;
    if (is_mma) wmma::fill_fragment(acc, 0.0f);
    for (int ks = 0; ks < k_steps; ++ks) {
      const int k0 = ks * kTileK;
      if (warp == 0) {
        stage_tile(g.A, g.M, g.K, g.W, g.N, m0, k0, n0, sA, sW, lane);
        cp_async_commit();
        cp_async_wait_all();
      }
      __syncthreads();
      if (is_mma) {
        const int wn = (warp - 1) * kWmmaN;
#pragma unroll
        for (int kk = 0; kk < kTileK / kWmmaK; ++kk) {
          wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, elem_t, wmma::row_major> fa;
          wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, elem_t, wmma::row_major> fb;
          wmma::load_matrix_sync(fa, sA + kk * kWmmaK, kLdA);
          wmma::load_matrix_sync(fb, sW + (kk * kWmmaK) * kLdW + wn, kLdW);
          wmma::mma_sync(acc, fa, fb, acc);
        }
      }
      __syncthreads();
    }
    if (is_mma) {
      const int wn = (warp - 1) * kWmmaN;
      const int col0 = n0 + wn;
      wmma::store_matrix_sync(s_epi[warp - 1], acc, kWmmaN, wmma::mem_row_major);
      __syncwarp();
      for (int idx = lane; idx < kWmmaM * kWmmaN; idx += 32) {
        int r = idx / kWmmaN, c = idx % kWmmaN;
        int gr = m0 + r, gc = col0 + c;
        if (gr < g.M && gc < g.N) {
          float val = s_epi[warp - 1][idx];
          g.C[(size_t)gr * g.N + gc] = val;
          // Cbf is the next-region activation in bf16; GEMM-2 has no bf16
          // consumer (the contraction reads the f32 C), so Cbf may be null —
          // skip the store to avoid a read/write alias on the input buffer.
          if (g.Cbf != nullptr) g.Cbf[(size_t)gr * g.N + gc] = __float2bfloat16(val);
        }
      }
    }
    __syncthreads();
  }
}

// Grid-stride over the N-tiles of one GEMM region, distributed across a COMPUTE
// pool of `pool_size` CTAs indexed by `rank` (0-based). When the whole grid
// computes, rank=blockIdx.x and pool_size=gridDim.x. When a producer group is
// carved off for prefetch, the COMPUTE pool is smaller and MUST still cover ALL
// tiles — hence the explicit rank/pool_size (a plain blockIdx/gridDim stride
// would leave the producer CTAs' tiles uncomputed).
__device__ __forceinline__ void gemm_region_gridstride(const GemmRegion& g,
                                                       int rank, int pool_size,
                                                       elem_t* sA, elem_t* sW) {
  const int n_tiles = (g.N + kTileN - 1) / kTileN;
  for (int t = rank; t < n_tiles; t += pool_size) {
    gemm_one_tile(g, t * kTileN, sA, sW);
  }
}

// ---------------------------------------------------------------------------
// Inter-layer contraction. After GEMM-2 produces Y2[M, Nm], the next layer's
// input X_next[M, Kp] is a fixed, deterministic linear contraction of Y2 with a
// shared (layer-independent) projection Wd[Nm, Kp], plus a bounded squash so the
// activation neither vanishes nor explodes across N=61 layers. Reduction order
// is fixed (sequential over Nm in the kernel AND the CPU does the same in f64 by
// cosine) so the megakernel and baseline agree bit-for-bit in their wmma outputs
// and the recurrence stays stable. ONE CTA computes the whole contraction (M*Kp
// is tiny: 32*512); other CTAs no-op but still hit the surrounding barriers.
//
//   Xn[m,p] = squash( sum_n Y2[m,n] * Wd[n,p] * kDownScale )
//   squash(v) = v / (1 + |v|)    (bounded, libm-free, exactly reproducible)
// ---------------------------------------------------------------------------
constexpr float kDownScale = 1.0f / 512.0f;  // keep the contraction O(1)

struct Contraction {
  const float* Y2;     // [M, Nm]  (f32 GEMM-2 output)
  const elem_t* Wd;    // [Nm, Kp] shared down-projection
  elem_t* Xn;          // [M, Kp]  next-layer input (bf16)
  float* Xn_f;         // [M, Kp]  f32 mirror (for the checksum / cosine)
  int M;
};

__device__ __forceinline__ float squash(float v) { return v / (1.0f + fabsf(v)); }

// Single-CTA contraction. warp w owns output columns p = w, w+kNumWarps, ...;
// each thread reduces a strided slice of Nm then warp-reduces. Deterministic.
__device__ __forceinline__ void contract_one_cta(const Contraction& c) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  for (int m = 0; m < c.M; ++m) {
    const float* y = c.Y2 + (size_t)m * kNm;
    for (int p = warp; p < kKp; p += kNumWarps) {
      float acc = 0.0f;
      // fixed reduction order: lane l accumulates n = l, l+32, ... then tree-reduce.
      for (int n = lane; n < kNm; n += 32) acc += y[n] * to_f(c.Wd[(size_t)n * kKp + p]);
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
      if (lane == 0) {
        float v = squash(acc * kDownScale);
        c.Xn_f[(size_t)m * kKp + p] = v;
        c.Xn[(size_t)m * kKp + p] = __float2bfloat16(v);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// CROSS-LAYER WEIGHT PREFETCH — the staged hot-weight double-buffer ("ring").
//
// Two slots, each big enough for ONE layer's weights laid out as
// [W1 (Kp*H) | W2 (H*Nm)] contiguously. Producer (DMA) warps across the COPY
// CTA-group stream layer L+1's weights from GMEM into the inactive slot while the
// MMA CTA-group computes layer L out of the active slot. After the grid barrier
// the slots swap. This is the G4 COPY∥COMPUTE overlap at LAYER granularity; it
// is UNBLOCKED because the L+1 addresses are known during layer L (static
// weights), unlike the dependency-blocked KV prefetch.
//
// The staged buffer lives in DEVICE global memory but is the canonical target of
// an L2 persisting-access window the host installs, so re-reads during the GEMM
// hit L2 rather than HBM (the realistic "on-chip-resident weight tile"); the
// point measured is that the STREAM of L+1 overlaps the COMPUTE of L.
// ---------------------------------------------------------------------------
struct WeightRing {
  elem_t* slot[2];    // each [kWLayerElems]
  int n_layers;
};
__device__ __forceinline__ const elem_t* ring_w1(const WeightRing& r, int s) {
  return r.slot[s];
}
__device__ __forceinline__ const elem_t* ring_w2(const WeightRing& r, int s) {
  return r.slot[s] + kW1Elems;
}

// Stream layer L's full weight block (W1 then W2) from GMEM into ring slot `s`.
// Cooperative across a producer pool of `n_producers` CTAs; `producer_rank` is
// this CTA's 0-based index within that pool. The flat thread id is computed from
// the RANK (not blockIdx) so the strided coverage is exact regardless of which
// physical CTAs form the pool (e.g. CTAs [1, n_copy)). Vectorized float4
// (8 bf16 / 16 B). W1_all and W2_all are not adjacent in GMEM, so the two ranges
// are streamed separately into the contiguous ring slot [W1 | W2].
__device__ __forceinline__ void stream_layer_weights(const LayerWeights& w,
                                                     const WeightRing& r, int L,
                                                     int s, int producer_rank,
                                                     int n_producers) {
  if (n_producers < 1) return;
  elem_t* dst = r.slot[s];
  const elem_t* w1 = w1_of(w, L);
  const size_t v1 = kW1Elems / 8;
  const float4* s1 = reinterpret_cast<const float4*>(w1);
  float4* d1 = reinterpret_cast<float4*>(dst);
  const size_t tid = (size_t)producer_rank * blockDim.x + threadIdx.x;
  const size_t nthreads = (size_t)n_producers * blockDim.x;
  for (size_t i = tid; i < v1; i += nthreads) d1[i] = s1[i];
  const elem_t* w2 = w2_of(w, L);
  const size_t v2 = kW2Elems / 8;
  const float4* s2 = reinterpret_cast<const float4*>(w2);
  float4* d2 = reinterpret_cast<float4*>(dst + kW1Elems);
  for (size_t i = tid; i < v2; i += nthreads) d2[i] = s2[i];
}

// ===========================================================================
// PER-LAYER ACTIVATION I/O. The megakernel and baseline both carry the layer
// activations through DEVICE buffers, never the host:
//   X      : [M, Kp]   current layer input (layer 0 = the prompt activation)
//   Y1/Y1b : [M, H ]   GEMM-1 out (f32 + bf16)
//   Y2     : [M, Nm]   GEMM-2 out (f32)
//   Xn/Xnf : [M, Kp]   next-layer input (becomes X for layer L+1)
// (We checksum Xn_f at the FINAL layer; that f32 stream is the gate quantity.)
// ===========================================================================
struct LayerIO {
  elem_t* X;      // [M, Kp]  (also reused as Xn target each layer; see swap)
  float* Y1_f;    // [M, H]
  elem_t* Y1_b;   // [M, H]
  float* Y2_f;    // [M, Nm]
  elem_t* Xn;     // [M, Kp]
  float* Xn_f;    // [M, Kp]
  int M;
};

struct StackParams {
  LayerWeights w;
  LayerIO io;
  const elem_t* Wd;    // [Nm, Kp] shared down-proj
  WeightRing ring;     // staged hot-weight ring (prefetch path)
  int n_layers;
  int use_prefetch;    // 1 = cross-layer weight prefetch ON; 0 = read GMEM direct
  int n_copy;          // # COPY CTAs for the prefetch producer (rest = MMA/compute)
};

// One layer's compute (GEMM-1 -> GEMM-2 -> contraction), reading weights from
// `w1src`/`w2src` (either GMEM-direct or a ring slot). Whole grid grid-strides
// the GEMM tiles; CTA 0 does the single-CTA contraction. The two GEMMs are
// separated by a grid barrier (GEMM-2 reads GEMM-1's output Y1 from L2), and a
// final barrier publishes Xn before the next layer. ALL CTAs execute every
// barrier (uniform). We use cg::grid.sync() — the G1/G2/G4-validated cooperative
// grid barrier — which is robust across the hundreds of back-to-back barriers an
// N=61 stack issues (the hand-rolled sense-reversing barrier is fragile under
// rapid re-entry; grid.sync is purpose-built for it).
// `participate`: a CTA in the COMPUTE group runs the GEMM/contraction work; a CTA
// in the PRODUCER group (prefetch path) passes participate=false and SKIPS the
// work but STILL executes all 3 grid.sync()s in lockstep, so the cooperative
// barrier stays well-formed while the producer's weight stream (issued BEFORE
// this call) runs CONCURRENTLY with the compute group's GEMM — the actual
// cross-layer overlap. CTA 0 is always in the compute group (it does the
// single-CTA contraction).
__device__ __forceinline__ void run_layer_compute(const StackParams& p,
                                                  const elem_t* w1src,
                                                  const elem_t* w2src,
                                                  const cg::grid_group& grid,
                                                  bool participate, int crank,
                                                  int cpool, elem_t* sA,
                                                  elem_t* sW) {
  // GEMM-1: Y1[M,H] = X[M,Kp] * W1[Kp,H]
  if (participate) {
    GemmRegion g1{p.io.X, w1src, p.io.Y1_f, p.io.Y1_b, p.io.M, kKp, kH};
    gemm_region_gridstride(g1, crank, cpool, sA, sW);
  }
  cg_grid_barrier(grid);                        // Y1 visible to all
  // GEMM-2: Y2[M,Nm] = Y1[M,H] * W2[H,Nm]. Cbf=nullptr (no bf16 consumer; the
  // contraction reads the f32 Y2). Must NOT alias Y1_b (read/write hazard).
  if (participate) {
    GemmRegion g2{p.io.Y1_b, w2src, p.io.Y2_f, /*Cbf=*/nullptr, p.io.M, kH, kNm};
    gemm_region_gridstride(g2, crank, cpool, sA, sW);
  }
  cg_grid_barrier(grid);                        // Y2 visible to all
  // Contraction: Xn[M,Kp] = squash(Y2 * Wd) — single CTA (always CTA 0).
  if (blockIdx.x == 0) {
    Contraction c{p.io.Y2_f, p.Wd, p.io.Xn, p.io.Xn_f, p.io.M};
    contract_one_cta(c);
  }
  cg_grid_barrier(grid);                        // Xn visible before it becomes X
}

// ===========================================================================
// (A) FULL-STACK PERSISTENT MEGAKERNEL — ONE cooperative launch, N layers,
//     cross-layer weight prefetch.
//
// Grid roles (only when use_prefetch=1): CTAs [0,n_copy) are the WEIGHT-PRODUCER
// group that streams layer L+1's weights into the inactive ring slot; ALL CTAs
// (including the producers) participate in the per-layer GEMM grid-stride compute
// of layer L out of the active slot. Because the producer's stream and the
// compute's GEMM are independent (L+1 weights vs L weights), they overlap; a grid
// barrier at the layer boundary joins them and swaps slots.
//
// use_prefetch=0: no ring, every layer reads its weights straight from GMEM
// (the in-megakernel control for the prefetch lever — isolates launch-elim from
// weight-prefetch-overlap).
//
// Barrier discipline: cg::grid.sync(). The COPY producers do a DIFFERENT amount
// of work than the compute path, but every CTA calls grid.sync() the SAME number
// of times (the prefetch stream is issued BEFORE run_layer_compute, which all
// CTAs enter identically), so the cooperative barrier is well-formed. Each layer
// issues exactly 4 grid.sync()s (after GEMM-1, GEMM-2, contraction, then the
// prefetch JOIN) on EVERY CTA, plus 1 prime barrier in the prefetch path. The
// prefetch-OFF path issues a matching no-op 4th barrier so the COUNT is identical
// and an A/B timing diff reflects only the weight-prefetch overlap (plus, for the
// OFF path, the prime barrier is replaced by an equivalent leading grid.sync()).
// ===========================================================================
__global__ void __launch_bounds__(kBlockThreads)
    kFullStackMega(StackParams p, GlobalBarrier /*unused*/) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ unsigned char smem[];
  elem_t* sA = reinterpret_cast<elem_t*>(smem);
  elem_t* sW = sA + kSmemAelems;

  const bool prefetch = (p.use_prefetch != 0);
  // Grid partition (prefetch path): the TOP `n_prod` CTAs [grid-n_prod, grid) are
  // the PRODUCER group that streams layer L+1's weights concurrently with the
  // compute; the BOTTOM CTAs [0, grid-n_prod) are the COMPUTE group that runs the
  // GEMMs from the ring (CTA 0 is in it, owning the single-CTA contraction). A
  // CONTIGUOUS split lets the compute group rank densely as [0, cpool) and cover
  // ALL GEMM tiles (a producer carve-out from the middle would leave holes). When
  // prefetch is OFF, every CTA computes (n_prod=0).
  const int n_prod = prefetch ? (p.n_copy - 0) : 0;   // p.n_copy producers
  const int cpool = gridDim.x - n_prod;               // compute-pool size
  const bool is_producer = prefetch && (blockIdx.x >= cpool);
  const bool participate = !is_producer;              // compute group runs GEMMs
  const int crank = participate ? blockIdx.x : 0;     // dense compute rank [0,cpool)
  const int prank = is_producer ? (blockIdx.x - cpool) : 0;  // producer rank
  int active = 0;  // ring slot holding the CURRENT layer's weights

  // Prime barrier (both paths, so the total barrier count matches exactly): in
  // the prefetch path the WHOLE grid streams layer 0's weights into the active
  // slot (rank = blockIdx.x over all gridDim.x CTAs); in the no-prefetch path it
  // is a bare grid.sync() placeholder.
  if (prefetch) {
    stream_layer_weights(p.w, p.ring, /*L=*/0, active, /*rank=*/blockIdx.x,
                         /*n_producers=*/gridDim.x);
  }
  cg_grid_barrier(grid);

  for (int L = 0; L < p.n_layers; ++L) {
    const elem_t* w1src;
    const elem_t* w2src;
    if (prefetch) {
      w1src = ring_w1(p.ring, active);
      w2src = ring_w2(p.ring, active);
      // PREFETCH layer L+1 into the inactive slot. The PRODUCER group issues this
      // stream and then enters run_layer_compute with participate=false, so the
      // stream's GMEM traffic OVERLAPS the compute group's GEMM of layer L (read
      // from the active ring slot). This is the cross-layer weight prefetch.
      const int inactive = active ^ 1;
      if (is_producer && (L + 1) < p.n_layers) {
        stream_layer_weights(p.w, p.ring, L + 1, inactive, /*rank=*/prank,
                             /*n_producers=*/n_prod);
      }
      run_layer_compute(p, w1src, w2src, grid, participate, crank, cpool, sA, sW);
      // JOIN producers + compute, then swap slots for the next layer.
      cg_grid_barrier(grid);
      active = inactive;
    } else {
      // No prefetch: read this layer's weights straight from GMEM, whole grid
      // computes (crank=blockIdx.x, cpool=gridDim.x).
      w1src = w1_of(p.w, L);
      w2src = w2_of(p.w, L);
      run_layer_compute(p, w1src, w2src, grid, /*participate=*/true,
                        /*crank=*/blockIdx.x, /*cpool=*/gridDim.x, sA, sW);
      cg_grid_barrier(grid);  // keep barrier COUNT identical to the prefetch path
    }
    // Hand off: Xn becomes the next layer's X. CTA-uniform pointer swap (both
    // paths do the same swap). Xn_f always holds the LATEST layer's f32 output.
    elem_t* tmpX = p.io.X; p.io.X = p.io.Xn; p.io.Xn = tmpX;
  }
}

// ===========================================================================
// (B) PER-LAYER-LAUNCH BASELINE — N separate kernel launches (today's pattern).
//
// Each launch runs ONE layer: GEMM-1, then (a second launch) GEMM-2, then (a
// third launch) the contraction. The host loops L=0..N-1 and relaunches; weights
// are read straight from GMEM every layer with NO cross-layer prefetch (each
// launch starts cold). We split a layer into 3 launches because a non-cooperative
// grid cannot grid-barrier internally — exactly the per-op launch granularity of
// today's decode forward. The activation buffers persist in GMEM across launches
// (the host swaps X/Xn pointers between layers, never reading them back).
// ===========================================================================
__global__ void __launch_bounds__(kBlockThreads) kBaseGemm(GemmRegion g) {
  extern __shared__ unsigned char smem[];
  elem_t* sA = reinterpret_cast<elem_t*>(smem);
  elem_t* sW = sA + kSmemAelems;
  const int n_tiles = (g.N + kTileN - 1) / kTileN;
  for (int t = blockIdx.x; t < n_tiles; t += gridDim.x) gemm_one_tile(g, t * kTileN, sA, sW);
}

__global__ void __launch_bounds__(kBlockThreads) kBaseContract(Contraction c) {
  // single-CTA contraction (only CTA 0 does work; launched with grid>=1).
  if (blockIdx.x == 0) contract_one_cta(c);
}

inline size_t mega_smem_bytes() {
  return (size_t)kSmemAelems * sizeof(elem_t) + (size_t)kSmemWelems * sizeof(elem_t);
}

}  // namespace g5
}  // namespace pde

// PDE G2 — warp-specialized DOUBLE-BUFFERED (N-stage) overlap megakernel
// (standalone, sm_100).
//
// G1 proved the persistent warp-specialized two-region STRUCTURE + on-chip L2
// handoff + correctness, but its producer->consumer handoff is a SINGLE-BUFFER
// __syncthreads(): warp 0 stages a K-tile, the CTA syncs, warps 1..N wmma it,
// the CTA syncs before warp 0 overwrites it. The cp.async W/A load latency is
// therefore FULLY EXPOSED — measured 1.08-1.10x slower than two separate
// kernels.
//
// G2 hides that latency with a software-pipelined N-stage cp.async ring: the
// DMA producer (warp 0) runs kStages-1 K-tiles AHEAD, prefetching tile k+1
// (cp.async, no wait) WHILE the MMA consumers (warps 1..N) wmma tile k.
//
// Pipeline (hand-rolled cp.async.commit_group / cp.async.wait_group ring):
//   * kStages rotating SMEM buffers for A and W.
//   * Prologue: producer issues+commits the first kStages-1 tiles (no wait).
//   * Steady state, per k-step ks:
//       - producer issues+commits tile ks+(kStages-1) into ring slot
//         (ks+kStages-1)%kStages  [if in K range],
//       - cp.async.wait_group (kStages-1)  -> at most kStages-1 groups remain
//         in flight, so the group feeding slot ks%kStages is guaranteed landed,
//       - __syncthreads()  -> landed buffer visible to all warps AND no consumer
//         reads a slot the producer is about to overwrite,
//       - consumers wmma slot ks%kStages,
//       - __syncthreads()  -> consumers done before that slot is reused kStages
//         iters later.
//
// CORRECTNESS NOTE (why this does NOT repeat the G1 agent's deadlock): the G1
// agent used a cuda::barrier full/empty scheme whose arrival count was the whole
// CTA and which was never reset across work-queue tiles -> phase drift ->
// deadlock. Here:
//   * cp.async.wait_group is a PER-THREAD instruction over THAT thread's
//     committed groups; ONLY warp 0 issues/commits cp.async, so ONLY warp 0
//     waits. Consumers never wait_group (they have 0 groups). Cross-warp
//     visibility/ordering is carried by the two plain __syncthreads() that EVERY
//     thread of the CTA executes uniformly (same loop trip count for all warps).
//   * The ring re-issues into a fixed slot each tile and the wait is RELATIVE to
//     the outstanding group count, so accounting self-resets per tile — no
//     persistent phase state to drift. Tested across many work-queue tiles +
//     B=1/8/32.
//
// Everything else is inherited from G1 verbatim: two regions, grid barrier,
// atomic work-queue, M-tiling in 16-row bands, L2 inter-region handoff, bf16
// nvcuda::wmma 16x16x16 (HMMA, no CUTLASS) so it builds standalone with
// nvcc -arch=sm_100. Does NOT touch the decode/model path; ABI-frozen files
// untouched.
#pragma once

#include "pde_substrate.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>

namespace pde {
namespace g2 {

namespace cg = cooperative_groups;
namespace wmma = nvcuda::wmma;

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

constexpr int kTileM = 16;     // decode batch is small (B<=32) -> M-band <= 16
constexpr int kNumMmaWarps = 4;
constexpr int kTileN = kWmmaN * kNumMmaWarps;  // 64: one 16-wide N sub-tile/warp
constexpr int kTileK = 32;     // K consumed per staged step (2 wmma K atoms)

constexpr int kNumDmaWarps = 1;
constexpr int kNumWarps = kNumMmaWarps + kNumDmaWarps;   // warp0=DMA, 1..4=MMA
constexpr int kBlockThreads = kNumWarps * 32;            // 160

// SMEM staging, padded leading dim to cut wmma-load bank conflicts. Same as G1;
// G2 just keeps kStages copies of each (a ring), selected by a template param.
constexpr int kLdA = kTileK + 8;
constexpr int kLdW = kTileN + 8;
constexpr int kSmemAelems = kTileM * kLdA;   // per stage
constexpr int kSmemWelems = kTileK * kLdW;   // per stage

__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* gmem_ptr) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s),
               "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}
// Leave at most N committed-but-unfinished cp.async groups in flight.
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n");
}

// Stage one K-tile of A and W into the given ring-slot SMEM (called by warp 0).
// IDENTICAL byte layout to G1's stage_tile; out-of-range zero-filled, vectorized
// 16B cp.async where the source row is fully in-range, scalar tail otherwise.
// Does NOT commit (caller commits, so prologue/steady-state control the groups).
__device__ __forceinline__ void stage_tile_async(const __nv_bfloat16* A, int M,
                                                  int K, const __nv_bfloat16* W,
                                                  int N, int m0, int k0, int n0,
                                                  __nv_bfloat16* sA,
                                                  __nv_bfloat16* sW, int lane) {
  for (int v = lane; v < (kTileM * kTileK) / 8; v += 32) {
    int row = (v * 8) / kTileK;     // local row within the m0 band [0,kTileM)
    int col = (v * 8) % kTileK;
    if ((m0 + row) < M && (k0 + col) + 7 < K) {
      cp_async_16(sA + row * kLdA + col,
                  A + (size_t)(m0 + row) * K + (k0 + col));
    } else {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        bool ok = ((m0 + row) < M) && ((k0 + col + e) < K);
        sA[row * kLdA + col + e] =
            ok ? A[(size_t)(m0 + row) * K + (k0 + col + e)]
               : __float2bfloat16(0.0f);
      }
    }
  }
  for (int v = lane; v < (kTileK * kTileN) / 8; v += 32) {
    int row = (v * 8) / kTileN;     // k
    int col = (v * 8) % kTileN;     // n
    if ((k0 + row) < K && (n0 + col) + 7 < N) {
      cp_async_16(sW + row * kLdW + col,
                  W + (size_t)(k0 + row) * N + (n0 + col));
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

struct GemmRegion {
  const __nv_bfloat16* A;   // [M, K]  row-major
  const __nv_bfloat16* W;   // [K, N]  row-major (K-major)
  float* C;                 // [M, N]  row-major f32 out (checksum reference)
  __nv_bfloat16* Cbf;       // [M, N]  row-major bf16 out (next-region activation)
  int M;
  int K;
  int N;
};

// One N-tile of C[M,N] = A[M,K]*W[K,N] with an N-stage cp.async OVERLAP pipeline.
// warp 0 prefetches kStages-1 K-tiles ahead; warps 1..kNumMmaWarps wmma the
// current stage. Handoff/visibility via __syncthreads() (ALL CTA threads run
// every sync; the producer-only wait_group is hidden behind the first sync).
// Math is byte-identical to G1's gemm_one_tile -> same result, just overlapped.
template <int kStages>
__device__ __forceinline__ void gemm_one_tile_pipe(const GemmRegion& g, int n0,
                                                   __nv_bfloat16* sA_ring,
                                                   __nv_bfloat16* sW_ring) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const bool is_mma = (warp >= 1 && warp <= kNumMmaWarps);
  const bool is_dma = (warp == 0);
  const int k_steps = (g.K + kTileK - 1) / kTileK;

  __shared__ float s_epi[kNumMmaWarps][kWmmaM * kWmmaN];

  // Ring-slot base pointers.
  auto slotA = [&](int s) { return sA_ring + (size_t)s * kSmemAelems; };
  auto slotW = [&](int s) { return sW_ring + (size_t)s * kSmemWelems; };

  // Tile M (batch) in kTileM-row bands; all warps run the same #iterations so
  // the __syncthreads stay uniform across the whole CTA.
  for (int m0 = 0; m0 < g.M; m0 += kTileM) {
    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc;
    if (is_mma) wmma::fill_fragment(acc, 0.0f);

    // ---- Prologue: prefetch the first (kStages-1) K-tiles, committing one
    // cp.async group per tile. No wait. Out-of-range tiles still issue a (fully
    // zero-filled / no-op) group so the group count is uniform and the steady
    // state wait_group<kStages-1> accounting is exact.
    if (is_dma) {
#pragma unroll
      for (int s = 0; s < kStages - 1; ++s) {
        int k0 = s * kTileK;
        if (s < k_steps)
          stage_tile_async(g.A, g.M, g.K, g.W, g.N, m0, k0, n0, slotA(s),
                           slotW(s), lane);
        cp_async_commit();   // one group per prologue stage (uniform count)
      }
    }

    // ---- Steady state: for each consumed k-step ks, prefetch ks+(kStages-1)
    // then consume ks.
    for (int ks = 0; ks < k_steps; ++ks) {
      const int cur = ks % kStages;                  // slot to consume now
      const int fetch_step = ks + (kStages - 1);     // tile to prefetch now
      const int fetch_slot = fetch_step % kStages;

      if (is_dma) {
        if (fetch_step < k_steps) {
          stage_tile_async(g.A, g.M, g.K, g.W, g.N, m0, fetch_step * kTileK, n0,
                           slotA(fetch_slot), slotW(fetch_slot), lane);
        }
        cp_async_commit();             // keep one group per steady step
        // Leave kStages-1 groups in flight -> the group feeding `cur` is done.
        cp_async_wait_group<kStages - 1>();
      }
      __syncthreads();                 // landed `cur` visible to all warps;
                                       // also fences consumers vs producer reuse

      if (is_mma) {
        const __nv_bfloat16* sA = slotA(cur);
        const __nv_bfloat16* sW = slotW(cur);
        const int wn = (warp - 1) * kWmmaN;
#pragma unroll
        for (int kk = 0; kk < kTileK / kWmmaK; ++kk) {
          wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16,
                         wmma::row_major>
              fa;
          wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16,
                         wmma::row_major>
              fb;
          wmma::load_matrix_sync(fa, sA + kk * kWmmaK, kLdA);
          wmma::load_matrix_sync(fb, sW + (kk * kWmmaK) * kLdW + wn, kLdW);
          wmma::mma_sync(acc, fa, fb, acc);
        }
      }
      __syncthreads();                 // consumers done with `cur` before the
                                       // producer overwrites it kStages later
    }

    // Drain any still-in-flight prefetch groups for this band before reusing the
    // ring in the next m0 band (keeps the steady-state group count exact).
    if (is_dma) cp_async_wait_all();
    __syncthreads();

    // ---- Epilogue: store this warp's 16x16 N sub-tile.
    if (is_mma) {
      const int wn = (warp - 1) * kWmmaN;
      const int col0 = n0 + wn;
      wmma::store_matrix_sync(s_epi[warp - 1], acc, kWmmaN, wmma::mem_row_major);
      __syncwarp();
      for (int idx = lane; idx < kWmmaM * kWmmaN; idx += 32) {
        int r = idx / kWmmaN;
        int c = idx % kWmmaN;
        int gr = m0 + r;
        int gc = col0 + c;
        if (gr < g.M && gc < g.N) {
          float val = s_epi[warp - 1][idx];
          g.C[(size_t)gr * g.N + gc] = val;
          g.Cbf[(size_t)gr * g.N + gc] = __float2bfloat16(val);
        }
      }
    }
    __syncthreads();   // all warps done with this m0 band before advancing
  }
}

// Persistent-grid region body: pull N-tiles from the work-queue until drained.
template <int kStages>
__device__ __forceinline__ void gemm_region_body_pipe(const GemmRegion& g,
                                                      WorkQueue q,
                                                      __nv_bfloat16* sA_ring,
                                                      __nv_bfloat16* sW_ring) {
  const int n_tiles = (g.N + kTileN - 1) / kTileN;
  __shared__ unsigned int s_tile;
  __shared__ int s_have;
  while (true) {
    if (threadIdx.x == 0) {
      unsigned int it;
      s_have = work_queue_pop(q, &it) ? 1 : 0;
      s_tile = it;
    }
    __syncthreads();
    int have = s_have;
    unsigned int tile_id = s_tile;
    if (!have || (int)tile_id >= n_tiles) break;
    gemm_one_tile_pipe<kStages>(g, (int)tile_id * kTileN, sA_ring, sW_ring);
  }
}

inline size_t fused_smem_bytes(int kStages) {
  return (size_t)kStages * kSmemAelems * sizeof(__nv_bfloat16) +
         (size_t)kStages * kSmemWelems * sizeof(__nv_bfloat16);
}

// ---------------------------------------------------------------------------
// THE FUSED TWO-REGION PERSISTENT OVERLAP MEGAKERNEL (templated on stage count).
// Region A -> grid barrier -> region B, ONE cooperative launch. regB.A aliases
// region A's bf16 output (C_A) and is read from L2. Each region uses the same
// kStages-deep cp.async ring.
// ---------------------------------------------------------------------------
template <int kStages>
__global__ void __launch_bounds__(kBlockThreads)
    kFusedTwoRegionPipe(GemmRegion regA, GemmRegion regB, WorkQueue qA,
                        WorkQueue qB) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ unsigned char smem[];
  __nv_bfloat16* sA_ring = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* sW_ring = sA_ring + (size_t)kStages * kSmemAelems;

  gemm_region_body_pipe<kStages>(regA, qA, sA_ring, sW_ring);   // Region A
  cg_grid_barrier(grid);                                        // G0 barrier
  gemm_region_body_pipe<kStages>(regB, qB, sA_ring, sW_ring);   // Region B (L2)
}

// Single-region OVERLAP reference: NON-persistent / NON-cooperative grid-stride
// over N-tiles. Same gemm_one_tile_pipe math as the fused path, so two
// sequential launches form the overlapped "separate" reference (apples to
// apples with the fused overlap). Takes only GemmRegion (grid-stride needs no
// work-queue -> safe to re-launch in tight timing loops).
template <int kStages>
__global__ void __launch_bounds__(kBlockThreads) kSingleRegionPipe(GemmRegion g) {
  extern __shared__ unsigned char smem[];
  __nv_bfloat16* sA_ring = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* sW_ring = sA_ring + (size_t)kStages * kSmemAelems;
  const int n_tiles = (g.N + kTileN - 1) / kTileN;
  for (int tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
    gemm_one_tile_pipe<kStages>(g, tile * kTileN, sA_ring, sW_ring);
  }
}

}  // namespace g2
}  // namespace pde

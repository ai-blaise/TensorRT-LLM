// PDE G1 — two-region warp-specialized persistent megakernel (standalone, sm_100).
//
// ONE persistent cooperative kernel runs TWO real tensor-core GEMM regions
// separated by a grid-wide barrier, with intra-CTA warp specialization:
//
//   Region A  ("attention output projection"): C_A[M,N_A] = A_A[M,K_A] * W_A[K_A,N_A]
//             M = decode batch tokens, K_A = 512 (kv_lora_rank / v-path),
//             N_A = 7168 (model hidden). bf16 inputs, f32 accumulate.
//   GRID BARRIER (pde::cg_grid_barrier) — region A's output globally visible.
//   Region B  ("MoE expert FC1"): C_B[M,N_B] = C_A[M,N_A] * W_B[N_A,N_B]
//             K_B = N_A = 7168 (region B reads region A's activation),
//             N_B = 2048 (intermediate). bf16 inputs, f32 accumulate.
//
// The region-A -> region-B activation (C_A, M*7168*2B = tens-to-hundreds of KB)
// is read by region B FROM L2: the host pins a persisting-L2 access-policy
// window on C_A so its lines stay resident across the barrier; region B never
// round-trips HBM for it. C_A is materialized in bf16 (the "activation" the next
// region consumes) AND f32 (for the correctness checksum).
//
// Warp specialization within each CTA:
//   warp 0            = DMA producer: cp.async-stages bf16 A/W K-tiles GMEM->SMEM.
//   warps 1..kNumMma  = MMA consumers: nvcuda::wmma bf16 16x16x16 tensor-core
//                       math (HMMA), f32 accumulators; each warp owns a 16-wide
//                       N sub-tile.
//
// Producer->consumer handoff is __syncthreads() (single-buffer): warp 0 stages a
// K-tile, the CTA syncs, warps 1..kNumMma consume it, the CTA syncs before warp 0
// overwrites it. Robustly correct (no fragile mbarrier phase accounting across
// tiles). It does NOT pipeline producer ahead of consumer — real cp.async /
// mbarrier double-buffered OVERLAP is gate G2. G1 proves the persistent
// warp-specialized two-region STRUCTURE + on-chip handoff + correctness.
//
// Tensor core: nvcuda::wmma 16x16x16 bf16/f32 is native (HMMA), no CUTLASS, so
// the kernel builds fully standalone with nvcc -arch=sm_100. The frozen FlashMLA
// tcgen05/UMMA path + NVFP4 are LATER gates.
//
// Does NOT touch the decode/model path; compiled standalone. ABI-frozen files
// untouched.
#pragma once

#include "pde_substrate.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>

namespace pde {
namespace g1 {

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

// SMEM staging (single buffer), padded leading dim to cut wmma-load bank conflicts.
constexpr int kLdA = kTileK + 8;
constexpr int kLdW = kTileN + 8;
constexpr int kSmemAelems = kTileM * kLdA;
constexpr int kSmemWelems = kTileK * kLdW;

__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* gmem_ptr) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s),
               "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n");
}

// Stage one K-tile of A and W into SMEM (called by warp 0). Out-of-range is
// zero-filled. Vectorized 16B copies where the source row is fully in-range.
__device__ __forceinline__ void stage_tile(const __nv_bfloat16* A, int M, int K,
                                            const __nv_bfloat16* W, int N,
                                            int m0, int k0, int n0,
                                            __nv_bfloat16* sA, __nv_bfloat16* sW,
                                            int lane) {
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

// One N-tile of C[M,N] = A[M,K]*W[K,N]. warp 0 stages each K-tile; warps
// 1..kNumMmaWarps wmma their 16-wide N sub-tile; handoff via __syncthreads.
// Shared by the fused (work-queue) and single-region (grid-stride) kernels so
// the math is byte-identical. ALL threads of the CTA execute every __syncthreads.
__device__ __forceinline__ void gemm_one_tile(const GemmRegion& g, int n0,
                                              __nv_bfloat16* sA,
                                              __nv_bfloat16* sW) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const bool is_mma = (warp >= 1 && warp <= kNumMmaWarps);
  const int k_steps = (g.K + kTileK - 1) / kTileK;

  __shared__ float s_epi[kNumMmaWarps][kWmmaM * kWmmaN];

  // Tile the M (batch) dimension in kTileM-row bands (one wmma M-atom each), so
  // M > kTileM (e.g. decode B=32 > 16) is handled correctly. All threads of the
  // CTA run the same number of m0 iterations -> the __syncthreads stay uniform.
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
      __syncthreads();   // staged tile visible to all consumers
      if (is_mma) {
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
      __syncthreads();   // consumers done before warp 0 overwrites sA/sW
    }

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

// Persistent-grid region body: pull N-tiles from a work-queue until drained.
__device__ __forceinline__ void gemm_region_body(const GemmRegion& g, WorkQueue q,
                                                 __nv_bfloat16* sA,
                                                 __nv_bfloat16* sW) {
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
    gemm_one_tile(g, (int)tile_id * kTileN, sA, sW);
  }
}

// ---------------------------------------------------------------------------
// THE FUSED TWO-REGION PERSISTENT MEGAKERNEL.
// Region A -> grid barrier -> region B, ONE cooperative launch. regB.A aliases
// region A's bf16 output (C_A) and is read from L2.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kBlockThreads)
    kFusedTwoRegion(GemmRegion regA, GemmRegion regB, WorkQueue qA,
                    WorkQueue qB) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ unsigned char smem[];
  __nv_bfloat16* sA = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* sW = sA + kSmemAelems;

  gemm_region_body(regA, qA, sA, sW);   // Region A
  cg_grid_barrier(grid);                // G0 substrate barrier
  gemm_region_body(regB, qB, sA, sW);   // Region B (reads C_A from L2)
}

inline size_t fused_smem_bytes() {
  return (size_t)kSmemAelems * sizeof(__nv_bfloat16) +
         (size_t)kSmemWelems * sizeof(__nv_bfloat16);
}

// Single-region reference kernel: NON-persistent / NON-cooperative, grid-stride
// over N-tiles on blockIdx. Same gemm_one_tile math as the fused path, so two
// sequential launches form the TRUE reference. Signature takes only GemmRegion
// (grid-stride needs no work-queue -> safe to re-launch in tight timing loops).
__global__ void __launch_bounds__(kBlockThreads) kSingleRegion(GemmRegion g) {
  extern __shared__ unsigned char smem[];
  __nv_bfloat16* sA = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* sW = sA + kSmemAelems;
  const int n_tiles = (g.N + kTileN - 1) / kTileN;
  for (int tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
    gemm_one_tile(g, tile * kTileN, sA, sW);
  }
}

}  // namespace g1
}  // namespace pde

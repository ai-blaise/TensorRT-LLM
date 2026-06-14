// PDE G3 — device-resident DATA-DEPENDENT CONTROL FLOW (standalone, sm_100).
//
// This gate is NOT about fusing GEMMs (G1/G2 showed that is net-negative at decode
// scale). G3 attacks the actual decode "execution gap": host round-trips for
// data-dependent control flow, which are exactly what is CUDA-graph-capture-illegal
// (a d2h readback inside graph capture is illegal). The DeepSeek-V3.2 decode path's
// highest-value boundary ("Barrier 2", Indexer -> attention) is:
//
//   DSA Indexer computes per-query scores over candidate KV blocks
//     -> TOP-K selection (index_topk)
//       -> attention reads ONLY the selected blocks.
//
// Today that selection round-trips host orchestration. G3 proves it can stay
// device-resident in ONE persistent cooperative grid, dissolving the d2h wall, and
// quantifies the host-round-trip latency removed.
//
// Shapes (representative decode): Q queries (= decode batch M in {1,8,32}); C
// candidate blocks/query (C in {2048,4096}); each block a D-dim vector (D=128);
// select top-K (K in {256,2048}, like DSA index_topk).
//
//   Region 1 (SCORE + SELECT): score[q,c] = dot(q_query[q], block[c])  (GEMV-like
//             per query over C candidates); then device-side EXACT top-K selection
//             per query, indices kept in device global memory (NEVER d2h).
//   GRID BARRIER (pde::cg_grid_barrier) — selected indices globally visible.
//   Region 2 (GATHER + REDUCE): out[q,:] = sum over the K selected blocks of
//             softmax_like_weight(score) * block[idx][:]  (stand-in for attention
//             over the selected KV). Data-dependent: which blocks are read depends
//             on region 1's on-device decision.
//
// (A) DEVICE-RESIDENT path = ONE persistent cooperative kernel doing region1 ->
//     grid barrier -> region2; the selection never leaves the device.
// (B) HOST-ORCHESTRATED baseline (today's pattern) = score kernel -> cudaMemcpy
//     scores d2h -> host top-K -> cudaMemcpy indices h2d -> gather kernel. Real
//     copies + stream syncs = the "execution gap" being eliminated.
//
// Device top-K method: one CTA owns one query. It loads all C scores, then runs an
// iterative warp/block argmax-and-evict ("selection") that extracts the top-K in
// descending order with a DETERMINISTIC tiebreak (higher score wins; on an exact
// score tie the LOWER block index wins). The CPU reference uses the identical
// tiebreak so the selected INDEX SET matches bit-for-bit (ties resolved the same).
//
// Tensor cores are not used here on purpose: the indexer score is a thin GEMV and
// the value of G3 is the CONTROL-FLOW boundary, not GEMM throughput. Standalone
// nvcc -arch=sm_100. Does NOT touch the decode/model path; ABI-frozen files
// untouched.
#pragma once

#include "pde_substrate.cuh"

#include <cooperative_groups.h>
#include <cstdint>
#include <cfloat>

namespace pde {
namespace g3 {

namespace cg = cooperative_groups;

constexpr int kD = 128;          // per-block vector dim (DSA index head dim-ish)
constexpr int kBlockThreads = 256;
constexpr int kWarps = kBlockThreads / 32;

// Problem descriptor shared by every kernel + the CPU reference.
struct Problem {
  const float* q;        // [Q, D]   query vectors (row-major)
  const float* blocks;   // [C, D]   candidate block vectors (row-major)
  float* scores;         // [Q, C]   region-1 output: score[q,c]
  int* sel_idx;          // [Q, K]   selected block indices (descending score)
  float* sel_score;      // [Q, K]   the selected scores (for the weighted reduce)
  float* out;            // [Q, D]   region-2 output: weighted sum of selected
  int Q;
  int C;
  int K;
};

// ---------------------------------------------------------------------------
// Region 1a — SCORE. score[q,c] = sum_d q[q,d] * blocks[c,d].
// One CTA per query (grid-stride over queries). Each warp strides candidates;
// each thread reduces D with a strided load, then a warp shuffle finishes the dot.
// D=128 = 4 floats/lane across a warp -> 4 fma + a 5-step shuffle reduce.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_dot_qd(const float* qrow,
                                              const float* brow) {
  const int lane = threadIdx.x & 31;
  float acc = 0.0f;
#pragma unroll
  for (int d = lane; d < kD; d += 32) acc += qrow[d] * brow[d];
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  return __shfl_sync(0xffffffffu, acc, 0);  // broadcast lane-0 result
}

// Compute all C scores for the queries this CTA owns (grid-stride over Q).
__device__ __forceinline__ void score_query_range(const Problem& p) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  for (int q = blockIdx.x; q < p.Q; q += gridDim.x) {
    const float* qrow = p.q + (size_t)q * kD;
    for (int c = warp; c < p.C; c += kWarps) {
      const float* brow = p.blocks + (size_t)c * kD;
      float s = warp_dot_qd(qrow, brow);
      if (lane == 0) p.scores[(size_t)q * p.C + c] = s;
    }
  }
}

// ---------------------------------------------------------------------------
// Region 1b — DEVICE TOP-K SELECTION (exact, deterministic tiebreak).
//
// One CTA owns one query; it reads that query's C scores from global memory into
// a per-block running view and extracts the top-K in descending order:
//
//   for j in [0,K):
//     blockwide argmax over (score, index) with key = (score, -index) so the
//     LOWER index wins ties (deterministic); record (idx,score); mark chosen.
//
// "Mark chosen" = set that score to -inf in a SMEM scratch copy so it is not
// re-picked. To avoid an O(C) SMEM array for large C, the chosen scores are
// evicted by writing -inf back over the per-warp-local reduction each round; we
// keep a compact SMEM bitmap-free approach: each round does a full blockwide
// argmax over global scores while skipping already-chosen indices via a SMEM
// "min acceptable" is NOT valid under duplicates, so we use an explicit chosen
// test against the running selected set in SMEM (K small enough: K<=2048).
//
// Implementation chosen for correctness + simplicity: copy this query's C scores
// into SMEM once (C<=4096 floats = 16KB), then K rounds of blockwide argmax that
// overwrite the winner's SMEM slot with -inf. Tiebreak by index encoded into the
// argmax compare. O(K*C) but C,K modest and it is exact + matches CPU exactly.
// ---------------------------------------------------------------------------

// Blockwide argmax over s_scores[0..C) returning the winning (value,index) with
// the deterministic tiebreak (higher value; lower index on tie). Uses a SMEM
// reduction across warps. All threads participate; result broadcast via SMEM.
struct ArgMax { float val; int idx; };

__device__ __forceinline__ ArgMax block_argmax(const float* s_scores, int C,
                                               ArgMax* s_warp /*[kWarps]*/) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  ArgMax best{-FLT_MAX, INT_MAX};
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    float v = s_scores[c];
    // higher value wins; on exact tie, lower index wins.
    if (v > best.val || (v == best.val && c < best.idx)) {
      best.val = v;
      best.idx = c;
    }
  }
  // warp reduce
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    float ov = __shfl_down_sync(0xffffffffu, best.val, off);
    int oi = __shfl_down_sync(0xffffffffu, best.idx, off);
    if (ov > best.val || (ov == best.val && oi < best.idx)) {
      best.val = ov;
      best.idx = oi;
    }
  }
  if (lane == 0) s_warp[warp] = best;
  __syncthreads();
  // first warp reduces the per-warp winners
  ArgMax g{-FLT_MAX, INT_MAX};
  if (warp == 0) {
    ArgMax w = (lane < kWarps) ? s_warp[lane] : ArgMax{-FLT_MAX, INT_MAX};
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      float ov = __shfl_down_sync(0xffffffffu, w.val, off);
      int oi = __shfl_down_sync(0xffffffffu, w.idx, off);
      if (ov > w.val || (ov == w.val && oi < w.idx)) {
        w.val = ov;
        w.idx = oi;
      }
    }
    if (lane == 0) {
      s_warp[0] = w;  // reuse slot 0 to broadcast
    }
    g = w;
  }
  __syncthreads();
  g = s_warp[0];
  __syncthreads();
  return g;
}

// Select top-K for the queries this CTA owns (grid-stride over Q). Indices in
// descending score order written to p.sel_idx/p.sel_score. SMEM holds this
// query's C scores (caller provides s_scores sized >= C) + per-warp argmax slots.
__device__ __forceinline__ void select_topk_query_range(const Problem& p,
                                                        float* s_scores,
                                                        ArgMax* s_warp) {
  for (int q = blockIdx.x; q < p.Q; q += gridDim.x) {
    const float* g_scores = p.scores + (size_t)q * p.C;
    for (int c = threadIdx.x; c < p.C; c += blockDim.x) s_scores[c] = g_scores[c];
    __syncthreads();
    for (int j = 0; j < p.K; ++j) {
      ArgMax w = block_argmax(s_scores, p.C, s_warp);
      if (threadIdx.x == 0) {
        p.sel_idx[(size_t)q * p.K + j] = w.idx;
        p.sel_score[(size_t)q * p.K + j] = w.val;
      }
      __syncthreads();
      if (threadIdx.x == 0) s_scores[w.idx] = -FLT_MAX;  // evict the winner
      __syncthreads();
    }
  }
}

// ---------------------------------------------------------------------------
// Region 2 — DATA-DEPENDENT GATHER + REDUCE (stand-in for attention over the
// selected KV). out[q,d] = sum_{j in selected} w_j * blocks[sel_idx[q,j], d],
// with w_j a softmax-like weight over the selected scores (numerically stable).
// Which blocks are read is decided by region 1 ON THE DEVICE.
// One CTA per query (grid-stride). Pass 1: blockwide max + sum of exp over the K
// selected scores (SMEM). Pass 2: accumulate weighted D-vectors.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void gather_reduce_query_range(const Problem& p,
                                                          float* s_red /*[kWarps]*/,
                                                          float* s_acc /*[kD]*/) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  for (int q = blockIdx.x; q < p.Q; q += gridDim.x) {
    const int* idxrow = p.sel_idx + (size_t)q * p.K;
    const float* scorerow = p.sel_score + (size_t)q * p.K;

    // --- pass 1a: max of selected scores (blockwide) ---
    float m = -FLT_MAX;
    for (int j = threadIdx.x; j < p.K; j += blockDim.x)
      m = fmaxf(m, scorerow[j]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
    if (lane == 0) s_red[warp] = m;
    __syncthreads();
    if (warp == 0) {
      float v = (lane < kWarps) ? s_red[lane] : -FLT_MAX;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
      if (lane == 0) s_red[0] = v;
    }
    __syncthreads();
    const float gmax = s_red[0];
    __syncthreads();

    // --- pass 1b: sum of exp(score - max) (blockwide) ---
    float ssum = 0.0f;
    for (int j = threadIdx.x; j < p.K; j += blockDim.x)
      ssum += __expf(scorerow[j] - gmax);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      ssum += __shfl_down_sync(0xffffffffu, ssum, off);
    if (lane == 0) s_red[warp] = ssum;
    __syncthreads();
    if (warp == 0) {
      float v = (lane < kWarps) ? s_red[lane] : 0.0f;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
      if (lane == 0) s_red[0] = v;
    }
    __syncthreads();
    const float denom = s_red[0];
    const float inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    __syncthreads();

    // --- pass 2: accumulate weighted D-vectors into SMEM, then store ---
    for (int d = threadIdx.x; d < kD; d += blockDim.x) s_acc[d] = 0.0f;
    __syncthreads();
    // Each warp handles a stripe of the K selected blocks; atomically adds its
    // weighted contribution to the shared D-accumulator. (D=128, K up to 2048;
    // SMEM atomics on 128 floats are cheap relative to the global reads.)
    for (int j = warp; j < p.K; j += kWarps) {
      const float w = __expf(scorerow[j] - gmax) * inv_denom;
      const int bidx = idxrow[j];
      const float* brow = p.blocks + (size_t)bidx * kD;
#pragma unroll
      for (int d = lane; d < kD; d += 32)
        atomicAdd(&s_acc[d], w * brow[d]);
    }
    __syncthreads();
    for (int d = threadIdx.x; d < kD; d += blockDim.x)
      p.out[(size_t)q * kD + d] = s_acc[d];
    __syncthreads();
  }
}

// ===========================================================================
// (A) DEVICE-RESIDENT persistent cooperative kernel: region1 -> barrier ->
//     region2. The selection (sel_idx/sel_score) lives only in device memory.
// ===========================================================================
__global__ void __launch_bounds__(kBlockThreads)
    kDeviceResident(Problem p) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ unsigned char smem[];
  // layout: [C floats scores][kWarps ArgMax][kWarps floats red][kD floats acc]
  float* s_scores = reinterpret_cast<float*>(smem);
  ArgMax* s_warp =
      reinterpret_cast<ArgMax*>(s_scores + p.C);
  float* s_red = reinterpret_cast<float*>(s_warp + kWarps);
  float* s_acc = s_red + kWarps;

  score_query_range(p);
  cg_grid_barrier(grid);                      // scores globally visible
  select_topk_query_range(p, s_scores, s_warp);
  cg_grid_barrier(grid);                      // sel_idx/sel_score visible
  gather_reduce_query_range(p, s_red, s_acc);
}

inline size_t device_resident_smem_bytes(int C) {
  return (size_t)C * sizeof(float) + (size_t)kWarps * sizeof(ArgMax) +
         (size_t)kWarps * sizeof(float) + (size_t)kD * sizeof(float);
}

// ===========================================================================
// (B) HOST-ORCHESTRATED baseline kernels (today's pattern): three separate
//     kernels with d2h scores + host top-K + h2d indices between them.
// ===========================================================================

// B1: score only (non-cooperative, grid-stride over queries).
__global__ void __launch_bounds__(kBlockThreads) kScoreOnly(Problem p) {
  score_query_range(p);
}

// B3: gather+reduce only. Reads sel_idx/sel_score that the HOST copied in (h2d).
__global__ void __launch_bounds__(kBlockThreads) kGatherOnly(Problem p) {
  extern __shared__ unsigned char smem[];
  float* s_red = reinterpret_cast<float*>(smem);
  float* s_acc = s_red + kWarps;
  gather_reduce_query_range(p, s_red, s_acc);
}

inline size_t gather_only_smem_bytes() {
  return (size_t)kWarps * sizeof(float) + (size_t)kD * sizeof(float);
}

// ===========================================================================
// (C) DEVICE-SELECT-ONLY kernel — used so the host baseline can reuse the SAME
//     device top-K when ONLY the host orchestration (the d2h/h2d/sync) is what
//     we want to isolate vs an honest host-side top-K. Not part of (A)/(B); a
//     control to separate "device top-k algo" from "control-flow boundary".
//     Persistent so it can also report selection-kernel occupancy.
// ===========================================================================
__global__ void __launch_bounds__(kBlockThreads) kSelectOnlyPersistent(Problem p) {
  extern __shared__ unsigned char smem[];
  float* s_scores = reinterpret_cast<float*>(smem);
  ArgMax* s_warp = reinterpret_cast<ArgMax*>(s_scores + p.C);
  select_topk_query_range(p, s_scores, s_warp);
}

inline size_t select_only_smem_bytes(int C) {
  return (size_t)C * sizeof(float) + (size_t)kWarps * sizeof(ArgMax);
}

}  // namespace g3
}  // namespace pde

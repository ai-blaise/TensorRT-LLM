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
// Today that selection round-trips host orchestration. G3 proved it can stay
// device-resident in ONE persistent cooperative grid, dissolving the d2h wall, and
// quantified the host-round-trip latency removed.
//
// ============================ G3-OPT (this file) ============================
// G3 proved device-residency WINS at batch (5.47x @M32 small-k) but LOST at M=1
// (0.54x small-k, 0.27x large-k) because the device top-k was a simple O(K*C)
// blockwide argmax-and-evict (K rounds x C elements) that dominates at large K,
// plus 1 query = 1 CTA = 147 idle SMs. G3-OPT replaces that with:
//
//   (1) An EXACT O(C) device top-k via 64-bit COMPOSITE-KEY MSD RADIX-SELECT.
//       Each score is mapped to a monotone uint32 key (larger float -> larger
//       key), then packed with the (bit-inverted) candidate index into a 64-bit
//       composite  key64 = (score_key << 32) | (~index & 0xffffffff).
//       Descending order on key64 == (score DESC, then index ASC) — EXACTLY the
//       CPU reference's deterministic tiebreak — and because the index is unique,
//       ALL composites are DISTINCT, so the K-th largest is unique and the
//       selected set is { key64 >= threshold }: no boundary/tie special-casing,
//       provably index-set-identical to the CPU. Cost: 8 radix passes (8-bit
//       digits over 64 bits) x C, i.e. O(C) vs the old O(K*C) (~512x less inner
//       work at K=2048,C=4096).
//
//   (2) MULTI-CTA-PER-QUERY parallelism so M=1 is not 1-CTA-bound. A "query
//       group" of G CTAs cooperates on ONE query's score + radix-select via a
//       per-query 256-bin GLOBAL histogram (atomically accumulated across the
//       group), with grid.sync() between radix passes. At M=1 the whole device
//       works the single query (all SMs busy) instead of 1 CTA / 147 idle.
//
// Everything else is unchanged: the persistent-grid score -> grid.sync -> top-k
// -> grid.sync -> gather structure, and the host-orchestrated baseline (B). The
// gather (region 2) is still the softmax-weighted D-vector sum stand-in.
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

// Radix configuration: 8-bit digits over a 64-bit composite key = 8 passes.
constexpr int kRadixBits = 8;
constexpr int kRadixBins = 1 << kRadixBits;   // 256
constexpr int kRadixPasses = 64 / kRadixBits; // 8

// Problem descriptor shared by every kernel + the CPU reference.
//
// G3-OPT adds GLOBAL scratch needed by the multi-CTA radix-select:
//   hist     [Q * kRadixBins]  per-query digit histogram (reset each pass)
//   thr_key  [Q]               per-query threshold composite key64
//   out_cnt  [Q]               per-query atomic output fill counter
// These are pure device scratch — never copied to host; the selection stays
// device-resident exactly as in G3.
struct Problem {
  const float* q;        // [Q, D]   query vectors (row-major)
  const float* blocks;   // [C, D]   candidate block vectors (row-major)
  float* scores;         // [Q, C]   region-1 output: score[q,c]
  int* sel_idx;          // [Q, K]   selected block indices (descending score)
  float* sel_score;      // [Q, K]   the selected scores (for the weighted reduce)
  float* out;            // [Q, D]   region-2 output: weighted sum of selected
  // --- G3-OPT global scratch ---
  unsigned int* hist;            // [Q, kRadixBins]
  unsigned long long* thr_key;   // [Q]  threshold composite key
  unsigned int* out_cnt;         // [Q]  atomic output position counter
  int Q;
  int C;
  int K;
  int ctas_per_query;    // G: CTAs cooperating on one query (group size)
};

// ---------------------------------------------------------------------------
// Composite-key helpers.
//
// Monotone float->uint32: for finite x, key(a) > key(b) <=> a > b. Flip sign bit
// for positives, invert all bits for negatives (standard radix float ordering).
// Then pack with ~index so descending key64 == (score DESC, index ASC), and all
// key64 are DISTINCT (index unique) => exact, tie-free top-K.
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned int float_to_okey(float f) {
  unsigned int u = __float_as_uint(f);
  unsigned int mask = (unsigned int)(-(int)(u >> 31)) | 0x80000000u;
  return u ^ mask;
}
__device__ __forceinline__ float okey_to_float(unsigned int k) {
  unsigned int mask = ((k >> 31) - 1u) | 0x80000000u;
  return __uint_as_float(k ^ mask);
}
__device__ __forceinline__ unsigned long long make_key64(float score, int idx) {
  unsigned int sk = float_to_okey(score);
  unsigned int ik = ~(unsigned int)idx;          // smaller idx -> larger ik
  return ((unsigned long long)sk << 32) | (unsigned long long)ik;
}
__device__ __forceinline__ int key64_index(unsigned long long k) {
  return (int)(~(unsigned int)(k & 0xffffffffu));
}

// ---------------------------------------------------------------------------
// Region 1a — SCORE. score[q,c] = sum_d q[q,d] * blocks[c,d].
// Multi-CTA-per-query: a query group of G CTAs splits the candidate range; each
// warp in each CTA strides its slice, each thread reduces D with a strided load,
// then a warp shuffle finishes the dot. D=128 = 4 floats/lane across a warp.
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

// Score the candidates owned by THIS CTA for its query (grid-stride over query
// groups so #queries may exceed #groups resident). Each group of gsize CTAs
// partitions [0,C); within a CTA, warps stride candidates.
__device__ __forceinline__ void score_query_range(const Problem& p) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int gsize = p.ctas_per_query;
  const int n_groups = gridDim.x / gsize;
  const int my_group = blockIdx.x / gsize;
  const int my_rank = blockIdx.x % gsize;
  for (int q = my_group; q < p.Q; q += n_groups) {
    const float* qrow = p.q + (size_t)q * kD;
    for (int c = my_rank * kWarps + warp; c < p.C; c += gsize * kWarps) {
      const float* brow = p.blocks + (size_t)c * kD;
      float s = warp_dot_qd(qrow, brow);
      if (lane == 0) p.scores[(size_t)q * p.C + c] = s;
    }
  }
}

// ---------------------------------------------------------------------------
// Region 1b — DEVICE TOP-K SELECTION (exact, deterministic, O(C) radix-select).
//
// 64-bit composite-key MSD radix-select cooperating across a query group:
//
//   threshold prefix starts empty; for pass t = 0..7 (digit = bits [56-8t..]):
//     1) each CTA in the group histograms its slice's digit over the ACTIVE set
//        (active iff key64's already-fixed high bits == the running prefix) into
//        a per-query GLOBAL 256-bin histogram via atomicAdd.
//     2) grid.sync(): histogram complete + visible to the whole group.
//     3) EVERY CTA independently walks the 256 bins high->low; the bin where the
//        cumulative count first reaches the (reduced) K is the split bin -> append
//        its digit to the prefix, reduce K by the strictly-higher bins' counts.
//        (Deterministic: all CTAs compute the identical prefix.) Reset histogram.
//     4) grid.sync(): histogram reset visible before the next pass's atomics.
//
//   After 8 passes the prefix IS the threshold composite key64 (unique, since
//   composites are distinct). The selected set = { key64 >= threshold } = EXACTLY
//   the top-K with (score DESC, index ASC): no boundary/tie special-casing. Each
//   CTA scans its slice and appends qualifying (idx,score) via an atomic fill
//   counter. Exactly K elements qualify. Cost ~ 8*C/gsize per CTA + 16 barriers.
//
// WAVE-LOCKED: every resident CTA must issue the SAME number of grid.sync()s. We
// pass an `active` flag; inactive CTAs (groups with no query this wave) still hit
// every barrier but contribute zero histogram and skip the threshold write.
// ---------------------------------------------------------------------------

// One radix-select for ONE query across its group, wave-locked. Writes the
// threshold key64 to p.thr_key[q] (all active CTAs of the group compute the same)
// and emits the selected (idx,score). `active==false` => skip all writes/atomics
// but STILL execute every grid.sync() to keep the cooperative grid in lockstep.
__device__ __forceinline__ void radix_select_wavelocked(
    cg::grid_group& grid, const Problem& p, int q, int my_rank, int gsize,
    unsigned int* s_hist, bool active) {
  unsigned long long prefix = 0ull;
  unsigned long long prefix_mask = 0ull;
  int k_remain = p.K;
  const float* g_scores = active ? (p.scores + (size_t)q * p.C) : nullptr;
  unsigned int* g_hist = active ? (p.hist + (size_t)q * kRadixBins) : nullptr;

  // Reset this query's output fill counter once at the start (rank0/thread0).
  // Global histogram is left all-zero by the previous invocation's per-pass reset
  // (including the last pass), so it needs no host memset between timing reps; the
  // out_cnt does, hence this reset. 16 grid.syncs separate it from the emit's
  // atomicAdd below, so the zero is globally visible before emit reads it.
  if (active && my_rank == 0 && threadIdx.x == 0) p.out_cnt[q] = 0u;

  for (int t = 0; t < kRadixPasses; ++t) {
    const int shift = 64 - kRadixBits * (t + 1);
    // --- build SMEM histogram over the active set, then flush to global ---
    for (int b = threadIdx.x; b < kRadixBins; b += blockDim.x) s_hist[b] = 0u;
    __syncthreads();
    if (active) {
      for (int c = my_rank * blockDim.x + threadIdx.x; c < p.C;
           c += gsize * blockDim.x) {
        unsigned long long k = make_key64(g_scores[c], c);
        if ((k & prefix_mask) == prefix) {
          unsigned int d = (unsigned int)((k >> shift) & (kRadixBins - 1));
          atomicAdd(&s_hist[d], 1u);
        }
      }
      __syncthreads();
      for (int b = threadIdx.x; b < kRadixBins; b += blockDim.x) {
        unsigned int v = s_hist[b];
        if (v) atomicAdd(&g_hist[b], v);
      }
    }
    grid.sync();  // (A) global histogram complete + visible to the whole group

    // --- every active CTA copies the global histogram into its OWN SMEM and
    //     walks it identically (read-only on global) to find the split bin. The
    //     SMEM copy decouples the walk from rank0's reset below so they cannot
    //     race even within the group. ---
    if (active) {
      for (int b = threadIdx.x; b < kRadixBins; b += blockDim.x)
        s_hist[b] = g_hist[b];
      __syncthreads();
      int acc = 0, digit = 0;
      for (int b = kRadixBins - 1; b >= 0; --b) {
        int cnt = (int)s_hist[b];
        if (acc + cnt >= k_remain) { digit = b; break; }
        acc += cnt;
      }
      k_remain -= acc;
      prefix |= ((unsigned long long)digit) << shift;
      prefix_mask |= ((unsigned long long)(kRadixBins - 1)) << shift;
    }
    grid.sync();  // (B) all walks (global reads) done before any reset writes

    // --- reset this query's global histogram for the next pass (rank0 only);
    //     after (B) no peer is still reading it. ---
    if (active && my_rank == 0) {
      for (int b = threadIdx.x; b < kRadixBins; b += blockDim.x) g_hist[b] = 0u;
    }
    grid.sync();  // (C) reset visible before next pass's atomics
  }

  // prefix now == the K-th largest composite key (the threshold). Emit.
  if (active) {
    if (threadIdx.x == 0 && my_rank == 0) p.thr_key[q] = prefix;
    const unsigned long long thr = prefix;
    unsigned int* g_cnt = p.out_cnt + q;
    int* idx_out = p.sel_idx + (size_t)q * p.K;
    float* sc_out = p.sel_score + (size_t)q * p.K;
    for (int c = my_rank * blockDim.x + threadIdx.x; c < p.C;
         c += gsize * blockDim.x) {
      float s = g_scores[c];
      unsigned long long k = make_key64(s, c);
      if (k >= thr) {
        unsigned int pos = atomicAdd(g_cnt, 1u);
        if (pos < (unsigned int)p.K) {   // exactly K qualify; defensive guard
          idx_out[pos] = c;
          sc_out[pos] = s;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Region 2 — DATA-DEPENDENT GATHER + REDUCE (stand-in for attention over the
// selected KV). out[q,d] = sum_{j in selected} w_j * blocks[sel_idx[q,j], d],
// with w_j a softmax-like weight over the selected scores (numerically stable).
// Which blocks are read is decided by region 1 ON THE DEVICE. The group leader
// (rank 0) performs the (BW-light, K<=2048) reduce — bit-identical to G3's
// single-CTA gather. The expensive O(C) parts (score, radix-select) are what we
// parallelize across the group; the gather is not where the M=1 win lives.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void gather_reduce_single(const Problem& p, int q,
                                                      float* s_red, float* s_acc) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int* idxrow = p.sel_idx + (size_t)q * p.K;
  const float* scorerow = p.sel_score + (size_t)q * p.K;

  float m = -FLT_MAX;
  for (int j = threadIdx.x; j < p.K; j += blockDim.x) m = fmaxf(m, scorerow[j]);
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

  for (int d = threadIdx.x; d < kD; d += blockDim.x) s_acc[d] = 0.0f;
  __syncthreads();
  for (int j = warp; j < p.K; j += kWarps) {
    const float w = __expf(scorerow[j] - gmax) * inv_denom;
    const int bidx = idxrow[j];
    const float* brow = p.blocks + (size_t)bidx * kD;
#pragma unroll
    for (int d = lane; d < kD; d += 32) atomicAdd(&s_acc[d], w * brow[d]);
  }
  __syncthreads();
  for (int d = threadIdx.x; d < kD; d += blockDim.x)
    p.out[(size_t)q * kD + d] = s_acc[d];
  __syncthreads();
}

// Grouped gather for the device-resident path: rank-0 of each group reduces.
__device__ __forceinline__ void gather_reduce_query_range(const Problem& p,
                                                          float* s_red,
                                                          float* s_acc) {
  const int gsize = p.ctas_per_query;
  const int n_groups = gridDim.x / gsize;
  const int my_group = blockIdx.x / gsize;
  const int my_rank = blockIdx.x % gsize;
  if (my_rank != 0) return;
  for (int q = my_group; q < p.Q; q += n_groups)
    gather_reduce_single(p, q, s_red, s_acc);
}

// Plain (non-grouped) gather for the HOST baseline's standalone gather kernel.
__device__ __forceinline__ void gather_reduce_plain(const Problem& p,
                                                     float* s_red, float* s_acc) {
  for (int q = blockIdx.x; q < p.Q; q += gridDim.x)
    gather_reduce_single(p, q, s_red, s_acc);
}

// ===========================================================================
// (A) DEVICE-RESIDENT persistent cooperative kernel: region1 -> barrier ->
//     radix top-k -> barrier -> region2. The selection lives only in device mem.
//
// Cooperative-grid invariant: every resident CTA executes the SAME sequence of
// grid.sync()s. The radix-select does kRadixPasses*2 syncs PER WAVE; we loop a
// FIXED #waves = ceil(Q / n_groups) for ALL CTAs (groups without a query in a
// wave still hit every barrier, gated by `active`). The bench sizes n_groups>=Q
// (one wave) for M<=32.
// ===========================================================================
__global__ void __launch_bounds__(kBlockThreads)
    kDeviceResident(Problem p) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ unsigned char smem[];
  // layout: [256 u32 hist][kWarps floats red][kD floats acc]
  unsigned int* s_hist = reinterpret_cast<unsigned int*>(smem);
  float* s_red = reinterpret_cast<float*>(s_hist + kRadixBins);
  float* s_acc = s_red + kWarps;

  const int gsize = p.ctas_per_query;
  const int n_groups = gridDim.x / gsize;
  const int my_group = blockIdx.x / gsize;
  const int my_rank = blockIdx.x % gsize;
  const int waves = (p.Q + n_groups - 1) / n_groups;

  score_query_range(p);
  grid.sync();  // all scores globally visible

  for (int w = 0; w < waves; ++w) {
    const int q = my_group + w * n_groups;
    const bool active = (q < p.Q);
    radix_select_wavelocked(grid, p, active ? q : 0, my_rank, gsize, s_hist,
                            active);
  }
  grid.sync();  // sel_idx/sel_score (+ thresholds) globally visible

  gather_reduce_query_range(p, s_red, s_acc);
}

inline size_t device_resident_smem_bytes(int /*C*/) {
  // No longer scales with C (radix needs only a 256-bin SMEM histogram), so the
  // device-resident kernel's SMEM is tiny + constant -> higher occupancy.
  return (size_t)kRadixBins * sizeof(unsigned int) +
         (size_t)kWarps * sizeof(float) + (size_t)kD * sizeof(float);
}

// ===========================================================================
// (B) HOST-ORCHESTRATED baseline kernels (today's pattern): three separate
//     kernels with d2h scores + host top-K + h2d indices between them.
// ===========================================================================

// B1: score only — plain one-CTA-per-query grid-stride (original G3 behavior).
__global__ void __launch_bounds__(kBlockThreads) kScoreOnly(Problem p) {
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

// B3: gather+reduce only. Reads sel_idx/sel_score that the HOST copied in (h2d).
__global__ void __launch_bounds__(kBlockThreads) kGatherOnly(Problem p) {
  extern __shared__ unsigned char smem[];
  float* s_red = reinterpret_cast<float*>(smem);
  float* s_acc = s_red + kWarps;
  gather_reduce_plain(p, s_red, s_acc);
}

inline size_t gather_only_smem_bytes() {
  return (size_t)kWarps * sizeof(float) + (size_t)kD * sizeof(float);
}

// ===========================================================================
// (C) DEVICE-SELECT-ONLY persistent kernel — isolates the device top-k cost
//     (radix-select) alone, so we can report the new top-k kernel time vs the
//     old O(K*C) argmax. Cooperative; same wave-locked barrier discipline.
// ===========================================================================
__global__ void __launch_bounds__(kBlockThreads) kSelectOnlyPersistent(Problem p) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ unsigned char smem[];
  unsigned int* s_hist = reinterpret_cast<unsigned int*>(smem);

  const int gsize = p.ctas_per_query;
  const int n_groups = gridDim.x / gsize;
  const int my_group = blockIdx.x / gsize;
  const int my_rank = blockIdx.x % gsize;
  const int waves = (p.Q + n_groups - 1) / n_groups;
  for (int w = 0; w < waves; ++w) {
    const int q = my_group + w * n_groups;
    const bool active = (q < p.Q);
    radix_select_wavelocked(grid, p, active ? q : 0, my_rank, gsize, s_hist,
                            active);
  }
}

inline size_t select_only_smem_bytes(int /*C*/) {
  return (size_t)kRadixBins * sizeof(unsigned int);
}

}  // namespace g3
}  // namespace pde

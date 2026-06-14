// PDE G4 — HETEROGENEOUS-WORKER copy ∥ compute overlap (standalone, sm_100).
//
// G4 proves the second half of the persistent-decode thesis (after G3's
// device-resident control flow): a LATENCY-bound copy can be HIDDEN under
// compute by partitioning the resident cooperative grid into two heterogeneous
// worker groups that run CONCURRENTLY (TileRT "Heterogeneous Workers").
//
// The decode workload this models: the HiSparse decode path swaps cold KV blocks
// into a hot device buffer every step (~212-489 blocks/step, packed kvarn_k2v2
// block ~208 B/tok). With SWAP-AHEAD semantics, this-step attention reads only
// blocks that are ALREADY hot, while the copy group stages NEXT-step's cold
// blocks into a separate hot region. The copy is therefore off the critical path
// and should hide under the attention compute.
//
//   COPY group   (CTAs [0, n_copy)):    stage N_swap cold KV blocks
//                                        (cold device buffer OR host-pinned cold)
//                                        -> a hot device buffer, vectorized /
//                                        cp.async-style. Writes hot_staged[].
//   COMPUTE group(CTAs [n_copy, grid)):  attention-like softmax-weighted reduction
//                                        over M_hot resident hot blocks (stand-in
//                                        for the FlashMLA hot-read). Writes out[].
//
//   (A) OVERLAPPED : ONE persistent cooperative kernel. COPY and COMPUTE groups
//       run at the SAME TIME (swap-ahead: compute does NOT depend on this step's
//       staged blocks). A single grid barrier at the END joins the two groups.
//   (B) SERIAL     : copy ALL blocks -> grid barrier -> compute ALL. (Today's
//       implicit ordering when the swap-in is on the critical path.)
//
// Correctness: (A) hot_staged + out == (B) == an INDEPENDENT CPU reference
// (byte-identical / cos >= 0.999999). Both the STAGED bytes and the REDUCTION
// output are checked — the overlap must not corrupt either.
//
// Overlap magnitude: copy-only, compute-only, serial(=their sum + a join),
// overlapped latency; overlapped should approach max(copy,compute), not the sum.
// hidden_fraction = (serial - overlapped) / min(copy, compute)  in [0,1].
//
// Standalone nvcc -arch=sm_100. Does NOT touch the decode/model path; the
// ABI-frozen hot-read op/header are untouched.
#pragma once

#include "pde_substrate.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cfloat>

namespace pde {
namespace g4 {

namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// Geometry. A "block" is one KV slot to stage / reduce. We use D bf16 elements
// per block; D=128 bf16 = 256 B/block matches the kvarn_k2v2 ~208-512 B/tok
// regime. The COMPUTE reduction uses the SAME element type so a single hot
// buffer layout serves both (staged region + resident hot region).
// ---------------------------------------------------------------------------
constexpr int kD = 128;                 // elements per KV block (bf16) -> 256 B
constexpr int kBlockThreads = 256;
constexpr int kWarps = kBlockThreads / 32;

using elem_t = __nv_bfloat16;

__device__ __forceinline__ float to_f(elem_t x) { return __bfloat162float(x); }
__device__ __forceinline__ elem_t to_e(float x) { return __float2bfloat16(x); }

// Host-side bf16 converter for filling reference buffers (the device intrinsics
// above are __device__-only). Plain round-to-nearest via the bf16 ctor, which is
// host-callable in CUDA's <cuda_bf16.h>.
__host__ inline elem_t to_e_host(float x) { return __nv_bfloat16(x); }

// Problem descriptor. Shared by the kernels and the CPU reference.
//
//   cold_dev   : [N_swap, D]  cold KV blocks resident in DEVICE global memory.
//   cold_pin   : [N_swap, D]  the SAME cold blocks in HOST-PINNED memory (the
//                             real swap-in source on a capacity miss); copied
//                             over PCIe/NVLink C2C. May be nullptr (device path).
//   hot_staged : [N_swap, D]  destination hot buffer the COPY group fills.
//   hot_resident:[M_hot, D]   already-hot KV the COMPUTE group reduces over.
//   q          : [D]          single query vector for the attention stand-in
//                             (decode = one query/step per (batch,head); we model
//                             one head's hot-read; M scaled via M_hot).
//   sims       : [M_hot]      scratch: per-block score q·hot (region buffer).
//   out        : [D]          COMPUTE output: softmax(q·hot)-weighted sum of hot.
//
// use_pinned selects the COPY source (cold_pin when true, else cold_dev).
struct Problem {
  const elem_t* cold_dev;
  const elem_t* cold_pin;
  elem_t* hot_staged;
  const elem_t* hot_resident;
  const float* q;
  float* sims;
  float* out;
  int n_swap;     // blocks to stage
  int m_hot;      // resident hot blocks to reduce
  int n_copy;     // # CTAs in the COPY group (rest are COMPUTE)
  int use_pinned; // 0 = cold_dev source, 1 = cold_pin source
};

// ---------------------------------------------------------------------------
// COPY group body. CTAs [0, n_copy) cooperatively stage all N_swap*D elements
// from the chosen cold source into hot_staged. Vectorized as float4 (8 bf16 per
// transaction) over a flat element grid, strided across the COPY CTAs' threads.
// This is the "cp.async / vectorized DMA" stand-in; on SM100 the L2/TMA path
// makes a wide coalesced float4 copy the right baseline for a staging memcpy.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void copy_group_body(const Problem& p) {
  const elem_t* src = p.use_pinned ? p.cold_pin : p.cold_dev;
  // total bf16 elements; copy 8 at a time (float4 = 16 B).
  const size_t total = (size_t)p.n_swap * kD;
  const size_t vec = total / 8;                  // # float4 chunks (D=128 -> exact)
  const float4* s4 = reinterpret_cast<const float4*>(src);
  float4* d4 = reinterpret_cast<float4*>(p.hot_staged);
  // Flatten COPY CTAs into one cooperative pool.
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t nthreads = (size_t)p.n_copy * blockDim.x;
  for (size_t i = tid; i < vec; i += nthreads) d4[i] = s4[i];
  // tail (only if D*N not divisible by 8 — here it always is, kept for safety).
  for (size_t i = vec * 8 + tid; i < total; i += nthreads)
    p.hot_staged[i] = src[i];
}

// ---------------------------------------------------------------------------
// COMPUTE group body. CTAs [n_copy, grid) cooperatively do a softmax-weighted
// reduction over the M_hot resident hot blocks (attention hot-read stand-in):
//   score[b] = q · hot_resident[b]      (b in [0, M_hot))
//   w[b]     = softmax(score)[b]
//   out[d]   = sum_b w[b] * hot_resident[b][d]
//
// Online-softmax over a GRID of CTAs is awkward; instead we split the work in
// two device passes joined by reductions in GLOBAL memory so the math is exact
// and matches the CPU reference: pass-1 each COMPUTE CTA writes its blocks'
// scores to p.sims; a grid barrier; pass-2 the FIRST compute CTA reduces sims
// (max + sum-exp) and accumulates the weighted D-vector. M_hot is modest
// (<= a few thousand) so the single-CTA epilogue is fine and keeps the result
// bit-stable vs the CPU. The point of G4 is the COPY∥COMPUTE overlap, not a
// distributed-softmax algorithm; this keeps correctness unambiguous.
//
// `compute_rank` = blockIdx.x - n_copy (0-based within the COMPUTE group);
// `n_compute`    = gridDim.x - n_copy.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void compute_scores(const Problem& p,
                                               int compute_rank,
                                               int n_compute) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  // one warp owns one hot block; grid-stride over M_hot across COMPUTE CTAs.
  const int warps_per = n_compute * kWarps;
  const int my_warp = compute_rank * kWarps + warp;
  for (int b = my_warp; b < p.m_hot; b += warps_per) {
    const elem_t* hb = p.hot_resident + (size_t)b * kD;
    float acc = 0.0f;
#pragma unroll
    for (int d = lane; d < kD; d += 32) acc += p.q[d] * to_f(hb[d]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) p.sims[b] = acc;
  }
}

// Single-CTA softmax epilogue over p.sims[0..M_hot) -> weighted sum into out.
// Runs on COMPUTE rank 0 only. SMEM: [kWarps] reduction + [kD] accumulator.
__device__ __forceinline__ void compute_epilogue(const Problem& p,
                                                 float* s_red /*[kWarps]*/,
                                                 float* s_acc /*[kD]*/) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  // pass 1a: max of scores
  float m = -FLT_MAX;
  for (int b = threadIdx.x; b < p.m_hot; b += blockDim.x) m = fmaxf(m, p.sims[b]);
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
  // pass 1b: sum exp(score - max)
  float ssum = 0.0f;
  for (int b = threadIdx.x; b < p.m_hot; b += blockDim.x)
    ssum += __expf(p.sims[b] - gmax);
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
  const float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
  __syncthreads();
  // pass 2: weighted accumulate into SMEM, then store
  for (int d = threadIdx.x; d < kD; d += blockDim.x) s_acc[d] = 0.0f;
  __syncthreads();
  for (int b = warp; b < p.m_hot; b += kWarps) {
    const float w = __expf(p.sims[b] - gmax) * inv;
    const elem_t* hb = p.hot_resident + (size_t)b * kD;
#pragma unroll
    for (int d = lane; d < kD; d += 32)
      atomicAdd(&s_acc[d], w * to_f(hb[d]));
  }
  __syncthreads();
  for (int d = threadIdx.x; d < kD; d += blockDim.x) p.out[d] = s_acc[d];
}

// ===========================================================================
// (A) OVERLAPPED — ONE persistent cooperative kernel. COPY group and COMPUTE
//     group run concurrently; a single grid barrier at the END joins them.
//     Swap-ahead: the COMPUTE reduction reads hot_resident (already hot), which
//     does NOT depend on the copy filling hot_staged — so the two are truly
//     independent and may execute fully overlapped.
// ===========================================================================
__global__ void __launch_bounds__(kBlockThreads) kOverlapped(Problem p) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ unsigned char smem[];
  float* s_red = reinterpret_cast<float*>(smem);
  float* s_acc = s_red + kWarps;

  if (blockIdx.x < p.n_copy) {
    // COPY worker
    copy_group_body(p);
  } else {
    // COMPUTE worker
    const int compute_rank = blockIdx.x - p.n_copy;
    const int n_compute = gridDim.x - p.n_copy;
    compute_scores(p, compute_rank, n_compute);
  }
  // JOIN: both groups complete (staged bytes written AND all scores written).
  cg_grid_barrier(grid);
  // softmax epilogue on COMPUTE rank 0 (needs the full sims[] from all CTAs).
  if (blockIdx.x == p.n_copy) compute_epilogue(p, s_red, s_acc);
}

// ===========================================================================
// (B) SERIAL baseline — copy ALL (whole grid) -> grid barrier -> compute ALL
//     (whole grid) -> grid barrier -> epilogue. The copy is on the critical
//     path: total ≈ copy + compute.
// ===========================================================================
__global__ void __launch_bounds__(kBlockThreads) kSerial(Problem p) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ unsigned char smem[];
  float* s_red = reinterpret_cast<float*>(smem);
  float* s_acc = s_red + kWarps;

  // Phase 1: the WHOLE grid stages the copy (n_copy := gridDim for this phase).
  {
    const elem_t* src = p.use_pinned ? p.cold_pin : p.cold_dev;
    const size_t total = (size_t)p.n_swap * kD;
    const size_t vec = total / 8;
    const float4* s4 = reinterpret_cast<const float4*>(src);
    float4* d4 = reinterpret_cast<float4*>(p.hot_staged);
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nthreads = (size_t)gridDim.x * blockDim.x;
    for (size_t i = tid; i < vec; i += nthreads) d4[i] = s4[i];
    for (size_t i = vec * 8 + tid; i < total; i += nthreads)
      p.hot_staged[i] = src[i];
  }
  cg_grid_barrier(grid);
  // Phase 2: the WHOLE grid computes scores.
  compute_scores(p, blockIdx.x, gridDim.x);
  cg_grid_barrier(grid);
  if (blockIdx.x == 0) compute_epilogue(p, s_red, s_acc);
}

// ===========================================================================
// Isolation kernels: copy-only and compute-only (each over the WHOLE grid),
// so overlapped latency can be compared to max(copy,compute) and to the sum.
// ===========================================================================
__global__ void __launch_bounds__(kBlockThreads) kCopyOnly(Problem p) {
  const elem_t* src = p.use_pinned ? p.cold_pin : p.cold_dev;
  const size_t total = (size_t)p.n_swap * kD;
  const size_t vec = total / 8;
  const float4* s4 = reinterpret_cast<const float4*>(src);
  float4* d4 = reinterpret_cast<float4*>(p.hot_staged);
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t nthreads = (size_t)gridDim.x * blockDim.x;
  for (size_t i = tid; i < vec; i += nthreads) d4[i] = s4[i];
  for (size_t i = vec * 8 + tid; i < total; i += nthreads)
    p.hot_staged[i] = src[i];
}

__global__ void __launch_bounds__(kBlockThreads) kComputeOnly(Problem p) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ unsigned char smem[];
  float* s_red = reinterpret_cast<float*>(smem);
  float* s_acc = s_red + kWarps;
  compute_scores(p, blockIdx.x, gridDim.x);
  cg_grid_barrier(grid);
  if (blockIdx.x == 0) compute_epilogue(p, s_red, s_acc);
}

inline size_t epilogue_smem_bytes() {
  return (size_t)kWarps * sizeof(float) + (size_t)kD * sizeof(float);
}

}  // namespace g4
}  // namespace pde

// PDE G6 — in-kernel tensor-parallel all-reduce for the persistent decode
// megakernel (standalone, sm_100, model-free).
//
// The decode megakernel must perform its TP all-reduce (after attn out-proj and
// after MoE down-proj) WITHOUT exiting to the launch boundary.
//
// IMPL LADDER (see pde_g6_results.md for the measured tier):
//   (a) NVLS `multimem` over a CUDA-driver multicast object — the *best* B200
//       path (fabric reduces in the NVSwitch). Wrappers kept below
//       (`multimem_*`). REQUIRES the NVLink-fabric multicast team to be
//       established, which needs nv-fabricmanager + IMEX channels. On a GPU-
//       passthrough VM without the fabric exposed, `cuMulticastBindMem` returns
//       CUDA_ERROR_INVALID_VALUE / NOT_PERMITTED even though every GPU reports
//       multicast_supported=1 — so this tier is environment-blocked there.
//   (b) P2P-NVLink in-kernel one-shot all-reduce (THIS is what the bench runs
//       when (a) is blocked): every rank reads its peers' unicast buffers
//       directly over NVLink (peer-mapped device pointers after
//       cudaDeviceEnablePeerAccess) and sums in-kernel. Fully in-kernel, fully
//       model-free. `one_shot_all_reduce_f32_body` below.
//
// This file does NOT touch the model/decode path; compiled standalone by the
// G6 bench.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace pde {
namespace g6 {

static constexpr int kMaxRanks = 8;

// ===========================================================================
// (a) NVLS multimem PTX wrappers (tier-a path; used iff a multicast team binds).
// ===========================================================================
__device__ __forceinline__ void multimem_ld_reduce_add_v4f32(
    float4* out, const float4* mc_ptr) {
  asm volatile(
      "multimem.ld_reduce.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"
      : "=f"(out->x), "=f"(out->y), "=f"(out->z), "=f"(out->w)
      : "l"(mc_ptr)
      : "memory");
}
__device__ __forceinline__ void multimem_st_v4f32(float4* mc_ptr,
                                                  const float4& v) {
  asm volatile(
      "multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};"
      :
      : "l"(mc_ptr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
      : "memory");
}

__device__ __forceinline__ void all_reduce_f32_multimem_body(
    float* mc_f32, int n_f32, int comms_tid, int comms_nthreads) {
  const int n_vec = n_f32 / 4;
  float4* mc_v = reinterpret_cast<float4*>(mc_f32);
  for (int i = comms_tid; i < n_vec; i += comms_nthreads) {
    float4 acc;
    multimem_ld_reduce_add_v4f32(&acc, mc_v + i);
    multimem_st_v4f32(mc_v + i, acc);
  }
  for (int i = n_vec * 4 + comms_tid; i < n_f32; i += comms_nthreads) {
    float v;
    asm volatile("multimem.ld_reduce.global.add.f32 %0, [%1];"
                 : "=f"(v) : "l"(mc_f32 + i) : "memory");
    asm volatile("multimem.st.global.f32 [%0], %1;"
                 : : "l"(mc_f32 + i), "f"(v) : "memory");
  }
}

// ===========================================================================
// (2) Cross-rank barrier over a small P2P-mapped flag region (used by BOTH
//     tiers). peer_flags[r] points (in THIS rank's address space, via P2P
//     NVLink) to the base of rank r's own flag array, a uint64[world] block, all
//     P2P-accessible from here. A rank announces arrival at `phase` by storing
//     `phase+1` into the slot indexed by its own rank on EVERY peer (incl self),
//     then spins until every rank's slot in its OWN array reaches `phase+1`.
//     system-scope release/acquire atomics make prior region writes visible
//     across ranks. No host involvement; safe between in-kernel regions.
// ===========================================================================
struct RankBarrier {
  unsigned long long* peer_flags[kMaxRanks];  // peer_flags[r] = rank r's flags
  int rank;
  int world;
};

__device__ __forceinline__ void rank_barrier_sync(const RankBarrier& b,
                                                  unsigned long long phase) {
  const unsigned long long target = phase + 1ull;
  // RELEASE: publish all of this rank's prior region writes (the inputs peers
  // are about to read) to system scope BEFORE signaling arrival. ONE system
  // fence here replaces per-atom release ordering on the flag stores — measured
  // ~6x cheaper for the rendezvous (1.6us relaxed+fence vs 10us st.release.sys
  // per flag) on this 4-GPU NVLink config.
  __threadfence_system();
  // Signal arrival with RELAXED system-scope atomics (visibility/ordering is
  // provided by the fences, not the atoms). Remote stores fire-and-forget.
  for (int r = 0; r < b.world; ++r) {
    asm volatile("st.global.relaxed.sys.u64 [%0], %1;"
                 :
                 : "l"(b.peer_flags[r] + b.rank), "l"(target)
                 : "memory");
  }
  // Spin on our OWN (local) array — peers push their arrival into our slots, so
  // each spin load is a cheap local HBM read, not an NVLink round-trip.
  for (int r = 0; r < b.world; ++r) {
    volatile unsigned long long* slot =
        reinterpret_cast<volatile unsigned long long*>(b.peer_flags[b.rank] + r);
    while (*slot < target) { /* tight local spin */ }
  }
  // ACQUIRE: all peers have arrived; make their published region writes visible
  // to us before we read them past the barrier.
  __threadfence_system();
}

// ---------------------------------------------------------------------------
// Intra-grid (single-GPU, all-CTA) sense-reversing barrier. Needed so that ALL
// resident CTAs/warps wait for the leader's cross-rank handshake before reading
// peers — otherwise non-leader comms warps would race ahead. `arrive`/`sense`
// are device globals (zero-init once); `expected` = resident CTA count.
// ---------------------------------------------------------------------------
struct GridBarrier {
  unsigned int* arrive;  // [1]
  unsigned int* sense;   // [1]
  int expected;          // participating CTA count
};

__device__ __forceinline__ void grid_barrier(const GridBarrier& g,
                                              int* local_sense) {
  __syncthreads();
  const int my = *local_sense;
  if (threadIdx.x == 0) {
    unsigned int c = atomicAdd(g.arrive, 1u) + 1u;
    if (c == (unsigned)g.expected) {
      *g.arrive = 0u;
      __threadfence();
      atomicExch(g.sense, (unsigned)(my ^ 1));
    } else {
      volatile unsigned int* vs = g.sense;
      while (*vs == (unsigned)my) { /* spin */ }
    }
  }
  __syncthreads();
  *local_sense = my ^ 1;
}

// Combined CROSS-RANK + INTRA-GRID barrier, fused into a SINGLE grid sync.
//
// Every CTA arrives (atomicAdd on `arrive`). The LAST CTA to arrive (the one
// that pushes the counter to `expected`) is the one thread that drives the
// cross-rank flag handshake — at that moment every CTA of this rank has reached
// the barrier, so its inputs are produced. Only after the cross-GPU handshake
// completes does it flip the grid `sense`, releasing all local CTAs. This folds
// the previous two grid barriers into one (the cross-rank latency is paid by the
// already-waiting CTAs, not added serially), roughly halving the barrier cost.
//
// On return every thread of every CTA on every rank is past the barrier and
// peers' published writes are visible (rank_barrier_sync ends in a
// threadfence_system; we also acquire-fence here for the local release).
__device__ __forceinline__ void cross_rank_grid_barrier(
    const RankBarrier& rb, const GridBarrier& gb, int* local_sense,
    unsigned long long phase) {
  __syncthreads();
  const int my = *local_sense;
  if (threadIdx.x == 0) {
    unsigned int c = atomicAdd(gb.arrive, 1u) + 1u;
    if (c == (unsigned)gb.expected) {
      // Last local arriver: all CTAs here have produced. Do the cross-GPU sync,
      // then release the whole local grid.
      *gb.arrive = 0u;
      rank_barrier_sync(rb, phase);  // cross-rank (ends with threadfence_system)
      atomicExch(gb.sense, (unsigned)(my ^ 1));
    } else {
      volatile unsigned int* vs = gb.sense;
      while (*vs == (unsigned)my) { /* spin on local sense flag */ }
    }
  }
  __syncthreads();
  __threadfence();  // acquire: peers' + leader's writes visible before we read
  *local_sense = my ^ 1;
}

// ===========================================================================
// (b) P2P-NVLink in-kernel one-shot all-reduce (tier-b path; what runs when the
//     fabric multicast team is unavailable).
//
//     `peer_in[r]`  = peer r's UNICAST input buffer, P2P-mapped into this rank's
//                     address space (directly dereferenceable over NVLink).
//     `my_out`      = this rank's output buffer (its own device memory).
//     Each thread strides over the buffer; for element i it reads i from all
//     `world` peers over NVLink and writes the sum to my_out[i]. The caller
//     brackets this with rank_barrier_sync (before: all ranks finished writing
//     their inputs; after: not needed for correctness of my_out since each rank
//     reads inputs that are already published by the pre-barrier, but a post-
//     barrier keeps inputs alive until all readers are done before the next
//     region overwrites them).
//
//     This is the classic small-message one-shot all-reduce (read-all-sum). It
//     moves (world-1)/world * size per rank over NVLink, same traffic class as
//     NCCL's NVLink one-shot for small payloads.
// ===========================================================================
struct PeerPtrs {
  const float* in[kMaxRanks];  // peer unicast input buffers (P2P-mapped)
  int world;
};

__device__ __forceinline__ void one_shot_all_reduce_f32_body(
    const PeerPtrs& pp, float* my_out, int n_f32, int comms_tid,
    int comms_nthreads) {
  const int world = pp.world;
  const int n_vec = n_f32 / 4;
  for (int i = comms_tid; i < n_vec; i += comms_nthreads) {
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int r = 0; r < kMaxRanks; ++r) {
      if (r >= world) break;
      const float4 v = reinterpret_cast<const float4*>(pp.in[r])[i];
      acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    reinterpret_cast<float4*>(my_out)[i] = acc;
  }
  for (int i = n_vec * 4 + comms_tid; i < n_f32; i += comms_nthreads) {
    float acc = 0.f;
    for (int r = 0; r < world; ++r) acc += pp.in[r][i];
    my_out[i] = acc;
  }
}

}  // namespace g6
}  // namespace pde

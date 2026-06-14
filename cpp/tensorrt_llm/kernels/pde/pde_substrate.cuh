// PDE G0 — Persistent Decode Engine device substrate (standalone, sm_100).
//
// Reusable device primitives for the warp-specialized persistent decode megakernel:
//   1. Persistent cooperative-grid launch sized from device props.
//   2. Grid-wide region barrier, two impls:
//        (a) cooperative-groups grid.sync()
//        (b) hand-rolled global arrival/release barrier (atomic + sense reversal +
//            release/acquire threadfence).
//   3. Warp-role scaffold (role enum) + intra-CTA producer->consumer SMEM mailbox
//      via mbarrier (cuda::barrier).
//   4. Atomic global work-stealing queue (atomic head index) handing M items to the
//      resident grid with no double-processing and no drops.
//
// This file does NOT touch the existing decode/model path; it is compiled standalone.
#pragma once

#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cstdint>
#include <cstdio>

namespace pde {

namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// Warp roles for the warp-specialized megakernel.
// ---------------------------------------------------------------------------
enum class WarpRole : int {
  kDma = 0,       // producer: async global->shared copies (TMA/cp.async)
  kMma = 1,       // tensor-core math
  kEpilogue = 2,  // scale / quantize / activation / store
  kComms = 3,     // NVLink all-reduce / a2a
  kCopy = 4,      // KV staging copies (hisparse swap-in)
  kNumRoles = 5,
};

// Assign a role to a warp from its CTA-local warp index. Layout: warp 0 = DMA,
// warp 1 = MMA, warp 2 = epilogue, warp 3 = comms, rest = copy. The concrete
// megakernel will override this; G0 only needs a deterministic, testable map.
__device__ __forceinline__ WarpRole role_of_warp(int warp_id_in_cta) {
  switch (warp_id_in_cta) {
    case 0: return WarpRole::kDma;
    case 1: return WarpRole::kMma;
    case 2: return WarpRole::kEpilogue;
    case 3: return WarpRole::kComms;
    default: return WarpRole::kCopy;
  }
}

// ---------------------------------------------------------------------------
// (2b) Hand-rolled grid-wide region barrier.
//
// Centralized sense-reversing barrier across all resident CTAs. One CTA-leader
// thread arrives; the last arriver flips a shared sense flag and releases.
// Memory ordering: a release fence before publishing arrival makes all of this
// CTA's region-r writes visible; an acquire fence after release makes all peers'
// region-r writes visible before region r+1 reads them.
//
// `arrive`  = device-global atomic counter (zero-initialized once).
// `sense`   = device-global sense flag (zero-initialized once).
// `expected`= number of participating CTAs (== grid.x for a 1-D persistent grid).
// Each thread keeps a thread-local `local_sense` it flips every call.
// ---------------------------------------------------------------------------
struct GlobalBarrier {
  unsigned int* arrive;  // [1]
  unsigned int* sense;   // [1]
  int expected;          // participating CTA count
};

__device__ __forceinline__ void global_barrier_sync(const GlobalBarrier& b,
                                                     int* local_sense) {
  // Publish this CTA's region writes to all peers before announcing arrival.
  __threadfence();
  __syncthreads();  // all warps in this CTA reach the barrier together

  const int my_sense = *local_sense;
  if (threadIdx.x == 0) {
    // atomicAdd returns the pre-increment value; the CTA that pushes the counter
    // to `expected` is the last arriver and flips the sense to release everyone.
    unsigned int count = atomicAdd(b.arrive, 1u) + 1u;
    if (count == static_cast<unsigned int>(b.expected)) {
      *b.arrive = 0u;            // reset for the next region boundary
      __threadfence();          // ensure reset is visible before release
      atomicExch(b.sense, static_cast<unsigned int>(my_sense ^ 1));
    } else {
      // Spin on a plain acquire load of the sense flag (not an atomic RMW: a
      // read-only spin keeps the cacheline shared and avoids serializing the
      // releaser). volatile forces the reload each iteration.
      volatile unsigned int* vsense = b.sense;
      while (*vsense == static_cast<unsigned int>(my_sense)) {
        __nanosleep(20);  // light backoff to ease the spin-read pressure
      }
    }
  }
  __syncthreads();  // broadcast release to all warps of this CTA
  // Acquire: make every peer CTA's region-r writes visible before we read them.
  __threadfence();
  *local_sense = my_sense ^ 1;
}

// ---------------------------------------------------------------------------
// (2a) Cooperative-groups grid barrier — thin wrapper for symmetry / A-B test.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void cg_grid_barrier(const cg::grid_group& grid) {
  grid.sync();
}

// ---------------------------------------------------------------------------
// (3) Intra-CTA producer->consumer SMEM mailbox via mbarrier (cuda::barrier).
//
// A producer warp fills a SMEM tile then `arrive()`s; a consumer warp `wait()`s
// before reading. One barrier object per mailbox slot lives in shared memory.
// This is the async-barrier handshake the megakernel uses between the DMA warp
// and the MMA warp for double-buffered tiles.
// ---------------------------------------------------------------------------
using MailboxBarrier = cuda::barrier<cuda::thread_scope_block>;

// Initialize a mailbox barrier (call once, single thread) with the number of
// threads that will `arrive` (producers) before consumers may proceed.
__device__ __forceinline__ void mailbox_init(MailboxBarrier* mbar,
                                             int arrive_count) {
  init(mbar, arrive_count);
}

// Producer: signal tile-ready. Returns the arrival token for the phase.
__device__ __forceinline__ MailboxBarrier::arrival_token mailbox_produce(
    MailboxBarrier* mbar) {
  return mbar->arrive();
}

// Consumer: block until the producers for this phase have arrived.
__device__ __forceinline__ void mailbox_consume(
    MailboxBarrier* mbar, MailboxBarrier::arrival_token&& tok) {
  mbar->wait(cuda::std::move(tok));
}

// ---------------------------------------------------------------------------
// (4) Atomic global work-stealing queue.
//
// A single monotonically increasing head index hands out item ids to the
// resident grid. Each persistent worker pops with atomicAdd(head, 1); ids >=
// total are out of range (queue drained). No item is handed out twice (atomic
// fetch-add is a total order), and none is dropped (workers loop until drained).
// ---------------------------------------------------------------------------
struct WorkQueue {
  unsigned int* head;  // [1] next item index, zero-initialized
  unsigned int total;  // number of items M
};

// Pop the next item id; returns true and sets `item` while items remain.
__device__ __forceinline__ bool work_queue_pop(const WorkQueue& q,
                                                unsigned int* item) {
  unsigned int idx = atomicAdd(q.head, 1u);
  if (idx >= q.total) return false;
  *item = idx;
  return true;
}

// ---------------------------------------------------------------------------
// Persistent cooperative-grid launch sizing.
//
// Returns a grid size that fully occupies the device for a cooperative launch:
//   grid.x = SM_count * max_resident_CTAs_per_SM(kernel, block, smem)
// clamped to the cooperative-launch device limit. Caller passes the kernel ptr,
// block size, and dynamic smem so the occupancy query matches the real launch.
// ---------------------------------------------------------------------------
struct GridPlan {
  int sm_count;
  int blocks_per_sm;
  int grid_blocks;  // = sm_count * blocks_per_sm (the resident CTA count)
  int block_threads;
  size_t dynamic_smem;
};

inline cudaError_t plan_persistent_grid(const void* kernel, int block_threads,
                                        size_t dynamic_smem, int device,
                                        GridPlan* out) {
  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) return err;

  int blocks_per_sm = 0;
  err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, kernel, block_threads, dynamic_smem);
  if (err != cudaSuccess) return err;
  if (blocks_per_sm < 1) blocks_per_sm = 1;

  out->sm_count = prop.multiProcessorCount;
  out->blocks_per_sm = blocks_per_sm;
  out->grid_blocks = prop.multiProcessorCount * blocks_per_sm;
  out->block_threads = block_threads;
  out->dynamic_smem = dynamic_smem;
  return cudaSuccess;
}

}  // namespace pde

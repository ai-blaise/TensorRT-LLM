// PDE G7 — in-kernel EP (expert-parallel) all-to-all dispatch+combine for the
// persistent decode megakernel (standalone, sm_100, model-free).
//
// The MoE layer routes each token to the rank(s) owning its top-k expert(s) and
// must do the resulting expert all-to-all WITHOUT exiting the megakernel:
//   dispatch  — each rank pushes its tokens to the owning rank's recv buffer
//               (DATA-DEPENDENT, variable per-(src,dst) counts);
//   expert    — a synthetic per-expert transform on the received tokens;
//   combine   — scatter the expert outputs back to the originating rank.
// Every rank ends with its tokens' combined expert outputs, never having left
// the kernel.
//
// IMPL LADDER (see pde_g7_results.md for the measured tier):
//   (a) NVLS `multimem` — irrelevant for a *gather/scatter* a2a (multimem is a
//       reduction primitive; a2a moves disjoint payloads, no fabric reduce), and
//       in any case `cuMulticastBindMem` is environment-blocked on this
//       passthrough VM (no fabric-manager / IMEX / NVSwitch — verified in G6,
//       NCCL fails identically). Not applicable.
//   (b) P2P-NVLink in-kernel push a2a (THIS is the tier the bench runs): each
//       rank writes its tokens straight into the destination rank's recv buffer
//       over peer-mapped NVLink pointers at DEVICE-COMPUTED offsets. Fully in-
//       kernel, fully model-free. `dispatch_push_body` / `combine_push_body`.
//   (c) staged P2P — same push path with an explicit local staging buffer; the
//       direct push (b) subsumes it here (the recv buffer IS the stage).
//
// DATA-DEPENDENT COUNTS — handled entirely ON-DEVICE (the crux vs G6's fixed-
// size all-reduce):
//   1. count   — each rank histograms its tokens by destination rank
//                (send_count[src][dst]).
//   2. publish — each rank pushes its send_count row to a P2P-shared world×world
//                count matrix on EVERY peer, so every rank can read the full
//                matrix and derive its recv counts (recv_count[dst][src] =
//                send_count[src][dst]) — no host, no cudaMemcpy.
//   3. offsets — exclusive prefix sums over the matrix give, on-device, every
//                rank's send base into each peer AND each peer's recv base for
//                this rank's payload. The push lands tokens contiguously.
//   4. push    — variable-count gather into per-dst send order, then a strided
//                NVLink store into the destination's recv region at its base.
//
// Builds on `pde_g6_allreduce.cuh` (the relaxed-atomic cross-rank barrier
// `rank_barrier_sync` + `cross_rank_grid_barrier`) and the G0 substrate. Does
// NOT touch the model/decode path; compiled standalone by the G7 bench.
#pragma once

#include "pde_g6_allreduce.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace pde {
namespace g7 {

static constexpr int kMaxRanks = 8;

// Reuse G6's verified cross-rank rendezvous primitives unchanged.
using RankBarrier = pde::g6::RankBarrier;
using GridBarrier = pde::g6::GridBarrier;
using pde::g6::cross_rank_grid_barrier;  // grid + cross-GPU (peer-read points)
using pde::g6::grid_barrier;             // intra-grid only (local-only stages)
using pde::g6::rank_barrier_sync;

// ===========================================================================
// Peer handles for the in-kernel a2a. Every pointer below is P2P-mapped into
// THIS rank's address space (directly dereferenceable over NVLink after
// cudaDeviceEnablePeerAccess). `[r]` indexes peer rank r.
//
//   send_cnt_peer[r]  -> rank r's count matrix (int[world*world], row-major
//                        [src][dst]). This rank publishes its send-count row by
//                        storing into column-block of EVERY peer's matrix.
//   recv_buf_peer[r]  -> rank r's dispatch RECV region (tokens land here),
//                        int payload, row-major [slot][hidden].
//   recv_meta_peer[r] -> rank r's per-recv-slot source-token id (int[capacity]),
//                        written alongside the payload so combine can scatter
//                        back to the exact origin slot.
//   combine_peer[r]   -> rank r's COMBINE region (expert outputs land back here),
//                        row-major [token][hidden]; the origin rank reads its
//                        own combined tokens from here after the barrier.
// ===========================================================================
struct A2AHandles {
  int* send_cnt_peer[kMaxRanks];      // world*world int matrix per peer
  int* recv_buf_peer[kMaxRanks];      // dispatch recv payload per peer
  int* recv_meta_peer[kMaxRanks];     // per-recv-slot origin token id per peer
  int* combine_peer[kMaxRanks];       // combine payload per peer
  int world;
  int hidden;                          // ints per token
  int recv_capacity;                   // max recv slots per rank (per-rank cap)
};

// ===========================================================================
// (1) COUNT — histogram this rank's tokens by destination rank.
//   route[t] in [0,world): destination rank of local token t (from top-k expert
//   ownership, computed by the caller's synthetic router). Writes the per-dst
//   send count into this rank's OWN matrix row [my_rank][*]. Single CTA, single
//   warp drives it (counts are tiny: world entries).
// ===========================================================================
__device__ __forceinline__ void count_by_dst(const int* route, int n_tokens,
                                              int* my_send_row, int world,
                                              int tid, int nthreads) {
  // Zero the row, then atomic-accumulate. my_send_row = &matrix[my_rank*world].
  for (int d = tid; d < world; d += nthreads) my_send_row[d] = 0;
  __syncthreads();
  for (int t = tid; t < n_tokens; t += nthreads) {
    int d = route[t];
    atomicAdd(&my_send_row[d], 1);
  }
}

// ===========================================================================
// (2) PUBLISH — push this rank's send-count row to EVERY peer's matrix (so each
// peer can read recv_count[dst][src] = send_count[src][dst]). Relaxed system-
// scope stores over P2P; the surrounding rank barrier provides ordering. One
// thread per (peer) entry.
// ===========================================================================
__device__ __forceinline__ void publish_send_counts(const A2AHandles& h,
                                                     int my_rank,
                                                     const int* my_send_row,
                                                     int tid, int nthreads) {
  const int world = h.world;
  // Each peer r needs OUR row written into row [my_rank] of ITS matrix.
  for (int idx = tid; idx < world * world; idx += nthreads) {
    int r = idx / world;          // destination peer
    int d = idx % world;          // column (dst rank)
    int* dst = h.send_cnt_peer[r] + my_rank * world + d;
    int v = my_send_row[d];
    asm volatile("st.global.relaxed.sys.s32 [%0], %1;"
                 : : "l"(dst), "r"(v) : "memory");
  }
}

// ===========================================================================
// On-device offset derivation from the full (now-published) count matrix.
//   matrix[src*world + dst] = tokens src sends to dst.
//   send_base[d] = exclusive prefix over d of OUR row  -> where in our gathered
//                  send order the block for dst d begins (local packing).
//   recv_base_at_dst[d] = where, inside dst d's recv region, OUR payload starts
//                  = sum over src<my_rank of matrix[src*world + d]
//                  (exclusive prefix over the COLUMN d, up to my_rank).
//   recv_total = sum over src of matrix[src*world + my_rank] (what we receive).
// All computed from the shared matrix; no host involvement. Returns via out
// params (small: world entries). Single thread is fine (world<=8).
// ===========================================================================
struct A2AOffsets {
  int send_base[kMaxRanks];          // local pack base per dst
  int send_count[kMaxRanks];         // our tokens to each dst
  int recv_base_at_dst[kMaxRanks];   // our payload base inside each dst's recv
  int recv_count_from[kMaxRanks];    // tokens we recv from each src
  int send_total;
  int recv_total;
};

__device__ __forceinline__ void derive_offsets(const int* matrix, int world,
                                               int my_rank, A2AOffsets* o) {
  // our send row
  int acc = 0;
  for (int d = 0; d < world; ++d) {
    o->send_base[d] = acc;
    int c = matrix[my_rank * world + d];
    o->send_count[d] = c;
    acc += c;
  }
  o->send_total = acc;
  // our recv: column my_rank. recv_base_at_dst[d] is, for each destination d we
  // SEND to, the exclusive column-prefix of column d up to my_rank (= bytes from
  // lower-ranked srcs already occupying d's recv region).
  for (int d = 0; d < world; ++d) {
    int base = 0;
    for (int src = 0; src < my_rank; ++src) base += matrix[src * world + d];
    o->recv_base_at_dst[d] = base;
  }
  // what WE receive, per source (column my_rank), and the running recv base into
  // our own region per source (used by the expert/compute + reference).
  int racc = 0;
  for (int src = 0; src < world; ++src) {
    int c = matrix[src * world + my_rank];
    o->recv_count_from[src] = c;
    racc += c;
  }
  o->recv_total = racc;
}

// ===========================================================================
// (3) LOCAL PACK — gather this rank's tokens into per-dst-contiguous send order.
//   For token t with dst d, its position in the packed send buffer is
//   send_base[d] + (its rank among this rank's tokens going to d). We compute
//   that rank with an atomic cursor per dst (seeded from send_base). Writes the
//   packed payload AND the origin token id (for combine scatter-back).
//   send_pack    : int[send_total * hidden]
//   send_pack_src: int[send_total]   (origin local token id)
//   send_dst_of  : int[send_total]   (dst rank of each packed slot) -> for push
//   cursor       : int[world]        (atomic cursors, pre-seeded to send_base)
// ===========================================================================
__device__ __forceinline__ void local_pack(const int* tokens, const int* route,
                                            int n_tokens, int hidden,
                                            const A2AOffsets& o, int* cursor,
                                            int* send_pack, int* send_pack_src,
                                            int* send_dst_of, int tid,
                                            int nthreads) {
  // One thread per token claims a packed slot; then the whole grid copies the
  // payloads. Two-pass to keep slot assignment race-free across the grid.
  for (int t = tid; t < n_tokens; t += nthreads) {
    int d = route[t];
    int slot = atomicAdd(&cursor[d], 1);  // cursor pre-seeded to send_base[d]
    send_pack_src[slot] = t;
    send_dst_of[slot] = d;
  }
}

// Copy payloads into the packed order (grid-strided over slot*hidden). Must run
// AFTER local_pack assigned every slot (separated by a grid barrier).
__device__ __forceinline__ void pack_payloads(const int* tokens, int hidden,
                                              int send_total,
                                              const int* send_pack_src,
                                              int* send_pack, int gtid,
                                              int gnthreads) {
  const long total = (long)send_total * hidden;
  for (long e = gtid; e < total; e += gnthreads) {
    int slot = (int)(e / hidden);
    int col = (int)(e % hidden);
    int src_tok = send_pack_src[slot];
    send_pack[e] = tokens[(long)src_tok * hidden + col];
  }
}

// ===========================================================================
// (4) DISPATCH PUSH — write the packed per-dst blocks into each destination's
// recv region over P2P NVLink at the device-computed recv base. Also writes the
// per-recv-slot origin id (global-ish: we store the SOURCE rank in the high bits
// and the source local token id in the low bits so combine can route back
// exactly). Grid-strided over our send_total*hidden elements.
//
//   For packed slot s (dst d = send_dst_of[s]), its position WITHIN d's recv
//   region = recv_base_at_dst[d] + (s - send_base[d]).  We push payload+meta to
//   peer d's recv_buf/recv_meta at that slot.
// ===========================================================================
__device__ __forceinline__ void dispatch_push_body(const A2AHandles& h,
                                                    int my_rank,
                                                    const A2AOffsets& o,
                                                    const int* send_pack,
                                                    const int* send_pack_src,
                                                    const int* send_dst_of,
                                                    int gtid, int gnthreads) {
  const int hidden = h.hidden;
  const long total = (long)o.send_total * hidden;
  for (long e = gtid; e < total; e += gnthreads) {
    int s = (int)(e / hidden);
    int col = (int)(e % hidden);
    int d = send_dst_of[s];
    int within = (s - o.send_base[d]) + o.recv_base_at_dst[d];  // slot in dst recv
    int* dst_payload = h.recv_buf_peer[d] + (long)within * hidden + col;
    int v = send_pack[e];
    // P2P store over NVLink (relaxed; rank barrier orders vs the consumer).
    asm volatile("st.global.relaxed.sys.s32 [%0], %1;"
                 : : "l"(dst_payload), "r"(v) : "memory");
    // meta (origin) once per slot (col 0): pack (src_rank<<20 | src_tok).
    if (col == 0) {
      int meta = (my_rank << 20) | (send_pack_src[s] & 0xFFFFF);
      int* dst_meta = h.recv_meta_peer[d] + within;
      asm volatile("st.global.relaxed.sys.s32 [%0], %1;"
                   : : "l"(dst_meta), "r"(meta) : "memory");
    }
  }
}

// ===========================================================================
// EXPERT COMPUTE — synthetic per-expert transform on the tokens this rank
// received (now contiguous in our OWN recv_buf, recv_total slots). Stand-in for
// the grouped expert GEMM. Deterministic + invertible-checkable: out = in*2 + 1
// per element (the reference applies the identical transform), so the gate is
// bit-exact. Grid-strided over recv_total*hidden of OUR own recv buffer.
// ===========================================================================
__device__ __forceinline__ void expert_compute_body(int* my_recv_buf,
                                                     int* my_expert_out,
                                                     int recv_total, int hidden,
                                                     int gtid, int gnthreads) {
  const long total = (long)recv_total * hidden;
  for (long e = gtid; e < total; e += gnthreads) {
    my_expert_out[e] = my_recv_buf[e] * 2 + 1;
  }
}

// ===========================================================================
// (5) COMBINE PUSH — scatter each expert output back to its ORIGIN rank/slot.
// For recv slot k (origin meta = src_rank<<20 | src_tok), push expert_out[k] to
// the origin rank's combine region at row src_tok. Grid-strided over OUR
// recv_total*hidden. Reads meta from our OWN recv_meta. Origin rank reads its
// combined tokens from its own combine region after the barrier.
// ===========================================================================
__device__ __forceinline__ void combine_push_body(const A2AHandles& h,
                                                   const int* my_expert_out,
                                                   const int* my_recv_meta,
                                                   int recv_total, int gtid,
                                                   int gnthreads) {
  const int hidden = h.hidden;
  const long total = (long)recv_total * hidden;
  for (long e = gtid; e < total; e += gnthreads) {
    int k = (int)(e / hidden);
    int col = (int)(e % hidden);
    int meta = my_recv_meta[k];
    int src_rank = (meta >> 20) & 0xFFF;
    int src_tok = meta & 0xFFFFF;
    int* dst = h.combine_peer[src_rank] + (long)src_tok * hidden + col;
    int v = my_expert_out[e];
    asm volatile("st.global.relaxed.sys.s32 [%0], %1;"
                 : : "l"(dst), "r"(v) : "memory");
  }
}

}  // namespace g7
}  // namespace pde

// PDE G7 — in-kernel EP=4 all-to-all dispatch+combine microbench (model-free,
// GPU 0-3).
//
// TIER REACHED: (b) P2P-NVLink in-kernel push all-to-all.
//   Tier (a) NVLS `multimem` is (i) not a fit for a gather/scatter a2a (multimem
//   is a *reduction* primitive — a2a moves disjoint payloads with no fabric
//   reduce) and (ii) environment-blocked anyway: cuMulticastBindMem is rejected
//   on this GPU-passthrough VM (no fabric-manager / IMEX / NVSwitch — verified
//   in G6, NCCL fails identically). See pde_g7_results.md.
//
// One process, `world` host threads (one per GPU 0..world-1). Each thread owns
// its device and a persistent kernel. P2P enabled between all pairs, so each
// rank's kernel writes its tokens straight into peers' recv buffers over NVLink.
// The MoE expert all-to-all is data-dependent: a synthetic deterministic top-1
// router gives each token a destination rank, producing VARIABLE per-(src,dst)
// counts. The in-kernel kernel: count -> publish counts (P2P) -> derive offsets
// on-device -> local pack -> DISPATCH push -> synthetic expert compute ->
// COMBINE push back. Every rank ends with its tokens' combined expert outputs,
// never having left the kernel.
//
// GATE
//   CORRECTNESS (hard): in-kernel dispatch+combine == INDEPENDENT references on
//     ALL ranks. Two independent refs:
//       (i)  NCCL grouped ncclSend/ncclRecv all-to-all (variable counts) + the
//            same synthetic expert transform applied via NCCL routing back;
//       (ii) host gather-route-scatter of the known integer payloads.
//     Integer payloads => bit-exact. Also verifies the data-dependent routing
//     delivers every token to the right rank with NO drops / dupes / misroutes
//     (per-(src,dst) count matrix checked against the analytic router; received
//     payload identity checked per token).
//   PERF: in-kernel a2a (dispatch+combine) latency vs LAUNCH-BOUNDARY NCCL
//     all-to-all at representative decode sizes; reports in-kernel latency, NCCL
//     baseline, comms∥compute overlap, achieved NVLink BW, tier reached.
//
// Build/run: see build_run_g7.sh.

#include "pde_g7_alltoall.cuh"

#include <nccl.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA err %s:%d: %s\n", __FILE__, __LINE__,             \
              cudaGetErrorString(e));                                          \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

#define NCCL_CHECK(x)                                                          \
  do {                                                                         \
    ncclResult_t e = (x);                                                      \
    if (e != ncclSuccess) {                                                    \
      fprintf(stderr, "NCCL err %s:%d: %s\n", __FILE__, __LINE__,             \
              ncclGetErrorString(e));                                          \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

static constexpr int kWorld = 4;     // EP=4, GPU 0..3
static constexpr int kExperts = 32;  // total experts; kExperts/kWorld per rank
static constexpr int kBlock = 256;
static constexpr int kIters = 100;
static constexpr int kWarmup = 20;

// Representative decode MoE a2a sizes. hidden = ints/token; T = tokens/rank.
// recv_capacity is per-rank max recv slots (worst case all ranks -> one rank);
// we size to T*world to never overflow under skew.
struct SizeCase { const char* name; int T; int hidden; };
static const SizeCase kSizes[] = {
    {"T8_h512",    8,   512},
    {"T16_h1024",  16,  1024},
    {"T32_h1024",  32,  1024},
    {"T64_h2048",  64,  2048},
    {"T128_h2048", 128, 2048},
    {"T256_h2048", 256, 2048},
};
static constexpr int kNumSizes = sizeof(kSizes) / sizeof(kSizes[0]);
static constexpr int kMaxT = 256;
static constexpr int kMaxHidden = 2048;
static constexpr int kMaxRecvCap = kMaxT * kWorld;  // per-rank recv capacity

// ---------------------------------------------------------------------------
// Synthetic DATA-DEPENDENT router (deterministic). Token t on rank r maps to a
// top-1 expert; the owning rank = expert / (kExperts/kWorld). Skewed so the
// per-(src,dst) counts are genuinely variable (not uniform): a quadratic hash
// biased by (rank, t). This is the routing the in-kernel path and BOTH
// references all use, so any disagreement is a real bug.
// ---------------------------------------------------------------------------
__host__ __device__ static inline int route_expert(int rank, int t) {
  unsigned h = (unsigned)(rank * 2654435761u + t * 40503u);
  h ^= h >> 13; h *= 0x9E3779B1u; h ^= h >> 16;
  // bias toward a rank-dependent cluster of experts to force count skew
  int base = (rank * 7) % kExperts;
  int span = kExperts / 2 + (int)(h % (kExperts / 2));
  return (base + (int)(h % span)) % kExperts;
}
__host__ __device__ static inline int route_rank(int rank, int t) {
  return route_expert(rank, t) / (kExperts / kWorld);
}
// Known integer payload: token (rank,t), element c -> deterministic value.
__host__ __device__ static inline int payload_val(int rank, int t, int c) {
  return (rank + 1) * 1000003 + t * 131 + c;  // distinct per (rank,t,c)
}
// expert transform (matches expert_compute_body): out = in*2 + 1.
__host__ __device__ static inline int expert_xform(int v) { return v * 2 + 1; }

// ===========================================================================
// IN-KERNEL a2a kernel — ONE grid-wide pass: count -> publish -> derive ->
// pack-assign -> [barrier] -> pack-copy -> [barrier] -> dispatch-push ->
// [barrier] -> expert -> [barrier] -> combine-push -> [barrier]. Each cross-rank
// barrier publishes the prior P2P writes to peers. Validated vs NCCL + host.
// ===========================================================================
__global__ void pde_g7_a2a_kernel(
    pde::g7::A2AHandles h, int my_rank, const int* tokens, const int* route,
    int n_tokens, int* my_send_row, int* matrix, int* cursor, int* send_pack,
    int* send_pack_src, int* send_dst_of, int* my_recv_buf, int* my_recv_meta,
    int* my_expert_out, pde::g7::A2AOffsets* o_out, pde::g7::RankBarrier bar,
    pde::g7::GridBarrier gb, unsigned long long phase) {
  const int hidden = h.hidden;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const int gnthreads = gridDim.x * blockDim.x;
  int local_sense = 0;

  // BARRIER PLACEMENT: a stage transition needs a CROSS-RANK barrier ONLY when
  // the next stage reads data a PEER produced over P2P; purely local transitions
  // need just an INTRA-GRID barrier (much cheaper — no NVLink rendezvous). Only 3
  // of the 8 transitions are peer-read points: publish→derive (peers wrote the
  // matrix), dispatch→expert (peers wrote our recv buffer), combine→end (peers
  // wrote our combine region). The other 5 are intra-grid. This is the dominant
  // G7 latency lever (8 cross-rank barriers -> 3).

  // --- COUNT (CTA 0 owns the tiny world-entry histogram) ---
  if (blockIdx.x == 0) {
    pde::g7::count_by_dst(route, n_tokens, my_send_row, h.world, tid, nthreads);
  }
  pde::g7::grid_barrier(gb, &local_sense);  // local: publish reads our own row

  // --- PUBLISH counts to all peers' matrices (P2P) ---
  if (blockIdx.x == 0) {
    pde::g7::publish_send_counts(h, my_rank, my_send_row, tid, nthreads);
  }
  pde::g7::cross_rank_grid_barrier(bar, gb, &local_sense, phase + 0);  // peers wrote matrix

  // --- DERIVE offsets on-device from the full matrix; seed pack cursors ---
  __shared__ pde::g7::A2AOffsets o_sh;
  if (blockIdx.x == 0 && tid == 0) {
    pde::g7::derive_offsets(matrix, h.world, my_rank, &o_sh);
    for (int d = 0; d < h.world; ++d) cursor[d] = o_sh.send_base[d];
    *o_out = o_sh;  // publish for all CTAs + host readback
  }
  pde::g7::grid_barrier(gb, &local_sense);  // local: every CTA reads o_out
  pde::g7::A2AOffsets o = *o_out;

  // --- LOCAL PACK assign (one thread/token claims a packed slot) ---
  if (blockIdx.x == 0) {
    pde::g7::local_pack(tokens, route, n_tokens, hidden, o, cursor, send_pack,
                        send_pack_src, send_dst_of, tid, nthreads);
  }
  pde::g7::grid_barrier(gb, &local_sense);  // local: pack-copy reads slot assign

  // --- PACK copy payloads (grid-strided) ---
  pde::g7::pack_payloads(tokens, hidden, o.send_total, send_pack_src, send_pack,
                         gtid, gnthreads);
  pde::g7::grid_barrier(gb, &local_sense);  // local: dispatch reads our send_pack

  // --- DISPATCH push to peers' recv buffers (P2P, device offsets) ---
  pde::g7::dispatch_push_body(h, my_rank, o, send_pack, send_pack_src,
                              send_dst_of, gtid, gnthreads);
  pde::g7::cross_rank_grid_barrier(bar, gb, &local_sense, phase + 1);  // peers wrote our recv

  // --- EXPERT compute on our received tokens (now contiguous in my_recv_buf) ---
  pde::g7::expert_compute_body(my_recv_buf, my_expert_out, o.recv_total, hidden,
                               gtid, gnthreads);
  pde::g7::grid_barrier(gb, &local_sense);  // local: combine reads our expert_out

  // --- COMBINE push expert outputs back to origin ranks (P2P) ---
  pde::g7::combine_push_body(h, my_expert_out, my_recv_meta, o.recv_total, gtid,
                             gnthreads);
  pde::g7::cross_rank_grid_barrier(bar, gb, &local_sense, phase + 2);  // peers wrote our combine
}

// ---------------------------------------------------------------------------
// PERSISTENT-LOOP timing kernel: ONE launch performs `inner` in-kernel a2a
// rounds in a device loop, NO relaunch between them (engine-representative — the
// megakernel never relaunches mid-decode-step). `comms_enable` toggles the push
// a2a; `compute_iters` adds independent FMA work in the dispatch∥compute window
// to measure overlap; `barriers_only` isolates the rendezvous floor.
// ---------------------------------------------------------------------------
__global__ void pde_g7_persistent(
    pde::g7::A2AHandles h, int my_rank, const int* tokens, const int* route,
    int n_tokens, int* my_send_row, int* matrix, int* cursor, int* send_pack,
    int* send_pack_src, int* send_dst_of, int* my_recv_buf, int* my_recv_meta,
    int* my_expert_out, pde::g7::A2AOffsets* o_out, pde::g7::RankBarrier bar,
    pde::g7::GridBarrier gb, unsigned long long base_phase, int inner,
    int comms_enable, int mode, int compute_iters, float* dummy_out) {
  // mode 0 = full a2a pipeline; 1 = barriers-only floor; 2 = overlap-window-only
  // (dispatch ∥ compute between 2 barriers, offsets derived once up front).
  const int hidden = h.hidden;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const int gnthreads = gridDim.x * blockDim.x;
  int local_sense = 0;
  float cacc = 1.0f + 1e-3f * (gtid & 1023);
  const float cb = 0.9999f;
  const int NB = 8;  // barriers per round (matches the 8 in the corr kernel)

  // mode 2: derive offsets + pack ONCE (count/publish/derive/pack), then the
  // loop times only the dispatch ∥ compute window between two barriers so the
  // comms∥compute overlap is measured without the other 7 stages diluting it.
  if (mode == 2) {
    if (blockIdx.x == 0)
      pde::g7::count_by_dst(route, n_tokens, my_send_row, h.world, tid, nthreads);
    pde::g7::cross_rank_grid_barrier(bar, gb, &local_sense, base_phase + 0);
    if (blockIdx.x == 0)
      pde::g7::publish_send_counts(h, my_rank, my_send_row, tid, nthreads);
    pde::g7::cross_rank_grid_barrier(bar, gb, &local_sense, base_phase + 1);
    if (blockIdx.x == 0 && tid == 0) {
      pde::g7::A2AOffsets ot;
      pde::g7::derive_offsets(matrix, h.world, my_rank, &ot);
      for (int d = 0; d < h.world; ++d) cursor[d] = ot.send_base[d];
      *o_out = ot;
    }
    pde::g7::cross_rank_grid_barrier(bar, gb, &local_sense, base_phase + 2);
    pde::g7::A2AOffsets o = *o_out;
    if (blockIdx.x == 0)
      pde::g7::local_pack(tokens, route, n_tokens, hidden, o, cursor, send_pack,
                          send_pack_src, send_dst_of, tid, nthreads);
    pde::g7::cross_rank_grid_barrier(bar, gb, &local_sense, base_phase + 3);
    pde::g7::pack_payloads(tokens, hidden, o.send_total, send_pack_src, send_pack,
                           gtid, gnthreads);
    pde::g7::cross_rank_grid_barrier(bar, gb, &local_sense, base_phase + 4);
    // WARP-SPECIALIZED overlap window: half the warps drive the dispatch push
    // over NVLink, the other half run independent compute — concurrently in one
    // resident grid (same scheme G6 used for comms∥compute). comms warps stride
    // the push over a comms-only thread space; compute warps do the FMA.
    const int n_warps = blockDim.x / 32;
    const int n_comms_warps = (n_warps > 1) ? n_warps / 2 : 1;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const bool is_comms = (warp >= n_warps - n_comms_warps);
    const int comms_widx = warp - (n_warps - n_comms_warps);
    const int comms_tid =
        blockIdx.x * (n_comms_warps * 32) + comms_widx * 32 + lane;
    const int comms_nthreads = gridDim.x * n_comms_warps * 32;
    for (int it = 0; it < inner; ++it) {
      unsigned long long ph = base_phase + 8ull + (unsigned long long)it * 4ull;
      pde::g7::cross_rank_grid_barrier(bar, gb, &local_sense, ph + 0);
      if (comms_enable && is_comms)
        pde::g7::dispatch_push_body(h, my_rank, o, send_pack, send_pack_src,
                                    send_dst_of, comms_tid, comms_nthreads);
      if (compute_iters > 0 && !is_comms) {
#pragma unroll 1
        for (int i = 0; i < compute_iters; ++i) {
          cacc = fmaf(cacc, cb, 1e-7f);
          cacc = fmaf(cacc, cb, 1e-7f);
        }
      }
      pde::g7::cross_rank_grid_barrier(bar, gb, &local_sense, ph + 1);
    }
    if (cacc == -123456.0f && dummy_out) dummy_out[gtid % 16] = cacc;
    return;
  }

  for (int it = 0; it < inner; ++it) {
    unsigned long long ph = base_phase + (unsigned long long)it * 16ull;
    if (mode == 1) {
      // Pipeline barrier floor: the REAL mix of the a2a pipeline — 3 cross-rank
      // (the peer-read points) + 5 intra-grid (local-only transitions). This is
      // the floor the full pipeline actually pays, not an all-cross-rank upper
      // bound.
#pragma unroll 1
      for (int b = 0; b < 3; ++b)
        pde::g7::cross_rank_grid_barrier(bar, gb, &local_sense, ph + b);
#pragma unroll 1
      for (int b = 0; b < 5; ++b) pde::g7::grid_barrier(gb, &local_sense);
      continue;
    }
    (void)NB;
    // Same barrier placement as the correctness kernel: cross-rank ONLY at the 3
    // peer-read points (publish→derive, dispatch→expert, combine→end); intra-grid
    // for the 5 local-only transitions. Barrier mix is identical for all
    // comms_enable so the nocomms baseline reflects the real pipeline floor.
    // count
    if (blockIdx.x == 0)
      pde::g7::count_by_dst(route, n_tokens, my_send_row, h.world, tid, nthreads);
    pde::g7::grid_barrier(gb, &local_sense);                              // local
    // publish
    if (blockIdx.x == 0)
      pde::g7::publish_send_counts(h, my_rank, my_send_row, tid, nthreads);
    pde::g7::cross_rank_grid_barrier(bar, gb, &local_sense, ph + 0);      // peers wrote matrix
    // derive
    if (blockIdx.x == 0 && tid == 0) {
      pde::g7::A2AOffsets ot;
      pde::g7::derive_offsets(matrix, h.world, my_rank, &ot);
      for (int d = 0; d < h.world; ++d) cursor[d] = ot.send_base[d];
      *o_out = ot;
    }
    pde::g7::grid_barrier(gb, &local_sense);                              // local
    pde::g7::A2AOffsets o = *o_out;
    // pack assign
    if (blockIdx.x == 0)
      pde::g7::local_pack(tokens, route, n_tokens, hidden, o, cursor, send_pack,
                          send_pack_src, send_dst_of, tid, nthreads);
    pde::g7::grid_barrier(gb, &local_sense);                              // local
    // pack copy
    if (comms_enable)
      pde::g7::pack_payloads(tokens, hidden, o.send_total, send_pack_src,
                             send_pack, gtid, gnthreads);
    pde::g7::grid_barrier(gb, &local_sense);                              // local
    // dispatch ∥ optional independent compute (overlap window)
    if (comms_enable)
      pde::g7::dispatch_push_body(h, my_rank, o, send_pack, send_pack_src,
                                  send_dst_of, gtid, gnthreads);
    if (compute_iters > 0) {
#pragma unroll 1
      for (int i = 0; i < compute_iters; ++i) {
        cacc = fmaf(cacc, cb, 1e-7f);
        cacc = fmaf(cacc, cb, 1e-7f);
      }
    }
    pde::g7::cross_rank_grid_barrier(bar, gb, &local_sense, ph + 1);      // peers wrote our recv
    // expert
    if (comms_enable)
      pde::g7::expert_compute_body(my_recv_buf, my_expert_out, o.recv_total,
                                   hidden, gtid, gnthreads);
    pde::g7::grid_barrier(gb, &local_sense);                              // local
    // combine
    if (comms_enable)
      pde::g7::combine_push_body(h, my_expert_out, my_recv_meta, o.recv_total,
                                 gtid, gnthreads);
    pde::g7::cross_rank_grid_barrier(bar, gb, &local_sense, ph + 2);      // peers wrote our combine
  }
  if (cacc == -123456.0f && dummy_out) dummy_out[gtid % 16] = cacc;  // keep live
}

// ---------------------------------------------------------------------------
// Cross-thread shared handoff.
// ---------------------------------------------------------------------------
struct Shared {
  std::atomic<int> arrive{0};
  std::atomic<int> sense{0};
  // a2a peer buffers per rank
  int* send_cnt_ptr[kWorld];
  int* recv_buf_ptr[kWorld];
  int* recv_meta_ptr[kWorld];
  int* combine_ptr[kWorld];
  std::atomic<int> buf_ready{0};
  unsigned long long* flag_ptr[kWorld];
  std::atomic<int> flag_ready{0};
  ncclUniqueId nccl_id;
  std::atomic<int> nccl_id_ready{0};
  // results per rank (smallest size)
  int corr_nccl_ok[kWorld];
  int corr_host_ok[kWorld];
  int corr_route_ok[kWorld];  // count matrix + no drop/dupe/misroute
  double corr_cos[kWorld];
};

static void cpu_barrier(Shared* s) {
  int g = s->sense.load();
  if (s->arrive.fetch_add(1) == kWorld - 1) {
    s->arrive.store(0);
    s->sense.store(g ^ 1);
  } else {
    while (s->sense.load() == g) std::this_thread::yield();
  }
}

// ---------------------------------------------------------------------------
// Per-rank worker.
// ---------------------------------------------------------------------------
static void worker(int rank, Shared* s) {
  CUDA_CHECK(cudaSetDevice(rank));

  // Enable P2P to all peers.
  for (int r = 0; r < kWorld; ++r) {
    if (r == rank) continue;
    int can = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can, rank, r));
    if (!can) { fprintf(stderr, "rank %d cannot P2P to %d\n", rank, r); std::abort(); }
    cudaError_t e = cudaDeviceEnablePeerAccess(r, 0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
      fprintf(stderr, "rank %d enablePeer %d: %s\n", rank, r, cudaGetErrorString(e));
    cudaGetLastError();
  }

  // Persistent device buffers (own device memory; some P2P-shared).
  int *d_tokens, *d_route, *d_send_row, *d_matrix, *d_cursor;
  int *d_send_pack, *d_send_pack_src, *d_send_dst_of;
  int *d_recv_buf, *d_recv_meta, *d_expert_out, *d_combine;
  pde::g7::A2AOffsets* d_off;
  CUDA_CHECK(cudaMalloc(&d_tokens, sizeof(int) * (long)kMaxT * kMaxHidden));
  CUDA_CHECK(cudaMalloc(&d_route, sizeof(int) * kMaxT));
  CUDA_CHECK(cudaMalloc(&d_send_row, sizeof(int) * kWorld));      // our matrix row alias unused
  CUDA_CHECK(cudaMalloc(&d_matrix, sizeof(int) * kWorld * kWorld));  // P2P-shared count matrix
  CUDA_CHECK(cudaMalloc(&d_cursor, sizeof(int) * kWorld));
  CUDA_CHECK(cudaMalloc(&d_send_pack, sizeof(int) * (long)kMaxT * kMaxHidden));
  CUDA_CHECK(cudaMalloc(&d_send_pack_src, sizeof(int) * kMaxT));
  CUDA_CHECK(cudaMalloc(&d_send_dst_of, sizeof(int) * kMaxT));
  CUDA_CHECK(cudaMalloc(&d_recv_buf, sizeof(int) * (long)kMaxRecvCap * kMaxHidden));  // P2P
  CUDA_CHECK(cudaMalloc(&d_recv_meta, sizeof(int) * kMaxRecvCap));                    // P2P
  CUDA_CHECK(cudaMalloc(&d_expert_out, sizeof(int) * (long)kMaxRecvCap * kMaxHidden));
  CUDA_CHECK(cudaMalloc(&d_combine, sizeof(int) * (long)kMaxT * kMaxHidden));         // P2P
  CUDA_CHECK(cudaMalloc(&d_off, sizeof(pde::g7::A2AOffsets)));

  // The published matrix row is matrix[rank*world ..]; count_by_dst writes there.
  // Share the P2P buffers.
  s->send_cnt_ptr[rank] = d_matrix;
  s->recv_buf_ptr[rank] = d_recv_buf;
  s->recv_meta_ptr[rank] = d_recv_meta;
  s->combine_ptr[rank] = d_combine;
  s->buf_ready.fetch_add(1);
  while (s->buf_ready.load() < kWorld) std::this_thread::yield();
  cpu_barrier(s);

  pde::g7::A2AHandles h{};
  h.world = kWorld;
  for (int r = 0; r < kWorld; ++r) {
    h.send_cnt_peer[r] = s->send_cnt_ptr[r];
    h.recv_buf_peer[r] = s->recv_buf_ptr[r];
    h.recv_meta_peer[r] = s->recv_meta_ptr[r];
    h.combine_peer[r] = s->combine_ptr[r];
  }

  // Cross-rank barrier flags (uint64[world], P2P-shared) — same as G6.
  unsigned long long* my_flags = nullptr;
  CUDA_CHECK(cudaMalloc(&my_flags, sizeof(unsigned long long) * kWorld));
  CUDA_CHECK(cudaMemset(my_flags, 0, sizeof(unsigned long long) * kWorld));
  s->flag_ptr[rank] = my_flags;
  s->flag_ready.fetch_add(1);
  while (s->flag_ready.load() < kWorld) std::this_thread::yield();
  cpu_barrier(s);
  pde::g7::RankBarrier bar{};
  bar.rank = rank;
  bar.world = kWorld;
  for (int r = 0; r < kWorld; ++r) bar.peer_flags[r] = s->flag_ptr[r];

  // Intra-grid barrier globals.
  unsigned int *gb_arrive, *gb_sense;
  CUDA_CHECK(cudaMalloc(&gb_arrive, sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&gb_sense, sizeof(unsigned int)));
  pde::g7::GridBarrier gb{};
  gb.arrive = gb_arrive;
  gb.sense = gb_sense;

  // NCCL communicator (independent reference + perf baseline).
  if (rank == 0) {
    NCCL_CHECK(ncclGetUniqueId(&s->nccl_id));
    s->nccl_id_ready.store(1);
  }
  while (s->nccl_id_ready.load() == 0) std::this_thread::yield();
  cpu_barrier(s);
  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, kWorld, s->nccl_id, rank));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  int sm_count = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, rank));
  int grid = sm_count;  // 1 CTA/SM -> co-resident for the grid-wide barrier
  if (const char* g = getenv("PDE_G7_GRID")) {
    int gv = atoi(g);
    if (gv > 0 && gv <= sm_count) grid = gv;
  }
  gb.expected = grid;
  auto reset_gb = [&]() {
    CUDA_CHECK(cudaMemset(gb_arrive, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(gb_sense, 0, sizeof(unsigned int)));
  };

  // Host staging.
  int* h_tokens = (int*)malloc(sizeof(int) * (long)kMaxT * kMaxHidden);
  int* h_route = (int*)malloc(sizeof(int) * kMaxT);
  int* h_combine = (int*)malloc(sizeof(int) * (long)kMaxT * kMaxHidden);
  int* h_combine_ref = (int*)malloc(sizeof(int) * (long)kMaxT * kMaxHidden);
  int* h_matrix = (int*)malloc(sizeof(int) * kWorld * kWorld);
  float* dummy_out = nullptr;
  CUDA_CHECK(cudaMalloc(&dummy_out, sizeof(float) * 16));

  // NCCL a2a scratch (per rank): send buffer packed by dst, recv buffer.
  int* d_nccl_send; int* d_nccl_recv; int* d_nccl_combine;
  CUDA_CHECK(cudaMalloc(&d_nccl_send, sizeof(int) * (long)kMaxRecvCap * kMaxHidden));
  CUDA_CHECK(cudaMalloc(&d_nccl_recv, sizeof(int) * (long)kMaxRecvCap * kMaxHidden));
  CUDA_CHECK(cudaMalloc(&d_nccl_combine, sizeof(int) * (long)kMaxRecvCap * kMaxHidden));

  unsigned long long phase = 16;

  for (int sc = 0; sc < kNumSizes; ++sc) {
    int T = kSizes[sc].T;
    int hidden = kSizes[sc].hidden;
    h.hidden = hidden;
    h.recv_capacity = kMaxRecvCap;

    // Build routing + payload on host (deterministic), upload.
    for (int t = 0; t < T; ++t) h_route[t] = route_rank(rank, t);
    for (int t = 0; t < T; ++t)
      for (int c = 0; c < hidden; ++c)
        h_tokens[(long)t * hidden + c] = payload_val(rank, t, c);
    CUDA_CHECK(cudaMemcpy(d_route, h_route, sizeof(int) * T, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens, sizeof(int) * (long)T * hidden,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_matrix, 0, sizeof(int) * kWorld * kWorld));
    CUDA_CHECK(cudaMemset(d_combine, 0, sizeof(int) * (long)T * hidden));
    CUDA_CHECK(cudaMemset(d_recv_buf, 0, sizeof(int) * (long)kMaxRecvCap * hidden));
    CUDA_CHECK(cudaMemset(d_recv_meta, -1, sizeof(int) * kMaxRecvCap));
    CUDA_CHECK(cudaDeviceSynchronize());
    cpu_barrier(s);

    // ===================== CORRECTNESS =================================
    reset_gb();
    cpu_barrier(s);
    pde_g7_a2a_kernel<<<grid, kBlock, 0, stream>>>(
        h, rank, d_tokens, d_route, T, d_matrix + rank * kWorld, d_matrix,
        d_cursor, d_send_pack, d_send_pack_src, d_send_dst_of, d_recv_buf,
        d_recv_meta, d_expert_out, d_off, bar, gb, phase);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cpu_barrier(s);
    phase += 16;

    CUDA_CHECK(cudaMemcpy(h_combine, d_combine, sizeof(int) * (long)T * hidden,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_matrix, d_matrix, sizeof(int) * kWorld * kWorld,
                          cudaMemcpyDeviceToHost));

    // ---- INDEPENDENT ref #2: host gather-route-scatter ----
    // Every rank's token t -> route_rank -> expert_xform(payload) -> back to t.
    // Because the expert transform is element-wise and routing is a permutation
    // of (rank,t) ids, the combined result for OUR token t is simply
    // expert_xform(payload_val(rank,t,c)). (The a2a + scatter-back is identity
    // on token identity; only the transform changes the value.)
    for (int t = 0; t < T; ++t)
      for (int c = 0; c < hidden; ++c)
        h_combine_ref[(long)t * hidden + c] =
            expert_xform(payload_val(rank, t, c));
    int ok_host = 1;
    double dot = 0, na = 0, nb = 0;
    for (long e = 0; e < (long)T * hidden; ++e) {
      if (h_combine[e] != h_combine_ref[e]) ok_host = 0;
      dot += (double)h_combine[e] * h_combine_ref[e];
      na += (double)h_combine[e] * h_combine[e];
      nb += (double)h_combine_ref[e] * h_combine_ref[e];
    }
    double cos = dot / (sqrt(na) * sqrt(nb) + 1e-30);

    // ---- routing check: count matrix vs analytic router (no drop/dupe/misroute)
    // analytic send_count[rank][d] = #tokens of OUR rank routed to d.
    int an_row[kWorld]; for (int d = 0; d < kWorld; ++d) an_row[d] = 0;
    for (int t = 0; t < T; ++t) an_row[route_rank(rank, t)]++;
    int ok_route = 1;
    for (int d = 0; d < kWorld; ++d)
      if (h_matrix[rank * kWorld + d] != an_row[d]) ok_route = 0;
    // The DEVICE-published full matrix (every rank wrote its own row over P2P)
    // is in h_matrix. At rank 0, on the first size, dump the per-(src,dst) count
    // matrix to evidence the data-dependent variable counts + conservation
    // (sum of all sends == sum of all recvs == world*T, no drops/dupes).
    if (rank == 0 && sc == 0) {
      // compare device-published matrix to the analytic full matrix for ALL
      // (src,dst) (not just our row) — the device routing is correct globally.
      int gfull[kWorld][kWorld];
      int sum_send = 0;
      printf("\n--- per-(src,dst) count matrix [%s] (rows=src, cols=dst) ---\n",
             kSizes[sc].name);
      for (int sr = 0; sr < kWorld; ++sr) {
        printf("  src%d:", sr);
        int rowsum = 0;
        for (int d = 0; d < kWorld; ++d) {
          gfull[sr][d] = 0;
          for (int t = 0; t < T; ++t) if (route_rank(sr, t) == d) gfull[sr][d]++;
          int devv = h_matrix[sr * kWorld + d];
          printf(" %4d%s", devv, (devv == gfull[sr][d]) ? "" : "!MISMATCH!");
          rowsum += devv;
          sum_send += devv;
        }
        printf("   (src%d sends %d/%d)\n", sr, rowsum, T);
      }
      int sum_recv = 0;
      printf("  recv:");
      for (int d = 0; d < kWorld; ++d) {
        int col = 0; for (int sr = 0; sr < kWorld; ++sr) col += h_matrix[sr * kWorld + d];
        printf(" %4d", col); sum_recv += col;
      }
      printf("   (dst totals)\n");
      printf("  conservation: sum_send=%d sum_recv=%d expected=%d -> %s\n",
             sum_send, sum_recv, kWorld * T,
             (sum_send == kWorld * T && sum_recv == kWorld * T) ? "OK (no drop/dupe)"
                                                                : "VIOLATION");
      fflush(stdout);
    }

    // ---- INDEPENDENT ref #1: NCCL grouped ncclSend/ncclRecv a2a ----
    // Pack our tokens by dst into d_nccl_send using the SAME analytic offsets,
    // send variable counts to each peer, recv into d_nccl_recv, apply the expert
    // transform, then send the transformed tokens BACK to origin (combine), and
    // compare the NCCL-combined result to the in-kernel combine. Fully
    // independent of the in-kernel push path (uses NCCL transport + host packing).
    // send offsets (analytic, host):
    int send_base[kWorld]; int sb = 0;
    for (int d = 0; d < kWorld; ++d) { send_base[d] = sb; sb += an_row[d]; }
    // pack on host then upload (independent of device pack).
    {
      static thread_local std::vector<int> packed;
      packed.assign((size_t)sb * hidden, 0);
      int cur[kWorld]; for (int d = 0; d < kWorld; ++d) cur[d] = send_base[d];
      // also remember, per packed slot, origin token id for the combine return.
      static thread_local std::vector<int> packed_src; packed_src.assign(sb, 0);
      for (int t = 0; t < T; ++t) {
        int d = route_rank(rank, t);
        int slot = cur[d]++;
        packed_src[slot] = t;
        for (int c = 0; c < hidden; ++c)
          packed[(long)slot * hidden + c] = payload_val(rank, t, c);
      }
      CUDA_CHECK(cudaMemcpy(d_nccl_send, packed.data(),
                            sizeof(int) * (long)sb * hidden,
                            cudaMemcpyHostToDevice));
      // We need each peer's recv offsets for OUR data. Gather the full matrix:
      // recv from src s lands at column-prefix of column rank. Compute recv plan.
      // First everyone must agree on the matrix; we already have h_matrix (ours)
      // — but NCCL send needs the GLOBAL matrix. Exchange via NCCL allgather of
      // rows would work; simpler: recompute every rank's row analytically (we
      // know route_rank for all ranks deterministically).
      int gmat[kWorld][kWorld];
      for (int sr = 0; sr < kWorld; ++sr) {
        for (int d = 0; d < kWorld; ++d) gmat[sr][d] = 0;
        for (int t = 0; t < T; ++t) gmat[sr][route_rank(sr, t)]++;  // same T all ranks
      }
      // our recv: from each src s we receive gmat[s][rank] tokens; recv base for
      // src s = sum_{s'<s} gmat[s'][rank].
      int recv_base[kWorld]; int rb = 0;
      for (int sr = 0; sr < kWorld; ++sr) { recv_base[sr] = rb; rb += gmat[sr][rank]; }
      int recv_total_nccl = rb;
      // Grouped a2a: send our block for dst d, recv block from src s. NCCL
      // matches sends/recvs by (peer,order), so explicit dst-side offsets aren't
      // needed here — each recv lands at our recv_base[src].
      NCCL_CHECK(ncclGroupStart());
      for (int d = 0; d < kWorld; ++d) {
        int cnt = gmat[rank][d];
        if (cnt > 0 && d != rank)
          NCCL_CHECK(ncclSend(d_nccl_send + (long)send_base[d] * hidden,
                              (size_t)cnt * hidden, ncclInt32, d, comm, stream));
        if (cnt > 0 && d == rank)  // self copy
          CUDA_CHECK(cudaMemcpyAsync(
              d_nccl_recv + (long)recv_base[rank] * hidden,
              d_nccl_send + (long)send_base[rank] * hidden,
              sizeof(int) * (long)cnt * hidden, cudaMemcpyDeviceToDevice, stream));
      }
      for (int sr = 0; sr < kWorld; ++sr) {
        int cnt = gmat[sr][rank];
        if (cnt > 0 && sr != rank)
          NCCL_CHECK(ncclRecv(d_nccl_recv + (long)recv_base[sr] * hidden,
                              (size_t)cnt * hidden, ncclInt32, sr, comm, stream));
      }
      NCCL_CHECK(ncclGroupEnd());
      // expert transform on the NCCL-received tokens (in place into combine src).
      // do it on host for full independence from the device expert kernel.
      static thread_local std::vector<int> h_recv; h_recv.assign((size_t)recv_total_nccl * hidden, 0);
      CUDA_CHECK(cudaMemcpy(h_recv.data(), d_nccl_recv,
                            sizeof(int) * (long)recv_total_nccl * hidden,
                            cudaMemcpyDeviceToHost));
      for (long e = 0; e < (long)recv_total_nccl * hidden; ++e)
        h_recv[e] = expert_xform(h_recv[e]);
      CUDA_CHECK(cudaMemcpy(d_nccl_recv, h_recv.data(),
                            sizeof(int) * (long)recv_total_nccl * hidden,
                            cudaMemcpyHostToDevice));
      // combine: send transformed tokens BACK to origin. src s sent us
      // gmat[s][rank] tokens at recv_base[s]; we return them to s, which places
      // them at ITS send_base for dst=rank ... but origin needs them back at the
      // ORIGINAL token position. Easiest independent check: origin reconstructs
      // by the inverse pack. We send the contiguous block back; origin unpacks
      // using packed_src order. To keep it simple+independent, gather to origin
      // via ncclSend back to s and have s unpack with ITS packed_src.
      NCCL_CHECK(ncclGroupStart());
      for (int sr = 0; sr < kWorld; ++sr) {  // return to each source
        int cnt = gmat[sr][rank];
        if (cnt > 0 && sr != rank)
          NCCL_CHECK(ncclSend(d_nccl_recv + (long)recv_base[sr] * hidden,
                              (size_t)cnt * hidden, ncclInt32, sr, comm, stream));
        if (cnt > 0 && sr == rank)
          CUDA_CHECK(cudaMemcpyAsync(
              d_nccl_combine + (long)send_base[rank] * hidden,
              d_nccl_recv + (long)recv_base[rank] * hidden,
              sizeof(int) * (long)cnt * hidden, cudaMemcpyDeviceToDevice, stream));
      }
      // we receive back, from each dst d we sent to, our own gmat[rank][d] tokens
      // (they come back in the SAME order we sent = our packed order for d).
      for (int d = 0; d < kWorld; ++d) {
        int cnt = gmat[rank][d];
        if (cnt > 0 && d != rank)
          NCCL_CHECK(ncclRecv(d_nccl_combine + (long)send_base[d] * hidden,
                              (size_t)cnt * hidden, ncclInt32, d, comm, stream));
      }
      NCCL_CHECK(ncclGroupEnd());
      CUDA_CHECK(cudaStreamSynchronize(stream));
      // d_nccl_combine is in OUR packed order; unpack to token order via packed_src.
      static thread_local std::vector<int> h_nc; h_nc.assign((size_t)sb * hidden, 0);
      CUDA_CHECK(cudaMemcpy(h_nc.data(), d_nccl_combine,
                            sizeof(int) * (long)sb * hidden, cudaMemcpyDeviceToHost));
      int ok_nccl = 1;
      for (int slot = 0; slot < sb; ++slot) {
        int t = packed_src[slot];
        for (int c = 0; c < hidden; ++c) {
          int got = h_nc[(long)slot * hidden + c];
          if (got != h_combine[(long)t * hidden + c]) ok_nccl = 0;
        }
      }
      if (sc == 0) {
        s->corr_nccl_ok[rank] = ok_nccl;
        s->corr_host_ok[rank] = ok_host;
        s->corr_route_ok[rank] = ok_route;
        s->corr_cos[rank] = cos;
      }
    }
    cpu_barrier(s);

    // ===================== PERF =======================================
    auto time_persistent = [&](int inner, int comms_en, int mode,
                               int citers) -> double {
      reset_gb();
      cpu_barrier(s);
      pde_g7_persistent<<<grid, kBlock, 0, stream>>>(
          h, rank, d_tokens, d_route, T, d_matrix + rank * kWorld, d_matrix,
          d_cursor, d_send_pack, d_send_pack_src, d_send_dst_of, d_recv_buf,
          d_recv_meta, d_expert_out, d_off, bar, gb, phase, 8, comms_en,
          mode, citers, dummy_out);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(stream));
      phase += 8ull * 16ull + 64ull;
      reset_gb();
      cpu_barrier(s);
      cudaEvent_t p0, p1;
      CUDA_CHECK(cudaEventCreate(&p0));
      CUDA_CHECK(cudaEventCreate(&p1));
      CUDA_CHECK(cudaEventRecord(p0, stream));
      pde_g7_persistent<<<grid, kBlock, 0, stream>>>(
          h, rank, d_tokens, d_route, T, d_matrix + rank * kWorld, d_matrix,
          d_cursor, d_send_pack, d_send_pack_src, d_send_dst_of, d_recv_buf,
          d_recv_meta, d_expert_out, d_off, bar, gb, phase, inner, comms_en,
          mode, citers, dummy_out);
      CUDA_CHECK(cudaEventRecord(p1, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      phase += (unsigned long long)inner * 16ull + 64ull;
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, p0, p1));
      CUDA_CHECK(cudaEventDestroy(p0));
      CUDA_CHECK(cudaEventDestroy(p1));
      cpu_barrier(s);
      return (double)ms * 1000.0 / inner;  // us per inner round
    };

    // Full-pipeline a2a latency (8 stages, 8 barriers) and its decomposition.
    double t_ik = time_persistent(kIters, /*comms=*/1, /*mode=*/0, /*c=*/0);
    double t_bars = time_persistent(kIters, /*comms=*/0, /*mode=*/1, /*c=*/0);
    double t_nocomms = time_persistent(kIters, /*comms=*/0, /*mode=*/0, /*c=*/0);

    // Overlap (mode 2): isolate the dispatch ∥ compute window (2 barriers).
    //   d_only   = dispatch push alone in the window
    //   c_only   = independent compute alone, calibrated to ~match d_only
    //   both     = dispatch ∥ compute in the SAME window
    //   overlap factor = (window_floor + d_net + c_net) / both
    double w_floor = time_persistent(kIters, /*comms=*/0, /*mode=*/2, /*c=*/0);
    double d_only = time_persistent(kIters, /*comms=*/1, /*mode=*/2, /*c=*/0);
    double d_net = d_only - w_floor; if (d_net < 0) d_net = 0;
    int citers = 4000;
    double c_only = time_persistent(kIters, /*comms=*/0, /*mode=*/2, citers);
    double c_net = c_only - w_floor;
    if (c_net > 0.2 && d_net > 0.2) {
      citers = (int)(citers * (d_net / c_net));
      if (citers < 1) citers = 1;
      c_only = time_persistent(kIters, /*comms=*/0, /*mode=*/2, citers);
      c_net = c_only - w_floor;
    }
    double both = time_persistent(kIters, /*comms=*/1, /*mode=*/2, citers);
    double serial_win = w_floor + (d_net > 0 ? d_net : 0) + (c_net > 0 ? c_net : 0);
    double overlap = (both > 1e-6) ? serial_win / both : 1.0;

    // launch-boundary NCCL a2a baseline (dispatch + combine = 2 grouped a2a).
    // Reuse the analytic plan computed above is out of scope; recompute counts.
    int gmat[kWorld][kWorld];
    for (int sr = 0; sr < kWorld; ++sr) {
      for (int d = 0; d < kWorld; ++d) gmat[sr][d] = 0;
      for (int t = 0; t < T; ++t) gmat[sr][route_rank(sr, t)]++;
    }
    int sbase[kWorld], rbase[kWorld]; int ss = 0, rr = 0;
    for (int d = 0; d < kWorld; ++d) { sbase[d] = ss; ss += gmat[rank][d]; }
    for (int sr = 0; sr < kWorld; ++sr) { rbase[sr] = rr; rr += gmat[sr][rank]; }
    auto nccl_a2a_round = [&]() {
      // dispatch
      NCCL_CHECK(ncclGroupStart());
      for (int d = 0; d < kWorld; ++d) {
        int cnt = gmat[rank][d];
        if (cnt > 0 && d != rank)
          NCCL_CHECK(ncclSend(d_nccl_send + (long)sbase[d] * hidden,
                              (size_t)cnt * hidden, ncclInt32, d, comm, stream));
      }
      for (int sr = 0; sr < kWorld; ++sr) {
        int cnt = gmat[sr][rank];
        if (cnt > 0 && sr != rank)
          NCCL_CHECK(ncclRecv(d_nccl_recv + (long)rbase[sr] * hidden,
                              (size_t)cnt * hidden, ncclInt32, sr, comm, stream));
      }
      NCCL_CHECK(ncclGroupEnd());
      // combine (return)
      NCCL_CHECK(ncclGroupStart());
      for (int sr = 0; sr < kWorld; ++sr) {
        int cnt = gmat[sr][rank];
        if (cnt > 0 && sr != rank)
          NCCL_CHECK(ncclSend(d_nccl_recv + (long)rbase[sr] * hidden,
                              (size_t)cnt * hidden, ncclInt32, sr, comm, stream));
      }
      for (int d = 0; d < kWorld; ++d) {
        int cnt = gmat[rank][d];
        if (cnt > 0 && d != rank)
          NCCL_CHECK(ncclRecv(d_nccl_combine + (long)sbase[d] * hidden,
                              (size_t)cnt * hidden, ncclInt32, d, comm, stream));
      }
      NCCL_CHECK(ncclGroupEnd());
    };
    for (int it = 0; it < kWarmup; ++it) { cpu_barrier(s); nccl_a2a_round(); CUDA_CHECK(cudaStreamSynchronize(stream)); }
    cpu_barrier(s);
    cudaEvent_t q0, q1;
    CUDA_CHECK(cudaEventCreate(&q0));
    CUDA_CHECK(cudaEventCreate(&q1));
    CUDA_CHECK(cudaEventRecord(q0, stream));
    for (int it = 0; it < kIters; ++it) nccl_a2a_round();
    CUDA_CHECK(cudaEventRecord(q1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float qms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&qms, q0, q1));
    CUDA_CHECK(cudaEventDestroy(q0));
    CUDA_CHECK(cudaEventDestroy(q1));
    double t_nccl = (double)qms * 1000.0 / kIters;
    cpu_barrier(s);

    if (rank == 0) {
      // bytes moved per rank in dispatch+combine (send + recv of variable data).
      double sent = 0, recvd = 0;
      for (int d = 0; d < kWorld; ++d) if (d != 0) sent += (double)gmat[0][d] * hidden * sizeof(int);
      for (int sr = 0; sr < kWorld; ++sr) if (sr != 0) recvd += (double)gmat[sr][0] * hidden * sizeof(int);
      double bytes_round = (sent + recvd) * 2.0;  // *2 = dispatch + combine
      double move_net = t_ik - t_nocomms; if (move_net < 1e-3) move_net = t_ik;
      double bw_move = bytes_round / (move_net * 1e-6) / 1e9;
      double bw_ik = bytes_round / (t_ik * 1e-6) / 1e9;
      double bw_nccl = bytes_round / (t_nccl * 1e-6) / 1e9;
      printf(
          "[%-11s h=%4d] ik=%7.2fus nccl=%7.2fus speedup=%5.2fx | ov=%4.2fx | "
          "[bars=%6.2f nocomms=%6.2f move_net=%6.2f]us | BW_move=%6.1f "
          "BW_ik=%5.1f BW_nccl=%5.1f GB/s\n",
          kSizes[sc].name, hidden, t_ik, t_nccl, t_nccl / t_ik, overlap, t_bars,
          t_nocomms, move_net, bw_move, bw_ik, bw_nccl);
      printf("              overlap-window: floor=%5.2f dispatch_net=%5.2f "
             "compute_net=%5.2f both=%5.2f -> serial=%5.2f ov=%4.2fx\n",
             w_floor, d_net, c_net, both, serial_win, overlap);
      fflush(stdout);
    }
  }

  // gather full matrix at rank 0 for a conservation print (no drop/dupe global).
  cpu_barrier(s);

  NCCL_CHECK(ncclCommDestroy(comm));
  CUDA_CHECK(cudaStreamDestroy(stream));
  cudaFree(d_tokens); cudaFree(d_route); cudaFree(d_send_row); cudaFree(d_matrix);
  cudaFree(d_cursor); cudaFree(d_send_pack); cudaFree(d_send_pack_src);
  cudaFree(d_send_dst_of); cudaFree(d_recv_buf); cudaFree(d_recv_meta);
  cudaFree(d_expert_out); cudaFree(d_combine); cudaFree(d_off);
  cudaFree(my_flags); cudaFree(gb_arrive); cudaFree(gb_sense); cudaFree(dummy_out);
  cudaFree(d_nccl_send); cudaFree(d_nccl_recv); cudaFree(d_nccl_combine);
  free(h_tokens); free(h_route); free(h_combine); free(h_combine_ref); free(h_matrix);
}

int main() {
  int n = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n));
  printf("=== PDE G7 in-kernel EP=%d all-to-all dispatch+combine (tier b: "
         "P2P-NVLink) ===\n", kWorld);
  printf("visible_devices=%d block=%d experts=%d grid=1CTA/SM\n", n, kBlock,
         kExperts);
  if (n < kWorld) { fprintf(stderr, "need %d GPUs, have %d\n", kWorld, n); return 2; }

  Shared* s = new Shared();
  std::vector<std::thread> threads;
  for (int r = 0; r < kWorld; ++r) threads.emplace_back(worker, r, s);
  for (auto& t : threads) t.join();

  printf("\n=== CORRECTNESS (per rank, smallest size, vs INDEPENDENT refs) ===\n");
  int all_ok = 1;
  for (int r = 0; r < kWorld; ++r) {
    int ok = s->corr_nccl_ok[r] && s->corr_host_ok[r] && s->corr_route_ok[r];
    all_ok &= ok;
    printf("rank %d: inkernel==NCCL_a2a: %-3s  inkernel==host: %-3s  "
           "route_matrix_ok: %-3s  cos_vs_host=%.9f -> %s\n",
           r, s->corr_nccl_ok[r] ? "YES" : "NO", s->corr_host_ok[r] ? "YES" : "NO",
           s->corr_route_ok[r] ? "YES" : "NO", s->corr_cos[r],
           ok ? "PASS" : "FAIL");
  }
  printf("\n=== GATE: %s ===\n", all_ok ? "CORRECTNESS PASS" : "CORRECTNESS FAIL");
  delete s;
  return all_ok ? 0 : 1;
}

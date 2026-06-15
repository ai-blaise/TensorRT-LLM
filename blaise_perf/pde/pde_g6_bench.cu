// PDE G6 — in-kernel TP=4 all-reduce microbench (model-free, GPU 0-3).
//
// TIER REACHED: (b) P2P-NVLink in-kernel one-shot all-reduce.
//   Tier (a) NVLS `multimem` is environment-blocked on this VM: every GPU
//   reports multicast_supported=1 and `multimem` PTX assembles, but
//   cuMulticastBindMem returns CUDA_ERROR_INVALID_VALUE because the NVLink-
//   fabric multicast team cannot form without nv-fabricmanager + IMEX channels,
//   which are not exposed in this GPU-passthrough VM (lspci shows 0 NVSwitch,
//   no /dev/nvidia-caps-imex-channels). See pde_g6_results.md.
//
// One process, `world` host threads (one per GPU, GPU 0..world-1). Each thread
// owns its device and a persistent kernel. P2P access is enabled between all
// pairs, so each rank can directly dereference its peers' input buffers over
// NVLink. The persistent decode-shaped kernel:
//   - COMPUTE warps run independent FLOP-y work (stand-in for the rest of the
//     decode step) — what comms must overlap with;
//   - COMMS warps do an IN-KERNEL all-reduce: read every peer's input over
//     NVLink and sum, bracketed by a cross-rank barrier (system-scope atomics
//     over P2P flags). On return every rank holds the element-wise sum across
//     all `world` ranks, never having left the kernel.
//
// GATE
//   CORRECTNESS (hard): in-kernel result == INDEPENDENT reference on all ranks.
//     Two independent refs: (i) NCCL all-reduce, (ii) analytic gather-sum of the
//     known integer inputs. Integer test data => bit-exact.
//   PERF: in-kernel all-reduce latency vs LAUNCH-BOUNDARY NCCL all-reduce (the
//     baseline the engine replaces), at representative decode sizes; reports
//     in-kernel latency, NCCL baseline, comms∥compute overlap factor, achieved
//     NVLink bus bandwidth.
//
// Build/run: see build_run_g6.sh.

#include "pde_g6_allreduce.cuh"

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

static constexpr int kWorld = 4;   // TP=4, GPU 0..3
static constexpr int kBlock = 256;
static constexpr int kIters = 100;
static constexpr int kWarmup = 20;

// Representative decode all-reduce sizes (hidden 7168), element counts (f32).
struct SizeCase { const char* name; int n_f32; };
static const SizeCase kSizes[] = {
    {"b1_h7168", 7168 * 1},
    {"b2_h7168", 7168 * 2},
    {"b4_h7168", 7168 * 4},
    {"b8_h7168", 7168 * 8},
    {"b16_h7168", 7168 * 16},
    {"b32_h7168", 7168 * 32},
};
static constexpr int kNumSizes = sizeof(kSizes) / sizeof(kSizes[0]);
static constexpr int kMaxN = 7168 * 32;

// ---------------------------------------------------------------------------
// Persistent decode-shaped kernel.
//
// warps [n_warps-n_comms .. n_warps-1] = COMMS; the rest = COMPUTE.
//   mode 0 = comms only (pure in-kernel all-reduce latency)
//   mode 1 = compute only (the independent compute alone)
//   mode 2 = overlapped (both)  -> overlap factor = (t0+t1)/t2
// ---------------------------------------------------------------------------
__global__ void pde_g6_kernel(pde::g6::PeerPtrs pp, float* my_out, int n_f32,
                              float* dummy_out, int compute_iters, int mode,
                              pde::g6::RankBarrier bar, pde::g6::GridBarrier gb,
                              unsigned long long phase, int n_comms_warps) {
  // Correctness kernel: ONE grid-wide cross-rank all-reduce. All CTAs converge
  // at the barrier; comms warps reduce; all converge again. Validated vs NCCL.
  const int tid_in_cta = threadIdx.x;
  const int warp = tid_in_cta / 32;
  const int n_warps = blockDim.x / 32;
  const bool is_comms = (warp >= n_warps - n_comms_warps);
  int local_sense = 0;
  (void)compute_iters; (void)mode; (void)dummy_out;

  pde::g6::cross_rank_grid_barrier(bar, gb, &local_sense, phase + 0);
  if (is_comms) {
    int comms_warp_idx = warp - (n_warps - n_comms_warps);
    int lane = tid_in_cta % 32;
    int comms_tid =
        blockIdx.x * (n_comms_warps * 32) + comms_warp_idx * 32 + lane;
    int comms_nthreads = gridDim.x * n_comms_warps * 32;
    pde::g6::one_shot_all_reduce_f32_body(pp, my_out, n_f32, comms_tid,
                                          comms_nthreads);
  }
  pde::g6::cross_rank_grid_barrier(bar, gb, &local_sense, phase + 1);
}

// ---------------------------------------------------------------------------
// PERSISTENT-LOOP kernel: ONE launch performs `inner_iters` in-kernel all-
// reduces in a device-side loop, with NO relaunch between them. This is the
// engine-representative measurement — in the real persistent decode megakernel
// the TP all-reduce happens mid-step and never pays a kernel launch. Each
// iteration is barrier -> peer-read-sum -> barrier; the grid stays resident.
// `base_phase` must be unique per call so flag values keep increasing.
// ---------------------------------------------------------------------------
__global__ void pde_g6_persistent(pde::g6::PeerPtrs pp, float* my_out, int n_f32,
                                  float* dummy_out, pde::g6::RankBarrier bar,
                                  pde::g6::GridBarrier gb,
                                  unsigned long long base_phase,
                                  int n_comms_warps, int inner_iters,
                                  int reduce_enable, int two_barriers,
                                  int compute_iters) {
  const int tid_in_cta = threadIdx.x;
  const int warp = tid_in_cta / 32;
  const int n_warps = blockDim.x / 32;
  const bool is_comms = (warp >= n_warps - n_comms_warps);
  int comms_warp_idx = warp - (n_warps - n_comms_warps);
  int lane = tid_in_cta % 32;
  int comms_tid = blockIdx.x * (n_comms_warps * 32) + comms_warp_idx * 32 + lane;
  int comms_nthreads = gridDim.x * n_comms_warps * 32;
  int compute_warps_per_cta = n_warps - n_comms_warps;
  int gtid = blockIdx.x * (compute_warps_per_cta * 32) + warp * 32 + lane;
  int local_sense = 0;
  float cacc = 1.0f + 1e-3f * (gtid & 1023);
  const float cb = 0.9999f;

  for (int it = 0; it < inner_iters; ++it) {
    unsigned long long ph = base_phase + (unsigned long long)it;
    // Pre-barrier: grid-wide + cross-rank — every CTA on every rank converges
    // and peers' published inputs are visible before any comms warp reads them.
    pde::g6::cross_rank_grid_barrier(bar, gb, &local_sense, ph * 2 + 0);
    // ----- comms ∥ compute region (between the two barriers) -----
    if (reduce_enable && is_comms) {
      pde::g6::one_shot_all_reduce_f32_body(pp, my_out, n_f32, comms_tid,
                                            comms_nthreads);
    }
    if (compute_iters > 0 && !is_comms) {
      // independent FMA work that overlaps the all-reduce (the rest of decode)
#pragma unroll 1
      for (int i = 0; i < compute_iters; ++i) {
        cacc = fmaf(cacc, cb, 1e-7f);
        cacc = fmaf(cacc, cb, 1e-7f);
      }
    }
    if (two_barriers) {
      // Post-barrier: keep peer inputs alive until all ranks done reading.
      pde::g6::cross_rank_grid_barrier(bar, gb, &local_sense, ph * 2 + 1);
    }
  }
  if (cacc == -123456.0f && dummy_out) dummy_out[gtid % n_f32] = cacc;  // live
}

// ---------------------------------------------------------------------------
// Cross-thread shared handoff.
// ---------------------------------------------------------------------------
struct Shared {
  std::atomic<int> arrive{0};
  std::atomic<int> sense{0};
  float* in_ptr[kWorld];                 // each rank's unicast input buffer
  std::atomic<int> in_ready{0};
  unsigned long long* flag_ptr[kWorld];  // each rank's barrier flag array
  std::atomic<int> flag_ready{0};
  ncclUniqueId nccl_id;
  std::atomic<int> nccl_id_ready{0};
  int corr_nccl_ok[kWorld];
  int corr_sum_ok[kWorld];
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

  // Enable P2P to all peers (precondition for in-kernel peer reads + NCCL NVL).
  for (int r = 0; r < kWorld; ++r) {
    if (r == rank) continue;
    int can = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can, rank, r));
    if (!can) { fprintf(stderr, "rank %d cannot P2P to %d\n", rank, r); std::abort(); }
    cudaError_t e = cudaDeviceEnablePeerAccess(r, 0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
      fprintf(stderr, "rank %d enablePeer %d: %s\n", rank, r,
              cudaGetErrorString(e));
    }
    cudaGetLastError();
  }

  // This rank's input + output buffers (own device memory).
  float* my_in = nullptr;
  float* my_out = nullptr;
  CUDA_CHECK(cudaMalloc(&my_in, sizeof(float) * kMaxN));
  CUDA_CHECK(cudaMalloc(&my_out, sizeof(float) * kMaxN));
  s->in_ptr[rank] = my_in;
  s->in_ready.fetch_add(1);
  while (s->in_ready.load() < kWorld) std::this_thread::yield();
  cpu_barrier(s);

  // Peer input pointers (directly dereferenceable over NVLink after P2P enable).
  pde::g6::PeerPtrs pp{};
  pp.world = kWorld;
  for (int r = 0; r < kWorld; ++r) pp.in[r] = s->in_ptr[r];

  // Cross-rank barrier flags: each rank allocs uint64[kWorld], P2P-shared.
  unsigned long long* my_flags = nullptr;
  CUDA_CHECK(cudaMalloc(&my_flags, sizeof(unsigned long long) * kWorld));
  CUDA_CHECK(cudaMemset(my_flags, 0, sizeof(unsigned long long) * kWorld));
  s->flag_ptr[rank] = my_flags;
  s->flag_ready.fetch_add(1);
  while (s->flag_ready.load() < kWorld) std::this_thread::yield();
  cpu_barrier(s);
  pde::g6::RankBarrier bar{};
  bar.rank = rank;
  bar.world = kWorld;
  for (int r = 0; r < kWorld; ++r) bar.peer_flags[r] = s->flag_ptr[r];

  // Intra-grid barrier globals (single-GPU, all-CTA). Reset to 0 before each
  // launch so local_sense (=0 at kernel entry) matches the global sense.
  unsigned int* gb_arrive = nullptr;
  unsigned int* gb_sense = nullptr;
  CUDA_CHECK(cudaMalloc(&gb_arrive, sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&gb_sense, sizeof(unsigned int)));
  pde::g6::GridBarrier gb{};
  gb.arrive = gb_arrive;
  gb.sense = gb_sense;
  // gb.expected set after grid size is known (below).

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

  float* dummy_out = nullptr;
  CUDA_CHECK(cudaMalloc(&dummy_out, sizeof(float) * kMaxN));
  float* nccl_buf = nullptr;
  CUDA_CHECK(cudaMalloc(&nccl_buf, sizeof(float) * kMaxN));
  float* host_in = (float*)malloc(sizeof(float) * kMaxN);
  float* host_res = (float*)malloc(sizeof(float) * kMaxN);
  float* host_nccl = (float*)malloc(sizeof(float) * kMaxN);

  int sm_count = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                    rank));
  int grid = sm_count;  // 1 CTA/SM -> all CTAs guaranteed co-resident for the
                        // grid-wide barrier (no deadlock).
  if (const char* g = getenv("PDE_G6_GRID")) {
    int gv = atoi(g);
    if (gv > 0 && gv <= sm_count) grid = gv;  // clamp: must stay co-resident
  }
  gb.expected = grid;
  const int n_warps = kBlock / 32;
  // Half the warps drive comms (peer NVLink reads + sum), half are free for the
  // overlapped compute. Sweep (PDE_G6_COMMS_WARPS) showed 4/8 gives near-NCCL
  // small-message latency + ~57 GB/s reduce BW while preserving overlap room.
  int n_comms_warps = n_warps / 2;
  if (const char* c = getenv("PDE_G6_COMMS_WARPS")) {
    int cv = atoi(c);
    if (cv >= 1 && cv < n_warps) n_comms_warps = cv;
  }
  auto reset_gb = [&]() {
    CUDA_CHECK(cudaMemset(gb_arrive, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(gb_sense, 0, sizeof(unsigned int)));
  };

  // input[i] on rank r = (r+1)*((i%64)+1); sum over 4 ranks = 10*((i%64)+1),
  // all exact integers in f32 -> bit-exact all-reduce.
  auto fill_inputs = [&](int n) {
    for (int i = 0; i < n; ++i)
      host_in[i] = (float)((rank + 1) * ((i % 64) + 1));
  };

  unsigned long long phase = 0;

  for (int sc = 0; sc < kNumSizes; ++sc) {
    int n = kSizes[sc].n_f32;

    // ===================== CORRECTNESS =================================
    fill_inputs(n);
    CUDA_CHECK(cudaMemcpy(my_in, host_in, sizeof(float) * n,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(my_out, 0, sizeof(float) * n));
    CUDA_CHECK(cudaDeviceSynchronize());
    cpu_barrier(s);

    reset_gb();
    cpu_barrier(s);
    pde_g6_kernel<<<grid, kBlock, 0, stream>>>(pp, my_out, n, dummy_out,
                                               /*citers=*/0, /*mode=*/0, bar, gb,
                                               phase, n_comms_warps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cpu_barrier(s);
    phase += 4;

    CUDA_CHECK(cudaMemcpy(host_res, my_out, sizeof(float) * n,
                          cudaMemcpyDeviceToHost));

    // INDEPENDENT ref #1: NCCL all-reduce of the SAME inputs.
    CUDA_CHECK(cudaMemcpy(nccl_buf, host_in, sizeof(float) * n,
                          cudaMemcpyHostToDevice));
    cpu_barrier(s);
    NCCL_CHECK(ncclAllReduce(nccl_buf, nccl_buf, n, ncclFloat32, ncclSum, comm,
                             stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(host_nccl, nccl_buf, sizeof(float) * n,
                          cudaMemcpyDeviceToHost));

    // Compare: in-kernel vs NCCL (bit-exact), and vs analytic gather-sum.
    int ok_nccl = 1, ok_sum = 1;
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) {
      float expect = (float)(10 * ((i % 64) + 1));
      if (host_res[i] != host_nccl[i]) ok_nccl = 0;
      if (host_res[i] != expect) ok_sum = 0;
      dot += (double)host_res[i] * host_nccl[i];
      na += (double)host_res[i] * host_res[i];
      nb += (double)host_nccl[i] * host_nccl[i];
    }
    double cos = dot / (sqrt(na) * sqrt(nb) + 1e-30);
    if (sc == 0) {
      s->corr_nccl_ok[rank] = ok_nccl;
      s->corr_sum_ok[rank] = ok_sum;
      s->corr_cos[rank] = cos;
    }
    cpu_barrier(s);

    // ===================== PERF =======================================
    // All in-kernel timings use the PERSISTENT kernel: ONE launch performs
    // `inner` in-kernel all-reduces in a device loop, zero per-call relaunch —
    // the engine-relevant cost (the megakernel never relaunches mid-decode-step).
    // Wall-clock / inner. reduce_en/two_bar/citers decompose the cost.
    auto time_persistent = [&](int inner, int reduce_en, int two_bar,
                               int citers) -> double {
      reset_gb();
      cpu_barrier(s);
      pde_g6_persistent<<<grid, kBlock, 0, stream>>>(
          pp, my_out, n, dummy_out, bar, gb, phase, n_comms_warps, 16, reduce_en,
          two_bar, citers);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(stream));
      phase += 16 * 2 + 8;
      reset_gb();
      cpu_barrier(s);
      cudaEvent_t p0, p1;
      CUDA_CHECK(cudaEventCreate(&p0));
      CUDA_CHECK(cudaEventCreate(&p1));
      CUDA_CHECK(cudaEventRecord(p0, stream));
      pde_g6_persistent<<<grid, kBlock, 0, stream>>>(
          pp, my_out, n, dummy_out, bar, gb, phase, n_comms_warps, inner,
          reduce_en, two_bar, citers);
      CUDA_CHECK(cudaEventRecord(p1, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      phase += inner * 2 + 8;
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, p0, p1));
      CUDA_CHECK(cudaEventDestroy(p0));
      CUDA_CHECK(cudaEventDestroy(p1));
      cpu_barrier(s);
      return (double)ms * 1000.0 / inner;  // us per inner all-reduce
    };

    // in-kernel all-reduce latency (2 barriers + reduce, no compute)
    double t_persist = time_persistent(kIters, /*reduce=*/1, /*2bar=*/1, /*c=*/0);
    // decomposition
    double t_bar1 = time_persistent(kIters, /*reduce=*/0, /*2bar=*/0, /*c=*/0);
    double t_bar2 = time_persistent(kIters, /*reduce=*/0, /*2bar=*/1, /*c=*/0);
    // compute-only (calibrate to ~match the reduce+2bar latency for overlap demo)
    int citers = 4000;
    double t_comp = time_persistent(kIters, /*reduce=*/0, /*2bar=*/1, citers);
    double t_comp_net = t_comp - t_bar2;  // compute beyond the two barriers
    if (t_comp_net > 0.3) {
      double target = t_persist - t_bar2;  // the reduce's net cost to hide
      if (target < 0.3) target = t_persist;  // fall back to full latency
      citers = (int)(citers * (target / t_comp_net));
      if (citers < 1) citers = 1;
      t_comp = time_persistent(kIters, /*reduce=*/0, /*2bar=*/1, citers);
    }
    // overlapped: reduce ∥ compute, both between the two barriers
    double t_overlap = time_persistent(kIters, /*reduce=*/1, /*2bar=*/1, citers);
    // overlap factor: how much of (reduce + compute) is hidden by running them
    // concurrently. serial = t_persist + (t_comp - t_bar2); overlapped = t_overlap.
    double serial = t_persist + (t_comp - t_bar2);
    double overlap = serial / t_overlap;

    // launch-boundary NCCL baseline (what the engine replaces).
    for (int it = 0; it < kWarmup; ++it) {
      cpu_barrier(s);
      NCCL_CHECK(ncclAllReduce(nccl_buf, nccl_buf, n, ncclFloat32, ncclSum, comm,
                               stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    cpu_barrier(s);
    cudaEvent_t q0, q1;
    CUDA_CHECK(cudaEventCreate(&q0));
    CUDA_CHECK(cudaEventCreate(&q1));
    CUDA_CHECK(cudaEventRecord(q0, stream));
    for (int it = 0; it < kIters; ++it) {
      NCCL_CHECK(ncclAllReduce(nccl_buf, nccl_buf, n, ncclFloat32, ncclSum, comm,
                               stream));
    }
    CUDA_CHECK(cudaEventRecord(q1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float qms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&qms, q0, q1));
    CUDA_CHECK(cudaEventDestroy(q0));
    CUDA_CHECK(cudaEventDestroy(q1));
    double t_nccl = (double)qms * 1000.0 / kIters;
    cpu_barrier(s);

    if (rank == 0) {
      double bytes = (double)n * sizeof(float);
      double busfac = 2.0 * (kWorld - 1) / kWorld;  // all-reduce algBW->busBW
      // reduce-net BW (subtract the fixed 2-barrier cost so BW reflects the data
      // movement, not the sync floor).
      double reduce_net = t_persist - t_bar2;
      if (reduce_net < 1e-3) reduce_net = t_persist;
      double bw_red = busfac * bytes / (reduce_net * 1e-6) / 1e9;  // GB/s
      double bw_persist = busfac * bytes / (t_persist * 1e-6) / 1e9;
      double bw_nccl = busfac * bytes / (t_nccl * 1e-6) / 1e9;
      printf(
          "[%-10s %7.1fKB] ik=%7.2fus nccl=%7.2fus speedup=%5.2fx | ov=%4.2fx | "
          "[bar1=%5.2f bar2=%5.2f reduce_net=%6.2f]us | BW_red=%6.1f "
          "BW_ik=%6.1f BW_nccl=%6.1f GB/s\n",
          kSizes[sc].name, bytes / 1024.0, t_persist, t_nccl,
          t_nccl / t_persist, overlap, t_bar1, t_bar2, reduce_net, bw_red,
          bw_persist, bw_nccl);
      fflush(stdout);
    }
  }

  NCCL_CHECK(ncclCommDestroy(comm));
  CUDA_CHECK(cudaStreamDestroy(stream));
  cudaFree(my_in);
  cudaFree(my_out);
  cudaFree(my_flags);
  cudaFree(gb_arrive);
  cudaFree(gb_sense);
  cudaFree(dummy_out);
  cudaFree(nccl_buf);
  free(host_in);
  free(host_res);
  free(host_nccl);
}

int main() {
  int n = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n));
  printf("=== PDE G6 in-kernel TP=%d all-reduce (tier b: P2P-NVLink) ===\n",
         kWorld);
  printf("visible_devices=%d block=%d comms_warps=%d/%d grid=1CTA/SM\n", n,
         kBlock, (kBlock / 32) / 2, kBlock / 32);
  if (n < kWorld) {
    fprintf(stderr, "need %d GPUs, have %d\n", kWorld, n);
    return 2;
  }

  Shared* s = new Shared();
  std::vector<std::thread> threads;
  for (int r = 0; r < kWorld; ++r) threads.emplace_back(worker, r, s);
  for (auto& t : threads) t.join();

  printf("\n=== CORRECTNESS (per rank, smallest size, vs INDEPENDENT refs) ===\n");
  int all_ok = 1;
  for (int r = 0; r < kWorld; ++r) {
    int ok = s->corr_nccl_ok[r] && s->corr_sum_ok[r];
    all_ok &= ok;
    printf(
        "rank %d: inkernel==NCCL: %-3s  inkernel==analytic_sum: %-3s  "
        "cos_vs_nccl=%.9f -> %s\n",
        r, s->corr_nccl_ok[r] ? "YES" : "NO",
        s->corr_sum_ok[r] ? "YES" : "NO", s->corr_cos[r],
        ok ? "PASS" : "FAIL");
  }
  printf("\n=== GATE: %s ===\n",
         all_ok ? "CORRECTNESS PASS" : "CORRECTNESS FAIL");
  delete s;
  return all_ok ? 0 : 1;
}

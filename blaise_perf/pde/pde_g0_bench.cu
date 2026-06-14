// PDE G0 gate microbench (standalone, sm_100). Real numbers on GPU0.
//
// Validates the substrate and measures the G0 bar:
//   Correctness A: K-region cross-grid-barrier reduction == CPU reference,
//                  for BOTH barrier impls (cg.grid_sync and hand-rolled global).
//   Correctness B: work-queue distributes M items, each processed exactly once
//                  (checksum == CPU reference).
//   Perf:          cost of K separate trivial kernel LAUNCHES  vs  K in-kernel
//                  grid-barriers in ONE persistent cooperative launch.
//                  PASS iff in-kernel barrier per-boundary cost < launch cost.
//
// Build: nvcc -std=c++17 -arch=sm_100 -O3 -o pde_g0_bench pde_g0_bench.cu
#include "pde_substrate.cuh"

#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cg = cooperative_groups;
using namespace pde;

#define CK(x)                                                              \
  do {                                                                     \
    cudaError_t e_ = (x);                                                  \
    if (e_ != cudaSuccess) {                                              \
      printf("CUDA_ERR %s:%d %s -> %s\n", __FILE__, __LINE__, #x,         \
             cudaGetErrorString(e_));                                     \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

// ===========================================================================
// Shared per-region work: a cross-grid dependency. Region r writes scratch[i],
// region r+1 reads ALL of region r's output via a strided partial sum. This is
// a real multi-stage reduction: a wrong barrier (missing a peer CTA's writes)
// changes the result, so EXACT match vs CPU proves the barrier.
// ===========================================================================

// At region r each thread updates its own cell from the GRID-WIDE running sum
// of the previous region. acc accumulates a checksum of the full vector each
// region so cross-CTA visibility is required for correctness.
__device__ __forceinline__ void region_step(int* scratch, long long* acc,
                                             int n, int region,
                                             int tid, int nthreads,
                                             unsigned long long* gsum) {
  // Phase 1: each thread folds the previous region's global sum into its cells.
  unsigned long long prev = *gsum;  // read prev region's grid-wide sum
  for (int i = tid; i < n; i += nthreads) {
    scratch[i] = (int)((scratch[i] + (int)(prev & 0x3FF) + region * 7 + i) & 0x7FFFFFFF);
  }
}

// CPU reference for the K-region reduction (bit-exact mirror of the device math).
static long long cpu_reference_reduction(int n, int K, std::vector<int>& vec) {
  // gsum starts 0 (matches device init). Each region: fold, then recompute gsum.
  unsigned long long gsum = 0;
  for (int r = 0; r < K; ++r) {
    unsigned long long prev = gsum;
    for (int i = 0; i < n; ++i) {
      vec[i] = (int)((vec[i] + (int)(prev & 0x3FF) + r * 7 + i) & 0x7FFFFFFF);
    }
    unsigned long long s = 0;
    for (int i = 0; i < n; ++i) s += (unsigned long long)vec[i];
    gsum = s;
  }
  long long checksum = 0;
  for (int i = 0; i < n; ++i) checksum += vec[i];
  return checksum;
}

// ---------------------------------------------------------------------------
// Correctness A, impl (a): cooperative-groups grid.sync().
// Persistent kernel: K regions, grid barrier between each. After each region,
// CTA 0 recomputes the grid-wide sum into gsum (guarded by a grid barrier so
// every CTA's writes are visible first). gsum feeds the next region.
// ---------------------------------------------------------------------------
__global__ void kReductionCG(int* scratch, int n, int K,
                             unsigned long long* gsum, long long* out) {
  cg::grid_group grid = cg::this_grid();
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (int r = 0; r < K; ++r) {
    region_step(scratch, nullptr, n, r, tid, nthreads, gsum);
    grid.sync();  // all region-r writes globally visible
    // CTA 0 thread 0 recomputes the grid-wide sum for region r.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      unsigned long long s = 0;
      for (int i = 0; i < n; ++i) s += (unsigned long long)scratch[i];
      *gsum = s;
    }
    grid.sync();  // new gsum visible to all before region r+1 reads it
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    long long c = 0;
    for (int i = 0; i < n; ++i) c += scratch[i];
    *out = c;
  }
}

// ---------------------------------------------------------------------------
// Correctness A, impl (b): hand-rolled global barrier.
// Same structure, substituting GlobalBarrier for grid.sync(). NOT a cooperative
// launch requirement for correctness of the barrier itself, but we launch it
// cooperatively too so all CTAs are co-resident (a spinning global barrier
// deadlocks if some CTAs are not resident).
// ---------------------------------------------------------------------------
__global__ void kReductionGlobal(int* scratch, int n, int K,
                                 unsigned long long* gsum, long long* out,
                                 GlobalBarrier gb) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;
  int sense = 0;

  for (int r = 0; r < K; ++r) {
    region_step(scratch, nullptr, n, r, tid, nthreads, gsum);
    global_barrier_sync(gb, &sense);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      unsigned long long s = 0;
      for (int i = 0; i < n; ++i) s += (unsigned long long)scratch[i];
      *gsum = s;
    }
    global_barrier_sync(gb, &sense);
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    long long c = 0;
    for (int i = 0; i < n; ++i) c += scratch[i];
    *out = c;
  }
}

// ---------------------------------------------------------------------------
// Correctness B: atomic work-queue. M items; the resident grid pops ids and
// records ownership + adds the item value to a global checksum exactly once.
// `times_processed[i]` must be exactly 1 for all i; checksum must match CPU.
// One thread per pop (atomicAdd on head gives a total order) -> no item handed
// out twice, none dropped (every thread loops until the queue drains).
// ---------------------------------------------------------------------------
__global__ void kWorkQueue(WorkQueue q, const int* values,
                           int* times_processed, unsigned long long* checksum) {
  unsigned int item;
  while (work_queue_pop(q, &item)) {
    atomicAdd(&times_processed[item], 1);
    atomicAdd(checksum, (unsigned long long)values[item]);
  }
}

// ---------------------------------------------------------------------------
// Correctness C: intra-CTA producer->consumer SMEM mailbox via mbarrier.
// Launched with EXACTLY 2 warps (64 threads): warp 0 = producer, warp 1 =
// consumer. Two barriers, each with arrival count = 64 (all threads):
//   `ready`    orders producer's tile store BEFORE the consumer's tile read.
//   `consumed` orders the consumer's read BEFORE the producer overwrites it.
// Standard cuda::barrier producer/consumer idiom: arrive_and_wait() on a
// barrier whose expected count covers BOTH warps blocks each side until the
// other has arrived, making the producer's writes visible to the consumer.
// A broken handshake corrupts the consumer's sum -> EXACT vs CPU proves it.
// ---------------------------------------------------------------------------
__global__ void kMailbox(const int* tiles, int tile_len, int num_tiles,
                         long long* out) {
  __shared__ MailboxBarrier ready;     // producer-store -> consumer-read
  __shared__ MailboxBarrier consumed;  // consumer-read  -> producer-refill
  extern __shared__ int smem_tile[];   // tile_len ints (single buffer)

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int total = blockDim.x;  // == 64
  if (threadIdx.x == 0) {
    mailbox_init(&ready, total);
    mailbox_init(&consumed, total);
  }
  __syncthreads();

  long long local = 0;
  for (int t = 0; t < num_tiles; ++t) {
    if (warp == 0) {  // producer: fill the tile, then publish via `ready`.
      for (int i = lane; i < tile_len; i += 32)
        smem_tile[i] = tiles[t * tile_len + i];
    }
    // Both warps arrive+wait on `ready`: the consumer cannot pass until the
    // producer warp has arrived (its stores complete & visible).
    ready.arrive_and_wait();

    if (warp == 1) {  // consumer: read the published tile.
      for (int i = lane; i < tile_len; i += 32) local += smem_tile[i];
    }
    // Both arrive+wait on `consumed`: the producer cannot overwrite the buffer
    // for tile t+1 until the consumer has finished reading tile t.
    consumed.arrive_and_wait();
  }
  if (warp == 1) {  // reduce the consumer lane-local sums and emit.
    for (int o = 16; o > 0; o >>= 1)
      local += __shfl_down_sync(0xFFFFFFFFu, local, o);
    if (lane == 0) *out = local;
  }
}

// ---------------------------------------------------------------------------
// Perf (i): K SEPARATE trivial kernel launches (the per-op launch overhead the
// megakernel replaces). Each kernel does a tiny touch so it is not elided.
// ---------------------------------------------------------------------------
__global__ void kTrivial(int* scratch, int region) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) scratch[0] = scratch[0] + region;
}

// Perf (ii): K in-kernel grid-barriers in ONE persistent launch. Same trivial
// touch per region, separated by the shipped barrier (hand-rolled global).
__global__ void kBarrierLoopGlobal(int* scratch, int K, GlobalBarrier gb) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int sense = 0;
  for (int r = 0; r < K; ++r) {
    if (tid == 0) scratch[0] = scratch[0] + r;
    global_barrier_sync(gb, &sense);
  }
}

// Perf (ii-cg): same with cg.grid_sync(), for the A/B comparison.
__global__ void kBarrierLoopCG(int* scratch, int K) {
  cg::grid_group grid = cg::this_grid();
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int r = 0; r < K; ++r) {
    if (tid == 0) scratch[0] = scratch[0] + r;
    grid.sync();
  }
}

// ===========================================================================
int main(int argc, char** argv) {
  int device = 0;
  CK(cudaSetDevice(device));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, device));
  printf("=== PDE G0 microbench ===\n");
  printf("device: %s  SMs=%d  cc=%d.%d  coopLaunch=%d\n", prop.name,
         prop.multiProcessorCount, prop.major, prop.minor,
         prop.cooperativeLaunch);

  const int BLK = 256;
  const int N = 4096;     // reduction vector length
  const int K = 64;       // number of regions / barriers

  // ----- Plan persistent grid from device props -----
  GridPlan plan{};
  CK(plan_persistent_grid((const void*)kReductionGlobal, BLK, 0, device, &plan));
  printf("grid plan: blocks_per_sm=%d  resident_CTAs=%d  block_threads=%d\n",
         plan.blocks_per_sm, plan.grid_blocks, plan.block_threads);

  // Verify the cooperative grid actually fits (the launch-able coop grid size).
  int coop_blocks_cg = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&coop_blocks_cg,
                                                   (void*)kReductionCG, BLK, 0));
  int max_coop_grid = coop_blocks_cg * prop.multiProcessorCount;
  printf("cooperative occupancy (kReductionCG): blocks_per_sm=%d  max_coop_grid=%d\n",
         coop_blocks_cg, max_coop_grid);

  int grid = plan.grid_blocks;  // resident CTA count for the persistent launches
  if (max_coop_grid < grid) grid = max_coop_grid;  // never exceed coop limit
  printf("USING resident grid = %d CTAs (%d threads)\n", grid, grid * BLK);

  // ================= Correctness A =================
  std::vector<int> h_init(N);
  for (int i = 0; i < N; ++i) h_init[i] = (i * 2654435761u) & 0xFFFF;

  int* d_scratch;
  unsigned long long* d_gsum;
  long long* d_out;
  CK(cudaMalloc(&d_scratch, N * sizeof(int)));
  CK(cudaMalloc(&d_gsum, sizeof(unsigned long long)));
  CK(cudaMalloc(&d_out, sizeof(long long)));

  // CPU reference.
  std::vector<int> ref_vec = h_init;
  long long ref_checksum = cpu_reference_reduction(N, K, ref_vec);

  // -- impl (a) cg.grid_sync --
  CK(cudaMemcpy(d_scratch, h_init.data(), N * sizeof(int),
                cudaMemcpyHostToDevice));
  CK(cudaMemset(d_gsum, 0, sizeof(unsigned long long)));
  {
    void* args[] = {&d_scratch, (void*)&N, (void*)&K, &d_gsum, &d_out};
    CK(cudaLaunchCooperativeKernel((void*)kReductionCG, grid, BLK, args, 0, 0));
    CK(cudaDeviceSynchronize());
  }
  long long got_cg = 0;
  CK(cudaMemcpy(&got_cg, d_out, sizeof(long long), cudaMemcpyDeviceToHost));
  bool passA_cg = (got_cg == ref_checksum);
  printf("[CorrA cg.grid_sync]   got=%lld  ref=%lld  -> %s\n", got_cg,
         ref_checksum, passA_cg ? "PASS" : "FAIL");

  // -- impl (b) hand-rolled global --
  unsigned int* d_arrive;
  unsigned int* d_sense;
  CK(cudaMalloc(&d_arrive, sizeof(unsigned int)));
  CK(cudaMalloc(&d_sense, sizeof(unsigned int)));
  CK(cudaMemset(d_arrive, 0, sizeof(unsigned int)));
  CK(cudaMemset(d_sense, 0, sizeof(unsigned int)));
  GlobalBarrier gb{d_arrive, d_sense, grid};

  CK(cudaMemcpy(d_scratch, h_init.data(), N * sizeof(int),
                cudaMemcpyHostToDevice));
  CK(cudaMemset(d_gsum, 0, sizeof(unsigned long long)));
  {
    void* args[] = {&d_scratch, (void*)&N, (void*)&K, &d_gsum, &d_out, &gb};
    CK(cudaLaunchCooperativeKernel((void*)kReductionGlobal, grid, BLK, args, 0,
                                   0));
    CK(cudaDeviceSynchronize());
  }
  long long got_gl = 0;
  CK(cudaMemcpy(&got_gl, d_out, sizeof(long long), cudaMemcpyDeviceToHost));
  bool passA_gl = (got_gl == ref_checksum);
  printf("[CorrA global-barrier] got=%lld  ref=%lld  -> %s\n", got_gl,
         ref_checksum, passA_gl ? "PASS" : "FAIL");

  // ================= Correctness B (work-queue) =================
  const int M = 100000;  // items
  std::vector<int> h_vals(M);
  long long ref_wq = 0;
  for (int i = 0; i < M; ++i) {
    h_vals[i] = (int)((i * 1103515245u + 12345u) & 0xFFFF);
    ref_wq += h_vals[i];
  }
  int* d_vals;
  int* d_times;
  unsigned int* d_head;
  unsigned long long* d_check;
  CK(cudaMalloc(&d_vals, M * sizeof(int)));
  CK(cudaMalloc(&d_times, M * sizeof(int)));
  CK(cudaMalloc(&d_head, sizeof(unsigned int)));
  CK(cudaMalloc(&d_check, sizeof(unsigned long long)));
  CK(cudaMemcpy(d_vals, h_vals.data(), M * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemset(d_times, 0, M * sizeof(int)));
  CK(cudaMemset(d_head, 0, sizeof(unsigned int)));
  CK(cudaMemset(d_check, 0, sizeof(unsigned long long)));
  WorkQueue q{d_head, (unsigned int)M};
  // Plain (non-cooperative) launch is fine for the work-queue.
  kWorkQueue<<<grid, BLK>>>(q, d_vals, d_times, d_check);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<int> h_times(M);
  unsigned long long got_check = 0;
  CK(cudaMemcpy(h_times.data(), d_times, M * sizeof(int),
                cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&got_check, d_check, sizeof(unsigned long long),
                cudaMemcpyDeviceToHost));
  int bad = 0, total_proc = 0;
  for (int i = 0; i < M; ++i) {
    if (h_times[i] != 1) ++bad;
    total_proc += h_times[i];
  }
  bool passB = (bad == 0) && (total_proc == M) &&
               (got_check == (unsigned long long)ref_wq);
  printf("[CorrB work-queue]     items=%d  exactly-once-violations=%d  "
         "total_processed=%d  checksum got=%llu ref=%lld  -> %s\n",
         M, bad, total_proc, got_check, ref_wq, passB ? "PASS" : "FAIL");

  // ================= Correctness C (mbarrier mailbox) =================
  const int TILE = 512;       // ints per tile
  const int NTILES = 256;     // streamed tiles
  std::vector<int> h_tiles((size_t)TILE * NTILES);
  long long ref_mbox = 0;
  for (size_t i = 0; i < h_tiles.size(); ++i) {
    h_tiles[i] = (int)((i * 2246822519u + 3266489917u) & 0x3FFF);
    ref_mbox += h_tiles[i];  // consumer sums every element of every tile
  }
  int* d_tiles;
  long long* d_mbox;
  CK(cudaMalloc(&d_tiles, h_tiles.size() * sizeof(int)));
  CK(cudaMalloc(&d_mbox, sizeof(long long)));
  CK(cudaMemcpy(d_tiles, h_tiles.data(), h_tiles.size() * sizeof(int),
                cudaMemcpyHostToDevice));
  CK(cudaMemset(d_mbox, 0, sizeof(long long)));
  // EXACTLY 2 warps (producer + consumer), 1 CTA, dynamic smem = one tile.
  kMailbox<<<1, 64, TILE * sizeof(int)>>>(d_tiles, TILE, NTILES, d_mbox);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  long long got_mbox = 0;
  CK(cudaMemcpy(&got_mbox, d_mbox, sizeof(long long), cudaMemcpyDeviceToHost));
  bool passC = (got_mbox == ref_mbox);
  printf("[CorrC mbarrier-mailbox] tiles=%d x %d  got=%lld  ref=%lld  -> %s\n",
         NTILES, TILE, got_mbox, ref_mbox, passC ? "PASS" : "FAIL");

  // ================= Perf: K launches vs K in-kernel barriers =================
  const int REPS = 200;       // outer timed reps
  const int WARMUP = 20;
  cudaEvent_t t0, t1;
  CK(cudaEventCreate(&t0));
  CK(cudaEventCreate(&t1));

  // (i) K separate kernel launches.
  for (int w = 0; w < WARMUP; ++w) {
    for (int r = 0; r < K; ++r) kTrivial<<<grid, BLK>>>(d_scratch, r);
  }
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(t0));
  for (int rep = 0; rep < REPS; ++rep) {
    for (int r = 0; r < K; ++r) kTrivial<<<grid, BLK>>>(d_scratch, r);
  }
  CK(cudaEventRecord(t1));
  CK(cudaEventSynchronize(t1));
  float ms_launch = 0;
  CK(cudaEventElapsedTime(&ms_launch, t0, t1));
  double per_boundary_launch_us =
      (double)ms_launch * 1000.0 / ((double)REPS * K);

  // (ii) K in-kernel grid-barriers, ONE persistent launch (hand-rolled global).
  CK(cudaMemset(d_arrive, 0, sizeof(unsigned int)));
  CK(cudaMemset(d_sense, 0, sizeof(unsigned int)));
  for (int w = 0; w < WARMUP; ++w) {
    void* args[] = {&d_scratch, (void*)&K, &gb};
    CK(cudaLaunchCooperativeKernel((void*)kBarrierLoopGlobal, grid, BLK, args, 0,
                                   0));
  }
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(t0));
  for (int rep = 0; rep < REPS; ++rep) {
    void* args[] = {&d_scratch, (void*)&K, &gb};
    CK(cudaLaunchCooperativeKernel((void*)kBarrierLoopGlobal, grid, BLK, args, 0,
                                   0));
  }
  CK(cudaEventRecord(t1));
  CK(cudaEventSynchronize(t1));
  float ms_barrier_gl = 0;
  CK(cudaEventElapsedTime(&ms_barrier_gl, t0, t1));
  // Subtract the single persistent-launch overhead per rep so we isolate the
  // per-boundary barrier cost (REPS launches, each doing K barriers).
  double per_boundary_barrier_gl_us =
      (double)ms_barrier_gl * 1000.0 / ((double)REPS * K);

  // (ii-cg) same with cg.grid_sync().
  for (int w = 0; w < WARMUP; ++w) {
    void* args[] = {&d_scratch, (void*)&K};
    CK(cudaLaunchCooperativeKernel((void*)kBarrierLoopCG, grid, BLK, args, 0,
                                   0));
  }
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(t0));
  for (int rep = 0; rep < REPS; ++rep) {
    void* args[] = {&d_scratch, (void*)&K};
    CK(cudaLaunchCooperativeKernel((void*)kBarrierLoopCG, grid, BLK, args, 0,
                                   0));
  }
  CK(cudaEventRecord(t1));
  CK(cudaEventSynchronize(t1));
  float ms_barrier_cg = 0;
  CK(cudaEventElapsedTime(&ms_barrier_cg, t0, t1));
  double per_boundary_barrier_cg_us =
      (double)ms_barrier_cg * 1000.0 / ((double)REPS * K);

  printf("\n=== PERF (per region-boundary, averaged over %d reps x %d boundaries) ===\n",
         REPS, K);
  printf("(i)   K separate kernel LAUNCHES : %.4f us/boundary  (total %.3f ms/%d-region-step)\n",
         per_boundary_launch_us, ms_launch / REPS, K);
  printf("(ii)  in-kernel global barrier   : %.4f us/boundary  (total %.3f ms/%d-region-step)\n",
         per_boundary_barrier_gl_us, ms_barrier_gl / REPS, K);
  printf("(ii') in-kernel cg.grid_sync     : %.4f us/boundary  (total %.3f ms/%d-region-step)\n",
         per_boundary_barrier_cg_us, ms_barrier_cg / REPS, K);
  double ratio_gl = per_boundary_launch_us / per_boundary_barrier_gl_us;
  double ratio_cg = per_boundary_launch_us / per_boundary_barrier_cg_us;
  printf("RATIO launch/barrier  global=%.2fx  cg=%.2fx\n", ratio_gl, ratio_cg);

  // Ship decision: pick the cheaper viable barrier at this occupancy.
  const char* shipped;
  double shipped_us;
  if (per_boundary_barrier_gl_us <= per_boundary_barrier_cg_us) {
    shipped = "hand-rolled-global";
    shipped_us = per_boundary_barrier_gl_us;
  } else {
    shipped = "cg.grid_sync";
    shipped_us = per_boundary_barrier_cg_us;
  }
  bool perf_pass = shipped_us < per_boundary_launch_us;
  printf("SHIPPED barrier: %s  (%.4f us/boundary)\n", shipped, shipped_us);
  printf("PERF GATE: in-kernel barrier %s than separate launches -> %s\n",
         perf_pass ? "CHEAPER" : "NOT cheaper", perf_pass ? "PASS" : "FAIL");

  bool all_pass = passA_cg && passA_gl && passB && passC && perf_pass;
  printf("\n=== G0 GATE: %s ===\n", all_pass ? "PASS" : "FAIL");
  printf("SUMMARY_JSON {\"sm\":%d,\"resident_ctas\":%d,\"blocks_per_sm\":%d,"
         "\"launch_us\":%.4f,\"barrier_global_us\":%.4f,\"barrier_cg_us\":%.4f,"
         "\"ratio_global\":%.3f,\"ratio_cg\":%.3f,\"shipped\":\"%s\","
         "\"corrA_cg\":%d,\"corrA_global\":%d,\"corrB\":%d,\"corrC\":%d,"
         "\"perf_pass\":%d,\"gate\":\"%s\"}\n",
         prop.multiProcessorCount, grid, plan.blocks_per_sm,
         per_boundary_launch_us, per_boundary_barrier_gl_us,
         per_boundary_barrier_cg_us, ratio_gl, ratio_cg, shipped,
         (int)passA_cg, (int)passA_gl, (int)passB, (int)passC, (int)perf_pass,
         all_pass ? "PASS" : "FAIL");

  cudaFree(d_scratch); cudaFree(d_gsum); cudaFree(d_out);
  cudaFree(d_arrive); cudaFree(d_sense);
  cudaFree(d_vals); cudaFree(d_times); cudaFree(d_head); cudaFree(d_check);
  cudaFree(d_tiles); cudaFree(d_mbox);
  return all_pass ? 0 : 2;
}

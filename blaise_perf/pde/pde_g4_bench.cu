// PDE G4 gate microbench (standalone, sm_100). REAL numbers on GPU0.
//
// G4 = HETEROGENEOUS-WORKER copy ∥ compute overlap. Partition the resident
// cooperative grid into a COPY group (stage cold KV -> hot buffer) and a COMPUTE
// group (softmax-weighted hot-read reduction) running CONCURRENTLY. Prove the
// copy hides under the compute (swap-ahead), with correctness vs an independent
// CPU reference.
//
//   (A) OVERLAPPED : one persistent cooperative kernel, copy ∥ compute, end barrier.
//   (B) SERIAL     : copy ALL -> barrier -> compute ALL.
//   isolations     : copy-only, compute-only (whole grid each).
//
// Gate:
//   Correctness: (A) staged bytes == (B) staged bytes == CPU cold copy (byte-exact)
//                AND (A) out == (B) out == CPU softmax-weighted reduction
//                (cos >= 0.999999).
//   Overlap    : overlapped ≈ max(copy,compute), NOT copy+compute (serial). Report
//                hidden_fraction = (serial - overlapped)/min(copy,compute).
//   Also: device->hot and pinned->hot copy BW; kernel occupancy; SM-split sweep.
//
// Build: nvcc -std=c++17 -arch=sm_100 -O3 -o pde_g4_bench pde_g4_bench.cu
#include "pde_g4_het_overlap.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace pde;

#define CK(x)                                                     \
  do {                                                            \
    cudaError_t e_ = (x);                                         \
    if (e_ != cudaSuccess) {                                      \
      printf("CUDA_ERR %s:%d %s -> %s\n", __FILE__, __LINE__, #x, \
             cudaGetErrorString(e_));                             \
      std::exit(1);                                               \
    }                                                             \
  } while (0)

using elem_t = g4::elem_t;

static inline float fill_val(uint64_t i, uint64_t salt) {
  uint64_t z = (i + salt) * 0x9E3779B97F4A7C15ull;
  z ^= z >> 29;
  z *= 0xBF58476D1CE4E5B9ull;
  z ^= z >> 32;
  return ((float)(z & 0xFFFF) / 32768.0f) - 1.0f;
}

struct CmpStat { double cos; double max_abs_err; };
static CmpStat compare_f(const std::vector<float>& got,
                         const std::vector<float>& ref) {
  double dot = 0, ng = 0, nr = 0, maxe = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double a = got[i], b = ref[i];
    dot += a * b; ng += a * a; nr += b * b;
    maxe = std::max(maxe, std::fabs(a - b));
  }
  CmpStat s;
  s.cos = (ng > 0 && nr > 0) ? dot / (std::sqrt(ng) * std::sqrt(nr)) : 0.0;
  s.max_abs_err = maxe;
  return s;
}

// Byte-exact comparison of two bf16 buffers (raw bit pattern). Returns #mismatches.
static size_t compare_bytes(const std::vector<elem_t>& a,
                            const std::vector<elem_t>& b) {
  size_t mism = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    uint16_t ua, ub;
    std::memcpy(&ua, &a[i], 2);
    std::memcpy(&ub, &b[i], 2);
    if (ua != ub) mism++;
  }
  return mism;
}

static int occ_blocks_per_sm(const void* kernel, int bt, size_t smem) {
  int b = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, kernel, bt, smem));
  return b;
}

// Time a cooperative launch (closure) over REPS with WARM warmups.
template <class F>
static double time_us(F&& launch, cudaStream_t stream, int REPS, int WARM) {
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  for (int w = 0; w < WARM; ++w) launch();
  CK(cudaStreamSynchronize(stream));
  CK(cudaEventRecord(e0, stream));
  for (int r = 0; r < REPS; ++r) launch();
  CK(cudaEventRecord(e1, stream)); CK(cudaEventSynchronize(e1));
  float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return ms * 1000.0 / REPS;
}

// ---------------------------------------------------------------------------
// Independent CPU reference: (1) the staged bytes are exactly the cold blocks;
// (2) the softmax-weighted reduction over the hot resident blocks.
// ---------------------------------------------------------------------------
struct CpuRef {
  std::vector<elem_t> staged;   // == cold blocks bit-for-bit
  std::vector<float> out;       // [D]
};
static CpuRef cpu_reference(const std::vector<elem_t>& cold,
                            const std::vector<elem_t>& hot,
                            const std::vector<float>& q, int M_hot) {
  const int D = g4::kD;
  CpuRef r;
  r.staged = cold;  // staging is a pure copy
  r.out.assign(D, 0.0f);
  // scores
  std::vector<double> sc(M_hot, 0.0);
  double m = -1e300;
  for (int b = 0; b < M_hot; ++b) {
    double acc = 0.0;
    for (int d = 0; d < D; ++d)
      acc += (double)q[d] * (double)__bfloat162float(hot[(size_t)b * D + d]);
    sc[b] = acc;
    m = std::max(m, acc);
  }
  double denom = 0.0;
  for (int b = 0; b < M_hot; ++b) denom += std::exp(sc[b] - m);
  double inv = denom > 0 ? 1.0 / denom : 0.0;
  std::vector<double> acc(D, 0.0);
  for (int b = 0; b < M_hot; ++b) {
    double w = std::exp(sc[b] - m) * inv;
    for (int d = 0; d < D; ++d)
      acc[d] += w * (double)__bfloat162float(hot[(size_t)b * D + d]);
  }
  for (int d = 0; d < D; ++d) r.out[d] = (float)acc[d];
  return r;
}

int main() {
  int dev = 0;
  CK(cudaSetDevice(dev));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, dev));
  printf("device: %s  SMs=%d  cc=%d.%d  coopLaunch=%d  smemPerSM=%zu KB\n",
         prop.name, prop.multiProcessorCount, prop.major, prop.minor,
         prop.cooperativeLaunch, prop.sharedMemPerMultiprocessor / 1024);

  const int D = g4::kD;
  const size_t smem = g4::epilogue_smem_bytes();

  // Cooperative launch requires the grid to fit EVERY launched kernel
  // co-resident. Sizing from kOverlapped alone over-sizes kSerial / kCopyOnly /
  // kComputeOnly ("too many blocks in cooperative launch"); use the MIN
  // occupancy across all 4 cooperative kernels.
  GridPlan plan;
  CK(plan_persistent_grid((void*)g4::kOverlapped, g4::kBlockThreads, smem, dev,
                          &plan));
  int occ = std::min(
      std::min(plan.blocks_per_sm,
               occ_blocks_per_sm((void*)g4::kSerial, g4::kBlockThreads, smem)),
      std::min(occ_blocks_per_sm((void*)g4::kCopyOnly, g4::kBlockThreads, smem),
               occ_blocks_per_sm((void*)g4::kComputeOnly, g4::kBlockThreads,
                                 smem)));
  int grid = occ * prop.multiProcessorCount;
  cudaFuncAttributes fa;
  CK(cudaFuncGetAttributes(&fa, (void*)g4::kOverlapped));
  printf("OCC: kOverlapped %d blk/SM (%d CTAs) regs=%d smem=%zuB\n",
         occ, grid, fa.numRegs, smem);

  // Sweep configs. N_swap blocks (256-512 B each at D=128 bf16 = 256 B), M_hot
  // resident hot blocks sized so compute ≈ copy (to make the overlap visible:
  // hiding is only meaningful when neither side trivially dominates).
  struct Cfg { int n_swap; int m_hot; const char* name; };
  // M_hot picked empirically so compute is the same order as the copy at this
  // copy size; the SM-split sweep then finds the best partition.
  Cfg cfgs[2] = {
      {256, 4096, "Nswap256_Mhot4096"},
      {512, 8192, "Nswap512_Mhot8192"},
  };

  bool all_corr = true;

  // JSON accumulators (per cfg).
  double j_copy[2] = {0}, j_comp[2] = {0}, j_serial[2] = {0}, j_over[2] = {0};
  double j_over_best[2] = {0};
  int j_split_best[2] = {0};
  double j_hidden[2] = {0}, j_hidden_best[2] = {0};
  double j_bw_dev[2] = {0}, j_bw_pin[2] = {0};
  double j_cosA[2] = {0}, j_cosB[2] = {0};
  size_t j_byteA[2] = {0}, j_byteB[2] = {0};
  double j_overP[2] = {0}, j_serialP[2] = {0}, j_hiddenP[2] = {0};

  cudaStream_t stream;
  CK(cudaStreamCreate(&stream));

  for (int ci = 0; ci < 2; ++ci) {
    const int N = cfgs[ci].n_swap;
    const int M = cfgs[ci].m_hot;
    printf("\n############ CFG %s : N_swap=%d (%d B/blk) M_hot=%d D=%d ########\n",
           cfgs[ci].name, N, D * 2, M, D);

    // ---- host data ----
    std::vector<elem_t> hCold((size_t)N * D), hHot((size_t)M * D);
    std::vector<float> hQ(D);
    for (size_t i = 0; i < hCold.size(); ++i)
      hCold[i] = g4::to_e_host(fill_val(i, 211 + ci));
    for (size_t i = 0; i < hHot.size(); ++i)
      hHot[i] = g4::to_e_host(fill_val(i, 877 + ci));
    for (int d = 0; d < D; ++d) hQ[d] = fill_val(d, 13 + ci);

    // ---- CPU reference ----
    CpuRef ref = cpu_reference(hCold, hHot, hQ, M);

    // ---- device buffers ----
    elem_t *dCold, *dHotStaged, *dHotResident, *dColdPin;
    float *dQ, *dSims, *dOut;
    CK(cudaMalloc(&dCold, hCold.size() * sizeof(elem_t)));
    CK(cudaMalloc(&dHotStaged, hCold.size() * sizeof(elem_t)));
    CK(cudaMalloc(&dHotResident, hHot.size() * sizeof(elem_t)));
    CK(cudaMalloc(&dQ, D * sizeof(float)));
    CK(cudaMalloc(&dSims, (size_t)M * sizeof(float)));
    CK(cudaMalloc(&dOut, D * sizeof(float)));
    CK(cudaMemcpy(dCold, hCold.data(), hCold.size() * sizeof(elem_t),
                  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dHotResident, hHot.data(), hHot.size() * sizeof(elem_t),
                  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dQ, hQ.data(), D * sizeof(float), cudaMemcpyHostToDevice));
    // host-pinned cold source (the real swap-in path on a capacity miss)
    CK(cudaHostAlloc((void**)&dColdPin, hCold.size() * sizeof(elem_t),
                     cudaHostAllocDefault));
    std::memcpy(dColdPin, hCold.data(), hCold.size() * sizeof(elem_t));

    auto make_problem = [&](int n_copy, int use_pinned) {
      g4::Problem p;
      p.cold_dev = dCold;
      p.cold_pin = dColdPin;
      p.hot_staged = dHotStaged;
      p.hot_resident = dHotResident;
      p.q = dQ;
      p.sims = dSims;
      p.out = dOut;
      p.n_swap = N;
      p.m_hot = M;
      p.n_copy = n_copy;
      p.use_pinned = use_pinned;
      return p;
    };

    // Default split: ~1/4 of CTAs to COPY (a staging memcpy is bandwidth-bound
    // and saturates with few CTAs), rest to COMPUTE. Swept below.
    int default_copy = std::max(1, grid / 4);

    // ============ correctness: (A) overlapped vs (B) serial vs CPU ============
    // (A) overlapped, device-source.
    g4::Problem pA = make_problem(default_copy, 0);
    CK(cudaMemsetAsync(dHotStaged, 0, hCold.size() * sizeof(elem_t), stream));
    CK(cudaMemsetAsync(dOut, 0, D * sizeof(float), stream));
    {
      void* args[] = {&pA};
      CK(cudaLaunchCooperativeKernel((void*)g4::kOverlapped, grid,
                                     g4::kBlockThreads, args, smem, stream));
    }
    CK(cudaStreamSynchronize(stream));
    std::vector<elem_t> A_staged(hCold.size());
    std::vector<float> A_out(D);
    CK(cudaMemcpy(A_staged.data(), dHotStaged, A_staged.size() * sizeof(elem_t),
                  cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(A_out.data(), dOut, D * sizeof(float), cudaMemcpyDeviceToHost));

    // (B) serial, device-source.
    g4::Problem pB = make_problem(grid, 0);  // n_copy unused by kSerial
    CK(cudaMemsetAsync(dHotStaged, 0, hCold.size() * sizeof(elem_t), stream));
    CK(cudaMemsetAsync(dOut, 0, D * sizeof(float), stream));
    {
      void* args[] = {&pB};
      CK(cudaLaunchCooperativeKernel((void*)g4::kSerial, grid,
                                     g4::kBlockThreads, args, smem, stream));
    }
    CK(cudaStreamSynchronize(stream));
    std::vector<elem_t> B_staged(hCold.size());
    std::vector<float> B_out(D);
    CK(cudaMemcpy(B_staged.data(), dHotStaged, B_staged.size() * sizeof(elem_t),
                  cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(B_out.data(), dOut, D * sizeof(float), cudaMemcpyDeviceToHost));

    // (A) overlapped, PINNED-source (mirror the real swap-in cold->hot).
    g4::Problem pAp = make_problem(default_copy, 1);
    CK(cudaMemsetAsync(dHotStaged, 0, hCold.size() * sizeof(elem_t), stream));
    CK(cudaMemsetAsync(dOut, 0, D * sizeof(float), stream));
    {
      void* args[] = {&pAp};
      CK(cudaLaunchCooperativeKernel((void*)g4::kOverlapped, grid,
                                     g4::kBlockThreads, args, smem, stream));
    }
    CK(cudaStreamSynchronize(stream));
    std::vector<elem_t> Ap_staged(hCold.size());
    std::vector<float> Ap_out(D);
    CK(cudaMemcpy(Ap_staged.data(), dHotStaged,
                  Ap_staged.size() * sizeof(elem_t), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(Ap_out.data(), dOut, D * sizeof(float),
                  cudaMemcpyDeviceToHost));

    // compare
    size_t byteA = compare_bytes(A_staged, ref.staged);
    size_t byteB = compare_bytes(B_staged, ref.staged);
    size_t byteAp = compare_bytes(Ap_staged, ref.staged);
    size_t byteAB = compare_bytes(A_staged, B_staged);
    CmpStat cA = compare_f(A_out, ref.out);
    CmpStat cB = compare_f(B_out, ref.out);
    CmpStat cAp = compare_f(Ap_out, ref.out);
    CmpStat cAB = compare_f(A_out, B_out);
    bool ok = (byteA == 0) && (byteB == 0) && (byteAp == 0) && (byteAB == 0) &&
              (cA.cos >= 0.999999) && (cB.cos >= 0.999999) &&
              (cAp.cos >= 0.999999) && (cAB.cos >= 0.999999);
    all_corr = all_corr && ok;
    j_cosA[ci] = cA.cos; j_cosB[ci] = cB.cos;
    j_byteA[ci] = byteA; j_byteB[ci] = byteB;

    printf("CORR staged-bytes vs CPU: A=%zu mism  B=%zu mism  A(pinned)=%zu mism "
           " A-vs-B=%zu mism\n", byteA, byteB, byteAp, byteAB);
    printf("CORR reduction vs CPU: A cos=%.8f maxabs=%.3e | B cos=%.8f "
           "maxabs=%.3e | A(pin) cos=%.8f | A-vs-B cos=%.8f -> %s\n",
           cA.cos, cA.max_abs_err, cB.cos, cB.max_abs_err, cAp.cos, cAB.cos,
           ok ? "PASS" : "FAIL");

    // ============ isolation timings (device-source) ============
    g4::Problem pCopy = make_problem(grid, 0);
    g4::Problem pComp = make_problem(0, 0);  // n_copy=0 -> all compute
    double copy_us = time_us(
        [&]() {
          void* a[] = {&pCopy};
          CK(cudaLaunchCooperativeKernel((void*)g4::kCopyOnly, grid,
                                         g4::kBlockThreads, a, smem, stream));
        },
        stream, 200, 40);
    double comp_us = time_us(
        [&]() {
          void* a[] = {&pComp};
          CK(cudaLaunchCooperativeKernel((void*)g4::kComputeOnly, grid,
                                         g4::kBlockThreads, a, smem, stream));
        },
        stream, 200, 40);
    double serial_us = time_us(
        [&]() {
          void* a[] = {&pB};
          CK(cudaLaunchCooperativeKernel((void*)g4::kSerial, grid,
                                         g4::kBlockThreads, a, smem, stream));
        },
        stream, 200, 40);

    // ============ SM-split sweep for the OVERLAPPED kernel ============
    // Try several COPY-group sizes; record best (lowest) overlapped latency.
    int splits[6];
    splits[0] = std::max(1, grid / 16);
    splits[1] = std::max(1, grid / 8);
    splits[2] = std::max(1, grid / 4);
    splits[3] = std::max(1, grid / 3);
    splits[4] = std::max(1, grid / 2);
    splits[5] = std::max(1, (grid * 2) / 3);
    double over_default = 0.0, over_best = 1e300;
    int best_split = splits[0];
    printf("OVERLAP SM-split sweep (copy CTAs / %d total):\n", grid);
    for (int s = 0; s < 6; ++s) {
      int nc = std::min(grid - 1, splits[s]);  // keep >=1 compute CTA
      if (nc < 1) nc = 1;
      g4::Problem ps = make_problem(nc, 0);
      double us = time_us(
          [&]() {
            void* a[] = {&ps};
            CK(cudaLaunchCooperativeKernel((void*)g4::kOverlapped, grid,
                                           g4::kBlockThreads, a, smem, stream));
          },
          stream, 200, 40);
      double maxcc = std::max(copy_us, comp_us);
      printf("  copy=%4d compute=%4d : overlapped=%8.3f us  (max(copy,comp)="
             "%8.3f, serial=%8.3f)\n", nc, grid - nc, us, maxcc, serial_us);
      if (nc == default_copy) over_default = us;
      if (us < over_best) { over_best = us; best_split = nc; }
    }
    if (over_default == 0.0) over_default = over_best;

    // ============ pinned-source overlapped + serial (real swap-in mirror) ====
    g4::Problem pOverPin = make_problem(best_split, 1);
    g4::Problem pSerPin = make_problem(grid, 1);
    double over_pin_us = time_us(
        [&]() {
          void* a[] = {&pOverPin};
          CK(cudaLaunchCooperativeKernel((void*)g4::kOverlapped, grid,
                                         g4::kBlockThreads, a, smem, stream));
        },
        stream, 100, 20);
    double serial_pin_us = time_us(
        [&]() {
          void* a[] = {&pSerPin};
          CK(cudaLaunchCooperativeKernel((void*)g4::kSerial, grid,
                                         g4::kBlockThreads, a, smem, stream));
        },
        stream, 100, 20);

    // ============ copy BW (device->hot, pinned->hot) ============
    // bytes moved = read N*D*2 + write N*D*2 (device->hot reads + writes);
    // for pinned->hot, the read crosses C2C/PCIe. Report effective GB/s using
    // the WRITE volume (hot buffer filled) as the canonical figure + total.
    const double bytes_rw = 2.0 * (double)N * D * 2.0;  // read+write bf16
    const double bytes_w = (double)N * D * 2.0;
    double bw_dev = bytes_rw / (copy_us * 1e-6) / 1e9;       // GB/s (R+W)
    double bw_dev_w = bytes_w / (copy_us * 1e-6) / 1e9;      // GB/s (W only)
    // pinned copy-only timing
    double copy_pin_us = time_us(
        [&]() {
          void* a[] = {&pSerPin};  // serial copies whole grid; reuse copy-only
          // use kCopyOnly with pinned
          (void)a;
          g4::Problem pcp = make_problem(grid, 1);
          void* a2[] = {&pcp};
          CK(cudaLaunchCooperativeKernel((void*)g4::kCopyOnly, grid,
                                         g4::kBlockThreads, a2, smem, stream));
        },
        stream, 100, 20);
    double bw_pin = bytes_w / (copy_pin_us * 1e-6) / 1e9;  // GB/s host->device

    // ============ hidden fractions ============
    // hidden_fraction = (serial - overlapped) / min(copy,compute), clamped [0,1].
    double mincc = std::min(copy_us, comp_us);
    auto hidden_of = [&](double over) {
      double h = (serial_us - over) / (mincc > 0 ? mincc : 1.0);
      if (h < 0) h = 0; if (h > 1) h = 1; return h;
    };
    double hidden_default = hidden_of(over_default);
    double hidden_best = hidden_of(over_best);
    double mincc_pin = std::min(copy_pin_us, comp_us);
    double hidden_pin = (serial_pin_us - over_pin_us) /
                        (mincc_pin > 0 ? mincc_pin : 1.0);
    if (hidden_pin < 0) hidden_pin = 0; if (hidden_pin > 1) hidden_pin = 1;

    printf("PERF (us/iter, device-source):\n");
    printf("  copy-only      : %.3f us  (dev->hot BW: %.1f GB/s R+W, %.1f GB/s W)\n",
           copy_us, bw_dev, bw_dev_w);
    printf("  compute-only   : %.3f us\n", comp_us);
    printf("  serial(copy+compute) : %.3f us\n", serial_us);
    printf("  overlapped (split=%d): %.3f us   best(split=%d): %.3f us\n",
           default_copy, over_default, best_split, over_best);
    printf("  sum(copy+comp) : %.3f us   max(copy,comp): %.3f us\n",
           copy_us + comp_us, std::max(copy_us, comp_us));
    printf("  HIDDEN fraction: default=%.1f%%  best=%.1f%%  "
           "(overlapped vs max => %s)\n",
           hidden_default * 100.0, hidden_best * 100.0,
           (over_best <= std::max(copy_us, comp_us) * 1.10) ? "≈max (HIDDEN)"
                                                            : "> max");
    printf("PERF (pinned cold->hot swap-in mirror):\n");
    printf("  copy-only(pin) : %.3f us  (host->dev BW: %.1f GB/s W)\n",
           copy_pin_us, bw_pin);
    printf("  serial(pin)    : %.3f us   overlapped(pin,split=%d): %.3f us  "
           "HIDDEN=%.1f%%\n", serial_pin_us, best_split, over_pin_us,
           hidden_pin * 100.0);

    j_copy[ci] = copy_us; j_comp[ci] = comp_us; j_serial[ci] = serial_us;
    j_over[ci] = over_default; j_over_best[ci] = over_best;
    j_split_best[ci] = best_split;
    j_hidden[ci] = hidden_default; j_hidden_best[ci] = hidden_best;
    j_bw_dev[ci] = bw_dev_w; j_bw_pin[ci] = bw_pin;
    j_overP[ci] = over_pin_us; j_serialP[ci] = serial_pin_us;
    j_hiddenP[ci] = hidden_pin;

    cudaFree(dCold); cudaFree(dHotStaged); cudaFree(dHotResident);
    cudaFree(dQ); cudaFree(dSims); cudaFree(dOut);
    cudaFreeHost(dColdPin);
  }

  printf("\n=== G4 GATE: %s ===\n",
         all_corr ? "PASS(correctness)" : "FAIL(correctness)");
  printf("SUMMARY_JSON {\"sm\":%d,\"occ\":%d,\"grid\":%d,\"regs\":%d,"
         "\"c0_name\":\"%s\",\"c1_name\":\"%s\","
         "\"c0_copy_us\":%.3f,\"c0_compute_us\":%.3f,\"c0_serial_us\":%.3f,"
         "\"c0_over_us\":%.3f,\"c0_over_best_us\":%.3f,\"c0_split_best\":%d,"
         "\"c0_hidden\":%.4f,\"c0_hidden_best\":%.4f,"
         "\"c0_bw_dev_w_GBs\":%.1f,\"c0_bw_pin_GBs\":%.1f,"
         "\"c0_over_pin_us\":%.3f,\"c0_serial_pin_us\":%.3f,\"c0_hidden_pin\":%.4f,"
         "\"c0_cosA\":%.8f,\"c0_cosB\":%.8f,\"c0_byteA\":%zu,\"c0_byteB\":%zu,"
         "\"c1_copy_us\":%.3f,\"c1_compute_us\":%.3f,\"c1_serial_us\":%.3f,"
         "\"c1_over_us\":%.3f,\"c1_over_best_us\":%.3f,\"c1_split_best\":%d,"
         "\"c1_hidden\":%.4f,\"c1_hidden_best\":%.4f,"
         "\"c1_bw_dev_w_GBs\":%.1f,\"c1_bw_pin_GBs\":%.1f,"
         "\"c1_over_pin_us\":%.3f,\"c1_serial_pin_us\":%.3f,\"c1_hidden_pin\":%.4f,"
         "\"c1_cosA\":%.8f,\"c1_cosB\":%.8f,\"c1_byteA\":%zu,\"c1_byteB\":%zu,"
         "\"corr_gate\":\"%s\"}\n",
         prop.multiProcessorCount, occ, grid, fa.numRegs,
         cfgs[0].name, cfgs[1].name,
         j_copy[0], j_comp[0], j_serial[0], j_over[0], j_over_best[0],
         j_split_best[0], j_hidden[0], j_hidden_best[0], j_bw_dev[0], j_bw_pin[0],
         j_overP[0], j_serialP[0], j_hiddenP[0],
         j_cosA[0], j_cosB[0], j_byteA[0], j_byteB[0],
         j_copy[1], j_comp[1], j_serial[1], j_over[1], j_over_best[1],
         j_split_best[1], j_hidden[1], j_hidden_best[1], j_bw_dev[1], j_bw_pin[1],
         j_overP[1], j_serialP[1], j_hiddenP[1],
         j_cosA[1], j_cosB[1], j_byteA[1], j_byteB[1],
         all_corr ? "PASS" : "FAIL");

  cudaStreamDestroy(stream);
  return all_corr ? 0 : 2;
}

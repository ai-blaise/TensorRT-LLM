// PDE G2 gate microbench (standalone, sm_100). Real numbers on GPU0.
//
// G2 = warp-specialized DOUBLE-BUFFERED (N-stage) cp.async OVERLAP. This bench
// compares THREE fused/handoff variants on the same two-region decode-ish GEMM,
// all against the SAME independent CPU f32 reference:
//
//   (1) G2 overlapped-fused      (kFusedTwoRegionPipe<kStages>, this gate)
//   (2) G1 single-buffer-fused   (kFusedTwoRegion, the prior gate's baseline)
//   (3) separate two kernels      (overlap-pipe single-region x2, the ceiling)
//
// Gate:
//   Correctness: overlapped-fused C_B == CPU f32 reference (cos >= 0.999999) at
//                B=1,8,32, AND == G1 single-buffer-fused output. Region A and the
//                intermediate C_A are also checked vs CPU.
//   Perf (the G2 point): overlapped-fused vs G1-single-buffer-fused vs separate,
//                at B=1,8,32. Report overlap speedup (overlapped/single-buffer)
//                and overlapped/separate.
//   Stage sweep: kStages = 2 and 3 — report best + achieved occupancy
//                (blocks/SM, regs/thread, smem) for each.
//
// Build: nvcc -std=c++17 -arch=sm_100 -O3 -o pde_g2_bench pde_g2_bench.cu
#include "pde_g1_two_region.cuh"   // G1 single-buffer baseline (kFusedTwoRegion)
#include "pde_g2_overlap.cuh"      // G2 overlap pipeline (kFusedTwoRegionPipe)

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace pde;

#define CK(x)                                                         \
  do {                                                                \
    cudaError_t e_ = (x);                                             \
    if (e_ != cudaSuccess) {                                          \
      printf("CUDA_ERR %s:%d %s -> %s\n", __FILE__, __LINE__, #x,     \
             cudaGetErrorString(e_));                                 \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

// DeepSeek-V3.2 decode-ish shapes for the two regions (same as G1).
constexpr int kKA = 512;
constexpr int kNA = 7168;
constexpr int kKB = 7168;   // == kNA
constexpr int kNB = 2048;

static inline float bf16_to_f32(__nv_bfloat16 x) { return __bfloat162float(x); }
static inline __nv_bfloat16 f32_to_bf16(float x) { return __float2bfloat16(x); }

static inline float fill_val(uint64_t i, uint64_t salt) {
  uint64_t z = (i + salt) * 0x9E3779B97F4A7C15ull;
  z ^= z >> 29;
  z *= 0xBF58476D1CE4E5B9ull;
  z ^= z >> 32;
  return ((float)(z & 0xFFFF) / 32768.0f) - 1.0f;
}

// Independent CPU f32 reference GEMM (operands pre-rounded to bf16). NEVER a
// self-comparison — this is the correctness ground truth.
static void cpu_gemm(const std::vector<__nv_bfloat16>& A,
                     const std::vector<__nv_bfloat16>& W, std::vector<float>& C,
                     int M, int K, int N) {
  C.assign((size_t)M * N, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += bf16_to_f32(A[(size_t)m * K + k]) *
               bf16_to_f32(W[(size_t)k * N + n]);
      }
      C[(size_t)m * N + n] = acc;
    }
  }
}

struct CmpStat { double cos; double max_abs_err; double ref_norm; };
static CmpStat compare(const std::vector<float>& got,
                       const std::vector<float>& ref) {
  double dot = 0, ng = 0, nr = 0, maxe = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double a = got[i], b = ref[i];
    dot += a * b; ng += a * a; nr += b * b;
    double e = std::fabs(a - b);
    if (e > maxe) maxe = e;
  }
  CmpStat s;
  s.cos = (ng > 0 && nr > 0) ? dot / (std::sqrt(ng) * std::sqrt(nr)) : 0.0;
  s.max_abs_err = maxe;
  s.ref_norm = std::sqrt(nr);
  return s;
}

static int occ_blocks_per_sm(const void* kernel, int block_threads,
                             size_t dyn_smem) {
  int b = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, kernel, block_threads,
                                                   dyn_smem));
  return b;
}

// ---- per-stage-count function-pointer table (instantiate the templates) ----
struct StageKernels {
  int stages;
  void* fused;          // kFusedTwoRegionPipe<stages>
  void* single;         // kSingleRegionPipe<stages>
  size_t smem;          // fused_smem_bytes(stages)
};

template <int S>
static StageKernels make_stage() {
  StageKernels k;
  k.stages = S;
  k.fused = (void*)g2::kFusedTwoRegionPipe<S>;
  k.single = (void*)g2::kSingleRegionPipe<S>;
  k.smem = g2::fused_smem_bytes(S);
  return k;
}

struct OccInfo { int blocks_per_sm; int regs; size_t smem; };
static OccInfo stage_occ(const StageKernels& sk, cudaFuncAttributes& fa_out) {
  OccInfo o;
  CK(cudaFuncSetAttribute(sk.fused,
                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                          (int)sk.smem));
  CK(cudaFuncSetAttribute(sk.single,
                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                          (int)sk.smem));
  o.blocks_per_sm = occ_blocks_per_sm(sk.fused, g2::kBlockThreads, sk.smem);
  CK(cudaFuncGetAttributes(&fa_out, sk.fused));
  o.regs = fa_out.numRegs;
  o.smem = sk.smem;
  return o;
}

// Run + time a fused-pipe variant for a given stage count, returning us/iter and
// the device C_B (f32) + C_A (f32) for correctness.
struct FusedRun { double us; std::vector<float> CB; std::vector<float> CA; };

static FusedRun run_fused_pipe(const StageKernels& sk, int grid,
                               g2::GemmRegion regA, g2::GemmRegion regB,
                               unsigned int* dHeadA, unsigned int* dHeadB,
                               int nA_tiles, int nB_tiles, int M,
                               float* dCB_f, float* dCA_f, cudaStream_t stream) {
  auto launch = [&]() {
    CK(cudaMemsetAsync(dHeadA, 0, sizeof(unsigned int), stream));
    CK(cudaMemsetAsync(dHeadB, 0, sizeof(unsigned int), stream));
    WorkQueue qA{dHeadA, (unsigned int)nA_tiles};
    WorkQueue qB{dHeadB, (unsigned int)nB_tiles};
    void* args[] = {&regA, &regB, &qA, &qB};
    CK(cudaLaunchCooperativeKernel(sk.fused, grid, g2::kBlockThreads, args,
                                   sk.smem, stream));
  };
  launch();
  CK(cudaStreamSynchronize(stream));
  FusedRun fr;
  fr.CB.resize((size_t)M * kNB);
  fr.CA.resize((size_t)M * kNA);
  CK(cudaMemcpy(fr.CB.data(), dCB_f, fr.CB.size() * sizeof(float),
                cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(fr.CA.data(), dCA_f, fr.CA.size() * sizeof(float),
                cudaMemcpyDeviceToHost));
  const int REPS = 100, WARM = 20;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  for (int w = 0; w < WARM; ++w) launch();
  CK(cudaStreamSynchronize(stream));
  CK(cudaEventRecord(e0, stream));
  for (int r = 0; r < REPS; ++r) launch();
  CK(cudaEventRecord(e1, stream));
  CK(cudaEventSynchronize(e1));
  float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
  fr.us = ms * 1000.0 / REPS;
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return fr;
}

int main() {
  int dev = 0;
  CK(cudaSetDevice(dev));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, dev));
  printf("device: %s  SMs=%d  cc=%d.%d  coopLaunch=%d  L2=%.1f MB  "
         "smemPerSM=%zu KB\n",
         prop.name, prop.multiProcessorCount, prop.major, prop.minor,
         prop.cooperativeLaunch, prop.l2CacheSize / 1048576.0,
         prop.sharedMemPerMultiprocessor / 1024);

  // ---- Stage variants under test: 2 and 3 (G2 sweep) ----
  StageKernels s2 = make_stage<2>();
  StageKernels s3 = make_stage<3>();

  // ---- Occupancy / register / smem per stage count (+ G1 single-buffer) ----
  const size_t g1_smem = g1::fused_smem_bytes();
  CK(cudaFuncSetAttribute((void*)g1::kFusedTwoRegion,
                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                          (int)g1_smem));
  CK(cudaFuncSetAttribute((void*)g1::kSingleRegion,
                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                          (int)g1_smem));
  int occ_g1 = occ_blocks_per_sm((void*)g1::kFusedTwoRegion, g1::kBlockThreads,
                                 g1_smem);
  cudaFuncAttributes fa_g1; CK(cudaFuncGetAttributes(&fa_g1,
                                                     (void*)g1::kFusedTwoRegion));
  cudaFuncAttributes fa2, fa3;
  OccInfo o2 = stage_occ(s2, fa2);
  OccInfo o3 = stage_occ(s3, fa3);

  printf("\n=== OCCUPANCY / PRESSURE (@ %d threads) ===\n", g2::kBlockThreads);
  printf("  G1 single-buffer fused : %d blocks/SM  (%d CTAs)  regs=%d  smem=%.1f KB\n",
         occ_g1, occ_g1 * prop.multiProcessorCount, fa_g1.numRegs,
         g1_smem / 1024.0);
  printf("  G2 fused  2-stage      : %d blocks/SM  (%d CTAs)  regs=%d  smem=%.1f KB\n",
         o2.blocks_per_sm, o2.blocks_per_sm * prop.multiProcessorCount, o2.regs,
         o2.smem / 1024.0);
  printf("  G2 fused  3-stage      : %d blocks/SM  (%d CTAs)  regs=%d  smem=%.1f KB\n",
         o3.blocks_per_sm, o3.blocks_per_sm * prop.multiProcessorCount, o3.regs,
         o3.smem / 1024.0);

  // Persistent grids sized PER kernel/smem (occupancy differs by stage count).
  GridPlan plan_g1, plan2, plan3;
  CK(plan_persistent_grid((void*)g1::kFusedTwoRegion, g1::kBlockThreads, g1_smem,
                          dev, &plan_g1));
  CK(plan_persistent_grid(s2.fused, g2::kBlockThreads, s2.smem, dev, &plan2));
  CK(plan_persistent_grid(s3.fused, g2::kBlockThreads, s3.smem, dev, &plan3));
  printf("persistent grids: G1=%d  G2s2=%d  G2s3=%d CTAs\n", plan_g1.grid_blocks,
         plan2.grid_blocks, plan3.grid_blocks);

  const int Bs[3] = {1, 8, 32};
  bool all_corr = true;

  // ---- shared weights ----
  std::vector<__nv_bfloat16> hWA((size_t)kKA * kNA), hWB((size_t)kKB * kNB);
  for (size_t i = 0; i < hWA.size(); ++i) hWA[i] = f32_to_bf16(fill_val(i, 11));
  for (size_t i = 0; i < hWB.size(); ++i) hWB[i] = f32_to_bf16(fill_val(i, 22));
  __nv_bfloat16 *dWA, *dWB;
  CK(cudaMalloc(&dWA, hWA.size() * sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&dWB, hWB.size() * sizeof(__nv_bfloat16)));
  CK(cudaMemcpy(dWA, hWA.data(), hWA.size() * sizeof(__nv_bfloat16),
                cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dWB, hWB.data(), hWB.size() * sizeof(__nv_bfloat16),
                cudaMemcpyHostToDevice));

  cudaStream_t stream; CK(cudaStreamCreate(&stream));

  // Accumulators for the final JSON summary.
  double best_speedup_vs_g1[3] = {0, 0, 0};
  int best_stage_per_b[3] = {0, 0, 0};

  for (int bi = 0; bi < 3; ++bi) {
    const int M = Bs[bi];
    std::vector<__nv_bfloat16> hAA((size_t)M * kKA);
    for (size_t i = 0; i < hAA.size(); ++i)
      hAA[i] = f32_to_bf16(fill_val(i, 33 + bi));

    // ---- CPU reference (independent ground truth) ----
    std::vector<float> CA_ref, CB_ref;
    cpu_gemm(hAA, hWA, CA_ref, M, kKA, kNA);
    std::vector<__nv_bfloat16> CA_ref_bf((size_t)M * kNA);
    for (size_t i = 0; i < CA_ref.size(); ++i) CA_ref_bf[i] = f32_to_bf16(CA_ref[i]);
    cpu_gemm(CA_ref_bf, hWB, CB_ref, M, kKB, kNB);

    // ---- device buffers (shared input AA + weights; per-variant outputs) ----
    __nv_bfloat16 *dAA, *dCA_bf, *dCB_bf;
    float *dCA_f, *dCB_f;
    CK(cudaMalloc(&dAA, (size_t)M * kKA * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dCA_bf, (size_t)M * kNA * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dCA_f, (size_t)M * kNA * sizeof(float)));
    CK(cudaMalloc(&dCB_bf, (size_t)M * kNB * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dCB_f, (size_t)M * kNB * sizeof(float)));
    CK(cudaMemcpy(dAA, hAA.data(), hAA.size() * sizeof(__nv_bfloat16),
                  cudaMemcpyHostToDevice));

    const int nA_tiles = (kNA + g2::kTileN - 1) / g2::kTileN;
    const int nB_tiles = (kNB + g2::kTileN - 1) / g2::kTileN;
    unsigned int *dHeadA, *dHeadB;
    CK(cudaMalloc(&dHeadA, sizeof(unsigned int)));
    CK(cudaMalloc(&dHeadB, sizeof(unsigned int)));

    // ===== (2) G1 single-buffer fused (baseline) =====
    g1::GemmRegion g1A{dAA, dWA, dCA_f, dCA_bf, M, kKA, kNA};
    g1::GemmRegion g1B{dCA_bf, dWB, dCB_f, dCB_bf, M, kKB, kNB};
    auto launch_g1 = [&]() {
      CK(cudaMemsetAsync(dHeadA, 0, sizeof(unsigned int), stream));
      CK(cudaMemsetAsync(dHeadB, 0, sizeof(unsigned int), stream));
      WorkQueue qA{dHeadA, (unsigned int)nA_tiles};
      WorkQueue qB{dHeadB, (unsigned int)nB_tiles};
      void* args[] = {&g1A, &g1B, &qA, &qB};
      CK(cudaLaunchCooperativeKernel((void*)g1::kFusedTwoRegion,
                                     plan_g1.grid_blocks, g1::kBlockThreads, args,
                                     g1_smem, stream));
    };
    launch_g1();
    CK(cudaStreamSynchronize(stream));
    std::vector<float> g1_CB((size_t)M * kNB);
    CK(cudaMemcpy(g1_CB.data(), dCB_f, g1_CB.size() * sizeof(float),
                  cudaMemcpyDeviceToHost));
    {
      const int REPS = 100, WARM = 20;
      cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
      for (int w = 0; w < WARM; ++w) launch_g1();
      CK(cudaStreamSynchronize(stream));
      CK(cudaEventRecord(e0, stream));
      for (int r = 0; r < REPS; ++r) launch_g1();
      CK(cudaEventRecord(e1, stream)); CK(cudaEventSynchronize(e1));
      float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
      double us_g1 = ms * 1000.0 / REPS;
      cudaEventDestroy(e0); cudaEventDestroy(e1);

      // ===== (1) G2 overlapped fused, stage = 2 and 3 =====
      g2::GemmRegion p2A{dAA, dWA, dCA_f, dCA_bf, M, kKA, kNA};
      g2::GemmRegion p2B{dCA_bf, dWB, dCB_f, dCB_bf, M, kKB, kNB};
      FusedRun r2 = run_fused_pipe(s2, plan2.grid_blocks, p2A, p2B, dHeadA, dHeadB,
                                   nA_tiles, nB_tiles, M, dCB_f, dCA_f, stream);
      std::vector<float> r2_CA = r2.CA;  // grabbed before overwrite by s3
      FusedRun r3 = run_fused_pipe(s3, plan3.grid_blocks, p2A, p2B, dHeadA, dHeadB,
                                   nA_tiles, nB_tiles, M, dCB_f, dCA_f, stream);

      // ===== (3) separate two kernels (overlap-pipe), the ceiling =====
      // Use the BEST stage count's single-region kernel for the separate path.
      __nv_bfloat16 *dCA_bf2, *dCB_bf2;
      float *dCA_f2, *dCB_f2;
      CK(cudaMalloc(&dCA_bf2, (size_t)M * kNA * sizeof(__nv_bfloat16)));
      CK(cudaMalloc(&dCA_f2, (size_t)M * kNA * sizeof(float)));
      CK(cudaMalloc(&dCB_bf2, (size_t)M * kNB * sizeof(__nv_bfloat16)));
      CK(cudaMalloc(&dCB_f2, (size_t)M * kNB * sizeof(float)));
      g2::GemmRegion sepA{dAA, dWA, dCA_f2, dCA_bf2, M, kKA, kNA};
      g2::GemmRegion sepB{dCA_bf2, dWB, dCB_f2, dCB_bf2, M, kKB, kNB};
      const StageKernels& sbest = (r2.us <= r3.us) ? s2 : s3;
      const GridPlan& pbest = (r2.us <= r3.us) ? plan2 : plan3;
      int gA = nA_tiles < pbest.grid_blocks ? nA_tiles : pbest.grid_blocks;
      int gB = nB_tiles < pbest.grid_blocks ? nB_tiles : pbest.grid_blocks;
      auto launch_sep = [&]() {
        if (sbest.stages == 2) {
          g2::kSingleRegionPipe<2><<<gA, g2::kBlockThreads, sbest.smem, stream>>>(sepA);
          g2::kSingleRegionPipe<2><<<gB, g2::kBlockThreads, sbest.smem, stream>>>(sepB);
        } else {
          g2::kSingleRegionPipe<3><<<gA, g2::kBlockThreads, sbest.smem, stream>>>(sepA);
          g2::kSingleRegionPipe<3><<<gB, g2::kBlockThreads, sbest.smem, stream>>>(sepB);
        }
      };
      launch_sep();
      CK(cudaGetLastError());
      CK(cudaStreamSynchronize(stream));
      std::vector<float> sep_CB((size_t)M * kNB);
      CK(cudaMemcpy(sep_CB.data(), dCB_f2, sep_CB.size() * sizeof(float),
                    cudaMemcpyDeviceToHost));
      {
        const int REPS = 100, WARM = 20;
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        for (int w = 0; w < WARM; ++w) launch_sep();
        CK(cudaStreamSynchronize(stream));
        CK(cudaEventRecord(e0, stream));
        for (int r = 0; r < REPS; ++r) launch_sep();
        CK(cudaEventRecord(e1, stream)); CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
        double us_sep = ms * 1000.0 / REPS;
        cudaEventDestroy(e0); cudaEventDestroy(e1);

        // ---- correctness (all vs CPU; overlap vs G1 secondary) ----
        CmpStat cA2 = compare(r2_CA, CA_ref);          // s2 region A vs CPU
        CmpStat c2 = compare(r2.CB, CB_ref);           // s2 fused B vs CPU
        CmpStat c3 = compare(r3.CB, CB_ref);           // s3 fused B vs CPU
        CmpStat c2_vs_g1 = compare(r2.CB, g1_CB);      // s2 vs G1 single-buf
        CmpStat c3_vs_g1 = compare(r3.CB, g1_CB);      // s3 vs G1 single-buf
        CmpStat csep = compare(sep_CB, CB_ref);        // separate vs CPU
        bool ok = (c2.cos >= 0.999999) && (c3.cos >= 0.999999) &&
                  (cA2.cos >= 0.999999) && (c2_vs_g1.cos >= 0.999999) &&
                  (c3_vs_g1.cos >= 0.999999) && (csep.cos >= 0.999999);
        all_corr = all_corr && ok;

        printf("\n=== B=%d (M=%d) CORRECTNESS (all vs independent CPU f32) ===\n",
               M, M);
        printf("  s2 regionA vs CPU : cos=%.8f  maxabs=%.3e\n", cA2.cos,
               cA2.max_abs_err);
        printf("  s2 fusedB  vs CPU : cos=%.8f  maxabs=%.3e\n", c2.cos,
               c2.max_abs_err);
        printf("  s3 fusedB  vs CPU : cos=%.8f  maxabs=%.3e\n", c3.cos,
               c3.max_abs_err);
        printf("  separate   vs CPU : cos=%.8f  maxabs=%.3e\n", csep.cos,
               csep.max_abs_err);
        printf("  s2 fusedB  vs G1  : cos=%.8f   s3 fusedB vs G1 : cos=%.8f  -> %s\n",
               c2_vs_g1.cos, c3_vs_g1.cos, ok ? "PASS" : "FAIL");

        // ---- perf ----
        double best_us = (r2.us <= r3.us) ? r2.us : r3.us;
        int best_stage = (r2.us <= r3.us) ? 2 : 3;
        double sp_vs_g1_best = us_g1 / best_us;
        double sp_vs_g1_s2 = us_g1 / r2.us;
        double sp_vs_g1_s3 = us_g1 / r3.us;
        best_speedup_vs_g1[bi] = sp_vs_g1_best;
        best_stage_per_b[bi] = best_stage;

        printf("=== B=%d PERF (us/iter, lower=better) ===\n", M);
        printf("  G1 single-buffer fused : %.3f us\n", us_g1);
        printf("  G2 overlap  2-stage    : %.3f us   (overlap/single-buf = %.3fx, %.1f%% %s)\n",
               r2.us, r2.us / us_g1, std::fabs(1.0 - r2.us / us_g1) * 100.0,
               (r2.us < us_g1) ? "FASTER" : "slower");
        printf("  G2 overlap  3-stage    : %.3f us   (overlap/single-buf = %.3fx, %.1f%% %s)\n",
               r3.us, r3.us / us_g1, std::fabs(1.0 - r3.us / us_g1) * 100.0,
               (r3.us < us_g1) ? "FASTER" : "slower");
        printf("  separate 2 kernels(%d-st): %.3f us\n", best_stage, us_sep);
        printf("  BEST overlap = %d-stage @ %.3f us  ->  vs G1 %.3fx  |  "
               "vs separate %.3fx\n",
               best_stage, best_us, sp_vs_g1_best, best_us / us_sep);
        printf("  speedups vs G1: s2=%.3fx  s3=%.3fx\n", sp_vs_g1_s2,
               sp_vs_g1_s3);
      }
      cudaFree(dCA_bf2); cudaFree(dCA_f2); cudaFree(dCB_bf2); cudaFree(dCB_f2);
    }

    cudaFree(dAA); cudaFree(dCA_bf); cudaFree(dCA_f); cudaFree(dCB_bf);
    cudaFree(dCB_f); cudaFree(dHeadA); cudaFree(dHeadB);
  }

  printf("\n=== G2 GATE: %s ===\n",
         all_corr ? "PASS(correctness)" : "FAIL(correctness)");
  printf("SUMMARY_JSON {\"sm\":%d,"
         "\"occ_g1\":%d,\"regs_g1\":%d,\"smem_g1_kb\":%.1f,"
         "\"occ_s2\":%d,\"regs_s2\":%d,\"smem_s2_kb\":%.1f,"
         "\"occ_s3\":%d,\"regs_s3\":%d,\"smem_s3_kb\":%.1f,"
         "\"best_stage_b1\":%d,\"speedup_vs_g1_b1\":%.3f,"
         "\"best_stage_b8\":%d,\"speedup_vs_g1_b8\":%.3f,"
         "\"best_stage_b32\":%d,\"speedup_vs_g1_b32\":%.3f,"
         "\"corr_gate\":\"%s\"}\n",
         prop.multiProcessorCount, occ_g1, fa_g1.numRegs, g1_smem / 1024.0,
         o2.blocks_per_sm, o2.regs, o2.smem / 1024.0,
         o3.blocks_per_sm, o3.regs, o3.smem / 1024.0,
         best_stage_per_b[0], best_speedup_vs_g1[0],
         best_stage_per_b[1], best_speedup_vs_g1[1],
         best_stage_per_b[2], best_speedup_vs_g1[2],
         all_corr ? "PASS" : "FAIL");

  cudaFree(dWA); cudaFree(dWB);
  cudaStreamDestroy(stream);
  return all_corr ? 0 : 2;
}

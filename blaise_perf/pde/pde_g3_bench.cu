// PDE G3 gate microbench (standalone, sm_100). REAL numbers on GPU0.
//
// G3 = device-resident DATA-DEPENDENT CONTROL FLOW (Indexer -> top-k -> attention
// gather). Compares:
//
//   (A) DEVICE-RESIDENT : ONE persistent cooperative kernel doing
//        score -> grid barrier -> device top-k select -> grid barrier ->
//        data-dependent gather+reduce. The selection NEVER leaves the device.
//   (B) HOST-ORCHESTRATED (today's pattern): score kernel -> cudaMemcpy scores
//        d2h -> HOST top-k -> cudaMemcpy indices h2d -> gather kernel. The real
//        d2h+h2d copies + stream syncs are the "execution gap" being removed.
//
// Both are checked against an INDEPENDENT CPU reference (CPU computes scores,
// exact top-k with the same deterministic tiebreak, and the weighted gather).
// We also isolate the raw d2h+h2d+sync cost that (A) eliminates, and report the
// persistent control-flow kernel's occupancy.
//
// Gate:
//   Correctness: A == B == CPU (out cos >= 0.999999) AND the device top-k INDEX
//                SET == CPU top-k index set, at M in {1,8,32}.
//   Perf: latency(A) vs latency(B) at M in {1,8,32}; speedup; plus the measured
//         d2h+h2d+sync cost removed.
//
// Build: nvcc -std=c++17 -arch=sm_100 -O3 -o pde_g3_bench pde_g3_bench.cu
#include "pde_g3_dev_ctrl.cuh"

#include <cooperative_groups.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <set>
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

// Representative decode/index shapes. C = candidate blocks/query; K = top-k.
// Two (C,K) settings exercised: a "small-k" and a "large-k" like DSA index_topk.
struct Shape { int C; int K; const char* name; };

static inline float fill_val(uint64_t i, uint64_t salt) {
  uint64_t z = (i + salt) * 0x9E3779B97F4A7C15ull;
  z ^= z >> 29;
  z *= 0xBF58476D1CE4E5B9ull;
  z ^= z >> 32;
  return ((float)(z & 0xFFFF) / 32768.0f) - 1.0f;
}

// ---------------------------------------------------------------------------
// Independent CPU reference: scores, exact top-k (deterministic tiebreak =
// higher score, lower index on tie), and the softmax-weighted gather+reduce.
// This is the ground truth — NEVER a device self-comparison.
// ---------------------------------------------------------------------------
struct CpuRef {
  std::vector<float> scores;   // [Q,C]
  std::vector<int> sel_idx;    // [Q,K]
  std::vector<float> out;      // [Q,D]
};

static CpuRef cpu_reference(const std::vector<float>& q,
                            const std::vector<float>& blocks, int Q, int C,
                            int K) {
  const int D = g3::kD;
  CpuRef r;
  r.scores.assign((size_t)Q * C, 0.0f);
  r.sel_idx.assign((size_t)Q * K, -1);
  r.out.assign((size_t)Q * D, 0.0f);
  for (int qi = 0; qi < Q; ++qi) {
    // scores
    for (int c = 0; c < C; ++c) {
      double acc = 0.0;
      for (int d = 0; d < D; ++d)
        acc += (double)q[(size_t)qi * D + d] * (double)blocks[(size_t)c * D + d];
      r.scores[(size_t)qi * C + c] = (float)acc;
    }
    // exact top-k by (score desc, index asc) — partial_sort on indices with the
    // SAME tiebreak the device uses, so the selected INDEX SET matches exactly.
    std::vector<int> ord(C);
    std::iota(ord.begin(), ord.end(), 0);
    const float* sc = &r.scores[(size_t)qi * C];
    std::partial_sort(
        ord.begin(), ord.begin() + K, ord.end(), [&](int a, int b) {
          if (sc[a] != sc[b]) return sc[a] > sc[b];
          return a < b;  // lower index wins ties (matches device)
        });
    std::vector<float> selv(K);
    for (int j = 0; j < K; ++j) {
      r.sel_idx[(size_t)qi * K + j] = ord[j];
      selv[j] = sc[ord[j]];
    }
    // softmax-weighted gather+reduce over the selected blocks
    float m = -FLT_MAX;
    for (int j = 0; j < K; ++j) m = std::max(m, selv[j]);
    double denom = 0.0;
    for (int j = 0; j < K; ++j) denom += std::exp((double)selv[j] - m);
    double inv = denom > 0 ? 1.0 / denom : 0.0;
    std::vector<double> acc(D, 0.0);
    for (int j = 0; j < K; ++j) {
      double w = std::exp((double)selv[j] - m) * inv;
      const float* brow = &blocks[(size_t)ord[j] * D];
      for (int d = 0; d < D; ++d) acc[d] += w * (double)brow[d];
    }
    for (int d = 0; d < D; ++d) r.out[(size_t)qi * D + d] = (float)acc[d];
  }
  return r;
}

// HOST top-k for the host-orchestrated baseline (the realistic "today" step the
// host runs on the d2h'd scores). Same deterministic tiebreak. Returns indices
// [Q,K] AND the selected scores [Q,K] (so the gather kernel can read them h2d).
static void host_topk(const std::vector<float>& scores, int Q, int C, int K,
                      std::vector<int>& idx, std::vector<float>& sel) {
  idx.assign((size_t)Q * K, -1);
  sel.assign((size_t)Q * K, 0.0f);
  std::vector<int> ord(C);
  for (int qi = 0; qi < Q; ++qi) {
    std::iota(ord.begin(), ord.end(), 0);
    const float* sc = &scores[(size_t)qi * C];
    std::partial_sort(ord.begin(), ord.begin() + K, ord.end(),
                      [&](int a, int b) {
                        if (sc[a] != sc[b]) return sc[a] > sc[b];
                        return a < b;
                      });
    for (int j = 0; j < K; ++j) {
      idx[(size_t)qi * K + j] = ord[j];
      sel[(size_t)qi * K + j] = sc[ord[j]];
    }
  }
}

struct CmpStat { double cos; double max_abs_err; };
static CmpStat compare(const std::vector<float>& got,
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

// Compare two top-k INDEX SETS per query. Returns (#queries with identical set,
// total queries, #queries whose ordered list also matches, max set-diff size).
struct IdxCmp { int set_match; int ordered_match; int total; int max_diff; };
static IdxCmp compare_idx(const std::vector<int>& got, const std::vector<int>& ref,
                          int Q, int K) {
  IdxCmp r{0, 0, Q, 0};
  for (int qi = 0; qi < Q; ++qi) {
    std::set<int> sg(got.begin() + (size_t)qi * K,
                     got.begin() + (size_t)(qi + 1) * K);
    std::set<int> sr(ref.begin() + (size_t)qi * K,
                     ref.begin() + (size_t)(qi + 1) * K);
    if (sg == sr) r.set_match++;
    else {
      int diff = 0;
      for (int x : sg) if (!sr.count(x)) diff++;
      r.max_diff = std::max(r.max_diff, diff);
    }
    bool ord = true;
    for (int j = 0; j < K; ++j)
      if (got[(size_t)qi * K + j] != ref[(size_t)qi * K + j]) { ord = false; break; }
    if (ord) r.ordered_match++;
  }
  return r;
}

static int occ_blocks_per_sm(const void* kernel, int bt, size_t smem) {
  int b = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, kernel, bt, smem));
  return b;
}

int main() {
  int dev = 0;
  CK(cudaSetDevice(dev));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, dev));
  printf("device: %s  SMs=%d  cc=%d.%d  coopLaunch=%d  smemPerSM=%zu KB  "
         "smemPerBlockOptin=%d KB\n",
         prop.name, prop.multiProcessorCount, prop.major, prop.minor,
         prop.cooperativeLaunch, prop.sharedMemPerMultiprocessor / 1024,
         (int)(prop.sharedMemPerBlockOptin / 1024));

  const int D = g3::kD;
  const int Ms[3] = {1, 8, 32};
  Shape shapes[2] = {{2048, 256, "C2048_K256"}, {4096, 2048, "C4096_K2048"}};

  bool all_corr = true;

  // For the JSON summary (per shape, per M): speedup A-vs-B and removed copy us.
  double sj_speedup[2][3] = {{0}};
  double sj_removed_us[2][3] = {{0}};
  double sj_A_us[2][3] = {{0}};
  double sj_B_us[2][3] = {{0}};
  int sj_idx_match[2][3] = {{0}};
  int occ_dev = 0, occ_sel = 0, regs_dev = 0;

  cudaStream_t stream;
  CK(cudaStreamCreate(&stream));

  for (int si = 0; si < 2; ++si) {
    const int C = shapes[si].C;
    const int K = shapes[si].K;
    printf("\n############ SHAPE %s : C=%d K=%d D=%d ############\n",
           shapes[si].name, C, K, D);

    // ---- candidate blocks shared across M (depend only on C) ----
    std::vector<float> hBlocks((size_t)C * D);
    for (size_t i = 0; i < hBlocks.size(); ++i) hBlocks[i] = fill_val(i, 101 + si);
    float* dBlocks;
    CK(cudaMalloc(&dBlocks, hBlocks.size() * sizeof(float)));
    CK(cudaMemcpy(dBlocks, hBlocks.data(), hBlocks.size() * sizeof(float),
                  cudaMemcpyHostToDevice));

    // ---- smem + occupancy (depends on C only) ----
    const size_t devA_smem = g3::device_resident_smem_bytes(C);
    const size_t gather_smem = g3::gather_only_smem_bytes();
    const size_t sel_smem = g3::select_only_smem_bytes(C);
    CK(cudaFuncSetAttribute((void*)g3::kDeviceResident,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            (int)devA_smem));
    CK(cudaFuncSetAttribute((void*)g3::kSelectOnlyPersistent,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            (int)sel_smem));
    int o_dev = occ_blocks_per_sm((void*)g3::kDeviceResident, g3::kBlockThreads,
                                  devA_smem);
    int o_sel = occ_blocks_per_sm((void*)g3::kSelectOnlyPersistent,
                                  g3::kBlockThreads, sel_smem);
    cudaFuncAttributes fa_dev;
    CK(cudaFuncGetAttributes(&fa_dev, (void*)g3::kDeviceResident));
    if (si == 0) { occ_dev = o_dev; occ_sel = o_sel; regs_dev = fa_dev.numRegs; }
    printf("OCC: kDeviceResident %d blk/SM (%d CTAs) regs=%d smem=%.1fKB | "
           "kSelectOnly %d blk/SM smem=%.1fKB\n",
           o_dev, o_dev * prop.multiProcessorCount, fa_dev.numRegs,
           devA_smem / 1024.0, o_sel, sel_smem / 1024.0);

    // Persistent grid for (A) sized to occupancy; capped at Q later per-M.
    GridPlan planA;
    CK(plan_persistent_grid((void*)g3::kDeviceResident, g3::kBlockThreads,
                            devA_smem, dev, &planA));

    for (int mi = 0; mi < 3; ++mi) {
      const int Q = Ms[mi];
      printf("\n==== M(Q)=%d  shape=%s ====\n", Q, shapes[si].name);

      // ---- queries ----
      std::vector<float> hQ((size_t)Q * D);
      for (size_t i = 0; i < hQ.size(); ++i) hQ[i] = fill_val(i, 7 + mi * 3 + si);

      // ---- CPU reference (ground truth) ----
      CpuRef ref = cpu_reference(hQ, hBlocks, Q, C, K);

      // ---- device buffers ----
      float *dQ, *dScores, *dSelScore, *dOut;
      int* dSelIdx;
      CK(cudaMalloc(&dQ, (size_t)Q * D * sizeof(float)));
      CK(cudaMalloc(&dScores, (size_t)Q * C * sizeof(float)));
      CK(cudaMalloc(&dSelIdx, (size_t)Q * K * sizeof(int)));
      CK(cudaMalloc(&dSelScore, (size_t)Q * K * sizeof(float)));
      CK(cudaMalloc(&dOut, (size_t)Q * D * sizeof(float)));
      CK(cudaMemcpy(dQ, hQ.data(), hQ.size() * sizeof(float),
                    cudaMemcpyHostToDevice));

      g3::Problem p{dQ, dBlocks, dScores, dSelIdx, dSelScore, dOut, Q, C, K};

      // Cooperative grid is capped at Q CTAs (one CTA owns a query; extra CTAs
      // would just idle, and cg grid sync requires every launched CTA resident).
      int gridA = std::min(planA.grid_blocks, Q);
      if (gridA < 1) gridA = 1;

      // =================== (A) DEVICE-RESIDENT ===================
      auto launchA = [&]() {
        void* args[] = {&p};
        CK(cudaLaunchCooperativeKernel((void*)g3::kDeviceResident, gridA,
                                       g3::kBlockThreads, args, devA_smem,
                                       stream));
      };
      // functional run -> copy out + selected indices for correctness
      CK(cudaMemsetAsync(dOut, 0, (size_t)Q * D * sizeof(float), stream));
      launchA();
      CK(cudaStreamSynchronize(stream));
      std::vector<float> A_out((size_t)Q * D);
      std::vector<int> A_idx((size_t)Q * K);
      CK(cudaMemcpy(A_out.data(), dOut, A_out.size() * sizeof(float),
                    cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(A_idx.data(), dSelIdx, A_idx.size() * sizeof(int),
                    cudaMemcpyDeviceToHost));

      // timing (A): pure on-device, no d2h/h2d inside the loop.
      double A_us;
      {
        const int REPS = 100, WARM = 20;
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        for (int w = 0; w < WARM; ++w) launchA();
        CK(cudaStreamSynchronize(stream));
        CK(cudaEventRecord(e0, stream));
        for (int r = 0; r < REPS; ++r) launchA();
        CK(cudaEventRecord(e1, stream)); CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
        A_us = ms * 1000.0 / REPS;
        cudaEventDestroy(e0); cudaEventDestroy(e1);
      }

      // =================== (B) HOST-ORCHESTRATED ===================
      // score kernel -> d2h scores -> HOST top-k -> h2d (idx,sel) -> gather kernel.
      CK(cudaFuncSetAttribute((void*)g3::kGatherOnly,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              (int)gather_smem));
      int score_grid = std::min(prop.multiProcessorCount * 4, Q * 32);
      if (score_grid < Q) score_grid = Q;     // >= one CTA per query is fine (grid-stride)
      if (score_grid < 1) score_grid = 1;
      int gather_grid = std::min(prop.multiProcessorCount * 4, Q);
      if (gather_grid < 1) gather_grid = 1;

      std::vector<float> hScores((size_t)Q * C);
      std::vector<int> hSelIdx;
      std::vector<float> hSelScore;
      std::vector<float> B_out((size_t)Q * D);

      auto runB = [&](std::vector<float>* out_capture) {
        // 1) score on device
        g3::kScoreOnly<<<score_grid, g3::kBlockThreads, 0, stream>>>(p);
        // 2) d2h scores (REAL copy + implied sync to read on host)
        CK(cudaMemcpyAsync(hScores.data(), dScores,
                           hScores.size() * sizeof(float),
                           cudaMemcpyDeviceToHost, stream));
        CK(cudaStreamSynchronize(stream));
        // 3) HOST top-k (the data-dependent decision on the host)
        host_topk(hScores, Q, C, K, hSelIdx, hSelScore);
        // 4) h2d selected idx + scores (REAL copies)
        CK(cudaMemcpyAsync(dSelIdx, hSelIdx.data(),
                           hSelIdx.size() * sizeof(int), cudaMemcpyHostToDevice,
                           stream));
        CK(cudaMemcpyAsync(dSelScore, hSelScore.data(),
                           hSelScore.size() * sizeof(float),
                           cudaMemcpyHostToDevice, stream));
        // 5) gather+reduce on device
        g3::kGatherOnly<<<gather_grid, g3::kBlockThreads, gather_smem, stream>>>(p);
        if (out_capture) {
          CK(cudaMemcpyAsync(out_capture->data(), dOut,
                             out_capture->size() * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
        }
        CK(cudaStreamSynchronize(stream));
      };
      CK(cudaMemsetAsync(dOut, 0, (size_t)Q * D * sizeof(float), stream));
      runB(&B_out);
      std::vector<int> B_idx = hSelIdx;  // host-selected idx set for this run

      double B_us;
      {
        const int REPS = 50, WARM = 10;     // B includes host work -> fewer reps
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        for (int w = 0; w < WARM; ++w) runB(nullptr);
        // wall-clock around the whole orchestrated sequence (events bracket the
        // device work but the loop body includes the host top-k + syncs).
        CK(cudaEventRecord(e0, stream));
        for (int r = 0; r < REPS; ++r) runB(nullptr);
        CK(cudaEventRecord(e1, stream)); CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
        B_us = ms * 1000.0 / REPS;
        cudaEventDestroy(e0); cudaEventDestroy(e1);
      }

      // =========== isolate the d2h + h2d + sync cost (A removes this) ===========
      // Measure ONLY the copies + syncs the host path needs (no host top-k, no
      // kernels) so the "execution gap removed" is explicit.
      double copy_us;
      {
        const int REPS = 100, WARM = 20;
        auto copies = [&]() {
          CK(cudaMemcpyAsync(hScores.data(), dScores,
                             hScores.size() * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
          CK(cudaStreamSynchronize(stream));   // host must see scores
          CK(cudaMemcpyAsync(dSelIdx, hSelIdx.data(),
                             hSelIdx.size() * sizeof(int),
                             cudaMemcpyHostToDevice, stream));
          CK(cudaMemcpyAsync(dSelScore, hSelScore.data(),
                             hSelScore.size() * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
          CK(cudaStreamSynchronize(stream));   // device must see indices
        };
        for (int w = 0; w < WARM; ++w) copies();
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        CK(cudaEventRecord(e0, stream));
        for (int r = 0; r < REPS; ++r) copies();
        CK(cudaEventRecord(e1, stream)); CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
        copy_us = ms * 1000.0 / REPS;
        cudaEventDestroy(e0); cudaEventDestroy(e1);
      }

      // =================== CORRECTNESS ===================
      CmpStat cA = compare(A_out, ref.out);
      CmpStat cB = compare(B_out, ref.out);
      IdxCmp ia = compare_idx(A_idx, ref.sel_idx, Q, K);   // device vs CPU
      IdxCmp ib = compare_idx(B_idx, ref.sel_idx, Q, K);   // host  vs CPU
      bool ok = (cA.cos >= 0.999999) && (cB.cos >= 0.999999) &&
                (ia.set_match == Q) && (ib.set_match == Q);
      all_corr = all_corr && ok;
      sj_idx_match[si][mi] = ia.set_match;

      printf("CORR vs CPU: A.out cos=%.8f maxabs=%.3e | B.out cos=%.8f "
             "maxabs=%.3e\n", cA.cos, cA.max_abs_err, cB.cos, cB.max_abs_err);
      printf("TOPK index-set vs CPU: device %d/%d sets match (ordered %d/%d, "
             "max_set_diff=%d) | host %d/%d sets match (ordered %d/%d) -> %s\n",
             ia.set_match, ia.total, ia.ordered_match, ia.total, ia.max_diff,
             ib.set_match, ib.total, ib.ordered_match, ib.total,
             ok ? "PASS" : "FAIL");

      // =================== PERF ===================
      double speedup = (A_us > 0) ? B_us / A_us : 0.0;
      sj_speedup[si][mi] = speedup;
      sj_removed_us[si][mi] = copy_us;
      sj_A_us[si][mi] = A_us;
      sj_B_us[si][mi] = B_us;
      printf("PERF (us/iter, lower=better):\n");
      printf("  (A) device-resident   : %.3f us\n", A_us);
      printf("  (B) host-orchestrated : %.3f us   -> A is %.3fx %s\n", B_us,
             speedup, (A_us < B_us) ? "FASTER" : "slower");
      printf("  d2h+h2d+sync removed  : %.3f us  (the capture-illegal round-trip "
             "(A) dissolves)\n", copy_us);
      printf("  grids: (A) coop=%d CTAs | (B) score=%d gather=%d CTAs\n", gridA,
             score_grid, gather_grid);

      cudaFree(dQ); cudaFree(dScores); cudaFree(dSelIdx); cudaFree(dSelScore);
      cudaFree(dOut);
    }
    cudaFree(dBlocks);
  }

  printf("\n=== G3 GATE: %s ===\n",
         all_corr ? "PASS(correctness)" : "FAIL(correctness)");
  printf("SUMMARY_JSON {\"sm\":%d,\"occ_dev_resident\":%d,\"occ_select\":%d,"
         "\"regs_dev\":%d,"
         "\"s0_name\":\"%s\",\"s1_name\":\"%s\","
         "\"s0_A_us\":[%.3f,%.3f,%.3f],\"s0_B_us\":[%.3f,%.3f,%.3f],"
         "\"s0_speedup\":[%.3f,%.3f,%.3f],\"s0_removed_us\":[%.3f,%.3f,%.3f],"
         "\"s0_idxmatch_per_M\":[%d,%d,%d],"
         "\"s1_A_us\":[%.3f,%.3f,%.3f],\"s1_B_us\":[%.3f,%.3f,%.3f],"
         "\"s1_speedup\":[%.3f,%.3f,%.3f],\"s1_removed_us\":[%.3f,%.3f,%.3f],"
         "\"s1_idxmatch_per_M\":[%d,%d,%d],"
         "\"M\":[1,8,32],\"corr_gate\":\"%s\"}\n",
         prop.multiProcessorCount, occ_dev, occ_sel, regs_dev,
         shapes[0].name, shapes[1].name,
         sj_A_us[0][0], sj_A_us[0][1], sj_A_us[0][2],
         sj_B_us[0][0], sj_B_us[0][1], sj_B_us[0][2],
         sj_speedup[0][0], sj_speedup[0][1], sj_speedup[0][2],
         sj_removed_us[0][0], sj_removed_us[0][1], sj_removed_us[0][2],
         sj_idx_match[0][0], sj_idx_match[0][1], sj_idx_match[0][2],
         sj_A_us[1][0], sj_A_us[1][1], sj_A_us[1][2],
         sj_B_us[1][0], sj_B_us[1][1], sj_B_us[1][2],
         sj_speedup[1][0], sj_speedup[1][1], sj_speedup[1][2],
         sj_removed_us[1][0], sj_removed_us[1][1], sj_removed_us[1][2],
         sj_idx_match[1][0], sj_idx_match[1][1], sj_idx_match[1][2],
         all_corr ? "PASS" : "FAIL");

  cudaStreamDestroy(stream);
  return all_corr ? 0 : 2;
}

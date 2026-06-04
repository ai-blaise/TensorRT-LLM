// SPDX-License-Identifier: Apache-2.0
// In-kernel KVarN dequant-on-read microbenchmark for the op-trt MLA latent path.
//
// Question (per round-4 finding): folding KVarN's Sinkhorn-var-norm + RTN dequant
// directly into the decode KV read (zero python staging, no extra HBM round-trip)
// should remove the 242%-of-budget staging overhead the python path showed at b32.
//
// This standalone kernel reproduces EXACTLY the per-read work the fused decode
// path must do, on the real DeepSeek-V3.2 MLA latent dims, and times it against
// (a) the fp8 dequantCopy cost (native fp8 read baseline) and
// (b) a plain fp16 latent read (lower bound: just move the bytes).
//
// KVarN dequant per block (group=64 tokens, ckv=512, k_pe=64):
//   ckv: unpack 4-bit -> (q*s_row[token] + zp[token]) * s_col[ch]  (V-orient rows=token)
//        -> inverse Hadamard along the 512 channel axis (FWHT, O(D log D), in smem)
//   k_pe: unpack 2-bit -> (q*s_row[ch] + zp[ch]) * s_col[token]   (K-orient rows=ch)
//        -> inverse Hadamard along the 64 channel axis
// Output: fp16 ckv[group,512] + k_pe[group,64], the exact tensors BatchMLA decode consumes.
//
// KEY EFFICIENCY CHOICE vs the python staging path: the python path un-rotated with
// a dense [N,64,512]@[512,512] matmul. In-kernel we use the Fast Walsh-Hadamard
// Transform (butterfly, 9 stages for 512) entirely in shared memory -> O(D log D),
// no global round-trip, no matmul. That is the lever that makes in-kernel a win.

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

// DeepSeek-V3.2 MLA latent dims
static constexpr int GROUP   = 64;   // tokens_per_block == KVarN tile rows
static constexpr int DCKV    = 512;  // compressed_kv latent dim (power of 2)
static constexpr int DPE     = 64;   // k_pe rope dim (power of 2)
static constexpr int CKV_BITS = 4;
static constexpr int PE_BITS  = 2;
static constexpr int CKV_PACK = 8 / CKV_BITS;  // 2 nibbles/byte
static constexpr int PE_PACK  = 8 / PE_BITS;   // 4 vals/byte
static constexpr float INV_SQRT_DCKV = 0.044194173824159f; // 1/sqrt(512)
static constexpr float INV_SQRT_DPE  = 0.125f;             // 1/sqrt(64)

// ---- Packed record layout (matches KVarNLatentPool._compute_layout) ----
// Per block, byte-contiguous:
//   ckv_q [G * DCKV/CKV_PACK] u8 | ckv_srow[G] f16 | ckv_zp[G] f16 | ckv_scol[DCKV] f16 |
//   pe_q  [DPE * G/PE_PACK]   u8 | pe_srow[DPE] f16| pe_zp[DPE] f16 | pe_scol[G] f16
struct BlockOffsets {
    int ckv_q, ckv_srow, ckv_zp, ckv_scol, pe_q, pe_srow, pe_zp, pe_scol, total;
};
__host__ __device__ inline BlockOffsets layout() {
    BlockOffsets o{};
    int off = 0;
    o.ckv_q    = off; off += GROUP * (DCKV / CKV_PACK);
    o.ckv_srow = off; off += GROUP * 2;
    o.ckv_zp   = off; off += GROUP * 2;
    o.ckv_scol = off; off += DCKV * 2;
    o.pe_q     = off; off += DPE * (GROUP / PE_PACK);
    o.pe_srow  = off; off += DPE * 2;
    o.pe_zp    = off; off += DPE * 2;
    o.pe_scol  = off; off += GROUP * 2;
    o.total = off;
    return o;
}

// In-shared-memory FWHT over the leading `n` (power of 2) dim of `data` laid out
// [rows][n], one row per token. Each thread owns one (row, j) lane; butterfly
// stages combine pairs. Normalization (1/sqrt(n)) applied by caller.
__device__ inline void fwht_rows(float* sm, int rows, int n, int tid, int nthreads) {
    for (int len = 1; len < n; len <<= 1) {
        for (int idx = tid; idx < rows * (n / 2); idx += nthreads) {
            int row = idx / (n / 2);
            int k = idx % (n / 2);
            int blk = (k / len) * (2 * len);
            int j = k % len;
            int a = blk + j;
            int b = a + len;
            float* r = sm + row * n;
            float u = r[a], v = r[b];
            r[a] = u + v;
            r[b] = u - v;
        }
        __syncthreads();
    }
}

// ===================================================================
// KVarN in-kernel dequant: one CTA per latent block.
// ===================================================================
__global__ void kvarn_dequant_kernel(
    uint8_t const* __restrict__ store,  // [num_blocks, total_bytes]
    int const* __restrict__ block_ids,  // [N] which blocks to restore this step
    __half* __restrict__ ckv_out,       // [N, GROUP, DCKV]
    __half* __restrict__ kpe_out,       // [N, GROUP, DPE]
    int total_bytes)
{
    BlockOffsets L = layout();
    int n = blockIdx.x;
    int bid = block_ids[n];
    uint8_t const* slot = store + (size_t)bid * total_bytes;
    int tid = threadIdx.x;
    int nth = blockDim.x;

    // smem: ckv rows [GROUP][DCKV] + pe rows [DPE][GROUP] (pe stored channel-major).
    extern __shared__ float sm[];
    float* sm_ckv = sm;                 // GROUP*DCKV floats
    float* sm_pe  = sm + GROUP * DCKV;  // DPE*GROUP floats

    uint8_t const* ckv_q   = slot + L.ckv_q;
    __half const*  ckv_srow= reinterpret_cast<__half const*>(slot + L.ckv_srow);
    __half const*  ckv_zp  = reinterpret_cast<__half const*>(slot + L.ckv_zp);
    __half const*  ckv_scol= reinterpret_cast<__half const*>(slot + L.ckv_scol);
    uint8_t const* pe_q    = slot + L.pe_q;
    __half const*  pe_srow = reinterpret_cast<__half const*>(slot + L.pe_srow);
    __half const*  pe_zp   = reinterpret_cast<__half const*>(slot + L.pe_zp);
    __half const*  pe_scol = reinterpret_cast<__half const*>(slot + L.pe_scol);

    // ---- ckv: unpack 4-bit + RTN-dequant into rotated frame -> sm_ckv[token][ch] ----
    // V-orient: rows=token (s_row per token), cols=ch (s_col per ch).
    for (int idx = tid; idx < GROUP * DCKV; idx += nth) {
        int tok = idx / DCKV;
        int ch  = idx % DCKV;
        int byte_i = ch / CKV_PACK;
        int nib    = ch % CKV_PACK;
        uint8_t packed = ckv_q[tok * (DCKV / CKV_PACK) + byte_i];
        int q = (packed >> (nib * CKV_BITS)) & ((1 << CKV_BITS) - 1);
        float s_row = __half2float(ckv_srow[tok]);
        float zp    = __half2float(ckv_zp[tok]);
        float s_col = __half2float(ckv_scol[ch]);
        sm_ckv[idx] = ((float)q * s_row + zp) * s_col;
    }
    // ---- k_pe: unpack 2-bit + RTN into rotated frame -> sm_pe[ch][token] ----
    // K-orient: rows=ch (s_row per ch), cols=token (s_col per token).
    for (int idx = tid; idx < DPE * GROUP; idx += nth) {
        int ch  = idx / GROUP;
        int tok = idx % GROUP;
        int byte_i = tok / PE_PACK;
        int slot_i = tok % PE_PACK;
        uint8_t packed = pe_q[ch * (GROUP / PE_PACK) + byte_i];
        int q = (packed >> (slot_i * PE_BITS)) & ((1 << PE_BITS) - 1);
        float s_row = __half2float(pe_srow[ch]);
        float zp    = __half2float(pe_zp[ch]);
        float s_col = __half2float(pe_scol[tok]);
        sm_pe[idx] = ((float)q * s_row + zp) * s_col;
    }
    __syncthreads();

    // ---- inverse Hadamard along channel axis ----
    // ckv: FWHT along DCKV for each of GROUP rows.
    fwht_rows(sm_ckv, GROUP, DCKV, tid, nth);
    // pe: channel axis is the LEADING dim here (sm_pe[ch][token]); we need FWHT
    // along ch for each token. Treat as GROUP rows of DPE by striding: do it as
    // a transposed-FWHT. Simpler: FWHT over DPE with token as the row stride.
    // sm_pe is [DPE][GROUP]; for fixed token, elements at stride GROUP. We run a
    // strided butterfly.
    for (int len = 1; len < DPE; len <<= 1) {
        for (int idx = tid; idx < GROUP * (DPE / 2); idx += nth) {
            int tok = idx / (DPE / 2);
            int k = idx % (DPE / 2);
            int blk = (k / len) * (2 * len);
            int j = k % len;
            int a = blk + j;
            int b = a + len;
            float u = sm_pe[a * GROUP + tok];
            float v = sm_pe[b * GROUP + tok];
            sm_pe[a * GROUP + tok] = u + v;
            sm_pe[b * GROUP + tok] = u - v;
        }
        __syncthreads();
    }

    // ---- write fp16 out, applying 1/sqrt(D) Hadamard normalization ----
    for (int idx = tid; idx < GROUP * DCKV; idx += nth) {
        ckv_out[(size_t)n * GROUP * DCKV + idx] = __float2half(sm_ckv[idx] * INV_SQRT_DCKV);
    }
    for (int idx = tid; idx < GROUP * DPE; idx += nth) {
        int ch = idx / GROUP;     // sm_pe is [DPE][GROUP] = [ch][token]
        int tok = idx % GROUP;
        // out layout [N, GROUP, DPE] = [token, ch]
        kpe_out[(size_t)n * GROUP * DPE + tok * DPE + ch] =
            __float2half(sm_pe[idx] * INV_SQRT_DPE);
    }
}

// ===================================================================
// Baseline A: fp8 dequantCopy cost (native fp8 latent read, 1 byte/elem,
// single global scale). Mirrors mlaKernels.cu dequantCopy: load fp8, *scale,
// store fp16. This is the "fp8 dense KV read" the directive compares against.
// ===================================================================
__global__ void fp8_dequant_kernel(
    uint8_t const* __restrict__ store,  // [num_blocks, GROUP*(DCKV+DPE)] fp8 (1B/elem)
    int const* __restrict__ block_ids,
    __half* __restrict__ out,           // [N, GROUP, DCKV+DPE]
    float scale)
{
    constexpr int LAT = DCKV + DPE;  // 576
    int n = blockIdx.x;
    int bid = block_ids[n];
    uint8_t const* slot = store + (size_t)bid * GROUP * LAT;
    for (int idx = threadIdx.x; idx < GROUP * LAT; idx += blockDim.x) {
        // interpret byte as e4m3 via __nv_fp8_e4m3
        __nv_fp8_e4m3 v = reinterpret_cast<__nv_fp8_e4m3 const*>(slot)[idx];
        out[(size_t)n * GROUP * LAT + idx] = __float2half(float(v) * scale);
    }
}

// Baseline B: plain fp16 latent copy (lower bound: just move bytes, no quant).
__global__ void fp16_copy_kernel(
    __half const* __restrict__ store,   // [num_blocks, GROUP*576]
    int const* __restrict__ block_ids,
    __half* __restrict__ out)
{
    constexpr int LAT = DCKV + DPE;
    int n = blockIdx.x;
    int bid = block_ids[n];
    __half const* slot = store + (size_t)bid * GROUP * LAT;
    for (int idx = threadIdx.x; idx < GROUP * LAT; idx += blockDim.x) {
        out[(size_t)n * GROUP * LAT + idx] = slot[idx];
    }
}

template <typename F>
float time_ms(F fn, int iters, int warmup) {
    cudaEvent_t ev_s, ev_e; CK(cudaEventCreate(&ev_s)); CK(cudaEventCreate(&ev_e));
    for (int i = 0; i < warmup; i++) fn();
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(ev_s));
    for (int i = 0; i < iters; i++) fn();
    CK(cudaEventRecord(ev_e)); CK(cudaEventSynchronize(ev_e));
    float ms; CK(cudaEventElapsedTime(&ms, ev_s, ev_e));
    CK(cudaEventDestroy(ev_s)); CK(cudaEventDestroy(ev_e));
    return ms / iters;
}

// Load a binary file into a host vector<uint8_t>; returns false if missing.
static bool load_bin(char const* path, std::vector<uint8_t>& out) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    out.resize(sz);
    size_t r = fread(out.data(), 1, sz, f); fclose(f);
    return r == (size_t)sz;
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 32;   // blocks restored per step (sparse topK)
    int iters = 200, warmup = 50;
    BlockOffsets L = layout();

    // Try to load the python-generated fixture (proves correctness). Fall back to
    // random bytes (timing-only) if absent.
    std::vector<uint8_t> h_store, h_ref_ckv_u8, h_ref_kpe_u8;
    bool have_fixture = load_bin("fixture_store.bin", h_store)
        && load_bin("fixture_ref_ckv.bin", h_ref_ckv_u8)
        && load_bin("fixture_ref_kpe.bin", h_ref_kpe_u8);
    int NUM_BLOCKS;
    if (have_fixture) {
        NUM_BLOCKS = (int)(h_store.size() / L.total);
        if (NUM_BLOCKS < N) N = NUM_BLOCKS;
        printf("[fixture loaded] %d blocks, total_bytes/block=%d\n", NUM_BLOCKS, L.total);
    } else {
        NUM_BLOCKS = std::max(N * 4, 2048);
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> ub(0, 255);
        h_store.assign((size_t)NUM_BLOCKS * L.total, 0);
        for (auto& b : h_store) b = ub(rng);
        printf("[no fixture] timing-only with %d random blocks\n", NUM_BLOCKS);
    }

    printf("=== KVarN in-kernel dequant microbench (B200) ===\n");
    printf("dims: group=%d ckv=%d k_pe=%d  ckv_bits=%d pe_bits=%d\n",
           GROUP, DCKV, DPE, CKV_BITS, PE_BITS);
    printf("packed bytes/block=%d  fp16 bytes/block=%d  fp8 bytes/block=%d\n",
           L.total, 2 * GROUP * (DCKV + DPE), GROUP * (DCKV + DPE));
    printf("N (blocks restored/step) = %d, pool=%d blocks\n\n", N, NUM_BLOCKS);

    // when fixture present, restore the FIRST N blocks in order (ids 0..N-1) so we
    // can compare against ref. Otherwise stride.
    std::vector<int> h_ids(N);
    for (int i = 0; i < N; i++) h_ids[i] = have_fixture ? i : (i * 7) % NUM_BLOCKS;

    uint8_t* d_store; int* d_ids;
    __half *d_ckv, *d_kpe;
    CK(cudaMalloc(&d_store, h_store.size()));
    CK(cudaMemcpy(d_store, h_store.data(), h_store.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_ids, N * sizeof(int)));
    CK(cudaMemcpy(d_ids, h_ids.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_ckv, (size_t)N * GROUP * DCKV * sizeof(__half)));
    CK(cudaMalloc(&d_kpe, (size_t)N * GROUP * DPE * sizeof(__half)));

    size_t smem = ((size_t)GROUP * DCKV + (size_t)DPE * GROUP) * sizeof(float);
    int threads = 256;
    CK(cudaFuncSetAttribute(kvarn_dequant_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    printf("kvarn kernel smem = %zu bytes (%.1f KB), threads=%d\n\n", smem, smem/1024.0, threads);

    auto run_kvarn = [&]() {
        kvarn_dequant_kernel<<<N, threads, smem>>>(d_store, d_ids, d_ckv, d_kpe, L.total);
    };
    run_kvarn(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    // ---- correctness vs python fixture (cosine over all restored blocks) ----
    if (have_fixture) {
        std::vector<__half> out_ckv((size_t)N * GROUP * DCKV), out_kpe((size_t)N * GROUP * DPE);
        CK(cudaMemcpy(out_ckv.data(), d_ckv, out_ckv.size()*sizeof(__half), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(out_kpe.data(), d_kpe, out_kpe.size()*sizeof(__half), cudaMemcpyDeviceToHost));
        __half const* ref_ckv = reinterpret_cast<__half const*>(h_ref_ckv_u8.data());
        __half const* ref_kpe = reinterpret_cast<__half const*>(h_ref_kpe_u8.data());
        auto cos = [](__half const* a, __half const* b, size_t n){
            double dot=0, na=0, nb=0;
            for (size_t i=0;i<n;i++){ double x=__half2float(a[i]), y=__half2float(b[i]); dot+=x*y; na+=x*x; nb+=y*y; }
            return dot / (sqrt(na)*sqrt(nb)+1e-12);
        };
        double cos_ckv = cos(out_ckv.data(), ref_ckv, out_ckv.size());
        double cos_kpe = cos(out_kpe.data(), ref_kpe, out_kpe.size());
        // also max abs err on ckv
        double maxerr=0;
        for (size_t i=0;i<out_ckv.size();i++)
            maxerr = std::max(maxerr, fabs((double)__half2float(out_ckv[i]) - (double)__half2float(ref_ckv[i])));
        printf("--- correctness vs python dequant_latent_block ---\n");
        printf("cos_ckv (in-kernel FWHT vs python matmul-Hadamard) = %.6f\n", cos_ckv);
        printf("cos_kpe = %.6f   ckv_max_abs_err = %.4g\n\n", cos_kpe, maxerr);
    }

    float t_kvarn = time_ms(run_kvarn, iters, warmup);

    // fp8 baseline
    uint8_t* d_fp8; __half* d_fp8out;
    CK(cudaMalloc(&d_fp8, (size_t)NUM_BLOCKS * GROUP * (DCKV + DPE)));
    CK(cudaMemset(d_fp8, 0x38, (size_t)NUM_BLOCKS * GROUP * (DCKV + DPE))); // ~0.5 in e4m3
    CK(cudaMalloc(&d_fp8out, (size_t)N * GROUP * (DCKV + DPE) * sizeof(__half)));
    auto run_fp8 = [&]() {
        fp8_dequant_kernel<<<N, threads>>>(d_fp8, d_ids, d_fp8out, 1.0f);
    };
    run_fp8(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    float t_fp8 = time_ms(run_fp8, iters, warmup);

    // fp16 copy baseline
    __half* d_fp16; __half* d_fp16out;
    CK(cudaMalloc(&d_fp16, (size_t)NUM_BLOCKS * GROUP * (DCKV + DPE) * sizeof(__half)));
    CK(cudaMemset(d_fp16, 0, (size_t)NUM_BLOCKS * GROUP * (DCKV + DPE) * sizeof(__half)));
    CK(cudaMalloc(&d_fp16out, (size_t)N * GROUP * (DCKV + DPE) * sizeof(__half)));
    auto run_fp16 = [&]() {
        fp16_copy_kernel<<<N, threads>>>(d_fp16, d_ids, d_fp16out);
    };
    run_fp16(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    float t_fp16 = time_ms(run_fp16, iters, warmup);

    printf("--- per-step latency (N=%d blocks) ---\n", N);
    printf("KVarN in-kernel dequant : %8.2f us   (%6.3f us/block)\n", t_kvarn*1e3, t_kvarn*1e3/N);
    printf("fp8 dequantCopy read    : %8.2f us   (%6.3f us/block)\n", t_fp8*1e3, t_fp8*1e3/N);
    printf("fp16 plain copy (floor) : %8.2f us   (%6.3f us/block)\n", t_fp16*1e3, t_fp16*1e3/N);
    printf("\nKVarN / fp16-floor = %.2fx   KVarN / fp8 = %.2fx\n", t_kvarn/t_fp16, t_kvarn/t_fp8);

    // budget context: round-4 logged ~341 us/layer/tok budget; python staged b32 = 808.4us = 237%
    double budget_us = 341.0;
    printf("\n--- vs round-4 decode budget (~%.0f us/layer/tok) ---\n", budget_us);
    printf("KVarN in-kernel  : %.1f us = %.1f%% of budget\n", t_kvarn*1e3, t_kvarn*1e3/budget_us*100);
    printf("python staged b32: 808.4 us = 237.1%% of budget  (round-4 logged)\n");

    CK(cudaFree(d_store)); CK(cudaFree(d_ids)); CK(cudaFree(d_ckv)); CK(cudaFree(d_kpe));
    CK(cudaFree(d_fp8)); CK(cudaFree(d_fp8out)); CK(cudaFree(d_fp16)); CK(cudaFree(d_fp16out));
    return 0;
}

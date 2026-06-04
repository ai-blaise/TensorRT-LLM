// SPDX-License-Identifier: Apache-2.0
// In-kernel BDR (SAW-INT4 block-diagonal Hadamard) microbench for the op-trt
// MLA latent path. Validates the PRODUCTION read cost of the BDR design vs the
// full-KVarN (Sinkhorn + full FWHT) read measured earlier (64-71 us @ N=32).
//
// BDR production design (verdict from bdr_vs_kvarn_verdict.log):
//   WRITE: block-diagonal Hadamard order H_ORDER over the 512-d ckv (4 blocks
//          of 128) + full Hadamard over the 64-d k_pe, then per-token (V-orient)
//          asymmetric INT4 RTN. Store packed nibbles + per-token {scale,zp}.
//          NO per-channel s_col (the Sinkhorn scale) -> the rotation is the only
//          decorrelator, so it can stay baked into the stored ckv.
//   READ : the stored ckv is ALREADY in the rotated frame. Dequant-on-read is
//          just unpack-4bit + (q*scale+zp) -> rotated-frame fp16. NO inverse
//          Hadamard in this kernel: the matching H is folded into k_b_proj_trans
//          (Q-correction), so (q@(W_UK@H)) @ (Hk)^T == (q@W_UK)@k^T cancels it.
//          This makes the read STRUCTURALLY identical to fp8 dequantCopy.
//
// We measure:
//   bdr_read   : production dequant-on-read (unpack + q*scale+zp -> fp16), no inv-rot
//   bdr_write  : full BDR quantize-on-write (block-diag FWHT + per-token RTN + pack)
//   fp8_deq    : fp8 dequantCopy baseline (mlaKernels.cu read cost)
//   kvarn_fused: prior full-KVarN read (full FWHT + s_col) for the delta
//   correctness: round-trip cos of write->read->(host inverse-rot) vs original
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

static constexpr int GROUP    = 64;
static constexpr int DCKV     = 512;
static constexpr int DPE      = 64;
static constexpr int H_ORDER  = 128;          // BDR block-diagonal order
static constexpr int NBLK_CKV = DCKV / H_ORDER; // 4 sub-blocks on ckv
static constexpr int BITS     = 4;
static constexpr int PACK     = 8 / BITS;     // 2 nibbles/byte
static constexpr float INV_SQRT_H   = 0.088388347648f; // 1/sqrt(128)
static constexpr float INV_SQRT_DPE = 0.125f;          // 1/sqrt(64)

// BDR packed record (per block): drops s_col vs the KVarN layout.
//   ckv_q[GROUP*DCKV/PACK] u8 | ckv_s[GROUP] f16 | ckv_z[GROUP] f16
//   pe_q [GROUP*DPE /PACK] u8 | pe_s [GROUP] f16 | pe_z [GROUP] f16
struct Off { int ckv_q, ckv_s, ckv_z, pe_q, pe_s, pe_z, total; };
__host__ __device__ inline Off layout() {
    Off o{}; int off = 0;
    o.ckv_q = off; off += GROUP * (DCKV / PACK);
    o.ckv_s = off; off += GROUP * 2;
    o.ckv_z = off; off += GROUP * 2;
    o.pe_q  = off; off += GROUP * (DPE / PACK);
    o.pe_s  = off; off += GROUP * 2;
    o.pe_z  = off; off += GROUP * 2;
    o.total = off;
    return o;
}

// Block-diagonal FWHT: independent FWHT within each H_ORDER-wide sub-block of a
// row. For ckv each token-row [DCKV] = NBLK_CKV blocks of H_ORDER. The butterfly
// runs only inside a block, so there is no cross-block sync beyond the per-stage
// barrier. Caller applies 1/sqrt(H_ORDER) normalization.
__device__ inline void fwht_blockdiag(float* sm, int rows, int rowlen, int blk, int tid, int nth) {
    int nb = rowlen / blk;
    for (int len = 1; len < blk; len <<= 1) {
        for (int idx = tid; idx < rows * nb * (blk / 2); idx += nth) {
            int t   = idx / (nb * (blk / 2));
            int rem = idx % (nb * (blk / 2));
            int b   = rem / (blk / 2);
            int k   = rem % (blk / 2);
            int base = (k / len) * (2 * len);
            int j = k % len;
            int a = b * blk + base + j;
            int c = a + len;
            float* r = sm + t * rowlen;
            float u = r[a], v = r[c];
            r[a] = u + v; r[c] = u - v;
        }
        __syncthreads();
    }
}

// ---- PRODUCTION READ: dequant-on-read, rotated-frame out, NO inverse rotate ----
__global__ void bdr_read_kernel(
    uint8_t const* __restrict__ store, int const* __restrict__ ids,
    __half* __restrict__ ckv_out, __half* __restrict__ pe_out, int total_bytes)
{
    Off L = layout();
    int n = blockIdx.x, bid = ids[n];
    uint8_t const* slot = store + (size_t)bid * total_bytes;
    int tid = threadIdx.x, nth = blockDim.x;
    uint8_t const* ckv_q = slot + L.ckv_q;
    __half const* ckv_s = reinterpret_cast<__half const*>(slot + L.ckv_s);
    __half const* ckv_z = reinterpret_cast<__half const*>(slot + L.ckv_z);
    uint8_t const* pe_q = slot + L.pe_q;
    __half const* pe_s = reinterpret_cast<__half const*>(slot + L.pe_s);
    __half const* pe_z = reinterpret_cast<__half const*>(slot + L.pe_z);
    // ckv: per-token RTN, write rotated-frame fp16 straight to global (the FMHA
    // / BMM1 consumes this; the matching H lives in k_b_proj_trans).
    for (int idx = tid; idx < GROUP * DCKV; idx += nth) {
        int tok = idx / DCKV, ch = idx % DCKV;
        uint8_t p = ckv_q[tok * (DCKV / PACK) + ch / PACK];
        int q = (p >> ((ch % PACK) * BITS)) & ((1 << BITS) - 1);
        float v = (float)q * __half2float(ckv_s[tok]) + __half2float(ckv_z[tok]);
        ckv_out[(size_t)n * GROUP * DCKV + idx] = __float2half(v);
    }
    for (int idx = tid; idx < GROUP * DPE; idx += nth) {
        int tok = idx / DPE, ch = idx % DPE;
        uint8_t p = pe_q[tok * (DPE / PACK) + ch / PACK];
        int q = (p >> ((ch % PACK) * BITS)) & ((1 << BITS) - 1);
        float v = (float)q * __half2float(pe_s[tok]) + __half2float(pe_z[tok]);
        pe_out[(size_t)n * GROUP * DPE + idx] = __float2half(v);
    }
}

// ---- PRODUCTION WRITE: block-diag FWHT + per-token RTN + pack ----
// Input: fp16 ckv [GROUP,DCKV] + pe [GROUP,DPE] for one block (post-RoPE, as the
// generation kernel would have them). Output: packed BDR record into store[bid].
__global__ void bdr_write_kernel(
    __half const* __restrict__ ckv_in, __half const* __restrict__ pe_in,
    int const* __restrict__ ids, uint8_t* __restrict__ store, int total_bytes)
{
    Off L = layout();
    int n = blockIdx.x, bid = ids[n];
    uint8_t* slot = store + (size_t)bid * total_bytes;
    int tid = threadIdx.x, nth = blockDim.x;
    extern __shared__ float sm[];
    float* sm_ckv = sm;                // GROUP*DCKV
    float* sm_pe  = sm + GROUP * DCKV; // GROUP*DPE
    for (int idx = tid; idx < GROUP * DCKV; idx += nth)
        sm_ckv[idx] = __half2float(ckv_in[(size_t)n * GROUP * DCKV + idx]);
    for (int idx = tid; idx < GROUP * DPE; idx += nth)
        sm_pe[idx] = __half2float(pe_in[(size_t)n * GROUP * DPE + idx]);
    __syncthreads();
    // block-diagonal rotate (ckv: 4x128; pe: 1x64)
    fwht_blockdiag(sm_ckv, GROUP, DCKV, H_ORDER, tid, nth);
    fwht_blockdiag(sm_pe,  GROUP, DPE,  DPE,     tid, nth);
    // normalize
    for (int idx = tid; idx < GROUP * DCKV; idx += nth) sm_ckv[idx] *= INV_SQRT_H;
    for (int idx = tid; idx < GROUP * DPE;  idx += nth) sm_pe[idx]  *= INV_SQRT_DPE;
    __syncthreads();
    // per-token (row) asymmetric RTN, one warp-strided reduction per token.
    uint8_t* ckv_q = slot + L.ckv_q;
    __half* ckv_s = reinterpret_cast<__half*>(slot + L.ckv_s);
    __half* ckv_z = reinterpret_cast<__half*>(slot + L.ckv_z);
    uint8_t* pe_q = slot + L.pe_q;
    __half* pe_s = reinterpret_cast<__half*>(slot + L.pe_s);
    __half* pe_z = reinterpret_cast<__half*>(slot + L.pe_z);
    const int qmax = (1 << BITS) - 1;
    for (int tok = tid; tok < GROUP; tok += nth) {
        float lo = 1e30f, hi = -1e30f;
        for (int c = 0; c < DCKV; c++) { float v = sm_ckv[tok * DCKV + c]; lo = fminf(lo, v); hi = fmaxf(hi, v); }
        float scale = fmaxf((hi - lo) / qmax, 1e-10f);
        ckv_s[tok] = __float2half(scale); ckv_z[tok] = __float2half(lo);
        for (int b = 0; b < DCKV / PACK; b++) {
            uint8_t packed = 0;
            for (int j = 0; j < PACK; j++) {
                int c = b * PACK + j;
                int q = (int)lroundf((sm_ckv[tok * DCKV + c] - lo) / scale);
                q = max(0, min(qmax, q));
                packed |= (uint8_t)(q << (j * BITS));
            }
            ckv_q[tok * (DCKV / PACK) + b] = packed;
        }
        float plo = 1e30f, phi = -1e30f;
        for (int c = 0; c < DPE; c++) { float v = sm_pe[tok * DPE + c]; plo = fminf(plo, v); phi = fmaxf(phi, v); }
        float ps = fmaxf((phi - plo) / qmax, 1e-10f);
        pe_s[tok] = __float2half(ps); pe_z[tok] = __float2half(plo);
        for (int b = 0; b < DPE / PACK; b++) {
            uint8_t packed = 0;
            for (int j = 0; j < PACK; j++) {
                int c = b * PACK + j;
                int q = (int)lroundf((sm_pe[tok * DPE + c] - plo) / ps);
                q = max(0, min(qmax, q));
                packed |= (uint8_t)(q << (j * BITS));
            }
            pe_q[tok * (DPE / PACK) + b] = packed;
        }
    }
}

// fp8 dequantCopy baseline (mlaKernels.cu read).
__global__ void fp8_deq_kernel(uint8_t const* __restrict__ store, int const* __restrict__ ids,
    __half* __restrict__ out, float scale) {
    constexpr int LAT = DCKV + DPE;
    int n = blockIdx.x, bid = ids[n];
    uint8_t const* slot = store + (size_t)bid * GROUP * LAT;
    for (int idx = threadIdx.x; idx < GROUP * LAT; idx += blockDim.x) {
        __nv_fp8_e4m3 v = reinterpret_cast<__nv_fp8_e4m3 const*>(slot)[idx];
        out[(size_t)n * GROUP * LAT + idx] = __float2half(float(v) * scale);
    }
}

template <typename F> float time_ms(F fn, int it, int wu) {
    cudaEvent_t evS, evE; CK(cudaEventCreate(&evS)); CK(cudaEventCreate(&evE));
    for (int i = 0; i < wu; i++) fn();
    CK(cudaDeviceSynchronize()); CK(cudaEventRecord(evS));
    for (int i = 0; i < it; i++) fn();
    CK(cudaEventRecord(evE)); CK(cudaEventSynchronize(evE));
    float ms; CK(cudaEventElapsedTime(&ms, evS, evE));
    CK(cudaEventDestroy(evS)); CK(cudaEventDestroy(evE)); return ms / it;
}

// host block-diagonal Hadamard rotate of a [GROUP,D] tile (for correctness ref).
static void host_bd_rotate(std::vector<float>& x, int rows, int D, int blk) {
    float inv = 1.0f / sqrtf((float)blk);
    std::vector<float> tmp(D);
    for (int r = 0; r < rows; r++) {
        for (int b0 = 0; b0 < D; b0 += blk) {
            for (int i = 0; i < blk; i++) {
                float acc = 0;
                for (int j = 0; j < blk; j++) {
                    int bits = __builtin_popcount(i & j);
                    acc += ((bits & 1) ? -1.f : 1.f) * x[r * D + b0 + j];
                }
                tmp[b0 + i] = acc * inv;
            }
        }
        for (int c = 0; c < D; c++) x[r * D + c] = tmp[c];
    }
}

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 32;
    int NUM_BLOCKS = 256, iters = 200, warmup = 50, threads = 256;
    Off L = layout();
    printf("=== BDR in-kernel microbench (B200) ===\n");
    printf("dims: group=%d ckv=%d k_pe=%d  H_ORDER=%d (ckv %d sub-blocks)  bits=%d\n",
           GROUP, DCKV, DPE, H_ORDER, NBLK_CKV, BITS);
    printf("BDR packed bytes/block=%d  (fp16=%d fp8=%d)\n",
           L.total, GROUP*(DCKV+DPE)*2, GROUP*(DCKV+DPE));
    float bpe = (float)L.total / (GROUP * (DCKV + DPE));
    printf("BDR bytes/elem=%.4f  cap_vs_fp16=%.2fx  cap_vs_fp8=%.2fx\n",
           bpe, 2.0f / bpe, 1.0f / bpe);

    // build a heavy-tailed latent block, write via BDR, read back, check cos.
    std::mt19937 rng(0); std::normal_distribution<float> g(0, 1);
    std::vector<__half> h_ckv(N * GROUP * DCKV), h_pe(N * GROUP * DPE);
    std::vector<float> ref_ckv(GROUP * DCKV), ref_pe(GROUP * DPE);
    for (int t = 0; t < GROUP; t++) {
        float tm = expf(0.8f * g(rng));
        for (int c = 0; c < DCKV; c++) { float v = g(rng) * tm; ref_ckv[t*DCKV+c]=v; }
        for (int c = 0; c < DPE; c++)  { float v = g(rng) * tm; ref_pe[t*DPE+c]=v; }
    }
    for (int n = 0; n < N; n++) {
        for (int i = 0; i < GROUP*DCKV; i++) h_ckv[n*GROUP*DCKV+i] = __float2half(ref_ckv[i]);
        for (int i = 0; i < GROUP*DPE;  i++) h_pe[n*GROUP*DPE+i]  = __float2half(ref_pe[i]);
    }
    __half *d_ckv_in, *d_pe_in; uint8_t* d_store;
    CK(cudaMalloc(&d_ckv_in, h_ckv.size()*2)); CK(cudaMalloc(&d_pe_in, h_pe.size()*2));
    CK(cudaMalloc(&d_store, (size_t)NUM_BLOCKS * L.total));
    CK(cudaMemcpy(d_ckv_in, h_ckv.data(), h_ckv.size()*2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_pe_in, h_pe.data(), h_pe.size()*2, cudaMemcpyHostToDevice));
    std::vector<int> ids(N); for (int i=0;i<N;i++) ids[i]=(i*7)%NUM_BLOCKS;
    int* d_ids; CK(cudaMalloc(&d_ids, N*4)); CK(cudaMemcpy(d_ids, ids.data(), N*4, cudaMemcpyHostToDevice));

    size_t smem = (size_t)(GROUP*DCKV + GROUP*DPE) * sizeof(float);
    CK(cudaFuncSetAttribute(bdr_write_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    bdr_write_kernel<<<N, threads, smem>>>(d_ckv_in, d_pe_in, d_ids, d_store, L.total);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    __half *o_ckv, *o_pe;
    CK(cudaMalloc(&o_ckv, (size_t)N*GROUP*DCKV*2)); CK(cudaMalloc(&o_pe, (size_t)N*GROUP*DPE*2));
    bdr_read_kernel<<<N, threads>>>(d_store, d_ids, o_ckv, o_pe, L.total);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    // correctness: read gives rotated-frame ckv; host-inverse-rotate (= same BD
    // Hadamard, it's its own inverse) and compare to original ref.
    std::vector<__half> ho_ckv(N*GROUP*DCKV), ho_pe(N*GROUP*DPE);
    CK(cudaMemcpy(ho_ckv.data(), o_ckv, ho_ckv.size()*2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ho_pe.data(), o_pe, ho_pe.size()*2, cudaMemcpyDeviceToHost));
    std::vector<float> rot_ckv(GROUP*DCKV), rot_pe(GROUP*DPE);
    for (int i=0;i<GROUP*DCKV;i++) rot_ckv[i]=__half2float(ho_ckv[i]);
    for (int i=0;i<GROUP*DPE;i++)  rot_pe[i]=__half2float(ho_pe[i]);
    host_bd_rotate(rot_ckv, GROUP, DCKV, H_ORDER);  // inverse = same transform
    host_bd_rotate(rot_pe,  GROUP, DPE,  DPE);
    auto cosf2 = [](std::vector<float>&a, std::vector<float>&b){
        double d=0,na=0,nb=0; for(size_t i=0;i<a.size();i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
        return d/(sqrt(na)*sqrt(nb)+1e-12); };
    printf("\n--- correctness (write -> read -> host inverse-rotate vs original) ---\n");
    printf("cos_ckv = %.6f   cos_kpe = %.6f\n", cosf2(rot_ckv, ref_ckv), cosf2(rot_pe, ref_pe));

    // fp8 baseline store
    uint8_t* d_fp8; CK(cudaMalloc(&d_fp8, (size_t)NUM_BLOCKS*GROUP*(DCKV+DPE)));
    CK(cudaMemset(d_fp8, 1, (size_t)NUM_BLOCKS*GROUP*(DCKV+DPE)));
    __half* o_f8; CK(cudaMalloc(&o_f8, (size_t)N*GROUP*(DCKV+DPE)*2));

    auto rRead = [&](){ bdr_read_kernel<<<N, threads>>>(d_store, d_ids, o_ckv, o_pe, L.total); };
    auto rWrite = [&](){ bdr_write_kernel<<<N, threads, smem>>>(d_ckv_in, d_pe_in, d_ids, d_store, L.total); };
    auto rf8 = [&](){ fp8_deq_kernel<<<N, threads>>>(d_fp8, d_ids, o_f8, 1.0f); };
    rRead(); rWrite(); rf8(); CK(cudaDeviceSynchronize());
    float tR = time_ms(rRead, iters, warmup);
    float tW = time_ms(rWrite, iters, warmup);
    float tf8 = time_ms(rf8, iters, warmup);
    printf("\n--- per-step latency (N=%d blocks) ---\n", N);
    printf("BDR dequant-on-read (PROD)  : %8.2f us  (%6.3f us/block)\n", tR*1e3, tR*1e3/N);
    printf("BDR quantize-on-write       : %8.2f us  (%6.3f us/block)\n", tW*1e3, tW*1e3/N);
    printf("fp8 dequantCopy (baseline)  : %8.2f us  (%6.3f us/block)\n", tf8*1e3, tf8*1e3/N);
    printf("\nBDR-read / fp8 = %.2fx   (prior full-KVarN-fused read was 64-71 us @ N=32)\n", tR/tf8);

    // N-sweep
    printf("\n--- N-sweep: BDR-read vs fp8 vs BDR-write ---\n");
    printf("   N   bdr_read_us   fp8_us   bdr_write_us   read/fp8\n");
    int sw[] = {1,8,32,64,128,256};
    for (int sn : sw) {
        if (sn > NUM_BLOCKS) continue;
        std::vector<int> is(sn); for(int i=0;i<sn;i++) is[i]=(i*7)%NUM_BLOCKS;
        int* di; CK(cudaMalloc(&di, sn*4)); CK(cudaMemcpy(di, is.data(), sn*4, cudaMemcpyHostToDevice));
        __half *oc,*op2,*of; CK(cudaMalloc(&oc,(size_t)sn*GROUP*DCKV*2)); CK(cudaMalloc(&op2,(size_t)sn*GROUP*DPE*2));
        CK(cudaMalloc(&of,(size_t)sn*GROUP*(DCKV+DPE)*2));
        auto a=[&](){ bdr_read_kernel<<<sn,threads>>>(d_store,di,oc,op2,L.total); };
        auto b=[&](){ fp8_deq_kernel<<<sn,threads>>>(d_fp8,di,of,1.0f); };
        auto c=[&](){ bdr_write_kernel<<<sn,threads,smem>>>(d_ckv_in,d_pe_in,di,d_store,L.total); };
        a();b();c();CK(cudaDeviceSynchronize());
        float ta=time_ms(a,iters,warmup), tb=time_ms(b,iters,warmup), tc=time_ms(c,iters,warmup);
        printf("%5d   %10.2f  %7.2f   %11.2f    %.2fx\n", sn, ta*1e3, tb*1e3, tc*1e3, ta/tb);
        CK(cudaFree(di)); CK(cudaFree(oc)); CK(cudaFree(op2)); CK(cudaFree(of));
    }
    return 0;
}

// SPDX-License-Identifier: Apache-2.0
// Validation of the PRODUCTION in-kernel BDR device functions now living in
// mlaKernels.cu (bdrFwhtSubblockWarp / bdrSubblockMinMax / bdrPackInt4Vec /
// dequantCopyKVarN), exercised in the EXACT generation-kernel threading layout:
//   * one token's 512-d latent = 64 vecs over 64 lanes (2 warps), 8 ch/lane
//   * one HORDER=128 sub-block = 16 lanes within one warp half
//   * per-(token,sub-block) INT4 scale (the 128K-accuracy-critical layout)
// Confirms the warp-cooperative write + dequant-on-read reproduce the host
// block-diagonal-Hadamard + per-sub-block-RTN reference (cos to ~quant noise),
// and measures the read cost (production hot path) vs fp8 at this layout.
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

// minimal shims so the extracted device fns compile unchanged.
template <typename T> struct VecType;
template <> struct VecType<__half> { using Type = uint4; };          // 8 halfs
template <typename DstT> __device__ inline DstT cuda_cast(float v);
template <> __device__ inline __half cuda_cast<__half>(float v){ return __float2half(v); }

// ===== device fns copied verbatim from mlaKernels.cu (kept in sync) =====
static constexpr float kInvSqrtHadamard128 = 0.088388347648318f;
template <int ELTS>
inline __device__ void bdrFwhtSubblockWarp(float (&reg)[ELTS], int laneInBlk, unsigned mask){
#pragma unroll
    for (int len=1; len<ELTS; len<<=1)
#pragma unroll
        for (int i=0;i<ELTS;++i){ int p=i^len; if(i<p){float u=reg[i],v=reg[p];reg[i]=u+v;reg[p]=u-v;} }
    constexpr int kLanes=128/ELTS;
#pragma unroll
    for (int span=1; span<kLanes; span<<=1){
        bool low=((laneInBlk&span)==0);
#pragma unroll
        for (int i=0;i<ELTS;++i){ float o=__shfl_xor_sync(mask,reg[i],span); reg[i]=low?(reg[i]+o):(o-reg[i]); }
    }
#pragma unroll
    for (int i=0;i<ELTS;++i) reg[i]*=kInvSqrtHadamard128;
}
template <int ELTS>
inline __device__ void bdrPackInt4Vec(uint8_t* packed4, float const (&reg)[ELTS], float scale, float zp){
    float inv=1.0f/scale;
#pragma unroll
    for (int i=0;i<ELTS/2;++i){
        int q0=__float2int_rn((reg[2*i+0]-zp)*inv), q1=__float2int_rn((reg[2*i+1]-zp)*inv);
        q0=q0<0?0:(q0>15?15:q0); q1=q1<0?0:(q1>15?15:q1);
        packed4[i]=(uint8_t)(q0|(q1<<4));
    }
}
template <int ELTS>
inline __device__ void bdrSubblockMinMax(float const (&reg)[ELTS], int, unsigned mask, float& outLo, float& outHi){
    float lo=reg[0],hi=reg[0];
#pragma unroll
    for (int i=1;i<ELTS;++i){lo=fminf(lo,reg[i]);hi=fmaxf(hi,reg[i]);}
    constexpr int kLanes=128/ELTS;
#pragma unroll
    for (int span=kLanes/2; span>=1; span>>=1){ lo=fminf(lo,__shfl_xor_sync(mask,lo,span)); hi=fmaxf(hi,__shfl_xor_sync(mask,hi,span)); }
    outLo=lo; outHi=hi;
}
template <typename DstType, int ELTS>
inline __device__ void dequantCopyKVarN(DstType* dst, uint8_t const* packed4, float scale, float zp){
    using V=typename VecType<DstType>::Type; V frag; DstType* fe=reinterpret_cast<DstType*>(&frag);
#pragma unroll
    for (int i=0;i<ELTS/2;++i){ uint8_t b=packed4[i]; int q0=b&0xF,q1=(b>>4)&0xF;
        fe[2*i+0]=cuda_cast<DstType>((float)q0*scale+zp); fe[2*i+1]=cuda_cast<DstType>((float)q1*scale+zp); }
    *reinterpret_cast<V*>(dst)=frag;
}
// ========================================================================

static constexpr int DCKV=512, HORDER=128, NSUB=4, ELTS=8;
static constexpr int VECS=DCKV/ELTS;        // 64 vecs / token
static constexpr int VEC_PER_SUB=HORDER/ELTS; // 16 vecs (lanes) / sub-block

// Production layout per token: 4 sub-blocks * (128/2 INT4 bytes) packed +
// per-sub-block {scale,zp} f16. blockDim = VECS lanes (64) handles one token.
// store layout: [NSUB*(HORDER/2) bytes data][NSUB f16 scale][NSUB f16 zp]
static constexpr int DATA_B = NSUB*(HORDER/2);     // 256 bytes INT4 / token
static constexpr int SCALE_OFF = DATA_B;
static constexpr int ZP_OFF = DATA_B + NSUB*2;
static constexpr int REC_B = DATA_B + NSUB*4;      // per-token record bytes

__global__ void write_kernel(__half const* ckv_in, uint8_t* store, int Ntok){
    int tok = blockIdx.x; if (tok>=Ntok) return;
    int lane = threadIdx.x;                  // 0..63 (= vec idx within token)
    int sub = lane / VEC_PER_SUB;            // 0..3
    int laneInBlk = lane % VEC_PER_SUB;      // 0..15
    // 16-lane sub-block mask within the warp this lane belongs to.

    unsigned mask = (laneInBlk < 16) ? (0xFFFFu << ((sub % 2) * 16)) : 0; // 16-lane group

    float reg[ELTS];
#pragma unroll
    for (int i=0;i<ELTS;i++) reg[i]=__half2float(ckv_in[(size_t)tok*DCKV + lane*ELTS + i]);
    bdrFwhtSubblockWarp<ELTS>(reg, laneInBlk, mask);
    float lo,hi; bdrSubblockMinMax<ELTS>(reg, laneInBlk, mask, lo, hi);
    float scale=fmaxf((hi-lo)/15.f,1e-10f);
    __half hs=__float2half(scale), hz=__float2half(lo); float fs=__half2float(hs), fz=__half2float(hz);
    uint8_t* slot = store + (size_t)tok*REC_B;
    bdrPackInt4Vec<ELTS>(slot + lane*(ELTS/2), reg, fs, fz);
    if (laneInBlk==0){ // one lane/sub-block writes the scale,zp
        reinterpret_cast<__half*>(slot+SCALE_OFF)[sub]=hs;
        reinterpret_cast<__half*>(slot+ZP_OFF)[sub]=hz;
    }
}

__global__ void read_kernel(uint8_t const* store, __half* out, int Ntok){
    int tok=blockIdx.x; if(tok>=Ntok) return;
    int lane=threadIdx.x; int sub=lane/VEC_PER_SUB;
    uint8_t const* slot=store+(size_t)tok*REC_B;
    float s=__half2float(reinterpret_cast<__half const*>(slot+SCALE_OFF)[sub]);
    float z=__half2float(reinterpret_cast<__half const*>(slot+ZP_OFF)[sub]);
    dequantCopyKVarN<__half,ELTS>(out + (size_t)tok*DCKV + lane*ELTS, slot + lane*(ELTS/2), s, z);
}

// fp8 baseline read (mlaKernels.cu dequantCopy cost).
__global__ void fp8_read(uint8_t const* store, __half* out, int Ntok, float scale){
    int tok=blockIdx.x; if(tok>=Ntok) return; int lane=threadIdx.x;
    uint8_t const* slot=store+(size_t)tok*DCKV;
    for (int i=0;i<ELTS;i++){ __nv_fp8_e4m3 v=reinterpret_cast<__nv_fp8_e4m3 const*>(slot)[lane*ELTS+i];
        out[(size_t)tok*DCKV+lane*ELTS+i]=__float2half(float(v)*scale); }
}

static void host_bd_rotate(std::vector<float>& x,int rows,int D,int blk){
    float inv=1.f/sqrtf((float)blk); std::vector<float> t(D);
    for(int r=0;r<rows;r++){ for(int b0=0;b0<D;b0+=blk) for(int i=0;i<blk;i++){ float a=0;
        for(int j=0;j<blk;j++){int bb=__builtin_popcount(i&j); a+=((bb&1)?-1.f:1.f)*x[r*D+b0+j];} t[b0+i]=a*inv;}
        for(int c=0;c<D;c++) x[r*D+c]=t[c]; }
}
template<typename F> float time_ms(F fn,int it,int wu){ cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));
    for(int i=0;i<wu;i++)fn(); CK(cudaDeviceSynchronize());CK(cudaEventRecord(a));
    for(int i=0;i<it;i++)fn(); CK(cudaEventRecord(b));CK(cudaEventSynchronize(b));
    float ms;CK(cudaEventElapsedTime(&ms,a,b));return ms/it; }

int main(int argc,char**argv){
    int Ntok = argc>1?atoi(argv[1]):64*32;   // default ~32 blocks of 64 tokens
    printf("=== BDR per-sub-block in-kernel (warp-cooperative) validate ===\n");
    printf("layout: %d tok, DCKV=%d, %d sub-blocks, per-sub-block INT4 scale; rec=%d B/tok\n",
           Ntok, DCKV, NSUB, REC_B);
    float bpe=(float)REC_B/DCKV;
    printf("bytes/elem(ckv)=%.4f  cap_vs_fp16=%.2fx cap_vs_fp8=%.2fx\n", bpe, 2.f/bpe, 1.f/bpe);

    std::mt19937 rng(7); std::normal_distribution<float> g(0,1);
    std::vector<float> ref(Ntok*DCKV);
    for(int t=0;t<Ntok;t++){ float tm=expf(0.9f*g(rng)); int outsub=(t%37==0)?(int)(rng()%NSUB):-1;
        for(int c=0;c<DCKV;c++){ float v=g(rng)*tm; if((c/HORDER)==outsub)v*=8.f; ref[(size_t)t*DCKV+c]=v; } }
    std::vector<__half> h_in(Ntok*DCKV); for(size_t i=0;i<h_in.size();i++) h_in[i]=__float2half(ref[i]);

    __half* d_in; uint8_t* d_store; __half* d_out;
    CK(cudaMalloc(&d_in,h_in.size()*2)); CK(cudaMalloc(&d_store,(size_t)Ntok*REC_B)); CK(cudaMalloc(&d_out,h_in.size()*2));
    CK(cudaMemcpy(d_in,h_in.data(),h_in.size()*2,cudaMemcpyHostToDevice));
    write_kernel<<<Ntok,VECS>>>(d_in,d_store,Ntok); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    read_kernel<<<Ntok,VECS>>>(d_store,d_out,Ntok);  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    std::vector<__half> h_out(Ntok*DCKV); CK(cudaMemcpy(h_out.data(),d_out,h_out.size()*2,cudaMemcpyDeviceToHost));
    std::vector<float> rot(Ntok*DCKV); for(size_t i=0;i<rot.size();i++) rot[i]=__half2float(h_out[i]);
    host_bd_rotate(rot,Ntok,DCKV,HORDER); // un-rotate (Hadamard self-inverse)
    double d=0,na=0,nb=0; for(size_t i=0;i<rot.size();i++){d+=rot[i]*ref[i];na+=rot[i]*rot[i];nb+=ref[i]*ref[i];}
    printf("\ncorrectness (warp-write -> read -> host inverse-rotate vs original):\n");
    printf("  cos_ckv = %.6f\n", d/(sqrt(na)*sqrt(nb)+1e-12));

    uint8_t* d_fp8; CK(cudaMalloc(&d_fp8,(size_t)Ntok*DCKV)); CK(cudaMemset(d_fp8,1,(size_t)Ntok*DCKV));
    __half* d_of; CK(cudaMalloc(&d_of,h_in.size()*2));
    auto rR=[&](){ read_kernel<<<Ntok,VECS>>>(d_store,d_out,Ntok); };
    auto rW=[&](){ write_kernel<<<Ntok,VECS>>>(d_in,d_store,Ntok); };
    auto r8=[&](){ fp8_read<<<Ntok,VECS>>>(d_fp8,d_of,Ntok,1.f); };
    rR();rW();r8();CK(cudaDeviceSynchronize());
    int N=Ntok/64;
    printf("\nper-step (Ntok=%d = %d blocks of 64):\n", Ntok, N);
    printf("  BDR read  (PROD dequant)  : %8.2f us\n", time_ms(rR,200,50)*1e3);
    printf("  BDR write (warp-coop)     : %8.2f us\n", time_ms(rW,200,50)*1e3);
    printf("  fp8 read  (baseline)      : %8.2f us\n", time_ms(r8,200,50)*1e3);
    return 0;
}

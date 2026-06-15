#!/usr/bin/env python3
# PDE radix-select top-k — ITER 3 (v5). Attacks the PASS COUNT (dominant cost at
# final-topk = 8 full re-scans of C). Two variants vs v4:
#  A) SCORE-ONLY 32-bit radix (4 passes of 8-bit) + exact index tiebreak emit.
#     The composite key's top 32 bits ARE the score. Radix-select to the score
#     threshold (4 passes), then among elements AT the threshold-score, break ties
#     by index to pick exactly the right ones. With random fp scores ties are ~0,
#     so this is ~2x fewer passes. Still EXACT (full composite tiebreak in emit).
#  B) 11-bit / 6-pass (2048-bin single SMEM histogram) over full 64-bit key.
# Toggle via env PDE_VARIANT in {A,B}. Keeps warp-private hist (A) + float4 loads.
import os, time, math
import torch
import tensorrt_llm

import sys
VARIANT="A"
for a in sys.argv[1:]:
    if a.startswith("PDE_VARIANT="): VARIANT=a.split("=")[1]
VARIANT=os.environ.get("PDE_VARIANT",VARIANT)

# ---------- Variant A: 32-bit score radix (4 passes) + exact index tiebreak ----------
SRC_A = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
typedef unsigned long long ull;
__device__ __forceinline__ unsigned int float_to_okey(float f){unsigned int u=__float_as_uint(f);unsigned int m=(unsigned int)(-(int)(u>>31))|0x80000000u;return u^m;}

// 256 threads = 8 warps, warp-private 256-bin histograms. Radix-select on the
// 32-bit SCORE key only (4 passes). Then resolve: count strictly-greater scores
// (cnt_gt) and equal-score elements; emit all strictly-greater, then pick the
// lowest-index (cnt_eq tiebreak) equal-score elements to fill to k. Exact.
__global__ void __launch_bounds__(256) topk_v5a(const float* __restrict__ scores,
    const int* __restrict__ seq_lens, int* __restrict__ out_idx, int B,int C,int k){
  int row=blockIdx.x; if(row>=B) return; int tid=threadIdx.x; int warp=tid>>5;
  int sl=seq_lens[row];
  const float* srow=scores+(size_t)row*C; int* orow=out_idx+(size_t)row*k;
  extern __shared__ unsigned char smem[];
  float* s_sc=(float*)smem; unsigned int* s_key=(unsigned int*)(s_sc+C); // 32-bit score keys
  unsigned int* s_hist=(unsigned int*)(s_key+C); // 8*256 bins
  __shared__ int s_digit,s_acc; __shared__ unsigned int s_fill; __shared__ int s_keq_need;
  int valid=sl<C?sl:C; int kk=k<valid?k:valid;

  int C4=C&~3;
  const float4* srow4=reinterpret_cast<const float4*>(srow);
  for(int c4=tid;c4<(C4>>2);c4+=blockDim.x){ float4 v=srow4[c4]; int base=c4<<2;
    float v0=(base+0<sl)?v.x:-FLT_MAX, v1=(base+1<sl)?v.y:-FLT_MAX, v2=(base+2<sl)?v.z:-FLT_MAX, v3=(base+3<sl)?v.w:-FLT_MAX;
    s_sc[base+0]=v0; s_key[base+0]=float_to_okey(v0);
    s_sc[base+1]=v1; s_key[base+1]=float_to_okey(v1);
    s_sc[base+2]=v2; s_key[base+2]=float_to_okey(v2);
    s_sc[base+3]=v3; s_key[base+3]=float_to_okey(v3); }
  for(int c=C4+tid;c<C;c+=blockDim.x){ float v=(c<sl)?srow[c]:-FLT_MAX; s_sc[c]=v; s_key[c]=float_to_okey(v); }
  __syncthreads();

  unsigned int prefix=0,pmask=0; int krem=kk;
  for(int t=0;t<4;++t){ int sh=32-8*(t+1);
    for(int b=tid;b<8*256;b+=blockDim.x) s_hist[b]=0u; __syncthreads();
    unsigned int* myh=s_hist+warp*256;
    for(int c=tid;c<valid;c+=blockDim.x){ unsigned int key=s_key[c];
      if((key&pmask)==prefix) atomicAdd(&myh[(unsigned)((key>>sh)&255)],1u); }
    __syncthreads();
    for(int b=tid;b<256;b+=blockDim.x){ unsigned int s=s_hist[b];
      #pragma unroll
      for(int w=1;w<8;++w) s+=s_hist[w*256+b]; s_hist[b]=s; }
    __syncthreads();
    if(tid==0){int acc=0,dg=0;for(int bn=255;bn>=0;--bn){int cc=(int)s_hist[bn];if(acc+cc>=krem){dg=bn;break;}acc+=cc;}s_digit=dg;s_acc=acc;}
    __syncthreads();
    krem-=s_acc; prefix|=((unsigned int)s_digit)<<sh; pmask|=((unsigned int)255)<<sh; __syncthreads();
  }
  // prefix == threshold SCORE key (32-bit). krem = how many of the equal-score
  // (key==prefix) elements we still need (>=0). Elements with key>prefix: emit all.
  // Elements with key==prefix: emit the krem lowest-index ones (matches ~idx DESC tiebreak == idx ASC).
  unsigned int thr=prefix;
  if(tid==0){s_fill=0u; s_keq_need=krem;} __syncthreads();
  // strictly-greater: emit unconditionally
  for(int c=tid;c<valid;c+=blockDim.x){ unsigned int key=s_key[c];
    if(key>thr){unsigned p=atomicAdd(&s_fill,1u); if(p<(unsigned)k)orow[p]=c;} }
  __syncthreads();
  // equal-score: need the krem with SMALLEST index. Iterate indices ascending in
  // thread order; a thread emits c if it's among the first krem equals. We do a
  // simple device-side selection: each equal-c does atomicAdd to a shared eq-rank
  // counter; rank<krem => emit. Iterating c ascending across the grid is naturally
  // index-ordered because we sweep c = tid, tid+blockDim, ... but multiple warps
  // race. To stay EXACT (lowest indices), do a 2-pass: count equals < c is too slow;
  // instead: collect equals, then pick lowest. Equals are tiny (~ties=0 typ), so
  // single-thread serial pick is fine.
  __shared__ int s_eqcnt; if(tid==0) s_eqcnt=0; __syncthreads();
  // gather equal-score indices into the tail of s_key region? reuse s_hist as scratch is too small.
  // Simpler: thread0 scans for equals in ascending c and emits up to krem. valid<=8192 -> ok-ish but serial.
  if(tid==0){ int need=s_keq_need; unsigned p=s_fill;
    for(int c=0;c<valid && need>0;++c){ if(s_key[c]==thr){ if(p<(unsigned)k) orow[p]=c; p++; need--; } }
    s_fill=p; }
  __syncthreads();
  for(int j=(int)s_fill+tid;j<k;j+=blockDim.x) orow[j]=-1;
}
void launch_topk(torch::Tensor scores,torch::Tensor seq_lens,torch::Tensor out_idx,int k){
  int B=scores.size(0),C=scores.size(1);
  size_t smem=(size_t)C*4 + (size_t)C*4 + 8*256*4; // s_sc + s_key + hist
  cudaStream_t st=at::cuda::getCurrentCUDAStream();
  auto kern=topk_v5a;
  if(smem>48*1024)cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
  kern<<<B,256,smem,st>>>(scores.data_ptr<float>(),seq_lens.data_ptr<int>(),out_idx.data_ptr<int>(),B,C,k);
}
'''

# ---------- Variant B: 11-bit / 6-pass single 2048-bin SMEM histogram, full 64b key ----------
SRC_B = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
typedef unsigned long long ull;
__device__ __forceinline__ unsigned int float_to_okey(float f){unsigned int u=__float_as_uint(f);unsigned int m=(unsigned int)(-(int)(u>>31))|0x80000000u;return u^m;}
__device__ __forceinline__ ull make_key64(float s,int idx){return ((ull)float_to_okey(s)<<32)|(ull)(~(unsigned int)idx);}
// 6 passes of ~11-bit digits over 64 bits: 6*11=66 >= 64. Use digit widths summing to 64:
// {11,11,11,11,11,9} shifts. 2048-bin single SMEM histogram (8KB).
__global__ void __launch_bounds__(256) topk_v5b(const float* __restrict__ scores,
    const int* __restrict__ seq_lens, int* __restrict__ out_idx, int B,int C,int k){
  int row=blockIdx.x; if(row>=B) return; int tid=threadIdx.x; int sl=seq_lens[row];
  const float* srow=scores+(size_t)row*C; int* orow=out_idx+(size_t)row*k;
  extern __shared__ unsigned char smem[];
  float* s_sc=(float*)smem; unsigned int* s_hist=(unsigned int*)(s_sc+C); // 2048 bins
  __shared__ int s_digit,s_acc; __shared__ unsigned int s_fill;
  int valid=sl<C?sl:C; int kk=k<valid?k:valid;
  int C4=C&~3; const float4* srow4=reinterpret_cast<const float4*>(srow);
  for(int c4=tid;c4<(C4>>2);c4+=blockDim.x){ float4 v=srow4[c4]; int base=c4<<2;
    s_sc[base+0]=(base+0<sl)?v.x:-FLT_MAX; s_sc[base+1]=(base+1<sl)?v.y:-FLT_MAX;
    s_sc[base+2]=(base+2<sl)?v.z:-FLT_MAX; s_sc[base+3]=(base+3<sl)?v.w:-FLT_MAX; }
  for(int c=C4+tid;c<C;c+=blockDim.x) s_sc[c]=(c<sl)?srow[c]:-FLT_MAX;
  __syncthreads();
  const int widths[6]={11,11,11,11,11,9};
  ull prefix=0,pmask=0; int krem=kk; int donebits=0;
  for(int t=0;t<6;++t){ int w=widths[t]; int sh=64-donebits-w; int nb=1<<w; unsigned int bmask=nb-1;
    for(int b=tid;b<nb;b+=blockDim.x) s_hist[b]=0u; __syncthreads();
    for(int c=tid;c<valid;c+=blockDim.x){ ull key=make_key64(s_sc[c],c);
      if((key&pmask)==prefix) atomicAdd(&s_hist[(unsigned)((key>>sh)&bmask)],1u); }
    __syncthreads();
    if(tid==0){int acc=0,dg=0;for(int bn=nb-1;bn>=0;--bn){int cc=(int)s_hist[bn];if(acc+cc>=krem){dg=bn;break;}acc+=cc;}s_digit=dg;s_acc=acc;}
    __syncthreads();
    krem-=s_acc; prefix|=((ull)s_digit)<<sh; pmask|=((ull)bmask)<<sh; donebits+=w; __syncthreads();
  }
  ull thr=prefix;
  if(tid==0)s_fill=0u; __syncthreads();
  for(int c=tid;c<valid;c+=blockDim.x){ if(make_key64(s_sc[c],c)>=thr){unsigned p=atomicAdd(&s_fill,1u); if(p<(unsigned)k)orow[p]=c;} }
  __syncthreads();
  for(int j=(int)s_fill+tid;j<k;j+=blockDim.x) orow[j]=-1;
}
void launch_topk(torch::Tensor scores,torch::Tensor seq_lens,torch::Tensor out_idx,int k){
  int B=scores.size(0),C=scores.size(1);
  size_t smem=(size_t)C*4 + 2048*4;
  cudaStream_t st=at::cuda::getCurrentCUDAStream();
  auto kern=topk_v5b;
  if(smem>48*1024)cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
  kern<<<B,256,smem,st>>>(scores.data_ptr<float>(),seq_lens.data_ptr<int>(),out_idx.data_ptr<int>(),B,C,k);
}
'''

def build(variant):
    from torch.utils.cpp_extension import load_inline
    d=f"/tmp/torch_ext_pde5{variant}"; os.environ["TORCH_EXTENSIONS_DIR"]=d; os.makedirs(d,exist_ok=True)
    t0=time.time()
    src=SRC_A if variant=="A" else SRC_B
    m=load_inline(name=f"pde_topk_v5{variant}",cpp_sources="void launch_topk(torch::Tensor,torch::Tensor,torch::Tensor,int);",
        cuda_sources=src,functions=["launch_topk"],extra_cuda_cflags=["-O3","--use_fast_math","-arch=sm_100"],verbose=False)
    print(f"[build {variant}] {time.time()-t0:.1f}s",flush=True); return m
def setmatch(a,b):
    B=a.shape[0];fr=[]
    for r in range(B):
        sa=set(x for x in a[r].tolist() if x>=0); sb=set(x for x in b[r].tolist() if x>=0)
        fr.append(1.0 if not sb else len(sa&sb)/len(sb))
    return sum(fr)/len(fr)
def t_ms(fn,it=50,wu=15):
    for _ in range(wu): fn()
    torch.cuda.synchronize();s=torch.cuda.Event(True);e=torch.cuda.Event(True);s.record()
    for _ in range(it): fn()
    e.record();torch.cuda.synchronize();return s.elapsed_time(e)/it*1000.0
def main():
    dev="cuda";torch.manual_seed(0);m=build(VARIANT)
    op=torch.ops.trtllm.cute_dsl_indexer_topk_decode
    print(f"VARIANT={VARIANT}")
    print(f"{'case':>22} {'cute_us':>8} {'pde_us':>8} {'speedup':>8} {'setmatch':>8}",flush=True)
    for (C,k,label) in [(1032,64,"block nb1032 k64"),(8192,1024,"final C8192 k1024")]:
        for B in [1,8,32,64]:
            scores=torch.randn(B,C,device=dev,dtype=torch.float32)
            sl=torch.full((B,),C,device=dev,dtype=torch.int32)
            out_p=torch.full((B,k),-1,device=dev,dtype=torch.int32)
            out_c=torch.full((B,k),-1,device=dev,dtype=torch.int32)
            m.launch_topk(scores,sl,out_p,k); op(scores,sl,out_c,k); torch.cuda.synchronize()
            gold=torch.topk(scores.float(),k,dim=1).indices
            sm=setmatch(out_p,gold)
            pu=t_ms(lambda: m.launch_topk(scores,sl,out_p,k))
            cu=t_ms(lambda: op(scores,sl,out_c,k))
            print(f"{label:>16} B={B:>2} {cu:8.1f} {pu:8.1f} {cu/pu:7.2f}x {sm:8.4f}",flush=True)
if __name__=="__main__": main()

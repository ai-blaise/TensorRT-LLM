#!/usr/bin/env python3
# PDE radix-select top-k — ITER 6 (v8). Target = captured cute 8.2us(block)/
# 10.3us(final). v6 (single-CTA, warp-priv hist + float4 + compaction) is the best
# at 22-41us captured. Multi-CTA-coop (v7) was FATAL (grid.sync x25 = 55-111us).
# v8 attacks v6's PER-PASS FIXED OVERHEAD (single-CTA, no coop):
#   (a) PARALLEL threshold-bin walk: replace thread0's serial 256-step scan/pass
#       (8 passes => 2048 serial steps/row) with a block-parallel inclusive
#       suffix-sum over the 256 bins + first-crossing find. Removes the serial tail.
#   (b) keep warp-private hist + float4 + active-set compaction.
#   (c) cheaper histogram reset: only the 8*256 region, vectorized.
import os, time, math, sys
import torch
import tensorrt_llm

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
typedef unsigned long long ull;
__device__ __forceinline__ unsigned int f2o(float f){unsigned int u=__float_as_uint(f);unsigned int m=(unsigned int)(-(int)(u>>31))|0x80000000u;return u^m;}
__device__ __forceinline__ ull mk(float s,int i){return ((ull)f2o(s)<<32)|(ull)(~(unsigned int)i);}

// Parallel "find first bin (scanning from 255 down) where suffix-count >= krem".
// We have reduced hist in s_red[0..255]. Compute, for each bin b, the count of
// elements in bins (b..255] strictly above b: suffix_excl[b] = sum_{j>b} hist[j].
// The chosen digit = largest b with suffix_excl[b] < krem (i.e. including bin b
// reaches/exceeds krem). acc = suffix_excl[digit]. Done with a 256-elem block scan.
// 256 threads, one bin each. Use a simple shared-memory Hillis-Steele suffix sum.
__device__ __forceinline__ void walk_parallel(unsigned int* s_red, int krem,
                                              int tid, int* out_digit, int* out_acc,
                                              unsigned int* s_scan){
  // s_scan[b] = hist[b]; we want suffix_excl. Load.
  unsigned int v = (tid<256)? s_red[tid] : 0u;
  s_scan[tid]=v; __syncthreads();
  // inclusive suffix sum over 256 (Hillis-Steele, descending): after this,
  // s_scan[b] = sum_{j>=b} hist[j].
  #pragma unroll
  for(int off=1; off<256; off<<=1){
    unsigned int add = (tid+off<256)? s_scan[tid+off] : 0u;
    __syncthreads();
    s_scan[tid]+=add; __syncthreads();
  }
  // suffix_excl[b] = s_scan[b+1] (0 if b==255). digit = largest b with
  // suffix_excl[b] < krem.  Equivalent: smallest b with s_scan[b] >= krem? No:
  // we want the bin that, when included, first reaches krem scanning 255->0.
  // s_scan[b] = count in bins >= b. The digit is the LARGEST b with s_scan[b] >= krem.
  // Because scanning high->low, the first bin (largest b) whose inclusive suffix
  // reaches krem is where the krem-th element lands.
  // find largest b with s_scan[b] >= krem:
  int qualifies = (tid<256 && (int)s_scan[tid] >= krem) ? tid : -1;
  // block-max reduce of qualifies
  // warp reduce then shared
  for(int off=16; off>0; off>>=1){ int o=__shfl_down_sync(0xffffffffu,qualifies,off); qualifies = o>qualifies?o:qualifies; }
  __shared__ int s_wm[8];
  if((tid&31)==0) s_wm[tid>>5]=qualifies; __syncthreads();
  if(tid==0){ int mx=-1; for(int w=0;w<8;++w) if(s_wm[w]>mx) mx=s_wm[w]; if(mx<0)mx=0;
    *out_digit=mx; *out_acc=(int)((mx+1<256)? s_scan[mx+1] : 0u); }
}

__global__ void __launch_bounds__(256) topk_v8(const float* __restrict__ scores,
    const int* __restrict__ seq_lens, int* __restrict__ out_idx, int B,int C,int k){
  int row=blockIdx.x; if(row>=B) return; int tid=threadIdx.x; int warp=tid>>5;
  int sl=seq_lens[row];
  const float* srow=scores+(size_t)row*C; int* orow=out_idx+(size_t)row*k;
  extern __shared__ unsigned char smem[];
  float* s_sc=(float*)smem; unsigned int* s_hist=(unsigned int*)(s_sc+C); // 8*256
  unsigned int* s_red=s_hist; // reuse low 256 of s_hist after reduce
  unsigned int* s_scan=(unsigned int*)(s_hist+8*256); // 256
  int* s_act=(int*)(s_scan+256);
  __shared__ int s_digit,s_acc; __shared__ unsigned int s_fill; __shared__ int s_nact;
  int valid=sl<C?sl:C; int kk=k<valid?k:valid;

  int C4=C&~3; const float4* srow4=reinterpret_cast<const float4*>(srow);
  for(int c4=tid;c4<(C4>>2);c4+=blockDim.x){ float4 v=srow4[c4]; int base=c4<<2;
    s_sc[base+0]=(base+0<sl)?v.x:-FLT_MAX; s_sc[base+1]=(base+1<sl)?v.y:-FLT_MAX;
    s_sc[base+2]=(base+2<sl)?v.z:-FLT_MAX; s_sc[base+3]=(base+3<sl)?v.w:-FLT_MAX; }
  for(int c=C4+tid;c<C;c+=blockDim.x) s_sc[c]=(c<sl)?srow[c]:-FLT_MAX;
  __syncthreads();

  ull prefix=0,pmask=0; int krem=kk;
  // PASS 0: full-C hist + compact survivors
  {
    int sh=56;
    for(int b=tid;b<8*256;b+=blockDim.x) s_hist[b]=0u; __syncthreads();
    unsigned int* myh=s_hist+warp*256;
    for(int c=tid;c<valid;c+=blockDim.x){ ull key=mk(s_sc[c],c); atomicAdd(&myh[(unsigned)((key>>sh)&255)],1u); }
    __syncthreads();
    for(int b=tid;b<256;b+=blockDim.x){ unsigned int s=s_hist[b];
      #pragma unroll
      for(int w=1;w<8;++w) s+=s_hist[w*256+b]; s_red[b]=s; }
    __syncthreads();
    walk_parallel(s_red,krem,tid,&s_digit,&s_acc,s_scan);
    __syncthreads();
    if(tid==0) s_nact=0; __syncthreads();
    krem-=s_acc; prefix|=((ull)s_digit)<<sh; pmask|=((ull)255)<<sh; int dg=s_digit;
    for(int c=tid;c<valid;c+=blockDim.x){ ull key=mk(s_sc[c],c);
      if((unsigned)((key>>sh)&255)==(unsigned)dg){ int p=atomicAdd(&s_nact,1); s_act[p]=c; } }
    __syncthreads();
  }
  int nact=s_nact;
  for(int t=1;t<8;++t){ int sh=64-8*(t+1);
    for(int b=tid;b<8*256;b+=blockDim.x) s_hist[b]=0u; __syncthreads();
    unsigned int* myh=s_hist+warp*256;
    for(int i=tid;i<nact;i+=blockDim.x){ int c=s_act[i]; ull key=mk(s_sc[c],c);
      if((key&pmask)==prefix) atomicAdd(&myh[(unsigned)((key>>sh)&255)],1u); }
    __syncthreads();
    for(int b=tid;b<256;b+=blockDim.x){ unsigned int s=s_hist[b];
      #pragma unroll
      for(int w=1;w<8;++w) s+=s_hist[w*256+b]; s_red[b]=s; }
    __syncthreads();
    walk_parallel(s_red,krem,tid,&s_digit,&s_acc,s_scan);
    __syncthreads();
    krem-=s_acc; prefix|=((ull)s_digit)<<sh; pmask|=((ull)255)<<sh; __syncthreads();
  }
  ull thr=prefix;
  if(tid==0)s_fill=0u; __syncthreads();
  for(int c=tid;c<valid;c+=blockDim.x){ if(mk(s_sc[c],c)>=thr){unsigned p=atomicAdd(&s_fill,1u); if(p<(unsigned)k)orow[p]=c;} }
  __syncthreads();
  for(int j=(int)s_fill+tid;j<k;j+=blockDim.x) orow[j]=-1;
}
void launch_topk(torch::Tensor scores,torch::Tensor seq_lens,torch::Tensor out_idx,int k){
  int B=scores.size(0),C=scores.size(1);
  size_t smem=(size_t)C*4 + 8*256*4 + 256*4 + (size_t)C*4;
  cudaStream_t st=at::cuda::getCurrentCUDAStream();
  auto kern=topk_v8;
  if(smem>48*1024)cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
  kern<<<B,256,smem,st>>>(scores.data_ptr<float>(),seq_lens.data_ptr<int>(),out_idx.data_ptr<int>(),B,C,k);
}
'''
def build():
    from torch.utils.cpp_extension import load_inline
    d="/tmp/torch_ext_pde8"; os.environ["TORCH_EXTENSIONS_DIR"]=d; os.makedirs(d,exist_ok=True)
    t0=time.time()
    m=load_inline(name="pde_topk_v8",cpp_sources="void launch_topk(torch::Tensor,torch::Tensor,torch::Tensor,int);",
        cuda_sources=CUDA_SRC,functions=["launch_topk"],extra_cuda_cflags=["-O3","--use_fast_math","-arch=sm_100"],verbose=False)
    print(f"[build] {time.time()-t0:.1f}s",flush=True); return m
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
def cap_time(fn,it=80):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5): fn()
    torch.cuda.current_stream().wait_stream(s)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(15): g.replay()
    torch.cuda.synchronize();st=torch.cuda.Event(True);e=torch.cuda.Event(True);st.record()
    for _ in range(it): g.replay()
    e.record();torch.cuda.synchronize();return st.elapsed_time(e)/it*1000.0
def main():
    dev="cuda";torch.manual_seed(0);m=build()
    op=torch.ops.trtllm.cute_dsl_indexer_topk_decode
    CAP=("CAP" in sys.argv)
    print(f"{'case':>20} {'eC':>7} {'eV8':>7} {'capC':>8} {'capV8':>8} {'cV8/cC':>8} {'sm':>6}",flush=True)
    for (C,k,label) in [(1032,64,"block C1032 k64"),(8192,1024,"final C8192 k1024")]:
        for B in [1,8,32,64]:
            scores=torch.randn(B,C,device=dev,dtype=torch.float32)
            sl=torch.full((B,),C,device=dev,dtype=torch.int32)
            out_p=torch.full((B,k),-1,device=dev,dtype=torch.int32)
            out_c=torch.full((B,k),-1,device=dev,dtype=torch.int32)
            m.launch_topk(scores,sl,out_p,k); op(scores,sl,out_c,k); torch.cuda.synchronize()
            gold=torch.topk(scores.float(),k,dim=1).indices
            sm=setmatch(out_p,gold)
            eV=t_ms(lambda: m.launch_topk(scores,sl,out_p,k))
            eC=t_ms(lambda: op(scores,sl,out_c,k))
            capC=capV=float('nan')
            if CAP:
                try: capC=cap_time(lambda: op(scores,sl,out_c,k))
                except Exception as ex: print(f"  [capC FAIL B={B}] {str(ex)[:80]}",flush=True)
                try: capV=cap_time(lambda: m.launch_topk(scores,sl,out_p,k))
                except Exception as ex: print(f"  [capV FAIL B={B}] {str(ex)[:80]}",flush=True)
            r=(capV/capC) if (capC==capC and capV==capV) else float('nan')
            print(f"{label:>16} B={B:>2} {eC:7.1f} {eV:7.1f} {capC:8.1f} {capV:8.1f} {r:7.2f}x {sm:6.3f}",flush=True)
if __name__=="__main__": main()

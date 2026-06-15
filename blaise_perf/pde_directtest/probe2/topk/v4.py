#!/usr/bin/env python3
# PDE device-resident radix-select top-k — ITER 2 (v4).
# Changes vs v3 (single-CTA-per-row, 256-bin SMEM atomic, full re-scan/pass):
#   (1) WARP-PRIVATE histograms: 8 warps each own a 256-bin SMEM histogram, then
#       reduce 8->1 per pass. Cuts 256-way SMEM-atomic contention to 32-way.
#   (2) VECTORIZED float4 loads of scores into SMEM.
# Same composite-key 8-bit/8-pass MSD radix (exact -> setmatch 1.0).
import os, time, math
import torch
import tensorrt_llm  # registers cute_dsl ops

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
typedef unsigned long long ull;
__device__ __forceinline__ unsigned int float_to_okey(float f){unsigned int u=__float_as_uint(f);unsigned int m=(unsigned int)(-(int)(u>>31))|0x80000000u;return u^m;}
__device__ __forceinline__ ull make_key64(float s,int idx){return ((ull)float_to_okey(s)<<32)|(ull)(~(unsigned int)idx);}

// v4: 256 threads = 8 warps; each warp has a private 256-bin histogram.
// SMEM layout: [ s_sc: C floats ][ s_hist: 8*256 uints ]
__global__ void __launch_bounds__(256) topk_v4(const float* __restrict__ scores,
    const int* __restrict__ seq_lens, int* __restrict__ out_idx, int B,int C,int k){
  int row=blockIdx.x; if(row>=B) return; int tid=threadIdx.x; int warp=tid>>5; int lane=tid&31;
  int sl=seq_lens[row];
  const float* srow=scores+(size_t)row*C; int* orow=out_idx+(size_t)row*k;
  extern __shared__ unsigned char smem[];
  float* s_sc=(float*)smem; unsigned int* s_hist=(unsigned int*)(s_sc+C); // 8*256 bins
  __shared__ int s_digit,s_acc; __shared__ unsigned int s_fill;
  int valid=sl<C?sl:C; int kk=k<valid?k:valid;

  // (2) vectorized float4 load of scores -> SMEM (C assumed multiple of 4; both prod shapes are even, pad tail scalar)
  int C4=C&~3;
  const float4* srow4=reinterpret_cast<const float4*>(srow);
  for(int c4=tid;c4<(C4>>2);c4+=blockDim.x){ float4 v=srow4[c4]; int base=c4<<2;
    s_sc[base+0]=(base+0<sl)?v.x:-FLT_MAX; s_sc[base+1]=(base+1<sl)?v.y:-FLT_MAX;
    s_sc[base+2]=(base+2<sl)?v.z:-FLT_MAX; s_sc[base+3]=(base+3<sl)?v.w:-FLT_MAX; }
  for(int c=C4+tid;c<C;c+=blockDim.x) s_sc[c]=(c<sl)?srow[c]:-FLT_MAX;
  __syncthreads();

  ull prefix=0,pmask=0; int krem=kk;
  for(int t=0;t<8;++t){ int sh=64-8*(t+1);
    // reset all 8 private histograms
    for(int b=tid;b<8*256;b+=blockDim.x) s_hist[b]=0u; __syncthreads();
    unsigned int* myh=s_hist+warp*256;
    // (1) each thread atomicAdds into ITS OWN WARP's histogram -> 32-way contention
    for(int c=tid;c<valid;c+=blockDim.x){ ull key=make_key64(s_sc[c],c);
      if((key&pmask)==prefix) atomicAdd(&myh[(unsigned)((key>>sh)&255)],1u); }
    __syncthreads();
    // reduce 8 private histograms -> warp0's histogram (bin-parallel over 256 bins)
    for(int b=tid;b<256;b+=blockDim.x){ unsigned int s=s_hist[b];
      #pragma unroll
      for(int w=1;w<8;++w) s+=s_hist[w*256+b];
      s_hist[b]=s; }
    __syncthreads();
    if(tid==0){int acc=0,dg=0;for(int bn=255;bn>=0;--bn){int cc=(int)s_hist[bn];if(acc+cc>=krem){dg=bn;break;}acc+=cc;}s_digit=dg;s_acc=acc;}
    __syncthreads();
    krem-=s_acc; prefix|=((ull)s_digit)<<sh; pmask|=((ull)255)<<sh; __syncthreads();
  }
  ull thr=prefix;
  if(tid==0)s_fill=0u; __syncthreads();
  for(int c=tid;c<valid;c+=blockDim.x){ if(make_key64(s_sc[c],c)>=thr){unsigned p=atomicAdd(&s_fill,1u); if(p<(unsigned)k)orow[p]=c;} }
  __syncthreads();
  for(int j=(int)s_fill+tid;j<k;j+=blockDim.x) orow[j]=-1;
}
void launch_topk(torch::Tensor scores,torch::Tensor seq_lens,torch::Tensor out_idx,int k){
  int B=scores.size(0),C=scores.size(1);
  size_t smem=(size_t)C*4+8*256*4;
  cudaStream_t st=at::cuda::getCurrentCUDAStream();
  auto kern=topk_v4;
  if(smem>48*1024)cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
  kern<<<B,256,smem,st>>>(scores.data_ptr<float>(),seq_lens.data_ptr<int>(),out_idx.data_ptr<int>(),B,C,k);
}
'''
def build():
    from torch.utils.cpp_extension import load_inline
    os.environ.setdefault("TORCH_EXTENSIONS_DIR","/tmp/torch_ext_pde4"); os.makedirs("/tmp/torch_ext_pde4",exist_ok=True)
    t0=time.time()
    m=load_inline(name="pde_topk_v4",cpp_sources="void launch_topk(torch::Tensor,torch::Tensor,torch::Tensor,int);",
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
def main():
    dev="cuda";torch.manual_seed(0);m=build()
    op=torch.ops.trtllm.cute_dsl_indexer_topk_decode
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

#!/usr/bin/env python3
# PDE single-level device-resident radix-select top-k (the prod-integratable
# primitive: block-topk and final-topk are SEPARATE ops, split by the candidate
# GEMM). Compares vs cute_dsl_indexer_topk_decode at the two real prod shapes.
import os, time, math
import torch
import tensorrt_llm  # registers cute_dsl ops

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cfloat>
typedef unsigned long long ull;
__device__ __forceinline__ unsigned int float_to_okey(float f){unsigned int u=__float_as_uint(f);unsigned int m=(unsigned int)(-(int)(u>>31))|0x80000000u;return u^m;}
__device__ __forceinline__ ull make_key64(float s,int idx){return ((ull)float_to_okey(s)<<32)|(ull)(~(unsigned int)idx);}
__device__ __forceinline__ float to_float(float x){return x;}
__device__ __forceinline__ float to_float(__nv_bfloat16 x){return __bfloat162float(x);}

template<typename ScalarT>
__global__ void __launch_bounds__(256) topk_one(const ScalarT* __restrict__ scores,
    const int* __restrict__ seq_lens, int* __restrict__ out_idx, int B,int C,int k){
  int row=blockIdx.x; if(row>=B) return; int tid=threadIdx.x; int sl=seq_lens[row];
  const ScalarT* srow=scores+(size_t)row*C; int* orow=out_idx+(size_t)row*k;
  extern __shared__ unsigned char smem[];
  float* s_sc=(float*)smem; unsigned int* s_hist=(unsigned int*)(s_sc+C);
  __shared__ int s_digit,s_acc; __shared__ unsigned int s_fill;
  for(int c=tid;c<C;c+=blockDim.x) s_sc[c]=(c<sl)?to_float(srow[c]):-FLT_MAX;
  __syncthreads();
  int valid=sl<C?sl:C; int kk=k<valid?k:valid;
  ull prefix=0,pmask=0; int krem=kk;
  for(int t=0;t<8;++t){ int sh=64-8*(t+1);
    for(int b=tid;b<256;b+=blockDim.x) s_hist[b]=0u; __syncthreads();
    for(int c=tid;c<valid;c+=blockDim.x){ ull key=make_key64(s_sc[c],c);
      if((key&pmask)==prefix) atomicAdd(&s_hist[(unsigned)((key>>sh)&255)],1u); }
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
  size_t smem=(size_t)C*4+256*4;
  cudaStream_t st=at::cuda::getCurrentCUDAStream();
  if(scores.scalar_type()==torch::kFloat32){auto kern=topk_one<float>;
    if(smem>48*1024)cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
    kern<<<B,256,smem,st>>>(scores.data_ptr<float>(),seq_lens.data_ptr<int>(),out_idx.data_ptr<int>(),B,C,k);
  }else{auto kern=topk_one<__nv_bfloat16>;
    if(smem>48*1024)cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
    kern<<<B,256,smem,st>>>((const __nv_bfloat16*)scores.data_ptr<at::BFloat16>(),seq_lens.data_ptr<int>(),out_idx.data_ptr<int>(),B,C,k);}
}
'''
def build():
    from torch.utils.cpp_extension import load_inline
    os.environ.setdefault("TORCH_EXTENSIONS_DIR","/tmp/torch_ext_pde3"); os.makedirs("/tmp/torch_ext_pde3",exist_ok=True)
    t0=time.time()
    m=load_inline(name="pde_topk_v3",cpp_sources="void launch_topk(torch::Tensor,torch::Tensor,torch::Tensor,int);",
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

#!/usr/bin/env python3
# Does the device radix-select EVER beat cute_dsl_indexer_topk_decode, captured,
# at a prod-plausible regime? Probe (a) long live-kv final-topk (the dsa.py DSL
# crossover note says cute/C++ cross ~12-16K), and (b) a MULTI-CTA-per-row radix
# (split candidates across G CTAs via a global per-row histogram + grid.sync,
# i.e. the G3-OPT shape) to remove the single-CTA serialization that loses at
# C8192. Captured-vs-captured only (both are capturable here).
import os, time
import torch
import tensorrt_llm

# Multi-CTA-per-row cooperative radix-select (G3-OPT shape, condensed). gridDim =
# G*B; group of G CTAs cooperates on one row via a global [B,256] histogram.
CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cfloat>
namespace cg=cooperative_groups;
typedef unsigned long long ull;
__device__ __forceinline__ unsigned int f2o(float f){unsigned int u=__float_as_uint(f);unsigned int m=(unsigned int)(-(int)(u>>31))|0x80000000u;return u^m;}
__device__ __forceinline__ ull mk(float s,int i){return ((ull)f2o(s)<<32)|(ull)(~(unsigned int)i);}

// global scratch: hist[B*256], outcnt[B]
__global__ void __launch_bounds__(256) topk_mcta(const float* __restrict__ sc,const int* __restrict__ sl,
    int* __restrict__ out,unsigned int* __restrict__ ghist,unsigned int* __restrict__ outcnt,int B,int C,int k,int G){
  cg::grid_group grid=cg::this_grid();
  int row=blockIdx.x/G, rank=blockIdx.x%G, tid=threadIdx.x;
  extern __shared__ unsigned int sh[]; // 256
  int L=sl[row]; int v=L<C?L:C; int kk=k<v?k:v;
  const float* s=sc+(size_t)row*C; int* o=out+(size_t)row*k;
  unsigned int* gh=ghist+(size_t)row*256;
  ull pf=0,pm=0; int kr=kk;
  if(rank==0&&tid==0) outcnt[row]=0u;
  for(int t=0;t<8;++t){int shift=64-8*(t+1);
    for(int b=tid;b<256;b+=blockDim.x) sh[b]=0u; __syncthreads();
    for(int c=rank*blockDim.x+tid;c<v;c+=G*blockDim.x){ull key=mk(s[c],c);if((key&pm)==pf)atomicAdd(&sh[(unsigned)((key>>shift)&255)],1u);}
    __syncthreads();
    for(int b=tid;b<256;b+=blockDim.x){unsigned val=sh[b];if(val)atomicAdd(&gh[b],val);}
    grid.sync();
    for(int b=tid;b<256;b+=blockDim.x) sh[b]=gh[b]; __syncthreads();
    int acc=0,dg=0;for(int b=255;b>=0;--b){int cc=(int)sh[b];if(acc+cc>=kr){dg=b;break;}acc+=cc;}
    kr-=acc; pf|=((ull)dg)<<shift; pm|=((ull)255)<<shift;
    grid.sync();
    if(rank==0){for(int b=tid;b<256;b+=blockDim.x) gh[b]=0u;}
    grid.sync();
  }
  ull thr=pf;
  for(int c=rank*blockDim.x+tid;c<v;c+=G*blockDim.x){if(mk(s[c],c)>=thr){unsigned p=atomicAdd(&outcnt[row],1u);if(p<(unsigned)k)o[p]=c;}}
  grid.sync();
  for(int j=(int)outcnt[row]+rank*blockDim.x+tid;j<k;j+=G*blockDim.x) o[j]=-1;
}

void launch_mcta(torch::Tensor sc,torch::Tensor sl,torch::Tensor out,torch::Tensor ghist,torch::Tensor outcnt,int k,int G){
  int B=sc.size(0),C=sc.size(1); size_t sm=256*4;
  void* args[]={(void*)&(*(float**)sc.data_ptr()),0};
  dim3 grid(B*G),block(256);
  float* scp=sc.data_ptr<float>(); int* slp=sl.data_ptr<int>(); int* op=out.data_ptr<int>();
  unsigned int* ghp=(unsigned int*)ghist.data_ptr<int>(); unsigned int* ocp=(unsigned int*)outcnt.data_ptr<int>();
  void* a[]={&scp,&slp,&op,&ghp,&ocp,&B,&C,&k,&G};
  cudaLaunchCooperativeKernel((void*)topk_mcta,grid,block,a,sm,at::cuda::getCurrentCUDAStream());
}
'''

def build():
    from torch.utils.cpp_extension import load_inline
    os.environ.setdefault("TORCH_EXTENSIONS_DIR","/tmp/torch_ext_longc"); os.makedirs("/tmp/torch_ext_longc",exist_ok=True)
    return load_inline(name="longc_topk",cpp_sources="void launch_mcta(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int,int);",
        cuda_sources=CUDA_SRC,functions=["launch_mcta"],extra_cuda_cflags=["-O3","--use_fast_math","-arch=sm_100","--expt-relaxed-constexpr"],verbose=True)

def t_ms(fn,it=80,wu=20):
    for _ in range(wu): fn()
    torch.cuda.synchronize();s=torch.cuda.Event(True);e=torch.cuda.Event(True);s.record()
    for _ in range(it): fn()
    e.record();torch.cuda.synchronize();return s.elapsed_time(e)/it*1000.0

def setmatch(a,gold):
    B=a.shape[0];tot=0.0
    for r in range(B):
        sa=set(x for x in a[r].tolist() if x>=0); sg=set(gold[r].tolist())
        tot+=len(sa&sg)/len(sg)
    return tot/B

def main():
    dev="cuda";torch.manual_seed(0);m=build()
    cute=torch.ops.trtllm.cute_dsl_indexer_topk_decode
    # long-C final-topk regime, multi-CTA radix vs captured cute
    print(f"\n{'case':>22} {'G':>3} {'cap_cute_us':>11} {'cap_mcta_us':>11} {'mcta/cute':>10} {'setmatch':>9}",flush=True)
    for (C,k) in [(8192,1024),(32768,1024),(131072,1024)]:
        for B in [1,8,32]:
            for G in ([8,32] if C>=32768 else [4,8]):
                if B*G>132: continue
                sc=torch.randn(B,C,device=dev,dtype=torch.float32)
                sl=torch.full((B,),C,device=dev,dtype=torch.int32)
                oV=torch.full((B,k),-1,device=dev,dtype=torch.int32)
                oC=torch.full((B,k),-1,device=dev,dtype=torch.int32)
                gh=torch.zeros(B,256,device=dev,dtype=torch.int32)
                oc=torch.zeros(B,device=dev,dtype=torch.int32)
                f_v=lambda: m.launch_mcta(sc,sl,oV,gh,oc,k,G)
                f_c=lambda: cute(sc,sl,oC,k)
                f_v(); torch.cuda.synchronize()
                gold=torch.topk(sc,k,dim=1).indices
                sm=setmatch(oV,gold)
                # capture both
                def cap(fn):
                    st=torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(st):
                        for _ in range(3): fn()
                    torch.cuda.current_stream().wait_stream(st)
                    g=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g): fn()
                    return g
                try: gC=cap(f_c); cu=t_ms(lambda:gC.replay())
                except Exception as ex: cu=float('nan')
                try: gV=cap(f_v); vu=t_ms(lambda:gV.replay())
                except Exception as ex: vu=float('nan'); print("MCTA CAPFAIL",str(ex)[:50])
                print(f"final C{C} k{k} B={B:>2} {G:>3} {cu:11.1f} {vu:11.1f} {vu/cu:9.2f}x {sm:9.4f}",flush=True)
if __name__=="__main__": main()

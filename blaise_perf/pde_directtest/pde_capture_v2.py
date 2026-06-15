#!/usr/bin/env python3
# Apples-to-apples capture-overhead delta for the PDE device-resident radix top-k
# vs the in-image production decode top-k (cute_dsl_indexer_topk_decode).
#
# Three regimes per (shape, batch):
#   1. eager-cute      : current prod pattern, op called eagerly
#   2. captured-cute   : cute op inside a CUDA graph (if capturable)
#   3. captured-v3     : device-resident radix-select inside a CUDA graph
# We also probe whether cute is capturable on its own (the load-bearing claim:
# the device-resident path's value is removing a capture-illegal d2h). We measure
# the topk op IN ISOLATION (no synthetic layer) so the delta is attributable.
import os, time
import torch
import tensorrt_llm  # cute_dsl ops

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
typedef unsigned long long ull;
__device__ __forceinline__ unsigned int f2o(float f){unsigned int u=__float_as_uint(f);unsigned int m=(unsigned int)(-(int)(u>>31))|0x80000000u;return u^m;}
__device__ __forceinline__ ull mk(float s,int i){return ((ull)f2o(s)<<32)|(ull)(~(unsigned int)i);}
__global__ void __launch_bounds__(256) topk_one(const float* __restrict__ sc,const int* __restrict__ sl,int* __restrict__ out,int B,int C,int k){
  int row=blockIdx.x; if(row>=B)return; int tid=threadIdx.x; int L=sl[row];
  const float* s=sc+(size_t)row*C; int* o=out+(size_t)row*k;
  extern __shared__ unsigned char sm[]; float* ss=(float*)sm; unsigned int* h=(unsigned int*)(ss+C);
  __shared__ int sd,sa; __shared__ unsigned int sf;
  for(int c=tid;c<C;c+=blockDim.x) ss[c]=(c<L)?s[c]:-FLT_MAX; __syncthreads();
  int v=L<C?L:C; int kk=k<v?k:v; ull pf=0,pm=0; int kr=kk;
  for(int t=0;t<8;++t){int sh=64-8*(t+1);
    for(int b=tid;b<256;b+=blockDim.x)h[b]=0u;__syncthreads();
    for(int c=tid;c<v;c+=blockDim.x){ull key=mk(ss[c],c);if((key&pm)==pf)atomicAdd(&h[(unsigned)((key>>sh)&255)],1u);}__syncthreads();
    if(tid==0){int a=0,d=0;for(int b=255;b>=0;--b){int cc=(int)h[b];if(a+cc>=kr){d=b;break;}a+=cc;}sd=d;sa=a;}__syncthreads();
    kr-=sa;pf|=((ull)sd)<<sh;pm|=((ull)255)<<sh;__syncthreads();}
  ull thr=pf; if(tid==0)sf=0u;__syncthreads();
  for(int c=tid;c<v;c+=blockDim.x){if(mk(ss[c],c)>=thr){unsigned p=atomicAdd(&sf,1u);if(p<(unsigned)k)o[p]=c;}}__syncthreads();
  for(int j=(int)sf+tid;j<k;j+=blockDim.x)o[j]=-1;
}
void launch_topk(torch::Tensor sc,torch::Tensor sl,torch::Tensor out,int k){
  int B=sc.size(0),C=sc.size(1); size_t sm=(size_t)C*4+256*4;
  auto kern=topk_one; if(sm>48*1024)cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm);
  kern<<<B,256,sm,at::cuda::getCurrentCUDAStream()>>>(sc.data_ptr<float>(),sl.data_ptr<int>(),out.data_ptr<int>(),B,C,k);
}
'''

def build():
    from torch.utils.cpp_extension import load_inline
    os.environ.setdefault("TORCH_EXTENSIONS_DIR","/tmp/torch_ext_cap2"); os.makedirs("/tmp/torch_ext_cap2",exist_ok=True)
    return load_inline(name="cap2_topk",cpp_sources="void launch_topk(torch::Tensor,torch::Tensor,torch::Tensor,int);",
        cuda_sources=CUDA_SRC,functions=["launch_topk"],extra_cuda_cflags=["-O3","--use_fast_math","-arch=sm_100"],verbose=False)

def t_ms(fn,it=80,wu=20):
    for _ in range(wu): fn()
    torch.cuda.synchronize();s=torch.cuda.Event(True);e=torch.cuda.Event(True);s.record()
    for _ in range(it): fn()
    e.record();torch.cuda.synchronize();return s.elapsed_time(e)/it*1000.0

def try_capture(fn):
    """Return (graph or None, err). Warms on a side stream then captures."""
    try:
        s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): fn()
        torch.cuda.current_stream().wait_stream(s)
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): fn()
        return g,None
    except Exception as ex:
        return None,str(ex)[:60]

def main():
    dev="cuda";torch.manual_seed(0);m=build()
    cute=torch.ops.trtllm.cute_dsl_indexer_topk_decode
    print("ISOLATED top-k op: eager vs captured, cute vs device-radix(v3)\n")
    print(f"{'case':>20} {'eag_cute':>9} {'cap_cute':>9} {'eag_v3':>9} {'cap_v3':>9} {'capv3/capcute':>14} {'capv3/eagcute':>14} {'cute_cap?':>20}",flush=True)
    for (C,k,label) in [(1032,64,"block C1032 k64"),(8192,1024,"final C8192 k1024")]:
        for B in [1,8,32,64]:
            sc=torch.randn(B,C,device=dev,dtype=torch.float32)
            sl=torch.full((B,),C,device=dev,dtype=torch.int32)
            oC=torch.full((B,k),-1,device=dev,dtype=torch.int32)
            oV=torch.full((B,k),-1,device=dev,dtype=torch.int32)
            f_cute=lambda: cute(sc,sl,oC,k)
            f_v3  =lambda: m.launch_topk(sc,sl,oV,k)
            f_cute(); f_v3(); torch.cuda.synchronize()
            eag_cute=t_ms(f_cute); eag_v3=t_ms(f_v3)
            gC,errC=try_capture(f_cute); gV,errV=try_capture(f_v3)
            cap_cute=t_ms(lambda: gC.replay()) if gC else float('nan')
            cap_v3  =t_ms(lambda: gV.replay()) if gV else float('nan')
            cute_cap = "yes" if gC else f"NO:{errC}"
            r_vc = (cap_v3/cap_cute) if gC and gV else float('nan')
            r_ve = (cap_v3/eag_cute) if gV else float('nan')
            print(f"{label:>14} B={B:>2} {eag_cute:9.1f} {cap_cute:9.1f} {eag_v3:9.1f} {cap_v3:9.1f} {r_vc:13.2f}x {r_ve:13.2f}x {cute_cap:>20}",flush=True)
if __name__=="__main__": main()

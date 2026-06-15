#!/usr/bin/env python3
# Two go/no-go probes:
#  (1) Captured cute_dsl CORRECTNESS: capture cute topk, replay with NEW inputs,
#      confirm output set-matches torch.topk (i.e. capture isn't a stale no-op).
#  (2) Cooperative-kernel CAPTURE feasibility: does cudaLaunchCooperativeKernel
#      with grid.sync() run, and can it be captured in a CUDA graph on sm_100?
import os, time, sys
import torch
import tensorrt_llm

COOP_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cfloat>
namespace cg=cooperative_groups;
// trivial cooperative kernel: each block adds blockIdx to a global accumulator,
// grid.sync(), then block0 reads sum. Tests coop-launch + grid.sync + capture.
__global__ void coop_test(int* acc, int* out, int nblocks){
  cg::grid_group grid=cg::this_grid();
  if(threadIdx.x==0) atomicAdd(acc, blockIdx.x);
  grid.sync();
  if(blockIdx.x==0 && threadIdx.x==0) *out = *acc;
}
int launch_coop(torch::Tensor acc, torch::Tensor out, int nblocks){
  cudaStream_t st=at::cuda::getCurrentCUDAStream();
  void* k=(void*)coop_test;
  // need acc zeroed each call -> do it via cudaMemsetAsync (capturable)
  cudaMemsetAsync(acc.data_ptr<int>(),0,sizeof(int),st);
  int nb=nblocks; int* pacc=acc.data_ptr<int>(); int* pout=out.data_ptr<int>();
  void* args[]={&pacc,&pout,&nb};
  cudaError_t e=cudaLaunchCooperativeKernel(k, dim3(nblocks),dim3(32),args,0,st);
  return (int)e;
}
int max_coop_blocks(){
  int n=0; cudaError_t e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,(void*)coop_test,32,0);
  int dev=0; cudaGetDevice(&dev); int sms=0; cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev);
  return n*sms;
}
'''
def build():
    from torch.utils.cpp_extension import load_inline
    d="/tmp/torch_ext_coop"; os.environ["TORCH_EXTENSIONS_DIR"]=d; os.makedirs(d,exist_ok=True)
    return load_inline(name="coop_probe",cpp_sources="int launch_coop(torch::Tensor,torch::Tensor,int); int max_coop_blocks();",
        cuda_sources=COOP_SRC,functions=["launch_coop","max_coop_blocks"],extra_cuda_cflags=["-O3","-arch=sm_100"],verbose=False)

def setmatch(a,b):
    B=a.shape[0];fr=[]
    for r in range(B):
        sa=set(x for x in a[r].tolist() if x>=0); sb=set(x for x in b[r].tolist() if x>=0)
        fr.append(1.0 if not sb else len(sa&sb)/len(sb))
    return sum(fr)/len(fr)

def main():
    dev="cuda";torch.manual_seed(0);m=build()
    # ---- probe 1: captured cute correctness with changing inputs ----
    cute=torch.ops.trtllm.cute_dsl_indexer_topk_decode
    C,k,B=8192,1024,8
    sc=torch.randn(B,C,device=dev,dtype=torch.float32); sl=torch.full((B,),C,device=dev,dtype=torch.int32)
    oc=torch.full((B,k),-1,device=dev,dtype=torch.int32)
    cute(sc,sl,oc,k); torch.cuda.synchronize()
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): cute(sc,sl,oc,k)
    torch.cuda.current_stream().wait_stream(s)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): cute(sc,sl,oc,k)
    # now CHANGE the input in-place and replay -> output must track new input
    sc.copy_(torch.randn(B,C,device=dev,dtype=torch.float32))
    g.replay(); torch.cuda.synchronize()
    gold=torch.topk(sc.float(),k,dim=1).indices
    sm=setmatch(oc,gold)
    print(f"[probe1] captured-cute correctness after input change: setmatch={sm:.4f}  (1.0 => capture is real, not stale)",flush=True)

    # ---- probe 2: cooperative launch + grid.sync, eager then captured ----
    mb=m.max_coop_blocks()
    print(f"[probe2] max cooperative blocks (32 thr): {mb}",flush=True)
    acc=torch.zeros(1,device=dev,dtype=torch.int32); out=torch.zeros(1,device=dev,dtype=torch.int32)
    NB=min(64,mb)
    e=m.launch_coop(acc,out,NB); torch.cuda.synchronize()
    exp=NB*(NB-1)//2
    print(f"[probe2] eager coop launch err={e} out={int(out.item())} expected={exp} ok={int(out.item())==exp}",flush=True)
    # capture it
    try:
        s2=torch.cuda.Stream(); s2.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s2):
            for _ in range(3): m.launch_coop(acc,out,NB)
        torch.cuda.current_stream().wait_stream(s2)
        g2=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2): m.launch_coop(acc,out,NB)
        out.zero_(); g2.replay(); torch.cuda.synchronize()
        print(f"[probe2] CAPTURED coop replay out={int(out.item())} expected={exp} ok={int(out.item())==exp}  => coop kernels ARE capturable",flush=True)
    except Exception as ex:
        print(f"[probe2] CAPTURED coop FAILED: {type(ex).__name__}: {str(ex)[:120]}  => coop NOT capturable, must use 2-phase",flush=True)
if __name__=="__main__": main()

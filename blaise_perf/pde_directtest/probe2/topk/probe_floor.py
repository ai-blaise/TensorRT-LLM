#!/usr/bin/env python3
# Establish the COMPUTE FLOOR: a kernel that only (a) loads C scores to SMEM via
# float4 and (b) writes k dummy outputs -- NO radix. If captured this is already
# ~10us, the radix is nearly free and we're load/occupancy-bound. Also probe SMEM
# occupancy + report which tools exist. Compares vs cute captured.
import os, time, sys, shutil
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
// LOAD-ONLY floor: float4-load C scores -> s_key (okey), then write first k indices.
__global__ void __launch_bounds__(256) loadonly(const float* __restrict__ scores,
    const int* __restrict__ seq_lens, int* __restrict__ out_idx, int B,int C,int k){
  int row=blockIdx.x; if(row>=B) return; int tid=threadIdx.x; int sl=seq_lens[row];
  const float* srow=scores+(size_t)row*C; int* orow=out_idx+(size_t)row*k;
  extern __shared__ unsigned char smem[]; unsigned int* s_key=(unsigned int*)smem;
  int C4=C&~3; const float4* srow4=reinterpret_cast<const float4*>(srow);
  for(int c4=tid;c4<(C4>>2);c4+=blockDim.x){ float4 v=srow4[c4]; int base=c4<<2;
    s_key[base+0]=f2o(v.x); s_key[base+1]=f2o(v.y); s_key[base+2]=f2o(v.z); s_key[base+3]=f2o(v.w); }
  for(int c=C4+tid;c<C;c+=blockDim.x) s_key[c]=f2o(srow[c]);
  __syncthreads();
  // trivial "use" of s_key so it's not optimized away: write argmax-ish (just index)
  unsigned int acc=0; for(int c=tid;c<C;c+=blockDim.x) acc+=s_key[c];
  __shared__ unsigned int sink; if(tid==0) sink=0; __syncthreads();
  atomicAdd(&sink,acc); __syncthreads();
  for(int j=tid;j<k;j+=blockDim.x) orow[j]=(int)((j+sink)%C);
}
void launch_loadonly(torch::Tensor scores,torch::Tensor seq_lens,torch::Tensor out_idx,int k){
  int B=scores.size(0),C=scores.size(1); size_t smem=(size_t)C*4;
  auto kern=loadonly; if(smem>48*1024)cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
  kern<<<B,256,smem,at::cuda::getCurrentCUDAStream()>>>(scores.data_ptr<float>(),seq_lens.data_ptr<int>(),out_idx.data_ptr<int>(),B,C,k);
}
int occ(int C){ size_t smem=(size_t)C*4; auto kern=loadonly;
  if(smem>48*1024)cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
  int nb=0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb,(void*)kern,256,smem); return nb; }
'''
def build():
    from torch.utils.cpp_extension import load_inline
    d="/tmp/torch_ext_floor"; os.environ["TORCH_EXTENSIONS_DIR"]=d; os.makedirs(d,exist_ok=True)
    return load_inline(name="floork",cpp_sources="void launch_loadonly(torch::Tensor,torch::Tensor,torch::Tensor,int); int occ(int);",
        cuda_sources=CUDA_SRC,functions=["launch_loadonly","occ"],extra_cuda_cflags=["-O3","--use_fast_math","-arch=sm_100"],verbose=False)
def cap_time(fn,it=100):
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
    print("tools:", "ncu",shutil.which("ncu"),"nsys",shutil.which("nsys"),flush=True)
    dev="cuda";torch.manual_seed(0);m=build()
    op=torch.ops.trtllm.cute_dsl_indexer_topk_decode
    print(f"{'case':>20} {'capLoadOnly':>12} {'capCute':>10} {'occ_blk/SM':>11}",flush=True)
    for (C,k,label) in [(1032,64,"block C1032 k64"),(8192,1024,"final C8192 k1024")]:
        for B in [1,8,32,64]:
            scores=torch.randn(B,C,device=dev,dtype=torch.float32)
            sl=torch.full((B,),C,device=dev,dtype=torch.int32)
            out_p=torch.full((B,k),-1,device=dev,dtype=torch.int32)
            out_c=torch.full((B,k),-1,device=dev,dtype=torch.int32)
            m.launch_loadonly(scores,sl,out_p,k); op(scores,sl,out_c,k); torch.cuda.synchronize()
            o=m.occ(C)
            cl=cap_time(lambda: m.launch_loadonly(scores,sl,out_p,k))
            cc=cap_time(lambda: op(scores,sl,out_c,k))
            print(f"{label:>16} B={B:>2} {cl:12.1f} {cc:10.1f} {o:11d}",flush=True)
if __name__=="__main__": main()

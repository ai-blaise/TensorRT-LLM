#!/usr/bin/env python3
# Capture lever direct test: a prod-shaped decode-layer run EAGER (cute_dsl topk,
# which does an internal d2h sync -> un-capturable) vs CUDA-graph CAPTURED (v3
# device-resident radix topk -> no d2h -> capturable). The point: a capturable
# topk lets the WHOLE layer be captured, removing per-op launch overhead across
# all ops -- even though the device topk is slightly slower per-op.
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
    os.environ.setdefault("TORCH_EXTENSIONS_DIR","/tmp/torch_ext_cap"); os.makedirs("/tmp/torch_ext_cap",exist_ok=True)
    return load_inline(name="cap_topk",cpp_sources="void launch_topk(torch::Tensor,torch::Tensor,torch::Tensor,int);",
        cuda_sources=CUDA_SRC,functions=["launch_topk"],extra_cuda_cflags=["-O3","--use_fast_math","-arch=sm_100"],verbose=False)

def t_ms(fn,it=50,wu=15):
    for _ in range(wu): fn()
    torch.cuda.synchronize();s=torch.cuda.Event(True);e=torch.cuda.Event(True);s.record()
    for _ in range(it): fn()
    e.record();torch.cuda.synchronize();return s.elapsed_time(e)/it*1000.0

def main():
    dev="cuda";torch.manual_seed(0);m=build()
    H=7168; nb=1032; cand=8192; KVD=512
    cute=torch.ops.trtllm.cute_dsl_indexer_topk_decode
    print(f"{'B':>3} {'eager_cute_us':>14} {'eager_v3_us':>12} {'captured_v3_us':>14} {'cap_vs_eagercute':>16}",flush=True)
    for B in [1,8,32,64]:
        # static buffers (prod-ish decode layer)
        hidden=torch.randn(B,H,device=dev,dtype=torch.bfloat16)
        w1=torch.randn(H,1536,device=dev,dtype=torch.bfloat16); w2=torch.randn(1536,H,device=dev,dtype=torch.bfloat16)
        sB=torch.randn(B,nb,device=dev,dtype=torch.float32); slB=torch.full((B,),nb,device=dev,dtype=torch.int32)
        sF=torch.randn(B,cand,device=dev,dtype=torch.float32); slF=torch.full((B,),cand,device=dev,dtype=torch.int32)
        kvp=torch.randn(cand,KVD,device=dev,dtype=torch.bfloat16)
        oB=torch.full((B,64),-1,device=dev,dtype=torch.int32); oF=torch.full((B,1024),-1,device=dev,dtype=torch.int32)
        out=torch.zeros(B,H,device=dev,dtype=torch.bfloat16)
        def layer(topk_kind):
            x=hidden@w1; x=x@w2                       # proj GEMMs
            if topk_kind=="cute":
                cute(sB,slB,oB,64); cute(sF,slF,oF,1024)
            else:
                m.launch_topk(sB,slB,oB,64); m.launch_topk(sF,slF,oF,1024)
            g=kvp[oF.clamp_min(0).long()]             # sparse gather [B,1024,KVD]
            red=g.float().mean(1).to(torch.bfloat16)  # weighted-sum stand-in [B,KVD]
            out.copy_(x + torch.nn.functional.pad(red,(0,H-KVD)))
            return out
        # correctness sanity
        layer("cute"); layer("v3"); torch.cuda.synchronize()
        eu_cute=t_ms(lambda: layer("cute"))
        eu_v3=t_ms(lambda: layer("v3"))
        # capture with v3 (capturable: no d2h)
        s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): layer("v3")
        torch.cuda.current_stream().wait_stream(s)
        gph=torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(gph): layer("v3")
            for _ in range(15): gph.replay()
            torch.cuda.synchronize(); st=torch.cuda.Event(True);en=torch.cuda.Event(True);st.record()
            for _ in range(50): gph.replay()
            en.record();torch.cuda.synchronize(); cu=st.elapsed_time(en)/50*1000.0
            cap=f"{cu:14.1f}"; ratio=f"{eu_cute/cu:15.2f}x"
        except Exception as ex:
            cap=f"CAPFAIL"; ratio=str(ex)[:30]; cu=float('nan')
        # confirm cute is NOT capturable
        capfail_cute="n/a"
        try:
            s2=torch.cuda.Stream(); s2.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s2):
                for _ in range(2): layer("cute")
            torch.cuda.current_stream().wait_stream(s2)
            g2=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g2): layer("cute")
            capfail_cute="CAPTURED(unexpected)"
        except Exception as ex:
            capfail_cute="cute_NOT_capturable(expected)"
        print(f"B={B:>2} {eu_cute:14.1f} {eu_v3:12.1f} {cap:>14} {ratio:>16}  [{capfail_cute}]",flush=True)
if __name__=="__main__": main()

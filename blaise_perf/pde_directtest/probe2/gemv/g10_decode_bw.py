"""g10: pin the compute ceiling. Read W + DECODE only (no x, no scales), sum.
   Compare 3 decoders: int-arith, hw-cvt-half2, hw-cvt then sum. Find decode throughput.
   This tells us the BEST achievable if compute fully overlaps the 4.1us read.
"""
import torch, tensorrt_llm
from torch.utils.cpp_extension import load_inline
dev="cuda"; torch.manual_seed(0); PEAK=7.67e12
CPP="torch::Tensor dec(torch::Tensor W, int64_t N, int64_t K, int64_t wpb, int64_t mode);"
CUDA=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
__device__ __forceinline__ float e2m1f(unsigned v){
  unsigned e=(v>>1)&3u,m=v&1u,s=(v>>3)&1u,b;
  if(e==0u) b=m?((s<<31)|(126u<<23)):(s<<31); else b=(s<<31)|((127u+e-1u)<<23)|(m<<22);
  return __int_as_float((int)b);
}
__device__ __forceinline__ half2 cvt2(uint8_t b){
  __half2_raw h=__nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b,__NV_E2M1);
  return *reinterpret_cast<half2*>(&h);
}
template<int WPB,int MODE>
__global__ void __launch_bounds__(WPB*32) deck(const uint8_t* __restrict__ W,int N,int K,__nv_bfloat16* __restrict__ Y){
  int lane=threadIdx.x&31,warp=threadIdx.x>>5; int n=blockIdx.x*WPB+warp; if(n>=N)return;
  int Kb=K/2,nU=Kb/16; const uint4* wrow=reinterpret_cast<const uint4*>(W+(size_t)n*Kb);
  float acc=0.f; half2 hacc=__floats2half2_rn(0.f,0.f);
  for(int u=lane;u<nU;u+=32){
    uint4 wv=wrow[u];
    if(MODE==0){ // int-arith decode all 32, sum
      uint32_t ws[4]={wv.x,wv.y,wv.z,wv.w};
      #pragma unroll
      for(int j=0;j<4;j++){ uint32_t w=ws[j];
        #pragma unroll
        for(int i=0;i<8;i++) acc+=e2m1f((w>>(i*4))&0xF);
      }
    } else { // hw cvt half2, accumulate half2
      uint32_t ws[4]={wv.x,wv.y,wv.z,wv.w};
      #pragma unroll
      for(int j=0;j<4;j++){ uint32_t w=ws[j];
        #pragma unroll
        for(int i=0;i<4;i++){ hacc=__hadd2(hacc,cvt2((w>>(i*8))&0xFF)); }
      }
    }
  }
  if(MODE!=0) acc=__low2float(hacc)+__high2float(hacc);
  for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o);
  if(lane==0) Y[n]=__float2bfloat16(acc);
}
torch::Tensor dec(torch::Tensor W,int64_t N,int64_t K,int64_t wpb,int64_t mode){
  auto Y=torch::empty({N},torch::dtype(torch::kBFloat16).device(W.device())); auto st=c10::cuda::getCurrentCUDAStream();
  auto wp=W.data_ptr<uint8_t>(); auto yp=reinterpret_cast<__nv_bfloat16*>(Y.data_ptr());
  #define LD(WPB,M) deck<WPB,M><<<(N+WPB-1)/WPB,WPB*32,0,st>>>(wp,N,K,yp)
  if(mode==0){ if(wpb==4)LD(4,0); else if(wpb==8)LD(8,0); else if(wpb==16)LD(16,0);}
  else { if(wpb==4)LD(4,1); else if(wpb==8)LD(8,1); else if(wpb==16)LD(16,1);}
  return Y;
}
'''
mod=load_inline(name="dec_g10",cpp_sources=CPP,cuda_sources=CUDA,functions=["dec"],extra_cuda_cflags=["-arch=sm_100","-O3"],verbose=False)
def cap_us(fn,windows=7,it=50):
    for _ in range(8): fn()
    torch.cuda.synchronize(); s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g=torch.cuda.CUDAGraph(); h={}
    with torch.cuda.graph(g): h["o"]=fn()
    for _ in range(5): g.replay()
    torch.cuda.synchronize(); best=float("inf")
    for _ in range(windows):
        a=torch.cuda.Event(True);b=torch.cuda.Event(True);a.record()
        for _ in range(it): g.replay()
        b.record(); torch.cuda.synchronize(); best=min(best,a.elapsed_time(b)/it*1000.0)
    return best
def main():
    print(f"dev={torch.cuda.get_device_name(0)} (moe_up read floor=4.1us)",flush=True)
    K,N=7168,2048; W=torch.randint(0,255,(N,K//2),device=dev,dtype=torch.uint8)
    for mode,lbl in [(0,"int-arith decode"),(1,"hw-cvt half2")]:
        for wpb in (4,8,16):
            cu=cap_us(lambda: mod.dec(W,N,K,wpb,mode))
            print(f"  {lbl} w{wpb}: {cu:.2f}us",flush=True)
    print("DONE",flush=True)
if __name__=="__main__": main()

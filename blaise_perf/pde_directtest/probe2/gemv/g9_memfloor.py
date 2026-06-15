"""g9: isolate the memory-access-pattern floor for warp-per-row WITH scale reads.
   Same structure as GEMV but trivial compute (xor-fold bytes). Reveals if scales/pattern
   are the ceiling vs compute. Also test: scales read as uint32 (4 blocks coalesced) vs byte.
"""
import torch, tensorrt_llm
from torch.utils.cpp_extension import load_inline
dev="cuda"; torch.manual_seed(0); PEAK=7.67e12
CPP="torch::Tensor mem(torch::Tensor W, torch::Tensor Wsf, torch::Tensor X, torch::Tensor Xsf, int64_t N, int64_t K, int64_t wpb, int64_t mode);"
CUDA=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
// mode0: read weights only (uint4). mode1: weights+scales(byte). mode2: weights+scales(uint32).
template<int WPB,int MODE>
__global__ void __launch_bounds__(WPB*32) memk(
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ Wsf,
    const uint8_t* __restrict__ X, const uint8_t* __restrict__ Xsf,
    int N, int K, __nv_bfloat16* __restrict__ Y){
  int lane=threadIdx.x&31, warp=threadIdx.x>>5;
  int n=blockIdx.x*WPB+warp; if(n>=N) return;
  int Kb=K/2, Ks=K/16, nU=Kb/16;
  const uint4* wrow=reinterpret_cast<const uint4*>(W+(size_t)n*Kb);
  const uint8_t* wsfb=Wsf+(size_t)n*Ks;
  const uint32_t* wsfw=reinterpret_cast<const uint32_t*>(Wsf+(size_t)n*Ks);
  uint32_t s=0;
  for(int u=lane; u<nU; u+=32){
    uint4 wv=wrow[u]; s^=wv.x^wv.y^wv.z^wv.w;
    if(MODE==1){ int blk=u*2; s^=wsfb[blk]; s^=wsfb[blk+1]; }
    if(MODE==2){ s^=wsfw[u>>1]; } // 1 uint32 covers 4 blocks=2 uint4
  }
  for(int o=16;o>0;o>>=1) s^=__shfl_down_sync(0xffffffff,s,o);
  if(lane==0) Y[n]=__float2bfloat16((float)(s&0xff));
}
torch::Tensor mem(torch::Tensor W, torch::Tensor Wsf, torch::Tensor X, torch::Tensor Xsf, int64_t N, int64_t K, int64_t wpb, int64_t mode){
  auto Y=torch::empty({N},torch::dtype(torch::kBFloat16).device(W.device()));
  auto st=c10::cuda::getCurrentCUDAStream();
  auto wp=W.data_ptr<uint8_t>(),wsfp=Wsf.data_ptr<uint8_t>(),xp=X.data_ptr<uint8_t>(),xsfp=Xsf.data_ptr<uint8_t>();
  auto yp=reinterpret_cast<__nv_bfloat16*>(Y.data_ptr());
  #define LM(WPB,M) memk<WPB,M><<<(N+WPB-1)/WPB,WPB*32,0,st>>>(wp,wsfp,xp,xsfp,N,K,yp)
  if(mode==0){ if(wpb==4)LM(4,0); else if(wpb==8)LM(8,0); else if(wpb==16)LM(16,0);}
  else if(mode==1){ if(wpb==4)LM(4,1); else if(wpb==8)LM(8,1); else if(wpb==16)LM(16,1);}
  else { if(wpb==4)LM(4,2); else if(wpb==8)LM(8,2); else if(wpb==16)LM(16,2);}
  return Y;
}
'''
mod=load_inline(name="mem_g9",cpp_sources=CPP,cuda_sources=CUDA,functions=["mem"],extra_cuda_cflags=["-arch=sm_100","-O3"],verbose=False)
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
    print(f"dev={torch.cuda.get_device_name(0)}",flush=True)
    for name,K,N in [("moe_up",7168,2048),("o_proj",16384,7168)]:
        W=torch.randint(0,255,(N,K//2),device=dev,dtype=torch.uint8)
        Wsf=torch.randint(0,255,(N*K//16,),device=dev,dtype=torch.uint8)
        X=torch.randint(0,255,(1,K//2),device=dev,dtype=torch.uint8)
        Xsf=torch.randint(0,255,(K//16,),device=dev,dtype=torch.uint8)
        rb=N*K/2
        for mode,lbl in [(0,"W-only"),(1,"W+sf-byte"),(2,"W+sf-u32")]:
            for wpb in (4,8,16):
                cu=cap_us(lambda: mod.mem(W,Wsf,X,Xsf,N,K,wpb,mode))
                print(f"  {name} {lbl} w{wpb}: {cu:.2f}us  ({rb/(cu*1e-6)/1e12:.2f}TB/s, {rb/(cu*1e-6)/PEAK*100:.1f}%)",flush=True)
    print("DONE",flush=True)
if __name__=="__main__": main()

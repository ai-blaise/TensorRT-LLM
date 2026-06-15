"""g3b: diagnose why warp GEMV is 30x slow. Compile with -Xptxas -v to see regs/spills/lmem.
   Also test a pure-copy kernel (just read all weight bytes, write nothing meaningful) to
   isolate raw read BW from compute."""
import torch, tensorrt_llm
from torch.utils.cpp_extension import load_inline
dev="cuda"; torch.manual_seed(0); PEAK=7.67e12
CPP="torch::Tensor wread(torch::Tensor W, int64_t N, int64_t K);"
CUDA=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
// pure read-bw test: each warp streams one weight row, sums bytes (vectorized uint4), writes 1 val.
template<int WPB>
__global__ void wread_k(const uint8_t* __restrict__ W, int N, int K, __nv_bfloat16* __restrict__ Y){
  int lane=threadIdx.x&31, warp=threadIdx.x>>5;
  int n=blockIdx.x*WPB+warp; if(n>=N) return;
  int Kb=K/2;
  const uint4* wrow=reinterpret_cast<const uint4*>(W+(size_t)n*Kb); // Kb/16 uint4
  int nv=Kb/16;
  uint32_t s=0;
  for(int i=lane;i<nv;i+=32){ uint4 v=wrow[i]; s+=v.x+v.y+v.z+v.w; }
  for(int o=16;o>0;o>>=1) s+=__shfl_down_sync(0xffffffff,s,o);
  if(lane==0) Y[n]=__float2bfloat16((float)(s&0xffff));
}
torch::Tensor wread(torch::Tensor W, int64_t N, int64_t K){
  auto Y=torch::empty({N},torch::dtype(torch::kBFloat16).device(W.device()));
  const int WPB=8; int blocks=(N+WPB-1)/WPB;
  auto st=c10::cuda::getCurrentCUDAStream();
  wread_k<WPB><<<blocks,WPB*32,0,st>>>(W.data_ptr<uint8_t>(),N,K,reinterpret_cast<__nv_bfloat16*>(Y.data_ptr()));
  return Y;
}
'''
mod=load_inline(name="wread_g3b",cpp_sources=CPP,cuda_sources=CUDA,functions=["wread"],
                extra_cuda_cflags=["-arch=sm_100","-O3","-Xptxas","-v"],verbose=True)
def cap_us(fn,windows=5,it=50):
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
    for name,K,N in [("o_proj",16384,7168),("moe_up",7168,2048),("kv_a",7168,2112)]:
        w=torch.randint(0,255,(N,K//2),device=dev,dtype=torch.uint8)
        cu=cap_us(lambda: mod.wread(w,N,K))
        rb=N*K/2
        print(f"PUREREAD {name} K={K} N={N}: {cu:.2f}us  readBW={rb/(cu*1e-6)/PEAK*100:.1f}% ({rb/(cu*1e-6)/1e12:.2f}TB/s)",flush=True)
    print("DONE",flush=True)
if __name__=="__main__": main()

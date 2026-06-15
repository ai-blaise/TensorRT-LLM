"""g11: split-K to raise occupancy on the decode ALUs. moe_up has only N=2048 rows
   -> only 2048 warps -> under-occupies decode. Split each row's K across SK warps.
   Test decode-only throughput vs SK to see if decode floor drops toward 4.1us mem floor.
"""
import torch, tensorrt_llm
from torch.utils.cpp_extension import load_inline
dev="cuda"; torch.manual_seed(0); PEAK=7.67e12
CPP="torch::Tensor dec(torch::Tensor W, int64_t N, int64_t K, int64_t sk);"
CUDA=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
__device__ __forceinline__ float e2m1f(unsigned v){
  unsigned e=(v>>1)&3u,m=v&1u,s=(v>>3)&1u,b;
  if(e==0u) b=m?((s<<31)|(126u<<23)):(s<<31); else b=(s<<31)|((127u+e-1u)<<23)|(m<<22);
  return __int_as_float((int)b);
}
// grid.x = N (one row per blockIdx.x's blockIdx.y group); SK warps per row split K.
// block = SK warps. warp w handles K-slice w of row n=blockIdx.x.
template<int SK>
__global__ void __launch_bounds__(SK*32) deck(const uint8_t* __restrict__ W,int N,int K,float* __restrict__ Y){
  int lane=threadIdx.x&31, warp=threadIdx.x>>5;
  int n=blockIdx.x; if(n>=N) return;
  int Kb=K/2,nU=Kb/16;
  const uint4* wrow=reinterpret_cast<const uint4*>(W+(size_t)n*Kb);
  float acc=0.f;
  // warp w, lane l : stride over uint4 by SK*32, offset warp*32+lane
  for(int u=warp*32+lane; u<nU; u+=SK*32){
    uint4 wv=wrow[u]; uint32_t ws[4]={wv.x,wv.y,wv.z,wv.w};
    #pragma unroll
    for(int j=0;j<4;j++){ uint32_t w=ws[j];
      #pragma unroll
      for(int i=0;i<8;i++) acc+=e2m1f((w>>(i*4))&0xF);
    }
  }
  for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o);
  // block-reduce across warps via atomicAdd to Y[n] (decode-only test)
  if(lane==0) atomicAdd(&Y[n],acc);
}
torch::Tensor dec(torch::Tensor W,int64_t N,int64_t K,int64_t sk){
  auto Y=torch::zeros({N},torch::dtype(torch::kFloat32).device(W.device())); auto st=c10::cuda::getCurrentCUDAStream();
  auto wp=W.data_ptr<uint8_t>(); auto yp=Y.data_ptr<float>();
  #define LD(SK) deck<SK><<<N,SK*32,0,st>>>(wp,N,K,yp)
  if(sk==1)LD(1); else if(sk==2)LD(2); else if(sk==4)LD(4); else if(sk==8)LD(8); else if(sk==16)LD(16);
  return Y;
}
'''
mod=load_inline(name="dec_g11",cpp_sources=CPP,cuda_sources=CUDA,functions=["dec"],extra_cuda_cflags=["-arch=sm_100","-O3"],verbose=False)
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
    print(f"dev={torch.cuda.get_device_name(0)} (moe_up mem floor 4.1us, decode-SK1 ~10us)",flush=True)
    for name,K,N in [("moe_up",7168,2048),("kv_a",7168,2112)]:
        W=torch.randint(0,255,(N,K//2),device=dev,dtype=torch.uint8)
        print(f" {name} K={K} N={N}:",flush=True)
        for sk in (1,2,4,8,16):
            cu=cap_us(lambda: mod.dec(W,N,K,sk))
            print(f"   SK={sk} ({N*sk} warps): {cu:.2f}us",flush=True)
    print("DONE",flush=True)
if __name__=="__main__": main()

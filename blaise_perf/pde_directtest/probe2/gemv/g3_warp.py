"""g3: warp-per-N-row GEMV. Coalesced K streaming, warp-reduce. Full-occupancy grid.
   Each warp -> one n. 32 lanes stride over K-blocks (8 bytes each), coalesced.
   Activations read from global (tiny, L2-resident). M rows in registers per lane.
"""
import torch, tensorrt_llm
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from torch.utils.cpp_extension import load_inline
dev="cuda"; torch.manual_seed(0); PEAK=7.67e12

E2M1_POS=torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.],device=dev)
def plain_q(w,vec=16):
    N,K=w.shape; wf=w.float(); gs=(448.0*6.0)/wf.abs().max(); wq=wf*gs
    wb=wq.view(N,K//vec,vec); amax=wb.abs().amax(dim=-1,keepdim=True)
    bs=(amax/6.0).clamp(min=1e-6); bsf=bs.to(torch.float8_e4m3fn).float()
    wn=(wb/bsf).clamp(-6,6); sign=(wn<0); mag=wn.abs()
    idx=(mag.unsqueeze(-1)-E2M1_POS.view(1,1,1,8)).abs().argmin(dim=-1)
    nib=(idx+sign.long()*8).view(N,K); packed=(nib[:,0::2]|(nib[:,1::2]<<4)).to(torch.uint8)
    su8=bsf.view(N,K//vec).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    return packed.contiguous(),su8,gs.reshape(1).contiguous()
def trt_q(x):
    g=(448.0*6.0)/x.abs().max().float(); fp4,sf=torch.ops.trtllm.fp4_quantize(x,g.reshape(1),16,False); return fp4,sf,g

CPP="torch::Tensor gemv(torch::Tensor W, torch::Tensor Wsf, torch::Tensor X, torch::Tensor Xsf, double alpha, int64_t N, int64_t K, int64_t M);"
CUDA=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
__device__ __forceinline__ float e2m1(unsigned v){
  const float t[16]={0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-0.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
  return t[v&15];
}
// One warp per output n. 32 lanes stride over K-blocks (each block=16 fp4=8 bytes=2 uint32).
// Coalesced: at step s, lane L reads block (s*32+L) -> consecutive 8B -> 256B transaction.
template<int M, int WPB>
__global__ void gemv_warp(const uint8_t* __restrict__ W, const uint8_t* __restrict__ Wsf,
                          const uint8_t* __restrict__ X, const uint8_t* __restrict__ Xsf,
                          float alpha, int N, int K, __nv_bfloat16* __restrict__ Y){
  int lane=threadIdx.x&31, warp=threadIdx.x>>5;
  int n=blockIdx.x*WPB+warp; if(n>=N) return;
  int Kb=K/2, Ks=K/16;
  const uint32_t* wrow=reinterpret_cast<const uint32_t*>(W+(size_t)n*Kb);  // K/8 uint32
  const __nv_fp8_e4m3* wsf=reinterpret_cast<const __nv_fp8_e4m3*>(Wsf)+(size_t)n*Ks;
  float acc[M];
  #pragma unroll
  for(int m=0;m<M;m++) acc[m]=0.f;
  // each block = 8 bytes = 2 uint32. lane L handles blocks L, L+32, ...
  for(int b=lane; b<Ks; b+=32){
    float ws=(float)wsf[b];
    uint32_t w_lo=wrow[b*2], w_hi=wrow[b*2+1];
    // 16 fp4 vals in [w_lo,w_hi]: byte j (0..7) -> nibbles 2j,2j+1
    #pragma unroll
    for(int m=0;m<M;m++){
      const uint32_t* xrow=reinterpret_cast<const uint32_t*>(X+(size_t)m*Kb);
      const __nv_fp8_e4m3* xsf=reinterpret_cast<const __nv_fp8_e4m3*>(Xsf)+(size_t)m*Ks;
      float xs=(float)xsf[b]; float wsxs=ws*xs;
      uint32_t x_lo=xrow[b*2], x_hi=xrow[b*2+1];
      float s=0.f;
      #pragma unroll
      for(int q=0;q<4;q++){
        uint8_t wbb=(w_lo>>(q*8))&0xFF, xbb=(x_lo>>(q*8))&0xFF;
        s+=e2m1(wbb&0xF)*e2m1(xbb&0xF)+e2m1(wbb>>4)*e2m1(xbb>>4);
      }
      #pragma unroll
      for(int q=0;q<4;q++){
        uint8_t wbb=(w_hi>>(q*8))&0xFF, xbb=(x_hi>>(q*8))&0xFF;
        s+=e2m1(wbb&0xF)*e2m1(xbb&0xF)+e2m1(wbb>>4)*e2m1(xbb>>4);
      }
      acc[m]+=s*wsxs;
    }
  }
  // warp reduce each acc[m]
  #pragma unroll
  for(int m=0;m<M;m++){
    #pragma unroll
    for(int o=16;o>0;o>>=1) acc[m]+=__shfl_down_sync(0xffffffff,acc[m],o);
    if(lane==0) Y[m*N+n]=__float2bfloat16(acc[m]*alpha);
  }
}
torch::Tensor gemv(torch::Tensor W, torch::Tensor Wsf, torch::Tensor X, torch::Tensor Xsf,
                   double alpha, int64_t N, int64_t K, int64_t M){
  auto Y=torch::empty({M,N}, torch::dtype(torch::kBFloat16).device(W.device()));
  const int WPB=8; int threads=WPB*32; int blocks=(N+WPB-1)/WPB;
  auto st=c10::cuda::getCurrentCUDAStream();
  auto wp=W.data_ptr<uint8_t>(),wsfp=Wsf.data_ptr<uint8_t>(),xp=X.data_ptr<uint8_t>(),xsfp=Xsf.data_ptr<uint8_t>();
  auto yp=reinterpret_cast<__nv_bfloat16*>(Y.data_ptr()); float a=(float)alpha;
  if(M==1) gemv_warp<1,WPB><<<blocks,threads,0,st>>>(wp,wsfp,xp,xsfp,a,N,K,yp);
  else if(M==4) gemv_warp<4,WPB><<<blocks,threads,0,st>>>(wp,wsfp,xp,xsfp,a,N,K,yp);
  else if(M==8) gemv_warp<8,WPB><<<blocks,threads,0,st>>>(wp,wsfp,xp,xsfp,a,N,K,yp);
  return Y;
}
'''
mod=load_inline(name="gemv_g3",cpp_sources=CPP,cuda_sources=CUDA,functions=["gemv"],
                extra_cuda_cflags=["-arch=sm_100","-O3","--use_fast_math"],verbose=False)

def cos(a,b):
    a=a.float().flatten();b=b.float().flatten();n=a.norm()*b.norm(); return (a@b/n).item() if n>0 else float('nan')
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
def bw(N,K,M,us): return (N*K/2+N*K/16+M*K/2+M*K/16+N*M*2)/(us*1e-6)/PEAK*100.0
shapes=[("kv_a",7168,2112),("q_b",1536,24576),("o_proj",16384,7168),("moe_up",7168,2048),("moe_down",2048,7168)]
def main():
    print(f"dev={torch.cuda.get_device_name(0)}",flush=True)
    print("ROW shape K N M backend cap_us bw% cos",flush=True)
    for name,K,N in shapes:
        w=torch.randn(N,K,device=dev,dtype=torch.bfloat16); wp,wsf,wg=plain_q(w); wf_t,wsf_t,wgt=trt_q(w)
        for M in (1,4,8):
            x=torch.randn(M,K,device=dev,dtype=torch.bfloat16); xp,xsf,xg=plain_q(x); xf_t,xsf_t,xgt=trt_q(x)
            am=1.0/(wg.item()*xg.item()); at=(1.0/(wgt*xgt)).reshape(1)
            ref=torch.ops.trtllm.fp4_gemm(xf_t,wf_t,xsf_t,wsf_t,at,fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4,torch.bfloat16).clone()
            y=mod.gemv(wp,wsf,xp,xsf,am,N,K,M).clone(); torch.cuda.synchronize(); c=cos(y,ref)
            cu=cap_us(lambda: mod.gemv(wp,wsf,xp,xsf,am,N,K,M))
            cl=cap_us(lambda: torch.ops.trtllm.cute_dsl_nvfp4_gemm_blackwell(xf_t,wf_t,xsf_t,wsf_t,at,torch.bfloat16))
            print(f"ROW {name} {K} {N} {M} mine {cu:.2f} {bw(N,K,M,cu):.1f} {c:.4f}   [cutedsl {cl:.2f} {bw(N,K,M,cl):.1f}]",flush=True)
    print("DONE",flush=True)
if __name__=="__main__": main()

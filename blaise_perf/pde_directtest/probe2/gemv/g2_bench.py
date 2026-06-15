"""g2: captured benchmark. my naive GEMV vs cutlass/cuda_core/cutedsl across shapes x M.
   Reports cap_us + achieved HBM BW% (peak 7.67 TB/s). Establishes baseline gap.
   Competitors use fp4_quantize swizzled output (their required layout); mine uses own
   plain quant. Cross-correctness: my-GEMV vs fp4_gemm oracle (same logical W) cos.
"""
import torch, tensorrt_llm
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from torch.utils.cpp_extension import load_inline
dev="cuda"; torch.manual_seed(0)
PEAK_TBs=7.67e12

E2M1_POS = torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.], device=dev)
def plain_q(w, vec=16):
    N,K=w.shape; wf=w.float()
    gs=(448.0*6.0)/wf.abs().max(); wq=wf*gs
    wb=wq.view(N,K//vec,vec); amax=wb.abs().amax(dim=-1,keepdim=True)
    bs=(amax/6.0).clamp(min=1e-6); bsf=bs.to(torch.float8_e4m3fn).float()
    wn=(wb/bsf).clamp(-6,6); sign=(wn<0); mag=wn.abs()
    idx=(mag.unsqueeze(-1)-E2M1_POS.view(1,1,1,8)).abs().argmin(dim=-1)
    nib=(idx+sign.long()*8).view(N,K)
    packed=(nib[:,0::2]|(nib[:,1::2]<<4)).to(torch.uint8)
    su8=bsf.view(N,K//vec).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    return packed.contiguous(), su8, gs.reshape(1).contiguous()

# swizzled quant for trtllm ops (their required layout)
def trt_q(x):
    g=(448.0*6.0)/x.abs().max().float()
    fp4,sf=torch.ops.trtllm.fp4_quantize(x,g.reshape(1),16,False)
    return fp4,sf,g

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
template<int M>
__global__ void gemv_naive(const uint8_t* __restrict__ W, const uint8_t* __restrict__ Wsf,
                           const uint8_t* __restrict__ X, const uint8_t* __restrict__ Xsf,
                           float alpha, int N, int K, __nv_bfloat16* __restrict__ Y){
  int n=blockIdx.x*blockDim.x+threadIdx.x; if(n>=N) return;
  int Kb=K/2, Ks=K/16;
  const uint8_t* wrow=W+(size_t)n*Kb;
  const __nv_fp8_e4m3* wsf=reinterpret_cast<const __nv_fp8_e4m3*>(Wsf)+(size_t)n*Ks;
  float acc[M];
  #pragma unroll
  for(int m=0;m<M;m++) acc[m]=0.f;
  for(int b=0;b<Ks;b++){
    float ws=(float)wsf[b];
    #pragma unroll
    for(int j=0;j<8;j++){
      uint8_t wb=wrow[b*8+j];
      float w0=e2m1(wb&0xF)*ws, w1=e2m1(wb>>4)*ws;
      #pragma unroll
      for(int m=0;m<M;m++){
        const uint8_t* xrow=X+(size_t)m*Kb;
        const __nv_fp8_e4m3* xsf=reinterpret_cast<const __nv_fp8_e4m3*>(Xsf)+(size_t)m*Ks;
        float xs=(float)xsf[b]; uint8_t xb=xrow[b*8+j];
        acc[m]+=(e2m1(xb&0xF)*xs)*w0+(e2m1(xb>>4)*xs)*w1;
      }
    }
  }
  #pragma unroll
  for(int m=0;m<M;m++) Y[m*N+n]=__float2bfloat16(acc[m]*alpha);
}
torch::Tensor gemv(torch::Tensor W, torch::Tensor Wsf, torch::Tensor X, torch::Tensor Xsf,
                   double alpha, int64_t N, int64_t K, int64_t M){
  auto Y=torch::empty({M,N}, torch::dtype(torch::kBFloat16).device(W.device()));
  int threads=256, blocks=(N+threads-1)/threads;
  auto st=c10::cuda::getCurrentCUDAStream();
  auto wp=W.data_ptr<uint8_t>(),wsfp=Wsf.data_ptr<uint8_t>(),xp=X.data_ptr<uint8_t>(),xsfp=Xsf.data_ptr<uint8_t>();
  auto yp=reinterpret_cast<__nv_bfloat16*>(Y.data_ptr()); float a=(float)alpha;
  if(M==1) gemv_naive<1><<<blocks,threads,0,st>>>(wp,wsfp,xp,xsfp,a,N,K,yp);
  else if(M==4) gemv_naive<4><<<blocks,threads,0,st>>>(wp,wsfp,xp,xsfp,a,N,K,yp);
  else if(M==8) gemv_naive<8><<<blocks,threads,0,st>>>(wp,wsfp,xp,xsfp,a,N,K,yp);
  return Y;
}
'''
mod=load_inline(name="gemv_g2",cpp_sources=CPP,cuda_sources=CUDA,functions=["gemv"],
                extra_cuda_cflags=["-arch=sm_100","-O3"],verbose=False)

def cos(a,b):
    a=a.float().flatten();b=b.float().flatten();n=a.norm()*b.norm()
    return (a@b/n).item() if n>0 else float('nan')

def cap_us(fn,windows=5,it=50):
    for _ in range(8): fn()
    torch.cuda.synchronize()
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g=torch.cuda.CUDAGraph(); h={}
    with torch.cuda.graph(g): h["o"]=fn()
    for _ in range(5): g.replay()
    torch.cuda.synchronize()
    best=float("inf")
    for _ in range(windows):
        a=torch.cuda.Event(True);b=torch.cuda.Event(True);a.record()
        for _ in range(it): g.replay()
        b.record(); torch.cuda.synchronize(); best=min(best,a.elapsed_time(b)/it*1000.0)
    return best, h["o"]

def bytes_moved(N,K,M):
    # weight fp4 = N*K/2 bytes; weight scale = N*K/16 bytes (fp8); act tiny; out N*M*2
    return N*K/2 + N*K/16 + M*K/2 + M*K/16 + N*M*2

def bwpct(N,K,M,us):
    return bytes_moved(N,K,M)/(us*1e-6)/PEAK_TBs*100.0

shapes=[("kv_a",7168,2112),("q_b",1536,24576),("o_proj",16384,7168),("moe_up",7168,2048),("moe_down",2048,7168)]

def main():
    print(f"dev={torch.cuda.get_device_name(0)} peak={PEAK_TBs/1e12:.2f}TB/s",flush=True)
    print("ROW shape K N M backend cap_us bw% cos",flush=True)
    for name,K,N in shapes:
        w=torch.randn(N,K,device=dev,dtype=torch.bfloat16)
        wp,wsf,wg=plain_q(w)                       # mine
        wf_t,wsf_t,wgt=trt_q(w)                     # trtllm swizzled
        for M in (1,4,8):
            x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
            xp,xsf,xg=plain_q(x)
            xf_t,xsf_t,xgt=trt_q(x)
            alpha_mine=1.0/(wg.item()*xg.item())
            alpha_t=(1.0/(wgt*xgt)).reshape(1)
            # oracle ref (cutlass-quant logical product)
            ref=torch.ops.trtllm.fp4_gemm(xf_t,wf_t,xsf_t,wsf_t,alpha_t,fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4,torch.bfloat16).clone()
            backends={
              "mine":   lambda: mod.gemv(wp,wsf,xp,xsf,alpha_mine,N,K,M),
              "cutlass":lambda: torch.ops.trtllm.nvfp4_gemm_cutlass(xf_t,wf_t,xsf_t,wsf_t,alpha_t,torch.bfloat16),
              "cutedsl":lambda: torch.ops.trtllm.cute_dsl_nvfp4_gemm_blackwell(xf_t,wf_t,xsf_t,wsf_t,alpha_t,torch.bfloat16),
            }
            if M<=8:
                backends["cuda_core"]=lambda: torch.ops.trtllm.nvfp4_gemm(xf_t,wf_t,xsf_t,wsf_t,alpha_t,torch.bfloat16,0,"cuda_core",None)
            for bn,fn in backends.items():
                try:
                    y=fn().clone(); torch.cuda.synchronize(); c=cos(y,ref)
                    cu,_=cap_us(fn)
                    print(f"ROW {name} {K} {N} {M} {bn} {cu:.2f} {bwpct(N,K,M,cu):.1f} {c:.4f}",flush=True)
                except Exception as ex:
                    print(f"ROW {name} {K} {N} {M} {bn} ERR ERR {type(ex).__name__}:{str(ex)[:40]}",flush=True)
    print("DONE",flush=True)

if __name__=="__main__": main()

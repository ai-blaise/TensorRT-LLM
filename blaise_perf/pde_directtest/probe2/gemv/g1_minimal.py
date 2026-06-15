"""g1: MINIMAL custom NVFP4 W4A4 GEMV. Validate loop + correctness vs bf16 ref."""
import torch, tensorrt_llm
from torch.utils.cpp_extension import load_inline
dev="cuda"; torch.manual_seed(0)

E2M1_POS = torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.], device=dev)
def plain_nvfp4_quant(w, vec=16):
    N,K=w.shape
    wf=w.float()
    gs=(448.0*6.0)/wf.abs().max()
    wq=wf*gs
    wb=wq.view(N,K//vec,vec)
    amax=wb.abs().amax(dim=-1,keepdim=True)
    blk_scale=(amax/6.0).clamp(min=1e-6)
    blk_scale_fp8=blk_scale.to(torch.float8_e4m3fn).float()
    wn=(wb/blk_scale_fp8).clamp(-6,6)
    sign=(wn<0); mag=wn.abs()
    idx=(mag.unsqueeze(-1)-E2M1_POS.view(1,1,1,8)).abs().argmin(dim=-1)
    nib=idx + sign.long()*8
    nib=nib.view(N,K)
    packed=(nib[:,0::2] | (nib[:,1::2]<<4)).to(torch.uint8)
    scale_u8=blk_scale_fp8.view(N,K//vec).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    return packed.contiguous(), scale_u8, gs.reshape(1).contiguous()

def plain_dequant(packed, scale_u8, gs, K, vec=16):
    N=packed.shape[0]
    lo=(packed&0xF).long(); hi=(packed>>4).long()
    E=torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.,-0.,-0.5,-1.,-1.5,-2.,-3.,-4.,-6.],device=dev)
    vals=torch.empty(N,K,device=dev)
    vals[:,0::2]=E[lo]; vals[:,1::2]=E[hi]
    sc=scale_u8.view(torch.float8_e4m3fn).float().view(N,K//vec,1)
    deq=(vals.view(N,K//vec,vec)*sc).view(N,K)/gs
    return deq

CPP_SRC = "torch::Tensor gemv(torch::Tensor W, torch::Tensor Wsf, torch::Tensor X, torch::Tensor Xsf, double alpha, int64_t N, int64_t K, int64_t M);"
CUDA_SRC = r'''
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
  int n = blockIdx.x*blockDim.x + threadIdx.x;
  if(n>=N) return;
  int Kb=K/2, Ks=K/16;
  const uint8_t* wrow = W + (size_t)n*Kb;
  const __nv_fp8_e4m3* wsf = reinterpret_cast<const __nv_fp8_e4m3*>(Wsf) + (size_t)n*Ks;
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
        float xs=(float)xsf[b];
        uint8_t xb=xrow[b*8+j];
        float x0=e2m1(xb&0xF)*xs, x1=e2m1(xb>>4)*xs;
        acc[m]+=w0*x0+w1*x1;
      }
    }
  }
  #pragma unroll
  for(int m=0;m<M;m++) Y[m*N+n]=__float2bfloat16(acc[m]*alpha);
}
torch::Tensor gemv(torch::Tensor W, torch::Tensor Wsf, torch::Tensor X, torch::Tensor Xsf,
                   double alpha, int64_t N, int64_t K, int64_t M){
  auto Y=torch::empty({M,N}, torch::dtype(torch::kBFloat16).device(W.device()));
  int threads=256; int blocks=(N+threads-1)/threads;
  auto st=c10::cuda::getCurrentCUDAStream();
  auto wp=W.data_ptr<uint8_t>(); auto wsfp=Wsf.data_ptr<uint8_t>();
  auto xp=X.data_ptr<uint8_t>(); auto xsfp=Xsf.data_ptr<uint8_t>();
  auto yp=reinterpret_cast<__nv_bfloat16*>(Y.data_ptr());
  float a=(float)alpha;
  if(M==1) gemv_naive<1><<<blocks,threads,0,st>>>(wp,wsfp,xp,xsfp,a,N,K,yp);
  else if(M==4) gemv_naive<4><<<blocks,threads,0,st>>>(wp,wsfp,xp,xsfp,a,N,K,yp);
  else if(M==8) gemv_naive<8><<<blocks,threads,0,st>>>(wp,wsfp,xp,xsfp,a,N,K,yp);
  return Y;
}
'''
mod=load_inline(name="gemv_g1b", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC,
                functions=["gemv"], extra_cuda_cflags=["-arch=sm_100","-O3"], verbose=False)

def cos(a,b):
    a=a.float().flatten();b=b.float().flatten();n=a.norm()*b.norm()
    return (a@b/n).item() if n>0 else float('nan')

def main():
    print(f"dev={torch.cuda.get_device_name(0)}",flush=True)
    for (name,K,N) in [("moe_up",7168,2048),("q_a",7168,1536)]:
        w=torch.randn(N,K,device=dev,dtype=torch.bfloat16)
        wp,wsf,wg=plain_nvfp4_quant(w)
        wdeq=plain_dequant(wp,wsf,wg,K)
        print(f"{name}: weight self-quant cos={cos(wdeq,w):.5f}",flush=True)
        for M in (1,4,8):
            x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
            xp,xsf,xg=plain_nvfp4_quant(x)
            xdeq=plain_dequant(xp,xsf,xg,K)
            ref=(xdeq@wdeq.t())
            alpha=1.0/(wg.item()*xg.item())
            y=mod.gemv(wp,wsf,xp,xsf,alpha,N,K,M)
            print(f"  M={M}: kernel-vs-ref cos={cos(y,ref):.5f}",flush=True)
    print("DONE",flush=True)

if __name__=="__main__": main()

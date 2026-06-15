"""g12: FINAL consolidated table. Best custom GEMV (int-arith decode, per-shape WPB)
   vs cutlass/cuda_core/cutedsl, captured us + BW%, for all 5 shapes x M in {1,4,8}.
   Supports M>1 (each lane holds M activation rows). Picks best WPB per (shape,M).
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
CPP="torch::Tensor gemv(torch::Tensor W, torch::Tensor Wsf, torch::Tensor X, torch::Tensor Xsf, double alpha, int64_t N, int64_t K, int64_t M, int64_t wpb);"
CUDA=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
__device__ __forceinline__ float e2m1f(unsigned v){
  unsigned e=(v>>1)&3u,m=v&1u,s=(v>>3)&1u,b;
  if(e==0u) b=m?((s<<31)|(126u<<23)):(s<<31); else b=(s<<31)|((127u+e-1u)<<23)|(m<<22);
  return __int_as_float((int)b);
}
template<int M,int WPB>
__global__ void __launch_bounds__(WPB*32) gemv(
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ Wsf,
    const uint8_t* __restrict__ X, const uint8_t* __restrict__ Xsf,
    float alpha, int N, int K, __nv_bfloat16* __restrict__ Y){
  int lane=threadIdx.x&31, warp=threadIdx.x>>5;
  int n=blockIdx.x*WPB+warp; if(n>=N) return;
  int Kb=K/2, Ks=K/16, nU=Kb/16;
  const uint4* wrow=reinterpret_cast<const uint4*>(W+(size_t)n*Kb);
  const __nv_fp8_e4m3* wsf=reinterpret_cast<const __nv_fp8_e4m3*>(Wsf)+(size_t)n*Ks;
  float acc[M];
  #pragma unroll
  for(int m=0;m<M;m++) acc[m]=0.f;
  for(int u=lane; u<nU; u+=32){
    uint4 wv=wrow[u]; int blk=u*2;
    float ws0=(float)wsf[blk], ws1=(float)wsf[blk+1];
    // decode 32 weight fp4 once
    float wd[32];
    #pragma unroll
    for(int i=0;i<8;i++) wd[i]=e2m1f((wv.x>>(i*4))&0xF);
    #pragma unroll
    for(int i=0;i<8;i++) wd[8+i]=e2m1f((wv.y>>(i*4))&0xF);
    #pragma unroll
    for(int i=0;i<8;i++) wd[16+i]=e2m1f((wv.z>>(i*4))&0xF);
    #pragma unroll
    for(int i=0;i<8;i++) wd[24+i]=e2m1f((wv.w>>(i*4))&0xF);
    #pragma unroll
    for(int m=0;m<M;m++){
      const uint4* xrow=reinterpret_cast<const uint4*>(X+(size_t)m*Kb);
      const __nv_fp8_e4m3* xsf=reinterpret_cast<const __nv_fp8_e4m3*>(Xsf)+(size_t)m*Ks;
      uint4 xv=xrow[u];
      float xs0=(float)xsf[blk], xs1=(float)xsf[blk+1];
      float p0=0.f,p1=0.f;
      #pragma unroll
      for(int i=0;i<8;i++) p0+=wd[i]*e2m1f((xv.x>>(i*4))&0xF);
      #pragma unroll
      for(int i=0;i<8;i++) p0+=wd[8+i]*e2m1f((xv.y>>(i*4))&0xF);
      #pragma unroll
      for(int i=0;i<8;i++) p1+=wd[16+i]*e2m1f((xv.z>>(i*4))&0xF);
      #pragma unroll
      for(int i=0;i<8;i++) p1+=wd[24+i]*e2m1f((xv.w>>(i*4))&0xF);
      acc[m]+=p0*(ws0*xs0)+p1*(ws1*xs1);
    }
  }
  #pragma unroll
  for(int m=0;m<M;m++){
    #pragma unroll
    for(int o=16;o>0;o>>=1) acc[m]+=__shfl_down_sync(0xffffffff,acc[m],o);
    if(lane==0) Y[m*N+n]=__float2bfloat16(acc[m]*alpha);
  }
}
torch::Tensor gemv(torch::Tensor W, torch::Tensor Wsf, torch::Tensor X, torch::Tensor Xsf,
                   double alpha, int64_t N, int64_t K, int64_t M, int64_t wpb){
  auto Y=torch::empty({M,N}, torch::dtype(torch::kBFloat16).device(W.device()));
  auto st=c10::cuda::getCurrentCUDAStream();
  auto wp=W.data_ptr<uint8_t>(),wsfp=Wsf.data_ptr<uint8_t>(),xp=X.data_ptr<uint8_t>(),xsfp=Xsf.data_ptr<uint8_t>();
  auto yp=reinterpret_cast<__nv_bfloat16*>(Y.data_ptr()); float a=(float)alpha;
  #define L(MM,WPB) gemv<MM,WPB><<<(N+WPB-1)/WPB,WPB*32,0,st>>>(wp,wsfp,xp,xsfp,a,N,K,yp)
  if(M==1){ if(wpb==4)L(1,4); else if(wpb==8)L(1,8); else if(wpb==16)L(1,16); else L(1,12);}
  else if(M==4){ if(wpb==4)L(4,4); else if(wpb==8)L(4,8); else if(wpb==16)L(4,16); else L(4,12);}
  else if(M==8){ if(wpb==4)L(8,4); else if(wpb==8)L(8,8); else if(wpb==16)L(8,16); else L(8,12);}
  return Y;
}
'''
mod=load_inline(name="gemv_g12",cpp_sources=CPP,cuda_sources=CUDA,functions=["gemv"],extra_cuda_cflags=["-arch=sm_100","-O3"],verbose=False)
def cos(a,b):
    a=a.float().flatten();b=b.float().flatten();n=a.norm()*b.norm(); return (a@b/n).item() if n>0 else float('nan')
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
def bw(N,K,M,us): return (N*K/2+N*K/16+M*K/2+M*K/16+N*M*2)/(us*1e-6)/PEAK*100.0
shapes=[("kv_a",7168,2112),("q_b",1536,24576),("o_proj",16384,7168),("moe_up",7168,2048),("moe_down",2048,7168)]
def main():
    print(f"dev={torch.cuda.get_device_name(0)} peak={PEAK/1e12:.2f}TB/s",flush=True)
    print("TBL shape KxN M | mine_us(wpb,bw%) cutlass(bw%) cuda_core(bw%) cutedsl(bw%) | mine_cos best",flush=True)
    for name,K,N in shapes:
        w=torch.randn(N,K,device=dev,dtype=torch.bfloat16); wp,wsf,wg=plain_q(w); wf_t,wsf_t,wgt=trt_q(w)
        for M in (1,4,8):
            x=torch.randn(M,K,device=dev,dtype=torch.bfloat16); xp,xsf,xg=plain_q(x); xf_t,xsf_t,xgt=trt_q(x)
            am=1.0/(wg.item()*xg.item()); at=(1.0/(wgt*xgt)).reshape(1)
            ref=torch.ops.trtllm.fp4_gemm(xf_t,wf_t,xsf_t,wsf_t,at,fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4,torch.bfloat16).clone()
            # mine: pick best wpb
            mb=1e9; mw=None; mc=0
            for wpb in (4,8,12,16):
                y=mod.gemv(wp,wsf,xp,xsf,am,N,K,M,wpb).clone(); torch.cuda.synchronize(); c=cos(y,ref)
                if c<0.99: continue
                cu=cap_us(lambda: mod.gemv(wp,wsf,xp,xsf,am,N,K,M,wpb))
                if cu<mb: mb=cu; mw=wpb; mc=c
            ct=cap_us(lambda: torch.ops.trtllm.nvfp4_gemm_cutlass(xf_t,wf_t,xsf_t,wsf_t,at,torch.bfloat16))
            cd=cap_us(lambda: torch.ops.trtllm.cute_dsl_nvfp4_gemm_blackwell(xf_t,wf_t,xsf_t,wsf_t,at,torch.bfloat16))
            try: cc=cap_us(lambda: torch.ops.trtllm.nvfp4_gemm(xf_t,wf_t,xsf_t,wsf_t,at,torch.bfloat16,0,"cuda_core",None))
            except Exception: cc=float('nan')
            allus={"mine":mb,"cutlass":ct,"cuda_core":cc,"cutedsl":cd}
            best=min((v,k) for k,v in allus.items() if v==v)[1]
            print(f"TBL {name} {K}x{N} M{M} | mine {mb:.2f}(w{mw},{bw(N,K,M,mb):.0f}%) cutlass {ct:.2f}({bw(N,K,M,ct):.0f}%) cuda_core {cc:.2f}({bw(N,K,M,cc):.0f}%) cutedsl {cd:.2f}({bw(N,K,M,cd):.0f}%) | cos{mc:.4f} BEST={best}",flush=True)
    print("DONE",flush=True)
if __name__=="__main__": main()

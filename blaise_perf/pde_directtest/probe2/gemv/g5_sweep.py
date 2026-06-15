"""g5: optimize the M=1 small-N case. Levers: bit-construction decode, WPB sweep,
   multi-load software pipelining for memory-level parallelism. Focus moe_up+kv_a.
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
// bit-construction E2M1 decode (pure int ops -> fp32 bits)
__device__ __forceinline__ float e2m1f(unsigned v){
  unsigned e=(v>>1)&3u, m=v&1u, s=(v>>3)&1u;
  unsigned bits;
  if(e==0u) bits = m ? ((s<<31)|(126u<<23)) : (s<<31);   // 0.5 or 0
  else bits = (s<<31) | ((127u+e-1u)<<23) | (m<<22);
  return __int_as_float((int)bits);
}
__device__ __forceinline__ void dec8(uint32_t w, float* o){
  #pragma unroll
  for(int i=0;i<8;i++) o[i]=e2m1f((w>>(i*4))&0xF);
}
// M=1 specialized, templated WPB. Each warp -> one n. Lane reads PF uint4 per step (pipelined).
template<int WPB, int PF>
__global__ void gemv1(const uint8_t* __restrict__ W, const uint8_t* __restrict__ Wsf,
                      const uint8_t* __restrict__ X, const uint8_t* __restrict__ Xsf,
                      float alpha, int N, int K, __nv_bfloat16* __restrict__ Y){
  int lane=threadIdx.x&31, warp=threadIdx.x>>5;
  int n=blockIdx.x*WPB+warp; if(n>=N) return;
  int Kb=K/2, Ks=K/16, nU=Kb/16;   // uint4 per row
  const uint4* wrow=reinterpret_cast<const uint4*>(W+(size_t)n*Kb);
  const uint4* xrow=reinterpret_cast<const uint4*>(X);   // M=1
  const __nv_fp8_e4m3* wsf=reinterpret_cast<const __nv_fp8_e4m3*>(Wsf)+(size_t)n*Ks;
  const __nv_fp8_e4m3* xsf=reinterpret_cast<const __nv_fp8_e4m3*>(Xsf);
  float acc=0.f;
  // grid-stride by 32*PF uint4
  for(int base=lane*PF; base<nU; base+=32*PF){
    uint4 wv[PF]; uint4 xv[PF];
    #pragma unroll
    for(int p=0;p<PF;p++){ int u=base+p; if(u<nU){ wv[p]=wrow[u]; xv[p]=xrow[u]; } }
    #pragma unroll
    for(int p=0;p<PF;p++){
      int u=base+p; if(u>=nU) break;
      int blk=u*2;
      float ws0=(float)wsf[blk], ws1=(float)wsf[blk+1];
      float xs0=(float)xsf[blk], xs1=(float)xsf[blk+1];
      float wf[32], xf[32];
      dec8(wv[p].x,wf); dec8(wv[p].y,wf+8); dec8(wv[p].z,wf+16); dec8(wv[p].w,wf+24);
      dec8(xv[p].x,xf); dec8(xv[p].y,xf+8); dec8(xv[p].z,xf+16); dec8(xv[p].w,xf+24);
      float p0=0.f,p1=0.f;
      #pragma unroll
      for(int i=0;i<16;i++) p0+=wf[i]*xf[i];
      #pragma unroll
      for(int i=16;i<32;i++) p1+=wf[i]*xf[i];
      acc+=p0*ws0*xs0+p1*ws1*xs1;
    }
  }
  #pragma unroll
  for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o);
  if(lane==0) Y[n]=__float2bfloat16(acc*alpha);
}
torch::Tensor gemv(torch::Tensor W, torch::Tensor Wsf, torch::Tensor X, torch::Tensor Xsf,
                   double alpha, int64_t N, int64_t K, int64_t M, int64_t wpb){
  auto Y=torch::empty({M,N}, torch::dtype(torch::kBFloat16).device(W.device()));
  int threads=wpb*32, blocks=(N+wpb-1)/wpb;
  auto st=c10::cuda::getCurrentCUDAStream();
  auto wp=W.data_ptr<uint8_t>(),wsfp=Wsf.data_ptr<uint8_t>(),xp=X.data_ptr<uint8_t>(),xsfp=Xsf.data_ptr<uint8_t>();
  auto yp=reinterpret_cast<__nv_bfloat16*>(Y.data_ptr()); float a=(float)alpha;
  #define LAUNCH(WPB,PF) gemv1<WPB,PF><<<(N+WPB-1)/WPB,WPB*32,0,st>>>(wp,wsfp,xp,xsfp,a,N,K,yp)
  if(wpb==8)  LAUNCH(8,2);
  else if(wpb==16) LAUNCH(16,2);
  else if(wpb==32) LAUNCH(32,2);
  else if(wpb==161) LAUNCH(16,1);
  else if(wpb==164) LAUNCH(16,4);
  else if(wpb==324) LAUNCH(32,4);
  return Y;
}
'''
mod=load_inline(name="gemv_g5",cpp_sources=CPP,cuda_sources=CUDA,functions=["gemv"],
                extra_cuda_cflags=["-arch=sm_100","-O3"],verbose=False)
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
shapes=[("moe_up",7168,2048),("kv_a",7168,2112),("o_proj",16384,7168),("moe_down",2048,7168)]
def main():
    print(f"dev={torch.cuda.get_device_name(0)}",flush=True)
    M=1
    for name,K,N in shapes:
        w=torch.randn(N,K,device=dev,dtype=torch.bfloat16); wp,wsf,wg=plain_q(w); wf_t,wsf_t,wgt=trt_q(w)
        x=torch.randn(M,K,device=dev,dtype=torch.bfloat16); xp,xsf,xg=plain_q(x); xf_t,xsf_t,xgt=trt_q(x)
        am=1.0/(wg.item()*xg.item()); at=(1.0/(wgt*xgt)).reshape(1)
        ref=torch.ops.trtllm.fp4_gemm(xf_t,wf_t,xsf_t,wsf_t,at,fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4,torch.bfloat16).clone()
        cl=cap_us(lambda: torch.ops.trtllm.cute_dsl_nvfp4_gemm_blackwell(xf_t,wf_t,xsf_t,wsf_t,at,torch.bfloat16))
        best=1e9; bestcfg=None
        for wpb in (161,8,16,32,164,324):
            y=mod.gemv(wp,wsf,xp,xsf,am,N,K,M,wpb).clone(); torch.cuda.synchronize(); c=cos(y,ref)
            cu=cap_us(lambda: mod.gemv(wp,wsf,xp,xsf,am,N,K,M,wpb))
            tag={161:"w16p1",8:"w8p2",16:"w16p2",32:"w32p2",164:"w16p4",324:"w32p4"}[wpb]
            print(f"  {name} {tag}: {cu:.2f}us bw{bw(N,K,M,cu):.1f}% cos{c:.4f}",flush=True)
            if c>0.99 and cu<best: best=cu; bestcfg=tag
        print(f"ROW {name} K{K} N{N} M1: BEST mine {best:.2f}us ({bestcfg}) vs cutedsl {cl:.2f}us  ratio {cl/best:.2f}x",flush=True)
    print("DONE",flush=True)
if __name__=="__main__": main()

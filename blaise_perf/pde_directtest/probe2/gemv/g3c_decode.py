"""g3c: verify branchless E2M1 decode (no LUT, pure arithmetic) matches the table, on GPU."""
import torch
from torch.utils.cpp_extension import load_inline
dev="cuda"
CPP="torch::Tensor dec(torch::Tensor nib);"
CUDA=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
// branchless E2M1 nibble (0..15) -> float. bit3=sign, b2b1=exp, b0=mant.
__device__ __forceinline__ float e2m1_arith(unsigned v){
  unsigned e=(v>>1)&3u, m=v&1u;
  // mag: e==0 -> 0.5*m ; else (1+0.5*m)*2^(e-1)
  float base = (e==0u) ? (0.5f*m) : ( (1.0f + 0.5f*(float)m) * (float)(1u<<(e-1)) );
  float sign = (v&8u) ? -1.0f : 1.0f;
  return sign*base;
}
// Alternative: fully branchless via fp32 bit construction
__device__ __forceinline__ float e2m1_bits(unsigned v){
  unsigned e=(v>>1)&3u, m=v&1u, s=(v>>3)&1u;
  // denormal case e==0: value = m? 0.5:0. normal: exp_field = 127 + (e-1); mant top bit = m.
  unsigned is0 = (e==0u);
  unsigned expf = 127u + e - 1u;          // for e>=1
  unsigned bits_norm = (s<<31) | (expf<<23) | (m<<22);
  unsigned bits_half = (s<<31) | (126u<<23);  // 0.5 with sign
  unsigned bits = is0 ? (m?bits_half:(s<<31)) : bits_norm;
  return __int_as_float((int)bits);
}
__global__ void deck(const uint8_t* nib, float* out, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  out[2*i]=e2m1_arith(nib[i]); out[2*i+1]=e2m1_bits(nib[i]);
}
torch::Tensor dec(torch::Tensor nib){
  int n=nib.numel(); auto out=torch::empty({n,2},torch::dtype(torch::kFloat32).device(nib.device()));
  deck<<<(n+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(nib.data_ptr<uint8_t>(),out.data_ptr<float>(),n);
  return out;
}
'''
mod=load_inline(name="dec_g3c",cpp_sources=CPP,cuda_sources=CUDA,functions=["dec"],extra_cuda_cflags=["-arch=sm_100","-O3"],verbose=False)
ref=torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.,-0.,-0.5,-1.,-1.5,-2.,-3.,-4.,-6.],device=dev)
nib=torch.arange(16,device=dev,dtype=torch.uint8)
out=mod.dec(nib)
print("nibble | ref | arith | bits",flush=True)
ok_a=ok_b=True
for i in range(16):
    a=out[i,0].item(); b=out[i,1].item(); r=ref[i].item()
    if abs(a-r)>1e-6: ok_a=False
    if abs(b-r)>1e-6: ok_b=False
    print(f"  {i:2d} {r:5.1f} {a:5.1f} {b:5.1f}",flush=True)
print(f"ARITH correct={ok_a}  BITS correct={ok_b}",flush=True)
print("DONE",flush=True)

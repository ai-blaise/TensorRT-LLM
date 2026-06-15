"""g7: probe hardware FP4->half conversion intrinsics + bf16x2 dot, available on SM100."""
import torch
from torch.utils.cpp_extension import load_inline
dev="cuda"
CPP="torch::Tensor t(torch::Tensor x);"
CUDA=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
// test: decode a uint8 (2 fp4) -> half2 via hardware cvt
__global__ void tk(const uint8_t* x, float* o, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  __nv_fp4x2_storage_t s = (__nv_fp4x2_storage_t)x[i];
  __half2_raw h = __nv_cvt_fp4x2_to_halfraw2(s, __NV_E2M1);
  half2 hh = *reinterpret_cast<half2*>(&h);
  o[2*i]=__low2float(hh); o[2*i+1]=__high2float(hh);
}
torch::Tensor t(torch::Tensor x){
  int n=x.numel(); auto o=torch::empty({n,2},torch::dtype(torch::kFloat32).device(x.device()));
  tk<<<(n+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(x.data_ptr<uint8_t>(),o.data_ptr<float>(),n);
  return o;
}
'''
try:
    mod=load_inline(name="cvt_g7",cpp_sources=CPP,cuda_sources=CUDA,functions=["t"],
                    extra_cuda_cflags=["-arch=sm_100","-O3"],verbose=True)
    # nibble layout: byte = (hi<<4)|lo. fp4x2 storage: which nibble is .x?
    x=torch.arange(256,device=dev,dtype=torch.uint8)
    o=mod.t(x)
    # print a few that map to known values: byte 0x21 = nibbles 1,2 -> 0.5,1.0
    print("byte 0x00:",o[0x00].tolist(),"(expect 0,0)",flush=True)
    print("byte 0x21:",o[0x21].tolist(),"(nibbles lo=1->0.5 hi=2->1.0)",flush=True)
    print("byte 0x70:",o[0x70].tolist(),"(lo=0->0 hi=7->6.0)",flush=True)
    print("byte 0x0F:",o[0x0F].tolist(),"(lo=15->-6 hi=0->0)",flush=True)
    print("HW_FP4_CVT=OK",flush=True)
except Exception as e:
    print("HW_FP4_CVT_ERR:",str(e)[:400],flush=True)
print("DONE",flush=True)

#!/usr/bin/env python3
# PDE top-k — ITER 12 (v14). v13 final = 16.4us (1.59x), floored by pass0 (full-C
# hist) + phase1 emit (full-C scan + ~1024 atomicAdds on a single SMEM counter).
# v14 WARP-AGGREGATES the emit atomics: per loop-iter, each warp ballots which
# lanes qualify, the leader does ONE atomicAdd of the popcount, lanes write at
# base+rank. Cuts ~1024 atomics -> ~(C/256) warp atomics. Same for phase2.
# NPASS configurable {2,3}. Warp-private 8-hist + warp-walk + compaction.
import os, time, math, sys
import torch
import tensorrt_llm
NPASS=2
for a in sys.argv[1:]:
    if a.startswith("NPASS="): NPASS=int(a.split("=")[1])
CUDA_SRC_TMPL = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
typedef unsigned long long ull;
__device__ __forceinline__ unsigned int f2o(float f){unsigned int u=__float_as_uint(f);unsigned int m=(unsigned int)(-(int)(u>>31))|0x80000000u;return u^m;}
#define NPASS __NPASS__
__device__ __forceinline__ void walk_warp(unsigned int* s_red,int krem,int lane,int* od,int* oa){
  unsigned int carry=0u; int fd=-1; unsigned int fa=0u;
  #pragma unroll
  for(int ch=7;ch>=0;--ch){ int b=ch*32+lane; unsigned int v=s_red[b]; unsigned int suf=v;
    #pragma unroll
    for(int off=1;off<32;off<<=1){ unsigned int up=__shfl_down_sync(0xffffffffu,suf,off); if(lane+off<32) suf+=up; }
    unsigned int ct=__shfl_sync(0xffffffffu,suf,0); unsigned int inc=suf+carry;
    unsigned int mask=__ballot_sync(0xffffffffu,(int)inc>=krem);
    if(mask&&fd<0){ int top=31-__clz(mask); unsigned int it=__shfl_sync(0xffffffffu,inc,top); unsigned int vt=__shfl_sync(0xffffffffu,v,top); fd=ch*32+top; fa=it-vt; }
    carry+=ct; }
  if(fd<0){ fd=0; fa=carry; }
  if(lane==0){ *od=fd; *oa=(int)fa; }
}
__global__ void __launch_bounds__(256) topk_v14(const float* __restrict__ scores,
    const int* __restrict__ seq_lens, int* __restrict__ out_idx, int B,int C,int k){
  int row=blockIdx.x; if(row>=B) return; int tid=threadIdx.x; int warp=tid>>5; int lane=tid&31;
  int sl=seq_lens[row];
  const float* srow=scores+(size_t)row*C; int* orow=out_idx+(size_t)row*k;
  extern __shared__ unsigned char smem[];
  unsigned int* s_key=(unsigned int*)smem; unsigned int* s_hist=s_key+C; unsigned int* s_red=s_hist;
  int* s_act=(int*)(s_hist+8*256);
  __shared__ int s_digit,s_acc; __shared__ unsigned int s_fill; __shared__ int s_nact;
  int valid=sl<C?sl:C; int kk=k<valid?k:valid;
  int C4=C&~3; const float4* srow4=reinterpret_cast<const float4*>(srow);
  for(int c4=tid;c4<(C4>>2);c4+=blockDim.x){ float4 v=srow4[c4]; int base=c4<<2;
    s_key[base+0]=f2o((base+0<sl)?v.x:-FLT_MAX); s_key[base+1]=f2o((base+1<sl)?v.y:-FLT_MAX);
    s_key[base+2]=f2o((base+2<sl)?v.z:-FLT_MAX); s_key[base+3]=f2o((base+3<sl)?v.w:-FLT_MAX); }
  for(int c=C4+tid;c<C;c+=blockDim.x) s_key[c]=f2o((c<sl)?srow[c]:-FLT_MAX);
  __syncthreads();
  unsigned int prefixS=0,pmaskS=0; int krem=kk;
  { int sh=24;
    for(int b=tid;b<8*256;b+=blockDim.x) s_hist[b]=0u; __syncthreads();
    unsigned int* myh=s_hist+warp*256;
    for(int c=tid;c<valid;c+=blockDim.x) atomicAdd(&myh[(s_key[c]>>sh)&255u],1u);
    __syncthreads();
    for(int b=tid;b<256;b+=blockDim.x){ unsigned int s=s_hist[b];
      #pragma unroll
      for(int w=1;w<8;++w) s+=s_hist[w*256+b]; s_red[b]=s; }
    __syncthreads();
    if(warp==0) walk_warp(s_red,krem,lane,&s_digit,&s_acc); __syncthreads();
    if(tid==0) s_nact=0; __syncthreads();
    krem-=s_acc; prefixS|=((unsigned int)s_digit)<<sh; pmaskS|=255u<<sh; int dg=s_digit;
    // warp-aggregated compaction. Loop bound padded to a multiple of blockDim so
    // EVERY lane calls __ballot_sync each iteration (q=false when out of range).
    int vpad=((valid+blockDim.x-1)/blockDim.x)*blockDim.x;
    for(int c=tid;c<vpad;c+=blockDim.x){ bool q=(c<valid)&&(((s_key[c]>>sh)&255u)==(unsigned)dg);
      unsigned int bm=__ballot_sync(0xffffffffu,q); int cnt=__popc(bm);
      int base=0; if(lane==0 && cnt) base=atomicAdd(&s_nact,cnt); base=__shfl_sync(0xffffffffu,base,0);
      if(q){ int r=__popc(bm & ((1u<<lane)-1)); s_act[base+r]=c; } }
    __syncthreads();
  }
  int nact=s_nact;
  for(int t=1;t<NPASS;++t){ int sh=24-8*t;
    for(int b=tid;b<8*256;b+=blockDim.x) s_hist[b]=0u; __syncthreads();
    unsigned int* myh=s_hist+warp*256;
    for(int i=tid;i<nact;i+=blockDim.x){ int c=s_act[i]; unsigned int key=s_key[c];
      if((key&pmaskS)==prefixS) atomicAdd(&myh[(key>>sh)&255u],1u); }
    __syncthreads();
    for(int b=tid;b<256;b+=blockDim.x){ unsigned int s=s_hist[b];
      #pragma unroll
      for(int w=1;w<8;++w) s+=s_hist[w*256+b]; s_red[b]=s; }
    __syncthreads();
    if(warp==0) walk_warp(s_red,krem,lane,&s_digit,&s_acc); __syncthreads();
    krem-=s_acc; prefixS|=((unsigned int)s_digit)<<sh; pmaskS|=255u<<sh; __syncthreads();
  }
  if(tid==0)s_fill=0u; __syncthreads();
  int vpad2=((valid+blockDim.x-1)/blockDim.x)*blockDim.x;
  // phase1 warp-aggregated emit: strictly-greater. Padded loop -> warp-uniform ballot.
  for(int c=tid;c<vpad2;c+=blockDim.x){ bool q=(c<valid)&&((s_key[c]&pmaskS)>prefixS);
    unsigned int bm=__ballot_sync(0xffffffffu,q); int cnt=__popc(bm);
    unsigned base=0; if(lane==0 && cnt) base=atomicAdd(&s_fill,(unsigned)cnt); base=__shfl_sync(0xffffffffu,base,0);
    if(q){ unsigned p=base+__popc(bm&((1u<<lane)-1)); if(p<(unsigned)k)orow[p]=c; } }
  __syncthreads();
  if((int)s_fill < k){
    int npad=((nact+blockDim.x-1)/blockDim.x)*blockDim.x;
    for(int i=tid;i<npad;i+=blockDim.x){ bool q=false; int c=0;
      if(i<nact){ c=s_act[i]; q=((s_key[c]&pmaskS)==prefixS); }
      unsigned int bm=__ballot_sync(0xffffffffu,q); int cnt=__popc(bm);
      unsigned base=0; if(lane==0 && cnt) base=atomicAdd(&s_fill,(unsigned)cnt); base=__shfl_sync(0xffffffffu,base,0);
      if(q){ unsigned p=base+__popc(bm&((1u<<lane)-1)); if(p<(unsigned)k)orow[p]=c; } }
  }
  __syncthreads();
  for(int j=(int)s_fill+tid;j<k;j+=blockDim.x) orow[j]=-1;
}
void launch_topk(torch::Tensor scores,torch::Tensor seq_lens,torch::Tensor out_idx,int k){
  int B=scores.size(0),C=scores.size(1);
  size_t smem=(size_t)C*4 + 8*256*4 + (size_t)C*4;
  cudaStream_t st=at::cuda::getCurrentCUDAStream();
  auto kern=topk_v14;
  if(smem>48*1024)cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
  kern<<<B,256,smem,st>>>(scores.data_ptr<float>(),seq_lens.data_ptr<int>(),out_idx.data_ptr<int>(),B,C,k);
}
'''
def build(npass):
    from torch.utils.cpp_extension import load_inline
    d=f"/tmp/torch_ext_pde14_{npass}"; os.environ["TORCH_EXTENSIONS_DIR"]=d; os.makedirs(d,exist_ok=True)
    src=CUDA_SRC_TMPL.replace("__NPASS__",str(npass)); t0=time.time()
    m=load_inline(name=f"pde_topk_v14_{npass}",cpp_sources="void launch_topk(torch::Tensor,torch::Tensor,torch::Tensor,int);",
        cuda_sources=src,functions=["launch_topk"],extra_cuda_cflags=["-O3","--use_fast_math","-arch=sm_100"],verbose=False)
    print(f"[build NPASS={npass}] {time.time()-t0:.1f}s",flush=True); return m
def setmatch(a,b):
    B=a.shape[0];fr=[]
    for r in range(B):
        sa=set(x for x in a[r].tolist() if x>=0); sb=set(x for x in b[r].tolist() if x>=0)
        fr.append(1.0 if not sb else len(sa&sb)/len(sb))
    return sum(fr)/len(fr)
def t_ms(fn,it=50,wu=15):
    for _ in range(wu): fn()
    torch.cuda.synchronize();s=torch.cuda.Event(True);e=torch.cuda.Event(True);s.record()
    for _ in range(it): fn()
    e.record();torch.cuda.synchronize();return s.elapsed_time(e)/it*1000.0
def cap_time(fn,it=80):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5): fn()
    torch.cuda.current_stream().wait_stream(s)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(15): g.replay()
    torch.cuda.synchronize();st=torch.cuda.Event(True);e=torch.cuda.Event(True);st.record()
    for _ in range(it): g.replay()
    e.record();torch.cuda.synchronize();return st.elapsed_time(e)/it*1000.0
def main():
    dev="cuda";torch.manual_seed(0);m=build(NPASS)
    op=torch.ops.trtllm.cute_dsl_indexer_topk_decode
    print(f"NPASS={NPASS}")
    print(f"{'case':>20} {'eC':>7} {'eV14':>7} {'capC':>8} {'capV14':>8} {'r':>8} {'sm':>6}",flush=True)
    for (C,k,label) in [(1032,64,"block C1032 k64"),(8192,1024,"final C8192 k1024")]:
        for B in [1,8,32,64]:
            scores=torch.randn(B,C,device=dev,dtype=torch.float32)
            sl=torch.full((B,),C,device=dev,dtype=torch.int32)
            out_p=torch.full((B,k),-1,device=dev,dtype=torch.int32)
            out_c=torch.full((B,k),-1,device=dev,dtype=torch.int32)
            m.launch_topk(scores,sl,out_p,k); op(scores,sl,out_c,k); torch.cuda.synchronize()
            gold=torch.topk(scores.float(),k,dim=1).indices
            sm=setmatch(out_p,gold)
            eV=t_ms(lambda: m.launch_topk(scores,sl,out_p,k))
            eC=t_ms(lambda: op(scores,sl,out_c,k))
            capC=cap_time(lambda: op(scores,sl,out_c,k))
            capV=cap_time(lambda: m.launch_topk(scores,sl,out_p,k))
            print(f"{label:>16} B={B:>2} {eC:7.1f} {eV:7.1f} {capC:8.1f} {capV:8.1f} {capV/capC:7.2f}x {sm:6.3f}",flush=True)
if __name__=="__main__": main()

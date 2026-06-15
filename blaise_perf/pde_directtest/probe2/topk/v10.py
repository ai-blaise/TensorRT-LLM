#!/usr/bin/env python3
# PDE radix-select top-k — ITER 8 (v10). v8/v9 plateaued at 12.3us block / 24.6us
# final captured. ROOT CAUSE of the floor: walk_parallel's Hillis-Steele suffix
# sum over 256 bins uses 16 __syncthreads per walk; x8 passes = 128 block barriers.
# v10 replaces it with a WARP-ONLY walk (warp 0, 32 threads, pure __shfl scans, no
# __syncthreads): warp 0 loads the 256 reduced bins as 8 chunks of 32, computes the
# descending suffix-sum across chunks in registers (running total carried high->low),
# and finds the crossing bin with __ballot/__shfl. One __syncthreads to publish.
# Cuts ~120 block barriers -> should approach cute's 8us.
import os, time, math, sys
import torch
import tensorrt_llm

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
typedef unsigned long long ull;
__device__ __forceinline__ unsigned int f2o(float f){unsigned int u=__float_as_uint(f);unsigned int m=(unsigned int)(-(int)(u>>31))|0x80000000u;return u^m;}

// WARP-ONLY descending-suffix walk over 256 bins in s_red. Only warp 0 runs.
// Finds largest bin b with sum_{j>=b} hist[j] >= krem; writes digit + acc(=sum_{j>b}).
// Processes 8 chunks of 32 bins from HIGH chunk (224..255) to LOW (0..31), carrying
// a running suffix total across chunks. Within a chunk: inclusive suffix-sum via
// shuffles (descending within the 32 lanes), compare to krem with the carry added.
__device__ __forceinline__ void walk_warp(unsigned int* s_red, int krem, int lane,
                                          int* out_digit, int* out_acc){
  unsigned int carry = 0u;       // sum of all bins in chunks strictly ABOVE current
  int found_digit = -1;
  unsigned int found_acc = 0u;
  // iterate chunks high->low: chunk ch covers bins [ch*32 , ch*32+31]
  #pragma unroll
  for(int ch=7; ch>=0; --ch){
    int b = ch*32 + lane;                 // this lane's bin (lane 0..31 -> low..high in chunk)
    unsigned int v = s_red[b];
    // we want, within the chunk, descending inclusive suffix: suf[b] = sum_{j in chunk, j>=b}.
    // lane maps to bin b=ch*32+lane, so higher lane = higher bin. Descending suffix over bins
    // = inclusive scan from high lane down to low lane. Use shfl_down accumulation:
    unsigned int suf = v;
    #pragma unroll
    for(int off=1; off<32; off<<=1){
      unsigned int up = __shfl_down_sync(0xffffffffu, suf, off); // lane+off (higher bin)
      if(lane+off < 32) suf += up;
    }
    // suf[lane] = sum_{j in chunk, bin>=b}. Total chunk sum = suf at lane 0.
    unsigned int chunk_total = __shfl_sync(0xffffffffu, suf, 0);
    // inclusive suffix INCLUDING higher chunks = suf + carry
    unsigned int inc = suf + carry;
    // largest bin b with inc >= krem. higher lane = higher bin. find max lane where inc>=krem.
    unsigned int mask = __ballot_sync(0xffffffffu, (int)inc >= krem);
    if(mask){
      int top = 31 - __clz(mask);          // highest lane satisfying
      int cand_b = ch*32 + top;
      // acc = sum_{j>b} = inc(at top) - hist[b]... = (suf[top]+carry) - v[top]. Get via shfl.
      unsigned int inc_top = __shfl_sync(0xffffffffu, inc, top);
      unsigned int v_top   = __shfl_sync(0xffffffffu, v,   top);
      if(cand_b > found_digit){ found_digit=cand_b; found_acc = inc_top - v_top; }
      // since we go high chunk first, the FIRST chunk with a hit gives the global max bin.
      // (a higher chunk = strictly higher bins). So we can break.
      if(lane==0){} // no-op
      // record + stop scanning lower chunks
      carry += chunk_total;
      // break out: but pragma unroll loop; use a flag
      ch = -1; // force loop end next iter (ch-- -> -2 < 0)
      continue;
    }
    carry += chunk_total;
  }
  if(found_digit<0){ found_digit=0; found_acc = carry; } // krem>total (shouldn't happen); pick bin0
  if(lane==0){ *out_digit=found_digit; *out_acc=(int)found_acc; }
}

__global__ void __launch_bounds__(256) topk_v10(const float* __restrict__ scores,
    const int* __restrict__ seq_lens, int* __restrict__ out_idx, int B,int C,int k){
  int row=blockIdx.x; if(row>=B) return; int tid=threadIdx.x; int warp=tid>>5; int lane=tid&31;
  int sl=seq_lens[row];
  const float* srow=scores+(size_t)row*C; int* orow=out_idx+(size_t)row*k;
  extern __shared__ unsigned char smem[];
  unsigned int* s_key=(unsigned int*)smem; unsigned int* s_hist=s_key+C;
  unsigned int* s_red=s_hist; int* s_act=(int*)(s_hist+8*256);
  __shared__ int s_digit,s_acc; __shared__ unsigned int s_fill; __shared__ int s_nact;
  int valid=sl<C?sl:C; int kk=k<valid?k:valid;

  int C4=C&~3; const float4* srow4=reinterpret_cast<const float4*>(srow);
  for(int c4=tid;c4<(C4>>2);c4+=blockDim.x){ float4 v=srow4[c4]; int base=c4<<2;
    s_key[base+0]=f2o((base+0<sl)?v.x:-FLT_MAX); s_key[base+1]=f2o((base+1<sl)?v.y:-FLT_MAX);
    s_key[base+2]=f2o((base+2<sl)?v.z:-FLT_MAX); s_key[base+3]=f2o((base+3<sl)?v.w:-FLT_MAX); }
  for(int c=C4+tid;c<C;c+=blockDim.x) s_key[c]=f2o((c<sl)?srow[c]:-FLT_MAX);
  __syncthreads();

  unsigned int prefixS=0,pmaskS=0; int krem=kk;
  // PASS 0 (score MSD) full-C + compact
  {
    int sh=24;
    for(int b=tid;b<8*256;b+=blockDim.x) s_hist[b]=0u; __syncthreads();
    unsigned int* myh=s_hist+warp*256;
    for(int c=tid;c<valid;c+=blockDim.x) atomicAdd(&myh[(s_key[c]>>sh)&255u],1u);
    __syncthreads();
    for(int b=tid;b<256;b+=blockDim.x){ unsigned int s=s_hist[b];
      #pragma unroll
      for(int w=1;w<8;++w) s+=s_hist[w*256+b]; s_red[b]=s; }
    __syncthreads();
    if(warp==0) walk_warp(s_red,krem,lane,&s_digit,&s_acc);
    __syncthreads();
    if(tid==0) s_nact=0; __syncthreads();
    krem-=s_acc; prefixS|=((unsigned int)s_digit)<<sh; pmaskS|=255u<<sh; int dg=s_digit;
    for(int c=tid;c<valid;c+=blockDim.x){ if(((s_key[c]>>sh)&255u)==(unsigned)dg){ int p=atomicAdd(&s_nact,1); s_act[p]=c; } }
    __syncthreads();
  }
  int nact=s_nact;
  for(int t=1;t<4;++t){ int sh=24-8*t;
    for(int b=tid;b<8*256;b+=blockDim.x) s_hist[b]=0u; __syncthreads();
    unsigned int* myh=s_hist+warp*256;
    for(int i=tid;i<nact;i+=blockDim.x){ int c=s_act[i]; unsigned int key=s_key[c];
      if((key&pmaskS)==prefixS) atomicAdd(&myh[(key>>sh)&255u],1u); }
    __syncthreads();
    for(int b=tid;b<256;b+=blockDim.x){ unsigned int s=s_hist[b];
      #pragma unroll
      for(int w=1;w<8;++w) s+=s_hist[w*256+b]; s_red[b]=s; }
    __syncthreads();
    if(warp==0) walk_warp(s_red,krem,lane,&s_digit,&s_acc);
    __syncthreads();
    krem-=s_acc; prefixS|=((unsigned int)s_digit)<<sh; pmaskS|=255u<<sh; __syncthreads();
  }
  unsigned int prefixI=0,pmaskI=0;
  for(int t=4;t<8;++t){ int sh=64-8*(t+1);
    for(int b=tid;b<8*256;b+=blockDim.x) s_hist[b]=0u; __syncthreads();
    unsigned int* myh=s_hist+warp*256;
    for(int i=tid;i<nact;i+=blockDim.x){ int c=s_act[i];
      if((s_key[c]&pmaskS)==prefixS){ unsigned int ic=~(unsigned int)c;
        if((ic&pmaskI)==prefixI) atomicAdd(&myh[(ic>>sh)&255u],1u); } }
    __syncthreads();
    for(int b=tid;b<256;b+=blockDim.x){ unsigned int s=s_hist[b];
      #pragma unroll
      for(int w=1;w<8;++w) s+=s_hist[w*256+b]; s_red[b]=s; }
    __syncthreads();
    if(warp==0) walk_warp(s_red,krem,lane,&s_digit,&s_acc);
    __syncthreads();
    krem-=s_acc; prefixI|=((unsigned int)s_digit)<<sh; pmaskI|=255u<<sh; __syncthreads();
  }
  ull thr=((ull)prefixS<<32)|(ull)prefixI;
  if(tid==0)s_fill=0u; __syncthreads();
  for(int c=tid;c<valid;c+=blockDim.x){ ull key=((ull)s_key[c]<<32)|(ull)(~(unsigned int)c);
    if(key>=thr){unsigned p=atomicAdd(&s_fill,1u); if(p<(unsigned)k)orow[p]=c;} }
  __syncthreads();
  for(int j=(int)s_fill+tid;j<k;j+=blockDim.x) orow[j]=-1;
}
void launch_topk(torch::Tensor scores,torch::Tensor seq_lens,torch::Tensor out_idx,int k){
  int B=scores.size(0),C=scores.size(1);
  size_t smem=(size_t)C*4 + 8*256*4 + (size_t)C*4;
  cudaStream_t st=at::cuda::getCurrentCUDAStream();
  auto kern=topk_v10;
  if(smem>48*1024)cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
  kern<<<B,256,smem,st>>>(scores.data_ptr<float>(),seq_lens.data_ptr<int>(),out_idx.data_ptr<int>(),B,C,k);
}
'''
def build():
    from torch.utils.cpp_extension import load_inline
    d="/tmp/torch_ext_pde10"; os.environ["TORCH_EXTENSIONS_DIR"]=d; os.makedirs(d,exist_ok=True)
    t0=time.time()
    m=load_inline(name="pde_topk_v10",cpp_sources="void launch_topk(torch::Tensor,torch::Tensor,torch::Tensor,int);",
        cuda_sources=CUDA_SRC,functions=["launch_topk"],extra_cuda_cflags=["-O3","--use_fast_math","-arch=sm_100"],verbose=False)
    print(f"[build] {time.time()-t0:.1f}s",flush=True); return m
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
    dev="cuda";torch.manual_seed(0);m=build()
    op=torch.ops.trtllm.cute_dsl_indexer_topk_decode
    CAP=("CAP" in sys.argv)
    print(f"{'case':>20} {'eC':>7} {'eV10':>7} {'capC':>8} {'capV10':>8} {'r':>8} {'sm':>6}",flush=True)
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
            capC=capV=float('nan')
            if CAP:
                try: capC=cap_time(lambda: op(scores,sl,out_c,k))
                except Exception as ex: print(f"  [capC FAIL B={B}] {str(ex)[:80]}",flush=True)
                try: capV=cap_time(lambda: m.launch_topk(scores,sl,out_p,k))
                except Exception as ex: print(f"  [capV FAIL B={B}] {str(ex)[:80]}",flush=True)
            r=(capV/capC) if (capC==capC and capV==capV) else float('nan')
            print(f"{label:>16} B={B:>2} {eC:7.1f} {eV:7.1f} {capC:8.1f} {capV:8.1f} {r:7.2f}x {sm:6.3f}",flush=True)
if __name__=="__main__": main()

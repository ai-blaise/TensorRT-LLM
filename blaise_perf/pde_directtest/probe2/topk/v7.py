#!/usr/bin/env python3
# PDE radix-select top-k — ITER 5 (v7): MULTI-CTA-PER-ROW cooperative.
# Captured cute_dsl = 8.2us(block)/10.3us(final) is the REAL target (eager 27us is
# all launch overhead; both capture). v6 single-CTA is compute-bound at 22-41us
# captured = 3-4x too slow. Fix: G CTAs cooperate per row so the O(C) scan is
# divided by G. One persistent cooperative grid (grid.sync between passes); fully
# CUDA-graph capturable (probe confirmed coop kernels capture on sm_100).
#
# Grid: B*G blocks. group g_row = blockIdx.x / G handles row g_row; my_rank =
# blockIdx.x % G. Per-row global scratch: hist[B*256], thr/digit/acc via per-row
# small global arrays, out_cnt[B]. 3 grid.syncs/pass (reference discipline).
import os, time, math, sys
import torch
import tensorrt_llm

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cfloat>
namespace cg=cooperative_groups;
typedef unsigned long long ull;
__device__ __forceinline__ unsigned int f2o(float f){unsigned int u=__float_as_uint(f);unsigned int m=(unsigned int)(-(int)(u>>31))|0x80000000u;return u^m;}
__device__ __forceinline__ ull mk(float s,int i){return ((ull)f2o(s)<<32)|(ull)(~(unsigned int)i);}

// Per-row global scratch:
//   ghist[B*256]  digit histogram (reset each pass)
//   gdigit[B], gacc[B]  threshold-walk result broadcast (rank0 of group writes)
//   gcnt[B]  emit fill counter
// Each GROUP of G CTAs cooperates on ONE row.
__global__ void __launch_bounds__(256) topk_v7(const float* __restrict__ scores,
    const int* __restrict__ seq_lens, int* __restrict__ out_idx,
    unsigned int* __restrict__ ghist, int* __restrict__ gdigit, int* __restrict__ gacc,
    unsigned int* __restrict__ gcnt, int B,int C,int k,int G){
  cg::grid_group grid=cg::this_grid();
  int tid=threadIdx.x; int warp=tid>>5;
  int row=blockIdx.x / G; int my_rank=blockIdx.x % G;
  bool active = (row < B);
  int sl = active ? seq_lens[row] : 0;
  const float* srow = active ? scores+(size_t)row*C : nullptr;
  int* orow = active ? out_idx+(size_t)row*k : nullptr;
  unsigned int* myhist = active ? ghist+(size_t)row*256 : nullptr;
  int valid = active ? (sl<C?sl:C) : 0;
  int kk = active ? (k<valid?k:valid) : 0;

  // warp-private SMEM histogram (8*256) to cut atomic contention before the
  // single global flush per CTA.
  extern __shared__ unsigned char smem[];
  unsigned int* s_hist=(unsigned int*)smem; // 8*256
  unsigned int* s_red=(unsigned int*)(s_hist+8*256); // 256 (group-local reduced)

  // reset emit counter once
  if(active && my_rank==0 && tid==0) gcnt[row]=0u;

  ull prefix=0,pmask=0; int krem=kk;
  for(int t=0;t<8;++t){ int sh=64-8*(t+1);
    // reset this row's global histogram (rank0 only)
    if(active && my_rank==0){ for(int b=tid;b<256;b+=blockDim.x) myhist[b]=0u; }
    grid.sync(); // (R) global hist zeroed visible to whole group
    // warp-private SMEM hist over my slice of C
    for(int b=tid;b<8*256;b+=blockDim.x) s_hist[b]=0u; __syncthreads();
    if(active){ unsigned int* myh=s_hist+warp*256;
      for(int c=my_rank*blockDim.x+tid; c<valid; c+=G*blockDim.x){ ull key=mk(srow[c],c);
        if((key&pmask)==prefix) atomicAdd(&myh[(unsigned)((key>>sh)&255)],1u); }
      __syncthreads();
      // reduce 8 warp-private -> s_red[256]
      for(int b=tid;b<256;b+=blockDim.x){ unsigned int s=s_hist[b];
        #pragma unroll
        for(int w=1;w<8;++w) s+=s_hist[w*256+b]; s_red[b]=s; }
      __syncthreads();
      // flush s_red to global per-row hist
      for(int b=tid;b<256;b+=blockDim.x){ if(s_red[b]) atomicAdd(&myhist[b],s_red[b]); }
    }
    grid.sync(); // (A) global hist complete
    // every active CTA walks global hist identically (read-only) -> digit
    if(active){
      for(int b=tid;b<256;b+=blockDim.x) s_red[b]=myhist[b]; __syncthreads();
      if(tid==0){int acc=0,dg=0;for(int bn=255;bn>=0;--bn){int cc=(int)s_red[bn];if(acc+cc>=krem){dg=bn;break;}acc+=cc;}
        gdigit[row]=dg; gacc[row]=acc;}
    }
    grid.sync(); // (B) walk (global read) done before any race; broadcast via gdigit
    int dg=active?gdigit[row]:0; int acc=active?gacc[row]:0;
    krem-=acc; prefix|=((ull)dg)<<sh; pmask|=((ull)255)<<sh;
    // no separate reset grid.sync: the (R) at loop top re-zeros before next atomics
  }
  ull thr=prefix;
  // emit: each CTA scans its slice, atomicAdd into per-row global counter
  if(active){ unsigned int* cnt=gcnt+row;
    for(int c=my_rank*blockDim.x+tid; c<valid; c+=G*blockDim.x){ if(mk(srow[c],c)>=thr){
      unsigned p=atomicAdd(cnt,1u); if(p<(unsigned)k) orow[p]=c; } }
  }
  grid.sync(); // (E) all emits done
  // pad -1 (rank0 of group)
  if(active && my_rank==0){ unsigned f=gcnt[row]; for(int j=(int)f+tid;j<k;j+=blockDim.x) orow[j]=-1; }
}

struct Scratch{ unsigned int* ghist; int* gdigit; int* gacc; unsigned int* gcnt; int B; };
static Scratch g_s={nullptr,nullptr,nullptr,nullptr,0};
static int g_maxblocks=0;

void launch_topk(torch::Tensor scores,torch::Tensor seq_lens,torch::Tensor out_idx,int k){
  int B=scores.size(0),C=scores.size(1);
  cudaStream_t st=at::cuda::getCurrentCUDAStream();
  size_t smem=(8*256+256)*sizeof(unsigned int);
  auto kern=topk_v7;
  if(g_maxblocks==0){ int nb=0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb,(void*)kern,256,smem);
    int dev=0;cudaGetDevice(&dev);int sms=0;cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev);
    g_maxblocks=nb*sms; }
  // choose G: as many CTAs/row as fit, capped so threads cover C reasonably.
  int max_per_row = g_maxblocks / B; if(max_per_row<1) max_per_row=1;
  int want = (C + 256 - 1)/256; // 1 CTA covers 256 elems/pass-stride; cap useful G
  int G = max_per_row < want ? max_per_row : want; if(G<1)G=1;
  int total = B*G;
  // (re)alloc scratch
  if(g_s.B < B){
    if(g_s.ghist) cudaFree(g_s.ghist);
    cudaMalloc(&g_s.ghist,(size_t)B*256*sizeof(unsigned int));
    cudaMalloc(&g_s.gdigit,(size_t)B*sizeof(int));
    cudaMalloc(&g_s.gacc,(size_t)B*sizeof(int));
    cudaMalloc(&g_s.gcnt,(size_t)B*sizeof(unsigned int));
    cudaMemset(g_s.ghist,0,(size_t)B*256*sizeof(unsigned int));
    g_s.B=B;
  }
  const float* psc=scores.data_ptr<float>(); const int* psl=seq_lens.data_ptr<int>();
  int* po=out_idx.data_ptr<int>(); unsigned int* ph=g_s.ghist; int* pd=g_s.gdigit; int* pa=g_s.gacc; unsigned int* pc=g_s.gcnt;
  int BB=B,CC=C,kk=k,GG=G;
  void* args[]={&psc,&psl,&po,&ph,&pd,&pa,&pc,&BB,&CC,&kk,&GG};
  cudaLaunchCooperativeKernel((void*)kern, dim3(total),dim3(256),args,smem,st);
}
int chosen_G(int B,int C){
  if(g_maxblocks==0) return -1;
  int max_per_row=g_maxblocks/B; if(max_per_row<1)max_per_row=1;
  int want=(C+255)/256; int G=max_per_row<want?max_per_row:want; if(G<1)G=1; return G;
}
'''
def build():
    from torch.utils.cpp_extension import load_inline
    d="/tmp/torch_ext_pde7"; os.environ["TORCH_EXTENSIONS_DIR"]=d; os.makedirs(d,exist_ok=True)
    t0=time.time()
    m=load_inline(name="pde_topk_v7",cpp_sources="void launch_topk(torch::Tensor,torch::Tensor,torch::Tensor,int); int chosen_G(int,int);",
        cuda_sources=CUDA_SRC,functions=["launch_topk","chosen_G"],extra_cuda_cflags=["-O3","--use_fast_math","-arch=sm_100"],verbose=False)
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
    hdr_extra = "capC capV7 capV7/capC" if CAP else ""
    print(f"{'case':>20} {'eC':>7} {'eV7':>7} {'capC':>8} {'capV7':>8} {'cV7/cC':>8} {'G':>4} {'sm':>6}",flush=True)
    for (C,k,label) in [(1032,64,"block C1032 k64"),(8192,1024,"final C8192 k1024")]:
        for B in [1,8,32,64]:
            scores=torch.randn(B,C,device=dev,dtype=torch.float32)
            sl=torch.full((B,),C,device=dev,dtype=torch.int32)
            out_p=torch.full((B,k),-1,device=dev,dtype=torch.int32)
            out_c=torch.full((B,k),-1,device=dev,dtype=torch.int32)
            m.launch_topk(scores,sl,out_p,k); op(scores,sl,out_c,k); torch.cuda.synchronize()
            G=m.chosen_G(B,C)
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
            print(f"{label:>16} B={B:>2} {eC:7.1f} {eV:7.1f} {capC:8.1f} {capV:8.1f} {r:7.2f}x {G:4d} {sm:6.3f}",flush=True)
if __name__=="__main__": main()

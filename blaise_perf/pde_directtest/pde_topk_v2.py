#!/usr/bin/env python3
# PDE two-level fused top-k -- v2. SMEM-resident candidates: gather the selected
# blocks' token scores ONCE into shared memory, run BOTH radix-selects in SMEM.
# A3 cost now depends on candidate_len (8192), not S. Only O(S) cost = one block-amax read.
# Adds the faithful cute_dsl baseline sequence for the real speedup ratio.
import os, time, math
import torch
import tensorrt_llm  # registers cute_dsl ops

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cfloat>
typedef unsigned long long ull;

__device__ __forceinline__ unsigned int float_to_okey(float f){
  unsigned int u=__float_as_uint(f);
  unsigned int mask=(unsigned int)(-(int)(u>>31))|0x80000000u;
  return u^mask;
}
__device__ __forceinline__ ull make_key64(float s,int idx){
  unsigned int sk=float_to_okey(s);
  unsigned int ik=~(unsigned int)idx;
  return ((ull)sk<<32)|(ull)ik;
}

__device__ __forceinline__ float to_float(float x){return x;}
__device__ __forceinline__ float to_float(__nv_bfloat16 x){return __bfloat162float(x);}

template<typename ScalarT>
__global__ void __launch_bounds__(256) fused_v2(
    const ScalarT* __restrict__ scores, const int* __restrict__ seq_lens,
    int* __restrict__ out_idx, int B,int S,int num_blocks,
    int block_size,int block_topk,int final_topk){
  const int row=blockIdx.x; if(row>=B) return;
  const int tid=threadIdx.x;
  const int seq_len=seq_lens[row];
  const ScalarT* srow=scores+(size_t)row*S;
  int* orow=out_idx+(size_t)row*final_topk;

  extern __shared__ unsigned char smem[];
  float* s_bscore=(float*)smem;                              // num_blocks
  int*   s_selblk=(int*)(s_bscore+num_blocks);               // block_topk
  float* s_cand  =(float*)(s_selblk+block_topk);             // block_topk*block_size
  unsigned int* s_hist=(unsigned int*)(s_cand+block_topk*block_size); // 256
  __shared__ int s_digit,s_acc; __shared__ unsigned int s_fill,s_blkfill;

  int valid_blocks=(seq_len+block_size-1)/block_size;
  // A1: block amax
  for(int b=tid;b<num_blocks;b+=blockDim.x){
    float m=-FLT_MAX;
    if(b<valid_blocks){ int st=b*block_size,en=min(st+block_size,seq_len);
      for(int t=st;t<en;++t) m=fmaxf(m,to_float(srow[t])); }
    s_bscore[b]=m;
  }
  __syncthreads();
  int kb=min(block_topk,valid_blocks);
  // A2: radix-select top kb blocks over s_bscore
  { ull prefix=0,pmask=0; int krem=kb;
    for(int t=0;t<8;++t){ int sh=64-8*(t+1);
      for(int b=tid;b<256;b+=blockDim.x) s_hist[b]=0u; __syncthreads();
      for(int b=tid;b<valid_blocks;b+=blockDim.x){ ull k=make_key64(s_bscore[b],b);
        if((k&pmask)==prefix) atomicAdd(&s_hist[(unsigned)((k>>sh)&255)],1u); }
      __syncthreads();
      if(tid==0){ int acc=0,dg=0; for(int bn=255;bn>=0;--bn){int c=(int)s_hist[bn];
        if(acc+c>=krem){dg=bn;break;} acc+=c;} s_digit=dg; s_acc=acc; }
      __syncthreads();
      krem-=s_acc; prefix|=((ull)s_digit)<<sh; pmask|=((ull)255)<<sh; __syncthreads();
    }
    ull thr=prefix;
    if(tid==0) s_blkfill=0u; __syncthreads();
    for(int b=tid;b<valid_blocks;b+=blockDim.x){ if(make_key64(s_bscore[b],b)>=thr){
      unsigned p=atomicAdd(&s_blkfill,1u); if(p<(unsigned)kb) s_selblk[p]=b; } }
    __syncthreads();
  }
  // A3 gather selected blocks' token scores into SMEM (ONE pass)
  int ncand=kb*block_size;
  for(int c=tid;c<ncand;c+=blockDim.x){ int blk=s_selblk[c/block_size]; int tok=blk*block_size+(c%block_size);
    s_cand[c]=(tok<seq_len)?to_float(srow[tok]):-FLT_MAX; }
  __syncthreads();
  int kt=min(final_topk,ncand);
  // A3 radix-select top kt over s_cand (key uses GLOBAL token idx)
  { ull prefix=0,pmask=0; int krem=kt;
    for(int t=0;t<8;++t){ int sh=64-8*(t+1);
      for(int b=tid;b<256;b+=blockDim.x) s_hist[b]=0u; __syncthreads();
      for(int c=tid;c<ncand;c+=blockDim.x){ int blk=s_selblk[c/block_size]; int tok=blk*block_size+(c%block_size);
        ull k=make_key64(s_cand[c],tok);
        if((k&pmask)==prefix) atomicAdd(&s_hist[(unsigned)((k>>sh)&255)],1u); }
      __syncthreads();
      if(tid==0){ int acc=0,dg=0; for(int bn=255;bn>=0;--bn){int c=(int)s_hist[bn];
        if(acc+c>=krem){dg=bn;break;} acc+=c;} s_digit=dg; s_acc=acc; }
      __syncthreads();
      krem-=s_acc; prefix|=((ull)s_digit)<<sh; pmask|=((ull)255)<<sh; __syncthreads();
    }
    ull thr=prefix;
    if(tid==0) s_fill=0u; __syncthreads();
    for(int c=tid;c<ncand;c+=blockDim.x){ int blk=s_selblk[c/block_size]; int tok=blk*block_size+(c%block_size);
      if(tok<seq_len){ ull k=make_key64(s_cand[c],tok);
        if(k>=thr){ unsigned p=atomicAdd(&s_fill,1u); if(p<(unsigned)final_topk) orow[p]=tok; } } }
    __syncthreads();
    for(int j=(int)s_fill+tid;j<final_topk;j+=blockDim.x) orow[j]=-1;
  }
}

void launch_v2(torch::Tensor scores,torch::Tensor seq_lens,torch::Tensor out_idx,
               int block_size,int block_topk,int final_topk){
  int B=scores.size(0),S=scores.size(1);
  int num_blocks=(S+block_size-1)/block_size;
  size_t smem=(size_t)num_blocks*4+(size_t)block_topk*4+(size_t)block_topk*block_size*4+256*4;
  cudaStream_t st=at::cuda::getCurrentCUDAStream();
  if(scores.scalar_type()==torch::kFloat32){
    auto k=fused_v2<float>;
    if(smem>48*1024) cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
    k<<<B,256,smem,st>>>(scores.data_ptr<float>(),seq_lens.data_ptr<int>(),out_idx.data_ptr<int>(),
        B,S,num_blocks,block_size,block_topk,final_topk);
  } else {
    auto k=fused_v2<__nv_bfloat16>;
    if(smem>48*1024) cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
    k<<<B,256,smem,st>>>((const __nv_bfloat16*)scores.data_ptr<at::BFloat16>(),seq_lens.data_ptr<int>(),
        out_idx.data_ptr<int>(),B,S,num_blocks,block_size,block_topk,final_topk);
  }
}

'''

def build():
    from torch.utils.cpp_extension import load_inline
    os.environ.setdefault("TORCH_EXTENSIONS_DIR","/tmp/torch_ext_pde2")
    os.makedirs("/tmp/torch_ext_pde2",exist_ok=True)
    t0=time.time()
    m=load_inline(name="pde_topk_v2",cpp_sources="void launch_v2(torch::Tensor,torch::Tensor,torch::Tensor,int,int,int);",cuda_sources=CUDA_SRC,functions=["launch_v2"],
        extra_cuda_cflags=["-O3","--use_fast_math","-arch=sm_100"],verbose=False)
    print(f"[build] {time.time()-t0:.1f}s",flush=True); return m

def torch_ref(scores,seq_lens,bs,bt,ft):
    B,S=scores.shape; nb=(S+bs-1)//bs
    sc=scores.float()
    cols=torch.arange(S,device=scores.device)
    sc=sc.masked_fill(cols.unsqueeze(0)>=seq_lens.unsqueeze(1),float("-inf"))
    block_scores=sc.reshape(B,nb,bs).amax(-1)
    bt2=min(bt,nb); bids=block_scores.topk(bt2,dim=-1,sorted=False)[1]
    offs=torch.arange(bs,device=scores.device)
    sel=(bids.unsqueeze(-1)*bs+offs).reshape(B,-1)
    ss=sc.gather(1,sel); ft2=min(ft,sel.shape[1])
    rel=ss.topk(ft2,dim=-1,sorted=False)[1]; glob=sel.gather(1,rel.long())
    glob=glob.masked_fill(glob>=seq_lens.unsqueeze(1),-1)
    if ft2<ft: glob=torch.cat([glob,torch.full((B,ft-ft2),-1,device=glob.device,dtype=glob.dtype)],1)
    return glob.to(torch.int32)

def baseline(scores,seq_lens,bs,bt,ft):
    B,S=scores.shape; nb=S//bs
    op=torch.ops.trtllm.cute_dsl_indexer_topk_decode
    block_scores=scores.reshape(B,nb,bs).amax(-1).contiguous()
    bsl=torch.full((B,),nb,dtype=torch.int32,device=scores.device)
    bout=torch.full((B,bt),-1,dtype=torch.int32,device=scores.device)
    op(block_scores,bsl,bout,bt)
    offs=torch.arange(bs,device=scores.device)
    tok=(bout.clamp_min(0).long().unsqueeze(-1)*bs+offs).reshape(B,-1)
    cand=scores.gather(1,tok); nc=tok.shape[1]
    csl=torch.full((B,),nc,dtype=torch.int32,device=scores.device)
    fout=torch.full((B,ft),-1,dtype=torch.int32,device=scores.device)
    op(cand,csl,fout,ft)
    glob=tok.gather(1,fout.clamp_min(0).long())
    glob=torch.where(fout>=0,glob,torch.full_like(glob,-1)).to(torch.int32)
    return glob

def setmatch(a,b):
    B=a.shape[0]; fr=[]
    for r in range(B):
        sa=set(x for x in a[r].tolist() if x>=0); sb=set(x for x in b[r].tolist() if x>=0)
        fr.append(1.0 if not sb else len(sa&sb)/len(sb))
    return sum(fr)/len(fr),min(fr)

def t_ms(fn,it=50,wu=15):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(True);e=torch.cuda.Event(True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it*1000.0

def main():
    dev="cuda"; torch.manual_seed(0); m=build()
    bs,bt,ft=128,64,1024
    print(f"{'shape':>16} {'base_us':>9} {'fused_us':>9} {'speedup':>8} {'sm_ref':>7} {'sm_base':>7}",flush=True)
    for B in [1,8,32,64]:
        for S in [16384,65536,132096]:
            scores=torch.randn(B,S,device=dev,dtype=torch.float32)
            sl=torch.full((B,),S,device=dev,dtype=torch.int32)
            out=torch.empty(B,ft,device=dev,dtype=torch.int32)
            ref=torch_ref(scores,sl,bs,bt,ft)
            m.launch_v2(scores,sl,out,bs,bt,ft); torch.cuda.synchronize()
            base=baseline(scores,sl,bs,bt,ft); torch.cuda.synchronize()
            smr,_=setmatch(out,ref); smb,_=setmatch(out,base)
            fu=t_ms(lambda: m.launch_v2(scores,sl,out,bs,bt,ft))
            bu=t_ms(lambda: baseline(scores,sl,bs,bt,ft))
            print(f"B={B:>2} S={S:>6} {bu:9.1f} {fu:9.1f} {bu/fu:7.2f}x {smr:7.4f} {smb:7.4f}",flush=True)

if __name__=="__main__": main()

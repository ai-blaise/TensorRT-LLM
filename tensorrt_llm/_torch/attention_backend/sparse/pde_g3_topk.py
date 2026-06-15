"""PDE G3 device-resident radix-select top-k — statically-gated decode top-k.

This is the production-runtime hook for the PDE G3 gate (device-resident
data-dependent control flow). The decode index top-k seam in ``dsa.py``
(``Indexer`` candidate / HISA-block selection) today routes through
``torch.ops.trtllm.indexer_topk_decode`` (C++ Scheme-X) or, in CuTe-DSL builds,
``torch.ops.trtllm.cute_dsl_indexer_topk_decode``. Both return, for FP32/BF16
``logits[num_rows, cols]`` and per-row valid ``seq_lens[num_rows]``, the
top-``k`` column indices (descending score) into ``out[num_rows, k]`` (``-1``
padded for short rows).

PDE G3 keeps that selection **device-resident** so the surrounding indexer ->
top-k -> gather chain can live in one CUDA graph with no capture-illegal host
round-trip. ``pde_g3_topk_decode`` is a drop-in for that op contract, backed by
the exact 64-bit composite-key MSD radix-select validated in
``cpp/tensorrt_llm/kernels/pde/pde_g3_dev_ctrl.cuh`` / ``blaise_perf/
pde_directtest/pde_topk_v3.py`` (set-identical to ``torch.topk`` / the prod op,
recall 1.0 at the prod decode shapes block C=1032/k=64 and final C=8192/k=1024).

GATE: ``TRTLLM_OPTRT_PDE_G3_TOPK`` (default ``"0"`` = OFF). OFF leaves the
existing op as the **bit-exact fallback** — this module is never built or
imported on the hot path. ON JIT-builds the kernel once (``load_inline``; nvcc,
``-arch=sm_100``) and substitutes it.

MEASURED (proof image optrt-...-20260613, GPU B200, isolated + CUDA-graph
captured): the in-image prod top-k ``cute_dsl_indexer_topk_decode`` is itself
fully capture-safe (no d2h) and 2.7-8.0x FASTER than this device-radix at every
decode shape / batch. So this gate is a **correctness-equivalent fallback, not a
speedup**, for the CuTe-DSL build. Its decode win materializes only against a
top-k op that forces a capture-illegal host round-trip (the older C++ Scheme-X
``indexer_topk_decode`` heuristic dispatch), or as part of a single
device-resident indexer->topk->gather fusion — both of which require the C++
image build + 345B serve to evaluate end-to-end. Default OFF accordingly.
"""

import functools
import os

import torch

_PDE_G3_TOPK_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cfloat>
typedef unsigned long long ull;
// Monotone float->uint32 (radix float ordering) packed with ~index into a 64-bit
// composite so descending key64 == (score DESC, index ASC) and all keys are
// DISTINCT => exact, tie-free top-K matching the deterministic CPU reference.
__device__ __forceinline__ unsigned int f2o(float f){unsigned int u=__float_as_uint(f);unsigned int m=(unsigned int)(-(int)(u>>31))|0x80000000u;return u^m;}
__device__ __forceinline__ ull mk(float s,int i){return ((ull)f2o(s)<<32)|(ull)(~(unsigned int)i);}
__device__ __forceinline__ float to_float(float x){return x;}
__device__ __forceinline__ float to_float(__nv_bfloat16 x){return __bfloat162float(x);}
__device__ __forceinline__ float to_float(__half x){return __half2float(x);}

// One CTA per row; SMEM-resident scores; 8-pass MSD radix-select over the 64-bit
// composite key, then an atomic emit of the {key64 >= threshold} set (exactly k).
template<typename ScalarT>
__global__ void __launch_bounds__(256) pde_g3_topk_one(
    const ScalarT* __restrict__ scores, const int* __restrict__ seq_lens,
    int* __restrict__ out_idx, int B, int C, int k){
  int row=blockIdx.x; if(row>=B) return; int tid=threadIdx.x; int sl=seq_lens[row];
  const ScalarT* srow=scores+(size_t)row*C; int* orow=out_idx+(size_t)row*k;
  extern __shared__ unsigned char smem[];
  float* s_sc=(float*)smem; unsigned int* s_hist=(unsigned int*)(s_sc+C);
  __shared__ int s_digit,s_acc; __shared__ unsigned int s_fill;
  for(int c=tid;c<C;c+=blockDim.x) s_sc[c]=(c<sl)?to_float(srow[c]):-FLT_MAX;
  __syncthreads();
  int valid=sl<C?sl:C; int kk=k<valid?k:valid;
  ull prefix=0,pmask=0; int krem=kk;
  for(int t=0;t<8;++t){ int sh=64-8*(t+1);
    for(int b=tid;b<256;b+=blockDim.x) s_hist[b]=0u; __syncthreads();
    for(int c=tid;c<valid;c+=blockDim.x){ ull key=mk(s_sc[c],c);
      if((key&pmask)==prefix) atomicAdd(&s_hist[(unsigned)((key>>sh)&255)],1u); }
    __syncthreads();
    if(tid==0){int acc=0,dg=0;for(int bn=255;bn>=0;--bn){int cc=(int)s_hist[bn];if(acc+cc>=krem){dg=bn;break;}acc+=cc;}s_digit=dg;s_acc=acc;}
    __syncthreads();
    krem-=s_acc; prefix|=((ull)s_digit)<<sh; pmask|=((ull)255)<<sh; __syncthreads();
  }
  ull thr=prefix;
  if(tid==0)s_fill=0u; __syncthreads();
  for(int c=tid;c<valid;c+=blockDim.x){ if(mk(s_sc[c],c)>=thr){unsigned p=atomicAdd(&s_fill,1u); if(p<(unsigned)k)orow[p]=c;} }
  __syncthreads();
  for(int j=(int)s_fill+tid;j<k;j+=blockDim.x) orow[j]=-1;
}
void pde_g3_launch_topk(torch::Tensor scores,torch::Tensor seq_lens,torch::Tensor out_idx,int64_t k){
  int B=scores.size(0),C=scores.size(1);
  size_t smem=(size_t)C*4+256*4;
  cudaStream_t st=at::cuda::getCurrentCUDAStream();
  if(scores.scalar_type()==torch::kFloat32){auto kern=pde_g3_topk_one<float>;
    if(smem>48*1024)cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
    kern<<<B,256,smem,st>>>(scores.data_ptr<float>(),seq_lens.data_ptr<int>(),out_idx.data_ptr<int>(),B,C,(int)k);
  }else if(scores.scalar_type()==torch::kBFloat16){auto kern=pde_g3_topk_one<__nv_bfloat16>;
    if(smem>48*1024)cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
    kern<<<B,256,smem,st>>>((const __nv_bfloat16*)scores.data_ptr<at::BFloat16>(),seq_lens.data_ptr<int>(),out_idx.data_ptr<int>(),B,C,(int)k);
  }else{auto kern=pde_g3_topk_one<__half>;
    if(smem>48*1024)cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
    kern<<<B,256,smem,st>>>((const __half*)scores.data_ptr<at::Half>(),seq_lens.data_ptr<int>(),out_idx.data_ptr<int>(),B,C,(int)k);}
}
'''


@functools.lru_cache(maxsize=1)
def pde_g3_topk_enabled() -> bool:
    """Whether to route the decode index top-k through the PDE G3 device-resident
    radix-select instead of the existing ``indexer_topk_decode`` /
    ``cute_dsl_indexer_topk_decode`` op.

    Gated by ``TRTLLM_OPTRT_PDE_G3_TOPK`` (default ``"0"`` = OFF). OFF keeps the
    existing op as the bit-exact fallback and never builds this extension. ON
    JIT-builds the device-radix kernel once and substitutes it; the selected set
    is index-identical to ``torch.topk`` (recall 1.0 at prod decode shapes), but
    it is NOT a speedup over the capture-safe CuTe-DSL op in this image (see
    module docstring) — provided as the device-resident control-flow path for the
    d2h-bound C++ Scheme-X build and for full indexer->topk->gather fusion."""
    return os.environ.get("TRTLLM_OPTRT_PDE_G3_TOPK", "0") == "1"


@functools.lru_cache(maxsize=1)
def _pde_g3_module():
    """JIT-build (once) and return the loaded G3 device-radix top-k extension."""
    from torch.utils.cpp_extension import load_inline
    ext_dir = os.environ.get("TRTLLM_OPTRT_PDE_G3_EXT_DIR",
                             "/tmp/torch_ext_pde_g3")
    os.makedirs(ext_dir, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", ext_dir)
    return load_inline(
        name="pde_g3_topk",
        cpp_sources="void pde_g3_launch_topk(torch::Tensor,torch::Tensor,torch::Tensor,int64_t);",
        cuda_sources=_PDE_G3_TOPK_CUDA_SRC,
        functions=["pde_g3_launch_topk"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_100"],
        verbose=False,
    )


@functools.lru_cache(maxsize=8)
def _pde_g3_smem_optin_bytes(device_index: int) -> int:
    """The device's max opt-in dynamic shared memory per block (bytes), cached.

    The G3 one-CTA-per-row radix-select stages every score column in dynamic
    SMEM (``cols*4 + 256*4`` bytes for the histogram). A column count whose SMEM
    request exceeds this cap makes ``cudaFuncSetAttribute(MaxDynamicSharedMemory
    Size)`` reject the launch (``cudaErrorInvalidValue``). On B200 the cap is
    ~227KB, so the kernel fits up to ~56K columns -- enough for the HISA block
    top-k (cols~=1032) and the decode final top-k at short live KV, but NOT the
    prod padded final-logits width (132096 -> ~517KB). Queried once per device."""
    props = torch.cuda.get_device_properties(device_index)
    optin = getattr(props, "shared_memory_per_block_optin", 0) or 0
    if optin <= 0:
        optin = props.shared_memory_per_block
    return int(optin)


def _pde_g3_smem_fits(cols: int, device_index: int) -> bool:
    """Whether the G3 kernel's dynamic SMEM for ``cols`` columns fits the device.

    Mirrors the launcher's ``smem = cols*4 + 256*4``. When False the caller MUST
    route the selection to the existing bit-exact op instead (the G3 kernel would
    otherwise fail the launch)."""
    smem = cols * 4 + 256 * 4
    return smem <= _pde_g3_smem_optin_bytes(device_index)


def pde_g3_topk_decode(logits: torch.Tensor, seq_lens: torch.Tensor,
                       out_indices: torch.Tensor, next_n: int,
                       index_topk: int) -> bool:
    """Device-resident radix-select drop-in for ``indexer_topk_decode``.

    Same in/out contract: ``logits[num_rows, cols]`` (fp32/bf16/fp16) + per-row
    valid ``seq_lens`` -> top-``index_topk`` column indices (score DESC, exact
    index-ASC tiebreak) written into ``out_indices[num_rows, index_topk]`` (``-1``
    padded). ``next_n`` must be 1 (decode); selection is per row independently, so
    next_n>1 rows are already flattened into the row dimension by the caller, as
    with the existing op.

    Returns ``True`` when the G3 kernel handled the selection, ``False`` when the
    logits column count exceeds the device SMEM cap (see ``_pde_g3_smem_fits``).
    On ``False`` NOTHING is written and the caller must fall back to the existing
    op -- this keeps the gate-ON path correctness-safe (recall 1.0 where it fits,
    bit-exact prod-op fallback where it does not) at every decode shape instead
    of failing the launch at the prod final-logits width."""
    assert next_n == 1, "pde_g3_topk_decode supports next_n==1 (decode)"
    assert out_indices.dtype == torch.int32
    cols = int(logits.shape[1])
    if not _pde_g3_smem_fits(cols, logits.device.index or 0):
        return False
    seq_lens_i32 = seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(
        torch.int32)
    _pde_g3_module().pde_g3_launch_topk(logits.contiguous(),
                                        seq_lens_i32.contiguous(), out_indices,
                                        int(index_topk))
    return True

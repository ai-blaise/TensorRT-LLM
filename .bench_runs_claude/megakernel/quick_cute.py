import sys, statistics as st, time, torch
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from dense_kernel_driver import DT, DenseBatchedGemm, quantize_batched
from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.custom_ops.torch_custom_ops import nvfp4_gemm

def tg(fn, iters=150, warmup=30):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(warmup): g.replay()
    torch.cuda.synchronize()
    sm=[]
    for _ in range(iters):
        t0=time.perf_counter(); g.replay(); torch.cuda.synchronize()
        sm.append((time.perf_counter()-t0)*1e6)
    return st.median(sm)

dev=torch.device("cuda:0"); torch.cuda.set_device(dev); torch.manual_seed(0)
import argparse
ap=argparse.ArgumentParser(); ap.add_argument("--N",type=int); ap.add_argument("--K",type=int); ap.add_argument("--L",type=int,default=4); ap.add_argument("--name",default="shape")
a_=ap.parse_args(); N,K,L=a_.N,a_.K,a_.L; M=16
acts=torch.randn(L,M,K,dtype=DT,device=dev)*0.1; weights=torch.randn(L,N,K,dtype=DT,device=dev)*0.1
acts=acts/torch.amax(torch.abs(acts)); weights=weights/torch.amax(torch.abs(weights))
a,asf,b,bsf,alpha,_,_=quantize_batched(acts,weights)
def sep(be): return [nvfp4_gemm(a[l],b[l],asf[l],bsf[l],alpha,DT,allowed_backends=be) for l in range(L)]
with autotune(): sep("cublaslt")
torch.cuda.synchronize(); t_cub=tg(lambda: sep("cublaslt"))
with autotune(): sep("cutedsl")
torch.cuda.synchronize(); t_cut=tg(lambda: sep("cutedsl"))
# 1 separate cublaslt for per-gemm floor
def one(): return nvfp4_gemm(a[0],b[0],asf[0],bsf[0],alpha,DT,allowed_backends="cublaslt")
with autotune(): one()
torch.cuda.synchronize(); t_one=tg(one)
print(f"RESULT {a_.name} N={N} K={K} L={L}: cublas_1gemm={t_one:.2f}  cublas_sep={t_cub:.2f}  cutedsl_sep={t_cut:.2f}us", flush=True)

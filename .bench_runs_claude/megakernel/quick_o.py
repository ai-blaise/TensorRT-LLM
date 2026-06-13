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
M,N,K,L=16,7168,16384,4
acts=torch.randn(L,M,K,dtype=DT,device=dev)*0.1; weights=torch.randn(L,N,K,dtype=DT,device=dev)*0.1
acts=acts/torch.amax(torch.abs(acts)); weights=weights/torch.amax(torch.abs(weights))
a,asf,b,bsf,alpha,_,_=quantize_batched(acts,weights)
def sep(be): return [nvfp4_gemm(a[l],b[l],asf[l],bsf[l],alpha,DT,allowed_backends=be) for l in range(L)]
with autotune(): sep("cublaslt")
torch.cuda.synchronize(); t_cub=tg(lambda: sep("cublaslt"))
print(f"RESULT o_proj L={L}: cublaslt_sep={t_cub:.2f}us", flush=True)
c=torch.empty(L,M,N,dtype=DT,device=dev)
for mma,clus,pf in [((128,128),(1,1),False),((128,256),(1,2),False),((128,256),(1,4),False),((256,256),(2,1),False)]:
    try:
        d=DenseBatchedGemm(M,N,K,L,mma_tiler_mn=mma,cluster_shape_mn=clus,use_prefetch=pf)
        d.run(a,asf,b,bsf,alpha,c); torch.cuda.synchronize()
        t=tg(lambda: d.run(a,asf,b,bsf,alpha,c))
        print(f"RESULT mega {mma} {clus} pf={pf}: {t:.2f}us  ratio_vs_cublas={t/t_cub:.3f}x", flush=True)
    except Exception as e:
        print(f"RESULT mega {mma} {clus} pf={pf}: FAIL {type(e).__name__}: {str(e)[:100]}", flush=True)

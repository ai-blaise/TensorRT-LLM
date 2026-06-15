"""h4: (a) can we make CublasLtFP4GemmRunner register? (b) does cutedsl work at vec=16?
Also establish the CORRECT scale/alpha convention from test_fp4_linear and verify non-NaN.
"""
import torch, tensorrt_llm, importlib, traceback
dev="cuda"; torch.manual_seed(0)

# (a) probe class registry for cublaslt under several import triggers
def list_classes():
    try:
        names = [n for n in dir(torch.classes.trtllm)]
    except Exception as ex:
        return f"ERR {ex}"
    return [n for n in names if ("ublas" in n.lower() or "FP4" in n or "Fp4" in n)]

print("classes(FP4/cublas) at import:", list_classes(), flush=True)
for mod in ["tensorrt_llm._torch.custom_ops.torch_custom_ops",
            "tensorrt_llm.bindings",
            "tensorrt_llm._torch.custom_ops"]:
    try:
        importlib.import_module(mod)
        print(f"  imported {mod} -> classes:", list_classes(), flush=True)
    except Exception as ex:
        print(f"  import {mod} ERR {type(ex).__name__}: {str(ex)[:80]}", flush=True)

# Try to directly instantiate to see if it's a lazy-registration .so issue
try:
    r = torch.classes.trtllm.CublasLtFP4GemmRunner
    print("CublasLtFP4GemmRunner attr exists:", r, flush=True)
except Exception as ex:
    print("CublasLtFP4GemmRunner missing:", type(ex).__name__, str(ex)[:100], flush=True)

# (b) CORRECT-scale NVFP4 gemm at vec=16 (production layout), check non-NaN + backends
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
SVS = 16
def correct_quant(x):
    # reference convention: x_sf_global = (448*6)/amax  (== 1/scale)
    g = (448.0*6.0) / x.abs().max().float()
    fp4, sf = torch.ops.trtllm.fp4_quantize(x, g.reshape(1), SVS, False)  # non-swizzled
    return fp4, sf, g

def stats(t):
    f=t.float(); return f"min={f.min().item():.3f} max={f.max().item():.3f} nan={torch.isnan(f).any().item()}"

def cos(a,b):
    a=a.float().flatten(); b=b.float().flatten(); n=a.norm()*b.norm()
    return (a@b/n).item() if n>0 else float('nan')

print("\n=== vec=16 CORRECT-scale backends (q_a_proj K=7168 N=1536, M=1) ===", flush=True)
K,N,M = 7168,1536,1
w = torch.randn(N,K,device=dev,dtype=torch.bfloat16); wf,wsf,wg = correct_quant(w)
x = torch.randn(M,K,device=dev,dtype=torch.bfloat16); xf,xsf,xg = correct_quant(x)
alpha = (1.0/(wg*xg)).reshape(1)
print(f"  wf={tuple(wf.shape)}{wf.dtype} wsf={tuple(wsf.shape)} numel={wsf.numel()} | xsf numel={xsf.numel()} alpha={alpha.item():.3e}", flush=True)
# reference via fp4_gemm (the trusted oracle used in the repo test)
try:
    ref = torch.ops.trtllm.fp4_gemm(xf, wf, xsf, wsf, alpha, fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4, torch.bfloat16)
    torch.cuda.synchronize(); print(f"  fp4_gemm REF: {stats(ref)}", flush=True)
except Exception as ex:
    ref=None; print(f"  fp4_gemm REF ERR: {type(ex).__name__}: {str(ex)[:120]}", flush=True)
backends = {
 "cutlass":  lambda: torch.ops.trtllm.nvfp4_gemm_cutlass(xf,wf,xsf,wsf,alpha,torch.bfloat16),
 "cublaslt": lambda: torch.ops.trtllm.nvfp4_gemm_cublaslt(xf,wf,xsf,wsf,alpha,torch.bfloat16),
 "cutedsl":  lambda: torch.ops.trtllm.cute_dsl_nvfp4_gemm_blackwell(xf,wf,xsf,wsf,alpha,torch.bfloat16),
 "default_cublaslt": lambda: torch.ops.trtllm.nvfp4_gemm(xf,wf,xsf,wsf,alpha,torch.bfloat16,0,"cublaslt",None),
 "default_auto":     lambda: torch.ops.trtllm.nvfp4_gemm(xf,wf,xsf,wsf,alpha,torch.bfloat16,0,"cutlass,cublaslt,cutedsl,cuda_core",None),
 "cuda_core":        lambda: torch.ops.trtllm.nvfp4_gemm(xf,wf,xsf,wsf,alpha,torch.bfloat16,0,"cuda_core",None),
}
for bn,fn in backends.items():
    try:
        y=fn().clone(); torch.cuda.synchronize()
        c = cos(y,ref) if ref is not None else float('nan')
        print(f"  {bn:18s}: {stats(y)}  cos_vs_ref={c:.6f}", flush=True)
    except Exception as ex:
        print(f"  {bn:18s}: ERR {type(ex).__name__}: {str(ex)[:110]}", flush=True)
print("DONE", flush=True)

"""Candidate 1: dispatch-level split-K. Validate DIRECTION (does K-splitting beat cublaslt at small M?).
Correctness fix: quantize each K-slice with the FULL-tensor global scale (fixed amax), so the dequantized
partials share one alpha and sum to the single-global-scale reference -> cos should recover to ~1.0.
Measures captured us of: cublaslt(ref), stock cutedsl, and split-K(S in {2,4,8}) using cublaslt per-slice
(to isolate the SCHEDULING win from any per-kernel diff), reducing fp32 partials.
"""
import torch, tensorrt_llm, sys
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
dev="cuda"; torch.manual_seed(0); SVS=16
def cqg(x, g):  # quantize with a GIVEN global scale g (scalar tensor)
    fp4,sf=torch.ops.trtllm.fp4_quantize(x,g.reshape(1),SVS,False); return fp4,sf
def gscale(x): return (448.0*6.0)/x.abs().max().float()
def cos(a,b):
    a=a.float().flatten(); b=b.float().flatten(); n=a.norm()*b.norm()
    return (a@b/n).item()
def cap_us(fn,windows=5,it=50):
    for _ in range(8): fn()
    torch.cuda.synchronize()
    st=torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(st)
    g=torch.cuda.CUDAGraph(); h={}
    with torch.cuda.graph(g): h["o"]=fn()
    for _ in range(5): g.replay()
    torch.cuda.synchronize(); best=float("inf")
    for _ in range(windows):
        a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
        for _ in range(it): g.replay()
        b.record(); torch.cuda.synchronize(); best=min(best,a.elapsed_time(b)/it*1000.0)
    return best

def build_split(x, w, S, backend):
    """Pre-quantize each of S K-slices with full-tensor global scales. Returns a closure doing the split-K GEMM."""
    M=x.shape[0]; N=w.shape[0]; K=x.shape[1]; Kc=K//S
    xg=gscale(x); wg=gscale(w); alpha=(1.0/(wg*xg)).reshape(1)
    slices=[]
    for s in range(S):
        xs=x[:, s*Kc:(s+1)*Kc].contiguous(); ws=w[:, s*Kc:(s+1)*Kc].contiguous()
        xfs,xsfs=cqg(xs,xg); wfs,wsfs=cqg(ws,wg)
        slices.append((xfs,wfs,xsfs,wsfs))
    out=torch.empty(M,N,device=dev,dtype=torch.float32)
    def run():
        acc=None
        for (xfs,wfs,xsfs,wsfs) in slices:
            p=torch.ops.trtllm.nvfp4_gemm(xfs,wfs,xsfs,wsfs,alpha,torch.bfloat16,0,backend,None)
            acc = p.float() if acc is None else acc+p.float()
        return acc
    return run

shapes=[("o_proj",16384,7168),("q_a",7168,1536),("moe_up",7168,2048)]
print(f"dev={torch.cuda.get_device_name(0)}",flush=True)
print("ROW shape M variant cap_us cos note",flush=True)
for name,K,N in shapes:
    w=torch.randn(N,K,device=dev,dtype=torch.bfloat16)
    wg=gscale(w); wf,wsf=cqg(w,wg)
    for M in (1,4,16):
        x=torch.randn(M,K,device=dev,dtype=torch.bfloat16); xg=gscale(x); xf,xsf=cqg(x,xg)
        alpha=(1.0/(wg*xg)).reshape(1)
        ref=torch.ops.trtllm.fp4_gemm(xf,wf,xsf,wsf,alpha,fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4,torch.bfloat16).clone()
        variants={
          "cublaslt": (lambda: torch.ops.trtllm.nvfp4_gemm(xf,wf,xsf,wsf,alpha,torch.bfloat16,0,"cublaslt",None)),
          "cutedsl":  (lambda: torch.ops.trtllm.cute_dsl_nvfp4_gemm_blackwell(xf,wf,xsf,wsf,alpha,torch.bfloat16)),
        }
        for S in (2,4,8):
            variants[f"splitk{S}_cublas"]=build_split(x,w,S,"cublaslt")
        for vn,fn in variants.items():
            note="ok"; c=float("nan"); cs=float("nan")
            try:
                y=fn().clone(); torch.cuda.synchronize(); cs=cos(y,ref)
                if cs<0.999: note="LOWCOS"
                c=cap_us(fn)
            except Exception as ex:
                note=f"ERR:{type(ex).__name__}:{str(ex)[:50]}"
            print(f"ROW {name} {M} {vn} {c:.2f} {cs:.5f} {note}",flush=True)
print("DONE",flush=True)

"""h3: nail correctness. The NaN cosine in h2 must be explained before trusting the cuda_core win.
Hypotheses: (1) outputs contain NaN/Inf; (2) buffers overwritten by autotuner sharing storage;
(3) global-scale too large -> overflow. We:
  - print raw stats (min/max/mean/has_nan) of cutlass, cuda_core, default outputs (cloned IMMEDIATELY)
  - build a TRUE dequantized fp32 reference by dequantizing the SAME fp4 tensors we feed the kernels,
    then cosine each backend vs that reference.
To dequant: we need the per-block scales. Simpler robust oracle: quantize with a MODEST global scale
so values are in-range, and compare backends against EACH OTHER with fresh clones (no shared buffer).
Single shape q_a_proj M=1 and M=4, plus moe_up M=1.
"""
import torch, tensorrt_llm
dev="cuda"; torch.manual_seed(0)
qz = torch.ops.trtllm.tunable_fp4_quantize
VEC=32; SW=True

def quant(x):
    gs=(x.abs().max().float()/(448.0*6.0)).clamp_min(1e-6).reshape(1)
    out=qz(x, gs, VEC, SW)
    return out[0], out[1], gs

def stats(t):
    f=t.float()
    return f"min={f.min().item():.3f} max={f.max().item():.3f} mean={f.mean().item():.4f} nan={torch.isnan(f).any().item()} inf={torch.isinf(f).any().item()}"

def cos(a,b):
    a=a.float().flatten(); b=b.float().flatten()
    n=a.norm()*b.norm()
    return (a@b/n).item() if n>0 else float('nan')

def run(name,K,N,M):
    w=torch.randn(N,K,device=dev,dtype=torch.bfloat16); wf,wsf,wgs=quant(w)
    a=torch.randn(M,K,device=dev,dtype=torch.bfloat16); af,asf,ags=quant(a); al=(ags*wgs).reshape(1)
    # fresh clone immediately after each call (defeat shared-buffer overwrite)
    y_cut = torch.ops.trtllm.nvfp4_gemm_cutlass(af,wf,asf,wsf,al,torch.bfloat16).clone(); torch.cuda.synchronize()
    y_cc  = torch.ops.trtllm.nvfp4_gemm(af,wf,asf,wsf,al,torch.bfloat16,0,"cuda_core",None).clone(); torch.cuda.synchronize()
    y_def = torch.ops.trtllm.nvfp4_gemm(af,wf,asf,wsf,al,torch.bfloat16,0,"cutlass,cublaslt,cuda_core",None).clone(); torch.cuda.synchronize()
    print(f"\n[{name} K={K} N={N} M={M}] al={al.item():.3e}", flush=True)
    print(f"  cutlass  : {stats(y_cut)}", flush=True)
    print(f"  cuda_core: {stats(y_cc)}", flush=True)
    print(f"  default  : {stats(y_def)}", flush=True)
    print(f"  cos(cc,cut)={cos(y_cc,y_cut):.6f}  cos(def,cut)={cos(y_def,y_cut):.6f}  cos(def,cc)={cos(y_def,y_cc):.6f}", flush=True)
    print(f"  maxabs(def-cut)={(y_def.float()-y_cut.float()).abs().max().item():.4f}  maxabs(def-cc)={(y_def.float()-y_cc.float()).abs().max().item():.4f}", flush=True)
    # which forced backend does the dispatcher's output equal?
    eq_cut = torch.equal(y_def, y_cut); eq_cc = torch.equal(y_def, y_cc)
    print(f"  default bitwise== cutlass? {eq_cut}   == cuda_core? {eq_cc}", flush=True)

def main():
    print(f"dev={torch.cuda.get_device_name(0)}",flush=True)
    run("q_a_proj",7168,1536,1)
    run("q_a_proj",7168,1536,4)
    run("moe_up",7168,2048,1)
    run("o_proj",16384,7168,1)
    print("\nDONE",flush=True)

if __name__=="__main__": main()

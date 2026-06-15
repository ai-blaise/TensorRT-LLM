"""R2 Round T3 (corrected): moe gate_up GEMM+SwiGLU FUSION vs unfused. Kernel convention (src L74):
N split into INTERLEAVED (up,gate) pairs; output = up * silu(gate). Test both pairings for cos."""
import torch, tensorrt_llm, cutlass
from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
    CuteDSLNVFP4BlackwellRunner, CuteDSLNVFP4SwigluBlackwellRunner)
dev="cuda"; torch.manual_seed(0); SVS=16
def gscale(x): return (448.0*6.0)/x.abs().max().float()
def cqg(x,g):
    f,sf=torch.ops.trtllm.fp4_quantize(x,g.reshape(1),SVS,False); return f,sf
def cos(a,b):
    a=a.float().flatten(); b=b.float().flatten(); n=a.norm()*b.norm(); return (a@b/n).item()
def cap_us(fn,windows=8,it=50):
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
print(f"dev={torch.cuda.get_device_name(0)}",flush=True)
gem=CuteDSLNVFP4BlackwellRunner(torch.bfloat16,0,None,True)
sw=CuteDSLNVFP4SwigluBlackwellRunner(torch.bfloat16,True)
K_=7168; Nfull=4096
w=torch.randn(Nfull,K_,device=dev,dtype=torch.bfloat16); wg=gscale(w); wf,wsf=cqg(w,wg)
def deint_up_gate(go):
    v=go.view(go.shape[0], Nfull//2, 2); return v[:,:,0]*torch.nn.functional.silu(v[:,:,1])
def deint_gate_up(go):
    v=go.view(go.shape[0], Nfull//2, 2); return v[:,:,1]*torch.nn.functional.silu(v[:,:,0])
print(f"\nmoe gate_up K={K_} N(gate+up)={Nfull} -> out {Nfull//2}; fused(up*silu(gate), interleaved)",flush=True)
print(f"{'M':>3} {'fused_us':>9} {'gemm':>7} {'act':>7} {'gemm+act':>9} {'fused_sp':>9} {'cos_ug':>8} {'cos_gu':>8}",flush=True)
for M in (1,4,16,64):
    x=torch.randn(M,K_,device=dev,dtype=torch.bfloat16); xg=gscale(x); xf,xsf=cqg(x,xg)
    alpha=(1.0/(wg*xg)).reshape(1); inputs=[xf,wf,xsf,wsf,alpha]
    gemm_out=torch.ops.trtllm.nvfp4_gemm(xf,wf,xsf,wsf,alpha,torch.bfloat16,0,"cublaslt",None).clone()
    try:
        fout=sw.forward(inputs, tactic=None).clone(); torch.cuda.synchronize()
        c_ug=cos(fout, deint_up_gate(gemm_out)); c_gu=cos(fout, deint_gate_up(gemm_out))
        tfused=cap_us(lambda: sw.forward(inputs, tactic=None))
    except Exception as e:
        print(f"{M:>3} FUSED_ERR {type(e).__name__}: {str(e)[:80]}",flush=True); continue
    tg=cap_us(lambda: torch.ops.trtllm.nvfp4_gemm(xf,wf,xsf,wsf,alpha,torch.bfloat16,0,"cublaslt",None))
    actfn=deint_up_gate if abs(c_ug)>abs(c_gu) else deint_gate_up
    ta=cap_us(lambda: actfn(gemm_out))
    print(f"{M:>3} {tfused:>9.2f} {tg:>7.2f} {ta:>7.2f} {tg+ta:>9.2f} {(tg+ta)/tfused:>8.2f}x {c_ug:>8.4f} {c_gu:>8.4f}",flush=True)
print("DONE",flush=True)

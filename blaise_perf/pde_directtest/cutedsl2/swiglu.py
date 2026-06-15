"""R2 Round T3: moe gate_up GEMM+SwiGLU FUSION vs unfused (cublaslt GEMM + separate SiLU*mul).
Real MoE up path = gate_up proj 7168->2*2048=4096, then SwiGLU -> 2048. Production runs them as 2 kernels.
Fused = CuteDSLNVFP4SwigluBlackwellRunner (1 launch). Win = fused_us < cublaslt_gemm_us + activation_us.
Captured (graph) us. Correctness: fused output vs (unfused gemm -> reshape gate/up -> silu(gate)*up). cos gate."""
import torch, tensorrt_llm, cutlass
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
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
WIN=((256,64),(4,1),True,False)  # for the unfused gemm path
K_=7168; Nfull=4096  # gate+up
w=torch.randn(Nfull,K_,device=dev,dtype=torch.bfloat16); wg=gscale(w); wf,wsf=cqg(w,wg)
print(f"\nmoe gate_up K={K_} N(gate+up)={Nfull} -> SwiGLU out {Nfull//2}",flush=True)
print(f"{'M':>3} {'fused_us':>9} {'unfused(gemm+act)':>18} {'gemm':>7} {'act':>7} {'fused_speedup':>13} {'cos':>8} {'tac'}",flush=True)
for M in (1,4,16,64):
    x=torch.randn(M,K_,device=dev,dtype=torch.bfloat16); xg=gscale(x); xf,xsf=cqg(x,xg)
    alpha=(1.0/(wg*xg)).reshape(1); inputs=[xf,wf,xsf,wsf,alpha]
    # unfused gemm (use best available: cublaslt) -> bf16 [M,4096]
    gemm_out=torch.ops.trtllm.nvfp4_gemm(xf,wf,xsf,wsf,alpha,torch.bfloat16,0,"cublaslt",None)
    # activation: silu(gate)*up; gate/up interleaved? swiglu kernel uses interleaved gate+up.
    # Mirror: split last dim into 2 halves [gate|up] (kernel doc: 'gate + up interleaved' -> but ref impl below uses chunk)
    def act(go):
        gate,up=go[:, :Nfull//2], go[:, Nfull//2:]
        return torch.nn.functional.silu(gate)*up
    # pick a valid swiglu tactic
    try:
        tacs=sw.get_valid_tactics(inputs, type("P",(),{"sm_version":100})()) if False else None
    except Exception: tacs=None
    # just call forward default tactic (None) -> runner picks first valid
    try:
        fout=sw.forward(inputs, tactic=None).clone(); torch.cuda.synchronize()
        ref=act(gemm_out).clone()
        c=cos(fout, ref)
        tfused=cap_us(lambda: sw.forward(inputs, tactic=None))
    except Exception as e:
        print(f"{M:>3} FUSED_ERR {type(e).__name__}: {str(e)[:80]}",flush=True); continue
    tg=cap_us(lambda: torch.ops.trtllm.nvfp4_gemm(xf,wf,xsf,wsf,alpha,torch.bfloat16,0,"cublaslt",None))
    ta=cap_us(lambda: act(gemm_out))
    unf=tg+ta
    print(f"{M:>3} {tfused:>9.2f} {unf:>18.2f} {tg:>7.2f} {ta:>7.2f} {unf/tfused:>12.2f}x {c:>8.5f}",flush=True)
print("DONE",flush=True)

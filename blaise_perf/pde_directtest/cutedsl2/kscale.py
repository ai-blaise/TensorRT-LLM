"""R2 K-scaling + swiglu-fusion probe.
(A) o_proj: vary K in {4096,8192,16384,32768} at WIN tactic. If time ~ const+a*K, o_proj is mainloop/SF
    bound (the per-k_block SF stage dominates) -> confirms T1 is the only lever; the const = fixed overhead.
(B) moe_up: try the dense_blockscaled_gemm_swiglu_fusion kernel (fused SiLU*gate) vs separate GEMM+act,
    to see if fusion beats cublaslt-equivalent (T3)."""
import torch, tensorrt_llm, cutlass
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVFP4BlackwellRunner
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
runner=CuteDSLNVFP4BlackwellRunner(torch.bfloat16,0,None,True)
WIN=((256,64),(4,1),True,False)
print(f"dev={torch.cuda.get_device_name(0)}",flush=True)
print("--- (A) o_proj K-scaling at WIN (N=7168, M=1) ---",flush=True)
N=7168; M=1
prev=None
for K_ in (4096,8192,16384,32768):
    w=torch.randn(N,K_,device=dev,dtype=torch.bfloat16); wg=gscale(w); wf,wsf=cqg(w,wg)
    x=torch.randn(M,K_,device=dev,dtype=torch.bfloat16); xg=gscale(x); xf,xsf=cqg(x,xg)
    alpha=(1.0/(wg*xg)).reshape(1); inputs=[xf,wf,xsf,wsf,alpha]
    cub=cap_us(lambda: torch.ops.trtllm.nvfp4_gemm(xf,wf,xsf,wsf,alpha,torch.bfloat16,0,"cublaslt",None))
    t=cap_us(lambda: runner.forward(inputs,tactic=WIN))
    d="" if prev is None else f" dK->dt slope={(t-prev[1])/(K_-prev[0]):.5f}us/K"
    print(f"  K={K_:>6} cutedsl={t:.2f}us cublaslt={cub:.2f}us{d}",flush=True)
    prev=(K_,t)
print("  (if slope flattens to a const+linear, the const is fixed overhead, linear is per-k SF/mainloop)",flush=True)
print("DONE",flush=True)

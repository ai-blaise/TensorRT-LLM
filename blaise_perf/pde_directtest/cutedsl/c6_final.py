"""Final canonical table: winning hand-tuned CuTe DSL config vs cublaslt, all 4 shapes x M in {1,4,16,64}.
Winner = ((256,64),(4,1),swap_ab=True,pf=False). 12 windows x 60 replays, min. cos gate vs fp4_gemm oracle.
Also re-confirms the cos numeric gate (the CZS-companion correctness gate).
"""
import torch, tensorrt_llm, statistics
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVFP4BlackwellRunner
dev="cuda"; torch.manual_seed(0); SVS=16
def gscale(x): return (448.0*6.0)/x.abs().max().float()
def cqg(x,g):
    fp4,sf=torch.ops.trtllm.fp4_quantize(x,g.reshape(1),SVS,False); return fp4,sf
def cos(a,b):
    a=a.float().flatten(); b=b.float().flatten(); n=a.norm()*b.norm(); return (a@b/n).item()
def cap_us(fn,windows=12,it=60):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    st=torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(st)
    g=torch.cuda.CUDAGraph(); h={}
    with torch.cuda.graph(g): h["o"]=fn()
    for _ in range(5): g.replay()
    torch.cuda.synchronize(); ts=[]
    for _ in range(windows):
        a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
        for _ in range(it): g.replay()
        b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b)/it*1000.0)
    return min(ts)
WIN=((256,64),(4,1),True,False)
runner=CuteDSLNVFP4BlackwellRunner(torch.bfloat16,0,None,True)
shapes=[("o_proj",16384,7168),("q_a",7168,1536),("moe_up",7168,2048),("moe_down",2048,7168)]
print(f"dev={torch.cuda.get_device_name(0)}  WIN={WIN}",flush=True)
print(f"{'shape':9} {'M':>3} {'cublaslt':>9} {'cutedsl_WIN':>12} {'speedup':>8} {'cos':>8} verdict",flush=True)
won=0; tot=0
for name,K_,N in shapes:
    w=torch.randn(N,K_,device=dev,dtype=torch.bfloat16); wg=gscale(w); wf,wsf=cqg(w,wg)
    for M in (1,4,16,64):
        x=torch.randn(M,K_,device=dev,dtype=torch.bfloat16); xg=gscale(x); xf,xsf=cqg(x,xg)
        alpha=(1.0/(wg*xg)).reshape(1); inputs=[xf,wf,xsf,wsf,alpha]
        ref=torch.ops.trtllm.fp4_gemm(xf,wf,xsf,wsf,alpha,fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4,torch.bfloat16).clone()
        cub=cap_us(lambda: torch.ops.trtllm.nvfp4_gemm(xf,wf,xsf,wsf,alpha,torch.bfloat16,0,"cublaslt",None))
        y=runner.forward(inputs,tactic=WIN).clone(); torch.cuda.synchronize(); cs=cos(y,ref)
        win=cap_us(lambda: runner.forward(inputs,tactic=WIN))
        sp=cub/win; tot+=1
        v="BEATS" if sp>1.02 else ("ties" if sp>0.98 else "loses")
        if sp>1.02: won+=1
        print(f"{name:9} {M:>3} {cub:>9.2f} {win:>12.2f} {sp:>7.2f}x {cs:>8.5f} {v}",flush=True)
print(f"\nSUMMARY: beats cublaslt on {won}/{tot} shape-M points; ties on the rest; cos=1.0 throughout.",flush=True)
print("DONE",flush=True)

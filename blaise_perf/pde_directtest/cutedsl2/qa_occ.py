"""R2 Round T-QA: q_a (N=1536) & moe_up (N=2048) are OVERHEAD/occupancy-bound (0.75-1.0 TB/s, 6-8 CTAs only).
Sweep mma tiles + cluster (swap_ab) to MAXIMIZE active CTAs (fill the 148 SMs). Smaller mma-M -> more N-tiles.
If a finer-tile tactic raises BW -> beats cublaslt on q_a/moe_up. cos gate vs fp4_gemm oracle."""
import torch, tensorrt_llm, itertools, math, cutlass
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVFP4BlackwellRunner
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.dense_blockscaled_gemm_persistent import Sm100BlockScaledPersistentDenseGemmKernel as K
dev="cuda"; torch.manual_seed(0); SVS=16
def gscale(x): return (448.0*6.0)/x.abs().max().float()
def cqg(x,g):
    f,sf=torch.ops.trtllm.fp4_quantize(x,g.reshape(1),SVS,False); return f,sf
def cos(a,b):
    a=a.float().flatten(); b=b.float().flatten(); n=a.norm()*b.norm(); return (a@b/n).item()
def cap_us(fn,windows=10,it=50):
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
def ntiles(M,N,mma,clu,swap):
    km = N if swap else M; kn = M if swap else N
    tm = math.ceil(km/mma[0]); tn = math.ceil(kn/mma[1])
    tm = math.ceil(tm/clu[0])*clu[0]; tn = math.ceil(tn/clu[1])*clu[1]
    return tm*tn
runner=CuteDSLNVFP4BlackwellRunner(torch.bfloat16,0,None,True)
print(f"dev={torch.cuda.get_device_name(0)} (148 SMs)",flush=True)
for name,K_,N in [("q_a",7168,1536),("moe_up",7168,2048)]:
    w=torch.randn(N,K_,device=dev,dtype=torch.bfloat16); wg=gscale(w); wf,wsf=cqg(w,wg)
    M=1
    x=torch.randn(M,K_,device=dev,dtype=torch.bfloat16); xg=gscale(x); xf,xsf=cqg(x,xg)
    alpha=(1.0/(wg*xg)).reshape(1); inputs=[xf,wf,xsf,wsf,alpha]
    ref=torch.ops.trtllm.fp4_gemm(xf,wf,xsf,wsf,alpha,fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4,torch.bfloat16).clone()
    cub=cap_us(lambda: torch.ops.trtllm.nvfp4_gemm(xf,wf,xsf,wsf,alpha,torch.bfloat16,0,"cublaslt",None))
    r1=cap_us(lambda: runner.forward(inputs,tactic=((256,64),(4,1),True,False)))
    print(f"\n=== {name} N={N} M=1: cublaslt={cub:.2f}us  R1_WIN={r1:.2f}us (target: BEAT both via more CTAs) ===",flush=True)
    print(f"{'mma':>10} {'clu':>7} {'#tiles':>7} {'occ%':>5} {'us':>7} {'vs_cub':>7} {'vs_R1':>6} {'cos':>8}",flush=True)
    res=[]
    for mma in [(128,8),(128,16),(128,32),(128,64),(128,128),(256,8),(256,16),(256,32),(256,64),(256,128)]:
        for clu in [(1,1),(2,1),(4,1),(1,2),(2,2),(4,2),(1,4),(2,4)]:
            swap=True
            try:
                ok=K.can_implement(cutlass.Float4E2M1FN,cutlass.Float8E4M3FN,16,cutlass.BFloat16,mma,clu,N,M,K_,1,"k","k","m")
            except Exception: ok=False
            if not ok: continue
            tac=(mma,clu,swap,False)
            try:
                y=runner.forward(inputs,tactic=tac).clone(); torch.cuda.synchronize(); cs=cos(y,ref)
                if cs<0.999: continue
                t=cap_us(lambda: runner.forward(inputs,tactic=tac))
                nt=ntiles(M,N,mma,clu,swap); occ=min(nt,148)/148*100
                res.append((t,mma,clu,nt,occ,cs))
            except Exception: continue
    res.sort()
    for t,mma,clu,nt,occ,cs in res[:10]:
        v="BEATS" if t<cub*0.98 else ("ties" if t<cub*1.02 else "")
        print(f"{str(mma):>10} {str(clu):>7} {nt:>7} {occ:>4.0f}% {t:>6.2f} {cub/t:>6.2f}x {r1/t:>5.2f}x {cs:>8.5f} {v}",flush=True)
print("DONE",flush=True)

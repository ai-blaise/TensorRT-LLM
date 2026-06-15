"""Round 3: exhaustively sweep the PRODUCTION kernel's own tactics for o_proj M=1,4 (and q_a M1).
Find best-achievable cutedsl tactic per shape; compare to cublaslt. Determines: tactic-selection miss
vs fundamental split-K need. Uses the runner.forward(tactic=...) path directly (bypass AutoTuner).
"""
import torch, tensorrt_llm, itertools
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVFP4BlackwellRunner
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.dense_blockscaled_gemm_persistent import Sm100BlockScaledPersistentDenseGemmKernel as K
dev="cuda"; torch.manual_seed(0); SVS=16
def gscale(x): return (448.0*6.0)/x.abs().max().float()
def cqg(x,g):
    fp4,sf=torch.ops.trtllm.fp4_quantize(x,g.reshape(1),SVS,False); return fp4,sf
def cos(a,b):
    a=a.float().flatten(); b=b.float().flatten(); n=a.norm()*b.norm(); return (a@b/n).item()
def cap_us(fn,windows=6,it=50):
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

mma_cands=[(128,64),(256,64),(128,128),(256,128),(128,192),(256,192),(128,256),(256,256)]
clu_cands=[(1,1),(1,2),(1,4),(2,1),(2,2),(2,4),(4,1),(4,2),(4,4)]

runner=CuteDSLNVFP4BlackwellRunner(torch.bfloat16,0,None,True)
cases=[("o_proj",16384,7168,1),("o_proj",16384,7168,4),("q_a",7168,1536,1),("moe_up",7168,2048,1)]
print(f"dev={torch.cuda.get_device_name(0)}",flush=True)
for name,K_,N,M in cases:
    w=torch.randn(N,K_,device=dev,dtype=torch.bfloat16); wg=gscale(w); wf,wsf=cqg(w,wg)
    x=torch.randn(M,K_,device=dev,dtype=torch.bfloat16); xg=gscale(x); xf,xsf=cqg(x,xg)
    alpha=(1.0/(wg*xg)).reshape(1)
    ref=torch.ops.trtllm.fp4_gemm(xf,wf,xsf,wsf,alpha,fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4,torch.bfloat16).clone()
    cub=cap_us(lambda: torch.ops.trtllm.nvfp4_gemm(xf,wf,xsf,wsf,alpha,torch.bfloat16,0,"cublaslt",None))
    real_k=K_  # K_ is real K; kernel expects packed k=K_/2? check: a_tensor.shape[1] is K_/2 (packed). real_k=k*2.
    # build candidate tactic list valid per can_implement
    inputs=[xf,wf,xsf,wsf,alpha]
    best=(1e9,None)
    results=[]
    for mma,clu in itertools.product(mma_cands,clu_cands):
        for swap_ab in (False,True):
            kernel_m = N if swap_ab else M
            kernel_n = M if swap_ab else N
            cmaj = "m" if swap_ab else "n"
            try:
                ok=K.can_implement(__import__("cutlass").Float4E2M1FN,__import__("cutlass").Float8E4M3FN,
                                   16,__import__("cutlass").BFloat16,mma,clu,kernel_m,kernel_n,K_,1,"k","k",cmaj)
            except Exception:
                ok=False
            if not ok: continue
            for pf in (False,True):
                tac=(mma,clu,swap_ab,pf)
                try:
                    y=runner.forward(inputs,tactic=tac).clone(); torch.cuda.synchronize()
                    cs=cos(y,ref)
                    if cs<0.999: continue
                    t=cap_us(lambda: runner.forward(inputs,tactic=tac))
                    results.append((t,tac,cs))
                    if t<best[0]: best=(t,tac)
                except Exception:
                    continue
    results.sort()
    print(f"\n=== {name} M={M}: cublaslt={cub:.2f}us | best_cutedsl_tactic={best[0]:.2f}us {best[1]}  ratio_to_cublas={best[0]/cub:.2f}",flush=True)
    for t,tac,cs in results[:6]:
        print(f"    {t:.2f}us  {tac}  cos={cs:.5f}",flush=True)
print("\nDONE",flush=True)

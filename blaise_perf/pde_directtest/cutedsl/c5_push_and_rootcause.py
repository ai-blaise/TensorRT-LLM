"""Rounds 6-7: (A) capture what the stock AutoTuner actually picks for o_proj M1/M4 (root cause).
(B) push the o_proj win further with an extended tactic sweep (all valid mma x cluster x swap x pf),
ranking the top-8 for o_proj M=1 and M=4 to see if anything beats ~14.2us toward the 7.34us BW floor.
"""
import torch, tensorrt_llm, itertools, cutlass
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVFP4BlackwellRunner
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.dense_blockscaled_gemm_persistent import Sm100BlockScaledPersistentDenseGemmKernel as KCLS
dev="cuda"; torch.manual_seed(0); SVS=16
def gscale(x): return (448.0*6.0)/x.abs().max().float()
def cqg(x,g):
    fp4,sf=torch.ops.trtllm.fp4_quantize(x,g.reshape(1),SVS,False); return fp4,sf
def cos(a,b):
    a=a.float().flatten(); b=b.float().flatten(); n=a.norm()*b.norm(); return (a@b/n).item()
def cap_us(fn,windows=8,it=60):
    for _ in range(8): fn()
    torch.cuda.synchronize()
    st=torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(st)
    g=torch.cuda.CUDAGraph(); h={}
    with torch.cuda.graph(g): h["o"]=fn()
    for _ in range(5): g.replay()
    torch.cuda.synchronize(); best=1e9
    for _ in range(windows):
        a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
        for _ in range(it): g.replay()
        b.record(); torch.cuda.synchronize(); best=min(best,a.elapsed_time(b)/it*1000.0)
    return best

mma_cands=[(128,8),(256,8),(128,16),(256,16),(128,32),(256,32),(128,64),(256,64),(128,128),(256,128)]
clu_cands=[(1,1),(1,2),(1,4),(2,1),(2,2),(2,4),(4,1),(4,2),(4,4),(8,1)]
runner=CuteDSLNVFP4BlackwellRunner(torch.bfloat16,0,None,True)

# ROOT CAUSE: instrument the AutoTuner's get_valid_tactics + which it would choose. We can't easily read
# its choice, but we can list valid tactics and time the stock op, then find which tactic matches stock time.
K_,N=16384,7168
w=torch.randn(N,K_,device=dev,dtype=torch.bfloat16); wg=gscale(w); wf,wsf=cqg(w,wg)
print(f"dev={torch.cuda.get_device_name(0)}",flush=True)
for M in (1,4):
    x=torch.randn(M,K_,device=dev,dtype=torch.bfloat16); xg=gscale(x); xf,xsf=cqg(x,xg)
    alpha=(1.0/(wg*xg)).reshape(1); inputs=[xf,wf,xsf,wsf,alpha]
    ref=torch.ops.trtllm.fp4_gemm(xf,wf,xsf,wsf,alpha,fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4,torch.bfloat16).clone()
    stock=cap_us(lambda: torch.ops.trtllm.cute_dsl_nvfp4_gemm_blackwell(xf,wf,xsf,wsf,alpha,torch.bfloat16))
    cub=cap_us(lambda: torch.ops.trtllm.nvfp4_gemm(xf,wf,xsf,wsf,alpha,torch.bfloat16,0,"cublaslt",None))
    res=[]
    for mma,clu in itertools.product(mma_cands,clu_cands):
        for swap_ab in (True,False):
            kernel_m=N if swap_ab else M; kernel_n=M if swap_ab else N; cmaj="m" if swap_ab else "n"
            try:
                if not KCLS.can_implement(cutlass.Float4E2M1FN,cutlass.Float8E4M3FN,16,cutlass.BFloat16,
                                          mma,clu,kernel_m,kernel_n,K_,1,"k","k",cmaj): continue
            except Exception: continue
            for pf in (False,True):
                tac=(mma,clu,swap_ab,pf)
                try:
                    y=runner.forward(inputs,tactic=tac).clone(); torch.cuda.synchronize()
                    cs=cos(y,ref)
                    if cs<0.999: continue
                    t=cap_us(lambda: runner.forward(inputs,tactic=tac))
                    res.append((t,tac,cs))
                except Exception: continue
    res.sort()
    print(f"\n=== o_proj M={M}: cublaslt={cub:.2f} stockAutoTuner={stock:.2f} BWfloor=7.34 | {len(res)} valid tactics",flush=True)
    print("  TOP-8 fastest:",flush=True)
    for t,tac,cs in res[:8]:
        print(f"    {t:.2f}us  {tac}  cos={cs:.5f}  vs_cublas={t/cub:.2f}",flush=True)
    print("  SLOWEST-3 (what AutoTuner may be landing on):",flush=True)
    for t,tac,cs in res[-3:]:
        print(f"    {t:.2f}us  {tac}  cos={cs:.5f}",flush=True)
print("\nDONE",flush=True)

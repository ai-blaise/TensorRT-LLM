"""v0: minimal cutedsl-vs-cublaslt validation bench (lock the loop before deep iteration).
Confirms the image has BOTH cublaslt + cutedsl registered, reproduces baseline us under capture
at the 4 shapes x M in {1,4,16,64}, with cos gate vs torch.ops.trtllm.fp4_gemm oracle.
"""
import torch, tensorrt_llm, sys
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
dev="cuda"; torch.manual_seed(0); SVS=16

def cq(x):
    g=(448.0*6.0)/x.abs().max().float()
    fp4,sf=torch.ops.trtllm.fp4_quantize(x,g.reshape(1),SVS,False)
    return fp4,sf,g
def cos(a,b):
    a=a.float().flatten(); b=b.float().flatten(); n=a.norm()*b.norm()
    return (a@b/n).item() if n>0 else float('nan')
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
    torch.cuda.synchronize()
    best=float("inf")
    for _ in range(windows):
        a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
        for _ in range(it): g.replay()
        b.record(); torch.cuda.synchronize(); best=min(best,a.elapsed_time(b)/it*1000.0)
    return best

shapes=[("o_proj",16384,7168),("q_a",7168,1536),("moe_up",7168,2048),("moe_down",2048,7168)]
print(f"dev={torch.cuda.get_device_name(0)}",flush=True)
print("ROW shape M backend cap_us cos note",flush=True)
for name,K,N in shapes:
    w=torch.randn(N,K,device=dev,dtype=torch.bfloat16); wf,wsf,wg=cq(w)
    for M in (1,4,16,64):
        x=torch.randn(M,K,device=dev,dtype=torch.bfloat16); xf,xsf,xg=cq(x)
        alpha=(1.0/(wg*xg)).reshape(1)
        ref=torch.ops.trtllm.fp4_gemm(xf,wf,xsf,wsf,alpha,fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4,torch.bfloat16).clone()
        bks={
          "cublaslt": lambda: torch.ops.trtllm.nvfp4_gemm(xf,wf,xsf,wsf,alpha,torch.bfloat16,0,"cublaslt",None),
          "cutedsl":  lambda: torch.ops.trtllm.cute_dsl_nvfp4_gemm_blackwell(xf,wf,xsf,wsf,alpha,torch.bfloat16),
        }
        for bn,fn in bks.items():
            note="ok"; c=float("nan"); cs=float("nan")
            try:
                y=fn().clone(); torch.cuda.synchronize(); cs=cos(y,ref)
                if cs<0.999: note="LOWCOS"
                c=cap_us(fn)
            except Exception as ex:
                note=f"ERR:{type(ex).__name__}:{str(ex)[:60]}"
            print(f"ROW {name} {M} {bn} {c:.2f} {cs:.5f} {note}",flush=True)
print("DONE",flush=True)

# Prototype: does adding 'cutedsl' to the NVFP4 allowed_backends + warming the AutoTuner
# automatically realize the o_proj win? Measures warmed-dispatcher prodset vs prodset+cutedsl.
import torch, tensorrt_llm
from tensorrt_llm._torch.autotuner import autotune
dev="cuda"; torch.manual_seed(0); SVS=16
def cq(x):
    g=(448.0*6.0)/x.abs().max().float()
    fp4,sf=torch.ops.trtllm.fp4_quantize(x,g.reshape(1),SVS,False)
    return fp4,sf,g
def cap_build(fn,reps=1):
    for _ in range(8): fn()
    torch.cuda.synchronize()
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            for _ in range(reps): fn()
    torch.cuda.current_stream().wait_stream(s)
    g=torch.cuda.CUDAGraph(); h={}
    with torch.cuda.graph(g):
        for _ in range(reps): h["o"]=fn()
    for _ in range(5): g.replay()
    torch.cuda.synchronize(); return g
def event_us(g,it=200):
    a=torch.cuda.Event(True);b=torch.cuda.Event(True);a.record()
    for _ in range(it): g.replay()
    b.record();torch.cuda.synchronize();return a.elapsed_time(b)/it*1000.0
def measure(allowed,xf,wf,xsf,wsf,alpha):
    def call(): return torch.ops.trtllm.nvfp4_gemm(xf,wf,xsf,wsf,alpha,torch.bfloat16,0,allowed,None)
    with autotune():
        for _ in range(6): call()
    torch.cuda.synchronize()
    out=call(); torch.cuda.synchronize()
    return event_us(cap_build(call,1)), out
PROD="cutlass,cublaslt,cuda_core"; PCD="cutlass,cublaslt,cuda_core,cutedsl"
shapes=[("o_proj",16384,7168),("q_a",7168,1536),("moe_up",7168,2048),("moe_down",2048,7168)]
print(f"{'shape':>9} {'M':>3} {'prodset':>9} {'+cutedsl':>9} {'lift':>6} {'cos':>8}",flush=True)
for name,K,N in shapes:
  w=torch.randn(N,K,device=dev,dtype=torch.bfloat16); wf,wsf,wg=cq(w)
  for M in [1,16]:
    x=torch.randn(M,K,device=dev,dtype=torch.bfloat16); xf,xsf,xg=cq(x); alpha=(1.0/(wg*xg)).reshape(1)
    try:
        b,_=measure(PROD,xf,wf,xsf,wsf,alpha)
        c,out=measure(PCD,xf,wf,xsf,wsf,alpha)
        ref=torch.ops.trtllm.nvfp4_gemm_cutlass(xf,wf,xsf,wsf,alpha,torch.bfloat16)
        cos=torch.nn.functional.cosine_similarity(out.flatten().float(),ref.flatten().float(),dim=0).item()
        print(f"{name:>9} {M:>3} {b:9.2f} {c:9.2f} {b/c:6.2f} {cos:8.5f}",flush=True)
    except Exception as e:
        print(f"{name:>9} {M:>3} ERR {repr(e)[:60]}",flush=True)
print("DONE",flush=True)

import torch, tensorrt_llm, json, sys
dev="cuda"; ops=torch.ops.trtllm; VEC=16; DT=torch.bfloat16
torch.manual_seed(1)  # different seed to confirm stability
def qf(t):
    amax=t.abs().amax().float(); gs=2688.0/amax
    f,sf=ops.fp4_quantize(t, torch.tensor([gs],device=dev,dtype=torch.float32), VEC, False, True)
    return f,sf,gs
def run(tok,xf,wf,xsf,wsf,a,N):
    o=ops.nvfp4_gemm(xf,wf,xsf,wsf,a,DT,output_buffer_kind=0,allowed_backends=tok,group=None)
    return o[...,:N].contiguous() if o.shape[-1]>N else o
def bench(fn,iters=400,warmup=60):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters): fn()
    torch.cuda.synchronize()
    best=1e9
    for _ in range(8):
        e0=torch.cuda.Event(True); e1=torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        best=min(best,e0.elapsed_time(e1)/iters)
    g.reset(); return best*1000.0
# o_proj only, M in {1,64}; also confirm eager (no-capture) ratio to rule out capture artifact
K,N=16384,7168
W=(torch.randn(N,K,device=dev,dtype=DT)*0.05); wf,wsf,w_gs=qf(W)
for M in [1,64]:
    x=torch.randn(M,K,device=dev,dtype=DT); xf,xsf,x_gs=qf(x)
    a=torch.tensor([1.0/(w_gs*x_gs)],device=dev,dtype=torch.float32)
    row={"M":M}
    for tok in ["cutlass","cublaslt","cutedsl"]:
        row[tok+"_us_cap"]=round(bench(lambda: run(tok,xf,wf,xsf,wsf,a,N)),3)
    # eager timing (no graph) for cublaslt vs cutedsl
    import time
    def eager(tok,it=300):
        for _ in range(50): run(tok,xf,wf,xsf,wsf,a,N)
        torch.cuda.synchronize(); t=time.time()
        for _ in range(it): run(tok,xf,wf,xsf,wsf,a,N)
        torch.cuda.synchronize(); return (time.time()-t)/it*1e6
    row["cublaslt_us_eager"]=round(eager("cublaslt"),3)
    row["cutedsl_us_eager"]=round(eager("cutedsl"),3)
    print("RECHECK",json.dumps(row)); sys.stdout.flush()

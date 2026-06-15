import torch, tensorrt_llm, json, sys
dev="cuda"; ops=torch.ops.trtllm; VEC=16; DT=torch.bfloat16
torch.manual_seed(0)

e2m1=torch.tensor([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0],device=dev)

def qf(t):
    amax=t.abs().amax().float(); gs=2688.0/amax
    f,sf=ops.fp4_quantize(t, torch.tensor([gs],device=dev,dtype=torch.float32), VEC, False, True)  # prod: swizzled
    return f,sf,gs

def dequant(t):
    # verified decoder via UNSWIZZLED quant (matches identity-GEMM dequant cos 0.999999)
    R,C=t.shape
    amax=t.abs().amax().float(); gs=2688.0/amax
    f,sf=ops.fp4_quantize(t, torch.tensor([gs],device=dev,dtype=torch.float32), VEC, False, False)
    lo=(f & 0x0F).long(); hi=((f>>4)&0x0F).long()
    vals=torch.empty(R,C,device=dev); vals[:,0::2]=e2m1[lo]; vals[:,1::2]=e2m1[hi]
    sfb=sf.view(torch.float8_e4m3fn).float().view(R,C//VEC).repeat_interleave(VEC,dim=1)
    return (vals*sfb/gs)

def run(tok, xf, wf, xsf, wsf, alpha, N):
    o=ops.nvfp4_gemm(xf,wf,xsf,wsf,alpha,DT,output_buffer_kind=0,allowed_backends=tok,group=None)
    return o[...,:N].contiguous() if o.shape[-1]>N else o

def cos_min(o, ref):
    o=o.float()
    return ((o*ref).sum(-1)/(o.norm(dim=-1)*ref.norm(dim=-1)+1e-12)).min().item()

def bench(fn, iters=300, warmup=50):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters): fn()
    torch.cuda.synchronize()
    best=1e9
    for _ in range(6):
        e0=torch.cuda.Event(True); e1=torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        best=min(best, e0.elapsed_time(e1)/iters)
    g.reset()
    return best*1000.0  # us/op

SHAPES=[("o_proj",16384,7168),("q_a_proj",7168,1536),("moe_up",7168,2048),("moe_down",2048,7168)]
MS=[1,16,64]
results=[]
for sname,K,N in SHAPES:
    W=(torch.randn(N,K,device=dev,dtype=DT)*0.05)
    wf,wsf,w_gs=qf(W)
    W_dq=dequant(W)  # [N,K]
    for M in MS:
        x=torch.randn(M,K,device=dev,dtype=DT)
        xf,xsf,x_gs=qf(x)
        x_dq=dequant(x)
        ref=(x_dq @ W_dq.T)  # dequant(x)@dequant(W).T  pure-arith reference
        alpha=torch.tensor([1.0/(w_gs*x_gs)],device=dev,dtype=torch.float32)
        row={"shape":sname,"K":K,"N":N,"M":M}
        for tok in ["cutlass","cublaslt","cutedsl"]:
            try:
                o=run(tok,xf,wf,xsf,wsf,alpha,N)
                row[tok+"_cos"]=round(cos_min(o,ref),5)
                row[tok+"_us"]=round(bench(lambda: run(tok,xf,wf,xsf,wsf,alpha,N)),3)
            except Exception as e:
                row[tok+"_cos"]=None; row[tok+"_us"]=None; row[tok+"_err"]=str(e)[:120]
        # decide winner among those passing gate
        cand={t:row.get(t+"_us") for t in ["cutlass","cublaslt","cutedsl"] if row.get(t+"_us") and (row.get(t+"_cos") or 0)>=0.999}
        if cand:
            win=min(cand,key=cand.get); row["winner"]=win
            if row.get("cublaslt_us") and row.get("cutedsl_us"):
                row["cutedsl_vs_cublaslt_speedup"]=round(row["cublaslt_us"]/row["cutedsl_us"],3)
        results.append(row); print("ROW",json.dumps(row)); sys.stdout.flush()
print("RESULTS_JSON",json.dumps(results))

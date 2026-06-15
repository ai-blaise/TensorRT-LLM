import torch, tensorrt_llm, json
dev="cuda"; ops=torch.ops.trtllm; DT=torch.bfloat16; VEC=16
torch.manual_seed(0)
SHAPES=[("o_proj",16384,7168),("q_a_proj",7168,1536),("moe_up",7168,2048),("moe_down",2048,7168)]
MS=[1,16,64]
def cos_rows(a,b):
    a=a.float();b=b.float()
    return ((a*b).sum(-1)/(a.norm(dim=-1)*b.norm(dim=-1)+1e-12))
# Production-faithful quant: sfUseUE8M0=False, isSfSwizzledLayout=True (the default).
def q(t, gs):
    return ops.fp4_quantize(t, torch.tensor([gs],device=dev,dtype=torch.float32), VEC, False, True)
for sname,K,N in SHAPES:
    W=(torch.randn(N,K,device=dev,dtype=DT)*0.05)
    amax_w=W.abs().amax().float(); w_gs=2688.0/amax_w
    wf,wsf=q(W,w_gs)
    for M in MS:
        x=torch.randn(M,K,device=dev,dtype=DT)
        amax_x=x.abs().amax().float(); x_gs=2688.0/amax_x
        xf,xsf=q(x,x_gs)
        alpha=torch.tensor([1.0/(w_gs*x_gs)],device=dev,dtype=torch.float32)
        ref=(x.float()@W.float().T).to(DT)
        out={}
        for tok in ["cutlass","cublaslt","cutedsl"]:
            try:
                o=ops.nvfp4_gemm(xf,wf,xsf,wsf,alpha,DT,output_buffer_kind=0,allowed_backends=tok,group=None)
                o=o[...,:N].contiguous() if o.shape[-1]>N else o
                out[tok]=round(cos_rows(o,ref).min().item(),5)
            except Exception as e:
                out[tok]="ERR:"+str(e)[:90]
        print("CORR", sname, "M",M, json.dumps(out)); 
        import sys; sys.stdout.flush()

import torch, tensorrt_llm
dev="cuda"; torch.manual_seed(0)
qz = torch.ops.trtllm.tunable_fp4_quantize
def quant(x, swizzled):
    gs=(x.abs().max().float()/(448.0*6.0)).clamp_min(1e-6).reshape(1)
    out=qz(x, gs, 32, swizzled)
    return out[0], out[1], gs
def bench(fn,it=100,wu=20):
    for _ in range(wu): fn()
    torch.cuda.synchronize();s=torch.cuda.Event(True);e=torch.cuda.Event(True);s.record()
    for _ in range(it): fn()
    e.record();torch.cuda.synchronize();return s.elapsed_time(e)/it*1000
def fmt(v): return f"{v:7.1f}" if isinstance(v,float) else f"{str(v)[:22]:>22}"
shapes=[("o_proj",16384,7168),("q_a_proj",7168,1536),("moe_up",7168,2048),("moe_down",2048,7168)]
SW=True  # swizzled SF layout for the GEMM
backends={
 "cutlass":lambda af,wf,asf,wsf,al: torch.ops.trtllm.nvfp4_gemm_cutlass(af,wf,asf,wsf,al,torch.bfloat16),
 "cublaslt":lambda af,wf,asf,wsf,al: torch.ops.trtllm.nvfp4_gemm_cublaslt(af,wf,asf,wsf,al,torch.bfloat16),
 "cutedsl":lambda af,wf,asf,wsf,al: torch.ops.trtllm.cute_dsl_nvfp4_gemm_blackwell(af,wf,asf,wsf,al,torch.bfloat16),
 "default":lambda af,wf,asf,wsf,al: torch.ops.trtllm.nvfp4_gemm(af,wf,asf,wsf,al,torch.bfloat16,0,"cutlass,cublaslt,cuda_core",None),
}
print(f"{'shape':>10} {'M':>3} "+" ".join(f"{b:>22}" for b in backends),flush=True)
for name,K,N in shapes:
    w=torch.randn(N,K,device=dev,dtype=torch.bfloat16); wf,wsf,wgs=quant(w,SW)
    for M in [1,4,16,64]:
        a=torch.randn(M,K,device=dev,dtype=torch.bfloat16); af,asf,ags=quant(a,SW)
        al=(ags*wgs).reshape(1)
        res={}
        for bn,fn in backends.items():
            try:
                fn(af,wf,asf,wsf,al); torch.cuda.synchronize()
                res[bn]=bench(lambda: fn(af,wf,asf,wsf,al))
            except Exception as ex: res[bn]=f"ERR:{str(ex)[:16]}"
        print(f"{name:>10} {M:>3} "+" ".join(fmt(res[b]) for b in backends),flush=True)

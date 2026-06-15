import torch, tensorrt_llm
dev="cuda"; ops=torch.ops.trtllm; VEC=16; DT=torch.bfloat16
torch.manual_seed(0)

def qf(t):
    amax=t.abs().amax().float(); gs=2688.0/amax
    f,sf=ops.fp4_quantize(t, torch.tensor([gs],device=dev,dtype=torch.float32), VEC, False, True)
    return f,sf,gs

def dequant_via_identity(tf, tsf, t_gs, R, C):
    # t is [R,C] quantized. dequant via cutlass GEMM with C-dim identity activation.
    I=torch.eye(C,device=dev,dtype=DT); 
    iamax=I.abs().amax().float(); i_gs=2688.0/iamax
    inf,insf=ops.fp4_quantize(I, torch.tensor([i_gs],device=dev,dtype=torch.float32), VEC, False, True)
    al=torch.tensor([1.0/(t_gs*i_gs)],device=dev,dtype=torch.float32)
    o=ops.nvfp4_gemm(inf,tf,insf,tsf,al,DT,output_buffer_kind=0,allowed_backends="cutlass",group=None)
    o=o[...,:R].contiguous() if o.shape[-1]>R else o   # [C,R]
    return o.T.contiguous().float()  # [R,C]

N,K=128,256
W=(torch.randn(N,K,device=dev,dtype=DT)*0.05)
wf,wsf,w_gs=qf(W)
W_dq=dequant_via_identity(wf,wsf,w_gs,N,K)  # [N,K]
for M in [1,16,64]:
    x=torch.randn(M,K,device=dev,dtype=DT)
    xf,xsf,x_gs=qf(x)
    x_dq=dequant_via_identity(xf,xsf,x_gs,M,K)  # [M,K]
    ref_xdq = (x_dq @ W_dq.T)          # dequant(x)@dequant(W).T  -> pure-arith ref
    ref_bf16= (x.float() @ W_dq.T)     # spec literal: x(bf16)@dequant(W).T
    al=torch.tensor([1.0/(w_gs*x_gs)],device=dev,dtype=torch.float32)
    res={}
    for tok in ["cutlass","cublaslt","cutedsl"]:
        o=ops.nvfp4_gemm(xf,wf,xsf,wsf,al,DT,output_buffer_kind=0,allowed_backends=tok,group=None)
        o=o[...,:N].contiguous() if o.shape[-1]>N else o
        c_xdq=((o.float()*ref_xdq).sum(-1)/(o.float().norm(dim=-1)*ref_xdq.norm(dim=-1)+1e-12)).min().item()
        c_bf =((o.float()*ref_bf16).sum(-1)/(o.float().norm(dim=-1)*ref_bf16.norm(dim=-1)+1e-12)).min().item()
        res[tok]=(round(c_xdq,6),round(c_bf,6))
    print("M",M,"(cos_vs_dqxdq, cos_vs_bf16x):",res)

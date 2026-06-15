import torch, tensorrt_llm
dev="cuda"; ops=torch.ops.trtllm; VEC=16; DT=torch.bfloat16
torch.manual_seed(0)
N,K=128,256
W=(torch.randn(N,K,device=dev,dtype=DT)*0.05)
amax_w=W.abs().amax().float(); w_gs=2688.0/amax_w
wf,wsf=ops.fp4_quantize(W, torch.tensor([w_gs],device=dev,dtype=torch.float32), VEC, False, True)  # swizzled (prod weight layout)
# Build dequant(W) via cutlass GEMM with FP4 identity activation:
# out[m,n] = sum_k Idq[m,k]*Wdq[n,k]; with I = K-dim identity (M=K rows), Idq≈I -> out[m,:]=Wdq[:,m]
# So out = Wdq.T  (shape [K, N]); then dequant(W) = out.T  shape [N,K].
I = torch.eye(K, device=dev, dtype=DT)  # [K,K], rows index k
amax_i=I.abs().amax().float(); i_gs=2688.0/amax_i
xf,xsf=ops.fp4_quantize(I, torch.tensor([i_gs],device=dev,dtype=torch.float32), VEC, False, True)
alpha=torch.tensor([1.0/(w_gs*i_gs)],device=dev,dtype=torch.float32)
out=ops.nvfp4_gemm(xf,wf,xsf,wsf,alpha,DT,output_buffer_kind=0,allowed_backends="cutlass",group=None)
out=out[...,:N].contiguous() if out.shape[-1]>N else out  # [K,N]
W_dq = out.T.contiguous().float()  # [N,K]
c=((W.float()*W_dq).sum()/(W.float().norm()*W_dq.norm()+1e-12)).item()
print("Wdq_via_identity cos_vs_W", round(c,6), "shape", tuple(W_dq.shape), "scale", (W.float().abs().mean()/(W_dq.abs().mean()+1e-12)).item())
# Now test: does cutlass(x) match x@W_dq.T to cos>=0.999 ?
for M in [1,16,64]:
    x=torch.randn(M,K,device=dev,dtype=DT)
    amax_x=x.abs().amax().float(); x_gs=2688.0/amax_x
    qxf,qxsf=ops.fp4_quantize(x, torch.tensor([x_gs],device=dev,dtype=torch.float32), VEC, False, True)
    al=torch.tensor([1.0/(w_gs*x_gs)],device=dev,dtype=torch.float32)
    ref=(x.float()@W_dq.T)  # x @ dequant(W).T
    o=ops.nvfp4_gemm(qxf,wf,qxsf,wsf,al,DT,output_buffer_kind=0,allowed_backends="cutlass",group=None)
    o=o[...,:N].contiguous() if o.shape[-1]>N else o
    cc=((o.float()*ref).sum(-1)/(o.float().norm(dim=-1)*ref.norm(dim=-1)+1e-12)).min().item()
    print("M",M,"cutlass cos vs x@Wdq.T", round(cc,6))

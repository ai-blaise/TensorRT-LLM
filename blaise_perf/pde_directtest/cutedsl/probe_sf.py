"""Probe quantized tensor + SF layout to design correct K-slicing for split-K."""
import torch, tensorrt_llm
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
dev="cuda"; torch.manual_seed(0); SVS=16
def cq(x):
    g=(448.0*6.0)/x.abs().max().float()
    fp4,sf=torch.ops.trtllm.fp4_quantize(x,g.reshape(1),SVS,False)
    return fp4,sf,g
def cos(a,b):
    a=a.float().flatten(); b=b.float().flatten(); n=a.norm()*b.norm()
    return (a@b/n).item()

K=16384; N=7168; M=4
w=torch.randn(N,K,device=dev,dtype=torch.bfloat16); wf,wsf,wg=cq(w)
x=torch.randn(M,K,device=dev,dtype=torch.bfloat16); xf,xsf,xg=cq(x)
print("xf",tuple(xf.shape),xf.dtype,"  wf",tuple(wf.shape),wf.dtype)
print("xsf",tuple(xsf.shape),xsf.dtype,"  wsf",tuple(wsf.shape),wsf.dtype,flush=True)
# xf is [M, K/2] uint8 (2 fp4/byte). SF for vec16 swizzled=False: expect [M, K/16] but may be padded/swizzled.
print("K/2",K//2,"K/16",K//16, "M*K/16",M*K//16, "N*K/16",N*K//16, flush=True)

alpha=(1.0/(wg*xg)).reshape(1)
ref=torch.ops.trtllm.fp4_gemm(xf,wf,xsf,wsf,alpha,fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4,torch.bfloat16).clone()

# Try slicing: does the kernel accept a re-quantized K-slice? Slice in ORIGINAL space, re-quantize each slice, sum.
S=4; Kc=K//S
acc=torch.zeros(M,N,device=dev,dtype=torch.float32)
for s in range(S):
    xs=x[:, s*Kc:(s+1)*Kc].contiguous(); ws=w[:, s*Kc:(s+1)*Kc].contiguous()
    xfs,xsfs,xgs=cq(xs); wfs,wsfs,wgs=cq(ws)
    al=(1.0/(wgs*xgs)).reshape(1)
    p=torch.ops.trtllm.nvfp4_gemm(xfs,wfs,xsfs,wsfs,al,torch.bfloat16,0,"cublaslt",None)
    acc+=p.float()
print("split-K(retquant per slice) cos vs full-quant ref:",cos(acc,ref),flush=True)

# Also: can we slice the ALREADY-quantized fp4 + sf along K directly (no requant)? Needs SF contiguous per-row.
# xf[:, s*Kc//2:(s+1)*Kc//2]; xsf slice = [:, s*(Kc//16):...] IF sf is [M,K/16] row-contiguous.
try:
    xsf2=xsf.view(M,-1); wsf2=wsf.view(N,-1)
    print("xsf view [M,?]:",tuple(xsf2.shape)," wsf view[N,?]:",tuple(wsf2.shape),flush=True)
except Exception as e:
    print("sf not simply [M,K/16]:",str(e)[:80],flush=True)

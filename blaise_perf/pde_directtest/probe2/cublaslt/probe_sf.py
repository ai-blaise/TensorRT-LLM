import torch, tensorrt_llm
dev="cuda"; ops=torch.ops.trtllm
from tensorrt_llm.quantization.utils import fp4_utils
M,K,N,VEC=64,256,128,16
torch.manual_seed(0)
W=(torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.05)
x=(torch.randn(M,K,device=dev,dtype=torch.bfloat16))
amax_w=W.abs().amax().float(); w_gs=2688.0/amax_w
amax_x=x.abs().amax().float(); x_gs=2688.0/amax_x
# unswizzled
wf_u, wsf_u = ops.fp4_quantize(W, torch.tensor([w_gs],device=dev), VEC, False)
xf_u, xsf_u = ops.fp4_quantize(x, torch.tensor([x_gs],device=dev), VEC, False)
print("W.shape",tuple(W.shape),"wf_u",tuple(wf_u.shape),wf_u.dtype,"wsf_u",tuple(wsf_u.shape),wsf_u.dtype)
print("x.shape",tuple(x.shape),"xf_u",tuple(xf_u.shape),"xsf_u",tuple(xsf_u.shape))
print("get_fp4_shape W False", fp4_utils.get_fp4_shape(W.shape, VEC, False))
print("get_fp4_shape W True ", fp4_utils.get_fp4_shape(W.shape, VEC, True))
print("get_fp4_shape x False", fp4_utils.get_fp4_shape(x.shape, VEC, False))
# SF dtype raw bytes
print("wsf_u numel", wsf_u.numel(), "expected NxK/VEC", N*(K//VEC))
print("xsf_u numel", xsf_u.numel(), "expected MxK/VEC", M*(K//VEC))

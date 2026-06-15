import torch, tensorrt_llm
dev="cuda"; ops=torch.ops.trtllm; VEC=16
torch.manual_seed(0)
N,K=128,256
W=(torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.05)
amax_w=W.abs().amax().float(); w_gs=(2688.0/amax_w)  # globalScale passed to quantizer
# Unswizzled quant (isSfSwizzledLayout=False), NVFP4 (sfUseUE8M0=False)
wf,wsf=ops.fp4_quantize(W, torch.tensor([w_gs],device=dev,dtype=torch.float32), VEC, False, False)
print("wf",tuple(wf.shape),wf.dtype,"wsf",tuple(wsf.shape),wsf.dtype,"numel",wsf.numel(),"=N*K/VEC?",N*K//VEC)

# E2M1 decode table (4-bit -> float), index 0..15
e2m1 = torch.tensor([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,
                     -0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0], device=dev)
# unpack two 4-bit per byte
lo = (wf & 0x0F).to(torch.long)
hi = ((wf >> 4) & 0x0F).to(torch.long)
vals = torch.empty(N, K, device=dev)
vals[:, 0::2] = e2m1[lo]
vals[:, 1::2] = e2m1[hi]
# SF: fp8 e4m3 per 16-block. Try interpret wsf bytes as float8_e4m3fn
sf = wsf.view(torch.float8_e4m3fn).float()  # length N*K/VEC, row-major [N, K/VEC]?
print("sf stats min/max/mean", sf.min().item(), sf.max().item(), sf.mean().item())
sf_blk = sf.view(N, K//VEC)
# Dequant candidate A: W_dq = vals * sf_per_block / w_gs   (since global scale was multiplied in)
sf_full = sf_blk.repeat_interleave(VEC, dim=1)  # [N,K]
for name, formula in [
    ("vals*sf/wgs", vals * sf_full / w_gs),
    ("vals*sf*amax/2688/... A", vals * sf_full / w_gs),
]:
    W_dq = formula
    c = ((W.float()*W_dq).sum()/((W.float().norm())*(W_dq.norm())+1e-12)).item()
    print(name, "cos_vs_W", round(c,6), "scale_ratio", (W.float().abs().mean()/ (W_dq.abs().mean()+1e-12)).item())

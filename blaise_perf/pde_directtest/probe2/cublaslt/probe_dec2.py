import torch, tensorrt_llm
dev="cuda"; ops=torch.ops.trtllm; VEC=16; DT=torch.bfloat16
torch.manual_seed(0)
N,K=128,256
W=(torch.randn(N,K,device=dev,dtype=DT)*0.05)
amax_w=W.abs().amax().float(); w_gs=2688.0/amax_w

# --- decoder on UNSWIZZLED quant ---
wf_u,wsf_u=ops.fp4_quantize(W, torch.tensor([w_gs],device=dev,dtype=torch.float32), VEC, False, False)
e2m1=torch.tensor([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0],device=dev)
lo=(wf_u & 0x0F).long(); hi=((wf_u>>4)&0x0F).long()
vals=torch.empty(N,K,device=dev); vals[:,0::2]=e2m1[lo]; vals[:,1::2]=e2m1[hi]
sf=wsf_u.view(torch.float8_e4m3fn).float().view(N,K//VEC)
sf_full=sf.repeat_interleave(VEC,dim=1)
# global scale: stored SF are e4m3(per-block-amax * gs / 6)? dequant = vals * sf / gs
Wdq_dec = (vals * sf_full / w_gs)

# --- identity-GEMM dequant on SWIZZLED quant ---
wf_s,wsf_s=ops.fp4_quantize(W, torch.tensor([w_gs],device=dev,dtype=torch.float32), VEC, False, True)
I=torch.eye(K,device=dev,dtype=DT); i_gs=2688.0/I.abs().amax().float()
inf,insf=ops.fp4_quantize(I, torch.tensor([i_gs],device=dev,dtype=torch.float32), VEC, False, True)
al=torch.tensor([1.0/(w_gs*i_gs)],device=dev,dtype=torch.float32)
o=ops.nvfp4_gemm(inf,wf_s,insf,wsf_s,al,DT,output_buffer_kind=0,allowed_backends="cutlass",group=None)
o=o[...,:N].contiguous() if o.shape[-1]>N else o
Wdq_id=o.T.contiguous().float()

def cos(a,b): return ((a*b).sum()/(a.norm()*b.norm()+1e-12)).item()
print("decoder vs W      ", round(cos(Wdq_dec, W.float()),6))
print("identity vs W     ", round(cos(Wdq_id,  W.float()),6))
print("decoder vs identity", round(cos(Wdq_dec, Wdq_id),6), "scale", (Wdq_id.abs().mean()/(Wdq_dec.abs().mean()+1e-12)).item())
# also check max abs elementwise diff after scale-match
s=(Wdq_id.abs().mean()/(Wdq_dec.abs().mean()+1e-12))
print("max|dec*s - id|", (Wdq_dec*s - Wdq_id).abs().max().item(), "vs typical", Wdq_id.abs().mean().item())

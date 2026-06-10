import torch, time
# DeepSeek-V3.2 MLA decode at N=1024 blocks = 65536 KV tokens, dim 576.
# Absorbed MLA: q has 128 heads x 576 (kv_lora+rope). Per decode step (1 query token).
dev=torch.device("cuda"); torch.cuda.set_device(0)
B=1; Hq=128; D=576; T=65536  # one request, full sparse-selected context at b32 share
q=torch.randn(Hq,D,device=dev,dtype=torch.bfloat16)
kv=torch.randn(T,D,device=dev,dtype=torch.bfloat16)
def qk():
    s=q@kv.t()            # [128,65536] QK^T scores
    p=torch.softmax(s.float(),dim=-1).to(torch.bfloat16)
    o=p@kv                # [128,576] PV
    return o
for _ in range(10): qk()
torch.cuda.synchronize(); t0=time.time()
for _ in range(50): qk()
torch.cuda.synchronize(); dt=(time.time()-t0)/50*1e6
print(f"FMHA-equivalent (QK+softmax+PV) over T={T} tok, Hq={Hq}: {dt:.1f} us/step")
dequant_us = 489.0  # fullFUSED re-dequant at N=1024 (kvarn_inkernel_bench)
print(f"  -> attention compute can hide at most {dt:.1f} of the {dequant_us:.0f} us "
      f"fused KVarN dequant ({100*min(dt/dequant_us,1):.0f}%); the remaining "
      f"{max(dequant_us-dt,0):.0f} us is EXPOSED unless amortized "
      f"(steadyAvg path) — overlap alone does NOT cover the fullFUSED cost")

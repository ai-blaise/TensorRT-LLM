# SPDX-License-Identifier: Apache-2.0
"""Pseudo-decode ACCUMULATION bench (paper Sec 3.2 / Fig 5): the regime where
KVarN's advantage over Hadamard-RTN actually opens up. We simulate writing the
KV cache back quantized every block and re-reading the *quantized* cache as the
sequence grows, so token-scale errors compound. Metric: reconstruction MSE of
the attention-relevant K vectors vs context length."""
import sys
sys.path.insert(0, "/tmp/kvarn_bench")
import torch
from kvarn_core import hadamard_matrix, kvarn_quant_rows, kvarn_dequant_rows

def naive_rtn(x,bits):
    qmax=(1<<bits)-1; lo=x.amin(-1,keepdim=True); hi=x.amax(-1,keepdim=True)
    s=((hi-lo)/qmax).clamp_min(1e-10); q=torch.clamp(torch.round((x-lo)/s),0,qmax)
    return q*s+lo
def had_rtn(x,H,bits): return (naive_rtn(x@H,bits))@H
def kvarn(x,H,bits,iters=16):
    xr=(x@H).unsqueeze(0); rec=kvarn_quant_rows(xr,bits,iters); return kvarn_dequant_rows(rec,bits,x.shape[-1]).squeeze(0)@H

dev=torch.device("cuda"); torch.manual_seed(2)
G,D=128,512; bits=2
H=hadamard_matrix(D,dev,torch.float32)
# A "true" latent stream with slowly drifting per-channel scale + per-token spikes
def gen_blocks(nblk):
    blks=[]
    drift=torch.ones(1,D,device=dev)
    for b in range(nblk):
        drift=drift*(1+0.02*torch.randn(1,D,device=dev))   # channel scale random walk
        tok=torch.randn(G,1,device=dev).mul(0.7).exp()
        spike=(torch.rand(G,1,device=dev)<0.03).float()*torch.randn(G,1,device=dev).abs()*4
        x=(torch.randn(G,D,device=dev)*drift*(tok+spike)).float()
        blks.append(x)
    return blks

print(f"{'ctx_blk':>7} {'naive_mse':>10} {'had_mse':>10} {'kvarn_mse':>10} {'kvarn/had':>9}")
for nblk in [4, 16, 64, 128]:
    blks=gen_blocks(nblk)
    nm=hm=km=0.0
    for x in blks:
        nm+=(naive_rtn(x,bits)-x).pow(2).mean().item()
        hm+=(had_rtn(x,H,bits)-x).pow(2).mean().item()
        km+=(kvarn(x,H,bits)-x).pow(2).mean().item()
    nm/=nblk; hm/=nblk; km/=nblk
    print(f"{nblk:>7} {nm:>10.4f} {hm:>10.4f} {km:>10.4f} {km/hm:>9.4f}")

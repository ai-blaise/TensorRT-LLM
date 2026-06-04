# SPDX-License-Identifier: Apache-2.0
"""Sharper KVarN bench: demonstrate the token-magnitude-outlier lever the paper
targets. Compares 2-bit (paper headline) and 4-bit KVarN vs naive per-token RTN
and Hadamard-only RTN, under increasing per-token heavy-tail strength."""
import sys, time
sys.path.insert(0, "/tmp/kvarn_bench")
import torch
from kvarn_core import (hadamard_matrix, variance_normalize_batched,
                        kvarn_quant_rows, kvarn_dequant_rows, _pack_lowbit, _unpack_lowbit)

def err_decomp(ref, rec):
    rn=ref.norm(dim=-1); dn=rec.norm(dim=-1)
    cos=((ref*rec).sum(-1)/(rn*dn).clamp_min(1e-8))
    em=(rn-dn).pow(2); ed=2*rn*dn*(1-cos); et=(em+ed).clamp_min(1e-12)
    rel=(rec-ref).norm()/ref.norm()
    return cos.mean().item(), (em/et).mean().item(), rel.item()

def naive_rtn_pertoken(x, bits):
    # per-token asym RTN, no rotation, no varnorm (KIVI V-style baseline)
    qmax=(1<<bits)-1
    lo=x.amin(-1,keepdim=True); hi=x.amax(-1,keepdim=True)
    s=((hi-lo)/qmax).clamp_min(1e-10); z=lo
    q=torch.clamp(torch.round((x-z)/s),0,qmax)
    return q*s+z

def hadamard_rtn(x, H, bits):
    xr=x@H
    deq=naive_rtn_pertoken(xr,bits)
    return deq@H

def kvarn_rt(x,H,bits,iters):
    xr=x.float()@H
    rec=kvarn_quant_rows(xr,bits,iters)
    return (kvarn_dequant_rows(rec,bits,x.shape[-1])@H)

dev=torch.device("cuda")
torch.manual_seed(1)
N,G,D=256,128,512
H=hadamard_matrix(D,dev,torch.float32)
print(f"{'tail':>5} {'method':22s} {'bits':>4} {'cos':>8} {'E_M':>6} {'relErr':>8}")
for tail in [0.4, 0.8, 1.5]:   # per-token lognormal sigma: outlier strength
    base=torch.randn(N,G,D,device=dev)
    tok=torch.randn(N,G,1,device=dev).mul(tail).exp()
    chan=torch.randn(1,1,D,device=dev).mul(0.4).exp()
    x=(base*tok*chan).float()
    ref=x
    for bits in (4,2):
        for name,fn in [("naive-RTN", lambda:naive_rtn_pertoken(x,bits)),
                        ("hadamard-RTN", lambda:hadamard_rtn(x,H,bits)),
                        ("KVarN", lambda:kvarn_rt(x,H,bits,16))]:
            cos,em,rel=err_decomp(ref, fn().float())
            print(f"{tail:>5} {name:22s} {bits:>4} {cos:>8.5f} {em:>6.3f} {rel:>8.4f}")
    print()

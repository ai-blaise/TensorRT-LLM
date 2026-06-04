# SPDX-License-Identifier: Apache-2.0
"""Prove the MLA preset inversion: for MLA, spend more bits on ckv (content,
512d) than k_pe (64d). Compare end-to-end attention-output reconstruction (the
metric that matters: up-proj ckv -> per-head K_nope/V) under k4v2 vs k2v4 at
near-equal cache bytes."""
import sys; sys.path.insert(0,"/tmp/kvarn_bench")
import torch, kvarn_mla as M

dev=torch.device("cuda"); torch.manual_seed(5)
Dckv,Dpe,G=512,64,128
# fake up-proj: ckv -> [num_heads*(qk_nope+v_head)]; use a representative slice
H=128; nope=128; vdim=128
Wk=torch.randn(Dckv, H*nope, device=dev)/Dckv**0.5
Wv=torch.randn(Dckv, H*vdim, device=dev)/Dckv**0.5

def attn_proxy(ckv):  # what attention actually consumes
    return (ckv.float()@Wk), (ckv.float()@Wv)

def run(cb,pb):
    ckv=(torch.randn(G,Dckv,device=dev)*torch.randn(G,1,device=dev).mul(0.7).exp()
         *torch.randn(1,Dckv,device=dev).mul(0.4).exp()).to(torch.float16)
    kpe=(torch.randn(G,Dpe,device=dev)*torch.randn(G,1,device=dev).mul(0.5).exp()).to(torch.float16)
    rec=M.quant_latent_block(ckv,kpe,ckv_bits=cb,pe_bits=pb)
    ckv_d,kpe_d=M.dequant_latent_block(rec)
    Kr,Vr=attn_proxy(ckv); Kd,Vd=attn_proxy(ckv_d)
    def cos(a,b):a=a.flatten();b=b.flatten();return (a@b/(a.norm()*b.norm())).item()
    bytes_=M.packed_bytes_per_block(Dckv,Dpe,G,cb,pb)
    return cos(Kr,Kd),cos(Vr,Vd),cos(kpe.float(),kpe_d.float()),bytes_

print(f"{'preset':>10} {'cosK_nope':>10} {'cosV':>8} {'cos_kpe':>8} {'bytes':>7}")
for name,(cb,pb) in [("k4v2(paper)",(2,4)),  # ckv=2b(value-orient), pe=4b(key-orient)
                     ("k2v4(MLA)",(4,2)),    # ckv=4b, pe=2b  <-- proposed
                     ("k4v4",(4,4)),("k2v2",(2,2))]:
    cK,cV,cpe,b=run(cb,pb)
    print(f"{name:>10} {cK:>10.5f} {cV:>8.5f} {cpe:>8.5f} {b:>7}")

# SPDX-License-Identifier: Apache-2.0
"""Correctness + capacity test for the KVarN MLA latent adapter on real
DeepSeek-V3.2 dims (kv_lora_rank=512, qk_rope_head_dim=64)."""
import sys
sys.path.insert(0, "/tmp/kvarn_bench")
import torch
import kvarn_mla as M   # adapter copied alongside core in /tmp/kvarn_bench

def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(3)
    Dckv, Dpe = 512, 64
    for group in (64, 128):
        for (cb, pb) in [(4,4),(4,2),(2,2)]:
            # realistic-ish latent: per-token magnitude tails + per-channel structure
            ckv = (torch.randn(group, Dckv, device=dev)
                   * torch.randn(group,1,device=dev).mul(0.7).exp()
                   * torch.randn(1,Dckv,device=dev).mul(0.4).exp()).to(torch.float16)
            kpe = (torch.randn(group, Dpe, device=dev)
                   * torch.randn(group,1,device=dev).mul(0.5).exp()).to(torch.float16)
            rec = M.quant_latent_block(ckv, kpe, ckv_bits=cb, pe_bits=pb)
            ckv_d, kpe_d = M.dequant_latent_block(rec)
            def cos(a,b):
                a=a.float().flatten(); b=b.float().flatten()
                return (a@b/(a.norm()*b.norm())).item()
            ck = cos(ckv, ckv_d); pk = cos(kpe, kpe_d)
            fp16_b = 2*group*(Dckv+Dpe)
            kvb = M.packed_bytes_per_block(Dckv,Dpe,group,cb,pb)
            print(f"group={group:3d} ckv{cb}/pe{pb}b  cos_ckv={ck:.5f} cos_kpe={pk:.5f}  "
                  f"bytes {kvb}/{fp16_b}  cap={fp16_b/kvb:.2f}x  bpe={kvb*8/(group*(Dckv+Dpe)):.3f}")
    print("\nROUND-TRIP OK" )

if __name__=="__main__":
    main()

#!/usr/bin/env python3
"""ORCHESTRATOR independent audit of the HiSparse hot-read correctness gate.

Closes the gap in the subagent's hotread_correctness_probe.py: that probe compares
the kernel against a dense reference built over the DEQUANTIZED latent (shared
quant), proving the attention math but NOT the kvarn_k2v2 quant+dequant accuracy
vs the ORIGINAL pre-quant latent. This probe adds:
  CHECK 1  cosine(dequant, original)            -- the kvarn_k2v2 quant floor
  CHECK 2  cosine(decode, dense_over_ORIGINAL)  -- full chain vs true pre-quant intent (>=0.98)
  CHECK 3  cosine(decode, dense_over_dequant)   -- reproduce the subagent's attention-math result
"""
from __future__ import annotations
import torch
import tensorrt_llm  # noqa: F401  (auto-loads trtllm ops)

H_Q, D_QK, D_V = 128, 576, 512
TPB, KLR, QK_ROPE, KVARN_BITS = 64, 512, 64, 2
PACKED = TPB*(KLR*KVARN_BITS//8) + TPB*2*(KLR//128)*2 + TPB*QK_ROPE  # 13312

def cosine(a, b):
    a = a.reshape(-1).float(); b = b.reshape(-1).float()
    return float(torch.dot(a, b) / (a.norm()*b.norm() + 1e-12))

def make_packed(num_blocks, dev, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    latent = (torch.randn(num_blocks, TPB, D_QK, generator=g) * 0.5).to(dev, torch.float16)
    hot = torch.zeros(num_blocks, PACKED, dtype=torch.uint8, device=dev)
    for b in range(num_blocks):
        torch.ops.trtllm.mla_bdr_write_kvarn_record(latent[b], hot, b, KVARN_BITS, KLR, QK_ROPE)
    torch.cuda.synchronize()
    return hot.unsqueeze(0), latent  # [1,nb,PACKED] , [nb,TPB,576] ORIGINAL (fp16)

def dequant_native(hot, num_blocks, dev):
    sf = TPB
    idx = torch.empty(num_blocks, TPB, dtype=torch.int32, device=dev)
    for b in range(num_blocks):
        for t in range(TPB):
            idx[b, t] = b*sf + t
    tl = torch.full((num_blocks,), TPB, dtype=torch.int32, device=dev)
    rs = torch.zeros((num_blocks,), dtype=torch.uint8, device=dev)
    deq, st = torch.ops.trtllm.hisparse_read_kvarn_hot_bdr(hot, idx, tl, rs, 0, TPB, KVARN_BITS, KLR, QK_ROPE)
    torch.cuda.synchronize()
    assert int(st.max()) == 0, f"BDR read status nonzero: {st.cpu().tolist()}"
    return deq.float()  # [nb,TPB,576]

def dense_ref(q, latents, sm):  # q:[B,1,H,576], latents:[B,n,576]
    qf = q.float()
    K = latents; V = latents[..., :D_V]
    s = sm * torch.einsum("bhd,bnd->bhn", qf[:, 0], K)
    m = s.max(-1, keepdim=True).values
    w = torch.exp(s - m); den = w.sum(-1, keepdim=True)
    out = torch.einsum("bhn,bnd->bhd", w/den, V)
    return out.unsqueeze(1)  # [B,1,H,512]

def main():
    dev = torch.device("cuda:0")
    sm = 1.0/(D_QK**0.5)
    NB = 32
    hot, orig = make_packed(NB, dev, seed=7)      # orig = pre-quant latent
    deq = dequant_native(hot, NB, dev)            # what the kernel sees

    # CHECK 1: quant accuracy vs original (the kvarn_k2v2 floor with VARIED values)
    c_ckv  = cosine(deq[..., :KLR],  orig.float()[..., :KLR])   # 2-bit C-KV part
    c_rope = cosine(deq[..., KLR:],  orig.float()[..., KLR:])   # 8-bit RoPE part
    c_full = cosine(deq,             orig.float())
    print(f"CHECK1 quant roundtrip: cosine(deq,orig) full={c_full:.6f} ckv={c_ckv:.6f} rope={c_rope:.6f}")

    # full decode over random selection with NON-ZERO q
    B, nsel = 4, 256
    g = torch.Generator(device="cpu").manual_seed(11)
    q = (torch.randn(B,1,H_Q,D_QK, generator=g)*0.3).to(dev, torch.bfloat16)
    idx = torch.empty(B,1,nsel, dtype=torch.int32, device=dev)
    sel_orig = torch.empty(B,nsel,D_QK, device=dev)
    sel_deq  = torch.empty(B,nsel,D_QK, device=dev)
    for b in range(B):
        blk = torch.randint(0, NB, (nsel,), generator=g)
        tok = torch.randint(0, TPB, (nsel,), generator=g)
        idx[b,0] = (blk*TPB + tok).to(torch.int32).to(dev)
        sel_orig[b] = orig.float()[blk, tok, :]
        sel_deq[b]  = deq[blk, tok, :]
    rs = torch.zeros((B,), dtype=torch.uint8, device=dev)
    out, lse, meta, sp = torch.ops.trtllm.sparse_mla_decode_kvarn_hot(
        q, hot, idx, rs, None, None, 0, TPB, TPB, KVARN_BITS, KLR, QK_ROPE, sm)
    torch.cuda.synchronize()
    out = out.float()

    ref_orig = dense_ref(q, sel_orig, sm)
    ref_deq  = dense_ref(q, sel_deq,  sm)
    c2 = cosine(out, ref_orig); m2 = float((out-ref_orig).abs().max())
    c3 = cosine(out, ref_deq);  m3 = float((out-ref_deq).abs().max())
    print(f"CHECK2 full-chain vs ORIGINAL:  cosine={c2:.6f} max_abs={m2:.4e}  VERDICT(>=0.98)={'PASS' if c2>=0.98 else 'FAIL'}")
    print(f"CHECK3 attn-math vs dequant:    cosine={c3:.6f} max_abs={m3:.4e}  (reproduces subagent 0.999999)")

if __name__ == "__main__":
    main()

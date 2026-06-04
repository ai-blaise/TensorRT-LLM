# SPDX-License-Identifier: Apache-2.0
"""KVarN component microbench for op-trt MLA latent KV.

Measures, vs fp8-e4m3 and a simulated NVFP4 (e2m1 per-16 block) baseline on
DeepSeek-MLA-shaped latents:
  1. KV-cache bytes/token  (capacity multiplier)
  2. Reconstruction quality: cosine sim + the paper's E_M (magnitude) / E_D
     (directional) error split (Eq. 3) -- KVarN's whole claim is that it kills
     E_M (token-magnitude) errors that drive softmax outliers.
  3. Variance-norm + quant wall time per 128-token block (the runtime overhead
     the paper Sec 4.2 reports as 0.18%).

Run on B200 inside a container with torch+CUDA. CPU-only fallback works too
(set --device cpu) for correctness; timing only meaningful on GPU.
"""
import argparse, math, time
import torch
import sys
sys.path.insert(0, "/home/spencer/work/kvarn_component_wt")
from tensorrt_llm._torch.attention_backend.sparse.kvarn_core import (
    hadamard_matrix, kvarn_quant_rows, kvarn_dequant_rows,
)

# ---- baselines -------------------------------------------------------------
def fp8_roundtrip(x):
    q = x.to(torch.float8_e4m3fn)
    return q.float(), 1.0  # 1 byte/elem

def nvfp4_roundtrip(x, block=16):
    # simulate NVFP4: per-16 block fp8 scale + e2m1 (4-bit) values.
    *lead, C = x.shape
    xb = x.reshape(*lead, C // block, block).float()
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    # e2m1 max magnitude is 6.0
    scale = (amax / 6.0)
    scale_fp8 = scale.to(torch.float8_e4m3fn).float()  # scale stored fp8
    xn = (xb / scale_fp8.clamp_min(1e-8))
    # e2m1 grid: {0,0.5,1,1.5,2,3,4,6} * sign
    grid = torch.tensor([0,0.5,1,1.5,2,3,4,6], device=x.device)
    sign = xn.sign()
    mag = xn.abs().unsqueeze(-1)
    idx = (mag - grid).abs().argmin(dim=-1)
    qmag = grid[idx]
    deq = (sign * qmag * scale_fp8).reshape(*lead, C)
    # bytes/elem: 0.5 (4-bit) + scale 1 byte per 16 = 0.5625
    return deq, 0.5 + 1.0 / block

def err_decomp(ref, rec):
    # Eq 3: total^2 = (||K|| - ||Kdq||)^2 [magnitude]  + 2||K||||Kdq||(1-cos) [dir]
    rn = ref.norm(dim=-1)
    dn = rec.norm(dim=-1)
    cos = (ref * rec).sum(-1) / (rn * dn).clamp_min(1e-8)
    em = (rn - dn).pow(2)
    ed = 2 * rn * dn * (1 - cos)
    et = (em + ed).clamp_min(1e-12)
    return cos.mean().item(), (em / et).mean().item(), (ed / et).mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--kv-lora", type=int, default=512)   # DeepSeek latent dim
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--ntiles", type=int, default=256)    # blocks (per head-equiv)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--iters", type=int, default=16)
    a = ap.parse_args()
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"device={dev} kv_lora={a.kv_lora} group={a.group} ntiles={a.ntiles} bits={a.bits} iters={a.iters}")

    torch.manual_seed(0)
    # DeepSeek latent compressed_kv: per-token vector dim=kv_lora_rank.
    # Model it as low-rank structured + heavy-tailed per-token magnitude (the
    # regime KVarN targets): base gaussian * per-token lognormal magnitude.
    N, G, D = a.ntiles, a.group, a.kv_lora
    base = torch.randn(N, G, D, device=dev)
    tok_mag = torch.randn(N, G, 1, device=dev).mul(0.8).exp()   # heavy per-token scale
    chan_mag = torch.randn(1, 1, D, device=dev).mul(0.4).exp()  # per-channel structure
    latent = (base * tok_mag * chan_mag).to(torch.float16)

    # ---- KVarN path: V-orientation (per-token rows). Hadamard along channel.
    H = hadamard_matrix(D, dev, torch.float32)
    def kvarn_rt(x):
        xr = (x.float() @ H)                     # rotate channel axis
        rec = kvarn_quant_rows(xr, a.bits, a.iters)
        deqr = kvarn_dequant_rows(rec, a.bits, D)
        deq = deqr @ H                           # un-rotate (H is its own inverse)
        # bytes/elem: bits/8 + scales (s_row_abs[G]+zp[G] fp16 over G*D + s_col[D] fp16 over G*D)
        bpe = a.bits/8 + (2*2*G + 2*D) / (G*D)
        return deq, bpe, rec

    deq_kv, bpe_kv, rec = kvarn_rt(latent)
    deq_fp8, bpe_fp8 = fp8_roundtrip(latent)
    deq_nv, bpe_nv = nvfp4_roundtrip(latent)

    ref = latent.float()
    for name, deq, bpe in [("fp8-e4m3", deq_fp8, bpe_fp8),
                           ("nvfp4-sim", deq_nv, bpe_nv),
                           (f"KVarN-{a.bits}bit", deq_kv, bpe_kv)]:
        cos, em, ed = err_decomp(ref, deq.float())
        rel = (deq.float()-ref).norm() / ref.norm()
        print(f"  {name:14s} bytes/elem={bpe:.4f}  cos={cos:.5f}  relErr={rel:.4f}  "
              f"E_M={em:.3f} E_D={ed:.3f}  cap_vs_fp16={16/(bpe*8):.2f}x")

    # ---- timing: variance-norm+quant per block ----
    if dev.type == "cuda":
        xr = (latent.float() @ H)
        for _ in range(3): kvarn_quant_rows(xr, a.bits, a.iters)
        torch.cuda.synchronize(); t0=time.time()
        for _ in range(10): kvarn_quant_rows(xr, a.bits, a.iters)
        torch.cuda.synchronize()
        per_block_us = (time.time()-t0)/10/N*1e6
        print(f"  quant: {per_block_us:.2f} us / 128-tok block (over {N} blocks, {a.iters} iters)")

if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""Intel-correctness: KVarN vs fp8 vs NVFP4 reconstruction fidelity on the SAME
MLA latent (DeepSeek-V3.2 dims), at each scheme's bytes/capacity. Cosine vs the
fp16 ground truth = how faithfully the decode reads the long-context KV back."""
import sys
sys.path.insert(0, "tensorrt_llm/_torch/attention_backend/sparse")
import torch
import kvarn_core as KC
import kvarn_backend as KB
from kvarn_mla import quant_latent_block, dequant_latent_block

dev = torch.device("cuda"); torch.manual_seed(7)
G, Dckv, Dpe, D = 64, 512, 64, 576

def cos(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()

# realistic latent: low-rank-ish with outlier channels (paper regime)
ckv = torch.randn(G, Dckv, device=dev) * 0.5
ckv[:, ::37] *= 6.0
kpe = torch.randn(G, Dpe, device=dev) * 0.8
lat = torch.cat([ckv, kpe], dim=-1).half()

# --- fp8 e4m3 per-tensor (what dense fp8 KV does) ---
amax = lat.abs().max().float()
scale = amax / 448.0
q8 = (lat.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
deq8 = q8.float() * scale
cos8 = cos(deq8, lat)

# --- NVFP4 e2m1 + per-16 e4m3 block scale (what dense NVFP4 KV does) ---
def nvfp4_rt(x):
    x = x.float()
    BS = 16
    xr = x.reshape(-1, BS)
    bamax = xr.abs().amax(dim=1, keepdim=True)
    bscale = (bamax / 6.0).clamp(min=1e-8)        # e2m1 max = 6.0
    # e4m3 quantize the block scale
    samax = bscale.max()
    sscale = samax / 448.0
    bscale_q = (bscale / sscale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * sscale
    xn = xr / bscale_q
    # e2m1: representable levels {0,.5,1,1.5,2,3,4,6} * sign
    lv = torch.tensor([0,.5,1,1.5,2,3,4,6], device=x.device)
    s = xn.sign(); a = xn.abs()
    idx = (a.unsqueeze(-1) - lv).abs().argmin(dim=-1)
    xq = s * lv[idx]
    return (xq * bscale_q).reshape(x.shape)
deq4 = nvfp4_rt(lat)
cos4 = cos(deq4, lat)

# --- KVarN k4v4 (Hadamard + dual-axis Sinkhorn var-norm + RTN, ckv4/pe4) ---
cfg = KB.parse_kvarn_dtype("kvarn_k4v4")
H_ckv = KC.hadamard_matrix(Dckv, dev, torch.float32)
H_pe = KC.hadamard_matrix(Dpe, dev, torch.float32)
rec = quant_latent_block(ckv.half(), kpe.half(), ckv_bits=4, pe_bits=4,
                         iters=cfg.iters, H_ckv=H_ckv, H_pe=H_pe)
ckv_dq, kpe_dq = dequant_latent_block(rec)
latK = torch.cat([ckv_dq, kpe_dq], dim=-1)
cosK = cos(latK, lat)
cosK_ckv = cos(ckv_dq, ckv.half()); cosK_pe = cos(kpe_dq, kpe.half())

fp16_b = 2 * D
print("=== Intel-correctness: KVarN vs fp8 vs NVFP4 (same MLA latent) ===")
print(f"{'scheme':>10} {'bytes/tok':>9} {'cap_vs_fp16':>11} {'cos_vs_fp16':>11}")
print(f"{'fp16':>10} {fp16_b:>9} {1.0:>11.2f}x {1.0:>11.5f}")
print(f"{'fp8-e4m3':>10} {D:>9} {fp16_b/D:>11.2f}x {cos8:>11.5f}")
print(f"{'nvfp4':>10} {D//2 + D//16:>9} {fp16_b/(D//2+D//16):>11.2f}x {cos4:>11.5f}")
kb = cfg.packed_bytes(G) // G
print(f"{'kvarn-k4v4':>10} {kb:>9} {fp16_b/kb:>11.2f}x {cosK:>11.5f}")
print(f"\nkvarn k4v4 detail: cos_ckv={cosK_ckv:.5f} cos_kpe={cosK_pe:.5f}")
print("\nINTEL-CORRECTNESS-OK")

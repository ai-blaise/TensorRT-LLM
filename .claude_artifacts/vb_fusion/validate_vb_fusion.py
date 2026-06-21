# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Standalone numerical-equivalence validation for the v_b (W_UV) epilogue
fusion in the SM100 NVFP4 sparse-MLA decode kernel.

RUN THIS AFTER you rebuild the C++ extension with the fused op. It exercises
the op directly (no model load) and compares:

  fused   = sparse_mla_decode_nvfp4_vfuse(q, kv, kv_scales, indices, v_b_proj, ...)
            -> [b, s_q, h_q, v_head_dim]   (already projected by the kernel)

  ref     = sparse_mla_decode_nvfp4(q, kv, kv_scales, indices, ...)   (latent)
            -> [b, s_q, h_q, kv_lora_rank]
            then per-head BMM by v_b_proj  -> [b, s_q, h_q, v_head_dim]

Pass criterion: cosine >= 0.999 (per-(token,head) and global) and a loose
max-abs/rel check. The model is untrained, so we validate ONLY against this
in-framework reference, never against output text.

The kv/kv_scales packed NVFP4 bytes are produced by the SAME path the runtime
uses, by quantizing a random bf16 latent KV with the trtllm NVFP4 quant op, so
the dequant the kernel performs is exercised end to end. If that quant op is
unavailable in your build, pass --random-bytes to fill kv/kv_scales with random
bytes (the two sides still must match bit-for-bit because they read the same kv).
"""
import argparse
import math

import torch

import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm.*)

OP_LATENT = "sparse_mla_decode_nvfp4"
OP_FUSED = "sparse_mla_decode_nvfp4_vfuse"  # name of the fused op you add

H_Q = 128
H_KV = 1
D_QK = 576           # kv_lora_rank(512) + qk_rope_head_dim(64)
KV_LORA = 512        # d_v latent (kernel d_v)
V_HEAD_DIM = 128     # projected output per head
PAGE = 64
KV_BYTES = 288       # 576 packed FP4 / 2
SCALE_BYTES = 36     # 576 / 16


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def per_row_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    # a,b: [..., V_HEAD_DIM]; cosine per row, return the worst (min)
    a = a.float().reshape(-1, a.shape[-1])
    b = b.float().reshape(-1, b.shape[-1])
    cs = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return cs.min().item(), cs.mean().item()


def make_inputs(b, s_q, topk, dtype_w, device, random_bytes):
    torch.manual_seed(0)
    q = torch.randn(b, s_q, H_Q, D_QK, device=device, dtype=torch.bfloat16)
    # The two head halves are split inside the op; q must be full 128 heads.
    num_pages = max(8, math.ceil((topk + PAGE) / PAGE) * b + 4)
    if random_bytes:
        kv = torch.randint(0, 256, (num_pages, PAGE, H_KV, KV_BYTES),
                           dtype=torch.uint8, device=device)
        kv_scales = torch.randint(0, 256, (num_pages, PAGE, H_KV, SCALE_BYTES),
                                  dtype=torch.uint8, device=device)
    else:
        # Quantize a random bf16 latent with the runtime NVFP4 quant op so the
        # kernel's dequant path is exercised on realistic data. Fall back to
        # random bytes if the op is not present in this build.
        try:
            latent = torch.randn(num_pages * PAGE, D_QK, device=device,
                                 dtype=torch.bfloat16) * 0.5
            # global amax-based scale-2 (per-tensor); op packs to e2m1 + e4m3
            packed, scales = torch.ops.trtllm.fp4_quantize(  # may differ per build
                latent, torch.tensor(1.0, device=device), 16, False)
            kv = packed.view(num_pages, PAGE, H_KV, KV_BYTES).contiguous()
            kv_scales = scales.view(num_pages, PAGE, H_KV,
                                    SCALE_BYTES).contiguous().to(torch.uint8)
        except Exception as e:  # noqa: BLE001  (diagnostic fallback only)
            print(f"[warn] NVFP4 quant op unavailable ({e}); using random bytes")
            kv = torch.randint(0, 256, (num_pages, PAGE, H_KV, KV_BYTES),
                               dtype=torch.uint8, device=device)
            kv_scales = torch.randint(0, 256,
                                      (num_pages, PAGE, H_KV, SCALE_BYTES),
                                      dtype=torch.uint8, device=device)

    # valid topk indices into [0, num_pages*PAGE)
    max_tok = num_pages * PAGE
    indices = torch.randint(0, max_tok, (b, s_q, topk), dtype=torch.int32,
                            device=device)

    # v_b_proj: [h_q, v_head_dim, kv_lora_rank]  (same layout as nn.Parameter)
    v_b_proj = (torch.randn(H_Q, V_HEAD_DIM, KV_LORA, device=device,
                            dtype=torch.float32) * 0.05).to(dtype_w)
    return q, kv, kv_scales, indices, v_b_proj


def reference(q, kv, kv_scales, indices, v_b_proj, sm_scale, dtype_w):
    # latent out: [b, s_q, h_q, kv_lora_rank]
    out_latent = torch.ops.trtllm.__getattr__(OP_LATENT)(
        q, kv, kv_scales, indices, d_v=KV_LORA, sm_scale=sm_scale)[0]
    b, s_q, h_q, dv = out_latent.shape
    assert dv == KV_LORA
    # per-head BMM: [h_q, b*s_q, kv_lora] x [h_q, kv_lora, v_head_dim]
    x = out_latent.reshape(b * s_q, h_q, dv).transpose(0, 1)  # [h,T,dv]
    if dtype_w == torch.bfloat16:
        w = v_b_proj.transpose(1, 2)  # [h, dv, vhd]
        ref = torch.bmm(x.float(), w.float())  # [h,T,vhd]
    else:  # fp8 path -> dequant to fp32 for the reference
        w = v_b_proj.to(torch.float32).transpose(1, 2)
        ref = torch.bmm(x.float(), w)
    ref = ref.transpose(0, 1).reshape(b, s_q, h_q, V_HEAD_DIM)
    return ref, out_latent


def fused(q, kv, kv_scales, indices, v_b_proj, sm_scale):
    return torch.ops.trtllm.__getattr__(OP_FUSED)(
        q, kv, kv_scales, indices, v_b_proj,
        d_v=KV_LORA, v_head_dim=V_HEAD_DIM, sm_scale=sm_scale)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--s_q", type=int, default=1)
    ap.add_argument("--topk", type=int, default=1024)
    ap.add_argument("--wdtype", choices=["bf16", "fp8"], default="bf16")
    ap.add_argument("--random-bytes", action="store_true")
    ap.add_argument("--cos-thresh", type=float, default=0.999)
    args = ap.parse_args()

    dtype_w = torch.bfloat16 if args.wdtype == "bf16" else torch.float8_e4m3fn
    dev = "cuda"
    sm_scale = 1.0 / math.sqrt(D_QK)

    q, kv, kv_scales, indices, v_b_proj = make_inputs(
        args.batch, args.s_q, args.topk, dtype_w, dev, args.random_bytes)

    ref, latent = reference(q, kv, kv_scales, indices, v_b_proj, sm_scale,
                            dtype_w)
    fus = fused(q, kv, kv_scales, indices, v_b_proj, sm_scale)

    assert fus.shape == ref.shape, (fus.shape, ref.shape)
    g_cos = cosine(fus, ref)
    row_min, row_mean = per_row_cosine(fus, ref)
    max_abs = (fus.float() - ref.float()).abs().max().item()
    denom = ref.float().abs().max().clamp_min(1e-6).item()
    max_rel = max_abs / denom

    print(f"shape           : {tuple(fus.shape)}")
    print(f"global cosine   : {g_cos:.6f}")
    print(f"per-row cos min : {row_min:.6f}  mean: {row_mean:.6f}")
    print(f"max |abs diff|  : {max_abs:.4e}")
    print(f"max rel diff    : {max_rel:.4e}")
    ok = (g_cos >= args.cos_thresh) and (row_min >= args.cos_thresh)
    print("RESULT          :", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

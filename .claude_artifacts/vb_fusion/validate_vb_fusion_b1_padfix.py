# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""B1 split-path validator with the NVFP4 scale-padding fix.

The committed validate_vb_fusion.py assumes torch.ops.trtllm.fp4_quantize returns
a scale tensor of exactly rows*36 bytes. For row counts that are not a multiple of
128 (e.g. B1 -> num_pages=21 -> 1344 rows) the op pads the *scale-factor* tensor to
a 128-row-swizzled layout (1408 rows -> 50688 bytes), so the harness .view() throws
and it silently falls back to RANDOM BYTES, which make the attention softmax produce
NaN (both ref and fused) -> cosine NaN -> spurious FAIL.

This driver slices the padded scale tensor back to the needed rows*36 bytes so the
B1 split path runs on REALISTIC quantized data, then reuses the committed harness'
reference()/fused()/cosine() to compare. This is the true B1/topk1024 split-corruption
canary (cross-split LSE reduction in combine.cu).
"""
import argparse
import math
import torch
import tensorrt_llm  # noqa: F401

import importlib.util
spec = importlib.util.spec_from_file_location(
    "vbh", "/src/.claude_artifacts/vb_fusion/validate_vb_fusion.py")
vbh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vbh)

H_KV, KV_BYTES, SCALE_BYTES, PAGE, D_QK, KV_LORA, V_HEAD_DIM, H_Q = (
    vbh.H_KV, vbh.KV_BYTES, vbh.SCALE_BYTES, vbh.PAGE, vbh.D_QK,
    vbh.KV_LORA, vbh.V_HEAD_DIM, vbh.H_Q)


def make_inputs_padfix(b, s_q, topk, dtype_w, device):
    torch.manual_seed(0)
    q = torch.randn(b, s_q, H_Q, D_QK, device=device, dtype=torch.bfloat16)
    num_pages = max(8, math.ceil((topk + PAGE) / PAGE) * b + 4)
    rows = num_pages * PAGE
    latent = torch.randn(rows, D_QK, device=device, dtype=torch.bfloat16) * 0.5
    packed, scales = torch.ops.trtllm.fp4_quantize(
        latent, torch.tensor(1.0, device=device), 16, False)
    kv = packed.reshape(-1)[:rows * KV_BYTES].view(
        num_pages, PAGE, H_KV, KV_BYTES).contiguous()
    # KEY FIX: scales may be padded to a 128-row-swizzled layout; take only the
    # leading rows*SCALE_BYTES bytes (row-major prefix is the unpadded data).
    need = rows * SCALE_BYTES
    sc = scales.reshape(-1)
    assert sc.numel() >= need, (sc.numel(), need)
    kv_scales = sc[:need].view(num_pages, PAGE, H_KV,
                               SCALE_BYTES).contiguous().to(torch.uint8)
    max_tok = rows
    indices = torch.randint(0, max_tok, (b, s_q, topk), dtype=torch.int32,
                            device=device)
    v_b_proj = (torch.randn(H_Q, V_HEAD_DIM, KV_LORA, device=device,
                            dtype=torch.float32) * 0.05).to(dtype_w)
    return q, kv, kv_scales, indices, v_b_proj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--s_q", type=int, default=1)
    ap.add_argument("--topk", type=int, default=1024)
    ap.add_argument("--cos-thresh", type=float, default=0.999)
    args = ap.parse_args()
    dtype_w = torch.bfloat16
    dev = "cuda"
    sm_scale = 1.0 / math.sqrt(D_QK)

    q, kv, kv_scales, indices, v_b_proj = make_inputs_padfix(
        args.batch, args.s_q, args.topk, dtype_w, dev)
    ref, _ = vbh.reference(q, kv, kv_scales, indices, v_b_proj, sm_scale, dtype_w)
    fus = vbh.fused(q, kv, kv_scales, indices, v_b_proj, sm_scale)
    assert fus.shape == ref.shape, (fus.shape, ref.shape)
    g_cos = vbh.cosine(fus, ref)
    row_min, row_mean = vbh.per_row_cosine(fus, ref)
    max_abs = (fus.float() - ref.float()).abs().max().item()
    has_nan = bool(torch.isnan(fus).any() or torch.isnan(ref).any())
    print(f"shape           : {tuple(fus.shape)}")
    print(f"any NaN (ref/fus): {has_nan}")
    print(f"global cosine   : {g_cos:.6f}")
    print(f"per-row cos min : {row_min:.6f}  mean: {row_mean:.6f}")
    print(f"max |abs diff|  : {max_abs:.4e}")
    ok = (not has_nan) and (g_cos >= args.cos_thresh) and (row_min >= args.cos_thresh)
    print("RESULT          :", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

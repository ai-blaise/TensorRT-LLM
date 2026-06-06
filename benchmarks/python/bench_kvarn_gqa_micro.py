#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Microbench for KVarN GQA packed tile store/restore primitives.

This is intentionally small and non-invasive: it allocates one 128-token tile per
KV head and measures the reference pack/restore path. It is not a production
throughput benchmark for the fused decode kernel; it gives the next CUDA/Triton
implementation a correctness/perf baseline with odd SMC draft query shapes.
"""

from __future__ import annotations

import argparse
import time

import torch

from tensorrt_llm._torch.attention_backend.kvarn_gqa import (
    KVarNGQAConfig,
    dequantize_gqa_tile,
    quantize_gqa_tile,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bench(fn, iters: int, device: torch.device) -> float:
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - start) * 1e6 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--sinkhorn-iters", type=int, default=4)
    parser.add_argument("--queries", type=int, nargs="+", default=[1, 5, 25])
    parser.add_argument("--min-cosine", type=float, default=0.55)
    parser.add_argument("--require-fused", action="store_true",
                        help="fail unless a fused KVarN GQA op is registered")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = KVarNGQAConfig(sinkhorn_iters=args.sinkhorn_iters)
    torch.manual_seed(11)
    k = torch.randn(cfg.group, args.kv_heads, cfg.head_dim, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    q_by_m = {
        m: torch.randn(m, args.kv_heads, cfg.head_dim, device=device, dtype=torch.float16)
        for m in args.queries
    }

    records = quantize_gqa_tile(k, v, cfg)
    k_restore, v_restore = dequantize_gqa_tile(records, cfg)
    k_cos = torch.nn.functional.cosine_similarity(
        k.float().flatten(), k_restore.float().flatten(), dim=0).item()
    v_cos = torch.nn.functional.cosine_similarity(
        v.float().flatten(), v_restore.float().flatten(), dim=0).item()
    if min(k_cos, v_cos) < args.min_cosine:
        raise SystemExit(
            f"KVarN GQA restore correctness below floor: k_cos={k_cos:.4f} "
            f"v_cos={v_cos:.4f} floor={args.min_cosine:.4f}")

    pack_us = _bench(lambda: quantize_gqa_tile(k, v, cfg), args.iters, device)
    restore_us = _bench(lambda: dequantize_gqa_tile(records, cfg), args.iters, device)
    trtllm_ops = getattr(torch.ops, "trtllm", object())
    fused_registered = (hasattr(trtllm_ops, "kvarn_gqa_store")
                        and hasattr(trtllm_ops, "kvarn_gqa_decode")
                        and hasattr(trtllm_ops, "kvarn_gqa_backend_ready"))
    fused_ready = False
    if fused_registered:
        try:
            fused_ready = bool(trtllm_ops.kvarn_gqa_backend_ready())
        except Exception:
            fused_ready = False
    if args.require_fused and not fused_ready:
        raise SystemExit(
            "--require-fused was set, but torch.ops.trtllm.kvarn_gqa_store, "
            "kvarn_gqa_decode, and kvarn_gqa_backend_ready() are not all "
            "present and production-ready; do not promote the reference path as fused")

    print(f"dtype={cfg.dtype} tile_bytes={cfg.tile_bytes_aligned} bytes_per_token_slot={cfg.bytes_per_token_slot}")
    print(f"restore_cosine_k={k_cos:.4f} restore_cosine_v={v_cos:.4f}")
    print(f"pack_us={pack_us:.2f} restore_us={restore_us:.2f} kv_heads={args.kv_heads} fused_registered={fused_registered} fused_ready={fused_ready}")
    for m, q in q_by_m.items():
        def score_once():
            k_restore, v_restore = dequantize_gqa_tile(records, cfg)
            logits = torch.einsum("mhd,thd->hmt", q.float(), k_restore.float())
            probs = torch.softmax(logits / (cfg.head_dim ** 0.5), dim=-1)
            return torch.einsum("hmt,thd->mhd", probs, v_restore.float())
        score_us = _bench(score_once, max(args.iters // 5, 1), device)
        print(f"odd_m={m} restore_plus_score_us={score_us:.2f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Attention-region direct bench for DeepSeek-V3.2-REAP-345B decode.

Times the sparse-MLA decode kernels at prod decode shapes (num_heads=128, kv_lora_rank=512,
qk_rope=64 -> head_dim=576, topk=1024, tokens_per_block=64), under CUDA-graph capture:
  A) flash_mla_sparse_fwd (bf16-KV FlashMLA sparse decode; the served fp8/bf16 generation read)
  B) sparse_mla_decode_nvfp4 (NVFP4-KV FlashMLA decode) with metadata-hoist gate OFF vs ON
     (TRTLLM_OPTRT_HOIST_SPARSE_MLA_META: skips the per-call serial tile-scheduler-metadata kernel
      on 15/16 F-layers/step -> candidate #3 cross-region per-op overhead lever).

Builds synthetic KV pool + global topk indices directly (no full attn-metadata). Correctness here is
self-consistency (hoist ON vs OFF must be bit-exact); cross-kernel cosine is NOT expected (different
KV dtypes/layouts). The deliverable is the per-layer attention COST to ground the decode system map.
"""
from __future__ import annotations
import os, sys, statistics, argparse
import torch
import torch.nn.functional as F

import tensorrt_llm  # noqa: F401
to = torch.ops.trtllm
try:
    from tensorrt_llm.flash_mla import flash_mla_sparse_fwd
except Exception:
    flash_mla_sparse_fwd = None

DEV = "cuda"


def _bench(launch, iters=200, blocks=10, warm=30):
    for _ in range(warm):
        launch()
    torch.cuda.synchronize()
    ms = []
    for _ in range(blocks):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            launch()
        e.record(); torch.cuda.synchronize()
        ms.append(s.elapsed_time(e) / iters * 1000.0)
    return statistics.median(ms), min(ms)


def _capture(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=128)
    ap.add_argument("--kv-lora", type=int, default=512)
    ap.add_argument("--qk-rope", type=int, default=64)
    ap.add_argument("--topk", type=int, default=1024)
    ap.add_argument("--tpb", type=int, default=64)  # tokens_per_block
    ap.add_argument("--seqlen", type=int, default=8192)  # KV length to size the pool
    ap.add_argument("--batches", type=str, default="1,8,32,64")
    args = ap.parse_args()

    head_dim = args.kv_lora + args.qk_rope  # 576
    sm_scale = 1.0 / (head_dim ** 0.5)
    print(f"DEVICE: {torch.cuda.get_device_name(0)}")
    print(f"heads={args.heads} kv_lora={args.kv_lora} qk_rope={args.qk_rope} head_dim={head_dim} "
          f"topk={args.topk} tpb={args.tpb} seqlen={args.seqlen}")
    print(f"flash_mla_sparse_fwd available: {flash_mla_sparse_fwd is not None}")
    print("=" * 100)

    # KV pool sized to hold seqlen tokens per sequence (max batch). Single layer view.
    # bf16 pool for flash_mla_sparse_fwd: [num_pages, tpb, 1, head_dim]
    max_b = max(int(x) for x in args.batches.split(","))
    num_pages = ((args.seqlen + args.tpb - 1) // args.tpb) * max_b + 8
    torch.manual_seed(0)

    for M in [int(x) for x in args.batches.split(",")]:
        # ---- A) flash_mla_sparse_fwd (bf16 KV) ---- API: q (s_q_total, h_q, d_qk),
        # kv (s_kv, h_kv=1, d_qk), indices (s_q_total, h_kv=1, topk).
        try:
            with torch.device("cuda:0"):
                s_kv = args.seqlen
                q_concat = torch.randn(M, args.heads, head_dim, dtype=torch.bfloat16, device=DEV)
                kv_flat = torch.randn(s_kv, 1, head_dim, dtype=torch.bfloat16, device=DEV) * 0.1
                idx = torch.randint(0, s_kv, (M, 1, args.topk), dtype=torch.int32, device=DEV)
                def call_a(_q=q_concat, _kv=kv_flat, _i=idx):
                    return flash_mla_sparse_fwd(_q, _kv, _i, sm_scale)[0]
                _ = call_a(); torch.cuda.synchronize()
                ga, _ = _capture(call_a)
                med_a, min_a = _bench(lambda: ga.replay())
        except Exception as ex:
            print("flash_mla:", type(ex).__name__, str(ex)[:90])
            med_a = min_a = float('nan')

        # ---- B) sparse_mla_decode_nvfp4 (NVFP4 KV), hoist OFF then ON ----
        def run_nvfp4(hoist):
            os.environ["TRTLLM_OPTRT_HOIST_SPARSE_MLA_META"] = "1" if hoist else "0"
            with torch.device("cuda:0"):
                s_q = 1
                q = torch.randn(M, s_q, args.heads, head_dim, dtype=torch.bfloat16, device=DEV)
                kv = torch.randint(0, 255, (num_pages, args.tpb, 1, head_dim // 2),
                                   dtype=torch.uint8, device=DEV)
                kv_scales = torch.randint(0, 255, (num_pages, args.tpb, 1, head_dim // 16),
                                          dtype=torch.uint8, device=DEV).view(torch.float8_e4m3fn)
                maxtok = num_pages * args.tpb
                indices = torch.randint(0, min(args.seqlen * M, maxtok), (M, s_q, args.topk),
                                        dtype=torch.int32, device=DEV)
                def call_b(_q=q, _kv=kv, _ks=kv_scales, _i=indices):
                    return to.sparse_mla_decode_nvfp4(_q, _kv, _ks, _i,
                                                      d_v=args.kv_lora, sm_scale=sm_scale)[0]
                out = call_b(); torch.cuda.synchronize()
                gb, _ = _capture(call_b)
                med, mn = _bench(lambda: gb.replay())
                return med, mn, out.float().flatten().clone()

        try:
            med_off, min_off, vec_off = run_nvfp4(False)
        except Exception as ex:
            import traceback; traceback.print_exc()
            med_off = min_off = float('nan'); vec_off = None
        try:
            med_on, min_on, vec_on = run_nvfp4(True)
            cos = (F.cosine_similarity(vec_on, vec_off, dim=0).item()
                   if (vec_on is not None and vec_off is not None) else float('nan'))
        except Exception as ex:
            med_on = min_on = float('nan'); cos = float('nan')

        print(f"M={M:>3} | flash_mla(bf16): {med_a:7.2f}us | nvfp4 hoist-OFF: {med_off:7.2f}us "
              f"| nvfp4 hoist-ON: {med_on:7.2f}us | hoist_win={med_off/med_on if med_on==med_on and med_on>0 else float('nan'):.3f}x "
              f"| hoist_cos={cos:.5f}")

    print("=" * 100)


if __name__ == "__main__":
    main()

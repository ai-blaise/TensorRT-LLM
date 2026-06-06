# SPDX-License-Identifier: Apache-2.0
"""Focused KVarN k2v2 MLA latent microbench.

Safe defaults are intentionally small. On a free B200, run for the production
shape with an isolated GPU, for example:

CUDA_VISIBLE_DEVICES=0 python benchmarks/python/bench_kvarn_k2v2_micro.py \
  --device cuda --blocks 128 --group 64 --iters 2
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from tensorrt_llm._torch.attention_backend.sparse.kvarn_backend import (
    KVarNLatentPool,
    parse_kvarn_dtype,
)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().reshape(1, -1), b.float().reshape(1, -1)
    ).item()


def _time(fn, repeat: int, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(end) / repeat
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / repeat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--group", type=int, default=64)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260606)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    cfg = parse_kvarn_dtype(
        "kvarn_k2v2", kv_lora_rank=512, qk_rope_head_dim=64, iters=args.iters
    )
    pool = KVarNLatentPool(args.blocks, args.group, cfg, device)
    ckv = torch.randn(args.blocks, args.group, cfg.kv_lora_rank, device=device, dtype=torch.float16) * 0.35
    kpe = torch.randn(args.blocks, args.group, cfg.qk_rope_head_dim, device=device, dtype=torch.float16) * 0.20
    ids = torch.arange(args.blocks, device=device)

    def store_all() -> None:
        for block_id in range(args.blocks):
            pool.store_block(block_id, ckv[block_id], kpe[block_id])

    store_all()

    def load_all() -> None:
        pool.load_blocks(ids)

    store_ms = _time(store_all, args.repeat, device)
    load_ms = _time(load_all, args.repeat, device)
    ckv_rt, kpe_rt = pool.load_blocks(ids)

    print(json.dumps({
        "dtype": cfg.name,
        "device": str(device),
        "blocks": args.blocks,
        "group": args.group,
        "iters": args.iters,
        "bytes_per_block": cfg.packed_bytes(args.group),
        "bits_per_elem": cfg.bits_per_elem(args.group),
        "store_ms": store_ms,
        "load_ms": load_ms,
        "cos_ckv": _cosine(ckv_rt, ckv),
        "cos_kpe": _cosine(kpe_rt, kpe),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

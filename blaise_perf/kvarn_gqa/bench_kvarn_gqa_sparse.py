#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Low-memory GQA KVarN packed store/decode/sparse-decode probe.

This script intentionally avoids importing the full tensorrt_llm Python package.
It builds the local THOP extension, packs a single 128-token K/V block, compares
fused sparse top-k decode against dense packed decode when top-k enumerates the
whole block, then reports latency for store, dense decode, sparse-full, and a
smaller sparse top-k path.  It is meant for the next isolated B200 GPU window.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def build_ops(repo: Path) -> None:
    libs = "/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/libs"
    load(
        name="kvarn_gqa_sparse_bench_ext",
        sources=[
            str(repo / "cpp/tensorrt_llm/thop/kvarnGqaOp.cpp"),
            str(repo / "cpp/tensorrt_llm/kernels/kvarnGqaKernels.cu"),
        ],
        extra_include_paths=[
            str(repo / "cpp/include"),
            str(repo / "cpp"),
            str(repo / "cpp/tensorrt_llm"),
            "/usr/local/tensorrt/include",
        ],
        extra_cflags=["-std=c++17"],
        extra_cuda_cflags=["-std=c++17", "-arch=sm_100"],
        extra_ldflags=[
            f"-L{libs}",
            f"-Wl,-rpath,{libs}",
            "-ltensorrt_llm",
            "-lth_common",
        ],
        is_python_module=False,
        verbose=False,
    )


def timed_us(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def capture_tensor_op(fn):
    eager = fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fn()
    graph.replay()
    torch.cuda.synchronize()
    return eager, captured, graph


def capture_store_op(fn, packed: torch.Tensor):
    fn()
    eager = packed.detach().clone()
    packed.zero_()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    packed.zero_()
    graph.replay()
    torch.cuda.synchronize()
    return eager, packed.detach().clone(), graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/workspace")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--m", type=int, nargs="+", default=[1, 5, 25])
    parser.add_argument("--sparse-topk", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=1,
                        help="resident committed 128-token packed blocks to store/read")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--graph-replay", action="store_true",
                        help="capture/replay store, dense decode, and sparse decode and compare against eager outputs")
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    repo = Path(args.repo)
    build_ops(repo)
    assert hasattr(torch.ops.trtllm, "kvarn_gqa_store")
    assert hasattr(torch.ops.trtllm, "kvarn_gqa_decode")
    assert hasattr(torch.ops.trtllm, "kvarn_gqa_decode_sparse")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = torch.device("cuda", args.device)
    group = 128
    head_dim = 128
    num_blocks = args.blocks
    if num_blocks <= 0:
        raise ValueError("--blocks must be positive")
    total_tokens = num_blocks * group
    torch.manual_seed(1234)

    k = torch.randn((num_blocks, group, args.kv_heads, head_dim), device=device, dtype=dtype)
    v = torch.randn_like(k)
    packed = torch.empty((num_blocks, 1, group, args.kv_heads, 76), device=device, dtype=torch.uint8)
    block_ids = torch.arange(num_blocks, device=device, dtype=torch.long)
    empty = torch.empty((0,), device=device, dtype=dtype)

    store = lambda: torch.ops.trtllm.kvarn_gqa_store(k, v, packed, block_ids, 0, head_dim, group)
    store()
    store_us = timed_us(store, args.iters, args.warmup)
    store()
    if args.graph_replay:
        eager_packed, replay_packed, _ = capture_store_op(store, packed)
        store_graph_diff = (replay_packed.to(torch.int16) - eager_packed.to(torch.int16)).abs().max().item()
        store()
        print(f"GRAPH_STORE dtype={args.dtype} max_abs_byte={store_graph_diff}")

    print(f"STORE dtype={args.dtype} blocks={num_blocks} kv_heads={args.kv_heads} store_us={store_us:.2f}")
    for m in args.m:
        q = torch.randn((m, args.heads, head_dim), device=device, dtype=dtype)
        seq_lens = torch.full((m,), total_tokens, device=device, dtype=torch.int32)
        dense = lambda: torch.ops.trtllm.kvarn_gqa_decode(
            q, packed, block_ids, empty, empty, empty, empty, seq_lens,
            args.heads, args.kv_heads, head_dim, group)
        sparse_full = None
        if total_tokens <= 256:
            sparse_full_idx = torch.arange(total_tokens, device=device, dtype=torch.long).view(1, 1, total_tokens)
            sparse_full_idx = sparse_full_idx.expand(args.kv_heads, m, total_tokens).contiguous()
            sparse_full = lambda: torch.ops.trtllm.kvarn_gqa_decode_sparse(
                q, packed, block_ids, empty, empty, empty, empty, seq_lens, sparse_full_idx,
                args.heads, args.kv_heads, head_dim, group)
        topk = min(args.sparse_topk, total_tokens)
        sparse_idx = torch.arange(topk, device=device, dtype=torch.long).view(1, 1, topk)
        sparse_idx = sparse_idx.expand(args.kv_heads, m, topk).contiguous()
        sparse = lambda: torch.ops.trtllm.kvarn_gqa_decode_sparse(
            q, packed, block_ids, empty, empty, empty, empty, seq_lens, sparse_idx,
            args.heads, args.kv_heads, head_dim, group)

        ref = dense()
        if sparse_full is not None:
            got = sparse_full()
            torch.cuda.synchronize()
            max_abs = (got - ref).abs().max().item()
        else:
            torch.cuda.synchronize()
            max_abs = float("nan")
        dense_us = timed_us(dense, args.iters, args.warmup)
        sparse_full_us = timed_us(sparse_full, args.iters, args.warmup) if sparse_full is not None else float("nan")
        sparse_us = timed_us(sparse, args.iters, args.warmup)
        if args.graph_replay:
            eager_dense, replay_dense, _ = capture_tensor_op(dense)
            if sparse_full is not None:
                eager_sparse_full, replay_sparse_full, _ = capture_tensor_op(sparse_full)
                graph_sparse_full = (replay_sparse_full - eager_sparse_full).abs().max().item()
            else:
                graph_sparse_full = float("nan")
            eager_sparse, replay_sparse, _ = capture_tensor_op(sparse)
            graph_dense = (replay_dense - eager_dense).abs().max().item()
            graph_sparse = (replay_sparse - eager_sparse).abs().max().item()
            print(
                f"GRAPH_DECODE dtype={args.dtype} M={m} topk={topk} "
                f"dense_replay_max_abs={graph_dense:.6f} "
                f"sparse_full_replay_max_abs={graph_sparse_full:.6f} "
                f"sparse_topk_replay_max_abs={graph_sparse:.6f}"
            )
        print(
            f"DECODE dtype={args.dtype} blocks={num_blocks} M={m} topk={topk} max_abs_full={max_abs:.6f} "
            f"dense_us={dense_us:.2f} sparse_full_us={sparse_full_us:.2f} sparse_topk_us={sparse_us:.2f}"
        )


if __name__ == "__main__":
    main()

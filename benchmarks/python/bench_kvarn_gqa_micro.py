#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Microbench and parity harness for SMC-SD/GQA KVarN.

This script is intentionally safe by default: without ``--try-*`` it only runs
small reference pack/restore/score timings. The ``--try-store-op``,
``--try-decode-op``, and ``--try-side-op`` switches exercise the experimental
THOP kernels against a torch reference for FP16/BF16, odd SMC query counts,
compact records, paged KV-cache layout, and sink/tail partial-block sequences.
Those switches require a built TRT-LLM torch extension and an idle CUDA device.
"""

from __future__ import annotations

import argparse
import time

import torch

from tensorrt_llm._torch.attention_backend.kvarn_gqa import (
    KVarNGQAConfig,
    dequantize_gqa_tile,
    dequantize_gqa_tiles,
    quantize_gqa_tile,
)
from tensorrt_llm._torch.attention_backend.kvarn_gqa_attention import (
    _KVarNGQASidePool,
    _write_record_to_page,
)

_RUNTIME_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


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


def _records_to_container(records: torch.Tensor, cfg: KVarNGQAConfig, layout: str) -> torch.Tensor:
    if layout == "compact":
        return records.unsqueeze(0).contiguous()
    if layout != "paged":
        raise ValueError(f"unexpected layout {layout!r}")
    kv_heads = records.shape[0]
    return (records.reshape(kv_heads, cfg.group, cfg.bytes_per_token_slot)
            .permute(1, 0, 2)
            .unsqueeze(0)
            .unsqueeze(1)
            .contiguous())


def _container_to_records(container: torch.Tensor, cfg: KVarNGQAConfig, layout: str) -> torch.Tensor:
    if layout == "compact":
        return container[0].contiguous()
    if layout != "paged":
        raise ValueError(f"unexpected layout {layout!r}")
    return (container[0, 0]
            .permute(1, 0, 2)
            .contiguous()
            .reshape(container.size(3), cfg.tile_bytes_aligned))


def _empty_side(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty((0,), device=device, dtype=dtype)


def _ref_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cfg: KVarNGQAConfig) -> torch.Tensor:
    logits = torch.einsum("mhd,thd->hmt", q.float(), k.float())
    probs = torch.softmax(logits / (cfg.head_dim ** 0.5), dim=-1)
    return torch.einsum("hmt,thd->mhd", probs, v.float())


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--sinkhorn-iters", type=int, default=4)
    parser.add_argument("--runtime-dtype", choices=sorted(_RUNTIME_DTYPES), default="fp16")
    parser.add_argument("--layouts", nargs="+", choices=["compact", "paged"], default=["compact", "paged"])
    parser.add_argument("--queries", type=int, nargs="+", default=[1, 5, 25])
    parser.add_argument("--sink-side-tokens", type=int, default=16)
    parser.add_argument("--tail-side-tokens", type=int, default=7)
    parser.add_argument("--min-cosine", type=float, default=0.55)
    parser.add_argument("--require-fused", action="store_true",
                        help="fail unless a production-ready fused KVarN GQA backend is registered")
    parser.add_argument("--try-decode-op", action="store_true",
                        help="run the experimental decode op even while backend_ready() is false")
    parser.add_argument("--try-store-op", action="store_true",
                        help="run the experimental store op even while backend_ready() is false")
    parser.add_argument("--try-side-op", action="store_true",
                        help="exercise decode over fp16/bf16 sink + packed full block + fp16/bf16 tail")
    parser.add_argument("--decode-op-atol", type=float, default=7.5e-2)
    parser.add_argument("--store-op-atol", type=float, default=1.25e-1)
    parser.add_argument("--bdr-working-set-blocks", type=int, default=16,
                        help="committed full blocks to include in the BDR restore timing")
    parser.add_argument("--bdr-churn-blocks", type=int, default=1,
                        help="committed physical blocks to mark stale each BDR churn timing iteration")
    parser.add_argument("--decode-op-iters", type=int, default=20,
                        help="iterations for CUDA packed decode op timing when --try-decode-op/--try-side-op is set")
    args = parser.parse_args()

    device = torch.device(args.device)
    runtime_dtype = _RUNTIME_DTYPES[args.runtime_dtype]
    cfg = KVarNGQAConfig(sinkhorn_iters=args.sinkhorn_iters)
    torch.manual_seed(11)
    k = torch.randn(cfg.group, args.kv_heads, cfg.head_dim, device=device, dtype=runtime_dtype)
    v = torch.randn_like(k)
    q_by_m = {
        m: torch.randn(m, args.kv_heads, cfg.head_dim, device=device, dtype=runtime_dtype)
        for m in args.queries
    }

    records = quantize_gqa_tile(k, v, cfg)
    k_restore, v_restore = dequantize_gqa_tile(records, cfg)
    k_cos = torch.nn.functional.cosine_similarity(
        k.float().flatten(), k_restore.flatten(), dim=0).item()
    v_cos = torch.nn.functional.cosine_similarity(
        v.float().flatten(), v_restore.flatten(), dim=0).item()
    if min(k_cos, v_cos) < args.min_cosine:
        raise SystemExit(
            f"KVarN GQA restore correctness below floor: k_cos={k_cos:.4f} "
            f"v_cos={v_cos:.4f} floor={args.min_cosine:.4f}")

    pack_us = _bench(lambda: quantize_gqa_tile(k, v, cfg), args.iters, device)
    restore_us = _bench(lambda: dequantize_gqa_tile(records, cfg), args.iters, device)

    # BDR fold reference timing: packed pages remain authoritative, while the
    # persistent readable pool is restored only for physical blocks whose commit
    # generation changed. This mirrors the production CUDA kernel gate: full
    # restore scales with the working set, steady restore is metadata-only, and
    # churn restore scales with changed blocks.
    bdr_blocks = max(1, args.bdr_working_set_blocks)
    bdr_churn = max(0, min(args.bdr_churn_blocks, bdr_blocks))
    bdr_pages = torch.zeros((bdr_blocks + 1, 1, cfg.group, args.kv_heads,
                             cfg.bytes_per_token_slot),
                            device=device, dtype=torch.uint8)
    bdr_state = _KVarNGQASidePool(cfg, num_layers=1, max_batch_size=1,
                                  max_blocks_per_seq=bdr_blocks + 1,
                                  num_kv_heads=args.kv_heads,
                                  dtype=runtime_dtype, device=device)
    bdr_state.ensure_bdr_pool(bdr_pages.shape[0])
    bdr_slot = bdr_state.slot_for_request(11)
    bdr_block_ids = list(range(bdr_blocks + 1))
    for logical_block in range(1, bdr_blocks + 1):
        _write_record_to_page(bdr_pages[logical_block, 0], records, cfg)
        bdr_state.mark_committed(0, bdr_slot, 11, logical_block * cfg.group,
                                 physical_block_id=logical_block)
    bdr_seq_len = (bdr_blocks + 1) * cfg.group
    bdr_state.restore_committed_blocks_amortized(
        0, bdr_slot, bdr_block_ids, bdr_seq_len, bdr_pages, amortize=True)

    bdr_full_restore_us = _bench(
        lambda: bdr_state.restore_committed_blocks_amortized(
            0, bdr_slot, bdr_block_ids, bdr_seq_len, bdr_pages, amortize=False),
        max(args.iters // 10, 1), device)
    bdr_steady_restore_us = _bench(
        lambda: bdr_state.restore_committed_blocks_amortized(
            0, bdr_slot, bdr_block_ids, bdr_seq_len, bdr_pages, amortize=True),
        args.iters, device)

    def bump_and_restore_churn():
        if bdr_churn:
            churn_ids = torch.arange(1, bdr_churn + 1, device=device, dtype=torch.long)
            bdr_state.physical_commit_gen[0, churn_ids] += 1
        return bdr_state.restore_committed_blocks_amortized(
            0, bdr_slot, bdr_block_ids, bdr_seq_len, bdr_pages, amortize=True)

    bdr_churn_restore_us = _bench(bump_and_restore_churn,
                                  max(args.iters // 10, 1), device)
    trtllm_ops = getattr(torch.ops, "trtllm", object())
    fused_registered = (hasattr(trtllm_ops, "kvarn_gqa_store")
                        and hasattr(trtllm_ops, "kvarn_gqa_decode")
                        and hasattr(trtllm_ops, "kvarn_gqa_dequant_amortized")
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
            "kvarn_gqa_decode, kvarn_gqa_dequant_amortized, and kvarn_gqa_backend_ready() are not all "
            "present and production-ready; do not promote the reference path as fused")

    if args.try_store_op or args.try_decode_op or args.try_side_op:
        if device.type != "cuda":
            raise SystemExit("--try-store-op/--try-decode-op/--try-side-op require a CUDA device")
        if args.try_store_op and not hasattr(trtllm_ops, "kvarn_gqa_store"):
            raise SystemExit("torch.ops.trtllm.kvarn_gqa_store is not registered")
        if (args.try_decode_op or args.try_side_op) and not hasattr(trtllm_ops, "kvarn_gqa_decode"):
            raise SystemExit("torch.ops.trtllm.kvarn_gqa_decode is not registered")
        block_ids = torch.zeros((1,), device=device, dtype=torch.int64)
        empty_side = _empty_side(device, runtime_dtype)

    if args.try_store_op:
        if cfg.sinkhorn_iters != 16:
            raise SystemExit("--try-store-op compares against the C++ store preset and requires --sinkhorn-iters 16")
        for layout in args.layouts:
            op_container = torch.zeros_like(_records_to_container(records, cfg, layout))
            trtllm_ops.kvarn_gqa_store(k.unsqueeze(0).contiguous(), v.unsqueeze(0).contiguous(),
                                       op_container, block_ids, 0, cfg.head_dim, cfg.group)
            op_records = _container_to_records(op_container, cfg, layout)
            op_k, op_v = dequantize_gqa_tile(op_records, cfg)
            ref_k, ref_v = dequantize_gqa_tile(records, cfg)
            store_max_abs = max(_max_abs(op_k, ref_k), _max_abs(op_v, ref_v))
            if store_max_abs > args.store_op_atol:
                raise SystemExit(
                    f"store op restore mismatch layout={layout} dtype={args.runtime_dtype}: "
                    f"max_abs={store_max_abs:.6f} atol={args.store_op_atol:.6f}")
            store_us = _bench(lambda: trtllm_ops.kvarn_gqa_store(
                k.unsqueeze(0).contiguous(), v.unsqueeze(0).contiguous(),
                op_container, block_ids, 0, cfg.head_dim, cfg.group),
                max(args.decode_op_iters, 1), device)
            print(f"store_op_layout={layout} dtype={args.runtime_dtype} "
                  f"restore_max_abs={store_max_abs:.6f} store_us={store_us:.2f}")

    if args.try_decode_op:
        for layout in args.layouts:
            packed_container = _records_to_container(records, cfg, layout)
            for m, q in q_by_m.items():
                seq_lens = torch.full((m,), cfg.group, device=device, dtype=torch.int32)
                q_contig = q.contiguous()
                op_out = trtllm_ops.kvarn_gqa_decode(
                    q_contig, packed_container, block_ids, empty_side, empty_side,
                    empty_side, empty_side, seq_lens, args.kv_heads, args.kv_heads,
                    cfg.head_dim, cfg.group)
                ref_out = _ref_attention(q, k_restore, v_restore, cfg)
                max_abs = _max_abs(op_out, ref_out)
                if max_abs > args.decode_op_atol:
                    raise SystemExit(
                        f"decode op mismatch layout={layout} dtype={args.runtime_dtype} odd_m={m}: "
                        f"max_abs={max_abs:.6f} atol={args.decode_op_atol:.6f}")
                decode_us = _bench(lambda: trtllm_ops.kvarn_gqa_decode(
                    q_contig, packed_container, block_ids, empty_side, empty_side,
                    empty_side, empty_side, seq_lens, args.kv_heads, args.kv_heads,
                    cfg.head_dim, cfg.group), max(args.decode_op_iters, 1), device)
                ref_us = _bench(lambda: _ref_attention(q, k_restore, v_restore, cfg),
                                max(args.decode_op_iters, 1), device)
                print(f"decode_op_layout={layout} dtype={args.runtime_dtype} odd_m={m} "
                      f"max_abs={max_abs:.6f} decode_op_us={decode_us:.2f} "
                      f"restore_score_ref_us={ref_us:.2f}")

    if args.try_side_op:
        sink_tokens = max(0, min(args.sink_side_tokens, cfg.sink_tokens))
        tail_tokens = max(0, min(args.tail_side_tokens, cfg.group - 1))
        sink_k = torch.randn(1, sink_tokens, args.kv_heads, cfg.head_dim, device=device, dtype=runtime_dtype)
        sink_v = torch.randn_like(sink_k)
        tail_k = torch.randn(1, tail_tokens, args.kv_heads, cfg.head_dim, device=device, dtype=runtime_dtype)
        tail_v = torch.randn_like(tail_k)
        all_k = torch.cat([sink_k[0].float(), k_restore.float(), tail_k[0].float()], dim=0)
        all_v = torch.cat([sink_v[0].float(), v_restore.float(), tail_v[0].float()], dim=0)
        side_seq_len = sink_tokens + cfg.group + tail_tokens
        for layout in args.layouts:
            packed_container = _records_to_container(records, cfg, layout)
            for m, q in q_by_m.items():
                seq_lens = torch.full((m,), side_seq_len, device=device, dtype=torch.int32)
                q_contig = q.contiguous()
                sink_k_m = sink_k.expand(m, -1, -1, -1).contiguous()
                sink_v_m = sink_v.expand(m, -1, -1, -1).contiguous()
                tail_k_m = tail_k.expand(m, -1, -1, -1).contiguous()
                tail_v_m = tail_v.expand(m, -1, -1, -1).contiguous()
                op_out = trtllm_ops.kvarn_gqa_decode(
                    q_contig, packed_container, block_ids,
                    sink_k_m, sink_v_m, tail_k_m, tail_v_m,
                    seq_lens, args.kv_heads, args.kv_heads, cfg.head_dim, cfg.group)
                ref_out = _ref_attention(q, all_k, all_v, cfg)
                max_abs = _max_abs(op_out, ref_out)
                if max_abs > args.decode_op_atol:
                    raise SystemExit(
                        f"side decode op mismatch layout={layout} dtype={args.runtime_dtype} odd_m={m}: "
                        f"max_abs={max_abs:.6f} atol={args.decode_op_atol:.6f}")
                decode_us = _bench(lambda: trtllm_ops.kvarn_gqa_decode(
                    q_contig, packed_container, block_ids,
                    sink_k_m, sink_v_m, tail_k_m, tail_v_m,
                    seq_lens, args.kv_heads, args.kv_heads, cfg.head_dim, cfg.group),
                    max(args.decode_op_iters, 1), device)
                ref_us = _bench(lambda: _ref_attention(q, all_k, all_v, cfg),
                                max(args.decode_op_iters, 1), device)
                print(f"side_decode_op_layout={layout} dtype={args.runtime_dtype} odd_m={m} "
                      f"seq_len={side_seq_len} max_abs={max_abs:.6f} "
                      f"decode_op_us={decode_us:.2f} restore_score_ref_us={ref_us:.2f}")

    print(f"dtype={cfg.dtype} runtime_dtype={args.runtime_dtype} tile_bytes={cfg.tile_bytes_aligned} bytes_per_token_slot={cfg.bytes_per_token_slot}")
    print(f"restore_cosine_k={k_cos:.4f} restore_cosine_v={v_cos:.4f}")
    print(f"pack_us={pack_us:.2f} restore_us={restore_us:.2f} kv_heads={args.kv_heads} fused_registered={fused_registered} fused_ready={fused_ready}")
    print(f"bdr_working_set_blocks={bdr_blocks} bdr_churn_blocks={bdr_churn} full_restore_us={bdr_full_restore_us:.2f} steady_restore_us={bdr_steady_restore_us:.2f} churn_restore_us={bdr_churn_restore_us:.2f}")
    for m, q in q_by_m.items():
        def score_once():
            return _ref_attention(q, k_restore, v_restore, cfg)
        score_us = _bench(score_once, max(args.iters // 5, 1), device)
        print(f"odd_m={m} restore_plus_score_us={score_us:.2f}")


if __name__ == "__main__":
    main()

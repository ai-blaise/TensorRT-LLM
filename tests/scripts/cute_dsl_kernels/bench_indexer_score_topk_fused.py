# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone microbench: Indexer decode score -> top-k pipeline (prod CuTe-DSL path).

Measures the per-step launch count + median us/step of the production decode
indexer pipeline at REAP decode shapes, and establishes the correctness
baseline (selected top-k index SET + gathered latent-KV) so a fused variant
can be compared bit/cosine-identically.

Prod decode pipeline (tensorrt_llm/_torch/attention_backend/sparse/dsa.py:3956-4097):
  1. cute_dsl_fp4_paged_mqa_logits(q, sf_q, kv, w, ctx_lens, bt, sched, width)
        -> logits_decode [B*next_n, width] fp32   (SCORE)
  2. cute_dsl_indexer_topk_decode(logits_decode, gen_kv_lens, out_idx,
        index_topk, next_n, single_pass_multi_cta=True,
        single_pass_multi_cta_cluster=True)        (MASK folded in + TOP-K)
  3. gather selected latent-KV (downstream; modelled here for output-equality)

NOTE: the mask (kv-length boundary) is folded INTO the top-k kernel via
gen_kv_lens (the cluster kernel walks only [0, length) per row). There is NO
separate mask launch on the prod decode path -> already 2 launches.

The FP4 quant + schedule-metadata helpers are imported verbatim from
``run_fp4.py`` in this directory (single source of truth).

Run (in the wave2 image, GPU device=4):
  python bench_indexer_score_topk_fused.py --variant baseline
  python bench_indexer_score_topk_fused.py --variant fused      # if FUSED op present
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paged_mqa_logits.run_fp4 import (  # noqa: E402
    _compute_schedule_metadata,
    _kv_cache_cast_to_fp4,
    _per_token_cast_to_fp4,
    _cast_back_from_fp4,
)

import tensorrt_llm  # noqa: E402,F401  (registers torch.ops.trtllm)

# ---------------------------------------------------------------------------
# Prod REAP indexer dims (dsa.py: index_n_heads=64, index_head_dim=128).
# index_topk: task spec = 1024 for REAP prod (DeepSeek-V3.2 default 2048 also
# swept). phys_block_kv=128 is the prod fused-KV page (FP4 chunk-layout atom).
# ---------------------------------------------------------------------------
N_HEADS = 64
HEAD_DIM = 128
PHYS_BLOCK_KV = 128
NUM_SMS = 148


def _next_pow2(x: int) -> int:
    b = 1
    while b < x:
        b <<= 1
    return b


def _indexer_logits_width(max_gen_kv_len: int, hard_cap: int) -> int:
    """Mirror dsa.py:_indexer_logits_width (pow2 bucket clamped to hard_cap)."""
    if max_gen_kv_len <= 0 or hard_cap <= 0:
        return hard_cap
    if max_gen_kv_len > hard_cap // 2:
        return hard_cap
    bucket = _next_pow2(max_gen_kv_len)
    return bucket if bucket < hard_cap else hard_cap


def make_inputs(batch_size, next_n, kv_len, seed=42, device="cuda"):
    """Build FP4 indexer inputs at a fixed kv_len (all rows same length, the
    decode-graph common case where kv_lens_cuda_2d broadcasts a single band)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    context_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)

    n_blk = (kv_len + PHYS_BLOCK_KV - 1) // PHYS_BLOCK_KV
    num_total_blocks = n_blk * batch_size + batch_size * 2
    block_table = torch.zeros((batch_size, n_blk), dtype=torch.int32, device=device)
    pool = torch.randperm(num_total_blocks, device=device, dtype=torch.int32)
    off = 0
    for i in range(batch_size):
        block_table[i, :n_blk] = pool[off:off + n_blk]
        off += n_blk

    q = torch.randn((batch_size, next_n, N_HEADS, HEAD_DIM), device=device, dtype=torch.bfloat16)
    kv_cache = torch.randn((num_total_blocks, PHYS_BLOCK_KV, 1, HEAD_DIM),
                           device=device, dtype=torch.bfloat16)
    weights = torch.randn((batch_size * next_n, N_HEADS), device=device, dtype=torch.float32)

    q_packed, sf_q_packed = _per_token_cast_to_fp4(q.view(-1, HEAD_DIM), gran_k=32)
    q_fp4 = q_packed.view(torch.uint8).view(batch_size, next_n, N_HEADS, HEAD_DIM // 2)
    sf_q = sf_q_packed.view(torch.int32).view(batch_size, next_n, N_HEADS)

    # FP4 fused KV (prod uses online SF transpose -> remove_online_sf_transpose=False).
    kv_fused, kv_sim = _kv_cache_cast_to_fp4(kv_cache, remove_online_sf_transpose=False)

    return dict(
        context_lens=context_lens,
        block_table=block_table,
        q_fp4=q_fp4,
        sf_q=sf_q,
        weights=weights,
        kv_fused=kv_fused,
        kv_sim=kv_sim,         # bf16 dequant view of the fp4 KV, for ref gather
        batch_size=batch_size,
        next_n=next_n,
        kv_len=kv_len,
    )


def prep_dsl_score_args(inp, hard_cap):
    """Reproduce dsa.py's FP4 DSL-path tensor prep for cute_dsl_fp4_paged_mqa_logits.

    All host-side work (schedule build via .cpu(), reshapes) happens HERE, once,
    outside any captured/timed region -- mirroring prod, where the schedule is a
    graph-frozen input built once per step in Indexer.prepare(). The returned
    tuple is then fed to the op with zero host syncs (graph-capture safe)."""
    bs, nn = inp["batch_size"], inp["next_n"]
    q_fp4 = inp["q_fp4"]
    # uint8 reinterpret (dsa.py: q came via FP8 plumbing as int8/uint8).
    dsl_q = q_fp4.view(torch.uint8)
    decode_q_scale = inp["sf_q"].view(bs, nn, N_HEADS)
    kv_fused_flat = inp["kv_fused"]  # [num_blocks, phys, 1, D//2+4]
    context_lens = inp["context_lens"]
    block_table = inp["block_table"]

    width = _indexer_logits_width(inp["kv_len"], hard_cap)

    # Schedule metadata (prod builds once/step via get_paged_mqa_logits_metadata).
    sched = _compute_schedule_metadata(context_lens.cpu(), 128, NUM_SMS).to(context_lens.device)
    return dict(dsl_q=dsl_q, sf_q=decode_q_scale, kv=kv_fused_flat,
                w=inp["weights"], cl=context_lens, bt=block_table, sched=sched, width=width)


def run_score(sa, out_dtype=torch.float32):
    """Pure-launch: only the op call (graph-capture safe). sa from prep_dsl_score_args.

    out_dtype selects the logits element type the score kernel emits. fp32 is
    prod default; bf16/fp16 halve the score->topk round-trip AND let the top-k
    radix kernel run 2 rounds instead of fp32's 4 (the dominant decode stage).
    epi_dtype is kept fp32 (max accumulation precision) regardless.
    """
    return torch.ops.trtllm.cute_dsl_fp4_paged_mqa_logits(
        sa["dsl_q"], sa["sf_q"], sa["kv"], sa["w"], sa["cl"], sa["bt"], sa["sched"],
        sa["width"], 1, torch.float32, out_dtype, False)


def run_topk(logits, gen_kv_lens, out_idx, index_topk, next_n):
    torch.ops.trtllm.cute_dsl_indexer_topk_decode(
        logits, gen_kv_lens, out_idx, index_topk, next_n,
        single_pass_multi_cta=True, single_pass_multi_cta_cluster=True)


# ---------------------------------------------------------------------------
# Reference: gather selected latent-KV given the top-k indices, to allow
# output bit/cosine comparison between baseline & fused (the indices ARE the
# load-bearing output of the indexer; the gather is downstream sparse-MLA).
# ---------------------------------------------------------------------------
def gather_selected_kv(inp, out_idx):
    """Gather the dequantized KV rows at the selected indices (per row), in
    ascending-index order so the result is a function of the index SET only.

    The indexer's load-bearing output is the SELECTED SET; the downstream
    sparse-MLA gathers those KV rows and does a softmax-weighted sum, which is
    order-invariant. So we sort each row's valid indices ascending before
    gathering: two runs that select the identical set (IoU=1) -> bit-identical
    gathered tensors, regardless of the within-top-k emission order.

    out_idx: [B*next_n, index_topk] int32, local kv positions in [0, kv_len),
    -1 = padding. Returns [B*next_n, index_topk, HEAD_DIM] bf16 (zeros for -1).
    """
    bs, nn = inp["batch_size"], inp["next_n"]
    kv_sim = inp["kv_sim"]            # [num_blocks, phys, 1, HEAD_DIM] bf16
    block_table = inp["block_table"]  # [B, n_blk] int32
    rows = bs * nn
    K = out_idx.shape[1]

    idx = out_idx.clone().long()                       # [rows, K]
    # Push -1 padding to the end and sort valid indices ascending (set-canonical).
    idx_sorted = idx.masked_fill(idx < 0, torch.iinfo(torch.int64).max).sort(dim=1).values
    valid = idx_sorted < torch.iinfo(torch.int64).max
    idx_c = idx_sorted.clamp(0, kv_sim.shape[0] * PHYS_BLOCK_KV - 1)
    batch_of_row = (torch.arange(rows, device=idx.device) // nn)  # [rows]

    pages = idx_c // PHYS_BLOCK_KV                     # [rows, K]
    offs = idx_c % PHYS_BLOCK_KV
    phys = block_table[batch_of_row].gather(1, pages)  # [rows, K]
    gathered = kv_sim[phys.reshape(-1), offs.reshape(-1), 0]  # [rows*K, HEAD_DIM]
    gathered = gathered.view(rows, K, HEAD_DIM)
    gathered = gathered * valid.unsqueeze(-1)
    return gathered


def selected_set_iou(a_idx, b_idx):
    """Per-row IoU of the selected index SETS (excluding -1). Returns min IoU."""
    rows = a_idx.shape[0]
    ious = []
    for r in range(rows):
        sa = set(x for x in a_idx[r].cpu().tolist() if x >= 0)
        sb = set(x for x in b_idx[r].cpu().tolist() if x >= 0)
        if not sa and not sb:
            ious.append(1.0)
            continue
        inter = len(sa & sb)
        union = len(sa | sb)
        ious.append(inter / union if union else 1.0)
    return min(ious), sum(ious) / len(ious)


def cosine(a, b):
    a = a.double().reshape(-1)
    b = b.double().reshape(-1)
    denom = (a.norm() * b.norm())
    if denom == 0:
        return 1.0
    return float((a @ b) / denom)


# ---------------------------------------------------------------------------
# CUDA-graph timing
# ---------------------------------------------------------------------------
def time_graphed(fn, iters=200, warmup=30):
    """Capture fn() into a CUDA graph and return median us/replay."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()

    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        g.replay()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(starts[i].elapsed_time(ends[i]) * 1000.0 for i in range(iters))  # us
    return times[len(times) // 2], g


_DT = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def _select(sa, gen_kv_lens, index_topk, nn, out_dtype, dev):
    """Run score(out_dtype)->topk eagerly; return (logits, out_idx, width)."""
    logits = run_score(sa, out_dtype=out_dtype)
    rows = logits.shape[0]
    out_idx = torch.full((rows, index_topk), -1, dtype=torch.int32, device=dev)
    run_topk(logits, gen_kv_lens, out_idx, index_topk, nn)
    torch.cuda.synchronize()
    return logits, out_idx, sa["width"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--next_n", type=int, default=1)
    ap.add_argument("--index_topk", type=int, default=1024)
    ap.add_argument("--kv_lens", type=int, nargs="+", default=[8192, 33000, 66000])
    ap.add_argument("--hard_cap", type=int, default=132096, help="kv_cache max_seq_len")
    ap.add_argument("--logits_dtype", choices=_DT.keys(), default="fp32",
                    help="score-kernel logits emit dtype (the A/B variable)")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--eager", action="store_true",
                    help="also time the eager (non-graph) pipe to bound launch overhead")
    args = ap.parse_args()

    out_dtype = _DT[args.logits_dtype]
    dev = torch.device("cuda")
    print(f"device={torch.cuda.get_device_name(0)} logits_dtype={args.logits_dtype} "
          f"B={args.batch_size} next_n={args.next_n} index_topk={args.index_topk} "
          f"hard_cap={args.hard_cap}")
    print(f"{'kv_len':>7} {'width':>7} {'rows':>5} {'logitsMB':>9} "
          f"{'score_us':>9} {'topk_us':>8} {'pipe_us':>8} {'sumstg_us':>9} "
          f"{'eager_us':>8} {'IoUvsFP32':>9} {'IoUvsRef':>9} {'cos':>8} {'verdict':>8}")

    all_pass = True
    for kv_len in args.kv_lens:
        inp = make_inputs(args.batch_size, args.next_n, kv_len, device=dev)
        bs, nn = args.batch_size, args.next_n
        rows = bs * nn
        gen_kv_lens = inp["context_lens"]   # decode top-k uses kv_lens (total cache len)
        sa = prep_dsl_score_args(inp, args.hard_cap)
        width = sa["width"]

        # ---- fp32 reference selection (prod baseline) ----
        logits_fp32, idx_fp32, _ = _select(sa, gen_kv_lens, args.index_topk, nn,
                                            torch.float32, dev)
        # PyTorch reference top-k on the fp32 logits (mask to live kv).
        positions = torch.arange(width, device=dev).unsqueeze(0).expand(rows, -1)
        row_idx = torch.arange(rows, device=dev) // nn
        nn_off = torch.arange(rows, device=dev) % nn
        row_end = (gen_kv_lens[row_idx] - nn + nn_off + 1).unsqueeze(1)
        masked = logits_fp32.float().masked_fill(positions >= row_end, float("-inf"))
        ref_idx = masked.topk(min(args.index_topk, width), dim=-1)[1].to(torch.int32)
        ref_idx = ref_idx.masked_fill(ref_idx >= row_end, -1)

        # ---- variant selection (logits_dtype under test) ----
        if out_dtype == torch.float32:
            logits, out_idx = logits_fp32, idx_fp32
        else:
            logits, out_idx, _ = _select(sa, gen_kv_lens, args.index_topk, nn,
                                         out_dtype, dev)
        logits_mb = logits.numel() * logits.element_size() / 1e6

        # IoU of variant-selection vs prod fp32-selection (the real A/B), and
        # vs the torch reference. cos on the set-canonical gathered KV.
        iou_vs_fp32, _ = selected_set_iou(out_idx, idx_fp32)
        iou_vs_ref, _ = selected_set_iou(out_idx, ref_idx)
        cos = cosine(gather_selected_kv(inp, out_idx), gather_selected_kv(inp, idx_fp32))

        # ---- timing: per-stage + full pipe under CUDA graph (variant dtype) ----
        out_idx_t = torch.full((rows, args.index_topk), -1, dtype=torch.int32, device=dev)
        score_us, _ = time_graphed(lambda: run_score(sa, out_dtype=out_dtype), iters=args.iters)
        topk_us, _ = time_graphed(
            lambda: run_topk(logits, gen_kv_lens, out_idx_t, args.index_topk, nn),
            iters=args.iters)
        pipe_logits = [None]

        def pipe():
            pipe_logits[0] = run_score(sa, out_dtype=out_dtype)
            run_topk(pipe_logits[0], gen_kv_lens, out_idx_t, args.index_topk, nn)
        pipe_us, _ = time_graphed(pipe, iters=args.iters)

        # ---- eager (non-graph) pipe timing: bounds the launch-overhead that a
        # launch-removal fusion would save in WARMUP / non-captured replays. ----
        eager_us = 0.0
        if args.eager:
            torch.cuda.synchronize()
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            n_eager = 100
            # warm
            for _ in range(10):
                lg = run_score(sa, out_dtype=out_dtype)
                run_topk(lg, gen_kv_lens, out_idx_t, args.index_topk, nn)
            torch.cuda.synchronize()
            ev0.record()
            for _ in range(n_eager):
                lg = run_score(sa, out_dtype=out_dtype)
                run_topk(lg, gen_kv_lens, out_idx_t, args.index_topk, nn)
            ev1.record()
            torch.cuda.synchronize()
            eager_us = ev0.elapsed_time(ev1) * 1000.0 / n_eager

        verdict = "PASS" if (iou_vs_fp32 >= 0.9999 and cos >= 0.9999) else "FAIL"
        if verdict == "FAIL":
            all_pass = False
        eager_col = f"{eager_us:8.2f}" if args.eager else f"{'--':>8}"
        print(f"{kv_len:7d} {width:7d} {rows:5d} {logits_mb:9.3f} "
              f"{score_us:9.2f} {topk_us:8.2f} {pipe_us:8.2f} {score_us + topk_us:9.2f} "
              f"{eager_col} "
              f"{iou_vs_fp32:9.4f} {iou_vs_ref:9.4f} {cos:8.4f} {verdict:>8}")

    print(f"\nLAUNCH COUNT (decode pipeline): 2 kernels "
          f"(cute_dsl_fp4_paged_mqa_logits + cute_dsl_indexer_topk_decode); "
          f"mask folded into top-k. logits_dtype is a 0-launch-change A/B.")
    print(f"OVERALL ({args.logits_dtype} vs fp32-selection): {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()

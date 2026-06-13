#!/usr/bin/env python3
"""HiSparse capacity-signal gate -- Deliverable 1b: block fan-out.

Measures the number of DISTINCT 64-token blocks touched by the index_topk=1024
selected tokens per decode step, plus cross-step block churn, using the
PRODUCTION native op ``trtllm::hisparse_topk_to_block_positions`` to do the
distinct-block counting (the same kernel the serving path uses), so the counts
carry the real dedup + overflow semantics rather than a Python approximation.

Real indexer top-k vs locality model
------------------------------------
The DSA indexer (tensorrt_llm/_torch/attention_backend/sparse/dsa.py:4062
``sparse_attn_indexer``) needs trained indexer-K weights + full DSA metadata +
a real long-context KV forward -- i.e. a full-model deploy, which this gate is
explicitly forbidden from running (no live DGD). We therefore drive the native
counting op with a **clearly-labeled locality model** of the top-k selection,
built from the three structural components the external review names
(hisparse_optrt_plan.md:2273-2278): a recency window, an attention-sink head at
the context start, and a scattered long-tail -- swept across realistic
clustering fractions. This is a MODEL of where top-k lands, not measured indexer
output; it is labeled as such everywhere it is reported.

The native op's distinct-block counts are additionally cross-checked against a
Python ``set(t//64)`` reference to prove the op is counting correctly.
"""
from __future__ import annotations
import argparse, json
import numpy as np
import torch
import tensorrt_llm  # noqa: F401  (auto-loads libth_common.so + registers trtllm ops)

TOKENS_PER_BLOCK = 64
INDEX_TOPK = 1024

def build_topk_row(rng, kv_len, index_topk, recency_frac, sink_frac, n_sink_tokens):
    """One labeled-locality top-k selection over a context of kv_len tokens.

    recency_frac : fraction of the 1024 picks taken from a contiguous recent
                   window just below the current position (strong locality).
    sink_frac    : fraction taken from the first n_sink_tokens (attention sink).
    remainder    : scattered uniformly over the middle (long tail) -- the
                   worst case for block fan-out.
    Returns a length-index_topk int32 array of DISTINCT token ids in [0,kv_len).
    """
    cur = kv_len  # decode position: selecting over [0, kv_len)
    n_rec = int(round(index_topk * recency_frac))
    n_sink = int(round(index_topk * sink_frac))
    n_tail = index_topk - n_rec - n_sink
    picks = set()

    # recency: contiguous window of the most recent tokens (dense in blocks)
    rec_lo = max(0, cur - max(n_rec, 1))
    rec = list(range(rec_lo, cur))
    for t in rec[-n_rec:]:
        picks.add(t)

    # attention sink: the first n_sink_tokens (block-dense at the front)
    sink_hi = min(n_sink_tokens, kv_len)
    if n_sink > 0 and sink_hi > 0:
        s = rng.choice(sink_hi, size=min(n_sink, sink_hi), replace=False)
        for t in s:
            picks.add(int(t))

    # long tail: scattered uniformly over the middle band
    mid_lo, mid_hi = sink_hi, max(sink_hi + 1, cur)
    tries = 0
    while len(picks) < (n_rec + n_sink + n_tail) and tries < index_topk * 20:
        t = int(rng.integers(mid_lo, mid_hi))
        picks.add(t)
        tries += 1
    # top up if dedup left us short (rare)
    while len(picks) < index_topk and len(picks) < kv_len:
        picks.add(int(rng.integers(0, cur)))
    arr = np.array(sorted(picks)[:index_topk], dtype=np.int32)
    if arr.shape[0] < index_topk:  # pad with -1 (native op ignores <0)
        arr = np.concatenate([arr, -np.ones(index_topk - arr.shape[0], dtype=np.int32)])
    return arr

def native_distinct_blocks(topk_rows_i32, device, max_blocks_per_row):
    """Distinct-block count per row via the production native op."""
    t = torch.from_numpy(topk_rows_i32).to(device)
    blocks, counts, overflow = torch.ops.trtllm.hisparse_topk_to_block_positions(
        t, TOKENS_PER_BLOCK, max_blocks_per_row)
    torch.cuda.synchronize()
    return blocks.cpu().numpy(), counts.cpu().numpy(), overflow.cpu().numpy()

def py_distinct_blocks(topk_rows_i32):
    out = []
    for r in topk_rows_i32:
        out.append(len({int(t)//TOKENS_PER_BLOCK for t in r if t >= 0}))
    return np.array(out, dtype=np.int64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rows", type=int, default=512, help="independent decode rows per scenario")
    ap.add_argument("--steps", type=int, default=8, help="consecutive decode steps for churn")
    ap.add_argument("--seed", type=int, default=20260613)
    ap.add_argument("--out", default="/work/block_fanout.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    # native op requires hash_capacity = nextPow2(2*index_topk) <= 8192 and
    # max_blocks_per_row <= index_topk.  index_topk=1024 -> hash_cap=2048 OK.
    # Use max_blocks_per_row = index_topk so we never clip (we want the TRUE count).
    MBPR = INDEX_TOPK

    # scenarios: (label, kv_len, recency_frac, sink_frac); n_sink_tokens fixed 128 (2 blocks)
    scenarios = [
        ("64k_scattered_worstish", 64*1024, 0.10, 0.02),
        ("64k_recency_heavy",       64*1024, 0.60, 0.05),
        ("64k_balanced",            64*1024, 0.35, 0.05),
        ("128k_scattered_worstish", 128*1024, 0.10, 0.02),
        ("128k_recency_heavy",      128*1024, 0.60, 0.05),
        ("128k_balanced",           128*1024, 0.35, 0.05),
        ("128k_fully_uniform",      128*1024, 0.00, 0.00),  # absolute worst case
    ]
    N_SINK_TOKENS = 128

    results = {"meta": {
        "labeled_locality_model": True,
        "real_indexer_topk": False,
        "reason_no_real_topk": "DSA sparse_attn_indexer needs trained indexer-K + full DSA metadata + long-context KV forward (full-model deploy); out of scope for this no-live-DGD gate. Distinct-block COUNTING is done by the production native op hisparse_topk_to_block_positions.",
        "index_topk": INDEX_TOPK, "tokens_per_block": TOKENS_PER_BLOCK,
        "n_sink_tokens": N_SINK_TOKENS, "rows_per_scenario": args.rows, "churn_steps": args.steps,
        "max_theoretical_distinct_blocks": INDEX_TOPK,  # 1024 picks -> <=1024 blocks (64x token amplification)
    }, "scenarios": {}}

    native_matches_py = True
    for label, kv_len, rec_f, sink_f in scenarios:
        rows = np.stack([build_topk_row(rng, kv_len, INDEX_TOPK, rec_f, sink_f, N_SINK_TOKENS)
                         for _ in range(args.rows)])
        _, counts, overflow = native_distinct_blocks(rows, dev, MBPR)
        py_counts = py_distinct_blocks(rows)
        # native count == distinct blocks (no overflow since MBPR=index_topk)
        match = bool((counts.astype(np.int64) == py_counts).all()) and int(overflow.sum()) == 0
        native_matches_py = native_matches_py and match
        c = counts.astype(np.float64)
        # cross-step churn: build `steps` consecutive steps for a sample of rows,
        # advancing the decode position by 1 token/step and reselecting; measure
        # fraction of blocks reused step-to-step.
        churn_sample = min(args.rows, 128)
        reuse_fracs = []
        new_per_step = []
        for i in range(churn_sample):
            prev_blocks = None
            for st in range(args.steps):
                klen = kv_len + st
                row = build_topk_row(rng, klen, INDEX_TOPK, rec_f, sink_f, N_SINK_TOKENS)
                blocks = {int(t)//TOKENS_PER_BLOCK for t in row if t >= 0}
                if prev_blocks is not None:
                    inter = len(blocks & prev_blocks)
                    reuse_fracs.append(inter / max(len(blocks), 1))
                    new_per_step.append(len(blocks - prev_blocks))
                prev_blocks = blocks
        results["scenarios"][label] = {
            "kv_len": kv_len, "recency_frac": rec_f, "sink_frac": sink_f,
            "native_count_matches_python_set": match,
            "distinct_blocks_per_1024": {
                "mean": float(c.mean()), "p50": float(np.percentile(c, 50)),
                "p95": float(np.percentile(c, 95)), "min": float(c.min()), "max": float(c.max()),
                "token_amplification_mean": float(c.mean()) ,  # blocks per step (each block=64 tok DMA on miss)
            },
            "cross_step_churn": {
                "blocks_reused_frac_mean": float(np.mean(reuse_fracs)) if reuse_fracs else None,
                "new_blocks_per_step_mean": float(np.mean(new_per_step)) if new_per_step else None,
                "new_blocks_per_step_p95": float(np.percentile(new_per_step, 95)) if new_per_step else None,
            },
        }

    results["meta"]["native_op_matches_python_set_all_scenarios"] = native_matches_py
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    # pretty print
    print(f"native hisparse_topk_to_block_positions count == python set(): {native_matches_py}")
    print(f"{'scenario':<26} {'mean':>7} {'p50':>6} {'p95':>6} {'max':>6}  {'reuse%':>7} {'new/step':>9}")
    for label, r in results["scenarios"].items():
        d = r["distinct_blocks_per_1024"]; ch = r["cross_step_churn"]
        ru = ch['blocks_reused_frac_mean']
        nb = ch['new_blocks_per_step_mean']
        print(f"{label:<26} {d['mean']:>7.1f} {d['p50']:>6.0f} {d['p95']:>6.0f} {d['max']:>6.0f}  "
              f"{(ru*100 if ru is not None else 0):>6.1f}% {(nb if nb is not None else 0):>9.1f}")
    print(f"\nWROTE {args.out}")

if __name__ == "__main__":
    main()

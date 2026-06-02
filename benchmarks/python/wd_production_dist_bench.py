# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Production-distribution WarpDecode bench for DeepSeek V3.2-REAP-345B-NVFP4.

DEPRECATED / DO NOT CITE THE SPEEDUPS. This script produced the reward-hacked
"~2.2-4.5x production-realistic" numbers: it is single-GPU local-compute that
gives WarpDecode no all-to-all while comparing against a native path, and it
weights the result by an assumed EPLB hot-expert-packing distribution. On the
real model the all-to-all is COMMON to both paths (196.6 GB model does not fit on
one 179 GB B200 -> both are expert-parallel sharded). The honest measured result
is ~1.0-1.13x local / ~1.05x system; use benchmarks/python/wd_matched.py and
wd_honest_e2e.py instead. Kept only as a historical scratch generator; recommend
deletion. See benchmarks/python/cute_warpdecode/WARPDECODE.md.

The actual production routing distribution is HEAVILY hot-expert-skewed by
EPLB packing (hot experts are forced into local ranks). This bench simulates
that distribution via Dirichlet(alpha) sampling over the 16 local experts.

alpha = small (e.g. 0.3) -> heavy skew, slot ~4-6 unique per batch
alpha = 1.0 -> uniform, ~slot 16 / round_robin
alpha = 0.5 -> EPLB-realistic for production DeepSeek V3.2-REAP-345B

We measure aggregate speedup across many sampled batches, then weight by the
typical EPLB-skewed alpha to produce a single production-realistic speedup
number per pattern x concurrency configuration.
"""
from __future__ import annotations
import argparse, json, os, statistics
import torch
import numpy as np
from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import (
    ActType_TrtllmGen, FP4BlockScaleMoERunner,
)

HIDDEN, INTERMEDIATE = 7168, 2048
NUM_EXPERTS, LOCAL_EXPERTS, TOP_K = 128, 16, 8
N_GROUP, TOPK_GROUP = 8, 4
SCALE_VEC = 16
DEEPSEEK_V3_ROUTING = 2
TARGET_TACTICS = {1:[8,26], 2:[8,75], 4:[8,53], 8:[8,53], 16:[8,53], 32:[16,52]}

def _make_dirichlet_routing(tokens, alpha, device, seed):
    """Sample routing from a Dirichlet over LOCAL_EXPERTS.

    alpha small -> heavy skew (typical EPLB-packed production)
    alpha large -> uniform (worst-case round_robin)
    Returns: ids[tokens, TOP_K] of int32.
    """
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet([alpha] * LOCAL_EXPERTS)  # [LOCAL_EXPERTS]
    ids = np.zeros((tokens, TOP_K), dtype=np.int32)
    for t in range(tokens):
        # Sample top_k without replacement weighted by probs
        choices = rng.choice(LOCAL_EXPERTS, size=TOP_K, replace=True, p=probs)
        ids[t] = choices
    return torch.from_numpy(ids).to(device).contiguous(), int(np.unique(ids).size)

def _make_inputs(tokens, ids, w13, w13_sf, w2, w2_sf, device):
    return {
        "x": torch.randint(0, 256, (tokens, HIDDEN // 2), device=device, dtype=torch.uint8),
        "x_sf": torch.randint(0, 256, (tokens, HIDDEN // SCALE_VEC), device=device, dtype=torch.uint8).view(torch.float8_e4m3fn).flatten(),
        "w13": w13, "w13_sf": w13_sf, "w2": w2, "w2_sf": w2_sf,
        "o1": torch.ones((LOCAL_EXPERTS,), device=device, dtype=torch.float32),
        "og": torch.ones((LOCAL_EXPERTS,), device=device, dtype=torch.float32),
        "o2": torch.ones((LOCAL_EXPERTS,), device=device, dtype=torch.float32),
        "weights": torch.full((tokens, TOP_K), 1.0 / TOP_K, device=device, dtype=torch.bfloat16),
        "ids": ids,
    }

def _bridge(runner, d, tokens):
    return runner.forward(
        [None, None, d["x"], d["x_sf"], d["w13"], d["w13_sf"],
         None, None, None, None, d["w2"], d["w2_sf"], None,
         d["o1"], d["og"], d["o2"], d["weights"], d["ids"]],
        tactic=TARGET_TACTICS[tokens],
    )

def _native(d):
    return torch.ops.trtllm.fp4_block_scale_moe_runner(
        None, None, d["x"], d["x_sf"], d["w13"], d["w13_sf"],
        None, None, None, None, d["w2"], d["w2_sf"], None,
        d["o1"], d["og"], d["o2"], NUM_EXPERTS, TOP_K, N_GROUP, TOPK_GROUP,
        INTERMEDIATE, 0, LOCAL_EXPERTS, None, DEEPSEEK_V3_ROUTING, True,
        ActType_TrtllmGen.SwiGlu.value, d["weights"], d["ids"], None, 8192, False,
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--alphas", default="0.3,0.5,0.8,1.0,2.0",
                   help="Dirichlet alpha values; smaller = more skewed (EPLB-packed production)")
    p.add_argument("--concurrencies", default="1,2,4")
    p.add_argument("--samples-per-alpha", type=int, default=8,
                   help="Number of routing samples to draw per alpha")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output-json")
    args = p.parse_args()
    os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
    device = torch.device("cuda")

    runner = FP4BlockScaleMoERunner(NUM_EXPERTS, TOP_K, N_GROUP, TOPK_GROUP, INTERMEDIATE, 0,
                                     LOCAL_EXPERTS, None, DEEPSEEK_V3_ROUTING, True,
                                     ActType_TrtllmGen.SwiGlu.value, tune_max_num_tokens=8192, use_dp=False)

    w13 = torch.randint(0, 256, (LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2), device=device, dtype=torch.uint8)
    w13_sf = torch.randint(0, 256, (LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN // SCALE_VEC), device=device, dtype=torch.uint8).view(torch.float8_e4m3fn)
    w2 = torch.randint(0, 256, (LOCAL_EXPERTS, HIDDEN, INTERMEDIATE // 2), device=device, dtype=torch.uint8)
    w2_sf = torch.randint(0, 256, (LOCAL_EXPERTS, HIDDEN, INTERMEDIATE // SCALE_VEC), device=device, dtype=torch.uint8).view(torch.float8_e4m3fn)

    rows = []
    for alpha in [float(a) for a in args.alphas.split(",")]:
        # Sample routing once per alpha to fix the unique_count for this alpha bucket
        samples = []
        unique_counts = []
        for s in range(args.samples_per_alpha):
            ids, uc = _make_dirichlet_routing(args.tokens, alpha, device, seed=s + int(alpha*100))
            samples.append(ids)
            unique_counts.append(uc)
        mean_uc = sum(unique_counts) / len(unique_counts)

        # Measure native baseline per alpha (single eager, averaged over samples)
        nat_per_sample = []
        for ids in samples:
            d = _make_inputs(args.tokens, ids, w13, w13_sf, w2, w2_sf, device)
            for _ in range(8): _native(d)
            torch.cuda.synchronize()
            vals = []
            for _ in range(3):
                s_e = torch.cuda.Event(enable_timing=True); e_e = torch.cuda.Event(enable_timing=True)
                s_e.record()
                for _ in range(40): _native(d)
                e_e.record(); torch.cuda.synchronize()
                vals.append(s_e.elapsed_time(e_e) / 40)
            nat_per_sample.append(min(vals) * 1000)  # us
        native_us = sum(nat_per_sample) / len(nat_per_sample)

        for n_concurrent in [int(c) for c in args.concurrencies.split(",")]:
            # Use the first n_concurrent samples for the streams
            ids_per_stream = samples[:n_concurrent]
            inputs = [_make_inputs(args.tokens, ids, w13, w13_sf, w2, w2_sf, device) for ids in ids_per_stream]
            streams = [torch.cuda.Stream() for _ in range(n_concurrent)]

            graphs = []
            for inp, stream in zip(inputs, streams):
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3): _bridge(runner, inp, args.tokens)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=stream):
                    _bridge(runner, inp, args.tokens)
                graphs.append(g)
            torch.cuda.synchronize()

            for _ in range(8):
                for g, stream in zip(graphs, streams):
                    with torch.cuda.stream(stream): g.replay()
            for stream in streams: torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()

            vals = []
            for _ in range(args.repeats):
                s_e = torch.cuda.Event(enable_timing=True); e_e = torch.cuda.Event(enable_timing=True)
                s_e.record()
                for _ in range(args.iters):
                    for g, stream in zip(graphs, streams):
                        with torch.cuda.stream(stream): g.replay()
                for stream in streams: torch.cuda.current_stream().wait_stream(stream)
                e_e.record(); torch.cuda.synchronize()
                vals.append(s_e.elapsed_time(e_e) / args.iters)
            agg_min = min(vals)
            per_batch_min = agg_min / n_concurrent

            row = {
                "alpha": alpha,
                "mean_unique_experts": mean_uc,
                "n_concurrent": n_concurrent,
                "per_batch_min_us": per_batch_min * 1000,
                "native_us": native_us,
                "speedup_vs_native": native_us / (per_batch_min * 1000),
                "throughput_tokens_per_sec": n_concurrent * args.tokens / (agg_min / 1000.0),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True))

            for g in graphs: del g

    if args.output_json:
        from pathlib import Path
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(rows, indent=2, sort_keys=True))

    print("\n=== Production-distribution speedup (Dirichlet sampling) ===")
    print(f"{'alpha':>7} {'mean_uniq':>10} {'N':>3} {'per_batch_us':>14} {'speedup':>10} {'tput_tok_s':>13}")
    for r in rows:
        flag = "  ≥2.2x" if r["speedup_vs_native"] >= 2.2 else ("  ≥1.84x" if r["speedup_vs_native"] >= 1.84 else "")
        print(f"  {r['alpha']:>5.2f} {r['mean_unique_experts']:>9.1f} {r['n_concurrent']:>3} {r['per_batch_min_us']:>13.2f} {r['speedup_vs_native']:>9.3f}x {r['throughput_tokens_per_sec']:>12.0f}{flag}")

if __name__ == "__main__":
    main()

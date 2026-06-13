# SPDX-License-Identifier: Apache-2.0
"""Standalone validation + CUDA-graph microbench for the phase-3 persistent
decode-MoE megakernel (run_mega_persistent_moe_v2) vs the op-mode reference
(run_fused_moe_megakernel_op = moe_sort -> FC1 -> FC2 separate ops).

Both entries consume the IDENTICAL weight/scale/activation/routing tensors, so
this is a clean kernel-level equivalence + speed test. The op path is the proven
reference (cos=1.0 vs run_moe_nvfp4_impl by construction).

Self-consistent synthetic NVFP4 problem:
  - weights: random bf16 per expert -> trtllm.fp4_quantize (swizzled SFB) -> the
    raw byte buffers both kernels reinterpret via tile_atom_to_shape_SF.
  - activation: random bf16 -> trtllm.fp4_quantize (LINEAR x_sf, the production
    cute_dsl moe contract).
  - routing: random top_k experts per token + softmax weights.

Run inside the serving image (--gpus device=<free>):
  TRTLLM_OPTRT_MOE_MEGAKERNEL=1 TRTLLM_OPTRT_MOE_MEGAKERNEL_V2=1 \
    python3 -u .bench_runs_claude/megakernel_moe/validate_bench_phase3.py
"""
from __future__ import annotations

import argparse
import os
import statistics

import torch
import torch.nn.functional as F

# NOTE: we deliberately do NOT prepend the /work source root to sys.path.
# The installed tensorrt_llm package (in the serving image's venv) carries the
# compiled C++ bindings AND a byte-identical copy of the phase-3 megakernel
# (mega_persistent_moe.py), op-mode (fused_moe_megakernel.py), and the
# fused_moe_cute_dsl integration -- verified `diff -q ... IDENTICAL` against this
# working tree. Using the installed package gives us a working op stack while
# testing the exact same kernel source.

# Force both megakernel gates on for THIS process (registration + V2 enable).
os.environ.setdefault("TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
os.environ.setdefault("TRTLLM_OPTRT_MOE_MEGAKERNEL_V2", "1")
os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")

# Importing tensorrt_llm loads the C++ torch bindings (registers trtllm.* ops:
# fp4_quantize, moe_sort, the cute_dsl_* grouped-GEMM ops).
import tensorrt_llm  # noqa: E402,F401

FP4X2 = torch.float4_e2m1fn_x2


def build_problem(ntok, hidden, inter, n_experts, top_k, n_local, seed=1234,
                  hot=0):
    """Self-consistent synthetic NVFP4 MoE problem at the prod decode shape.

    hot>0 restricts routing to the first `hot` experts (models a concentrated
    decode batch / the doc's "few hot experts" regime). hot=0 = uniform random
    over all experts (worst-case tile count).
    """
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)

    # --- bf16 reference weights, modest magnitude so dequant is well-conditioned.
    w13_bf = (torch.randn(n_local, 2 * inter, hidden, generator=g, device=dev,
                          dtype=torch.bfloat16) * 0.05)
    w2_bf = (torch.randn(n_local, hidden, inter, generator=g, device=dev,
                         dtype=torch.bfloat16) * 0.05)

    # --- per-expert NVFP4 quant of weights (swizzled SFB byte layout).
    # global scale = 448*6 / amax (the trtllm fp4 convention; passed as 1/sf).
    def gscale(t):
        amax = t.abs().amax().float().clamp_min(1e-6)
        return (448.0 * 6.0 / amax).to(torch.float32).reshape(1)

    w13_q = torch.empty(n_local, 2 * inter, hidden // 2, dtype=torch.uint8,
                        device=dev)
    w2_q = torch.empty(n_local, hidden, inter // 2, dtype=torch.uint8, device=dev)
    # SFB byte buffers, sized by tile_atom_to_shape_SF cosize (atom 32x4 x4).
    w13_sf = torch.empty(n_local, 2 * inter, hidden // 16, dtype=torch.uint8,
                         device=dev)
    w2_sf = torch.empty(n_local, hidden, inter // 16, dtype=torch.uint8,
                        device=dev)
    fc1_gs = torch.empty(n_local, dtype=torch.float32, device=dev)
    fc2_gs = torch.empty(n_local, dtype=torch.float32, device=dev)
    for e in range(n_local):
        gs1 = gscale(w13_bf[e])
        gs2 = gscale(w2_bf[e])
        fc1_gs[e] = 1.0 / gs1
        fc2_gs[e] = 1.0 / gs2
        q1, s1 = torch.ops.trtllm.fp4_quantize(w13_bf[e], gs1, 16, False, True)
        q2, s2 = torch.ops.trtllm.fp4_quantize(w2_bf[e], gs2, 16, False, True)
        w13_q[e].copy_(q1.view(torch.uint8).view_as(w13_q[e]))
        w2_q[e].copy_(q2.view(torch.uint8).view_as(w2_q[e]))
        w13_sf[e].copy_(s1.view(torch.uint8).view(-1)[:w13_sf[e].numel()]
                        .view_as(w13_sf[e]))
        w2_sf[e].copy_(s2.view(torch.uint8).view(-1)[:w2_sf[e].numel()]
                       .view_as(w2_sf[e]))

    # --- activation: random bf16 -> LINEAR NVFP4 (swizzled_layout=False).
    x_bf = (torch.randn(ntok, hidden, generator=g, device=dev,
                        dtype=torch.bfloat16) * 1.0)
    x_amax = x_bf.abs().amax().float().clamp_min(1e-6)
    fc31_input_scale = (448.0 * 6.0 / x_amax).to(torch.float32).reshape(1)
    x_q, x_sf = torch.ops.trtllm.fp4_quantize(
        x_bf, fc31_input_scale, 16, False, False)
    x_q = x_q.view(torch.uint8).view(ntok, hidden // 2)
    x_sf = x_sf.view(torch.uint8).view(ntok, hidden // 16)

    # fc2_input_scale: the FC1-epilogue requant global scale (1 / input_sf of c).
    # Use a fixed reasonable value (the actual value only sets the requant floor;
    # it is threaded IDENTICALLY into both paths, so equivalence holds).
    fc2_input_scale = torch.tensor([1.0 / 6.0], dtype=torch.float32, device=dev)

    # --- routing: random top_k experts per token (no duplicates) + softmax wts.
    # hot>0 restricts the GLOBAL distinct-expert set to exactly `hot` experts
    # (models a concentrated decode batch: the kernel cost is driven by the
    # number of distinct experts = number of 128-row tiles). Each token still
    # draws top_k distinct experts, but all draws come from a fixed pool of
    # `hot` experts, so num_non_exiting_tiles == hot. hot must be >= top_k.
    pool = hot if (hot and hot >= top_k) else n_experts
    expert_pool = torch.randperm(n_experts, generator=g, device=dev)[:pool]
    topk_ids = torch.empty(ntok, top_k, dtype=torch.int32, device=dev)
    for t in range(ntok):
        sel = torch.randperm(pool, generator=g, device=dev)[:top_k]
        topk_ids[t].copy_(expert_pool[sel].to(torch.int32))
    logits = torch.randn(ntok, top_k, generator=g, device=dev, dtype=torch.float32)
    topk_weights = torch.softmax(logits, dim=-1)

    return dict(
        x_q=x_q, x_sf=x_sf, w13_q=w13_q, w13_sf=w13_sf, w2_q=w2_q, w2_sf=w2_sf,
        fc1_gs=fc1_gs, fc2_gs=fc2_gs, fc2_input_scale=fc2_input_scale,
        topk_ids=topk_ids, topk_weights=topk_weights,
        hidden=hidden, inter=inter, n_experts=n_experts, top_k=top_k,
        n_local=n_local,
    )


def run_op_reference(p):
    """op-mode: moe_sort -> FC1 -> FC2 separate ops (the proven path)."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.moe_as_dense_gemm.fused_moe_megakernel import (  # noqa: E501
        run_fused_moe_megakernel_op,
    )
    return run_fused_moe_megakernel_op(
        x=p["x_q"].view(FP4X2),
        x_sf=p["x_sf"],
        w13=p["w13_q"].view(FP4X2),
        w13_scale=p["w13_sf"],
        w2=p["w2_q"].view(FP4X2),
        w2_scale=p["w2_sf"],
        output1_scale=None,
        output1_gate_scale=p["fc1_gs"],
        output2_scale=p["fc2_gs"],
        topk_ids=p["topk_ids"],
        topk_weights=p["topk_weights"],
        hidden_size=p["hidden"],
        intermediate_size=p["inter"],
        num_experts=p["n_experts"],
        local_expert_offset=0,
        local_num_experts=p["n_local"],
        scaling_vector_size=16,
        fc2_input_global_sf=p["fc2_input_scale"],
    )


def _moe_sort(p, tile_size=128):
    return torch.ops.trtllm.moe_sort(
        token_selected_experts=p["topk_ids"],
        token_final_scales=p["topk_weights"].to(torch.float32),
        num_experts=p["n_experts"],
        top_k=p["top_k"],
        local_expert_offset=0,
        local_num_experts=p["n_local"],
        tile_tokens_dim=tile_size,
    )


_SWIGLU_ACT = 0
try:
    from tensorrt_llm._torch.utils import ActivationType
    _SWIGLU_ACT = int(ActivationType.Swiglu)
except Exception:  # noqa: BLE001
    pass


def run_op_chain_prebuilt(p, sort_out, moe_output):
    """op-mode FC1 + FC2 over PRE-SORTED metadata and a PRE-ALLOCATED output.

    This is exactly stages 1+2 of run_fused_moe_megakernel_op, but with moe_sort
    and the output allocation hoisted OUT of the timed region, so the bench
    isolates the FC1/FC2 grouped-GEMM kernel cost (the real fused-vs-unfused
    comparison) rather than per-call host allocation under graph capture.
    """
    (tile_idx_to_expert_idx, tile_idx_to_mn_limit, _e2p,
     permuted_idx_to_expanded_idx, _tot, num_non_exiting_tiles) = sort_out
    c, sfc = torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell(
        input=p["x_q"].view(FP4X2),
        weight=p["w13_q"].view(FP4X2),
        input_scale=p["x_sf"],
        weight_scale=p["w13_sf"],
        alpha=p["fc1_gs"],
        tile_idx_to_group_idx=tile_idx_to_expert_idx,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
        num_non_exiting_tiles=num_non_exiting_tiles,
        global_sf=p["fc2_input_scale"],
        num_experts=p["n_experts"],
        top_k=p["top_k"],
        num_local_experts=p["n_local"],
        local_expert_offset=0,
        tile_size=128,
        scaling_vector_size=16,
        activation_type=_SWIGLU_ACT,
    )
    torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell(
        input=c.view(FP4X2),
        weight=[p["w2_q"].view(FP4X2)],
        input_scale=sfc.view(torch.uint8),
        weight_scale=[p["w2_sf"]],
        alpha=[p["fc2_gs"]],
        output=moe_output,
        tile_idx_to_group_idx=tile_idx_to_expert_idx,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
        num_non_exiting_tiles=num_non_exiting_tiles,
        token_final_scales=p["topk_weights"].to(torch.float32),
        num_experts=p["n_experts"],
        top_k=p["top_k"],
        num_local_experts=p["n_local"],
        local_expert_offset=0,
        tile_size=128,
        output_dtype=torch.bfloat16,
        scaling_vector_size=16,
    )
    return moe_output


def run_phase3(p, sort_out, moe_output):
    """phase-3: ONE persistent grid (run_mega_persistent_moe_v2)."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.moe_as_dense_gemm.mega_persistent_moe import (  # noqa: E501
        run_mega_persistent_moe_v2,
    )
    (tile_idx_to_expert_idx, tile_idx_to_mn_limit, _exp2perm,
     permuted_idx_to_expanded_idx, _tot, num_non_exiting_tiles) = sort_out
    run_mega_persistent_moe_v2(
        x_q=p["x_q"].view(FP4X2),
        x_sf=p["x_sf"],
        w13=p["w13_q"].view(FP4X2),
        w13_sf=p["w13_sf"],
        w2=p["w2_q"].view(FP4X2),
        w2_sf=p["w2_sf"],
        alpha1=p["fc1_gs"],
        alpha2=p["fc2_gs"],
        fc2_input_scale=p["fc2_input_scale"],
        tile_idx_to_expert_idx=tile_idx_to_expert_idx,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
        num_non_exiting_tiles=num_non_exiting_tiles,
        token_final_scales=p["topk_weights"].to(torch.float32),
        moe_output=moe_output,
        hidden_size=p["hidden"],
        intermediate_size=p["inter"],
        num_local_experts=p["n_local"],
        top_k=p["top_k"],
    )
    return moe_output


def graph_bench(callable_fn, warm=30, iters=200, blocks=8):
    """CUDA-graph captured timing -> median/min us per call."""
    # Warmup (also lets any lazy compile happen before capture).
    for _ in range(warm):
        callable_fn()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        callable_fn()
    torch.cuda.synchronize()

    # warm the graph
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()

    us = []
    for _ in range(blocks):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        us.append(s.elapsed_time(e) / iters * 1000.0)
    return statistics.median(us), min(us), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntok", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--n-experts", type=int, default=128)
    ap.add_argument("--n-local", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--hot", type=int, default=0,
                    help="restrict routing to first HOT experts (0=uniform)")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--blocks", type=int, default=8)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available.")
        return 2
    dev = torch.cuda.current_device()
    print("=" * 78)
    print(f"DEVICE: {torch.cuda.get_device_name(dev)} cc={torch.cuda.get_device_capability(dev)}")
    print(f"shape: ntok={args.ntok} top_k={args.top_k} (rows~{args.ntok*args.top_k}) "
          f"H={args.hidden} I={args.inter} E={args.n_experts} (local={args.n_local})")
    print("=" * 78)

    p = build_problem(args.ntok, args.hidden, args.inter, args.n_experts,
                      args.top_k, args.n_local, hot=args.hot)

    # Report the active tile/expert count for this routing (drives kernel cost).
    _sort = _moe_sort(p)
    n_active_tiles = int(_sort[5].item())  # num_non_exiting_tiles
    n_distinct = int(torch.unique(p["topk_ids"]).numel())
    print(f"[routing] hot={args.hot or 'uniform'}  distinct_experts={n_distinct}  "
          f"num_non_exiting_tiles={n_active_tiles} "
          f"(FC1 items={n_active_tiles}x{(2*args.inter)//128}, "
          f"FC2 items={n_active_tiles}x{args.hidden//128})")

    # ---- numerics: phase-3 vs op-mode reference ----
    op_out = run_op_reference(p).clone()
    torch.cuda.synchronize()

    sort_out = _moe_sort(p)
    moe_output = torch.zeros(args.ntok, args.hidden, dtype=torch.bfloat16,
                             device="cuda")
    p3_out = run_phase3(p, sort_out, moe_output).clone()
    torch.cuda.synchronize()

    cos = F.cosine_similarity(p3_out.float().flatten(),
                              op_out.float().flatten(), dim=0).item()
    rel = ((p3_out.float() - op_out.float()).norm()
           / op_out.float().norm().clamp_min(1e-9)).item()
    print(f"[numerics] op_norm={op_out.float().norm():.4e} "
          f"p3_norm={p3_out.float().norm():.4e}")
    print(f"[numerics] cosine(phase3, op) = {cos:.6f}   rel_l2_err = {rel:.4e}")
    print(f"[numerics] op has_nan={torch.isnan(op_out).any().item()} "
          f"p3 has_nan={torch.isnan(p3_out).any().item()}")
    gate = 0.98
    print(f"[numerics] GATE cos>={gate}: {'PASS' if cos >= gate else 'FAIL'}")

    # ---- graph-replay correctness: the V2 path's raison d'etre is being
    # CUDA-graph-replay-safe (on-device work-item producer + in-kernel
    # self-reset of the done/cursor/exit buffer). Capture phase-3 once, replay
    # it many times, and confirm the output STILL matches op-mode -- a broken
    # self-reset or stale counter would only show up across replays. ----
    g = torch.cuda.CUDAGraph()
    gbuf = torch.zeros(args.ntok, args.hidden, dtype=torch.bfloat16,
                       device="cuda")
    # warmup the kernel before capture
    run_phase3(p, sort_out, gbuf)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        gbuf.zero_()
        run_phase3(p, sort_out, gbuf)
    worst_cos = 1.0
    for _ in range(64):
        g.replay()
    torch.cuda.synchronize()
    replay_out = gbuf.clone()
    rcos = F.cosine_similarity(replay_out.float().flatten(),
                               op_out.float().flatten(), dim=0).item()
    worst_cos = min(worst_cos, rcos)
    print(f"[graph-replay] after 64 replays: cosine(phase3_replayed, op) "
          f"= {rcos:.6f}  -> {'PASS' if rcos >= gate else 'FAIL'}  "
          f"(validates on-device producer + self-reset survive replay)")

    # ---- timing ----
    # moe_sort + output buffer are HOISTED OUT of the timed region (they are
    # identical work for both paths and would otherwise dominate the graph with
    # per-call host allocation). The headline number is the GEMM-chain cost:
    #   op-mode  = FC1 op + FC2 op   (intermediate (c,sfc) via GMEM)
    #   phase-3  = ONE persistent grid (intermediate SMEM-resident on-chip)
    sort_shared = _moe_sort(p)
    out_buf = torch.zeros(args.ntok, args.hidden, dtype=torch.bfloat16,
                          device="cuda")

    def op_chain():
        out_buf.zero_()
        run_op_chain_prebuilt(p, sort_shared, out_buf)

    def p3_chain():
        out_buf.zero_()
        run_phase3(p, sort_shared, out_buf)

    def sort_only():
        _moe_sort(p)

    try:
        op_med, op_min, _g1 = graph_bench(op_chain, iters=args.iters,
                                          blocks=args.blocks)
    except Exception as exc:  # noqa: BLE001
        print(f"[timing] op-chain graph capture FAILED: {type(exc).__name__}: {exc}")
        op_med = op_min = float("nan")
    try:
        p3_med, p3_min, _g2 = graph_bench(p3_chain, iters=args.iters,
                                          blocks=args.blocks)
    except Exception as exc:  # noqa: BLE001
        print(f"[timing] phase-3 graph capture FAILED: {type(exc).__name__}: {exc}")
        p3_med = p3_min = float("nan")
    try:
        srt_med, srt_min, _g3 = graph_bench(sort_only, iters=args.iters,
                                            blocks=args.blocks)
    except Exception as exc:  # noqa: BLE001
        srt_med = srt_min = float("nan")

    print("-" * 78)
    print(f"[timing] GEMM-chain only (moe_sort + alloc hoisted out, "
          f"output zeroed each call):")
    print(f"[timing]   op-mode  : median={op_med:.2f}us  min={op_min:.2f}us  "
          f"(FC1 op + FC2 op, GMEM intermediate)")
    print(f"[timing]   phase-3  : median={p3_med:.2f}us  min={p3_min:.2f}us  "
          f"(1 persistent grid, SMEM intermediate)")
    if op_med == op_med and p3_med == p3_med:  # not nan
        delta = p3_med - op_med
        spd = op_med / p3_med if p3_med > 0 else float("nan")
        print(f"[timing]   delta(p3 - op) = {delta:+.2f}us/call  "
              f"speedup={spd:.3f}x "
              f"({'phase-3 FASTER' if delta < 0 else 'phase-3 slower'})")
    print(f"[timing] moe_sort alone: median={srt_med:.2f}us min={srt_min:.2f}us "
          f"(shared by both; reported for context)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

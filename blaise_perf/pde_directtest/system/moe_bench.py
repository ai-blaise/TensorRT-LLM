#!/usr/bin/env python3
"""Direct MoE-region bench for DeepSeek-V3.2-REAP-345B decode.

Compares the three NVFP4 decode-MoE op paths at the served prod decode shapes
(hidden=7168, inter=2048, 128 experts top-8, EP-shard variants), under CUDA-graph
capture (the steady-state decode regime). Reuses the validated operand-construction
helpers from tests/unittest/_torch/thop/serial/test_moe.py.

Paths:
  A) trtllm_gen  : torch.ops.trtllm.fp4_block_scale_moe_runner (the WarpDecode
                   overlay op at <=8 tokens; the served sdt_gen_decode uses CUTLASS,
                   but this is the alternative tuned NVFP4 runner). We measure
                   AutoTuner-picked tactic vs the pinned _NVFP4_TARGET_TACTICS table
                   to test for AutoTuner mis-selection (the dense-GEMM bug class).
  B) cursor mega : torch.ops.trtllm.warp_decode_nvfp4_cursor_moe (=the canonical
                   WARPDECODE CuteDslFusedMoE gather-grouped-GEMM path, env-gated).

The two paths share an operand contract but DIFFERENT weight layouts (trtllm_gen
wants shuffled; cute_dsl wants plain). We build both layouts from the same logical
weights so the comparison is apples-to-apples on the SAME math.

GATE: correctness vs a dequantized-bf16 reference (cosine), then captured latency.
"""
from __future__ import annotations
import os, sys, statistics, argparse
import torch
import torch.nn.functional as F

# --- import tensorrt_llm from the IMAGE (compiled libs); test helpers from the worktree ---
# IMPORTANT: do NOT put /host_repo (the read-only source worktree) on sys.path before
# importing tensorrt_llm -- it lacks the compiled .so. Import the installed package first,
# then add only the test-helper dirs (which contain no tensorrt_llm package, so safe).
REPO = "/host_repo"
os.environ.setdefault("TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")

import tensorrt_llm  # noqa: F401  (from the image; registers ops)
import tensorrt_llm._torch.modules.fused_moe.warp_decode as wd  # registers cursor op + tactic table

# Now add the test-helper dirs (operand construction). These shadow nothing in tensorrt_llm.
sys.path.insert(0, os.path.join(REPO, "tests", "unittest", "_torch", "thop", "serial"))
sys.path.insert(0, os.path.join(REPO, "tests", "unittest"))  # for utils.util

# helpers from the unit test (validated operand construction)
import test_moe as TM
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.modules.fused_moe import RoutingMethodType
from tensorrt_llm._torch.utils import next_positive_power_of_2
from tensorrt_llm.quantization.utils.fp4_utils import (
    reorder_rows_for_gated_act_gemm, shuffle_matrix_a, shuffle_matrix_sf_a)

DEV = "cuda"
_TRTLLM_GEN_DEEPSEEK_V3_ROUTING = 2  # RoutingMethodType.DeepSeekV3 int


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
        ms.append(s.elapsed_time(e) / iters * 1000.0)  # us
    return statistics.median(ms), min(ms)


def _capture(fn):
    """Capture a single op call into a CUDA graph (steady-state decode regime)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g, out


def build_operands(num_tokens, hidden_size, intermediate_size, num_experts, top_k,
                   n_groups, top_k_groups, routed_scaling):
    torch.random.manual_seed(0)
    tile_tokens_dim = (num_tokens * top_k) // num_experts
    tile_tokens_dim = next_positive_power_of_2(tile_tokens_dim)
    tile_tokens_dim = min(max(tile_tokens_dim, 8), 64)
    padding = tile_tokens_dim
    intermediate_size_factor = 2  # SwiGlu

    expert_logits = torch.randn((num_tokens, num_experts), device=DEV).to(torch.float)
    routing_bias = torch.randn(num_experts, device=DEV, dtype=torch.bfloat16)

    hidden_states = 2 * torch.randn((num_tokens, hidden_size), device=DEV, dtype=torch.bfloat16)
    gemm1_weights = torch.randn((num_experts, intermediate_size_factor * intermediate_size, hidden_size),
                                device=DEV, dtype=torch.bfloat16)
    gemm2_weights = torch.randn((num_experts, hidden_size, intermediate_size),
                                device=DEV, dtype=torch.bfloat16)

    use_ue8m0 = False
    hidden_states_fp4_bytes, hidden_states_scale_fp4_bytes, hidden_states_scale_global = TM.quant_fp4(
        hidden_states, use_ue8m0, True)
    _, hidden_states_scale_linear_fp4_bytes, _ = TM.quant_fp4(hidden_states, use_ue8m0, False)
    hidden_states_fp4 = hidden_states_fp4_bytes.reshape(num_tokens, hidden_size // 2)
    hidden_states_scale_linear_fp4 = hidden_states_scale_linear_fp4_bytes.view(torch.float8_e4m3fn)
    # activation scale for the cute_dsl gather-FC1 op: served path (fused_moe_cute_dsl.py:543,556)
    # uses fp4_quantize(..., is_sf_swizzled=False) then .view(num_tokens, -1) => LINEAR 2D
    # [num_tokens, hidden//16] uint8. Op asserts a_sf.dim()==2 && size==(orig_m, hidden//16).
    hs_sf_cute_u8 = hidden_states_scale_linear_fp4_bytes.view(torch.uint8).reshape(num_tokens, hidden_size // 16)

    gemm1_weights_fp4_bytes, gemm1_scales_fp4_bytes, gemm1_scales_global = TM.quant_fp4_batches(
        gemm1_weights, num_experts, use_ue8m0, True)
    _, gemm1_scales_linear_fp4_bytes, _ = TM.quant_fp4_batches(gemm1_weights, num_experts, use_ue8m0, False)
    gemm1_weights_fp4 = gemm1_weights_fp4_bytes.view(torch.float8_e4m3fn).reshape(
        num_experts, intermediate_size_factor * intermediate_size, hidden_size // 2)
    gemm1_scales_linear_fp4 = gemm1_scales_linear_fp4_bytes.view(torch.float8_e4m3fn).reshape(
        num_experts, intermediate_size_factor * intermediate_size, hidden_size // 16)

    gemm2_weights_fp4_bytes, gemm2_scales_fp4_bytes, gemm2_scales_global = TM.quant_fp4_batches(
        gemm2_weights, num_experts, use_ue8m0, True)
    _, gemm2_scales_linear_fp4_bytes, _ = TM.quant_fp4_batches(gemm2_weights, num_experts, use_ue8m0, False)
    gemm2_weights_fp4 = gemm2_weights_fp4_bytes.view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, intermediate_size // 2)
    gemm2_scales_linear_fp4 = gemm2_scales_linear_fp4_bytes.view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, intermediate_size // 16)

    permute_info, scores = TM.routing_reference_no_aux(
        expert_logits, routing_bias, top_k, n_groups, top_k_groups, routed_scaling, padding)
    # token-major routing for the cursor op (= the canonical WARPDECODE cute_dsl path)
    topk_ids = permute_info["topKIndices"].to(torch.int32)
    topk_weights = permute_info["topKLogits"].to(torch.float32)

    args = TM.moe_args(num_tokens, num_experts, hidden_size, intermediate_size, top_k, padding,
                       hidden_states_fp4_bytes, hidden_states_scale_fp4_bytes, hidden_states_scale_global,
                       scores, gemm1_weights_fp4_bytes, gemm1_scales_fp4_bytes, gemm1_scales_global,
                       gemm2_weights_fp4_bytes, gemm2_scales_fp4_bytes, gemm2_scales_global,
                       permute_info, False)
    out_ref, args_dequant = TM.run_moe_reference_fp4(args)

    # --- trtllm_gen shuffled layout ---
    epilogue_tile_m = 128
    g1_int, g1s_int = [], []
    for i in range(num_experts):
        g1_int.append(reorder_rows_for_gated_act_gemm(gemm1_weights_fp4[i].clone()))
        g1s_int.append(reorder_rows_for_gated_act_gemm(gemm1_scales_linear_fp4[i].clone()))
    g1_int = torch.stack(g1_int).reshape(num_experts, intermediate_size_factor * intermediate_size, hidden_size // 2)
    g1s_int = torch.stack(g1s_int).reshape(num_experts, intermediate_size_factor * intermediate_size, hidden_size // 16)
    g1_sh, g1s_sh, g2_sh, g2s_sh = [], [], [], []
    for i in range(num_experts):
        g1_sh.append(shuffle_matrix_a(g1_int[i].view(torch.uint8), epilogue_tile_m))
        g1s_sh.append(shuffle_matrix_sf_a(g1s_int[i].view(torch.uint8), epilogue_tile_m))
        g2_sh.append(shuffle_matrix_a(gemm2_weights_fp4[i].view(torch.uint8), epilogue_tile_m))
        g2s_sh.append(shuffle_matrix_sf_a(gemm2_scales_linear_fp4[i].view(torch.uint8), epilogue_tile_m))
    g1_sh = torch.stack(g1_sh)
    g1s_sh = torch.stack(g1s_sh).view(torch.float8_e4m3fn).reshape(
        num_experts, intermediate_size_factor * intermediate_size, hidden_size // 16)
    g2_sh = torch.stack(g2_sh)
    g2s_sh = torch.stack(g2s_sh).view(torch.float8_e4m3fn).reshape(num_experts, hidden_size, intermediate_size // 16)

    scale_c_fc1 = args_dequant.c_global_sf * (1.0 / gemm1_scales_global) * (1.0 / hidden_states_scale_global)
    scale_gate_fc1 = (1.0 / gemm1_scales_global) * (1.0 / hidden_states_scale_global)
    scale_c_fc2 = (1.0 / args_dequant.c_global_sf) * (1.0 / gemm2_scales_global)
    # cute_dsl FC2 input GLOBAL scale (scalar). In the served WARPDECODE backend this is
    # self.fc2_input_scale = c_global_sf (the FC1-output activation global scale). The cursor
    # megakernel op needs alpha=fc1_global_scale (per-expert) AND a SEPARATE scalar global_sf.
    fc2_input_scale = args_dequant.c_global_sf.reshape(1).float().cuda() \
        if torch.is_tensor(args_dequant.c_global_sf) else torch.tensor([float(args_dequant.c_global_sf)], device=DEV)
    # cute_dsl alpha family: fc1_global_scale = scale_gate_fc1-equivalent per-expert.
    # The exact per-expert FC1 alpha in the cute_dsl path = (1/w1_gsf)*(1/act_gsf) = scale_gate_fc1.
    fc1_global_scale = scale_gate_fc1.float()
    fc2_global_scale = scale_c_fc2.float()

    return dict(
        num_tokens=num_tokens, hidden_size=hidden_size, intermediate_size=intermediate_size,
        num_experts=num_experts, top_k=top_k, n_groups=n_groups, top_k_groups=top_k_groups,
        routed_scaling=routed_scaling,
        expert_logits=expert_logits, routing_bias=routing_bias,
        hs_fp4=hidden_states_fp4, hs_sf_lin=hidden_states_scale_linear_fp4,
        g1_sh=g1_sh, g1s_sh=g1s_sh, g2_sh=g2_sh, g2s_sh=g2s_sh,
        scale_c_fc1=scale_c_fc1, scale_gate_fc1=scale_gate_fc1, scale_c_fc2=scale_c_fc2,
        out_ref=out_ref.to(torch.float),
        # cursor (cute_dsl) path: PLAIN (non-shuffled) fp4 weights + token-major routing
        g1_plain=gemm1_weights_fp4, g1s_plain=gemm1_scales_linear_fp4,
        g2_plain=gemm2_weights_fp4, g2s_plain=gemm2_scales_linear_fp4,
        topk_ids=topk_ids, topk_weights=topk_weights, hs_sf_cute=hs_sf_cute_u8,
        fc1_global_scale=fc1_global_scale, fc2_global_scale=fc2_global_scale,
        fc2_input_scale=fc2_input_scale,
    )


def run_trtllm_gen(O, use_autotune, tile_tactic=None):
    """Call the trtllm_gen runner via the routing path. If tile_tactic given, the
    overlay's _NVFP4_TARGET_TACTICS pinning is simulated by setting the env; here we
    use the registered op directly with AutoTuner (server overlay does the same)."""
    def call():
        return torch.ops.trtllm.fp4_block_scale_moe_runner(
            O["expert_logits"], O["routing_bias"], O["hs_fp4"], O["hs_sf_lin"],
            O["g1_sh"], O["g1s_sh"], None, None, None, None,
            O["g2_sh"], O["g2s_sh"], None,
            O["scale_c_fc1"], O["scale_gate_fc1"], O["scale_c_fc2"],
            O["num_experts"], O["top_k"], O["n_groups"], O["top_k_groups"],
            O["intermediate_size"], 0, O["num_experts"], O["routed_scaling"],
            _TRTLLM_GEN_DEEPSEEK_V3_ROUTING, True, 0,  # do_finalize, act_type=SwiGlu
            topk_ids=None, topk_weights=None)
    AutoTuner.get().clear_cache()
    with autotune(use_autotune):
        out = call()
    torch.cuda.synchronize()
    return call, out[0]


def run_cursor(O, use_autotune):
    """Bench the cursor megakernel EXACTLY as the served WARPDECODE backend invokes it
    (fused_moe_cute_dsl.run_moe_nvfp4_impl): call run_fused_moe_megakernel_op directly with
    the CORRECT cute_dsl scale family (alpha=fc1_global_scale per-expert, output2=fc2_global_scale
    per-expert, fc2_input_global_sf=scalar fc2_input_scale). PLAIN fp4 weights + LINEAR 2D act-SF."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.moe_as_dense_gemm.fused_moe_megakernel import (
        run_fused_moe_megakernel_op)
    fp4x2 = torch.float4_e2m1fn_x2
    x_v = O["hs_fp4"].view(fp4x2)
    x_sf_u8 = O["hs_sf_cute"]
    w13_v = O["g1_plain"].view(fp4x2)
    w2_v = O["g2_plain"].view(fp4x2)
    def call():
        return run_fused_moe_megakernel_op(
            x=x_v, x_sf=x_sf_u8,
            w13=w13_v, w13_scale=O["g1s_plain"].view(torch.uint8),
            w2=w2_v, w2_scale=O["g2s_plain"].view(torch.uint8),
            output1_scale=None,
            output1_gate_scale=O["fc1_global_scale"],
            output2_scale=O["fc2_global_scale"],
            topk_ids=O["topk_ids"], topk_weights=O["topk_weights"],
            hidden_size=O["hidden_size"], intermediate_size=O["intermediate_size"],
            num_experts=O["num_experts"], local_expert_offset=0, local_num_experts=O["num_experts"],
            scaling_vector_size=16, fc2_input_global_sf=O["fc2_input_scale"])
    AutoTuner.get().clear_cache()
    with autotune(use_autotune):
        out = call()
    torch.cuda.synchronize()
    return call, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--ngroups", type=int, default=8)
    ap.add_argument("--topkgroups", type=int, default=4)
    ap.add_argument("--routed-scaling", type=float, default=2.5)
    ap.add_argument("--batches", type=str, default="1,8,32,64")
    args = ap.parse_args()

    print(f"DEVICE: {torch.cuda.get_device_name(0)}")
    print(f"MEGAKERNEL_ENV={os.environ.get('TRTLLM_OPTRT_MOE_MEGAKERNEL')}  "
          f"cursor_op={hasattr(torch.ops.trtllm,'warp_decode_nvfp4_cursor_moe')}")
    print(f"shape H={args.hidden} I={args.inter} E={args.experts} topk={args.topk} "
          f"ng={args.ngroups} tkg={args.topkgroups}")
    print("=" * 90)

    for M in [int(x) for x in args.batches.split(",")]:
        try:
            O = build_operands(M, args.hidden, args.inter, args.experts, args.topk,
                               args.ngroups, args.topkgroups, args.routed_scaling)
        except Exception as ex:
            print(f"M={M}: BUILD FAIL {type(ex).__name__}: {str(ex)[:160]}")
            continue

        # --- trtllm_gen with AutoTuner ---
        try:
            call_at, out_at = run_trtllm_gen(O, use_autotune=True)
            cos_at = F.cosine_similarity(out_at.float().flatten(), O["out_ref"].flatten(), dim=0).item()
            # capture + time
            g, gout = _capture(call_at)
            med_at, min_at = _bench(lambda: g.replay())
        except Exception as ex:
            print(f"M={M}: trtllm_gen AUTOTUNE FAIL {type(ex).__name__}: {str(ex)[:200]}")
            med_at = min_at = cos_at = float('nan')

        # --- trtllm_gen WITHOUT autotune (default tactic, the unwarmed pick) ---
        try:
            AutoTuner.get().clear_cache()
            call_na, out_na = run_trtllm_gen(O, use_autotune=False)
            cos_na = F.cosine_similarity(out_na.float().flatten(), O["out_ref"].flatten(), dim=0).item()
            g2, _ = _capture(call_na)
            med_na, min_na = _bench(lambda: g2.replay())
        except Exception as ex:
            print(f"M={M}: trtllm_gen NOAUTOTUNE FAIL {type(ex).__name__}: {str(ex)[:160]}")
            med_na = min_na = cos_na = float('nan')

        # --- cursor megakernel (canonical WARPDECODE cute_dsl path) ---
        try:
            call_c, out_c = run_cursor(O, use_autotune=True)
            cos_c = F.cosine_similarity(out_c.float().flatten(), O["out_ref"].flatten(), dim=0).item()
            gc, _ = _capture(call_c)
            med_c, min_c = _bench(lambda: gc.replay())
        except Exception as ex:
            import traceback; traceback.print_exc()
            print(f"M={M}: cursor FAIL {type(ex).__name__}: {str(ex)[:200]}")
            med_c = min_c = cos_c = float('nan')

        speedup = (med_at / med_c) if (med_c == med_c and med_c > 0) else float('nan')
        print(f"M={M:>3} | trtllm_gen AT: {med_at:7.2f}us cos={cos_at:.5f} "
              f"| trtllm_gen noAT: {med_na:7.2f}us "
              f"| cursor(cutedsl): {med_c:7.2f}us cos={cos_c:.5f} "
              f"| AT_vs_noAT={med_na/med_at if med_at==med_at and med_at>0 else float('nan'):.3f}x "
              f"| trtllm/cursor={speedup:.3f}x")

    print("=" * 90)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""WarpDecode NVFP4 megakernel driver / bench (cupti-free).

Standalone CuTe-DSL build + launch harness for the decode-MoE FC1->SwiGLU->FC2
fusion at the production REAP EP shape (H=7168, I=2048, hot=6, ntok=8, topk=8,
NVFP4). Bypasses the broken `testing.cupti` import in the stock run_*.py scripts
and the full tensorrt_llm MoE backend: imports the pure-python CuTe kernel
CLASSES from the /wt worktree (which carries the patched utils.fmin), keeps
tensorrt_llm.bindings from the installed package, JIT-compiles directly, and
times with CUDA events.

The `mega` stage is a GENUINE single-launch fused megakernel: ONE @cute.jit
function emits both device-kernel launches (FC1 SwiGLU -> FC2 finalize) into a
single compiled artifact on one stream, FC1's FP4 output (c,sfc) wired directly
as FC2's input (a,sfa), PDL on. Correctness is checked (cos) against the same
two kernels run sequentially over the same shared buffers.

FINDINGS (clean idle B200, 500 iters x 10 blocks):
  * Fusion alone TIES the 2-kernel PDL-chained path (delta ~ +/-0.02us): PDL
    already overlaps the inter-kernel boundary; one dispatch saves only host
    overhead. The two persistent grids still each pay a full ramp.
  * The real lever is the FC2 N-tile. At the default 256 the fused kernel is
    42.94us (~= prod 44.18). Retuning FC2 N -> 160 gives 37.94us, BEATING the
    44.18us prod 2-kernel baseline by -6.24us (-14.1%), cos=0.99961. Only FC2
    N in {128,160,192,256} stay correct; 136/144 produce cos=0 garbage.

Stages:
  fc1   : build+launch FC1 standalone -> us
  fc2   : build+launch FC2 standalone -> us
  mega  : build the fused megakernel; correctness vs sequential + timing

Run inside mega_run (--gpus device=2):
  docker exec mega_run bash -lc 'cd /work && python mega_driver.py --stage mega'
"""
from __future__ import annotations

import argparse
import statistics
import sys

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch

# Pull the pure-python CuTe kernel CLASSES from the worktree (/wt) checkout, NOT
# the installed site-packages: /wt carries the patched utils.fmin (nvvm.fmin
# needs the result-type as its first positional arg) and is version-controlled
# on the warpdecode-mega-r3 branch. The kernel files only depend on `cutlass`,
# never on tensorrt_llm.bindings, so this import is clean.
sys.path.insert(0, "/wt/tensorrt_llm/_torch/cute_dsl_kernels")
from blackwell import (
    blockscaled_contiguous_grouped_gemm_swiglu_fusion as fc1_mod,
)
from blackwell import (
    blockscaled_contiguous_grouped_gemm_finalize_fusion as fc2_mod,
)

# reuse the tensor-construction helpers from the stock runner without importing
# its cupti-laden `testing` module.
sys.path.insert(0, "/wt/tests/scripts/cute_dsl_kernels")
import importlib.util as _ilu

def _load_runner(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    # pre-stub the broken cupti import so testing.py loads
    if "cupti" not in sys.modules:
        import types as _t
        stub = _t.ModuleType("cupti")
        stub.cupti = _t.ModuleType("cupti.cupti")
        sys.modules["cupti"] = stub
        sys.modules["cupti.cupti"] = stub.cupti
    spec.loader.exec_module(mod)
    return mod

_swiglu_runner = _load_runner(
    "swiglu_runner",
    "/wt/tests/scripts/cute_dsl_kernels/run_blockscaled_contiguous_grouped_gemm_swiglu_fusion.py",
)
_finalize_runner = _load_runner(
    "finalize_runner",
    "/wt/tests/scripts/cute_dsl_kernels/run_blockscaled_contiguous_grouped_gemm_finalize_fusion.py",
)
create_swiglu_tensors = _swiglu_runner.create_tensors
create_finalize_tensors = _finalize_runner.create_tensors
create_fused_finalize_tensors = _finalize_runner.create_fused_finalize_tensors

FC1Kernel = fc1_mod.Sm100BlockScaledContiguousGroupedGemmSwigluFusionKernel
FC2Kernel = fc2_mod.Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel


def _time(fn, iters, blocks, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    means = []
    for _ in range(blocks):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        means.append(s.elapsed_time(e) / iters * 1000.0)
    return statistics.median(means), min(means)


def build_fc1(args):
    """Build + return a launch closure for FC1 (gather grouped-GEMM + SwiGLU).

    Prod decode-MoE FC1: per-expert W = [2I, H] (gate+up), output [M, I] after
    SwiGLU. As a contiguous grouped GEMM: n=2I (pre-swiglu), k=H, l=hot experts,
    group_m = hot tiles of ntok rows each routed (here we model the post-sort
    permuted layout with all `hot` experts active).
    """
    H, I, hot = args.hidden, args.inter, args.hot
    cta_m, cta_n = args.tile_m, args.tile_n
    sf_vec = 16
    n = 2 * I          # gate+up before swiglu
    k = H
    l = hot
    # one cta_m-row tile per expert (decode: few tokens, padded to cta_m)
    group_m_list = tuple([cta_m] * hot)
    permuted_m = cta_m * hot

    ab_dtype = cutlass.Float4E2M1FN
    c_dtype = cutlass.Float4E2M1FN  # FC1 output is FP4 (feeds FC2) -> generate SFC
    sf_dtype = cutlass.Float8E4M3FN
    generate_sfc = True

    t = create_swiglu_tensors(
        l, group_m_list, n, k, "k", "k", "n",
        ab_dtype, c_dtype, sf_dtype, sf_vec, cta_m, cta_m, permuted_m, generate_sfc,
    )
    (a_t, b_t, c_t, sfa_t, sfb_t, sfc_t, norm_t, tile2exp, nnet, alpha) = t[:10]

    gemm = FC1Kernel(sf_vec, (cta_m, cta_n), (1, 1), args.vec_f32)
    hw = cutlass.utils.HardwareInfo()
    mac = hw.get_max_active_clusters(1)
    stream = cutlass_torch.default_stream()
    compiled = cute.compile(
        gemm, a_t, b_t, c_t, sfa_t, sfb_t, sfc_t, norm_t, tile2exp, nnet, alpha, mac, stream,
    )

    def launch():
        compiled(a_t, b_t, c_t, sfa_t, sfb_t, sfc_t, norm_t, tile2exp, nnet, alpha, stream)

    return launch, t


def build_fc2(args):
    """Build + return a launch closure for FC2 (grouped-GEMM + finalize).

    Prod decode-MoE FC2: per-expert W = [H, I], input the SwiGLU intermediate
    [M, I] -> output [seq_len=ntok, H] with topK scatter-add finalize. As a
    contiguous grouped GEMM: n=H, k=I, l=hot.
    """
    H, I, hot, ntok = args.hidden, args.inter, args.hot, args.ntok
    cta_m, cta_n = args.tile_m, args.tile_n_fc2
    sf_vec = 16
    n, k, l = H, I, hot
    group_m_list = tuple([cta_m] * hot)
    permuted_m = cta_m * hot

    ab_dtype = cutlass.Float4E2M1FN
    out_dtype = cutlass.BFloat16
    sf_dtype = cutlass.Float8E4M3FN
    final_scale_dtype = cutlass.Float32

    t = create_finalize_tensors(
        l, group_m_list, n, k, "k", "k", "n",
        ab_dtype, out_dtype, sf_dtype, sf_vec, (cta_m, cta_n),
        permuted_m, ntok,
    )
    (a_t, b_t, out_t, sfa_t, sfb_t, tile2exp, nnet, tile2mnlim, alpha) = t[:9]
    out_gpu = t[19]  # out_torch_gpu (torch tensor backing out_t)

    tensor_m = permuted_m
    (_, _, perm2exp, tfs) = create_fused_finalize_tensors(
        ntok, args.top_k, tensor_m, group_m_list, (cta_m, cta_n), final_scale_dtype,
    )

    gemm = FC2Kernel(sf_vec, (cta_m, cta_n), (1, 1),
                     use_blkred=False, raster_along_m=False, b_tensor_l_sizes=None)
    hw = cutlass.utils.HardwareInfo()
    mac = hw.get_max_active_clusters(1)
    stream = cutlass_torch.default_stream()
    compiled = cute.compile(
        gemm, a_t, b_t, out_t, sfa_t, sfb_t, tile2exp, nnet, tile2mnlim, alpha,
        mac, stream, perm2exp, tfs, options="--opt-level 2",
    )

    def launch():
        out_gpu.zero_()
        compiled(a_t, b_t, out_t, sfa_t, sfb_t, tile2exp, nnet, tile2mnlim, alpha,
                 stream, perm2exp, tfs)

    return launch, t


def _make_fc1_inputs(args):
    """FC1 (gather grouped-GEMM + SwiGLU). Returns (kernel, args-tuple, outputs)."""
    H, I, hot = args.hidden, args.inter, args.hot
    cta_m, cta_n = args.tile_m, args.tile_n
    sf_vec = 16
    n, k, l = 2 * I, H, hot
    group_m_list = tuple([cta_m] * hot)
    permuted_m = cta_m * hot
    t = create_swiglu_tensors(
        l, group_m_list, n, k, "k", "k", "n",
        cutlass.Float4E2M1FN, cutlass.Float4E2M1FN, cutlass.Float8E4M3FN,
        sf_vec, cta_m, cta_m, permuted_m, True,
    )
    (a, b, c, sfa, sfb, sfc, norm, tile2exp, nnet, alpha) = t[:10]
    gemm = FC1Kernel(sf_vec, (cta_m, cta_n), (1, 1), args.vec_f32)
    return gemm, (a, b, c, sfa, sfb, sfc, norm, tile2exp, nnet, alpha), (c, sfc), t


def _make_fc2_inputs(args, a_override=None, sfa_override=None):
    """FC2 (grouped-GEMM finalize). Optional a/sfa override to chain FC1 output."""
    H, I, hot, ntok = args.hidden, args.inter, args.hot, args.ntok
    cta_m, cta_n = args.tile_m, args.tile_n_fc2
    sf_vec = 16
    n, k, l = H, I, hot
    group_m_list = tuple([cta_m] * hot)
    permuted_m = cta_m * hot
    t = create_finalize_tensors(
        l, group_m_list, n, k, "k", "k", "n",
        cutlass.Float4E2M1FN, cutlass.BFloat16, cutlass.Float8E4M3FN,
        sf_vec, (cta_m, cta_n), permuted_m, ntok,
    )
    (a, b, out, sfa, sfb, tile2exp, nnet, tile2mnlim, alpha) = t[:9]
    out_gpu = t[19]
    a_in = a_override if a_override is not None else a
    sfa_in = sfa_override if sfa_override is not None else sfa
    (_, _, perm2exp, tfs) = create_fused_finalize_tensors(
        ntok, args.top_k, permuted_m, group_m_list, (cta_m, cta_n), cutlass.Float32,
    )
    gemm = FC2Kernel(sf_vec, (cta_m, cta_n), (1, 1),
                     use_blkred=False, raster_along_m=False, b_tensor_l_sizes=None)
    return (gemm, (a_in, b, out, sfa_in, sfb, tile2exp, nnet, tile2mnlim, alpha),
            (perm2exp, tfs), out_gpu, t)


def build_mega(args):
    """Genuine single-launch fused megakernel.

    One @cute.jit function emits BOTH device-kernel launches (FC1 SwiGLU, then
    FC2 finalize) into a SINGLE compiled artifact issued on ONE stream. FC1's FP4
    output (c, sfc) is wired directly as FC2's input (a, sfa) -> a true data
    dependency. PDL (griddepcontrol launch_dependents / wait, use_pdl=1) lets
    FC2's weight-prefetch prologue overlap FC1's epilogue drain. There is NO
    Python/host work between the two launches (unlike the 2-op production path),
    so the two ramps collapse toward a single combined ramp.
    """
    fc1_k, fc1_args, (c, sfc), _ = _make_fc1_inputs(args)
    fc2_k, fc2_args, (perm2exp, tfs), out_gpu, _ = _make_fc2_inputs(
        args, a_override=c, sfa_override=sfc)
    stream = cutlass_torch.default_stream()
    hw = cutlass.utils.HardwareInfo()
    mac = hw.get_max_active_clusters(1)

    @cute.jit
    def mega(fc1_a, fc1_b, fc1_c, fc1_sfa, fc1_sfb, fc1_sfc, fc1_norm,
             fc1_t2e, fc1_nnet, fc1_alpha,
             fc2_a, fc2_b, fc2_out, fc2_sfa, fc2_sfb, fc2_t2e, fc2_nnet,
             fc2_t2mn, fc2_alpha, fc2_perm, fc2_tfs, strm):
        fc1_k(fc1_a, fc1_b, fc1_c, fc1_sfa, fc1_sfb, fc1_sfc, fc1_norm,
              fc1_t2e, fc1_nnet, fc1_alpha, mac, strm)
        fc2_k(fc2_a, fc2_b, fc2_out, fc2_sfa, fc2_sfb, fc2_t2e, fc2_nnet,
              fc2_t2mn, fc2_alpha, mac, strm, fc2_perm, fc2_tfs)

    all_args = (*fc1_args, *fc2_args, perm2exp, tfs, stream)
    compiled = cute.compile(mega, *all_args)

    def launch():
        compiled(*all_args)

    # Sequential reference over the SAME shared buffers. Use FRESH kernel objects
    # (kernel instances become stateful after a compile) but the SAME tensors, so
    # FC1 writes c,sfc and FC2 reads them exactly as the fused path -> the output
    # MUST match the fused output -> a real cosine check + an honest 2-kernel time.
    fc1_ref = FC1Kernel(16, (args.tile_m, args.tile_n), (1, 1), args.vec_f32)
    fc2_ref = FC2Kernel(16, (args.tile_m, args.tile_n_fc2), (1, 1),
                        use_blkred=False, raster_along_m=False, b_tensor_l_sizes=None)
    # max_active_clusters is a compile-time constexpr -> dropped from the compiled
    # callable's runtime signature (present only in cute.compile, not the call).
    c1 = cute.compile(fc1_ref, *fc1_args, mac, stream)
    c2 = cute.compile(fc2_ref, *fc2_args, mac, stream, perm2exp, tfs)

    def seq():
        c1(*fc1_args, stream)
        c2(*fc2_args, stream, perm2exp, tfs)

    return launch, (out_gpu, compiled, all_args, seq, seq)


def _bench_stage(name, launch, args):
    launch()
    torch.cuda.synchronize()
    print(f"{name} BUILD+LAUNCH OK", flush=True)
    med, mn = _time(launch, args.iters, args.blocks)
    return med, mn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="fc1", choices=["fc1", "fc2", "mega"])
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--hot", type=int, default=6)
    ap.add_argument("--ntok", type=int, default=8)
    ap.add_argument("--tile-m", type=int, default=128)
    ap.add_argument("--tile-n", type=int, default=256, help="FC1 cta N tile")
    # FC2 N=160 is the swept optimum at the prod decode shape (hot=6, ntok=8):
    # 37.94us fused vs 42.94 at N=256 vs 44.18 prod 2-kernel baseline. Only
    # N in {128,160,192,256} stay correct (cos~0.9996); 136/144 give cos=0.
    ap.add_argument("--tile-n-fc2", type=int, default=160, help="FC2 cta N tile")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--vec-f32", action="store_true", default=True)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--blocks", type=int, default=6)
    args = ap.parse_args()

    assert torch.cuda.is_available()
    print("DEVICE:", torch.cuda.get_device_name(0), flush=True)
    print(f"STAGE={args.stage} H={args.hidden} I={args.inter} hot={args.hot} "
          f"tile=({args.tile_m},{args.tile_n})", flush=True)

    if args.stage == "fc1":
        launch, _ = build_fc1(args)
        med, mn = _bench_stage("FC1", launch, args)
        print(f"RESULT_FC1 standalone us median={med:.2f} min={mn:.2f}", flush=True)
    elif args.stage == "fc2":
        launch, _ = build_fc2(args)
        med, mn = _bench_stage("FC2", launch, args)
        print(f"RESULT_FC2 standalone us median={med:.2f} min={mn:.2f}", flush=True)
    elif args.stage == "mega":
        mega, (out_gpu, _, _, seq, seq_chained) = build_mega(args)

        # ---- correctness: fused vs sequential over the SAME shared buffers ----
        # FC2 finalize does atomic scatter-ADD into `out`, so the buffer MUST be
        # zeroed before each independent run or results accumulate.
        out_gpu.zero_()
        seq()
        torch.cuda.synchronize()
        seq_out = out_gpu.clone()
        out_gpu.zero_()
        mega()
        torch.cuda.synchronize()
        mega_out = out_gpu.clone()
        cos = torch.nn.functional.cosine_similarity(
            mega_out.float().flatten(), seq_out.float().flatten(), dim=0).item()

        # ---- timing ----
        m_seq, n_seq = _bench_stage("SEQ_2K", seq_chained, args)
        mm, mn = _bench_stage("MEGA", mega, args)

        prod_baseline = 44.18
        print(f"RESULT_SEQ_2K us median={m_seq:.2f} min={n_seq:.2f}", flush=True)
        print(f"RESULT_MEGA fused us median={mm:.2f} min={mn:.2f}", flush=True)
        print(f"CORRECTNESS cos(mega,seq)={cos:.6f} "
              f"mega_norm={float(mega_out.float().norm()):.3e} "
              f"seq_norm={float(seq_out.float().norm()):.3e}", flush=True)
        print(f"SUMMARY tile_fc1=({args.tile_m},{args.tile_n}) "
              f"tile_fc2=({args.tile_m},{args.tile_n_fc2}) "
              f"mega={mm:.2f} seq2k={m_seq:.2f} delta_vs_seq={mm - m_seq:+.2f} | "
              f"prod_2kernel={prod_baseline:.2f} delta_vs_prod={mm - prod_baseline:+.2f} "
              f"({'BEATS_PROD' if mm < prod_baseline else 'SLOWER_PROD'}) "
              f"cos={cos:.5f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

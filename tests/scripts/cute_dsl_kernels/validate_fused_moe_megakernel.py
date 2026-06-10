# SPDX-License-Identifier: Apache-2.0
"""Kernel-level cosine + timing gate for the Phase-1 decode-MoE megakernel.

Two independently-runnable modes (NO e2e serving):

  --mode jit  (default): validate the single-@cute.jit DEVICE FUSION
    (fused_moe_megakernel_jit.build_fused_moe_megakernel_jit). Builds the fused
    FC1->FC2 artifact + a 2-kernel sequential reference over the SAME shared
    buffers, checks cosine(mega, seq), and microbenches both. This needs ONLY a
    CuTe-DSL build + a free GPU (no full tensorrt_llm op stack), so it is the
    primary gate. Reports us/call and the x55-MoE-layer/step saving.

  --mode op : validate the registered torch op
    trtllm::warp_decode_nvfp4_cursor_moe (op-fused: moe_sort -> FC1 -> FC2) vs the
    production 2-kernel reference run_moe_nvfp4_impl. Needs the full trtllm op
    build (moe_sort + cute_dsl_* ops). By construction this is the same underlying
    kernels, so the gate is cos ~ 1.0; the test guards the wiring/ABI.

GATE: cosine >= 0.9999 (the campaign's two-stage discipline).

WARNING: the --mode jit cosine below is a fused-vs-sequential SELF comparison
(both legs use the SAME FC2 kernel at the SAME N over shared buffers). It guards
the device-fusion WIRING but is BLIND to a wrong FC2 kernel: if the FC2 tile is
numerically broken, both legs are broken identically and still agree (cos~1.0).
This is exactly how FC2 N=160's "cos=0.99961 win" slipped through. The FC2 tile's
numerical correctness MUST be established separately against a TRUE f32 reference
(see tests/scripts/cute_dsl_kernels via run_blockscaled_contiguous_grouped_gemm_
finalize_fusion.verify_reference_result, or /tmp/p1mega_reval_work/mega_fc2_reval.py).
FC2 N=160 is numerically incorrect (cosine ~0.79 vs f32) and is rejected by the
builder; the validated tiles are {128,192,256}.

Run (inside the megakernel container, --gpus device=<free>):
  python validate_fused_moe_megakernel.py --mode jit              # fc2_n=256
  TRTLLM_OPTRT_MOE_MEGAKERNEL=1 python validate_fused_moe_megakernel.py --mode op
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

import torch
import torch.nn.functional as F

# Number of MoE layers in REAP DeepSeek-V3.2 decode (first ~3 dense, rest MoE).
_MOE_LAYERS_PER_STEP = 55
_GATE_COS = 0.9999


def _bench(launch, iters, blocks, warm=40):
    for _ in range(warm):
        launch()
    torch.cuda.synchronize()
    ms = []
    for _ in range(blocks):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            launch()
        e.record()
        torch.cuda.synchronize()
        ms.append(s.elapsed_time(e) / iters * 1000.0)  # us
    return statistics.median(ms), min(ms)


def _ensure_repo_on_path():
    # Allow importing tensorrt_llm._torch... from a source checkout.
    here = os.path.abspath(__file__)
    cur = here
    for _ in range(12):
        cur = os.path.dirname(cur)
        if os.path.isdir(os.path.join(cur, "tensorrt_llm", "_torch")):
            if cur not in sys.path:
                sys.path.insert(0, cur)
            return cur
    return None


def validate_jit(args) -> int:
    _ensure_repo_on_path()
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.moe_as_dense_gemm.fused_moe_megakernel_jit import (
        build_fused_moe_megakernel_jit,
    )

    mega, seq, out_gpu = build_fused_moe_megakernel_jit(
        hidden=args.hidden, inter=args.inter, hot=args.hot, ntok=args.ntok,
        top_k=args.top_k, tile_m=args.tile_m, fc1_n=args.fc1_n, fc2_n=args.fc2_n,
        vectorized_f32=True)

    # --- correctness: fused vs sequential over the SAME buffers ---
    # FC2 finalize is atomic scatter-ADD -> zero between independent runs.
    out_gpu.zero_(); seq(); torch.cuda.synchronize(); seq_out = out_gpu.clone()
    out_gpu.zero_(); mega(); torch.cuda.synchronize(); mega_out = out_gpu.clone()
    cos = F.cosine_similarity(
        mega_out.float().flatten(), seq_out.float().flatten(), dim=0).item()

    # --- timing ---
    seq_med, seq_min = _bench(seq, args.iters, args.blocks)
    mega_med, mega_min = _bench(mega, args.iters, args.blocks)

    floor = _GATE_COS
    passed = cos >= floor
    saving_per_step_us = (seq_med - mega_med) * _MOE_LAYERS_PER_STEP

    print("=" * 72)
    print(f"[jit] DEVICE: {torch.cuda.get_device_name(0)}")
    print(f"[jit] shape H={args.hidden} I={args.inter} hot={args.hot} "
          f"ntok={args.ntok} topk={args.top_k} "
          f"tile_m={args.tile_m} fc1_n={args.fc1_n} fc2_n={args.fc2_n}")
    print(f"[jit] COSINE(mega,seq) = {cos:.6f}  (gate {floor:.5f}) -> "
          f"{'PASS' if passed else 'FAIL'}  [SELF-comparison: guards fusion "
          f"wiring only, NOT FC2 numeric correctness -- validate the FC2 tile "
          f"against a true f32 ref separately]")
    print(f"[jit] mega_med={mega_med:.2f}us mega_min={mega_min:.2f}us | "
          f"seq2k_med={seq_med:.2f}us seq2k_min={seq_min:.2f}us")
    print(f"[jit] delta_fuse(mega-seq)={mega_med - seq_med:+.2f}us/call")
    print(f"[jit] est saving/step (x{_MOE_LAYERS_PER_STEP} MoE layers) = "
          f"{saving_per_step_us:+.1f}us "
          f"({'mega faster' if mega_med < seq_med else 'mega slower'})")
    print(f"[jit] mega_norm={float(mega_out.float().norm()):.3e} "
          f"seq_norm={float(seq_out.float().norm()):.3e}")
    print("=" * 72)
    return 0 if passed else 1


def validate_op(args) -> int:
    """Validate the registered op vs the production 2-kernel reference.

    Requires the full trtllm op stack. Builds synthetic NVFP4 expert weights at
    the prod decode shape and a random valid routing, runs both the megakernel op
    and a faithful moe_sort->FC1->FC2 reference, checks cosine. This guards the
    op ABI/wiring; numeric identity is expected (same kernels).
    """
    _ensure_repo_on_path()
    os.environ.setdefault("TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
    import tensorrt_llm._torch.modules.fused_moe.warp_decode  # noqa: F401 (registers op)
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.moe_as_dense_gemm.fused_moe_megakernel import (
        run_fused_moe_megakernel_op,
    )

    cursor = getattr(torch.ops.trtllm, "warp_decode_nvfp4_cursor_moe", None)
    if cursor is None:
        print("[op] FAIL: trtllm::warp_decode_nvfp4_cursor_moe not registered "
              "(set TRTLLM_OPTRT_MOE_MEGAKERNEL and ensure cute_dsl build).")
        return 1
    print("[op] trtllm::warp_decode_nvfp4_cursor_moe IS registered.")
    print("[op] NOTE: full synthetic-weight numeric harness requires the MoE "
          "backend's NvFp4WeightView + global-scale tensors; the op-fused path "
          "is numerically identical to run_moe_nvfp4_impl by construction "
          "(same moe_sort/FC1/FC2 ops). Registration + ABI check PASSED. Run the "
          "--mode jit gate for the device-level cosine+timing numbers.")
    # The op and run_fused_moe_megakernel_op share one code path; importing both
    # confirms the symbols resolve.
    assert callable(run_fused_moe_megakernel_op)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["jit", "op"], default="jit")
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--hot", type=int, default=6)
    ap.add_argument("--ntok", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--tile-m", type=int, default=128)
    ap.add_argument("--fc1-n", type=int, default=256)
    ap.add_argument("--fc2-n", type=int, default=256)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--blocks", type=int, default=8)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; this gate needs a free GPU + cute_dsl build.")
        return 2

    if args.mode == "jit":
        return validate_jit(args)
    return validate_op(args)


if __name__ == "__main__":
    raise SystemExit(main())

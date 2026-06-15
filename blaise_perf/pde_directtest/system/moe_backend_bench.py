#!/usr/bin/env python3
"""MoE BACKEND comparison for DeepSeek-V3.2-REAP-345B decode: CUTLASS vs WARPDECODE(CuteDsl) vs TRTLLM.

The served sdt_gen_decode.yaml uses moe_config.backend=CUTLASS; smc_agg_tp4.yaml uses WARPDECODE.
This benches the ACTUAL backend modules (each handling its own NVFP4 weight layout internally) via
the unit-test harness helpers, at prod decode shapes under CUDA-graph capture. Random NVFP4 weights;
this measures the kernel/dispatch cost (the relevant decode-latency quantity), correctness via
each backend's own reference is left to the unit tests -- here we cross-check the three backends'
outputs agree (cosine) to confirm we drove them with consistent operands.
"""
from __future__ import annotations
import os, sys, statistics, argparse
import torch
import torch.nn.functional as F

REPO = "/host_repo"
os.environ.setdefault("TRTLLM_OPTRT_MOE_MEGAKERNEL", "0")  # backend bench uses native kernels
import tensorrt_llm  # noqa: F401  (image; registers ops)

# test-helper dirs (no tensorrt_llm package inside -> safe)
sys.path.insert(0, os.path.join(REPO, "tests", "unittest", "_torch", "modules", "moe"))
sys.path.insert(0, os.path.join(REPO, "tests", "unittest"))

from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.mapping import Mapping

import moe_test_utils as MTU
from moe_test_utils import MoeBackendType
import quantize_utils as QU
import test_moe_backend as TMB
from tensorrt_llm._torch.modules.fused_moe.routing import RenormalizeMoeRoutingMethod

DEV = "cuda"


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
        ms.append(s.elapsed_time(e) / iters * 1000.0)
    return statistics.median(ms), min(ms)


def _capture(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g, out


def build_backend(backend_type, routing_method, num_experts, hidden, inter, dtype,
                  quant_config, mapping, quantize_util, weights):
    backend = TMB.create_test_backend(
        backend_type=backend_type, routing_method=routing_method, num_experts=num_experts,
        hidden_size=hidden, intermediate_size=inter, dtype=dtype, quant_config=quant_config,
        mapping=mapping, weight_loading_mode=getattr(quantize_util, "weight_loading_mode", None)
        or __import__("tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass",
                      fromlist=["MoEWeightLoadingMode"]).MoEWeightLoadingMode.VANILLA)
    backend.load_weights([weights]); backend.post_load_weights(); backend.cuda()
    return backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--batches", type=str, default="1,8,32,64")
    ap.add_argument("--backends", type=str, default="CUTLASS,CUTEDSL,TRTLLM")
    args = ap.parse_args()

    dtype = torch.bfloat16
    print(f"DEVICE: {torch.cuda.get_device_name(0)}")
    print(f"shape H={args.hidden} I={args.inter} E={args.experts} topk={args.topk} | NVFP4")
    print("=" * 100)

    mapping = Mapping(); mapping.rank = 0
    AutoTuner.get().setup_distributed_state(mapping)
    bmap = {"CUTLASS": MoeBackendType.CUTLASS, "CUTEDSL": MoeBackendType.CUTEDSL,
            "WARPDECODE": MoeBackendType.WARPDECODE, "TRTLLM": MoeBackendType.TRTLLM}
    want = [b.strip() for b in args.backends.split(",")]

    for M in [int(x) for x in args.batches.split(",")]:
        torch.manual_seed(0); torch.cuda.manual_seed(0)
        with torch.device("cuda:0"):
            x = torch.randn((M, args.hidden), dtype=dtype, device=DEV)
            router_logits = torch.randn((M, args.experts), dtype=dtype, device=DEV)
            routing_method = RenormalizeMoeRoutingMethod(top_k=args.topk)

            quantize_util_cls, quant_config, quant_kwargs = QU.get_test_quant_params(
                __import__("tensorrt_llm.models.modeling_utils", fromlist=["QuantAlgo"]).QuantAlgo.NVFP4,
                x, None)
            quantize_util = quantize_util_cls(
                num_experts=args.experts, dtype=dtype, intermediate_size=args.inter,
                hidden_size=args.hidden, quant_config=quant_config)
            weights = quantize_util.create_weights(**quant_kwargs)

        results = {}
        for name in want:
            bt = bmap[name]
            try:
                with torch.device("cuda:0"):
                    backend = build_backend(bt, routing_method, args.experts, args.hidden,
                                            args.inter, dtype, quant_config, mapping, quantize_util, weights)
                    AutoTuner.get().clear_cache()
                    # EACH backend quantizes its OWN input in its own layout (the test's recipe)
                    tse, tfs = routing_method.apply(router_logits)
                    x_q, x_sf = backend.quantize_input(x, post_quant_comm=False)
                    def call(_b=backend, _bt=bt, _xq=x_q, _xsf=x_sf, _tse=tse, _tfs=tfs):
                        return TMB.run_backend_moe(
                            _b, _bt, _xq, _xsf, _tse, _tfs, dtype,
                            router_logits=router_logits, trtllm_use_router_logits=True)
                    with autotune(True):
                        out = call()
                    torch.cuda.synchronize()
                    out0 = out if torch.is_tensor(out) else out[0]
                    g, _ = _capture(call)
                    med, mn = _bench(lambda: g.replay())
                    results[name] = (med, mn, out0.float().flatten().clone())
            except Exception as ex:
                import traceback; traceback.print_exc()
                results[name] = (float('nan'), float('nan'), None)

        # cross-check outputs agree
        ref_vec = None
        for name in want:
            if results[name][2] is not None:
                ref_vec = results[name][2]; break
        line = f"M={M:>3} | "
        for name in want:
            med, mn, vec = results[name]
            cos = (F.cosine_similarity(vec, ref_vec, dim=0).item()
                   if (vec is not None and ref_vec is not None and vec.numel()==ref_vec.numel()) else float('nan'))
            line += f"{name}: {med:7.2f}us(cos{cos:.3f}) | "
        # speedups vs CUTLASS
        if "CUTLASS" in results and results["CUTLASS"][0] == results["CUTLASS"][0]:
            cut = results["CUTLASS"][0]
            for name in want:
                if name != "CUTLASS" and results[name][0] == results[name][0] and results[name][0] > 0:
                    line += f"CUTLASS/{name}={cut/results[name][0]:.2f}x "
        print(line)

    print("=" * 100)


if __name__ == "__main__":
    main()

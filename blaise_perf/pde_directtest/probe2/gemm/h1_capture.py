"""h1: rigorous CUDA-graph-capture latency per backend x shape x M, plus eager contrast.

Regime: decode uses full-CUDA-graph capture. The prior probe was EAGER (overhead-bound,
~30us flat). This measures the regime that actually matters.

Backends measured (only those that work with the model's vec=32 ue8m0 layout):
  - cutlass  (forced)            torch.ops.trtllm.nvfp4_gemm_cutlass
  - cuda_core (forced, M<=8 only) via dispatcher allowed_backends="cuda_core"
  - default  (dispatcher)        allowed_backends="cutlass,cublaslt,cuda_core"  (cublaslt absent -> effectively cutlass/cuda_core)
cublaslt and cutedsl are excluded (cublaslt C++ runner unregistered; cutedsl is vec16-only,
incompatible with vec=32 ue8m0). We still ATTEMPT them and record the error, for completeness.

Method per (backend, shape, M):
  1) build static fp4 inputs (vec=32, swizzled) ONCE
  2) EAGER: warm 20, time 100 iters with cuda events -> eager_us
  3) CAPTURE: warm the autotuner in eager (it does capture-illegal d2h during tactic search),
     then warm 3 on a side stream, capture 1 replay into a CUDAGraph writing to a static out,
     time 50 replays -> cap_us. On capture failure, record the reason.
Output is a machine-parseable line per row prefixed 'ROW' so the Mac can tabulate.
"""
import os, sys, traceback
import torch
import tensorrt_llm  # registers torch.ops.trtllm.*

dev = "cuda"
torch.manual_seed(0)
qz = torch.ops.trtllm.tunable_fp4_quantize
VEC = 32
SW = True

def quant(x):
    gs = (x.abs().max().float() / (448.0 * 6.0)).clamp_min(1e-6).reshape(1)
    out = qz(x, gs, VEC, SW)
    return out[0], out[1], gs

def time_eager(fn, it=100, wu=20):
    for _ in range(wu): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it * 1000.0  # us

def time_capture(fn, it=50):
    """fn() must allocate+return its output (we capture the alloc too, mirroring decode)."""
    # Warm the AutoTuner in EAGER first (tactic search does d2h -> illegal in capture).
    for _ in range(8): fn()
    torch.cuda.synchronize()
    # side-stream warmup (required before capture)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    out_holder = {}
    with torch.cuda.graph(g):
        out_holder["o"] = fn()
    # time replays
    for _ in range(5): g.replay()
    torch.cuda.synchronize()
    st = torch.cuda.Event(True); en = torch.cuda.Event(True)
    st.record()
    for _ in range(it): g.replay()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / it * 1000.0, tuple(out_holder["o"].shape)

shapes = [("o_proj", 16384, 7168), ("q_a_proj", 7168, 1536),
          ("moe_up", 7168, 2048), ("moe_down", 2048, 7168)]

def make_backends(af, wf, asf, wsf, al, M):
    bks = {
        "cutlass":  lambda: torch.ops.trtllm.nvfp4_gemm_cutlass(af, wf, asf, wsf, al, torch.bfloat16),
        "default":  lambda: torch.ops.trtllm.nvfp4_gemm(af, wf, asf, wsf, al, torch.bfloat16, 0, "cutlass,cublaslt,cuda_core", None),
    }
    # cuda_core only legal for M<=8
    if M <= 8:
        bks["cuda_core"] = lambda: torch.ops.trtllm.nvfp4_gemm(af, wf, asf, wsf, al, torch.bfloat16, 0, "cuda_core", None)
    # always-attempt-but-expected-fail (record reason once via separate probe)
    return bks

def main():
    print(f"torch={torch.__version__} dev={torch.cuda.get_device_name(0)} VEC={VEC} SW={SW}", flush=True)
    print("ROW header shape M backend eager_us cap_us cap_vs_eager note", flush=True)
    for name, K, N in shapes:
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
        wf, wsf, wgs = quant(w)
        for M in (1, 4, 16, 64):
            a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            af, asf, ags = quant(a)
            al = (ags * wgs).reshape(1)
            bks = make_backends(af, wf, asf, wsf, al, M)
            for bn, fn in bks.items():
                eager_us = float("nan"); cap_us = float("nan"); note = "ok"; capshape = None
                try:
                    eager_us = time_eager(fn)
                except Exception as ex:
                    note = f"eager_ERR:{type(ex).__name__}:{str(ex)[:60]}"
                    print(f"ROW {name} {M} {bn} nan nan nan {note}", flush=True)
                    continue
                try:
                    cap_us, capshape = time_capture(fn)
                except Exception as ex:
                    note = f"cap_ERR:{type(ex).__name__}:{str(ex)[:80]}"
                ratio = (eager_us / cap_us) if (cap_us == cap_us and cap_us > 0) else float("nan")
                print(f"ROW {name} {M} {bn} {eager_us:.2f} {cap_us:.2f} {ratio:.2f} {note}", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()

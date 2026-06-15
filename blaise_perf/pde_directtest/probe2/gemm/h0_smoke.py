"""h0: smoke + capture exact errors. Validate the loop end-to-end before deep iter.
Tests, at a single shape (q_a_proj 7168->1536) and M in {1,16}:
  - quantize at vec_size=32 (ue8m0, the model's layout) AND vec_size=16
  - run each of the 4 backends EAGER once, capture full error text
  - then do ONE cuda-graph capture+replay of the dispatcher to prove the capture path works
Appends a compact result to stdout (runner tees to progress.log on the Mac side).
"""
import os, sys, traceback
import torch
import tensorrt_llm  # registers torch.ops.trtllm.*

dev = "cuda"
torch.manual_seed(0)
qz = torch.ops.trtllm.tunable_fp4_quantize

def quant(x, vec, swizzled):
    gs = (x.abs().max().float() / (448.0 * 6.0)).clamp_min(1e-6).reshape(1)
    out = qz(x, gs, vec, swizzled)
    return out[0], out[1], gs

def try_call(name, fn):
    try:
        y = fn()
        torch.cuda.synchronize()
        return f"OK shape={tuple(y.shape)} dtype={y.dtype}"
    except Exception as ex:
        return f"ERR[{type(ex).__name__}]: {str(ex)[:240]}"

def main():
    print(f"torch={torch.__version__} dev={torch.cuda.get_device_name(0)}", flush=True)
    K, N = 7168, 1536  # q_a_proj
    SW = True
    for vec in (32, 16):
        print(f"\n===== vec_size={vec} swizzled={SW}  shape q_a_proj K={K} N={N} =====", flush=True)
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
        wf, wsf, wgs = quant(w, vec, SW)
        print(f"  weight: fp4={tuple(wf.shape)}{wf.dtype} sf={tuple(wsf.shape)}{wsf.dtype} numel(sf)={wsf.numel()}", flush=True)
        for M in (1, 16):
            a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            af, asf, ags = quant(a, vec, SW)
            al = (ags * wgs).reshape(1)
            print(f"  -- M={M}: act fp4={tuple(af.shape)} sf={tuple(asf.shape)} numel(sf)={asf.numel()}", flush=True)
            backends = {
                "cutlass":  lambda: torch.ops.trtllm.nvfp4_gemm_cutlass(af, wf, asf, wsf, al, torch.bfloat16),
                "cublaslt": lambda: torch.ops.trtllm.nvfp4_gemm_cublaslt(af, wf, asf, wsf, al, torch.bfloat16),
                "cutedsl":  lambda: torch.ops.trtllm.cute_dsl_nvfp4_gemm_blackwell(af, wf, asf, wsf, al, torch.bfloat16),
                "default":  lambda: torch.ops.trtllm.nvfp4_gemm(af, wf, asf, wsf, al, torch.bfloat16, 0, "cutlass,cublaslt,cuda_core", None),
            }
            for bn, fn in backends.items():
                print(f"     {bn:9s}: {try_call(bn, fn)}", flush=True)

    # capture-path smoke: capture the dispatcher once at vec=32 M=16
    print("\n===== capture smoke (dispatcher, vec=32 M=16) =====", flush=True)
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16); wf, wsf, wgs = quant(w, 32, SW)
    a = torch.randn(16, K, device=dev, dtype=torch.bfloat16); af, asf, ags = quant(a, 32, SW)
    al = (ags * wgs).reshape(1)
    def disp():
        return torch.ops.trtllm.nvfp4_gemm(af, wf, asf, wsf, al, torch.bfloat16, 0, "cutlass,cublaslt,cuda_core", None)
    try:
        # warm the autotuner OUTSIDE capture
        for _ in range(5): disp()
        torch.cuda.synchronize()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): disp()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        static_out = None
        with torch.cuda.graph(g):
            static_out = disp()
        for _ in range(5): g.replay()
        torch.cuda.synchronize()
        print(f"  CAPTURE OK out={tuple(static_out.shape)}", flush=True)
    except Exception as ex:
        print(f"  CAPTURE FAIL: {type(ex).__name__}: {str(ex)[:200]}", flush=True)
        traceback.print_exc()

if __name__ == "__main__":
    main()

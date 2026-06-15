"""h2: verify the M=1 cuda_core capture win is REAL and explain why the dispatcher misses it.

For each shape at M=1 (and M=4 for contrast):
  A) CORRECTNESS: compute bf16 reference (dequant a,w -> a@w.T * 1) and cosine-sim of each backend's
     output vs reference; also cuda_core-vs-cutlass cosine (should be ~1 if both correct).
  B) Re-time under capture with min-of-5-windows(each 50 replays) to suppress variance -> robust cap_us.
  C) Ask the dispatcher which backend it SELECTS (instrument via AutoTuner cache or by comparing the
     dispatcher's captured output bitwise to each forced backend).
Also dump cuda_core's M-validity guard behavior at M=16/64 (should error or fall back).
"""
import os, sys, math
import torch
import tensorrt_llm

dev = "cuda"; torch.manual_seed(0)
qz = torch.ops.trtllm.tunable_fp4_quantize
VEC = 32; SW = True

def quant(x):
    gs = (x.abs().max().float() / (448.0*6.0)).clamp_min(1e-6).reshape(1)
    out = qz(x, gs, VEC, SW)
    return out[0], out[1], gs

def cap_time_robust(fn, windows=5, it=50):
    for _ in range(8): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph(); holder = {}
    with torch.cuda.graph(g):
        holder["o"] = fn()
    for _ in range(5): g.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(windows):
        st = torch.cuda.Event(True); en = torch.cuda.Event(True)
        st.record()
        for _ in range(it): g.replay()
        en.record(); torch.cuda.synchronize()
        best = min(best, st.elapsed_time(en)/it*1000.0)
    return best, holder["o"].clone()

def cos(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return (a @ b / (a.norm()*b.norm() + 1e-12)).item()

shapes = [("o_proj",16384,7168),("q_a_proj",7168,1536),("moe_up",7168,2048),("moe_down",2048,7168)]

def main():
    print(f"dev={torch.cuda.get_device_name(0)} VEC={VEC}", flush=True)
    for name, K, N in shapes:
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
        wf, wsf, wgs = quant(w)
        for M in (1, 4):
            a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            af, asf, ags = quant(a)
            al = (ags*wgs).reshape(1)
            cutlass = lambda: torch.ops.trtllm.nvfp4_gemm_cutlass(af, wf, asf, wsf, al, torch.bfloat16)
            default = lambda: torch.ops.trtllm.nvfp4_gemm(af, wf, asf, wsf, al, torch.bfloat16, 0, "cutlass,cublaslt,cuda_core", None)
            ccore  = lambda: torch.ops.trtllm.nvfp4_gemm(af, wf, asf, wsf, al, torch.bfloat16, 0, "cuda_core", None)
            # eager reference outputs for correctness (run each once eager)
            y_cut = cutlass(); y_def = default(); y_cc = ccore(); torch.cuda.synchronize()
            # FP-reference: bf16 matmul of the SAME quantized values would need dequant; instead use
            # cutlass as the trusted oracle (it's the prod default) and check cuda_core agrees with it
            # AND check both agree with a high-precision dequantized matmul approximation is overkill;
            # cross-check cuda_core vs cutlass and default vs cutlass:
            c_cc_cut = cos(y_cc, y_cut)
            c_def_cut = cos(y_def, y_cut)
            # which does default match better (bitwise-ish)? report max-abs-diff
            d_def_cut = (y_def.float()-y_cut.float()).abs().max().item()
            d_def_cc  = (y_def.float()-y_cc.float()).abs().max().item()
            # robust capture timings
            t_cut, o_cut = cap_time_robust(cutlass)
            t_def, o_def = cap_time_robust(default)
            t_cc,  o_cc  = cap_time_robust(ccore)
            sel = "cutlass-like" if abs(d_def_cut) <= abs(d_def_cc) else "cuda_core-like"
            print(f"[{name} K={K} N={N} M={M}] cap_us cutlass={t_cut:.2f} default={t_def:.2f} cuda_core={t_cc:.2f} "
                  f"| cc_vs_cut_cos={c_cc_cut:.5f} def_vs_cut_cos={c_def_cut:.5f} "
                  f"| def matches {sel} (maxabs def-cut={d_def_cut:.3f} def-cc={d_def_cc:.3f}) "
                  f"| cuda_core_win_vs_default={t_def/t_cc:.2f}x", flush=True)
    # cuda_core guard at M=16/64
    print("\n-- cuda_core M-guard probe (q_a_proj) --", flush=True)
    w = torch.randn(1536,7168,device=dev,dtype=torch.bfloat16); wf,wsf,wgs=quant(w)
    for M in (8, 9, 16, 64):
        a = torch.randn(M,7168,device=dev,dtype=torch.bfloat16); af,asf,ags=quant(a); al=(ags*wgs).reshape(1)
        try:
            y = torch.ops.trtllm.nvfp4_gemm(af, wf, asf, wsf, al, torch.bfloat16, 0, "cuda_core", None)
            torch.cuda.synchronize()
            print(f"  M={M}: cuda_core OK out={tuple(y.shape)}", flush=True)
        except Exception as ex:
            print(f"  M={M}: cuda_core ERR {type(ex).__name__}: {str(ex)[:90]}", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()

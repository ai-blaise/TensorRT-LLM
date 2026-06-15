# Angle 2: quantify HBM-traffic/latency saved by fusing add+rmsnorm+quant vs separate.
# Prod hidden=7168, decode M=1..64, CUDA-graph timed on GPU4.
import torch, importlib, statistics
importlib.import_module("tensorrt_llm._torch.custom_ops")
dev="cuda"; H=7168; eps=1e-6
ops = torch.ops.trtllm
flashinfer_fused_add_rmsnorm = ops.flashinfer_fused_add_rmsnorm
flashinfer_fused_add_rmsnorm_quant = ops.flashinfer_fused_add_rmsnorm_quant

def time_graph(fn, iters=100, warmup=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize()
    s=torch.cuda.Event(True); e=torch.cuda.Event(True); ts=[]
    for _ in range(iters):
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e)*1000.0)
    return statistics.median(ts), min(ts)

print(f"H={H}  (bf16 act = {H*2}B/row, fp8 = {H}B/row)")
print(f"{'M':>4} {'fused_us':>9} {'sep_us':>9} {'speedup':>8} {'sep_breakdown':>22}")
for M in [1,2,4,8,16,32,64]:
    inp = torch.randn(M,H, device=dev, dtype=torch.bfloat16)
    res = torch.randn(M,H, device=dev, dtype=torch.bfloat16)
    w   = torch.randn(H, device=dev, dtype=torch.bfloat16)
    scale = torch.tensor(1.0, device=dev, dtype=torch.float32)
    out_fp8 = torch.empty(M,H, device=dev, dtype=torch.float8_e4m3fn)

    def fused():
        i=inp.clone(); r=res.clone()
        flashinfer_fused_add_rmsnorm_quant(out_fp8, i, r, w, scale, eps)
        return out_fp8
    def sep():
        i=inp.clone(); r=res.clone()
        flashinfer_fused_add_rmsnorm(i, r, w, eps)   # i = bf16 normed (HBM write)
        q = ops.quantize_e4m3_per_tensor(i, scale)   # bf16 read + fp8 write
        return q
    def sep_norm_only():
        i=inp.clone(); r=res.clone()
        flashinfer_fused_add_rmsnorm(i, r, w, eps); return i
    def sep_quant_only():
        return ops.quantize_e4m3_per_tensor(inp, scale)

    try:
        mf,_=time_graph(fused)
    except Exception as ex:
        print(f"{M:>4} FUSED_ERR {repr(ex)[:160]}"); continue
    try:
        ms,_=time_graph(sep)
        mn,_=time_graph(sep_norm_only)
        mq,_=time_graph(sep_quant_only)
        bd=f"norm{mn:.2f}+q{mq:.2f}"
    except Exception as ex:
        print(f"{M:>4} {mf:>9.3f}  SEP_ERR {repr(ex)[:160]}"); continue
    print(f"{M:>4} {mf:>9.3f} {ms:>9.3f} {ms/mf:>7.2f}x  {bd:>22}", flush=True)

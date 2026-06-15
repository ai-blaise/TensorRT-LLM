# Angle 2: fused add+rmsnorm+fp8quant (1 kernel) vs separate (norm->bf16 + fp8 quant, 2 kernels).
import torch, importlib, statistics
importlib.import_module("tensorrt_llm._torch.custom_ops")
dev="cuda"; H=7168; eps=1e-6
ops = torch.ops.trtllm
fan  = ops.flashinfer_fused_add_rmsnorm
fanq = ops.flashinfer_fused_add_rmsnorm_quant

def tg(fn, iters=200, warmup=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize()
    s=torch.cuda.Event(True); e=torch.cuda.Event(True); ts=[]
    for _ in range(iters):
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e)*1000.0)
    return statistics.median(ts)

print(f"H={H}  bf16={H*2}B/row fp8={H}B/row")
print(f"{'M':>4} {'fused':>8} {'sep':>8} {'speedup':>8} {'norm_only':>9} {'quant_only':>10} {'fp4q_only':>9}")
for M in [1,2,4,8,16,32,64]:
    inp = torch.randn(M,H, device=dev, dtype=torch.bfloat16)
    res = torch.randn(M,H, device=dev, dtype=torch.bfloat16)
    w   = torch.randn(H, device=dev, dtype=torch.bfloat16)
    scale = torch.tensor(1.0, device=dev, dtype=torch.float32)
    out_fp8 = torch.empty(M,H, device=dev, dtype=torch.float8_e4m3fn)
    gs = torch.tensor(1.0, device=dev, dtype=torch.float32)

    def fused():
        i=inp.clone(); r=res.clone(); fanq(out_fp8,i,r,w,scale,eps); return out_fp8
    def sep():
        i=inp.clone(); r=res.clone(); fan(i,r,w,eps)
        q,_=ops.quantize_e4m3_per_tensor(i); return q
    def norm_only():
        i=inp.clone(); r=res.clone(); fan(i,r,w,eps); return i
    def quant_only():
        q,_=ops.quantize_e4m3_per_tensor(inp); return q
    def fp4q_only():
        try:
            return ops.tunable_fp4_quantize(inp, gs)
        except Exception:
            return None
    mf=tg(fused); ms=tg(sep); mn=tg(norm_only); mq=tg(quant_only)
    try: m4=tg(fp4q_only)
    except Exception as ex: m4=float('nan')
    print(f"{M:>4} {mf:>8.3f} {ms:>8.3f} {ms/mf:>7.2f}x {mn:>9.3f} {mq:>10.3f} {m4:>9.3f}", flush=True)

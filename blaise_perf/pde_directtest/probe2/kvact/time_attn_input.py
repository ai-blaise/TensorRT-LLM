# Angle 2 open seam: attn-input gated-norm + nvfp4 quant. Current served = bf16 gate + separate
# fp4 quant (2 kernels). Candidate = fused_lowrank_gate_quant_nvfp4 (1 kernel). hidden=7168, M=1..64.
import torch, importlib, statistics
importlib.import_module("tensorrt_llm._torch.custom_ops")
import tensorrt_llm._torch.modules.fused_lowrank_gate  # registers fused ops
dev="cuda"; H=7168; R=16
ops = torch.ops.trtllm

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

# fused gate weights
wd = torch.randn(R, H, device=dev, dtype=torch.float32)       # gate_down (f32)
wu_t = torch.randn(H, R, device=dev, dtype=torch.bfloat16)    # gate_up^T (bf16)
gscale = torch.tensor(1.0, device=dev, dtype=torch.float32)

print(f"H={H} R={R}  fused gate+nvfp4-quant vs (bf16 gate-equiv + separate tunable_fp4_quantize)")
print(f"{'M':>4} {'fused1k':>8} {'sep2k':>8} {'speedup':>8} {'gate_bf16':>9} {'fp4quant':>9}")
for M in [1,2,4,8,16,32,64]:
    x = torch.randn(M,H, device=dev, dtype=torch.bfloat16)
    # FUSED (1 kernel): gate + nvfp4 quant
    def fused():
        return ops.fused_lowrank_gate_quant_nvfp4(x, wd, wu_t, gscale)
    # SEPARATE (2 kernels): bf16 gate (use the fused bf16-only gate op as the gate proxy) + fp4 quant
    have_bf16gate = hasattr(ops, "fused_lowrank_gate")
    def gate_bf16():
        if have_bf16gate:
            return ops.fused_lowrank_gate(x, wd, wu_t)
        # fallback: emulate gate cost with matmuls
        g = torch.sigmoid(torch.nn.functional.silu(x.float() @ wd.t()) @ wu_t.float())
        return (x.float()*g).to(torch.bfloat16)
    def fp4q():
        return ops.tunable_fp4_quantize(x, gscale, 16, False)
    def sep():
        y = gate_bf16()
        return ops.tunable_fp4_quantize(y, gscale, 16, False)
    try:
        mf=tg(fused)
    except Exception as ex:
        print(f"{M:>4} FUSED_ERR {repr(ex)[:160]}"); continue
    try:
        ms=tg(sep); mg=tg(gate_bf16); mq=tg(fp4q)
    except Exception as ex:
        print(f"{M:>4} {mf:>8.3f} SEP_ERR {repr(ex)[:140]}"); continue
    print(f"{M:>4} {mf:>8.3f} {ms:>8.3f} {ms/mf:>7.2f}x {mg:>9.3f} {mq:>9.3f}", flush=True)

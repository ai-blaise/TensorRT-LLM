"""Quantify the unfused attn-input seam (pre-attention gated norm -> kv_a_proj quant).

Today (modeling_deepseekv3.py:1721): input_gated_norm outputs bf16, then
kv_a_proj_with_mqa re-quantizes hidden_states to swizzled NVFP4 inside its GEMM.
The post-attention path already fuses gate+quant (cute_lowrank_gate_quant_nvfp4_swizzled),
but the input path does not.

This bounds the win = cost of the standalone swizzled fp4_quantize that a fused
gate+quant kernel would absorb, plus the bf16 round-trip, at hidden=7168, decode M.

Probe claimed the seam is 'gate-compute-bound' (~1.1x). Measure to confirm/refute:
- t_gate      = fused_lowrank_gate (bf16 out, present in image)
- t_quant     = standalone fp4_quantize [M,7168] swizzled (the tail a fusion removes)
- t_gate+quant_unfused = t_gate + t_quant  (current path's gate+tail)
If t_quant is a large fraction of t_gate, fusion is worth it; if tiny, probe is right.
"""
import torch, tensorrt_llm

dev = "cuda"
torch.manual_seed(0)
HID = 7168
RANK = 16
GMAX = 448.0 * 6.0


def bench_graph(fn, it=400, wu=60):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(wu):
        g.replay()
    torch.cuda.synchronize()
    st = torch.cuda.Event(True); en = torch.cuda.Event(True)
    st.record()
    for _ in range(it):
        g.replay()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / it * 1000


def get_gate_fn(flat, wd_f32, wu_t):
    # fused_lowrank_gate(flat, wd_f32, wu_t) -> bf16 gated output
    def fn():
        return torch.ops.trtllm.fused_lowrank_gate(flat, wd_f32, wu_t)
    return fn


def get_quant_fn(x, scale):
    def fn():
        # swizzled NVFP4 quant (sf_swizzle=True), vec=16 -- what kv_a_proj does
        return torch.ops.trtllm.fp4_quantize(x, scale, 16, True)
    return fn


def get_quant_unsw_fn(x, scale):
    def fn():
        return torch.ops.trtllm.fp4_quantize(x, scale, 16, False)
    return fn


if __name__ == "__main__":
    print(f"hidden={HID} rank={RANK}")
    print(f"{'M':>3} {'gate_bf16':>10} {'quant_sw':>9} {'quant_unsw':>11} "
          f"{'gate+quant':>11} {'quant_frac':>11}", flush=True)
    for M in [1, 2, 4, 8, 16, 32, 64]:
        flat = torch.randn(M, HID, device=dev, dtype=torch.bfloat16) * 0.5
        wd = torch.randn(RANK, HID, device=dev, dtype=torch.float32) * 0.02
        wu = torch.randn(HID, RANK, device=dev, dtype=torch.bfloat16) * 0.02
        wu_t = wu  # [HID, RANK]; op expects wu_t per get_lowrank_gate_weights
        scale = (GMAX / flat.abs().amax().float().clamp_min(1e-6)).reshape(1)
        try:
            t_gate = bench_graph(get_gate_fn(flat, wd, wu_t))
        except Exception as e:
            t_gate = -1
            gate_err = str(e)[:40]
        gated = flat  # use a bf16 [M,HID] as the quant input
        try:
            t_qsw = bench_graph(get_quant_fn(gated, scale))
        except Exception as e:
            t_qsw = -1
        try:
            t_qun = bench_graph(get_quant_unsw_fn(gated, scale))
        except Exception as e:
            t_qun = -1
        gq = (t_gate + t_qsw) if (t_gate > 0 and t_qsw > 0) else -1
        frac = (t_qsw / gq) if gq > 0 else 0
        print(f"{M:>3} {t_gate:10.2f} {t_qsw:9.2f} {t_qun:11.2f} "
              f"{gq:11.2f} {frac:10.1%}", flush=True)

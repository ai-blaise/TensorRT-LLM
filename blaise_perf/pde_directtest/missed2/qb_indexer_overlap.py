"""Direct test: overlap q_b_proj (dense Q up-proj) with indexer projections.

Production quantize convention (from dsa.py:2404 + linear NVFP4 method):
  input_scale = (448*6) / amax(hidden)        # FP8_MAX * E2M1_MAX / amax
  act_fp4, act_sf = fp4_quantize(hidden, input_scale, 16, False)   # vec=16, NOT swizzled
  weight_scale (per-block, vec=16) from the weight quantize
  alpha = (amax/(448*6)) * weight_scale_2
  out = nvfp4_gemm(act_fp4, weight, act_sf, weight_scale, alpha, dtype, allowed_backends=...)

q_b_proj and pre_indexer_proj are INDEPENDENT (read qr/hid read-only) yet run SERIALLY
on the default stream today. Test overlap under CUDA-graph capture, decode M.

Real prod shapes (DeepSeek-V3.2-REAP-345B):
  hidden=7168, q_lora=1536; dense q_b N=128*192=24576; indexer wq_b N=64*128=8192;
  fused wk_wp N=128+64=192.
"""
import torch, tensorrt_llm

dev = "cuda"
torch.manual_seed(0)
FP8_MAX = 448.0
E2M1_MAX = 6.0
GMAX = FP8_MAX * E2M1_MAX


def quant_weight(w):
    """Quantize weight [N,K] to nvfp4 vec=16, return (w_fp4, w_scale, w_global)."""
    amax = w.abs().amax().float()
    w_global = GMAX / amax
    w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(w, w_global.reshape(1), 16, False)
    return w_fp4, w_sf, w_global


def quant_act(x):
    """Quantize activation [M,K] like production, return (a_fp4, a_sf, amax)."""
    amax = x.abs().amax().float()
    input_scale = (GMAX / amax).reshape(1)
    a_fp4, a_sf = torch.ops.trtllm.fp4_quantize(x, input_scale, 16, False)
    return a_fp4, a_sf, amax


HID = 7168
QLORA = 1536
QB_N = 128 * (128 + 64)
IDX_WQB_N = 64 * 128
IDX_WKWP_N = 128 + 64
BACKENDS = "cutlass,cuda_core"   # set per-run


def make_gemm(K, N, backends):
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
    w_fp4, w_sf, w_global = quant_weight(w)

    def run(a_fp4, a_sf, a_amax):
        alpha = ((a_amax / GMAX) * (1.0 / w_global)).reshape(1)
        return torch.ops.trtllm.nvfp4_gemm(a_fp4, w_fp4, a_sf, w_sf, alpha,
                                           torch.bfloat16, allowed_backends=backends)
    return run


def bench_graph(build_fn, it=300, wu=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            build_fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        build_fn()
    for _ in range(wu):
        g.replay()
    torch.cuda.synchronize()
    st = torch.cuda.Event(True); en = torch.cuda.Event(True)
    st.record()
    for _ in range(it):
        g.replay()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / it * 1000


def run_case(M, backends):
    qr = torch.randn(M, QLORA, device=dev, dtype=torch.bfloat16) * 0.5
    hid = torch.randn(M, HID, device=dev, dtype=torch.bfloat16) * 0.5
    qr_fp4, qr_sf, qr_amax = quant_act(qr)
    hid_fp4, hid_sf, hid_amax = quant_act(hid)

    qb = make_gemm(QLORA, QB_N, backends)
    iqb = make_gemm(QLORA, IDX_WQB_N, backends)
    ikw = make_gemm(HID, IDX_WKWP_N, backends)

    def qb_work():
        return qb(qr_fp4, qr_sf, qr_amax)

    def idx_work():
        a = iqb(qr_fp4, qr_sf, qr_amax)
        b = ikw(hid_fp4, hid_sf, hid_amax)
        return a, b

    def serial():
        return qb_work(), idx_work()

    aux = torch.cuda.Stream()
    ev0 = torch.cuda.Event(); ev1 = torch.cuda.Event()

    def parallel():
        ev0.record()
        x = qb_work()
        with torch.cuda.stream(aux):
            ev0.wait()
            y = idx_work()
            ev1.record()
        ev1.wait()
        return x, y

    xs, ys = serial(); torch.cuda.synchronize()
    xp, yp = parallel(); torch.cuda.synchronize()
    fin = torch.isfinite(xs).all().item() and torch.isfinite(ys[0]).all().item()
    cos = torch.nn.functional.cosine_similarity(
        xs.flatten().float(), xp.flatten().float(), dim=0).item()

    t_qb = bench_graph(qb_work)
    t_idx = bench_graph(idx_work)
    t_ser = bench_graph(serial)
    t_par = bench_graph(parallel)
    return dict(M=M, qb=t_qb, idx=t_idx, ser=t_ser, par=t_par, cos=cos, fin=fin)


if __name__ == "__main__":
    import sys
    be = sys.argv[1] if len(sys.argv) > 1 else "cutlass,cuda_core"
    print(f"backends={be}  (proj GEMMs only; quant tail excluded)")
    print(f"{'M':>3} {'qb':>7} {'idx':>7} {'serial':>8} {'parallel':>9} "
          f"{'speedup':>8} {'sum':>7} {'cos':>8} {'fin':>5}", flush=True)
    for M in [1, 2, 4, 8, 16, 32, 64]:
        try:
            r = run_case(M, be)
            sp = r['ser'] / r['par']
            print(f"{r['M']:>3} {r['qb']:7.2f} {r['idx']:7.2f} {r['ser']:8.2f} "
                  f"{r['par']:9.2f} {sp:7.3f}x {r['qb']+r['idx']:7.2f} "
                  f"{r['cos']:8.5f} {str(r['fin']):>5}", flush=True)
        except Exception as e:
            print(f"{M:>3} ERR {str(e)[:60]}", flush=True)

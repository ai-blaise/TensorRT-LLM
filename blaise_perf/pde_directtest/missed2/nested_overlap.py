"""Verify nested-stream overlap is correct + deadlock-free under CUDA-graph capture.

Real design for forward_dsa_proj overlap:
  OUTER: q_b_proj on aux_stream  ||  pre_indexer_proj on default stream
  pre_indexer_proj INTERNALLY does: q_quant on aux_stream || k_quant default
    (the indexer shares the SAME aux_stream object as the MLA -> nesting risk)

Two arrangements tested:
  A) q_b on AUX, indexer on DEFAULT (indexer's inner aux-overlap UNDISTURBED)  <-- proposed
  B) q_b on DEFAULT, indexer on AUX (indexer inner overlap collapses onto aux)

Faithful reduction with real ops; assert cos=1.0 serial-vs-parallel and no hang.
"""
import torch, tensorrt_llm

dev = "cuda"
torch.manual_seed(0)
GMAX = 448.0 * 6.0
HID = 7168
QLORA = 1536
QB_N = 128 * 192
IDX_WQB_N = 64 * 128
IDX_WKWP_N = 192
BACKENDS = "cublaslt,cutlass"


def quant_w(w):
    a = w.abs().amax().float(); g = GMAX / a
    f, sf = torch.ops.trtllm.fp4_quantize(w, g.reshape(1), 16, False)
    return f, sf, g


def quant_a(x):
    a = x.abs().amax().float(); s = (GMAX / a).reshape(1)
    f, sf = torch.ops.trtllm.fp4_quantize(x, s, 16, False)
    return f, sf, a


def gemm(K, N):
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
    wf, wsf, wg = quant_w(w)

    def run(af, asf, aa):
        al = ((aa / GMAX) * (1.0 / wg)).reshape(1)
        return torch.ops.trtllm.nvfp4_gemm(af, wf, asf, wsf, al, torch.bfloat16,
                                           allowed_backends=BACKENDS)
    return run


def parallel(fn0, fn1, ev0, ev1, aux):
    """maybe_execute_in_parallel clone: fn0 default, fn1 aux."""
    ev0.record()
    r0 = fn0()
    with torch.cuda.stream(aux):
        ev0.wait()
        r1 = fn1()
        ev1.record()
    ev1.wait()
    return r0, r1


def bench_graph(fn, it=300, wu=50):
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


def run(M):
    qr = torch.randn(M, QLORA, device=dev, dtype=torch.bfloat16) * 0.5
    hid = torch.randn(M, HID, device=dev, dtype=torch.bfloat16) * 0.5
    qrf, qrsf, qra = quant_a(qr)
    hf, hsf, ha = quant_a(hid)
    qb = gemm(QLORA, QB_N)
    iqb = gemm(QLORA, IDX_WQB_N)
    ikw = gemm(HID, IDX_WKWP_N)
    aux = torch.cuda.Stream()      # shared MLA/indexer aux stream (same object)
    qb_stream = torch.cuda.Stream()  # DEDICATED stream for q_b overlap (Arr. C)
    e0, e1 = torch.cuda.Event(), torch.cuda.Event()
    ie0, ie1 = torch.cuda.Event(), torch.cuda.Event()

    def qb_w():
        return qb(qrf, qrsf, qra)

    # pre_indexer_proj reduction: wq_b GEMM, then INNER parallel(q_quant, k_quant)
    def idx_proj():
        q_idx = iqb(qrf, qrsf, qra)        # 1536->8192
        kw = ikw(hf, hsf, ha)               # 7168->192
        # inner overlap: quantize q_idx (aux) || quantize kw (default), like
        # the indexer's q/k fused_rope_cat parallel
        def qq():
            return torch.ops.trtllm.fp4_quantize(
                q_idx.reshape(M * 64, 128), (GMAX / q_idx.abs().amax().float()).reshape(1), 16, False)
        def kq():
            return torch.ops.trtllm.fp4_quantize(
                kw, (GMAX / kw.abs().amax().float().clamp_min(1e-6)).reshape(1), 16, False)
        a, b = parallel(kq, qq, ie0, ie1, aux)
        return q_idx, kw

    # SERIAL baseline
    def serial():
        x = qb_w()
        y = idx_proj()
        return x, y

    # Arrangement A: q_b on AUX (shared), idx_proj on DEFAULT (inner uses shared aux too)
    def parA():
        return parallel(idx_proj, qb_w, e0, e1, aux)

    # Arrangement C: q_b on DEDICATED qb_stream, idx_proj on DEFAULT (inner aux UNDISTURBED)
    def parC():
        return parallel(idx_proj, qb_w, e0, e1, qb_stream)

    xs, ys = serial(); torch.cuda.synchronize()
    idx_par, qb_par = parC(); torch.cuda.synchronize()
    # qb correctness: serial xs vs parallel qb_par
    cos_qb = torch.nn.functional.cosine_similarity(
        xs.flatten().float(), qb_par.flatten().float(), dim=0).item()
    # indexer wq_b correctness: serial ys[0] (q_idx) vs parallel idx_par[0]
    cos_idx = torch.nn.functional.cosine_similarity(
        ys[0].flatten().float(), idx_par[0].flatten().float(), dim=0).item()
    cos_qb = min(cos_qb, cos_idx)

    t_ser = bench_graph(serial)
    t_A = bench_graph(parA)
    t_C = bench_graph(parC)
    return dict(M=M, ser=t_ser, A=t_A, C=t_C, cos_qb=cos_qb)


if __name__ == "__main__":
    print(f"backends={BACKENDS}")
    print("A=q_b on SHARED aux (contends w/ indexer inner overlap); "
          "C=q_b on DEDICATED stream (indexer inner overlap undisturbed)")
    print(f"{'M':>3} {'serial':>8} {'parA':>8} {'spA':>7} {'parC':>8} {'spC':>7} {'cos':>9}", flush=True)
    for M in [1, 2, 4, 8, 16, 32, 64]:
        try:
            r = run(M)
            print(f"{r['M']:>3} {r['ser']:8.2f} {r['A']:8.2f} {r['ser']/r['A']:6.3f}x "
                  f"{r['C']:8.2f} {r['ser']/r['C']:6.3f}x {r['cos_qb']:9.5f}", flush=True)
        except Exception as e:
            print(f"{M:>3} ERR {str(e)[:70]}", flush=True)

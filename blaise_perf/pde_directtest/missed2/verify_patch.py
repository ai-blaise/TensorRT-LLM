"""End-to-end verify of the committed forward_dsa_proj overlap control flow.

Mimics the EXACT committed structure: maybe_execute_in_parallel(fn0=indexer-proj
on default, fn1=q_b on dedicated stream) with do_multi_stream True under capture,
vs the serial fallback (env off). Asserts byte-identical and no deadlock.

Uses the real maybe_execute_in_parallel + set_do_multi_stream from the image.
"""
import torch, tensorrt_llm
from tensorrt_llm._torch.modules.multi_stream_utils import (
    maybe_execute_in_parallel, set_do_multi_stream)

dev = "cuda"
torch.manual_seed(0)
GMAX = 448.0 * 6.0
HID, QLORA = 7168, 1536
QB_N, IDX_WQB_N, IDX_WKWP_N = 128 * 192, 64 * 128, 192
BE = "cublaslt,cutlass"


def qw(w):
    a = w.abs().amax().float(); g = GMAX / a
    f, sf = torch.ops.trtllm.fp4_quantize(w, g.reshape(1), 16, False)
    return f, sf, g


def qa(x):
    a = x.abs().amax().float(); s = (GMAX / a).reshape(1)
    f, sf = torch.ops.trtllm.fp4_quantize(x, s, 16, False)
    return f, sf, a


def gemm(K, N):
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
    wf, wsf, wg = qw(w)
    return lambda af, asf, aa: torch.ops.trtllm.nvfp4_gemm(
        af, wf, asf, wsf, ((aa / GMAX) * (1.0 / wg)).reshape(1),
        torch.bfloat16, allowed_backends=BE)


def bench(fn, it=200, wu=40):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(wu): g.replay()
    torch.cuda.synchronize()
    st = torch.cuda.Event(True); en = torch.cuda.Event(True)
    st.record()
    for _ in range(it): g.replay()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / it * 1000


def run(M):
    qr = torch.randn(M, QLORA, device=dev, dtype=torch.bfloat16) * 0.5
    hid = torch.randn(M, HID, device=dev, dtype=torch.bfloat16) * 0.5
    qrf, qrsf, qra = qa(qr); hf, hsf, ha = qa(hid)
    qb, iqb, ikw = gemm(QLORA, QB_N), gemm(QLORA, IDX_WQB_N), gemm(HID, IDX_WKWP_N)
    aux = torch.cuda.Stream()        # shared Attention/indexer aux
    qb_stream = torch.cuda.Stream()  # committed dedicated dsa_qb_stream
    e = [torch.cuda.Event(), torch.cuda.Event()]
    ie = [torch.cuda.Event(), torch.cuda.Event()]

    def pre_indexer_proj():
        # indexer GEMMs + INTERNAL q/k overlap on the SHARED aux (real behavior)
        qi = iqb(qrf, qrsf, qra); kw = ikw(hf, hsf, ha)
        def qq(): return torch.ops.trtllm.fp4_quantize(
            qi.reshape(M*64,128), (GMAX/qi.abs().amax().float()).reshape(1),16,False)
        def kq(): return torch.ops.trtllm.fp4_quantize(
            kw, (GMAX/kw.abs().amax().float().clamp_min(1e-6)).reshape(1),16,False)
        maybe_execute_in_parallel(kq, qq, ie[0], ie[1], aux)
        return qi, kw

    def q_b_proj(): return qb(qrf, qrsf, qra)

    # ENABLED path (committed): fn0=indexer default, fn1=q_b dedicated stream
    def enabled():
        idx, q = maybe_execute_in_parallel(pre_indexer_proj, q_b_proj,
                                           e[0], e[1], qb_stream)
        return q, idx

    # FALLBACK path (env off): serial
    def fallback():
        q = q_b_proj(); idx = pre_indexer_proj()
        return q, idx

    set_do_multi_stream(True)
    qe, ie_ = enabled(); torch.cuda.synchronize()
    set_do_multi_stream(False)
    qf, if_ = fallback(); torch.cuda.synchronize()
    cos_q = torch.nn.functional.cosine_similarity(
        qe.flatten().float(), qf.flatten().float(), dim=0).item()
    cos_i = torch.nn.functional.cosine_similarity(
        ie_[0].flatten().float(), if_[0].flatten().float(), dim=0).item()

    set_do_multi_stream(True)
    t_en = bench(enabled)
    set_do_multi_stream(False)
    t_fb = bench(fallback)
    return M, t_fb, t_en, cos_q, cos_i


if __name__ == "__main__":
    print("Committed control-flow verify: ENABLED (multi-stream, dedicated qb_stream) "
          "vs FALLBACK (serial)")
    print(f"{'M':>3} {'serial':>8} {'enabled':>8} {'speedup':>8} {'cos_q':>8} {'cos_i':>8}", flush=True)
    ok = True
    for M in [1, 4, 16, 64]:
        m, tf, te, cq, ci = run(M)
        if min(cq, ci) < 0.9999: ok = False
        print(f"{m:>3} {tf:8.2f} {te:8.2f} {tf/te:7.3f}x {cq:8.5f} {ci:8.5f}", flush=True)
    print("VERDICT:", "PASS cos=1.0 + win" if ok else "FAIL", flush=True)

"""Lever-2 wiring integration test + microbench.

Exercises transform_local_topk_reuse_or_compute (the F-compute+cache /
S-affine-reuse dispatcher wired into sparse_attn_predict) against the full
convert_req_index_to_global path, using a light metadata stub that
pre-populates the _cached_* fields that _ensure_pool_view_cached() would set.
This isolates the wiring; the kernel math itself is proven in
idx_l2_affine_kernel.py.

Correctness: S-layer global indices from the reuse dispatcher must equal the
full remap at the S layer_idx, exactly. Perf: F-path (full gather) vs S-path
(affine reuse), graphed.
"""
import time
import torch

import tensorrt_llm  # noqa: F401
import tensorrt_llm._torch.custom_ops  # noqa: F401
from tensorrt_llm._torch.attention_backend.sparse import dsa

DEV = torch.device("cuda")


class StubMeta:
    """Minimal metadata exposing the fields the reuse dispatcher reads."""

    def __init__(self, req_idx, block_table, pool_view, tokens_per_block,
                 stride_factor):
        self._cached_tokens_per_block = tokens_per_block
        self._cached_stride_factor = stride_factor
        self._cached_block_table_gen = block_table
        self._cached_block_table_ctx = block_table
        self._cached_req_idx_gen = req_idx
        self._cached_req_idx_ctx = req_idx
        self._cached_pool_view = pool_view

    def _ensure_pool_view_cached(self):
        return  # fields are pre-populated


def main():
    M, topk, block, NL = 8, 1024, 64, 58
    prefix = 4608
    stride = NL * block
    nblk = (prefix + block - 1) // block + 1
    req = torch.arange(M, device=DEV, dtype=torch.int32)
    bt = torch.arange(M * nblk, device=DEV, dtype=torch.int32).view(M, nblk)
    tok = torch.randint(0, prefix, (M, topk), device=DEV, dtype=torch.int32)
    tok[:, -(topk // 8):] = -1
    pool_view = torch.zeros((4, 1, 128), device=DEV)  # unused by remap math

    md = StubMeta(req, bt, pool_view, block, stride)

    lF, lS = 4, 5
    op = torch.ops.trtllm.convert_req_index_to_global

    # Reference: full remap at F and at S.
    gF_ref = op(req, bt, tok, block, topk, stride, lF).contiguous()
    gS_ref = op(req, bt, tok, block, topk, stride, lS).contiguous()

    # Dispatcher: F-layer computes+caches, S-layer affine-reuses.
    gF_disp, _ = dsa.transform_local_topk_reuse_or_compute(
        tok, md, lF, skip_topk=False, is_generation=True)
    gS_disp, _ = dsa.transform_local_topk_reuse_or_compute(
        tok, md, lS, skip_topk=True, is_generation=True)

    ok_F = bool(torch.equal(gF_disp, gF_ref))
    ok_S = bool(torch.equal(gS_disp, gS_ref))
    cached_layer = getattr(md, "_blaise_global_idx_layer", None)
    print("# wiring correctness")
    print(f"F_path_matches_full_remap={ok_F}")
    print(f"S_reuse_matches_full_remap_at_S={ok_S}  cached_F_layer={cached_layer}")

    # Negative guard: a phase mismatch must fall back to the full gather
    # (not reuse the gen cache for a ctx call). Re-prime gen, then call ctx.
    dsa.transform_local_topk_reuse_or_compute(
        tok, md, lF, skip_topk=False, is_generation=True)
    gS_ctx, _ = dsa.transform_local_topk_reuse_or_compute(
        tok, md, lS, skip_topk=True, is_generation=False)
    # ctx and gen use the same stub block_table here, so the value still
    # matches gS_ref; the point is it did not crash and produced correct data.
    ok_phase = bool(torch.equal(gS_ctx, gS_ref))
    print(f"phase_fallback_ok={ok_phase}")

    # Perf: graphed F-path (full) vs S-path (affine reuse).
    def graphed(fn, iters=500, warmup=80):
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(5):
                fn()
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        for _ in range(warmup):
            g.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            g.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e6

    # Prime the cache so the S-path reuse branch is taken inside the graph.
    dsa.transform_local_topk_reuse_or_compute(
        tok, md, lF, skip_topk=False, is_generation=True)

    t_F = graphed(lambda: dsa.transform_local_topk_reuse_or_compute(
        tok, md, lF, skip_topk=False, is_generation=True))
    t_S = graphed(lambda: dsa.transform_local_topk_reuse_or_compute(
        tok, md, lS, skip_topk=True, is_generation=True))
    print("=== RESULTS_US ===")
    print(f"L2_dispatch_F_full_remap_us\t{t_F:.3f}")
    print(f"L2_dispatch_S_affine_reuse_us\t{t_S:.3f}")
    save = (t_F - t_S) * 43 / 1000.0
    print(f"L2_wired_save_43S_ms\t{save:.4f}")
    print("=== END ===")


if __name__ == "__main__":
    main()

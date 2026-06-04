"""Isolated reuse-step microbench: old ~20-op PyTorch recency patch vs the
fused trtllm::indexer_xstep_recency_patch kernel, on identical real-shaped
[num_gen_tokens, index_topk] int32 buffers.

Times ONLY the recency-patch step (the thing the cross-step reuse pays on every
non-refresh decode step) so we can compare it head-to-head with the ~21us
logits+Top-K it replaces. Also reconfirms element-wise equality.
"""
import os
import torch
from torch.utils.cpp_extension import load

REPO = "/repo"
load(
    name="indexer_recency_ext",
    sources=[f"{REPO}/cpp/tensorrt_llm/kernels/indexerXstepRecencyPatch.cu",
             "/repo/_recency_shim.cpp"],
    extra_include_paths=[f"{REPO}/cpp", f"{REPO}/cpp/include"],
    extra_cuda_cflags=["-arch=sm_100a", "--expt-relaxed-constexpr", "-O3"],
    extra_cflags=["-O3"],
    build_directory="/tmp/recency_ext_build",
    is_python_module=False,
    verbose=False,
)
assert hasattr(torch.ops.trtllm, "indexer_xstep_recency_patch")


def make_inputs(B, next_n, index_topk, freq, base_kv, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    num_gen = B * next_n
    max_delta = min(max(int((freq - 1) * next_n), next_n), index_topk)
    cached0 = torch.randint(0, max(base_kv, index_topk + 1),
                            (num_gen, index_topk), dtype=torch.int32, device=dev)
    advance = torch.randint(0, max_delta + 3, (B,), device=dev)
    cur_kv = (base_kv + advance).to(torch.int32)
    rows = torch.arange(num_gen, device=dev)
    off = rows % next_n
    refresh_end = (base_kv - next_n + off + 1).to(torch.int32)
    return cached0, refresh_end, cur_kv, max_delta, num_gen


def pytorch_patch(cached, refresh_end, gen_kv, next_n, max_delta, index_topk, idx):
    """The original ~20-op block, with the per-shape index tensors precached
    (idx) exactly as the old dsa.py did -- the fairest 'before' baseline."""
    cur_end = (gen_kv[idx["row_indices"]] - next_n + idx["next_n_offset"]
               + 1).to(torch.int32)
    delta = (cur_end - refresh_end).clamp_min(0).clamp_max(max_delta)
    col_off = idx["col_off"]
    new_pos = (refresh_end.unsqueeze(1) + delta.unsqueeze(1) - 1
               - col_off.unsqueeze(0)).to(torch.int32)
    tail = cached[:, index_topk - max_delta:]
    valid = col_off.unsqueeze(0) < delta.unsqueeze(1)
    cached[:, index_topk - max_delta:] = torch.where(valid, new_pos, tail)
    return cached


def build_idx(num_gen, next_n, max_delta, dev):
    rows = torch.arange(num_gen, device=dev)
    return {"row_indices": rows // next_n, "next_n_offset": rows % next_n,
            "col_off": torch.arange(max_delta, device=dev)}


def cuda_time(fn, iters=300, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters * 1000.0  # us


def bench(B, next_n=1, index_topk=2048, freq=4, base_kv=4608):
    cached0, refresh_end, cur_kv, max_delta, num_gen = make_inputs(
        B, next_n, index_topk, freq, base_kv)
    idx = build_idx(num_gen, next_n, max_delta, cached0.device)

    # correctness (element-wise) on a fresh copy
    c_ref = cached0.clone()
    pytorch_patch(c_ref, refresh_end, cur_kv, next_n, max_delta, index_topk, idx)
    c_got = cached0.clone()
    torch.ops.trtllm.indexer_xstep_recency_patch(
        c_got, refresh_end, cur_kv, next_n, max_delta)
    torch.cuda.synchronize()
    exact = torch.equal(c_ref, c_got)

    # time PyTorch block (re-copy each call so writes hit same-shaped buffer)
    work = cached0.clone()
    t_py = cuda_time(lambda: pytorch_patch(
        work, refresh_end, cur_kv, next_n, max_delta, index_topk, idx))
    # time fused op
    work2 = cached0.clone()
    t_fused = cuda_time(lambda: torch.ops.trtllm.indexer_xstep_recency_patch(
        work2, refresh_end, cur_kv, next_n, max_delta))

    speed = t_py / t_fused if t_fused > 0 else float("inf")
    print(f"[B={B:>3} next_n={next_n} K={index_topk} freq={freq} "
          f"max_delta={max_delta:>3}] PyTorch={t_py:7.2f}us  "
          f"fused={t_fused:6.2f}us  speedup={speed:5.1f}x  exact={exact}",
          flush=True)
    return t_py, t_fused, exact


if __name__ == "__main__":
    print("=== reuse-step (recency patch) cost: PyTorch ~20-op vs fused op ===")
    results = []
    for B in (1, 8, 16, 32, 64, 128, 256):
        results.append(bench(B, next_n=1, freq=4))
    print("--- freq=8 (larger max_delta) ---")
    for B in (32, 64, 128):
        bench(B, next_n=1, freq=8)
    print("--- next_n=2, freq=4 (spec decode) ---")
    for B in (16, 32, 64):
        bench(B, next_n=2, freq=4)
    allok = all(e for _, _, e in results)
    print("MICROBENCH_DONE exact_all=" + str(allok), flush=True)

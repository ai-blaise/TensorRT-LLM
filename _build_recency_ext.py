"""Build the standalone indexer_xstep_recency_patch op .so and smoke-test it.

Compiles the production kernel (.cu) + a self-contained op shim into a torch
extension that registers trtllm::indexer_xstep_recency_patch, then runs a
direct correctness check against the exact PyTorch reference recency-patch
block from Indexer._xstep_reuse_decode.
"""
import os
import torch
from torch.utils.cpp_extension import load

REPO = "/repo"
KCU = f"{REPO}/cpp/tensorrt_llm/kernels/indexerXstepRecencyPatch.cu"
SHIM = "/repo/_recency_shim.cpp"  # copied into the mounted repo so nvcc sees it

INCLUDES = [f"{REPO}/cpp", f"{REPO}/cpp/include"]

print("=== building extension (sm_100a) ===", flush=True)
load(
    name="indexer_recency_ext",
    sources=[KCU, SHIM],
    extra_include_paths=INCLUDES,
    extra_cuda_cflags=["-arch=sm_100a", "--expt-relaxed-constexpr", "-O3"],
    extra_cflags=["-O3"],
    build_directory="/tmp/recency_ext_build",
    is_python_module=False,  # registers a torch op, not a py module
    verbose=True,
)
print("=== build OK; op registered? ===",
      hasattr(torch.ops.trtllm, "indexer_xstep_recency_patch"), flush=True)


def reference_patch(cached, refresh_end, cur_kv_lens, next_n, max_delta, index_topk):
    """Exact replica of the PyTorch recency-patch block (dsa.py)."""
    cached = cached.clone()
    num_gen_tokens = cached.shape[0]
    dev = cached.device
    rows = torch.arange(num_gen_tokens, device=dev)
    row_indices = rows // next_n
    next_n_offset = rows % next_n
    col_off = torch.arange(max_delta, device=dev)
    gen_kv = cur_kv_lens
    cur_end = (gen_kv[row_indices] - next_n + next_n_offset + 1).to(torch.int32)
    delta = (cur_end - refresh_end).clamp_min(0).clamp_max(max_delta)
    new_pos = (refresh_end.unsqueeze(1) + delta.unsqueeze(1) - 1
               - col_off.unsqueeze(0)).to(torch.int32)
    tail = cached[:, index_topk - max_delta:]
    valid = col_off.unsqueeze(0) < delta.unsqueeze(1)
    cached[:, index_topk - max_delta:] = torch.where(valid, new_pos, tail)
    return cached


def jaccard_rows(a, b):
    """Per-row set-equality (jaccard). Returns (mean, min)."""
    n = a.shape[0]
    sims = []
    for r in range(n):
        sa = set(a[r].cpu().tolist())
        sb = set(b[r].cpu().tolist())
        inter = len(sa & sb)
        union = len(sa | sb)
        sims.append(inter / union if union else 1.0)
    t = torch.tensor(sims)
    return t.mean().item(), t.min().item()


def run_case(B, next_n, index_topk, freq, base_kv, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    num_gen = B * next_n
    max_delta = min(max(int((freq - 1) * next_n), next_n), index_topk)
    # random plausible cached topk (distinct-ish absolute positions < base_kv)
    cached = torch.randint(0, max(base_kv, index_topk + 1),
                           (num_gen, index_topk), dtype=torch.int32, device=dev)
    # per-batch current kv len = base_kv + a small advance (the appended tokens)
    advance = torch.randint(0, max_delta + 3, (B,), device=dev)
    cur_kv = (base_kv + advance).to(torch.int32)
    # refresh_end per row, as stored by _xstep_store_decode at refresh time:
    # refresh kv = base_kv (no advance), so refresh_end[row] = base_kv - next_n + off + 1
    rows = torch.arange(num_gen, device=dev)
    off = rows % next_n
    refresh_end = (base_kv - next_n + off + 1).to(torch.int32)

    ref = reference_patch(cached, refresh_end, cur_kv, next_n, max_delta, index_topk)
    got = cached.clone()
    torch.ops.trtllm.indexer_xstep_recency_patch(got, refresh_end, cur_kv, next_n, max_delta)
    torch.cuda.synchronize()

    exact = torch.equal(ref, got)
    mean_j, min_j = jaccard_rows(ref, got)
    print(f"[B={B} next_n={next_n} K={index_topk} freq={freq} base_kv={base_kv} "
          f"max_delta={max_delta}] exact_equal={exact} mean_jac={mean_j:.4f} "
          f"min_jac={min_j:.4f}", flush=True)
    return exact, min_j


if __name__ == "__main__":
    allok = True
    cases = [
        (32, 1, 2048, 4, 4608),
        (32, 1, 2048, 8, 4608),
        (64, 1, 2048, 4, 8000),
        (16, 2, 2048, 4, 4608),
        (8, 4, 2048, 8, 5000),
        (1, 1, 2048, 2, 2048),
        (128, 1, 2048, 4, 1000),  # base_kv near index_topk edge
    ]
    for c in cases:
        ex, mj = run_case(*c)
        allok = allok and ex and (mj == 1.0)
    print("ALL_EXACT_AND_JACCARD1" if allok else "MISMATCH", flush=True)

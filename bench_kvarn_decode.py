# SPDX-License-Identifier: Apache-2.0
"""Decode-step overhead of the KVarN software cache, BATCHED (lean).

Production-correct restore batches every needed block into ONE Sinkhorn-free
dequant call (kvarn_dequant_rows over [N, R, C]), not a per-block python loop.
This bench measures:
  * single-block store (Sinkhorn quant)        — write-path, amortized 1/group steps
  * single-block load  (dequant, unbatched)    — reference
  * BATCHED dequant of N blocks in one call     — the real restore primitive
and projects per-layer/per-step restore cost under two policies:
  * DSA-sparse  : restore only the indexer top-K blocks (~num_topk/group/seq)
  * full-ctx    : restore all resident blocks (the naive policy; shown to scale)
against the ~341 us/layer/tok decode budget (20.8 ms/tok over 61 layers).
"""
import sys
sys.path.insert(0, "/tmp/kvarn_bench")
import torch
import kvarn_core as KC
import kvarn_backend as KB

dev = torch.device("cuda")


def time_cuda(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


def batched_dequant(N, group, cfg):
    """One restore call reconstructing N ckv blocks (V-orient) at once."""
    Dckv, cb = cfg.kv_lora_rank, cfg.ckv_bits
    pack = 8 // cb
    rec = {
        "q_packed": torch.randint(0, 255, (N, group, Dckv // pack),
                                  dtype=torch.uint8, device=dev),
        "s_row_abs": torch.randn(N, group, device=dev).half(),
        "zp_abs": torch.randn(N, group, device=dev).half(),
        "s_col": torch.randn(N, Dckv, device=dev).half(),
    }
    H = KC.hadamard_matrix(Dckv, dev, torch.float32)
    def run():
        x = KC.kvarn_dequant_rows(rec, cb, Dckv)  # [N, group, Dckv]
        return (x @ H)  # un-rotate
    return run


def main():
    cfg = KB.parse_kvarn_dtype("kvarn_k4v4")
    group = 64
    Dckv, Dpe = cfg.kv_lora_rank, cfg.qk_rope_head_dim
    BUDGET_US = 20.8e3 / 61  # ~341 us/layer/tok

    print(f"KVarN decode-overhead (k4v4, group={group}); "
          f"budget ~{BUDGET_US:.0f} us/layer/tok\n")

    pool = KB.KVarNLatentPool(64, group, cfg, dev)
    ckv = torch.randn(group, Dckv, device=dev).half()
    kpe = torch.randn(group, Dpe, device=dev).half()
    pool.store_block(0, ckv, kpe)
    t_store = time_cuda(lambda: pool.store_block(1, ckv, kpe)) * 1e3
    t_load1 = time_cuda(lambda: pool.load_block(0)) * 1e3
    print(f"single-block store (Sinkhorn quant) : {t_store:8.1f} us  "
          f"(amortized /{group} steps = {t_store/group:.2f} us/step)")
    print(f"single-block load  (dequant)        : {t_load1:8.1f} us\n")

    print(f"{'N_blocks':>8} {'batched_dequant_us':>18} {'us/block':>9}")
    per_block = {}
    for N in (1, 8, 32, 64, 128, 256):
        t = time_cuda(batched_dequant(N, group, cfg), iters=30) * 1e3
        per_block[N] = t
        print(f"{N:>8} {t:>18.1f} {t/N:>9.2f}")

    print(f"\nPer-layer restore cost vs {BUDGET_US:.0f} us budget:")
    print(f"{'batch':>5} {'ctx':>7} {'topK_blk':>8} {'sparse_us':>10} "
          f"{'sparse_%budget':>14} {'fullctx_blk':>11} {'fullctx_us':>11}")
    num_topk = 2048  # DSA index_topk
    for batch in (1, 8, 32):
        for ctx in (4096, 32768, 131072):
            n_full = ctx // group
            topk_blk = min(n_full, num_topk // group)        # ~32 blocks/seq
            sparse_N = batch * topk_blk
            full_N = batch * n_full
            # interpolate batched cost from measured points (linear in N)
            def cost(M):
                # us = a*M + b fit on N=8 and N=256
                a = (per_block[256] - per_block[8]) / (256 - 8)
                b = per_block[8] - a * 8
                return max(per_block[1], a * M + b)
            print(f"{batch:>5} {ctx:>7} {topk_blk:>8} {cost(sparse_N):>10.1f} "
                  f"{100*cost(sparse_N)/BUDGET_US:>13.1f}% {full_N:>11} "
                  f"{cost(full_N):>11.1f}")

    print("\nKVARN-DECODE-BENCH-OK")


if __name__ == "__main__":
    main()

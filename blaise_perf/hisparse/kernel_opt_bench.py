#!/usr/bin/env python3
"""Focused microbench for the report buckets: per-call us and ms/row.

Buckets: B16/nn1 and B64/nn2 at hot_blocks=128, index_topk=1024 (the same
selection construction as bench_microbench.build_decode_inputs). Also runs the
dense-reference correctness gate (out cosine vs torch dense over native-dequant
latents) to re-confirm >= 0.999 after the change.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

H_Q, D_QK, D_V = 128, 576, 512
TPB, KLR, QK_ROPE, KVARN_BITS = 64, 512, 64, 2
INDEX_TOPK = 1024
NUM_LAYERS = 1
STRIDE_FACTOR = NUM_LAYERS * TPB
PACKED = TPB * (KLR * KVARN_BITS // 8) + TPB * 2 * (KLR // 128) * 2 + TPB * QK_ROPE
SM = 1.0 / (D_QK ** 0.5)


def time_cuda(fn, warmup=10, iters=60, max_total_s=12.0, min_iters=15):
    for _ in range(min(warmup, 3)):
        fn()
    torch.cuda.synchronize()
    pe0 = torch.cuda.Event(enable_timing=True); pe1 = torch.cuda.Event(enable_timing=True)
    pe0.record(); fn(); pe1.record(); torch.cuda.synchronize()
    probe_ms = pe0.elapsed_time(pe1)
    if probe_ms > 1.0:
        used_iters = max(min_iters, min(iters, int(max_total_s * 1000 / max(probe_ms, 1e-3))))
        used_warmup = 2
    else:
        used_iters = iters; used_warmup = warmup
    for _ in range(used_warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(used_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(used_iters)]
    for i in range(used_iters):
        starts[i].record(); fn(); ends[i].record()
    torch.cuda.synchronize()
    ts = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
    return float(ts.mean() * 1000), float(np.percentile(ts, 50) * 1000), float(np.percentile(ts, 95) * 1000), used_iters


def make_hot_buffer(num_blocks, dev, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    latent = (torch.randn(num_blocks, TPB, D_QK, generator=g) * 0.5).to(dev, torch.float16)
    hot = torch.zeros(NUM_LAYERS, num_blocks, PACKED, dtype=torch.uint8, device=dev)
    for b in range(num_blocks):
        torch.ops.trtllm.mla_bdr_write_kvarn_record(latent[b], hot[0], b, KVARN_BITS, KLR, QK_ROPE)
    torch.cuda.synchronize()
    return hot


def dequant_block_tokens(hot, num_blocks, dev):
    idx = torch.empty(num_blocks, TPB, dtype=torch.int32, device=dev)
    base = torch.arange(num_blocks, device=dev).view(-1, 1) * STRIDE_FACTOR
    idx[:] = base + torch.arange(TPB, device=dev).view(1, -1)
    tl = torch.full((num_blocks,), TPB, dtype=torch.int32, device=dev)
    rs = torch.zeros((num_blocks,), dtype=torch.uint8, device=dev)
    deq, st = torch.ops.trtllm.hisparse_read_kvarn_hot_bdr(hot, idx, tl, rs, 0, TPB, KVARN_BITS, KLR, QK_ROPE)
    torch.cuda.synchronize()
    assert int(st.max()) == 0
    return deq.float()


def cosine(a, b):
    a = a.reshape(-1).float(); b = b.reshape(-1).float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def build_decode_inputs(B, next_n, hot_blocks, dev, seed, recency_frac=0.6):
    g = torch.Generator(device="cpu").manual_seed(seed)
    rows = B * next_n
    q = (torch.randn(B, next_n, H_Q, D_QK, generator=g) * 0.3).to(dev, torch.bfloat16)
    n_rec = int(INDEX_TOPK * recency_frac)
    n_tail = INDEX_TOPK - n_rec
    rec_lo = max(0, hot_blocks - 8)
    rec_span = hot_blocks - rec_lo
    ar = torch.arange(n_rec)
    rec_blk = (rec_lo + (ar % rec_span)).view(1, -1).expand(rows, -1)
    rec_tok = ((ar // rec_span) % TPB).view(1, -1).expand(rows, -1)
    tail_blk = torch.randint(0, hot_blocks, (rows, n_tail), generator=g)
    tail_tok = torch.randint(0, TPB, (rows, n_tail), generator=g)
    sel_block = torch.cat([rec_blk, tail_blk], dim=1).to(torch.int64)
    sel_tok = torch.cat([rec_tok, tail_tok], dim=1).to(torch.int64)
    gidx = (sel_block * STRIDE_FACTOR + 0 * TPB + sel_tok).to(torch.int32)
    indices = gidx.view(B, next_n, INDEX_TOPK).to(dev).contiguous()
    rs = torch.zeros((rows,), dtype=torch.uint8, device=dev)
    return q, indices, rs, sel_block.numpy(), sel_tok.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.ops.load_library(str(args.library))
    dev = torch.device("cuda:0")
    HB = 128
    hot = make_hot_buffer(HB, dev, seed=7)
    deq = dequant_block_tokens(hot, HB, dev)

    # correctness gate vs dense reference at B16/nn1
    qC, idxC, rsC, sbC, stC = build_decode_inputs(16, 1, HB, dev, seed=101)
    outC, lseC, _, _ = torch.ops.trtllm.sparse_mla_decode_kvarn_hot(
        qC, hot, idxC, rsC, None, None, 0, TPB, STRIDE_FACTOR, KVARN_BITS, KLR, QK_ROPE, SM)
    torch.cuda.synchronize()
    deq_sel = torch.empty(16, INDEX_TOPK, D_QK, dtype=torch.float32, device=dev)
    for rr in range(16):
        deq_sel[rr] = deq[torch.from_numpy(sbC[rr]).to(dev), torch.from_numpy(stC[rr]).to(dev), :]
    qf = qC.float().reshape(16, H_Q, D_QK)
    scores = SM * torch.einsum("rhd,rnd->rhn", qf, deq_sel)
    m = scores.max(-1, keepdim=True).values
    w = torch.exp(scores - m); den = w.sum(-1, keepdim=True); wn = w / den
    out_ref = torch.einsum("rhn,rnd->rhd", wn, deq_sel[..., :D_V]).reshape(16, 1, H_Q, D_V)
    lse_ref = (torch.log(den.squeeze(-1)) + m.squeeze(-1)).reshape(16, 1, H_Q)
    cos = cosine(outC.float(), out_ref)
    lse_cos = cosine(lseC.float().transpose(1, 2).reshape(16, 1, H_Q), lse_ref)
    print(f"[dense-ref {args.tag}] out cosine={cos:.6f} lse cosine={lse_cos:.6f} "
          f"-> {'PASS' if cos >= 0.999 else 'FAIL'} (gate>=0.999)", flush=True)

    # focused buckets
    for (B, nn) in [(16, 1), (64, 2)]:
        q, idx, rs, _, _ = build_decode_inputs(B, nn, HB, dev, seed=300 + B + nn)
        fn = lambda: torch.ops.trtllm.sparse_mla_decode_kvarn_hot(
            q, hot, idx, rs, None, None, 0, TPB, STRIDE_FACTOR, KVARN_BITS, KLR, QK_ROPE, SM)
        mean_us, p50, p95, ui = time_cuda(fn)
        rows = B * nn
        print(f"[bench {args.tag}] B={B} nn={nn} rows={rows} hb={HB}: "
              f"{mean_us:.1f} us/call  ms/row={mean_us/1000/rows:.4f}  "
              f"(p50={p50:.1f} p95={p95:.1f} us, {ui}it)", flush=True)


if __name__ == "__main__":
    main()

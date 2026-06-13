#!/usr/bin/env python3
"""1c correctness PROBE (hot-read): establish the TRUE dense reference.

Validates torch.ops.trtllm.sparse_mla_decode_kvarn_hot against a dense MLA
attention reference built from the SAME dequantized packed records, read via the
native BDR reader torch.ops.trtllm.hisparse_read_kvarn_hot_bdr. Quant error is
thus SHARED between kernel and reference, so the test isolates the kernel's
attention/softmax/reduction math (not a fused-vs-sequential self-comparison).

Kernel math (sparse_mla_decode_kvarn_hot.cu:170-423), per (row, head):
  score[k] = sm_scale * dot(q[row,head,0:576], Klatent[token_k, 0:576])
  w        = softmax_k(score)              # over rowTopK active tokens
  out[row,head,0:512] = sum_k w[k] * Vlatent[token_k, 0:512]   # V = first 512 dims
where Klatent/Vlatent come from readHisparseKvarnK2v2BdrLatentValue (the same
function hisparse_read_kvarn_hot_bdr exposes).
"""
from __future__ import annotations
import torch
import tensorrt_llm  # noqa: F401

H_Q, D_QK, D_V = 128, 576, 512
TPB, KLR, QK_ROPE, KVARN_BITS = 64, 512, 64, 2
PACKED_PER_BLOCK = TPB*(KLR*KVARN_BITS//8) + TPB*2*(KLR//128)*2 + TPB*QK_ROPE  # 13312

def make_packed_blocks(num_blocks, device, seed=0):
    """Write a known random latent into packed kvarn_k2v2 records via the native
    BDR writer. Returns (hot_packed[num_layers=1, num_blocks, PACKED], latent_src
    [num_blocks, TPB, 576])."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    # modest dynamic range so 2-bit C-KV quant is representative, not saturated
    latent = (torch.randn(num_blocks, TPB, D_QK, generator=g) * 0.5)
    latent = latent.to(device=device, dtype=torch.float16)
    hot = torch.zeros(num_blocks, PACKED_PER_BLOCK, dtype=torch.uint8, device=device)
    for b in range(num_blocks):
        torch.ops.trtllm.mla_bdr_write_kvarn_record(
            latent[b], hot, b, KVARN_BITS, KLR, QK_ROPE)
    torch.cuda.synchronize()
    return hot.unsqueeze(0), latent  # [1, num_blocks, PACKED]

def dequant_via_native(hot_packed, num_blocks, device):
    """Read the dequantized latent the kernel SEES, via the native BDR reader.
    Returns deq[num_blocks, TPB, 576] bf16 -> float."""
    # build hot_indices that address (slot=b, layer=0, tokenOffset=t):
    # global = slot*strideFactor + 0*TPB + t ; strideFactor = numLayers*TPB = TPB
    stride_factor = 1 * TPB
    rows = num_blocks
    idx = torch.full((rows, TPB), -1, dtype=torch.int32, device=device)
    for b in range(num_blocks):
        for t in range(TPB):
            idx[b, t] = b*stride_factor + 0*TPB + t
    topk_len = torch.full((rows,), TPB, dtype=torch.int32, device=device)
    row_status = torch.zeros((rows,), dtype=torch.uint8, device=device)
    latent_out, out_status = torch.ops.trtllm.hisparse_read_kvarn_hot_bdr(
        hot_packed, idx, topk_len, row_status, 0, TPB, KVARN_BITS, KLR, QK_ROPE)
    torch.cuda.synchronize()
    assert int(out_status.max()) == 0, f"BDR read status nonzero: {out_status.cpu().tolist()}"
    return latent_out.float()  # [num_blocks, TPB, 576]

def dense_reference(q, deq_latent_per_token, sm_scale):
    """q:[B,1,H,576] bf16; deq_latent_per_token:[B, n_tok, 576] float (the
    selected tokens' dequantized latents, same order as `indices`).
    Returns out_ref:[B,1,H,512] float and lse_ref:[B,H] float."""
    qf = q.float()  # [B,1,H,576]
    B = qf.shape[0]; H = qf.shape[2]
    K = deq_latent_per_token  # [B, n, 576]
    V = deq_latent_per_token[..., :D_V]  # [B, n, 512]
    # scores[b,h,n] = sm_scale * (qf[b,0,h,:] . K[b,n,:])
    scores = sm_scale * torch.einsum("bhd,bnd->bhn", qf[:,0], K)  # [B,H,n]
    m = scores.max(dim=-1, keepdim=True).values
    w = torch.exp(scores - m)
    denom = w.sum(dim=-1, keepdim=True)
    wn = w / denom
    out = torch.einsum("bhn,bnd->bhd", wn, V)  # [B,H,512]
    lse = (torch.log(denom.squeeze(-1)) + m.squeeze(-1))  # [B,H]
    return out.unsqueeze(1), lse

def cosine(a, b):
    a = a.reshape(-1).float(); b = b.reshape(-1).float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))

def main():
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    sm_scale = 1.0 / (D_QK ** 0.5)

    # Build a hot buffer of N blocks; each decode row selects n_sel tokens that
    # all map into resident hot slots (committed-hot path; no resident sink/tail).
    N_BLOCKS = 32
    hot_packed, latent_src = make_packed_blocks(N_BLOCKS, dev, seed=7)
    deq = dequant_via_native(hot_packed, N_BLOCKS, dev)  # [N_BLOCKS, TPB, 576]
    stride_factor = 1 * TPB

    B = 4
    n_sel = 256  # selected tokens per row
    g = torch.Generator(device="cpu").manual_seed(11)
    q = (torch.randn(B, 1, H_Q, D_QK, generator=g) * 0.3).to(device=dev, dtype=torch.bfloat16)

    # choose n_sel random (block, tokenOffset) per row -> hot global indices
    indices = torch.empty(B, 1, n_sel, dtype=torch.int32, device=dev)
    deq_sel = torch.empty(B, n_sel, D_QK, dtype=torch.float32, device=dev)
    for b in range(B):
        blk = torch.randint(0, N_BLOCKS, (n_sel,), generator=g)
        tok = torch.randint(0, TPB, (n_sel,), generator=g)
        gidx = blk*stride_factor + 0*TPB + tok
        indices[b, 0] = gidx.to(torch.int32).to(dev)
        deq_sel[b] = deq[blk, tok, :]  # exact dequantized latent the kernel reads
    row_status = torch.zeros((B,), dtype=torch.uint8, device=dev)

    out, lse, meta, splits = torch.ops.trtllm.sparse_mla_decode_kvarn_hot(
        q, hot_packed, indices, row_status, None, None,
        0, TPB, stride_factor, KVARN_BITS, KLR, QK_ROPE, sm_scale)
    torch.cuda.synchronize()
    # lse returned transposed to [B, H, s_q]
    out = out.float()                  # [B,1,H,512]
    lse_k = lse.float()                # [B,H,1]

    out_ref, lse_ref = dense_reference(q, deq_sel, sm_scale)  # [B,1,H,512], [B,H]

    cos = cosine(out, out_ref)
    max_abs = float((out - out_ref).abs().max())
    lse_cos = cosine(lse_k.squeeze(-1), lse_ref)
    lse_maxabs = float((lse_k.squeeze(-1) - lse_ref).abs().max())
    print(f"hot-read committed-path:  out cosine = {cos:.6f}  max_abs_diff = {max_abs:.4e}")
    print(f"hot-read committed-path:  lse cosine = {lse_cos:.6f}  lse_max_abs = {lse_maxabs:.4e}")
    print(f"VERDICT cosine>=0.98: {'PASS' if cos >= 0.98 else 'FAIL'}")

if __name__ == "__main__":
    main()

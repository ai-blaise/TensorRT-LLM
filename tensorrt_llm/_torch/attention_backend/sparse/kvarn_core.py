# SPDX-License-Identifier: Apache-2.0
"""KVarN variance-normalized KV-cache quantization, ported for op-trt.

Component-level port of Huawei KVarN (arXiv:2606.03458) into the
TensorRT-LLM op-trt sparse/MLA KV-cache path.

Method (verbatim from the paper + Huawei vLLM reference):
  1. Hadamard rotation along head_dim (orthonormal, ``x @ H``; H is its own
     inverse so dequant un-rotates with the same matmul). Applied to the
     channel axis only; never the token axis (would cost group^2 ops/token).
  2. Iterative log-domain variance-normalization (SINQ/Sinkhorn-style):
     alternating column-std and row-std normalization in log space over the
     [R, C] tile, keeping the lowest-imbalance state seen (best-so-far).
  3. Asymmetric per-row RTN at ``bits`` (zero-point = row min).
  4. Absorb the per-row RTN scale+zp into the matching Sinkhorn axis so the
     dequant is two multiplies and one add (the paper's "second scale s2",
     fused so no extra HBM round-trip).

Orientation (KIVI convention, preserved here):
  - K tile is [D, group] (channels x tokens): per-channel RTN rows.
  - V tile is [group, D] (tokens x channels): per-token RTN rows.

For the op-trt MLA latent path, the per-token compressed_kv latent vector
(dim = kv_lora_rank) is quantized with the *V orientation* (per-token rows),
because the latent is already a learned low-rank projection with no
per-channel softmax-exponential sensitivity — token-magnitude is the error
driver (paper Sec 3.1, Fig 1). The rope-pe key sub-vector uses the *K
orientation* (per-channel). See kvarn_mla.py for the latent wiring.

This module is pure-torch and import-light so it unit-tests standalone.
"""

from __future__ import annotations

import torch

_CLIP_STD_MIN = 1e-3
_CLIP_STD_MAX = 1e3
_LOG_S_MIN = -0.3
_LOG_S_MAX = 10.0


def hadamard_matrix(d: int, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    """Normalized Sylvester-Hadamard [d, d]; d must be a power of two.

    H @ H.T == I, and (H / sqrt(d)) is symmetric+orthonormal so it is its own
    inverse: rotating with ``x @ H`` and un-rotating with ``y @ H`` round-trips.
    """
    assert d & (d - 1) == 0, f"head_dim {d} must be a power of two for Hadamard"
    H = torch.ones(1, 1)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    import math
    return (H / math.sqrt(d)).to(device=device, dtype=dtype)


def _imbalance(tile: torch.Tensor) -> torch.Tensor:
    """Column-std spread + row-std spread; 2.0 == perfectly balanced."""
    sc = tile.std(dim=-2)
    sr = tile.std(dim=-1)
    return (
        sc.amax(dim=-1) / sc.amin(dim=-1).clamp_min(1e-8)
        + sr.amax(dim=-1) / sr.amin(dim=-1).clamp_min(1e-8)
    )


def variance_normalize_batched(
    tiles: torch.Tensor, iterations: int = 16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched log-domain Sinkhorn variance-norm.

    Args:
        tiles: [N, R, C] real; cast to fp32 internally.
        iterations: alternating col/row passes (paper default 16; ~4 in practice).

    Returns (balanced [N,R,C], s_col [N,1,C], s_row [N,R,1]) with
    ``balanced = tiles / s_col / s_row``, best-imbalance state selected per tile.
    """
    m = tiles.float()
    N, R, C = m.shape
    dev = m.device
    log_s_col = torch.zeros(N, 1, C, device=dev)
    log_s_row = torch.zeros(N, R, 1, device=dev)

    cur = m / log_s_col.exp() / log_s_row.exp()
    imb_best = _imbalance(cur)
    sc_best = log_s_col.exp().clone()
    sr_best = log_s_row.exp().clone()

    for _ in range(iterations):
        col_std = cur.std(dim=1, keepdim=True).clamp(_CLIP_STD_MIN, _CLIP_STD_MAX)
        log_s_col = (log_s_col + col_std.log()).clip(_LOG_S_MIN, _LOG_S_MAX)
        cur = m / log_s_col.exp() / log_s_row.exp()

        row_std = cur.std(dim=2, keepdim=True).clamp(_CLIP_STD_MIN, _CLIP_STD_MAX)
        log_s_row = (log_s_row + row_std.log()).clip(_LOG_S_MIN, _LOG_S_MAX)
        cur = m / log_s_col.exp() / log_s_row.exp()

        imb = _imbalance(cur)
        better = imb <= imb_best
        if better.any():
            mask = better.view(N, 1, 1).to(log_s_col.dtype)
            sc_best = mask * log_s_col.exp() + (1 - mask) * sc_best
            sr_best = mask * log_s_row.exp() + (1 - mask) * sr_best
            imb_best = torch.where(better, imb, imb_best)

    balanced = m / sc_best / sr_best
    return balanced, sc_best, sr_best


def _pack_lowbit(q: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack ``bits``-bit ints into uint8, 8//bits values per byte, low-first."""
    pack = 8 // bits
    C = q.shape[-1]
    assert C % pack == 0, f"last dim {C} must be divisible by {pack} for {bits}-bit"
    q = (q.to(torch.uint8) & ((1 << bits) - 1)).reshape(*q.shape[:-1], C // pack, pack)
    out = q[..., 0].clone()
    for j in range(1, pack):
        out = out | (q[..., j] << (j * bits))
    return out.to(torch.uint8)


def _unpack_lowbit(packed: torch.Tensor, bits: int, orig_last: int) -> torch.Tensor:
    """Inverse of _pack_lowbit -> [..., orig_last] uint8 in [0, 2^bits-1]."""
    pack = 8 // bits
    mask = (1 << bits) - 1
    cols = []
    for j in range(pack):
        cols.append(((packed >> (j * bits)) & mask).unsqueeze(-1))
    out = torch.cat(cols, dim=-1).reshape(*packed.shape[:-1], orig_last)
    return out.to(torch.uint8)


def kvarn_quant_rows(
    tiles: torch.Tensor, bits: int, iterations: int = 16
) -> dict[str, torch.Tensor]:
    """Variance-normalize + asymmetric per-row RTN + pack, batched.

    Args:
        tiles: [N, R, C] real, already Hadamard-rotated along the channel axis.
               RTN is per-row (axis 1 = R rows). For K orientation R=D
               (per-channel); for V orientation R=group (per-token).
        bits: 2/3/4.
        iterations: Sinkhorn iters.

    Returns packed record:
        q_packed : [N, R, C/pack] uint8
        s_row_abs: [N, R] fp16   absorbed per-row scale  (= s_row_sinkhorn*rtn_scale)
        zp_abs   : [N, R] fp16   absorbed per-row zero    (= s_row_sinkhorn*rtn_zp)
        s_col    : [N, C] fp16   untouched per-col sinkhorn scale
    Dequant: ``x[n,r,c] = (q*s_row_abs[n,r] + zp_abs[n,r]) * s_col[n,c]``.
    """
    balanced, s_col, s_row = variance_normalize_batched(tiles, iterations)
    qmax = (1 << bits) - 1
    lo = balanced.amin(dim=2, keepdim=True)
    hi = balanced.amax(dim=2, keepdim=True)
    scale = ((hi - lo) / qmax).clamp_min(1e-10)
    zp = lo
    q = torch.clamp(torch.round((balanced - zp) / scale), 0, qmax).to(torch.int32)
    s_row_abs = (s_row.squeeze(-1) * scale.squeeze(-1)).to(torch.float16)
    zp_abs = (s_row.squeeze(-1) * zp.squeeze(-1)).to(torch.float16)
    s_col_out = s_col.squeeze(1).to(torch.float16)
    q_packed = _pack_lowbit(q, bits)
    return {"q_packed": q_packed, "s_row_abs": s_row_abs, "zp_abs": zp_abs, "s_col": s_col_out}


def kvarn_dequant_rows(
    rec: dict[str, torch.Tensor], bits: int, orig_last: int
) -> torch.Tensor:
    """Inverse of kvarn_quant_rows -> [N, R, C] fp32 in the rotated frame."""
    q = _unpack_lowbit(rec["q_packed"], bits, orig_last).float()
    s_row = rec["s_row_abs"].float().unsqueeze(-1)
    zp = rec["zp_abs"].float().unsqueeze(-1)
    s_col = rec["s_col"].float().unsqueeze(1)
    return (q * s_row + zp) * s_col

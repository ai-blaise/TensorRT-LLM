"""Fused low-rank gate (REAP gated-norm) and sigmoid-mul decode kernels.

The REAP graft adds two glue chains to every DeepSeek-V3.2 decoder layer:

1. Low-rank gated norm (2x/layer, modeling_deepseekv3._maybe_apply_gated_norm):
       y = x * sigmoid(silu(x.float() @ Wd.float().T) @ Wu.T)
   Eagerly this is 9 kernels per call, including a per-call fp32 cast of the
   down weight and a pathological cuBLAS SGEMM (gemmSN_TN, ~40us for a
   [tokens,7168]x[7168,16] product). `fused_lowrank_gate` runs the whole chain
   as a split-K Triton pair (partial down-proj, then reduce + silu + up-proj +
   sigmoid + mul) against weights pre-cast once at first use.

2. Attention output gate (1x/layer, attention.py MLA forward):
       attn_output = attn_output * sigmoid(gate)
   `fused_sigmoid_mul` collapses the two elementwise kernels into one.

Rounding points mirror the eager chain (silu -> bf16, up-GEMM output -> bf16,
sigmoid -> bf16, final mul in fp32 -> bf16) so outputs match at bf16 ulp level.
"""

import os
from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False

_GATE_MAX_FUSED_TOKENS = int(
    os.environ.get("TRTLLM_OPTRT_LOWRANK_GATE_MAX_TOKENS", "256"))
_GATE_DISABLED = os.environ.get("TRTLLM_OPTRT_FUSED_LOWRANK_GATE",
                                "1") in ("0", "false", "False")
_SIGMOID_MUL_DISABLED = os.environ.get("TRTLLM_OPTRT_FUSED_SIGMOID_MUL",
                                       "1") in ("0", "false", "False")

_KSPLITS = 7
_BLOCK_K = 1024
_BLOCK_N = 1024

if HAS_TRITON:

    @triton.jit
    def _lowrank_gate_phase1(
        x_ptr,
        wd_ptr,
        ws_ptr,
        N,
        stride_xm,
        KSPLITS: tl.constexpr,
        R: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # Partial down-projection: ws[row, ksplit, :] = x[row, kslice] @ Wd.T.
        # Split-K keeps the K=7168 reduction parallel instead of one
        # latency-bound serial loop per row.
        row = tl.program_id(0)
        ks = tl.program_id(1)
        r_idx = tl.arange(0, R)
        acc = tl.zeros([R], dtype=tl.float32)
        chunk = tl.cdiv(N, KSPLITS * BLOCK_K) * BLOCK_K
        for k0 in range(0, tl.cdiv(chunk, BLOCK_K)):
            offs = ks * chunk + k0 * BLOCK_K + tl.arange(0, BLOCK_K)
            kmask = offs < N
            xk = tl.load(x_ptr + row * stride_xm + offs,
                         mask=kmask,
                         other=0.0).to(tl.float32)
            wd = tl.load(wd_ptr + r_idx[:, None] * N + offs[None, :],
                         mask=kmask[None, :],
                         other=0.0)
            acc += tl.sum(wd * xk[None, :], axis=1)
        tl.store(ws_ptr + (row * KSPLITS + ks) * R + r_idx, acc)

    @triton.jit
    def _lowrank_gate_phase2(
        x_ptr,
        ws_ptr,
        wu_ptr,
        y_ptr,
        N,
        stride_xm,
        stride_ym,
        KSPLITS: tl.constexpr,
        R: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        # Reduce split-K partials (deterministic), then silu -> up-proj ->
        # sigmoid -> mul for one BLOCK_N slice of the row. Rounding points
        # mirror the eager chain (silu/up-GEMM/sigmoid each round to bf16).
        row = tl.program_id(0)
        nc = tl.program_id(1)
        r_idx = tl.arange(0, R)
        KS_POW2: tl.constexpr = triton.next_power_of_2(KSPLITS)
        ks_idx = tl.arange(0, KS_POW2)
        part = tl.load(ws_ptr + row * KSPLITS * R + ks_idx[:, None] * R +
                       r_idx[None, :],
                       mask=ks_idx[:, None] < KSPLITS,
                       other=0.0)
        acc = tl.sum(part, axis=0)
        g = acc * (1.0 / (1.0 + tl.exp(-acc)))
        g = g.to(tl.bfloat16).to(tl.float32)
        offs = nc * BLOCK_N + tl.arange(0, BLOCK_N)
        nmask = offs < N
        wu = tl.load(wu_ptr + r_idx[:, None] * N + offs[None, :],
                     mask=nmask[None, :],
                     other=0.0).to(tl.float32)
        dot = tl.sum(g[:, None] * wu, axis=0)
        dot = dot.to(tl.bfloat16).to(tl.float32)
        gate = 1.0 / (1.0 + tl.exp(-dot))
        gate = gate.to(tl.bfloat16).to(tl.float32)
        xv = tl.load(x_ptr + row * stride_xm + offs, mask=nmask,
                     other=0.0).to(tl.float32)
        tl.store(y_ptr + row * stride_ym + offs, (xv * gate).to(tl.bfloat16),
                 mask=nmask)

    @triton.jit
    def _sigmoid_mul_kernel(
        x_ptr,
        g_ptr,
        y_ptr,
        numel,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        s = 1.0 / (1.0 + tl.exp(-g))
        s = s.to(tl.bfloat16).to(tl.float32)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_ptr + offs, (x * s).to(tl.bfloat16), mask=mask)


@torch.library.custom_op("trtllm::fused_lowrank_gate", mutates_args=())
def fused_lowrank_gate(x: torch.Tensor, wd_f32: torch.Tensor,
                       wu_t_bf16: torch.Tensor) -> torch.Tensor:
    """y = x * sigmoid(silu(x @ wd_f32.T) @ wu_t_bf16) for rank-R gates.

    x: [M, N] bf16, row-contiguous. wd_f32: [R, N] fp32 contiguous (pre-cast
    gate_down.weight). wu_t_bf16: [R, N] bf16 contiguous (gate_up.weight.T).
    """
    M, N = x.shape
    R = wd_f32.shape[0]
    y = torch.empty_like(x)
    ws = torch.empty(M * _KSPLITS * R, device=x.device, dtype=torch.float32)
    _lowrank_gate_phase1[(M, _KSPLITS)](
        x,
        wd_f32,
        ws,
        N,
        x.stride(0),
        KSPLITS=_KSPLITS,
        R=R,
        BLOCK_K=_BLOCK_K,
        num_warps=4,
    )
    _lowrank_gate_phase2[(M, triton.cdiv(N, _BLOCK_N))](
        x,
        ws,
        wu_t_bf16,
        y,
        N,
        x.stride(0),
        y.stride(0),
        KSPLITS=_KSPLITS,
        R=R,
        BLOCK_N=_BLOCK_N,
        num_warps=4,
    )
    return y


@fused_lowrank_gate.register_fake
def _(x, wd_f32, wu_t_bf16):
    return torch.empty_like(x)


@torch.library.custom_op("trtllm::fused_sigmoid_mul", mutates_args=())
def fused_sigmoid_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """y = x * sigmoid(gate), both bf16 contiguous with equal numel."""
    y = torch.empty_like(x)
    numel = x.numel()
    BLOCK = 1024
    _sigmoid_mul_kernel[(triton.cdiv(numel, BLOCK), )](
        x,
        gate,
        y,
        numel,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return y


@fused_sigmoid_mul.register_fake
def _(x, gate):
    return torch.empty_like(x)


def get_lowrank_gate_weights(
        gate_down: torch.nn.Linear,
        gate_up: torch.nn.Linear) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-cast gate weights once; the eager chain re-cast them every call."""
    cached = getattr(gate_down, "_lowrank_gate_cached", None)
    if cached is None or cached[2] is not gate_down.weight:
        wd_f32 = gate_down.weight.detach().float().contiguous()
        wu_t = gate_up.weight.detach().t().to(torch.bfloat16).contiguous()
        cached = (wd_f32, wu_t, gate_down.weight)
        gate_down._lowrank_gate_cached = cached
    return cached[0], cached[1]


def lowrank_gate_supported(flat: torch.Tensor, rank: int) -> bool:
    return (HAS_TRITON and not _GATE_DISABLED and flat.is_cuda
            and flat.dtype == torch.bfloat16
            and flat.shape[0] <= _GATE_MAX_FUSED_TOKENS and rank in (8, 16, 32,
                                                                     64))


def sigmoid_mul_supported(x: torch.Tensor, gate: torch.Tensor) -> bool:
    return (HAS_TRITON and not _SIGMOID_MUL_DISABLED and x.is_cuda
            and x.dtype == torch.bfloat16 and gate.dtype == torch.bfloat16
            and x.is_contiguous() and gate.is_contiguous()
            and x.shape == gate.shape)

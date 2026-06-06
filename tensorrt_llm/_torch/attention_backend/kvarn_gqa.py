# SPDX-License-Identifier: Apache-2.0
"""KVarN 2-bit GQA KV-cache layout and reference store/restore primitives.

This module ports the Huawei KVarN GQA tile contract into op-trt's generic KV
cache vocabulary. It is deliberately backend-neutral: one 128-token page maps
to one packed record per ``(layer, block, kv_head)`` and partial/sink tokens are
kept outside the record in fp16 by the runtime that owns request state.

The production decode path still needs a CUDA/Triton attention kernel that reads
these records directly. The primitives here are the correctness oracle and byte
layout used by allocation, transfer metadata, and future kernels.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

import torch

from tensorrt_llm._torch.attention_backend.sparse.kvarn_core import (
    hadamard_matrix,
    variance_normalize_batched,
)

_KVARN_GQA_RE = re.compile(r"^kvarn_k(?P<k>[234])v(?P<v>[234])_g(?P<g>\d+)$")


@dataclass(frozen=True)
class KVarNGQAConfig:
    """KVarN GQA tile preset.

    ``group`` must equal ``tokens_per_block``. ``head_dim`` is fixed to 128 for
    the Huawei public GQA design and Blaise SMC-SD production preset.
    """

    key_bits: int = 2
    value_bits: int = 2
    group: int = 128
    head_dim: int = 128
    sinkhorn_iters: int = 16
    sink_tokens: int = 128

    @property
    def dtype(self) -> str:
        return f"kvarn_k{self.key_bits}v{self.value_bits}_g{self.group}"

    @property
    def k_packed_bytes(self) -> int:
        return math.ceil(self.head_dim * self.group * self.key_bits / 8)

    @property
    def v_packed_bytes(self) -> int:
        return math.ceil(self.group * self.head_dim * self.value_bits / 8)

    @property
    def k_scale_bytes(self) -> int:
        return (2 * self.head_dim + self.group) * 2

    @property
    def v_scale_bytes(self) -> int:
        return (self.head_dim + 2 * self.group) * 2

    @property
    def tile_bytes(self) -> int:
        return (self.k_packed_bytes + self.k_scale_bytes +
                self.v_packed_bytes + self.v_scale_bytes)

    @property
    def tile_bytes_aligned(self) -> int:
        return ((self.tile_bytes + 7) // 8) * 8

    @property
    def bytes_per_token_slot(self) -> int:
        if self.tile_bytes_aligned % self.group != 0:
            raise ValueError(
                f"KVarN GQA tile_bytes_aligned={self.tile_bytes_aligned} is not "
                f"divisible by group={self.group}")
        return self.tile_bytes_aligned // self.group

    @property
    def k_packed_offset(self) -> int:
        return 0

    @property
    def k_s_col_offset(self) -> int:
        return self.k_packed_offset + self.k_packed_bytes

    @property
    def k_zp_offset(self) -> int:
        return self.k_s_col_offset + self.head_dim * 2

    @property
    def k_s_row_offset(self) -> int:
        return self.k_zp_offset + self.head_dim * 2

    @property
    def v_packed_offset(self) -> int:
        return self.k_s_row_offset + self.group * 2

    @property
    def v_s_col_offset(self) -> int:
        return self.v_packed_offset + self.v_packed_bytes

    @property
    def v_s_row_offset(self) -> int:
        return self.v_s_col_offset + self.head_dim * 2

    @property
    def v_zp_offset(self) -> int:
        return self.v_s_row_offset + self.group * 2

    def validate_runtime(self, *, tokens_per_block: int, head_dim: int) -> None:
        if tokens_per_block != self.group:
            raise ValueError(
                f"KVarN GQA requires tokens_per_block={self.group}, got "
                f"{tokens_per_block}")
        if head_dim != self.head_dim:
            raise ValueError(
                f"KVarN GQA requires head_dim={self.head_dim}, got {head_dim}")


def is_kvarn_gqa_dtype(dtype: object) -> bool:
    return isinstance(dtype, str) and _KVARN_GQA_RE.match(dtype.lower()) is not None


def parse_kvarn_gqa_dtype(dtype: str,
                          *,
                          head_dim: int = 128,
                          sinkhorn_iters: int = 16,
                          sink_tokens: int = 128) -> KVarNGQAConfig:
    match = _KVARN_GQA_RE.match(dtype.lower()) if isinstance(dtype, str) else None
    if match is None:
        raise ValueError(
            "KVarN GQA dtype must be 'kvarn_k<2|3|4>v<2|3|4>_g128', "
            f"got {dtype!r}")
    group = int(match.group("g"))
    if group != 128:
        raise ValueError(f"KVarN GQA currently supports only g128, got g{group}")
    if head_dim != 128:
        raise ValueError(f"KVarN GQA currently supports only head_dim=128, got {head_dim}")
    return KVarNGQAConfig(
        key_bits=int(match.group("k")),
        value_bits=int(match.group("v")),
        group=group,
        head_dim=head_dim,
        sinkhorn_iters=sinkhorn_iters,
        sink_tokens=sink_tokens,
    )


def _pack_bits_flat(q: torch.Tensor, bits: int, packed_bytes: int) -> torch.Tensor:
    """Pack each leading row of integer values into a contiguous bitstream."""
    if bits not in (2, 3, 4):
        raise ValueError(f"KVarN supports 2/3/4-bit packing, got {bits}")
    flat = q.reshape(q.shape[0], -1).to(torch.int32)
    n_values = flat.shape[1]
    needed = math.ceil(n_values * bits / 8)
    if needed != packed_bytes:
        raise ValueError(f"packed byte mismatch: layout={packed_bytes}, values need {needed}")
    offsets = torch.arange(n_values, device=flat.device, dtype=torch.int64) * bits
    byte_idx = offsets // 8
    shifts = (offsets % 8).to(torch.int32)
    out = torch.zeros((flat.shape[0], packed_bytes), dtype=torch.int32, device=flat.device)
    shifted = flat << shifts.unsqueeze(0)
    out.scatter_add_(1, byte_idx.expand(flat.shape[0], -1), shifted & 0xFF)
    spill = (shifts + bits) > 8
    if spill.any():
        spill_idx = byte_idx[spill] + 1
        spill_val = shifted[:, spill] >> 8
        out.scatter_add_(1, spill_idx.expand(flat.shape[0], -1), spill_val)
    return out.to(torch.uint8)


def _unpack_bits_flat(packed: torch.Tensor, bits: int, shape: tuple[int, int, int]) -> torch.Tensor:
    """Unpack ``[N, packed_bytes]`` into ``shape`` uint8 values."""
    n, rows, cols = shape
    n_values = rows * cols
    offsets = torch.arange(n_values, device=packed.device, dtype=torch.int64) * bits
    byte_idx = offsets // 8
    shifts = (offsets % 8).to(torch.int32)
    src = packed.to(torch.int32)
    gathered = src[:, byte_idx].clone()
    spill = (shifts + bits) > 8
    if spill.any():
        gathered[:, spill] = gathered[:, spill] | (src[:, byte_idx[spill] + 1] << 8)
    q = (gathered >> shifts.unsqueeze(0)) & ((1 << bits) - 1)
    return q.to(torch.uint8).reshape(n, rows, cols)


def _quant_rows(tiles: torch.Tensor, bits: int, packed_bytes: int,
                iterations: int) -> dict[str, torch.Tensor]:
    balanced, s_col, s_row = variance_normalize_batched(tiles, iterations)
    qmax = (1 << bits) - 1
    lo = balanced.amin(dim=2, keepdim=True)
    hi = balanced.amax(dim=2, keepdim=True)
    scale = ((hi - lo) / qmax).clamp_min(1e-10)
    zp = lo
    q = torch.clamp(torch.round((balanced - zp) / scale), 0, qmax).to(torch.int32)
    return {
        "q_packed": _pack_bits_flat(q, bits, packed_bytes),
        "s_row_abs": (s_row.squeeze(-1) * scale.squeeze(-1)).to(torch.float16),
        "zp_abs": (s_row.squeeze(-1) * zp.squeeze(-1)).to(torch.float16),
        "s_col": s_col.squeeze(1).to(torch.float16),
    }


def _dequant_rows(rec: dict[str, torch.Tensor], bits: int,
                  shape: tuple[int, int, int]) -> torch.Tensor:
    q = _unpack_bits_flat(rec["q_packed"], bits, shape).float()
    s_row = rec["s_row_abs"].float().unsqueeze(-1)
    zp = rec["zp_abs"].float().unsqueeze(-1)
    s_col = rec["s_col"].float().unsqueeze(1)
    return (q * s_row + zp) * s_col


def _bytes_from_fp16(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8).reshape(t.shape[0], -1)


def _fp16_from_bytes(t: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    return t.contiguous().view(torch.float16).reshape(shape)


def quantize_gqa_tile(k_tile: torch.Tensor, v_tile: torch.Tensor,
                      cfg: Optional[KVarNGQAConfig] = None) -> torch.Tensor:
    """Quantize one full GQA tile.

    Args:
        k_tile: ``[group, num_kv_heads, head_dim]`` fp16/bf16/fp32.
        v_tile: ``[group, num_kv_heads, head_dim]`` fp16/bf16/fp32.
        cfg: KVarN GQA config, defaults to Blaise k2v2 g128.

    Returns:
        ``[num_kv_heads, cfg.tile_bytes_aligned]`` uint8 packed records.
    """
    cfg = cfg or KVarNGQAConfig()
    if k_tile.shape != v_tile.shape or k_tile.ndim != 3:
        raise ValueError(
            f"expected k/v [group, num_kv_heads, head_dim], got {k_tile.shape}/{v_tile.shape}")
    group, num_kv_heads, head_dim = k_tile.shape
    cfg.validate_runtime(tokens_per_block=group, head_dim=head_dim)

    H = hadamard_matrix(head_dim, k_tile.device, torch.float32)
    k_rot = torch.matmul(k_tile.float(), H).permute(1, 2, 0).contiguous()
    v_rot = torch.matmul(v_tile.float(), H).permute(1, 0, 2).contiguous()
    k_rec = _quant_rows(k_rot, cfg.key_bits, cfg.k_packed_bytes,
                        cfg.sinkhorn_iters)
    v_rec = _quant_rows(v_rot, cfg.value_bits, cfg.v_packed_bytes,
                        cfg.sinkhorn_iters)

    records = torch.zeros((num_kv_heads, cfg.tile_bytes_aligned),
                          dtype=torch.uint8,
                          device=k_tile.device)

    def put(offset: int, data: torch.Tensor) -> None:
        records[:, offset:offset + data.shape[1]] = data

    put(cfg.k_packed_offset, k_rec["q_packed"])
    put(cfg.k_s_col_offset, _bytes_from_fp16(k_rec["s_row_abs"]))
    put(cfg.k_zp_offset, _bytes_from_fp16(k_rec["zp_abs"]))
    put(cfg.k_s_row_offset, _bytes_from_fp16(k_rec["s_col"]))
    put(cfg.v_packed_offset, v_rec["q_packed"])
    put(cfg.v_s_col_offset, _bytes_from_fp16(v_rec["s_col"]))
    put(cfg.v_s_row_offset, _bytes_from_fp16(v_rec["s_row_abs"]))
    put(cfg.v_zp_offset, _bytes_from_fp16(v_rec["zp_abs"]))
    return records


def dequantize_gqa_tile(records: torch.Tensor,
                        cfg: Optional[KVarNGQAConfig] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Restore a full GQA tile from KVarN packed records.

    Args:
        records: ``[num_kv_heads, cfg.tile_bytes_aligned]`` uint8.

    Returns:
        ``(k, v)`` as ``[group, num_kv_heads, head_dim]`` fp32 tensors in the
        original, unrotated frame.
    """
    cfg = cfg or KVarNGQAConfig()
    if records.ndim != 2 or records.shape[1] < cfg.tile_bytes:
        raise ValueError(
            f"expected records [num_kv_heads, >= {cfg.tile_bytes}], got {records.shape}")
    num_kv_heads = records.shape[0]

    def get(offset: int, nbytes: int) -> torch.Tensor:
        return records[:, offset:offset + nbytes].contiguous()

    k_rec = {
        "q_packed": get(cfg.k_packed_offset, cfg.k_packed_bytes),
        "s_row_abs": _fp16_from_bytes(get(cfg.k_s_col_offset, cfg.head_dim * 2),
                                      (num_kv_heads, cfg.head_dim)),
        "zp_abs": _fp16_from_bytes(get(cfg.k_zp_offset, cfg.head_dim * 2),
                                   (num_kv_heads, cfg.head_dim)),
        "s_col": _fp16_from_bytes(get(cfg.k_s_row_offset, cfg.group * 2),
                                  (num_kv_heads, cfg.group)),
    }
    v_rec = {
        "q_packed": get(cfg.v_packed_offset, cfg.v_packed_bytes),
        "s_col": _fp16_from_bytes(get(cfg.v_s_col_offset, cfg.head_dim * 2),
                                  (num_kv_heads, cfg.head_dim)),
        "s_row_abs": _fp16_from_bytes(get(cfg.v_s_row_offset, cfg.group * 2),
                                      (num_kv_heads, cfg.group)),
        "zp_abs": _fp16_from_bytes(get(cfg.v_zp_offset, cfg.group * 2),
                                   (num_kv_heads, cfg.group)),
    }
    k_rot = _dequant_rows(k_rec, cfg.key_bits,
                          (num_kv_heads, cfg.head_dim, cfg.group))
    v_rot = _dequant_rows(v_rec, cfg.value_bits,
                          (num_kv_heads, cfg.group, cfg.head_dim))
    H = hadamard_matrix(cfg.head_dim, records.device, torch.float32)
    k = torch.matmul(k_rot.permute(2, 0, 1).contiguous(), H)
    v = torch.matmul(v_rot.permute(1, 0, 2).contiguous(), H)
    return k, v


class KVarNGQAPackedPool:
    """Reference packed-record pool for one model/rank.

    The production cache manager should expose the same logical records through
    its byte-backed KV pages, while request-owned fp16 sink/tail buffers remain
    separate. This class is small enough for unit tests and transfer metadata
    shape checks.
    """

    def __init__(self, num_layers: int, num_blocks: int, num_kv_heads: int,
                 cfg: Optional[KVarNGQAConfig] = None,
                 device: Optional[torch.device] = None):
        self.cfg = cfg or KVarNGQAConfig()
        device = device or torch.device("cpu")
        self.records = torch.zeros(
            (num_layers, num_blocks, num_kv_heads, self.cfg.tile_bytes_aligned),
            dtype=torch.uint8,
            device=device,
        )
        self.valid = torch.zeros((num_layers, num_blocks), dtype=torch.bool, device=device)
        self.commit_gen = torch.zeros((num_layers, num_blocks), dtype=torch.int64, device=device)

    def store_block(self, layer_idx: int, block_idx: int, k_tile: torch.Tensor,
                    v_tile: torch.Tensor) -> None:
        record = quantize_gqa_tile(k_tile, v_tile, self.cfg)
        expected_heads = self.records.shape[2]
        if record.shape[0] != expected_heads:
            raise ValueError(f"expected {expected_heads} kv heads, got {record.shape[0]}")
        self.records[layer_idx, block_idx].copy_(record)
        self.valid[layer_idx, block_idx] = True
        self.commit_gen[layer_idx, block_idx] += 1

    def load_block(self, layer_idx: int, block_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(self.valid[layer_idx, block_idx].item()):
            raise ValueError(f"KVarN GQA block layer={layer_idx} block={block_idx} is not committed")
        return dequantize_gqa_tile(self.records[layer_idx, block_idx], self.cfg)

    def transfer_view(self, layer_idx: int, block_indices: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return metadata tensors that disagg transfer must move opaquely."""
        return {
            "format": torch.tensor([1], dtype=torch.int32, device=self.records.device),
            "tile_bytes": torch.tensor([self.cfg.tile_bytes_aligned], dtype=torch.int32,
                                       device=self.records.device),
            "records": self.records[layer_idx, block_indices],
            "valid": self.valid[layer_idx, block_indices],
            "commit_gen": self.commit_gen[layer_idx, block_indices],
        }

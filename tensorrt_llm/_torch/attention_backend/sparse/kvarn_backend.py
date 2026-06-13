# SPDX-License-Identifier: Apache-2.0
"""KVarN system-tier backend: mla_latent_kv_dtype="kvarn_*" for the op-trt MLA path.

This is the *system-level* wiring that turns the validated component
(``kvarn_core`` + ``kvarn_mla``) into a selectable dense MLA latent-cache
backend, parallel to the existing fp8 / nvfp4 latent-cache modes. It does not
change Indexer K storage.

Design (faithful to ``kvarn_mla.SYSTEM_INTEGRATION``, Stage-a end-state):

  * ``KVarNConfig`` — the preset (bits per ckv / k_pe, Sinkhorn iters, sink
    tokens) plus the byte accounting. Selected by an
    ``mla_latent_kv_dtype`` string ``kvarn_k{cb}v{pb}`` (e.g.
    ``kvarn_k4v2``) and/or env ``TRTLLM_MLA_LATENT_KV_DTYPE``.

  * ``KVarNLatentPool`` — a per-layer side-pool, allocated by the cache manager
    exactly like the indexer-K side-pool (plain torch tensors, indexed by the
    same paged block_offsets). It holds the *packed* KVarN record bytes for
    each fully-filled MLA latent block, so the realized KV memory is the packed
    footprint (measurable now), not fp16/fp8.

  * Quant-on-write / dequant-on-read operate on the fp16 latent block. The
    write hook fires when a paged block fills to ``group == tokens_per_block``
    tokens; the read hook reconstructs the fp16 ckv/k_pe a decode kernel
    consumes. The partially-filled tail block and the first ``sink_tokens``
    stay fp16 (attention-sink preservation + you cannot Sinkhorn a partial
    tile stably).

Tile reconciliation (the group=128 vs tokens_per_block=64 question):
  KVarN's Sinkhorn tile is ``[group, dim]``. We bind ``group == tokens_per_block``
  so one op-trt paged block == one KVarN tile == one packed record. Production
  op-trt uses ``tokens_per_block == 64``; the component round-trip log shows
  group=64 holds full accuracy (cos_ckv 0.9935 @ 4-bit), so there is no need to
  re-block to 128. The 576-dim latent splits as ckv=512 (power of two) +
  k_pe=64 (power of two) — both Hadamard-legal — exactly as ``quant_latent_block``
  already does.

The C++ end-state (folding the Sinkhorn+RTN store/load into the
``mla_rope_generation`` / dsv3Rope kernel that already exists on origin/op-trt
for NVFP4 dense KV) is the zero-staging production target; this module is the
measurable Stage-a backend and the drop-in point for that fold.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import torch

try:
    from tensorrt_llm._torch.attention_backend.sparse.kvarn_mla import (
        dequant_latent_block,
        packed_bytes_per_block,
        quant_latent_block,
    )
    from tensorrt_llm._torch.attention_backend.sparse.kvarn_core import (
        hadamard_matrix,
        kvarn_dequant_rows,
    )
except ImportError:  # standalone / unit-test
    from kvarn_mla import (  # noqa: F401
        dequant_latent_block,
        packed_bytes_per_block,
        quant_latent_block,
    )
    from kvarn_core import hadamard_matrix, kvarn_dequant_rows  # noqa: F401


# ---------------------------------------------------------------------------
# Config / dtype-string surface
# ---------------------------------------------------------------------------

_KVARN_PREFIX = "kvarn_"
KVARN_LEGACY_SIDEPOOL_LAYOUT = "legacy_sinkhorn_v1"
KVARN_BDR_HISPARSE_LAYOUT = "bdr_ckv_lowbit_fp8_pe_v1"


@dataclass(frozen=True)
class KVarNBDRLayout:
    """Production BDR layout contract for dense-MLA HiSparse hot records.

    This describes the C++ BDR helper in ``mlaKernels.cu`` rather than the
    older Python/Sinkhorn side-pool record. C-KV is low-bit packed after the
    block-diagonal Hadamard rotation, with per-token/sub-block fp16
    ``{scale,zp}``; the RoPE component is carried as byte storage in the same
    hot record until a fused low-bit PE path exists.
    """

    tokens_per_block: int
    ckv_bits: int
    requested_pe_bits: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    name: str = KVARN_BDR_HISPARSE_LAYOUT
    bdr_order: int = 128
    bytes_per_scale: int = 2
    pe_storage_bytes_per_elem: int = 1

    @property
    def num_subblocks(self) -> int:
        return self.kv_lora_rank // self.bdr_order

    @property
    def ckv_packed_bytes_per_token(self) -> int:
        return self.kv_lora_rank * self.ckv_bits // 8

    @property
    def ckv_scale_zp_bytes_per_token(self) -> int:
        return 2 * self.num_subblocks * self.bytes_per_scale

    @property
    def pe_payload_bytes_per_token(self) -> int:
        return self.qk_rope_head_dim * self.pe_storage_bytes_per_elem

    @property
    def pe_storage_bits(self) -> int:
        return 8 * self.pe_storage_bytes_per_elem

    @property
    def packed_bytes_per_token(self) -> int:
        return (self.ckv_packed_bytes_per_token +
                self.ckv_scale_zp_bytes_per_token +
                self.pe_payload_bytes_per_token)

    @property
    def packed_bytes_per_block(self) -> int:
        return self.tokens_per_block * self.packed_bytes_per_token

    @property
    def field_offsets(self) -> dict[str, tuple[int, int]]:
        ckv_q = self.tokens_per_block * self.ckv_packed_bytes_per_token
        ckv_scale_zp = (
            self.tokens_per_block * self.ckv_scale_zp_bytes_per_token)
        pe = self.tokens_per_block * self.pe_payload_bytes_per_token
        return {
            "ckv_q": (0, ckv_q),
            "ckv_scale_zp": (ckv_q, ckv_q + ckv_scale_zp),
            "pe_byte": (ckv_q + ckv_scale_zp, ckv_q + ckv_scale_zp + pe),
        }


@dataclass(frozen=True)
class KVarNConfig:
    """KVarN MLA-latent backend preset.

    ``ckv_bits`` is spent on the 512-dim content latent (drives K_nope/V
    reconstruction; see mla_preset_ablation: 4 bits -> cos 0.994 vs 2 bits ->
    0.87). ``pe_bits`` is the 64-dim RoPE key (cheap, smaller quality lever).
    """

    ckv_bits: int = 4
    pe_bits: int = 2
    iters: int = 16
    sink_tokens: int = 128
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64

    @property
    def name(self) -> str:
        return f"{_KVARN_PREFIX}k{self.ckv_bits}v{self.pe_bits}"

    @property
    def latent_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim  # 576 for DeepSeek-V3.2

    def packed_bytes(self, group: int) -> int:
        """Packed bytes for one full block of ``group`` tokens."""
        return packed_bytes_per_block(self.kv_lora_rank, self.qk_rope_head_dim,
                                      group, self.ckv_bits, self.pe_bits)

    def fp16_bytes(self, group: int) -> int:
        return 2 * group * self.latent_dim

    def fp8_bytes(self, group: int) -> int:
        return group * self.latent_dim  # 1 byte/elem, no extra scale in fp8 latent

    def bits_per_elem(self, group: int) -> float:
        return self.packed_bytes(group) * 8 / (group * self.latent_dim)

    def hisparse_bdr_layout(self, group: int) -> KVarNBDRLayout:
        """Return the production BDR hot-record layout for HiSparse.

        This is intentionally separate from :meth:`packed_bytes`, which still
        describes the Python/Sinkhorn side-pool used by the amortized restore
        path. HiSparse sparse-MLA hot reads must use this BDR contract or fail
        closed before allocating host/hot tiers.
        """
        if self.ckv_bits not in (2, 4):
            raise ValueError(
                "HiSparse dense MLA BDR hot records support ckv_bits 2 or 4, "
                f"got {self.ckv_bits}.")
        if self.kv_lora_rank % 128 != 0:
            raise ValueError(
                "HiSparse dense MLA BDR hot records require kv_lora_rank to "
                f"be divisible by 128, got {self.kv_lora_rank}.")
        return KVarNBDRLayout(
            tokens_per_block=int(group),
            ckv_bits=int(self.ckv_bits),
            requested_pe_bits=int(self.pe_bits),
            kv_lora_rank=int(self.kv_lora_rank),
            qk_rope_head_dim=int(self.qk_rope_head_dim),
        )


def is_kvarn_dtype(mla_latent_kv_dtype) -> bool:
    return isinstance(mla_latent_kv_dtype, str) and mla_latent_kv_dtype.startswith(_KVARN_PREFIX)


def parse_kvarn_dtype(mla_latent_kv_dtype: str, **overrides) -> KVarNConfig:
    """``"kvarn_k4v2"`` -> KVarNConfig(ckv_bits=4, pe_bits=2).

    Accepts model-dim overrides (kv_lora_rank, qk_rope_head_dim, sink_tokens).
    """
    assert is_kvarn_dtype(mla_latent_kv_dtype), mla_latent_kv_dtype
    body = mla_latent_kv_dtype[len(_KVARN_PREFIX):]  # "k4v2"
    cb = pb = None
    if body.startswith("k") and "v" in body:
        kpart, vpart = body[1:].split("v", 1)
        cb, pb = int(kpart), int(vpart)
    if cb is None or pb is None:
        raise ValueError(
            f"bad dense MLA latent KVarN dtype {mla_latent_kv_dtype!r}; "
            "want kvarn_k<ckv>v<pe>")
    for bits in (cb, pb):
        if bits not in (2, 3, 4):
            raise ValueError(f"kvarn bits must be 2/3/4, got {bits}")
    fields = {"ckv_bits": cb, "pe_bits": pb}
    for k in ("iters", "sink_tokens", "kv_lora_rank", "qk_rope_head_dim"):
        if k in overrides and overrides[k] is not None:
            fields[k] = overrides[k]
    return KVarNConfig(**fields)


def resolve_kvarn_config(mla_latent_kv_dtype=None, **overrides):
    """Resolve a KVarNConfig from an explicit dtype string or the env override.

    Returns ``None`` when KVarN is not selected (caller keeps fp8/nvfp4 path).
    """
    if is_kvarn_dtype(mla_latent_kv_dtype):
        return parse_kvarn_dtype(mla_latent_kv_dtype, **overrides)
    env = os.environ.get("TRTLLM_MLA_LATENT_KV_DTYPE", "")
    if is_kvarn_dtype(env):
        return parse_kvarn_dtype(env, **overrides)
    return None


# ---------------------------------------------------------------------------
# Per-layer packed side-pool
# ---------------------------------------------------------------------------

class KVarNLatentPool:
    """Packed KVarN records for the MLA latent cache of ONE layer.

    Mirrors the indexer-K side-pool: a flat byte tensor of
    ``[num_blocks, bytes_per_block]`` indexed by the same paged block id the
    main KV pool uses. A separate ``valid`` flag marks blocks that hold a
    committed KVarN record (vs. blocks still in the fp16 staging tail).

    The packed record for a block is laid out contiguously as:
        ckv q_packed | ckv s_row_abs | ckv zp_abs | ckv s_col |
        kpe q_packed | kpe s_row_abs | kpe zp_abs | kpe s_col
    Field sizes are fixed by (group, dims, bits), so a block is sliced back out
    without any per-block metadata.
    """

    def __init__(self, num_blocks: int, group: int, cfg: KVarNConfig,
                 device: torch.device):
        self.num_blocks = num_blocks
        self.group = group
        self.cfg = cfg
        self.device = device
        self._layout = self._compute_layout(group, cfg)
        self.storage_layout_name = KVARN_LEGACY_SIDEPOOL_LAYOUT
        self.bytes_per_block = self._layout["total_bytes"]
        # Flat uint8 store. One alloc/layer; same shape contract as indexer-K.
        self.store = torch.zeros((num_blocks, self.bytes_per_block),
                                  dtype=torch.uint8, device=device)
        self.valid = torch.zeros((num_blocks,), dtype=torch.bool, device=device)
        # Per-block content epoch: bumped on every (re)commit. The amortized
        # decode restore reconstructs a block into the fp16 main pool only when
        # its restored epoch lags commit_gen (committed blocks are immutable
        # until the block-id is recycled and re-committed). Device int tensor so
        # the per-step stale-block set-diff is a vectorized GPU compare, not a
        # Python loop (the loop dominates at batch>=8, see bench_amort_e2e).
        self.commit_gen = torch.zeros((num_blocks,), dtype=torch.int64,
                                      device=device)
        # Host mirrors of valid / commit_gen plus the restore epoch consumed
        # by the pre-replay delta restore. ``store_block`` is the only writer
        # of the device flags and takes a host block id, so the mirrors stay
        # exact with zero device readback. ``restored_gen_host`` records the
        # commit epoch most recently reconstructed into the fp16 main-pool
        # slot by the delta walk; committed blocks are immutable until their
        # id is recycled and re-committed, so ``restored == commit`` means the
        # fp16 slot already holds the block's dequantized content.
        self.valid_host = np.zeros((num_blocks,), dtype=bool)
        self.commit_gen_host = np.zeros((num_blocks,), dtype=np.int64)
        self.restored_gen_host = np.full((num_blocks,), -1, dtype=np.int64)
        # Hadamard matrices cached once per layer (shared across all blocks).
        self.H_ckv = hadamard_matrix(cfg.kv_lora_rank, device, torch.float32)
        self.H_pe = hadamard_matrix(cfg.qk_rope_head_dim, device, torch.float32)

    @staticmethod
    def _compute_layout(group: int, cfg: KVarNConfig) -> dict:
        """Byte offsets of each record field within one block's slot."""
        Dckv, Dpe = cfg.kv_lora_rank, cfg.qk_rope_head_dim
        cb, pb = cfg.ckv_bits, cfg.pe_bits
        ckv_pack = 8 // cb
        pe_pack = 8 // pb
        # ckv V-orient: rows=group, cols=Dckv -> q_packed [group, Dckv/ckv_pack]
        ckv_q = group * (Dckv // ckv_pack)
        ckv_srow = group * 2      # fp16
        ckv_zp = group * 2
        ckv_scol = Dckv * 2
        # kpe K-orient: rows=Dpe, cols=group -> q_packed [Dpe, group/pe_pack]
        pe_q = Dpe * (group // pe_pack)
        pe_srow = Dpe * 2
        pe_zp = Dpe * 2
        pe_scol = group * 2
        off = 0
        fields = {}
        for nm, sz in [("ckv_q", ckv_q), ("ckv_srow", ckv_srow),
                       ("ckv_zp", ckv_zp), ("ckv_scol", ckv_scol),
                       ("pe_q", pe_q), ("pe_srow", pe_srow),
                       ("pe_zp", pe_zp), ("pe_scol", pe_scol)]:
            fields[nm] = (off, off + sz)
            off += sz
        fields["total_bytes"] = off
        return fields

    # -- write -------------------------------------------------------------

    def _serialize_into(self, block_id: int, rec: dict) -> None:
        L = self._layout
        slot = self.store[block_id]
        ckv, kpe = rec["ckv"], rec["kpe"]

        def put(field, t):
            b0, b1 = L[field]
            slot[b0:b1] = t.reshape(-1).view(torch.uint8)

        put("ckv_q", ckv["q_packed"])
        put("ckv_srow", ckv["s_row_abs"])
        put("ckv_zp", ckv["zp_abs"])
        put("ckv_scol", ckv["s_col"])
        put("pe_q", kpe["q_packed"])
        put("pe_srow", kpe["s_row_abs"])
        put("pe_zp", kpe["zp_abs"])
        put("pe_scol", kpe["s_col"])

    def store_block(self, block_id: int, ckv: torch.Tensor,
                    k_pe: torch.Tensor) -> None:
        """Quantize a full fp16 latent block [group, *] and commit its record."""
        assert ckv.shape[0] == self.group, (ckv.shape, self.group)
        rec = quant_latent_block(ckv, k_pe, ckv_bits=self.cfg.ckv_bits,
                                 pe_bits=self.cfg.pe_bits, iters=self.cfg.iters,
                                 H_ckv=self.H_ckv, H_pe=self.H_pe)
        self._serialize_into(block_id, rec)
        self.valid[block_id] = True
        self.commit_gen[block_id] += 1  # bump content epoch (device tensor)
        bid = int(block_id)
        self.valid_host[bid] = True
        self.commit_gen_host[bid] += 1

    def packed_source_fragments(self, block_ids) -> tuple[np.ndarray, np.ndarray]:
        """Return VRAM source pointers for committed packed KVarN blocks.

        The fragments point directly at the authoritative packed byte records.
        This is the only valid source for HiSparse direct-to-host writes; an
        uncommitted block is still sink/tail fp16 state and must not be
        published as a packed host record.
        """
        ids = np.asarray(block_ids, dtype=np.int64)
        if ids.ndim != 1:
            raise ValueError("KVarN packed source fragments require 1D block ids.")
        if ids.size == 0:
            return (np.array([], dtype=np.int64),
                    np.array([], dtype=np.int64))
        if int(ids.min()) < 0 or int(ids.max()) >= int(self.num_blocks):
            raise ValueError(
                f"KVarN packed source block id out of range: "
                f"min={int(ids.min())}, max={int(ids.max())}, "
                f"num_blocks={self.num_blocks}.")
        uncommitted = ids[~self.valid_host[ids]]
        if uncommitted.size:
            sample = ", ".join(str(int(x)) for x in uncommitted[:8])
            raise RuntimeError(
                "HiSparse direct-to-host requires committed dense MLA KVarN "
                f"records; uncommitted block id(s): {sample}.")
        ptrs = int(self.store.data_ptr()) + ids * int(self.bytes_per_block)
        sizes = np.full(ids.size, int(self.bytes_per_block), dtype=np.int64)
        return ptrs.astype(np.int64, copy=False), sizes

    def stale_committed_host(self, block_ids) -> list:
        """Filter ``block_ids`` (host ints) down to committed blocks whose
        fp16 main-pool slot lags their commit epoch. Pure host; no syncs."""
        ids = np.asarray(block_ids, dtype=np.int64)
        if ids.size == 0:
            return []
        keep = self.valid_host[ids] & (self.restored_gen_host[ids] !=
                                       self.commit_gen_host[ids])
        return ids[keep].tolist()

    def mark_restored_host(self, block_ids) -> None:
        """Record that ``block_ids`` were reconstructed at their current
        commit epoch (call after the restore kernels were launched)."""
        ids = np.asarray(block_ids, dtype=np.int64)
        self.restored_gen_host[ids] = self.commit_gen_host[ids]

    def invalidate_blocks(self, block_ids, dev_ids=None) -> None:
        """Drop the committed records for ``block_ids`` (host ints): the
        free/recycle hook. The cache manager calls this when paged block ids
        return to the allocator (request free / kv rewind). The packed record
        describes the dying owner's content, so the next owner of a recycled
        id must neither skip its own commit (commit idempotence is keyed on
        ``valid``) nor have the stale record restored over its fresh fp16
        block. ``commit_gen`` is left monotonic -- never reset -- so module
        restore epochs (``_kvarn_restored_gen`` / ``restored_gen_host``)
        cannot alias a re-committed id at an old epoch value.

        ``dev_ids`` optionally carries the ids as a device long tensor so a
        caller invalidating across many layer pools materializes it once."""
        ids = np.asarray(block_ids, dtype=np.int64)
        if ids.size == 0:
            return
        live = ids[self.valid_host[ids]]
        if live.size:
            self.valid_host[live] = False
            self.restored_gen_host[live] = -1
        if dev_ids is None:
            if live.size == 0:
                return
            dev_ids = torch.as_tensor(live, dtype=torch.long,
                                      device=self.device)
        # A superset id write is fine (False over False); sharing one device
        # tensor across all layer pools keeps the per-layer cost to a single
        # small index_put launch.
        self.valid[dev_ids] = False

    # -- read --------------------------------------------------------------

    def _deserialize(self, block_id: int) -> dict:
        L = self._layout
        slot = self.store[block_id]
        cfg = self.cfg
        Dckv, Dpe, G = cfg.kv_lora_rank, cfg.qk_rope_head_dim, self.group
        ckv_pack = 8 // cfg.ckv_bits
        pe_pack = 8 // cfg.pe_bits

        def get_u8(field, shape):
            b0, b1 = L[field]
            return slot[b0:b1].view(torch.uint8).reshape(shape)

        def get_f16(field, n):
            b0, b1 = L[field]
            return slot[b0:b1].view(torch.float16).reshape(n)

        ckv_rec = {
            "q_packed": get_u8("ckv_q", (1, G, Dckv // ckv_pack)),
            "s_row_abs": get_f16("ckv_srow", (1, G)),
            "zp_abs": get_f16("ckv_zp", (1, G)),
            "s_col": get_f16("ckv_scol", (1, Dckv)),
        }
        pe_rec = {
            "q_packed": get_u8("pe_q", (1, Dpe, G // pe_pack)),
            "s_row_abs": get_f16("pe_srow", (1, Dpe)),
            "zp_abs": get_f16("pe_zp", (1, Dpe)),
            "s_col": get_f16("pe_scol", (1, G)),
        }
        return {
            "ckv": ckv_rec, "ckv_bits": cfg.ckv_bits, "ckv_dim": Dckv,
            "kpe": pe_rec, "pe_bits": cfg.pe_bits, "pe_dim": Dpe, "group": G,
            "H_ckv": self.H_ckv, "H_pe": self.H_pe,
        }

    def load_block(self, block_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct (ckv [group, Dckv], k_pe [group, Dpe]) fp16."""
        return dequant_latent_block(self._deserialize(block_id))

    def load_blocks(self, block_ids) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched restore of N committed blocks in ONE dequant call each for
        ckv/k_pe -> (ckv [N, group, Dckv], k_pe [N, group, Dpe]) fp16.

        This is the production restore primitive: the per-block python-loop
        load is ~277 us/block, while this batched path is ~1.2-2.4 us/block at
        N>=32 (see system_decode_overhead.log). Restore the indexer-selected
        (sparse top-K) blocks per step, not the full context.
        """
        cfg = self.cfg
        Dckv, Dpe, G = cfg.kv_lora_rank, cfg.qk_rope_head_dim, self.group
        ckv_pack, pe_pack = 8 // cfg.ckv_bits, 8 // cfg.pe_bits
        ids = torch.as_tensor(block_ids, dtype=torch.long, device=self.device)
        N = ids.numel()
        if N == 0:
            z = torch.empty(0, G, 0, device=self.device, dtype=torch.float16)
            return (z.reshape(0, G, Dckv), z.reshape(0, G, Dpe))
        L = self._layout
        slots = self.store[ids]  # [N, bytes_per_block]

        def gu8(field, shape):
            b0, b1 = L[field]
            return slots[:, b0:b1].reshape(shape)

        def gf16(field, shape):
            b0, b1 = L[field]
            return slots[:, b0:b1].view(torch.float16).reshape(shape)

        ckv_rec = {
            "q_packed": gu8("ckv_q", (N, G, Dckv // ckv_pack)),
            "s_row_abs": gf16("ckv_srow", (N, G)),
            "zp_abs": gf16("ckv_zp", (N, G)),
            "s_col": gf16("ckv_scol", (N, Dckv)),
        }
        pe_rec = {
            "q_packed": gu8("pe_q", (N, Dpe, G // pe_pack)),
            "s_row_abs": gf16("pe_srow", (N, Dpe)),
            "zp_abs": gf16("pe_zp", (N, Dpe)),
            "s_col": gf16("pe_scol", (N, G)),
        }
        # ckv: rotated-frame dequant [N,G,Dckv] then un-rotate channels.
        ckv_rot = kvarn_dequant_rows(ckv_rec, cfg.ckv_bits, Dckv)
        ckv = (ckv_rot @ self.H_ckv).to(torch.float16)            # [N, G, Dckv]
        # k_pe: K-orient [N,Dpe,G] then un-rotate + transpose back to [N,G,Dpe].
        pe_rot = kvarn_dequant_rows(pe_rec, cfg.pe_bits, G)        # [N, Dpe, G]
        k_pe = (pe_rot.transpose(1, 2) @ self.H_pe).to(torch.float16)
        return ckv, k_pe


# ---------------------------------------------------------------------------
# Memory accounting (parallels DSACacheManager.get_cache_bytes_per_token)
# ---------------------------------------------------------------------------

def kvarn_latent_bytes_per_token(cfg: KVarNConfig, group: int,
                                 num_attention_layers: int) -> float:
    """Per-token packed bytes for the MLA latent across all layers (KVarN)."""
    return cfg.packed_bytes(group) / group * num_attention_layers


def latent_bytes_per_token_baseline(latent_dim: int, num_attention_layers: int,
                                    bytes_per_elem: int) -> float:
    """fp16(=2)/fp8(=1) per-token latent bytes across all layers."""
    return latent_dim * bytes_per_elem * num_attention_layers

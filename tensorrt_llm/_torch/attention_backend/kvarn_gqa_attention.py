# SPDX-License-Identifier: Apache-2.0
"""Functional KVarN GQA attention backend.

This is the first runnable generic/GQA KVarN path for SMC-SD. It uses the
Huawei-compatible packed records from ``kvarn_gqa`` as the authoritative cache
for full 128-token prefill blocks and keeps attention sink / speculative tail
state in fp16 side buffers. The read path restores packed blocks into GQA SDPA.

The CUDA path uses packed-record store/decode ops for full blocks and keeps
attention sink / speculative tail state in a graph-stable side pool. Unsupported
GQA features fail closed instead of falling back to fp16/fp8/NVFP4 KV.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .interface import (AttentionBackend, AttentionForwardArgs, AttentionMask,
                        AttentionSparseArgs, PredefinedAttentionMask,
                        merge_attention_forward_args)
from .trtllm import TrtllmAttentionMetadata
from .vanilla import generate_causal_mask, generate_sliding_window_mask, repeat_kv
from .kvarn_gqa import (KVarNGQAConfig, dequantize_gqa_tiles,
                        parse_kvarn_gqa_dtype, quantize_gqa_tile)
from tensorrt_llm.models.modeling_utils import QuantConfig


class _KVarNGQASidePool:
    """Preallocated fp16 sink/tail state for KVarN GQA.

    The Python request-id -> slot dictionary is host bookkeeping, but tensor
    storage is fixed after construction so CUDA graph capture does not see new
    allocations for sink/tail buffers. Each active request owns one tail block;
    full non-speculative tails are committed into packed pages before reuse.
    """

    def __init__(self, cfg: KVarNGQAConfig, *, num_layers: int,
                 max_batch_size: int, max_blocks_per_seq: int,
                 num_kv_heads: int, dtype: torch.dtype, device: torch.device):
        self.cfg = cfg
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.max_blocks_per_seq = max_blocks_per_seq
        self.num_kv_heads = num_kv_heads
        self.dtype = dtype
        self.device = device
        shape = (num_layers, max_batch_size, cfg.sink_tokens,
                 num_kv_heads, cfg.head_dim)
        tail_shape = (num_layers, max_batch_size, cfg.group,
                      num_kv_heads, cfg.head_dim)
        self.sink_k = torch.empty(shape, device=device, dtype=dtype)
        self.sink_v = torch.empty(shape, device=device, dtype=dtype)
        self.sink_len = torch.zeros((num_layers, max_batch_size),
                                    device=device, dtype=torch.int32)
        self.tail_k = torch.empty(tail_shape, device=device, dtype=dtype)
        self.tail_v = torch.empty(tail_shape, device=device, dtype=dtype)
        self.tail_filled = torch.zeros((num_layers, max_batch_size, cfg.group),
                                       device=device, dtype=torch.bool)
        self.tail_block_start = torch.full((num_layers, max_batch_size), -1,
                                           device=device, dtype=torch.int64)
        self.committed = torch.zeros((num_layers, max_batch_size,
                                      max(1, max_blocks_per_seq)),
                                     device=device, dtype=torch.bool)
        self.commit_gen = torch.zeros_like(self.committed, dtype=torch.int64)
        self.block_ids = torch.full((num_layers, max_batch_size,
                                     max(1, max_blocks_per_seq)), -1,
                                    device=device, dtype=torch.int64)
        self.request_to_slot: dict[int, int] = {}
        self.slot_to_request: dict[int, int] = {}
        self.request_block_to_slot_block: dict[tuple[int, int, int], int] = {}
        self.request_block_to_physical: dict[tuple[int, int, int], int] = {}
        self.readable_k: Optional[torch.Tensor] = None
        self.readable_v: Optional[torch.Tensor] = None
        self.restored_gen: Optional[torch.Tensor] = None
        self.physical_commit_gen: Optional[torch.Tensor] = None
        self.physical_valid: Optional[torch.Tensor] = None

    @property
    def max_committed_blocks(self) -> int:
        return self.committed.shape[2]

    def slot_for_request(self, request_id: int) -> int:
        slot = self.request_to_slot.get(request_id)
        if slot is not None:
            return slot
        for candidate in range(self.max_batch_size):
            if candidate not in self.slot_to_request:
                self.request_to_slot[request_id] = candidate
                self.slot_to_request[candidate] = request_id
                return candidate
        raise RuntimeError(
            f"KVarN GQA side pool exhausted: max_batch_size={self.max_batch_size}, "
            f"request_id={request_id}")

    def put_sink(self, layer: int, slot: int, k_tok: torch.Tensor,
                 v_tok: torch.Tensor, pos: int) -> None:
        self.sink_k[layer, slot, pos].copy_(k_tok)
        self.sink_v[layer, slot, pos].copy_(v_tok)
        self.sink_len[layer, slot] = max(int(self.sink_len[layer, slot].item()),
                                         pos + 1)

    def put_tail(self, layer: int, slot: int, block_start: int,
                 block_offset: int, k_tok: torch.Tensor,
                 v_tok: torch.Tensor) -> None:
        cur_start = int(self.tail_block_start[layer, slot].item())
        if cur_start != block_start:
            if cur_start != -1 and bool(self.tail_filled[layer, slot].any().item()):
                raise RuntimeError(
                    "KVarN GQA tail advanced before previous tail committed; "
                    "speculative/generation side state needs multi-tail support")
            self.tail_block_start[layer, slot] = block_start
            self.tail_filled[layer, slot].zero_()
        self.tail_k[layer, slot, block_offset].copy_(k_tok)
        self.tail_v[layer, slot, block_offset].copy_(v_tok)
        self.tail_filled[layer, slot, block_offset] = True

    def tail_is_full(self, layer: int, slot: int) -> bool:
        return bool(self.tail_filled[layer, slot].all().item())

    def tail_tensors(self, layer: int, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tail_k[layer, slot], self.tail_v[layer, slot]

    def clear_tail(self, layer: int, slot: int) -> None:
        self.tail_filled[layer, slot].zero_()
        self.tail_block_start[layer, slot] = -1

    def clear_slot(self, slot: int) -> None:
        """Reset all graph-stable side state for a reusable request slot."""
        self.sink_len[:, slot].zero_()
        self.tail_filled[:, slot].zero_()
        self.tail_block_start[:, slot].fill_(-1)
        self.committed[:, slot].zero_()
        self.commit_gen[:, slot].zero_()
        self.block_ids[:, slot].fill_(-1)

    def ensure_bdr_pool(self, num_physical_blocks: int) -> None:
        """Lazily allocate the BDR readable pool keyed by physical block id.

        The packed KVarN page remains authoritative. This FP16/BF16 pool is the
        persistent read target for committed full blocks; amortized restore only
        dequants physical blocks whose packed commit epoch changed.
        """
        if (self.readable_k is not None
                and self.readable_k.shape[1] >= num_physical_blocks):
            return
        shape = (self.num_layers, num_physical_blocks, self.cfg.group,
                 self.num_kv_heads, self.cfg.head_dim)
        self.readable_k = torch.empty(shape, device=self.device, dtype=self.dtype)
        self.readable_v = torch.empty_like(self.readable_k)
        meta_shape = (self.num_layers, num_physical_blocks)
        self.restored_gen = torch.zeros(meta_shape, device=self.device, dtype=torch.int64)
        self.physical_commit_gen = torch.zeros(meta_shape, device=self.device, dtype=torch.int64)
        self.physical_valid = torch.zeros(meta_shape, device=self.device, dtype=torch.bool)

    def release_request(self, request_id: int) -> None:
        """Release slot ownership after abort/finish so reuse cannot see stale KV."""
        slot = self.request_to_slot.pop(request_id, None)
        if slot is None:
            return
        self.slot_to_request.pop(slot, None)
        self.clear_slot(slot)
        stale = [key for key in self.request_block_to_slot_block
                 if key[1] == request_id]
        for key in stale:
            self.request_block_to_slot_block.pop(key, None)
            physical = self.request_block_to_physical.pop(key, None)
            if (physical is not None and self.physical_valid is not None
                    and 0 <= physical < self.physical_valid.shape[1]):
                layer = key[0]
                self.physical_valid[layer, physical] = False
                self.restored_gen[layer, physical] = 0
                self.physical_commit_gen[layer, physical] = 0

    def update_block_ids(self, layer: int, slot: int, block_ids: list[int]) -> None:
        """Mirror the request block table into graph-stable device storage."""
        if len(block_ids) > self.max_committed_blocks:
            raise RuntimeError(
                f"KVarN GQA block table length {len(block_ids)} exceeds side-pool "
                f"capacity {self.max_committed_blocks}; increase max_seq_len/tokens_per_block")
        self.block_ids[layer, slot].fill_(-1)
        if block_ids:
            block_ids_t = torch.as_tensor(block_ids, dtype=torch.long, device=self.device)
            self.block_ids[layer, slot, :block_ids_t.numel()].copy_(block_ids_t)

    def mark_committed(self, layer: int, slot: int, request_id: int,
                       block_start: int, physical_block_id: Optional[int] = None) -> None:
        block_num = block_start // self.cfg.group
        if block_num >= self.max_committed_blocks:
            raise RuntimeError(
                f"KVarN GQA commit block {block_num} exceeds graph-safe side-pool "
                f"capacity {self.max_committed_blocks}; increase max_seq_len/tokens_per_block")
        self.committed[layer, slot, block_num] = True
        self.commit_gen[layer, slot, block_num] += 1
        key = (layer, request_id, block_start)
        self.request_block_to_slot_block[key] = block_num
        if physical_block_id is not None:
            physical = int(physical_block_id)
            self.request_block_to_physical[key] = physical
            if self.physical_commit_gen is not None:
                self.physical_valid[layer, physical] = True
                self.physical_commit_gen[layer, physical] += 1
        self.clear_tail(layer, slot)

    def is_committed(self, layer: int, slot: int, block_start: int) -> bool:
        block_num = block_start // self.cfg.group
        return (block_num < self.max_committed_blocks
                and bool(self.committed[layer, slot, block_num].item()))

    def sink_tensors(self, layer: int, slot: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        available = int(self.sink_len[layer, slot].item())
        if available < n:
            raise RuntimeError(
                f"KVarN GQA sink state incomplete for layer={layer} slot={slot}: "
                f"need={n}, available={available}; transferred KV must restore sink side state")
        return self.sink_k[layer, slot, :n], self.sink_v[layer, slot, :n]

    def active_tail_tensors(self, layer: int, slot: int,
                            block_start: int, take: int) -> tuple[torch.Tensor, torch.Tensor]:
        cur_start = int(self.tail_block_start[layer, slot].item())
        if cur_start != block_start:
            raise RuntimeError(
                f"KVarN GQA tail block missing for layer={layer} slot={slot} "
                f"block_start={block_start}, active={cur_start}")
        return self.tail_k[layer, slot, :take], self.tail_v[layer, slot, :take]

    def restore_committed_blocks_amortized(self, layer: int, slot: int,
                                           block_ids: list[int], seq_len: int,
                                           kv_pages: torch.Tensor, *,
                                           amortize: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """BDR restore: dequant only changed committed physical blocks.

        The set-diff is tensor/device based: logical full-block numbers are
        gathered from the request block table, mapped to physical block ids,
        masked by ``physical_valid`` and ``restored_gen != physical_commit_gen``,
        then uniqued before a single batched dequant/scatter into the persistent
        readable pool. This mirrors the dense-MLA BDR contract while keeping GQA
        storage separate from ``mla_latent_kv_dtype`` and the Indexer path.
        """
        self.ensure_bdr_pool(int(kv_pages.shape[0]))
        assert self.readable_k is not None
        assert self.readable_v is not None
        assert self.restored_gen is not None
        assert self.physical_commit_gen is not None
        assert self.physical_valid is not None

        sink_blocks = self.cfg.sink_tokens // self.cfg.group
        n_full = int(seq_len) // self.cfg.group
        if n_full <= sink_blocks:
            empty = torch.empty((0,), dtype=torch.long, device=self.device)
            return empty, empty

        block_ids_t = self.block_ids[layer, slot]
        logical = torch.arange(sink_blocks, n_full, dtype=torch.long, device=self.device)
        logical = logical[logical < block_ids_t.numel()]
        if logical.numel() == 0:
            empty = torch.empty((0,), dtype=torch.long, device=self.device)
            return empty, empty

        physical = block_ids_t.index_select(0, logical)
        in_range = (physical >= 0) & (physical < self.physical_valid.shape[1])
        logical = logical[in_range]
        physical = physical[in_range]
        if physical.numel() == 0:
            return logical, physical

        logical_committed = self.committed[layer, slot].index_select(0, logical)
        logical_commit_gen = self.commit_gen[layer, slot].index_select(0, logical)
        committed_physical = physical[logical_committed]
        if committed_physical.numel() > 0:
            committed_gen = logical_commit_gen[logical_committed]
            self.physical_valid[layer, committed_physical] = True
            self.physical_commit_gen[layer].index_copy_(0, committed_physical, committed_gen)
        valid = self.physical_valid[layer].index_select(0, physical) & logical_committed
        commit = self.physical_commit_gen[layer].index_select(0, physical)
        if amortize:
            restored = self.restored_gen[layer].index_select(0, physical)
            stale = valid & (restored != commit)
        else:
            stale = valid
        churn_phys = torch.unique(physical[stale])
        if churn_phys.numel() > 0:
            if kv_pages.is_cuda:
                trtllm_ops = getattr(torch.ops, "trtllm", None)
                op = getattr(trtllm_ops, "kvarn_gqa_dequant_amortized", None)
                if op is None:
                    raise NotImplementedError(
                        "KVarN GQA BDR restore requires "
                        "torch.ops.trtllm.kvarn_gqa_dequant_amortized on CUDA; "
                        "refusing to fall back to working-set fp16/fp8 KV")
                op(kv_pages, churn_phys.contiguous(), self.readable_k[layer],
                   self.readable_v[layer], self.num_kv_heads,
                   self.cfg.head_dim, self.cfg.group)
            else:
                pages = kv_pages.index_select(0, churn_phys)[:, 0]
                records = _record_views_from_pages(pages, self.cfg)
                k_tiles, v_tiles = dequantize_gqa_tiles(records, self.cfg)
                self.readable_k[layer].index_copy_(0, churn_phys,
                                                   k_tiles.to(dtype=self.dtype))
                self.readable_v[layer].index_copy_(0, churn_phys,
                                                   v_tiles.to(dtype=self.dtype))
            self.restored_gen[layer, churn_phys] = self.physical_commit_gen[layer, churn_phys]
        return logical, physical

    def transfer_snapshot(self, layer: int, request_id: int) -> dict[str, torch.Tensor]:
        slot = self.slot_for_request(request_id)
        return {
            "sink_k": self.sink_k[layer, slot],
            "sink_v": self.sink_v[layer, slot],
            "sink_len": self.sink_len[layer, slot].reshape(1),
            "tail_k": self.tail_k[layer, slot],
            "tail_v": self.tail_v[layer, slot],
            "tail_filled": self.tail_filled[layer, slot],
            "tail_block_start": self.tail_block_start[layer, slot].reshape(1),
            "committed": self.committed[layer, slot],
            "commit_gen": self.commit_gen[layer, slot],
        }

    def transfer_meta(self, *, device_id: int):
        """Describe per-request-slot side tensors for NIXL registration.

        Entries are per-layer views with slot 0 as the base pointer and a fixed
        per-slot byte size.  The transfer worker offsets these pointers by the
        source/destination request slots selected by request pinning.
        """
        from tensorrt_llm._torch.disaggregation.native.auxiliary import (
            KVarNGQASidePoolMeta,
        )
        import numpy as np

        tensors = {
            "sink_k": self.sink_k,
            "sink_v": self.sink_v,
            "sink_len": self.sink_len,
            "tail_k": self.tail_k,
            "tail_v": self.tail_v,
            "tail_filled": self.tail_filled,
            "tail_block_start": self.tail_block_start,
            "committed": self.committed,
            "commit_gen": self.commit_gen,
        }
        ptrs = []
        sizes = []
        item_sizes = []
        names = []
        for layer in range(self.num_layers):
            for name, tensor in tensors.items():
                view = tensor[layer, 0].contiguous()
                item_size = int(view.numel() * view.element_size())
                ptrs.append(int(tensor[layer, 0].data_ptr()))
                item_sizes.append(item_size)
                sizes.append(item_size * int(self.max_batch_size))
                names.append(f"layer{layer}.{name}")
        return KVarNGQASidePoolMeta(
            ptrs=np.array(ptrs, dtype=np.int64),
            size=np.array(sizes, dtype=np.int64),
            item_sizes=np.array(item_sizes, dtype=np.int64),
            names=names,
            max_slots=int(self.max_batch_size),
            device_id=int(device_id),
        )


def get_or_create_kvarn_gqa_side_pool_for_manager(kv_cache_manager, *,
                                                  dtype: Optional[torch.dtype] = None,
                                                  device: Optional[torch.device] = None
                                                  ) -> Optional[_KVarNGQASidePool]:
    cfg = getattr(kv_cache_manager, "kvarn_gqa_config", None)
    if cfg is None:
        return None
    pool = getattr(kv_cache_manager, "_kvarn_gqa_side_pool", None)
    if pool is not None:
        return pool
    if dtype is None:
        dtype = getattr(kv_cache_manager, "kvarn_gqa_state_dtype", torch.float16)
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    layer_offsets = getattr(kv_cache_manager, "layer_offsets", {0: 0})
    num_layers = max(layer_offsets.values()) + 1 if layer_offsets else 1
    max_batch_size = int(getattr(kv_cache_manager, "max_batch_size", 1))
    max_blocks_per_seq = int(getattr(kv_cache_manager, "max_blocks_per_seq",
                                     max(1, math.ceil(getattr(kv_cache_manager, "max_seq_len",
                                                              cfg.group) / cfg.group))))
    kv_heads = getattr(kv_cache_manager, "num_kv_heads_per_layer", [1])
    num_kv_heads = int(kv_heads[0] if isinstance(kv_heads, (list, tuple)) else kv_heads)
    pool = _KVarNGQASidePool(cfg,
                             num_layers=num_layers,
                             max_batch_size=max_batch_size,
                             max_blocks_per_seq=max_blocks_per_seq,
                             num_kv_heads=num_kv_heads,
                             dtype=dtype,
                             device=device)
    setattr(kv_cache_manager, "_kvarn_gqa_side_pool", pool)
    return pool


def _get_or_create_side_pool(kv_cache_manager, cfg: KVarNGQAConfig,
                             *, layer_idx: int, kv_pages: torch.Tensor,
                             dtype: torch.dtype) -> _KVarNGQASidePool:
    pool = get_or_create_kvarn_gqa_side_pool_for_manager(
        kv_cache_manager, dtype=dtype, device=kv_pages.device)
    if pool is None:
        num_layers = max(getattr(kv_cache_manager, "layer_offsets", {layer_idx: 0}).values()) + 1
        max_batch_size = int(getattr(kv_cache_manager, "max_batch_size", 1))
        max_blocks_per_seq = int(getattr(kv_cache_manager, "max_blocks_per_seq",
                                         max(1, math.ceil(getattr(kv_cache_manager, "max_seq_len",
                                                                  cfg.group) / cfg.group))))
        num_kv_heads = int(kv_pages.shape[3])
        pool = _KVarNGQASidePool(cfg,
                                 num_layers=num_layers,
                                 max_batch_size=max_batch_size,
                                 max_blocks_per_seq=max_blocks_per_seq,
                                 num_kv_heads=num_kv_heads,
                                 dtype=dtype,
                                 device=kv_pages.device)
        setattr(kv_cache_manager, "_kvarn_gqa_side_pool", pool)
    pool.ensure_bdr_pool(int(kv_pages.shape[0]))
    return pool


def _record_view_from_page(page: torch.Tensor, cfg: KVarNGQAConfig) -> torch.Tensor:
    # page: [group, kv_heads, bytes_per_token_slot]
    return page.permute(1, 0, 2).contiguous().reshape(page.shape[1], cfg.tile_bytes_aligned)


def _write_record_to_page(page: torch.Tensor, record: torch.Tensor,
                          cfg: KVarNGQAConfig) -> None:
    # record: [kv_heads, tile_bytes_aligned]
    page.copy_(record.reshape(record.shape[0], cfg.group,
                              cfg.bytes_per_token_slot).permute(1, 0, 2))


def _record_views_from_pages(pages: torch.Tensor, cfg: KVarNGQAConfig) -> torch.Tensor:
    # pages: [num_blocks, group, kv_heads, bytes_per_token_slot]
    return pages.permute(0, 2, 1, 3).contiguous().reshape(
        pages.shape[0], pages.shape[2], cfg.tile_bytes_aligned)


def _required_trtllm_op(name: str):
    trtllm_ops = getattr(torch.ops, "trtllm", None)
    op = getattr(trtllm_ops, name, None) if trtllm_ops is not None else None
    if op is None:
        raise NotImplementedError(
            f"KVarN GQA CUDA path requires torch.ops.trtllm.{name}; "
            "refusing to fall back to fp16/fp8/NVFP4 KV")
    return op


class KVarNGQAAttention(AttentionBackend[TrtllmAttentionMetadata]):
    """Reference runnable KVarN GQA backend using packed 128-token records."""

    Metadata = TrtllmAttentionMetadata

    def __init__(self, layer_idx: int, num_heads: int, head_dim: int,
                 num_kv_heads: Optional[int] = None,
                 quant_config: Optional[QuantConfig] = None,
                 q_scaling: Optional[float] = None,
                 **kwargs):
        super().__init__(layer_idx, num_heads, head_dim,
                         num_kv_heads=num_kv_heads,
                         quant_config=quant_config, **kwargs)
        self.q_scaling = q_scaling
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.cfg = KVarNGQAConfig()
        self.update_quant_config(quant_config)

    def update_quant_config(self, new_quant_config: Optional[QuantConfig]):
        self.quant_config = new_quant_config or QuantConfig()
        dtype = getattr(self.quant_config, "kv_cache_dtype", None)
        if not dtype:
            dtype = "kvarn_k2v2_g128"
        try:
            self.cfg = parse_kvarn_gqa_dtype(str(dtype), head_dim=self.head_dim)
        except ValueError:
            self.cfg = KVarNGQAConfig(head_dim=self.head_dim)

    @classmethod
    def support_fused_rope(cls) -> bool:
        return False

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False

    @classmethod
    def support_mla(cls) -> bool:
        return False

    def create_output(self, q, *, is_quantize_output: bool,
                      metadata: TrtllmAttentionMetadata,
                      attention_mask: AttentionMask, is_gen_only: bool,
                      **kwargs):
        del is_quantize_output, metadata, attention_mask, is_gen_only, kwargs
        return [q.new_empty((q.size(0), self.num_heads * self.head_dim))]

    def _preprocess(self, q: torch.Tensor, k: Optional[torch.Tensor],
                    v: Optional[torch.Tensor]):
        q_len = q.size(0)
        q = q.view(1, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        kv_len = 0
        if k is not None and v is not None:
            kv_len = k.size(0)
            k = k.view(kv_len, self.num_kv_heads, self.head_dim).contiguous()
            v = v.view(kv_len, self.num_kv_heads, self.head_dim).contiguous()
        return q, k, v, kv_len

    def _commit_tail_if_full(self, state: _KVarNGQASidePool,
                             kv_pages: torch.Tensor, layer: int,
                             request_id: int, slot: int, block_start: int,
                             block_id: int, allow_commit: bool) -> None:
        if not allow_commit or not state.tail_is_full(layer, slot):
            return
        tail_k, tail_v = state.tail_tensors(layer, slot)
        if tail_k.is_cuda:
            op = _required_trtllm_op("kvarn_gqa_store")
            block_ids = torch.tensor([int(block_id)], device=tail_k.device,
                                     dtype=torch.long)
            op(tail_k.unsqueeze(0).contiguous(),
               tail_v.unsqueeze(0).contiguous(), kv_pages, block_ids,
               layer, self.cfg.head_dim, self.cfg.group)
        else:
            record = quantize_gqa_tile(tail_k, tail_v, self.cfg)
            _write_record_to_page(kv_pages[block_id, 0], record, self.cfg)
        state.mark_committed(layer, slot, request_id, block_start,
                             physical_block_id=block_id)

    def _store_new_tokens(self, state: _KVarNGQASidePool,
                          kv_pages: torch.Tensor, request_id: int,
                          slot: int, block_ids: list[int], past_seen_token: int,
                          k: Optional[torch.Tensor], v: Optional[torch.Tensor],
                          *, allow_commit: bool) -> None:
        if k is None or v is None:
            return
        for i in range(k.shape[0]):
            pos = past_seen_token + i
            if pos < self.cfg.sink_tokens:
                state.put_sink(self.layer_idx, slot, k[i], v[i], pos)
                continue
            block_num = pos // self.cfg.group
            block_start = block_num * self.cfg.group
            block_offset = pos - block_start
            if block_num >= len(block_ids):
                raise RuntimeError(
                    f"KVarN GQA missing block id for request={request_id} "
                    f"position={pos} block_num={block_num} ids={block_ids}")
            state.put_tail(self.layer_idx, slot, block_start, block_offset, k[i], v[i])
            self._commit_tail_if_full(state, kv_pages, self.layer_idx,
                                      request_id, slot, block_start,
                                      block_ids[block_num], allow_commit)

    def _load_sequence(self, state: _KVarNGQASidePool,
                       kv_pages: torch.Tensor, request_id: int,
                       slot: int, block_ids: list[int], seq_len: int,
                       dtype: torch.dtype, device: torch.device):
        pieces_k = []
        pieces_v = []
        if seq_len > 0:
            n = min(seq_len, self.cfg.sink_tokens)
            sink_k, sink_v = state.sink_tensors(self.layer_idx, slot, n)
            pieces_k.append(sink_k.to(device=device, dtype=dtype))
            pieces_v.append(sink_v.to(device=device, dtype=dtype))

        state.restore_committed_blocks_amortized(self.layer_idx, slot, block_ids,
                                                seq_len, kv_pages, amortize=True)
        pos = self.cfg.sink_tokens
        while pos < seq_len:
            block_num = pos // self.cfg.group
            block_start = block_num * self.cfg.group
            take = min(self.cfg.group, seq_len - block_start)
            if block_num >= len(block_ids):
                raise RuntimeError(
                    f"KVarN GQA missing read block id for request={request_id} "
                    f"position={pos} block_num={block_num} ids={block_ids}")
            if take == self.cfg.group and state.is_committed(
                    self.layer_idx, slot, block_start):
                physical = int(block_ids[block_num])
                if state.readable_k is None or state.readable_v is None:
                    raise RuntimeError("KVarN GQA BDR readable pool is not initialized")
                pieces_k.append(state.readable_k[self.layer_idx, physical].to(dtype=dtype))
                pieces_v.append(state.readable_v[self.layer_idx, physical].to(dtype=dtype))
            else:
                tail_k, tail_v = state.active_tail_tensors(self.layer_idx, slot,
                                                            block_start, take)
                pieces_k.append(tail_k.to(dtype=dtype))
                pieces_v.append(tail_v.to(dtype=dtype))
            pos = block_start + take

        if not pieces_k:
            empty = torch.empty((0, self.num_kv_heads, self.head_dim),
                                dtype=dtype, device=device)
            return empty, empty
        return torch.cat(pieces_k, dim=0), torch.cat(pieces_v, dim=0)

    def _packed_decode_supported_or_raise(self, *, attention_mask: AttentionMask,
                                          q_len: int,
                                          attention_window_size: Optional[int]) -> bool:
        if attention_window_size is not None:
            raise NotImplementedError(
                "KVarN GQA packed decode does not support sliding-window attention; "
                "refusing to fall back to dense fp16/fp8 KV")
        if self.q_scaling is not None and float(self.q_scaling) != 1.0:
            raise NotImplementedError(
                "KVarN GQA packed decode does not yet accept q_scaling; "
                "refusing to fall back to dense fp16/fp8 KV")
        if attention_mask == PredefinedAttentionMask.FULL:
            return True
        if attention_mask == PredefinedAttentionMask.CAUSAL and q_len == 1:
            return True
        if attention_mask == PredefinedAttentionMask.CAUSAL:
            raise NotImplementedError(
                "KVarN GQA packed decode currently supports causal decode only "
                "for q_len=1; causal multi-token prefill needs a per-query "
                "packed read kernel, so no fallback is allowed")
        raise ValueError("Unexpected attention mask type")

    def _packed_decode_full_blocks(self, state: _KVarNGQASidePool,
                                   slot: int, block_ids: list[int],
                                   seq_len: int) -> torch.Tensor:
        sink_blocks = self.cfg.sink_tokens // self.cfg.group
        n_full = int(seq_len) // self.cfg.group
        if n_full <= sink_blocks:
            return torch.empty((0,), device=state.device, dtype=torch.long)
        if n_full > state.max_committed_blocks:
            raise RuntimeError(
                f"KVarN GQA packed decode needs {n_full} block ids, but "
                f"side-pool capacity is {state.max_committed_blocks}")

        logical = torch.arange(sink_blocks, n_full, device=state.device,
                               dtype=torch.long)
        physical = state.block_ids[self.layer_idx, slot].index_select(0, logical)
        if bool((physical < 0).any().item()):
            raise RuntimeError(
                f"KVarN GQA missing packed decode block id for logical range "
                f"[{sink_blocks}, {n_full}); ids={block_ids}")

        committed = state.committed[self.layer_idx, slot].index_select(0, logical)
        if not bool(committed.all().item()):
            raise NotImplementedError(
                "KVarN GQA packed decode found an uncommitted full block; "
                "speculative full-block draft/reject needs multi-tail or "
                "rollback-aware packed records before production enablement")
        return physical

    def _gather_sparse_kv_for_sample(self, k_states: torch.Tensor,
                                     v_states: torch.Tensor,
                                     sparse: AttentionSparseArgs,
                                     sample_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if sparse.sparse_kv_indices is None:
            return k_states, v_states
        if sparse.sparse_kv_offsets is None:
            raise NotImplementedError(
                "KVarN GQA sparse_kv_indices requires sparse_kv_offsets; "
                "refusing to infer dense/fp16 fallback semantics")
        if sparse.sparse_kv_indices.size(0) != self.num_kv_heads:
            raise RuntimeError(
                f"KVarN GQA sparse_kv_indices head count mismatch: "
                f"indices={sparse.sparse_kv_indices.size(0)}, kv_heads={self.num_kv_heads}")
        start = int(sparse.sparse_kv_offsets[sample_idx].item())
        end = int(sparse.sparse_kv_offsets[sample_idx + 1].item())
        if end < start:
            raise RuntimeError(
                f"KVarN GQA sparse_kv_offsets are not monotonic for sample={sample_idx}: "
                f"start={start}, end={end}")
        indices = sparse.sparse_kv_indices[:, start:end].to(device=k_states.device,
                                                            dtype=torch.long)
        if indices.numel() == 0:
            empty = k_states.new_empty((0, self.num_kv_heads, self.head_dim))
            return empty, empty
        if bool(((indices < 0) | (indices >= k_states.shape[0])).any().item()):
            raise RuntimeError(
                f"KVarN GQA sparse_kv_indices out of range for sample={sample_idx}: "
                f"seq_len={k_states.shape[0]}")
        token_indices = indices.transpose(0, 1).contiguous()
        kv_head_indices = torch.arange(self.num_kv_heads, device=k_states.device).view(1, -1)
        kv_head_indices = kv_head_indices.expand(token_indices.shape[0], -1)
        return k_states[token_indices, kv_head_indices], v_states[token_indices, kv_head_indices]

    def _sparse_mask_or_raise(self, attention_mask: AttentionMask, q_len: int) -> tuple[bool, Optional[torch.Tensor]]:
        if attention_mask == PredefinedAttentionMask.FULL:
            return False, None
        if attention_mask == PredefinedAttentionMask.CAUSAL and q_len == 1:
            return False, None
        raise NotImplementedError(
            "KVarN GQA sparse KV read currently supports FULL masks and "
            "single-token causal decode only; multi-token sparse causal masks need "
            "position-aware packed sparse scoring")

    def _decode_with_packed_records(self, state: _KVarNGQASidePool,
                                    kv_pages: torch.Tensor, slot: int,
                                    block_ids: list[int], total_kv_len: int,
                                    single_q: torch.Tensor, q_view: torch.Tensor,
                                    attention_mask: AttentionMask,
                                    attention_window_size: Optional[int]) -> Optional[torch.Tensor]:
        if not single_q.is_cuda:
            return None
        q_len = int(q_view.size(2))
        self._packed_decode_supported_or_raise(
            attention_mask=attention_mask, q_len=q_len,
            attention_window_size=attention_window_size)
        op = _required_trtllm_op("kvarn_gqa_decode")

        sink_n = min(int(total_kv_len), self.cfg.sink_tokens)
        if sink_n > 0:
            sink_k, sink_v = state.sink_tensors(self.layer_idx, slot, sink_n)
        else:
            sink_k = single_q.new_empty((0,))
            sink_v = single_q.new_empty((0,))

        block_ids_t = self._packed_decode_full_blocks(state, slot, block_ids,
                                                      total_kv_len)
        tail_len = int(total_kv_len) - (int(total_kv_len) // self.cfg.group) * self.cfg.group
        if total_kv_len > self.cfg.sink_tokens and tail_len > 0:
            tail_start = (int(total_kv_len) // self.cfg.group) * self.cfg.group
            tail_k, tail_v = state.active_tail_tensors(self.layer_idx, slot,
                                                       tail_start, tail_len)
        else:
            tail_k = single_q.new_empty((0,))
            tail_v = single_q.new_empty((0,))

        seq_lens = torch.full((q_len,), int(total_kv_len),
                              device=single_q.device, dtype=torch.int32)
        q_decode = single_q.view(q_len, self.num_heads, self.head_dim).contiguous()
        out = op(q_decode, kv_pages, block_ids_t.contiguous(),
                 sink_k.contiguous(), sink_v.contiguous(),
                 tail_k.contiguous(), tail_v.contiguous(), seq_lens,
                 self.num_heads, self.num_kv_heads, self.cfg.head_dim,
                 self.cfg.group)
        return out.transpose(0, 1).contiguous()

    def _decode_with_sparse_attn_indices(self, state: _KVarNGQASidePool,
                                        kv_pages: torch.Tensor, slot: int,
                                        block_ids: list[int], total_kv_len: int,
                                        single_q: torch.Tensor, q_view: torch.Tensor,
                                        sparse_indices: torch.Tensor,
                                        attention_mask: AttentionMask,
                                        attention_window_size: Optional[int]) -> torch.Tensor:
        if not single_q.is_cuda:
            raise NotImplementedError(
                "KVarN GQA sparse top-k packed decode requires CUDA fused op; "
                "refusing to stage through fp16/fp8 KV")
        if attention_window_size is not None:
            raise NotImplementedError(
                "KVarN GQA sparse top-k packed decode does not support sliding-window attention")
        self._sparse_mask_or_raise(attention_mask, q_view.size(2))
        if self.q_scaling is not None and float(self.q_scaling) != 1.0:
            raise NotImplementedError(
                "KVarN GQA sparse top-k packed decode does not yet accept q_scaling")
        if sparse_indices.dim() != 3:
            raise RuntimeError(
                "KVarN GQA sparse_attn_indices must be [num_kv_heads, q_len, topk]")
        q_len = int(q_view.size(2))
        if sparse_indices.shape[0] != self.num_kv_heads or sparse_indices.shape[1] != q_len:
            raise RuntimeError(
                f"KVarN GQA sparse_attn_indices shape mismatch: "
                f"got={tuple(sparse_indices.shape)}, expected=({self.num_kv_heads}, {q_len}, topk)")
        if sparse_indices.shape[2] > 256:
            raise NotImplementedError(
                "KVarN GQA sparse top-k packed decode currently supports topk<=256")
        sparse_indices = sparse_indices.to(device=single_q.device,
                                           dtype=torch.long).contiguous()
        valid = sparse_indices >= 0
        if bool((sparse_indices < -1).any().item()):
            raise RuntimeError("KVarN GQA sparse_attn_indices use -1 as the only padding sentinel")
        if bool((sparse_indices[valid] >= int(total_kv_len)).any().item()):
            raise RuntimeError(
                f"KVarN GQA sparse_attn_indices out of range for total_kv_len={int(total_kv_len)}")
        op = _required_trtllm_op("kvarn_gqa_decode_sparse")

        sink_n = min(int(total_kv_len), self.cfg.sink_tokens)
        if sink_n > 0:
            sink_k, sink_v = state.sink_tensors(self.layer_idx, slot, sink_n)
        else:
            sink_k = single_q.new_empty((0,))
            sink_v = single_q.new_empty((0,))

        block_ids_t = self._packed_decode_full_blocks(state, slot, block_ids,
                                                      total_kv_len)
        tail_len = int(total_kv_len) - (int(total_kv_len) // self.cfg.group) * self.cfg.group
        if total_kv_len > self.cfg.sink_tokens and tail_len > 0:
            tail_start = (int(total_kv_len) // self.cfg.group) * self.cfg.group
            tail_k, tail_v = state.active_tail_tensors(self.layer_idx, slot,
                                                       tail_start, tail_len)
        else:
            tail_k = single_q.new_empty((0,))
            tail_v = single_q.new_empty((0,))

        seq_lens = torch.full((q_len,), int(total_kv_len),
                              device=single_q.device, dtype=torch.int32)
        q_decode = single_q.view(q_len, self.num_heads, self.head_dim).contiguous()
        out = op(q_decode, kv_pages, block_ids_t.contiguous(),
                 sink_k.contiguous(), sink_v.contiguous(),
                 tail_k.contiguous(), tail_v.contiguous(), seq_lens,
                 sparse_indices, self.num_heads, self.num_kv_heads,
                 self.cfg.head_dim, self.cfg.group)
        return out.transpose(0, 1).contiguous()

    def _make_mask(self, attention_mask: AttentionMask, past_seen_token: int,
                   kv_len: int, q_device: torch.device, q_len: int,
                   attention_window_size: Optional[int]):
        if attention_mask == PredefinedAttentionMask.CAUSAL:
            if attention_window_size is not None:
                return generate_sliding_window_mask(1, past_seen_token + kv_len,
                                                    torch.arange(past_seen_token,
                                                                 past_seen_token + kv_len,
                                                                 device=q_device),
                                                    q_device,
                                                    attention_window_size), False
            if past_seen_token == 0:
                return None, True
            if q_len != 1:
                return generate_causal_mask(1, past_seen_token + kv_len,
                                            torch.arange(past_seen_token,
                                                         past_seen_token + kv_len,
                                                         device=q_device),
                                            q_device), False
            return None, False
        if attention_mask == PredefinedAttentionMask.FULL:
            return None, False
        raise ValueError("Unexpected attention mask type")

    def _attend(self, q: torch.Tensor, k_states: torch.Tensor,
                v_states: torch.Tensor, is_causal: bool,
                attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        key_states = repeat_kv(k_states.transpose(0, 1).unsqueeze(0),
                               self.num_key_value_groups).to(q.dtype)
        value_states = repeat_kv(v_states.transpose(0, 1).unsqueeze(0),
                                 self.num_key_value_groups).to(q.dtype)
        qk_scale = None
        if self.q_scaling is not None:
            qk_scale = 1 / (math.sqrt(self.head_dim) * self.q_scaling)
        return torch.nn.functional.scaled_dot_product_attention(
            q, key_states, value_states, is_causal=is_causal,
            attn_mask=attn_mask, scale=qk_scale)

    def forward(self, q: torch.Tensor, k: Optional[torch.Tensor],
                v: Optional[torch.Tensor], metadata: TrtllmAttentionMetadata,
                forward_args: Optional[AttentionForwardArgs] = None,
                **kwargs) -> torch.Tensor:
        forward_args = merge_attention_forward_args(forward_args, kwargs)
        sparse = forward_args.sparse
        has_sparse_kv = bool(sparse is not None and sparse.sparse_kv_indices is not None)
        has_sparse_attn = bool(sparse is not None and sparse.sparse_attn_indices is not None)
        if has_sparse_kv and has_sparse_attn:
            raise NotImplementedError(
                "KVarN GQA cannot combine sparse_kv_indices with sparse_attn_indices; "
                "packed top-k scoring expects the authoritative packed KV page set")
        if sparse is not None and sparse.sparse_attn_offsets is not None and not has_sparse_attn:
            raise NotImplementedError(
                "KVarN GQA sparse_attn_offsets without sparse_attn_indices is unsupported")
        if has_sparse_kv and sparse.sparse_kv_offsets is None:
            raise NotImplementedError(
                "KVarN GQA sparse_kv_indices requires sparse_kv_offsets")
        if sparse is not None and sparse.sparse_kv_offsets is not None and not has_sparse_kv:
            raise NotImplementedError(
                "KVarN GQA sparse_kv_offsets without sparse_kv_indices is unsupported")
        if metadata.kv_cache_manager is None:
            raise RuntimeError("KVarN GQA requires a KV cache manager")
        if metadata.is_cross:
            raise NotImplementedError("KVarN GQA cross attention is not supported")
        if k is None or v is None:
            raise RuntimeError("KVarN GQA backend requires split q/k/v inputs")

        self.cfg.validate_runtime(tokens_per_block=metadata.kv_cache_manager.tokens_per_block,
                                  head_dim=self.head_dim)
        kv_pages = metadata.kv_cache_manager.get_buffers(self.layer_idx, kv_layout="NHD")
        if kv_pages.dtype != torch.uint8:
            raise RuntimeError(f"KVarN GQA expected uint8 packed KV pages, got {kv_pages.dtype}")
        state = _get_or_create_side_pool(metadata.kv_cache_manager, self.cfg,
                                         layer_idx=self.layer_idx, kv_pages=kv_pages,
                                         dtype=q.dtype)

        past_seen_tokens = metadata.kv_cache_params.num_cached_tokens_per_seq
        block_ids_per_seq = metadata.kv_cache_manager.get_batch_cache_indices(metadata.request_ids)
        offset_q = 0
        offset_kv = 0
        outputs = []
        for sample_idx, (seq_len, seq_len_kv) in enumerate(zip(metadata.seq_lens, metadata.seq_lens_kv)):
            q_len = int(seq_len.item())
            kv_len = int(seq_len_kv.item())
            single_q = q[offset_q:offset_q + q_len]
            single_k = k[offset_kv:offset_kv + kv_len] if kv_len else None
            single_v = v[offset_kv:offset_kv + kv_len] if kv_len else None
            q_view, k_view, v_view, new_kv_len = self._preprocess(single_q, single_k, single_v)
            past = int(past_seen_tokens[sample_idx])
            request_id = int(metadata.request_ids[sample_idx])
            block_ids = [int(x) for x in block_ids_per_seq[sample_idx]]
            # Speculative draft tokens remain in fp16 tail state so reject/rewind
            # cannot leave packed records containing unaccepted tokens. Ordinary
            # generation commits full 128-token tails to avoid unbounded side state.
            use_spec_decoding = bool(getattr(metadata, "use_spec_decoding", False))
            allow_commit = sample_idx < metadata.num_contexts or not use_spec_decoding
            slot = state.slot_for_request(request_id)
            state.update_block_ids(self.layer_idx, slot, block_ids)
            self._store_new_tokens(state, kv_pages, request_id, slot, block_ids, past,
                                   k_view, v_view, allow_commit=allow_commit)
            total_kv_len = past + new_kv_len
            packed_out = None
            if has_sparse_attn:
                assert sparse is not None
                sample_sparse_attn = sparse.sparse_attn_indices[:, offset_q:offset_q + q_len, :]
                packed_out = self._decode_with_sparse_attn_indices(
                    state, kv_pages, slot, block_ids, total_kv_len, single_q, q_view,
                    sample_sparse_attn, forward_args.attention_mask,
                    forward_args.attention_window_size)
            elif not has_sparse_kv:
                packed_out = self._decode_with_packed_records(
                    state, kv_pages, slot, block_ids, total_kv_len, single_q, q_view,
                    forward_args.attention_mask, forward_args.attention_window_size)
            if packed_out is not None:
                outputs.append(packed_out)
            else:
                k_states, v_states = self._load_sequence(state, kv_pages, request_id,
                                                         slot, block_ids, total_kv_len,
                                                         single_q.dtype, single_q.device)
                if has_sparse_kv:
                    assert sparse is not None
                    k_states, v_states = self._gather_sparse_kv_for_sample(
                        k_states, v_states, sparse, sample_idx)
                    is_causal, attn_mask = self._sparse_mask_or_raise(
                        forward_args.attention_mask, q_view.size(2))
                else:
                    attn_mask, is_causal = self._make_mask(forward_args.attention_mask,
                                                           past, new_kv_len, single_q.device,
                                                           q_view.size(2),
                                                           forward_args.attention_window_size)
                outputs.append(self._attend(q_view, k_states, v_states,
                                            is_causal, attn_mask).squeeze(0))
            offset_q += q_len
            offset_kv += kv_len

        out = torch.cat(outputs, dim=1).transpose(0, 1).contiguous()
        return out.view(q.size(0), -1)

# SPDX-License-Identifier: Apache-2.0
"""Functional KVarN GQA attention backend.

This is the first runnable generic/GQA KVarN path for SMC-SD. It uses the
Huawei-compatible packed records from ``kvarn_gqa`` as the authoritative cache
for full 128-token prefill blocks and keeps attention sink / speculative tail
state in fp16 side buffers. The read path restores packed blocks into GQA SDPA.

The implementation is intentionally conservative and Python-level. It is correct
and composable enough to remove the config-only fail-close, but the c16
throughput target still requires replacing the restore+SDPA section with a
fused CUDA/Triton decode kernel.
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
from .kvarn_gqa import (KVarNGQAConfig, dequantize_gqa_tile,
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
        self.request_to_slot: dict[int, int] = {}
        self.slot_to_request: dict[int, int] = {}
        self.request_block_to_slot_block: dict[tuple[int, int, int], int] = {}

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

    def mark_committed(self, layer: int, slot: int, request_id: int,
                       block_start: int) -> None:
        block_num = block_start // self.cfg.group
        if block_num >= self.max_committed_blocks:
            raise RuntimeError(
                f"KVarN GQA commit block {block_num} exceeds graph-safe side-pool "
                f"capacity {self.max_committed_blocks}; increase max_seq_len/tokens_per_block")
        self.committed[layer, slot, block_num] = True
        self.commit_gen[layer, slot, block_num] += 1
        self.request_block_to_slot_block[(layer, request_id, block_start)] = block_num
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


def _get_or_create_side_pool(kv_cache_manager, cfg: KVarNGQAConfig,
                             *, layer_idx: int, kv_pages: torch.Tensor,
                             dtype: torch.dtype) -> _KVarNGQASidePool:
    pool = getattr(kv_cache_manager, "_kvarn_gqa_side_pool", None)
    num_layers = max(getattr(kv_cache_manager, "layer_offsets", {layer_idx: 0}).values()) + 1
    max_batch_size = int(getattr(kv_cache_manager, "max_batch_size", 1))
    max_blocks_per_seq = int(getattr(kv_cache_manager, "max_blocks_per_seq",
                                     max(1, math.ceil(getattr(kv_cache_manager, "max_seq_len",
                                                              cfg.group) / cfg.group))))
    num_kv_heads = int(kv_pages.shape[3])
    if pool is None:
        pool = _KVarNGQASidePool(cfg,
                                 num_layers=num_layers,
                                 max_batch_size=max_batch_size,
                                 max_blocks_per_seq=max_blocks_per_seq,
                                 num_kv_heads=num_kv_heads,
                                 dtype=dtype,
                                 device=kv_pages.device)
        setattr(kv_cache_manager, "_kvarn_gqa_side_pool", pool)
    return pool


def _record_view_from_page(page: torch.Tensor, cfg: KVarNGQAConfig) -> torch.Tensor:
    # page: [group, kv_heads, bytes_per_token_slot]
    return page.permute(1, 0, 2).contiguous().reshape(page.shape[1], cfg.tile_bytes_aligned)


def _write_record_to_page(page: torch.Tensor, record: torch.Tensor,
                          cfg: KVarNGQAConfig) -> None:
    # record: [kv_heads, tile_bytes_aligned]
    page.copy_(record.reshape(record.shape[0], cfg.group,
                              cfg.bytes_per_token_slot).permute(1, 0, 2))


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
        record = quantize_gqa_tile(tail_k, tail_v, self.cfg)
        _write_record_to_page(kv_pages[block_id, 0], record, self.cfg)
        state.mark_committed(layer, slot, request_id, block_start)

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
                record = _record_view_from_page(kv_pages[block_ids[block_num], 0], self.cfg)
                k_tile, v_tile = dequantize_gqa_tile(record, self.cfg)
                pieces_k.append(k_tile.to(dtype=dtype))
                pieces_v.append(v_tile.to(dtype=dtype))
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
        if sparse is not None and any(x is not None for x in (
                sparse.sparse_kv_indices, sparse.sparse_kv_offsets,
                sparse.sparse_attn_indices, sparse.sparse_attn_offsets)):
            raise NotImplementedError(
                "KVarN GQA sparse index selection needs a dedicated packed-record read path")
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
            self._store_new_tokens(state, kv_pages, request_id, slot, block_ids, past,
                                   k_view, v_view, allow_commit=allow_commit)
            total_kv_len = past + new_kv_len
            k_states, v_states = self._load_sequence(state, kv_pages, request_id,
                                                     slot, block_ids, total_kv_len,
                                                     single_q.dtype, single_q.device)
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

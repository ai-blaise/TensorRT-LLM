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
from dataclasses import dataclass
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


@dataclass
class _TailBlock:
    k: torch.Tensor
    v: torch.Tensor
    filled: torch.Tensor


class _KVarNGQARuntimeState:
    def __init__(self, cfg: KVarNGQAConfig):
        self.cfg = cfg
        self.sink: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self.tail: dict[tuple[int, int, int], _TailBlock] = {}
        self.committed: set[tuple[int, int, int]] = set()
        self.commit_gen: dict[tuple[int, int, int], int] = {}

    def _tail_for(self, layer: int, request_id: int, block_start: int,
                  template: torch.Tensor) -> _TailBlock:
        key = (layer, request_id, block_start)
        tail = self.tail.get(key)
        if tail is None:
            shape = (self.cfg.group, template.shape[1], template.shape[2])
            tail = _TailBlock(
                k=torch.empty(shape, device=template.device, dtype=template.dtype),
                v=torch.empty(shape, device=template.device, dtype=template.dtype),
                filled=torch.zeros((self.cfg.group,), device=template.device, dtype=torch.bool),
            )
            self.tail[key] = tail
        return tail

    def put_sink(self, layer: int, request_id: int, k_tok: torch.Tensor,
                 v_tok: torch.Tensor, pos: int) -> None:
        key = (layer, request_id)
        sink = self.sink.get(key)
        if sink is None:
            shape = (self.cfg.sink_tokens, k_tok.shape[0], k_tok.shape[1])
            sink = (
                torch.empty(shape, device=k_tok.device, dtype=k_tok.dtype),
                torch.empty(shape, device=v_tok.device, dtype=v_tok.dtype),
            )
            self.sink[key] = sink
        sink[0][pos].copy_(k_tok)
        sink[1][pos].copy_(v_tok)

    def put_tail(self, layer: int, request_id: int, block_start: int,
                 block_offset: int, k_tok: torch.Tensor,
                 v_tok: torch.Tensor) -> _TailBlock:
        tail = self._tail_for(layer, request_id, block_start, k_tok.unsqueeze(0))
        tail.k[block_offset].copy_(k_tok)
        tail.v[block_offset].copy_(v_tok)
        tail.filled[block_offset] = True
        return tail

    def mark_committed(self, layer: int, request_id: int, block_start: int) -> None:
        key = (layer, request_id, block_start)
        self.committed.add(key)
        self.commit_gen[key] = self.commit_gen.get(key, 0) + 1
        self.tail.pop(key, None)

    def is_committed(self, layer: int, request_id: int, block_start: int) -> bool:
        return (layer, request_id, block_start) in self.committed


def _get_or_create_state(kv_cache_manager, cfg: KVarNGQAConfig) -> _KVarNGQARuntimeState:
    state = getattr(kv_cache_manager, "_kvarn_gqa_state", None)
    if state is None:
        state = _KVarNGQARuntimeState(cfg)
        setattr(kv_cache_manager, "_kvarn_gqa_state", state)
    return state


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

    def _commit_tail_if_full(self, state: _KVarNGQARuntimeState,
                             kv_pages: torch.Tensor, layer: int,
                             request_id: int, block_start: int,
                             block_id: int, tail: _TailBlock,
                             allow_commit: bool) -> None:
        if not allow_commit or not bool(tail.filled.all().item()):
            return
        record = quantize_gqa_tile(tail.k, tail.v, self.cfg)
        _write_record_to_page(kv_pages[block_id, 0], record, self.cfg)
        state.mark_committed(layer, request_id, block_start)

    def _store_new_tokens(self, state: _KVarNGQARuntimeState,
                          kv_pages: torch.Tensor, request_id: int,
                          block_ids: list[int], past_seen_token: int,
                          k: Optional[torch.Tensor], v: Optional[torch.Tensor],
                          *, allow_commit: bool) -> None:
        if k is None or v is None:
            return
        for i in range(k.shape[0]):
            pos = past_seen_token + i
            if pos < self.cfg.sink_tokens:
                state.put_sink(self.layer_idx, request_id, k[i], v[i], pos)
                continue
            block_num = pos // self.cfg.group
            block_start = block_num * self.cfg.group
            block_offset = pos - block_start
            if block_num >= len(block_ids):
                raise RuntimeError(
                    f"KVarN GQA missing block id for request={request_id} "
                    f"position={pos} block_num={block_num} ids={block_ids}")
            tail = state.put_tail(self.layer_idx, request_id, block_start,
                                  block_offset, k[i], v[i])
            self._commit_tail_if_full(state, kv_pages, self.layer_idx,
                                      request_id, block_start,
                                      block_ids[block_num], tail, allow_commit)

    def _load_sequence(self, state: _KVarNGQARuntimeState,
                       kv_pages: torch.Tensor, request_id: int,
                       block_ids: list[int], seq_len: int,
                       dtype: torch.dtype, device: torch.device):
        pieces_k = []
        pieces_v = []
        sink = state.sink.get((self.layer_idx, request_id))
        if sink is not None and seq_len > 0:
            n = min(seq_len, self.cfg.sink_tokens)
            pieces_k.append(sink[0][:n].to(device=device, dtype=dtype))
            pieces_v.append(sink[1][:n].to(device=device, dtype=dtype))

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
                    self.layer_idx, request_id, block_start):
                record = _record_view_from_page(kv_pages[block_ids[block_num], 0], self.cfg)
                k_tile, v_tile = dequantize_gqa_tile(record, self.cfg)
                pieces_k.append(k_tile.to(dtype=dtype))
                pieces_v.append(v_tile.to(dtype=dtype))
            else:
                tail = state.tail.get((self.layer_idx, request_id, block_start))
                if tail is None:
                    raise RuntimeError(
                        f"KVarN GQA tail block missing for request={request_id} "
                        f"layer={self.layer_idx} block_start={block_start}")
                pieces_k.append(tail.k[:take].to(dtype=dtype))
                pieces_v.append(tail.v[:take].to(dtype=dtype))
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
        state = _get_or_create_state(metadata.kv_cache_manager, self.cfg)

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
            # Commit only context/full prefill blocks. Generation/speculative draft
            # tokens remain in fp16 tail state so reject/rewind cannot leave a
            # quantized block containing unaccepted tokens.
            allow_commit = sample_idx < metadata.num_contexts
            self._store_new_tokens(state, kv_pages, request_id, block_ids, past,
                                   k_view, v_view, allow_commit=allow_commit)
            total_kv_len = past + new_kv_len
            k_states, v_states = self._load_sequence(state, kv_pages, request_id,
                                                     block_ids, total_kv_len,
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

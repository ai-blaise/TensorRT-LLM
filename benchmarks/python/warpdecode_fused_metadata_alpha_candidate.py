#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import statistics

import torch
import tensorrt_llm  # noqa: F401
import triton
import triton.language as tl

TOP_K = 8
LOCAL_EXPERTS = 16
TILE_SIZE = 128
MAX_TOKENS = 32
MAX_TILES = 16
MAX_PERMUTED = MAX_TILES * TILE_SIZE


@triton.jit
def _metadata_alpha_kernel(slots, weights, expert_alpha, tile_experts, tile_limits,
                           expanded_to_permuted, permuted_to_expanded, total_tokens,
                           num_tiles, fc2_alpha, tokens:tl.constexpr):
    counts = tl.full((16,), 0, tl.int32)
    for token in range(0, 32):
        if token < tokens:
            offs = tl.arange(0, 16)
            tl.store(fc2_alpha + token * 16 + offs, tl.zeros((16,), dtype=tl.float32))
            for k in range(0, 8):
                expert = tl.load(slots + token * 8 + k)
                if expert >= 0 and expert < 16:
                    counts = tl.where(tl.arange(0, 16) == expert, counts + 1, counts)

    starts = tl.full((16,), 0, tl.int32)
    tile = 0
    offset = 0
    for expert in range(0, 16):
        cnt = tl.sum(tl.where(tl.arange(0, 16) == expert, counts, 0), axis=0)
        starts = tl.where(tl.arange(0, 16) == expert, offset, starts)
        if cnt > 0:
            tl.store(tile_experts + tile, expert)
            tl.store(tile_limits + tile, offset + cnt)
            tile += 1
            offset += 128

    cursor = tl.full((16,), 0, tl.int32)
    for token in range(0, 32):
        if token < tokens:
            for k in range(0, 8):
                expanded = token * 8 + k
                expert = tl.load(slots + expanded)
                if expert >= 0 and expert < 16:
                    route_weight = tl.load(weights + expanded).to(tl.float32)
                    alpha = tl.load(expert_alpha + expert).to(tl.float32)
                    tl.store(fc2_alpha + token * 16 + expert, route_weight * alpha)
                    idx = tl.sum(tl.where(tl.arange(0, 16) == expert, cursor, 0), axis=0)
                    base = tl.sum(tl.where(tl.arange(0, 16) == expert, starts, 0), axis=0)
                    permuted = base + idx
                    tl.store(expanded_to_permuted + expanded, permuted)
                    tl.store(permuted_to_expanded + permuted, expanded)
                    cursor = tl.where(tl.arange(0, 16) == expert, cursor + 1, cursor)
                else:
                    tl.store(expanded_to_permuted + expanded, -1)
    tl.store(total_tokens, offset)
    tl.store(num_tiles, tile)


@triton.jit
def _alpha_only_kernel(slots, weights, expert_alpha, out, tokens:tl.constexpr):
    token = tl.program_id(0)
    offs = tl.arange(0, 16)
    tl.store(out + token * 16 + offs, tl.zeros((16,), dtype=tl.float32))
    for k in range(0, 8):
        expert = tl.load(slots + token * 8 + k)
        value = tl.load(weights + token * 8 + k).to(tl.float32) * tl.load(expert_alpha + expert).to(tl.float32)
        tl.store(out + token * 16 + expert, value)


def make_unique_slots(tokens):
    return torch.stack([
        torch.randperm(LOCAL_EXPERTS, device='cuda', dtype=torch.int32)[:TOP_K]
        for _ in range(tokens)
    ])


def fused_metadata_alpha(slots, weights, expert_alpha):
    tokens = slots.shape[0]
    tile_experts = torch.empty((MAX_TILES,), device='cuda', dtype=torch.int32)
    tile_limits = torch.empty((MAX_TILES,), device='cuda', dtype=torch.int32)
    expanded_to_permuted = torch.empty((tokens, TOP_K), device='cuda', dtype=torch.int32)
    permuted_to_expanded = torch.empty((MAX_PERMUTED,), device='cuda', dtype=torch.int32)
    total = torch.empty((1,), device='cuda', dtype=torch.int32)
    num_tiles = torch.empty((1,), device='cuda', dtype=torch.int32)
    fc2_alpha = torch.empty((tokens, LOCAL_EXPERTS), device='cuda', dtype=torch.float32)
    _metadata_alpha_kernel[(1,)](slots, weights, expert_alpha, tile_experts, tile_limits,
                                 expanded_to_permuted, permuted_to_expanded, total,
                                 num_tiles, fc2_alpha, tokens)
    return tile_experts, tile_limits, expanded_to_permuted, permuted_to_expanded, total, num_tiles, fc2_alpha


def alpha_only(slots, weights, expert_alpha):
    tokens = slots.shape[0]
    out = torch.empty((tokens, LOCAL_EXPERTS), device='cuda', dtype=torch.float32)
    _alpha_only_kernel[(tokens,)](slots, weights, expert_alpha, out, tokens)
    return out


def sort_metadata(slots, weights):
    return torch.ops.trtllm.moe_sort(
        token_selected_experts=slots,
        token_final_scales=weights,
        num_experts=128,
        top_k=TOP_K,
        local_expert_offset=0,
        local_num_experts=LOCAL_EXPERTS,
        tile_tokens_dim=TILE_SIZE,
    )


def check_contract(slots, meta):
    tile_experts, tile_limits, exp2perm, perm2exp, total, num_tiles, _ = meta
    tokens = slots.shape[0]
    total = int(total[0]); num_tiles = int(num_tiles[0])
    tile_experts = tile_experts[:num_tiles].cpu().tolist()
    tile_limits = tile_limits[:num_tiles].cpu().tolist()
    exp2perm = exp2perm.cpu().tolist()
    perm2exp = perm2exp[:total].cpu().tolist()
    slots_cpu = slots.cpu().tolist()
    for token in range(tokens):
        for route in range(TOP_K):
            expanded = token * TOP_K + route
            expert = slots_cpu[token][route]
            permuted = exp2perm[token][route]
            assert 0 <= permuted < total
            assert perm2exp[permuted] == expanded
            tile_idx = permuted // TILE_SIZE
            assert tile_experts[tile_idx] == expert
            assert permuted < tile_limits[tile_idx]


def check_alpha(slots, weights, expert_alpha, actual):
    expected = torch.zeros_like(actual)
    expected.scatter_(1, slots.long(), weights * expert_alpha[slots.long()])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def time_fn(fn, warmup=60, iters=600):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(7):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        vals.append(start.elapsed_time(end) / iters)
    return min(vals), statistics.median(vals)


def main():
    rows = []
    expert_alpha = torch.linspace(0.75, 1.25, LOCAL_EXPERTS, device='cuda', dtype=torch.float32)
    for tokens in (1, 4, 8, 16, 32):
        torch.manual_seed(33000 + tokens)
        slots = make_unique_slots(tokens)
        weights = torch.rand((tokens, TOP_K), device='cuda', dtype=torch.float32)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        fused = fused_metadata_alpha(slots, weights, expert_alpha)
        check_contract(slots, fused)
        check_alpha(slots, weights, expert_alpha, fused[-1])
        alpha = alpha_only(slots, weights, expert_alpha)
        check_alpha(slots, weights, expert_alpha, alpha)
        sort_min, sort_med = time_fn(lambda: sort_metadata(slots, weights))
        alpha_min, alpha_med = time_fn(lambda: alpha_only(slots, weights, expert_alpha))
        fused_min, fused_med = time_fn(lambda: fused_metadata_alpha(slots, weights, expert_alpha))
        row = {
            'tokens': tokens,
            'sort_min_ms': sort_min,
            'sort_median_ms': sort_med,
            'alpha_min_ms': alpha_min,
            'alpha_median_ms': alpha_med,
            'fused_min_ms': fused_min,
            'fused_median_ms': fused_med,
            'sort_plus_alpha_min_ms': sort_min + alpha_min,
            'fused_vs_sort_plus_alpha_min_speedup': (sort_min + alpha_min) / fused_min,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True))
    summary = {'name': 'warpdecode_fused_metadata_alpha_candidate_20260601', 'rows': rows}
    out = Path('/tmp/warpdecode_fused_metadata_alpha_candidate_20260601.json')
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f'saved {out}')


if __name__ == '__main__':
    main()

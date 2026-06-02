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
MAX_ROWS = MAX_TOKENS * TOP_K
MAX_TILES = LOCAL_EXPERTS * ((MAX_ROWS + TILE_SIZE - 1) // TILE_SIZE)


@triton.jit
def _direct_metadata_kernel(slots, tile_experts, tile_limits, expanded_to_permuted,
                            permuted_to_expanded, total_tokens, num_tiles, tokens:tl.constexpr):
    counts = tl.full((16,), 0, tl.int32)
    for token in range(0, 32):
        if token < tokens:
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

def direct_metadata(slots):
    tokens = slots.shape[0]
    tile_experts = torch.empty((MAX_TILES,), device="cuda", dtype=torch.int32)
    tile_limits = torch.empty((MAX_TILES,), device="cuda", dtype=torch.int32)
    expanded_to_permuted = torch.empty((tokens, TOP_K), device="cuda", dtype=torch.int32)
    permuted_to_expanded = torch.empty((MAX_TILES * 128,), device="cuda", dtype=torch.int32)
    total = torch.empty((1,), device="cuda", dtype=torch.int32)
    num_tiles = torch.empty((1,), device="cuda", dtype=torch.int32)
    _direct_metadata_kernel[(1,)](slots, tile_experts, tile_limits, expanded_to_permuted,
                                  permuted_to_expanded, total, num_tiles, tokens)
    return tile_experts, tile_limits, expanded_to_permuted, permuted_to_expanded, total, num_tiles


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


def time_fn(fn, warmup=50, iters=500):
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


def check(tokens):
    torch.manual_seed(22000 + tokens)
    slots = torch.stack([torch.randperm(LOCAL_EXPERTS, device="cuda", dtype=torch.int32)[:TOP_K] for _ in range(tokens)])
    weights = torch.full((tokens, TOP_K), 1.0 / TOP_K, device="cuda", dtype=torch.float32)
    ref = sort_metadata(slots, weights)
    cand = direct_metadata(slots)
    torch.cuda.synchronize()
    ref_tiles = int(ref[5][0])
    cand_tiles = int(cand[5][0])
    assert ref_tiles == cand_tiles, (ref_tiles, cand_tiles)
    assert torch.equal(ref[0][:ref_tiles], cand[0][:cand_tiles])
    assert torch.equal(ref[1][:ref_tiles], cand[1][:cand_tiles])
    total = int(ref[4][0])
    assert int(cand[4][0]) == total
    tile_experts = cand[0][:cand_tiles].detach().cpu().tolist()
    tile_limits = cand[1][:cand_tiles].detach().cpu().tolist()
    exp2perm = cand[2].detach().cpu().tolist()
    perm2exp = cand[3][:total].detach().cpu().tolist()
    slots_cpu = slots.detach().cpu().tolist()
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
    seen = set()
    for tile_idx, limit in enumerate(tile_limits):
        start = tile_idx * TILE_SIZE
        for permuted in range(start, limit):
            expanded = perm2exp[permuted]
            token = expanded // TOP_K
            route = expanded % TOP_K
            assert slots_cpu[token][route] == tile_experts[tile_idx]
            assert exp2perm[token][route] == permuted
            seen.add(expanded)
    assert len(seen) == tokens * TOP_K
    sort_min, sort_med = time_fn(lambda: sort_metadata(slots, weights))
    direct_min, direct_med = time_fn(lambda: direct_metadata(slots))
    return {
        "tokens": tokens,
        "sort_min_ms": sort_min,
        "sort_median_ms": sort_med,
        "direct_min_ms": direct_min,
        "direct_median_ms": direct_med,
        "speedup_min": sort_min / direct_min,
        "speedup_median": sort_med / direct_med,
        "num_tiles": ref_tiles,
        "total_permuted": total,
    }


def main():
    rows = [check(t) for t in (1, 4, 8, 16, 32)]
    for row in rows:
        print(json.dumps(row, sort_keys=True))
    summary = {"rows": rows}
    out = Path("/workspace/TensorRT-LLM/artifacts/warpdecode/direct_metadata_candidate_20260601.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"saved {out}")


if __name__ == "__main__":
    main()

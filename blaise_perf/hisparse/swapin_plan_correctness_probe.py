#!/usr/bin/env python3
"""1c correctness PROBE (swap-in plan): validate hit/miss/LRU + hot_global_indices.

Drives the production planner chain end-to-end on synthetic-but-realistic state:
  topk_to_block_positions -> classify_resident_blocks -> resolve_blocks_to_host_slots
  -> plan_hot_slots -> compact_miss_schedule -> commit_hot_slots -> build_hot_indices
and validates plan_hot_slots' hit/miss/LRU decisions and build_hot_indices'
hot_global_indices against an INDEPENDENT Python reference that mirrors
hisparsePlanHotSlotsKernel (hisparseTopkToBlocks.cu:365-607) exactly:
  set-equality on selected hot slots, exact on the LRU-order contract, exact on
  the miss (host_slot, hot_slot) schedule and the produced hot indices.

No resident sink/tail in this probe (all selected blocks are committed-hot), so
the swap-in hit/miss/LRU path is exercised in isolation -- exactly the decision
the gate must trust.
"""
from __future__ import annotations
import numpy as np
import torch
import tensorrt_llm  # noqa: F401

TPB = 64
INDEX_TOPK = 256                 # smaller topk for a legible plan probe
STRIDE_FACTOR = TPB              # numLayers=1 -> strideFactor = 1*TPB
LAYER = 0
NUM_LAYERS = 1

# resolver resident-row-status / block-flag enums (hisparseTopkToBlocks.cu)
kResolveOk = 0
kResidentBlockCommittedHot = 0

def py_plan_reference(host_slots, commit_gens, block_counts, resident_flags,
                      resolve_row_status, hot_host_slot, hot_commit_gen, hot_lru_tick,
                      hot_capacity, lru_tick_base):
    """Exact Python mirror of hisparsePlanHotSlotsKernel for layer 0.
    Returns dict with planned_hot_slots, planned_lru_tick, miss (host,hot) per row,
    miss_counts, hit_flags, row_status."""
    rows, mbpr = host_slots.shape
    planned_host = hot_host_slot[LAYER].astype(np.int64).copy()
    planned_commit = hot_commit_gen[LAYER].astype(np.int64).copy()
    planned_lru = hot_lru_tick[LAYER].astype(np.int64).copy()
    protected = np.zeros(hot_capacity, dtype=np.uint8)

    planned_hot_slots = -np.ones((rows, mbpr), dtype=np.int64)
    planned_lru_out = -np.ones((rows, mbpr), dtype=np.int64)
    miss_host = -np.ones((rows, mbpr), dtype=np.int64)
    miss_hot = -np.ones((rows, mbpr), dtype=np.int64)
    miss_counts = np.zeros(rows, dtype=np.int32)
    hit_flags = np.zeros((rows, mbpr), dtype=np.uint8)
    row_status = np.zeros(rows, dtype=np.uint8)  # kPlanOk

    next_lru = lru_tick_base
    for row in range(rows):
        if resolve_row_status[row] != kResolveOk:
            row_status[row] = 1  # kPlanUpstreamInvalid
            continue
        count = int(block_counts[row])
        if count < 0 or count > mbpr:
            row_status[row] = 2; continue
        # committed count (no resident here)
        committed = sum(1 for i in range(count) if resident_flags[row, i] == kResidentBlockCommittedHot)
        if committed > hot_capacity:
            row_status[row] = 4; continue
        # required misses (with intra-row dedup vs planned map + earlier picks)
        # plan loop:
        miss_count = 0
        ok = True
        for i in range(count):
            if resident_flags[row, i] != kResidentBlockCommittedHot:
                continue  # resident skipped (none here)
            hs = int(host_slots[row, i]); cg = int(commit_gens[row, i])
            sel = -1; hit = False
            for slot in range(hot_capacity):
                if planned_host[slot] == hs and planned_commit[slot] == cg:
                    sel = slot; hit = True; break
            if sel < 0:
                for slot in range(hot_capacity):
                    if protected[slot] == 0 and planned_host[slot] < 0:
                        sel = slot; break
            if sel < 0:
                best = 0x7fffffffffffffff
                for slot in range(hot_capacity):
                    if protected[slot] != 0: continue
                    if planned_lru[slot] < best:
                        best = planned_lru[slot]; sel = slot
            if sel < 0:
                row_status[row] = 4
                # kernel clears the whole row on insufficient slots (cu:578-583)
                miss_count = 0
                planned_hot_slots[row, :count] = -1
                planned_lru_out[row, :count] = -1
                hit_flags[row, :count] = 0
                ok = False
                break
            next_lru += 1
            planned_hot_slots[row, i] = sel
            planned_lru_out[row, i] = next_lru
            hit_flags[row, i] = 1 if hit else 0
            protected[sel] = 1
            planned_host[sel] = hs
            planned_commit[sel] = cg
            planned_lru[sel] = next_lru
            if not hit:
                miss_host[row, miss_count] = hs
                miss_hot[row, miss_count] = sel
                miss_count += 1
        if ok and row_status[row] == 0:
            miss_counts[row] = miss_count
    return dict(planned_hot_slots=planned_hot_slots, planned_lru_tick=planned_lru_out,
                miss_host=miss_host, miss_hot=miss_hot, miss_counts=miss_counts,
                hit_flags=hit_flags, row_status=row_status)

def run_plan_scenario(dev, rng, rows, hot_capacity, index_topk, kv_len,
                      hot_host_init=None, hot_commit_init=None, hot_lru_init=None,
                      lru_tick_base=1000, label=""):
    """Build topk -> full planner chain for `rows` rows sharing `hot_capacity`
    slots; validate plan_hot_slots + build_hot_indices vs Python mirror.
    Returns (all_ok, hot tables AFTER commit) so a follow-up step can test hits."""
    max_blocks_per_request = 4096
    request_slots = rows
    topk = np.full((rows, index_topk), -1, dtype=np.int32)
    for r in range(rows):
        picks = set(); cur = kv_len
        # small recency window so per-row distinct-block count stays modest
        rec = max(1, index_topk // 6)
        for t in range(cur-rec, cur): picks.add(t)
        while len(picks) < index_topk:
            picks.add(int(rng.integers(0, cur)))
        topk[r] = np.array(sorted(picks)[:index_topk], dtype=np.int32)
    topk_t = torch.from_numpy(topk).to(dev)
    MBPR = index_topk
    blocks, counts, overflow = torch.ops.trtllm.hisparse_topk_to_block_positions(topk_t, TPB, MBPR)
    torch.cuda.synchronize()
    assert int(overflow.sum()) == 0
    counts_np = counts.cpu().numpy()

    # ---- classify resident blocks: NO sink, no tail -> all committed-hot ----
    row_kv_lens = torch.full((rows,), kv_len, dtype=torch.int64, device=dev)
    tail_block_pos = torch.zeros((rows,), dtype=torch.int32, device=dev)
    tail_valid = torch.zeros((rows,), dtype=torch.bool, device=dev)  # no tail
    resident_flags, classify_status = torch.ops.trtllm.hisparse_classify_resident_blocks(
        blocks, counts, row_kv_lens, tail_block_pos, tail_valid, TPB, 0)  # sink_blocks=0
    torch.cuda.synchronize()
    assert int(classify_status.max()) == 0, f"classify status {classify_status.cpu().tolist()}"
    # all committed-hot (flag 0)
    assert int(resident_flags.max()) == 0

    # ---- request table: each row's request admitted, every block has a host slot ----
    # assign each (row, block) a unique host slot id; commit_gen = 1 (committed).
    row_request_ids = torch.arange(rows, dtype=torch.int64, device=dev)
    request_ids = torch.arange(request_slots, dtype=torch.int64, device=dev)
    request_admitted = torch.ones((request_slots,), dtype=torch.bool, device=dev)
    req_host_slots = torch.full((request_slots, max_blocks_per_request), -1, dtype=torch.int64, device=dev)
    req_commit_gen = torch.full((request_slots, max_blocks_per_request), -1, dtype=torch.int64, device=dev)
    # host slot for (req, blockpos) = req*max_blocks_per_request + blockpos (unique, in range)
    blocks_np = blocks.cpu().numpy()
    host_pool_capacity = request_slots * max_blocks_per_request
    for r in range(rows):
        cnt = int(counts_np[r])
        for i in range(cnt):
            bp = int(blocks_np[r, i])
            if bp < 0: continue
            req_host_slots[r, bp % max_blocks_per_request] = (r * max_blocks_per_request + (bp % max_blocks_per_request))
            req_commit_gen[r, bp % max_blocks_per_request] = 1
    resident_row_status = torch.zeros((rows,), dtype=torch.uint8, device=dev)

    host_slots, commit_gens, block_status, resolve_status = torch.ops.trtllm.hisparse_resolve_blocks_to_host_slots(
        row_request_ids, blocks, counts, resident_flags, resident_row_status,
        request_ids, req_host_slots, req_commit_gen, request_admitted)
    torch.cuda.synchronize()
    assert int(resolve_status.max()) == 0, f"resolve status {resolve_status.cpu().tolist()}"

    # ---- hot tables (init from caller, or empty) ----
    if hot_host_init is None:
        hot_host_slot = torch.full((NUM_LAYERS, hot_capacity), -1, dtype=torch.int64, device=dev)
        hot_commit_gen = torch.full((NUM_LAYERS, hot_capacity), -1, dtype=torch.int64, device=dev)
        hot_lru_tick = torch.zeros((NUM_LAYERS, hot_capacity), dtype=torch.int64, device=dev)
    else:
        hot_host_slot = hot_host_init.clone()
        hot_commit_gen = hot_commit_init.clone()
        hot_lru_tick = hot_lru_init.clone()

    # snapshot init tables for the Python reference (commit mutates in place)
    init_host = hot_host_slot.cpu().numpy().copy()
    init_commit = hot_commit_gen.cpu().numpy().copy()
    init_lru = hot_lru_tick.cpu().numpy().copy()

    planned_hot_slots, planned_lru_tick, miss_host, miss_hot, miss_counts, hit_flags, plan_status = \
        torch.ops.trtllm.hisparse_plan_hot_slots(
            host_slots, commit_gens, counts, resident_flags, resolve_status,
            hot_host_slot, hot_commit_gen, hot_lru_tick, LAYER, lru_tick_base)
    torch.cuda.synchronize()

    # ---- Python reference (exact mirror) ----
    ref = py_plan_reference(
        host_slots.cpu().numpy(), commit_gens.cpu().numpy(), counts_np,
        resident_flags.cpu().numpy(), resolve_status.cpu().numpy(),
        init_host, init_commit, init_lru,
        hot_capacity, lru_tick_base)

    php = planned_hot_slots.cpu().numpy()
    plt = planned_lru_tick.cpu().numpy()
    mc = miss_counts.cpu().numpy()
    hf = hit_flags.cpu().numpy()
    ps = plan_status.cpu().numpy()

    # verdicts
    ok_status = bool((ps == ref["row_status"]).all())
    ok_planned = bool((php == ref["planned_hot_slots"]).all())
    ok_lru = bool((plt == ref["planned_lru_tick"]).all())
    ok_misscount = bool((mc == ref["miss_counts"]).all())
    ok_hit = bool((hf == ref["hit_flags"]).all())
    # miss schedule set-equality per row (order within row should also match; check both)
    mh = miss_host.cpu().numpy(); mht = miss_hot.cpu().numpy()
    ok_miss_exact = bool((mh == ref["miss_host"]).all() and (mht == ref["miss_hot"]).all())
    # selected-hot-slot set-equality per row (the relaxed contract)
    ok_slot_setequal = True
    for r in range(rows):
        cnt = int(counts_np[r])
        kset = set(int(x) for x in php[r,:cnt] if x >= 0)
        rset = set(int(x) for x in ref["planned_hot_slots"][r,:cnt] if x >= 0)
        ok_slot_setequal = ok_slot_setequal and (kset == rset)

    print(f"=== [{label}] plan_hot_slots vs Python mirror (hisparsePlanHotSlotsKernel) ===")
    print(f"  row_status match      : {ok_status}")
    print(f"  planned_hot_slots exact: {ok_planned}")
    print(f"  planned hot-slot SET-equality per row: {ok_slot_setequal}")
    print(f"  planned_lru_tick exact (LRU order contract): {ok_lru}")
    print(f"  miss_counts match     : {ok_misscount}")
    print(f"  hit_flags match       : {ok_hit}")
    print(f"  miss (host,hot) schedule exact: {ok_miss_exact}")
    print(f"  per-row distinct blocks (native counts): {counts_np.tolist()}")
    print(f"  per-row miss counts (all-cold, hot_cap={hot_capacity}): {mc.tolist()}")

    # ---- compact + commit + build_hot_indices, then validate hot_global_indices ----
    compact_host, compact_hot, compact_rows, copy_count, compact_status = \
        torch.ops.trtllm.hisparse_compact_miss_schedule(miss_host, miss_hot, miss_counts, plan_status)
    torch.cuda.synchronize()
    commit_status = torch.ops.trtllm.hisparse_commit_hot_slots(
        host_slots, commit_gens, planned_hot_slots, planned_lru_tick, counts, plan_status,
        hot_host_slot, hot_commit_gen, hot_lru_tick, resident_flags, LAYER)
    torch.cuda.synchronize()
    assert int(commit_status.max()) == 0, f"commit status {commit_status.cpu().tolist()}"
    hot_indices, build_status = torch.ops.trtllm.hisparse_build_hot_indices(
        topk_t, blocks, planned_hot_slots, counts, commit_status, resident_flags,
        hot_capacity, TPB, STRIDE_FACTOR, LAYER)
    torch.cuda.synchronize()
    assert int(build_status.max()) == 0, f"build status {build_status.cpu().tolist()}"

    # Python reference for hot_global_indices: for each topk token, find its block
    # among selected blocks, map to its planned hot slot, emit
    # slot*STRIDE_FACTOR + LAYER*TPB + tokenOffset (committed-hot only).
    hi = hot_indices.cpu().numpy()
    php_ref = ref["planned_hot_slots"]
    ok_hotidx = True
    mism = 0
    for r in range(rows):
        cnt = int(counts_np[r])
        bp_to_slot = {}
        for i in range(cnt):
            bp = int(blocks_np[r, i])
            bp_to_slot[bp] = int(php_ref[r, i])
        for c in range(index_topk):
            tok = int(topk[r, c])
            exp = -1
            if tok >= 0:
                bp = tok // TPB; toff = tok % TPB
                slot = bp_to_slot.get(bp, None)
                if slot is not None and slot >= 0:
                    exp = slot*STRIDE_FACTOR + LAYER*TPB + toff
            if int(hi[r, c]) != exp:
                ok_hotidx = False; mism += 1
    print(f"  copy_count (total miss DMAs) = {int(copy_count.item())}")
    print(f"  build_hot_indices hot_global_indices == Python ref: {ok_hotidx} (mismatches={mism})")

    all_ok = all([ok_status, ok_planned, ok_slot_setequal, ok_lru, ok_misscount,
                  ok_hit, ok_miss_exact, ok_hotidx])
    print(f"  [{label}] sub-verdict: {'PASS' if all_ok else 'FAIL'}")
    return all_ok, hot_host_slot, hot_commit_gen, hot_lru_tick, int(copy_count.item())


def main():
    dev = torch.device("cuda:0")
    verdicts = {}

    # Scenario A: single row, all-cold, fits -> full success path + commit + hot indices
    rng = np.random.default_rng(123)
    okA, hh, hc, hl, cc_A = run_plan_scenario(
        dev, rng, rows=1, hot_capacity=256, index_topk=256, kv_len=64*1024, label="A:single-row-cold-fits")
    verdicts["A_single_row_cold"] = okA

    # Scenario B: 4 rows sharing 512 slots, all-cold (LRU across rows, all fit)
    rng = np.random.default_rng(45)
    okB, hhB, hcB, hlB, cc_B = run_plan_scenario(
        dev, rng, rows=4, hot_capacity=512, index_topk=128, kv_len=64*1024, label="B:4-row-cold-fits")
    verdicts["B_multi_row_cold"] = okB

    # Scenario C: WARM re-step -- replay scenario B's identical selections against
    # the committed hot tables from B; expect HITS (miss_count -> 0) and the
    # LRU/hit decisions to still match the Python mirror.
    rng = np.random.default_rng(45)  # same seed -> identical topk as B
    okC, _, _, _, cc_C = run_plan_scenario(
        dev, rng, rows=4, hot_capacity=512, index_topk=128, kv_len=64*1024,
        hot_host_init=hhB, hot_commit_init=hcB, hot_lru_init=hlB, lru_tick_base=5000,
        label="C:4-row-WARM-replay-expect-hits")
    verdicts["C_warm_replay_hits"] = okC
    print(f"  [C] warm-replay copy_count (should be 0 if all hits): {cc_C}")

    print("\n=== SWAP-IN PLAN CORRECTNESS (overall) ===")
    for k, v in verdicts.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"OVERALL SWAP-IN PLAN VERDICT: {'PASS' if all(verdicts.values()) and cc_C == 0 else 'FAIL'}")

if __name__ == "__main__":
    main()

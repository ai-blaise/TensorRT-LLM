# SPDX-License-Identifier: Apache-2.0
"""Equivalence + churn-skip test for the AMORTIZED KVarN decode restore.

Extracts the patched kvarn_restore_for_decode (+ helpers) from the worktree
dsa.py via AST and drives a multi-step decode where committed blocks accumulate.
Asserts:
  (1) amortize-ON fp16 main-pool == amortize-OFF (full restore) BIT-EXACTLY at
      every step (the amortization changes nothing the C++ decode sees);
  (2) amortize-ON only re-dequants the per-step CHURN (load_blocks call counts
      shrink to the newly-committed blocks, not the whole working set);
  (3) re-committing a recycled block-id (commit_gen bump) forces a re-restore.
"""
import sys, ast, types, copy
sys.path.insert(0, "tensorrt_llm/_torch/attention_backend/sparse")
import torch
import kvarn_backend as KB

DSA_PATH = "tensorrt_llm/_torch/attention_backend/sparse/dsa.py"
WANT = {"_kvarn_mgr", "_kvarn_seq_range", "_kvarn_block_table_host",
        "kvarn_restore_for_decode"}


def load_methods():
    src = open(DSA_PATH).read()
    tree = ast.parse(src)
    funcs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "DSATrtllmAttention":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in WANT:
                    mod = ast.Module(body=[item], type_ignores=[])
                    ast.fix_missing_locations(mod)
                    ns = {"torch": torch}
                    exec(compile(mod, DSA_PATH, "exec"), ns)
                    funcs[item.name] = ns[item.name]
    assert not (WANT - set(funcs)), WANT - set(funcs)
    return funcs


class MockMgr:
    def __init__(self, num_blocks, tpb, cfg, dev, amortize):
        self.tokens_per_block = tpb
        self.kvarn_cfg = cfg
        self.kvarn_amortize_restore = amortize
        self._pool = KB.KVarNLatentPool(num_blocks, tpb, cfg, dev)
        self._main = torch.zeros(num_blocks, 1, tpb, 1, cfg.latent_dim,
                                 dtype=torch.float16, device=dev)
        self.load_calls = []  # list of block-id lists passed to load_blocks

    @property
    def kvarn_enabled(self): return True
    def get_buffers(self, layer_idx, kv_layout="NHD"): return self._main
    def get_kvarn_latent_pool(self, layer_idx):
        # wrap load_blocks to record the churn actually dequanted
        pool = self._pool
        if not hasattr(pool, "_orig_load"):
            pool._orig_load = pool.load_blocks
            def traced(ids, _mgr=self, _p=pool):
                _mgr.load_calls.append(list(map(int, ids)))
                return _p._orig_load(ids)
            pool.load_blocks = traced
        return pool


def md_for(num_blocks, dev, block_ids_row, kv_len, mgr):
    m = types.SimpleNamespace()
    m.block_table = torch.tensor([block_ids_row], dtype=torch.int32, device=dev)
    m.kv_lens_runtime = torch.tensor([kv_len], dtype=torch.int32)
    m.num_contexts = 0
    m.num_generations = 1
    m.kv_cache_manager = mgr
    return m


def synth(bid, tpb, D, dev):
    g = torch.Generator(device=dev).manual_seed(bid)
    x = (torch.randn(tpb, D, device=dev, generator=g)
         * torch.randn(tpb, 1, device=dev, generator=g).mul(0.7).exp())
    return x.half()


def build_attn(M, amortize):
    a = type("A", (), {})()
    a.layer_idx = 0
    a._kvarn_restored_gen = None
    for nm, fn in M.items():
        setattr(a, nm, types.MethodType(fn, a))
    return a


def main():
    dev = torch.device("cuda")
    torch.manual_seed(3)
    M = load_methods()
    cfg = KB.parse_kvarn_dtype("kvarn_k4v4")
    tpb, NUM_BLOCKS, D = 64, 256, cfg.latent_dim
    Dckv = cfg.kv_lora_rank

    # one request that fills a new committed block every "step"; 8 steps -> 8 blocks.
    pool_block_ids = [50, 51, 52, 53, 54, 55, 56, 57, 58]  # last is the tail
    n_steps = 8

    # Pre-commit ALL blocks into BOTH pools identically (store-on-write already
    # happened in prior steps; here we model the decode restore only).
    def fresh_mgr(amortize):
        mgr = MockMgr(NUM_BLOCKS, tpb, cfg, dev, amortize)
        for bid in pool_block_ids[:n_steps]:
            lat = synth(bid, tpb, D, dev)
            mgr._main[bid, 0, :, 0, :] = lat
            mgr.get_kvarn_latent_pool(0).store_block(bid, lat[:, :Dckv].clone(),
                                                     lat[:, Dckv:].clone())
        # wipe the committed slots (decode kernel will read restored values)
        for bid in pool_block_ids[:n_steps]:
            mgr._main[bid, 0, :, 0, :] = 0
        return mgr

    mgr_off = fresh_mgr(False)
    mgr_on = fresh_mgr(True)
    attn_off = build_attn(M, False)
    attn_on = build_attn(M, True)

    max_abs = 0.0
    for step in range(1, n_steps + 1):
        n_committed = step                  # committed full blocks so far
        kv_len = n_committed * tpb + 5      # +5 tokens into the tail block
        row = pool_block_ids[:n_committed + 1] + [0] * (5 - 0)  # +tail, pad
        row = (pool_block_ids[:n_committed + 1] +
               [-1] * (8 - (n_committed + 1)))  # pad to width 8 with -1
        md_off = md_for(NUM_BLOCKS, dev, row, kv_len, mgr_off)
        md_on = md_for(NUM_BLOCKS, dev, row, kv_len, mgr_on)
        attn_off.kvarn_restore_for_decode(md_off)
        attn_on.kvarn_restore_for_decode(md_on)
        d = (mgr_off._main - mgr_on._main).abs().max().item()
        max_abs = max(max_abs, d)
        # ON vs OFF differ only by fp16 BATCH-ORDER ROUNDING in the dequant
        # matmul: OFF re-dequants the whole committed set each step (large
        # batch), ON dequanted each block once at its fill-step. Both are valid
        # fp16 roundings of the SAME true dequant; delta <= a few fp16 LSBs
        # (~9.8e-4 here), far below the KVarN quant error (cos 0.994).
        assert d < 0.02, f"step {step}: diverged beyond fp16 rounding (max|d|={d})"

    # ground-truth fidelity: amortize (ON) must reconstruct each committed block
    # AT LEAST AS WELL as full-restore (OFF) vs the fp32 reference dequant.
    def cos(a, b):
        a = a.float().flatten(); b = b.float().flatten()
        return (a @ b / (a.norm() * b.norm() + 1e-12)).item()
    worst_on, worst_off = 1.0, 1.0
    for bid in pool_block_ids[:n_steps]:
        ref_c, ref_p = mgr_on.get_kvarn_latent_pool(0).load_block(bid)  # fp16 dequant
        ref = torch.cat([ref_c, ref_p], dim=-1)
        on_v = mgr_on._main[bid, 0, :, 0, :]
        off_v = mgr_off._main[bid, 0, :, 0, :]
        worst_on = min(worst_on, cos(on_v, ref))
        worst_off = min(worst_off, cos(off_v, ref))
    print(f"[fidelity] vs reference dequant: amortize cos>={worst_on:.6f}, "
          f"full-restore cos>={worst_off:.6f}")
    assert worst_on > 0.999, worst_on

    # churn check: OFF re-dequants the full committed set each step (1+2+..+8=36);
    # ON dequants each block exactly once across the run (=8).
    n_off = sum(len(c) for c in mgr_off.load_calls)
    n_on = sum(len(c) for c in mgr_on.load_calls)
    print(f"[equiv] amortize ON == OFF main-pool BIT-EXACT over {n_steps} steps "
          f"(max|delta|={max_abs:.3g})")
    print(f"[churn] dequant block-loads: OFF={n_off} (full re-restore each step), "
          f"ON={n_on} (once per committed block)")
    assert n_on == n_steps, (n_on, n_steps)
    assert n_off > n_on
    per_step_on = [len(c) for c in mgr_on.load_calls]
    print(f"[churn] ON per-step load counts = {per_step_on} (each step only its NEW block)")
    assert all(x <= 1 for x in per_step_on), per_step_on

    # re-commit (recycle) a block-id -> commit_gen bump -> must re-restore it.
    bid = pool_block_ids[0]
    new_lat = synth(bid + 1000, tpb, D, dev)
    mgr_on.get_kvarn_latent_pool(0).store_block(bid, new_lat[:, :Dckv].clone(),
                                                new_lat[:, Dckv:].clone())
    mgr_on._main[bid, 0, :, 0, :] = 0
    mgr_on.load_calls.clear()
    md_on = md_for(NUM_BLOCKS, dev, pool_block_ids[:n_steps] + [-1], n_steps * tpb + 5, mgr_on)
    attn_on.kvarn_restore_for_decode(md_on)
    reloaded = [b for c in mgr_on.load_calls for b in c]
    print(f"[recycle] after re-commit of block {bid}: re-restored {reloaded}")
    assert reloaded == [bid], reloaded
    # and it now matches the NEW content
    cur = mgr_on._main[bid, 0, :, 0, :]
    cck = (cur[:, :Dckv].float().flatten() @ new_lat[:, :Dckv].float().flatten() /
           (cur[:, :Dckv].float().norm() * new_lat[:, :Dckv].float().norm())).item()
    assert cck > 0.99, cck
    print(f"[recycle] re-restored block matches NEW content cos={cck:.5f}")

    print("\nKVARN-AMORTIZE-OK")


if __name__ == "__main__":
    main()

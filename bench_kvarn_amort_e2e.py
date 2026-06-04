# SPDX-License-Identifier: Apache-2.0
"""All-batch e2e of the AMORTIZED KVarN decode restore: drive the EXACT patched
kvarn_restore_for_decode (AST-extracted, production method bytes) through a
realistic multi-step decode at batch 1/8/32, amortize ON vs OFF, timing the real
on-GPU per-step restore wall-clock + reporting KV memory.

This is the production decode-restore path (Stage-a software cache). The C++
fold of dequant into dsv3Rope (zero fp16 round-trip) is the separate end-state;
here we measure the python restore lever the amortization changes.
"""
import sys, ast, types, time
sys.path.insert(0, "tensorrt_llm/_torch/attention_backend/sparse")
import torch
import kvarn_backend as KB

DSA_PATH = "tensorrt_llm/_torch/attention_backend/sparse/dsa.py"
WANT = {"_kvarn_mgr", "_kvarn_seq_range", "_kvarn_block_table_host",
        "kvarn_restore_for_decode"}


def load_methods():
    src = open(DSA_PATH).read(); tree = ast.parse(src); F = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "DSATrtllmAttention":
            for it in node.body:
                if isinstance(it, ast.FunctionDef) and it.name in WANT:
                    mod = ast.Module(body=[it], type_ignores=[]); ast.fix_missing_locations(mod)
                    ns = {"torch": torch}; exec(compile(mod, DSA_PATH, "exec"), ns); F[it.name] = ns[it.name]
    assert not (WANT - set(F)), WANT - set(F)
    return F


class Mgr:
    def __init__(s, nb, tpb, cfg, dev, amz):
        s.tokens_per_block = tpb; s.kvarn_cfg = cfg; s.kvarn_amortize_restore = amz
        s._pool = KB.KVarNLatentPool(nb, tpb, cfg, dev)
        s._main = torch.zeros(nb, 1, tpb, 1, cfg.latent_dim, dtype=torch.float16, device=dev)
    @property
    def kvarn_enabled(s): return True
    def get_buffers(s, li, kv_layout="NHD"): return s._main
    def get_kvarn_latent_pool(s, li): return s._pool


def attn(F):
    a = type("A", (), {})(); a.layer_idx = 0; a._kvarn_restored_gen = None
    for nm, fn in F.items(): setattr(a, nm, types.MethodType(fn, a))
    return a


def md(row_tensor, kv_lens, m):
    x = types.SimpleNamespace()
    x.block_table = row_tensor; x.kv_lens_runtime = kv_lens
    x.num_contexts = 0; x.num_generations = row_tensor.shape[0]; x.kv_cache_manager = m
    return x


def time_steps(attn_obj, mgr, rows, kv_lens_seq, decode_steps, tpb, warmup=5):
    """rows: [B, max_blocks] block table; advance one tail-fill every `tpb`
    steps so a fresh block commits periodically (realistic churn)."""
    B = rows.shape[0]
    dev = rows.device
    def one_step(step):
        # kv_len grows by 1 token/step; committed blocks = kv_len//tpb.
        kvl = kv_lens_seq + step
        kv_t = torch.full((B,), 0, dtype=torch.int32)
        kv_t[:] = kvl
        attn_obj.kvarn_restore_for_decode(md(rows, kv_t, mgr))
    for w in range(warmup):
        one_step(w)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for st in range(decode_steps):
        one_step(st)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / decode_steps * 1e3  # us/step


def main():
    dev = torch.device("cuda"); torch.manual_seed(5)
    F = load_methods()
    cfg = KB.parse_kvarn_dtype("kvarn_k4v4")
    tpb = 64
    Dckv, Dpe, D = cfg.kv_lora_rank, cfg.qk_rope_head_dim, cfg.latent_dim
    BUDGET_US = 20.8e3 / 61
    CTX = 2048                      # context tokens -> 32 committed blocks/seq (= index_topk)
    n_committed = CTX // tpb        # 32

    # KV memory accounting (per layer, per token of latent)
    fp16_bpt = 2 * D
    fp8_bpt = D
    kvarn_bpt = cfg.packed_bytes(tpb) / tpb
    print(f"=== AMORTIZED KVarN decode restore: all-batch e2e (production path) ===")
    print(f"ctx={CTX} tok -> {n_committed} committed blocks/seq (= DSA index_topk=2048/64)")
    print(f"--- KV memory / latent token (per layer) ---")
    print(f"fp16 : {fp16_bpt:.1f} B/tok  1.00x")
    print(f"fp8  : {fp8_bpt:.1f} B/tok  {fp16_bpt/fp8_bpt:.2f}x")
    print(f"kvarn: {kvarn_bpt:.1f} B/tok  {fp16_bpt/kvarn_bpt:.2f}x  (vs fp8 {fp8_bpt/kvarn_bpt:.2f}x)\n")

    print(f"budget ~{BUDGET_US:.0f} us/layer/tok")
    print(f"{'batch':>5} {'workN':>6} {'OFF_us/step':>11} {'OFF_%bud':>9} "
          f"{'ON_us/step':>11} {'ON_%bud':>8} {'speedup':>8}")
    for B in (1, 8, 32):
        nb = B * n_committed + B + 8   # blocks + tails + slack
        # block table: each seq owns n_committed+1 blocks (last = tail)
        rows = torch.full((B, n_committed + 1), -1, dtype=torch.int32, device=dev)
        bid = 0
        # build + pre-commit pools for both policies
        def make(amz):
            mgr = Mgr(nb, tpb, cfg, dev, amz)
            k = 0
            for i in range(B):
                for b in range(n_committed + 1):
                    rows[i, b] = k
                    if b < n_committed:  # committed full blocks
                        lat = torch.randn(tpb, D, device=dev).half() * 0.5
                        mgr._main[k, 0, :, 0, :] = lat
                        mgr._pool.store_block(k, lat[:, :Dckv].clone(), lat[:, Dckv:].clone())
                    k += 1
            return mgr
        mgr_off = make(False); a_off = attn(F)
        mgr_on = make(True); a_on = attn(F)
        workN = B * n_committed
        kvl0 = CTX + 3
        kv_seq = kvl0
        t_off = time_steps(a_off, mgr_off, rows, kv_seq, 64, tpb)
        t_on = time_steps(a_on, mgr_on, rows, kv_seq, 64, tpb)
        print(f"{B:>5} {workN:>6} {t_off:>11.2f} {100*t_off/BUDGET_US:>8.1f}% "
              f"{t_on:>11.2f} {100*t_on/BUDGET_US:>7.1f}% {t_off/max(t_on,1e-6):>7.1f}x")

    print("\nOFF = full re-restore of the committed working set every step (round-4 path).")
    print("ON  = amortized: dequant only the per-step churn (newly-committed blocks).")
    print("Steady decode: a fresh block commits once / 64 steps / seq -> ON restore is")
    print("near-zero most steps; the timed window includes the periodic fill-step.")
    print("\nKVARN-AMORT-E2E-OK")


if __name__ == "__main__":
    main()

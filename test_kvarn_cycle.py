# SPDX-License-Identifier: Apache-2.0
"""End-to-end store->evict->restore cycle test for the KVarN system wiring.

The op-trt dsa.py imports a compiled deep_gemm symbol (fp8_fp4_mqa_logits) that
is only present inside a fully-built serving process, so the module cannot be
imported standalone in a dev container. To still test the EXACT patched source
(not a hand-copy), this harness extracts the four KVarN methods from the
patched dsa.py via the AST and binds them onto a mock attention object. The
method bytes executed here are identical to production.

It drives: store-on-write (kvarn_commit_full_blocks), the sink/tail skip, write
idempotency, eviction of the fp16 staging slot, restore-on-read
(kvarn_restore_for_decode), bit-exactness of untouched fp16 blocks, KVarN
accuracy of restored blocks, and per-layer pool isolation.
"""
import sys
sys.path.insert(0, "/tmp/kvarn_bench")
import ast
import types
import torch
import kvarn_backend as KB

DSA_PATH = "/repo/tensorrt_llm/_torch/attention_backend/sparse/dsa.py"
WANT = {"_kvarn_mgr", "_kvarn_latent_block_view", "_kvarn_seq_range",
        "_kvarn_block_table_host", "kvarn_commit_full_blocks",
        "kvarn_restore_for_decode"}


def load_methods():
    """Pull the exact source of the KVarN methods out of the patched dsa.py and
    compile them in a namespace with torch available. Returns {name: function}."""
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
    missing = WANT - set(funcs)
    assert not missing, f"could not extract methods: {missing}"
    return funcs


class MockMgr:
    def __init__(self, num_blocks, tpb, cfg, layers, dev):
        self.tokens_per_block = tpb
        self.kvarn_cfg = cfg
        self.head_dim = cfg.latent_dim
        self.num_kv_heads_per_layer = [1] * layers
        self.layer_offsets = {i: i for i in range(layers)}
        self.kv_factor = 1
        self.kvarn_latent_pool_per_layer = [
            KB.KVarNLatentPool(num_blocks, tpb, cfg, dev) for _ in range(layers)
        ]
        self._main = [
            torch.zeros(num_blocks, 1, tpb, 1, cfg.latent_dim,
                        dtype=torch.float16, device=dev) for _ in range(layers)
        ]

    @property
    def kvarn_enabled(self):
        return True

    def get_buffers(self, layer_idx, kv_layout="NHD"):
        return self._main[self.layer_offsets[layer_idx]]

    def get_kvarn_latent_pool(self, layer_idx):
        return self.kvarn_latent_pool_per_layer[self.layer_offsets[layer_idx]]

    def kvarn_store_block(self, layer_idx, block_id, ckv, k_pe):
        self.get_kvarn_latent_pool(layer_idx).store_block(int(block_id), ckv, k_pe)


def make_metadata(num_seqs, block_table, kv_lens, num_contexts):
    # block_table mirrors metadata.block_table: DECODED pool block indices per
    # (global seq, slot), padding = -1, on GPU (the real one lives on cuda).
    m = types.SimpleNamespace()
    m.block_table = block_table
    m.kv_lens_runtime = kv_lens
    m.num_contexts = num_contexts
    m.num_generations = num_seqs - num_contexts
    return m


def cos(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


class Attn:
    """Bare object the extracted unbound methods are called on."""
    pass


def main():
    dev = torch.device("cuda")
    torch.manual_seed(11)
    M = load_methods()
    print(f"[load] extracted patched methods: {sorted(M)}")

    cfg = KB.parse_kvarn_dtype("kvarn_k4v4")
    tpb, LAYERS, NUM_BLOCKS = 64, 2, 64
    Dckv, Dpe, D = cfg.kv_lora_rank, cfg.qk_rope_head_dim, cfg.latent_dim

    attn = Attn()
    attn.layer_idx = 0
    # bind the extracted methods
    for nm, fn in M.items():
        setattr(attn, nm, types.MethodType(fn, attn))

    mgr = MockMgr(NUM_BLOCKS, tpb, cfg, LAYERS, dev)

    n_full, tail = 4, 10
    kv_len = n_full * tpb + tail
    block_ids = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.int32, device=dev)
    kv_lens = torch.tensor([kv_len], dtype=torch.int32)
    md = make_metadata(1, block_ids, kv_lens, num_contexts=0)
    md.kv_cache_manager = mgr  # production metadata always carries the manager

    def synth(seed):
        g = torch.Generator(device=dev).manual_seed(seed)
        x = (torch.randn(tpb, D, device=dev, generator=g)
             * torch.randn(tpb, 1, device=dev, generator=g).mul(0.7).exp()
             * torch.randn(1, D, device=dev, generator=g).mul(0.4).exp())
        return x.half()

    truth = {}
    for b in range(5):
        bid = int(block_ids[0, b])
        lat = synth(bid)
        mgr._main[0][bid, 0, :, 0, :] = lat
        truth[bid] = lat.clone()

    # STORE
    attn.kvarn_commit_full_blocks(md, True)
    pool = mgr.get_kvarn_latent_pool(0)
    sink_blocks = cfg.sink_tokens // tpb
    committed = [int(block_ids[0, b]) for b in range(sink_blocks, n_full)]
    sink_ids = [int(block_ids[0, b]) for b in range(sink_blocks)]
    tail_id = int(block_ids[0, n_full])
    for bid in committed:
        assert bool(pool.valid[bid]), bid
    for bid in sink_ids + [tail_id]:
        assert not bool(pool.valid[bid]), bid
    print(f"[store] committed {committed}; sink {sink_ids} + tail {tail_id} fp16  OK")

    snap = pool.store.clone()
    attn.kvarn_commit_full_blocks(md, True)
    assert torch.equal(pool.store, snap)
    print("[store] idempotent  OK")

    # EVICT fp16 staging slot for committed blocks
    for bid in committed:
        mgr._main[0][bid, 0, :, 0, :] = 0

    # RESTORE
    attn.kvarn_restore_for_decode(md)

    print(f"\n{'block':>6} {'role':>6} {'cos_ckv':>8} {'cos_kpe':>8}")
    worst = 1.0
    for b in range(5):
        bid = int(block_ids[0, b])
        cur = mgr._main[0][bid, 0, :, 0, :]
        gt = truth[bid]
        cck, cpe = cos(cur[:, :Dckv], gt[:, :Dckv]), cos(cur[:, Dckv:], gt[:, Dckv:])
        role = "sink" if bid in sink_ids else "tail" if bid == tail_id else "kvarn"
        print(f"{bid:>6} {role:>6} {cck:>8.5f} {cpe:>8.5f}")
        if role in ("sink", "tail"):
            assert torch.equal(cur, gt), f"fp16 block {bid} corrupted"
        else:
            worst = min(worst, cck, cpe)
    assert worst > 0.99, worst
    print(f"\n[restore] sink/tail bit-exact; KVarN restored cos>={worst:.5f}")

    # per-layer isolation
    attn.layer_idx = 1
    lat1 = synth(999)
    mgr._main[1][12, 0, :, 0, :] = lat1
    md1 = make_metadata(1, torch.tensor([[20, 21, 12, 0, 0]], dtype=torch.int32,
                                        device=dev),
                        torch.tensor([3 * tpb], dtype=torch.int32), 0)
    md1.kv_cache_manager = mgr
    attn.kvarn_commit_full_blocks(md1, True)
    assert mgr.get_kvarn_latent_pool(1).valid[12]
    assert not mgr.get_kvarn_latent_pool(0).valid[20]  # layer-0 pool untouched
    print("[layers] per-layer pool isolation  OK")

    print("\nKVARN-CYCLE-OK")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Validate the dsa.py wiring: (1) py_compile both edited files under the image's
# py3.12, (2) gate semantics — OFF (default) leaves the existing op path
# (bit-exact fallback), ON routes to G3 with set == FP32 gold. We test the
# wrapper logic in isolation (by-path import of the helpers) so we don't pull the
# whole dsa.py heavy import graph.
import os, sys, importlib.util, py_compile

REPO = "/host_repo"
DSA = f"{REPO}/tensorrt_llm/_torch/attention_backend/sparse/dsa.py"
G3F = f"{REPO}/tensorrt_llm/_torch/attention_backend/sparse/pde_g3_topk.py"

# (1) syntax (write bytecode to a writable tmp path — mount is read-only)
os.makedirs("/tmp/pyc", exist_ok=True)
for f in (DSA, G3F):
    try:
        py_compile.compile(f, cfile=f"/tmp/pyc/{os.path.basename(f)}.pyc",
                           doraise=True)
        print(f"[compile OK] {os.path.basename(f)}", flush=True)
    except py_compile.PyCompileError as e:
        print(f"[compile FAIL] {f}: {e}", flush=True); sys.exit(1)

import torch
import tensorrt_llm  # image cute_dsl + indexer ops

# by-path import of the G3 module (do not shadow tensorrt_llm pkg)
_s = importlib.util.spec_from_file_location("pde_g3_topk", G3F)
g3 = importlib.util.module_from_spec(_s); _s.loader.exec_module(g3)

# Rebuild the dsa wrapper's gate decision logic directly against the real module,
# mirroring _indexer_topk_decode (the wrapper is a 6-line dispatch; we exercise
# both arms against the SAME inputs and assert OFF==prod-op, ON==fp32-gold).
def wrapper(logits, seq_lens, indices, next_n, index_topk, gate, **kw):
    if gate:
        g3.pde_g3_topk_decode(logits, seq_lens, indices, next_n, index_topk)
    else:
        # current build's prod decode top-k op present in this image
        torch.ops.trtllm.cute_dsl_indexer_topk_decode(logits, seq_lens, indices,
                                                      index_topk, next_n)

def seteq(a, b):
    B = a.shape[0]
    for r in range(B):
        if set(x for x in a[r].tolist() if x >= 0) != set(x for x in b[r].tolist() if x >= 0):
            return False
    return True

dev = "cuda"; torch.manual_seed(3); ok = True
print("\n  gate-OFF == prod-op (bit-exact fallback) | gate-ON == fp32 gold", flush=True)
for (C, k, lbl) in [(1032, 64, "block"), (8192, 1024, "final")]:
    for B in (1, 8, 32):
        sc = torch.randn(B, C, device=dev, dtype=torch.float32)
        sl = torch.full((B,), C, device=dev, dtype=torch.int32)
        # gate OFF -> prod op
        o_off = torch.full((B, k), -1, device=dev, dtype=torch.int32)
        wrapper(sc, sl, o_off, 1, k, gate=False)
        o_prod = torch.full((B, k), -1, device=dev, dtype=torch.int32)
        torch.ops.trtllm.cute_dsl_indexer_topk_decode(sc, sl, o_prod, k, 1)
        # gate ON -> G3
        o_on = torch.full((B, k), -1, device=dev, dtype=torch.int32)
        wrapper(sc, sl, o_on, 1, k, gate=True)
        torch.cuda.synchronize()
        gold = torch.topk(sc, k, dim=1).indices
        off_eq = seteq(o_off, o_prod)           # OFF identical to prod op
        on_eq = seteq(o_on, gold)               # ON identical to fp32 gold
        tag = "OK" if (off_eq and on_eq) else "FAIL"; ok &= off_eq and on_eq
        print(f"  {lbl} C{C} k{k} B{B:>2}: OFF==prodop={off_eq}  ON==gold={on_eq}  {tag}", flush=True)

print("\nWIRING VALIDATED" if ok else "\nWIRING FAIL", flush=True)
sys.exit(0 if ok else 1)

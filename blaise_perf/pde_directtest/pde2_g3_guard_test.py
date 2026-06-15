# Round-2 correctness validation of the G3 SMEM-capacity guard.
#  (1) py_compile both edited files under the image py3.12.
#  (2) gate-OFF: the wrapper logic uses the prod op, bit-exact (no G3 build).
#  (3) gate-ON small C (HISA block 1032/64; final 8192..32768/1024): G3 HANDLES
#      (returns True) and the set == fp32 gold (recall 1.0).
#  (4) gate-ON prod final width 132096 (and 65536): G3 DECLINES (returns False),
#      writes nothing, and the wrapper FALL-BACK op produces set == gold with NO
#      launch crash -- the exact failure this guard fixes.
import importlib.util
import os
import py_compile
import sys

import torch
import tensorrt_llm  # image cute_dsl + indexer ops

REPO = "/host_repo"
DSA = f"{REPO}/tensorrt_llm/_torch/attention_backend/sparse/dsa.py"
G3F = f"{REPO}/tensorrt_llm/_torch/attention_backend/sparse/pde_g3_topk.py"

os.makedirs("/tmp/pyc", exist_ok=True)
for f in (DSA, G3F):
    try:
        py_compile.compile(f, cfile=f"/tmp/pyc/{os.path.basename(f)}.pyc",
                           doraise=True)
        print(f"[compile OK] {os.path.basename(f)}", flush=True)
    except py_compile.PyCompileError as e:
        print(f"[compile FAIL] {f}: {e}", flush=True)
        sys.exit(1)

_s = importlib.util.spec_from_file_location("pde_g3_topk", G3F)
g3 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(g3)


def prod_op(sc, sl, out, k):
    # The bit-exact fallback op the wrapper uses (present in this image).
    torch.ops.trtllm.indexer_topk_decode(sc, sl, out, 1, k)


def wrapper(sc, sl, out, k, gate):
    """Mirror of dsa._indexer_topk_decode after the round-2 patch: when gated ON,
    try G3; if it DECLINES (SMEM cap), fall back to the prod op."""
    if gate:
        if g3.pde_g3_topk_decode(sc, sl, out, 1, k):
            return "G3"
        prod_op(sc, sl, out, k)
        return "FALLBACK"
    prod_op(sc, sl, out, k)
    return "PRODOP"


def set_eq(a, b):
    for r in range(a.shape[0]):
        if set(x for x in a[r].tolist() if x >= 0) != \
           set(x for x in b[r].tolist() if x >= 0):
            return False
    return True


dev = "cuda"
torch.manual_seed(3)
ok = True
optin = g3._pde_g3_smem_optin_bytes(0)
print(f"\ndevice optin SMEM = {optin} bytes (~{optin/1024:.0f}KB); "
      f"G3 fits up to ~{(optin - 1024)//4} cols\n", flush=True)

# (C, k, expect_route_when_ON)
cases = [
    (1032, 64, "G3"),      # HISA block top-k (prod)
    (8192, 1024, "G3"),    # final, fits
    (32768, 1024, "G3"),   # final, fits (129KB)
    (65536, 1024, "FALLBACK"),    # over cap -> decline + fall back
    (132096, 1024, "FALLBACK"),   # PROD padded final width -> decline + fall back
]
print("  C       k    gate-OFF==prodop | gate-ON route | ON-set==gold | crash?",
      flush=True)
for C, k, expect in cases:
    B = 4
    sc = torch.randn(B, C, device=dev, dtype=torch.float32)
    sl = torch.full((B,), C, device=dev, dtype=torch.int32)
    gold = torch.topk(sc, k, dim=1).indices

    # gate OFF == prod op (the wrapper's bit-exact fallback)
    o_off = torch.full((B, k), -1, device=dev, dtype=torch.int32)
    wrapper(sc, sl, o_off, k, gate=False)
    o_prod = torch.full((B, k), -1, device=dev, dtype=torch.int32)
    prod_op(sc, sl, o_prod, k)
    torch.cuda.synchronize()
    off_eq = set_eq(o_off, o_prod)

    # gate ON: route + correctness + no-crash
    crashed = ""
    o_on = torch.full((B, k), -1, device=dev, dtype=torch.int32)
    try:
        route = wrapper(sc, sl, o_on, k, gate=True)
        torch.cuda.synchronize()
        on_eq = set_eq(o_on, gold)
    except Exception as e:
        route = "EXC"
        on_eq = False
        crashed = f"{type(e).__name__}:{str(e)[:60]}"
    route_ok = (route == expect)
    good = off_eq and on_eq and route_ok and not crashed
    ok &= good
    tag = "OK" if good else "FAIL"
    print(f"  {C:>6} {k:>4}   OFF==prod={off_eq!s:>5}   {route:>9}("
          f"want {expect})={route_ok!s:>5}   ON==gold={on_eq!s:>5}   "
          f"{crashed or 'none':>10}  {tag}", flush=True)

print("\nG3_GUARD VALIDATED" if ok else "\nG3_GUARD FAIL", flush=True)
sys.exit(0 if ok else 1)

# Final integration check: exercise the REAL installed-package dispatch function
# tensorrt_llm._torch.attention_backend.sparse.dsa._indexer_topk_decode (with the
# round-2 patch overlaid onto the installed paths), proving the gate routing +
# SMEM-decline fallback works at the genuine import site -- not a mirror.
import os
import torch
import tensorrt_llm  # GPU import OK
from tensorrt_llm._torch.attention_backend.sparse.dsa import _indexer_topk_decode
from tensorrt_llm._torch.attention_backend.sparse import pde_g3_topk

dev = "cuda"
torch.manual_seed(7)
print(f"dsa module: {_indexer_topk_decode.__module__}", flush=True)
print(f"g3 enabled (gate {os.environ.get('TRTLLM_OPTRT_PDE_G3_TOPK')}): "
      f"{pde_g3_topk.pde_g3_topk_enabled()}", flush=True)


def set_eq(a, b):
    for r in range(a.shape[0]):
        if set(x for x in a[r].tolist() if x >= 0) != \
           set(x for x in b[r].tolist() if x >= 0):
            return False
    return True


ok = True
for C, k, lbl in [(1032, 64, "block-fits"), (132096, 1024, "final-PROD-decline")]:
    B = 4
    sc = torch.randn(B, C, device=dev, dtype=torch.float32)
    sl = torch.full((B,), C, device=dev, dtype=torch.int32)
    out = torch.full((B, k), -1, device=dev, dtype=torch.int32)
    gold = torch.topk(sc, k, dim=1).indices
    crashed = ""
    try:
        # The REAL prod dispatch (gate read inside). Should route to G3 where it
        # fits and silently fall back to indexer_topk_decode at prod width.
        _indexer_topk_decode(sc, sl, out, 1, k)
        torch.cuda.synchronize()
        eq = set_eq(out, gold)
    except Exception as e:
        eq = False
        crashed = f"{type(e).__name__}:{str(e)[:70]}"
    good = eq and not crashed
    ok &= good
    print(f"  C={C:>6} k={k:>4} [{lbl}] set==gold={eq} crash={crashed or 'none'} "
          f"{'OK' if good else 'FAIL'}", flush=True)

print("\nREAL_IMPORT_DISPATCH OK" if ok else "\nREAL_IMPORT_DISPATCH FAIL",
      flush=True)
import sys
sys.exit(0 if ok else 1)

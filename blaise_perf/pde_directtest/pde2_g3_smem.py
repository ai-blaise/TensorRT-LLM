# Probe the G3 kernel's SMEM ceiling: confirm whether pde_g3_topk_decode handles
# the PROD final-topk logits WIDTH. The kernel loads all C score columns into
# dynamic SMEM (C*4 + 1KB). B200 hard cap ~227KB/block => C beyond ~56K columns
# cannot fit. Block-topk C~1032 is safe; final-topk prod width is large. This
# decides whether the G3 hook needs a SMEM-aware fallback guard to stay
# correctness-safe (recall 1.0 OR clean fallback) at ALL gated-ON decode shapes.
import importlib.util
import torch

G3F = "/host_repo/tensorrt_llm/_torch/attention_backend/sparse/pde_g3_topk.py"
_s = importlib.util.spec_from_file_location("pde_g3_topk", G3F)
g3 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(g3)

dev = "cuda"
props = torch.cuda.get_device_properties(0)
print(f"device={props.name} sharedMemPerBlockOptin="
      f"{getattr(props, 'shared_memory_per_block_optin', 'n/a')} "
      f"sharedMemPerBlock={props.shared_memory_per_block}", flush=True)


def set_eq(a, b):
    for r in range(a.shape[0]):
        if set(x for x in a[r].tolist() if x >= 0) != \
           set(x for x in b[r].tolist() if x >= 0):
            return False
    return True


# Sweep widths from the block-topk regime up to the prod final padded width.
# k follows the seam: block uses k=64; final uses k=index_topk=1024.
cases = [
    (1032, 64, "block (prod)"),
    (8192, 1024, "final (docstring-claimed)"),
    (16384, 1024, "final 16k"),
    (32768, 1024, "final 32k"),
    (65536, 1024, "final 64k"),
    (132096, 1024, "final (PROD padded width)"),
]
torch.manual_seed(0)
for C, k, lbl in cases:
    smem_kb = (C * 4 + 256 * 4) / 1024
    sc = torch.randn(4, C, device=dev, dtype=torch.float32)
    sl = torch.full((4,), C, device=dev, dtype=torch.int32)
    out = torch.full((4, k), -1, device=dev, dtype=torch.int32)
    status = "ok"
    try:
        g3.pde_g3_topk_decode(sc, sl, out, 1, k)
        torch.cuda.synchronize()
        gold = torch.topk(sc, k, dim=1).indices
        rec = set_eq(out, gold)
        status = f"recall1.0={rec}"
    except Exception as e:
        status = f"FAIL: {type(e).__name__}: {str(e)[:120]}"
    print(f"  C={C:>6} k={k:>4} smem={smem_kb:7.1f}KB  [{lbl}]  -> {status}",
          flush=True)
print("G3_SMEM_PROBE DONE", flush=True)

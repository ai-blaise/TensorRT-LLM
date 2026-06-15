"""Probe for production NVFP4 GEMM ops (real 0.5 B/elt decode primitive)."""
import torch
print("torch", torch.__version__)
# look for trtllm custom ops
try:
    import tensorrt_llm  # noqa
    print("tensorrt_llm imported", getattr(tensorrt_llm, "__version__", "?"))
except Exception as e:
    print("trtllm import err", repr(e)[:160])

names = [n for n in dir(torch.ops) ]
print("torch.ops namespaces:", [n for n in names if not n.startswith("_")][:40])

for ns in ["trtllm"]:
    try:
        ops = torch.ops.__getattr__(ns)
        opnames = [o for o in dir(ops) if not o.startswith("_")]
        fp4 = [o for o in opnames if "fp4" in o.lower() or "nvfp4" in o.lower()
               or "fp8" in o.lower() or "gemm" in o.lower() or "scaled" in o.lower()]
        print(f"ns {ns}: {len(opnames)} ops; fp4/gemm-ish:", fp4[:40])
    except Exception as e:
        print(f"ns {ns} err", repr(e)[:120])
print("OPS PROBE OK")

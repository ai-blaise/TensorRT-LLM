import torch, sys, os
sys.path.insert(0, "/host_repo")
print("TORCH", torch.__version__, flush=True)
# Trigger op registration. The C++ custom ops register on import of the compiled lib.
import importlib
reg = False
for modname in ["tensorrt_llm", "tensorrt_llm.bindings", "tensorrt_llm._torch.custom_ops"]:
    try:
        importlib.import_module(modname)
        print("IMPORTED", modname, flush=True)
        reg = True
    except Exception as e:
        print("IMPORTERR", modname, repr(e)[:200], flush=True)
names = sorted(dir(torch.ops.trtllm))
print("TOTAL_TRTLLM_OPS", len(names), flush=True)
pats = ["kvarn","hot","dequant","fp8","fp4","nvfp4","rmsnorm","rms_norm","fused_add","fused","indexer","gather","quant","mla","scale","sparse","attention","mqa","bmm","silu","gemm"]
for p in pats:
    hh = [x for x in names if p.lower() in x.lower()]
    if hh:
        print(f"== {p} ({len(hh)}) ==")
        for x in hh:
            print("   ", x)

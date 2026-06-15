import torch, sys
print("TORCH", torch.__version__, flush=True)
# Do NOT shadow with /host_repo. Use the image's installed tensorrt_llm (matching .so).
import importlib
ok = False
for modname in ["tensorrt_llm._torch.custom_ops", "tensorrt_llm"]:
    try:
        importlib.import_module(modname)
        print("IMPORTED", modname, flush=True)
        ok = True
        break
    except Exception as e:
        print("IMPORTERR", modname, repr(e)[:160], flush=True)
import tensorrt_llm as t
print("TRTLLM_FILE", t.__file__, flush=True)
names = sorted(dir(torch.ops.trtllm))
print("TOTAL_TRTLLM_OPS", len(names), flush=True)
pats = ["kvarn","hot","dequant","fp8","fp4","nvfp4","rmsnorm","rms_norm","fused_add","fused","indexer","gather","quant","mla","scale","sparse","mqa","bmm","silu"]
for p in pats:
    hh = [x for x in names if p.lower() in x.lower()]
    if hh:
        print(f"== {p} ({len(hh)}) ==")
        for x in hh:
            print("   ", x)

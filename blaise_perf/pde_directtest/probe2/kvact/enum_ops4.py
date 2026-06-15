import torch, importlib
importlib.import_module("tensorrt_llm._torch.custom_ops")
names = sorted(dir(torch.ops.trtllm))
for p in ["sparse_mla","mla_rope","mla_gen","kv","cache","paged","logits","topk","fmha","mla_"]:
    hh=[x for x in names if p.lower() in x.lower()]
    if hh:
        print(f"== {p} ==")
        for x in hh: print("   ",x)
print("ALL98:")
for x in names: print("  ", x)

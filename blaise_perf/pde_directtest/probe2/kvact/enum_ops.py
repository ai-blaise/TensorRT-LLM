import torch, re, sys
print("TORCH", torch.__version__, "CUDA", torch.cuda.is_available(), flush=True)
try:
    print("DEV", torch.cuda.get_device_name(0), flush=True)
except Exception as e:
    print("DEVERR", e, flush=True)
# enumerate trtllm ops of interest
import torch.ops
names = []
try:
    for n in dir(torch.ops.trtllm):
        names.append(n)
except Exception as e:
    print("ENUMERR", e)
pats = ["kvarn", "hot", "dequant", "fp8", "fp4", "nvfp4", "rmsnorm", "rms_norm", "fused_add", "indexer", "gather", "quant", "mla", "scale", "fp4_quant", "act"]
hits = {}
for p in pats:
    hits[p] = sorted([x for x in names if p.lower() in x.lower()])
for p in pats:
    print(f"== {p} ==")
    for x in hits[p]:
        print("  ", x)
print("TOTAL_TRTLLM_OPS", len(names), flush=True)

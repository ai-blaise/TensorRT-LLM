import torch, tensorrt_llm
import tensorrt_llm.quantization.utils.fp4_utils as f
print("FP4UTILS:", [n for n in dir(f) if not n.startswith("_")],flush=True)
for n in dir(f):
    if any(k in n.lower() for k in ("swizzle","scale","sf","block","reorder","interleave")):
        print("  HELPER:", n,flush=True)
ops=[o for o in dir(torch.ops.trtllm) if not o.startswith("_")]
print("OPS swizzle/sf/scale/reorder:", [o for o in ops if any(k in o.lower() for k in ("swizzle","scale","_sf","block_scale","reorder","unpack","dequant"))],flush=True)
print("OPS fp4:", [o for o in ops if "fp4" in o.lower()],flush=True)
print("DONE",flush=True)

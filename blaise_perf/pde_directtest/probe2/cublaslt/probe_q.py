import torch, tensorrt_llm
ov=torch.ops.trtllm.fp4_quantize
for o in (ov.overloads() if hasattr(ov,'overloads') else []):
    print("OVERLOAD",o,"::",getattr(ov,o)._schema)
try:
    print("DEFAULT_SCHEMA::", ov.default._schema)
except Exception as e:
    print("nodefault",repr(e)[:80])

import torch, tensorrt_llm  # noqa
ops = torch.ops.tensorrt_llm
opnames = [o for o in dir(ops) if not o.startswith("_")]
print(f"tensorrt_llm: {len(opnames)} ops")
kw = ["fp4","nvfp4","fp8","gemm","scaled","quant","mm","linear","moe"]
hit = [o for o in opnames if any(k in o.lower() for k in kw)]
for o in sorted(hit):
    print(" ", o)
print("---also flashinfer---")
try:
    fi = torch.ops.flashinfer
    fin = [o for o in dir(fi) if not o.startswith("_")]
    h2 = [o for o in fin if any(k in o.lower() for k in kw)]
    for o in sorted(h2): print("  fi:", o)
except Exception as e:
    print("flashinfer err", repr(e)[:120])
print("OPS2 OK")

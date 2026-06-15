import torch, importlib
importlib.import_module("tensorrt_llm._torch.custom_ops")
names = sorted(dir(torch.ops.trtllm))
for q in ["fused_add_rms_norm_quant","fused_add_rmsnorm","tunable_fp4_quantize","flashinfer_fused_add_rmsnorm","flashinfer_rmsnorm","cuda_tile_rms_norm_fuse_residual_"]:
    print(q, "PRESENT" if q in names else "ABSENT")
# show all rms / quant / fp4 quantize names
print("---rms/quant---")
for x in names:
    if "rms" in x.lower() or ("quant" in x.lower() and "fp4" in x.lower()) or "fp4_quant" in x.lower():
        print("  ",x)

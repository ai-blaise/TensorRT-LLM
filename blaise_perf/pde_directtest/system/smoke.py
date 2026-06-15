import os, torch
print("TORCH", torch.__version__, "CUDA", torch.cuda.is_available(), torch.cuda.get_device_name(0))
to = getattr(torch.ops, "trtllm", None)
names = ["fp4_block_scale_moe_runner", "nvfp4_gemm", "fp4_quantize", "warp_decode_nvfp4_cursor_moe", "warp_decode_nvfp4_moe"]
for n in names:
    print(f"op {n}:", hasattr(to, n) if to else "NO trtllm ns")
# FP4BlockScaleMoERunner class
try:
    r = torch.classes.trtllm.FP4BlockScaleMoERunner(0)
    print("FP4BlockScaleMoERunner class: OK")
except Exception as e:
    print("FP4BlockScaleMoERunner class:", type(e).__name__, str(e)[:120])
# CutlassFusedMoE / CuteDslFusedMoE import
for mod in ["CutlassFusedMoE","CuteDslFusedMoE","TRTLLMGenFusedMoE","DenseGEMMFusedMoE"]:
    try:
        m = __import__("tensorrt_llm._torch.modules.fused_moe", fromlist=[mod])
        print(f"backend {mod}:", "OK" if hasattr(m, mod) else "MISSING")
    except Exception as e:
        print(f"backend {mod}: import-err", type(e).__name__)
print("MEGAKERNEL_ENV", os.environ.get("TRTLLM_OPTRT_MOE_MEGAKERNEL"))

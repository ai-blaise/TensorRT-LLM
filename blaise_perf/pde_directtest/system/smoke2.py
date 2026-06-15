import os, torch
import tensorrt_llm  # triggers op registration
to = torch.ops.trtllm
names = ["fp4_block_scale_moe_runner","nvfp4_gemm","fp4_quantize","warp_decode_nvfp4_cursor_moe","warp_decode_nvfp4_moe"]
for n in names:
    print(f"op {n}:", hasattr(to, n))
try:
    r = torch.classes.trtllm.FP4BlockScaleMoERunner(0)
    print("FP4BlockScaleMoERunner: OK")
except Exception as e:
    print("FP4BlockScaleMoERunner:", type(e).__name__, str(e)[:80])
print("MEGAKERNEL_ENV", os.environ.get("TRTLLM_OPTRT_MOE_MEGAKERNEL"))

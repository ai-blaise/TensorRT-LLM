import torch, tensorrt_llm
to = torch.ops.trtllm
for n in ["sparse_mla_decode_nvfp4","sparse_mla_decode_kvarn_hot","fp8_block_scaling_bmm_out","fp8_block_scaling_gemm"]:
    print(f"op {n}:", hasattr(to, n))
try:
    from tensorrt_llm.flash_mla import flash_mla_sparse_fwd
    print("flash_mla_sparse_fwd:", flash_mla_sparse_fwd is not None)
except Exception as e:
    print("flash_mla import:", type(e).__name__, str(e)[:80])

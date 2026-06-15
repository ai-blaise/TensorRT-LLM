import torch, tensorrt_llm
for o in ["nvfp4_gemm","nvfp4_gemm_cutlass","nvfp4_gemm_cublaslt","cute_dsl_nvfp4_gemm_blackwell",
          "warp_decode_nvfp4_moe","fp4_block_scale_moe_runner","cute_dsl_nvfp4_grouped_gemm_blackwell",
          "tunable_fp4_quantize"]:
    op=getattr(torch.ops.trtllm,o,None)
    if op is None: print("MISSING",o); continue
    try: print(f"{o} :: {str(op.default._schema)}")
    except Exception as e: print(f"{o} SCHEMA_ERR {e}")

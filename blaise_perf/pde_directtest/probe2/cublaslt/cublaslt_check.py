import torch, tensorrt_llm
print("TRTLLM", getattr(tensorrt_llm, "__version__", "?"))
try:
    from tensorrt_llm._torch.cublaslt_utils import IS_CUBLASLT_AVAILABLE
    print("IS_CUBLASLT_AVAILABLE", IS_CUBLASLT_AVAILABLE)
except Exception as e:
    print("cublaslt_utils_import_err", repr(e)[:120])
print("has_nvfp4_gemm_cublaslt_op", hasattr(torch.ops.trtllm, "nvfp4_gemm_cublaslt"))
try:
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import CublasLtFP4GemmRunner
    r = CublasLtFP4GemmRunner(0, torch.bfloat16)
    print("CublasLtFP4GemmRunner_INSTANTIATE_OK")
except Exception as e:
    print("CublasLtFP4GemmRunner_ERR", repr(e)[:150])

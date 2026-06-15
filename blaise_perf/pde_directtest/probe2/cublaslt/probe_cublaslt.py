import torch, sys, traceback
try:
    import tensorrt_llm
    ver = getattr(tensorrt_llm, "__version__", "?")
except Exception as e:
    print("IMPORT_FAIL", repr(e)); sys.exit(0)

ops = torch.ops.trtllm
names = list(dir(ops))
relevant = sorted([n for n in names if "nvfp4_gemm" in n or "cute_dsl_nvfp4" in n])
print("TRTLLM_VER", ver)
print("RELEVANT_OPS", relevant)
print("HAS_cublaslt_attr", "nvfp4_gemm_cublaslt" in names)
print("HAS_cutlass_attr", "nvfp4_gemm" in names)
print("HAS_cutedsl_attr", any("cute_dsl_nvfp4" in n for n in names))

# Probe registration of cublaslt by calling it. If the op object exists but the C++
# backend (CublasLtFP4GemmRunner) is missing, the call raises with a "not registered"/
# "no kernel"/"unsupported" style error; if the op object itself is absent, attr access fails.
status = "ATTR_ABSENT"
if "nvfp4_gemm_cublaslt" in names:
    try:
        M, K, N = 16, 256, 256
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        alpha = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        # Intentionally minimal/typed args; we only classify the error, not correctness.
        out = torch.ops.trtllm.nvfp4_gemm_cublaslt(x, w, x, w, alpha, False)
        status = "CALLED_OK"
    except Exception as e:
        msg = str(e).lower()
        if ("not registered" in msg) or ("no such operator" in msg) or ("could not find" in msg) or ("no kernel" in msg) or ("not implemented" in msg):
            status = "RUNNER_MISSING: " + str(e)[:200]
        else:
            # A type/shape/dtype error means the op + C++ runner ARE present (it got far enough to validate args).
            status = "RUNNER_PRESENT(argerr): " + str(e)[:200]
print("CUBLASLT_STATUS", status)

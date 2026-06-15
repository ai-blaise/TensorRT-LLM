import torch, tensorrt_llm, inspect
for n in ["nvfp4_gemm","nvfp4_gemm_cutlass","nvfp4_gemm_cublaslt","cute_dsl_nvfp4_gemm_blackwell"]:
    ov = getattr(torch.ops.trtllm, n)
    try:
        print("SCHEMA", n, "::", ov.default._schema)
    except Exception as e:
        # multiple overloads
        try:
            for o in ov.overloads():
                print("SCHEMA", n, o, "::", getattr(ov,o)._schema)
        except Exception as e2:
            print("SCHEMA", n, "ERR", repr(e)[:120], repr(e2)[:120])
# print _input_prepare source
from tensorrt_llm._torch.modules.linear import NVFP4LinearMethod
print("=== _input_prepare ===")
print(inspect.getsource(NVFP4LinearMethod._input_prepare))
# tunable_fp4_quantize signature
try:
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import tunable_fp4_quantize
    print("tunable_fp4_quantize sig:", inspect.signature(tunable_fp4_quantize))
except Exception as e:
    print("tunable_fp4_quantize import ERR", repr(e)[:160])

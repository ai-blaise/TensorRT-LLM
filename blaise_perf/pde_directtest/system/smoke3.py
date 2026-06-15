import os, sys
REPO="/host_repo"
import tensorrt_llm
sys.path.insert(0, os.path.join(REPO,"tests","unittest","_torch","modules","moe"))
sys.path.insert(0, os.path.join(REPO,"tests","unittest"))
try:
    from tensorrt_llm._torch.modules.fused_moe.routing import RenormalizeMoeRoutingMethod
    print("RenormalizeMoeRoutingMethod OK")
except Exception as e: print("routing import:", type(e).__name__, str(e)[:120])
try:
    import moe_test_utils as MTU
    from moe_test_utils import MoeBackendType
    print("MoeBackendType OK:", [m.value for m in MoeBackendType][:8])
except Exception as e:
    import traceback; traceback.print_exc()
try:
    import quantize_utils as QU
    print("quantize_utils OK; get_test_quant_params:", hasattr(QU,"get_test_quant_params"))
except Exception as e:
    print("quantize_utils:", type(e).__name__, str(e)[:200])
try:
    import test_moe_backend as TMB
    print("test_moe_backend OK; create_test_backend:", hasattr(TMB,"create_test_backend"), "run_backend_moe:", hasattr(TMB,"run_backend_moe"))
except Exception as e:
    print("test_moe_backend:", type(e).__name__, str(e)[:200])

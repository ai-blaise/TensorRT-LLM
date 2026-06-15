import os, sys
REPO="/host_repo"
os.environ.setdefault("TRTLLM_OPTRT_MOE_MEGAKERNEL","0")
import tensorrt_llm
sys.path.insert(0, os.path.join(REPO,"tests","unittest","_torch","modules","moe"))
sys.path.insert(0, os.path.join(REPO,"tests","unittest"))
import torch
import test_moe_backend as TMB
from moe_test_utils import MoeBackendType, get_backend_class
# Confirm WARPDECODE and CUTEDSL resolve to CuteDslFusedMoE; CUTLASS to CutlassFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cute_dsl import CuteDslFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass import CutlassFusedMoE
print("CUTEDSL ->", get_backend_class(MoeBackendType.CUTEDSL).__name__)
print("WARPDECODE ->", get_backend_class(MoeBackendType.WARPDECODE).__name__)
print("CUTLASS ->", get_backend_class(MoeBackendType.CUTLASS).__name__)
print("CUTEDSL is CuteDslFusedMoE:", get_backend_class(MoeBackendType.CUTEDSL) is CuteDslFusedMoE)
print("WARPDECODE is CuteDslFusedMoE:", get_backend_class(MoeBackendType.WARPDECODE) is CuteDslFusedMoE)

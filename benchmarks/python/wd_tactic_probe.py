"""Confirm the WarpDecode FC1 autotuner genuinely explores the 2-CTA/256x256 tactic at the decode
shape and report which tactic it selects. Cell: G8 conc32 -> LE=16 experts/rank, ntok=32 tokens."""
import os, torch
os.environ["TRTLLM_ENABLE_PDL"] = "1"
from tensorrt_llm.logger import logger
logger.set_level("debug")
import tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops as cdc
from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm._torch.autotuner import autotune, AutoTuner
HIDDEN, INTERMEDIATE, NE, TK, SV = 7168, 2048, 128, 8, 16
dev = torch.device("cuda")
LE, ntok = 16, 32
w13 = torch.randint(0, 256, (LE, 2 * INTERMEDIATE, HIDDEN // 2), device=dev, dtype=torch.uint8)
w13sf = torch.randint(1, 8, (LE, 2 * INTERMEDIATE, HIDDEN // SV), device=dev, dtype=torch.uint8)
a1 = torch.ones((LE,), device=dev, dtype=torch.float32); gsf = torch.ones((1,), device=dev, dtype=torch.float32)
x4 = torch.randint(0, 256, (ntok, HIDDEN // 2), device=dev, dtype=torch.uint8)
xsf = torch.randint(1, 8, (ntok, HIDDEN // SV), device=dev, dtype=torch.uint8)
ids = (torch.arange(ntok * TK, device=dev, dtype=torch.int32).reshape(ntok, TK) % LE).contiguous()
wf32 = torch.full((ntok, TK), 1.0 / TK, device=dev, dtype=torch.float32)
t2e, t2lim, e2p, p2e, tot, nt = torch.ops.trtllm.moe_sort(token_selected_experts=ids, token_final_scales=wf32, num_experts=NE, top_k=TK, local_expert_offset=0, local_num_experts=LE, tile_tokens_dim=128)
def fc1():
    return torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell(
        input=x4.view(torch.float4_e2m1fn_x2), weight=w13.view(torch.float4_e2m1fn_x2), input_scale=xsf.view(torch.uint8),
        weight_scale=w13sf.view(torch.uint8), alpha=a1, tile_idx_to_group_idx=t2e, tile_idx_to_mn_limit=t2lim,
        permuted_idx_to_expanded_idx=p2e, num_non_exiting_tiles=nt, global_sf=gsf, num_experts=NE, top_k=TK,
        num_local_experts=LE, local_expert_offset=0, tile_size=128, scaling_vector_size=SV, activation_type=int(ActivationType.Swiglu))
with autotune():
    for _ in range(6): fc1()
torch.cuda.synchronize()
print("\n===== PROFILING CACHE (cache_key -> (runner_id, tactic_idx, min_time_ms)) =====", flush=True)
for k, v in AutoTuner.get().profiling_cache.cache.items():
    if "gather_grouped_gemm" in str(k[0]):
        print("ENTRY:", k[0], "| key_tail:", k[1:], "| value:", v, flush=True)

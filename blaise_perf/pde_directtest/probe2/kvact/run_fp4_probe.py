# Drives the existing run_fp4.py harness internals + adds CUDA-graph timing.
# Measures cute_dsl FP4 paged MQA logits (indexer KV-scan) at prod B=1 shape.
import sys, time
sys.argv = ["probe"]  # neutralize argparse in imported module
HARNESS_DIR = "/host_repo/tests/scripts/cute_dsl_kernels/paged_mqa_logits"
sys.path.insert(0, HARNESS_DIR)
sys.path.insert(0, "/host_repo/tensorrt_llm/_torch/cute_dsl_kernels")
import torch, cutlass
import run_fp4 as H

# Quick validation run at B=1 prod-ish ctx to confirm kernel builds in-image.
print("=== validation B=1 avg_ctx=4608 (fixed) ===", flush=True)
t0=time.time()
diff = H.run(batch_size=1, next_n=1, avg_ctx=4608, num_heads=64, head_dim=128,
             phys_block_kv=64, epi_dtype=cutlass.Float32, output_dtype=cutlass.Float16,
             fix_length=True, num_sms=148)
print(f"VALIDATION_DIFF={diff:.3e} wall={time.time()-t0:.1f}s", flush=True)

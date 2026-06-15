# Empirical op-inventory probe of the proof image: establishes which C++ ops the
# G4 overlap / hisparse-hot / G8 draft paths require are PRESENT vs ABSENT, and
# whether the copy-schedule op advertises the G4 overlap args. Ground truth for
# "JIT-tractable now" vs "needs the C++ build/serve".
import torch
import tensorrt_llm  # registers the image's trtllm ops


def has_op(qual):
    ns, name = qual.split("::")
    try:
        return hasattr(getattr(torch.ops, ns), name)
    except Exception:
        return False


def schema_args(qual):
    ns, name = qual.split("::")
    try:
        op = getattr(getattr(torch.ops, ns), name)
        return {a.name for a in op.default._schema.arguments}
    except Exception as e:
        return f"<no-schema: {type(e).__name__}>"


print("=== topk ops (G3 seam: present => G3 substitutes here when gated ON) ===", flush=True)
for q in ("trtllm::indexer_topk_decode",
          "trtllm::cute_dsl_indexer_topk_decode"):
    print(f"  {q}: present={has_op(q)}", flush=True)

print("\n=== G4 copy-overlap seam (hisparse) ===", flush=True)
COPY = "trtllm::hisparse_submit_packed_kvarn_copy_schedule"
print(f"  {COPY}: present={has_op(COPY)}", flush=True)
if has_op(COPY):
    args = schema_args(COPY)
    g4_args = {"overlap_copy_stream", "copy_stream_handle"}
    print(f"    schema args: {sorted(args) if isinstance(args, set) else args}", flush=True)
    print(f"    advertises G4 overlap args {g4_args}: "
          f"{g4_args.issubset(args) if isinstance(args, set) else False}", flush=True)

print("\n=== G4 hisparse hot-pool op chain (map_topk_to_hot_pool requires ALL) ===", flush=True)
for q in ("trtllm::hisparse_topk_to_block_positions",
          "trtllm::hisparse_resolve_blocks_to_host_slots",
          "trtllm::hisparse_classify_resident_blocks",
          "trtllm::hisparse_plan_hot_slots",
          "trtllm::hisparse_compact_miss_schedule",
          "trtllm::hisparse_commit_hot_slots",
          "trtllm::hisparse_build_hot_indices",
          "trtllm::sparse_mla_decode_kvarn_hot"):
    print(f"  {q}: present={has_op(q)}", flush=True)

print("\n=== G9 cross-step reuse seam (recency patch op) ===", flush=True)
print(f"  trtllm::indexer_xstep_recency_patch: "
      f"present={has_op('trtllm::indexer_xstep_recency_patch')}", flush=True)

print("\n=== G8 SMC draft loop op (extract real draft tokens, CUDA-graph path) ===", flush=True)
print(f"  trtllm::extract_real_draft_tokens_op: "
      f"present={has_op('trtllm::extract_real_draft_tokens_op')}", flush=True)

print("\n=== HISA fused block-score op (block-topk feeds _indexer_topk_decode=>G3) ===", flush=True)
print(f"  trtllm::indexer_hisa_block_scores_nvfp4: "
      f"present={has_op('trtllm::indexer_hisa_block_scores_nvfp4')}", flush=True)
print("\nOPCHECK DONE", flush=True)

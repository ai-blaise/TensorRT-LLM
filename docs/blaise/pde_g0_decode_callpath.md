# PDE G0 — Decode forward call-path map (sparse-MLA path, DeepSeek-V3.2-REAP-NVFP4)

Repo: `/home/spencer/work/pde-wt` @ `op-trt-pde` (base 818065cb). Factual map of the
**per-op kernel-launch sequence in ONE decode layer**, with `file:line` anchors, used
to place the megakernel grid-barriers. Line numbers verified against the worktree on
2026-06-14 (not from prior stale notes).

## Top-level decode dispatch (one decoder layer)

`DeepseekV3DecoderLayer.forward` — `tensorrt_llm/_torch/models/modeling_deepseekv3.py:1709`
1. `input_layernorm` (RMSNorm) + optional gated-norm — `modeling_deepseekv3.py:1720,1721`
2. `self.self_attn(...)` (DSA + MLA attention block) — `modeling_deepseekv3.py:1725`
3. `forward_MoE(...)` (expert FFN / WarpDecode) — `modeling_deepseekv3.py:1740` (`forward_MoE` body from :1760)
4. `post_attention_layernorm` + (optional) PRE/POST MoE all-reduce fusion — `modeling_deepseekv3.py:1781,1795`

The decode model forward over ALL layers is captured as **one CUDA graph**:
`tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py:503` —
`with torch.cuda.graph(graph, pool=self.memory_pool):` (capture region wraps the full
`forward_fn`). Replayed via `replay()` at `cuda_graph_runner.py:513`. So the launch
sequence below is fixed at capture; the *execution gap* is the intra-op host round-trips
and inter-kernel activation HBM round-trips replayed each step.

## Attention block: `MLA.forward_impl_with_dsa` — `attention.py:1914`

Delegates to `forward_dsa_proj` (token-wise projections, graph-safe) then
`forward_dsa_attn` (batch-dependent attention dispatch).

### A. `forward_dsa_proj` — `attention.py:1939`  (REGION: PROJECTIONS, pre-Indexer)
- `kv_a_proj_with_mqa(hidden_states)` GEMM + split — `attention.py:1963`
- `q_a_layernorm` / `kv_a_layernorm` (parallel on aux stream) — `attention.py:1967`
- `q_b_proj(q)` GEMM — `attention.py:1975`
- `indexer.pre_indexer_proj(qr, hidden_states, position_ids)` — `attention.py:1992`
  (cublas_mm + rope + FP4/FP8 quantize + weight scaling; CUDA-graph-safe token-wise)

### B. `forward_dsa_attn` -> Indexer top-k — `attention.py:1999` -> `dsa.py:4062`  (REGION: INDEXER  <<< BARRIER BOUNDARY 1 after this)
`topk_indices = self.mqa.indexer.sparse_attn_indexer(...)` — `attention.py:2048`
Inside `Indexer.sparse_attn_indexer` (`dsa.py:4062`), the **decode** branch:
- width-correct logits buffer to graph-safe bucket — `dsa.py` `_indexer_logits_width` (~4283)
- **paged MQA logits kernel** (the indexer's per-token×candidate score GEMM):
  - prod path: `torch.ops.trtllm.fp8_paged_mqa_logits` via `_call_mqa_logits` (DeepGEMM), or
  - `torch.ops.trtllm.cute_dsl_fp4_paged_mqa_logits` — `dsa.py:4411`
  - `torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits` — `dsa.py:4437`
- **top-k selection** over logits — one of:
  - `torch.ops.trtllm.cute_dsl_indexer_topk_decode` — `dsa.py:4539` (num_gen_tokens<=256 path)
  - `torch.ops.trtllm.indexer_topk_decode` — `dsa.py:4547` (prod C++ top-k; index_topk=1024/2048)
  - (HISA-from-logits decode path `_should_use_hisa_logits` is statically False in prod)
  Result: `topk_indices` [num_gen_tokens, index_topk].

### C. HiSparse hot/cold swap-in — `hisparse.py`  (REGION: KV STAGING, overlaps B/D)
Hoisted to the **top of** `forward_absorption_generation` (`attention.py:2966-2978`) via
`hisparse_coordinator.prepare_hot_pool_overlapped` (`hisparse.py:2274`), which runs
`map_topk_to_hot_pool` (`hisparse.py:2373`). Op chain (all `torch.ops.trtllm.*`):
- `hisparse_topk_to_block_positions` — `hisparse.py:2547`
- `hisparse_classify_resident_blocks` — `hisparse.py:2553`
- `hisparse_resolve_blocks_to_host_slots` — `hisparse.py:2563`
- `hisparse_plan_hot_slots` — `hisparse.py:2579`
- `hisparse_compact_miss_schedule` — `hisparse.py:2592`
- `hisparse_commit_hot_slots` — `hisparse.py:2610`
- `hisparse_build_hot_indices` — `hisparse.py:2624`
- `hisparse_submit_packed_kvarn_copy_schedule` — `hisparse.py:2188/2271` (the H2D/D2D KV copy)
  Issued on the coordinator copy stream BEFORE bmm+rope; join DEFERRED into the hot-read.

### D. MLA decode core — `forward_absorption_generation` — `attention.py:2933`  (REGION: ATTENTION  <<< BARRIER BOUNDARY 2 after this)
- bmm (q_nope · k_b) ‖ `mla_rope_generation` in parallel (`maybe_execute_in_parallel`),
  writing the fused_q buffer — `attention.py:3037-3056` (bf16) / `:3061-3083` (fp8):
  - `_bmm_bf16_out` (bmm into `fused_q[..., :kv_lora_rank]`) — `attention.py:3038`
  - `mqa.mla_rope_generation(fused_q, q_pe, latent_cache, ...)` — `attention.py:3041`
    (applies rope into the rope slice of fused_q; quantizes K write to KV cache)
- **FlashMLA hot-read** decode (the sparse-MLA attention kernel):
  `attn_out_latent = self._sparse_mla_decode_kvarn_hot(fused_q, attn_metadata, topk_indices, num_tokens)`
  — `attention.py:3094` -> `torch.ops.trtllm.sparse_mla_decode_kvarn_hot` (ABI-frozen op
  `cpp/.../thop/SparseMlaDecodeKvarnHotOp.cpp`, kernel `cpp/.../kernels/hisparseKvarnBdrRead.cuh`)
  (NVFP4 dense-KV alternative: `_sparse_mla_decode_nvfp4` — `attention.py:3100`)
- output `o_proj` / out-absorption GEMM (back to hidden_size) — in `forward_absorption_generation` tail (after :3094).

## MoE block: `forward_MoE` -> WarpDecode — `modeling_deepseekv3.py:1760`  (REGION: MoE  <<< BARRIER BOUNDARY 3 after this)
- `post_attention_layernorm` (+ optional gated-norm quant to NVFP4) — `modeling_deepseekv3.py:1795` / `_apply_post_attention_gated_norm_quant` :1639
- router gate GEMM + top-k expert select (`self.mlp.gate` / routing) — inside `Deepseekv3MoE` forward
- **WarpDecode NVFP4 MoE** (the fused gather-grouped-GEMM + SwiGLU + FC2/down-proj + finalize):
  `torch.ops.trtllm.warp_decode_nvfp4_moe` (registered `warp_decode.py:516`) ->
  `torch.ops.trtllm.fp4_block_scale_moe_runner` — `warp_decode.py:591`
  (FC1 gather-grouped-GEMM -> SwiGLU/silu -> FC2 grouped-GEMM -> grouped-finalize/scatter)
  Megakernel variant exists at `cute_dsl_kernels/blackwell/moe_as_dense_gemm/fused_moe_megakernel.py`.
- shared-expert GatedMLP (FC1->silu->FC2) added in parallel for the dense shared expert.
- optional POST_MOE all-reduce fusion — `modeling_deepseekv3.py` forward_MoE tail.

## Ordered per-op launch list for ONE decode layer (megakernel target)

```
# === REGION: PROJECTIONS ===
 1. kv_a_proj_with_mqa GEMM                    attention.py:1963
 2. q_a_layernorm / kv_a_layernorm (RMSNorm)   attention.py:1967
 3. q_b_proj GEMM                              attention.py:1975
 4. indexer.pre_indexer_proj (mm+rope+quant)   attention.py:1992
# ---------------- GRID-BARRIER 1 (proj -> indexer) ----------------
# === REGION: INDEXER ===
 5. fp8/fp4 paged_mqa_logits (score GEMM)      dsa.py:4411/4437 (or fp8_paged_mqa_logits)
 6. indexer_topk_decode (top-k select)         dsa.py:4539/4547
# ---------------- GRID-BARRIER 2 (indexer -> attention) ----------------
# === REGION: KV STAGING (overlaps, issued pre-bmm) ===
 7. hisparse map_topk_to_hot_pool op-chain     hisparse.py:2547-2624
 8. hisparse_submit_packed_kvarn_copy_schedule hisparse.py:2188/2271
# === REGION: ATTENTION ===
 9. _bmm_bf16_out (q_nope · k_b -> fused_q)     attention.py:3038
10. mla_rope_generation (rope + K-write)        attention.py:3041
11. sparse_mla_decode_kvarn_hot (FlashMLA)      attention.py:3094  [ABI-FROZEN op]
12. o_proj / out-absorption GEMM                attention.py: forward_absorption_generation tail
# ---------------- GRID-BARRIER 3 (attention -> MoE) ----------------
# === REGION: MoE ===
13. post_attention_layernorm (+gated-norm quant) modeling_deepseekv3.py:1795
14. router gate GEMM + expert top-k             Deepseekv3MoE.forward
15. warp_decode_nvfp4_moe (FC1->SwiGLU->FC2->fin) warp_decode.py:516 -> 591
16. shared-expert GatedMLP (FC1->silu->FC2)      Deepseekv3MoE.forward
# ---------------- GRID-BARRIER (layer N -> layer N+1) ----------------
```

## Megakernel grid-barrier placement (region boundaries)

Three intra-layer grid-barriers separate the data-dependent regions; a fourth closes the
layer. These are exactly the points where the current implementation today pays a kernel
launch + (in the indexer/hisparse ops) host round-trips:

- **Barrier 1 — PROJECTIONS -> INDEXER**: topk needs the indexer-projected q/k FP8 + weights.
- **Barrier 2 — INDEXER -> ATTENTION**: the FlashMLA hot-read consumes `topk_indices`
  (and the hisparse swap-in keyed on it). This is the highest-value boundary — it currently
  carries the indexer top-k + the hisparse planner host syncs.
- **Barrier 3 — ATTENTION -> MoE**: the MoE consumes the attention output (post-attn-norm).
- **Barrier (layer boundary)**: MoE output (+residual) feeds the next layer's input-norm.

G0 provides the *substrate* (persistent cooperative grid, grid-wide region barrier,
warp-role scaffold, atomic work-queue) that makes consolidating launches 1-16 into one
resident grid with these barriers possible; it does NOT modify any of the above decode code.

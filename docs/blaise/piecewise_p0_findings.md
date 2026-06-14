# P0 findings — Piecewise (PCG) for the DSA prefill path (op-trt-hisparse)

Branch `op-trt-hisparse-piecewise` (off `818065cb`), node 001, worktree
`/home/spencer/work/piecewise-wt`. Read-only characterization + feasibility.

## TL;DR — the plan's central premise does NOT hold for this op-trt codebase

The plan assumes op-trt is in SGLang's *pre-#23351* state for DSA (DSA forcibly
excluded from piecewise via a disabled-arch list / `is_deepseek_dsa` guard, with
only the indexer top-k bundled eager). **That is not the case.** op-trt has
already implemented the #23351 fix-set — and a *more complete* version of it:

- DSA is already split into two custom ops (`tensorrt_llm/_torch/modules/
  attention.py`):
  - **Op 1 `trtllm::mla_dsa_proj`** (line 1034) — token-wise projections
    (kv_a_proj, layernorms, q_b_proj, indexer pre-proj/quant). Explicitly
    *CUDA-graph-capturable*, straight-line under compile.
  - **Op 2 `trtllm::mla_dsa_attn_inplace`** (line 1109) — the indexer
    (top-k + index-K-cache scatter/gather) + sparse-attention dispatch.
    Explicitly the *eager* piece ("This op is excluded from CUDA graph
    capture", line 1131). This is the op-trt analog of SGLang's split op —
    but it bundles the *entire* indexer+attention, not just the top-k.
- Op 2 is ALREADY registered as a split point in the piecewise partitioner
  (`tensorrt_llm/_torch/compilation/piecewise_optimizer.py:265`,
  `torch.ops.trtllm.mla_dsa_attn_inplace.default` in the `is_call_function`
  list, with the "safe to continue splitting after attention" branch).
- The #23351 custom-op + fake-impl wrappers exist: `mla_dsa_proj.register_fake`
  (attention.py:1058) forces a fixed-length (9-tensor) straight-line output
  with `_should_use_short_mha` == False under compile — exactly #23351's
  `use_mha=False` trick to keep control flow capture-legal.
- The DSA forward already dispatches through these custom ops when
  `register_to_config` is True, which it is for DSA layers (attention.py:378,
  1306, 3459).

## There is NO DSA piecewise-exclusion construct to remove (the P1 task)

Exhaustive grep across `tensorrt_llm/` for `disabled_model_archs`,
`is_deepseek_dsa`, `is_piecewise_cuda_graph_disabled`, `DeepseekV32`,
`GlmMoeDsa` (in a piecewise context), `dsa.*exclude` etc. finds **no
exclusion in the piecewise enable path**.

- The enable gate `model_engine.py:2465` (`can_run_piecewise_cuda_graph`) is
  purely `has_ctx_requests and max(tokens) <= max_captured_num_tokens`. No
  arch guard, no DSA special-case. It is generic.
- The only `GlmMoeDsa` hits are in the `auto_deploy` modeling registry
  (`auto_deploy/models/custom/...`) — an unrelated AD model definition, not a
  piecewise disabled list.
- The `dsa.py` "excluded from CUDA-graph capture" strings (lines 472, 2606)
  are *comments* describing that Op 2 is intentionally the eager piece — i.e.
  the design is already #23351-correct, not a gap.

=> **P1 ("un-exclude DSA from piecewise") is a no-op for op-trt: there is
nothing to un-exclude.** The plan's P1 does not apply.

## The dsa.py:196 "always eager" comment is NOT a removable gap

The comment lives inside `_layersplit_read_block_ids_step` — a per-step memo
of the LayerSplit read-set. Its memo key includes
`torch.cuda.is_current_stream_capturing()`. The comment ("in practice this
path is always eager — batches with context requests never run under CUDA
graphs, and the unique() output shape is data-dependent") explains that this
memo's capture-flag key element is always False, because the read-set / unique()
computation lives in the *eager* Op 2 (`mla_dsa_attn_inplace`). It documents
*correct existing behavior of an internal memo inside the already-eager op* —
not a switch that forces the whole prefill eager. It is descriptive, not a
gate to flip.

## The REAL reason DSA prefill runs eager in the deployment

`enable_piecewise_cuda_graph` defaults to **False**
(`llmapi/llm_args.py:4292`) and is enabled only when a `TorchCompileConfig`
sets it. **The r20 prefill deploy config `deploy/disagg_pd_r20/prefill.yaml`
sets NO `torch_compile_config`/piecewise block at all** (it has
`cuda_graph_config` + `sparse_attention_config` only). So piecewise is simply
*never turned on* on the prefill worker. The full DSA-ready split-op machinery
exists but is dormant in production.

Therefore the actual lever for the validated benefit (capture the graph-safe
prefill spans, run only Op 2 eager → cut per-op launch overhead → TTFT /
prefill-throughput win) is:
  **enable piecewise on the prefill path** (set a `torch_compile_config` with
  `enable_piecewise_cuda_graph: true` + appropriate `capture_num_tokens`), then
  verify (a) the prefill spans actually capture, (b) correctness vs eager,
  (c) the launch-count / latency win — and resolve whatever capture-illegality
  surfaces at runtime (the #23351 P2-style construct-stripping), if any.

This is a re-scoping of the plan: P1 (un-exclude) is empty; the work is
"enable + make-actually-capture + gate" — which IS the spirit of the benefit.

## Capture-illegal constructs already handled vs still-open

Already handled by op-trt's design:
- Indexer top-k (`indexer_topk_prefill`/`_decode`, `cute_dsl_indexer_topk_*`),
  index-K-cache scatter/gather, `torch.unique`, data-dependent `logits.topk`,
  `.masked_fill`, chunked-prefill gathers — ALL live inside Op 2 (eager).
- `.item()` host syncs in `dsa.py` (470, 713, 1769-1969, 2342, 2994, ...) are
  in metadata `prepare()` (eager, outside the captured forward) or inside Op 2.
- CP: the r20 prefill is `context_parallel_size: 2` + LayerSplit. #23351 says
  CP is NOT supported under PCG. **This is the key open risk** — the prefill
  worker that would benefit is exactly a CP worker; enabling PCG there needs CP
  asserted-out-of / excluded-from the captured region, or a non-CP prefill
  config to demonstrate the win. MUST verify the CP prefill path is guarded.

Still-open (to confirm at runtime under capture, P2/P3):
- Whether the captured spans (projections / MoE / norms between Op 1 and Op 2,
  and across layers) actually form for a prefill batch, or whether an
  `aten.index.Tensor` / `aten.cumsum` node (the two `stop_partition` triggers
  in `piecewise_optimizer.py:266-267`) appears EARLY in the prefill forward and
  forces everything after it eager. `stop_partition` is the op-trt-specific
  subtlety the plan didn't anticipate: index/cumsum are split points that STOP
  further splitting (only attention ops continue). If prefill emits an early
  index/cumsum on the captured path, coverage collapses to ~nothing.
- `data_ptr()` compares / metadata-identity guards on the captured forward.

## FEASIBILITY VERDICT

- **Layer-level harness: BUILDABLE.** Existing unit tests
  `tests/unittest/_torch/attention/sparse/test_dsa_indexer.py` (2941 lines) and
  `test_dsa_fp4_indexer.py` directly construct `Indexer`,
  `DSAtrtllmAttentionMetadata`, `DSACacheManager` (imported from
  `_torch.attention_backend.sparse.dsa`) at real prefill shapes. A decoder-layer
  forward harness through `mla_dsa_proj`/`mla_dsa_attn_inplace` is constructible
  on the same primitives. The proof image
  `localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-aaa7e2b542b2-hisparse-current-head-proof-20260613T155354Z`
  is present on 001 for running it on GPU 1.
- **Model weights present:** `~/.cache/huggingface/hub/
  models--BlaiseAI--DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft`
  (+ GLM-4-9B DSA draft) — full-model exercise is possible if needed.
- **The true gating boundary** for the *benefit* (launch-count/latency of a
  piecewise-captured prefill vs eager) is NOT the indexer kernel — it is whether
  the piecewise PARTITIONER produces non-trivial captured spans for a DSA
  prefill graph. That requires either (a) a module-level forward of a stack of
  DSA decoder layers traced through the torch.compile backend with
  `enable_piecewise_cuda_graph=True` (mid-weight harness), or (b) the full
  serving stack with a prefill `torch_compile_config`. A single Indexer.forward
  in isolation CANNOT gate the benefit (the indexer is the eager piece by
  design; the win is in the captured spans around it).

## What P1+ becomes, given the findings

1. (was P1) — empty: no exclusion to remove.
2. Add an off-by-default `torch_compile_config` w/ `enable_piecewise_cuda_graph`
   to a prefill config (a NON-CP prefill variant first, to dodge the CP-under-PCG
   block), capture-num-tokens tuned to prefill buckets. Off-switch = simply not
   setting it (byte-identical to today; decode + non-DSA untouched).
3. Bring up the piecewise capture for a DSA prefill forward and measure whether
   spans form; if `stop_partition` (early index/cumsum) kills coverage, that is
   the real op-trt-specific fix (move/guard those nodes), analogous in spirit to
   #23351's construct-stripping.
4. Gate: prefill output bit-exact vs eager (TRUE ref) + per-op launch-count /
   per-step latency eager-vs-piecewise.
5. Guard CP-prefill out of PCG (assert not-in-piecewise on the CP path).

## Files of record
- `tensorrt_llm/_torch/compilation/piecewise_optimizer.py` (partitioner, split
  points, `stop_partition`).
- `tensorrt_llm/_torch/compilation/backend.py` (`enable_piecewise_cuda_graph`).
- `tensorrt_llm/_torch/modules/attention.py` (mla_dsa_proj / mla_dsa_attn_inplace
  custom ops + fake-impls + DSA forward dispatch at 3459).
- `tensorrt_llm/_torch/attention_backend/sparse/dsa.py` (Indexer, prefill topk
  path 4117-4260, sparse_attn_indexer 4063+, the :196 memo).
- `tensorrt_llm/_torch/pyexecutor/model_engine.py` (enable gate 2465, warmup
  capture 1560, `_filter_piecewise_capture_num_tokens` 277).
- `tensorrt_llm/llmapi/llm_args.py` (TorchCompileConfig 4288+, defaults).
- `deploy/disagg_pd_r20/prefill.yaml` (NO torch_compile block — the dormancy).

## P3 gate result + the established gating boundary (measured)

Ran in the proof image on GPU 1 (flock /tmp/gpu001_lock_b).

### What WAS measured (partition-logic gate, blaise_perf/piecewise/gate_partition.py)
The load-bearing op-trt-specific question -- does a DSA prefill graph keep
captured spans, or does stop_partition collapse coverage -- was gated by
replicating the EXACT node-classification loop of piecewise_optimizer against
a DSA decoder-layer prefill op skeleton (mla_dsa_proj -> mla_dsa_attn_inplace
between pointwise/MoE spans):
  - clean DSA prefill skeleton: captured_spans=2, excluded_eager=1
    (pre-attn proj span + post-attn MoE span around the single eager
    mla_dsa_attn_inplace op). Coverage forms; NO collapse. PASS.
  - early raw aten.cumsum injected before attention: captured_spans=1,
    stop_partition tripped -- confirms the coverage risk IS the presence of a
    raw aten.index/aten.cumsum on the captured path. The real DeepSeek-V3
    decoder forward (modeling_deepseekv3.py:1709 DeepseekV3DecoderLayer.forward
    / forward_MoE) has NO such raw op on the captured path (the only
    cumsum+index is in DeepseekV3MTPHead.get_last_token_states:758, a SEPARATE
    @torch.compile region after the decoder stack, not in the per-layer span).
Also verified in-image: both custom ops registered, mla_dsa_attn_inplace IS a
registered split point, the toolchain imports cleanly.

### What was NOT measured, and WHY (the honest gating boundary)
A TRUE full-forward bit-exactness gate (eager DSA layer vs piecewise-captured
DSA layer) is NOT runnable on 001:
  - The only DSA-architecture model on the box is
    BlaiseAI/DeepSeek-V3.2-REAP-345B-...-NVFP4-NextN-Graft, and its checkpoint
    is metadata-only (18M: config.json + index.json; the .safetensors weight
    shards are absent). Loading the full DSA model is impossible without
    downloading ~170GB of NVFP4 weights, which the VM cannot reach and which
    would exceed any bounded run.
  - The other complete checkpoint, GLM-4-9B-0414-FP8-DeepSeekV32-OMP (12G,
    real shards), is Glm4ForCausalLM / model_type glm4 with NO DSA/sparse
    keys -- it is the SMC draft, a plain Glm4, and exercises the NON-DSA
    attention path. It does NOT touch mla_dsa_proj/mla_dsa_attn_inplace, so it
    cannot gate the DSA prefill path.
  - A bespoke single-DSA-layer forward harness under torch.compile is
    buildable in principle (test_dsa_indexer.py shows the Indexer/metadata/
    DSACacheManager scaffolding) but requires replicating the entire DSA
    metadata.prepare() + FP8/FP4 indexer + paged-KV pipeline with correct
    quantized inputs under capture -- a large harness, and still only a
    correctness gate (the perf benefit needs the real model under concurrency).

### Boundary statement
The DSA-prefill piecewise enable is partition-correct and machinery-validated
(the partitioner keeps captured spans around the eager indexer op; the enable
config is valid; off == baseline byte-identical). It is perf-unverified and
full-forward-correctness-unverified on 001 because no runnable DSA model with
weights exists here. The remaining gate (bit-exact DSA prefill output
eager-vs-piecewise + launch-count/TTFT win under concurrency) must run where the
345B NVFP4 weights are resident (the GPU VM with the real checkpoint), via the
opt-in deploy/disagg_pd_r20/prefill_piecewise.yaml on a non-CP prefill worker.

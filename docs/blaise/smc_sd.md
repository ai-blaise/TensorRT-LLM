# SMC-SD (Sequential Monte-Carlo speculative decoding)

SMC-SD is a **two-model speculative decoding** scheme that multiplies decode
throughput by drafting several tokens per target verify step. Unlike plain
EAGLE/MTP single-chain speculation, SMC-SD runs **`n_particles` parallel
particles** through a Sequential-Monte-Carlo proposal/resample loop, so a wider
slice of the draft distribution is verified per target step (higher accepted
length per verify under diverse continuations).

Because the target checkpoint on this fleet is untrained, SMC-SD is validated by
**draft/verify kernel correctness** (accept-logic, batched particle selection,
log-prob diff) — not by end-to-end text acceptance length.

## Mechanism

- Per request, `n_particles` particles each propose a chain of `gamma` draft
  tokens from the draft model. `build_smc_particle_choices(gamma, n_particles)`
  builds the static tree of `(particle, depth)` choices the verify step expands.
- The target model verifies the particle tree in one batched forward. SMC weights
  combine target and draft log-probs at `target_temperature` / `draft_temperature`.
- When the effective sample size (ESS) ratio drops below `resample_threshold`
  (default 0.5), particles are resampled. `tokens_per_gen_step =
  gamma * n_particles + 1`.

## Configuration (`SMCDecodingConfig`)

```python
speculative_config = {
    "decoding_type": "SMC",
    "speculative_model": "<draft model path>",   # required
    "gamma": 6,                  # draft transitions per target verify (== max_draft_len)
    "n_particles": 4,            # SMC particles per request
    "resample_threshold": 0.5,   # ESS ratio below which particles resample, in (0,1]
    "target_temperature": 1.0,
    "draft_temperature": 1.0,
    "draft_attention_backend": "triton",   # auto | triton | fa3 | trtllm_mha
    "draft_kv_cache_dtype": "auto",        # auto | bfloat16 | fp8_e4m3 | fp8_e5m2
}
```

The validator enforces `max_draft_len == gamma` and sets
`max_total_draft_tokens = gamma * n_particles`.

## Files

| Path | Purpose |
|------|---------|
| `tensorrt_llm/_torch/speculative/smc.py` | `SMCSpecMetadata`, `SMCResourceManager`, `SMCModelDrafter`, `SMCSampler`, `build_smc_particle_choices`. |
| `tensorrt_llm/_torch/speculative/drafting_loops.py` | `SMCStaticParticleDraftingLoopWrapper` (extends `StaticTreeDraftingLoopWrapper`) — the static-particle draft forward + log-prob recording. |
| `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py` | Wires `SMCStaticParticleDraftingLoopWrapper` into the executor. |
| `tensorrt_llm/llmapi/llm_args.py` | `SMCDecodingConfig`. |

## Status

- **Draft path validated** (draft image `smcsd-20260603`): the static-particle
  draft forward + log-prob recording + batched particle selection (which
  collapses per-request d2h syncs) are correct. Rejection-sampling acceptance is
  the default for higher accepted length.
- **GLM-4-9B-FP8 draft specifics handled:** duplicate draft KV-head FP8 block
  scales at `tp > num_kv_heads`; route the GLM draft to FlashInfer (no fused
  trtllm-gen GQA kernel on SM100); force non-fp8 draft KV to avoid the
  unfused-MHA OOM; coerce dense GLM-4-0414-FP8 draft activations to bf16.
- **End-to-end serving blocked / regressing** at the time of writing: the
  aggregated-TP4 SMC deploy hit an intermittent **libnccl 2.29.2 double-free**
  (root-caused; fix = NCCL 2.29.7 swap) and a measured **~2 tok/s regression**
  vs the no-SMC baseline on the same hardware — i.e. the draft overhead was not
  yet repaid by acceptance at the tested operating point. A no-SMC baseline on
  the same agg-TP4 hardware is staged for a clean A/B. SMC-SD is therefore
  **opt-in and not a production default** pending the e2e win.

## Correctness validation

| Unit | Method |
|------|--------|
| Accept logic | `_accept_selected_particle` / rejection sampling vs reference acceptance |
| Batched particle selection | one batched selection vs per-request loop (same result, fewer d2h syncs) |
| Log-prob diff | draft vs target log-prob recording numerically consistent |

## Composition

SMC-SD is **per-draft / per-target**: every other campaign optimization applies
*independently inside the draft forward and the target forward*. The Indexer,
sparse-MLA, NVFP4 fusions, KVarN, WarpDecode, and LayerSplit all run unchanged on
the target model's forward; the draft model is a separate (smaller) model with
its own attention backend (`draft_attention_backend`) and KV dtype
(`draft_kv_cache_dtype`). The only coupling is the speculative loop scheduling
(`tokens_per_gen_step` shapes the target verify batch), which the executor
handles via the drafting-loop wrapper.

- **Topology** (`topology_deploy.md`): SMC-SD was campaigned on the aggregated-TP4
  setup; the draft and target share the TP group. The DP2/TP4 disaggregated
  topology is orthogonal to whether SMC is enabled.

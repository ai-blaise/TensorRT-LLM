# Moondream SMC overlap audit

Audit date: 2026-06-06

Scope: commit `f98885b` and current `origin/op-trt` head `a416a8f`. This audit
checks the SMC-SD integration against the Moondream/Photon pipeline mechanics:
pinned host buffers, event-gated D2H copy, deferred commit, zombie release,
prefill sharing the overlap loop, and SMC draft payload preservation.

## Proven in code and focused tests

- **SMC overlap admission:** `SpeculativeDecodingMode.SMC` is admitted through
  `has_draft_model()`; the executor creator has no SMC-specific force-disable
  path.
- **Pinned/evented draft D2H:** static draft overlap allocates an explicit
  pinned CPU `new_draft_tokens` landing buffer and records a CUDA event after
  the non-blocking copy.
- **SMC payload preservation:** `SMCModelDrafter` overrides the generic static
  draft overlap payload so SMC carries `draft_token_log_probs` through delayed
  commit and does not require or synthesize generic `draft_logits`.
- **Deferred commit ordering:** previous SMC draft results are consumed only
  after the current draft forward is launched; `process_static_draft_outputs`
  waits on the sample event before reading the pinned host token buffer.
- **Zombie / abort safety:** SMC draft commit skips any target request that is
  not `GENERATION_IN_PROGRESS`; SMC resource state is retained through
  `GENERATION_TO_COMPLETE` and freed only on `GENERATION_COMPLETE`.
- **Prefill-in-pipeline:** context-init requests are skipped by SMC draft commit,
  so prefill can ride the overlap loop without stale particle mutation.
- **Metadata composability:** focused tests verify SMC overlap commit does not
  mutate request-pinning/disagg metadata, KVarN metadata, or SMC group identity.

Focused tests added/maintained in
`tests/unittest/_torch/speculative/hw_agnostic/test_smc.py`:

- `test_smc_mode_admits_overlap_scheduler`
- `test_smc_resource_manager_keeps_zombie_until_complete`
- `test_smc_overlap_static_draft_commit_uses_evented_host_tokens`
- `test_smc_overlap_pack_does_not_require_generic_draft_logits`
- `test_smc_overlap_static_draft_commit_skips_prefill_context`
- `test_smc_overlap_static_draft_commit_skips_aborted_or_zombie_request`
- `test_smc_overlap_commit_preserves_disagg_pin_and_kvarn_metadata`

## Remaining blockers / not yet production-proven

- **Live E2E is still required.** These tests prove scheduler/drafter ordering
  and metadata safety, not a full SMC-SD deployment with LayerSplit prefill,
  dense MLA KVarN, WarpDecode TP/EP, HISA/indexcache, CUDA graphs, and real
  disaggregated KV transfer.
- **Standard request pinning is not implemented by this patch.** The current
  tree has disaggregated request ids, KV transfer sessions, and rank consensus,
  but this audit did not find a complete deterministic producer/consumer
  request-affinity layer that pins remote KV block ownership across prefill and
  decode. Treat that as a pre-A/B blocker, especially with KVarN dense block
  ownership and LayerSplit.
- **Generic GQA KVarN remains documented as blocked.** Dense MLA KVarN
  composability is covered by metadata preservation here; SMC-SD GQA KVarN
  still needs its own storage/read kernel path before it can be claimed.
- **MORI/NIXL/Mooncake transport wins are not proven here.** This audit does not
  replace the separate transport benchmark/integration gate.

## Rollout log signals to check

After building an image that includes this audit patch and rolling it to the
SMC-SD decode worker, verify:

- Engine args show `disable_overlap_scheduler: False` with
  `speculative_config.decoding_type: "SMC"`.
- No warning says `Disable overlap scheduler for speculation mode SMC`.
- No SMC error says `SMC-SD requires target token probabilities` or
  `SMC-SD requires selected draft token log probabilities`.
- Logs show the SMC static-particle draft path running, not generic greedy
  fallback. Absence of `draft_logits` fallback errors is required.
- Generation requests that finish under overlap do not release KV/slot early;
  look for clean completion without zombie corruption, stale guided-mask errors,
  or KV-cache transfer/cancel errors.
- Prefill remains schedulable in the overlap loop and decode remains ready after
  warmup/autotune.
- KVarN remains selected for dense MLA (`mla_latent_kv_dtype="kvarn_k2v2"`,
  `mla_latent_kv_amortize=True`) and WarpDecode remains forced with TP/EP active
  and no kernel-backend fallback.

Do not mark this production-complete until live E2E passes with the full custom
stack and request-pinning/transport gates satisfied.

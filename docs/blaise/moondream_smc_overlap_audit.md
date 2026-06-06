# Moondream SMC overlap audit

Audit date: 2026-06-06

Scope: current `origin/op-trt`. This audit checks the SMC-SD integration
against the Moondream/Photon pipeline mechanics: pinned host buffers,
event-gated D2H copy, deferred commit, zombie release, prefill sharing the
overlap loop, SMC draft payload preservation, and the non-MORI request-pinning
gate that prevents unknown-DP KV receive fanout before A/B.

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
- **Non-MORI request pinning:** `OpenAIDisaggregatedService` now fails closed
  unless `ctx_request_id`, `disagg_request_id`, and `ctx_dp_rank` are present
  and tied to the expected disaggregated request id before decode asks for KV.
  This prevents the Python/native ADP `REQUEST_DATA` broadcast path from being
  reachable via an unpinned request; the current r20 canary uses C++ UCX, whose
  exact rank fanout is formatter/layout driven, but the same stable
  `ContextPhaseParams` metadata is passed to the executor.

Focused tests added/maintained in
`tests/unittest/_torch/speculative/hw_agnostic/test_smc.py`:

- `test_smc_mode_admits_overlap_scheduler`
- `test_smc_resource_manager_keeps_zombie_until_complete`
- `test_smc_overlap_static_draft_commit_uses_evented_host_tokens`
- `test_smc_overlap_pack_does_not_require_generic_draft_logits`
- `test_smc_overlap_static_draft_commit_skips_prefill_context`
- `test_smc_overlap_static_draft_commit_skips_aborted_or_zombie_request`
- `test_smc_overlap_commit_preserves_disagg_pin_and_kvarn_metadata`

Focused request-pinning tests are in
`tests/unittest/disaggregated/test_openai_disagg_service.py`; deployment gates
are in `tests/unittest/disaggregated/test_disagg_pd_r20_gates.py`.

## Remaining blockers / not yet production-proven

- **Live E2E is still required.** These tests prove scheduler/drafter ordering
  and metadata safety, not a full SMC-SD deployment with LayerSplit prefill,
  dense MLA KVarN, WarpDecode TP/EP, HISA/indexcache, CUDA graphs, and real
  disaggregated KV transfer.
- **Live E2E request-pinning proof is still required.** Unit tests and source
  audit prove fail-closed metadata handling; the rollout must still show paired
  `disagg request pin established` / `disagg request pin cleared` logs and no
  Python/native `ADP broadcast path` logs under target traffic.
- **Generic GQA KVarN remains pre-production.** Dense MLA KVarN composability is
  covered by metadata preservation here; the GQA path now has packed-record
  primitives and HF config admission work, but still needs full production E2E
  proof before it can replace dense MLA KVarN in this canary.
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

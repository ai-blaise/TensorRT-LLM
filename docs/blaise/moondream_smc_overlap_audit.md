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
  reachable via an unpinned request; the current r20 pre-A/B gate uses the C++
  NIXL transceiver, whose exact rank fanout is formatter/layout driven, and the
  same stable `ContextPhaseParams` metadata is passed to the executor.
- **Overlay/source parity:** the r20 overlay carries the request-pinning service,
  OpenAI client/server trace points, OpenAI protocol conversion, and
  `DisaggregatedParams` dataclass together, so `ctx_dp_rank` serialization is
  not dependent on whatever happens to be present in the base runtime image.
- **Moondream/SMC overlay source parity:** the r20 overlay now explicitly
  carries `sampler.py`, `guided_decoder.py`, `speculative/interface.py`,
  `speculative/model_drafter.py`, `speculative/smc.py`, and
  `speculative/drafting_loops.py`, and the r20 gate test asserts those COPY
  entries so SMC overlap/pinning cannot silently disappear in a thin overlay
  rebuild.
- **Live proof gate:** `deploy/disagg_pd_r20/smoke_request_pinning.sh` is the
  canonical pre-A/B smoke. It refuses to run before DGD/endpoints are ready,
  sends one normal request and one early-closed stream, then requires matching
  frontend pin lifecycle, frontend outbound generation metadata, prefill
  context receipt, decode generation receipt, no bad fallback/overlap logs, and
  zero prefill/decode restarts. In `SMC_GATE_MODE=required`, the SMC handoff
  marker must also correlate to the same pinned request id and completed-prefill
  `ctx_dp_rank` / `ctx_info_endpoint` that Dynamo emitted outbound to decode;
  unrelated SMC log lines cannot satisfy the gate.
- **SGLang/GLM kernel assumptions are below the scheduler contract.** The GLM
  draft path can choose FlashInfer decode or the SGLang-derived Triton prefill
  shim for draft attention, but the Moondream scheduler only consumes the
  wrapper contract: `SMCStaticParticleDraftingLoopWrapper` must return
  `draft_token_log_probs`, `SMCModelDrafter` must carry those through delayed
  commit, and the commit must use the evented pinned host token buffer. If a
  remaining GLM kernel port changes logits layout, draft attention backend, or
  warmup behavior, it must preserve that wrapper contract and cannot bypass the
  request-pin validation or unpinned-commit guard.
- **Runtime-focused proof:** the CPU-safe SMC/Moondream drafter tests were run
  inside the TRT-LLM runtime image with the patched `smc.py` overlaid into
  site-packages. This caught the missing `logger` import for the live handoff
  marker and now proves the evented pinned commit path, generation-only request
  pin validation before mutation, valid pin metadata preservation, and the
  fail-closed unpinned fallback guard without launching model traffic.

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
  `disagg request pin established` / `disagg request pin cleared` logs,
  `disagg request pin outbound` from the frontend, `disagg request pin received`
  in prefill and decode, and no Python/native `ADP broadcast path` logs under
  target traffic.
- **Generic GQA KVarN remains pre-production.** Dense MLA KVarN composability is
  covered by metadata preservation here; the GQA path now has packed-record
  primitives and HF config admission work, but still needs full production E2E
  proof before it can replace dense MLA KVarN in this canary.
- **Transport wins are not proven here.** This audit does not replace the NIXL
  pre-A/B gate or the later UCX/Mooncake/MORI A/B transport benchmark.
- **Current live SMC blocker:** SMC-SD decode is no longer on the immediate gate
  path. The next live proof should validate LayerSplit, NIXL transfer, and
  request pinning first, then re-enable SMC-SD for A/B once the GLM draft-kernel
  path is stable.

## Rollout log signals to check

After building an image that includes this audit patch and rolling it to the
SMC-SD decode worker, verify:

- Engine args show `disable_overlap_scheduler: False` with
  `speculative_config.decoding_type: "SMC"`.
- `deploy/disagg_pd_r20/smoke_request_pinning.sh` passes without weakening its
  fail-closed checks.
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
- For transport variants, keep the same request-pinning proof (`rid`,
  `ctx_dp_rank`, cleanup on close/abort) and repeat it under the UCX baseline,
  Python/C++ NIXL, Mooncake, and MORI-IO images before accepting any throughput
  win. Request pinning is service/protocol-level and should be transport
  independent; transfer-backend replacement is only accepted after E2E proves
  the backend wrapper preserves the same metadata and cleanup behavior.

Do not mark this production-complete until live E2E passes with the full custom
stack and request-pinning/transport gates satisfied.

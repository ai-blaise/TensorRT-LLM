# Moondream-style pipelined decoding (forward-now-sample-later) — verification

op-trt's decode already implements the "pop the GPU bubble" pipelined-decoding
pattern: forward-now-sample-later overlap via `_executor_loop_overlap`
(`tensorrt_llm/_torch/pyexecutor/py_executor.py`), with the sampled-token D→H copy
on a separate side-stream anchored to a CUDA event, pinned host buffers, and a
finalize/refcount path that releases KV/slot only after results are read
(zombie/EOS-mid-flight safe).

This records the round-20 rigor + max-optimization sweep confirming it is **correct
and maximally optimized as shipped** (op-trt `a974423f0`), so a future change does
not regress it or re-derive the analysis.

## Correctness — 5 edge cases (all CORRECT)

1. **Guided/constrained mask — no synchronous D→H.** The mask build is event-gated
   (`guided_decoder.py:357`, token_event/bitmask_event, no `.synchronize()`);
   `CapturableGuidedDecoder` copies into a pinned buffer (`:477`) and the `.tolist()`
   runs inside an `@hostfunc` CUDA callback (`:543`) gated by token_event. (A stray
   sync here would negate the entire overlap — the moondream caught-bug. It is absent.)
2. **Zombie / EOS mid-flight.** `GENERATION_TO_COMPLETE` latch
   (`py_executor.py:4150-4153/4179-4182`) holds KV/slot; terminate happens in
   `_handle_responses` (`:4691`) after enqueue, N-1 deferred; `update_resources`
   skips COMPLETE/CONTEXT_INIT; zombie skip at `sampler.py:3605`.
3. **Prefill-in-pipeline.** A new prefill (`py_batch_idx is None`) uses host tokens
   (`model_engine.py:2645-2655`), never the in-flight device tensor; N-1 device
   tokens scatter only into the disjoint `input_ids_cuda[num_tokens:]`.
4. **Batch ramp (c1→c16→c32).** Static buffers at `max_num_tokens`, only `[:n]`
   slices vary; the incremental fast-path is disabled when the request set changes;
   CUDA-graph pads to captured sizes via dummy requests.
5. **Slot reuse.** `SlotManager` free/add; `setup_sampler_step` resets per-slot state
   on reuse (finish-reason, beam/logprob buffers, `end_ids`); `new_tokens` overwritten
   write-before-read.

Unit tests (B200 / SM100): `test_py_executor` + `test_request_utils` = 38 passed / 0
failed; `test_torch_sampler` = 262 passed / 162 skipped (skips are `@force_ampere`
SM90-only).

## Maximum optimization

- **No residual D→H sync on the overlap decode hot path.** Every real sync is warmup,
  perf-timing (`enable_timing`), capture-guarded (`is_current_stream_capturing()`),
  HISA-off-at-prod, prefill-once, or the non-overlap PP loop. Metadata
  `.item()/.tolist()` are all on host (`device='cpu'`) tensors. The one mandatory
  sync (`sampler.py:3573`) is the deferred-sample gate that *defines* the overlap.
- `disable_overlap_scheduler` defaults to `False` (`llm_args.py:4128`) — overlap ON,
  optimal. Composes safely with WarpDecode (orthogonal fused-MoE kernel) and the DSA
  indexer (native sync-free path).
- Residual ~20 ms/tok decode overhead is kernel-launch/execution-bound (small
  DSA/MoE/MLA kernels), not a CPU-sync bubble — addressed by CUDA graphs +
  megakernel/indexer-floor work, outside the overlap scheduler's scope.

## Interaction with SMC-SD (important)

SMC-SD now participates in the overlap scheduler instead of force-disabling it.
`SpeculativeDecodingMode.support_overlap_scheduler()` admits SMC via the
two-model `has_draft_model()` path, and the SMC static-draft overlap payload is
SMC-aware: it preserves `draft_token_log_probs` plus the event-gated pinned host
copy of `new_draft_tokens` rather than assuming generic `draft_logits`.

The required invariants are:

- **No greedy fallback.** SMC keeps particle-weighted verification;
  `SMCModelDrafter.process_static_draft_outputs()` consumes SMC draft log-probs,
  not generic draft logits.
- **Ping-pong/deferred commit safe.** The delayed previous-draft commit waits on
  `SampleState.sampler_event` before reading the pinned host token buffer, so
  draft tokens cannot be read before the side-stream D2H copy lands.
- **Zombie safe.** `SMCResourceManager.update_resources()` frees particle state
  only after `GENERATION_COMPLETE`; already-finalized requests in the next
  in-flight batch are skipped by sampler update and released on the late resource
  update, matching the finalize-early/release-late rule.
- **Prefill-in-pipeline.** Context requests remain admitted through the same
  `_executor_loop_overlap` path; SMC draft preparation still skips context-init
  requests until generation state, so prefill can occupy a slot without stale SMC
  particle mutation.

## Verdict

Correctness is now implemented for SMC-aware Moondream-style overlap at the
scheduler/drafter boundary. Keep the focused SMC overlap tests green before
starting A/B sweeps, and require runtime E2E on the SMC-SD deployment before
claiming production completion. See `moondream_smc_overlap_audit.md` for the
current proven-vs-E2E checklist and rollout log signals.

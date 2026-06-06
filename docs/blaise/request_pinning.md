# Disaggregated request pinning and Moondream overlap gates

## Purpose

The non-MORI P/D path must bind each generation request to the exact prefill
producer that owns its KV transfer metadata. The request is not allowed to fall
back to an unknown-DP broadcast path before A/B testing.

## Runtime invariants

- `disagg_request_id` is the stable request identity across context and
  generation.
- `ctx_request_id` is present before decode asks for KV.
- `ctx_request_id` and `disagg_request_id` must both match the expected
  disaggregated request id. A present-but-mismatched id is a hard error because
  it can associate decode with the wrong prefill producer.
- `ctx_dp_rank` is present before decode asks for KV. Missing `ctx_dp_rank` is
  a hard error because it would otherwise broadcast `REQUEST_DATA` across
  context DP groups.
- `ctx_info_endpoint`, when provided by the transceiver runtime, is treated as
  request-local transfer metadata. Context response metadata wins over static
  server metadata; static server metadata only backfills missing fields.
- Request pins are cleared on normal completion, error, generation-first
  validation failure, generation-first context errors, and when a streaming
  response is consumed or closed.

## KV transfer integration

- OpenAI protocol serialization preserves `ctx_request_id`,
  `disagg_request_id`, `ctx_dp_rank`, and `ctx_info_endpoint`.
- `DisaggregatedParams.get_context_phase_params()` forwards those fields into
  executor `ContextPhaseParams`; it prefers `disagg_request_id` as the executor
  request id so context and generation use the same stable request identity.
- The Python/native NIXL receiver has an explicit ADP broadcast branch when
  `ctx_dp_rank is None`. The service-level fail-closed gate prevents normal
  non-MORI traffic from reaching that branch without an explicitly proven
  override.
- The r20 canary uses the C++ UCX transceiver. Its receive fanout is computed
  from `DataTransceiverState` and MLA/cache formatter rank layout; request
  pinning still supplies the stable producer request id and transfer metadata
  to the executor before `requestAndReceive*` starts.

## Moondream-style overlap invariants

- Prefill uses the same overlap scheduler pipeline as decode:
  `disable_overlap_scheduler: false`.
- SMC-SD decode is allowed to use overlap; it must preserve
  `draft_token_log_probs` and event-gated pinned host draft tokens. It must not
  fall back to greedy draft verification.
- The delayed commit waits on `SampleState.sampler_event` before reading pinned
  host tokens.
- Zombie requests keep SMC particle state until `GENERATION_COMPLETE`.

## R20 canary gates

- Prefill config:
  - `cp_config.cp_type: LAYERSPLIT`
  - `sparse_attention_config.layersplit_enabled: true`
  - `sparse_attention_config.layersplit_all_cp_ranks_transfer: true`
  - `disable_overlap_scheduler: false`
  - `mla_latent_kv_dtype: kvarn_k2v2`
- Decode config:
  - `disable_overlap_scheduler: false`
  - `speculative_config.decoding_type: SMC`
  - `moe_config.backend: WARPDECODE`
  - `warp_decode.policy: force`
  - `warp_decode.allow_parallelism_fallback: false`
  - `mla_latent_kv_dtype: kvarn_k2v2`
- No `cp_type: HELIX` or implicit HELIX fallback.
- UCX remains explicit only for the current non-MORI baseline image. NIXL,
  Mooncake, or MORI may replace it only after wrapper availability and E2E
  throughput wins are proven.

## Rollout proof points

Collect these from the live rollout before A/B:

- Frontend/request logs include `disagg request pin established` with
  `ctx_server`, `ctx_dp_rank`, and `gen_server`.
- Matching `disagg request pin cleared` appears for every established pin.
- No `Request pinning requires ctx_dp_rank` errors.
- No `ADP broadcast path` logs from native transfer in the target canary.
- Prefill/decode engine args show `disable_overlap_scheduler: False`.
- Prefill engine args show `cp_config={'cp_type': 'LAYERSPLIT'}` and
  `layersplit_enabled: True`.
- Decode engine args show `decoding_type='SMC'`, WarpDecode enabled with
  `policy='force'`, and no backend fallback.
- KVarN dense MLA shows `mla_latent_kv_dtype='kvarn_k2v2'` and amortized restore.
- Worker pods have zero restarts through smoke and 16-concurrency warmup.

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
- The r20 canary uses the C++ NIXL transceiver as the pre-A/B gate. Its receive
  fanout is computed from `DataTransceiverState` and MLA/cache formatter rank
  layout; request pinning still supplies the stable producer request id and
  transfer metadata to the executor before `requestAndReceive*` starts. UCX is
  retained only as an A/B comparison candidate.
- The r20 canary emits request-pinning trace logs at both boundaries:
  `disagg request pin outbound` from the frontend's OpenAI client before it
  sends context/generation requests, and `disagg request pin received` from the
  worker OpenAI server after prefill/decode deserialize the request body.

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
  - `sparse_attention_config.layersplit_owner_local_alloc: true`
  - `cache_transceiver_config.backend: NIXL`
  - `disable_overlap_scheduler: false`
  - `mla_latent_kv_dtype: kvarn_k2v2`
- Decode config:
  - `disable_overlap_scheduler: false`
  - `speculative_config.decoding_type: SMC` for the full production proof; use the explicit `SMC_GATE_MODE=deferred` smoke only while the SMC-SD kernel is behind the NIXL/LayerSplit gate.
  - `moe_config.backend: WARPDECODE`
  - `warp_decode.policy: force`
  - `warp_decode.allow_parallelism_fallback: false`
  - `mla_latent_kv_dtype: kvarn_k2v2`
- No `cp_type: HELIX` or implicit HELIX fallback.
- NIXL is explicit for the current non-MORI pre-A/B baseline image. UCX,
  Mooncake, and MORI are A/B-phase comparison candidates rather than gates.

## Rollout proof points

Collect these from the live rollout before A/B:

- Frontend/request logs include `disagg request pin established` with
  `ctx_server`, `ctx_dp_rank`, and `gen_server`.
- Frontend logs include `disagg request pin outbound` for a
  `request_type=generation_only` request with matching `disagg_request_id` and
  non-null `ctx_dp_rank`.
- Prefill logs include `disagg request pin received` for
  `request_type=context_only`.
- Decode logs include `disagg request pin received` for
  `request_type=generation_only` with matching `disagg_request_id` and
  `ctx_dp_rank`.
- Matching `disagg request pin cleared` appears for every established pin.
- No `Request pinning requires ctx_dp_rank` errors.
- No `ADP broadcast path` logs from native transfer in the target canary.
- Prefill/decode engine args show `disable_overlap_scheduler: False`.
- Prefill engine args show `cp_config={'cp_type': 'LAYERSPLIT'}` and
  `layersplit_enabled: True`.
- Full production proof: decode engine args show `decoding_type='SMC'`, WarpDecode enabled with
  `policy='force'`, and no backend fallback. The temporary NIXL/LayerSplit smoke may run with `SMC_GATE_MODE=deferred`; that mode does not clear the SMC-SD A/B item.
- KVarN dense MLA shows `mla_latent_kv_dtype='kvarn_k2v2'` and amortized restore.
- Worker pods have zero restarts through smoke and 16-concurrency warmup.

## Reproducible live smoke

Run this only after the DGD is ready and the decode service has an endpoint. The
script refuses to run otherwise. The first request proves the normal close path;
the second opens a stream and closes it early to prove pin cleanup/abort safety
without putting load on the canary.

```bash
deploy/disagg_pd_r20/smoke_request_pinning.sh

# Temporary critical-path smoke while SMC-SD is deferred behind NIXL/LayerSplit:
SMC_GATE_MODE=deferred deploy/disagg_pd_r20/smoke_request_pinning.sh
```

Equivalent manual command sequence:

```bash
KC='sudo -E /usr/local/bin/k3s kubectl -n dynamo-system'
DGD=topo-c1-dp2tp4-disagg-r20
MODEL=BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft

$KC get dgd "$DGD"
$KC get endpoints "${DGD}-decode" "${DGD}-prefill" "${DGD}-frontend"
START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
FE=$($KC get pods -o name | grep "${DGD}-0-frontend" | tail -1)

$KC exec "$FE" -- python3 - <<PY
import json, urllib.request
payload = {
    "model": "$MODEL",
    "prompt": "Request pinning smoke. Count to five.",
    "max_tokens": 32,
    "temperature": 0,
    "stream": False,
}
req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
print(urllib.request.urlopen(req, timeout=300).read().decode()[:1000])
PY

$KC exec "$FE" -- python3 - <<PY
import json, urllib.request
payload = {
    "model": "$MODEL",
    "prompt": "Streaming request pinning abort smoke. Continue briefly.",
    "max_tokens": 128,
    "temperature": 0,
    "stream": True,
}
req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
response = urllib.request.urlopen(req, timeout=300)
print(response.readline().decode(errors="ignore")[:500])
response.close()
PY

DEC=$($KC get pods -o name | grep "${DGD}-0-decode" | tail -1)
PRE=$($KC get pods -o name | grep "${DGD}-0-prefill" | tail -1)

$KC logs "$FE" --since-time="$START" \
  | egrep -i 'disagg request pin established|disagg request pin cleared|disagg request pin outbound|ctx_dp_rank|Request pinning requires|abort|stream'
$KC logs "$DEC" --since-time="$START" \
  | egrep -i 'disagg request pin received|SMC|overlap|Disable overlap|HELIX|fallback|kvarn|WARPDECODE|illegal|Traceback|ERROR'
$KC logs "$PRE" --since-time="$START" \
  | egrep -i 'disagg request pin received|LAYERSPLIT|HELIX|kvarn|ctx_dp_rank|transfer|ERROR|Traceback'
```

Pass criteria:

- Each `disagg request pin established` has a matching
  `disagg request pin cleared` for the same `rid`, including the early stream
  close.
- The established pin includes `ctx_server`, `ctx_dp_rank`, and `gen_server`.
- Frontend has matching `disagg request pin outbound` for generation; prefill
  has matching context-only `disagg request pin received`; decode has matching
  generation-only `disagg request pin received` with the same `rid` and
  `ctx_dp_rank`.
- No log contains `Request pinning requires`, `ctx_dp_rank is None`, HELIX
  selection, SMC overlap disable, WarpDecode backend fallback, or CUDA illegal
  memory access.
- Decode and prefill remain ready with zero restarts after the two requests.

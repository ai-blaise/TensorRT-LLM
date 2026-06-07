#!/usr/bin/env bash
set -euo pipefail

DGD="${DGD:-topo-c1-dp2tp4-disagg-r20}"
MODEL="${MODEL:-BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft}"
KC="${KC:-sudo -E /usr/local/bin/k3s kubectl -n dynamo-system}"

# Set REQUIRE_DYNAMO_PIN_MARKERS=0 only for a router-selector dry run against an
# image that has not yet rebuilt Dynamo with the fail-closed marker patch. The
# pre-A/B gate must leave this at 1.
REQUIRE_DYNAMO_PIN_MARKERS="${REQUIRE_DYNAMO_PIN_MARKERS:-1}"
REQUIRE_POSITIVE_TRANSFER_METRICS="${REQUIRE_POSITIVE_TRANSFER_METRICS:-1}"
REQUIRE_ABORT_CLEANUP_MARKER="${REQUIRE_ABORT_CLEANUP_MARKER:-1}"
SMC_GATE_MODE="${SMC_GATE_MODE:-deferred}"
SMOKE_TMPDIR=""

cleanup() {
  if [[ -n "${SMOKE_TMPDIR:-}" ]]; then
    rm -rf "$SMOKE_TMPDIR"
  fi
}

die() {
  echo "request-pinning smoke failed: $*" >&2
  exit 1
}

need_ready() {
  local ready
  ready="$($KC get dgd "$DGD" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || true)"
  [[ "$ready" == "True" ]] || die "$DGD is not Ready (Ready=${ready:-missing})"

  local svc endpoint
  for svc in "${DGD}-frontend" "${DGD}-prefill" "${DGD}-decode"; do
    endpoint="$($KC get endpoints "$svc" -o jsonpath='{.subsets[0].addresses[0].ip}' 2>/dev/null || true)"
    [[ -n "$endpoint" ]] || die "$svc has no ready endpoint"
  done
}

pod_for() {
  local component="$1"
  $KC get pods -o name | grep "${DGD}-0-${component}" | tail -1
}

restart_count() {
  local pod="$1"
  $KC get "$pod" -o jsonpath='{range .status.containerStatuses[*]}{.restartCount}{"\n"}{end}' \
    | awk '{sum += $1} END {print sum + 0}'
}

require_runtime_nixl_gate() {
  local pre="$1" dec="$2"
  local pod env_dump log_dump

  for pod in "$pre" "$dec"; do
    env_dump="$($KC get "$pod" -o jsonpath='{range .spec.containers[*].env[*]}{.name}{"="}{.value}{"\n"}{end}' 2>/dev/null || true)"
    ! grep -Eq '^TRTLLM_USE_(UCX|MOONCAKE|MPI)_KVCACHE=1$' <<<"$env_dump" \
      || die "$pod has a legacy env backend override that conflicts with the NIXL gate"

    log_dump="$($KC logs "$pod" 2>/dev/null || true)"
    grep -q 'Initializing NIXL Connect' <<<"$log_dump" || die "$pod did not initialize NIXL Connect"
    grep -Eq "cache_transceiver_config.*backend.*NIXL|cache_transceiver_config: \{'backend': 'NIXL'" <<<"$log_dump" \
      || die "$pod logs do not prove cache_transceiver_config.backend=NIXL"
    ! grep -Eq "cache_transceiver_config.*backend.*UCX|cache_transceiver_config: \{'backend': 'UCX'|Using UCX kv-cache transceiver" <<<"$log_dump" \
      || die "$pod selected UCX cache transceiver in logs"
    grep -q 'OPTRT_LAYERSPLIT_XFER_DEBUG' <<<"$log_dump" || die "$pod missing LayerSplit transfer debug proof"
    grep -q 'global_layers=61' <<<"$log_dump" || die "$pod did not advertise global_layers=61 to CacheTransceiver"
  done

  grep -q 'transfer_attr=True' <<<"$($KC logs "$pre" 2>/dev/null || true)" \
    || die "prefill did not expose DSACacheManager transfer_attr=True global metadata"
}

require_config_gate() {
  local cfg
  cfg="$($KC get cm "${DGD}-config" -o yaml)"
  grep -q 'disable_overlap_scheduler: false' <<<"$cfg" || die "Moondream-style overlap scheduler is not enabled"
  grep -q 'cp_type: LAYERSPLIT' <<<"$cfg" || die "prefill LayerSplit is not configured"
  grep -q 'layersplit_enabled: true' <<<"$cfg" || die "prefill LayerSplit sparse config is not enabled"
  grep -q 'layersplit_all_cp_ranks_transfer: true' <<<"$cfg" || die "LayerSplit all-rank transfer is not configured"
  grep -q 'layersplit_transfer_backend: nixl' <<<"$cfg" || die "LayerSplit transfer backend is not NIXL"
  grep -q 'layersplit_owner_local_alloc: true' <<<"$cfg" || die "LayerSplit owner-local allocation is not enabled"
  [[ "$(grep -c 'backend: NIXL' <<<"$cfg")" -ge 2 ]] || die "prefill/decode NIXL cache transceivers are not both configured"
  ! grep -q 'backend: UCX' <<<"$cfg" || die "UCX cache transceiver backend is present; NIXL is the pre-A/B baseline"
  grep -q 'mla_latent_kv_dtype: kvarn_k2v2' <<<"$cfg" || die "dense MLA KVarN kvarn_k2v2 is not configured"
  ! grep -q 'mla_latent_kv_dtype: auto' <<<"$cfg" || die "dense MLA KVarN fell back to auto dtype"
  grep -q 'mla_latent_kv_amortize: true' <<<"$cfg" || die "dense MLA KVarN amortization is not configured"
  grep -q 'indexer_k_dtype: fp4' <<<"$cfg" || die "Indexer K is not FP4/HISA"
  ! grep -q 'indexer_k_dtype: kvarn' <<<"$cfg" || die "Indexer K was incorrectly routed to KVarN"
  ! grep -q 'layersplit_transfer_backend: ucx' <<<"$cfg" || die "LayerSplit transfer backend fell back to UCX"
  grep -q 'backend: WARPDECODE' <<<"$cfg" || die "WarpDecode is not configured"
  grep -q 'allow_parallelism_fallback: false' <<<"$cfg" || die "WarpDecode kernel backend fallback is not fail-closed"
  ! grep -q 'cp_type: HELIX' <<<"$cfg" || die "HELIX is present in production config"

  case "$SMC_GATE_MODE" in
    deferred)
      ! grep -q 'decoding_type: SMC' <<<"$cfg" || die "SMC-SD must remain deferred for the current NIXL/LayerSplit gate"
      ! grep -q 'speculative_model:' <<<"$cfg" || die "speculative draft model must remain absent while SMC is deferred"
      ! grep -q 'draft_attention_backend:' <<<"$cfg" || die "draft attention backend must remain absent while SMC is deferred"
      ;;
    required)
      grep -q 'decoding_type: SMC' <<<"$cfg" || die "SMC-SD is required for this smoke but not configured"
      ;;
    *) die "SMC_GATE_MODE must be deferred or required, got $SMC_GATE_MODE" ;;
  esac
}

run_non_streaming_smoke() {
  local fe="$1" response_file="$2"
  $KC exec -i "$fe" -- python3 - "$MODEL" <<'PY_REQ' >"$response_file"
import json
import sys
import urllib.request

model = sys.argv[1]
payload = {
    "model": model,
    "prompt": "NIXL request pinning smoke. Count to five.",
    "max_tokens": 32,
    "temperature": 0,
    "stream": False,
    "nvext": {"extra_fields": ["worker_id", "timing"]},
}
req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
body = urllib.request.urlopen(req, timeout=300).read().decode()
print(body)
PY_REQ
}

run_stream_abort_smoke() {
  local fe="$1"
  $KC exec -i "$fe" -- python3 - "$MODEL" <<'PY_REQ'
import json
import sys
import urllib.request

model = sys.argv[1]
payload = {
    "model": model,
    "prompt": "NIXL request pinning early-close smoke. Continue briefly.",
    "max_tokens": 128,
    "temperature": 0,
    "stream": True,
    "nvext": {"extra_fields": ["worker_id", "timing"]},
}
req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
response = urllib.request.urlopen(req, timeout=300)
print(response.readline().decode(errors="ignore")[:500])
response.close()
PY_REQ
}

fetch_perf_metrics() {
  local fe="$1" metrics_file="$2"
  $KC exec -i "$fe" -- python3 - "$DGD" <<'PY_METRICS' >"$metrics_file"
import json
import sys
import urllib.error
import urllib.request

# The Dynamo frontend does not serve TRT-LLM /perf_metrics.  Query all likely
# in-cluster surfaces and preserve failures as data so the parser can still use
# response-carried nvext timing as the primary proof source.
dgd = sys.argv[1]
urls = [
    ("frontend", "http://127.0.0.1:8000/perf_metrics"),
    ("prefill-service", f"http://{dgd}-prefill:9090/perf_metrics"),
    ("decode-service", f"http://{dgd}-decode:9090/perf_metrics"),
]
out = []
for source, url in urls:
    try:
        body = urllib.request.urlopen(url, timeout=30).read().decode()
        try:
            payload = json.loads(body)
        except Exception:
            payload = body
        out.append({"source": source, "url": url, "payload": payload})
    except urllib.error.HTTPError as exc:
        out.append({"source": source, "url": url, "error": f"HTTP {exc.code}: {exc.reason}"})
    except Exception as exc:
        out.append({"source": source, "url": url, "error": repr(exc)})
print(json.dumps(out))
PY_METRICS
}

parse_logs() {
  local frontend_log="$1"
  local prefill_log="$2"
  local decode_log="$3"
  local response_json="$4"
  local metrics_json="$5"
  python3 - "$frontend_log" "$prefill_log" "$decode_log" "$response_json" "$metrics_json" "$REQUIRE_DYNAMO_PIN_MARKERS" "$REQUIRE_POSITIVE_TRANSFER_METRICS" "$REQUIRE_ABORT_CLEANUP_MARKER" "$SMC_GATE_MODE" <<'PY_PARSE'
import json
import re
import sys
from pathlib import Path

ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
frontend = ansi_re.sub("", Path(sys.argv[1]).read_text(errors="ignore"))
prefill = ansi_re.sub("", Path(sys.argv[2]).read_text(errors="ignore"))
decode = ansi_re.sub("", Path(sys.argv[3]).read_text(errors="ignore"))
response_text = Path(sys.argv[4]).read_text(errors="ignore")
metrics_text = Path(sys.argv[5]).read_text(errors="ignore")
require_dynamo = sys.argv[6] == "1"
require_positive_transfer_metrics = sys.argv[7] == "1"
require_abort_cleanup_marker = sys.argv[8] == "1"
smc_gate_mode = sys.argv[9]
all_logs = "\n".join([frontend, prefill, decode])

bad = [
    r"Request pinning requires",
    r"ctx_dp_rank is None",
    r"Pinned worker .* could not resolve dp_rank",
    r"Routing to specified worker without resolved dp_rank",
    r"NoBootstrapEndpoint",
    r"Prefill router not activated",
    r"Disable overlap scheduler.*SMC",
    r"Disable overlap scheduler for speculation mode SMC",
    r"cp_type:\s*HELIX",
    r"\bHELIX\b.*fallback",
    r"WarpDecode.*fallback",
    r"LayerSplit: layer .* no local KV pool slot",
    r"no scratch routing",
    r"SMC-SD requires target token probabilities",
    r"SMC-SD requires selected draft token log probabilities",
    r"host_pinned_blocks[=: ]+0\b",
    r"cache_state_layers[=: ]+0\b",
    r"pinned KV handoff.*0 blocks",
    r"KV cache transfer timeout",
    r"Terminating .* due to KV cache transfer timeout",
    r"illegal memory access",
    r"MLACacheFormatter::inquireSupport",
    r"only support same number of layers",
    r"CacheTransferLayer::validateSupport",
    r"NIXL.*(?:failed|failure|error)",
    r"(?:failed|failure|error).*NIXL",
    r"Traceback",
]
for pattern in bad:
    if re.search(pattern, all_logs, re.IGNORECASE):
        raise SystemExit(f"bad log pattern present: {pattern}")

try:
    response = json.loads(response_text)
except Exception as exc:
    raise SystemExit(f"completion response was not valid JSON: {exc}: {response_text[:400]}")
worker = response.get("nvext", {}).get("worker_id")
if not isinstance(worker, dict):
    raise SystemExit("completion response missing nvext.worker_id; request did not expose router metadata")
required = ["prefill_worker_id", "prefill_dp_rank", "decode_worker_id", "decode_dp_rank"]
missing = [key for key in required if worker.get(key) is None]
if missing:
    raise SystemExit(f"completion nvext.worker_id missing {missing}: {worker}")

selected_prefill = re.findall(
    r"Selected worker: worker_type=prefill, worker_id=(\d+) dp_rank=(\d+)",
    frontend,
)
selected_decode = re.findall(
    r"Selected worker: worker_type=decode, worker_id=(\d+) dp_rank=(\d+)",
    frontend,
)
route_selected = re.findall(
    r"dynamo request pin route selected.*request_id[= ]([^, ]+).*worker_id[= ](\d+).*dp_rank[= ](\d+).*phase[= ](Prefill|Decode|Aggregated)",
    frontend,
)
established = [
    (rid, worker_id, dp_rank, "bootstrap", f"{host}:{port}")
    for rid, worker_id, dp_rank, host, port in re.findall(
        r"dynamo disagg request pin established.*request_id[= ]([^, ]+).*prefill_worker_id[= ](\d+).*prefill_dp_rank[= ](?:Some\()?([0-9]+).*bootstrap_host[= ]([^, ]+).*bootstrap_port[= ](\d+)",
        frontend,
    )
]
established.extend(
    (rid, worker_id, dp_rank, "completed_prefill", ctx_info_endpoint)
    for rid, worker_id, dp_rank, ctx_info_endpoint in re.findall(
        r"dynamo disagg request pin established.*request_id[= ]([^, ]+).*prefill_worker_id[= ](\d+).*prefill_dp_rank[= ](?:Some\()?([0-9]+).*ctx_info_endpoint[= ]([^, ]+).*handoff_mode[= ]\"?completed_prefill\"?",
        frontend,
    )
)
outbound = [
    (rid, "bootstrap", f"{host}:{port}")
    for rid, host, port in re.findall(
        r"dynamo disagg request pin outbound to decode.*request_id[= ]([^, ]+).*bootstrap_host[= ]([^, ]+).*bootstrap_port[= ](\d+)",
        frontend,
    )
]
outbound.extend(
    (rid, "completed_prefill", ctx_info_endpoint)
    for rid, ctx_info_endpoint in re.findall(
        r"dynamo disagg request pin outbound to decode.*request_id[= ]([^, ]+).*ctx_info_endpoint[= ]([^, ]+).*handoff_mode[= ]\"?completed_prefill\"?",
        frontend,
    )
)
cleared = re.findall(r"dynamo request pin cleared|disagg request pin cleared", all_logs)
cleared_rids = set(re.findall(
    r"dynamo request pin cleared.*request_id[= ]([^, ]+)|disagg request pin cleared.*request_id[= ]([^, ]+)",
    all_logs,
))
cleared_rids = {rid for pair in cleared_rids for rid in pair if rid}
cleanup_scheduled = re.findall(r"dynamo request pin cleanup scheduled", all_logs)
cleanup_scheduled_rids = set(re.findall(
    r"dynamo request pin cleanup scheduled.*request_id[= ]([^, ]+)",
    all_logs,
))

prefill_pair = (str(worker["prefill_worker_id"]), str(worker["prefill_dp_rank"]))
decode_pair = (str(worker["decode_worker_id"]), str(worker["decode_dp_rank"]))
route_prefill = [(rid, w, r) for rid, w, r, phase in route_selected if phase == "Prefill"]
route_decode = [(rid, w, r) for rid, w, r, phase in route_selected if phase == "Decode"]
if prefill_pair not in selected_prefill and prefill_pair not in [(w, r) for _rid, w, r in route_prefill]:
    raise SystemExit(f"response prefill worker/rank {prefill_pair} not found in frontend Rust route logs")
if selected_decode and decode_pair not in selected_decode:
    raise SystemExit(f"response decode worker/rank {decode_pair} not found in frontend decode selection logs")
if route_decode and decode_pair not in [(w, r) for _rid, w, r in route_decode]:
    raise SystemExit(f"response decode worker/rank {decode_pair} not found in Dynamo decode route-selected logs")

if require_dynamo:
    route_prefill_rids = {rid for rid, _w, _r in route_prefill}
    route_decode_rids = {rid for rid, _w, _r in route_decode}
    route_prefill_matching_response = {
        rid for rid, w, r in route_prefill if (w, r) == prefill_pair
    }
    route_decode_matching_response = {
        rid for rid, w, r in route_decode if (w, r) == decode_pair
    }

    if not established:
        raise SystemExit("missing Dynamo pin-established marker; route-selected logs alone are not a pre-A/B pinning proof")
    if not outbound:
        raise SystemExit("missing Dynamo outbound-to-decode marker; route-selected logs alone are not a pre-A/B pinning proof")
    placeholder_completed = [
        item for item in [*established, *outbound]
        if item[-2] == "completed_prefill" and item[-1] in {"", "completed_prefill", "None", "null"}
    ]
    if placeholder_completed:
        raise SystemExit(f"completed-prefill pin marker missing real ctx_info_endpoint: {placeholder_completed}")
    established_pairs = [(w, r) for _rid, w, r, _mode, _anchor in established]
    if prefill_pair not in established_pairs:
        raise SystemExit(f"response prefill worker/rank {prefill_pair} not present in Dynamo pin-established markers: {established}")
    established_rids = {rid for rid, _w, _r, _mode, _anchor in established}
    outbound_rids = {rid for rid, _mode, _anchor in outbound}
    shared_pin_rids = established_rids & outbound_rids
    if not shared_pin_rids:
        raise SystemExit(f"no request id appears in both pin-established and outbound-to-decode markers: established={established_rids} outbound={outbound_rids}")
    if not (shared_pin_rids & route_prefill_rids):
        raise SystemExit(f"no request id appears in both route-selected prefill and pin lifecycle markers: route_prefill={route_prefill_rids} pin={shared_pin_rids}")
    if not (shared_pin_rids & route_decode_rids):
        raise SystemExit(f"no request id appears in both route-selected decode and pin lifecycle markers: route_decode={route_decode_rids} pin={shared_pin_rids}")
    lifecycle_rids = shared_pin_rids

    if not (lifecycle_rids & cleared_rids):
        raise SystemExit(
            f"request pin cleanup proof incomplete: lifecycle_rids={lifecycle_rids} "
            f"cleared_rids={cleared_rids} cleanup_scheduled={cleanup_scheduled_rids}"
        )
    if require_abort_cleanup_marker and not cleanup_scheduled_rids:
        raise SystemExit("early-close abort cleanup proof missing: no dynamo request pin cleanup scheduled marker")
else:
    print("WARNING: REQUIRE_DYNAMO_PIN_MARKERS=0; selector-only dry run is not a pre-A/B proof")




def _walk(obj):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk(value)

positive_transfer_metrics = []
try:
    perf_metrics = json.loads(metrics_text) if metrics_text.strip() else []
except Exception as exc:
    if require_positive_transfer_metrics:
        raise SystemExit(f"perf_metrics response was not valid JSON: {exc}: {metrics_text[:400]}")
    perf_metrics = []

def _maybe_record_transfer_metric(item, source):
    if not isinstance(item, dict):
        return
    candidates = []
    timing = item.get("timing_metrics")
    if isinstance(timing, dict):
        candidates.append(timing)
    nvext = item.get("nvext")
    if isinstance(nvext, dict):
        nvext_timing = nvext.get("timing") or nvext.get("timing_metrics")
        if isinstance(nvext_timing, dict):
            candidates.append(nvext_timing)
    candidates.append(item)
    for timing in candidates:
        size = timing.get("kv_cache_size", 0) or 0
        start = timing.get("kv_cache_transfer_start", 0) or 0
        end = timing.get("kv_cache_transfer_end", 0) or 0
        try:
            size = float(size)
            start = float(start)
            end = float(end)
        except (TypeError, ValueError):
            continue
        if size > 0 and start > 0 and end >= start:
            positive_transfer_metrics.append((source, size, start, end))

for item in _walk(response):
    _maybe_record_transfer_metric(item, "response")
for item in _walk(perf_metrics):
    _maybe_record_transfer_metric(item, "perf_metrics")

proof_starts = {
    rid: int(blocks)
    for rid, blocks in re.findall(
        r"OPTRT_NIXL_TRANSFER_PROOF.*phase=context_send_start.*request_id=(\S+).*cache_blocks=([0-9]+)",
        all_logs,
    )
}
proof_ctx_complete = set(re.findall(
    r"OPTRT_NIXL_TRANSFER_PROOF.*phase=context_send_complete.*request_id=(\S+)",
    all_logs,
))
proof_gen_complete = set(re.findall(
    r"OPTRT_NIXL_TRANSFER_PROOF.*phase=gen_recv_complete.*request_id=(\S+)",
    all_logs,
))
incomplete_transfer_proof_ids = {
    rid: blocks for rid, blocks in proof_starts.items()
    if blocks > 0 and rid not in proof_ctx_complete and rid not in proof_gen_complete
}
if require_positive_transfer_metrics and incomplete_transfer_proof_ids:
    raise SystemExit(
        "incomplete KV transfer proof: context_send_start without matching "
        "context_send_complete/gen_recv_complete in smoke window; "
        f"incomplete={incomplete_transfer_proof_ids} "
        f"ctx_complete={proof_ctx_complete} gen_complete={proof_gen_complete}"
    )
positive_transfer_proof_ids = {
    rid for rid, blocks in proof_starts.items()
    if blocks > 0 and (rid in proof_ctx_complete or rid in proof_gen_complete)
}
for rid in sorted(positive_transfer_proof_ids):
    positive_transfer_metrics.append(("log_proof", float(proof_starts[rid]), 1.0, 1.0))

if require_positive_transfer_metrics and not positive_transfer_metrics:
    raise SystemExit(
        "positive KV transfer proof missing: positive KV transfer metrics missing; response nvext timing, worker /perf_metrics, "
        "and OPTRT_NIXL_TRANSFER_PROOF logs did not expose a nonzero completed transfer; "
        f"starts={proof_starts} ctx_complete={proof_ctx_complete} gen_complete={proof_gen_complete} "
        f"perf_metrics_probe={metrics_text[:600]}"
    )

if smc_gate_mode == "required":
    if re.search(r"Disable overlap scheduler.*SMC", all_logs):
        raise SystemExit("SMC overlap scheduler was disabled")
    required_smc_markers = [
        "SMC Moondream decode handoff preserved",
        "draft_token_log_probs",
        "sample_state.sampler_event",
        "pinned_host_tokens=True",
        "ctx_dp_rank=",
        "ctx_info_endpoint=",
    ]
    missing = [marker for marker in required_smc_markers if marker not in all_logs]
    if missing:
        raise SystemExit(
            f"SMC-SD is required but decode handoff markers are missing: {missing}"
        )
    if re.search(r"SMC Moondream decode handoff preserved.*ctx_dp_rank=None", all_logs):
        raise SystemExit("SMC-SD decode handoff marker has ctx_dp_rank=None")
    if re.search(r"SMC Moondream decode handoff preserved.*ctx_info_endpoint=(?:None|null|$)", all_logs):
        raise SystemExit("SMC-SD decode handoff marker has empty ctx_info_endpoint")

print(
    "request pinning live proof ok: "
    f"prefill={prefill_pair} decode={decode_pair} "
    f"dynamo_required={require_dynamo} established={len(established)} outbound={len(outbound)} "
    f"cleared={len(cleared)} cleanup_scheduled={len(cleanup_scheduled)} "
    f"positive_transfer_metrics={len(positive_transfer_metrics)}"
)
PY_PARSE
}

main() {
  need_ready
  require_config_gate

  local fe pre dec
  fe="$(pod_for frontend)"
  pre="$(pod_for prefill)"
  dec="$(pod_for decode)"
  [[ -n "$fe" && -n "$pre" && -n "$dec" ]] || die "could not resolve frontend/prefill/decode pods"
  require_runtime_nixl_gate "$pre" "$dec"

  local pre_restart dec_restart
  pre_restart="$(restart_count "$pre")"
  dec_restart="$(restart_count "$dec")"

  local start
  start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  SMOKE_TMPDIR="$(mktemp -d)"
  trap cleanup EXIT

  run_non_streaming_smoke "$fe" "$SMOKE_TMPDIR/response.json"
  run_stream_abort_smoke "$fe"
  fetch_perf_metrics "$fe" "$SMOKE_TMPDIR/perf_metrics.json"

  $KC logs "$fe" --since-time="$start" >"$SMOKE_TMPDIR/frontend.log"
  $KC logs "$pre" --since-time="$start" >"$SMOKE_TMPDIR/prefill.log"
  $KC logs "$dec" --since-time="$start" >"$SMOKE_TMPDIR/decode.log"
  parse_logs "$SMOKE_TMPDIR/frontend.log" "$SMOKE_TMPDIR/prefill.log" "$SMOKE_TMPDIR/decode.log" "$SMOKE_TMPDIR/response.json" "$SMOKE_TMPDIR/perf_metrics.json"

  [[ "$(restart_count "$pre")" == "$pre_restart" ]] || die "prefill restarted during smoke"
  [[ "$(restart_count "$dec")" == "$dec_restart" ]] || die "decode restarted during smoke"
  echo "NIXL request pinning smoke passed for $DGD"
}

main "$@"

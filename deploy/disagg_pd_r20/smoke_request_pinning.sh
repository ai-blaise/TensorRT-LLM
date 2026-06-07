#!/usr/bin/env bash
set -euo pipefail

DGD="${DGD:-topo-c1-dp2tp4-disagg-r20}"
MODEL="${MODEL:-BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft}"
KC="${KC:-sudo -E /usr/local/bin/k3s kubectl -n dynamo-system}"

# Set REQUIRE_DYNAMO_PIN_MARKERS=0 only for a router-selector dry run against an
# image that has not yet rebuilt Dynamo with the fail-closed marker patch. The
# pre-A/B gate must leave this at 1.
REQUIRE_DYNAMO_PIN_MARKERS="${REQUIRE_DYNAMO_PIN_MARKERS:-1}"
SMC_GATE_MODE="${SMC_GATE_MODE:-deferred}"

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
  grep -q 'mla_latent_kv_amortize: true' <<<"$cfg" || die "dense MLA KVarN amortization is not configured"
  grep -q 'backend: WARPDECODE' <<<"$cfg" || die "WarpDecode is not configured"
  grep -q 'allow_parallelism_fallback: false' <<<"$cfg" || die "WarpDecode kernel backend fallback is not fail-closed"
  ! grep -q 'cp_type: HELIX' <<<"$cfg" || die "HELIX is present in production config"

  case "$SMC_GATE_MODE" in
    deferred)
      ! grep -q 'decoding_type: SMC' <<<"$cfg" || die "SMC-SD must remain deferred for the current NIXL/LayerSplit gate"
      ! grep -q 'speculative_model:' <<<"$cfg" || die "speculative draft model must remain absent while SMC is deferred"
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
    "prompt": "Non-MORI request pinning smoke. Count to five.",
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
    "prompt": "Non-MORI request pinning early-close smoke. Continue briefly.",
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

parse_logs() {
  local frontend_log="$1"
  local prefill_log="$2"
  local decode_log="$3"
  local response_json="$4"
  python3 - "$frontend_log" "$prefill_log" "$decode_log" "$response_json" "$REQUIRE_DYNAMO_PIN_MARKERS" "$SMC_GATE_MODE" <<'PY_PARSE'
import json
import re
import sys
from pathlib import Path

frontend = Path(sys.argv[1]).read_text(errors="ignore")
prefill = Path(sys.argv[2]).read_text(errors="ignore")
decode = Path(sys.argv[3]).read_text(errors="ignore")
response_text = Path(sys.argv[4]).read_text(errors="ignore")
require_dynamo = sys.argv[5] == "1"
smc_gate_mode = sys.argv[6]
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
    r"illegal memory access",
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
    r"dynamo request pin route selected.*worker_id[= ](\d+).*dp_rank[= ](\d+).*phase[= ](Prefill|Decode|Aggregated)",
    frontend,
)
established = re.findall(
    r"dynamo disagg request pin established.*prefill_worker_id[= ](\d+).*prefill_dp_rank[= ](?:Some\()?([0-9]+)",
    frontend,
)
outbound = re.findall(r"dynamo disagg request pin outbound to decode", frontend)

prefill_pair = (str(worker["prefill_worker_id"]), str(worker["prefill_dp_rank"]))
decode_pair = (str(worker["decode_worker_id"]), str(worker["decode_dp_rank"]))
route_prefill = [(w, r) for w, r, phase in route_selected if phase == "Prefill"]
route_decode = [(w, r) for w, r, phase in route_selected if phase == "Decode"]
if prefill_pair not in selected_prefill and prefill_pair not in route_prefill:
    raise SystemExit(f"response prefill worker/rank {prefill_pair} not found in frontend Rust route logs")
if selected_decode and decode_pair not in selected_decode:
    raise SystemExit(f"response decode worker/rank {decode_pair} not found in frontend decode selection logs")
if route_decode and decode_pair not in route_decode:
    raise SystemExit(f"response decode worker/rank {decode_pair} not found in Dynamo decode route-selected logs")

if require_dynamo:
    if not established:
        raise SystemExit("no Dynamo pin-established marker found; rebuild Dynamo router image/base with the request-pinning patch")
    if not outbound:
        raise SystemExit("no Dynamo pin outbound-to-decode marker found; request-level KV handoff was not proven")
    if prefill_pair not in [(w, r) for w, r in established]:
        raise SystemExit(f"response prefill worker/rank {prefill_pair} not present in Dynamo pin-established markers: {established}")
else:
    print("WARNING: REQUIRE_DYNAMO_PIN_MARKERS=0; selector-only dry run is not a pre-A/B proof")

if smc_gate_mode == "required":
    if re.search(r"Disable overlap scheduler.*SMC", all_logs):
        raise SystemExit("SMC overlap scheduler was disabled")
    if "draft_token_log_probs" not in all_logs and "SMC" in all_logs:
        print("WARNING: SMC logprob payload marker not found in logs; rely on focused tests plus deeper E2E parser")

print(
    "request pinning live proof ok: "
    f"prefill={prefill_pair} decode={decode_pair} "
    f"dynamo_required={require_dynamo} established={len(established)} outbound={len(outbound)}"
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

  local pre_restart dec_restart
  pre_restart="$(restart_count "$pre")"
  dec_restart="$(restart_count "$dec")"

  local start tmpdir
  start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "$tmpdir"' EXIT

  run_non_streaming_smoke "$fe" "$tmpdir/response.json"
  run_stream_abort_smoke "$fe"

  $KC logs "$fe" --since-time="$start" >"$tmpdir/frontend.log"
  $KC logs "$pre" --since-time="$start" >"$tmpdir/prefill.log"
  $KC logs "$dec" --since-time="$start" >"$tmpdir/decode.log"
  parse_logs "$tmpdir/frontend.log" "$tmpdir/prefill.log" "$tmpdir/decode.log" "$tmpdir/response.json"

  [[ "$(restart_count "$pre")" == "$pre_restart" ]] || die "prefill restarted during smoke"
  [[ "$(restart_count "$dec")" == "$dec_restart" ]] || die "decode restarted during smoke"
  echo "non-MORI NIXL request pinning smoke passed for $DGD"
}

main "$@"

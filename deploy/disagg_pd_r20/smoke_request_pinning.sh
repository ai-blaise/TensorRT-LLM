#!/usr/bin/env bash
set -euo pipefail

DGD="${DGD:-topo-c1-dp2tp4-disagg-r20}"
MODEL="${MODEL:-BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft}"
KC="${KC:-sudo -E /usr/local/bin/k3s kubectl -n dynamo-system}"

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
  grep -q 'disable_overlap_scheduler: false' <<<"$cfg" || die "overlap scheduler is not enabled in config"
  grep -q 'cp_type: LAYERSPLIT' <<<"$cfg" || die "prefill LayerSplit is not configured"
  grep -q 'layersplit_enabled: true' <<<"$cfg" || die "prefill LayerSplit sparse config is not enabled"
  grep -q 'mla_latent_kv_dtype: kvarn_k2v2' <<<"$cfg" || die "dense MLA KVarN kvarn_k2v2 is not configured"
  grep -q 'decoding_type: SMC' <<<"$cfg" || die "SMC-SD is not configured"
  grep -q 'backend: WARPDECODE' <<<"$cfg" || die "WarpDecode is not configured"
  grep -q 'allow_parallelism_fallback: false' <<<"$cfg" || die "WarpDecode kernel fallback is not fail-closed"
  ! grep -q 'cp_type: HELIX' <<<"$cfg" || die "HELIX is present in production config"
}

run_non_streaming_smoke() {
  local fe="$1"
  $KC exec "$fe" -- python3 - "$MODEL" <<'PY'
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
}
req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
body = urllib.request.urlopen(req, timeout=300).read().decode()
print(body[:1000])
PY
}

run_stream_abort_smoke() {
  local fe="$1"
  $KC exec "$fe" -- python3 - "$MODEL" <<'PY'
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
}

parse_logs() {
  local frontend_log="$1"
  local prefill_log="$2"
  local decode_log="$3"
  python3 - "$frontend_log" "$prefill_log" "$decode_log" <<'PY'
import re
import sys
from pathlib import Path

frontend = Path(sys.argv[1]).read_text(errors="ignore")
prefill = Path(sys.argv[2]).read_text(errors="ignore")
decode = Path(sys.argv[3]).read_text(errors="ignore")
all_logs = "\n".join([frontend, prefill, decode])

bad = [
    r"Request pinning requires",
    r"ctx_dp_rank is None",
    r"Disable overlap scheduler.*SMC",
    r"cp_type:\s*HELIX",
    r"\bHELIX\b.*fallback",
    r"WarpDecode.*fallback",
    r"SMC-SD requires target token probabilities",
    r"SMC-SD requires selected draft token log probabilities",
    r"illegal memory access",
    r"Traceback",
]
for pattern in bad:
    if re.search(pattern, all_logs, re.IGNORECASE):
        raise SystemExit(f"bad log pattern present: {pattern}")

established = re.findall(
    r"disagg request pin established: rid=(\d+) pin=(\{[^\n]*\})",
    frontend,
)
cleared = set(re.findall(r"disagg request pin cleared: rid=(\d+)", frontend))
if not established:
    raise SystemExit("no request pin established logs found")

ranked_pins = []
for rid, pin in established:
    rank = re.search(r"ctx_dp_rank['\"]?:\s*(\d+)", pin)
    if rank and "gen_server" in pin and "ctx_server" in pin:
        ranked_pins.append((rid, rank.group(1), pin))

if not ranked_pins:
    raise SystemExit("no complete pin with ctx_server, gen_server, and ctx_dp_rank")

for rid, _rank, pin in ranked_pins:
    if rid not in cleared:
        raise SystemExit(f"pin was not cleared for rid={rid}: {pin}")

outbound_gen = {
    (rid, rank)
    for _ctx, rid, rank in re.findall(
        r"disagg request pin outbound: .*request_type=generation_only "
        r"ctx_request_id=(\d+) disagg_request_id=(\d+) ctx_dp_rank=(\d+)",
        frontend,
    )
}
if not outbound_gen:
    raise SystemExit("no outbound generation request with ctx_dp_rank in frontend logs")

prefill_context = re.findall(
    r"disagg request pin received: .*request_type=context_only .*disagg_request_id=(\d+)",
    prefill,
)
if not prefill_context:
    raise SystemExit("prefill did not log context_only request receipt")

decode_gen = {
    (rid, rank)
    for _ctx, rid, rank in re.findall(
        r"disagg request pin received: .*request_type=generation_only "
        r"ctx_request_id=(\d+) disagg_request_id=(\d+) ctx_dp_rank=(\d+)",
        decode,
    )
}
if not decode_gen:
    raise SystemExit("decode did not log generation_only request receipt with ctx_dp_rank")

for rid, rank, pin in ranked_pins:
    if (rid, rank) not in outbound_gen:
        raise SystemExit(f"frontend did not log outbound generation for rid={rid} rank={rank}: {pin}")
    if (rid, rank) not in decode_gen:
        raise SystemExit(f"decode did not log matching generation receipt for rid={rid} rank={rank}: {pin}")

print(f"request pinning live proof ok: {len(ranked_pins)} complete pinned request(s)")
PY
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

  local start
  start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  run_non_streaming_smoke "$fe"
  run_stream_abort_smoke "$fe"

  local tmpdir
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "$tmpdir"' EXIT
  $KC logs "$fe" --since-time="$start" >"$tmpdir/frontend.log"
  $KC logs "$pre" --since-time="$start" >"$tmpdir/prefill.log"
  $KC logs "$dec" --since-time="$start" >"$tmpdir/decode.log"
  parse_logs "$tmpdir/frontend.log" "$tmpdir/prefill.log" "$tmpdir/decode.log"

  [[ "$(restart_count "$pre")" == "$pre_restart" ]] || die "prefill restarted during smoke"
  [[ "$(restart_count "$dec")" == "$dec_restart" ]] || die "decode restarted during smoke"
  echo "non-MORI request pinning smoke passed for $DGD"
}

main "$@"

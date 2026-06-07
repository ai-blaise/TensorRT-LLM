#!/usr/bin/env bash
set -euo pipefail

DGD="${DGD:-topo-c1-dp2tp4-disagg-r20}"
MODEL="${MODEL:-BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft}"
KC="${KC:-sudo -E /usr/local/bin/k3s kubectl -n dynamo-system}"
BACKEND="nixl"
LENGTHS="1024,4096,8192,16384,32768,65536,131072"
CONCURRENCY=16
MAX_TOKENS=128
MIN_TOK_PER_USER="${MIN_TOK_PER_USER:-150}"
OUTPUT_DIR="${BENCH_OUT:-}"

usage() {
  cat <<USAGE
Usage: $0 [--backend nixl|ucx|mooncake|mori] [--lengths csv] [--concurrency n] [--max-tokens n] [--min-tok-per-user n] [--output-dir dir]

Runs a request-pinned disaggregated transport profile through the live frontend.
NIXL is the pre-A/B gate. Non-NIXL backends require ALLOW_TRANSPORT_AB=1 and
must already be explicitly deployed in the DGD config; this script never toggles
backend env vars or applies manifests.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2 ;;
    --lengths) LENGTHS="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --min-tok-per-user) MIN_TOK_PER_USER="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="/tmp/${BACKEND}_c${CONCURRENCY}_transport_bench_$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "$OUTPUT_DIR"

if [[ "$BACKEND" != "nixl" && "${ALLOW_TRANSPORT_AB:-0}" != "1" ]]; then
  echo "Refusing $BACKEND benchmark without ALLOW_TRANSPORT_AB=1; NIXL is the pre-A/B gate" >&2
  exit 2
fi

pod_for() {
  local component="$1"
  $KC get pods -o name | grep "${DGD}-0-${component}" | tail -1
}

restart_count() {
  local pod="$1"
  $KC get "$pod" -o jsonpath='{range .status.containerStatuses[*]}{.restartCount}{"\n"}{end}' \
    | awk '{sum += $1} END {print sum + 0}'
}

require_ready_and_backend() {
  local ready cfg lower_backend
  lower_backend="$(tr '[:upper:]' '[:lower:]' <<<"$BACKEND")"
  ready="$($KC get dgd "$DGD" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || true)"
  [[ "$ready" == "True" ]] || { echo "$DGD is not Ready (Ready=${ready:-missing})" >&2; exit 1; }
  cfg="$($KC get cm "${DGD}-config" -o yaml)"
  case "$lower_backend" in
    nixl)
      grep -q 'backend: NIXL' <<<"$cfg" || { echo "config does not contain backend: NIXL" >&2; exit 1; }
      [[ "$(grep -c 'backend: NIXL' <<<"$cfg")" -ge 2 ]] || { echo "prefill/decode are not both NIXL" >&2; exit 1; }
      grep -q 'layersplit_transfer_backend: nixl' <<<"$cfg" || { echo "LayerSplit transfer backend is not nixl" >&2; exit 1; }
      ! grep -q 'backend: UCX' <<<"$cfg" || { echo "UCX backend present in NIXL gate config" >&2; exit 1; }
      ;;
    ucx)
      grep -q 'backend: UCX' <<<"$cfg" || { echo "UCX A/B requested but config is not explicitly UCX" >&2; exit 1; }
      grep -q 'layersplit_transfer_backend: ucx' <<<"$cfg" || { echo "UCX A/B requested but LayerSplit backend is not ucx" >&2; exit 1; }
      ;;
    mooncake|mori)
      echo "$BACKEND is A/B-only and requires a runnable wrapper/API; no production gate is defined here" >&2
      exit 2
      ;;
    *) echo "unknown backend: $BACKEND" >&2; exit 2 ;;
  esac
  grep -q 'cp_type: LAYERSPLIT' <<<"$cfg" || { echo "LayerSplit missing" >&2; exit 1; }
  grep -q 'layersplit_owner_local_alloc: true' <<<"$cfg" || { echo "owner-local LayerSplit missing" >&2; exit 1; }
  grep -q 'mla_latent_kv_dtype: kvarn_k2v2' <<<"$cfg" || { echo "dense MLA KVarN k2v2 missing" >&2; exit 1; }
  grep -q 'indexer_k_dtype: fp4' <<<"$cfg" || { echo "Indexer fp4 missing" >&2; exit 1; }
  grep -q 'allow_parallelism_fallback: false' <<<"$cfg" || { echo "WarpDecode fail-closed setting missing" >&2; exit 1; }
  ! grep -q 'cp_type: HELIX' <<<"$cfg" || { echo "HELIX present" >&2; exit 1; }
}

require_ready_and_backend
FE="$(pod_for frontend)"
PRE="$(pod_for prefill)"
DEC="$(pod_for decode)"
[[ -n "$FE" && -n "$PRE" && -n "$DEC" ]] || { echo "could not resolve frontend/prefill/decode pods" >&2; exit 1; }

START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PRE_RESTART_BEFORE="$(restart_count "$PRE")"
DEC_RESTART_BEFORE="$(restart_count "$DEC")"

$KC get pods -o wide | grep "$DGD" >"$OUTPUT_DIR/pods_before.txt" || true
$KC get cm "${DGD}-config" -o yaml >"$OUTPUT_DIR/config.yaml" || true
ip -s link >"$OUTPUT_DIR/ip_link_before.txt" 2>&1 || true
nvidia-smi dmon -s pucm -d 1 -o TD >"$OUTPUT_DIR/nvidia_smi_dmon.txt" 2>&1 &
DMON_PID="$!"
cleanup_dmon() {
  if kill -0 "$DMON_PID" 2>/dev/null; then
    kill "$DMON_PID" 2>/dev/null || true
    wait "$DMON_PID" 2>/dev/null || true
  fi
}
trap cleanup_dmon EXIT

$KC exec -i "$FE" -- python3 - "$MODEL" "$LENGTHS" "$CONCURRENCY" "$MAX_TOKENS" >"$OUTPUT_DIR/results.jsonl" <<'PY_BENCH'
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.request

model = sys.argv[1]
lengths = [int(x) for x in sys.argv[2].split(',') if x]
concurrency = int(sys.argv[3])
max_tokens = int(sys.argv[4])


def make_prompt(length):
    seed = 'nixl layersplit pinned kv transfer profile '
    unit = ' '.join([seed, 'token']) + ' '
    return (unit * ((length // 6) + 2))[: max(1, length * 7)]


def stream_request(length, idx):
    prompt = make_prompt(length)
    payload = {
        'model': model,
        'prompt': prompt,
        'max_tokens': max_tokens,
        'temperature': 0,
        'stream': True,
        'nvext': {'extra_fields': ['worker_id', 'timing']},
    }
    req = urllib.request.Request(
        'http://127.0.0.1:8000/v1/completions',
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
    )
    start = time.perf_counter()
    first = None
    last = start
    itls = []
    chunks = 0
    text_chars = 0
    worker = None
    error = None
    status = 'ok'
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for raw in resp:
                line = raw.decode(errors='ignore').strip()
                if not line or not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    break
                now = time.perf_counter()
                if first is None:
                    first = now
                else:
                    itls.append(now - last)
                last = now
                try:
                    obj = json.loads(data)
                except Exception:
                    obj = None
                if isinstance(obj, dict):
                    worker = obj.get('nvext', {}).get('worker_id') or worker
                    choices = obj.get('choices') or []
                    if choices:
                        delta = choices[0].get('text') or choices[0].get('delta', {}).get('content') or ''
                        text_chars += len(delta)
                chunks += 1
    except Exception as exc:
        status = 'error'
        error = repr(exc)
    end = time.perf_counter()
    decode_time = max(0.0, end - (first or end))
    tok_per_user_after_first = (max(0, chunks - 1) / decode_time) if decode_time > 0 else 0.0
    return {
        'length': length,
        'idx': idx,
        'status': status,
        'error': error,
        'chunks': chunks,
        'text_chars': text_chars,
        'ttft_s': None if first is None else first - start,
        'total_s': end - start,
        'decode_s_after_first': decode_time,
        'tok_per_user_after_first': tok_per_user_after_first,
        'itl_p50_s': statistics.median(itls) if itls else None,
        'itl_p95_s': sorted(itls)[int(0.95 * (len(itls) - 1))] if itls else None,
        'itl_p99_s': sorted(itls)[int(0.99 * (len(itls) - 1))] if itls else None,
        'worker_id': worker,
    }

for length in lengths:
    wave_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(stream_request, length, i) for i in range(concurrency)]
        rows = [f.result() for f in concurrent.futures.as_completed(futures)]
    wave_end = time.perf_counter()
    ok = [r for r in rows if r['status'] == 'ok']
    summary = {
        'type': 'summary',
        'length': length,
        'concurrency': concurrency,
        'ok': len(ok),
        'failed': len(rows) - len(ok),
        'wall_s': wave_end - wave_start,
        'ttft_p50_s': statistics.median([r['ttft_s'] for r in ok if r['ttft_s'] is not None]) if ok else None,
        'ttft_p95_s': sorted([r['ttft_s'] for r in ok if r['ttft_s'] is not None])[int(0.95 * (len(ok) - 1))] if ok else None,
        'tok_per_user_after_first_p50': statistics.median([r['tok_per_user_after_first'] for r in ok]) if ok else 0.0,
        'tok_per_user_after_first_min': min([r['tok_per_user_after_first'] for r in ok], default=0.0),
        'itl_p50_s': statistics.median([r['itl_p50_s'] for r in ok if r['itl_p50_s'] is not None]) if ok else None,
        'itl_p95_s': statistics.median([r['itl_p95_s'] for r in ok if r['itl_p95_s'] is not None]) if ok else None,
        'itl_p99_s': statistics.median([r['itl_p99_s'] for r in ok if r['itl_p99_s'] is not None]) if ok else None,
    }
    print(json.dumps(summary), flush=True)
    for row in sorted(rows, key=lambda r: r['idx']):
        row['type'] = 'request'
        print(json.dumps(row), flush=True)
PY_BENCH

$KC exec -i "$FE" -- python3 - <<'PY_METRICS' >"$OUTPUT_DIR/perf_metrics.json" || true
import urllib.request
print(urllib.request.urlopen('http://127.0.0.1:8000/perf_metrics', timeout=120).read().decode())
PY_METRICS

cleanup_dmon
trap - EXIT
ip -s link >"$OUTPUT_DIR/ip_link_after.txt" 2>&1 || true
$KC logs "$FE" --since-time="$START_UTC" >"$OUTPUT_DIR/frontend.log" || true
$KC logs "$PRE" --since-time="$START_UTC" >"$OUTPUT_DIR/prefill.log" || true
$KC logs "$DEC" --since-time="$START_UTC" >"$OUTPUT_DIR/decode.log" || true

PRE_RESTART_AFTER="$(restart_count "$PRE")"
DEC_RESTART_AFTER="$(restart_count "$DEC")"
{
  echo "backend=$BACKEND"
  echo "dgd=$DGD"
  echo "frontend=$FE"
  echo "prefill=$PRE restart_before=$PRE_RESTART_BEFORE restart_after=$PRE_RESTART_AFTER"
  echo "decode=$DEC restart_before=$DEC_RESTART_BEFORE restart_after=$DEC_RESTART_AFTER"
  echo "lengths=$LENGTHS"
  echo "concurrency=$CONCURRENCY"
  echo "max_tokens=$MAX_TOKENS"
  echo "min_tok_per_user=$MIN_TOK_PER_USER"
  echo "output_dir=$OUTPUT_DIR"
} >"$OUTPUT_DIR/metadata.txt"

if [[ "$PRE_RESTART_BEFORE" != "$PRE_RESTART_AFTER" || "$DEC_RESTART_BEFORE" != "$DEC_RESTART_AFTER" ]]; then
  echo "worker restarted during benchmark; see $OUTPUT_DIR" >&2
  exit 1
fi

python3 - "$OUTPUT_DIR" "$BACKEND" "$CONCURRENCY" "$MIN_TOK_PER_USER" <<'PY_VERIFY'
import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
backend = sys.argv[2].lower()
concurrency = int(sys.argv[3])
min_tok = float(sys.argv[4])
logs = "\n".join((out / name).read_text(errors="ignore") for name in ("frontend.log", "prefill.log", "decode.log") if (out / name).exists())
ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
logs_clean = ansi.sub("", logs)

bad_patterns = [
    r"KV cache transfer timeout",
    r"Terminating .* due to KV cache transfer timeout",
    r"MLACacheFormatter::inquireSupport",
    r"CacheTransferLayer::validateSupport",
    r"only support same number of layers",
    r"illegal memory access",
    r"Traceback",
    r"NIXL.*(?:failed|failure|error)",
    r"(?:failed|failure|error).*NIXL",
]
if backend == "nixl":
    bad_patterns.extend([
        r"Using UCX kv-cache transceiver",
        r"cache_transceiver_config.*backend.*UCX",
        r"layersplit_transfer_backend: ucx",
    ])
for pattern in bad_patterns:
    if re.search(pattern, logs_clean, re.IGNORECASE):
        raise SystemExit(f"bad log pattern present during {backend} benchmark: {pattern}")

rows = []
for line in (out / "results.jsonl").read_text().splitlines():
    if line.strip():
        rows.append(json.loads(line))
summaries = [row for row in rows if row.get("type") == "summary"]
requests = [row for row in rows if row.get("type") == "request"]
if not summaries:
    raise SystemExit("benchmark emitted no summary rows")
failed = [row for row in summaries if row.get("failed") != 0 or row.get("ok") != concurrency]
if failed:
    raise SystemExit(f"benchmark request failures: {failed}")
slow = [row for row in summaries if float(row.get("tok_per_user_after_first_min") or 0.0) < min_tok]

proof_starts = {
    rid: int(blocks)
    for rid, blocks in re.findall(
        r"OPTRT_NIXL_TRANSFER_PROOF.*phase=context_send_start.*request_id=(\S+).*cache_blocks=([0-9]+)",
        logs_clean,
    )
}
proof_ctx_complete = set(re.findall(
    r"OPTRT_NIXL_TRANSFER_PROOF.*phase=context_send_complete.*request_id=(\S+)", logs_clean
))
proof_gen_complete = set(re.findall(
    r"OPTRT_NIXL_TRANSFER_PROOF.*phase=gen_recv_complete.*request_id=(\S+)", logs_clean
))
positive = sorted(rid for rid, blocks in proof_starts.items() if blocks > 0 and (rid in proof_ctx_complete or rid in proof_gen_complete))
if backend == "nixl" and not positive:
    raise SystemExit(
        "missing positive NIXL transfer proof: no nonzero OPTRT_NIXL_TRANSFER_PROOF "
        f"start+complete pair found; starts={proof_starts} ctx_complete={proof_ctx_complete} gen_complete={proof_gen_complete}"
    )

worker_missing = [row for row in requests if row.get("status") == "ok" and not row.get("worker_id")]
if worker_missing:
    raise SystemExit(f"request-pinning worker metadata missing from {len(worker_missing)} completed requests")

report = {
    "backend": backend,
    "concurrency": concurrency,
    "min_tok_per_user": min_tok,
    "summaries": summaries,
    "positive_transfer_proof_ids": positive[:20],
    "positive_transfer_proof_count": len(positive),
    "request_count": len(requests),
    "slow_lengths": slow,
}
(out / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True))
if slow:
    raise SystemExit(f"tok/s/user floor not met; see {out / 'summary.json'}")
PY_VERIFY

echo "transport benchmark complete: $OUTPUT_DIR"
echo "summary: $OUTPUT_DIR/summary.json"

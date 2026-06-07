#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-dynamo-system}"
CANARY_DGD="${CANARY_DGD:-topo-c1-dp2tp4-hook-canary}"
SNAPSHOT_HOOK_PROOF_DIR="${SNAPSHOT_HOOK_PROOF_DIR:-/tmp/optrt-snapshot-hooks-canary}"
SIGNAL_TIMEOUT_S="${SIGNAL_TIMEOUT_S:-120}"
DRY_RUN=1
STRICT_READINESS=0
ALLOW_NON_CANARY=0

usage() {
  cat <<'EOF'
Usage: deploy/disagg_pd_r20/snapshot_hook_signal_probe.sh [options]

Canary-only proof runner for OPTRT TensorRT-LLM snapshot hook files. It finds
prefill/decode pods for a hook-enabled canary DGD, verifies the OPTRT hook envs,
sends SIGRTMIN+5 (pre-snapshot) and SIGRTMIN+6 (post-restore) to the worker
Python process, waits for JSON ready proof files, and optionally runs
snapshot_readiness.sh for the canary.

Defaults to --dry-run. It refuses the production r20 DGD name unless
--allow-non-canary is passed, and it never deletes, scales, restarts, or patches
workloads.

Options:
  --namespace NS             Kubernetes namespace (default: dynamo-system)
  --canary-dgd NAME          Hook canary DGD name
                             (default: topo-c1-dp2tp4-hook-canary)
  --hook-proof-dir PATH      Host/container path for hook proof files
                             (default: /tmp/optrt-snapshot-hooks-canary)
  --signal-timeout-s SEC     Wait timeout for proof files (default: 120)
  --execute                  Actually send hook signals to canary pods
  --strict-readiness         Run snapshot_readiness.sh --strict after proof
  --allow-non-canary         Permit DGD names without "canary" or "hook"
  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --namespace) NAMESPACE="$2"; shift 2 ;;
    --canary-dgd) CANARY_DGD="$2"; shift 2 ;;
    --hook-proof-dir) SNAPSHOT_HOOK_PROOF_DIR="$2"; shift 2 ;;
    --signal-timeout-s) SIGNAL_TIMEOUT_S="$2"; shift 2 ;;
    --execute) DRY_RUN=0; shift ;;
    --strict-readiness) STRICT_READINESS=1; shift ;;
    --allow-non-canary) ALLOW_NON_CANARY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$SIGNAL_TIMEOUT_S" in
  ''|*[!0-9]*) echo "--signal-timeout-s must be an integer" >&2; exit 2 ;;
esac
case "$SNAPSHOT_HOOK_PROOF_DIR" in
  /*) ;;
  *) echo "--hook-proof-dir must be an absolute path" >&2; exit 2 ;;
esac
if [[ "$ALLOW_NON_CANARY" != 1 && ! "$CANARY_DGD" =~ (canary|hook) ]]; then
  echo "refusing non-canary DGD name without --allow-non-canary: $CANARY_DGD" >&2
  exit 2
fi
if [[ "$CANARY_DGD" == "topo-c1-dp2tp4-disagg-r20" && "$ALLOW_NON_CANARY" != 1 ]]; then
  echo "refusing production r20 DGD without --allow-non-canary" >&2
  exit 2
fi

if [[ -x /usr/local/bin/k3s ]]; then
  KUBECTL=(sudo -E /usr/local/bin/k3s kubectl)
elif command -v kubectl >/dev/null 2>&1; then
  KUBECTL=(kubectl)
else
  echo "kubectl unavailable" >&2
  exit 2
fi

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

echo "snapshot_hook_signal_probe=1"
echo "mode=$([[ "$DRY_RUN" == 1 ]] && echo dry_run || echo execute)"
echo "namespace=$NAMESPACE"
echo "canary_dgd=$CANARY_DGD"
echo "hook_proof_dir=$SNAPSHOT_HOOK_PROOF_DIR"
echo "signal_timeout_s=$SIGNAL_TIMEOUT_S"

ready="$("${KUBECTL[@]}" -n "$NAMESPACE" get dgd "$CANARY_DGD" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || true)"
if [[ "$ready" != "True" ]]; then
  echo "probe_ready=blocked"
  echo "reason=canary_dgd_not_ready:${ready:-missing}"
  exit 3
fi

pod_for() {
  local component="$1"
  "${KUBECTL[@]}" -n "$NAMESPACE" get pods -o name \
    | grep "${CANARY_DGD}-0-${component}" \
    | tail -1
}

prefill_pod="$(pod_for prefill || true)"
decode_pod="$(pod_for decode || true)"
if [[ -z "$prefill_pod" || -z "$decode_pod" ]]; then
  echo "probe_ready=blocked"
  echo "reason=missing_prefill_or_decode_pod:prefill=${prefill_pod:-missing},decode=${decode_pod:-missing}"
  exit 3
fi
echo "prefill_pod=$prefill_pod"
echo "decode_pod=$decode_pod"

require_hook_envs() {
  local component="$1"
  local pod="$2"
  local env_dump
  env_dump="$("${KUBECTL[@]}" -n "$NAMESPACE" get "$pod" -o jsonpath='{range .spec.containers[*].env[*]}{.name}{"="}{.value}{"\n"}{end}')"
  grep -q '^OPTRT_SNAPSHOT_HOOKS=1$' <<<"$env_dump" || {
    echo "probe_ready=blocked"
    echo "reason=missing_OPTRT_SNAPSHOT_HOOKS:$pod"
    exit 3
  }
  grep -q "^OPTRT_SNAPSHOT_HOOK_DIR=${SNAPSHOT_HOOK_PROOF_DIR}$" <<<"$env_dump" || {
    echo "probe_ready=blocked"
    echo "reason=missing_OPTRT_SNAPSHOT_HOOK_DIR:$pod"
    exit 3
  }
  grep -q "^DYN_COMPONENT=${component}$" <<<"$env_dump" || {
    echo "probe_ready=blocked"
    echo "reason=missing_DYN_COMPONENT_${component}:$pod"
    exit 3
  }
}
require_hook_envs prefill "$prefill_pod"
require_hook_envs decode "$decode_pod"

if [[ "$DRY_RUN" == 1 ]]; then
  echo "signals_skipped=1"
  echo "next_execute_command=deploy/disagg_pd_r20/snapshot_hook_signal_probe.sh --canary-dgd $CANARY_DGD --hook-proof-dir $SNAPSHOT_HOOK_PROOF_DIR --execute"
  exit 0
fi

signal_worker() {
  local pod="$1"
  local phase="$2"
  local offset="$3"
  "${KUBECTL[@]}" -n "$NAMESPACE" exec "$pod" -- python3 - "$offset" <<'PY'
import os
import signal
import sys
import time

offset = int(sys.argv[1])
sig_rtmin = signal.SIGRTMIN
sig = (sig_rtmin() if callable(sig_rtmin) else int(sig_rtmin)) + offset

me = os.getpid()
targets = []
for entry in os.listdir("/proc"):
    if not entry.isdigit():
        continue
    pid = int(entry)
    if pid == me:
        continue
    try:
        cmdline = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="ignore")
    except OSError:
        continue
    if "dynamo.trtllm" in cmdline or "tensorrt_llm" in cmdline:
        targets.append((pid, cmdline.strip()))
if not targets:
    raise SystemExit("no TensorRT-LLM worker python process found")
pid, cmdline = sorted(targets)[0]
os.kill(pid, sig)
print(f"sent_signal={sig} pid={pid} cmdline={cmdline[:180]}")
time.sleep(0.2)
PY
  echo "signal_phase=${phase}:pod=${pod}:offset=${offset}"
}

mkdir -p "$SNAPSHOT_HOOK_PROOF_DIR"
probe_marker="$SNAPSHOT_HOOK_PROOF_DIR/optrt_snapshot_probe_${CANARY_DGD}_$$_start.marker"
touch "$probe_marker"
echo "probe_marker=$probe_marker"

signal_worker "$prefill_pod" pre_snapshot 5
signal_worker "$decode_pod" pre_snapshot 5

wait_for_phase_ready() {
  local phase="$1"
  local min_count="$2"
  local label="$3"
  local deadline count err_count

  deadline=$((SECONDS + SIGNAL_TIMEOUT_S))
  while (( SECONDS < deadline )); do
    count="$(find "$SNAPSHOT_HOOK_PROOF_DIR" -maxdepth 1 -type f -newer "$probe_marker" -name "optrt_snapshot_*_${phase}.ready.json" 2>/dev/null | wc -l | tr -d ' ')"
    err_count="$(find "$SNAPSHOT_HOOK_PROOF_DIR" -maxdepth 1 -type f -newer "$probe_marker" -name 'optrt_snapshot_*.error.json' 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "$err_count" != 0 ]]; then
      echo "probe_ready=failed"
      echo "reason=hook_error_files_present"
      find "$SNAPSHOT_HOOK_PROOF_DIR" -maxdepth 1 -type f -newer "$probe_marker" -name 'optrt_snapshot_*.error.json' -print
      exit 4
    fi
    if (( count >= min_count )); then
      printf 'hook_%s_ready_count=%s\n' "$label" "$count"
      return 0
    fi
    sleep 2
  done

  echo "probe_ready=failed"
  echo "reason=timeout_waiting_for_${phase}_proof_files"
  printf 'hook_%s_ready_count=%s\n' "$label" "${count:-0}"
  exit 4
}

wait_for_phase_ready pre_snapshot 2 pre

signal_worker "$prefill_pod" post_restore 6
signal_worker "$decode_pod" post_restore 6
wait_for_phase_ready post_restore 2 post
echo "probe_ready=ok"

readiness_args=(
  --vm local
  --dgd-namespace "$NAMESPACE"
  --dgd-name "$CANARY_DGD"
  --hook-proof-dir "$SNAPSHOT_HOOK_PROOF_DIR"
)
if [[ "$STRICT_READINESS" == 1 ]]; then
  readiness_args+=(--strict)
fi
deploy/disagg_pd_r20/snapshot_readiness.sh "${readiness_args[@]}"

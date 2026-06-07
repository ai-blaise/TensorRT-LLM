#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-dynamo-system}"
SOURCE_DGD="${SOURCE_DGD:-topo-c1-dp2tp4-disagg-r20}"
CANARY_DGD="${CANARY_DGD:-topo-c1-dp2tp4-hook-canary}"
TARGET_NODE="${TARGET_NODE:-a4-us-001-rl9}"
SNAPSHOT_HOOK_PROOF_DIR="${SNAPSHOT_HOOK_PROOF_DIR:-/tmp/optrt-snapshot-hooks-canary}"
SNAPSHOT_HOOK_TIMEOUT_S="${SNAPSHOT_HOOK_TIMEOUT_S:-60}"
GPU_MEMORY_USED_MAX_MIB="${GPU_MEMORY_USED_MAX_MIB:-8192}"
OUT="${OUT:-}"
APPLY=0
SERVER_DRY_RUN=1
WAIT_READY=0
READINESS_STRICT=0

usage() {
  cat <<'EOF'
Usage: deploy/disagg_pd_r20/snapshot_hook_canary.sh [options]

Render and optionally apply an isolated r20 DGD with OPTRT snapshot hooks
enabled on the prefill/decode workers. This helper is fail-closed: the default
mode is render + API-server dry-run only, and --apply refuses to run when the
canary already exists or the target node's GPUs are not idle enough.

Run it on the target B200 VM checkout. It does not build, import, push, delete,
scale, restart, or patch the live r20 DGD.

Options:
  --namespace NS             Kubernetes namespace (default: dynamo-system)
  --source-dgd NAME          Existing DGD used to resolve the active image
                             (default: topo-c1-dp2tp4-disagg-r20)
  --canary-dgd NAME          Hook canary DGD name
                             (default: topo-c1-dp2tp4-hook-canary)
  --target-node NAME         Kubernetes nodeSelector hostname
                             (default: a4-us-001-rl9)
  --hook-proof-dir PATH      Host/container path for hook proof files
                             (default: /tmp/optrt-snapshot-hooks-canary)
  --hook-timeout-s SEC       OPTRT hook quiescence timeout (default: 60)
  --gpu-memory-used-max-mib MIB
                             Maximum per-GPU memory allowed before --apply
                             (default: 8192)
  --out PATH                 Rendered manifest path
                             (default: /tmp/<user>-<canary-dgd>.yaml)
  --apply                    Apply the canary after all safety checks pass
  --wait-ready               After --apply, wait for the canary DGD Ready=True
  --readiness-strict         After --apply/--wait-ready, run snapshot_readiness
                             with --strict for the canary proof directory
  --no-server-dry-run        Skip API-server dry-run before apply
  -h, --help                 Show this help

Environment overrides use the same names as the options.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --namespace) NAMESPACE="$2"; shift 2 ;;
    --source-dgd) SOURCE_DGD="$2"; shift 2 ;;
    --canary-dgd) CANARY_DGD="$2"; shift 2 ;;
    --target-node) TARGET_NODE="$2"; shift 2 ;;
    --hook-proof-dir) SNAPSHOT_HOOK_PROOF_DIR="$2"; shift 2 ;;
    --hook-timeout-s) SNAPSHOT_HOOK_TIMEOUT_S="$2"; shift 2 ;;
    --gpu-memory-used-max-mib) GPU_MEMORY_USED_MAX_MIB="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --apply) APPLY=1; shift ;;
    --wait-ready) WAIT_READY=1; shift ;;
    --readiness-strict) READINESS_STRICT=1; shift ;;
    --no-server-dry-run) SERVER_DRY_RUN=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$APPLY:$WAIT_READY" in
  0:1) echo "--wait-ready requires --apply" >&2; exit 2 ;;
esac
case "$APPLY:$READINESS_STRICT" in
  0:1) echo "--readiness-strict requires --apply" >&2; exit 2 ;;
esac
case "$SNAPSHOT_HOOK_PROOF_DIR" in
  /*) ;;
  *) echo "--hook-proof-dir must be an absolute path" >&2; exit 2 ;;
esac
case "$GPU_MEMORY_USED_MAX_MIB" in
  ''|*[!0-9]*) echo "--gpu-memory-used-max-mib must be an integer" >&2; exit 2 ;;
esac

if (( ${#CANARY_DGD} + 8 > 45 )); then
  echo "canary DGD name too long for r20 Frontend pod naming: ${#CANARY_DGD}+8 > 45 ($CANARY_DGD)" >&2
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

if [[ -z "$OUT" ]]; then
  OUT="/tmp/${USER:-optrt}-${CANARY_DGD}.yaml"
fi

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

echo "snapshot_hook_canary=1"
echo "mode=$([[ "$APPLY" == 1 ]] && echo apply || echo dry_run)"
echo "namespace=$NAMESPACE"
echo "source_dgd=$SOURCE_DGD"
echo "canary_dgd=$CANARY_DGD"
echo "target_node=$TARGET_NODE"
echo "hook_proof_dir=$SNAPSHOT_HOOK_PROOF_DIR"
echo "gpu_memory_used_max_mib=$GPU_MEMORY_USED_MAX_MIB"

render_args=(
  --namespace "$NAMESPACE"
  --image-from-dgd "$SOURCE_DGD"
  --dgd-name "$CANARY_DGD"
  --target-node "$TARGET_NODE"
  --enable-snapshot-hooks
  --snapshot-hook-proof-dir "$SNAPSHOT_HOOK_PROOF_DIR"
  --snapshot-hook-timeout-s "$SNAPSHOT_HOOK_TIMEOUT_S"
  --out "$OUT"
)
if [[ "$SERVER_DRY_RUN" == 1 ]]; then
  render_args+=(--server-dry-run)
fi

deploy/disagg_pd_r20/render_dgd.sh "${render_args[@]}"

if [[ "$APPLY" != 1 ]]; then
  echo "apply_skipped=1"
  echo "next_apply_command=deploy/disagg_pd_r20/snapshot_hook_canary.sh --source-dgd $SOURCE_DGD --canary-dgd $CANARY_DGD --target-node $TARGET_NODE --hook-proof-dir $SNAPSHOT_HOOK_PROOF_DIR --apply --wait-ready"
  exit 0
fi

if "${KUBECTL[@]}" -n "$NAMESPACE" get dgd "$CANARY_DGD" >/dev/null 2>&1; then
  echo "apply_safety=blocked"
  echo "reason=canary_dgd_already_exists:$CANARY_DGD"
  exit 3
fi
existing_canary_pods="$("${KUBECTL[@]}" -n "$NAMESPACE" get pods --no-headers 2>/dev/null | awk -v dgd="$CANARY_DGD" '$1 ~ dgd {print $1":"$3}' | paste -sd, - || true)"
if [[ -n "$existing_canary_pods" ]]; then
  echo "apply_safety=blocked"
  echo "reason=canary_pods_already_exist:$existing_canary_pods"
  exit 3
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "apply_safety=blocked"
  echo "reason=nvidia_smi_unavailable"
  exit 3
fi
busy_gpus="$(
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, -v limit="$GPU_MEMORY_USED_MAX_MIB" '
        {
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
          if (($2 + 0) > limit) {
            printf "%s:%sMiB,", $1, $2
          }
        }
      '
)"
busy_gpus="${busy_gpus%,}"
if [[ -n "$busy_gpus" ]]; then
  echo "apply_safety=blocked"
  echo "reason=gpu_memory_not_idle:$busy_gpus"
  exit 3
fi

echo "apply_safety=passed"
"${KUBECTL[@]}" -n "$NAMESPACE" apply -f "$OUT"

if [[ "$WAIT_READY" == 1 ]]; then
  "${KUBECTL[@]}" -n "$NAMESPACE" wait "dgd/$CANARY_DGD" \
    --for=jsonpath='{.status.conditions[?(@.type=="Ready")].status}'=True \
    --timeout=1800s
fi

if [[ "$READINESS_STRICT" == 1 ]]; then
  deploy/disagg_pd_r20/snapshot_readiness.sh \
    --vm local \
    --dgd-namespace "$NAMESPACE" \
    --dgd-name "$CANARY_DGD" \
    --target-node "$TARGET_NODE" \
    --hook-proof-dir "$SNAPSHOT_HOOK_PROOF_DIR" \
    --strict
fi

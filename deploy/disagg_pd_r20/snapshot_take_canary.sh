#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-dynamo-system}"
CANARY_DGD="${CANARY_DGD:-topo-c1-dp2tp4-hook-canary}"
SNAPSHOT_HOOK_PROOF_DIR="${SNAPSHOT_HOOK_PROOF_DIR:-/tmp/optrt-snapshot-hooks-canary}"
SNAPSHOT_NAMESPACE="${SNAPSHOT_NAMESPACE:-dynamo-system}"
SNAPSHOT_NAME_PREFIX="${SNAPSHOT_NAME_PREFIX:-${CANARY_DGD}}"
OCI_REPO="${OCI_REPO:-localhost:5000/optrt-snapshots/${CANARY_DGD}}"
TAG_TEMPLATE="${TAG_TEMPLATE:-}"
if [[ -z "$TAG_TEMPLATE" ]]; then
  TAG_TEMPLATE='${component}-${timestamp}'
fi
HOOK_PROOF_MAX_AGE_S="${HOOK_PROOF_MAX_AGE_S:-1800}"
MAX_IN_FLIGHT="${MAX_IN_FLIGHT:-0}"
PRE_STOP_GRACE_PERIOD_SECONDS="${PRE_STOP_GRACE_PERIOD_SECONDS:-30}"
VALIDATION_PROMPT="${VALIDATION_PROMPT:-OPTRT snapshot restore probe. Count to three.}"
VALIDATION_MAX_NEW_TOKENS="${VALIDATION_MAX_NEW_TOKENS:-16}"
VALIDATION_TIMEOUT_SECONDS="${VALIDATION_TIMEOUT_SECONDS:-300}"
OUT_DIR="${OUT_DIR:-/tmp/optrt-snapshot-take-canary}"
SERVER_DRY_RUN=1
APPLY=0
ALLOW_NON_CANARY=0

usage() {
  cat <<'EOF'
Usage: deploy/disagg_pd_r20/snapshot_take_canary.sh [options]

Render and optionally create component DynamoGraphDeploymentSnapshot resources
for a hook-proven r20 canary. This helper is fail-closed: it requires a Ready
canary DGD, recent pre/post OPTRT hook proof files, and API-server dry-run by
default. It refuses production/non-canary DGD names unless explicitly overridden.

Options:
  --namespace NS             Namespace containing the canary DGD
                             (default: dynamo-system)
  --snapshot-namespace NS    Namespace for snapshot resources
                             (default: dynamo-system)
  --canary-dgd NAME          Hook canary DGD name
                             (default: topo-c1-dp2tp4-hook-canary)
  --hook-proof-dir PATH      Host-visible hook proof directory
                             (default: /tmp/optrt-snapshot-hooks-canary)
  --hook-proof-max-age-s SEC Maximum age for hook ready proof files
                             (default: 1800)
  --snapshot-name-prefix PFX Snapshot resource name prefix
                             (default: <canary-dgd>)
  --oci-repo REPO            Snapshot OCI repository
                             (default: localhost:5000/optrt-snapshots/<canary>)
  --tag-template TEMPLATE    Snapshot tagTemplate (default: component-timestamp)
  --out-dir DIR              Rendered YAML output dir
                             (default: /tmp/optrt-snapshot-take-canary)
  --apply                    Create snapshot resources after all gates pass
  --no-server-dry-run        Skip API-server dry-run before apply
  --allow-non-canary         Permit DGD names without "canary" or "hook"
  -h, --help                 Show this help

Environment overrides use the same names as the options.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --namespace) NAMESPACE="$2"; shift 2 ;;
    --snapshot-namespace) SNAPSHOT_NAMESPACE="$2"; shift 2 ;;
    --canary-dgd) CANARY_DGD="$2"; shift 2 ;;
    --hook-proof-dir) SNAPSHOT_HOOK_PROOF_DIR="$2"; shift 2 ;;
    --hook-proof-max-age-s) HOOK_PROOF_MAX_AGE_S="$2"; shift 2 ;;
    --snapshot-name-prefix) SNAPSHOT_NAME_PREFIX="$2"; shift 2 ;;
    --oci-repo) OCI_REPO="$2"; shift 2 ;;
    --tag-template) TAG_TEMPLATE="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --apply) APPLY=1; shift ;;
    --no-server-dry-run) SERVER_DRY_RUN=0; shift ;;
    --allow-non-canary) ALLOW_NON_CANARY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$SNAPSHOT_HOOK_PROOF_DIR" in
  /*) ;;
  *) echo "--hook-proof-dir must be an absolute path" >&2; exit 2 ;;
esac
case "$HOOK_PROOF_MAX_AGE_S" in
  ''|*[!0-9]*) echo "--hook-proof-max-age-s must be an integer" >&2; exit 2 ;;
esac
if [[ "$OCI_REPO" =~ [[:space:]] || "$TAG_TEMPLATE" =~ [[:space:]] ]]; then
  echo "--oci-repo and --tag-template must not contain whitespace" >&2
  exit 2
fi
if [[ "$VALIDATION_PROMPT" == *$'\n'* || "$VALIDATION_PROMPT" == *'"'* ]]; then
  echo "VALIDATION_PROMPT must not contain newlines or double quotes" >&2
  exit 2
fi
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

echo "snapshot_take_canary=1"
echo "mode=$([[ "$APPLY" == 1 ]] && echo apply || echo dry_run)"
echo "namespace=$NAMESPACE"
echo "snapshot_namespace=$SNAPSHOT_NAMESPACE"
echo "canary_dgd=$CANARY_DGD"
echo "hook_proof_dir=$SNAPSHOT_HOOK_PROOF_DIR"
echo "oci_repo=$OCI_REPO"
echo "tag_template=$TAG_TEMPLATE"

ready="$("${KUBECTL[@]}" -n "$NAMESPACE" get dgd "$CANARY_DGD" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || true)"
if [[ "$ready" != "True" ]]; then
  echo "snapshot_take_ready=blocked"
  echo "reason=canary_dgd_not_ready:${ready:-missing}"
  exit 3
fi

if [[ ! -d "$SNAPSHOT_HOOK_PROOF_DIR" ]]; then
  echo "snapshot_take_ready=blocked"
  echo "reason=hook_proof_dir_missing:$SNAPSHOT_HOOK_PROOF_DIR"
  exit 3
fi
proof_cutoff="$(mktemp /tmp/optrt-snapshot-proof-cutoff.XXXXXX)"
touch -d "@$(( $(date +%s) - HOOK_PROOF_MAX_AGE_S ))" "$proof_cutoff"
pre_count="$(find "$SNAPSHOT_HOOK_PROOF_DIR" -maxdepth 1 -type f -newer "$proof_cutoff" -name 'optrt_snapshot_*_pre_snapshot.ready.json' 2>/dev/null | wc -l | tr -d ' ')"
post_count="$(find "$SNAPSHOT_HOOK_PROOF_DIR" -maxdepth 1 -type f -newer "$proof_cutoff" -name 'optrt_snapshot_*_post_restore.ready.json' 2>/dev/null | wc -l | tr -d ' ')"
err_count="$(find "$SNAPSHOT_HOOK_PROOF_DIR" -maxdepth 1 -type f -newer "$proof_cutoff" -name 'optrt_snapshot_*.error.json' 2>/dev/null | wc -l | tr -d ' ')"
rm -f "$proof_cutoff"
echo "hook_pre_ready_recent_count=$pre_count"
echo "hook_post_ready_recent_count=$post_count"
echo "hook_error_recent_count=$err_count"
if [[ "$err_count" != 0 ]]; then
  echo "snapshot_take_ready=blocked"
  echo "reason=recent_hook_error_files_present"
  exit 3
fi
if (( pre_count < 2 || post_count < 2 )); then
  echo "snapshot_take_ready=blocked"
  echo "reason=missing_recent_hook_pre_post_proof"
  exit 3
fi

mkdir -p "$OUT_DIR"
render_snapshot() {
  local component="$1"
  local out="$OUT_DIR/${SNAPSHOT_NAME_PREFIX}-${component}-snapshot.yaml"
  cat >"$out" <<EOF_YAML
apiVersion: snapshots.ai-blaise.io/v1alpha1
kind: DynamoGraphDeploymentSnapshot
metadata:
  name: ${SNAPSHOT_NAME_PREFIX}-${component}
  namespace: ${SNAPSHOT_NAMESPACE}
  labels:
    ai-blaise.io/source-dgd: ${CANARY_DGD}
    ai-blaise.io/component: ${component}
    ai-blaise.io/proof: optrt-r20-hook-canary
spec:
  source:
    dynamoGraphDeployment: ${CANARY_DGD}
    component: ${component}
  drain:
    maxInFlight: ${MAX_IN_FLIGHT}
    preStopGracePeriodSeconds: ${PRE_STOP_GRACE_PERIOD_SECONDS}
  storage:
    ociRepo: ${OCI_REPO}
    tagTemplate: ${TAG_TEMPLATE}
  validation:
    probe:
      type: openai-completion
      prompt: "${VALIDATION_PROMPT}"
      maxNewTokens: ${VALIDATION_MAX_NEW_TOKENS}
      timeoutSeconds: ${VALIDATION_TIMEOUT_SECONDS}
EOF_YAML
  echo "$out"
}

prefill_yaml="$(render_snapshot prefill)"
decode_yaml="$(render_snapshot decode)"
echo "prefill_snapshot_yaml=$prefill_yaml"
echo "decode_snapshot_yaml=$decode_yaml"

if [[ "$SERVER_DRY_RUN" == 1 ]]; then
  "${KUBECTL[@]}" -n "$SNAPSHOT_NAMESPACE" apply --dry-run=server -f "$prefill_yaml"
  "${KUBECTL[@]}" -n "$SNAPSHOT_NAMESPACE" apply --dry-run=server -f "$decode_yaml"
fi

if [[ "$APPLY" != 1 ]]; then
  echo "apply_skipped=1"
  echo "next_apply_command=deploy/disagg_pd_r20/snapshot_take_canary.sh --canary-dgd $CANARY_DGD --hook-proof-dir $SNAPSHOT_HOOK_PROOF_DIR --oci-repo $OCI_REPO --apply"
  exit 0
fi

for component in prefill decode; do
  name="${SNAPSHOT_NAME_PREFIX}-${component}"
  if "${KUBECTL[@]}" -n "$SNAPSHOT_NAMESPACE" get dgds "$name" >/dev/null 2>&1; then
    echo "snapshot_take_ready=blocked"
    echo "reason=snapshot_resource_already_exists:$SNAPSHOT_NAMESPACE/$name"
    exit 3
  fi
done

echo "snapshot_take_ready=ok"
"${KUBECTL[@]}" -n "$SNAPSHOT_NAMESPACE" apply -f "$prefill_yaml"
"${KUBECTL[@]}" -n "$SNAPSHOT_NAMESPACE" apply -f "$decode_yaml"

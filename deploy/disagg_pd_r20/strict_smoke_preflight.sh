#!/usr/bin/env bash
set -euo pipefail

VM_HOST="${VM_HOST:-local}"
TARGET_NODE="${TARGET_NODE:-a4-us-001-rl9}"
DGD_NAME="${DGD_NAME:-topo-c1-dp2tp4-disagg-r20}"
IMAGE="${IMAGE:-}"
IMAGE_FROM_DGD="${IMAGE_FROM_DGD:-$DGD_NAME}"
IMAGE_PULL_POLICY="${IMAGE_PULL_POLICY:-IfNotPresent}"
REQUIRE_CACHES="${REQUIRE_CACHES:-triton,deep_gemm}"
REGISTRY_REPO="${REGISTRY_REPO:-local/dynamo-trtllm-optrt-custom}"
REGISTRY_TAG_TAIL="${REGISTRY_TAG_TAIL:-12}"
OUT_DIR="${OUT_DIR:-/tmp/r20-strict-smoke-preflight-$(date -u +%Y%m%dT%H%M%SZ)}"
SMC_GATE_MODE="${SMC_GATE_MODE:-required}"
NAMESPACE="${NAMESPACE:-dynamo-system}"
SKIP_CACHE_REQUIRE=0
SKIP_NIXL_GATE_AUDIT=0
SKIP_PREWARM_SERVER_DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: deploy/disagg_pd_r20/strict_smoke_preflight.sh [options]

Run the non-mutating R20 strict-smoke/C16 handoff preflight bundle. It verifies
exact image handoff, renders and API-server-validates the DGD, validates the
prewarm Job with server dry-run, optionally requires populated persistent kernel
caches, and writes proof logs plus the exact next commands to an output dir.

This helper never applies a DGD, creates/deletes a Job, builds/imports/pushes an
image, restarts pods, or mutates the live workload.

Options:
  --image IMAGE          Exact image to validate and render
  --image-from-dgd NAME  Use first running pod image whose name contains NAME
                         when --image is omitted (default: $DGD_NAME)
  --vm HOST              Target VM for helper calls; use local on the VM
                         (default: local)
  --target-node NAME     Rendered nodeSelector hostname (default: $TARGET_NODE)
  --dgd-name NAME        Existing DGD name / render source prefix (default: r20)
  --image-pull-policy P  Pull policy for rendered DGD/prewarm validation
                         (default: IfNotPresent)
  --require-caches LIST  Comma-separated /var/lib/optrt-cache subdirs that must
                         contain files (default: triton,deep_gemm)
  --smc-gate-mode MODE   required or deferred for the local NIXL/custom-stack
                         gate audit (default: required)
  --skip-nixl-gate-audit Do not run the local NIXL/custom-stack gate audit
  --registry-repo NAME  Local registry repository for cache/image report
  --registry-tag-tail N Number of recent-looking registry tags to report
  --skip-cache-require   Do not fail closed on cache population
  --skip-prewarm-server-dry-run
                         Skip prewarm Job API-server validation
  --namespace NAME       Namespace for image lookup and server dry-runs
                         (default: dynamo-system)
  --out-dir PATH         Proof artifact directory (default: /tmp/r20-...)
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --image-from-dgd) IMAGE_FROM_DGD="$2"; shift 2 ;;
    --vm) VM_HOST="$2"; shift 2 ;;
    --target-node) TARGET_NODE="$2"; shift 2 ;;
    --dgd-name) DGD_NAME="$2"; shift 2 ;;
    --image-pull-policy) IMAGE_PULL_POLICY="$2"; shift 2 ;;
    --require-caches) REQUIRE_CACHES="$2"; shift 2 ;;
    --registry-repo) REGISTRY_REPO="$2"; shift 2 ;;
    --smc-gate-mode) SMC_GATE_MODE="$2"; shift 2 ;;
    --skip-nixl-gate-audit) SKIP_NIXL_GATE_AUDIT=1; shift ;;
    --registry-tag-tail) REGISTRY_TAG_TAIL="$2"; shift 2 ;;
    --skip-cache-require) SKIP_CACHE_REQUIRE=1; shift ;;
    --skip-prewarm-server-dry-run) SKIP_PREWARM_SERVER_DRY_RUN=1; shift ;;
    --namespace) NAMESPACE="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$IMAGE_PULL_POLICY" in
  Always|IfNotPresent|Never) ;;
  *) echo "unsupported --image-pull-policy: $IMAGE_PULL_POLICY" >&2; exit 2 ;;
esac
case "$SMC_GATE_MODE" in
  required|deferred) ;;
  *) echo "unsupported --smc-gate-mode: $SMC_GATE_MODE" >&2; exit 2 ;;
esac

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"
mkdir -p "$OUT_DIR"
source "deploy/disagg_pd_r20/target_node_guard.sh"
optrt_r20_reject_disallowed_target_node "$TARGET_NODE"

run_step() {
  local name="$1"
  shift
  printf 'preflight_step=%s\n' "$name" | tee -a "$OUT_DIR/summary.log"
  "$@" >"$OUT_DIR/${name}.out" 2>"$OUT_DIR/${name}.err"
}

if [[ "$SKIP_NIXL_GATE_AUDIT" != 1 ]]; then
  run_step nixl_gate_local_audit \
    env \
      NIXL_AUDIT_MODE=local \
      SMC_GATE_MODE="$SMC_GATE_MODE" \
      CHECK_RUNTIME_LIBS=0 \
      deploy/disagg_pd_r20/audit_nixl_gate_readiness.sh
fi

if [[ -z "$IMAGE" ]]; then
  run_step image_handoff_from_dgd \
    deploy/disagg_pd_r20/check_image_handoff.sh \
      --vm "$VM_HOST" \
      --image-from-dgd "$IMAGE_FROM_DGD" \
      --mode auto \
      --require
  IMAGE="$(awk -F= '$1 == "image" {print $2; exit}' "$OUT_DIR/image_handoff_from_dgd.out")"
else
  run_step image_handoff \
    deploy/disagg_pd_r20/check_image_handoff.sh \
      --vm "$VM_HOST" \
      --image "$IMAGE" \
      --mode auto \
      --require
fi

if [[ -z "$IMAGE" ]]; then
  echo "failed to resolve image" >&2
  exit 2
fi

printf 'resolved_image=%s\n' "$IMAGE" | tee -a "$OUT_DIR/summary.log"

render_name="r20-preflight"
run_step render_dgd_server_dry_run \
  deploy/disagg_pd_r20/render_dgd.sh \
    --image "$IMAGE" \
    --target-node "$TARGET_NODE" \
    --dgd-name "$render_name" \
    --image-pull-policy "$IMAGE_PULL_POLICY" \
    --namespace "$NAMESPACE" \
    --out "$OUT_DIR/${render_name}.yaml" \
    --server-dry-run

if [[ "$SKIP_PREWARM_SERVER_DRY_RUN" != 1 ]]; then
  run_step prewarm_server_dry_run \
    deploy/disagg_pd_r20/prewarm_caches.sh \
      --vm "$VM_HOST" \
      --image "$IMAGE" \
      --image-pull-policy "$IMAGE_PULL_POLICY" \
      --target-node "$TARGET_NODE" \
      --require-image-handoff \
      --server-dry-run
fi

if [[ "$SKIP_CACHE_REQUIRE" != 1 && -n "$REQUIRE_CACHES" ]]; then
  run_step cache_requirements \
    deploy/disagg_pd_r20/cache_report.sh \
      --vm "$VM_HOST" \
      --dgd-name "$DGD_NAME" \
      --image-filter dynamo-trtllm-optrt-custom \
      --registry-repo "$REGISTRY_REPO" \
      --registry-tag-tail "$REGISTRY_TAG_TAIL" \
      --require-populated "$REQUIRE_CACHES"
else
  run_step cache_report \
    deploy/disagg_pd_r20/cache_report.sh \
      --vm "$VM_HOST" \
      --dgd-name "$DGD_NAME" \
      --image-filter dynamo-trtllm-optrt-custom \
      --registry-repo "$REGISTRY_REPO" \
      --registry-tag-tail "$REGISTRY_TAG_TAIL"
fi

audit_block=""
if [[ "$SKIP_NIXL_GATE_AUDIT" != 1 ]]; then
  audit_block="
# Re-run the local NIXL/custom-stack config audit before consuming rollout time.
NIXL_AUDIT_MODE=local \\
  SMC_GATE_MODE=$SMC_GATE_MODE \\
  CHECK_RUNTIME_LIBS=0 \\
  deploy/disagg_pd_r20/audit_nixl_gate_readiness.sh
"
fi

cat >"$OUT_DIR/next_commands.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
$audit_block

# Reconfirm exact image handoff before consuming rollout/prewarm time.
deploy/disagg_pd_r20/check_image_handoff.sh \\
  --vm $VM_HOST \\
  --image '$IMAGE' \\
  --mode auto \\
  --require

# Validate the prewarm Job spec without creating a Job.
deploy/disagg_pd_r20/prewarm_caches.sh \\
  --vm $VM_HOST \\
  --image '$IMAGE' \\
  --image-pull-policy $IMAGE_PULL_POLICY \\
  --target-node $TARGET_NODE \\
  --require-image-handoff \\
  --server-dry-run

# Reuse the validated image in the normal fast-iterate flow. Remove --dry-run
# only when the strict smoke owner is ready to prewarm/deploy.
deploy/disagg_pd_r20/fast_iterate.sh \\
  --vm $VM_HOST \\
  --target-node $TARGET_NODE \\
  --deploy-image '$IMAGE' \\
  --use-local-registry \\
  --require-image-handoff \\
  --prewarm \\
  --deploy \\
  --dry-run

# After warmup, require expected persistent kernel caches.
deploy/disagg_pd_r20/cache_report.sh \\
  --vm $VM_HOST \\
  --dgd-name $DGD_NAME \\
  --image-filter dynamo-trtllm-optrt-custom \\
  --registry-repo $REGISTRY_REPO \\
  --registry-tag-tail $REGISTRY_TAG_TAIL \\
  --require-populated $REQUIRE_CACHES
EOF
chmod +x "$OUT_DIR/next_commands.sh"

{
  printf 'strict_smoke_preflight=ok\n'
  printf 'out_dir=%s\n' "$OUT_DIR"
  printf 'image=%s\n' "$IMAGE"
  printf 'target_node=%s\n' "$TARGET_NODE"
  printf 'dgd_name=%s\n' "$DGD_NAME"
  printf 'image_pull_policy=%s\n' "$IMAGE_PULL_POLICY"
  printf 'require_caches=%s\n' "$REQUIRE_CACHES"
  printf 'smc_gate_mode=%s\n' "$SMC_GATE_MODE"
  printf 'skip_nixl_gate_audit=%s\n' "$SKIP_NIXL_GATE_AUDIT"
  printf 'registry_repo=%s\n' "$REGISTRY_REPO"
  printf 'registry_tag_tail=%s\n' "$REGISTRY_TAG_TAIL"
  printf 'next_commands=%s\n' "$OUT_DIR/next_commands.sh"
} | tee -a "$OUT_DIR/summary.log"

#!/usr/bin/env bash
set -euo pipefail

TARGET_NODE="${TARGET_NODE:-a4-us-001-rl9}"
DGD_NAME="${DGD_NAME:-topo-c1-dp2tp4-disagg-r20}"
IMAGE="${IMAGE:-}"
IMAGE_FROM_DGD="${IMAGE_FROM_DGD:-}"
IMAGE_PULL_POLICY="${IMAGE_PULL_POLICY:-template}"
OUT="${OUT:-}"
SERVER_DRY_RUN=0
NAMESPACE="${NAMESPACE:-dynamo-system}"

usage() {
  cat <<'EOF'
Usage: deploy/disagg_pd_r20/render_dgd.sh [options]

Render the r20 DGD manifest with an existing image and optionally ask the API
server to validate it with kubectl apply --dry-run=server. This helper never
applies, deletes, builds, imports, pushes, or restarts anything.

Options:
  --image IMAGE          Unified image to render into the DGD
  --image-from-dgd NAME  Read the first running pod image whose name contains NAME
                         and reuse it as --image
  --target-node NAME     Kubernetes nodeSelector hostname (default: $TARGET_NODE)
  --dgd-name NAME        Rendered DGD/ConfigMap name (default: $DGD_NAME)
  --image-pull-policy P  Override all imagePullPolicy entries; use template to
                         keep the manifest value (default: template)
  --namespace NAME       Namespace for --image-from-dgd and server dry-run
                         (default: dynamo-system)
  --out PATH             Output path (default: /tmp/<dgd-name>-render.yaml)
  --server-dry-run       Run kubectl apply --dry-run=server against the output
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --image-from-dgd) IMAGE_FROM_DGD="$2"; shift 2 ;;
    --target-node) TARGET_NODE="$2"; shift 2 ;;
    --dgd-name) DGD_NAME="$2"; shift 2 ;;
    --image-pull-policy) IMAGE_PULL_POLICY="$2"; shift 2 ;;
    --namespace) NAMESPACE="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --server-dry-run) SERVER_DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$IMAGE_PULL_POLICY" in
  template|Always|IfNotPresent|Never) ;;
  *) echo "unsupported --image-pull-policy: $IMAGE_PULL_POLICY" >&2; exit 2 ;;
esac

if (( ${#DGD_NAME} + 8 > 45 )); then
  echo "DGD name too long for r20 Frontend pod naming: ${#DGD_NAME}+8 > 45 ($DGD_NAME)" >&2
  exit 2
fi

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

if [[ -z "$IMAGE" && -n "$IMAGE_FROM_DGD" ]]; then
  if [[ -x /usr/local/bin/k3s ]]; then
    KUBECTL=(sudo -E /usr/local/bin/k3s kubectl)
  elif command -v kubectl >/dev/null 2>&1; then
    KUBECTL=(kubectl)
  else
    echo "kubectl unavailable for --image-from-dgd" >&2
    exit 2
  fi
  IMAGE="$(
    "${KUBECTL[@]}" -n "$NAMESPACE" get pods \
      -o custom-columns='POD:.metadata.name,PHASE:.status.phase,IMAGE:.spec.containers[*].image' \
      --no-headers 2>/dev/null \
      | awk -v dgd="$IMAGE_FROM_DGD" '$1 ~ dgd && $2 == "Running" {print $3; exit}'
  )"
  if [[ -z "$IMAGE" ]]; then
    echo "no active pod image found for --image-from-dgd=$IMAGE_FROM_DGD in namespace=$NAMESPACE" >&2
    exit 2
  fi
fi

if [[ -z "$IMAGE" ]]; then
  echo "--image or --image-from-dgd is required" >&2
  exit 2
fi

if [[ -z "$OUT" ]]; then
  OUT="/tmp/${DGD_NAME}-render.yaml"
fi
out_dir="$(dirname "$OUT")"
if [[ ! -d "$out_dir" ]]; then
  echo "output directory does not exist: $out_dir" >&2
  exit 2
fi
if [[ -e "$OUT" && ! -w "$OUT" ]]; then
  echo "output path exists but is not writable: $OUT" >&2
  exit 2
fi
if [[ ! -e "$OUT" && ! -w "$out_dir" ]]; then
  echo "output directory is not writable: $out_dir" >&2
  exit 2
fi

TARGET_NODE="$TARGET_NODE" \
UNIFIED_IMAGE="$IMAGE" \
DGD_NAME="$DGD_NAME" \
IMAGE_PULL_POLICY="$IMAGE_PULL_POLICY" \
OUT="$OUT" \
python3 - <<'INNER_PY'
import os
import re
from pathlib import Path

src = Path("deploy/disagg_pd_r20/topo-c1-dp2tp4-disagg-r20.yaml")
text = src.read_text()
text = text.replace("${TARGET_NODE}", os.environ["TARGET_NODE"])
text = text.replace("${UNIFIED_IMAGE}", os.environ["UNIFIED_IMAGE"])
text = text.replace("topo-c1-dp2tp4-disagg-r20", os.environ["DGD_NAME"])
policy = os.environ["IMAGE_PULL_POLICY"]
if policy != "template":
    text = re.sub(r"imagePullPolicy: (Always|IfNotPresent|Never)", f"imagePullPolicy: {policy}", text)
Path(os.environ["OUT"]).write_text(text)
INNER_PY

printf 'rendered_dgd=%s\n' "$OUT"
printf 'dgd_name=%s\n' "$DGD_NAME"
printf 'target_node=%s\n' "$TARGET_NODE"
printf 'image=%s\n' "$IMAGE"
printf 'image_pull_policy=%s\n' "$IMAGE_PULL_POLICY"

if [[ "$SERVER_DRY_RUN" == 1 ]]; then
  if [[ -x /usr/local/bin/k3s ]]; then
    sudo -E /usr/local/bin/k3s kubectl -n "$NAMESPACE" apply --dry-run=server -f "$OUT"
  elif command -v kubectl >/dev/null 2>&1; then
    kubectl -n "$NAMESPACE" apply --dry-run=server -f "$OUT"
  else
    echo "kubectl unavailable for --server-dry-run" >&2
    exit 2
  fi
fi

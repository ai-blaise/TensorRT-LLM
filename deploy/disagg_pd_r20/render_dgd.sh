#!/usr/bin/env bash
set -euo pipefail

TARGET_NODE="${TARGET_NODE:-a4-us-001-rl9}"
DGD_NAME="${DGD_NAME:-topo-c1-dp2tp4-disagg-r20}"
IMAGE="${IMAGE:-}"
IMAGE_FROM_DGD="${IMAGE_FROM_DGD:-}"
IMAGE_PULL_POLICY="${IMAGE_PULL_POLICY:-template}"
OUT="${OUT:-}"
ENABLE_SNAPSHOT_HOOKS="${ENABLE_SNAPSHOT_HOOKS:-0}"
SNAPSHOT_HOOK_PROOF_DIR="${SNAPSHOT_HOOK_PROOF_DIR:-/tmp/optrt-snapshot-hooks}"
SNAPSHOT_HOOK_TIMEOUT_S="${SNAPSHOT_HOOK_TIMEOUT_S:-60}"
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
  --enable-snapshot-hooks
                         Render OPTRT pre/post snapshot hook envs and a
                         hostPath proof directory on prefill/decode workers.
                         Defaults off so production r20 renders are unchanged.
  --snapshot-hook-proof-dir PATH
                         Host/container path for hook proof files
                         (default: $SNAPSHOT_HOOK_PROOF_DIR)
  --snapshot-hook-timeout-s SEC
                         OPTRT hook quiescence timeout
                         (default: $SNAPSHOT_HOOK_TIMEOUT_S)
  --namespace NAME       Namespace for --image-from-dgd and server dry-run
                         (default: dynamo-system)
  --out PATH             Output path (default: /tmp/<user>-<dgd-name>-render.yaml)
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
    --enable-snapshot-hooks) ENABLE_SNAPSHOT_HOOKS=1; shift ;;
    --snapshot-hook-proof-dir) SNAPSHOT_HOOK_PROOF_DIR="$2"; shift 2 ;;
    --snapshot-hook-timeout-s) SNAPSHOT_HOOK_TIMEOUT_S="$2"; shift 2 ;;
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
case "$ENABLE_SNAPSHOT_HOOKS" in
  0|1) ;;
  *) echo "ENABLE_SNAPSHOT_HOOKS must be 0 or 1" >&2; exit 2 ;;
esac
if [[ "$ENABLE_SNAPSHOT_HOOKS" == 1 ]]; then
  case "$SNAPSHOT_HOOK_PROOF_DIR" in
    /*) ;;
    *) echo "--snapshot-hook-proof-dir must be an absolute path" >&2; exit 2 ;;
  esac
  if [[ "$SNAPSHOT_HOOK_PROOF_DIR" == *$'\n'* || "$SNAPSHOT_HOOK_PROOF_DIR" == *"'"* || "$SNAPSHOT_HOOK_PROOF_DIR" =~ [[:space:]] ]]; then
    echo "--snapshot-hook-proof-dir must not contain whitespace, newlines, or single quotes" >&2
    exit 2
  fi
  case "$SNAPSHOT_HOOK_TIMEOUT_S" in
    ''|*[!0-9.]*)
      echo "--snapshot-hook-timeout-s must be numeric" >&2
      exit 2
      ;;
  esac
fi

if (( ${#DGD_NAME} + 8 > 45 )); then
  echo "DGD name too long for r20 Frontend pod naming: ${#DGD_NAME}+8 > 45 ($DGD_NAME)" >&2
  exit 2
fi

source "$(dirname "${BASH_SOURCE[0]}")/target_node_guard.sh"
optrt_r20_reject_disallowed_target_node "$TARGET_NODE"

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
  OUT="/tmp/${USER:-optrt}-${DGD_NAME}-render.yaml"
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
ENABLE_SNAPSHOT_HOOKS="$ENABLE_SNAPSHOT_HOOKS" \
SNAPSHOT_HOOK_PROOF_DIR="$SNAPSHOT_HOOK_PROOF_DIR" \
SNAPSHOT_HOOK_TIMEOUT_S="$SNAPSHOT_HOOK_TIMEOUT_S" \
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
enable_snapshot_hooks = os.environ["ENABLE_SNAPSHOT_HOOKS"] == "1"
hook_dir = os.environ["SNAPSHOT_HOOK_PROOF_DIR"]
hook_timeout_s = os.environ["SNAPSHOT_HOOK_TIMEOUT_S"]


def snapshot_env_block(component: str) -> str:
    if not enable_snapshot_hooks:
        return ""
    return f"""      - name: OPTRT_SNAPSHOT_HOOKS
        value: '1'
      - name: OPTRT_SNAPSHOT_HOOK_DIR
        value: {hook_dir}
      - name: OPTRT_SNAPSHOT_HOOK_TIMEOUT_S
        value: '{hook_timeout_s}'
      - name: DYN_COMPONENT
        value: {component}
"""


if enable_snapshot_hooks:
    env_marker = "      - name: PYTHONUNBUFFERED\n        value: '1'\n"
    env_parts = text.split(env_marker)
    if len(env_parts) != 3:
        raise SystemExit(
            f"expected exactly two PYTHONUNBUFFERED env markers, found {len(env_parts) - 1}"
        )
    text = (
        env_parts[0]
        + env_marker
        + snapshot_env_block("prefill")
        + env_parts[1]
        + env_marker
        + snapshot_env_block("decode")
        + env_parts[2]
    )

    init_old = (
        "          - mkdir -p /cache/optrt/hf_modules /cache/optrt/transformers "
        "/cache/optrt/hf_datasets /cache/optrt/xdg /cache/optrt/pip "
        "/cache/optrt/torch_extensions /cache/optrt/torchinductor "
        "/cache/optrt/triton /cache/optrt/cuda /cache/optrt/deep_gemm "
        "/cache/optrt/tensorrt_llm/dg /cache/optrt/tensorrt_llm/llmapi_build "
        "&& chmod -R 0777 /cache/optrt"
    )
    init_new = (
        "          - mkdir -p /cache/optrt/hf_modules /cache/optrt/transformers "
        "/cache/optrt/hf_datasets /cache/optrt/xdg /cache/optrt/pip "
        "/cache/optrt/torch_extensions /cache/optrt/torchinductor "
        "/cache/optrt/triton /cache/optrt/cuda /cache/optrt/deep_gemm "
        "/cache/optrt/tensorrt_llm/dg /cache/optrt/tensorrt_llm/llmapi_build "
        f"{hook_dir} && chmod -R 0777 /cache/optrt {hook_dir}"
    )
    init_count = text.count(init_old)
    if init_count != 2:
        raise SystemExit(f"expected two optrt cache init commands, found {init_count}")
    text = text.replace(init_old, init_new)

    mount_old = "          - mountPath: /cache/optrt\n            name: optrt-cache"
    mount_new = f"""{mount_old}
          - mountPath: {hook_dir}
            name: optrt-snapshot-hooks
"""
    mount_count = text.count(mount_old)
    if mount_count != 4:
        raise SystemExit(f"expected four optrt cache mounts, found {mount_count}")
    text = text.replace(mount_old, mount_new.rstrip("\n"))

    volume_old = (
        "        - name: optrt-cache\n"
        "          hostPath:\n"
        "            path: /var/lib/optrt-cache\n"
        "            type: DirectoryOrCreate"
    )
    volume_new = f"""{volume_old}
        - name: optrt-snapshot-hooks
          hostPath:
            path: {hook_dir}
            type: DirectoryOrCreate
"""
    volume_count = text.count(volume_old)
    if volume_count != 2:
        raise SystemExit(f"expected two optrt cache volumes, found {volume_count}")
    text = text.replace(volume_old, volume_new.rstrip("\n"))
Path(os.environ["OUT"]).write_text(text)
INNER_PY

printf 'rendered_dgd=%s\n' "$OUT"
printf 'dgd_name=%s\n' "$DGD_NAME"
printf 'target_node=%s\n' "$TARGET_NODE"
printf 'image=%s\n' "$IMAGE"
printf 'image_pull_policy=%s\n' "$IMAGE_PULL_POLICY"
printf 'snapshot_hooks_enabled=%s\n' "$ENABLE_SNAPSHOT_HOOKS"
if [[ "$ENABLE_SNAPSHOT_HOOKS" == 1 ]]; then
  printf 'snapshot_hook_proof_dir=%s\n' "$SNAPSHOT_HOOK_PROOF_DIR"
  printf 'snapshot_hook_timeout_s=%s\n' "$SNAPSHOT_HOOK_TIMEOUT_S"
fi

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

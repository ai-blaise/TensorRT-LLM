#!/usr/bin/env bash
set -euo pipefail

VM_HOST="${VM_HOST:-34.106.33.128}"
VM_USER="${VM_USER:-spencergarnets}"
REMOTE_REPO="${REMOTE_REPO:-/tmp/tensorrt-llm-op-trt-fast}"
IMAGE_REPO="${IMAGE_REPO:-docker.io/local/dynamo-trtllm-optrt-custom}"
BASE_IMAGE="${BASE_IMAGE:-local/dynamo-trtllm-optrt-custom:canonical-smc-r20-cpfix-ucx-20260605}"
TARGET_NODE="${TARGET_NODE:-a4-us-001-rl9}"
DGD_NAME="${DGD_NAME:-topo-c1-dp2tp4-disagg-r20}"
DEPLOY=0
SYNC=1
FULL_SYNC=0
BUILD=1
TAG_SUFFIX="${TAG_SUFFIX:-fast}"
SSH_OPTS=(
  -o BatchMode=yes
  -o IdentitiesOnly=yes
  -o UserKnownHostsFile=/Users/spencer/.ssh/google_compute_known_hosts
  -o CheckHostIP=no
)

usage() {
  cat <<'EOF'
Usage: deploy/disagg_pd_r20/fast_iterate.sh [options]

Build a thin op-trt overlay image on the target B200 VM, import it directly into
k3s containerd when needed, and optionally apply the disagg r20 DGD.

Options:
  --vm HOST             Target VM IP or hostname (default: $VM_HOST)
  --user USER           SSH user (default: $VM_USER)
  --remote-repo PATH    Remote rsync/build directory (default: $REMOTE_REPO)
  --image-repo NAME     Image repository (default: $IMAGE_REPO)
  --base-image IMAGE    Existing runtime image used as overlay base
  --target-node NAME    Kubernetes nodeSelector hostname (default: $TARGET_NODE)
  --dgd-name NAME       DGD/ConfigMap name; use a suffix for warm canaries
  --tag-suffix TEXT     Human suffix added after the git sha (default: fast)
  --deploy              Apply the DGD after build/import
  --full-sync           Sync the whole repo instead of the overlay build subset
  --no-sync             Reuse the existing remote repo
  --no-build            Reuse the computed image tag and only apply when --deploy
  -h, --help            Show this help

Environment overrides use the same names as the options.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vm) VM_HOST="$2"; shift 2 ;;
    --user) VM_USER="$2"; shift 2 ;;
    --remote-repo) REMOTE_REPO="$2"; shift 2 ;;
    --image-repo) IMAGE_REPO="$2"; shift 2 ;;
    --base-image) BASE_IMAGE="$2"; shift 2 ;;
    --target-node) TARGET_NODE="$2"; shift 2 ;;
    --dgd-name) DGD_NAME="$2"; shift 2 ;;
    --tag-suffix) TAG_SUFFIX="$2"; shift 2 ;;
    --deploy) DEPLOY=1; shift ;;
    --full-sync) FULL_SYNC=1; shift ;;
    --no-sync) SYNC=0; shift ;;
    --no-build) BUILD=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"
SHA="$(git rev-parse --short=12 HEAD)"
STAMP="$(date -u +%Y%m%d%H%M%S)"
IMAGE_TAG="${IMAGE_TAG:-${IMAGE_REPO}:optrt-${SHA}-${TAG_SUFFIX}-${STAMP}}"
SSH_TARGET="${VM_USER}@${VM_HOST}"

if [[ "$SYNC" == 1 ]]; then
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "mkdir -p '$REMOTE_REPO'"
  if ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "command -v rsync >/dev/null 2>&1"; then
    if [[ "$FULL_SYNC" == 1 ]]; then
      rsync -az --delete \
        --exclude .git \
        --exclude build \
        --exclude cpp/build \
        --exclude dist \
        --exclude '*.egg-info' \
        --exclude __pycache__ \
        --exclude '.pytest_cache' \
        --exclude '.ruff_cache' \
        ./ "$SSH_TARGET:$REMOTE_REPO/"
    else
      rsync -az --delete \
        --relative \
        tensorrt_llm deploy .dockerignore \
        "$SSH_TARGET:$REMOTE_REPO/"
    fi
  elif [[ "$FULL_SYNC" == 1 ]]; then
    tar --exclude ./.git \
      --exclude ./build \
      --exclude ./cpp/build \
      --exclude ./dist \
      --exclude './*.egg-info' \
      --exclude './**/__pycache__' \
      --exclude './.pytest_cache' \
      --exclude './.ruff_cache' \
      -czf - . | ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
        "rm -rf '$REMOTE_REPO' && mkdir -p '$REMOTE_REPO' && tar -xzf - -C '$REMOTE_REPO'"
  else
    ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
      "rm -rf '$REMOTE_REPO/tensorrt_llm' '$REMOTE_REPO/deploy' '$REMOTE_REPO/.dockerignore' && mkdir -p '$REMOTE_REPO'"
    tar -czf - tensorrt_llm deploy .dockerignore | ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
      "tar -xzf - -C '$REMOTE_REPO'"
  fi
fi

read -r -d '' REMOTE_SCRIPT <<'EOS' || true
set -euo pipefail
cd "$REMOTE_REPO"

sudo mkdir -p \
  /var/lib/optrt-cache/hf_modules \
  /var/lib/optrt-cache/xdg \
  /var/lib/optrt-cache/torch_extensions \
  /var/lib/optrt-cache/triton \
  /var/lib/optrt-cache/cuda \
  /var/lib/optrt-cache/tensorrt_llm/dg \
  /var/lib/optrt-cache/tensorrt_llm/llmapi_build
sudo chmod -R 0777 /var/lib/optrt-cache

if [[ "$BUILD" == 1 ]]; then
  if command -v nerdctl >/dev/null 2>&1; then
    sudo nerdctl -n k8s.io build \
      --build-arg "BASE_IMAGE=$BASE_IMAGE" \
      -f deploy/disagg_pd_r20/Dockerfile.r20-overlay \
      -t "$IMAGE_TAG" .
  else
    if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
      tmp_base="/tmp/optrt-base-${BASE_IMAGE##*:}.tar"
      sudo /usr/local/bin/k3s ctr -n k8s.io images export "$tmp_base" "$BASE_IMAGE"
      docker load -i "$tmp_base"
      rm -f "$tmp_base"
    fi
    DOCKER_BUILDKIT=1 docker build \
      --build-arg "BASE_IMAGE=$BASE_IMAGE" \
      -f deploy/disagg_pd_r20/Dockerfile.r20-overlay \
      -t "$IMAGE_TAG" .
    if ! sudo /usr/local/bin/k3s ctr -n k8s.io images ls name=="$IMAGE_TAG" | grep -F "$IMAGE_TAG" >/dev/null 2>&1; then
      docker save "$IMAGE_TAG" | sudo /usr/local/bin/k3s ctr -n k8s.io images import -
    fi
  fi
fi

if [[ "$DEPLOY" == 1 ]]; then
  OUT="/tmp/${DGD_NAME}-${IMAGE_TAG##*:}.yaml"
  TARGET_NODE="$TARGET_NODE" UNIFIED_IMAGE="$IMAGE_TAG" DGD_NAME="$DGD_NAME" python3 - <<'PY'
import os
from pathlib import Path

src = Path("deploy/disagg_pd_r20/topo-c1-dp2tp4-disagg-r20.yaml")
text = src.read_text()
text = text.replace("${TARGET_NODE}", os.environ["TARGET_NODE"])
text = text.replace("${UNIFIED_IMAGE}", os.environ["UNIFIED_IMAGE"])
text = text.replace("topo-c1-dp2tp4-disagg-r20", os.environ["DGD_NAME"])
Path(os.environ["OUT"]).write_text(text)
PY
  sudo -E /usr/local/bin/k3s kubectl -n dynamo-system apply --dry-run=server -f "$OUT"
  sudo -E /usr/local/bin/k3s kubectl -n dynamo-system apply -f "$OUT"
  sudo -E /usr/local/bin/k3s kubectl -n dynamo-system get dgd "$DGD_NAME" \
    -o jsonpath='generation={.metadata.generation} observed={.status.observedGeneration}'; echo
else
  echo "built_image=$IMAGE_TAG"
  echo "deploy_skipped=1"
fi
EOS

ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
  "REMOTE_REPO='$REMOTE_REPO' IMAGE_TAG='$IMAGE_TAG' BASE_IMAGE='$BASE_IMAGE' TARGET_NODE='$TARGET_NODE' DGD_NAME='$DGD_NAME' DEPLOY='$DEPLOY' BUILD='$BUILD' OUT='/tmp/${DGD_NAME}-${IMAGE_TAG##*:}.yaml' bash -s" \
  <<<"$REMOTE_SCRIPT"

echo "image=$IMAGE_TAG"

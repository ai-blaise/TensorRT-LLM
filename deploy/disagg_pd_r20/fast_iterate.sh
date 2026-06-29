#!/usr/bin/env bash
set -euo pipefail

VM_HOST="${VM_HOST:-34.106.33.128}"
VM_USER="${VM_USER:-spencergarnets}"
REMOTE_REPO="${REMOTE_REPO:-/tmp/tensorrt-llm-op-trt-fast}"
IMAGE_REPO="${IMAGE_REPO:-docker.io/local/dynamo-trtllm-optrt-custom}"
BASE_IMAGE="${BASE_IMAGE:-local/dynamo-trtllm-optrt-custom:r20-nixl-layersplit-base}"
TARGET_NODE="${TARGET_NODE:-a4-us-001-rl9}"
DGD_NAME="${DGD_NAME:-topo-c1-dp2tp4-disagg-r20}"
REQUIRED_TRANSPORT_WRAPPERS="${REQUIRED_TRANSPORT_WRAPPERS:-ucx,nixl}"
DEPLOY=0
SYNC=1
FULL_SYNC=0
BUILD=1
PREWARM=0
DRY_RUN=0
USE_LOCAL_REGISTRY=0
LOCAL_REGISTRY="${LOCAL_REGISTRY:-localhost:5000}"
LOCAL_REGISTRY_MODE="${LOCAL_REGISTRY_MODE:-push}"
ALLOW_CHAINED_OVERLAY="${ALLOW_CHAINED_OVERLAY:-0}"
REQUIRE_IMAGE_HANDOFF="${REQUIRE_IMAGE_HANDOFF:-0}"
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
  --vm HOST             Target VM IP or hostname (default: $VM_HOST);
                        use local to run directly from the current VM checkout
  --user USER           SSH user (default: $VM_USER)
  --remote-repo PATH    Remote rsync/build directory (default: $REMOTE_REPO)
  --image-repo NAME     Image repository (default: $IMAGE_REPO)
  --deploy-image IMAGE  Existing image to prewarm/deploy; skips build/import
  --base-image IMAGE    Existing runtime image used as overlay base
  --target-node NAME    Kubernetes nodeSelector hostname (default: $TARGET_NODE)
  --dgd-name NAME       DGD/ConfigMap name; use a suffix for warm canaries
  --tag-suffix TEXT     Human suffix added after the git sha (default: fast)
  --required-transport-wrappers LIST
                        Comma-separated wrapper libs required before deploy
                        (default: ucx,nixl; set empty to skip)
  --deploy              Apply the DGD after build/import
  --prewarm             Run the lightweight cache/model visibility prewarm job
                        after build and before deploy
  --dry-run             Print the resolved sync/build/deploy plan and exit before
                        SSH, sync, build, prewarm, or deploy
  --use-local-registry  Tag the thin image with a VM-local registry name and
                        render pods with imagePullPolicy=IfNotPresent
  --local-registry HOST Registry host:port (default: $LOCAL_REGISTRY)
  --local-registry-mode MODE
                        push: push to registry (default, multi-node safe);
                        resident: keep exact deploy tag in k3s containerd and
                        skip registry push (single-node iteration only)
  --allow-chained-overlay
                        Allow using a previous r20 overlay image as the base
  --require-image-handoff
                        Before transport checks, prewarm, or deploy, fail unless
                        the exact deploy image is resident/registry-ready
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
    --deploy-image) DEPLOY_IMAGE_TAG="$2"; BUILD=0; shift 2 ;;
    --base-image) BASE_IMAGE="$2"; shift 2 ;;
    --target-node) TARGET_NODE="$2"; shift 2 ;;
    --dgd-name) DGD_NAME="$2"; shift 2 ;;
    --tag-suffix) TAG_SUFFIX="$2"; shift 2 ;;
    --required-transport-wrappers) REQUIRED_TRANSPORT_WRAPPERS="$2"; shift 2 ;;
    --deploy) DEPLOY=1; shift ;;
    --prewarm) PREWARM=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --use-local-registry) USE_LOCAL_REGISTRY=1; shift ;;
    --local-registry) LOCAL_REGISTRY="$2"; shift 2 ;;
    --local-registry-mode) LOCAL_REGISTRY_MODE="$2"; shift 2 ;;
    --allow-chained-overlay) ALLOW_CHAINED_OVERLAY=1; shift ;;
    --require-image-handoff) REQUIRE_IMAGE_HANDOFF=1; shift ;;
    --full-sync) FULL_SYNC=1; shift ;;
    --no-sync) SYNC=0; shift ;;
    --no-build) BUILD=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"
if [[ "$VM_HOST" == "local" ]]; then
  REMOTE_REPO="$ROOT_DIR"
  SYNC=0
fi
SHA="$(git rev-parse --short=12 HEAD)"
STAMP="$(date -u +%Y%m%d%H%M%S)"
IMAGE_TAG="${IMAGE_TAG:-${IMAGE_REPO}:optrt-${SHA}-${TAG_SUFFIX}-${STAMP}}"
DEPLOY_IMAGE_TAG="${DEPLOY_IMAGE_TAG:-$IMAGE_TAG}"
if [[ "$USE_LOCAL_REGISTRY" == 1 && "$DEPLOY_IMAGE_TAG" != "$LOCAL_REGISTRY/"* ]]; then
  image_path="${DEPLOY_IMAGE_TAG#docker.io/}"
  image_path="${image_path#${LOCAL_REGISTRY}/}"
  DEPLOY_IMAGE_TAG="${LOCAL_REGISTRY}/${image_path}"
fi
case "$LOCAL_REGISTRY_MODE" in
  push|resident) ;;
  *) echo "unknown --local-registry-mode: $LOCAL_REGISTRY_MODE" >&2; exit 2 ;;
esac
if [[ "$LOCAL_REGISTRY_MODE" == "resident" && "$USE_LOCAL_REGISTRY" != 1 ]]; then
  echo "--local-registry-mode=resident requires --use-local-registry" >&2
  exit 2
fi
source "$(dirname "${BASH_SOURCE[0]}")/target_node_guard.sh"
optrt_r20_reject_disallowed_target_node "$TARGET_NODE"

IMAGE_HANDOFF_MODE=resident
if [[ "$USE_LOCAL_REGISTRY" == 1 && "$LOCAL_REGISTRY_MODE" == "push" ]]; then
  IMAGE_HANDOFF_MODE=registry
fi
if (( ${#DGD_NAME} + 8 > 45 )); then
  echo "DGD name too long for r20 Frontend pod naming: ${#DGD_NAME}+8 > 45 ($DGD_NAME)" >&2
  exit 2
fi
SSH_TARGET="${VM_USER}@${VM_HOST}"

if [[ "$DRY_RUN" == 1 ]]; then
  OVERLAY_SYNC_PATHS=(deploy .dockerignore)
  while IFS= read -r overlay_path; do
    OVERLAY_SYNC_PATHS+=("$overlay_path")
  done < <(
    awk '
      /^COPY[[:space:]]/ {
        for (i = 1; i <= NF; i++) {
          if ($i ~ /^tensorrt_llm\//) {
            print $i
          }
        }
      }
    ' deploy/disagg_pd_r20/Dockerfile.r20-overlay
  )
  image_pull_policy="Never"
  if [[ "$USE_LOCAL_REGISTRY" == 1 ]]; then
    image_pull_policy="IfNotPresent"
  fi
  cat <<EOF
fast_iterate_dry_run=1
root_dir=$ROOT_DIR
vm_host=$VM_HOST
vm_user=$VM_USER
remote_repo=$REMOTE_REPO
sync=$SYNC
full_sync=$FULL_SYNC
build=$BUILD
prewarm=$PREWARM
deploy=$DEPLOY
dgd_name=$DGD_NAME
target_node=$TARGET_NODE
base_image=$BASE_IMAGE
image_tag=$IMAGE_TAG
deploy_image_tag=$DEPLOY_IMAGE_TAG
image_pull_policy=$image_pull_policy
use_local_registry=$USE_LOCAL_REGISTRY
local_registry=$LOCAL_REGISTRY
local_registry_mode=$LOCAL_REGISTRY_MODE
require_image_handoff=$REQUIRE_IMAGE_HANDOFF
image_handoff_mode=$IMAGE_HANDOFF_MODE
allow_chained_overlay=$ALLOW_CHAINED_OVERLAY
required_transport_wrappers=$REQUIRED_TRANSPORT_WRAPPERS
transport_check_image=$DEPLOY_IMAGE_TAG
overlay_sync_path_count=${#OVERLAY_SYNC_PATHS[@]}
EOF
  printf 'overlay_sync_paths='
  printf '%s ' "${OVERLAY_SYNC_PATHS[@]}"
  printf '\n'
  exit 0
fi

if [[ "$SYNC" == 1 ]]; then
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "mkdir -p '$REMOTE_REPO'"
  OVERLAY_SYNC_PATHS=(deploy .dockerignore)
  while IFS= read -r overlay_path; do
    OVERLAY_SYNC_PATHS+=("$overlay_path")
  done < <(
    awk '
      /^COPY[[:space:]]/ {
        for (i = 1; i <= NF; i++) {
          if ($i ~ /^tensorrt_llm\//) {
            print $i
          }
        }
      }
    ' deploy/disagg_pd_r20/Dockerfile.r20-overlay
  )
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
      ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
        "rm -rf '$REMOTE_REPO/tensorrt_llm' '$REMOTE_REPO/deploy' '$REMOTE_REPO/.dockerignore' && mkdir -p '$REMOTE_REPO'"
      rsync -az --delete \
        --relative \
        "${OVERLAY_SYNC_PATHS[@]}" \
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
    tar -czf - "${OVERLAY_SYNC_PATHS[@]}" | ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
      "tar -xzf - -C '$REMOTE_REPO'"
  fi
fi

read -r -d '' REMOTE_SCRIPT <<'EOS' || true
set -euo pipefail
cd "$REMOTE_REPO"
BUILD_IMAGE_TAG="$IMAGE_TAG"
OVERLAY_DOCKERIGNORE="deploy/disagg_pd_r20/Dockerfile.r20-overlay.dockerignore"
if [[ -f "$OVERLAY_DOCKERIGNORE" ]]; then
  if [[ -d .git ]]; then
    echo "overlay_dockerignore=using_dockerfile_specific_ignore"
  else
    cp "$OVERLAY_DOCKERIGNORE" .dockerignore
    echo "overlay_dockerignore=installed_build_root_ignore"
  fi
fi

if [[ "$PREWARM" == 1 || "$DEPLOY" == 1 ]]; then
  sudo mkdir -p \
    /var/lib/optrt-cache/hf_modules \
    /var/lib/optrt-cache/transformers \
    /var/lib/optrt-cache/hf_datasets \
    /var/lib/optrt-cache/xdg \
    /var/lib/optrt-cache/pip \
    /var/lib/optrt-cache/torch_extensions \
    /var/lib/optrt-cache/torchinductor \
    /var/lib/optrt-cache/triton \
    /var/lib/optrt-cache/cuda \
    /var/lib/optrt-cache/deep_gemm \
    /var/lib/optrt-cache/tensorrt_llm/dg \
    /var/lib/optrt-cache/tensorrt_llm/llmapi_build
  sudo chmod -R 0777 /var/lib/optrt-cache
fi

if [[ "$BUILD" == 1 ]]; then
  if [[ "$ALLOW_CHAINED_OVERLAY" != 1 ]]; then
    base_overlay_label=""
    if command -v nerdctl >/dev/null 2>&1; then
      base_overlay_label="$(sudo nerdctl -n k8s.io image inspect "$BASE_IMAGE" \
        --format '{{ index .Config.Labels "ai.blaise.r20-overlay" }}' 2>/dev/null || true)"
    elif command -v docker >/dev/null 2>&1 && docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
      base_overlay_label="$(docker image inspect "$BASE_IMAGE" \
        --format '{{ index .Config.Labels "ai.blaise.r20-overlay" }}' 2>/dev/null || true)"
    fi
    if [[ "$base_overlay_label" == "true" ]]; then
      echo "refusing chained r20 overlay base: $BASE_IMAGE" >&2
      echo "Use a stable runtime/canonical base, or pass --allow-chained-overlay after checking containerd mount limits." >&2
      exit 2
    fi
  fi
  if command -v nerdctl >/dev/null 2>&1; then
    sudo nerdctl -n k8s.io build \
      --build-arg "BASE_IMAGE=$BASE_IMAGE" \
      -f deploy/disagg_pd_r20/Dockerfile.r20-overlay \
      -t "$BUILD_IMAGE_TAG" .
    if [[ "$USE_LOCAL_REGISTRY" == 1 && "$DEPLOY_IMAGE_TAG" != "$BUILD_IMAGE_TAG" ]]; then
      sudo nerdctl -n k8s.io tag "$BUILD_IMAGE_TAG" "$DEPLOY_IMAGE_TAG"
      if [[ "$LOCAL_REGISTRY_MODE" == "push" ]]; then
        if command -v docker >/dev/null 2>&1; then
          if ! docker ps --format '{{.Names}}' | grep -qx optrt-registry; then
            docker rm -f optrt-registry >/dev/null 2>&1 || true
            docker run -d --restart=always -p "${LOCAL_REGISTRY##*:}:5000" --name optrt-registry registry:2 >/dev/null
          fi
        fi
        sudo nerdctl -n k8s.io push "$DEPLOY_IMAGE_TAG"
      else
        echo "resident_local_image=$DEPLOY_IMAGE_TAG"
        echo "registry_push_skipped=1"
      fi
    fi
  else
    if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
      tmp_base="/tmp/optrt-base-${BASE_IMAGE##*:}.tar"
      sudo /usr/local/bin/k3s ctr -n k8s.io images export "$tmp_base" "$BASE_IMAGE"
      sudo chmod 0644 "$tmp_base"
      docker load -i "$tmp_base"
      rm -f "$tmp_base" || sudo rm -f "$tmp_base"
    fi
    DOCKER_BUILDKIT=1 docker build \
      --build-arg "BASE_IMAGE=$BASE_IMAGE" \
      -f deploy/disagg_pd_r20/Dockerfile.r20-overlay \
      -t "$BUILD_IMAGE_TAG" .
    if [[ "$USE_LOCAL_REGISTRY" == 1 ]]; then
      docker tag "$BUILD_IMAGE_TAG" "$DEPLOY_IMAGE_TAG"
      if [[ "$LOCAL_REGISTRY_MODE" == "push" ]]; then
        if ! docker ps --format '{{.Names}}' | grep -qx optrt-registry; then
          docker rm -f optrt-registry >/dev/null 2>&1 || true
          docker run -d --restart=always -p "${LOCAL_REGISTRY##*:}:5000" --name optrt-registry registry:2 >/dev/null
        fi
        docker push "$DEPLOY_IMAGE_TAG"
      else
        docker save "$DEPLOY_IMAGE_TAG" | sudo /usr/local/bin/k3s ctr -n k8s.io images import -
        echo "resident_local_image=$DEPLOY_IMAGE_TAG"
        echo "registry_push_skipped=1"
      fi
    elif ! sudo /usr/local/bin/k3s ctr -n k8s.io images ls name=="$BUILD_IMAGE_TAG" | grep -F "$BUILD_IMAGE_TAG" >/dev/null 2>&1; then
      docker save "$BUILD_IMAGE_TAG" | sudo /usr/local/bin/k3s ctr -n k8s.io images import -
    fi
  fi
fi

if [[ "$REQUIRE_IMAGE_HANDOFF" == 1 ]]; then
  echo "image_handoff_required=1"
  deploy/disagg_pd_r20/check_image_handoff.sh \
    --vm local \
    --image "$DEPLOY_IMAGE_TAG" \
    --mode "$IMAGE_HANDOFF_MODE" \
    --local-registry "$LOCAL_REGISTRY" \
    --require
fi

if [[ -n "$REQUIRED_TRANSPORT_WRAPPERS" ]]; then
  check_image="$DEPLOY_IMAGE_TAG"
  check_script='
set -euo pipefail
base="/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/libs"
IFS="," read -ra wrappers <<<"$REQUIRED_TRANSPORT_WRAPPERS"
missing=0
for wrapper in "${wrappers[@]}"; do
  wrapper="${wrapper//[[:space:]]/}"
  [[ -z "$wrapper" ]] && continue
  path="$base/libtensorrt_llm_${wrapper}_wrapper.so"
  if [[ -s "$path" ]]; then
    echo "transport_wrapper_ok=$wrapper:$path"
  else
    echo "transport_wrapper_missing=$wrapper:$path" >&2
    missing=1
  fi
done
exit "$missing"
'
  if command -v nerdctl >/dev/null 2>&1; then
    sudo nerdctl -n k8s.io run --rm --user root --entrypoint /bin/bash \
      -e REQUIRED_TRANSPORT_WRAPPERS="$REQUIRED_TRANSPORT_WRAPPERS" \
      "$check_image" -lc "$check_script"
  elif command -v docker >/dev/null 2>&1 && docker image inspect "$check_image" >/dev/null 2>&1; then
    docker run --rm --user root --entrypoint /bin/bash \
      -e REQUIRED_TRANSPORT_WRAPPERS="$REQUIRED_TRANSPORT_WRAPPERS" \
      "$check_image" -lc "$check_script"
  else
    echo "unable to preflight transport wrappers for $check_image: no runnable local image engine found" >&2
    exit 2
  fi
fi

if [[ "$PREWARM" == 1 ]]; then
  pull_policy=Never
  if [[ "$USE_LOCAL_REGISTRY" == 1 ]]; then
    pull_policy=IfNotPresent
  fi
  JOB_NAME="optrt-cache-prewarm-$(date -u +%Y%m%d%H%M%S)"
  cat >/tmp/"$JOB_NAME".yaml <<YAML
apiVersion: batch/v1
kind: Job
metadata:
  name: $JOB_NAME
  namespace: dynamo-system
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 900
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        kubernetes.io/hostname: $TARGET_NODE
      tolerations:
      - operator: Exists
      containers:
      - name: prewarm
        image: $DEPLOY_IMAGE_TAG
        imagePullPolicy: $pull_policy
        command: [python3, -c]
        args:
        - |
          import importlib
          import os
          from pathlib import Path
          os.environ.setdefault("HF_HOME", "/models")
          os.environ.setdefault("HF_HUB_OFFLINE", "1")
          os.environ.setdefault("HF_MODULES_CACHE", "/cache/optrt/hf_modules")
          os.environ.setdefault("TRANSFORMERS_CACHE", "/cache/optrt/transformers")
          os.environ.setdefault("HF_DATASETS_CACHE", "/cache/optrt/hf_datasets")
          os.environ.setdefault("XDG_CACHE_HOME", "/cache/optrt/xdg")
          os.environ.setdefault("PIP_CACHE_DIR", "/cache/optrt/pip")
          os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/cache/optrt/torch_extensions")
          os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/cache/optrt/torchinductor")
          os.environ.setdefault("TRITON_CACHE_DIR", "/cache/optrt/triton")
          os.environ.setdefault("CUDA_CACHE_PATH", "/cache/optrt/cuda")
          os.environ.setdefault("DG_JIT_CACHE_DIR", "/cache/optrt/deep_gemm")
          os.environ.setdefault("TRTLLM_DG_CACHE_DIR", "/cache/optrt/tensorrt_llm/dg")
          os.environ.setdefault("TLLM_LLMAPI_BUILD_CACHE", "1")
          os.environ.setdefault("TLLM_LLMAPI_BUILD_CACHE_ROOT", "/cache/optrt/tensorrt_llm/llmapi_build")
          for path in [
              "/cache/optrt/hf_modules", "/cache/optrt/transformers",
              "/cache/optrt/hf_datasets", "/cache/optrt/xdg", "/cache/optrt/pip",
              "/cache/optrt/torch_extensions", "/cache/optrt/torchinductor",
              "/cache/optrt/triton", "/cache/optrt/cuda",
              "/cache/optrt/deep_gemm", "/cache/optrt/tensorrt_llm/dg",
              "/cache/optrt/tensorrt_llm/llmapi_build",
          ]:
              Path(path).mkdir(parents=True, exist_ok=True)
          def _exception_chain_contains(exc, needle):
              seen = set()
              stack = [exc]
              while stack:
                  cur = stack.pop()
                  if cur is None or id(cur) in seen:
                      continue
                  seen.add(id(cur))
                  if needle in str(cur):
                      return True
                  stack.extend([getattr(cur, "__cause__", None), getattr(cur, "__context__", None)])
              return False
          for module in ["torch", "transformers", "tensorrt_llm"]:
              try:
                  importlib.import_module(module)
              except ImportError as exc:
                  if module == "tensorrt_llm" and _exception_chain_contains(exc, "libcuda.so.1"):
                      print(f"prewarm_import_warning={module}: {exc}")
                      continue
                  raise
          from transformers import AutoConfig, AutoTokenizer
          for model in [
              "/models/BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft",
          ]:
              print(f"prewarm_model={model}")
              AutoConfig.from_pretrained(model, trust_remote_code=True, local_files_only=True)
              try:
                  AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
              except Exception as exc:
                  print(f"tokenizer_prewarm_warning={model}: {exc}")
          print("optrt_cache_prewarm=ok")
        volumeMounts:
        - mountPath: /models
          name: models
          readOnly: true
        - mountPath: /cache/optrt
          name: optrt-cache
      volumes:
      - name: models
        hostPath:
          path: /models
          type: Directory
      - name: optrt-cache
        hostPath:
          path: /var/lib/optrt-cache
          type: DirectoryOrCreate
YAML
  sudo -E /usr/local/bin/k3s kubectl -n dynamo-system apply -f /tmp/"$JOB_NAME".yaml
  sudo -E /usr/local/bin/k3s kubectl -n dynamo-system wait --for=condition=complete --timeout=300s job/"$JOB_NAME"
  sudo -E /usr/local/bin/k3s kubectl -n dynamo-system logs job/"$JOB_NAME"
  sudo -E /usr/local/bin/k3s kubectl -n dynamo-system delete job "$JOB_NAME" --ignore-not-found=true >/dev/null
fi

if [[ "$DEPLOY" == 1 ]]; then
  OUT="/tmp/${DGD_NAME}-${DEPLOY_IMAGE_TAG##*:}.yaml"
  TARGET_NODE="$TARGET_NODE" UNIFIED_IMAGE="$DEPLOY_IMAGE_TAG" DGD_NAME="$DGD_NAME" USE_LOCAL_REGISTRY="$USE_LOCAL_REGISTRY" OUT="$OUT" python3 - <<'PY'
import os
from pathlib import Path

src = Path("deploy/disagg_pd_r20/topo-c1-dp2tp4-disagg-r20.yaml")
text = src.read_text()
text = text.replace("${TARGET_NODE}", os.environ["TARGET_NODE"])
text = text.replace("${UNIFIED_IMAGE}", os.environ["UNIFIED_IMAGE"])
text = text.replace("topo-c1-dp2tp4-disagg-r20", os.environ["DGD_NAME"])
if os.environ.get("USE_LOCAL_REGISTRY") == "1":
    text = text.replace("imagePullPolicy: Never", "imagePullPolicy: IfNotPresent")
Path(os.environ["OUT"]).write_text(text)
PY
  sudo -E /usr/local/bin/k3s kubectl -n dynamo-system apply --dry-run=server -f "$OUT"
  sudo -E /usr/local/bin/k3s kubectl -n dynamo-system apply -f "$OUT"
  sudo -E /usr/local/bin/k3s kubectl -n dynamo-system get dgd "$DGD_NAME" \
    -o jsonpath='generation={.metadata.generation} observed={.status.observedGeneration}'; echo
else
  echo "built_image=$BUILD_IMAGE_TAG"
  echo "deploy_image=$DEPLOY_IMAGE_TAG"
  echo "deploy_skipped=1"
fi
EOS

if [[ "$VM_HOST" == "local" ]]; then
  REMOTE_REPO="$REMOTE_REPO" IMAGE_TAG="$IMAGE_TAG" DEPLOY_IMAGE_TAG="$DEPLOY_IMAGE_TAG" \
    BASE_IMAGE="$BASE_IMAGE" TARGET_NODE="$TARGET_NODE" DGD_NAME="$DGD_NAME" \
    REQUIRED_TRANSPORT_WRAPPERS="$REQUIRED_TRANSPORT_WRAPPERS" DEPLOY="$DEPLOY" \
    BUILD="$BUILD" PREWARM="$PREWARM" USE_LOCAL_REGISTRY="$USE_LOCAL_REGISTRY" \
    LOCAL_REGISTRY="$LOCAL_REGISTRY" LOCAL_REGISTRY_MODE="$LOCAL_REGISTRY_MODE" \
    ALLOW_CHAINED_OVERLAY="$ALLOW_CHAINED_OVERLAY" REQUIRE_IMAGE_HANDOFF="$REQUIRE_IMAGE_HANDOFF" \
    IMAGE_HANDOFF_MODE="$IMAGE_HANDOFF_MODE" \
    OUT="/tmp/${DGD_NAME}-${DEPLOY_IMAGE_TAG##*:}.yaml" bash -s <<<"$REMOTE_SCRIPT"
else
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
    "REMOTE_REPO='$REMOTE_REPO' IMAGE_TAG='$IMAGE_TAG' DEPLOY_IMAGE_TAG='$DEPLOY_IMAGE_TAG' BASE_IMAGE='$BASE_IMAGE' TARGET_NODE='$TARGET_NODE' DGD_NAME='$DGD_NAME' REQUIRED_TRANSPORT_WRAPPERS='$REQUIRED_TRANSPORT_WRAPPERS' DEPLOY='$DEPLOY' BUILD='$BUILD' PREWARM='$PREWARM' USE_LOCAL_REGISTRY='$USE_LOCAL_REGISTRY' LOCAL_REGISTRY='$LOCAL_REGISTRY' LOCAL_REGISTRY_MODE='$LOCAL_REGISTRY_MODE' ALLOW_CHAINED_OVERLAY='$ALLOW_CHAINED_OVERLAY' REQUIRE_IMAGE_HANDOFF='$REQUIRE_IMAGE_HANDOFF' IMAGE_HANDOFF_MODE='$IMAGE_HANDOFF_MODE' OUT='/tmp/${DGD_NAME}-${DEPLOY_IMAGE_TAG##*:}.yaml' bash -s" \
    <<<"$REMOTE_SCRIPT"
fi

echo "built_image=$IMAGE_TAG"
echo "deploy_image=$DEPLOY_IMAGE_TAG"

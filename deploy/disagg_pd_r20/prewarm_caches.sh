#!/usr/bin/env bash
set -euo pipefail

VM_HOST="${VM_HOST:-34.106.33.128}"
VM_USER="${VM_USER:-spencergarnets}"
TARGET_NODE="${TARGET_NODE:-a4-us-001-rl9}"
IMAGE="${IMAGE:-}"
IMAGE_PULL_POLICY="${IMAGE_PULL_POLICY:-Never}"
MODEL_PATH="${MODEL_PATH:-/models/BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft}"
DRY_RUN=0
SERVER_DRY_RUN=0
REQUIRE_IMAGE_HANDOFF=0
LOCAL_REGISTRY="${LOCAL_REGISTRY:-localhost:5000}"
SSH_OPTS=(
  -o BatchMode=yes
  -o IdentitiesOnly=yes
  -o UserKnownHostsFile=/Users/spencer/.ssh/google_compute_known_hosts
  -o CheckHostIP=no
)

usage() {
  cat <<'EOF'
Usage: deploy/disagg_pd_r20/prewarm_caches.sh --image IMAGE [options]

Prepare the persistent op-trt cache hostPath and run a lightweight cache prewarm
job on the target B200 node. This warms Python/HF remote-code/module caches and
validates that the mounted model artifacts are visible before a full DGD rollout.

Options:
  --image IMAGE          Image already present in k3s containerd
  --image-pull-policy P  Kubernetes image pull policy (default: Never;
                         use IfNotPresent for VM-local registry images)
  --vm HOST              Target VM IP or hostname (default: $VM_HOST);
                         use local to run directly from the current VM
  --user USER            SSH user (default: $VM_USER)
  --target-node NAME     Kubernetes nodeSelector hostname (default: $TARGET_NODE)
  --model PATH           Main model path (default: production DeepSeek path)
  --dry-run              Render the prewarm Job YAML and exit without applying it
  --server-dry-run       Validate the Job with kubectl apply --dry-run=server
                         without creating a pod or touching cache dirs
  --require-image-handoff
                         Fail before prewarm if IMAGE is not resident/registry-ready
  --local-registry HOST  Registry host:port for image handoff checks
                         (default: $LOCAL_REGISTRY)
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --image-pull-policy) IMAGE_PULL_POLICY="$2"; shift 2 ;;
    --vm) VM_HOST="$2"; shift 2 ;;
    --user) VM_USER="$2"; shift 2 ;;
    --target-node) TARGET_NODE="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --server-dry-run) SERVER_DRY_RUN=1; shift ;;
    --require-image-handoff) REQUIRE_IMAGE_HANDOFF=1; shift ;;
    --local-registry) LOCAL_REGISTRY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$IMAGE" ]]; then
  echo "--image is required" >&2
  exit 2
fi

source "$(dirname "${BASH_SOURCE[0]}")/target_node_guard.sh"
optrt_r20_reject_disallowed_target_node "$TARGET_NODE"

SSH_TARGET="${VM_USER}@${VM_HOST}"
JOB_NAME="optrt-cache-prewarm-$(date -u +%Y%m%d%H%M%S)"

read -r -d '' REMOTE_SCRIPT <<'EOS' || true
set -euo pipefail

if [[ "$REQUIRE_IMAGE_HANDOFF" == 1 ]]; then
  containerd_detail=""
  if command -v nerdctl >/dev/null 2>&1; then
    containerd_detail="$(
      sudo nerdctl -n k8s.io images --format '{{.Repository}}:{{.Tag}}	{{.Size}}	{{.Digest}}' 2>/dev/null \
        | awk -F '\t' -v img="$IMAGE" '$1 == img {print; found=1} END {exit found ? 0 : 1}' \
        || true
    )"
  else
    containerd_detail="$(
      sudo /usr/local/bin/k3s ctr -n k8s.io images ls 2>/dev/null \
        | awk -v img="$IMAGE" '$1 == img {print; found=1} END {exit found ? 0 : 1}' \
        || true
    )"
  fi

  containerd_resident=no
  if [[ -n "$containerd_detail" ]]; then
    containerd_resident=yes
  fi

  registry_available=unknown
  registry_tag_available=not_applicable
  registry_ref=none
  if command -v curl >/dev/null 2>&1; then
    if curl -fsS "http://${LOCAL_REGISTRY}/v2/" >/dev/null 2>&1; then
      registry_available=yes
    else
      registry_available=no
    fi
  fi
  if [[ "$IMAGE" == "$LOCAL_REGISTRY/"* ]]; then
    registry_ref="${IMAGE#${LOCAL_REGISTRY}/}"
    if [[ "$registry_ref" == *@sha256:* ]]; then
      registry_repo="${registry_ref%@sha256:*}"
      registry_ref_name="sha256:${registry_ref##*@sha256:}"
    else
      registry_repo="${registry_ref%:*}"
      registry_ref_name="${registry_ref##*:}"
    fi
    if [[ "$registry_available" == yes ]]; then
      if curl -fsSI \
        -H 'Accept: application/vnd.oci.image.index.v1+json' \
        -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
        "http://${LOCAL_REGISTRY}/v2/${registry_repo}/manifests/${registry_ref_name}" >/dev/null 2>&1; then
        registry_tag_available=yes
      else
        registry_tag_available=no
      fi
    elif [[ "$registry_available" == no ]]; then
      registry_tag_available=no
    fi
  fi

  handoff_ready=no
  reason="image_not_resident_or_registry_available"
  if [[ "$IMAGE_PULL_POLICY" == Never ]]; then
    if [[ "$containerd_resident" == yes ]]; then
      handoff_ready=yes
      reason=""
    else
      reason="image_not_resident_in_containerd"
    fi
  elif [[ "$registry_tag_available" == yes || "$containerd_resident" == yes ]]; then
    handoff_ready=yes
    reason=""
  fi

  printf 'image_handoff_check=1\n'
  printf 'image=%s\n' "$IMAGE"
  printf 'image_pull_policy=%s\n' "$IMAGE_PULL_POLICY"
  printf 'containerd_resident=%s\n' "$containerd_resident"
  printf 'registry_available=%s\n' "$registry_available"
  printf 'registry_ref=%s\n' "$registry_ref"
  printf 'registry_tag_available=%s\n' "$registry_tag_available"
  printf 'handoff_ready=%s\n' "$handoff_ready"
  if [[ -n "$reason" ]]; then
    printf 'reason=%s\n' "$reason"
    exit 3
  fi
fi


if [[ "$DRY_RUN" != 1 && "$SERVER_DRY_RUN" != 1 ]]; then
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
        image: $IMAGE
        imagePullPolicy: $IMAGE_PULL_POLICY
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
          for path in ["/cache/optrt/hf_modules", "/cache/optrt/transformers", "/cache/optrt/hf_datasets", "/cache/optrt/xdg", "/cache/optrt/pip", "/cache/optrt/torch_extensions", "/cache/optrt/torchinductor", "/cache/optrt/triton", "/cache/optrt/cuda", "/cache/optrt/deep_gemm", "/cache/optrt/tensorrt_llm/dg", "/cache/optrt/tensorrt_llm/llmapi_build"]:
              Path(path).mkdir(parents=True, exist_ok=True)
          for module in ["torch", "transformers", "tensorrt_llm"]:
              try:
                  importlib.import_module(module)
              except ImportError as exc:
                  if module == "tensorrt_llm" and "libcuda.so.1" in str(exc):
                      print(f"prewarm_import_warning={module}: {exc}")
                      continue
                  raise
          from transformers import AutoConfig, AutoTokenizer
          for model in ["$MODEL_PATH"]:
              print(f"prewarm_model={model}")
              AutoConfig.from_pretrained(model, trust_remote_code=True, local_files_only=True)
              try:
                  AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
              except Exception as exc:
                  print(f"tokenizer_prewarm_warning={model}: {exc}")
          print("optrt_cache_prewarm=ok")
        env:
        - name: HF_HOME
          value: /models
        - name: HF_HUB_OFFLINE
          value: "1"
        - name: HF_MODULES_CACHE
          value: /cache/optrt/hf_modules
        - name: TRANSFORMERS_CACHE
          value: /cache/optrt/transformers
        - name: HF_DATASETS_CACHE
          value: /cache/optrt/hf_datasets
        - name: XDG_CACHE_HOME
          value: /cache/optrt/xdg
        - name: PIP_CACHE_DIR
          value: /cache/optrt/pip
        - name: TORCH_EXTENSIONS_DIR
          value: /cache/optrt/torch_extensions
        - name: TORCHINDUCTOR_CACHE_DIR
          value: /cache/optrt/torchinductor
        - name: TRITON_CACHE_DIR
          value: /cache/optrt/triton
        - name: CUDA_CACHE_PATH
          value: /cache/optrt/cuda
        - name: DG_JIT_CACHE_DIR
          value: /cache/optrt/deep_gemm
        - name: TRTLLM_DG_CACHE_DIR
          value: /cache/optrt/tensorrt_llm/dg
        - name: TLLM_LLMAPI_BUILD_CACHE
          value: "1"
        - name: TLLM_LLMAPI_BUILD_CACHE_ROOT
          value: /cache/optrt/tensorrt_llm/llmapi_build
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

if [[ "$DRY_RUN" == 1 ]]; then
  cat /tmp/"$JOB_NAME".yaml
  rm -f /tmp/"$JOB_NAME".yaml
  exit 0
fi

if [[ "$SERVER_DRY_RUN" == 1 ]]; then
  sudo -E /usr/local/bin/k3s kubectl -n dynamo-system apply --dry-run=server -f /tmp/"$JOB_NAME".yaml
  rm -f /tmp/"$JOB_NAME".yaml
  exit 0
fi

sudo -E /usr/local/bin/k3s kubectl -n dynamo-system apply -f /tmp/"$JOB_NAME".yaml
sudo -E /usr/local/bin/k3s kubectl -n dynamo-system wait --for=condition=complete --timeout=300s job/"$JOB_NAME"
sudo -E /usr/local/bin/k3s kubectl -n dynamo-system logs job/"$JOB_NAME"
sudo -E /usr/local/bin/k3s kubectl -n dynamo-system delete job "$JOB_NAME" --ignore-not-found=true >/dev/null
EOS

if [[ "$VM_HOST" == "local" ]]; then
  IMAGE="$IMAGE" IMAGE_PULL_POLICY="$IMAGE_PULL_POLICY" TARGET_NODE="$TARGET_NODE" \
    MODEL_PATH="$MODEL_PATH" JOB_NAME="$JOB_NAME" DRY_RUN="$DRY_RUN" SERVER_DRY_RUN="$SERVER_DRY_RUN" \
    REQUIRE_IMAGE_HANDOFF="$REQUIRE_IMAGE_HANDOFF" LOCAL_REGISTRY="$LOCAL_REGISTRY" bash -s <<<"$REMOTE_SCRIPT"
else
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
    "IMAGE='$IMAGE' IMAGE_PULL_POLICY='$IMAGE_PULL_POLICY' TARGET_NODE='$TARGET_NODE' MODEL_PATH='$MODEL_PATH' JOB_NAME='$JOB_NAME' DRY_RUN='$DRY_RUN' SERVER_DRY_RUN='$SERVER_DRY_RUN' REQUIRE_IMAGE_HANDOFF='$REQUIRE_IMAGE_HANDOFF' LOCAL_REGISTRY='$LOCAL_REGISTRY' bash -s" \
    <<<"$REMOTE_SCRIPT"
fi

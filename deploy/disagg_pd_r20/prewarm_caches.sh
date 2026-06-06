#!/usr/bin/env bash
set -euo pipefail

VM_HOST="${VM_HOST:-34.106.33.128}"
VM_USER="${VM_USER:-spencergarnets}"
TARGET_NODE="${TARGET_NODE:-a4-us-001-rl9}"
IMAGE="${IMAGE:-}"
IMAGE_PULL_POLICY="${IMAGE_PULL_POLICY:-Never}"
MODEL_PATH="${MODEL_PATH:-/models/BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/models/BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP}"
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
  --vm HOST              Target VM IP or hostname (default: $VM_HOST)
  --user USER            SSH user (default: $VM_USER)
  --target-node NAME     Kubernetes nodeSelector hostname (default: $TARGET_NODE)
  --model PATH           Main model path (default: production DeepSeek path)
  --draft-model PATH     SMC draft model path (default: production GLM path)
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
    --draft-model) DRAFT_MODEL_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$IMAGE" ]]; then
  echo "--image is required" >&2
  exit 2
fi

SSH_TARGET="${VM_USER}@${VM_HOST}"
JOB_NAME="optrt-cache-prewarm-$(date -u +%Y%m%d%H%M%S)"

read -r -d '' REMOTE_SCRIPT <<'EOS' || true
set -euo pipefail

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
          for model in ["$MODEL_PATH", "$DRAFT_MODEL_PATH"]:
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

sudo -E /usr/local/bin/k3s kubectl -n dynamo-system apply -f /tmp/"$JOB_NAME".yaml
sudo -E /usr/local/bin/k3s kubectl -n dynamo-system wait --for=condition=complete --timeout=300s job/"$JOB_NAME"
sudo -E /usr/local/bin/k3s kubectl -n dynamo-system logs job/"$JOB_NAME"
sudo -E /usr/local/bin/k3s kubectl -n dynamo-system delete job "$JOB_NAME" --ignore-not-found=true >/dev/null
EOS

ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
  "IMAGE='$IMAGE' IMAGE_PULL_POLICY='$IMAGE_PULL_POLICY' TARGET_NODE='$TARGET_NODE' MODEL_PATH='$MODEL_PATH' DRAFT_MODEL_PATH='$DRAFT_MODEL_PATH' JOB_NAME='$JOB_NAME' bash -s" \
  <<<"$REMOTE_SCRIPT"

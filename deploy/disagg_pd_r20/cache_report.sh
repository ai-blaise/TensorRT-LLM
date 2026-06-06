#!/usr/bin/env bash
set -euo pipefail

VM_HOST="${VM_HOST:-34.106.33.128}"
VM_USER="${VM_USER:-spencergarnets}"
LOCAL_REGISTRY="${LOCAL_REGISTRY:-localhost:5000}"
IMAGE_FILTER="${IMAGE_FILTER:-dynamo-trtllm-optrt-custom}"
SSH_OPTS=(
  -o BatchMode=yes
  -o IdentitiesOnly=yes
  -o UserKnownHostsFile=/Users/spencer/.ssh/google_compute_known_hosts
  -o CheckHostIP=no
)

usage() {
  cat <<'EOF'
Usage: deploy/disagg_pd_r20/cache_report.sh [options]

Print a read-only cache/image residency report from the target B200 VM. This is
safe to run while a DGD is warming; it does not mutate pods, images, or caches.

Options:
  --vm HOST              Target VM IP or hostname (default: $VM_HOST)
  --user USER            SSH user (default: $VM_USER)
  --local-registry HOST  Registry host:port to probe (default: $LOCAL_REGISTRY)
  --image-filter TEXT    Image substring for k3s/containerd listing
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vm) VM_HOST="$2"; shift 2 ;;
    --user) VM_USER="$2"; shift 2 ;;
    --local-registry) LOCAL_REGISTRY="$2"; shift 2 ;;
    --image-filter) IMAGE_FILTER="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

SSH_TARGET="${VM_USER}@${VM_HOST}"

read -r -d '' REMOTE_SCRIPT <<'EOS' || true
set -euo pipefail

echo "== op-trt persistent cache =="
if [[ -d /var/lib/optrt-cache ]]; then
  sudo du -sh /var/lib/optrt-cache 2>/dev/null || true
  for path in \
    hf_modules transformers hf_datasets xdg pip torch_extensions \
    torchinductor triton cuda deep_gemm tensorrt_llm/dg \
    tensorrt_llm/llmapi_build; do
    full="/var/lib/optrt-cache/$path"
    if [[ -d "$full" ]]; then
      size="$(sudo du -sh "$full" 2>/dev/null | awk '{print $1}')"
      files="$(sudo find "$full" -type f 2>/dev/null | wc -l | tr -d ' ')"
      newest="$(
        { sudo find "$full" -type f -printf '%T@ %p\n' 2>/dev/null || true; } \
          | sort -nr \
          | awk 'NR == 1 {sub(/^[^ ]+ /, ""); print; exit}'
      )"
      printf '%-32s size=%-8s files=%-8s newest=%s\n' "$path" "${size:-0}" "${files:-0}" "${newest:-none}"
    else
      printf '%-32s missing\n' "$path"
    fi
  done
else
  echo "missing /var/lib/optrt-cache"
fi

echo
echo "== local registry =="
if command -v docker >/dev/null 2>&1; then
  docker ps --filter name=optrt-registry --format 'registry={{.Names}} status={{.Status}} ports={{.Ports}}' || true
fi
if command -v curl >/dev/null 2>&1; then
  curl -fsS "http://${LOCAL_REGISTRY}/v2/_catalog" 2>/dev/null || echo "registry_catalog_unavailable"
  echo
fi

echo
echo "== k3s/containerd image residency =="
if command -v nerdctl >/dev/null 2>&1; then
  sudo nerdctl -n k8s.io images --format '{{.Repository}}:{{.Tag}}\t{{.Size}}' 2>/dev/null | grep -F "$IMAGE_FILTER" | tail -20 || true
else
  sudo /usr/local/bin/k3s ctr -n k8s.io images ls 2>/dev/null | grep -F "$IMAGE_FILTER" | tail -20 || true
fi
EOS

ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
  "LOCAL_REGISTRY='$LOCAL_REGISTRY' IMAGE_FILTER='$IMAGE_FILTER' bash -s" \
  <<<"$REMOTE_SCRIPT"

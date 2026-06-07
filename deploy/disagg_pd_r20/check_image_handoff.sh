#!/usr/bin/env bash
set -euo pipefail

VM_HOST="${VM_HOST:-34.106.33.128}"
VM_USER="${VM_USER:-spencergarnets}"
LOCAL_REGISTRY="${LOCAL_REGISTRY:-localhost:5000}"
IMAGE="${IMAGE:-}"
IMAGE_FROM_DGD="${IMAGE_FROM_DGD:-}"
MODE="${MODE:-auto}"
REQUIRE=0
NAMESPACE="${NAMESPACE:-dynamo-system}"
SSH_OPTS=(
  -o BatchMode=yes
  -o IdentitiesOnly=yes
  -o UserKnownHostsFile=/Users/spencer/.ssh/google_compute_known_hosts
  -o CheckHostIP=no
)

usage() {
  cat <<'EOF'
Usage: deploy/disagg_pd_r20/check_image_handoff.sh [options]

Read-only preflight for r20 image handoff before prewarm/deploy. It verifies
whether an exact image is resident in k3s/containerd and, for VM-local registry
images, whether the tag is available from the local registry. It never builds,
imports, pushes, applies, deletes, or restarts anything.

Options:
  --image IMAGE          Exact image tag/digest to check
  --image-from-dgd NAME  Read the first running pod image whose name contains NAME
                         and check that image
  --mode MODE            auto, resident, or registry (default: auto)
  --require              Exit non-zero if the selected mode is not handoff-ready
  --vm HOST              Target VM IP or hostname (default: $VM_HOST);
                         use local to run directly from the current VM
  --user USER            SSH user (default: $VM_USER)
  --local-registry HOST  Registry host:port to probe (default: $LOCAL_REGISTRY)
  --namespace NAME       Namespace for --image-from-dgd (default: dynamo-system)
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --image-from-dgd) IMAGE_FROM_DGD="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --require) REQUIRE=1; shift ;;
    --vm) VM_HOST="$2"; shift 2 ;;
    --user) VM_USER="$2"; shift 2 ;;
    --local-registry) LOCAL_REGISTRY="$2"; shift 2 ;;
    --namespace) NAMESPACE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$MODE" in
  auto|resident|registry) ;;
  *) echo "unsupported --mode: $MODE" >&2; exit 2 ;;
esac

SSH_TARGET="${VM_USER}@${VM_HOST}"

read -r -d '' REMOTE_SCRIPT <<'EOS' || true
set -euo pipefail

if [[ -z "$IMAGE" && -n "$IMAGE_FROM_DGD" ]]; then
  if [[ -x /usr/local/bin/k3s ]]; then
    KUBECTL=(sudo -E /usr/local/bin/k3s kubectl)
  elif command -v kubectl >/dev/null 2>&1; then
    KUBECTL=(kubectl)
  else
    echo "kubectl_unavailable=1"
    exit 2
  fi
  IMAGE="$(
    "${KUBECTL[@]}" -n "$NAMESPACE" get pods \
      -o custom-columns='POD:.metadata.name,PHASE:.status.phase,IMAGE:.spec.containers[*].image' \
      --no-headers 2>/dev/null \
      | awk -v dgd="$IMAGE_FROM_DGD" '$1 ~ dgd && $2 == "Running" {print $3; exit}'
  )"
  if [[ -z "$IMAGE" ]]; then
    echo "handoff_ready=no"
    echo "reason=no_running_pod_image_for_dgd:$IMAGE_FROM_DGD"
    exit "$([[ "$REQUIRE" == 1 ]] && echo 3 || echo 0)"
  fi
fi

if [[ -z "$IMAGE" ]]; then
  echo "--image or --image-from-dgd is required" >&2
  exit 2
fi

containerd_detail=""
if command -v nerdctl >/dev/null 2>&1; then
  containerd_detail="$(
    sudo nerdctl -n k8s.io images --format '{{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.Digest}}' 2>/dev/null \
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
registry_repo=""
registry_ref=""
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
    registry_digest="sha256:${registry_ref##*@sha256:}"
    registry_url="http://${LOCAL_REGISTRY}/v2/${registry_repo}/manifests/${registry_digest}"
  else
    registry_repo="${registry_ref%:*}"
    registry_tag="${registry_ref##*:}"
    registry_url="http://${LOCAL_REGISTRY}/v2/${registry_repo}/manifests/${registry_tag}"
  fi
  if [[ -n "$registry_repo" && "$registry_available" == yes ]]; then
    if curl -fsSI \
      -H 'Accept: application/vnd.oci.image.index.v1+json' \
      -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
      "$registry_url" >/dev/null 2>&1; then
      registry_tag_available=yes
    else
      registry_tag_available=no
    fi
  elif [[ "$registry_available" == no ]]; then
    registry_tag_available=no
  fi
fi

recommended_pull_policy=unavailable
handoff_ready=no
reason=""
case "$MODE" in
  resident)
    recommended_pull_policy=Never
    if [[ "$containerd_resident" == yes ]]; then
      handoff_ready=yes
    else
      reason="image_not_resident_in_containerd"
    fi
    ;;
  registry)
    recommended_pull_policy=IfNotPresent
    if [[ "$registry_tag_available" == yes ]]; then
      handoff_ready=yes
    else
      reason="image_not_available_from_local_registry"
    fi
    ;;
  auto)
    if [[ "$registry_tag_available" == yes ]]; then
      recommended_pull_policy=IfNotPresent
      handoff_ready=yes
    elif [[ "$containerd_resident" == yes ]]; then
      recommended_pull_policy=Never
      handoff_ready=yes
    else
      reason="image_not_resident_or_registry_available"
    fi
    ;;
esac

printf 'image=%s\n' "$IMAGE"
printf 'mode=%s\n' "$MODE"
printf 'containerd_resident=%s\n' "$containerd_resident"
if [[ -n "$containerd_detail" ]]; then
  printf 'containerd_detail=%s\n' "$containerd_detail"
fi
printf 'local_registry=%s\n' "$LOCAL_REGISTRY"
printf 'registry_available=%s\n' "$registry_available"
printf 'registry_ref=%s\n' "${registry_ref:-none}"
printf 'registry_tag_available=%s\n' "$registry_tag_available"
printf 'recommended_pull_policy=%s\n' "$recommended_pull_policy"
printf 'handoff_ready=%s\n' "$handoff_ready"
if [[ -n "$reason" ]]; then
  printf 'reason=%s\n' "$reason"
fi

if [[ "$REQUIRE" == 1 && "$handoff_ready" != yes ]]; then
  exit 3
fi
EOS

if [[ "$VM_HOST" == local ]]; then
  IMAGE="$IMAGE" IMAGE_FROM_DGD="$IMAGE_FROM_DGD" MODE="$MODE" REQUIRE="$REQUIRE" \
    LOCAL_REGISTRY="$LOCAL_REGISTRY" NAMESPACE="$NAMESPACE" bash -s <<<"$REMOTE_SCRIPT"
else
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
    "IMAGE='$IMAGE' IMAGE_FROM_DGD='$IMAGE_FROM_DGD' MODE='$MODE' REQUIRE='$REQUIRE' LOCAL_REGISTRY='$LOCAL_REGISTRY' NAMESPACE='$NAMESPACE' bash -s" \
    <<<"$REMOTE_SCRIPT"
fi

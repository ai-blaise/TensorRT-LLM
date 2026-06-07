#!/usr/bin/env bash
set -euo pipefail

BUILD_BASE="${BUILD_BASE:-}"
RUNTIME_BASE="${RUNTIME_BASE:-}"
IMAGE_REPO="${IMAGE_REPO:-localhost:5000/local/dynamo-trtllm-optrt-custom}"
TAG_SUFFIX="${TAG_SUFFIX:-fullsource-r20}"
PUSH_LOCAL_REGISTRY="${PUSH_LOCAL_REGISTRY:-1}"
JOBS="${JOBS:-48}"

usage() {
  cat <<'EOF'
Usage: deploy/disagg_pd_r20/build_fullsource_image.sh [options]

Build a full-source TensorRT-LLM runtime image for the R20 NIXL/LayerSplit gate.
The build stage compiles TRT-LLM C++/Python artifacts from the current op-trt
checkout; the runtime stage layers them onto the selected Dynamo runtime image.

Options:
  --build-base IMAGE      Build-stage base image with CUDA/TRT-LLM build deps
  --runtime-base IMAGE    Runtime base image, usually the current routerpin image
  --image-repo REPO       Output repository (default: localhost:5000/local/dynamo-trtllm-optrt-custom)
  --tag-suffix SUFFIX     Output tag suffix (default: fullsource-r20)
  --no-push               Do not docker push the resulting local-registry tag
  -h, --help              Show this help

Environment equivalents: BUILD_BASE, RUNTIME_BASE, IMAGE_REPO, TAG_SUFFIX,
PUSH_LOCAL_REGISTRY.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-base) BUILD_BASE="$2"; shift 2 ;;
    --runtime-base) RUNTIME_BASE="$2"; shift 2 ;;
    --image-repo) IMAGE_REPO="$2"; shift 2 ;;
    --tag-suffix) TAG_SUFFIX="$2"; shift 2 ;;
    --no-push) PUSH_LOCAL_REGISTRY=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$BUILD_BASE" ]]; then
  echo "--build-base or BUILD_BASE is required" >&2
  exit 2
fi
if [[ -z "$RUNTIME_BASE" ]]; then
  echo "--runtime-base or RUNTIME_BASE is required" >&2
  exit 2
fi

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

SHA="$(git rev-parse --short=12 HEAD)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
IMAGE_TAG="${IMAGE_REPO}:optrt-${SHA}-${TAG_SUFFIX}-${STAMP}"

DOCKER_BUILDKIT=1 docker build \
  --build-arg "BUILD_BASE=${BUILD_BASE}" \
  --build-arg "RUNTIME_BASE=${RUNTIME_BASE}" \
  --build-arg "OPTRT_SOURCE_SHA=$(git rev-parse HEAD)" \
  --build-arg "FULL_BUILD_JOBS=${JOBS}" \
  -f deploy/disagg_pd_r20/Dockerfile.r20-fullsource \
  -t "$IMAGE_TAG" \
  .

if [[ "$PUSH_LOCAL_REGISTRY" == 1 ]]; then
  docker push "$IMAGE_TAG"
fi

printf 'image=%s\n' "$IMAGE_TAG"
printf 'source_sha=%s\n' "$(git rev-parse HEAD)"
printf 'build_base=%s\n' "$BUILD_BASE"
printf 'runtime_base=%s\n' "$RUNTIME_BASE"

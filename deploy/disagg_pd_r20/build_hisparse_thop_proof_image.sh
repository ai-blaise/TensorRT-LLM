#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build a tiny deployment-runtime proof image that contains the exact
# branch-built HiSparse smoke thop and its CUDA planner/copy smoke script.
#
# This is intentionally not a replacement for the full r20 serving image/wheel
# gate. It is the low-cost intermediate proof that the deployment runtime can
# load and execute the branch-built native HiSparse thops without bind-mounting
# the library at test time.

set -euo pipefail

BASE_IMAGE="${BASE_IMAGE:-}"
THOP_LIB="${THOP_LIB:-/home/spencer/work/build-cache/hisparse-thop/cpp-build/tensorrt_llm/thop/libth_hisparse_smoke.so}"
IMAGE_REPO="${IMAGE_REPO:-localhost:5000/local/dynamo-trtllm-optrt-custom}"
TAG_SUFFIX="${TAG_SUFFIX:-hisparse-thop-proof}"
PUSH_LOCAL_REGISTRY="${PUSH_LOCAL_REGISTRY:-0}"

usage() {
  cat <<'EOF'
Usage: deploy/disagg_pd_r20/build_hisparse_thop_proof_image.sh [options]

Options:
  --base-image IMAGE   Deployment runtime base image to extend
  --thop-lib PATH      Branch-built libth_hisparse_smoke.so path
  --image-repo REPO    Output repository
  --tag-suffix TEXT    Human suffix after the git sha
  --push               Push the resulting tag to the local registry
  -h, --help           Show this help

Environment equivalents: BASE_IMAGE, THOP_LIB, IMAGE_REPO, TAG_SUFFIX,
PUSH_LOCAL_REGISTRY.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-image) BASE_IMAGE="$2"; shift 2 ;;
    --thop-lib) THOP_LIB="$2"; shift 2 ;;
    --image-repo) IMAGE_REPO="$2"; shift 2 ;;
    --tag-suffix) TAG_SUFFIX="$2"; shift 2 ;;
    --push) PUSH_LOCAL_REGISTRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$BASE_IMAGE" ]]; then
  echo "--base-image or BASE_IMAGE is required" >&2
  exit 2
fi
if [[ ! -s "$THOP_LIB" ]]; then
  echo "HiSparse thop library not found or empty: $THOP_LIB" >&2
  exit 2
fi

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

SMOKE_SCRIPT="blaise_perf/hisparse/native_planner_copy_smoke.py"
if [[ ! -s "$SMOKE_SCRIPT" ]]; then
  echo "HiSparse planner/copy smoke script missing: $SMOKE_SCRIPT" >&2
  exit 2
fi

SHA="$(git rev-parse --short=12 HEAD)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
IMAGE_TAG="${IMAGE_REPO}:optrt-${SHA}-${TAG_SUFFIX}-${STAMP}"
CTX="$(mktemp -d "${TMPDIR:-/tmp}/hisparse-thop-proof.XXXXXX")"
cleanup() {
  rm -rf "$CTX"
}
trap cleanup EXIT

cp "$THOP_LIB" "$CTX/libth_hisparse_smoke.so"
cp "$SMOKE_SCRIPT" "$CTX/native_planner_copy_smoke.py"
cat >"$CTX/Dockerfile" <<'DOCKERFILE'
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}
USER root
ARG SITE=/opt/ai-blaise/hisparse
ARG OPTRT_SOURCE_SHA=unknown
RUN mkdir -p "${SITE}"
COPY --chown=dynamo:0 libth_hisparse_smoke.so ${SITE}/libth_hisparse_smoke.so
COPY --chown=dynamo:0 native_planner_copy_smoke.py ${SITE}/native_planner_copy_smoke.py
RUN chmod 0755 ${SITE}/native_planner_copy_smoke.py \
    && echo "${OPTRT_SOURCE_SHA}" > /opt/ai-blaise/optrt_hisparse_thop_source_sha \
    && chown dynamo:0 /opt/ai-blaise/optrt_hisparse_thop_source_sha
LABEL ai.blaise.hisparse.thop_proof="true"
USER dynamo
DOCKERFILE

DOCKER_BUILDKIT=1 docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "OPTRT_SOURCE_SHA=$(git rev-parse HEAD)" \
  -t "$IMAGE_TAG" \
  "$CTX"

if [[ "$PUSH_LOCAL_REGISTRY" == 1 ]]; then
  docker push "$IMAGE_TAG"
fi

printf 'image=%s\n' "$IMAGE_TAG"
printf 'source_sha=%s\n' "$(git rev-parse HEAD)"
printf 'base_image=%s\n' "$BASE_IMAGE"
printf 'thop_lib=%s\n' "$THOP_LIB"

#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build a deployment-runtime proof image that installs the current branch's
# Python package into the normal serving site-packages location and installs a
# branch-built libth_common.so under tensorrt_llm/libs. The smoke must import
# tensorrt_llm normally and run HiSparse native ops without an explicit
# torch.ops.load_library path.
#
# This is a low-cost package/import gate. It does not replace the heavier
# fullsource image gate when branch-built generated bindings or plugin libs are
# required, but it can carry those already-built artifacts from the persistent
# VM cache when they are available.

set -euo pipefail

BASE_IMAGE="${BASE_IMAGE:-}"
TH_COMMON_LIB="${TH_COMMON_LIB:-/home/spencer/work/build-cache/hisparse-thop/cpp-build/tensorrt_llm/thop/libth_common.so}"
EXTRA_LIBS="${EXTRA_LIBS:-}"
PACKAGE_ROOT_FILES="${PACKAGE_ROOT_FILES:-}"
PACKAGE_LIB_FILES="${PACKAGE_LIB_FILES:-}"
PACKAGE_DIR="${PACKAGE_DIR:-}"
SITE_PACKAGES="${SITE_PACKAGES:-/opt/dynamo/venv/lib/python3.12/site-packages}"
IMAGE_REPO="${IMAGE_REPO:-localhost:5000/local/dynamo-trtllm-optrt-custom}"
TAG_SUFFIX="${TAG_SUFFIX:-hisparse-serving-import-proof}"
PUSH_LOCAL_REGISTRY="${PUSH_LOCAL_REGISTRY:-0}"
RUN_SMOKE="${RUN_SMOKE:-0}"
GPU_DEVICE="${GPU_DEVICE:-0}"

usage() {
  cat <<'EOF'
Usage: deploy/disagg_pd_r20/build_hisparse_serving_import_proof_image.sh [options]

Options:
  --base-image IMAGE       Deployment runtime base image to extend
  --th-common-lib PATH     Branch-built libth_common.so path
  --extra-libs LIST        Colon-separated native libraries copied to tensorrt_llm/libs
  --package-root-files LIST
                            Colon-separated files copied to tensorrt_llm/
                            (for example bindings*.so or transfer-agent binding)
  --package-lib-files LIST  Colon-separated files copied to tensorrt_llm/libs
                            (for example plugin/wrapper libraries)
  --package-dir PATH       Branch tensorrt_llm package dir (default: repo/tensorrt_llm)
  --site-packages PATH     Runtime site-packages root
  --image-repo REPO        Output repository
  --tag-suffix TEXT        Human suffix after the git sha
  --push                   Push the resulting tag to the local registry
  --run-smoke              Run the serving-import CUDA smoke after image build
  --gpu-device ID          GPU id for --run-smoke (default: 0)
  -h, --help               Show this help

Environment equivalents: BASE_IMAGE, TH_COMMON_LIB, EXTRA_LIBS, PACKAGE_DIR,
PACKAGE_ROOT_FILES, PACKAGE_LIB_FILES, SITE_PACKAGES, IMAGE_REPO, TAG_SUFFIX,
PUSH_LOCAL_REGISTRY, RUN_SMOKE, GPU_DEVICE.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-image) BASE_IMAGE="$2"; shift 2 ;;
    --th-common-lib) TH_COMMON_LIB="$2"; shift 2 ;;
    --extra-libs) EXTRA_LIBS="$2"; shift 2 ;;
    --package-root-files) PACKAGE_ROOT_FILES="$2"; shift 2 ;;
    --package-lib-files) PACKAGE_LIB_FILES="$2"; shift 2 ;;
    --package-dir) PACKAGE_DIR="$2"; shift 2 ;;
    --site-packages) SITE_PACKAGES="$2"; shift 2 ;;
    --image-repo) IMAGE_REPO="$2"; shift 2 ;;
    --tag-suffix) TAG_SUFFIX="$2"; shift 2 ;;
    --push) PUSH_LOCAL_REGISTRY=1; shift ;;
    --run-smoke) RUN_SMOKE=1; shift ;;
    --gpu-device) GPU_DEVICE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$BASE_IMAGE" ]]; then
  echo "--base-image or BASE_IMAGE is required" >&2
  exit 2
fi
if [[ ! -s "$TH_COMMON_LIB" ]]; then
  echo "libth_common.so not found or empty: $TH_COMMON_LIB" >&2
  exit 2
fi

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

if [[ -z "$PACKAGE_DIR" ]]; then
  PACKAGE_DIR="$ROOT_DIR/tensorrt_llm"
fi
if [[ ! -d "$PACKAGE_DIR" ]]; then
  echo "tensorrt_llm package dir not found: $PACKAGE_DIR" >&2
  exit 2
fi

append_colon_file() {
  local var_name="$1"
  local path="$2"
  [[ -s "$path" ]] || return 0
  local current="${!var_name:-}"
  if [[ -z "$current" ]]; then
    printf -v "$var_name" '%s' "$path"
  else
    printf -v "$var_name" '%s:%s' "$current" "$path"
  fi
}

append_matching_files() {
  local var_name="$1"
  local pattern="$2"
  local match
  while IFS= read -r match; do
    append_colon_file "$var_name" "$match"
  done < <(compgen -G "$pattern" || true)
}

build_root="$(cd "$(dirname "$TH_COMMON_LIB")/.." && pwd)"

if [[ -z "$EXTRA_LIBS" ]]; then
  default_libs=(
    "$build_root/libtensorrt_llm.so"
    "$build_root/runtime/utils/libpg_utils.so"
    "$build_root/kernels/decoderMaskedMultiheadAttention/libdecoder_attention_0.so"
    "$build_root/kernels/decoderMaskedMultiheadAttention/libdecoder_attention_1.so"
    "$build_root/executor/cache_transmission/nixl_utils/libtensorrt_llm_nixl_wrapper.so"
    "$build_root/executor/cache_transmission/ucx_utils/libtensorrt_llm_ucx_wrapper.so"
    "$build_root/executor/cache_transmission/mooncake_utils/libtensorrt_llm_mooncake_wrapper.so"
  )
  for lib in "${default_libs[@]}"; do
    append_colon_file EXTRA_LIBS "$lib"
  done
fi

if [[ -z "$PACKAGE_ROOT_FILES" ]]; then
  append_matching_files PACKAGE_ROOT_FILES "$build_root/nanobind/bindings*.so"
  append_matching_files PACKAGE_ROOT_FILES "$build_root/executor/cache_transmission/nixl_utils/tensorrt_llm_transfer_agent_binding*.so"
fi

if [[ -z "$PACKAGE_LIB_FILES" ]]; then
  append_matching_files PACKAGE_LIB_FILES "$build_root/plugins/libnvinfer_plugin_tensorrt_llm.so"
fi

SMOKE_SCRIPT="blaise_perf/hisparse/native_planner_copy_smoke.py"
SPARSE_MLA_SMOKE_SCRIPT="blaise_perf/hisparse/sparse_mla_kvarn_hot_smoke.py"
SERVING_SMOKE_SCRIPT="blaise_perf/hisparse/serving_import_smoke.py"
for script in "$SMOKE_SCRIPT" "$SPARSE_MLA_SMOKE_SCRIPT" "$SERVING_SMOKE_SCRIPT"; do
  if [[ ! -f "$script" ]]; then
    echo "missing smoke script: $script" >&2
    exit 2
  fi
done

SHA="$(git rev-parse --short=12 HEAD)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
IMAGE_TAG="${IMAGE_REPO}:optrt-${SHA}-${TAG_SUFFIX}-${STAMP}"
CTX="$(mktemp -d)"
cleanup() {
  rm -rf "$CTX"
}
trap cleanup EXIT

mkdir -p "$CTX/libs" "$CTX/smoke"
tar --exclude='__pycache__' --exclude='*.pyc' -C "$(dirname "$PACKAGE_DIR")" \
  -cf - "$(basename "$PACKAGE_DIR")" | tar -C "$CTX" -xf -
cp "$TH_COMMON_LIB" "$CTX/libs/libth_common.so"
copy_colon_files() {
  local files="$1"
  local dest="$2"
  local label="$3"
  [[ -z "$files" ]] && return 0
  IFS=':' read -ra paths <<<"$files"
  for path in "${paths[@]}"; do
    [[ -z "$path" ]] && continue
    if [[ ! -s "$path" ]]; then
      echo "$label not found or empty: $path" >&2
      exit 2
    fi
    cp "$path" "$dest/$(basename "$path")"
  done
}

copy_colon_files "$EXTRA_LIBS" "$CTX/libs" "extra native library"
copy_colon_files "$PACKAGE_LIB_FILES" "$CTX/libs" "package library"
copy_colon_files "$PACKAGE_ROOT_FILES" "$CTX/tensorrt_llm" "package root artifact"
cp "$SMOKE_SCRIPT" "$CTX/smoke/native_planner_copy_smoke.py"
cp "$SPARSE_MLA_SMOKE_SCRIPT" "$CTX/smoke/sparse_mla_kvarn_hot_smoke.py"
cp "$SERVING_SMOKE_SCRIPT" "$CTX/smoke/serving_import_smoke.py"

cat >"$CTX/Dockerfile" <<'DOCKERFILE'
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}
USER root

ARG SITE_PACKAGES=/opt/dynamo/venv/lib/python3.12/site-packages
ARG OPTRT_SOURCE_SHA=unknown

RUN mkdir -p "${SITE_PACKAGES}/tensorrt_llm/libs" /opt/ai-blaise/hisparse
COPY --chown=dynamo:0 tensorrt_llm/ ${SITE_PACKAGES}/tensorrt_llm/
COPY --chown=dynamo:0 libs/ ${SITE_PACKAGES}/tensorrt_llm/libs/
COPY --chown=dynamo:0 smoke/ /opt/ai-blaise/hisparse/
RUN chmod 0755 /opt/ai-blaise/hisparse/native_planner_copy_smoke.py \
    /opt/ai-blaise/hisparse/sparse_mla_kvarn_hot_smoke.py \
    /opt/ai-blaise/hisparse/serving_import_smoke.py \
    && echo "${OPTRT_SOURCE_SHA}" > /opt/ai-blaise/optrt_hisparse_serving_import_source_sha \
    && /opt/dynamo/venv/bin/python3 -m py_compile \
      /opt/ai-blaise/hisparse/native_planner_copy_smoke.py \
      /opt/ai-blaise/hisparse/sparse_mla_kvarn_hot_smoke.py \
      /opt/ai-blaise/hisparse/serving_import_smoke.py

LABEL ai.blaise.hisparse.serving_import_proof="true"
USER dynamo
DOCKERFILE

DOCKER_BUILDKIT=1 docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "SITE_PACKAGES=$SITE_PACKAGES" \
  --build-arg "OPTRT_SOURCE_SHA=$(git rev-parse HEAD)" \
  -f "$CTX/Dockerfile" \
  -t "$IMAGE_TAG" \
  "$CTX"

if [[ "$PUSH_LOCAL_REGISTRY" == 1 ]]; then
  docker push "$IMAGE_TAG"
fi

printf 'image=%s\n' "$IMAGE_TAG"
printf 'source_sha=%s\n' "$(git rev-parse HEAD)"
printf 'base_image=%s\n' "$BASE_IMAGE"
printf 'site_packages=%s\n' "$SITE_PACKAGES"
printf 'th_common_lib=%s\n' "$TH_COMMON_LIB"
printf 'extra_libs=%s\n' "$EXTRA_LIBS"
printf 'package_lib_files=%s\n' "$PACKAGE_LIB_FILES"
printf 'package_root_files=%s\n' "$PACKAGE_ROOT_FILES"
printf 'serving_smoke=/opt/ai-blaise/hisparse/serving_import_smoke.py\n'

if [[ "$RUN_SMOKE" == 1 ]]; then
  docker run --rm --gpus "device=${GPU_DEVICE}" --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --cpus="${SMOKE_CPUS:-4}" --memory="${SMOKE_MEMORY:-24g}" \
    --entrypoint /bin/bash \
    "$IMAGE_TAG" \
    -lc "set -euo pipefail; export LD_LIBRARY_PATH='${SITE_PACKAGES}/tensorrt_llm/libs':\${LD_LIBRARY_PATH:-}; /opt/dynamo/venv/bin/python3 /opt/ai-blaise/hisparse/serving_import_smoke.py --device cuda:0 --expect-site-packages '${SITE_PACKAGES}'"
fi

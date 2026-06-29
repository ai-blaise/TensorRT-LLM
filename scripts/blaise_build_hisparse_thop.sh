#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Configure and build the native th_common library for the OP-TRT HiSparse
# thops in a persistent VM-side Docker build cache.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CACHE_DIR="${CACHE_DIR:-${HOME}/build-cache/hisparse-thop}"
IMAGE="${IMAGE:-local/dynamo-trtllm-optrt-custom:hisa-buildtools-20260531}"
JOBS="${JOBS:-8}"
MEMORY="${MEMORY:-64g}"
CPUS="${CPUS:-8}"
BUILD_USER="${BUILD_USER:-}"
APT_INSTALL_CMAKE="${APT_INSTALL_CMAKE:-0}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-100-real}"
TARGETS="${TARGETS:-th_common}"
WHEEL_TARGETS="${WHEEL_TARGETS:-th_common}"
BUILD_DEEP_GEMM="${BUILD_DEEP_GEMM:-OFF}"
ENABLE_CUBLASLT_FP4_GEMM="${ENABLE_CUBLASLT_FP4_GEMM:-ON}"
NCCL_INCLUDE_DIR="${NCCL_INCLUDE_DIR:-/opt/dynamo/venv/lib/python3.12/site-packages/nvidia/nccl/include}"
NCCL_LIBRARY="${NCCL_LIBRARY:-/usr/lib/x86_64-linux-gnu/libnccl.so}"
TENSORRT_ROOT="${TENSORRT_ROOT:-/usr/local/tensorrt}"

BUILD_DIR="${CACHE_DIR}/cpp-build"
HOME_LOCAL="${CACHE_DIR}/home-dynamo-local"
CUTLASS_PY="${BUILD_DIR}/_deps/cutlass-src/python"
USER_SITE="/home/dynamo/.local/lib/python3.12/site-packages"

mkdir -p "${BUILD_DIR}" "${HOME_LOCAL}/lib/python3.12/site-packages"

docker run --rm \
  --entrypoint /bin/bash \
  --cpus="${CPUS}" \
  --memory="${MEMORY}" \
  ${BUILD_USER:+--user="${BUILD_USER}"} \
  -e HISPARSE_WHEEL_TARGETS="${WHEEL_TARGETS}" \
  -e PYTHONPATH="${USER_SITE}:${CUTLASS_PY}" \
  -v "${REPO_DIR}:/work" \
  -v "${CACHE_DIR}:/build" \
  -v "${HOME_LOCAL}:/home/dynamo/.local" \
  -w /build/cpp-build \
  "${IMAGE}" \
  -lc "
set -euo pipefail
if [[ \"${APT_INSTALL_CMAKE}\" == \"1\" ]] && ! command -v cmake >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends cmake
fi
export PYTHONPATH=${USER_SITE}:${CUTLASS_PY}:\${PYTHONPATH:-}
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PYT=ON \
  -DBUILD_DEEP_EP=OFF \
  -DBUILD_DEEP_GEMM=${BUILD_DEEP_GEMM} \
  -DBUILD_FLASH_MLA=OFF \
  -DNVTX_DISABLE=ON \
  -DBUILD_MICRO_BENCHMARKS=OFF \
  -DBUILD_WHEEL_TARGETS=\"\${HISPARSE_WHEEL_TARGETS}\" \
  -DPython_EXECUTABLE=/opt/dynamo/venv/bin/python3 \
  -DPython3_EXECUTABLE=/opt/dynamo/venv/bin/python3 \
  -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
  -DTensorRT_ROOT=${TENSORRT_ROOT} \
  -DCMAKE_CXX_COMPILER_LAUNCHER= \
  -DCMAKE_CUDA_COMPILER_LAUNCHER= \
  -DFAST_BUILD=ON \
  -DNVRTC_DYNAMIC_LINKING=ON \
  -DUSING_OSS_CUTLASS_LOW_LATENCY_GEMM=OFF \
  -DUSING_OSS_CUTLASS_FP4_GEMM=OFF \
  -DUSING_OSS_CUTLASS_MOE_GEMM=ON \
  -DUSING_OSS_CUTLASS_ALLREDUCE_GEMM=OFF \
  -DENABLE_CUBLASLT_FP4_GEMM=${ENABLE_CUBLASLT_FP4_GEMM} \
  -DTRTLLM_FETCHCONTENT_CACHE=/work/3rdparty/.cache_3rdparty \
  -DNCCL_INCLUDE_DIR=${NCCL_INCLUDE_DIR} \
  -DNCCL_LIBRARY=${NCCL_LIBRARY} \
  -GNinja \
  -S /work/cpp
cmake --build . --config Release --parallel ${JOBS} --target ${TARGETS}
"

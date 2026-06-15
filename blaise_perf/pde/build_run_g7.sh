#!/usr/bin/env bash
# PDE G7 — build + run the IN-KERNEL EP=4 all-to-all dispatch+combine microbench.
# Runs inside the proof container, worktree mounted at /wt. GPU 0-3 ONLY.
#
# Host invocation (001):
#   flock -n /tmp/gpu001_lock_a \
#     timeout -s KILL 220 docker run --rm --init --name pde_g7_run \
#       --gpus all -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
#       -v /home/spencer/work/pde-wt:/wt --entrypoint /bin/bash <proof-image> \
#       /wt/blaise_perf/pde/build_run_g7.sh
set -euo pipefail

WT="${WT:-/wt}"
SRC_DIR="$WT/cpp/tensorrt_llm/kernels/pde"
BENCH="$WT/blaise_perf/pde/pde_g7_bench.cu"
OUT="/tmp/pde_g7_bench"

# NCCL (independent reference + launch-boundary baseline).
NCCL_ROOT="/opt/dynamo/venv/lib/python3.12/site-packages/nvidia/nccl"
NCCL_INC="$NCCL_ROOT/include"
NCCL_LIB="$NCCL_ROOT/lib"
CUDA_STUB="/usr/local/cuda/lib64/stubs"

echo "=== nvcc version ==="
nvcc --version | tail -2
echo "=== building PDE G7 in-kernel all-to-all bench (sm_100) ==="
nvcc -std=c++17 -arch=sm_100 -O3 \
  -I"$SRC_DIR" -I"$NCCL_INC" \
  "$BENCH" -o "$OUT" \
  -L"$NCCL_LIB" -lnccl \
  -L"$CUDA_STUB" -lcuda \
  -lpthread
echo "=== build OK -> $OUT ==="

# Real driver libcuda at runtime (stub is link-only).
export LD_LIBRARY_PATH="$NCCL_LIB:${LD_LIBRARY_PATH:-}"
# NVLS multicast cannot bind in this VM (no fabric manager / NVSwitch) — force
# NCCL onto its P2P/ring NVLink transport (the launch-boundary baseline the
# in-kernel engine replaces). Without this NCCL aborts at init binding NVLS.
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_P2P_LEVEL=NVL

echo "=== nvidia-smi (visible devices) ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader || true

echo "=== running G7 gate (GPU 0-3) ==="
"$OUT"
RC=$?
echo "=== bench exit code: $RC ==="
exit $RC

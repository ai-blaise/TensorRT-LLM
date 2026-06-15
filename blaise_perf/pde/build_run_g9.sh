#!/usr/bin/env bash
# PDE G9 — build + run the CROSS-STEP PERSISTENCE microbench (one persistent
# cooperative engine that loops over decode steps device-side vs per-step
# relaunch). Runs inside the proof container, worktree mounted at /wt. GPU 0 ONLY.
#
# Host invocation (001):
#   flock -n /tmp/gpu001_lock_a \
#     timeout -s KILL 200 docker run --rm --init --name pde_g9_run \
#       --gpus all -e CUDA_VISIBLE_DEVICES=0 \
#       -v /home/spencer/work/pde-wt:/wt --entrypoint /bin/bash <proof-image> \
#       /wt/blaise_perf/pde/build_run_g9.sh
set -euo pipefail

WT="${WT:-/wt}"
SRC_DIR="$WT/cpp/tensorrt_llm/kernels/pde"
BENCH="$WT/blaise_perf/pde/pde_g9_bench.cu"
OUT="/tmp/pde_g9_bench"

echo "=== nvcc version ==="
nvcc --version | tail -2
echo "=== building PDE G9 cross-step persistence bench (sm_100) ==="
# --threads for host pthreads (host producer/consumer); -lpthread for std::thread.
nvcc -std=c++17 -arch=sm_100 -O3 \
  -I"$SRC_DIR" \
  "$BENCH" -o "$OUT" \
  -lpthread
echo "=== build OK -> $OUT ==="

echo "=== nvidia-smi (visible device) ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader || true

echo "=== running G9 gate (GPU 0) ==="
"$OUT"
RC=$?
echo "=== bench exit code: $RC ==="
exit $RC

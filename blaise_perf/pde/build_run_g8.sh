#!/usr/bin/env bash
# PDE G8 — build + run the MTP / SPECULATIVE-DECODE DEVICE-LOOP microbench (one
# persistent cooperative engine that runs the draft -> verify -> VARIABLE-LENGTH
# accept loop device-side vs a host-orchestrated accept-length d2h + branch + h2d
# per step). Runs inside the proof container, worktree mounted at /wt. GPU 0 ONLY.
#
# Host invocation (001):
#   flock -n /tmp/gpu001_lock_a \
#     timeout -s KILL 200 docker run --rm --init --name pde_g8_run \
#       --gpus all -e CUDA_VISIBLE_DEVICES=0 \
#       -v /home/spencer/work/pde-wt:/wt --entrypoint /bin/bash <proof-image> \
#       /wt/blaise_perf/pde/build_run_g8.sh
set -euo pipefail

WT="${WT:-/wt}"
SRC_DIR="$WT/cpp/tensorrt_llm/kernels/pde"
BENCH="$WT/blaise_perf/pde/pde_g8_bench.cu"
OUT="/tmp/pde_g8_bench"

echo "=== nvcc version ==="
nvcc --version | tail -2
echo "=== building PDE G8 MTP device-loop bench (sm_100) ==="
# -lpthread for the host producer/consumer std::threads.
nvcc -std=c++17 -arch=sm_100 -O3 \
  -I"$SRC_DIR" \
  "$BENCH" -o "$OUT" \
  -lpthread
echo "=== build OK -> $OUT ==="

echo "=== nvidia-smi (visible device) ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader || true

echo "=== running G8 gate (GPU 0) ==="
"$OUT"
RC=$?
echo "=== bench exit code: $RC ==="
exit $RC

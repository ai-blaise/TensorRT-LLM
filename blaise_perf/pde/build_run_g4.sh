#!/usr/bin/env bash
# PDE G4 — build + run the HETEROGENEOUS-WORKER copy ∥ compute overlap microbench.
# Run inside the proof container (entrypoint /bin/bash), mounting the worktree at
# /wt:
#   docker run --rm --init --gpus all -e CUDA_VISIBLE_DEVICES=0 \
#     -v /home/spencer/work/pde-wt:/wt --entrypoint /bin/bash <proof-image> \
#     /wt/blaise_perf/pde/build_run_g4.sh
# On the host, wrap with: flock -n /tmp/gpu001_lock_a timeout -s KILL 150 docker run ...
set -euo pipefail

WT="${WT:-/wt}"
SRC_DIR="$WT/cpp/tensorrt_llm/kernels/pde"
BENCH="$WT/blaise_perf/pde/pde_g4_bench.cu"
OUT="/tmp/pde_g4_bench"

echo "=== nvcc version ==="
nvcc --version | tail -2
echo "=== building PDE G4 het-worker copy∥compute overlap bench (sm_100) ==="
nvcc -std=c++17 -arch=sm_100 -O3 \
  -I"$SRC_DIR" \
  "$BENCH" -o "$OUT"
echo "=== build OK -> $OUT ==="

echo "=== nvidia-smi (visible device) ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader || true

echo "=== running G4 gate ==="
"$OUT"
RC=$?
echo "=== bench exit code: $RC ==="
exit $RC

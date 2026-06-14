#!/usr/bin/env bash
# PDE G0 — build + run the persistent-grid substrate microbench.
# Run inside the proof container (entrypoint /bin/bash), mounting the worktree at /wt:
#   docker run --rm --gpus '"device=0"' -e CUDA_VISIBLE_DEVICES=0 \
#     -v /home/spencer/work/pde-wt:/wt --entrypoint /bin/bash <proof-image> \
#     /wt/blaise_perf/pde/build_run_g0.sh
# On the host, wrap the docker run with: flock -n /tmp/gpu001_lock_a -c '<docker run ...>'
set -euo pipefail

WT="${WT:-/wt}"
SRC_DIR="$WT/cpp/tensorrt_llm/kernels/pde"
BENCH="$WT/blaise_perf/pde/pde_g0_bench.cu"
OUT="/tmp/pde_g0_bench"

echo "=== nvcc version ==="
nvcc --version | tail -2
echo "=== building PDE G0 microbench (sm_100) ==="
nvcc -std=c++17 -arch=sm_100 -O3 \
  -I"$SRC_DIR" \
  "$BENCH" -o "$OUT"
echo "=== build OK -> $OUT ==="

echo "=== nvidia-smi (visible device) ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader || true

echo "=== running G0 gate ==="
"$OUT"
RC=$?
echo "=== bench exit code: $RC ==="
exit $RC

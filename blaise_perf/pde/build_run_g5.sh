#!/usr/bin/env bash
# PDE G5 — build + run the FULL-LAYER-STACK persistent megakernel microbench:
# N synthetic decoder layers under ONE persistent cooperative launch with
# cross-layer weight prefetch, vs a per-layer-launch baseline vs a CPU reference.
# Runs inside the proof container, worktree mounted at /wt. GPU 0 ONLY.
#
# Host invocation (001):
#   flock -n /tmp/gpu001_lock_a \
#     timeout -s KILL 220 docker run --rm --init --name pde_g5_run \
#       --gpus all -e CUDA_VISIBLE_DEVICES=0 \
#       -v /home/spencer/work/pde-wt:/wt --entrypoint /bin/bash <proof-image> \
#       /wt/blaise_perf/pde/build_run_g5.sh
set -euo pipefail

WT="${WT:-/wt}"
SRC_DIR="$WT/cpp/tensorrt_llm/kernels/pde"
BENCH="$WT/blaise_perf/pde/pde_g5_bench.cu"
OUT="/tmp/pde_g5_bench"

echo "=== nvcc version ==="
nvcc --version | tail -2
echo "=== building PDE G5 full-stack megakernel bench (sm_100) ==="
# -Xcompiler -fopenmp + -lgomp: the f64 CPU reference parallelizes the per-layer
# GEMMs (the N=61 stack reference is multi-GFLOP single-threaded otherwise).
nvcc -std=c++17 -arch=sm_100 -O3 \
  -I"$SRC_DIR" \
  -Xcompiler -fopenmp \
  "$BENCH" -o "$OUT" -lgomp
echo "=== build OK -> $OUT ==="

echo "=== nvidia-smi (visible device) ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader || true

echo "=== running G5 gate (GPU 0) ==="
"$OUT"
RC=$?
echo "=== bench exit code: $RC ==="
exit $RC

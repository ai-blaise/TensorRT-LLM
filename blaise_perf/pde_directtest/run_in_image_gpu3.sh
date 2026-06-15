#!/bin/bash
# GPU3 runner for the PDE-pdewire JIT validation. Mounts the pdewire worktree
# read-only and runs the given python in the proof image under flock gpu3.
set -uo pipefail
PYF="$1"; shift || true
IMG=localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-aaa7e2b542b2-hisparse-current-head-proof-20260613T155354Z
flock -n /tmp/gpu001_lock_3 docker run --rm --gpus device=3 --init \
  -e CUDA_VISIBLE_DEVICES=0 -e PYTHONUNBUFFERED=1 -e HF_HUB_OFFLINE=1 \
  -v "$PYF":/tmp/run.py:ro \
  -v /home/spencer/work/pde-wt-pdewire:/host_repo:ro \
  --entrypoint bash "$IMG" -c "timeout ${PDE_TIMEOUT:-420} python /tmp/run.py $*"

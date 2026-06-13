#!/bin/bash
# Helper: run a megakernel script in the serving container on a chosen device.
# Usage: run_in_docker.sh <device> <script.py> <out.log> [args...]
set -euo pipefail
DEV="$1"; SCRIPT="$2"; OUT="$3"; shift 3
IMG=localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-19d82b488-fullsource-msgpack-0426
docker run --rm --gpus "\"device=${DEV}\"" \
  -v /home/sjpat/TensorRT-LLM:/work --entrypoint bash "$IMG" \
  -c "cd /work && /opt/dynamo/venv/bin/python3 -u /work/.bench_runs_claude/megakernel/${SCRIPT} $* 2>&1" \
  | tee "/work/.bench_runs_claude/megakernel/out/${OUT}"

#!/bin/bash
# Direct TensorRT-LLM TP4 serve path for Corsaire benchmarks.
# This expects trtllm-serve from the current environment, not a Dynamo launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TRTLLM_SERVE="${TRTLLM_SERVE:-trtllm-serve}"
MODEL="${MODEL:-/models/BlaiseAI/corsaire-research-preview}"
CONFIG="${CONFIG:-$SCRIPT_DIR/corsaire_tp4.yaml}"
GPU_DEVICES="${GPU_DEVICES:-0,1,2,3}"
PORT="${PORT:-8077}"
COPY_ID="${COPY_ID:-gpus_${GPU_DEVICES//,/}}"
BENCH_ROOT="${BENCH_ROOT:-$HOME/wt/_hb}"
SDIR="${SDIR:-$BENCH_ROOT/corsaire_direct_${COPY_ID}}"
LOGD="${LOGD:-$SDIR/logs}"
HB="${HB:-$BENCH_ROOT/corsaire-direct-${COPY_ID}.log}"
PID_FILE="${PID_FILE:-$SDIR/serve.pid}"
FOREGROUND="${FOREGROUND:-0}"

mkdir -p "$LOGD" "$(dirname "$HB")"

if [[ "$TRTLLM_SERVE" != */* ]] && ! command -v "$TRTLLM_SERVE" >/dev/null 2>&1; then
  echo "trtllm-serve not found. Set TRTLLM_SERVE=/path/to/trtllm-serve or activate the built TensorRT-LLM env." >&2
  exit 127
fi

if [[ ! -d "$MODEL" ]]; then
  echo "model path does not exist: $MODEL" >&2
  exit 2
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "config path does not exist: $CONFIG" >&2
  exit 2
fi

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE")"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" >/dev/null 2>&1; then
    echo "[$(date -u +%H:%M:%SZ)] direct serve: stopping existing pid $OLD_PID" | tee -a "$HB"
    kill "$OLD_PID" >/dev/null 2>&1 || true
    wait "$OLD_PID" >/dev/null 2>&1 || true
  fi
fi

CMD=(
  "$TRTLLM_SERVE" serve "$MODEL"
  --host 0.0.0.0
  --port "$PORT"
  --backend pytorch
  --tp_size 4
  --max_batch_size 64
  --max_num_tokens 2048
  --max_seq_len 132096
  --trust_remote_code
  --extra_llm_api_options "$CONFIG"
)

{
  echo "[$(date -u +%H:%M:%SZ)] direct serve: launching Corsaire on GPUs $GPU_DEVICES port $PORT"
  echo "model=$MODEL"
  echo "config=$CONFIG"
  echo "repo_root=$REPO_ROOT"
  echo "trtllm_serve=$TRTLLM_SERVE"
} | tee -a "$HB"

if [[ "$FOREGROUND" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
  exec "${CMD[@]}"
fi

(
  export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
  exec "${CMD[@]}"
) > "$LOGD/run.log" 2>&1 &

PID="$!"
echo "$PID" > "$PID_FILE"
echo "[$(date -u +%H:%M:%SZ)] direct serve started pid=$PID log=$LOGD/run.log" | tee -a "$HB"

#!/bin/bash
# Aggregated TP4 benchmark serve path for the Corsaire target model.
# This measures the target model as-is, without an SMC/speculative draft model.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

IMG="${IMG:-local/dynamo-trtllm-optrt-custom:canonical-smc-r20-20260605}"
WT="${WT:-$REPO_ROOT}"
BENCH_ROOT="${BENCH_ROOT:-$HOME/wt/_hb}"
GPU_DEVICES="${GPU_DEVICES:-0,1,2,3}"
COPY_ID="${COPY_ID:-gpus_${GPU_DEVICES//,/}}"
SDIR="${SDIR:-$BENCH_ROOT/corsaire_bench_${COPY_ID}}"
LOGD="${LOGD:-$SDIR/logs}"
HB="${HB:-$BENCH_ROOT/corsaire-bench-${COPY_ID}.log}"
MODEL="${MODEL:-/models/BlaiseAI/corsaire-research-preview}"
PORT="${PORT:-8077}"
NAME="${NAME:-corsaire_bench_${COPY_ID}}"

mkdir -p "$LOGD" "$(dirname "$HB")"
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "[$(date -u +%H:%M:%SZ)] benchmark serve: launching $NAME on GPUs $GPU_DEVICES port $PORT" | tee -a "$HB"

docker run -d --name "$NAME" --gpus "\"device=${GPU_DEVICES}\"" \
  --ipc=host --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 \
  -p ${PORT}:8000 \
  -e CUDA_VISIBLE_DEVICES="$GPU_DEVICES" \
  -e HF_HOME=/models -e HF_HUB_OFFLINE=1 -e HF_MODULES_CACHE=/tmp/hf_modules \
  -e PYTHONUNBUFFERED=1 -e TLLM_LOG_LEVEL=INFO \
  -e TRTLLM_SERVER_DISABLE_GC=1 -e TRTLLM_WORKER_DISABLE_GC=1 -e TRTLLM_ENABLE_PDL=1 \
  -e NCCL_IB_DISABLE=1 -e NCCL_MNNVL_ENABLE=0 -e NCCL_CUMEM_ENABLE=1 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_GRAPH_MIXING_SUPPORT=0 -e NCCL_NET_PLUGIN=none \
  -e NCCL_DEBUG=WARN \
  -e NVIDIA_GDRCOPY=1 -e UCX_CUDA_IPC_ENABLE_MNNVL=0 \
  -e OMPI_MCA_pml=ob1 -e OMPI_MCA_btl=self,tcp,vader \
  -e UCX_TLS=tcp,self,sm,cuda_copy,cuda_ipc \
  -e OMPI_ALLOW_RUN_AS_ROOT=1 -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
  -e TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER=1 \
  -v /models:/models:ro \
  -v "$SDIR":/host_serve \
  -v "$WT/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/utils.py":/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/utils.py:ro \
  -v "$WT/tensorrt_llm/_torch/modules/attention.py":/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/_torch/modules/attention.py:ro \
  --entrypoint bash "$IMG" -c "
    set -uo pipefail
    echo '[serve] overlay utils.py:' \$(md5sum /opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/utils.py | cut -d' ' -f1)
    echo '[serve] starting trtllm-serve TP4 aggregated benchmark (model load ~8-12min)'
    exec timeout 2400 trtllm-serve serve '$MODEL' --host 0.0.0.0 --port 8000 --backend pytorch \
      --tp_size 4 --max_batch_size 64 --max_num_tokens 2048 --max_seq_len 132096 \
      --trust_remote_code --extra_llm_api_options /host_serve/smc_agg_tp4.yaml
  " > "$LOGD/run.log" 2>&1

CID=$(docker ps -q -f name=$NAME)
echo "[$(date -u +%H:%M:%SZ)] r20 e2e serve container started cid=$CID" | tee -a "$HB"
echo "$CID" > "$SDIR/serve.cid"

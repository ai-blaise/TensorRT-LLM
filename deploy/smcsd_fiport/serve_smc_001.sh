#!/bin/bash
# r20 SMC-ON e2e serve gate (a4-us-001 docker). REAP-345B NVFP4 target +
# GLM-4-9B-0414-FP8 SMC draft. Uses the canonical-smc-r20 image with the r20
# worktree tensorrt_llm overlaid (python-only, no rebuild). GPUs 0-3.
set -uo pipefail
IMG=local/dynamo-trtllm-optrt-custom:canonical-smc-r20-20260605
WT=/home/spencer/wt/smcsd-complete-r20
SDIR=/home/spencer/wt/_hb/r20serve
LOGD=$SDIR/logs
HB=/home/spencer/wt/_hb/smcsd-complete-r20.log
MODEL=/models/BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft
PORT=8077
NAME=r20_smc_serve

mkdir -p "$LOGD"
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "[$(date -u +%H:%M:%SZ)] r20 e2e serve: launching $NAME on GPUs 0-3 port $PORT" | tee -a "$HB"

docker run -d --name "$NAME" --gpus '"device=0,1,2,3"' \
  --ipc=host --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 \
  -p ${PORT}:8000 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e HF_HOME=/models -e HF_HUB_OFFLINE=1 -e HF_MODULES_CACHE=/tmp/hf_modules \
  -e PYTHONUNBUFFERED=1 -e TLLM_LOG_LEVEL=INFO \
  -e SMC_REJECTION_ACCEPT=1 \
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
  -v "$WT/tensorrt_llm/_torch/speculative":/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/_torch/speculative:ro \
  -v "$WT/tensorrt_llm/_torch/models/modeling_glm.py":/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/_torch/models/modeling_glm.py:ro \
  -v "$WT/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/utils.py":/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/utils.py:ro \
  -v "$WT/tensorrt_llm/_torch/modules/attention.py":/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/_torch/modules/attention.py:ro \
  --entrypoint bash "$IMG" -c "
    set -uo pipefail
    echo '[serve] overlay smc.py:' \$(md5sum /opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/_torch/speculative/smc.py | cut -d' ' -f1)
    echo '[serve] overlay utils.py:' \$(md5sum /opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/utils.py | cut -d' ' -f1)
    echo '[serve] starting trtllm-serve TP4 agg SMC-ON (model load ~8-12min)'
    exec timeout 2400 trtllm-serve serve '$MODEL' --host 0.0.0.0 --port 8000 --backend pytorch \
      --tp_size 4 --max_batch_size 64 --max_num_tokens 2048 --max_seq_len 132096 \
      --trust_remote_code --extra_llm_api_options /host_serve/smc_agg_tp4.yaml
  " > "$LOGD/run.log" 2>&1

CID=$(docker ps -q -f name=$NAME)
echo "[$(date -u +%H:%M:%SZ)] r20 e2e serve container started cid=$CID" | tee -a "$HB"
echo "$CID" > "$SDIR/serve.cid"

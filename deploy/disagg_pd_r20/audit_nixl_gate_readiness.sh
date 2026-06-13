#!/usr/bin/env bash
set -euo pipefail

DGD="${DGD:-topo-c1-dp2tp4-disagg-r20}"
KC="${KC:-sudo -E /usr/local/bin/k3s kubectl -n dynamo-system}"
MODE="${NIXL_AUDIT_MODE:-live}"
MIN_MAX_TOKENS_IN_BUFFER="${MIN_MAX_TOKENS_IN_BUFFER:-131072}"
CHECK_RUNTIME_LIBS="${CHECK_RUNTIME_LIBS:-1}"
EXPECTED_NIXL_PLUGIN_BACKEND="${EXPECTED_NIXL_PLUGIN_BACKEND:-UCX}"
SMC_GATE_MODE="${SMC_GATE_MODE:-required}"
OUTPUT_DIR="${NIXL_AUDIT_OUT:-/tmp/nixl_gate_audit_${MODE}_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
LOCAL_DGD_MANIFEST="${LOCAL_DGD_MANIFEST:-deploy/disagg_pd_r20/topo-c1-dp2tp4-disagg-r20.yaml}"

case "$EXPECTED_NIXL_PLUGIN_BACKEND" in
  UCX|LIBFABRIC) ;;
  *) echo "nixl gate audit failed: EXPECTED_NIXL_PLUGIN_BACKEND must be UCX or LIBFABRIC, got $EXPECTED_NIXL_PLUGIN_BACKEND" >&2; exit 2 ;;
esac

mkdir -p "$OUTPUT_DIR"

fail() {
  echo "nixl gate audit failed: $*" >&2
  echo "artifact_dir=$OUTPUT_DIR" >&2
  exit 1
}

note() {
  echo "[nixl-audit] $*"
}

pod_for() {
  local component="$1"
  $KC get pods -o name | grep "${DGD}-0-${component}" | tail -1
}

count_fixed() {
  local needle="$1" file="$2"
  grep -F -c -- "$needle" "$file" || true
}

require_fixed() {
  local needle="$1" file="$2" label="$3"
  grep -F -q -- "$needle" "$file" || fail "$label missing: $needle"
}

reject_fixed() {
  local needle="$1" file="$2" label="$3"
  ! grep -F -q -- "$needle" "$file" || fail "$label forbidden: $needle"
}

require_regex() {
  local pattern="$1" file="$2" label="$3"
  grep -E -q -- "$pattern" "$file" || fail "$label missing regex: $pattern"
}

reject_regex() {
  local pattern="$1" file="$2" label="$3"
  ! grep -E -q -- "$pattern" "$file" || fail "$label forbidden regex: $pattern"
}

collect_local() {
  [[ -f "$LOCAL_DGD_MANIFEST" ]] || fail "local DGD manifest not found: $LOCAL_DGD_MANIFEST"
  cat "$LOCAL_DGD_MANIFEST" \
      deploy/disagg_pd_r20/prefill.yaml \
      deploy/disagg_pd_r20/decode.yaml >"$OUTPUT_DIR/config_combined.yaml"
  cp "$LOCAL_DGD_MANIFEST" "$OUTPUT_DIR/dgd.yaml"
}

collect_live() {
  local ready
  ready="$($KC get dgd "$DGD" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || true)"
  [[ "$ready" == "True" ]] || fail "$DGD is not Ready (Ready=${ready:-missing})"

  $KC get dgd "$DGD" -o yaml >"$OUTPUT_DIR/dgd.yaml"
  $KC get cm "${DGD}-config" -o yaml >"$OUTPUT_DIR/configmap.yaml"
  cat "$OUTPUT_DIR/dgd.yaml" "$OUTPUT_DIR/configmap.yaml" >"$OUTPUT_DIR/config_combined.yaml"
  $KC get pods -o wide | grep "$DGD" >"$OUTPUT_DIR/pods.txt" || true

  for component in frontend prefill decode; do
    local pod
    pod="$(pod_for "$component")"
    [[ -n "$pod" ]] || fail "could not resolve $component pod"
    echo "$pod" >"$OUTPUT_DIR/${component}_pod.txt"
    $KC get "$pod" -o yaml >"$OUTPUT_DIR/${component}_pod.yaml" || true
    $KC logs "$pod" >"$OUTPUT_DIR/${component}.log" 2>&1 || true
    $KC get "$pod" -o jsonpath='{range .spec.containers[*].env[*]}{.name}{"="}{.value}{"\n"}{end}' >"$OUTPUT_DIR/${component}.env" 2>/dev/null || true
  done
}

require_config_shape() {
  local cfg="$OUTPUT_DIR/config_combined.yaml"

  [[ "$(count_fixed 'backend: NIXL' "$cfg")" -ge 2 ]] || fail "prefill/decode cache_transceiver_config.backend are not both NIXL"
  [[ "$(count_fixed 'transceiver_runtime: PYTHON' "$cfg")" -ge 2 ]] \
    || fail "prefill/decode NIXL transceiver runtime must be PYTHON for generation-first/write-mode handoff"
  reject_fixed 'backend: UCX' "$cfg" "direct UCX backend in NIXL gate"
  reject_fixed 'backend: MOONCAKE' "$cfg" "Mooncake backend in NIXL gate"
  require_fixed 'layersplit_transfer_backend: nixl' "$cfg" "LayerSplit NIXL transfer"
  reject_fixed 'layersplit_transfer_backend: ucx' "$cfg" "LayerSplit UCX fallback"
  require_fixed 'cp_type: LAYERSPLIT' "$cfg" "LayerSplit CP type"
  reject_fixed 'cp_type: HELIX' "$cfg" "HELIX fallback"
  require_fixed 'layersplit_enabled: true' "$cfg" "LayerSplit enabled"
  require_fixed 'layersplit_owner_local_alloc: true' "$cfg" "LayerSplit owner-local allocation"
  require_fixed 'layersplit_all_cp_ranks_transfer: true' "$cfg" "LayerSplit all-rank transfer"
  require_fixed 'mla_latent_kv_dtype: kvarn_k2v2' "$cfg" "dense MLA KVarN k2v2"
  require_fixed 'mla_latent_kv_amortize: true' "$cfg" "dense MLA KVarN amortize"
  require_fixed 'indexer_k_dtype: fp4' "$cfg" "Indexer FP4"
  reject_fixed 'indexer_k_dtype: kvarn' "$cfg" "Indexer KVarN misuse"
  require_fixed 'backend: WARPDECODE' "$cfg" "WarpDecode backend"
  require_fixed 'allow_parallelism_fallback: false' "$cfg" "WarpDecode fail-closed fallback"
  require_fixed 'disable_overlap_scheduler: false' "$cfg" "Moondream overlap scheduler"

  [[ "$(count_fixed 'max_tokens_in_buffer: 131072' "$cfg")" -ge 2 ]] \
    || fail "max_tokens_in_buffer must be at least ${MIN_MAX_TOKENS_IN_BUFFER} for 128k NIXL transfer buffer coverage"
  if [[ "$MODE" == "live" ]]; then
    require_fixed "TRTLLM_NIXL_KVCACHE_BACKEND=${EXPECTED_NIXL_PLUGIN_BACKEND}" "$OUTPUT_DIR/prefill.env" "prefill NIXL plugin backend env"
    require_fixed "TRTLLM_NIXL_KVCACHE_BACKEND=${EXPECTED_NIXL_PLUGIN_BACKEND}" "$OUTPUT_DIR/decode.env" "decode NIXL plugin backend env"
    require_fixed 'TRTLLM_NIXL_ENABLE_COALESCE=1' "$OUTPUT_DIR/prefill.env" "prefill NIXL descriptor coalescing env"
    require_fixed 'TRTLLM_NIXL_ENABLE_COALESCE=1' "$OUTPUT_DIR/decode.env" "decode NIXL descriptor coalescing env"
    require_fixed 'TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=0' "$OUTPUT_DIR/prefill.env" "prefill NIXL transfer overlap env"
    require_fixed 'TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=0' "$OUTPUT_DIR/decode.env" "decode NIXL transfer overlap env"
    require_fixed 'TRTLLM_ENABLE_KVCACHE_RECEIVE_PARALLEL=1' "$OUTPUT_DIR/prefill.env" "prefill NIXL parallel receive env"
    require_fixed 'TRTLLM_ENABLE_KVCACHE_RECEIVE_PARALLEL=1' "$OUTPUT_DIR/decode.env" "decode NIXL parallel receive env"
  else
    [[ "$(count_fixed 'TRTLLM_NIXL_KVCACHE_BACKEND' "$cfg")" -ge 2 ]] \
      || fail "TRTLLM_NIXL_KVCACHE_BACKEND must be explicit on prefill and decode"
    [[ "$(count_fixed 'TRTLLM_NIXL_ENABLE_COALESCE' "$cfg")" -ge 2 ]] \
      || fail "TRTLLM_NIXL_ENABLE_COALESCE must be explicit on prefill and decode"
    [[ "$(count_fixed 'TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP' "$cfg")" -ge 2 ]] \
      || fail "TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP must be explicit on prefill and decode"
    [[ "$(count_fixed 'TRTLLM_ENABLE_KVCACHE_RECEIVE_PARALLEL' "$cfg")" -ge 2 ]] \
      || fail "TRTLLM_ENABLE_KVCACHE_RECEIVE_PARALLEL must be explicit on prefill and decode"
    require_fixed "value: ${EXPECTED_NIXL_PLUGIN_BACKEND}" "$cfg" "NIXL plugin backend value"
    require_fixed "value: '1'" "$cfg" "enabled boolean env values"
    require_fixed "value: '0'" "$cfg" "disabled boolean env values"
  fi
  require_fixed 'UCX_CUDA_IPC_ENABLE_MNNVL' "$cfg" "UCX CUDA IPC MNNVL guard"
  require_fixed 'NVIDIA_GDRCOPY' "$cfg" "GDRCopy env"
  require_fixed 'NCCL_NET_PLUGIN' "$cfg" "NCCL net plugin guard"
  require_fixed 'TRTLLM_FORCE_COMM_METHOD' "$cfg" "explicit MoE comm method"
  require_fixed 'NVLINK_TWO_SIDED' "$cfg" "prefill NVLink two-sided comm method"
  require_fixed 'DEEPEPLOWLATENCY' "$cfg" "decode DeepEP low-latency comm method"
  require_fixed 'TRTLLM_DEEP_EP_TOKEN_LIMIT' "$cfg" "decode DeepEP token limit"
  require_fixed "value: '64'" "$cfg" "decode DeepEP token limit value"
  require_fixed 'TRTLLM_DEEP_EP_DISABLE_P2P_FOR_LOW_LATENCY_MODE' "$cfg" "decode DeepEP P2P mode"
  require_fixed 'TRTLLM_MOE_POST_QUANT_ALLTOALLV' "$cfg" "decode post-quant alltoallv"

  case "$SMC_GATE_MODE" in
    deferred)
      reject_fixed 'decoding_type: SMC' "$cfg" "SMC-SD in deferred NIXL gate"
      reject_fixed 'speculative_model:' "$cfg" "speculative model in deferred NIXL gate"
      ;;
    required)
      require_fixed 'decoding_type: SMC' "$cfg" "SMC-SD required gate"
      require_fixed 'speculative_model: /models/BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP' "$cfg" "SMC-SD GLM draft"
      require_fixed 'draft_attention_backend: triton' "$cfg" "SMC-SD draft attention backend"
      require_fixed 'draft_kv_cache_dtype: bfloat16' "$cfg" "SMC-SD bf16 draft KV"
      reject_fixed 'draft_kv_cache_dtype: kvarn' "$cfg" "GQA KVarN draft KV before readiness promotion"
      require_fixed 'use_low_precision_moe_combine: false' "$cfg" "DeepEP low-latency combine gate"
      ;;
    *) fail "SMC_GATE_MODE must be deferred or required, got $SMC_GATE_MODE" ;;
  esac
}

require_live_runtime() {
  local pre dec pod logs all_logs
  pre="$(cat "$OUTPUT_DIR/prefill_pod.txt")"
  dec="$(cat "$OUTPUT_DIR/decode_pod.txt")"
  all_logs="$OUTPUT_DIR/all_worker_logs.txt"
  cat "$OUTPUT_DIR/prefill.log" "$OUTPUT_DIR/decode.log" >"$all_logs"

  for component in prefill decode; do
    reject_regex '^TRTLLM_USE_(UCX|MOONCAKE|MPI)_KVCACHE=1$' "$OUTPUT_DIR/${component}.env" "legacy backend env override for $component"
  done

  require_fixed 'Initializing NIXL Connect' "$all_logs" "NIXL Connect startup"
  require_regex "cache_transceiver_config.*backend.*NIXL|cache_transceiver_config: \{'backend': 'NIXL'" "$all_logs" "runtime NIXL backend log"
  reject_regex "cache_transceiver_config.*backend.*UCX|cache_transceiver_config: \{'backend': 'UCX'|Using UCX kv-cache transceiver" "$all_logs" "direct UCX runtime fallback"
  require_regex 'dynamo disagg request pin established.*handoff_mode="?generation_first"?' "$OUTPUT_DIR/frontend.log" "generation-first request pin marker"
  require_regex 'dynamo disagg request pin outbound to decode.*handoff_mode="?generation_first"?' "$OUTPUT_DIR/frontend.log" "generation-first outbound marker"
  reject_regex 'handoff_mode="?completed_prefill"?' "$OUTPUT_DIR/frontend.log" "completed-prefill handoff in NIXL write-mode gate"
  require_fixed 'OPTRT_LAYERSPLIT_XFER_DEBUG' "$all_logs" "LayerSplit transfer debug marker"
  require_fixed 'global_layers=61' "$all_logs" "global 61-layer transfer metadata"
  require_fixed 'transfer_attr=True' "$OUTPUT_DIR/prefill.log" "prefill DSACacheManager transfer metadata"

  reject_regex 'KV cache transfer timeout|MLACacheFormatter::inquireSupport|CacheTransferLayer::validateSupport|only support same number of layers|illegal memory access|Traceback|NIXL.*(failed|failure|error)|(failed|failure|error).*NIXL' "$all_logs" "known transport failure marker"

  if [[ "$CHECK_RUNTIME_LIBS" == "1" ]]; then
    for pod in "$pre" "$dec"; do
      note "checking NIXL runtime libs in $pod"
      $KC exec "$pod" -- sh -lc 'py=$(command -v python3 || command -v python); "$py" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec(\"nixl\") and importlib.util.find_spec(\"msgpack\") else 1)" && "$py" -c "import tensorrt_llm._torch.disaggregation.native.transfer" && find /usr /opt -name "libtensorrt_llm_nixl_wrapper.so*" -print -quit | grep -q .' \
        >"$OUTPUT_DIR/$(basename "$pod")_nixl_runtime.txt" 2>&1 \
        || fail "$pod missing Python nixl/msgpack, native NIXL transfer import, or libtensorrt_llm_nixl_wrapper.so; see $OUTPUT_DIR/$(basename "$pod")_nixl_runtime.txt"
    done
  fi
}

case "$MODE" in
  local) collect_local ;;
  live) collect_live ;;
  *) fail "NIXL_AUDIT_MODE must be local or live, got $MODE" ;;
esac

require_config_shape
if [[ "$MODE" == "live" ]]; then
  require_live_runtime
fi

cat >"$OUTPUT_DIR/summary.txt" <<EOF_SUMMARY
mode=$MODE
dgd=$DGD
min_max_tokens_in_buffer=$MIN_MAX_TOKENS_IN_BUFFER
check_runtime_libs=$CHECK_RUNTIME_LIBS
expected_nixl_plugin_backend=$EXPECTED_NIXL_PLUGIN_BACKEND
local_dgd_manifest=$LOCAL_DGD_MANIFEST
smc_gate_mode=$SMC_GATE_MODE
status=pass
EOF_SUMMARY

note "NIXL gate audit passed; artifacts: $OUTPUT_DIR"

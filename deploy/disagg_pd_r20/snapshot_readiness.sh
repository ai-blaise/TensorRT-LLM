#!/usr/bin/env bash
set -euo pipefail

VM_HOST="${VM_HOST:-34.106.33.128}"
VM_USER="${VM_USER:-spencergarnets}"
DGD_NAMESPACE="${DGD_NAMESPACE:-dynamo-system}"
DGD_NAME="${DGD_NAME:-topo-c1-dp2tp4-disagg-r20}"
SNAPSHOT_NAMESPACE="${SNAPSHOT_NAMESPACE:-criu-snapshots}"
TARGET_NODE="${TARGET_NODE:-a4-us-001-rl9}"
RUN_HOST_PREFLIGHT="${RUN_HOST_PREFLIGHT:-0}"
STRICT="${STRICT:-0}"
SSH_OPTS=(
  -o BatchMode=yes
  -o IdentitiesOnly=yes
  -o UserKnownHostsFile=/Users/spencer/.ssh/google_compute_known_hosts
  -o CheckHostIP=no
)

usage() {
  cat <<'EOF'
Usage: deploy/disagg_pd_r20/snapshot_readiness.sh [options]

Print a read-only CRIU snapshot composition report for the r20 DGD. This does
not create DynamoGraphDeploymentSnapshot resources, patch pods, drain traffic,
or restart workloads. It is a preflight for deciding whether the existing
ai-blaise/criu-snapshots path can be composed with op-trt.

Options:
  --vm HOST                 Target VM IP or hostname (default: $VM_HOST);
                            use local to run directly on the current VM
  --user USER               SSH user (default: $VM_USER)
  --dgd-namespace NS        DGD namespace (default: $DGD_NAMESPACE)
  --dgd-name NAME           DGD name (default: $DGD_NAME)
  --snapshot-namespace NS   criu-snapshots namespace (default: $SNAPSHOT_NAMESPACE)
  --target-node NAME        Expected Kubernetes node name (default: $TARGET_NODE)
  --run-host-preflight      Run /opt/criu-snapshots/bin/snapshot-preflight if present
  --strict                  Exit nonzero when a blocking prerequisite is missing
  -h, --help                Show this help

Environment overrides use the same names as the options.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vm) VM_HOST="$2"; shift 2 ;;
    --user) VM_USER="$2"; shift 2 ;;
    --dgd-namespace) DGD_NAMESPACE="$2"; shift 2 ;;
    --dgd-name) DGD_NAME="$2"; shift 2 ;;
    --snapshot-namespace) SNAPSHOT_NAMESPACE="$2"; shift 2 ;;
    --target-node) TARGET_NODE="$2"; shift 2 ;;
    --run-host-preflight) RUN_HOST_PREFLIGHT=1; shift ;;
    --strict) STRICT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

SSH_TARGET="${VM_USER}@${VM_HOST}"

read -r -d '' REMOTE_SCRIPT <<'EOS' || true
set -euo pipefail
export PATH="/opt/criu-snapshots/bin:/opt/criu-snapshots/libexec:$PATH"

kv() { printf '%s=%s\n' "$1" "$2"; }
one_line() { tr '\n' ' ' | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//'; }
count_lines() { awk 'NF {n += 1} END {print n + 0}'; }
have() { command -v "$1" >/dev/null 2>&1; }

KUBECTL=""
if have kubectl; then
  KUBECTL=kubectl
elif [[ -x /usr/local/bin/k3s ]]; then
  KUBECTL="sudo /usr/local/bin/k3s kubectl"
fi

blockers=()
warns=()

kv report snapshot_readiness
kv mode read_only
kv composition_path criu_snapshots_sidecar
kv proof_criteria_doc docs/blaise/r20_snapshot_proof_criteria.md
kv dgd_namespace "$DGD_NAMESPACE"
kv dgd_name "$DGD_NAME"
kv snapshot_namespace "$SNAPSHOT_NAMESPACE"
kv target_node "$TARGET_NODE"
kv live_workload_modified 0
kv snapshot_resource_created 0

kernel="$(uname -r 2>/dev/null || echo missing)"
kv kernel "$kernel"
case "$kernel" in
  6.[2-9]*|6.[1-9][0-9]*|[7-9].*) kv kernel_checkpoint_floor ok ;;
  5.14.0-*el9*_ciq*) kv kernel_checkpoint_floor ok_backport ;;
  *) kv kernel_checkpoint_floor needs_review; warns+=(kernel_checkpoint_floor) ;;
esac

if have nvidia-smi; then
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
  kv nvidia_driver "${driver:-missing}"
  driver_major="${driver%%.*}"
  if [[ -n "${driver:-}" && "$driver_major" =~ ^[0-9]+$ && "$driver_major" -ge 580 ]]; then
    kv nvidia_driver_floor ok_r580_or_newer
  elif [[ -n "${driver:-}" && "$driver_major" =~ ^[0-9]+$ && "$driver_major" -ge 555 ]]; then
    kv nvidia_driver_floor single_gpu_floor_only
    warns+=(nvidia_driver_below_r580)
  else
    kv nvidia_driver_floor missing_or_too_old
    blockers+=(nvidia_driver)
  fi
else
  kv nvidia_driver missing
  blockers+=(nvidia_smi)
fi

if have criu; then
  criu_version="$(criu --version 2>/dev/null | head -1 | awk '{print $2}' || true)"
  kv criu_status present
  kv criu_version "${criu_version:-unknown}"
else
  kv criu_status missing
  blockers+=(criu)
fi

if have cuda-checkpoint; then
  kv cuda_checkpoint_status present
else
  kv cuda_checkpoint_status missing
  blockers+=(cuda_checkpoint)
fi

for tool in checkpointctl buildah oras; do
  if have "$tool"; then
    kv "${tool}_status" present
  else
    kv "${tool}_status" missing
    warns+=("${tool}_missing")
  fi
done
if have checkpointctl; then
  kv checkpointctl_restore_image_tooling ready
else
  kv checkpointctl_restore_image_tooling missing_for_restore_image_materialization
fi

if [[ -d /opt/criu-snapshots ]]; then
  kv host_install_dir present
else
  kv host_install_dir missing
  blockers+=(host_install_dir)
fi
if [[ -x /opt/criu-snapshots/bin/snapshot-preflight ]]; then
  kv host_preflight_binary present
  if [[ "$RUN_HOST_PREFLIGHT" == 1 ]]; then
    if sudo /opt/criu-snapshots/bin/snapshot-preflight >/tmp/optrt-snapshot-preflight.out 2>/tmp/optrt-snapshot-preflight.err; then
      kv host_preflight_result ok
    else
      kv host_preflight_result failed
      kv host_preflight_error "$(tail -20 /tmp/optrt-snapshot-preflight.err 2>/dev/null | one_line)"
      blockers+=(host_preflight)
    fi
  else
    kv host_preflight_result not_run
  fi
else
  kv host_preflight_binary missing
  blockers+=(host_preflight_binary)
fi

if [[ -z "$KUBECTL" ]]; then
  kv kubectl_status missing
  blockers+=(kubectl)
else
  kv kubectl_status present
  if $KUBECTL get crd dynamographdeploymentsnapshots.snapshots.ai-blaise.io >/dev/null 2>&1; then
    kv dgds_crd present
  else
    kv dgds_crd missing
    blockers+=(dgds_crd)
  fi
  if $KUBECTL get ns "$SNAPSHOT_NAMESPACE" >/dev/null 2>&1; then
    kv snapshot_namespace_status present
  else
    kv snapshot_namespace_status missing
    blockers+=(snapshot_namespace)
  fi
  if $KUBECTL get ns "$DGD_NAMESPACE" >/dev/null 2>&1; then
    kv dgd_namespace_status present
  else
    kv dgd_namespace_status missing
    blockers+=(dgd_namespace)
  fi

  controller_refs="$($KUBECTL -n "$SNAPSHOT_NAMESPACE" get deploy,sts --ignore-not-found -o name 2>/dev/null | grep -Ei 'criu|snapshot|controller' || true)"
  daemon_refs="$($KUBECTL -n "$SNAPSHOT_NAMESPACE" get ds --ignore-not-found -o name 2>/dev/null | grep -Ei 'criu|snapshot|daemon|agent' || true)"
  kv snapshot_controller_refs "$(printf '%s\n' "$controller_refs" | paste -sd, -)"
  kv snapshot_daemon_refs "$(printf '%s\n' "$daemon_refs" | paste -sd, -)"
  [[ -n "$controller_refs" ]] || blockers+=(snapshot_controller)
  [[ -n "$daemon_refs" ]] || blockers+=(snapshot_daemon)

  snapshot_pods="$($KUBECTL -n "$SNAPSHOT_NAMESPACE" get pods --ignore-not-found --no-headers 2>/dev/null || true)"
  kv snapshot_pod_count "$(printf '%s\n' "$snapshot_pods" | count_lines)"
  not_ready_snapshot_pods="$(printf '%s\n' "$snapshot_pods" | awk 'NF && $3 !~ /^(Running|Completed)$/ {print $1":"$3}' | paste -sd, -)"
  kv snapshot_not_ready_pods "$not_ready_snapshot_pods"
  [[ -z "$not_ready_snapshot_pods" ]] || warns+=(snapshot_pods_not_ready)

  dgd_status=missing
  dgd_yaml=""
  if dgd_yaml="$($KUBECTL -n "$DGD_NAMESPACE" get dynamographdeployment "$DGD_NAME" -o yaml 2>/dev/null)"; then
    dgd_status=present
  elif dgd_yaml="$($KUBECTL -n "$DGD_NAMESPACE" get dgd "$DGD_NAME" -o yaml 2>/dev/null)"; then
    dgd_status=present
  fi
  kv target_dgd "$dgd_status"
  [[ "$dgd_status" == present ]] || blockers+=(target_dgd)

  cm_name="${DGD_NAME}-config"
  config_yaml=""
  if config_yaml="$($KUBECTL -n "$DGD_NAMESPACE" get cm "$cm_name" -o yaml 2>/dev/null)"; then
    kv target_configmap present
  else
    kv target_configmap missing
    blockers+=(target_configmap)
  fi
  target_yaml="${dgd_yaml}
${config_yaml}"
  if grep -Eq 'backendFramework:[[:space:]]*trtllm|dynamo\.trtllm' <<<"$target_yaml"; then
    kv runtime_engine trtllm
  else
    kv runtime_engine unknown
    warns+=(runtime_engine_unknown)
  fi
  for marker in SGLANG_SNAPSHOT_HOOKS OPTRT_SNAPSHOT_HOOKS TRTLLM_SNAPSHOT_HOOKS; do
    if grep -q "$marker" <<<"$target_yaml"; then
      kv "${marker}_configured" 1
    else
      kv "${marker}_configured" 0
    fi
  done
  if grep -Eiq 'foundry|LD_PRELOAD|libcuda_hook|graph_extension' <<<"$target_yaml"; then
    kv foundry_live_markers present
    warns+=(foundry_live_markers)
  else
    kv foundry_live_markers absent
  fi
  if grep -q '/cache/optrt' <<<"$target_yaml" && grep -q '/var/lib/optrt-cache' <<<"$target_yaml"; then
    kv optrt_cache_mount configured
  else
    kv optrt_cache_mount missing
    blockers+=(optrt_cache_mount)
  fi
  if grep -Eq 'backend:[[:space:]]*NIXL|cache_transceiver_config.*NIXL|layersplit_transfer_backend:[[:space:]]*nixl' <<<"$target_yaml"; then
    kv nixl_backend_configured 1
  else
    kv nixl_backend_configured 0
    warns+=(nixl_backend_not_detected)
  fi
  if grep -q 'layersplit_enabled: true' <<<"$target_yaml"       && grep -q 'tensor_parallel_size: 2' <<<"$target_yaml"       && grep -q 'context_parallel_size: 2' <<<"$target_yaml"; then
    kv layersplit_tp2cp2_configured 1
  else
    kv layersplit_tp2cp2_configured 0
    warns+=(layersplit_tp2cp2_not_detected)
  fi
  if grep -Eq 'mla_latent_kv_dtype:[[:space:]]*kvarn_k2v2|mla_latent_kv_dtype.*kvarn' <<<"$target_yaml"; then
    kv dense_kvarn_configured 1
  else
    kv dense_kvarn_configured 0
    warns+=(dense_kvarn_not_detected)
  fi

  pod_lines="$($KUBECTL -n "$DGD_NAMESPACE" get pods --ignore-not-found -o wide --no-headers 2>/dev/null | grep -F "$DGD_NAME" || true)"
  kv target_pod_count "$(printf '%s\n' "$pod_lines" | count_lines)"
  kv target_pods "$(printf '%s\n' "$pod_lines" | awk '{print $1":"$3}' | paste -sd, -)"
  if [[ -n "$pod_lines" ]]; then
    target_node_mismatch="$(printf '%s\n' "$pod_lines" | awk -v node="$TARGET_NODE" 'NF && $7 != node {print $1":"$7}' | paste -sd, -)"
    kv target_node_mismatch "$target_node_mismatch"
    [[ -z "$target_node_mismatch" ]] || warns+=(target_node_mismatch)
  fi

  existing_dgds="$($KUBECTL -n "$SNAPSHOT_NAMESPACE" get dgds --ignore-not-found -o name 2>/dev/null | head -20 | paste -sd, - || true)"
  kv existing_dgds "$existing_dgds"
fi

if [[ -d /var/lib/optrt-cache ]]; then
  kv optrt_cache_hostpath present
else
  kv optrt_cache_hostpath missing
  blockers+=(optrt_cache_hostpath)
fi

kv trtllm_snapshot_hook_status missing
kv trtllm_snapshot_hook_proof missing
kv nixl_inflight_restore_proof missing
kv layersplit_tp2cp2_restore_proof missing
kv kvarn_cuda_graph_scratch_restore_proof missing
kv checkpointctl_restore_proof missing
blockers+=(trtllm_snapshot_hook_proof)
blockers+=(nixl_inflight_restore_proof)
blockers+=(layersplit_tp2cp2_restore_proof)
blockers+=(kvarn_cuda_graph_scratch_restore_proof)
blockers+=(checkpointctl_restore_proof)
kv safe_to_take_snapshot 0
kv recommended_next_action add_gated_trtllm_pre_snapshot_post_restore_hooks_then_run_dgds_canary

if ((${#warns[@]})); then
  kv warnings "$(IFS=,; echo "${warns[*]}")"
else
  kv warnings none
fi
if ((${#blockers[@]})); then
  kv blockers "$(IFS=,; echo "${blockers[*]}")"
else
  kv blockers none
fi

if [[ "$STRICT" == 1 && ${#blockers[@]} -gt 0 ]]; then
  exit 1
fi
EOS

if [[ "$VM_HOST" == "local" ]]; then
  DGD_NAMESPACE="$DGD_NAMESPACE" DGD_NAME="$DGD_NAME" \
    SNAPSHOT_NAMESPACE="$SNAPSHOT_NAMESPACE" TARGET_NODE="$TARGET_NODE" \
    RUN_HOST_PREFLIGHT="$RUN_HOST_PREFLIGHT" STRICT="$STRICT" bash -s \
    <<<"$REMOTE_SCRIPT"
else
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
    "DGD_NAMESPACE='$DGD_NAMESPACE' DGD_NAME='$DGD_NAME' SNAPSHOT_NAMESPACE='$SNAPSHOT_NAMESPACE' TARGET_NODE='$TARGET_NODE' RUN_HOST_PREFLIGHT='$RUN_HOST_PREFLIGHT' STRICT='$STRICT' bash -s" \
    <<<"$REMOTE_SCRIPT"
fi

import re
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_DIR = REPO_ROOT / "deploy" / "disagg_pd_r20"


def _load_yaml(name: str):
    return yaml.safe_load((DEPLOY_DIR / name).read_text())


def test_r20_prefill_canary_gates():
    cfg = _load_yaml("prefill.yaml")

    assert cfg["disable_overlap_scheduler"] is False
    assert cfg["cp_config"]["cp_type"] == "LAYERSPLIT"
    assert cfg["sparse_attention_config"]["layersplit_enabled"] is True
    assert cfg["sparse_attention_config"]["layersplit_all_cp_ranks_transfer"] is True
    assert cfg["sparse_attention_config"]["layersplit_owner_local_alloc"] is True
    assert cfg["sparse_attention_config"]["layersplit_transfer_backend"] == "nixl"
    assert cfg["sparse_attention_config"]["mla_latent_kv_dtype"] == "kvarn_k2v2"
    assert cfg["sparse_attention_config"]["mla_latent_kv_amortize"] is True
    assert cfg["cache_transceiver_config"]["backend"] == "NIXL"
    assert cfg["cache_transceiver_config"]["transceiver_runtime"] == "PYTHON"
    assert cfg["moe_config"]["warp_decode"]["enabled"] is False


def test_r20_decode_canary_gates():
    cfg = _load_yaml("decode.yaml")

    assert cfg["disable_overlap_scheduler"] is False
    assert cfg["enable_attention_dp"] is True
    assert cfg["context_parallel_size"] == 1
    assert cfg["cache_transceiver_config"]["backend"] == "NIXL"
    assert cfg["cache_transceiver_config"]["transceiver_runtime"] == "PYTHON"
    assert cfg["moe_config"]["backend"] == "WARPDECODE"
    assert cfg["moe_config"]["warp_decode"]["enabled"] is True
    assert cfg["moe_config"]["warp_decode"]["policy"] == "force"
    assert cfg["moe_config"]["warp_decode"]["allow_parallelism_fallback"] is False
    assert cfg["sparse_attention_config"]["layersplit_enabled"] is False
    assert cfg["sparse_attention_config"]["mla_latent_kv_dtype"] == "kvarn_k2v2"
    assert cfg["sparse_attention_config"]["mla_latent_kv_amortize"] is True
    assert "speculative_config" not in cfg


def test_r20_manifest_has_no_helix_or_smc_fallback():
    manifest = (DEPLOY_DIR / "topo-c1-dp2tp4-disagg-r20.yaml").read_text()
    assert "cp_type: HELIX" not in manifest
    assert "decoding_type: SMC" not in manifest
    assert "speculative_model:" not in manifest
    assert "draft_attention_backend:" not in manifest
    assert "GLM-4-9B" not in manifest
    assert "SMC_REJECTION_ACCEPT" not in manifest
    assert "TRTLLM_USE_PRELOADED_TRITON_SWAPAB_ODD_M" not in manifest
    assert "SMC_CUDA_SYNC_PROBE" not in manifest
    assert "CUDA_LAUNCH_BLOCKING" not in manifest
    assert "layersplit_enabled: true" in manifest
    assert "layersplit_transfer_backend: nixl" in manifest
    assert "layersplit_owner_local_alloc: true" in manifest
    assert "backend: NIXL" in manifest
    assert manifest.count("transceiver_runtime: PYTHON") >= 2
    assert manifest.count("TRTLLM_NIXL_KVCACHE_BACKEND") >= 2
    assert manifest.count("value: LIBFABRIC") >= 2
    assert "value: UCX" not in manifest
    assert manifest.count("TRTLLM_NIXL_ENABLE_COALESCE") >= 2
    assert manifest.count("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP") >= 2
    assert manifest.count("TRTLLM_ENABLE_KVCACHE_RECEIVE_PARALLEL") >= 2
    assert "UCX_CUDA_IPC_ENABLE_MNNVL" in manifest
    assert "NVIDIA_GDRCOPY" in manifest
    assert "TRTLLM_FORCE_COMM_METHOD" in manifest
    assert "NVLINK_TWO_SIDED" in manifest
    assert "allow_parallelism_fallback: false" in manifest


def test_r20_overlay_carries_layersplit_and_request_pinning_sources():
    dockerfile = (DEPLOY_DIR / "Dockerfile.r20-overlay").read_text()

    assert "msgpack==1.1.1" in dockerfile
    assert "tensorrt_llm/_torch/pyexecutor/py_executor_creator.py" in dockerfile
    assert "tensorrt_llm/_torch/pyexecutor/kv_cache_transceiver.py" in dockerfile
    assert "tensorrt_llm/_torch/pyexecutor/snapshot_hooks.py" in dockerfile
    assert "tensorrt_llm/serve/openai_disagg_service.py" in dockerfile
    assert "tensorrt_llm/serve/openai_client.py" in dockerfile
    assert "tensorrt_llm/serve/openai_protocol.py" in dockerfile
    assert "tensorrt_llm/serve/openai_server.py" in dockerfile
    assert "tensorrt_llm/disaggregated_params.py" in dockerfile


def test_r20_overlay_carries_moondream_smc_overlap_sources():
    dockerfile = (DEPLOY_DIR / "Dockerfile.r20-overlay").read_text()

    for source in [
        "tensorrt_llm/_torch/pyexecutor/sampler.py",
        "tensorrt_llm/_torch/pyexecutor/guided_decoder.py",
        "tensorrt_llm/_torch/speculative/interface.py",
        "tensorrt_llm/_torch/speculative/model_drafter.py",
        "tensorrt_llm/_torch/speculative/smc.py",
        "tensorrt_llm/_torch/speculative/drafting_loops.py",
    ]:
        assert source in dockerfile


def test_r20_overlay_py_compile_block_is_docker_parseable():
    dockerfile = (DEPLOY_DIR / "Dockerfile.r20-overlay").read_text()

    assert "\\\\\n" not in dockerfile
    assert "RUN /opt/dynamo/venv/bin/python -m py_compile \\\n" in dockerfile


def test_r20_fullsource_build_path_exists_for_native_fixes():
    dockerfile = (DEPLOY_DIR / "Dockerfile.r20-fullsource").read_text()
    script = (DEPLOY_DIR / "build_fullsource_image.sh").read_text()
    readme = (DEPLOY_DIR / "README.md").read_text()

    assert "ARG BUILD_BASE" in dockerfile
    assert "ARG RUNTIME_BASE" in dockerfile
    assert "msgpack==1.1.1" in dockerfile
    assert "--configure-only" in dockerfile
    assert "--target tensorrt_llm th_common bindings" in dockerfile
    assert "BUILD_DEEP_EP=OFF" in dockerfile
    assert "ENABLE_NVSHMEM=OFF" in dockerfile
    assert "OPTRT_SOURCE_SHA" in dockerfile
    assert "ai.blaise.optrt.fullsource" in dockerfile
    assert "--build-base" in script
    assert "--runtime-base" in script
    assert "Dockerfile.r20-fullsource" in script
    assert "C++/CUDA/native-library changes" in readme


def test_r20_nixl_gate_requires_generation_first_write_mode():
    smoke = (DEPLOY_DIR / "smoke_request_pinning.sh").read_text()
    audit = (DEPLOY_DIR / "audit_nixl_gate_readiness.sh").read_text()

    assert "tensorrt_llm._torch.disaggregation.native.transfer" in smoke
    assert "importlib.util.find_spec(\\\"msgpack\\\")" in smoke
    assert "handoff_mode=\"?generation_first\"?" in audit
    assert "completed-prefill handoff in NIXL write-mode gate" in audit
    assert "NIXL write-mode gate forbids completed-prefill handoff markers" in smoke


def test_r20_snapshot_hooks_are_opt_in_and_readiness_checked():
    source = (
        REPO_ROOT / "tensorrt_llm" / "_torch" / "pyexecutor" / "snapshot_hooks.py"
    ).read_text()
    executor = (
        REPO_ROOT / "tensorrt_llm" / "_torch" / "pyexecutor" / "py_executor.py"
    ).read_text()
    readiness = (DEPLOY_DIR / "snapshot_readiness.sh").read_text()

    assert "OPTRT_SNAPSHOT_HOOKS" in source
    assert "SIGRTMIN" in source
    assert "pre_snapshot" in source
    assert "post_restore" in source
    assert "queue.active = False" in source
    assert "queue.active = True" in source
    assert "has_any_inflight_requests" in source
    assert "check_gen_transfer_complete" in source
    assert "SnapshotHookController.maybe_install" in executor
    assert "SNAPSHOT_HOOK_PROOF_DIR" in readiness
    assert "trtllm_snapshot_hook_status configured" in readiness
    assert "optrt_snapshot_*_pre_snapshot.ready.json" in readiness
    assert "optrt_snapshot_*_post_restore.ready.json" in readiness


def test_r20_render_snapshot_hooks_are_canary_only(tmp_path):
    script = DEPLOY_DIR / "render_dgd.sh"
    image = "localhost:5000/local/dynamo-trtllm-optrt-custom:test"
    default_out = tmp_path / "default.yaml"
    hook_out = tmp_path / "hook.yaml"
    hook_dir = "/tmp/optrt-snapshot-hooks-canary"

    subprocess.run(
        [
            str(script),
            "--image",
            image,
            "--dgd-name",
            "topo-c1-dp2tp4-disagg-r20",
            "--out",
            str(default_out),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    default_text = default_out.read_text()
    assert "OPTRT_SNAPSHOT_HOOKS" not in default_text
    assert "optrt-snapshot-hooks" not in default_text
    assert "${SNAPSHOT_HOOK_" not in default_text

    subprocess.run(
        [
            str(script),
            "--image",
            image,
            "--dgd-name",
            "topo-c1-dp2tp4-hook-canary",
            "--enable-snapshot-hooks",
            "--snapshot-hook-proof-dir",
            hook_dir,
            "--snapshot-hook-timeout-s",
            "60",
            "--out",
            str(hook_out),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    docs = list(yaml.safe_load_all(hook_out.read_text()))
    dgd = next(doc for doc in docs if doc and doc["kind"] == "DynamoGraphDeployment")
    services = dgd["spec"]["services"]

    for component in ["prefill", "decode"]:
        envs = {item["name"]: item["value"] for item in services[component]["envs"]}
        assert envs["OPTRT_SNAPSHOT_HOOKS"] == "1"
        assert envs["OPTRT_SNAPSHOT_HOOK_DIR"] == hook_dir
        assert envs["OPTRT_SNAPSHOT_HOOK_TIMEOUT_S"] == "60"
        assert envs["DYN_COMPONENT"] == component

        pod_spec = services[component]["extraPodSpec"]
        mounts = pod_spec["mainContainer"]["volumeMounts"]
        assert {"mountPath": hook_dir, "name": "optrt-snapshot-hooks"} in mounts
        init_mounts = pod_spec["initContainers"][0]["volumeMounts"]
        assert {"mountPath": hook_dir, "name": "optrt-snapshot-hooks"} in init_mounts
        volumes = pod_spec["volumes"]
        assert {
            "name": "optrt-snapshot-hooks",
            "hostPath": {"path": hook_dir, "type": "DirectoryOrCreate"},
        } in volumes

    frontend_text = yaml.dump(services["Frontend"])
    assert "OPTRT_SNAPSHOT_HOOKS" not in frontend_text


def test_r20_snapshot_hook_canary_runner_is_fail_closed():
    script = (DEPLOY_DIR / "snapshot_hook_canary.sh").read_text()

    assert "APPLY=0" in script
    assert "SERVER_DRY_RUN=1" in script
    assert "--enable-snapshot-hooks" in script
    assert "--server-dry-run" in script
    assert "canary_dgd_already_exists" in script
    assert "canary_pods_already_exist" in script
    assert "gpu_memory_not_idle" in script
    assert "nvidia-smi --query-gpu=index,memory.used" in script
    assert "deploy/disagg_pd_r20/snapshot_readiness.sh" in script
    assert "snapshot_resource_created" not in script

    forbidden = [
        " delete ",
        " scale ",
        " rollout restart",
        " patch dgd",
        " patch dynamographdeployment",
    ]
    for token in forbidden:
        assert token not in script


def test_r20_snapshot_hook_signal_probe_is_canary_only():
    script = (DEPLOY_DIR / "snapshot_hook_signal_probe.sh").read_text()

    assert "DRY_RUN=1" in script
    assert "refusing production r20 DGD" in script
    assert "refusing non-canary DGD name" in script
    assert "OPTRT_SNAPSHOT_HOOKS=1" in script
    assert "OPTRT_SNAPSHOT_HOOK_DIR" in script
    assert "DYN_COMPONENT" in script
    assert "SIGRTMIN+5" in script
    assert "SIGRTMIN+6" in script
    assert "dynamo.trtllm" in script
    assert "optrt_snapshot_*_${phase}.ready.json" in script
    assert "optrt_snapshot_probe_${CANARY_DGD}_$$_start.marker" in script
    assert "-newer \"$probe_marker\"" in script
    assert "wait_for_phase_ready pre_snapshot 2 pre" in script
    assert script.index("wait_for_phase_ready pre_snapshot 2 pre") < script.index("signal_worker \"$prefill_pod\" post_restore 6")
    assert "wait_for_phase_ready post_restore 2 post" in script
    assert "deploy/disagg_pd_r20/snapshot_readiness.sh" in script

    forbidden = [
        " delete ",
        " scale ",
        " rollout restart",
        " patch dgd",
        " patch dynamographdeployment",
        " apply -f",
    ]
    for token in forbidden:
        assert token not in script


def test_r20_snapshot_take_canary_is_proof_gated_and_fail_closed():
    script = (DEPLOY_DIR / "snapshot_take_canary.sh").read_text()

    assert "APPLY=0" in script
    assert "SERVER_DRY_RUN=1" in script
    assert "refusing production r20 DGD" in script
    assert "refusing non-canary DGD name" in script
    assert "canary_dgd_not_ready" in script
    assert "hook_pre_ready_recent_count" in script
    assert "hook_post_ready_recent_count" in script
    assert "missing_recent_hook_pre_post_proof" in script
    assert "recent_hook_error_files_present" in script
    assert "snapshot_resource_already_exists" in script
    assert "DynamoGraphDeploymentSnapshot" in script
    assert "apiVersion: snapshots.ai-blaise.io/v1alpha1" in script
    assert "component: ${component}" in script
    assert "render_snapshot prefill" in script
    assert "render_snapshot decode" in script
    assert "maxInFlight: ${MAX_IN_FLIGHT}" in script
    assert "type: openai-completion" in script
    assert "apply --dry-run=server" in script

    forbidden = [
        " delete ",
        " scale ",
        " rollout restart",
        " patch dgd",
        " patch dynamographdeployment",
    ]
    for token in forbidden:
        assert token not in script


def test_r20_transceiver_backend_selection_is_fail_closed():
    source = (REPO_ROOT / "tensorrt_llm" / "_torch" / "pyexecutor" / "kv_cache_transceiver.py").read_text()

    assert "_resolve_cache_transceiver_backend" in source
    assert "TRTLLM_USE_UCX_KVCACHE" in source
    assert "TRTLLM_USE_MOONCAKE_KVCACHE" in source
    assert "legacy env backend selector(s)" in source
    assert "implicit transport fallback is" in source
    assert "received multiple" in source


def test_r20_executor_emits_nixl_transfer_proof_markers():
    source = (REPO_ROOT / "tensorrt_llm" / "_torch" / "pyexecutor" / "py_executor.py").read_text()

    assert "OPTRT_NIXL_TRANSFER_PROOF" in source
    assert "phase=context_send_start" in source
    assert "phase=context_send_complete" in source
    assert "phase=gen_recv_start" in source
    assert "phase=gen_recv_complete" in source
    assert "cache_blocks=" in source


def test_r20_cpp_nixl_backend_selection_rejects_implicit_ucx_fallback():
    source = (
        REPO_ROOT / "cpp/tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.cpp"
    ).read_text()

    assert "kSUPPORTED_BACKENDS = {\"UCX\", \"LIBFABRIC\"}" in source
    assert "Unsupported NIXL backend: %s. Supported backends: UCX, LIBFABRIC" in source
    assert "fallback to UCX" not in source


def test_r20_smc_moondream_decode_pinning_is_fail_closed():
    source = (REPO_ROOT / "tensorrt_llm" / "_torch" / "speculative" / "smc.py").read_text()
    smoke = (DEPLOY_DIR / "smoke_request_pinning.sh").read_text()
    offline = (DEPLOY_DIR / "offline_request_pinning_smoke.py").read_text()
    audit = (DEPLOY_DIR / "audit_smc_decode_pinning_static.py").read_text()

    assert "validate_smc_decode_request_pin" in source
    assert "TRTLLM_SMC_REQUIRE_REQUEST_PIN" in source
    assert "generation_only" in source
    assert "SMC-SD decode requires request pin metadata" in source
    assert "disagg_request_id" in source
    assert "ctx_dp_rank" in source
    assert "ctx_info_endpoint" in source
    assert "sample_state.sampler_event.synchronize()" in source
    assert "sample_state.host.new_tokens" in source
    assert "used_pinned_host_tokens = True" in source
    assert "TRTLLM_SMC_ALLOW_UNPINNED_DRAFT_COMMIT" in source
    assert "SMC-SD Moondream decode requires evented pinned host" in source
    assert "SMC Moondream decode handoff preserved" in source
    assert source.index("validate_smc_decode_request_pin(target_model_req)") < source.index("target_model_req.py_draft_tokens = []")
    assert source.index("sample_state.sampler_event.synchronize()") < source.index("draft_tokens_host = sample_state.host.new_tokens")

    assert "SMC-SD is required but decode handoff markers are missing" in smoke
    assert "SMC Moondream decode handoff preserved" in smoke
    assert "pinned_host_tokens=True" in smoke
    assert "ctx_dp_rank=None" in smoke
    assert "ctx_info_endpoint=(?:None|null|$)" in smoke
    assert "WARNING: SMC logprob payload marker" not in smoke

    assert "missing_smc_ctx_dp_rank" in offline
    assert "missing_smc_ctx_info_endpoint" in offline
    assert "pinned_host_tokens=True" in offline
    assert "positive KV transfer" in offline
    assert "audit_smc_decode_pinning_static" not in audit


def test_r20_cpp_context_prefill_stamps_pin_endpoint():
    source = (
        REPO_ROOT / "cpp" / "tensorrt_llm" / "batch_manager" / "cacheTransceiver.cpp"
    ).read_text()
    audit = (DEPLOY_DIR / "audit_cpp_context_endpoint_static.py").read_text()
    result = (REPO_ROOT / "tensorrt_llm" / "executor" / "result.py").read_text()

    assert "std::optional<std::string> disaggInfoEndpoint" in source
    assert "mCommState->toString()" in source
    assert "if (!commEndpoint.empty())" in source
    assert "disaggInfoEndpoint = std::move(commEndpoint)" in source
    assert source.count("disaggInfoEndpoint") >= 4
    assert "ctx_info_endpoint=context_phase_params.disagg_info_endpoint" in result
    assert "C++ completed-prefill endpoint audit" in audit


def test_r20_request_pinning_smoke_is_fail_closed():
    script = (DEPLOY_DIR / "smoke_request_pinning.sh").read_text()

    assert "dynamo disagg request pin established" in script
    assert "dynamo disagg request pin outbound to decode" in script
    assert "dynamo request pin route selected" in script
    assert "dynamo request pin cleared" in script
    assert "request pin cleanup proof incomplete" in script
    assert "early-close abort cleanup proof missing" in script
    assert "host_pinned_blocks" in script
    assert "cache_state_layers" in script
    assert "no request id appears in both pin-established and outbound-to-decode" in script
    assert "route-selected prefill" in script
    assert "route-selected decode" in script
    assert "Selected worker: worker_type=prefill" in script
    assert "nvext.worker_id" in script
    assert "prefill-service" in script
    assert "decode-service" in script
    assert "REQUIRE_DYNAMO_PIN_MARKERS" in script
    assert "REQUIRE_POSITIVE_TRANSFER_METRICS" in script
    assert "REQUIRE_ABORT_CLEANUP_MARKER" in script
    assert "positive KV transfer proof missing" in script
    assert "response nvext timing" in script
    assert "worker /perf_metrics" in script
    assert "KV cache transfer timeout" in script
    assert "OPTRT_NIXL_TRANSFER_PROOF" in script
    assert r"request_id=(\S+)" in script
    assert "ctx_dp_rank is None" in script
    assert "SMC-SD must remain deferred" in script
    assert "SMC_GATE_MODE" in script
    assert "SMC-SD is required" in script
    assert "mla_latent_kv_dtype: kvarn_k2v2" in script
    assert "mla_latent_kv_dtype: auto" in script
    assert "mla_latent_kv_amortize: true" in script
    assert "indexer_k_dtype: fp4" in script
    assert "indexer_k_dtype: kvarn" in script
    assert "layersplit_transfer_backend: nixl" in script
    assert "layersplit_transfer_backend: ucx" in script
    assert "Initializing NIXL Connect" in script
    assert "global_layers=61" in script
    assert "TRTLLM_USE_(UCX|MOONCAKE|MPI)_KVCACHE=1" in script
    assert "selected UCX cache transceiver" in script
    assert "layersplit_owner_local_alloc: true" in script
    assert "backend: NIXL" in script
    assert "backend: UCX" in script
    assert "layersplit_all_cp_ranks_transfer: true" in script
    assert "no scratch routing" in script
    assert "LayerSplit: layer .* no local KV pool slot" in script
    assert "MLACacheFormatter::inquireSupport" in script
    assert "CacheTransferLayer::validateSupport" in script
    assert "speculative_model" in script
    assert "draft_attention_backend" in script
    assert "cp_type: HELIX" in script

def test_r20_nixl_gate_readiness_audit_is_read_only_and_fail_closed():
    script = (DEPLOY_DIR / "audit_nixl_gate_readiness.sh").read_text()

    assert "NIXL_AUDIT_MODE" in script
    assert "CHECK_RUNTIME_LIBS" in script
    assert "MIN_MAX_TOKENS_IN_BUFFER" in script
    assert "EXPECTED_NIXL_PLUGIN_BACKEND" in script
    assert "LOCAL_DGD_MANIFEST" in script
    assert "nixl_gate_audit_${MODE}" in script
    assert "TRTLLM_NIXL_KVCACHE_BACKEND" in script
    assert 'EXPECTED_NIXL_PLUGIN_BACKEND="${EXPECTED_NIXL_PLUGIN_BACKEND:-LIBFABRIC}"' in script
    assert "UCX|LIBFABRIC" in script
    assert "TRTLLM_NIXL_ENABLE_COALESCE" in script
    assert "TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP" in script
    assert "TRTLLM_ENABLE_KVCACHE_RECEIVE_PARALLEL" in script
    assert "max_tokens_in_buffer must be at least" in script
    assert "Initializing NIXL Connect" in script
    assert "OPTRT_LAYERSPLIT_XFER_DEBUG" in script
    assert "global_layers=61" in script
    assert "transfer_attr=True" in script
    assert "KV cache transfer timeout" in script
    assert "MLACacheFormatter::inquireSupport" in script
    assert "CacheTransferLayer::validateSupport" in script
    assert "libtensorrt_llm_nixl_wrapper.so" in script
    assert "find_spec" in script
    assert "nixl" in script
    assert "/v1/completions" not in script
    assert "kubectl apply" not in script
    assert "kubectl delete" not in script


def test_r20_nixl_plugin_probe_is_no_traffic_and_checks_vram_plugins():
    script = (DEPLOY_DIR / "probe_nixl_plugins.sh").read_text()

    assert 'PLUGINS="${PLUGINS:-UCX,LIBFABRIC}"' in script
    assert "getAvailPlugins" in script
    assert "getPluginParams" in script
    assert "createBackend" in script
    assert "VRAM_SEG" in script
    assert "plugin_probe.json" in script
    assert "plugin_probe.stderr" in script
    assert "/v1/completions" not in script
    assert "kubectl apply" not in script
    assert "kubectl delete" not in script


def test_r20_nixl_plugin_matrix_probe_is_no_traffic_and_checks_cleanup():
    script = (DEPLOY_DIR / "probe_nixl_plugin_matrix.sh").read_text()
    readme = (DEPLOY_DIR / "README.md").read_text()

    assert 'PLUGINS="${PLUGINS:-UCX,LIBFABRIC,GDS,GDS_MT}"' in script
    assert 'REQUIRE_PLUGINS="${REQUIRE_PLUGINS:-UCX,LIBFABRIC}"' in script
    assert "r20_nixl_plugin_matrix_" in script
    assert "getAvailPlugins" in script
    assert "getPluginParams" in script
    assert "createBackend" in script
    assert "VRAM_SEG" in script
    assert "cleanup_warning" in script
    assert "fi_close" in script
    assert "Device or resource busy" in script
    assert "current_nixl_gate_plugin" in script
    assert "current_nixl_gate_plugin_cleanup_risk" in script
    assert "not_peer_kv_gate" in script
    assert "summary.json" in script
    assert "result.json" in script
    assert "stderr" in script
    assert "/v1/completions" not in script
    assert "kubectl apply" not in script
    assert "kubectl delete" not in script
    assert "probe_nixl_plugin_matrix.sh" in readme
    assert "REQUIRE_PLUGINS=UCX,LIBFABRIC" in readme
    assert "NIXL `UCX` plugin remains available only as an A/B comparison candidate" in readme


def test_r20_nixl_plugin_variant_renderer_is_fail_closed():
    script = (DEPLOY_DIR / "render_nixl_plugin_variant.sh").read_text()

    assert "--plugin UCX|LIBFABRIC" in script
    assert "TRTLLM_NIXL_KVCACHE_BACKEND" in script
    assert "backend: UCX" in script
    assert "backend: MOONCAKE" in script
    assert "cp_type: HELIX" in script
    assert "layersplit_transfer_backend: ucx" in script
    assert "kubectl apply" not in script
    assert "kubectl delete" not in script


def test_r20_transport_bench_is_nixl_first_and_fail_closed():
    script = (DEPLOY_DIR / "run_c16_transport_bench.sh").read_text()

    assert 'BACKEND="nixl"' in script
    assert 'CONCURRENCY=16' in script
    assert '1024,4096,8192,16384,32768,65536,131072' in script
    assert 'ALLOW_TRANSPORT_AB=1' in script
    assert 'NIXL is the pre-A/B gate' in script
    assert 'backend: NIXL' in script
    assert 'layersplit_transfer_backend: nixl' in script
    assert 'backend: UCX' in script
    assert 'layersplit_transfer_backend: ucx' in script
    assert 'mla_latent_kv_dtype: kvarn_k2v2' in script
    assert 'indexer_k_dtype: fp4' in script
    assert 'cp_type: HELIX' in script
    assert 'tok_per_user_after_first' in script
    assert 'ttft_p95_s' in script
    assert 'itl_p99_s' in script
    assert 'perf_metrics' in script
    assert 'nvidia-smi dmon' in script
    assert 'ip -s link' in script
    assert 'MLACacheFormatter::inquireSupport' in script
    assert 'CacheTransferLayer::validateSupport' in script
    assert 'Using UCX kv-cache transceiver' in script
    assert '--min-tok-per-user' in script
    assert 'MIN_TOK_PER_USER="${MIN_TOK_PER_USER:-150}"' in script
    assert 'summary.json' in script
    assert 'positive_transfer_proof_count' in script
    assert 'KV cache transfer timeout' in script
    assert 'logs = \"\\n\".join' in script
    assert r'request_id=(\S+)' in script

def test_r20_transport_bench_embedded_verifier_compiles():
    script = (DEPLOY_DIR / "run_c16_transport_bench.sh").read_text()

    match = re.search(r"<<'PY_VERIFY'\n(.*?)\nPY_VERIFY", script, re.DOTALL)
    assert match is not None
    compile(match.group(1), "run_c16_transport_bench.py_verify", "exec")


def test_r20_cpp_cache_sender_completes_cancelled_promises():
    source = (REPO_ROOT / "cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp").read_text()
    cancel_branch = source[source.index("if (mCancelledRequests.find(reqId)"):source.index("void response() noexcept")]
    assert "it->second.mPromise.set_value()" in cancel_branch
    assert "mReadyResponses.erase(it)" in cancel_branch


def test_r20_cache_transceiver_timeout_config_roundtrips_sender_future_timeout():
    config_source = (REPO_ROOT / "cpp/tensorrt_llm/executor/cacheTransceiverConfig.cpp").read_text()
    nanobind_source = (REPO_ROOT / "cpp/tensorrt_llm/nanobind/executor/executorConfig.cpp").read_text()
    llm_args = (REPO_ROOT / "tensorrt_llm/llmapi/llm_args.py").read_text()

    operator_eq = config_source[
        config_source.index("bool CacheTransceiverConfig::operator=="):
        config_source.index("void CacheTransceiverConfig::setBackendType")
    ]
    assert "mKvTransferSenderFutureTimeoutMs == other.mKvTransferSenderFutureTimeoutMs" in operator_eq
    assert "getKvTransferSenderFutureTimeoutMs" in nanobind_source
    assert "state.size() != 3 && state.size() != 4" in nanobind_source
    assert "senderFutureTimeoutMs" in nanobind_source
    assert "kv_transfer_sender_future_timeout_ms" in llm_args
    assert "Requests exceeding this timeout will be cancelled" in llm_args


def test_r20_live_smoke_strips_ansi_router_logs():
    smoke = (REPO_ROOT / "deploy/disagg_pd_r20/smoke_request_pinning.sh").read_text()
    assert "ansi_re = re.compile" in smoke
    assert "ansi_re.sub" in smoke

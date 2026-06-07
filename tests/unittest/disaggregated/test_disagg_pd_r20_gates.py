import re
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
    assert cfg["moe_config"]["warp_decode"]["enabled"] is False


def test_r20_decode_canary_gates():
    cfg = _load_yaml("decode.yaml")

    assert cfg["disable_overlap_scheduler"] is False
    assert cfg["enable_attention_dp"] is True
    assert cfg["context_parallel_size"] == 1
    assert cfg["cache_transceiver_config"]["backend"] == "NIXL"
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
    assert manifest.count("TRTLLM_NIXL_KVCACHE_BACKEND") >= 2
    assert manifest.count("TRTLLM_NIXL_ENABLE_COALESCE") >= 2
    assert "UCX_CUDA_IPC_ENABLE_MNNVL" in manifest
    assert "NVIDIA_GDRCOPY" in manifest
    assert "TRTLLM_FORCE_COMM_METHOD" in manifest
    assert "NVLINK_TWO_SIDED" in manifest
    assert "allow_parallelism_fallback: false" in manifest


def test_r20_overlay_carries_layersplit_and_request_pinning_sources():
    dockerfile = (DEPLOY_DIR / "Dockerfile.r20-overlay").read_text()

    assert "tensorrt_llm/_torch/pyexecutor/py_executor_creator.py" in dockerfile
    assert "tensorrt_llm/_torch/pyexecutor/kv_cache_transceiver.py" in dockerfile
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
    assert "UCX|LIBFABRIC" in script
    assert "TRTLLM_NIXL_ENABLE_COALESCE" in script
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


def test_r20_live_smoke_strips_ansi_router_logs():
    smoke = (REPO_ROOT / "deploy/disagg_pd_r20/smoke_request_pinning.sh").read_text()
    assert "ansi_re = re.compile" in smoke
    assert "ansi_re.sub" in smoke

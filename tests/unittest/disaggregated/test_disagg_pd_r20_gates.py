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
    assert "allow_parallelism_fallback: false" in manifest


def test_r20_overlay_carries_layersplit_and_request_pinning_sources():
    dockerfile = (DEPLOY_DIR / "Dockerfile.r20-overlay").read_text()

    assert "tensorrt_llm/_torch/pyexecutor/py_executor_creator.py" in dockerfile
    assert "tensorrt_llm/serve/openai_disagg_service.py" in dockerfile
    assert "tensorrt_llm/serve/openai_client.py" in dockerfile
    assert "tensorrt_llm/serve/openai_protocol.py" in dockerfile
    assert "tensorrt_llm/serve/openai_server.py" in dockerfile
    assert "tensorrt_llm/disaggregated_params.py" in dockerfile


def test_r20_request_pinning_smoke_is_fail_closed():
    script = (DEPLOY_DIR / "smoke_request_pinning.sh").read_text()

    assert "dynamo disagg request pin established" in script
    assert "dynamo disagg request pin outbound to decode" in script
    assert "dynamo request pin route selected" in script
    assert "Selected worker: worker_type=prefill" in script
    assert "nvext.worker_id" in script
    assert "REQUIRE_DYNAMO_PIN_MARKERS" in script
    assert "ctx_dp_rank is None" in script
    assert "SMC-SD must remain deferred" in script
    assert "mla_latent_kv_dtype: kvarn_k2v2" in script
    assert "layersplit_transfer_backend: nixl" in script
    assert "layersplit_owner_local_alloc: true" in script
    assert "backend: NIXL" in script
    assert "backend: UCX" in script
    assert "layersplit_all_cp_ranks_transfer: true" in script
    assert "no scratch routing" in script
    assert "LayerSplit: layer .* no local KV pool slot" in script
    assert "SMC_GATE_MODE" in script
    assert "speculative_model" in script
    assert "cp_type: HELIX" in script

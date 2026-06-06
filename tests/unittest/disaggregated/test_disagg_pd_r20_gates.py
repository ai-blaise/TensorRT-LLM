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
    assert cfg["sparse_attention_config"]["layersplit_transfer_backend"] == "ucx"
    assert cfg["sparse_attention_config"]["mla_latent_kv_dtype"] == "kvarn_k2v2"
    assert cfg["sparse_attention_config"]["mla_latent_kv_amortize"] is True
    assert cfg["cache_transceiver_config"]["backend"] == "UCX"
    assert cfg["moe_config"]["warp_decode"]["enabled"] is False


def test_r20_decode_canary_gates():
    cfg = _load_yaml("decode.yaml")

    assert cfg["disable_overlap_scheduler"] is False
    assert cfg["enable_attention_dp"] is True
    assert cfg["context_parallel_size"] == 1
    assert cfg["cache_transceiver_config"]["backend"] == "UCX"
    assert cfg["moe_config"]["backend"] == "WARPDECODE"
    assert cfg["moe_config"]["warp_decode"]["enabled"] is True
    assert cfg["moe_config"]["warp_decode"]["policy"] == "force"
    assert cfg["moe_config"]["warp_decode"]["allow_parallelism_fallback"] is False
    assert cfg["sparse_attention_config"]["layersplit_enabled"] is False
    assert cfg["sparse_attention_config"]["mla_latent_kv_dtype"] == "kvarn_k2v2"
    assert cfg["sparse_attention_config"]["mla_latent_kv_amortize"] is True
    assert cfg["speculative_config"]["decoding_type"] == "SMC"


def test_r20_manifest_has_no_helix_fallback():
    manifest = (DEPLOY_DIR / "topo-c1-dp2tp4-disagg-r20.yaml").read_text()
    assert "cp_type: HELIX" not in manifest
    assert "layersplit_enabled: true" in manifest
    assert "allow_parallelism_fallback: false" in manifest


def test_r20_overlay_carries_moondream_and_request_pinning_sources():
    dockerfile = (DEPLOY_DIR / "Dockerfile.r20-overlay").read_text()

    assert "tensorrt_llm/_torch/speculative/model_drafter.py" in dockerfile
    assert "tensorrt_llm/_torch/speculative/smc.py" in dockerfile
    assert "tensorrt_llm/_torch/pyexecutor/py_executor_creator.py" in dockerfile
    assert "tensorrt_llm/serve/openai_disagg_service.py" in dockerfile

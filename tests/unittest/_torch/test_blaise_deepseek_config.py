from types import SimpleNamespace

import pytest

import tensorrt_llm._torch.model_config as model_config
from tensorrt_llm._torch.model_config import (
    _deepseek_sparse_attention_config, _get_blaise_indexer_overrides,
    _is_deepseek_dsa_config)
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig


def test_blaise_nvfp4_hisa_model_card_maps_to_trt_dsa_config():
    pretrained_config = SimpleNamespace(
        architectures=["DeepseekV3ForCausalLM"],
        index_topk=1024,
        quantization_config={
            "indexer_quantization": {
                "quant_method": "nvfp4_e2m1_ue8m0",
                "indexcache": {
                    "enabled": True,
                    "freq": 4,
                    "pattern": "fsss",
                },
                "hisa": {
                    "enabled": True,
                    "mode": "indexcache-hisa",
                    "block_size": 128,
                    "block_topk": 64,
                    "compression_ratio": 4.0,
                    "min_seq_len": 32768,
                    "execution_mode": "optimized",
                },
            },
            "kv_cache_scheme": {
                "quant_method": "higgs_dense_2bit",
            },
        },
    )

    overrides = _get_blaise_indexer_overrides(pretrained_config)

    assert _is_deepseek_dsa_config(pretrained_config)
    assert overrides == {
        "indexer_k_dtype": "fp4",
        "indexer_mode": "indexcache-hisa",
        "index_topk_freq": 4,
        "index_topk_pattern": "FSSS",
        "enable_nvfp4_hisa": True,
        "hisa_block_size": 128,
        "hisa_block_topk": 64,
        "hisa_compression_ratio": 4.0,
        "hisa_min_seq_len": 32768,
        "hisa_execution_mode": "optimized",
    }


def test_indexcache_hisa_requires_trt_fp4_indexer_cache():
    with pytest.raises(ValueError, match="indexcache-hisa requires"):
        DeepSeekSparseAttentionConfig(
            indexer_mode="indexcache-hisa",
            index_head_dim=128,
            indexer_k_dtype="fp8",
            enable_nvfp4_hisa=True,
        )


def test_higgs_kv_scheme_does_not_affect_trt_indexer_overrides():
    pretrained_config = SimpleNamespace(
        architectures=["DeepseekV3ForCausalLM"],
        index_topk=1024,
        quantization_config={
            "kv_cache_scheme": {
                "quant_method": "higgs_dense_2bit",
                "preset": "eden2_16",
            }
        },
    )

    assert _get_blaise_indexer_overrides(pretrained_config) == {}


def test_generic_fp4_indexer_method_is_not_treated_as_nvfp4():
    pretrained_config = SimpleNamespace(
        architectures=["DeepseekV3ForCausalLM"],
        index_topk=1024,
        quantization_config={
            "indexer_quantization": {
                "quant_method": "fp4",
            },
        },
    )

    assert _get_blaise_indexer_overrides(pretrained_config) == {}


def test_layersplit_is_not_enabled_from_hf_model_card():
    # LayerSplit is a runtime/system topology feature and must not be enabled
    # from the HF model card. The HF parser must drop every layersplit_* key
    # so production enablement only flows through SparseAttentionConfig.
    pretrained_config = SimpleNamespace(
        architectures=["DeepseekV3ForCausalLM"],
        index_topk=1024,
        quantization_config={
            "indexer_quantization": {
                "quant_method": "nvfp4_e2m1_ue8m0",
                "layersplit": {
                    "enabled": True,
                    "layout": "interleaved",
                    "owner_assignment": "round_robin",
                    "transfer_backend": "ucx",
                    "all_cp_ranks_transfer": True,
                },
            },
        },
    )

    overrides = _get_blaise_indexer_overrides(pretrained_config)

    assert all(not key.startswith("layersplit") for key in overrides), (
        f"HF model card must not enable LayerSplit; leaked keys: "
        f"{[k for k in overrides if k.startswith('layersplit')]}")


def test_hisa_overrides_ignore_sibling_layersplit_block():
    # When HF indexer_quantization contains both a HISA block and a LayerSplit
    # block, only the HISA fields propagate; LayerSplit must stay runtime-only.
    pretrained_config = SimpleNamespace(
        architectures=["DeepseekV3ForCausalLM"],
        index_topk=1024,
        quantization_config={
            "indexer_quantization": {
                "quant_method": "nvfp4_e2m1_ue8m0",
                "hisa": {
                    "enabled": True,
                    "mode": "indexcache-hisa",
                    "block_size": 128,
                    "block_topk": 64,
                },
                "layersplit": {
                    "enabled": True,
                    "owner_assignment": "contiguous",
                },
            },
        },
    )

    overrides = _get_blaise_indexer_overrides(pretrained_config)

    assert overrides["indexer_mode"] == "indexcache-hisa"
    assert overrides["enable_nvfp4_hisa"] is True
    assert overrides["hisa_block_size"] == 128
    assert overrides["hisa_block_topk"] == 64
    assert all(not key.startswith("layersplit") for key in overrides)


def test_layersplit_owner_assignment_defaults_to_contiguous():
    # 1B-i: the Pydantic default must be 'contiguous' so the in-engine
    # broadcast owner-map agrees with the C++ CP->non-CP reassembly
    # (cacheSplitConcat.cu) and the NIXL handoff, which both REQUIRE
    # contiguous balanced layer spans. A round_robin default would desync the
    # owner-map from the contiguous-shard reassembly at the disagg handoff.
    cfg = DeepSeekSparseAttentionConfig(index_head_dim=128)
    assert cfg.layersplit_owner_assignment == "contiguous"


def test_layersplit_owner_assignment_round_robin_still_selectable():
    # contiguous is only the DEFAULT; round_robin must remain a valid choice
    # (it matches the SGLang op-ls reference) for non-disaggregated use.
    cfg = DeepSeekSparseAttentionConfig(index_head_dim=128,
                                        layersplit_owner_assignment="round_robin")
    assert cfg.layersplit_owner_assignment == "round_robin"
    # Both values pass validation (the validator gates all_cp_ranks_transfer,
    # not the owner-assignment policy).
    cfg_c = DeepSeekSparseAttentionConfig(index_head_dim=128,
                                          layersplit_enabled=True,
                                          layersplit_owner_assignment="contiguous")
    assert cfg_c.layersplit_owner_assignment == "contiguous"


def _valid_hisparse_sparse_config(**overrides):
    kwargs = {
        "index_head_dim": 128,
        "index_topk": 1024,
        "indexer_k_dtype": "fp4",
        "indexer_mode": "indexcache-hisa",
        "enable_nvfp4_hisa": True,
        "mla_latent_kv_dtype": "kvarn_k2v2",
        "mla_latent_kv_amortize": True,
        "hisparse_enabled": True,
    }
    kwargs.update(overrides)
    return DeepSeekSparseAttentionConfig(**kwargs)


def test_hisparse_accepts_production_dense_mla_kvarn_config():
    cfg = _valid_hisparse_sparse_config()

    assert cfg.hisparse_enabled is True
    assert cfg.hisparse_mode == "dense_mla_kvarn"
    assert cfg.hisparse_topk == cfg.index_topk
    assert cfg.hisparse_direct_to_host is True
    assert cfg.hisparse_indexer_host_tier is False
    assert cfg.mla_latent_kv_dtype == "kvarn_k2v2"
    assert cfg.indexer_k_dtype == "fp4"


def test_hisparse_rejects_non_kvarn_dense_mla_storage():
    with pytest.raises(ValueError, match="dense MLA KVarN"):
        _valid_hisparse_sparse_config(mla_latent_kv_dtype="auto")


def test_hisparse_rejects_indexer_host_tier_and_direct_to_host_off():
    with pytest.raises(ValueError, match="Indexer K"):
        _valid_hisparse_sparse_config(hisparse_indexer_host_tier=True)

    with pytest.raises(ValueError, match="hisparse_direct_to_host=true"):
        _valid_hisparse_sparse_config(hisparse_direct_to_host=False)


def test_hisparse_rejects_topk_mismatch_and_open_fallback_policy():
    with pytest.raises(ValueError, match="hisparse_topk must match index_topk"):
        _valid_hisparse_sparse_config(hisparse_topk=512)

    with pytest.raises(ValueError, match="hisparse_fail_closed"):
        _valid_hisparse_sparse_config(hisparse_fail_closed=False)


def test_hisparse_runtime_requires_nixl_python_and_no_block_reuse():
    cfg = _valid_hisparse_sparse_config()

    with pytest.raises(ValueError, match="backend='NIXL'"):
        cfg.validate_hisparse_runtime_config(
            kv_cache_config=SimpleNamespace(enable_block_reuse=False),
            cache_transceiver_config=SimpleNamespace(
                backend="UCX", transceiver_runtime="PYTHON"))

    with pytest.raises(ValueError, match="transceiver_runtime='PYTHON'"):
        cfg.validate_hisparse_runtime_config(
            kv_cache_config=SimpleNamespace(enable_block_reuse=False),
            cache_transceiver_config=SimpleNamespace(
                backend="NIXL", transceiver_runtime="CPP"))

    with pytest.raises(ValueError, match="enable_block_reuse=false"):
        cfg.validate_hisparse_runtime_config(
            kv_cache_config=SimpleNamespace(enable_block_reuse=True),
            cache_transceiver_config=SimpleNamespace(
                backend="NIXL", transceiver_runtime="PYTHON"))

    cfg.validate_hisparse_runtime_config(
        kv_cache_config=SimpleNamespace(enable_block_reuse=False),
        cache_transceiver_config=SimpleNamespace(
            backend="NIXL", transceiver_runtime="PYTHON"))


def test_sparse_attention_config_attaches_blaise_runtime_fields(monkeypatch):
    class RuntimeSparseAttentionConfig:
        model_fields = {
            "index_head_dim": object(),
            "index_topk": object(),
        }

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            for key, value in kwargs.items():
                setattr(self, key, value)

    monkeypatch.setattr(model_config, "DeepSeekSparseAttentionConfig",
                        RuntimeSparseAttentionConfig)

    sparse_attention_config = _deepseek_sparse_attention_config(
        index_head_dim=128,
        index_topk=1024,
        indexer_mode="indexcache-hisa",
        enable_nvfp4_hisa=True,
    )

    assert sparse_attention_config.kwargs == {
        "index_head_dim": 128,
        "index_topk": 1024,
    }
    assert sparse_attention_config.indexer_mode == "indexcache-hisa"
    assert sparse_attention_config.enable_nvfp4_hisa is True

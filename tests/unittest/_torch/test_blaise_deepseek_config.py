from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.model_config import (
    _get_blaise_indexer_overrides, _is_deepseek_dsa_config)
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

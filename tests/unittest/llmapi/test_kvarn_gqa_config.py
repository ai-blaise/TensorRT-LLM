"""GQA KVarN KV-cache config plumbing coverage.

These tests cover HF/default routing and fail-closed config behavior. The
reference GQA backend is intentionally isolated from dense MLA and Indexer paths.
"""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tensorrt_llm._torch.pyexecutor.model_loader import (
    _hf_kvarn_gqa_kv_dtype,
    validate_and_set_kv_cache_quant,
)
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.quantization.mode import QuantAlgo


def test_kvarn_gqa_dtype_requires_group128_block_size():
    cfg = KvCacheConfig(dtype="kvarn_k2v2_g128", tokens_per_block=128)

    assert cfg.dtype == "kvarn_k2v2_g128"
    assert cfg.tokens_per_block == 128

    with pytest.raises(ValidationError, match="tokens_per_block=128"):
        KvCacheConfig(dtype="kvarn_k2v2_g128")


def test_kvarn_gqa_dtype_rejects_bad_shape():
    with pytest.raises(ValidationError, match="g128"):
        KvCacheConfig(dtype="kvarn_k2v2_g64", tokens_per_block=64)

    with pytest.raises(ValidationError, match="2/3/4-bit"):
        KvCacheConfig(dtype="kvarn_k8v2_g128", tokens_per_block=128)


def test_hf_config_can_request_gqa_kvarn():
    cfg = SimpleNamespace(kv_cache_dtype="kvarn_k2v2_g128")
    assert _hf_kvarn_gqa_kv_dtype(cfg) == "kvarn_k2v2_g128"

    cfg = SimpleNamespace(
        quantization_config={
            "kvarn": {"gqa": {"enabled": True, "dtype": "kvarn_k2v2_g128"}}
        }
    )
    assert _hf_kvarn_gqa_kv_dtype(cfg) == "kvarn_k2v2_g128"


def test_hf_config_defaults_gqa_kvarn_when_model_declares_support():
    cfg = SimpleNamespace(supports_kvarn_gqa=True)
    assert _hf_kvarn_gqa_kv_dtype(cfg) == "kvarn_k2v2_g128"

    cfg = SimpleNamespace(quantization_config={"kvarn": {"gqa": {"enabled": True}}})
    assert _hf_kvarn_gqa_kv_dtype(cfg) == "kvarn_k2v2_g128"

    cfg = SimpleNamespace(quantization_config={"kvarn": {"gqa": {"enabled": False}}})
    assert _hf_kvarn_gqa_kv_dtype(cfg) is None


def test_gqa_kvarn_request_sets_kvarn_quant_mode():
    model_config = SimpleNamespace(
        quant_config=SimpleNamespace(kv_cache_quant_algo=None),
        pretrained_config=SimpleNamespace(),
    )

    validate_and_set_kv_cache_quant(model_config, "kvarn_k2v2_g128")

    assert model_config.quant_config.kv_cache_quant_algo == QuantAlgo.KVARN.value
    assert model_config.quant_config.kv_cache_dtype == "kvarn_k2v2_g128"


def test_hf_dense_mla_kvarn_config_does_not_enable_gqa_kvarn():
    cfg = SimpleNamespace(
        quantization_config={
            "kvarn": {"path": "dense_mla", "dtype": "kvarn_k2v2"}
        }
    )

    assert _hf_kvarn_gqa_kv_dtype(cfg) is None


def test_hf_indexer_kvarn_config_does_not_enable_gqa_kvarn():
    cfg = SimpleNamespace(
        quantization_config={
            "kvarn": {
                "indexer": {"enabled": True, "dtype": "fp4_hisa"},
                "gqa": {"enabled": False},
            }
        }
    )

    assert _hf_kvarn_gqa_kv_dtype(cfg) is None

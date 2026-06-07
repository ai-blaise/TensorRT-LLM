"""GQA KVarN KV-cache config plumbing coverage.

These tests cover HF/default routing and fail-closed config behavior. The
reference GQA backend is intentionally isolated from dense MLA and Indexer paths.
"""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tensorrt_llm._torch.pyexecutor import model_loader
from tensorrt_llm._torch.pyexecutor.model_loader import (
    _KVARN_GQA_FUSED_OPS,
    _KVARN_GQA_READY_OP,
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


def test_gqa_kvarn_request_fails_closed_without_fused_ops(monkeypatch):
    model_config = SimpleNamespace(
        quant_config=SimpleNamespace(kv_cache_quant_algo=None),
        pretrained_config=SimpleNamespace(),
    )
    monkeypatch.setattr(model_loader, "_has_kvarn_gqa_fused_backend", lambda: False)

    with pytest.raises(NotImplementedError, match="production fused"):
        validate_and_set_kv_cache_quant(model_config, "kvarn_k2v2_g128")


def test_gqa_kvarn_fused_ops_gate_sets_kvarn_quant_mode(monkeypatch):
    model_config = SimpleNamespace(
        quant_config=SimpleNamespace(kv_cache_quant_algo=None),
        pretrained_config=SimpleNamespace(),
    )
    monkeypatch.setattr(model_loader, "_has_kvarn_gqa_fused_backend", lambda: True)

    validate_and_set_kv_cache_quant(model_config, "kvarn_k2v2_g128")

    assert _KVARN_GQA_FUSED_OPS == ("kvarn_gqa_store", "kvarn_gqa_decode", "kvarn_gqa_dequant_amortized")
    assert _KVARN_GQA_READY_OP == "kvarn_gqa_backend_ready"
    assert model_config.quant_config.kv_cache_quant_algo == QuantAlgo.KVARN.value
    assert model_config.quant_config.kv_cache_dtype == "kvarn_k2v2_g128"


def test_gqa_kvarn_fused_backend_probe_requires_ready_op(monkeypatch):
    class Ready:
        def __init__(self, value):
            self.value = value

        def __call__(self):
            return self.value

    trtllm = SimpleNamespace(
        kvarn_gqa_store=object(),
        kvarn_gqa_decode=object(),
        kvarn_gqa_dequant_amortized=object(),
    )
    monkeypatch.setattr(model_loader.torch, "ops", SimpleNamespace(trtllm=trtllm))
    assert model_loader._has_kvarn_gqa_fused_backend() is False

    trtllm.kvarn_gqa_dequant_amortized = object()
    trtllm.kvarn_gqa_backend_ready = Ready(False)
    assert model_loader._has_kvarn_gqa_fused_backend() is False

    trtllm.kvarn_gqa_backend_ready = Ready(True)
    assert model_loader._has_kvarn_gqa_fused_backend() is True


def test_gqa_kvarn_fused_backend_probe_fails_closed_on_ready_error(monkeypatch):
    def raises():
        raise RuntimeError("backend probe failed")

    trtllm = SimpleNamespace(
        kvarn_gqa_store=object(),
        kvarn_gqa_decode=object(),
        kvarn_gqa_dequant_amortized=object(),
        kvarn_gqa_backend_ready=raises,
    )
    monkeypatch.setattr(model_loader.torch, "ops", SimpleNamespace(trtllm=trtllm))

    assert model_loader._has_kvarn_gqa_fused_backend() is False


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

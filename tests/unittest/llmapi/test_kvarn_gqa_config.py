"""GQA KVarN KV-cache config plumbing coverage.

These tests intentionally stop at config/fail-fast behavior. The generic GQA
store/read/dequant kernels are not present in op-trt yet, so KVarN GQA requests
must not silently fall through to fp16/fp8/nvfp4 cache behavior.
"""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tensorrt_llm._torch.pyexecutor.model_loader import (
    _hf_kvarn_gqa_kv_dtype,
    validate_and_set_kv_cache_quant,
)
from tensorrt_llm.llmapi.llm_args import KvCacheConfig


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


def test_gqa_kvarn_request_fails_until_generic_backend_exists():
    model_config = SimpleNamespace(
        quant_config=SimpleNamespace(kv_cache_quant_algo=None),
        pretrained_config=SimpleNamespace(),
    )

    with pytest.raises(NotImplementedError, match="generic paged K/V KVarN"):
        validate_and_set_kv_cache_quant(model_config, "kvarn_k2v2_g128")

import types

import pytest
import torch

from tensorrt_llm._torch.modules.fused_moe.warp_decode import (
    get_warp_decode_guard_failure, try_run_warp_decode)


class CutlassFusedMoE:
    pass


class TRTLLMGenFusedMoE:
    pass


def _config(policy="auto"):
    return types.SimpleNamespace(
        enabled=True,
        max_batch_size=8,
        policy=policy,
        allow_parallelism_fallback=(policy != "force"),
    )


def _moe(**overrides):
    backend = CutlassFusedMoE()
    backend.w3_w1_weight = torch.empty((2, 32, 16), dtype=torch.bfloat16)
    backend.w2_weight = torch.empty((2, 16, 16), dtype=torch.bfloat16)
    backend.quant_method = types.SimpleNamespace(use_shuffled_weight=False)
    values = dict(
        model_config=types.SimpleNamespace(warp_decode_config=_config()),
        routing_method=types.SimpleNamespace(experts_per_token=8),
        has_nvfp4=False,
        backend=backend,
        comm=None,
        enable_dwdp=False,
        warp_decode_is_decode_only=True,
        warp_decode_is_cuda_graph=False,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _inputs(num_tokens=4):
    return dict(
        x=torch.empty((num_tokens, 16), dtype=torch.bfloat16),
        token_selected_experts=torch.empty((num_tokens, 8), dtype=torch.int32),
        token_final_scales=torch.empty((num_tokens, 8), dtype=torch.float32),
        x_sf=None,
        do_finalize=True,
        all_rank_num_tokens=[num_tokens],
    )


def test_warp_decode_requires_runtime_decode_signal():
    moe = _moe(warp_decode_is_decode_only=False)
    assert get_warp_decode_guard_failure(moe, **_inputs()) == "not_decode_only"


def test_warp_decode_reports_nvfp4_kernel_gap():
    backend = TRTLLMGenFusedMoE()
    moe = _moe(backend=backend, has_nvfp4=True)
    inputs = _inputs()
    inputs["x_sf"] = torch.empty((4, 1), dtype=torch.uint8)
    assert get_warp_decode_guard_failure(moe, **inputs) == (
        "nvfp4_warp_decode_kernel_missing")


def test_warp_decode_rejects_eplb_slots():
    moe = _moe(layer_load_balancer=object())
    assert get_warp_decode_guard_failure(moe, **_inputs()) == "eplb_not_supported"


def test_warp_decode_force_raises_exact_guard_reason():
    moe = _moe(
        model_config=types.SimpleNamespace(warp_decode_config=_config("force")),
        warp_decode_is_decode_only=False,
    )
    with pytest.raises(NotImplementedError, match="not_decode_only"):
        try_run_warp_decode(moe, **_inputs())
    assert moe.warp_decode_last_status == "fallback"
    assert moe.warp_decode_last_reason == "not_decode_only"


def test_warp_decode_disabled_records_status():
    moe = _moe(model_config=types.SimpleNamespace(warp_decode_config=None))
    assert try_run_warp_decode(moe, **_inputs()) is None
    assert moe.warp_decode_last_status == "disabled"
    assert moe.warp_decode_last_reason == "disabled"


def test_warp_decode_disallows_fallback_when_requested():
    config = _config("auto")
    config.allow_parallelism_fallback = False
    moe = _moe(
        model_config=types.SimpleNamespace(warp_decode_config=config),
        warp_decode_is_decode_only=False,
    )
    with pytest.raises(NotImplementedError, match="not_decode_only"):
        try_run_warp_decode(moe, **_inputs())
    assert moe.warp_decode_last_status == "fallback"
    assert moe.warp_decode_last_reason == "not_decode_only"

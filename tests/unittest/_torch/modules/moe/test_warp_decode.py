import types

import pytest
import torch

from tensorrt_llm._torch.modules.fused_moe import warp_decode
from tensorrt_llm._torch.modules.fused_moe.warp_decode import (
    get_cursor_warp_decode_plan, get_warp_decode_guard_failure,
    try_run_warp_decode)


class CutlassFusedMoE:
    pass


class TRTLLMGenFusedMoE:
    pass


def _populate_nvfp4_backend(
    backend,
    *,
    weights=True,
    local_experts=128,
    packed_hidden_size=3584,
    weight_storage_dtype=torch.uint8,
    scale_storage_dtype=torch.uint8,
):
    backend.hidden_size = 7168
    backend.intermediate_size = 2048
    backend.num_experts = 128
    backend.num_slots = 128
    backend.expert_size_per_partition = local_experts
    backend.slot_start = 0
    backend.scaling_vector_size = 16
    if weights:
        padded_hidden_size = packed_hidden_size * 2
        storage_factor = torch.empty((), dtype=weight_storage_dtype).element_size()
        assert packed_hidden_size % storage_factor == 0
        assert 1024 % storage_factor == 0
        scale_storage_factor = torch.empty((), dtype=scale_storage_dtype).element_size()
        assert (padded_hidden_size // 16) % scale_storage_factor == 0
        assert 128 % scale_storage_factor == 0
        backend.w3_w1_weight = torch.empty(
            (local_experts, 4096, packed_hidden_size // storage_factor),
            dtype=weight_storage_dtype,
        )
        backend.w3_w1_weight_scale = torch.empty(
            (
                local_experts,
                4096,
                (padded_hidden_size // 16) // scale_storage_factor,
            ),
            dtype=scale_storage_dtype,
        )
        backend.w2_weight = torch.empty(
            (local_experts, padded_hidden_size, 1024 // storage_factor),
            dtype=weight_storage_dtype,
        )
        backend.w2_weight_scale = torch.empty(
            (local_experts, padded_hidden_size, 128 // scale_storage_factor),
            dtype=scale_storage_dtype,
        )
        backend.fc31_alpha = torch.ones((local_experts,), dtype=torch.float32)
        backend.fc2_alpha = torch.ones((local_experts,), dtype=torch.float32)
        backend.fc2_input_scale = torch.ones((1,), dtype=torch.float32)
    return backend


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


@pytest.mark.parametrize(
    ("num_tokens", "bucket_tokens"),
    [(1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (8, 8), (9, 16), (16, 16), (17, 32), (32, 32)],
)
def test_cursor_warp_decode_plan_uses_graph_buckets(num_tokens, bucket_tokens):
    plan = get_cursor_warp_decode_plan(num_tokens)
    assert plan.requested_tokens == num_tokens
    assert plan.bucket_tokens == bucket_tokens
    assert plan.route_slots_shape == (bucket_tokens, 8)
    assert plan.route_scales_shape == (bucket_tokens, 8)
    assert plan.activation_scale_shape == (bucket_tokens, 448)
    assert plan.exact_expanded_rows == bucket_tokens * 8
    assert plan.gate_up_warps == bucket_tokens * 8 * 2048
    assert plan.down_warps == bucket_tokens * 7168


def test_cursor_warp_decode_plan_records_required_eliminations():
    plan = get_cursor_warp_decode_plan(32)
    assert plan.intermediate_shape == (32, 8, 2048)
    assert plan.output_shape == (32, 7168)
    assert plan.warps_per_cta == 8
    assert set(plan.eliminated_stages) == {
        "expert_major_batches",
        "expert_padding",
        "moe_sort",
        "scatter_combine",
        "activation_gather_buffer",
        "per_expert_output_buffer",
    }


def test_cursor_warp_decode_plan_rejects_unbucketed_decode():
    with pytest.raises(ValueError, match="does not cover 33 tokens"):
        get_cursor_warp_decode_plan(33)


def test_cursor_warp_decode_plan_covers_only_campaign_concurrency():
    plan = get_cursor_warp_decode_plan(32)
    assert plan.bucket_tokens == 32
    assert plan.exact_expanded_rows == 256
    with pytest.raises(ValueError, match="does not cover 64 tokens"):
        get_cursor_warp_decode_plan(64)


def test_cursor_warp_decode_plan_uses_post_dispatch_slot_metadata():
    plan = get_cursor_warp_decode_plan(16)
    assert plan.route_slots_shape == (16, 8)
    assert plan.route_scales_shape == (16, 8)
    assert "moe_sort" in plan.eliminated_stages
    assert "expert_major_batches" in plan.eliminated_stages
    assert "scatter_combine" in plan.eliminated_stages


def test_warp_decode_reports_nvfp4_op_gap_after_contract_guards(monkeypatch):
    backend = _populate_nvfp4_backend(TRTLLMGenFusedMoE())
    moe = _moe(backend=backend, has_nvfp4=True)
    inputs = _inputs()
    inputs["x"] = torch.empty((4, 3584), dtype=torch.uint8)
    inputs["x_sf"] = torch.empty((4, 448), dtype=torch.uint8)
    monkeypatch.setattr(warp_decode, "_required_nvfp4_ops_available", lambda: False)
    assert get_warp_decode_guard_failure(moe, **inputs) == (
        "nvfp4_warp_decode_op_unavailable")


def test_warp_decode_selects_explicit_tactic_nvfp4_op(monkeypatch):
    backend = _populate_nvfp4_backend(TRTLLMGenFusedMoE())
    moe = _moe(backend=backend, has_nvfp4=True)
    inputs = _inputs(num_tokens=2)
    inputs["x"] = torch.empty((2, 3584), dtype=torch.uint8)
    inputs["x_sf"] = torch.empty((2, 448), dtype=torch.uint8)

    def _op(*args):
        assert args[0] is inputs["x"]
        assert args[1] is inputs["x_sf"]
        assert args[9] is inputs["token_selected_experts"]
        assert args[10] is inputs["token_final_scales"]
        assert args[11:17] == (7168, 2048, 128, 0, 128, 16)
        return torch.empty((2, 7168), dtype=torch.bfloat16)

    monkeypatch.setattr(warp_decode, "_get_nvfp4_op", lambda: _op)
    output = try_run_warp_decode(moe, **inputs)

    assert output.shape == (2, 7168)
    assert output.dtype == torch.bfloat16
    assert moe.warp_decode_last_status == "selected"
    assert moe.warp_decode_last_reason == "nvfp4_explicit_tactic_op"


def test_warp_decode_explicit_tactic_pads_nvfp4_hidden_to_weight_contract(monkeypatch):
    backend = _populate_nvfp4_backend(TRTLLMGenFusedMoE(), packed_hidden_size=4096)
    moe = _moe(backend=backend, has_nvfp4=True)
    inputs = _inputs(num_tokens=2)
    inputs["x"] = torch.empty((2, 3584), dtype=torch.uint8)
    inputs["x_sf"] = torch.empty((2, 448), dtype=torch.uint8)

    class _Runner:

        def run_moe(self, *args):
            assert args[2].shape == (2, 4096)
            assert args[3].numel() == 2 * 512
            assert args[4].shape[-1] == 4096
            assert args[10].shape[1] == 8192
            assert args[11].shape[1] == 8192
            assert args[-1].shape == (2, 7168)
            return [args[-1]]

    monkeypatch.setattr(warp_decode, "_nvfp4_torch_runner", lambda: _Runner())

    assert get_warp_decode_guard_failure(moe, **inputs) is None
    output = warp_decode._run_nvfp4_explicit_tactic(
        inputs["x"],
        inputs["x_sf"],
        backend.w3_w1_weight,
        backend.w3_w1_weight_scale,
        backend.w2_weight,
        backend.w2_weight_scale,
        backend.fc31_alpha,
        backend.fc31_alpha,
        backend.fc2_alpha,
        inputs["token_selected_experts"],
        inputs["token_final_scales"],
        7168,
        2048,
        128,
        0,
        128,
        16,
    )

    assert output.shape == (2, 7168)


def test_warp_decode_explicit_tactic_views_stored_nvfp4_weights_as_bytes(monkeypatch):
    backend = _populate_nvfp4_backend(
        TRTLLMGenFusedMoE(), weight_storage_dtype=torch.int64)
    moe = _moe(backend=backend, has_nvfp4=True)
    inputs = _inputs(num_tokens=2)
    inputs["x"] = torch.empty((2, 3584), dtype=torch.uint8)
    inputs["x_sf"] = torch.empty((2, 448), dtype=torch.uint8)

    class _Runner:

        def run_moe(self, *args):
            assert args[4].dtype == torch.uint8
            assert args[4].shape == (128, 4096, 3584)
            assert args[10].dtype == torch.uint8
            assert args[10].shape == (128, 7168, 1024)
            assert args[-1].shape == (2, 7168)
            return [args[-1]]

    monkeypatch.setattr(warp_decode, "_nvfp4_torch_runner", lambda: _Runner())

    assert backend.w3_w1_weight.dtype == torch.int64
    assert backend.w3_w1_weight.shape[-1] == 448
    assert get_warp_decode_guard_failure(moe, **inputs) is None
    output = warp_decode._run_nvfp4_explicit_tactic(
        inputs["x"],
        inputs["x_sf"],
        backend.w3_w1_weight,
        backend.w3_w1_weight_scale,
        backend.w2_weight,
        backend.w2_weight_scale,
        backend.fc31_alpha,
        backend.fc31_alpha,
        backend.fc2_alpha,
        inputs["token_selected_experts"],
        inputs["token_final_scales"],
        7168,
        2048,
        128,
        0,
        128,
        16,
    )

    assert output.shape == (2, 7168)


def test_warp_decode_explicit_tactic_views_stored_nvfp4_scales_as_fp8(monkeypatch):
    backend = _populate_nvfp4_backend(
        TRTLLMGenFusedMoE(), scale_storage_dtype=torch.int64)
    moe = _moe(backend=backend, has_nvfp4=True)
    inputs = _inputs(num_tokens=2)
    inputs["x"] = torch.empty((2, 3584), dtype=torch.uint8)
    inputs["x_sf"] = torch.empty((2, 448), dtype=torch.uint8)

    class _Runner:

        def run_moe(self, *args):
            assert args[5].dtype == torch.float8_e4m3fn
            assert args[5].shape == (128, 4096, 448)
            assert args[11].dtype == torch.float8_e4m3fn
            assert args[11].shape == (128, 7168, 128)
            assert args[-1].shape == (2, 7168)
            return [args[-1]]

    monkeypatch.setattr(warp_decode, "_nvfp4_torch_runner", lambda: _Runner())

    assert backend.w3_w1_weight_scale.dtype == torch.int64
    assert backend.w3_w1_weight_scale.shape[-1] == 56
    assert get_warp_decode_guard_failure(moe, **inputs) is None
    output = warp_decode._run_nvfp4_explicit_tactic(
        inputs["x"],
        inputs["x_sf"],
        backend.w3_w1_weight,
        backend.w3_w1_weight_scale,
        backend.w2_weight,
        backend.w2_weight_scale,
        backend.fc31_alpha,
        backend.fc31_alpha,
        backend.fc2_alpha,
        inputs["token_selected_experts"],
        inputs["token_final_scales"],
        7168,
        2048,
        128,
        0,
        128,
        16,
    )

    assert output.shape == (2, 7168)


def test_warp_decode_allows_nvfp4_cuda_graph_bucket(monkeypatch):
    backend = _populate_nvfp4_backend(TRTLLMGenFusedMoE())
    moe = _moe(
        backend=backend,
        has_nvfp4=True,
        warp_decode_is_cuda_graph=True,
    )
    inputs = _inputs(num_tokens=8)
    inputs["x"] = torch.empty((8, 3584), dtype=torch.uint8)
    inputs["x_sf"] = torch.empty((8, 448), dtype=torch.uint8)

    monkeypatch.setattr(warp_decode, "_required_nvfp4_ops_available", lambda: True)

    assert get_warp_decode_guard_failure(moe, **inputs) is None


def test_warp_decode_nvfp4_allows_explicit_tactic_above_crossover(monkeypatch):
    backend = _populate_nvfp4_backend(TRTLLMGenFusedMoE())
    config = _config()
    config.max_batch_size = 64
    moe = _moe(
        backend=backend,
        has_nvfp4=True,
        model_config=types.SimpleNamespace(warp_decode_config=config),
    )
    inputs = _inputs(num_tokens=16)
    inputs["x"] = torch.empty((16, 3584), dtype=torch.uint8)
    inputs["x_sf"] = torch.empty((16, 448), dtype=torch.uint8)
    monkeypatch.setattr(warp_decode, "_get_nvfp4_op", lambda: object())

    assert get_warp_decode_guard_failure(moe, **inputs) is None


def test_warp_decode_selects_cursor_nvfp4_op_for_c32(monkeypatch):
    backend = _populate_nvfp4_backend(TRTLLMGenFusedMoE())
    config = _config()
    config.max_batch_size = 64
    moe = _moe(
        backend=backend,
        has_nvfp4=True,
        model_config=types.SimpleNamespace(warp_decode_config=config),
    )
    inputs = _inputs(num_tokens=32)
    inputs["x"] = torch.empty((32, 3584), dtype=torch.uint8)
    inputs["x_sf"] = torch.empty((32, 448), dtype=torch.uint8)

    def _op(*args):
        assert args[0] is inputs["x"]
        assert args[1] is inputs["x_sf"]
        assert args[9] is inputs["token_selected_experts"]
        assert args[10] is inputs["token_final_scales"]
        return torch.empty((32, 7168), dtype=torch.bfloat16)

    monkeypatch.setattr(warp_decode, "_get_nvfp4_cursor_op", lambda: _op)

    assert get_warp_decode_guard_failure(moe, **inputs) is None
    output = try_run_warp_decode(moe, **inputs)
    assert output.shape == (32, 7168)
    assert moe.warp_decode_last_status == "selected"
    assert moe.warp_decode_last_reason == "nvfp4_cursor_op"


def test_warp_decode_selects_explicit_nvfp4_op_for_c32_without_cursor(monkeypatch):
    backend = _populate_nvfp4_backend(TRTLLMGenFusedMoE())
    config = _config()
    config.max_batch_size = 64
    moe = _moe(
        backend=backend,
        has_nvfp4=True,
        model_config=types.SimpleNamespace(warp_decode_config=config),
    )
    inputs = _inputs(num_tokens=32)
    inputs["x"] = torch.empty((32, 3584), dtype=torch.uint8)
    inputs["x_sf"] = torch.empty((32, 448), dtype=torch.uint8)

    def _op(*args):
        assert args[0] is inputs["x"]
        assert args[1] is inputs["x_sf"]
        assert args[9] is inputs["token_selected_experts"]
        assert args[10] is inputs["token_final_scales"]
        return torch.empty((32, 7168), dtype=torch.bfloat16)

    monkeypatch.setattr(warp_decode, "_get_nvfp4_op", lambda: _op)
    monkeypatch.setattr(warp_decode, "_get_nvfp4_cursor_op", lambda: None)

    assert get_warp_decode_guard_failure(moe, **inputs) is None
    output = try_run_warp_decode(moe, **inputs)
    assert output.shape == (32, 7168)
    assert moe.warp_decode_last_status == "selected"
    assert moe.warp_decode_last_reason == "nvfp4_explicit_tactic_op"


def test_warp_decode_uses_explicit_nvfp4_autotune_above_cursor_buckets(monkeypatch):
    backend = _populate_nvfp4_backend(TRTLLMGenFusedMoE())
    config = _config()
    config.max_batch_size = 64
    moe = _moe(
        backend=backend,
        has_nvfp4=True,
        model_config=types.SimpleNamespace(warp_decode_config=config),
    )
    inputs = _inputs(num_tokens=128)
    inputs["x"] = torch.empty((128, 3584), dtype=torch.uint8)
    inputs["x_sf"] = torch.empty((128, 448), dtype=torch.uint8)

    def _op(*args):
        assert args[0] is inputs["x"]
        assert args[1] is inputs["x_sf"]
        assert args[9] is inputs["token_selected_experts"]
        assert args[10] is inputs["token_final_scales"]
        return torch.empty((128, 7168), dtype=torch.bfloat16)

    monkeypatch.setattr(warp_decode, "_get_nvfp4_cursor_op", lambda: object())
    monkeypatch.setattr(warp_decode, "_get_nvfp4_op", lambda: _op)

    assert get_warp_decode_guard_failure(moe, **inputs) is None
    output = try_run_warp_decode(moe, **inputs)
    assert output.shape == (128, 7168)
    assert moe.warp_decode_last_status == "selected"
    assert moe.warp_decode_last_reason == "nvfp4_explicit_tactic_op"


def test_warp_decode_rejects_bf16_batch_above_config_max():
    moe = _moe()
    inputs = _inputs(num_tokens=9)

    assert get_warp_decode_guard_failure(moe, **inputs) == "batch_too_large"


def test_warp_decode_allows_nvfp4_after_external_dispatch(monkeypatch):
    backend = _populate_nvfp4_backend(TRTLLMGenFusedMoE())
    moe = _moe(backend=backend, has_nvfp4=True, comm=object())
    inputs = _inputs(num_tokens=4)
    inputs["x"] = torch.empty((4, 3584), dtype=torch.uint8)
    inputs["x_sf"] = torch.empty((4, 448), dtype=torch.uint8)

    monkeypatch.setattr(warp_decode, "_required_nvfp4_ops_available", lambda: True)

    assert get_warp_decode_guard_failure(moe, **inputs) is None


def test_warp_decode_allows_nvfp4_eplb_slots(monkeypatch):
    backend = _populate_nvfp4_backend(TRTLLMGenFusedMoE())
    moe = _moe(
        backend=backend,
        has_nvfp4=True,
        layer_load_balancer=object(),
    )
    inputs = _inputs(num_tokens=4)
    inputs["x"] = torch.empty((4, 3584), dtype=torch.uint8)
    inputs["x_sf"] = torch.empty((4, 448), dtype=torch.uint8)

    monkeypatch.setattr(warp_decode, "_required_nvfp4_ops_available", lambda: True)

    assert get_warp_decode_guard_failure(moe, **inputs) is None


def test_warp_decode_rejects_incomplete_nvfp4_backend():
    backend = _populate_nvfp4_backend(TRTLLMGenFusedMoE(), weights=False)
    moe = _moe(backend=backend, has_nvfp4=True)
    inputs = _inputs()
    inputs["x"] = torch.empty((4, 3584), dtype=torch.uint8)
    inputs["x_sf"] = torch.empty((4, 448), dtype=torch.uint8)
    assert get_warp_decode_guard_failure(moe, **inputs) == (
        "nvfp4_missing_weights_or_scales")


def test_warp_decode_rejects_eplb_slots():
    moe = _moe(layer_load_balancer=object())
    assert get_warp_decode_guard_failure(moe, **_inputs()) == "eplb_not_supported"


def test_warp_decode_force_raises_exact_guard_reason():
    moe = _moe(
        model_config=types.SimpleNamespace(warp_decode_config=_config("force")),
        layer_load_balancer=object(),
    )
    with pytest.raises(NotImplementedError, match="eplb_not_supported"):
        try_run_warp_decode(moe, **_inputs())
    assert moe.warp_decode_last_status == "fallback"
    assert moe.warp_decode_last_reason == "eplb_not_supported"


def test_warp_decode_force_skips_context_or_warmup_batches():
    moe = _moe(
        model_config=types.SimpleNamespace(warp_decode_config=_config("force")),
        warp_decode_is_decode_only=False,
    )
    assert try_run_warp_decode(moe, **_inputs()) is None
    assert moe.warp_decode_last_status == "not_applicable"
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
        layer_load_balancer=object(),
    )
    with pytest.raises(NotImplementedError, match="eplb_not_supported"):
        try_run_warp_decode(moe, **_inputs())
    assert moe.warp_decode_last_status == "fallback"
    assert moe.warp_decode_last_reason == "eplb_not_supported"

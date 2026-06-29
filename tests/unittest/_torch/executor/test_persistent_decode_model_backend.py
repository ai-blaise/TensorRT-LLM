#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.persistent_decode_model_backend import (
    DeepSeekResidentModelBackendNativeV1,
    DeepSeekResidentModelBackendV0,
    DeepSeekResidentModelBackendPythonV1,
    PersistentDecodeModelForwardRequest,
    PersistentDecodeModelWindowRequest,
    PersistentDecodeModelWindowResult,
    build_deepseek_resident_body_contract,
    build_deepseek_resident_invocation_contract,
    build_persistent_decode_model_window_contract,
    create_persistent_decode_model_backend,
)
from tensorrt_llm._torch.pyexecutor import deepseek_resident_native


def test_deepseek_native_request_id_list_sanitizes_unsigned_dummies():
    request_ids = deepseek_resident_native._torchbind_request_id_list(
        (123, -1, (1 << 64) - 1, (1 << 64) - 100002))

    assert request_ids == [123, 0, 0, 0]


def test_deepseek_resident_window_cuda_graph_indexer_width_bucket():
    layer_plan = deepseek_resident_native.DeepSeekResidentDsaWindowLayerPlan(
        layer_idx=7,
        static_metadata_tensors=(),
        runtime_tensors={},
        runtime_config={"max_seq_len": 131072},
        runtime_scalars={},
        scratch_shapes=(),
    )

    assert deepseek_resident_native._dsa_window_indexer_widths(
        cached_tokens=(4161, 4163),
        input_tokens=2,
        owned_steps=16,
        dsa_window_plan=(layer_plan, ),
    ) == ((7, 8192), )
    assert deepseek_resident_native._dsa_window_indexer_widths(
        cached_tokens=(8191, ),
        input_tokens=1,
        owned_steps=16,
        dsa_window_plan=(layer_plan, ),
    ) == ((7, 16384), )


def test_deepseek_resident_window_cuda_graph_key_ignores_request_ids():
    engine = object.__new__(deepseek_resident_native.DeepSeekResidentNativeEngine)
    layer_plan = deepseek_resident_native.DeepSeekResidentDsaWindowLayerPlan(
        layer_idx=7,
        static_metadata_tensors=(),
        runtime_tensors={},
        runtime_config={"max_seq_len": 131072},
        runtime_scalars={},
        scratch_shapes=(),
    )
    payload = {
        "layer_indices": [7],
        "metadata_offsets": [0, 1],
        "metadata_tensors": [_fake_native_tensor()],
        "runtime_tensors": [{}],
        "runtime_config": [{"max_seq_len": 131072}],
        "runtime_scalars": [{}],
        "scratch_offsets": [0, 1],
        "scratch_tensors": [_fake_native_tensor()],
    }
    common = {
        "stable_shape_key": ("deepseek", 4),
        "input_tokens": 4,
        "cached_tokens": (4096, 4096, 4096, 4096),
    }
    invocation_a = SimpleNamespace(request_ids=(10, 11, 12, 13), **common)
    invocation_b = SimpleNamespace(
        request_ids=(
            (1 << 64) - 1,
            (1 << 64) - 2,
            (1 << 64) - 3,
            (1 << 64) - 4,
        ),
        **common,
    )
    tensors = {
        "initial_tokens": _fake_native_tensor((4, ), "torch.int64"),
        "input_ids": _fake_native_tensor((4, ), "torch.int64"),
        "hidden_states": _fake_native_tensor((4, 7168)),
        "logits": _fake_native_tensor((4, 129280)),
        "position_ids": _fake_native_tensor((4, ), "torch.int64"),
        "seq_lens_cuda": _fake_native_tensor((4, ), "torch.int32"),
        "kv_lens_cuda": _fake_native_tensor((4, ), "torch.int32"),
        "window_tokens": _fake_native_tensor((16, 4), "torch.int64"),
    }

    key_a = engine._window_cuda_graph_key(
        invocation=invocation_a,
        max_owned_steps=16,
        dsa_window_plan=(layer_plan, ),
        payload=payload,
        **tensors,
    )
    key_b = engine._window_cuda_graph_key(
        invocation=invocation_b,
        max_owned_steps=16,
        dsa_window_plan=(layer_plan, ),
        payload=payload,
        **tensors,
    )

    assert key_a == key_b


def test_deepseek_resident_window_cuda_graph_key_ignores_runtime_payload_pointers(
):
    engine = object.__new__(deepseek_resident_native.DeepSeekResidentNativeEngine)
    layer_plan = deepseek_resident_native.DeepSeekResidentDsaWindowLayerPlan(
        layer_idx=7,
        static_metadata_tensors=(),
        runtime_tensors={},
        runtime_config={"max_seq_len": 131072},
        runtime_scalars={},
        scratch_shapes=(),
    )
    payload_a = _cuda_graph_payload(metadata_ptr=11,
                                    runtime_ptr=12,
                                    scratch_ptr=13)
    payload_b = _cuda_graph_payload(metadata_ptr=11,
                                    runtime_ptr=22,
                                    scratch_ptr=13)
    invocation = SimpleNamespace(
        stable_shape_key=("deepseek", 4),
        input_tokens=4,
        cached_tokens=(4096, 4096, 4096, 4096),
        request_ids=(10, 11, 12, 13),
    )
    tensors = {
        "initial_tokens": _fake_native_tensor((4, ), "torch.int64"),
        "input_ids": _fake_native_tensor((4, ), "torch.int64"),
        "hidden_states": _fake_native_tensor((4, 7168)),
        "logits": _fake_native_tensor((4, 129280)),
        "position_ids": _fake_native_tensor((4, ), "torch.int64"),
        "seq_lens_cuda": _fake_native_tensor((4, ), "torch.int32"),
        "kv_lens_cuda": _fake_native_tensor((4, ), "torch.int32"),
        "window_tokens": _fake_native_tensor((16, 4), "torch.int64"),
    }

    key_a = engine._window_cuda_graph_key(
        invocation=invocation,
        max_owned_steps=16,
        dsa_window_plan=(layer_plan, ),
        payload=payload_a,
        **tensors,
    )
    key_b = engine._window_cuda_graph_key(
        invocation=invocation,
        max_owned_steps=16,
        dsa_window_plan=(layer_plan, ),
        payload=payload_b,
        **tensors,
    )

    assert key_a == key_b


def test_deepseek_resident_window_cuda_graph_static_payload_reuses_buffers():
    engine = object.__new__(deepseek_resident_native.DeepSeekResidentNativeEngine)
    graph_state = deepseek_resident_native.DeepSeekResidentWindowCudaGraphState(
        key=("test", ))
    payload_a = _cuda_graph_payload(metadata_ptr=11,
                                    runtime_ptr=12,
                                    scratch_ptr=13)
    payload_b = _cuda_graph_payload(metadata_ptr=21,
                                    runtime_ptr=22,
                                    scratch_ptr=23)

    static_a = engine._ensure_window_cuda_graph_static_payload(
        graph_state=graph_state,
        payload=payload_a,
    )
    static_b = engine._ensure_window_cuda_graph_static_payload(
        graph_state=graph_state,
        payload=payload_b,
    )

    assert graph_state.static_payload is not None
    assert static_a["runtime_tensors"][0]["kv_lens_runtime"] is (
        static_b["runtime_tensors"][0]["kv_lens_runtime"])
    assert static_b["metadata_tensors"][0] is (
        payload_b["metadata_tensors"][0])
    assert static_b["runtime_tensors"][0]["kv_lens_runtime"].copied_from is (
        payload_b["runtime_tensors"][0]["kv_lens_runtime"])
    assert static_b["scratch_tensors"][0] is payload_b["scratch_tensors"][0]


def _cuda_graph_payload(
    *,
    metadata_ptr: int,
    runtime_ptr: int,
    scratch_ptr: int,
):
    return {
        "layer_indices": [7],
        "metadata_offsets": [0, 1],
        "metadata_tensors": [
            _fake_native_tensor((4, ), "torch.int32", data_ptr=metadata_ptr)
        ],
        "runtime_tensors": [{
            "kv_lens_runtime":
            _fake_native_tensor((4, ), "torch.int32", data_ptr=runtime_ptr),
        }],
        "runtime_config": [{
            "max_seq_len": 131072
        }],
        "runtime_scalars": [{}],
        "scratch_offsets": [0, 1],
        "scratch_tensors": [
            _fake_native_tensor((16, 4), "torch.int32", data_ptr=scratch_ptr)
        ],
    }


def _forward_request(
    *,
    eligible: bool = True,
    reason: str = "eligible_decode_only_deepseek_batch",
    can_run_graph: bool = True,
    real_generation_requests: int = 8,
    padded_generation_requests: int | None = None,
    request_ids: tuple[int, ...] | None = None,
    model=None,
    preprocess_inputs=None,
    model_forward=None,
    without_logits: bool = False,
):
    if preprocess_inputs is None:
        preprocess_inputs = lambda inputs: inputs
    if model_forward is None:
        model_forward = lambda **kwargs: {"logits": "fake_logits"}
    request_ids = tuple(range(100, 108)) if request_ids is None else request_ids
    padded_generation_requests = (
        len(request_ids)
        if padded_generation_requests is None else padded_generation_requests)
    seq_lens = (1, ) * len(request_ids)
    cached_tokens = tuple(range(2048, 2048 + len(request_ids)))
    attn_metadata = SimpleNamespace(
        request_ids=request_ids,
        seq_lens=seq_lens,
        kv_cache_params=SimpleNamespace(
            num_cached_tokens_per_seq=cached_tokens),
    )
    return PersistentDecodeModelForwardRequest(
        plan=SimpleNamespace(
            eligible=eligible,
            reason=reason,
            real_generation_requests=real_generation_requests,
            padded_generation_requests=padded_generation_requests,
            request_ids=request_ids,
            seq_lens=seq_lens,
            cached_tokens=cached_tokens,
            input_tokens=len(request_ids),
            cuda_graph_replay=can_run_graph,
            cuda_graph_padding=False,
            attention_metadata_type="DSAtrtllmAttentionMetadata",
            kv_cache_manager_type="DSACacheManager",
        ),
        real_requests=SimpleNamespace(),
        padded_requests=SimpleNamespace(),
        inputs={
            "input_ids":
            _fake_tensor((len(request_ids), ), "torch.int64", "cuda:0"),
            "attn_metadata": object(),
        },
        gather_ids=None,
        attn_metadata=attn_metadata,
        spec_metadata=None,
        kv_cache_manager=object(),
        draft_kv_cache_manager=None,
        resource_manager=object(),
        model=model if model is not None else _fake_deepseek_model(),
        cuda_graph_key=(1, 8),
        can_run_graph=can_run_graph,
        gather_context_logits=False,
        preprocess_inputs=preprocess_inputs,
        model_forward=model_forward,
        without_logits=without_logits,
    )


def _fake_tensor(shape, dtype, device):
    return SimpleNamespace(shape=shape, dtype=dtype, device=device)


class _FakeNativeTensor:

    def __init__(self,
                 shape=(1, ),
                 dtype="torch.bfloat16",
                 device="cuda:0",
                 data_ptr=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self._data_ptr = id(self) if data_ptr is None else int(data_ptr)

    def stride(self):
        stride = []
        running = 1
        for dim in reversed(self.shape):
            stride.insert(0, running)
            running *= int(dim)
        return tuple(stride)

    def data_ptr(self):
        return self._data_ptr

    def new_empty(self, shape):
        return _FakeNativeTensor(shape=shape,
                                 dtype=self.dtype,
                                 device=self.device)

    def copy_(self, other):
        self.copied_from = other
        return self


def _fake_native_tensor(shape=(1, ),
                        dtype="torch.bfloat16",
                        device="cuda:0",
                        data_ptr=None):
    return _FakeNativeTensor(shape=shape,
                             dtype=dtype,
                             device=device,
                             data_ptr=data_ptr)


def _fake_linear(out_features=16):
    return SimpleNamespace(weight=SimpleNamespace(shape=(out_features, 128)))


def _fake_native_weight_module(shape=(1, )):
    return SimpleNamespace(weight=_fake_native_tensor(shape))


def _fake_native_indexer():
    return SimpleNamespace(
        skip_topk=False,
        n_heads=64,
        head_dim=128,
        rope_dim=64,
        index_topk=2048,
        wq_b=_fake_native_weight_module((8192, 512)),
        wk=_fake_native_weight_module((128, 7168)),
        weights_proj=_fake_native_weight_module((64, 7168)),
        k_norm=_fake_native_weight_module((128, )),
        rotary_emb=SimpleNamespace(
            rotary_cos_sin=_fake_native_tensor((4096, 64),
                                               dtype="torch.float32")),
    )


def _fake_native_resident_model():
    layer = SimpleNamespace(
        input_layernorm=_fake_native_weight_module((7168, )),
        post_attention_layernorm=_fake_native_weight_module((7168, )),
        next_layer_layernorm=_fake_native_weight_module((7168, )),
        input_gated_norm_down=_fake_native_weight_module((16, 7168)),
        input_gated_norm_up=_fake_native_weight_module((7168, 16)),
        post_attention_gated_norm_down=_fake_native_weight_module((16, 7168)),
        post_attention_gated_norm_up=_fake_native_weight_module((7168, 16)),
        self_attn=SimpleNamespace(
            kv_a_proj_with_mqa=_fake_native_weight_module((576, 7168)),
            k_b_proj_trans=_fake_native_tensor((128, 512, 128)),
            v_b_proj=_fake_native_tensor((128, 128, 512)),
            gate_proj=_fake_native_weight_module((16384, 7168)),
            o_proj=_fake_native_weight_module((7168, 16384)),
            mqa=SimpleNamespace(indexer=_fake_native_indexer()),
        ),
        mlp=SimpleNamespace(
            gate_up_proj=_fake_native_weight_module((4096, 7168)),
            down_proj=_fake_native_weight_module((7168, 2048)),
        ),
    )
    first_parameter = _fake_native_tensor((7168, ))
    return SimpleNamespace(
        parameters=lambda: iter((first_parameter, )),
        model=SimpleNamespace(
            embed_tokens=_fake_native_weight_module((129280, 7168)),
            norm=_fake_native_weight_module((7168, )),
            layers=(layer, ),
        ),
        lm_head=_fake_native_weight_module((129280, 7168)),
    )


def _fake_native_resident_model_with_moe():
    dense_layer = _fake_native_resident_model().model.layers[0]
    moe_layer = SimpleNamespace(
        input_layernorm=_fake_native_weight_module((7168, )),
        post_attention_layernorm=_fake_native_weight_module((7168, )),
        next_layer_layernorm=_fake_native_weight_module((7168, )),
        input_gated_norm_down=_fake_native_weight_module((16, 7168)),
        input_gated_norm_up=_fake_native_weight_module((7168, 16)),
        post_attention_gated_norm_down=_fake_native_weight_module(
            (16, 7168)),
        post_attention_gated_norm_up=_fake_native_weight_module((7168, 16)),
        self_attn=SimpleNamespace(
            kv_a_proj_with_mqa=_fake_native_weight_module((576, 7168)),
            k_b_proj_trans=_fake_native_tensor((128, 512, 128)),
            v_b_proj=_fake_native_tensor((128, 128, 512)),
            gate_proj=_fake_native_weight_module((16384, 7168)),
            o_proj=_fake_native_weight_module((7168, 16384)),
            mqa=SimpleNamespace(indexer=_fake_native_indexer()),
        ),
        mlp=SimpleNamespace(gate=SimpleNamespace(
            weight=_fake_native_tensor((128, 7168)),
            e_score_correction_bias=_fake_native_tensor(
                (128, ), dtype="torch.float32"),
        )),
    )
    first_parameter = _fake_native_tensor((7168, ))
    return SimpleNamespace(
        parameters=lambda: iter((first_parameter, )),
        model=SimpleNamespace(
            embed_tokens=_fake_native_weight_module((129280, 7168)),
            norm=_fake_native_weight_module((7168, )),
            layers=(dense_layer, moe_layer),
        ),
        lm_head=_fake_native_weight_module((129280, 7168)),
    )


def _fake_layer(layer_idx: int, *, missing_attention_gate: bool = False):
    gate_proj = None if missing_attention_gate else object()
    return SimpleNamespace(
        self_attn=SimpleNamespace(
            gate_proj=gate_proj,
            kv_a_proj_with_mqa=object(),
        ),
        mlp=SimpleNamespace() if layer_idx < 3 else type("FakeMoE", (), {})(),
        input_layernorm=object(),
        post_attention_layernorm=object(),
        next_layer_layernorm=object(),
        input_gated_norm_down=_fake_linear(16),
        input_gated_norm_up=_fake_linear(128),
        post_attention_gated_norm_down=_fake_linear(16),
        post_attention_gated_norm_up=_fake_linear(128),
        fusion_config=SimpleNamespace(
            PRE_MLP_FUSION=False,
            PRE_MOE_FUSION=False,
            POST_MLP_FUSION=False,
            POST_MOE_FUSION=False,
        ),
        disable_attn_allreduce=True,
        num_experts=128,
        top_k=8,
    )


def _fake_deepseek_model(*, missing_attention_gate: bool = False):
    config = SimpleNamespace(
        model_type="deepseek_v32",
        architectures=("DeepseekV32ForCausalLM", ),
        num_hidden_layers=4,
        hidden_size=7168,
        vocab_size=129280,
        rms_norm_eps=1e-6,
        gated_norm=True,
        attention_output_gate=True,
        n_routed_experts=128,
        num_experts_per_tok=8,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        first_k_dense_replace=3,
        moe_layer_freq=1,
    )
    layers = [
        _fake_layer(idx, missing_attention_gate=missing_attention_gate)
        for idx in range(config.num_hidden_layers)
    ]
    return SimpleNamespace(
        model_config=SimpleNamespace(pretrained_config=config),
        model=SimpleNamespace(layers=layers),
    )


def _fake_window_sample_state(*, request_count: int = 8):
    requests = []
    for request_id in range(100, 100 + request_count):
        requests.append(
            SimpleNamespace(
                py_request_id=request_id,
                py_seq_slot=request_id - 100,
                py_max_new_tokens=128,
                py_decoding_iter=8,
                py_beam_width=1,
                streaming=False,
                py_return_log_probs=False,
                py_return_generation_logits=False,
                py_stop_words_list=None,
                py_end_id=-1,
                py_draft_tokens=None,
                state=SimpleNamespace(name="GENERATION_IN_PROGRESS"),
                is_attention_dp_dummy=False,
                is_cuda_graph_dummy=False,
                is_dummy_request=False,
            ))
    return SimpleNamespace(requests=tuple(requests))


def _fake_dummy_window_sample_state():
    return SimpleNamespace(requests=(SimpleNamespace(
        py_request_id=0,
        py_seq_slot=0,
        py_max_new_tokens=0,
        py_decoding_iter=0,
        py_beam_width=1,
        streaming=False,
        py_return_log_probs=False,
        py_return_generation_logits=False,
        py_stop_words_list=None,
        py_end_id=-1,
        py_draft_tokens=None,
        state=SimpleNamespace(name="GENERATION_IN_PROGRESS"),
        is_attention_dp_dummy=True,
        is_cuda_graph_dummy=False,
        is_dummy_request=False,
    ), ))


def _fake_mixed_adp_window_sample_state():
    real_request = SimpleNamespace(
        py_request_id=100,
        py_seq_slot=0,
        py_max_new_tokens=128,
        py_decoding_iter=8,
        py_beam_width=1,
        streaming=False,
        py_return_log_probs=False,
        py_return_generation_logits=False,
        py_stop_words_list=None,
        py_end_id=-1,
        py_draft_tokens=None,
        state=SimpleNamespace(name="GENERATION_IN_PROGRESS"),
        is_attention_dp_dummy=False,
        is_cuda_graph_dummy=False,
        is_dummy_request=False,
    )
    dummy_request = SimpleNamespace(
        py_request_id=0,
        py_seq_slot=1,
        py_max_new_tokens=0,
        py_decoding_iter=0,
        py_beam_width=1,
        streaming=False,
        py_return_log_probs=False,
        py_return_generation_logits=False,
        py_stop_words_list=None,
        py_end_id=-1,
        py_draft_tokens=None,
        state=SimpleNamespace(name="GENERATION_IN_PROGRESS"),
        is_attention_dp_dummy=True,
        is_cuda_graph_dummy=False,
        is_dummy_request=False,
    )
    return SimpleNamespace(requests=(real_request, dummy_request))


def test_model_backend_disabled_by_default(monkeypatch):
    monkeypatch.delenv("TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND",
                       raising=False)

    backend = create_persistent_decode_model_backend(
        SimpleNamespace(rank=0, tp_rank=0))

    assert backend is None


def test_model_backend_factory_rejects_unknown(monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND",
                       "unknown")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND_RANKS",
                       "0")

    with pytest.raises(ValueError, match="Unsupported"):
        create_persistent_decode_model_backend(
            SimpleNamespace(rank=0, tp_rank=0))


def test_deepseek_resident_v0_declines_until_body_exists(monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND",
                       "deepseek_resident_v0")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND_RANKS",
                       "0")
    backend = create_persistent_decode_model_backend(
        SimpleNamespace(rank=0, tp_rank=0))

    assert isinstance(backend, DeepSeekResidentModelBackendV0)
    assert backend.try_execute(_forward_request()) is None
    assert (
        backend._reasons[
            "resident_model_body_contract_ready_not_implemented"] == 1)


def test_deepseek_resident_v0_keeps_plan_rejection_reason(monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND",
                       "deepseek_resident_v0")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND_RANKS",
                       "0")
    backend = create_persistent_decode_model_backend(
        SimpleNamespace(rank=0, tp_rank=0))

    assert isinstance(backend, DeepSeekResidentModelBackendV0)
    assert backend.try_execute(
        _forward_request(
            eligible=False,
            reason="context_requests_present",
        )) is None
    assert backend._reasons["context_requests_present"] == 1


def test_deepseek_resident_python_v1_executes_contract_ready_body(monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND",
                       "deepseek_resident_python_v1")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND_RANKS",
                       "0")
    backend = create_persistent_decode_model_backend(
        SimpleNamespace(rank=0, tp_rank=0))
    calls = []

    def preprocess_inputs(inputs):
        calls.append(("preprocess", inputs["input_ids"]))
        return {
            **inputs,
            "preprocessed": True,
        }

    def model_forward(**kwargs):
        calls.append(("model_forward", kwargs["preprocessed"],
                      kwargs["return_context_logits"]))
        return "fake_logits"

    assert isinstance(backend, DeepSeekResidentModelBackendPythonV1)
    result = backend.try_execute(
        _forward_request(
            preprocess_inputs=preprocess_inputs,
            model_forward=model_forward,
        ))

    assert result is not None
    assert result.backend == "deepseek_resident_python_v1"
    assert result.outputs == {"logits": "fake_logits"}
    assert calls[0][0] == "preprocess"
    assert calls[1] == ("model_forward", True, False)
    assert backend._reasons["resident_model_body_python_v1_executed"] == 1


def test_deepseek_resident_native_v1_falls_back_without_native_module(
        monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND",
                       "deepseek_resident_native_v1")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND_RANKS",
                       "0")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_NATIVE_MODULE",
                       "missing_native_backend_for_test")
    backend = create_persistent_decode_model_backend(
        SimpleNamespace(rank=0, tp_rank=0))

    assert isinstance(backend, DeepSeekResidentModelBackendNativeV1)
    assert backend.try_execute(_forward_request()) is None
    assert (backend._reasons[
        "resident_native_engine_module_import_failed:ModuleNotFoundError"] == 1)


def test_deepseek_resident_native_v1_executes_loaded_engine(monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND",
                       "deepseek_resident_native_v1")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND_RANKS",
                       "0")
    module_name = "fake_native_backend_for_test"
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_NATIVE_MODULE",
                       module_name)
    captured = {}

    class FakeNativeEngine:

        def execute(self, **kwargs):
            captured.update(kwargs)
            return {"logits": "native_logits"}

    module = SimpleNamespace(create_engine=lambda **kwargs: FakeNativeEngine())
    monkeypatch.setitem(sys.modules, module_name, module)
    backend = create_persistent_decode_model_backend(
        SimpleNamespace(rank=0, tp_rank=0))

    result = backend.try_execute(_forward_request())

    assert result is not None
    assert result.backend == "deepseek_resident_native_v1"
    assert result.outputs == {"logits": "native_logits"}
    assert captured["invocation"].real_batch_size == 8
    assert captured["invocation"].padded_batch_size == 8
    assert captured["invocation"].request_ids == tuple(range(100, 108))
    assert captured["invocation"].input_tensor_specs[0].name == "input_ids"
    assert backend._reasons["resident_native_engine_executed"] == 1


def test_deepseek_resident_native_v1_respects_min_local_batch(monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND",
                       "deepseek_resident_native_v1")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND_RANKS",
                       "0")
    monkeypatch.setenv("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MIN_LOCAL_BATCH", "4")
    module_name = "fake_native_min_batch_backend_for_test"
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_NATIVE_MODULE",
                       module_name)

    class FakeNativeEngine:

        def execute(self, **kwargs):
            raise AssertionError("native engine should not run")

    module = SimpleNamespace(create_engine=lambda **kwargs: FakeNativeEngine())
    monkeypatch.setitem(sys.modules, module_name, module)
    backend = create_persistent_decode_model_backend(
        SimpleNamespace(rank=0, tp_rank=0))

    result = backend.try_execute(
        _forward_request(
            real_generation_requests=1,
            request_ids=(100, ),
        ))

    assert result is None
    assert (backend._reasons[
        "resident_native_min_local_batch_not_met:1<4"] == 1)


def test_deepseek_resident_v0_declines_window_execution(monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND",
                       "deepseek_resident_v0")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND_RANKS",
                       "0")
    backend = create_persistent_decode_model_backend(
        SimpleNamespace(rank=0, tp_rank=0))
    forward_request = _forward_request()

    assert isinstance(backend, DeepSeekResidentModelBackendV0)
    assert backend.window_backend_state() == {
        "backend": None,
        "ready": False,
        "reason": "resident_window_model_backend_not_native",
        "metadata": {
            "model_backend": "deepseek_resident_v0",
        },
    }
    assert backend.try_execute_window(
        PersistentDecodeModelWindowRequest(
            forward_request=forward_request,
            sample_state=_fake_window_sample_state(),
            requested_window_steps=4,
        )) is None
    assert backend._reasons["resident_window_model_backend_not_native"] == 1


def test_deepseek_resident_native_v1_delegates_window_to_loaded_engine(
        monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND",
                       "deepseek_resident_native_v1")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND_RANKS",
                       "0")
    module_name = "fake_native_window_backend_for_test"
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_NATIVE_MODULE",
                       module_name)
    captured = {}

    class FakeNativeEngine:

        def execute(self, **kwargs):
            return {"logits": "unused_logits"}

        def window_backend_state(self):
            return {
                "backend": "deepseek_resident_window_native_v1",
                "ready": True,
                "reason": "resident_window_native_ready",
                "metadata": {
                    "fake": True,
                },
            }

        def execute_window(self, **kwargs):
            captured.update(kwargs)
            window_request = kwargs["request"]
            return PersistentDecodeModelWindowResult(
                sample_state="native_window_sample_state",
                owned_steps=3,
                requested_window_steps=(
                    window_request.requested_window_steps),
                break_reason="resident_window_native_executed",
            )

    module = SimpleNamespace(create_engine=lambda **kwargs: FakeNativeEngine())
    monkeypatch.setitem(sys.modules, module_name, module)
    backend = create_persistent_decode_model_backend(
        SimpleNamespace(rank=0, tp_rank=0))
    forward_request = _forward_request()
    sample_state = _fake_window_sample_state()

    assert isinstance(backend, DeepSeekResidentModelBackendNativeV1)
    assert backend.window_backend_state() == {
        "backend": "deepseek_resident_window_native_v1",
        "ready": False,
        "reason": "resident_window_native_engine_not_created",
        "metadata": {
            "model_backend": "deepseek_resident_native_v1",
            "native_engine_rejection": None,
        },
    }

    result = backend.try_execute_window(
        PersistentDecodeModelWindowRequest(
            forward_request=forward_request,
            sample_state=sample_state,
            requested_window_steps=4,
        ))

    assert result is not None
    assert result.sample_state == "native_window_sample_state"
    assert result.owned_steps == 3
    assert result.requested_window_steps == 4
    assert result.break_reason == "resident_window_native_executed"
    assert captured["request"].sample_state is sample_state
    assert captured["window_contract"].request_ids == tuple(range(100, 108))
    assert captured["window_contract"].seq_slots == tuple(range(8))
    assert captured["window_contract"].max_safe_owned_steps == 4
    assert captured["invocation"].real_batch_size == 8
    assert captured["invocation"].padded_batch_size == 8
    assert captured["inputs"]["input_ids"].shape == (8, )
    assert backend.window_backend_state()["ready"]
    assert backend._reasons["resident_window_native_executed"] == 1


def test_deepseek_resident_native_v1_window_respects_min_local_batch(
        monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND",
                       "deepseek_resident_native_v1")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND_RANKS",
                       "0")
    monkeypatch.setenv("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MIN_LOCAL_BATCH", "4")
    module_name = "fake_native_window_min_batch_backend_for_test"
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_NATIVE_MODULE",
                       module_name)

    class FakeNativeEngine:

        def execute(self, **kwargs):
            return {"logits": "unused_logits"}

        def execute_window(self, **kwargs):
            raise AssertionError("native window should not run")

    module = SimpleNamespace(create_engine=lambda **kwargs: FakeNativeEngine())
    monkeypatch.setitem(sys.modules, module_name, module)
    backend = create_persistent_decode_model_backend(
        SimpleNamespace(rank=0, tp_rank=0))

    result = backend.try_execute_window(
        PersistentDecodeModelWindowRequest(
            forward_request=_forward_request(
                real_generation_requests=1,
                request_ids=(100, ),
            ),
            sample_state=_fake_window_sample_state(request_count=1),
            requested_window_steps=4,
        ))

    assert result is not None
    assert result.executed is False
    assert result.break_reason == "resident_native_min_local_batch_not_met:1<4"
    assert (backend._reasons[
        "resident_native_min_local_batch_not_met:1<4"] == 1)


def test_deepseek_resident_native_v1_window_allows_dummy_only_rank(
        monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND",
                       "deepseek_resident_native_v1")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND_RANKS",
                       "0")
    monkeypatch.setenv("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MIN_LOCAL_BATCH", "1")
    module_name = "fake_native_window_dummy_only_rank_backend_for_test"
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_NATIVE_MODULE",
                       module_name)

    captured = {}

    class FakeNativeEngine:

        def execute(self, **kwargs):
            return {"logits": "unused_logits"}

        def execute_window(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                sample_state="dummy_rank_native_window_sample_state",
                owned_steps=3,
                requested_window_steps=4,
                break_reason="resident_window_native_executed",
                executed=True,
            )

    module = SimpleNamespace(create_engine=lambda **kwargs: FakeNativeEngine())
    monkeypatch.setitem(sys.modules, module_name, module)
    backend = create_persistent_decode_model_backend(
        SimpleNamespace(rank=0, tp_rank=0))

    result = backend.try_execute_window(
        PersistentDecodeModelWindowRequest(
            forward_request=_forward_request(
                real_generation_requests=0,
                padded_generation_requests=1,
                request_ids=(0, ),
            ),
            sample_state=_fake_dummy_window_sample_state(),
            requested_window_steps=4,
        ))

    assert result is not None
    assert result.executed is True
    assert result.sample_state == "dummy_rank_native_window_sample_state"
    assert result.break_reason == "resident_window_native_executed"
    assert captured["invocation"].real_batch_size == 0
    assert captured["invocation"].padded_batch_size == 1
    assert captured["window_contract"].request_ids == (0, )


def test_resident_window_contract_allows_adp_dummy_only_rank():
    forward_request = _forward_request(
        real_generation_requests=0,
        padded_generation_requests=1,
        request_ids=(0, ),
    )
    invocation, invocation_rejection = (
        build_deepseek_resident_invocation_contract(forward_request))
    assert invocation_rejection is None
    assert invocation is not None

    window_contract, window_rejection = (
        build_persistent_decode_model_window_contract(
            PersistentDecodeModelWindowRequest(
                forward_request=forward_request,
                sample_state=_fake_dummy_window_sample_state(),
                requested_window_steps=128,
            ),
            invocation,
        ))

    assert window_rejection is None
    assert window_contract is not None
    assert window_contract.request_ids == (0, )
    assert window_contract.max_safe_owned_steps == 128
    assert window_contract.sequences[0].remaining_decode_steps == 129


@pytest.mark.parametrize("padding_request_id", (0, -1, (1 << 64) - 1))
def test_resident_window_contract_ignores_adp_dummy_request_ids_for_real_rank(
        padding_request_id):
    forward_request = _forward_request(
        real_generation_requests=1,
        padded_generation_requests=2,
        request_ids=(100, padding_request_id),
    )
    invocation, invocation_rejection = (
        build_deepseek_resident_invocation_contract(forward_request))
    assert invocation_rejection is None
    assert invocation is not None

    window_contract, window_rejection = (
        build_persistent_decode_model_window_contract(
            PersistentDecodeModelWindowRequest(
                forward_request=forward_request,
                sample_state=_fake_mixed_adp_window_sample_state(),
                requested_window_steps=128,
            ),
            invocation,
        ))

    assert window_rejection is None
    assert window_contract is not None
    assert window_contract.request_ids == (100, )
    assert window_contract.seq_lens == (1, )
    assert window_contract.cached_tokens == (2048, )
    assert window_contract.seq_slots == (0, 1)
    assert len(window_contract.sequences) == 2
    assert window_contract.sequences[1].remaining_decode_steps == 129
    assert window_contract.max_safe_owned_steps == 119


def test_resident_window_contract_reports_request_id_mismatch_details():
    forward_request = _forward_request(
        real_generation_requests=2,
        padded_generation_requests=2,
        request_ids=(100, 101),
    )
    invocation, invocation_rejection = (
        build_deepseek_resident_invocation_contract(forward_request))
    assert invocation_rejection is None
    assert invocation is not None

    window_contract, window_rejection = (
        build_persistent_decode_model_window_contract(
            PersistentDecodeModelWindowRequest(
                forward_request=forward_request,
                sample_state=_fake_window_sample_state(request_count=1),
                requested_window_steps=128,
            ),
            invocation,
        ))

    assert window_contract is None
    assert window_rejection == (
        "request_ids_mismatch:invocation=[100,101;n=2],"
        "invocation_real=[100,101;n=2],request=[100;n=1],"
        "all=[100;n=1]")


def test_resident_window_contract_can_own_terminal_tail(monkeypatch):
    monkeypatch.setenv(
        "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_TERMINAL_WINDOW",
        "1")
    forward_request = _forward_request(
        real_generation_requests=1,
        request_ids=(100, ),
    )
    invocation, invocation_rejection = (
        build_deepseek_resident_invocation_contract(forward_request))
    assert invocation_rejection is None
    assert invocation is not None

    sample_state = _fake_window_sample_state(request_count=1)
    sample_state.requests[0].py_max_new_tokens = 32
    sample_state.requests[0].py_decoding_iter = 3
    window_contract, window_rejection = (
        build_persistent_decode_model_window_contract(
            PersistentDecodeModelWindowRequest(
                forward_request=forward_request,
                sample_state=sample_state,
                requested_window_steps=128,
            ),
            invocation,
        ))

    assert window_rejection is None
    assert window_contract is not None
    assert window_contract.max_safe_owned_steps == 29
    assert window_contract.sequences[0].remaining_decode_steps == 29


def test_deepseek_resident_native_engine_window_stub_declines():
    contract = SimpleNamespace(
        hidden_size=7168,
        vocab_size=129280,
        layers=(SimpleNamespace(layer_idx=0, layer_kind="dense"), ),
    )
    engine = deepseek_resident_native.create_engine(
        model=_fake_native_resident_model(),
        contract=contract,
        dist=None,
    )

    state = engine.window_backend_state()
    assert state["backend"] == "deepseek_resident_window_native_v1"
    assert state["ready"] is False
    assert state["reason"] == "resident_window_native_not_implemented"
    assert state["metadata"]["native"] is True
    assert state["metadata"]["native_window_ready"] is False
    assert state["metadata"]["window_stage_scheduler_ready"] is False
    assert state["metadata"]["window_stage_calls"] == 0
    assert state["metadata"]["window_scheduler_calls"] == 0
    assert state["metadata"]["stage_scheduler_calls"] == 0
    native_contract = state["metadata"]["native_window_contract"]
    assert native_contract["ready"] is False
    assert native_contract["first_missing_component"] == "native_window_handle"
    assert native_contract["layer_counts"] == {
        "total": 1,
        "dense": 1,
        "moe": 0,
    }
    assert native_contract["asset_counts"]["resident_tensors"] > 0
    assert engine.execute_window(
        request=None,
        contract=contract,
        invocation=None,
        window_contract=None,
        inputs={},
    ) is None
    state = engine.window_backend_state()
    assert state["ready"] is False
    assert state["reason"] == "resident_window_sample_state_factory_missing"
    assert state["metadata"]["window_stage_calls"] == 1


def test_deepseek_resident_python_v1_falls_back_when_contract_rejects(
        monkeypatch):
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND",
                       "deepseek_resident_python_v1")
    monkeypatch.setenv("TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND_RANKS",
                       "0")
    backend = create_persistent_decode_model_backend(
        SimpleNamespace(rank=0, tp_rank=0))

    result = backend.try_execute(
        _forward_request(model=_fake_deepseek_model(
            missing_attention_gate=True)))

    assert result is None
    assert (backend._reasons[
        "resident_model_contract_layer_0_missing_attention_output_gate"] == 1)


def test_deepseek_resident_contract_summarizes_layer_stack():
    contract, rejection = build_deepseek_resident_body_contract(
        _fake_deepseek_model())

    assert rejection is None
    assert contract is not None
    assert contract.num_layers == 4
    assert contract.rms_norm_eps == 1e-6
    assert contract.layer_kind_hist == {
        "dense": 3,
        "moe": 1,
    }
    assert contract.layers[0].has_attention_output_gate
    assert contract.layers[3].num_experts == 128
    assert contract.layers[3].top_k == 8


def test_deepseek_resident_invocation_contract_captures_shape_key():
    invocation, rejection = build_deepseek_resident_invocation_contract(
        _forward_request())

    assert rejection is None
    assert invocation is not None
    assert invocation.real_batch_size == 8
    assert invocation.padded_batch_size == 8
    assert invocation.cached_tokens == tuple(range(2048, 2056))
    assert invocation.input_tensor_specs[0].shape == (8, )
    assert invocation.stable_shape_key[0] == 8


def test_deepseek_resident_invocation_contract_defaults_decode_seq_lens():
    forward_request = _forward_request(
        real_generation_requests=2,
        padded_generation_requests=4,
        request_ids=(100, 101, 0, (1 << 64) - 1),
    )
    delattr(forward_request.plan, "seq_lens")
    delattr(forward_request.attn_metadata, "seq_lens")

    invocation, rejection = build_deepseek_resident_invocation_contract(
        forward_request)

    assert rejection is None
    assert invocation is not None
    assert invocation.seq_lens == (1, 1, 1, 1)


def test_deepseek_native_engine_reuses_shape_scratch(monkeypatch):
    calls = []
    prepare_calls = []

    class FakeTensor:

        def __init__(self, shape, *, dtype="torch.bfloat16", device="cuda:0"):
            self.shape = shape
            self.dtype = dtype
            self.device = device

    def fake_empty(shape, *, device=None, dtype=None):
        return FakeTensor(tuple(shape), dtype=dtype, device=device)

    def fake_native_op(*args):
        calls.append(args)
        return args[2]

    def fake_prepare_op(*args):
        prepare_calls.append(args)
        return True

    fake_torch = SimpleNamespace(
        empty=fake_empty,
        ops=SimpleNamespace(trtllm=SimpleNamespace(
            deepseek_resident_decode=fake_native_op,
            deepseek_resident_decode_prepare=fake_prepare_op,
        )),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("TRTLLM_OPTRT_DEEPSEEK_RESIDENT_VALIDATE_MANIFEST",
                       "1")

    contract = SimpleNamespace(
        hidden_size=7168,
        vocab_size=129280,
        layers=(SimpleNamespace(layer_idx=0, layer_kind="dense"), ),
    )
    engine = deepseek_resident_native.create_engine(
        model=_fake_native_resident_model(),
        contract=contract,
        dist=None,
    )
    invocation = SimpleNamespace(
        stable_shape_key=("shape", 16),
        real_batch_size=16,
        padded_batch_size=16,
        input_tokens=16,
        request_ids=tuple(range(16)),
        seq_lens=(1, ) * 16,
        cached_tokens=tuple(range(2048, 2064)),
    )
    input_ids = FakeTensor((16, ), dtype="torch.int64")

    first = engine.execute(
        request=None,
        contract=contract,
        invocation=invocation,
        inputs={"input_ids": input_ids},
    )
    second = engine.execute(
        request=None,
        contract=contract,
        invocation=invocation,
        inputs={"input_ids": input_ids},
    )

    assert first["logits"].shape == (16, 129280)
    assert first["logits"] is second["logits"]
    assert len(calls) == 2
    assert calls[0][3:6] == (16, 16, 16)
    assert calls[0][6] == list(range(16))
    assert calls[0][8] == list(range(2048, 2064))
    assert calls[0][9] == [3, 22]
    assert calls[0][10] == [0]
    assert len(calls[0][11]) == 22
    assert len(prepare_calls) == 1
    assert prepare_calls[0][0] == [3, 22]
    assert prepare_calls[0][1] == [0]
    assert len(prepare_calls[0][2]) == 22


def test_deepseek_native_window_token_scratch_reuses_capacity(monkeypatch):
    allocations = []

    class FakeTensor:

        def __init__(self, shape, *, dtype="torch.bfloat16", device="cuda:0"):
            self.shape = tuple(shape)
            self.dtype = dtype
            self.device = device

    def fake_empty(shape, *, device=None, dtype=None):
        tensor = FakeTensor(tuple(shape), dtype=dtype, device=device)
        allocations.append(tensor)
        return tensor

    fake_torch = SimpleNamespace(empty=fake_empty, int32="torch.int32")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    engine = object.__new__(deepseek_resident_native.DeepSeekResidentNativeEngine)
    state = deepseek_resident_native.DeepSeekResidentShapeState(
        shape_key=("shape", 4),
        batch_size=4,
        hidden_size=7168,
        vocab_size=129280,
        device="cuda:0",
        dtype="torch.bfloat16",
    )

    large = engine._ensure_window_new_tokens_scratch(state, 121)
    smaller = engine._ensure_window_new_tokens_scratch(state, 117)
    larger = engine._ensure_window_new_tokens_scratch(state, 125)

    assert smaller is large
    assert larger is not large
    assert large.shape == (121, 4, 1)
    assert larger.shape == (125, 4, 1)
    assert len(allocations) == 2


def test_deepseek_native_engine_prefers_resident_handle(monkeypatch):
    handle_creations = []
    handle_decode_calls = []
    op_calls = []

    class FakeTensor:

        def __init__(self, shape, *, dtype="torch.bfloat16", device="cuda:0"):
            self.shape = shape
            self.dtype = dtype
            self.device = device

    class FakeHandle:

        def __init__(self, layer_offsets, layer_kinds, layer_site_offsets,
                     layer_site_ids, layer_site_tensor_indices,
                     resident_tensors):
            handle_creations.append(
                (layer_offsets, layer_kinds, layer_site_offsets,
                 layer_site_ids, layer_site_tensor_indices,
                 resident_tensors))

        def decode(self, *args):
            handle_decode_calls.append(args)
            return args[2]

    def fake_empty(shape, *, device=None, dtype=None):
        return FakeTensor(tuple(shape), dtype=dtype, device=device)

    def fake_native_op(*args):
        op_calls.append(args)
        return args[2]

    fake_torch = SimpleNamespace(
        empty=fake_empty,
        classes=SimpleNamespace(trtllm=SimpleNamespace(
            DeepseekResidentDecodeHandle=FakeHandle)),
        ops=SimpleNamespace(trtllm=SimpleNamespace(
            deepseek_resident_decode=fake_native_op)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    contract = SimpleNamespace(
        hidden_size=7168,
        vocab_size=129280,
        layers=(SimpleNamespace(layer_idx=0, layer_kind="dense"), ),
    )
    engine = deepseek_resident_native.create_engine(
        model=_fake_native_resident_model(),
        contract=contract,
        dist=None,
    )
    invocation = SimpleNamespace(
        stable_shape_key=("shape", 16),
        real_batch_size=16,
        padded_batch_size=16,
        input_tokens=16,
        request_ids=tuple(range(16)),
        seq_lens=(1, ) * 16,
        cached_tokens=tuple(range(2048, 2064)),
    )

    result = engine.execute(
        request=None,
        contract=contract,
        invocation=invocation,
        inputs={"input_ids": FakeTensor((16, ), dtype="torch.int64")},
    )

    assert result["logits"].shape == (16, 129280)
    assert len(handle_creations) == 1
    assert handle_creations[0][0] == [3, 22]
    assert handle_creations[0][1] == [0]
    assert handle_creations[0][2] == [0, 19]
    assert handle_creations[0][3] == [
        0, 1, 2, 3, 4, 5, 6, 23, 26, 7, 19, 15, 83, 88, 93, 98, 100, 40,
        44
    ]
    assert handle_creations[0][4] == [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21
    ]
    assert len(handle_creations[0][5]) == 22
    assert len(handle_decode_calls) == 1
    assert op_calls == []


def test_deepseek_native_engine_declines_single_step_attention_dp(monkeypatch):
    handle_creations = []
    handle_decode_calls = []
    op_calls = []

    class FakeTensor:

        def __init__(self, shape, *, dtype="torch.bfloat16", device="cuda:0"):
            self.shape = shape
            self.dtype = dtype
            self.device = device

    class FakeHandle:

        def __init__(self, *args):
            handle_creations.append(args)

        def decode(self, *args):
            handle_decode_calls.append(args)
            return args[2]

    def fake_empty(shape, *, device=None, dtype=None):
        return FakeTensor(tuple(shape), dtype=dtype, device=device)

    def fake_native_op(*args):
        op_calls.append(args)
        return args[2]

    fake_torch = SimpleNamespace(
        empty=fake_empty,
        classes=SimpleNamespace(trtllm=SimpleNamespace(
            DeepseekResidentDecodeHandle=FakeHandle)),
        ops=SimpleNamespace(trtllm=SimpleNamespace(
            deepseek_resident_decode=fake_native_op)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    contract = SimpleNamespace(
        hidden_size=7168,
        vocab_size=129280,
        layers=(SimpleNamespace(layer_idx=0, layer_kind="dense"), ),
    )
    engine = deepseek_resident_native.create_engine(
        model=_fake_native_resident_model(),
        contract=contract,
        dist=SimpleNamespace(mapping=SimpleNamespace(enable_attention_dp=True)),
    )
    invocation = SimpleNamespace(
        stable_shape_key=("shape", 1),
        real_batch_size=1,
        padded_batch_size=1,
        input_tokens=1,
        request_ids=(123, ),
        seq_lens=(1, ),
        cached_tokens=(2048, ),
    )

    result = engine.execute(
        request=None,
        contract=contract,
        invocation=invocation,
        inputs={"input_ids": FakeTensor((1, ), dtype="torch.int64")},
    )

    assert result is None
    assert handle_creations == []
    assert handle_decode_calls == []
    assert op_calls == []
    assert (engine.execution_state()["reason"] ==
            "resident_native_single_step_attention_dp_disabled")


def test_deepseek_native_engine_executes_resident_window_handle(monkeypatch):
    window_calls = []
    sample_factory_calls = []

    class FakeTensor:

        def __init__(self, shape, *, dtype="torch.bfloat16", device="cuda:0"):
            self.shape = tuple(shape)
            self.dtype = dtype
            self.device = device
            self.window_written = False

    class FakeHandle:

        def __init__(self, *args):
            pass

        def run_decode_window_ready(self):
            return True

        def run_decode_window(self, initial_tokens, window_tokens, input_ids,
                              hidden_states, logits, position_ids,
                              seq_lens_cuda, kv_lens_cuda, owned_steps,
                              input_tokens, request_ids, seq_lens,
                              cached_tokens):
            window_calls.append(
                (initial_tokens, window_tokens, input_ids, hidden_states,
                 logits, position_ids, seq_lens_cuda, kv_lens_cuda,
                 owned_steps, input_tokens, request_ids, seq_lens,
                 cached_tokens))
            window_tokens.window_written = True
            return window_tokens

    def fake_empty(shape, *, device=None, dtype=None):
        return FakeTensor(tuple(shape), dtype=dtype, device=device)

    fake_torch = SimpleNamespace(
        int32="torch.int32",
        empty=fake_empty,
        classes=SimpleNamespace(trtllm=SimpleNamespace(
            DeepseekResidentDecodeHandle=FakeHandle)),
        ops=SimpleNamespace(trtllm=SimpleNamespace(
            cute_dsl_fp4_paged_mqa_logits=object())),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    contract = SimpleNamespace(
        hidden_size=7168,
        vocab_size=129280,
        layers=(SimpleNamespace(layer_idx=0, layer_kind="dense"), ),
    )
    engine = deepseek_resident_native.create_engine(
        model=_fake_native_resident_model(),
        contract=contract,
        dist=None,
    )
    invocation = SimpleNamespace(
        stable_shape_key=("shape", 16),
        real_batch_size=16,
        padded_batch_size=16,
        input_tokens=16,
        request_ids=tuple(range(16)),
        seq_lens=(1, ) * 16,
        cached_tokens=tuple(range(2048, 2064)),
    )
    initial_tokens = FakeTensor((1, 16, 1), dtype="torch.int32")
    sample_state = SimpleNamespace(
        device=SimpleNamespace(new_tokens=initial_tokens),
        host=None,
        requests=tuple(range(16)),
    )

    def make_sample_state(**kwargs):
        sample_factory_calls.append(kwargs)
        return SimpleNamespace(
            produced_tokens=kwargs["new_tokens"],
            owned_steps=kwargs["owned_steps"],
        )

    window_contract = SimpleNamespace(
        requested_window_steps=4,
        max_safe_owned_steps=3,
    )
    position_ids = FakeTensor((1, 16), dtype="torch.int64")
    seq_lens_cuda = FakeTensor((16, ), dtype="torch.int32")
    kv_lens_cuda = FakeTensor((16, ), dtype="torch.int32")
    attn_metadata = SimpleNamespace(
        seq_lens_cuda=seq_lens_cuda,
        kv_lens_cuda_runtime=kv_lens_cuda,
    )

    assert engine.window_backend_state()["ready"]
    result = engine.execute_window(
        request=SimpleNamespace(
            sample_state=sample_state,
            make_sample_state=make_sample_state,
        ),
        contract=contract,
        invocation=invocation,
        window_contract=window_contract,
        inputs={
            "input_ids": FakeTensor((16, ), dtype="torch.int64"),
            "position_ids": position_ids,
            "attn_metadata": attn_metadata,
        },
    )

    assert result is not None
    assert result.owned_steps == 3
    assert result.requested_window_steps == 4
    assert result.break_reason == "resident_window_native_executed"
    assert len(window_calls) == 1
    assert window_calls[0][0] is initial_tokens
    assert window_calls[0][1].shape == (3, 16, 1)
    assert window_calls[0][1].window_written
    assert window_calls[0][2].shape == (16, )
    assert window_calls[0][3].shape == (16, 7168)
    assert window_calls[0][4].shape == (16, 129280)
    assert window_calls[0][5] is position_ids
    assert window_calls[0][6] is seq_lens_cuda
    assert window_calls[0][7] is kv_lens_cuda
    assert window_calls[0][8:] == (
        3,
        16,
        list(range(16)),
        [1] * 16,
        list(range(2048, 2064)),
    )
    assert len(sample_factory_calls) == 1
    assert sample_factory_calls[0]["source_sample_state"] is sample_state
    assert sample_factory_calls[0]["owned_steps"] == 3
    assert result.sample_state.produced_tokens is window_calls[0][1]
    assert engine.window_backend_state()["reason"] == (
        "resident_window_native_ready")


def test_deepseek_native_engine_rejects_window_handle_until_ready(monkeypatch):
    window_calls = []

    class FakeTensor:

        def __init__(self, shape, *, dtype="torch.bfloat16", device="cuda:0"):
            self.shape = tuple(shape)
            self.dtype = dtype
            self.device = device

    class FakeHandle:

        def __init__(self, *args):
            pass

        def run_decode_window_ready(self):
            return False

        def run_decode_window(self, *args):
            window_calls.append(args)
            return args[1]

    def fake_empty(shape, *, device=None, dtype=None):
        return FakeTensor(tuple(shape), dtype=dtype, device=device)

    fake_torch = SimpleNamespace(
        int32="torch.int32",
        empty=fake_empty,
        classes=SimpleNamespace(trtllm=SimpleNamespace(
            DeepseekResidentDecodeHandle=FakeHandle)),
        ops=SimpleNamespace(trtllm=SimpleNamespace(
            cute_dsl_fp4_paged_mqa_logits=object())),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    contract = SimpleNamespace(
        hidden_size=7168,
        vocab_size=129280,
        layers=(SimpleNamespace(layer_idx=0, layer_kind="dense"), ),
    )
    engine = deepseek_resident_native.create_engine(
        model=_fake_native_resident_model(),
        contract=contract,
        dist=None,
    )
    invocation = SimpleNamespace(
        stable_shape_key=("shape", 16),
        real_batch_size=16,
        padded_batch_size=16,
        input_tokens=16,
        request_ids=tuple(range(16)),
        seq_lens=(1, ) * 16,
        cached_tokens=tuple(range(2048, 2064)),
    )
    sample_state = SimpleNamespace(
        device=SimpleNamespace(
            new_tokens=FakeTensor((1, 16, 1), dtype="torch.int32")),
        host=None,
        requests=tuple(range(16)),
    )

    assert not engine.window_backend_state()["ready"]
    result = engine.execute_window(
        request=SimpleNamespace(
            sample_state=sample_state,
            make_sample_state=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
        contract=contract,
        invocation=invocation,
        window_contract=SimpleNamespace(
            requested_window_steps=4,
            max_safe_owned_steps=3,
        ),
        inputs={
            "input_ids": FakeTensor((16, ), dtype="torch.int64"),
            "position_ids": FakeTensor((1, 16), dtype="torch.int64"),
            "attn_metadata": SimpleNamespace(
                seq_lens_cuda=FakeTensor((16, ), dtype="torch.int32"),
                kv_lens_cuda_runtime=FakeTensor((16, ), dtype="torch.int32"),
            ),
        },
    )

    assert result is None
    assert window_calls == []
    assert engine.window_backend_state()["reason"] == (
        "resident_window_native_not_ready")


def test_deepseek_native_engine_reports_window_missing_native_boundary(
        monkeypatch):
    window_calls = []

    class FakeTensor:

        def __init__(self, shape, *, dtype="torch.bfloat16", device="cuda:0"):
            self.shape = tuple(shape)
            self.dtype = dtype
            self.device = device

    class FakeHandle:

        def __init__(self, *args):
            pass

        def run_decode_window_ready(self):
            return False

        def run_decode_window_not_ready_reason(self):
            return "resident_window_native_missing_dsa_attention_dispatch"

        def run_decode_window_attention_metadata_device_refresh(self, *args):
            return args[4]

        def run_decode_window(self, *args):
            window_calls.append(args)
            return args[1]

    def fake_empty(shape, *, device=None, dtype=None):
        return FakeTensor(tuple(shape), dtype=dtype, device=device)

    fake_torch = SimpleNamespace(
        int32="torch.int32",
        empty=fake_empty,
        classes=SimpleNamespace(trtllm=SimpleNamespace(
            DeepseekResidentDecodeHandle=FakeHandle)),
        ops=SimpleNamespace(trtllm=SimpleNamespace()),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    contract = SimpleNamespace(
        hidden_size=7168,
        vocab_size=129280,
        layers=(SimpleNamespace(layer_idx=0, layer_kind="dense"), ),
    )
    engine = deepseek_resident_native.create_engine(
        model=_fake_native_resident_model(),
        contract=contract,
        dist=None,
    )
    invocation = SimpleNamespace(
        stable_shape_key=("shape", 16),
        real_batch_size=16,
        padded_batch_size=16,
        input_tokens=16,
        request_ids=tuple(range(16)),
        seq_lens=(1, ) * 16,
        cached_tokens=tuple(range(2048, 2064)),
    )
    sample_state = SimpleNamespace(
        device=SimpleNamespace(
            new_tokens=FakeTensor((1, 16, 1), dtype="torch.int32")),
        host=None,
        requests=tuple(range(16)),
    )

    assert engine.window_backend_state()["reason"] == (
        "resident_window_native_missing_dsa_attention_dispatch")
    native_contract = engine.window_backend_state()["metadata"][
        "native_window_contract"]
    assert native_contract["first_missing_component"] == "native_window_body"
    assert "native_dsa_attention_dispatch" in native_contract[
        "missing_components"]
    assert native_contract["component_ready"]["native_dsa_indexer_assets"]
    assert "native_dsa_indexer_assets" not in native_contract[
        "missing_components"]
    assert native_contract["component_reasons"][
        "native_window_body"].startswith(
            "resident_window_native_missing_dsa_attention_dispatch")
    assert native_contract["component_ready"][
        "native_attention_metadata_refresh"]
    assert "native_attention_metadata_refresh" not in native_contract[
        "missing_components"]
    result = engine.execute_window(
        request=SimpleNamespace(
            sample_state=sample_state,
            make_sample_state=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
        contract=contract,
        invocation=invocation,
        window_contract=SimpleNamespace(
            requested_window_steps=4,
            max_safe_owned_steps=3,
        ),
        inputs={
            "input_ids": FakeTensor((16, ), dtype="torch.int64"),
            "position_ids": FakeTensor((1, 16), dtype="torch.int64"),
            "attn_metadata": SimpleNamespace(
                seq_lens_cuda=FakeTensor((16, ), dtype="torch.int32"),
                kv_lens_cuda_runtime=FakeTensor((16, ), dtype="torch.int32"),
            ),
        },
    )

    assert result is None
    assert window_calls == []
    assert engine.window_backend_state()["reason"] == (
        "resident_window_native_missing_dsa_attention_dispatch")


def test_deepseek_native_engine_window_stage_body_owns_steps(monkeypatch):
    calls = []
    sample_factory_calls = []

    class FakeTensor:

        def __init__(self, shape, *, dtype="torch.bfloat16", device="cuda:0"):
            self.shape = tuple(shape)
            self.dtype = dtype
            self.device = device
            self.sampled_steps = []

    class FakeHandle:

        def __init__(self, *args):
            pass

        def run_decode_window_ready(self):
            return False

        def run_decode_window_not_ready_reason(self):
            return "resident_window_native_missing_window_execution_plan"

        def run_layer_dsa_attention_dispatch_ready(self):
            return True

        def run_layer_dsa_attention_dispatch_not_ready_reason(self):
            return "resident_attention_dsa_dispatch_native_ready"

        def run_decode_window(self, *args):
            calls.append(("native_window", args))
            return args[1]

        def run_decode_window_attention_metadata_device_refresh(self, *args):
            calls.append(("metadata_native_refresh", args[-8:]))
            return args[4]

        def run_decode_window_advance_state(self, *args):
            calls.append(("window_advance", args))
            return args[2]

        def run_decode_window_prepare_step(
                self, initial_tokens, window_tokens_scratch, input_ids,
                hidden_states_scratch, position_ids, kv_lens_cuda,
                output_step_idx, input_tokens):
            calls.append(("window_prepare", output_step_idx, initial_tokens,
                          window_tokens_scratch, input_ids, position_ids,
                          kv_lens_cuda, input_tokens))
            hidden_states_scratch.prepared_step = output_step_idx
            return hidden_states_scratch

        def run_input_embedding(self, input_ids, hidden_states_scratch,
                                input_tokens):
            calls.append(("embedding", input_tokens))
            return hidden_states_scratch

        def run_layer_input_rmsnorm(self, layer_idx, hidden_states_scratch,
                                    norm_hidden_states_scratch, input_tokens,
                                    eps, use_gemma):
            calls.append(("input_rmsnorm", layer_idx,
                          hidden_states_scratch.prepared_step))
            norm_hidden_states_scratch.prepared_step = (
                hidden_states_scratch.prepared_step)
            return norm_hidden_states_scratch

        def run_layer_input_gated_norm(self, layer_idx,
                                       norm_hidden_states_scratch,
                                       gated_hidden_states_scratch,
                                       input_tokens):
            calls.append(("input_gated_norm", layer_idx, input_tokens))
            gated_hidden_states_scratch.prepared_step = (
                norm_hidden_states_scratch.prepared_step)
            return gated_hidden_states_scratch

        def run_layer_attention_output_tail(
                self, layer_idx, attention_core_output_scratch,
                attention_input_scratch, attention_gate_scratch,
                attention_hidden_states_scratch, input_tokens):
            calls.append(("attention_tail", layer_idx,
                          attention_core_output_scratch.attended_step))
            return attention_hidden_states_scratch

        def run_layer_post_attention_rmsnorm(
                self, layer_idx, attention_hidden_states_scratch,
                residual_input_scratch, post_attention_norm_scratch,
                post_attention_residual_scratch, input_tokens, eps,
                use_gemma):
            calls.append(("post_attention_rmsnorm", layer_idx))
            return post_attention_norm_scratch

        def run_layer_post_attention_gated_norm(
                self, layer_idx, post_attention_norm_scratch,
                post_attention_gated_scratch, input_tokens):
            calls.append(("post_attention_gated_norm", layer_idx))
            return post_attention_gated_scratch

        def run_layer_dense_mlp(self, layer_idx,
                                post_attention_gated_scratch,
                                dense_mlp_intermediate_scratch,
                                dense_mlp_output_scratch, input_tokens):
            calls.append(("dense_mlp", layer_idx))
            return dense_mlp_output_scratch

        def run_layer_post_ffn_rmsnorm(
                self, layer_idx, dense_mlp_output_scratch,
                post_attention_residual_scratch, next_layer_hidden_scratch,
                next_layer_residual_scratch, input_tokens, eps, use_gemma):
            calls.append(("post_ffn_rmsnorm", layer_idx))
            return next_layer_hidden_scratch

        def run_lm_head_logits(self, next_layer_hidden_scratch,
                               logits_scratch, input_tokens):
            calls.append(("lm_head", input_tokens))
            return logits_scratch

        def run_decode_window_sample_step(
                self, logits_scratch, window_tokens_scratch, output_step_idx,
                input_tokens):
            calls.append(("window_sample", output_step_idx, input_tokens))
            window_tokens_scratch.sampled_steps.append(output_step_idx)
            return window_tokens_scratch

    def fake_empty(shape, *, device=None, dtype=None):
        return FakeTensor(tuple(shape), dtype=dtype, device=device)

    fake_torch = SimpleNamespace(
        float32="torch.float32",
        int32="torch.int32",
        empty=fake_empty,
        classes=SimpleNamespace(trtllm=SimpleNamespace(
            DeepseekResidentDecodeHandle=FakeHandle)),
        ops=SimpleNamespace(trtllm=SimpleNamespace()),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    model = _fake_native_resident_model()

    def forward_impl_with_dsa(position_ids,
                              hidden_states,
                              attn_metadata,
                              output,
                              kv_proj_input=None):
        calls.append(("attention_core", position_ids, hidden_states.shape,
                      attn_metadata.refresh_calls, kv_proj_input))
        output.attended_step = hidden_states.prepared_step

    model.model.layers[0].self_attn.forward_impl_with_dsa = forward_impl_with_dsa
    contract = SimpleNamespace(
        hidden_size=7168,
        vocab_size=129280,
        rms_norm_eps=1e-6,
        layers=(SimpleNamespace(layer_idx=0, layer_kind="dense"), ),
    )
    engine = deepseek_resident_native.create_engine(
        model=model,
        contract=contract,
        dist=None,
    )
    invocation = SimpleNamespace(
        stable_shape_key=("shape", 16),
        real_batch_size=16,
        padded_batch_size=16,
        input_tokens=16,
        request_ids=tuple(range(16)),
        seq_lens=(1, ) * 16,
        cached_tokens=tuple(range(2048, 2064)),
    )
    initial_tokens = FakeTensor((1, 16, 1), dtype="torch.int32")
    sample_state = SimpleNamespace(
        device=SimpleNamespace(new_tokens=initial_tokens),
        host=None,
        requests=tuple(range(16)),
    )

    def make_sample_state(**kwargs):
        sample_factory_calls.append(kwargs)
        return SimpleNamespace(
            produced_tokens=kwargs["new_tokens"],
            owned_steps=kwargs["owned_steps"],
        )

    input_ids = FakeTensor((16, ), dtype="torch.int64")
    position_ids = FakeTensor((1, 16), dtype="torch.int64")
    kv_lens_cuda = FakeTensor((16, ), dtype="torch.int32")
    attn_metadata = SimpleNamespace(
        seq_lens_cuda=FakeTensor((16, ), dtype="torch.int32"),
        kv_lens_cuda_runtime=kv_lens_cuda,
        req_idx_per_token=FakeTensor((16, ), dtype="torch.int32"),
        indexer_k_cache_block_offsets=FakeTensor((16, 4),
                                                 dtype="torch.int32"),
        slot_mapping_fp8=FakeTensor((16, ), dtype="torch.int64"),
        slot_mapping_scale=FakeTensor((16, ), dtype="torch.int64"),
        gen_kv_indptr=FakeTensor((17, ), dtype="torch.int64"),
        gen_cached_token_indptr=FakeTensor((17, ), dtype="torch.int64"),
        kv_lens_cuda_2d=FakeTensor((16, 2), dtype="torch.int32"),
        kv_cache_manager=SimpleNamespace(
            index_head_dim=16,
            tokens_per_block=4,
            quant_block_size=4,
            use_fp4=True,
        ),
        num_tokens=16,
        num_seqs=16,
        num_contexts=0,
        num_generations=16,
        refresh_calls=0,
    )

    def on_update_kv_lens():
        attn_metadata.refresh_calls += 1

    attn_metadata.on_update_kv_lens = on_update_kv_lens
    result = engine.execute_window(
        request=SimpleNamespace(
            sample_state=sample_state,
            make_sample_state=make_sample_state,
        ),
        contract=contract,
        invocation=invocation,
        window_contract=SimpleNamespace(
            requested_window_steps=4,
            max_safe_owned_steps=3,
        ),
        inputs={
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attn_metadata": attn_metadata,
        },
    )

    assert result is not None
    assert result.owned_steps == 3
    assert result.requested_window_steps == 4
    assert result.break_reason == "resident_window_stage_scheduler_executed"
    assert ("native_window", ) not in [call[:1] for call in calls]
    assert ("embedding", 16) not in calls
    assert [call[1] for call in calls if call[0] == "window_prepare"] == [1, 2]
    assert [call[1] for call in calls if call[0] == "window_sample"] == [1, 2]
    assert [call[2] for call in calls if call[0] == "input_rmsnorm"] == [1, 2]
    assert [call[3] for call in calls if call[0] == "attention_core"] == [1, 2]
    assert [call[1] for call in calls
            if call[0] == "metadata_native_refresh"] == [
                (16, 16, 0, 16, 16, 4, 4, 8),
                (16, 16, 0, 16, 16, 4, 4, 8),
            ]
    assert len(sample_factory_calls) == 1
    assert sample_factory_calls[0]["source_sample_state"] is sample_state
    assert sample_factory_calls[0]["owned_steps"] == 3
    assert result.sample_state.produced_tokens.sampled_steps == [1, 2]
    assert engine._scheduler_state.window_scheduler_calls == 1
    assert engine._scheduler_state.window_scheduler_completions == 1
    assert engine._scheduler_state.stage_scheduler_completions == 2
    assert engine._scheduler_state.window_prepare_stage_calls == 2
    assert engine._scheduler_state.window_sample_stage_calls == 2
    assert engine._scheduler_state.window_metadata_refresh_calls == 2
    assert (
        engine._scheduler_state.window_metadata_native_device_refresh_calls
        == 2)
    state = engine.window_backend_state()
    assert state["backend"] == "deepseek_resident_window_native_v1"
    assert state["ready"] is True
    assert state["reason"] == "resident_window_stage_scheduler_ready"
    assert state["metadata"]["native"] is True
    assert state["metadata"]["window_stage_body_ready"] is True
    assert state["metadata"]["window_stage_scheduler_ready"] is True
    assert state["metadata"]["window_stage_scheduler_debug_only"] is False
    native_contract = state["metadata"]["native_window_contract"]
    assert native_contract["ready"] is False
    assert native_contract["first_missing_component"] == "native_window_body"
    assert native_contract["component_ready"]["native_window_body"] is False
    assert native_contract["component_ready"]["native_dsa_indexer_assets"]
    assert state["metadata"]["last_window_reason"] == (
        "resident_window_stage_scheduler_executed")


def test_deepseek_native_engine_plan_window_flattens_dsa_payload(monkeypatch):
    calls = []

    class FakeTensor:

        def __init__(self, shape, *, dtype="torch.bfloat16", device="cuda:0"):
            self.shape = tuple(shape)
            self.dtype = dtype
            self.device = device

    class FakeHandle:

        def __init__(self, *args):
            pass

        def run_decode_window_with_dsa_plan_ready(self):
            return True

        def run_decode_window_with_dsa_plan(self, *args):
            calls.append(args)
            return args[1]

    def fake_empty(shape, *, device=None, dtype=None):
        return FakeTensor(tuple(shape), dtype=dtype, device=device)

    fake_torch = SimpleNamespace(
        float32="torch.float32",
        int32="torch.int32",
        uint32="torch.uint32",
        uint8="torch.uint8",
        empty=fake_empty,
        classes=SimpleNamespace(trtllm=SimpleNamespace(
            DeepseekResidentDecodeHandle=FakeHandle)),
        ops=SimpleNamespace(trtllm=SimpleNamespace()),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    contract = SimpleNamespace(
        hidden_size=7168,
        vocab_size=129280,
        layers=(SimpleNamespace(layer_idx=0, layer_kind="dense"), ),
    )
    engine = deepseek_resident_native.create_engine(
        model=_fake_native_resident_model(),
        contract=contract,
        dist=None,
    )
    invocation = SimpleNamespace(
        stable_shape_key=("shape", 16),
        real_batch_size=16,
        padded_batch_size=16,
        input_tokens=16,
        request_ids=tuple(range(16)),
        seq_lens=(1, ) * 16,
        cached_tokens=tuple(range(2048, 2064)),
    )
    input_ids = FakeTensor((16, ), dtype="torch.int64")
    plan = (deepseek_resident_native.DeepSeekResidentDsaWindowLayerPlan(
        layer_idx=0,
        static_metadata_tensors=(
            FakeTensor((16, 2048), dtype="torch.int32"),
            FakeTensor((4096, 128, 1, 64), dtype="torch.int8"),
        ),
        runtime_tensors={
            "rotary_cos_sin": FakeTensor((132096, 64),
                                         dtype="torch.float32"),
            "dense_kv_pool": FakeTensor((4096, 1, 1, 1, 64, 288),
                                        dtype="torch.int8"),
        },
        runtime_config={
            "num_seqs": 16,
            "num_generations": 16,
        },
        runtime_scalars={
            "softmax_scale": 1.0,
        },
        scratch_shapes=((16, 128, 576), (16, 128, 512), (17, ), (17, ),
                        (1, ), (16, )),
    ), )
    monkeypatch.setattr(engine, "_build_dsa_window_plan",
                        lambda **kwargs: plan)
    initial_tokens = FakeTensor((1, 16, 1), dtype="torch.int32")
    sample_state = SimpleNamespace(
        device=SimpleNamespace(new_tokens=initial_tokens),
        host=None,
        requests=tuple(range(16)),
    )

    result = engine.execute_window(
        request=SimpleNamespace(
            sample_state=sample_state,
            make_sample_state=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
        contract=contract,
        invocation=invocation,
        window_contract=SimpleNamespace(
            requested_window_steps=4,
            max_safe_owned_steps=3,
        ),
        inputs={
            "input_ids": input_ids,
            "position_ids": FakeTensor((1, 16), dtype="torch.int64"),
            "attn_metadata": SimpleNamespace(
                seq_lens_cuda=FakeTensor((16, ), dtype="torch.int32"),
                kv_lens_cuda_runtime=FakeTensor((16, ), dtype="torch.int32"),
            ),
        },
    )

    assert result is not None
    assert result.break_reason == "resident_window_native_plan_executed"
    assert len(calls) == 1
    call = calls[0]
    assert call[8] == [0]
    assert call[9] == [0, 2]
    assert [tensor.shape for tensor in call[10]] == [(16, 2048),
                                                     (4096, 128, 1, 64)]
    assert call[11] == [{
        "rotary_cos_sin": plan[0].runtime_tensors["rotary_cos_sin"],
        "dense_kv_pool": plan[0].runtime_tensors["dense_kv_pool"],
    }]
    assert call[12] == [{"num_seqs": 16, "num_generations": 16}]
    assert call[13] == [{"softmax_scale": 1.0}]
    assert call[14] == [0, 6]
    assert [tensor.shape for tensor in call[15]] == [
        (16, 128, 576), (16, 128, 512), (17, ), (17, ), (1, ), (16, )
    ]
    assert call[16:21] == (3, 16, list(range(16)), [1] * 16,
                           list(range(2048, 2064)))


def test_deepseek_native_engine_plan_window_not_ready_reason_wins(
        monkeypatch):
    calls = []

    class FakeTensor:

        def __init__(self, shape, *, dtype="torch.bfloat16", device="cuda:0"):
            self.shape = tuple(shape)
            self.dtype = dtype
            self.device = device

    class FakeHandle:

        def __init__(self, *args):
            pass

        def run_decode_window_ready(self):
            return False

        def run_decode_window_not_ready_reason(self):
            return "resident_window_native_missing_window_execution_plan"

        def run_decode_window_with_dsa_plan_ready(self):
            return False

        def run_decode_window_with_dsa_plan_not_ready_reason(self):
            return "resident_window_native_body_disabled"

        def run_decode_window_with_dsa_plan(self, *args):
            calls.append(("plan_window", args))
            return args[1]

        def run_decode_window(self, *args):
            calls.append(("legacy_window", args))
            return args[1]

    def fake_empty(shape, *, device=None, dtype=None):
        return FakeTensor(tuple(shape), dtype=dtype, device=device)

    fake_torch = SimpleNamespace(
        float32="torch.float32",
        int32="torch.int32",
        uint32="torch.uint32",
        uint8="torch.uint8",
        empty=fake_empty,
        classes=SimpleNamespace(trtllm=SimpleNamespace(
            DeepseekResidentDecodeHandle=FakeHandle)),
        ops=SimpleNamespace(trtllm=SimpleNamespace()),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    contract = SimpleNamespace(
        hidden_size=7168,
        vocab_size=129280,
        layers=(SimpleNamespace(layer_idx=0, layer_kind="dense"), ),
    )
    engine = deepseek_resident_native.create_engine(
        model=_fake_native_resident_model(),
        contract=contract,
        dist=None,
    )
    monkeypatch.setattr(engine, "_build_dsa_window_plan", lambda **kwargs: ())
    invocation = SimpleNamespace(
        stable_shape_key=("shape", 16),
        real_batch_size=16,
        padded_batch_size=16,
        input_tokens=16,
        request_ids=tuple(range(16)),
        seq_lens=(1, ) * 16,
        cached_tokens=tuple(range(2048, 2064)),
    )
    initial_tokens = FakeTensor((1, 16, 1), dtype="torch.int32")
    sample_state = SimpleNamespace(
        device=SimpleNamespace(new_tokens=initial_tokens),
        host=None,
        requests=tuple(range(16)),
    )

    state = engine.window_backend_state()
    assert state["reason"] == "resident_window_native_body_disabled"

    result = engine.execute_window(
        request=SimpleNamespace(
            sample_state=sample_state,
            make_sample_state=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
        contract=contract,
        invocation=invocation,
        window_contract=SimpleNamespace(
            requested_window_steps=4,
            max_safe_owned_steps=3,
        ),
        inputs={
            "input_ids": FakeTensor((16, ), dtype="torch.int64"),
            "position_ids": FakeTensor((1, 16), dtype="torch.int64"),
            "attn_metadata": SimpleNamespace(
                seq_lens_cuda=FakeTensor((16, ), dtype="torch.int32"),
                kv_lens_cuda_runtime=FakeTensor((16, ), dtype="torch.int32"),
            ),
        },
    )

    assert result is None
    assert calls == []
    assert engine.window_backend_state()["reason"] == (
        "resident_window_native_body_disabled")


def test_deepseek_native_engine_debug_window_stage_scheduler_bridges_moe(
        monkeypatch):
    monkeypatch.setenv(
        "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_STAGE_SCHEDULER", "1")
    calls = []
    sample_factory_calls = []

    class FakeTensor:

        def __init__(self, shape, *, dtype="torch.bfloat16", device="cuda:0"):
            self.shape = tuple(shape)
            self.dtype = dtype
            self.device = device
            self.copied_from = None
            self.sampled_steps = []

        def add_(self, other):
            calls.append(("add", self.shape, other.shape))
            return FakeTensor(self.shape, dtype=self.dtype, device=self.device)

        def copy_(self, other):
            self.copied_from = other
            calls.append(("copy", self.shape, other.shape))
            return self

    class FakeExperts:
        has_nvfp4 = False

        def __call__(self, hidden_states, router_logits, **kwargs):
            calls.append(("experts", hidden_states.shape, router_logits.shape,
                          kwargs["do_finalize"],
                          kwargs["all_rank_num_tokens"]))
            return FakeTensor(hidden_states.shape,
                              dtype=hidden_states.dtype,
                              device=hidden_states.device)

    class FakeSharedExperts:

        def __call__(self, hidden_states):
            calls.append(("shared_experts", hidden_states.shape))
            return FakeTensor(hidden_states.shape,
                              dtype=hidden_states.dtype,
                              device=hidden_states.device)

    class FakeHandle:

        def __init__(self, *args):
            pass

        def run_decode_window_ready(self):
            return False

        def run_decode_window(self, *args):
            calls.append(("native_window", args))
            return args[1]

        def run_decode_window_prepare_step(
                self, initial_tokens, window_tokens_scratch, input_ids,
                hidden_states_scratch, position_ids, kv_lens_cuda,
                output_step_idx, input_tokens):
            calls.append(("window_prepare", output_step_idx))
            hidden_states_scratch.prepared_step = output_step_idx
            return hidden_states_scratch

        def run_layer_input_rmsnorm(self, layer_idx, hidden_states_scratch,
                                    norm_hidden_states_scratch, input_tokens,
                                    eps, use_gemma):
            calls.append(("input_rmsnorm", layer_idx,
                          hidden_states_scratch.prepared_step))
            norm_hidden_states_scratch.prepared_step = (
                hidden_states_scratch.prepared_step)
            return norm_hidden_states_scratch

        def run_layer_input_gated_norm(self, layer_idx,
                                       norm_hidden_states_scratch,
                                       gated_hidden_states_scratch,
                                       input_tokens):
            calls.append(("input_gated_norm", layer_idx, input_tokens))
            gated_hidden_states_scratch.prepared_step = (
                norm_hidden_states_scratch.prepared_step)
            return gated_hidden_states_scratch

        def run_layer_attention_output_tail(
                self, layer_idx, attention_core_output_scratch,
                attention_input_scratch, attention_gate_scratch,
                attention_hidden_states_scratch, input_tokens):
            calls.append(("attention_tail", layer_idx,
                          attention_core_output_scratch.attended_step))
            return attention_hidden_states_scratch

        def run_layer_post_attention_rmsnorm(
                self, layer_idx, attention_hidden_states_scratch,
                residual_input_scratch, post_attention_norm_scratch,
                post_attention_residual_scratch, input_tokens, eps,
                use_gemma):
            calls.append(("post_attention_rmsnorm", layer_idx))
            return post_attention_norm_scratch

        def run_layer_post_attention_gated_norm(
                self, layer_idx, post_attention_norm_scratch,
                post_attention_gated_scratch, input_tokens):
            calls.append(("post_attention_gated_norm", layer_idx))
            return post_attention_gated_scratch

        def run_layer_moe_router(
                self, layer_idx, post_attention_gated_scratch,
                router_logits_scratch, router_scores_scratch,
                router_topk_indices_scratch, router_topk_weights_scratch,
                input_tokens, top_k, n_group, topk_group,
                routed_scaling_factor):
            calls.append(("moe_router", layer_idx, router_logits_scratch.shape,
                          top_k, n_group, topk_group, routed_scaling_factor))
            return router_topk_weights_scratch

        def run_layer_post_ffn_rmsnorm(
                self, layer_idx, dense_mlp_output_scratch,
                post_attention_residual_scratch, next_layer_hidden_scratch,
                next_layer_residual_scratch, input_tokens, eps, use_gemma):
            calls.append(("post_ffn_rmsnorm", layer_idx,
                          dense_mlp_output_scratch.copied_from.shape))
            return next_layer_hidden_scratch

        def run_lm_head_logits(self, next_layer_hidden_scratch,
                               logits_scratch, input_tokens):
            calls.append(("lm_head", input_tokens))
            return logits_scratch

        def run_decode_window_sample_step(
                self, logits_scratch, window_tokens_scratch, output_step_idx,
                input_tokens):
            calls.append(("window_sample", output_step_idx, input_tokens))
            window_tokens_scratch.sampled_steps.append(output_step_idx)
            return window_tokens_scratch

    def fake_empty(shape, *, device=None, dtype=None):
        return FakeTensor(tuple(shape), dtype=dtype, device=device)

    fake_torch = SimpleNamespace(
        float32="torch.float32",
        int32="torch.int32",
        empty=fake_empty,
        classes=SimpleNamespace(trtllm=SimpleNamespace(
            DeepseekResidentDecodeHandle=FakeHandle)),
        ops=SimpleNamespace(trtllm=SimpleNamespace()),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    model = _fake_native_resident_model_with_moe()
    moe = model.model.layers[1].mlp
    moe.experts = FakeExperts()
    moe.shared_experts = FakeSharedExperts()
    moe.use_dp = False
    moe.mapping = SimpleNamespace(tp_size=1, is_multi_node=lambda: False)
    moe.shared_output_scale = None
    model.model.layers[1].fusion_config = SimpleNamespace(
        POST_MOE_FUSION=False)

    def forward_impl_with_dsa(position_ids,
                              hidden_states,
                              attn_metadata,
                              output,
                              kv_proj_input=None):
        calls.append(("attention_core", hidden_states.prepared_step,
                      attn_metadata.refresh_calls, kv_proj_input))
        output.attended_step = hidden_states.prepared_step

    model.model.layers[1].self_attn.forward_impl_with_dsa = forward_impl_with_dsa
    contract = SimpleNamespace(
        hidden_size=7168,
        vocab_size=129280,
        rms_norm_eps=1e-6,
        num_experts_per_tok=8,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        layers=(SimpleNamespace(layer_idx=1, layer_kind="moe", top_k=8), ),
    )
    engine = deepseek_resident_native.create_engine(
        model=model,
        contract=contract,
        dist=None,
    )
    invocation = SimpleNamespace(
        stable_shape_key=("shape", 16),
        real_batch_size=16,
        padded_batch_size=16,
        input_tokens=16,
        request_ids=tuple(range(16)),
        seq_lens=(1, ) * 16,
        cached_tokens=tuple(range(2048, 2064)),
    )
    initial_tokens = FakeTensor((1, 16, 1), dtype="torch.int32")
    sample_state = SimpleNamespace(
        device=SimpleNamespace(new_tokens=initial_tokens),
        host=None,
        requests=tuple(range(16)),
    )

    def make_sample_state(**kwargs):
        sample_factory_calls.append(kwargs)
        return SimpleNamespace(
            produced_tokens=kwargs["new_tokens"],
            owned_steps=kwargs["owned_steps"],
        )

    attn_metadata = SimpleNamespace(
        seq_lens_cuda=FakeTensor((16, ), dtype="torch.int32"),
        kv_lens_cuda_runtime=FakeTensor((16, ), dtype="torch.int32"),
        all_rank_num_tokens=[16],
        refresh_calls=0,
    )

    def on_update_kv_lens():
        attn_metadata.refresh_calls += 1

    attn_metadata.on_update_kv_lens = on_update_kv_lens
    result = engine.execute_window(
        request=SimpleNamespace(
            sample_state=sample_state,
            make_sample_state=make_sample_state,
        ),
        contract=contract,
        invocation=invocation,
        window_contract=SimpleNamespace(
            requested_window_steps=4,
            max_safe_owned_steps=3,
        ),
        inputs={
            "input_ids": FakeTensor((16, ), dtype="torch.int64"),
            "position_ids": FakeTensor((1, 16), dtype="torch.int64"),
            "attn_metadata": attn_metadata,
        },
    )

    assert result is not None
    assert result.owned_steps == 3
    assert result.break_reason == "resident_window_stage_scheduler_executed"
    assert ("native_window", ) not in [call[:1] for call in calls]
    assert [call[1] for call in calls if call[0] == "window_prepare"] == [1, 2]
    assert [call[2] for call in calls if call[0] == "input_rmsnorm"] == [1, 2]
    assert [call[2] for call in calls if call[0] == "attention_core"] == [1, 2]
    assert calls.count(("moe_router", 1, (16, 128), 8, 8, 4, 2.5)) == 2
    assert calls.count(("experts", (16, 7168), (16, 128), True, [16])) == 2
    assert calls.count(("shared_experts", (16, 7168))) == 2
    assert calls.count(("add", (16, 7168), (16, 7168))) == 2
    assert [call[1] for call in calls if call[0] == "window_sample"] == [1, 2]
    assert result.sample_state.produced_tokens.sampled_steps == [1, 2]
    assert len(sample_factory_calls) == 1
    assert engine._scheduler_state.stage_scheduler_completions == 2
    assert engine._scheduler_state.window_metadata_refresh_calls == 2
    shape_state = engine._shape_states[invocation.stable_shape_key]
    assert shape_state.scratch["last_moe_experts_reason"] == (
        "moe_bridge_executed")


def test_deepseek_native_engine_runs_input_embedding_stage(monkeypatch):
    embedding_calls = []
    rmsnorm_calls = []
    gated_norm_calls = []
    attention_output_tail_calls = []
    post_attention_rmsnorm_calls = []
    post_attention_gated_norm_calls = []
    moe_router_calls = []
    dense_mlp_calls = []
    post_ffn_rmsnorm_calls = []
    lm_head_logits_calls = []
    sampling_calls = []
    window_sample_calls = []
    window_prepare_calls = []

    class FakeTensor:

        def __init__(self, shape, *, dtype="torch.bfloat16", device="cuda:0"):
            self.shape = shape
            self.dtype = dtype
            self.device = device
            self.embedding_written = False

    class FakeHandle:

        def __init__(self, layer_offsets, layer_kinds, layer_site_offsets,
                     layer_site_ids, layer_site_tensor_indices,
                     resident_tensors):
            self.layer_offsets = layer_offsets
            self.layer_kinds = layer_kinds
            self.layer_site_offsets = layer_site_offsets
            self.layer_site_ids = layer_site_ids
            self.layer_site_tensor_indices = layer_site_tensor_indices
            self.resident_tensors = resident_tensors

        def run_input_embedding(self, input_ids, hidden_states_scratch,
                                input_tokens):
            embedding_calls.append(
                (input_ids, hidden_states_scratch, input_tokens,
                 self.layer_offsets, self.layer_kinds,
                 self.resident_tensors))
            hidden_states_scratch.embedding_written = True
            return hidden_states_scratch

        def run_layer_input_rmsnorm(self, layer_idx, hidden_states_scratch,
                                    norm_hidden_states_scratch, input_tokens,
                                    eps, use_gemma):
            rmsnorm_calls.append(
                (layer_idx, hidden_states_scratch,
                 norm_hidden_states_scratch, input_tokens, eps, use_gemma))
            norm_hidden_states_scratch.rmsnorm_written = True
            return norm_hidden_states_scratch

        def run_layer_input_gated_norm(self, layer_idx,
                                       norm_hidden_states_scratch,
                                       gated_hidden_states_scratch,
                                       input_tokens):
            gated_norm_calls.append(
                (layer_idx, norm_hidden_states_scratch,
                 gated_hidden_states_scratch, input_tokens))
            gated_hidden_states_scratch.gated_norm_written = True
            return gated_hidden_states_scratch

        def run_layer_attention_output_tail(
                self, layer_idx, attention_core_output_scratch,
                attention_input_scratch, attention_gate_scratch,
                attention_hidden_states_scratch, input_tokens):
            attention_output_tail_calls.append(
                (layer_idx, attention_core_output_scratch,
                 attention_input_scratch, attention_gate_scratch,
                 attention_hidden_states_scratch, input_tokens))
            attention_core_output_scratch.attention_core_output_written = True
            attention_gate_scratch.attention_gate_written = True
            attention_hidden_states_scratch.attention_tail_written = True
            return attention_hidden_states_scratch

        def run_layer_post_attention_rmsnorm(
                self, layer_idx, attention_hidden_states_scratch,
                residual_input_scratch, post_attention_norm_scratch,
                post_attention_residual_scratch, input_tokens, eps,
                use_gemma):
            post_attention_rmsnorm_calls.append(
                (layer_idx, attention_hidden_states_scratch,
                 residual_input_scratch, post_attention_norm_scratch,
                 post_attention_residual_scratch, input_tokens, eps,
                 use_gemma))
            post_attention_norm_scratch.post_attention_rmsnorm_written = True
            post_attention_residual_scratch.post_attention_residual_written = True
            return post_attention_norm_scratch

        def run_layer_post_attention_gated_norm(
                self, layer_idx, post_attention_norm_scratch,
                post_attention_gated_scratch, input_tokens):
            post_attention_gated_norm_calls.append(
                (layer_idx, post_attention_norm_scratch,
                 post_attention_gated_scratch, input_tokens))
            post_attention_gated_scratch.post_attention_gated_norm_written = True
            return post_attention_gated_scratch

        def run_layer_moe_router(
                self, layer_idx, post_attention_gated_scratch,
                router_logits_scratch, router_scores_scratch,
                router_topk_indices_scratch, router_topk_weights_scratch,
                input_tokens, top_k, n_group, topk_group,
                routed_scaling_factor):
            moe_router_calls.append(
                (layer_idx, post_attention_gated_scratch,
                 router_logits_scratch, router_scores_scratch,
                 router_topk_indices_scratch, router_topk_weights_scratch,
                 input_tokens, top_k, n_group, topk_group,
                 routed_scaling_factor))
            router_logits_scratch.router_logits_written = True
            router_scores_scratch.router_scores_written = True
            router_topk_indices_scratch.router_topk_indices_written = True
            router_topk_weights_scratch.router_topk_weights_written = True
            return router_topk_weights_scratch

        def run_layer_dense_mlp(self, layer_idx,
                                post_attention_gated_scratch,
                                dense_mlp_intermediate_scratch,
                                dense_mlp_output_scratch, input_tokens):
            dense_mlp_calls.append(
                (layer_idx, post_attention_gated_scratch,
                 dense_mlp_intermediate_scratch, dense_mlp_output_scratch,
                 input_tokens))
            dense_mlp_intermediate_scratch.dense_mlp_intermediate_written = True
            dense_mlp_output_scratch.dense_mlp_output_written = True
            return dense_mlp_output_scratch

        def run_layer_post_ffn_rmsnorm(
                self, layer_idx, dense_mlp_output_scratch,
                post_attention_residual_scratch, next_layer_hidden_scratch,
                next_layer_residual_scratch, input_tokens, eps, use_gemma):
            post_ffn_rmsnorm_calls.append(
                (layer_idx, dense_mlp_output_scratch,
                 post_attention_residual_scratch, next_layer_hidden_scratch,
                 next_layer_residual_scratch, input_tokens, eps, use_gemma))
            next_layer_hidden_scratch.next_layer_hidden_written = True
            next_layer_residual_scratch.next_layer_residual_written = True
            return next_layer_hidden_scratch

        def run_lm_head_logits(self, next_layer_hidden_scratch,
                               logits_scratch, input_tokens):
            lm_head_logits_calls.append(
                (next_layer_hidden_scratch, logits_scratch, input_tokens))
            logits_scratch.logits_written = True
            return logits_scratch

        def run_greedy_sample(self, logits_scratch, new_tokens_scratch,
                              input_tokens):
            sampling_calls.append(
                (logits_scratch, new_tokens_scratch, input_tokens))
            new_tokens_scratch.greedy_sample_written = True
            return new_tokens_scratch

        def run_decode_window_sample_step(
                self, logits_scratch, window_tokens_scratch, output_step_idx,
                input_tokens):
            window_sample_calls.append(
                (logits_scratch, window_tokens_scratch, output_step_idx,
                 input_tokens))
            window_tokens_scratch.window_sample_written = True
            return window_tokens_scratch

        def run_decode_window_prepare_step(
                self, initial_tokens, window_tokens_scratch, input_ids,
                hidden_states_scratch, position_ids, kv_lens_cuda,
                output_step_idx, input_tokens):
            window_prepare_calls.append(
                (initial_tokens, window_tokens_scratch, input_ids,
                 hidden_states_scratch, position_ids, kv_lens_cuda,
                 output_step_idx, input_tokens))
            hidden_states_scratch.window_prepare_written = True
            return hidden_states_scratch

    def fake_empty(shape, *, device=None, dtype=None):
        return FakeTensor(tuple(shape), dtype=dtype, device=device)

    fake_torch = SimpleNamespace(
        empty=fake_empty,
        classes=SimpleNamespace(trtllm=SimpleNamespace(
            DeepseekResidentDecodeHandle=FakeHandle)),
        ops=SimpleNamespace(trtllm=SimpleNamespace(
            cute_dsl_fp4_paged_mqa_logits=object())),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    contract = SimpleNamespace(
        hidden_size=7168,
        vocab_size=129280,
        layers=(SimpleNamespace(layer_idx=0, layer_kind="dense"), ),
    )
    engine = deepseek_resident_native.create_engine(
        model=_fake_native_resident_model(),
        contract=contract,
        dist=None,
    )
    invocation = SimpleNamespace(
        stable_shape_key=("shape", 16),
        real_batch_size=16,
        padded_batch_size=16,
        input_tokens=16,
        request_ids=tuple(range(16)),
        seq_lens=(1, ) * 16,
        cached_tokens=tuple(range(2048, 2064)),
    )
    input_ids = FakeTensor((16, ), dtype="torch.int64")

    first = engine.run_input_embedding_stage(
        input_ids=input_ids,
        invocation=invocation,
    )
    second = engine.run_input_embedding_stage(
        input_ids=input_ids,
        invocation=invocation,
    )
    normed = engine.run_layer_input_rmsnorm_stage(
        layer_idx=0,
        invocation=invocation,
        eps=1e-6,
    )
    gated = engine.run_layer_input_gated_norm_stage(
        layer_idx=0,
        invocation=invocation,
    )
    attention_tail = engine.run_layer_attention_output_tail_stage(
        layer_idx=0,
        invocation=invocation,
    )
    post_attention_normed = engine.run_layer_post_attention_rmsnorm_stage(
        layer_idx=0,
        invocation=invocation,
        eps=1e-6,
    )
    post_attention_gated = engine.run_layer_post_attention_gated_norm_stage(
        layer_idx=0,
        invocation=invocation,
    )
    dense_mlp_output = engine.run_layer_dense_mlp_stage(
        layer_idx=0,
        invocation=invocation,
    )
    next_layer_hidden = engine.run_layer_post_ffn_rmsnorm_stage(
        layer_idx=0,
        invocation=invocation,
        eps=1e-6,
    )
    logits = engine.run_lm_head_logits_stage(invocation=invocation)
    new_tokens = engine.run_greedy_sample_stage(invocation=invocation)
    initial_tokens = FakeTensor((1, 16, 1), dtype="torch.int32")
    position_ids = FakeTensor((1, 16), dtype="torch.int64")
    kv_lens_cuda = FakeTensor((16, ), dtype="torch.int32")
    prepared_hidden = engine.run_decode_window_prepare_step_stage(
        invocation=invocation,
        initial_tokens=initial_tokens,
        input_ids=input_ids,
        position_ids=position_ids,
        kv_lens_cuda=kv_lens_cuda,
        owned_steps=4,
        output_step_idx=1,
    )
    window_tokens = engine.run_decode_window_sample_step_stage(
        invocation=invocation,
        owned_steps=4,
        output_step_idx=2,
    )

    assert first is second
    assert first.shape == (16, 7168)
    assert first.embedding_written
    assert normed.shape == (16, 7168)
    assert normed is not first
    assert normed.rmsnorm_written
    assert gated.shape == (16, 7168)
    assert gated is not first
    assert gated is not normed
    assert gated.gated_norm_written
    assert attention_tail.shape == (16, 7168)
    assert attention_tail is not first
    assert attention_tail is not gated
    assert attention_tail.attention_tail_written
    assert post_attention_normed.shape == (16, 7168)
    assert post_attention_normed is not first
    assert post_attention_normed is not normed
    assert post_attention_normed is not gated
    assert post_attention_normed.post_attention_rmsnorm_written
    assert post_attention_gated.shape == (16, 7168)
    assert post_attention_gated is not first
    assert post_attention_gated is not normed
    assert post_attention_gated is not gated
    assert post_attention_gated is not post_attention_normed
    assert post_attention_gated.post_attention_gated_norm_written
    assert dense_mlp_output.shape == (16, 7168)
    assert dense_mlp_output is not first
    assert dense_mlp_output is not normed
    assert dense_mlp_output is not gated
    assert dense_mlp_output is not post_attention_normed
    assert dense_mlp_output is not post_attention_gated
    assert dense_mlp_output.dense_mlp_output_written
    assert next_layer_hidden.shape == (16, 7168)
    assert next_layer_hidden is not first
    assert next_layer_hidden is not dense_mlp_output
    assert next_layer_hidden.next_layer_hidden_written
    assert logits.shape == (16, 129280)
    assert logits.logits_written
    assert new_tokens.shape == (1, 16, 1)
    assert new_tokens.greedy_sample_written
    assert prepared_hidden is first
    assert prepared_hidden.window_prepare_written
    assert window_tokens.shape == (4, 16, 1)
    assert window_tokens.window_sample_written
    assert len(embedding_calls) == 2
    assert embedding_calls[0][2] == 16
    assert embedding_calls[0][3] == [3, 22]
    assert embedding_calls[0][4] == [0]
    assert len(embedding_calls[0][5]) == 22
    assert len(rmsnorm_calls) == 1
    assert rmsnorm_calls[0][0] == 0
    assert rmsnorm_calls[0][1] is first
    assert rmsnorm_calls[0][2] is normed
    assert rmsnorm_calls[0][3] == 16
    assert rmsnorm_calls[0][4] == 1e-6
    assert not rmsnorm_calls[0][5]
    assert len(gated_norm_calls) == 1
    assert gated_norm_calls[0][0] == 0
    assert gated_norm_calls[0][1] is normed
    assert gated_norm_calls[0][2] is gated
    assert gated_norm_calls[0][3] == 16
    assert len(attention_output_tail_calls) == 1
    assert attention_output_tail_calls[0][0] == 0
    assert attention_output_tail_calls[0][1].shape == (16, 16384)
    assert attention_output_tail_calls[0][1].attention_core_output_written
    assert attention_output_tail_calls[0][2] is gated
    assert attention_output_tail_calls[0][3].shape == (16, 16384)
    assert attention_output_tail_calls[0][3].attention_gate_written
    assert attention_output_tail_calls[0][4] is attention_tail
    assert attention_output_tail_calls[0][5] == 16
    assert len(post_attention_rmsnorm_calls) == 1
    assert post_attention_rmsnorm_calls[0][0] == 0
    assert post_attention_rmsnorm_calls[0][1] is attention_tail
    assert post_attention_rmsnorm_calls[0][2] is first
    assert post_attention_rmsnorm_calls[0][3] is post_attention_normed
    assert post_attention_rmsnorm_calls[0][5] == 16
    assert post_attention_rmsnorm_calls[0][6] == 1e-6
    assert not post_attention_rmsnorm_calls[0][7]
    assert len(post_attention_gated_norm_calls) == 1
    assert post_attention_gated_norm_calls[0][0] == 0
    assert post_attention_gated_norm_calls[0][1] is post_attention_normed
    assert post_attention_gated_norm_calls[0][2] is post_attention_gated
    assert post_attention_gated_norm_calls[0][3] == 16
    assert len(dense_mlp_calls) == 1
    assert dense_mlp_calls[0][0] == 0
    assert dense_mlp_calls[0][1] is post_attention_gated
    assert dense_mlp_calls[0][2].shape == (16, 2048)
    assert dense_mlp_calls[0][2].dense_mlp_intermediate_written
    assert dense_mlp_calls[0][3] is dense_mlp_output
    assert dense_mlp_calls[0][4] == 16
    assert len(post_ffn_rmsnorm_calls) == 1
    assert post_ffn_rmsnorm_calls[0][0] == 0
    assert post_ffn_rmsnorm_calls[0][1] is dense_mlp_output
    assert post_ffn_rmsnorm_calls[0][3] is next_layer_hidden
    assert post_ffn_rmsnorm_calls[0][5] == 16
    assert post_ffn_rmsnorm_calls[0][6] == 1e-6
    assert not post_ffn_rmsnorm_calls[0][7]
    assert len(lm_head_logits_calls) == 1
    assert lm_head_logits_calls[0][0] is next_layer_hidden
    assert lm_head_logits_calls[0][1] is logits
    assert lm_head_logits_calls[0][2] == 16
    assert len(sampling_calls) == 1
    assert sampling_calls[0][0] is logits
    assert sampling_calls[0][1] is new_tokens
    assert sampling_calls[0][2] == 16
    assert len(window_prepare_calls) == 1
    assert window_prepare_calls[0][0] is initial_tokens
    assert window_prepare_calls[0][1] is window_tokens
    assert window_prepare_calls[0][2] is input_ids
    assert window_prepare_calls[0][3] is first
    assert window_prepare_calls[0][4] is position_ids
    assert window_prepare_calls[0][5] is kv_lens_cuda
    assert window_prepare_calls[0][6] == 1
    assert window_prepare_calls[0][7] == 16
    assert len(window_sample_calls) == 1
    assert window_sample_calls[0][0] is logits
    assert window_sample_calls[0][1] is window_tokens
    assert window_sample_calls[0][2] == 2
    assert window_sample_calls[0][3] == 16
    assert engine._scheduler_state.embedding_stage_calls == 2
    assert engine._scheduler_state.input_rmsnorm_stage_calls == 1
    assert engine._scheduler_state.input_gated_norm_stage_calls == 1
    assert engine._scheduler_state.attention_output_tail_stage_calls == 1
    assert engine._scheduler_state.post_attention_rmsnorm_stage_calls == 1
    assert engine._scheduler_state.post_attention_gated_norm_stage_calls == 1
    assert engine._scheduler_state.moe_router_stage_calls == 0
    assert engine._scheduler_state.dense_mlp_stage_calls == 1
    assert engine._scheduler_state.post_ffn_rmsnorm_stage_calls == 1
    assert engine._scheduler_state.lm_head_logits_stage_calls == 1
    assert engine._scheduler_state.sampling_stage_calls == 1
    assert engine._scheduler_state.window_prepare_stage_calls == 1
    assert engine._scheduler_state.window_sample_stage_calls == 1
    assert engine.execution_state()["window_prepare_reason"] == (
        "resident_window_prepare_native_executed")
    assert engine.execution_state()["window_sample_reason"] == (
        "resident_window_sample_native_executed")
    assert engine.sample_backend_state() == {
        "backend": "deepseek_resident_sampler_native_v1",
        "ready": True,
        "reason": "resident_sampling_native_executed",
        "metadata": {
            "native": True,
            "sampling_stage_calls": 1,
            "last_sampling_reason": "resident_sampling_native_executed",
        },
    }
    assert moe_router_calls == []


def test_deepseek_native_engine_runs_moe_router_stage(monkeypatch):
    embedding_calls = []
    moe_router_calls = []

    class FakeTensor:

        def __init__(self, shape, *, dtype="torch.bfloat16", device="cuda:0"):
            self.shape = shape
            self.dtype = dtype
            self.device = device

    class FakeHandle:

        def __init__(self, layer_offsets, layer_kinds, layer_site_offsets,
                     layer_site_ids, layer_site_tensor_indices,
                     resident_tensors):
            self.layer_offsets = layer_offsets
            self.layer_kinds = layer_kinds
            self.layer_site_offsets = layer_site_offsets
            self.layer_site_ids = layer_site_ids
            self.layer_site_tensor_indices = layer_site_tensor_indices
            self.resident_tensors = resident_tensors

        def run_input_embedding(self, input_ids, hidden_states_scratch,
                                input_tokens):
            embedding_calls.append(
                (input_ids, hidden_states_scratch, input_tokens))
            return hidden_states_scratch

        def run_layer_moe_router(
                self, layer_idx, post_attention_gated_scratch,
                router_logits_scratch, router_scores_scratch,
                router_topk_indices_scratch, router_topk_weights_scratch,
                input_tokens, top_k, n_group, topk_group,
                routed_scaling_factor):
            moe_router_calls.append(
                (layer_idx, post_attention_gated_scratch,
                 router_logits_scratch, router_scores_scratch,
                 router_topk_indices_scratch, router_topk_weights_scratch,
                 input_tokens, top_k, n_group, topk_group,
                 routed_scaling_factor))
            router_logits_scratch.router_logits_written = True
            router_scores_scratch.router_scores_written = True
            router_topk_indices_scratch.router_topk_indices_written = True
            router_topk_weights_scratch.router_topk_weights_written = True
            return router_topk_weights_scratch

    def fake_empty(shape, *, device=None, dtype=None):
        return FakeTensor(tuple(shape), dtype=dtype, device=device)

    fake_torch = SimpleNamespace(
        float32="torch.float32",
        int32="torch.int32",
        empty=fake_empty,
        classes=SimpleNamespace(trtllm=SimpleNamespace(
            DeepseekResidentDecodeHandle=FakeHandle)),
        ops=SimpleNamespace(trtllm=SimpleNamespace()),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    contract = SimpleNamespace(
        hidden_size=7168,
        vocab_size=129280,
        num_experts_per_tok=8,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        layers=(
            SimpleNamespace(layer_idx=0, layer_kind="dense"),
            SimpleNamespace(layer_idx=1, layer_kind="moe", top_k=8),
        ),
    )
    engine = deepseek_resident_native.create_engine(
        model=_fake_native_resident_model_with_moe(),
        contract=contract,
        dist=None,
    )
    invocation = SimpleNamespace(
        stable_shape_key=("shape", 16),
        real_batch_size=16,
        padded_batch_size=16,
        input_tokens=16,
        request_ids=tuple(range(16)),
        seq_lens=(1, ) * 16,
        cached_tokens=tuple(range(2048, 2064)),
    )
    input_ids = FakeTensor((16, ), dtype="torch.int64")

    assert engine.run_input_embedding_stage(
        input_ids=input_ids,
        invocation=invocation,
    ).shape == (16, 7168)
    topk_weights = engine.run_layer_moe_router_stage(
        layer_idx=1,
        invocation=invocation,
    )

    assert topk_weights.shape == (16, 8)
    assert topk_weights.router_topk_weights_written
    assert len(embedding_calls) == 1
    assert len(moe_router_calls) == 1
    call = moe_router_calls[0]
    assert call[0] == 1
    assert call[1].shape == (16, 7168)
    assert call[2].shape == (16, 128)
    assert call[2].dtype == "torch.float32"
    assert call[2].router_logits_written
    assert call[3].shape == (16, 128)
    assert call[3].dtype == "torch.float32"
    assert call[3].router_scores_written
    assert call[4].shape == (16, 8)
    assert call[4].dtype == "torch.int32"
    assert call[4].router_topk_indices_written
    assert call[5] is topk_weights
    assert call[6:] == (16, 8, 8, 4, 2.5)
    assert engine._scheduler_state.moe_router_stage_calls == 1


def test_deepseek_native_decode_step_scheduler_stops_at_attention_core(
        monkeypatch):
    calls = []

    class FakeTensor:

        def __init__(self, shape, *, dtype="torch.bfloat16", device="cuda:0"):
            self.shape = shape
            self.dtype = dtype
            self.device = device

    class FakeHandle:

        def __init__(self, *args):
            pass

        def run_input_embedding(self, input_ids, hidden_states_scratch,
                                input_tokens):
            calls.append(("embedding", input_tokens))
            return hidden_states_scratch

        def run_layer_input_rmsnorm(self, layer_idx, hidden_states_scratch,
                                    norm_hidden_states_scratch, input_tokens,
                                    eps, use_gemma):
            calls.append(("input_rmsnorm", layer_idx, eps, use_gemma))
            return norm_hidden_states_scratch

        def run_layer_input_gated_norm(self, layer_idx,
                                       norm_hidden_states_scratch,
                                       gated_hidden_states_scratch,
                                       input_tokens):
            calls.append(("input_gated_norm", layer_idx, input_tokens))
            return gated_hidden_states_scratch

    def fake_empty(shape, *, device=None, dtype=None):
        return FakeTensor(tuple(shape), dtype=dtype, device=device)

    fake_torch = SimpleNamespace(
        empty=fake_empty,
        classes=SimpleNamespace(trtllm=SimpleNamespace(
            DeepseekResidentDecodeHandle=FakeHandle)),
        ops=SimpleNamespace(trtllm=SimpleNamespace(
            cute_dsl_fp4_paged_mqa_logits=object())),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    contract = SimpleNamespace(
        hidden_size=7168,
        vocab_size=129280,
        rms_norm_eps=1e-5,
        layers=(SimpleNamespace(layer_idx=0, layer_kind="dense"), ),
    )
    engine = deepseek_resident_native.create_engine(
        model=_fake_native_resident_model(),
        contract=contract,
        dist=None,
    )
    invocation = SimpleNamespace(
        stable_shape_key=("shape", 16),
        real_batch_size=16,
        padded_batch_size=16,
        input_tokens=16,
        request_ids=tuple(range(16)),
        seq_lens=(1, ) * 16,
        cached_tokens=tuple(range(2048, 2064)),
    )

    result = engine.run_decode_step_scheduler(
        input_ids=FakeTensor((16, ), dtype="torch.int64"),
        invocation=invocation,
    )

    assert not result.completed
    assert result.reason == "layer_0_attention_core_missing"
    assert result.stage == "attention_core"
    assert result.layer_idx == 0
    assert result.completed_layers == 0
    assert calls == [
        ("embedding", 16),
        ("input_rmsnorm", 0, 1e-5, False),
        ("input_gated_norm", 0, 16),
    ]
    assert engine._scheduler_state.stage_scheduler_calls == 1
    assert engine._scheduler_state.stage_scheduler_declines == 1
    assert (engine._scheduler_state.last_stage_scheduler_reason ==
            "layer_0_attention_core_missing")


def test_deepseek_native_decode_step_scheduler_reaches_moe_boundary(
        monkeypatch):
    calls = []

    class FakeTensor:

        def __init__(self, shape, *, dtype="torch.bfloat16", device="cuda:0"):
            self.shape = shape
            self.dtype = dtype
            self.device = device

    class FakeHandle:

        def __init__(self, *args):
            pass

        def run_input_embedding(self, input_ids, hidden_states_scratch,
                                input_tokens):
            calls.append(("embedding", input_tokens))
            return hidden_states_scratch

        def run_layer_input_rmsnorm(self, layer_idx, hidden_states_scratch,
                                    norm_hidden_states_scratch, input_tokens,
                                    eps, use_gemma):
            calls.append(("input_rmsnorm", layer_idx))
            return norm_hidden_states_scratch

        def run_layer_input_gated_norm(self, layer_idx,
                                       norm_hidden_states_scratch,
                                       gated_hidden_states_scratch,
                                       input_tokens):
            calls.append(("input_gated_norm", layer_idx,
                          norm_hidden_states_scratch.shape))
            return gated_hidden_states_scratch

        def run_layer_attention_output_tail(
                self, layer_idx, attention_core_output_scratch,
                attention_input_scratch, attention_gate_scratch,
                attention_hidden_states_scratch, input_tokens):
            calls.append(("attention_tail", layer_idx))
            return attention_hidden_states_scratch

        def run_layer_post_attention_rmsnorm(
                self, layer_idx, attention_hidden_states_scratch,
                residual_input_scratch, post_attention_norm_scratch,
                post_attention_residual_scratch, input_tokens, eps,
                use_gemma):
            calls.append(("post_attention_rmsnorm", layer_idx,
                          residual_input_scratch.shape))
            return post_attention_norm_scratch

        def run_layer_post_attention_gated_norm(
                self, layer_idx, post_attention_norm_scratch,
                post_attention_gated_scratch, input_tokens):
            calls.append(("post_attention_gated_norm", layer_idx))
            return post_attention_gated_scratch

        def run_layer_dense_mlp(self, layer_idx,
                                post_attention_gated_scratch,
                                dense_mlp_intermediate_scratch,
                                dense_mlp_output_scratch, input_tokens):
            calls.append(("dense_mlp", layer_idx))
            return dense_mlp_output_scratch

        def run_layer_moe_router(
                self, layer_idx, post_attention_gated_scratch,
                router_logits_scratch, router_scores_scratch,
                router_topk_indices_scratch, router_topk_weights_scratch,
                input_tokens, top_k, n_group, topk_group,
                routed_scaling_factor):
            calls.append(("moe_router", layer_idx, top_k, n_group,
                          topk_group, routed_scaling_factor))
            return router_topk_weights_scratch

        def run_layer_post_ffn_rmsnorm(
                self, layer_idx, dense_mlp_output_scratch,
                post_attention_residual_scratch, next_layer_hidden_scratch,
                next_layer_residual_scratch, input_tokens, eps, use_gemma):
            calls.append(("post_ffn_rmsnorm", layer_idx))
            return next_layer_hidden_scratch

        def run_lm_head_logits(self, next_layer_hidden_scratch,
                               logits_scratch, input_tokens):
            calls.append(("lm_head", input_tokens))
            return logits_scratch

    def fake_empty(shape, *, device=None, dtype=None):
        return FakeTensor(tuple(shape), dtype=dtype, device=device)

    def fake_attention_core(**kwargs):
        calls.append(("attention_core", kwargs["layer_idx"]))
        return True

    fake_torch = SimpleNamespace(
        float32="torch.float32",
        int32="torch.int32",
        empty=fake_empty,
        classes=SimpleNamespace(trtllm=SimpleNamespace(
            DeepseekResidentDecodeHandle=FakeHandle)),
        ops=SimpleNamespace(trtllm=SimpleNamespace()),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    contract = SimpleNamespace(
        hidden_size=7168,
        vocab_size=129280,
        rms_norm_eps=1e-6,
        num_experts_per_tok=8,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        layers=(
            SimpleNamespace(layer_idx=0, layer_kind="dense"),
            SimpleNamespace(layer_idx=1, layer_kind="moe", top_k=8),
        ),
    )
    engine = deepseek_resident_native.create_engine(
        model=_fake_native_resident_model_with_moe(),
        contract=contract,
        dist=None,
    )
    invocation = SimpleNamespace(
        stable_shape_key=("shape", 16),
        real_batch_size=16,
        padded_batch_size=16,
        input_tokens=16,
        request_ids=tuple(range(16)),
        seq_lens=(1, ) * 16,
        cached_tokens=tuple(range(2048, 2064)),
    )

    result = engine.run_decode_step_scheduler(
        input_ids=FakeTensor((16, ), dtype="torch.int64"),
        invocation=invocation,
        attention_core_runner=fake_attention_core,
    )

    assert not result.completed
    assert result.reason == "layer_1_moe_experts_missing"
    assert result.stage == "moe_experts"
    assert result.layer_idx == 1
    assert result.completed_layers == 1
    assert calls.count(("input_rmsnorm", 0)) == 1
    assert ("input_rmsnorm", 1) not in calls
    assert ("moe_router", 1, 8, 8, 4, 2.5) in calls
    assert ("lm_head", 16) not in calls
    assert engine._scheduler_state.stage_scheduler_calls == 1
    assert engine._scheduler_state.stage_scheduler_declines == 1
    assert engine._scheduler_state.moe_router_stage_calls == 1
    assert (engine._scheduler_state.last_stage_scheduler_reason ==
            "layer_1_moe_experts_missing")


def test_deepseek_native_decode_step_scheduler_uses_dsa_attention_bridge(
        monkeypatch):
    calls = []
    attention_core_calls = []

    class FakeTensor:

        def __init__(self, shape, *, dtype="torch.bfloat16", device="cuda:0"):
            self.shape = shape
            self.dtype = dtype
            self.device = device

    class FakeHandle:

        def __init__(self, *args):
            pass

        def run_input_embedding(self, input_ids, hidden_states_scratch,
                                input_tokens):
            calls.append(("embedding", input_tokens))
            return hidden_states_scratch

        def run_layer_input_rmsnorm(self, layer_idx, hidden_states_scratch,
                                    norm_hidden_states_scratch, input_tokens,
                                    eps, use_gemma):
            calls.append(("input_rmsnorm", layer_idx))
            return norm_hidden_states_scratch

        def run_layer_input_gated_norm(self, layer_idx,
                                       norm_hidden_states_scratch,
                                       gated_hidden_states_scratch,
                                       input_tokens):
            calls.append(("input_gated_norm", layer_idx))
            return gated_hidden_states_scratch

        def run_layer_attention_output_tail(
                self, layer_idx, attention_core_output_scratch,
                attention_input_scratch, attention_gate_scratch,
                attention_hidden_states_scratch, input_tokens):
            calls.append(("attention_tail", layer_idx,
                          attention_core_output_scratch.core_layer_idx))
            return attention_hidden_states_scratch

        def run_layer_post_attention_rmsnorm(
                self, layer_idx, attention_hidden_states_scratch,
                residual_input_scratch, post_attention_norm_scratch,
                post_attention_residual_scratch, input_tokens, eps,
                use_gemma):
            calls.append(("post_attention_rmsnorm", layer_idx))
            return post_attention_norm_scratch

        def run_layer_post_attention_gated_norm(
                self, layer_idx, post_attention_norm_scratch,
                post_attention_gated_scratch, input_tokens):
            calls.append(("post_attention_gated_norm", layer_idx))
            return post_attention_gated_scratch

        def run_layer_dense_mlp(self, layer_idx,
                                post_attention_gated_scratch,
                                dense_mlp_intermediate_scratch,
                                dense_mlp_output_scratch, input_tokens):
            calls.append(("dense_mlp", layer_idx))
            return dense_mlp_output_scratch

        def run_layer_moe_router(
                self, layer_idx, post_attention_gated_scratch,
                router_logits_scratch, router_scores_scratch,
                router_topk_indices_scratch, router_topk_weights_scratch,
                input_tokens, top_k, n_group, topk_group,
                routed_scaling_factor):
            calls.append(("moe_router", layer_idx))
            return router_topk_weights_scratch

        def run_layer_post_ffn_rmsnorm(
                self, layer_idx, dense_mlp_output_scratch,
                post_attention_residual_scratch, next_layer_hidden_scratch,
                next_layer_residual_scratch, input_tokens, eps, use_gemma):
            calls.append(("post_ffn_rmsnorm", layer_idx))
            return next_layer_hidden_scratch

    def fake_empty(shape, *, device=None, dtype=None):
        return FakeTensor(tuple(shape), dtype=dtype, device=device)

    def make_attention(layer_idx):

        def forward_impl_with_dsa(position_ids,
                                  hidden_states,
                                  attn_metadata,
                                  output,
                                  kv_proj_input=None):
            attention_core_calls.append(
                (layer_idx, position_ids, hidden_states, attn_metadata,
                 output, kv_proj_input))
            output.core_layer_idx = layer_idx

        return SimpleNamespace(
            kv_a_proj_with_mqa=_fake_native_weight_module((576, 7168)),
            gate_proj=_fake_native_weight_module((16384, 7168)),
            o_proj=_fake_native_weight_module((7168, 16384)),
            forward_impl_with_dsa=forward_impl_with_dsa,
        )

    fake_torch = SimpleNamespace(
        float32="torch.float32",
        int32="torch.int32",
        empty=fake_empty,
        classes=SimpleNamespace(trtllm=SimpleNamespace(
            DeepseekResidentDecodeHandle=FakeHandle)),
        ops=SimpleNamespace(trtllm=SimpleNamespace()),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    model = _fake_native_resident_model_with_moe()
    model.model.layers[0].self_attn = make_attention(0)
    model.model.layers[1].self_attn = make_attention(1)
    contract = SimpleNamespace(
        hidden_size=7168,
        vocab_size=129280,
        rms_norm_eps=1e-6,
        num_experts_per_tok=8,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        layers=(
            SimpleNamespace(layer_idx=0, layer_kind="dense"),
            SimpleNamespace(layer_idx=1, layer_kind="moe", top_k=8),
        ),
    )
    engine = deepseek_resident_native.create_engine(
        model=model,
        contract=contract,
        dist=None,
    )
    invocation = SimpleNamespace(
        stable_shape_key=("shape", 16),
        real_batch_size=16,
        padded_batch_size=16,
        input_tokens=16,
        request_ids=tuple(range(16)),
        seq_lens=(1, ) * 16,
        cached_tokens=tuple(range(2048, 2064)),
    )
    position_ids = FakeTensor((1, 16), dtype="torch.int64")
    attn_metadata = object()

    result = engine.run_decode_step_scheduler(
        input_ids=FakeTensor((16, ), dtype="torch.int64"),
        invocation=invocation,
        inputs={
            "position_ids": position_ids,
            "attn_metadata": attn_metadata,
        },
    )

    assert not result.completed
    assert result.reason == "layer_1_moe_experts_missing"
    assert len(attention_core_calls) == 2
    assert attention_core_calls[0][0] == 0
    assert attention_core_calls[0][1] is position_ids
    assert attention_core_calls[0][2].shape == (16, 7168)
    assert attention_core_calls[0][3] is attn_metadata
    assert attention_core_calls[0][4].shape == (16, 16384)
    assert attention_core_calls[0][5] is None
    assert attention_core_calls[1][0] == 1
    assert ("attention_tail", 0, 0) in calls
    assert ("attention_tail", 1, 1) in calls
    assert ("moe_router", 1) in calls


def test_deepseek_native_attention_core_prefers_split_dsa_bridge(monkeypatch):
    calls = []

    class FakeTensor:

        def __init__(self,
                     shape,
                     *,
                     dtype="torch.bfloat16",
                     device="cuda:0",
                     name="tensor"):
            self.shape = tuple(shape)
            self.dtype = dtype
            self.device = device
            self.name = name

    def fake_empty(shape, *, device=None, dtype=None):
        return FakeTensor(tuple(shape), dtype=dtype, device=device)

    fake_torch = SimpleNamespace(
        empty=fake_empty,
        classes=SimpleNamespace(trtllm=SimpleNamespace()),
        ops=SimpleNamespace(trtllm=SimpleNamespace()),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    model = _fake_native_resident_model()

    def forward_dsa_proj(position_ids,
                         hidden_states,
                         attn_metadata,
                         kv_proj_input=None):
        calls.append(("dsa_proj", position_ids, hidden_states.name,
                      attn_metadata, kv_proj_input))
        return [
            FakeTensor((16, 16384), name="q"),
            FakeTensor((16, 512), name="compressed_kv"),
            FakeTensor((16, 64), name="k_pe"),
            FakeTensor((16, 576), name="latent_cache"),
            FakeTensor((16, 8, 128), dtype="torch.float8_e4m3fn",
                       name="q_fp8"),
        ]

    def forward_dsa_attn(q, compressed_kv, k_pe, latent_cache,
                         indexer_intermediates, position_ids, attn_metadata,
                         output):
        calls.append(("dsa_attn", q.name, compressed_kv.name, k_pe.name,
                      latent_cache.name,
                      [tensor.name for tensor in indexer_intermediates],
                      position_ids, attn_metadata, output.shape))
        output.core_written = True

    model.model.layers[0].self_attn = SimpleNamespace(
        kv_a_proj_with_mqa=_fake_native_weight_module((576, 7168)),
        gate_proj=_fake_native_weight_module((16384, 7168)),
        o_proj=_fake_native_weight_module((7168, 16384)),
        forward_dsa_proj=forward_dsa_proj,
        forward_dsa_attn=forward_dsa_attn,
        forward_impl_with_dsa=lambda *args, **kwargs: calls.append(
            ("forward_impl_with_dsa", )),
    )
    contract = SimpleNamespace(
        hidden_size=7168,
        vocab_size=129280,
        layers=(SimpleNamespace(layer_idx=0, layer_kind="dense"), ),
    )
    engine = deepseek_resident_native.create_engine(
        model=model,
        contract=contract,
        dist=None,
    )
    invocation = SimpleNamespace(
        stable_shape_key=("shape", 16),
        real_batch_size=16,
        padded_batch_size=16,
        input_tokens=16,
        request_ids=tuple(range(16)),
        seq_lens=(1, ) * 16,
        cached_tokens=tuple(range(2048, 2064)),
    )
    input_ids = FakeTensor((16, ), dtype="torch.int64", name="input_ids")
    state = engine._get_shape_state(invocation, input_ids)
    state.scratch["gated_hidden_states"] = FakeTensor((16, 7168),
                                                      name="gated_hidden")
    position_ids = FakeTensor((1, 16), dtype="torch.int64", name="positions")
    attn_metadata = object()

    assert engine.run_layer_attention_core_stage(
        layer_idx=0,
        invocation=invocation,
        inputs={
            "position_ids": position_ids,
            "attn_metadata": attn_metadata,
        },
    )

    assert calls[0] == ("dsa_proj", position_ids, "gated_hidden",
                        attn_metadata, None)
    assert calls[1] == ("dsa_attn", "q", "compressed_kv", "k_pe",
                        "latent_cache", ["q_fp8"], position_ids,
                        attn_metadata, (16, 16384))
    assert ("forward_impl_with_dsa", ) not in calls
    proj_outputs = state.scratch["attention_dsa_proj_outputs"][0]
    assert [tensor.name for tensor in proj_outputs] == [
        "q", "compressed_kv", "k_pe", "latent_cache", "q_fp8"
    ]
    tail_scratch = state.scratch["attention_output_tail_states"][0]
    assert tail_scratch["attention_core_output"].core_written
    state_dict = engine.execution_state()
    assert state_dict["attention_dsa_proj_stage_calls"] == 1
    assert state_dict["attention_dsa_proj_reason"] == (
        "resident_attention_dsa_proj_bridge_executed")
    assert state_dict["attention_dsa_attn_stage_calls"] == 1
    assert state_dict["attention_dsa_attn_reason"] == (
        "resident_attention_dsa_attn_bridge_executed")


def test_deepseek_native_attention_core_records_dsa_dispatch_not_ready(
        monkeypatch):
    calls = []

    class FakeTensor:

        def __init__(self,
                     shape,
                     *,
                     dtype="torch.bfloat16",
                     device="cuda:0",
                     name="tensor"):
            self.shape = tuple(shape)
            self.dtype = dtype
            self.device = device
            self.name = name

    class FakeHandle:

        def __init__(self, *args):
            pass

        def run_layer_dsa_attention_dispatch_ready(self):
            return False

        def run_layer_dsa_attention_dispatch_not_ready_reason(self):
            return "resident_attention_dsa_dispatch_native_not_implemented"

        def run_layer_dsa_attention_dispatch(self, *args):
            calls.append(("native_dispatch", args))
            return args[8]

    def fake_empty(shape, *, device=None, dtype=None):
        return FakeTensor(tuple(shape), dtype=dtype, device=device)

    fake_torch = SimpleNamespace(
        empty=fake_empty,
        classes=SimpleNamespace(trtllm=SimpleNamespace(
            DeepseekResidentDecodeHandle=FakeHandle)),
        ops=SimpleNamespace(trtllm=SimpleNamespace(
            cute_dsl_fp4_paged_mqa_logits=object())),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def forward_dsa_proj(*args, **kwargs):
        return [
            FakeTensor((16, 16384), name="q"),
            FakeTensor((16, 512), name="compressed_kv"),
            FakeTensor((16, 64), name="k_pe"),
            FakeTensor((16, 576), name="latent_cache"),
        ]

    def forward_dsa_attn(*args):
        calls.append(("python_dispatch", args[0].name, args[7].shape))
        args[7].python_dispatch_written = True

    model = _fake_native_resident_model()
    model.model.layers[0].self_attn = SimpleNamespace(
        kv_a_proj_with_mqa=_fake_native_weight_module((576, 7168)),
        gate_proj=_fake_native_weight_module((16384, 7168)),
        o_proj=_fake_native_weight_module((7168, 16384)),
        forward_dsa_proj=forward_dsa_proj,
        forward_dsa_attn=forward_dsa_attn,
        forward_impl_with_dsa=lambda *args, **kwargs: None,
    )
    engine = deepseek_resident_native.create_engine(
        model=model,
        contract=SimpleNamespace(
            hidden_size=7168,
            vocab_size=129280,
            layers=(SimpleNamespace(layer_idx=0, layer_kind="dense"), ),
        ),
        dist=None,
    )
    invocation = SimpleNamespace(
        stable_shape_key=("shape", 16),
        real_batch_size=16,
        padded_batch_size=16,
        input_tokens=16,
        request_ids=tuple(range(16)),
        seq_lens=(1, ) * 16,
        cached_tokens=tuple(range(2048, 2064)),
    )
    state = engine._get_shape_state(
        invocation, FakeTensor((16, ), dtype="torch.int64"))
    state.scratch["gated_hidden_states"] = FakeTensor((16, 7168),
                                                      name="gated_hidden")

    assert engine.run_layer_attention_core_stage(
        layer_idx=0,
        invocation=invocation,
        inputs={
            "position_ids":
            FakeTensor((1, 16), dtype="torch.int64", name="position_ids"),
            "attn_metadata":
            SimpleNamespace(
                seq_lens_cuda=FakeTensor((16, ), dtype="torch.int32"),
                kv_lens_cuda_runtime=FakeTensor((16, ), dtype="torch.int32"),
            ),
        },
    )

    assert calls == [("python_dispatch", "q", (16, 16384))]
    state_dict = engine.execution_state()
    assert state_dict["attention_dsa_native_dispatch_stage_calls"] == 0
    assert state_dict["attention_dsa_native_dispatch_reason"] == (
        "resident_attention_dsa_dispatch_native_not_implemented")
    assert state_dict["attention_dsa_attn_reason"] == (
        "resident_attention_dsa_attn_bridge_executed")


def test_deepseek_native_attention_core_uses_ready_native_dsa_dispatch(
        monkeypatch):
    calls = []

    class FakeTensor:

        def __init__(self,
                     shape,
                     *,
                     dtype="torch.bfloat16",
                     device="cuda:0",
                     name="tensor"):
            self.shape = tuple(shape)
            self.dtype = dtype
            self.device = device
            self.name = name

    class FakeHandle:

        def __init__(self, *args):
            pass

        def run_layer_dsa_attention_dispatch_ready(self):
            return True

        def run_layer_dsa_attention_dispatch(self, layer_idx, q,
                                             compressed_kv, k_pe,
                                             latent_cache,
                                             indexer_intermediates,
                                             position_ids, seq_lens_cuda,
                                             kv_lens_cuda,
                                             metadata_tensors,
                                             runtime_tensors,
                                             runtime_config,
                                             runtime_scalars,
                                             scratch_tensors, output,
                                             input_tokens):
            calls.append(("native_dispatch", layer_idx, q.name,
                          compressed_kv.name, k_pe.name, latent_cache.name,
                          [tensor.name for tensor in indexer_intermediates],
                          position_ids.name, seq_lens_cuda.name,
                          kv_lens_cuda.name,
                          [tensor.name for tensor in metadata_tensors],
                          sorted(runtime_tensors),
                          runtime_config["tokens_per_block"],
                          runtime_config["kv_lora_rank"],
                          runtime_scalars["softmax_scale"],
                          [tensor.shape for tensor in scratch_tensors],
                          output.shape, input_tokens))
            output.native_dispatch_written = True
            return output

    def fake_empty(shape, *, device=None, dtype=None):
        return FakeTensor(tuple(shape), dtype=dtype, device=device)

    fake_torch = SimpleNamespace(
        empty=fake_empty,
        classes=SimpleNamespace(trtllm=SimpleNamespace(
            DeepseekResidentDecodeHandle=FakeHandle)),
        ops=SimpleNamespace(trtllm=SimpleNamespace(
            cute_dsl_fp4_paged_mqa_logits=object())),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def forward_dsa_proj(*args, **kwargs):
        return [
            FakeTensor((16, 16384), name="q"),
            FakeTensor((16, 512), name="compressed_kv"),
            FakeTensor((16, 64), name="k_pe"),
            FakeTensor((16, 576), name="latent_cache"),
            FakeTensor((16, 8, 128), name="q_fp8"),
            FakeTensor((16, 8, 128), name="k_fp8"),
            FakeTensor((16, 8), dtype="torch.float32", name="k_scale"),
            FakeTensor((16, 8), dtype="torch.float32", name="weights"),
            FakeTensor((16, 8), dtype="torch.int32", name="q_scale"),
        ]

    def sparse_attn_indexer(attn_metadata,
                            q,
                            q_fp8,
                            k_fp8,
                            k_scale,
                            weights,
                            *,
                            q_scale=None):
        calls.append(("topk", q.name, q_fp8.name, k_fp8.name, k_scale.name,
                      weights.name, q_scale.name))
        return FakeTensor((16, 2048),
                          dtype="torch.int32",
                          name="topk_indices")

    def forward_dsa_attn(*args):
        calls.append(("python_dispatch", ))

    def kvarn_restore_for_decode(attn_metadata):
        calls.append(("kvarn_restore", attn_metadata.kv_cache_manager.dtype.name))

    model = _fake_native_resident_model()
    model.model.layers[0].self_attn = SimpleNamespace(
        kv_a_proj_with_mqa=_fake_native_weight_module((576, 7168)),
        k_b_proj_trans=FakeTensor((128, 512, 128), name="k_b_proj_trans"),
        v_b_proj=FakeTensor((128, 128, 512), name="v_b_proj"),
        gate_proj=_fake_native_weight_module((16384, 7168)),
        o_proj=_fake_native_weight_module((7168, 16384)),
        forward_dsa_proj=forward_dsa_proj,
        forward_dsa_attn=forward_dsa_attn,
        forward_impl_with_dsa=lambda *args, **kwargs: None,
        mqa=SimpleNamespace(indexer=SimpleNamespace(
            n_heads=64,
            head_dim=128,
            rope_dim=64,
            index_topk=2048,
            index_topk_step_freq=8,
            _xstep_recency_patch=True,
            sparse_attn_indexer=sparse_attn_indexer),
                            rotary_cos_sin=FakeTensor(
                                (132096, 64),
                                dtype="torch.float32",
                                name="rotary_cos_sin"),
                            predicted_tokens_per_seq=1,
                            num_heads=128,
                            num_kv_heads=1,
                            head_dim=576,
                            quant_mode=0,
                            q_lora_rank=1536,
                            kv_lora_rank=512,
                            qk_nope_head_dim=128,
                            qk_rope_head_dim=64,
                            v_head_dim=128,
                            rope_append=False,
                            q_scaling=1.0,
                            kvarn_restore_for_decode=kvarn_restore_for_decode),
    )
    engine = deepseek_resident_native.create_engine(
        model=model,
        contract=SimpleNamespace(
            hidden_size=7168,
            vocab_size=129280,
            layers=(SimpleNamespace(layer_idx=0, layer_kind="dense"), ),
        ),
        dist=None,
    )
    invocation = SimpleNamespace(
        stable_shape_key=("shape", 16),
        real_batch_size=16,
        padded_batch_size=16,
        input_tokens=16,
        request_ids=tuple(range(16)),
        seq_lens=(1, ) * 16,
        cached_tokens=tuple(range(2048, 2064)),
    )
    state = engine._get_shape_state(
        invocation, FakeTensor((16, ), dtype="torch.int64"))
    state.scratch["gated_hidden_states"] = FakeTensor((16, 7168),
                                                      name="gated_hidden")

    attention_inputs = {
        "position_ids":
        FakeTensor((1, 16), dtype="torch.int64", name="position_ids"),
        "attn_metadata":
        SimpleNamespace(
            block_table=FakeTensor((16, 4096),
                                   dtype="torch.int32",
                                   name="block_table"),
            kv_cache_manager=SimpleNamespace(
                get_indexer_k_cache_buffers=lambda layer_idx: FakeTensor(
                    (4096, 128, 1, 64),
                    dtype="torch.int8",
                    name="indexer_k_cache"),
                get_unique_primary_pool=lambda: FakeTensor(
                    (4096, 1, 1, 1, 64, 288),
                    dtype="torch.int8",
                    name="dense_kv_pool"),
                get_dense_block_scale_pool=lambda: FakeTensor(
                    (4096, 1, 1, 1, 64, 36),
                    dtype="torch.uint8",
                    name="dense_kv_scale_pool"),
                dtype=SimpleNamespace(name="NVFP4"),
                kvarn_enabled=False,
                kvarn_cfg=None,
                use_fp4=True,
                tokens_per_block=64,
                index_head_dim=128,
                quant_block_size=128),
            indexer_k_cache_block_offsets=FakeTensor(
                (16, 4096),
                dtype="torch.int32",
                name="indexer_offsets"),
            scheduler_metadata_buffer=FakeTensor(
                (16, 8),
                dtype="torch.int32",
                name="scheduler_meta"),
            slot_mapping_fp8=FakeTensor((16, ),
                                        dtype="torch.int64",
                                        name="slot_mapping_fp8"),
            slot_mapping_scale=FakeTensor((16, ),
                                          dtype="torch.int64",
                                          name="slot_mapping_scale"),
            gen_kv_indptr=FakeTensor((17, ),
                                     dtype="torch.int32",
                                     name="gen_kv_indptr"),
            gen_cached_token_indptr=FakeTensor(
                (17, ),
                dtype="torch.int32",
                name="gen_cached_token_indptr"),
            kv_lens_cuda_2d=FakeTensor((16, 1),
                                       dtype="torch.int32",
                                       name="kv_lens_cuda_2d"),
            seq_lens_cuda=FakeTensor((16, ),
                                     dtype="torch.int32",
                                     name="seq_lens"),
            kv_lens_cuda_runtime=FakeTensor((16, ),
                                            dtype="torch.int32",
                                            name="kv_lens"),
            req_idx_per_token=FakeTensor((16, ),
                                         dtype="torch.int32",
                                         name="req_idx_per_token"),
            kv_cache_block_offsets=FakeTensor(
                (1, ),
                dtype="torch.int64",
                name="kv_cache_block_offsets"),
            host_kv_cache_pool_pointers=FakeTensor(
                (1, ),
                dtype="torch.int64",
                device="cpu",
                name="host_pool_pointers"),
            host_kv_cache_pool_mapping=FakeTensor(
                (1, ),
                dtype="torch.int32",
                device="cpu",
                name="host_pool_mapping"),
            kv_lens_runtime=FakeTensor((16, ),
                                       dtype="torch.int32",
                                       device="cpu",
                                       name="kv_lens_runtime"),
            prompt_lens_cpu_runtime=FakeTensor(
                (16, ),
                dtype="torch.int32",
                device="cpu",
                name="prompt_lens_cpu"),
            max_seq_len=132096,
            num_seqs=16,
            num_contexts=0,
            num_ctx_tokens=0,
            num_generations=16,
            num_sparse_topk=2048,
            beam_width=1,
        ),
    }
    plan = engine._build_dsa_window_plan(
        invocation=invocation,
        inputs=attention_inputs,
        state=state,
    )

    assert plan is not None
    assert [layer_plan.layer_idx for layer_plan in plan] == [0]
    assert [tensor.name for tensor in plan[0].static_metadata_tensors] == [
        "block_table", "indexer_k_cache", "indexer_offsets",
        "scheduler_meta", "slot_mapping_fp8", "slot_mapping_scale",
        "gen_kv_indptr", "gen_cached_token_indptr", "kv_lens_cuda_2d"
    ]
    assert sorted(plan[0].runtime_tensors) == [
        "dense_kv_pool", "dense_kv_scale_pool", "dsa_req_idx_per_token",
        "host_kv_cache_pool_mapping", "host_kv_cache_pool_pointers",
        "kv_cache_block_offsets", "kv_lens_runtime",
        "prompt_lens_cpu_runtime", "rotary_cos_sin"
    ]
    assert plan[0].runtime_config["tokens_per_block"] == 64
    assert plan[0].runtime_config["kv_lora_rank"] == 512
    assert plan[0].runtime_config["num_heads"] == 128
    assert plan[0].runtime_config["resident_indexer_num_heads"] == 64
    assert plan[0].runtime_config["resident_indexer_step_freq"] == 8
    assert plan[0].runtime_config["resident_indexer_step_recency_patch"] == 1
    assert plan[0].runtime_scalars["softmax_scale"] == 1.0
    assert plan[0].scratch_shapes == ((16, 128, 576), (16, 128, 512),
                                      (17, ), (17, ), (1, ), (16, 64, 64),
                                      (16, 64), (16, 1), (16, 64),
                                      (16, 64, 1), (16, 2048), (16, ))
    assert engine.execution_state()["dsa_window_plan_reason"] == (
        "resident_dsa_window_plan_ready")

    assert engine.run_layer_attention_core_stage(
        layer_idx=0,
        invocation=invocation,
        inputs=attention_inputs,
    )

    assert calls == [
        ("topk", "q", "q_fp8", "k_fp8", "k_scale", "weights", "q_scale"),
        ("native_dispatch", 0, "q", "compressed_kv", "k_pe", "latent_cache", [
            "q_fp8", "k_fp8", "k_scale", "weights", "q_scale"
        ], "position_ids", "seq_lens", "kv_lens", [
            "topk_indices", "block_table", "indexer_k_cache",
            "indexer_offsets", "scheduler_meta", "slot_mapping_fp8",
            "slot_mapping_scale", "gen_kv_indptr",
            "gen_cached_token_indptr", "kv_lens_cuda_2d"
        ], [
            "dense_kv_pool", "dense_kv_scale_pool", "dsa_req_idx_per_token",
            "host_kv_cache_pool_mapping", "host_kv_cache_pool_pointers",
            "kv_cache_block_offsets", "kv_lens_runtime",
            "prompt_lens_cpu_runtime", "rotary_cos_sin"
        ], 64, 512, 1.0, [(16, 128, 576), (16, 128, 512), (17, ), (17, ),
                          (1, )],
         (16, 16384), 16),
    ]
    output = state.scratch["attention_output_tail_states"][0][
        "attention_core_output"]
    assert output.native_dispatch_written
    state_dict = engine.execution_state()
    assert state_dict["attention_dsa_native_dispatch_stage_calls"] == 1
    assert state_dict["attention_dsa_native_dispatch_reason"] == (
        "resident_attention_dsa_dispatch_native_executed")
    assert state_dict["attention_dsa_attn_stage_calls"] == 1
    assert state_dict["attention_dsa_attn_reason"] == (
        "resident_attention_dsa_attn_native_executed")


def test_deepseek_native_dsa_window_plan_uses_standard_mla_for_kvarn_fp8(
        monkeypatch):
    class FakeTensor:

        def __init__(self,
                     shape,
                     *,
                     dtype="torch.bfloat16",
                     device="cuda:0",
                     name="tensor"):
            self.shape = tuple(shape)
            self.dtype = dtype
            self.device = device
            self.name = name

    class FakeHandle:

        def __init__(self, *args):
            pass

        def run_layer_dsa_attention_dispatch_ready(self):
            return True

        def run_layer_dsa_attention_dispatch(self, *args):
            return args[-2]

    def fake_empty(shape, *, device=None, dtype=None):
        return FakeTensor(tuple(shape), dtype=dtype, device=device)

    def get_dense_block_scale_pool():
        raise AssertionError(
            "Dense block-scale pool is only present for NVFP4 KV cache")

    fake_torch = SimpleNamespace(
        empty=fake_empty,
        classes=SimpleNamespace(trtllm=SimpleNamespace(
            DeepseekResidentDecodeHandle=FakeHandle)),
        ops=SimpleNamespace(trtllm=SimpleNamespace(
            cute_dsl_fp4_paged_mqa_logits=object())),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    model = _fake_native_resident_model()
    model.model.layers[0].self_attn = SimpleNamespace(
        k_b_proj_trans=FakeTensor((128, 512, 128), name="k_b_proj_trans"),
        v_b_proj=FakeTensor((128, 128, 512), name="v_b_proj"),
        softmax_scale=1.0,
        mqa=SimpleNamespace(
            indexer=SimpleNamespace(n_heads=64,
                                    head_dim=128,
                                    rope_dim=64,
                                    index_topk=2048),
            rotary_cos_sin=FakeTensor((132096, 64),
                                      dtype="torch.float32",
                                      name="rotary_cos_sin"),
            kvarn_restore_for_decode=lambda attn_metadata: None,
            predicted_tokens_per_seq=1,
            num_heads=128,
            num_kv_heads=1,
            head_dim=576,
            quant_mode=0,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            rope_append=False,
            q_scaling=1.0,
        ), )
    engine = deepseek_resident_native.create_engine(
        model=model,
        contract=SimpleNamespace(
            hidden_size=7168,
            vocab_size=129280,
            layers=(SimpleNamespace(layer_idx=0, layer_kind="dense"), ),
        ),
        dist=None,
    )
    invocation = SimpleNamespace(
        stable_shape_key=("shape", 16),
        real_batch_size=16,
        padded_batch_size=16,
        input_tokens=16,
        request_ids=tuple(range(16)),
        seq_lens=(1, ) * 16,
        cached_tokens=tuple(range(2048, 2064)),
    )
    state = engine._get_shape_state(
        invocation, FakeTensor((16, ), dtype="torch.int64"))
    attention_inputs = {
        "attn_metadata":
        SimpleNamespace(
            block_table=FakeTensor((16, 4096),
                                   dtype="torch.int32",
                                   name="block_table"),
            effective_workspace=FakeTensor((4096, ),
                                           dtype="torch.uint8",
                                           name="attention_workspace"),
            host_total_kv_lens=FakeTensor((2, ),
                                          dtype="torch.int32",
                                          device="cpu",
                                          name="host_total_kv_lens"),
            prompt_lens_cuda_runtime=FakeTensor(
                (16, ),
                dtype="torch.int32",
                name="prompt_lens_cuda"),
            host_request_types_runtime=FakeTensor(
                (16, ),
                dtype="torch.int32",
                device="cpu",
                name="host_request_types"),
            kv_cache_manager=SimpleNamespace(
                get_indexer_k_cache_buffers=lambda layer_idx: FakeTensor(
                    (4096, 128, 1, 64),
                    dtype="torch.int8",
                    name="indexer_k_cache"),
                get_unique_primary_pool=lambda: FakeTensor(
                    (4096, 1, 1, 1, 64, 288),
                    dtype="torch.int8",
                    name="dense_kv_pool"),
                get_dense_block_scale_pool=get_dense_block_scale_pool,
                dtype=SimpleNamespace(name="FP8"),
                kvarn_enabled=True,
                kvarn_cfg=SimpleNamespace(),
                tokens_per_block=64,
                index_head_dim=128,
                quant_block_size=128,
            ),
            indexer_k_cache_block_offsets=FakeTensor(
                (16, 4096),
                dtype="torch.int32",
                name="indexer_offsets"),
            scheduler_metadata_buffer=FakeTensor(
                (16, 8),
                dtype="torch.int32",
                name="scheduler_meta"),
            slot_mapping_fp8=FakeTensor((16, ),
                                        dtype="torch.int64",
                                        name="slot_mapping_fp8"),
            slot_mapping_scale=FakeTensor((16, ),
                                          dtype="torch.int64",
                                          name="slot_mapping_scale"),
            gen_kv_indptr=FakeTensor((17, ),
                                     dtype="torch.int32",
                                     name="gen_kv_indptr"),
            gen_cached_token_indptr=FakeTensor(
                (17, ),
                dtype="torch.int32",
                name="gen_cached_token_indptr"),
            kv_lens_cuda_2d=FakeTensor((16, 1),
                                       dtype="torch.int32",
                                       name="kv_lens_cuda_2d"),
            req_idx_per_token=FakeTensor((16, ),
                                         dtype="torch.int32",
                                         name="req_idx_per_token"),
            kv_cache_block_offsets=FakeTensor(
                (1, ),
                dtype="torch.int64",
                name="kv_cache_block_offsets"),
            host_kv_cache_pool_pointers=FakeTensor(
                (1, ),
                dtype="torch.int64",
                device="cpu",
                name="host_pool_pointers"),
            host_kv_cache_pool_mapping=FakeTensor(
                (1, ),
                dtype="torch.int32",
                device="cpu",
                name="host_pool_mapping"),
            kv_lens_runtime=FakeTensor((16, ),
                                       dtype="torch.int32",
                                       device="cpu",
                                       name="kv_lens_runtime"),
            prompt_lens_cpu_runtime=FakeTensor(
                (16, ),
                dtype="torch.int32",
                device="cpu",
                name="prompt_lens_cpu"),
            max_seq_len=132096,
            max_num_requests=16,
            max_context_length=132096,
            num_sparse_topk=2048,
            num_seqs=16,
            num_contexts=0,
            num_ctx_tokens=0,
            num_generations=16,
            beam_width=1,
        ),
    }
    plan = engine._build_dsa_window_plan(
        invocation=invocation,
        inputs=attention_inputs,
        state=state,
    )

    assert plan is not None
    assert plan[0].runtime_config["kv_dispatch_mode"] == 1
    assert plan[0].runtime_config["num_sparse_topk"] == 2048
    assert plan[0].runtime_config["num_heads"] == 128
    assert plan[0].runtime_config["resident_indexer_num_heads"] == 64
    assert plan[0].runtime_config["resident_indexer_step_freq"] == 1
    assert plan[0].runtime_config["resident_indexer_step_recency_patch"] == 0
    assert "dense_kv_scale_pool" not in plan[0].runtime_tensors
    assert sorted(plan[0].runtime_tensors) == [
        "attention_workspace", "dense_kv_pool", "dsa_req_idx_per_token",
        "host_kv_cache_pool_mapping", "host_kv_cache_pool_pointers",
        "host_request_types_runtime", "host_total_kv_lens",
        "kv_cache_block_offsets", "kv_lens_runtime",
        "prompt_lens_cpu_runtime", "prompt_lens_cuda_runtime",
        "rotary_cos_sin"
    ]
    assert plan[0].scratch_shapes == ((16, 128, 576), (16, 128, 512),
                                      (17, ), (17, ), (1, ), (2, ), (1, ),
                                      (16, 128, 576), (16, 64, 64),
                                      (16, 64), (16, 1), (16, 64),
                                      (16, 64, 1), (16, 2048), (16, ))
    assert engine.execution_state()["dsa_window_plan_reason"] == (
        "resident_dsa_window_plan_ready")


def test_deepseek_native_attention_core_ready_native_dispatch_requires_metadata(
        monkeypatch):
    calls = []

    class FakeTensor:

        def __init__(self,
                     shape,
                     *,
                     dtype="torch.bfloat16",
                     device="cuda:0",
                     name="tensor"):
            self.shape = tuple(shape)
            self.dtype = dtype
            self.device = device
            self.name = name

    class FakeHandle:

        def __init__(self, *args):
            pass

        def run_layer_dsa_attention_dispatch_ready(self):
            return True

        def run_layer_dsa_attention_dispatch(self, *args):
            calls.append(("native_dispatch", args))
            return args[-2]

    def fake_empty(shape, *, device=None, dtype=None):
        return FakeTensor(tuple(shape), dtype=dtype, device=device)

    fake_torch = SimpleNamespace(
        empty=fake_empty,
        classes=SimpleNamespace(trtllm=SimpleNamespace(
            DeepseekResidentDecodeHandle=FakeHandle)),
        ops=SimpleNamespace(trtllm=SimpleNamespace()),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    model = _fake_native_resident_model()
    model.model.layers[0].self_attn = SimpleNamespace(
        kv_a_proj_with_mqa=_fake_native_weight_module((576, 7168)),
        gate_proj=_fake_native_weight_module((16384, 7168)),
        o_proj=_fake_native_weight_module((7168, 16384)),
        forward_dsa_proj=lambda *args, **kwargs: [
            FakeTensor((16, 16384), name="q"),
            FakeTensor((16, 512), name="compressed_kv"),
            FakeTensor((16, 64), name="k_pe"),
            FakeTensor((16, 576), name="latent_cache"),
            FakeTensor((16, 8, 128), name="q_fp8"),
            FakeTensor((16, 8, 128), name="k_fp8"),
            FakeTensor((16, 8), dtype="torch.float32", name="k_scale"),
            FakeTensor((16, 8), dtype="torch.float32", name="weights"),
            FakeTensor((16, 8), dtype="torch.int32", name="q_scale"),
        ],
        forward_dsa_attn=lambda *args: calls.append(("python_dispatch", )),
        forward_impl_with_dsa=lambda *args, **kwargs: None,
        mqa=SimpleNamespace(indexer=SimpleNamespace(
            n_heads=64,
            head_dim=128,
            rope_dim=64,
            index_topk=2048,
            sparse_attn_indexer=lambda *args, **kwargs: calls.append(
                ("topk", )) or FakeTensor((16, 2048),
                                          dtype="torch.int32",
                                          name="topk_indices")),
                            has_fp4_kv_cache=True),
    )
    engine = deepseek_resident_native.create_engine(
        model=model,
        contract=SimpleNamespace(
            hidden_size=7168,
            vocab_size=129280,
            layers=(SimpleNamespace(layer_idx=0, layer_kind="dense"), ),
        ),
        dist=None,
    )
    invocation = SimpleNamespace(
        stable_shape_key=("shape", 16),
        real_batch_size=16,
        padded_batch_size=16,
        input_tokens=16,
        request_ids=tuple(range(16)),
        seq_lens=(1, ) * 16,
        cached_tokens=tuple(range(2048, 2064)),
    )
    state = engine._get_shape_state(
        invocation, FakeTensor((16, ), dtype="torch.int64"))
    state.scratch["gated_hidden_states"] = FakeTensor((16, 7168),
                                                      name="gated_hidden")

    assert not engine.run_layer_attention_core_stage(
        layer_idx=0,
        invocation=invocation,
        inputs={
            "position_ids":
            FakeTensor((1, 16), dtype="torch.int64", name="position_ids"),
            "attn_metadata":
            SimpleNamespace(
                seq_lens_cuda=FakeTensor((16, ),
                                         dtype="torch.int32",
                                         name="seq_lens"),
                kv_lens_cuda_runtime=FakeTensor((16, ),
                                                dtype="torch.int32",
                                                name="kv_lens"),
            ),
        },
    )

    assert calls == [("topk", )]
    state_dict = engine.execution_state()
    assert state_dict["attention_dsa_native_dispatch_stage_calls"] == 0
    assert state_dict["attention_dsa_native_dispatch_reason"] == (
        "resident_attention_dsa_dispatch_block_table_missing")
    assert state_dict["attention_dsa_attn_stage_calls"] == 0


def test_deepseek_native_decode_step_scheduler_bridges_moe_experts(
        monkeypatch):
    calls = []

    class FakeTensor:

        def __init__(self, shape, *, dtype="torch.bfloat16", device="cuda:0"):
            self.shape = shape
            self.dtype = dtype
            self.device = device
            self.copied_from = None

        def add_(self, other):
            calls.append(("add", self.shape, other.shape))
            return FakeTensor(self.shape, dtype=self.dtype, device=self.device)

        def copy_(self, other):
            self.copied_from = other
            calls.append(("copy", self.shape, other.shape))
            return self

    class FakeExperts:
        has_nvfp4 = False

        def __call__(self, hidden_states, router_logits, **kwargs):
            calls.append(("experts", hidden_states.shape, router_logits.shape,
                          kwargs["do_finalize"],
                          kwargs["all_rank_num_tokens"]))
            return FakeTensor(hidden_states.shape,
                              dtype=hidden_states.dtype,
                              device=hidden_states.device)

    class FakeSharedExperts:

        def __call__(self, hidden_states):
            calls.append(("shared_experts", hidden_states.shape))
            return FakeTensor(hidden_states.shape,
                              dtype=hidden_states.dtype,
                              device=hidden_states.device)

    class FakeHandle:

        def __init__(self, *args):
            pass

        def run_input_embedding(self, input_ids, hidden_states_scratch,
                                input_tokens):
            calls.append(("embedding", input_tokens))
            return hidden_states_scratch

        def run_layer_input_rmsnorm(self, layer_idx, hidden_states_scratch,
                                    norm_hidden_states_scratch, input_tokens,
                                    eps, use_gemma):
            calls.append(("input_rmsnorm", layer_idx))
            return norm_hidden_states_scratch

        def run_layer_input_gated_norm(self, layer_idx,
                                       norm_hidden_states_scratch,
                                       gated_hidden_states_scratch,
                                       input_tokens):
            calls.append(("input_gated_norm", layer_idx))
            return gated_hidden_states_scratch

        def run_layer_attention_output_tail(
                self, layer_idx, attention_core_output_scratch,
                attention_input_scratch, attention_gate_scratch,
                attention_hidden_states_scratch, input_tokens):
            calls.append(("attention_tail", layer_idx))
            return attention_hidden_states_scratch

        def run_layer_post_attention_rmsnorm(
                self, layer_idx, attention_hidden_states_scratch,
                residual_input_scratch, post_attention_norm_scratch,
                post_attention_residual_scratch, input_tokens, eps,
                use_gemma):
            calls.append(("post_attention_rmsnorm", layer_idx))
            return post_attention_norm_scratch

        def run_layer_post_attention_gated_norm(
                self, layer_idx, post_attention_norm_scratch,
                post_attention_gated_scratch, input_tokens):
            calls.append(("post_attention_gated_norm", layer_idx))
            return post_attention_gated_scratch

        def run_layer_moe_router(
                self, layer_idx, post_attention_gated_scratch,
                router_logits_scratch, router_scores_scratch,
                router_topk_indices_scratch, router_topk_weights_scratch,
                input_tokens, top_k, n_group, topk_group,
                routed_scaling_factor):
            calls.append(("moe_router", layer_idx, router_logits_scratch.shape))
            return router_topk_weights_scratch

        def run_layer_post_ffn_rmsnorm(
                self, layer_idx, dense_mlp_output_scratch,
                post_attention_residual_scratch, next_layer_hidden_scratch,
                next_layer_residual_scratch, input_tokens, eps, use_gemma):
            calls.append(("post_ffn_rmsnorm", layer_idx,
                          dense_mlp_output_scratch.copied_from.shape))
            return next_layer_hidden_scratch

        def run_lm_head_logits(self, next_layer_hidden_scratch,
                               logits_scratch, input_tokens):
            calls.append(("lm_head", input_tokens))
            return logits_scratch

    def fake_empty(shape, *, device=None, dtype=None):
        return FakeTensor(tuple(shape), dtype=dtype, device=device)

    def fake_attention_core(**kwargs):
        calls.append(("attention_core", kwargs["layer_idx"]))
        return True

    fake_torch = SimpleNamespace(
        float32="torch.float32",
        int32="torch.int32",
        empty=fake_empty,
        classes=SimpleNamespace(trtllm=SimpleNamespace(
            DeepseekResidentDecodeHandle=FakeHandle)),
        ops=SimpleNamespace(trtllm=SimpleNamespace()),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    model = _fake_native_resident_model_with_moe()
    moe = model.model.layers[1].mlp
    moe.experts = FakeExperts()
    moe.shared_experts = FakeSharedExperts()
    moe.use_dp = False
    moe.mapping = SimpleNamespace(tp_size=1, is_multi_node=lambda: False)
    moe.shared_output_scale = None
    model.model.layers[1].fusion_config = SimpleNamespace(
        POST_MOE_FUSION=False)
    contract = SimpleNamespace(
        hidden_size=7168,
        vocab_size=129280,
        rms_norm_eps=1e-6,
        num_experts_per_tok=8,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        layers=(SimpleNamespace(layer_idx=1, layer_kind="moe", top_k=8), ),
    )
    engine = deepseek_resident_native.create_engine(
        model=model,
        contract=contract,
        dist=None,
    )
    invocation = SimpleNamespace(
        stable_shape_key=("shape", 16),
        real_batch_size=16,
        padded_batch_size=16,
        input_tokens=16,
        request_ids=tuple(range(16)),
        seq_lens=(1, ) * 16,
        cached_tokens=tuple(range(2048, 2064)),
    )
    attn_metadata = SimpleNamespace(all_rank_num_tokens=[16])

    result = engine.run_decode_step_scheduler(
        input_ids=FakeTensor((16, ), dtype="torch.int64"),
        invocation=invocation,
        inputs={"attn_metadata": attn_metadata},
        attention_core_runner=fake_attention_core,
    )

    assert result.completed
    assert result.reason == "resident_stage_scheduler_completed"
    assert result.outputs.shape == (16, 129280)
    assert engine.execution_state()["reason"] == (
        "resident_stage_scheduler_completed")
    assert ("moe_router", 1, (16, 128)) in calls
    assert ("experts", (16, 7168), (16, 128), True, [16]) in calls
    assert ("shared_experts", (16, 7168)) in calls
    assert ("add", (16, 7168), (16, 7168)) in calls
    assert ("post_ffn_rmsnorm", 1, (16, 7168)) in calls
    assert ("lm_head", 16) in calls
    assert engine._scheduler_state.stage_scheduler_completions == 1


def test_deepseek_native_decode_step_scheduler_bridges_deferred_moe_allreduce(
        monkeypatch):
    calls = []

    class FakeTensor:

        def __init__(self,
                     shape,
                     *,
                     dtype="torch.bfloat16",
                     device="cuda:0",
                     name="tensor"):
            self.shape = tuple(shape)
            self.dtype = dtype
            self.device = device
            self.name = name

    class FakeExperts:
        has_nvfp4 = True

        def __call__(self, hidden_states, router_logits, **kwargs):
            calls.append(("experts", hidden_states.shape, router_logits.shape,
                          kwargs["do_finalize"],
                          kwargs["all_rank_num_tokens"]))
            assert not kwargs["do_finalize"]
            return (
                FakeTensor(hidden_states.shape,
                           dtype=hidden_states.dtype,
                           device=hidden_states.device,
                           name="fc2"),
                FakeTensor((hidden_states.shape[0], 8),
                           dtype="torch.float32",
                           device=hidden_states.device,
                           name="expert_scale"),
                FakeTensor((hidden_states.shape[0], 8),
                           dtype="torch.int32",
                           device=hidden_states.device,
                           name="expanded_idx"),
            )

    class FakeSharedExperts:

        def __call__(self, hidden_states):
            calls.append(("shared_experts", hidden_states.shape))
            return FakeTensor(hidden_states.shape,
                              dtype=hidden_states.dtype,
                              device=hidden_states.device,
                              name="shared")

    class FakeMoEAllReduceParams:

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeMoEAllReduce:
        max_token = 32

        def __call__(self, fc2_output, *, all_reduce_params):
            calls.append(("moe_allreduce", fc2_output.name,
                          all_reduce_params.kwargs["shared_expert_output"].name,
                          all_reduce_params.kwargs["residual"].name,
                          all_reduce_params.kwargs["norm_weight"].name,
                          all_reduce_params.kwargs["eps"],
                          all_reduce_params.kwargs["is_cutlass_min_latency"]))
            return (
                FakeTensor(fc2_output.shape,
                           dtype=fc2_output.dtype,
                           device=fc2_output.device,
                           name="fused_next_hidden"),
                FakeTensor(fc2_output.shape,
                           dtype=fc2_output.dtype,
                           device=fc2_output.device,
                           name="fused_next_residual"),
            )

    monkeypatch.setattr(
        deepseek_resident_native,
        "_resolve_moe_all_reduce_params_class",
        lambda: FakeMoEAllReduceParams,
    )

    model = _fake_native_resident_model_with_moe()
    layer = model.model.layers[1]
    moe = layer.mlp
    moe.experts = FakeExperts()
    moe.shared_experts = FakeSharedExperts()
    moe.use_dp = False
    moe.mapping = SimpleNamespace(tp_size=8, is_multi_node=lambda: False)
    moe.shared_output_scale = None
    layer.fusion_config = SimpleNamespace(POST_MOE_FUSION=True)
    layer.model_config = SimpleNamespace(moe_backend="TRTLLM")
    layer.is_p2p_supported = True
    layer.moe_allreduce = FakeMoEAllReduce()
    layer.next_layer_layernorm = SimpleNamespace(
        weight=SimpleNamespace(name="next_norm_weight"),
        variance_epsilon=1e-6,
    )
    contract = SimpleNamespace(
        hidden_size=7168,
        vocab_size=129280,
        rms_norm_eps=1e-6,
        num_experts_per_tok=8,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        layers=(SimpleNamespace(layer_idx=1, layer_kind="moe", top_k=8), ),
    )
    engine = deepseek_resident_native.create_engine(
        model=model,
        contract=contract,
        dist=None,
    )
    invocation = SimpleNamespace(
        stable_shape_key=("shape", 16),
        real_batch_size=16,
        padded_batch_size=16,
        input_tokens=16,
        request_ids=tuple(range(16)),
        seq_lens=(1, ) * 16,
        cached_tokens=tuple(range(2048, 2064)),
    )
    state = deepseek_resident_native.DeepSeekResidentShapeState(
        shape_key=invocation.stable_shape_key,
        batch_size=16,
        hidden_size=7168,
        vocab_size=129280,
        device="cuda:0",
        dtype="torch.bfloat16",
        scratch={
            "hidden_states": FakeTensor((16, 7168), name="hidden"),
            "next_layer_hidden_states":
            FakeTensor((16, 7168), name="scratch_next_hidden"),
            "next_layer_residual_states":
            FakeTensor((16, 7168), name="scratch_next_residual"),
            "logits": FakeTensor((16, 129280), name="logits"),
        },
    )
    engine._shape_states[invocation.stable_shape_key] = state

    def set_input_norm(**kwargs):
        calls.append(("input_rmsnorm", kwargs["layer_idx"]))
        state.scratch["norm_hidden_states"] = FakeTensor((16, 7168),
                                                         name="input_norm")
        return state.scratch["norm_hidden_states"]

    def set_input_gate(**kwargs):
        calls.append(("input_gated_norm", kwargs["layer_idx"]))
        state.scratch["gated_hidden_states"] = FakeTensor((16, 7168),
                                                          name="input_gate")
        return state.scratch["gated_hidden_states"]

    def set_attention_tail(**kwargs):
        calls.append(("attention_tail", kwargs["layer_idx"]))
        state.scratch["attention_hidden_states"] = FakeTensor(
            (16, 7168), name="attention_tail")
        return state.scratch["attention_hidden_states"]

    def set_post_attention_norm(**kwargs):
        calls.append(("post_attention_rmsnorm", kwargs["layer_idx"]))
        state.scratch["post_attention_residual_states"] = FakeTensor(
            (16, 7168), name="post_attention_residual")
        return FakeTensor((16, 7168), name="post_attention_norm")

    def set_post_attention_gate(**kwargs):
        calls.append(("post_attention_gated_norm", kwargs["layer_idx"]))
        state.scratch["post_attention_gated_hidden_states"] = FakeTensor(
            (16, 7168), name="post_attention_gated")
        return state.scratch["post_attention_gated_hidden_states"]

    def set_router(**kwargs):
        calls.append(("moe_router", kwargs["layer_idx"]))
        state.scratch["moe_router_states"] = {
            1: {
                "logits": FakeTensor((16, 128), name="router_logits"),
            },
        }
        return state.scratch["moe_router_states"][1]["logits"]

    def fail_post_ffn(**kwargs):
        calls.append(("post_ffn_rmsnorm", kwargs["layer_idx"]))
        return FakeTensor((16, 7168), name="unexpected_post_ffn")

    def run_lm_head(**kwargs):
        calls.append(("lm_head",
                      state.scratch["next_layer_hidden_states"].name,
                      state.scratch["next_layer_residual_states"].name))
        return state.scratch["logits"]

    engine.run_layer_input_rmsnorm_stage = set_input_norm
    engine.run_layer_input_gated_norm_stage = set_input_gate
    engine.run_layer_attention_core_stage = lambda **kwargs: True
    engine.run_layer_attention_output_tail_stage = set_attention_tail
    engine.run_layer_post_attention_rmsnorm_stage = set_post_attention_norm
    engine.run_layer_post_attention_gated_norm_stage = set_post_attention_gate
    engine.run_layer_moe_router_stage = set_router
    engine.run_layer_post_ffn_rmsnorm_stage = fail_post_ffn
    engine.run_lm_head_logits_stage = run_lm_head

    result = engine.run_decode_step_scheduler(
        input_ids=FakeTensor((16, ), dtype="torch.int64", name="input_ids"),
        invocation=invocation,
        inputs={"attn_metadata": SimpleNamespace(all_rank_num_tokens=[16])},
        skip_input_embedding=True,
    )

    assert result.completed
    assert result.outputs is state.scratch["logits"]
    assert ("experts", (16, 7168), (16, 128), False, [16]) in calls
    assert ("shared_experts", (16, 7168)) in calls
    assert ("moe_allreduce", "fc2", "shared", "post_attention_residual",
            "next_norm_weight", 1e-6, False) in calls
    assert ("post_ffn_rmsnorm", 1) not in calls
    assert ("lm_head", "fused_next_hidden", "fused_next_residual") in calls
    assert state.scratch["last_moe_experts_reason"] == (
        "post_moe_fusion_deferred_allreduce_executed")


def test_deepseek_native_decode_step_scheduler_reports_moe_decline_reason():

    class FakeTensor:

        def __init__(self, shape, *, dtype="torch.bfloat16", device="cuda:0"):
            self.shape = tuple(shape)
            self.dtype = dtype
            self.device = device

    contract = SimpleNamespace(
        hidden_size=7168,
        vocab_size=129280,
        layers=(SimpleNamespace(layer_idx=1, layer_kind="moe", top_k=8), ),
    )
    engine = deepseek_resident_native.create_engine(
        model=_fake_native_resident_model_with_moe(),
        contract=contract,
        dist=None,
    )
    invocation = SimpleNamespace(
        stable_shape_key=("shape", 16),
        real_batch_size=16,
        padded_batch_size=16,
        input_tokens=16,
        request_ids=tuple(range(16)),
        seq_lens=(1, ) * 16,
        cached_tokens=tuple(range(2048, 2064)),
    )
    state = deepseek_resident_native.DeepSeekResidentShapeState(
        shape_key=invocation.stable_shape_key,
        batch_size=16,
        hidden_size=7168,
        vocab_size=129280,
        device="cuda:0",
        dtype="torch.bfloat16",
        scratch={
            "hidden_states": FakeTensor((16, 7168)),
            "moe_router_states": {
                1: {
                    "topk_weights": FakeTensor((16, 8)),
                },
            },
        },
    )
    engine._shape_states[invocation.stable_shape_key] = state
    engine.run_layer_input_rmsnorm_stage = (
        lambda **kwargs: FakeTensor((16, 7168)))
    engine.run_layer_input_gated_norm_stage = (
        lambda **kwargs: FakeTensor((16, 7168)))
    engine.run_layer_attention_core_stage = lambda **kwargs: True
    engine.run_layer_attention_output_tail_stage = (
        lambda **kwargs: FakeTensor((16, 7168)))
    engine.run_layer_post_attention_rmsnorm_stage = (
        lambda **kwargs: FakeTensor((16, 7168)))
    engine.run_layer_post_attention_gated_norm_stage = (
        lambda **kwargs: FakeTensor((16, 7168)))
    engine.run_layer_moe_router_stage = lambda **kwargs: FakeTensor((16, 8))

    def decline_moe(**kwargs):
        state.scratch["last_moe_experts_reason"] = (
            "post_moe_fusion_deferred_finalize_unimplemented")
        return False

    engine.run_layer_moe_experts_stage = decline_moe

    result = engine.run_decode_step_scheduler(
        input_ids=FakeTensor((16, ), dtype="torch.int64"),
        invocation=invocation,
        inputs={},
        skip_input_embedding=True,
    )

    assert not result.completed
    assert result.reason == (
        "layer_1_moe_experts_declined:"
        "post_moe_fusion_deferred_finalize_unimplemented")
    assert engine.execution_state()["stage_scheduler_reason"] == result.reason


def test_deepseek_native_engine_honors_ready_guard(monkeypatch):
    calls = []

    class FakeTensor:

        def __init__(self, shape, *, dtype="torch.bfloat16", device="cuda:0"):
            self.shape = shape
            self.dtype = dtype
            self.device = device

    def fake_empty(shape, *, device=None, dtype=None):
        return FakeTensor(tuple(shape), dtype=dtype, device=device)

    def fake_native_op(*args):
        calls.append(args)
        return {
            "logits": args[2],
        }

    fake_torch = SimpleNamespace(
        empty=fake_empty,
        ops=SimpleNamespace(trtllm=SimpleNamespace(
            deepseek_resident_decode_ready=lambda: False,
            deepseek_resident_decode=fake_native_op,
        )),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    contract = SimpleNamespace(
        hidden_size=7168,
        vocab_size=129280,
        layers=(SimpleNamespace(layer_idx=0, layer_kind="dense"), ),
    )
    engine = deepseek_resident_native.create_engine(
        model=_fake_native_resident_model(),
        contract=contract,
        dist=None,
    )
    invocation = SimpleNamespace(
        stable_shape_key=("shape", 16),
        real_batch_size=16,
        padded_batch_size=16,
        input_tokens=16,
        request_ids=tuple(range(16)),
        seq_lens=(1, ) * 16,
        cached_tokens=tuple(range(2048, 2064)),
    )

    result = engine.execute(
        request=None,
        contract=contract,
        invocation=invocation,
        inputs={"input_ids": FakeTensor((16, ), dtype="torch.int64")},
    )

    assert result is None
    assert calls == []


def test_deepseek_native_asset_table_includes_layer_sites():
    contract = SimpleNamespace(
        layers=(SimpleNamespace(layer_idx=0, layer_kind="dense"), ),
    )

    assets = deepseek_resident_native.build_deepseek_resident_model_assets(
        _fake_native_resident_model(), contract)

    assert "model.embed_tokens.weight" in assets.tensor_names
    assert "model.norm.weight" in assets.tensor_names
    assert "lm_head.weight" in assets.tensor_names
    assert "model.layers.0.input_layernorm.weight" in assets.tensor_names
    assert (
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"
        in assets.tensor_names)
    assert "model.layers.0.self_attn.k_b_proj_trans" in assets.tensor_names
    assert "model.layers.0.self_attn.v_b_proj" in assets.tensor_names
    assert "model.layers.0.self_attn.gate_proj.weight" in assets.tensor_names
    assert "model.layers.0.self_attn.o_proj.weight" in assets.tensor_names
    assert "model.layers.0.mlp.gate_up_proj.weight" in assets.tensor_names
    assert assets.layers[0].layer_idx == 0
    assert assets.layers[0].layer_kind == "dense"
    assert assets.layers[0].start == 3
    assert assets.layers[0].stop == len(assets.tensors)
    assert assets.layer_offsets == (3, len(assets.tensors))
    assert assets.layer_kinds == (0, )
    assert assets.layer_site_offsets == (0, 19)
    assert assets.layer_site_ids == (0, 1, 2, 3, 4, 5, 6, 23, 26, 7, 19, 15,
                                    83, 88, 93, 98, 100, 40, 44)
    assert assets.layer_site_tensor_indices == (3, 4, 5, 6, 7, 8, 9, 10,
                                                11, 12, 13, 14, 15, 16, 17,
                                                18, 19, 20, 21)


def test_deepseek_native_asset_table_pins_plain_indexer_rope_tensor():

    class FakeTorchModule:

        def __init__(self, **modules):
            self._modules = modules

        def __getattr__(self, name):
            try:
                return self._modules[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def named_parameters(self, recurse=True):
            return ()

        def named_buffers(self, recurse=True):
            return ()

    rope = _fake_native_tensor((4096, 64), dtype="torch.float32")
    model = _fake_native_resident_model()
    model.model.layers[0].self_attn.mqa.indexer = FakeTorchModule(
        rotary_emb=SimpleNamespace(rotary_cos_sin=rope))
    contract = SimpleNamespace(
        layers=(SimpleNamespace(layer_idx=0, layer_kind="dense"), ),
    )

    assets = deepseek_resident_native.build_deepseek_resident_model_assets(
        model, contract)

    tensor_name = (
        "model.layers.0.self_attn.mqa.indexer.rotary_emb.rotary_cos_sin")
    assert tensor_name in assets.tensor_names
    site = next(
        site for site in assets.layer_tensor_sites if site.site_id ==
        deepseek_resident_native._SITE_ATTN_INDEXER_ROTARY_COS_SIN)
    assert assets.tensor_specs[site.tensor_idx].name == tensor_name
    assert assets.tensors[site.tensor_idx] is rope


def test_deepseek_native_asset_table_includes_moe_router_sites():
    contract = SimpleNamespace(
        layers=(
            SimpleNamespace(layer_idx=0, layer_kind="dense"),
            SimpleNamespace(layer_idx=1, layer_kind="moe"),
        ),
    )

    assets = deepseek_resident_native.build_deepseek_resident_model_assets(
        _fake_native_resident_model_with_moe(), contract)

    assert "model.layers.1.mlp.gate.weight" in assets.tensor_names
    assert (
        "model.layers.1.mlp.gate.e_score_correction_bias"
        in assets.tensor_names)
    assert assets.layers[1].layer_idx == 1
    assert assets.layers[1].layer_kind == "moe"
    assert assets.layer_offsets == (3, 22, len(assets.tensors))
    assert assets.layer_kinds == (0, 1)
    assert assets.layer_site_offsets == (0, 19, 38)
    assert assets.layer_site_ids[19:38] == (0, 1, 2, 3, 4, 5, 6, 23, 26, 7,
                                            19, 15, 83, 88, 93, 98, 100, 30,
                                            31)


def test_deepseek_native_asset_table_aliases_warpdecode_moe_experts():
    model = _fake_native_resident_model_with_moe()
    model.model.layers[1].mlp.shared_experts = SimpleNamespace(
        gate_up_proj=_fake_native_weight_module((4096, 7168)),
        down_proj=_fake_native_weight_module((7168, 2048)),
    )
    model.model.layers[1].mlp.experts = SimpleNamespace(backend=SimpleNamespace(
        w3_w1_weight=_fake_native_tensor((32, 4096, 3584)),
        w3_w1_weight_scaling_factor=_fake_native_tensor(
            (32, 4096, 448), dtype="torch.float8_e4m3fn"),
        fc31_input_scale=_fake_native_tensor((1, ), dtype="torch.float32"),
        fc31_scale_c=_fake_native_tensor((32, ), dtype="torch.float32"),
        fc31_alpha=_fake_native_tensor((32, ), dtype="torch.float32"),
        w2_weight=_fake_native_tensor((32, 7168, 2048)),
        w2_weight_scaling_factor=_fake_native_tensor(
            (32, 7168, 256), dtype="torch.float8_e4m3fn"),
        fc2_input_scale=_fake_native_tensor((1, ), dtype="torch.float32"),
        fc2_alpha=_fake_native_tensor((32, ), dtype="torch.float32"),
    ))
    contract = SimpleNamespace(
        layers=(
            SimpleNamespace(layer_idx=0, layer_kind="dense"),
            SimpleNamespace(layer_idx=1, layer_kind="moe"),
        ),
    )

    assets = deepseek_resident_native.build_deepseek_resident_model_assets(
        model, contract)
    sites = {
        site.site_id: assets.tensor_specs[site.tensor_idx].name
        for site in assets.layer_tensor_sites
        if site.layer_idx == 1
    }

    assert sites[deepseek_resident_native._SITE_EXPERT_GATE_UP_WEIGHT] == (
        "model.layers.1.mlp.experts.backend.w3_w1_weight")
    assert sites[deepseek_resident_native._SITE_EXPERT_GATE_UP_WEIGHT_SCALE] == (
        "model.layers.1.mlp.experts.backend.w3_w1_weight_scaling_factor")
    assert sites[deepseek_resident_native._SITE_EXPERT_GATE_UP_INPUT_SCALE] == (
        "model.layers.1.mlp.experts.backend.fc31_input_scale")
    assert sites[deepseek_resident_native._SITE_EXPERT_GATE_UP_OUTPUT_SCALE] == (
        "model.layers.1.mlp.experts.backend.fc31_scale_c")
    assert sites[deepseek_resident_native._SITE_EXPERT_GATE_UP_ALPHA] == (
        "model.layers.1.mlp.experts.backend.fc31_alpha")
    assert sites[deepseek_resident_native._SITE_EXPERT_DOWN_WEIGHT] == (
        "model.layers.1.mlp.experts.backend.w2_weight")
    assert sites[deepseek_resident_native._SITE_EXPERT_DOWN_WEIGHT_SCALE] == (
        "model.layers.1.mlp.experts.backend.w2_weight_scaling_factor")
    assert sites[deepseek_resident_native._SITE_EXPERT_DOWN_INPUT_SCALE] == (
        "model.layers.1.mlp.experts.backend.fc2_input_scale")
    assert sites[deepseek_resident_native._SITE_EXPERT_DOWN_ALPHA] == (
        "model.layers.1.mlp.experts.backend.fc2_alpha")


def test_deepseek_resident_contract_rejects_missing_attention_gate():
    contract, rejection = build_deepseek_resident_body_contract(
        _fake_deepseek_model(missing_attention_gate=True))

    assert contract is None
    assert rejection == "layer_0_missing_attention_output_gate"

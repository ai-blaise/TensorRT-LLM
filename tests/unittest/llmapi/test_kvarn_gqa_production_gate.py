"""Production-gate coverage for fail-closed GQA KVarN rollout."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


_GATE_PATH = (Path(__file__).resolve().parents[3] / "benchmarks" / "python" /
              "bench_kvarn_gqa_production_gate.py")
_SPEC = importlib.util.spec_from_file_location("bench_kvarn_gqa_production_gate", _GATE_PATH)
_gate = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _gate
_SPEC.loader.exec_module(_gate)


class _Ready:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


def test_gate_matrix_covers_post_gen56_composability_defaults():
    payload = _gate.build_payload(
        seq_lens=_gate.DEFAULT_SEQ_LENS,
        odd_m=_gate.DEFAULT_ODD_M,
        transports=_gate.DEFAULT_TRANSPORTS,
        concurrency=16,
        min_tok_s_per_user=150.0,
    )

    assert payload["dtype"] == "kvarn_k2v2_g128"
    assert payload["dense_mla_dtype"] == "kvarn_k2v2"
    assert payload["dense_mla_amortize"] is True
    assert payload["indexer_quantized_by_kvarn"] is False
    assert payload["target_concurrency"] == 16
    assert payload["min_tok_s_per_user"] == 150.0
    assert payload["runtime_dtypes"] == ["fp16", "bf16"]
    assert payload["kv_layouts"] == ["compact", "paged"]
    assert payload["bf16_reference_required"] is True
    assert payload["paged_kv_required"] is True
    assert payload["abort_reuse_required"] is True
    assert {case["tail_tokens"] for case in payload["partial_block_cases"]} == {0, 1, 7, 127}
    assert {case["sink_tokens"] for case in payload["partial_block_cases"]} == {0, 16, 128}

    cases = payload["cases"]
    assert len(cases) == len(_gate.DEFAULT_SEQ_LENS) * len(_gate.DEFAULT_ODD_M) * len(_gate.DEFAULT_TRANSPORTS)
    assert {case["seq_len"] for case in cases} == set(_gate.DEFAULT_SEQ_LENS)
    assert {case["odd_m"] for case in cases} == {1, 5, 25}
    assert {case["transport"] for case in cases} == {"ucx", "nixl", "mooncake", "mori"}
    assert all(case["layersplit"] for case in cases)
    assert all(case["request_pinning"] for case in cases)
    assert all(case["moondream_overlap"] for case in cases)
    assert all(case["smc"] for case in cases)
    assert all(case["warpdecode"] for case in cases)


def test_gate_requires_fused_ops_and_backend_readiness(monkeypatch):
    fake_torch = SimpleNamespace(ops=SimpleNamespace(trtllm=SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(SystemExit, match="kvarn_gqa_store"):
        _gate._require_fused_ready()

    fake_torch.ops.trtllm.kvarn_gqa_store = object()
    fake_torch.ops.trtllm.kvarn_gqa_decode = object()
    fake_torch.ops.trtllm.kvarn_gqa_backend_ready = _Ready(False)
    with pytest.raises(SystemExit, match="not production-ready"):
        _gate._require_fused_ready()

    fake_torch.ops.trtllm.kvarn_gqa_backend_ready = _Ready(True)
    _gate._require_fused_ready()


def test_gate_fails_closed_when_backend_probe_raises(monkeypatch):
    def raises():
        raise RuntimeError("probe failed")

    fake_torch = SimpleNamespace(ops=SimpleNamespace(trtllm=SimpleNamespace(
        kvarn_gqa_store=object(),
        kvarn_gqa_decode=object(),
        kvarn_gqa_backend_ready=raises,
    )))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(SystemExit, match="probe failed"):
        _gate._require_fused_ready()

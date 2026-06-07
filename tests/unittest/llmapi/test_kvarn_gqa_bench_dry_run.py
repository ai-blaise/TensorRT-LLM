"""Source-only coverage for the GQA KVarN sparse benchmark dry-run."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


_BENCH_PATH = (Path(__file__).resolve().parents[3] / "blaise_perf" / "kvarn_gqa" /
               "bench_kvarn_gqa_sparse.py")
_SPEC = importlib.util.spec_from_file_location("bench_kvarn_gqa_sparse", _BENCH_PATH)
_bench = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _bench
_SPEC.loader.exec_module(_bench)


def test_sparse_bench_dry_run_prints_store_decode_graph_matrix(capsys):
    args = SimpleNamespace(
        repo="/workspace",
        device=0,
        dtype="fp16",
        heads=8,
        kv_heads=2,
        m=[1, 5, 25],
        sparse_topk=64,
        blocks=1,
        graph_replay=True,
    )

    _bench.dry_run(args)

    out = capsys.readouterr().out
    assert "KVARN_GQA_BENCH_DRY_RUN" in out
    assert "tokens=128" in out
    assert "sparse_full_check=1" in out
    assert "graph_replay=1" in out
    assert "PLAN STORE+DECODE dtype=fp16 M=1" in out
    assert "PLAN STORE+DECODE dtype=fp16 M=5" in out
    assert "PLAN STORE+DECODE dtype=fp16 M=25" in out
    assert "No CUDA context was created" in out

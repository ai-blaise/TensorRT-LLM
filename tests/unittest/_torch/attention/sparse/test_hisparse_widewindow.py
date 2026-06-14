"""Wide-window swap-in (lever c) control-flow unit tests (CPU, no model).

These cover the Python orchestration of prepare_hot_pool_overlapped /
consume_prepare_join_event / the on_copy_stream threading -- the parts that gate
eligibility and manage the deferred-join bookkeeping -- WITHOUT needing a CUDA
forward. The byte-exact + graph-capture proof of the actual overlapped device chain
is blaise_perf/hisparse/swapin_widewindow_probe.py (GPU). Here we assert:
  - prepare falls back (returns None) when the hoist is not eligible (no CUDA tensor
    / disabled / wrong dtype / older build without overlap args),
  - consume_prepare_join_event is a no-op (returns False) when no prepare ran,
  - the eligibility predicate honors the overlap_swap_in switch and the op-arg gate,
which together guarantee the fail-closed, byte-identical fallback to the in-method
G1 path whenever the wide hoist cannot run.
"""
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.attention_backend.sparse.hisparse import (
    OPTRTHiSparseCoordinator)


def _cfg(enabled=True, overlap=True):
    return SimpleNamespace(
        hisparse_enabled=enabled,
        hisparse_mode="dense_mla_kvarn",
        hisparse_hot_blocks_per_req=2,
        hisparse_host_to_device_ratio=8,
        hisparse_overlap_swap_in=overlap,
    )


def test_prepare_returns_none_when_disabled():
    coord = OPTRTHiSparseCoordinator(_cfg(enabled=False))
    out = coord.prepare_hot_pool_overlapped(
        topk_indices=object(), metadata=SimpleNamespace(), layer_idx=0,
        skip_topk=False, is_generation=True)
    assert out is None


def test_prepare_returns_none_when_overlap_switch_off():
    coord = OPTRTHiSparseCoordinator(_cfg(enabled=True, overlap=False))
    assert coord.overlap_swap_in is False
    # eligibility must be False purely from the switch (no CUDA needed)
    assert coord._prepare_overlap_eligible(object()) is False  # noqa: SLF001
    out = coord.prepare_hot_pool_overlapped(
        topk_indices=object(), metadata=SimpleNamespace(), layer_idx=0,
        skip_topk=False, is_generation=True)
    assert out is None


def test_prepare_returns_none_for_non_cuda_topk():
    # Default overlap on, but a non-CUDA TopK (e.g. a plain object / CPU tensor) is
    # ineligible -> fall back. This is the path a CPU-only unit run always takes.
    coord = OPTRTHiSparseCoordinator(_cfg(enabled=True, overlap=True))
    assert coord._prepare_overlap_eligible(object()) is False  # noqa: SLF001
    out = coord.prepare_hot_pool_overlapped(
        topk_indices=object(), metadata=SimpleNamespace(), layer_idx=0,
        skip_topk=False, is_generation=True)
    assert out is None


def test_eligibility_requires_overlap_op_args(monkeypatch):
    import torch
    coord = OPTRTHiSparseCoordinator(_cfg(enabled=True, overlap=True))

    class _FakeCudaTopk:
        is_cuda = True
        dtype = torch.int32

        def dim(self):
            return 2

    # Pretend CUDA is available so we reach the op-arg gate.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    # Force the op-arg detection to report "older build" (no overlap args).
    coord._swap_in_overlap_supported = False  # noqa: SLF001
    assert coord._prepare_overlap_eligible(_FakeCudaTopk()) is False  # noqa: SLF001
    # And to report supported -> eligible (we don't run the chain here).
    coord._swap_in_overlap_supported = True  # noqa: SLF001
    assert coord._prepare_overlap_eligible(_FakeCudaTopk()) is True  # noqa: SLF001


def test_consume_prepare_join_event_is_noop_without_pending():
    coord = OPTRTHiSparseCoordinator(_cfg(enabled=True))
    # No prepare ran -> no pending join for this (step, layer) -> False, no error.
    assert coord.consume_prepare_join_event(0) is False
    assert coord.consume_prepare_join_event(5) is False


def test_pending_join_map_is_keyed_by_step_and_layer():
    # The bookkeeping dict is keyed (step_id, layer_idx); a manually-seeded entry is
    # consumed exactly once and only for its key (mirrors the real prepare/consume).
    coord = OPTRTHiSparseCoordinator(_cfg(enabled=True))
    sentinel = SimpleNamespace(waited=False)

    class _FakeEvent:
        pass

    class _FakeStream:

        def wait_event(self, ev):
            sentinel.waited = True

    import torch
    monkeypatch_stream = _FakeStream()
    # Seed a pending join for (step_id=0, layer=3).
    coord._pending_prepare_join[(int(coord.step_id), 3)] = _FakeEvent()  # noqa: SLF001
    orig = torch.cuda.current_stream
    torch.cuda.current_stream = lambda *a, **k: monkeypatch_stream
    try:
        # wrong layer -> no-op
        assert coord.consume_prepare_join_event(2) is False
        assert sentinel.waited is False
        # right layer -> consumes, waits, and clears
        assert coord.consume_prepare_join_event(3) is True
        assert sentinel.waited is True
        assert coord.consume_prepare_join_event(3) is False  # cleared
    finally:
        torch.cuda.current_stream = orig


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

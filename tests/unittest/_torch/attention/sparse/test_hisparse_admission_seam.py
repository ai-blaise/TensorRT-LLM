"""P4 freed-HBM admission-seam unit tests (pure accounting, no CUDA, no model).

Covers the HiSparse host-backed readmission gate that a scheduler consults after a
finished request is released:
  - OPTRTHiSparseCoordinator.can_admit_request / admittable_token_capacity
    accounting through reserve -> release,
  - OPTRTHiSparseCoordinator.filter_admissible_requests (batch budget consistency,
    oversized rejection, free-slot exhaustion, freed-HBM re-open after release,
    unconfigured no-op),
  - the scheduler-side glue: hisparse_num_prompt_blocks (prompt -> host-block
    ceil) and SimpleScheduler._apply_hisparse_gate (strict no-op when no coordinator
    is attached).

These exercise the seam end-to-end at the Python boundary. The LIVE wiring -- the
multi-rank DSA/NIXL forward that actually reserves/admits/releases through the
coordinator (Gate 5-7) -- is not available in this harness; this bounds the seam to
what is unit-testable in isolation.
"""
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.attention_backend.sparse.hisparse import (
    OPTRTHiSparseCoordinator)


def _cfg(enabled=True):
    return SimpleNamespace(
        hisparse_enabled=enabled,
        hisparse_mode="dense_mla_kvarn",
        hisparse_hot_blocks_per_req=2,
        hisparse_host_to_device_ratio=8,
    )


def _coordinator(*, host_blocks=8, hot_blocks=2, tokens_per_block=64,
                 request_slots=None, max_blocks=None, num_layers=1):
    coord = OPTRTHiSparseCoordinator(_cfg(enabled=True))
    coord.configure_packed_tiers(
        num_layers=num_layers,
        tokens_per_block=tokens_per_block,
        packed_bytes_per_block=2048,
        logical_host_capacity_blocks=host_blocks,
        hot_device_capacity_blocks=hot_blocks,
        request_slot_capacity=request_slots,
        max_blocks_per_request=max_blocks,
    )
    return coord


# ---------------------------------------------------------------------------
# can_admit_request / admittable_token_capacity accounting
# ---------------------------------------------------------------------------
def test_admittable_token_capacity_is_host_backed():
    coord = _coordinator(host_blocks=8, tokens_per_block=64)
    # host-backed sequence budget = logical_host_capacity_blocks * tokens_per_block
    assert coord.admittable_token_capacity() == 8 * 64
    # device-resident budget is the small hot tier (2 blocks)
    assert coord.device_resident_token_capacity() == 2 * 64


def test_admittable_token_capacity_zero_when_unconfigured():
    coord = OPTRTHiSparseCoordinator(_cfg(enabled=True))
    assert coord.admittable_token_capacity() == 0
    assert coord.can_admit_request(1) is False


def test_can_admit_request_tracks_reserve_and_release():
    coord = _coordinator(host_blocks=8, request_slots=4, max_blocks=8)
    # fresh: 8 free host blocks, 4 free request slots
    assert coord.can_admit_request(3) is True
    assert coord.can_admit_request(8) is True
    assert coord.can_admit_request(9) is False  # exceeds max_blocks_per_request

    coord.reserve_request(req_pool_idx=1, num_prompt_blocks=5)
    # 3 host blocks + 3 request slots left
    assert coord.can_admit_request(3) is True
    assert coord.can_admit_request(4) is False  # only 3 host blocks free

    coord.reserve_request(req_pool_idx=2, num_prompt_blocks=3)
    # 0 host blocks free now
    assert coord.can_admit_request(1) is False

    # release frees the 5 host blocks + the request slot -> admission re-opens
    coord.release_request(1)
    assert coord.can_admit_request(5) is True
    assert coord.can_admit_request(6) is False  # 5 freed + 0 remaining


def test_can_admit_request_needs_a_free_request_slot():
    coord = _coordinator(host_blocks=16, request_slots=1, max_blocks=16)
    assert coord.can_admit_request(2) is True
    coord.reserve_request(req_pool_idx=10, num_prompt_blocks=2)
    # plenty of host blocks remain, but the single request slot is taken
    assert coord.can_admit_request(1) is False
    coord.release_request(10)
    assert coord.can_admit_request(1) is True


# ---------------------------------------------------------------------------
# filter_admissible_requests
# ---------------------------------------------------------------------------
def _req(name, blocks):
    return SimpleNamespace(name=name, blocks=blocks)


def _blocks_of(r):
    return r.blocks


def test_filter_admits_in_order_until_host_budget_exhausted():
    coord = _coordinator(host_blocks=8, request_slots=8, max_blocks=8)
    reqs = [_req("a", 3), _req("b", 3), _req("c", 3)]  # 3+3 fit (6<=8), 3rd defers
    admitted, deferred = coord.filter_admissible_requests(
        reqs, num_prompt_blocks_of=_blocks_of)
    assert [r.name for r in admitted] == ["a", "b"]
    assert [r.name for r in deferred] == ["c"]


def test_filter_batch_budget_is_summed_not_per_request():
    # Each request fits in isolation (<= 8) but the SUM must fit the free budget.
    coord = _coordinator(host_blocks=10, request_slots=8, max_blocks=8)
    reqs = [_req("a", 4), _req("b", 4), _req("c", 4)]  # 4+4=8<=10, +4=12>10
    admitted, deferred = coord.filter_admissible_requests(
        reqs, num_prompt_blocks_of=_blocks_of)
    assert [r.name for r in admitted] == ["a", "b"]
    assert [r.name for r in deferred] == ["c"]


def test_filter_request_slot_limit_caps_admissions():
    coord = _coordinator(host_blocks=64, request_slots=2, max_blocks=64)
    reqs = [_req("a", 1), _req("b", 1), _req("c", 1)]  # host fine; only 2 req slots
    admitted, deferred = coord.filter_admissible_requests(
        reqs, num_prompt_blocks_of=_blocks_of)
    assert [r.name for r in admitted] == ["a", "b"]
    assert [r.name for r in deferred] == ["c"]


def test_filter_defers_oversized_request_but_keeps_scanning():
    coord = _coordinator(host_blocks=8, request_slots=8, max_blocks=4)
    # "big" exceeds max_blocks_per_request (deferred), smaller ones still admitted.
    reqs = [_req("big", 5), _req("a", 2), _req("b", 2)]
    admitted, deferred = coord.filter_admissible_requests(
        reqs, num_prompt_blocks_of=_blocks_of)
    assert [r.name for r in admitted] == ["a", "b"]
    assert [r.name for r in deferred] == ["big"]


def test_filter_freed_hbm_reopens_admission_after_release():
    coord = _coordinator(host_blocks=6, request_slots=4, max_blocks=6)
    # Fill the host tier with one resident request.
    coord.reserve_request(req_pool_idx=99, num_prompt_blocks=6)
    reqs = [_req("a", 3), _req("b", 3)]
    admitted, deferred = coord.filter_admissible_requests(
        reqs, num_prompt_blocks_of=_blocks_of)
    assert admitted == []  # nothing fits, host tier full
    assert [r.name for r in deferred] == ["a", "b"]

    # Release the resident request -> 6 host blocks freed -> both now fit.
    coord.release_request(99)
    admitted, deferred = coord.filter_admissible_requests(
        reqs, num_prompt_blocks_of=_blocks_of)
    assert [r.name for r in admitted] == ["a", "b"]
    assert deferred == []


def test_filter_is_side_effect_free():
    coord = _coordinator(host_blocks=8, request_slots=8, max_blocks=8)
    before_host = len(coord._free_host_slots)  # noqa: SLF001
    before_req = len(coord._free_request_slots)  # noqa: SLF001
    coord.filter_admissible_requests(
        [_req("a", 3), _req("b", 3)], num_prompt_blocks_of=_blocks_of)
    # the gate decides admissibility but does NOT reserve -> free pools unchanged
    assert len(coord._free_host_slots) == before_host  # noqa: SLF001
    assert len(coord._free_request_slots) == before_req  # noqa: SLF001


def test_filter_noop_when_unconfigured():
    coord = OPTRTHiSparseCoordinator(_cfg(enabled=True))  # no tiers configured
    reqs = [_req("a", 3), _req("b", 100)]
    admitted, deferred = coord.filter_admissible_requests(
        reqs, num_prompt_blocks_of=_blocks_of)
    assert admitted == reqs
    assert deferred == []


# ---------------------------------------------------------------------------
# scheduler-side glue: hisparse_num_prompt_blocks + SimpleScheduler no-op
# ---------------------------------------------------------------------------
def test_num_prompt_blocks_ceil_division():
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import (
        hisparse_num_prompt_blocks)
    assert hisparse_num_prompt_blocks(SimpleNamespace(prompt_len=0), 64) == 0
    assert hisparse_num_prompt_blocks(SimpleNamespace(prompt_len=1), 64) == 1
    assert hisparse_num_prompt_blocks(SimpleNamespace(prompt_len=64), 64) == 1
    assert hisparse_num_prompt_blocks(SimpleNamespace(prompt_len=65), 64) == 2
    assert hisparse_num_prompt_blocks(SimpleNamespace(prompt_len=128), 64) == 2
    # unknown length -> 0 (never spuriously rejects)
    assert hisparse_num_prompt_blocks(SimpleNamespace(), 64) == 0
    # falls back to token list length
    assert hisparse_num_prompt_blocks(
        SimpleNamespace(prompt_tokens=list(range(130))), 64) == 3
    # guard tokens_per_block <= 0
    assert hisparse_num_prompt_blocks(SimpleNamespace(prompt_len=128), 0) == 0


def test_simple_scheduler_gate_is_noop_without_coordinator():
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import SimpleScheduler
    # Build a SimpleScheduler with no coordinator; the gate must be a pure pass-through.
    sched = SimpleScheduler.__new__(SimpleScheduler)
    sched.hisparse_coordinator = None
    reqs = [_req("a", 3), _req("b", 3)]
    assert sched._apply_hisparse_gate(reqs) is reqs  # noqa: SLF001


def test_simple_scheduler_gate_filters_with_coordinator():
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import SimpleScheduler
    coord = _coordinator(host_blocks=6, request_slots=4, max_blocks=6,
                         tokens_per_block=64)
    coord.reserve_request(req_pool_idx=1, num_prompt_blocks=6)  # host tier full
    sched = SimpleScheduler.__new__(SimpleScheduler)
    sched.hisparse_coordinator = coord
    # requests sized via prompt_len -> blocks; both need >0 blocks, none fit while full
    reqs = [SimpleNamespace(name="a", prompt_len=130),
            SimpleNamespace(name="b", prompt_len=64)]
    gated = sched._apply_hisparse_gate(reqs)  # noqa: SLF001
    assert gated == []
    coord.release_request(1)  # free the host tier
    gated = sched._apply_hisparse_gate(reqs)  # noqa: SLF001
    assert [r.name for r in gated] == ["a", "b"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

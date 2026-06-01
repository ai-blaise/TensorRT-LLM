"""Unit tests for tensorrt_llm._torch.attention_backend.sparse.layersplit."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm._torch.attention_backend.sparse.layersplit import (
    LayerSplitOwnership, LayerSplitRuntimeState, build_layersplit_layer_mask,
    compute_owner_assignment)


def test_cp_size_1_collapses_all_layers_to_rank_0():
    own = compute_owner_assignment(num_layers=61, cp_size=1)
    assert own.owner_map == tuple([0] * 61)
    assert own.cp_size == 1
    assert own.num_layers == 61
    assert own.layers_per_rank() == (61, )
    assert own.owned_layers(0) == tuple(range(61))


def test_round_robin_matches_sglang_op_ls_reference():
    own = compute_owner_assignment(num_layers=4, cp_size=2, policy="round_robin")
    assert own.owner_map == (0, 1, 0, 1)
    assert own.is_owner(0, 0) and not own.is_owner(0, 1)
    assert own.is_owner(1, 1) and not own.is_owner(1, 0)
    assert own.owned_layers(0) == (0, 2)
    assert own.owned_layers(1) == (1, 3)
    assert own.layers_per_rank() == (2, 2)


def test_contiguous_distributes_remainder_to_low_ranks():
    own = compute_owner_assignment(num_layers=7, cp_size=3, policy="contiguous")
    # per=2, extra=1 -> rank 0 owns 3, rank 1 owns 2, rank 2 owns 2
    assert own.owner_map == (0, 0, 0, 1, 1, 2, 2)
    assert own.layers_per_rank() == (3, 2, 2)
    assert own.owned_layers(0) == (0, 1, 2)
    assert own.owned_layers(1) == (3, 4)
    assert own.owned_layers(2) == (5, 6)


def test_round_robin_deepseek_v32_61_layers_cp_size_8():
    own = compute_owner_assignment(num_layers=61, cp_size=8, policy="round_robin")
    # 61 = 8*7 + 5, so ranks 0..4 own 8 layers and 5..7 own 7 layers under round_robin
    assert own.layers_per_rank() == (8, 8, 8, 8, 8, 7, 7, 7)
    for layer in range(61):
        assert own.owner_of(layer) == layer % 8


def test_contiguous_deepseek_v32_61_layers_cp_size_8():
    own = compute_owner_assignment(num_layers=61, cp_size=8, policy="contiguous")
    # per=7, extra=5 -> ranks 0..4 own 8 contiguous layers, ranks 5..7 own 7
    assert own.layers_per_rank() == (8, 8, 8, 8, 8, 7, 7, 7)
    assert own.owned_layers(0) == tuple(range(0, 8))
    assert own.owned_layers(1) == tuple(range(8, 16))
    assert own.owned_layers(4) == tuple(range(32, 40))
    assert own.owned_layers(5) == tuple(range(40, 47))
    assert own.owned_layers(7) == tuple(range(54, 61))


def test_cp_size_greater_than_num_layers_leaves_idle_owners():
    own = compute_owner_assignment(num_layers=2, cp_size=4, policy="round_robin")
    assert own.owner_map == (0, 1)
    assert own.layers_per_rank() == (1, 1, 0, 0)
    assert own.owned_layers(2) == ()
    assert own.owned_layers(3) == ()


def test_contiguous_cp_size_greater_than_num_layers():
    own = compute_owner_assignment(num_layers=2, cp_size=4, policy="contiguous")
    # per=0, extra=2 -> ranks 0,1 own 1 each, ranks 2,3 own 0
    assert own.owner_map == (0, 1)
    assert own.layers_per_rank() == (1, 1, 0, 0)


def test_ownership_is_hashable_so_it_can_sit_on_metadata():
    own_a = compute_owner_assignment(8, 2, "round_robin")
    own_b = compute_owner_assignment(8, 2, "round_robin")
    own_c = compute_owner_assignment(8, 2, "contiguous")
    assert hash(own_a) == hash(own_b)
    assert hash(own_a) != hash(own_c)
    assert {own_a, own_b, own_c} == {own_a, own_c}


def test_explicit_construction_matches_compute_result():
    expected = LayerSplitOwnership(
        owner_map=(0, 1, 0, 1, 0, 1),
        cp_size=2,
        policy="round_robin",
    )
    actual = compute_owner_assignment(6, 2, "round_robin")
    assert actual == expected


def test_invalid_num_layers_raises():
    with pytest.raises(ValueError, match="num_layers must be >= 1"):
        compute_owner_assignment(0, 2)
    with pytest.raises(ValueError, match="num_layers must be >= 1"):
        compute_owner_assignment(-3, 2)


def test_invalid_cp_size_raises():
    with pytest.raises(ValueError, match="cp_size must be >= 1"):
        compute_owner_assignment(8, 0)
    with pytest.raises(ValueError, match="cp_size must be >= 1"):
        compute_owner_assignment(8, -1)


def test_unknown_policy_raises():
    with pytest.raises(ValueError, match="unknown LayerSplit"):
        compute_owner_assignment(8, 2, policy="block_cyclic")


def test_default_policy_is_round_robin():
    own = compute_owner_assignment(8, 4)
    assert own.policy == "round_robin"
    assert own.owner_map == (0, 1, 2, 3, 0, 1, 2, 3)


# ---------------------------------------------------------------------------
# LayerSplitRuntimeState (M3) tests
# ---------------------------------------------------------------------------


def _sparse_cfg(**overrides):
    cfg = {
        "layersplit_enabled": True,
        "layersplit_owner_assignment": "round_robin",
        "layersplit_transfer_backend": "auto",
        "layersplit_all_cp_ranks_transfer": True,
    }
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def test_runtime_state_disabled_path_is_inert():
    state = LayerSplitRuntimeState.disabled()
    assert state.enabled is False
    assert state.ownership is None
    assert state.comm_stream is None
    # The off path acts as "everyone owns every layer" so callers can use the
    # ownership query unconditionally without branching on `enabled`.
    assert state.is_owner(0) is True
    assert state.is_owner(60) is True
    assert state.owner_of(0) == 0
    # The no-op transfer must be safe to call on the off path.
    state.layersplit_noop_transfer(layer_idx=0)


def test_runtime_state_built_from_disabled_sparse_config():
    state = LayerSplitRuntimeState.from_sparse_config(
        sparse_attn_config=_sparse_cfg(layersplit_enabled=False),
        num_layers=61,
        cp_size=8,
        cp_rank=3,
    )
    assert state.enabled is False
    assert state.ownership is None
    assert state.comm_stream is None


def test_runtime_state_built_round_robin_cp_size_8():
    state = LayerSplitRuntimeState.from_sparse_config(
        sparse_attn_config=_sparse_cfg(),
        num_layers=61,
        cp_size=8,
        cp_rank=3,
        create_comm_stream=False,
    )
    assert state.enabled is True
    assert state.ownership is not None
    assert state.ownership.policy == "round_robin"
    assert state.cp_size == 8
    assert state.cp_rank == 3
    assert state.transfer_backend == "auto"
    assert state.all_cp_ranks_transfer is True
    # owner_of(layer) on the off path returned 0; on the on path it returns
    # the actual owner per the policy table.
    assert state.owner_of(0) == 0
    assert state.owner_of(3) == 3
    assert state.owner_of(8) == 0
    # is_owner reflects the cp_rank we built the state with.
    assert state.is_owner(3) is True
    assert state.is_owner(0) is False


def test_runtime_state_contiguous_policy_propagates():
    state = LayerSplitRuntimeState.from_sparse_config(
        sparse_attn_config=_sparse_cfg(
            layersplit_owner_assignment="contiguous"),
        num_layers=61,
        cp_size=8,
        cp_rank=5,
        create_comm_stream=False,
    )
    assert state.ownership.policy == "contiguous"
    # ranks 0..4 own 8 layers each (40 total); rank 5 owns layers 40..46
    assert state.is_owner(40) is True
    assert state.is_owner(46) is True
    assert state.is_owner(47) is False


def test_runtime_state_rejects_partial_rank_transfer():
    # The SparseAttentionConfig validator should already reject this, but
    # belt-and-suspenders: the runtime state factory rejects it too so a
    # programmatic constructor cannot smuggle the case past the validator.
    with pytest.raises(ValueError,
                       match="layersplit_all_cp_ranks_transfer=False"):
        LayerSplitRuntimeState.from_sparse_config(
            sparse_attn_config=_sparse_cfg(
                layersplit_all_cp_ranks_transfer=False),
            num_layers=61,
            cp_size=8,
            cp_rank=0,
            create_comm_stream=False,
        )


def test_runtime_state_rejects_unknown_transfer_backend():
    with pytest.raises(ValueError, match="unknown layersplit_transfer_backend"):
        LayerSplitRuntimeState.from_sparse_config(
            sparse_attn_config=_sparse_cfg(
                layersplit_transfer_backend="mpi"),
            num_layers=8,
            cp_size=2,
            cp_rank=0,
            create_comm_stream=False,
        )


def test_runtime_state_skips_comm_stream_on_cpu_only_host():
    state = LayerSplitRuntimeState.from_sparse_config(
        sparse_attn_config=_sparse_cfg(),
        num_layers=4,
        cp_size=2,
        cp_rank=0,
        create_comm_stream=False,
    )
    assert state.comm_stream is None
    # Even without a comm stream the no-op transfer must not raise.
    state.layersplit_noop_transfer(layer_idx=0)


def test_runtime_state_cp_size_1_collapses_to_owns_everything():
    state = LayerSplitRuntimeState.from_sparse_config(
        sparse_attn_config=_sparse_cfg(),
        num_layers=61,
        cp_size=1,
        cp_rank=0,
        create_comm_stream=False,
    )
    assert state.enabled is True
    assert state.ownership.owner_map == tuple([0] * 61)
    assert state.is_owner(0) and state.is_owner(60)


def test_runtime_state_carries_all_transfer_backends():
    for backend in ("auto", "ucx", "nixl"):
        state = LayerSplitRuntimeState.from_sparse_config(
            sparse_attn_config=_sparse_cfg(
                layersplit_transfer_backend=backend),
            num_layers=8,
            cp_size=2,
            cp_rank=0,
            create_comm_stream=False,
        )
        assert state.transfer_backend == backend


# ---------------------------------------------------------------------------
# build_layersplit_layer_mask (M4: owner-local cache allocation) tests
# ---------------------------------------------------------------------------


def test_layer_mask_none_on_disabled():
    mask = build_layersplit_layer_mask(
        num_layers=61,
        sparse_attn_config=_sparse_cfg(layersplit_enabled=False),
        cp_size=8,
        cp_rank=0,
    )
    assert mask is None


def test_layer_mask_none_when_sparse_attn_config_missing():
    assert build_layersplit_layer_mask(num_layers=61,
                                       sparse_attn_config=None,
                                       cp_size=8,
                                       cp_rank=0) is None


def test_layer_mask_none_when_cp_size_is_1():
    # cp_size <= 1: no reason to mask any layers because there is only one
    # rank; the helper returns None so the caller falls back to the regular
    # "all layers on this rank" path.
    mask = build_layersplit_layer_mask(
        num_layers=61,
        sparse_attn_config=_sparse_cfg(),
        cp_size=1,
        cp_rank=0,
    )
    assert mask is None


def test_layer_mask_round_robin_rank_0_owns_every_8th_layer():
    mask = build_layersplit_layer_mask(
        num_layers=61,
        sparse_attn_config=_sparse_cfg(
            layersplit_owner_assignment="round_robin"),
        cp_size=8,
        cp_rank=0,
    )
    assert mask is not None
    assert len(mask) == 61
    # Rank 0 in round-robin owns layers 0, 8, 16, 24, 32, 40, 48, 56 → 8 layers
    expected = [(layer % 8 == 0) for layer in range(61)]
    assert mask == expected
    assert sum(mask) == 8


def test_layer_mask_round_robin_rank_5_owns_layers_5_13_21_etc():
    mask = build_layersplit_layer_mask(
        num_layers=61,
        sparse_attn_config=_sparse_cfg(),
        cp_size=8,
        cp_rank=5,
    )
    # Rank 5 owns layers 5, 13, 21, 29, 37, 45, 53 → 7 layers (61 % 8 = 5,
    # so ranks 0..4 own one extra layer each)
    expected = [((layer - 5) >= 0 and (layer - 5) % 8 == 0)
                for layer in range(61)]
    assert mask == expected
    assert sum(mask) == 7


def test_layer_mask_contiguous_rank_3_owns_block():
    mask = build_layersplit_layer_mask(
        num_layers=61,
        sparse_attn_config=_sparse_cfg(
            layersplit_owner_assignment="contiguous"),
        cp_size=8,
        cp_rank=3,
    )
    # contiguous: ranks 0..4 own 8 layers each, ranks 5..7 own 7 each. Rank
    # 3 owns layers 24..31 inclusive.
    expected = [(24 <= layer <= 31) for layer in range(61)]
    assert mask == expected
    assert sum(mask) == 8


# ---------------------------------------------------------------------------
# maybe_broadcast_for_layer (M5: owner -> peers per-layer broadcast) tests
# ---------------------------------------------------------------------------


def _make_state(cp_size=8, cp_rank=0, policy="round_robin"):
    return LayerSplitRuntimeState.from_sparse_config(
        sparse_attn_config=_sparse_cfg(
            layersplit_owner_assignment=policy),
        num_layers=61,
        cp_size=cp_size,
        cp_rank=cp_rank,
        create_comm_stream=False,
    )


def test_broadcast_noop_on_disabled_path():
    state = LayerSplitRuntimeState.disabled()
    # Even with a payload + group provided, the disabled path must do
    # absolutely nothing — no broadcast call, no side effects.
    with patch("torch.distributed.broadcast") as mock_bcast:
        state.maybe_broadcast_for_layer(
            layer_idx=3,
            payload=MagicMock(),
            cp_group=MagicMock(),
        )
    mock_bcast.assert_not_called()


def test_broadcast_noop_on_cp_size_1():
    state = _make_state(cp_size=1, cp_rank=0)
    with patch("torch.distributed.broadcast") as mock_bcast:
        state.maybe_broadcast_for_layer(
            layer_idx=3,
            payload=MagicMock(),
            cp_group=MagicMock(),
        )
    mock_bcast.assert_not_called()


def test_broadcast_noop_when_payload_or_group_missing():
    # M5a scaffolding posture: callers stage the wiring without paying for
    # an NCCL call. The runtime collapses to layersplit_noop_transfer.
    state = _make_state(cp_size=4, cp_rank=0)
    with patch("torch.distributed.broadcast") as mock_bcast:
        state.maybe_broadcast_for_layer(layer_idx=3,
                                        payload=None,
                                        cp_group=MagicMock())
        state.maybe_broadcast_for_layer(layer_idx=3,
                                        payload=MagicMock(),
                                        cp_group=None)
    mock_bcast.assert_not_called()


class _FakeDistContext:
    """Context manager that patches torch.distributed.broadcast (and the
    availability/initialization predicates) so unit tests can observe the
    broadcast call without spinning up an actual process group.

    Using patch.dict(sys.modules, {"torch.distributed": fake}) does not
    work here because `import torch.distributed as dist` resolves through
    Python's import machinery which checks the real `torch` package and
    bypasses the sys.modules override. Patching the real attributes on
    the real torch.distributed module is the reliable interception point.
    """

    def __init__(self, is_initialized=True):
        self.is_initialized = is_initialized
        self.broadcast = MagicMock(name="dist.broadcast")
        self._patches = []

    def __enter__(self):
        import torch as _torch
        import torch.distributed as _dist
        self._patches = [
            patch.object(_torch.cuda, "is_available", return_value=True),
            patch.object(_dist, "is_available", return_value=True),
            patch.object(_dist,
                         "is_initialized",
                         return_value=self.is_initialized),
            patch.object(_dist, "broadcast", self.broadcast),
        ]
        for p in self._patches:
            p.__enter__()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.__exit__(*exc)
        return False


def test_broadcast_uses_correct_owner_rank_round_robin():
    # round_robin policy: owner_of(layer) == layer % cp_size. Verify that
    # the broadcast call uses src=owner regardless of which CP rank is
    # invoking it (both owner and non-owner must call broadcast — NCCL
    # collectives require every participant).
    payload = MagicMock(name="kv_payload")
    cp_group = MagicMock(name="cp_group_pg")
    with _FakeDistContext() as fake:
        # Owner rank 0 broadcasts layer 0
        state0 = _make_state(cp_size=8, cp_rank=0)
        state0.maybe_broadcast_for_layer(layer_idx=0,
                                         payload=payload,
                                         cp_group=cp_group,
                                         async_op=False)
        # Non-owner rank 5 also calls (NCCL collective) for layer 0
        state5 = _make_state(cp_size=8, cp_rank=5)
        state5.maybe_broadcast_for_layer(layer_idx=0,
                                         payload=payload,
                                         cp_group=cp_group,
                                         async_op=False)
        # For layer 5, owner is rank 5
        state5.maybe_broadcast_for_layer(layer_idx=5,
                                         payload=payload,
                                         cp_group=cp_group,
                                         async_op=False)
        assert fake.broadcast.call_count == 3
        calls = fake.broadcast.call_args_list
        assert calls[0].kwargs["src"] == 0
        assert calls[1].kwargs["src"] == 0
        assert calls[2].kwargs["src"] == 5
        for c in calls:
            assert c.kwargs["group"] is cp_group


def test_broadcast_uses_correct_owner_rank_contiguous():
    payload = MagicMock()
    cp_group = MagicMock()
    with _FakeDistContext() as fake:
        # cp_size=8 contiguous → ranks 0..4 own 8 layers each, ranks 5..7
        # own 7 each. Layer 30 belongs to rank 3 (which owns 24..31).
        state = LayerSplitRuntimeState.from_sparse_config(
            sparse_attn_config=_sparse_cfg(
                layersplit_owner_assignment="contiguous"),
            num_layers=61,
            cp_size=8,
            cp_rank=0,
            create_comm_stream=False,
        )
        state.maybe_broadcast_for_layer(layer_idx=30,
                                        payload=payload,
                                        cp_group=cp_group,
                                        async_op=False)
        state.maybe_broadcast_for_layer(layer_idx=54,
                                        payload=payload,
                                        cp_group=cp_group,
                                        async_op=False)
        calls = fake.broadcast.call_args_list
        assert len(calls) == 2
        assert calls[0].kwargs["src"] == 3, calls[0]
        assert calls[1].kwargs["src"] == 7, calls[1]  # rank 7 owns 54..60


def test_heartbeat_payload_none_on_disabled():
    assert LayerSplitRuntimeState.disabled().ensure_heartbeat_payload() is None


def test_heartbeat_payload_none_on_cp_size_1():
    state = _make_state(cp_size=1, cp_rank=0)
    assert state.ensure_heartbeat_payload() is None


def test_heartbeat_payload_none_when_cuda_unavailable():
    state = _make_state(cp_size=4, cp_rank=0)
    import torch as _torch
    with patch.object(_torch.cuda, "is_available", return_value=False):
        assert state.ensure_heartbeat_payload() is None


def test_heartbeat_payload_returns_cached_cuda_tensor():
    state = _make_state(cp_size=4, cp_rank=0)
    import torch as _torch
    fake_tensor = MagicMock(spec=_torch.Tensor, name="heartbeat")
    with patch.object(_torch.cuda, "is_available", return_value=True), \
         patch.object(_torch, "zeros", return_value=fake_tensor) as mzeros:
        p1 = state.ensure_heartbeat_payload()
        p2 = state.ensure_heartbeat_payload()
    assert p1 is fake_tensor and p2 is fake_tensor
    # Allocation is lazy and cached — torch.zeros is invoked exactly once
    # across both ensure_* calls.
    assert mzeros.call_count == 1


def test_broadcast_skipped_when_dist_not_initialized():
    payload = MagicMock()
    cp_group = MagicMock()
    with _FakeDistContext(is_initialized=False) as fake:
        state = _make_state(cp_size=4, cp_rank=0)
        state.maybe_broadcast_for_layer(layer_idx=0,
                                        payload=payload,
                                        cp_group=cp_group,
                                        async_op=False)
        fake.broadcast.assert_not_called()


# ---------------------------------------------------------------------------
# build_layersplit_layer_mask sum-across-ranks invariant (M4 cross-check)
# ---------------------------------------------------------------------------


def test_layer_mask_sum_across_ranks_covers_every_layer_exactly_once():
    # Critical correctness invariant for the C++ side: every layer must be
    # owned by exactly one rank, otherwise some layers go un-allocated and
    # the broadcast graph cannot reconstruct the full KV pool.
    for policy in ("round_robin", "contiguous"):
        cp_size = 8
        num_layers = 61
        masks = [
            build_layersplit_layer_mask(
                num_layers=num_layers,
                sparse_attn_config=_sparse_cfg(
                    layersplit_owner_assignment=policy),
                cp_size=cp_size,
                cp_rank=rank,
            ) for rank in range(cp_size)
        ]
        for rank, mask in enumerate(masks):
            assert mask is not None and len(mask) == num_layers, policy
        # Each layer is True in exactly one rank's mask.
        per_layer_owners = [
            sum(masks[rank][layer] for rank in range(cp_size))
            for layer in range(num_layers)
        ]
        assert per_layer_owners == [1] * num_layers, (policy, per_layer_owners)
        # Total layers owned across ranks equals num_layers.
        assert sum(sum(m) for m in masks) == num_layers, policy

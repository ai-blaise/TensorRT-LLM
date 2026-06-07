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



def test_layersplit_owner_map_uses_global_layer_domain_with_local_mask():
    # Regression for the r20 TP2xCP2 prefill crash: DSACacheManager receives
    # num_layers=sum(layer_mask) for its trimmed local pool, but ownership is
    # queried with global layer ids. The owner table must be built over
    # len(layer_mask), not the local pool length.
    layer_mask = [False, False, True, True]
    local_num_layers = sum(layer_mask)
    ownership_num_layers = len(layer_mask)
    state = LayerSplitRuntimeState.from_sparse_config(
        sparse_attn_config=_sparse_cfg(
            layersplit_owner_assignment="contiguous",
            layersplit_owner_local_alloc=True,
        ),
        num_layers=ownership_num_layers,
        cp_size=2,
        cp_rank=1,
        create_comm_stream=False,
    )

    assert local_num_layers == 2
    assert state.ownership.num_layers == 4
    assert state.is_owner(0) is False
    assert state.is_owner(2) is True
    assert state.is_owner(3) is True


def test_dsa_non_owned_helper_ignores_out_of_domain_pool_offsets():
    # Regression for the owner-local dense scratch warmup crash: some dense/KVarN
    # callers probe local pool offsets while the LayerSplit ownership table is in
    # the global layer-id domain. Those local offsets must not index the global
    # owner tuple and crash before the scratch route can be used.
    from tensorrt_llm._torch.attention_backend.sparse.dsa import DSACacheManager

    state = LayerSplitRuntimeState.from_sparse_config(
        sparse_attn_config=_sparse_cfg(
            layersplit_owner_assignment="contiguous",
            layersplit_owner_local_alloc=True,
        ),
        num_layers=4,
        cp_size=2,
        cp_rank=1,
        create_comm_stream=False,
    )
    mgr = DSACacheManager.__new__(DSACacheManager)
    mgr.layersplit_state = state
    mgr.layer_offsets = {2: 0, 4: 1, 99: 2}

    assert mgr._layersplit_non_owned(0) is True
    assert mgr._layersplit_non_owned(2) is False
    assert mgr._layersplit_non_owned(4) is False
    assert mgr._layersplit_non_owned(99) is False
    assert mgr._layersplit_non_owned(100) is True


def test_owner_local_transfer_state_uses_real_model_layers_not_padded_mask():
    # r20 TP2xCP2 prefill -> TP4 decode uses a padded owner-local mask so
    # every CP rank has the same local pool row count. The C++ transfer state
    # must still advertise the real attention-layer domain (61), not the
    # padded mask/ownership domain or the local pool row count.
    from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import (
        _normalise_layersplit_total_kv_heads_per_layer,
    )

    mgr = SimpleNamespace(
        layersplit_state=SimpleNamespace(
            enabled=True,
            owner_local_alloc=True,
            ownership=SimpleNamespace(num_layers=62),
        ),
        layersplit_model_num_layers=61,
        layersplit_cache_transfer_model_layers=61,
        layersplit_local_pool_layers=31,
    )

    transfer_heads = _normalise_layersplit_total_kv_heads_per_layer(
        mgr, [1] * 31)

    assert len(transfer_heads) == 61
    assert set(transfer_heads) == {1}

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


def test_runtime_state_translates_cp_local_owner_to_global_src_rank():
    state = LayerSplitRuntimeState.from_sparse_config(
        sparse_attn_config=_sparse_cfg(),
        num_layers=4,
        cp_size=2,
        cp_rank=0,
        create_comm_stream=False,
    )
    state.bind_cp_group(cp_group=object(), cp_group_ranks=(4, 5))

    assert state.owner_of(0) == 0
    assert state.broadcast_src_rank(0) == 4
    assert state.owner_of(1) == 1
    assert state.broadcast_src_rank(1) == 5


def test_runtime_state_rejects_owner_outside_bound_cp_group_ranks():
    state = LayerSplitRuntimeState.from_sparse_config(
        sparse_attn_config=_sparse_cfg(),
        num_layers=2,
        cp_size=2,
        cp_rank=0,
        create_comm_stream=False,
    )
    state.bind_cp_group(cp_group=object(), cp_group_ranks=(7,))

    with pytest.raises(ValueError, match="outside cp_group_ranks"):
        state.broadcast_src_rank(1)

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
    # Balanced layer_mask (M5d-tight-v2): length = ceil(61/8)*8 = 64.
    # Rank 0 owns the 8 real layers 0,8,...,56; the 3 phantom slots
    # (61, 62, 63) are False because rank 0 already hits target=8.
    # The trimmed mask is only produced under owner_local_alloc=True.
    mask = build_layersplit_layer_mask(
        num_layers=61,
        sparse_attn_config=_sparse_cfg(
            layersplit_owner_assignment="round_robin",
            layersplit_owner_local_alloc=True),
        cp_size=8,
        cp_rank=0,
    )
    assert mask is not None
    assert len(mask) == 64  # ceil(61/8)*8
    expected = [(layer % 8 == 0) for layer in range(61)] + [False, False,
                                                              False]
    assert mask == expected
    assert sum(mask) == 8  # target_per_rank = ceil(61/8) = 8


def test_layer_mask_round_robin_rank_5_owns_layers_5_13_21_etc():
    # Balanced layer_mask: rank 5 owns 7 real layers (5, 13, ..., 53) +
    # 1 phantom True at index 61 to hit target_per_rank = 8. Indices 62,
    # 63 stay False (already at target).
    mask = build_layersplit_layer_mask(
        num_layers=61,
        sparse_attn_config=_sparse_cfg(layersplit_owner_local_alloc=True),
        cp_size=8,
        cp_rank=5,
    )
    assert len(mask) == 64
    real_owned = [((layer - 5) >= 0 and (layer - 5) % 8 == 0)
                  for layer in range(61)]
    # The single missing layer is filled by a phantom True at the first
    # padding slot (idx 61); remaining padding slots stay False.
    expected = real_owned + [True, False, False]
    assert mask == expected
    assert sum(mask) == 8


def test_layer_mask_contiguous_rank_3_owns_block():
    # Balanced layer_mask: rank 3 owns 8 contiguous real layers
    # (24..31). target_per_rank = 8, no padding needed.
    mask = build_layersplit_layer_mask(
        num_layers=61,
        sparse_attn_config=_sparse_cfg(
            layersplit_owner_assignment="contiguous",
            layersplit_owner_local_alloc=True),
        cp_size=8,
        cp_rank=3,
    )
    assert len(mask) == 64
    expected = [(24 <= layer <= 31) for layer in range(61)] + [False, False,
                                                                False]
    assert mask == expected
    assert sum(mask) == 8


def test_layer_mask_none_when_owner_local_alloc_off():
    # Default (replicated) posture: even with LayerSplit enabled + cp_size>1,
    # the mask is None so every CP rank keeps the full pool. This is the
    # correctness-first prefill posture that lets the dense-MLA C++ attention
    # path resolve a pool slot for every layer (layer_offsets stays complete).
    for policy in ("round_robin", "contiguous"):
        mask = build_layersplit_layer_mask(
            num_layers=61,
            sparse_attn_config=_sparse_cfg(
                layersplit_owner_assignment=policy),  # owner_local_alloc unset
            cp_size=8,
            cp_rank=0,
        )
        assert mask is None, policy


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


def test_heartbeat_default_payload_is_16_bytes_int32():
    # The default M5c heartbeat must remain int32 × 4 = 16 B until
    # callers explicitly opt into the larger M5d-bandwidth payload.
    # This protects the existing wiring + bench baselines from a silent
    # payload-size regression when the API expands.
    state = _make_state(cp_size=4, cp_rank=0)
    import torch as _torch
    with patch.object(_torch.cuda, "is_available", return_value=True), \
         patch.object(_torch, "zeros") as mzeros:
        mzeros.return_value = MagicMock(spec=_torch.Tensor)
        state.ensure_heartbeat_payload()  # payload_bytes=None
    # zeros called with (4, dtype=int32, device='cuda')
    args, kwargs = mzeros.call_args
    assert args == (4, ) or kwargs.get("size") == 4 or args[0] == 4
    assert kwargs.get("dtype") == _torch.int32
    assert kwargs.get("device") == "cuda"


def test_heartbeat_bandwidth_mode_allocates_uint8_block():
    # Explicit payload_bytes > 16 (M5d-bandwidth posture): the broadcast
    # publishes a realistic-sized payload (uint8 block) so the comm
    # stream / NCCL channels see production-shaped traffic.
    state = _make_state(cp_size=4, cp_rank=0)
    import torch as _torch
    with patch.object(_torch.cuda, "is_available", return_value=True), \
         patch.object(_torch, "zeros") as mzeros:
        mzeros.return_value = MagicMock(spec=_torch.Tensor)
        state.ensure_heartbeat_payload(payload_bytes=1024 * 1024)  # 1 MB
    args, kwargs = mzeros.call_args
    assert args[0] == 1024 * 1024
    assert kwargs.get("dtype") == _torch.uint8
    assert kwargs.get("device") == "cuda"


def test_heartbeat_bandwidth_size_is_cached():
    # First-call payload_bytes wins; subsequent calls return the cached
    # tensor regardless of payload_bytes argument. Callers that want a
    # different size must reset the runtime state.
    state = _make_state(cp_size=4, cp_rank=0)
    import torch as _torch
    fake_a = MagicMock(spec=_torch.Tensor, name="big")
    with patch.object(_torch.cuda, "is_available", return_value=True), \
         patch.object(_torch, "zeros", return_value=fake_a) as mzeros:
        first = state.ensure_heartbeat_payload(payload_bytes=8192)
        second = state.ensure_heartbeat_payload(payload_bytes=16)
        third = state.ensure_heartbeat_payload(payload_bytes=2 * 1024 * 1024)
    assert first is fake_a
    assert second is fake_a
    assert third is fake_a
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
    # Critical correctness invariant: in the REAL layer range (0..num_layers)
    # every layer is owned by exactly one rank. Phantom slots at indices
    # >= num_layers are owned by AT MOST one rank — they exist purely to
    # balance sum(mask) per rank for consistent num_blocks at the C++
    # pool layer.
    for policy in ("round_robin", "contiguous"):
        cp_size = 8
        num_layers = 61
        total_length = ((num_layers + cp_size - 1) // cp_size) * cp_size
        masks = [
            build_layersplit_layer_mask(
                num_layers=num_layers,
                sparse_attn_config=_sparse_cfg(
                    layersplit_owner_assignment=policy,
                    layersplit_owner_local_alloc=True),
                cp_size=cp_size,
                cp_rank=rank,
            ) for rank in range(cp_size)
        ]
        for rank, mask in enumerate(masks):
            assert mask is not None and len(mask) == total_length, policy
        # Real-layer rows: exactly one rank owns each.
        per_layer_owners = [
            sum(masks[rank][layer] for rank in range(cp_size))
            for layer in range(num_layers)
        ]
        assert per_layer_owners == [1] * num_layers, (policy, per_layer_owners)
        # Phantom rows are pure padding to balance sum(mask) per rank —
        # multiple ranks may "own" the same phantom slot because their
        # layer counts differed before padding. That's correctness-safe
        # because the C++ pool allocates the slot on every rank that has
        # it True, but the model's per-layer loop only iterates
        # 0..num_layers and so the phantom slots are never read. The only
        # invariant that matters at this layer is sum(mask) per rank
        # being identical (checked below) so num_blocks is consistent
        # across ranks.
        # Each rank's sum is identical and equals target_per_rank — the
        # core invariant that makes num_blocks consistent at the C++ pool.
        target = (num_layers + cp_size - 1) // cp_size
        for rank, m in enumerate(masks):
            assert sum(m) == target, (policy, rank, sum(m), target)


def test_balanced_layer_mask_keeps_num_layers_divisible_case_unpadded():
    # When num_layers % cp_size == 0 the balanced mask reduces exactly to
    # the unbalanced "owned-only" mask (no phantom slots are needed
    # because target_per_rank * cp_size == num_layers).
    mask = build_layersplit_layer_mask(
        num_layers=8,
        sparse_attn_config=_sparse_cfg(layersplit_owner_local_alloc=True),
        cp_size=4,
        cp_rank=2,
    )
    assert len(mask) == 8
    expected = [(layer % 4 == 2) for layer in range(8)]
    assert mask == expected
    assert sum(mask) == 2


def test_balanced_layer_mask_extreme_cp_greater_than_num_layers():
    # CP > num_layers: target_per_rank = 1, ranks 0..num_layers-1 own
    # their natural layer plus 0 phantoms; ranks >= num_layers own only
    # a phantom slot (so the C++ pool size is consistent).
    mask = build_layersplit_layer_mask(
        num_layers=2,
        sparse_attn_config=_sparse_cfg(layersplit_owner_local_alloc=True),
        cp_size=4,
        cp_rank=3,
    )
    # length = ceil(2/4)*4 = 4
    # ownership for cp_size=4: owner_map = (0, 1) — only ranks 0 and 1
    # have a real owned layer. Rank 3 needs target=1 True, all from
    # phantom slots.
    assert len(mask) == 4
    # No real layers owned by rank 3.
    assert not mask[0] and not mask[1]
    # Exactly one phantom is True to hit target=1.
    assert sum(mask[2:]) == 1
    assert sum(mask) == 1


# ---------------------------------------------------------------------------
# prefetch_for_layer / wait_for_prefetched_layer (M6 cross-layer overlap) tests
# ---------------------------------------------------------------------------


class _FakeM6DistContext(_FakeDistContext):
    """Variant that ALSO injects a fake comm_stream + cuda.Event so the
    prefetch path runs without a real CUDA device.

    The base _FakeDistContext patches dist.broadcast and dist availability;
    M6 additionally needs torch.cuda.stream(...) and torch.cuda.Event() to
    behave without an active CUDA context.
    """

    def __enter__(self):
        super().__enter__()
        import torch as _torch
        from contextlib import contextmanager

        @contextmanager
        def _fake_stream_ctx(stream):
            yield

        # Fake Event: records the comm_stream it was recorded on so the
        # test can assert prefetch wired the right stream.
        class _FakeEvent:
            def __init__(self):
                self.recorded_on = None

            def record(self, stream=None):
                self.recorded_on = stream

        # Fake current_stream that swallows wait_event() calls so
        # wait_for_prefetched_layer's `current_stream().wait_event(e)`
        # works without a real CUDA context.
        self.current_stream_mock = MagicMock(name="current_stream")
        self.current_stream_mock.wait_event = MagicMock(name="wait_event")

        self._extra_patches = [
            patch.object(_torch.cuda, "stream", side_effect=_fake_stream_ctx),
            patch.object(_torch.cuda, "Event", _FakeEvent),
            patch.object(_torch.cuda,
                         "current_stream",
                         return_value=self.current_stream_mock),
        ]
        for p in self._extra_patches:
            p.__enter__()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._extra_patches):
            p.__exit__(*exc)
        return super().__exit__(*exc)


def _make_state_with_comm_stream(cp_size=4, cp_rank=0, policy="round_robin"):
    """Like _make_state but injects a MagicMock comm_stream so the M6
    prefetch path uses the side-stream branch."""
    state = LayerSplitRuntimeState.from_sparse_config(
        sparse_attn_config=_sparse_cfg(
            layersplit_owner_assignment=policy),
        num_layers=8,
        cp_size=cp_size,
        cp_rank=cp_rank,
        create_comm_stream=False,
    )
    state.comm_stream = MagicMock(name="comm_stream")
    return state


def test_prefetch_noop_on_disabled_path():
    state = LayerSplitRuntimeState.disabled()
    with patch("torch.distributed.broadcast") as mock_bcast:
        assert state.prefetch_for_layer(layer_idx=0,
                                        payload=MagicMock(),
                                        cp_group=MagicMock()) is False
    mock_bcast.assert_not_called()


def test_prefetch_noop_on_cp_size_1():
    state = _make_state_with_comm_stream(cp_size=1, cp_rank=0)
    with patch("torch.distributed.broadcast") as mock_bcast:
        assert state.prefetch_for_layer(layer_idx=0,
                                        payload=MagicMock(),
                                        cp_group=MagicMock()) is False
    mock_bcast.assert_not_called()


def test_prefetch_noop_when_payload_or_group_missing():
    state = _make_state_with_comm_stream(cp_size=4, cp_rank=0)
    with patch("torch.distributed.broadcast") as mock_bcast:
        assert state.prefetch_for_layer(layer_idx=0,
                                        payload=None,
                                        cp_group=MagicMock()) is False
        assert state.prefetch_for_layer(layer_idx=0,
                                        payload=MagicMock(),
                                        cp_group=None) is False
    mock_bcast.assert_not_called()


def test_prefetch_records_event_on_comm_stream_and_uses_correct_owner():
    payload = MagicMock(name="kv_payload")
    cp_group = MagicMock(name="cp_group_pg")
    state = _make_state_with_comm_stream(cp_size=4, cp_rank=0)
    with _FakeM6DistContext() as fake:
        # Layer 3 owner = 3 % 4 = 3
        assert state.prefetch_for_layer(layer_idx=3,
                                        payload=payload,
                                        cp_group=cp_group) is True
    fake.broadcast.assert_called_once()
    call = fake.broadcast.call_args
    assert call.kwargs["src"] == 3
    assert call.kwargs["group"] is cp_group
    assert call.kwargs["async_op"] is True
    # The event must have landed in _prefetched_events for (layer=3,
    # channel='kv') (the default channel) and be recorded on the comm
    # stream.
    assert (3, "kv") in state._prefetched_events
    event = state._prefetched_events[(3, "kv")]
    assert event.recorded_on is state.comm_stream


def test_wait_for_prefetched_layer_no_op_when_no_prefetch():
    state = _make_state_with_comm_stream(cp_size=4, cp_rank=0)
    # No prefetch in flight: caller must do a sync broadcast.
    with _FakeM6DistContext() as fake:
        assert state.wait_for_prefetched_layer(layer_idx=0) is False
    # current_stream().wait_event must NOT be called.
    fake.current_stream_mock.wait_event.assert_not_called()


def test_wait_for_prefetched_layer_consumes_event_and_signals_stream():
    payload = MagicMock(name="kv_payload")
    cp_group = MagicMock(name="cp_group_pg")
    state = _make_state_with_comm_stream(cp_size=4, cp_rank=0)
    with _FakeM6DistContext() as fake:
        state.prefetch_for_layer(layer_idx=3,
                                 payload=payload,
                                 cp_group=cp_group)
        # Wait on layer 3: returns True, removes the event, signals current_stream
        assert state.wait_for_prefetched_layer(layer_idx=3) is True
        # Calling again returns False since the event was popped.
        assert state.wait_for_prefetched_layer(layer_idx=3) is False
    assert 3 not in state._prefetched_events
    # current_stream().wait_event should have been called exactly once
    fake.current_stream_mock.wait_event.assert_called_once()


def test_prefetch_pipeline_overlaps_consecutive_layers():
    # Exercise the M6 pipeline: at layer L we prefetch L+1 while computing
    # L; at layer L+1 we consume the prefetch and prefetch L+2. After 4
    # iterations we should have exactly one event in flight (for the next
    # layer) and the previous events should have all been consumed.
    payload = MagicMock(name="kv_payload")
    cp_group = MagicMock(name="cp_group_pg")
    state = _make_state_with_comm_stream(cp_size=4, cp_rank=0)
    with _FakeM6DistContext() as fake:
        # Bootstrap: no prefetch for layer 0 yet, so it's a sync miss.
        assert state.wait_for_prefetched_layer(0) is False
        state.maybe_broadcast_for_layer(layer_idx=0,
                                        payload=payload,
                                        cp_group=cp_group,
                                        async_op=False)
        state.prefetch_for_layer(layer_idx=1,
                                 payload=payload,
                                 cp_group=cp_group)

        for L in range(1, 4):
            assert state.wait_for_prefetched_layer(L) is True, L
            state.prefetch_for_layer(layer_idx=L + 1,
                                     payload=payload,
                                     cp_group=cp_group)

    # Only layer 4's broadcast remains in flight after the loop. Keys
    # are (layer_idx, channel) tuples since M8b — the default channel
    # is "kv".
    assert list(state._prefetched_events.keys()) == [(4, "kv")]
    # Total broadcast calls: 1 (sync layer 0) + 4 prefetches (layers 1..4)
    assert fake.broadcast.call_count == 5
    # The 4 prefetches must have been async; the bootstrap was sync.
    async_calls = [
        c for c in fake.broadcast.call_args_list
        if c.kwargs.get("async_op") is True
    ]
    sync_calls = [
        c for c in fake.broadcast.call_args_list
        if c.kwargs.get("async_op") is False
    ]
    assert len(async_calls) == 4
    assert len(sync_calls) == 1


def test_clear_prefetched_events_empties_dict_without_waiting():
    payload = MagicMock(name="kv_payload")
    cp_group = MagicMock(name="cp_group_pg")
    state = _make_state_with_comm_stream(cp_size=4, cp_rank=0)
    with _FakeM6DistContext():
        state.prefetch_for_layer(layer_idx=1,
                                 payload=payload,
                                 cp_group=cp_group)
        state.prefetch_for_layer(layer_idx=2,
                                 payload=payload,
                                 cp_group=cp_group)
        assert len(state._prefetched_events) == 2
        state.clear_prefetched_events()
        assert state._prefetched_events == {}


def test_per_layer_payloads_are_distinct_across_layers():
    # M6 invariant: with prefetch, two consecutive layers' broadcasts can
    # be in flight on the comm stream simultaneously. If they shared the
    # same payload tensor the second broadcast would corrupt the first
    # one's data. Verify ensure_heartbeat_payload returns a fresh tensor
    # for each layer_idx.
    import torch as _torch
    with patch.object(_torch.cuda, "is_available", return_value=True), \
         patch.object(_torch, "zeros", side_effect=lambda *a, **k: MagicMock(
             name=f"payload-{a}-{k}")):
        state = _make_state_with_comm_stream(cp_size=4, cp_rank=0)
        p0 = state.ensure_heartbeat_payload(layer_idx=0)
        p1 = state.ensure_heartbeat_payload(layer_idx=1)
        p0_again = state.ensure_heartbeat_payload(layer_idx=0)
        shared = state.ensure_heartbeat_payload()  # legacy shared path
    assert p0 is not p1
    assert p0 is p0_again  # cached per layer
    assert shared is not p0  # shared path != per-layer path
    # Both layer-0 and layer-1 payloads sit in the per-layer dict under
    # the default 'kv' channel (M8b key shape is (layer_idx, channel)).
    assert (0, "kv") in state._per_layer_payloads
    assert (1, "kv") in state._per_layer_payloads


# ---------------------------------------------------------------------------
# 2-phase per-layer broadcast (M8b: indexer-first + dense-KV channels)
# ---------------------------------------------------------------------------


def _make_state_with_two_streams(cp_size=4, cp_rank=0, policy="round_robin"):
    """Like _make_state_with_comm_stream but also injects an
    indexer_comm_stream so the M8b two-stream path is exercised."""
    state = _make_state_with_comm_stream(cp_size=cp_size,
                                         cp_rank=cp_rank,
                                         policy=policy)
    state.indexer_comm_stream = MagicMock(name="indexer_comm_stream")
    return state


def test_channel_rejects_unknown_value():
    state = _make_state_with_comm_stream(cp_size=4, cp_rank=0)
    with pytest.raises(ValueError, match="unknown layersplit channel"):
        state.ensure_heartbeat_payload(layer_idx=0, channel="bogus")
    with pytest.raises(ValueError, match="unknown layersplit channel"):
        state.prefetch_for_layer(layer_idx=0,
                                 payload=MagicMock(),
                                 cp_group=MagicMock(),
                                 channel="bogus")
    with pytest.raises(ValueError, match="unknown layersplit channel"):
        state.wait_for_prefetched_layer(layer_idx=0, channel="bogus")


def test_per_layer_payloads_are_distinct_per_channel():
    import torch as _torch
    with patch.object(_torch.cuda, "is_available", return_value=True), \
         patch.object(_torch, "zeros", side_effect=lambda *a, **k: MagicMock(
             name=f"payload-{a}-{k}")):
        state = _make_state_with_comm_stream(cp_size=4, cp_rank=0)
        p0_kv = state.ensure_heartbeat_payload(layer_idx=0, channel="kv")
        p0_ix = state.ensure_heartbeat_payload(layer_idx=0, channel="indexer")
        p1_kv = state.ensure_heartbeat_payload(layer_idx=1, channel="kv")
    # All four combinations are distinct.
    assert p0_kv is not p0_ix
    assert p0_kv is not p1_kv
    assert p0_ix is not p1_kv
    # And the dict carries the right keys.
    assert (0, "kv") in state._per_layer_payloads
    assert (0, "indexer") in state._per_layer_payloads
    assert (1, "kv") in state._per_layer_payloads


def test_prefetch_indexer_channel_uses_indexer_comm_stream():
    payload = MagicMock(name="kv_payload")
    cp_group = MagicMock(name="cp_group_pg")
    state = _make_state_with_two_streams(cp_size=4, cp_rank=0)
    with _FakeM6DistContext():
        # kv channel records on the primary comm stream
        state.prefetch_for_layer(layer_idx=1,
                                 payload=payload,
                                 cp_group=cp_group,
                                 channel="kv")
        # indexer channel records on the indexer comm stream
        state.prefetch_for_layer(layer_idx=1,
                                 payload=payload,
                                 cp_group=cp_group,
                                 channel="indexer")
    assert (1, "kv") in state._prefetched_events
    assert (1, "indexer") in state._prefetched_events
    kv_event = state._prefetched_events[(1, "kv")]
    ix_event = state._prefetched_events[(1, "indexer")]
    assert kv_event.recorded_on is state.comm_stream
    assert ix_event.recorded_on is state.indexer_comm_stream
    # The two streams must be distinct objects.
    assert state.comm_stream is not state.indexer_comm_stream


def test_prefetch_indexer_channel_falls_back_to_primary_stream():
    # If indexer_comm_stream is not allocated (single-stream posture),
    # the indexer channel runs on the primary comm_stream.
    payload = MagicMock(name="kv_payload")
    cp_group = MagicMock(name="cp_group_pg")
    state = _make_state_with_comm_stream(cp_size=4, cp_rank=0)
    state.indexer_comm_stream = None  # explicit single-stream posture
    with _FakeM6DistContext():
        state.prefetch_for_layer(layer_idx=1,
                                 payload=payload,
                                 cp_group=cp_group,
                                 channel="indexer")
    event = state._prefetched_events[(1, "indexer")]
    assert event.recorded_on is state.comm_stream


def test_wait_indexer_channel_returns_false_when_only_kv_prefetched():
    # A channel without a prefetch must NOT pop another channel's event.
    payload = MagicMock(name="payload")
    cp_group = MagicMock(name="cp_group_pg")
    state = _make_state_with_two_streams(cp_size=4, cp_rank=0)
    with _FakeM6DistContext():
        state.prefetch_for_layer(layer_idx=3,
                                 payload=payload,
                                 cp_group=cp_group,
                                 channel="kv")
        # waiting on indexer channel for layer 3 should return False
        assert state.wait_for_prefetched_layer(layer_idx=3,
                                               channel="indexer") is False
        # the kv event is still in flight
        assert (3, "kv") in state._prefetched_events
        # waiting on kv channel for layer 3 returns True
        assert state.wait_for_prefetched_layer(layer_idx=3,
                                               channel="kv") is True




def test_two_phase_pipeline_keeps_both_channels_pipelined():
    # Drive the 2-phase pattern by hand: at layer L we wait on both
    # channels for L (popping their events) and prefetch both for L+1.
    payload = MagicMock(name="payload")
    cp_group = MagicMock(name="cp_group_pg")
    state = _make_state_with_two_streams(cp_size=4, cp_rank=0)
    num_layers = 4
    with _FakeM6DistContext() as fake:
        # Bootstrap layer 0 on both channels.
        for ch in ("indexer", "kv"):
            assert state.wait_for_prefetched_layer(0, channel=ch) is False
            state.maybe_broadcast_for_layer(layer_idx=0,
                                            payload=payload,
                                            cp_group=cp_group,
                                            async_op=False,
                                            channel=ch)
        # Prefetch layer 1 on both channels
        for ch in ("indexer", "kv"):
            state.prefetch_for_layer(layer_idx=1,
                                     payload=payload,
                                     cp_group=cp_group,
                                     channel=ch)
        # Pipeline 1..num_layers-1
        for L in range(1, num_layers):
            for ch in ("indexer", "kv"):
                assert state.wait_for_prefetched_layer(L,
                                                      channel=ch) is True, (L,
                                                                            ch)
            next_layer = L + 1
            if next_layer < num_layers:
                for ch in ("indexer", "kv"):
                    state.prefetch_for_layer(layer_idx=next_layer,
                                             payload=payload,
                                             cp_group=cp_group,
                                             channel=ch)
    # No leftover events after a clean pipeline run.
    assert state._prefetched_events == {}
    # Total broadcast calls: 2 sync (layer 0 indexer+kv) + 2*3 prefetches
    # (layers 1..3 on both channels)
    assert fake.broadcast.call_count == 2 + 6


def test_owner_local_indexer_scratch_lazily_allocates_for_non_owned_layer():
    import torch

    from tensorrt_llm._torch.attention_backend.sparse.dsa import DSACacheManager

    mgr = DSACacheManager.__new__(DSACacheManager)
    mgr.layersplit_state = SimpleNamespace(
        enabled=True,
        cp_size=2,
        owner_local_alloc=True,
        is_owner=lambda layer_idx: layer_idx >= 2,
    )
    mgr._layersplit_indexer_k_scratch = None
    mgr.index_head_dim = 128
    mgr.quant_block_size = 128
    mgr.tokens_per_block = 4
    mgr.num_blocks = 3
    mgr.use_fp4 = True
    mgr.indexer_k_cache_pool_per_layer = [
        torch.empty((mgr.num_blocks, mgr.tokens_per_block * 68),
                    dtype=torch.uint8)
    ]
    mgr.layer_offsets = {2: 0, 3: 1}

    scratch = mgr.get_indexer_k_cache_buffers(0)

    assert scratch is mgr._layersplit_indexer_k_scratch
    assert scratch.shape == (mgr.num_blocks, mgr.tokens_per_block, 1, 68)


def test_owner_local_hisa_scratch_lazily_allocates_for_non_owned_layer():
    import torch

    from tensorrt_llm._torch.attention_backend.sparse.dsa import DSACacheManager

    mgr = DSACacheManager.__new__(DSACacheManager)
    mgr.layersplit_state = SimpleNamespace(
        enabled=True,
        cp_size=2,
        owner_local_alloc=True,
        is_owner=lambda layer_idx: layer_idx >= 2,
    )
    mgr.enable_hisa_page_reps = True
    mgr._layersplit_hisa_pagerep_scratch = None
    mgr._layersplit_hisa_pagecount_scratch = None
    mgr.indexer_hisa_page_reps_per_layer = [torch.empty((3, 128))]
    mgr.indexer_hisa_page_counts_per_layer = [torch.empty((3,), dtype=torch.int32)]
    mgr.layer_offsets = {2: 0, 3: 1}

    reps, counts = mgr.get_indexer_hisa_page_rep_buffers(0)

    assert reps is mgr._layersplit_hisa_pagerep_scratch
    assert counts is mgr._layersplit_hisa_pagecount_scratch
    assert reps.shape == (3, 128)
    assert counts.shape == (3,)



def test_owner_local_dense_scratch_routing_lazily_builds_nonowned_rows():
    import torch

    from tensorrt_llm._torch.attention_backend.sparse.dsa import DSACacheManager

    class _Impl:
        def get_primary_pool_data(self, local_offset):
            assert local_offset == 0
            return torch.empty((3, 4, 2), dtype=torch.uint8)

    mgr = DSACacheManager.__new__(DSACacheManager)
    mgr.layersplit_state = SimpleNamespace(
        enabled=True,
        cp_size=2,
        cp_rank=1,
        owner_local_alloc=True,
        is_owner=lambda layer_idx: layer_idx >= 2,
        ownership=SimpleNamespace(
            num_layers=4,
            owned_layers=lambda cp_rank: (2, 3),
        ),
    )
    mgr._layersplit_dense_kv_scratch = None
    mgr._layersplit_dense_scratch_pool_index = None
    mgr._layersplit_kv_cache_pool_pointers_ls = None
    mgr._layersplit_kv_cache_pool_mapping_ls = None
    mgr._layersplit_nonowned_layer_rows = {}
    mgr.layer_offsets = {2: 0, 3: 1}
    mgr.impl = _Impl()
    mgr.kv_cache_pool_pointers = torch.tensor([[123, 0]], dtype=torch.int64)
    mgr.kv_cache_pool_mapping = torch.tensor([[0, 0], [0, 1]], dtype=torch.int32)
    mgr.host_kv_cache_block_offsets = torch.zeros((1, 2, 2, 4), dtype=torch.int32)
    mgr.num_local_layers = 2
    mgr.dtype = None

    assert mgr._ensure_layersplit_dense_scratch_routing() is True

    assert mgr._layersplit_dense_kv_scratch.shape == (3, 4, 2)
    assert mgr._layersplit_dense_scratch_pool_index == 1
    assert mgr._layersplit_nonowned_layer_rows == {0: 2, 1: 3}
    assert mgr._layersplit_kv_cache_pool_pointers_ls.shape == (2, 2)
    assert mgr._layersplit_kv_cache_pool_mapping_ls.tolist() == [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 0],
    ]
    assert mgr.host_kv_cache_block_offsets.shape[0] == 2
    assert mgr.get_buffers(0) is mgr._layersplit_dense_kv_scratch


def test_owner_local_global_owned_layer_bypasses_owner_map_bounds_check():
    from tensorrt_llm._torch.attention_backend.sparse.dsa import DSACacheManager

    mgr = DSACacheManager.__new__(DSACacheManager)
    mgr.layersplit_state = SimpleNamespace(
        enabled=True,
        cp_size=2,
        cp_rank=1,
        owner_local_alloc=True,
        ownership=SimpleNamespace(
            num_layers=31,
            is_owner=lambda layer_idx, cp_rank: (_ for _ in ()).throw(IndexError()),
        ),
        is_owner=lambda layer_idx: (_ for _ in ()).throw(IndexError()),
    )
    mgr.layer_offsets = {31: 0}

    assert mgr._layersplit_non_owned(31) is False
    assert mgr._layersplit_non_owned(0) is True
    assert mgr._layersplit_non_owned(62) is True

"""Unit tests for tensorrt_llm._torch.attention_backend.sparse.layersplit."""
import pytest

from tensorrt_llm._torch.attention_backend.sparse.layersplit import (
    LayerSplitOwnership, compute_owner_assignment)


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

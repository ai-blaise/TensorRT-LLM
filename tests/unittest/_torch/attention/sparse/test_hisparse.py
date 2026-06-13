from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.attention_backend.sparse.hisparse import (
    OPTRTHiSparseCoordinator)


def _cfg(enabled=False):
    return SimpleNamespace(
        hisparse_enabled=enabled,
        hisparse_mode="dense_mla_kvarn",
        hisparse_hot_blocks_per_req=2,
        hisparse_host_to_device_ratio=8,
    )


def test_hisparse_coordinator_disabled_path_is_noop():
    coordinator = OPTRTHiSparseCoordinator(_cfg())

    assert coordinator.enabled is False
    assert coordinator.step_id == 0
    coordinator.assert_startup_ready()
    coordinator.reset_step()
    assert coordinator.step_id == 1
    assert coordinator.map_topk_to_hot_pool(topk_indices=None,
                                            metadata=None,
                                            layer_idx=0,
                                            skip_topk=False,
                                            is_generation=True) is None


def test_hisparse_coordinator_enabled_path_fails_closed_until_kernel_ready():
    coordinator = OPTRTHiSparseCoordinator(_cfg(enabled=True))

    assert coordinator.enabled is True
    with pytest.raises(NotImplementedError, match="fail-closed"):
        coordinator.assert_startup_ready()

    coordinator.configure_packed_tiers(num_layers=1,
                                       tokens_per_block=64,
                                       packed_bytes_per_block=2048,
                                       logical_host_capacity_blocks=4,
                                       hot_device_capacity_blocks=2)
    with pytest.raises(NotImplementedError, match="swap-in kernel"):
        coordinator.assert_startup_ready()
    with pytest.raises(NotImplementedError, match="hot-pool TopK mapping"):
        coordinator.map_topk_to_hot_pool(topk_indices=None,
                                         metadata=None,
                                         layer_idx=0,
                                         skip_topk=False,
                                         is_generation=True)


def test_hisparse_request_allocation_commit_and_release():
    coordinator = OPTRTHiSparseCoordinator(_cfg())
    tier = coordinator.configure_packed_tiers(num_layers=2,
                                             tokens_per_block=64,
                                             packed_bytes_per_block=2048,
                                             logical_host_capacity_blocks=4,
                                             hot_device_capacity_blocks=2)

    assert tier.num_layers == 2
    state = coordinator.reserve_request(req_pool_idx=7, num_prompt_blocks=3)
    assert state.host_slots_by_block_pos == {0: 0, 1: 1, 2: 2}
    assert coordinator.stats()["host_used"] == 3
    with pytest.raises(ValueError, match="already reserved"):
        coordinator.reserve_request(req_pool_idx=7, num_prompt_blocks=1)
    with pytest.raises(MemoryError, match="Insufficient"):
        coordinator.reserve_request(req_pool_idx=8, num_prompt_blocks=2)

    record = coordinator.mark_host_block_committed(7,
                                                  1,
                                                  logical_block_id=1234)
    assert record.valid is True
    assert record.commit_gen == 1
    assert record.logical_block_id == 1234

    coordinator.release_request(7)
    assert coordinator.stats() == {
        "configured": 1,
        "requests": 0,
        "host_used": 0,
        "host_free": 4,
        "hot_used": 0,
    }


def test_hisparse_configure_from_kv_cache_manager_uses_kvarn_shape():
    coordinator = OPTRTHiSparseCoordinator(_cfg())
    coordinator.kv_cache_manager = SimpleNamespace(
        kvarn_cfg=SimpleNamespace(packed_bytes=lambda tokens_per_block: 3072),
        tokens_per_block=64,
        num_local_layers=3,
        blocks_in_primary_pool=16,
        max_batch_size=4,
    )

    tier = coordinator.configure_from_kv_cache_manager()

    assert tier.num_layers == 3
    assert tier.tokens_per_block == 64
    assert tier.packed_bytes_per_block == 3072
    assert tier.hot_device_capacity_blocks == 2
    assert tier.logical_host_capacity_blocks == 64


def test_hisparse_hot_selection_hits_misses_and_lru_eviction():
    coordinator = OPTRTHiSparseCoordinator(_cfg())
    coordinator.configure_packed_tiers(num_layers=1,
                                       tokens_per_block=64,
                                       packed_bytes_per_block=2048,
                                       logical_host_capacity_blocks=8,
                                       hot_device_capacity_blocks=2)
    coordinator.reserve_request(req_pool_idx=3, num_prompt_blocks=4)
    for block_pos in range(4):
        coordinator.mark_host_block_committed(3, block_pos)

    first = coordinator.select_hot_blocks(layer_idx=0,
                                          req_pool_idx=3,
                                          block_positions=[0, 1, 1])
    assert first.block_positions == (0, 1)
    assert first.hits == 0
    assert first.misses == 2
    assert first.hot_slots == (0, 1)

    second = coordinator.select_hot_blocks(layer_idx=0,
                                           req_pool_idx=3,
                                           block_positions=[1])
    assert second.hits == 1
    assert second.misses == 0
    assert second.hot_slots == (1, )

    third = coordinator.select_hot_blocks(layer_idx=0,
                                          req_pool_idx=3,
                                          block_positions=[2])
    assert third.hits == 0
    assert third.misses == 1
    assert third.hot_slots == (0, )

    # Block 1 stayed hot because it was touched after block 0.
    fourth = coordinator.select_hot_blocks(layer_idx=0,
                                           req_pool_idx=3,
                                           block_positions=[1])
    assert fourth.hits == 1
    assert fourth.hot_slots == (1, )


def test_hisparse_hot_selection_rejects_uncommitted_and_respects_commit_gen():
    coordinator = OPTRTHiSparseCoordinator(_cfg())
    coordinator.configure_packed_tiers(num_layers=1,
                                       tokens_per_block=64,
                                       packed_bytes_per_block=2048,
                                       logical_host_capacity_blocks=4,
                                       hot_device_capacity_blocks=2)
    coordinator.reserve_request(req_pool_idx=11, num_prompt_blocks=2)
    with pytest.raises(RuntimeError, match="uncommitted"):
        coordinator.select_hot_blocks(layer_idx=0,
                                      req_pool_idx=11,
                                      block_positions=[0])

    coordinator.mark_host_block_committed(11, 0)
    first = coordinator.select_hot_blocks(layer_idx=0,
                                          req_pool_idx=11,
                                          block_positions=[0])
    assert first.misses == 1

    coordinator.mark_host_block_committed(11, 0)
    second = coordinator.select_hot_blocks(layer_idx=0,
                                           req_pool_idx=11,
                                           block_positions=[0])
    assert second.hits == 0
    assert second.misses == 1

    coordinator.invalidate_host_blocks(11, [0])
    assert coordinator.stats()["hot_used"] == 0

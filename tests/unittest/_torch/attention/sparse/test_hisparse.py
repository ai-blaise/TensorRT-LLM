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


class FakeTensor:

    def __init__(self, shape, dtype, device, pin_memory, base_ptr):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.pin_memory = pin_memory
        self.fill_value = None
        self._base_ptr = int(base_ptr)

    def fill_(self, value):
        self.fill_value = value
        return self

    def data_ptr(self):
        return self._base_ptr

    def element_size(self):
        dtype = str(self.dtype)
        if "int64" in dtype:
            return 8
        return 1

    @property
    def nbytes(self):
        numel = 1
        for dim in self.shape:
            numel *= dim
        return numel * self.element_size()


def _fake_tensor_factory(calls):
    next_ptr = {"value": 0x100000}

    def factory(shape, *, dtype, device, pin_memory=False):
        tensor = FakeTensor(shape, dtype, device, pin_memory,
                            next_ptr["value"])
        next_ptr["value"] += tensor.nbytes + 0x1000
        calls.append(tensor)
        return tensor
    return factory


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
    with pytest.raises(NotImplementedError, match="tensors are not allocated"):
        coordinator.assert_startup_ready()
    calls = []
    coordinator.allocate_packed_tensors(device="cuda:0",
                                        host_pinned=True,
                                        tensor_factory=_fake_tensor_factory(calls))
    with pytest.raises(NotImplementedError, match="swap-in kernel"):
        coordinator.assert_startup_ready()
    with pytest.raises(NotImplementedError):
        coordinator.map_topk_to_hot_pool(topk_indices=None,
                                         metadata=SimpleNamespace(request_ids=[0]),
                                         layer_idx=0,
                                         skip_topk=False,
                                         is_generation=True)

    coordinator._torch_cuda_op_registered = (  # noqa: SLF001
        lambda name: name == "trtllm::hisparse_topk_to_block_positions")
    with pytest.raises(NotImplementedError,
                       match="resolve_blocks_to_host_slots"):
        coordinator.map_topk_to_hot_pool(topk_indices=object(),
                                         metadata=SimpleNamespace(request_ids=[0]),
                                         layer_idx=0,
                                         skip_topk=False,
                                         is_generation=True)

    coordinator._torch_cuda_op_registered = (  # noqa: SLF001
        lambda name: name in {
            "trtllm::hisparse_topk_to_block_positions",
            "trtllm::hisparse_resolve_blocks_to_host_slots",
        })
    with pytest.raises(NotImplementedError,
                       match="plan_hot_slots"):
        coordinator.map_topk_to_hot_pool(topk_indices=object(),
                                         metadata=SimpleNamespace(request_ids=[0]),
                                         layer_idx=0,
                                         skip_topk=False,
                                         is_generation=True)

    coordinator._torch_cuda_op_registered = (  # noqa: SLF001
        lambda name: name in {
            "trtllm::hisparse_topk_to_block_positions",
            "trtllm::hisparse_resolve_blocks_to_host_slots",
            "trtllm::hisparse_plan_hot_slots",
        })
    with pytest.raises(NotImplementedError,
                       match="compact_miss_schedule"):
        coordinator.map_topk_to_hot_pool(topk_indices=object(),
                                         metadata=SimpleNamespace(request_ids=[0]),
                                         layer_idx=0,
                                         skip_topk=False,
                                         is_generation=True)

    coordinator._torch_cuda_op_registered = (  # noqa: SLF001
        lambda name: name in {
            "trtllm::hisparse_topk_to_block_positions",
            "trtllm::hisparse_resolve_blocks_to_host_slots",
            "trtllm::hisparse_plan_hot_slots",
            "trtllm::hisparse_compact_miss_schedule",
        })
    with pytest.raises(NotImplementedError,
                       match="swap_in_packed_kvarn"):
        coordinator.map_topk_to_hot_pool(topk_indices=object(),
                                         metadata=SimpleNamespace(request_ids=[0]),
                                         layer_idx=0,
                                         skip_topk=False,
                                         is_generation=True)

    coordinator._torch_cuda_op_registered = (  # noqa: SLF001
        lambda name: name in {
            "trtllm::hisparse_topk_to_block_positions",
            "trtllm::hisparse_resolve_blocks_to_host_slots",
            "trtllm::hisparse_plan_hot_slots",
            "trtllm::hisparse_compact_miss_schedule",
            "trtllm::hisparse_swap_in_packed_kvarn",
        })
    with pytest.raises(NotImplementedError,
                       match="commit_hot_slots"):
        coordinator.map_topk_to_hot_pool(topk_indices=object(),
                                         metadata=SimpleNamespace(request_ids=[0]),
                                         layer_idx=0,
                                         skip_topk=False,
                                         is_generation=True)

    coordinator._torch_cuda_op_registered = (  # noqa: SLF001
        lambda name: name in {
            "trtllm::hisparse_topk_to_block_positions",
            "trtllm::hisparse_resolve_blocks_to_host_slots",
            "trtllm::hisparse_plan_hot_slots",
            "trtllm::hisparse_compact_miss_schedule",
            "trtllm::hisparse_swap_in_packed_kvarn",
            "trtllm::hisparse_commit_hot_slots",
        })
    with pytest.raises(NotImplementedError,
                       match="build_hot_indices"):
        coordinator.map_topk_to_hot_pool(topk_indices=object(),
                                         metadata=SimpleNamespace(request_ids=[0]),
                                         layer_idx=0,
                                         skip_topk=False,
                                         is_generation=True)

    coordinator._torch_cuda_op_registered = (  # noqa: SLF001
        lambda name: name in {
            "trtllm::hisparse_topk_to_block_positions",
            "trtllm::hisparse_resolve_blocks_to_host_slots",
            "trtllm::hisparse_plan_hot_slots",
            "trtllm::hisparse_compact_miss_schedule",
            "trtllm::hisparse_swap_in_packed_kvarn",
            "trtllm::hisparse_commit_hot_slots",
            "trtllm::hisparse_build_hot_indices",
        })
    with pytest.raises(NotImplementedError,
                       match="native orchestration"):
        coordinator.map_topk_to_hot_pool(topk_indices=object(),
                                         metadata=SimpleNamespace(request_ids=[0]),
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
        "tensors_allocated": 0,
    }


def test_hisparse_host_write_commit_waits_for_all_local_layers():
    coordinator = OPTRTHiSparseCoordinator(_cfg())
    coordinator.configure_packed_tiers(num_layers=2,
                                       tokens_per_block=64,
                                       packed_bytes_per_block=2048,
                                       logical_host_capacity_blocks=4,
                                       hot_device_capacity_blocks=2)
    state = coordinator.reserve_request(req_pool_idx=7, num_prompt_blocks=2)
    coordinator.begin_host_write(7)
    assert state.pending_writes == 1
    assert coordinator.request_ready_for_admission(7) is False
    with pytest.raises(RuntimeError, match="pending host writes"):
        coordinator.release_request(7)

    newly_full = coordinator.mark_host_write_committed(
        7,
        layer_indices=[0],
        block_positions=[1],
    )

    assert newly_full == ()
    assert coordinator.host_block_committed(7, 1) is False
    with pytest.raises(RuntimeError, match="uncommitted"):
        coordinator.select_hot_blocks(layer_idx=0,
                                      req_pool_idx=7,
                                      block_positions=[1])
    coordinator.finish_host_write(7)
    assert state.pending_writes == 0
    assert coordinator.request_ready_for_admission(7) is False
    with pytest.raises(RuntimeError, match="cannot be admitted"):
        coordinator.mark_request_admitted(7)

    newly_full = coordinator.mark_host_write_committed(
        7,
        layer_indices=[0, 1, 1],
        block_positions=[0, 0, 1],
    )

    assert len(newly_full) == 2
    assert [record.block_pos for record in newly_full] == [0, 1]
    assert all(record.valid for record in newly_full)
    assert all(record.commit_gen == 1 for record in newly_full)
    assert coordinator.host_block_committed(7, 1) is True

    duplicate = coordinator.mark_host_write_committed(
        7,
        layer_indices=[0, 1],
        block_positions=[1, 1],
    )

    assert duplicate == ()
    assert newly_full[0].commit_gen == 1
    assert coordinator.request_ready_for_admission(7) is True
    admitted = coordinator.mark_request_admitted(7)
    assert admitted.admitted is True


def test_hisparse_request_reservation_is_idempotent_for_published_slots():
    coordinator = OPTRTHiSparseCoordinator(_cfg())
    coordinator.configure_packed_tiers(num_layers=1,
                                       tokens_per_block=64,
                                       packed_bytes_per_block=1024,
                                       logical_host_capacity_blocks=6,
                                       hot_device_capacity_blocks=2)

    first = coordinator.reserve_or_get_request(req_pool_idx=17,
                                               num_prompt_blocks=3)
    second = coordinator.reserve_or_get_request(req_pool_idx=17,
                                                num_prompt_blocks=3)

    assert second is first
    assert coordinator.host_slots_for_request(17) == (0, 1, 2)
    assert coordinator.host_slots_for_request(17,
                                              num_prompt_blocks=2) == (0, 1)
    with pytest.raises(RuntimeError, match="already has 3"):
        coordinator.reserve_or_get_request(req_pool_idx=17,
                                           num_prompt_blocks=4)


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
    assert tier.request_slot_capacity == 4
    assert tier.max_blocks_per_request == 16


def test_hisparse_allocate_packed_tensors_shapes_and_initializers():
    coordinator = OPTRTHiSparseCoordinator(_cfg())
    coordinator.configure_packed_tiers(num_layers=2,
                                       tokens_per_block=64,
                                       packed_bytes_per_block=3072,
                                       logical_host_capacity_blocks=5,
                                       hot_device_capacity_blocks=3)
    calls = []

    tensors = coordinator.allocate_packed_tensors(
        device="cuda:1",
        host_pinned=True,
        tensor_factory=_fake_tensor_factory(calls),
    )

    assert coordinator.packed_tensors_allocated is True
    assert coordinator.tensors is tensors
    assert tensors.host_packed.shape == (2, 5, 3072)
    assert tensors.hot_packed.shape == (2, 3, 3072)
    assert tensors.host_valid.shape == (2, 5)
    assert tensors.host_commit_gen.shape == (2, 5)
    assert tensors.hot_host_slot.shape == (2, 3)
    assert tensors.hot_commit_gen.shape == (2, 3)
    assert tensors.hot_lru_tick.shape == (2, 3)
    assert tensors.request_ids_host.shape == (5, )
    assert tensors.request_ids_device.shape == (5, )
    assert tensors.request_block_host_slots_host.shape == (5, 5)
    assert tensors.request_block_host_slots_device.shape == (5, 5)
    assert tensors.request_block_commit_gen_host.shape == (5, 5)
    assert tensors.request_block_commit_gen_device.shape == (5, 5)
    assert tensors.request_admitted_host.shape == (5, )
    assert tensors.request_admitted_device.shape == (5, )
    assert tensors.host_packed.pin_memory is True
    assert tensors.hot_packed.pin_memory is False
    assert tensors.hot_packed.device == "cuda:1"
    assert [tensor.fill_value for tensor in calls] == [
        0, 0, 0, 0, -1, -1, 0, -1, -1, -1, -1, -1, -1, 0, 0
    ]
    assert coordinator.stats()["tensors_allocated"] == 1


def test_hisparse_request_table_slots_are_stable_and_reused():
    coordinator = OPTRTHiSparseCoordinator(_cfg())
    coordinator.configure_packed_tiers(num_layers=1,
                                       tokens_per_block=64,
                                       packed_bytes_per_block=1024,
                                       logical_host_capacity_blocks=6,
                                       hot_device_capacity_blocks=2,
                                       request_slot_capacity=2,
                                       max_blocks_per_request=3)

    first = coordinator.reserve_request(req_pool_idx=101, num_prompt_blocks=3)
    second = coordinator.reserve_request(req_pool_idx=202, num_prompt_blocks=1)

    assert first.table_slot == 0
    assert second.table_slot == 1
    assert coordinator.request_table_snapshot(101) == {
        "table_slot": 0,
        "req_pool_idx": 101,
        "host_slots_by_block_pos": {
            0: 0,
            1: 1,
            2: 2,
        },
        "admitted": False,
    }
    with pytest.raises(MemoryError, match="request table slots"):
        coordinator.reserve_request(req_pool_idx=303, num_prompt_blocks=1)
    with pytest.raises(MemoryError, match="request table width"):
        coordinator.reserve_or_get_request(req_pool_idx=404,
                                           num_prompt_blocks=4)

    coordinator.mark_host_block_committed(101, 0)
    coordinator.mark_host_block_committed(101, 1)
    coordinator.mark_host_block_committed(101, 2)
    coordinator.mark_request_admitted(101)
    assert coordinator.request_table_snapshot(101)["admitted"] is True

    coordinator.release_request(101)
    reused = coordinator.reserve_request(req_pool_idx=303,
                                         num_prompt_blocks=1)

    assert reused.table_slot == 0


def test_hisparse_host_registration_descs_and_slot_ptrs():
    coordinator = OPTRTHiSparseCoordinator(_cfg())
    coordinator.configure_packed_tiers(num_layers=2,
                                       tokens_per_block=64,
                                       packed_bytes_per_block=16,
                                       logical_host_capacity_blocks=4,
                                       hot_device_capacity_blocks=2)
    calls = []
    tensors = coordinator.allocate_packed_tensors(
        device="cuda:1",
        host_pinned=True,
        tensor_factory=_fake_tensor_factory(calls),
    )

    descs = coordinator.host_registration_descs()

    host_packed_layer_bytes = 4 * 16
    assert descs[:2] == [
        (tensors.host_packed.data_ptr(), host_packed_layer_bytes, 0,
         "hisparse_host.host_packed.layer0"),
        (tensors.host_packed.data_ptr() + host_packed_layer_bytes,
         host_packed_layer_bytes, 0, "hisparse_host.host_packed.layer1"),
    ]
    assert len(descs) == 6

    coordinator.reserve_request(req_pool_idx=5, num_prompt_blocks=3)
    ptrs, sizes = coordinator.host_packed_ptrs_for_blocks(
        layer_idx=1,
        req_pool_idx=5,
        block_positions=[0, 2],
    )

    layer_one_base = tensors.host_packed.data_ptr() + host_packed_layer_bytes
    assert ptrs.tolist() == [layer_one_base, layer_one_base + 2 * 16]
    assert sizes.tolist() == [16, 16]


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


def test_hisparse_hot_planning_is_non_mutating_until_commit():
    coordinator = OPTRTHiSparseCoordinator(_cfg())
    coordinator.configure_packed_tiers(num_layers=1,
                                       tokens_per_block=64,
                                       packed_bytes_per_block=2048,
                                       logical_host_capacity_blocks=8,
                                       hot_device_capacity_blocks=2)
    coordinator.reserve_request(req_pool_idx=13, num_prompt_blocks=2)
    for block_pos in range(2):
        coordinator.mark_host_block_committed(13, block_pos)
    coordinator.mark_request_admitted(13)

    planned = coordinator.plan_hot_blocks(layer_idx=0,
                                          req_pool_idx=13,
                                          block_positions=[0, 1],
                                          require_admitted=True)

    assert planned.misses == 2
    assert planned.miss_block_positions == (0, 1)
    assert planned.miss_host_slots == (0, 1)
    assert planned.miss_hot_slots == (0, 1)
    assert coordinator.stats()["hot_used"] == 0

    committed = coordinator.commit_hot_selection(planned)

    assert committed is planned
    assert coordinator.stats()["hot_used"] == 2


def test_hisparse_hot_planning_requires_admission_for_production_path():
    coordinator = OPTRTHiSparseCoordinator(_cfg())
    coordinator.configure_packed_tiers(num_layers=1,
                                       tokens_per_block=64,
                                       packed_bytes_per_block=2048,
                                       logical_host_capacity_blocks=4,
                                       hot_device_capacity_blocks=2)
    coordinator.reserve_request(req_pool_idx=23, num_prompt_blocks=1)
    coordinator.mark_host_block_committed(23, 0)

    with pytest.raises(RuntimeError, match="not been admitted"):
        coordinator.plan_hot_blocks(layer_idx=0,
                                    req_pool_idx=23,
                                    block_positions=[0],
                                    require_admitted=True)


def test_hisparse_swap_in_plan_dedupes_tokens_and_builds_packed_pointers():
    coordinator = OPTRTHiSparseCoordinator(_cfg())
    coordinator.configure_packed_tiers(num_layers=1,
                                       tokens_per_block=64,
                                       packed_bytes_per_block=16,
                                       logical_host_capacity_blocks=4,
                                       hot_device_capacity_blocks=4)
    calls = []
    tensors = coordinator.allocate_packed_tensors(
        device="cuda:1",
        host_pinned=True,
        tensor_factory=_fake_tensor_factory(calls),
    )
    coordinator.reserve_request(req_pool_idx=31, num_prompt_blocks=3)
    for block_pos in range(3):
        coordinator.mark_host_block_committed(31, block_pos)
    coordinator.mark_request_admitted(31)

    plan = coordinator.plan_swap_in_for_token_positions(
        layer_idx=0,
        req_pool_idx=31,
        token_positions=[0, 63, 64, 130],
        require_admitted=True,
    )

    assert plan.selection.block_positions == (0, 1, 2)
    assert plan.selection.misses == 3
    assert plan.selection.miss_hot_slots == (0, 1, 2)
    assert plan.host_ptrs.tolist() == [
        tensors.host_packed.data_ptr(),
        tensors.host_packed.data_ptr() + 16,
        tensors.host_packed.data_ptr() + 32,
    ]
    assert plan.hot_ptrs.tolist() == [
        tensors.hot_packed.data_ptr(),
        tensors.hot_packed.data_ptr() + 16,
        tensors.hot_packed.data_ptr() + 32,
    ]
    assert plan.sizes.tolist() == [16, 16, 16]
    assert plan.has_misses is True
    assert coordinator.stats()["hot_used"] == 0


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

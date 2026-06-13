from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.hisparse import (
    HiSparseResidentTokenDescriptor,
    HiSparseSparseMlaKvarnHotDescriptor,
    OPTRTHiSparseCoordinator)
from tensorrt_llm._torch.attention_backend.sparse.kvarn_backend import (
    KVARN_BDR_HISPARSE_LAYOUT, KVARN_LEGACY_SIDEPOOL_LAYOUT)


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
    with pytest.raises(NotImplementedError, match="Missing CUDA op"):
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
                       match="classify_resident_blocks"):
        coordinator.map_topk_to_hot_pool(topk_indices=object(),
                                         metadata=SimpleNamespace(request_ids=[0]),
                                         layer_idx=0,
                                         skip_topk=False,
                                         is_generation=True)

    coordinator._torch_cuda_op_registered = (  # noqa: SLF001
        lambda name: name in {
            "trtllm::hisparse_topk_to_block_positions",
            "trtllm::hisparse_resolve_blocks_to_host_slots",
            "trtllm::hisparse_classify_resident_blocks",
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
            "trtllm::hisparse_classify_resident_blocks",
            "trtllm::hisparse_plan_hot_slots",
            "trtllm::hisparse_compact_miss_schedule",
        })
    with pytest.raises(NotImplementedError,
                       match="submit_packed_kvarn_copy_schedule"):
        coordinator.map_topk_to_hot_pool(topk_indices=object(),
                                         metadata=SimpleNamespace(request_ids=[0]),
                                         layer_idx=0,
                                         skip_topk=False,
                                         is_generation=True)

    coordinator._torch_cuda_op_registered = (  # noqa: SLF001
        lambda name: name in {
            "trtllm::hisparse_topk_to_block_positions",
            "trtllm::hisparse_resolve_blocks_to_host_slots",
            "trtllm::hisparse_classify_resident_blocks",
            "trtllm::hisparse_plan_hot_slots",
            "trtllm::hisparse_compact_miss_schedule",
            "trtllm::hisparse_submit_packed_kvarn_copy_schedule",
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
            "trtllm::hisparse_classify_resident_blocks",
            "trtllm::hisparse_plan_hot_slots",
            "trtllm::hisparse_compact_miss_schedule",
            "trtllm::hisparse_submit_packed_kvarn_copy_schedule",
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
            "trtllm::hisparse_classify_resident_blocks",
            "trtllm::hisparse_plan_hot_slots",
            "trtllm::hisparse_compact_miss_schedule",
            "trtllm::hisparse_submit_packed_kvarn_copy_schedule",
            "trtllm::hisparse_commit_hot_slots",
            "trtllm::hisparse_build_hot_indices",
            "trtllm::hisparse_read_kvarn_hot_bdr",
            "trtllm::sparse_mla_decode_kvarn_hot",
        })
    with pytest.raises(NotImplementedError,
                       match="sink/tail resident-token ABI"):
        coordinator.map_topk_to_hot_pool(topk_indices=object(),
                                         metadata=SimpleNamespace(request_ids=[0]),
                                         layer_idx=0,
                                         skip_topk=False,
                                         is_generation=True)


def test_hisparse_sparse_mla_readiness_ladder(monkeypatch):
    coordinator = OPTRTHiSparseCoordinator(_cfg(enabled=True))
    coordinator.configure_packed_tiers(num_layers=1,
                                       tokens_per_block=64,
                                       packed_bytes_per_block=2048,
                                       logical_host_capacity_blocks=4,
                                       hot_device_capacity_blocks=2)
    calls = []
    coordinator.allocate_packed_tensors(device="cuda:0",
                                        host_pinned=True,
                                        tensor_factory=_fake_tensor_factory(calls))

    planner_ops = {
        "trtllm::hisparse_publish_request_table_slots",
        "trtllm::hisparse_topk_to_block_positions",
        "trtllm::hisparse_resolve_blocks_to_host_slots",
        "trtllm::hisparse_classify_resident_blocks",
        "trtllm::hisparse_plan_hot_slots",
        "trtllm::hisparse_compact_miss_schedule",
        "trtllm::hisparse_submit_packed_kvarn_copy_schedule",
        "trtllm::hisparse_commit_hot_slots",
        "trtllm::hisparse_build_hot_indices",
    }
    hot_reader = "trtllm::hisparse_read_kvarn_hot_bdr"
    sparse_mla = "trtllm::sparse_mla_decode_kvarn_hot"

    monkeypatch.setattr(coordinator, "_torch_cuda_op_registered",
                        lambda name: name in planner_ops)
    monkeypatch.setattr(coordinator, "_torch_bool_op_ready",
                        lambda name: False)
    with pytest.raises(NotImplementedError,
                       match="KVarN-hot BDR reader primitive"):
        coordinator.assert_startup_ready()

    monkeypatch.setattr(coordinator, "_torch_cuda_op_registered",
                        lambda name: name in planner_ops | {hot_reader})
    with pytest.raises(NotImplementedError,
                       match="sparse_mla_decode_kvarn_hot"):
        coordinator.assert_startup_ready()

    monkeypatch.setattr(coordinator, "_torch_cuda_op_registered",
                        lambda name: name in planner_ops | {
                            hot_reader,
                            sparse_mla,
                        })
    with pytest.raises(NotImplementedError,
                       match="sink/tail resident-token ABI"):
        coordinator.assert_startup_ready()

    monkeypatch.setattr(coordinator, "_torch_bool_op_ready",
                        lambda name: name == "hisparse_sparse_mla_resident_v1_ready")
    coordinator.assert_startup_ready()


def test_hisparse_resident_token_readiness_is_hard_startup_gate(monkeypatch):
    coordinator = OPTRTHiSparseCoordinator(_cfg(enabled=True))
    coordinator.configure_packed_tiers(num_layers=1,
                                       tokens_per_block=64,
                                       packed_bytes_per_block=2048,
                                       logical_host_capacity_blocks=4,
                                       hot_device_capacity_blocks=2,
                                       kvarn_bits=2)
    calls = []
    coordinator.allocate_packed_tensors(device="cuda:0",
                                        host_pinned=True,
                                        tensor_factory=_fake_tensor_factory(calls))
    monkeypatch.setattr(coordinator, "_torch_cuda_op_registered",
                        lambda name: name in {
                            "trtllm::hisparse_publish_request_table_slots",
                            "trtllm::hisparse_topk_to_block_positions",
                            "trtllm::hisparse_resolve_blocks_to_host_slots",
                            "trtllm::hisparse_classify_resident_blocks",
                            "trtllm::hisparse_plan_hot_slots",
                            "trtllm::hisparse_compact_miss_schedule",
                            "trtllm::hisparse_submit_packed_kvarn_copy_schedule",
                            "trtllm::hisparse_commit_hot_slots",
                            "trtllm::hisparse_build_hot_indices",
                            "trtllm::hisparse_read_kvarn_hot_bdr",
                            "trtllm::sparse_mla_decode_kvarn_hot",
                        })
    monkeypatch.setattr(coordinator, "_torch_bool_op_ready",
                        lambda name: False)

    with pytest.raises(NotImplementedError,
                       match="explicit sink/tail resident-token ABI"):
        coordinator.assert_startup_ready()

    monkeypatch.setattr(coordinator, "_torch_bool_op_ready",
                        lambda name: name == "hisparse_sparse_mla_resident_v1_ready")
    coordinator.assert_startup_ready()


def test_hisparse_sparse_mla_descriptor_is_production_k2v2_contract():
    desc = HiSparseSparseMlaKvarnHotDescriptor(
        hot_packed=object(),
        hot_indices=object(),
        row_status=object(),
        request_topk_indices=object(),
        resident_block_flags=object(),
        resident_block_status=object(),
        topk_length=None,
        layer_idx=3,
        index_topk=1024,
        max_blocks_per_row=64,
        tokens_per_block=64,
        stride_factor=5120,
        packed_bytes_per_block=126976,
        hot_capacity_blocks=128,
    )

    assert desc.kvarn_bits == 2
    assert desc.kv_lora_rank == 512
    assert desc.qk_rope_head_dim == 64
    assert desc.resident_token_policy == "explicit_sink_tail_v1"
    assert desc.resident_tokens is None
    assert desc.request_topk_indices is not None
    assert desc.resident_block_flags is not None
    assert desc.resident_block_status is not None
    assert desc.topk_length is None
    assert desc.step_id == -1


def test_hisparse_resident_token_descriptor_derives_sink_tail_geometry():
    coordinator = OPTRTHiSparseCoordinator(
        _cfg(),
        kv_cache_manager=SimpleNamespace(kvarn_cfg=SimpleNamespace(
            sink_tokens=128)))
    coordinator.configure_packed_tiers(num_layers=1,
                                       tokens_per_block=64,
                                       packed_bytes_per_block=126976,
                                       logical_host_capacity_blocks=4,
                                       hot_device_capacity_blocks=2,
                                       kvarn_bits=2)
    metadata = SimpleNamespace(
        kv_lens_cuda_runtime=torch.tensor([512, 130, 192, 257],
                                          dtype=torch.int32),
        num_contexts=1,
        num_generations=3,
        _cached_pool_view=torch.empty((32, 1, 576), dtype=torch.bfloat16),
        _cached_block_table_gen=torch.arange(12,
                                             dtype=torch.int32).view(3, 4),
    )
    req_idx = torch.tensor([0, 1, 1, 2], dtype=torch.int64)
    row_request_ids = torch.tensor([7001, 7002, 7002, 7003],
                                   dtype=torch.int64)

    desc = coordinator._make_resident_token_descriptor(  # noqa: SLF001
        metadata=metadata,
        req_idx=req_idx,
        row_request_ids=row_request_ids,
        is_generation=True,
    )

    assert isinstance(desc, HiSparseResidentTokenDescriptor)
    assert desc.policy == "explicit_sink_tail_v1"
    assert desc.source == "normal_decode_kv"
    assert desc.sink_tokens == 128
    assert desc.sink_blocks == 2
    assert desc.tokens_per_block == 64
    assert desc.kv_pool is metadata._cached_pool_view
    assert desc.block_table is metadata._cached_block_table_gen
    assert desc.row_kv_lens.tolist() == [130, 192, 192, 257]
    assert desc.row_req_idx.tolist() == [0, 1, 1, 2]
    assert desc.row_request_ids.tolist() == [7001, 7002, 7002, 7003]
    assert desc.tail_block_pos.tolist() == [2, 3, 3, 4]
    assert desc.tail_token_count.tolist() == [2, 0, 0, 1]
    assert desc.tail_valid.tolist() == [True, False, False, True]


def test_hisparse_sparse_mla_descriptor_records_coordinator_step():
    coordinator = OPTRTHiSparseCoordinator(_cfg())
    coordinator.configure_packed_tiers(num_layers=1,
                                       tokens_per_block=64,
                                       packed_bytes_per_block=126976,
                                       logical_host_capacity_blocks=4,
                                       hot_device_capacity_blocks=2,
                                       kvarn_bits=2)
    coordinator.allocate_packed_tensors(device="cpu", host_pinned=False)
    coordinator.reset_step()

    desc = coordinator._make_sparse_mla_kvarn_hot_descriptor(  # noqa: SLF001
        hot_indices=object(),
        row_status=object(),
        layer_idx=0,
        index_topk=1024,
        max_blocks_per_row=2,
        stride_factor=64,
    )

    assert desc.step_id == coordinator.step_id
    assert desc.resident_token_policy == "explicit_sink_tail_v1"


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
    layout = SimpleNamespace(name=KVARN_BDR_HISPARSE_LAYOUT,
                             packed_bytes_per_block=13312,
                             ckv_bits=2)
    coordinator.kv_cache_manager = SimpleNamespace(
        kvarn_cfg=SimpleNamespace(
            hisparse_bdr_layout=lambda tokens_per_block: layout),
        kvarn_hisparse_source_layout=KVARN_BDR_HISPARSE_LAYOUT,
        tokens_per_block=64,
        num_local_layers=3,
        blocks_in_primary_pool=16,
        max_batch_size=4,
    )

    tier = coordinator.configure_from_kv_cache_manager()

    assert tier.num_layers == 3
    assert tier.tokens_per_block == 64
    assert tier.packed_bytes_per_block == 13312
    assert tier.packed_layout == KVARN_BDR_HISPARSE_LAYOUT
    assert tier.kvarn_bits == 2
    assert tier.hot_device_capacity_blocks == 2
    assert tier.logical_host_capacity_blocks == 64
    assert tier.request_slot_capacity == 4
    assert tier.max_blocks_per_request == 16


def test_hisparse_configure_from_kv_cache_manager_rejects_legacy_kvarn_layout():
    coordinator = OPTRTHiSparseCoordinator(_cfg())
    layout = SimpleNamespace(name=KVARN_BDR_HISPARSE_LAYOUT,
                             packed_bytes_per_block=13312,
                             ckv_bits=2)
    coordinator.kv_cache_manager = SimpleNamespace(
        kvarn_cfg=SimpleNamespace(
            hisparse_bdr_layout=lambda tokens_per_block: layout),
        kvarn_hisparse_source_layout=KVARN_LEGACY_SIDEPOOL_LAYOUT,
        tokens_per_block=64,
        num_local_layers=3,
        blocks_in_primary_pool=16,
        max_batch_size=4,
    )

    with pytest.raises(NotImplementedError, match="production BDR KVarN"):
        coordinator.configure_from_kv_cache_manager()


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


def test_hisparse_request_table_syncs_host_rows_to_device_rows_cpu():
    coordinator = OPTRTHiSparseCoordinator(_cfg())
    coordinator.configure_packed_tiers(num_layers=2,
                                       tokens_per_block=64,
                                       packed_bytes_per_block=1024,
                                       logical_host_capacity_blocks=6,
                                       hot_device_capacity_blocks=2,
                                       request_slot_capacity=2,
                                       max_blocks_per_request=4)
    tensors = coordinator.allocate_packed_tensors(device="cpu",
                                                  host_pinned=False)

    state = coordinator.reserve_request(req_pool_idx=101, num_prompt_blocks=3)
    slot = state.table_slot

    assert int(tensors.request_ids_host[slot]) == 101
    assert int(tensors.request_ids_device[slot]) == 101
    assert tensors.request_block_host_slots_host[slot].tolist() == [0, 1, 2, -1]
    assert tensors.request_block_host_slots_device[slot].tolist() == [
        0, 1, 2, -1
    ]
    assert torch.equal(tensors.request_block_commit_gen_device[slot],
                       tensors.request_block_commit_gen_host[slot])

    coordinator.mark_host_block_committed(101, 1)
    assert int(tensors.request_block_commit_gen_host[slot, 1]) == 1
    assert int(tensors.request_block_commit_gen_device[slot, 1]) == 1
    assert bool(tensors.request_admitted_device[slot]) is False

    coordinator.mark_host_block_committed(101, 0)
    coordinator.mark_host_block_committed(101, 2)
    coordinator.mark_request_admitted(101)
    assert bool(tensors.request_admitted_host[slot]) is True
    assert bool(tensors.request_admitted_device[slot]) is True

    coordinator.release_request(101)
    assert int(tensors.request_ids_device[slot]) == -1
    assert bool(tensors.request_admitted_device[slot]) is False
    assert tensors.request_block_host_slots_device[slot].tolist() == [-1] * 4
    assert tensors.request_block_commit_gen_device[slot].tolist() == [-1] * 4


def test_hisparse_enabled_cuda_request_table_publish_requires_native_op(
        monkeypatch):
    coordinator = OPTRTHiSparseCoordinator(_cfg(enabled=True))
    coordinator.configure_packed_tiers(num_layers=1,
                                       tokens_per_block=64,
                                       packed_bytes_per_block=1024,
                                       logical_host_capacity_blocks=2,
                                       hot_device_capacity_blocks=1,
                                       request_slot_capacity=1,
                                       max_blocks_per_request=2)
    calls = []
    coordinator.allocate_packed_tensors(device="cuda:0",
                                        host_pinned=True,
                                        tensor_factory=_fake_tensor_factory(calls))
    monkeypatch.setattr(coordinator, "_request_table_device_is_cuda",
                        lambda: True)
    monkeypatch.setattr(coordinator, "_torch_cuda_op_registered",
                        lambda name: False)

    with pytest.raises(NotImplementedError,
                       match="hisparse_publish_request_table_slots"):
        coordinator.reserve_request(req_pool_idx=101, num_prompt_blocks=1)


def test_hisparse_request_table_native_publisher_is_registered_in_sources():
    root = Path(__file__).resolve().parents[5]
    thop = root / "cpp/tensorrt_llm/thop/hisparseRequestTableOp.cpp"
    cmake = root / "cpp/tensorrt_llm/thop/CMakeLists.txt"
    fake = root / "tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py"

    thop_source = thop.read_text()
    assert "hisparse_publish_request_table_slots" in thop_source
    assert "cudaMemcpyAsync" in thop_source
    assert "request_ids_host" in thop_source
    assert "request_block_commit_gen_device" in thop_source
    assert "request_admitted_device" in thop_source
    assert "hisparseRequestTableOp.cpp" in cmake.read_text()
    assert "hisparse_publish_request_table_slots" in fake.read_text()


def test_hisparse_kvarn_hot_bdr_reader_is_registered_in_sources():
    root = Path(__file__).resolve().parents[5]
    device_header = root / "cpp/tensorrt_llm/kernels/hisparseKvarnBdrRead.cuh"
    kernel = root / "cpp/tensorrt_llm/kernels/hisparseKvarnHotRead.cu"
    header = root / "cpp/tensorrt_llm/kernels/hisparseKvarnHotRead.h"
    thop = root / "cpp/tensorrt_llm/thop/hisparseKvarnHotReadOp.cpp"
    cmake = root / "cpp/tensorrt_llm/thop/CMakeLists.txt"
    fake = root / "tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py"

    device_header_source = device_header.read_text()
    kernel_source = kernel.read_text()
    thop_source = thop.read_text()
    assert "hisparseKvarnBdrRead.cuh" in kernel_source
    assert "decodeHisparseKvarnHotIndex" in device_header_source
    assert "readHisparseKvarnK2v2PackedCkvValue" in device_header_source
    assert "readHisparseKvarnK2v2BdrLatentValue" in device_header_source
    assert "hisparseKvarnK2v2BdrRecordBytes" in device_header_source
    assert "readLowBitKvarnValue" not in kernel_source
    assert "readFp8E4m3Byte" not in kernel_source
    assert "kvarnBits == 2" in kernel_source
    assert "kvarnBits == 2 || kvarnBits == 4" not in kernel_source
    assert "tokensPerBlock == 64" in kernel_source
    assert "kvLoraRank == 512" in thop_source
    assert "qkRopeHeadDim == 64" in thop_source
    assert "hisparse_read_kvarn_hot_bdr" in header.read_text()
    assert "hisparseKvarnHotReadOp.cpp" in cmake.read_text()
    assert "hisparse_read_kvarn_hot_bdr" in fake.read_text()


def test_hisparse_sparse_mla_kvarn_hot_op_is_registered_in_sources():
    root = Path(__file__).resolve().parents[5]
    kernel = root / "cpp/tensorrt_llm/kernels/flashMLA/sparse_mla_decode_kvarn_hot.cu"
    header = root / "cpp/tensorrt_llm/kernels/flashMLA/sparse_mla_decode_kvarn_hot.h"
    planner_kernel = root / "cpp/tensorrt_llm/kernels/hisparseTopkToBlocks.cu"
    thop = root / "cpp/tensorrt_llm/thop/SparseMlaDecodeKvarnHotOp.cpp"
    planner_thop = root / "cpp/tensorrt_llm/thop/hisparseTopkToBlocksOp.cpp"
    flash_cmake = root / "cpp/tensorrt_llm/kernels/flashMLA/CMakeLists.txt"
    thop_cmake = root / "cpp/tensorrt_llm/thop/CMakeLists.txt"
    fake = root / "tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py"
    docs = root / "docs/blaise/hisparse_optrt_plan.md"

    kernel_source = kernel.read_text()
    planner_source = planner_kernel.read_text()
    thop_source = thop.read_text()
    planner_thop_source = planner_thop.read_text()
    docs_source = docs.read_text()
    assert "hisparseKvarnBdrRead.cuh" in kernel_source
    assert "hisparseClassifyResidentBlocksKernel" in planner_source
    assert "kResidentBlockSink" in planner_source
    assert "kResidentBlockTail" in planner_source
    assert "kResidentClassBlockPastKvLen" in planner_source
    assert "kResolveResidentInvalid" in planner_source
    assert "kPlanInvalidResidentFlag" in planner_source
    assert "kBuildHotIndexInvalidResidentFlag" in planner_source
    assert "hisparseIsResidentBlockFlag" in planner_source
    assert "residentBlockFlags" in planner_source
    assert "plannedHotSlots[rowOffset + i] = -1" in planner_source
    assert "readHisparseKvarnK2v2BdrLatentValue" in kernel_source
    assert "decodeResidentTokenAddress" in kernel_source
    assert "readResidentLatentValue" in kernel_source
    assert "requestTopkIndices" in kernel_source
    assert "residentKvPoolDtype" in kernel_source
    assert "residentBlockTableRows" in kernel_source
    assert "residentKvPoolTokens" in kernel_source
    assert "hotIndex < 0" in kernel_source
    assert "resident normal-KV reads" in docs_source
    assert "decodeHisparseKvarnHotIndex" in kernel_source
    assert "atomicCAS(&rowCode" in kernel_source
    assert "__shared__ int32_t valueCode" in kernel_source
    assert "atomicCAS(&valueCode" in kernel_source
    assert "if (valueCode != kHotReadOk)" in kernel_source
    assert "writeBf16(params.out, outBase + dim, 0.0F)" in kernel_source
    assert "head] = kNegInf" in kernel_source
    assert "kvarn_k2v2" in kernel_source
    assert "sparse_mla_decode_kvarn_hot" in header.read_text()
    assert "sparse_mla_decode_kvarn_hot" in thop_source
    assert "kvarn_bits must be 2" in thop_source
    assert "expectedBdrBytesPerBlock" in thop_source
    assert "hotPacked.size(2) >= expectedBytes" in thop_source
    assert "hot_packed record bytes are smaller than production BDR layout" in thop_source
    assert "checkHotPackedStrides" in thop_source
    assert "checkResidentTokenAbi" in thop_source
    assert "hisparse_sparse_mla_resident_v1_ready" in thop_source
    assert "return false" in thop_source
    assert "explicit_sink_tail_v1 requires resident_kv_lens" in thop_source
    assert "and request_topk_indices" in thop_source
    assert "request_topk_indices must be int32" in thop_source
    assert "same [batch, s_q, topk] shape as hot indices" in thop_source
    assert "resident_kv_pool must be bf16 or fp16 normal decode KV" in thop_source
    assert "resident_kv_pool must have shape [global_tokens, 1, 576]" in thop_source
    assert "resident_tail_valid must be bool" in thop_source
    assert "resident_sink_blocks must equal resident_sink_tokens" in thop_source
    assert "resident_block_table must have shape [seqs, blocks]" in thop_source
    assert "hisparse_classify_resident_blocks" in planner_thop_source
    assert "resident_row_status must be uint8" in planner_thop_source
    assert "resident_block_flags shape must match host_slots" in planner_thop_source
    assert "resident_block_flags shape must match block_positions" in planner_thop_source
    assert "tail_valid must be bool" in planner_thop_source
    assert "resident_kv_lens=None" in thop_source
    assert "resident_kv_pool=None" in thop_source
    assert "request_topk_indices=None" in thop_source
    assert "prevent overlapping hot records" in thop_source
    assert "prevent overlapping layers" in thop_source
    assert "stride_factor must cover all layer token ranges" in thop_source
    assert "residentKvLens" in header.read_text()
    assert "residentKvPool" in header.read_text()
    assert "requestTopkIndices" in header.read_text()
    assert "residentTailTokenCount" in header.read_text()
    assert "residentKvPoolDtype" in header.read_text()
    assert "residentBlockTableBlocks" in header.read_text()
    assert "sparse_mla_decode_kvarn_hot.cu" in flash_cmake.read_text()
    assert "SparseMlaDecodeKvarnHotOp.cpp" in thop_cmake.read_text()
    fake_source = fake.read_text()
    assert "sparse_mla_decode_kvarn_hot" in fake_source
    assert "hisparse_classify_resident_blocks" in fake_source
    assert "resident_block_flags, resident_row_status" in fake_source
    assert "resident_kv_lens=None" in fake_source
    assert "resident_kv_pool=None" in fake_source
    assert "resident_tail_valid=None" in fake_source
    assert "request_topk_indices=None" in fake_source


def test_hisparse_schedule_copy_bridge_fails_closed_on_bad_row_ids():
    root = Path(__file__).resolve().parents[5]
    thop = root / "cpp/tensorrt_llm/thop/hisparseSwapInPackedKvarnOp.cpp"
    cmake = root / "cpp/tensorrt_llm/thop/CMakeLists.txt"
    fake = root / "tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py"

    source = thop.read_text()
    assert "hisparse_submit_packed_kvarn_copy_schedule" in source
    assert "compactRowIds" in source
    assert "row < 0 || row >= numRows" in source
    assert "statusRow < numRows" in source
    assert "rowStatus[statusRow] = kCopyInvalidRow" in source
    assert "commit stage cannot" in source
    assert "slot stride must be at least packed_bytes_per_block" in source
    assert "prevent overlapping packed records" in source
    assert "layer stride must cover all packed slots" in source
    assert "prevent overlapping layers" in source
    assert "hisparseSwapInPackedKvarnOp.cpp" in cmake.read_text()
    assert "hisparse_submit_packed_kvarn_copy_schedule" in fake.read_text()


def test_hisparse_native_mapping_allows_hot_capacity_larger_than_topk_source_contract():
    root = Path(__file__).resolve().parents[5]
    hisparse = root / "tensorrt_llm/_torch/attention_backend/sparse/hisparse.py"

    source = hisparse.read_text()
    assert "topk_indices.dim() != 2" in source
    assert "topk_indices.dtype != torch.int32" in source
    assert "TopK rows to match" in source
    assert "req_idx.numel()" in source
    assert "index_topk = int(topk_indices.shape[1])" in source
    assert "min(int(tier.hot_device_capacity_blocks)," in source
    assert "index_topk)" in source
    assert "positive TopK width " in source
    assert "and hot capacity." in source
    assert "index_topk=index_topk" in source


def test_hisparse_hot_planner_counts_duplicate_misses_once_source_contract():
    root = Path(__file__).resolve().parents[5]
    kernel = root / "cpp/tensorrt_llm/kernels/hisparseTopkToBlocks.cu"

    source = kernel.read_text()
    assert "int32_t requiredMisses = 0" in source
    assert "for (int32_t prev = 0; prev < i; ++prev)" in source
    assert "hostSlots[rowOffset + prev] == hostSlot" in source
    assert "commitGens[rowOffset + prev] == commitGen" in source
    assert "++requiredMisses" in source


def test_hisparse_attention_dispatch_consumes_kvarn_hot_descriptor():
    root = Path(__file__).resolve().parents[5]
    attention = root / "tensorrt_llm/_torch/modules/attention.py"
    hisparse = root / "tensorrt_llm/_torch/attention_backend/sparse/hisparse.py"
    source = attention.read_text()

    assert "def _sparse_mla_decode_kvarn_hot" in source
    assert "map_topk_to_hot_pool" in source
    assert "expected_layer_idx" in source
    assert "descriptor_is_current" in source
    assert "descriptor.step_id" in source
    assert "hisparse_coordinator.step_id" in source
    assert "int(descriptor.layer_idx) != expected_layer_idx" in source
    assert "descriptor.row_status.shape[0]" in source
    assert "descriptor.hot_indices.shape[1]" in source
    assert "descriptor.request_topk_indices is None" in source
    assert "descriptor.resident_block_flags is None" in source
    assert "descriptor.resident_block_status is None" in source
    assert "descriptor.request_topk_indices.shape[1]" in source
    assert "descriptor.resident_block_flags.shape[0]" in source
    assert "resident_block_flags," in source
    assert "resident_block_status," in source
    assert "resident = getattr(descriptor, \"resident_tokens\", None)" in source
    assert "explicit_sink_tail_v1" in source
    assert "resident.row_kv_lens.shape[0]" in source
    assert "resident.kv_pool.device" in source
    assert "resident.block_table.device" in source
    assert "resident.tail_token_count.device" in source
    assert "resident.row_req_idx" in source
    assert "resident.row_request_ids" in source
    assert "resident.kv_pool" in source
    assert "resident.tail_block_pos" in source
    assert "resident.tail_valid" in source
    assert "resident.sink_blocks" in source
    assert "request_topk_indices = descriptor.request_topk_indices.reshape" in source
    assert "assert_resident_token_policy_ready" in hisparse.read_text()
    assert "HiSparseResidentTokenDescriptor" in hisparse.read_text()
    assert "torch.ops.trtllm.sparse_mla_decode_kvarn_hot" in source
    assert "getattr(attn_metadata, \"num_generations\", 0)" in source
    assert "hisparse_sparse_mla_kvarn_hot" in source
    assert "for the current layer, row set, TopK " in source
    assert "width, and CUDA device; refusing" in source
    assert "refusing to route through NVFP4 or " in source
    assert "full-HBM sparse MLA" in source
    assert "HiSparse readiness returned unexpectedly" not in source


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

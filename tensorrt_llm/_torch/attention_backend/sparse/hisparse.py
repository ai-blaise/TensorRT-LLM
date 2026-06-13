"""OP-TRT HiSparse coordinator scaffolding.

The production HiSparse path is intentionally fail-closed until the packed
dense-MLA KVarN host/hot tiers, swap-in kernel, and NIXL direct-to-host
writer are wired. This module gives DSA a stable extension point without
introducing an FP16 staging path or a silent full-HBM fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Tuple

if TYPE_CHECKING:
    import numpy as np
    import torch


@dataclass(frozen=True)
class HiSparseTopKMapping:
    """Result of mapping request-relative TopK through the HiSparse hot pool."""

    topk_indices_global: "torch.Tensor"
    pool_view: Optional["torch.Tensor"] = None
    hot_block_ids: Optional["torch.Tensor"] = None


@dataclass(frozen=True)
class HiSparsePackedTierDescriptor:
    """Shape contract for packed dense-MLA KVarN HiSparse tiers."""

    num_layers: int
    tokens_per_block: int
    packed_bytes_per_block: int
    logical_host_capacity_blocks: int
    hot_device_capacity_blocks: int


@dataclass(frozen=True)
class HiSparsePackedTierTensors:
    """Packed dense-MLA KVarN storage owned by the HiSparse coordinator."""

    host_packed: "torch.Tensor"
    hot_packed: "torch.Tensor"
    host_valid: "torch.Tensor"
    host_commit_gen: "torch.Tensor"
    hot_host_slot: "torch.Tensor"
    hot_commit_gen: "torch.Tensor"
    hot_lru_tick: "torch.Tensor"
    host_pinned: bool
    device: object


@dataclass
class HiSparseHostBlockRecord:
    """Host-tier record for one request-relative committed block."""

    host_slot: int
    req_pool_idx: int
    block_pos: int
    logical_block_id: int = -1
    valid: bool = False
    commit_gen: int = 0
    owner_epoch: int = 0


@dataclass
class HiSparseHotBlockRecord:
    """Hot-tier metadata for one packed KVarN block slot."""

    hot_slot: int
    layer_idx: int
    req_pool_idx: Optional[int] = None
    block_pos: Optional[int] = None
    host_slot: Optional[int] = None
    commit_gen: int = -1
    lru_tick: int = 0

    @property
    def empty(self) -> bool:
        return self.req_pool_idx is None

    def clear(self) -> None:
        self.req_pool_idx = None
        self.block_pos = None
        self.host_slot = None
        self.commit_gen = -1
        self.lru_tick = 0


@dataclass
class HiSparseRequestState:
    """Request-owned host slots and lifecycle metadata."""

    req_pool_idx: int
    request_epoch: int
    host_slots_by_block_pos: Dict[int, int] = field(default_factory=dict)
    committed_layers_by_block_pos: Dict[int, set[int]] = field(default_factory=dict)
    pending_writes: int = 0
    admitted: bool = False


@dataclass(frozen=True)
class HiSparseHotSelection:
    """Metadata result of selecting request blocks into layer-local hot slots."""

    layer_idx: int
    req_pool_idx: int
    block_positions: Tuple[int, ...]
    host_slots: Tuple[int, ...]
    hot_slots: Tuple[int, ...]
    hits: int
    misses: int


class OPTRTHiSparseCoordinator:
    """Owns OP-TRT HiSparse request and step state.

    The disabled path is a no-op. The enabled path currently raises at startup
    and at the mapping call site so an accidentally enabled manifest cannot
    proceed via a non-production fallback.
    """

    def __init__(self, sparse_attention_config, kv_cache_manager=None) -> None:
        self.sparse_attention_config = sparse_attention_config
        self.kv_cache_manager = kv_cache_manager
        self.enabled = bool(
            getattr(sparse_attention_config, "hisparse_enabled", False))
        self.mode = getattr(sparse_attention_config, "hisparse_mode",
                            "dense_mla_kvarn")
        self.step_id = 0
        self._tier: Optional[HiSparsePackedTierDescriptor] = None
        self._free_host_slots: List[int] = []
        self._host_records: Dict[int, HiSparseHostBlockRecord] = {}
        self._requests: Dict[int, HiSparseRequestState] = {}
        self._hot_records_by_layer: Dict[int, List[HiSparseHotBlockRecord]] = {}
        self._lru_clock = 0
        self._tensors: Optional[HiSparsePackedTierTensors] = None

    @property
    def packed_tiers_configured(self) -> bool:
        return self._tier is not None

    @property
    def tier(self) -> Optional[HiSparsePackedTierDescriptor]:
        return self._tier

    @property
    def packed_tensors_allocated(self) -> bool:
        return self._tensors is not None

    @property
    def tensors(self) -> Optional[HiSparsePackedTierTensors]:
        return self._tensors

    def assert_startup_ready(self) -> None:
        if not self.enabled:
            return
        if not self.packed_tiers_configured:
            raise NotImplementedError(
                "HiSparse is enabled, but packed KVarN host/hot tiers are not "
                "configured yet. This is fail-closed by design: do not use an "
                "FP16 staging path, full-HBM fallback, or direct-to-host-off "
                "runtime for enabled HiSparse.")
        if not self.packed_tensors_allocated:
            raise NotImplementedError(
                "HiSparse is enabled and packed tier metadata is configured, "
                "but host-pinned/device packed KVarN tensors are not allocated "
                "yet. This remains fail-closed for serving.")
        raise NotImplementedError(
            "HiSparse packed tiers are configured, but the production packed "
            "KVarN swap-in kernel and sparse MLA hot-pool read path are not "
            "complete. This remains fail-closed for serving.")

    def reset_step(self) -> None:
        self.step_id += 1

    def configure_packed_tiers(
        self,
        *,
        num_layers: int,
        tokens_per_block: int,
        packed_bytes_per_block: int,
        logical_host_capacity_blocks: int,
        hot_device_capacity_blocks: int,
    ) -> HiSparsePackedTierDescriptor:
        """Install packed KVarN host/hot tier metadata.

        This method allocates metadata only. The actual host-pinned packed
        records, device hot packed records, and NIXL descriptors are Phase 2/3
        tensor work; callers must not treat this as a serving-ready data path.
        """
        values = {
            "num_layers": num_layers,
            "tokens_per_block": tokens_per_block,
            "packed_bytes_per_block": packed_bytes_per_block,
            "logical_host_capacity_blocks": logical_host_capacity_blocks,
            "hot_device_capacity_blocks": hot_device_capacity_blocks,
        }
        bad = [name for name, value in values.items() if int(value) <= 0]
        if bad:
            raise ValueError(
                "HiSparse packed tier dimensions must be positive; got "
                + ", ".join(f"{name}={values[name]}" for name in bad))

        self._tier = HiSparsePackedTierDescriptor(
            num_layers=int(num_layers),
            tokens_per_block=int(tokens_per_block),
            packed_bytes_per_block=int(packed_bytes_per_block),
            logical_host_capacity_blocks=int(logical_host_capacity_blocks),
            hot_device_capacity_blocks=int(hot_device_capacity_blocks),
        )
        self._free_host_slots = list(
            range(self._tier.logical_host_capacity_blocks))
        self._host_records.clear()
        self._requests.clear()
        self._hot_records_by_layer = {
            layer: [
                HiSparseHotBlockRecord(hot_slot=slot, layer_idx=layer)
                for slot in range(self._tier.hot_device_capacity_blocks)
            ]
            for layer in range(self._tier.num_layers)
        }
        self._lru_clock = 0
        self._tensors = None
        return self._tier

    def configure_from_kv_cache_manager(self) -> HiSparsePackedTierDescriptor:
        if self.kv_cache_manager is None:
            raise RuntimeError(
                "Cannot configure HiSparse tiers without a KV cache manager.")
        kv_cache_manager = self.kv_cache_manager
        kvarn_cfg = getattr(kv_cache_manager, "kvarn_cfg", None)
        if kvarn_cfg is None:
            raise RuntimeError(
                "HiSparse packed tiers require dense MLA KVarN to be selected.")
        tokens_per_block = int(getattr(kv_cache_manager, "tokens_per_block"))
        num_layers = int(getattr(kv_cache_manager, "num_local_layers"))
        num_blocks = getattr(kv_cache_manager, "num_blocks", None)
        if num_blocks is None:
            num_blocks = getattr(kv_cache_manager, "blocks_in_primary_pool")
        num_blocks = int(num_blocks)
        hot_blocks_per_req = int(
            getattr(self.sparse_attention_config, "hisparse_hot_blocks_per_req",
                    64))
        host_to_device_ratio = int(
            getattr(self.sparse_attention_config,
                    "hisparse_host_to_device_ratio", 8))
        max_batch_size = int(getattr(kv_cache_manager, "max_batch_size", 1))
        return self.configure_packed_tiers(
            num_layers=num_layers,
            tokens_per_block=tokens_per_block,
            packed_bytes_per_block=kvarn_cfg.packed_bytes(tokens_per_block),
            logical_host_capacity_blocks=max(num_blocks,
                                             max_batch_size
                                             * hot_blocks_per_req
                                             * host_to_device_ratio),
            hot_device_capacity_blocks=hot_blocks_per_req,
        )

    def allocate_packed_tensors(
        self,
        *,
        device=None,
        host_pinned: bool = True,
        tensor_factory: Optional[Callable[..., object]] = None,
    ) -> HiSparsePackedTierTensors:
        """Allocate production-shaped packed KVarN host/hot tensors.

        The host tier is CPU `uint8` and pinned when requested so NIXL can
        publish it as writable remote memory. The hot tier is device `uint8`.
        Metadata tensors are allocated alongside the packed bytes so the native
        swap-in kernel can consume a compact ABI later. This does not enable
        serving by itself; startup remains fail-closed until the swap-in/read
        kernels are wired.
        """
        tier = self._require_configured()
        if tensor_factory is None:
            import torch

            def tensor_factory(shape, *, dtype, device, pin_memory=False):
                kwargs = {
                    "dtype": getattr(torch, dtype),
                    "device": device,
                }
                if pin_memory:
                    kwargs["pin_memory"] = True
                return torch.empty(shape, **kwargs)

        if device is None:
            device = "cuda"
        host_shape = (tier.num_layers, tier.logical_host_capacity_blocks,
                      tier.packed_bytes_per_block)
        hot_shape = (tier.num_layers, tier.hot_device_capacity_blocks,
                     tier.packed_bytes_per_block)
        host_meta_shape = (tier.num_layers, tier.logical_host_capacity_blocks)
        hot_meta_shape = (tier.num_layers, tier.hot_device_capacity_blocks)
        tensors = HiSparsePackedTierTensors(
            host_packed=tensor_factory(host_shape,
                                       dtype="uint8",
                                       device="cpu",
                                       pin_memory=host_pinned),
            hot_packed=tensor_factory(hot_shape,
                                      dtype="uint8",
                                      device=device,
                                      pin_memory=False),
            host_valid=tensor_factory(host_meta_shape,
                                      dtype="bool",
                                      device="cpu",
                                      pin_memory=host_pinned),
            host_commit_gen=tensor_factory(host_meta_shape,
                                           dtype="int64",
                                           device="cpu",
                                           pin_memory=host_pinned),
            hot_host_slot=tensor_factory(hot_meta_shape,
                                         dtype="int64",
                                         device=device,
                                         pin_memory=False),
            hot_commit_gen=tensor_factory(hot_meta_shape,
                                          dtype="int64",
                                          device=device,
                                          pin_memory=False),
            hot_lru_tick=tensor_factory(hot_meta_shape,
                                        dtype="int64",
                                        device=device,
                                        pin_memory=False),
            host_pinned=host_pinned,
            device=device,
        )
        self._zero_or_fill(tensors.host_packed, 0)
        self._zero_or_fill(tensors.hot_packed, 0)
        self._zero_or_fill(tensors.host_valid, 0)
        self._zero_or_fill(tensors.host_commit_gen, 0)
        self._zero_or_fill(tensors.hot_host_slot, -1)
        self._zero_or_fill(tensors.hot_commit_gen, -1)
        self._zero_or_fill(tensors.hot_lru_tick, 0)
        self._tensors = tensors
        return tensors

    @staticmethod
    def _zero_or_fill(tensor, value: int) -> None:
        fill = getattr(tensor, "fill_", None)
        if fill is not None:
            fill(value)
            return
        zero = getattr(tensor, "zero_", None)
        if value == 0 and zero is not None:
            zero()

    def host_registration_descs(
        self,
        *,
        prefix: str = "hisparse_host",
        include_metadata: bool = True,
    ) -> List[Tuple[int, int, int, str]]:
        """Return DRAM registration descriptors for NIXL writable host tiers.

        The descriptors intentionally describe CPU host-pinned memory only. The
        transfer path must keep these DRAM regions out of the existing VRAM KV
        write batch until a dedicated HiSparse host-write meta path is added.
        """
        entries = self._host_tier_entries(include_metadata=include_metadata)
        descs: List[Tuple[int, int, int, str]] = []
        for name, tensor, item_size in entries:
            base_ptr = self._tensor_data_ptr(tensor)
            for layer in range(self._require_configured().num_layers):
                layer_ptr = base_ptr + layer * self._host_layer_stride_bytes(
                    name, item_size)
                descs.append(
                    (layer_ptr, self._host_layer_stride_bytes(name, item_size),
                     0, f"{prefix}.{name}.layer{layer}"))
        return descs

    def transfer_meta(self):
        """Build serializable transfer metadata for the HiSparse host tier."""
        import numpy as np

        from tensorrt_llm._torch.disaggregation.native.auxiliary import (
            HiSparseHostTierMeta,
        )

        entries = self._host_tier_entries(include_metadata=True)
        ptrs: List[int] = []
        sizes: List[int] = []
        item_sizes: List[int] = []
        names: List[str] = []
        for name, tensor, item_size in entries:
            base_ptr = self._tensor_data_ptr(tensor)
            for layer in range(self._require_configured().num_layers):
                ptrs.append(base_ptr +
                            layer * self._host_layer_stride_bytes(
                                name, item_size))
                sizes.append(self._host_layer_stride_bytes(name, item_size))
                item_sizes.append(item_size)
                names.append(f"{name}.layer{layer}")
        tier = self._require_configured()
        tensors = self._require_tensors()
        return HiSparseHostTierMeta(
            ptrs=np.array(ptrs, dtype=np.int64),
            size=np.array(sizes, dtype=np.int64),
            item_sizes=np.array(item_sizes, dtype=np.int64),
            names=names,
            num_layers=tier.num_layers,
            host_slots=tier.logical_host_capacity_blocks,
            packed_bytes_per_block=tier.packed_bytes_per_block,
            device=str(getattr(tensors.host_packed, "device", "cpu")),
        )

    def host_packed_ptrs_for_blocks(
        self,
        *,
        layer_idx: int,
        req_pool_idx: int,
        block_positions: Iterable[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return host-pinned packed-KVarN destinations for request blocks."""
        import numpy as np

        tier = self._require_configured()
        tensors = self._require_tensors()
        layer_idx = int(layer_idx)
        if layer_idx < 0 or layer_idx >= tier.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} outside configured HiSparse layer "
                f"range [0, {tier.num_layers}).")
        base_ptr = self._tensor_data_ptr(tensors.host_packed)
        layer_base = (
            base_ptr +
            layer_idx * tier.logical_host_capacity_blocks
            * tier.packed_bytes_per_block)
        ptrs = []
        sizes = []
        for block_pos in block_positions:
            record = self._host_record(req_pool_idx, int(block_pos))
            ptrs.append(layer_base +
                        record.host_slot * tier.packed_bytes_per_block)
            sizes.append(tier.packed_bytes_per_block)
        return (np.array(ptrs, dtype=np.int64),
                np.array(sizes, dtype=np.int64))

    def _require_tensors(self) -> HiSparsePackedTierTensors:
        if self._tensors is None:
            raise RuntimeError(
                "HiSparse packed tensors are not allocated. Call "
                "allocate_packed_tensors() before requesting transfer "
                "descriptors.")
        return self._tensors

    def _host_tier_entries(
        self,
        *,
        include_metadata: bool,
    ) -> List[Tuple[str, object, int]]:
        tier = self._require_configured()
        tensors = self._require_tensors()
        entries = [("host_packed", tensors.host_packed,
                    tier.packed_bytes_per_block)]
        if include_metadata:
            entries.extend([
                ("host_valid", tensors.host_valid,
                 self._tensor_element_size(tensors.host_valid)),
                ("host_commit_gen", tensors.host_commit_gen,
                 self._tensor_element_size(tensors.host_commit_gen)),
            ])
        for name, tensor, _item_size in entries:
            device = str(getattr(tensor, "device", "cpu"))
            if not device.startswith("cpu"):
                raise RuntimeError(
                    f"HiSparse host tier tensor {name} must be CPU memory, got "
                    f"device={device!r}.")
        return entries

    def _host_layer_stride_bytes(self, name: str, item_size: int) -> int:
        tier = self._require_configured()
        if name == "host_packed":
            return tier.logical_host_capacity_blocks * tier.packed_bytes_per_block
        return tier.logical_host_capacity_blocks * int(item_size)

    @staticmethod
    def _tensor_data_ptr(tensor) -> int:
        data_ptr = getattr(tensor, "data_ptr", None)
        if data_ptr is None:
            raise RuntimeError(
                "HiSparse host tensor does not expose data_ptr(); cannot build "
                "NIXL descriptors.")
        return int(data_ptr())

    @staticmethod
    def _tensor_element_size(tensor) -> int:
        element_size = getattr(tensor, "element_size", None)
        if element_size is not None:
            return int(element_size())
        dtype = str(getattr(tensor, "dtype", ""))
        if "int64" in dtype:
            return 8
        if "int32" in dtype or "float32" in dtype:
            return 4
        if "float16" in dtype or "bfloat16" in dtype:
            return 2
        return 1

    def _require_configured(self) -> HiSparsePackedTierDescriptor:
        if self._tier is None:
            raise RuntimeError(
                "HiSparse packed tiers are not configured. Call "
                "configure_packed_tiers() before reserving requests.")
        return self._tier

    def reserve_request(
        self,
        req_pool_idx: int,
        num_prompt_blocks: int,
        *,
        request_epoch: Optional[int] = None,
    ) -> HiSparseRequestState:
        self._require_configured()
        req_pool_idx = int(req_pool_idx)
        num_prompt_blocks = int(num_prompt_blocks)
        if num_prompt_blocks < 0:
            raise ValueError("num_prompt_blocks must be non-negative.")
        if req_pool_idx in self._requests:
            raise ValueError(
                f"HiSparse request {req_pool_idx} is already reserved.")
        if len(self._free_host_slots) < num_prompt_blocks:
            raise MemoryError(
                "Insufficient HiSparse host slots: requested "
                f"{num_prompt_blocks}, available {len(self._free_host_slots)}.")
        state = HiSparseRequestState(
            req_pool_idx=req_pool_idx,
            request_epoch=self.step_id if request_epoch is None else
            int(request_epoch),
        )
        for block_pos in range(num_prompt_blocks):
            host_slot = self._free_host_slots.pop(0)
            self._host_records[host_slot] = HiSparseHostBlockRecord(
                host_slot=host_slot,
                req_pool_idx=req_pool_idx,
                block_pos=block_pos,
                owner_epoch=state.request_epoch,
            )
            state.host_slots_by_block_pos[block_pos] = host_slot
        self._requests[req_pool_idx] = state
        return state

    def reserve_or_get_request(
        self,
        req_pool_idx: int,
        num_prompt_blocks: int,
        *,
        request_epoch: Optional[int] = None,
    ) -> HiSparseRequestState:
        """Reserve request host rows once and validate later publications.

        Generation-first receive metadata may be requested more than once for a
        session as the transfer lifecycle is retried or extended. Reusing the
        same request-relative host slots keeps direct-to-host descriptors
        stable; changing the prompt block count for an active request is a
        correctness error because it would make previously published writable
        offsets ambiguous.
        """
        req_pool_idx = int(req_pool_idx)
        existing = self._requests.get(req_pool_idx)
        if existing is None:
            return self.reserve_request(req_pool_idx,
                                        num_prompt_blocks,
                                        request_epoch=request_epoch)
        existing_blocks = len(existing.host_slots_by_block_pos)
        if existing_blocks != int(num_prompt_blocks):
            raise RuntimeError(
                f"HiSparse request {req_pool_idx} already has "
                f"{existing_blocks} host block slot(s), cannot republish "
                f"{num_prompt_blocks}.")
        return existing

    def host_slots_for_request(
        self,
        req_pool_idx: int,
        *,
        num_prompt_blocks: Optional[int] = None,
    ) -> Tuple[int, ...]:
        state = self._requests.get(int(req_pool_idx))
        if state is None:
            raise KeyError(f"HiSparse request {req_pool_idx} is not reserved.")
        if num_prompt_blocks is None:
            num_prompt_blocks = len(state.host_slots_by_block_pos)
        return tuple(state.host_slots_by_block_pos[block_pos]
                     for block_pos in range(int(num_prompt_blocks)))

    def mark_host_block_committed(
        self,
        req_pool_idx: int,
        block_pos: int,
        *,
        logical_block_id: Optional[int] = None,
    ) -> HiSparseHostBlockRecord:
        tier = self._require_configured()
        state = self._requests.get(int(req_pool_idx))
        if state is None:
            raise KeyError(f"HiSparse request {req_pool_idx} is not reserved.")
        record = self._host_record(req_pool_idx, block_pos)
        if logical_block_id is not None:
            record.logical_block_id = int(logical_block_id)
        record.valid = True
        record.commit_gen += 1
        state.committed_layers_by_block_pos[int(block_pos)] = set(
            range(tier.num_layers))
        self._write_host_commit_metadata(record,
                                         range(tier.num_layers),
                                         valid=True)
        return record

    def begin_host_write(self, req_pool_idx: int) -> HiSparseRequestState:
        state = self._request_state(req_pool_idx)
        state.pending_writes += 1
        state.admitted = False
        return state

    def finish_host_write(self, req_pool_idx: int) -> HiSparseRequestState:
        state = self._request_state(req_pool_idx)
        if state.pending_writes <= 0:
            raise RuntimeError(
                f"HiSparse request {req_pool_idx} has no pending host writes "
                "to finish.")
        state.pending_writes -= 1
        return state

    def abort_request(self, req_pool_idx: int) -> None:
        """Clear admission state after a terminal failed/cancelled transfer."""
        state = self._requests.get(int(req_pool_idx))
        if state is None:
            return
        state.pending_writes = 0
        state.admitted = False

    def mark_host_write_committed(
        self,
        req_pool_idx: int,
        *,
        layer_indices: Iterable[int],
        block_positions: Iterable[int],
    ) -> Tuple[HiSparseHostBlockRecord, ...]:
        """Record successful packed host writes and publish full-block commits.

        ``layer_indices`` and ``block_positions`` are parallel arrays of
        successfully transferred packed KVarN records. A request-relative block
        becomes globally selectable only after every local layer for that block
        has reported a successful host write. Replayed coverage is idempotent
        and does not bump ``commit_gen`` again.
        """
        tier = self._require_configured()
        req_pool_idx = int(req_pool_idx)
        state = self._requests.get(req_pool_idx)
        if state is None:
            raise KeyError(f"HiSparse request {req_pool_idx} is not reserved.")
        layers = [int(layer) for layer in layer_indices]
        blocks = [int(block_pos) for block_pos in block_positions]
        if len(layers) != len(blocks):
            raise ValueError(
                "HiSparse host-write commit coverage requires parallel "
                f"layer/block arrays, got {len(layers)} layer entries and "
                f"{len(blocks)} block entries.")
        if not layers:
            return ()
        bad_layers = [
            layer for layer in layers
            if layer < 0 or layer >= int(tier.num_layers)
        ]
        if bad_layers:
            sample = ", ".join(str(layer) for layer in bad_layers[:8])
            raise IndexError(
                "HiSparse host-write commit layer index out of range: "
                f"{sample}; num_layers={tier.num_layers}.")

        newly_full: List[HiSparseHostBlockRecord] = []
        all_layers = set(range(tier.num_layers))
        for layer, block_pos in dict.fromkeys(zip(layers, blocks)):
            record = self._host_record(req_pool_idx, block_pos)
            committed = state.committed_layers_by_block_pos.setdefault(
                int(block_pos), set())
            if layer in committed:
                continue
            committed.add(layer)
            if not record.valid and committed >= all_layers:
                record.valid = True
                record.commit_gen += 1
                self._write_host_commit_metadata(record,
                                                 all_layers,
                                                 valid=True)
                newly_full.append(record)
        return tuple(newly_full)

    def host_block_committed(self, req_pool_idx: int, block_pos: int) -> bool:
        return self._host_record(req_pool_idx, block_pos).valid

    def request_ready_for_admission(
        self,
        req_pool_idx: int,
        *,
        num_prompt_blocks: Optional[int] = None,
    ) -> bool:
        state = self._request_state(req_pool_idx)
        if state.pending_writes != 0:
            return False
        if num_prompt_blocks is None:
            num_prompt_blocks = len(state.host_slots_by_block_pos)
        return all(
            self.host_block_committed(state.req_pool_idx, block_pos)
            for block_pos in range(int(num_prompt_blocks)))

    def mark_request_admitted(
        self,
        req_pool_idx: int,
        *,
        num_prompt_blocks: Optional[int] = None,
    ) -> HiSparseRequestState:
        state = self._request_state(req_pool_idx)
        if not self.request_ready_for_admission(
                req_pool_idx, num_prompt_blocks=num_prompt_blocks):
            raise RuntimeError(
                "HiSparse request cannot be admitted until all reserved prompt "
                "host blocks are committed and no host writes are pending: "
                f"req={req_pool_idx}, pending_writes={state.pending_writes}.")
        state.admitted = True
        return state

    def release_request(self, req_pool_idx: int, *, force: bool = False) -> None:
        req_pool_idx = int(req_pool_idx)
        state = self._requests.get(req_pool_idx)
        if state is None:
            return
        if state.pending_writes and not force:
            raise RuntimeError(
                "Cannot release HiSparse request with pending host writes: "
                f"req={req_pool_idx}, pending_writes={state.pending_writes}.")
        if force:
            self.abort_request(req_pool_idx)
        state = self._requests.pop(req_pool_idx)
        released_slots = sorted(state.host_slots_by_block_pos.values())
        released_set = set(released_slots)
        for records in self._hot_records_by_layer.values():
            for hot in records:
                if hot.host_slot in released_set:
                    hot.clear()
        for host_slot in released_slots:
            self._host_records.pop(host_slot, None)
        self._free_host_slots.extend(released_slots)
        self._free_host_slots.sort()

    def invalidate_host_blocks(self, req_pool_idx: int,
                               block_positions: Iterable[int]) -> None:
        state = self._requests.get(int(req_pool_idx))
        if state is None:
            raise KeyError(f"HiSparse request {req_pool_idx} is not reserved.")
        released_host_slots = set()
        for block_pos in block_positions:
            record = self._host_record(req_pool_idx, int(block_pos))
            record.valid = False
            state.committed_layers_by_block_pos.pop(int(block_pos), None)
            self._write_host_commit_metadata(
                record,
                range(self._require_configured().num_layers),
                valid=False,
            )
            released_host_slots.add(record.host_slot)
        for records in self._hot_records_by_layer.values():
            for hot in records:
                if hot.host_slot in released_host_slots:
                    hot.clear()

    def _write_host_commit_metadata(
        self,
        record: HiSparseHostBlockRecord,
        layers: Iterable[int],
        *,
        valid: bool,
        commit_gen: Optional[int] = None,
    ) -> None:
        tensors = self._tensors
        if tensors is None:
            return
        if commit_gen is None:
            commit_gen = record.commit_gen
        for layer in layers:
            self._set_tensor_value(tensors.host_valid, int(layer),
                                   record.host_slot, bool(valid))
            self._set_tensor_value(tensors.host_commit_gen, int(layer),
                                   record.host_slot, int(commit_gen))

    @staticmethod
    def _set_tensor_value(tensor, layer: int, host_slot: int, value) -> None:
        try:
            tensor[int(layer), int(host_slot)] = value
        except (AttributeError, TypeError, IndexError):
            # Lightweight test fakes expose only shape/data_ptr/fill_. Real
            # torch tensors take this branch only on unsupported tensor-like
            # objects, and serving remains fail-closed until kernels are wired.
            return

    def select_hot_blocks(
        self,
        *,
        layer_idx: int,
        req_pool_idx: int,
        block_positions: Iterable[int],
    ) -> HiSparseHotSelection:
        tier = self._require_configured()
        layer_idx = int(layer_idx)
        if layer_idx < 0 or layer_idx >= tier.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} outside configured HiSparse layer "
                f"range [0, {tier.num_layers}).")
        deduped = tuple(dict.fromkeys(int(pos) for pos in block_positions))
        hot_records = self._hot_records_by_layer[layer_idx]
        host_slots: List[int] = []
        hot_slots: List[int] = []
        hits = 0
        misses = 0
        for block_pos in deduped:
            host = self._host_record(req_pool_idx, block_pos)
            if not host.valid:
                raise RuntimeError(
                    "Cannot select an uncommitted HiSparse host block: "
                    f"req={req_pool_idx}, block_pos={block_pos}.")
            hot = self._find_hot_hit(hot_records, host)
            if hot is not None:
                hits += 1
            else:
                misses += 1
                hot = self._pick_hot_victim(hot_records)
                hot.req_pool_idx = host.req_pool_idx
                hot.block_pos = host.block_pos
                hot.host_slot = host.host_slot
                hot.commit_gen = host.commit_gen
            self._touch_hot(hot)
            host_slots.append(host.host_slot)
            hot_slots.append(hot.hot_slot)
        return HiSparseHotSelection(
            layer_idx=layer_idx,
            req_pool_idx=int(req_pool_idx),
            block_positions=deduped,
            host_slots=tuple(host_slots),
            hot_slots=tuple(hot_slots),
            hits=hits,
            misses=misses,
        )

    def _host_record(self, req_pool_idx: int,
                     block_pos: int) -> HiSparseHostBlockRecord:
        state = self._request_state(req_pool_idx)
        try:
            host_slot = state.host_slots_by_block_pos[int(block_pos)]
        except KeyError as exc:
            raise KeyError(
                f"HiSparse request {req_pool_idx} has no block_pos "
                f"{block_pos}.") from exc
        return self._host_records[host_slot]

    def _request_state(self, req_pool_idx: int) -> HiSparseRequestState:
        state = self._requests.get(int(req_pool_idx))
        if state is None:
            raise KeyError(f"HiSparse request {req_pool_idx} is not reserved.")
        return state

    @staticmethod
    def _find_hot_hit(
        hot_records: List[HiSparseHotBlockRecord],
        host: HiSparseHostBlockRecord,
    ) -> Optional[HiSparseHotBlockRecord]:
        for hot in hot_records:
            if (hot.req_pool_idx == host.req_pool_idx
                    and hot.block_pos == host.block_pos
                    and hot.host_slot == host.host_slot
                    and hot.commit_gen == host.commit_gen):
                return hot
        return None

    @staticmethod
    def _pick_hot_victim(
        hot_records: List[HiSparseHotBlockRecord],
    ) -> HiSparseHotBlockRecord:
        for hot in hot_records:
            if hot.empty:
                return hot
        return min(hot_records, key=lambda hot: hot.lru_tick)

    def _touch_hot(self, hot: HiSparseHotBlockRecord) -> None:
        self._lru_clock += 1
        hot.lru_tick = self._lru_clock

    def stats(self) -> Dict[str, int]:
        configured = self._tier is not None
        hot_used = 0
        if configured:
            hot_used = sum(not hot.empty
                           for records in self._hot_records_by_layer.values()
                           for hot in records)
        return {
            "configured": int(configured),
            "requests": len(self._requests),
            "host_used": len(self._host_records),
            "host_free": len(self._free_host_slots),
            "hot_used": hot_used,
            "tensors_allocated": int(self._tensors is not None),
        }

    def map_topk_to_hot_pool(
        self,
        *,
        topk_indices,
        metadata,
        layer_idx: int,
        skip_topk: bool,
        is_generation: bool,
    ) -> Optional[HiSparseTopKMapping]:
        if not self.enabled:
            return None
        raise NotImplementedError(
            "HiSparse hot-pool TopK mapping is not implemented yet. The next "
            "phase must map Indexer/HISA request-relative TopK into packed "
            "KVarN hot blocks before sparse MLA reads them.")

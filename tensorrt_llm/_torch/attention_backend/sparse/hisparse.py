"""OP-TRT HiSparse coordinator scaffolding.

The production HiSparse path is intentionally fail-closed until the packed
dense-MLA KVarN host/hot tiers, swap-in kernel, and NIXL direct-to-host
writer are wired. This module gives DSA a stable extension point without
introducing an FP16 staging path or a silent full-HBM fallback.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Tuple

if TYPE_CHECKING:
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

    def mark_host_block_committed(
        self,
        req_pool_idx: int,
        block_pos: int,
        *,
        logical_block_id: Optional[int] = None,
    ) -> HiSparseHostBlockRecord:
        record = self._host_record(req_pool_idx, block_pos)
        if logical_block_id is not None:
            record.logical_block_id = int(logical_block_id)
        record.valid = True
        record.commit_gen += 1
        return record

    def release_request(self, req_pool_idx: int) -> None:
        req_pool_idx = int(req_pool_idx)
        state = self._requests.pop(req_pool_idx, None)
        if state is None:
            return
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
        released_host_slots = set()
        for block_pos in block_positions:
            record = self._host_record(req_pool_idx, int(block_pos))
            record.valid = False
            released_host_slots.add(record.host_slot)
        for records in self._hot_records_by_layer.values():
            for hot in records:
                if hot.host_slot in released_host_slots:
                    hot.clear()

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
        state = self._requests.get(int(req_pool_idx))
        if state is None:
            raise KeyError(f"HiSparse request {req_pool_idx} is not reserved.")
        try:
            host_slot = state.host_slots_by_block_pos[int(block_pos)]
        except KeyError as exc:
            raise KeyError(
                f"HiSparse request {req_pool_idx} has no block_pos "
                f"{block_pos}.") from exc
        return self._host_records[host_slot]

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

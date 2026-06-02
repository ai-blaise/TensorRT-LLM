"""LayerSplit owner-assignment policies for DSA KV / indexer cache.

LayerSplit (z.ai "Scaling Pain" §4) partitions per-layer DSA KV cache and
indexer K cache across context-parallel ranks. Each layer has exactly one
owning CP rank; the owner stores the dense cache and broadcasts it to the
other ranks before that layer's attention compute.

This module owns the policy that decides which CP rank owns which layer.
Owner assignment is a pure function of (num_layers, cp_size, policy) and
runs at metadata setup time, not in the hot path. The returned
``LayerSplitOwnership`` object is hashable so it can sit on CUDA-graph-stable
metadata structures without copies and is safe to share across decoding
steps within the same model load.

Two policies are supported today:

``round_robin``
    ``owner_map[L] = L mod cp_size``. Matches the SGLang op-ls reference
    implementation and the ``layout="interleaved"`` HF model-card alias.
    Spreads heterogeneous layer cost (DSA vs MoE, dense vs sparse) across
    CP ranks at the granularity of every layer.

``contiguous``
    ``owner_map[0..k0) = 0, owner_map[k0..k0+k1) = 1, ...`` with the
    remainder distributed to the lowest-rank owners. Keeps each rank's
    owned layers contiguous, which matches a few comm patterns (windowed
    KV refresh, prefix-cache hand-off) better than round-robin.

Edge cases the policies must handle without surprise:

* ``cp_size == 1`` collapses to "all layers owned by rank 0" regardless
  of policy. The runtime should never reach LayerSplit broadcast code in
  this case but the ownership map must still be well-defined for the
  metadata path that always builds it.
* ``cp_size > num_layers`` is legal: extra ranks own zero layers and the
  broadcast for those ranks is a no-op.
* ``num_layers % cp_size != 0`` is the common case for DeepSeek-V3.2 (61
  layers) on cp_size in {2,4,8}.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

try:
    import torch
except ImportError:  # pragma: no cover - torch is always available in prod
    torch = None  # type: ignore[assignment]

_VALID_POLICIES = ("round_robin", "contiguous")
_VALID_TRANSFER_BACKENDS = ("auto", "ucx", "nixl")

# LayerSplit's per-layer broadcast publishes two logically-independent
# payloads, mirroring the z.ai "Scaling Pain" Figure 4(b) protocol:
# - "indexer": the indexer K cache slice (small; ~1/8 dense KV size).
#   Broadcasting this first lets the receiver start the indexer compute as
#   soon as it lands while the larger dense KV broadcast is still in
#   flight on a sibling stream.
# - "kv":      the dense KV cache slice (the bulk of the bytes). Required
#   before the sparse-attention compute reads any of layer-L's KV.
# Single-channel callers default to ``channel="kv"`` so the M5c / M6 call
# sites are byte-identical to their channel-aware behavior.
_VALID_CHANNELS = ("kv", "indexer")


@dataclass(frozen=True)
class LayerSplitOwnership:
    """Immutable per-layer CP-rank ownership table for LayerSplit."""

    owner_map: Tuple[int, ...]
    cp_size: int
    policy: str

    @property
    def num_layers(self) -> int:
        return len(self.owner_map)

    def is_owner(self, layer: int, cp_rank: int) -> bool:
        return self.owner_map[layer] == cp_rank

    def owner_of(self, layer: int) -> int:
        return self.owner_map[layer]

    def owned_layers(self, cp_rank: int) -> Tuple[int, ...]:
        return tuple(layer for layer, owner in enumerate(self.owner_map)
                     if owner == cp_rank)

    def layers_per_rank(self) -> Tuple[int, ...]:
        counts = [0] * self.cp_size
        for owner in self.owner_map:
            counts[owner] += 1
        return tuple(counts)


def compute_owner_assignment(
        num_layers: int,
        cp_size: int,
        policy: str = "round_robin") -> LayerSplitOwnership:
    """Return the LayerSplit ownership table for ``num_layers`` across
    ``cp_size`` CP ranks under the requested ``policy``.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}")
    if policy not in _VALID_POLICIES:
        raise ValueError(
            f"unknown LayerSplit owner_assignment policy {policy!r}; "
            f"expected one of {_VALID_POLICIES}")

    if cp_size == 1:
        owner_map: Tuple[int, ...] = tuple([0] * num_layers)
    elif policy == "round_robin":
        owner_map = tuple(layer % cp_size for layer in range(num_layers))
    else:  # contiguous
        per = num_layers // cp_size
        extra = num_layers % cp_size
        flat = []
        for rank in range(cp_size):
            count = per + (1 if rank < extra else 0)
            flat.extend([rank] * count)
        owner_map = tuple(flat)

    return LayerSplitOwnership(owner_map=owner_map,
                               cp_size=cp_size,
                               policy=policy)


@dataclass
class LayerSplitRuntimeState:
    """All LayerSplit state the DSA runtime needs for one model load.

    This object is built once per ``DSACacheManager`` instance and lives for
    the model's lifetime. Holding it as a separate dataclass keeps
    ``DSACacheManager``'s diff tiny and lets the runtime logic
    (ownership lookups, the side-stream synchronization placeholder, the
    transfer-backend selection) be unit-tested without the C++ KV cache
    manager parent.

    Fields:
    - ``enabled``: whether LayerSplit is configured on. When False all the
      other fields are inert (``ownership`` is None, no comm stream is
      created) so the runtime path collapses to the regular DSA behavior.
    - ``ownership``: the per-layer owner table (see :class:`LayerSplitOwnership`).
    - ``transfer_backend``: ``"auto" | "ucx" | "nixl"``. Auto resolves at
      M5+ when the broadcast path is wired; M3 only records the selection.
    - ``all_cp_ranks_transfer``: must be True today; partial-rank transfer
      is not implemented and is rejected by ``SparseAttentionConfig``.
    - ``cp_size`` / ``cp_rank``: cached from the model's ``Mapping`` so the
      runtime does not have to re-traverse the parallel config on every
      layer call.
    - ``comm_stream``: a dedicated ``torch.cuda.Stream`` for owner -> peers
      broadcasts. Created at model-load time so it stays graph-stable
      (CUDA-graph capture freezes the stream identity at capture time).
      ``None`` when LayerSplit is disabled or when CUDA is unavailable
      (CPU-only tests, doc generation).
    - ``cp_group``: the ``torch.distributed.ProcessGroup`` for the CP
      collective. The DSACacheManager resolves it from
      ``mapping.cp_group_pg`` (via the device-mesh path) and stashes it
      here so the per-layer hook does not have to re-traverse the
      mapping on every call. ``None`` when LayerSplit is disabled,
      when ``cp_size <= 1``, or when the mapping has not yet built the
      process group (e.g. CPU-only unit tests, model-load smoke tests
      before ``init_process_group``).
    """

    enabled: bool
    ownership: Optional[LayerSplitOwnership]
    transfer_backend: str
    all_cp_ranks_transfer: bool
    cp_size: int
    cp_rank: int
    comm_stream: Optional[Any] = field(default=None, repr=False)
    cp_group: Optional[Any] = field(default=None, repr=False)
    cp_group_ranks: Optional[Tuple[int, ...]] = field(default=None, repr=False)
    # M8b: optional second comm stream dedicated to the "indexer" channel
    # so the small indexer broadcast does not queue behind the larger
    # dense KV broadcast on the single comm stream. None falls back to
    # the primary ``comm_stream`` (the M6 single-stream posture).
    indexer_comm_stream: Optional[Any] = field(default=None, repr=False)
    # Backwards-compat: M5c callers ask for a single shared payload tensor
    # (one allocation reused across every layer's sync broadcast).
    _heartbeat_payload: Optional[Any] = field(default=None,
                                              repr=False,
                                              init=False)
    # M6 / M8b: per-(layer, channel) payload tensors so the layer-L
    # broadcasts (one per channel) and the prefetched layer-L+1
    # broadcasts can all be in flight at the same time without aliasing.
    # Keyed by (layer_idx, channel) — kept as a single dict for clarity.
    _per_layer_payloads: Dict[Tuple[int, str], Any] = field(
        default_factory=dict, repr=False, init=False)
    # M6 / M8b: in-flight prefetched broadcasts. The layer-L hook
    # consumes the events for layer L on every channel (if any) and then
    # prefetches the L+1 events on every channel. Keyed by
    # (layer_idx, channel).
    _prefetched_events: Dict[Tuple[int, str], Any] = field(
        default_factory=dict, repr=False, init=False)

    def bind_cp_group(self,
                      cp_group: Any,
                      cp_group_ranks: Optional[Any] = None) -> None:
        """Late-bind the CP process group resolved from ``mapping``.

        The DSACacheManager resolves the group eagerly at construction
        time when possible; tests that build the runtime state directly
        (without a mapping) can bind the group post-hoc. ``torch.distributed``
        broadcast expects a global source rank, not a CP-local rank, so cache
        the group's global ranks when available.
        """
        self.cp_group = cp_group
        if cp_group_ranks is not None:
            self.cp_group_ranks = tuple(int(rank) for rank in cp_group_ranks)
            return
        try:
            import torch.distributed as dist
            self.cp_group_ranks = tuple(
                int(rank) for rank in dist.get_process_group_ranks(cp_group))
        except Exception:
            self.cp_group_ranks = None

    def broadcast_src_rank(self, layer_idx: int) -> int:
        """Return the global distributed rank that owns ``layer_idx``.

        ``LayerSplitOwnership`` stores owners in CP-local rank space. PyTorch
        collectives take a global ``src`` rank when a process group is passed,
        so translate through the CP group's rank list for production TP/CP
        meshes such as groups ``[2, 3]`` or ``[4, 5]``.
        """
        local_owner = self.owner_of(layer_idx)
        if self.cp_group_ranks is None:
            return local_owner
        if local_owner < 0 or local_owner >= len(self.cp_group_ranks):
            raise ValueError(
                f"LayerSplit owner {local_owner} is outside cp_group_ranks "
                f"of size {len(self.cp_group_ranks)}")
        return self.cp_group_ranks[local_owner]

    def ensure_heartbeat_payload(
            self,
            layer_idx: Optional[int] = None,
            payload_bytes: Optional[int] = None,
            channel: str = "kv") -> Optional[Any]:
        """Lazily allocate a CUDA tensor used as the per-layer broadcast
        payload.

        ``layer_idx=None`` (default) returns a single shared payload
        reused across all layers — used by the M5c sync-broadcast caller
        where layer-L's broadcast always completes before layer-L+1's
        broadcast starts.

        ``layer_idx=int`` returns a layer-specific payload tensor — used
        by the M6 overlap caller so that the layer-L broadcast (in flight
        on the comm stream) and the prefetched layer-L+1 broadcast (also
        on the comm stream) do not alias each other's data.

        ``payload_bytes=None`` (default) keeps the M5c posture: a 16-byte
        heartbeat (4 × int32) that exercises the NCCL broadcast path end
        to end so configuration errors surface at first decode step. The
        16-byte payload is too small to matter for bandwidth measurement
        but proves the wiring is correct.

        ``payload_bytes > 16`` (M5d-bandwidth posture) grows the payload
        to a realistic per-layer KV-slice size so the production decode
        actually pays NCCL bandwidth proportional to what the M5d real
        active-KV broadcast will pay. Used to:
        - Measure the M6 cross-layer overlap upside ahead of M5d shipping.
        - Stress the comm stream / cp_group at realistic sizes (e.g. 1
          MB / layer) and surface any latent NCCL configuration issues
          (NVSwitch contention, channel exhaustion) before M5d activates
          the active-KV path.
        - Provide an apples-to-apples baseline for "what would real M5d
          cost" without yet touching the attention compute (which would
          require exact-token CP=2 validation).

        The payload is allocated as uint8 zeros for ``payload_bytes`` not
        equal to the heartbeat size; the broadcast publishes uninitialized
        bytes (since the receiver discards the payload in the M5d-partial
        posture anyway). Once M5d wires the real active-KV slice the
        payload will be a view over the cache pool rather than a separate
        allocation.

        Returns None on the disabled / non-CUDA / single-CP path so the
        caller short-circuits before the broadcast call.
        """
        if not self.enabled or self.cp_size <= 1:
            return None
        if torch is None or not torch.cuda.is_available():
            return None
        if channel not in _VALID_CHANNELS:
            raise ValueError(f"unknown layersplit channel {channel!r}; "
                             f"expected one of {_VALID_CHANNELS}")

        # Production decode (M5d) broadcasts the real indexer-K cache slot
        # directly via maybe_broadcast_for_layer, so this heartbeat path
        # is only used by tests + the M9 bench scaffold (which need a
        # synthetic payload of configurable size).
        def _alloc() -> Any:
            if payload_bytes is None or payload_bytes <= 16:
                return torch.zeros(4, dtype=torch.int32, device="cuda")
            return torch.zeros(int(payload_bytes),
                               dtype=torch.uint8,
                               device="cuda")

        if layer_idx is None:
            # Legacy shared payload (M5c sync-broadcast path). One
            # allocation reused across every layer and channel; safe
            # because the M5c caller never has two broadcasts in flight.
            if self._heartbeat_payload is None:
                self._heartbeat_payload = _alloc()
            return self._heartbeat_payload

        key = (layer_idx, channel)
        if key not in self._per_layer_payloads:
            self._per_layer_payloads[key] = _alloc()
        return self._per_layer_payloads[key]

    def prefetch_for_layer(self,
                           layer_idx: int,
                           payload: Optional[Any] = None,
                           cp_group: Optional[Any] = None,
                           channel: str = "kv") -> bool:
        """Kick off the per-layer broadcast on the comm stream and record
        a CUDA event so a later ``wait_for_prefetched_layer`` can have the
        default stream wait on it.

        This is the M6 cross-layer overlap primitive: the indexer hook for
        layer L issues a ``prefetch_for_layer(L+1)`` while layer L's own
        pre-indexer projection + sparse-attention indexer runs on the
        default stream. By the time layer L+1's hook fires, the broadcast
        has either already finished (zero wait) or is mostly done (small
        wait), so the cross-layer broadcast cost is amortized into the
        layer-L compute window.

        Returns True iff the broadcast was actually issued (caller may
        skip the symmetric ``wait_for_prefetched_layer`` if False). Same
        no-op cases as ``maybe_broadcast_for_layer``.
        """
        if not self.enabled or self.ownership is None:
            return False
        if self.cp_size <= 1 or cp_group is None or payload is None:
            return False
        if channel not in _VALID_CHANNELS:
            raise ValueError(f"unknown layersplit channel {channel!r}; "
                             f"expected one of {_VALID_CHANNELS}")
        if torch is None or not torch.cuda.is_available():
            return False
        try:
            import torch.distributed as dist
        except ImportError:
            return False
        if not dist.is_available() or not dist.is_initialized():
            return False

        src_rank = self.broadcast_src_rank(layer_idx)
        stream = self._stream_for_channel(channel)
        if stream is None:
            # Fall back to a synchronous broadcast on the default stream;
            # there's no side stream to record an event on.
            dist.broadcast(payload,
                           src=src_rank,
                           group=cp_group,
                           async_op=False)
            return True
        with torch.cuda.stream(stream):
            dist.broadcast(payload,
                           src=src_rank,
                           group=cp_group,
                           async_op=True)
            event = torch.cuda.Event()
            event.record(stream)
        self._prefetched_events[(layer_idx, channel)] = event
        return True

    def wait_for_prefetched_layer(self,
                                  layer_idx: int,
                                  channel: str = "kv") -> bool:
        """Have the default (current) stream wait on the comm-stream event
        recorded by an earlier ``prefetch_for_layer(layer_idx)``.

        Returns True if a prefetched event was found and waited on
        (caller should NOT re-issue a sync broadcast); returns False if no
        prefetch was in flight for this layer (caller should fall back to
        a synchronous broadcast).
        """
        if channel not in _VALID_CHANNELS:
            raise ValueError(f"unknown layersplit channel {channel!r}; "
                             f"expected one of {_VALID_CHANNELS}")
        event = self._prefetched_events.pop((layer_idx, channel), None)
        if event is None:
            return False
        if torch is None or not torch.cuda.is_available():
            return True  # nothing to wait on, but treat as handled
        torch.cuda.current_stream().wait_event(event)
        return True

    def clear_prefetched_events(self) -> None:
        """Drop any pending prefetched events without waiting (test/tear-down).

        Used by tests and by error-recovery paths that abort a partial
        forward; production decode steps consume every event in order via
        ``wait_for_prefetched_layer`` so the dict empties naturally.
        """
        self._prefetched_events.clear()

    def maybe_broadcast_active_blocks(
            self,
            layer_idx: int,
            cache_slot: Optional[Any],
            active_block_ids: Optional[Any],
            cp_group: Optional[Any] = None) -> bool:
        """M5e: gather, broadcast, and scatter back the *active* blocks of
        layer L's cache pool — strictly stronger than M5d's
        broadcast-the-whole-pool-slot because it cuts the bytes-on-the-wire
        from the full per-layer pool size (~870 MB / layer at V3.2 long
        context) down to just the blocks touched by the current step's
        scatter (~2 MB / layer at decode batch=256).

        Owner-side:
        - Gathers ``cache_slot[active_block_ids]`` into a contiguous send
          buffer (one ``torch.index_select`` — vectorized GPU op, no host
          sync).
        - Calls ``dist.broadcast`` with ``src=owner_rank``; the buffer
          content is published to every peer in the cp_group.
        - Scatters the send buffer back into ``cache_slot[active_block_ids]``
          — a no-op write of the same bytes for the owner.

        Receiver-side:
        - Gathers ``cache_slot[active_block_ids]`` into a contiguous send
          buffer (the receiver's own stale content).
        - Calls ``dist.broadcast`` — the buffer is OVERWRITTEN in place
          with the owner's content.
        - Scatters the overwritten buffer back into
          ``cache_slot[active_block_ids]`` — this is the cache update.

        Returns True iff the broadcast was issued. No-ops on the disabled
        / cp_size=1 / no-group / no-CUDA / dist-not-initialized / None-args
        branches so this is safe to drop in unconditionally.
        """
        if not self.enabled or self.ownership is None:
            return False
        if self.cp_size <= 1 or cp_group is None:
            return False
        if cache_slot is None or active_block_ids is None:
            return False
        if torch is None or not torch.cuda.is_available():
            return False
        try:
            import torch.distributed as dist
        except ImportError:
            return False
        if not dist.is_available() or not dist.is_initialized():
            return False
        if active_block_ids.numel() == 0:
            return False

        src_rank = self.broadcast_src_rank(layer_idx)
        send_buffer = cache_slot.index_select(0, active_block_ids).contiguous()
        dist.broadcast(send_buffer,
                       src=src_rank,
                       group=cp_group,
                       async_op=False)
        cache_slot.index_copy_(0, active_block_ids, send_buffer)
        return True

    def maybe_broadcast_active_blocks_fused(
            self,
            layer_idx: int,
            cache_slots,
            active_block_ids: Optional[Any],
            cp_group: Optional[Any] = None) -> bool:
        """M5g: per-layer system-level fusion — pack N cache pools
        (e.g. the indexer-K pool + the dense KV pool) into a SINGLE
        contiguous send buffer and issue ONE ``dist.broadcast`` instead
        of N. Cuts NCCL launches per layer from N to 1; at the V3.2
        shape with the indexer + dense pair this halves per-layer
        broadcast launch count (and at 61 layers per forward step
        eliminates 61 NCCL launches per step).

        ``cache_slots`` is an iterable of ``(num_blocks, row_bytes_i)``
        tensors that share the first dimension (the block id) but may
        have different second-dim sizes. The helper gathers each into
        a contiguous (num_active, row_bytes_i) view, concatenates them
        along dim 1 into one (num_active, sum(row_bytes_i)) send buffer,
        broadcasts, then splits + scatters each chunk back to its
        respective cache pool.

        Returns True iff the broadcast was issued. Same no-op guards
        as ``maybe_broadcast_active_blocks``. Falls back gracefully
        when ``cache_slots`` is empty or contains None entries (those
        entries are skipped).
        """
        if not self.enabled or self.ownership is None:
            return False
        if self.cp_size <= 1 or cp_group is None:
            return False
        if active_block_ids is None:
            return False
        if torch is None or not torch.cuda.is_available():
            return False
        try:
            import torch.distributed as dist
        except ImportError:
            return False
        if not dist.is_available() or not dist.is_initialized():
            return False
        if active_block_ids.numel() == 0:
            return False

        # Filter None entries; keep originals for the scatter-back step.
        slots = [(idx, s) for idx, s in enumerate(cache_slots)
                 if s is not None]
        if not slots:
            return False

        # Gather each pool's active rows into a contiguous 2-D view.
        gathered = []
        row_widths = []
        for _, slot in slots:
            flat = slot if slot.dim() == 2 else slot.view(slot.shape[0], -1)
            g = flat.index_select(0, active_block_ids).contiguous()
            gathered.append(g)
            row_widths.append(g.shape[1])

        # Concatenate along the row dimension into ONE send buffer +
        # one broadcast — the system-level fusion.
        send_buffer = torch.cat(gathered, dim=1).contiguous()
        src_rank = self.broadcast_src_rank(layer_idx)
        dist.broadcast(send_buffer,
                       src=src_rank,
                       group=cp_group,
                       async_op=False)

        # Split + scatter back into each cache pool.
        offset = 0
        for (_, slot), width in zip(slots, row_widths):
            chunk = send_buffer[:, offset:offset + width].contiguous()
            offset += width
            flat = slot if slot.dim() == 2 else slot.view(slot.shape[0], -1)
            flat.index_copy_(0, active_block_ids, chunk)
            # When we reshaped above, ``flat`` is a view of ``slot`` so
            # the index_copy_ already updated ``slot`` in place.
        return True

    @classmethod
    def disabled(cls) -> "LayerSplitRuntimeState":
        """Inert state for the LayerSplit-off path."""
        return cls(
            enabled=False,
            ownership=None,
            transfer_backend="auto",
            all_cp_ranks_transfer=True,
            cp_size=1,
            cp_rank=0,
            comm_stream=None,
        )

    @classmethod
    def from_sparse_config(
            cls,
            sparse_attn_config: Any,
            num_layers: int,
            cp_size: int,
            cp_rank: int = 0,
            create_comm_stream: Optional[bool] = None,
            create_indexer_comm_stream: Optional[bool] = None,
    ) -> "LayerSplitRuntimeState":
        """Build runtime state from a ``SparseAttentionConfig`` instance.

        ``create_comm_stream`` defaults to "yes if torch.cuda is available"
        so unit tests on a CPU-only host can disable it explicitly.

        ``create_indexer_comm_stream`` defaults to "track create_comm_stream"
        so production constructs both streams by default and the unit
        tests get both streams disabled together. When True an additional
        ``torch.cuda.Stream`` is allocated for the "indexer" channel (M8b)
        so the small indexer broadcast doesn't queue behind the larger
        dense KV broadcast on the primary comm stream.
        """
        enabled = bool(
            getattr(sparse_attn_config, "layersplit_enabled", False))
        if not enabled:
            return cls.disabled()

        policy = str(
            getattr(sparse_attn_config, "layersplit_owner_assignment",
                    "round_robin"))
        transfer_backend = str(
            getattr(sparse_attn_config, "layersplit_transfer_backend", "auto"))
        all_cp_ranks_transfer = bool(
            getattr(sparse_attn_config, "layersplit_all_cp_ranks_transfer",
                    True))
        if transfer_backend not in _VALID_TRANSFER_BACKENDS:
            raise ValueError(
                f"unknown layersplit_transfer_backend {transfer_backend!r}; "
                f"expected one of {_VALID_TRANSFER_BACKENDS}")
        if not all_cp_ranks_transfer:
            raise ValueError(
                "layersplit_all_cp_ranks_transfer=False is not yet "
                "implemented at the DSA runtime layer; the SparseAttentionConfig "
                "validator should have rejected this earlier.")

        ownership = compute_owner_assignment(num_layers, cp_size, policy)

        if create_comm_stream is None:
            create_comm_stream = (torch is not None
                                  and torch.cuda.is_available())
        comm_stream = (torch.cuda.Stream()
                       if (create_comm_stream and torch is not None
                           and torch.cuda.is_available()) else None)

        if create_indexer_comm_stream is None:
            create_indexer_comm_stream = create_comm_stream
        indexer_comm_stream = (torch.cuda.Stream()
                               if (create_indexer_comm_stream
                                   and torch is not None
                                   and torch.cuda.is_available()) else None)

        return cls(
            enabled=True,
            ownership=ownership,
            transfer_backend=transfer_backend,
            all_cp_ranks_transfer=all_cp_ranks_transfer,
            cp_size=cp_size,
            cp_rank=cp_rank,
            comm_stream=comm_stream,
            indexer_comm_stream=indexer_comm_stream,
        )

    def _stream_for_channel(self, channel: str) -> Optional[Any]:
        """Return the comm stream dedicated to ``channel`` if any. Falls
        back to the primary ``comm_stream`` when the indexer stream is
        not allocated (single-stream posture)."""
        if channel == "indexer" and self.indexer_comm_stream is not None:
            return self.indexer_comm_stream
        return self.comm_stream

    def is_owner(self, layer_idx: int) -> bool:
        if not self.enabled:
            return True  # everyone owns every layer in the off path
        assert self.ownership is not None
        return self.ownership.is_owner(layer_idx, self.cp_rank)

    def owner_of(self, layer_idx: int) -> int:
        if not self.enabled:
            return 0
        assert self.ownership is not None
        return self.ownership.owner_of(layer_idx)

    def layersplit_noop_transfer(self, layer_idx: int) -> None:
        """Graph-stable no-op transfer placeholder.

        Records an event on the default stream and waits on the comm stream
        — the exact synchronization shape a real owner-to-peers broadcast
        would have, with no data movement. Used by callers that want the
        comm-stream sync without engaging the broadcast collective (e.g.
        the LayerSplit-off path, the cp_size=1 path, or unit tests where
        no process group is constructed).
        """
        del layer_idx  # only used by future milestones for keying state
        if not self.enabled:
            return
        if self.comm_stream is None or torch is None:
            return
        if not torch.cuda.is_available():
            return
        event = torch.cuda.Event()
        event.record()
        self.comm_stream.wait_event(event)

    def maybe_broadcast_for_layer(
            self,
            layer_idx: int,
            payload: Optional[Any] = None,
            cp_group: Optional[Any] = None,
            async_op: bool = True,
            channel: str = "kv") -> None:
        """Broadcast a per-layer payload from the owner CP rank to every CP
        peer, matching the z.ai "Scaling Pain" §4 LayerSplit broadcast
        protocol. Peers will receive into ``payload`` (which must already
        be a torch tensor of the right shape on every rank); the owner's
        tensor content is published. The broadcast runs on the dedicated
        comm stream (created at runtime-state init) so that M6 can overlap
        it with the indexer / sparse-attention compute on the default
        stream.

        Returns immediately as a no-op in any of these cases:

        - LayerSplit is disabled.
        - ``cp_size <= 1``: there are no peers to broadcast to.
        - ``cp_group`` is None: the caller has not wired the CP process
          group through yet (M5 callers may stage the hook in
          progressively; the method must not blow up before the group is
          available).
        - ``payload`` is None: M5a callers pass None to exercise the
          sync-shape (`layersplit_noop_transfer`) without paying for an
          NCCL call. M5b passes a real KV slice tensor.
        - ``torch.distributed`` is unavailable or the runtime is not
          initialized (CPU-only tests, doc generation).

        The owner rank for ``layer_idx`` is read from ``self.ownership``
        — so the cache descriptor (M4), the C++ allocation mask (M4), and
        this broadcast all agree on which rank owns the layer.
        """
        if not self.enabled or self.ownership is None:
            return
        if self.cp_size <= 1:
            return
        if channel not in _VALID_CHANNELS:
            raise ValueError(f"unknown layersplit channel {channel!r}; "
                             f"expected one of {_VALID_CHANNELS}")
        if cp_group is None or payload is None:
            # No group / no payload: defer to the sync-shape placeholder
            # so the consumer wait downstream still happens correctly.
            self.layersplit_noop_transfer(layer_idx)
            return
        if torch is None or not torch.cuda.is_available():
            return
        try:
            import torch.distributed as dist
        except ImportError:
            return
        if not dist.is_available() or not dist.is_initialized():
            return

        src_rank = self.broadcast_src_rank(layer_idx)
        stream = self._stream_for_channel(channel)
        if stream is not None:
            with torch.cuda.stream(stream):
                dist.broadcast(payload,
                               src=src_rank,
                               group=cp_group,
                               async_op=async_op)
        else:
            dist.broadcast(payload,
                           src=src_rank,
                           group=cp_group,
                           async_op=async_op)


def _balanced_layer_mask(num_layers: int, cp_size: int, cp_rank: int,
                         policy: str) -> list:
    """Build a layer_mask whose sum() is identical across every CP rank.

    The naive per-rank mask (just "True at indices this rank owns") has
    a sum that varies by ±1 across ranks when ``num_layers % cp_size != 0``
    — e.g. CP=2 with 61 layers gives rank 0 = 31 True, rank 1 = 30 True.
    The C++ ``KVCacheManager`` sizes ``num_blocks_per_layer`` as
    ``max_memory / (sum(layer_mask) × per_block_bytes)``, so ranks with
    different sums get different ``num_blocks`` — which makes the global
    block-id allocation diverge across ranks and breaks the
    block-id-based broadcast (block_X on the owner may map to a
    semantically-different position on the receiver).

    The fix is to pad both the mask length and per-rank True count out
    to ``ceil(num_layers / cp_size) * cp_size``: every rank has the same
    target sum (``target_per_rank = ceil(num_layers / cp_size)``); ranks
    that naturally own fewer layers fill the difference with phantom
    True slots at indices ``[num_layers, total_length)``. The model
    never references those layers (the actual layer-loop only iterates
    ``0..num_layers``), so the phantom slots are pure padding — they
    cost a small amount of allocated-but-unused memory in exchange for
    making ``num_blocks`` identical across ranks.

    For DeepSeek-V3.2's 61 layers on CP=2, the phantom slot count is
    exactly 1 (total length 62, target 31 per rank); on CP=4 it's 3
    (total 64, target 16 per rank); on CP=8 it's 3 (total 64, target
    8 per rank). The wasted memory is ~3 / 61 ≈ 5 % of one rank's
    pool, far smaller than the ~50–87 % savings from owner-local
    allocation itself.
    """
    target_per_rank = (num_layers + cp_size - 1) // cp_size  # ceil
    total_length = target_per_rank * cp_size
    ownership = compute_owner_assignment(num_layers, cp_size, policy)

    mask: list = []
    owned_so_far = 0
    for idx in range(total_length):
        if idx < num_layers:
            owns = ownership.is_owner(idx, cp_rank)
        else:
            # Phantom slot: fill it iff this rank still needs padding to
            # hit target_per_rank True count.
            owns = owned_so_far < target_per_rank
        if owns:
            owned_so_far += 1
        mask.append(owns)
    return mask


def build_layersplit_layer_mask(
        num_layers: int,
        sparse_attn_config: Optional[Any],
        cp_size: int,
        cp_rank: int = 0) -> Optional[list]:
    """Compute the per-layer ownership mask the KV cache manager passes to
    the C++ ``WindowBlockManager`` so that non-owner CP ranks skip pool
    allocation for non-owned layers (the M4 memory-savings step of the
    LayerSplit integration).

    Returns ``None`` when LayerSplit is disabled, when ``sparse_attn_config``
    is missing the LayerSplit fields, or when ``cp_size <= 1`` — those cases
    collapse to the regular all-layers-on-this-rank allocation path and the
    layer_mask machinery should stay out of them. When LayerSplit is on with
    ``cp_size > 1`` the result is a list of length ``num_layers`` where
    ``True`` marks layers this rank owns and ``False`` marks the rest. The
    caller (``_create_kv_cache_manager``) sums the mask to recompute
    ``num_hidden_layers`` for the cache manager constructor; the C++ side
    only allocates KV/indexer pools for the masked-in positions.

    Composes safely with the rest of TRT-LLM:
    - Returns ``None`` on the off path, so any other layer-mask consumer
      (e.g. KV sharing for Gemma4 hybrid, one-model draft KV separation)
      sees the same behavior as before.
    - Uses the same ``compute_owner_assignment`` that drives
      :class:`LayerSplitRuntimeState`; both code paths agree on which rank
      owns which layer, so the cache descriptor and the runtime broadcast
      sender will name the same owner for every layer.
    """
    if sparse_attn_config is None:
        return None
    if not bool(getattr(sparse_attn_config, "layersplit_enabled", False)):
        return None
    if cp_size <= 1:
        return None
    policy = str(
        getattr(sparse_attn_config, "layersplit_owner_assignment",
                "round_robin"))
    # M5d-tight-v2: return the *balanced* mask so every CP rank's
    # sum(layer_mask) is identical and the C++ KVCacheManager allocates
    # the same num_blocks on every rank — required for the global
    # block-id space to stay consistent across ranks so the M5e
    # active-block broadcast writes block_X on the receiver to the same
    # semantic position as the owner.
    return _balanced_layer_mask(num_layers=num_layers,
                                cp_size=cp_size,
                                cp_rank=cp_rank,
                                policy=policy)

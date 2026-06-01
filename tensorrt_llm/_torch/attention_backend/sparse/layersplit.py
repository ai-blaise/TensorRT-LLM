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
from typing import Any, Optional, Tuple

try:
    import torch
except ImportError:  # pragma: no cover - torch is always available in prod
    torch = None  # type: ignore[assignment]

_VALID_POLICIES = ("round_robin", "contiguous")
_VALID_TRANSFER_BACKENDS = ("auto", "ucx", "nixl")


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
    _heartbeat_payload: Optional[Any] = field(default=None,
                                              repr=False,
                                              init=False)

    def bind_cp_group(self, cp_group: Any) -> None:
        """Late-bind the CP process group resolved from ``mapping``.

        The DSACacheManager resolves the group eagerly at construction
        time when possible; tests that build the runtime state directly
        (without a mapping) can bind the group post-hoc.
        """
        self.cp_group = cp_group

    def ensure_heartbeat_payload(self) -> Optional[Any]:
        """Lazily allocate a tiny CUDA tensor used as the M5c heartbeat
        payload for the per-layer broadcast.

        The heartbeat is a small (4 × int32 = 16 B) tensor that the
        production Indexer hook publishes from the owner rank every layer
        every decode step. It exercises the NCCL broadcast path end-to-end
        so deployments with LayerSplit on and CP > 1 actually pay (and
        we measure) the per-layer collective cost. Once M5d plumbs the
        real active-KV slice this hook will be replaced with the KV
        payload; the heartbeat keeps the wiring under CI pressure in the
        meantime.

        Returns None on the disabled / non-CUDA / single-CP path so the
        caller short-circuits before the broadcast call.
        """
        if not self.enabled or self.cp_size <= 1:
            return None
        if torch is None or not torch.cuda.is_available():
            return None
        if self._heartbeat_payload is None:
            self._heartbeat_payload = torch.zeros(4,
                                                  dtype=torch.int32,
                                                  device="cuda")
        return self._heartbeat_payload

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
    ) -> "LayerSplitRuntimeState":
        """Build runtime state from a ``SparseAttentionConfig`` instance.

        ``create_comm_stream`` defaults to "yes if torch.cuda is available"
        so unit tests on a CPU-only host can disable it explicitly.
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

        return cls(
            enabled=True,
            ownership=ownership,
            transfer_backend=transfer_backend,
            all_cp_ranks_transfer=all_cp_ranks_transfer,
            cp_size=cp_size,
            cp_rank=cp_rank,
            comm_stream=comm_stream,
        )

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
            async_op: bool = True) -> None:
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

        src_rank = self.ownership.owner_of(layer_idx)
        if self.comm_stream is not None:
            with torch.cuda.stream(self.comm_stream):
                dist.broadcast(payload,
                               src=src_rank,
                               group=cp_group,
                               async_op=async_op)
        else:
            dist.broadcast(payload,
                           src=src_rank,
                           group=cp_group,
                           async_op=async_op)


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
    ownership = compute_owner_assignment(num_layers=num_layers,
                                         cp_size=cp_size,
                                         policy=policy)
    return [ownership.owner_of(layer) == cp_rank for layer in range(num_layers)]

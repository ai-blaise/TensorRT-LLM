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

from dataclasses import dataclass
from typing import Tuple

_VALID_POLICIES = ("round_robin", "contiguous")


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

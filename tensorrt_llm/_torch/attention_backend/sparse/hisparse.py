"""OP-TRT HiSparse coordinator scaffolding.

The production HiSparse path is intentionally fail-closed until the packed
dense-MLA KVarN host/hot tiers, swap-in kernel, and NIXL direct-to-host
writer are wired. This module gives DSA a stable extension point without
introducing an FP16 staging path or a silent full-HBM fallback.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class HiSparseTopKMapping:
    """Result of mapping request-relative TopK through the HiSparse hot pool."""

    topk_indices_global: "torch.Tensor"
    pool_view: Optional["torch.Tensor"] = None
    hot_block_ids: Optional["torch.Tensor"] = None


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

    def assert_startup_ready(self) -> None:
        if not self.enabled:
            return
        raise NotImplementedError(
            "HiSparse is enabled, but the production packed KVarN host/hot "
            "swap-in path is not complete. This is fail-closed by design: "
            "do not use an FP16 staging path, full-HBM fallback, or "
            "direct-to-host-off runtime for enabled HiSparse.")

    def reset_step(self) -> None:
        self.step_id += 1

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

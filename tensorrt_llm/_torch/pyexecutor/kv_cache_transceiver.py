from abc import ABC, abstractmethod
from os import getenv
from typing import Any, Dict, List, Optional

import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm.bindings import WorldConfig
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig
from tensorrt_llm.mapping import Mapping

from .llm_request import LlmRequest
from .mamba_cache_manager import (BaseMambaCacheManager,
                                  CppMambaHybridCacheManager)
from .resource_manager import KVCacheManager

CacheTransceiverCpp = tensorrt_llm.bindings.internal.batch_manager.CacheTransceiver
AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType
CacheTransBufferManagerCpp = tensorrt_llm.bindings.internal.batch_manager.CacheTransBufferManager
BackendTypeCpp = tensorrt_llm.bindings.executor.CacheTransceiverBackendType


_CACHE_TRANSCEIVER_ENV_BACKENDS = [
    ("TRTLLM_USE_NIXL_KVCACHE", "NIXL"),
    ("TRTLLM_USE_UCX_KVCACHE", "UCX"),
    ("TRTLLM_USE_MOONCAKE_KVCACHE", "MOONCAKE"),
    ("TRTLLM_USE_MPI_KVCACHE", "MPI"),
]


def _resolve_cache_transceiver_backend(
        cache_transceiver_config: CacheTransceiverConfig) -> str:
    """Resolve legacy env selectors without silently changing explicit YAML.

    The r20 NIXL gate must fail closed: an explicit YAML backend is the source
    of truth, and old TRTLLM_USE_*_KVCACHE toggles may not silently redirect it
    to UCX/Mooncake/MPI. The env selectors remain supported only for DEFAULT,
    where exactly one selector may choose an A/B backend.
    """
    explicit_backend = cache_transceiver_config.backend
    enabled_env_backends = [(name, backend)
                            for name, backend in _CACHE_TRANSCEIVER_ENV_BACKENDS
                            if getenv(name) == "1"]

    if explicit_backend == "DEFAULT":
        if len(enabled_env_backends) > 1:
            enabled = ", ".join(
                f"{name}={backend}" for name, backend in enabled_env_backends)
            raise RuntimeError(
                "cache_transceiver_config.backend=DEFAULT received multiple "
                f"TRTLLM_USE_*_KVCACHE selectors: {enabled}")
        if enabled_env_backends:
            env_var, backend = enabled_env_backends[0]
            logger.warning(
                f"{env_var}=1 is set, but explicit "
                "cache_transceiver_config.backend in YAML is preferred")
            return backend
        return "NIXL"

    conflicting_env_backends = [
        (name, backend) for name, backend in enabled_env_backends
        if backend != explicit_backend
    ]
    if conflicting_env_backends:
        enabled = ", ".join(
            f"{name}={backend}" for name, backend in conflicting_env_backends)
        raise RuntimeError(
            f"cache_transceiver_config.backend={explicit_backend} conflicts "
            f"with legacy env backend selector(s): {enabled}. Remove the env "
            "override or change YAML explicitly; implicit transport fallback is "
            "not allowed.")

    matching_env_backends = [name for name, backend in enabled_env_backends
                             if backend == explicit_backend]
    if matching_env_backends:
        logger.warning(
            "Ignoring redundant cache transceiver env selector(s) for explicit "
            f"backend {explicit_backend}: {matching_env_backends}")
    return explicit_backend


def _normalise_layersplit_total_kv_heads_per_layer(
        kv_cache_manager: KVCacheManager,
        total_num_kv_heads_per_layer: List[int]) -> List[int]:
    """Return the global attention-layer vector advertised to C++ transfer.

    Owner-local LayerSplit intentionally trims the local C++ KV/indexer/KVarN
    pools to the layers owned by this CP rank, but the disaggregated transfer
    metadata still needs the global layer domain. The C++ split/concat path
    uses mAttentionLayerNumPerPP for the local shard size and
    mNbKvHeadsPerLayer.size() for the model layer span; advertising only the
    local pool length makes MLA transfer reject TPxCP prefill -> TP decode
    handoff before the LayerSplit-aware concat path can run.
    """
    layersplit_state = getattr(kv_cache_manager, "layersplit_state", None)
    if not (layersplit_state is not None
            and getattr(layersplit_state, "enabled", False)
            and getattr(layersplit_state, "owner_local_alloc", False)):
        return total_num_kv_heads_per_layer

    ownership = getattr(layersplit_state, "ownership", None)
    global_num_layers = int(
        getattr(kv_cache_manager, "layersplit_model_num_layers", 0) or
        getattr(kv_cache_manager, "layersplit_cache_transfer_model_layers", 0) or
        getattr(ownership, "num_layers", 0) or 0)
    local_pool_layers = int(
        getattr(kv_cache_manager, "layersplit_local_pool_layers", 0) or
        len(total_num_kv_heads_per_layer))
    if global_num_layers <= 0:
        raise RuntimeError(
            "LayerSplit owner-local transfer requires a positive cache-transfer "
            "model layer count")
    if not total_num_kv_heads_per_layer:
        raise RuntimeError(
            "LayerSplit owner-local transfer cannot infer global KV-head "
            "metadata from an empty local layer vector")
    if len(total_num_kv_heads_per_layer) == global_num_layers:
        logger.info(
            "CacheTransceiver CacheState layers=%d local_pool_layers=%d",
            global_num_layers, local_pool_layers)
        return total_num_kv_heads_per_layer
    if len(total_num_kv_heads_per_layer) > global_num_layers:
        logger.warning(
            "LayerSplit owner-local cache-transfer trimming %d phantom "
            "CacheState layer(s): local_pool_layers=%d cache_state_layers=%d "
            "model_layers=%d",
            len(total_num_kv_heads_per_layer) - global_num_layers,
            local_pool_layers, len(total_num_kv_heads_per_layer),
            global_num_layers)
        return total_num_kv_heads_per_layer[:global_num_layers]
    if len(set(total_num_kv_heads_per_layer)) != 1:
        raise RuntimeError(
            "LayerSplit owner-local transfer requires an explicit global "
            "KV-head vector for heterogeneous per-layer heads")

    kv_heads = total_num_kv_heads_per_layer[0]
    logger.info(
        "Expanding LayerSplit owner-local transfer layer metadata from "
        f"{len(total_num_kv_heads_per_layer)} local layers to "
        f"{global_num_layers} global layers")
    return [kv_heads for _ in range(global_num_layers)]


def mapping_to_world_config(mapping: Mapping) -> WorldConfig:

    return WorldConfig(tensor_parallelism=mapping.tp_size,
                       pipeline_parallelism=mapping.pp_size,
                       context_parallelism=mapping.cp_size,
                       rank=mapping.rank,
                       gpus_per_node=mapping.gpus_per_node,
                       device_ids=None,
                       enable_attention_dp=mapping.enable_attention_dp)


def create_kv_cache_transceiver(
        mapping: Mapping,
        dist: Distributed,
        kv_cache_manager: KVCacheManager,
        attention_type: AttentionTypeCpp,
        cache_transceiver_config: CacheTransceiverConfig,
        mamba_cache_manager: Optional[BaseMambaCacheManager] = None):
    if cache_transceiver_config is None or cache_transceiver_config.backend is None:
        logger.info("cache_transceiver is disabled")
        return None

    cache_transceiver_config.backend = _resolve_cache_transceiver_backend(
        cache_transceiver_config)

    if cache_transceiver_config.backend == "MPI":
        logger.warning(
            "MPI CacheTransceiver is deprecated, UCX or NIXL is recommended")
    elif cache_transceiver_config.backend == "UCX":
        logger.info(
            f"Using UCX kv-cache transceiver. If your devices are not in the same domain, please consider setting "
            f"UCX_CUDA_IPC_ENABLE_MNNVL=n, UCX_RNDV_SCHEME=put_zcopy and/or unset UCX_NET_DEVICES upon server "
            f"hangs or lower-than-expected performance.")

    # Select transceiver implementation based on transceiver_runtime
    # transceiver_runtime == None or "CPP" -> use C++ transceiver (default)
    # transceiver_runtime == "PYTHON" -> use Python transceiver
    if cache_transceiver_config.transceiver_runtime == "PYTHON":
        # Python transceiver currently only supports NIXL and DEFAULT backend
        if cache_transceiver_config.backend not in ("DEFAULT", "NIXL"):
            raise ValueError(
                f"Python transceiver currently only supports NIXL or DEFAULT backend, "
                f"got {cache_transceiver_config.backend}. "
                f"Please use transceiver_runtime='CPP' for MPI, UCX, or MOONCAKE backends."
            )
        from tensorrt_llm._torch.disaggregation.transceiver import \
            KvCacheTransceiverV2
        logger.info("Using KvCacheTransceiverV2")
        return KvCacheTransceiverV2(mapping, dist, kv_cache_manager,
                                    cache_transceiver_config)

    # Default: use C++ transceiver (transceiver_runtime is None or "CPP")
    return BindKvCacheTransceiver(mapping, dist, kv_cache_manager,
                                  attention_type, cache_transceiver_config,
                                  mamba_cache_manager)


class KvCacheTransceiver(ABC):

    @abstractmethod
    def respond_and_send_async(self, req: LlmRequest):
        raise NotImplementedError

    @abstractmethod
    def request_and_receive_sync(self, req: LlmRequest):
        raise NotImplementedError

    @abstractmethod
    def request_and_receive_async(self, req: LlmRequest):
        raise NotImplementedError

    @abstractmethod
    def check_context_transfer_status(self, at_least_request_num: int):
        raise NotImplementedError

    @abstractmethod
    def check_gen_transfer_status(self, at_least_request_num: int):
        raise NotImplementedError

    @abstractmethod
    def check_gen_transfer_complete(self):
        raise NotImplementedError

    @abstractmethod
    def cancel_request(self, req: LlmRequest):
        raise NotImplementedError

    @abstractmethod
    def prepare_context_requests(self, requests: List[LlmRequest]):
        """
        Prepare the context request for the cache transceiver in generation-first mode.
        This method should set the context request state to DISAGG_CONTEXT_WAIT_SCHEDULER
        so that it won't be scheduled if the responding generation kvcache request is not
        yet received otherwise set it to CONTEXT_INIT.
        """
        ...

    @abstractmethod
    def get_disaggregated_params(self) -> Dict[str, Any]:
        """
        Return a dictionary form of DisaggregatedParams to be set in the generation request.
        The generation server will use it to get kvcache in generation-first mode.
        """
        ...

    def commit_blocks_for_reuse(self, req: LlmRequest) -> None:
        """Commit received KV blocks to the radix tree for prefix reuse. No-op by default."""

    def shutdown(self):
        """Shut down the transceiver and release registered resources."""


class BindKvCacheTransceiver(KvCacheTransceiver):

    def __init__(self,
                 mapping: Mapping,
                 dist: Distributed,
                 kv_cache_manager: KVCacheManager,
                 attention_type: AttentionTypeCpp,
                 cache_transceiver_config: CacheTransceiverConfig,
                 mamba_cache_manager: Optional[BaseMambaCacheManager] = None):
        world_config = mapping_to_world_config(mapping)
        # Filter out mamba/recurrent state layers (kv_heads == 0) so that
        # CacheState::ModelConfig::mNbKvHeadsPerLayer only contains attention
        # layers — matching the factory path (modelConfig.getNumKvHeadsPerLayer()).
        # This is critical: splitKVCacheDispatch uses mNbKvHeadsPerLayer.size()
        # as the layer count for the CUDA kernel grid dimension.
        layersplit_transfer_heads = getattr(
            kv_cache_manager, "layersplit_transfer_num_kv_heads_per_layer",
            None)
        if layersplit_transfer_heads is not None:
            total_num_kv_heads_per_layer = [
                h for h in layersplit_transfer_heads if h > 0
            ]
        else:
            total_num_kv_heads_per_layer = [
                h for h in kv_cache_manager.total_num_kv_heads_per_layer
                if h > 0
            ]
            total_num_kv_heads_per_layer = \
                _normalise_layersplit_total_kv_heads_per_layer(
                    kv_cache_manager, total_num_kv_heads_per_layer)
        head_dim = kv_cache_manager.head_dim
        tokens_per_block = kv_cache_manager.tokens_per_block
        dtype = kv_cache_manager.dtype
        # Get the *attention* layer count per PP rank (C++ uses this as
        # mAttentionLayerNumPerPP).  For CppMambaHybridCacheManager the local
        # pp_layers list includes mamba layers (kv_heads == 0); those must be
        # excluded so the C++ buffer-size calculations stay correct.
        pp_layer_num = sum(1 for h in kv_cache_manager.num_kv_heads_per_layer
                           if h > 0)
        pp_layer_num_per_pp_rank = dist.pp_allgather(pp_layer_num)
        logger.info(
            "CacheTransceiver transfer model config: global_attention_layers=%d, "
            "local_attention_layers=%d, local_pool_layers=%d, "
            "pp_layer_num_per_pp_rank=%s, tp=%d, cp=%d, attention_dp=%s",
            len(total_num_kv_heads_per_layer), pp_layer_num,
            len(getattr(kv_cache_manager, 'num_kv_heads_per_layer', [])),
            pp_layer_num_per_pp_rank, mapping.tp_size, mapping.cp_size,
            mapping.enable_attention_dp)
        print(
            "OPTRT_LAYERSPLIT_XFER_DEBUG "
            f"manager={type(kv_cache_manager).__name__} "
            f"transfer_attr={layersplit_transfer_heads is not None} "
            f"global_layers={len(total_num_kv_heads_per_layer)} "
            f"local_attention_layers={pp_layer_num} "
            f"local_pool_layers={len(getattr(kv_cache_manager, 'num_kv_heads_per_layer', []))} "
            f"pp_layers={pp_layer_num_per_pp_rank} "
            f"tp={mapping.tp_size} cp={mapping.cp_size} "
            f"attention_dp={mapping.enable_attention_dp} "
            f"layersplit_model={getattr(kv_cache_manager, 'layersplit_model_num_layers', None)} "
            f"layersplit_transfer_model={getattr(kv_cache_manager, 'layersplit_cache_transfer_model_layers', None)}",
            flush=True)

        self.kv_transfer_timeout_ms = cache_transceiver_config.kv_transfer_timeout_ms
        self.kv_transfer_sender_future_timeout_ms = cache_transceiver_config.kv_transfer_sender_future_timeout_ms

        # Get RNN state manager and layer distribution if mamba_cache_manager is provided.
        rnn_state_manager = None
        rnn_layer_num_per_pp_rank = []
        if mamba_cache_manager is not None:
            if isinstance(mamba_cache_manager, CppMambaHybridCacheManager):
                # Unified pool path: RNN model config is in LinearAttentionMetadata,
                # C++ reads it from BlockManager during CacheTransceiver construction.
                rnn_layer_num_per_pp_rank = dist.pp_allgather(
                    mamba_cache_manager.local_num_mamba_layers)
            else:
                rnn_state_manager = mamba_cache_manager._impl.mamba_impl
                # Get the number of local RNN layers and allgather across PP ranks
                rnn_local_layer_num = rnn_state_manager.get_num_local_layers()
                rnn_layer_num_per_pp_rank = dist.pp_allgather(
                    rnn_local_layer_num)
                logger.info(
                    f"RNN state transfer enabled: rnn_layer_num_per_pp={rnn_layer_num_per_pp_rank}"
                )

        self.impl = CacheTransceiverCpp(
            kv_cache_manager.impl, total_num_kv_heads_per_layer, head_dim,
            tokens_per_block, world_config,
            pp_layer_num_per_pp_rank, dtype, attention_type,
            cache_transceiver_config._to_pybind(), rnn_state_manager,
            rnn_layer_num_per_pp_rank)

    def respond_and_send_async(self, req: LlmRequest):
        return self.impl.respond_and_send_async(req)

    def request_and_receive_sync(self, req: LlmRequest):
        return self.impl.request_and_receive_sync(req)

    def request_and_receive_async(self, req: LlmRequest):
        return self.impl.request_and_receive_async(req)

    def check_context_transfer_status(self, at_least_request_num: int):
        return self.impl.check_context_transfer_status(at_least_request_num)

    def check_gen_transfer_status(self, at_least_request_num: int):
        return self.impl.check_gen_transfer_status(at_least_request_num)

    def check_gen_transfer_complete(self):
        return self.impl.check_gen_transfer_complete()

    def cancel_request(self, req: LlmRequest):
        return self.impl.cancel_request(req)

    def prepare_context_requests(self, requests: List[LlmRequest]):
        # not implemented, an empty placeholder to allow being invoked unconditionally
        ...

    def get_disaggregated_params(self):
        # Cpp kv cache transceiver will set the disaggregated params to context response
        # Only new py cache transceiver will support gen-first disagg
        return {}


class CacheTransBufferManager:

    def __init__(self, kv_cache_manager: KVCacheManager, max_num_tokens: int):
        self.impl = CacheTransBufferManagerCpp(kv_cache_manager.impl,
                                               max_num_tokens)

    @staticmethod
    def pre_alloc_buffer_size(
            kv_cache_size_bytes_per_token_per_window: dict[int, int],
            tokens_per_block: int,
            cache_transceiver_config: CacheTransceiverConfig):
        return CacheTransBufferManagerCpp.pre_alloc_buffer_size(
            kv_cache_size_bytes_per_token_per_window, tokens_per_block,
            cache_transceiver_config)

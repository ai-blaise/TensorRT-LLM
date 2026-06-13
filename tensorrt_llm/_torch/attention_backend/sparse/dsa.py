"""Dense Sparse Attention (DSA) backend for TRT-LLM with indexer-based TopK selection."""
import functools
import math
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs, AttentionInputType, MLAParams,
    PositionalEmbeddingParams)
from tensorrt_llm._torch.attention_backend.sparse.layersplit import (
    LayerSplitOwnership, LayerSplitRuntimeState, ensure_cp_process_group)
from tensorrt_llm._torch.attention_backend.sparse.hisparse import (
    OPTRTHiSparseCoordinator)
from tensorrt_llm._torch.attention_backend.sparse.kvarn_backend import (
    KVarNLatentPool, KVARN_LEGACY_SIDEPOOL_LAYOUT,
    kvarn_latent_bytes_per_token, resolve_kvarn_config)


def _layersplit_compute_active_block_ids(metadata):
    """M5e: compute the unique block ids touched by THIS step's scatter.

    Given the DSA attention metadata, return the int64 GPU tensor of
    unique block ids that were (or are about to be) written by the
    indexer-K scatter for the current batch. The LayerSplit hook gathers
    those blocks into a send buffer, broadcasts owner -> peers, and
    scatters back — broadcasting only the active blocks rather than the
    full per-layer pool slot cuts the bytes on the wire by 100×-1000×
    at typical batch / context sizes.

    For request ``i`` the new tokens span positions
    ``[kv_lens[i] - seq_lens[i], kv_lens[i])`` so the block range is
    ``[(kv_lens[i] - seq_lens[i]) // tokens_per_block,
       (kv_lens[i] - 1) // tokens_per_block]`` inclusive. We materialize
    the union of these ranges per request via vectorized masking against
    the request's ``block_table`` row, then ``torch.unique`` over the
    concatenated block ids.

    Returns ``None`` on any path that can't compute the set (no
    block_table, no kv_lens, num_seqs=0, missing kv_cache_manager) so
    the hook's broadcast helper short-circuits to a no-op rather than
    publishing garbage.
    """
    if metadata is None:
        return None
    kv_lens = getattr(metadata, "kv_lens", None)
    seq_lens = getattr(metadata, "seq_lens", None)
    block_table = getattr(metadata, "block_table", None)
    num_seqs = getattr(metadata, "num_seqs", 0)
    kv_cache_manager = getattr(metadata, "kv_cache_manager", None)
    if (kv_lens is None or seq_lens is None or block_table is None
            or num_seqs <= 0 or kv_cache_manager is None):
        return None
    tokens_per_block = getattr(kv_cache_manager, "tokens_per_block", None)
    if tokens_per_block is None or tokens_per_block <= 0:
        return None

    # Per-request block index ranges, on the same device as block_table.
    device = block_table.device
    kv_lens_slice = kv_lens[:num_seqs].to(device=device, dtype=torch.int64)
    seq_lens_slice = seq_lens[:num_seqs].to(device=device, dtype=torch.int64)
    end_block_in_seq = (kv_lens_slice - 1) // tokens_per_block  # (num_seqs,)
    start_block_in_seq = (kv_lens_slice - seq_lens_slice) // tokens_per_block
    # Clamp negatives that arise when seq_lens > kv_lens (shouldn't happen
    # in well-formed metadata but be defensive — a negative start would
    # silently include extra blocks).
    start_block_in_seq = torch.clamp_min(start_block_in_seq, 0)

    max_blocks_per_seq = block_table.shape[1]
    block_arange = torch.arange(max_blocks_per_seq,
                                device=device,
                                dtype=torch.int64).unsqueeze(0)  # (1, B)
    mask = ((block_arange >= start_block_in_seq.unsqueeze(1)) &
            (block_arange <= end_block_in_seq.unsqueeze(1)))  # (S, B)
    table_slice = block_table[:num_seqs].to(dtype=torch.int64)
    selected = table_slice[mask]  # 1-D, may include -1 padding
    selected = selected[selected >= 0]
    if selected.numel() == 0:
        return None
    return torch.unique(selected)


def _layersplit_compute_read_block_ids(metadata):
    """Compute the unique block ids the indexer-K kernel READS this step.

    Companion to :func:`_layersplit_compute_active_block_ids`. The active
    (write) set is only the blocks the current step SCATTERS into
    (``[kv_lens - seq_lens, kv_lens)`` per request). But the DSA indexer
    scores each query token causally against the FULL per-request KV
    prefix ``[0, kv_len[i])`` (confirmed: ``prepare_one_prefill_chunk``
    gathers indexer-K over ``[host_ctx_kv_indptr[req], +num_cached+chunk]``
    and ``cu_seqlen_ks`` starts at the request KV base, ``cu_seqlen_ke``
    runs to ``num_cached + local_q_pos + 1``), so under chunked prefill /
    prefix reuse the cached prefix blocks ``[0, num_cached)`` are read but
    never appear in the write set. A write-set-only broadcast leaves the
    non-owner's indexer-K scratch STALE for the prefix on chunk >= 2,
    yielding wrong logits / TopK.

    This returns the READ set: the union over requests of the block_table
    rows for the full range ``[0, kv_len[i])`` — i.e. exactly
    :func:`_layersplit_compute_active_block_ids` with
    ``start_block_in_seq = 0``. The owner broadcasts these so the
    non-owner presents a complete prefix to the (CP-unaware) indexer.

    Returns ``None`` on any path that can't compute the set, so the hook's
    broadcast helper short-circuits to a no-op rather than publishing
    garbage.
    """
    if metadata is None:
        return None
    kv_lens = getattr(metadata, "kv_lens", None)
    block_table = getattr(metadata, "block_table", None)
    num_seqs = getattr(metadata, "num_seqs", 0)
    kv_cache_manager = getattr(metadata, "kv_cache_manager", None)
    if (kv_lens is None or block_table is None or num_seqs <= 0
            or kv_cache_manager is None):
        return None
    tokens_per_block = getattr(kv_cache_manager, "tokens_per_block", None)
    if tokens_per_block is None or tokens_per_block <= 0:
        return None

    device = block_table.device
    kv_lens_slice = kv_lens[:num_seqs].to(device=device, dtype=torch.int64)
    # Full prefix [0, kv_len) -> blocks [0, (kv_lens - 1) // tpb] inclusive.
    end_block_in_seq = (kv_lens_slice - 1) // tokens_per_block  # (num_seqs,)

    max_blocks_per_seq = block_table.shape[1]
    block_arange = torch.arange(max_blocks_per_seq,
                                device=device,
                                dtype=torch.int64).unsqueeze(0)  # (1, B)
    mask = block_arange <= end_block_in_seq.unsqueeze(1)  # (S, B)
    table_slice = block_table[:num_seqs].to(dtype=torch.int64)
    selected = table_slice[mask]  # 1-D, may include -1 padding
    selected = selected[selected >= 0]
    if selected.numel() == 0:
        return None
    return torch.unique(selected)


def _layersplit_readset_hoist_enabled() -> bool:
    """L2 gate: memoize the LayerSplit indexer-K READ set once per step
    instead of recomputing it inside every layer's Indexer.forward (61x at
    DSV3.2). Default on; set TRTLLM_OPTRT_LAYERSPLIT_READSET_HOIST=0 to
    recompute per layer (pre-hoist behavior, identical sets either way)."""
    return os.environ.get("TRTLLM_OPTRT_LAYERSPLIT_READSET_HOIST",
                          "1") != "0"


def _layersplit_prefill_overlap_enabled() -> bool:
    """L1 gate: z.ai dense-broadcast overlap on pure-context steps.

    Indexer.forward issues the dense KV (+ NVFP4 scale) READ-SET broadcast
    on the LayerSplit comm stream right after the indexer-K broadcast, so
    it hides behind the indexer scoring; sparse_attn_predict waits on the
    comm-stream event just before the dense read instead of running the
    per-layer top-k-union broadcast (whose masked_select + unique host
    syncs dominate the sync path's exposed cost: measured 71.5 -> 6.9 ms
    exposed per 61-layer prefill step at kv=64k on the CP2 worker, equal
    wire bytes, byte-identical consumer-visible rows). Default on; set
    TRTLLM_OPTRT_LAYERSPLIT_PREFILL_OVERLAP=0 for the legacy sync path."""
    return os.environ.get("TRTLLM_OPTRT_LAYERSPLIT_PREFILL_OVERLAP",
                          "1") != "0"


def _layersplit_read_block_ids_step(metadata):
    """Per-step memo of :func:`_layersplit_compute_read_block_ids` (L2).

    The read set is a pure function of step-level metadata — host kv_lens
    values, block_table, num_seqs, tokens_per_block — none of which vary
    across the indexer layers of one forward step, yet every layer's
    Indexer.forward re-ran the full computation (H2D kv_lens copy, range
    mask, padding filter, torch.unique — the last two host-sync on their
    data-dependent output shapes). The first indexer layer of a step
    computes the set; the remaining layers reuse the tensor. The broadcast
    consumer only reads it (index_select / index_copy_ index argument), so
    cross-layer aliasing is safe.

    Correctness across steps rests on the same two legs as
    ``_hisa_step_invariants``: prepare() / on_update_kv_lens() /
    update_for_spec_dec() clear the slot every step, and the memo key
    revalidates. The key here is strictly stronger than the HISA one: it
    carries the host kv_lens VALUES, so chunked-prefill steps of the same
    request (same buffers and num_seqs, advanced kv progress — the read
    set legitimately grows chunk to chunk) and prefix-cache-hit admissions
    can never alias even if a clear were missed. The capture flag mirrors
    the template; in practice this path is always eager — batches with
    context requests never run under CUDA graphs, and the unique() output
    shape is data-dependent.

    Falls back to the plain per-layer compute when the gate is off, the
    metadata is unusable for keying, or kv_lens is not host-resident
    (fingerprinting a device tensor would itself sync).
    """
    if metadata is None or not _layersplit_readset_hoist_enabled():
        return _layersplit_compute_read_block_ids(metadata)
    kv_lens = getattr(metadata, "kv_lens", None)
    block_table = getattr(metadata, "block_table", None)
    num_seqs = getattr(metadata, "num_seqs", 0)
    if (kv_lens is None or block_table is None or num_seqs <= 0
            or kv_lens.device.type != "cpu"):
        return _layersplit_compute_read_block_ids(metadata)
    key = (kv_lens.data_ptr(), block_table.data_ptr(), int(num_seqs),
           tuple(kv_lens[:num_seqs].tolist()),
           torch.cuda.is_current_stream_capturing())
    cached = getattr(metadata, "_layersplit_step_read_set", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    read_set = _layersplit_compute_read_block_ids(metadata)
    metadata._layersplit_step_read_set = (key, read_set)
    return read_set


def _layersplit_topk_global_block_ids(topk_indices_global, stride_factor):
    """Map global-token TopK indices -> the unique global BLOCK ids they hit.

    ``transform_local_topk_reuse_or_compute`` (via convert_req_index_to_global)
    returns global *token* indices into the flat (block, layer, token) pool:
    ``g = base * stride_factor + (layer_id * block_size + tok % block_size)``
    where ``base`` is the global block id (block_table entry) and
    ``stride_factor = num_layers * tokens_per_block``. The in-block-plus-layer
    offset is strictly ``< stride_factor``, so ``g // stride_factor == base``
    recovers the global block id that indexes the dense KV / scale pool slot
    (``cache_slot[block_id]``) — exactly the index space
    ``maybe_broadcast_active_blocks`` scatters into. Padding entries (-1) are
    dropped. Returns ``None`` (broadcast no-ops) when nothing valid remains.

    This is the dense-KV READ set for the step: the union of every query
    token's TopK-selected blocks (prefill TopK is per-query-token, so the
    union spans all of them).
    """
    if (topk_indices_global is None or stride_factor is None
            or stride_factor <= 0 or topk_indices_global.numel() == 0):
        return None
    flat = topk_indices_global.reshape(-1)
    flat = flat[flat >= 0]
    if flat.numel() == 0:
        return None
    block_ids = (flat.to(torch.int64) // int(stride_factor))
    return torch.unique(block_ids)
from tensorrt_llm._torch.attention_backend.trtllm import (
    TrtllmAttention, TrtllmAttentionMetadata)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.distributed.ops import allgather
from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.multi_stream_utils import \
    maybe_execute_in_parallel
from tensorrt_llm._torch.modules.rotary_embedding import RotaryEmbedding
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.utils import maybe_compile
from tensorrt_llm._utils import get_size_in_bytes, get_sm_version, prefer_pinned
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.bindings.internal.batch_manager import \
    CacheType as CacheTypeCpp
from tensorrt_llm.deep_gemm import (fp8_fp4_mqa_logits,
                                    fp8_fp4_paged_mqa_logits, fp8_mqa_logits,
                                    fp8_paged_mqa_logits,
                                    get_paged_mqa_logits_metadata)
from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

ModelConfig = tensorrt_llm.bindings.ModelConfig

if TYPE_CHECKING:
    from tensorrt_llm._torch.speculative.interface import SpecMetadata
    from tensorrt_llm._torch.speculative.spec_tree_manager import SpecTreeManager
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig

# Optional import: fast-hadamard-transform causes CI build issues (requires wheel+torch pre-installed)
try:
    from fast_hadamard_transform import hadamard_transform
    HAS_FAST_HADAMARD = True
except ImportError:
    hadamard_transform = None
    HAS_FAST_HADAMARD = False

# Idempotency guard for warmup_heuristic_topk_decode — keyed by
# (device_index, top_k, hint_size, num_cols). Prevents repeated allocations
# and synchronizations when multiple Indexer modules invoke the warmup with
# the same parameters during model construction.
_HEURISTIC_TOPK_WARMUP_DONE: Set[Tuple[int, int, int, int,
                                       torch.dtype]] = set()
_HEURISTIC_TOPK_WARMUP_LOCK = threading.Lock()


def warmup_heuristic_topk_decode(top_k: int = 2048,
                                 hint_size: int = 2048,
                                 num_cols: int = 4096,
                                 logits_dtype: torch.dtype = torch.float32
                                 ) -> None:
    """Pre-initialize cached hardware attributes in the C++ Scheme X dispatcher.

    The dispatcher inside ``invokeIndexerTopKDecode`` lazily queries
    ``cudaDeviceGetAttribute`` for ``MultiProcessorCount`` and
    ``L2CacheSize`` on its first call. Those host-side queries must not
    be issued during ``cudaStreamBeginCapture / EndCapture``: the values
    captured there become frozen into the graph and cannot be refreshed
    across replays on a different device.

    This warmup issues one small heuristic decode call so the static
    caches are populated before any CUDA Graph capture begins. Must be
    called from the Indexer setup hook (``layer_idx == 0``) when
    ``enable_heuristic_topk`` is true.

    Repeated invocations with the same ``(device, top_k, hint_size,
    num_cols, logits_dtype)`` key are short-circuited so that constructing
    many Indexer modules in the same process does not re-allocate scratch
    tensors or issue redundant synchronizations.

    ``logits_dtype`` must match the dtype the decode pipeline feeds the
    C++ Top-K (see ``resolve_indexer_logits_dtype``); the kernel requires
    its heuristic scratch dtype to equal its logits dtype.
    """
    key = (torch.cuda.current_device(), top_k, hint_size, num_cols,
           logits_dtype)
    with _HEURISTIC_TOPK_WARMUP_LOCK:
        if key in _HEURISTIC_TOPK_WARMUP_DONE:
            return
        _HEURISTIC_TOPK_WARMUP_DONE.add(key)

    device = torch.device("cuda")
    logits = torch.zeros((1, num_cols), dtype=logits_dtype, device=device)
    seq_lens = torch.tensor([num_cols], dtype=torch.int32, device=device)
    indices = torch.empty((1, top_k), dtype=torch.int32, device=device)
    pre_idx = torch.zeros((1, hint_size), dtype=torch.int32, device=device)
    scratch = torch.empty((top_k, ), dtype=logits_dtype, device=device)
    torch.ops.trtllm.indexer_topk_decode(logits,
                                         seq_lens,
                                         indices,
                                         1,
                                         top_k,
                                         pre_idx=pre_idx,
                                         heuristic_scratch=scratch)
    torch.cuda.synchronize()


# `block_kv` arg passed to DeepGEMM's `get_paged_mqa_logits_metadata`. This is
# a SCHEDULE-granularity parameter, not the cache page size. The DG metadata
# kernel computes `SPLIT_KV = block_kv * 4` (the multiplier 4 is hardcoded;
# DeepGEMM commit 7f2a703 dropped the SM100-aware `arch == 10 ? 2 : 4` from
# nv_dev #fc97232 during the Public release 26/04 sync, leaving the formula
# uniform across SM90/SM100). Both the DG compute kernel (which hardcodes
# `split_kv = 256` at csrc/apis/attention.hpp:353) and our DSL paged-MQA-logits
# kernel (compute tile = 128 × kNumMathWarpGroups = 2 = 256) require
# `SPLIT_KV = 256`. So we must pass `block_kv = 256 / 4 = 64` here. Independent
# of the indexer K cache's physical page size (`tokens_per_block`), which only
# affects cache reads inside the compute kernel — the metadata wrapper does
# not read the cache. Passing the cache page size directly (the previous
# behavior) only works by accident when `tokens_per_block == 64`; for
# `tokens_per_block == 32` it produces a SPLIT_KV=128 schedule that the
# compute kernels misinterpret. TODO(remove once DeepGEMM restores the
# SM100-aware num_math_warpgroups in the metadata JIT impl).
_DG_SCHEDULE_BLOCK_KV = 64

# NVFP4 sparse-MLA decode tile-scheduler metadata hoist (candidate SM1).
# The `sparse_mla_decode_nvfp4` op recomputes its tile-scheduler metadata inside
# EVERY F-layer call (16/step). With prod topk=1024 and topk_length=None the
# metadata is data-independent — a pure function of (batch_size, s_q, topk) plus
# compile-time scheduler constants — so it is bit-identical across all 16 layers
# and every step at a fixed batch bucket (see `get_decoding_sched_meta.cu`:
# `cur_s_k = topk` when `topk != -1`). When enabled, layer 0 lets the op compute
# the metadata and we cache its returned tensors into the pre-allocated
# graph-stable buffers below; layers 1..15 pass those buffers back in so the op
# skips the serial metadata kernel (`computeSchedulerMetadata=false`).
# Gated OFF by default → byte-identical to the per-layer recompute.
# `DecodingSchedMeta` is 8 int32s (params.h); `kMaxNvfp4NumSmParts` (= 4096,
# sparse_mla_decode_nvfp4.cu) caps `num_sm_parts`, so the worst-case metadata
# buffer is [4096, 8] int32 — the analogue of the bf16 FlashMLA `sm_count * 8`.
_SPARSE_MLA_META_WIDTH = 8
_SPARSE_MLA_MAX_NUM_SM_PARTS = 4096


def _hoist_sparse_mla_meta_enabled() -> bool:
    """Whether to hoist the NVFP4 sparse-MLA tile-scheduler metadata.

    Gated by ``TRTLLM_OPTRT_HOIST_SPARSE_MLA_META`` (default ``"0"`` = OFF). OFF
    keeps the current per-F-layer recompute (each call passes
    ``tile_scheduler_metadata=None``), which is byte-identical to the historical
    behavior. ON computes the metadata once per step and reuses it across the 16
    F-layers.
    """
    return os.environ.get("TRTLLM_OPTRT_HOIST_SPARSE_MLA_META", "1") == "1"


# B200 guardrail for the scalar HISA block-score scorer. It only wins
# small block-count shapes; larger contexts use the DeepGEMM FP4 scorer.
_HISA_FUSED_BLOCK_SCORE_MAX_BLOCKS = 64


@functools.lru_cache(maxsize=1)
def _hisa_perrow_cand() -> bool:
    """H3b gate: when set, the HISA candidate-score GEMM and candidate top-k are
    driven by each row's live candidate width instead of the uniform band ceiling
    `candidate_len`, so each row walks only its live extent (and short rows flip the
    top-k from radix to the cheaper insertion final-sort). Candidate buffer
    *allocations* stay graph-stable; only the per-row length arguments shrink.
    Default off (0) = the uniform `full(candidate_len)` behavior, byte-identical."""
    return os.environ.get("TRTLLM_OPTRT_HISA_PERROW_CAND", "0") == "1"


def _hisa_step_memo_enabled() -> bool:
    """Gate for the per-step memo of layer-invariant HISA decode index math
    (row_to_batch / prefix_lens / block_counts / row spans). Default on; set
    TRTLLM_OPTRT_HISA_STEP_MEMO=0 to recompute per layer (pre-memo behavior,
    byte-identical math either way)."""
    return os.environ.get("TRTLLM_OPTRT_HISA_STEP_MEMO", "1") != "0"


@functools.lru_cache(maxsize=1)
def _hoist_hisa_sched() -> bool:
    """SM3 gate: when set, the HISA candidate-score schedule
    (`get_paged_mqa_logits_metadata`) is built ONCE per decode step in the
    metadata prepare path instead of being rebuilt inside every recompute-"F"
    indexer layer (16x/step at prod). The candidate-score GEMM only consumes the
    schedule; the schedule is a pure function of `candidate_context_lens`,
    `_DG_SCHEDULE_BLOCK_KV`, and `num_sms`, none of which change across the F
    layers of a single step. Hoisting only redistributes where that build runs
    (1x vs 16x), not what it computes. The per-layer consumer revalidates a small
    `(num_rows, next_n, candidate_len)` provenance signature before substituting
    the prebuilt buffer and otherwise falls back to the in-line rebuild, so the
    result is bit-identical whether on or off. Default off (0) = the in-line
    per-layer rebuild, byte-identical."""
    return os.environ.get("TRTLLM_OPTRT_HOIST_HISA_SCHED", "1") == "1"


def _hisa_block_topk_value(num_blocks: int, index_topk: int, hisa_block_size: int,
                           hisa_compression_ratio: float,
                           hisa_block_topk: int) -> int:
    """Free-function mirror of ``Indexer._hisa_block_topk`` for the once-per-step
    schedule build in the metadata prepare path (which has no ``Indexer`` handle,
    only ``sparse_attention_config`` scalars). MUST stay identical to the method
    at the class so the prebuilt schedule's ``candidate_len`` matches the
    per-layer value exactly."""
    min_blocks = math.ceil(index_topk / hisa_block_size)
    if hisa_compression_ratio > 0:
        block_topk = math.ceil(num_blocks / hisa_compression_ratio)
    else:
        block_topk = hisa_block_topk
    return min(max(block_topk, min_blocks), num_blocks)


def _build_hisa_candidate_schedule(sparse_attention_config, kv_lens_gen: torch.Tensor,
                                   next_n: int, num_generations: int,
                                   max_gen_kv_len: int, num_sms: int):
    """Build the HISA candidate-score schedule once per decode step (SM3 hoist).

    Replicates the EXACT length math from ``_hisa_topk_from_nvfp4_cache`` so the
    returned schedule is bit-identical to the per-layer rebuild:

      max_blocks   = ceil(max_kv_len / hisa_block_size)              (== per-layer)
      block_topk   = _hisa_block_topk(max_blocks)                    (== per-layer)
      candidate_len= block_topk * hisa_block_size                    (== per-layer)
      prefix_lens  = kv_lens[row // next_n] - next_n + row % next_n + 1
      candidate_context_lens = H3b-perrow-clamped count   (or full(candidate_len))

    ``max_gen_kv_len`` is the host int already computed sync-free in
    ``DSAtrtllmAttentionMetadata.prepare`` (== ``int(kv_lens.max().item())`` over
    the gen slice), which equals the per-layer eager ``max_kv_len`` because the
    HISA decode path runs in ``mla_dsa_attn_inplace`` (excluded from CUDA-graph
    capture), so the per-layer code always takes its eager ``kv_lens.max()`` branch.

    Returns ``(schedule, sig)`` where ``sig = (num_rows, next_n, candidate_len)``
    for the consumer's provenance check, or ``None`` when HISA would not run for
    this shape (so the consumer falls back to its own path)."""
    hisa_block_size = getattr(sparse_attention_config, "hisa_block_size", 128)
    index_topk = sparse_attention_config.index_topk
    hisa_compression_ratio = getattr(sparse_attention_config,
                                     "hisa_compression_ratio", 0)
    hisa_block_topk = getattr(sparse_attention_config, "hisa_block_topk", 1)

    if max_gen_kv_len <= 0 or next_n <= 0 or num_generations <= 0:
        return None
    max_blocks = math.ceil(max_gen_kv_len / hisa_block_size)
    if max_blocks <= 0:
        return None
    block_topk = _hisa_block_topk_value(max_blocks, index_topk, hisa_block_size,
                                        hisa_compression_ratio, hisa_block_topk)
    candidate_len = block_topk * hisa_block_size
    # Same early-out as the per-layer path (`candidate_len < index_topk` returns
    # None there): a too-short candidate band means HISA pre-indexer is skipped.
    if candidate_len < index_topk:
        return None

    num_rows = num_generations * next_n
    device = kv_lens_gen.device
    row_idx = torch.arange(num_rows, device=device)
    row_to_batch = torch.div(row_idx, next_n, rounding_mode="floor")
    row_offset = row_idx % next_n
    prefix_lens = (kv_lens_gen[row_to_batch] - next_n + row_offset + 1).to(
        torch.int32)
    if _hisa_perrow_cand():
        block_counts_row = torch.div(prefix_lens + hisa_block_size - 1,
                                     hisa_block_size,
                                     rounding_mode="floor")
        cand_count = (block_counts_row.clamp_max(block_topk) *
                      hisa_block_size).to(torch.int32)
        candidate_context_lens = cand_count.view(num_rows, 1)
    else:
        candidate_context_lens = torch.full((num_rows, 1),
                                            candidate_len,
                                            dtype=torch.int32,
                                            device=device)
    schedule = get_paged_mqa_logits_metadata(candidate_context_lens,
                                             _DG_SCHEDULE_BLOCK_KV, num_sms)
    return schedule, (num_rows, next_n, candidate_len)


# Decode top-k kernel crossover on LIVE KV length (B200, index_topk=1024,
# CUDA-graph replay, 3-seed, prod width 132096). The C++ indexer_topk_decode
# walks only [0, live_kv) per row (cpp/.../indexerTopK.cu topKPerRowJob), so its
# latency is ~independent of the padded logits width and grows ~linearly in live
# kv. The CuTe DSL kernel runs a kv-parallel select whose cost scales with the
# padded WIDTH. Measured at prod width 132096: C++ is ~1.7x FASTER at live kv
# ~4608 (~11us vs ~19us); the paths cross at ~12-16K live kv; the (cluster) DSL
# kernel wins by ~1.3-1.4x only at long live kv (>= ~16K up to 131072). Both
# produce identical selected sets (recall 1.0). Below this threshold prefer C++.
_DSL_TOPK_MIN_KV_LEN = 16384

# Width-correct the decode logits buffer to a power-of-2 bucket of the real
# max-kv per graph band. With the top-k dispatch keyed on live kv only (no
# consumer keys on the logits width anymore), the bucket is NOT a latency
# lever: measured on B200 (2026-06-11, live kv 4608, B=4/16, fp16 logits +
# C++ top-k, CUDA-graph replay, interleaved widths) the scoring+top-k
# pipeline is flat (~15.4us) from width 8192 to 132096 and logits[:, :kv) is
# bitwise identical -- the scoring kernel walks ceil(kv/block_kv) tiles from
# the context_lens schedule and the C++ top-k walks [0, live_kv), so the
# [kv, width) pad is never written or read; width is only the output row
# stride. The bucket survives as the graph-safety upper bound for a band's
# replays plus a captured-allocation trim (rows x width x 2B fp16, e.g.
# 4.2MB -> 0.26MB at 16 rows) when an operator opts into a short band above
# index_topk. The bucket is a captured constant under CUDA graphs: at capture
# max_gen_kv_len equals the per-graph warmup kv (the upper bound of that
# graph's short/long seq-len band), so every replay's real kv <= the captured
# width and the topk seq_lens mask the [kv, width) tail. Power-of-2 bucketing
# keeps the captured-width set small and stable across batches.


def _indexer_logits_width(max_gen_kv_len: int, hard_cap: int) -> int:
    """Power-of-2 bucket of the live max-kv for the decode logits buffer,
    clamped to ``hard_cap`` (== kv_cache_manager.max_seq_len).

    Graph-safety upper-bound guarantee. Under CUDA graphs this value is
    frozen at capture, where ``max_gen_kv_len`` equals the per-graph warmup
    kv. A warmup request reserves a few fewer tokens than the band ceiling
    (``max_seq_len - 1 - num_extra_kv_tokens - draft``), so a live replay at
    the true ceiling can exceed the warmup kv by that small epsilon. To keep
    the captured width a safe upper bound for every replay we collapse to
    ``hard_cap`` for any kv in the top bucket band (``> hard_cap // 2``);
    the long-sequence graph (warmup kv ~= hard_cap) therefore always keeps
    its full max_model_len width (no narrowing, no regression, no tail
    miss). Only genuinely short bands (kv <= hard_cap // 2) are narrowed,
    where the next-pow2 bucket strictly exceeds max_gen_kv_len and thus
    bounds the whole band.
    """
    if max_gen_kv_len <= 0 or hard_cap <= 0:
        return hard_cap
    if max_gen_kv_len > hard_cap // 2:
        return hard_cap
    bucket = 1
    while bucket < max_gen_kv_len:
        bucket <<= 1
    return bucket if bucket < hard_cap else hard_cap


_INDEXER_LOGITS_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def resolve_indexer_logits_dtype(sparse_attention_config) -> torch.dtype:
    """Decode-logits element type for the DSL score -> Top-K pipeline.

    The indexer scoring inputs are already fp8/fp4 quantized (see
    ``indexer_k_dtype``), so fp32 logits carry far more precision than the
    scores contain. 16-bit logits halve the logits store/load traffic and
    cut the radix Top-K from 4 rounds (fp32) to 2; the scoring kernels keep
    fp32 accumulation and only convert at the epilogue store. ``auto``
    resolves to fp16: 3 extra mantissa bits over bf16 give measurably
    better top-1024 selection fidelity, and the score magnitudes (weighted
    ReLU sums over 64 heads) sit orders of magnitude below the fp16 range.

    Only the CuTe DSL paged-MQA-logits kernels can emit 16-bit logits; the
    DeepGEMM fallback emits fp32 only, so everything resolves to fp32 when
    that path is active. This keeps the heuristic-topk scratch dtype (which
    the C++ Top-K requires to match its logits input) consistent with the
    logits actually produced.
    """
    if not (getattr(sparse_attention_config, "use_cute_dsl_paged_mqa_logits",
                    False) and IS_CUTLASS_DSL_AVAILABLE):
        return torch.float32
    choice = getattr(sparse_attention_config, "indexer_logits_dtype", "auto")
    if choice == "auto":
        return torch.float16
    return _INDEXER_LOGITS_DTYPE_MAP[choice]


def _pick_dsl_expand(
    next_n: int,
    num_sms: int,
    batch_size: int = 0,
    max_ctx: int = 0,
    kernel_atoms: Tuple[int, ...] = (1, 2, 3)) -> Tuple[int, int]:
    """Pick (expand_factor, effective_next_n) for the DSL paged kernel
    using a wave-aware strategy. Used by both FP4 and FP8 DSL paths.

    The DSL kernel natively supports ``effective_next_n ∈ kernel_atoms``
    (FP4: ``(1, 2, 3)``; FP8: ``(1, 2, 3, 4)``). For ``next_n`` not natively
    supported or when SM utilization can be improved, reshape
    ``[B, next_n, ...]`` -> ``[B * expand_factor, effective_next_n, ...]``
    caller-side.

    Strategy: enumerate ``(expand_factor, effective_next_n)`` pairs with
    ``expand_factor * effective_next_n == next_n`` and ``effective_next_n
    in kernel_atoms``. Score each by ``(waves, -expand_factor)`` where
    ``waves = ceil(B * expand_factor * ceil(max_ctx/256) / num_sms)``.
    Pick min waves; on tie, prefer LARGER expand_factor (more SMs busy per
    wave; pays HBM cost of expand_factor x KV re-reads).

    When ``batch_size == 0`` or ``max_ctx == 0`` (workload unknown), fall
    back to the legacy HBM-minimizing heuristic: largest effective_next_n
    that divides next_n cleanly (still constrained to ``kernel_atoms``).

    Examples (wave-aware, num_sms=148 [B200], SPLIT_KV=256 tokens):
        FP4, next_n=4, B=1,  ctx=4096   -> (4, 1): ntask=64<148, 1 wave, max factor
        FP4, next_n=4, B=32, ctx=4096   -> (2, 2): ntask=1024>148, multi-wave, min factor
        FP4, next_n=2, B=1,  ctx=4096   -> (2, 1): wave-tie, larger factor
        FP8, next_n=4, B=1,  ctx=4096   -> (4, 1): kernel_atoms incl. 4 doesn't change small-B pick
    """
    # Legacy fallback when workload is unknown.
    if batch_size <= 0 or max_ctx <= 0:
        for eff in sorted(kernel_atoms, reverse=True):
            if next_n % eff == 0:
                return next_n // eff, eff
        return next_n, 1

    SPLIT_KV_TOKENS = 256
    cands = []
    for eff in kernel_atoms:
        if next_n % eff == 0:
            factor = next_n // eff
            ntask = batch_size * factor * (
                (max_ctx + SPLIT_KV_TOKENS - 1) // SPLIT_KV_TOKENS)
            waves = (ntask + num_sms - 1) // num_sms
            cands.append((waves, factor, eff))
    if not cands:
        return next_n, 1
    cands.sort(key=lambda x: (x[0], -x[1]))  # min waves, max factor
    _, factor, eff = cands[0]
    return factor, eff


def _compute_slot_mappings(
    global_positions: torch.Tensor,
    block_offsets: torch.Tensor,
    req_indices: torch.Tensor,
    head_dim: int,
    tokens_per_block: int,
    quant_block_size: int,
    data_bytes_per_token: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute flat byte indices for indexer K data and scales from global token positions.

    Shared by Indexer.prepare() (CPU) and on_update_kv_lens() (GPU) to avoid
    duplicating the slot mapping arithmetic.

    Args:
        global_positions: Per-token absolute position in the KV sequence.
        block_offsets: [num_seqs, max_blocks_per_seq] block offset table.
        req_indices: Per-token request index.
        head_dim: Indexer head dimension (used for the scale-size formula).
        tokens_per_block: Tokens stored per cache block.
        quant_block_size: Quantization block size.
        data_bytes_per_token: Bytes of quantized data per token in the cache
            pool. FP8 stores one byte per element (= head_dim). FP4 packs two
            E2M1 codes per byte (= head_dim // 2). Defaults to ``head_dim``
            when unset, preserving the FP8 layout for callers that haven't
            threaded the FP4 dtype through.

    Returns:
        (fp8_indices, scale_indices): Flat byte offsets into the cache pool.
    """
    if data_bytes_per_token is None:
        data_bytes_per_token = head_dim
    scale_size = head_dim // quant_block_size * 4  # float32 = 4 bytes
    block_stride = tokens_per_block * (data_bytes_per_token + scale_size)
    scale_base_offset = tokens_per_block * data_bytes_per_token

    block_indices_in_seq = global_positions // tokens_per_block
    pos_in_blocks = global_positions % tokens_per_block

    max_blocks = block_offsets.shape[1]
    if block_indices_in_seq.is_cuda:
        # Clamp to prevent OOB from stale token-to-seq mappings during
        # CUDA graph capture/replay with MTP + DSA.
        block_indices_in_seq = block_indices_in_seq.clamp(0, max_blocks - 1)
    else:
        assert (block_indices_in_seq < max_blocks).all(), \
            f"Block index out of bounds: max={max_blocks}, got indices up to {block_indices_in_seq.max().item()}"

    block_ids = block_offsets[req_indices, block_indices_in_seq].to(torch.int64)

    fp8_indices = block_ids * block_stride + pos_in_blocks * data_bytes_per_token
    scale_indices = (block_ids * block_stride + scale_base_offset +
                     pos_in_blocks * scale_size)
    return fp8_indices, scale_indices


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard rotation to activation tensor for DSA sparse attention."""
    assert x.dtype == torch.bfloat16

    if not HAS_FAST_HADAMARD:
        # Fallback: skip transformation (acceptable for test/dev)
        logger.warning_once(
            "fast-hadamard-transform not available. DSA sparse attention will skip "
            "hadamard transformation. Install with: "
            "pip install git+https://github.com/Dao-AILab/fast-hadamard-transform.git",
            key="fast_hadamard_import_missing")
        return x

    hidden_size = x.size(-1)
    assert (hidden_size & (hidden_size - 1)
            ) == 0, "Hidden size must be a power of 2 for Hadamard transform."
    return hadamard_transform(x, scale=hidden_size**-0.5)


def transform_local_topk_and_prepare_pool_view(
    topk_indices: torch.Tensor,
    attn_metadata: "DSAtrtllmAttentionMetadata",
    layer_idx: int,
    is_generation: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert local topk indices to global pool indices and prepare KV pool.

    Uses cached values from attn_metadata._ensure_pool_view_cached()
    to avoid redundant Python/CUDA overhead across layers.
    """
    assert topk_indices.dtype == torch.int32

    attn_metadata._ensure_pool_view_cached()

    if is_generation:
        block_table = attn_metadata._cached_block_table_gen
        req_idx = attn_metadata._cached_req_idx_gen
    else:
        block_table = attn_metadata._cached_block_table_ctx
        req_idx = attn_metadata._cached_req_idx_ctx

    global_indices = torch.ops.trtllm.convert_req_index_to_global(
        req_idx,
        block_table,
        topk_indices,
        attn_metadata._cached_tokens_per_block,
        topk_indices.shape[1],
        attn_metadata._cached_stride_factor,
        layer_idx,
    )

    return global_indices, attn_metadata._cached_pool_view


# JIT fallback for trtllm::fused_rope_cat_fp4. The AOT C++ op
# (cpp/tensorrt_llm/thop/fusedRopeCatFp4Op.cpp + kernels/fusedRopeCatFp4.cu) is
# the production path; when the loaded .so predates it we register an equivalent
# load_inline kernel under the same qualified name so the fused RoPE+cat+FP4
# proj path runs without a rebuild. Output is bit-identical to flashinfer RoPE
# followed by fused_cat_fp4.
_FUSED_ROPE_CAT_FP4_READY = False
_FUSED_ROPE_CAT_FP4_LOCK = threading.Lock()


def _ensure_fused_rope_cat_fp4_op():
    global _FUSED_ROPE_CAT_FP4_READY
    if _FUSED_ROPE_CAT_FP4_READY:
        return
    with _FUSED_ROPE_CAT_FP4_LOCK:
        if _FUSED_ROPE_CAT_FP4_READY:
            return
        if hasattr(torch.ops.trtllm, "fused_rope_cat_fp4"):
            _FUSED_ROPE_CAT_FP4_READY = True
            return
        from torch.utils.cpp_extension import load_inline
        cuda_src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cstdint>
namespace {
constexpr int HEAD_DIM=128; constexpr int WARP_SIZE=32; constexpr int ELEMS_PER_THREAD=4;
constexpr int ROWS_PER_BLOCK=8; constexpr float INV_FP4_E2M1_MAX=1.0f/6.0f; constexpr float MIN_AMAX=1.0e-12f;
union BF16x4 { int2 vec; __nv_bfloat162 bf16x2[2]; };
__device__ __forceinline__ uint32_t qFp4(float scaled){
  float ax=fminf(fabsf(scaled),6.0f);
  uint32_t idx=(uint32_t)((ax>0.25f)+(ax>0.75f)+(ax>1.25f)+(ax>1.75f)+(ax>2.5f)+(ax>3.5f)+(ax>5.0f));
  uint32_t code=idx&0x7u; uint32_t sign=(scaled<0.0f&&idx!=0u)?1u:0u; return code|(sign<<3);
}
__global__ __launch_bounds__(WARP_SIZE*ROWS_PER_BLOCK)
void frcKernel(int8_t* __restrict__ packed_out,int32_t* __restrict__ scale_out,
  __nv_bfloat16 const* __restrict__ pe,__nv_bfloat16 const* __restrict__ nope,
  float const* __restrict__ cos_sin,int32_t const* __restrict__ pos,
  int32_t M,int32_t pe_dim,int32_t nope_dim,int32_t pe_rs,int32_t nope_rs,int32_t cs_s){
  int w=threadIdx.x/WARP_SIZE,lane=threadIdx.x%WARP_SIZE,row=blockIdx.x*ROWS_PER_BLOCK+w;
  if(row>=M) return;
  int rope_half=pe_dim>>1, plo=rope_half/ELEMS_PER_THREAD, base=lane*ELEMS_PER_THREAD;
  float v0,v1,v2,v3;
  bool from_pe=(base<pe_dim);
  if(from_pe){
    __nv_bfloat16 const* pr=pe+(int64_t)row*pe_rs;
    BF16x4 sl; sl.vec=*reinterpret_cast<int2 const*>(pr+base);
    float2 s0=__bfloat1622float2(sl.bf16x2[0]),s1=__bfloat1622float2(sl.bf16x2[1]);
    float a0=s0.x,a1=s0.y,a2=s1.x,a3=s1.y;
    bool lower=(base<rope_half); int pl=lower?(lane+plo):(lane-plo);
    unsigned pe_mask=(1u<<(pe_dim/ELEMS_PER_THREAD))-1u;
    float p0=__shfl_sync(pe_mask,a0,pl),p1=__shfl_sync(pe_mask,a1,pl),
          p2=__shfl_sync(pe_mask,a2,pl),p3=__shfl_sync(pe_mask,a3,pl);
    float const* cs=cos_sin+(int64_t)pos[row]*cs_s; int ci=lower?base:(base-rope_half);
    float c0=cs[ci],c1=cs[ci+1],c2=cs[ci+2],c3=cs[ci+3];
    float n0=cs[rope_half+ci],n1=cs[rope_half+ci+1],n2=cs[rope_half+ci+2],n3=cs[rope_half+ci+3];
    float sg=lower?-1.0f:1.0f;
    v0=a0*c0+sg*p0*n0; v1=a1*c1+sg*p1*n1; v2=a2*c2+sg*p2*n2; v3=a3*c3+sg*p3*n3;
    v0=__bfloat162float(__float2bfloat16(v0)); v1=__bfloat162float(__float2bfloat16(v1));
    v2=__bfloat162float(__float2bfloat16(v2)); v3=__bfloat162float(__float2bfloat16(v3));
  } else {
    __nv_bfloat16 const* nr=nope+(int64_t)row*nope_rs; int col=base-pe_dim;
    BF16x4 ld; ld.vec=*reinterpret_cast<int2 const*>(nr+col);
    float2 f0=__bfloat1622float2(ld.bf16x2[0]),f1=__bfloat1622float2(ld.bf16x2[1]);
    v0=f0.x;v1=f0.y;v2=f1.x;v3=f1.y;
  }
  float lm=fmaxf(fmaxf(fabsf(v0),fabsf(v1)),fmaxf(fabsf(v2),fabsf(v3)));
  float amax=lm;
  amax=fmaxf(amax,__shfl_xor_sync(0xFFFFFFFFu,amax,1));
  amax=fmaxf(amax,__shfl_xor_sync(0xFFFFFFFFu,amax,2));
  amax=fmaxf(amax,__shfl_xor_sync(0xFFFFFFFFu,amax,4));
  amax=fmaxf(amax,MIN_AMAX);
  float ratio=amax*INV_FP4_E2M1_MAX; uint32_t bits=__float_as_uint(ratio);
  uint32_t eb=bits&0x7F800000u; if((bits&0x007FFFFFu)!=0u) eb+=0x00800000u;
  float scale=__uint_as_float(eb);
  uint32_t c0=qFp4(v0/scale),c1=qFp4(v1/scale),c2=qFp4(v2/scale),c3=qFp4(v3/scale);
  uint8_t b0=(uint8_t)(c0|(c1<<4)),b1=(uint8_t)(c2|(c3<<4));
  int bo=row*(HEAD_DIM/2)+lane*2; packed_out[bo]=(int8_t)b0; packed_out[bo+1]=(int8_t)b1;
  uint32_t me=(__float_as_uint(scale)>>23)&0xFFu;
  uint32_t e0=__shfl_sync(0xFFFFFFFFu,me,0),e1=__shfl_sync(0xFFFFFFFFu,me,8),
           e2=__shfl_sync(0xFFFFFFFFu,me,16),e3=__shfl_sync(0xFFFFFFFFu,me,24);
  if(lane==0) scale_out[row]=(int32_t)(e0|(e1<<8)|(e2<<16)|(e3<<24));
}
}
std::tuple<at::Tensor,at::Tensor> fused_rope_cat_fp4(at::Tensor const& pe, at::Tensor const& nope,
    at::Tensor const& cos_sin, at::Tensor const& pos){
  c10::cuda::CUDAGuard g{pe.device()};
  auto pd=(int32_t)pe.size(-1), nd=(int32_t)nope.size(-1); auto M=(int32_t)(pe.numel()/pd);
  auto packed=at::empty({M,(pd+nd)/2}, pe.options().dtype(at::kChar));
  auto scale=at::empty({M,1}, pe.options().dtype(at::kInt));
  auto st=at::cuda::getCurrentCUDAStream(pe.get_device());
  int nb=(M+ROWS_PER_BLOCK-1)/ROWS_PER_BLOCK;
  frcKernel<<<nb,WARP_SIZE*ROWS_PER_BLOCK,0,st>>>(
    reinterpret_cast<int8_t*>(packed.data_ptr()),reinterpret_cast<int32_t*>(scale.data_ptr()),
    reinterpret_cast<__nv_bfloat16 const*>(pe.data_ptr()),reinterpret_cast<__nv_bfloat16 const*>(nope.data_ptr()),
    cos_sin.data_ptr<float>(),pos.data_ptr<int32_t>(),M,pd,nd,(int32_t)pe.stride(-2),(int32_t)nope.stride(-2),(int32_t)cos_sin.stride(0));
  C10_CUDA_CHECK(cudaGetLastError());
  return {packed,scale};
}
'''
        cpp_src = '#include <torch/extension.h>\n#include <tuple>\nstd::tuple<at::Tensor,at::Tensor> fused_rope_cat_fp4(at::Tensor const&,at::Tensor const&,at::Tensor const&,at::Tensor const&);\n'
        mod = load_inline(name="trtllm_fused_rope_cat_fp4_jit",
                          cpp_sources=cpp_src, cuda_sources=cuda_src,
                          functions=["fused_rope_cat_fp4"],
                          extra_cuda_cflags=["-O3"], verbose=False)
        torch.library.define(
            "trtllm::fused_rope_cat_fp4",
            "(Tensor pe, Tensor nope, Tensor cos_sin, Tensor pos) -> (Tensor, Tensor)")
        torch.library.impl(
            "trtllm::fused_rope_cat_fp4", "CUDA",
            lambda pe, nope, cos_sin, pos: mod.fused_rope_cat_fp4(pe, nope, cos_sin, pos))

        @torch.library.register_fake("trtllm::fused_rope_cat_fp4")
        def _(pe, nope, cos_sin, pos):
            pe_dim = pe.shape[-1]
            head_dim = pe_dim + nope.shape[-1]
            M = pe.numel() // pe_dim
            return (pe.new_empty((M, head_dim // 2), dtype=torch.int8),
                    pe.new_empty((M, 1), dtype=torch.int32))

        _FUSED_ROPE_CAT_FP4_READY = True


# JIT fallback for trtllm::indexer_affine_reuse. The AOT C++ op
# (cpp/tensorrt_llm/thop/indexerAffineReuseOp.cpp) is the production path; when
# the loaded .so predates that op we register an equivalent load_inline kernel
# under the same qualified name so the reuse path runs without a rebuild. Both
# compute out = (g < 0) ? -1 : g + delta elementwise.
_AFFINE_REUSE_READY = False
_AFFINE_REUSE_LOCK = threading.Lock()


def _ensure_indexer_affine_reuse_op():
    global _AFFINE_REUSE_READY
    if _AFFINE_REUSE_READY:
        return
    with _AFFINE_REUSE_LOCK:
        if _AFFINE_REUSE_READY:
            return
        if hasattr(torch.ops.trtllm, "indexer_affine_reuse"):
            _AFFINE_REUSE_READY = True
            return
        from torch.utils.cpp_extension import load_inline
        cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void _idx_affine_reuse_k(const int* __restrict__ g, int* __restrict__ o,
                                    long n, int delta) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    int v = g[i];
    o[i] = (v < 0) ? -1 : v + delta;
}
torch::Tensor indexer_affine_reuse(torch::Tensor g, int64_t delta) {
    auto o = torch::empty_like(g);
    long n = g.numel();
    if (n > 0) {
        int t = 256; long b = (n + t - 1) / t;
        _idx_affine_reuse_k<<<b, t, 0, at::cuda::getCurrentCUDAStream()>>>(
            g.data_ptr<int>(), o.data_ptr<int>(), n, (int)delta);
    }
    return o;
}
"""
        cpp_src = "torch::Tensor indexer_affine_reuse(torch::Tensor g, int64_t delta);"
        mod = load_inline(name="trtllm_indexer_affine_reuse_jit",
                          cpp_sources=cpp_src, cuda_sources=cuda_src,
                          functions=["indexer_affine_reuse"], verbose=False)
        torch.library.define(
            "trtllm::indexer_affine_reuse",
            "(Tensor global_indices_f, int delta) -> Tensor")
        torch.library.impl(
            "trtllm::indexer_affine_reuse", "CUDA",
            lambda g, delta: mod.indexer_affine_reuse(g, delta))

        @torch.library.register_fake("trtllm::indexer_affine_reuse")
        def _(g, delta):
            return torch.empty_like(g)

        _AFFINE_REUSE_READY = True


def transform_local_topk_reuse_or_compute(
    topk_indices: torch.Tensor,
    attn_metadata: "DSAtrtllmAttentionMetadata",
    layer_idx: int,
    skip_topk: bool,
    is_generation: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Local->global TopK remap with FSSS reuse-layer short-circuit.

    On an owning ("F") indexer layer (``skip_topk`` False) this runs the full
    convert_req_index_to_global gather and caches the resulting global indices
    plus this layer's index. On a reuse ("S") layer (``skip_topk`` True) the
    local TopK is identical to the owning layer's, so the global indices are
    the cached F-layer globals plus the constant per-layer offset
    ``(layer_idx - cached_layer_idx) * tokens_per_block`` (with -1 preserved);
    a single fused elementwise op replaces the gather. ``skip_topk`` is a static
    per-layer property so this branch is constant under CUDA-graph capture.
    """
    attn_metadata._ensure_pool_view_cached()
    block_size = attn_metadata._cached_tokens_per_block
    cached_global = getattr(attn_metadata, "_blaise_global_idx_cache", None)
    cached_layer = getattr(attn_metadata, "_blaise_global_idx_layer", None)
    cached_gen = getattr(attn_metadata, "_blaise_global_idx_is_gen", None)

    if (skip_topk and cached_global is not None
            and cached_gen == is_generation
            and cached_global.shape == topk_indices.shape):
        _ensure_indexer_affine_reuse_op()
        delta = (layer_idx - cached_layer) * block_size
        global_indices = torch.ops.trtllm.indexer_affine_reuse(
            cached_global, delta)
        return global_indices, attn_metadata._cached_pool_view

    global_indices, pool_view = transform_local_topk_and_prepare_pool_view(
        topk_indices, attn_metadata, layer_idx, is_generation)
    if not skip_topk:
        # Owning layer: publish its global indices for the reuse layers that
        # follow it in the same step (the local TopK they reuse is identical).
        attn_metadata._blaise_global_idx_cache = global_indices
        attn_metadata._blaise_global_idx_layer = layer_idx
        attn_metadata._blaise_global_idx_is_gen = is_generation
    return global_indices, pool_view


def split_prefill_chunks(
    seq_lens: torch.Tensor,
    max_chunk_size: int,
    start_idx: int = 0,
) -> List[List[Tuple[int, int, int, int]]]:
    """
    Split prefill requests into chunks based on max_chunk_size.
    Supports two-level chunking:
    1. Request-boundary chunking: group multiple small requests into one chunk
    2. Intra-request chunking: split large requests into multiple Q-block chunks

    Args:
        seq_lens: Sequence lengths for all requests
        max_chunk_size: Maximum number of tokens per chunk
        start_idx: Starting index for prefill requests

    Returns:
        List of chunk groups, where each group is a list of chunk specs.
        Each chunk spec is (req_idx, token_start_in_req, token_end_in_req, req_cum_start)

        - For multi-request chunks: group contains multiple specs (one per request)
        - For intra-request chunks: each Q-block is a separate group with single spec
    """
    chunk_groups = []
    num_reqs = len(seq_lens)

    current_req = start_idx
    # Compute cumulative token positions
    query_start_loc_cpu = torch.cat([
        torch.zeros(1, dtype=torch.int32, device='cpu'),
        seq_lens.cumsum(dim=0).to(torch.int32)
    ])

    while current_req < num_reqs:
        seq_len = seq_lens[current_req].item()
        req_cum_start = query_start_loc_cpu[current_req].item()

        if seq_len <= max_chunk_size:
            # This request fits in one chunk - try to pack with others
            current_size = seq_len
            chunk_specs = [(current_req, 0, seq_len, req_cum_start)]
            next_req = current_req + 1

            # Try to add more requests to this chunk
            while next_req < num_reqs:
                next_seq_len = seq_lens[next_req].item()
                if next_seq_len > max_chunk_size:
                    # Next request is large, stop packing
                    break
                if current_size + next_seq_len <= max_chunk_size:
                    next_cum_start = query_start_loc_cpu[next_req].item()
                    chunk_specs.append(
                        (next_req, 0, next_seq_len, next_cum_start))
                    current_size += next_seq_len
                    next_req += 1
                else:
                    break

            # Add as one multi-request chunk group
            chunk_groups.append(chunk_specs)
            current_req = next_req
        else:
            # Large request - split into Q-blocks
            # Each Q-block is a separate chunk group (processed in separate iteration)
            num_q_blocks = (seq_len + max_chunk_size - 1) // max_chunk_size
            for q_block_idx in range(num_q_blocks):
                token_start = q_block_idx * max_chunk_size
                token_end = min(token_start + max_chunk_size, seq_len)
                q_block_spec = [(current_req, token_start, token_end,
                                 req_cum_start)]
                chunk_groups.append(q_block_spec)

            current_req += 1

    return chunk_groups


def compute_cu_seqlen_kv_bounds_with_cache(
    seq_lens: torch.Tensor,
    num_contexts: int,
    num_ctx_tokens: int,
    cached_token_lens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention window bounds for batched sequences with causal attention,
    accounting for cached KV tokens.

    Args:
        seq_lens: current token lengths [num_contexts], dtype=torch.int32
        num_contexts: Number of sequences in the batch
        num_ctx_tokens: Total number of context tokens across all sequences in current batch
        cached_token_lens: Cached KV token lengths [num_contexts], dtype=torch.int32 (optional)

    Returns:
        cu_seqlen_ks: Start index in KV for each Q token [num_ctx_tokens]
        cu_seqlen_ke: End index (exclusive) in KV for each Q token [num_ctx_tokens]
    """
    device = seq_lens.device
    # Total KV lengths per request
    kv_lens = seq_lens if cached_token_lens is None else cached_token_lens + seq_lens  # [num_contexts]

    # Cumulative KV offsets: where each request's KV sequence starts in global KV space
    cu_kv_offsets = torch.cat([
        torch.zeros(1, device=device, dtype=torch.int32),
        torch.cumsum(kv_lens, dim=0).to(torch.int32)
    ])  # [num_contexts + 1]

    # Map each Q token to its request: [0,0,...,0, 1,1,...,1, ..., B-1,B-1,...,B-1]
    batch_ids = torch.repeat_interleave(
        torch.arange(num_contexts, device=device, dtype=torch.int32),
        seq_lens)  # [num_ctx_tokens]

    # Each Q token's KV window starts at its request's KV sequence start
    cu_seqlen_ks = cu_kv_offsets[batch_ids]  # [num_ctx_tokens]

    # Compute local Q position within each request (0-based, relative to current batch context tokens)
    cu_q_offsets = torch.cat([
        torch.zeros(1, device=device, dtype=torch.int32),
        torch.cumsum(seq_lens, dim=0).to(torch.int32)
    ])  # [num_contexts + 1]

    global_q_positions = torch.arange(num_ctx_tokens,
                                      device=device,
                                      dtype=torch.int32)
    local_q_positions = global_q_positions - torch.repeat_interleave(
        cu_q_offsets[:-1], seq_lens)  # [num_ctx_tokens]

    if cached_token_lens is not None:
        cached_per_token = torch.repeat_interleave(cached_token_lens,
                                                   seq_lens)  # [num_ctx_tokens]
        cu_seqlen_ke = cu_seqlen_ks + cached_per_token + local_q_positions + 1  # [num_ctx_tokens]
    else:
        cu_seqlen_ke = cu_seqlen_ks + local_q_positions + 1  # [num_ctx_tokens]

    return cu_seqlen_ks, cu_seqlen_ke


@dataclass
class IndexerPrefillChunkMetadata:
    """Metadata for a single prefill chunk in the indexer"""
    cu_seqlen_ks: torch.Tensor  # Attention window start for each token
    cu_seqlen_ke: torch.Tensor  # Attention window end for each token
    token_start: int  # Q token start index in batch
    token_end: int  # Q token end index in batch
    k_token_start: int  # K token start index in batch
    k_token_end: int  # K token end index in batch


class DSAtrtllmAttentionMetadata(TrtllmAttentionMetadata):
    """Attention metadata for DSA (Dense Sparse Attention) with indexer state."""

    # Store reference to indexer for preparation stage
    indexer: Optional["Indexer"] = None
    # Chunked prefill metadata for indexer (prefill-only, no CUDA graph needed)
    indexer_prefill_chunks: Optional[List[IndexerPrefillChunkMetadata]] = None
    # Max chunk size for two-level chunking:
    # 1. Request-level: Pack multiple small requests into one chunk (up to indexer_max_chunk_size)
    # 2. Intra-request: Split large requests into Q-blocks when seq_len > max_chunk_size
    indexer_max_chunk_size: int
    # Topk for sparse MLA
    num_sparse_topk: int
    # max number of draft tokens
    max_draft_tokens: int = 0
    # Enable indexer skip for short sequences
    enable_indexer_skip: bool = False
    # Whether skip the indexer for context requests
    skip_indexer_for_ctx_reqs: bool = False
    # Whether skip the indexer for generation requests
    skip_indexer_for_gen_reqs: bool = False
    # Whether to use the expanded buffers for MTP support
    use_expanded_buffers_for_mtp: bool = False
    # Whether to reshape the DSL paged MQA logits Q tensor into a kernel-
    # supported `effective_next_n` via caller-side atom-split (FP4: {1,2,3};
    # FP8: {1,2,3,4}; see `_pick_dsl_expand`). Reuses
    # `kv_lens_expanded_cuda` / `block_table_expanded` /
    # `scheduler_metadata_buffer_expanded`; runtime mutually exclusive with
    # `use_expanded_buffers_for_mtp` (the latter requires `not _use_dsl`).
    expand_for_dsl: bool = False
    # Cached (expand_factor, atom) decision from the wave-aware picker. Set at
    # `prepare()` time and read by forward call sites — avoids re-running the
    # picker per call and guarantees prepare/forward use the SAME decision
    # (otherwise the populated buffers would mismatch the kernel reshape).
    dsl_expand_factor: int = 1
    dsl_atom: int = 1

    def __init__(self, *args, **kwargs):
        """Initialize DSA metadata with SM count and indexer chunk size."""
        self.num_sms = tensorrt_llm.deep_gemm.get_num_sms()
        # Cached step-invariant values for transform_local_topk_and_prepare_pool_view.
        # These are recomputed once per step in _ensure_pool_view_cached() and
        # reused across all layers to avoid redundant Python/CUDA overhead.
        # Initialized here as plain instance attributes (not class-level
        # annotations) to stay invisible to dataclass/torch.compile introspection.
        self._pool_cache_valid = False
        self._cached_kv_mgr_id = 0
        self._cached_pool_view = None
        self._cached_stride_factor = 0
        self._cached_tokens_per_block = 0
        self._cached_block_table_ctx = None
        self._cached_block_table_gen = None
        self._cached_req_idx_ctx = None
        self._cached_req_idx_gen = None
        # Hoisted NVFP4 sparse-MLA tile-scheduler metadata gate (candidate SM1);
        # see `_hoist_sparse_mla_meta_enabled`. Pre-init so the gate is defined
        # before `__post_init__` allocates the backing buffers.
        self._sparse_mla_meta_valid = False
        self._sparse_mla_meta_num_sm_parts = 0
        self._sparse_mla_meta_sig = None
        # Per-step memo slots for layer-invariant HISA decode index math
        # (row_to_batch / prefix_lens / block_counts / row spans). Cleared in
        # prepare() and on kv-len updates; rebuilt by the first indexer layer.
        self._hisa_step_invariants = None
        self._hisa_step_rowspan = None
        # Per-step memo for the LayerSplit indexer-K read set (candidate L2);
        # same lifecycle as the HISA slots above.
        self._layersplit_step_read_set = None
        self.hisparse_coordinator = None
        self.hisparse_request_ids = None
        super().__init__(*args, **kwargs)
        if self.sparse_attention_config.indexer_max_chunk_size is not None:
            self.indexer_max_chunk_size = self.sparse_attention_config.indexer_max_chunk_size
        else:
            self.indexer_max_chunk_size = 32768  # Default to 32K tokens for the indexer

    def __post_init__(self):
        """Allocate indexer K-cache buffers and heuristic TopK metadata."""
        super().__post_init__()
        assert isinstance(self.kv_cache_manager, DSACacheManager), \
            f"DSAtrtllmAttentionMetadata requires DSACacheManager, got {type(self.kv_cache_manager)}"
        self.hisparse_coordinator = getattr(self.kv_cache_manager,
                                            "hisparse_coordinator", None)

        self.num_sparse_topk = self.sparse_attention_config.index_topk
        self.enable_indexer_skip = self.sparse_attention_config.skip_indexer_for_short_seqs
        capture_graph = self.is_cuda_graph

        self.indexer_k_cache_block_offsets = self.get_empty(
            self.cuda_graph_buffers,
            [self.max_num_sequences, self.kv_cache_manager.max_blocks_per_seq],
            cache_name="indexer_k_cache_block_offsets",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.host_indexer_k_cache_block_offsets = torch.zeros_like(
            self.indexer_k_cache_block_offsets,
            device='cpu',
            pin_memory=prefer_pinned(),
        )

        if not self.enable_context_mla_with_cached_kv:
            self.ctx_cached_token_indptr = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_requests + 1, ),
                cache_name="ctx_cached_token_indptr",
                dtype=torch.int64,
                capture_graph=capture_graph,
            )
            self.host_ctx_cached_token_indptr = torch.zeros_like(
                self.ctx_cached_token_indptr,
                device='cpu',
                pin_memory=prefer_pinned(),
            )
            self.ctx_kv_indptr = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_requests + 1, ),
                cache_name="ctx_kv_indptr",
                dtype=torch.int64,
                capture_graph=capture_graph,
            )
            self.host_ctx_kv_indptr = torch.zeros_like(
                self.ctx_kv_indptr,
                device='cpu',
                pin_memory=prefer_pinned(),
            )

        # Only when MLA chunked prefill is enabled, we need to gather the full KV for indexer's logit computation.
        # These buffers will be allocated dynamically in Indexer.prepare() based on actual total_kv_len to save memory.
        if self.enable_context_mla_with_cached_kv:
            self.slot_mapping_fp8_fullkv = None
            self.slot_mapping_scale_fullkv = None

        # New generation buffers for dsa
        self.gen_cached_token_indptr = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_requests + 1, ),
            cache_name="gen_cached_token_indptr",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self.host_gen_cached_token_indptr = torch.zeros_like(
            self.gen_cached_token_indptr,
            device='cpu',
            pin_memory=prefer_pinned(),
        )
        self.gen_kv_indptr = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_requests + 1, ),
            cache_name="gen_kv_indptr",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self.host_gen_kv_indptr = torch.zeros_like(
            self.gen_kv_indptr,
            device='cpu',
            pin_memory=prefer_pinned(),
        )
        # Indexer metadata
        # Separate slot mappings for non-interleaved layout (flat byte indices)
        self.slot_mapping_fp8 = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens, ),
            cache_name="slot_mapping_fp8",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self.host_slot_mapping_fp8 = torch.zeros_like(
            self.slot_mapping_fp8,
            device='cpu',
            pin_memory=prefer_pinned(),
        )
        self.slot_mapping_scale = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens, ),
            cache_name="slot_mapping_scale",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self.host_slot_mapping_scale = torch.zeros_like(
            self.slot_mapping_scale,
            device='cpu',
            pin_memory=prefer_pinned(),
        )
        # Per-token request index buffer for topk_indices conversion
        self.req_idx_per_token = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens, ),
            cache_name="req_idx_per_token",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        # Block table for topk_indices conversion (shared for context and generation)
        self.block_table = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_requests, self.kv_cache_manager.max_blocks_per_seq),
            cache_name="block_table",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.scheduler_metadata_buffer = self.get_empty(
            self.cuda_graph_buffers,
            (self.num_sms + 1, 2),
            cache_name="scheduler_metadata_buffer",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        # SM3 hoist: pre-allocated graph-stable destination for the HISA
        # candidate-score schedule, mirroring `scheduler_metadata_buffer`'s
        # (num_sms + 1, 2) int32 shape. Built once per step in the prepare /
        # on_update_kv_lens path (instead of 16x/step inside each recompute-"F"
        # indexer layer) when `TRTLLM_OPTRT_HOIST_HISA_SCHED` is set;
        # `_hisa_topk_from_nvfp4_cache` consumes it after a provenance check.
        # `hisa_candidate_schedule_sig` is the `(num_rows, next_n, candidate_len)`
        # provenance of the last build (None until the first decode build, or
        # whenever the hoist is off / HISA does not apply for the shape).
        self.hisa_candidate_schedule_buffer = self.get_empty(
            self.cuda_graph_buffers,
            (self.num_sms + 1, 2),
            cache_name="hisa_candidate_schedule_buffer",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.hisa_candidate_schedule_sig = None
        # When MTP runs without the expanded-tokens path, the same forward step
        # alternates between full-window calls (next_n == 1 + max_draft_tokens)
        # and per-token draft calls (next_n == 1). The 2D DeepGEMM metadata
        # API encodes next_n into the schedule, so the precomputed schedule
        # for one shape cannot be reused for the other. Maintain a second
        # buffer holding the schedule for the full next_n window; the draft
        # path keeps using `scheduler_metadata_buffer`. Always allocate (a
        # few KB) so transitions in `update_spec_dec_param` don't have to
        # special-case its existence.
        self.scheduler_metadata_buffer_full_next_n = self.get_empty(
            self.cuda_graph_buffers,
            (self.num_sms + 1, 2),
            cache_name="scheduler_metadata_buffer_full_next_n",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        # Pre-allocated graph-stable buffers for the hoisted NVFP4 sparse-MLA
        # tile-scheduler metadata (candidate SM1). Worst-case sized to
        # [kMaxNvfp4NumSmParts, DecodingSchedMeta width] int32 and [batch + 1]
        # int32 — the analogue of the bf16 FlashMLA `flash_mla_tile_scheduler_
        # metadata` (sm_count * 8) / `flash_mla_num_splits` (max_batch + 1)
        # buffers. Always allocated (a few KB) so the env flag can be toggled
        # without touching buffer layout. `_sparse_mla_meta_valid` /
        # `_sparse_mla_meta_sig` gate the once-per-step compute (reset in
        # `prepare()`); the populated slices are read by every F-layer's
        # `sparse_mla_decode_nvfp4` call. The op writes the metadata once on the
        # step's first layer and we copy it here; subsequent layers pass these
        # buffers back in so the op skips its serial metadata kernel.
        self.sparse_mla_tile_scheduler_metadata = self.get_empty(
            self.cuda_graph_buffers,
            (_SPARSE_MLA_MAX_NUM_SM_PARTS, _SPARSE_MLA_META_WIDTH),
            cache_name="sparse_mla_tile_scheduler_metadata",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.sparse_mla_num_splits = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1, ),
            cache_name="sparse_mla_num_splits",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        # Once-per-step validity gate + batch signature for the hoisted sparse-
        # MLA metadata. Plain instance attrs (not dataclass annotations) so they
        # stay invisible to torch.compile/CUDA-graph introspection, mirroring the
        # cached pool-view state above.
        self._sparse_mla_meta_valid = False
        self._sparse_mla_meta_num_sm_parts = 0
        self._sparse_mla_meta_sig = None
        # Pre-allocated 2D kv_lens buffer for the new DeepGEMM 2D context_lens
        # API. Shape: (max_num_sequences, 1 + max_draft_tokens). Each row
        # broadcasts the same kv_len across next_n positions; kernel reads a
        # slice per forward. Avoids per-forward .expand().contiguous()
        # allocations that would break CUDA graphs.
        self._create_kv_lens_2d_buffer(capture_graph=capture_graph)
        self.cu_seqlen_ks = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens, ),
            cache_name="cu_seqlen_ks",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.cu_seqlen_ke = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens, ),
            cache_name="cu_seqlen_ke",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        # Topk indices buffer to support skip indexer for requests with short sequence lengths
        if self.enable_indexer_skip:
            self.topk_indices_buffer = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_tokens, self.num_sparse_topk),
                cache_name="topk_indices_buffer",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            self.host_topk_indices_buffer = torch.zeros_like(
                self.topk_indices_buffer,
                device='cpu',
                pin_memory=prefer_pinned(),
            )
        # Per-layer persistent buffers for heuristic TopK pre_idx.
        # Indexed by [local_layer_idx, generation_position, :].
        # The graph captures reads/writes on these stable-address buffers;
        # each replay's write becomes the next replay's read (feedback loop).
        self.enable_heuristic_topk = (
            self.sparse_attention_config.enable_heuristic_topk
            and get_sm_version() >= 100)
        if self.enable_heuristic_topk:
            num_local_layers = self.kv_cache_manager.num_local_layers
            self.heuristic_prev_topk = self.get_empty(
                self.cuda_graph_buffers,
                (num_local_layers, self.max_num_sequences,
                 self.num_sparse_topk),
                cache_name="heuristic_prev_topk",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            # Zero-initialize so the first decode step's pre_idx (kernel
            # adds +1 offset) points to index 1 — a valid but benign candidate.
            # Without this, uninitialized memory produces random hint indices.
            self.heuristic_prev_topk.zero_()
            # Scratch buffer for heuristic TopK kernel output values.
            # Pre-allocated with stable address for CUDA Graph compatibility
            # (replaces cudaMallocAsync/cudaFreeAsync inside the kernel launcher).
            # Shape: [max_gen_tokens, topK] where max_gen_tokens = max_batch * (1 + max_draft).
            # The C++ Top-K requires scratch dtype == logits dtype, so this
            # follows the resolved decode-logits dtype.
            max_gen_tokens = self.max_num_sequences * (1 +
                                                       self.max_draft_tokens)
            self.heuristic_scratch_values = self.get_empty(
                self.cuda_graph_buffers,
                (max_gen_tokens, self.num_sparse_topk),
                cache_name="heuristic_scratch_values",
                dtype=resolve_indexer_logits_dtype(
                    self.sparse_attention_config),
                capture_graph=capture_graph,
            )

        # Create expanded buffers for MTP support
        self.create_expanded_buffers(capture_graph=capture_graph)

    def _create_kv_lens_2d_buffer(self, capture_graph=False):
        """Pre-allocated buffer for the DeepGEMM 2D context_lens API.

        Avoids per-forward .expand().contiguous() allocations that break CUDA
        graphs. The buffer is written in-place via .copy_() inside
        on_update_kv_lens so its address stays stable across replays.
        """
        self.kv_lens_cuda_2d = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences, 1 + self.max_draft_tokens),
            cache_name="kv_lens_cuda_2d",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )

    # TODO: remove these expanded buffers when fp8_paged_mqa_logits supports an arbitrary number of MTP draft tokens.
    def create_expanded_buffers(self, capture_graph=False):
        """Create expanded KV-length and block-table buffers for speculative decoding."""
        self.kv_lens_expanded_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences * (1 + self.max_draft_tokens), ),
            cache_name="kv_lens_expanded_cuda",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.kv_lens_expanded_host = torch.zeros_like(
            self.kv_lens_expanded_cuda,
            device='cpu',
            pin_memory=prefer_pinned(),
        )
        self.block_table_expanded = self.get_empty(
            self.cuda_graph_buffers,
            [
                self.max_num_sequences * (1 + self.max_draft_tokens),
                self.kv_cache_manager.max_blocks_per_seq
            ],
            cache_name="block_table_expanded",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.host_block_table_expanded = torch.zeros_like(
            self.block_table_expanded,
            device='cpu',
            pin_memory=prefer_pinned(),
        )
        self.scheduler_metadata_buffer_expanded = self.get_empty(
            self.cuda_graph_buffers,
            (self.num_sms + 1, 2),
            cache_name="scheduler_metadata_buffer_expanded",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )

    # This function is only used to create the expanded buffers when the max_draft_tokens is changed.
    # TODO: remove this function once fp8_paged_mqa_logits supports an arbitrary number of MTP draft tokens.
    def update_spec_dec_param(
        self,
        batch_size,
        is_spec_decoding_enabled,
        is_spec_dec_tree,
        is_spec_dec_dynamic_tree,
        max_draft_len,
        max_total_draft_tokens,
        model_is_wrapped: bool = False,
        spec_metadata: Optional['SpecMetadata'] = None,
        spec_tree_manager: Optional['SpecTreeManager'] = None,
        num_contexts: int = 0,
    ):
        """Update speculative decoding parameters and create expanded buffers."""
        super().update_spec_dec_param(batch_size,
                                      is_spec_decoding_enabled,
                                      is_spec_dec_tree,
                                      is_spec_dec_dynamic_tree,
                                      max_draft_len,
                                      max_total_draft_tokens,
                                      model_is_wrapped,
                                      spec_metadata,
                                      spec_tree_manager,
                                      num_contexts=num_contexts)
        # DSA's decode buffers (kv_lens_cuda_2d, kv_lens_expanded_cuda,
        # block_table_expanded, heuristic scratch) and the DSL paged-MQA-logits
        # next_n / atom-split are all sized as `1 + self.max_draft_tokens`, i.e.
        # the per-request verified-token width. That width is the number of
        # draft-tree NODES (max_total_draft_tokens), not the tree DEPTH
        # (max_draft_len). For linear-tree / MTP / parallel-draft these are equal,
        # so this is a no-op there. For SMC the tree is n_particles*gamma nodes
        # (e.g. 24) at depth gamma (e.g. 6); sizing by max_draft_len undersizes the
        # buffers vs the decode q buffer (which carries 1 + max_total_draft_tokens
        # positions/req) and crashes the DSL atom-split reshape.
        self.max_draft_tokens = max_total_draft_tokens
        capture_graph = self.is_cuda_graph
        if self.kv_lens_cuda_2d.shape[1] != 1 + self.max_draft_tokens:
            self._create_kv_lens_2d_buffer(capture_graph=capture_graph)
        init_shape = self.kv_lens_expanded_host.shape[0]
        if self.max_num_sequences * (1 + self.max_draft_tokens) != init_shape:
            self.create_expanded_buffers(capture_graph=capture_graph)
            # Resize heuristic scratch buffer for new max_draft_tokens.
            if self.enable_heuristic_topk:
                max_gen_tokens = self.max_num_sequences * (
                    1 + self.max_draft_tokens)
                self.heuristic_scratch_values = self.get_empty(
                    self.cuda_graph_buffers,
                    (max_gen_tokens, self.num_sparse_topk),
                    cache_name="heuristic_scratch_values",
                    dtype=resolve_indexer_logits_dtype(
                        self.sparse_attention_config),
                    capture_graph=capture_graph,
                )

    def _invalidate_pool_view_cache(self):
        """Invalidate the cached pool view and related step-invariant values.

        Must be called at the start of each forward step (in prepare()) so that
        _ensure_pool_view_cached() recomputes them for the new batch.
        """
        self._pool_cache_valid = False

    def _ensure_pool_view_cached(self):
        """Compute and cache values used by
        transform_local_topk_and_prepare_pool_view().

        These values (pool view, stride factor, block table slices, request
        index slices) are constant across all layers sharing the same KV pool
        and batch dimensions within a forward pass. Caching them avoids
        redundant Python/CUDA overhead per layer.

        Safety: _invalidate_pool_view_cache() is called unconditionally at the
        start of every step (prepare() and on_update_kv_lens()), so the boolean
        flag is always cleared before the first per-layer call within a step.
        """
        if self._pool_cache_valid and self._cached_kv_mgr_id == id(
                self.kv_cache_manager):
            return

        pool = self.kv_cache_manager.get_unique_primary_pool()
        kv_cache_manager = self.kv_cache_manager
        num_blocks, num_layers, _, _ = pool.shape
        self._cached_tokens_per_block = kv_cache_manager.tokens_per_block
        head_dim = kv_cache_manager.head_dim
        self._cached_pool_view = pool.squeeze(2).view(-1, 1, head_dim)
        self._cached_stride_factor = (num_layers *
                                      self._cached_tokens_per_block)
        self._cached_block_table_ctx = self.block_table[:self.num_contexts]
        self._cached_block_table_gen = self.block_table[self.num_contexts:self.
                                                        num_seqs]
        self._cached_req_idx_ctx = self.req_idx_per_token[:self.num_ctx_tokens]
        self._cached_req_idx_gen = (
            self.req_idx_per_token[self.num_ctx_tokens:self.num_tokens] -
            self.num_contexts)
        self._cached_kv_mgr_id = id(kv_cache_manager)
        self._pool_cache_valid = True

    @maybe_compile(dynamic=True)
    def _get_dense_topk_indices(self, seq_lens, kv_lens, num_tokens):
        device = kv_lens.device
        past_kv_lens = kv_lens - seq_lens
        # get position ids
        seq_ends = torch.cumsum(seq_lens, dim=0)
        seq_starts = seq_ends - seq_lens
        per_seq_offsets = past_kv_lens - seq_starts  # Shape: [batch_size]
        global_indices = torch.arange(num_tokens, device=device)
        batch_indices = torch.searchsorted(seq_ends,
                                           global_indices,
                                           side='right')
        repeated_offsets = per_seq_offsets[batch_indices]
        position_ids = global_indices + repeated_offsets
        # get the dense topk indices with causal mask
        range_row = torch.arange(self.num_sparse_topk, device=device)
        mask = range_row <= position_ids.unsqueeze(1)
        return torch.where(mask, range_row, -1)

    def prepare_dense_topk_indices(self,
                                   kv_lens,
                                   device=False):  # device=False means use CPU
        """Prepare dense TopK indices for short sequences that skip the indexer."""

        if self.num_contexts > 0 and self.skip_indexer_for_ctx_reqs:
            ctx_range = slice(self.num_ctx_tokens)
            if device:
                self.topk_indices_buffer[ctx_range, :].copy_(
                    self._get_dense_topk_indices(
                        self.seq_lens_cuda[:self.num_contexts],
                        kv_lens[:self.num_contexts], self.num_ctx_tokens),
                    non_blocking=True)
            else:
                self.host_topk_indices_buffer[
                    ctx_range, :] = self._get_dense_topk_indices(
                        self.seq_lens[:self.num_contexts],
                        kv_lens[:self.num_contexts], self.num_ctx_tokens)
                self.topk_indices_buffer[ctx_range, :].copy_(
                    self.host_topk_indices_buffer[ctx_range, :],
                    non_blocking=True)

        if self.num_generations > 0 and self.skip_indexer_for_gen_reqs:
            gen_range = slice(self.num_ctx_tokens, self.num_tokens)
            if device:
                self.topk_indices_buffer[gen_range, :].copy_(
                    self._get_dense_topk_indices(
                        self.seq_lens_cuda[self.num_contexts:self.num_seqs],
                        kv_lens[self.num_contexts:self.num_seqs],
                        self.num_tokens - self.num_ctx_tokens),
                    non_blocking=True)
            else:
                self.host_topk_indices_buffer[
                    gen_range, :] = self._get_dense_topk_indices(
                        self.seq_lens[self.num_contexts:self.num_seqs],
                        kv_lens[self.num_contexts:self.num_seqs],
                        self.num_tokens - self.num_ctx_tokens)
                self.topk_indices_buffer[gen_range, :].copy_(
                    self.host_topk_indices_buffer[gen_range, :],
                    non_blocking=True)

    def _get_pool_block_indices(self) -> torch.Tensor:
        """Extract memory pool block indices from host_kv_cache_block_offsets.

        The C++ setOffsets() encodes offsets as:
            encoded = memPoolBlockIndex * numLayers * kvFactor
        For SELFKONLY (MLA/DSA), kvFactor=1, so:
            memPoolBlockIndex = encoded // num_local_layers

        Returns a (num_seqs, max_blocks_per_seq) int32 CPU tensor with valid
        pool indices clamped to [0, blocks_in_primary_pool - 1].
        """
        num_local_layers = self.kv_cache_manager.num_local_layers
        max_pool_idx = self.kv_cache_manager.blocks_in_primary_pool - 1
        # DSA uses SELFKONLY mode where only key cache is stored (kv_factor=1).
        # host_kv_cache_block_offsets shape: (num_pools, max_batch*beam, 2, max_blocks_per_seq)
        # Note: dim=2 is always 2 in the tensor layout (K and V slots), but for
        # SELFKONLY only the K slot (index 0) contains valid data.
        assert self.kv_cache_manager.kv_factor == 1, \
            f"DSA requires SELFKONLY mode (kv_factor=1), got kv_factor={self.kv_cache_manager.kv_factor}"
        # Pool 0, first num_seqs entries, field 0 (key offsets)
        encoded = self.kv_cache_manager.host_kv_cache_block_offsets[
            0, :self.num_seqs, 0, :]
        pool_indices = encoded // num_local_layers
        # Clamp for safety: handles garbage padding from torch.empty in uninitialized slots
        pool_indices = pool_indices.clamp(min=0,
                                          max=max_pool_idx).to(torch.int32)
        return pool_indices

    def prepare(self):
        """Prepare DSA metadata: compute slot mappings, block tables, and prefill chunks."""
        super().prepare()
        self._invalidate_pool_view_cache()
        if self.hisparse_coordinator is not None:
            self.hisparse_coordinator.reset_step()

        # Get kv lengths
        assert self.kv_cache_params.use_cache is True, "DSA requires use_cache to be True"
        cached_token_lens = torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq,
            dtype=torch.int,
            device='cpu',
        )
        if self.enable_helix:
            # For Helix CP, inactive ranks only attend to previously cached
            # tokens (no new token appended), while active ranks add new tokens.
            # This mirrors the kv_lens logic in TrtllmAttentionMetadata.prepare().
            active_rank = ~self.helix_is_inactive_rank_cpu[:self.num_seqs]
            kv_lens = cached_token_lens.clone()
            kv_lens[active_rank] += self.seq_lens_kv[active_rank]
        else:
            kv_lens = cached_token_lens + self.seq_lens_kv

        # Prepare to support skip indexer
        num_extra_kv_tokens = self.kv_cache_params.num_extra_kv_tokens
        if self.num_contexts > 0 and self.enable_indexer_skip:
            # Minus the number of extra KV tokens because when using one-model MTP, the
            # draft layers needs more KV tokens for the next draft forwards.
            self.skip_indexer_for_ctx_reqs = kv_lens[:self.num_contexts].max(
            ).item() <= self.num_sparse_topk - num_extra_kv_tokens
        else:
            self.skip_indexer_for_ctx_reqs = False

        if self.num_generations > 0 and self.enable_indexer_skip:
            # Minus the number of extra KV tokens because when using one-model MTP, the
            # draft layers needs more KV tokens for the next draft forwards.
            self.skip_indexer_for_gen_reqs = kv_lens[
                self.num_contexts:self.num_seqs].max().item(
                ) <= self.num_sparse_topk - num_extra_kv_tokens
        else:
            self.skip_indexer_for_gen_reqs = False
        self.prepare_dense_topk_indices(kv_lens)

        # Build indexer_k_cache_block_offsets using pool block indices derived
        # from host_kv_cache_block_offsets (populated by super().prepare()).
        # This correctly resolves block IDs to memory pool indices, which is
        # required when host cache offload is enabled (block IDs != pool indices
        # for onboarded secondary blocks).
        if self.kv_cache_manager is not None:
            pool_indices = self._get_pool_block_indices()
            self.host_indexer_k_cache_block_offsets[:self.num_seqs].copy_(
                pool_indices)
            self.indexer_k_cache_block_offsets[:self.num_seqs].copy_(
                self.host_indexer_k_cache_block_offsets[:self.num_seqs],
                non_blocking=True)
            # Safety clamp: prevent OOB from CUDA graph padding entries which
            # may contain stale negative or out-of-range values after block
            # eviction/onboarding with host cache offload.
            self.indexer_k_cache_block_offsets.clamp_(min=0)

        # Build req_idx_per_token for topk_indices conversion
        host_req_idx_per_token = torch.repeat_interleave(torch.arange(
            self.num_seqs, dtype=torch.int32),
                                                         self.seq_lens,
                                                         dim=0)
        self.req_idx_per_token[:self.num_tokens].copy_(host_req_idx_per_token,
                                                       non_blocking=True)

        # Build block_table for topk_indices conversion (actual block allocation)
        if self.kv_cache_manager is not None:
            tokens_per_block = self.kv_cache_manager.tokens_per_block
            num_blocks_per_seq = (kv_lens[:self.num_seqs] + tokens_per_block -
                                  1) // tokens_per_block
            max_blocks_used = num_blocks_per_seq.max().item(
            ) if self.num_seqs > 0 else 1
            # pool_indices already has correct values; set padding to -1
            host_block_table = pool_indices[:, :max_blocks_used].clone()
            for i in range(self.num_seqs):
                if num_blocks_per_seq[i] < max_blocks_used:
                    host_block_table[i, num_blocks_per_seq[i]:] = -1
            # Copy to GPU
            self.block_table[:self.num_seqs, :max_blocks_used].copy_(
                host_block_table, non_blocking=True)
            # Host-side view for the KVarN pre-replay delta restore: the
            # delta walk derives newly-full block ids from request metadata
            # without reading the device block table back.
            self.kvarn_host_block_table = host_block_table

        # For mla_rope_append_paged_kv_assign_q
        if self.num_contexts > 0:
            self.num_ctx_cached_tokens = cached_token_lens[:self.
                                                           num_contexts].sum(
                                                           ).item()
            self.max_ctx_kv_len = kv_lens[:self.num_contexts].max().item()
            self.max_ctx_seq_len = self.seq_lens[:self.num_contexts].max().item(
            )
            # context cached token indptr
            torch.cumsum(
                cached_token_lens[:self.num_contexts],
                dim=0,
                dtype=torch.int64,
                out=self.host_ctx_cached_token_indptr[1:self.num_contexts + 1])
            self.ctx_cached_token_indptr[:self.num_contexts + 1].copy_(
                self.host_ctx_cached_token_indptr[:self.num_contexts + 1],
                non_blocking=True)
            # context kv indptr
            torch.cumsum(kv_lens[:self.num_contexts],
                         dim=0,
                         dtype=torch.int64,
                         out=self.host_ctx_kv_indptr[1:self.num_contexts + 1])
            self.ctx_kv_indptr[:self.num_contexts + 1].copy_(
                self.host_ctx_kv_indptr[:self.num_contexts + 1],
                non_blocking=True)
        else:
            self.num_ctx_cached_tokens = 0
            self.max_ctx_kv_len = 0
            self.max_ctx_seq_len = 0

        if self.num_generations > 0:
            # seq_lens is a host (CPU) tensor, so these are host reads, not
            # d2h syncs. Compute the decode-length stats once per step here
            # rather than per recompute-F indexer layer in
            # sparse_attn_indexer. All generation requests must share a
            # single decode length (the paged MQA logits + topk kernels
            # assume no padding); assert that step-level invariant once.
            gen_seq_lens = self.seq_lens[self.num_contexts:self.num_seqs]
            self.max_gen_seq_len = gen_seq_lens.max().item()
            assert self.max_gen_seq_len == gen_seq_lens.min().item(), \
                "generation seq_lens are non-uniform; decode requires padding"
            # Longest total kv length over the decode batch (host int; kv_lens
            # is a CPU tensor so this is sync-free). Used to pick the decode
            # top-k kernel; see _DSL_TOPK_MIN_KV_LEN.
            self.max_gen_kv_len = int(
                kv_lens[self.num_contexts:self.num_seqs].max().item())
            # generation cached token indptr
            torch.cumsum(
                cached_token_lens[self.num_contexts:self.num_seqs],
                dim=0,
                dtype=torch.int64,
                out=self.host_gen_cached_token_indptr[1:self.num_generations +
                                                      1])
            self.gen_cached_token_indptr[:self.num_generations + 1].copy_(
                self.host_gen_cached_token_indptr[:self.num_generations + 1],
                non_blocking=True)
            # generation kv indptr
            torch.cumsum(kv_lens[self.num_contexts:self.num_seqs],
                         dim=0,
                         dtype=torch.int64,
                         out=self.host_gen_kv_indptr[1:self.num_generations +
                                                     1])
            self.gen_kv_indptr[:self.num_generations + 1].copy_(
                self.host_gen_kv_indptr[:self.num_generations + 1],
                non_blocking=True)
        else:
            self.max_gen_seq_len = 0
            self.max_gen_kv_len = 0

        # Because the fp8_paged_mqa_logits only supports seq_len == 1/2/4 (i.e., max_draft_tokens == 0/1/3) on sm100, and
        # seq_len == 1/2 (i.e., max_draft_tokens == 0/1) on sm90, for other cases, we need to flatten the q tensor and
        # expand the kv_lens and block_table for MTP support.
        # The CuTe DSL kernel supports arbitrary next_n natively, so it never needs expansion.
        # TODO:
        # - No distinction between sm90 and sm100 is needed once MTP3 is supported on sm90.
        # - Remove this once fp8_paged_mqa_logits supports an arbitrary number of MTP draft tokens.
        _use_dsl = self.sparse_attention_config.use_cute_dsl_paged_mqa_logits
        self.use_expanded_buffers_for_mtp = (not _use_dsl and (
            (self.max_draft_tokens > 1 and get_sm_version() == 90) or
            ((self.max_draft_tokens == 2 or self.max_draft_tokens > 3)
             and get_sm_version() >= 100)))
        if self.use_expanded_buffers_for_mtp:
            # Expand kv_lens_cuda (only generation)
            num_tokens = self.num_generations * (1 + self.max_draft_tokens)
            gen_kv_lens = kv_lens[self.num_contexts:self.num_seqs]
            gen_kv_lens_expanded = torch.stack([gen_kv_lens] *
                                               (1 + self.max_draft_tokens),
                                               dim=0)
            gen_kv_lens_expanded = gen_kv_lens_expanded.transpose(
                0, 1).contiguous().flatten()
            self.kv_lens_expanded_host[:num_tokens].copy_(gen_kv_lens_expanded)
            self.kv_lens_expanded_cuda[:num_tokens].copy_(
                self.kv_lens_expanded_host[:num_tokens], non_blocking=True)

            # Expand indexer_k_cache_block_offsets (only generation)
            # host_indexer_k_cache_block_offsets already contains correct pool
            # indices from _get_pool_block_indices() above.
            if self.kv_cache_manager is not None and self.num_generations > 0:
                max_len = self.host_indexer_k_cache_block_offsets.shape[1]
                gen_block_tensor = self.host_indexer_k_cache_block_offsets[
                    self.num_contexts:self.num_seqs, :max_len]
                expanded_blocks = gen_block_tensor.repeat_interleave(
                    1 + self.max_draft_tokens, dim=0)
                self.host_block_table_expanded[:num_tokens, :max_len].copy_(
                    expanded_blocks, non_blocking=True)
                self.block_table_expanded[:num_tokens].copy_(
                    self.host_block_table_expanded[:num_tokens],
                    non_blocking=True)
                self.block_table_expanded.clamp_(min=0)

        # CuTe DSL FP4 paged MQA logits kernel natively supports
        # next_n ∈ {1, 2, 3} only. For next_n ≥ 4 atom-split is mandatory.
        # For next_n ∈ {2, 3} atom-split is also beneficial when the wave-aware
        # picker decides more SM utilization outweighs the (expand_factor)x
        # HBM cost (e.g., low batch with idle SMs). The expanded buffers
        # (`kv_lens_expanded_cuda` / `block_table_expanded` /
        # `scheduler_metadata_buffer_expanded`, all sized for the worst-case
        # `1+max_draft_tokens` factor) are reused: under `_use_dsl=True` the
        # existing FP8 DG expand path never writes to them (gated on
        # `not _use_dsl`), so there's no conflict.
        # Trigger relaxed to `max_draft_tokens >= 1` (i.e., next_n >= 2) so the
        # picker can choose to expand when waves vs HBM trade-off favors it.
        # Trigger atom-split for both FP4 and FP8 DSL paths. FP4 kernel
        # supports atom ∈ {1, 2, 3}; FP8 supports {1, 2, 3, 4}. Picker is
        # given the appropriate kernel_atoms set so it only enumerates
        # decompositions the kernel can handle.
        self.expand_for_dsl = (_use_dsl and self.kv_cache_manager is not None
                               and self.max_draft_tokens >= 1)
        if self.expand_for_dsl and self.num_generations > 0:
            next_n = 1 + self.max_draft_tokens
            kernel_atoms = (1, 2,
                            3) if self.kv_cache_manager.use_fp4 else (1, 2, 3,
                                                                      4)
            # Wave-aware picker. max_ctx ≈ longest gen kv_len (decode iter
            # upper-bound observed at this prepare). num_sms is hardware.
            gen_kv_lens = kv_lens[self.num_contexts:self.num_seqs]
            max_ctx = int(
                gen_kv_lens.max().item()) if gen_kv_lens.numel() else 0
            expand_factor, atom = _pick_dsl_expand(
                next_n,
                batch_size=self.num_generations,
                max_ctx=max_ctx,
                num_sms=self.num_sms,
                kernel_atoms=kernel_atoms,
            )
            self.dsl_expand_factor = expand_factor
            self.dsl_atom = atom
            # Only populate when picker chose to actually split (factor > 1);
            # factor=1 means kernel-native, no expansion needed.
            if expand_factor > 1:
                num_tokens = self.num_generations * expand_factor
                gen_kv_lens_expanded = gen_kv_lens.repeat_interleave(
                    expand_factor)
                self.kv_lens_expanded_host[:num_tokens].copy_(
                    gen_kv_lens_expanded)
                self.kv_lens_expanded_cuda[:num_tokens].copy_(
                    self.kv_lens_expanded_host[:num_tokens], non_blocking=True)
                if self.kv_cache_manager is not None:
                    max_len = self.host_indexer_k_cache_block_offsets.shape[1]
                    gen_block_tensor = self.host_indexer_k_cache_block_offsets[
                        self.num_contexts:self.num_seqs, :max_len]
                    expanded_blocks = gen_block_tensor.repeat_interleave(
                        expand_factor, dim=0)
                    self.host_block_table_expanded[:num_tokens, :max_len].copy_(
                        expanded_blocks, non_blocking=True)
                    self.block_table_expanded[:num_tokens].copy_(
                        self.host_block_table_expanded[:num_tokens],
                        non_blocking=True)
                    self.block_table_expanded.clamp_(min=0)
        else:
            # Reset cache; forward path uses kernel-native next_n.
            self.dsl_expand_factor = 1
            self.dsl_atom = 1 + self.max_draft_tokens

        # Prepare metadata for indexer
        Indexer.prepare(metadata=self)

    def _maybe_build_hisa_candidate_schedule(self, gen_kv_lens: torch.Tensor):
        """SM3 hoist: build the HISA candidate-score schedule once per decode
        step into `hisa_candidate_schedule_buffer` (instead of 16x/step inside
        each recompute-"F" indexer layer). No-op unless
        `TRTLLM_OPTRT_HOIST_HISA_SCHED` is set. `gen_kv_lens` is the gen slice of
        the (runtime-corrected) device kv_lens — the same tensor the per-layer
        GEMM derives `prefix_lens` from, so the prebuilt schedule is bit-identical
        to the in-line rebuild for the matching `(num_rows, next_n, candidate_len)`.
        `next_n = 1 + max_draft_tokens` is the main-forward window in which the
        HISA pre-indexer runs (the consumer revalidates the signature and falls
        back to its own rebuild on any mismatch, e.g. a draft-only next_n==1
        call, so a wrong guess can never produce a wrong result).
        `self.max_gen_kv_len` is the host int already computed sync-free in
        prepare()."""
        # Always reset provenance first: a stale signature must never let a
        # later step consume a schedule built for a different shape.
        self.hisa_candidate_schedule_sig = None
        if not _hoist_hisa_sched():
            return
        if self.num_generations <= 0 or self.sparse_attention_config is None:
            return
        next_n = 1 + self.max_draft_tokens
        built = _build_hisa_candidate_schedule(self.sparse_attention_config,
                                               gen_kv_lens, next_n,
                                               self.num_generations,
                                               self.max_gen_kv_len, self.num_sms)
        if built is None:
            return
        schedule, sig = built
        self.hisa_candidate_schedule_buffer.copy_(schedule, non_blocking=True)
        self.hisa_candidate_schedule_sig = sig

    def on_update_kv_lens(self):
        """Refresh indexer slot mappings after KV lengths change at runtime."""
        self._hisa_step_invariants = None
        self._hisa_step_rowspan = None
        self._layersplit_step_read_set = None
        if self.hisparse_coordinator is not None:
            self.hisparse_coordinator.reset_step()
        # After changing the kv_lens/kv_lens_cuda, we may need to update other metadatas.
        # Especially for the changes in the _preprocess_inputs() of model_engine.py.
        #
        # NOTE:
        # In overlap scheduler + speculative decoding, kv_lens_cuda can be corrected at runtime
        # (inside _preprocess_inputs) to account for variable accepted tokens. The indexer
        # slot_mapping_* buffers also depend on these effective cached lengths. If we do not
        # refresh slot mappings here, indexer K-cache updates can be written with stale offsets.

        # _preprocess_inputs() also uses this as a general hook to "invalidate per-forward-pass
        # caches so they are recomputed (and captured) on every _forward_step". Invalidate the
        # pool_view cache here so it is recomputed on the next
        # transform_local_topk_and_prepare_pool_view() call.
        self._invalidate_pool_view_cache()

        if self.kv_cache_manager is not None and self.num_tokens > 0:
            seq_lens = self.seq_lens_cuda[:self.num_seqs]
            # Runtime cached lengths after overlap/spec-dec correction.
            start_positions = self.kv_lens_cuda[:self.num_seqs] - seq_lens

            # Reuse request-per-token mapping prepared in metadata.prepare().
            # This avoids repeat_interleave in graph-capture mode.
            req_indices = self.req_idx_per_token[:self.num_tokens].to(
                dtype=torch.int64)
            seq_starts = torch.cumsum(
                seq_lens, dim=0, dtype=torch.int64) - seq_lens.to(torch.int64)
            token_offsets = torch.arange(
                self.num_tokens, device=seq_lens.device,
                dtype=torch.int64) - seq_starts[req_indices]

            global_positions = start_positions[req_indices] + token_offsets
            # Under FP4 the indexer cache stores two E2M1 codes per byte, so
            # the per-token data footprint is head_dim // 2; otherwise it is
            # head_dim (one FP8 byte per element). Feed the real byte count
            # into _compute_slot_mappings so scatter/gather see offsets that
            # match the pool layout produced by createIndexerKCachePools.
            use_fp4 = self.kv_cache_manager.use_fp4
            index_head_dim = self.kv_cache_manager.index_head_dim
            data_bytes_per_token = index_head_dim // 2 if use_fp4 else index_head_dim
            fp8_indices, scale_indices = _compute_slot_mappings(
                global_positions,
                self.indexer_k_cache_block_offsets,
                req_indices,
                index_head_dim,
                self.kv_cache_manager.tokens_per_block,
                self.kv_cache_manager.quant_block_size,
                data_bytes_per_token=data_bytes_per_token,
            )
            self.slot_mapping_fp8[:self.num_tokens] = fp8_indices
            self.slot_mapping_scale[:self.num_tokens] = scale_indices

        if self.num_generations > 0:
            torch.cumsum(
                self.kv_lens_cuda[self.num_contexts:self.
                                  num_seqs],  # num_contexts should be 0
                dim=0,
                dtype=torch.int64,
                out=self.gen_kv_indptr[1:self.num_generations + 1])
            torch.cumsum(
                (self.kv_lens_cuda[self.num_contexts:self.num_seqs] -
                 self.seq_lens_cuda[self.num_contexts:self.num_seqs]),
                dim=0,
                dtype=torch.int64,
                out=self.gen_cached_token_indptr[1:self.num_generations + 1])
            # Write 2D kv_lens in-place (broadcast same kv_len across next_n
            # positions). .expand() returns a view and .copy_() writes into the
            # pre-allocated destination, so this is CUDA-graph-friendly.
            gen_kv_lens = self.kv_lens_cuda[self.num_contexts:self.num_seqs]
            next_n_cap = self.kv_lens_cuda_2d.shape[1]
            self.kv_lens_cuda_2d[:self.num_generations, :next_n_cap].copy_(
                gen_kv_lens.unsqueeze(-1).expand(-1, next_n_cap))
            # Build the next_n=1 schedule (used by MTP draft layers and any
            # non-MTP forward). Reshape the contiguous gen slice of
            # kv_lens_cuda to (num_gen, 1) — slicing kv_lens_cuda_2d's first
            # column would be non-contiguous and would fail the metadata
            # kernel's is_contiguous assertion.
            context_lens_next_n1 = gen_kv_lens.view(-1, 1)
            # `_DG_SCHEDULE_BLOCK_KV` (= 64) instead of cache `tokens_per_block`:
            # see module-level constant comment for the SPLIT_KV=256 alignment.
            scheduler_metadata_buffer = get_paged_mqa_logits_metadata(
                context_lens_next_n1, _DG_SCHEDULE_BLOCK_KV, self.num_sms)
            self.scheduler_metadata_buffer.copy_(scheduler_metadata_buffer,
                                                 non_blocking=True)
            # When MTP is on without the expanded-tokens path, also populate
            # the full-next_n schedule for the main forward call. The metadata
            # kernel reads next_n from context_lens.size(1), so we must pass
            # the wider slice here.
            if (self.max_draft_tokens > 0
                    and not self.use_expanded_buffers_for_mtp):
                context_lens_full_next_n = self.kv_lens_cuda_2d[:self.
                                                                num_generations, :
                                                                next_n_cap]
                scheduler_metadata_buffer_full_next_n = get_paged_mqa_logits_metadata(
                    context_lens_full_next_n, _DG_SCHEDULE_BLOCK_KV,
                    self.num_sms)
                self.scheduler_metadata_buffer_full_next_n.copy_(
                    scheduler_metadata_buffer_full_next_n, non_blocking=True)
            if self.use_expanded_buffers_for_mtp:
                num_draft_tokens = 1 + self.max_draft_tokens
                num_tokens = self.num_generations * num_draft_tokens
                kv_lens_expanded = torch.stack([gen_kv_lens] * num_draft_tokens,
                                               dim=0)
                self.kv_lens_expanded_cuda[:num_tokens] = \
                    kv_lens_expanded.transpose(0, 1).contiguous().flatten()
                # New API requires 2D; each expanded token becomes a (1,) row.
                kv_lens_expanded_2d = self.kv_lens_expanded_cuda[:
                                                                 num_tokens].view(
                                                                     -1, 1)
                scheduler_metadata_buffer_expanded = get_paged_mqa_logits_metadata(
                    kv_lens_expanded_2d, _DG_SCHEDULE_BLOCK_KV, self.num_sms)
                self.scheduler_metadata_buffer_expanded.copy_(
                    scheduler_metadata_buffer_expanded, non_blocking=True)
            # DSL atom-split path: mirror the prepare()-time build so that
            # overlap-scheduler / spec-dec runtime corrections to kv_lens_cuda
            # propagate into kv_lens_expanded_cuda and the matching schedule.
            # Reuse the cached (dsl_expand_factor, dsl_atom) — re-running the
            # picker here would let the split decision drift between prepare
            # and forward, breaking CUDA graph capture.
            if self.expand_for_dsl and self.dsl_expand_factor > 1:
                expand_factor = self.dsl_expand_factor
                num_tokens = self.num_generations * expand_factor
                gen_kv_lens_expanded = gen_kv_lens.repeat_interleave(
                    expand_factor)
                self.kv_lens_expanded_cuda[:num_tokens].copy_(
                    gen_kv_lens_expanded)
                kv_lens_expanded_2d = self.kv_lens_expanded_cuda[:
                                                                 num_tokens].view(
                                                                     -1, 1)
                scheduler_metadata_buffer_expanded = get_paged_mqa_logits_metadata(
                    kv_lens_expanded_2d, _DG_SCHEDULE_BLOCK_KV, self.num_sms)
                self.scheduler_metadata_buffer_expanded.copy_(
                    scheduler_metadata_buffer_expanded, non_blocking=True)
            # SM3 hoist: build the HISA candidate-score schedule once here, from
            # the same runtime-corrected gen kv_lens the F-layer GEMM will use.
            self._maybe_build_hisa_candidate_schedule(gen_kv_lens)
            # Invalidate the hoisted NVFP4 sparse-MLA tile-scheduler metadata so
            # the step's first F-layer recomputes it (candidate SM1). The
            # metadata is data-independent (a pure function of batch_size, s_q
            # and topk), so this once-per-step reset plus the caller's batch
            # signature guard is sufficient; the eager prepare() runs before any
            # CUDA-graph replay, keeping the buffer addresses stable.
            self._sparse_mla_meta_valid = False
        self._hisa_step_invariants = None
        self._hisa_step_rowspan = None
        self._layersplit_step_read_set = None
        self.prepare_dense_topk_indices(self.kv_lens_cuda, device=True)

    def update_for_spec_dec(self):
        """Reset context/generation counters and refresh slot mappings for speculative decoding."""
        super().update_for_spec_dec()
        self._hisa_step_invariants = None
        self._hisa_step_rowspan = None
        self._layersplit_step_read_set = None
        # host
        self.max_ctx_kv_len = 0
        self.num_ctx_cached_tokens = 0
        self.max_gen_seq_len = 1
        self.max_gen_kv_len = 0

        # device
        self.on_update_kv_lens()


@maybe_compile(dynamic=True)
def _scale(weights: torch.Tensor, q_scale: torch.Tensor,
           s: float) -> torch.Tensor:
    """Scale attention weights by quantization scale and constant factor."""
    return weights * q_scale.squeeze(-1) * s


@maybe_compile(dynamic=True)
def _to_float(hidden_states: torch.Tensor) -> torch.Tensor:
    """Cast hidden states to float32 for TF32 GEMM computation."""
    return hidden_states.float()


@contextmanager
def _tf32_matmul_enabled():
    """Temporarily enable TF32 tensor cores for FP32 matmul in this scope.

    Forces PyTorch/cuBLASLt to use CUBLAS_COMPUTE_32F_FAST_TF32, which
    guarantees TF32 tensor cores. Plain CUBLAS_COMPUTE_32F (used by
    torch.ops.trtllm.cublas_mm) falls back to SIMT SGEMM on CUDA cores
    based on cuBLASLt heuristics for small M.
    """
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


class _FusedWkWpNvfp4:
    """Single NVFP4 GEMM for the Indexer's wk + weights_proj projections.

    wk [head_dim <- hidden] and weights_proj [n_heads <- hidden] consume the
    SAME hidden-state tensor, so the unfused pair quantizes the activation
    twice and streams the [M, hidden] activation through two skinny GEMMs.
    This runs one [head_dim + n_heads <- hidden] GEMM instead: the packed FP4
    weights and the swizzled block scales are concatenated along N at load
    time and the output is split back into the two projections.

    Scale handling: the activation quantization (and therefore the FP4 codes
    and per-block activation scales) is shared, which is exact because both
    parts see the same input. The GEMM's scalar dequant alpha uses wk's
    per-tensor weight scale; the weights_proj part is corrected by
    weight_scale_2_wp / weight_scale_2_wk on its f32 output slice. That fold
    is a per-element f32 multiply (1 ulp), NOT an e4m3 block-scale
    requantization, so no quantization error is introduced.

    Block-scale layout: trtllm swizzled scales order 128-row blocks
    outermost (block_scale_interleave pads each part's N to a multiple of
    128), so concatenating the two flat swizzled tensors is the swizzled
    layout of the [head_dim + n_heads, hidden] weight iff head_dim is a
    multiple of 128. The trailing pad rows of the weights_proj block only
    back output columns >= head_dim + n_heads, which are never computed
    (GEMM N = head_dim + n_heads).
    """

    # Mirrors NVFP4LinearMethod._input_prepare quantization constants.
    _FP8_MAX = 448.0
    _E2M1_MAX = 6.0

    def __init__(self, weight: torch.Tensor, weight_scale: torch.Tensor,
                 wk_out: int, wp_out: int, out_dtype: torch.dtype,
                 weight_scale_2_wk: torch.Tensor, wp_out_scale: float,
                 input_scale: Optional[torch.Tensor],
                 alpha: Optional[torch.Tensor], allowed_backends: str):
        self.weight = weight
        self.weight_scale = weight_scale
        self.wk_out = wk_out
        self.wp_out = wp_out
        self.out_dtype = out_dtype
        self.weight_scale_2_wk = weight_scale_2_wk
        self.wp_out_scale = wp_out_scale
        # Static activation quantization when the checkpoint carries an
        # activation scale; dynamic (per-call amax) otherwise.
        self.input_scale = input_scale
        self.alpha = alpha
        self.allowed_backends = allowed_backends

    @classmethod
    def build(cls, wk: Linear, wp: Linear,
              allowed_backends: Optional[List[str]]
              ) -> Optional["_FusedWkWpNvfp4"]:
        """Build from two loaded NVFP4 Linears; None when fusion is unsafe."""

        def _is_nvfp4(m: Linear) -> bool:
            w = getattr(m, "weight", None)
            ws = getattr(m, "weight_scale", None)
            ws2 = getattr(m, "weight_scale_2", None)
            return (w is not None and w.dtype == torch.uint8 and w.dim() == 2
                    and ws is not None and ws.dim() == 1 and ws2 is not None
                    and m.bias is None
                    and getattr(m, "pre_quant_scale", None) is None)

        if not (_is_nvfp4(wk) and _is_nvfp4(wp)):
            return None
        if wk.weight.shape[1] != wp.weight.shape[1]:
            return None
        if wk.dtype != wp.dtype:
            return None
        # Swizzled scales order 128-row blocks outermost; the concat is only
        # the fused layout when wk's rows fill whole blocks.
        if wk.out_features % 128 != 0:
            return None
        k_blocks = (wk.weight.shape[1] * 2) // 16
        k_blocks_padded = (k_blocks + 3) // 4 * 4
        for m in (wk, wp):
            n_padded = (m.out_features + 127) // 128 * 128
            if m.weight.shape[0] != m.out_features:
                return None
            if m.weight_scale.numel() != n_padded * k_blocks_padded:
                return None

        wk_static = getattr(wk, "input_scale", None) is not None
        wp_static = getattr(wp, "input_scale", None) is not None
        if wk_static != wp_static:
            return None
        if wk_static:
            # Both parts must quantize the shared activation identically for
            # the fused GEMM to reproduce the unfused outputs.
            if not torch.allclose(wk.input_scale, wp.input_scale):
                logger.warning(
                    "DSA indexer wk/weights_proj input_scale mismatch; "
                    "keeping the projections unfused.")
                return None

        weight = torch.cat([wk.weight.data, wp.weight.data], dim=0)
        weight_scale = torch.cat(
            [wk.weight_scale.data, wp.weight_scale.data], dim=0)
        wp_out_scale = float(
            (wp.weight_scale_2.float() / wk.weight_scale_2.float()).item())
        return cls(
            weight=weight,
            weight_scale=weight_scale,
            wk_out=wk.out_features,
            wp_out=wp.out_features,
            out_dtype=wk.dtype,
            weight_scale_2_wk=wk.weight_scale_2.data,
            wp_out_scale=wp_out_scale,
            input_scale=wk.input_scale if wk_static else None,
            alpha=wk.alpha if wk_static else None,
            allowed_backends=','.join(allowed_backends or
                                      ['cutlass', 'cublaslt', 'cuda_core']))

    def __call__(
            self,
            hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the fused GEMM; returns (wk_out, weights_proj_out) slices."""
        if self.input_scale is not None:
            input_scale = self.input_scale
            alpha = self.alpha
        else:
            # Dynamic activation quantization, computed ONCE for both parts
            # (the unfused pair pays the amax reduction + quantize twice).
            global_max = self._FP8_MAX * self._E2M1_MAX
            amax = torch.amax(torch.abs(hidden_states)).float()
            input_scale = global_max / amax
            alpha = (amax / global_max) * self.weight_scale_2_wk
        act_fp4, act_sf = torch.ops.trtllm.fp4_quantize(
            hidden_states, input_scale, 16, False)
        out = torch.ops.trtllm.nvfp4_gemm(
            act_fp4,
            self.weight,
            act_sf,
            self.weight_scale,
            alpha,
            self.out_dtype,
            allowed_backends=self.allowed_backends)
        indexer_k = out[..., :self.wk_out]
        weights = out[..., self.wk_out:self.wk_out + self.wp_out]
        if self.wp_out_scale != 1.0:
            weights = weights * self.wp_out_scale
        return indexer_k, weights


class Indexer(nn.Module):
    """DSA sparse attention indexer that selects top-K KV cache entries per token."""

    def __init__(self,
                 quant_config: Optional[QuantConfig],
                 pos_embd_params: Optional[PositionalEmbeddingParams],
                 mla_params: Optional[MLAParams],
                 skip_create_weights_in_init: bool,
                 sparse_attention_config: "SparseAttentionConfig",
                 dtype: Optional[torch.dtype],
                 layer_idx: int = 0,
                 aux_stream: Optional[torch.cuda.Stream] = None):
        """Initialize indexer with projection weights, norms, and TopK configuration."""
        super().__init__()
        self.hidden_size = mla_params.hidden_size
        self.q_lora_rank = mla_params.q_lora_rank
        self.rope_dim = mla_params.qk_rope_head_dim
        self.n_heads = sparse_attention_config.index_n_heads  # 64
        self.head_dim = sparse_attention_config.index_head_dim  # 128
        self.index_topk = sparse_attention_config.index_topk  # 2048
        self.layer_idx = layer_idx
        # Per-block (device int64) epoch last reconstructed into this layer's
        # fp16 main pool; lazily sized to the pool on first amortized restore.
        # None until then / when amortize is off.
        self._kvarn_restored_gen = None
        self.indexer_mode = getattr(sparse_attention_config, "indexer_mode",
                                    "vanilla")
        self.index_topk_freq = getattr(sparse_attention_config,
                                       "index_topk_freq", None)
        self.index_topk_pattern = getattr(sparse_attention_config,
                                          "index_topk_pattern", None)
        # Cross-STEP TopK reuse: the DSA Top-K block selection is stable across
        # consecutive decode steps, so recompute the full logits MQA + Top-K
        # only every `index_topk_step_freq` decode steps and reuse the cached
        # selection in between (patching in the newly appended KV position(s) so
        # recency is exact). This runs in sparse_attn_indexer, which executes in
        # the trtllm::mla_dsa_attn_inplace custom op -- explicitly EXCLUDED from
        # CUDA graph capture -- so a Python step counter gating the reuse is
        # safe (no graph break). None / <=1 disables it (no behavior change).
        self.index_topk_step_freq = getattr(sparse_attention_config,
                                            "index_topk_step_freq", None)
        self._xstep_enabled = (self.index_topk_step_freq is not None
                               and self.index_topk_step_freq > 1)
        # Recency patch trades extra launches for guaranteed inclusion of the
        # newest positions on reuse steps. It is launch-bound (~20 tiny ops) and
        # can cost more than the kernels it replaces, so default to frozen reuse
        # (no patch) -- the win comes from skipping the logits MQA + Top-K.
        self._xstep_recency_patch = bool(
            getattr(sparse_attention_config, "index_topk_step_recency_patch",
                    False))
        # Per-layer cross-step state (this module is per-layer). Populated on a
        # refresh step and reused on the following (freq-1) steps.
        self._xstep_counter = 0
        self._xstep_cache: Optional[torch.Tensor] = None
        self._xstep_refresh_kvlen: Optional[torch.Tensor] = None
        self._xstep_batch_tokens = 0
        self._xstep_idx_cache: Dict[Tuple, int] = {}
        self.enable_nvfp4_hisa = getattr(sparse_attention_config,
                                         "enable_nvfp4_hisa", False)
        self.hisa_block_size = getattr(sparse_attention_config,
                                       "hisa_block_size", 128)
        self.hisa_block_topk = getattr(sparse_attention_config,
                                       "hisa_block_topk", 64)
        self.hisa_compression_ratio = getattr(sparse_attention_config,
                                              "hisa_compression_ratio", 4.0)
        self.hisa_min_seq_len = getattr(sparse_attention_config,
                                        "hisa_min_seq_len", 32768)
        self.hisa_execution_mode = getattr(sparse_attention_config,
                                           "hisa_execution_mode", "optimized")
        self._hisa_range_cache: Dict[Tuple[torch.device, int], torch.Tensor] = {}
        self._hisa_full_cache: Dict[Tuple[torch.device, int, int], torch.Tensor] = {}
        self._hisa_e2m1_cache: Dict[torch.device, torch.Tensor] = {}
        self._hisa_rep_cache: Dict[Tuple[int, int, int, Tuple[int, ...], int],
                                   Dict[str, torch.Tensor]] = {}
        self.skip_topk = self._should_reuse_previous_topk()

        # NVFP4 backend override for the indexer projection GEMMs. With an
        # NVFP4-quantized indexer (e.g. the REAP indexer overlay) the GEMM
        # AutoTuner picks the cutlass backend for wq_b/wk/weights_proj at
        # decode token counts; cuBLASLt is bit-identical (max|diff| == 0 on
        # real weights at M in {4,16}) and 1.16-1.30x faster. Ignored when the
        # indexer is unquantized. Comma-separated to override, empty to
        # restore auto-selection.
        _idx_backends_env = os.environ.get('TRTLLM_INDEXER_NVFP4_BACKENDS',
                                           'cublaslt')
        self._indexer_nvfp4_backends = ([
            b.strip() for b in _idx_backends_env.split(',') if b.strip()
        ] or None)

        self.wq_b = Linear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True,
            nvfp4_allowed_backends=self._indexer_nvfp4_backends)
        self.wk = Linear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            dtype=torch.float32,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True,
            nvfp4_allowed_backends=self._indexer_nvfp4_backends)
        self.k_norm = LayerNorm(hidden_size=self.head_dim, eps=1e-6)
        self.weights_proj = Linear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            dtype=torch.float32,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True,
            nvfp4_allowed_backends=self._indexer_nvfp4_backends)

        # Fused wk + weights_proj weight for single F.linear FP32 GEMM under allow_tf32.
        # Maps to TF32 tensor cores on Ampere+.
        self._fused_wk_wp_weight: Optional[torch.Tensor] = None
        # NVFP4 counterpart (quantized indexer): one [head_dim + n_heads <-
        # hidden] NVFP4 GEMM with a single activation quantize, built from
        # the loaded wk/weights_proj tensors in post_load_weights.
        # TRTLLM_INDEXER_FUSE_WK_WP=0 disables.
        self._fused_wk_wp_nvfp4: Optional[_FusedWkWpNvfp4] = None

        indexer_rope_interleave = getattr(sparse_attention_config,
                                          'indexer_rope_interleave', False)
        self.rotary_emb = RotaryEmbedding(
            pos_embd_params.rope,
            head_dim=self.rope_dim,
            is_neox=not indexer_rope_interleave,
        )

        self.softmax_scale = self.head_dim**-0.5
        # indexer_k_dtype controls both Q and K precision. DeepGEMM's
        # fp8_fp4_mqa_logits / fp8_fp4_paged_mqa_logits kernels only dispatch
        # to FP4xFP4 or FP8xFP8 (no mixed-precision variant). The DeepGEMM
        # kernel asserts SM100 + head_dim=128 at launch time under FP4.
        # Must precede _rope_cat_fuse_ok below, which reads self.use_fp4.
        self.use_fp4 = sparse_attention_config.indexer_k_dtype == "fp4"
        # Fused RoPE+cat+FP4-quant eligibility: the fused kernel folds the
        # standalone flashinfer RoPE launch (+ its BF16 q_pe/k_pe write-back
        # and reload) into fused_cat_fp4. It requires neox RoPE, head_dim 128,
        # an even rope_dim whose half is a multiple of 4, and the
        # flashinfer-compatible cos/sin cache (cos first half, sin second).
        self._rope_cat_fuse_ok = bool(
            self.use_fp4 and self.head_dim == 128 and self.rope_dim == 64
            and (self.rope_dim % 4 == 0) and ((self.rope_dim // 2) % 4 == 0)
            and not indexer_rope_interleave
            and getattr(self.rotary_emb, 'rotary_cos_sin', None) is not None)
        self._rope_cat_cos_sin = None
        # TODO: make it configurable from hf config
        self.scale_fmt = "ue8m0"
        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]
        self.use_cute_dsl_topk = (sparse_attention_config.use_cute_dsl_topk
                                  and IS_CUTLASS_DSL_AVAILABLE)
        self.use_cute_dsl_paged_mqa_logits = (
            sparse_attention_config.use_cute_dsl_paged_mqa_logits
            and IS_CUTLASS_DSL_AVAILABLE)
        # Decode-logits dtype for the DSL score -> Top-K pipeline. fp16 at
        # prod (config-matched: scoring inputs are fp8/fp4 quantized);
        # fp32 whenever the DeepGEMM scoring fallback is active.
        self.logits_dtype = resolve_indexer_logits_dtype(
            sparse_attention_config)
        self.weight_scale_factor = self.softmax_scale * self.n_heads**-0.5

        self._enable_heuristic_topk = (
            sparse_attention_config.enable_heuristic_topk
            and get_sm_version() >= 100)

        if (self.use_cute_dsl_topk
                or self.use_cute_dsl_paged_mqa_logits) and layer_idx == 0:
            from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

            if self.use_cute_dsl_topk:
                cute_dsl_custom_ops.warmup_cute_dsl_indexer_topk(
                    dtype=self.logits_dtype, top_k=self.index_topk)

        if self._enable_heuristic_topk and layer_idx == 0:
            # Populate static caches (sm_count, L2 cache size) inside the C++
            # Scheme X dispatcher before any CUDA Graph capture so the host
            # attribute queries do not end up frozen into a captured graph.
            warmup_heuristic_topk_decode(top_k=self.index_topk,
                                         logits_dtype=self.logits_dtype)

    def _should_reuse_previous_topk(self) -> bool:
        if self.indexer_mode not in ("indexcache", "indexcache-hisa"):
            return False
        if self.index_topk_pattern:
            role = self.index_topk_pattern[self.layer_idx %
                                           len(self.index_topk_pattern)]
            return role == "S"
        freq = 1 if self.index_topk_freq is None else self.index_topk_freq
        return max(self.layer_idx - 1, 0) % freq != 0

    def _get_indexcache_topk(
            self, metadata: DSAtrtllmAttentionMetadata,
            num_tokens: int) -> Optional[torch.Tensor]:
        if not self.skip_topk:
            return None
        cached = getattr(metadata, "_blaise_indexcache_topk", None)
        if cached is None or cached.shape[0] < num_tokens:
            return None
        if cached.shape[1] != self.index_topk:
            return None
        return cached[:num_tokens]

    def _maybe_store_indexcache_topk(
            self, metadata: DSAtrtllmAttentionMetadata,
            topk_indices_buffer: torch.Tensor) -> None:
        if self.indexer_mode not in ("indexcache", "indexcache-hisa"):
            return
        if not self.skip_topk:
            metadata._blaise_indexcache_topk = topk_indices_buffer

    # --- Cross-step Top-K reuse (decode only) ---------------------------------
    # sparse_attn_indexer runs in trtllm::mla_dsa_attn_inplace, which is excluded
    # from CUDA graph capture, so the Python step counter below is evaluated
    # eagerly every decode step and does not bake a stale decision into a graph.

    def _xstep_reuse_active(self, num_contexts: int,
                            num_generations: int) -> bool:
        """True when this step should reuse the cached cross-step Top-K.

        Only pure-decode steps (no prefill in the batch) participate; mixing in
        a prefill changes num_tokens / buffer shape and is rare, so we refresh
        on those. The decision is purely a function of the per-layer step
        counter modulo the configured stride.
        """
        if not self._xstep_enabled:
            return False
        if num_contexts != 0 or num_generations <= 0:
            return False
        if self._xstep_cache is None or self._xstep_refresh_kvlen is None:
            return False
        # counter==0 is a refresh step; 1..freq-1 reuse.
        return (self._xstep_counter % self.index_topk_step_freq) != 0

    def _xstep_reuse_decode(self, metadata: DSAtrtllmAttentionMetadata,
                            num_gen_tokens: int, num_contexts: int,
                            num_generations: int,
                            next_n: int) -> Optional[torch.Tensor]:
        """Zero-copy cross-step reuse for a pure-decode step.

        Returns the per-layer cached Top-K tensor directly (no fresh buffer, no
        full copy), after an in-place recency patch of only its trailing columns
        for the positions appended since the last refresh. The cache window is
        short, so the patch touches at most (freq-1)*next_n columns -- a handful
        of writes vs the ~24us logits MQA + Top-K kernels it replaces. Returns
        None if the cache shape no longer matches (caller recomputes).
        """
        cached = self._xstep_cache
        if (cached is None or cached.shape[0] != num_gen_tokens
                or cached.shape[1] != self.index_topk
                or self._xstep_batch_tokens != num_gen_tokens):
            return None

        if not self._xstep_recency_patch:
            # Frozen reuse: return the cached selection unchanged (0 extra
            # launches). The newest <=(freq-1)*next_n positions may be absent;
            # at the short strides used in practice this is within the same
            # approximation budget as cross-layer FSSS reuse.
            return cached

        # Recency patch: overwrite the trailing `delta` columns with the newly
        # appended absolute positions so every new token is selectable. The
        # previous PyTorch implementation issued ~20 dependent tiny launches
        # (~125us, a NET LOSS vs the ~21us logits+Top-K it replaces). It is now a
        # single fused kernel launch (trtllm::indexer_xstep_recency_patch): one
        # block per cached row computes delta = clamp(cur_end - refresh_end, 0,
        # max_delta) and writes refresh_end + delta - 1 - col_off into the last
        # max_delta columns where col_off < delta, leaving the rest untouched
        # (element-wise identical to the old block; verified jaccard=1.0). Only
        # the scalar max_delta is cached per shape.
        key = (cached.device, num_gen_tokens, next_n)
        max_delta = self._xstep_idx_cache.get(key)
        if max_delta is None:
            max_delta = min(
                max(int((self.index_topk_step_freq - 1) * next_n), next_n),
                self.index_topk)
            self._xstep_idx_cache[key] = max_delta
        gen_kv = metadata.kv_lens_cuda_runtime[
            num_contexts:num_contexts + num_generations]
        if gen_kv.dtype != torch.int32:
            gen_kv = gen_kv.to(torch.int32)
        gen_kv = gen_kv.contiguous()
        torch.ops.trtllm.indexer_xstep_recency_patch(
            cached, self._xstep_refresh_kvlen, gen_kv, next_n, max_delta)
        return cached

    def _xstep_store_decode(self, metadata: DSAtrtllmAttentionMetadata,
                            topk_indices_buffer: torch.Tensor,
                            num_ctx_tokens: int, num_gen_tokens: int,
                            num_contexts: int, num_generations: int,
                            next_n: int) -> None:
        """Snapshot the freshly computed decode Top-K + the per-row refresh end
        position so subsequent reuse steps can patch recency."""
        if not self._xstep_enabled:
            return
        gen_slice = slice(num_ctx_tokens, num_ctx_tokens + num_gen_tokens)
        self._xstep_cache = topk_indices_buffer[gen_slice, :].clone()
        gen_kv = metadata.kv_lens_cuda_runtime[
            num_contexts:num_contexts + num_generations]
        row_indices = (torch.arange(num_gen_tokens, device=gen_kv.device)
                       // next_n)
        next_n_offset = (torch.arange(num_gen_tokens, device=gen_kv.device)
                         % next_n)
        self._xstep_refresh_kvlen = (
            gen_kv[row_indices] - next_n + next_n_offset + 1).to(torch.int32)
        self._xstep_batch_tokens = num_gen_tokens

    def _hisa_block_topk(self, num_blocks: int) -> int:
        min_blocks = math.ceil(self.index_topk / self.hisa_block_size)
        if self.hisa_compression_ratio > 0:
            block_topk = math.ceil(num_blocks / self.hisa_compression_ratio)
        else:
            block_topk = self.hisa_block_topk
        return min(max(block_topk, min_blocks), num_blocks)

    def _hisa_arange(self, length: int, device: torch.device) -> torch.Tensor:
        key = (device, length)
        value = self._hisa_range_cache.get(key)
        if value is None:
            value = torch.arange(length, device=device)
            self._hisa_range_cache[key] = value
        return value

    def _hisa_full_int32(self, length: int, fill: int,
                         device: torch.device) -> torch.Tensor:
        key = (device, length, fill)
        value = self._hisa_full_cache.get(key)
        if value is None:
            value = torch.full((length, ),
                               fill,
                               dtype=torch.int32,
                               device=device)
            self._hisa_full_cache[key] = value
        return value

    def _hisa_step_invariants(
        self, metadata: Optional["DSAtrtllmAttentionMetadata"],
        kv_lens: torch.Tensor, num_rows: int, next_n: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        """Layer-invariant HISA decode index math, memoized per step.

        row_to_batch / prefix_lens / block_counts depend only on the step's
        kv_lens and shapes, never on the layer; eagerly they re-ran as ~7
        small int kernels in every layer's indexer. The first indexer layer
        of a step computes them; the rest reuse the tensors. metadata
        prepare()/kv-len updates clear the slot, and under CUDA graphs the
        layer-0 kernels are captured once and replay against the live
        kv_lens buffer, so every step still sees fresh values.
        """
        # The capture flag in the key forces the first captured layer to
        # rebuild inside the graph instead of hitting tensors memoized by the
        # eager warmup pass (which the replayed graph would never refresh).
        use_memo = metadata is not None and _hisa_step_memo_enabled()
        key = (kv_lens.data_ptr(), num_rows, next_n, self.hisa_block_size,
               torch.cuda.is_current_stream_capturing())
        cached = getattr(metadata, "_hisa_step_invariants",
                         None) if use_memo else None
        if cached is not None and cached[0] == key:
            return cached[1]
        rows = self._hisa_arange(num_rows, device)
        row_to_batch = torch.div(rows, next_n, rounding_mode="floor")
        row_offset = rows % next_n
        prefix_lens = (kv_lens[row_to_batch] - next_n + row_offset + 1).to(
            torch.int32)
        block_counts = torch.div(prefix_lens + self.hisa_block_size - 1,
                                 self.hisa_block_size,
                                 rounding_mode="floor")
        vals = (row_to_batch, prefix_lens, block_counts,
                row_to_batch.to(torch.int32), kv_lens.to(torch.int64))
        if use_memo:
            metadata._hisa_step_invariants = (key, vals)
            # The rowspan memo keys off id(block_counts); drop it together
            # with the invariants so a rebuilt block_counts can never collide
            # with a stale entry through CPython address reuse.
            metadata._hisa_step_rowspan = None
        return vals

    def _should_use_hisa(self, max_kv_len: int) -> bool:
        if not self.enable_nvfp4_hisa:
            return False
        if self.indexer_mode != "indexcache-hisa":
            return False
        if self.hisa_execution_mode not in ("auto", "optimized", "reference",
                                            "preindexer_reference"):
            return False
        num_blocks = math.ceil(max_kv_len / self.hisa_block_size)
        if self.hisa_block_size * self._hisa_block_topk(
                num_blocks) < self.index_topk:
            return False
        return max_kv_len >= self.hisa_min_seq_len

    def _should_use_hisa_logits(self, max_kv_len: int) -> bool:
        return False

    def _should_use_hisa_pre_indexer(self, max_kv_len: int) -> bool:
        if not self.use_fp4:
            return False
        if self.hisa_block_size != 128:
            return False
        return self._should_use_hisa(max_kv_len)

    def _hisa_e2m1_values(self, device: torch.device) -> torch.Tensor:
        values = self._hisa_e2m1_cache.get(device)
        if values is None:
            values = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
                                  dtype=torch.float32,
                                  device=device)
            self._hisa_e2m1_cache[device] = values
        return values

    def _dequantize_indexer_nvfp4(self, values: torch.Tensor,
                                  scales: torch.Tensor) -> torch.Tensor:
        low = values & 0x0f
        high = (values >> 4) & 0x0f
        codes = torch.empty((*values.shape[:-1], values.shape[-1] * 2),
                            dtype=torch.long,
                            device=values.device)
        codes[..., 0::2] = low.long()
        codes[..., 1::2] = high.long()
        magnitudes = self._hisa_e2m1_values(values.device)[codes & 0x07]
        signs = torch.where((codes & 0x08) != 0, -1.0, 1.0)
        dim_ids = self._hisa_arange(codes.shape[-1], values.device)
        scale_shifts = ((dim_ids // 32) * 8).to(torch.int32)
        scale_exp = ((scales.to(torch.int32).unsqueeze(-1) >>
                      scale_shifts.view(*([1] * scales.dim()), -1)) & 0xff)
        scale = torch.pow(2.0, scale_exp.to(torch.float32) - 127.0)
        return magnitudes * signs * scale

    def _hisa_mean_pool_indexer_cache(self, k_cache: torch.Tensor,
                                      block_table: torch.Tensor,
                                      kv_lens: torch.Tensor,
                                      max_blocks: int) -> torch.Tensor:
        if (k_cache.is_cuda
                and hasattr(torch.ops.trtllm, "indexer_hisa_mean_pool_nvfp4")):
            return torch.ops.trtllm.indexer_hisa_mean_pool_nvfp4(
                k_cache, block_table.contiguous(), kv_lens.to(torch.int32),
                max_blocks)
        block_size = self.hisa_block_size
        page_size = k_cache.shape[1]
        num_batches = block_table.shape[0]
        token_offsets = self._hisa_arange(block_size, k_cache.device).view(
            1, 1, block_size)
        block_starts = self._hisa_arange(max_blocks, k_cache.device).view(
            1, max_blocks, 1) * block_size
        token_indices = block_starts + token_offsets
        token_valid = token_indices < kv_lens.view(-1, 1, 1)
        logical_pages = torch.div(token_indices,
                                  page_size,
                                  rounding_mode="floor")
        logical_pages = logical_pages.clamp_max(block_table.shape[1] - 1)
        logical_pages = logical_pages.expand(num_batches, -1, -1)
        physical_pages = block_table.gather(
            1, logical_pages.reshape(num_batches, -1)).reshape(
                num_batches, max_blocks, block_size)
        physical_pages = physical_pages.clamp_min(0)
        page_offsets = token_indices % page_size
        page_offsets = page_offsets.expand(num_batches, -1, -1)
        tokens = k_cache[physical_pages, page_offsets, 0]
        token_values = tokens[..., :self.head_dim // 2]
        token_scales = tokens[..., self.head_dim // 2:].contiguous().view(
            torch.int32).view(
            num_batches, max_blocks, block_size)
        denom = token_valid.sum(dim=2).clamp_min(1).to(torch.float32)
        decoded = self._dequantize_indexer_nvfp4(token_values, token_scales)
        decoded = decoded.masked_fill(~token_valid.unsqueeze(-1), 0.0)
        return decoded.sum(dim=2) / denom.unsqueeze(-1)

    def _hisa_block_reps_from_page_cache(
        self,
        page_reps: torch.Tensor,
        page_counts: torch.Tensor,
        block_table: torch.Tensor,
        kv_lens: torch.Tensor,
        max_blocks: int,
        page_size: int,
    ) -> torch.Tensor:
        if (page_reps.is_cuda
                and hasattr(torch.ops.trtllm,
                            "indexer_hisa_block_reps_from_pages_nvfp4")):
            return torch.ops.trtllm.indexer_hisa_block_reps_from_pages_nvfp4(
                page_reps, page_counts, block_table.contiguous(),
                kv_lens.to(torch.int32), max_blocks, page_size)

        pages_per_hisa_block = self.hisa_block_size // page_size
        num_batches = block_table.shape[0]
        block_ids = self._hisa_arange(max_blocks, block_table.device).view(
            1, max_blocks, 1)
        page_offsets = self._hisa_arange(pages_per_hisa_block,
                                         block_table.device).view(
                                             1, 1, pages_per_hisa_block)
        logical_pages = block_ids * pages_per_hisa_block + page_offsets
        max_pages = torch.div(kv_lens + page_size - 1,
                              page_size,
                              rounding_mode="floor").view(-1, 1, 1)
        page_valid = logical_pages < max_pages
        logical_pages = logical_pages.clamp_max(block_table.shape[1] - 1)
        logical_pages = logical_pages.expand(num_batches, -1, -1)
        physical_pages = block_table.gather(
            1, logical_pages.reshape(num_batches, -1)).reshape(
                num_batches, max_blocks, pages_per_hisa_block)
        physical_pages = physical_pages.clamp_min(0)
        reps = page_reps[physical_pages.long()]
        counts = page_counts[physical_pages.long()].clamp_min(0).to(
            torch.float32)
        counts = counts.masked_fill(~page_valid, 0.0)
        denom = counts.sum(dim=2).clamp_min(1.0)
        return (reps * counts.unsqueeze(-1)).sum(dim=2) / denom.unsqueeze(-1)

    def _hisa_quantized_block_reps_from_page_cache(
        self,
        page_reps: torch.Tensor,
        page_counts: torch.Tensor,
        block_table: torch.Tensor,
        kv_lens: torch.Tensor,
        max_blocks: int,
        page_size: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not (page_reps.is_cuda and hasattr(
                torch.ops.trtllm,
                "indexer_hisa_quantized_block_reps_from_pages_nvfp4")):
            return None
        return torch.ops.trtllm.indexer_hisa_quantized_block_reps_from_pages_nvfp4(
            page_reps, page_counts, block_table.contiguous(),
            kv_lens.to(torch.int32), max_blocks, page_size)

    def _hisa_mean_pool_selected_blocks(self, k_cache: torch.Tensor,
                                        block_table: torch.Tensor,
                                        kv_lens: torch.Tensor,
                                        block_ids: torch.Tensor) -> torch.Tensor:
        block_size = self.hisa_block_size
        page_size = k_cache.shape[1]
        num_batches, num_blocks = block_ids.shape
        token_offsets = self._hisa_arange(block_size, k_cache.device).view(
            1, 1, block_size)
        token_indices = (block_ids.to(torch.int64).unsqueeze(-1) *
                         block_size + token_offsets)
        token_valid = token_indices < kv_lens.view(-1, 1, 1)
        logical_pages = torch.div(token_indices,
                                  page_size,
                                  rounding_mode="floor")
        logical_pages = logical_pages.clamp_max(block_table.shape[1] - 1)
        physical_pages = block_table.gather(
            1, logical_pages.reshape(num_batches, -1)).reshape(
                num_batches, num_blocks, block_size)
        physical_pages = physical_pages.clamp_min(0)
        page_offsets = token_indices % page_size
        tokens = k_cache[physical_pages, page_offsets, 0]
        token_values = tokens[..., :self.head_dim // 2]
        token_scales = tokens[..., self.head_dim // 2:].contiguous().view(
            torch.int32).view(num_batches, num_blocks, block_size)
        denom = token_valid.sum(dim=2).clamp_min(1).to(torch.float32)
        decoded = self._dequantize_indexer_nvfp4(token_values, token_scales)
        decoded = decoded.masked_fill(~token_valid.unsqueeze(-1), 0.0)
        return decoded.sum(dim=2) / denom.unsqueeze(-1)

    def _hisa_request_key(self, request_ids: Optional[Tuple[Union[int, str], ...]],
                          batch_size: int) -> Tuple[int, ...]:
        if request_ids is None:
            return tuple(range(batch_size))
        return tuple(hash(value) for value in request_ids)

    def _hisa_cached_block_reps(
        self,
        k_cache: torch.Tensor,
        block_table: torch.Tensor,
        kv_lens: torch.Tensor,
        max_blocks: int,
        request_ids: Optional[Tuple[Union[int, str], ...]],
        page_reps: Optional[torch.Tensor] = None,
        page_counts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if page_reps is not None and page_counts is not None:
            return self._hisa_block_reps_from_page_cache(
                page_reps, page_counts, block_table, kv_lens, max_blocks,
                k_cache.shape[1])
        if torch.cuda.is_current_stream_capturing():
            return self._hisa_mean_pool_indexer_cache(k_cache, block_table,
                                                      kv_lens, max_blocks)
        batch_size = block_table.shape[0]
        key = (k_cache.device.index or 0, k_cache.data_ptr(),
               block_table.data_ptr(),
               self._hisa_request_key(request_ids, batch_size), max_blocks)
        cached = self._hisa_rep_cache.get(key)
        if len(self._hisa_rep_cache) > 4 and cached is None:
            self._hisa_rep_cache.clear()
        kv_lens_i32 = kv_lens.to(torch.int32)
        if cached is None:
            reps = self._hisa_mean_pool_indexer_cache(k_cache, block_table,
                                                      kv_lens_i32, max_blocks)
            self._hisa_rep_cache[key] = {
                "reps": reps,
                "kv_lens": kv_lens_i32.clone(),
            }
            return reps

        reps = cached["reps"]
        cached_lens = cached["kv_lens"]
        if (reps.shape[0] != batch_size or reps.shape[1] < max_blocks
                or torch.any(kv_lens_i32 < cached_lens).item()):
            reps = self._hisa_mean_pool_indexer_cache(k_cache, block_table,
                                                      kv_lens_i32, max_blocks)
            cached["reps"] = reps
            cached["kv_lens"] = kv_lens_i32.clone()
            return reps

        changed = torch.nonzero(kv_lens_i32 != cached_lens,
                                as_tuple=False).flatten()
        if changed.numel() == 0:
            return reps

        row_lens = kv_lens_i32.index_select(0, changed)
        block_ids = torch.div(row_lens.clamp_min(1) - 1,
                              self.hisa_block_size,
                              rounding_mode="floor").clamp_max(
                                  max_blocks - 1).view(-1, 1)
        updated = self._hisa_mean_pool_selected_blocks(
            k_cache, block_table.index_select(0, changed), row_lens,
            block_ids)
        reps[changed.long(), block_ids.flatten().long(), :] = updated[:, 0, :]
        cached_lens[changed] = row_lens
        return reps

    def _hisa_select_blocks_tensor_ops(
        self,
        q_flat: torch.Tensor,
        q_scale_flat: torch.Tensor,
        weights_flat: torch.Tensor,
        reps: torch.Tensor,
        prefix_lens: torch.Tensor,
        block_topk: int,
        next_n: int,
        block_counts: Optional[torch.Tensor] = None,
        row_to_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q_dequant = self._dequantize_indexer_nvfp4(q_flat, q_scale_flat)
        if row_to_batch is None:
            row_to_batch = torch.div(
                self._hisa_arange(q_flat.shape[0], q_flat.device),
                next_n,
                rounding_mode="floor")
        reps_rows = reps.index_select(0, row_to_batch.long())
        with _tf32_matmul_enabled():
            dots = torch.bmm(q_dequant, reps_rows.transpose(1, 2))
        block_scores = (dots.clamp_min_(0.0) *
                        weights_flat.unsqueeze(-1)).sum(dim=1)
        if block_counts is None:
            block_counts = torch.div(prefix_lens + self.hisa_block_size - 1,
                                     self.hisa_block_size,
                                     rounding_mode="floor")
        block_ids = self._hisa_arange(reps.shape[1], q_flat.device).view(1, -1)
        block_scores = block_scores.masked_fill(
            block_ids >= block_counts.view(-1, 1), float("-inf"))
        top_blocks = torch.empty((q_flat.shape[0], block_topk),
                                 dtype=torch.int32,
                                 device=q_flat.device)
        torch.ops.trtllm.indexer_topk_decode(block_scores, block_counts,
                                             top_blocks, 1, block_topk)
        return top_blocks

    def _hisa_select_blocks_deepgemm_fp4(
        self,
        q_flat: torch.Tensor,
        q_scale_flat: torch.Tensor,
        weights_flat: torch.Tensor,
        reps: Optional[torch.Tensor],
        prefix_lens: torch.Tensor,
        block_topk: int,
        next_n: int,
        quantized_reps: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        max_blocks: Optional[int] = None,
        block_counts: Optional[torch.Tensor] = None,
        row_to_batch_i32: Optional[torch.Tensor] = None,
        metadata: Optional["DSAtrtllmAttentionMetadata"] = None,
    ) -> Optional[torch.Tensor]:
        if quantized_reps is None:
            if not hasattr(torch.ops.trtllm,
                           "indexer_hisa_quantize_block_reps_nvfp4"):
                return None
            assert reps is not None
            max_blocks = reps.shape[1]
            block_rep_values, block_rep_scales = (
                torch.ops.trtllm.indexer_hisa_quantize_block_reps_nvfp4(
                    reps.contiguous()))
        else:
            block_rep_values, block_rep_scales = quantized_reps
            assert max_blocks is not None
        if block_counts is None:
            block_counts = torch.div(prefix_lens + self.hisa_block_size - 1,
                                     self.hisa_block_size,
                                     rounding_mode="floor")
        if row_to_batch_i32 is None:
            row_to_batch_i32 = torch.div(
                self._hisa_arange(q_flat.shape[0], q_flat.device),
                next_n,
                rounding_mode="floor").to(torch.int32)
        # Layer-invariant row spans (and the width-correction gather index),
        # memoized per step alongside _hisa_step_invariants.
        use_rowspan_memo = metadata is not None and _hisa_step_memo_enabled()
        rowspan_key = (id(block_counts), int(max_blocks), q_flat.shape[0])
        rowspan = getattr(metadata, "_hisa_step_rowspan",
                          None) if use_rowspan_memo else None
        if rowspan is not None and rowspan[0] == rowspan_key:
            row_starts, row_ends, gather_idx = rowspan[1]
        else:
            row_starts = row_to_batch_i32 * max_blocks
            row_ends = row_starts + block_counts
            gather_idx = None
        block_scores = fp8_fp4_mqa_logits(
            (q_flat.contiguous().view(torch.int8), q_scale_flat.contiguous()),
            (block_rep_values.reshape(-1, self.head_dim // 2),
             block_rep_scales.reshape(-1)),
            weights_flat.contiguous(), row_starts, row_ends, False, max_blocks)
        if block_scores.shape[1] != max_blocks:
            if gather_idx is None:
                block_ids = self._hisa_arange(max_blocks,
                                              q_flat.device).view(1, -1)
                gather_idx = row_starts.to(torch.int64).view(-1,
                                                             1) + block_ids
            block_scores = block_scores.gather(1, gather_idx)
        if use_rowspan_memo:
            metadata._hisa_step_rowspan = (rowspan_key,
                                           (row_starts, row_ends, gather_idx))
        top_blocks = torch.empty((q_flat.shape[0], block_topk),
                                 dtype=torch.int32,
                                 device=q_flat.device)
        torch.ops.trtllm.indexer_topk_decode(block_scores, block_counts,
                                             top_blocks, 1, block_topk)
        return top_blocks

    def _hisa_select_blocks(
        self,
        q_flat: torch.Tensor,
        q_scale_flat: torch.Tensor,
        weights_flat: torch.Tensor,
        reps: torch.Tensor,
        prefix_lens: torch.Tensor,
        block_topk: int,
        next_n: int,
        block_counts: Optional[torch.Tensor] = None,
        row_to_batch: Optional[torch.Tensor] = None,
        row_to_batch_i32: Optional[torch.Tensor] = None,
        metadata: Optional["DSAtrtllmAttentionMetadata"] = None,
    ) -> torch.Tensor:
        if q_flat.is_cuda and reps.shape[1] > _HISA_FUSED_BLOCK_SCORE_MAX_BLOCKS:
            top_blocks = self._hisa_select_blocks_deepgemm_fp4(
                q_flat,
                q_scale_flat,
                weights_flat,
                reps,
                prefix_lens,
                block_topk,
                next_n,
                block_counts=block_counts,
                row_to_batch_i32=row_to_batch_i32,
                metadata=metadata)
            if top_blocks is not None:
                return top_blocks

        use_fused_scores = (q_flat.is_cuda
                            and reps.shape[1] <= _HISA_FUSED_BLOCK_SCORE_MAX_BLOCKS
                            and hasattr(torch.ops.trtllm,
                                        "indexer_hisa_block_scores_nvfp4"))
        if use_fused_scores:
            block_scores = torch.ops.trtllm.indexer_hisa_block_scores_nvfp4(
                q_flat.contiguous(), q_scale_flat.contiguous(),
                weights_flat.contiguous(), reps.contiguous(), prefix_lens,
                block_topk, next_n, self.hisa_block_size)
            if block_counts is None:
                block_counts = torch.div(prefix_lens + self.hisa_block_size -
                                         1,
                                         self.hisa_block_size,
                                         rounding_mode="floor")
            top_blocks = torch.empty((q_flat.shape[0], block_topk),
                                     dtype=torch.int32,
                                     device=q_flat.device)
            torch.ops.trtllm.indexer_topk_decode(block_scores, block_counts,
                                                 top_blocks, 1, block_topk)
            return top_blocks

        return self._hisa_select_blocks_tensor_ops(q_flat,
                                                   q_scale_flat,
                                                   weights_flat,
                                                   reps,
                                                   prefix_lens,
                                                   block_topk,
                                                   next_n,
                                                   block_counts=block_counts,
                                                   row_to_batch=row_to_batch)

    def _hisa_topk_from_nvfp4_cache(
        self, q_values: torch.Tensor, q_scales: torch.Tensor,
        k_cache: torch.Tensor, block_table: torch.Tensor,
        kv_lens: torch.Tensor, weights: torch.Tensor, next_n: int,
        num_sms: int,
        request_ids: Optional[Tuple[Union[int, str], ...]] = None,
        page_reps: Optional[torch.Tensor] = None,
        page_counts: Optional[torch.Tensor] = None,
        metadata: Optional["DSAtrtllmAttentionMetadata"] = None,
    ) -> Optional[torch.Tensor]:
        if not q_values.is_cuda:
            return None
        capturing = torch.cuda.is_current_stream_capturing()
        # Candidate budget must track live kv, not the static block-table width
        # (block_table.shape[1] * k_cache.shape[1] is the absolute KV allocation
        # = 132096, which froze candidate_len at 33024 and broke HISA scaling).
        # metadata.max_gen_kv_len is the capture-frozen, sync-free per-graph kv
        # ceiling — it is `int(kv_lens[gen].max())` computed once in prepare()
        # from the host kv_lens, so it is identical to the live device max here
        # whenever no overlap/spec-dec runtime correction split them (the prod
        # decode path). Reuse it in BOTH the captured and eager branches so the
        # eager (warmup / non-bucketed-batch) decode step no longer pays a D2H
        # `kv_lens.max().item()` sync. block_topk is monotonic in max_blocks, so
        # using the frozen ceiling only ever shrinks the candidate band (never
        # OOB). Fall back to the original live device read only when the host
        # int is unavailable (metadata None / not yet populated), which keeps
        # the eager corner bit-identical to the pre-hoist behavior.
        if metadata is not None and metadata.max_gen_kv_len > 0:
            max_kv_len = metadata.max_gen_kv_len
        elif capturing:
            max_kv_len = block_table.shape[1] * k_cache.shape[1]
        else:
            max_kv_len = int(kv_lens.max().item())
        if not self._should_use_hisa_pre_indexer(max_kv_len):
            return None
        if q_values.dim() != 4 or q_values.shape[-1] != self.head_dim // 2:
            return None

        num_batches = q_values.shape[0]
        num_rows = q_values.shape[0] * q_values.shape[1]
        max_blocks = math.ceil(max_kv_len / self.hisa_block_size)
        block_topk = self._hisa_block_topk(max_blocks)
        candidate_len = block_topk * self.hisa_block_size
        topk = min(self.index_topk, candidate_len)
        if candidate_len < self.index_topk:
            return None

        q_flat = q_values.reshape(num_rows, self.n_heads, self.head_dim // 2)
        q_scale_flat = q_scales.reshape(num_rows, self.n_heads)
        weights_flat = weights.reshape(num_rows, self.n_heads)
        (row_to_batch, prefix_lens, block_counts, row_to_batch_i32,
         kv_lens_i64) = self._hisa_step_invariants(metadata, kv_lens,
                                                   num_rows, next_n,
                                                   q_values.device)
        # Eager-only guard: bail out of HISA when the shortest row's causal
        # prefix is below index_topk (the candidate band could not fill a full
        # selection). Under the uniform-decode invariant asserted in prepare()
        # (all gen requests share one kv length, dsa.py:1732) every row's prefix
        # equals `max_gen_kv_len - next_n + row_offset + 1`, whose minimum over
        # row_offset >= 0 is `max_gen_kv_len - next_n + 1`. Derive it from the
        # capture-frozen host int so the eager (warmup / non-bucketed-batch)
        # decode step no longer pays a `prefix_lens.min().item()` D2H sync. Fall
        # back to the exact device reduction only when the host ceiling is
        # absent, keeping the non-uniform corner bit-identical to before.
        if not capturing:
            if metadata is not None and metadata.max_gen_kv_len > 0:
                min_prefix = metadata.max_gen_kv_len - next_n + 1
            else:
                min_prefix = int(prefix_lens.min().item())
            if min_prefix < self.index_topk:
                return None

        top_blocks = None
        quantized_reps = None
        if page_reps is not None and page_counts is not None:
            quantized_reps = self._hisa_quantized_block_reps_from_page_cache(
                page_reps, page_counts, block_table, kv_lens, max_blocks,
                k_cache.shape[1])
        if quantized_reps is not None:
            top_blocks = self._hisa_select_blocks_deepgemm_fp4(
                q_flat,
                q_scale_flat,
                weights_flat,
                None,
                prefix_lens,
                block_topk,
                next_n,
                quantized_reps,
                max_blocks,
                block_counts=block_counts,
                row_to_batch_i32=row_to_batch_i32,
                metadata=metadata)
        if top_blocks is None:
            reps = self._hisa_cached_block_reps(k_cache, block_table,
                                                kv_lens_i64, max_blocks,
                                                request_ids, page_reps,
                                                page_counts)
            top_blocks = self._hisa_select_blocks(
                q_flat,
                q_scale_flat,
                weights_flat,
                reps,
                prefix_lens,
                block_topk,
                next_n,
                block_counts=block_counts,
                row_to_batch=row_to_batch,
                row_to_batch_i32=row_to_batch_i32,
                metadata=metadata)
        # H3b: per-row live candidate count (TRTLLM_OPTRT_HISA_PERROW_CAND). Bound to
        # None here so it is defined across every execution-mode branch; populated lazily
        # at the first per-row use below.
        cand_count = None
        if self.hisa_execution_mode in ("auto", "optimized"):
            pages_per_hisa_block = self.hisa_block_size // k_cache.shape[1]
            if hasattr(torch.ops.trtllm, "indexer_hisa_candidate_pages"):
                candidate_page_table = torch.ops.trtllm.indexer_hisa_candidate_pages(
                    top_blocks, block_table.contiguous(), next_n,
                    pages_per_hisa_block)
            else:
                top_blocks_i64 = top_blocks.to(torch.int64)
                page_offsets = self._hisa_arange(pages_per_hisa_block,
                                                 q_values.device)
                candidate_pages = (top_blocks_i64.unsqueeze(-1) *
                                   pages_per_hisa_block +
                                   page_offsets).reshape(num_rows, -1)
                candidate_page_table = block_table[row_to_batch.long()].gather(
                    1, candidate_pages.clamp_min(0).long())
            # H3b: per-row live-length candidate scaling (TRTLLM_OPTRT_HISA_PERROW_CAND).
            # Walk only each row's live candidate width instead of the uniform band
            # ceiling `candidate_len`. The candidate-score GEMM derives per-q
            # `num_kv = ceil(mContextLens[q] / block_kv)` and walks only that range,
            # so a shorter per-row `candidate_context_lens` shrinks the GEMM walk while
            # the schedule (recomputed below from the same tensor) stays consistent.
            # Buffer allocations are unchanged: only the length-arg value shrinks; the
            # `candidate_scores` width stays `candidate_len` (tail rows ride as -inf,
            # already masked downstream). `cand_count` is pure device arithmetic on
            # `prefix_lens` (no host sync) so it is recomputed correctly each graph replay.
            if _hisa_perrow_cand():
                block_counts_row = block_counts
                cand_count = (block_counts_row.clamp_max(block_topk) *
                              self.hisa_block_size).to(torch.int32)
                candidate_context_lens = cand_count.view(num_rows, 1)
            else:
                candidate_context_lens = torch.full((num_rows, 1),
                                                    candidate_len,
                                                    dtype=torch.int32,
                                                    device=q_values.device)
            # SM3 hoist: consume the once-per-step prebuilt candidate schedule
            # instead of rebuilding it here on every recompute-"F" layer. The
            # schedule is a pure function of `candidate_context_lens` (whose
            # values are reproduced bit-for-bit by the prepare-time build from
            # the same `kv_lens`/`prefix_lens` and the same H3b length policy),
            # `_DG_SCHEDULE_BLOCK_KV`, and `num_sms`. A `(num_rows, next_n,
            # candidate_len)` provenance signature guards the substitution: if it
            # does not match what prepare built (e.g. a draft-only next_n call, or
            # the flag was off at prepare time), fall back to the in-line rebuild
            # so the result is always identical. `num_sms == metadata.num_sms`
            # here (same value passed at the call site), so the prebuilt buffer's
            # `num_sms`-dependent shape matches.
            candidate_schedule = None
            if _hoist_hisa_sched() and metadata is not None and \
                    metadata.hisa_candidate_schedule_sig == (num_rows, next_n,
                                                             candidate_len):
                candidate_schedule = metadata.hisa_candidate_schedule_buffer
            if candidate_schedule is None:
                candidate_schedule = get_paged_mqa_logits_metadata(
                    candidate_context_lens, _DG_SCHEDULE_BLOCK_KV, num_sms)
            candidate_scores = fp8_fp4_paged_mqa_logits(
                (q_flat.reshape(num_rows, 1, self.n_heads,
                                self.head_dim // 2).view(torch.int8),
                 q_scale_flat.reshape(num_rows, 1, self.n_heads)),
                k_cache,
                weights_flat,
                candidate_context_lens,
                candidate_page_table,
                candidate_schedule,
                candidate_len,
            )
            if hasattr(torch.ops.trtllm, "indexer_hisa_mask_scores"):
                torch.ops.trtllm.indexer_hisa_mask_scores(
                    candidate_scores, top_blocks, prefix_lens,
                    self.hisa_block_size)
            else:
                top_blocks_i64 = top_blocks.to(torch.int64)
                offsets = self._hisa_arange(self.hisa_block_size,
                                            q_values.device)
                candidate_indices = (top_blocks_i64.unsqueeze(-1) *
                                     self.hisa_block_size + offsets).reshape(
                                         num_rows, candidate_len)
                # -1-padded top_blocks slots derive negative candidate
                # indices; they alias page 0 upstream and must be invalid.
                candidate_valid = ((candidate_indices >= 0) &
                                   (candidate_indices
                                    < prefix_lens.view(-1, 1)))
                candidate_scores = candidate_scores.masked_fill(
                    ~candidate_valid, float("-inf"))
        else:
            q_dequant = self._dequantize_indexer_nvfp4(q_flat, q_scale_flat)
            top_blocks_i64 = top_blocks.to(torch.int64)
            offsets = self._hisa_arange(self.hisa_block_size, q_values.device)
            candidate_indices = (top_blocks_i64.unsqueeze(-1) *
                                 self.hisa_block_size + offsets).reshape(
                                     num_rows, candidate_len)
            # -1-padded top_blocks slots derive negative candidate indices;
            # they alias page 0 upstream and must be invalid.
            candidate_valid = ((candidate_indices >= 0) &
                               (candidate_indices
                                < prefix_lens.view(-1, 1)))
            candidate_pages = torch.div(candidate_indices,
                                        k_cache.shape[1],
                                        rounding_mode="floor")
            candidate_offsets = candidate_indices % k_cache.shape[1]
            physical_pages = block_table[row_to_batch.long()].gather(
                1, candidate_pages.clamp_min(0).long())
            candidate_cache = k_cache[physical_pages.clamp_min(0),
                                      candidate_offsets, 0]
            cand_values = candidate_cache[..., :self.head_dim // 2]
            cand_scales = candidate_cache[...,
                                          self.head_dim // 2:].contiguous(
                                          ).view(torch.int32).view(
                                              num_rows, candidate_len)
            candidate_scores = torch.zeros((num_rows, candidate_len),
                                           dtype=torch.float32,
                                           device=q_values.device)
            for group in range(4):
                lo = group * 16
                hi = lo + 16
                k_dequant = self._dequantize_indexer_nvfp4(
                    cand_values[..., lo:hi], cand_scales)
                q_group = q_dequant[..., group * 32:(group + 1) * 32]
                dots = torch.matmul(q_group, k_dequant.transpose(1, 2))
                candidate_scores += (dots.clamp_min_(0.0) *
                                     weights_flat.unsqueeze(-1)).sum(dim=1)
            candidate_scores = candidate_scores.masked_fill(~candidate_valid,
                                                            float("-inf"))

        selected = torch.empty((num_rows, topk),
                               dtype=torch.int32,
                               device=q_values.device)
        # H3b: feed the per-row live candidate width to the candidate top-k. A shorter
        # per-row length both restricts the walk to [0, cand_count) and (via the kernel's
        # on-device adaptive final-sort) flips slow radix -> fast insertion for short rows.
        # Valid candidates already occupy the contiguous front [0, cand_count) (block ids
        # are front-packed with -1 padding, tokens past prefix_lens are -inf), so the
        # truncation drops only padding that cannot enter the top-k -> SET-identical.
        # `cand_count` is reused from the GEMM branch when present, else recomputed from
        # the in-scope `prefix_lens` (pure device arithmetic, graph-safe, no host sync).
        if _hisa_perrow_cand():
            if cand_count is None:
                cand_count = (block_counts.clamp_max(block_topk) *
                              self.hisa_block_size).to(torch.int32)
            selected_lengths = cand_count
        else:
            selected_lengths = self._hisa_full_int32(num_rows, candidate_len,
                                                     q_values.device)
        torch.ops.trtllm.indexer_topk_decode(candidate_scores, selected_lengths,
                                             selected, 1, topk)
        if (self.hisa_execution_mode in ("auto", "optimized")
                and hasattr(torch.ops.trtllm, "indexer_hisa_remap_selected")):
            return torch.ops.trtllm.indexer_hisa_remap_selected(
                selected, top_blocks, prefix_lens, self.hisa_block_size,
                self.index_topk)

        top_blocks_i64 = top_blocks.to(torch.int64)
        offsets = self._hisa_arange(self.hisa_block_size, q_values.device)
        candidate_indices = (top_blocks_i64.unsqueeze(-1) *
                             self.hisa_block_size + offsets).reshape(
                                 num_rows, candidate_len)
        # indexer_topk_decode pads short rows' `selected` with -1; clamp for
        # the gather and sentinel them, along with negative candidate indices
        # from -1-padded top_blocks slots (the -1 sentinel is what
        # convert_req_index_to_global expects for invalid entries).
        topk_indices = candidate_indices.gather(1,
                                                selected.clamp_min(0).long())
        topk_indices = topk_indices.masked_fill(
            (selected < 0) | (topk_indices < 0)
            | (topk_indices >= prefix_lens.view(-1, 1)), -1)
        if topk < self.index_topk:
            padding = torch.full((num_rows, self.index_topk - topk),
                                 -1,
                                 dtype=torch.int64,
                                 device=q_values.device)
            topk_indices = torch.cat((topk_indices, padding), dim=1)
        return topk_indices.to(torch.int32)

    def _hisa_topk_from_logits(
            self, logits: torch.Tensor, row_starts: torch.Tensor,
            row_ends: torch.Tensor,
            row_starts_are_zero: bool = False) -> Optional[torch.Tensor]:
        if logits.numel() == 0:
            return None
        if logits.is_cuda and torch.cuda.is_current_stream_capturing():
            max_kv_len = logits.shape[1]
        else:
            max_kv_len = int((row_ends - row_starts).max().item())
        if not self._should_use_hisa_logits(max_kv_len):
            return None

        num_rows, num_cols = logits.shape
        block_size = self.hisa_block_size
        num_blocks = math.ceil(num_cols / block_size)
        block_topk = self._hisa_block_topk(num_blocks)
        topk = min(self.index_topk, num_cols)

        full_rows = False
        if logits.is_cuda and not torch.cuda.is_current_stream_capturing():
            full_rows = (torch.count_nonzero(row_starts).item() == 0
                         and torch.count_nonzero(row_ends != num_cols).item()
                         == 0)

        if full_rows:
            scores = logits.float()
        elif row_starts_are_zero:
            cols = self._hisa_arange(num_cols, logits.device)
            valid = cols.unsqueeze(0) < row_ends.unsqueeze(1)
            scores = logits.float().masked_fill(~valid, float("-inf"))
        else:
            cols = self._hisa_arange(num_cols, logits.device)
            valid = (cols.unsqueeze(0) >= row_starts.unsqueeze(1)) & (
                cols.unsqueeze(0) < row_ends.unsqueeze(1))
            scores = logits.float().masked_fill(~valid, float("-inf"))

        pad = num_blocks * block_size - num_cols
        if pad == 0:
            padded_scores = scores
        else:
            padded_scores = F.pad(scores, (0, pad), value=float("-inf"))
        block_scores = padded_scores.reshape(num_rows, num_blocks, block_size)
        block_scores = block_scores.amax(dim=-1)

        block_ids = block_scores.topk(block_topk, dim=-1, sorted=False)[1]
        offsets = self._hisa_arange(block_size, logits.device)
        selected_indices = (block_ids.unsqueeze(-1) * block_size +
                            offsets).reshape(num_rows, -1)
        selected_scores = padded_scores.gather(1, selected_indices)
        if selected_scores.is_cuda:
            selected_relative = torch.empty((num_rows, topk),
                                            dtype=torch.int32,
                                            device=logits.device)
            selected_lengths = self._hisa_full_int32(
                num_rows, selected_scores.shape[1], logits.device)
            torch.ops.trtllm.indexer_topk_decode(
                selected_scores, selected_lengths, selected_relative, 1, topk)
        else:
            selected_relative = selected_scores.topk(
                topk, dim=-1, sorted=False)[1]
        relative = selected_indices.gather(1, selected_relative.long())
        if not full_rows:
            if row_starts_are_zero:
                relative = relative.masked_fill(
                    relative >= row_ends.unsqueeze(1), -1)
            else:
                relative = relative - row_starts.unsqueeze(1)
                lengths = row_ends - row_starts
                relative = relative.masked_fill(
                    (relative < 0) | (relative >= lengths.unsqueeze(1)), -1)

        result = torch.full((num_rows, self.index_topk),
                            -1,
                            dtype=torch.int32,
                            device=logits.device)
        result[:, :topk] = relative.to(torch.int32)
        return result

    def post_load_weights(self):
        """Fuse wk + weights_proj into a single GEMM over their shared input.

        NVFP4-quantized indexer: one [head_dim + n_heads <- hidden] NVFP4
        GEMM (single activation quantize + single weight stream); see
        _FusedWkWpNvfp4. Unquantized indexer: single FP32 weight for an
        F.linear GEMM under allow_tf32 (TF32 tensor cores on Ampere+).
        """
        # Verified vs the split Linears on real layer-5 REAP weights at
        # M in {4, 16}: wk slice bit-identical (max|diff| == 0), weights_proj
        # slice cos >= 0.999995 (sole delta: the wp_out_scale fold rounding;
        # exact in f32 out_dtype), cutlass and cuBLASLt agree, 1.96-1.97x vs
        # the split pair. The earlier cos=0.0 gate failure was a harness
        # artifact, not a kernel bug: this checkpoint carries no indexer
        # activation scales and linear.py builds before 5bc2b2cb8 left
        # module.input_scale as an UNINITIALIZED Parameter, so the driver's
        # split reference and the fused path both quantized with a garbage
        # static scale and emitted exact zeros (cosine(0, 0) == 0).
        # TRTLLM_INDEXER_FUSE_WK_WP=0 disables.
        if os.environ.get('TRTLLM_INDEXER_FUSE_WK_WP', '1') != '0':
            self._fused_wk_wp_nvfp4 = _FusedWkWpNvfp4.build(
                self.wk, self.weights_proj, self._indexer_nvfp4_backends)
            if self._fused_wk_wp_nvfp4 is not None:
                self._fused_wk_wp_weight = None
                return
        # wk: [head_dim, hidden_size] + weights_proj: [n_heads, hidden_size]
        # → fused: [head_dim + n_heads, hidden_size]
        wk_weight = self.wk.weight.data
        weights_proj_weight = self.weights_proj.weight.data
        if (wk_weight.shape[-1] == self.hidden_size and
                weights_proj_weight.shape[-1] == self.hidden_size):
            self._fused_wk_wp_weight = torch.cat(
                [wk_weight, weights_proj_weight], dim=0)
        else:
            self._fused_wk_wp_weight = None

    @staticmethod
    def prepare_one_prefill_chunk(
        metadata: DSAtrtllmAttentionMetadata,
        chunk_specs: List[Tuple[int, int, int, int]],
    ) -> IndexerPrefillChunkMetadata:
        """
        Build metadata for one prefill chunk for indexer forward pass.
        Handles both multi-request chunks and intra-request Q-block chunks.

        Args:
            metadata: Attention metadata
            chunk_specs: List of (req_idx, token_start_in_req, token_end_in_req, req_cum_start)
                        - token_start_in_req, token_end_in_req are indices into current batch context tokens
                        - For multi-request: multiple specs from different requests (full requests)
                        - For intra-request: single spec from one request's Q-block

        Note: Cached token counts are derived from metadata.host_ctx_cached_token_indptr
        """
        device = metadata.cu_seqlen_ks.device
        if len(chunk_specs) == 1:
            # Single request or intra-request Q-block
            req_idx, token_start_in_req, token_end_in_req, req_cum_start = chunk_specs[
                0]
            num_q_tokens = token_end_in_req - token_start_in_req

            # Get cached token count for this request from metadata
            num_cached = (
                metadata.host_ctx_cached_token_indptr[req_idx + 1] -
                metadata.host_ctx_cached_token_indptr[req_idx]).item()

            # For intra-request chunks: Q block attends to all previous K in the request
            # Q tokens [token_start_in_req:token_end_in_req] within the request's current tokens
            # K tokens [0:num_cached + token_end_in_req] within the request (causal attention)
            cu_seqlen_ks = torch.zeros(num_q_tokens,
                                       dtype=torch.int32,
                                       device='cpu')
            cu_seqlen_ke = torch.arange(token_start_in_req + 1,
                                        token_end_in_req + 1,
                                        dtype=torch.int32,
                                        device='cpu') + num_cached

            # Q token range in batch (indices into context tokens in the current batch)
            token_start = req_cum_start + token_start_in_req
            token_end = req_cum_start + token_end_in_req

            # K token range: index into full KV slot mapping (cached + current batch context tokens)
            kv_offset_in_extended = metadata.host_ctx_kv_indptr[req_idx].item()
            total_kv_for_req = num_cached + token_end_in_req
            k_token_start = kv_offset_in_extended
            k_token_end = kv_offset_in_extended + total_kv_for_req

        else:
            # Multi-request chunk: batch multiple full requests together
            # Extract sequence lengths for these requests
            req_seq_lens = []
            req_cached_lens = []
            first_req_idx = chunk_specs[0][0]

            for spec in chunk_specs:
                req_idx, token_start_in_req, token_end_in_req, _ = spec
                req_seq_lens.append(token_end_in_req - token_start_in_req)
                # Get cached token count from metadata
                num_cached = (
                    metadata.host_ctx_cached_token_indptr[req_idx + 1] -
                    metadata.host_ctx_cached_token_indptr[req_idx]).item()
                req_cached_lens.append(num_cached)

            req_seq_lens_tensor = torch.tensor(req_seq_lens,
                                               dtype=torch.int32,
                                               device='cpu')
            req_cached_lens_tensor = torch.tensor(req_cached_lens,
                                                  dtype=torch.int32,
                                                  device='cpu')
            num_q_tokens = sum(req_seq_lens)

            # Compute causal attention bounds for batched requests
            cu_seqlen_ks, cu_seqlen_ke = compute_cu_seqlen_kv_bounds_with_cache(
                req_seq_lens_tensor, len(chunk_specs), num_q_tokens,
                req_cached_lens_tensor)

            # Global Q token ranges (indices into ctx tokens in the current batch)
            token_start = chunk_specs[0][3]  # req_cum_start of first request
            token_end = token_start + num_q_tokens

            # K token range: index into full kv slot mapping (cached + current ctx tokens within the batch)
            kv_offset_in_extended = metadata.host_ctx_kv_indptr[
                first_req_idx].item()
            total_kv_len = sum(req_seq_lens_tensor +
                               req_cached_lens_tensor).item()
            k_token_start = kv_offset_in_extended
            k_token_end = kv_offset_in_extended + total_kv_len

        assert cu_seqlen_ks.shape[0] == num_q_tokens == token_end - token_start, \
            f"Indexer.prepare_one_prefill_chunk - cu_seqlen_ks length mismatch: {cu_seqlen_ks.shape[0]} != {num_q_tokens}"
        assert cu_seqlen_ke.shape[0] == num_q_tokens == token_end - token_start, \
            f"Indexer.prepare_one_prefill_chunk - cu_seqlen_ke length mismatch: {cu_seqlen_ke.shape[0]} != {num_q_tokens}"

        return IndexerPrefillChunkMetadata(
            cu_seqlen_ks=cu_seqlen_ks.to(device, non_blocking=True),
            cu_seqlen_ke=cu_seqlen_ke.to(device, non_blocking=True),
            token_start=token_start,
            token_end=token_end,
            k_token_start=k_token_start,
            k_token_end=k_token_end,
        )

    @staticmethod
    def recompute_slot_mappings(metadata: DSAtrtllmAttentionMetadata):
        """Recompute only slot_mapping_fp8/scale from the current block offsets.

        This is the subset of prepare() that maps each token to its flat cache
        position.  It is safe to call in isolation (e.g. during draft KV-cache
        replay) because it only touches slot-mapping buffers and reads
        block-offset / sequence metadata that the caller has already set up.
        """
        kv_cache_manager = metadata.kv_cache_manager
        if kv_cache_manager is None or not hasattr(kv_cache_manager,
                                                   'index_head_dim'):
            return

        seq_lens = metadata.seq_lens
        head_dim = kv_cache_manager.index_head_dim
        tokens_per_block = kv_cache_manager.tokens_per_block
        quant_block_size = kv_cache_manager.quant_block_size
        use_fp4 = kv_cache_manager.use_fp4
        # FP4 packs two E2M1 codes per byte; FP8 stores one byte per element.
        data_bytes_per_token = head_dim // 2 if use_fp4 else head_dim
        cached_tokens = metadata.kv_cache_params.num_cached_tokens_per_seq
        total_tokens = seq_lens.sum().item()

        start_positions = torch.tensor(cached_tokens, dtype=torch.int32)
        batch_size = len(metadata.request_ids)

        req_indices = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.int64, device='cpu'), seq_lens)

        token_offsets = torch.cat([
            torch.arange(seq_lens[i].item(), dtype=torch.int64, device='cpu')
            for i in range(batch_size)
        ])

        global_positions = start_positions[req_indices] + token_offsets

        fp8_flat_indices, scale_flat_indices = _compute_slot_mappings(
            global_positions,
            metadata.host_indexer_k_cache_block_offsets,
            req_indices,
            head_dim,
            tokens_per_block,
            quant_block_size,
            data_bytes_per_token=data_bytes_per_token,
        )

        metadata.host_slot_mapping_fp8[:total_tokens] = fp8_flat_indices
        metadata.host_slot_mapping_scale[:total_tokens] = scale_flat_indices

        metadata.slot_mapping_fp8[:total_tokens].copy_(
            metadata.host_slot_mapping_fp8[:total_tokens], non_blocking=True)
        metadata.slot_mapping_scale[:total_tokens].copy_(
            metadata.host_slot_mapping_scale[:total_tokens], non_blocking=True)

    @staticmethod
    def prepare(metadata: DSAtrtllmAttentionMetadata):
        """
        Prepare indexer for the forward pass.
        This should be called during metadata.prepare() stage.

        - Computes slot_mapping for KV cache updates
        - Prepares schedule_metadata for fp8_paged_mqa_logits
        - Stores generation request IDs for decode phase
        """
        kv_cache_manager = metadata.kv_cache_manager
        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        num_ctx_tokens = metadata.num_ctx_tokens
        seq_lens = metadata.seq_lens
        tokens_per_block = kv_cache_manager.tokens_per_block

        # Prepare for prefill phase if there are context requests
        if num_contexts > 0:
            # Compute attention window bounds for each query token in batched sequences
            # cu_seqlen_ks[i]: start index in global KV for query token i
            # cu_seqlen_ke[i]: end index (exclusive) in global KV for query token i
            host_seq_lens = seq_lens[:num_contexts]
            cached_tokens = metadata.kv_cache_params.num_cached_tokens_per_seq
            host_cached_tokens = torch.tensor(cached_tokens[:num_contexts],
                                              dtype=torch.int32,
                                              device='cpu')

            # When MLA chunked prefill is active, it already handles chunking
            # Indexer should just process the current MLA chunk as a single chunk
            has_mla_chunked_prefill = (
                metadata.enable_context_mla_with_cached_kv
                and metadata.runtime_features.chunked_prefill)

            if has_mla_chunked_prefill:
                # MLA chunked prefill is active - use single-chunk pattern for
                # indexer prefill chunks.
                chunk_specs = [(i, 0, host_seq_lens[i].item(),
                                host_seq_lens[:i].sum().item() if i > 0 else 0)
                               for i in range(num_contexts)]
                metadata.indexer_prefill_chunks = [
                    Indexer.prepare_one_prefill_chunk(
                        metadata,
                        chunk_specs,
                    )
                ]
            else:
                # Use indexer's own chunking logic to prevent L^2 complexity of indexer MQA logits computation for long sequences.
                # This is only used when MLA chunked prefill is not enabled.
                chunk_groups = split_prefill_chunks(
                    host_seq_lens,
                    metadata.indexer_max_chunk_size,
                    start_idx=0,
                )

                if len(chunk_groups
                       ) > 1 or metadata.enable_context_mla_with_cached_kv:
                    metadata.indexer_prefill_chunks = [
                        Indexer.prepare_one_prefill_chunk(
                            metadata,
                            chunk_specs,
                        ) for chunk_specs in chunk_groups
                    ]
                else:
                    metadata.indexer_prefill_chunks = None

            host_cu_seqlen_ks, host_cu_seqlen_ke = compute_cu_seqlen_kv_bounds_with_cache(
                host_seq_lens, num_contexts, num_ctx_tokens, host_cached_tokens)

            metadata.cu_seqlen_ks[:num_ctx_tokens].copy_(host_cu_seqlen_ks,
                                                         non_blocking=True)
            metadata.cu_seqlen_ke[:num_ctx_tokens].copy_(host_cu_seqlen_ke,
                                                         non_blocking=True)

        # Prepare for decode phase if there are generation requests
        if num_generations > 0:
            # Prepare schedule metadata for fp8_paged_mqa_logits
            # This is a preprocessing step that computes scheduling information for the kernel
            if not metadata.use_expanded_buffers_for_mtp:
                # Write 2D kv_lens (broadcast same kv_len across next_n positions).
                gen_seq_lens = metadata.kv_lens_cuda_runtime[
                    num_contexts:num_contexts + num_generations]
                next_n_cap = metadata.kv_lens_cuda_2d.shape[1]
                metadata.kv_lens_cuda_2d[:num_generations, :next_n_cap].copy_(
                    gen_seq_lens.unsqueeze(-1).expand(-1, next_n_cap))
                # Build the next_n=1 schedule (used by MTP draft layers).
                # Use the contiguous 1D gen slice reshaped to (num_gen, 1);
                # slicing kv_lens_cuda_2d's first column would be a strided
                # view that fails the metadata kernel's contiguous assertion.
                context_lens_next_n1 = gen_seq_lens.view(-1, 1)
                # `_DG_SCHEDULE_BLOCK_KV` (= 64) instead of cache `tokens_per_block`:
                # see module-level constant comment for the SPLIT_KV=256 alignment.
                scheduler_metadata_buffer = get_paged_mqa_logits_metadata(
                    context_lens_next_n1, _DG_SCHEDULE_BLOCK_KV,
                    metadata.num_sms)
                metadata.scheduler_metadata_buffer.copy_(
                    scheduler_metadata_buffer, non_blocking=True)
                # MTP main forward uses next_n = 1 + max_draft_tokens; build
                # a separate schedule because the metadata kernel reads next_n
                # from context_lens.size(1).
                if metadata.max_draft_tokens > 0:
                    context_lens_full_next_n = metadata.kv_lens_cuda_2d[:
                                                                        num_generations, :
                                                                        next_n_cap]
                    scheduler_metadata_buffer_full_next_n = get_paged_mqa_logits_metadata(
                        context_lens_full_next_n, _DG_SCHEDULE_BLOCK_KV,
                        metadata.num_sms)
                    metadata.scheduler_metadata_buffer_full_next_n.copy_(
                        scheduler_metadata_buffer_full_next_n,
                        non_blocking=True)
            else:
                # Expand schedule metadata buffer (only generation). The new
                # DeepGEMM API requires 2D; each expanded token becomes a (1,)
                # row.
                num_tokens = metadata.num_generations * (
                    1 + metadata.max_draft_tokens)
                kv_lens_expanded_2d = metadata.kv_lens_expanded_cuda[:
                                                                     num_tokens].view(
                                                                         -1, 1)
                scheduler_metadata_buffer_expanded = get_paged_mqa_logits_metadata(
                    kv_lens_expanded_2d, _DG_SCHEDULE_BLOCK_KV,
                    metadata.num_sms)
                metadata.scheduler_metadata_buffer_expanded.copy_(
                    scheduler_metadata_buffer_expanded, non_blocking=True)

            # DSL atom-split schedule. Picker decision was cached on
            # `metadata.dsl_{expand_factor, atom}` at metadata prepare
            # time; only build the expanded schedule when picker chose to
            # split (factor > 1). Runtime mutually exclusive with the `else`
            # branch above (latter requires `use_expanded_buffers_for_mtp`
            # which is False under DSL).
            if metadata.expand_for_dsl and metadata.num_generations > 0 \
                    and metadata.dsl_expand_factor > 1:
                expand_factor = metadata.dsl_expand_factor
                num_tokens = metadata.num_generations * expand_factor
                kv_lens_expanded_2d = metadata.kv_lens_expanded_cuda[:
                                                                     num_tokens].view(
                                                                         -1, 1)
                scheduler_metadata_buffer_expanded = get_paged_mqa_logits_metadata(
                    kv_lens_expanded_2d, _DG_SCHEDULE_BLOCK_KV,
                    metadata.num_sms)
                metadata.scheduler_metadata_buffer_expanded.copy_(
                    scheduler_metadata_buffer_expanded, non_blocking=True)

            # SM3 hoist: build the HISA candidate-score schedule once here, from
            # the gen slice of the device kv_lens (the same tensor the F-layer
            # GEMM derives `prefix_lens` from). `on_update_kv_lens` rebuilds it
            # again after any runtime kv_lens correction (overlap / spec-dec), so
            # the consumed schedule always reflects the corrected lengths.
            gen_kv_lens = metadata.kv_lens_cuda_runtime[
                num_contexts:num_contexts + num_generations]
            metadata._maybe_build_hisa_candidate_schedule(gen_kv_lens)

        # Compute slot_mapping for all requests (both context and generation)
        Indexer.recompute_slot_mappings(metadata)

        # When chunked prefill or KVCache reuse is enabled, we need to gather the full KV for indexer's logit computation.
        # Indexer's own chunking does not need full KV gathering, instead it gathers only the current chunk with loop-based gathering.
        _need_full_kv_gathering = num_contexts > 0 and metadata.enable_context_mla_with_cached_kv
        if _need_full_kv_gathering:
            head_dim = kv_cache_manager.index_head_dim
            quant_block_size = kv_cache_manager.quant_block_size
            use_fp4 = kv_cache_manager.use_fp4
            data_bytes_per_token = head_dim // 2 if use_fp4 else head_dim
            cached_tokens = metadata.kv_cache_params.num_cached_tokens_per_seq
            start_positions = torch.tensor(cached_tokens, dtype=torch.int32)

            total_kv_len = metadata.host_ctx_kv_indptr[num_contexts].item()
            total_kv_per_request = seq_lens[:
                                            num_contexts] + start_positions[:
                                                                            num_contexts]
            host_slot_mapping_fp8_fullkv = torch.empty(
                total_kv_len, dtype=torch.int64, pin_memory=prefer_pinned())
            host_slot_mapping_scale_fullkv = torch.empty(
                total_kv_len, dtype=torch.int64, pin_memory=prefer_pinned())

            fullkv_req_indices = torch.repeat_interleave(
                torch.arange(num_contexts, dtype=torch.int64, device='cpu'),
                total_kv_per_request)

            kv_positions = torch.cat([
                torch.arange(total_kv_per_request[i].item(),
                             dtype=torch.int64,
                             device='cpu') for i in range(num_contexts)
            ])

            fp8_flat_indices, scale_flat_indices = _compute_slot_mappings(
                kv_positions,
                metadata.host_indexer_k_cache_block_offsets,
                fullkv_req_indices,
                head_dim,
                tokens_per_block,
                quant_block_size,
                data_bytes_per_token=data_bytes_per_token,
            )

            host_slot_mapping_fp8_fullkv[:total_kv_len] = fp8_flat_indices
            host_slot_mapping_scale_fullkv[:total_kv_len] = scale_flat_indices

            assert len(fp8_flat_indices) == total_kv_len, \
                f"host_slot_mapping_fp8_fullkv/host_slot_mapping_scale_fullkv length mismatch: {len(fp8_flat_indices)} != total_kv_len={total_kv_len}"

            # Store extended mappings for indexer full KV gathering
            metadata.slot_mapping_fp8_fullkv = host_slot_mapping_fp8_fullkv.cuda(
                non_blocking=True)
            metadata.slot_mapping_scale_fullkv = host_slot_mapping_scale_fullkv.cuda(
                non_blocking=True)
        else:
            metadata.slot_mapping_fp8_fullkv = metadata.slot_mapping_fp8
            metadata.slot_mapping_scale_fullkv = metadata.slot_mapping_scale

    def _update_k_cache(self, k_fp8: torch.Tensor, k_scale: torch.Tensor,
                        metadata: DSAtrtllmAttentionMetadata) -> None:
        """
        Insert/append k values and scales into the indexer k cache using pre-computed slot mappings.
        Uses flat byte indices with vectorized scatter.

        Args:
            k_fp8: FP8 quantized k tensor, shape [total_tokens, head_dim]
            k_scale: Scaling factors, shape [total_tokens, head_dim // quant_block_size]
        """
        if metadata.kv_cache_manager is None or metadata.slot_mapping_fp8 is None:
            return

        # LayerSplit note (owner_local_alloc + CpType.LAYERSPLIT): this write is
        # ownership-blind ON PURPOSE and is LOAD-BEARING, not wasted work.
        #   - get_indexer_k_cache_buffers returns the shared dense/indexer
        #     scratch for a non-owned layer, the real pool slot for an owned
        #     one. Either way this scatter is the SOLE writer of THIS step's
        #     new-token K into that slot/scratch.
        #   - The per-layer owner->peer broadcast (Indexer.forward, M5e) runs
        #     BEFORE this write, so it cannot carry K that has not been
        #     computed yet; it only resyncs previously-committed active blocks.
        #     The indexer then gathers the FULL KV range (slot_mapping_*_fullkv)
        #     from this slot, so the new tokens' K MUST be written here.
        #   - It is correct on every CP rank because the indexer K projection
        #     (Indexer.wk) carries no TP/CP mapping and runs over the
        #     CP-replicated full hidden state, so each rank computes
        #     bit-identical K. The HISA page-rep recompute below is likewise
        #     correct (built from this same replicated K). Do NOT "optimize"
        #     this away for non-owners -- skipping it drops the new tokens' K
        #     on non-owner ranks. (The separate question of whether cached
        #     PREFIX blocks of non-owned layers are covered by the active-block
        #     broadcast set is a broadcast-COVERAGE concern, not a write
        #     concern -- see the 1B report.)
        k_cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(
            self.layer_idx)

        num_tokens = k_fp8.shape[0]

        # The C++ op reinterprets k_fp8 (FP8) and k_scale (float32) as raw
        # bytes internally and only reads the first num_tokens entries from
        # the slot mapping buffers, avoiding Python-side view/slice overhead.
        torch.ops.trtllm.indexer_k_cache_scatter_op(k_fp8, k_scale, k_cache,
                                                    metadata.slot_mapping_fp8,
                                                    metadata.slot_mapping_scale,
                                                    num_tokens)
        if (self.use_fp4 and self.enable_nvfp4_hisa
                and hasattr(metadata.kv_cache_manager,
                            "get_indexer_hisa_page_rep_buffers")
                and hasattr(torch.ops.trtllm,
                            "indexer_hisa_update_page_reps_nvfp4")):
            page_reps, page_counts = (
                metadata.kv_cache_manager.get_indexer_hisa_page_rep_buffers(
                    self.layer_idx))
            torch.ops.trtllm.indexer_hisa_update_page_reps_nvfp4(
                k_cache, page_reps, page_counts, metadata.slot_mapping_fp8,
                num_tokens)

    def _call_mqa_logits(self, q_fp8: torch.Tensor, k_fp8: torch.Tensor,
                         k_scale: torch.Tensor, weights: torch.Tensor,
                         cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor,
                         q_scale: Optional[torch.Tensor]) -> torch.Tensor:
        """Dispatch to fp8_mqa_logits or fp8_fp4_mqa_logits based on use_fp4.

        For FP4 the gather output is typed as FP8 for historical reasons;
        reinterpret the bytes as the expected int8 / int32 layouts. DeepGEMM
        asserts kv_sf is 1D in both modes, so flatten the scale here.
        """
        if self.use_fp4:
            k_fp4_bytes = k_fp8.view(torch.int8)
            k_scale_int32 = k_scale.view(torch.int32).reshape(-1)
            # q_scale arrives here as (chunk_tokens, n_heads, 1) — fused_cat_fp4
            # emits one int32 per (token, head) carrying four UE8M0 exponents,
            # pre_indexer_proj reshapes back to (N, n_heads, 1), and
            # sparse_attn_indexer chunk-slices on axis 0. The DeepGEMM FP4
            # kernel asserts q_sf is 2D, so collapse the trailing unit axis.
            q_scale_2d = q_scale.reshape(-1, self.n_heads)
            return fp8_fp4_mqa_logits(
                (q_fp8, q_scale_2d),
                (k_fp4_bytes, k_scale_int32),
                weights,
                cu_seqlen_ks,
                cu_seqlen_ke,
            )
        return fp8_mqa_logits(q_fp8, (k_fp8, k_scale.reshape(-1)), weights,
                              cu_seqlen_ks, cu_seqlen_ke)

    def _call_paged_mqa_logits(self, q_decode: torch.Tensor,
                               k_cache: torch.Tensor,
                               weights_decode: torch.Tensor,
                               context_lens: torch.Tensor,
                               block_table: torch.Tensor,
                               scheduler_metadata_buffer: torch.Tensor,
                               max_seq_len: int,
                               q_scale: Optional[torch.Tensor]) -> torch.Tensor:
        """Dispatch to fp8_paged_mqa_logits or fp8_fp4_paged_mqa_logits."""
        if self.use_fp4:
            return fp8_fp4_paged_mqa_logits(
                (q_decode, q_scale), k_cache, weights_decode, context_lens,
                block_table, scheduler_metadata_buffer, max_seq_len)
        return fp8_paged_mqa_logits(q_decode, k_cache, weights_decode,
                                    context_lens, block_table,
                                    scheduler_metadata_buffer, max_seq_len)

    def sparse_attn_indexer(
        self,
        metadata: DSAtrtllmAttentionMetadata,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
        weights: torch.Tensor,
        use_custom_topk: bool = True,
        q_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the indexer TopK kernel for both prefill and decode phases.

        q_scale is only consumed by the FP4 dispatch; FP8 path ignores it.
        """
        # DSACacheManager hardcodes quant_block_size=128 (see its __init__);
        # FP4 uses per-block-32 UE8M0 scales but packs four of them into one
        # int32 so the cache-layout contribution is the same 4 bytes/token as
        # FP8 per-block-128 at head_dim=128.
        assert metadata.kv_cache_manager is None or \
            metadata.kv_cache_manager.quant_block_size == 128, \
            f"Unexpected quant_block_size {metadata.kv_cache_manager.quant_block_size if metadata.kv_cache_manager else 'N/A'}"
        cached_topk = self._get_indexcache_topk(metadata,
                                                hidden_states.shape[0])
        if cached_topk is not None:
            return cached_topk
        # A reuse ("S") layer that misses the indexcache would fall through
        # to the compute path below, but pre_indexer_proj returns
        # uninitialized q/k buffers on reuse layers, so that path would
        # score garbage. The owning ("F") layer for each reuse group runs
        # earlier in the same step (FSSS layer 0 is always "F"; no PP split
        # at prod) and stores a same-shape cache, so the hit above is
        # guaranteed at prod. Fail loudly if that invariant is ever broken
        # (e.g. an unexpected layer pattern or pipeline split) rather than
        # silently emitting wrong TopK.
        assert not self.skip_topk, (
            f"indexer layer {self.layer_idx}: skip_topk reuse layer missed "
            "the indexcache; pre_indexer_proj projections are not computed "
            "on reuse layers so the compute path cannot run")

        # Update the indexer k cache before prefill chunks gather from it.
        self._update_k_cache(k_fp8, k_scale, metadata)

        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        num_ctx_tokens = metadata.num_ctx_tokens
        num_tokens = metadata.num_tokens

        has_decode = num_generations > 0
        has_prefill = num_contexts > 0
        num_gen_tokens = num_tokens - num_ctx_tokens

        # Cross-step reuse fast path (pure decode only). The K cache was already
        # appended above, so the logits MQA + Top-K can be skipped: return the
        # per-layer cached selection directly (zero-copy) with a tiny in-place
        # recency patch. Runs in the non-graph-captured mla_dsa_attn_inplace op,
        # so this Python branch is eager and safe. The counter still advances
        # below via the unified increment.
        if (self._xstep_reuse_active(num_contexts, num_generations)
                and not has_prefill):
            next_n = num_gen_tokens // num_generations
            reused = self._xstep_reuse_decode(metadata, num_gen_tokens,
                                              num_contexts, num_generations,
                                              next_n)
            if reused is not None:
                self._xstep_counter += 1
                self._maybe_store_indexcache_topk(metadata, reused)
                return reused

        topk_indices_buffer = torch.empty(
            (hidden_states.shape[0], self.index_topk),
            dtype=torch.int32,
            device=hidden_states.device)
        if not use_custom_topk or self.use_fp4:
            topk_indices_buffer[:hidden_states.shape[0]] = -1

        if has_prefill and not metadata.skip_indexer_for_ctx_reqs:
            # Use chunked prefill to reduce memory footprint
            if metadata.indexer_prefill_chunks is not None:

                # Default to 8192 if sparse_attention_config is not available (e.g., in unit tests)
                q_split_threshold = metadata.sparse_attention_config.q_split_threshold if metadata.sparse_attention_config is not None else 8192
                q_split_eligible = q_split_threshold >= 0 and metadata.mapping is not None and not metadata.mapping.enable_attention_dp and metadata.mapping.tp_size > 1

                if q_split_eligible:
                    tp_rank = metadata.mapping.tp_rank
                    tp_size = metadata.mapping.tp_size

                k_cache_4d = metadata.kv_cache_manager.get_indexer_k_cache_buffers(
                    self.layer_idx)

                gather_head_dim = self.head_dim // 2 if self.use_fp4 else self.head_dim
                for chunk in metadata.indexer_prefill_chunks:
                    num_k_tokens = chunk.k_token_end - chunk.k_token_start
                    chunk_k_fp8, chunk_k_scale = torch.ops.trtllm.indexer_k_cache_gather_op(
                        k_cache_4d, metadata.slot_mapping_fp8_fullkv,
                        metadata.slot_mapping_scale_fullkv, chunk.k_token_start,
                        num_k_tokens, gather_head_dim)

                    chunk_num_token = chunk.token_end - chunk.token_start
                    apply_q_split = q_split_eligible and chunk_num_token >= q_split_threshold
                    if apply_q_split:
                        chunk_q_start = chunk_num_token * tp_rank // tp_size
                        chunk_q_end = chunk_num_token * (tp_rank + 1) // tp_size
                    else:
                        chunk_q_start = 0
                        chunk_q_end = chunk_num_token

                    global_q_start = chunk.token_start + chunk_q_start
                    global_q_end = chunk.token_start + chunk_q_end

                    chunk_q_scale = q_scale[global_q_start:global_q_end,
                                            ...] if self.use_fp4 else None
                    logits = self._call_mqa_logits(
                        q_fp8[global_q_start:global_q_end, ...],
                        chunk_k_fp8,
                        chunk_k_scale,
                        weights[global_q_start:global_q_end, ...],
                        chunk.cu_seqlen_ks[chunk_q_start:chunk_q_end],
                        chunk.cu_seqlen_ke[chunk_q_start:chunk_q_end],
                        chunk_q_scale,
                    )
                    hisa_topk = self._hisa_topk_from_logits(
                        logits, chunk.cu_seqlen_ks[chunk_q_start:chunk_q_end],
                        chunk.cu_seqlen_ke[chunk_q_start:chunk_q_end])
                    if hisa_topk is not None:
                        topk_indices_buffer[global_q_start:global_q_end, :] = \
                            hisa_topk
                    elif use_custom_topk and not self.use_fp4:
                        torch.ops.trtllm.indexer_topk_prefill(
                            logits.contiguous(),
                            chunk.cu_seqlen_ks[chunk_q_start:chunk_q_end],
                            chunk.cu_seqlen_ke[chunk_q_start:chunk_q_end],
                            topk_indices_buffer[global_q_start:global_q_end, :])
                    else:
                        topk_indices = logits.topk(min(self.index_topk,
                                                       logits.shape[-1]),
                                                   dim=-1)[1]
                        topk_indices -= chunk.cu_seqlen_ks[
                            chunk_q_start:chunk_q_end][:, None]

                        mask_lo = topk_indices >= 0
                        mask_hi = topk_indices - (
                            chunk.cu_seqlen_ke[chunk_q_start:chunk_q_end] -
                            chunk.cu_seqlen_ks[chunk_q_start:chunk_q_end]
                        )[:, None] < 0
                        mask = mask_lo & mask_hi

                        # local indices per sequence
                        topk_indices = topk_indices.masked_fill(~mask, -1)

                        topk_indices_buffer[
                            global_q_start:global_q_end, :topk_indices.
                            shape[-1]] = topk_indices.to(dtype=torch.int32)

                    if apply_q_split:
                        q_sizes = [(r + 1) * chunk_num_token // tp_size -
                                   r * chunk_num_token // tp_size
                                   for r in range(tp_size)]
                        topk_indices_buffer[
                            chunk.token_start:chunk.token_end, :] = allgather(
                                topk_indices_buffer[
                                    global_q_start:global_q_end, :],
                                metadata.mapping,
                                dim=0,
                                sizes=q_sizes)
            else:
                # Fallback: single-pass indexer prefill (TODO: remove this once chunked prefill is fully tested)
                cu_seqlen_ks = metadata.cu_seqlen_ks[:num_ctx_tokens]
                cu_seqlen_ke = metadata.cu_seqlen_ke[:num_ctx_tokens]

                ctx_q_scale = q_scale[:num_ctx_tokens,
                                      ...] if self.use_fp4 else None
                logits = self._call_mqa_logits(
                    q_fp8[:num_ctx_tokens, ...],
                    k_fp8[:num_ctx_tokens, ...],
                    k_scale[:num_ctx_tokens, ...],
                    weights[:num_ctx_tokens, ...],
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    ctx_q_scale,
                )
                hisa_topk = self._hisa_topk_from_logits(
                    logits, cu_seqlen_ks, cu_seqlen_ke)
                if hisa_topk is not None:
                    topk_indices_buffer[:num_ctx_tokens, :] = hisa_topk
                elif use_custom_topk and not self.use_fp4:
                    torch.ops.trtllm.indexer_topk_prefill(
                        logits.contiguous(), cu_seqlen_ks, cu_seqlen_ke,
                        topk_indices_buffer[:num_ctx_tokens, :])
                else:
                    topk_indices = logits.topk(min(self.index_topk,
                                                   logits.shape[-1]),
                                               dim=-1)[1]
                    topk_indices -= cu_seqlen_ks[:, None]
                    mask_lo = topk_indices >= 0
                    mask_hi = topk_indices - (cu_seqlen_ke -
                                              cu_seqlen_ks)[:, None] < 0
                    mask = mask_lo & mask_hi

                    # local indices per sequence
                    topk_indices = topk_indices.masked_fill(~mask, -1)
                    topk_indices_buffer[:num_ctx_tokens, :topk_indices.
                                        shape[-1]] = topk_indices.to(
                                            dtype=torch.int32)
        elif has_prefill and metadata.skip_indexer_for_ctx_reqs:
            # Fill topk_indices_buffer with pre-defined dense topk indices
            topk_indices_buffer[:num_ctx_tokens, :] = \
                metadata.topk_indices_buffer[:num_ctx_tokens, :]

        if has_decode and not metadata.skip_indexer_for_gen_reqs:
            max_seq_len = metadata.kv_cache_manager.max_seq_len
            # Width-correct the decode logits buffer to a graph-safe bucket of
            # the live max-kv (see _indexer_logits_width). Allocation trim
            # only: no kernel cost or dispatch keys on the width (the topk
            # dispatch is gated on live kv, _DSL_TOPK_MIN_KV_LEN).
            # max_gen_kv_len is the captured constant under CUDA graphs
            # (== per-graph warmup kv).
            logits_width = _indexer_logits_width(metadata.max_gen_kv_len,
                                                 max_seq_len)
            # The all-generation-requests-share-one-decode-length invariant
            # (needed because the paged MQA logits + topk kernels assume no
            # padding) is asserted once per step in metadata.prepare(); no
            # per-layer host read here.

            # Reshape q for decode phase: [num_gen_tokens, ...] -> [batch_size, next_n, ...]
            q_decode = q_fp8[num_ctx_tokens:num_ctx_tokens + num_gen_tokens,
                             ...]
            batch_size = num_generations
            next_n = num_gen_tokens // num_generations

            # Because fp8_paged_mqa_logits can only support next_n == 1/2/4 on sm100, and
            # next_n == 1/2 on sm90, for other next_n, we need to flatten the q_decode tensor
            # and expand the corresponding metadata.
            if not metadata.use_expanded_buffers_for_mtp or next_n == 1:
                q_decode = q_decode.view(num_generations, -1, *q_fp8.shape[1:])
                # 2D context_lens slice from the pre-allocated buffer; matches
                # q_decode's (batch, next_n) layout required by the new
                # DeepGEMM paged MQA logits API.
                context_lens = metadata.kv_lens_cuda_2d[:num_generations, :
                                                        next_n].contiguous()
                block_table = metadata.indexer_k_cache_block_offsets[
                    num_contexts:num_contexts + num_generations]
                # The 2D-context_lens metadata kernel encodes next_n into the
                # schedule (via num_next_n_atoms). MTP forwards alternate
                # between the full-window call (next_n == 1+max_draft_tokens)
                # and per-token draft calls (next_n == 1), so we must select
                # the buffer that was populated for this next_n. The DSL path
                # uses its own schedule buffer (built with num_next_n_atoms=1
                # via a (num_gen, 1) input shape) and overrides this below.
                if next_n == 1:
                    scheduler_metadata_buffer = metadata.scheduler_metadata_buffer
                else:
                    scheduler_metadata_buffer = metadata.scheduler_metadata_buffer_full_next_n
            else:
                q_decode = q_decode.view(-1, 1, *q_fp8.shape[1:])
                num_tokens = q_decode.shape[0]
                # New API requires 2D; each expanded token becomes a (1,) row.
                context_lens = metadata.kv_lens_expanded_cuda[:num_tokens].view(
                    -1, 1)
                block_table = metadata.block_table_expanded[:num_tokens]
                scheduler_metadata_buffer = metadata.scheduler_metadata_buffer_expanded

            assert num_gen_tokens == batch_size * next_n
            weights_decode = weights[num_ctx_tokens:num_ctx_tokens +
                                     num_gen_tokens, ...]

            # Get k cache and call fp8_paged_mqa_logits with prepared decode metadata
            # [num_blocks, tokens_per_block, 1, head_dim + scale_size]
            k_cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(
                self.layer_idx)

            pre_hisa_topk = None
            if (use_custom_topk and self.use_fp4 and q_decode.shape[0]
                    == num_generations and q_decode.shape[1] == next_n):
                pre_hisa_q_scale = q_scale[num_ctx_tokens:num_ctx_tokens +
                                           num_gen_tokens, ...]
                pre_hisa_q_scale = pre_hisa_q_scale.view(
                    q_decode.shape[0], q_decode.shape[1], self.n_heads)
                pre_hisa_request_ids = None
                if metadata.request_ids is not None:
                    pre_hisa_request_ids = tuple(
                        metadata.request_ids[num_contexts:num_contexts +
                                             num_generations])
                page_reps = None
                page_counts = None
                if hasattr(metadata.kv_cache_manager,
                           "get_indexer_hisa_page_rep_buffers"):
                    page_reps, page_counts = (
                        metadata.kv_cache_manager.
                        get_indexer_hisa_page_rep_buffers(self.layer_idx))
                pre_hisa_topk = self._hisa_topk_from_nvfp4_cache(
                    q_decode.view(torch.uint8), pre_hisa_q_scale, k_cache,
                    block_table,
                    metadata.kv_lens_cuda_runtime[num_contexts:num_contexts +
                                                   num_generations],
                    weights_decode, next_n, metadata.num_sms,
                    pre_hisa_request_ids, page_reps, page_counts,
                    metadata=metadata)

            if pre_hisa_topk is None and self.use_cute_dsl_paged_mqa_logits:
                # DSL kernel design: 1 atom per q (atom = real next_n positions),
                # kNumNextNAtoms = 1 for any real next_n. The matching schedule
                # is `scheduler_metadata_buffer` — built in `Indexer.prepare()`
                # with a (num_gen, 1) input shape, which makes DeepGEMM's wrapper
                # compute `num_next_n_atoms = 1`. (DeepGEMM uses the same buffer
                # for its own next_n=1 kernel; DSL piggy-backs on it for all
                # real next_n values.) All next_n positions of a batch share
                # the same KV length on this path (kv_lens_cuda_2d broadcasts),
                # so passing the 1D contiguous kv_lens slice for context_lens
                # avoids materializing a 2D contiguous tensor per call.
                dsl_context_lens = metadata.kv_lens_cuda_runtime[
                    num_contexts:num_contexts + num_generations]
                if self.use_fp4:
                    # FP4 DSL signature splits DG's (q, sf_q) tuple into two
                    # separate args and requires q.dtype == uint8 (q_decode
                    # came in via the FP8 plumbing as int8; reinterpret with
                    # no copy). sf_q is the q_scale slice reshaped to
                    # (B, next_n, H) int32 — mirrors the non-DSL FP4 branch.
                    decode_q_scale = q_scale[num_ctx_tokens:num_ctx_tokens +
                                             num_gen_tokens, ...]
                    decode_q_scale = decode_q_scale.view(
                        q_decode.shape[0], q_decode.shape[1], self.n_heads)
                    dsl_q = q_decode.view(torch.uint8)
                    dsl_block_table = block_table
                    dsl_schedule_meta = metadata.scheduler_metadata_buffer

                    # DSL FP4 kernel natively supports next_n ∈ {1, 2, 3}.
                    # The wave-aware picker in `_pick_dsl_expand` is run
                    # once per metadata prepare and the result cached on
                    # `metadata.dsl_{expand_factor, atom}`. Trigger expand
                    # whenever the picker decided to split (factor > 1),
                    # regardless of next_n — this lets next_n ∈ {2, 3} also
                    # benefit from atom-split when low-batch leaves SMs idle,
                    # in addition to the mandatory next_n=4 case.
                    if metadata.dsl_expand_factor > 1:
                        factor = metadata.dsl_expand_factor
                        eff_next_n = metadata.dsl_atom
                        exp_B = num_generations * factor
                        dsl_q = dsl_q.reshape(exp_B, eff_next_n, self.n_heads,
                                              self.head_dim // 2)
                        decode_q_scale = decode_q_scale.reshape(
                            exp_B, eff_next_n, self.n_heads)
                        dsl_context_lens = metadata.kv_lens_expanded_cuda[:
                                                                          exp_B]
                        dsl_block_table = metadata.block_table_expanded[:exp_B]
                        dsl_schedule_meta = (
                            metadata.scheduler_metadata_buffer_expanded)

                    logits_decode = torch.ops.trtllm.cute_dsl_fp4_paged_mqa_logits(
                        dsl_q, decode_q_scale, k_cache, weights_decode,
                        dsl_context_lens, dsl_block_table, dsl_schedule_meta,
                        logits_width,
                        output_dtype=self.logits_dtype)
                else:
                    # FP8 DSL kernel natively supports next_n ∈ {1, 2, 3, 4}.
                    # Apply wave-aware atom-split when the picker decided to
                    # split (factor > 1) — typically benefits small-batch /
                    # low-ntask configs by raising SM utilization at the cost
                    # of factor× KV HBM re-reads. Picker decision was cached
                    # on metadata.{dsl_expand_factor, dsl_atom} during prepare.
                    dsl_q = q_decode
                    fp8_ctx_lens = dsl_context_lens
                    fp8_block_table = block_table
                    fp8_schedule_meta = metadata.scheduler_metadata_buffer
                    if metadata.dsl_expand_factor > 1:
                        factor = metadata.dsl_expand_factor
                        atom = metadata.dsl_atom
                        exp_B = num_generations * factor
                        dsl_q = q_decode.reshape(exp_B, atom, self.n_heads,
                                                 self.head_dim)
                        fp8_ctx_lens = metadata.kv_lens_expanded_cuda[:exp_B]
                        fp8_block_table = metadata.block_table_expanded[:exp_B]
                        fp8_schedule_meta = (
                            metadata.scheduler_metadata_buffer_expanded)
                    logits_decode = torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits(
                        dsl_q, k_cache, weights_decode, fp8_ctx_lens,
                        fp8_block_table, fp8_schedule_meta, logits_width,
                        output_dtype=self.logits_dtype)
            elif pre_hisa_topk is None:
                decode_q_scale = q_scale[num_ctx_tokens:num_ctx_tokens +
                                         num_gen_tokens,
                                         ...] if self.use_fp4 else None
                if self.use_fp4:
                    # q_decode shape is either (num_generations, next_n, n_heads,
                    # head_dim/2) [non-expanded] or (batch*next_n, 1, n_heads,
                    # head_dim/2) [expanded]. Match q_scale's batch/next_n dims.
                    decode_q_scale = decode_q_scale.view(
                        q_decode.shape[0], q_decode.shape[1], self.n_heads)
                logits_decode = self._call_paged_mqa_logits(
                    q_decode, k_cache, weights_decode, context_lens,
                    block_table, scheduler_metadata_buffer, logits_width,
                    decode_q_scale)

            if use_custom_topk:
                # Kernel expects kv_lens (total cache length), not seq_lens (new tokens)
                # This is because rowEnd = seq_len - next_n + offset + 1
                gen_kv_lens_cuda = metadata.kv_lens_cuda_runtime[
                    num_contexts:num_contexts + num_generations]

                pre_idx = None
                heuristic_scratch = None
                # heuristic_prev_topk is sized to the number of local (owned)
                # layers and indexed by the local pool offset. Under LayerSplit
                # owner-local alloc a non-owned layer has no offset and no row,
                # so skip the temporal hint for it (its KV arrives via the
                # owner broadcast; the hint is a perf accelerator, not a
                # correctness input). layer_offsets always contains the layer
                # in the replicated posture, so this is a no-op there.
                local_layer = metadata.kv_cache_manager.layer_offsets.get(
                    self.layer_idx) if self._enable_heuristic_topk else None
                if local_layer is not None:
                    # Pass prev_topk directly; the +1 temporal offset is
                    # handled inside the C++ kernel (preIdxOffset += 1).
                    pre_idx = metadata.heuristic_prev_topk[
                        local_layer, :num_generations]
                    heuristic_scratch = \
                        metadata.heuristic_scratch_values[
                            :num_gen_tokens]

                # CuTE DSL top-k allocates O(num_gen_tokens * kv_len) global
                # memory. Beyond 256 tokens the extra memory becomes significant,
                # so we cap it at 256 for now and fall back to the CUDA C++
                # indexer_topk_decode. This limit can be removed if GPU memory
                # is not a bottleneck.
                hisa_topk = pre_hisa_topk
                # The HISA-from-logits decode path is statically gated by
                # _should_use_hisa_logits (currently always False -- HISA at
                # decode runs from the NVFP4 cache via pre_hisa_topk, not from
                # dense logits). Building row_indices / row_starts / row_ends
                # to feed _hisa_topk_from_logits is then dead work: four extra
                # device ops captured into the decode graph per F-layer whose
                # result is discarded (the function returns None). Gate the
                # whole preamble on the same cheap, capture-safe predicate the
                # callee uses (max_kv_len = logits column count needs no .item()
                # / host sync), so the non-HISA-logits prod path skips it
                # entirely and only the genuinely-enabled path pays for it.
                if (hisa_topk is None
                        and logits_decode.shape[0] == num_gen_tokens
                        and self._should_use_hisa_logits(
                            logits_decode.shape[1])):
                    row_indices = torch.arange(
                        num_gen_tokens, device=logits_decode.device) // next_n
                    next_n_offset = torch.arange(
                        num_gen_tokens, device=logits_decode.device) % next_n
                    row_starts = torch.zeros(num_gen_tokens,
                                             device=logits_decode.device,
                                             dtype=gen_kv_lens_cuda.dtype)
                    row_ends = (gen_kv_lens_cuda[row_indices] - next_n +
                                next_n_offset + 1)
                    hisa_topk = self._hisa_topk_from_logits(
                        logits_decode, row_starts, row_ends,
                        row_starts_are_zero=True)
                if hisa_topk is not None:
                    topk_indices_buffer[num_ctx_tokens:num_ctx_tokens +
                                        num_gen_tokens, :] = hisa_topk
                elif (self.use_cute_dsl_topk and num_gen_tokens <= 256
                      and metadata.max_gen_kv_len >= _DSL_TOPK_MIN_KV_LEN):
                    # DSL allocates O(num_gen_tokens * live_kv_len) scratch, so
                    # it is capped at 256 tokens. It beats the C++ kernel only at
                    # long live kv (>= _DSL_TOPK_MIN_KV_LEN); below that the C++
                    # kernel wins -- it walks only [0, live_kv) per row, while the
                    # DSL kernel's cost scales with the padded logits WIDTH. The
                    # earlier width-based override (route to DSL whenever logits
                    # width >= 12288) was REMOVED: it forced DSL at
                    # prod (width 132096, live kv ~4.6k) on the false premise that
                    # the C++ kernel slows at width >= 12288. Direct B200
                    # measurement (3-seed, CUDA-graph replay) shows C++ is
                    # ~width-independent (~11us flat) and ~1.7x FASTER than DSL at
                    # the prod short-kv regime, with a bit-identical selected set
                    # (recall 1.0). So prod now routes to the C++ kernel below.
                    # Enable the fused single-pass multi-CTA cluster radix top-k:
                    # ~1.3-1.6x faster than the default 2-pass+merge at the prod
                    # logits width 132096 with a bit-identical selected set
                    # (IoU=1.0, same radix, same lower-index tie-break). The op
                    # falls back to single-CTA/distributed internally when a shape
                    # rejects the cluster launch, so it is safe in this branch.
                    torch.ops.trtllm.cute_dsl_indexer_topk_decode(
                        logits_decode, gen_kv_lens_cuda,
                        topk_indices_buffer[num_ctx_tokens:num_ctx_tokens +
                                            num_gen_tokens, :], self.index_topk,
                        next_n,
                        single_pass_multi_cta=True,
                        single_pass_multi_cta_cluster=True)
                else:
                    torch.ops.trtllm.indexer_topk_decode(
                        logits_decode,
                        gen_kv_lens_cuda,
                        topk_indices_buffer[num_ctx_tokens:num_ctx_tokens +
                                            num_gen_tokens, :],
                        next_n,
                        self.index_topk,
                        pre_idx=pre_idx,
                        heuristic_scratch=heuristic_scratch)
            else:
                # padded
                positions = torch.arange(
                    logits_decode.shape[-1],
                    device=q_decode.device).unsqueeze(0).expand(
                        num_gen_tokens, -1)
                row_indices = torch.arange(num_gen_tokens,
                                           device=q_decode.device) // next_n
                next_n_offset = torch.arange(num_gen_tokens,
                                             device=q_decode.device) % next_n
                index_end_pos = (
                    metadata.kv_lens_cuda_runtime[num_contexts + row_indices] -
                    next_n + next_n_offset).unsqueeze(1)
                # index_end_pos: [B * N, 1]
                mask = positions <= index_end_pos
                # mask: [B * N, L]
                logits_decode = logits_decode.masked_fill(~mask, float('-inf'))
                topk_indices_decode = logits_decode.topk(
                    min(self.index_topk, logits_decode.shape[-1]),
                    dim=-1)[1].to(torch.int32)  # [B * N, K]
                # ensure we don't set indices for the top k
                # that is out of range(masked already)
                # this will happen if context length is shorter than K
                mask_decode = topk_indices_decode <= index_end_pos

                # local indices per sequence
                topk_indices_decode = topk_indices_decode.masked_fill(
                    ~mask_decode, -1)
                # Store in buffer
                topk_indices_buffer[num_ctx_tokens:num_ctx_tokens +
                                    num_gen_tokens, :topk_indices_decode.
                                    shape[-1]] = topk_indices_decode.to(
                                        dtype=torch.int32)

            if self._enable_heuristic_topk:
                # Mirror the read-side guard: a non-owned LayerSplit layer has
                # no heuristic_prev_topk row, so there is nothing to feed back.
                # layer_offsets always contains the layer in the replicated
                # posture, so this is a no-op there.
                local_layer = metadata.kv_cache_manager.layer_offsets.get(
                    self.layer_idx)
                if local_layer is not None:
                    decode_topk = topk_indices_buffer[
                        num_ctx_tokens:num_ctx_tokens + num_gen_tokens]
                    last_mtp_topk = decode_topk[next_n - 1::next_n]
                    metadata.heuristic_prev_topk[
                        local_layer, :num_generations].copy_(last_mtp_topk)

            # Refresh step: snapshot the freshly computed decode Top-K so the
            # next (freq-1) steps can reuse it. Only runs on the compute path.
            self._xstep_store_decode(metadata, topk_indices_buffer,
                                     num_ctx_tokens, num_gen_tokens,
                                     num_contexts, num_generations, next_n)

        elif has_decode and metadata.skip_indexer_for_gen_reqs:
            # Fill topk_indices_buffer with pre-defined dense topk indices
            topk_indices_buffer[num_ctx_tokens:num_tokens, :] = \
                metadata.topk_indices_buffer[num_ctx_tokens:num_tokens, :]

        # Advance the per-layer cross-step counter once per decode step. A pure
        # prefill / skip step does not advance it (the cache stays valid for the
        # next decode). Done after both reuse and compute so step 0 computes.
        if self._xstep_enabled and has_decode \
                and not metadata.skip_indexer_for_gen_reqs:
            self._xstep_counter += 1

        self._maybe_store_indexcache_topk(metadata, topk_indices_buffer)
        return topk_indices_buffer

    def _weight_scale(self, weights: torch.Tensor,
                      q_scale: torch.Tensor) -> torch.Tensor:
        """Apply quantization scale to indexer attention weights."""
        weights = _scale(weights, q_scale, self.weight_scale_factor)
        return weights

    def _qk_projection_and_rope(self, qr: torch.Tensor, indexer_k: torch.Tensor,
                                position_ids: torch.Tensor):
        """Project Q/K and apply RoPE"""
        q = self.wq_b(qr)
        k = self.k_norm(indexer_k)
        q = q.view(-1, self.n_heads, self.head_dim)
        q_pe, q_nope = q.split([self.rope_dim, self.head_dim - self.rope_dim],
                               dim=-1)
        k_pe, k_nope = k.split([self.rope_dim, self.head_dim - self.rope_dim],
                               dim=-1)
        q_pe, k_pe = self.rotary_emb(position_ids, [q_pe, k_pe.unsqueeze(1)])
        k_pe = k_pe[:, 0, :]
        return q_pe, q_nope, k_pe, k_nope

    def _qk_projection_no_rope(self, qr: torch.Tensor,
                               indexer_k: torch.Tensor):
        """Project Q/K and split pe/nope WITHOUT applying RoPE.

        Used by the fused RoPE+cat+FP4-quant path, which folds RoPE into the
        quantize kernel and so needs the pre-RoPE pe slices.
        """
        q = self.wq_b(qr)
        k = self.k_norm(indexer_k)
        q = q.view(-1, self.n_heads, self.head_dim)
        q_pe, q_nope = q.split([self.rope_dim, self.head_dim - self.rope_dim],
                               dim=-1)
        k_pe, k_nope = k.split([self.rope_dim, self.head_dim - self.rope_dim],
                               dim=-1)
        return q_pe, q_nope, k_pe, k_nope

    def _prep_q_or_k(self, qk_pe: torch.Tensor, qk_nope: torch.Tensor):
        """Concatenate and quantize for Q or K.

        FP8 mode: fused cat + FP8 quantize via CUDA kernel.
        FP4 mode: fused cat + per-block-32 FP4 E2M1 quantize via CUDA kernel.
        The returned packed bytes are int8 (two FP4 codes per byte) and the
        scale is int32 (four UE8M0 exponents packed little-endian).
        """
        if self.use_fp4:
            return torch.ops.trtllm.fused_cat_fp4(qk_pe, qk_nope)
        fp8_out, scale = torch.ops.trtllm.fused_cat_fp8(
            qk_pe, qk_nope, self.scale_fmt == "ue8m0")
        return fp8_out, scale

    def pre_indexer_proj(
        self, qr: torch.Tensor, hidden_states: torch.Tensor,
        position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        """Pure token-wise projections (CUDA-graph-capturable).

        Runs cublas_mm, qk_projection_and_rope, FP8 quantize, and weight
        scaling.  Does NOT touch the k cache or any batch-specific metadata,
        so this can safely run inside a captured CUDA graph partition.

        Returns (q_fp_bytes, k_fp_bytes, k_scale, weights, q_scale). The last
        tensor is only consumed by the FP4 kernel dispatch; the FP8 path
        ignores it. It is returned unconditionally so the two-op CUDA graph
        split in MLA.forward_dsa_proj sees a stable signature.
        """
        # FSSS / index_topk_freq reuse: on a TopK-reuse ("S") layer the
        # downstream sparse_attn_indexer returns the cached TopK from the
        # owning ("F") layer and never reads these projections, so computing
        # them is dead work (~32us/layer of wq_b + fused wk/wp GEMMs and the
        # fused_cat_fp4 quantize at decode batch). skip_topk is a static
        # per-layer property, so this branch is constant for a given layer
        # object and stays straight-line under CUDA graph capture. Return
        # uninitialized buffers matching the _mla_dsa_proj_fake contract
        # (shape + dtype) so the graph-captured signature is unchanged; the
        # values are never consumed on the reuse path.
        if self.skip_topk:
            num_tokens = hidden_states.shape[0]
            if self.use_fp4:
                q_fp8 = hidden_states.new_empty(
                    (num_tokens, self.n_heads, self.head_dim // 2),
                    dtype=torch.int8)
                k_fp8 = hidden_states.new_empty(
                    (num_tokens, self.head_dim // 2), dtype=torch.int8)
                k_scale = hidden_states.new_empty((num_tokens, 1),
                                                  dtype=torch.int32)
                q_scale = hidden_states.new_empty(
                    (num_tokens, self.n_heads, 1), dtype=torch.int32)
            else:
                q_fp8 = hidden_states.new_empty(
                    (num_tokens, self.n_heads, self.head_dim),
                    dtype=torch.float8_e4m3fn)
                k_fp8 = hidden_states.new_empty((num_tokens, self.head_dim),
                                                dtype=torch.float8_e4m3fn)
                k_scale = hidden_states.new_empty((num_tokens, 1),
                                                  dtype=torch.float32)
                q_scale = hidden_states.new_empty(
                    (num_tokens, self.n_heads, 1), dtype=torch.float32)
            weights = hidden_states.new_empty((num_tokens, self.n_heads),
                                              dtype=torch.float32)
            return q_fp8, k_fp8, k_scale, weights, q_scale

        if (self._fused_wk_wp_nvfp4 is not None
                and isinstance(hidden_states, torch.Tensor)):
            indexer_k, weights = self._fused_wk_wp_nvfp4(hidden_states)
        elif self._fused_wk_wp_weight is not None:
            hidden_float = _to_float(hidden_states)
            with _tf32_matmul_enabled():
                # F.linear computes input @ weight.T internally; no explicit .t() needed.
                # _fused_wk_wp_weight is [head_dim + n_heads, hidden_size] (nn.Linear convention).
                # Goes through PyTorch's cuBLAS handle which respects allow_tf32 and
                # dispatches CUBLAS_COMPUTE_32F_FAST_TF32, unlike torch.ops.trtllm.cublas_mm
                # which uses its own handle and always falls back to CUDA-core SGEMM.
                fused_out = F.linear(hidden_float, self._fused_wk_wp_weight)
            indexer_k, weights = fused_out.split([self.head_dim, self.n_heads],
                                                 dim=-1)
        else:
            indexer_k = self.wk(hidden_states)
            weights = self.weights_proj(hidden_states)
        # Cast indexer_k back to model dtype for downstream ops (k_norm, RoPE, FP8 quantize)
        indexer_k = indexer_k.to(hidden_states.dtype)

        if self._rope_cat_fuse_ok:
            _ensure_fused_rope_cat_fp4_op()
            # Fused RoPE+cat+FP4-quant: project (no RoPE), then fold RoPE into
            # the quantize kernel for Q and K in parallel. Bit-identical to
            # rotary_emb -> _prep_q_or_k, but removes the RoPE kernel launch
            # and the BF16 q_pe/k_pe round-trip (~3-4 us/F-layer, graphed).
            q_pe, q_nope, k_pe, k_nope = self._qk_projection_no_rope(
                qr, indexer_k)
            if self._rope_cat_cos_sin is None:
                self._rope_cat_cos_sin = self.rotary_emb.rotary_cos_sin.view(
                    self.rotary_emb.max_positions, -1).to(torch.float32)
            cos_sin = self._rope_cat_cos_sin
            pos = position_ids.view(-1).to(torch.int32)
            num_tokens = pos.shape[0]
            pos_q = pos.repeat_interleave(self.n_heads)
            q_pe_2d = q_pe.reshape(num_tokens * self.n_heads, self.rope_dim)
            q_nope_2d = q_nope.reshape(num_tokens * self.n_heads,
                                       self.head_dim - self.rope_dim)
            q, k = maybe_execute_in_parallel(
                lambda: torch.ops.trtllm.fused_rope_cat_fp4(
                    q_pe_2d, q_nope_2d, cos_sin, pos_q),
                lambda: torch.ops.trtllm.fused_rope_cat_fp4(
                    k_pe, k_nope, cos_sin, pos),
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )
            q_fp8, q_scale = q
            k_fp8, k_scale = k
        else:
            q_pe, q_nope, k_pe, k_nope = self._qk_projection_and_rope(
                qr, indexer_k, position_ids)
            q, k = maybe_execute_in_parallel(
                lambda: self._prep_q_or_k(q_pe, q_nope),
                lambda: self._prep_q_or_k(k_pe, k_nope),
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )
            q_fp8, q_scale = q
            k_fp8, k_scale = k
        if self.use_fp4:
            # FP4 packs two codes per byte, so the trailing dim is head_dim // 2.
            # fused_cat_fp4 flattens the leading dims to M=N*n_heads; restore
            # the (N, n_heads, ...) shape so downstream slicing in
            # sparse_attn_indexer (which indexes by token) lines up with
            # q_fp8. The DeepGEMM FP4 kernel applies the per-block q_scale
            # internally, so weights carry only softmax_scale * n_heads^-0.5.
            q_fp8 = q_fp8.view(-1, self.n_heads, self.head_dim // 2)
            q_scale = q_scale.view(-1, self.n_heads, 1)
            weights = weights * self.weight_scale_factor
        else:
            q_fp8 = q_fp8.view(-1, self.n_heads, self.head_dim)
            q_scale = q_scale.view(-1, self.n_heads, 1)
            weights = self._weight_scale(weights, q_scale)

        return q_fp8, k_fp8, k_scale, weights, q_scale

    @torch.inference_mode()
    def forward(self, qr: torch.Tensor, hidden_states: torch.Tensor,
                metadata: DSAtrtllmAttentionMetadata,
                position_ids: torch.Tensor):
        # LayerSplit indexer-K READ-SET broadcast: the owner CP rank for
        # layer L publishes the cache blocks the indexer KERNEL READS this
        # step — the FULL per-request KV prefix [0, kv_len) — not just the
        # write set (the new-token blocks). z.ai "Scaling Pain" §4: the
        # indexer scores every query token causally against the whole
        # prefix, so under chunked prefill / prefix reuse a write-set-only
        # broadcast (the old M5e) leaves the non-owner's indexer-K scratch
        # STALE for the cached prefix on chunk >= 2, yielding wrong logits
        # and wrong TopK. The read set is ~1/8 of total KV (z.ai's exposed
        # broadcast cost) and is REQUIRED for correctness.
        #
        # The dense-KV + NVFP4-scale broadcasts are NOT here: the dense
        # read set is the TopK-selected blocks, which are only known AFTER
        # the indexer runs. They are issued in
        # DSATrtllmAttention.sparse_attn_predict, after topk_indices_global
        # is computed and before the sparse-MLA dense read consumes it.
        #
        # All ranks see the same metadata, so they compute identical read
        # sets — required for the NCCL broadcast to agree on buffer shape.
        # All paths are no-ops on the LayerSplit-off / cp_size=1 / no
        # process-group / no-CUDA branches so this is safe to drop in
        # unconditionally — and it ONLY engages for DSA models because this
        # file is the DSA attention backend (LayerSplit's only home;
        # non-DSA models never construct a DSACacheManager).
        kv_cache_manager = getattr(metadata, "kv_cache_manager", None)
        layersplit_state = getattr(kv_cache_manager, "layersplit_state",
                                   None) if kv_cache_manager is not None else None
        if layersplit_state is not None and layersplit_state.enabled:
            # L2 hoist: the read set is layer-invariant within a step, so the
            # first indexer layer computes it and the rest reuse the memo.
            read_block_ids = _layersplit_read_block_ids_step(metadata)

            indexer_slot = kv_cache_manager.get_indexer_k_cache_buffers(
                self.layer_idx)
            layersplit_state.maybe_broadcast_active_blocks(
                layer_idx=self.layer_idx,
                cache_slot=indexer_slot,
                active_block_ids=read_block_ids,
                cp_group=layersplit_state.cp_group,
            )

            # L1 (z.ai Fig 4(b) overlap): on pure-context steps the dense
            # top-k union equals the read set, so the dense KV (+ NVFP4
            # scale) broadcast does not need to wait for the TopK — issue
            # it NOW on the comm stream, hidden behind the indexer scoring
            # below. sparse_attn_predict consumes the comm-stream event
            # just before the dense read (read-before-ready) and skips the
            # legacy union broadcast and its per-layer masked_select +
            # unique host syncs. Generation steps keep the legacy path:
            # at decode the top-k union is ~32 blocks/seq, far smaller
            # than the read set, so the early superset broadcast would
            # multiply wire bytes there.
            if (metadata.num_generations == 0
                    and _layersplit_prefill_overlap_enabled()):
                try:
                    dense_kv_slot = kv_cache_manager.get_buffers(
                        self.layer_idx)
                except (AttributeError, IndexError, KeyError):
                    dense_kv_slot = None
                dense_scale_slot = None
                get_scale_slot = getattr(kv_cache_manager,
                                         "get_dense_scale_slot", None)
                if get_scale_slot is not None:
                    try:
                        dense_scale_slot = get_scale_slot(self.layer_idx)
                    except (AttributeError, IndexError, KeyError,
                            RuntimeError):
                        dense_scale_slot = None
                if dense_kv_slot is not None:
                    flat_dense = dense_kv_slot.view(dense_kv_slot.shape[0],
                                                    -1)
                    layersplit_state.overlap_broadcast_readset(
                        layer_idx=self.layer_idx,
                        cache_slots=(flat_dense, dense_scale_slot),
                        read_block_ids=read_block_ids,
                        cp_group=layersplit_state.cp_group,
                    )

        q_fp8, k_fp8, k_scale, weights, q_scale = self.pre_indexer_proj(
            qr, hidden_states, position_ids)

        # Return topk indices buffer for sparse attention [num_tokens, index_topk]
        return self.sparse_attn_indexer(metadata,
                                        hidden_states,
                                        q_fp8,
                                        k_fp8,
                                        k_scale,
                                        weights,
                                        q_scale=q_scale)


class DSATrtllmAttention(TrtllmAttention):
    """TRT-LLM attention layer with DSA sparse indexer for MLA models."""

    Metadata = DSAtrtllmAttentionMetadata

    def __init__(
            self,
            layer_idx: int,
            num_heads: int,
            head_dim: int,
            num_kv_heads: Optional[int] = None,
            quant_config: Optional[QuantConfig] = None,
            q_scaling: Optional[float] = None,
            pos_embd_params: Optional[PositionalEmbeddingParams] = None,
            mla_params: Optional[MLAParams] = None,
            skip_create_weights_in_init: bool = False,
            attention_chunk_size: Optional[int] = None,
            sparse_attention_config: Optional["SparseAttentionConfig"] = None,
            dtype: Optional[torch.dtype] = None,
            aux_stream: Optional[torch.cuda.Stream] = None,
            **kwargs):
        """Initialize DSA attention with an Indexer sub-module for sparse TopK selection."""
        if sparse_attention_config is None:
            raise ValueError(
                "sparse_attention_config is required for DSATrtllmAttention and cannot be None"
            )
        TrtllmAttention.__init__(
            self,
            layer_idx,
            num_heads,
            head_dim,
            sparse_attention_config=sparse_attention_config,
            num_kv_heads=num_kv_heads,
            quant_config=quant_config,
            q_scaling=q_scaling,
            pos_embd_params=pos_embd_params,
            mla_params=mla_params,
            skip_create_weights_in_init=skip_create_weights_in_init,
            attention_chunk_size=attention_chunk_size,
            **kwargs)

        self.indexer = Indexer(quant_config, pos_embd_params, mla_params,
                               skip_create_weights_in_init,
                               sparse_attention_config, dtype, layer_idx,
                               aux_stream)
        # Per-block (device int64) epoch last reconstructed into this layer's
        # fp16 main pool; lazily sized to the KVarN pool on first amortized
        # restore. This state belongs to the attention layer, because decode
        # restore runs before the Indexer path.
        self._kvarn_restored_gen = None

    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DSAtrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Transform local TopK indices to global paged KV cache indices."""
        # Transform the local topk indices to global topk indices in paged kv cache
        is_generation = (forward_args.attention_input_type ==
                         AttentionInputType.generation_only)
        local_layer_idx = self.get_local_layer_idx(metadata)
        hisparse_mapping = None
        hisparse_coordinator = getattr(metadata, "hisparse_coordinator", None)
        if hisparse_coordinator is not None:
            hisparse_mapping = hisparse_coordinator.map_topk_to_hot_pool(
                topk_indices=forward_args.topk_indices,
                metadata=metadata,
                layer_idx=local_layer_idx,
                skip_topk=self.indexer.skip_topk,
                is_generation=is_generation,
            )
        if hisparse_mapping is not None:
            topk_indices_global = hisparse_mapping.topk_indices_global
        else:
            topk_indices_global, _ = transform_local_topk_reuse_or_compute(
                forward_args.topk_indices, metadata, local_layer_idx,
                self.indexer.skip_topk, is_generation)

        # LayerSplit dense-KV + NVFP4-scale READ-SET broadcast. The indexer-K
        # broadcast (the full prefix) ran in Indexer.forward; the dense read
        # set is the TopK-SELECTED blocks, knowable only now. The owner CP rank
        # for this layer publishes exactly those blocks into the non-owner's
        # dense scratch (and its sibling NVFP4 block-scale scratch) BEFORE the
        # sparse-MLA read consumes topk_indices_global in the same forward
        # (TrtllmAttention.forward: sparse_attn_predict -> _run, the dense
        # read). Map the global-token TopK indices to their unique global block
        # ids (g // stride_factor) — the index space the broadcast scatters
        # into. No-ops on LayerSplit-off / cp_size<=1 / no-group / empty set.
        kv_cache_manager = getattr(metadata, "kv_cache_manager", None)
        layersplit_state = getattr(kv_cache_manager, "layersplit_state",
                                   None) if kv_cache_manager is not None else None
        if (layersplit_state is not None and layersplit_state.enabled
                and layersplit_state.consume_overlap_event(self.layer_idx)):
            # L1 overlap consumed: Indexer.forward already broadcast this
            # layer's dense KV (+ scale) READ set on the comm stream — a
            # byte-equivalent superset of the top-k union below — and the
            # event wait just ordered it before the dense sparse-MLA read.
            # Skip the legacy union broadcast (and its masked_select +
            # unique host syncs).
            pass
        elif layersplit_state is not None and layersplit_state.enabled:
            stride_factor = getattr(metadata, "_cached_stride_factor", None)
            dense_block_ids = _layersplit_topk_global_block_ids(
                topk_indices_global, stride_factor)

            # M5f dense KV broadcast. get_buffers may raise for layers outside
            # the current manager (PP-partitioned drafts etc.) — skip the dense
            # broadcast silently in that case.
            try:
                dense_kv_slot = kv_cache_manager.get_buffers(self.layer_idx)
            except (AttributeError, IndexError, KeyError):
                dense_kv_slot = None
            if dense_kv_slot is not None:
                flat_dense = dense_kv_slot.view(dense_kv_slot.shape[0], -1)
                layersplit_state.maybe_broadcast_active_blocks(
                    layer_idx=self.layer_idx,
                    cache_slot=flat_dense,
                    active_block_ids=dense_block_ids,
                    cp_group=layersplit_state.cp_group,
                )

            # M5f-scale: under NVFP4 the dense KV has a sibling block-scale pool
            # whose non-owned-layer scratch the dense-MLA kernel also reads (via
            # the augmented pool pointers' scale column). Broadcast the same
            # TopK blocks' scales; no-op when there is no dense scale scratch
            # (non-NVFP4 / replicated / owner-only). Owners share the block-
            # offset table for data and scale, so the block ids index both.
            get_scale_slot = getattr(kv_cache_manager, "get_dense_scale_slot",
                                     None)
            if get_scale_slot is not None:
                try:
                    dense_scale_slot = get_scale_slot(self.layer_idx)
                except (AttributeError, IndexError, KeyError, RuntimeError):
                    dense_scale_slot = None
                if dense_scale_slot is not None:
                    # Pass the slot unflattened: the broadcast indexes dim 0
                    # only, and reshape on a non-contiguous slice would copy —
                    # silently dropping the index_copy_ write-through into the
                    # real scale pool.
                    layersplit_state.maybe_broadcast_active_blocks(
                        layer_idx=self.layer_idx,
                        cache_slot=dense_scale_slot,
                        active_block_ids=dense_block_ids,
                        cp_group=layersplit_state.cp_group,
                    )

        # TODO: Use sparse_attn_indexer to predict the indices for DSA attention
        # return self.indexer(q, k, metadata, hidden_states, qr, position_ids)
        return topk_indices_global, None

    def sparse_kv_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DSAtrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """No-op KV prediction; DSA uses indexer-based selection instead."""
        return None, None

    # -- KVarN dense-MLA-latent store/restore (Stage-a software cache) ------

    def _kvarn_mgr(self, metadata):
        mgr = getattr(metadata, "kv_cache_manager", None)
        if mgr is None or not getattr(mgr, "kvarn_enabled", False):
            return None
        return mgr

    def _kvarn_latent_block_view(self, mgr, metadata, block_id):
        """[tokens_per_block, kv_lora_rank + qk_rope_head_dim] fp16 view of one
        paged latent block in the main pool, split into (ckv, k_pe)."""
        buf = mgr.get_buffers(self.layer_idx, kv_layout="NHD")  # [P,1,tpb,1,D]
        blk = buf[int(block_id), 0, :, 0, :]                    # [tpb, D]
        ckv = blk[:, :mgr.kvarn_cfg.kv_lora_rank]
        k_pe = blk[:, mgr.kvarn_cfg.kv_lora_rank:]
        return blk, ckv, k_pe

    def _kvarn_seq_range(self, metadata, is_generation):
        """(start, stop) global sequence indices for this phase. Context seqs
        occupy rows 0..num_contexts of block_table / kv_lens_runtime;
        generation seqs occupy num_contexts..num_seqs."""
        nc = int(metadata.num_contexts)
        ns = nc + int(metadata.num_generations)
        return (nc, ns) if is_generation else (0, nc)

    def _kvarn_block_table_host(self, metadata):
        """metadata.block_table is the DECODED pool block index per (seq, slot)
        (padding = -1; see _get_pool_block_indices). Pull to host once."""
        bt = metadata.block_table
        return bt.to("cpu") if bt.is_cuda else bt

    def _kvarn_step_cand_host(self, metadata, tpb: int, lo: int,
                              hi: int) -> Optional["np.ndarray"]:
        """Step-invariant host candidate set for the eager decode restore.

        Returns the sorted-unique block ids (np.int64) covered by the
        generation rows' FULL blocks, derived purely from host state: the
        host block table stashed by prepare() plus the host kv lens. Returns
        None when that state is unavailable or inconsistent (caller falls
        back to the device scan). The set is identical for every layer
        within a step, so it is memoized on the metadata keyed by
        (request ids, per-row full-block counts) -- the same invariant the
        pre-replay step gate relies on: a row's already-full block ids
        cannot change while its request id and full-block count are stable.
        """
        hbt = getattr(metadata, "kvarn_host_block_table", None)
        if (hbt is None or not torch.is_tensor(hbt) or hbt.is_cuda
                or hbt.dim() != 2):
            return None
        req_ids = getattr(metadata, "request_ids", None)
        kv_lens = metadata.kv_lens_runtime
        if (req_ids is None or len(req_ids) < hi or hbt.shape[0] < hi
                or kv_lens is None or len(kv_lens) < hi):
            return None
        n_full = tuple(int(kv_lens[i]) // tpb for i in range(lo, hi))
        key = (tuple(req_ids[lo:hi]), n_full, lo, hi, tpb)
        memo = getattr(metadata, "_blaise_kvarn_step_cand", None)
        if memo is not None and memo[0] == key:
            return memo[1]
        table = hbt.numpy()
        width = table.shape[1]
        segs = []
        for r, nf in enumerate(n_full):
            if nf <= 0:
                continue
            if nf > width:
                return None
            seg = table[lo + r, :nf]
            if (seg < 0).any():
                # Padding inside the full range: the table and kv lens
                # disagree; let the device scan decide rather than guess.
                return None
            segs.append(seg)
        if segs:
            cand = np.unique(np.concatenate(segs).astype(np.int64,
                                                         copy=False))
        else:
            cand = np.empty((0, ), dtype=np.int64)
        metadata._blaise_kvarn_step_cand = (key, cand)
        return cand

    def kvarn_commit_full_blocks(self, metadata, is_generation):
        """KVarN-store every newly-FULL latent block (skip the sink + the
        in-progress tail block, which stay fp16). Idempotent via pool.valid."""
        mgr = self._kvarn_mgr(metadata)
        if mgr is None:
            return
        tpb = mgr.tokens_per_block
        sink_blocks = mgr.kvarn_cfg.sink_tokens // tpb
        pool = mgr.get_kvarn_latent_pool(self.layer_idx)
        if pool is None:
            return
        bt = self._kvarn_block_table_host(metadata)
        kv_lens = metadata.kv_lens_runtime  # host, per (all) seqs
        lo, hi = self._kvarn_seq_range(metadata, is_generation)
        for i in range(lo, hi):
            klen = int(kv_lens[i])
            n_full = klen // tpb            # number of FULL blocks for this seq
            for b in range(sink_blocks, n_full):
                block_id = int(bt[i, b])
                if block_id < 0 or bool(pool.valid[block_id]):
                    continue
                _, ckv, k_pe = self._kvarn_latent_block_view(mgr, metadata,
                                                             block_id)
                mgr.kvarn_store_block(self.layer_idx, block_id,
                                      ckv.to(torch.float16),
                                      k_pe.to(torch.float16))

    def kvarn_restore_for_decode(self, metadata):
        """Reconstruct committed blocks into the main-pool fp16 slot so the C++
        decode kernel reads correct latent values, using the BATCHED restore
        primitive (one pool.load_blocks() for the step; ~1.2-2.4 us/block).

        AMORTIZATION (mla_latent_kv_amortize, default OFF):
        committed full blocks are immutable -- their packed KVarN bytes never
        change until the block-id is recycled and re-committed (pool.commit_gen
        bumps then). During steady decode the only main-pool write is to the
        in-progress TAIL block (not yet committed, stays fp16); committed-block
        fp16 slots are never overwritten. So once a block is reconstructed into
        its fp16 slot it stays correct, and we only re-dequant blocks whose
        restored epoch lags commit_gen -- the per-step CHURN (one fresh block
        per request per 64 decode steps), not the full B*32 working set.

        Microbench (kvarn_inkernel, B200): batch=32 un-amortized full restore
        481 us = 141% of the 341 us/layer/tok budget; amortized fill-step (all
        32 reqs commit a fresh block the SAME step) 71.7 us = 21% budget (6.7x);
        steady-state 1.12 us = 0.33% budget. cos_ckv=cos_kpe=1.000000.

        Set mla_latent_kv_amortize=False (or leave the AMORTIZE env off) to
        fall back to the always-correct full per-step restore."""
        mgr = self._kvarn_mgr(metadata)
        if mgr is None:
            return
        if torch.cuda.is_current_stream_capturing():
            return
        tpb = mgr.tokens_per_block
        pool = mgr.get_kvarn_latent_pool(self.layer_idx)
        if pool is None:
            return
        kv_lens = metadata.kv_lens_runtime
        lo, hi = self._kvarn_seq_range(metadata, True)
        if hi <= lo:
            return
        amortize = bool(getattr(mgr, "kvarn_amortize_restore", False))

        # Host-mirror fast path: derive the stale set purely from host state
        # (the host block table stashed by prepare() + the pool's numpy
        # mirrors), the same machinery the pre-replay delta walk trusts. The
        # device scan below costs 3 boolean-index nonzeros (cub DeviceSelect
        # sweep + compact-init + count-reduce + write_indices) plus a unique
        # per layer per step -- ~20 us GPU and 3 forced d2h syncs per layer,
        # ~1.24 ms GPU / 183 syncs per eager step at 61 layers (the
        # "indexer_select_cub" 3.0% line in the EAGER_C16 decode profile) --
        # in the common case only to discover the stale set is empty. The
        # candidate set is layer-invariant within a step, so it is memoized
        # on the metadata; the per-layer residual is one numpy filter over
        # ~B*n_full ids and zero launches when nothing is stale.
        cand_host = self._kvarn_step_cand_host(metadata, tpb, lo, hi)
        if cand_host is not None and hasattr(pool, "stale_committed_host"):
            if cand_host.size == 0:
                return
            if amortize:
                todo = pool.stale_committed_host(cand_host)
            else:
                # Un-amortized semantics: restore every committed candidate
                # each step (no epoch filter), matching the device scan.
                todo = cand_host[pool.valid_host[cand_host]].tolist()
            if not todo:
                return
            self.kvarn_restore_block_ids(metadata, todo)
            pool.mark_restored_host(todo)
            return

        dev = pool.valid.device

        # Device-scan fallback (host block table unavailable): gather the
        # committed block ids of the generation rows straight off the (device)
        # block table, mask out padding / uncommitted / (under amortize) blocks
        # whose fp16 slot already holds their current content. The old Python
        # double-loop over B*32 entries dominated decode at batch>=8
        # (host-scan-only 2779 us/step @ b32, vs ~16 us for the d2h); this
        # keeps the whole set-diff on-device.
        bt = metadata.block_table  # [num_all_seqs, max_blocks], device, -1 pad
        bt_gen = bt[lo:hi].to(dev, non_blocking=True)            # [B, max_blocks]
        kv_t = torch.as_tensor(kv_lens[lo:hi], device=dev, dtype=torch.long)
        n_full = torch.div(kv_t, tpb, rounding_mode="floor")     # [B]
        max_blocks = bt_gen.shape[1]
        col = torch.arange(max_blocks, device=dev)
        in_ctx = col.unsqueeze(0) < n_full.unsqueeze(1)          # [B, max_blocks]
        cand = bt_gen[in_ctx]                                    # 1-D candidate ids
        cand = cand[cand >= 0].to(torch.long)
        if cand.numel() == 0:
            return
        keep = pool.valid[cand]
        if amortize:
            rg = self._kvarn_restored_gen
            if (not torch.is_tensor(rg)) or rg.numel() != pool.num_blocks:
                rg = torch.full((pool.num_blocks,), -1, dtype=torch.int64,
                                device=dev)
                self._kvarn_restored_gen = rg
            keep = keep & (rg[cand] != pool.commit_gen[cand])
        cand = cand[keep]
        if cand.numel() == 0:
            return
        block_ids = torch.unique(cand)                           # sorted, device
        ckv_d, kpe_d = pool.load_blocks(block_ids)  # [N,G,Dckv] / [N,G,Dpe]
        buf = mgr.get_buffers(self.layer_idx, kv_layout="NHD")  # [P,1,tpb,1,D]
        Dckv = mgr.kvarn_cfg.kv_lora_rank
        ids = block_ids.to(buf.device)
        # scatter the reconstructed latent back: blk[:, :Dckv]=ckv, [:, Dckv:]=k_pe
        buf[ids, 0, :, 0, :Dckv] = ckv_d.to(buf.dtype)
        buf[ids, 0, :, 0, Dckv:] = kpe_d.to(buf.dtype)
        if amortize:
            self._kvarn_restored_gen[block_ids] = pool.commit_gen[block_ids]
        # Mirror the restore epoch on host for the pre-replay delta walk.
        # This path already pays masked-select syncs, so the small ids d2h
        # is noise here; without it the delta walk would re-restore these
        # blocks on their owner's next boundary crossing -- idempotent for
        # live blocks but wrong for a recycled id whose stale committed
        # record the amortized path would correctly leave alone.
        pool.mark_restored_host(block_ids.cpu().numpy())

    def kvarn_restore_block_ids(self, metadata, block_ids) -> None:
        """Reconstruct a host-precomputed set of committed blocks into this
        layer's fp16 main-pool slots.

        Sync-free counterpart of ``kvarn_restore_for_decode`` for the
        pre-replay delta walk (``model_engine._restore_kvarn_before_cuda_
        graph_replay``): the caller derived the stale-block set from host
        request metadata plus the pool's host mirrors, so this body is one
        batched dequant and two scatters -- no masked-select / ``unique``
        readbacks, no device->host syncs. Every id must be committed
        (``pool.valid``) and stale; restoring an already-restored committed
        block is an idempotent rewrite of identical bytes.
        """
        mgr = self._kvarn_mgr(metadata)
        if mgr is None:
            return
        pool = mgr.get_kvarn_latent_pool(self.layer_idx)
        if pool is None:
            return
        ids = torch.as_tensor(block_ids, dtype=torch.long, device=pool.device)
        if ids.numel() == 0:
            return
        ckv_d, kpe_d = pool.load_blocks(ids)  # [N,G,Dckv] / [N,G,Dpe]
        buf = mgr.get_buffers(self.layer_idx, kv_layout="NHD")  # [P,1,tpb,1,D]
        Dckv = mgr.kvarn_cfg.kv_lora_rank
        ids = ids.to(buf.device)
        buf[ids, 0, :, 0, :Dckv] = ckv_d.to(buf.dtype)
        buf[ids, 0, :, 0, Dckv:] = kpe_d.to(buf.dtype)
        # Keep the eager amortized path's epoch view coherent so a later
        # eager step does not redo this work (device-to-device, no sync).
        rg = self._kvarn_restored_gen
        if torch.is_tensor(rg) and rg.numel() == pool.num_blocks:
            rg[ids] = pool.commit_gen[ids]

    def mla_rope_generation(
        self,
        fused_q: torch.Tensor,
        q_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        metadata: "DSAtrtllmAttentionMetadata",
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        fmha_scheduler_counter: torch.Tensor,
        mla_bmm1_scale: torch.Tensor,
        mla_bmm2_scale: torch.Tensor,
        quant_q_buffer: torch.Tensor,
        out_scale=None,
    ) -> None:
        """DSA decode: reconstruct KVarN-committed latent blocks into the fp16
        main pool (no-op when KVarN is off), then run the standard MLA decode."""
        self.kvarn_restore_for_decode(metadata)
        return super().mla_rope_generation(
            fused_q, q_pe, latent_cache, metadata, cu_q_seqlens, cu_kv_seqlens,
            fmha_scheduler_counter, mla_bmm1_scale, mla_bmm2_scale,
            quant_q_buffer, out_scale)

    def mla_rope_append_paged_kv_assign_q(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        metadata: DSAtrtllmAttentionMetadata,
        is_generation: bool = False,
        **kwargs,
    ) -> None:
        """Apply RoPE, append latent cache to paged KV, and assign query for MLA."""
        if is_generation:
            cached_token_indptr = metadata.gen_cached_token_indptr
            kv_indptr = metadata.gen_kv_indptr
            num_seqs = metadata.num_generations
            max_seq_len = metadata.max_gen_seq_len
            block_offsets = metadata.kv_cache_block_offsets[:, metadata.
                                                            num_contexts:]
        else:
            cached_token_indptr = metadata.ctx_cached_token_indptr
            kv_indptr = metadata.ctx_kv_indptr
            num_seqs = metadata.num_contexts
            max_seq_len = metadata.max_ctx_seq_len
            block_offsets = metadata.kv_cache_block_offsets
        assert self.is_mla_enable and self.mla_params is not None
        assert metadata.kv_cache_manager is not None

        beam_width = 1

        torch.ops.trtllm.mla_rope_append_paged_kv_assign_q(
            q,
            latent_cache,
            num_seqs,
            cached_token_indptr,
            kv_indptr,
            max_seq_len,
            self.rotary_cos_sin,
            self.num_heads,
            self.mla_params.qk_nope_head_dim,
            self.mla_params.qk_rope_head_dim,
            self.mla_params.kv_lora_rank,
            block_offsets,
            # LayerSplit owner-local: use the augmented pool pointers/mapping
            # (the metadata property appends the shared dense scratch pool +
            # one mapping row per non-owned layer) so the local_layer_idx
            # returned by get_local_layer_idx for a non-owned layer -- which
            # is that augmented mapping row -- indexes a valid pool. Reading
            # the raw manager attribute here would feed an augmented row index
            # into an UNaugmented mapping, an out-of-bounds / wrong-pool read.
            # Identical to the unaugmented tensors in the replicated posture.
            metadata.host_kv_cache_pool_pointers,
            metadata.host_kv_cache_pool_mapping,
            None,  # kv_scale_orig_quant
            self.get_local_layer_idx(metadata),
            metadata.kv_cache_manager.tokens_per_block,
            metadata.kv_cache_manager.max_seq_len,
            beam_width,
            self.quant_mode,
        )

        # KVarN: compress any blocks that just filled (no-op when off).
        self.kvarn_commit_full_blocks(metadata, is_generation)


class DSACacheManager(KVCacheManager):
    """KV cache manager for DSA with additional indexer K-cache pools."""

    def __init__(
        self,
        kv_cache_config: KvCacheConfig,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        num_kv_heads: Union[int, List[Optional[int]]],
        head_dim: int,
        tokens_per_block: int,
        # Note that max_seq_len is not necessarily equal to kv_cache_config.num_tokens.
        # It's derived from the model's BuildConfig for consistency with the C++ backend.
        max_seq_len: int,
        max_batch_size: int,
        mapping: Mapping,
        dtype: DataType = DataType.HALF,
        spec_config: Optional["DecodingBaseConfig"] = None,
        layer_mask: Optional[List[bool]] = None,
        max_num_tokens: int = 8192,
        model_config: Optional[ModelConfig] = None,
        max_beam_width: int = 1,
        sparse_attn_config: "SparseAttentionConfig",
        layersplit_model_num_layers: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Initialize cache manager with indexer K-cache pool per layer."""
        self.quant_block_size = 128
        self.index_head_dim = sparse_attn_config.index_head_dim

        # LayerSplit runtime state: owner_map + side-stream + transfer-backend
        # selection + (M5b) the CP process group used for owner -> peers
        # broadcasts. The state object is inert when layersplit_enabled is
        # False, so the regular DSA path is unchanged in that case. The
        # ownership table sits on a separate helper so it can be unit-tested
        # without instantiating the C++ WindowBlockManager parent.
        cp_size = getattr(mapping, "cp_size", 1) if mapping is not None else 1
        cp_rank = getattr(mapping, "cp_rank", 0) if mapping is not None else 0
        # ``num_layers`` has already been trimmed to sum(layer_mask) when the
        # LayerSplit owner-local mask is active, but the runtime queries
        # ownership with global layer ids (for example layer_idx=0 in
        # get_buffers()). Build the owner table over the global layer domain so
        # state.is_owner(global_layer_idx) remains valid on every CP rank.
        layersplit_owner_num_layers = (
            layersplit_model_num_layers
            if layersplit_model_num_layers is not None else
            (len(layer_mask) if layer_mask is not None and bool(
                getattr(sparse_attn_config, "layersplit_enabled", False)) else
             num_layers))
        self.layersplit_model_num_layers = layersplit_owner_num_layers
        self.layersplit_local_pool_layers = num_layers
        self.layersplit_cache_transfer_model_layers = layersplit_owner_num_layers
        self.layersplit_cache_transfer_local_pool_layers = num_layers
        self.layersplit_cache_transfer_layer_mask_len = len(layer_mask) if layer_mask is not None else None
        self.layersplit_state = LayerSplitRuntimeState.from_sparse_config(
            sparse_attn_config=sparse_attn_config,
            num_layers=layersplit_owner_num_layers,
            cp_size=cp_size,
            cp_rank=cp_rank,
        )
        if self.layersplit_state.enabled and mapping is not None:
            # Resolve (or create) the CP process group the per-layer
            # owner->peer broadcast collectives ride on. ensure_cp_process_group
            # prefers the canonical DeviceMesh cp_group_pg (Ray / mpi_disabled
            # runtimes) and, under MPI -- where cp_group_pg raises
            # NotImplementedError and torch.distributed is otherwise never
            # initialized (the TP/EP collectives use the C++ custom-allreduce
            # path) -- bootstraps a NCCL world and collectively carves the CP
            # subgroup from mapping.cp_groups. Runs once at model load so the
            # group is graph-stable. Failures are non-fatal: it returns
            # (None, None) and the broadcast gracefully collapses to a no-op.
            cp_group, cp_group_ranks = ensure_cp_process_group(mapping)
            if cp_group is not None:
                self.layersplit_state.bind_cp_group(cp_group, cp_group_ranks)
        if self.layersplit_state.enabled:
            logger.info(
                "LayerSplit enabled: %d layers across %d CP ranks via "
                "policy=%s, transfer_backend=%s, cp_group=%s. "
                "Replicated-materialization (M3) + owner-local alloc (M4) "
                "+ broadcast scaffold (M5) all installed; KV payload "
                "plumbing is M5c.",
                layersplit_owner_num_layers,
                cp_size,
                self.layersplit_state.ownership.policy,
                self.layersplit_state.transfer_backend,
                "bound" if self.layersplit_state.cp_group is not None else
                "unbound (broadcast collapses to noop)",
            )

        # FP4 mode packs the indexer K cache as head_dim/2 data bytes + 4
        # scale bytes (vs. head_dim + 4 for FP8). The C++ WindowBlockManager
        # allocates the pool with this smaller stride when the flag is set.
        self.use_fp4 = sparse_attn_config.indexer_k_dtype == "fp4"

        # Accessors can be called during construction (for example KVarN uses
        # get_buffers to choose the dense side-pool device), so LayerSplit
        # scratch state must exist before any base/KVarN setup path can read it.
        self._layersplit_indexer_k_scratch = None
        self._layersplit_dense_kv_scratch = None
        self._layersplit_hisa_pagerep_scratch = None
        self._layersplit_hisa_pagecount_scratch = None

        super().__init__(
            kv_cache_config,
            kv_cache_type,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=dtype,
            spec_config=spec_config,
            layer_mask=layer_mask,
            max_num_tokens=max_num_tokens,
            model_config=model_config,
            max_beam_width=max_beam_width,
            enable_indexer_k_cache=True,
            indexer_k_cache_quant_block_size=128,
            indexer_k_cache_index_head_dim=self.index_head_dim,
            indexer_k_cache_use_fp4=self.use_fp4,
            **kwargs,
        )

        # Owner-local LayerSplit trims the actual C++ KV/indexer/KVarN pools
        # to this CP rank's owned layers. Disaggregated transfer metadata is a
        # different contract: C++ target-rank selection and MLA split/concat use
        # mNbKvHeadsPerLayer.size() as the global attention-layer domain and
        # mAttentionLayerNumPerPP as the local shard size. Publish an explicit
        # transfer-only global vector so the transceiver does not serialize the
        # trimmed local pool shape and make decode reject TP2xCP2 -> TP4xCP1
        # handoff as a layer-count mismatch.
        self.layersplit_transfer_num_kv_heads_per_layer = None
        if (self.layersplit_state.enabled
                and self.layersplit_state.cp_size > 1
                and self.layersplit_state.owner_local_alloc
                and self.layersplit_state.ownership is not None):
            local_attention_heads = [
                int(h) for h in self.num_kv_heads_per_layer if int(h) > 0
            ]
            if not local_attention_heads:
                raise RuntimeError(
                    "LayerSplit owner-local transfer cannot infer global "
                    "KV-head metadata from an empty local layer vector")
            if len(set(local_attention_heads)) != 1:
                raise RuntimeError(
                    "LayerSplit owner-local transfer requires an explicit "
                    "global KV-head vector for heterogeneous per-layer heads")
            transfer_layers = int(
                getattr(self, "layersplit_cache_transfer_model_layers", 0)
                or getattr(self, "layersplit_model_num_layers", 0)
                or self.layersplit_state.ownership.num_layers)
            self.layersplit_transfer_num_kv_heads_per_layer = [
                local_attention_heads[0] for _ in range(transfer_layers)
            ]
            logger.info(
                "LayerSplit owner-local transfer metadata: local_layers=%d, "
                "global_layers=%d, kv_heads_per_layer=%d",
                len(local_attention_heads), transfer_layers,
                local_attention_heads[0])
            if os.environ.get("TRTLLM_OPTRT_DSA_LAYERSPLIT_DEBUG",
                              "0") == "1":
                print(
                    "OPTRT_DSA_LAYERSPLIT_DEBUG "
                    f"cp_rank={self.layersplit_state.cp_rank} "
                    f"cp_size={self.layersplit_state.cp_size} "
                    f"local_attention_layers={len(local_attention_heads)} "
                    f"global_layers={transfer_layers} "
                    f"local_pool_layers={len(getattr(self, 'num_kv_heads_per_layer', []))} "
                    f"layer_offsets={list(getattr(self, 'layer_offsets', {}).keys())[:8]}...",
                    flush=True)
        self.num_blocks = self.blocks_in_primary_pool

        # KVarN dense-MLA-latent side-pool (parallels the indexer-K pool):
        # when sparse_attn_config.mla_latent_kv_dtype is "kvarn_k<ckv>v<pe>",
        # each fully-filled MLA latent block (compressed_kv + k_pe) is stored
        # variance-normalized + low-bit in a per-layer flat uint8 side-pool,
        # and reconstructed to fp16 on read. The tile == one paged block, so
        # the KVarN group is bound to tokens_per_block. The main C++ latent
        # pool stays as the fp16/quant staging target the append/decode
        # kernels write/read; KVarN holds the committed compressed long
        # context (this is the Stage-a wiring; the C++ fold into dsv3Rope is
        # the zero-staging end-state). Inert (None) when not selected.
        self.kvarn_cfg = resolve_kvarn_config(
            getattr(sparse_attn_config, "mla_latent_kv_dtype", "auto"),
            kv_lora_rank=getattr(sparse_attn_config, "kv_lora_rank", None),
            qk_rope_head_dim=getattr(sparse_attn_config, "qk_rope_head_dim",
                                     None),
        )
        # AMORTIZED restore (default OFF for safety / clean main default).
        # When on, kvarn_restore_for_decode only re-dequants blocks whose fp16
        # main-pool slot is stale vs pool.commit_gen (the per-step churn), not
        # the full B*32 working set. Toggle via mla_latent_kv_amortize on the
        # sparse-attn config or the TRTLLM_KVARN_AMORTIZE env var.
        self.kvarn_amortize_restore = bool(
            getattr(sparse_attn_config, "mla_latent_kv_amortize", False)
            or os.environ.get("TRTLLM_KVARN_AMORTIZE", "") in ("1", "true", "True"))
        self.kvarn_latent_pool_per_layer = []
        self.kvarn_hisparse_source_layout = None
        if self.kvarn_cfg is not None:
            # The KVarN side-pool mirrors the dense MLA KV pool, not the
            # Indexer-K pool.  Build it on the primary KV device; the
            # Indexer-K side-pool is intentionally allocated later and remains
            # controlled only by indexer_k_dtype.
            if self.num_local_layers > 0 and self.layer_offsets:
                first_layer_idx = next(iter(self.layer_offsets))
                dev = self.get_buffers(first_layer_idx, kv_layout="NHD").device
            else:
                dev = torch.device("cuda")
            self.kvarn_latent_pool_per_layer = [
                KVarNLatentPool(self.num_blocks, self.tokens_per_block,
                                self.kvarn_cfg, dev)
                for _ in range(self.num_local_layers)
            ]
            self.kvarn_hisparse_source_layout = KVARN_LEGACY_SIDEPOOL_LAYOUT
            logger.info(
                "KVarN MLA-latent backend ENABLED (%s): group=%d, "
                "%d B/block, %.3f bits/elem, side-pool %d blocks x %d layers "
                "= %.2f GiB; vs fp16 latent %.2fx, vs fp8 %.2fx.",
                self.kvarn_cfg.name, self.tokens_per_block,
                self.kvarn_cfg.packed_bytes(self.tokens_per_block),
                self.kvarn_cfg.bits_per_elem(self.tokens_per_block),
                self.num_blocks, self.num_local_layers,
                self.kvarn_cfg.packed_bytes(self.tokens_per_block)
                * self.num_blocks * self.num_local_layers / 2**30,
                self.kvarn_cfg.fp16_bytes(self.tokens_per_block)
                / self.kvarn_cfg.packed_bytes(self.tokens_per_block),
                self.kvarn_cfg.fp8_bytes(self.tokens_per_block)
                / self.kvarn_cfg.packed_bytes(self.tokens_per_block))

        self.hisparse_coordinator = OPTRTHiSparseCoordinator(
            sparse_attn_config, kv_cache_manager=self)
        if self.hisparse_coordinator.enabled:
            hisparse_device = dev if self.kvarn_cfg is not None else torch.device(
                "cuda")
            self.hisparse_coordinator.configure_from_kv_cache_manager()
            self.hisparse_coordinator.allocate_packed_tensors(
                device=hisparse_device, host_pinned=prefer_pinned())
        self.hisparse_coordinator.assert_startup_ready()
        if self.hisparse_coordinator.enabled:
            logger.info(
                "OP-TRT HiSparse enabled: mode=%s, topk=%s, hot_blocks_per_req=%s, "
                "host_to_device_ratio=%s. Packed tiers are allocated; serving "
                "remains fail-closed until swap-in/read kernels are ready.",
                getattr(sparse_attn_config, "hisparse_mode", None),
                getattr(sparse_attn_config, "hisparse_topk", None),
                getattr(sparse_attn_config, "hisparse_hot_blocks_per_req", None),
                getattr(sparse_attn_config, "hisparse_host_to_device_ratio",
                        None),
            )

        # Indexer K cache pool for DSA attention
        # Shape: [num_blocks, self.tokens_per_block * (index_head_dim + scale_size)]
        # Non-interleaved layout: [fp8_tok0 | fp8_tok1 | ... | scale_tok0 | scale_tok1 | ...]
        # Store FP8-quantized k values from the indexer
        self.indexer_k_cache_pool_per_layer = [
            self.get_indexer_k_cache_pool_data(layer_idx)
            for layer_idx in range(self.num_local_layers)
        ]
        self.enable_hisa_page_reps = (
            self.use_fp4 and sparse_attn_config.indexer_mode == "indexcache-hisa"
            and getattr(sparse_attn_config, "enable_nvfp4_hisa", False))
        if self.enable_hisa_page_reps:
            self.indexer_hisa_page_reps_per_layer = [
                torch.empty((self.num_blocks, self.index_head_dim),
                            dtype=torch.float32,
                            device=pool.device)
                for pool in self.indexer_k_cache_pool_per_layer
            ]
            self.indexer_hisa_page_counts_per_layer = [
                torch.zeros((self.num_blocks, ),
                            dtype=torch.int32,
                            device=pool.device)
                for pool in self.indexer_k_cache_pool_per_layer
            ]
        else:
            self.indexer_hisa_page_reps_per_layer = []
            self.indexer_hisa_page_counts_per_layer = []

        # LayerSplit M5d-tight: when LayerSplit is enabled and CP > 1
        # AND this rank owns at least one layer, allocate a single
        # shared scratch buffer per cache type for non-owned layers.
        # The scratch buffer is sized identically to a per-layer pool
        # slot so the M5e broadcast (gather → broadcast → scatter back)
        # can index into it via the same active_block_ids that index
        # into pool slots on owner ranks, and the downstream attention
        # kernel reads from it via the same get_indexer_k_cache_buffers
        # / get_buffers accessors (overridden below to dispatch on
        # ownership). One scratch tensor suffices because layers
        # execute sequentially in the forward — layer L+1's broadcast
        # overwrites layer L's scratch bytes after layer L's attention
        # completes, and CUDA-graph capture serializes this dependency
        # chain correctly.
        self._layersplit_indexer_k_scratch = None
        self._layersplit_dense_kv_scratch = None
        self._layersplit_hisa_pagerep_scratch = None
        self._layersplit_hisa_pagecount_scratch = None
        # Only allocate the non-owned-layer scratch in the owner-local-alloc
        # posture. In the default (replicated) posture every layer has a real
        # pool slot, so the accessors must return the pool slot (not scratch)
        # and the per-layer broadcast lands directly in the slot the dense-MLA
        # C++ kernels read. Leaving the scratch tensors None makes
        # get_indexer_k_cache_buffers / get_buffers / get_indexer_hisa_page_
        # rep_buffers fall through to the pool-backed path for every layer.
        if (self.layersplit_state.enabled
                and self.layersplit_state.cp_size > 1
                and self.layersplit_state.owner_local_alloc):
            owned = self._layersplit_local_pool_layers()
            expected_owned = tuple(
                self.layersplit_state.ownership.owned_layers(
                    self.layersplit_state.cp_rank))
            if expected_owned and set(expected_owned) != set(owned):
                logger.warning(
                    "LayerSplit owner-local pool ownership differs from policy "
                    "helper: policy=%s actual_pool_layers=%s. Using "
                    "layer_offsets as the dense/indexer read-path source of "
                    "truth.", expected_owned, owned)
            if owned:
                # Use the first actual local pool layer's pool-slot tensor as the
                # template — same shape / dtype / device as every
                # non-owned layer's slot would have if it were allocated.
                first_owned = owned[0]
                try:
                    indexer_template = self._get_indexer_k_cache_buffers_owned(
                        first_owned)
                    self._layersplit_indexer_k_scratch = torch.empty_like(
                        indexer_template)
                except (KeyError, IndexError, AttributeError):
                    # Pool not yet initialized in this draft / spec
                    # path; LayerSplit broadcast will short-circuit
                    # when the scratch is None (M5e helper no-ops on
                    # None cache_slot).
                    pass
                try:
                    # The dense scratch must mirror the *raw* per-layer pool
                    # slot that the dense-MLA C++ attention reads through the
                    # pool pointer (get_primary_pool_data is a view of one C++
                    # pool slot: [num_blocks, kv_factor, block_size]). Under
                    # NVFP4 the base get_buffers() NHD reshape uses the logical
                    # latent width, which does not match the packed pool bytes,
                    # so always mirror the raw pool row. The scratch gets its
                    # OWN storage (torch.empty_like, not a pool view) so its
                    # .data_ptr() can be handed to the C++ attention as a
                    # standalone single-layer pool for non-owned layers. See
                    # _build_layersplit_dense_scratch_pool below.
                    first_owned_offset = self.layer_offsets[first_owned]
                    dense_template = self.impl.get_primary_pool_data(
                        first_owned_offset)
                    if dense_template is not None:
                        self._layersplit_dense_kv_scratch = torch.empty_like(
                            dense_template)
                except (KeyError, IndexError, AttributeError, RuntimeError):
                    pass
                if self.enable_hisa_page_reps:
                    # Non-owned layers recompute HISA page reps locally from
                    # the broadcast indexer-K scratch, so they need their own
                    # shared page-rep/count scratch (one slot, reused per layer
                    # like the indexer-K scratch above).
                    try:
                        pr_tmpl, pc_tmpl = (
                            self._get_indexer_hisa_page_rep_buffers_owned(
                                first_owned))
                        self._layersplit_hisa_pagerep_scratch = torch.empty_like(
                            pr_tmpl)
                        self._layersplit_hisa_pagecount_scratch = (
                            torch.empty_like(pc_tmpl))
                    except (KeyError, IndexError, AttributeError):
                        pass
                if self._layersplit_indexer_k_scratch is not None:
                    total_layers = self.layersplit_state.ownership.num_layers
                    logger.info(
                        "LayerSplit M5d-tight scratch buffers allocated for "
                        "%d non-owned layers (indexer_k: %s, dense_kv: %s); "
                        "per-rank memory savings ≈ %.0f%% vs replicated.",
                        total_layers - len(owned),
                        tuple(self._layersplit_indexer_k_scratch.shape),
                        tuple(self._layersplit_dense_kv_scratch.shape)
                        if self._layersplit_dense_kv_scratch is not None
                        else "n/a",
                        100.0 * (1.0 - len(owned) / total_layers),
                    )
        # Wire the dense scratch into the C++ dense-MLA attention as a real
        # single-layer pool the kernel can address (the owner-local-alloc
        # memory-saving read path). See _build_layersplit_dense_scratch_pool.
        self._layersplit_dense_scratch_pool_index = None
        self._layersplit_kv_cache_pool_pointers_ls = None
        self._layersplit_kv_cache_pool_mapping_ls = None
        self._layersplit_nonowned_layer_rows = {}
        if (self.layersplit_state.enabled
                and self.layersplit_state.cp_size > 1
                and self.layersplit_state.owner_local_alloc
                and self._layersplit_dense_kv_scratch is not None):
            self._build_layersplit_dense_scratch_pool()

    def _layersplit_local_pool_layers(self) -> Tuple[int, ...]:
        """Global layer ids with real local C++ KV/indexer/KVarN pool slots."""
        return tuple(sorted(int(layer) for layer in self.layer_offsets.keys()))

    def _layersplit_should_use_scratch(self, layer_idx: int) -> bool:
        return bool(self.layersplit_state.enabled
                    and self.layersplit_state.cp_size > 1
                    and self.layersplit_state.owner_local_alloc
                    and int(layer_idx) not in self.layer_offsets)

    def _layersplit_nonlocal_pool_layers(self) -> Tuple[int, ...]:
        if not (self.layersplit_state.enabled
                and self.layersplit_state.owner_local_alloc
                and self.layersplit_state.ownership is not None):
            return ()
        owned_set = set(self._layersplit_local_pool_layers())
        total_layers = self.layersplit_state.ownership.num_layers
        return tuple(layer for layer in range(total_layers)
                     if layer not in owned_set)

    def _build_layersplit_dense_scratch_pool(self) -> None:
        """Expose the non-owned-layer dense scratch as a real C++-addressable pool.

        Under ``owner_local_alloc=True`` each CP rank trims its dense KV pool
        to the layers it owns, so a non-owned layer has no pool slot. The
        dense-MLA C++ attention op resolves its KV byte address purely from
        ``(host_kv_cache_pool_pointers, host_kv_cache_pool_mapping,
        kv_cache_block_offsets, local_layer_idx)`` — it never reads the Python
        ``get_buffers()`` accessor — so returning a separate Python scratch
        tensor from ``get_buffers`` (the M5d-tight scaffold) does NOT feed the
        dense kernel. To make non-owned layers correct we append the dense
        scratch as one extra *single-layer* pool:

        * ``kv_cache_pool_pointers``: append a row holding the scratch
          ``data_ptr`` (and, under NVFP4, the scratch block-scale ``data_ptr``)
          with the same ``[primary, secondary]`` / ``[..., data, scale]`` shape
          as the real rows. The scratch pool is primary-only (secondary = 0).
        * ``kv_cache_pool_mapping``: append one row per non-owned layer, each
          mapping to ``(scratch_pool_index, layer_idx_in_pool=0)`` so the C++
          ``intra_pool_offset`` is 0 (the scratch is single-layer). Each
          non-owned layer gets a *distinct* mapping row index (its
          ``local_layer_idx``) so the C++ attention op's per-layer config cache
          key stays unique, while all rows point at the same scratch storage.
        * ``kv_cache_block_offsets``: the scratch pool's per-step block table is
          re-encoded from pool 0's (decode the global block id, re-encode with
          the single-layer stride). Handled in the metadata prepare via
          :meth:`fill_layersplit_scratch_block_offsets`; the scratch tensor row
          ``b`` holds global block ``b`` so the offset value is the global block
          id (kv_factor=1 for SELFKONLY).

        The owner-layer rows of both pointers and mapping are unchanged, so
        owned layers keep reading their real trimmed-pool slots. The indexer-K
        path is unaffected (it uses the Python pool list + scratch, not these
        C++ pool pointers).
        """
        if (self._layersplit_dense_scratch_pool_index is not None
                and self._layersplit_nonowned_layer_rows):
            return
        owned = self._layersplit_local_pool_layers()
        non_owned = list(self._layersplit_nonlocal_pool_layers())
        if not non_owned:
            return

        base_pointers = self.kv_cache_pool_pointers
        base_mapping = self.kv_cache_pool_mapping
        if base_pointers is None or base_mapping is None:
            return

        # --- Augmented pool pointers: one extra single-layer scratch pool. ---
        # base_pointers shape is [num_pools, 2] (primary, secondary) or
        # [num_pools, 2, 2] (primary/secondary x data/scale) under NVFP4.
        scratch_primary = int(self._layersplit_dense_kv_scratch.data_ptr())
        if base_pointers.dim() == 3:
            # NVFP4: [num_pools, 2, 2]; columns are (data, scale).
            scratch_scale = 0
            if self.dtype == DataType.NVFP4:
                try:
                    # Mirror the dense block-scale pool row for the scratch so
                    # the kernel reads matching E4M3 scales for the broadcast
                    # latent. One shared scratch scale slot, reused per layer.
                    scale_pool = self.get_dense_block_scale_pool()
                    first_owned_offset = self.layer_offsets[owned[0]]
                    scale_template = scale_pool.index_select(
                        0 if scale_pool.shape[0] == self.num_blocks else 1,
                        torch.tensor([0], device=scale_pool.device))
                    self._layersplit_dense_scale_scratch = torch.empty(
                        (self.num_blocks, ) +
                        tuple(self._dense_scale_row_shape(scale_pool)),
                        dtype=scale_pool.dtype,
                        device=scale_pool.device)
                    scratch_scale = int(
                        self._layersplit_dense_scale_scratch.data_ptr())
                except (KeyError, IndexError, AttributeError, RuntimeError,
                        AssertionError):
                    self._layersplit_dense_scale_scratch = None
            scratch_row = torch.tensor(
                [[[scratch_primary, scratch_scale], [0, 0]]],
                dtype=base_pointers.dtype,
                device=base_pointers.device)
        else:
            scratch_row = torch.tensor([[scratch_primary, 0]],
                                       dtype=base_pointers.dtype,
                                       device=base_pointers.device)
        self._layersplit_dense_scratch_pool_index = int(base_pointers.shape[0])
        self._layersplit_kv_cache_pool_pointers_ls = torch.cat(
            [base_pointers, scratch_row], dim=0).contiguous()

        # --- Augmented pool mapping: one row per non-owned layer. ---
        # base_mapping is a CPU int32 tensor [num_local_layers, 2].
        scratch_pool_idx = self._layersplit_dense_scratch_pool_index
        extra_rows = []
        next_row = int(base_mapping.shape[0])
        for layer_idx in non_owned:
            self._layersplit_nonowned_layer_rows[layer_idx] = next_row
            extra_rows.append([scratch_pool_idx, 0])
            next_row += 1
        extra = torch.tensor(extra_rows,
                             dtype=base_mapping.dtype,
                             device=base_mapping.device)
        self._layersplit_kv_cache_pool_mapping_ls = torch.cat(
            [base_mapping, extra], dim=0).contiguous()

        # The scratch pool needs a per-step block table row appended to
        # host/device kv_cache_block_offsets. Grow the host tensor by one pool
        # row (the metadata sizes its device tensor from the host tensor's pool
        # dim, and copy_batch_block_offsets loops over every host pool row). The
        # indexer path keeps using the real num_pools via num_local_layers and
        # only ever reads pool 0, so the extra row is inert for it.
        old_host = self.host_kv_cache_block_offsets
        grown = torch.zeros((old_host.shape[0] + 1, ) + tuple(old_host.shape[1:]),
                            dtype=old_host.dtype,
                            pin_memory=old_host.is_pinned(),
                            device=old_host.device)
        grown[:old_host.shape[0]].copy_(old_host)
        self.host_kv_cache_block_offsets = grown
        logger.info(
            "LayerSplit owner-local dense read path: scratch pool index=%d, "
            "%d non-owned layers routed to mapping rows %d..%d (shared single "
            "scratch slot).", scratch_pool_idx, len(non_owned),
            int(base_mapping.shape[0]), next_row - 1)

    def _ensure_layersplit_dense_scratch_routing(self) -> bool:
        """Ensure owner-local non-owned dense layers have C++ scratch routing.

        Some runtime constructions make the C++ pool pointer/mapping tensors
        available only after ``DSACacheManager.__init__`` has already attempted
        the eager build. Repair that lazily before metadata buffer allocation,
        block-offset copies, or attention local-layer lookup. The repair keeps
        owner-local memory savings: one shared dense scratch slot is appended as
        a single-layer pool and every non-owned layer maps to that slot.
        """
        if not (self.layersplit_state.enabled
                and self.layersplit_state.cp_size > 1
                and self.layersplit_state.owner_local_alloc):
            return False
        if (self._layersplit_dense_scratch_pool_index is not None
                and self._layersplit_nonowned_layer_rows):
            return True

        if self._layersplit_dense_kv_scratch is None:
            try:
                local_offsets = list(getattr(self, 'layer_offsets', {}).values())
                local_offset = local_offsets[0] if local_offsets else 0
                dense_template = self.impl.get_primary_pool_data(local_offset)
                if dense_template is not None:
                    self._layersplit_dense_kv_scratch = torch.empty_like(
                        dense_template)
                    logger.info(
                        "LayerSplit owner-local lazily allocated dense KV scratch "
                        "from local pool offset %d: %s.", local_offset,
                        tuple(self._layersplit_dense_kv_scratch.shape))
            except (KeyError, IndexError, AttributeError, RuntimeError):
                return False

        if self._layersplit_dense_kv_scratch is None:
            return False
        self._build_layersplit_dense_scratch_pool()
        return (self._layersplit_dense_scratch_pool_index is not None
                and bool(self._layersplit_nonowned_layer_rows))

    @staticmethod
    def _dense_scale_row_shape(scale_pool: torch.Tensor) -> Tuple[int, ...]:
        """Per-block row shape of a dense block-scale pool slot.

        The block-scale pool mirrors the data pool geometry
        ``[num_blocks, num_layers, kv_factor, ...]`` (block-first), so one
        layer's per-block row is everything after the (block, layer) axes.
        """
        # Drop the leading block axis and the layer axis.
        return tuple(scale_pool.shape[2:])

    def copy_batch_block_offsets(self, dst_tensor: torch.Tensor,
                                 request_ids, beam_width: int,
                                 num_context: int, num_seqs: int):
        """Fill block-offset tables, including the LayerSplit dense scratch pool.

        The base implementation fills the real (trimmed) pools' rows in
        ``self.host_kv_cache_block_offsets`` from the C++ block manager and
        copies every host pool row into ``dst_tensor``. Under owner-local alloc
        we first populate the appended scratch-pool row so it is copied in the
        same loop: the single-layer scratch pool's offset value for a logical
        block is that block's global memory-pool index. Pool 0 encodes
        ``global_block_id * num_local_layers`` (block-first layout, see
        BlockManager::setOffsets), so dividing pool 0's freshly-filled row by
        ``num_local_layers`` recovers the global block id (the scratch tensor
        row ``b`` holds global block ``b``; kv_factor=1 for SELFKONLY). When the
        scratch pool is inactive this is exactly the base behavior.
        """
        self._ensure_layersplit_dense_scratch_routing()
        scratch_idx = self._layersplit_dense_scratch_pool_index
        host = self.host_kv_cache_block_offsets
        if scratch_idx is not None and host.shape[0] > scratch_idx:
            # The C++ copy below only writes the real pools; fill the real
            # pools first by reusing the base host-fill, then derive scratch.
            self.impl.copy_batch_block_offsets(host, request_ids[:num_context],
                                               1, 0)
            self.impl.copy_batch_block_offsets(host,
                                               request_ids[num_context:],
                                               beam_width, num_context)
            num_local_layers = max(1, self.num_local_layers)
            host[scratch_idx, :num_seqs].copy_(
                host[0, :num_seqs] // num_local_layers)
            for pool_idx in range(host.shape[0]):
                dst_tensor[pool_idx, :num_seqs].copy_(host[pool_idx, :num_seqs],
                                                      non_blocking=True)
            return
        super().copy_batch_block_offsets(dst_tensor, request_ids, beam_width,
                                         num_context, num_seqs)

    def _get_indexer_k_cache_buffers_owned(self, layer_idx: int):
        """Pool-backed accessor for layers this CP rank owns.

        Bypasses the LayerSplit M5d-tight dispatch in
        :meth:`get_indexer_k_cache_buffers`; intended for the scratch
        allocator and for callers that have already verified ownership.
        """
        block_size = self.tokens_per_block
        data_bytes = self.index_head_dim // 2 if self.use_fp4 else self.index_head_dim
        per_token_size = data_bytes + self.index_head_dim // self.quant_block_size * 4
        layer_offset = self.layer_offsets[layer_idx]
        return self.indexer_k_cache_pool_per_layer[layer_offset].view(
            self.num_blocks, block_size, 1, per_token_size)

    def _layersplit_non_owned(self, layer_idx: int) -> bool:
        return self._layersplit_should_use_scratch(layer_idx)

    def _ensure_layersplit_indexer_k_scratch(self):
        """Lazily allocate the non-owned-layer indexer-K scratch slot.

        The eager constructor path normally builds this from the first owned
        layer. Some production mappings trim ``layer_offsets`` before the global
        owner table can be queried, so the eager path can miss even though this
        rank still has a local indexer pool. The scratch only needs to mirror one
        pool slot; building it from local offset 0 preserves owner-local memory
        savings and keeps non-owned writes/reads on the LayerSplit scratch path.
        """
        if self._layersplit_indexer_k_scratch is None:
            pools = getattr(self, "indexer_k_cache_pool_per_layer", None)
            if pools:
                block_size = self.tokens_per_block
                data_bytes = (self.index_head_dim // 2
                              if self.use_fp4 else self.index_head_dim)
                per_token_size = (data_bytes + self.index_head_dim
                                  // self.quant_block_size * 4)
                template = pools[0].view(self.num_blocks, block_size, 1,
                                         per_token_size)
                self._layersplit_indexer_k_scratch = torch.empty_like(template)
                logger.info(
                    "LayerSplit owner-local lazily allocated indexer-K scratch "
                    "from local pool offset 0: %s.",
                    tuple(self._layersplit_indexer_k_scratch.shape))
        return self._layersplit_indexer_k_scratch

    def _ensure_layersplit_hisa_scratch(self):
        """Lazily allocate HISA page-rep scratch for non-owned layers."""
        if (self._layersplit_hisa_pagerep_scratch is None
                and self.enable_hisa_page_reps
                and self.indexer_hisa_page_reps_per_layer
                and self.indexer_hisa_page_counts_per_layer):
            self._layersplit_hisa_pagerep_scratch = torch.empty_like(
                self.indexer_hisa_page_reps_per_layer[0])
            self._layersplit_hisa_pagecount_scratch = torch.empty_like(
                self.indexer_hisa_page_counts_per_layer[0])
            logger.info(
                "LayerSplit owner-local lazily allocated HISA scratch from "
                "local pool offset 0: reps=%s counts=%s.",
                tuple(self._layersplit_hisa_pagerep_scratch.shape),
                tuple(self._layersplit_hisa_pagecount_scratch.shape))
        return (self._layersplit_hisa_pagerep_scratch,
                self._layersplit_hisa_pagecount_scratch)

    @property
    def kvarn_enabled(self) -> bool:
        return bool(self.kvarn_latent_pool_per_layer)

    def get_kvarn_latent_pool(self, layer_idx: int) -> "KVarNLatentPool":
        """KVarN side-pool for a local dense-MLA layer.

        Owner-local LayerSplit trims dense MLA side-pools to the same actual
        local layer set as the C++ KV pool. Non-local layers read dense KV from
        the broadcast scratch and must not commit/restore KVarN entries on the
        receiving rank; the owner rank handles the side-pool for that layer.
        """
        if not self.kvarn_enabled:
            return None
        layer_offset = self.layer_offsets.get(layer_idx)
        if layer_offset is None:
            return None
        return self.kvarn_latent_pool_per_layer[layer_offset]

    def kvarn_store_block(self, layer_idx: int, block_id: int,
                          ckv, k_pe) -> None:
        """Quantize+commit one full fp16 latent block into the side-pool."""
        pool = self.get_kvarn_latent_pool(layer_idx)
        if pool is not None:
            pool.store_block(int(block_id), ckv, k_pe)

    def kvarn_load_block(self, layer_idx: int, block_id: int):
        """Reconstruct (ckv, k_pe) fp16 for one committed block, else None."""
        pool = self.get_kvarn_latent_pool(layer_idx)
        if pool is None or not bool(pool.valid[int(block_id)]):
            return None
        return pool.load_block(int(block_id))

    def kvarn_packed_source_fragments(self, layer_indices,
                                      block_ids) -> Tuple[np.ndarray, np.ndarray]:
        """Layer-major source fragments for committed packed KVarN records."""
        if not self.kvarn_enabled:
            raise RuntimeError(
                "HiSparse direct-to-host requires dense MLA KVarN source pools.")
        ptr_parts = []
        size_parts = []
        for layer_idx in layer_indices:
            pool = self.get_kvarn_latent_pool(int(layer_idx))
            if pool is None:
                raise RuntimeError(
                    "HiSparse direct-to-host cannot source non-local KVarN "
                    f"layer {int(layer_idx)} from this rank.")
            ptrs, sizes = pool.packed_source_fragments(block_ids)
            ptr_parts.append(ptrs)
            size_parts.append(sizes)
        if not ptr_parts:
            return (np.array([], dtype=np.int64),
                    np.array([], dtype=np.int64))
        return (np.concatenate(ptr_parts).astype(np.int64, copy=False),
                np.concatenate(size_parts).astype(np.int64, copy=False))

    def kvarn_bytes_per_token(self, num_attention_layers: int) -> float:
        if not self.kvarn_enabled:
            return 0.0
        return kvarn_latent_bytes_per_token(self.kvarn_cfg,
                                            self.tokens_per_block,
                                            num_attention_layers)

    # -- KVarN side-pool lifecycle (block free / recycle) -------------------

    def _kvarn_invalidate_on_free(self) -> bool:
        """Correctness gate (default ON): clear the KVarN side-pool records
        of paged blocks the allocator reclaims. Without it a recycled block
        id keeps the dying owner's ``valid=True`` record: the new owner's
        commit is skipped (idempotence keys on ``valid``) and the full-scan
        restore rewrites the stale record over the new owner's fresh fp16
        latent -- a silent cross-request KV clobber. Set
        TRTLLM_KVARN_INVALIDATE_ON_FREE=0 to restore the old behavior."""
        return (self.kvarn_enabled and os.environ.get(
            "TRTLLM_KVARN_INVALIDATE_ON_FREE", "1") != "0")

    def _kvarn_invalidate_block_ids(self, block_ids) -> None:
        """Clear committed side-pool records for ``block_ids`` (host ints)
        on every local layer pool. Block ids equal primary-pool slot indices
        in the KVarN posture (no host offload; see _get_pool_block_indices --
        with an empty secondary pool the encoded offset decode is the
        identity on cache-block ids)."""
        if getattr(self, "blocks_in_secondary_pool", 0):
            # Host-offload posture: cache-block ids are not pool slots and a
            # free-time decode is unavailable. KVarN does not support
            # offload; leave the pools untouched rather than guess.
            return
        ids = [b for b in block_ids if 0 <= b < self.num_blocks]
        if not ids:
            return
        pools = self.kvarn_latent_pool_per_layer
        dev_ids = torch.as_tensor(ids, dtype=torch.long,
                                  device=pools[0].device)
        for pool in pools:
            pool.invalidate_blocks(ids, dev_ids)

    def free_resources(self, request, pin_on_release: bool = False):
        """Release a request's blocks; drop their KVarN records first.

        The ids are fetched while the request still owns its sequence.
        Invalidating before the C++ free also covers the pinned-release
        path: pinned blocks stay unreclaimable until unpinned, and the
        later unpin-time free has no Python hook."""
        if self._kvarn_invalidate_on_free():
            try:
                ids = self.get_cache_indices(request)
            except (IndexError, RuntimeError):
                # No sequence for this request id (double-free / freed before
                # resource prepare). removeSequence no-ops on the same input,
                # so degrade to the pre-hook behavior instead of raising.
                ids = []
            self._kvarn_invalidate_block_ids(ids)
        return super().free_resources(request, pin_on_release)

    def rewind_kv_cache(self, request, rewind_len: int):
        """Rewind (spec-decode reject) is the other path returning blocks to
        the allocator without ``free_resources``. Invalidate the records of
        every block the rewind freed plus the new tail block: a block that
        committed when it filled and then shrank back to partial re-fills
        with different tokens, so its record is stale for the SAME owner."""
        if not (rewind_len > 0 and self._kvarn_invalidate_on_free()):
            return super().rewind_kv_cache(request, rewind_len)
        pre = self.get_cache_indices(request)
        ret = super().rewind_kv_cache(request, rewind_len)
        post_n = len(self.get_cache_indices(request))
        self._kvarn_invalidate_block_ids(pre[max(post_n - 1, 0):])
        return ret

    def get_indexer_k_cache_buffers(self, layer_idx: int):
        """Get indexer K cache buffer for a layer.

        M5d-tight dispatch: when LayerSplit is enabled and this rank is
        NOT the owner for ``layer_idx``, return the per-rank scratch
        buffer (one shared tensor for all non-owned layers; sequential
        layer execution means each layer's broadcast overwrites the
        prior layer's content in place). The downstream
        ``sparse_attn_indexer`` reads from this buffer unchanged because
        the shape mirrors a pool slot exactly. On the off-path or for
        owned layers, fall through to the pool-backed accessor.
        """
        if self._layersplit_non_owned(layer_idx):
            scratch = self._ensure_layersplit_indexer_k_scratch()
            if scratch is None:
                raise RuntimeError(
                    "LayerSplit owner-local indexer-K scratch is unavailable "
                    f"for non-owned layer {layer_idx}; cannot fall through to "
                    "the owned layer pool without breaking ownership.")
            return scratch
        return self._get_indexer_k_cache_buffers_owned(layer_idx)

    def get_buffers(self,
                    layer_idx: int,
                    kv_layout: str = "NHD"):
        """Get dense KV cache buffer for a layer.

        M5d-tight dispatch: a non-owned layer under owner-local alloc has no
        real pool slot (the C++ pool is trimmed to owned layers), so it must
        read the shared ``_layersplit_dense_kv_scratch`` -- the same storage
        whose ``data_ptr`` the augmented pool pointers expose to the dense-MLA
        C++ kernel, filled in place by this layer's owner broadcast (M5f).

        The scratch mirrors the RAW per-layer pool row (``[num_blocks,
        kv_factor, block_size]``), not the kv_layout-dependent NHD/HND reshape
        the base accessor applies for owned layers: the only consumers of a
        NON-owned slot are (a) the M5f broadcast, which flattens to 2-D before
        gather/scatter, and (b) the C++ kernel via the augmented pool pointer,
        which addresses raw bytes -- neither depends on the logical NHD/HND
        view. Routing every layout to the scratch is therefore correct AND
        closes the latent fall-through: the previous ``kv_layout == "NHD"``
        guard let an HND (or any non-NHD) request for a non-owned layer fall
        through to ``super().get_buffers``, which does
        ``self.layer_offsets[layer_idx]`` and KeyErrors on a trimmed layer
        (silently reading the wrong KV would be worse). Owned layers always
        fall through to the base accessor with the requested layout.
        """
        if self._layersplit_non_owned(layer_idx):
            if self._layersplit_dense_kv_scratch is None:
                self._ensure_layersplit_dense_scratch_routing()
            if self._layersplit_dense_kv_scratch is not None:
                return self._layersplit_dense_kv_scratch
            raise RuntimeError(
                "LayerSplit owner-local dense KV scratch is unavailable for "
                f"non-owned layer {layer_idx} (kv_layout={kv_layout!r}); "
                "cannot fall through to the owned dense pool without breaking "
                "ownership. Run with layersplit_owner_local_alloc=False "
                "(replicated pools) if scratch allocation cannot be repaired.")
        return super().get_buffers(layer_idx, kv_layout=kv_layout)

    def get_dense_block_scale_pool(self) -> torch.Tensor:
        """All-layer NVFP4 block-scale pool, matching the dense data pool shape.

        The dense KV data pool (``get_unique_primary_pool``) carries the packed
        E2M1 latent (288 bytes/token); under NVFP4 there is a sibling
        block-scale pool with one E4M3 scale per 16 elements (36 bytes/token).
        The base V1 ``KVCacheManager`` exposes only the data pool, while the C++
        block-scale pool (a parallel set of ``KVCacheBlockPool`` entries sharing
        the data pool's block-offset table and num_blocks x num_layers geometry)
        is reachable through ``get_block_scale_pool``. It is returned with the
        same ``[num_blocks, num_layers, kv_factor, ...]`` layout as
        ``get_unique_primary_pool`` (Float8_e4m3fn storage), so flattening it
        over the leading (block, layer, token) axes lets the same global token
        indices (from ``convert_req_index_to_global``) address both pools.
        """
        assert self.dtype == DataType.NVFP4, \
            "Dense block-scale pool is only present for NVFP4 KV cache"
        # KV data pool index 0 = the single dense MLA window.
        return self.impl.get_block_scale_pool(0)

    def get_dense_scale_slot(self, layer_idx: int):
        """Per-layer dense block-scale slot for the LayerSplit broadcast.

        Returns the ``[num_blocks, scale_row...]`` block-scale rows for
        ``layer_idx``: a view of the C++ scale pool slot for an owned layer, or
        the shared scale scratch (whose ``data_ptr`` the augmented pool pointers
        expose to the dense-MLA kernel) for a non-owned layer. Returns None when
        the dense scratch scale pool is not active (non-NVFP4 or replicated).
        """
        scale_scratch = getattr(self, "_layersplit_dense_scale_scratch", None)
        if scale_scratch is None or self.dtype != DataType.NVFP4:
            return None
        if self._layersplit_non_owned(layer_idx):
            return scale_scratch
        layer_offset = self.layer_offsets.get(layer_idx)
        if layer_offset is None:
            return None
        scale_pool = self.get_dense_block_scale_pool()
        # Block-first layout [num_blocks, num_layers, kv_factor, scale...].
        return scale_pool[:, layer_offset]

    def _get_indexer_hisa_page_rep_buffers_owned(self, layer_idx: int):
        """Pool-backed HISA page-rep accessor for layers this CP rank owns.

        Bypasses the LayerSplit M5d-tight dispatch in
        :meth:`get_indexer_hisa_page_rep_buffers`; intended for the scratch
        allocator and for callers that have already verified ownership.
        """
        layer_offset = self.layer_offsets[layer_idx]
        return (self.indexer_hisa_page_reps_per_layer[layer_offset],
                self.indexer_hisa_page_counts_per_layer[layer_offset])

    def get_indexer_hisa_page_rep_buffers(self, layer_idx: int):
        """Get maintained HISA page representatives for a layer.

        M5d-tight dispatch (mirrors :meth:`get_indexer_k_cache_buffers`):
        under LayerSplit the page-rep/count pools are sized to ``num_local_
        layers`` (owned only), so a non-owned ``layer_idx`` is absent from
        ``layer_offsets`` and would KeyError. Non-owned layers instead use a
        shared scratch, recomputed in place from that layer's broadcast
        indexer-K scratch by ``indexer_hisa_update_page_reps_nvfp4`` — the
        same one-slot-reused-per-layer scheme the indexer-K scratch uses.
        """
        if not self.enable_hisa_page_reps:
            raise RuntimeError("HISA page representatives are not enabled")
        if self._layersplit_non_owned(layer_idx):
            page_reps, page_counts = self._ensure_layersplit_hisa_scratch()
            if page_reps is None or page_counts is None:
                raise RuntimeError(
                    "LayerSplit owner-local HISA scratch is unavailable for "
                    f"non-owned layer {layer_idx}; cannot fall through to the "
                    "owned HISA pool without breaking ownership.")
            return (page_reps, page_counts)
        return self._get_indexer_hisa_page_rep_buffers_owned(layer_idx)

    def shutdown(self):
        """Release indexer cache pool references before C++ buffer cleanup."""
        self.indexer_hisa_page_reps_per_layer = []
        self.indexer_hisa_page_counts_per_layer = []
        self.indexer_k_cache_pool_per_layer = []
        super().shutdown()

    @staticmethod
    def get_cache_size_per_token(model_config: ModelConfig,
                                 mapping: Mapping,
                                 num_layers: Optional[int] = None,
                                 **kwargs):
        """Estimate total cache bytes per token including indexer K-cache overhead."""
        config = model_config.pretrained_config
        sparse_attn_config = model_config.sparse_attention_config
        index_head_dim = sparse_attn_config.index_head_dim
        quant_block_size = 128
        # Under FP4 the indexer stores two E2M1 codes per byte, so the
        # per-token data footprint halves (132 B -> 68 B at index_head_dim=128);
        # the scale bytes are unchanged (4 per token, one int32 holding four
        # UE8M0 exponents at quant_block_size=32 after packing).
        use_fp4 = sparse_attn_config.indexer_k_dtype == "fp4"
        indexer_data_dim = index_head_dim // 2 if use_fp4 else index_head_dim

        # get head dim (the dense MLA latent: kv_lora_rank + qk_rope_head_dim)
        head_dim = config.kv_lora_rank + config.qk_rope_head_dim

        num_attention_layers = KVCacheManager._resolve_num_attention_layers(
            model_config, mapping, num_layers)

        # The indexer K cache is a separate, already-packed payload; its
        # per-token footprint is a fixed byte count (indexer data bytes + int32
        # scale bytes per kv head). It must be added as raw bytes rather than
        # folded into the dense element count — under NVFP4 the dense path packs
        # two codes per byte and feeds a scale-factor sizer that requires a
        # 16-divisible element count (head_dim=576 is, head_dim+surcharge is
        # not).
        indexer_bytes_per_token = (indexer_data_dim +
                                   index_head_dim // quant_block_size * 4)

        quant_config = model_config.quant_config
        quant_mode = quant_config.quant_mode if quant_config is not None else None
        if quant_mode is not None and quant_mode.has_fp4_kv_cache():
            # Dense latent stored as NVFP4 data (4 bits/elem) + one E4M3 scale
            # per 16 elements.
            dense_bytes = get_size_in_bytes(head_dim, DataType.NVFP4)
            dense_bytes += KVCacheManager.calculate_scaling_factor_size_bytes(
                head_dim,
                quant_vector_size=16,
                scaling_factor_dtype=DataType.FP8)
            mem_per_token = num_attention_layers * (dense_bytes +
                                                    indexer_bytes_per_token)
            return mem_per_token

        # get kv cache dtype bytes (1 for FP8, 2 otherwise)
        mem_per_token = 2
        if quant_mode is not None and quant_mode.has_fp8_kv_cache():
            mem_per_token = 1
        mem_per_token *= num_attention_layers * (head_dim +
                                                 indexer_bytes_per_token)
        return mem_per_token

    def get_cache_bytes_per_token(self):
        """Compute actual cache bytes per token from instance configuration."""
        # The dense MLA latent (self.kv_factor * head_dim, i.e. the 512+64=576
        # kv_lora_rank + qk_rope_head_dim payload) is the part that is stored in
        # the configured KV dtype. The indexer K cache is a separate,
        # already-packed payload whose per-token footprint is a fixed byte
        # count (indexer data bytes + int32 scale bytes); it must NOT be folded
        # into the dense element count, otherwise an NVFP4 KV dtype both
        # mis-sizes those bytes (E2M1 packs two codes per byte) and feeds a
        # non-16-divisible element count into the scale-factor sizer (the dense
        # 576 latent is 16-divisible, but 576 + indexer surcharge is not).
        # Under FP4 the indexer data portion is halved (two E2M1 codes per
        # byte); the int32 scale bytes are unchanged.
        if self.dtype not in (DataType.FP8, DataType.HALF, DataType.BF16,
                              DataType.FLOAT, DataType.NVFP4):
            raise ValueError(f'Cannot support {self.dtype} KV cache.')

        num_kv_heads = sum(self.num_kv_heads_per_layer)
        dense_size_per_token = self.kv_factor * num_kv_heads * self.head_dim

        indexer_data_dim = self.index_head_dim // 2 if self.use_fp4 else self.index_head_dim
        indexer_bytes_per_token = (
            indexer_data_dim +
            self.index_head_dim // self.quant_block_size * 4) * num_kv_heads

        if self.dtype == DataType.NVFP4:
            # Size only the dense 576-wide latent through the NVFP4 data + scale
            # sizers (dense_size_per_token is 16-divisible); add the indexer K
            # bytes as raw bytes.
            dense_size_per_token = math.ceil(dense_size_per_token)
            cache_size_bytes_per_token = get_size_in_bytes(
                dense_size_per_token, self.dtype)
            cache_size_bytes_per_token += self.calculate_scaling_factor_size_bytes(
                dense_size_per_token,
                quant_vector_size=16,
                scaling_factor_dtype=DataType.FP8)
            cache_size_bytes_per_token += indexer_bytes_per_token
        else:
            cache_size_per_token = math.ceil(dense_size_per_token +
                                             indexer_bytes_per_token)
            cache_size_bytes_per_token = get_size_in_bytes(
                cache_size_per_token, self.dtype)
        return cache_size_bytes_per_token

# OP-TRT HiSparse Integration Plan

This document is the second-pass integration plan for building a HiSparse-style
hierarchical sparse-attention KV path on top of the current `op-trt` custom
stack. It is based on direct review of:

- LMSYS/SGLang HiSparse blog:
  https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/
- SGLang HiSparse guide:
  https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/hisparse_guide.md
- SGLang implementation:
  - `python/sglang/srt/managers/hisparse_coordinator.py`
  - `python/sglang/srt/mem_cache/allocator/hisparse.py`
  - `python/sglang/srt/mem_cache/hisparse_memory_pool.py`
  - `python/sglang/jit_kernel/csrc/hisparse.cuh`
  - `python/sglang/jit_kernel/hisparse.py`
  - `python/sglang/srt/layers/attention/dsa_backend.py`
  - `python/sglang/srt/layers/attention/dsv4/indexer.py`
  - `python/sglang/srt/disaggregation/decode.py`
  - `sgl-kernel/python/sgl_kernel/top_k.py`
- OP-TRT implementation:
  - `tensorrt_llm/_torch/attention_backend/sparse/dsa.py`
  - `tensorrt_llm/_torch/attention_backend/sparse/kvarn_backend.py`
  - `tensorrt_llm/_torch/pyexecutor/kv_cache_transceiver.py`
  - `tensorrt_llm/_torch/disaggregation/transceiver.py`
  - `tensorrt_llm/_torch/disaggregation/resource/kv_extractor.py`
  - `tensorrt_llm/_torch/disaggregation/resource/utils.py`
- Dynamo/NIXL write-mode docs:
  - `docs/api/nixl-connect/writable-operation.md`
  - `docs/api/nixl-connect/write-operation.md`

## Executive Decision

Do not copy SGLang HiSparse wholesale. Build an OP-TRT HiSparse subsystem that
uses SGLang's proven architecture:

1. full logical KV in host-pinned memory;
2. fixed-size hot KV buffer in GPU memory;
3. top-k-driven swap-in with hit/miss/LRU in CUDA;
4. direct-to-host PD transfer through write-mode metadata;
5. eager backup of newly produced decode KV.

But adapt it around OP-TRT's custom contracts:

1. Indexer/HISA scoring remains the source of truth;
2. Indexer K remains resident `fp4` and is not KVarN;
3. dense MLA latent KV remains production `kvarn_k2v2`;
4. host/hot ownership must be block-oriented, because KVarN restore,
   LayerSplit broadcasts, NIXL page tables, and sparse MLA all key on paged
   block ids;
5. LayerSplit owner-local prefill and TP4/CP1 decode must remain valid;
6. SMC-SD/Moondream decode must not reuse host slots before speculative cleanup;
7. no silent fallback to non-custom paths is allowed.

## Relevant SGLang Facts

SGLang HiSparse is decode-side hierarchical memory for DSA/DSv4 models. The
guide states that prefill is transparent, decode keeps only a small hot device
buffer, and the complete KV lives in CPU pinned memory. In PD mode, prefill
writes KV directly to decode host memory via RDMA. For DeepSeek V4, SGLang
writes only C4 KV to host and keeps the indexer/C128 path device-to-device.

Implementation details worth preserving:

- `HiSparseTokenToKVPoolAllocator` separates logical capacity from hot device
  capacity. `alloc_logical_only()` is used by direct-to-host transfer.
- `HiSparseCoordinator` owns request-to-host rows, request hot buffers,
  `full_to_hisparse_device_index_mapping`, LRU slots, raw top-k capture buffer,
  graph-safe output buffers, staging queues, eager backup stream, and cleanup.
- `swap_in_selected_pages()` launches one CUDA kernel per layer and returns
  device locations for attention.
- The CUDA kernel has a short-sequence fast path, newest-token reserved slot,
  shared-memory top-k hash, LRU hit/miss compaction, host-to-device miss copy,
  and `num_real_reqs` early exit for padded CUDA-graph batches.
- DSv4 top-k captures raw request-relative token positions separately from
  physical page-table locations so the swap-in path can target logical host
  rows.
- SGLang requires decode radix cache disabled with HiSparse.

## OP-TRT Facts That Change The Design

Current production r20 already enables:

- NIXL Python/native generation-first handoff;
- LayerSplit TP2xCP2 owner-local prefill;
- TP4/EP4 decode with `enable_attention_dp=true`;
- dense MLA latent KVarN `kvarn_k2v2` plus amortized restore;
- FP4 Indexer K;
- `indexcache-hisa`, HISA page reps/counts, FSSS reuse, cross-step IndexCache;
- CuTe/C++ top-k and paged MQA logits;
- WarpDecode forced on decode;
- SMC-SD with GLM draft path.

The clean OP-TRT insertion point is in
`DSATrtllmAttention.sparse_attn_predict()` after `Indexer.forward()` has filled
local request-relative top-k and before local top-k is converted to global pool
indices. The current transform path is:

1. `Indexer.sparse_attn_indexer()` produces `topk_indices_buffer` as
   request-relative token positions.
2. `transform_local_topk_reuse_or_compute()` either converts those positions to
   global pool indices or applies FSSS affine reuse.
3. `sparse_attn_predict()` uses global indices to trigger LayerSplit dense-KV
   read-set broadcast, then sparse MLA reads the global indices.

HiSparse should hook between steps 1 and 2 for owning F layers, and must provide
equivalent cached output for reuse S layers.

## Architecture

Add an `OPTRTHiSparseCoordinator` owned by `DSACacheManager` and exposed through
`DSAtrtllmAttentionMetadata`.

The coordinator owns:

- logical request rows:
  - `req_to_logical_blocks[req_pool_idx, block_pos] -> logical/full block id`
  - `req_to_host_blocks[req_pool_idx, block_pos] -> host block slot`
  - `req_to_hot_blocks[req_pool_idx, slot] -> hot device block slot`
- per-layer hot state:
  - `hot_block_tokens_or_block_pos[layer, req_pool_idx, slot]`
  - `hot_block_locs[layer, req_pool_idx, slot]`
  - `lru_slots[layer, req_pool_idx, hot_slot]`
- KVarN state:
  - host KVarN packed blocks;
  - host mirrors for `valid`, `commit_gen`, and restored epochs;
  - newest/tail fp16 blocks that are not committed yet;
- graph-safe buffers:
  - `hot_global_indices_buffer[B * next_n, index_topk]`;
  - `hot_block_ids_buffer[B * next_n, ceil(index_topk / tokens_per_block)]`;
  - `num_real_rows`;
  - optional `miss_count`, `hit_count`, and debug counters;
- lifecycle:
  - staging admission;
  - direct-to-host admission;
  - eager backup after decode;
  - abort/retract cleanup;
  - block recycle invalidation;
  - request-pin cleanup integration.

The hot buffer should be sized in blocks, not tokens:

```text
index_topk = 1024 tokens
tokens_per_block = 64
selected blocks/request/layer <= 16, usually fewer after block dedupe

candidate defaults:
  hisparse_hot_blocks_per_req = 64
  hisparse_hot_tokens_equivalent = 4096
  A/B: 32, 64, 96, 128 hot blocks/request
```

This block-level design matches OP-TRT's sparse MLA global-index format,
LayerSplit read-set broadcast, KVarN commit/restore, and NIXL descriptors. It
also reduces LRU pressure because many top-k tokens share a block.

## Config Surface

Add these fields under `sparse_attention_config`:

```yaml
hisparse_enabled: false
hisparse_mode: dense_mla_kvarn
hisparse_direct_to_host: true
hisparse_indexer_host_tier: false
hisparse_topk: 1024
hisparse_hot_blocks_per_req: 64
hisparse_host_to_device_ratio: 8
hisparse_min_seq_len: 65536
hisparse_block_lru: true
hisparse_eager_backup: true
hisparse_fail_closed: true
```

Validation:

- `hisparse_topk` must equal `index_topk` unless an explicit override is tested.
- `hisparse_mode=dense_mla_kvarn` requires `mla_latent_kv_dtype=kvarn_k2v2` or
  another supported dense MLA KVarN dtype.
- `hisparse_indexer_host_tier=false` in the first production candidate.
- decode `kv_cache_config.enable_block_reuse` must remain false unless a
  HiSparse-aware reuse adapter is implemented.
- direct-to-host requires `cache_transceiver_config.backend=NIXL` and
  `transceiver_runtime=PYTHON`.
- if HiSparse setup fails and `hisparse_fail_closed=true`, startup should fail
  rather than silently using full-HBM sparse attention.

## Host Pool Layout

Use a dense-MLA host pool parallel to KVarN side-pool format, not a plain fp16
pool as the optimized end state.

For each local layer and physical block:

```text
host_kvarn_ckv_packed[block_id]
host_kvarn_kpe_packed[block_id]
host_kvarn_meta[block_id]
host_block_valid[block_id]
host_commit_gen[block_id]
host_owner_request_epoch[block_id]
```

Tail/sink policy:

- sink blocks stay resident or are preloaded into the hot buffer;
- the in-progress tail block stays fp16 until it becomes full;
- once full, it is committed to KVarN host storage and invalidates any stale
  hot/restored epoch.

Bring-up can include an explicit non-production `hisparse_host_format=fp16`
oracle to compare byte-for-byte against sparse MLA. Production should target
packed KVarN host storage plus BDR/on-read dequant.

## Swap-In Kernel

Port the SGLang algorithm into OP-TRT as a CUDA/C++ op, but make it block-based:

Inputs:

```text
local_topk_tokens: int32 [rows, index_topk]
block_table: int32 [seqs, max_blocks]
req_idx_per_row: int32 [rows]
kv_lens: int32/int64 [seqs]
req_pool_indices: int32 [seqs]
req_to_host_blocks: int64 [req_slots, max_blocks]
hot_block_tokens: int32 [layers, req_slots, hot_blocks]
hot_block_locs: int32 [layers, req_slots, hot_blocks]
lru_slots: int16 [layers, req_slots, hot_blocks]
host_kvarn_pool pointers
device_hot_pool pointers
num_real_rows: int32[1]
```

Outputs:

```text
hot_global_indices: int32 [rows, index_topk]
hot_block_ids: int32 [rows, max_selected_blocks]
hit/miss stats optional
```

Algorithm:

1. One CTA per row or per `(row, layer)` depending measured occupancy.
2. Convert each selected local token to its logical block position:
   `block_pos = token // tokens_per_block`, `offset = token % tokens_per_block`.
3. Deduplicate selected block positions in shared memory.
4. Short path: if `seq_len <= hot_blocks_per_req * tokens_per_block`, map tokens
   directly to preloaded hot blocks.
5. Long path:
   - hash selected block positions;
   - scan LRU slots for hits;
   - assign misses to evictable slots;
   - update LRU order;
   - copy host KVarN packed block to hot KVarN block slot;
   - update `hot_global_indices` so sparse MLA reads from hot physical block ids
     plus original token offset.
6. Latest token:
   - reserve a hot tail slot like SGLang's newest-token slot;
   - update it from the decode append path;
   - do not evict it until committed/backed up.
7. CUDA graph:
   - no allocations;
   - no host reads;
   - `num_real_rows` guards padded graph rows;
   - fixed buckets for hot block count and top-k.

Initial kernels to compile:

```text
SM100, index_topk=1024, tokens_per_block=64
hot_blocks_per_req in {32, 64, 96, 128}
row shapes: B in graph buckets, next_n in {1, 2, 3, 4, 1 + gamma}
```

## KVarN Integration

Current KVarN restores committed dense MLA blocks into the fp16 main pool before
decode. HiSparse should change this into two tiers:

1. cold host tier: packed KVarN records for full committed blocks;
2. hot device tier: packed KVarN records or restored fp16 blocks for selected
   blocks.

The optimized target is:

- host-to-hot copies packed KVarN records;
- sparse MLA reads through a hot-pool view;
- BDR/in-kernel dequant-on-read handles selected hot blocks;
- `commit_gen` and `restored_gen` remain block-id keyed;
- `KVarNLatentPool.invalidate_blocks()` is called for host and hot tiers when
  `free_resources()` or `rewind_kv_cache()` recycles a block id.

Implementation sequence:

1. Phase A oracle: host/hot fp16 block copy, sparse MLA output equivalence.
2. Phase B packed KVarN host/hot copy, explicit dequant into hot fp16 before
   sparse MLA.
3. Phase C packed KVarN hot read with BDR/on-read dequant, no fp16 staging for
   committed blocks.

Only Phase C should be considered optimized.

## Indexer And HISA Integration

HiSparse must not perturb scoring.

Keep these paths unchanged:

- FP4 Indexer K cache;
- HISA candidate page reps/counts;
- HISA min-seq gate;
- CuTe/C++ top-k dispatch policy;
- FSSS `index_topk_freq`;
- cross-step IndexCache reuse;
- short-sequence indexer skip.

Add:

- `metadata.hisparse_coordinator`;
- `metadata.hisparse_local_topk_cache`;
- `metadata.hisparse_hot_global_idx_cache`;
- `metadata.hisparse_hot_block_ids_cache`;
- per-step cache invalidation in `prepare()`, `on_update_kv_lens()`, and
  `update_for_spec_dec()`.

F layer:

1. consume local request-relative top-k from Indexer/HISA;
2. run block swap-in;
3. produce hot global indices;
4. cache hot global indices for S layers.

S layer:

1. skip scoring as today;
2. reuse cached selected local tokens or hot global indices;
3. apply layer offset only if hot block layout is layer-contiguous in the same
   way as the full pool. If each layer has independent hot block slots, reuse
   must rerun only the cheap mapping for that layer, not the top-k scoring.

The second option is safer: keep FSSS scoring reuse, but run the per-layer
swap-in/mapping because each layer's hot residency differs.

## LayerSplit Composition

Prefill:

- LayerSplit owner-local remains on.
- Owner CP ranks own full Indexer K and dense/KVarN blocks for their layers.
- Direct-to-host NIXL writes must publish global layer metadata, preserving the
  existing owner-local transfer fix.
- All CP ranks continue participating until partial-rank transfer is explicit.

Decode:

- Current r20 decode has CP1, so HiSparse decode can be local per TP rank.
- If CP decode is enabled later, swap-in should happen on the layer owner and
  the selected hot blocks should be broadcast to peer CP ranks.

Important ordering:

1. Indexer K LayerSplit broadcast still happens before scoring.
2. HiSparse swap-in happens after top-k selection.
3. LayerSplit dense-KV broadcast must use the hot selected block ids, or be
   bypassed when decode CP1 owns the hot pool.
4. Sparse MLA consumes hot global indices.

## NIXL Direct-To-Host

Use Dynamo/NIXL write-mode semantics:

- decode creates writable descriptors for host-pinned HiSparse pool regions;
- decode publishes `RdmaMetadata` through the existing generation-first
  `ctx_info_endpoint`/request pin path;
- prefill creates write operations using local descriptors and remote writable
  metadata;
- transfer begins immediately and decode admits the request when the write
  completes.

OP-TRT changes:

1. Extend `KVRegionExtractorV1` or add `HiSparseRegionExtractor` so the transfer
   worker can describe host-pinned HiSparse pools, not just GPU KV pools.
2. Extend `get_unique_pool_memory_descs()` to distinguish device memory from
   host-pinned memory. The current tuple `(ptr, size, device_id, name)` is not
   expressive enough if the native NIXL wrapper needs memory type.
3. Add request-level host slot allocation before `prepare_context_requests()`
   promotes a generation-first context request.
4. Include host block rows in aux metadata so prefill writes exact destination
   offsets.
5. Keep cancel behavior strict: if any NIXL task is mid-write, do not free host
   or hot slots until `cancel_request()` reports safe.

Fallback policy:

- Direct-to-host failure should fail closed for production.
- A staging path can exist as a debug mode, but must require
  `hisparse_allow_staging_debug=true`.

## SMC-SD And Moondream Decode

SMC-SD changes the row geometry. HiSparse must treat speculative rows as query
rows over the same request host table.

Requirements:

- `num_real_rows` is `num_generations * next_n` or the expanded MTP row count.
- `req_idx_per_row` maps every speculative row back to the base request.
- selected top-k token positions are request-relative, not draft-row-relative.
- newest/tail backup is committed-token aware:
  - back up accepted target tokens;
  - do not commit rejected draft branches into the host table;
  - keep draft model GQA KVarN separate and fail-closed until proven.
- Moondream pinning markers must remain tied to the same `disagg_request_id`,
  `ctx_dp_rank`, and `ctx_info_endpoint`.

Do not gate first HiSparse implementation on GQA KVarN. Dense MLA target
HiSparse is the priority path.

## Scheduling And Admission

Admission should be governed by separate capacities:

```text
logical_host_capacity_blocks
hot_device_capacity_blocks
request_slot_capacity
metadata_buffer_capacity
NIXL writable-session capacity
```

For a new decode request:

1. reserve request slot;
2. allocate logical host rows for all prompt blocks;
3. allocate or reserve hot blocks for short-sequence preload/newest slot;
4. publish writable host descriptors through request pin metadata;
5. prefill writes host KVarN blocks;
6. decode admits request and preloads short sequences if needed;
7. first decode step skips eager backup for prefill tokens already in host.

Retraction:

- wait for pending backup;
- cancel NIXL session;
- if mid-write, defer free;
- clear hot maps;
- free hot slots;
- invalidate KVarN host/hot records for recycled blocks;
- free host slots;
- clear request pin metadata.

## Tests

Unit tests:

- allocator separates logical host and hot device capacity;
- host slot allocation and cleanup restore all counters;
- direct-to-host admission does not allocate full GPU KV;
- short sequence preload maps exact token offsets;
- long sequence hit/miss/LRU against a naive oracle;
- newest-token reserved slot;
- duplicate top-k tokens and duplicate blocks;
- padded CUDA graph rows via `num_real_rows`;
- abort while staging;
- abort while NIXL write is transferring;
- request recycle invalidates KVarN host/hot records;
- FSSS S layers reuse scoring but still map per-layer hot slots.

Correctness tests:

- Indexer selected SET unchanged with HiSparse off/on.
- HISA candidate selection unchanged.
- sparse MLA output equal to no-HiSparse fp16 oracle in Phase A.
- sparse MLA output within KVarN quant tolerance in Phase B/C.
- KVarN full restore vs HiSparse hot restore block equivalence.
- LayerSplit TP2xCP2 prefill to TP4/CP1 decode E2E.
- generation-first NIXL direct-to-host E2E.
- streaming cancel/cleanup E2E.
- SMC-SD speculative rows with accept/reject cleanup.

Performance tests:

- swap-in microbench:
  - hit rate sweep;
  - hot blocks per request sweep;
  - top-k 1024 and 2048;
  - seq len 1k, 4k, 16k, 32k, 64k, 128k;
  - B 1, 4, 8, 16, 32, 64.
- NIXL direct-to-host:
  - host registration cost;
  - descriptor coalescing on/off;
  - LIBFABRIC vs UCX plugin;
  - CPU NUMA placement for host pool.
- end-to-end:
  - target concurrency 16;
  - input lengths 1k to 128k;
  - compare full-HBM sparse attention, KVarN only, HiSparse fp16 oracle,
    HiSparse packed KVarN, and HiSparse direct-to-host.

## Implementation Phases

### Phase 0: Branch and Docs

- Work branch: `op-trt-hisparse`.
- Keep production r20 manifests unchanged until proof gates pass.
- Add this plan and keep a running implementation checklist.

### Phase 1: Metadata And Allocator Skeleton

Files:

- `tensorrt_llm/_torch/attention_backend/sparse/dsa.py`
- new `tensorrt_llm/_torch/attention_backend/sparse/hisparse.py`
- `tensorrt_llm/llmapi/llm_args.py` or relevant sparse config parser

Deliverables:

- config parse/validation;
- coordinator object creation;
- graph-safe buffers;
- request lifecycle hooks;
- no-op disabled path;
- fail-closed startup validation.

### Phase 2: FP16 Oracle Hot Buffer

Deliverables:

- block-level host fp16 pool;
- block-level hot fp16 pool;
- Python/torch naive swap-in oracle;
- C++/CUDA swap-in kernel for fp16 block copy;
- sparse MLA consumes hot global indices;
- unit tests prove equivalence.

### Phase 3: KVarN Host/Hot Path

Deliverables:

- host packed KVarN block storage;
- hot packed KVarN block storage;
- explicit dequant-to-hot-fp16 transitional path;
- KVarN record invalidation on free/rewind;
- packed host/hot tests.

### Phase 4: BDR/On-Read Optimized Path

Deliverables:

- sparse MLA hot read supports packed KVarN records;
- BDR fold/on-read dequant;
- no full-working-set fp16 restore for committed cold blocks;
- performance proof that KVarN+HiSparse beats KVarN-only at long context and
  concurrency 16.

### Phase 5: NIXL Direct-To-Host

Files:

- `tensorrt_llm/_torch/disaggregation/resource/kv_extractor.py`
- `tensorrt_llm/_torch/disaggregation/resource/utils.py`
- `tensorrt_llm/_torch/disaggregation/transceiver.py`
- native NIXL wrapper if memory-type descriptors are required
- Dynamo router/request-pin docs/tests if sideband metadata expands

Deliverables:

- host-pinned descriptors;
- decode writable metadata publish;
- prefill write operation into decode host pool;
- generation-first pin proof;
- cancel safety.

### Phase 6: LayerSplit, SMC, Moondream Hardening

Deliverables:

- owner-local LayerSplit direct-to-host proof;
- decode CP1 proof;
- CP>1 design guard or fail-closed validation;
- SMC row mapping;
- Moondream pin preservation;
- no draft rejected-token host pollution.

### Phase 7: A/B And Promotion

Promotion candidate:

```yaml
sparse_attention_config:
  hisparse_enabled: true
  hisparse_mode: dense_mla_kvarn
  hisparse_direct_to_host: true
  hisparse_indexer_host_tier: false
  hisparse_topk: 1024
  hisparse_hot_blocks_per_req: 64
  hisparse_host_to_device_ratio: 8
  hisparse_min_seq_len: 65536
  hisparse_fail_closed: true
```

A/B matrix:

- hot blocks/request: 32, 64, 96, 128;
- host/device ratio: 5, 8, 10;
- NIXL plugin: LIBFABRIC, UCX;
- direct-to-host on/off;
- KVarN packed on-read vs explicit hot fp16 restore;
- TP4 vs alternate TP/EP settings;
- `free_gpu_memory_fraction` decode sweep;
- SMC on/off;
- Moondream overlap on/off.

Promotion requires:

- no correctness regression;
- no fallback logs;
- no leaked request pins;
- no leaked host/hot blocks;
- no NIXL mid-write free;
- tokens/second/user improvement at concurrency 16 for long context;
- no short-context regression large enough to lower the aggregate target.

## Open Risks

1. NIXL host-pinned memory registration may need native wrapper changes because
   current memory descriptors do not encode memory kind.
2. Packed KVarN sparse MLA read may need a dedicated hot-pool ABI instead of
   reusing the existing main-pool pointer layout.
3. FSSS reuse layers cannot blindly affine-shift hot global indices if each
   layer has independent hot slots. Safer first implementation reruns per-layer
   swap-in using reused local top-k.
4. SMC rejected draft branches must never be backed up to host as committed
   target KV.
5. HiSparse benefits appear mostly under high concurrency and long context.
   `hisparse_min_seq_len` should prevent low-concurrency/short-context overhead
   from hurting the default path.

## First Code Change To Make

Start with Phase 1 and Phase 2 only:

1. add config fields and validation;
2. add coordinator skeleton and disabled no-op path;
3. implement fp16 block-hot oracle;
4. hook `sparse_attn_predict()` so local top-k maps through the coordinator;
5. prove output equivalence before touching KVarN/NIXL.

That gives a correctness rail before optimizing the hot path.

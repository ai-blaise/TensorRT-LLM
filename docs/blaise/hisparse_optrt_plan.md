# OP-TRT HiSparse Integration Plan

This document is the second-pass integration plan for building a HiSparse-style
hierarchical sparse-attention KV path on top of the current `op-trt` custom
stack. It is based on direct review of:

- LMSYS/SGLang HiSparse blog:
  https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/
- SGLang HiSparse guide:
  https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/hisparse_guide.md
- SGLang implementation, re-checked against `sgl-project/sglang` main
  `eb18416` on June 13, 2026:
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

The full implementation path must be based on the target model's production
architecture from the first executable serving path. That means dense MLA
`kvarn_k2v2` cold/hot storage, FP4 Indexer K + HISA scoring, sparse MLA with
BDR/on-read dequant, NIXL generation-first direct-to-host, and the r20
LayerSplit/SMC/Moondream wiring. Do not implement FP16 host/hot tiers in
serving code. Independent test fixtures may compare against reference tensors
outside the HiSparse coordinator/transceiver path, but there is no FP16
block-hot oracle in the serving implementation, runtime fallback, config mode,
or deployment candidate.

Current branch posture after the June 13 final correctness sweep: the branch
has a production-shaped, fail-closed partial implementation, not a deployable
HiSparse serving candidate. The config validation, packed KVarN tier
allocation, host metadata publication, NIXL DRAM registration, request
host-slot sideband, packed KVarN source/destination fragment derivation, typed
`HISPARSE_HOST` write submission, decode admission state, and two-stage
hot-block planning ABI are implemented. A device-visible request table now
publishes disaggregated request ids, request-relative block-to-host-slot rows,
host commit generations, and admission flags for the future native hot-slot
planner. This table is the required production ABI shape, but its current
Python lifecycle writer is not the final serving publication mechanism; before
the startup guard is relaxed, request-table updates must be batched,
stream-ordered, and native or async-copy driven so decode does not pay
per-block Python synchronization cost. The sender now returns explicit
`(local_layer, request_block_pos)` commit coverage only after the normal KV
write and typed host write both succeed, and the receiver accumulates that
coverage before marking host records committed. Admission is explicit: a
request cannot be marked HiSparse-ready unless all reserved prompt host blocks
are committed and no host writes are pending. Host-to-hot planning is also
explicit: the coordinator may plan packed KVarN miss copies, but it does not
publish hot residency until the planned native copy is accepted. A strict
native thop now exists for packed KVarN host-to-hot copies, and a native
CUDA-side TopK-to-block dedupe primitive now exists for the first planner stage.
A native request-table resolver now maps device block rows and row request ids
to committed host slots and commit generations, returning explicit invalid
status flags for missing, unadmitted, out-of-range, or uncommitted rows rather
than falling back to Python request-table extraction. A native non-mutating
hot-slot planner now consumes those resolved rows and layer-local hot metadata
to produce hit/miss/LRU slot decisions, copy schedules, and row status without
publishing residency before packed copies succeed. A native compact miss
schedule op now turns row-major device miss tensors into contiguous device
`(host_slot, hot_slot, row_id)` vectors plus a device copy count. A native
`trtllm::hisparse_submit_packed_kvarn_copy_schedule` bridge now consumes that
compact device schedule directly and copies from mapped pinned host KVarN
storage into hot HBM in stream order, returning per-row copy status for the
post-copy commit stage. It intentionally fails closed if the host tier is not
device-addressable; it does not use a CUDA host callback to enqueue copies and
does not synchronously read the schedule back to Python. A native post-copy hot
metadata commit op now publishes `hot_host_slot`, `hot_commit_gen`, and
`hot_lru_tick` on device only after the packed-copy stage has accepted the plan.
The branch also has a native hot-index builder that preserves the existing
`base * stride_factor + layer_idx * tokens_per_block + token_offset` sparse-MLA
index contract while targeting HiSparse hot slots instead of full-pool blocks.
Startup and runtime mapping still intentionally reject `hisparse_enabled=true`
before serving because the full mapping orchestration, sparse MLA hot-pool
reading, BDR/on-read dequant, and live NIXL/cancel E2E proofs are not complete.
This is the correct failure mode: no manifest should get an implicit full-HBM,
FP16-staging, Python TopK extraction, or direct-to-host-off substitute.

## Final Correctness Sweep

The full implementation must remain production-architecture-first. Partial code
may exist only when it is behind fail-closed startup/mapping guards and has the
same ABI shape as the final serving path. The following are hard invariants:

- Dense MLA cold and hot storage is packed KVarN `kvarn_k2v2`, not FP16 and not
  an FP16 staging tier.
- The Indexer/HISA path stays device-resident FP4/HISA.
- Indexer K is not moved into KVarN or host HiSparse storage.
- HiSparse uses request-relative top-k token positions from the existing
  Indexer/HISA path and maps them into selected packed KVarN hot blocks before
  sparse MLA.
- The hot tier is block-oriented for OP-TRT, even though SGLang's generic DSA
  implementation is token-slot-oriented, because OP-TRT KVarN, BDR/on-read
  dequant, paged sparse MLA, LayerSplit ownership, and NIXL page metadata all
  key on paged blocks.
- Direct-to-host is the production path: prefill writes packed dense-MLA KVarN
  records into decode host-pinned slots through typed NIXL `HISPARSE_HOST`
  writes, and decode admission waits for commit coverage.
- SGLang's naive top-k loader/debug oracle is not a model for OP-TRT serving.
  Any offline references used by tests must stay outside the coordinator,
  transceiver, kernel ABI, and deployment config.
- There is no HELIX, completed-prefill staging fallback, full-HBM fallback, or
  direct-to-host-off fallback when `hisparse_enabled=true`.
- There is no FP16 block-hot oracle in enabled serving.
- There is no runtime downgrade when `hisparse_enabled=true`.
- LayerSplit owner-local prefill, TP4/EP4 decode, SMC-SD row expansion,
  Moondream pinning, request pinning, cancellation, and request recycle must
  compose with HiSparse before the startup guard is relaxed.
- MORI-IO remains an A/B candidate only; the gate path is NIXL write-mode/direct
  host writes.
- Promotion requires live VM proof and A/B data, not just unit tests.
- The miss-copy boundary must be explicit. CUDA kernels must not pretend that
  CPU pinned host KVarN storage is ordinary device memory. The production path
  dedupes and plans misses on device, then hands a compact miss schedule to a
  native stream-ordered copy bridge and only then commits hot metadata.
  Python-side token extraction, Python request-table extraction, synchronous
  schedule reads, and CUDA host callbacks that enqueue CUDA work are not valid
  serving paths. The current bridge is a mapped pinned-host kernel path; a
  future copy-engine variant may replace it only if it consumes the same compact
  device schedule without host synchronization.

CUDA API note: NVIDIA documents `cudaMemcpyBatchAsync()` as a host API over
host-visible source pointer, destination pointer, and size arrays, and documents
that `cudaLaunchHostFunc()` callbacks must not make CUDA API calls. Therefore a
callback that waits for device schedule compaction and then enqueues copy-engine
work would be invalid, and a schedule readback before copy submission would
violate the no-sync production path. CUDA also documents mapped registered host
memory as device-addressable when the device supports
`cudaDevAttrCanUseHostPointerForRegisteredMem`; the current bridge uses exactly
that stream-ordered mapped-host path and fails closed otherwise.

## Relevant SGLang Facts

SGLang HiSparse is decode-side hierarchical memory for DSA/DSv4 models. The
guide states that prefill model execution is transparent, decode keeps only a
small hot device buffer, and the complete KV lives in CPU pinned memory. In PD
mode, prefill writes KV directly to decode host memory via RDMA. For DeepSeek
V4, SGLang writes only C4 KV to host and keeps the indexer/C128 path
device-to-device. In OP-TRT, "prefill transparent" means no target-model compute
detour: the prefill transceiver still has to expose/write NIXL host descriptors
for the production packed KVarN host tier.

Implementation details worth preserving or adapting:

- `HiSparseTokenToKVPoolAllocator` separates logical capacity from hot device
  capacity. `alloc_logical_only()` is used by direct-to-host transfer.
- `HiSparseCoordinator` owns request-to-host rows, request hot buffers,
  `full_to_hisparse_device_index_mapping`, LRU slots, raw top-k capture buffer,
  graph-safe output buffers, request-admission queues, eager backup stream, and
  cleanup.
- `swap_in_selected_pages()`/`load_cache_to_device_buffer_*` launches one CUDA
  kernel per layer and returns device locations for attention.
- The CUDA kernel has a short-sequence fast path, newest-token reserved slot,
  shared-memory top-k hash, LRU hit/miss compaction, host-to-device miss copy,
  and `num_real_reqs` early exit for padded CUDA-graph batches.
- DSv4 top-k captures raw request-relative token positions separately from
  physical page-table locations so the swap-in path can target logical host
  rows.
- SGLang validates model/backend constraints, requires radix cache disabled for
  HiSparse, and pairs DSA backends with the selected KV dtype.

Implementation details not to copy directly:

- SGLang's generic DSA hot buffer is token-slot based; OP-TRT's first serving
  candidate must be block-based packed KVarN.
- SGLang's BF16/FP8 FlashMLA hot tier is not OP-TRT's dense MLA KVarN hot tier.
- SGLang's staging and naive debug loader are useful for understanding
  correctness, but they are not acceptable OP-TRT deployment modes.
- DeepSeek V4 C4 layout handling is relevant as an example of architecture
  specialization, not as the OP-TRT target path for the current dense-MLA model.

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

Target-model assumption for this branch: the serving path is the current
production DSA/dense-MLA target path, with `index_topk=1024`,
`tokens_per_block=64`, FP4 Indexer K/HISA, dense MLA latent KVarN
`kvarn_k2v2`, sparse MLA decode, NIXL Python/native generation-first handoff,
LayerSplit owner-local prefill, and TP4/EP4 decode. Draft-model GQA KVarN is a
separate fail-closed path and should not determine the first HiSparse design.

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
hisparse_enabled: false  # schema default while gated; promotion candidate sets true
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
- `hisparse_enabled=true` production manifests must set
  `hisparse_direct_to_host=true`; completed-prefill/full-HBM variants are
  separate baselines, not an enabled-HiSparse fallback.
- if HiSparse setup fails and `hisparse_fail_closed=true`, startup should fail
  rather than silently using full-HBM sparse attention.
- startup validation should reject FP16 host/hot serving tiers, direct-to-host
  disabled with HiSparse enabled, Indexer K KVarN selection, or any staging
  fallback marker in production manifests.

## Host Pool Layout

Use a dense-MLA host pool parallel to KVarN side-pool format. The serving path
must never allocate a plain FP16 host/hot tier for committed blocks.

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

Correctness comparisons may use offline reference buffers or the existing
full-HBM production KVarN path, but no `hisparse_host_format=fp16` serving mode
should be added. Production targets packed KVarN host storage plus BDR/on-read
dequant.

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
   - emit a compact miss schedule for packed host KVarN block to hot KVarN
     block copies;
   - submit that schedule to a native stream-ordered copy bridge that uses
     host-to-device copy-engine transfers from pinned DRAM to hot HBM;
   - commit hot metadata only after copy submission succeeds;
   - update `hot_global_indices` so sparse MLA reads from hot physical block ids
     plus original token offset.
6. Latest token:
   - reserve a hot tail slot like SGLang's newest-token slot;
   - update it from the decode append path;
   - do not evict it until committed/backed up.
7. CUDA graph:
   - no allocations;
   - no Python token/request-table reads;
   - no synchronous host schedule readback;
   - no CUDA-kernel dereference of CPU pinned KVarN storage;
   - `num_real_rows` guards padded graph rows;
   - fixed buckets for hot block count and top-k.

Initial kernels to compile:

```text
SM100, index_topk=1024, tokens_per_block=64
hot_blocks_per_req in {32, 64, 96, 128}
row shapes: B in graph buckets, next_n in {1, 2, 3, 4, 1 + gamma}
```

## KVarN Integration

Current KVarN can restore committed dense MLA blocks into the FP16 main pool
before decode. HiSparse should move the production serving path to two packed
tiers:

1. cold host tier: packed KVarN records for full committed blocks;
2. hot device tier: packed KVarN records for selected committed blocks, plus
   the already-required resident FP16 sink/tail blocks that have not been
   committed yet.

The optimized target is:

- host-to-hot copies packed KVarN records;
- sparse MLA reads through a hot-pool view;
- BDR/in-kernel dequant-on-read handles selected hot blocks;
- `commit_gen` and `restored_gen` remain block-id keyed;
- `KVarNLatentPool.invalidate_blocks()` is called for host and hot tiers when
  `free_resources()` or `rewind_kv_cache()` recycles a block id.

Implementation sequence:

1. Install production packed host/hot KVarN allocation and metadata first.
2. Implement packed KVarN host-to-hot swap-in and hot global-index mapping.
3. Make sparse MLA consume the hot packed KVarN view through BDR/on-read dequant.
4. Keep external FP16/KVarN references in tests only; do not add a serving
   staging path that dequants committed cold blocks into a hot FP16 pool.

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

1. Publish HiSparse host-pinned pool metadata through `RankInfo` separately
   from the existing GPU page-table metadata.
2. Register HiSparse host-pinned pools with the transfer agent as `DRAM`
   descriptors, not as part of the existing `VRAM` KV-cache descriptor set.
3. Add a dedicated HiSparse host-write meta path, or another explicit
   `DRAM`-typed write batch, before scheduling prefill writes into host slots.
   The current `WriteMetaType.KV` path is VRAM-only and must not silently mix
   host-pinned descriptors into a GPU KV transfer.
4. Add request-level host slot allocation before `prepare_context_requests()`
   promotes a generation-first context request.
5. Include host block rows in request-pin/sideband metadata so prefill writes
   exact destination offsets.
6. Keep cancel behavior strict: if any NIXL task is mid-write, do not free host
   or hot slots until `cancel_request()` reports safe.

Fallback policy:

- Direct-to-host failure should fail closed for production.
- No staging/debug fallback should exist in the serving path. Unit tests may
  inject synthetic host-pool contents directly, but deployment config should
  expose only the production NIXL direct-to-host path.

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
- long sequence hit/miss/LRU against a pure reference model of the hot-block
  state machine;
- newest-token reserved slot;
- duplicate top-k tokens and duplicate blocks;
- padded CUDA graph rows via `num_real_rows`;
- abort while NIXL write is transferring;
- request recycle invalidates KVarN host/hot records;
- FSSS S layers reuse scoring but still map per-layer hot slots.

Correctness tests:

- Indexer selected SET unchanged with HiSparse off/on.
- HISA candidate selection unchanged.
- sparse MLA output within KVarN quant tolerance versus the existing production
  full-HBM KVarN path.
- KVarN full restore vs HiSparse hot restore block equivalence.
- independent offline references may be used only as test fixtures; no FP16
  block-hot oracle path may be wired into the coordinator, transceiver, kernel
  ABI, or deployment config.
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
  - compare full-HBM sparse attention, production KVarN only, HiSparse packed
    KVarN, and HiSparse packed KVarN with direct-to-host.

## Production Workstreams And Gates

These labels describe ordered workstreams and proof gates. They are not
deployable states. Any incomplete workstream remains fail-closed and must not
become a serving candidate until all promotion gates pass.

### Gate 0: Branch And Docs

- Work branch: `op-trt-hisparse`.
- Keep production r20 manifests unchanged until proof gates pass.
- Add this plan and keep a running implementation checklist.

### Gate 1: Metadata And Allocator

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

Current branch status:

- implemented config fields and sparse-config validation for
  `hisparse_enabled`, `hisparse_mode`, direct-to-host, Indexer device
  residency, KVarN dense MLA storage, hot-block sizing, eager backup, and
  fail-closed policy;
- implemented outer runtime validation requiring
  `cache_transceiver_config.backend="NIXL"`,
  `transceiver_runtime="PYTHON"`, and
  `kv_cache_config.enable_block_reuse=false`;
- added `OPTRTHiSparseCoordinator` as the DSA-owned extension point;
- wired coordinator ownership into `DSACacheManager`, per-step metadata reset,
  and the `sparse_attn_predict()` TopK mapping seam;
- disabled HiSparse remains a no-op and preserves current behavior;
- enabled HiSparse intentionally raises before serving until packed KVarN
  host/hot allocation, NIXL commit, host-to-hot swap-in, and sparse MLA hot-read
  support are all complete and live-validated.

### Gate 2: Production Packed KVarN Cold/Hot Tiers

Deliverables:

- host-pinned packed KVarN cold pool for committed dense MLA blocks;
- hot device packed KVarN pool for selected committed blocks;
- resident sink/tail policy for the uncommitted FP16 blocks already required by
  dense MLA KVarN;
- host/hot `valid`, `commit_gen`, request epoch, and recycle invalidation;
- sparse-attention metadata exposes hot block tables without changing Indexer
  scoring.

Current branch status:

- implemented the coordinator's packed-tier descriptor for
  `num_layers`, `tokens_per_block`, `packed_bytes_per_block`,
  logical host capacity, and hot device capacity;
- implemented request host-row reservation and release, with stable
  request-relative `block_pos -> host_slot` ownership;
- implemented a host-pinned plus device-mirrored request table keyed by stable
  coordinator table slots:
  `request_ids`, `request_block_host_slots`, `request_block_commit_gen`, and
  `request_admitted`;
- kept that request table in the final CUDA-planner ABI shape, while requiring
  a later batched/native metadata writer before enabled serving so table
  publication is not a Python per-cell hot path;
- bounded direct `configure_packed_tiers()` request-table defaults to avoid
  quadratic host-block allocation, while production `configure_from_kv_cache_manager()`
  derives table capacity from `max_batch_size` and width from
  `max_blocks_per_seq`;
- implemented host block commit metadata with `valid`, `commit_gen`,
  `logical_block_id`, and request epoch tracking;
- implemented layer-local hot-slot metadata and LRU hit/miss selection keyed by
  `(req_pool_idx, block_pos, host_slot, commit_gen)`;
- split hot-slot handling into a non-mutating `plan_hot_blocks()` phase and a
  `commit_hot_selection()` phase so miss residency is published only after the
  future native packed KVarN copy succeeds;
- added `HiSparseSwapInPlan` and pointer helpers that produce parallel host
  DRAM pointers, hot HBM pointers, and byte sizes for packed KVarN miss blocks;
- added `trtllm::hisparse_swap_in_packed_kvarn`, a strict native thop that
  copies only packed `uint8` KVarN records from pinned host memory into the hot
  CUDA tier, coalescing consecutive slot runs when both tiers are compact;
- final-sweep caveat: the current packed-copy thop accepts CPU slot vectors and
  is therefore only a partial building block behind fail-closed guards. The
  serving path still requires a native device-plan-to-copy bridge that consumes
  `hisparse_plan_hot_slots` miss tensors without Python materialization or a
  synchronous host readback, schedules stream-ordered host-to-device copies, and
  feeds `hisparse_commit_hot_slots` only after copy submission succeeds;
- added coordinator `execute_swap_in_plan()` so native copy acceptance and hot
  metadata publication are sequenced through one production-shaped path;
- added request-relative token-position planning that dedupes top-k tokens into
  paged block positions without changing Indexer/HISA scoring;
- added `trtllm::hisparse_topk_to_block_positions`, a native CUDA shared-memory
  hash dedupe primitive that maps request-relative TopK tokens to unique
  request-relative block positions, emits device overflow flags without a host
  sync, and marks overflowed rows with an invalid block count so downstream
  native status checks reject clipped hot sets fail-closed;
- added `trtllm::hisparse_resolve_blocks_to_host_slots`, a native CUDA request
  table resolver that consumes row request ids, block rows/counts, device
  request ids, block-to-host-slot rows, commit generations, and admission flags
  to produce host slots, commit generations, per-block status, and per-row
  status without host extraction;
- added `trtllm::hisparse_plan_hot_slots`, a native non-mutating CUDA planner
  that consumes resolved host slots/commit generations plus layer-local
  `hot_host_slot`, `hot_commit_gen`, and `hot_lru_tick` metadata, protects
  slots selected earlier in the same batch, and emits planned hot slots,
  planned LRU ticks, miss host/hot copy schedules, hit flags, miss counts, and
  row status without publishing hot residency;
- added `trtllm::hisparse_compact_miss_schedule`, a native CUDA schedule
  compactor that consumes planner miss tensors and row status, validates
  upstream rows/counts/slots, and emits contiguous device `host_slot` and
  `hot_slot` vectors, row ids, and a device copy count for the copy bridge;
- added `trtllm::hisparse_submit_packed_kvarn_copy_schedule`, a native mapped
  pinned-host copy bridge that consumes the compact device schedule, copies
  packed KVarN records into the hot HBM tier in stream order, and returns
  per-row copy status so post-copy metadata commit can remain fail-closed;
- added `trtllm::hisparse_commit_hot_slots`, a native post-copy CUDA metadata
  commit op that mutates device `hot_host_slot`, `hot_commit_gen`, and
  `hot_lru_tick` only for rows whose native plan succeeded;
- added `trtllm::hisparse_build_hot_indices`, a native CUDA hot-index builder
  that remaps request-relative TopK token positions through selected
  request-relative block rows and planned hot slots into sparse-MLA-compatible
  hot global indices with explicit row status;
- synchronized device hot metadata (`hot_host_slot`, `hot_commit_gen`, and
  `hot_lru_tick`) whenever hot records are committed or cleared;
- implemented invalidation that clears hot records when host records are
  invalidated or request slots are released;
- implemented production-shaped packed tensor allocation for host `uint8`
  KVarN records, device hot `uint8` KVarN records, host commit metadata, and
  device hot-slot metadata;
- wired `DSACacheManager` so an explicitly enabled HiSparse config derives
  packed tier sizes from dense MLA KVarN, allocates the host/hot tensors, and
  then still fails closed before serving until the swap-in/read kernels exist;
- added CPU-level unit tests for allocation, duplicate reservation, capacity
  failure, uncommitted-block rejection, commit-generation refresh, LRU eviction,
  non-mutating plan/commit, admitted-request enforcement, packed pointer-plan
  ABI, tensor allocation ABI, and cleanup.

Still pending before serving enablement:

- live E2E proof that the host-write completion handoff marks host `valid` and
  `commit_gen` only after typed HiSparse host writes succeed for the relevant
  layer/block coverage;
- VM compile and live validation of the native
  `trtllm::hisparse_swap_in_packed_kvarn` CPU-schedule packed-copy helper;
- VM compile and live validation of the native
  `trtllm::hisparse_topk_to_block_positions` planner primitive;
- VM compile and live validation of the native
  `trtllm::hisparse_resolve_blocks_to_host_slots` request-table resolver;
- VM compile and live validation of the native
  `trtllm::hisparse_plan_hot_slots` non-mutating hot-slot planner;
- VM compile and live validation of the native
  `trtllm::hisparse_compact_miss_schedule` copy-schedule compactor;
- VM compile and live validation of the native
  `trtllm::hisparse_commit_hot_slots` post-copy metadata commit op;
- VM compile and live validation of the native
  `trtllm::hisparse_build_hot_indices` hot global-index builder;
- VM compile and live validation of the native device-plan-to-copy bridge,
  `trtllm::hisparse_submit_packed_kvarn_copy_schedule`, between
  `hisparse_compact_miss_schedule` and packed KVarN host-to-hot copy submission,
  including proof that the host tier is mapped/device-addressable on the B200
  deployment image;
- replacement of scalar lifecycle request-table writes with a stream-ordered
  batched/native publication path for admission, commit-generation, and cleanup
  updates;
- sparse MLA hot-pool ABI and BDR/on-read dequant hookup.

### Gate 3: Swap-In Kernel And Sparse MLA Hook

Deliverables:

- SM100 block-level swap-in kernel over packed KVarN records;
- block dedupe from local top-k token positions;
- hit/miss/LRU/newest-slot updates with graph-safe buffers;
- hot global-index output consumed by sparse MLA;
- BDR/on-read dequant for hot packed KVarN records;
- FSSS reuse layers reuse scoring but rerun per-layer hot-slot mapping when hot
  residency is layer-local.

Current branch status:

- `DSAtrtllmAttention.sparse_attn_predict()` already has the HiSparse mapping
  seam immediately after Indexer/HISA top-k production and before the existing
  full-pool index transform;
- the coordinator now exposes device-side request/admission tables that the
  next native hot-slot planner can use with device TopK block rows, without
  Python token or request-table extraction;
- attention metadata now carries `hisparse_request_ids`, keyed by
  `disagg_request_id` when present, so the decode-side planner uses the same
  request key that NIXL direct-to-host admission reserved;
- incremental update paths refresh the HiSparse request-id vector along with
  normal request ids to avoid stale admission keys under overlap/CUDA-graph
  reuse;
- runtime mapping requires configured packed tiers, allocated tensors,
  admission-compatible request ids, and the native
  `trtllm::hisparse_topk_to_block_positions`,
  `trtllm::hisparse_resolve_blocks_to_host_slots`,
  `trtllm::hisparse_plan_hot_slots`,
  `trtllm::hisparse_compact_miss_schedule`,
  `trtllm::hisparse_submit_packed_kvarn_copy_schedule`,
  `trtllm::hisparse_commit_hot_slots`, and
  `trtllm::hisparse_build_hot_indices` ops before it can proceed;
- if the native op, CUDA-side planner, or sparse MLA hot-pool read path is
  absent, mapping raises rather than falling back to the full-HBM transform.

Still pending before serving enablement:

- full mapping orchestration that chains device TopK block rows, request-table
  resolution, hot-slot planning, packed copy, and hot global-index construction
  without Python-side token or table extraction;
- full native orchestration that passes planner miss schedules into packed-copy
  scheduling without synchronous host readback, calls post-copy hot metadata
  commit, rejects stale generations, validates row status across every native
  stage, and returns the hot-index output to sparse MLA;
- live validation and microbenchmarking of native packed KVarN host-to-hot
  copy plus hot metadata update;
- hot global-index output buffers for sparse MLA;
- sparse MLA packed hot-pool read with BDR/on-read dequant;
- FSSS reuse-layer remap over layer-local hot slots.

### Gate 4: Production Optimization Hardening

Deliverables:

- precompiled SM100 variants for the production buckets;
- no full-working-set FP16 restore for committed cold blocks;
- hit/miss telemetry, hot-buffer pressure counters, and request cleanup counters;
- performance proof that KVarN+HiSparse beats KVarN-only at long context and
  concurrency 16.

### Gate 5: NIXL Direct-To-Host

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

Current branch status:

- added `HiSparseHostTierMeta` for serializing host-pinned packed KVarN tier
  pointers, per-slot item sizes, names, layer count, host-slot count, and
  packed bytes per block;
- added coordinator helpers that expose DRAM registration descriptors for
  host `uint8` packed KVarN blocks plus host validity/commit metadata;
- added coordinator helpers that compute exact writable host-packed block
  destinations for `(layer_idx, req_pool_idx, block_pos)` without changing the
  existing Indexer/HISA or sparse MLA path;
- added idempotent request host-row reservation plus request-relative
  `hisparse_host_slots` publication in receiver request metadata;
- wired receive-session cleanup so HiSparse host rows are released on the same
  safe-close boundary as existing KV receive sessions;
- added layer-major destination-fragment construction for packed KVarN host
  writes, including bounds validation against the published host-slot capacity;
- added source-fragment construction from the production dense-MLA
  `KVarNLatentPool.store` byte records, with fail-closed rejection of
  uncommitted sink/tail blocks;
- added sender-side validation that aligns source packed KVarN block fragments
  with request-relative destination host slots for dense KV-cache pool pairs,
  skipping indexer, block-scale, and non-attention pools;
- extended native transfer metadata/request construction so typed HiSparse
  writes use distinct source and destination memory types
  (`VRAM -> DRAM` or `DRAM -> DRAM`) instead of overloading uniform KV/AUX
  descriptor assumptions;
- wired packed HiSparse `VRAM -> DRAM` host writes into the KV sender path so
  the receiver is not notified of KV success until the normal KV write and the
  HiSparse host write have both completed;
- added a backward-compatible `KV_AGENT_RESULT` commit payload that carries
  paired `(local_layer, request_block_pos)` coverage for each successful typed
  HiSparse host write;
- added receiver-side coverage accumulation and coordinator commit handoff:
  replayed coverage is idempotent, partial layer coverage remains unselectable,
  and a host block becomes globally valid only after all local layers for that
  block have been written;
- added a fail-closed receiver guard so a HiSparse-enabled request that
  reserved host slots cannot complete without successful host-write commit
  coverage;
- added explicit pending-write and admission state to the coordinator:
  request host writes begin when decode publishes host slots, finish on
  terminal host-write result, block request release while writes are pending,
  and mark a request admitted only after every reserved prompt block is
  committed;
- hardened disaggregated receive cleanup so cancelled/failed sessions are not
  treated as processable while KV/HiSparse writes are still `TRANSFERRING`, and
  `RxSession.close()` refuses to release HiSparse host rows until those writes
  reach a terminal state;
- extended `RankInfo` serialization so peers can publish/consume HiSparse host
  tier metadata through the existing rank-info handshake;
- extended `TransferWorker` so allocated HiSparse host tiers are registered
  with NIXL as a separate `DRAM` registration group;
- sender-side HiSparse fragments are intentionally not appended to the normal
  `WriteMetaType.KV` `VRAM -> VRAM` request. They are carried on `WriteMeta`,
  validated against request-relative host slots, and submitted as a separate
  typed `WriteMetaType.HISPARSE_HOST` request with `VRAM -> DRAM` descriptors.

Still pending before serving enablement:

- live E2E validation of the completion/commit handoff, including multi-rank
  and partial-slice cases;
- live decode-admission validation that proves request-visible host slots are
  committed before sparse MLA can select them;
- live cancel/abort testing that proves host slots remain pinned until
  in-flight DRAM writes finish;
- E2E proof that NIXL writes land directly in decode host slots before decode
  admits the request.

### Gate 6: LayerSplit, SMC, Moondream Hardening

Deliverables:

- owner-local LayerSplit direct-to-host proof;
- decode CP1 proof;
- CP>1 design guard or fail-closed validation;
- SMC row mapping;
- Moondream pin preservation;
- no draft rejected-token host pollution.

### Gate 7: A/B And Promotion

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
- direct-to-host on for every HiSparse candidate; completed-prefill/full-HBM
  paths may be measured as separate baselines, not runtime fallbacks;
- packed KVarN hot-pool ABI and BDR/on-read kernel variants only;
- TP4 vs alternate TP/EP settings;
- `free_gpu_memory_fraction` decode sweep;
- SMC on/off;
- Moondream overlap on/off.

Promotion requires:

- no correctness regression;
- no fallback logs;
- startup logs prove the packed KVarN HiSparse production path, NIXL
  direct-to-host, BDR/on-read dequant, and Indexer K device residency are active;
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

## Immediate Execution Plan

The next implementation work should continue from the current fail-closed
production ABI:

1. Finish the SM100 packed KVarN planner/copy ABI:
   - input request-relative top-k tokens, request rows, committed host slots,
     per-layer hot metadata, and graph row count;
   - dedupe tokens to paged block positions;
   - resolve request ids through the device-mirrored request table and reject
     missing, unadmitted, or stale commit-generation rows;
   - hit/miss/LRU over block slots;
   - emit compact device miss schedules;
   - bridge those schedules into stream-ordered host-to-device copy-engine
     submissions without Python materialization or synchronous schedule readback;
   - copy only packed KVarN records from pinned host DRAM to hot HBM;
   - commit hot metadata after copy submission succeeds;
   - output hot global indices and selected hot block ids for sparse MLA.
2. Wire the sparse MLA hot-pool read path:
   - consume hot packed KVarN records directly;
   - add BDR/on-read dequant in the sparse MLA path;
   - keep sink/tail resident policy separate from committed packed blocks;
   - remove any need for full-working-set restore of committed cold blocks.
3. Prove and harden NIXL direct-to-host on live B200 VMs:
   - decode publishes writable host-pinned slots;
   - prefill writes exact packed KVarN records into those slots;
   - commit coverage is multi-rank and partial-slice safe;
   - decode admission remains blocked until all reserved prompt blocks commit.
4. Prove and harden cancellation/retraction/recycle:
   - no host or hot slot is freed while a DRAM write can still complete;
   - failed/partial writes never become selectable;
   - request recycle invalidates KVarN host/hot records and pin metadata.
5. Compose with the custom stack:
   - LayerSplit owner-local prefill and CP1 decode first, CP>1 decode guarded
     or implemented explicitly;
   - SMC-SD row geometry maps every speculative row to the base request host
     table;
   - Moondream pinning stays tied to the same `disagg_request_id`,
     `ctx_dp_rank`, and `ctx_info_endpoint`;
   - FSSS reuse keeps scoring reuse but reruns per-layer hot mapping.
6. Add production tests:
   - unit tests for block dedupe, hit/miss/LRU, commit coverage, admission,
     cancel, recycle, and FSSS reuse;
   - VM E2E for generation-first NIXL direct-to-host, LayerSplit prefill,
     TP4/EP4 decode, SMC-SD accept/reject, and Moondream pin preservation;
   - correctness comparison against the existing production full-HBM KVarN path
     within KVarN tolerance, with no FP16 serving oracle.
7. Optimize before promotion:
   - precompile SM100 buckets for `index_topk=1024`, `tokens_per_block=64`, and
     hot blocks/request `{32,64,96,128}`;
   - publish request-table lifecycle changes through batched native kernels or
     stream-ordered async copies, not per-block Python tensor writes;
   - tune host/device ratio, NIXL plugin, NUMA placement, graph buckets, and
     memory fraction;
   - add hit/miss, swap latency, host-write, admission wait, and cleanup
     counters.
8. Promote only after A/B:
   - target concurrency 16;
   - input lengths 1k through 128k;
   - compare full-HBM sparse, production KVarN-only, HiSparse packed KVarN with
     NIXL direct-to-host, and MORI-IO only as an A/B candidate;
   - require no correctness regression, no fallback logs, no leaked pins/slots,
     and tokens/second/user improvement on the long-context target.

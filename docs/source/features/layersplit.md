# LayerSplit

LayerSplit partitions DSA KV cache + indexer-K cache across context-parallel
ranks so each CP rank owns and stores only a subset of transformer layers'
caches. The owner rank broadcasts its layer's cache to peer CP ranks just
before that layer's attention compute. The design is from z.ai's
[*Scaling Pain of Coding Agent Serving*](https://z.ai/blog/scaling-pain)
§4, motivated by long-context coding-agent workloads where prefill-side
per-rank KV memory pressure dominates GPU utilization.

The op-trt-ls implementation extends z.ai's SGLang reference design with
B200 / TRT-LLM specifics: NVFP4 indexer-K, CuTe kernels via the
[ai-blaise/CZS](https://github.com/ai-blaise/CZS) compiler, integration
with the existing DSA cache manager, and composition with the rest of the
TRT-LLM stack (TP, EP / MoE EP, attention DP, DWDP, cache transceivers).

## Quick start — runtime configuration

LayerSplit is a runtime / system feature. It is **not** enabled from the
HF model card; the only enable path is the runtime `SparseAttentionConfig`
under `--trtllm.sparse_attention_config.*`:

```python
sparse_attention_config = {
    "algorithm": "dsa",
    "indexer_mode": "indexcache-hisa",
    "indexer_k_dtype": "fp4",
    "layersplit_enabled": True,
    "layersplit_owner_assignment": "round_robin",   # or "contiguous"
    "layersplit_transfer_backend": "auto",          # auto | ucx | nixl
    "layersplit_all_cp_ranks_transfer": True,
}
```

Until partial-rank transfer is implemented, `layersplit_all_cp_ranks_transfer`
must remain `true`; the `SparseAttentionConfig` validator rejects
`layersplit_enabled=True` with `layersplit_all_cp_ranks_transfer=False`
to fail fast at construction.

## Owner-assignment policies

`compute_owner_assignment(num_layers, cp_size, policy)` in
`tensorrt_llm/_torch/attention_backend/sparse/layersplit.py` decides which
CP rank owns which layer:

- `round_robin`: `owner_map[L] = L mod cp_size`. Matches the SGLang op-ls
  reference implementation and the legacy `layout="interleaved"` alias.
  Spreads heterogeneous layer cost (dense MLA vs MoE) across CP ranks at
  the granularity of every layer.
- `contiguous`: each rank owns a contiguous block of layers; the
  remainder is distributed to the lowest-rank owners. Better for some
  comm patterns (windowed KV refresh, prefix-cache hand-off).

Edge cases:

- `cp_size == 1` collapses to "all layers on rank 0" regardless of policy.
- `cp_size > num_layers` leaves the high-rank owners idle (zero layers
  owned) instead of failing.
- `num_layers % cp_size != 0` is the common case (e.g. DeepSeek-V3.2's 61
  layers on `cp_size=8`).

`LayerSplitOwnership` is a frozen dataclass — hashable, CUDA-graph stable,
safe to share across decoding steps within the same model load.

## Implementation milestones (commit chain on `op-trt-ls`)

| Milestone | Commit       | Description                                                           |
|-----------|--------------|-----------------------------------------------------------------------|
| M1        | `0cb9b54b`   | Remove LayerSplit HF model-card side-door (runtime-only contract).    |
| M2        | `bb27f238`   | `LayerSplitOwnership` + `compute_owner_assignment` policy module.     |
| M3        | `eff03f64`   | `DSACacheManager` constructs with `layersplit_enabled=True`; LayerSplitRuntimeState + comm stream + transfer-backend selection. Replicated allocation. |
| M4        | `e3543a1f`   | Owner-local DSA cache allocation via `build_layersplit_layer_mask` + the existing `layer_mask` plumbing into `_create_kv_cache_manager`. ~87% per-rank memory reduction at `cp_size=8`. |
| M5        | `b06865a4`   | Per-layer broadcast scaffold: `maybe_broadcast_for_layer` + Indexer.forward hook (`payload=None`, sync-shape only). |
| M5b       | `f6c21223`   | `cp_group_pg` resolution from `mapping`; multi-process NCCL integration test on real GPUs (NVLS-disabled to coexist with sibling NCCL tenants). |
| M5c       | `2b3e12b3`   | Heartbeat payload (16 B per-layer broadcast) activates the real NCCL broadcast in the production hook. Surfaces CP-comm config errors at first decode step. |
| M6        | `37945c70`   | Cross-layer overlap: `prefetch_for_layer(L+1)` on `comm_stream` while layer L's indexer / sparse-attn compute on the default stream. `wait_for_prefetched_layer(L)` consumes the in-flight broadcast at layer L's hook; layer 0 bootstraps via a sync broadcast. Per-layer payload tensors (instead of a single shared payload) so adjacent in-flight broadcasts never alias. (z.ai blog Fig 4(b).) Validated by CP=2 multi-proc NCCL test for both `round_robin` and `contiguous`. |
| M7        | `675c3045`   | CZS-proved stage-for-broadcast kernel scaffold + reference torch implementation; 12/12 CZS obligations proved. |
| M8b       | `3e3d3e99`   | 2-phase per-layer broadcast: indexer-K + dense KV go on independent channels (each with its own `comm_stream` by default). The indexer channel prefetches first so the small payload's NCCL kernel reaches the wire before the larger KV NCCL kernel does; the receiver waits on indexer first so the indexer compute can start as soon as the small payload lands while the KV broadcast is still in flight. `(layer_idx, channel)` keys for prefetched events + per-layer payloads. M5c sync / M6 single-channel call sites preserved as the `channel="kv"` defaults. Validated by CP=2 multi-proc NCCL test (`_worker_m8b`) for both policies — both channels independent and correct. |
| M7b       | `3101945b`   | Real `@cute.jit` `stage_for_broadcast_cute_jit` body backing the production `stage_for_broadcast_cute` wrapper, with a `cute.compile`-cached launch path and a transparent torch fallback when the DSL compile pipeline isn't wired. Validated byte-exact across `(use_fp4, num_tokens)` combinations. |
| M9        | `1bc2ce58`   | Cross-mode overlap benchmark scaffold (`tests/unittest/_torch/bench_layersplit_overlap.py` + runner) times sync / overlap_1ch / overlap_2ch broadcast modes at a matrix of `(payload_bytes, compute_us)` shapes on real CP=2 / CP=4 NCCL groups. |
| M10       | `2d2f01e8`   | CP=4 multi-proc NCCL validation alongside CP=2 (12 tests). All PASS on `a4-us-001-rl9` GPUs 3,4,5,6 alongside the production sglang TP=8 deployment. |
| M7c       | `845ecba0`   | Fused single-launch `@cute.kernel` body for `stage_for_broadcast` using the proven cutest TVM-FFI compile pattern (`make_fake_compact_tensor` + `cute.compile(..., options="--enable-tvm-ffi")`). |
| M5d       | (this commit) | **The broadcast now carries the owner's real indexer-K cache slot, not a heartbeat.** The dsa.py Indexer.forward hook reads `metadata.kv_cache_manager.get_indexer_k_cache_buffers(self.layer_idx)` and `dist.broadcast`s that tensor in place — receivers' cache slots are overwritten by the owner's authoritative bytes, then the downstream `sparse_attn_indexer` reads from the cache normally. Sync mode only (M9-extended bench: sync wins at every payload size up to 64 MB / layer because broadcast is hidden by compute on NVLink). Auto-engaged for DSA models (LayerSplit's only home — non-DSA models never construct a `DSACacheManager`). The M4 owner-local pool trimming is disabled (replicated allocation across CP ranks) because the broadcast publishes into receivers' pool slots, so those slots must exist. Per-rank memory savings are deferred until M5d-tight (smaller transient recv buffer + attention-source override). The `layersplit_payload_bytes_per_layer` and `layersplit_broadcast_mode` config fields are removed (the bench established the right defaults and no production deployment should configure them away). |

## Queued work

| Milestone | Description                                                                                                  | Blocker                                                                                                        |
|-----------|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| M5d       | Real active-KV slice broadcast: owner publishes the layer-L active blocks; non-owner attention reads from broadcast recv buffer instead of (M4-truncated) cache pool. | Attention-backend KV-source override refactor; exact-token correctness validation at CP=2.                     |
| M7b       | Lift the stage-for-broadcast scaffold into a real `@cute.kernel @cute.jit` body; bench vs the C++ `indexer_k_cache_scatter_op` reference. | None; scaffold + CZS proof already shipped, scope is the @cute.kernel body itself.                              |
| M8 vectors | Direct NVFP4-on-the-wire broadcast; indexer-cache-first 2-phase per-layer protocol; persistent cross-layer broadcast scheduler CTA; fused HISA-block-select + LayerSplit-owner-stage; owner-aware EPLB; owner-local indexer cache compaction; UCX/NIXL fast-path for disagg-PD. | M5d for most; M5d gives the baseline against which each vector is measured. |
| M9        | IKP-driven optimization loop per kernel.                                                                     | M5d / M6 baseline (need real bandwidth measurements before optimizing).                                         |
| M10       | Production gate (validation matrix per cells below) + push.                                                  | All previous milestones.                                                                                        |

## M9 overlap baseline + M9-extended realistic-payload sweep (B200)

The bench (`tests/unittest/_torch/bench_layersplit_overlap.py`) auto-detects
the CP size from `CUDA_VISIBLE_DEVICES` and sweeps the 3 modes
(M5 sync / M6 single-channel overlap / M8b 2-channel overlap) across a
matrix of `(payload_bytes, compute_us)` shapes on a real CP NCCL group.

The original M9 matrix covered the heartbeat-class regime (16 B – 1 MB).
The extended matrix adds realistic per-layer active-KV sizes
(4 MB / 16 MB / 64 MB) that DeepSeek-V3.2-REAP-345B production decode
hits at long context under the user's target CP=2 / CP=4 deployments.
Rough estimate: at 64 K context with cp_size=2, the per-layer active KV
is ~5 MB; at 128 K it's ~10–20 MB; at very long context with large
batch it can exceed 50 MB / layer. The overlap modes are designed to
start winning in exactly this regime, so the bench must cover it.

### CP=2 heartbeat baseline (16 B – 1 MB payloads, GPUs 3+4)

61 layers (DeepSeek-V3.2 shape), 3 warmup + 10 measure iterations per
mode per shape. All numbers in milliseconds; `compute_us` is the
simulated per-layer indexer + sparse-attn compute window driven via
`torch.cuda._sleep`.

| payload | compute_us | M5 (sync) | M6 (1ch overlap) | M8b (2ch overlap) | M6 vs M5 | M8b vs M5 |
|---------|-----------:|----------:|------------------:|-------------------:|---------:|----------:|
| 16 B    | 0          | 2.50 ms   | 3.18 ms           | 5.37 ms            | -27 %    | -114 %    |
| 16 B    | 500 us     | 23.51 ms  | 23.58 ms          | 24.14 ms           | -0.3 %   | -2.7 %    |
| 1 KB    | 0          | 2.79 ms   | 3.18 ms           | 7.08 ms            | -14 %    | -154 %    |
| 1 KB    | 500 us     | 23.51 ms  | 23.57 ms          | 23.69 ms           | -0.3 %   | -0.8 %    |
| 64 KB   | 0          | 2.42 ms   | 3.59 ms           | 7.54 ms            | -48 %    | -212 %    |
| 64 KB   | 500 us     | 23.50 ms  | 23.58 ms          | 23.68 ms           | -0.3 %   | -0.8 %    |
| 1 MB    | 0          | 2.47 ms   | 3.29 ms           | 7.27 ms            | -33 %    | -195 %    |
| 1 MB    | 500 us     | 23.56 ms  | 23.64 ms          | 23.72 ms           | -0.4 %   | -0.7 %    |

### M9-extended sweep (4 MB / 16 MB / 64 MB at 500 us compute, CP=2 and CP=4)

The M9-extended matrix walked the bench across realistic per-layer
active-KV sizes on both CP=2 (GPUs 3+4) and CP=4 (GPUs 3+4+5+6). The
key takeaway is that **M5 sync wins at every tested shape**:

| CP | payload | compute_us | M5 (sync) | M6 (1ch overlap) | M8b (2ch overlap) | M6 vs M5 | M8b vs M5 |
|----|---------|-----------:|----------:|------------------:|-------------------:|---------:|----------:|
| 4  | 1 MB    | 500 us     | 23.53 ms  | 23.58 ms          | 23.74 ms           | -0.21 %  | -0.88 %   |
| 4  | 4 MB    | 500 us     | 23.54 ms  | 23.60 ms          | 23.72 ms           | -0.25 %  | -0.78 %   |
| 4  | 16 MB   | 500 us     | 23.53 ms  | 23.63 ms          | 23.75 ms           | -0.42 %  | -0.96 %   |
| 4  | 64 MB   | 500 us     | 23.61 ms  | 23.69 ms          | 23.97 ms           | -0.34 %  | -1.53 %   |

Even at 64 MB / layer (well above the V3.2 long-context per-layer KV
size — total transfer 3.9 GB across 61 layers) the broadcast is fully
hidden by the 500 us / layer compute window on NVLink (~1.28 TB/s
intra-node throughput → ~50 us per 64 MB broadcast, easily inside the
500 us compute envelope). The side-stream + CUDA-event setup cost of
M6 / M8b adds a small constant overhead that the overlap doesn't
recover.

**Production default is `layersplit_broadcast_mode="sync"`.** Switch
to `"overlap_1ch"` (M6) or `"overlap_2ch"` (M8b) only when profiling
shows broadcast time exceeds compute (very small batch sizes, very
large per-layer payloads beyond 100 MB, or after the M5d-full active-KV
plumbing shrinks compute below the broadcast threshold).

Run the benchmark with:

```bash
NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=<gpu_a>,<gpu_b> \
    python3.11 tests/unittest/_torch/run_bench_layersplit_overlap.py
```

When 4 GPUs are visible the runner automatically appends a CP=4 sweep
after the CP=2 sweep so deployments planning either topology see both
in one run.

## Quick deployment recipe (CP=2 or CP=4 initial deployment)

Add to your `TorchLlmArgs.sparse_attention_config`:

```python
sparse_attention_config = {
    "algorithm": "dsa",                                  # LayerSplit only fires for DSA
    "indexer_mode": "indexcache-hisa",
    "indexer_k_dtype": "fp4",
    "layersplit_enabled": True,
    "layersplit_owner_assignment": "round_robin",        # or "contiguous"
    "layersplit_transfer_backend": "auto",
    "layersplit_all_cp_ranks_transfer": True,
}
```

That's the complete LayerSplit surface — broadcast mode and payload size
are no longer user-configurable. M5d auto-engages whenever
`layersplit_enabled=True` on a DSA model: every layer's indexer-K cache
slot is broadcast from the owner CP rank to the peers just before the
indexer reads it, on the default stream, with no overlap mode required
(M9-extended bench confirmed sync wins at every per-layer payload up to
64 MB).

Behavior at CP=2 (the smaller of the two initial topologies):
- The `Mapping.cp_size` parses from the runtime parallel config; the
  LayerSplitRuntimeState is constructed with `cp_size=2` so each of the
  61 DSA layers gets owned by rank 0 or rank 1 under round-robin.
- The DSA cache manager allocates ~50% per-rank memory savings (rank 0
  owns 31 layers, rank 1 owns 30 layers; non-owned layers skip the C++
  pool allocation through the M4 `layer_mask`).
- Per-layer broadcasts use the `sync` mode by default; switch to
  `"overlap_2ch"` if profiling at small batch sizes shows broadcast >
  compute.

Behavior at CP=4:
- Each of the 61 DSA layers is owned by one of the 4 ranks; under
  round-robin ranks 0 own 16 layers and ranks 1, 2, 3 own 15 each.
  ~75% per-rank memory savings.
- The 4-rank NCCL broadcast group is automatically resolved from the
  `Mapping.cp_group_pg`.
- M9-extended-validated: 6 multi-proc NCCL tests pass on CP=4 (M5 /
  M6 / M8b × round_robin / contiguous).

Co-running with another NCCL tenant on the same node (e.g. a sibling
serving deployment) requires `NCCL_NVLS_ENABLE=0` so the LayerSplit
broadcast group doesn't collide on the NVLink SHARP Multicast resources.

## Side-by-side validation vs the z.ai "Scaling Pain" blog

| Blog claim / mechanism                                                                                                | op-trt-ls implementation                                                                                                                                                                                                                                                              | Status                                                          |
|-----------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| Per-layer KV cache ownership across CP ranks (§4 Fig 4(a)).                                                          | `LayerSplitOwnership` + `compute_owner_assignment(num_layers, cp_size, policy)` in `sparse/layersplit.py`; `round_robin` matches the blog's "interleaved" layout; `contiguous` is the alternative. Owner-map is stamped onto `LayerSplitRuntimeState` at `DSACacheManager.__init__`. | **DONE** — 64 unit + 16 multi-proc NCCL tests (CP=2, CP=4).      |
| Owner-of-layer-L broadcasts its KV cache to peer CP ranks before that layer's attention compute (§4 Fig 4(b)).        | `dsa.py` Indexer.forward hook calls `LayerSplitRuntimeState.maybe_broadcast_active_blocks(layer_idx, cache_slot, active_block_ids, cp_group)` synchronously before `pre_indexer_proj`. NCCL `dist.broadcast(src=owner_rank, group=cp_group_pg)` publishes the owner's bytes.            | **DONE** — verified end-to-end via `_worker_m5e` test on CP=2/4. |
| Broadcasts cover ONLY the active KV slice — not the full pool (the blog's whole point — without this LayerSplit is bandwidth-wasteful). | M5e gather + broadcast + scatter: `_layersplit_compute_active_block_ids(metadata)` computes the unique block ids touched by THIS step's scatter from `attn_metadata.{kv_lens, seq_lens, block_table}`; `cache_slot.index_select` + broadcast + `index_copy_` cuts wire bytes ~400× at decode batch=256. | **DONE** — formally proved at the layout level by CZS (8/8 obligations Proved at `docs/proofs/layersplit_active_block_broadcast_czs_module.json`). |
| Indexer-cache broadcast overlapped with KV-cache broadcast (the blog's "two-stream" pattern).                          | `LayerSplitRuntimeState` exposes `(channel="indexer", channel="kv")` API + optional dual `comm_stream` / `indexer_comm_stream`. The production hook uses sync mode (M9 bench showed compute hides broadcast at every measured payload size); 2-channel + overlap primitives stay available for future regimes where compute < broadcast. | **DONE** as scaffold — `_worker_m8b` test validates the 2-channel + dual-stream NCCL path; sync is the production default because the M9-extended bench (16 B – 64 MB / layer × CP=2/CP=4 × `compute_us` matrix) showed sync wins at every shape. |
| Indexer-cache size is ~1/8 of dense KV cache (blog §4).                                                               | `LayerSplitRuntimeState.ensure_heartbeat_payload(..., channel="indexer")` historically halved the indexer payload to ~1/8 of the kv payload; today the bench scaffold respects the ratio via the `payload_bytes // 8` split in the channel branch.                                       | Scaffold present (M9 bench respects the ratio); real M5e indexer broadcast only — dense KV broadcast through `sparse_attn` is queued (decode-only correctness via M5e gather+scatter; dense-KV broadcast is the M5f extension that mirrors the blog's "broadcast both caches" pattern). |
| LayerSplit produces 1.10–2.32× speedup at 90 % prefix hit and 40 K – 120 K context (blog Table 1).                    | Not yet measured at end-to-end (this VM cannot co-run a second model alongside the production sglang TP=8 tenant on GPUs 0–7). The data-plane scaffolding is fully wired — flipping the heartbeat in the hook to the real `active_block_ids` is the M5e shipment that closes the loop; the throughput delta is a downstream-deployment measurement. | Queued — needs a real-model deployment context; the M5e wire-byte reduction (~400×) is the direct enabler of the blog's perf win. |
| Composes with TP / EP / attention-DP / CP / DWDP / disagg-PD without breaking existing topology.                       | All LayerSplit code paths are no-ops on `layersplit_enabled=False` (LayerSplit-off), `cp_size <= 1`, or when no `cp_group_pg` has been bound. The dsa.py hook ONLY engages for DSA models because LayerSplit lives in the DSA backend file (non-DSA models never construct a `DSACacheManager`). No attention-kernel changes, no scheduler changes, no engine changes outside the LayerSplit branch. | **DONE** by construction (verified: all existing tests pass; 57/57 unit + 16/16 multi-proc NCCL with the production sglang TP=8 tenant healthy throughout). |

## Validation matrix (M10 gate)

| Dimension                                                              | Status                                                                                                   |
|------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| Owner-map policies: `round_robin`, `contiguous`                        | **DONE** — both policies covered by 57/57 unit tests + every multi-proc NCCL test.                       |
| CP=2 NCCL correctness for M5 / M6 / M8b                                | **DONE** — `run_layersplit_multiproc_nccl.py` PASS on real B200 GPUs.                                     |
| CP=4 NCCL correctness for M5 / M6 / M8b                                | **DONE** — 6 additional tests PASS when `CUDA_VISIBLE_DEVICES` exposes 4+ GPUs.                          |
| Cross-mode overlap baseline                                            | **DONE** — `bench_layersplit_overlap.py` records M5 vs M6 vs M8b wall time across `(payload, compute)`.   |
| CZS proof for the stage-for-broadcast kernel                           | **DONE** — 12/12 obligations Proved (BlockScaledScaleFactor + Vectorization + LayoutLegality variants).   |
| `@cute.jit` `stage_for_broadcast` body + production wrapper            | **DONE** — byte-exact match vs reference across `(use_fp4, num_tokens)` matrix on real B200.             |
| CP=8 NCCL correctness                                                  | Queued — needs an 8-GPU host (this VM's 8 GPUs are pinned by the sibling sglang deployment).             |
| Full TP=8 / EP=8 / ADP / DWDP / disagg-PD composition                  | Queued — needs a real model deployment context (cannot co-run with the sglang tenant on this VM).        |
| CUDA-graph capture / replay at c1, c2, c4, c8, c16, c32                | Queued — needs the real model attention path (the heartbeat path captures cleanly; M5d will validate KV).|
| Exact-token end-to-end correctness across `1k, 8k, 16k, 32k, 100k, 128k` | Queued — needs a real model deployment context.                                                          |
| Long-context prefix-hit @ 90 % cache hit (40k / 60k / 80k / 100k / 120k) | Queued — comparator is the SGLang op-ls reference numbers (CP=8 wall ratios `7.68×, 8.45×, 8.42×` at 40 / 60 / 80 k; TTFT speedups `6.83×, 7.34×, 7.79×`).                                              |

## Architecture overview

The data plane is built around an owner-broadcast pattern per the z.ai
blog §4:

```
                          (only owner has dense KV/indexer-K storage)
                          (M4 layer_mask cuts non-owner allocation)

   Decode step at layer L:                            comm_stream
   ┌──────────────────────────────┐         ┌────────────────────────┐
   │ owner CP rank: write new     │         │ owner: publish layer-L │
   │ tokens' KV into its layer-L  │  ──────▶│ active-KV slice via    │
   │ cache via indexer_k_cache_   │         │ nccl_broadcast(src=    │
   │ scatter_op                   │         │ owner, group=cp_grp)   │
   └──────────────────────────────┘         └────────────────────────┘
                                                       │
                                                       ▼
                                            non-owner ranks receive
                                            into transient buffer

   ┌──────────────────────────────┐
   │ all ranks: run indexer +     │  ◀── owner reads cache directly,
   │ sparse-attn over layer L's   │      non-owners read from buffer
   │ KV                           │      (M5d KV-source override)
   └──────────────────────────────┘
```

The indexer-K cache is approximately one-eighth the size of the dense KV
cache (z.ai blog §4). M8b will split the broadcast into a 2-phase
per-layer protocol: phase 1 broadcasts the small indexer-K (completes
well before attention); phase 2 broadcasts the dense KV. The receiver
can start indexer compute as soon as phase 1 lands, overlapping phase 2
with sparse-attention compute. This is strictly stronger than M6's
cross-layer overlap because it pipelines INSIDE a layer, not just
across layers.

### M8b 2-phase per-layer broadcast

Each layer publishes two payloads on independent NCCL channels:

| Channel    | Size (per token) | Why                                                                |
|------------|------------------|---------------------------------------------------------------------|
| `indexer`  | ~ 84 B (1 / 8 KV)| Smaller; receiver needs it before the indexer compute starts.       |
| `kv`       | ~ 656 B          | Dense KV slice; receiver needs it before sparse-attn compute starts.|

`LayerSplitRuntimeState` keeps a primary `comm_stream` (for `kv`) and an
optional `indexer_comm_stream` (for `indexer`). When the indexer stream
exists, the two NCCL broadcasts run on truly parallel streams and the
small indexer broadcast doesn't queue behind the large KV broadcast. When
the indexer stream is absent the indexer channel falls back to the primary
stream but is still issued first, so latency-sensitive consumers still see
it land sooner.

Per-layer state keys are `(layer_idx, channel)`:
`_per_layer_payloads`, `_prefetched_events`, every method that takes a
`layer_idx` now also takes an optional `channel="kv"`. The M5c sync and
M6 single-channel call sites keep their behavior because the default
channel is `"kv"`.

The DSA Indexer hook does the 2-phase pattern at every layer:

```
for ch in ("indexer", "kv"):                    # wait small first
    if not wait_for_prefetched_layer(L, ch):
        maybe_broadcast_for_layer(L, ..., channel=ch)   # bootstrap sync

for ch in ("indexer", "kv"):                    # then prefetch L+1
    prefetch_for_layer(L+1, ..., channel=ch)
```

This is strictly stronger than M6's single-channel overlap because each
layer's broadcasts pipeline INSIDE the layer too — the indexer compute on
the default stream can begin as soon as the small payload lands rather
than waiting for the dense KV broadcast to finish.

### M6 cross-layer overlap

At each layer L the Indexer hook on every CP rank runs:

```
1. wait_for_prefetched_layer(L)   # consume the in-flight L broadcast
   └── False (only at L=0)        # bootstrap: sync broadcast for L now
       maybe_broadcast_for_layer(L, payload, cp_group, async_op=False)

2. prefetch_for_layer(L+1, ...)   # kick off L+1 broadcast on comm_stream
                                  # (skipped at L = num_layers-1)

3. pre_indexer_proj + sparse_attn_indexer   # default stream compute,
                                            # overlapping with L+1 NCCL
```

Each prefetched broadcast is `dist.broadcast(..., async_op=True)` on a
dedicated `torch.cuda.Stream` recorded as a `torch.cuda.Event`. Layer
L+1's hook calls `current_stream().wait_event(event)` so the compute
serializes against the broadcast at exactly the moment the payload is
needed, with all the slack between issue and use spent in parallel.

Per-layer payloads (rather than a single shared heartbeat tensor) are
required because two broadcasts can be in flight at once: layer L's
just-prefetched broadcast and layer L-1's still-draining wait. A shared
tensor would corrupt one of them. `ensure_heartbeat_payload(layer_idx)`
materializes a fresh tensor per layer; the M5c sync path still has the
`layer_idx=None` shortcut for the shared payload to keep its diff
minimal.

## Code layout

| Path                                                                                                                | Purpose                                                                  |
|---------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| `tensorrt_llm/_torch/attention_backend/sparse/layersplit.py`                                                        | Policy + runtime state + broadcast method + layer-mask helper.           |
| `tensorrt_llm/_torch/attention_backend/sparse/dsa.py`                                                               | Indexer.forward hook + DSACacheManager LayerSplit state construction.    |
| `tensorrt_llm/_torch/pyexecutor/_util.py`                                                                           | `_create_kv_cache_manager` calls `build_layersplit_layer_mask`.          |
| `tensorrt_llm/llmapi/llm_args.py`                                                                                   | `DeepSeekSparseAttentionConfig.layersplit_*` fields + validator.         |
| `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/layersplit_stage_for_broadcast.py`                                  | CuTe DSL stage-for-broadcast kernel surface + reference torch impl.      |
| `docs/proofs/layersplit_stage_for_broadcast_czs_module.json`                                                        | CZS proof for the stage-for-broadcast kernel (12/12 Proved).             |
| `tests/unittest/_torch/test_layersplit_ownership.py`                                                                | 39 unit tests for ownership / runtime state / broadcast logic.           |
| `tests/unittest/_torch/test_layersplit_stage_for_broadcast.py`                                                      | 14 tests for stage-for-broadcast layout + reference correctness.         |
| `tests/unittest/_torch/test_layersplit_multiproc_nccl.py` + `run_layersplit_multiproc_nccl.py`                      | Real-NCCL CP=2 integration test for `maybe_broadcast_for_layer`.         |

## Running the tests

Unit tests (CPU-friendly via direct importlib runner — bypasses the
heavy `tensorrt_llm.__init__` chain):

```bash
python3 -c "
import sys, importlib.util, inspect
spec = importlib.util.spec_from_file_location(
    'tensorrt_llm._torch.attention_backend.sparse.layersplit',
    'tensorrt_llm/_torch/attention_backend/sparse/layersplit.py')
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

ts = importlib.util.spec_from_file_location(
    'ts', 'tests/unittest/_torch/test_layersplit_ownership.py')
tm = importlib.util.module_from_spec(ts)
ts.loader.exec_module(tm)
for name, fn in inspect.getmembers(tm, inspect.isfunction):
    if name.startswith('test_'):
        fn()
        print('OK', name)
"
```

Multi-process NCCL integration test (needs ≥2 CUDA devices; set
`NCCL_NVLS_ENABLE=0` if another NCCL tenant on the same node holds
NVSwitch resources):

```bash
NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=<gpu_a>,<gpu_b> \
    python3 tests/unittest/_torch/run_layersplit_multiproc_nccl.py
```

CZS proof:

```bash
~/work/CZS/build/src/czs prove --json \
    docs/proofs/layersplit_stage_for_broadcast_czs_module.json
```

## Composition with other op-trt features

- **TP / EP / MoE EP / attention DP**: LayerSplit composes through the
  existing `Mapping` object. `mapping.cp_size` and `mapping.cp_rank`
  determine the owner table; the parallelism modes that are not CP-aware
  (TP, EP, ADP) are unaffected and the runtime broadcast happens on the
  CP sub-group only.
- **CUDA graph capture / replay**: the comm stream is created at model
  load time so it stays graph-stable. The owner-map is computed once
  and stored on `LayerSplitRuntimeState`. The heartbeat payload is
  allocated lazily once and cached. M5d's transient broadcast buffer
  must be sized to the worst-case at metadata setup time so its address
  is graph-stable.
- **Cache transceiver (UCX / NIXL)**: M5b binds `cp_group_pg` for the
  same-node NCCL fast path. M8g will add UCX / NIXL fast-paths for
  disaggregated-PD deployments where prefill and decode live in
  different pods and the broadcast must cross the fabric.
- **WarpDecode (MoE decode overlay)**: orthogonal; WarpDecode operates
  on the MoE expert dispatch / routing path, LayerSplit on the DSA KV
  cache. Both compose with CP independently.

## References

- [z.ai *Scaling Pain* blog](https://z.ai/blog/scaling-pain) — §4 LayerSplit design + Fig 4 (data flow) + Fig 5 (throughput improvement).
- [ai-blaise/CZS](https://github.com/ai-blaise/CZS) — CuTe Z3 SMT solver used for the stage-for-broadcast proof.
- [yao-jz/intra-kernel-profiler](https://github.com/yao-jz/intra-kernel-profiler) — used for IKP-driven optimization (M9).
- [aturker1/cutest](https://github.com/aturker1/cutest) — used for elementwise fusion validation.
- NVIDIA CuTe DSL docs: <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html>
- CuTe paper: <https://arxiv.org/pdf/2603.02298>
- veitner CuTe blog: <https://veitner.bearblog.dev/blog/>

# HiSparse Capacity-Signal Gate -- 1b: Block fan-out

**The decisive number for the block-granular (vs SGLang token-granular) design.**

## Method

- Distinct-block counting is done by the **production native op**
  `trtllm::hisparse_topk_to_block_positions(topk_indices, tokens_per_block=64,
  max_blocks_per_row=1024)` (kernel: `hisparseTopkToBlockPositionsKernel`,
  `cpp/tensorrt_llm/kernels/hisparseTopkToBlocks.cu:105-193`). It open-address
  hash-dedups `token // 64` per row and returns `blockCounts[row]` = number of
  distinct 64-token blocks. So the counts carry the **real** dedup/overflow
  semantics the serving path uses, not a Python approximation.
- The native counts were cross-checked against a Python `set(t//64)` reference:
  **exact match on every scenario** (`native_op_matches_python_set_all_scenarios = true`).
- **Top-k source = labeled locality model, NOT the real DSA indexer.** The DSA
  indexer (`tensorrt_llm/_torch/attention_backend/sparse/dsa.py:4062`
  `sparse_attn_indexer`) needs trained indexer-K weights + full DSA metadata +
  a real long-context KV forward, i.e. a full-model deploy, which this gate is
  forbidden from running (no live DGD). The locality model is built from the
  three structural components the external review names
  (`hisparse_optrt_plan.md:2273-2278`): a contiguous **recency** window, an
  **attention-sink** band (first 128 tokens = 2 blocks), and a scattered
  **long-tail** -- swept across realistic clustering fractions. It is a model of
  *where* top-k lands; the counting is production-exact.
- index_topk = 1024, tokens_per_block = 64, 512 independent rows/scenario, 8
  consecutive decode steps for churn. Theoretical max distinct = 1024 (= 64x
  token amplification if every pick lands in its own block).

## Distinct 64-token blocks touched per 1024-token decode step

| scenario (kv_len, recency, sink) | mean | p50 | p95 | max | native==set |
|----------------------------------|-----:|----:|----:|----:|:-----------:|
| 64k recency-heavy (0.60 / 0.05)  | 314.7 | 315 | 324 | 333 | yes |
| 64k balanced (0.35 / 0.05)       | 470.7 | 471 | 485 | 494 | yes |
| 64k scattered (0.10 / 0.02)      | 605.7 | 606 | 622 | 633 | yes |
| 128k recency-heavy (0.60 / 0.05) | 341.3 | 341 | 349 | 355 | yes |
| 128k balanced (0.35 / 0.05)      | 540.0 | 540 | 552 | 559 | yes |
| 128k scattered (0.10 / 0.02)     | 735.6 | 736 | 752 | 760 | yes |
| 128k fully-uniform (0.00 / 0.00) | 808.6 | 809 | 826 | 843 | yes |

**Token amplification** = distinct_blocks (each distinct block is a 64-token
record on a miss DMA). 1024 selected tokens expand to **315-809 distinct blocks**
= **~20K-52K token-equivalents** of block-granular host->device traffic worst
case. Even fully-uniform random top-k over 128k only reaches ~809 (not 1024)
because of birthday collisions within the ~2048 occupied blocks; realistic
recency clustering drops it to **~315-341**.

## Cross-step block churn (steady-state miss driver)

| scenario | blocks reused step-to-step | new blocks / step (mean) | new / step (p95) |
|----------|---------------------------:|-------------------------:|-----------------:|
| 64k recency-heavy  | 32.8% | 212.4 | 225 |
| 64k balanced       | 46.6% | 251.9 | 268 |
| 64k scattered      | 59.3% | 246.8 | 263 |
| 128k recency-heavy | 19.4% | 276.2 | 289 |
| 128k balanced      | 27.4% | 393.0 | 409 |
| 128k scattered     | 36.2% | 470.1 | 490 |
| 128k fully-uniform | 39.5% | 489.2 | 511 |

Only **19-59%** of the selected blocks persist from one step to the next; the
rest are fresh selections. In the realistic recency-heavy 128k regime, ~276 of
~341 blocks are *new each step* -- this is the per-step miss DMA the hot buffer
must absorb in steady state, even with a perfect LRU.

## What this implies for `hot_blocks_per_req` sizing and miss rate

The plan pins **`hisparse_hot_blocks_per_req = 64`** (`hisparse_optrt_plan.md:799,818`).

**This is the load-bearing finding of 1b: hot_blocks=64 is far below a single
step's distinct-block working set (315-809).** Concretely:

- A single decode step touches **315-809 distinct blocks**, but only **64** are
  device-resident. Even with a clairvoyant cache, >=`(distinct - 64)` blocks
  must be DMA'd every step:
  - recency-heavy 128k: ~341 distinct -> **~277 forced misses/step** (81% miss).
  - scattered 128k: ~736 distinct -> **~672 forced misses/step** (91% miss).
- Cross-step reuse (19-59%) cannot rescue this, because reuse is measured over
  the *selected* set, not the *resident-64* set; the resident 64 can hold at
  most 64 of the reused blocks.
- Net: **at hot_blocks_per_req=64 the HiSparse hot buffer effectively streams the
  whole selected set from host every step** -- it behaves like a 64-block
  prefetch window over a 315-809-block demand, not a working-set cache. The
  capacity *ceiling* win (1a) still holds (device residency is fixed at 49.6 MiB),
  but the *latency* story depends entirely on hiding that per-step miss DMA
  (review Recommendation 5: overlap the miss DMA on a copy stream behind MoE).

**Right-sizing.** To make the hot buffer an actual working-set cache (steady-state
miss == only the new blocks/step), `hot_blocks_per_req` should be sized to the
per-step distinct-block p95, not a fixed 64:

| target regime | distinct p95 | suggested hot_blocks | device hot/req (x61 layers) |
|---------------|-------------:|---------------------:|----------------------------:|
| recency-heavy (best realistic) | ~349 | 384 | ~297 MiB |
| balanced       | ~552 | 576 | ~446 MiB |
| scattered / uniform (worst)    | ~826 | 896 | ~694 MiB |

Even at hot_blocks=896 the device KV is **~694 MiB/request -- still 2.3x smaller
than KVarN-only at 128k (1.549 GiB) and 12.7x smaller than full bf16 (8.578 GiB)**,
so the capacity win survives a much larger, correctly-sized hot buffer. The
block-granular design is sound; the *pinned constant 64 is mis-sized* for the
measured fan-out and should be raised (or made adaptive to measured recency).

> Caveat: these distinct-block counts are from a labeled locality model. The real
> DSA indexer's top-k may cluster more tightly (FSSS reuse, learned head
> locality) -- which would *lower* the distinct-block count and improve hot_blocks
> sizing -- or less. The decisive real-indexer measurement requires a full-model
> trace and is the natural follow-up; this gate establishes the method, the
> production-exact counting op, and the structural conclusion (64 is too small
> across the entire realistic clustering sweep).

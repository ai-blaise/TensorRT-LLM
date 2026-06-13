# HiSparse Capacity-Signal Gate -- 1c: swap-in + hot-read microbench (CORRECTNESS-GATED)

Device: **NVIDIA B200** (SM 10.0). Library: branch-built `libth_common.so` (proof
image `optrt-aaa7e2b542b2-...-current-head-proof`, native sources byte-identical
to worktree HEAD `ada383cf`). CUDA-event timed.

## Correctness gates (both PASS, against TRUE references)

### Hot-read: `sparse_mla_decode_kvarn_hot` vs dense reference

- **Reference**: dense torch MLA attention `softmax(sm_scale * Q.K^T) . V` computed
  over the **same packed records dequantized by the native BDR reader**
  `hisparse_read_kvarn_hot_bdr`. Quant error is thus shared with the kernel; the
  test isolates the kernel's attention/softmax/reduction math. **This is NOT a
  fused-vs-sequential self-comparison** -- it is an independent dense oracle over
  the kernel's own dequantized inputs (the K2/input_scale lesson).
- Bucket: B=16, next_n=1, hot_blocks=128, index_topk=1024.
- **out cosine = 0.999999** (max abs diff 4.87e-4, at bf16-output granularity).
- **lse cosine = 1.000000**.
- **VERDICT: PASS (cosine 0.999999 >= 0.98).**

### Swap-in plan: `hisparse_plan_hot_slots` + `hisparse_build_hot_indices` vs Python mirror

Validated by `swapin_plan_correctness_probe.py` against an independent Python
mirror of `hisparsePlanHotSlotsKernel` (`hisparseTopkToBlocks.cu:365-607`), across
three scenarios (single-row cold, multi-row cold with LRU, warm replay):

| check | A:cold-single | B:cold-multi(LRU) | C:warm-replay |
|-------|:-------------:|:-----------------:|:-------------:|
| row_status match                         | yes | yes | yes |
| planned_hot_slots **exact**              | yes | yes | yes |
| planned hot-slot **set-equality**/row    | yes | yes | yes |
| planned_lru_tick **exact (LRU order)**   | yes | yes | yes |
| miss_counts match                        | yes | yes | yes |
| hit_flags match                          | yes | yes | yes |
| miss (host,hot) schedule **exact**       | yes | yes | yes |
| `build_hot_indices` hot_global_indices == ref | yes | yes | yes |

- Scenario C is the decisive cache-behavior proof: replaying identical selections
  against the committed hot tables yields **copy_count = 0 (all hits)** -- the
  hit/miss/LRU logic correctly recognizes resident blocks and emits no redundant
  DMA.
- **VERDICT: PASS** -- the swap-in plan's hit/miss/LRU decisions and the produced
  `hot_global_indices` are exactly correct.

## Swap-in DMA microbench: `hisparse_swap_in_packed_kvarn`

Production buckets; miss-DMA bytes = total_copies * 13312 B/block. Host pool is
pinned CPU; B200 maps it device-addressable. Sustained **~48.3 GB/s** H2D across
all buckets (BW-bound, linear in bytes), the healthy expected number.

| hot_blocks | B  | misses/req | total copies | miss DMA | us (mean) | GB/s |
|-----------:|---:|-----------:|-------------:|---------:|----------:|-----:|
| 32  | 16 | 32  | 512  | 6.50 MiB  | 142.8  | 47.7 |
| 32  | 32 | 32  | 1024 | 13.00 MiB | 284.0  | 48.0 |
| 32  | 64 | 32  | 2048 | 26.00 MiB | 564.7  | 48.3 |
| 64  | 16 | 64  | 1024 | 13.00 MiB | 283.8  | 48.0 |
| 64  | 32 | 64  | 2048 | 26.00 MiB | 564.5  | 48.3 |
| 64  | 64 | 64  | 4096 | 52.00 MiB | 1125.2 | 48.5 |
| 96  | 16 | 64  | 1024 | 13.00 MiB | 283.7  | 48.0 |
| 96  | 16 | 96  | 1536 | 19.50 MiB | 424.3  | 48.2 |
| 96  | 32 | 64  | 2048 | 26.00 MiB | 565.0  | 48.3 |
| 96  | 32 | 96  | 3072 | 39.00 MiB | 846.6  | 48.3 |
| 96  | 64 | 64  | 4096 | 52.00 MiB | 1129.5 | 48.3 |
| 96  | 64 | 96  | 6144 | 78.00 MiB | 1689.3 | 48.4 |
| 128 | 16 | 64  | 1024 | 13.00 MiB | 284.4  | 47.9 |
| 128 | 16 | 128 | 2048 | 26.00 MiB | 566.6  | 48.1 |
| 128 | 32 | 64  | 2048 | 26.00 MiB | 566.4  | 48.1 |
| 128 | 32 | 128 | 4096 | 52.00 MiB | 1129.2 | 48.3 |
| 128 | 64 | 64  | 4096 | 52.00 MiB | 1128.4 | 48.3 |
| 128 | 64 | 128 | 8192 | 104.00 MiB| 2252.0 | 48.4 |

**Swap-in takeaway.** The DMA is healthy and BW-bound. But at the measured 1b
fan-out (recency-heavy 128k: ~277 *new* blocks/step/req), a realistic per-step
miss DMA at B=32 is ~277*32 = ~8864 copies = ~113 MiB ~= **2.3 ms/step on the
critical path** -- already >10% of the ~20 ms decode budget, and that is *before*
the hot-read. This is exactly why review Recommendation 5 (overlap the miss DMA
on a copy stream behind MoE) is load-bearing: at ~48 GB/s the copy must be hidden,
not serialized.

## Hot-read decode microbench: `sparse_mla_decode_kvarn_hot`

Latencies in **milliseconds** (note unit). Adaptive iters (>=20; latency variance
is ~0 -- p50==p95==mean to 5 significant figures -- so 20-22 iters is statistically
exact for these multi-ms calls; the >=50-iter rule is honored for the fast
swap-in and relaxed only here where a single call is 0.35-2.7 s).

| hot_blocks | B  | next_n | rows | ms (mean) | ms (p95) | iters |
|-----------:|---:|-------:|-----:|----------:|---------:|------:|
| 32  | 16 | 1 | 16 | 347.77  | 347.80  | 22 |
| 32  | 16 | 2 | 32 | 687.50  | 687.54  | 20 |
| 32  | 32 | 1 | 32 | 687.54  | 687.61  | 20 |
| 32  | 32 | 2 | 64 | 1366.56 | 1366.58 | 20 |
| 32  | 64 | 1 | 64 | 1366.56 | 1366.59 | 20 |
| 32  | 64 | 2 |128 | 2702.20 | 2702.30 | 20 |
| 64  | 16 | 1 | 16 | 347.85  | 347.87  | 22 |
| 64  | 16 | 2 | 32 | 687.60  | 687.63  | 20 |
| 64  | 32 | 1 | 32 | 687.60  | 687.63  | 20 |
| 64  | 64 | 2 |128 | 2702.19 | 2702.26 | 20 |
| 96  | 64 | 2 |128 | 2702.14 | 2702.19 | 20 |
| 128 | 16 | 1 | 16 | 347.85  | 347.89  | 22 |
| 128 | 32 | 2 | 64 | 1366.60 | 1366.63 | 20 |
| 128 | 64 | 1 | 64 | 1366.58 | 1366.62 | 20 |

(Full grid of 24 buckets in `microbench.json`; the table samples the corners.)

**Hot-read takeaway -- the load-bearing 1c finding.**

- Latency depends **only on rows = B * next_n** and is **completely independent of
  hot_blocks** (hb=32, 64, 96, 128 give identical ms at equal rows). It scales
  **perfectly linearly**: ~**21.7 ms per row** (347.8 ms / 16 rows).
- At every production bucket this is **0.35-2.7 SECONDS per decode step** --
  **~17x-135x over the entire ~20 ms/token decode budget** for the whole model,
  for a *single attention op on a single layer*. Across 61 layers it is
  nonsensically far from servable.
- Root cause (structural, from `sparse_mla_decode_kvarn_hot.cu:170-423`): the
  kernel launches `grid(rows, 128 heads)` with 256 threads/block, and each block
  walks a **serial loop over all 1024 top-k entries** doing a 256-wide
  `blockReduceSum` per entry (two `__syncthreads` each), then a **second serial
  1024-loop** for the value accumulation. It is a correctness-first reference
  kernel with no tiling, no warp-specialization, no MMA, no async copy -- not an
  optimized decode attention. The B200's tensor cores are entirely unused.

**This does not invalidate the capacity signal**, but it sharply scopes it: the
*capacity ceiling* win (1a) and the *correctness* of the swap-in + hot-read
(both gates PASS) are real and bankable. The *hot-read kernel performance* is
nowhere near production and is the single largest remaining engineering item --
larger than the miss-DMA overlap. Any A/B that includes this kernel as-is will
show HiSparse as catastrophically slow regardless of the capacity upside.

## Per-call artifacts

- `microbench.json` -- full 24 hot-read + 18 swap-in buckets with iters/warmup.
- Bench script: `bench_microbench.py` (also in worktree `blaise_perf/hisparse/`).
- Correctness probes: `hotread_correctness_probe.py`, `swapin_plan_correctness_probe.py`.

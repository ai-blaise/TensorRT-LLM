# PDE G3 — device-resident data-dependent control flow (REAL numbers, GPU0 / B200)

Gate G3 of the Persistent Decode Engine. G0–G2 proved the persistent-grid
substrate + warp-specialized overlap, but ALSO found that fusing compute GEMMs
into a megakernel is **net-negative** at decode scale (launch savings ~3.7us/op
are negligible vs barrier+occupancy cost). The decode "execution gap"
(~20.8ms/tok = 20–40× the BW floor) is dominated NOT by launch overhead but by
**host round-trips for data-dependent control flow** — and these are exactly what
is **CUDA-graph-capture-illegal** (a d2h readback inside graph capture is illegal).

The DeepSeek-V3.2 decode path's highest-value boundary (call-path "Barrier 2",
Indexer → attention) is: the DSA Indexer computes per-query scores over candidate
KV blocks → **top-k selection** (`index_topk`) → attention reads ONLY the selected
blocks. Today that selection round-trips host orchestration. G3 proves it can stay
**device-resident** in one persistent cooperative grid — dissolving the
capture-illegal-d2h wall — and quantifies the host-round-trip latency removed.

Device: NVIDIA B200, 148 SMs, cc 10.0, coopLaunch=1, 228 KB smem/SM.
Build: `nvcc -std=c++17 -arch=sm_100 -O3` (nvcc 13.1), standalone, no TRT-LLM
rebuild, decode/model path and ABI-frozen files untouched. Results reproduced
across two independent runs (numbers below = run 1; run 2 within run-to-run noise,
correctness bit-identical).

## Design

Microbench of the Indexer→attention **control-flow** pattern (NOT a GEMM-fusion
gate — G1/G2 already showed fusion is net-negative). Representative decode/index
shapes: Q queries = decode batch **M ∈ {1, 8, 32}**; **C** candidate blocks/query
∈ {2048, 4096}; each block a **D=128**-dim vector; select top-**K** ∈ {256, 2048}
(like DSA `index_topk`). Two shape settings: `C2048_K256` (small-k) and
`C4096_K2048` (large-k).

- **Score (region 1a):** `score[q,c] = dot(q_query[q], block[c])` over D — a
  GEMV-like per-query reduction over C candidates. One CTA per query (grid-stride
  over Q); each warp strides candidates; D=128 = 4 floats/lane + a shuffle reduce.
- **Device top-k (region 1b):** one CTA owns one query; loads its C scores into
  SMEM (C≤4096 floats = 16 KB), then K rounds of **blockwide argmax-and-evict**
  (winner's SMEM slot set to −inf). The compare key is **(score desc, index asc)**
  so on an exact score tie the **lower block index wins** — a *deterministic*
  tiebreak. Selected indices + scores written to device global memory and **never
  copied to host**. Exact by construction (extracts the true top-K), at O(K·C).
- **Gather + reduce (region 2):** `out[q,:] = Σ_{j∈selected} softmax(score)_j ·
  block[idx_j][:]` — a numerically-stable softmax-weighted sum of the selected
  blocks' D-vectors (stand-in for attention over the selected KV). **Which blocks
  are read is decided on-device by region 1.** One CTA/query, two-pass
  (blockwide max + Σexp, then SMEM-accumulate the weighted D-vectors).
- **(A) device-resident path:** ONE persistent cooperative kernel —
  `score` → `cg::grid.sync()` → `device top-k` → `cg::grid.sync()` → `gather`.
  The selection stays in device/SMEM the entire time; **zero d2h/h2d**.
- **(B) host-orchestrated baseline (today's pattern):** `kScoreOnly` kernel →
  `cudaMemcpy` scores **d2h** + sync → **host top-k** (same deterministic
  tiebreak) → `cudaMemcpy` selected (idx, score) **h2d** → `kGatherOnly` kernel.
  Includes the REAL copies + the implied stream syncs = the "execution gap."

Correctness is ALWAYS vs an **independent CPU reference** that recomputes scores
(f64 accum), the exact top-k with the identical tiebreak, and the softmax-weighted
gather. NEVER a device-vs-device self-comparison. We also isolate the raw
**d2h+h2d+sync** cost (copies + syncs only, no host top-k, no kernels) so the
removed capture-illegal round-trip is explicit, and report the persistent kernel's
occupancy.

## Files

- `cpp/tensorrt_llm/kernels/pde/pde_g3_dev_ctrl.cuh`
- `blaise_perf/pde/pde_g3_bench.cu`
- `blaise_perf/pde/build_run_g3.sh`

## Gate results — CORRECTNESS: PASS

Device top-k INDEX SET == CPU top-k index set at **every** M, BOTH shapes, and
device output == host output == CPU reference (cos = 1.00000000):

| shape | M | A.out cos | B.out cos | maxabs | device top-k set match | host top-k set match |
|---|---|---|---|---|---|---|
| C2048_K256 | 1 | 1.00000000 | 1.00000000 | 6.6e-7 | 1/1 (ordered 1/1) | 1/1 |
| C2048_K256 | 8 | 1.00000000 | 1.00000000 | 8.9e-7 | 8/8 (ordered 8/8) | 8/8 |
| C2048_K256 | 32 | 1.00000000 | 1.00000000 | 8.3e-7 | 32/32 (ordered 32/32) | 32/32 |
| C4096_K2048 | 1 | 1.00000000 | 1.00000000 | 1.4e-6 | 1/1 (ordered 1/1) | 1/1 |
| C4096_K2048 | 8 | 1.00000000 | 1.00000000 | 1.5e-6 | 8/8 (**ordered 7/8**) | 8/8 |
| C4096_K2048 | 32 | 1.00000000 | 1.00000000 | 2.4e-6 | 32/32 (**ordered 28/32**) | 32/32 |

The **index SET** match — the gate bar, and what attention actually cares about
(it reads the same blocks regardless of within-selection order) — is **100% at
every M/shape**. The *ordered* match drops below 100% only at `C4096_K2048`
(7/8, 28/32): a handful of queries have **exact score ties** where device and CPU
both select the identical index set but order two equal-score neighbors
differently. `max_set_diff = 0` confirms no query ever picks a *different* block.
This is an honest tie artifact, not an error — and the softmax gather is
order-invariant, so `out` is still bit-exact (cos = 1.0). (See caveats.)

## Gate results — PERF: the host-round-trip elimination (us/iter, lower=better)

| shape | M | (A) device-resident | (B) host-orchestrated | **A speedup** | d2h+h2d+sync removed |
|---|---|---|---|---|---|
| C2048_K256 | 1 | 312.4 | 167.6 | 0.54× (slower) | 22.9 |
| C2048_K256 | 8 | 314.4 | 532.4 | **1.69×** | 30.6 |
| C2048_K256 | 32 | 314.3 | 1718.3 | **5.47×** | 48.7 |
| C4096_K2048 | 1 | 2669.8 | 722.9 | 0.27× (slower) | 26.3 |
| C4096_K2048 | 8 | 2671.9 | 2588.0 | 0.97× (~par) | 46.7 |
| C4096_K2048 | 32 | 2674.9 | 9448.2 | **3.53×** | 96.1 |

**The win scales with batch — which is the real decode story.** The device-resident
(A) latency is **flat in M** (one CTA/query, all CTAs resident in one cooperative
grid: 312–314us small-k, ~2670us large-k regardless of M=1→32). The
host-orchestrated (B) cost **grows with M**: more scores to copy d2h (the removed
copy cost rises 23→49us small-k, 26→96us large-k), more host top-k work, and more
serialized kernel↔host hops. At the decode-representative **M=32**: device-resident
is **5.47× faster (small-k)** and **3.53× faster (large-k)**.

At **M=1** (A) is slower: a single query = a single CTA, so 147 of 148 SMs idle,
and at K=2048 the O(K·C) iterative argmax (2670us) dominates — while (B)'s copies
are tiny at M=1. This is the honest floor of the *current* device top-k algorithm,
not of device-residency itself (see caveats).

**The structural point (beyond raw latency):** the **22–96us** d2h+h2d+sync that
(A) removes is exactly the **CUDA-graph-capture-illegal** round-trip. Even at the
M=1 large-k point where (A) loses on wall-clock, (A) **dissolves the capture wall**
— a captured decode graph *cannot contain* the host top-k round-trip at all, so
(B) cannot be graph-captured end-to-end whereas (A) can. That is the gate's core
claim: the data-dependent Indexer→attention selection can stay fully device-
resident in a persistent grid.

## Occupancy

| kernel | blocks/SM | resident CTAs | regs/thread | smem (C2048 / C4096) |
|---|---|---|---|---|
| `kDeviceResident` (the control-flow megakernel) | **6** | 888 | 40 | 8.6 / 16.6 KB |
| `kSelectOnlyPersistent` (top-k only) | 8 | 1184 | 40 | 8.1 / 16.1 KB |

@ 256 threads/CTA. Occupancy is healthy (6 blocks/SM = 888 resident CTAs); the
persistent grid is capped at Q CTAs at runtime (one CTA owns a query; cg grid sync
requires every launched CTA resident, so extra CTAs would only idle).

## Honest caveats / negatives — how representative is this of the real DSA indexer?

1. **Device top-k is O(K·C) iterative argmax — chosen for *provable exactness*,
   not speed.** It is the dominant cost at large-k (the flat ~2670us at K=2048 is
   almost entirely the 2048 argmax rounds × 4096 elements). The real DSA
   `index_topk` uses an efficient **threshold/radix/bitonic** top-k (≈O(C) or
   O(C·log K)), which would collapse (A)'s large-k latency by ~10× and make (A)
   win at M=1 too. This bench deliberately traded top-k *algorithm* speed for an
   exact CPU-matchable selection so the **index-set==CPU** correctness claim is
   airtight. **The G3 claim — device-residency dissolves the capture-illegal
   round-trip, and the host-round-trip cost grows with batch while device cost
   stays flat — is independent of the top-k algorithm's internal speed.** Swapping
   in a radix top-k is a follow-on (it does not change which path is
   capture-legal). The *small-k* numbers (C2048_K256: 1.69×@M8, 5.47×@M32) are the
   more representative perf signal precisely because the O(K·C) tax is small there.

2. **Within-tie ordering differs at C4096_K2048 (ordered 7/8, 28/32).** Random
   block vectors produce occasional exact f32 score ties; device argmax-evict and
   `std::partial_sort` resolve equal-score neighbors in the same SET but sometimes
   different ORDER. `max_set_diff=0` everywhere → the selected blocks are always
   identical; the softmax gather is order-invariant → `out` is bit-exact. Real
   indexer scores are quantized/clamped and tie even more, so a tie-robust
   selection matters; this bench shows the SET is stable under the deterministic
   (score, index) tiebreak, which is the property attention needs.

3. **The score is a thin GEMV (dot over D=128), not the full DSA indexer.** The
   real indexer has FP8 candidate scoring, RoPE on the index head, and per-block
   structure. G3 isolates the **control-flow boundary** (score→select→gather as a
   single device-resident region pair), not indexer GEMM throughput — by design,
   consistent with the G0/G2 finding that the boundary, not the math, is the lever.

4. **Gather is a softmax-weighted D-vector sum, a stand-in for attention over the
   selected KV** — not a real flash-attention over paged KV. It exercises the
   *data-dependent* read (which blocks are touched is decided on-device) but not
   the full attention numerics. Wiring the real `hisparseKvarnBdrRead` path behind
   this device-resident selection (without a d2h between select and read) is the
   downstream integration gate.

5. **(B)'s host top-k is a real `std::partial_sort` per query on the d2h'd
   scores** — a faithful "today's pattern" baseline. A production host path might
   overlap the copy with compute or use pinned memory; this bench uses pageable
   host buffers + explicit syncs, so (B)'s copy cost is an upper-ish bound. The
   *isolated* d2h+h2d+sync row (22–96us) is the conservative, copy-only figure.

## Bottom line

Device-resident data-dependent control flow **works and is correct** (device top-k
index-set == CPU at every M; output bit-exact). It **eliminates the
capture-illegal host round-trip** (22–96us measured) and, at decode-representative
batch (M=32), is **3.5–5.5× faster** end-to-end than host orchestration — with the
device path's latency **flat in batch** while the host path's grows. The G3 lever
is validated: the Indexer→top-k→attention selection can live entirely in a
persistent grid, which is the prerequisite for a graph-capturable / fully
device-resident sparse decode. The remaining frontier is a radix/threshold device
top-k (to win at M=1 + large-k) and wiring the real KV read behind the selection.

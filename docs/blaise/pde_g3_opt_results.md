# PDE G3-OPT — device-resident control flow now WINS at every batch size (REAL numbers, GPU0 / B200)

Optimization of Gate G3 of the Persistent Decode Engine. G3 proved that the
DeepSeek-V3.2 decode "Barrier 2" (Indexer → top-k → attention) can stay
**device-resident** in one persistent cooperative grid — dissolving the
CUDA-graph-capture-illegal host round-trip — and won big at batch (5.47×@M32
small-k). **But G3 LOST at M=1** (0.54× small-k, **0.27× large-k**) because its
device top-k was a simple **O(K·C)** blockwide argmax-and-evict (K rounds × C
elements) that dominates at large K, and 1 query = 1 CTA = 147 idle SMs. G3's own
caveat #1 predicted a radix/threshold top-k (~O(C)) would collapse that ~10× and
make device-residency win at M=1 too.

**G3-OPT does exactly that.** It replaces the device top-k with an **exact O(C)
64-bit composite-key MSD radix-select** and adds **multi-CTA-per-query**
parallelism. Result: device-resident is now **≥ 1.85× faster than host
orchestration at EVERY measured point**, including the M=1 large-k point that G3
lost — which goes from **0.27× → 2.33× (an ~8.6× swing)**. Correctness is
unchanged-or-better: device top-k index-set == CPU at every M/shape with
`max_set_diff = 0`, output bit-exact (cos = 1.00000000).

Device: NVIDIA B200, 148 SMs, cc 10.0, coopLaunch=1, 228 KB smem/SM.
Build: `nvcc -std=c++17 -arch=sm_100 -O3` (nvcc 13.1), standalone, no TRT-LLM
rebuild; decode/model path and ABI-frozen files (`SparseMlaDecodeKvarnHotOp.cpp`,
`hisparseKvarnBdrRead.cuh`) untouched. Numbers below = the canonical run
(`PDE_MAX_CPQ=16`); reproduced across runs within run-to-run noise, correctness
bit-identical.

## What changed vs G3

### 1. Device top-k: O(K·C) argmax-evict → exact O(C) 64-bit radix-select

Each score is mapped to a **monotone uint32 key** (`float_to_okey`: flip sign bit
for positives, invert all bits for negatives — larger float ⇒ larger key), then
packed with the **bit-inverted candidate index** into a **64-bit composite**:

```
key64 = (score_key << 32) | (~index & 0xffffffff)
```

Descending order on `key64` is **exactly (score DESC, then index ASC)** — the
identical deterministic tiebreak the CPU reference uses. Crucially, because the
index is unique, **all composites are DISTINCT**: the K-th largest composite is
unique, so the selected set is simply `{ key64 ≥ threshold }` with **no
boundary/tie special-casing at all**. This makes the index-set **provably**
identical to the CPU's `partial_sort` by the same key — the airtight
`max_set_diff = 0` bar.

The threshold is found by **MSD radix-select**: 8 passes of 8-bit digits over the
64-bit key. Each pass histograms the current digit over the still-active set
(active ⇔ the key's already-fixed high bits equal the running threshold prefix)
into a 256-bin histogram, walks the bins high→low to find the bin where the
cumulative count crosses the (reduced) K, and fixes that digit into the prefix.
After 8 passes the prefix IS the threshold key64. Cost: **O(8·C)** vs the old
**O(K·C)** — at K=2048, C=4096 that is ~512× less inner work.

### 2. Multi-CTA-per-query (small-M parallelism)

A **query group** of `cpq` CTAs cooperates on ONE query's score + radix-select.
The per-query 256-bin histogram lives in **global memory**, accumulated across the
group via atomics (each CTA first builds a private SMEM histogram, then flushes ≤
256 atomics to global — not one atomic per element). `grid.sync()` between radix
sub-phases makes the global histogram visible; every CTA then copies it into its
own SMEM and walks it identically (so all CTAs of a group derive the same
threshold deterministically). At M=1 the device works the single query with
`cpq` CTAs instead of 1 CTA / 147 idle SMs.

**Cooperative-grid lockstep:** every resident CTA issues the SAME number of
`grid.sync()`s. The radix-select is **wave-locked**: it loops a fixed
`ceil(Q / n_groups)` waves for ALL CTAs; groups with no query in a wave still hit
every barrier (gated by an `active` flag that suppresses their writes/atomics).
The bench sizes `n_groups ≥ Q` (one wave) for M≤32.

`cpq` heuristic: `cpq = clamp(grid_blocks / Q, 1, 16)`, i.e. more CTAs/query when
there are few queries to fill the device, fewer when the queries already fill it.
A measured sweep over the cap (below) showed M=1 is insensitive above ~16 and
larger caps HURT M=8/M=32, so the default cap is **16** (overridable via
`PDE_MAX_CPQ`).

Side effect: the radix top-k needs only a **256-int SMEM histogram** (1 KB), so
`kDeviceResident` SMEM dropped from C-scaling 8.6/16.6 KB to a **constant 1.5 KB**
(it no longer loads all C scores into SMEM) — strictly higher SMEM headroom.

## Gate results — CORRECTNESS: PASS

Device top-k INDEX SET == CPU at **every** M, BOTH shapes, `max_set_diff = 0`,
output bit-exact:

| shape | M | A.out cos | B.out cos | maxabs | device top-k set match | max_set_diff |
|---|---|---|---|---|---|---|
| C2048_K256 | 1 | 1.00000000 | 1.00000000 | 3.6e-7 | 1/1 | 0 |
| C2048_K256 | 8 | 1.00000000 | 1.00000000 | 4.5e-7 | 8/8 | 0 |
| C2048_K256 | 32 | 1.00000000 | 1.00000000 | 7.2e-7 | 32/32 | 0 |
| C4096_K2048 | 1 | 1.00000000 | 1.00000000 | 1.0e-6 | 1/1 | 0 |
| C4096_K2048 | 8 | 1.00000000 | 1.00000000 | 1.2e-6 | 8/8 | 0 |
| C4096_K2048 | 32 | 1.00000000 | 1.00000000 | 2.1e-6 | 32/32 | 0 |

**Ordered match is 0/M** for the device path now (vs G3's mostly-100%). This is an
*expected, honest* artifact of the parallel emit: qualifying candidates are
appended to the output in **candidate-scan order via an atomic fill counter**, not
in descending-score order. The SET is identical (`max_set_diff = 0`) — which is
the gate bar and all attention needs (it reads the same blocks regardless of
within-row order) — and the softmax gather is order-invariant, so `out` is
bit-exact (cos = 1.0). If a downstream consumer ever needs the selected list
*sorted*, a cheap per-row sort of the K emitted entries can be added; it is not
needed for the attention read. (This is strictly a stronger set-correctness story
than G3, whose ordered mismatches came from genuine score ties; G3-OPT's
distinct-composite threshold has no ties to resolve.)

## Gate results — PERF: device-resident now WINS at EVERY point

us/iter, lower = better; **speedup = host(B) / device-resident(A)**:

| shape | M | (A) device-resident | (B) host-orchestrated | **G3-OPT speedup** | G3 OLD speedup | crossed 1.0×? |
|---|---|---|---|---|---|---|
| C2048_K256 | 1 | 90.6 | 167.8 | **1.85×** | 0.54× (slower) | ✅ |
| C2048_K256 | 8 | 100.3 | 535.0 | **5.34×** | 1.69× | ✅ |
| C2048_K256 | 32 | 119.6 | 1732.6 | **14.48×** | 5.47× | ✅ |
| C4096_K2048 | 1 | 303.2 | 706.1 | **2.33×** | **0.27× (slower)** | ✅ **(the big win)** |
| C4096_K2048 | 8 | 337.9 | 2732.1 | **8.09×** | 0.97× (~par) | ✅ |
| C4096_K2048 | 32 | 381.6 | 9232.1 | **24.19×** | 3.53× | ✅ |

**Every point crossed 1.0×.** The M=1 large-k point that G3 lost at **0.27×** now
**wins at 2.33×** — the targeted ~8.6× swing. The win still grows with batch (the
real decode story): device-resident A stays nearly flat in M (90→120us small-k,
303→382us large-k) while host B grows steeply (more scores to d2h, more host
top-k, more serialized hops), reaching **14.5× / 24.2× at M=32**.

### Old-vs-new device top-k cost (the ~10× the doc predicted — beaten)

Isolated device top-k kernel time (cooperative select-only kernel):

| shape | M | **G3-OPT radix top-k** | G3 OLD O(K·C) argmax (≈ A's flat cost) | collapse |
|---|---|---|---|---|
| C2048_K256 | 1/8/32 | 51 / 61 / 77 us | ~310 us (flat) | ~4–6× |
| C4096_K2048 | 1/8/32 | 51 / 76 / 117 us | **~2640 us (flat)** | **~22–52×** |

At large-k the old top-k was ~2640us of A's ~2670us total; the radix-select does
the same selection in **51–117us — a ~22–52× collapse**, exceeding the doc's ~10×
prediction. The residual A latency at M=1 large-k (~303us) is now dominated by the
**score region + cooperative-launch/grid-barrier fixed overhead**, not the top-k —
which is why M=1 is insensitive to `cpq` above ~16 (see sweep).

### d2h+h2d+sync removed (the capture-illegal round-trip)

Unchanged structural win: the isolated copy+sync cost that (A) dissolves is
**23–93us** (rises with M: 23→51us small-k, 26→93us large-k). This is exactly the
CUDA-graph-capture-illegal host round-trip — (B) cannot be graph-captured
end-to-end, (A) can — independent of the top-k algorithm.

### cpq cap sweep (why default = 16)

Speedup (A-vs-B) at the cap values measured:

| point | cap=16 | cap=32 | cap=64 | cap=128 |
|---|---|---|---|---|
| C2048_K256 M=1 | 1.84× | 1.94× | 1.96× | 1.95× |
| C2048_K256 M=8 | 5.34× | 5.32× | 4.92× | 3.90× |
| C2048_K256 M=32 | **15.12×** | 12.09× | 11.99× | 12.00× |
| C4096_K2048 M=1 | 2.45× | 2.37× | 2.38× | 2.40× |
| C4096_K2048 M=8 | 8.56× | 7.84× | 7.39× | 7.12× |
| C4096_K2048 M=32 | **24.12×** | 22.62× | 23.98× | 22.47× |

M=1 is flat above ~16 (its floor is fixed launch/barrier overhead, not
candidate-parallel work); larger caps HURT M=8/M=32 (too many CTAs/query ⇒ more
barrier + global-histogram atomic contention once queries already fill the
device). cap=16 ties the higher caps at M=1 and is fastest at M=8/M=32.

## Occupancy

| kernel | blocks/SM | resident CTAs | regs/thread | smem |
|---|---|---|---|---|
| `kDeviceResident` (control-flow megakernel) | **6** | 888 | 40 | **1.5 KB (const)** |
| `kSelectOnlyPersistent` (radix top-k only) | **8** | 1184 | 40 | **1.0 KB (const)** |

@ 256 threads/CTA. SMEM is now constant (256-int histogram), not C-scaling, so
occupancy no longer degrades with C. The persistent grid is launched at
`cpq·Q ≤ 888` CTAs (one wave); `cpq = clamp(888/Q, 1, 16)`.

## Honest caveats / negatives

1. **Ordered output is candidate-scan order, not score-descending.** The parallel
   atomic-fill emit trades within-row ordering for parallelism. The index SET is
   exact (`max_set_diff=0`) and `out` is bit-exact (order-invariant softmax
   gather), so this satisfies the gate and the attention read. A downstream that
   needs a *sorted* selection would need a cheap per-row sort of K entries (not
   added — unnecessary for the KV read).

2. **M=1 is launch/barrier-bound, not compute-bound — `cpq` past ~16 buys
   nothing there.** A's M=1 floor (~90us small-k, ~303us large-k) is dominated by
   the cooperative-launch + 26 grid-barrier fixed cost and the score GEMV, not the
   radix-select (now 51us). Squeezing M=1 further means cutting *barrier count*
   (e.g. fusing the 3-barriers/pass to 2 via double-buffered histograms) or the
   cooperative-launch overhead — a separate lever from the top-k algorithm. Still,
   M=1 already WINS (1.85×/2.33×), which was the gate.

3. **3 grid barriers per radix pass (24 total).** Chosen for provable correctness
   (decoupling the histogram walk from rank0's reset so they cannot race within a
   group). Barriers are ~µs-scale and dwarfed by the ~22–52× top-k collapse, but a
   2-barrier double-buffered histogram is the obvious next micro-opt.

4. **Score is still a thin GEMV (dot over D=128) and gather is a softmax-weighted
   D-vector sum** — the same stand-ins as G3 (isolating the *control-flow*
   boundary, not the full DSA indexer FP8 scoring / RoPE nor real flash-attention
   over paged KV). The radix-select operates on f32 scores; real indexer scores
   are quantized — the composite-key approach is agnostic to the score dtype (any
   monotone key works), but a production wiring should key off the actual indexer
   score representation. Wiring the real `hisparseKvarnBdrRead` behind this
   device-resident selection (no d2h between select and read) remains the
   downstream integration gate.

5. **`out_cnt` defensive guard.** Emit guards `pos < K` — exactly K composites
   qualify by construction, so the guard never fires; it is belt-and-suspenders
   against a hypothetical mis-sized threshold. The per-query global histogram is
   self-reset to zero on kernel exit (every pass resets, including the last), so
   no host memset is needed between timing reps; `out_cnt` is reset at kernel
   entry (separated from the emit by 24 barriers for visibility).

## Bottom line

The G3 frontier item is closed: an **exact O(C) device radix top-k** +
**multi-CTA-per-query** makes device-resident data-dependent control flow **win at
EVERY batch size**, including the M=1 large-k point G3 lost (**0.27× → 2.33×**),
peaking at **24.2× @M32 large-k**. The device top-k collapsed **~22–52×** vs the
old O(K·C) argmax (beating the doc's ~10× estimate), with index-set bit-identical
to the CPU (`max_set_diff=0`) and output bit-exact. The Indexer→top-k→attention
selection is now fully device-resident AND fast enough to be the unconditional
choice — the prerequisite for a graph-capturable / fully device-resident sparse
decode. Next levers: cut M=1's barrier/launch floor (double-buffered histogram,
fewer barriers) and wire the real KV read behind the selection.

## Files

- `cpp/tensorrt_llm/kernels/pde/pde_g3_dev_ctrl.cuh`  (radix-select device top-k + multi-CTA-per-query)
- `blaise_perf/pde/pde_g3_bench.cu`  (scratch alloc, cpq sizing, isolated radix-topk timing, `PDE_MAX_CPQ` knob)
- `blaise_perf/pde/build_run_g3.sh`  (unchanged build+run wrapper)

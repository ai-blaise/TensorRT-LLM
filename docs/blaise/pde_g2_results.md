# PDE G2 — warp-specialized double-buffered cp.async OVERLAP: gate results

Gate for **G2** of the Persistent Decode Engine: hide the producer weight/activation
load latency that G1 left fully exposed, by running the DMA producer (warp 0)
`kStages-1` K-tiles AHEAD of the MMA consumers (warps 1–4) via a software-pipelined
cp.async ring — the first PDE gate that makes the persistent path actually FASTER.

Kernel: `cpp/tensorrt_llm/kernels/pde/pde_g2_overlap.cuh`; bench:
`blaise_perf/pde/pde_g2_bench.cu` (build/run `blaise_perf/pde/build_run_g2.sh`).
Standalone `nvcc -arch=sm_100` (nvcc 13.1); no TRT-LLM rebuild; decode/model path
untouched; ABI-frozen files untouched; builds on G0 substrate + G1 baseline.

## Pipeline design

- **N-stage cp.async ring (hand-rolled `commit_group` / `wait_group`).** Each region
  keeps `kStages` rotating SMEM buffers for A and W. Producer (warp 0) prefetches
  K-tile `ks+(kStages-1)` into ring slot `(ks+kStages-1)%kStages` while the MMA
  consumers wmma slot `ks%kStages`.
  - Prologue: producer issues+commits the first `kStages-1` tiles (one cp.async
    group each), **no wait**.
  - Steady state per k-step `ks`: producer issues+commits tile `ks+(kStages-1)`,
    then `cp.async.wait_group <kStages-1>` (leaves ≤ `kStages-1` groups in flight →
    the group feeding slot `ks` is guaranteed landed); `__syncthreads()`; consumers
    wmma slot `ks`; `__syncthreads()`.
  - Per-M-band drain: `cp.async.wait_group 0` before the ring is reused in the next
    16-row band, so the steady-state group count stays exact.
- **Why this does NOT repeat the G1 agent's `cuda::barrier` deadlock.**
  `cp.async.wait_group` is a **per-thread** instruction over *that thread's*
  committed groups; **only warp 0** issues/commits cp.async, so **only warp 0**
  waits — consumers never `wait_group`. Cross-warp visibility/ordering is carried
  entirely by the two plain `__syncthreads()` that **every** CTA thread executes
  uniformly (identical loop trip counts for all warps, M tiled in 16-row bands).
  The ring re-issues into a fixed slot each tile and the wait is *relative* to the
  outstanding count, so accounting self-resets per tile — no persistent phase state
  to drift. Verified correct across many work-queue tiles × B=1/8/32, no hang
  (clean rc=0).
- **`kStages` is a template parameter** → the bench instantiates and sweeps 2 and 3.
- Everything else inherited verbatim from G1: two regions, grid barrier, atomic
  work-queue, M-tiling in 16-row bands, L2 inter-region handoff, bf16
  `nvcuda::wmma` 16×16×16 (HMMA, no CUTLASS). The math is byte-identical to G1's
  `gemm_one_tile` — G2 only changes *when* the loads happen, not the arithmetic.

## Verification (GPU0 build-gate; orchestrator re-gates on GPU7)

```
device: NVIDIA B200  SMs=148  cc=10.0  coopLaunch=1  L2=126.5 MB  smemPerSM=228 KB

=== OCCUPANCY / PRESSURE (@ 160 threads) ===
  G1 single-buffer fused : 8 blocks/SM  (1184 CTAs)  regs=48  smem=5.8 KB
  G2 fused  2-stage      : 7 blocks/SM  (1036 CTAs)  regs=56  smem=11.5 KB
  G2 fused  3-stage      : 7 blocks/SM  (1036 CTAs)  regs=56  smem=17.2 KB
```

### Correctness — ALL vs the independent CPU f32 reference (`cpu_gemm`), cos ≥ 0.999999

| B  | s2 regionA vs CPU | s2 fusedB vs CPU | s3 fusedB vs CPU | separate vs CPU | overlap vs G1 single-buf | gate |
|----|-------------------|------------------|------------------|-----------------|--------------------------|------|
| 1  | 1.00000000        | 1.00000000       | 1.00000000       | 1.00000000      | 1.00000000               | PASS |
| 8  | 1.00000000        | 1.00000000       | 1.00000000       | 1.00000000      | 1.00000000               | PASS |
| 32 | 1.00000000        | 1.00000000       | 1.00000000       | 1.00000000      | 1.00000000               | PASS |

The overlapped output is bit-equivalent to the G1 single-buffer output AND to the
independent CPU f32 GEMM — overlap changed scheduling, not the result.

### Perf — the G2 point (us/iter, 100 reps, 20 warmup, lower = better)

| B  | G1 single-buf | G2 2-stage | G2 3-stage | separate (2 kernels) | best overlap/single-buf | best overlap/separate |
|----|---------------|------------|------------|----------------------|-------------------------|-----------------------|
| 1  | 332.27        | 274.20     | **272.59** | 233.42               | **1.219× faster**       | 1.168×                |
| 8  | 306.55        | 257.46     | **250.18** | 217.02               | **1.225× faster**       | 1.153×                |
| 32 | 561.09        | 484.62     | **473.18** | 401.27               | **1.186× faster**       | 1.179×                |

- **Overlap beats single-buffer at every B**: 1.19–1.23× (the gate metric). The
  exposed cp.async W/A load latency that made G1 slow is now hidden under the wmma.
- **3-stage > 2-stage at every B** (one more in-flight prefetch buys 1–2 pp despite
  +5.7 KB smem) → **best stage count = 3**.
- **Reverses the G1 regression.** G1 was 1.08–1.10× *slower* than two separate
  kernels; G2 closes that to 1.15–1.18× of separate — the persistent fused path is
  now within ~15–18% of the unconstrained two-kernel ceiling while keeping the
  on-chip region→region handoff (no HBM round-trip, no second launch).

### Occupancy / pressure

- 3-stage: 7 blocks/SM (1036 CTAs), 56 regs/thread, 17.2 KB dynamic smem. The extra
  ring buffers cost **one** block/SM vs G1's 8 (48 regs, 5.8 KB) — the overlap more
  than pays for the lost occupancy (net 1.19–1.23× faster). 2-stage has the same
  7 blocks/SM and regs but is slower, so depth (not occupancy) is the live lever
  here; 3 stages is the sweet spot at these decode-small M and B200's 228 KB/SM.

## Honest caveats / negatives

- **Still 1.15–1.18× slower than two separate kernels.** Overlap narrows but does
  not close the gap. The residual is the grid-barrier + persistent-grid
  serialization + the lower occupancy (7 vs 8 blocks/SM); separate kernels get
  fresh full-occupancy launches with no barrier. Closing the rest is the job of the
  later gates (device-resident control flow / warp-specialized cross-region overlap
  / the real tcgen05 path), per the G0 finding that the win is control-flow
  elimination + overlap, not fusion for its own sake.
- This is the **bf16 `nvcuda::wmma` substrate**, not the frozen FlashMLA
  tcgen05/UMMA + NVFP4 path. G2 proves the overlap *mechanism* is correct and
  net-positive on the standalone substrate; porting the ring onto the production
  tensor-core path is a later gate.
- Persisting-L2 timing remains inconclusive at this scale (carried from G1): C_A is
  448 KB = 0.35% of L2, so normal reuse already holds it; the size argument is the
  evidence, pinning becomes load-bearing only under L2 contention.
- 4+ stages not swept (3 already best and smem grows linearly); revisit only if a
  larger tile or the production path changes the latency/occupancy balance.

## Carried forward to G3

G2 gives a correct, net-positive intra-region producer/consumer overlap. The
remaining gap to the separate-kernel ceiling is barrier + launch + occupancy
overhead → G3 targets device-resident control flow (dissolve the cooperative
relaunch/barrier cost) and cross-region overlap, on the path to the whole-decode
megakernel.

---

## Orchestrator audit correction (independent GPU7 re-gate)

Independently re-gated on GPU7: correctness PASS (cos 1.0 vs CPU + vs G1, all B);
overlap-vs-G1-single-buffer reproduces at **1.19–1.25×** (GPU7: B1 1.246×, B8
1.219×, B32 1.189×). The double-buffered cp.async ring is a real, correct overlap.

**Correction to the "reverses the G1 regression" claim above:** that comparison is
not supportable. The G1 bench measured the 2-separate-kernel baseline at ~301 µs
(B=1) while the G2 bench measures it at ~233 µs — the separate baseline shifted
between benches (GPU clock/warm state), so `overlap/separate` (1.15–1.18×) and
G1's `fused/separate` (1.08–1.10×) are **not** comparable across benches, and G2
did **not** close the gap to separate. The only valid, same-bench claim is the
overlap-vs-single-buffer win (**1.19–1.25×**).

**The honest, load-bearing finding:** even with overlap, the fused persistent path
is still ~1.15–1.18× **slower than two separate kernels** for these representative
*compute-bound* decode GEMMs at this scale. The launch overhead the fused path
saves (~3.7 µs/op, per G0) is negligible against ~250–560 µs kernel times, while
the grid-barrier + reduced occupancy (7 vs 8 blocks/SM) are not. So **fusing
compute GEMMs into a persistent megakernel is net-negative at decode scale.**

This reinforces the G0 strategic finding and sharpens G3: the persistent engine's
value is **not** fusing GEMMs — it is (a) **device-resident control flow** that
eliminates the per-step host round-trips these microbenchmarks don't even contain,
and (b) overlapping the real **latency-bound** production kernels (the FlashMLA
hot-read, ~9 ms/step, which is latency- not compute-bound). G2 has proven the
overlap *mechanism* is correct and net-positive over a non-overlapped baseline;
the engine-level win must come from G3 (device control flow) applied to the
actual latency-bound decode path, not from fusing representative GEMMs.

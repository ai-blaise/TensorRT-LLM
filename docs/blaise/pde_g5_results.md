# PDE Gate G5 — Full-Layer-Stack Persistent Megakernel (cross-layer weight prefetch)

**Node:** a4-us-001-rl9 (GPU 0, NVIDIA B200, sm_100, nvcc 13.1)
**Branch:** op-trt-pde · **Image:** `…optrt-aaa7e2b542b2-hisparse-current-head-proof-20260613T155354Z`
**Status:** correctness gate **PASS** (all N, both M regimes) · perf finding **HONEST NET-NEGATIVE at decode scale** (see below)

Standalone microbench, model-free. ABI-frozen files untouched. Decode-realistic shapes
(H=7168, Kp=512, Nm=2048; 35.0 MB weights/layer in bf16). Composes the validated PDE
primitives: G0 substrate (cooperative grid + barrier), G1 GEMM region body, G4 het-worker
COPY∥COMPUTE overlap (lifted from KV-staging to **weight** staging), G9 device-resident
multi-unit persistence (lifted from cross-STEP to cross-LAYER).

## Design

**One persistent cooperative launch runs N synthetic decoder layers.** Each layer is two
weight-heavy GEMMs chained — GEMM-1 (attn out-proj, Kp→H) then GEMM-2 (MoE FC1, H→Nm) —
followed by a fixed deterministic contraction back to Kp (a residual-like down-projection
with a bounded `v/(1+|v|)` squash) so the activation recurs stably across up to 61 layers.
The dominant byte traffic per layer is the **weights** (Kp·H + H·Nm bf16 = 35 MB); the
decode batch M is small (16), so the layer is weight-streaming-bound — the regime where
weight prefetch is supposed to pay.

- **N-layer device-resident execution (cross-LAYER analog of G9):** the host launches ONCE.
  Layers loop *inside* the kernel; `cg::grid.sync()` separates the two GEMMs and the
  contraction and joins the layer boundary (4 grid barriers/layer + 1 prime, identical count
  on every CTA in both prefetch ON/OFF paths). Activations hand off layer→layer through
  device buffers (X/Xn ping-pong, Y1/Y2 scratch) kept resident — **no per-layer host launch
  or metadata rebuild.**
- **Cross-layer weight prefetch (cross-LAYER analog of G4 het-workers):** the resident grid is
  split CONTIGUOUSLY — the top `n_prod` CTAs are a PRODUCER group that streams layer **L+1**'s
  weights (GMEM→a double-buffered "hot-weight ring" slot) while the bottom CTAs are the
  COMPUTE group running layer **L**'s GEMMs out of the *active* ring slot. Producers issue the
  L+1 stream then enter the layer-compute path with `participate=false` (they skip the GEMM but
  still hit every `grid.sync()` in lockstep, so the cooperative barrier stays well-formed and
  the stream OVERLAPS the compute). After the join barrier the ring slots swap. This is the
  **unblocked** analog of the dependency-blocked KV prefetch: L+1's weight addresses are static
  and known a layer ahead.
- **Why not GEMM-fusion-for-its-own-sake:** per the G2 finding, fusing compute GEMMs purely to
  delete launches is net-negative at decode scale. G5 therefore isolates the *prefetch* lever
  (ON vs OFF) from the *launch-elim* lever (megakernel-noPF vs per-layer baseline).

**Correctness method (never a self-compare):** three independent device paths — megakernel
prefetch-ON, megakernel prefetch-OFF (GMEM-direct), and a per-layer-launch baseline (3N
separate non-cooperative launches, host swaps activation buffers, weights cold from GMEM each
layer) — plus an independent f64 CPU reference of the same N-layer recurrence.

## Gate results

### Correctness — PASS (all configs)

Two layers of evidence:
- **device-vs-device (the integration gate):** prefetch-ON vs prefetch-OFF megakernel f32
  outputs are **BIT-EXACT** (0 mismatched words at every N and M), and megakernel-vs-baseline
  **cos = 1.00000000**. The prefetch path does not perturb the result vs GMEM-direct or vs the
  relaunch baseline.
- **device-vs-CPU (corroboration):** all device paths match the independent f64 reference at
  bf16-natural precision (a deep bf16 two-GEMM/layer stack compounds rounding the f64 ref does
  not): cos 0.99999272 (N=4) · 0.99992781 (N=16) · 0.99963637 (N=61) · 0.99992449 (M=128).

| config            | N  | M   | dev↔dev (f32 mism / cos) | dev↔CPU cos | gate |
|-------------------|----|-----|--------------------------|-------------|------|
| N4_M16_decode     | 4  | 16  | 0/8192 · 1.00000000      | 0.99999272  | PASS |
| N16_M16_decode    | 16 | 16  | 0/8192 · 1.00000000      | 0.99992781  | PASS |
| N61_M16_decode    | 61 | 16  | 0/8192 · 1.00000000      | 0.99963637  | PASS |
| N16_M128_reuse    | 16 | 128 | 0/65536 · 1.00000000     | 0.99992449  | PASS |

### Performance — megakernel NET-LOSES at decode scale; prefetch is NOT the hoped-for lever

us/iter on GPU0 (cooperative megakernel = 1 launch; baseline = 3N launches). Three ratios:
**launch-elim** = baseline / megakernel-noPF (the persistence lever alone); **weight-prefetch**
= megakernel-noPF / megakernel-PF (the prefetch lever alone); **integrated** = baseline /
megakernel-PF.

| config         | baseline | mega noPF | mega PF | launch-elim | weight-prefetch | integrated |
|----------------|---------:|----------:|--------:|:-----------:|:---------------:|:----------:|
| N4_M16_decode  |    89008 |    129762 |  124360 | **0.686×**  |   **1.043×**    | **0.716×** |
| N16_M16_decode |   354785 |    515862 |  496366 | **0.688×**  |   **1.039×**    | **0.715×** |
| N61_M16_decode |  1353288 |   1964279 | 1887375 | **0.689×**  |   **1.041×**    | **0.717×** |
| N16_M128_reuse |  1296163 |   1830738 | 4522469 | **0.708×**  |   **0.405×**    | **0.287×** |

**Occupancy:** megakernel 4 blocks/SM (592 CTAs, 96 regs, 5888 B dynamic smem); per-layer
baseline GEMM 8 blocks/SM (1184 CTAs). The cooperative monolith runs at **half** the
baseline's occupancy.

## Honest read of the result

1. **Launch-elimination alone is net-NEGATIVE (~0.69×, i.e. ~31% slower).** Folding 3N host
   launches into one persistent cooperative launch *loses* at decode scale: the per-layer host
   launch it deletes (~a few µs) is dwarfed by (a) the cooperative monolith's halved occupancy
   (4 vs 8 blk/SM) and (b) the 4 `grid.sync()`/layer barrier cost across the full resident
   grid. This is exactly the G2 finding and the standing memory note ("launch-elim only
   ~5%/step → real win is device-control-flow + warp-specialized overlap"), now confirmed at
   full-stack scale: launch-elim is not ~+5%, it is **−31%** once the occupancy/barrier cost of
   a real N-layer monolith is paid.

2. **Cross-layer weight prefetch does NOT materialize as a win.** At decode batch (M=16) it is a
   marginal **+4%** (noPF→PF), and at the weight-reuse regime (M=128) it **inverts to −59%
   (0.41×)**. The reason is structural and is the load-bearing G5 finding:
   - In a weight-BW-bound decode layer the **GEMM already streams the weights from HBM exactly
     once** (M tiny ⇒ each weight read ~once). Pre-staging L+1 to a ring slot is a *redundant*
     read; the only thing it can buy is overlapping L+1's HBM fill under L's compute.
   - But carving a producer group out of a **fixed cooperative grid steals compute CTAs** from
     the GEMM (here 148 of 592, −25%). At M=16 the GEMM is so trivial that the lost CTAs barely
     matter and the overlap nets +4%; at M=128 the GEMM has 8 m-bands and *is* the bottleneck,
     so losing 25% of the compute group (plus the redundant 35 MB/layer stage traffic) **more
     than doubles** the layer time. The prefetch helps only in the exact regime where there is
     nothing worth hiding it under, and hurts the moment compute matters.

3. **The integrated full-stack megakernel loses to the per-layer baseline everywhere**
   (0.72× decode, 0.29× reuse). The +4% prefetch nudge at decode does not come close to
   recovering the −31% launch-elim/occupancy/barrier penalty.

**Verdict:** G5 is a clean correctness PASS (the N-layer device-resident execution + cross-layer
prefetch are bit-exact vs both an independent baseline and a CPU reference). The distinct G5
*performance* thesis — that cross-layer weight prefetch hides the dominant weight-load latency —
**does not hold** for the BW-bound decode regime on B200: the GEMM is already the weight stream,
and a dedicated producer group is a net loss of compute parallelism under a fixed cooperative
occupancy. The real decode lever remains warp-specialized intra-region overlap (G4) and
device-resident control flow (G3/G9/G8), **not** stacking the whole forward into one
grid-barrier'd monolith. Honest negative — recorded so the campaign does not re-pay this cost.

## Reproduce

```
flock -n /tmp/gpu001_lock_a timeout -s KILL 220 docker run --rm --init \
  --name pde_g5_run --gpus all -e CUDA_VISIBLE_DEVICES=0 \
  -v /home/spencer/work/pde-wt:/wt --entrypoint /bin/bash <proof-image> \
  /wt/blaise_perf/pde/build_run_g5.sh
# G5_ONLY=<0..3> re-times a single config in isolation (the N=61 / M=128 configs are slow).
```

Files: `cpp/tensorrt_llm/kernels/pde/pde_g5_fullstack.cuh`,
`blaise_perf/pde/pde_g5_bench.cu`, `blaise_perf/pde/build_run_g5.sh`.

---

## Orchestrator GPU7 re-gate (independent)

Re-ran the full G5 bench on a reserved GPU (physical GPU7).
- **Correctness: PASS reproduced** — dev-vs-dev bit-exact (mega prefetch-ON vs OFF: 0 mismatched f32 words at every N/M; mega vs per-layer baseline cos = 1.00000000) and dev-vs-CPU cos 0.99963637 (N61) / 0.99992449 (M128) etc.
- **Headline verdict reproduced:** integrated megakernel **LOSES** (megawin 0.719-0.723x across all configs); **launch-elim 0.695-0.698x (-31%)** — confirms the monolith's halved occupancy (4 vs 8 blk/SM) + grid-barriers exceed the per-layer launches it deletes.
- **One discrepancy recorded:** the M128 "reuse" **weight-prefetch** sub-attribution is **1.035x on GPU7**, NOT the 0.41x reported on GPU0 above — the GPU0 0.41x is a run-specific artifact (not reproduced). On GPU7 weight-prefetch is uniformly **marginal (~1.03x at every config)**, i.e. it neither helps nor actively hurts. This does not change the conclusion: **the cross-layer-weight-prefetch thesis does not hold (at best ~3%), and the integrated full-stack megakernel net-loses to the per-layer baseline.** The engine's real levers remain device-resident control flow (G3/G9/G8) + warp-specialized intra-region overlap (G4), not a full-stack grid-barriered monolith.

# PDE G1 — two-region warp-specialized persistent slice: gate results

Gate for **G1** of the Persistent Decode Engine: prove that two real tensor-core
GEMM regions can run under ONE persistent cooperative grid, warp-specialized, with
a grid-barrier between them and the inter-region activation handed off on-chip,
bit-correct vs an independent reference.

Kernel: `cpp/tensorrt_llm/kernels/pde/pde_g1_two_region.cuh`; bench:
`blaise_perf/pde/pde_g1_bench.cu` (build/run `blaise_perf/pde/build_run_g1.sh`).
Standalone `nvcc -arch=sm_100`; no TRT-LLM rebuild; decode/model path untouched;
ABI-frozen files untouched; builds on the G0 substrate (`pde_substrate.cuh`).

## Design

- **Region A** ("attn output proj"): C_A[M,512]·W_A[512,7168]; **grid barrier**;
  **Region B** ("MoE FC1"): C_B[M,7168]·W_B[7168,2048]. bf16 in, f32 accumulate.
- Warp-specialized per CTA: warp 0 = `cp.async` DMA producer; warps 1–4 = wmma
  16×16×16 MMA consumers (each owns a 16-wide N sub-tile). Single-buffer
  producer→consumer handoff via `__syncthreads()`. N-tiles distributed across the
  resident grid by the G0 atomic work-queue (fused) / grid-stride (reference).
  M (batch) tiled in 16-row bands so B>16 is correct.
- Region B reads C_A from L2 (host pins a persisting-L2 window on C_A).

## Verification (orchestrator, GPU0 build-gate + GPU7 independent re-gate — identical)

```
device: NVIDIA B200  SMs=148  cc=10.0  L2=126.5 MB
OCCUPANCY: fused 8 blocks/SM (1184 CTAs, 48 regs)  ==  single-region 8 blocks/SM (48 regs)

                 regionA vs CPU      fusedB vs CPU       fusedB vs SEP
  B=1   (M=1)    cos 1.00000000      cos 1.00000000      cos 1.0  PASS
  B=8   (M=8)    cos 1.00000000      cos 1.00000000      cos 1.0  PASS
  B=32  (M=32)   cos 1.00000000      cos 1.00000000      cos 1.0  PASS

G1 GATE: PASS (correctness)   [GPU0 and GPU7, bit-identical results]
```

All correctness is vs an **independent CPU f32 GEMM** (operands pre-rounded to
bf16 so the reference sees the same operands). The fused-vs-separate check is a
secondary consistency check, not the reference.

- **Occupancy:** the fused two-region kernel resides at 8 blocks/SM = 1184 CTAs,
  the SAME as the single-region kernel (48 regs/thread, 5.8 KB dynamic smem). No
  occupancy collapse from fusing the two regions — the central G1 risk did not
  materialize at this tile size.
- **L2 handoff:** C_A = 448 KB = **0.346% of the 126.5 MB L2** → the region-A→B
  activation fits L2 with enormous margin and never needs HBM. (The
  persisting-window-vs-no-persist timing came out 1.00× — inconclusive — because
  normal L2 reuse already holds 448 KB at this scale; the *size* argument is the
  real evidence, the pinning becomes load-bearing only when concurrent working
  sets contend for L2.)
- **Perf:** fused two-region is **1.08–1.10× SLOWER** than two separate kernels.
  This is expected and correct for G1: the fused kernel pays the grid-barrier and
  persistent-grid serialization **without the overlap benefit yet**. G1's gate is
  structure + correctness; the perf win comes from G2 (warp-specialized producer
  prefetch overlap) and G3 (device-resident control flow). This directly
  reaffirms the G0 finding: the value is in overlap + control-flow elimination,
  not fusion for its own sake.

## Bugs found and fixed (orchestrator finish after the impl agent stalled)

The implementation subagent wrote the kernel + bench but **stalled/was killed
without ever building, gating, or committing** (it sat idle ~1h45m). The
orchestrator audited the ground truth and finished it, finding two real bugs the
agent never caught because it never ran the code:

1. **Producer/consumer `cuda::barrier` deadlock.** The full/empty barriers
   (arrival count = blockThreads) desynced: the producer arrived on `empty` only
   when `ks>=kStages`, and the barriers were never reset across work-queue tiles,
   so phase accounting drifted and a waiter blocked forever (confirmed: a prior
   run hung 32 min at 100% GPU). The double-buffer gave no overlap anyway (all
   threads met every k-step), so it was replaced with a **single-buffer
   `__syncthreads()` handoff** — robustly correct. Real pipelined overlap is G2.

2. **M>kTileM correctness bug.** `kTileM=16` but B=32→M=32; the kernel computed
   only the first 16 rows (`regionA vs CPU cos=0.70`), while **fused-vs-separate
   still read cos=1.0** (both variants shared the bug). Caught **only by the
   independent CPU reference** — the canonical reason never to self-compare. Fixed
   by tiling M in 16-row bands inside `gemm_one_tile`.

## Process correction (deployment + monitoring)

- **Deployment:** runs are now `timeout -s KILL` bounded (SIGTERM cannot kill a
  hung cooperative kernel stuck in a GPU ioctl), in a `--init` named container so
  it can be force-killed, backgrounded so no SSH dependency.
- **Monitoring:** active in-turn polling of the live remote log + GPU util +
  container state on a short cadence, with terminal-marker / container-gone exit —
  not fire-and-forget waiting on a completion notification. This caught the
  deadlock (and the reclaim of GPU0) immediately instead of after ~1h45m.

## Carried forward to G2

The fused kernel's single-buffer `__syncthreads` handoff is correct but
non-overlapping. G2 = warp-specialized **double-buffered overlap** (producer
`cp.async` of tile k+1 hidden under the MMA of tile k, via correct mbarrier/
`cuda::pipeline` accounting) — the first gate that should make the persistent
path actually faster, per the G0 finding.

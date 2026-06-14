# PDE G0 — persistent-grid substrate: gate results

Gate for **G0 (the persistent-grid substrate)** of the Persistent Decode Engine
plan. Substrate: `cpp/tensorrt_llm/kernels/pde/pde_substrate.cuh`; microbench:
`blaise_perf/pde/pde_g0_bench.cu` (build/run `blaise_perf/pde/build_run_g0.sh`).

Standalone `nvcc -arch=sm_100` build inside the proof image; no TRT-LLM rebuild,
no change to the decode/model path, ABI-frozen files untouched.

## Verification (orchestrator-independent, GPU 7)

Built + run fresh on a reserved GPU (physical GPU 7), separate from the build GPU.

```
device: NVIDIA B200  SMs=148  cc=10.0  coopLaunch=1
grid plan: blocks_per_sm=8  resident_CTAs=1184  block_threads=256  (303104 threads)

[CorrA cg.grid_sync]     got=728791040   ref=728791040   -> PASS
[CorrA global-barrier]   got=728791040   ref=728791040   -> PASS
[CorrB work-queue]       items=100000  exactly-once-violations=0  checksum match -> PASS
[CorrC mbarrier-mailbox] tiles=256x512  got=1073676288  ref=1073676288  -> PASS

PERF (per region-boundary, 200 reps x 64 boundaries):
  (i)   K separate kernel LAUNCHES : 3.7174 us/boundary
  (ii)  in-kernel global barrier   : 3.1649 us/boundary   (1.17x cheaper)
  (ii') in-kernel cg.grid_sync     : 2.6101 us/boundary   (1.42x cheaper)  <- SHIPPED
PERF GATE: in-kernel barrier cheaper than separate launches -> PASS

G0 GATE: PASS
```

All correctness is vs an independent CPU reference (the K-region reduction is
barrier-sensitive: a dropped peer-CTA write changes the result, so an exact match
proves the barrier). No fused-vs-sequential self-comparison.

## What G0 establishes

- The **persistent cooperative grid** sizes correctly to full residency (8 CTAs/SM
  x 148 SMs = 1184 CTAs) and the **grid-wide region barrier** is correct under both
  impls. `cg.grid_sync` (2.61 us) is shipped as the default; the hand-rolled global
  arrival/release barrier (3.16 us, sense-reversal + release/acquire fences) is kept
  for the case where the megakernel's resource envelope makes a cooperative launch
  too restrictive.
- The **warp-role scaffold** (mbarrier producer->consumer mailbox) and the **atomic
  work-queue** (no double-handout, no drops) are correct — the substrate the
  warp-specialized megakernel and dynamic MoE routing build on.

## Strategic finding (re-weights the later gates)

The barrier-cheaper-than-launch premise holds, but the **margin is modest**:
~1.1 us/boundary saved (cg) over a 3.72 us launch. Across ~976 op-boundaries/step
(~16 ops/layer x 61 layers), pure launch-elimination saves only ~1 ms of the
~20.8 ms decode step (~5%).

So the 20-40x execution gap is **not** dominated by raw kernel-launch overhead. It
is dominated by (a) **host round-trips for data-dependent control flow** (the
indexer top-k / hisparse planner / metadata rebuilds replayed each step) and (b) the
**latency/occupancy-bound compute kernels themselves** (the FlashMLA hot-read alone
is ~0.147 ms/call x 61 = ~9 ms/step). Therefore the engine's headroom is in:

1. **G3 — device-resident control flow** (eliminate the host syncs), highest value.
2. **Region fusion** (fold RMSNorm/RoPE/KV-write/dequant into the big kernels).
3. **Warp-specialized overlap** of the compute-heavy regions (attention hot-read,
   MoE megakernel, comms) via heterogeneous workers + continuous (weight) prefetch.

Launch consolidation is a real but secondary (~5%) slice. This sharpens the build
order: front-load device control flow and the attention-region megakernel + overlap;
treat launch-elimination as a free side-effect, not the goal.

## Notes / minor cleanup carried forward

- Two benign nvcc warnings (#20054-D, dynamic init of a function-scope static
  `__shared__` `cuda::barrier`); functionally correct (CorrC passes), but the
  megakernel should use the explicit `mbarrier` init pattern to silence them.
- The substrate's `plan_persistent_grid` returns exactly the resident CTA count
  (`sm_count * occupancy`); the "clamp to coop limit" comment is descriptive — the
  value is already the launchable cooperative grid size (verified: max_coop_grid==1184).

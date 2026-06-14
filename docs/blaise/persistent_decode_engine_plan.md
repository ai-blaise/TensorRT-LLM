# Persistent Decode Engine for DeepSeek-V3.2-REAP-NVFP4 on 8xB200 — Build Plan

This document is the build plan for a persistent, warp-specialized decode
megakernel for `BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4` served via `op-trt`.
It instantiates TileRT's "First Leap" execution-model philosophy (a persistent
engine + tile-level pipelining + warp specialization + heterogeneous workers)
for this model and 8xB200, and folds the "Second Leap" co-design
(NVFP4-experts/FP8-rest, fused dequant, microsecond auxiliary-op triage,
DFlash/MTP) in as native regions rather than bolt-ons.

It is grounded on:

- This campaign's own decode measurement: `~20.8 ms/tok, 20-40x the
  memory-bandwidth floor` (decode is overhead-bound, not BW-bound).
- Direct reads of the TileRT blog
  (`https://www.tilert.ai/blog/breaking-1000-tps.html`, "The First Leap: The
  Execution Model Revolution") and the Cursor WarpDecode blog
  (`https://cursor.com/blog/warp-decode`).
- The existing `op-trt` custom decode stack (the FlashMLA hot-read decode,
  WarpDecode MoE, the DSA Indexer/HISA, HiSparse, LayerSplit, SMC-SD, the
  NVFP4 dense KV / KVarN paths, and the `SimpleScheduler` admission seam).

The branch `op-trt-pde` (off `op-trt-hisparse`) is where this work is built.

## Objective and the central thesis

Consolidate the entire per-token decode forward — MLA+DSA attention, MoE, the
auxiliary ops, the dynamic control flow, and the intra-node collectives — into
one continuously-resident grid, replacing the current operator-by-operator
launch path on the decode side.

**The quantitative justification is already in our own data.** Decode runs at
`~20.8 ms/tok, 20-40x the BW floor`. ~95% of every decode token's wall-clock is
not weight movement; it is the "Execution Gap": kernel-launch latency,
host<->device round-trips for data-dependent decisions, hardware syncs,
metadata rebuilds, and activation round-trips to HBM between isolated kernels.
No single-kernel optimization touches that 20-40x multiple, because it does not
live inside any one kernel — it lives in the seams between them. The persistent
engine is the only lever in the campaign whose headroom is that whole multiple.

**The model-specific thesis (why this fits DeepSeek-V3.2 in particular).** The
model is saturated with data-dependent sparsity: the DSA Indexer picks top-k KV
blocks per layer, the MoE router picks top-k experts per token, HiSparse picks
which cold blocks to swap in per step, MTP/DFlash accepts a variable token
count. Each of those decisions currently forces either a host round-trip or a
device-resident schedule that cannot be expressed under CUDA-graph capture
(capture-illegal d2h readback — the exact wall the HiSparse G1/G2 work hit). A
persistent kernel does device-side control flow natively: it can branch on a
device value, consume a device-resident schedule, and loop, with no graph and
no host sync. So the persistent engine is not merely a launch-overhead
optimization — it is the architecture that makes the full dynamic-sparse decode
device-resident, dissolving the graph-safety constraints that currently cap
HiSparse, the Indexer, and SMC-SD. That is the through-line of this plan.

## Scope decisions (the non-negotiable framing)

- **Decode-only engine.** Prefill is compute-bound; its big GEMMs already
  amortize launch overhead, and the WarpDecode blog's own logic says
  expert-centric packing wins there. The engine targets the decode regime
  exclusively — the low-batch, latency-first, overhead-bound path. Prefill keeps
  the current expert-centric / graph-captured path. This mirrors both TileRT
  (ultra-low-latency focus) and the r20 disagg split (TP2xCP2 LayerSplit
  prefill / TP4 decode). The engine is the decode half of the disagg
  deployment.
- **New path beside the frozen ABI, not a modification of it.**
  `SparseMlaDecodeKvarnHotOp.cpp` and `hisparseKvarnBdrRead.cuh` stay frozen and
  become the reference oracle and fallback. The megakernel is a parallel decode
  path selected by config; the per-op path remains the bit-exact truth source we
  gate against. Never self-compare fused-vs-sequential — always gate vs the true
  dense f32 reference.
- **Single-GPU persistent engine before intra-kernel collectives.** In-kernel
  TP/EP communication (NVLink-SHARP `multimem` / NVSHMEM device API) is the
  highest-risk component. Get the entire intra-GPU megakernel win first (correct
  on 1 rank), then add in-kernel comms, with a clean fallback to launch-boundary
  collectives if device-side comms is not robust.
- **Target the TP4 decode engine** (the disagg decode side, equivalently the TP4
  group inside the DP2/TP4 `64/49/41/40` single-node winner). Baselines to beat:
  that DP2/TP4 curve and the r20 disagg decode TPOT.

## The architecture

### 1. The substrate: persistent grid + warp specialization + heterogeneous workers

Launch once and stay resident. All SMs (occupancy-tuned — confirm the exact
resident-CTA count and per-SM SMEM budget on the target B200 SKU at G0) are
claimed by a single cooperative grid that lives for the whole decode step, and
ultimately across steps. Two orthogonal specializations:

- **Warp specialization within a CTA** (the producer/consumer pattern,
  Hopper/Blackwell-native): named warp-groups take fixed roles for their
  lifetime. DMA/producer warps drive TMA bulk loads (weights, KV tiles)
  GMEM->SMEM ahead of compute; MMA warps issue `tcgen05` tensor-core ops
  consuming SMEM, accumulating in TMEM; epilogue/reduction warps do
  dequant-expand, SwiGLU, RMSNorm-fold, requant-write; comms warps issue
  collectives. Coordination is via named barriers (`mbarrier`) and an SMEM
  mailbox — no cross-warp global sync on the hot path. This is the shape our
  kernels already have: the FlashMLA hot-read decode's head-grouped flash
  (dequant K/V once, shared across the head group) is a producer/consumer split,
  and WarpDecode's "each warp owns one output scalar for its lifetime" is warp
  independence. We are generalizing shapes we already validated.
- **Heterogeneous Workers across SM groups** (what makes a whole-layer
  megakernel feasible). A whole MLA+MoE layer's working set does not fit one
  CTA's SMEM, so we do not ask it to. The grid is partitioned into specialized
  SM groups: an Indexer group, an attention group (holds KV/latent tiles), a MoE
  group (holds expert-weight tiles), a copy group (HiSparse cold<->hot swap), and
  a comms group. Each group's per-SM working set stays bounded; groups hand off
  through L2-resident activation scratch, never through HBM. A device-side work
  queue assigns dynamic work (which expert, which KV block) to SMs within a group
  via atomic work-stealing — this absorbs MoE routing imbalance. LayerSplit
  already explored SM-group partitioning; it informs the static split, and REAP's
  reduced expert count shrinks the MoE group's pressure (a tailwind).

```
            +--------------- persistent cooperative grid (all SMs) ---------------+
 device      |  [Indexer SMs] -> [Attention SMs] -+         +-> [MoE SMs] -> ...    |
 request --->|        ^              ^   (L2 act scratch)    |     ^                |  --> completion
 queue       |   [Copy SMs: HiSparse hot/cold swap, overlapped]   [Comms SMs: NVLS]|      ring
            +----------------------------------------------------------------------+
   grid-wide barrier between dependent regions; warp-specialized producer/consumer within each SM
```

The grid-wide barrier between dependent regions is either cooperative-groups
`grid.sync()` (clean, but couples to a cooperative launch and a fixed resident
occupancy) or a hand-rolled global-memory arrival barrier (atomics +
release/acquire) if the warp-specialized occupancy makes cooperative launch too
restrictive. Decide at G0 by measuring achievable occupancy; the barrier
primitive is a substrate choice everything else depends on.

### 2. Continuous prefetch — and the precise line between what prefetches and what cannot

TileRT's headline benefit is end-to-end continuous prefetch
(GMEM->SMEM->registers ahead of time). For this model the prefetchable and
non-prefetchable axes must be drawn exactly, because one is blocked:

- **Weights are static -> cross-region and cross-layer weight prefetch is a
  real, unblocked win.** Because the grid is persistent, the producer warps of
  layer L+1's attention can begin TMA-streaming layer L+1's projection/expert
  weights while layer L's MoE tail still computes. The weight addresses are
  known; nothing data-dependent gates them. This is the biggest new lever the
  persistent engine unlocks that the per-op path structurally cannot.
- **Activations stay on-chip -> no HBM round-trip between regions.** A decode
  token's hidden state is tiny (hidden_dim x dtype x batch — tens of KB), so the
  attention-output->MoE-input handoff lives in L2 (tens of MB on B200), never
  touching HBM. Killing those inter-kernel activation round-trips is a direct
  slice of the 20-40x overhead.
- **KV selection is data-dependent -> cross-layer prefetch is dependency-BLOCKED,
  and we already proved it.** Each layer's DSA Indexer top-k depends on that
  layer's input, which is the previous layer's output. You cannot prefetch layer
  L+1's selected KV before layer L finishes. Cross-step speculative prefetch is
  locality-blocked too (~37% mean block reuse, and the reused blocks are already
  LRU-resident; ~63% is unpredictable churn). The plan must not chase this lever
  — the achievable KV win is in-layer overlap (the swap-in copy group running
  concurrently with the attention compute group), which is precisely the G1/G2
  mechanism, now made native and graph-constraint-free (section 4).

Drawing this line explicitly is what keeps the engine honest: weights and
activations pipeline; KV selection does not, so its only lever is concurrency,
which the heterogeneous-worker split provides for free.

### 3. Device-resident control flow (the crux)

Everything dynamic moves on-device and stays there for the whole step:

- **DSA Indexer top-k** runs as the Indexer SM group; its output (selected block
  ids, in SMEM/L2 scratch) feeds the attention group across a grid barrier. No
  host round-trip, no graph-illegal readback.
- **HiSparse swap-in** becomes the copy SM group consuming the device-resident
  `compact_miss_schedule` (copy_count + slot lists) directly — the thing that was
  capture-illegal under CUDA graphs is trivial inside a persistent kernel,
  because there is no capture and no host orchestration. The cold(host-pinned)<->
  hot(device) copies are issued by copy warps and overlapped with the attention
  group's compute. This is where the entire HiSparse Gate-5-7 frontier collapses
  into ordinary device code: live-overlap magnitude, P4 live wiring, and
  promotion all become device code paths in the engine.
- **MoE router top-k** feeds the device-side work queue; expert assignment to MoE
  SMs is dynamic work-stealing, so a hot expert does not stall the group.
- **Device-side metadata / step-advance.** Positions, block tables, KV lengths,
  and the sparse-attention metadata are kept device-resident and advanced on
  device per accepted token by a small step-advance region — eliminating the
  per-step host metadata rebuild that currently fragments the stream (the
  "in-graph metadata" lever taken all the way).
- **MTP/DFlash (SMC-SD)** draft->verify->accept runs as a device-side loop inside
  the engine; only the final accepted tokens cross to host. The variable accept
  length never becomes a host branch.

### 4. Intra-kernel communication (TP all-reduce, EP all-to-all)

The decode forward has two collective points per layer: the TP all-reduce after
attention out-proj and after MoE down-proj, and the EP all-to-all for expert
dispatch/combine. To keep them inside the kernel:

- **TP all-reduce via NVLink-SHARP `multimem`.** On B200's NVLink5 + NVSwitch,
  comms warps issue `multimem.ld_reduce` / `multimem.st` PTX against a
  multicast-mapped buffer — an in-kernel, switch-side-reduced all-reduce with no
  kernel exit. Compute warps proceed on independent work (the shared expert, the
  next region) while the reduction is in flight: in-kernel compute/comms overlap,
  which is TileRT's "communication dissected into finer-grained Tiles."
- **EP all-to-all via device-initiated NVSHMEM (`nvshmemx_*` / IBGDA-style).**
  Harder — the dispatch/combine volume is data-dependent on routing — so comms
  warps issue per-token-group puts to expert-owning ranks from inside the kernel.
  This is the single riskiest mechanism in the plan.
- **Fallback (explicit):** if device-side comms is not robust enough, the kernel
  exits at the collective boundary, the collective runs as a normal NCCL/NVLS
  launch, and the engine resumes — we still keep the intra-GPU megakernel win
  (every fusion and the device control flow), losing only the in-kernel comms
  overlap. The fallback is a per-collective config flag, not a rewrite.

### 5. Quantization co-design (the Second Leap, native)

We already ship NVFP4 experts + FP8 dense, which is exactly TileRT/MiMo's "FP4
experts, FP8 rest" trade-off — so this is co-design we have validated, now
fused:

- **Dequant fused into the GEMM prologue.** NVFP4 weights stay compressed in HBM
  (that is the BW floor we want to ride); producer warps TMA the compressed
  bytes, an expand step unpacks to the MMA input type in SMEM/registers just
  before `tcgen05`. No standalone dequant kernel, no dequantized-weight HBM
  round-trip.
- **KVarN latent-KV dequant as a shared producer role.** The inverse-Hadamard
  dequant in the FlashMLA hot-read is per-head-redundant (the proven
  bottleneck); the U7 optimization already dequants once per head-group and
  shares it. In the engine this is simply the attention group's producer-warp
  role — the optimization becomes structural.
- **No MXFP8 activation round-trip.** WarpDecode already keeps activations in
  BF16 with FP32 accumulate (the 1.4x-closer-to-FP32 accuracy win). The engine
  preserves that end-to-end: activations never get quantized between fused
  regions because they never leave the chip.

### 6. Microsecond auxiliary-op triage

TileRT names RMSNorm, RoPE, KV-cache writes, syncs, and metadata as the
microsecond killers. In the engine they stop being kernels and become
register-level steps folded into adjacent region prologues/epilogues: RMSNorm
folds into the next GEMM's prologue (normalize in registers, feed MMA); RoPE
folds into the q-up-proj epilogue and the k_pe path; the
KVarN-quantize+Hadamard+write folds into the attention epilogue; metadata
advance is the device-side region from section 3. Each is individually
negligible in FLOPs and individually a stream-fracture on a microsecond clock —
fusing them is most of the Second Leap for our model.

## Mapping onto existing op-trt pieces

| Existing piece | Role in the persistent engine |
|---|---|
| FlashMLA hot-read decode (`sparse_mla_decode_kvarn_hot.cu`) | Attention compute region; head-grouped flash + shared dequant is the producer/consumer template |
| WarpDecode (output-centric MoE, FC2) | MoE expert-GEMM region; warp-independence is already the persistent-grid shape; "single-CTA is the megakernel path" generalizes here |
| DSA Indexer / HISA | Indexer SM group (device-side top-k) |
| HiSparse (`hisparse.py`, `SparseMlaDecodeKvarnHotOp`, `hisparseKvarnBdrRead.cuh`) | Copy SM group + hot/cold KV; G1/G2 overlap becomes native, graph-constraint-free |
| LayerSplit | Informs the static SM-group partition (heterogeneous workers) |
| SMC-SD | Device-side MTP draft/verify/accept loop |
| NVFP4 dense KV + KVarN | Fused prologue dequant + shared-producer dequant |
| `SimpleScheduler` + HiSparse coordinator (P4 admission seam) | Producer of the device-visible request/admission queue (cross-step persistence) |
| NVFP4-experts/FP8-dense quant config | The Second-Leap co-design point, now fused into GEMM epilogues |

The engine is not green-field — it is a re-housing of pieces we already
validated into one resident grid, plus the substrate (grid barrier, warp-role
framework, work queue, device comms) that lets them share a launch.

## Load-bearing build sequence (ordered by dependency, each correctness-gated)

Ordered by what must exist before what — not by schedule. Each gate is a
capability that is bit-accuracy-gated against the frozen per-op path / true dense
f32 reference before the next is allowed to build on it, re-gated on the reserved
orchestrator GPU.

- **G0 — Substrate.** Confirm the exact current decode call path on 001 (the one
  safe, necessary repo-grounding step), then build the reusable device framework:
  persistent cooperative launch sized to measured B200 occupancy, the grid-wide
  barrier primitive (cooperative `grid.sync()` vs hand-rolled global barrier —
  chosen by measured occupancy), the warp-role scaffold (role enum, `mbarrier`
  mailboxes), and the atomic device work-queue. *Gate:* a no-op persistent grid
  that launches, barriers across regions, and exits, with a microbenchmark
  proving the barrier cost is below the per-launch overhead it replaces.
  Everything below depends on G0.
- **G1 — Vertical slice (one layer, dense).** MLA attention region -> grid barrier
  -> MoE region, one layer, one launch, HiSparse off, single GPU. *Gate:*
  bit-exact vs the per-op path for one dense layer. Proves the producer/consumer
  pipeline and the inter-region L2 handoff.
- **G2 — Auxiliary fusion.** Fold RMSNorm, RoPE, KV-write, and prologue dequant
  into G1's region boundaries. *Gate:* per-fusion bit-accuracy vs true dense f32
  reference (never fused-vs-sequential).
- **G3 — Device-resident control flow.** DSA Indexer region -> attention; MoE
  router -> work queue; device-side metadata/step-advance. *Gate:* device-side
  selections and metadata match the host-side path exactly. This is where
  graph-safety stops being a constraint.
- **G4 — HiSparse copy group.** Cold<->hot swap as in-kernel copy warps consuming
  the device schedule, overlapped with the attention group. *Gate:* byte-identical
  vs serial swap-in; overlap magnitude measured live (closes HiSparse Gate-5-7).
- **G5 — Full layer stack.** Loop all layers inside one persistent step, with
  cross-layer weight prefetch. *Gate:* full-model single-GPU decode step
  bit-accurate end-to-end.
- **G6 — Intra-kernel TP.** NVLS `multimem` all-reduce at the two collective
  points, overlapped with compute; fallback flag wired. *Gate:* TP4 multi-GPU
  decode correct vs the DP2/TP4 baseline.
- **G7 — Intra-kernel EP.** Device-initiated NVSHMEM all-to-all for expert
  dispatch/combine. *Gate:* EP correctness; if unstable, ship on the
  launch-boundary fallback and record the gap (no silent cap).
- **G8 — MTP/DFlash device loop.** SMC-SD draft/verify/accept inside the engine.
  *Gate:* accept-length distribution and outputs match the current SMC-SD path.
- **G9 — Cross-step persistence.** Device-visible request queue + completion ring
  + device-side continuous batching; host shrinks to admission (the
  `SimpleScheduler`/HiSparse-coordinator seam becomes the queue producer). *Gate:*
  end-to-end serving correctness + the per-step host overhead is gone from the
  trace.
- **G10 — Tuning loop (microsecond triage).** SM-group partition ratios, pipeline
  depth, warp-role counts, occupancy. *Gate:* tok/s/user-after-first-token vs
  DP2/TP4 `64/49/41/40` and r20 disagg, profiled to confirm we are closing the
  20-40x overhead multiple toward the BW floor.

## Correctness and measurement methodology

- **Two-stage gate throughout:** `linear_decode_tps` + intel-correctness, never
  one without the other.
- **True dense f32 reference only.** Every fusion and every device-side selection
  is gated against the frozen per-op path or a true dense reference — never
  fused-vs-sequential self-comparison. Cosine floor 0.98, target 0.999999.
- **Per-region unit probes** (self-contained, load only the fresh `.so`) for each
  region before it enters the stack; the per-op path stays the live oracle.
- **End-to-end vs the real baselines:** DP2/TP4 `64/49/41/40` and r20 disagg
  TPOT, profiled (Nsight) to attribute the recovered time to launch-elimination
  vs activation-residency vs comms-overlap — so the win is explained, not just
  observed.
- **Re-gate on the reserved orchestrator GPU** (separate from the build GPU),
  reproducible via the proof-image toolchain.

## Risk register and honest constraints

- **In-kernel multi-GPU comms (NVSHMEM EP) is the top risk.** Mitigation:
  per-collective launch-boundary fallback that preserves every other win; treat
  G7 as best-effort with a measured floor, not a blocker for G6/G8/G9.
- **Occupancy vs fusion tension.** A fatter fused kernel can lose more to reduced
  resident CTAs than it gains from launch elimination. Mitigation: heterogeneous
  workers bound per-SM working set; G0 measures the occupancy envelope before any
  region is fused; if a region is occupancy-toxic it stays a separate (still
  graph-captured) launch — fusion is per-region, not all-or-nothing.
- **Dynamic MoE imbalance** -> device work-stealing queue (static SM->expert
  assignment will stall on hot experts). REAP's smaller expert set eases this.
- **Blocked levers we must not re-attempt:** cross-layer KV-selection prefetch
  (per-layer Indexer dependency, proven) and cross-step speculative KV prefetch
  (locality-blocked, reused blocks already resident). Weight prefetch is the
  unblocked analog — pursue that, not those.
- **Megakernel debuggability.** A whole-model persistent kernel is hard to debug.
  Mitigation: the warp-role framework + per-region probes + keeping the per-op
  path as the always-available oracle and fallback.
- **CUDA graph vs persistent kernel** are alternative launch-overhead killers;
  the persistent kernel goes further (device control flow) but loses replay
  simplicity. Decide per region — some occupancy-toxic or comms-heavy regions may
  rationally stay graph-captured.

## What this explicitly does not change (preserved invariants)

The frozen ABI (`SparseMlaDecodeKvarnHotOp.cpp`, `hisparseKvarnBdrRead.cuh`) is
untouched and remains the oracle/fallback; `resident_v1_ready()` stays false
(fail-closed); prefill keeps its expert-centric / LayerSplit path; other tracks'
worktrees stay untouched; commits go to ai-blaise repos only (never upstream),
twin-history via the throwaway worktree off `origin/op-trt-hisparse` on the Mac,
FF-only.

## Why this is worth building as the decode path

Our own measurement says decode is ~95% execution gap, and DeepSeek-V3.2's
data-dependent sparsity is what manufactures that gap. The persistent engine is
the architecture that makes the dynamic decode device-resident — which is
simultaneously the largest perf lever left (the 20-40x multiple) and the thing
that dissolves the graph-safety wall HiSparse, the Indexer, and SMC-SD keep
hitting. That dual payoff is why this is the decode path, not another per-kernel
increment.

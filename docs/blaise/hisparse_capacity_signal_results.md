# HiSparse Capacity-Signal Gate — Results (2026-06-13)

The capacity-signal gate (external-review Recommendation 4) is the no-live-DGD
measurement front-loaded between plan Gate 3 and Gate 4 to decide, **before** the
remaining serving-integration hardening, whether the OP-TRT HiSparse subsystem
wins and where the real blocker is. It was implemented by a subagent on
`a4-us-001` against the current-head proof image (source SHA `aaa7e2b`, ops
verified NOT stale vs HEAD), and independently audited by the orchestrator
(separate GPU re-runs, not the subagent's self-report). Artifacts:
`blaise_perf/hisparse/{compute_capacity_math.py,bench_block_fanout.py,bench_microbench.py,hotread_correctness_probe.py,swapin_plan_correctness_probe.py}`
and `*.md/*.json` under the same dir.

## Verdict

**Capacity: GO. Correctness (attention math): GO. Serving viability: the hot-read
kernel blocker is now RESOLVED (Phase-3 → Phase-4).** The gate's original finding
held: the production-ABI `sparse_mla_decode_kvarn_hot` kernel was a naive scaffold
~3 orders of magnitude too slow to serve, and the gate's decisive output was to
invest in that kernel (Gate-4) — NOT the Gate 5-7 serving integration — exactly the
"measure before hardening" outcome Rec-4 was designed to force. That investment has
landed: the FlashMLA-style rewrite plus the FWHT dequant and warp/vectorization
levers took the hot-read from ~348 ms/call to 0.360 ms/call (M19, Phase-4/5), and the
tcgen05/UMMA tensor-core rewrite then took it to 0.217 ms/call (U4, Phase-6), and a
profile-driven wide-dequant + Hadamard-hoist round then to **0.147 ms/call (B16),
~9.0 ms/step** (U7, Phase-7) — comfortably under the ~20 ms c16 decode budget, at cos
0.999995 vs the true dense reference with the production ABI frozen. The ≤0.30 ms/call
(≤18.3 ms/step) target is beaten ~2×.

## 1a — Capacity (verified)

Packed `kvarn_k2v2` BDR record = **13312 B / 64-token block = 208 B/token/layer**
(`mlaBdrKvarnOp.cpp:32-41`, orchestrator-confirmed two ways), vs bf16 latent
1152 B/token (**5.54× denser**; ≈2.89 effective bits/elem over the 576-wide
latent).

Per-request device KV (L attention layers; reduction is **L-independent** =
`context / (hot_blocks·64)`):

| context | KVarN-only | HiSparse hot=64 (fixed) | reduction |
|--------:|-----------:|------------------------:|----------:|
| 32k  | 0.41 GB | 49.6 MiB | 8× |
| 64k  | 0.83 GB | 49.6 MiB | 16× |
| 128k | 1.62 GB | 49.6 MiB | **32×** |
| 256k | 3.25 GB | 49.6 MiB | 64× |

The device footprint is **fixed** at `hot_blocks·64·208·L` regardless of context;
the ceiling lift = `context/(hot_blocks·64)`. ~177× vs naive bf16 full-HBM @128k.
**This is a capacity/concurrency win, invisible at c16 (Rec 1).**

## 1b — Block fan-out (the decisive sizing number)

Distinct 64-token blocks touched by the 1024 selected tokens per step, counted by
the **production** `hisparse_topk_to_block_positions` op (cross-checked exact vs
`set(t//64)`), over a **labeled locality model** (recency+sink+scatter — real DSA
indexer needs a full-model trace, out of scope here):

| regime (128k) | distinct blocks/step (p50) | new blocks/step | forced miss @ hot=64 |
|---|---:|---:|---:|
| recency-heavy | 341 | 276 | ~81% |
| balanced | 540 | 393 | ~85% |
| scattered/uniform | 736–809 | 470–489 | ~91% |

**`hot_blocks_per_req=64` is 5–13× below a single step's working set.** At hot=64
the buffer is a 64-block prefetch window over a 315–809-block demand — it streams
most of the selection from host every step (81–91% miss). Two ways forward, a
real tradeoff curve, not a free 32×:

- **Keep hot=64 (full 32× capacity):** requires hiding the per-step miss DMA —
  **Rec-5 overlap becomes mandatory**, not optional. **Measured** mapped-host copy
  bandwidth is **~48 GB/s** (not the >100 GB/s I first assumed), so a B=32
  recency-128k miss set (~277 new blocks/req × 32 = ~8864 copies ≈ 113 MiB) is
  **~2.3 ms/layer** — already >10% of the ~20 ms decode budget *before* the
  hot-read, and multi-× that at higher B/scatter. It is still hideable behind the
  ~22 ms MoE window **if overlapped on a copy stream**; exposed and serialized
  otherwise. The ~48 GB/s copy BW is itself a secondary lever (a copy-engine
  variant over the same compact device schedule may beat the mapped-host kernel).
- **Right-size hot_blocks → ~384/576/896** (recency/balanced/scattered p95) for a
  real working-set cache (steady-state miss = only new blocks): device KV
  ~297/446/694 MiB/req — still **2.3–5× under KVarN-only @128k**, capacity win
  survives a correctly-sized buffer.

## 1c — Correctness (audited, with one important qualification)

- **Swap-in planner chain: exact.** `plan_hot_slots`+`build_hot_indices` match an
  exact Python mirror of the kernel (hot-slot assignment, LRU order, miss
  schedule, hit flags, hot global indices) across cold-single, cold-multi(LRU),
  and warm-replay(all-hit). ✓
- **Hot-read attention math: verified.** `sparse_mla_decode_kvarn_hot` vs an
  independent torch dense-attention reference over the same dequantized latents
  (non-zero q, varied values) = **cosine 0.999999** — a true reference, not a
  fused-vs-sequential self-comparison. Orchestrator reproduced this on a separate
  GPU. ✓
- **Quant accuracy: NOT closed by this gate (orchestrator catch).** The hot-read
  gate compares against the *dequantized* latent (shared quant), so it isolates
  the attention math but does **not** test the `kvarn_k2v2` write→read accuracy
  vs the *original pre-quant* latent. An independent orchestrator run added that
  check: on i.i.d. Gaussian inputs, `cosine(dequant, original)` = **0.906**
  (2-bit C-KV 0.896; 8-bit RoPE 0.9997) and full-chain-vs-original = **0.890
  (< 0.98)**. This is the **worst-case input artifact**, not a production
  failure — i.i.d. Gaussian is adversarial for any 2-bit quant, and `kvarn_k2v2`
  (Hadamard+Sinkhorn) reaches its ~FP16 accuracy only on the real, structured KV
  distribution (the existing KVarN component gate already established ~0.9996 on
  real latents). **A valid in-context end-to-end accuracy number requires real
  dense-MLA KV (a model forward)** — the same out-of-scope caveat as the
  real-indexer fan-out. The synthetic "PASS" must therefore be read as
  *attention-math-correct*, not *production-accuracy-proven*.

## 1c — Latency (the blocker)

`sparse_mla_decode_kvarn_hot` is a **naive serial-1024-topk kernel** (grid =
rows × 128 heads, serial reduce over topk, scalar on-read BDR dequant):

- **348 ms** at hb32/B16/next_n=1, scaling ~linearly with rows to **2.7 s** at
  B64/next_n=2 — **per layer call**.
- The op is invoked **per attention layer** (`layer_idx`), ~61/decode step ⇒ a
  decode step of **~21 s at B16** vs the ~25 ms target — **~800–1000× too slow**.

This is the plan's own acknowledged "direct per-row/head, pre-FlashMLA-split"
scaffold, now quantified. **Serving is impossible until the FlashMLA-split
hot-read kernel (Gate 4) lands** — the `sparse_mla_decode_kvarn_hot_split` variant
the plan notes exists but is "not merge-ready" is the critical path. The swap-in
copy, by contrast, is a bandwidth-bound mapped-host kernel (misses × 13312 B);
its cost is small (sub-ms/step) and overlap-hideable.

## What this changes in the plan (recommendations)

1. **Re-prioritize: Gate 4 (FlashMLA-split hot-read kernel) is the hard
   prerequisite, ahead of Gates 5–7.** No amount of NIXL/admission/cancel
   hardening matters while the hot-read is ~1000× too slow. This is the gate's
   single most actionable output.
2. **Decide hot_blocks deliberately from the measured fan-out**, not the pinned
   64: either commit to hot=64 + *mandatory* Rec-5 miss/compute overlap (full 32×,
   overlap-dependent), or right-size to ~384–896 (2.3–5×, low-miss). Make
   `hisparse_hot_blocks_per_req` a measured/adaptive parameter.
3. **Two measurements still need a model forward** (the only things this gate
   could not close without a live model): real-DSA-indexer block fan-out, and
   real-KV end-to-end quant accuracy vs the production full-HBM `kvarn_k2v2` path.
   Both are the natural Gate-4 companions.
4. **Keep the honest capacity framing (Rec 1):** the A/B must sweep concurrency
   up; c16 is the no-regression floor, and HiSparse is scored on the capacity
   axis, not c16 TPOT.

## Phase-2 — hot-read kernel optimization (bit-identical 3.41×, verified)

The gate identified the hot-read kernel as the blocker. A bounded, math-preserving
optimization was implemented (subagent) and **independently verified by the
orchestrator** (both `.so` run on GPU 7 with identical seeded inputs).

- **Transform** (`cpp/tensorrt_llm/kernels/flashMLA/sparse_mla_decode_kvarn_hot.cu`,
  +120/−31): the per-token C-KV base dequant (2-bit unpack) was recomputed
  O(dim×128) per token — once per output dim sharing a 128-wide inverse-Hadamard
  subblock, twice (score + V). Now each token's 512 C-KV base values are
  dequantized **once** into a shared cache (`buildCkvBaseCache`); the
  inverse-Hadamard (`ckvHadamardFromCache`) and the V accumulation read the cache.
  The Hadamard accumulation order (ascending j), signs, scale, the score/V
  reduction order, the per-thread dim partition, and every `__float2bfloat16_rn`
  rounding point are preserved → bit-identical. Production ABI + BDR layout frozen.
- **Bit-identical (verified two ways):** subagent strict before/after 8/8 bit-exact
  (out_maxabs=0.0); orchestrator's independent BEFORE/AFTER run on fixed-seed inputs
  produced an **identical output signature** (sum `-3.2716332487e+02`, identical head
  values) on both `.so`; dense-ref 0.999999; existing smoke PASS.
- **Speedup: 21.79 → 6.38 ms/row (3.41×)** at B16/nn1 (orchestrator-confirmed),
  3.50× at B64/nn2. The win is the removed 128× redundant unpack — a warp-shuffle
  reduction variant gave only −8% (the `__syncthreads` were not the bottleneck).
- **Honest scope:** a bit-identical down-payment, not the finish line. At 6.38 ms/row
  × 61 layers the kernel is still ~250× from a servable decode step; production speed
  requires the separate **Gate-4 FlashMLA-split / tensor-core** rewrite (multi-day,
  CZS/IKP-gated). This optimization does not touch the production ABI/BDR layout and
  does not flip any readiness gate.

## Phase-3 — FlashMLA-style hot-read rewrite (21.6×, verified; major step, not the finish line)

The Gate-4 rewrite (user-directed). The hot-read is MLA decode attention over the
selected tokens = two GEMMs with on-read KVarN-BDR dequant. The kernel was
restructured FlashMLA-style: the serial-1024 `blockReduceSum`-per-entry score
reduction replaced by a parallel warp-tiled reduction + online softmax, with
head-group occupancy tuning (split target 304 for B16) and the per-block C-KV
base dequant shared 8× across heads. (`sparse_mla_decode_kvarn_hot.{cu,h}`.)

- **6.39 → 0.295 ms/row (21.6×) at B16**, independently re-verified by the
  orchestrator on GPU 7 (102.10 → **4.72 ms/call**, cos **0.999999**); 271–296
  µs/**row** across all buckets. Correctness: dense-ref 0.999999, smoke PASS
  (committed-hot + resident sink/tail + fail-closed), no regression. Production
  ABI/BDR frozen; readiness gate stays false.
- **Honest gap (per-row ≠ per-step):** 295 µs/**row** is the headline, but per
  **call** at B16 is 4.72 ms → ×61 layers ≈ 288 ms/step, still above the ~20 ms
  c16 budget. The kernel is now **dequant-bound** — the inverse-Hadamard-128
  dominates (~67M FMA/block vs ~26M for score+V); B16 is also occupancy-limited
  (only 16 rows). Closing the rest needs **tensor-core Hadamard dequant**
  (numerically delicate vs the fp32 reference) — the verified ceiling for a
  math-faithful scalar-dequant kernel. This is a major step toward A/B-viability
  (74× under the original 348 ms/call), not the finish line.

## Phase-4 — FWHT + warp/vectorization to the c16 budget (12× over M4, verified; A/B-viable)

Phase-3 left the kernel dequant-bound and flagged "tensor-core Hadamard" as the next
lever. The ceiling was reached differently — and tensor cores were measured and
**rejected**. Continued from the Phase-3 M4 baseline (4.49 ms/call B16) to **M15 at
0.377 ms/call — a verified 12×, holding cos 0.999999** vs the true dense BDR-dequant
reference at every step, smoke PASS throughout (orchestrator re-gate on GPU 7:
0.375 ms/call). The kernel `.cu` is committed to `op-trt-hisparse` (GitHub
`4dc6c7de4`); ABI frozen.

| Step | ms/call | ms/row | × over M4 | note |
|---|---|---|---|---|
| M4 baseline | 4.505 | 0.2816 | 1.00× | |
| M5 FWHT dequant | 1.204 | 0.0751 | 3.74× | bit-identical |
| M8 warp-per-token dequant | 0.717 | 0.0448 | 6.28× | bit-identical |
| M9 SMEM softmax-weight precompute | 0.570 | 0.0356 | 7.90× | bit-identical |
| M11/M13 128-bit PV reads | 0.478 | 0.0299 | 9.43× | bit-identical |
| M14 kHeadsPerBlock 16→32 | 0.386 | 0.0242 | 11.67× | cos-equal |
| **M15 128-bit score reads** | **0.377** | **0.0235** | **11.95×** | cos-equal, FINAL |

- **FWHT (the decisive lever, exact):** the O(128²) per-dim inverse-Hadamard was
  replaced by the Fast Walsh-Hadamard butterfly (warp-cooperative: intra-lane low
  stages + `__shfl_xor` high stages), verified `FWHT(x) == H@x` in fp64 to 4.4e-15.
  This removed the dequant wall (the 3.74× M5 win, bit-identical). M8–M15 then
  removed structural waste: one warp-per-token dequant, SMEM-precomputed softmax
  weights, 128-bit (int4 / 8-bf16) vectorized Q/PV/score SMEM transactions, and
  kHeadsPerBlock/kTileTokens swept to 32.
- **Tensor cores attempted and rejected (honest):** a correct score-MMA (m16n8k16,
  fragment layout validated standalone at max_abs_err 0) **regressed** — global-Q
  A-fragment reload per k-step, half-warp idle, and no TMEM for the 512-wide V
  accumulator make the scalar path win. Component ablation confirms the kernel is
  occupancy/issue-bound at low arithmetic intensity, not GEMM-bound. A neutral M16
  (128-bit kTile write) confirmed the ceiling. Further gain needs a TMEM-accumulator
  UMMA mainloop or fp8 V (SM100 lacks the mixed atom) — not pursued, to keep every
  step at cos 0.999999.
- **Accuracy:** M5–M13 are bit-identical to M4 (output signature −327.16367); M14/M15
  differ ~3–5e-6 rel (split-count + dot-grouping reorder), max_abs 2.44e-4 = M4's own
  bf16 floor. Full 24-bucket sweep (B16-64 × nn1-2 × hb32-128) uniform 0.0223–0.0244
  ms/row, zero regressions.
- **Bottom line:** per-step (×61 layers) ~274 → **~23 ms**, within the ~20 ms c16
  decode budget — the hot-read is now **A/B-viable** (~920× under the original
  348 ms/call). Only the kernel `.cu` differs (+173/−70 vs the M4 commit);
  `SparseMlaDecodeKvarnHotOp.cpp` and `hisparseKvarnBdrRead.cuh` are git-diff-empty
  and `..._resident_v1_ready()` stays `false`.

## Phase-5 — further tuning + the measured non-tensor-core ceiling (M15 → M19, bit-identical)

A follow-up round chasing "comfortably under 20 ms/step" (≤ 0.30 ms/call). Result:
**M19 = 0.360 ms/call @ B16, bit-identical to M15** (SIG −3.2716296266e+02, cos
0.999999, smoke PASS, orchestrator-regated GPU 7), per-step ~23 → ~22 ms. The
≤ 0.30 target was NOT reached; the round's value is the measured evidence for why
~0.36 is the ceiling of a math-faithful scalar kernel. Committed `be2e620a5`.

- **What landed (bit-identical):** M16 single-pass ELTS=16 FWHT (cleaner, neutral);
  **M18** cache per-token `active` in a 32-byte SMEM array to kill a redundant
  per-token `params.indices[]` GMEM re-read in the score loop (+2.4%, the real win);
  M19 score/PV unroll-4 (+1.4%).
- **cp.async software-pipelined dequant — REJECTED, −23%:** the framework issue cost
  (2nd resolve + `__pipeline_commit`/`wait`, ~0.08 ms) dwarfs the ~0.04 ms / 11%
  hideable GMEM load latency at 25% occupancy. Same root cause that sank score-MMA.
- **Occupancy is dual-locked at 2 blocks/SM:** REG=128 *and* dynamic SMEM=104 KB
  (the 64 KB `acc[32][512]` fp32 V-accumulator is intrinsic). 3 blocks needs reg < 85
  AND smem < 77 KB; hpb16 fits 3 but is slower (2× dequant redundancy). The grid is
  already 2.16× over the 148 SMs, so split-K is saturated.
- **The only path to ≤ 0.30** is a TMEM-accumulator UMMA mainloop (sm_100 5th-gen
  tensor cores) — a high-risk rewrite vs the fp32 dense reference (cos-divergence
  risk), deferred to keep every step at cos 0.999999. The c16 A/B can proceed at M19
  (~22 ms/step): the real question is end-to-end tok/s/user, not the per-step
  microbench in isolation, and the per-step is now within ~10% of the budget.

## Phase-6 — tcgen05/UMMA tensor-core rewrite to 0.217 ms/call (40% over M19, under the c16 budget)

The "high-risk TMEM-UMMA rewrite" Phase-5 flagged as the only path below ~0.36 was built
and verified. **U4 = 0.217 ms/call @ B16, cos 0.999998, ~13.2 ms/step — 40% under M19
(0.360) and comfortably under the ~20 ms c16 budget** (orchestrator GPU-7 re-gate:
0.216–0.217 over 3 runs; smoke PASS; hotread probe out-cos 0.999997 / lse-cos 1.000000
VERDICT PASS). Committed `aaed1e17d`.

The kernel uses sm_100 5th-gen tensor cores (tcgen05/UMMA): both the score S[64,topk] and
value O[64,512] accumulators live in **TMEM**, M=64 heads/block (2 head-groups), bf16 WS
SS atoms (`SM100_MMA_F16BF16_WS_SS_NOELECT`, operands in SMEM), flash-decoding online
softmax reading S from TMEM. The frozen KVarN-BDR 2-bit dequant fills the bf16 operand
tiles. It auto-enables on sm_100 (`HISPARSE_UMMA_ENABLED` when `__CUDA_ARCH__ >= 1000`)
with a host-stub fallback; only the kernel `.cu` and the th_hisparse_smoke `CMakeLists.txt`
change.

The win was an **occupancy insight, not the GEMMs** — and it came from a falsified thesis:
- **The binding resource is TMEM, not SMEM.** The kernel allocates all 512 TMEM cols/block
  and a B200 SM has exactly 512, so two CTAs serialize on the single TMEM pool — proven by
  a spin-probe (two 512-col CTAs on one SM double walltime, ratio 2.00). The kernel is
  hard-capped at **1 effective block/SM regardless of SMEM**.
- **fp8 was built, validated, and rejected.** The fp8 WS atom (`SM100_MMA_F8F6F4_WS_SS_NOELECT`;
  the cutlass non-WS fp8 atom *hangs* the GPU on nested elect+barrier) computed correct GEMMs
  and held end-to-end cos 0.99939, and fp8 *did* halve SMEM to 2 blocks/SM-by-SMEM — but it
  can't win because the kernel is TMEM-capped to 1 block/SM and overhead/dequant-bound, so the
  fp8 GEMM 2× is swamped (decomposed: SW64 swizzle +0.011, fp8 itself +0.032). Shipped clean bf16.
- **The actual lever: stop over-splitting.** Because the kernel is TMEM-capped to 1 block/SM,
  the prior split factor (numSplits=10 → 320 CTAs ≈ 2.2 waves, sized for 2 blocks/SM) was
  2.5× over-split; each extra CTA pays redundant TMEM alloc/free + barrier init + Q-reload.
  Retargeting to a **single wave** (numSplits = SMs/baseBlocks; nS=4 → 128 CTAs at B16) removed
  that overhead: nS=10 0.347, nS=5 0.235, **nS=4 0.217**, nS=1 0.589. The split logic now
  auto-targets one wave from the queried SM count (`HISPARSE_TARGET_BLOCKS` override).

**Proven floor (the TMEM-split was built and rejected):** 0.217 is the ceiling. Going below
needs ≤256 TMEM cols/block for true 2-way concurrency — which *is* achievable (a co-residency
probe confirmed two 256-col CTAs co-reside on one SM, ratio 1.00 vs 2.00 at 512). The 2-pass
O-split kernel (U5: pass-1 dequant+score+softmax+O[0:256) caching scaledScore+V_hi to GMEM,
pass-2 value-UMMA→O[256:512) only) was built and gates correct (cos 0.999998) but its best is
0.299 ms/call (+38% over U4). Three-way proof it can't win: (1) adding the 2nd co-resident
block makes it *slower*, not faster (256-col→0.369, 320→0.405); (2) the kernel is
**dequant/compute-throughput-bound** (FWHT + 2-bit unpack + PE), **not TMEM-occupancy-bound** —
the 512-col TMEM cap was *masking* a compute bound, not creating one, so breaking it exposes no
headroom; (3) the O-split forces a V GMEM round-trip + a serial pass-2 (+66 µs by nsys) U4 never
pays. The 2-CTA-cluster variant would also lose (it duplicates the compute-bound dequant). The
precision (fp8), GEMM, and occupancy levers are all exhausted — **U4 (0.217) stands.**
(Superseded by Phase-7: those *were* exhausted, but a different axis — thread-level
latency hiding — was not yet explored.)

## Phase-7 — profile-driven wide-dequant + Hadamard-hoist to 0.147 ms/call (−32% over U4)

A research-grounded round (CuTe DSL docs + the CuTe-layout paper + Colfax FA-4/Blackwell
tutorials + Veitner, with cutest/CZS/IKP as tooling). The dequant-*arithmetic* levers the
plan centered on proved a ~2-10% ceiling — because **C0 profiling falsified the
dequant-bound premise** and found a new axis Phases 5-6 missed. Result: **U7 = 0.147 ms/call
@ B16, cos 0.999995, ~9.0 ms/step — a further 32% under U4 (0.217)** (orchestrator GPU-7
re-gate: 0.147-0.148 ×5; smoke PASS; dense-ref cos 0.999995; ABI diff-empty). Committed
`ff3b4f086`. The win **grows with topk**: at topk=2048, U4 0.342 → U7 0.218 (−36%).

**C0 — the measurement that redirected the round.** Ablation (resolution-immune compute
attribution) + IKP region trace (per-region ns; NVBit silently can't instrument Blackwell
tcgen05 SASS — a documented tooling negative; `ncu` unavailable). Dequant *compute* is only
~16% (FWHT 8.7%, score/value UMMA 5.5%, unpack/PE/exp ~2%); **~80% is memory-latency
stalls** — the gather *pattern* (random == sequential) and the load *count* (vectorized ==
scalar) were both proven *irrelevant* by ablation. Root cause: **latency-bound at 6% thread
occupancy** (128 threads, 1 block/SM, TMEM-capped) — *not* compute- or TMEM-occupancy-bound
as Phase-5/6 concluded.

- **Hadamard-hoist (C4), kept, −2.8%:** the inverse-Hadamard is orthogonal + symmetric +
  self-inverse, so it folds out of the per-token dequant into a one-time per-row transform —
  S = (Ĥ Q)·K_raw^T and O = Ĥ·(P·K_raw). The per-token hot path drops the FWHT and its
  cross-lane shuffles entirely. Proven exact in isolation (S cos 1.0, O cos 0.99999988); the
  resident cold path applies Ĥ⁻¹ on pool reads to keep the smoke consistent.
- **Wide-dequant (the dominant lever), −32%:** since the dequant is latency-bound at 6%
  occupancy, run it on **384 threads / 12 warps** (`kDequantThreads`; 8 tokens/warp vs 16) to
  hide load + barrier latency, while score/softmax/value/epilogue stay gated to the first 128
  threads (the UMMA/TMEM layout) with all `__syncthreads` block-wide (no hang). Sweep:
  128=0.211, 256=0.158, 320=0.153, **384=0.148**, 512=0.148 (plateau).
- **Rejected (all measured):** packed-cvt conversions (~2% ceiling), softmax O-rescale-skip
  (inert on near-uniform random scores), vectorized byte loads (not load-instr-bound),
  register-frugal epilogue (<2%; the 384-thread spill is intrinsic to the hot loop). Only the
  kernel `.cu` changes (+231/-120); ABI frozen.

**Lesson:** this kernel's true ceiling is **warp-level parallelism / latency hiding**, not
compute or TMEM occupancy — which only the IKP + ablation profiling surfaced, after
fp8/GEMM/occupancy/TMEM-split were all measured-dead in Phases 4-6.

## Systems-level swap-in (post-kernel) — overlap + bulk-coalesce + prefetch + sizing + capacity

With the hot-read at 0.147 ms (Phase-7), the bottleneck moved off the kernel and onto the
hot/cold swap-in machinery — exactly the capacity-gate's "Rec-5 overlap MANDATORY" item. A
direct read of the SGLang HiSparse source (`hisparse_coordinator.py`, `mem_cache/allocator/
hisparse.py`, `jit_kernel/csrc/hisparse.cuh`, `dsa_backend.py`) vs ours found that **nothing in
SGLang's non-kernel design is algorithmically ahead of ours — its only edge is that its path is
live while ours is fail-closed**, plus two axes where we can *beat* it (overlap, prefetch —
SGLang does neither). Implemented in the coordinator (`hisparse.py`, +471/−3, purely additive;
ABI + `resident_v1_ready()=false` preserved; every path bit-identical on the real coordinator;
gate probes pass). Committed `5e2d281ac`. **A later wiring + composability sweep (below) found P1/P3
as built were on a *dormant* Python-records path the captured forward never called; G1 ported the
overlap into the live native op graph-safely and removed the dormant code. Read the P1/P3 entries
below as the isolation findings, and the G1 block for what is live.**

- **P1 — miss-DMA overlap + bulk coalescing (the must-win).** The decisive finding: the swap-in
  path's per-block `cudaMemcpyAsync` is **100% launch-bound** for scattered misses — 8864 copies
  at B32/128k ≈ **52 ms / 2.9 GB/s** (the gate's "48 GB/s" was the *other* op, the SM-contending
  schedule kernel that **can't** overlap with matmul). Added a dedicated copy stream + event join
  and a **staging-gather bulk fast path** (dense hot run → CPU `index_select` gather into
  contiguous pinned staging → one bulk H2D at the 55 GB/s copy-engine ceiling, byte-identical),
  with a per-block fallback for steady-state scatter. Bulk copy alone **52 → 3.2 ms (16×)**;
  overlapped behind the compute window the exposed swap-in is effectively removed (**40.4 → ~0.02
  ms** microbench). The copy-engine path is the overlappable winner; the schedule kernel is faster
  alone but un-hideable (contends for SMs). This is what makes HiSparse viable — 52 ms/step would
  have blown the 20 ms budget 2.6×.
- **P2 — adaptive hot-buffer sizing.** Drove the real planner + fan-out op: working set D_p95 =
  recency 349 / balanced 551 / scattered 752 (matches §1b), so the knee H ≥ D_p95 = **384 / 576 /
  896**. `hot_blocks=64` is badly undersized (STREAM mode, 154-162% miss); the knee cuts miss-DMA
  ~50% and flips STREAM→CACHE while keeping a **12.7-29.5× capacity win** vs KVarN-only HBM @128k.
  `recommend_hot_blocks()` + int / "auto" / 0→regime-knee resolution (default 64, backward-compatible).
- **P3 — predictive prefetch (`prefetch_swap_in_plan` + `join_prefetch`).** Issue layer L+1's
  swap-in during L's compute, join (wait + commit) at the read — the clean deferred-overlap path
  (the synchronous `execute_swap_in_plan_overlapped` keeps an inline wait). Bit-identical across an
  8-layer pipeline; prefetched blocks == read-needs every layer. ~1-6% when P1's bulk copy already
  makes the DMA near-free, **1.23-1.38× when the copy exceeds a single layer's window** (scattered
  / larger batch) — the regime P1+bulk can't cover. SGLang does neither overlap nor prefetch.
- **P4 — freed-HBM capacity-admission accounting** (`admittable_token_capacity` = SGLang's
  host-backed `max_total_num_tokens` analogue, `can_admit_request`, `capacity_accounting`):
  admit-until-budget proven in isolation (host-backed 262144 vs device-resident 24576 tok =
  **10.7×**), release returns freed blocks + reopens headroom. **Live promotion is NOT done here** —
  `resident_v1_ready()` is a hardcoded `return false` by design pending a live multi-rank DSA/NIXL
  forward (Gate-5-7); the mechanism exists, the promotion is documented as needing the model stack.

### Wiring + composability sweep → G1 (the live graph-safe overlap)

A full wiring sweep (does each optimization reach the live captured forward, and compose with the
custom + production pieces?) found: **U7 hot-read kernel WIRED** (op → `invokeSparseMlaDecodeKvarnHot`
→ forward, gated on `coordinator.enabled`); **P2 sizing WIRED** (`recommend_hot_blocks` →
`configure_from_kv_cache_manager`); **P4 capacity AVAILABLE-ONLY** (complete accounting, no scheduler
hookup); and critically **P1/P3 DORMANT** — the live swap-in is a *device-native* CUDA-op chain
(`map_topk_to_hot_pool` → `… compact_miss_schedule → submit_packed_kvarn_copy_schedule`) whose copy is
a single mapped-host **kernel on the compute stream** (no copy stream, no overlap), while the P1/P3
Python `execute_swap_in_plan_overlapped` / `prefetch` were a *separate Python-records model the
forward never invoked* — and not graph-safe (CPU `index_select` gather + Python stream mgmt).
Composability otherwise clean: U7 ⊗ swap-in byte-identical (the Hadamard-hoist is algebraic, BDR
storage invariant), no op-registry collision with WarpDecode/NVFP4/Indexer-HISA/GQA/LayerSplit,
fail-closed intact, A+B compile + link together.

**G1 (committed `5124f823`)** closed the dormancy by porting the overlap **into the native op**,
graph-safe. The binding constraint: `compact_miss_schedule` emits `copy_count` + slot lists as
**device-resident** tensors and the forward runs **inside the DSA CUDA-graph-captured region**, so a
host-enumerated copy-engine `cudaMemcpyAsync`-per-run loop (the P1 staging-gather recipe) is **not
expressible graph-safely** — it needs a capture-illegal d2h readback, and no CUDA API issues a
copy-engine memcpy from device-resident pointers/sizes. So the copy stays a **byte-identical kernel**
moved onto a coordinator copy stream behind **capture-safe fork/join events** — the win is *overlap*,
not SM-elimination. `hisparse_submit_packed_kvarn_copy_schedule` gained 2 defaulted args
(`overlap_copy_stream`, `copy_stream_handle`); `hisparse.py` does Python-side fork/join via the torch
Stream/Event API and **removed the 214 dormant non-graph-safe lines**; `register_fake` extended 1:1.
Verified (re-gated independently on GPU7): native planner/copy byte-equal at rb=512 **and rb=13312
(production)** + row2 fail-closed + warm `copy_count==0`; hotread cos **0.999999** (kernel untouched);
**graph-capture probe — 16 replays, captured-graph bytes + copy_status byte-identical to serial
(graph-safety proven)**; overlap **0.10 → 0.005 ms exposed, step-segment 0.41 → 0.31 ms (~1.30-1.33×)**
in the realistic decode regime, ~1.0× under a fully SM-saturating kernel.

**Net:** the live per-step swap-in is now a graph-safe, byte-identical kernel **overlapped** on a copy
stream (the dormant Python staging-gather path is removed); the working-set knee (P2) is wired, and
the freed-HBM admission loop (P4) is built but needs the scheduler hookup. Honest residual: true
zero-SM copy-engine swap-in is unreachable graph-safely with a device-resident schedule (would need
re-architecting `compact_miss_schedule` to emit a host-side run table pre-capture); the cross-layer
prefetch hoist is seam-ready but deferred; and live promotion still needs the multi-rank serving
proof (Gate-5-7), a model forward not a microbench.

### G2 — wide-window overlap (hoist before bmm+rope) + P4 scheduler seam

The G1 mechanism was correct but its overlap window was ~nil in the *real* decode flow: the swap-in
issued **inside** `map_topk_to_hot_pool`, after the decode `bmm(q_nope·k_b)+mla_rope_generation`, and
joined almost immediately. A follow-up determination settled the achievable win from code + numbers:
the **clean cross-layer hoist is dependency-BLOCKED** (the DSA `Indexer` is a per-layer `nn.Module`
with per-layer trained weights — L's top-k is unknowable until L−1 completes); **cross-step
speculative prefetch is locality-blocked** (measured step-to-step block reuse only 19–59%, and the
reused blocks are already LRU-resident, so the ~63% churn is unpredictable); **miss-reduction is
churn-bounded**. The one real lever is **widening the in-method window** — and the `bmm+rope` that
produces `fused_q` depends only on q/q_pe/latent_cache, **not** on the swap-in.

**G2 (committed `012c096b4`)** issues the WHOLE swap-in chain (planners + in-stream copy + commit +
build) on the coordinator copy stream at the **top of `forward_absorption_generation`, before the
bmm+rope**, and **defers** the main-stream join to just before the hot-read (`prepare_hot_pool_
overlapped` + `consume_prepare_join_event`: a dedicated fork/done pair keyed `(step_id, layer_idx)`;
`map_topk_to_hot_pool` gained an `on_copy_stream` kwarg so the submit launches in-stream with no
nested fork/join; the descriptor is stamped so `_sparse_mla_decode_kvarn_hot` reuses it via its
`descriptor_is_current` fast path and only takes the join). Issue + join derive the layer index from
the same `_hisparse_local_layer_idx`, so the key always matches and the join fires. **EXACT** (serial
== wide == captured-wide, byte-identical on hot pool + hot_indices + build/commit status, all regimes,
index_topk 256 and 1024, worst-case 5900+ miss blocks, cos 1.0); **graph-capture PASS** (re-gated
independently on GPU7); **off-switchable** (falls back to the G1 in-method path). No .cu/ABI change.
Overlap collapse **1.4–1.7×** when the bmm-shaped window is comparable to the exposed chain, toward
fully-hidden for larger windows; the precise live magnitude (real bmm duration vs real per-layer miss
volume) is a Gate-5-7 measurement. Since the mechanism is exact + graph-safe + off-switchable, it can
only help or no-op.

**P4 scheduler-admission seam (also G2):** `filter_admissible_requests` (a pure, side-effect-free
freed-HBM budget gate over the existing `can_admit_request`/`admittable_token_capacity` accounting) +
a `SimpleScheduler` optional `hisparse_coordinator` hook (`_apply_hisparse_gate` filters the capacity
scheduler's fitting set; **strict no-op when no coordinator is attached**, so non-HiSparse serving is
byte-for-byte unchanged) + `hisparse_num_prompt_blocks` (returns 0 when unsizable, never spuriously
rejects). 14 admission unit tests + 6 lever-c control-flow tests pass; the live wiring (plumb the
coordinator into the PyExecutor scheduler construction + the C++ `BindCapacityScheduler` path +
multi-rank release/readmit validation) is the remaining Gate-5-7 integration.

**Updated residual:** the live overlap magnitude and the P4 live wiring both need the multi-rank DSA
decode forward (Gate-5-7); live promotion (`resident_v1_ready()`, left false) likewise. The swap-in
overlap mechanism, the working-set knee (P2), and the freed-HBM admission accounting + seam are all in
place and verified in isolation.

## Companion track — KVarN-GQA packed decode (24.7× dense + FWHT/sparse/split-K, bit-identical/graph-safe)

Run in parallel (separate worktree/GPU): the SMC-SD GQA-KVarN packed-decode kernel
(`kvarnGqaKernels.cu`). The first pass took **dense 1470 → 59.5 µs (24.7×),
bit-identical** (kill the per-(token,dim) shared-mem `atomicAdd` + double-K-dequant
M1, algebraic scale-factoring M2, SMEM code-plane staging M3). A second pass applied
the same FWHT insight plus SMEM-staged sparse and split-K — every lever independently
re-gated by the orchestrator on GPU 7 vs the PyTorch reference (atol 7.5e-2) and for
CUDA-graph replay:

| kernel | M3 → final | speedup | gate |
|---|---|---|---|
| STORE | 1601 → 437 µs | 3.67× | byte-exact, graph byte-delta 0 |
| BDR dequant | 205 → 73 µs | 2.82× | churn_max_abs 0, replay 0 |
| dense decode | 59.5 → 18.5 µs (16.4 graphed) | 3.22× | ref 1.2e-4, replay 0 |
| sparse top-k | 75.8 → 20.5 µs | 3.69× | ref 4.9e-4, replay 0 |
| blocks=20 dense | 133 → 24.7 µs | 5.4× | replay 0 |

- **FWHT (exact):** every O(N²)=128² natural-order Hadamard matrix-sum → in-place
  FWHT butterfly (7 stages), `FWHT(x)==H@x` to 7e-15; removed the O(N³) per-tile
  store rotate and the BDR inverse-rotate (the 3.67×/2.82×).
- **SMEM-staged sparse top-k** lifted the previously-stuck path (was 1.22×) to
  **3.69×**: stage resident blocks' K/V code planes + scale vectors into shared
  memory once, gather scattered top-k from SMEM with touched-block detection
  (numBlocks ≤ 16, else fall back to global gather).
- **Flash-decoding split-K** fixed the M=1 dense occupancy starvation, backed by a
  CUDA-graph-safe persistent workspace; bf16 also gated. ABI +
  `kvarn_gqa_backend_ready()=false` untouched. Distinct from the dense-MLA HiSparse
  path. Committed to `op-trt-hisparse` (GitHub `4b35fe6e0`).

## Audit provenance

Orchestrator independently: (a) derived the 208 B/token constant from source and
confirmed every 1a headline number; (b) reproduced the hot-read attention-math
cosine (0.999999) on a separate GPU; (c) **caught and quantified the quant-accuracy
gap** the subagent's gate left open (0.906/0.890 Gaussian worst-case, RoPE 0.9997)
and established the real-KV caveat; (d) verified the 1b method uses the production
counting op against a true reference; (e) confirmed the kernel-latency blocker is
per-layer (×61); (f) for Phases 3–4 independently re-gated each HiSparse milestone
(M5_fwht / M14 / M15) and the full KVarN-GQA ladder on GPU 7 — reproducing every
speedup and confirming bit-identity / cos 0.999999 and CUDA-graph replay-0 before
each commit. All correctness claims are gated against TRUE references, never
self-comparison (the K2 / input_scale lesson).

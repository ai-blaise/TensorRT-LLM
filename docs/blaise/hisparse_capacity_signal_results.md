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
levers took the hot-read from ~348 ms/call to **0.360 ms/call (B16), ~22 ms/step —
A/B-viable, within ~10% of the ~20 ms c16 decode budget** (M19, Phase-4/5), at cos
0.999999 vs the true dense reference with the production ABI frozen. Going
comfortably under 20 ms/step would require a high-risk TMEM-UMMA tensor-core rewrite
(Phase-5).

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

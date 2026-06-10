# tok/s/user optimization candidates (post-first-token, c16)

Living plan for hill-climbing **tokens/sec/user after first token at 16
concurrent users** on the r20 disaggregated deploy
(`BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft`,
8×B200: prefill TP2×CP2 LayerSplit, decode TP4+EP4+attention-DP). This is the
companion to the per-component docs ([indexer](indexer.md),
[warpdecode](warpdecode.md), [kvarn](kvarn.md), [sparse_mla](sparse_mla.md),
[nvfp4_fusions](nvfp4_fusions.md), [topology_deploy](topology_deploy.md),
[moondream_pipelining](moondream_pipelining.md),
[request_pinning](request_pinning.md)) — it ranks the *open* levers and records
the plan, evidence, expected win, correctness gate, and status for each.

No speculative decoding anywhere in this target (no SMC-SD, no NextN/MTP, no
EAGLE, no n-gram). The metric is single-user decode latency under a full c16
batch, not aggregate throughput.

---

## Measurement methodology (read first)

Two harnesses, both client-side streaming timestamps from inside the frontend
pod (the decode `:9090` `trtllm_*` gauges are **stale/laggy** — never use them):

- **Mixed harness** (8×512, 4×4k, 4×16k prompts): realistic but streams finish
  at staggered times → batch composition changes mid-window → **±1.5 tok/s
  (~4%) variance**. Cannot resolve sub-5% levers. Use only for coarse checks.
- **Tight harness** (`tight_measure.sh`: 16 **uniform** 2048-tok prompts, 512
  out, temp 0, `ignore_eos`): all 16 stay in-batch the whole window → constant
  batch=16 → **sd 0.05–0.11, 5-round spread 0.17 tok/s (~0.4%)**. This is the
  A/B instrument for every lever below.

**Baseline (tight harness, image `c1c3c9`):** 39.98 tok/s/user, TPOT 25.0 ms,
sd < 0.1. Mixed-harness c16 ≈ 35–37 (higher-variance, longer-KV workload).
c1 ≈ 49.5 tok/s/user, TPOT 20.1 ms.

### The overhead-bound diagnosis

c1→c16 step time grows only 20.1→27.4 ms (1.36× for 16× the batch): the decode
step is **overhead-bound, not bandwidth-bound**. Recomputed per-rank HBM floor
at c16 ≈ **5.3–5.5 ms** (MoE routed-expert weight reads dominate: ≈20 distinct
experts/rank/layer × 24.8 MB × 58 layers ≈ 4.3 ms; ADP-unsharded attention
weights ≈ 0.85 ms; replicated bf16 lm_head ≈ 0.27 ms). So of the ~25 ms TPOT,
**~15 ms is attackable host/launch/sync overhead** — that is the target.

### Where the GPU time goes (eager c16 profile — supersedes the Indexer-dominant premise)

A fresh eager c16 profile of the decode step (real REAP weights, B200) gives
the per-bucket GPU-time split: **MoE ≈ 60 % (of which EP comm ≈ 40 % of the
step), dense proj GEMMs ≈ 19.8 %, glue/elementwise ≈ 13.5 %, Indexer ≈ 4 %,
HISA ≈ 0.7 %**. This **replaces the campaign's original "Indexer is 50–74 % of
TPOT" premise**, which described the pre-campaign state — after the
[indexer.md](indexer.md) wins plus the fp16-logits and C++-top-k routing below,
the Indexer stack is a single-digit slice. The ranked levers now follow the
profile: MoE/EP comm (M3), proj GEMMs (B1, shipped), glue (G1 shipped / G2
open), host overhead.

### The two batch regimes (DP4 vs TP16)

Two attention-batch regimes are measured separately, because levers move
differently in each:

- **DP4 (attention-DP, the live r20 decode topology):** c16 splits to
  **bs=4/rank** attention.
- **TP16 (pure-TP attention):** every rank runs the full **bs=16**.

**The production plan targets WarpDecode + TP**: TP-rank attention measured
*faster* than ADP at equal load (the autotuner re-selects tactics per shape —
no mispick at bs=16), and pure TP drops the per-step ADP host collectives
entirely (which is what moots S2 below). Per-lever microbenches are therefore
taken at **bs=4 AND bs=16**; TP-regime-specific costs (e.g. the KVarN
pre-replay scan, C2) are called out explicitly.

### Correctness gating (untrained checkpoint)

The graft checkpoint emits garbage-class text and MoE routing is run-to-run
non-deterministic, so **end-to-end text parity is meaningless**. Every
correctness-affecting change is gated **kernel-level, two-stage**: (1) top-k
SET match (Jaccard / recall@`index_topk`) against the reference path, and (2)
bit-exact cache-slice / cosine ≥ 0.9999 on the dense read where applicable.

---

## Production target — all custom pieces ENABLED and maximally optimized

The production deployment runs the full custom stack **on**, each in its most
optimal mode — optimization means making each piece *engage intelligently*, not
disabling it:

| Piece | State | Optimal-mode note |
|-------|-------|-------------------|
| LayerSplit (prefill) | on, CP2×TP2, owner-local, read-set broadcast | + CP=2 IPC push primitive (C9) |
| WarpDecode (decode) | on, forced `decode_1cta`, fixed tactic | persistent-megakernel is the structural ceiling; megakernel FC2 N-tile default is now 256 (160 numerically broken, cycle 5) |
| dense KVarN `kvarn_k2v2` | on, amortized restore | host-gated pre-replay scan (C1); delta-restore (C2) is the TP-regime follow-up |
| Indexer IndexCache + FSSS | on, `index_topk_freq=4` | escalation to 8 under recall gate |
| **HISA** | **on whenever the Indexer is on; capture gate + candidate width track live kv via `metadata.max_gen_kv_len` (cycle 4); engages at kv ≥ `hisa_min_seq_len` (default 32768)** | **never slower than off, wins whenever active (forced-on gate=1024 proven never-slower, up to 2.56× at 33k); per-step invariant memo shipped `3e03d665d`** |
| NVFP4 indexer-K (MX E2M1+UE8M0) | on | score→top-k fusion measured net-zero under graphs — killed |
| Indexer decode top-k | on — prod live-kv routes to vanilla C++ (`841f9874a`), DSL only at kv ≥ 16k | fp16 logits (`indexer_logits_dtype=auto`→fp16 on the DSL path, `3e03d665d`) |
| MLA / MLP / indexer proj GEMM backend | on — cuBLASLt forced for the NVFP4 proj Linears (`TRTLLM_MLA_PROJ_NVFP4_BACKENDS` + `TRTLLM_DSV3_MLP_NVFP4_BACKENDS` + `TRTLLM_INDEXER_NVFP4_BACKENDS`, default `cublaslt`) | bit-identical, 1.2–2.1× per GEMM (B1, `4220bf4bd`+`3e03d665d`) |
| Gated-norm / glue | on — `fused_lowrank_gate` with the CuTe DSL kernel as default impl (`TRTLLM_OPTRT_LOWRANK_GATE_IMPL=cute`, `33e801fd1`) + fused sigmoid·mul at both attention gate sites | −91.6 % on the gated-norm chain (G1); PRE_MOE_FUSION absorb is the follow-up (G2) |
| Shared-expert swiglu+FP4-out fusion | on at decode M — `_FP4OUT_MIN_M=128` guard lifted (`fd705a6f5`) | exact vs TRUE-f32 (cos 1.0, max_abs 0.0) at every m |
| MoE EP comm | NVLINK_TWO_SIDED today; DeepEP low-latency enablement shipped (`51918fba2`), production flip is M3 | LL needs `TRTLLM_DEEP_EP_TOKEN_LIMIT` = per-rank concurrency (16), NOT max_batch_size |
| NIXL transport + request pinning + Moondream overlap | on | generation-first/write-mode is the open gate |

HISA stays **enabled whenever the Indexer is on**. HISA is a core part of the
model's serving path and, implemented correctly, *wins* at every indexed length
with cost that scales with sequence length. The cycle-4 capture-gate fix
(`metadata.max_gen_kv_len`, `8db5cd77f`) made our HISA scale (its candidate
width was frozen at the max-context value) and fixed wrong selection at
kv > 33k; it does **not** gate HISA off anywhere the Indexer runs. Track H
(deep HISA optimization) remains open but is **re-scoped by the eager c16
profile**: HISA is ≈ 0.7 % of the decode step, so H3x are small-candidate work,
no longer the top hill-climb priority.

---

## Priority sequence

1. **Finish LayerSplit (Part 1 of the goal):** C9 CP=2 IPC broadcast GPU
   re-validation → push; then L1 (z.ai dense-broadcast overlap re-measure) and
   L2 (read-set block-id hoist). LayerSplit must be correct + optimal before the
   decode hill-climb is the focus.
2. **MoE / EP comm is the top open hill-climb lever (per the eager c16
   profile: MoE ≈ 60 % of the step, EP comm ≈ 40 %).** M3 — the DeepEP
   low-latency production flip (enablement shipped `51918fba2`; the open item
   is the `TRTLLM_DEEP_EP_TOKEN_LIMIT` c16-vs-max_batch sizing decision).
3. **The TP-regime cost: C2 KVarN delta-restore** (the bs=16 pre-replay scan
   is ~4 ms/step amortized at TP — the single biggest TP-regime cost), then
   **G2 gated-norm → PRE_MOE_FUSION** (~1–1.5 ms/step) and **I5 indexer wk+wp
   fused GEMM** (verification in progress).
4. Track H (HISA) — re-scoped small by the profile (HISA ≈ 0.7 % of step);
   H3a/H3b remain valid low-risk candidates, no longer the lead. Then the
   remaining host/kernel levers (N1, I2, M1, K1), then the structural
   megakernel (P1). S2 is moot under the WarpDecode+TP production plan (held).

## Ranked candidates

| # | Candidate | Layer | Expected win @ c16 | Status |
|---|-----------|-------|--------------------|--------|
| **M3** | **DeepEP low-latency production flip** (`TRTLLM_FORCE_COMM_METHOD=DEEPEPLOWLATENCY` + `TRTLLM_DEEP_EP_TOKEN_LIMIT=16`) | comm | LL roundtrip 45 vs 115 µs/layer @ limit 16 (~2.5×) — **INVERTS at limit 64** | **enablement SHIPPED** `51918fba2`; flip pending the c16-vs-max_batch sizing decision |
| **C2** | **KVarN delta-restore** (restore only the changed rows/blocks) | scheduler | bs=16 pre-replay scan ~4 ms/step amortized at TP — the biggest TP-regime cost | impl, opt-in; **5-scenario equivalence verification pending** |
| G2 | Gated-norm → PRE_MOE_FUSION absorb/fold (gate measured 0.500 ± 0.0025) | glue | ~1–1.5 ms/step | design (absorb-or-fold) |
| I5 | Indexer wk+wp fused GEMM (one launch + one read of x per F-layer) | indexer | small per-F-layer | impl; **verification in progress** (`build()` gating vs the production loader under investigation) |
| C9 | CP=2 IPC push broadcast | prefill TTFT | 1.2–3× the per-layer broadcast | impl, GPU re-validating |
| B1 | cuBLASLt NVFP4 backend force: MLA proj + shared/dense MLP + indexer proj | GEMM | ~1.62 ms/tok (`4220bf4bd`) + 1.62 ms/tok incremental (`3e03d665d`, TP4); bit-identical | **SHIPPED** `4220bf4bd`+`3e03d665d` |
| G1 | Gated-norm + glue fusions (`fused_lowrank_gate` −91.6 %, fused sigmoid·mul, HISA invariant memo) + CuTe DSL port (−36 % vs Triton, default impl) | glue | −6.27 (bs4) / −9.38 (bs16) ms/step eager GPU | **SHIPPED** `3e03d665d`+`33e801fd1` |
| I6 | fp16 indexer logits (`indexer_logits_dtype`, auto→fp16 on the DSL path) | indexer | top-k −15…−22 % @ kv ≥ 33k; logits buffer halved | **SHIPPED** `3e03d665d` |
| I7 | Prod decode top-k → vanilla C++ (drop the stale width-override) | indexer | ~1.7× top-k @ prod live-kv (~29–33 % of the top-k pipeline) | **SHIPPED** `841f9874a` |
| K3 | Shared-expert swiglu+FP4-out fusion at decode M (`_FP4OUT_MIN_M` lift) | kernel | ~100 µs/step + 58 act-quant launches removed | **SHIPPED** `fd705a6f5` |
| C1 | KVarN pre-replay restore host-gate | scheduler | ~0.75–2 ms host (within harness noise) | **shipped** `a1b13ea78` |
| C3 | Cache debug env-gates / no eager kwargs | scheduler | ~0.1 ms/step | **shipped** `0adc87009` |
| H3a | PDL-chain the 5 HISA glue kernels (candidate_pages/mask/remap/block_reps/block_scores; indexerHisaNvfp4.cu has 0 PDL, indexerTopK.cu has 9) | indexer | ~0.11–0.17 ms/step (7 boundaries × ~1–1.5µs × 16F) | candidate (re-scoped: HISA ≈ 0.7 % of step) |
| H3b | Per-row live-length candidate scaling (caller-only: per-row `candidate_context_lens`/`selected_lengths` from `prefix_lens`; kernels already walk `[0,num_kv)` — verified fp4_paged_mqa_logits.py:1422 / indexerTopK.cu:663) | indexer | long-band topk radix→insertion 14→6.9µs (real); GEMM 7.2× (latency-bound caveat) | measure in mixed/long-band regime |
| H3c | Incremental block-rep quantize (only the boundary block changes/step; indexerHisaNvfp4.cu:279 rebuilds all) | indexer | ~0.08–0.11 ms/step | candidate |
| H4 | HISA `compression_ratio` sweep {4,6,8,12}, recall-gated (`hisa_block_topk=64` is dead config when ratio>0) | indexer | ckpt-specific; config-only | candidate |
| ~~H1~~ | ~~short-band candidate-width allocation shrink~~ | indexer | **REGRESSED −6% (40.28→37.82)** — short kv is latency-bound | **discarded (measured)** |
| S2 | Collapse 10 host MPI collectives → ~3 | scheduler | 150–400 µs + 7 barriers of jitter | **MOOT under WarpDecode+TP** (no ADP collectives); held with that note |
| N1 | NUMA-pin decode workers to node 1 | system | 0.3–1 ms + jitter | planned (rides manifest) |
| I2 | `index_topk_freq` 4→8 | indexer | indexer-cost −68% on S-steps (~) | planned (recall gate) |
| M1 | MoE A2A two-sided → one-sided + workspace combine | comm | 0.3–0.9 ms/step | planned (sequence after M3) |
| K1 | PDL coverage completion | kernel | +1–3% | planned |
| ~~K2~~ | ~~FC2 N-tile 256→160~~ | kernel | **N=160 is numerically broken** (SFB miscompute, cos 0.790 vs TRUE f32; + prefill M=1024 OOB) | **KILLED** `29f492b49`→`d01737397` — see killed list |
| L1 | z.ai dense-broadcast overlap | prefill | TTFT (exposed indexer-K broadcast) | planned |
| P1 | Persistent decode-layer megakernel | kernel | dispatch fusion itself Δ≈0 under PDL (the −13.9% previously attributed to it was K2's broken N=160); structural persistent-kernel case unchanged | **code shipped, opt-in** (`MOE_MEGAKERNEL`); FC2_N default fixed 160→256 `e105fd7a1` |
| MO1 | MORI-style generation-first / write-mode handoff | transport | TTFT (overlaps RDMA with prefill) | needs router build |

---

## H1 — HISA candidate-width band-scaling (MEASURED REGRESSION, discarded)

**Result (tight harness, sd<0.1): −6%.** Production HISA-on (candidate 33024,
gate 65536) = **40.28 tok/s/user** (TPOT 24.83 ms); HISA-scale (candidate
band-sized→2048, gate 1024) = **37.82** (TPOT 26.44 ms). Shrinking the candidate
width made decode *slower*. Why: at the harness kv (~2560) production's
`candidate_len=33024 ≥ live kv`, so HISA was doing *exact, unfiltered* selection
— yet faster than the 2048-wide *approximate* (filtered) version. Less work being
slower ⇒ the decode candidate score+topk are **launch/latency-bound** at b≈4/rank
(kernel launch + tcgen05 pipeline fill dominate), not width-bound; shrinking the
width buys ~nothing while the band/gate/filter path adds overhead. The reference
chart's width-scaling reflects a throughput-bound regime (large batch / long
context); c16 decode at b≈4 is latency-bound. **Code change discarded; not
pushed. `hisa_min_seq_len` reverted to 65536; production restored to 40.28.**

**Redirect → H3:** the decode HISA lever is the ~128 kernel launches/step (16
F-layers × ~8 kernels), not the candidate width. Fuse/PDL-chain the pipeline.

---

## (historical) original H1 framing

**The correct framing (HISA stays ENABLED; the defect is that our HISA doesn't
scale).** HISA, implemented right, *wins* tok/s/user at every indexed length and
its cost *scales* with sequence length (validated on a reference stack + the 4:1
compression chart: ~0.65 ms @8k → ~2.6 ms @64k, below the plain DSA path at all
lengths). Our HISA cost was instead **flat at the max-context value** for every
length — an implementation defect, not a reason to gate HISA off.

**Root cause (triple-confirmed: P2-I2, P2-I3 independently, + direct read).**
`candidate_len`, the width of the HISA candidate score + top-k, is *designed* to
scale: `candidate_len = _hisa_block_topk(ceil(max_kv_len/block)) * block`
(`dsa.py:2559-2561`), and `_hisa_block_topk` (`dsa.py:2140-2147`) scales with
`max_kv_len` via `hisa_compression_ratio`. But under CUDA-graph capture
`max_kv_len = block_table.shape[1] * k_cache.shape[1]` — the **static pool
width 132096** (`dsa.py:2549`) — not the per-graph band kv. So `candidate_len`
froze at `258 × 128 = 33024` for *every* sequence length, and the candidate GEMM
(`fp8_fp4_paged_mqa_logits`, `dsa.py:2632`) + candidate top-k
(`indexer_topk_decode`, `dsa.py:2684`) paid 33024-wide cost on a 4.6k context —
where they should pay ~2k. (Eager warmup used live kv, which is why HISA looked
"off at prod" in eager measurements.) The Indexer itself is on iff
kv > `index_topk` (1024) via `skip_indexer_for_gen_reqs` (`dsa.py:1365/1439`);
HISA should track that exactly.

**Cost recovered.** Short-band candidate work 33024 → ~2048 (16×). Per-F-layer
HISA was ~45–66 µs over the plain path × 16 F-layers ≈ 0.4–1.1 ms/step (~2–4%
TPOT); scaling the width recovers the bulk of it while keeping HISA on — so
HISA becomes a net *win* at prod kv (scores ~64 block-reps + ~2048 candidates
vs the plain path's ~4608), matching the chart.

**Fix landed (HISA stays ON, made to scale).** The bands already exist —
`seq_len_threshold` auto-defaults to 8192 (`llm_args.py:564-599`,
`needs_separate_short_long_cuda_graphs` returns `skip_indexer_for_short_seqs`),
so short-band graphs warm at ~8192 — but the HISA gate ignored it.

1. **Code** (`dsa.py:2536-2560`, landed): thread `metadata.max_gen_kv_len` (the
   per-graph band kv, the same value `_indexer_logits_width` uses at
   `dsa.py:3485/3721`) into `_hisa_topk_from_nvfp4_cache` and use it for
   `max_kv_len` under capture instead of the static pool width. Falls back to
   the pool width when unavailable. → `candidate_len` scales per band.
2. **Config** (landed): `hisa_min_seq_len` 65536 → 1024 (= `index_topk`) so HISA
   stays **on** for the short band, tracking the Indexer's gate. (Without this,
   the code fix would have turned HISA *off* on the short band — wrong.)

**Correctness (intel two-stage gate, required before push).** At short kv the
old `candidate_len ≥ live kv` meant *no* block filtering (accidentally exact);
the fix restores HISA's *intended* block-prefilter approximation there. Gate:
top-1024 SET recall of band-scaled HISA vs the exact plain path on synthetic
kv ∈ {2k, 4.6k, 8k} must be high (HISA preserves the top tokens by design); long
band (≥8192) unchanged.

**Measurement.** Tight uniform-prompt harness (sd < 0.1): production HISA-on
baseline 40.28 tok/s/user (TPOT 24.83 ms); HISA-scale image
(`hisascale`, candidate band-sized + gate 1024) A/B in flight — expect *higher*
(short-band candidate work drops 16×). Push to op-trt as a verified win once the
A/B shows the gain and the recall gate passes.

**Track H continues (deep HISA optimization — since re-scoped small by the
eager c16 profile, HISA ≈ 0.7 % of step):** H2
(per-row continuous candidate scaling — make the candidate GEMM/topk live-kv
length-aware like the plain path's `kAdaptiveFinalSort`, so cost scales
*continuously* with kv, not stepwise per band), H3 (8-kernel-pipeline
fusion + PDL), H4 (compression/block-topk/block-size tuning, recall-gated) are
under active deep investigation (probe P2-HISA).

**Composability.** Independent of WarpDecode, kvarn, ADP, LayerSplit. Composes
with I2 (freq) and the seq_len_threshold short-band (which also revives the
width-correct C++ insertion top-k that HISA currently shadows).

---

## C1 — KVarN pre-replay restore host-gate (SHIPPED)

`_restore_kvarn_before_cuda_graph_replay` looped all 61 layer modules before
**every** graph replay; each `kvarn_restore_for_decode` (`dsa.py:4249`) pays
3–4 implicit device syncs (masked-selects + `torch.unique`) before its
empty-set early-exit can fire — ~200 exposed host-blocking syncs/step in steady
state. Under amortized restore (`mla_latent_kv_amortize=true`, live) the pools'
`valid`/`commit_gen` sets change **only** when the scheduled row composition
changes (onboard/free) or a sequence crosses a block boundary — both visible
from host metadata. Fix: an O(B) host-integer step key
`(request_ids, kv_len//tokens_per_block per req)`; identical key ⇒ restore set
provably empty ⇒ skip the 61-layer scan. Key completeness:
`commit_gen` bumps ⟺ block-boundary crossing ⟺ `kv_len//tpb` increments;
row-set change ⟺ `request_ids` changes. Any missing attribute falls through to
the full scan. **Shipped** `a1b13ea78` (op-trt). Within tight-harness noise at
c16 (host-bound regimes benefit more); removes real sync work, cannot regress
correctness.

## C2 — KVarN delta-restore (impl opt-in, verification pending)

C1 skips the pre-replay restore scan when *nothing* changed; when something
*did* change, the scan still walks/restores far more than the delta. At **TP
bs=16** (every rank sees the full batch) the pre-replay scan is **~4 ms/step
amortized — the single biggest TP-regime cost** (at DP4/bs=4 it is much
smaller, which is why C1 alone sufficed there). Delta-restore restores **only
the changed rows/blocks** instead of re-deriving the full set. Implemented
**opt-in**; before any default flip it needs the **5-scenario equivalence
verification** (delta vs full restore: onboard, free, block-boundary crossing,
recycle/re-commit, mixed) — KVarN correctness is cache-content correctness, so
the gate is bit-equality of the restored pool, not a cosine. See
[kvarn.md](kvarn.md) for the restore architecture this extends.

## C3 — Cache debug env-gates (SHIPPED)

`TRTLLM_OPTRT_KV_DEBUG` / `TRTLLM_OPTRT_MODEL_ENGINE_ADP_DEBUG` were re-read from
the environment every call, and 50+ call sites (several per-step) evaluated
their kwargs eagerly (a scheduled-request-id list comprehension every iteration;
per-step request/metadata summary strings). Cache the gates at import,
short-circuit the summary helpers, gate the per-iteration executor call at the
call site. **Shipped** `0adc87009`. ~0.1 ms/step, zero correctness risk.

## C9 — CP=2 IPC push broadcast (impl, GPU re-validating)

The directive-7 bench showed a single cross-device copy beats NCCL
`dist.broadcast` 1.2–3× at CP=2 (NCCL wins serial fan-out at CP≥3). Owner-push
channel for the per-layer active-block broadcast: per-rank `[depth, slot]`
uint8 staging rings + int64 sequence mailboxes exported once via
`reduce_tensor`; steady state = gather → one cross-device copy into the peer
ring slot → 8-byte pinned-host `cuMemcpyAsync` sequence publish; consumer
`cuStreamWaitValue64(GEQ)` then scatters. Sequence mailboxes (not IPC events:
`cudaStreamWaitEvent` snapshots at host call time and reads stale on host skew)
make it host-skew-immune; the publish is a memcpy not `cuStreamWriteValue64`
because stream memops reject CUDA-IPC-imported addresses
(`CUDA_ERROR_INVALID_VALUE`, observed on B200 — the first GPU run; fix applied,
re-validation pending). 4-phase unanimous setup (loopback memop probe, handle
exchange w/ UUID + bidirectional-P2P check, agreement vote, live roundtrip),
NCCL fallback on cp_size≠2 / capture / oversize / kill-switch / any failure,
symmetric on both ranks. Local commit `8bcda0809`, **held until GPU
re-validation** (`c9_test.py`, 2-GPU, 5 cases incl. ring-wrap + float8 alias +
fallback).

---

## B1 — cuBLASLt NVFP4 backend force (SHIPPED `4220bf4bd` + `3e03d665d`)

The largest real-compute decode bucket after MoE: dense proj GEMMs are 19.8 %
of the eager c16 GPU profile, and Linear's NVFP4 auto-selection
(`nvfp4_allowed_backends=['cutlass','cublaslt','cuda_core']`) was picking
cutlass at decode shapes. cuBLASLt is **bit-identical** to cutlass on these
GEMMs (max|diff| == 0 vs cutlass AND vs a true-f32 reference at M ∈ {1,4,512})
and **1.33–2.07× faster, flat from M=1 to M=1024** — so it holds at decode
bs=4, bs=16, and prefill, under both attention-DP and pure TP.

- **`4220bf4bd`** forced `['cublaslt']` for the MLA q_a/q_b/kv_a/kv_b/o_proj
  Linears (`TRTLLM_MLA_PROJ_NVFP4_BACKENDS=cublaslt`, default-on): per-layer
  proj GEMMs 63.0 → 36.5 µs, **~1.62 ms/token (~6 % of the ~27 ms TPOT)** at 61
  layers.
- **`3e03d665d` (lane 1)** extended it to every remaining mispicked Linear —
  and fixed a bug that made `4220bf4bd` **partially dead**:
  `DeepseekV32Attention` RE-CREATES `kv_a_proj_with_mqa` after `MLA.__init__`,
  dropping the backend override. Passing the backends through gives 2.07–2.12×
  on the fused_a GEMM (verified through the runtime `DeepseekV3Linear`). Plus
  shared+dense GatedMLP (`TRTLLM_DSV3_MLP_NVFP4_BACKENDS`, 1.5–2.1×) and the
  indexer wq_b/wk/weights_proj (`TRTLLM_INDEXER_NVFP4_BACKENDS`, 1.18–1.30×).
  All bit-identical to the default backend (max|diff| == 0 at M ∈ {4,16}).
  **+1.62 ms/token (TP4) incremental** on top of `4220bf4bd`.

The bf16 GEMMs were audited in the same pass: lm_head / gate_proj / router are
already optimally dispatched — no lever there. Companion negative: NVFP4-
*quantizing* the remaining dense MLA proj GEMMs fails accuracy (cos 0.63–0.83;
see killed list) — the win is the backend, not more quantization.

## G1 — Gated-norm + glue fusions (SHIPPED `3e03d665d` + `33e801fd1`)

The REAP rank-16 gated-norm (`_maybe_apply_gated_norm`, 2/layer, plus the MLA
attention-output gate) was **~7 ms/step of fp32 SGEMM + cast/copy soup at c16
hiding in the dense-proj profile bucket**: a pathological `gemmSN_TN`
(40.6 µs/call × 122/step) that re-cast a 458 KB weight to fp32 on EVERY call.

- **`fused_lowrank_gate`** (`3e03d665d`): `x*sigmoid(silu(x@Wd)@Wu)` in 2
  kernels, split-K, weights pre-cast once — 49.7 → 6.05 µs bs4 / 74.3 → 6.27 µs
  bs16 (**−91.6 %**), bit-identical (fp64 ref cos 0.9999971 for fused AND
  eager). Plus **fused sigmoid·mul** at both attention gate sites and the
  **HISA per-step invariant memo** (row_to_batch / prefix_lens / block-counts /
  gather indices computed once per step, not 61×; capture-aware key). Measured
  **−6.27 ms/step (bs4) / −9.38 ms/step (bs16)** eager GPU total; all env-gated
  default-on.
- **CuTe DSL port** (`33e801fd1`, the repo-standard kernel language): single
  launch, CTA-cluster (grid (CN,M), CN=7/row; split-K rank-16 down-proj,
  128-bit bf16 loads + fp32 FMA, cross-CTA partials via st.async distributed
  SMEM + mbarrier — no global round-trip, no second launch; the full
  silu→up→sigmoid→mul epilogue fused, rounding points mirror eager exactly).
  Graph-replay B200: 3.96 µs (M=4) / 4.17 µs (M=16) vs Triton 6.18/6.74
  (**−36/−38 %**), wins at every M ∈ {1,4,16,64}; ~0.27 ms/token additional at
  2 calls × 61 layers. **BIT-IDENTICAL to the Triton kernel** at every
  (M, scale, CN) tested; vs eager bit-identical at M ≤ 16, 1 elem/458752 1-ulp
  at M=64 (same tie as triton-vs-eager). Default impl
  (`TRTLLM_OPTRT_LOWRANK_GATE_IMPL=cute|triton|eager`, auto-fallback to Triton
  when DSL unsupported; `TRTLLM_OPTRT_LOWRANK_GATE_CUTE_CLUSTER` tunes CN; JIT
  warmup helper compiles pre-capture). tcgen05/TMA were *rejected* at
  rank-16/M≤16 — latency-bound (CN sweep CN1 10.6 µs → CN7 3.95 µs confirms).

## G2 — Gated-norm → PRE_MOE_FUSION absorb/fold (design)

With G1 shipped, the residual lever is structural: the gate output is measured
**0.500 ± 0.0025** across real activations — i.e. the sigmoid sits at its
midpoint, so the gate is (near-)absorbable. Two designs: **absorb** the
constant 0.5 into the adjacent scale (validity gated on the ±0.0025 band being
checkpoint-stable), or **fold** the lowrank-gate computation into the
PRE_MOE_FUSION region (the fused add+RMSNorm+quant pass,
[nvfp4_fusions.md](nvfp4_fusions.md) #13) so the gate rides an existing kernel
instead of its own 2 launches × 122 sites/step. Expected **~1–1.5 ms/step**.
Gate: bit-equality (fold) or bounded-delta vs the G1 path (absorb).

## K3 — Shared-expert swiglu+FP4-out fusion at decode M (SHIPPED `fd705a6f5`)

The shared-expert swiglu+fp4-output fusion was gated to m ≥ 128
(`_FP4OUT_MIN_M`) by an OOB claim ("SFC epilogue does not predicate writes when
m < CTA tile height") that is **false for the production call path**: forward()
sizes C to pad_up(m, cta_m) rows and SFC to pad_up(padded_m, 128), covering
every full-tile and cluster-spill write; small-m partial tiles are the same
code path as the last partial tile of any m % cta_m ≠ 0 prefill shape.
B200-verified (real REAP shared-expert weights, GPU driver run):
**cos(fused vs TRUE-f32 reference of the same math) = 1.0 with max_abs = 0.0 at
every m ∈ {1,4,16,64,128}**; fused-vs-unfused 0.9998+ (the delta is the unfused
chain's own extra quantize step); **oob_demo** — direct kernel launches with
TIGHT (unpadded beyond contract) C/SFC allocations across tilers
{(128,128),(256,128)} × clusters at all five m — no fault. Timing: 1.7–1.9 µs/
layer at m=4/16 ⇒ **~100 µs/step across 58 MoE layers + 58 act-quant
launches/step removed**. Guard lifted so decode (m=4..16) takes the fusion.
Documented as [nvfp4_fusions.md](nvfp4_fusions.md) #13b.

---

## Scheduler / host overhead

### S2 — Collapse host MPI collectives (HELD — moot under WarpDecode+TP)
**Status update: MOOT under the WarpDecode+TP production plan** — the
collectives below are the *attention-DP* lockstep set; pure-TP attention has no
ADP collectives, so the lever evaporates at the target topology. Held with that
note (it returns only if the topology decision reverts to ADP); do not invest
while the production plan is WarpDecode+TP.

Original analysis (ADP regime): 10 python MPI object collectives/iter, 8
**before** forward launch (rank-state
AG `py_executor:3492`, can_queue AG `:2342`, first-token gather `:3089`, pad AG
`cuda_graph_runner:597`, maybe_get AG `:356`, num_tokens AG `model_engine:2001`,
ctx AG `:2014`, bcast `request_utils:582`); pad/maybe_get/can_queue/num_tokens/
ctx carry redundant scalars. Gather one tuple at schedule time and plumb. Win
150–400 µs/step **plus** removing 7 lockstep barriers (any-rank jitter ×10/step
currently stalls all 4). `TRTLLM_OPTRT_SKIP_ATTENTION_DP_CG_TP_ALLGATHER` exists
for 2 of them but is unsafe alone (padding needs the cross-rank max). Risk:
collective-count symmetry across ranks.

### N1 — NUMA-pin decode workers (planned, rides next manifest edit)
Decode GPUs 4–7 are NUMA node 1, but rank-0 had 110/115 threads on node 0
(unpinned drift; cpuset 0-223, no cpu limit, `nr_throttled=0`). Cross-UPI
pinned-buffer staging + wake jitter. Fix: cpuset/membind in the DGD
`extraPodSpec`. Zero code, 0.3–1 ms + jitter.

### S5 — Per-iter stats/perf-metrics decimation (planned)
`iteration_stats_interval=1` + per-iter `IterationStats` pickle on the rank-state
AG + per-request `append_step_metrics`/`update_perf_metrics` every iter + 3 CUDA
timing events/step. Sample 1/16; keep the cheap host/device step-time fields.
100–250 µs/step, hours, low risk.

---

## Indexer

The Indexer is ≈ 4 % of the eager c16 decode GPU profile (HISA ≈ 0.7 %) — the
remaining levers here are small and ranked accordingly.

### I6 — fp16 indexer logits (SHIPPED `3e03d665d`)
Config-matched precision: the model is bf16 with 4-bit indexer keys, and the
fp32 logits store was the only fp32 element in the scoring→top-k pipeline —
costing 2 extra radix rounds in the DSL top-k. New
`DeepSeekSparseAttentionConfig.indexer_logits_dtype` (`auto|fp32|fp16|bf16`;
auto → fp16 on the DSL path, fp32 on the DeepGEMM fallback). Gates: top-1024
recall vs a TRUE-f32-scored reference equals the pre-existing fp8-scoring
noise floor (|Δ| ≤ 0.0005); downstream attention cosine 1.000000; max|logit|
~360 vs the 65504 fp16 ceiling. **Top-k −15…−22 % at kv ≥ 33k** (e.g. B16/33k
27.9 → 21.7 µs), zero short-kv regression, logits buffer halved. This also
moved the C++→DSL top-k crossover (`_DSL_TOPK_MIN_KV_LEN`) 32768 → 16384.

### I7 — Prod decode top-k → vanilla C++ (SHIPPED `841f9874a`)
The decode top-k dispatch OR'd a kv-length gate (prefer C++ below
`_DSL_TOPK_MIN_KV_LEN`) with a width gate (force the CuTe DSL kernel whenever
the padded logits width ≥ `_DSL_TOPK_MIN_COLS`=12288). Production runs the
logits at max_seq_len=132096 every step, so the width gate fired
unconditionally and routed the entire prod regime to the CuTe kernel. The
premise behind the width gate ("C++ jumps to 16.4 µs at width ≥ 12288 and
stays flat") is **false**: direct B200 measurement (3-seed, CUDA-graph replay,
true head-to-head vs the C++ else-branch) shows the C++ `indexer_topk_decode`
is **~width-INDEPENDENT (~11 µs flat)** — it walks only `[0, live_kv)` per row
— while the CuTe kernel's cost scales with the padded width. At the prod
operating point (width 132096, live kv ~4.6k) **C++ is ~1.7× FASTER with a
bit-identical selected set** (top-1024 recall 1.0). Width-override removed;
**~29–33 % faster indexer top-k pipeline at prod**, selection unchanged. The
CuTe kernel (incl. the `81cfeb88c` fused single-pass cluster top-k) now runs
only for genuinely long live kv, which the kv gate already routes to it.

### I5 — Indexer wk+wp fused GEMM (impl, verification in progress)
Fuse the indexer `wk` and `weights_proj` GEMMs (same input x) into one launch
+ one read of x per F-layer. Implemented; **verification in progress** — the
open question is `build()` gating vs the production loader (whether the fused
weight is constructed on the path the production checkpoint loader actually
takes), under investigation before any default.

### I2 — `index_topk_freq` 4→8 (planned, recall-gated)
FSSS is **cross-layer** (not cross-step): the doc's −39/60/68% @ stride 2/4/8
(`indexer.md` win #4) is *indexer-cost-relative* and traces to the XSTEP
prototype (`llm_args.py:329-341` docstring flags accuracy-must-be-validated).
Escalating to 8 reuses the F-layer top-k across more S-layers. Gate:
top-1024 SET recall vs `freq=1` ground truth on real-shape synthetic. Open
question: recency drift between F-layers at stride 8 — read
`indexerXstepRecencyPatch.cu`'s window and whether it must widen.

### I3 — `seq_len_threshold` short band (was: folds into H1; still valid standalone)
Currently unset → one effective band → width always 132096. Setting 65536
revives the width-correct C++ insertion top-k (`indexer.md` win #1/#2/#6)
on short-band F-layers: ~3–4 µs/F-layer × 16 ≈ 50–60 µs/step for kv≤8k traffic,
at 2× graph count. Doc's "2.08–2.30×" is top-k-kernel-relative, not TPOT.
(H1 itself was discarded; this band lever stands on its own, and I7's C++
routing already captures most of the top-k side at prod.)

### ~~I4 — Score→top-k fusion~~ (KILLED — net-zero under graphs)
Measured with the purpose-built bench (`bench_indexer_score_topk_fused.py`,
shipped in `3e03d665d`): **net-zero under CUDA graphs**. The mask is already
fused into the top-k kernel, the logits round-trip is < 1 µs at decode B, and
the launch overhead — the only remaining win — is hidden by graph replay.
Moved to the killed list; do not rebuild.

---

## MoE / communication

### M3 — DeepEP low-latency production flip (enablement SHIPPED `51918fba2`; flip pending)
The top open lever per the eager c16 profile (EP comm ≈ 40 % of the step).
`51918fba2` shipped the two fixes that unblock
`TRTLLM_FORCE_COMM_METHOD=DEEPEPLOWLATENCY` on the WARPDECODE decode path
(both inert when LL is not active):

1. **warp_decode overlay guard:** the LL adapter flattens its padded
   `[num_local_experts, ep*token_limit, H]` recv into top-1 rows with a
   num_slots sentinel; the trtllm_gen overlay requires token-major top-k and
   RAISED under policy=force. Now detects the LL layout and returns a
   skip-reason — the canonical CuteDslFusedMoE backend consumes it natively;
   moe_sort drops sentinel rows without inflating tiles (tiles_equal at EP2,
   xlayout cosine 0.99999).
2. **ConfigurableMoE oversize park/restore:** an oversize forward (e.g. the
   max_num_tokens warmup pass) *destroyed* the comm strategy and pinned the
   AllGather fallback for the rest of the process — silently evicting LL.
   Now parks the primary strategy and restores it on the next in-limit
   forward. Validation 6/6 (in-limit restore, repeated park/restore, non-LL
   keeps the legacy destroy-and-replace).

**Caveats closed** (EP2, B200): LL combine cosine vs f32 = 0.9999985 (PASS —
better than normal's 0.9999975); NVSHMEM NVLink-P2P fallback init clean; int32
topk_idx matches the vendored kernels.

**The sizing caveat (why the flip is not defaulted):** the LL win — **45 µs vs
115 µs per-layer roundtrip at token_limit=16 (~2.5×) — INVERTS at
token_limit=64 (225 µs)**. The padded transfer scales with the limit, so
production must set `TRTLLM_DEEP_EP_TOKEN_LIMIT` to the **actual per-rank
concurrency (16 for the c16 target), not max_batch_size**. Deploy delta:
`TRTLLM_FORCE_COMM_METHOD=DEEPEPLOWLATENCY` + `TRTLLM_DEEP_EP_TOKEN_LIMIT=16`.
Open decision before the flip: the c16-vs-max_batch sizing policy (what
happens when scheduled tokens/rank exceed the limit — the park/restore
fallback engages, but the limit choice sets how often).

### M1 — A2A two-sided → one-sided + workspace combine (planned, sequence after M3)
Deploy forces `TRTLLM_FORCE_COMM_METHOD=NVLINK_TWO_SIDED` (FIFO handshake, 4
ops/MoE layer) while the factory priority puts OneSided first
(`communication_factory.py:66-72`); `CuteDslFusedMoE.supports_moe_output_in_
alltoall_workspace()` returns True for NVFP4 (`fused_moe_cute_dsl.py:514`) and
`moe_scheduler.py:651-691` already wires combine-into-workspace (one fewer
payload copy/layer); `ConfigurableMoE` pins it False (`configurable_moe.py:524`).
A/B the env flip after establishing why two-sided was forced. 0.3–0.9 ms/step.

### M2 — ADP analysis at c16 (research, interacts with M1/N1/lm_head)
NVIDIA's pareto config at conc16 (`examples/configs/database/nvidia/
DeepSeek-R1-0528-FP4-v2/B200/1k8k_tp8_conc16.yaml`) runs TP8+EP8 **without** ADP
(ADP enters at conc256). At our c16 ADP costs 4× attention-weight reads
(+0.6 ms), replicated bf16 lm_head (+0.27 ms), 4 host barriers — vs ~0.4–0.7 ms
of MNNVL-able allreduce it saves. Net possibly −0.5–1.5 ms but topology-level;
sequence after M1/N1. If ADP stays: `enable_lm_head_tp_in_adp=true`
(`mapping.py:155`, impl exists) shards the 1.85 GB/step bf16 lm_head read 4× for
a ~230 KB allgather (~0.2 ms, config-only).

---

## Kernel / structural

### K1 — PDL coverage completion (planned)
Add `griddepcontrol` to `fusedRopeCatFp4.cu` and `indexerXstepRecencyPatch.cu`
(`indexerTopK.cu` already has it); nsys-audit the captured graph for PDL-off
pairs. Precedent 1.05–1.12× on the MoE op. +1–3%, low risk.

### ~~K2 — FC2 N-tile 256→160~~ (KILLED — N=160 is numerically broken)
**The FC2 N-tile lever is DEAD.** Full post-mortem in the killed list. Short
form: K2 shipped default-on in `2833bb0c5` on a decode-shape cosine gate, then
(1) **crashed the prefill autotuner at M=1024** (illegal memory access; 7168 is
not a multiple of 160, the final tile overruns 32 cols) → reverted to opt-in
`29f492b49`; (2) re-validation against a **TRUE f32 reference** showed
**cos ~0.790 at BOTH decode and prefill** — a broad SFB (weight scale-factor)
miscompute (SFB GMEM tiled by round_up(160,128)=256; only the {192,64} tile
branches carry the compensating odd-tile TMEM shift, so N=160 reads the wrong
32-col SFB sub-block) → N=160 made unreachable `d01737397`
(`_FC2_VALID_MMA_TILER_N` now (64,128,192,256); `TRTLLM_OPTRT_FC2_NTILE_160`
is a warning no-op). The original "win" was an artifact of the validator
comparing the buggy kernel against itself (fused-vs-sequential). **N=256 is
already optimal**; a correct N=160 needs an SFB layout tiled at 160 granularity
(CUTLASS-internal), not an epilogue tweak — not worth it.

### P1 — Persistent decode-layer megakernel (strategic, multi-day)
The TileRT "Breaking 1000 TPS" Leap-1: a persistent GPU program eliminates
per-kernel grid ramps + cross-op SMEM/TMEM round-trips (NOT dispatch fusion —
PDL ties that, confirmed by the mega-driver: fusion itself Δ≈−0.01 µs).
**Figure correction (cycle 5):** the previously-claimed −13.9 % vs the prod
2-kernel pair was K2's broken N=160 tile, not the fusion — with N=160 dead the
shipped megakernel op carries **no current measured win**; its case is the
structural persistent-kernel ceiling, unchanged. `e105fd7a1` fixed the JIT
default `TRTLLM_OPTRT_MOE_MEGAKERNEL_FC2_N` 160→256 (same SFB bug: cos 0.790
→ 0.99984 at the REAP shape), raises ValueError on an explicit 160, and
de-blinded `validate_fused_moe_megakernel.py` to compare against a true f32
einsum reference (it previously compared the fused kernel against a sequential
run of the SAME FC2 kernel — fusion-equivalence, blind to the SFB bug). The
broken 160 never ran in production (op-mode routes FC2 through the AutoTuner's
{128,256}). ~300–500 kernels/step × 1–3 µs
ramp ⇒ +5–10% near-term; `CursorWarpDecodePlan` (`warp_decode.py:90-153`) is the
landing contract, kernels unbuilt. Strategic ceiling: TileRT serves the same
model on the same 8×B200 at ~600 tok/s single-request (MTP-assisted) vs our ~50
c1 — multi-× headroom, consistent with the 20–40×-above-floor diagnosis. Phase
1: persistent per-layer worker grid (norm+quant→FC1→FC2→combine) from the
single-`@cute.jit` artifact + `decode_1cta` tiles. Composes with WarpDecode on
the MoE side; attention-side persistent chain (q_a/q_b/rope/gather/FMHA) is the
companion.

---

## Prefill / LayerSplit

### L1 — z.ai dense-broadcast overlap (planned, after C9)
z.ai's design overlaps the dense-KV broadcast behind indexer compute so only the
~1/8 indexer-K broadcast is exposed. op-trt defaults sync (M9 bench used
`cuda._sleep`-simulated compute). With the 1G read-set broadcast the indexer-K
payload grew, so re-measure sync vs overlap at the real 128k/large-batch shape
with real compute; activate if it wins. Affects TTFT, not TPOT. Same code region
as C9 — design together.

### L2 — Read-set block-id hoist (planned)
`_layersplit_compute_read_block_ids` is metadata-derived and identical across all
61 layers in a step but recomputed per layer. Hoist once per step. (The dense
top-k set legitimately differs per layer — only the read-set computation hoists.)

---

## Transport

### MO1 — MORI-style generation-first / write-mode handoff (needs router build)
The MORI-IO blog's best mode is **write mode**: the proxy dispatches prefill and
decode concurrently and prefill pushes KV layer-by-layer, so the RDMA transfer
overlaps prefill compute and only its residual adds to TTFT (read mode serializes
→ +1 full prefill pass). Our analogue is `generation_first` handoff, and the
campaign's own audit *requires* the `handoff_mode="generation_first"` marker and
*rejects* `completed_prefill`. The live deploy logs `completed_prefill`: the
`generation_first` string is **absent from the deployed Dynamo router binary**
(`_core.abi3.so` `prefill_router/mod.rs` exposes only completed-prefill
dispatch). Closing this needs a **new router build** (Rust), not just the python
slice the status doc assumed. TTFT lever only (not post-first-token) — tracked
but lower priority for this metric.

---

## Killed candidates (considered space)

- **FC2 N-tile 160 (K2, all forms)** — **numerically broken**: TRUE-f32
  reference shows cos ~0.790 at BOTH decode and prefill (vs ~0.9999 for
  {128,192,256}); root cause is a broad SFB miscompute (SFB GMEM tiled by
  round_up(160,128)=256, only the {192,64} branches carry the odd-tile TMEM
  shift) plus a hard OOB at prefill M=1024 (7168 % 160 ≠ 0). The original
  "−14.1 %" was the validator comparing the buggy kernel against itself
  (fused-vs-sequential self-comparison — blind to the SFB bug). N=256 already
  optimal; megakernel FC2_N default fixed 160→256 and the validator de-blinded
  to a true f32 einsum. The fusion itself is timing-neutral under PDL. Chain:
  `2833bb0c5` (ship) → `29f492b49` (revert opt-in) → `d01737397` (unreachable)
  → `e105fd7a1` (megakernel default + validator). **Lesson: cosine-gate
  kernels against a TRUE reference, never the candidate's own
  fused-vs-sequential output.**
- **`use_cute_dsl_topk` flip / DSL top-k at prod** — the round-1 "width ≥12288
  ⇒ DSL wins" premise was **measured false** (`841f9874a`): the C++
  `indexer_topk_decode` is ~width-independent (~11 µs flat, walks
  `[0, live_kv)`), so at prod (width 132096, live kv ~4.6k) C++ is ~1.7×
  faster with a bit-identical set. The width-override is removed; the DSL
  kernel runs only at long live kv (kv gate, ≥16k after fp16 logits). Closed
  both ways — no flip in either direction is left.
- **MoE tactic mispick at bs=16** — none. The MoE tuner picks optimally at
  bs=16 (verified head-to-head against the forced alternatives); no re-pin
  lever.
- **Attention-tactic retune at TP bs=16** — already optimal: the autotuner
  re-selects correct tactics at the TP shapes, and TP-rank attention is
  *faster* than ADP at equal load (part of the WarpDecode+TP case). No lever.
- **NVFP4-quantizing the dense MLA proj GEMMs** — FAILS accuracy: cos
  0.63–0.83 vs the bf16 path. The proj win is the cuBLASLt backend (B1,
  bit-identical), not more quantization. Do not revisit without a new quant
  scheme.
- **Indexer score→top-k fusion (I4)** — net-zero under CUDA graphs: the mask
  is already fused into the top-k kernel, the logits round-trip is <1 µs at
  decode B, and the launch overhead is hidden by graph replay. Bench shipped
  (`bench_indexer_score_topk_fused.py`, `3e03d665d`).
- **L2 persistence hints** — working set (MoE ~30 GB, indexer-K ≥575 MB)
  ≫ 126 MB L2; no inter-step reuse window. Arithmetic kills it.
- **NVLS one-shot allreduce** — ADP decode has no attention allreduce; MoE is
  all-to-all; NVLS already on. Returns only if M2 flips topology.
- **TP2×2-replica decode** — same per-rank MoE bytes at c8, loses dense-GEMM
  batching. Net-negative.
- **Sampler argmax trim** — already grouped + flashinfer + side-stream D2H; no
  wasted work at temp 0.
- **First-step-after-NIXL slow path** — gen-init joins CG-eligible steps; eager
  fallback ~1.6% of steps.
- **Detok off-loop / cgroup throttling** — already off-loop (stream_interval=50,
  8 workers); `nr_throttled=0`.
- **Indexer `.item()` syncs at decode** — in-graph metadata already shipped
  (`indexer.md` win #5); python indexer runs only at capture.
- **Together-MSA bf16 AB-swap index scoring** — implemented, benched, REJECTED
  (regression vs the tcgen05 MXF4 block-scaled path). Do not rebuild.
- **All speculation** (DFlash, MTP/NextN, EAGLE, n-gram, SMC-SD) — excluded by
  mandate.
- **FP4-experts + FP8-rest** (TileRT Leap-2) — already exceeded (NVFP4 experts +
  fp8 KV + kvarn_k2v2 latent + NVFP4 indexer-K).

---

## Cycle log

- **Cycle 1** (shipped `a1b13ea78`): C1 + C3. Within tight-harness noise at c16;
  removed real host work; landed for correctness/composability.
- **Cycle 2** (in progress): H1 (HISA band-aware) is the headline — probing the
  cost with the zero-code decode `enable_nvfp4_hisa=false` A/B against the 39.98
  tight baseline, then the band-aware code fix keeps HISA enabled and optimal.
- **Cycle 3** (shipped `2833bb0c5`): K2 FC2 N=160 default-on (driver cos+timing
  verified, −27% standalone FC2); P1 megakernel op shipped opt-in.
- **Cycle 4** (HISA-scale, shipped this cycle): fixed the CUDA-graph capture-gate
  root cause in `dsa.py` — `max_kv_len` was the static block-table width (132096),
  freezing `candidate_len` at 33024 and **selecting wrong tokens at kv>33k**
  (cos 0.707 vs eager at 66k). Now reads `metadata.max_gen_kv_len` (capture-frozen
  live ceiling). B200-verified on the real `indexer_topk_decode`: cos 1.00000 at
  all lengths, HISA-on never slower than off, **1.87× at 66k / 2.09× at 132k**;
  forced-on proves "always wins when Indexer on" (up to 2.56× at 33k). This is the
  HISA-scaling directive realized + a latent correctness fix. Default-on.
- **NVFP4 index-scoring (rejected, not a vacuum win)**: the FP4 path already exists
  (`FP4MQALogitsKernel`, dsa.py:3925, gated `use_fp4 and use_cute_dsl_paged_mqa_logits`).
  Head-to-head at REAP B≈4 decode: only 1.0–1.02× faster (memory-bound 1.6× win
  needs B≈16, which REAP decode never reaches) **and degrades top-k selection**
  (IoU 0.69–0.83 vs FP8's 0.92–0.95) on random inputs. Do NOT default-on without an
  e2e accuracy eval on real indexer activations. Bench:
  `tests/scripts/cute_dsl_kernels/paged_mqa_logits/bench_fp4_vs_fp8_decode.py`.
- **Cycle 4b** (shipped `81cfeb88c`): fused single-pass cluster top-k
  (1.28–1.59× at width 132096, IoU 1.0) + HISA-gate eager d2h-sync unification
  (`max_gen_kv_len` in both branches, eager indexer host wall 33k 21.3→1.18 µs).
  The cluster-top-k lane was later **superseded at the prod operating point**
  by I7/`841f9874a` (C++ wins at short live kv); it still serves long live kv.
- **Cycle 5** (K2 post-mortem, shipped `29f492b49`→`d01737397`→`e105fd7a1`):
  the FC2 N-tile lever is **dead** — prefill M=1024 crash → opt-in revert; then
  TRUE-f32 re-validation exposed N=160 as numerically broken everywhere
  (SFB miscompute, cos ~0.790; the original gate was a fused-vs-sequential
  self-comparison, blind by construction); N=160 made unreachable; megakernel
  JIT default FC2_N 160→256 + `validate_fused_moe_megakernel.py` de-blinded to
  a true f32 einsum reference. See killed list for the full lesson.
- **Cycle 6** (shipped `841f9874a`): prod decode top-k routed to the vanilla
  C++ kernel — the width-override premise was false; C++ is ~width-independent
  (~11 µs) and **~1.7× faster at prod live kv (~4.6k)**, bit-identical set;
  ~29–33 % faster top-k pipeline. I7.
- **Cycle 7** (shipped `4220bf4bd` + `3e03d665d` lane 1): cuBLASLt forced for
  the NVFP4 proj GEMMs — MLA proj **~1.62 ms/tok (~6 % TPOT)**, bit-identical,
  1.33–2.07× flat M=1→1024; then extended everywhere (+**1.62 ms/tok
  incremental**, TP4) and fixed the `kv_a_proj_with_mqa` re-creation bug that
  had left `4220bf4bd` partially dead. B1.
- **Cycle 8** (shipped `3e03d665d` lanes 2–3 + `33e801fd1`): fp16 indexer
  logits (I6, recall at the fp8 noise floor, top-k −15…−22 % at kv ≥ 33k);
  gated-norm/glue fusions (G1: `fused_lowrank_gate` −91.6 % on a ~7 ms/step
  fp32 SGEMM+cast soup found hiding in the dense-proj bucket; fused
  sigmoid·mul; HISA per-step invariant memo; −6.27/−9.38 ms/step eager GPU at
  bs4/bs16); then the CuTe DSL lowrank-gate port (−36 % vs Triton,
  bit-identical, default impl).
- **Cycle 9** (shipped `51918fba2`): DeepEP low-latency enablement under
  WarpDecode (overlay LL-layout skip-reason + ConfigurableMoE oversize
  park/restore; caveats closed at EP2). Production flip is M3 — LL 45 vs
  115 µs/layer at token_limit=16 but **inverts at 64**, so the limit must be
  per-rank concurrency, not max_batch_size.
- **Cycle 10** (shipped `fd705a6f5`): `_FP4OUT_MIN_M=128` guard lifted — the
  swiglu+fp4-out fusion now engages at decode M (exact vs TRUE-f32, OOB demo
  clean, ~100 µs/step + 58 launches). K3.
- **SM1/SM3 schedule hoists — exposure check DONE**: the hoisted schedule work
  is **already OVERLAPPED at c16** — correct but flat, no exposed-time win.
  Status revised from "queued exposure check" to closed-no-win at c16 (revisit
  only if the overlap structure changes).
- **Queued**: M3 flip decision (LL token-limit sizing); C2 5-scenario
  equivalence verification; G2 absorb-or-fold design; I5 `build()`-gating
  investigation; then N1, I2, K1 PDL. Held: S2 (moot under WarpDecode+TP).

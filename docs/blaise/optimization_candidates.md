# tok/s/user optimization candidates (post-first-token, c16)

Living plan for hill-climbing **tokens/sec/user after first token at 16
concurrent users** on the r20 disaggregated deploy
(`BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft`,
8×B200: prefill TP2×CP2 LayerSplit, decode TP4+EP4+attention-DP). This is the
companion to the per-component docs ([indexer](indexer.md),
[warpdecode](warpdecode.md), [kvarn](kvarn.md), [sparse_mla](sparse_mla.md),
[topology_deploy](topology_deploy.md), [moondream_pipelining](moondream_pipelining.md),
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
| WarpDecode (decode) | on, forced `decode_1cta`, fixed tactic | persistent-megakernel is the structural ceiling |
| dense KVarN `kvarn_k2v2` | on, amortized restore | host-gated pre-replay scan (C1) |
| Indexer IndexCache + FSSS | on, `index_topk_freq=4` | escalation to 8 under recall gate |
| **HISA** | **on whenever the Indexer is on (kv > `index_topk`=1024); `enable_nvfp4_hisa=true`, gate `hisa_min_seq_len=1024`** | **candidate width scales with kv (H1 landed); deep-optimization track H is the top hill-climb priority** |
| NVFP4 indexer-K (MX E2M1+UE8M0) | on | score→top-k fusion candidate |
| NIXL transport + request pinning + Moondream overlap | on | generation-first/write-mode is the open gate |

HISA stays **enabled whenever the Indexer is on** (kv > `index_topk` = 1024).
HISA is a core part of the model's serving path and, implemented correctly,
*wins* at every indexed length with cost that scales with sequence length. The
H1 fix below makes our HISA scale (its candidate width was frozen at the
max-context value); it does **not** gate HISA off anywhere the Indexer runs.
Deep HISA optimization (track H) is the top hill-climb priority.

---

## Priority sequence

1. **Finish LayerSplit (Part 1 of the goal):** C9 CP=2 IPC broadcast GPU
   re-validation → push; then L1 (z.ai dense-broadcast overlap re-measure) and
   L2 (read-set block-id hoist). LayerSplit must be correct + optimal before the
   decode hill-climb is the focus.
2. **THEN — deep HISA optimization is the TOP hill-climb priority (track H).**
   HISA, implemented correctly, wins tok/s/user at *every* indexed length and
   its cost *scales* with sequence length (validated empirically + the 4:1
   compression chart); the Indexer (and thus HISA) is on whenever kv >
   `index_topk` (1024). Any length where our HISA loses is an implementation
   defect to fix, not a reason to gate HISA off. H1 (below) is the first fix;
   H2+ (the continuous-scaling, pipeline-fusion, and knob-tuning tiers) are
   under active deep investigation (probe P2-HISA) and become the lead Part-2
   work once LayerSplit lands.
3. Then the host/MoE/kernel levers (S2, N1, I2, M1, K1/K2), then the structural
   megakernel (P1).

## Ranked candidates

| # | Candidate | Layer | Expected win @ c16 | Status |
|---|-----------|-------|--------------------|--------|
| **H (track)** | **Deep HISA optimization — TOP hill-climb priority (after LayerSplit)** | indexer | **HISA must win at every length, scaling with kv** | **H1 in A/B; H2+ deep-probing** |
| **H1** | **HISA candidate-width band-scaling** (use `metadata.max_gen_kv_len`; gate→`index_topk`) | indexer | short-band candidate 33024→2048 (~2–4% TPOT) | **fix landed, A/B in flight** |
| H2 | HISA per-row continuous candidate scaling (Tier-2: live-kv-aware candidate GEMM + topk) | indexer | candidate cost → continuous-with-kv (chart) | probing (P2-HISA) |
| H3 | HISA 8-kernel-pipeline fusion + PDL (mask→score, remap→topk, block-score→block-topk) | indexer | 128 launches/step → fewer | probing (P2-HISA) |
| H4 | HISA knob tuning (`compression_ratio`/`block_topk`/`block_size`), recall-gated | indexer | faster compression that holds recall | probing (P2-HISA) |
| C1 | KVarN pre-replay restore host-gate | scheduler | ~0.75–2 ms host (within harness noise) | **shipped** `a1b13ea78` |
| C3 | Cache debug env-gates / no eager kwargs | scheduler | ~0.1 ms/step | **shipped** `0adc87009` |
| C9 | CP=2 IPC push broadcast | prefill TTFT | 1.2–3× the per-layer broadcast | impl, GPU re-validating |
| S2 | Collapse 10 host MPI collectives → ~3 | scheduler | 150–400 µs + 7 barriers of jitter | planned |
| N1 | NUMA-pin decode workers to node 1 | system | 0.3–1 ms + jitter | planned (rides manifest) |
| I2 | `index_topk_freq` 4→8 | indexer | indexer-cost −68% on S-steps (~) | planned (recall gate) |
| M1 | MoE A2A two-sided → one-sided + workspace combine | comm | 0.3–0.9 ms/step | planned |
| K1 | PDL coverage completion | kernel | +1–3% | planned |
| K2 | FC2 N-tile 256→160 | kernel | ~+1.5–2% e2e (−14.1% on the MoE pair) | planned |
| L1 | z.ai dense-broadcast overlap | prefill | TTFT (exposed indexer-K broadcast) | planned |
| P1 | Persistent decode-layer megakernel | kernel | +5–10% near-term, multi-× ceiling | strategic |
| MO1 | MORI-style generation-first / write-mode handoff | transport | TTFT (overlaps RDMA with prefill) | needs router build |

---

## H1 — HISA candidate-width band-scaling (HEADLINE; fix landed, A/B in flight)

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

**Track H continues (deep HISA optimization, top hill-climb priority):** H2
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

## Scheduler / host overhead

### S2 — Collapse host MPI collectives (planned)
10 python MPI object collectives/iter, 8 **before** forward launch (rank-state
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

## Indexer (beyond H1)

### I2 — `index_topk_freq` 4→8 (planned, recall-gated)
FSSS is **cross-layer** (not cross-step): the doc's −39/60/68% @ stride 2/4/8
(`indexer.md` win #4) is *indexer-cost-relative* and traces to the XSTEP
prototype (`llm_args.py:329-341` docstring flags accuracy-must-be-validated).
Escalating to 8 reuses the F-layer top-k across more S-layers. Gate:
top-1024 SET recall vs `freq=1` ground truth on real-shape synthetic. Open
question: recency drift between F-layers at stride 8 — read
`indexerXstepRecencyPatch.cu`'s window and whether it must widen.

### I3 — `seq_len_threshold` short band (folds into H1)
Currently unset → one effective band → width always 132096. Setting 65536 (for
H1) also revives the width-correct C++ insertion top-k (`indexer.md` win #1/#2/#6)
on short-band F-layers: ~3–4 µs/F-layer × 16 ≈ 50–60 µs/step for kv≤8k traffic,
at 2× graph count. Doc's "2.08–2.30×" is top-k-kernel-relative, not TPOT.

### I4 — Score→top-k fusion (research)
Stream `cand_score` tiles into the radix histogram without materializing the
`[B, width]` fp32 logits in HBM. At b=4 the round-trip is small (B4×132096×4B×2
×16F ≈ 67 MB/step ≈ 10 µs) so the win is **launch-count** (~5 µs×16 ≈ 0.1 ms)
not bandwidth; weigh against tcgen05-epilogue complexity. Read the DSL radix
kernel epilogue first.

---

## MoE / communication

### M1 — A2A two-sided → one-sided + workspace combine (planned)
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

### K2 — FC2 N-tile 256→160 (planned)
`warpdecode_mega_driver` FINDINGS: the fused FC1→FC2 kernel at N=256 is 42.94 µs
(≈ prod 44.18); retuning FC2 N→160 gives 37.94 µs, **−14.1%** on the MoE pair,
cos 0.99961. ≈ +1.5–2% e2e. Note: single-launch fusion **ties** the PDL-chained
pair — the win is the N-tile, not the dispatch fusion (PDL already overlaps the
boundary). Days (kernel retune + parity gate).

### P1 — Persistent decode-layer megakernel (strategic, multi-day)
The TileRT "Breaking 1000 TPS" Leap-1: a persistent GPU program eliminates
per-kernel grid ramps + cross-op SMEM/TMEM round-trips (NOT dispatch fusion —
PDL ties that, confirmed by the mega-driver). ~300–500 kernels/step × 1–3 µs
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

- **`use_cute_dsl_topk` flip** — width-aware dispatch (`dsa.py:292-352`) already
  optimal; c16 width ≥12288 ⇒ DSL wins ~4 µs/layer. Round-1 memory note stale.
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
- **Queued**: C9 GPU re-validation → push; then S2, N1, I2, M1, K1/K2.

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
| **HISA** | **on, `enable_nvfp4_hisa=true`** | **band-aware capture gate (H1) — HISA for ≥`hisa_min_seq_len`, plain exact top-k below** |
| NVFP4 indexer-K (MX E2M1+UE8M0) | on | score→top-k fusion candidate |
| NIXL transport + request pinning + Moondream overlap | on | generation-first/write-mode is the open gate |

HISA in particular stays **enabled at the architecture level**; the H1 fix
below makes it fire only where it wins (long context) and fall to the exact
plain path where that is both faster and more accurate (short context) — i.e.
*maximally optimized HISA*, not HISA-off.

---

## Ranked candidates

| # | Candidate | Layer | Expected win @ c16 | Status |
|---|-----------|-------|--------------------|--------|
| **H1** | **HISA capture-gate band-awareness** | indexer | **~0.4–1.1 ms/step (~2–4% TPOT)** | **probing** |
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

## H1 — HISA capture-gate band-awareness (HEADLINE)

**Problem (triple-confirmed: P2-I2, P2-I3 independently, + direct read).** The
HISA pre-indexer enable gate keys on the wrong length under CUDA-graph capture.
`dsa.py:2547-2551`:

```python
capturing = torch.cuda.is_current_stream_capturing()
if capturing:
    max_kv_len = block_table.shape[1] * k_cache.shape[1]   # STATIC = 132096
else:
    max_kv_len = int(kv_lens.max().item())                 # live kv (~4.6k prod)
if not self._should_use_hisa_pre_indexer(max_kv_len):      # >= hisa_min_seq_len (65536)
    return None
```

At capture the gate sees the **static block-table width** (132096 ≥ 65536), so
`_should_use_hisa` (`dsa.py:2168-2180`) returns True and HISA is baked into
**every** decode graph. Eager warmup uses the live kv (~4.6k < 65536 → plain
path), which is why HISA was believed "off at prod" — those were eager-mode
measurements. The candidate-score GEMM (`fp8_fp4_paged_mqa_logits`,
`dsa.py:2618`) and candidate top-k (`indexer_topk_decode`, `dsa.py:2684`) then
run at a **fixed** `candidate_len` (`candidate_context_lens = full(..)`
`dsa.py:2612`; `selected_lengths = full(candidate_len)` `dsa.py:2682`) —
regardless of live kv, and `kAdaptiveFinalSort` (which rescues the *plain*
path's static-width top-k) does **not** rescue HISA because the selected length
is forced to the full candidate width. So every graphed decode step at prod kv
(~4.6k, where the plain exact path is cheapest) pays the full HISA pipeline:
~8 kernels incl. a 33024-wide candidate GEMM + 33024-wide C++ top-k vs the
plain path's 2 kernels.

**Cost.** Per F-layer HISA excess ≈ 45–66 µs (block quant/score/top-k ~19 µs +
candidate score @33024 ~25–40 µs + candidate top-k @33024 ~16.4 µs + glue −
plain MQA+top-k ~21–24 µs) × 16 F-layers ≈ **0.75–1.09 ms/step** (P2-I3,
higher accounting) / **0.4–0.7 ms/step** (P2-I2, lower bound) ≈ **2–4% of
TPOT, ~5–7% of the attackable 15 ms.** Single largest clean indexer lever.

**Fix (HISA stays ENABLED — make it band-correct).** At capture, gate on the
per-graph warmup kv instead of the static table width. `metadata.max_gen_kv_len`
is exactly this value (`dsa.py:321-350` docstring; set `dsa.py:1539`) and is
already used by `_indexer_logits_width` for the identical band-aware purpose.
Two parts:

1. **Code (~5 lines).** Thread the per-graph `max_gen_kv_len` into
   `_hisa_topk_from_nvfp4_cache` (it is an `Indexer` method that does not
   currently receive `metadata`; the caller `dsa.py:3558` has it in scope) and
   use it for `max_kv_len` when `capturing`. Then the short-band graph
   (warmup kv < 65536) captures the **plain exact** path and the long-band
   graph (warmup kv ≥ 65536) captures genuine HISA.
2. **Config.** Set `seq_len_threshold: 65536` (currently unset → default split
   at 8192) so the graph-band boundary **aligns with** `hisa_min_seq_len`;
   otherwise the long band [8192, 132096] would still over-fire HISA on its
   8k–65k portion.

**Why this is correctness-safe (and an improvement at prod).** For kv < 33024
the HISA candidate set (33024 positions) already covers the entire context, so
HISA selection is *exact* there — switching to the plain path returns the same
top-k SET, just cheaper. For kv ≥ 65536 HISA is unchanged. The only band where
behavior changes is where HISA was pure wasted compute.

**Correctness gate.** Two-stage: (a) top-k SET Jaccard = 1.0 between HISA-on
and band-aware (HISA-off-for-short) on synthetic kv ∈ {2k, 8k, 32k} (must
match: candidate set complete); (b) on a ≥65k synthetic, confirm the long band
still selects the HISA set (HISA genuinely engaged).

**Measurement.** A zero-code probe (`enable_nvfp4_hisa=false` on the decode
worker only) quantifies the prod-kv cost end-to-end against the 39.98 tight
baseline before the code fix lands. The production version keeps
`enable_nvfp4_hisa=true` + the band-aware gate.

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

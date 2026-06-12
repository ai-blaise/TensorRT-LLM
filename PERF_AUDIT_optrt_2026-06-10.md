# op-trt Performance Audit — Bench Reproduction + Code Findings

**Date:** 2026-06-10 · **Branch:** `op-trt` @ `81cfeb88c` (merge-base with upstream `main` = `5f106dfab`)
**Hardware:** node 001, GPU 0 (B200, 183 GB), shared-idle with the resident R20 prefill worker (137 GB allocated, 0% util during runs). Clocks NOT locked.
**Software:** image `optrt-fusionproof-20260609134115` (`.so` ≈ commit `0be07d6df`-era C++, includes BDR/mlaKernels + all indexer ops) + Python overlay of the 28 `.py` files changed `0be07d6df..HEAD` — i.e. the same overlay scheme as `deploy/disagg_pd_r20/Dockerfile.r20-overlay`. TRT-LLM 1.3.0rc17, torch 2.11.0a0+nv26.02, CUDA 13.1.
**Method:** every suite under Spencer's docs (`kvarn_results/`, `kvarn_inkernel/`, `blaise_perf/*`, root `bench_*`/`test_kvarn_*`) was executed in a fresh container (`optrt-bench-claude`); in parallel, six audit sweeps covered KVarN, DSA/indexer/HISA, LayerSplit, SMC, MoE/WarpDecode, host/executor, and bench methodology. Findings are tagged:

- **[V]** — verified by me directly (read the code at HEAD and/or reproduced by running)
- **[A]** — produced by an audit sweep, code-cited, but not independently re-derived line-by-line
- **[C]** — an audit claim I checked and **corrected**; the corrected statement is what's written here

Severity is about *decode/prefill production impact*, not code style.

---

## 1. TL;DR

1. **The benches largely reproduce — with one big exception.** The KVarN decode-overhead bench matches the committed log within ~2% on every row; the 128K per-sub-block accuracy result, the 0.65×-of-fp8 production read kernel, the 8.3× amortizer, and the NVFP4 backend-selection conclusion (cutedsl 58/70) all reproduce; correctness suites pass. But the **lever-2 affine-reuse win is gone at HEAD**: the fused affine kernel measures **3.92 µs vs the documented 2.27 µs**, and the wired S-layer dispatch shows **no win at all** (4.07 µs vs F-path 4.11 µs; doc claims 2.45 vs 4.11 = 1.68×). The documented −0.071 ms/tok S-layer saving measures **−0.007 ms/tok** today. The R4 aggregate "−43.5% indexer TPOT" partially rests on this → needs re-validation.
2. **One ungated D2H sync per prefill chunk per layer feeds permanently-dead code** (`dsa.py:3048`) — the single cheapest high-value fix in the branch.
3. **FC2 N-tile=160 is default-on and *pins* the tactic list** (removes 128/256 entirely) for every user of the grouped-GEMM finalize op, not just the REAP shape it was validated on.
4. **The SMC draft (M=25) permanently lives on a Triton fallback GEMM + Triton quant** because every other odd-M path is hardcoded off after B200 faults. This — not "padding waste" — is the most likely chunk of the documented ~2 tok/s SMC regression.
5. **LayerSplit's indexer read-set broadcast exists only in `Indexer.forward`, which the live custom-op path bypasses** — and when it does run it has no `skip_topk` gate, so all 43 S-layers re-broadcast the full prefix every step.
6. **The KVarN software-cache store path wastes ~2× HBM passes in its Sinkhorn loop and runs 16 iterations where its own docstring says ~4 suffice**; the restore path (their own bench, reproduced) costs 52–234% of the per-layer decode budget — the in-kernel path is the only viable production read path.
7. **Methodology debt is real**: the NVFP4 GEMM microbench sorts samples and drops the slowest 20% before taking a median (≈P40, optimistically biased); `topk_scheme_probe.py` cannot run as committed (missing module); `test_kvarn_backend.py` is broken by an env-var rename and has demonstrably not been run since; several suites need undocumented env/paths; the FMHA overlap probe's printed conclusion contradicts its own numbers (489 µs dequant vs 132 µs hideable).

---

## 2. Bench reproduction matrix

Environment caveats applying to ALL rows: idle-but-resident serving worker on the same GPU, no clock locking, single seed, single GPU.

| # | Suite (doc) | Status | Headline result (this run) | vs Spencer's committed numbers |
|---|---|---|---|---|
| 01 | `test_kvarn_backend.py` | ❌→✅ **broken as committed** (F-46) | after env-key fix: ALL PASS; capacity @131072 ctx: KVarN 2.219 GiB vs fp8 4.289 vs fp16 8.578 (1.93×/3.87×) | test sets `TRTLLM_KV_CACHE_QUANT` but the resolver reads `TRTLLM_MLA_LATENT_KV_DTYPE` — stale env-var name, can never pass unmodified |
| 02 | `test_kvarn_cycle.py` (`system_cycle.log`) | ✅ PASS | sink/tail bit-exact; restored cos ≥ 0.99328 | matches committed (≥0.993) |
| 03 | `test_kvarn_mla.py` | ✅ PASS | — | consistent |
| 04 | `test_kvarn_mla_preset.py` (`mla_preset_ablation.log`) | ✅ PASS | k4v2 > k2v4 ordering holds | consistent |
| 05 | `test_kvarn_amortize.py` | ✅ after path fix | KVARN-AMORTIZE-OK; recycle re-restore cos 0.99361 | needs `/tmp/kvarn_bench` module layout AND cwd=repo root (relative `open()` of dsa.py) |
| 06 | `bench_kvarn_component.py` | ✅ | bytes/elem + cos table reproduced | wall-clock timing methodology — see F-43 |
| 07 | `bench_kvarn_decode.py` (`system_decode_overhead.log`) | ✅ | batched dequant 316.2 µs @256 blk (1.24 µs/blk); sparse restore 51.6→233.8 %budget @B=1→32 | **matches committed log within ~2% on every row** |
| 08 | `bench_kvarn_accum.py` | ✅ | — | consistent |
| 09 | `bench_kvarn_outlier.py` | ✅ | E_M/E_D split reproduced | note: this bench measures **no timing** (F-44) |
| 10 | `bench_kvarn_intel_correctness.py` | ✅ after path fix | fp8 0.9987 / nvfp4 0.9965 / kvarn 0.9949 cos ordering reproduced | consistent |
| 11 | `bench_kvarn_amort_e2e.py` | ✅ after path fix | B=32/1024 blk: OFF 1807.9 µs (530% budget) → ON 218.2 µs (64%) = **8.3×** | consistent with claimed 3.1–7.6× family (this point above it); 64-step window caveat (F-44) |
| 12 | `bench_sparse_mla.py` | ✅ | `sparse_mla_decode_nvfp4` B=32 topk=2048: **296.0 µs**, baseline saved | no committed reference number; recorded as new baseline |
| 13 | `idx_l2_affine_kernel.py` | ⚠️ **DISCREPANCY** | full remap 4.094 µs ✓, **fused affine 3.924 µs** (doc: 2.268), where 6.150 ✓ | **claimed 1.78× win is now 1.04×**; kernel correct=True |
| 14 | `bench_lever2_wiring.py` | ⚠️ **DISCREPANCY** | F full remap 4.110 µs ✓, **S affine reuse 4.074 µs** (doc: 2.450) | **claimed 1.68× wiring win is now 1.01× (none)**; correctness checks pass |
| 15 | `topk_scheme_probe.py` | ❌ broken as committed | `ModuleNotFoundError: logits_launch_probe` | Win B's width-crossover (12288) **not reproducible from the repo** |
| 16 | `check_kvarn_gqa_static_abi.py` | ✅ (`--repo /repo`) | static ABI consistent | — |
| 17 | `bench_kvarn_gqa_sparse.py` | ✅ after `--repo /repo` | decode M=25 (SMC draft shape): sparse-topk 96.3 µs vs dense 114.9 µs | default `--repo /workspace` wrong for this layout |
| 18 | `nvfp4_gemm_backend_microbench.py` | ✅ (623 s incl. JIT) | winner histogram **{cutedsl: 58, cublaslt: 11, cuda_core: 1}**; o_proj cutedsl +11.5/6.3/6.8% @M=1/2/4 | claimed {54, 14, 2, cutlass 0} → **conclusion reproduces** (cutedsl majority, cutlass never wins); o_proj wins in-family but M=2/4 land below the documented 8.5–12.3% band |
| 19 | `kvarn_inkernel/*.cu` (4 benches) | ✅ all build+run (`-arch=sm_100a`, CUDA 13.1) | see §2.1 | reproduce committed logs incl. the 128K result |
| 20 | `fmha_overlap_probe.py` | ✅ | FMHA-equiv 132.1 µs/step vs fused dequant 489 µs @N=1024 | probe's own "would overlap" conclusion is overstated — see §2.1 |

### 2.1 Long-running / in-kernel results

**NVFP4 GEMM microbench (#18):** full 70-cell sweep completed under CUDA-graph replay. Winner histogram `{cutedsl: 58, cublaslt: 11, cuda_core: 1}` vs documented `{54, 14, 2}` — the enable-CuteDSL decision is **supported**; per-cell win% wobbles a few points between runs (expected given F-40's P40-biased metric), so treat any individual cell's win% as ±3-4 points.

**KVarN in-kernel CUDA benches (#19):** all four compile and run on CUDA 13.1 / sm_100a.
- `bdr_persub_inkernel_validate` (PRODUCTION warp-coop layout, Ntok=2048): cos_ckv 0.995143; **BDR read 4.11 µs vs fp8 read 6.36 µs (0.65×)**, BDR write 4.65 µs → the commit `76204896f` claim "dequant-on-read 0.46–0.67× fp8" reproduces **for the production per-sub-block kernel**.
- `bdr_inkernel_bench` (earlier-generation standalone kernel): BDR read 15.7–17.9 µs vs fp8 10.5–12.3 µs = **1.33–1.54× SLOWER** at every N. Not a contradiction once you know the history — the committed logs capture three kernel generations (5.75× slower → ~1.5× → 0.65×) — but the directory does not label which log corresponds to which kernel, and a reader grabbing `bdr_inkernel_bench` numbers would conclude the read path loses to fp8. **Label the logs or delete the obsolete benches.**
- `bdr_longctx_bench` (32 s): **reproduces the headline 128K claim** — per-sub-block scales hold ~0.992 cos at 128K while naive-INT4 and per-token-scale BDR both collapse to ~0.866; accumulator dtype is confirmed not load-bearing.
- `kvarn_inkernel_bench`: fill-step worst case at B=32 = 490.7 µs (143.9% of layer budget) vs steady-state amortized 1.18 µs (0.35%), fullFUSED 75.8 µs (22.2%) — consistent with the amortizer being mandatory (F-13).

**FMHA overlap probe (#20):** measures FMHA-equivalent work at 132.1 µs/step (T=65536, Hq=128) and prints that the 489 µs fused dequant "would OVERLAP this attention compute." **The arithmetic doesn't support the wording**: attention can hide at most 132 of 489 µs (~27%); the remaining ~357 µs is exposed. The probe's conclusion line should be corrected before anyone budgets on it.

**Re-runs after environment fixes:** `test_kvarn_backend` ALL PASS once the env key is fixed (F-46); `test_kvarn_amortize` + `bench_kvarn_amort_e2e` PASS from repo-root cwd; `bench_kvarn_intel_correctness` PASS (k4v4 cos_ckv 0.99390 / cos_kpe 0.99582); `bench_kvarn_gqa_sparse` PASS with `--repo /repo`.

### 2.2 The lever-2 regression (most important reproduction result)

`LEVER2_RESULT.md` claims (B200, megamoe_dev, graphed, M=8 topk=1024 block=64):

```
full remap 4.044 µs   fused affine 2.268 µs (1.78×)   wired: F 4.106 → S 2.450 µs (1.68×)
savings: 0.0712 ms/token across 43 S-layers
```

This run (same GPU class, same shapes, same scripts, graphed, correctness checks all `True`):

```
full remap 4.094 µs   fused affine 3.924 µs (1.04×)   wired: F 4.110 → S 4.074 µs (1.01×)
savings: 0.0073 ms/token across 43 S-layers   (10× less than documented)
```

The full-remap baseline and the `torch.where` reference (6.150 vs 6.149 µs) reproduce to 3 decimal places, so clocks/environment are NOT the explanation — the affine path specifically lost its advantage. Plausible causes, in order: (a) both kernels are now sitting at a ~3.9–4.1 µs launch/latency floor at these tiny sizes, and the original 2.27 µs was measured on a build where the affine kernel avoided some fixed overhead that has since returned (e.g. AOT op dispatch layering vs raw `load_inline` kernel); (b) the AOT `indexer_affine_reuse` op in the current `.so` is slower than the JIT kernel that was originally measured; (c) a kernel change between measurement and HEAD. **Action:** profile one S-layer call with nsys (kernel name + duration will immediately distinguish dispatch overhead vs kernel regression). Until then, treat `-0.071 ms/tok` (and the R4 aggregate that embeds it) as stale.

**UPDATE (same day, post-optimization validation):** a re-run of `bench_lever2_wiring.py` immediately after a GPU-warming workload measured **F 3.811 / S 2.755 µs (1.38×)** — the win partially reappears under warm clocks. Across three same-day runs the S-path ranged 2.76–4.07 µs against a 3.8–4.1 µs F-path.

**RESOLVED (nsys, same day):** profiling the wiring bench gives the device-time truth: `convertReqIndexToGlobalKernel` (full remap) **1.60 µs median**, `indexerAffineReuseKernel` **1.30 µs median**. Both kernels are healthy — and both are far below every wall measurement, which is launch/graph-replay floor. Conclusions: (a) there was never a kernel regression; (b) the documented 1.78×/0.071 ms-per-token saving compared launch-floor-dominated wall times — the **true device-level saving is ~0.3 µs/layer ≈ 0.013 ms/token across 43 S-layers (~5× smaller than documented)**, and that device delta is what a captured prod graph actually realizes; (c) the run-to-run wall oscillation (1.0×–1.7×) is clock/launch noise, exactly the F-44 methodology gap. The R4 aggregate should be restated with the 0.013 ms/tok figure.

---

## 3. Findings — every perf issue, by subsystem

Format: `F-n [tag][severity] location — title`, then what/why/fix.

### A. DSA / Indexer / HISA

**F-1 [V][HIGH] `dsa.py:3048` + call sites `:3729`, `:3789` — ungated per-chunk-per-layer D2H sync feeding permanently-dead code.**
`_hisa_topk_from_logits()` computes `int((row_ends - row_starts).max().item())` on the eager path *before* checking `_should_use_hisa_logits()`, which is hardcoded `return False` (`dsa.py:2422`). Both prefill call sites invoke it unconditionally after `_call_mqa_logits`, so every prefill chunk × every layer pays one device→host sync (≈5–20 µs stall + pipeline break) to evaluate a gate that can never pass. The decode call site (`:4044`) was correctly pre-gated by the R4 "Win A" fix; the prefill sites were missed. At 61 layers × N chunks this is hundreds of pointless syncs per prefill step. **Fix (1 line):** check the static gate before computing `max_kv_len`, or early-return `None` at function top when the feature is off. Also removes the two `count_nonzero(...).item()` syncs at `:3060-3062` on the same dead path.

**F-2 [V][MED] `dsa.py:3329-3331` — O(B²) chunk-spec construction in MLA-chunked-prefill path.**
`chunk_specs = [(i, 0, host_seq_lens[i].item(), host_seq_lens[:i].sum().item() ...) for i in range(num_contexts)]` re-sums a growing prefix slice per request: quadratic tensor-slice sums plus 2 B `.item()` calls. `host_seq_lens` is a host tensor, so no device sync, but at B=64 contexts this is ~100–200 µs of host time per prefill step. **Fix:** one `torch.cumsum` + one `.tolist()`.

**F-3 [V][MED] `dsa.py:4099-4124` — silent full-width fallback top-k.**
The `else` ("padded") decode branch materializes `positions` `[B·N, 132096]`, a full-width boolean mask, `masked_fill`, and a dense `topk` over width 132096 — an O(B·N·L) memory-bound path that silently engages when the DSL/C++ top-k paths reject a shape. Nothing logs when this happens. **Fix:** `logger.warning_once` on entry + a debug assert in prod configs so a kernel-gating regression can't silently 10× the indexer cost.

**F-4 [V][INFO/design] logits width pinned to `max_seq_len` (132096) under CUDA graphs.**
The indexer scans full cache width regardless of live kv (~4.6 K in prod), because a captured graph must be replayable as kv grows (R4 explicitly probed tight-width and correctly rejected it as graph-incompatible). Residual structural cost ≈ 6.2 µs logits + 12.3 µs DSL top-k per F-layer. **Possible recovery:** bucket decode graphs by kv-length band (e.g. ≤8K / ≤32K / ≤132K) the same way they're bucketed by batch — replays inside a band scan only the band's width. Medium effort, touches graph cache keying.

**F-5 [A][MED] `dsa.py:3767-3772` — `allgather` inside the prefill chunk loop.**
When `q_split_eligible`, the TP allgather of top-k buffers runs once per chunk instead of once per step (N_chunks × collective launch + sync). **Fix:** accumulate per-chunk results and gather once after the loop.

**F-6 [A][MED] `dsa.py:2929` — HISA schedule fallback rebuild is silent.**
When the hoisted schedule's signature check misses, the per-step schedule rebuild silently runs in-line (the cost SM3 was supposed to eliminate). **Fix:** count misses and log once; in prod a nonzero count means the hoist key is wrong.

**F-7 [A][LOW] `dsa.py:1509`, `:2007` — transpose→`contiguous()` hidden copies; `:3576-3587` — q/k_scale reshaped per call (hoistable to `pre_indexer_proj`); `:2388-2406` — per-instance dict-keyed caches for tiny arange/full tensors (global per-device pool would be cheaper).**

**F-8 [V][LOW] residual eager `.item()` fallbacks when metadata absent (`dsa.py:2846`-style, also the `_hisa_topk_from_nvfp4_cache` eager gate).**
These are the *documented* fallbacks from the 81cfeb88c sync-free work and only fire in warmup / non-bucketed batches. Fine as-is; flagged so nobody re-routes a hot path through them.

### B. KVarN

**F-9 [V][HIGH] `kvarn_core.py:85-97` — Sinkhorn loop wastes ~2× full-tensor HBM passes.**
At init, `cur = m / log_s_col.exp() / log_s_row.exp()` divides the whole `[N,R,C]` tile by `exp(0)=1` twice (two wasted elementwise kernels + temporaries). Inside each of the 16 iterations, `cur` is recomputed *from scratch* twice (lines 93 and 97), including re-dividing by the factor that didn't change that half-step — ≈4 full-tensor divisions + 2 broadcast `exp`s per iteration where ~2 would do. This is the store path (amortized ≈148.6 µs/step measured in this run at k4v4/group=64). **Fix:** keep `m_col = m / log_s_col.exp()` as the half-updated intermediate, and/or fuse the normalize into one Triton kernel.

**F-10 [V][HIGH] `kvarn_backend.py:89` — production Sinkhorn `iters: int = 16` vs the algorithm's own "~4 in practice".**
`variance_normalize_batched`'s docstring (`kvarn_core.py:74`) says "paper default 16; **~4 in practice**", yet `KVarNConfig.iters` defaults to 16 and nothing in the dtype strings used by the R20 deploy overrides it. Store cost is ~linear in iterations → ~3–4× write-path reduction available if quality holds. **Fix:** re-run `test_kvarn_mla_preset` + the long-context ablation at `iters=4`; if cos holds (≥0.992 @128K), flip the default.

**F-11 [V][MED] `kvarn_core.py:111-131` — `_pack_lowbit`/`_unpack_lowbit` loop in Python over pack factor.**
2-bit packing runs 4 shifted full-tensor ops (+ `clone`); unpack builds a Python list of 4 shifted copies then `torch.cat` — several extra HBM round-trips per store/restore on what should be one pass. **Fix:** vectorized bit-twiddling via a wider integer view, or fold into the (already-planned) fused quant kernel.

**F-12 [A][MED] `kvarn_backend.py:342-346` — software-cache restore uses dense `[512,512]` matmul for the inverse Hadamard.**
O(N·G·D²) ≈ 536 MFLOP per 32-block restore vs O(D log D) for an FWHT butterfly; also materializes the full rotated tensor. The production read is the in-kernel `dequantCopyKVarN` (which inverts per-sub-block), so this only burns when the Python software cache is the active path — which is exactly the mode `bench_kvarn_decode.py` shows blowing the budget (F-13). **Fix:** Triton FWHT for the Python path, or accept the Python path as test-only and assert it off in prod.

**F-13 [V][HIGH-design, measured] software-cache restore cannot fit the decode budget — confirmed by reproducing Spencer's own bench.**
`bench_kvarn_decode.py` (this run ≈ committed log within 2%): even the DSA-sparse restore policy costs **51.6% of the 341 µs/layer/token budget at B=1, 92.7% at B=8, 233.8% at B=32**; full-context restore is 1.4–41 ms. This is presumably *why* the in-kernel BDR path and the amortizer exist — but the config surface still allows enabling the software cache in serving. **Fix:** hard-gate the Python restore path out of production configs (or emit a startup error above B=1) so nobody ships it by accident.

**F-14 [C][MED→LOW, corrected] `model_engine.py:705-714` — kvarn restore step-gate: no device syncs, host-only cost; gate semantics are correct.**
An audit sweep flagged `int(kv_lens[i])` as B CUDA syncs/step and a possible stale-skip correctness trap. Verified: `kv_lens_runtime` is the **host** tensor (`trtllm.py:600`, passed as `host_past_key_value_lengths` at `:1569`), so there are no device syncs; and the key `(req_ids, kv_lens[i]//tokens_per_block)` *does* change on block-boundary crossings, which is exactly when restore matters — semantics correct. Residual: O(B) per-element `int()` on a tensor is needlessly slow Python (~15 µs @B=8, ~100 µs @B=64). **Fix:** `tuple(kv_lens[:len(ids)].tolist())` — one C++ hop.

**F-15 [A][MED] `mlaKernels.cu` `dequantCopyKVarN` — scalar per-element unpack in the read kernel.**
Each thread unpacks ELTS values with per-element shifts/casts and no wide loads; int4 pairs could come in via uint16/uint32 vector loads with lane-shared scales. The read path is the latency-sensitive one (decode). **Fix:** vectorize the unpack; estimated 1.3–1.8× on the dequant-read microbench.

**F-16 [A][LOW] `mlaKernels.cu` `mlaBdrQuantizeLatentKernel` — fixed-kLanes reduction loop and single-lane scale writes; minor occupancy/ILP polish only.**

**F-17 [A][LOW] `kvarn_core.py:103-104` — best-state masked update recomputes `exp()` and runs even when `better.any()` is False on most tiles (the `.any()` is itself a device read on GPU tensors — but this code runs under the store path where a sync already exists; keep an eye on it if the store ever moves on-stream).**

### C. LayerSplit

**F-18 [V][HIGH] Indexer read-set broadcast exists only on a path the live model may not take.**
The broadcast (`dsa.py:4376-4385`, inside `Indexer.forward`) is the *only* call site of `_layersplit_compute_read_block_ids`. But the live MLA flow calls `self.mqa.indexer.sparse_attn_indexer(...)` directly (`modules/attention.py:1973`) after running `pre_indexer_proj` via custom op — `Indexer.forward` is bypassed (its only in-repo invocation is a commented-out line, `dsa.py:4527`). Two possibilities, both bad: (a) the prod CP prefill config drives the module path and pays it on **every** layer (see F-19), or (b) the prod config drives the custom-op path and the indexer-K prefix **silently goes stale on chunk ≥ 2 / prefix reuse** — the exact correctness bug the commit message for `f33372f54` says this broadcast prevents. **Fix:** move the read-set broadcast into `sparse_attn_indexer` (single entry point), or assert at init which path is active under `layersplit_enabled`.

**F-19 [V][HIGH] No `skip_topk` gate on the read-set broadcast.**
`Indexer.forward` issues the full-prefix indexer-K broadcast before any F/S short-circuit. S-layers (43 of 58) reuse the F-layer's top-k and never compute logits — their indexer-K cache is never read — yet on the forward path each S-layer re-broadcasts its full prefix read-set every step. **Fix:** `if not self.skip_topk:` around the broadcast (1 line) — saves up to 43/58 ≈ 74% of indexer-channel broadcast bytes.

**F-20 [V][MED-design] Full-prefix re-broadcast every step with no incremental tracking.**
The read set is `[0, kv_len)` per request — required once (peers' scratch holds no history under owner-local alloc), but at decode the prefix grows ≤1 block/step while the whole prefix is re-sent each step (gather + NCCL/IPC + scatter = 3 HBM touches of the full read-set bytes, per layer). z.ai §4 accepts ~1/8-of-KV exposed broadcast; the current implementation pays it **per step** instead of per new block. **Fix (substantial):** peer-side block-version cache (broadcast only blocks whose version changed) — turns steady-state decode broadcast into write-set-only + first-touch; or keep full-prefix only for the first step after onboard/restore.

**F-21 [V][MED] `dsa.py:65-66` (and the active-set twin) — per-layer H2D of `kv_lens`/`seq_lens` + arange/mask/`unique` recomputed per layer.**
`kv_lens[:num_seqs].to(device=...)` copies host→device on every call, and the read-set is identical across layers within a step. At 58–61 layers × 2 tensors that's ~120 H2D launches + 60 redundant `unique` kernels per step. **Fix:** compute once per step in `prepare()`, cache on metadata.

**F-22 [A][MED] `layersplit.py:838-846, 1009-1010` — per-`(layer, channel)` payload buffers (up to 122 persistent allocations, ~100+ MB/rank at prefill sizes).** Ping-pong pair per channel sized to max-layer payload instead.

**F-23 [A][MED] `layersplit.py:529-577` — IPC ring depth=2 lets the producer run at most one layer ahead before stalling on peer credits; depth should scale with payload (e.g. `max(2, ⌈payload/4MB⌉)`), or copies should move to a dedicated produce stream.**

**F-24 [A][LOW] `layersplit.py:1171-1178` — broadcast is synchronous on the default stream by design (M9 measured sync > overlap at current scale). Fine today; leave a config knob + doc note rather than code surgery. The M6/M8b overlap scaffolding exists but is intentionally dormant — the docs read as if overlap is active; they should say it is not.**

**F-25 [A][LOW] `layersplit.py:1256+` — M5g fused multi-pool broadcast measured 0.77× (cat/split overhead) and is correctly not used; consider deleting or env-gating to stop bitrot. Also: duplicated guard ladders across the two `maybe_broadcast_*` entry points; silent NCCL fallback when IPC setup fails on one rank (log it loudly).**

### D. SMC (Sequential Monte-Carlo speculative decoding)

**F-26 [C][HIGH, corrected] Odd-M draft GEMMs permanently route to the SGLang **Triton** packed-scale path — fallback-kernel throughput, not padding, is the cost.**
An audit sweep claimed every draft step pads M→128 (≈740 MB/step waste). Verified against `torch_custom_ops.py`: at the live draft shape (M=25, packed int32 UE8M0 scales, SM100) the dispatch in `fp8_swap_ab_gemm` (`:2006-2025`) hits `_should_use_packed_scale_triton_swap_ab_odd_m` → `_fp8_swap_ab_packed_scale_triton_matmul` (`:1939+`), whose scale preload **is cached** (`:1804-1821`) — no per-step conversion, no M-padding. The padded-DeepGEMM and dequantized routes are **hardcoded off** (`_should_pad_fp8_swap_ab_odd_m` and `_should_use_dequantized_swap_ab_odd_m` both `return False`, `:1724-1737`) because they faulted on B200; the CUDA quantizer is also bypassed for M%8≠0 (`:1752-1755`, illegal-access workaround) so quantization runs on Triton too. Net: **every draft forward, all GEMMs run on a Triton block-FP8 fallback instead of DeepGEMM/CUTLASS** — typically 1.5–3× slower at small M — plus `logger.warning_once` machinery on the hot path (cheap but present). This is the most credible single contributor to the documented ~2 tok/s SMC regression. **Fix options:** (a) make the draft batch 8-aligned by construction (choose `n_particles×gamma(+1)` ≡ 0 mod 8 — config-level, zero kernel work); (b) root-cause the CUDA-quant illegal access at odd M; (c) a CuteDSL grouped path for M∈[9,32].

**F-27 [V][LOW] `torch_custom_ops.py:1724-1760` — three dead routing predicates hardcoded `False` and their unreachable branches remain in the dispatch ladder; delete or gate behind env to keep the hot dispatch short.**

**F-28 [A][MED] `smc.py:640-665` + `:314-322` — per-request fallback particle selection still does 3 D2H syncs (argmax `.item()` + 2 in `select_particle`) when the batched pre-pass isn't cached; make the batched path unconditional.**

**F-29 [A][MED] `smc.py:809-830, 857-928` — acceptance walk is a per-depth Python loop; rejection-sampling path moves `(q,p,u)` host-side per depth (`.tolist()`). ~24 depth-iterations × 2–5 µs × batch — vectorize to one device pass + single transfer.**

**F-30 [A][INFO] The ~2 tok/s e2e regression decomposes (best estimate) as: F-26 (draft on Triton fallback) > F-28/F-29 (sync + host-walk serialization) > resample/ESS overhead. An nsys capture of one SMC decode step would settle the split in minutes; do that before optimizing.**

### E. MoE / FC2 / WarpDecode / megakernel

**F-31 [V][HIGH] FC2 N-tile=160 default-on **pins the tactic list to a single candidate** — for every caller of the op, not just the validated shape.**
`cute_dsl_custom_ops.py:2007-2014`: when `TRTLLM_OPTRT_FC2_NTILE_160` (default **"1"**) is set, `mma_tiler_mn_candidates = [(tile, 160)]` and clusters are restricted to `(tile//128, 1)` — the previously-validated `[128, 256]` sweep is *removed*, so the AutoTuner cannot recover if 160 loses on a different M/N/K (other models, EP configs, prefill-side use of the finalize op, skewed expert loads). It was driver-verified at exactly one shape (REAP decode H=7168, I=2048; −14.1%, cos 0.99961). `finalize_fusion.py:61-63` widens the *validity* set under the same env — that part is fine. **Fix:** append 160 to the sweep instead of replacing it (`[(t,128),(t,160),(t,256)]` + cluster variants); determinism for the REAP shape can come from the autotuner cache, not from amputating the search space.

**F-32 [A][MED] Megakernel JIT compiles in the decode hot path for any shape outside the warmed bucket set (`fused_moe_megakernel_jit.py`); opt-in today, but if enabled, first-hit compile is 10–50 ms inside a decode step. Precompile the standard buckets at init.**

**F-33 [A][MED] WarpDecode `_NVFP4_TARGET_TACTICS` covers token buckets ≤32 only (`warp_decode.py:114-121, 333-380`); >32 falls back to `[-1,-1]` AutoTuner inside the decode loop — tuning latency plus lost CUDA-graph determinism. Extend the table to 48/64 or add closest-bucket fallback.**

**F-34 [A][LOW] Megakernel/WarpDecode registration failures are swallowed into a `warning_once` and silently fall back (`warp_decode.py:46-78`); if the env var is explicitly set, fail hard.**

### F. Host / pyexecutor / disagg

**F-35 [V][OK] Snapshot hooks: installed once at executor init (`py_executor.py:567`), env-gated, zero steady-state cost when disabled. No action.**

**F-36 [A][MED] `py_executor.py:3408-3416` — idle disagg loop polls `has_any_inflight_requests()` every 0.1 s under a lock that scans the in-flight list; cheap now, scales with transfer count. Consider event-driven wakeup or a lock-free counter.**

**F-37 [A][MED] `resource_manager.py:1227-1289` — several `_optrt_kv_debug(...)` call sites build per-request tuple/lists as *arguments* before the gate inside the callee returns; at B=64 this is 50–100 µs/step of dead allocation when debug is off. Wrap call sites in `if _OPTRT_KV_DEBUG_ENABLED:` (the pattern already used at `py_executor.py:2407`).**

**F-38 [A][LOW] `cuda_graph_runner.py:336-392` — four separate debug-gated `tp_allgather` branches; cache the gate at module level (pattern exists elsewhere in the same commit) and collapse to one branch.**

**F-39 [V][OK] `0adc87009` debug env-gate caching and eager-kwarg removal are correctly done at module/instance scope; spot-checked, no per-step `os.environ` reads remain on the touched paths.**

### G. Bench methodology (what the numbers can and cannot support)

**F-40 [V][HIGH] `nvfp4_gemm_backend_microbench.py:83-85` — sorts samples, drops the slowest 20%, then medians ⇒ reports ≈P40 of a sync-bracketed wall-clock distribution.**
Each sample is `perf_counter` around `g.replay()+synchronize()` — fine for relative ranking, but absolute µs include per-replay sync overhead, and the sorted-tail drop biases every cell optimistically. The "cutedsl wins 54/70 cells" conclusion is *probably* directionally right (all backends share the bias) but per-cell win% under ~3% is inside the bias band. Also: `microbench_results_b200.log` referenced by the README is **not committed**. **Fix:** report untrimmed median + P90, commit the log.

**F-41 [V][HIGH] Lever-2 / R4 numbers do not reproduce at HEAD (see §2.2).** `INDEXER_R4_RESULT.md`'s aggregate (−43.5% indexer TPOT, A_F 36.92→16.43 µs) embeds the lever-2 S-layer saving; the standalone benches behind Win A/Win B are either not committed or broken (F-42), so the aggregate is currently a projection resting on one reproducible number (the floor) and one stale one (affine reuse).

**F-42 [V][HIGH] Scripts that cannot run as committed:** `topk_scheme_probe.py` imports a `logits_launch_probe` module that does not exist in the repo (Win B's 12288-width crossover is unreproducible); `bench_kvarn_*.py`/`test_kvarn_{backend,amortize}.py` assume a `/tmp/kvarn_bench` loose-module layout; `bench_kvarn_component.py` inserts `/home/spencer/work/...`; `test_kvarn_cycle.py`/`bench_kvarn_amort_e2e.py` assume the repo at `/repo`; `bench_kvarn_gqa_sparse.py` defaults `--repo /workspace`; `test_kvarn_backend.py` requires `TRTLLM_MLA_LATENT_KV_DTYPE` set. None of this is fatal in the original container, but it means **CI can never guard these claims**. Fix: relative paths + a tiny `conftest`/env-bootstrap, and commit the reference logs the docs cite.

**F-43 [V][MED] `bench_kvarn_component.py:106-112` — wall-clock (`time.time`) around 10 iterations for a multi-kernel eager pipeline; sync-bracketed so it's not wrong, but it folds Python dispatch of ~10 kernels/iter into "µs per block" and has no variance reporting. The committed README's "quant overhead" figures inherit this.**

**F-44 [A][MED] Cross-cutting bench-realism gaps:** L2-resident inputs in dequant/store loops (prod KV reads are cold) — KVarN read-path numbers are best-case; no GPU clock locking anywhere (B200 boost variance ±5–10%); lever-2/affine measured at a single shape (M=8, prefix 4608); `bench_kvarn_amort_e2e` uses a 64-step window (≈1 fill event); correctness gates are cosine-on-latents only — no end-to-end attention-output equivalence test exists in the tree (a magnitude-preserving error pattern that shifts softmax would pass every committed check); `bench_kvarn_outlier.py` measures quality only, despite the "0.18% overhead" claim living nearby.

**F-45 [V][POSITIVE] `bench_kvarn_decode.py` reproduces its committed log within ~2% on every row — the KVarN system benches are deterministic and replicable once paths/env are fixed. The correctness suites (cycle/MLA/preset/intel) all pass unmodified logic.**

**F-46 [V][HIGH] `test_kvarn_backend.py:46-49` — committed test broken by an env-var rename.**
The test sets/deletes `TRTLLM_KV_CACHE_QUANT`, but `resolve_kvarn_config()` reads `TRTLLM_MLA_LATENT_KV_DTYPE` (`kvarn_backend.py:153`). The env-resolve assertions can never pass as committed (and pre-setting the right var externally breaks the subsequent `is None` assert). The suite passes fully once the key is corrected — confirming this is a rename leftover, not a logic bug. It also means **this suite has not actually been run since the rename**, which weakens "tests pass" as evidence for any change after that point. Fix: one `sed`, plus run the suite in CI.

**F-47 [V][MED] `kvarn_inkernel/results/` mixes logs from three kernel generations without labels.**
`inkernel_dequant_sweep_v1.log` (5.75× slower than fp8), `bdr_inkernel_bench` (1.3–1.5× slower), and `bdr_persub_inkernel_sweep.log` (0.65×, production) all coexist; only the last reflects the shipped kernel. Anyone citing the directory can pick a stale number in either direction. Fix: prefix obsolete logs with `OBSOLETE_` or add a README index mapping log → kernel generation → commit.

**F-48 [V][MED] `kvarn_inkernel/fmha_overlap_probe.py` — conclusion line overstates overlap.**
Probe prints that the 489 µs fused dequant "would OVERLAP" 132 µs of attention compute; at most ~27% can be hidden. The planning docs that reference overlap-based budgets should be re-checked against the exposed ~357 µs.

---

## 4. Claim trust table (post-reproduction)

| Documented claim | Source | Reproduction result | Trust |
|---|---|---|---|
| KVarN restored cos ≥0.993; sink/tail bit-exact | `system_cycle.log` | ✅ reproduced | **HIGH** |
| KVarN dequant 1.24 µs/blk @256; restore %budget table | `system_decode_overhead.log` | ✅ within ~2% | **HIGH** |
| KVarN capacity multipliers (bytes/elem) | component bench | ✅ deterministic arithmetic | **HIGH** |
| KVarN 0.992 cos @128K (per-sub-block BDR) | `kvarn_inkernel/results` | ✅ **reproduced** (`bdr_longctx_bench`: 0.992 holds; naive/per-token collapse 0.866) | **HIGH** |
| KVarN BDR read 0.46–0.67× fp8 read | commit `76204896f` | ✅ reproduced for the production kernel (0.65×); obsolete in-tree logs show older kernels LOSING to fp8 (F-47) | **HIGH (prod kernel only)** |
| KVarN amortized restore 3.1–7.6× | `system_amortized_e2e.log` | ✅ reproduced (8.3× at B=32/1024 blk; ON = 64% of budget) | **HIGH** |
| Indexer floor: logits 6.17 µs graphed, A_F graphed ≪ eager | `INDEXER_R4_RESULT.md` | floor not directly re-run (needs the missing probe); eager-vs-graph reframing is sound | **MEDIUM** |
| Affine reuse 1.78×/1.68×, −0.071 ms/tok | `LEVER2_RESULT.md` | ❌ **1.04×/1.01×, −0.007 ms/tok at HEAD** | **LOW — stale** |
| Top-k DSL/C++ crossover at width 12288 | R4 Win B | ❌ probe script broken (missing module) | **LOW — unreproducible** |
| R4 aggregate −43.5% indexer TPOT | `INDEXER_R4_RESULT.md` | projection; one input stale (above) | **LOW–MEDIUM** |
| cutedsl wins 54/70 cells; o_proj +8.5–12.3% | nvfp4 README | ✅ conclusion reproduced (58/70 cutedsl, cutlass never wins); per-cell win% drifts a few points (P40-biased metric, log still not committed) | **MEDIUM+ (decision sound, cell values soft)** |
| FC2 N-tile −14.1% @REAP, cos 0.99961 | commit `2833bb0c5` | not independently re-run; *pinning* concern is orthogonal (F-31) | **MEDIUM** |
| SMC ~2 tok/s regression cause unknown | deploy notes | F-26 provides the most credible mechanism | n/a |

---

## 5. Top-10 actions, by value-per-effort

1. **`dsa.py:3048` one-line gate reorder** — kills hundreds of D2H syncs per prefill step (F-1).
2. **Re-validate lever-2 affine reuse with nsys at HEAD**; if the win is really gone, revert the S-layer dispatch complexity or fix the kernel (F-41/§2.2).
3. **FC2: append 160, don't pin** — removes a default-on global tactic-space amputation (F-31).
4. **Make the SMC draft batch 8-aligned by config** — likely recovers a large share of the 2 tok/s without kernel work (F-26).
5. **`skip_topk`-gate + per-step cache for the LayerSplit read-set; unify the indexer entry point** — up to ~74% indexer-channel broadcast bytes + removes a latent staleness correctness hazard (F-18/19/21).
6. **Sinkhorn: iters 16→4 (validate), fix the double-recompute loop, vectorize pack/unpack** — ~3–6× store-path reduction combined (F-9/10/11).
7. **`cumsum` the chunk-spec build** (F-2) and **hoist the chunk-loop allgather** (F-5).
8. **Fix the microbench tail-drop + commit reference logs + make scripts runnable from the repo** — restores CI-guardability of every perf claim (F-40/42).
9. **WarpDecode tactic table → 48/64 buckets; megakernel precompile at init** (F-32/33).
10. **Wrap debug-arg construction at gated call sites; event-driven idle disagg wakeup** (F-36/37).

---

## 5.1 Status update — fixes landed (2026-06-10, same day)

Upstream pulls (cherry-picked, all Python-side / overlay-deployable):
`33b0a3299` + `f0ba8c721` (DeepGemmFusedMoE Triton fusions — unit tests pass 10/10 on B200 in our container), `6254f3a16` (DSA indexer K-cache UINT8 sizing — merged with our independent fix of the same bug, adopting upstream's dtype-unscaled semantics), `57413893a` (DSA DSL atom-split MTP guard), `9bc43218b` (disagg quorum vote TP-scoped — our tree lacked the vote entirely and had the rank-divergence deadlock exposure), `3b4672876` (MLA decode workspace counter clear — helper ported into our diverged `run_mla_generation`). Deferred: `451dbb8b2` (routing-kernel BLOCK_SIZE, needs `.so` rebuild).

Audit findings fixed and validated:
- **F-1 + F-2** (`59c687199`): width-probe before the eager HISA sync; O(B) chunk-spec build.
- **F-19 + F-21** (`59c687199`): LayerSplit indexer broadcast gated on `skip_topk`; read-set computed once per step (cache invalidated in `prepare()`).
- **F-9** (`e5a095fd5`): Sinkhorn loop restructure — **bit-exact** (cycle/preset outputs identical), component-bench quant 35.05 → 30.11 µs/block at iters=16.
- **F-10** (follow-up commit): default Sinkhorn iters 16 → 4 after an 8-seed sweep showed quality flat-to-better (cos 0.99351→0.99369; cycle restored-cos improved 0.99328→0.99363; preset table bit-identical at 4 vs 16). **Net store path: 9509.7 → 3105.4 µs single-block (amortized 148.6 → 48.5 µs/step), 3.1×.**
- **F-14** (`c0105c924`): step-gate key via one `.tolist()`.
- **F-31** (`28bee8268`): FC2 160 appended to the sweep instead of replacing it; explicit skip of the invalid (160, cluster_n=2) combo — note `is_valid_mma_tiler_and_cluster_shape` does NOT reject it, contrary to what a hasty reading of the original comment suggests.

**Round 2 (same day, later):**
- **F-26 FIXED**: odd-M packed-scale SwapAB now pads to 8-aligned M and takes the autotuned DeepGEMM path (the historical fault was the odd-M CUDA quantizer, which padding sidesteps; it no longer reproduces). Measured: 1.35× on the M=25/N=18432 draft GEMM through the op, 1.17–1.19× at M=50/100; `TRTLLM_FP8_SWAPAB_ODD_M_TRITON=1` restores the old route. All `-k odd_m` unit tests pass incl. CUDA-graph replay.
- **F-18 FIXED**: the indexer read-set broadcast moved from the bypassed `Indexer.forward` into `sparse_attn_indexer` (both entry paths covered; reuse layers/steps skip it via the existing early returns). `test_dsa_indexer` failure set unchanged vs baseline; `test_layersplit_ownership` 81/81.
- **F-5 FIXED**: one deferred allgather for all q-split prefill chunks (R20 prefill is q_split-eligible: `enable_attention_dp=false`, TP2). Assembly math validated equal to the per-chunk reference across tp×rank×chunk-layout combinations.
- **F-3 FIXED** (log-once on the padded full-width top-k fallback), **F-37 FIXED** (per-request `get_token_count` C++ calls no longer paid with debug off), **F-38 FIXED** (cg debug gates lru_cached).
- **F-40/F-46/F-47/F-48 FIXED**: microbench reports untrimmed median; `test_kvarn_backend.py` env-var rename repaired (passes as-committed now); `kvarn_inkernel/results/README.md` indexes the three kernel generations; fmha probe prints honest overlap coverage.
- **F-41 RESOLVED** via nsys — see §2.2: both kernels healthy; true lever-2 saving ≈ 0.013 ms/token (device-level), 5× below the documented figure.
- **F-32/F-33 DOWNGRADED**: uncovered WarpDecode buckets resolve their tactic at *capture time* through the warmed autotuner cache, not per-replay — no decode-loop tuning in graph mode. Extending the table to 48/64 needs a `warpdecode_tactic_retune.py` run if such buckets ever ship; megakernel is opt-in experimental, precompile deferred.

Still open: LayerSplit CP=2 e2e revalidation before the next R20 image bake (multiproc NCCL machinery test run in a 2-GPU container; a real TP2xCP2 model A/B is the remaining gate), `topk_scheme_probe.py` reconstruction (its `logits_launch_probe` helper was never committed), F-20 incremental block-version broadcast (substantial design work), F-36 idle-poll wakeup (idle-time only, low value).

**F-49 [V][MED] `bench_sparse_mla.py` --compare diffs uninitialized output regions.**
The output tensor is allocated uninitialized and the kernel writes only valid positions, so cross-run comparison of the unwritten slots produces NaN diffs and `bit_identical=False` even when the kernel is unchanged (same `.so`, same seed: the LSE plane — fully written — diffs at exactly 0.0 while `o_max_abs=nan`). Fix: zero-init the output or mask the comparison to the valid region; until then treat `lse_max_abs` + timing as the equivalence signal.

**2026-06-11 full-suite re-run (v3, post-merge of Spencer cycles 5-10 + local work):** 28/29 entries pass (only the known-broken topk probe fails, F-42). KVarN system numbers reproduce yesterday's post-fix values within noise (cycle cos 0.99363, component 30.46 µs/blk, store 3245.9 µs ≈ 3.0× over original, amortize 8.8×); `sparse_mla` timing identical to the pre-merge baseline (295.88 vs 296.02 µs, LSE exact — see F-49 for the NaN artifact); NVFP4 winner histogram with untrimmed medians {cutedsl 55, cublaslt 13, cuda_core 2} — the CuteDSL-enable conclusion is robust to the F-40 methodology fix; per-sub-block BDR read 0.64× fp8 and the 128K accuracy verdict reproduce; the corrected fmha probe prints honest coverage (27%). Lever-2 wall numbers swung the OTHER way this run (wired S 2.524 µs ≈ the original doc claim; standalone full-remap an outlier at 14.7 µs) — third consecutive day of large swings, reinforcing that only the nsys device times (1.60/1.30 µs) are trustworthy at these scales without clock locking.

---

## 6. Raw artifacts

- Run driver + logs: `.bench_runs_claude/run_all.sh` (host) → logs inside container `optrt-bench-claude:/tmp/benchlogs/` (`docker cp optrt-bench-claude:/tmp/benchlogs <dest>` to extract).
- Bench container: `optrt-bench-claude` from image `optrt-fusionproof-20260609134115`, GPU 0 only, repo mounted read-only-in-practice at `/repo` (SELinux blocks container→host writes; logs were kept container-local for that reason).
- `sparse_mla_decode_nvfp4` baseline tensor saved at `optrt-bench-claude:/tmp/sparse_mla_baseline.pt` (B=32, s_q=1, topk=2048, pages=4096, seed=1234) for future `--compare` regression runs.
- Audit provenance: six parallel code sweeps (KVarN, DSA/HISA, LayerSplit, SMC+MoE, host/executor, methodology) + line-level verification passes on every [V]/[C] finding above.

---

## 7. E2E serving session (2026-06-11) — baseline numbers; integrated A/B blocked; six new findings

**Baseline (image `optrt-0be07d6df64e-smcquietadpfix-20260608203818`, DSV3.2-REAP-345B, disagg TP2xCP2 prefill + TP4 decode, ISL 4161 / OSL 256, streamed, token counts from usage):**

| C | TTFT p50 | TTFT p95 | tok/s/user p50 | aggregate out tok/s |
|---|---|---|---|---|
| 1 | 610 ms | 660 ms | 42.6 | 44 |
| 4 | 927 ms | 1.22 s | 41.9 | 162 |
| 8 | 1.29 s | 1.94 s | 32.9 | 246 |
| 16 | 1.56 s | 3.20 s | 34.0 | 454 |
| 32 | 1.68 s | 5.78 s | 29.3 | 719 |

Client: `.bench_runs_claude/e2e_sweep.py` (aiohttp streaming, per-request TTFT + steady rate, 2C requests per point).

**The integrated-branch A/B is BLOCKED** on a remote-prefill hang (F-54). Three images tried: merged HEAD, pre-merge (`bb58b7098`), pre-merge-minus-quorum — all overlays on the June-8 base; all hang/fail the >2048-token remote-prefill path while ≤2048 (decode-local prefill) works.

**F-50 [V][HIGH][pre-existing] Prefill engine sampler assert kills the worker.** The ORIGINAL baseline prefill (35h-old pod) died at 01:44 with `AssertionError: Sampling failed` (RANK 2, 2 requests inflight) — before any new code was deployed. Recurring engine bug in the June-8 lineage.

**F-51 [V][HIGH] Dead engines leave pods `Running`/ready.** Three zombie incidents in one session: engine fatal (MPI worker exit / event-loop death) while the pod's 9090 probes stay green, so the router keeps/regains the instance and requests blackhole. The liveness probe must reflect executor health.

**F-52 [V][HIGH] Decode local-prefill fallback fatally asserts.** A 4162-token prompt classified decode-local trips `total_num_tokens <= max_num_tokens (2048)` as an assert → MPI death — one mis-routed request kills the whole decode engine. Should reject the request (or chunk) instead.

**F-53 [V][MED] Discovery races route around the prefill leg.** A fresh frontend (or freshly-registered decode) routes decode-direct until a discovery snapshot lands: instant 500 `Disaggregated params are required for decode mode`; combined with F-52, the FIRST request after a worker boot can kill decode. The 4-minute settle is the operational workaround.

**F-54 [V][BLOCKER, unresolved] Post-June-8 Python overlay on the June-8 base hangs remote prefill.** Symptom: request received by the prefill dynamo handler, never enqueued to the engine — rank0 idle at `ipc.get`, ranks 1-3 parked at the request broadcast, handler coroutine await-parked; hang detector eventually kills the ranks. Ruled out as sole causes (each tested by deployment): Spencer's cycles 5-10 (pre-merge image also hangs), the TP-quorum cherry-pick (no-quorum image also fails), FC2 N-tile 160 (env-disabled). Remaining suspects: the June-7-9 LayerSplit/NIXL Python (`f33372f54`/`fc8cdba85` lineage) against June-8 NIXL binaries, and/or our dsa.py prefill-path changes (F-1/F-2/F-5/F-18) under real TP2xCP2 — neither testable single-GPU. Next steps: (a) pair the integrated tree with its own binaries via `build_fullsource_image.sh` (Spencer's flow — his cycles were validated that way), (b) build a 2-GPU CP=2 remote-prefill repro rig for cheap bisecting.

**F-55 [V][HIGH][confirmation] FC2 N-tile 160 faults at decode autotune.** The pre-merge image (which still carried F-31's default-on 160 sweep) crashed decode warmup with `CUDA illegal memory access` inside autotune — independently confirming `e105fd7a1`'s SFB/partial-tile analysis at decode shapes (his report covered prefill M=1024). The F-31 env gate (`TRTLLM_OPTRT_FC2_NTILE_160=0`) mitigated without a rebuild; the merged branch has 160 fully removed.

**Ops notes:** `render_dgd.sh` defaults `--target-node` to a4-us-001-rl9 — always pass the node; the curated overlay COPY list rotted (ImportError + stale-module TypeError) — use `Dockerfile.r20-overlay-fulltree` and verify by per-file content hash against the built image; the lab deployment was rolled back to the baseline image and verified at session end.

---

## 8. FINAL A/B (2026-06-12) — integrated branch validated e2e; +26-50% over baseline; root-cause chain closed

**Stack:** fullsource image `optrt-19d82b488-fullsource-msgpack-0426` (merged HEAD: Spencer cycles + our 17 commits, matched C++/Python), config: `max_num_tokens: 8192` (decode), **SMC drafting OFF** (as Spencer's own perf numbers run). Full program green end-to-end: probes, tight harness, sweep, profiling.

**Tight harness (Spencer's regime — uniform ~2055 ISL, OSL 512, C=16, his reference 39.98–40.28 tok/s/user):**
**51.6 tok/s/user p50, 760 tok/s aggregate** — reproduced ×3 (51.58 / 51.66 / 51.86). **+28% over the documented cycles-era number.**

**Full sweep (ISL 4161 / OSL 256) vs the June-8 baseline:**

| C | tok/s/user base→new | aggregate base→new | TTFT p50 base→new |
|---|---|---|---|
| 1 | 42.6 → 39.7 | 44 → 41 | 610 → 651 ms |
| 4 | 41.9 → **50.8** | 162 → 179 | 927 → 1309 ms |
| 8 | 32.9 → **47.7** | 246 → 329 | 1290 → 1238 ms |
| 16 | 34.0 → **46.5** | 454 → 547 | 1564 → 1627 ms |
| 32 | 29.3 → **44.0** | 719 → **907** | 1682 → 1574 ms |

Aggregate +26% at C=32; per-user +37–50% at C=8–32; TTFT comparable (better at high C). C=1 within variance of baseline. Decode GPUs sample near-idle under c16 (profiled rank parks in the request broadcast) — the stack remains **overhead-bound** per Spencer's composite; his a2a lever (4.3–4.7 ms/step) is the next headroom.

**The full root-cause chain behind two days of "hangs" (F-54 final disposition):**
1. **SMC drafting silently active** — `speculative_config` (SMC + GLM draft) ships ENABLED in the topo template and the live config; inert on June-8 Python, fully wired after `smc-sd` (`2634a162d`). The draft runs **single-rank + eager**: ~300 ms of python-dispatched draft GEMMs per decode step (py-spy: rank0 mid-`fp8_swap_ab_gemm`, peers parked; GPUs ~5%). Measured 2.7–2.9 vs 36–38 tok/s incl-prefill with the block present/absent on the SAME image. Template now ships it commented with rationale.
2. **Drafter context pass is unchunked** — `prepare_draft_tokens` asserts `total_num_tokens <= max_num_tokens` and the assert **kills the MPI worker** (zombie pod, F-51). With decode `max_num_tokens: 2048`, any prompt >2048 was a one-request engine kill; the tight harness's 2048-token prompts ride exactly on the edge. Fixed to 8192 in decode.yaml AND the topo template (the render's inline copy silently reverted the first fix). Code follow-up: chunk the draft context pass; reject, never assert.
3. **Template debt** ×3: inline decode config diverged from `decode.yaml`; orphaned SMC keys re-parented into `sparse_attention_config` on render (new-schema pydantic crashloop); `--target-node` defaults to the other cell node. All fixed/committed.
4. Acquitted along the way: the quorum cherry-pick (identical crawl without it), C9 IPC (opt-in + parked), L1 overlap, our dsa/prefill changes, python↔binary skew (fullsource pairing changed nothing), dynamo version (byte-identical across images), MO1 core (his legacy-path port covers it).
5. Infra fixes that made the loop workable: msgpack absent from the fullsource runtime stage (worker ImportError crashloop; one-layer patch + Dockerfile note pending); registry-vs-containerd image GC (the overnight ImagePullBackOff); crashloop/wall-cap bails + image-verified, name-guarded pod waits in the orchestration script.

**Artifacts:** `/tmp/e2e_FINAL2.log` (full program), `/tmp/decode_rank2.speedscope` (c16 host profile), `/tmp/dmon_window.log` (GPU telemetry), `.bench_runs_claude/e2e_full_program.sh` + `e2e_sweep.py` (the harness).

## 9. Kineto profile (2026-06-12) — per-iteration breakdown + the three follow-up fixes

**Capture:** built-in PyTorch/kineto hook (`TLLM_TORCH_PROFILE_TRACE` + `TLLM_PROFILE_START_STOP=3000-3120`), 4 rank traces at steady c16 (51.7–52.2 tok/s/user), `torch_trace-rank-{0..3}.json` (368 MB each) under `/var/lib/optrt-cache/nsys/`.

**Steady-state per-iteration anatomy (c16, 120-iter window):** GPU 95.8% busy in-step (span 17.5 ms, busy-union 16.7 ms, kernel-sum 23.1 ms → ~28% cross-stream overlap). Top costs/iter:
- **MoE comm + prep ~4.0 ms (24%)**: a2a 2.06 ms + prep trio 1.59 ms (`memsetExpertIds` 580 µs / `computeCountAndIndice` 590 µs / `cumsum` 420 µs, ~58 each = per-MoE-layer) + finalize 340 µs.
- nvjet 64x8 swarm 2.77 ms (122 calls ~23 µs each — cuBLAS-internal GEMM tiling).
- `quantize_with_block_size` 1.12 ms (392 launches, ~2.9 µs each — M=1 activation quant).
- `cudaGraphLaunch` 2.81 ms/launch host (1/iter; hidden under GPU work at c16, caps c1 latency).
- `cudaEventSynchronize` 2/iter ≈ 5.6 ms (pacing). aten glue ~2.9 ms/iter host.
- Autotuner warmup warning: `nvfp4_gemm (1,3584)×(2112,3584)` no valid tactic.

### The three fixes

**#1 — Fuse the MoE prep trio (DONE, committed `e14dfc859`).** Of the three prep kernels, `computeCountAndIndice` is the irreducible comm kernel (sender/receiver blocks over the FIFO workspace, a cross-block barrier it can't share). The clean, **bit-exact** fusion is `computeCumsum` → `moveIndice`: `computeCumsumDevice` was a trivial 2-block cub `BlockScan` over the per-rank counts whose only consumer was `moveIndice` (and the downstream pad). Folded the inclusive scan into `moveIndiceDevice` — each CTA reads the raw per-rank counts (rankCount ≤ 64) and scans them in shared memory before its gather; CTA0 publishes the cumsum to the output buffers before the PDL trigger so a dependent grid's `gridDependencySynchronize` observes it. Raw counts now land in dedicated scratch buffers (no read/write alias with the cumsum outputs). Arithmetic identical; op signature + returned tensors unchanged → no Python/`register_fake` change. **Removes one kernel launch per MoE layer (~58/iter) from the captured CUDA graph.**

  - `memsetExpertIds` (the third kernel) was *not* fused: it pads the recv expert-ids tail **after** the alltoall (the pad target is the alltoall's output buffer, allocated inside the comm op; the tail is stale until re-padded each step). Folding it would require the generic comm op to learn the expert-id field index + `invalidExpertId` and pad in its epilogue — invasive into a backend-shared op. Deferred with rationale.

**#2 — `cudaGraphLaunch` host cost (2.81 ms/launch).** This is the host cost of replaying a graph with a very large node count; it is *hidden* under GPU work at c16 (95.8% busy) and only bites c1 latency. It is not a single kernel to fix — it falls out of node-count reduction. #1 removes ~58 nodes/iter from the captured graph; the post-#1 kineto re-capture quantifies the delta. _[A/B: pending re-profile]_

**#3 — The quant storm (392 `quantize_with_block_size`/iter, 1.12 ms) — investigated, no safe drop-in.** This is **not** a batchable loop: the MoE input quant is a single `fp4_quantize` per layer (`fused_moe_wide_ep.py:523`); the 392 count is one activation-quant per fp4 GEMM (MLA projections, gate/up/down, indexer), i.e. the architectural floor. The only lever is **fusing the activation quant into the producing RMSNorm** — `trtllm::flashinfer_fused_add_rmsnorm_quant` exists, but (a) pulls a flashinfer runtime dependency the TRTLLM attention backend doesn't otherwise need, and (b) changes quant rounding, so it needs full e2e numeric revalidation on this custom NextN-graft model. Too risky for the validated baseline as a lab drop-in; flagged as a real headroom item, not landed.
  - The autotuner hole (`nvfp4_gemm (1,3584)×(2112,3584)` no valid tactic) is **benign**: `AutoTuner.search_cache` returns the fallback `(runner[0], tactic=-1)` (`autotuner.py:440,1018`), which `NVFP4GemmUnifiedRunner` implements — warmup warning only, no crash, runs at 52 tok/s. Not worth a code change.

### #1 validation (2026-06-12) — fused image deployed; correctness proven

Built `optrt-7cb0d17b8ef2-cumsumfuse-…`, deployed via `render_dgd.sh --target-node a4-us-002-rl9` (config unchanged: `max_num_tokens 8192`, no SMC). Results:

- **Throughput identical to baseline:** tight-c16 **52.08 / 51.68 tok/s/user**, agg 768/766 (baseline §8: 51.6 / 760). The fusion costs nothing.
- **Zero CUDA/kernel errors** through model load + warmup (0 pod restarts); the first real MoE forwards exercised the fused kernel cleanly.
- **Numerical correctness PROVEN** (`.bench_runs_claude/test_movefuse.cu`): the fused `moveIndiceDevice` (verbatim) vs a CPU golden reference — **8/8 cases PASS** (rankCount 2/4/8/16, incl. the deployment's ep_size=4 and edge cases), **0 mismatches** on the published send/recv cumsum AND all three gather outputs (send/backward/recv). This is the deterministic correctness proof.

**Validation gate (per `docs/blaise/`, not a new finding).** The target checkpoint is **untrained** and end-to-end text is gibberish, so **e2e text is explicitly NOT a correctness signal** — this is documented in `docs/blaise/README.md:29-33` ("Validation philosophy") and `docs/blaise/optimization_candidates.md:320` ("Correctness gating (untrained checkpoint; the 0.98 functional bar)"). The directive: gate every correctness-affecting change **kernel-level against a TRUE reference** (not the candidate's own fused-vs-sequential output — the K2 lesson), functional cosine ≥ 0.98; **state-machine / routing-index gates are exact-equivalence, not cosine**. #1 is gated exactly this way: `moveIndice`'s outputs are **integer routing indices + the cumsum**, so the gate is bit-exact equivalence vs an independent CPU golden reference (`test_movefuse.cu`, 8/8 PASS) — a true reference, not fused-vs-sequential. Throughput parity (52 tok/s) + zero kernel errors are the secondary signals. (I briefly chased output coherence before re-reading `docs/blaise/` — recorded here so the gate isn't re-litigated.)

### #1 + #2 perf A/B (2026-06-12) — kineto re-capture, fused vs baseline

Re-captured the kineto trace on the fused image (steady c16, same window). Analyzer (`.bench_runs_claude/analyze_trace.py`) over rank-0, baseline vs fused:

| metric (per iter) | baseline `0426` | fused | delta |
|---|---|---|---|
| `computeCumsumDevice` launches | 58 | **0** | **eliminated** |
| `moveIndiceDevice` signature | 9-arg | 12-arg `const*` | fused code live ✓ |
| total kernel nodes / iter | 3429 | 3372 | **−57 (−1.7%)** |
| tight-c16 tok/s/user | 51.6 | 51.9 | neutral (within noise) |
| `cudaGraphLaunch` host | 2810 µs | 2953 µs | +143 µs = run-to-run variance |

**What the A/B actually shows (honest read):**
- **#1 works and is correct:** the standalone `computeCumsum` kernel is gone (58→0/iter), `moveIndice` carries the new signature, throughput is unchanged. The per-kernel *durations* can't be compared across captures — they include PDL `gridDependencySynchronize` wait time and vary run-to-run (the **unchanged** `computeCountAndIndiceDevice` "rose" 593→748 µs purely from cross-capture variance; `moveIndice`'s 135→627 µs is the absorbed cumsum work + that same variance, not a compute regression — throughput parity confirms no net GPU-time loss).
- **#2 is falsified by measurement:** the fusion removes **1.7%** of the ~3429 graph nodes/iter. If `cudaGraphLaunch` scaled linearly that's ~48 µs of its 2.81 ms host cost — **below the ±150 µs measurement noise**. So a single per-layer kernel fusion does **not** measurably move the graph-launch host cost. Cutting that 2.81 ms requires removing a *large* fraction of the 3429 nodes (whole-layer megakernel fusion), not picking off individual prep kernels.

**Net:** #1 is a clean, proven-correct, throughput-neutral simplification (−1.7% graph nodes, marginally helps c1 launch latency). It is **not** the big lever. The profile is unambiguous about where the real headroom is: the **a2a itself (2.06 ms/iter, ~24% of the step with comm+prep)** — exactly the lever Spencer's own composite flags. Attacking it means changing the NVLINK_TWO_SIDED all-to-all algorithm/overlap, a substantially larger project than per-kernel fusion, and is the recommended next focus. The quant storm (#3) and graph-launch cost (#2) are both architectural (norm+quant fusion; whole-layer megakernels), not drop-in wins.

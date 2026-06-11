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

## CRITICAL — the uninitialized-`input_scale` production hazard (2026-06-10)

**Mechanism (runtime-proven).** Before `5bc2b2cb8`,
`NVFP4LinearMethod.create_weights` allocated `input_scale` as
`Parameter(torch.empty([1]))` and `process_weights_after_loading_vanilla`
never nulled it when the checkpoint carries **no activation scales** for the
module — so `_input_prepare` took the **STATIC quant branch with
uninitialized memory**. Every FP4 activation code quantizes to 0 and the GEMM
**silently outputs EXACT ZEROS** (no error, no NaN — probe-observed garbage
scales 1.4e-45…1.76e+22, output amax 0.0). `5bc2b2cb8` fixes it: no
in-checkpoint act scale ⇒ `input_scale = None` ⇒ the dynamic-quant branch.

**Exposed set on this checkpoint** (from `model.safetensors.index.json` —
decisive): `indexer.{wk, wq_b, weights_proj}` × 61 layers = 183 modules — the
**only** NVFP4 modules without `input_global_scale`. Every other NVFP4 module
(routed/shared experts, dense MLP, q_a/q_b/kv_a/kv_b/o_proj) carries
in-checkpoint act scales and is NOT exposed.

**The LIVE 002 r20 deploy is CONFIRMED EXPOSED.** Image
`optrt-0be07d6df64e-smcquietadpfix-20260608203818` (both P/D workers):
`/proc/<pid>/root` in-mount-namespace code check shows the pre-fix linear.py;
no `.py` overlay mounts in the pods; **no `force_dynamic_quantization` escape
hatch** anywhere (0 hits in worker env and served yamls); indexer NVFP4 is
active (`indexer_k_dtype: fp4`, `indexer_mode: indexcache-hisa`). GPU probe on
byte-identical linear.py (wave2 image, md5-equal): all three layer-5 indexer
projections output exact zeros (0/2048, 0/1024, 0/131072 nonzero) on
real-magnitude input; the same image with the 8-line fix overlaid outputs
dense nonzero. **Production DSA top-k therefore runs on all-zero index scores
for every request past `seq_len_threshold = 8192`** (degenerate selection);
sub-8192 traffic never engages the sparse path and is unaffected.

**Remediation** (runbook `/tmp/livehazard_work/REMEDIATION.md`, staged not
deployed): hot-patch overlay `linear_wave2_fixed.py` (hostPath-mount over the
venv linear.py — the deploys already use this .py-overlay pattern) **or**
rebuild from a tree ≥ `5bc2b2cb8`. The live tag exists in **no registry**
(only 002's containerd store), so the overlay is the fast path. Rollout
timing is the owner's call (active workload). **ALL 001-resident images and
the 001 host trees predate the fix** — any future build must include
≥ `5bc2b2cb8` or it re-introduces the hazard.

**Remediation STAGED (2026-06-11): the fixed image is built, pushed, and
verified.** Tag `optrt-34fe7aaec-fixed-20260611011615` (001's
`localhost:5000` registry + dockerd; base = the dynrouterpin full-wheel
image every wave/r20 overlay was built from, proven by RootFS layer-prefix)
carries a **47-file .py overlay = every `tensorrt_llm` .py differing from
the base** (recursive md5 diff, not a hand list) — python content exactly
commit `34fe7aaec` (HEAD at build time; `0a1504755`'s L1 + KVarN-host-path
python **postdates this image** — fold it into the next build). Verification
**4/4 PASS** (001 GPU 7, ephemeral `--rm`): imports; fix present + 47/47
md5 == the Mac tree; **the zero-probe passes — all three layer-5 indexer
projections output dense NONZERO on the real checkpoint** (wk 2048/2048,
weights_proj 1024/1024, wq_b 131072/131072; the pre-fix garbage alphas are
bypassed via `input_scale=None` → dynamic quant); env defaults clean.
`/tmp/fixed_image_work/ROLLOUT.md` stages the rest: 002 transport
(`docker save | zstd | ssh … k3s ctr import` — 002's registry is
loopback-only and lacks the base layers), the DGD image swap on CR
`topo-c1-dp2tp4-disagg-r20` (render from the in-tree
`deploy/disagg_pd_r20/render_dgd.sh`), the **N1 NUMA-pin snippet in the same
manifest edit** (`numactl --cpunodebind=1 --membind=1` on the decode
service; node-1 CPUs `56-111,168-223` verified on identical hardware), and
read-only post-roll checks (fix-in-mount-ns grep, Cpus_allowed_list,
long-context canary). **Applying it to the live 002 workload is the owner's
decision; nothing is deployed.** Note the staged image predates the
`31e0b5be7` HISA AOT fix (next section) — the **next full-source build**
must carry both.

**Invalidated claims** (zero-vs-zero comparisons are vacuously "equal"):

- `3e03d665d`'s indexer-proj backend claim "bit-identical (max|diff| == 0 at
  M ∈ {4,16})" is **VACUOUS for the indexer triple** — both sides were
  all-zero. The 1.18–1.30× timing stands (kernel time is value-independent).
  Post-fix re-proof under HEAD dynamic-quant semantics (`wkwp_driver2`):
  **wk and weights_proj re-proven** (cutlass↔cuBLASLt cos 1.000000 /
  0.999996); **wq_b re-proven PASS** (close-out B, 2026-06-10: bit-identical
  with liveness asserted — cos 1.000000, max|diff| 0 at M ∈ {4,16} × layers
  {5,30}; cuBLASLt 1.18–1.21×, stays the pick). The full indexer triple is
  re-proven. See B1.
- The `5bc2b2cb8` commit-message note "wkwp fused GEMM fails the cosine gate
  (cos = 0.000000), do not enable" — both sides were all-zero; cos(0,0) = 0.
  Root-caused and superseded by `68866e061` (default-on after the driver2
  PASS). Do not cite that commit-message claim.
- **Any output-quality / long-context-correctness observation through pre-fix
  images on this checkpoint is void** (every serve since the Graft checkpoint
  landed ran degenerate selection past 8192 tokens). Pure TPOT/throughput
  numbers stand mechanically (`index_topk` is fixed, so kernel work is
  value-independent) — the topology-sweep tok/s rankings survive as
  PERFORMANCE data but must not be cited as long-context correctness
  evidence. The planned A/B campaign runs on fixed code only.

**Lesson (extends the K2 lesson):** an equality gate is only as good as the
liveness of both sides — assert the outputs are nonzero/dense before
celebrating `max|diff| == 0`.

---

## CRITICAL — HISA decode selection corruption: pad-poisoning + logits-stride (FIXED `31e0b5be7`; AOT kernels ride the next image build)

Two live silent selection-quality bugs in the HISA NVFP4 decode candidate
path, both verified at HEAD on B200 with full gates (JIT twin kernels of the
verbatim vs patched `.cu`) and fixed in `31e0b5be7`. Artifacts: 001
`/tmp/hisa_pad_work/`. Neither crashes — both corrupt *which tokens the
selection returns*, every step they engage.

**RC1 — pad poisoning (mixed/heterogeneous bands).** Rows with
`ceil(prefix/128) < hisa_block_topk` get their `top_blocks` tail −1-padded
by `indexer_topk_decode`'s short-row path; the candidate-pages kernel clamps
−1 to logical page 0, so every pad slot re-scores the row's FIRST KV page —
attention sinks, high scores — under negative token identities; the mask
kernel only rejected `token >= prefix_len`, so the duplicate sink scores
DISPLACE real candidates in the candidate top-k (up to **987/1024 slots for
a 512-token row in a 66k band; 623–5277 poisoned selections/step in mixed
c16 bands**), and the remap kernel emits raw negatives into
`topk_indices_buffer`. Downstream convert maps negatives to −1 and sparse
MLA skips them — no crash, no OOB: short rows just attend FEWER and WRONG
tokens, every step, whenever the batch-max kv engages HISA (the gate keys on
**batch max** ≥ `hisa_min_seq_len`, default 32768; the eager min-prefix bail
neither protects mixed batches nor exists under graphs) and rows are
heterogeneous. These are exactly the `offpad` counts the H3b A/B flagged as
a pre-existing OFF-path artifact — now root-caused.

**RC2 — logits stride (every non-128-aligned prefix; uniform bands
included).** `fp8_fp4_paged_mqa_logits` returns a ROW-PADDED tensor (stride0
16640 vs 16512 cols); the mask wrapper indexed it FLAT, so every row > 0's
mask drifts by −128·row elements — wrong live slots −inf'd, true
out-of-range tokens left unmasked — for every non-128-aligned prefix, i.e.
every real production batch. (The H3b harness never saw it: its prefixes
were all 128-aligned, so the kernel never wrote.)

**Fix (minimal, the −1-sentinel contract):** the mask predicate rejects
`token < 0`; mask writes are stride-aware (`invoke` gains `scoreStride0`,
the wrapper passes `stride(0)`; torch schema unchanged); remap requires
`selectedOffset >= 0` and `token >= 0`, else −1 — which also kills the
`-1/128 == 0` truncation corner. The 3 non-AOT python fallback sites in
`dsa.py` mirror the same fixes bit-for-bit. The candidate-pages page-0 clamp
stays (memory safety; the scores are now masked).

**Gates (driver3, fixed image, GPU 6) — ALL PASS:** HEAD repro EXACT
(`offpad` 623/1473/5277 = the H3b counts; RC2 drift 122/840 cells); fix →
zero pad tokens / raw negatives / ≥prefix leaks; **every row set-equal to an
exact `torch.topk` reference on correctly-masked scores** (HEAD fails the
non-aligned bands, the fix matches); the existing off/on/onh equivalence
suite passes; timing FIX == HEAD within the 2.05 µs replay granularity.

**Deployment status:** the python fallbacks are live immediately; **the
`.cu`/`.h`/`.cpp` parts need the next full-source image build** — the staged
remediation image `optrt-34fe7aaec-fixed-20260611011615` predates this fix,
so the build floor is now **≥ `5bc2b2cb8` (input_scale) AND ≥ `31e0b5be7`
(HISA AOT)**.

**Prior-measurement caveat:** any HISA-path **selection-quality**
observation taken through pre-fix kernels is suspect — mixed/heterogeneous
bands via RC1, any non-128-aligned prefix via RC2. (The H3b set-equality
verdict survives only because its driver masked the pads in both arms on
aligned prefixes; its perf verdict is timing and unaffected.) Pure timing
numbers stand — the kernel work is value-independent.

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

### Where the GPU time goes (composite re-profile 2026-06-11 — the current ranking)

**Composite re-profile: eager c16 on the FIXED image
(`optrt-34fe7aaec-fixed-20260611011615` + the `8e44aeae1` overlay,
md5-verified), full shipped stack DEFAULT-ON (pure defaults — zero
`TRTLLM_OPTRT_*` env; only the prod companions `WARP_DECODE_FIXED_TACTIC=1`
+ `ENABLE_PDL=1`): 30.8 ms/step/GPU, −26.1 % vs the 2026-06-10 baseline
(41.7 ms), composition CLEAN** — full lifecycle rc=0; 14.4k-line log sweep
zero tracebacks / CUDA errors / OOM / NaN; capture window verified true c16
ADP steady state; two same-image legs reproduce throughput to +0.004 %.
Classifier coverage 99.99 %.

| Stage | µs/step/GPU | share | vs the 06-10 baseline |
|---|---|---|---|
| MoE | 22 965 | **74.6 %** | −8.2 % |
| — of which a2a (dispatch 11 042 + combine 3 848) | 14 889 | **48.4 %** | eager-exposed spin/skew |
| — of which expert-GEMM chain (expert 3 953 + fc1 2 081 + fc2 1 129) | 7 163 | **23.3 %** | flat |
| dense-MLA proj | 3 089 | 10.0 % | **−62.5 %** |
| norm / rope / quant | 1 672 | 5.4 % | −2.7 % |
| elementwise / glue | 1 642 | 5.3 % | **−58.1 %** |
| sparse-MLA attention | 784 | 2.5 % | −0.4 % (control: flat) |
| Indexer | 388 | **1.3 %** | **−76.7 %** |
| HISA | 227 | 0.7 % | −20.0 % |

Per-stage deltas map cleanly to commits (artifacts: 001
`/tmp/rerank_profile_work/RERANK_REPORT.md` + `compare_rerank.txt`; raw leg
`/tmp/nsys_decode_work/out_RERANK_C16/`):

- **dense-MLA proj −5 150 µs**: B1 cuBLASLt force (`4220bf4bd`+`3e03d665d`
  — the pathological f32 `gemmSN_TN` −4 953 µs is GONE) + I5 wk+wp fused
  GEMM (`68866e061`).
- **glue −2 279 µs net**: G1 + the CuTe gate (`3e03d665d`+`33e801fd1`,
  +756 µs NEW kernel replacing more), gate+quant chain (`68866e061`), L2
  read-set hoist + S5 (`5bc2b2cb8`), C4 KVarN host-path (`0a1504755`,
  direct_copy −925 µs).
- **Indexer −1 276 µs (−76.7 %)**: I7 C++ top-k (`841f9874a`, select_cub
  −1 235 µs = −99.2 %) + I6 fp16 logits (`3e03d665d`) **minus +280 µs of
  NEW real work** from the input_scale remediation (+146 µs dynamic-amax
  chain + +134 µs real wk+wp GEMM — the baseline fed zeros).
- **MoE −2 056 µs**: mostly notify_dispatch −1 630 µs (a barrier-spin
  kernel shrinking second-order as the rest of the step gets leaner) + K3
  swiglu+fp4out (`fd705a6f5`, moe_activation −178 µs); fc2/expert-GEMM flat
  (`e105fd7a1` FC2-N=256 default == the baseline's env-forced config).
- **norm/rope/quant −46 µs**: `68866e061` + `fd705a6f5` + the `8e44aeae1`
  dense-MLP GATED_PREMLP_QUANT handoff (−7.7 µs vs the isolation leg —
  matches the ~6–8 µs/tok ship estimate).
- Controls flat (fmha −0.1 %, rmsnorm −0.1 %, routing +0.9 %, sampler
  +0.0 %) — cross-leg comparability is sound. `34fe7aaec` is
  correctness-only (no stage delta); the `8e44aeae1` short-band restore has
  **zero eager effect** by design (the eager skip keys on live kv) — its
  win lands on graphed prod kv ≤ 1024 traffic.

> **Indexer callout (the directive predicted the share would RISE — it
> FELL, 4.0 % → 1.3 %):** the input_scale fix's real-output cost is now
> visible and quantified (**+146 µs/step** amax chain + **+134 µs/step**
> real wk+wp GEMM, both zero at the all-zero baseline), but the
> `841f9874a` top-k rebuild simultaneously deleted an order of magnitude
> more. Operating-point caveat: this dataset is 128in/512out (kv ≤ 640 ≤
> `index_topk`=1024), so the MQA/top-k skip is active in BOTH columns —
> the predicted input_scale-driven indexer-share rise materializes only on
> **long-kv traffic**; a kv ≫ 1024 profiling leg is the follow-up that
> would show it.

> **A2A measurement caveat:** dispatch/combine are dominated by notify/spin
> kernels (notify_dispatch 9.5 ms) — skew, not payload — and inflate under
> eager launch jitter (this stage alone moved ±15 % between two same-image
> legs while every compute stage reproduced < 1 %). **Re-measure the
> exposed comm under graphs + overlap before sizing further a2a work**, and
> the LL flip is REQUIRED to even engage it (the factory never auto-picks
> LL — see M3).

Lineage: the campaign's original "Indexer is 50–74 % of TPOT" premise
described the pre-campaign state. The first eager c16 profile (2026-06)
measured MoE ≈ 60 % / dense proj 19.8 % / glue 13.5 % / Indexer ≈ 4 % — of
which an "indexer FSSS cub select" slice (~3 %) was a **MISATTRIBUTION**
(KVarN restore host-path compaction — 671 ATen cub kernels + 183 D2H syncs
per step-rank selecting an EMPTY set — fixed as C4 in `0a1504755`). The
composite re-profile above is the current source of truth; the ranked
levers follow it: MoE a2a (M3 flip + graphed re-measure), the expert-GEMM
chain (P1 megakernel, phase-1 in flight), the dense-proj residual (B2, in
flight).

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

> **M3 ⇄ topology interaction (decision-relevant, 2026-06-11).** EP a2a —
> **any** strategy, including the DeepEP-LL flip now decided GO (M3) —
> engages **only under attention-DP with `moe_tp_size=1`**: with plain-TP
> attention the MoE comm factory returns **no strategy at all**
> (`tensorrt_llm/_torch/modules/fused_moe/communication/`
> `communication_factory.py:126` — `(not enable_attention_dp) or dp_size==1
> ⇒ None`; `moe_tp_size != 1 ⇒ AllGather/ReduceScatter`, no a2a). So **ADP
> unlocks the measured 2.4–2.5× a2a win; pure TP has no a2a path to
> optimize and forfeits M3 entirely.** The WarpDecode+TP case (faster
> attention at equal load, no ADP host collectives, S2 moot) must now be
> weighed against losing M3 — the ADP-vs-TP production call owns this
> trade. Also recorded in [topology_deploy.md](topology_deploy.md).

### Correctness gating (untrained checkpoint; the 0.98 functional bar)

The graft checkpoint emits garbage-class text and MoE routing is run-to-run
non-deterministic, so **end-to-end text parity is meaningless**. Every
correctness-affecting change is gated **kernel-level against a TRUE
reference** — never the candidate's own fused-vs-sequential output (the K2
lesson), and never an equality whose two sides could both be degenerate (the
input_scale lesson above).

**The bar (owner directive 2026-06-10): downstream functional correctness,
cosine ≥ 0.98. Bit-identical is never required.** This supersedes the earlier
"cosine ≥ 0.9999" preamble for functional gates. Two qualifications:

- **Format mandate:** quantization precision/formats are pinned to the
  checkpoint (`NVFP4-W4A4KV4-IndexerK4` + `HISA4to1`); FP/BF16 logits are
  acceptable. A lever that changes a checkpoint-pinned format is dead
  regardless of its cosine (see H4 in the killed list).
- **State-machine / cache-content gates are NOT cosine gates** and are
  unaffected: KVarN restore correctness is still judged by equivalence of the
  restored pool vs the reference universe (a cosine there would mask restore
  bugs).

Top-k selection changes are judged by their **downstream effect** (attention-
output cosine of the candidate-selected set vs the reference-selected set):
boundary swaps among near-tied tail tokens carry ~zero softmax mass, so set
IoU systematically understates functional agreement. See the 0.98-bar
re-screen section for what this revived and what stays dead.

---

## Production target — all custom pieces ENABLED and maximally optimized

The production deployment runs the full custom stack **on**, each in its most
optimal mode — optimization means making each piece *engage intelligently*, not
disabling it:

| Piece | State | Optimal-mode note |
|-------|-------|-------------------|
| LayerSplit (prefill) | on, CP2×TP2, owner-local, read-set broadcast + **dense-broadcast overlap default-on (L1, `0a1504755`)** | exposed broadcast 65–82 → 2.5–8.2 ms/step, TTFT −544 ms @64k; C9 IPC push stays **parked** — the IPC channel wedges under the overlap pattern, overlap traffic is NCCL-only |
| WarpDecode (decode) | on, forced `decode_1cta`, fixed tactic | persistent-megakernel is the structural ceiling; megakernel FC2 N-tile default is now 256 (160 numerically broken, cycle 5) |
| dense KVarN `kvarn_k2v2` | on, amortized + **delta** restore (C2 default-on `5bc2b2cb8`) + **eager decode-restore host-path (C4, `0a1504755`)** | C1 host-gate retained as the no-change fast path; stale-record-on-recycle invalidation default-on (`34fe7aaec`); eager restore 19.4× via host-mirror selection |
| Indexer IndexCache + FSSS | on, `index_topk_freq=4` | escalation to 8 under recall gate |
| **HISA** | **on whenever the Indexer is on; capture gate + candidate width track live kv via `metadata.max_gen_kv_len` (cycle 4); engages at kv ≥ `hisa_min_seq_len` (default 32768)** | **never slower than off, wins whenever active (forced-on gate=1024 proven never-slower, up to 2.56× at 33k); per-step invariant memo shipped `3e03d665d`; pad-poisoning + logits-stride selection bugs FIXED `31e0b5be7`** (python fallbacks live; AOT kernels ride the next image build — see the hazard section) |
| NVFP4 indexer-K (MX E2M1+UE8M0) | on | score→top-k fusion measured net-zero under graphs — killed |
| Indexer decode top-k | on — prod live-kv routes to vanilla C++ (`841f9874a`), DSL only at kv ≥ 16k; **short-band auto-default restored to `index_topk` (`8e44aeae1`)** — kv ≤ 1024 graphs capture indexer-FREE again; dead `_DSL_TOPK_MIN_COLS` removed | fp16 logits (`indexer_logits_dtype=auto`→fp16 on the DSL path, `3e03d665d`) |
| MLA / MLP / indexer proj GEMM backend | on — cuBLASLt forced for the NVFP4 proj Linears (`TRTLLM_MLA_PROJ_NVFP4_BACKENDS` + `TRTLLM_DSV3_MLP_NVFP4_BACKENDS` + `TRTLLM_INDEXER_NVFP4_BACKENDS`, default `cublaslt`) | 1.2–2.1× per GEMM (B1, `4220bf4bd`+`3e03d665d`); bit-identical on the MLA/MLP set — the indexer-triple equality was vacuous pre-input_scale-fix; full triple re-proven post-fix (wk/wp `wkwp_driver2`, wq_b close-out B; CRITICAL section) |
| Indexer wk+weights_proj fused GEMM | on (`68866e061`, `TRTLLM_INDEXER_FUSE_WK_WP=1` default) | 1.96–1.97× → **~2.0 ms/token**; also halves the dynamic amax+quantize work (I5) |
| Gated-norm / glue | on — `fused_lowrank_gate` with the CuTe DSL kernel as default impl (`TRTLLM_OPTRT_LOWRANK_GATE_IMPL=cute`, `33e801fd1`) + fused sigmoid·mul at both attention gate sites + **single-launch gate+NVFP4-quant on the MoE input** (`68866e061`) + **dense-MLP (layers 0–2) gate→quant handoff, swizzled-SF epilogue** (`TRTLLM_OPTRT_GATED_PREMLP_QUANT`, `8e44aeae1`) | −91.6 % on the gated-norm chain (G1); the quant epilogue takes the chain 7.04 → 4.19 µs/layer (−165 µs/tok, G2 chain); dense-MLP handoff 4→3 kernels, ~6–8 µs/tok, bit-exact; absorb **CLOSED PERMANENTLY** (MoE-output cosine FAIL, see G2) |
| Shared-expert swiglu+FP4-out fusion | on at decode M — `_FP4OUT_MIN_M=128` guard lifted (`fd705a6f5`) | exact vs TRUE-f32 (cos 1.0, max_abs 0.0) at every m |
| MoE EP comm | NVLINK_TWO_SIDED today; DeepEP low-latency **DECIDED GO** (M3, 2026-06-11) — env delta staged | `TRTLLM_FORCE_COMM_METHOD=DEEPEPLOWLATENCY` + **explicit `TRTLLM_DEEP_EP_TOKEN_LIMIT=64`** (the "inversion at 64" was an artifact — see M3); **topology precondition: ADP + `moe_tp_size=1`, inert under plain TP** |
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

1. **Remediate the live input_scale hazard** — the fixed image is **built +
   verified** (`optrt-34fe7aaec-fixed-20260611011615`, 4/4 incl. the NONZERO
   indexer probe; CRITICAL section); `ROLLOUT.md` stages 002 transport, the
   DGD image swap, the N1 NUMA pin, and post-roll checks. **Rollout is the
   owner's decision** (live workload). Every future build must carry
   ≥ `5bc2b2cb8` **and ≥ `31e0b5be7`** (the HISA AOT fix — second hazard
   section).
2. **MoE a2a — the top lever by the composite re-profile (48.4 % of the
   eager step)**: apply the staged M3 LL env delta
   (`/tmp/m3_sizing_work/production_env_delta.yaml`:
   `TRTLLM_FORCE_COMM_METHOD=DEEPEPLOWLATENCY` + explicit
   `TRTLLM_DEEP_EP_TOKEN_LIMIT=64`) — **REQUIRED, the comm factory never
   auto-picks LL at defaults**; topology precondition ADP + `moe_tp_size=1`
   (`communication_factory.py:126`, feeds the ADP-vs-TP call) — and
   **re-measure the exposed comm under graphs + overlap** (the eager spin
   kernels inflate ±15 % leg-to-leg; every compute stage reproduces < 1 %).
3. **Expert-GEMM chain (7.2 ms/step, 23.3 % of the re-profile)** → the
   single-CTA persistent megakernel — **phase-1 in flight** (P1).
4. **Dense-proj residual (B2, in flight)** — `nvjet_tst_128x8` 38 µs/call
   × 61/step is latency-bound, not FLOP-bound; cross-layer / per-layer
   batching.
5. The remainder: I2 GATE-B real-activation dump (unblocked by the fixed
   image); **MO1 is DEPLOY-READY** (release `.so` built + the activation
   gap closed + runbook staged; rollout = owner's call); the 0.98-bar
   re-screen stays settled (G2-absorb closed permanently, H3b dead, FP4MQA
   + wq_b close-outs passed); then N1 (rides the rollout manifest edit),
   M1, K1 PDL, H3a/H3c. C9 stays **parked** (the IPC channel wedges under
   the L1 overlap pattern); S2 held (moot under WarpDecode+TP). **Shipped
   this round: `8e44aeae1` (I3 closed dead + short-band default restore +
   dense-MLP gate+quant handoff) + `31e0b5be7` (the HISA pad-poisoning /
   logits-stride correctness fix).**

## Ranked candidates

| # | Candidate | Layer | Expected win @ c16 | Status |
|---|-----------|-------|--------------------|--------|
| **M3** | **DeepEP low-latency production flip** (`TRTLLM_FORCE_COMM_METHOD=DEEPEPLOWLATENCY` + explicit `TRTLLM_DEEP_EP_TOKEN_LIMIT=64`) | comm | LL wins at **every** point: 2.4–2.5× at steady c16, 2.06× worst case (actual=64); the prior "INVERTS at limit 64" was an **emulated-FFN artifact, REFUTED**; re-profile: a2a is **48.4 % of the eager step** and the env delta is **REQUIRED** (the factory never auto-picks LL) | **DECIDED GO** (2026-06-11, sizing sweep) — env delta staged; **requires ADP + `moe_tp_size=1`** (`communication_factory.py:126`); post-flip: re-measure exposed comm under graphs+overlap |
| C2 | KVarN delta-restore (restore only the changed rows/blocks) | scheduler | host 48.8 → 0.3 ms/fire (147–162×); **12.1 → 0.08 ms/step amortized at TP bs=16** | **SHIPPED default-on** `5bc2b2cb8` (5-scenario lockstep equivalence PASS) |
| G2 | Gated-norm → PRE_MOE_FUSION (chain quant-epilogue + absorb) | glue | chain: −165 µs/tok shipped; absorb floor: ~117 µs/tok — **irreducible** | **chain SHIPPED** `68866e061`; **absorb CLOSED PERMANENTLY** (2026-06-11) — MoE-output cosine FAILs every cell at the 0.98 bar (see G2) |
| I5 | Indexer wk+wp fused GEMM (one launch + one read of x per F-layer) | indexer | fused 23.5 vs split 46.5 µs @ M=4 (1.96×) → **~2.0 ms/token** | **SHIPPED default-on** `68866e061` (the v1 cos=0.0 gate-fail was the input_scale bug, not the fusion) |
| C4 | KVarN eager decode-restore host-path (`_kvarn_step_cand_host` host-mirror selection) | scheduler | 13.3 → 0.69 ms/61-layer step (19.4×) steady, 17.0 → 1.7 ms churn; kills the misattributed "indexer FSSS cub select ~3 %" profile slice | **SHIPPED default-on** `0a1504755` (300-trial set-equality, 0 failures) |
| C9 | CP=2 IPC push broadcast | prefill TTFT | 1.2–3× the per-layer broadcast | impl; **PARKED** — the IPC channel wedges both ranks under the L1 overlap pattern; L1 ships the win NCCL-only |
| B1 | cuBLASLt NVFP4 backend force: MLA proj + shared/dense MLP + indexer proj | GEMM | ~1.62 ms/tok (`4220bf4bd`) + 1.62 ms/tok incremental (`3e03d665d`, TP4); bit-identical | **SHIPPED** `4220bf4bd`+`3e03d665d`; re-profile: dense-proj bucket **−62.5 %** |
| **B2** | **Dense-proj residual batching** (cross-layer / per-layer consolidation of the M=4 proj GEMMs; CuTe persistent variant / absorption are the alternates) | GEMM | re-profile: `nvjet_tst_128x8` **38 µs/call × 61/step = 2.33 ms** — tiny-M latency-bound, not FLOP-bound (same attack covers the 269 µs once-per-step 128×8 tail) | **IN FLIGHT** (re-profile target #3) |
| G1 | Gated-norm + glue fusions (`fused_lowrank_gate` −91.6 %, fused sigmoid·mul, HISA invariant memo) + CuTe DSL port (−36 % vs Triton, default impl) | glue | −6.27 (bs4) / −9.38 (bs16) ms/step eager GPU | **SHIPPED** `3e03d665d`+`33e801fd1` |
| I6 | fp16 indexer logits (`indexer_logits_dtype`, auto→fp16 on the DSL path) | indexer | top-k −15…−22 % @ kv ≥ 33k; logits buffer halved | **SHIPPED** `3e03d665d` |
| I7 | Prod decode top-k → vanilla C++ (drop the stale width-override) | indexer | ~1.7× top-k @ prod live-kv (~29–33 % of the top-k pipeline) | **SHIPPED** `841f9874a` |
| K3 | Shared-expert swiglu+FP4-out fusion at decode M (`_FP4OUT_MIN_M` lift) | kernel | ~100 µs/step + 58 act-quant launches removed | **SHIPPED** `fd705a6f5` |
| C1 | KVarN pre-replay restore host-gate | scheduler | ~0.75–2 ms host (within harness noise) | **shipped** `a1b13ea78` |
| C3 | Cache debug env-gates / no eager kwargs | scheduler | ~0.1 ms/step | **shipped** `0adc87009` |
| H3a | PDL-chain the 5 HISA glue kernels (candidate_pages/mask/remap/block_reps/block_scores; indexerHisaNvfp4.cu has 0 PDL, indexerTopK.cu has 9) | indexer | ~0.11–0.17 ms/step (7 boundaries × ~1–1.5µs × 16F) | candidate (re-scoped: HISA ≈ 0.7 % of step) |
| H3b | Per-row live-length candidate scaling (caller-only: per-row `candidate_context_lens`/`selected_lengths` from `prefix_lens`; kernels already walk `[0,num_kv)` — verified fp4_paged_mqa_logits.py:1422 / indexerTopK.cu:663) | indexer | long-band A/B (driver v2): set-equality OK but **plain ON is SLOWER everywhere** (+1.2–4.1 µs full pipeline); hoisted variant breakeven (best +2.3 µs) | **DEAD again** (2026-06-11) — the HISA-scale fix already captured the width win; stays opt-in/off (`/tmp/h3b_ab_work/h3b_table_v2.txt`) |
| H3c | Incremental block-rep quantize (only the boundary block changes/step; indexerHisaNvfp4.cu:279 rebuilds all) | indexer | ~0.08–0.11 ms/step | candidate |
| ~~H4~~ | ~~HISA `compression_ratio` sweep {4,6,8,12}~~ | indexer | — | **KILLED by format mandate** — the checkpoint pins HISA4to1; only intra-4to1 tuning remains |
| ~~H1~~ | ~~short-band candidate-width allocation shrink~~ | indexer | **REGRESSED −6% (40.28→37.82)** — short kv is latency-bound | **discarded (measured)** |
| S2 | Collapse 10 host MPI collectives → ~3 | scheduler | 150–400 µs + 7 barriers of jitter | **MOOT under WarpDecode+TP** (no ADP collectives); held with that note |
| N1 | NUMA-pin decode workers to node 1 | system | 0.3–1 ms + jitter | planned (rides manifest) |
| I2 | `index_topk_freq` 4→8 (16 → 9 F-layers) | indexer | 0.23–0.46 ms/step measured ceiling | synthetic recall gate proven **non-predictive**; conditional GO on GATE-B real-activation dump — **unblocked**: the fixed image is staged (`optrt-34fe7aaec-fixed-…`, CRITICAL section), dump runnable on 001 |
| M1 | MoE A2A two-sided → one-sided + workspace combine | comm | 0.3–0.9 ms/step | planned (sequence after M3) |
| K1 | PDL coverage completion | kernel | +1–3% | planned |
| ~~K2~~ | ~~FC2 N-tile 256→160~~ | kernel | **N=160 is numerically broken** (SFB miscompute, cos 0.790 vs TRUE f32; + prefill M=1024 OOB) | **KILLED** `29f492b49`→`d01737397` — see killed list |
| L1 | z.ai dense-broadcast overlap | prefill | exposed broadcast 65–82 → 2.5–8.2 ms/step; **TTFT −544 ms @64k, ~−1.1 s @128k** | **SHIPPED default-on** `0a1504755` (20/20 correctness, both ranks, CP2 + real fp8_fp4 scoring) |
| P1 | Persistent decode-layer megakernel | kernel | dispatch fusion itself Δ≈0 under PDL (the −13.9% previously attributed to it was K2's broken N=160); structural persistent-kernel case unchanged; re-profile: the expert-GEMM chain is **7.2 ms/step (23.3 %)** | **code shipped, opt-in** (`MOE_MEGAKERNEL`); FC2_N default fixed 160→256 `e105fd7a1`; **phase-1 persistent worker grid IN FLIGHT** (re-profile target #2) |
| MO1 | MORI-style generation-first / write-mode handoff | transport | TTFT ~−12–17 ms @8k / ~−20–27 ms @64k intra-node fp8 (estimate) | **DEPLOY-READY** — release `_core.abi3.so` built in the target image (cargo test 18/18 `prefill_router`; GPU-7 smoke 39/39 incl. 5 gen-first tests); python activation gap **closed** (legacy-path port, +113 lines, ZERO manifest delta); `INSTALL_RUNBOOK.md` staged; rollout = owner's call (see MO1) |

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

## C2 — KVarN delta-restore (SHIPPED default-on `5bc2b2cb8`)

C1 skips the pre-replay restore scan when *nothing* changed; when something
*did* change, the 61-layer masked-select scan walk still re-derived far more
than the delta. Delta-restore (`TRTLLM_OPTRT_KVARN_DELTA_RESTORE=1`, default)
replaces it with an **O(batch) host-integer delta** — blocks newly full +
newly onboarded, filtered by the pools' host mirrors; **any uncertainty
returns False WITHOUT mutation** and falls through to the full scan.

- **Equivalence (state-machine gate, not cosine):** 5-scenario lockstep vs
  the full scan — no-op key, boundary crossing, multi-request churn,
  onboard/free/rewind/recycle, fallback-decline — **all PASS** (non-buffer
  state byte-exact; restored buffers reference-exact 0 ulp where the full
  scan itself wanders 3–9 ulp from batched-dequant nondeterminism).
- **Adversarial probes favored the delta**: the full scan CLOBBERS a
  recycled-id owner with the stale record — a pre-existing bug in the
  *fallback* path, fixed separately in `34fe7aaec` (next).
- **Win:** host cost/fire 48.8 → 0.3 ms (162× on the common uncommitted
  fire); amortized at TP bs=16 **12.1 → 0.08 ms/step** — the TP-regime cost
  this lever was ranked on is gone. See [kvarn.md](kvarn.md).

### C2b — KVarN stale-record-on-recycle fix (SHIPPED default-on `34fe7aaec`)

The adversarial probe above exposed a real production bug:
`KVarNLatentPool.valid` was set by `store_block` and **never cleared when the
paged KV block was freed and recycled** to a new request.
`kvarn_commit_full_blocks` skips valid blocks, so a recycled id kept the OLD
owner's packed record and was never re-committed; the full-scan restore path
(`keep = pool.valid[cand]`, no epoch check) then dequantized the OLD owner's
record **over the NEW owner's fresh fp16 latent** — stale-KV poisoning on
every restore fire for that block (harness scenario d3: restored content =
the old record, cos −0.02 vs the new owner's data). The delta path was immune
via its host-mirror epoch filter; the full scan is the fallback and the eager
default.

Fix (default-ON, `TRTLLM_KVARN_INVALIDATE_ON_FREE=0` escape hatch):
`KVarNLatentPool.invalidate_blocks()` clears `valid`/`valid_host` and resets
`restored_gen_host`; `DSACacheManager.free_resources` (ids snapshotted before
the C++ free) and `rewind_kv_cache` (freed blocks + the new tail block) fan
out to all local layer pools with one shared device-id tensor — restoring the
documented "re-committed when block-id is recycled" contract that no code
path actually honored. Gates (B200): d3 fixed — gate-ON restore is
**bit-equivalent to a never-recycled universe (cos 1.0) on all 4 restore
paths**; the full a–e scenario suite passes on BOTH delta and full-scan paths
(worst cos 0.99964 = the k2v2 quant floor); 7/7 unit tests. Cost: the decode
walk is byte-identical (no per-token change); 1.46 ms once per request-free
at the full 61-pool fan-out.

## C3 — Cache debug env-gates (SHIPPED)

`TRTLLM_OPTRT_KV_DEBUG` / `TRTLLM_OPTRT_MODEL_ENGINE_ADP_DEBUG` were re-read from
the environment every call, and 50+ call sites (several per-step) evaluated
their kwargs eagerly (a scheduled-request-id list comprehension every iteration;
per-step request/metadata summary strings). Cache the gates at import,
short-circuit the summary helpers, gate the per-iteration executor call at the
call site. **Shipped** `0adc87009`. ~0.1 ms/step, zero correctness risk.

## C4 — KVarN eager decode-restore host-path (SHIPPED default-on `0a1504755`)

The eager decode profile's "indexer FSSS cub select" line (~3 % of decode
GPU) was a **misattribution**: `kvarn_restore_for_decode`'s boolean-mask
indexing launched **671 ATen cub kernels + 183 D2H syncs per step-rank**
(61 layers × 3 `torch.nonzero`) — ~1.24 ms/step of GPU compaction that
selects an **EMPTY set** in steady state. Fix: host-mirror selection
(`_kvarn_step_cand_host`): candidates from `kvarn_host_block_table` + host
kv_lens, memoized on the step key; numpy filter via the same
`valid`/`commit_gen` host mirrors the C2 delta walk trusts. Empty set ⇒
zero launches/syncs; non-empty ⇒ the existing restore primitive; **any
host-state inconsistency ⇒ unchanged device-scan fallback**. Set-equality
vs an exact HEAD-path replica: **300 randomized trials (B 1/4/16, churn,
duplicates, recycled epochs, padding) = 0 failures**. Standalone 61-layer
step: **13.3 → 0.69 ms (19.4×) steady, 17.0 → 1.7 ms churn**. See
[kvarn.md](kvarn.md); the profile correction is in the methodology section
above.

## C9 — CP=2 IPC push broadcast (impl, PARKED — IPC wedges under the L1 overlap)

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
symmetric on both ranks. Local commit `8bcda0809`, held.

**Status update (`0a1504755`): PARKED.** During the L1 overlap work the C9
IPC channel **wedged both ranks** under the overlap traffic pattern
(comm-stream broadcast issued behind indexer compute) — the L1 win ships
**NCCL-only** and C9 stays parked. Revisit only with a root-cause for the
wedge; L1 already removes most of the exposed cost C9 targeted (the sync
path's cost was the masked_select+unique host syncs, not wire time — see
L1).

---

## B1 — cuBLASLt NVFP4 backend force (SHIPPED `4220bf4bd` + `3e03d665d`)

The largest real-compute decode bucket after MoE: dense proj GEMMs were
19.8 % of the first eager c16 GPU profile (10.0 % after this ship — the
composite re-profile measures the bucket **−62.5 %**, with the pathological
f32 `gemmSN_TN` −4.95 ms/step GONE from the timeline), and Linear's NVFP4
auto-selection
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

> **Post-audit correction (2026-06-10):** the `3e03d665d` indexer-triple
> equality ("all bit-identical, max|diff| == 0") was **VACUOUS** — both sides
> were all-zero through the pre-`5bc2b2cb8` input_scale bug (CRITICAL
> section). The 1.18–1.30× timing stands. Re-proof under HEAD dynamic-quant
> semantics (`wkwp_driver2`): **wk and weights_proj re-proven**
> (cutlass↔cuBLASLt cos 1.000000 / 0.999996); **wq_b re-proven** (close-out
> B, 2026-06-10: cos 1.000000 / max|diff| 0 with liveness asserted,
> M ∈ {4,16} × layers {5,30}; cuBLASLt 1.18–1.21× — remains the right wq_b
> pick; 001 `/tmp/fp4mqa_closeout_work/closeout_b.json`; **v2 re-gate adds a
> pure-torch true-f32 dequant reference: min cos vs f32 ref 0.999999, both
> backends identical** — `closeout_b_v2.json`). The full indexer
> triple is re-proven. The MLA proj (`4220bf4bd`) and MLP claims are
> unaffected — those modules have in-checkpoint `input_global_scale` and were
> gated against a true-f32 reference.

The bf16 GEMMs were audited in the same pass: lm_head / gate_proj / router are
already optimally dispatched — no lever there. The companion accuracy verdict
on the dense MLA proj W4A4 path ("fails accuracy, cos 0.63–0.83") has since
been **REVERSED** — the reference was corrupted; corrected cosines are 0.995+
everywhere (see the record-corrected list).

### B2 — Dense-proj residual batching (IN FLIGHT; re-profile target #3)

Post-B1 the dense-MLA proj bucket is 3.09 ms/step (10.0 %) and its floor is
structural, not a backend mispick: `nvjet_tst_128x8` costs **38 µs/call ×
61 calls/step = 2.33 ms** — a tiny-M=4 NVFP4 GEMM paying 38 µs/call is
**latency-bound, not FLOP-bound** (the once-per-step 269 µs 128×8 tail
kernel has the same shape problem). Attack: cross-layer / per-layer
batching of the small proj GEMMs into fewer launches (alternates: a CuTe
persistent variant, or absorption into adjacent kernels). In flight.

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

## G2 — Gated-norm → PRE_MOE_FUSION (chain SHIPPED `68866e061`; absorb CLOSED PERMANENTLY)

Of the two designs, **the chain landed and the absorb is closed for good —
it failed the decisive MoE-output-cosine gate in every cell**:

- **Chain — single-launch CuTe lowrank-gate + NVFP4-quant epilogue
  (`apply_fused_lowrank_gate_quant_nvfp4`, SHIPPED `68866e061`):** the
  gated-norm output now quantizes to a LINEAR-SF `Fp4QuantizedTensor`
  **inside the gate kernel itself**, replacing gate + separate
  `fp4_quantize` on the routed-MoE input path (the prior in-tree handoff was
  a 2-launch Triton pair). Chain cost 7.04 → 4.19 µs (M=4) / 7.17 → 4.48 µs
  (M=16): **−165 µs/token at M=4 across 58 MoE layers** (−254 µs/token vs the
  pre-lowrank-gate unfused chain). Gates: y cosine vs true reference
  ≥ 0.999997; fp4 codes + scales bit-exact vs `trtllm.fp4_quantize` on the
  same y; top-8 routing overlap 1.0 vs the current path (the fp4-vs-bf16
  quantization noise floor is shared with the existing production path, not
  added by this kernel); CUDA-graph capture + replay bit-exact. Full
  absorb-into-AR-quant is additionally blocked by kernel layout: the AR NVFP4
  epilogues emit SWIZZLED SF only while the MoE permute path requires LINEAR.
  Documented as [nvfp4_fusions.md](nvfp4_fusions.md) #13c.
- **Absorb (constant-0.5 fold) — CLOSED PERMANENTLY (2026-06-11, decisive
  MoE-output-cosine gate at the 0.98 bar):** the routing rejection above
  ("0.500 ± 0.0025" is layer-5-local; top-8 overlap 0.965–0.988 at layers
  20–60) was re-screened under the new bar with the **functional judge** —
  per-token MoE OUTPUT cosine through the real production expert chain
  (`run_moe_nvfp4_impl`, real checkpoint weights, EP=1 full combine, layers
  {5,20,45} × M {4,16}, 3–4k tokens/cell; instrument determinism floor
  ≥ 0.999994). **FAIL in every cell, under every reading of the bar**: mins
  0.60–0.80, p1 0.77–0.96, and at layer 45 even the MEAN fails (0.9695 <
  0.98). Two independent mechanisms, both real:
  1. **Routing flips are NOT benign** — the f32-functional control (no
     activation quant anywhere; pure gate-modulation + routing difference)
     refutes the near-tied-flip hypothesis: **one flipped expert costs
     ~3–6 % output cosine** functionally, two cost ~6–12 %, and the
     any-expert flip rate grows with depth 4.5 % → 8.2 % → **18.9 % at
     L45**. Near-tied ROUTER SCORES do not imply near-identical EXPERT
     OUTPUTS.
  2. **fp4 re-quantization decorrelation amplified by routed/shared
     cancellation** — present even at 8/8 routing agreement: the gate-vs-0.5
     input delta crosses NVFP4 bin boundaries (rel_l2 0.10–0.22 through
     both expert paths), and tokens where routed+shared partially cancel
     lose up to 25 % cosine with PERFECT routing.
  The shipped single-launch gate+quant chain kernel (`68866e061`) **stays**;
  the remaining **~117 µs/token to the absorb floor is irreducible**. Do not
  re-open without a fundamentally different design (not a constant fold).
  Artifacts: 001 `/tmp/g2_absorb_work/` (`CLOSEOUT.md`, `out_g2_full/`);
  patch stays recorded at `/tmp/premoe_work/absorb_premoe.REJECTED.patch`.

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

### S5 — Per-iter stats/perf-metrics decimation (SHIPPED `5bc2b2cb8`)
`TRTLLM_OPTRT_STATS_DECIMATE=16`: the full IterationStats / per-request
metrics / CUDA-timing-event triplet is sampled 1/16; first + finishing
per-request calls stay exact (TTFT/E2E preserved); sampling derives only from
`iter_counter` so all ranks agree (collective payloads are fixed-width either
way). 29/29 host tests incl. a 4-rank lockstep sim + a negative desync test.
Stats block 48.2 → 16.1 µs/iter, p99 69 → 35. **Honest accounting: the
exposed TPOT win is ~0 at c16** — the host work was overlapped; this removes
a jitter source between lockstep collectives. The earlier "100–250 µs/step"
premise here was stale.

---

## Indexer

The Indexer is **1.3 %** of the composite re-profile (−76.7 % vs the 06-10
baseline; HISA ≈ 0.7 %) — and that 1.3 % already includes the ~280 µs/step
of real work the input_scale remediation restored. The remaining levers here
are small and ranked accordingly.

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

### I5 — Indexer wk+wp fused GEMM (SHIPPED default-on `68866e061`)
Fuses the indexer `wk` and `weights_proj` GEMMs (same input x) into one NVFP4
GEMM launch + one read of x per F-layer (`TRTLLM_INDEXER_FUSE_WK_WP=1`,
default). The earlier "fails the cosine gate (cos = 0.000000), do not enable"
verdict was **the input_scale hazard, not the fusion**: the v1 driver
compared all-zero against all-zero through the pre-`5bc2b2cb8`
uninitialized-input_scale bug — cos(0,0) = 0 (CRITICAL section). Under HEAD
dynamic-quant semantics the gate is **cos(indexer_k) = 1.000000
(bit-identical)** and **cos(weights) = 0.999996** (1 bf16 ulp from the wp
out-scale fold; f32 in production) at M ∈ {4,16}, cutlass cross-backend
identical. Timing under CUDA-graph replay: M=4 fused 23.5 µs vs split 46.5,
M=16 34.6 vs 67.6 (**1.96–1.97×**) → **~2.0 ms/token across 61 layers** —
bigger than the static estimate because the fusion also halves the dynamic
amax+quantize work.

### I2 — `index_topk_freq` 4→8 (synthetic gate NON-PREDICTIVE; conditional GO — GATE-B dump now unblocked by the fixed image)
FSSS is **cross-layer** (not cross-step); freq=8 means **9** F-layers (not 8)
vs prod freq=4's 16, so the lever is 7 F-layers/step. Measured (B200, real
per-layer NVFP4 indexer weights):

- **The planned synthetic recall gate is non-predictive.** 8 configs × 3
  seeds: cross-layer top-1024 recall == **chance at every stride**
  (0.222 @ kv 4.6k = 1024/4608, 0.031 @ kv 33k), flat in F-distance 1..7 —
  including the freq=4 production setting. Harness sanity passed (dequant
  max|diff| 3e-8 vs the reference op; self-recall 0.99 at 1 % input noise),
  so the conclusion is structural: **cross-layer top-k agreement in prod is
  100 % activation-borne**; synthetic activations carry none of it.
- **Recency is a non-issue:** the recency patch is cross-step only and its
  window is independent of `index_topk_freq`; cross-layer reuse has ZERO
  intra-step staleness (code-firm).
- **Timing ceiling:** pipeline 12–33 µs/F-layer (graphed) + proj proxy
  15–27 µs ⇒ 7 F-layers ≈ **0.23–0.46 ms/step (~1.1–2.2 %)**.

Verdict: NO-GO as an immediate flip on the synthetic gate; **conditional GO**
under the 0.98-bar GATE-B judgment — downstream attention cosine of
freq=8-selected vs freq=1-selected sets on a **real-activation top-k dump**.
The dump had to wait for a FIXED image (the live 002 deploy's indexer
activations are the zero-poisoned path, CRITICAL section — any dump taken
from it gates garbage); **now unblocked**:
`optrt-34fe7aaec-fixed-20260611011615` is verified on 001 (NONZERO indexer
probe), so the dump can run there without waiting for the 002 rollout.

### I3 — `seq_len_threshold` short band (CLOSED DEAD `8e44aeae1` — the width side is spent; short-band default RESTORED)

Closed by measurement, plus a live regression the probe found:

- **Nothing in the decode scoring/top-k pipeline is width-dependent
  anymore**: the FP4 DSL scorer walks ceil(kv/block) per row with width as
  a runtime stride, the C++ top-k walks live kv, and the width dispatch
  override fell in `841f9874a`. Width curve at kv = 4 608: pipeline p50
  **15.39–15.42 µs for ALL widths {8k..132k}**, logits [0, kv) bitwise
  identical — width-bucketing saves 0.0 µs. (The "2.08–2.30×" of indexer.md
  win #1 was real against the pre-I7 width-dependent pipeline; that
  pipeline no longer exists.)
- **BUT the r16-era auto-default `seq_len_threshold=8192` was a
  regression**: the short-band graph warms at 8191 → indexer kernels get
  captured → ultra-short decodes (kv ≤ `index_topk`, which
  `skip_indexer_for_gen_reqs` used to skip outright) pay ~15.4 µs × 61
  layers ≈ **0.94 ms/step they used to skip**. `8e44aeae1` reverts the
  auto-default to `index_topk` so the kv ≤ 1024 band captures the
  indexer-FREE path again; an explicit `seq_len_threshold` stays as the
  operator escape hatch; the dead `_DSL_TOPK_MIN_COLS` is removed and the
  stale width-bucket comments rewritten (the helper survives only as a
  graph-safety bound). Zero eager effect by design (the eager skip keys on
  live kv) — the win lands on graphed prod kv ≤ 1024 traffic.

### ~~I4 — Score→top-k fusion~~ (KILLED — net-zero under graphs)
Measured with the purpose-built bench (`bench_indexer_score_topk_fused.py`,
shipped in `3e03d665d`): **net-zero under CUDA graphs**. The mask is already
fused into the top-k kernel, the logits round-trip is < 1 µs at decode B, and
the launch overhead — the only remaining win — is hidden by graph replay.
Moved to the killed list; do not rebuild.

---

## MoE / communication

### M3 — DeepEP low-latency production flip (enablement SHIPPED `51918fba2`; **DECIDED GO 2026-06-11**)
The top open lever per the composite re-profile: **a2a = 48.4 % of the eager
step** (dispatch 11.0 ms + combine 3.8 ms/step — dominated by notify/spin
kernels, i.e. skew not payload, and inflated under eager launch jitter:
±15 % leg-to-leg where every compute stage reproduces < 1 %). Two
consequences: **the flip is REQUIRED, not optional** — at pure defaults the
comm factory never auto-picks LL (the re-profile leg fell back to
DeepEP-normal; NVLink one/two-sided additionally die on `pidfd_getfd`
without SYS_PTRACE — a prod-container-spec candidate) — and the post-flip
exposed comm must be **re-measured under graphs + overlap** before sizing
further a2a work.
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

**The sizing decision (2026-06-11, EP2 GPUs 5,6 sweep): GO at
`TRTLLM_DEEP_EP_TOKEN_LIMIT=64`.** The prior "inversion at token_limit=64
(225 µs)" was a **measurement artifact, REFUTED**: both prior LL runs used
limit=64 (identical `rdma_bytes`); the 225 µs run's `full_roundtrip` included
`y = recv.float() * s_local` — an **emulated FFN over the FULL padded recv
buffer** (`[32, 2·64, 7168]` ≈ 29.4 M elements). That emulation scales with
the limit; **the a2a kernels do not**, and the real compute chain is
tile-count-driven (moe_sort drops sentinel rows, `tiles_equal=True`,
cross-layout cosine ≥ 0.999993 at ALL limits incl. ep4_prod L64). The real
padding curve is **FLAT** (limit > actual costs +0.2 µs p50 at t16, +1.8 µs
at t4) and **LL wins at EVERY measured point**:

| actual tok/rank | normal | LL L16 | LL L32 | LL L64 |
|---|---|---|---|---|
| 4 (ADP c16)     | 107.7  | 42.9   | 44.1   | 44.7   |
| 16 (TP c16)     | 113.3  | 45.4   | 45.4   | 45.6   |
| 32              | 117.8  | —      | 47.9   | 48.2   |
| 64 (max_batch)  | 119.7  | —      | —      | 58.2   |

(p50 µs/layer roundtrip; worst case 58.2 vs 119.7 = **2.06×** at a full
64-token batch; prod steady range t4–t16 = **2.4–2.5×**; LL tail max ≤ 96 µs
vs normal's ~150 µs + prior 3.7 ms skew pathology; a2a cosine 1.000000 on
graph replay, both ranks.)

**Why 64, and why set it explicitly:** the limit must cover the max
per-rank tokens ever dispatched (`deep_ep_low_latency.py` asserts
`all_rank_max_num_tokens <= limit`); per-iter cost follows **actual**
tokens, the env only sizes the reserved NVSHMEM buffer (~235 MB/rank at
L64, EP2 shape). **Leaving it UNSET is wrong** — the default sizes the
NVSHMEM symmetric heap to the engine `max_num_tokens` (multi-GB at
warmup-sized values). Forwards above the limit hit the feasibility guard
and park/restore (`51918fba2`) — they do not evict LL.

**Production env delta** (staged, `/tmp/m3_sizing_work/
production_env_delta.yaml`): `TRTLLM_FORCE_COMM_METHOD=DEEPEPLOWLATENCY` +
`TRTLLM_DEEP_EP_TOKEN_LIMIT=64` +
`TRTLLM_DEEP_EP_DISABLE_P2P_FOR_LOW_LATENCY_MODE=0` (pin NVLink-P2P
transport) + `TRTLLM_MOE_POST_QUANT_ALLTOALLV=1` (fp4 post-quant dispatch,
44–48 µs with bf16 combine). `use_low_precision_moe_combine` must stay
**false** (nvfp4 combine failed the 0.999 gate at 0.9957; bf16 passes at
0.9999985).

> **CRITICAL TOPOLOGY PRECONDITION:** LL — like **every** a2a strategy —
> engages only under `enable_attention_dp=true` **and** `moe_tp_size=1`
> (`communication_factory.py:126`: no ADP ⇒ the comm factory returns None;
> `moe_tp_size≠1` ⇒ AllGather/ReduceScatter). **Under plain-TP attention
> M3 is inert.** This feeds the ADP-vs-TP production topology decision:
> **ADP unlocks the 2.4–2.5× a2a win; pure TP has no a2a at all** — see
> the regimes section above and [topology_deploy.md](topology_deploy.md).

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

**Status (2026-06-11): phase-1 IN FLIGHT** — the composite re-profile
re-ranks the expert-GEMM chain at **7.2 ms/step (23.3 %**: expert 3.95 +
fc1 2.08 + fc2 1.13 ms, still spanning nvjet_ootst at 193 launches/step +
cutlass3x + DSL persistent launches**)**, making the persistent worker grid
the standing multi-day lever (re-profile target #2).

---

## Prefill / LayerSplit

### L1 — z.ai dense-broadcast overlap (SHIPPED default-on `0a1504755`)
z.ai's design overlaps the dense-KV broadcast behind indexer compute so only
the ~1/8 indexer-K broadcast is exposed. **Shipped default-on**
(`TRTLLM_OPTRT_LAYERSPLIT_PREFILL_OVERLAP`, `0` = legacy sync path).

- **What the sync path actually cost:** ~65–82 ms exposed per prefill step
  (~1.17 ms/layer × 61), dominated **not by wire time** but by the
  per-layer `masked_select`+`unique` top-k-union **host syncs**.
- **The overlap:** on pure-context steps the read set == the top-k union
  (measured `union_fraction=1.0` at every shape), so the dense+scale
  broadcast is issued for the **READ SET** on the comm stream right after
  the indexer-K broadcast — hidden behind real indexer compute — and
  consumed event-gated exactly where the legacy sync broadcast sat.
  Exposed drops to **2.5–8.2 ms/step**: **TTFT −544 ms per 64k prompt**
  (584.9 → 41.0 ms exposed over 8 chunk steps), **~−1.1 s at 128k**.
- **Correctness 20/20:** consumer-visible rows byte-equal AND equal to
  independently regenerated owner truth, both ranks, CP2 + real fp8_fp4
  scoring. **Decode keeps the legacy tiny-union path** (unaffected).
  Graceful fallback on capture / no-group / no-stream / empty-set.
- **C9 interaction:** the C9 IPC channel **wedges both ranks** under this
  traffic pattern — overlap routes via **NCCL only**; C9 stays parked
  (see C9).

TTFT lever only, not TPOT.

### L2 — Read-set block-id hoist (SHIPPED `5bc2b2cb8`)
`_layersplit_compute_read_block_ids` is per-step-invariant but ran
61×/prefill-step with 2 host syncs each. Now memoized once per step keyed on
(ptrs, num_seqs, host kv_lens values, capturing) —
`TRTLLM_OPTRT_LAYERSPLIT_READSET_HOIST=1` default. 4-scenario set-equality
PASS (single-chunk, multi-chunk incl. the defensive no-clear chunk,
prefix-hit, mixed batch). **11.6–12.4 → 0.4 ms per prefill step** (TTFT,
CP2×TP2 worker). (The dense top-k set legitimately differs per layer — only
the read-set computation hoists.)

---

## Transport

### MO1 — MORI-style generation-first / write-mode handoff (**DEPLOY-READY** — release built + activation gap closed; rollout = owner's call)
The MORI-IO blog's best mode is **write mode**: the proxy dispatches prefill and
decode concurrently and prefill pushes KV layer-by-layer, so the RDMA transfer
overlaps prefill compute and only its residual adds to TTFT (read mode serializes
→ +1 full prefill pass). Our analogue is `generation_first` handoff, and the
campaign's own audit *requires* the `handoff_mode="generation_first"` marker and
*rejects* `completed_prefill`.

**Status rewrite (2026-06-10/11 session, findings
`/tmp/mo1_router_work/MO1_FINDINGS.md`, Mac):**

- **The generation-first slice ALREADY EXISTS end-to-end on
  `ai-blaise/dynamo-prod-k8s` main** — tip `01359673f3` ("feat(trtllm):
  route NIXL generation-first handoff", 2026-06-07): the router checks the
  prefill worker's published
  `trtllm_generation_first_disaggregated_params` runtime_data, **mints a
  shared 63-bit `disagg_request_id`**, spawns the `context_only` prefill as
  a background task, **dispatches decode at t0** with `generation_only`
  params, and emits the audit marker `handoff_mode="generation_first"`. All
  op-trt engine dependencies exist (`DisaggScheduleStyle.GENERATION_FIRST`,
  the py_executor readiness gates, the V2 transceiver gen-first branch,
  LayerSplit-aware).
- **But it was NEVER COMPILED**: the commit message itself records `cargo
  check` never completed (the VM lacks `protoc`; etcd-client's build.rs
  fails first), and the **deployed `_core.abi3.so` predates the commit** —
  the live frontend can only do completed-prefill.
- **This session compiled it for the first time** (Mac, protoc 35.0 +
  pinned rustc 1.93.1, zig-cross to `x86_64-unknown-linux-gnu`): `cargo
  check -p dynamo-llm` **PASS**, `--tests` **PASS**, and the `dynamo-py3`
  crate (the exact `_core.abi3.so` that ships) **PASS**. Session edits
  (patch `/tmp/mo1_router_work/mo1_session_edits.patch`, Mac-local, not
  pushed): a **fail-close `handoff_mode="completed_prefill"` marker** on
  the serialized-fallback arm (before this, a silently-degraded deploy was
  invisible to the audit grep) + unit tests for
  `build_trtllm_generation_first_params` (endpoint string/array/empty
  forms, request-type split, shared ctx/disagg request id, dp_rank
  insert/null-strip, 63-bit id cap — previously never compiled).
- **Python activation gap — CLOSED (legacy-path port, 2026-06-10/11):**
  the r20 manifests launch the **legacy worker entrypoint** (`python3 -m
  dynamo.trtllm` → `llm_worker.py` / `handlers.py`), which never published
  the gen-first runtime_data (that code lived only in `llm_engine.py` /
  `unified_main`) — so the router's gen-first arm could never fire. Rather
  than the entrypoint flip (flag-parity risk vs the r20 args), the port
  lands the capability in the legacy path itself (**+113 lines, ZERO
  manifest delta**): `llm_worker.py` publishes the
  `trtllm_generation_first_disaggregated_params` runtime_data
  (`ctx_info_endpoint` + `schedule_style=generation_first`
  [+ `ctx_dp_rank`]) at registration via
  `ModelRuntimeConfig.set_engine_specific`; `handler_base.py` consumes the
  router-built params — prefill takes
  `extra_args["trtllm_generation_first_disaggregated_params"]` (shared
  `disagg_request_id` with the concurrently-dispatched decode), decode maps
  wire `schedule_style` onto `DisaggScheduleStyle.GENERATION_FIRST`.
- **Release build — DONE, inside the target serving image** (001 container
  `mo1build` on `optrt-34fe7aaec-fixed-20260611011615`): `cargo build
  --release` of the `dynamo-py3` crate EXIT=0 (rustc 1.93.1, repo RUSTFLAGS
  replicated — the env-overrides-config gotcha is recorded in
  `TEST_RESULTS.md`); the 109.6 MB `_core.abi3.so` REAL-links the image's
  nixl (RUNPATH `/opt/nvidia/nvda_nixl/lib64`, `DT_NEEDED libnixl.so` — not
  the dlopen-stub mode the Mac cross-check silently used). Tests: `cargo
  test -p dynamo-llm --lib kv_router::prefill_router` **18/18 on the prod
  platform** (first execution — the prior session could only typecheck on
  darwin); **GPU-7 smoke 39/39 incl. the 5 new gen-first tests**.
- **Staged release** at `/tmp/mo1_router_work/RELEASE/` (Mac; mirrored to
  001 `/home/spencer/work/mo1_router_src/RELEASE`): `_core.abi3.so` +
  `python/llm_worker.py` + `python/handler_base.py` (the three MUST ship
  together) + `INSTALL_RUNBOOK.md` (exact site-packages target paths, the
  **load-bearing `__pycache__` purge** — stale .pyc shadow COPY'd sources —
  patch-image Dockerfile, deploy delta, rollback) + `TEST_RESULTS.md`.
  **Deploy-ready; rollout = owner's call.**

**Expected TTFT win** (prod shapes, 61 layers, MLA latent 576 elem/tok/layer;
hides t_xfer + the serialized decode-setup leg ≈ ½ decode iteration):
**~12–17 ms @8k, ~20–27 ms @64k** intra-node NVLink fp8 KV; kvarn_k2v2 is
setup-dominated ~10–18 ms across 8–64k; cross-node 1-rail up to ~60–67 ms
@64k. TTFT lever only (not post-first-token) — tracked but lower priority
for this metric.

---

## Re-screen at the 0.98 functional bar (2026-06-10)

The owner's new bar (downstream-functional cosine ≥ 0.98; formats pinned to
the checkpoint; bit-identical never required) triggered a full re-screen of
the killed list and the pending gates (`/tmp/rescreen_098/RESCREEN.md`).

**RESURRECTED (all four now resolved):**

1. **G2-absorb — re-screened and CLOSED PERMANENTLY (2026-06-11):** the
   MoE-output-cosine gate ran (the decisive experiment) and **FAILED every
   (layer, M) cell under every reading of the bar** — mins 0.60–0.80, p1
   0.77–0.96, L45 mean 0.9695 < 0.98. The f32-functional control refuted
   the near-tied-flip hypothesis (1 flipped expert = −3..6 % output cosine
   functionally; L45 any-expert flip rate 18.9 %), and a second
   bar-independent mechanism (fp4-requant decorrelation amplified by
   routed/shared cancellation) fails tokens even at 8/8 routing agreement.
   Full close-out in G2; artifacts 001 `/tmp/g2_absorb_work/`.
2. **I2 `index_topk_freq` 4→8** (0.23–0.46 ms/step): gate relaxed from set
   recall to GATE-B downstream attention cosine (freq=8-selected vs
   freq=1-selected sets). The real-activation dump it needs was blocked on
   a fixed image — **now unblocked**: `optrt-34fe7aaec-fixed-20260611011615`
   is verified on 001 (CRITICAL section). The one still-open re-screen item.
3. **H3b per-row candidate scaling — A/B done, DEAD again (2026-06-11):**
   set-equality OK on every band (driver v2 masks the OFF path's
   `-1`-padded `top_blocks` in both arms — those pad slots alias page 0 on
   mixed bands, a pre-existing OFF-path artifact surfaced by the gate, counts
   in the table's `offpad` column; not an H3b bug — **root-caused and FIXED
   in `31e0b5be7`**, see the HISA hazard section),
   but **plain ON is SLOWER everywhere** (+1.2–4.1 µs full pipeline,
   33k/66k/132k/mixed × B4/16/64) and the hoisted-cand_count variant is
   breakeven at best (+2.3 µs, mostly ~0). The HISA-scale fix
   (`max_gen_kv_len`, cycle 4) already captured the width win H3b targeted.
   Stays opt-in/off. Table: 001 `/tmp/h3b_ab_work/h3b_table_v2.txt`.
4. **FP4MQALogits index-scoring formal close-out** (0 incremental tok/s —
   already live): the cycle-4 kill ("IoU 0.69–0.83, do NOT default-on
   without an e2e accuracy eval") is **RETIRED**. The fp4 chain is the
   checkpoint-faithful setting (`IndexerK4`) and has been the r20 production
   config all along (`indexer_k_dtype: fp4` +
   `use_cute_dsl_paged_mqa_logits: true`); the correct judgment is
   downstream, and it already exists — GATE-B measured the attention cosine
   of the fp4-chain-selected set vs the TRUE-f32-selected set at **1.000000
   (6 dp) on all 6 shapes** (B {4,16} × kv {4.6k, 33k, 66k}): the IoU
   boundary disagreement is near-tied tail tokens with ~zero softmax mass.
   The "only 1.0–1.02×" speed leg was DP4-specific (bs=4/rank); bs=16 under
   the WarpDecode+TP plan **is** the 1.6× memory-bound regime.
   **CLOSED — GATE PASSED (2026-06-10):** the FP8-chain GATE-B leg ran on 001
   (wave2 image, production top-k routing: C++ at kv 4.6k, DSL at 33k/66k) —
   downstream attention cosine **1.000000 for fp4-set vs fp8-set AND each vs
   the TRUE-f32 set, at all 6 shapes** (B {4,16} × kv {4.6k, 33k, 66k},
   incl. B=16/kv=66k), while the same run reproduced the kill's divergence
   regime (fp4↔fp8 IoU 0.67–0.81) — confirming the wrong-metric diagnosis.
   The kill is formally retired. Artifacts: 001 `/tmp/fp4mqa_closeout_work/`
   (STATUS `OVERALL=PASS`).

**STAYS DEAD at the new bar:** K2 FC2 N=160 (true-f32 cos 0.790 < 0.98, and
the prefill M=1024 crash is bar-independent); the megakernel
**dispatch-fusion** win claim (timing-neutral under PDL — a timing fact,
bar-irrelevant; P1's structural persistent-kernel case is unchanged); the H4
ratio sweep (the format mandate pins HISA4to1 — moved to the killed list);
bf16 indexer logits (fp16 shipped and strictly dominant — equal time, better
boundary fidelity 0.994–0.996 vs bf16's 0.971–0.990, downstream cos
1.000000). Every other kill was speed / arithmetic / topology / mandate —
bar-irrelevant — and stands.

---

## Record-corrected (moved OUT of the killed list)

- **NVFP4 dense MLA proj GEMM accuracy** (was killed as "FAILS accuracy: cos
  0.63–0.83 vs the bf16 path") — **REVERSED 2026-06-10**. The 0.63–0.85
  cosines measured a **corrupted reference**: the harness's `f32_weight()`
  passed `isSfSwizzledLayout=True` to `e2m1_and_ufp8sf_scale_to_float_v2` on
  checkpoint `weight_scale` tensors that are LINEAR layout (the prod loader
  runs `block_scale_interleave` on them, which takes linear input) — the
  scrambled block scales corrupted the "true f32" reference itself
  (old-vs-corrected weight cos 0.65–0.87). With the corrected reference
  (validated against the CPU op at the correct flag, cos 0.99999994), the
  **unmodified production W4A4 path passes everywhere**: cos vs true-f32 at
  L5/M4 cutlass — **o_proj 0.9953 / q_b 0.9966 / q_a 0.9954 / kv_a 0.9953**;
  per-proj minima 0.9951–0.9954 over the full **n=144** sweep (layers
  {5,20,45} × M {4,16} × {spread, outlier} activations × 3 backends). **ALL
  PASS the 0.98 bar.** No production change follows: W4A4 on these projs is
  the checkpoint-faithful mode already in service (these modules ship
  NVFP4-packed with `input_global_scale`), so no faster mode is unlocked —
  the entry's value is the record correction plus one real caveat:
  **out-of-calibration inputs clip** (an activation with amax ≫ the
  calibrated `input_global_scale` clips against the static input scale;
  excluded-by-construction in calibrated traffic; the mitigation is
  recalibration, not code). Artifacts:
  001 `/tmp/dense_proj_work/corrected_accuracy.json`.

---

## Killed candidates (considered space)

- **G2-absorb (constant-0.5 gated-norm fold into PRE_MOE_FUSION)** —
  **CLOSED PERMANENTLY (2026-06-11)** at the 0.98 functional bar itself:
  MoE-output cosine through the real expert chain FAILs every (layer, M)
  cell (mins 0.60–0.80, p1 0.77–0.96, L45 mean 0.9695 < 0.98). Two
  mechanisms: routing flips are functionally expensive (f32 control: 1
  flipped expert = −3..6 % cosine; L45 flip rate 18.9 %) AND fp4-requant
  decorrelation + routed/shared cancellation fails tokens even at 8/8
  routing agreement. The ~117 µs/tok absorb floor is irreducible; the
  shipped chain kernel (`68866e061`) is the end state. Artifacts
  `/tmp/g2_absorb_work/`. **Lesson: near-tied router scores do not imply
  near-identical expert outputs — judge selection changes at the OUTPUT.**
- **H3b per-row live-length candidate scaling** — dead **twice**: perf-dead
  at the A/B (set-equality OK; plain ON +1.2–4.1 µs SLOWER at every band ×
  batch; hoisted variant breakeven at best) after the HISA-scale fix
  (cycle 4) already captured the width win. The candidate score+topk at
  decode B is launch/latency-bound (the H1 lesson again) — per-row width
  shaving buys nothing the band scaling didn't. Code stays opt-in/off.
  `/tmp/h3b_ab_work/h3b_table_v2.txt`. (The `offpad` artifact its gate
  surfaced was real — root-caused and fixed as the `31e0b5be7` HISA
  pad-poisoning hazard.)
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
- **H4 HISA `compression_ratio` sweep ({6,8,12} legs)** — dead by **format
  mandate**: the checkpoint pins HISA4to1, so non-4:1 ratios are a format
  change regardless of recall. Only intra-4to1 tuning remains in scope.
- **bf16 indexer logits** — dead by **dominance**, not by bar: timing equals
  fp16 (both take 2 radix rounds) with strictly worse top-boundary fidelity
  (set-overlap vs fp32 sets 0.971–0.990 vs fp16's 0.994–0.996); fp16 (I6) is
  shipped and already passes a stricter gate (downstream cos 1.000000, ~180×
  fp16 range margin).
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
  **[RETIRED 2026-06-10 at the 0.98-bar re-screen; close-out GATE PASSED** —
  the fp4 chain is the checkpoint-faithful setting (IndexerK4), is the live
  r20 production config, and is GATE-B-judged at downstream attention cos
  1.000000 (fp4-set vs fp8-set and each vs the TRUE-f32 set, all 6 shapes
  B {4,16} × kv {4.6k,33k,66k}); the IoU kill measured the wrong metric and
  the 1.0–1.02× was the wrong (DP4) regime. See the re-screen section.]
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
  per-rank concurrency, not max_batch_size. **[The inversion claim was
  REFUTED by the 2026-06-11 sizing sweep — an emulated-FFN measurement
  artifact; the decision is an explicit limit=64. See the M3 GO entry
  below.]**
- **Cycle 10** (shipped `fd705a6f5`): `_FP4OUT_MIN_M=128` guard lifted — the
  swiglu+fp4-out fusion now engages at decode M (exact vs TRUE-f32, OOB demo
  clean, ~100 µs/step + 58 launches). K3.
- **SM1/SM3 schedule hoists — exposure check DONE**: the hoisted schedule work
  is **already OVERLAPPED at c16** — correct but flat, no exposed-time win.
  Status revised from "queued exposure check" to closed-no-win at c16 (revisit
  only if the overlap structure changes).
- **Cycle 11** (shipped `5bc2b2cb8`): **C2 KVarN delta-restore DEFAULT-ON**
  (O(batch) host-integer delta; 5-scenario lockstep equivalence PASS; host
  48.8 → 0.3 ms/fire, 12.1 → 0.08 ms/step amortized at TP bs=16) + **L2
  read-set hoist** (11.6–12.4 → 0.4 ms/prefill step, CP2×TP2) + **S5 stats
  decimation** (48.2 → 16.1 µs/iter, p99 69 → 35; exposed TPOT win ~0 at c16
  — jitter removal; the old 100–250 µs premise was stale) + **the linear.py
  input_scale fix itself** (drop never-initialized placeholder Parameters —
  the fix whose absence is the CRITICAL hazard above). Also carried, both
  opt-in/off at this commit: the wk+wp fused GEMM (then mis-flagged by the
  zero-vs-zero gate) and the S2 ADP-collective fusion (held; moot under
  WarpDecode+TP).
- **Cycle 12** (shipped `68866e061`): **I5 wk+wp fused GEMM DEFAULT-ON**
  (1.96–1.97× → ~2.0 ms/token; the v1 cos=0.0 was the input_scale bug — the
  fusion was always correct, re-gated at cos 1.000000 / 0.999996) +
  **single-launch CuTe gate+quant on the MoE input** (7.04 → 4.19 µs/layer,
  −165 µs/token; LINEAR-SF `Fp4QuantizedTensor` handoff skips the MoE's own
  quant) + **G2-absorb measured and REJECTED on routing** (top-8 overlap
  0.965–0.988 at layers 20–60; patch recorded, not shipped).
- **Cycle 13** (shipped `34fe7aaec`): **KVarN stale-record-on-recycle fix,
  default-on** — pool records now invalidated on block free/recycle
  (`invalidate_blocks` + free/rewind fan-out); the d3 stale-KV clobber on the
  full-scan restore path is fixed; restore bit-equivalent to a never-recycled
  universe (cos 1.0) on all 4 paths; the documented re-commit contract is now
  actually honored. C2b.
- **Live-hazard audit (2026-06-10)**: the pre-`5bc2b2cb8` input_scale bug
  traced into the LIVE 002 r20 deploy — runtime-proven all-zero indexer-proj
  outputs, DSA top-k degenerate past 8192 tokens; no escape hatch in the
  served config. CRITICAL section added at the top of this doc; remediation
  runbook staged (`/tmp/livehazard_work/REMEDIATION.md`); `3e03d665d`
  indexer-triple equality claims marked vacuous (wk/wp re-proven same day;
  wq_b re-proven in the close-out entry below).
- **Dense-proj NVFP4 accuracy record-corrected (2026-06-10)**: the killed
  "cos 0.63–0.83" verdict measured a corrupted reference
  (`isSfSwizzledLayout=True` on linear-layout checkpoint scales); corrected
  W4A4-vs-true-f32 cosines are 0.995+ everywhere (n=144, layers 5/20/45) —
  moved to the record-corrected list. No production change (W4A4 already the
  checkpoint-faithful mode); out-of-calibration clipping caveat recorded.
- **0.98-bar re-screen (owner directive 2026-06-10)**: functional gates are
  now cosine ≥ 0.98 downstream (bit-identical never required; formats
  checkpoint-pinned: NVFP4-W4A4KV4-IndexerK4 + HISA4to1, FP/BF16 logits ok).
  Resurrected: G2-absorb (MoE-output-cosine gate), I2 (GATE-B), H3b A/B,
  FP4MQA formal close-out. Stays dead: K2 N=160, megakernel dispatch-fusion,
  H4 ratio sweep, bf16 logits. See the re-screen section.
- **FP4MQA + wq_b formal close-outs PASSED (2026-06-10, 001 GPU 4, wave2
  image)**: (A) fp4-vs-fp8 scoring-chain GATE-B — downstream attention
  cosine 1.000000 (fp4-set vs fp8-set, each vs the TRUE-f32 set) at all 6
  shapes B {4,16} × kv {4.6k,33k,66k} under the production top-k routing,
  while reproducing the cycle-4 divergence regime (fp4↔fp8 IoU 0.67–0.81) —
  the cycle-4 FP4-scoring kill is formally retired; (B) wq_b
  cutlass↔cuBLASLt re-gate under HEAD dynamic-quant semantics —
  bit-identical with liveness asserted (cos 1.000000, max|diff| 0; cuBLASLt
  1.18–1.21×, stays the pick), closing the last vacuous `3e03d665d` claim.
  Artifacts: 001 `/tmp/fp4mqa_closeout_work/`.
- **Cycle 14** (shipped `0a1504755`): **L1 z.ai prefill dense-broadcast
  overlap DEFAULT-ON** (the sync path's 65–82 ms/step exposed cost was the
  masked_select+unique host syncs, not wire; read-set broadcast on the comm
  stream hidden behind indexer compute; exposed 2.5–8.2 ms/step, **TTFT
  −544 ms @64k / ~−1.1 s @128k**; 20/20 correctness; decode unaffected; C9
  IPC wedges under the pattern → NCCL-only, C9 parked) + **C4 KVarN eager
  decode-restore host-path** (the "indexer FSSS cub select ~3 %" profile
  line was a MISATTRIBUTION — 671 cub kernels + 183 syncs/step-rank
  selecting an empty set; host-mirror selection 19.4×; 300-trial
  set-equality; profile-correction note added to the methodology section).
- **M3 sizing sweep → DECIDED GO (2026-06-11, EP2 GPUs 5,6)**: the
  "inversion at token_limit=64" was an emulated-FFN measurement artifact —
  the real padding curve is FLAT and LL wins at every point (2.4–2.5×
  steady c16, 2.06× at a full 64-token batch). Env delta staged
  (`production_env_delta.yaml`: FORCE_COMM_METHOD + explicit
  TOKEN_LIMIT=64 — unset would size the NVSHMEM heap to engine
  max_num_tokens). **Topology precondition recorded: ADP + moe_tp_size=1
  (`communication_factory.py:126`) — feeds the ADP-vs-TP call.**
- **G2-absorb CLOSED PERMANENTLY (2026-06-11)**: the decisive
  MoE-output-cosine gate FAILed every cell (mins 0.60–0.80, L45 mean
  0.9695); f32-functional control refuted the near-tied-flip hypothesis;
  second mechanism (fp4-requant + cancellation) fails even 8/8-routing
  tokens. ~117 µs/tok floor irreducible. Moved to the killed list.
- **H3b A/B → DEAD again (2026-06-11)**: set-equality OK, plain ON slower
  everywhere (+1.2–4.1 µs), hoisted breakeven; the HISA-scale fix already
  owns the width win. Moved to the killed list.
- **input_scale remediation STAGED (2026-06-11)**:
  `optrt-34fe7aaec-fixed-20260611011615` built + pushed to 001's registry,
  **4/4 verified incl. the NONZERO indexer probe** (the hazard is fixed in
  this image); ROLLOUT.md stages 002 transport, the DGD swap, the N1
  NUMA-pin snippet, and post-roll checks. **Rollout = owner's decision.**
- **MO1 router-slice validation (2026-06-10/11, Mac)**: the
  generation-first slice already exists on `dynamo-prod-k8s` main
  (`01359673f3`) but was never compiled (VM lacks protoc; deployed `.so`
  predates it); `cargo check` now PASSES for the prod platform (dynamo-llm
  + tests + the dynamo-py3 `.so` crate) with a fail-close
  `completed_prefill` marker + gen-first param unit tests added
  (Mac-local patch). TTFT estimate ~12–17 ms @8k / ~20–27 ms @64k.
  **[Both remaining gaps closed same session — next entry.]**
- **MO1 release built + activation gap closed (2026-06-10/11) —
  DEPLOY-READY**: release `_core.abi3.so` built **inside the target
  serving image** (cargo test 18/18 `kv_router::prefill_router` on the
  prod platform — first execution; GPU-7 smoke 39/39 incl. the 5 new
  gen-first tests); the python activation gap closed via the
  **legacy-path port** (+113 lines: `llm_worker.py` runtime_data
  publication + `handler_base.py` params consumption — ZERO manifest
  delta); `INSTALL_RUNBOOK.md` staged at `/tmp/mo1_router_work/RELEASE/`
  (exact site-packages paths, the load-bearing `__pycache__` purge,
  rollback). **Rollout = owner's call.**
- **Cycle 15** (shipped `8e44aeae1`): **I3 CLOSED DEAD by measurement** —
  nothing in the decode scoring/top-k pipeline is width-dependent anymore
  (width curve at kv=4608: pipeline p50 15.39–15.42 µs flat across widths
  {8k..132k}, logits bitwise identical; bucketing saves 0.0 µs) — **plus
  the regression the probe found**: the r16-era auto-default
  `seq_len_threshold=8192` forfeited the indexer-free capture for
  kv ≤ `index_topk` traffic (~0.94 ms/step back), REVERTED to `index_topk`
  (explicit threshold = operator escape hatch; dead `_DSL_TOPK_MIN_COLS`
  removed). Plus the **dense-MLP (layers 0–2) gate+quant handoff
  default-on** (`TRTLLM_OPTRT_GATED_PREMLP_QUANT`: swizzled-SF cute
  epilogue, new op `cute_lowrank_gate_quant_nvfp4_swizzled`, bit-exact
  incl. GEMM output, 4→3 kernels, ~6–8 µs/token). nvfp4_fusions.md #13d.
- **Composite re-profile on the FIXED image (2026-06-11)**: eager c16,
  full shipped stack default-on — **30.8 ms/step/GPU, −26.1 % vs the
  06-10 baseline (41.7 ms), composition CLEAN** (rc=0, zero
  tracebacks/NaN). New ranking: MoE 74.6 % (a2a 48.4 % eager-exposed,
  expert GEMMs 23.3 %), dense-proj 10.0 % (−62.5 %), norm/rope/quant
  5.4 %, glue 5.3 % (−58 %), sparse-MLA 2.5 %, indexer 1.3 % (−77 %),
  HISA 0.7 %. Per-stage deltas mapped to commits; the methodology section
  now carries these numbers. Indexer callout: real (non-zero) outputs ADD
  +146 µs amax + +134 µs real GEMM but the top-k rebuild deleted 10× more;
  the predicted indexer-share rise applies only to long-kv traffic
  (kv ≤ 640 here). Next targets set: a2a under graphs+overlap (M3 delta
  REQUIRED), expert-GEMM megakernel (P1 phase-1), dense-proj batching
  (B2). Artifacts: 001 `/tmp/rerank_profile_work/`.
- **Cycle 16** (shipped `31e0b5be7`): **TWO live HISA decode
  selection-corruption bugs fixed** — RC1 pad-poisoning (−1-padded
  `top_blocks` alias page 0 → duplicate sink scores displace up to
  987/1024 real candidates on short rows in mixed bands, every c16 step;
  raw negatives reach `topk_indices_buffer`) + RC2 logits-stride
  (row-padded scorer output indexed flat → every non-128-aligned prefix
  mis-masks, uniform bands included). Fix = the −1-sentinel contract +
  stride-aware writes; HEAD repro EXACT (offpad 623/1473/5277 = the H3b
  counts); fix passes the exact `torch.topk`-reference equality; timing
  FIX == HEAD. **AOT `.cu`/`.h`/`.cpp` parts ride the NEXT image build;
  python fallbacks live.** Hazard section added; all pre-fix HISA
  selection-quality observations marked suspect.
- **Queued**: input_scale remediation ROLLOUT (owner's call; image staged —
  the **next full-source build must carry ≥ `31e0b5be7`** for the HISA AOT
  fix too); **the M3 production flip + a2a re-measure under
  graphs+overlap** (env delta REQUIRED — the factory never auto-picks LL;
  ADP + `moe_tp_size=1` precondition feeds the ADP-vs-TP call); **P1
  single-CTA persistent megakernel phase-1 (in flight)**; **B2 dense-proj
  residual batching (in flight)**; I2 GATE-B real-activation dump
  (unblocked — fixed image on 001); MO1 ROLLOUT (deploy-ready; owner's
  call); a kv ≫ 1024 profiling leg (the long-kv indexer-share check); then
  N1 (rides the rollout manifest), M1, K1 PDL, H3a/H3c. Parked: C9 (IPC
  wedge under L1 overlap). Held: S2 (moot under WarpDecode+TP).

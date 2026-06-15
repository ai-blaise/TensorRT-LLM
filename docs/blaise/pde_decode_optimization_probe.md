# PDE decode optimization probe — 6-agent parallel sweep (2026-06-15)

Directive: push subagents on (1) thoroughness of prior conclusions and (2) orthogonal/creative
angles, for the DeepSeek-V3.2-REAP-345B decode on op-trt. Six agents, one B200 GPU each, fast
JIT/direct tests at real prod decode shapes, CUDA-graph-capture regime (production runs captured).

## Headline (actionable): cuBLASLt FP4 GEMM is disabled in the HiSparse build → ~2x dense-GEMM penalty

- `scripts/blaise_build_hisparse_thop.sh` (the fast th_common iteration build, FAST_BUILD=ON) passes
  **`-DENABLE_CUBLASLT_FP4_GEMM=OFF`**. The CMake option (`cpp/CMakeLists.txt:74`) defaults ON and only
  auto-disables below CUDA 12.8 (both images here are CUDA 13.1) — so this OFF is a deliberate
  build-speed shortcut, not a version gate.
- Result: `CublasLtFP4GemmRunner` is NOT registered in the current HiSparse images
  (`optrt-aaa7e2b542b2-...-proof`, `optrt-4d274bd53338-...-serving-import-proof`, both 20260613);
  `IS_CUBLASLT_AVAILABLE=False`. The older `optrt-34fe7aaec-fixed-20260611` HAS it.
- Production NVFP4 GEMM defaults to cublaslt (`TRTLLM_DSV3_MLP_NVFP4_BACKENDS` / `_MLA_PROJ_` default
  `'cublaslt'`, modeling_deepseekv3.py:810). With cublaslt absent, dense NVFP4 GEMMs fall back to
  **cutlass, ~2x slower than cublaslt** under capture (measured below).
- **Fix:** serving builds must build th_common with `ENABLE_CUBLASLT_FP4_GEMM=ON`. (Fast-iteration
  builds may keep it OFF for speed — but that th_common must not ship to serving.)
- **Caveats:** (a) confirm which th_common the live 345B decode actually ships (fast-build import vs a
  full build with the flag ON) — if it ships a full build, no regression; (b) end-to-end rebuild+serve
  validation could not be run here (the smoke build repo / cache / buildtools image are absent on this node).

## Captured NVFP4 dense-GEMM backend ranking (us/op, vec=16 swizzled, cos~1.0)

| shape (K->N) | M | cutlass | cublaslt | cutedsl | cuda_core |
|---|---|---|---|---|---|
| o_proj 16384->7168 | 1 | 31 | **15.4** | 23 | 51 |
| q_a 7168->1536 | 1 | 15.1 | **6.1** | 7.1 | 10.3 |
| moe_up 7168->2048 | 1 | 15.1 | **6.1** | 7.1 | 10.3 |
| moe_down 2048->7168 | 1 | 6.3 | **3.6** | 4.3 | 12.3 |

Ranking with all backends present: **cublaslt > cutedsl > cutlass > cuda_core**. cublaslt beats cutedsl in
11/12 shape×M cells; cutedsl beats cutlass ~1.66-2.13x (so cutedsl only matters where cublaslt is absent).
Dense GEMMs use scaling_vector_size=16 (NOT the ue8m0/vec=32 of the indexer path); cutedsl is vec=16-only.

## Other levers

- **KV bytes (real, build-gated):** served fp8 KV = 576 B/tok; nvfp4 = 324 B (1.78x fewer); the
  checkpoint's native higgs-2bit = 258 B (2.23x). Needs `sparse_mla_decode_nvfp4` compiled in (source
  exists in `cpp/kernels/flashMLA` + CMake; absent from the proof image) + `kv_cache_config.dtype` flip.
- **M=1 small-GEMM floor is intrinsic (~9-13us):** a hand-written CUDA-core NVFP4 GEMV cannot beat the
  tensor-core backends — software FP4 decode alone (~10us) equals cutedsl's total; the memory-only floor
  is ~4us (23% BW, small-N latency-bound). Tensor-core dequant-in-MMA is required; no custom kernel wins.

## Confirmed negatives (now rigorous)

- **Layer-level weight prefetch/overlap: dead.** HBM saturates on one stream (81% peak); a 2nd concurrent
  read serializes (1.87x). Nothing to prefetch around at M=1 (compute ~free, the GEMM IS the weight read).
- **topk device-resident: loses.** Pushed hard (warp-private hist, parallel threshold walk, compaction,
  8->2/3 passes); parity on block-topk, 1.59-1.80x slower on final-topk vs cute_dsl captured. cute is
  already optimal (STS.128-vectorized CuTe-DSL); no capturability dividend (cute captures fine).
- **Activation/scale fusion: already comprehensive.** rmsnorm+quant fusion is load-bearing (2.0-2.5x) and
  applied at every paying boundary; only the attn-input seam is unfused (~1.1x, gate-compute-bound).
- **Indexer KV-scan at floor** (fp4-K, fp16 logits, live-kv-only, ~6.2us). **cuda_core@M<=8 already an
  AutoTuner candidate.** **Decode already full-graph-captured.**

## Net
One real high-impact actionable win (restore cublaslt FP4 in serving builds, ~2x dense GEMMs, pending
deploy-provenance), one real-but-build-gated lever (nvfp4/higgs KV bytes), and a thorough confirmation
that the rest of the decode is already well-optimized. Harnesses: blaise_perf/pde_directtest/.

## CORRECTION — cutedsl CAN beat cublaslt on o_proj (tactic-selection fix; 2026-06-15)

A follow-up agent (deep-read CuTe DSL docs + arXiv 2603.02298 + Veitner + Colfax; CZS-gated) overturned the
"cutedsl never beats cublaslt" claim above. Independently re-verified on GPU6 (image -fixed-20260611, cos=1.0):

| shape (K->N) | M | cublaslt | cutedsl WIN_TACTIC | result |
|---|---|---|---|---|
| o_proj 16384->7168 | 1/4/16/64 | 16.4-16.5 | **14.25 / 13.28 / 14.36 / 14.36** | **BEATS 1.14-1.24x** |
| q_a 7168->1536 | * | 8.24 | 8.23 | ties |
| moe_up 7168->2048 | * | 8.24 | 8.23 | ties |
| moe_down 2048->7168 | * | 6.17 | 6.17 | ties |

WIN_TACTIC = `((256,64), cluster(4,1), swap_ab=True, prefetch=False)` on the EXISTING
`Sm100BlockScaledPersistentDenseGemmKernel` (tactic-tuning, not a new kernel). Lever: swap_ab=True puts
N=7168 on the kernel-M axis -> 28-56 M-tiles fill the 148 SMs (wave-quantization, the #1 Colfax-CLC technique
for small M). **The gap was AutoTuner MIS-SELECTION**: stock `cute_dsl_nvfp4_gemm_blackwell` picks
swap_ab=False/big-tile = 26.6us on o_proj; `prefetch=True` is catastrophic at small M (246-343us).
Negatives: dispatch-split-K is S-x slower (serializes under capture); 7.34us pure-BW floor unreachable
(~14.3us = 1.95x = real block-scaled-GEMM plateau). CZS: czs_py pybind not built on 001 -> CLI on a
hand-encoded Module (14 Proved / 1 Disproved-artifact, legality-only); cos=1.0 is the correctness gate.

**Scope (honest):** o_proj only (1x/layer x 61 ~= 130us/tok ~= 0.65% of decode = modest). Deploy needs BOTH
(a) 'cutedsl' added to the NVFP4 allowed_backends AND (b) the AutoTuner fixed to prefer swap_ab=True + disable
prefetch at small-M tall-N shapes (else the stock mis-pick makes cutedsl WORSE, which is why prod excludes it
today). The AutoTuner small-M-tall-N fix also lifts the stock op 26.6->14.2us (o_proj) / 10.27->8.23 (q_a/moe_up).
This does NOT change the headline GEMM lever (ensure warmed cublaslt in serving) — it's an additional o_proj-only edge.

## Round 2 (deeper squeeze, 2026-06-15): R1 tactic is the kernel optimum + CZS real-pybind upgrade

A round-2 agent (fresh direct read of the 4 sources for techniques BEYOND tactic-selection; CZS-gated)
found NO further kernel-level win — and rigorously bounded why:
- **q_a / moe_up / moe_down: BEATING cublaslt is config-impossible.** FLAT ~8.24us (q_a/moe_up) / ~6.17us
  (moe_down) across all occupancy(4-22%)/tile/cluster; cublaslt hits the SAME fixed NVFP4-blockscaled
  per-launch floor (TMEM-lifecycle + pipeline prologue/epilogue at K=7168). R1's "tie" IS the ceiling.
- **o_proj 14.3us is a structural plateau** = 1.95x the 7.34us pure-W-read floor; the gap is the
  HARDWARE-IMPLICIT per-k_block `tcgen05.cp(SF)->tcgen05.mma` serialization (Colfax block-scaling tutorial:
  no overlap) + the checkpoint-fixed nvf4 vec=16 SF (4x heavier TMEM than mxf4). Occupancy is NOT the
  bottleneck (sweep flat; 76% occ is WORSE); intra-kernel split-K is dead (re-serializes SF per partial).
  A from-scratch 2-deep SF-TMEM-ring is the only lever but the implicit pipeline likely prevents the overlap
  -> high-risk/multi-day/low-yield for o_proj's ~0.65%/tok share. NOT worth it.
- Best config UNCHANGED from R1: `((256,64),cluster(4,1),swap_ab=True,prefetch=False)`.

**Verification upgrade:** built the `czs_py` pybind (pybind11 3.0.4) -> `/home/spencer/work/CZS/python/czs/_native.cpython-311-*.so`
(REAL_PYBIND confirmed). The R1 winning config is now structurally verified against the REAL compiled kernel
via `run_all_passes`: **6 Proved / 0 Disproved / 0 Unknown** (clean — removes R1's hand-encoded bank-conflict
artifact). cos=1.0 throughout. NET: the cutedsl squeeze is exhausted at the kernel level; R1's o_proj win
(1.14-1.24x) stands and is now real-kernel-CZS-verified; q_a/moe_up/moe_down are provably at the hardware floor.

## AutoTuner config-change prototype (2026-06-15): adding 'cutedsl' to allowed_backends auto-realizes the o_proj win

Measured the WARMED nvfp4_gemm dispatcher (autotune() context, captured, cos=1.0), prodset vs prodset+cutedsl:
| shape | M | prodset (cutlass,cublaslt,cuda_core) | +cutedsl | lift |
|---|---|---|---|---|
| o_proj | 1/16 | 16.39/16.40us | **14.32/14.35us** | **1.14x** |
| q_a / moe_up / moe_down | * | 8.21 / 8.21 / 6.16 | same | tie |

**The warmed AutoTuner auto-selects cutedsl on o_proj (16.4->14.3) and keeps cublaslt elsewhere — strictly
non-regressive.** So the fix is a CONFIG change, NOT a get_valid_tactics code edit: add 'cutedsl' to the three
`TRTLLM_DSV3_MLP_NVFP4_BACKENDS` / `_MLA_PROJ_` / `_INDEXER_NVFP4_BACKENDS` env vars (or the _dsv3_mlp/_mla_proj/
_indexer defaults in modeling_deepseekv3.py / attention.py / dsa.py). The earlier 26.6us "stock" was the
UN-warmed default tactic, not the AutoTuner's warmed pick (the winner IS in get_valid_tactics already).
Prereq (same as the headline GEMM lever): the serving image must have cutedsl available (CuTe DSL JIT, present)
AND the AutoTuner warmed at model load (prod model-load does this). **Decode-level impact: o_proj ~= 5% of the
~20.8ms step -> ~0.6% tok/s/user. Real + clean + deployable (config-only) but modest.** Harness: blaise_perf/pde_directtest/autotune_proto/proto.py.

## Broader TileRT/PDE system probe (2026-06-15): production decode is comprehensively optimized; the lever is the ~65% inter-kernel overhead

System map (per-layer, M=1, measured under capture on B200): dense GEMMs ~50us + attention (flash_mla sparse
decode) ~18us FLAT + indexer ~6us + MoE (WARPDECODE cute_dsl grouped-GEMM) ~45us = **~120us measured compute
vs ~340us/layer budget (20.8ms/61L) -> ~65% is inter-kernel launch/overhead/sync + bmm/rope/quant glue.** This
confirms the "decode is overhead-bound, 20-40x over BW floor" thesis with measured numbers.

- **MoE: WARPDECODE beats CUTLASS 1.77x(M1)/1.38x(M8)/1.30x(M32), cos=1.0 — but NOT a new win:** ALL production
  decode configs (decode.yaml, smc_agg_tp4.yaml, topo-c1-dp2tp4-r20) ALREADY use WARPDECODE; the only CUTLASS
  user is sdt_gen_decode.yaml, a DENSE-GEMM MICROBENCH (CUTLASS pinned to isolate the GEMM variable; README +
  header confirm). Corrects the earlier read of sdt_gen_decode as the live serve config — it's a microbench.
  MoE tactic is NOT AutoTuner-mis-picked (trtllm_gen AutoTuner == no-autotune 1.00-1.02x, unlike the dense GEMM).
- **Attention** (standard FMHA MLA-gen on fp8 KV): ~18us, small + flat in batch — not a lever.
- **SMC spec-decode (serve-gated, biggest potential):** draft = 6 GLM-9B-FP8 forwards per verify (gamma=6);
  forwards 1-5 each process batch x 25 tokens (n_particles=4 x gamma + root tree) = the dominant added cost.
  tok/s/user = accepted_len / (6x draft + 345B verify); acceptance is data-dependent -> serve-gated. Cheap
  config lever: smc_vectorize_logprob_record (off; saves ~15 launches/step, bit-exact within 1 ULP) + gamma/
  n_particles tradeoff.
- Topology already DP2/TP4-tuned; overlap scheduler on.

**BIGGEST REMAINING tok/s/user LEVER (honest): NOT any single compute kernel** (MoE/attn/GEMM/indexer all
optimal). It's the **~65% inter-kernel overhead** — exactly what the validated PDE primitives (G3 device-control-
flow, G9 cross-step persistence) target, but those are NOT yet wired into the live serving runtime. The other
lever is SMC acceptance x draft-cost. **Both require the model/serve** (wire PDE control-flow into the live
runtime + measure, or tune SMC vs live acceptance). No new op/backend-level no-serve serving win remains.

## PDE-stack wiring audit ROUND 2 (2026-06-15): seam-level, what is JIT-tractable vs build/serve-gated

Deeper pass than R1 (which wired only the G3 top-k hook). Located the REAL runtime seam for every
PDE primitive and proved the gating empirically (proof image op-schema probe, GPU3 B200).

### Empirical op inventory (proof image optrt-aaa7e2b542b2-...-20260613, blaise_perf/pde_directtest/pde2_opcheck.py)
- topk ops present: indexer_topk_decode + cute_dsl_indexer_topk_decode -> G3 substitutes at BOTH the HISA
  block top-k AND the final top-k (all 5 _indexer_topk_decode call-sites share the wrapper).
- G4 hisparse hot chain: ALL 8 ops PRESENT (hisparse_topk_to_block_positions ... sparse_mla_decode_kvarn_hot).
- G4 copy-overlap: hisparse_submit_packed_kvarn_copy_schedule PRESENT but does NOT advertise the overlap args
  (overlap_copy_stream / copy_stream_handle ABSENT from the schema) -> _swap_in_overlap_args_supported()==False.
- G9 cross-step: indexer_xstep_recency_patch PRESENT. G8: extract_real_draft_tokens_op PRESENT.

### Per-primitive verdict
- G3 device-control-flow top-k = JIT-tractable, WIRED (default-OFF, now prod-safe). Seam: dsa.py
  _indexer_topk_decode (the chokepoint for all 5 top-k sites: block-topk 3085/3155, two-level 3475, from-logits
  3561, final 4559) + the explicit _pde_g3_topk_active() branch at dsa.py:4552. R2 FIX: the G3 kernel stages
  every score column in dynamic SMEM (cols*4 + 1KB); B200 optin cap ~227KB, so it FAILS the launch
  (cudaErrorInvalidValue) above ~57.8K cols. R1 routed the FINAL top-k (prod padded width 132096) through G3 when
  gated ON -> would hard-crash the live decode on enable. pde_g3_topk_decode now returns bool (False = SMEM over
  cap, writes nothing) and the wrapper falls back to indexer_topk_decode. Gate-ON now recall=1.0 where G3 fits +
  bit-exact prod-op fallback where it does not, at EVERY shape. Verified GPU3: C in {1032,8192,32768} -> G3
  set==gold; C in {65536,132096} -> decline+fallback set==gold, no crash; gate-OFF==prodop unchanged. Real-import
  test confirms the genuine site. (Still not a perf win in this capture-safe-cute image; win is C++-Scheme-X /
  full indexer->topk->gather fusion = serve-gated.)
- G4 het copy/compute overlap = ALREADY FULLY WIRED natively; BUILD-gated on ONE op signature. Seam:
  hisparse.py submit_packed_kvarn_copy_schedule (2168, P1 per-step miss-DMA overlap) + prepare_hot_pool_overlapped
  (2291, wide-window hoist) forked at attention.py:2956 forward_absorption_generation, deferred join consumed at
  attention.py:2789. Gated by hisparse_overlap_swap_in (default ON) AND the C++ .so advertising
  overlap_copy_stream/copy_stream_handle. The proof image op LACKS those args -> _swap_in_overlap_args_supported
  ==False -> both P1 and the wide-window hoist fail-CLOSE to byte-identical serial. NOT JIT-wirable: the overlap
  needs the copy-schedule op recompiled with the overlap-arg launch-stream signature (cpp/.../hisparse). All other
  G4 chain ops are present, so the ONLY missing piece is that op variant. (Even wired, the overlap is only live
  when HiSparse hot-pool serves -- not on the dense 345B decode path without the serve.)
- G8 MTP device variable-accept-len loop = SERVE/BUILD-gated. Two host loops: drafting_loops.py:758
  (SMCStaticParticleDraftingLoopWrapper.forward "for layer_idx in range(1, max_draft_len)" = gamma=6 FULL GLM-9B
  forwards, each internally CUDA-graph captured) + smc.py:855 (SMCSampler._accept_selected_particle
  "for depth in token_indices" = the accept walk). The accept side is intrinsically HOST/scheduler-resident (it
  mutates LlmRequest python objects: add_new_token, _handle_stop_criteria, py_num_accepted...). The draft side is
  already device-resident + graph-captured per forward with sync-free inter-forward glue (_host_draft_layout memo;
  extract_real_draft_tokens_op CUDA-graph path; .item() only behind SMC_CUDA_SYNC_PROBE). A device-side
  variable-accept-len LOOP = fusing the gamma graph replays into one persistent kernel that early-exits on a
  device-resident accept decision -> needs the C++ engine (model forwards in-kernel) + the live model. No bit-exact
  JIT hook exists.
- G9 cross-step persistence = ALREADY has a runtime-level analog WIRED (config-gated, APPROXIMATE). Seam:
  dsa.py cross-step Top-K reuse (_xstep_reuse_active 2644 / _xstep_reuse_decode 2662 / _xstep_store_decode 2714;
  short-circuit at 4154; counter advance 4661). Persists the decode Top-K selection across index_topk_step_freq
  steps and SKIPS the ~24us logits-MQA + Top-K recompute on reuse steps -- the application-level form of G9. Runs in
  the non-graph-captured mla_dsa_attn_inplace eager region (why it is JIT-safe). Gated by index_topk_step_freq
  (llm_args.py:329, default None=OFF) + index_topk_step_recency_patch (default OFF). It is NOT a bit-exact hook:
  frozen reuse drops the newest up to (freq-1)*next_n positions (approximation budget); the recency patch op
  restores them (indexer_xstep_recency_patch, present, jaccard=1.0 vs the old block) at a launch cost. So
  G9-as-shipped is an accuracy/throughput LEVER, not a correctness-gated substitution; tuning it vs live
  acceptance is serve-gated.

### Net R2
Only G3 is JIT-tractable as a default-OFF correctness-gated hook, and R2 made it prod-safe (SMEM guard;
was a latent gate-ON crash at the prod final width). G4 is wired but build-gated on the single copy-schedule
overlap-arg op variant (every other hot-pool op is present). G8 is serve/build-gated (draft = full-model
graph replays; accept = host/scheduler-resident). G9 already has a config-gated approximate cross-step-reuse
analog in the eager indexer region; the bit-exact device-resident form needs the engine. Harness:
blaise_perf/pde_directtest/pde2_*.py. tip op-trt-pde-pdewire2.
---

## MISSED-OPTIMIZATIONS round (2026-06-15, op-trt-pde-missed2): two new wins + one rigorous serve-gated lever

Directive: find decode optimizations the prior rounds MISSED — orthogonal/creative angles not on the
"confirmed-optimal" map. Direct op/kernel tests at real prod shapes under CUDA-graph capture, B200 GPU4.
Harnesses: blaise_perf/pde_directtest/missed2/.

### WIN 1 (IMPLEMENTED, verified cos=1.0): q_b_proj || pre_indexer_proj on a DEDICATED stream

The prior probe noted "the codebase already uses multi-stream" but did NOT find that the
`forward_dsa_proj` (captured Op-1) seam runs `q_b_proj` (dense Q up-proj 1536->24576) and the indexer's
`pre_indexer_proj` (wq_b 1536->8192 + fused wk/wp 7168->192 + fused_rope_cat quant) **serially on the
default stream**, although they are mutually independent (both read qr/hidden_states read-only; neither
writes the other's inputs).

KEY FINDING — the dedicated stream is load-bearing: `pre_indexer_proj` ALREADY uses `self.aux_stream`
(== the shared Attention aux stream, same object as the indexer's) for its internal q/k quant overlap.
Reusing that aux stream for the q_b overlap makes the two overlaps contend and the GEMM-level win
COLLAPSES. Measured under capture (cublaslt image, real shapes, full faithful Op-1 incl indexer inner
overlap):

| arrangement | M=1 | M=8 | M=32 | M=64 |
|---|---|---|---|---|
| q_b on SHARED aux (contends) | 0.980x | 0.971x | 0.997x | 1.000x |
| **q_b on DEDICATED stream** | **1.160x** | **1.144x** | **1.153x** | **1.176x** |

Pure proj-GEMM region (cublaslt, q_b||indexer GEMMs, no inner quant): serial 32.8us -> parallel 24.6us =
**1.33x**, cos=1.00000. Full Op-1 (incl indexer inner overlap): serial 59.4us -> 51.2us = 1.16x @M=1.
q_b (~12us cublaslt) is fully hidden behind the indexer-proj region (~47us) -> near-optimal; the result
lands at ~indexer-alone time. cos=1.00000 serial-vs-parallel at every M=1..64; no CUDA-graph capture
deadlock under the nested-stream pattern.

Scope: DSA F-layers only (where the indexer runs). On S-layers (skip_topk) pre_indexer_proj returns dead
buffers -> no GEMMs -> nothing to overlap (q_b kept serial, byte-identical). Short-MHA path also serial.
With index_topk_freq=4 (~15 F-layers of 61): ~8us/F-layer x ~15 ~= 120us/step ~= 0.6% tok/s/user. Real,
clean, deployable.

Impl: `tensorrt_llm/_torch/modules/attention.py` — MLA.__init__ allocates `self.dsa_qb_stream` +
`self.dsa_qb_events` (gated by `mqa is not None` and env); `forward_dsa_proj` overlaps via
`maybe_execute_in_parallel(pre_indexer_proj [default], q_b_proj [dsa_qb_stream])`. Kill switch
`TRTLLM_OPTRT_DSA_QB_OVERLAP=0`. Commit b4acac8e on op-trt-pde-missed2.
Harness: missed2/qb_indexer_overlap.py, missed2/nested_overlap.py.

### WIN 2 (IMPLEMENTED, log-only, zero numerical change): gate per-step SMC handoff logger.info

`smc.py` `process_static_draft_outputs` emitted an unconditional 5-field f-string `logger.info` per
request per draft-commit on the SMC decode hot path (fires even single-node where the pin fields are
None). At INFO level (usually on in prod) that is host string-build + emit every step under the overlap
scheduler. Gated behind `TRTLLM_OPTRT_SMC_DEBUG` (default off); the fail-closed pin VALIDATION is
unchanged. Commit fe650edc. (Small host-overhead trim on the serve-gated SMC path; not a measured
tok/s number — no draft model on node for direct timing.)

### SERVE-GATED LEVER (rigorous, NOT shipped unverified): fuse the attn-INPUT gated-norm with NVFP4 quant

The prior probe flagged "the attn-input seam is unfused (~1.1x, gate-compute-bound)" but did not pursue
or quantify it. Decomposed here:

- `modeling_deepseekv3.py` layer forward: the PRE-attention `input_gated_norm` (`_maybe_apply_gated_norm`)
  outputs **bf16**; `self_attn -> forward_dsa_proj -> kv_a_proj_with_mqa(hidden_states)` then RE-quantizes
  to swizzled NVFP4 inside its GEMM. The POST-attention path ALREADY fuses gate+quant
  (`_apply_post_attention_gated_norm_quant_dense` -> `cute_lowrank_gate_quant_nvfp4_swizzled`, returns
  (bf16, fp4)) — the INPUT path does not.
- Refuting "gate-compute-bound": standalone fp4_quantize of [M,7168] is ~4.1us (measured), and the
  existing post-attn fused kernel's own benchmark is 4.2us fused vs 7.0us unfused (gate+quant) = **~2.8us
  saved/layer**. This applies to ALL 61 layers (input gate runs every layer) -> ~170us/step ~= **0.8%
  tok/s/user** — LARGER than WIN 1 because it is not F-layer-gated.
- Feasibility PROVEN: `Linear.forward` accepts `Fp4QuantizedTensor` input (linear.py:1391), so
  `kv_a_proj_with_mqa(fp4)` works; the swizzled fused op exists in serving builds and is the same kernel
  the dense-MLP post-attn handoff already uses.
- Complication (why it's a real change, not a one-liner): `forward_dsa_proj` consumes `hidden_states`
  TWICE — `kv_a_proj_with_mqa` (wants fp4) AND `indexer.pre_indexer_proj` (needs bf16 + dynamic amax).
  The fused op returns BOTH (bf16, fp4); the plumbing must thread the fp4 through
  `self_attn.forward -> forward_impl_with_dsa -> forward_dsa_proj` while keeping bf16 for the indexer.

NOT shipped here: the cute lowrank gate kernels (`cute_lowrank_gate_quant_nvfp4*`) are ABSENT from both
direct-test images, so the fused kernel's correctness (cos=1.0) CANNOT be verified on this node, and the
signature change is invasive. Per "a plausible-but-unverified win is not valuable," this is documented as
a high-confidence serve-gated lever (seam + feasibility + win estimate all proven) for implementation +
validation in a real serving build, NOT committed as unverified plumbing.
SEAM: `modeling_deepseekv3.py` DeepseekV3DecoderLayer.forward (input gated norm) + attention.py
forward_dsa_proj kv_a_proj input. NEED: serving build with the cute swizzled gate kernel + a real-model
forward to confirm bit-exactness and the per-layer delta.

### Rigorously-confirmed NEGATIVES this round
- q_b overlap on the SHARED Attention aux stream: 0.97-1.00x (NO win) — contends with the indexer's
  existing internal q/k quant overlap. The dedicated stream is mandatory.
- Extending the overlap to also cover kv_a_proj: no headroom — kv_a_proj is the dependency ROOT that
  feeds BOTH the dense and indexer paths; q_b is already fully hidden behind the indexer region, so the
  Op-1 floor is ~max(kv_a + q_b, kv_a + indexer) which the current overlap already reaches.
- proof image (no-cublaslt) dense NVFP4 GEMM is ~2x the cublaslt-image cost (35-47us vs 15-31us),
  re-confirming the headline "restore cublaslt in serving builds" lever from the other direction.

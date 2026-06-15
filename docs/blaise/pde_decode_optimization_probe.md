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

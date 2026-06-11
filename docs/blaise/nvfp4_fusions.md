# NVFP4 elementwise + quant fusions

Decode on the NVFP4 target is overhead-bound, so removing standalone kernel
launches and HBM round-trips on the per-layer elementwise+quant paths is a
direct win. Six fusions land here:

| # | Fusion | Op / file | Figure | Default |
|---|--------|-----------|--------|---------|
| 13 | add + RMSNorm + NVFP4 quant | `residual_add_norm.py` (torch.compile pattern) | −48…−54 % norm→quant sub-path (113–144 µs/step) | on |
| 13b | shared-expert SwiGLU + FP4-output (decode-M guard lift) | `cute_dsl_custom_ops.py` / `gated_mlp.py` | ~100 µs/step + 58 act-quant launches | on (guard lifted `fd705a6f5`) |
| 13c | lowrank-gate + NVFP4-quant single-launch epilogue (MoE input) | `cute_lowrank_gate.py` / `fused_lowrank_gate.py` | chain 7.04 → 4.19 µs/layer ⇒ −165 µs/tok | on (`68866e061`) |
| 13d | dense-MLP gated-norm + NVFP4-quant handoff (swizzled-SF) | `cute_lowrank_gate.py` / `modeling_deepseekv3.py` | 4 → 3 kernels per dense-layer input; ~6–8 µs/tok | on (`TRTLLM_OPTRT_GATED_PREMLP_QUANT`, `8e44aeae1`) |
| 14 | fused RoPE-cat-FP4 | `fusedRopeCatFp4Op.cpp` / `fusedRopeCatFp4.cu` | −3.2…−4.1 µs / F-layer (graphed) | on when shape matches |
| 15 | KVarN-BDR fold | see `kvarn.md` | (capacity, not latency) | opt-in |

All are validated by **numerical match vs the unfused decomposition** (residual
bit-identical; NVFP4 quant error reported), not end-to-end text.

---

## add + RMSNorm + quant fusion

**Idea.** On every MoE layer the decode graph runs the triple
`aten.add(h, residual)` → `trtllm.flashinfer_rmsnorm` →
`trtllm.fp4_quantize(., ., 16)` as three separate kernels — a residual add, an
RMSNorm, and a standalone NVFP4 quant (with its own swizzled scale-factor
write). The quant launch and the intermediate HBM round-trip between norm and
quant are pure overhead.

**Fix.** A torch.compile pattern that rewrites the triple into the single
`trtllm.fused_add_rms_norm_quant` kernel (residual-add + RMSNorm + NVFP4 quant +
swizzled SF in one launch). No `dsa.py` / `attention.py` hook is needed:
production `RMSNorm(h, residual)` already decomposes under `torch.compile` to
`add + flashinfer_rmsnorm`, and the MoE `fp4_quantize` follows, so the triple
appears on the real decode path unmodified and the pattern matcher folds it.

- **Files (production diff is only 2 files, +117 lines):**
  - `tensorrt_llm/_torch/compilation/patterns/residual_add_norm.py`
    (`register_add_norm_fp4_quant`, +111) — registers the `target_pattern`
    `(input, residual, gamma, sf_scale?, use_rms_norm=True, eps,
    output_hp_norm=False) → 4-tuple` into the `PatternMatcherPass`.
  - `tensorrt_llm/_torch/compilation/backend.py` (+6) — imports and registers
    the pattern **before** the fp8 / bf16 patterns so the FP4 triple is matched
    first.
- **Win:** **saved 113–144 µs per decode step (−48…−54 %)** on the
  norm→quant sub-path (graphed e2e bench `bench_fusion_graphed_e2e`, L = 58
  DeepSeek-V3.2 MoE layers, each side in a CUDA graph). The FP4 pass fires
  **58×/step** (once per MoE layer).
- **Enable:** **on** — it is a `torch.compile` graph pass, so it engages
  automatically whenever the decode graph is compiled and the triple is present.
- **Correctness:** **numerically equivalent** — fused vs unfused dequant to the
  same **7.06–7.13 % NVFP4 error**, and the **residual is bit-identical**.
  Build-verified: target op `trtllm.fused_add_rms_norm_quant` exists with the
  matching signature; decode-shaped (N = 8) `torch.compile` fire test shows
  `match_count = 1`, the fused op in the POST graph, and **0 remaining**
  standalone `fp4_quantize` nodes.
- **Composes with:** WarpDecode (`warpdecode.md`) — both operate on the MoE
  decode path but at different stages (this fuses the *pre-MoE* norm+quant;
  WarpDecode is the *expert* GEMM path). The pattern fires per MoE layer
  regardless of which MoE backend runs.

## shared-expert SwiGLU + FP4-output at decode M (guard lift)

**Idea.** The shared-expert GEMM chain can emit its SwiGLU output directly as
NVFP4 (FP4 codes + swizzled SF) in the GEMM epilogue, removing the standalone
`swiglu bf16 → fp4_quantize` pair. The fusion existed but was gated to
**m ≥ 128** (`_FP4OUT_MIN_M`), so decode (m = 4..16) never took it.

**Fix (`fd705a6f5`).** The guard's OOB claim ("SFC epilogue does not predicate
writes when m < CTA tile height") is **false for the production call path**:
`forward()` sizes C to `pad_up(m, cta_m)` rows and SFC to
`pad_up(padded_m, 128)`, which covers every full-tile and cluster-spill write;
small-m partial tiles are the same code path as the last partial tile of any
`m % cta_m != 0` prefill shape. Guard removed; decode M takes the fusion.

- **Files:** `tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py`,
  `tensorrt_llm/_torch/modules/gated_mlp.py`.
- **Win:** 1.7–1.9 µs/layer at m = 4/16 ⇒ **~100 µs/step across 58 MoE layers,
  plus 58 act-quant launches/step removed**.
- **Enable:** on (the guard is gone; the fusion engages at every m).
- **Correctness (B200, real REAP shared-expert weights, GPU driver run):**
  **cos(fused vs TRUE-f32 reference of the same math) = 1.0 with
  max_abs = 0.0 at every m ∈ {1,4,16,64,128}**; fused-vs-unfused 0.9998+ (the
  delta is the unfused chain's own extra quantize step). **OOB demo:** direct
  kernel launches with TIGHT (unpadded beyond contract) C/SFC allocations
  across tilers {(128,128),(256,128)} × clusters at all five m — no fault; the
  guard's claimed failure mode does not reproduce.
- **Composes with:** WarpDecode (`warpdecode.md` — shared-expert vs routed-
  expert paths are independent), #13 (different fusion sites on the same MoE
  layer).

## lowrank-gate + NVFP4-quant single-launch epilogue (MoE input)

**Idea.** The routed-MoE input path ran the REAP lowrank gated-norm and then a
standalone `fp4_quantize` of the gated output (the in-tree handoff was a
2-launch split-K Triton pair feeding a separate quant). The gate kernel
already touches every output element last — the NVFP4 quantization can ride
its epilogue.

**Fix (`68866e061`).** `apply_fused_lowrank_gate_quant_nvfp4`: the existing
single-launch CuTe cluster lowrank-gate kernel
([optimization_candidates.md](optimization_candidates.md) G1) gains an NVFP4
epilogue — e2m1 packing via the same `cvt.rn.satfinite.e2m1x2.f32` hardware
instruction as `quantization.cuh::fp32_vec_to_e2m1` (LINEAR SF layout), e4m3
block scales via `cvt.rn.satfinite.e4m3x2.f32` with the exact-arithmetic
zero-block guard mirroring the Triton epilogue, 16-element blocks pairing
adjacent lanes (one butterfly shuffle for the block amax; CTA K-slices proven
never to split a block at the supported shapes). The gated bf16 plus a
LINEAR-SF `Fp4QuantizedTensor` go straight into the MoE, which skips its own
`fp4_quantize`.

- **Files:** `tensorrt_llm/_torch/modules/cute_lowrank_gate.py`,
  `tensorrt_llm/_torch/modules/fused_lowrank_gate.py` (dispatch),
  `tensorrt_llm/_torch/models/modeling_deepseekv3.py` (handoff).
- **Win:** chain cost 7.04 → 4.19 µs (M=4) / 7.17 → 4.48 µs (M=16) per MoE
  layer ⇒ **−165 µs/token at M=4 across 58 MoE layers** (−254 µs/token vs the
  pre-lowrank-gate unfused chain).
- **Enable:** on (default impl of the gate+quant handoff). WarpDecode is a
  mode of `CuteDslFusedMoE`, so the handoff engages under the production
  pure-TP decode plan.
- **Correctness:** y bitmatch vs eager 99.99988 % of elements with y cosine
  vs a **true-f32 reference** min 0.9999971; **fp4 codes + scales bit-exact
  vs `trtllm.fp4_quantize` on the same y** (layers 5/20/50,
  M ∈ {1,4,16,256}, through the production dispatcher with real `nn.Linear`
  modules); top-8 routing overlap 1.0 vs the eager-input path; CUDA-graph
  capture + replay bit-exact. The fp4-vs-bf16 quantization noise floor is
  shared with the existing production path, not added by this kernel.
- **Note:** full absorb-into-AR-quant (quantizing inside the allreduce
  epilogue instead) is blocked by layout — the AR NVFP4 epilogues emit
  SWIZZLED SF only while the MoE permute path requires LINEAR. The
  constant-0.5 gate ABSORB variant was measured and rejected on routing
  accuracy (optimization_candidates.md G2).
- **Composes with:** #13 (different fusion site on the same layer: #13 is the
  post-AR residual+norm+quant, this is the gated-branch MoE input), WarpDecode
  (consumes the `Fp4QuantizedTensor` natively), #13d (the same kernel family
  with the swizzled-SF epilogue for the dense-MLP consumer).

## dense-MLP gated-norm + NVFP4-quant handoff (swizzled-SF)

**Idea.** #13c gives the routed-MoE input a single-launch gate+quant whose
SF output is LINEAR (the MoE permute path requires LINEAR). The dense-MLP
layers (0–2) run the same lowrank gated-norm, but their `gate_up_proj` is a
plain NVFP4 Linear whose quantized-activation path expects the **swizzled**
(tiled) SF layout — so the gate output still paid a separate in-MLP
`fp4_quantize` launch.

**Fix (`8e44aeae1`).** A swizzled-SF epilogue variant of the cute
lowrank-gate+quant kernel — new op `cute_lowrank_gate_quant_nvfp4_swizzled`,
the SF store routed through the `computeSFIndex` tiled layout — feeds
`gate_up_proj` a ready `Fp4QuantizedTensor`, replacing gate + in-MLP
`fp4_quantize`: **4 → 3 kernels per dense-layer input chain**. Two
alternatives were ruled out: (b)-direct (have the Linear consume LINEAR SF)
is **dead by code** — `NVFP4LinearMethod` has no linear-SF activation path
and cuBLASLt needs the tiled layout; (b)+interleave (LINEAR SF + a separate
interleave launch) is launch-neutral.

- **Files:** `tensorrt_llm/_torch/modules/cute_lowrank_gate.py` (the
  swizzled epilogue + op), `tensorrt_llm/_torch/modules/fused_lowrank_gate.py`
  (dispatch), `tensorrt_llm/_torch/models/modeling_deepseekv3.py` (the
  dense-MLP handoff).
- **Win:** ~2.0–2.7 µs/layer × 3 dense layers ≈ **6–8 µs/token** plus one
  launch per layer — a launch-count / composability win (confirmed on the
  composite re-profile: norm/rope/quant −7.7 µs vs the isolation leg).
- **Enable:** on (`TRTLLM_OPTRT_GATED_PREMLP_QUANT=1` default; `0` restores
  the unfused chain).
- **Correctness:** y / fp4 codes / SF / **GEMM output all bit-exact vs the
  unfused chain** at M ∈ {4,16} (block cosine 1.0) — the swizzled SF store
  is layout-only; the quantization math is #13c's.
- **Composes with:** #13c (LINEAR vs swizzled SF epilogue selected by the
  consumer), B1 (the cuBLASLt-forced `gate_up_proj` consumes the tiled SF
  natively), #13 (different site: #13 is post-AR, this is the dense-MLP
  gated branch).

## fused RoPE-cat-FP4

**Idea.** The indexer-K projection runs RoPE (a standalone flashinfer launch),
writes the BF16 `q_pe`/`k_pe` back to HBM, reloads them, concatenates the
positional (`pe`) and non-positional (`nope`) parts, and NVFP4-quantizes the
result. The standalone RoPE launch plus its BF16 write-back + reload are
overhead that can be folded into the cat+quant.

**Fix.** A fused CUDA op `trtllm::fused_rope_cat_fp4(pe, nope, cos_sin, pos)`
that applies RoPE, concatenates, and NVFP4-quantizes in one kernel — folding the
standalone flashinfer RoPE launch and its BF16 round-trip into `fused_cat_fp4`.

**Eligibility (`_rope_cat_fuse_ok`).** The fused kernel requires neox RoPE,
`head_dim == 128`, `rope_dim == 64` (even, with `rope_dim % 4 == 0` and
`(rope_dim // 2) % 4 == 0`), no indexer-rope-interleave, and the
flashinfer-compatible cos/sin cache (cos first half, sin second). When the shape
does not match, the path falls back to the standalone RoPE + cat + quant.

- **Files:**
  `cpp/tensorrt_llm/thop/fusedRopeCatFp4Op.cpp`,
  `cpp/tensorrt_llm/kernels/fusedRopeCatFp4.cu`; JIT/torch fallback +
  registration in `tensorrt_llm/_torch/attention_backend/sparse/dsa.py`
  (`_ensure_fused_rope_cat_fp4_op`, the `_rope_cat_fuse_ok` gate at dsa.py:1840,
  `_rope_cat_cos_sin` cache).
- **Win:** **−3.2 … −4.1 µs per F-layer (graphed)** — over ~15 F-layers this is
  ~50–60 µs/token.
- **Enable:** **on when the shape matches** (`_rope_cat_fuse_ok` true — the
  DeepSeek-V3.2 indexer shape qualifies: `use_fp4`, head_dim 128, rope_dim 64).
  A transparent JIT fallback builds the op via `load_inline` if the AOT C++ op
  is not compiled in; if even that is unavailable the standalone path runs.
- **Correctness:** validated against the standalone RoPE + cat + FP4-quant
  reference (the fused op reproduces the same FP4 codes + swizzled SF). The
  `use_fp4` field is hoisted above `_rope_cat_fuse_ok` to avoid a
  use-before-assign (`9208440d`).
- **Composes with:** the Indexer (this is the indexer-K projection's quant
  step), the native FP4 scoring path (indexer.md #6) that consumes the quantized
  indexer-K.

## KVarN-BDR fold

The KVarN variance-normalized KV quant folds its block-diagonal-rotation (BDR)
dequant into the read path / the add+RMSNorm path rather than running a
standalone dequant launch. Because it is part of the KV-cache capacity story
(not a decode-latency fusion per se), it is documented in full in
[`kvarn.md`](kvarn.md#bdr-fold-in-kernel-dequant-on-read). It is **opt-in**
(KVarN flag) and composes with the add+RMSNorm fusion above (the dequant can ride
the same kernel that produces the normed activation).

---

## Enabling the fusions

- **add+RMSNorm+quant (#13):** on automatically under `torch.compile` (no
  config). Verify it fired by checking the POST graph for
  `trtllm.fused_add_rms_norm_quant` with 0 remaining standalone `fp4_quantize`.
- **shared-expert SwiGLU+FP4-out (#13b):** on at every m (the `_FP4OUT_MIN_M`
  guard was lifted in `fd705a6f5`).
- **lowrank-gate+quant epilogue (#13c):** on (default impl of the MoE-input
  gate+quant handoff).
- **dense-MLP gate+quant handoff (#13d):** on
  (`TRTLLM_OPTRT_GATED_PREMLP_QUANT=1` default; `0` = unfused chain).
- **fused RoPE-cat-FP4 (#14):** on automatically when `_rope_cat_fuse_ok` is
  true for the model shape (DeepSeek-V3.2 NVFP4 indexer qualifies). No config;
  falls back transparently otherwise.
- **KVarN-BDR fold (#15):** opt-in via the KVarN flag (see `kvarn.md`).

## Correctness validation summary

| Fusion | Method | Result |
|--------|--------|--------|
| add + RMSNorm + quant | fused vs unfused dequant | 7.06–7.13 % NVFP4 error (same), residual bit-identical, match_count=1 |
| SwiGLU + FP4-out @ decode M | fused vs TRUE-f32 reference + OOB demo | cos 1.0, max_abs 0.0 at every m ∈ {1,4,16,64,128}; no fault on tight allocations |
| lowrank-gate + quant epilogue (#13c) | y vs true-f32 ref; fp4+SF vs `trtllm.fp4_quantize` | y cos ≥ 0.9999971; codes+scales bit-exact; routing overlap 1.0; graph replay bit-exact |
| dense-MLP gate+quant handoff (#13d) | fused vs unfused chain through the real `gate_up_proj` | y / fp4 / SF / GEMM output **bit-exact** at M ∈ {4,16} (block cosine 1.0) |
| fused RoPE-cat-FP4 | vs standalone RoPE+cat+quant | same FP4 codes + swizzled SF |
| KVarN-BDR fold | see kvarn.md | cos 0.992–1.0 (see kvarn.md) |

## Composition with the rest of the campaign

- **WarpDecode** (`warpdecode.md`): #13 fuses the pre-MoE norm+quant; #13b the
  shared-expert output quant; WarpDecode is the routed-expert GEMM. Independent
  stages of the MoE decode path.
- **Indexer** (`indexer.md`): #14 is the indexer-K projection's quant; it feeds
  the native FP4 scoring kernel.
- **KVarN** (`kvarn.md`): #15 is the KVarN dequant folded into the read /
  norm path.
- **CUDA graph:** #13 is a compile-time graph pass (fully graph-native); #14 is
  a single op with stable buffers, captured cleanly.

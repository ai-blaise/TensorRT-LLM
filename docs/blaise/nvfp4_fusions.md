# NVFP4 elementwise + quant fusions

Decode on the NVFP4 target is overhead-bound, so removing standalone kernel
launches and HBM round-trips on the per-layer elementwise+quant paths is a
direct win. Four fusions land here:

| # | Fusion | Op / file | Figure | Default |
|---|--------|-----------|--------|---------|
| 13 | add + RMSNorm + NVFP4 quant | `residual_add_norm.py` (torch.compile pattern) | −48…−54 % norm→quant sub-path (113–144 µs/step) | on |
| 13b | shared-expert SwiGLU + FP4-output (decode-M guard lift) | `cute_dsl_custom_ops.py` / `gated_mlp.py` | ~100 µs/step + 58 act-quant launches | on (guard lifted `fd705a6f5`) |
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
- **fused RoPE-cat-FP4 (#14):** on automatically when `_rope_cat_fuse_ok` is
  true for the model shape (DeepSeek-V3.2 NVFP4 indexer qualifies). No config;
  falls back transparently otherwise.
- **KVarN-BDR fold (#15):** opt-in via the KVarN flag (see `kvarn.md`).

## Correctness validation summary

| Fusion | Method | Result |
|--------|--------|--------|
| add + RMSNorm + quant | fused vs unfused dequant | 7.06–7.13 % NVFP4 error (same), residual bit-identical, match_count=1 |
| SwiGLU + FP4-out @ decode M | fused vs TRUE-f32 reference + OOB demo | cos 1.0, max_abs 0.0 at every m ∈ {1,4,16,64,128}; no fault on tight allocations |
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

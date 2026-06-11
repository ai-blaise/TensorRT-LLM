# DSA Indexer optimizations

The **DSA Indexer** is the lightning-attention pre-filter that selects, per
decode token, which KV positions the sparse-MLA attention will actually attend
to (the top-`index_topk` by indexer logit). On the DeepSeek-V3.2 NVFP4 decode
path the Indexer **was ~50–74 % of TPOT at campaign start** — by far the
biggest single lever — so this is where the campaign spent the most effort.
That premise is now **spent**: after the wins below (plus fp16 logits and the
C++ prod top-k routing), the 2026-06-11 composite re-profile (fixed image,
full stack default-on) measures the Indexer at **1.3 % of the decode step
(−76.7 % vs the 06-10 baseline; HISA ≈ 0.7 %)** — and that 1.3 % already
includes the ~280 µs/step of NEW real work the input_scale remediation
restored (the old all-zero indexer projections cost nothing). The
previously-reported ≈ 4 % included an "indexer FSSS cub select" slice (~3 %)
that was a **misattribution**: those cub kernels were KVarN's eager
decode-restore host-path, fixed in `0a1504755` (see the methodology section
in [optimization_candidates.md](optimization_candidates.md)). See that doc
for where the open levers moved (MoE a2a, expert-GEMM megakernel,
dense-proj batching).

The Indexer runs once per "F" (full-compute) layer. Its decode step is:

```
pre_indexer_proj (q/k proj + RoPE)                 -> indexer q,k
  → paged-MQA-logits  (q · cached k)               -> logits [rows, kv_len]
    → top-k over logits                            -> topk_indices [rows, index_topk]
      → (optional) HISA block pre-select
        → sparse_mla_decode reads only topk slots
```

Seven composable wins attack this path. They all live in
`tensorrt_llm/_torch/attention_backend/sparse/dsa.py` (the Python driver) plus
dedicated C++/CuTe kernels. All are validated by **top-k SET match** against a
torch reference (the Indexer's only externally-visible contract is *which*
positions it selects), not by end-to-end text.

| # | Win | Kernel / file | Figure @ c16 | Default |
|---|-----|---------------|--------------|---------|
| 1 | Width-correct decode logits buffer | `dsa.py`, `llm_args.py` | 2.08–2.30× top-k | opt-in `seq_len_threshold` |
| 2 | Adaptive per-row final-sort | `indexerTopK.cu` | 1.58× (~2× at kv≤8k) | on (graph-safe) |
| 3 | Fused cross-step recency-patch op | `indexerXstepRecencyPatch.cu` | 20× (~104→~5 µs) | on when reuse engaged |
| 4 | Cross-step IndexCache reuse | `dsa.py` | −39/60/68 % @ stride 2/4/8 | opt-in `index_topk_freq` |
| 5 | In-graph metadata / sync-free decode | `dsa.py` | −48 % TPOT (consolidated) | on |
| 6 | Native C++ / CuTe-DSL top-k dispatch | `dsa.py`, `indexerTopK.cu`, `*_paged_mqa_logits.py` | C++ ~1.7× @ prod live-kv; DSL at kv ≥ 16k | auto by live kv_len |
| 7 | fp16 indexer logits | `dsa.py`, `llm_args.py` (`indexer_logits_dtype`) | top-k −15…−22 % @ kv ≥ 33k; buffer halved | on (`auto` → fp16 on the DSL path) |

> **HISA candidate-path correctness (2026-06-11):** two live
> selection-corruption bugs in the HISA decode candidate pipeline —
> pad-poisoning (−1-padded `top_blocks` aliasing page 0) and a row-padded
> logits buffer indexed flat — were fixed in `31e0b5be7`; see the HISA
> hazard section in
> [optimization_candidates.md](optimization_candidates.md). Any HISA-path
> **selection-quality** observation taken through pre-fix kernels is
> suspect (timing observations stand). The AOT kernel parts land with the
> next full-source image build; the python fallbacks are live.

---

## Width-correct decode logits

**Problem.** The decode top-k kernel was allocated and launched over a logits
buffer sized to the *static* maximum model length (`max_model_len`, e.g.
132 096 columns), not the *actual* per-step KV length. The top-k histogram and
the launch grid were sized for the worst case on every step, so short-context
decode (the common case) paid full-width cost.

**Fix.** Size the decode logits buffer and the top-k launch to the real
`num_cols` (the live `kv_len`) instead of the static max, gated by a
`seq_len_threshold` config knob so the narrowing only engages below the
threshold where it is a strict win and never changes the captured CUDA-graph
shape above it. The `warmup_heuristic_topk_decode(top_k, hint_size, num_cols)`
helper (`dsa.py:130`) caches the warmed kernel per `(device, top_k, hint_size,
num_cols)` key so repeated Indexer constructions at the same width are
short-circuited.

- **Files:** `tensorrt_llm/_torch/attention_backend/sparse/dsa.py` (logits
  buffer width + warmup), `tensorrt_llm/llmapi/llm_args.py`
  (`seq_len_threshold` field on the sparse-attention config).
- **Win:** **2.08–2.30× on the top-k kernel** at production widths.
- **Enable:** opt-in via `sparse_attention_config.seq_len_threshold`. Off by
  default so the static-width graph capture is preserved unless the deployment
  opts into the narrow-width path.
- **Correctness:** identical top-k SET — the buffer is the same data, only its
  declared width changes; the histogram scan was already length-bounded.
- **Composes with:** the adaptive final-sort (next section) — width-correct
  cuts the *scan* domain, adaptive-sort cuts the *tie-break* domain; they stack.
  `_DSL_TOPK_MIN_COLS` was first demoted to a buffer-bucketing constant (the
  width-override fell in `841f9874a`; dispatch is live-kv-only, see #6) and
  then **removed dead in `8e44aeae1`** (the width-bucket helper survives only
  as a graph-safety bound).

> **Status (closed `8e44aeae1`):** the *width* side of this win is **spent**
> — nothing in the decode scoring/top-k pipeline is width-dependent anymore
> (the FP4 DSL scorer walks ceil(kv/block) per row with width as a runtime
> stride; the C++ top-k walks live kv). Direct measurement at kv = 4 608:
> pipeline p50 **15.39–15.42 µs for ALL widths {8k..132k}**, logits [0, kv)
> bitwise identical — width-bucketing saves 0.0 µs (the 2.08–2.30× above was
> real against the pre-`841f9874a` width-dependent pipeline). What still
> matters is the **band capture**: graphs warmed at or below `index_topk`
> capture the indexer-FREE path. The r16-era auto-default
> `seq_len_threshold=8192` forfeited that (the short-band graph warmed at
> 8191 → indexer kernels captured → ultra-short decodes paid ~15.4 µs × 61
> layers ≈ 0.94 ms/step they used to skip) — `8e44aeae1` reverts the
> auto-default to `index_topk` and keeps the explicit knob as an operator
> escape hatch. See optimization_candidates.md I3.

## Adaptive per-row final-sort

**Problem (the ~15 µs top-k floor).** The host dispatcher in
`invokeIndexerTopKDecode` selects the final tie-break sort (radix vs insertion)
from `numColumns` = static `max_model_len` (132 096). Production therefore
*always* gets the radix path — which sorts a fixed 2 048 slots (~14 µs) — even
when the real per-row KV (`seqLens`) is short (~4 608, where insertion over the
small candidate set is ~7 µs). The histogram scan is already `seqLens`-bounded;
only the final sort was paying the static-width tax.

**Fix.** A new `kAdaptiveFinalSort` template parameter threaded through
`topKPerRowJob` / `topKPerRowDecode`. When set, each row picks its final-sort
scheme **at runtime** from the on-device scanned `rowLen` (`useRadixFinalSort =
rowLen >= kRowLenRadixThreshold` with `kRowLenRadixThreshold = 12288`),
independent of the host-chosen `numColumns` scheme. A radix-launched kernel thus
behaves like the insertion path whenever the real KV is short.

- **File:** `cpp/tensorrt_llm/kernels/indexerTopK.cu`
  (`topKPerRowJob`, `topKPerRowDecode`, `invokeIndexerTopKDecode`); probe
  `blaise_perf/indexer_floor/topk_scheme_probe.py`.
- **Win:** **6.9 µs vs 14 µs at kv = 4 608, B = 32** (~2× the top-k kernel;
  ~1.58× in the consolidated bf16/fp16 dispatcher); ~7 µs saved per F-layer →
  ~100 µs/token over 15 F-layers.
- **Enable:** on. Extended from fp32 to the **bf16 / fp16** top-k dispatchers
  in a follow-up (`2b1089f5`) so all decode dtypes get parity.
- **Correctness:** **same histogram, same top-k SET** — only the
  intra-threshold-bin (tie-break) ordering path differs. fp32 correctness
  preserved across recompiles; bf16/fp16 reach parity with fp32.
- **Graph-safety:** fully CUDA-graph-safe — no host-side shape or launch-config
  change. The scheme switch is a pure device-side branch on a scanned length.
- **Composes with:** width-correct logits (orthogonal domains, stack), the
  native top-k dispatch (#6) which selects *this* kernel below the C++ threshold.

## Fused cross-step recency-patch

**Problem.** With cross-step IndexCache reuse (#4), on a *reuse* decode step we
keep the previous step's top-k instead of recomputing the logits-MQA + top-k.
But the cached top-k must still be **patched** so the KV positions appended
since the last refresh become selectable. The reference implementation
(`Indexer._xstep_reuse_decode`'s recency branch) does this with ~20 dependent,
launch-bound PyTorch tensor ops — ~104 µs of pure launch latency on the decode
critical path.

**Fix.** A single fused CUDA op, `trtllm::indexer_xstep_recency_patch`, that
patches the cached `[numRows, indexTopK]` int32 top-k buffer **in place** with
one launch (one block per cached top-k row). Per row `r` it computes
`curEnd = curKvLens[batch] - nextN + offset + 1`,
`delta = clamp(curEnd - refreshEnd[r], 0, maxDelta)`, and for each trailing
column writes `refreshEnd[r] + delta - 1 - c` when `c < delta`, else leaves the
cached value untouched — reproducing the PyTorch reference's column placement,
descending order, and clamp semantics exactly.

- **Files:**
  `cpp/tensorrt_llm/kernels/IndexerXstepRecencyPatch.h`,
  `cpp/tensorrt_llm/kernels/indexerXstepRecencyPatch.cu`,
  `cpp/tensorrt_llm/thop/indexerXstepRecencyPatchOp.cpp` (THOP op + CMake +
  fake-tensor meta registration); wired into the `_xstep_reuse_decode` recency
  path in `dsa.py`.
- **Win:** **~104 µs → ~5 µs, 20×** on the reuse-step recency patch (isolated
  microbench).
- **Enable:** engaged automatically whenever cross-step IndexCache reuse is on
  (i.e. `index_topk_freq > 1`) and the step is a reuse ("S") step. A transparent
  JIT/torch fallback exists if the AOT C++ op is not built.
- **Correctness:** **exact-equal** to the PyTorch reference (Jaccard = 1.0,
  bit-identical column placement) across the validated shapes.
- **Composes with:** cross-step IndexCache reuse (#4) — this op *is* the patch
  primitive that makes reuse correct without recompute; it has no effect on
  full-compute ("F") steps.

## Cross-step IndexCache reuse

**Problem.** The Indexer recomputes logits-MQA + top-k every decode step, but
across adjacent steps the selected top-k set is nearly stationary (only the few
newest positions change). Recomputing the full top-k every step is the dominant
Indexer cost.

**Fix.** Compute a fresh top-k only every `index_topk_freq` steps (the refresh
or "F" step); on the intervening reuse ("S") steps, restore the cached top-k
from the IndexCache and apply the fused recency patch (#3) so newly-appended
positions remain selectable. `_should_reuse_previous_topk()` decides per step;
`_get_indexcache_topk()` / `_maybe_store_indexcache_topk()` manage the cache;
the owning "F" layer for each reuse group is the one that writes the cache, so a
reuse layer that misses the cache fails fast rather than scoring uninitialized
buffers.

- **File:** `tensorrt_llm/_torch/attention_backend/sparse/dsa.py`
  (`_should_reuse_previous_topk`, `_get_indexcache_topk`,
  `_maybe_store_indexcache_topk`, `index_topk_freq`).
- **Win:** **−39 % / −60 % / −68 % TPOT at stride 2 / 4 / 8** (i.e.
  `index_topk_freq = 2 / 4 / 8`).
- **Enable:** opt-in, frozen. Set `sparse_attention_config.index_topk_freq` (1 =
  off / recompute every step; the default). Higher stride trades a small recall
  drift for larger savings.
- **Correctness:** top-k SET match within the recency window — the cached set
  plus the recency patch reproduces the exact positions the fresh top-k would
  select for the recency columns; older columns are stationary by construction.
- **Composes with:** the fused recency-patch op (#3, the patch primitive), the
  in-graph metadata (#5, supplies the per-step `curKvLens` / `refreshEnd` on
  device so the reuse path needs no d2h sync).

## In-graph metadata / sync-free decode

**Problem.** The Indexer decode path issued ~29 `.item()` / d2h
syncs per step (KV lengths, block counts, offsets) plus a dead HISA-logits
decode call. Each d2h sync stalls the launch pipeline; on an overhead-bound
decode step they dominate.

**Fix.** Three consolidated changes, landed as the R2 "syncfree + cdslswitch +
floor" set:
1. Keep decode metadata **on device** (in-graph) — derive `rowLen`, block
   ranges, and refresh ends from device tensors (`seqLens`, `kv_lens`,
   `block_table`) instead of pulling scalars to host.
2. Gate the **dead HISA-logits decode call** + its d2h sync behind
   `_should_use_hisa_logits()` (which returns `False` at prod) so eager mode
   stops paying for it.
3. Route the candidate-score path through the native fp8/fp4 kernel (see #6).

- **File:** `tensorrt_llm/_torch/attention_backend/sparse/dsa.py`
  (the `_should_use_hisa*` guards, device-tensor metadata derivation).
- **Win:** **decode TPOT ≈ −48 %** consolidated (R2); the dead-HISA-gate alone
  is **−67 % eager** on its sub-path.
- **Enable:** on. The guards are the production defaults
  (`_should_use_hisa_logits` → `False` — that *logits* path was dead). Note the
  HISA candidate gate itself is a different thing and is **not** off at prod:
  it now tracks live kv via `metadata.max_gen_kv_len` (`8db5cd77f`, which also
  fixed wrong selection at kv > 33k) and HISA wins whenever the Indexer is
  active — see optimization_candidates.md cycle 4.
- **Correctness:** the metadata is the same values, computed on device instead
  of host — top-k SET unchanged. The gated HISA-logits path was dead (its
  outputs were never consumed on the decode path).
- **Extension (`3e03d665d`):** the **HISA per-step invariant memo** —
  row_to_batch / prefix_lens / block-counts / gather indices are per-*step*
  invariants but were recomputed per *layer* (61×); now computed once per step
  under a capture-aware key. Part of the G1 glue lane.
- **Composes with:** cross-step reuse (#4, consumes the on-device `curKvLens` /
  `refreshEnd`), CUDA-graph capture (in-graph metadata is what *allows* the
  Indexer decode step to be captured — host scalars would break replay).

## Native C++ / CuTe-DSL top-k dispatch

**Problem.** Two dispatch decisions on the Indexer's hot path: (a) which top-k
kernel to use as a function of `kv_len`, and (b) which paged-MQA-logits scoring
kernel (native fp8/fp4 vs CuTe-DSL).

**Fix.**
- **Top-k (updated `841f9874a`):** route decode top-k by **live `kv_len` only**
  — C++ `indexerTopK` below `_DSL_TOPK_MIN_KV_LEN` (now **16384**, lowered from
  32768 by the fp16-logits win #7), the split-work / DSL path above it. The
  dispatch previously OR'd in a *width* gate (force DSL whenever the padded
  logits width ≥ `_DSL_TOPK_MIN_COLS`=12288), which at the prod width 132096
  fired unconditionally and overrode the kv gate. That width gate's premise
  was **measured false** (3-seed, CUDA-graph replay, true head-to-head): the
  C++ kernel is **~width-independent (~11 µs flat)** — it walks only
  `[0, live_kv)` per row — so at prod (live kv ~4.6k) **C++ is ~1.7× faster
  with a bit-identical selected set** (top-1024 recall 1.0); ~29–33 % off the
  top-k pipeline. The fused single-pass cluster DSL top-k (`81cfeb88c`,
  1.28–1.59× over the 2-pass DSL form, IoU 1.0000) now serves only genuinely
  long live kv. Separately, `use_cute_dsl_topk` remains opt-in and
  net-negative at small batch — measured a **net loss at ≤ b32**.
- **Scoring:** the native **fp8_fp4 candidate-score** kernel is already ahead of
  the SGLang WMMA path. `cute_dsl_fp4_paged_mqa_logits` was validated as a
  **proven floor** (a correctness/perf lower bound, not a kernel win), so the
  native path stays default.

- **Files:** `tensorrt_llm/_torch/attention_backend/sparse/dsa.py` (dispatch:
  `_DSL_TOPK_MIN_KV_LEN`, `use_cute_dsl_topk`), `cpp/tensorrt_llm/kernels/indexerTopK.cu`,
  `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/paged_mqa_logits/{fp4,fp8,bf16}_paged_mqa_logits.py`.
- **Win:** C++ is the prod path (**~1.7× at the prod operating point**, width
  132096 / live kv ~4.6k); the DSL floor is proven and owns long live kv.
- **Enable:** automatic — selected by **live** `kv_len`
  (`metadata.max_gen_kv_len`), no width clause. `use_cute_dsl_topk` is opt-in
  and intentionally *not* default (net-loss at small batch).
- **Correctness:** all paths validated by top-k SET match (the `841f9874a`
  re-route is bit-identical, recall 1.0); the DSL path is a proven floor with
  FP4 logits re-verified at B ≥ 64.
- **Composes with:** adaptive final-sort (#2, the C++ kernel this dispatch
  selects), width-correct logits (#1, the width the selected kernel runs at),
  fp16 logits (#7, which shifts the crossover), the AB-swapped scoring kernel
  (sparse_mla.md #9, the tcgen05 scoring path).

## fp16 indexer logits

**Problem.** The model is bf16 with 4-bit indexer keys, yet the decode indexer
logits were stored fp32 — the only fp32 element in the scoring→top-k pipeline.
The fp8/fp4-quantized scoring inputs carry far less information than 32 bits,
and the fp32 store costs the DSL top-k **2 extra radix rounds** plus double the
logits HBM traffic.

**Fix (`3e03d665d`).** New `DeepSeekSparseAttentionConfig.indexer_logits_dtype`
(`auto | fp32 | fp16 | bf16`): `auto` resolves to **fp16 on the CuTe-DSL path**
and stays fp32 on the DeepGEMM fallback. Config-matched precision — the logits
dtype now matches what the quantized scores actually carry.

- **Files:** `tensorrt_llm/_torch/attention_backend/sparse/dsa.py`,
  `tensorrt_llm/llmapi/llm_args.py` (`indexer_logits_dtype`).
- **Win:** **top-k −15…−22 % at kv ≥ 33k** (e.g. B16/33k 27.9 → 21.7 µs), zero
  short-kv regression, logits buffer halved. Also lowered the C++→DSL
  crossover `_DSL_TOPK_MIN_KV_LEN` 32768 → 16384 (#6).
- **Enable:** on (`auto`). Set `indexer_logits_dtype=fp32` to restore the old
  behavior.
- **Correctness:** top-1024 recall vs a **TRUE-f32-scored reference** equals
  the pre-existing fp8-scoring noise floor (|Δ| ≤ 0.0005 — fp16 adds nothing
  above the noise already inherent in fp8 scoring); downstream attention
  cosine 1.000000; max|logit| ~360 vs the 65504 fp16 ceiling (no overflow
  margin concern).
- **Composes with:** the top-k dispatch (#6 — the DSL top-k reads these
  logits; the crossover shift is this win propagating), width-correct logits
  (#1, orthogonal: width vs dtype).

---

## Enabling the Indexer stack (recommended decode config)

```python
sparse_attention_config = {
    "algorithm": "dsa",
    "indexer_mode": "indexcache-hisa",
    "indexer_k_dtype": "fp4",            # FP4 indexer-K (smaller cache, native score path)
    "index_topk_freq": 4,                # #4 cross-step reuse (−60% @ stride 4); 1 = off
    # "indexer_logits_dtype": "auto",    # #7 fp16 logits on the DSL path (default)
    # "seq_len_threshold": <int>,        # #1 width-correct logits (opt-in)
    # "use_cute_dsl_topk": False,        # #6 keep C++ top-k at short live kv (net-loss if True)
}
```

Defaults that are **on** without configuration: adaptive final-sort (#2),
in-graph / sync-free metadata (#5), native top-k+scoring dispatch (#6, live-kv
gated — prod routes to C++), fp16 logits (#7, `auto`), and the
fused recency-patch op (#3, whenever reuse is engaged). The two opt-ins worth
flipping per deployment are `index_topk_freq` (the single biggest Indexer lever)
and `seq_len_threshold` (width-correct logits).

## Correctness validation summary

| Win | Method | Result |
|-----|--------|--------|
| Width-correct logits | top-k SET vs full-width | identical set |
| Adaptive final-sort | top-k SET vs radix path | identical set (only tie-break order differs) |
| Fused recency-patch | exact-equal vs PyTorch ref | Jaccard = 1.0, bit-identical |
| IndexCache reuse | top-k SET within recency window | matches fresh top-k on recency cols |
| In-graph metadata | top-k SET vs host-scalar path | identical set |
| Native top-k/scoring dispatch | top-k SET + FP4 logits floor | identical set (`841f9874a` re-route recall 1.0); floor proven |
| fp16 logits | top-1024 recall vs TRUE-f32-scored ref + downstream cosine | recall at the fp8 noise floor (\|Δ\| ≤ 0.0005); attention cos 1.000000 |

## Composition with the rest of the campaign

- **Sparse-MLA attention** (`sparse_mla.md`) consumes the Indexer's
  `topk_indices`. The Indexer and the attention kernel are pipelined per layer;
  the MSA 2-stream split (#7) overlaps the two attention head groups while the
  Indexer top-k is being produced.
- **KVarN** (`kvarn.md`) stores the dense MLA latent KV that the Indexer scores
  against after dequantization. It is not an Indexer K-cache dtype; Indexer
  storage remains `indexer_k_dtype="fp8"` or `"fp4"`, so the two are independent
  at the selection level.
- **LayerSplit** (`../source/features/layersplit.md`) broadcasts the owner CP
  rank's indexer-K cache slot to peers before the Indexer reads it — the Indexer
  hook is exactly the LayerSplit broadcast point. The Indexer optimizations run
  unchanged on whichever rank owns the layer.
- **CUDA graph:** all six wins are graph-safe by construction (no host-side
  shape/launch change on the captured path). The in-graph metadata (#5) is the
  enabler; width-correct (#1) is gated above its threshold precisely to keep the
  captured shape static.

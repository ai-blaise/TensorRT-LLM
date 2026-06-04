# DSA Indexer optimizations

The **DSA Indexer** is the lightning-attention pre-filter that selects, per
decode token, which KV positions the sparse-MLA attention will actually attend
to (the top-`index_topk` by indexer logit). On the DeepSeek-V3.2 NVFP4 decode
path the Indexer is **~50–74 % of TPOT** — by far the biggest single lever — so
this is where the campaign spent the most effort.

The Indexer runs once per "F" (full-compute) layer. Its decode step is:

```
pre_indexer_proj (q/k proj + RoPE)                 -> indexer q,k
  → paged-MQA-logits  (q · cached k)               -> logits [rows, kv_len]
    → top-k over logits                            -> topk_indices [rows, index_topk]
      → (optional) HISA block pre-select
        → sparse_mla_decode reads only topk slots
```

Six composable wins attack this path. They all live in
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
| 6 | Native C++ / CuTe-DSL top-k dispatch | `dsa.py`, `indexerTopK.cu`, `*_paged_mqa_logits.py` | net-win ≤ b32 | auto by kv_len |

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
  Pairs with `_DSL_TOPK_MIN_COLS` (the CuTe-DSL top-k floor) which gates which
  rows take the DSL path.

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
  (`_should_use_hisa_logits` → `False`; HISA gated by
  `hisa_min_seq_len = 65536`, i.e. off at prod context lengths).
- **Correctness:** the metadata is the same values, computed on device instead
  of host — top-k SET unchanged. The gated HISA-logits path was dead (its
  outputs were never consumed on the decode path).
- **Composes with:** cross-step reuse (#4, consumes the on-device `curKvLens` /
  `refreshEnd`), CUDA-graph capture (in-graph metadata is what *allows* the
  Indexer decode step to be captured — host scalars would break replay).

## Native C++ / CuTe-DSL top-k dispatch

**Problem.** Two dispatch decisions on the Indexer's hot path: (a) which top-k
kernel to use as a function of `kv_len`, and (b) which paged-MQA-logits scoring
kernel (native fp8/fp4 vs CuTe-DSL).

**Fix.**
- **Top-k:** route decode top-k to the **C++ `indexerTopK` kernel below 32 K
  `kv_len`** (where it beats the alternatives) and let longer contexts take the
  split-work / DSL path. The `use_cute_dsl_topk` config selects the CuTe-DSL
  top-k for `num_gen_tokens ≤ 256` when enabled; the campaign measured the DSL
  top-k as a **net loss at ≤ b32** (`use_cute_dsl_topk` net-negative on small
  batch), so the C++ path is the production default below b32.
- **Scoring:** the native **fp8_fp4 candidate-score** kernel is already ahead of
  the SGLang WMMA path. `cute_dsl_fp4_paged_mqa_logits` was validated as a
  **proven floor** (a correctness/perf lower bound, not a kernel win), so the
  native path stays default.

- **Files:** `tensorrt_llm/_torch/attention_backend/sparse/dsa.py` (dispatch:
  `use_cute_dsl_topk`, the `< 32K` route), `cpp/tensorrt_llm/kernels/indexerTopK.cu`,
  `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/paged_mqa_logits/{fp4,fp8,bf16}_paged_mqa_logits.py`.
- **Win:** native C++ top-k is the **net-win path ≤ b32**; the DSL floor is
  proven (closes the question rather than adding a kernel win).
- **Enable:** automatic — selected by `kv_len` and `num_gen_tokens`.
  `use_cute_dsl_topk` is opt-in and intentionally *not* default (net-loss at
  small batch).
- **Correctness:** all paths validated by top-k SET match; the DSL path is a
  proven floor with FP4 logits re-verified at B ≥ 64.
- **Composes with:** adaptive final-sort (#2, the C++ kernel this dispatch
  selects), width-correct logits (#1, the width the selected kernel runs at),
  the AB-swapped scoring kernel (sparse_mla.md #9, the tcgen05 scoring path).

---

## Enabling the Indexer stack (recommended decode config)

```python
sparse_attention_config = {
    "algorithm": "dsa",
    "indexer_mode": "indexcache-hisa",
    "indexer_k_dtype": "fp4",            # FP4 indexer-K (smaller cache, native score path)
    "index_topk_freq": 4,                # #4 cross-step reuse (−60% @ stride 4); 1 = off
    # "seq_len_threshold": <int>,        # #1 width-correct logits (opt-in)
    # "use_cute_dsl_topk": False,        # #6 keep C++ top-k below b32 (net-loss if True)
}
```

Defaults that are **on** without configuration: adaptive final-sort (#2),
in-graph / sync-free metadata (#5), native top-k+scoring dispatch (#6), and the
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
| Native top-k/scoring dispatch | top-k SET + FP4 logits floor | identical set; floor proven |

## Composition with the rest of the campaign

- **Sparse-MLA attention** (`sparse_mla.md`) consumes the Indexer's
  `topk_indices`. The Indexer and the attention kernel are pipelined per layer;
  the MSA 2-stream split (#7) overlaps the two attention head groups while the
  Indexer top-k is being produced.
- **KVarN** (`kvarn.md`) shares the latent KV cache the Indexer scores against;
  the Indexer reads the (dequantized) latent, KVarN changes only how that latent
  is *stored*, so the two are independent at the selection level.
- **LayerSplit** (`../source/features/layersplit.md`) broadcasts the owner CP
  rank's indexer-K cache slot to peers before the Indexer reads it — the Indexer
  hook is exactly the LayerSplit broadcast point. The Indexer optimizations run
  unchanged on whichever rank owns the layer.
- **CUDA graph:** all six wins are graph-safe by construction (no host-side
  shape/launch change on the captured path). The in-graph metadata (#5) is the
  enabler; width-correct (#1) is gated above its threshold precisely to keep the
  captured shape static.

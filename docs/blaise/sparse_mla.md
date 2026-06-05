# Sparse-MLA attention optimizations

The sparse-MLA decode attention kernel consumes the DSA Indexer's `topk_indices`
(see `indexer.md`) and runs MLA attention over **only** the selected KV
positions. Three campaign wins target this stage: a 2-stream head-split that
overlaps the two attention head groups, a parallelized scheduler-metadata
kernel, and the AB-swapped index-scoring kernel (MSA #3) that turned out to be
already-optimal on the tcgen05 TMEM path.

These derive from MiniMax-M3 / Together-AI's MSA (Multi-head Sparse Attention)
kernels mapped onto our DSA + HISA stack (MSA ≈ DSA + HISA with GQA-not-MLA).
All are validated by **partial-O + LSE numerical match** and **top-k SET / score
match** against a torch reference, not end-to-end text.

| # | Win | Kernel / file | Figure @ c16 | Default |
|---|-----|---------------|--------------|---------|
| 7 | MSA 2-stream head-split | `dsa.py` + `multi_stream_utils.py` | −14.5…−19.5 % @ b1–4 | on @ small batch |
| 8 | `get_decoding_sched_meta` parallelization | `get_decoding_sched_meta.cu` | 2.5–3.3× the meta kernel | on |
| 9 | AB-swapped index-scoring (MSA #3) | `fp4_paged_mqa_logits.py` / tcgen05 | already-optimal | on (tcgen05 path) |

---

## MSA 2-stream head-split

**Idea (MiniMax-M3 MSA kernel #1 → our stack).** MLA decode does two
projection/attention sub-flows (the q and k indexer projections, then the two
attention head groups). They are independent, so issuing them on **two CUDA
streams** lets the second overlap the first instead of serializing — a direct
win at small batch where each sub-flow under-fills the GPU and launch/issue
latency dominates.

**Implementation.** `Indexer` / the sparse-MLA module take an
`aux_stream: torch.cuda.Stream`. The hot path calls
`maybe_execute_in_parallel(fn_a, fn_b, event_a, event_b, aux_stream)`
(`tensorrt_llm/_torch/modules/multi_stream_utils.py`) which runs `fn_a` on the
current stream and `fn_b` on `aux_stream`, with CUDA events to join. In
`dsa.py` the q/k indexer projections (and the head-split attention groups) are
the two halves (dsa.py:3848, 3862). The split is thread-local gated by
`do_multi_stream()` / `with_multi_stream(enable)` so it can be turned off
per-region for capture or A/B.

- **Files:** `tensorrt_llm/_torch/attention_backend/sparse/dsa.py`
  (`aux_stream` plumbing, the `maybe_execute_in_parallel` call sites),
  `tensorrt_llm/_torch/modules/multi_stream_utils.py`
  (`maybe_execute_in_parallel`, `do_multi_stream`, `with_multi_stream`).
- **Win:** **−14.5 % … −19.5 % at b = 1–4** (the small-batch regime where the
  two sub-flows under-fill the SMs). The win shrinks as batch grows and the GPU
  saturates; this is why it is a small-batch optimization.
- **Enable:** on at small batch via the `aux_stream` being bound at module
  construction; gated by `do_multi_stream()`. The aux stream is created at model
  load time so it is **CUDA-graph stable**.
- **Correctness:** the two streams compute independent tensors and join on
  events before any consumer reads them — numerically identical to the serial
  path (partial-O + LSE match). No race because the join events serialize the
  consumer against both producers.
- **Composes with:** the Indexer (the 2-stream split overlaps the indexer q/k
  projection with the attention head group while the Indexer top-k is in
  flight); `get_decoding_sched_meta` (#8, which produces the schedule both
  streams consume); the AB-swapped scoring (#9, runs inside one of the streams).
- **Status / next lever:** an extension to 4-way head-split / combine-overlap /
  higher-batch tuning is in progress (campaign ITEM#1a) to push the win past b4.

## `get_decoding_sched_meta` parallelization

**Problem.** Before the sparse-MLA decode kernel runs, a scheduler-metadata
kernel (`get_mla_metadata_kernel`) computes the per-split work assignment
(`DecodingSchedMeta`: which KV ranges each CTA split owns). The original kernel
seeded the schedule **serially on a single lane** with a struct store and an
inner end-state while-loop, then the whole-op decode waited on it.

**Fix.** Split `get_mla_metadata_kernel` into two:
- `get_mla_metadata_kernel_parallel` (**default**): lane-0 does only a minimal
  serial begin-seed (3 shared ints per split + `num_splits`, no struct store, no
  end inner-while), then a **256-thread parallel fill** recomputes each split's
  end-state and writes `DecodingSchedMeta` **direct-to-global**. The
  `extra_topk` / `ku::ceil` phase-1 paths are preserved.
- `get_mla_metadata_kernel_serial` (**fallback**): the verbatim original, used
  only when `base + 3 * num_splits` ints would exceed the **227 KB dynamic-smem
  ceiling**.

- **Files:**
  `cpp/tensorrt_llm/kernels/flashMLA/nvfp4_sparse/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu`
  (+ `.h`, `params.h`); standalone validator `sched_validate.cu`.
- **Win:** **2.5–3.3× on the metadata kernel** —
  - b32 / tk1024: **178 → 55 µs**
  - b64 / tk1024: 253 → 86 µs
  - b256 / tk2048: 786 → 313 µs

  Whole-op decode prediction: b32/tk1024 **292 → ~168 µs (1.74×)**, b64
  466 → ~296 µs.
- **Enable:** on (`_parallel` is the production default; `_serial` only when the
  smem ceiling would be exceeded).
- **Correctness:** **ALL BIT-IDENTICAL on consumed parts** across
  b = {1..256} × topk = {256..2048}, validated by `sched_validate.cu` against
  the exact production direct-to-global path.
- **Composes with:** the sparse-MLA decode kernel it feeds (unchanged consumer —
  the `DecodingSchedMeta` layout is identical), the 2-stream split (#7), and the
  combine kernel (`combine.cu`) that reduces the per-split partial-O + LSE.
- **Status:** landed into production `get_decoding_sched_meta.cu` (campaign
  ITEM#1b, completed). A further launch-fusion (fold the meta kernel into the
  decode launch) was explored on the same item.

## AB-swapped index-scoring (MSA #3)

**Idea (MiniMax-M3 MSA kernel #3 → our stack).** MiniMax's index-scoring kernel
swaps the A/B operand roles in the HMMA so the index-score matmul lands more
efficiently. Mapped onto our paged-MQA-logits scoring kernel.

**Finding: already-optimal on the tcgen05 TMEM path.** Our production FP4
scoring kernel (`fp4_paged_mqa_logits.py`) runs on the SM100 **tcgen05 TMEM**
path, which already expresses the efficient operand layout the AB-swap targets.
A bf16-HMMA rebuild of the AB-swapped kernel was implemented and benchmarked
(`bdde5eb5` / `416dd630`): it **regresses** versus the tcgen05 path, so it is
**not** adopted. The AB-swap idea is therefore "already-optimal" on our hardware
path — the win it captures on a bf16-HMMA GQA kernel is already captured by the
TMEM datapath for FP4 MLA.

- **Files:**
  `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/paged_mqa_logits/fp4_paged_mqa_logits.py`
  (production, tcgen05),
  `.../bf16_paged_mqa_logits.py` (the AB-swapped bf16-HMMA variant, benched but
  not promoted); benches under
  `tests/scripts/cute_dsl_kernels/paged_mqa_logits/` (`bench_fp4_scoring.py`,
  `bench_bf16_scoring.py`).
- **Win:** none to capture — the production tcgen05 FP4 path is already at the
  efficient layout; the bf16-HMMA rebuild **regresses**, so the production path
  is unchanged. (This is a "validated already-optimal" result, recorded so the
  AB-swap is not re-attempted.)
- **Enable:** on by virtue of the FP4 tcgen05 scoring path being the default;
  the bf16-HMMA AB-swap kernel exists for completeness/measurement only.
- **Correctness:** the AB-swapped bf16 kernel was validated for top-k SET match
  + B32 ctx8192 score correctness (`416dd630`) before being measured and
  rejected on perf.
- **Composes with:** the native top-k/scoring dispatch (indexer.md #6), which
  selects the tcgen05 FP4 scoring kernel.
- **Do-not-redo:** **do not** rebuild the bf16-HMMA AB-swap as a perf path — it
  regresses against tcgen05. (Recorded campaign learning.)

---

## Enabling the sparse-MLA stack

The 2-stream split and the parallel scheduler-meta are **on by default** (the
aux stream is bound at construction; `_parallel` is the default kernel). No
config knob is required for the production decode path. The AB-swapped bf16
scoring kernel is present but not selected (tcgen05 FP4 is the default).

To A/B the 2-stream split off (e.g. during graph-capture debugging):

```python
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
with with_multi_stream(False):
    ...  # forces the serial single-stream path for the enclosed region
```

## Correctness validation summary

| Win | Method | Result |
|-----|--------|--------|
| MSA 2-stream head-split | partial-O + LSE vs serial | numerically identical (event-joined) |
| `get_decoding_sched_meta` parallel | `sched_validate.cu` vs serial | bit-identical on consumed parts, b1–256 × tk256–2048 |
| AB-swapped scoring | top-k SET + B32 ctx8192 score | correct, but regresses → not promoted |

## Composition with the rest of the campaign

- **Indexer** (`indexer.md`) produces the `topk_indices` this attention reads;
  the 2-stream split overlaps the indexer projections with the attention head
  group.
- **KVarN** (`kvarn.md`) changes how the dense MLA latent KV is stored; the sparse-MLA
  kernel reads the dequantized latent, so KVarN is transparent to the scheduler
  and the head-split.
- **LayerSplit** (`../source/features/layersplit.md`) broadcasts the owner CP
  rank's KV before this attention runs; the scheduler-meta and head-split run
  per-rank on whichever rank owns the layer.
- **CUDA graph:** the aux stream and the `DecodingSchedMeta` buffers are
  allocated at load time so capture/replay is stable; the parallel meta kernel
  has no host-side shape change.

# WarpDecode Production Deployment Guide

## What WarpDecode is

WarpDecode is the **output-owned NVFP4 MoE decode** path for
`BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft` on NVIDIA
B200 (Blackwell, SM100). It is realized by **`CuteDslFusedMoE`**: the
`cute_dsl` gather-grouped-GEMM + SwiGLU fusion (FC1) followed by the
grouped-GEMM finalize-in-place fusion (FC2 + combine), sequenced by `moe_sort`.

Select it with `moe_backend="WARPDECODE"`. The tile policy is controlled by
`MoeConfig.warp_decode.tile_mode` (default `autotune`). The bandwidth section
below is the full performance reference.

## Model physics: why the all-to-all is common, not eliminated

The target model is **196.6 GB on disk** (108 shards), and a single B200 holds
**~179 GB**. Because 196.6 GB > 179 GB, the model **does not fit on one B200**.
WarpDecode is an output-owned kernel that loops a token's 8 routed experts
in-warp and therefore needs all of a token's experts resident locally. With the
model too large for one GPU, the experts **must be sharded (expert parallel)**
exactly like the native runner, which means WarpDecode pays the **same
all-to-all** as native. The all-to-all is **common to both paths, not
eliminated**.

The gather/scatter that the Cursor "warp-decode" blog describes are **local,
within-GPU HBM operations** (the blog measures a single-GPU kernel with zero
inter-GPU communication). op-trt's native NVFP4 runner already fuses those local
stages, so the realistic win here is the kernel structure, not the removal of
any communication. This is why the measured speedup is modest (~1.0-1.13×
local), not the blog's 1.84× (which was measured against an unfused
BF16-activation baseline, on a single GPU, with no all-to-all on either side).

> Do not reintroduce a "WarpDecode has no all-to-all" comparison. On this model
> the all-to-all is paid by both paths; giving WarpDecode zero a2a while the
> baseline pays full a2a is the reward-hacked framing this document replaces.

## Configuration

```python
from tensorrt_llm.llmapi import MoeConfig, WarpDecodeConfig

# Default: autotune selects the tile size per shape (1-CTA at decode).
moe_config = MoeConfig(backend="WARPDECODE")

# Explicit decode mode (pin 1-CTA / tile_size=128, skip profiling):
moe_config = MoeConfig(
    backend="WARPDECODE",
    warp_decode=WarpDecodeConfig(enabled=True, tile_mode="decode_1cta"),
)

# Explicit prefill / large-batch mode (pin 2-CTA / tile_size=256):
moe_config = MoeConfig(
    backend="WARPDECODE",
    warp_decode=WarpDecodeConfig(enabled=True, tile_mode="prefill_2cta"),
)
```

`tile_mode`:
- `autotune` (default) — the AutoTuner profiles `tile_size` in `{128, 256}` and
  caches the fastest per shape. At pure decode the per-expert row count is tiny
  (~1-2 tokens), so 256-row (2-CTA) tiles run mostly empty and the AutoTuner
  selects 1-CTA (`tile_size=128`).
- `decode_1cta` — pin `tile_size=128` (1-CTA), the decode-optimal kernel; skips
  profiling.
- `prefill_2cta` — pin `tile_size=256` (2-CTA). Only worthwhile for prefill /
  large-batch shapes; **slower at pure decode**.

## Explicit labeling

WarpDecode is named in the logs both at configuration time and at runtime:
- **Config/selection time** (`create_moe.get_moe_cls`): `Selecting
  CuteDslFusedMoE for WarpDecode (output-owned NVFP4 decode, tile_mode=...)`.
- **Backend init** (`CuteDslFusedMoE.__init__`): `WarpDecode backend active:
  CuteDslFusedMoE output-owned NVFP4 decode, tile_mode=...`.
- **Forced tile mode** (`run_moe_nvfp4`): `WarpDecode run_moe_nvfp4 SELECTED
  forced tile_mode=... (tile_size=..., 1-CTA/2-CTA)`.
- **Legacy overlay** (`warp_decode.try_run_warp_decode`, only when the optional
  `MoeConfig.warp_decode` overlay is enabled on another backend): `WarpDecode
  SELECTED (overlay): ...` / `WarpDecode FALLBACK to native MoE backend
  (reason=...)`.

## Measured performance (B200, graph + PDL, real NCCL all-to-all)

MoE decode is **context-independent** — the per-`(GPU count, users)` cell holds
across all context lengths `{1k .. 128k}`, because decode reads the same routed
expert weights regardless of KV length. Speedups below are WarpDecode-1CTA
(autotune-selected) relative to the native NVFP4 runner.

### Matched local (all-to-all excluded as common)

| GPUs | users | WD-1CTA vs native |
|------|-------|-------------------|
| 8    | 16    | 1.00× |
| 8    | 32    | 1.08× |
| 4    | 16    | 1.13× |
| 4    | 32    | 1.07× |
| 2    | 16    | 1.09× |
| 2    | 32    | 1.02× |

Mean ~1.06-1.10× local.

### System end-to-end (real all-to-all, common to both; ~50 µs in-graph ≈ 40% of e2e at 8 GPUs)

| GPUs | users | WD vs native (e2e) |
|------|-------|--------------------|
| 8    | 16    | 0.99× |
| 8    | 32    | 1.04× |
| 4    | 16    | 1.10× |
| 4    | 32    | 1.06× |
| 2    | 16    | 1.11× |
| 2    | 32    | 1.03× |

Mean ~1.05× system.

### Forced 2-CTA at decode

Numerically correct (cosine vs 1-CTA = 0.99999) but **slower at decode**
(0.69-0.84× of native): per-expert M is ~1-2 tokens, so 256-row tiles run mostly
empty. The AutoTuner correctly prunes 2-CTA at decode and picks 1-CTA
`(128,128)`. 2-CTA is the right choice only for prefill / large batch.

## Honest conclusion

For DeepSeek-V3.2-REAP-345B NVFP4 decode on B200:
- WarpDecode (output-owned `CuteDslFusedMoE`, 1-CTA) is **~1.0-1.13× local,
  ~1.05× system** over the native NVFP4 runner. The gain is the fused
  output-owned kernel structure, not communication removal.
- 1-CTA is decode-correct and the AutoTuner selects it automatically; keep
  `tile_mode="autotune"` for production.
- High-route-diversity decode is HBM-bandwidth-bound; the kernel reaches ~80% of
  the 6.8 TB/s peak at the larger per-rank loads (see the bandwidth table below),
  so no large additional kernel speedup is physically available there.

## Bandwidth efficiency (vs the B200 6.8 TB/s peak)

Measured against the B200's 6.8 TB/s peak (Cursor's copy-kernel figure). The
tensor-core FC1 GEMM is HBM-bound on the expert-weight load, and its efficiency
scales with the per-rank load because the fixed per-launch overhead amortizes:

| local experts (GPUs) | weight | latency | TB/s | % of 6.8 |
|---|---|---|---|---|
| 16 (8-GPU) | 235 MB | 49.1 µs | 4.79 | 70% |
| 32 (4-GPU) | 470 MB | 89.0 µs | 5.28 | 78% |
| 64 (2-GPU) | 940 MB | 173.3 µs | 5.43 | **80%** |

Fitting `time = fixed + bytes/BW` gives a load-asymptote of **~5.6 TB/s ≈ 82% of
peak** with ~4.8 µs fixed per-launch overhead — overhead that production removes
via CUDA-graph capture. So at graph-captured decode the kernel reaches ~80% of
the 6.8 TB/s peak at the larger per-rank loads (2-/4-GPU); the smallest 8-GPU
per-rank load is overhead-limited at ~70%. The AutoTuner-selected
`(128,128)/1-CTA` config is optimal at decode (2-CTA and prefetch are slower).

Backend selection is covered by the unit test
`tests/unittest/_torch/modules/moe/test_moe_backend.py::test_warpdecode_backend_selection`.

### CUDA graph is mandatory

The cute_dsl path carries ~264 µs of host dispatch per call (`moe_sort` + two JIT'd
ops + AutoTuner lookup + tvm-ffi marshaling). Eager that dominates — **15% of peak**;
under CUDA-graph capture (which TRT-LLM applies to the decode forward via
`cuda_graph_runner`, and the path is graph-compatible) it is **68% — a 4.47× cliff**.
With `tile_mode="autotune"` the AutoTuner host work must complete during the graph
runner's warmup before capture; the forced `decode_1cta`/`prefill_2cta` modes skip
autotune and are the most capture-clean.

### Where the remaining gap is (graph-captured decomposition, 8-GPU decode)

| component | latency | % of 6.8 |
|---|---|---|
| FC1 (gather + SwiGLU) | 50.9 µs | 68% |
| FC2 (finalize / combine) | 27.9 µs | **62%** (weakest) |
| moe_sort (routing) | 6.2 µs | separable overhead |
| FC1+FC2 | 76.1 µs | (~3% overlap) |

The remaining headroom is **not** launch overhead (graph removes that) and **not** HBM
bandwidth (a parallel-stream test reads 705 MB at 9.6 TB/s — large spare bandwidth). A
four-target parallel investigation (every variant cosine ≥ 0.999, CZS-verified,
graph-captured, peak fixed at 6.8) found the decode kernels are **~95%
fixed-overhead-bound**: FC1 ≈ 49 µs fixed + 0.056 µs/token, FC2 ≈ 27 µs fixed +
0.047 µs/token (4× the tokens adds ~1 µs). It is pipeline-ramp / occupancy bound, and
the easy levers are already exhausted in production:

- **moe_sort** is already PDL-overlapped with FC1
  (`cudaTriggerProgrammaticLaunchCompletion` ↔ `griddepcontrol_wait`); removable
  latency ≈ 0 at decode (NTOK ≥ 8).
- **M-padding** is already masked — the FC1 producer predicates padded A/SF-A loads on
  `mn_limit` (no global read); the kernel is weight-floor-bound and UMMA M-granularity
  is 128 (no smaller tile).
- **FC2** is wave-starved: efficiency rises with wave count (LE=16 → 60%, LE=64 → 76%);
  the deficit is the per-grid ramp/drain amortized over too few waves (FC2's 117 MB
  weight is half FC1's, K is 3.5× shorter) — intrinsic to the shapes, not config.

The **one genuine remaining win** is a **fused FC1→SwiGLU→FC2 persistent megakernel**
that pays the pipeline ramp **once** (streaming w13→w2, overlapping w2's TMA with FC1's
compute tail): ceiling ≈ **76 → 63 µs (~17%)**. The win is ramp amortization, *not* the
intermediate (the `h` round-trip is only ~0.74 µs). It is a large dual-GEMM TMEM-chain
rewrite of the gather kernels (highest effort / highest risk) and is the scoped next
step — the four-agent sweep confirmed no smaller lever substitutes for it.

# WarpDecode HBM Bandwidth Floor Analysis (B200)

*Measured on a4-us-002-rl9. This is the honest version; an earlier draft framed
it as a "2.5× feasibility" study and cited cherry-picked low-slot WarpDecode
wins. The HBM-floor physics below are correct and load-bearing; the speedup
claims have been corrected to the measured ~1.0-1.13× (see
`warpdecode_deployment_guide.md`).*

## Question

For DeepSeek-V3.2-REAP-345B NVFP4 decode at 32 concurrent users on B200, is there
headroom for a WarpDecode kernel to be *multiples* faster than the
production-optimized native NVFP4 TRTLLMGen MoE (PDL + CUDA graph + multi-stream)
across routing shapes?

## Answer: No — native already saturates HBM at moderate-to-high route diversity.

## Method

Both paths NVFP4 (e2m1 weights + e4m3 block scales, sf_vec=16), both
graph-captured. Per-expert NVFP4 weight bytes (w13 + w2 + e4m3 scales) =
24.77 MB. At `S` unique routed experts per call, weight bytes read = `S ×
24.77 MB`. B200 peak HBM = 6.8 TB/s; L2 = 50 MB (cannot hold even 3 experts).

## Measured native NVFP4 (production-optimized) achieved HBM bandwidth

| c-bucket | slot | native µs | HBM floor µs | achieved BW | % peak |
|----------|------|-----------|--------------|-------------|--------|
| c32 | 1  | 23.37 | 3.64  | 1.06 TB/s | 16% |
| c32 | 4  | 30.39 | 14.57 | 3.26 TB/s | 48% |
| c32 | 8  | 37.57 | 29.14 | 5.27 TB/s | 78% |
| c32 | 12 | 48.44 | 43.72 | 6.14 TB/s | 90% |
| c32 | 16 | 65.41 | 58.29 | 6.06 TB/s | 89% |
| c16 | 1  | 10.96 | 3.64  | 2.26 TB/s | 33% |
| c16 | 4  | 19.16 | 14.57 | 5.17 TB/s | 76% |
| c16 | 8  | 31.66 | 29.14 | 6.26 TB/s | 92% |
| c16 | 12 | 45.62 | 43.72 | 6.52 TB/s | 96% |
| c16 | 16 | 62.33 | 58.29 | 6.36 TB/s | 94% |

## Conclusions

1. **At moderate-to-high route diversity (slot ≥ ~8), the native NVFP4 runner is
   already at 89-96% of B200 peak HBM** on weight reads. No kernel — CuTe,
   Triton, hand-SASS, output-owned, tensor-core — can read the routed expert
   weights faster than the memory bus, so multiplicative speedups are physically
   impossible there.

2. **The Cursor blog's 1.84× was over an expert-centric grouped baseline that
   wasted bandwidth on padding/scatter**, measured on a single GPU with no
   all-to-all. The op-trt native NVFP4 (TRTLLMGen) runner already removes that
   padding/scatter waste, so the realistic WarpDecode win over op-trt native is
   the measured **~1.0-1.13× local / ~1.05× system** (output-owned
   `CuteDslFusedMoE` at 1-CTA), not a blog-scale multiple.

3. **At very low slot counts (route fully concentrated), native is
   overhead-bound** (16-33% of peak), so there is *latency* headroom in
   principle — but on this model both paths still pay the common all-to-all (the
   196.6 GB model does not fit on one 179 GB B200, forcing expert parallelism on
   both), so the route-concentrated case does not turn into a large WarpDecode
   system win. It is folded into the measured ~1.0-1.13× local range.

4. **The only way to beat the HBM floor at high slot counts** is to read fewer
   bytes: deeper sub-4-bit quantization, or cross-decode-step weight residency in
   L2 (which needs consecutive steps to route to the same ≤2 experts that fit in
   50 MB — far below the 8-16 active at high diversity). Neither is a WarpDecode
   kernel change.

## What this means for the deployment

Keep `moe_backend="WARPDECODE"` with `tile_mode="autotune"`. The AutoTuner
selects 1-CTA at decode, delivering the measured ~1.0-1.13× local / ~1.05× system
over native. At high route diversity native NVFP4 is already at the HBM wall and
no further multiplicative kernel speedup is available. There is no "2.5× across
all shapes" — that target is unreachable on B200 for this model because
high-diversity MoE decode is HBM-bandwidth-bound and native already saturates the
bus.

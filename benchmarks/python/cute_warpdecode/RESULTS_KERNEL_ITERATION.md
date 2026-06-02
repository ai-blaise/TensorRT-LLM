# WarpDecode per-thread kernel iteration (B200, NVFP4, cos=1.0 / CZS-verified)

Direct analysis of CuTe DSL docs, the CuTe paper (arxiv 2603.02298), Veitner, and Colfax
(3 passes) → 8 iteration rounds on the faithful output-owned per-thread kernels. Each round
verified by cosine vs an FP32 reference (correctness obligation) + a CZS index contract
(`docs/proofs/warpdecode_r4_*`, `warpdecode_r8_*`) + an event-timed B200 benchmark.

## gate_up (FC1), B=32
| round | change | TB/s | % of 6.8 peak |
|---|---|---|---|
| baseline | per-block scale, F32 reduce | 3.14 | 46% |
| R3 | F16 within-block accum (F32 across) | 3.75 | 55% |
| **R4** | + 8-wide vector + tree reduce (depth 4 vs 16) | **4.52** | **66%** |
| R8 | weight-amortized (2 tokens / (expert,neuron) warp) | 5.62 eff | 1.24× faster than R4 |

**R4 exceeds Cursor's reported 3.95 TB/s = 58% of peak** (cursor.com/blog/warp-decode), on the
same B200, correctness-preserving. The lever (grounded in Veitner's reduction posts + Colfax's
FA-4 "asymmetric hardware scaling": at tiny-M decode the bottleneck is the ALU/reduction, not the
tensor cores): F16 reduction throughput + collapsed dependency depth. R8 imports the grouped-GEMM
amortization principle (Veitner grouped-blockscaled-GEMM) — each expert's weight is loaded+dequantized
once per pair of tokens instead of per token, the redundant-reload that dominates at B=32.

## down (FC2), B=32
| round | change | TB/s | % peak |
|---|---|---|---|
| baseline (multi-h/warp) | F32 reduce | 3.09 | 45% |
| **R5** | F16 + tree reduce | **3.91** | **57%** |

## Negative rounds (also cos=1.0 — recorded honestly)
- R1 K-mode parallelization (Veitner method 1): worse — the per-output grid already *saturates* the
  B200 at decode (256·LP·8 warps), unlike Veitner's under-saturated GEMV.
- R2 8-wide vector accumulator with per-element scale: worse — added 8× scale mults.
- R6 SMEM-shared activation dequant: worse — 14 KB SMEM cut occupancy + devectorized the inner mult.
- R7 2× unroll + 2 accumulators: worse — register pressure + branch overhead.

## Honest scope
These are **kernel-bandwidth** results for the per-thread output-owned reference kernels — the direct
comparison to Cursor's reported figure. They do **not** change the system conclusion: the per-thread
design re-loads weights per (token,expert) pair, moving more bytes than the amortized tensor-core
grouped GEMM, so at the full MoE the cute_dsl tensor-core path (`CuteDslFusedMoE`, the WARPDECODE
backend) stays the production choice (~1.05× system over native; see WARPDECODE.md). R8 narrows that
gap by importing the amortization idea but converges toward — does not beat — the tensor-core path.

Optimized reference kernels: `23_gate_up_f16_tree_opt.py`, `24_down_f16_tree_opt.py`,
`25_gate_up_weight_amortized.py`.

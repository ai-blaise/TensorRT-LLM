# WarpDecode for DeepSeek-V3.2-REAP-345B NVFP4 — findings + kernels

Output-owned ("warp decode", Cursor) MoE decode for `DeepSeek-V3.2-REAP-345B-...-NVFP4` on B200,
faithfully implemented in CuTeDSL and measured against the op-trt native NVFP4 baseline
(`FP4BlockScaleMoERunner`, graph+PDL). HIDDEN=7168, INTER=2048, top_k=8, NVFP4 weights **and**
activations (ActKV-NVFP4).

## Result trajectory (all measured on B200, idle GPU, graph/event-timed)

1. **Faithful per-thread WarpDecode** (files 14,15): warp-per-output, `cvt_f4e2m1x8_to_f16x8`,
   `warp_reduction_sum` butterfly, no atomic/scatter, BF16 intermediate. **cosine 1.0 / 0.999999**
   (matches Cursor's >0.999996). Optimized: single GEMV 2.71 TB/s (file 09, beats Veitner improved
   2.1); down 1.06→3.09 TB/s via multi-h/warp (file 18); gate_up SF-block-aligned 2.52→2.88 (file 21).
2. **It is COMPUTE-bound, not memory-bound** (file 20, the decisive isolation): per-thread weight
   load = **6.69 TB/s** (98% of B200 peak), load+cvt = 6.33, but the full dot drops to **2.88** — the
   per-thread FFMA + reduction is the bottleneck. Software tricks (block-SF, tree-reduce,
   multi-accumulator) top out ~3 TB/s. Full matrix (file 19): per-thread WD is 0.26–0.51× of native.
3. **Tensor cores (UMMA `tcgen05`) are the lever** — confirmed by measurement. The tensor-core NVFP4
   MoE FC1 (`moe_as_dense_gemm/fc1.py`) at the G8 decode-equivalent (16 experts active, 235 MB weight)
   = **30.9 us (~7.6 TB/s)** vs the per-thread gate_up's 80 us = **2.6× faster**, and already most of
   the way under native's *full* MoE (78 us). FC2 weight ~half → ~15–20 us. **Tensor-core MoE projects
   to ~48 us vs native 78 us ≈ 1.5–1.6× decode win** (matches the 53 us HBM load floor; native sits at
   1.47× the floor due to tile-padding/bookkeeping the output-fused path approaches).

## Honest target calibration
Realistic win over the op-trt native baseline is **~1.5×** (approaching the HBM floor), not the 1.84×
Cursor reported — because op-trt's native is already a near-optimal *fused tensor-core* kernel, whereas
Cursor's 1.84× was vs a ~2.15 TB/s non-tensor-core baseline. 1.5× is still a comfortable systems win.

## The tensor-core WarpDecode is the production CuTe path
The output-fused tensor-core WarpDecode = the production ops
`cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell` (FC1+SwiGLU, gather-fused) +
`cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell` (FC2+combine, finalize-inplace), wired via
`moe_sort` (= `CuteDslFusedMoE`). gather+finalize eliminate scatter/combine on tensor cores.
The per-thread kernels here are the verified correctness reference + the compute-bound lesson.

## File index
00 fp4-vs-uint8 copy BW · 01 single GEMV · 02 batched gather GEMV · 03 gateup+SwiGLU · 04 down+scatter ·
05 full MoE pipeline · 06 uint8-decode GEMV · 07 coalesced+SMEM GEMV · 08 cvt GEMV · 09 cvt+coalesced
(2.71 TB/s) · 10 gateup-win · 11 down-win · 12 graph production latency · 13 down expert-batched ·
14 gate_up faithful · 15 down faithful · 16 full matrix WD-vs-native · 17 native baseline matrix ·
18 down multi-h optimized (3.09 TB/s) · 19 full matrix optimized · 20 load-ceiling compute-bound proof ·
21 gate_up block-aligned SF.

## Tensor-core optimization round (after direct analysis of arxiv/Colfax/Veitner)

Reading Colfax "Hardware-supported Block-scaling" (Part 4) surfaced the **`cta_group::2`** lever (two
SMs cooperate per UMMA). Applied to the tensor-core FC1 (NVFP4, 16-experts BW-bound):

| config | FC1 latency |
|---|---|
| mma_tiler 128×128, cluster 1,1 (1-CTA) | 30.96us (baseline) |
| mma_tiler 256×128, cluster 2,1 (2-CTA) | 38.47us |
| mma_tiler 128×256, cluster 1,1 | 32.47us |
| **mma_tiler 256×256, cluster 2,1 (2-CTA)** | **24.33us** ← best, 1.27× over baseline |
| 256×256, cluster 2,2 | 25.05us |

**FC1: 30.96 → 24.33us (1.27×)** via 2-CTA + 256×256 tile — and 3.3× over the per-thread gate_up (80us).
FC2 weight is ~half (117 MB) → ~12us at the same efficiency. **Optimized tensor-core MoE ≈ 36us GEMM
vs native 78us** (~2× GEMM-only; ~1.5× full-pipeline after the shared routing/finalize). The proper
decode FC2 is the grouped kernel (moe_as_dense FC2's `k=expert×INTER` framing is a concatenated GEMM,
wrong for decode). Remaining juice: pipeline stages, TMA multicast for shared weight, persistent tile
scheduler, and the grouped-kernel FC2. Resource analyses logged to agentmemory :3811.

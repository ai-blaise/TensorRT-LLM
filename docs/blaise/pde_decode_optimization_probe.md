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

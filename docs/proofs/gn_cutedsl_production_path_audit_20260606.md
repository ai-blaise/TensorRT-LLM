# gn-kernels CuTeDSL production-path audit

Source under review: `gau-nernst/gn-kernels` commit `94cdf76a25a8762c8daf989ec54a28a8607d872a`, directory `gn_kernels/cutedsl`.

op-trt parent review baseline: `origin/op-trt` at `e17ef7f1f904f3d31d658e83dcb16e0eb76f4791`.

Remote worktree: `/home/spencer/work/TensorRT-LLM-gn-cutedsl-audit-20260606` on `a4-us-002-rl9`.

## GPU occupancy

Recorded before any benchmark work on `a4-us-002-rl9`, 2026-06-06 16:10 UTC:

```text
0, NVIDIA B200, 137578, 183359, 0
1, NVIDIA B200, 136624, 183359, 0
2, NVIDIA B200, 136624, 183359, 0
3, NVIDIA B200, 136810, 183359, 0
4, NVIDIA B200, 160550, 183359, 0
5, NVIDIA B200, 159890, 183359, 0
6, NVIDIA B200, 159888, 183359, 0
7, NVIDIA B200, 159728, 183359, 0
```

`nvidia-smi pmon -c 1` showed active Python compute processes on every GPU. Per the work request, no heavy GPU benchmarks were run while the B200s were occupied.

## Inventory and mapping

| gn file | Kernel | Closest op-trt surface | Production-path decision |
|---|---|---|---|
| `sm100_mm_bf16.py` | `MatmulSm100`, plain SM100 BF16 GEMM, wrapper uses BN=256 and 2CTA | `trtllm::cute_dsl_bf16_gemm_blackwell`, `dense_gemm_persistent.py` | Out of scope. The live path does not enable generic BF16 CuteDSL GEMM; current live pieces are WarpDecode NVFP4 MoE plus DSA/HISA/KVarN attention. |
| `sm100_mm_mxfp8.py` | `MatmulMXFP8Sm100`, plain dense MXFP8 GEMM with MMA-layout scale factors and BF16 output | `trtllm::cute_dsl_fp8_gemm_blackwell`, `dense_blockscaled_gemm_persistent.py` | Out of scope. Generic dense MXFP8/blockscaled GEMM is not the live KVarN/DSA paged logits or WarpDecode path. |
| `sm100_mm_nvfp4.py` | `MatmulNVFP4Sm100`, plain dense NVFP4 GEMM with MMA-layout scale factors and BF16 output | Related to `trtllm::cute_dsl_nvfp4_gemm_blackwell`; not equivalent to WarpDecode grouped/fused MoE ops | Out of scope for promotion. Live WarpDecode needs gather/grouped routing, tile metadata, activation/quantize fusion, finalize/combine, EP composability, and fail-closed no-fallback semantics. |
| `sm80_mm_bf16.py` | SM80 BF16 dense GEMM | None for B200 SM100 production | Out of scope. |

## Correctness equivalence gates

No gn kernel reached benchmark eligibility because none was an equivalent production-path kernel. If a future live config enables one of the closest plain GEMM surfaces, equivalence must be proven before benchmarking:

- Shapes/layouts: include SMC draft `M=1/5/25`, normal decode, and prefill-ish M; verify K/N divisibility, transposed weight layout, contiguous/strided inputs, and output reshape behavior.
- Dtypes/scales: BF16 output for BF16/MXFP8/NVFP4; exact scale-factor dtype and MMA layout; no silent interpretation change between E8M0/E4M3 scale storage and TRT-LLM packed scale formats.
- Accuracy: compare against existing op-trt equivalent and a dequantized torch reference with dtype-specific tolerances; include odd/tiny M and boundary K/N cases.
- Runtime contract: CUDA graph capture/replay, stream semantics, no hidden fallback, compile/autotune warmup behavior, and failure mode when unsupported.
- Composability: TP/EP rank-local shapes, WarpDecode no-backend-fallback policy, DSA/HISA/KVarN interactions, and absence of cross-rank layout assumptions.

## Benchmark decision

No benchmark was run because strict production-path filtering produced zero candidates. This is intentional: benchmarking plain gn dense GEMMs against unrelated grouped MoE or paged-MQA kernels would not prove a production win.

Static audit command:

```bash
python3 benchmarks/python/gn_cutedsl_production_path_audit.py \
  --gn-root /home/spencer/work/gn-kernels-94cdf76-cutedsl \
  --json /tmp/gn_cutedsl_audit.json
```

## Integration decision

No gn kernel is integrated. There is no measured win on an equivalent production-path surface, and the only kernel that is superficially close to live WarpDecode NVFP4 (`sm100_mm_nvfp4.py`) lacks the grouped/fused/finalize semantics required by the live path.

Next step when GPUs are free: re-run the audit command to confirm occupancy, then only add a benchmark if the live config starts using a closest plain GEMM surface (`cute_dsl_bf16_gemm_blackwell`, `cute_dsl_fp8_gemm_blackwell`, or `cute_dsl_nvfp4_gemm_blackwell`). Until then, the fail-closed conclusion is no promotion.

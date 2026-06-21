# Megakernel Campaign

## Control Config

- Source of truth: `.bench_runs_claude/BEST_CONFIG/BEST_CONFIG.md`.
- Proven image: `localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-vbfuse-nvfp4-20260615T043037Z`.
- Required decode env:
  - `TRTLLM_OPTRT_MOE_MEGAKERNEL=1`
  - `TRTLLM_FORCE_COMM_METHOD=NVLINK_TWO_SIDED`
- Required model config:
  - `moe_config.use_low_precision_moe_combine: false`
  - WARPDECODE backend/policy/tile mode from BEST_CONFIG
  - `max_batch_size: 64`
- C32 is the primary acceptance point. Latest restored control sanity:
  - C32, 64 requests, ISL 2055, OSL 512
  - TTFT p50 1447 ms, p95 3370 ms
  - user tok/s p50 50.35
  - aggregate tok/s 1402.6

## Main Bottleneck

Dense/proj GEMM is still the biggest profile bucket:

- 9102 us/iter
- 37.8% of iteration time
- 640 launches/iter

The exposed decode-side opportunity is not just fewer launches. The gate GEMM currently overlaps on a side stream, so a replacement that removes launches but serializes overlapped work can lose throughput.

## Completed Experiments

### Shared-Expert SwiGLU FP4-Out

- Path fired and removed most standalone quant launches in the shared expert path.
- It did not produce an aggregate serving win.
- Keep the result as a caution: launch removal outside the exposed critical path is not enough.

### Monolithic Gate CuTe BF16 GEMM

- Replaced chunked gate projection with one full BF16 CuTeDSL GEMM.
- It compiled and ran, but lost throughput versus control.
- Likely cause: it destroyed useful side-stream overlap from the two-chunk gate path.
- Conclusion: do not optimize gate launch count by serializing currently overlapped work.

### Triton Fused Sigmoid-Mul + NVFP4 Pack Prototype

- Targeted the exposed path after attention:
  - `PersistentDenseGemmKernel`
  - `_sigmoid_mul_kernel`
  - `quantize_with_block_size`
  - FP4 `o_proj`
- Prototype fuses sigmoid-mul and swizzled NVFP4 activation packing before `o_proj`.
- Microbench signal was positive for the boundary:
  - `m=1,n=16384`: 37.54 us -> 28.37 us
  - `m=32,n=16384`: 31.47 us -> 28.28 us
- Scale-factor swizzle matched reference exactly.
- FP4 packed bytes did not match exactly because the Triton E2M1 rounding does not reproduce CUDA `__nv_fp4_e2m1` conversion.
- Conclusion: useful target, but production needs the exact CUDA/C++ conversion path.

## Current Target

Implement an exact CUDA/C++ op for:

```text
q, sf = fp4_quantize(fused_sigmoid_mul(attn_output, gate), o_proj.input_scale, 16, false, true)
```

Then pass `Fp4QuantizedTensor(q, sf, is_sf_swizzled=True)` into `o_proj`.

The kernel must preserve the current numerical contract:

- Input/output activation dtype: BF16.
- Match `fused_sigmoid_mul` rounding:
  - sigmoid result rounded to BF16
  - product rounded to BF16
- Match `fp4_quantize(..., is_sf_swizzled=True)` packing:
  - use the existing CUDA NVFP4 helpers instead of hand-coded approximate thresholds
  - produce identical packed bytes and identical scale-factor bytes for valid positions

## Current Implementation Checkpoint

Added guarded production-path plumbing for:

```text
torch.ops.trtllm.fused_sigmoid_mul_quant_nvfp4_swizzled(input, gate, sf_scale, sf_vec_size=16)
```

Implementation location:

- `cpp/tensorrt_llm/kernels/fusedActivationQuant.cu`
- `cpp/tensorrt_llm/thop/fusedActivationQuant.cpp`
- `tensorrt_llm/_torch/modules/fused_lowrank_gate.py`
- `tensorrt_llm/_torch/modules/attention.py`

The op is default-off behind:

```text
TRTLLM_OPTRT_FUSED_SIGMOID_MUL_QUANT_NVFP4=1
```

Validation completed in the benchmark container:

- C++/CUDA object rebuild passed after formatting.
- Exactness passed against the current two-op reference:
  - `q_ref, sf_ref = fp4_quantize(fused_sigmoid_mul(x, gate), scale, 16, false, true)`
  - tested `M in [1, 4, 16, 32, 64, 129]`
  - tested `N in [1024, 7168, 8192, 16384]`
  - packed FP4 byte diff: 0
  - swizzled scale-factor byte diff: 0
- Isolated boundary microbench:

```text
m= 1 n=16384 ref_us=33.469 fused_us=7.393 speedup=4.53x
m= 4 n=16384 ref_us=34.749 fused_us=7.699 speedup=4.51x
m=16 n=16384 ref_us=34.435 fused_us=7.807 speedup=4.41x
m=32 n=16384 ref_us=41.738 fused_us=8.147 speedup=5.12x
m=64 n=16384 ref_us=32.616 fused_us=7.756 speedup=4.21x
m=32 n= 7168 ref_us=35.030 fused_us=7.680 speedup=4.56x
m=32 n= 8192 ref_us=34.818 fused_us=7.817 speedup=4.45x
```

This is a real local win on the exposed boundary. It still must prove itself in serving, because the
dense/proj bucket contains many GEMMs and because previous wins outside the critical path did not translate.

## Production A/B Result: Sigmoid-Mul + NVFP4 Quant

Packaging note:

- Replacing `libth_common.so` is not viable for this experiment. It imports, but it drops existing
  `torch.classes.trtllm.*` class registrations such as `CublasLtFP4GemmRunner`.
- The viable image is additive:
  - `localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-sigmoid-quant-ext-20260618T011000Z`
  - keeps BEST_CONFIG `libth_common.so`
  - lazy-loads `libsigmoid_quant_ext.so`
  - keeps `TRTLLM_OPTRT_MOE_MEGAKERNEL=1`, `TRTLLM_FORCE_COMM_METHOD=NVLINK_TWO_SIDED`,
    and `use_low_precision_moe_combine=false`

Clean post-recycle serving A/B, C32, ISL 2055, OSL 128, 64 requests:

```text
BEST_CONFIG:
  TTFT p50 1243 ms, p95 3187 ms
  user tok/s p50 37.86
  aggregate tok/s 854.1
  wall 9.6 s

sigmoid_quant_ext:
  TTFT p50 1288 ms, p95 3365 ms
  user tok/s p50 38.87
  aggregate tok/s 850.9
  wall 9.6 s
```

Interpretation:

- The exact fused boundary kernel is real: isolated timing is ~4.2-5.1x faster and production
  serving shows a small per-user steady tok/s lift at C32/OSL128.
- It is not yet an aggregate throughput win. Aggregate is effectively flat/slightly down in this
  single clean A/B, so this does not solve the 9102 us/iter dense/proj GEMM bucket.
- The likely reason is scope: this op removes the post-MLA activation+quant boundary before `o_proj`,
  but the measured dense/proj bucket is dominated by many GEMM launches/tactics, not only that boundary.
- Next useful proof is a profile with the additive image to confirm:
  - `extFusedSigmoidMulQuantizeKernel` fires on the MLA gate path
  - the corresponding `_sigmoid_mul_kernel`/`quantize_with_block_size` launches disappear there
  - dense/proj GEMM launch count and time do or do not move

## Dense/Proj GEMM Follow-Up: MLA Output Gate

Trace mapping result:

- The largest dense/proj kernel in the C32/OSL128 profile,
  `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT`, is the MLA output-gate projection.
- It appears as 122 calls/iter on the gate side stream: two half-N BF16 GEMMs per layer.
- The `o_proj` NVFP4 path is a different kernel family (`cutlass3x...block_scaled...`) and is
  not the top single dense/proj bucket.

Two gate-GEMM experiments were run:

1. DSV3 fused-A strided gate op:
   - implemented as a guarded custom op, `trtllm::dsv3_gate_gemm_op`
   - default-off behind `TRTLLM_OPTRT_MLA_GATE_DSV3_FUSED_A=1`
   - numerically matched the current two-half GEMM reference
   - lost in isolated timing because the DSV3 fused-A tiling expands the 8192-output half into
     512 CTAs/launch, while cuBLASLt's nvjet kernel is already much more compact for this shape

```text
M=1   current two-half GEMM 42.71 us, dsv3 gate 48.18 us
M=4   current two-half GEMM 38.30 us, dsv3 gate 48.56 us
M=8   current two-half GEMM 38.38 us, dsv3 gate 48.88 us
M=16  current two-half GEMM 38.20 us, dsv3 gate 50.86 us
```

2. Full one-launch cuBLASLt gate GEMM:
   - implemented as a guarded Python path, `TRTLLM_OPTRT_MLA_GATE_TORCH_MM=full`
   - isolated and overlap-replay microbenches looked better than the two-half path
   - production serving still regressed at the C32/OSL128 acceptance point

```text
gate_full_mm_mega_cf:
  image: optrt-529374445d-codex-gate-full-mm-20260618T0310
  C32 / ISL 2055 / OSL 128 / 64 requests
  TTFT p50 1879 ms, p95 3147 ms
  user tok/s p50 36.98
  aggregate tok/s 764.6
  wall 10.7 s

BEST_CONFIG restored control, same command:
  C32 / ISL 2055 / OSL 128 / 64 requests
  TTFT p50 1241 ms, p95 3164 ms
  user tok/s p50 38.10
  aggregate tok/s 844.3
  wall 9.7 s
```

Interpretation:

- The existing two-half side-stream gate schedule remains the production winner for now.
- A naive full-GEMM replacement can win standalone but lose at serving level because it changes
  overlap/capture behavior and likely shifts critical-path waits.
- Stretching the DSV3 fused-A low-latency kernel to the 16k gate shape is not a viable kernel
  direction; it uses too many CTAs for this output width.
- The next principled dense/proj target should be a kernel that preserves the side-stream schedule
  but beats nvjet per half, or a deeper fusion that removes the gate materialization/activation
  dependency without collapsing the overlap window.

### Gate Kernel Follow-Up: Tiled DSV3 and CuTe Half-GEMM

The DSV3 fused-A gate kernel was extended so the CTA computes more than one
16-row output subtile:

- new `invokeFusedAGemmStridedTiled<T, HdIn, HdOut, TileM, TileN>`
- `TileM in {16, 32, 64, 128}` for the 7168 x 8192 gate-half shape
- opt-in selector: `TRTLLM_OPTRT_MLA_GATE_DSV3_TILE_M`
- gate hook remains default-off behind `TRTLLM_OPTRT_MLA_GATE_DSV3_FUSED_A=1`

This answered the "too many CTAs" objection directly. Wider `TileM` improves
the custom kernel at small M, but it still does not beat the current cuBLASLt
two-half path at the C32/C64 production token counts:

```text
M= 1 two_half_torch_us=60.70
  tile= 16 custom_us=57.38 ratio=0.945
  tile= 32 custom_us=49.44 ratio=0.814
  tile= 64 custom_us=48.82 ratio=0.804
  tile=128 custom_us=72.98 ratio=1.202

M=16 two_half_torch_us=58.29
  tile= 16 custom_us=59.15 ratio=1.015
  tile= 32 custom_us=52.70 ratio=0.904
  tile= 64 custom_us=53.10 ratio=0.911
  tile=128 custom_us=77.79 ratio=1.334

M=32 two_half_torch_us=56.48
  tile= 16 custom_us=99.39 ratio=1.760
  tile= 32 custom_us=85.28 ratio=1.510
  tile= 64 custom_us=85.14 ratio=1.507
  tile=128 custom_us=77.39 ratio=1.370

M=64 two_half_torch_us=56.37
  tile= 16 custom_us=178.35 ratio=3.164
  tile= 32 custom_us=146.27 ratio=2.595
  tile= 64 custom_us=153.09 ratio=2.716
  tile=128 custom_us=141.15 ratio=2.504
```

Numerics: cosine was ~1.0, max absolute diff was 1-2 BF16 ULP-ish versus
the current `torch.mm` two-half reference. That is close, but not bit-exact.

The existing Blackwell CuTe BF16 GEMM wrapper was also tested as a possible
more principled gate-half substrate. It is bit-exact, but slower than cuBLASLt
for the gate-half shape even with contiguous output:

```text
shape: [M, 7168] x [8192, 7168]^T -> [M, 8192]

M= 1 torch_half_us=30.14 cute_half_us=53.42 ratio=1.772
M= 4 torch_half_us=28.59 cute_half_us=56.70 ratio=1.983
M= 8 torch_half_us=28.26 cute_half_us=54.09 ratio=1.914
M=16 torch_half_us=28.77 cute_half_us=38.07 ratio=1.323
M=32 torch_half_us=28.40 cute_half_us=37.59 ratio=1.324
M=64 torch_half_us=28.61 cute_half_us=38.07 ratio=1.331
```

Follow-up tactic sweep script:

- `.bench_runs_claude/megakernel/bf16_gate_tactic_sweep.py`
- shape: `[M, 7168] x [8192, 7168]^T -> [M, 8192]`
- compares the production cuBLASLt/nvjet `torch.mm(..., out=...)` floor
  against direct `CuteDSLBf16BlackwellGemmRunner` tactics

The current production candidate set misses one useful tactic:

```text
current candidate set:
M=32 torch_us=29.30 default_cute_us=30.66 best_cute_us=30.87 ratio=1.054
M=64 torch_us=30.28 default_cute_us=31.38 best_cute_us=31.42 ratio=1.038

expanded quick set:
M=32 torch_us=29.16 default_cute_us=30.60 best_cute_us=30.55 ratio=1.048
  best_tactic=Tactic(use_2cta=True, mn=(128, 128), cluster=(2, 2))
M=64 torch_us=30.75 default_cute_us=31.64 best_cute_us=30.39 ratio=0.988
  best_tactic=Tactic(use_2cta=False, mn=(64, 64), cluster=(1, 1))
```

Action taken:

- added `(64, 64)` to `CuteDSLBf16BlackwellGemmRunner` so the autotuner can
  select the only tactic that beat the nvjet floor in the sweep

Interpretation:

- The old DSV3 HMMA/cp.async kernel is a useful experimental proof point but
  the wrong production substrate for gate at C32/C64.
- The existing CuTe persistent BF16 GEMM wrapper only has a marginal M64 tactic
  win and still loses at M32; it is not the primary replacement for the 9.1ms
  dense/proj bucket.
- The BF16 gate path needs either a new shape-specialized WGMMA/TMA kernel
  that beats cuBLASLt per half while preserving the side stream, or it should
  be bypassed by fusing the downstream gate activation/quant/projection boundary.

### Gate Kernel Follow-Up: Single-Launch DSV3 Full Output

Follow-up after the tiled DSV3 half-output test: the DSV3 gate op was extended
with an opt-in full-output launch:

- env knob: `TRTLLM_OPTRT_MLA_GATE_DSV3_SINGLE_OUTPUT=1`
- dispatch: one `invokeFusedAGemmStridedTiled<T, 7168, 16384, TileM, TileN>`
  call instead of two 8192-output calls
- template coverage: `TileM in {16, 32, 64, 128}`, `TileN in {8, 16}`
- build proof: `dsv3_min_latency_kernels` and `th_common` compile/link against
  the fresh symbols in `/repo/cpp/build`

Result file:

- `.bench_runs_claude/results/80_dsv3_gate_single_output.txt`

Graph replay microbench against the current exact two-half cuBLASLt path:

```text
split DSV3 path:
M= 1 two_half_us=45.23 full_out_us=40.79 dsv3_us=51.21 ratio_vs_two=1.132
M= 4 two_half_us=40.97 full_out_us=38.25 dsv3_us=51.22 ratio_vs_two=1.250
M= 8 two_half_us=40.95 full_out_us=38.94 dsv3_us=51.21 ratio_vs_two=1.251
M=16 two_half_us=40.96 full_out_us=39.01 dsv3_us=53.26 ratio_vs_two=1.300
M=32 two_half_us=41.05 full_out_us=40.97 dsv3_us=93.21 ratio_vs_two=2.271
M=64 two_half_us=44.12 full_out_us=40.93 dsv3_us=178.59 ratio_vs_two=4.048

single-output DSV3 path:
M= 1 two_half_us=45.30 full_out_us=40.77 dsv3_us=45.91 ratio_vs_two=1.013
M= 4 two_half_us=40.97 full_out_us=38.85 dsv3_us=47.09 ratio_vs_two=1.149
M= 8 two_half_us=40.95 full_out_us=38.92 dsv3_us=47.08 ratio_vs_two=1.150
M=16 two_half_us=40.95 full_out_us=39.01 dsv3_us=47.25 ratio_vs_two=1.154
M=32 two_half_us=41.04 full_out_us=40.95 dsv3_us=87.97 ratio_vs_two=2.144
M=64 two_half_us=43.46 full_out_us=40.62 dsv3_us=175.50 ratio_vs_two=4.038
```

Numerics match the earlier DSV3 behavior: cosine ~1.0, max absolute diff 1-2
BF16 ULP-ish against the current two-half reference.

Interpretation:

- Single-launch DSV3 answers the launch-collapse question directly, and it is
  better than split DSV3 at small M.
- It still does not beat the current cuBLASLt half-GEMM floor, so deploying this
  branch into serving is not justified.
- The gate branch is now constrained to either a new WGMMA/TMA gate-half kernel
  that beats cuBLASLt directly, or a broader fusion that removes work around the
  GEMM instead of replacing the GEMM with this old HMMA/cp.async substrate.

### Dense/Proj NVFP4 Projection Mapping

The major NVFP4 dense projection shapes were re-mapped with backend forcing.
The result was consistent: cuBLASLt/nvjet is already the fastest available
backend for the important production shapes. Forcing CUTLASS/CuTe is not the
missing lever.

```text
mla_fused_q_kv_a  N= 2112 K= 7168 -> cuBLASLt nvjet 128x128 family
q_b_proj          N=24576 K= 1536 -> cuBLASLt nvjet 256x128/128x128 family
o_proj            N= 7168 K=16384 -> cuBLASLt dispatching cutlass3x block-scaled kernel
shared_gate_up    N= 4096 K= 7168 -> cuBLASLt nvjet 128x128 family
shared_down       N= 7168 K= 2048 -> cuBLASLt nvjet 128x128 family
```

This means the next dense/proj kernel work cannot be "try a different existing
backend". It must remove work around the GEMM or beat the cuBLASLt kernel:

1. fused BF16-input -> NVFP4 projection for selected shapes, eliminating the
   standalone activation quantize launch and feeding the projection MMA directly;
2. grouped/persistent projection launch only where dependencies expose multiple
   independent same-shape GEMMs at the same point;
3. new gate-half WGMMA/TMA kernel only if a prototype clears the cuBLASLt
   half-GEMM floor above before any serving integration.

### Dense/Proj NVFP4 CuTeDSL Fallback Routing

Follow-up: the "existing backend" claim was tested more narrowly instead of
stopping at one broad attempt.

Implementation:

- added an env-gated fallback hook in `NVFP4GemmUnifiedRunner`:
  `TRTLLM_NVFP4_GEMM_FALLBACK_PREFER_CUTEDSL=1`
- added a JSON allow-list:
  `TRTLLM_NVFP4_GEMM_FALLBACK_CUTEDSL_TACTIC_TABLE`
- only table-listed `(M,N,K)` shapes route to CuTeDSL; non-listed shapes fall
  back to the normal cuBLASLt/CUTLASS small-M path
- added `.bench_runs_claude/extract_nvfp4_cutedsl_tactics.py` to sweep a small
  tactic grid against cuBLASLt for the production decode shapes

Isolated tactic sweep:

```text
o_proj       M=1  N= 7168 K=16384  cublaslt=38.809us cutedsl=37.192us  +4.17%
o_proj       M=2  N= 7168 K=16384  cublaslt=36.611us cutedsl=35.919us  +1.89%
o_proj       M=64 N= 7168 K=16384  cublaslt=37.058us cutedsl=33.420us  +9.82%
mlp_gate_up  M=1  N=36864 K= 7168  cublaslt=46.719us cutedsl=53.682us -14.90%
mlp_gate_up  M=2  N=36864 K= 7168  cublaslt=46.286us cutedsl=51.761us -11.83%
mlp_gate_up  M=64 N=36864 K= 7168  cublaslt=48.066us cutedsl=48.099us  -0.07%
mlp_down     M=1  N= 7168 K=18432  cublaslt=39.659us cutedsl=40.967us  -3.30%
mlp_down     M=2  N= 7168 K=18432  cublaslt=38.421us cutedsl=37.368us  +2.74%
mlp_down     M=64 N= 7168 K=18432  cublaslt=42.771us cutedsl=35.068us +18.01%
```

Production A/B, C32, ISL 2055, OSL 128, 64 requests:

```text
BEST_CONFIG warmed control:
  TTFT p50 1438 ms, p95 3202 ms
  user tok/s p50 38.45
  aggregate tok/s 834.5
  wall 9.8 s

high-impact fixed tactic route:
  repeats aggregate tok/s: 713.6, 838.6, 811.9, 839.0, 835.8
  post-warm median aggregate tok/s: 837.2

allow-list tactic route:
  repeats aggregate tok/s: 649.6, 779.9, 833.6, 820.7, 811.1
  post-warm median aggregate tok/s: 815.9
```

Interpretation:

- The per-op CuTeDSL wins are real for selected `o_proj` and `mlp_down` M
  buckets.
- They do not translate to a serving win. The narrow high-impact arm was
  effectively flat within noise; the measured allow-list arm was worse.
- This answers the backend-routing branch: the dense/proj bucket is not solved
  by choosing CuTeDSL for a few fallback misses. The branch still leaves the
  same Python/op/GEMM launch structure in place.
- The next kernel direction must either reduce launches at a legal grouping
  site or fuse a true dependency boundary into the projection, while preserving
  the overlap behavior that made the existing gate path competitive.

### Dense/Proj L-Batch Persistent Kernel Tests

The existing `Sm100BlockScaledPersistentDenseGemmKernel` has a native batch/L
dimension. The first version of the local harness timed tensor stacking inside
the graph, so it was repaired to prepack the L FP4/SF buffers before graph
timing:

- `.bench_runs_claude/megakernel/dense_kernel_driver.py`
- `.bench_runs_claude/megakernel/thesis_test.py`

Focused results, M=16, prepacked inputs, `cute_sep` skipped to avoid compile
noise, compared against L separate production `nvfp4_gemm(..., cublaslt)` calls
in one CUDA graph:

```text
kv_a_proj      N= 2112 K= 7168 L=2  cublas_sep=21.83us mega=16.20us 1.35x
kv_a_proj      N= 2112 K= 7168 L=4  cublas_sep=34.77us mega=16.38us 2.12x
shared_gate_up N= 4096 K= 7168 L=2  cublas_sep=21.76us mega=17.77us 1.22x
shared_gate_up N= 4096 K= 7168 L=4  cublas_sep=35.15us mega=17.62us 1.99x
q_b_proj       N=24576 K= 1536 L=2  cublas_sep=19.33us mega=19.54us 0.99x
q_b_proj       N=24576 K= 1536 L=4  cublas_sep=31.26us mega=23.45us 1.33x
o_proj         N= 7168 K=16384 L=2  cublas_sep=49.38us mega=34.96us 1.41x
```

Narrow rerun after killing an over-broad CuTeDSL tactic compile:

```text
o_proj         N= 7168 K=16384 L=2  cublas_sep=49.72us mega=35.89us 1.39x
o_proj         N= 7168 K=16384 L=4  cublas_sep=88.64us mega=61.57us 1.44x
```

Interpretation:

- The persistent L-batch substrate is real. It can amortize launch/ramp and
  beat separate cuBLASLt when independent same-shape problems exist.
- It is not automatically a serving win, because most repeated projection
  shapes are repeated across sequential layers, not concurrently groupable.
- The production integration task is to find or create legal grouping sites.
  Do not claim the L-batch win for cross-layer dense/proj until the scheduler
  can actually present multiple same-shape independent projections together.

### BF16 Gate L-Batch Prototype

The directly groupable BF16 dense/proj site is the MLA output gate split:
two `[M,7168] x [8192,7168]^T` half-GEMMs with the same A and different B.

Local harness:

- `.bench_runs_claude/megakernel/bf16_gate_lbatch.py`

Results:

```text
M=32 current two half torch.mm: 48.18us
  best L-batch persistent: 48.15us, ratio=0.999

M=64 current two half torch.mm: 49.44us
  best L-batch persistent: 47.82us, ratio=0.967
  tactic=(False, (64, 64), (1, 1))
```

Code prototype:

- `trtllm::cute_dsl_bf16_gate_lbatch_blackwell`
- default-off hook: `TRTLLM_OPTRT_MLA_GATE_CUTEDSL_BF16_LBATCH=1`
- uses `PersistentDenseGemmKernel.wrapper_strided` with A batch stride 0, so
  hidden states are not duplicated
- writes directly into the normal `[M, 16384]` gate buffer via an as-strided
  `[M, 8192, 2]` view, so there is no output concat/copy

Source-tree runtime test blocker:

- importing local `/repo/tensorrt_llm` currently fails while loading
  `/repo/cpp/build/tensorrt_llm/thop/libth_common.so`
- missing symbol:
  `invokeMLALoadPagedKV<__nv_bfloat16, __nv_bfloat16>`
- rebuilding `th_common` completed, but the import error remained
- syntax check passes for the edited Python files

Production overlay:

- `localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-gate-lbatch-20260618T055904Z`
- keeps BEST_CONFIG core settings:
  - `TRTLLM_FORCE_COMM_METHOD=NVLINK_TWO_SIDED`
  - `TRTLLM_OPTRT_MOE_MEGAKERNEL=1`
  - `moe_config.use_low_precision_moe_combine=false`
- enables only:
  - `TRTLLM_OPTRT_MLA_GATE_CUTEDSL_BF16_LBATCH=1`

Direct op validation in the benchmark image:

```text
max_diff=0, cosine=1.0 versus the two-half torch.mm reference
M=32 ref=48.96us lbatch=48.09us ratio=0.982
M=64 ref=49.38us lbatch=48.54us ratio=0.983
```

Serving A/B, C32, ISL 2055, OSL 128, 64 requests:

```text
BEST_CONFIG warmed control:
  TTFT p50 1438 ms, p95 3202 ms
  user tok/s p50 38.45
  aggregate tok/s 834.5
  wall 9.8 s

gate_lbatch run 1:
  cold-start cliff: TTFT p50 22248 ms, aggregate tok/s 215.6
  exclude from steady comparison

gate_lbatch run 2:
  TTFT p50 1252 ms, p95 3304 ms
  user tok/s p50 37.48
  aggregate tok/s 825.3
  wall 9.9 s

gate_lbatch run 3:
  TTFT p50 1364 ms, p95 3200 ms
  user tok/s p50 37.93
  aggregate tok/s 839.9
  wall 9.8 s
```

128-step C32/OSL128 profile:

```text
trace: /var/lib/optrt-cache/nsys/gate_lbatch_c32_128_20260618T0639-rank-*.json
analysis: .bench_runs_claude/profiles/kineto_analysis_gate_lbatch_c32_128_20260618T0639/gate_lbatch_c32_128_20260618T0639_exhaustive.txt

baseline Dense/proj GEMM: 9092 us/iter, 640 launches/iter
gate_lbatch Dense/proj GEMM: 9397 us/iter, 579 launches/iter

top shifted kernel:
  kernel_cutlass_kernel_tensorrt_llm_torchcute_dsl_kernelsblackwelldense
  3719.6 us/iter, 183.0 calls/iter, 20.33 us/call
```

Interpretation:

- The launch-count thesis was correct for this one site: the production path
  dropped by exactly one dense/proj launch per decode layer.
- The time thesis did not hold: the two cuBLASLt half-GEMMs were already near
  the same cost as the one CuTe l-batch grid, so the dense/proj bucket stayed
  flat/slightly worse.
- This is still useful proof. It validates the L-batch integration mechanics,
  but it also says the BF16 gate split is not the high-upside dense/proj target
  unless a new gate-half kernel beats the cuBLASLt floor directly.
- The larger remaining target is the NVFP4 projection family or a fused
  BF16-input-to-NVFP4-projection boundary that removes quant/packing plus GEMM
  launch overhead together.

### Dense/Proj NVFP4 Split-K Projection Kernel

New dense/proj kernel direction: split the large K dimension of small-M NVFP4
projection GEMMs, run the K slices as the native L dimension of the persistent
dense kernel in one grid, then reduce the partial outputs. This targets the
actual o_proj GEMM occupancy issue instead of swapping among existing backends.

Harness:

- `.bench_runs_claude/megakernel/splitk_nvfp4_proj.py`
- source substrate: `Sm100BlockScaledPersistentDenseGemmKernel`
- result log with PyTorch reduction:
  `.bench_runs_claude/results/60_splitk_o_proj_m16_64.txt`
- result log with a custom CUDA BF16 reducer:
  `.bench_runs_claude/results/61_splitk_o_proj_cuda_reduce.txt`

Expanded o_proj sweep with PyTorch reduction, `N=7168`, `K=16384`, splits
`2,4,8`, six tactics:

```text
M=16 best split+reduce: 23.18us -> 20.97us, 1.11x, S=2, tactic=((128,128),(1,2),False)
M=32 best split+reduce: 23.33us -> 21.44us, 1.09x, S=2, tactic=((128,128),(1,1),False)
M=64 best split+reduce: 23.54us -> 22.20us, 1.06x, S=4, tactic=((128,128),(1,1),False)
```

Focused rerun with the custom CUDA BF16 reducer:

```text
M=16 best split+reduce: 22.20us -> 19.76us, 1.12x, S=2, tactic=((128,128),(1,2),False)
M=32 best split+reduce: 22.97us -> 20.37us, 1.13x, S=2, tactic=((128,128),(1,2),False)
M=64 best split+reduce: 22.82us -> 21.90us, 1.04x, S=2, tactic=((128,128),(1,1),False)
```

No-copy production-shaped rerun:

- result log:
  `.bench_runs_claude/results/62_splitk_o_proj_strided_window.txt`
- patched kernel source loaded into the harness with
  `TRTLLM_SPLITK_DENSE_KERNEL_SOURCE=/repo/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/dense_blockscaled_gemm_persistent.py`
- A/B shape: o_proj, `N=7168`, `K=16384`
- implementation change: inputs expose one ordinary full-K production FP4/SF
  tensor (`L=1`), C exposes `L=split`, and the TMA producer offsets the source
  K-block window by output split id. This avoids per-split activation/weight/SF
  packing and avoids invalid strided-L TMA descriptors.

Best no-copy split+custom-reduce results:

```text
M=16 best split+reduce: 22.08us -> 19.47us, 1.13x, S=2, tactic=((128,128),(1,1),False)
M=32 best split+reduce: 22.36us -> 20.74us, 1.08x, S=2, tactic=((128,128),(1,1),False)
M=64 best split+reduce: 23.83us -> 21.85us, 1.09x, S=2, tactic=((128,128),(1,1),False)
```

Correctness proxy versus full-K cuBLASLt:

```text
cosine ~= 0.999994
max_abs = 0.125 for M<=32, 0.25 for M=64 with S=2
```

Interpretation:

- This is the first dense/proj kernel branch that consistently beats the
  current full-K cuBLASLt o_proj floor after including the necessary reduction.
- Kernel-only speedups are larger, roughly 1.24-1.31x on the winning S=2
  tactics, which means reduction/output handling is now the main polish point.
- Over-splitting is not broadly useful. S=2 is the production candidate for
  M16/M32/M64; S=4 can help some M64 cases with PyTorch reduction but is not
  consistently better once using the custom reducer.
- The no-copy windowed split-K rerun removed the previous concern that
  production would need K-chunk re-quantization or scale-factor repacking. The
  kernel can consume the existing full-K swizzled FP4/SF tensors and select the
  split window in the TMA producer.
- This branch is principled because it increases independent tile work for the
  underfilled small-M/large-K projection instead of rerouting the same launch
  structure.

Production integration work:

1. Move the CUDA reducer out of the harness or fold the reduction into the
   split-K epilogue. The external BF16 reduction costs roughly 2-4 us here and
   is now the largest remaining gap between kernel-only and end-to-end.
2. Initial guarded route added:
   - kernel entry: `Sm100BlockScaledPersistentDenseGemmKernel.wrapper_splitk`
   - runner/op: `trtllm::cute_dsl_nvfp4_gemm_splitk_blackwell`
   - `nvfp4_gemm` gate: `TRTLLM_NVFP4_GEMM_SPLITK_O_PROJ=1`
   - shape guards: default `N=7168`, `K=16384`, `M<=64`, `split=2`
   - current version reduces partials with a Triton BF16 reducer;
     this is functional for A/B but still costs a second launch.
   - hook-level result log:
     `.bench_runs_claude/results/63_splitk_nvfp4_gemm_hook.txt`
   - Triton-reducer hook-level result log:
     `.bench_runs_claude/results/64_splitk_nvfp4_gemm_hook_triton_reduce.txt`

Hook-level validation after copying the patched Python files into the bench
container site-packages, with explicit `allowed_backends="cublaslt"` preserved
as the reference path:

```text
M=16 cublaslt=22.80us split_hook=20.62us ratio=1.11x cos=0.999994 max_abs=0.125
M=32 cublaslt=23.12us split_hook=21.21us ratio=1.09x cos=0.999994 max_abs=0.125
M=64 cublaslt=22.53us split_hook=21.45us ratio=1.05x cos=0.999994 max_abs=0.125
```

Hook-level rerun with the Triton BF16 reducer:

```text
M=16 cublaslt=22.75us split_hook=20.59us ratio=1.10x cos=0.999994 max_abs=0.125
M=32 cublaslt=23.78us split_hook=20.45us ratio=1.16x cos=0.999994 max_abs=0.125
M=64 cublaslt=23.29us split_hook=20.81us ratio=1.12x cos=0.999994 max_abs=0.125
```

3. Serving A/B on top of BEST_CONFIG. This arm deliberately forced the route
   even when production passed an explicit single backend, using
   `TRTLLM_NVFP4_GEMM_SPLITK_OVERRIDE_EXPLICIT=1`, because the first two
   serving overlays were inactive and never reached the split-K hook.
4. Initial allow-list was conservative:
   `N=7168`, `K=16384`, `M<=64`, `split=2`, tactic `((128,128),(1,1),False)`.
   The 128x256 tactics repeatedly lost in the no-copy sweep.

Route proof from decode worker logs:

```text
NVFP4 split-K o_proj selected for M=5, N=7168, K=16384, split=2, output_buffer_kind=0
NVFP4 split-K o_proj selected for M=4, N=7168, K=16384, split=2, output_buffer_kind=0
NVFP4 split-K o_proj selected for M=3, N=7168, K=16384, split=2, output_buffer_kind=0
```

Serving A/B, C16/C24/C32, ISL 2055, OSL 128:

```text
splitk_o_proj_override_mega_cf:
  image: optrt-529374445d-codex-splitk-o-proj-20260618T104428Z
  C16: TTFT p50 1281 ms, user tok/s p50 38.11, aggregate tok/s 497.2
  C24: TTFT p50 1583 ms, user tok/s p50 39.68, aggregate tok/s 642.2
  C32: TTFT p50 1530 ms, user tok/s p50 38.28, aggregate tok/s 788.4

BEST_CONFIG restored control:
  image: optrt-529374445d-vbfuse-nvfp4-20260615T043037Z
  C16: TTFT p50 1200 ms, user tok/s p50 42.54, aggregate tok/s 521.2
  C24: TTFT p50 1467 ms, user tok/s p50 39.79, aggregate tok/s 661.9
  C32: TTFT p50 1860 ms, user tok/s p50 38.45, aggregate tok/s 818.2
```

Production interpretation:

- The split-K `o_proj` route is real and reachable in the production serving
  path. The earlier inactive overlays should not be used as A/B evidence.
- The current two-launch implementation is not a serving win. It loses
  aggregate throughput at all three measured concurrencies, with C32 dropping
  from 818.2 to 788.4 aggregate output tok/s.
- The likely problem is not correctness or routing; it is the extra partial
  output write plus separate reduction launch on very small decode M buckets.
  The route proof showed M=3/4/5 selections, where the isolated 2-3 us
  reduction overhead can erase the split-K occupancy gain.
- Therefore the next principled kernel step is not another backend toggle. It
  is either a single-kernel split-K design with an in-kernel reduction/epilogue,
  or a fused BF16-input-to-NVFP4 projection path that removes quant/packing and
  projection launch overhead together.

Atomic split-K follow-up:

- Implemented an in-kernel atomic BF16 accumulation path for the same split-K
  o_proj route:
  - wrapper: `Sm100BlockScaledPersistentDenseGemmKernel.wrapper_splitk_atomic`
  - custom-op switch: `TRTLLM_NVFP4_GEMM_SPLITK_ATOMIC=1`
  - serving arm: `splitk_o_proj_atomic_mega_cf`
  - image: `optrt-529374445d-codex-splitk-o-proj-atomic-20260618T111927Z`
- This still zero-initializes the output before the GEMM, but removes the
  `[split,M,N]` partial allocation and the separate BF16 reduction launch.
- Hook-level production-op timing improved over both cuBLASLt and the external
  reducer:

```text
M=16 base=22.53us external=19.00us atomic=18.27us atomic_ratio=1.23x
M=32 base=22.64us external=19.36us atomic=17.96us atomic_ratio=1.26x
M=64 base=23.42us external=21.19us atomic=19.34us atomic_ratio=1.21x
```

Production route proof from decode logs:

```text
NVFP4 split-K o_proj selected for M=15, N=7168, K=16384, split=2, atomic=True, output_buffer_kind=0
NVFP4 split-K o_proj selected for M=14, N=7168, K=16384, split=2, atomic=True, output_buffer_kind=0
NVFP4 split-K o_proj selected for M=13, N=7168, K=16384, split=2, atomic=True, output_buffer_kind=0
NVFP4 split-K o_proj selected for M=12, N=7168, K=16384, split=2, atomic=True, output_buffer_kind=0
```

Atomic serving repeats, C16/C24/C32, ISL 2055, OSL 128:

```text
BEST_CONFIG restored control:
  C16 aggregate tok/s 521.2
  C24 aggregate tok/s 661.9
  C32 aggregate tok/s 818.2

splitk_o_proj_atomic_mega_cf run 1:
  C16 aggregate tok/s 531.6
  C24 aggregate tok/s 683.4
  C32 aggregate tok/s 833.6

splitk_o_proj_atomic_mega_cf run 2:
  C16 aggregate tok/s 523.0
  C24 aggregate tok/s 693.0
  C32 aggregate tok/s 836.2

splitk_o_proj_atomic_mega_cf run 3:
  C16 aggregate tok/s 504.6
  C24 aggregate tok/s 685.4
  C32 aggregate tok/s 843.6
```

Interpretation after atomic follow-up:

- The principled split-K kernel direction is now serving-positive at the C32
  acceptance point in three warm repeats: 833.6/836.2/843.6 aggregate tok/s
  versus the fresh restored BEST_CONFIG control at 818.2.
- C24 is also repeat-positive: 683.4/693.0/685.4 versus 661.9.
- C16 is noisy/flat: 531.6/523.0/504.6 versus 521.2.
- The remaining overhead is the required output zeroing before atomic
  accumulation. A split0-store/split1-atomic design would avoid zeroing, but
  cannot be correct without ordering between split CTAs; a safe version needs a
  tile-local ordered split scheduler or a different epilogue contract.
- Next proof should be a 128-step Kineto profile of the atomic arm against the
  same restored control shape to verify the dense/proj bucket actually moves,
  not only the end-to-end aggregate number.

Nsight Compute note:

- Host `ncu` exists (`2026.1.1.0`), but profiling the containerized `docker exec`
  child did not attach to kernels (`No kernels were profiled`).
- The benchmark container itself does not have `ncu` on PATH.
- Do not report SOL% for this bucket until the container/tooling path is fixed.

## Phase-3 WARPDECODE MoE V2 Attempt

Goal: test the existing phase-3 persistent MoE kernel path against the production BEST_CONFIG
instead of stopping at a one-off microbench. The target was the exposed MoE FC1/FC2 chain from
the C32/OSL128 traces:

```text
blockscaled_contiguous_gather_grouped_gemm_act_fusion
blockscaled_contiguous_grouped_gemm_finalize_fusion
```

Initial result:

- Adding `TRTLLM_OPTRT_MOE_MEGAKERNEL_V2=1` on top of BEST_CONFIG did not change the trace.
- Root cause: `TRTLLM_OPTRT_MOE_MEGAKERNEL=1` returned through the phase-1 hook before the V2
  hook, so V2 was unreachable with the proven config.

Patch tested:

- Move the V2 gate before the phase-1 hook in `fused_moe_cute_dsl.py`.
- Suppress phase-1 only when V2 is explicitly enabled.
- Overlay image: `optrt-529374445d-codex-mega-v2-precedence-20260618T040332Z`.

Serving A/B after making V2 reachable:

```text
mega_v2_precedence_c32_osl128:
  C32 / ISL 2055 / OSL 128 / 64 requests
  TTFT p50 1598 ms, p95 3622 ms
  user tok/s p50 38.38
  aggregate tok/s 788.9
  wall 10.4 s

mega_v2_precedence_c32_osl128_r2:
  C32 / ISL 2055 / OSL 128 / 64 requests
  TTFT p50 1423 ms, p95 3617 ms
  user tok/s p50 39.45
  aggregate tok/s 811.3
  wall 10.1 s

Restored BEST_CONFIG reference from the same shape:
  aggregate tok/s ~844-847
```

128-step Kineto trace:

```text
trace: /var/lib/optrt-cache/nsys/mega_v2_precedence_c32_128_20260618T0421-rank-0.json
iters: 128
median decode step: 17894 us
median GPU idle: 625 us/iter (3.5%)

Dense/proj GEMM: 11723 us/iter, 1159 launches/iter
MoE a2a/comm:    4206 us/iter, 290 launches/iter
MoE expert GEMM: 2387 us/iter, 174 launches/iter
```

Important trace proof:

- No new persistent V2 kernel name appeared in the patched trace.
- The old FC1/FC2 kernels still launched at `58.0 launches/iter` each.
- Compared with the V2-unreachable trace, those kernels got faster, but were not eliminated:

```text
v2_unreachable:
  FC1 gather/act/quant: 1773.3 us/iter, 58.0 launches/iter
  FC2 finalize:          988.2 us/iter, 58.0 launches/iter

v2_precedence:
  FC1 gather/act/quant: 1361.2 us/iter, 58.0 launches/iter
  FC2 finalize:          772.5 us/iter, 58.0 launches/iter
```

Interpretation:

- The non-DWDP V2 hook is the wrong production integration point for this config.
- BEST_CONFIG is running the DWDP/multi-B Cute DSL path:
  `cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell_multi_b` followed by
  `cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell` with weight lists.
- Current `run_mega_persistent_moe_v2(...)` accepts a single `w13/w2` tensor pair and cannot
  replace the DWDP multi-B chain.
- Therefore the principled kernel work is to make the persistent MoE kernel DWDP-aware:
  accept weight/scale lists or an equivalent packed descriptor, preserve the output-owned
  fused-finalize contract, and then re-run the 128-step C32 trace looking for the FC1/FC2
  launch pair to disappear.

## Acceptance Bar

1. Exactness:
   - bit-exact packed FP4 bytes versus the current two-op reference
   - bit-exact swizzled scale factors versus reference
   - tested across representative `(M, N)` including decode shapes around C32
2. Microbench:
   - fused boundary must beat `fused_sigmoid_mul + fp4_quantize` in isolated timing
3. Serving:
   - guard with env var default-off
   - deploy only as an experiment on top of BEST_CONFIG
   - C32 must beat the restored control outside run-to-run noise
4. Regression check:
   - confirm gate overlap is still active
   - confirm existing WARPDECODE/MoE megakernel config is unchanged

## Next Steps

Dense/proj comes first because it is the largest measured bucket:

1. Keep L-batch work constrained to legal grouping sites:
   - BF16 gate halves are production-tested and launch-reducing, but not a
     throughput win with the current CuTe persistent substrate
   - cross-layer repetition does not count unless the scheduler can present
     independent projections together
   - for NVFP4, do not integrate until the module/scheduler boundary can pass
     L packed activations, weight pointers/scales, and output destinations
     without extra packing on the critical path
2. Continue the quant/projection fusion path where L-batching is not legal:
   - choose one shape family first, likely `o_proj` or `q_b_proj`
   - preserve cuBLASLt/nvjet as the baseline and compare against it directly
   - target removed quant/packing + GEMM launch overhead, not a backend toggle
   - keep hooks env-gated and default-off
3. Fix the Nsight Compute path:
   - make `ncu` available inside the benchmark container, or run the workload
     outside `docker exec` so host `ncu` attaches to kernels
   - collect `SpeedOfLight`, `LaunchStats`, and `Occupancy` for the top
     cuBLASLt/nvjet projection kernels before claiming a roofline diagnosis
4. Gate path is only worth continuing if a new prototype beats the per-half
   cuBLASLt floor:
   - BF16 gate-half floor is ~28-30 us for `[M,7168] x [8192,7168]^T`
   - existing CuTe BF16 GEMM is slower
   - tiled DSV3 fused-A is slower at M32/M64
5. Keep DWDP/multi-B MoE V2 as the secondary track:
   - accept the same weight/scale list contract used by the current DWDP Cute DSL ops
   - preserve output-owned fused finalize
   - old FC1/FC2 Cute DSL launches should disappear or materially drop
6. Benchmark C32/OSL128 outside profiler, then compare:
   - TTFT p50/p95
   - stable output tok/s/user
   - aggregate output tok/s
   - dense/proj GEMM bucket time and launch count
   - MoE expert GEMM bucket time and launch count when MoE V2 is touched
7. If serving regresses, collect traces before backing out:
   - confirm whether the op fires
   - confirm exact shape/backend/tactic for the replaced projections
   - inspect whether saved projection time is hidden by comm/indexer or graph sync

## 2026-06-18 Atomic Split-K O-Proj Profile

Artifacts:

```text
trace: /var/lib/optrt-cache/nsys/splitk_atomic_c32_128_20260618T114722Z-rank-{0,1,2,3}.json
analysis: .bench_runs_claude/profiles/kineto_analysis_splitk_atomic_c32_128_20260618T114722Z/splitk_atomic_c32_128_20260618T114722Z_exhaustive.txt
control: .bench_runs_claude/profiles/kineto_analysis_c32_128_actual_20260617T203140Z/c32_128_actual_20260617T203140Z_exhaustive.txt
```

Serving before the profile:

```text
BEST_CONFIG restored control, C32/ISL2055/OSL128:
  agg tok/s: 818.2

atomic split-K o_proj, C32/ISL2055/OSL128:
  r1 agg tok/s: 833.6
  r2 agg tok/s: 836.2
  r3 agg tok/s: 843.6
```

Route proof from the production decode pod:

```text
NVFP4 split-K o_proj selected for M=4, N=7168, K=16384, split=2, atomic=True, output_buffer_kind=0
NVFP4 split-K o_proj selected for M=3, N=7168, K=16384, split=2, atomic=True, output_buffer_kind=0
```

Profile comparison:

```text
BEST_CONFIG C32/128-step aggregate:
  total GPU kernel time: 24042 us/iter
  Dense/proj GEMM:       9102 us/iter, 640 launches/iter
  MoE a2a/comm:          3445 us/iter, 290 launches/iter
  MoE expert GEMM:       3067 us/iter, 174 launches/iter
  Quant/dequant:         1125 us/iter, 392 launches/iter

atomic split-K o_proj C32/128-step aggregate:
  total GPU kernel time: 23424 us/iter
  Dense/proj GEMM:       9005 us/iter, 640 launches/iter
  MoE a2a/comm:          3293 us/iter, 290 launches/iter
  MoE expert GEMM:       2927 us/iter, 174 launches/iter
  Quant/dequant:         1158 us/iter, 392 launches/iter
```

The o_proj split-K path is real but too narrow:

```text
control old block-scaled cutlass path:
  cutlass3x_sm100_bstensorop_s256x128x64gemm_block_scaled_ue4m3xf4_ue4m3
  1319.1 us/iter, 64.0 launches/iter

atomic split-K replacement/residual:
  kernel_cutlass_kernel_tensorrt_llm_torchcute_dsl_kernelsblackwelldense
  1134.9 us/iter, 61.0 launches/iter

  residual old cutlass path:
  64.1 us/iter, 3.0 launches/iter
```

So this replacement saves only about 120 us/iter inside the dense/proj bucket.
The bucket moves from 9102 to 9005 us/iter, which is real but not the scale of
win needed. The same 640 launches/iter remain because this is a per-layer
backend substitution, not a cross-layer or projection-family launch reduction.

The two dominant dense/proj families were not solved:

```text
control:
  nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT
    2858.7 us/iter, 122.0 launches/iter
  nvjet_sm100_ootst_128x128_256x7_4x1_2cta_v_bx_Avec16UE4M3_Bvec16UE4M3_
    2829.7 us/iter, 193.0 launches/iter

atomic split-K:
  nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT
    2860.6 us/iter, 122.0 launches/iter
  nvjet_sm100_ootst_128x128_256x7_4x1_2cta_v_bx_Avec16UE4M3_Bvec16UE4M3_
    2848.8 us/iter, 193.0 launches/iter
```

Interpretation:

- Atomic split-K o_proj is worth keeping as an experiment because it is
  production-reachable, correct in microbench, and serving-positive at C24/C32.
- It is not the main dense/proj answer. It changes the 64-call block-scaled
  o_proj slice but leaves the two biggest NVJIT families essentially unchanged.
- The next principled kernel target is one of those two NVJIT families, after
  mapping each family back to exact module/shape sites under production load.

Next kernel target:

1. Add low-volume projection-site shape/tactic logging around `NVFP4LinearMethod`
   and the custom cublas path, enough to map the `122` and `193` call families
   to module names and `(M, N, K)` without perturbing benchmark timing.
2. Prototype against the higher-value family first. The acceptance bar is not
   just a faster microbench: the C32/128-step dense/proj bucket must move by
   several hundred us/iter and the top NVJIT family must shrink in the kernel
   table.
3. Keep the split-K o_proj hook env-gated and default-off while this broader
   projection-family kernel is built.

## 2026-06-18 NVFP4 Production Site Mapping

Debug overlay:

```text
localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-nvfp4-debug-20260618T1414
```

Artifacts:

```text
.bench_runs_claude/results/81_nvfp4_debug_shape_sites.txt
.bench_runs_claude/results/82_nvfp4_debug_shape_summary.txt
```

The debug overlay enabled only shape/site logging:

```text
TRTLLM_NVFP4_LINEAR_DEBUG=1
TRTLLM_NVFP4_GEMM_DEBUG_SHAPES=1
```

The production trigger was intentionally not a throughput run. It was only to
force CUDA graph warmup/replay and collect unique NVFP4 call sites.

Mapped NVFP4 sites:

```text
MLA projection linears, all 61 layers:
  model.layers.*.self_attn.kv_a_proj_with_mqa  N=2112   K=7168
  model.layers.*.self_attn.q_b_proj            N=24576  K=1536
  model.layers.*.self_attn.o_proj              N=7168   K=16384

DSA sparse indexer:
  dsa_indexer.layer*.wq_b                       N=8192   K=1536
  dsa_indexer.layer*.fused_wk_weights_proj      N=192    K=7168

MLP/shared experts:
  model.layers.*.mlp.shared_experts.down_proj   N=7168   K=2048
  model.layers.0-2.mlp.gate_up_proj             N=36864  K=7168
  model.layers.0-2.mlp.down_proj                N=7168   K=18432
```

The formerly unlabeled `N=192,K=7168` GEMM is the DSA indexer fused
`wk + weights_proj` path in
`tensorrt_llm/_torch/attention_backend/sparse/dsa.py::_FusedWkWpNvfp4`.
It bypasses `Linear.apply`, so the normal `Linear` debug hook could not name
it. The formerly generic `N=8192,K=1536` `Linear` is DSA indexer `wq_b`.

I added a debug-only label patch in `dsa.py`:

```text
dsa_indexer.layer{layer_idx}.wq_b
dsa_indexer.layer{layer_idx}.wk
dsa_indexer.layer{layer_idx}.weights_proj
dsa_indexer.layer{layer_idx}.fused_wk_weights_proj
```

`python3 -m py_compile tensorrt_llm/_torch/attention_backend/sparse/dsa.py`
passes.

Current diagnosis:

- The o_proj split-K experiment only attacks the `N=7168,K=16384` block-scaled
  o_proj slice. It helps, but it cannot move the 193-call `nvjet_sm100_ootst`
  family because that family is mostly other 128x128 NVFP4 projection shapes.
- The next real kernel target is a family-level small-M NVFP4 projection path,
  not another isolated o_proj variant. The highest-confidence candidates are
  DSA indexer fused `N=192,K=7168`, DSA `wq_b N=8192,K=1536`, and MLA
  `kv_a_proj_with_mqa N=2112,K=7168`, because these are repeated across the
  layer stack and map to the dominant 128x128 cublasLt/NVJIT family.
- A cross-layer L-batch launch-reduction is unlikely to be legal without
  scheduler/model-graph surgery because these projections are layer-sequential.
  The principled route is either a shape-family CuTe kernel that beats cublasLt
  at small M, or a fused quantize+GEMM path for the repeated FP4 projection
  shapes so the quant/dequant bucket and dense/proj bucket move together.

## 2026-06-18 Split-K Generalization Check

I added the newly mapped production shapes to the split-K microbench harness:

```text
.bench_runs_claude/megakernel/splitk_nvfp4_proj.py
```

New shapes:

```text
dsa_fused_wk_wp  N=192   K=7168
dsa_wq_b         N=8192  K=1536
shared_down      N=7168  K=2048
```

Results:

```text
.bench_runs_claude/results/83_splitk_mapped_family_quick.txt
.bench_runs_claude/results/84_splitk_mapped_family_wqb_shared.txt
```

Summary, best split+reduce per M:

```text
kv_a_proj N=2112,K=7168:
  M=1   full=15.88us split+reduce=17.40us ratio=0.91x
  M=2   full=15.77us split+reduce=16.72us ratio=0.94x
  M=4   full=15.96us split+reduce=15.66us ratio=1.02x
  M=8   full=15.68us split+reduce=17.16us ratio=0.91x
  M=16  full=16.60us split+reduce=17.14us ratio=0.97x
  M=32  full=15.92us split+reduce=16.80us ratio=0.95x
  M=64  full=16.16us split+reduce=16.87us ratio=0.96x

dsa_fused_wk_wp N=192,K=7168:
  M=1   full=15.93us split+reduce=15.87us ratio=1.00x
  M=2   full=15.37us split+reduce=15.42us ratio=1.00x
  M=4   full=15.67us split+reduce=16.47us ratio=0.95x
  M=8   full=15.46us split+reduce=15.87us ratio=0.97x
  M=16  full=15.84us split+reduce=15.92us ratio=0.99x
  M=32  full=15.33us split+reduce=16.14us ratio=0.95x
  M=64  full=15.66us split+reduce=16.52us ratio=0.95x

dsa_wq_b N=8192,K=1536:
  all M=1..64 are losses, best ratios 0.73x-0.81x

shared_down N=7168,K=2048:
  all M=1..64 are losses, best ratios 0.74x-0.87x
```

Verdict:

- Split-K is not the broad answer for the 193-call mapped family. The kernel
  half can be slightly faster on large-K shapes, but reduction/zeroing removes
  the gain. On short-K shapes (`dsa_wq_b`, `shared_down`) even the split kernel
  is worse.
- Do not wire split-K to these mapped sites in production. Keep the o_proj
  split-K arm isolated as the one serving-positive special case.
- Continue with legal L-batch/fused-same-input opportunities or a true
  single-kernel FP4 quantize+GEMM path.

## 2026-06-18 DSA Pre-KV FP4 Reuse Check

Status update from the dense/proj GEMM campaign:

- The broad split-K path is not viable for the mapped 193-call NVFP4 family, so I moved to the repeated same-input quantization path around MLA + DSA.
- `kv_a_proj_with_mqa` can already consume a prequantized FP4 payload from the input gated norm when `TRTLLM_OPTRT_GATED_PREKV_QUANT=1`.
- The DSA fused `wk+weights_proj` indexer consumes the same BF16 hidden states, but the checkpoint has no indexer activation input scales:
  - `indexer.wk.input_global_scale`: 0 tensors
  - `indexer.weights_proj.input_global_scale`: 0 tensors
  - `indexer.wq_b.input_global_scale`: 0 tensors
  - `kv_a_proj_with_mqa.input_global_scale`: 61 tensors
  - `self_attn.indexer.*.weight_global_scale`: 183 tensors
- Layer-0 concrete values from the mounted checkpoint:
  - `kv_a_proj_with_mqa.input_global_scale = 430.0`
  - `indexer.wk.weight_global_scale = 24576.0`
  - `indexer.weights_proj.weight_global_scale = 11904.0`

Implication:

The safe exact reuse path is blocked for this checkpoint: `kv_a` is static-scale FP4, while the indexer fused `wk/wp` path is dynamic-activation FP4. Reusing `kv_a`'s FP4 bytes directly changes indexer numerics unless we deliberately force the indexer to use `kv_a`'s static activation scale.

Code added behind default-off gates:

```text
TRTLLM_OPTRT_GATED_PREKV_QUANT=1
TRTLLM_INDEXER_REUSE_PREKV_FP4=1
TRTLLM_INDEXER_REUSE_PREKV_FP4_STATIC=1   # required for this dynamic-indexer checkpoint
TRTLLM_INDEXER_REUSE_PREKV_FP4_DEBUG=1    # optional one-shot reason logs
```

The path passes the existing pre-KV `Fp4QuantizedTensor` and `kv_a.input_scale` into `Indexer.pre_indexer_proj()`. `_FusedWkWpNvfp4` only skips its local `fp4_quantize` when the flags above allow it. Without the STATIC flag, this checkpoint logs `dynamic_indexer` and falls back to the current path.

Next validation needed before any serving A/B:

1. Microbench the forced-static indexer path against current dynamic indexer for time.
2. Compare DSA `wk/wp` outputs and downstream top-k stability on real layer weights.
3. Only if top-k remains stable, deploy an overlay with pre-KV quant + static reuse and run C=32 serving throughput.

First real-weight microbench:

```text
.bench_runs_claude/results/85_dsa_prekv_reuse_compare.txt
```

Layer-0 fused indexer `wk+weights_proj`, backend `cublaslt`, real FP4 weights,
random BF16 hidden states normalized to fractions/multiples of the kv_a
calibration amax:

```text
kv_global_scale = 430.0
kv_calib_amax  = 6.251163
fused weight   = (192, 3584)

amax_factor=0.5:
  static_gemm is 1.67x-2.79x faster than dynamic quant+gemm
  wk_cos >= 0.99985993, wp_cos >= 0.99979764

amax_factor=1.0:
  static_gemm is 1.78x-2.76x faster than dynamic quant+gemm
  wk_cos >= 0.99993563, wp_cos >= 0.99986857

amax_factor=2.0:
  static_gemm is 1.77x-2.67x faster than dynamic quant+gemm
  wk_cos drops as low as 0.98395497, wp_cos drops as low as 0.98337930
```

Verdict:

- The speed lever is real: skipping the indexer-local amax+fp4_quantize cuts
  the fused indexer projection from roughly 27-41 us to roughly 15-16 us in
  this harness.
- Numerics are acceptable only when the hidden-state amax is near or below the
  kv_a calibration range. At 2x the calibration amax, output drift is too large
  to deploy without a real top-k stability check.
- Next step is to measure real decode hidden-state amax distribution or compare
  downstream DSA top-k stability with captured hidden states, not to jump
  straight to serving.

Second validation: standalone top-k stability.

```text
.bench_runs_claude/megakernel/dsa_prekv_topk_stability.py
.bench_runs_claude/results/86_dsa_prekv_topk_stability.txt
.bench_runs_claude/results/87_dsa_prekv_topk_stability_32k.txt
```

The 4k-cache run used 4096 cache tokens, 32 query tokens, top-1024. It showed
high token recall near calibration but block recall was saturated because 4096
tokens at 128/token blocks gives only 32 blocks, and top-1024 touches all of
them. The meaningful run is the 32k-cache sweep:

```text
cache_tokens=32768, query_tokens=16, topk=1024

factor=0.50:
  logits_cos=0.99987733
  token_recall_mean=0.985352, token_recall_min=0.955078
  block_recall_mean=0.999008, top1_match=1.000000

factor=1.00:
  logits_cos=0.99990547
  token_recall_mean=0.987366, token_recall_min=0.978516
  block_recall_mean=0.999009, top1_match=1.000000

factor=1.25:
  logits_cos=0.99466550
  token_recall_mean=0.885681, token_recall_min=0.819336
  block_recall_mean=0.992765, top1_match=0.562500

factor=1.50:
  token_recall_mean=0.873962, token_recall_min=0.695312
  block_recall_mean=0.993279, top1_match=0.750000

factor=2.00:
  token_recall_mean=0.895264, token_recall_min=0.709961
  block_recall_mean=0.995029, top1_match=0.625000
```

Verdict:

- The forced-static reuse is plausible only if real decode hidden amax is
  usually at or below the kv_a calibration amax. Near calibration, top-1024
  token recall is ~0.986 and top-1 is unchanged in this harness.
- Above ~1.25x calibration amax, token selection drift is too high for a safe
  serving A/B without stronger evidence.
- Added debug-only runtime amax logging:

```text
TRTLLM_OPTRT_GATED_PREKV_QUANT=1
TRTLLM_INDEXER_REUSE_PREKV_FP4=1
TRTLLM_INDEXER_REUSE_PREKV_FP4_AMAX_DEBUG=1
TRTLLM_INDEXER_REUSE_PREKV_FP4_AMAX_DEBUG_LIMIT=8

# Keep this OFF for amax sampling, so behavior stays current-path except for
# pre-KV quant itself:
TRTLLM_INDEXER_REUSE_PREKV_FP4_STATIC=0
```

The logger prints `amax_over_static_calib`; values <= 1.0 are in the stable
region from the harness, while values >= 1.25 are the danger zone.

Runtime amax sampling result:

```text
image:
  localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-prekv-amax-debug-cgsafe-20260618T1509

deploy:
  .bench_runs_claude/results/91_deploy_prekv_amax_debug_cgsafe.txt

decode log:
  .bench_runs_claude/results/92_decode_prekv_amax_debug_cgsafe.log

sample reduction:
  concrete amax samples: 9728
  capture-skipped samples: 1216
  min / avg / max amax_over_static_calib:
    0.0446792 / 0.237276 / 0.515625
  samples >= 1.0: 0
  samples >= 1.25: 0

top max buckets:
  layer44 M=2  max=0.515625
  layer56 M=7  max=0.510858
  layer56 M=6  max=0.508847
  layer40 M=2  max=0.508464
  layer56 M=4  max=0.504825
```

The first amax-debug image crashed decode warmup because the diagnostic path
called `.item()` while CUDA graph capture was active. The cgsafe image guards
amax logging with `torch.cuda.is_current_stream_capturing()`, reached ready
state, and preserved the production CUDA graph path. The runtime distribution
supports trying the forced-static pre-KV reuse A/B next.

## Production A/B Result: DSA Pre-KV Static Reuse

Serving arm:

```text
prekv_static_mega_cf
image:
  localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-prekv-amax-debug-cgsafe-20260618T1509

env delta on top of BEST_CONFIG:
  TRTLLM_OPTRT_GATED_PREKV_QUANT=1
  TRTLLM_INDEXER_REUSE_PREKV_FP4=1
  TRTLLM_INDEXER_REUSE_PREKV_FP4_STATIC=1
```

C32/ISL2055/OSL512 result:

```text
prekv_static_mega_cf:
  TTFT p50 1961 ms, p95 3686 ms
  user tok/s p50 50.73
  aggregate tok/s 1339.0

prekv_static_c32_hot1:
  TTFT p50 1988 ms, p95 3413 ms
  user tok/s p50 50.85
  aggregate tok/s 1348.7

profile-drive warm rounds:
  round1 aggregate tok/s 1344.0, user tok/s p50 50.84
  round2 aggregate tok/s 1410.2, user tok/s p50 50.81
  round3 aggregate tok/s 1105.9, user tok/s p50 50.01  # profile active
```

Compared with the latest restored control sanity at C32/OSL512:

```text
BEST_CONFIG:
  TTFT p50 1447 ms, p95 3370 ms
  user tok/s p50 50.35
  aggregate tok/s 1402.6
```

Interpretation:

- The forced-static DSA reuse is numerically plausible and safe enough to test
  under the measured amax distribution.
- It is not a reliable production throughput win at the C32 acceptance point.
  Per-user decode is roughly flat, but aggregate throughput is below or only
  noise-level competitive with BEST_CONFIG.
- Keep the code behind default-off gates. Do not add it to BEST_CONFIG.

Profile result for the static arm:

```text
profile:
  .bench_runs_claude/profiles/kineto_analysis_prekv_static_c32_512_20260618T1550Z/prekv_static_c32_512_20260618T1550Z_exhaustive.txt

aggregate static profile, C32/OSL512:
  Dense/proj GEMM       9370 us/iter, 39.5%, 640 launches/iter
  MoE a2a/comm          3446 us/iter, 14.5%, 290 launches/iter
  MoE expert GEMM       3257 us/iter, 13.7%, 174 launches/iter
  GEMM generic          1997 us/iter,  8.4%, 403 launches/iter
  DSA indexer           1743 us/iter,  7.3%, 263 launches/iter
  Elementwise/copy      1396 us/iter,  5.9%, 784 launches/iter
  Quant/dequant          771 us/iter,  3.3%, 257 launches/iter
  Attn generic           759 us/iter,  3.2%, 122 launches/iter
  Reduce/allreduce       573 us/iter,  2.4%, 133 launches/iter
```

Closest prior C32 profile for comparison:

```text
profile:
  .bench_runs_claude/profiles/kineto_analysis_c32_128_actual_20260617T203140Z/c32_128_actual_20260617T203140Z_exhaustive.txt

control profile, C32/OSL128:
  Dense/proj GEMM       9092 us/iter, 37.8%, 640 launches/iter
  MoE a2a/comm          3440 us/iter, 14.3%, 290 launches/iter
  MoE expert GEMM       3163 us/iter, 13.2%, 174 launches/iter
  GEMM generic          1944 us/iter,  8.1%, 403 launches/iter
  DSA indexer           1784 us/iter,  7.4%, 263 launches/iter
  Elementwise/copy      1638 us/iter,  6.8%, 992 launches/iter
  Quant/dequant         1125 us/iter,  4.7%, 392 launches/iter
  Attn generic           757 us/iter,  3.2%, 122 launches/iter
  Reduce/allreduce       700 us/iter,  2.9%, 149 launches/iter
```

The static reuse did reduce quant/dequant launches, but the production-critical
picture did not move enough. Dense/proj GEMM stayed at 640 launches/iter and
remained the dominant bucket. The largest kernel in the static trace is still
the MLA output-gate projection:

```text
nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT
  static trace: 3224 us/iter, 122 calls/iter, 26.42 us/call
  control trace: 2860 us/iter, 122 calls/iter, 23.44 us/call
```

Conclusion:

DSA pre-KV reuse is a scoped side optimization. It is not the missing
megakernel. The next principled target remains the dense/proj GEMM bucket,
especially the MLA output-gate projection and the surrounding projection/fusion
schedule, because that bucket is still ~9 ms/iter and ~38-40% of GPU kernel
time after the DSA experiment.

## Production A/B Result: q_b/wq_b Shared FP4 Reuse

Serving arm:

```text
image:
  localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-qb-wqb-reuse-20260618T164432Z

env delta:
  TRTLLM_INDEXER_REUSE_QB_FP4=1
  TRTLLM_INDEXER_REUSE_QB_FP4_STATIC=1
```

Rationale:

- `q_b_proj` and DSA `wq_b` consume the same hidden state and both land in the
  large NVFP4 projection family.
- The microbench showed a clear pair-level win from sharing q_b's quantized
  activation and scale for wq_b:

```text
pair-level speedup: ~1.5x-2.0x
q_b accuracy: exact
wq_b accuracy: close near calibration
runtime amax ratio: mostly below 1.0, observed max ~1.016
```

Serving result at C16/C24/C32:

```text
BEST_CONFIG fresh sanity:
  C24 aggregate tok/s 658.9
  C32 aggregate tok/s 820.6
  C16 row was cold/noisy; older warm C16 was ~521.2

q_b/wq_b static r1:
  C16 aggregate tok/s 518.8
  C24 aggregate tok/s 663.3
  C32 aggregate tok/s 795.6

q_b/wq_b static r2:
  C16 aggregate tok/s 506.9
  C24 aggregate tok/s 668.7
  C32 aggregate tok/s 744.7

q_b/wq_b static C32 hot:
  C32 aggregate tok/s 787.0
```

Verdict:

- The duplicated quantization hypothesis is real in isolation, but this direct
  production integration is not a win at the C32 acceptance point.
- The most likely issue is not numerical safety; it is integration shape:
  direct raw-op dispatch, graph/tactic selection, output-buffer behavior, or
  interaction with the surrounding DSA/top-k schedule.
- Keep the code default-off. Revisit only with a targeted profile that proves
  the large 193-call NVFP4 family moved in the right direction under the full
  production path.

## Gate BF16 Tactic Recheck

Artifact:

```text
.bench_runs_claude/results/123_bf16_gate_tactic_sweep_m32_m64_quick.txt
```

Quick tactic sweep, per-half gate projection shapes:

```text
RESULT M=32 torch_us=28.91 default_cute_us=30.44 best_cute_us=30.75 \
  best_ratio=1.064 best_tactic=Tactic(use_2cta=False, mn=(64, 64), cluster=(1, 2)) \
  best_cos=1.000000 best_max_abs=0.0000

RESULT M=64 torch_us=29.48 default_cute_us=30.71 best_cute_us=30.23 \
  best_ratio=1.025 best_tactic=Tactic(use_2cta=False, mn=(64, 64), cluster=(1, 2)) \
  best_cos=1.000000 best_max_abs=0.0000
```

Verdict:

- Existing CuTe BF16 gate runner/tactics do not beat the current cuBLASLt/nvjet
  per-half gate path at M32 or M64.
- This matches the serving experiments where full-gate and l-batch variants
  disturbed overlap more than they saved launch overhead.
- Do not wire the existing BF16 runner into production. Gate needs either a
  genuinely shape-specialized kernel with better co-run behavior or a broader
  fusion target that removes useful surrounding work, not another wrapper-level
  toggle.

## q_b + wq_b Same-Input Fused Projection Prototype

New branch:

```text
tensorrt_llm/_torch/attention_backend/sparse/dsa.py
tensorrt_llm/_torch/modules/attention.py
.bench_runs_claude/megakernel/qb_wqb_concat_projection.py
```

Env gates:

```text
TRTLLM_INDEXER_FUSE_QB_WQB=1
TRTLLM_INDEXER_FUSE_QB_WQB_STATIC=1
TRTLLM_INDEXER_FUSE_QB_WQB_POST_SCALE=0  # default: compensate via small weights tensor
```

Rationale:

- `q_b_proj` and DSA indexer `wq_b` both consume the post-`q_a_layernorm`
  q-lora activation.
- Their local production shapes concatenate cleanly:

```text
q_b_proj  M x 1536  -> 24576
wq_b      M x 1536  ->  8192
fused     M x 1536  -> 32768
```

- This is stronger than the previous q_b/wq_b shared-FP4 reuse check because
  it can remove a real `wq_b` GEMM launch, not just reuse activation quant.
- The class mirrors the existing `wk + weights_proj` fusion pattern:
  concatenate packed FP4 weights and swizzled scales along N, run one NVFP4
  GEMM with q_b's alpha, then split outputs.

Microbench artifact:

```text
.bench_runs_claude/results/124_qb_wqb_concat_projection.txt
.bench_runs_claude/results/125_qb_wqb_concat_projection_raw.txt
```

Corrected fused path, with explicit full `[M,8192]` `wq_b` post-scale:

```text
M=1  separate=18.73us fused=17.49us speedup=1.071
M=2  separate=18.89us fused=19.36us speedup=0.976
M=4  separate=18.25us fused=19.34us speedup=0.943
M=8  separate=18.57us fused=19.47us speedup=0.954
M=16 separate=18.36us fused=19.29us speedup=0.952
M=32 separate=18.14us fused=18.88us speedup=0.961
M=64 separate=18.12us fused=19.31us speedup=0.938
```

The explicit post-scale consumes the win. The better variant leaves the
`wq_b` BF16 slice unscaled and compensates the sparse-indexer logits by
multiplying the much smaller `[M,n_heads]` weights tensor:

```text
M=1  separate=18.21us raw_fused=15.93us raw_speedup=1.143
M=2  separate=18.93us raw_fused=16.23us raw_speedup=1.166
M=4  separate=18.38us raw_fused=15.70us raw_speedup=1.171
M=8  separate=18.21us raw_fused=15.42us raw_speedup=1.181
M=16 separate=18.49us raw_fused=15.77us raw_speedup=1.172
M=32 separate=18.42us raw_fused=15.87us raw_speedup=1.160
M=64 separate=19.10us raw_fused=16.58us raw_speedup=1.152
```

Status:

- Prototype is implemented behind default-off gates.
- `python3 -m py_compile` passes for both touched modules.
- Built and deployed the fused-arm image:
  `localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-qb-wqb-fused-20260618T173328Z`.
- The first deploy exposed a wiring bug:
  `_FusedQbWqBNvfp4` did not store `apply_wq_post_scale`. That is fixed.
- Corrected image warms up and reaches ready, but production traffic is not
  healthy. C16/OSL128 and single-request probes produced no benchmark rows.
  The prefill executor reported hang detection, and the frontend eventually
  returned `No disaggregated params in prefill response:
  Prefill router output missing disaggregated_params`.
- Artifacts:

```text
.bench_runs_claude/results/129_build_qb_wqb_fused_fix.txt
.bench_runs_claude/results/130_push_qb_wqb_fused_fix.txt
.bench_runs_claude/results/131_qb_wqb_fused_fix_deploy.txt
.bench_runs_claude/results/132_qb_wqb_fused_osl128_c16_24_32.txt
.bench_runs_claude/results/133_qb_wqb_fused_osl128_c16.txt
.bench_runs_claude/results/134_qb_wqb_fused_single_probe_after_hang.txt
.bench_runs_claude/results/135_restore_after_qb_wqb_fused_hang.txt
.bench_runs_claude/results/136_restored_best_single_probe.txt
```

Verdict:

- The fused q_b+wq_b direction is still a plausible dense/proj launch-reduction
  target because the isolated NVFP4 projection microbench saves about 15-18%.
- It is not production-safe yet. Treat the current implementation as a
  default-off experimental branch only.
- The next useful step is not another throughput sweep. Isolate
  `forward_dsa_proj` with precomputed `wq_b` under the same prefill/decode
  CUDA-graph path, add per-layer debug traces around sparse indexer entry/exit,
  and prove the fused tensor split plus sparse-weight scale compensation does
  not wedge request completion before testing throughput again.

Follow-up finding:

- BEST_CONFIG uses `index_topk_pattern: FSSS`, so only the F layers consume the
  full DSA indexer projection path. The first fused implementation built and
  ran q_b+wq_b for every layer before the static `skip_topk` early return in
  `pre_indexer_proj`.
- That was a bad production integration: it resurrected dead `wq_b` GEMMs on
  S layers and duplicated fused q_b+wq_b packed weights/scales on layers that
  never consume `wq_b`.
- Patched `MLA.post_load_weights()` and `MLA.forward_dsa_proj()` so fused
  q_b+wq_b only exists/runs when `indexer.skip_topk == False`.
- `python3 -m py_compile tensorrt_llm/_torch/modules/attention.py` and
  `python3 -m py_compile tensorrt_llm/_torch/attention_backend/sparse/dsa.py`
  pass after the patch.
- This is the next candidate to rebuild/test. It should reduce duplicate
  fused-weight memory by roughly the F-layer fraction and avoid doing
  non-consumed wq_b work on FSSS skip layers.

FSSS-gated rebuild and serving test:

- Built and pushed the gated image:
  `localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-qb-wqb-fused-fsss-20260618T180512Z`.
- Updated `.bench_runs_claude/deploy_arm.py` so `qb_wqb_fused_mega_cf` uses
  that tag.
- The hard single-request failure is fixed. The gated fused arm completed a
  single OSL32 request and C16/OSL128 probes.
- Warm C16/OSL128 is near baseline but not a win:

```text
qb_wqb_fused_fsss_osl128_c16_rerun: C16 agg=511.0 tok/s, TTFT_p50=1374 ms
qb_wqb_fused_fsss_osl128_c16_24_32: C16 agg=504.9 tok/s, TTFT_p50=1211 ms
qb_wqb_fused_fsss_osl128_c16_24_32: C24 agg=597.4 tok/s, TTFT_p50=1705 ms
```

- C32 did not produce a row before manual cancellation. Decode GPUs stayed
  busy and prefill logged repeated `num_fitting_reqs=0 ... may not have enough
  kvCache`, so the gated fused path is still not acceptable at the C32
  operating point.
- Restored BEST_CONFIG after the failed C32 point. Warm restored C32/OSL128
  completed normally:

```text
restored_best_after_fsss_c32_rerun: C32 agg=835.3 tok/s, TTFT_p50=1543 ms
```

- Artifacts:

```text
.bench_runs_claude/results/137_qb_wqb_fused_fsss_tag.txt
.bench_runs_claude/results/138_build_qb_wqb_fused_fsss.txt
.bench_runs_claude/results/139_push_qb_wqb_fused_fsss.txt
.bench_runs_claude/results/140_qb_wqb_fused_fsss_deploy.txt
.bench_runs_claude/results/141_qb_wqb_fused_fsss_single_probe.txt
.bench_runs_claude/results/142_qb_wqb_fused_fsss_osl128_c16.txt
.bench_runs_claude/results/143_qb_wqb_fused_fsss_osl128_c16_rerun.txt
.bench_runs_claude/results/144_qb_wqb_fused_fsss_osl128_c16_24_32.txt
.bench_runs_claude/results/145_restore_after_qb_wqb_fused_fsss_c32_stall.txt
.bench_runs_claude/results/146_restored_best_after_fsss_c32.txt
.bench_runs_claude/results/147_restored_best_after_fsss_c32_rerun.txt
```

Updated verdict:

- The FSSS gating fixed the correctness/liveness problem for small probes, but
  this fusion still loses or stalls at the C32 operating point.
- The remaining likely issue is memory/KV headroom and/or a scheduling
  interaction from materializing fused q_b+wq_b weights even on the active F
  layers. Since BEST_CONFIG is already close to the KV fitting edge, a few
  hundred MiB of duplicate decode weights can still be enough to change
  admission behavior at C32.
- Do not spend more time on the concat-weight version as the production
  megakernel. The principled next implementation is a two-output NVFP4 GEMM
  kernel/op for q_b+wq_b that consumes the original q_b and wq_b packed weights
  without creating a concatenated duplicate, or a narrower C32-safe profiling
  run proving the duplicate weights are not the admission trigger.

## 2026-06-18 - q_b/wq_b shared-packet overlap microbench

Added an auxiliary-stream variant to
`.bench_runs_claude/megakernel/qb_wqb_reuse_compare.py` and ran it on GPU 5
with the BEST_CONFIG image plus `/models` mounted:

```text
.bench_runs_claude/results/148_qb_wqb_shared_parallel_microbench.txt
```

M=16/24/32/64, amax factor 1.0, cublasLt:

```text
M=16 base=32.81us shared=20.09us parallel=19.65us
M=24 base=35.43us shared=19.46us parallel=19.24us
M=32 base=37.84us shared=18.75us parallel=18.89us
M=64 base=43.50us shared=18.86us parallel=19.02us
```

Interpretation:

- Reusing the q_b static FP4 packet is a real local microbench win and remains
  numerically stable for q_b exactly; wq_b differs only by using q_b's static
  activation scale (`wq_cos` around 0.9994-0.9996 for these samples).
- Running q_b and wq_b GEMMs on separate streams does not materially improve
  over the serial shared-packet path. At M=32 and M=64 it is marginally worse.
- This matches the serving result: the no-duplicate shared-packet path is
  memory-safe but not the missing end-to-end production win. The q_b/wq_b pair
  is too small a lever once the whole decode stack is active unless we replace
  the underlying GEMM launches with a genuine fused kernel that preserves memory
  headroom.

## 2026-06-18 - dense combo overlay: sigmoid quant + atomic split-K o_proj

Goal: combine the two dense/proj experiments that were individually
production-reachable:

- exact MLA gate `sigmoid_mul + NVFP4 pack` before `o_proj`;
- atomic split-K route for small-M `o_proj` (`N=7168,K=16384,split=2`).

Implementation notes:

- Patched `tensorrt_llm/_torch/modules/fused_lowrank_gate.py` so the fused
  sigmoid+quant op lazy-loads `tensorrt_llm/libs/libsigmoid_quant_ext.so`.
  Without that, the env flag could be set while the custom op was absent from
  `torch.ops.trtllm`.
- Built dense combo v1:
  `optrt-529374445d-codex-dense-combo-20260618T183958Z`.
  It crash-looped because the overlay copied the current
  `cpp_custom_ops.py`, which referenced optional
  `trtllm::dsv3_gate_gemm_op` symbols not present in the frozen BEST_CONFIG
  C++ library.
- Built dense combo v2:
  `optrt-529374445d-codex-dense-combo-20260618T184509Z`.
  It uses the known-good sigmoid-overlay `cpp_custom_ops.py`, current
  `fused_lowrank_gate.py`, and the split-K-capable dense GEMM files.
- Deployment arm: `dense_combo_mega_cf`, same BEST_CONFIG transport and
  `MEGAKERNEL=1`, `combine=false`; q_b/wq_b experiments explicitly disabled.

Artifacts:

```text
.bench_runs_claude/results/149_dense_combo_tag.txt
.bench_runs_claude/results/150_build_dense_combo.txt
.bench_runs_claude/results/151_push_dense_combo.txt
.bench_runs_claude/results/152_dense_combo_deploy.txt
.bench_runs_claude/results/153_dense_combo_v2_tag.txt
.bench_runs_claude/results/154_build_dense_combo_v2.txt
.bench_runs_claude/results/155_push_dense_combo_v2.txt
.bench_runs_claude/results/156_dense_combo_v2_deploy.txt
.bench_runs_claude/results/157_dense_combo_v2_handshake.txt
.bench_runs_claude/results/158_dense_combo_v2_c32_o128.txt
.bench_runs_claude/results/159_dense_combo_v2_c32_o128_isl2055.txt
.bench_runs_claude/results/160_dense_combo_v2_c16_24_32_o128_isl2055.txt
.bench_runs_claude/results/161_dense_combo_v2_c32_o128_isl2055_r2.txt
```

False alarm / measurement correction:

- First C32 run used `--isl-text-reps 1024`, which tokenized to `ISL=26625`.
  That is not the BEST_CONFIG operating-point shape. It hit prefill KV
  pressure (`num_fitting_reqs=0 ... may not have enough kvCache`) and produced
  `108.3 agg tok/s`, which is invalid for the C32/OSL128 comparison.
- Correct shape is `--isl-text-reps 79`, `ISL=2055`.

Corrected serving results, C32, ISL 2055, OSL 128:

```text
restored BEST_CONFIG reference:
  restored_best_after_fsss_c32_rerun: agg=835.3 tok/s, TTFT_p50=1543 ms

dense_combo_v2_c32_o128_isl2055:
  agg=848.6 tok/s, TTFT_p50=1330 ms, user_p50=39.16 tok/s

dense_combo_v2_c16_24_32_o128_isl2055:
  C16 agg=498.3 tok/s, TTFT_p50=1396 ms, user_p50=41.80 tok/s
  C24 agg=666.4 tok/s, TTFT_p50=1228 ms, user_p50=39.94 tok/s
  C32 agg=792.7 tok/s, TTFT_p50=1722 ms, user_p50=38.14 tok/s

dense_combo_v2_c32_o128_isl2055_r2:
  agg=838.4 tok/s, TTFT_p50=1381 ms, user_p50=38.76 tok/s
```

Verdict:

- The combo is runnable and the corrected C32 rows are in the control band.
- It is not a proven production win. The best row (`848.6`) is only about
  +1.6% over the latest restored control, while the next C32 rows were
  `792.7` and `838.4`.
- This agrees with the profile evidence from the split-K atomic run: isolated
  `o_proj` improvements are real but too narrow to move the full dense/proj
  bucket decisively.
- Next useful work should target a larger dense/proj family, especially the
  `nvjet_sm100_ootst_128x128_256x7_4x1_2cta_v_bx...` family or the BF16 MLA
  output gate family, and should require a 128-step C32 profile showing a
  multi-ms/iter dense/proj bucket reduction before more serving sweeps.

## 2026-06-18 - grouped NVFP4 op applicability check

Question: can the existing CuteDSL grouped/multi-B NVFP4 MoE kernels be reused
as the q_b/wq_b dense-projection fusion route?

Relevant entry points:

```text
tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py
  trtllm::cute_dsl_nvfp4_grouped_gemm_blackwell
  trtllm::cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell_multi_b

tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py
  cute_dsl_nvfp4_grouped_gemm_ref
  run_moe_nvfp4_impl_dwdp
```

Findings:

- `cute_dsl_nvfp4_grouped_gemm_blackwell` is an MoE grouped GEMM contract:
  `A[m,k]`, `B[l,n,k]`, per-tile `group/expert` ids, and a single output
  shape `C[m,n]`.
- The `multi_b` gather path accepts a list of B tensors, but the list only
  partitions the grouped/expert `L` dimension. Shape inference still takes
  `n = weight[0].size(1)` and returns one packed output plus one scale tensor.
- The path is also tied to gather/permutation metadata and activation/finalize
  behavior for MoE, not to independent dense MLA projections.
- q_b/wq_b need "same A, two independent B matrices, two independent outputs"
  with different N (`q_b: 24576`, `wq_b: 8192`) and no expert-row grouping.

Conclusion:

- Existing grouped/multi-B NVFP4 ops are useful implementation references but
  are not a direct q_b/wq_b solution.
- A real q_b/wq_b production attempt needs a no-duplicate two-output dense
  projection route that consumes the original packed q_b and wq_b weights and
  writes the two outputs separately, or a broader shape-family replacement for
  the large `nvjet_sm100_ootst...` NVFP4 projection bucket.

## 2026-06-18 - q_b/wq_b fused concat with aliased original storage

Hypothesis: the previous q_b/wq_b fused concat path was functionally correct at
low concurrency but hurt C32 admission/headroom because it kept the original
q_b/wq_b weights and also materialized a fused concatenated copy. The next
narrow test was to keep the same single-GEMM concat path while making the
original module parameters alias views into the fused tensor storage.

Implementation:

- Added `TRTLLM_INDEXER_FUSE_QB_WQB_ALIAS_ORIGINALS=1` to
  `_FusedQbWqBNvfp4`. When enabled, build creates the fused concatenated
  weight/scale tensors, then points `q_b.weight`, `wq_b.weight`,
  `q_b.weight_scale`, and `wq_b.weight_scale` at narrow views into that fused
  storage.
- Updated the `qb_wqb_fused_mega_cf` deploy arm with the alias flag.
- Built and pushed:
  `localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-qb-wqb-fused-alias-20260618T190718Z`.

Artifacts:

```text
.bench_runs_claude/results/162_qb_wqb_fused_alias_tag.txt
.bench_runs_claude/results/163_build_qb_wqb_fused_alias.txt
.bench_runs_claude/results/164_push_qb_wqb_fused_alias.txt
.bench_runs_claude/results/165_qb_wqb_fused_alias_deploy.txt
.bench_runs_claude/results/166_qb_wqb_fused_alias_decode_tail_unready.txt
.bench_runs_claude/results/167_qb_wqb_fused_alias_prefill_tail.txt
.bench_runs_claude/results/168_qb_wqb_fused_alias_handshake.txt
.bench_runs_claude/results/169_qb_wqb_fused_alias_c32_o128_isl2055.txt
.bench_runs_claude/results/170_qb_wqb_fused_alias_c32_o128_isl2055_r2.txt
.bench_runs_claude/results/171_qb_wqb_fused_alias_c16_24_32_o128_isl2055.txt
.bench_runs_claude/results/172_restore_best_after_qb_wqb_alias.txt
.bench_runs_claude/results/173_restore_best_decode_tail_waiting.txt
```

Readiness:

- Prefill reached ready first.
- Decode eventually reached ready at about 7m40s, with zero restarts.
- The decode tail while unready showed startup-probe 503s and CuTe DSL
  warnings, but no OOM or Python exception in the captured window.
- Production-route handshake succeeded:
  `C1, ISL=2055, OSL=32, TTFT_p50=21523 ms, user_p50=24.83 tok/s`.

Serving results, ISL 2055, OSL 128:

```text
first C32 pass:
  agg=625.1 tok/s, TTFT_p50=4048 ms, user_p50=38.27 tok/s

warmed C32 pass:
  agg=836.2 tok/s, TTFT_p50=1570 ms, user_p50=38.91 tok/s

C16/C24/C32 sweep:
  C16 agg=530.1 tok/s, TTFT_p50=1202 ms, user_p50=42.22 tok/s
  C24 agg=647.9 tok/s, TTFT_p50=1599 ms, user_p50=40.09 tok/s
  C32 agg=846.0 tok/s, TTFT_p50=1450 ms, user_p50=38.70 tok/s
```

Reference rows:

```text
restored BEST_CONFIG C16/C24/C32:
  C16 agg=521.2 tok/s, TTFT_p50=1200 ms, user_p50=42.54 tok/s
  C24 agg=661.9 tok/s, TTFT_p50=1467 ms, user_p50=39.79 tok/s
  C32 agg=818.2 tok/s, TTFT_p50=1860 ms, user_p50=38.45 tok/s

restored BEST_CONFIG later C32:
  C32 agg=835.3 tok/s, TTFT_p50=1543 ms, user_p50=38.29 tok/s
```

Verdict:

- Aliasing the original q_b/wq_b storage resolves the previous C32 stall and
  makes the fused concat path production-runnable.
- It is still not a real throughput win. Warmed C32 is in the restored
  BEST_CONFIG control band (`836-846` vs `818-835` depending on run), and C24
  is slightly worse than the control row.
- This confirms the q_b/wq_b pair is not large enough to move aggregate
  throughput by itself through a Python-level concat/split wrapper, even when
  the duplicate-weight headroom issue is removed.
- The next useful kernel work should not be another q_b/wq_b wrapper variant.
  It should replace a larger repeated projection family, most likely the
  `nvjet_sm100_ootst_128x128_256x7_4x1_2cta_v_bx...` NVFP4 family, or attack
  the BF16 MLA output gate family that accounts for a comparable per-iter
  share.
- After collecting the alias data, restored `vbfuse_nvfp4_mega_cf` /
  BEST_CONFIG. Decode and prefill both reached ready on
  `optrt-529374445d-vbfuse-nvfp4-20260615T043037Z` with zero restarts.

## 2026-06-18 - split-K atomic redeploy check after q_b/wq_b alias

Purpose:

- Recenter on the only dense/proj kernel arm that has shown a real production
  signal: atomic split-K for `o_proj` (`N=7168,K=16384,split=2`).
- Verify it still deploys cleanly after the q_b/wq_b alias experiment and
  compare the warmed C32/ISL2055/OSL128 row against restored BEST_CONFIG.

Deployment:

```text
arm: splitk_o_proj_atomic_mega_cf
image: localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-splitk-o-proj-atomic-20260618T111927Z
env: MEGAKERNEL=1, FORCE_COMM_METHOD=NVLINK_TWO_SIDED, use_low_precision_moe_combine=false
env: TRTLLM_NVFP4_GEMM_SPLITK_O_PROJ=1, TRTLLM_NVFP4_GEMM_SPLITK_ATOMIC=1
```

Artifacts:

```text
.bench_runs_claude/results/174_splitk_atomic_redeploy.txt
.bench_runs_claude/results/175_splitk_atomic_redeploy_c32_o128_isl2055.txt
.bench_runs_claude/results/176_splitk_atomic_redeploy_c32_o128_isl2055_warm.txt
.bench_runs_claude/results/177_splitk_atomic_route_logs.txt
```

Readiness:

- Prefill reached ready first.
- Decode reached ready at about 7m34s with zero restarts.

Serving results, C32, ISL 2055, OSL 128:

```text
split-K atomic cold first pass:
  TTFT_p50=21420 ms, TTFT_p95=26813 ms, user_p50=36.58 tok/s, agg=240.2 tok/s

split-K atomic warmed pass:
  TTFT_p50=1310 ms, TTFT_p95=2952 ms, user_p50=38.43 tok/s, agg=858.2 tok/s
```

Reference rows:

```text
restored BEST_CONFIG C32/ISL2055/OSL128:
  agg=818.2 tok/s in the C16/C24/C32 control sweep
  agg=835.3 tok/s in the later standalone rerun

previous split-K atomic warm repeats:
  agg=833.6, 836.2, 843.6 tok/s
```

Route proof:

```text
NVFP4 split-K o_proj selected for M=64, N=7168, K=16384, split=2, atomic=True, output_buffer_kind=0
NVFP4 split-K o_proj selected for M=7, N=7168, K=16384, split=2, atomic=True, output_buffer_kind=0
NVFP4 split-K o_proj selected for M=6, N=7168, K=16384, split=2, atomic=True, output_buffer_kind=0
NVFP4 split-K o_proj selected for M=5, N=7168, K=16384, split=2, atomic=True, output_buffer_kind=0
NVFP4 split-K o_proj selected for M=4, N=7168, K=16384, split=2, atomic=True, output_buffer_kind=0
NVFP4 split-K o_proj selected for M=3, N=7168, K=16384, split=2, atomic=True, output_buffer_kind=0
```

Interpretation:

- The redeploy is clean and the warmed row is the strongest C32/OSL128 serving
  row measured for this arm so far.
- This does not change the profile conclusion: the existing 128-step Kineto
  trace for this exact image/env shows dense/proj only moving from about
  `9102 us/iter` to `9005 us/iter`; the two dominant NVJIT families remain
  unchanged.
- Keep split-K atomic as an env-gated production-positive special case, but do
  not treat it as the broader dense/proj solution. The next kernel must target
  the 122-call BF16 family or the 193-call NVFP4 family directly.

## 2026-06-18 - padded kv_a + DSA wk/wp NVFP4 concat prototype

Purpose:

- Move beyond one-shape `o_proj` split-K and target a repeated 193-call NVFP4
  projection pattern with a real graph-shape reduction.
- Candidate pair: `self_attn.kv_a_proj_with_mqa` (`N=2112,K=7168`) and DSA
  fused `wk/weights_proj` (`N=192,K=7168`). Both consume the same gated hidden
  state on non-skip-topK DSA layers.

Key layout constraint:

- `kv_a` has `N=2112`, which is not a 128-row swizzled-scale boundary.
- Direct flat concat would misalign the DSA block scales.
- The valid concat inserts 64 dummy FP4 rows after `kv_a`, so the fused GEMM is
  `N=2368`: `kv_a[0:2112]`, pad `[2112:2176]`, DSA `[2176:2368]`.
- The scale tensor is `kv_a.weight_scale + fused_wk_wp.weight_scale`; this
  includes the trailing DSA pad-scale rows, matching `pad_up(2368, 128)`.

Prototype artifact:

```text
.bench_runs_claude/megakernel/kva_wkwp_concat_projection.py
```

Microbench, B200, forced cuBLASLt:

```text
M=1   separate=21.18 us  fused=16.59 us  speedup=1.276
M=2   separate=21.86 us  fused=17.67 us  speedup=1.238
M=4   separate=22.26 us  fused=17.60 us  speedup=1.265
M=8   separate=21.14 us  fused=17.88 us  speedup=1.182
M=16  separate=21.24 us  fused=17.65 us  speedup=1.204
M=32  separate=21.88 us  fused=18.60 us  speedup=1.176
M=64  separate=21.79 us  fused=18.55 us  speedup=1.175
```

Microbench, default allowed backends (`cutlass,cublaslt,cuda_core`):

```text
M=1   separate=20.41 us  fused=17.24 us  speedup=1.184
M=4   separate=22.73 us  fused=17.94 us  speedup=1.267
M=16  separate=21.47 us  fused=17.32 us  speedup=1.240
M=32  separate=21.83 us  fused=18.42 us  speedup=1.186
M=64  separate=22.18 us  fused=18.85 us  speedup=1.177
```

Correctness signal:

```text
kv_a slice: cos ~= 1.0, max_abs = 0 in the random-weight harness
DSA slice:  cos >= 0.999995 with scalar weight_scale_2 correction
```

Implemented production hook:

- `tensorrt_llm/_torch/attention_backend/sparse/dsa.py`
  - added `_FusedKvAWkWpNvfp4`
  - verifies NVFP4 weight/scale invariants
  - inserts 64 pad rows when needed
  - optional storage alias:
    `TRTLLM_INDEXER_FUSE_KVA_WKWP_ALIAS_ORIGINALS=1`
- `tensorrt_llm/_torch/modules/attention.py`
  - builds the helper in `MLA.post_load_weights()` behind
    `TRTLLM_INDEXER_FUSE_KVA_WKWP=1`
  - only fires on non-skip-topK DSA layers
  - passes precomputed `indexer_k` and `weights` into
    `Indexer.pre_indexer_proj()`
- `Indexer.pre_indexer_proj()`
  - accepts optional precomputed `indexer_k` / `weights`
  - leaves k_norm, RoPE/cat quant, and sparse-indexer weighting unchanged

Required experimental env for first production A/B:

```text
TRTLLM_INDEXER_FUSE_KVA_WKWP=1
TRTLLM_INDEXER_FUSE_KVA_WKWP_STATIC=1
TRTLLM_INDEXER_FUSE_KVA_WKWP_ALIAS_ORIGINALS=1
TRTLLM_INDEXER_FUSE_KVA_WKWP_DEBUG=1
```

Notes:

- `TRTLLM_INDEXER_FUSE_KVA_WKWP_STATIC=1` is required for this checkpoint
  because `kv_a` has a static activation scale while the indexer `wk/wp` path is
  dynamic. This is the same numeric assumption tested in the pre-KV static reuse
  work, now with a graph-shape reduction instead of only quantization reuse.
- Syntax validation passed with:
  `PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile ...`.
- A mounted-source import smoke hit local C++ loader-path issues
  (`libth_common.so` dependency chain) in the old microbench container; validate
  the full helper by building/deploying the normal production image next.

Production A/B, first deploy:

```text
image=localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-kva-wkwp-fused-20260618T195615Z
env=BEST_CONFIG + TRTLLM_INDEXER_FUSE_KVA_WKWP=1
```

Serving result, C32/ISL2055/OSL128:

```text
cold-ish run: agg=734.0 tok/s, TTFT_p50=1848 ms, user_p50=36.54 tok/s
warmed rerun: agg=810.0 tok/s, TTFT_p50=1549 ms, user_p50=38.18 tok/s
```

Profiler result:

```text
trace=/var/lib/optrt-cache/nsys/kva_wkwp_c32_128_20260618T2020-rank-*.json
analysis=.bench_runs_claude/profiles/kineto_analysis_kva_wkwp_c32_128_20260618T2020/kva_wkwp_c32_128_20260618T2020_exhaustive.txt

Dense/proj GEMM: 9097 us/iter, 37.2%, 640 launches/iter
Top dense kernels:
  nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT: 2856.7 us/iter, 122 calls/iter
  nvjet_sm100_ootst_128x128_256x7_4x1_2cta...: 2820.5 us/iter, 193 calls/iter
  cutlass3x_sm100_bstensorop_s256x128x64gemm...: 1324.1 us/iter, 64 calls/iter
```

Interpretation:

- The production kernel mix is effectively unchanged from the BEST_CONFIG
  control (`9102 us/iter`, `640 launches/iter` for dense/proj), so the new
  helper did not reduce the real decode-step projection count.
- Warmup logs still show the old `kv_a` shape:
  `input_shapes=((1, 3584), (2112, 3584), ...)`, which should not appear if the
  padded `N=2368` fused helper is selected for that path.
- The next action is not another throughput repeat. Add construction-time
  diagnostics for every `_FusedKvAWkWpNvfp4.build()` guard and redeploy a debug
  carrier image. The likely failure is one of the silent build guards, not the
  GEMM math itself.

Diagnostic redeploy result:

```text
image=localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-kva-wkwp-fused-diag-20260618T202232Z
log signal, all decode ranks:
  MLA kv_a/indexer wk/wp fusion disabled:
  dtype mismatch kv_a=torch.bfloat16 wkwp=torch.float32
```

Conclusion:

- The fused KVA/WK/WP helper is not firing in production. It is rejected during
  `MLA.post_load_weights()` because the MLA `kv_a` projection emits BF16, while
  DSA `wk` and `weights_proj` are declared as FP32 linears and the existing
  fused DSA helper reports FP32 output.
- This fully explains the profile result: the production trace still shows the
  old `N=2112` `kv_a` autotune shape and the dense/proj bucket remains at
  `~9097 us/iter`, `640 launches/iter`.
- A simple concatenated NVFP4 GEMM has one output dtype, so this is a real
  mixed-output problem, not just a missed env flag.
- Possible continuations:
  - test a BF16-output variant and cast only the DSA slice back to FP32 before
    indexer weighting, but this weakens an explicitly FP32 DSA path and needs
    accuracy/stability proof before a throughput run;
  - implement a true mixed-output fused kernel, BF16 for `kv_a` and FP32 for
    DSA slices, if we still believe this path is worth the scope;
  - redirect effort to the larger dense/proj families that dominate the trace:
    the 122-call BF16 MLA gate family and the 193-call NVFP4 family.

Recommendation:

- Do not spend more production runs on the current simple KVA/WK/WP concat.
  It has not actually activated, and even if fixed, it touches only the
  non-skip-topK DSA side path. The next principled target should be either a
  mixed-output correctness probe for this path or, more likely, the larger
  122/193-call dense/proj families.

## 2026-06-18 - BF16 gate DSV3 tile-N widening probe

Target:

- Revisit the 122-call BF16 MLA output-gate family at the kernel level instead
  of another wrapper-level full-gate/l-batch toggle.
- Existing `trtllm::dsv3_gate_gemm_op` only dispatches `tile_n=8` for
  `M<=8`, otherwise `tile_n=16`. The device kernel also asserted
  `n_iter_cnt <= 2`, so `tile_n=32` was never measured.

Code changes:

- `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu`
  - widened `MmaComputer` to allow up to 4 N-iterations;
  - added explicit `tile_n=32` instantiations for the DSV3 gate shapes.
- `cpp/tensorrt_llm/thop/dsv3FusedAGemmOp.cpp`
  - added env-gated tile-N dispatch:
    `TRTLLM_OPTRT_MLA_GATE_DSV3_TILE_N={8,16,32}`;
  - default behavior is unchanged (`M<=8 -> 8`, else `16`).
- `.bench_runs_claude/megakernel/gate_proj_compare.py`
  - extended the DSV3 sweep to cover `tile_n={8,16,32}`;
  - added `--skip-cute` for focused C++ gate-kernel timing.

Validation:

```text
docker exec -u root optrt-bench-claude \
  bash -lc 'cd /repo && cmake --build cpp/build --target th_common -j 16'

result: Built target th_common
```

Direct-load microbench against the rebuilt `libth_common.so`, B200, CUDA graph
replay, M in `{16,32,64}`:

```text
M=16 base two-half torch.mm: 48.36 us
  full torch.mm:        46.39 us, ratio=0.959
  best DSV3 tile-N:     53.78 us, ratio=1.112, tile_m=64,tile_n=16
  DSV3 tile_n=32 best:  56.01 us, ratio=1.158, tile_m=64

M=32 base two-half torch.mm: 49.62 us
  full torch.mm:        48.49 us, ratio=0.977
  best DSV3 tile-N:     57.79 us, ratio=1.165, tile_m=64,tile_n=32

M=64 base two-half torch.mm: 49.66 us
  full torch.mm:        46.09 us, ratio=0.928
  best DSV3 tile-N:     95.69 us, ratio=1.927, tile_m=64,tile_n=32
```

Correctness:

```text
cosine = 1.0 or 0.99999994
max_abs = 0.0156
```

Interpretation:

- The wider `tile_n=32` variant is real and improves the custom DSV3 gate
  kernel at M32/M64 versus its `tile_n=16` form, but it remains materially
  slower than the current two-half `torch.mm` gate baseline.
- BF16 gate is still not the next production path unless we write a genuinely
  different kernel. More tuning of this DSV3 kernel family is unlikely to move
  the 37-38% dense/proj bucket enough.
- The next high-leverage branch should stay on the 193-call NVFP4 family or a
  fused BF16-input-to-NVFP4 projection boundary that removes quantization and
  projection overhead together.

## 2026-06-18 - NVFP4 quant/projection boundary probe

Added:

```text
.bench_runs_claude/megakernel/nvfp4_quant_projection_boundary.py
```

Purpose:

- Measure the ceiling for a fused BF16-input-to-NVFP4 projection kernel before
  spending more time in CuTe/C++.
- Timed under CUDA graph replay on B200 with a conservative replay+sync timing
  mode.
- Shapes are the current dense/proj suspects:
  `q_b_proj`, `dsa_wq_b`, and `o_proj`, M in `{16,32,64}`.

Static activation-scale path, forced cuBLASLt:

```text
q_b_proj N=24576,K=1536:
  M=16 gemm_only=8.259us quant+gemm=10.274us boundary=2.015us ceiling=1.244x
  M=32 gemm_only=8.266us quant+gemm=10.092us boundary=1.826us ceiling=1.221x
  M=64 gemm_only=8.261us quant+gemm=10.314us boundary=2.052us ceiling=1.248x

dsa_wq_b N=8192,K=1536:
  M=16 gemm_only=6.202us quant+gemm=8.241us boundary=2.039us ceiling=1.329x
  M=32 gemm_only=6.210us quant+gemm=8.247us boundary=2.037us ceiling=1.328x
  M=64 gemm_only=6.212us quant+gemm=8.252us boundary=2.040us ceiling=1.328x

o_proj N=7168,K=16384:
  M=16 gemm_only=16.461us quant+gemm=20.543us boundary=4.082us ceiling=1.248x
  M=32 gemm_only=16.458us quant+gemm=20.544us boundary=4.086us ceiling=1.248x
  M=64 gemm_only=16.469us quant+gemm=20.536us boundary=4.066us ceiling=1.247x
```

Dynamic activation-scale path, forced cuBLASLt, same sync timer:

```text
q_b_proj:
  dynamic quant+gemm = 22.6us / 26.7us / 32.9us for M=16/32/64
  dynamic ideal ceiling = 2.74x / 3.23x / 3.95x

dsa_wq_b:
  dynamic quant+gemm = 21.3us / 22.6us / 30.1us for M=16/32/64
  dynamic ideal ceiling = 3.43x / 3.64x / 4.84x

o_proj:
  dynamic quant+gemm = 39.0us / 39.0us / 38.5us for M=16/32/64
  dynamic ideal ceiling = 2.37x / 2.37x / 2.34x
```

Notes:

- Static-scale projection fusion has real but bounded upside: usually
  `~1.22x-1.33x` if quantization became free. This alone will not explain the
  whole missing aggregate throughput.
- Dynamic activation quantization is much more expensive. That is consistent
  with the earlier DSA pre-KV and q_b/wq_b shared-packet microbenches, but
  production A/B already showed those reuse wrappers do not move aggregate
  throughput enough when dense/proj GEMM launch count stays at 640/iter.
- A production-allowed backend sweep (`cutlass,cublaslt,cuda_core`) matched the
  cuBLASLt rows for q_b/o_proj, but produced suspicious DSA static rows where
  quant+gemm measured equal to gemm-only. Treat the forced-cuBLASLt static-only
  run as the clean boundary measurement until the captured graph is profiled at
  kernel-count level.

Decision:

- Do not make the next branch another static packet reuse or concat wrapper.
- The next implementation should target the actual launch/tile substrate: a
  no-duplicate multi-problem NVFP4 projection kernel that can process same-A
  independent B matrices in one persistent grid and write separate outputs.
- Per the local CuTe guidance, start from the existing SM100 dense persistent
  NVFP4 kernel path and keep the first substrate plain GEMM-only. Avoid adding
  fused quantization/epilogue extras until the multi-problem launch substrate
  wins on CUDA-graph microbench.

Artifact rerun, 2026-06-19:

```text
.bench_runs_claude/results/324_nvfp4_quant_projection_boundary_static_event.txt
.bench_runs_claude/results/325_nvfp4_quant_projection_boundary_static_sync.txt
```

Sync-timed static rerun:

```text
q_b_proj:
  M16 gemm_only=8.258us quant+gemm=8.277us ceiling=1.002x
  M32 gemm_only=8.268us quant+gemm=10.313us ceiling=1.247x
  M64 gemm_only=8.259us quant+gemm=9.258us ceiling=1.121x

dsa_wq_b:
  M16 gemm_only=6.203us quant+gemm=8.247us ceiling=1.329x
  M32 gemm_only=6.213us quant+gemm=8.258us ceiling=1.329x
  M64 gemm_only=6.203us quant+gemm=7.220us ceiling=1.164x

o_proj:
  M16 gemm_only=16.452us quant+gemm=20.525us ceiling=1.248x
  M32 gemm_only=16.455us quant+gemm=20.546us ceiling=1.249x
  M64 gemm_only=16.456us quant+gemm=20.549us ceiling=1.249x
```

Interpretation update:

- The event timer is not trustworthy for every quant+GEMM graph shape; keep
  the sync artifact as the reference.
- `o_proj` remains the cleanest static quant/projection boundary: stable
  ~4.09 us over GEMM-only for M16/M32/M64.
- q_b/dsa short-K rows are shape-sensitive under graph replay. They are still
  valid as a family target, but not as a standalone proof that a fused static
  quant boundary will move serving.

## 2026-06-18 - q_b/wq_b variable-N L-batch kernel substrate

Implementation:

- Extended `Sm100BlockScaledPersistentDenseGemmKernel` with a default-off
  `variable_n_l1` parameter.
- The new path keeps the existing L-batch persistent grid, but for L=1 skips
  N tiles whose start is beyond the smaller problem's N. The skip is applied
  consistently in the TMA producer, MMA producer, and epilogue consumer loops.
- Added benchmark-only driver and harness:

```text
.bench_runs_claude/megakernel/dense_kernel_driver.py
.bench_runs_claude/megakernel/qb_wqb_variable_n_lbatch.py
```

Target shape:

```text
L=0: q_b_proj  M x 1536 -> 24576
L=1: wq_b      M x 1536 ->  8192
```

This is the first no-concat q_b/wq_b substrate:

- no fused concatenated weight allocation;
- no padded wq_b compute up to q_b's N;
- one persistent grid instead of two GEMM launches;
- current prototype still uses padded output storage for L=1 because the
  existing single C descriptor has row stride `Nmax`; production needs either
  a second C descriptor or an accepted padded-output scratch contract.

First smoke:

```text
M=16, tactic=((128,128),(1,1),False)
separate=17.96us concat=18.10us variable=16.90us
variable_speedup=1.063x vs separate, 1.071x vs concat
q_cos=0.99999630, wq_cos=0.99999529
```

Small tactic sweep, B200, CUDA graph replay, forced cuBLASLt for baselines:

```text
M=16:
  separate=17.28us concat=18.25us variable=14.88us
  variable_speedup=1.161x, variable_vs_concat=1.226x
  tactic=((128,256),(1,2),False)
  q_cos=0.99999630, wq_cos=0.99999529

M=32:
  separate=18.01us concat=18.71us variable=15.78us
  variable_speedup=1.141x, variable_vs_concat=1.186x
  tactic=((128,256),(1,2),False)
  q_cos=0.99999595, wq_cos=0.99999487

M=64:
  separate=16.84us concat=18.91us variable=15.68us
  variable_speedup=1.074x, variable_vs_concat=1.206x
  tactic=((128,256),(1,2),False)
  q_cos=0.99999595, wq_cos=0.99999458
```

Debug notes:

- CuTe DSL rejects early `continue`, so invalid N tiles are handled with a
  dynamic `tile_n_in_bounds` guard around each loop body.
- A 192-wide N tile tactic caused an illegal access because `8192 % 192 != 0`;
  the harness now skips tactics whose tile N does not divide the smaller
  problem's N unless we add B1 padding to the next tile boundary.
- The first wrong-output attempt exposed that the single C descriptor also
  imposes row stride `Nmax` for L=1. Padding only the output storage fixed
  correctness; production should remove that limitation with a second C pointer
  and TMA descriptor if the integration path needs exact output storage.

Decision:

- This is a real kernel-substrate win, unlike the previous q_b/wq_b concat and
  shared-packet wrapper paths.
- It is still narrower than the whole 193-call NVFP4 family, but it proves the
  "same A, independent B, separate outputs, one persistent grid" direction can
  beat both separate and concat baselines.
- Next work:
  1. turn the benchmark driver into a package custom op/runner behind an env
     flag;
  2. wire `_FusedQbWqBNvfp4` to call it without materializing fused weights;
  3. profile C32/OSL128 for launch-count and dense/proj bucket movement before
     doing another serving throughput sweep.

## 2026-06-18 - q_b/wq_b variable-N package op

Implementation:

- Added `trtllm::cute_dsl_nvfp4_qb_wqb_gemm_blackwell_out`, an in-place
  package custom op for the same-A variable-N q_b/wq_b substrate.
- The production path uses a padded output buffer shaped `[2, M, qb_out]` and
  slices the q_b and wq_b views outside the custom-op boundary. This avoids the
  PyTorch custom-op aliasing rejection from returning two views into one buffer.
- `_FusedQbWqBNvfp4` now enables this path behind
  `TRTLLM_INDEXER_FUSE_QB_WQB_VARIABLE_N=1`. The variable-N mode also aliases
  original q_b/wq_b weight storage into the fused storage to avoid keeping
  duplicate original and fused weights alive.
- An exact concat-output layout `[M, qb_out+wq_out]` was tested and rejected:
  correctness was fine, but the custom-stride C TMA store was slower than the
  standard padded row-major C layout.

Important measurement correction:

- The first package-op timings accidentally multiplied wq_b by the correction
  scale inside the timed CUDA graph. That was not apples-to-apples with
  `variable_us` and it is not the default production contract: DSA returns the
  wq scale separately unless `TRTLLM_INDEXER_FUSE_QB_WQB_POST_SCALE=1`.
- After moving that multiply out of the timed package path, the package op
  matches the direct same-A substrate.

Package-op sweep, B200, CUDA graph replay, forced cuBLASLt baselines:

```text
M=16:
  separate=17.97us concat=18.82us package=15.82us
  package_speedup=1.136x, package_vs_concat=1.190x
  q_cos=0.99999630, wq_cos=0.99999529

M=32:
  separate=18.58us concat=19.27us package=15.69us
  package_speedup=1.184x, package_vs_concat=1.228x
  q_cos=0.99999595, wq_cos=0.99999487

M=64:
  separate=18.22us concat=19.51us package=16.73us
  package_speedup=1.089x, package_vs_concat=1.166x
  q_cos=0.99999595, wq_cos=0.99999458
```

Notes:

- The micro win is now visible through the production `torch.ops` boundary.
- This still only removes one q_b + indexer wq_b pair. The next proof point is
  profile-level movement: C32/OSL128 should show one fewer NVFP4 projection
  launch at the fused DSA site and some reduction in the dense/proj bucket.
- Do not benchmark with `TRTLLM_INDEXER_FUSE_QB_WQB_POST_SCALE=1` unless the
  goal is specifically to include the extra correction multiply.

## 2026-06-18 - q_b/wq_b variable-N production A/B

Production arm:

```text
image=localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-qb-wqb-variable-n-20260618T215848Z
TRTLLM_FORCE_COMM_METHOD=NVLINK_TWO_SIDED
TRTLLM_OPTRT_MOE_MEGAKERNEL=1
TRTLLM_INDEXER_FUSE_QB_WQB=1
TRTLLM_INDEXER_FUSE_QB_WQB_STATIC=1
TRTLLM_INDEXER_FUSE_QB_WQB_VARIABLE_N=1
TRTLLM_INDEXER_FUSE_QB_WQB_POST_SCALE=0
TRTLLM_INDEXER_FUSE_QB_WQB_ALIAS_ORIGINALS=1
use_low_precision_moe_combine=false
```

Handshake:

```text
qb_wqb_variable_n-handshake, C=1, OSL=32, ISL=4161:
ok=8/8, TTFT_p50=754ms, TTFT_p95=22123ms, tok/s/user=24.82
```

The high p95 was first-request/cold-start noise; the disagg/NIXL path passed.

Control-matched serving sweep:

```text
file: .bench_runs_claude/results/20260618T220318Z_qb_wqb_variable_n.txt

  C    TTFT_p50   tok/s/user   agg_out_tok/s
  1       488        43.83          45.8
  4       769        57.41         226.9
  8      1015        55.13         422.1
 16      1144        53.08         780.0
 32      1273        50.61        1417.4
 64      1412        45.38        2322.2
```

C32 repeat, same live deployment:

```text
qb_wqb_variable_n_c32_r2:
  C=32, ok=64/64, TTFT_p50=1361ms, tok/s/user=50.51, agg_out_tok/s=1399.5
```

Comparison against BEST_CONFIG:

```text
vs BEST_CONFIG/sweep_confirmed.txt:
  C32 tok/s/user: 50.61 vs 51.07 (-0.9%)
  C32 agg_out:    1417.4 vs 1424.9 (-0.5%)

vs results/26_best_restored.txt:
  C32 tok/s/user: 50.61 vs 50.08 (+1.1%)
  C32 agg_out:    1417.4 vs 1386.6 (+2.2%)
```

Interpretation:

- Production throughput is parity/noise, not a reliable win.
- The microbench win is real, but the affected q_b/wq_b pair is too small a
  fraction of the serving step to move C32 throughput through run-to-run noise.

Profile result:

```text
label: qb_wqb_variable_n_c32_128_20260618T2220Z
traces: /var/lib/optrt-cache/nsys/qb_wqb_variable_n_c32_128_20260618T2220Z-rank-{0..3}.json
analysis: .bench_runs_claude/profiles/kineto_analysis_qb_wqb_variable_n_c32_128_20260618T2220Z/qb_wqb_variable_n_c32_128_20260618T2220Z_exhaustive.txt
```

Aggregate C32/128-window trace comparison:

```text
                        BEST c32_128_actual       q_b/wq_b variable-N
Dense/proj GEMM         9102 us/iter, 640 launch   9084 us/iter, 624 launch
MoE a2a/comm            3488 us/iter, 290 launch   3482 us/iter, 290 launch
MoE expert GEMM         3183 us/iter, 174 launch   3200 us/iter, 174 launch
Mean GPU idle           15.2%                      15.1%
```

Decision:

- This did remove the expected dense/proj launches: `640 -> 624`, so the fused
  path is active in the production decode graph.
- It did not move dense/proj time materially: `9102 -> 9084 us/iter`.
- Do not keep spending effort on this q_b/wq_b pair as the primary path to a
  throughput win.
- Next target stays the large dense/proj kernel family itself:
  `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT`,
  `nvjet_sm100_ootst_128x128_256x7_4x1_2cta_v_bx_*`, and the
  `cutlass3x_sm100_bstensorop_s256x128x64gemm_block_scaled_*` family. Those
  three alone remain roughly `6960 us/iter` in the aggregate table.

## 2026-06-18 - projection NCU probe and dense/proj geometry result

Added harness:

```text
.bench_runs_claude/megakernel/ncu_projection_probe.py
```

Purpose:

- isolate one production-shaped projection at a time without loading the full
  serving stack;
- profile BF16 MLA gate and NVFP4 q_b/o_proj/DSA/MLP projection shapes with
  Nsight Compute;
- use NVTX ranges (`projection_probe_direct`) so NCU can filter out setup
  quantization kernels and profile the projection call only.

Validation:

```text
python3 -m py_compile .bench_runs_claude/megakernel/ncu_projection_probe.py
git diff --check -- .bench_runs_claude/megakernel/ncu_projection_probe.py
```

Graph smoke in `optrt-bench-claude`:

```text
bf16_gate_half, M=32: 29.753 us
nvfp4_q_b, M=32, backend=cublaslt: 14.657 us
```

NCU setup:

- Host `ncu` around `docker exec` does not cross into the container process:

```text
.bench_runs_claude/results/260_ncu_bf16_gate_half_sol.csv
==WARNING== No kernels were profiled.
```

- Disposable container with the host NCU directory mounted works only with
  profiling privileges:

```text
docker run --rm --user root --gpus 'device=0' \
  --cap-add=SYS_ADMIN --security-opt seccomp=unconfined \
  -v /home/sjpat/TensorRT-LLM:/repo \
  -v /home/sjpat/.local/opt/nsight/extract/opt/nvidia/nsight-compute/2026.1.1:/opt/ncu \
  -w /repo \
  localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-fusionproof-20260609134115 \
  -lc '/opt/ncu/ncu ...'
```

Without those privileges, NCU reaches the target process but fails:

```text
.bench_runs_claude/results/261_ncu_bf16_gate_half_container_sol.csv
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
```

Usable raw NCU captures:

```text
.bench_runs_claude/results/262_ncu_bf16_gate_half_container_cap_sol.csv
.bench_runs_claude/results/267_ncu_nvfp4_qb_nvtx_pushpop_sol.csv
.bench_runs_claude/results/268_ncu_nvfp4_o_proj_splitk_nvtx_pushpop_sol.csv
```

Raw key rows:

```text
bf16_gate_half:
  kernel=nvjet_sm100_tst_64x32_64x16_2x1_2cta_v_bz_TNT
  duration=23.456-24.224 us
  memory=65.33-67.44%, dram=65.33-67.44%, sm=26.86-27.27%
  grid=128, block=256, cluster=2, regs/thread=255
  dyn_smem/block=180760, waves/sm=0.86
  theoretical_occupancy=12.50%, achieved_occupancy=9.48-9.55%

nvfp4_q_b:
  command used push/pop NVTX filter:
    --nvtx --nvtx-include "projection_probe_direct]"
  kernel=nvjet_sm100_ootst_256x128_256x4_4x1_2cta_v_bx_Avec16UE4M3_Bvec16UE4M3_TNT
  duration=10.656-10.944 us
  memory=25.53-26.24%, dram=25.53-26.24%, sm=11.39-11.82%
  grid=96, block=256, cluster=4, regs/thread=255
  dyn_smem/block=205104, waves/sm=0.65
  theoretical_occupancy=12.50%, achieved_occupancy=8.88-9.05%
  NCU rule: grid is too small; estimated launch-configuration speedup 35.14%

nvfp4_o_proj capture note:
  The disposable profiler image does not contain the in-container split-K
  modifications, so this capture is a normal o_proj baseline, not the split-K
  custom path.
  kernel=cutlass3x_sm100_bstensorop_s256x128x64gemm_block_scaled_ue4m3xf4_ue4m3xf4_f32_bf16_bf16_256x128x256_0_tnn_align32_o_vs16_2sm_bias_bf16_relu
  duration=21.824-22.208 us
  memory=39.44-40.15%, dram=39.44-40.15%, sm=18.45-19.01%
  grid=56, block=256, cluster=2, regs/thread=89
  dyn_smem/block=219136, waves/sm=0.38
  theoretical_occupancy=12.50%, achieved_occupancy=10.40-10.69%
```

Split-K profiling caveat:

- `optrt-bench-claude` has the modified package state and does report:

```text
NVFP4 split-K o_proj selected for M=32, N=7168, K=16384, split=2, atomic=True, output_buffer_kind=0
```

- A fresh disposable container from the same image does not have those
  in-container package edits; it profiles the image package and falls back to
  the normal cuBLASLt/CUTLASS path.
- Staging NCU into `optrt-bench-claude` reached the modified package, but the
  profiled Python process segfaulted during CUDA init:

```text
.bench_runs_claude/results/270_ncu_existing_nvfp4_o_proj_splitk_nvtx_pushpop_sol.csv
==PROF== Connected to process 362028 (/usr/bin/python3.12)
==ERROR== The application returned an error code (11).

.bench_runs_claude/results/271_ncu_existing_nvfp4_o_proj_splitk_fullpayload_nvtx_pushpop_sol.csv
==PROF== Connected to process 362182 (/usr/bin/python3.12)
==ERROR== The application returned an error code (11).
```

- The temporary `/tmp/ncu-min` profiler payload was removed after the failed
  attempts.

Backend spot check for q_b:

```text
backend=cublaslt:                 14.657 us graph replay
backend=cutlass,cublaslt,cuda_core: 15.601 us graph replay
backend=cutlass:                  15.476 us graph replay
backend=cuda_core:                invalid at M=32
backend=cutedsl:                  stopped after several minutes of tactic compilation
```

Targeted q_b CuTeDSL fallback-table tactics, bypassing full autotune:

```text
env:
  TRTLLM_NVFP4_GEMM_FALLBACK_PREFER_CUTEDSL=1
  TRTLLM_NVFP4_GEMM_FALLBACK_OVERRIDE_EXPLICIT=1

baseline cublaslt:                                14.657 us
((128,128),(1,1),swap_ab=false,prefetch=false):  16.709 us
((128,64), (1,1),swap_ab=false,prefetch=false):  19.402 us
((128,128),(1,1),swap_ab=true, prefetch=false):  16.913 us
((256,128),(1,1),swap_ab=false,prefetch=false):  invalid
```

The invalid `(256,128),(1,1)` tactic fails during Cute compile:

```text
ValueError: Cluster shape not divisible by MMA size: (1, 1) and 2:1
```

Conclusion from this tactic poke:

- The existing generic CuTeDSL dense kernel substrate does not beat cuBLASLt for
  q_b by just changing N tile or swap-AB.
- The q_b opportunity still exists, but it requires a more specific kernel
  design than the current `Sm100BlockScaledPersistentDenseGemmKernel` tactic
  space. The custom tactic needs to address the underfilled launch while
  preserving enough arithmetic intensity; simply increasing CTA count with a
  smaller N tile made runtime worse.

Interpretation:

- The dominant q_b NVFP4 projection is not compute-saturated. It is launch and
  geometry limited: only 96 blocks for 148 SMs, 0.65 waves/SM, about 11-12% SM
  throughput, and NCU explicitly flags launch configuration as the speedup
  opportunity.
- BF16 gate-half is also under one full wave and memory-heavy, but the current
  serving path overlaps the two gate halves on a side stream; prior attempts to
  collapse it into one full GEMM hurt overlap.
- The next principled kernel attempt should be a narrow q_b-shaped NVFP4 tactic
  that increases available parallel work per launch without adding an extra
  reduction launch: smaller N tile or split-N/grid expansion, lower cluster
  size, or a custom persistent schedule specialized for `M<=64, N=24576,
  K=1536`.
- Do not treat wrapper-level q_b/wq_b fusion as the answer anymore. It proved
  the production hook can remove launches, but the NCU result says the biggest
  per-call problem is the base GEMM launch geometry.

## 2026-06-18 - cuBLASLt tactic sweep and projection isolation follow-up

Added:

```text
.bench_runs_claude/megakernel/cublaslt_nvfp4_tactic_sweep.py
```

Purpose:

- Directly call `CublasLtFP4GemmRunner` for every heuristic tactic returned by
  cuBLASLt on production projection shapes.
- Separate tactic-selection problems from kernel-geometry problems before
  adding another production dispatch override.

Validation:

```text
python3 -m py_compile .bench_runs_claude/megakernel/cublaslt_nvfp4_tactic_sweep.py
```

q_b tactic sweep, CUDA graph timing:

```text
M=16: tactic0 14.673 us, tactic1 15.205 us -> best tactic0
M=24: tactic0 14.605 us, tactic1 16.039 us -> best tactic0
M=32: tactic0 14.745 us, tactic1 15.337 us -> best tactic0
M=64: tactic0 14.291 us, tactic1 14.549 us -> best tactic0
```

The first short M=16 sweep produced a 225 us outlier for one tactic, but it
disappeared with longer warmup and repeat count. Treat it as measurement noise,
not a production issue.

kv_a tactic sweep:

```text
M=16: only tactic0, 15.708 us after longer warmup
M=24: only tactic0, 15.107 us
M=32: only tactic0, 15.839 us
M=64: only tactic0, 15.903 us
```

o_proj cuBLASLt tactic sweep, CUDA graph timing:

```text
M=16: best tactic4, 22.730 us
M=24: best tactic2, 23.292 us
M=32: best tactic2/4 depending run, 23.011-23.436 us
M=64: best tactic3, 22.958 us
```

Direct timing has higher host+sync overhead but preserves the same direction:

```text
q_b M=32: best tactic0, 25.927 us
o_proj M=32: best tactic3, 35.716 us
```

Plain `nvfp4_gemm` projection isolation, CUDA graph timing:

```text
nvfp4_q_b:
  M=16 14.586 us
  M=24 14.670 us
  M=32 14.866 us
  M=64 14.759 us

nvfp4_kv_a:
  M=16 15.775 us
  M=24 15.854 us
  M=32 15.809 us
  M=64 15.715 us

nvfp4_o_proj baseline:
  M=16 23.016 us
  M=24 23.029 us
  M=32 23.049 us
  M=64 22.936 us

nvfp4_o_proj split-K atomic:
  M=16 19.483 us
  M=24 18.972 us
  M=32 19.283 us
  M=64 20.834 us
```

Current conclusion:

- q_b is not a hidden cuBLASLt tactic-selection issue. The best heuristic tactic
  is already the default/baseline path for the decode M range.
- o_proj split-K remains a real isolated-kernel win, about 3.6-4.1 us per call
  for M=16-32 and about 2.1 us for M=64. This matches the earlier production
  signal where the env-gated split-K path reduced the o_proj slice but did not
  solve the larger dense/proj bucket alone.
- Do not add a q_b cuBLASLt tactic override. It would force the same tactic.
- Do not add an o_proj tactic override before serving A/B. Plain `nvfp4_gemm`
  already lands close to the best cuBLASLt tactic in isolation, and split-K is
  the larger o_proj lever.
- Next q_b work must change launch geometry or scheduling, not tactic choice:
  the NCU issue remains 96 CTAs for 148 SMs, 0.65 waves/SM, and about 11-12% SM
  throughput.

## 2026-06-19 - q_b CuTeDSL geometry probe and guarded production carrier

Added:

```text
.bench_runs_claude/megakernel/cutedsl_nvfp4_tactic_probe.py
```

Purpose:

- Probe CuTeDSL NVFP4 tactics on the under-occupied q_b projection shape
  instead of staying inside cuBLASLt heuristic tactics.
- Keep the experiment narrow: only `M=16..64, N=24576, K=1536` and BF16
  output. Small warmup shapes stay on the existing path.

Key microbench result:

```text
q_b M=32 baseline cuBLASLt:              14.678 us
q_b M=32 CuTeDSL 128x64 cluster 4x1:    13.566 us

q_b M=16 baseline / CuTeDSL 4x1:        14.663 / 14.279 us
q_b M=24 baseline / CuTeDSL 4x1:        14.736 / 14.219 us
q_b M=64 baseline / CuTeDSL 4x1:        14.562 / 14.300 us
```

The direct production hook microbench, run through `nvfp4_gemm`, also showed
the direction:

```text
q_b M=32, no env:                         14.997 us
q_b M=32, TRTLLM_NVFP4_GEMM_QB_CUTEDSL=1: 14.378 us
```

Serving carrier attempts:

```text
optrt-529374445d-codex-qb-cutedsl-full-20260618T233744Z
optrt-529374445d-codex-qb-cutedsl-min16-20260618T234909Z
optrt-529374445d-codex-qb-cutedsl-min16-qbwqbbase-20260619T000257Z
```

Observed behavior:

- The first full carrier fired q_b CuTeDSL on CUDA graph warmup shapes below
  the measured range (`M=3..15`) and caused a startup compile storm. Tail:
  `.bench_runs_claude/results/274_qb_cutedsl_startup_hang_decode_tail.log`.
- The min-M carrier avoided the small-M q_b hook but still did not become a
  useful serving candidate. Tail:
  `.bench_runs_claude/results/275_qb_cutedsl_min16_startup_unready_decode_tail.log`.
- The qbwqb-base carrier did become ready after a long decode warmup, but the
  C32 production A/B did not complete. Client output:
  `.bench_runs_claude/results/276_qb_cutedsl_qbwqbbase_c32.txt`.
- During that C32 attempt, prefill and decode logs showed KV cache transfer
  timeouts and repeated `RxSession.close deferred ... TRANSFERRING` messages.
  The run was stopped and the deployment was restored to BEST_CONFIG.

Restored known-good deployment:

```text
arm:   vbfuse_nvfp4_mega_cf
image: localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-vbfuse-nvfp4-20260615T043037Z
state: decode and prefill both 1/1 ready, zero restarts
```

## 2026-06-19 - Dense-family combined arm vs original BEST_CONFIG

Current arm:

```text
arm:   dense_family_combo_mega_cf
image: localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-dense-family-combo-20260619T041459Z
digest: sha256:045053b7aa739a55e6845dc466effe1b334085dc511429b2d28c5ece6d27db1f
```

This combines the q_b/wq_b variable-N path with the KVA/WK/WP BF16 DSA path
on top of the original BEST_CONFIG transport and MoE settings:

```text
TRTLLM_OPTRT_MOE_MEGAKERNEL=1
TRTLLM_FORCE_COMM_METHOD=NVLINK_TWO_SIDED
use_low_precision_moe_combine=false
TRTLLM_INDEXER_FUSE_QB_WQB=1
TRTLLM_INDEXER_FUSE_QB_WQB_STATIC=1
TRTLLM_INDEXER_FUSE_QB_WQB_VARIABLE_N=1
TRTLLM_INDEXER_FUSE_QB_WQB_ALIAS_ORIGINALS=1
TRTLLM_INDEXER_FUSE_KVA_WKWP=1
TRTLLM_INDEXER_FUSE_KVA_WKWP_STATIC=1
TRTLLM_INDEXER_FUSE_KVA_WKWP_BF16_DSA=1
TRTLLM_INDEXER_FUSE_KVA_WKWP_ALIAS_ORIGINALS=1
```

Deployment note:

- The first attempt accidentally used a stale carrier image and crashed during
  CUDA graph capture because `torch.ops.trtllm` did not contain
  `cute_dsl_nvfp4_qb_wqb_gemm_blackwell_out`.
- Added a fallback guard in `_FusedQbWqBNvfp4.__call__` so variable-N only
  uses the custom op when it is registered; otherwise it falls back to the
  concatenated `nvfp4_gemm` route.
- Rebuilt and deployed the combined overlay image above. Decode and prefill
  reached 1/1 ready with zero restarts.

Benchmark files:

```text
.bench_runs_claude/results/286_dense_family_combo_c32_512_r1.txt
.bench_runs_claude/results/287_dense_family_combo_c32_512_r2.txt
.bench_runs_claude/results/288_dense_family_combo_osl512_full_hot.txt
```

Focused C32/OSL512 repeats:

```text
r1: C32 TTFT_p50=1884 ms, user_tok/s=51.84, agg_out_tok/s=1386.8
r2: C32 TTFT_p50=1900 ms, user_tok/s=52.21, agg_out_tok/s=1346.3
```

Full hot OSL512 sweep, ISL 2055:

```text
  C    TTFT_p50   user_tok/s   agg_out_tok/s
  1       483        44.79          46.6
  4       770        58.25         229.3
  8      1001        55.93         428.1
 16      1160        54.71         810.1
 32      1657        51.92        1395.6
 64      1481        46.48        2321.4
```

Comparison to original frozen BEST_CONFIG:

```text
BEST_CONFIG C32:      TTFT_p50=1396 ms, user_tok/s=51.07, agg_out_tok/s=1424.9
dense-family C32:     TTFT_p50=1657 ms, user_tok/s=51.92, agg_out_tok/s=1395.6
delta:                TTFT +18.7%, user tok/s +1.7%, aggregate -2.1%
```

Comparison to later restored sanity:

```text
restored C32:         TTFT_p50=1642 ms, user_tok/s=50.08, agg_out_tok/s=1386.6
dense-family C32:     TTFT_p50=1657 ms, user_tok/s=51.92, agg_out_tok/s=1395.6
delta:                TTFT +0.9%, user tok/s +3.7%, aggregate +0.6%
```

Interpretation:

- This is the first combined dense-family run that looks directionally useful
  in steady decode: user tok/s is above both the original frozen C32 row and
  the later restored-control C32 row.
- It is still not a promotion candidate because the primary C32 aggregate
  number remains below the original frozen best by about 2%. The C32 TTFT is
  also much worse than the original frozen sweep, which eats the decode-rate
  gain at the request-level aggregate metric.
- Outside C32, the hot sweep is broadly positive versus the original frozen
  table: C1/C4/C8/C16/C64 all beat the original aggregate values. C64 also
  beats the original aggregate but is roughly flat versus the later restored
  C64 sanity.
- Next work should separate whether the C32 aggregate miss is TTFT/admission
  behavior from the combined dense hooks, CUDA graph capture side effects, or
  KV-router/prefill timing. The useful profile question is no longer "does the
  steady decode rate move?" but "why does C32 wall/TTFT not convert that into
  aggregate throughput?"

Interpretation:

- CuTeDSL can improve the isolated q_b kernel by about 0.3-1.1 us depending on
  M, so the geometry hypothesis is real.
- The current Python-level q_b CuTeDSL carrier is not production-safe. Its
  startup cost is high, and the qbwqb-base serving A/B regressed into KV
  transfer timeouts before any throughput number could be collected.
- Do not keep pushing this as a drop-in Python dispatch override. The next
  principled step is either a precompiled/static q_b kernel path with no
  request-time CuTeDSL compile behavior, or a lower-risk target with an already
  validated production hook, such as o_proj split-K serving A/B.

## 2026-06-19 - o_proj split-K atomic production A/B

Arm:

```text
splitk_o_proj_atomic_mega_cf
image: localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-splitk-o-proj-atomic-20260618T111927Z
```

Delta versus BEST_CONFIG:

- Keeps `TRTLLM_OPTRT_MOE_MEGAKERNEL=1`, `NVLINK_TWO_SIDED`, and
  `use_low_precision_moe_combine=false`.
- Routes small-M `N=7168, K=16384` NVFP4 o_proj through atomic split-K=2.
- Leaves q_b, kv_a, MoE dense, and other projection shapes on the existing
  path.

Startup observation:

- Decode readiness was slow because the split-K hook fired during CUDA graph
  warmup for tiny shapes:

```text
M=9,8,7,6,5,4,3 selected for N=7168,K=16384, split=2, atomic=True
```

This did eventually become ready, but it is the same class of startup behavior
that made the q_b CuTeDSL hook risky. A min-M guard is required before this is
worth another production candidate pass.

C32 serving A/B:

```text
BEST_CONFIG control repeat before this test:
  C=32 reqs=64 ok=64 ISL=2055 TTFT_p50=1665 ms TTFT_p95=3404 ms
  user_tok_s_p50=50.68 agg_out_tok_s=1376.1 wall=23.8
  file: .bench_runs_claude/results/273_bestconfig_c32_control_repeat_before_qb_cutedsl.txt

split-K atomic first run, cold/warmup contaminated:
  C=32 reqs=64 ok=64 ISL=2055 TTFT_p50=21762 ms TTFT_p95=26658 ms
  user_tok_s_p50=49.60 agg_out_tok_s=686.7 wall=47.7
  file: .bench_runs_claude/results/277_splitk_o_proj_atomic_c32.txt

split-K atomic repeat:
  C=32 reqs=64 ok=64 ISL=2055 TTFT_p50=1601 ms TTFT_p95=3596 ms
  user_tok_s_p50=50.14 agg_out_tok_s=1349.0 wall=24.3
  file: .bench_runs_claude/results/278_splitk_o_proj_atomic_c32_repeat.txt

split-K atomic repeat2:
  C=32 reqs=64 ok=64 ISL=2055 TTFT_p50=1456 ms TTFT_p95=3481 ms
  user_tok_s_p50=50.13 agg_out_tok_s=1389.3 wall=23.6
  file: .bench_runs_claude/results/279_splitk_o_proj_atomic_c32_repeat2.txt
```

Restored known-good deployment after the test:

```text
arm:   vbfuse_nvfp4_mega_cf
image: localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-vbfuse-nvfp4-20260615T043037Z
state: decode and prefill both 1/1 ready, zero restarts
```

## 2026-06-19 - Linear-level o_proj CuTeDSL bypass

Hypothesis:

- The direct CuTeDSL o_proj tactic probe was positive, but the generic NVFP4
  fallback carrier regressed serving. Test whether routing at the
  `NVFP4LinearMethod` level avoids the generic fallback overhead.

Implementation:

```text
code:  tensorrt_llm/_torch/modules/linear.py
arm:   dense_family_sigmoid_o_linear_cutedsl_mega_cf
image: localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-dense-family-sigmoid-o-linear-cutedsl-20260619T074958Z
base:  dense_family_sigmoid_mega_cf
env:   TRTLLM_NVFP4_LINEAR_O_CUTEDSL=1
       MIN_M=16 MAX_M=64 N=7168 K=16384
       tactic=((128,64),(1,1),swap_ab=True,prefetch=False)
```

Artifacts:

```text
.bench_runs_claude/results/337_o_linear_hook_probe_m8_16_32_64.txt
.bench_runs_claude/results/338_deploy_dense_family_sigmoid_o_linear_cutedsl.txt
.bench_runs_claude/results/339_dense_family_sigmoid_o_linear_cutedsl_handshake.txt
.bench_runs_claude/results/340_dense_family_sigmoid_o_linear_cutedsl_c32_512_r1.txt
.bench_runs_claude/results/341_dense_family_sigmoid_o_linear_cutedsl_c32_512_r2.txt
.bench_runs_claude/results/342_dense_family_sigmoid_o_linear_cutedsl_c32_512_r3.txt
.bench_runs_claude/results/343_restore_dense_family_sigmoid_after_o_linear_cutedsl.txt
```

Isolated hook probe:

```text
M=8  routed=False baseline_us=23.918
M=16 routed=True  baseline_us=23.045 hook_us=20.648 cos=0.99999571
M=32 routed=True  baseline_us=23.604 hook_us=20.752 cos=0.99999619
M=64 routed=True  baseline_us=24.083 hook_us=20.574 cos=0.99999607
```

Serving result versus original BEST_CONFIG C32/OSL512:

```text
Original BEST_CONFIG:
  TTFT_p50=1396 ms, user_tok_s_p50=51.07, agg_out_tok_s=1424.9

dense_family_sigmoid_o_linear_cutedsl_mega_cf:
  r1: TTFT_p50=1800 ms, user_tok_s_p50=51.10, agg_out_tok_s=1390.1
  r2: TTFT_p50=1405 ms, user_tok_s_p50=50.92, agg_out_tok_s=1396.2
  r3: TTFT_p50=1537 ms, user_tok_s_p50=51.46, agg_out_tok_s=1409.3
```

Comparison:

- The Linear-level hook preserves the isolated o_proj microbench win and is
  less damaging than the generic fallback carrier, but it still does not beat
  either original BEST_CONFIG or the best dense-family+sigmoid focused C32
  rows.
- Best serving repeat is `1409.3` aggregate tok/s, about `1.1%` below original
  BEST_CONFIG and about `0.8%` below the best dense-family+sigmoid focused row
  at `1420.6`.
- Startup is worse: decode reached Ready after about 7.7 minutes and the
  handshake still had cold-tail TTFT (`p95=21664 ms`), consistent with extra
  CuTeDSL compilation.

Verdict:

- Do not promote the o_proj CuTeDSL route as a standalone serving change.
- Keep the finding as evidence that isolated GEMM wins are being lost in graph
  compile/runtime scheduling overhead. The next dense/proj attempt should
  target a precompiled or broader fused projection path, not another narrow
  runtime CuTeDSL route for one shape family.

Restored current known-good candidate after the test:

```text
arm:   dense_family_sigmoid_mega_cf
image: localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-dense-family-sigmoid-20260619T045508Z
state: decode and prefill both 1/1 ready, zero restarts
```

## 2026-06-19 - Recenter versus original BEST_CONFIG after gate misses

Original frozen BEST_CONFIG C32/OSL512 target:

```text
TTFT_p50=1396 ms, user_tok_s_p50=51.07, agg_out_tok_s=1424.9
```

Latest candidate rows:

```text
dense_family_sigmoid_mega_cf focused r1:
  TTFT_p50=1291 ms, user_tok_s_p50=51.76, agg_out_tok_s=1420.6

dense_family_sigmoid_mega_cf focused r2:
  TTFT_p50=1432 ms, user_tok_s_p50=52.56, agg_out_tok_s=1419.1

dense_family_sigmoid_mega_cf full hot sweep C32:
  TTFT_p50=1620 ms, user_tok_s_p50=51.88, agg_out_tok_s=1405.0

dense_family_sigmoid_mega_cf later repeat:
  TTFT_p50=1753 ms, user_tok_s_p50=50.88, agg_out_tok_s=1366.0

dense_family_sigmoid_fp4norm_mega_cf repeats:
  TTFT_p50=1603/1538 ms, user_tok_s_p50=50.95/50.89,
  agg_out_tok_s=1394.1/1403.0
```

Current profile read:

```text
dense_family_sigmoid C32/OSL512, 128-step Kineto:
  steady per-iteration idle: ~3%
  Dense/proj GEMM:           ~8.9 ms/iter, 608 launches/iter
  top BF16 gate half GEMM:   ~2.85 ms/iter, 122 calls/iter
  top NVFP4 projection GEMM: ~2.75 ms/iter, 177 calls/iter
  o_proj/block-scaled GEMM:  ~1.34 ms/iter, 64 calls/iter
```

Verdict:

- We are close in steady decode, but not ahead of original BEST_CONFIG on the
  primary C32 aggregate acceptance metric.
- The current stack is generally better than the later restored-control band,
  but the frozen original still has the best C32 aggregate row.
- The fp4norm pass-order patch may be a real compiler fix, but the live image
  is not a promotion candidate by serving numbers.
- The latest BF16 gate attempts did not produce a viable hook:
  cublas wrapper slower, cublasLt split-K slower, full-gate serving worse,
  CuTe tactic sweep slower, L-batch serving worse, naive WMMA slower, and the
  Triton prototype correct but 1.4-2.3x slower than the existing path.
- The unmapped `N=192,K=7168` projection is `weights_proj`; it is already
  covered by the DSA wk/weights and KVA/WK/WP fusion branches. It is not a new
  hidden projection family.

Next target:

- Stop spending cycles on wrappers around the current BF16 gate GEMM unless we
  write a genuinely different half-gate kernel that preserves side-stream
  overlap.
- If staying on NVFP4 projections, the remaining principled direction is a
  first-class/precompiled multi-problem projection substrate. The previous
  Python/CuTe carriers proved that small direct wins are erased by dispatch,
  startup, or admission effects.
- Any next serving candidate must show a profile-level movement larger than
  the current 32-launch / ~200-325 us dense/proj reductions; otherwise C32
  aggregate will stay inside noise or below the original target.

## 2026-06-19 - Narrow o_proj CuTeDSL fallback recenter

Goal: test whether the remaining NVFP4 `o_proj` bucket can be moved by the
direct CuTeDSL dense GEMM substrate instead of the previous split-K route. This
was intentionally scoped to the production `o_proj` shape
`N=7168,K=16384`; M1/M8 were checked to avoid hurting tiny decode shapes.

Microbench artifacts:

```text
.bench_runs_claude/results/326_cutedsl_o_proj_m32_recenter.txt
.bench_runs_claude/results/327_cutedsl_o_proj_m16_recenter.txt
.bench_runs_claude/results/328_cutedsl_o_proj_m64_recenter.txt
.bench_runs_claude/results/329_cutedsl_o_proj_m1_recenter.txt
.bench_runs_claude/results/330_cutedsl_o_proj_m8_recenter.txt
.bench_runs_claude/results/331_cutedsl_o_proj_m24_recenter.txt
```

Direct-kernel result, median us:

```text
M1:  cuBLASLt 24.227, CuTe 128x64 1x1 swap 25.645  -> slower
M8:  cuBLASLt 23.757, CuTe 128x64 1x1 swap 23.695  -> flat
M16: cuBLASLt 23.162, CuTe 128x64 1x1 swap 20.510  -> +11.4%
M24: cuBLASLt 24.188, CuTe 128x64 1x1 swap 19.847  -> +17.9%
M32: cuBLASLt 23.877, CuTe 128x64 1x1 swap 20.558  -> +13.9%
M64: cuBLASLt 23.556, CuTe 128x64 1x1 swap 21.087  -> +10.5%
```

Production hook:

```text
file: .bench_runs_claude/deploy_arm.py
arm:  dense_family_sigmoid_o_cutedsl_mega_cf
env:  dense_family_sigmoid_mega_cf plus
      TRTLLM_NVFP4_GEMM_FALLBACK_PREFER_CUTEDSL=1
      TRTLLM_NVFP4_GEMM_FALLBACK_OVERRIDE_EXPLICIT=1
      TRTLLM_NVFP4_GEMM_FALLBACK_CUTEDSL_MAX_M=64
      TRTLLM_NVFP4_GEMM_FALLBACK_CUTEDSL_POLICY=high_impact
      TRTLLM_NVFP4_GEMM_FALLBACK_CUTEDSL_TACTIC_TABLE only for
        M={16,24,32,64}, N=7168, K=16384 ->
        ((128,64),(1,1),swap_ab=True,prefetch=False)
```

Decode logs confirmed the intended route fired on all ranks for M16/M24/M32/M64:

```text
NVFP4 GEMM fallback selected CuTeDSL for M=64, N=7168, K=16384
NVFP4 GEMM fallback selected CuTeDSL for M=32, N=7168, K=16384
NVFP4 GEMM fallback selected CuTeDSL for M=24, N=7168, K=16384
NVFP4 GEMM fallback selected CuTeDSL for M=16, N=7168, K=16384
```

Serving artifacts:

```text
.bench_runs_claude/results/332_deploy_dense_family_sigmoid_o_cutedsl.txt
.bench_runs_claude/results/333_dense_family_sigmoid_o_cutedsl_handshake.txt
.bench_runs_claude/results/334_dense_family_sigmoid_o_cutedsl_c32_512_r1.txt
.bench_runs_claude/results/335_dense_family_sigmoid_o_cutedsl_c32_512_r2.txt
```

Serving result, C32/OSL512:

```text
r1: TTFT_p50=1989 ms, user_tok_s_p50=51.98, agg_out_tok_s=1342.5
r2: TTFT_p50=1801 ms, user_tok_s_p50=51.48, agg_out_tok_s=1363.8
```

Verdict:

- The direct CuTeDSL `o_proj` tactic is real in a DEFAULT-output isolated
  microbench, but it does not survive the production `Linear` path.
- User tok/s remains decent, but aggregate collapses versus both the original
  BEST_CONFIG `1424.9` and the best `dense_family_sigmoid_mega_cf` focused
  rows `1420.6/1419.1`.
- Do not promote the broad or narrow CuTe fallback carrier as-is.
- Follow-up debug evidence from
  `.bench_runs_claude/results/81_nvfp4_debug_shape_sites.txt` shows
  production `self_attn.o_proj` is still `output_buffer_kind=0`, `group=False`,
  and `tp_size=1`, so the miss is not NCCL-window output allocation.
- The likely missing piece is the fallback-carrier/runtime behavior under the
  full serving graph, not the tactic itself. Next o_proj work should either use
  a first-class package op/precompiled route with no generic fallback carrier,
  or compare the same CuTe tactic inside a captured production-layer replay
  before any further serving rollout.

## 2026-06-19 - Distributed add+RMSNorm+NVFP4-quant fallback check

Question: docs say add+RMSNorm+NVFP4 quant fusion is automatic under
`torch.compile`, but the production distributed profile still showed a large
standalone quant/dequant bucket. The compile backend confirmed the likely gap:
single-rank mode registered `register_add_norm_fp4_quant` before plain
`register_add_norm`, while `world_size > 1` only registered AR fusions and then
plain add+norm fallback. That means any non-AR add+norm+fp4-quant triple could be
consumed by the plain fallback before the FP4 quant fusion ever had a chance.

Patch:

```text
tensorrt_llm/_torch/compilation/backend.py
  world_size > 1:
    register_ar_fusions(...)
    register_add_norm_fp4_quant(...)
    register_add_norm(...)
```

Packaging:

```text
image:
  localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-dense-family-sigmoid-fp4normdist-20260619T063416Z
base:
  optrt-529374445d-codex-dense-family-sigmoid-20260619T045508Z
overlay:
  backend.py only
arm:
  dense_family_sigmoid_fp4norm_mega_cf
```

Validation:

```text
python3 -m py_compile:
  tensorrt_llm/_torch/compilation/backend.py
  tests/unittest/_torch/test_compilation_backend_passes.py
  .bench_runs_claude/deploy_arm.py

pytest inside patched serving image with GPU passthrough:
  docker run --rm --gpus all ... -m pytest -q /tmp/test_compilation_backend_passes.py
  result: 1 passed, 39 warnings
```

Serving A/B versus original BEST_CONFIG C32/OSL512:

```text
Original BEST_CONFIG:
  TTFT_p50=1396 ms, user_tok_s_p50=51.07, agg_out_tok_s=1424.9

Current pre-patch dense_family_sigmoid_mega_cf fresh repeat:
  file: .bench_runs_claude/results/318_dense_family_sigmoid_c32_512_r3.txt
  C=32 ok=64/64 TTFT_p50=1753 ms, user_tok_s_p50=50.88,
  agg_out_tok_s=1366.0

Patched fp4norm distributed fallback r1:
  file: .bench_runs_claude/results/320_dense_family_sigmoid_fp4norm_c32_512_r1.txt
  C=32 ok=64/64 TTFT_p50=1603 ms, user_tok_s_p50=50.95,
  agg_out_tok_s=1394.1

Patched fp4norm distributed fallback r2:
  file: .bench_runs_claude/results/321_dense_family_sigmoid_fp4norm_c32_512_r2.txt
  C=32 ok=64/64 TTFT_p50=1538 ms, user_tok_s_p50=50.89,
  agg_out_tok_s=1403.0
```

Interpretation:

- The compile-pass ordering bug is real and now guarded by a unit test.
- The patched image starts cleanly and the disagg handshake succeeds.
- E2E improves versus the worst immediately preceding dense-family repeat, but
  it is still below the original frozen BEST_CONFIG at the operating point:
  `1403.0` best patched repeat versus `1424.9` original aggregate tok/s.
- This is not a promotion candidate. Keep the fix only if later profiling shows
  the fusion actually fires and helps another combined arm; otherwise the next
  target remains the dominant dense/proj GEMM families, not this fallback.

## 2026-06-19 - Current campaign state

Current restored serving arm:

```text
arm:   dense_family_sigmoid_mega_cf
image: localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-dense-family-sigmoid-20260619T045508Z
state: decode=1/1, prefill=1/1, frontend=1/1, zero decode restarts
```

Latest comparison to original BEST_CONFIG C32/OSL512:

```text
BEST_CONFIG:
  TTFT_p50=1396 ms, user_tok_s_p50=51.07, agg_out_tok_s=1424.9

dense_family_sigmoid_mega_cf focused C32:
  results/293_dense_family_sigmoid_c32_512_r1.txt
    TTFT_p50=1291 ms, user_tok_s_p50=51.76, agg_out_tok_s=1420.6
  results/294_dense_family_sigmoid_c32_512_r2.txt
    TTFT_p50=1432 ms, user_tok_s_p50=52.56, agg_out_tok_s=1419.1

dense_family_sigmoid_mega_cf full sweep C32:
  results/295_dense_family_sigmoid_osl512_full_hot.txt
    TTFT_p50=1620 ms, user_tok_s_p50=51.88, agg_out_tok_s=1405.0
```

Latest profile:

```text
analysis: .bench_runs_claude/profiles/kineto_analysis_dense_family_sigmoid_c32_512_20260619T0520/analysis.txt

Dense/proj GEMM   8930 us/iter, 38.1%, 608 launches/iter
MoE a2a/comm      3388 us/iter, 14.5%, 290 launches/iter
MoE expert GEMM   3130 us/iter, 13.4%, 174 launches/iter
DSA indexer       1779 us/iter,  7.6%, 263 launches/iter
Elementwise/copy  1342 us/iter,  5.7%, 749 launches/iter
Quant/dequant     1238 us/iter,  5.3%, 360 launches/iter
```

Latest rejected A/Bs:

```text
full gate:
  results/301_dense_family_sigmoid_gate_full_c32_512_r1.txt
    agg_out_tok_s=1399.5
  results/302_dense_family_sigmoid_gate_full_c32_512_r2.txt
    agg_out_tok_s=1380.4

TRT-LLM cublas_mm wrapper for BF16 gate half:
  results/304_bf16_gate_cublasmm_compare.txt
  M32 torch_two=47.96 us, cublas_two=56.34 us, ratio=1.175
  M64 torch_two=50.00 us, cublas_two=57.82 us, ratio=1.156

dense-family+sigmoid+o_proj split-K atomic:
  results/307_dense_family_sigmoid_splitk_o_atomic_c32_512_r1.txt
    TTFT_p50=2203 ms, user_tok_s_p50=50.30, agg_out_tok_s=1314.2

cublasLt BF16 gate heuristic sweep:
  results/310_bf16_gate_cublaslt_heuristic_m32_m64_retry.txt
  M32 best splitK=2 candidate ratio=1.156; best no-split ratio=0.994
  M64 best splitK=2 candidate ratio=1.223; best no-split ratio=0.993

cublasLt manual NVJET tactic sweep:
  results/311_bf16_gate_cublaslt_manual_m32_m64.txt
  short run found one apparent M32 cga2 variant near ratio=0.973, but
  confirmation did not hold.

  results/312_bf16_gate_cublaslt_manual_m32_m64_confirm.txt
  M32 best explicit/default:
    algo=66 tile=13 stages=35 splitK=1 reduction=0 custom=1 cga=3
    19.243 us vs default 19.265 us
  M64 best explicit/default:
    algo=66 tile=15 stages=35 splitK=1 reduction=0 custom=1 cga=3
    19.552 us vs default 19.564 us

standalone WMMA BF16 half-gate prototypes:
  results/314_bf16_gate_wmma_proto_m16_m32_m64_retry.txt
  M32 best=167.904 us vs torch_half=19.407 us, ratio=8.652
  M64 best=258.200 us vs torch_half=19.667 us, ratio=13.129

  results/315_bf16_gate_wmma_shareda_proto_m16_m32_m64.txt
  M32 best=387.066 us vs torch_half=19.345 us, ratio=20.009
  M64 best=384.629 us vs torch_half=19.750 us, ratio=19.475

  results/316_bf16_gate_wmma_stage64_proto_m16_m32_m64.txt
  M32 best=383.789 us vs torch_half=19.447 us, ratio=19.735
  M64 best=387.619 us vs torch_half=19.754 us, ratio=19.622

full CuTe persistent BF16 gate tactic sweep:
  results/317_bf16_gate_cutedsl_full_tactic_m32_m64.txt
  M32 best=30.26 us vs torch=29.42 us, ratio=1.029
    best_tactic=Tactic(use_2cta=True, mn=(128, 128), cluster=(2, 1))
  M64 best=30.71 us vs torch=29.87 us, ratio=1.028
    best_tactic=Tactic(use_2cta=False, mn=(64, 64), cluster=(1, 1))

small-M CuTe persistent BF16 gate tactic sweep:
  results/322_bf16_gate_cutedsl_quick_m4_8_16_24_32.txt
  M4  best=31.27 us vs torch=28.25 us, ratio=1.107
    best_tactic=Tactic(use_2cta=False, mn=(64, 64), cluster=(1, 4))
  M8  best=33.13 us vs torch=29.17 us, ratio=1.136
    best_tactic=Tactic(use_2cta=False, mn=(64, 64), cluster=(1, 2))
  M16 best=30.22 us vs torch=29.25 us, ratio=1.033
    best_tactic=Tactic(use_2cta=False, mn=(64, 64), cluster=(1, 1))
  M24 best=30.19 us vs torch=29.70 us, ratio=1.016
    best_tactic=Tactic(use_2cta=False, mn=(64, 64), cluster=(1, 1))
  M32 best=30.48 us vs torch=29.58 us, ratio=1.031
    best_tactic=Tactic(use_2cta=False, mn=(64, 64), cluster=(1, 4))

benchmark-only Triton BF16 half-gate prototype:
  script:  .bench_runs_claude/megakernel/bf16_gate_triton_proto.py
  result:  .bench_runs_claude/results/323_bf16_gate_triton_quick_m8_16_24_32_64.txt
  shape:   X[M,7168] @ W_half[8192,7168].T -> [M,8192]
  M8  best=26.819 us vs torch=19.048 us, ratio=1.408
    best_tactic=Tactic(block_m=16, block_n=64, block_k=128, group_m=1, num_warps=4, num_stages=4)
  M16 best=26.993 us vs torch=19.192 us, ratio=1.406
    best_tactic=Tactic(block_m=16, block_n=64, block_k=128, group_m=1, num_warps=4, num_stages=4)
  M24 best=30.994 us vs torch=19.320 us, ratio=1.604
    best_tactic=Tactic(block_m=16, block_n=64, block_k=128, group_m=1, num_warps=4, num_stages=4)
  M32 best=30.942 us vs torch=19.345 us, ratio=1.600
    best_tactic=Tactic(block_m=16, block_n=64, block_k=128, group_m=1, num_warps=4, num_stages=4)
  M64 best=45.013 us vs torch=19.760 us, ratio=2.278
    best_tactic=Tactic(block_m=32, block_n=64, block_k=128, group_m=1, num_warps=4, num_stages=4)
```

Current conclusion:

- The current dense-family+sigmoid stack is close to BEST_CONFIG but does not
  clearly beat the C32 aggregate operating point.
- Wrapper/config combinations are exhausted for the dominant MLA gate family:
  full-gate collapse, generic CuTe l-batch, DSV3 gate, TRT-LLM cublas_mm,
  cublasLt split-K, manual cublasLt NVJET tactic tweaks, naive WMMA, and the
  expanded generic CuTe persistent tactic sweep all miss. The follow-up
  small-M sweep also misses, so there is no remaining evidence for a
  production hook that only changes the existing CuTe tactic list. A generic
  Triton small-M matmul prototype also misses by a wide margin, so Triton is
  not a useful production substrate for the half-gate GEMM in this form.
- The next implementation target should be a new BF16 half-gate kernel that
  preserves two-half overlap while improving the per-half floor, or a more
  radical dense/proj restructuring. A LUT/config-only change is not supported
  by the current evidence.

## 2026-06-19 - Dense-family + sigmoid profile and full-gate A/B

Current dense-family+sigmoid arm:

```text
arm:   dense_family_sigmoid_mega_cf
image: localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-dense-family-sigmoid-20260619T045508Z
env:   MEGAKERNEL=1, NVLINK_TWO_SIDED, combine=false
       TRTLLM_OPTRT_FUSED_SIGMOID_MUL_QUANT_NVFP4=1
       TRTLLM_INDEXER_FUSE_QB_WQB=1
       TRTLLM_INDEXER_FUSE_QB_WQB_STATIC=1
       TRTLLM_INDEXER_FUSE_QB_WQB_VARIABLE_N=1
       TRTLLM_INDEXER_FUSE_QB_WQB_POST_SCALE=0
       TRTLLM_INDEXER_FUSE_KVA_WKWP=1
       TRTLLM_INDEXER_FUSE_KVA_WKWP_STATIC=1
       TRTLLM_INDEXER_FUSE_KVA_WKWP_BF16_DSA=1
```

Serving result versus original BEST_CONFIG C32/OSL512:

```text
BEST_CONFIG:
  TTFT_p50=1396 ms, user_tok_s_p50=51.07, agg_out_tok_s=1424.9

dense_family_sigmoid_mega_cf focused C32 runs:
  results/293_dense_family_sigmoid_c32_512_r1.txt
    TTFT_p50=1291 ms, user_tok_s_p50=51.76, agg_out_tok_s=1420.6
  results/294_dense_family_sigmoid_c32_512_r2.txt
    TTFT_p50=1432 ms, user_tok_s_p50=52.56, agg_out_tok_s=1419.1

dense_family_sigmoid_mega_cf full OSL512 hot sweep:
  results/295_dense_family_sigmoid_osl512_full_hot.txt
    C1  agg_out_tok_s=46.4    (+2.4% vs BEST_CONFIG)
    C4  agg_out_tok_s=229.7   (+3.4%)
    C8  agg_out_tok_s=422.6   (+2.1%)
    C16 agg_out_tok_s=780.2   (-0.4%)
    C32 agg_out_tok_s=1405.0  (-1.4%)
    C64 agg_out_tok_s=2359.4  (+5.8%)
```

Kineto profile:

```text
trace:    /var/lib/optrt-cache/nsys/dense_family_sigmoid_c32_512_trace-rank-*.json
analysis: .bench_runs_claude/profiles/kineto_analysis_dense_family_sigmoid_c32_512_20260619T0520/analysis.txt
window:   TLLM_PROFILE_START_STOP=3000-3128
files:    results/296_dense_family_sigmoid_profile_c32_512.txt
          results/297_dense_family_sigmoid_profile_c32_512_extra.txt
```

Aggregate profile:

```text
Dense/proj GEMM   8930 us/iter, 38.1%, 608 launches/iter
MoE a2a/comm      3388 us/iter, 14.5%, 290 launches/iter
MoE expert GEMM   3130 us/iter, 13.4%, 174 launches/iter
GEMM generic      1934 us/iter,  8.3%, 403 launches/iter
DSA indexer       1779 us/iter,  7.6%, 263 launches/iter
Elementwise/copy  1342 us/iter,  5.7%, 749 launches/iter
Quant/dequant     1238 us/iter,  5.3%, 360 launches/iter
```

Top dense/proj kernels:

```text
nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT
  2846 us/iter, 122 calls/iter

nvjet_sm100_ootst_128x128_256x7_4x1_2cta_v_bx_Avec16UE4M3_Bvec16UE4M3_
  2757 us/iter, 177 calls/iter

cutlass3x_sm100_bstensorop_s256x128x64gemm_block_scaled_ue4m3xf4_ue4m3
  1340 us/iter, 64 calls/iter

kernel_cutlass_kernel_tensorrt_llm_torchcute_dsl_kernelsblackwelldense
  852 us/iter, 122 calls/iter
```

Fused sigmoid/quant proof:

```text
dense_family_sigmoid profile:
  extFusedSigmoidMulQuantizeKernel present
  _sigmoid_mul_kernel absent

prior dense_family_combo profile:
  extFusedSigmoidMulQuantizeKernel absent
  _sigmoid_mul_kernel present
```

Delta versus dense_family_combo profile:

```text
Dense/proj GEMM   8937 -> 8930 us/iter, unchanged launch count
Elementwise/copy  1471 -> 1342 us/iter, -129 us/iter
Quant/dequant     1070 -> 1238 us/iter, +168 us/iter
```

Interpretation:

- The sigmoid/quant boundary fusion is mechanically active and removes the old
  standalone `_sigmoid_mul_kernel`.
- It is not an end-to-end C32 win because the quant/dequant bucket gets worse
  by about the same amount the elementwise bucket improves.
- Dense/proj GEMM remains the right target. It is still about 38% of GPU kernel
  time, and the largest single family is the BF16 MLA output-gate projection.

Full-gate A/B:

```text
arm: dense_family_sigmoid_gate_full_mega_cf
delta versus dense_family_sigmoid_mega_cf:
  TRTLLM_OPTRT_MLA_GATE_TORCH_MM=full
  TRTLLM_OPTRT_MLA_GATE_CUTEDSL_BF16_LBATCH unset
  TRTLLM_OPTRT_MLA_GATE_DSV3_FUSED_A unset

results/300_dense_family_sigmoid_gate_full_handshake.txt:
  C1 OSL32 ok=8/8, TTFT_p50=686 ms, user_tok_s_p50=24.39

results/301_dense_family_sigmoid_gate_full_c32_512_r1.txt:
  C32 OSL512 ok=64/64, TTFT_p50=1283 ms, user_tok_s_p50=50.67, agg_out_tok_s=1399.5

results/302_dense_family_sigmoid_gate_full_c32_512_r2.txt:
  C32 OSL512 ok=64/64, TTFT_p50=1577 ms, user_tok_s_p50=49.93, agg_out_tok_s=1380.4
```

Interpretation:

- Collapsing the two half-N BF16 gate GEMMs into one full cuBLASLt GEMM is not
  viable in the current stack.
- The likely reason is that the split side-stream gate work is buying useful
  overlap; reducing launch count alone loses more overlap than it saves.
- Do not promote full-gate mode. The next gate attempt needs to beat the
  half-GEMM per-call floor while preserving overlap, not merely collapse the
  launch structure.

Restore after full-gate A/B:

```text
arm:   dense_family_sigmoid_mega_cf
file:  results/303_restore_dense_family_sigmoid_after_gate_full.txt
state: decode=1/1, prefill=1/1, frontend=1/1, zero decode restarts
```

## 2026-06-19 - Gate cublasLt wrapper check and split-K combo miss

BF16 gate cublasLt wrapper microbench:

```text
script: .bench_runs_claude/megakernel/bf16_gate_cublasmm_compare.py
file:   .bench_runs_claude/results/304_bf16_gate_cublasmm_compare.txt
shape:  gate two-half BF16, X[M,7168] @ W_half[8192,7168].T
```

Results:

```text
M=1   torch_two=48.39 us  cublas_two=51.82 us  ratio=1.071
M=4   torch_two=49.38 us  cublas_two=56.34 us  ratio=1.141
M=8   torch_two=47.45 us  cublas_two=54.96 us  ratio=1.158
M=16  torch_two=47.95 us  cublas_two=54.55 us  ratio=1.138
M=32  torch_two=47.96 us  cublas_two=56.34 us  ratio=1.175
M=64  torch_two=50.00 us  cublas_two=57.82 us  ratio=1.156
```

Interpretation:

- TRT-LLM's public `trtllm::cublas_mm` wrapper is slower than `torch.mm` for
  the exact gate half/two-half shapes, so it is not a useful production hook.
- The BF16 gate kernel still needs either a better cublasLt tactic exposure or
  a new split-K/low-occupancy-aware half-GEMM kernel. A plain wrapper swap will
  not get us there.

Dense-family+sigmoid plus o_proj split-K combo:

```text
arm:   dense_family_sigmoid_splitk_o_atomic_mega_cf
image: localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-dense-family-sigmoid-20260619T045508Z
env:   dense_family_sigmoid_mega_cf plus
       TRTLLM_NVFP4_GEMM_SPLITK_O_PROJ=1
       TRTLLM_NVFP4_GEMM_SPLITK_ATOMIC=1
       TRTLLM_NVFP4_GEMM_SPLITK_MIN_M=16
       TRTLLM_NVFP4_GEMM_SPLITK_MAX_M=64
       TRTLLM_NVFP4_GEMM_SPLITK_N=7168
       TRTLLM_NVFP4_GEMM_SPLITK_K=16384
       TRTLLM_NVFP4_GEMM_SPLITK_SPLIT=2
       TRTLLM_NVFP4_GEMM_SPLITK_DEBUG=0
```

Results:

```text
results/306_dense_family_sigmoid_splitk_o_atomic_handshake.txt
  C1 OSL32 ok=8/8, TTFT_p50=702 ms, user_tok_s_p50=24.84

results/307_dense_family_sigmoid_splitk_o_atomic_c32_512_r1.txt
  C32 OSL512 ok=64/64, TTFT_p50=2203 ms,
  user_tok_s_p50=50.30, agg_out_tok_s=1314.2
```

Interpretation:

- The guarded split-K o_proj path is not additive with the current dense-family
  stack. Even with tiny-shape guarding and debug logging disabled, C32 aggregate
  drops far below both BEST_CONFIG and dense_family_sigmoid_mega_cf.
- Do not combine o_proj split-K into the current best arm. The isolated o_proj
  microbench signal is dominated in serving by extra contention/latency.

Restore after split-K combo:

```text
arm:   dense_family_sigmoid_mega_cf
file:  results/308_restore_dense_family_sigmoid_after_splitk_combo.txt
state: decode=1/1, prefill=1/1, frontend=1/1, zero decode restarts
```

BF16 gate cublasLt heuristic sweep:

```text
script: .bench_runs_claude/megakernel/bf16_gate_cublaslt_heuristic_sweep.py
file:   .bench_runs_claude/results/310_bf16_gate_cublaslt_heuristic_m32_m64_retry.txt
shape:  X[M,7168] @ W_half[8192,7168].T, BF16 -> BF16
```

Best returned cublasLt tactics:

```text
M=32:
  torch_half_us=19.458
  default cublasLt: 19.314 us, ratio=0.993
  best explicit heuristic:
    algo=66 tile=13 stages=35 splitK=1 reduction=0 custom=1 cga=3
    19.336 us, ratio=0.994
  split-K candidates:
    splitK=2 best 22.493 us, ratio=1.156
    splitK=3 best 24.188 us, ratio=1.243
    splitK=4 best 24.797 us, ratio=1.274

M=64:
  torch_half_us=19.761
  best explicit heuristic:
    algo=66 tile=15 stages=35 splitK=1 reduction=0 custom=1 cga=3
    19.627 us, ratio=0.993
  default cublasLt: 19.774 us, ratio=1.001
  split-K candidates:
    splitK=2 best 24.169 us, ratio=1.223
    splitK=3 best 24.766 us, ratio=1.253
    splitK=4 best 24.708 us, ratio=1.250
```

Interpretation:

- cublasLt already exposes the sensible no-split tactic for the gate half
  shape; adding a BF16 LUT entry would at best reproduce default behavior.
- cublasLt split-K increases parallelism but loses badly to reduction/workspace
  overhead for this shape.
- The remaining gate opportunity requires a custom kernel that improves the
  half-GEMM per-call floor while preserving the current two-half overlap. The
  generic CuTe persistent runner, TRT-LLM cublas_mm wrapper, full-gate collapse,
  DSV3 low-latency kernel, and cublasLt split-K tactics have all failed the
  relevant M32/M64 evidence.

## 2026-06-19 - NVFP4 split-K shape-table production test

Code change:

- Generalized the guarded split-K NVFP4 projection hook in
  `tensorrt_llm/_torch/custom_ops/torch_custom_ops.py`.
- Backward-compatible narrow o_proj env still works:
  `TRTLLM_NVFP4_GEMM_SPLITK_O_PROJ=1`.
- New explicit allow-list:
  `TRTLLM_NVFP4_GEMM_SPLITK_SHAPES=N,K,split[,atomic];...`
- Added M bounds:
  `TRTLLM_NVFP4_GEMM_SPLITK_MIN_M`, `TRTLLM_NVFP4_GEMM_SPLITK_MAX_M`.
- The selector remains default-off and still respects explicit single-backend
  routing unless `TRTLLM_NVFP4_GEMM_SPLITK_OVERRIDE_EXPLICIT=1`.

Deploy arm:

```text
arm:   splitk_proj_family_atomic_mega_cf
image: localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-splitk-shape-table-20260619T032713Z
env:
  TRTLLM_OPTRT_MOE_MEGAKERNEL=1
  TRTLLM_FORCE_COMM_METHOD=NVLINK_TWO_SIDED
  TRTLLM_NVFP4_GEMM_SPLITK_SHAPES=7168,16384,2,true;2112,7168,2,true
  TRTLLM_NVFP4_GEMM_SPLITK_ATOMIC=1
  TRTLLM_NVFP4_GEMM_SPLITK_MIN_M=0
  TRTLLM_NVFP4_GEMM_SPLITK_MAX_M=64
  TRTLLM_NVFP4_GEMM_SPLITK_OVERRIDE_EXPLICIT=1
```

Activation proof:

- Decode warmup selected split-K for the intended large-K shapes:
  - `N=2112,K=7168` at `M=1,2,3,4,5,6,7,8,48,64`
  - `N=7168,K=16384` at `M=1,2,3,4,5,6,7,8,48,64`
- It explicitly skipped the known-bad short-K or MLP shapes:
  - `N=24576,K=1536`
  - `N=8192,K=1536`
  - `N=36864,K=7168`
  - `N=7168,K=18432`
  - `N=7168,K=2048`

Serving result, OSL512:

```text
file: .bench_runs_claude/results/283_splitk_proj_family_atomic_osl512.txt

  C    TTFT_p50   user_tok/s   agg_out_tok/s
  1       490        42.45          44.5
  4       788        55.86         221.4
  8      1037        54.13         414.2
 16      1224        52.15         763.7
 32      1570        49.84        1374.7
 64      1568        45.37        2294.2

warm C32 repeat:
file: .bench_runs_claude/results/284_splitk_proj_family_atomic_c32_repeat.txt
  C=32 TTFT_p50=1504 ms, user_tok/s=50.05, agg_out_tok/s=1375.3
```

Comparison to original BEST_CONFIG C32/OSL512:

```text
BEST_CONFIG:     user_tok/s=51.07, agg_out_tok/s=1424.9, TTFT_p50=1396 ms
shape-table run: user_tok/s=49.84, agg_out_tok/s=1374.7, TTFT_p50=1570 ms
warm repeat:     user_tok/s=50.05, agg_out_tok/s=1375.3, TTFT_p50=1504 ms
```

Interpretation:

- The generalized selector works in the production path and routes exactly the
  intended projection shapes.
- It does not beat BEST_CONFIG at the C32 operating point. It is about
  1.2 user tok/s and about 50 aggregate tok/s below the original frozen best.
- Adding the `kv_a`/`qkv_a`-family split-K route did not rescue the isolated
  o_proj split-K win. The extra route likely adds atomic/compile/capture
  overhead without enough exposed critical-path benefit.
- Do not promote split-K projection routing as a serving lever in this form.
  If revisited, the next version should either precompile/cache the CuTe
  specializations and use a stricter min-M guard, or move to a fused
  BF16-input-to-NVFP4-projection boundary instead of split-King standalone
  projection GEMMs.

Restored known-good deployment after the test:

```text
arm:   vbfuse_nvfp4_mega_cf
image: localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-vbfuse-nvfp4-20260615T043037Z
state: decode and prefill both 1/1 ready, zero restarts
```

## 2026-06-19 - Dense-family plus fused sigmoid-quant production check

Arm:

```text
arm:   dense_family_sigmoid_mega_cf
image: localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-dense-family-sigmoid-20260619T045508Z
digest: sha256:96fb2f2c547fb5dedcfb223fc6b67e1374673e3a5b1e5a8e17464a92daf853bb
env:
  TRTLLM_OPTRT_MOE_MEGAKERNEL=1
  TRTLLM_FORCE_COMM_METHOD=NVLINK_TWO_SIDED
  TRTLLM_OPTRT_FUSED_SIGMOID_MUL_QUANT_NVFP4=1
  TRTLLM_INDEXER_FUSE_QB_WQB=1
  TRTLLM_INDEXER_FUSE_QB_WQB_STATIC=1
  TRTLLM_INDEXER_FUSE_QB_WQB_VARIABLE_N=1
  TRTLLM_INDEXER_FUSE_QB_WQB_POST_SCALE=0
  TRTLLM_INDEXER_FUSE_QB_WQB_ALIAS_ORIGINALS=1
  TRTLLM_INDEXER_FUSE_KVA_WKWP=1
  TRTLLM_INDEXER_FUSE_KVA_WKWP_STATIC=1
  TRTLLM_INDEXER_FUSE_KVA_WKWP_BF16_DSA=1
  TRTLLM_INDEXER_FUSE_KVA_WKWP_ALIAS_ORIGINALS=1
  use_low_precision_moe_combine=false
```

Production readiness:

- Decode and prefill both reached `1/1`, zero restarts.
- Startup was slower than the previous dense-family image, roughly 7.5 minutes
  before both pods were ready.
- Pod env confirmed fused sigmoid-quant and both dense-family hooks active;
  split-K and q_b CuTeDSL routes were absent.

Handshake:

```text
file: .bench_runs_claude/results/292_dense_family_sigmoid_handshake.txt
C=1, OSL=32, ok=8/8, TTFT_p50=761 ms, user_tok/s=25.01
```

Focused C32/OSL512 repeats:

```text
file: .bench_runs_claude/results/293_dense_family_sigmoid_c32_512_r1.txt
C=32, TTFT_p50=1291 ms, user_tok/s=51.76, agg_out_tok/s=1420.6

file: .bench_runs_claude/results/294_dense_family_sigmoid_c32_512_r2.txt
C=32, TTFT_p50=1432 ms, user_tok/s=52.56, agg_out_tok/s=1419.1
```

Full hot OSL512 sweep:

```text
file: .bench_runs_claude/results/295_dense_family_sigmoid_osl512_full_hot.txt

  C    TTFT_p50   user_tok/s   agg_out_tok/s
  1       474        44.44          46.4
  4       798        58.13         229.7
  8      1108        55.62         422.6
 16      1450        54.28         780.2
 32      1620        51.88        1405.0
 64      1515        46.54        2359.4
```

Comparison to original frozen BEST_CONFIG:

```text
BEST_CONFIG C32:       TTFT_p50=1396 ms, user_tok/s=51.07, agg_out_tok/s=1424.9
focused r1 C32:        TTFT_p50=1291 ms, user_tok/s=51.76, agg_out_tok/s=1420.6
focused r2 C32:        TTFT_p50=1432 ms, user_tok/s=52.56, agg_out_tok/s=1419.1
full-sweep C32:        TTFT_p50=1620 ms, user_tok/s=51.88, agg_out_tok/s=1405.0

focused r1 delta:      TTFT -7.5%, user tok/s +1.4%, aggregate -0.3%
focused r2 delta:      TTFT +2.6%, user tok/s +2.9%, aggregate -0.4%
full-sweep C32 delta:  TTFT +16.0%, user tok/s +1.6%, aggregate -1.4%
```

Full-sweep aggregate delta versus original frozen BEST_CONFIG:

```text
  C     dense+sigmoid agg delta
  1     +2.4%
  4     +3.4%
  8     +2.1%
 16     -0.4%
 32     -1.4%
 64     +5.8%
```

Comparison to dense-family combo without sigmoid:

```text
prior dense-family C32 full sweep:      TTFT_p50=1657 ms, user_tok/s=51.92, agg_out_tok/s=1395.6
dense-family+sigmoid C32 full sweep:    TTFT_p50=1620 ms, user_tok/s=51.88, agg_out_tok/s=1405.0
delta versus prior combo:               TTFT -2.2%, user tok/s -0.1%, aggregate +0.7%

prior dense-family C64 full sweep:      agg_out_tok/s=2321.4
dense-family+sigmoid C64 full sweep:    agg_out_tok/s=2359.4
delta versus prior combo:               aggregate +1.6%
```

Interpretation:

- This arm is much healthier than the dense-family combo on the focused C32
  repeat: it recovers TTFT and keeps the steady per-user decode gain.
- It still does not beat the original frozen BEST_CONFIG at the primary C32
  aggregate acceptance point. Best focused C32 aggregate is 1420.6 tok/s,
  about 0.3% below BEST; the full-sweep C32 row is about 1.4% below BEST.
- The fused sigmoid-quant boundary helps enough to make the dense-family combo
  less bad at C32 and better at C64, but it is not a promotion candidate
  without a profile showing a real kernel-bound reduction and without a C32
  aggregate win outside run-to-run noise.
- Next proof needed: capture a 128-step C32 profile for this arm and confirm
  whether `extFusedSigmoidMulQuantizeKernel` replaces `_sigmoid_mul_kernel`,
  whether quant/dequant launch count drops, and whether the remaining C32
  aggregate miss is admission/TTFT or decode-kernel critical path.

## 2026-06-19 - Current comparison to original BEST_CONFIG

Live deployment after the q_b Linear CuTeDSL experiment is back on the frozen
BEST_CONFIG image on both workers:

```text
decode:  localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-vbfuse-nvfp4-20260615T043037Z
prefill: localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-vbfuse-nvfp4-20260615T043037Z
decode env: TRTLLM_OPTRT_MOE_MEGAKERNEL=1, TRTLLM_FORCE_COMM_METHOD=NVLINK_TWO_SIDED
q_b env: absent
```

OSL512 comparison:

```text
original BEST C16: 53.24 user,  783.2 agg
q_b Linear   C16: 52.56 user,  780.0 agg

original BEST C32: 51.07 user, 1424.9 agg
q_b Linear   C32: 50.46 / 50.83 / 50.40 user,
                   1409.7 / 1379.6 / 1401.8 agg
restored BEST C32: 50.08 user, 1386.6 agg

original BEST C64: 44.77 user, 2230.4 agg
q_b Linear   C64: 44.70 user, 2328.8 agg
restored BEST C64: 44.85 user, 2329.7 agg
```

Verdict:

- q_b Linear CuTeDSL is a valid serving path now that both workers use the same
  carrier image, but it is not a proven improvement over original BEST_CONFIG
  at the C32/OSL512 operating point.
- The standalone q_b slice is too narrow. It reduces one projection family, but
  the profile remains dominated by the 122-call BF16 MLA gate family and the
  193-call NVFP4 projection family.
- Keep q_b Linear CuTeDSL as a measured building block only. The next
  dense/proj branch needs to remove a larger family or combine multiple
  projection reductions without collapsing the overlap that the current serving
  path relies on.

## 2026-06-19 - q_b CuTe tactic recenter after BEST_CONFIG comparison

Live state check:

```text
decode image:  optrt-529374445d-vbfuse-nvfp4-20260615T043037Z
prefill image: optrt-529374445d-vbfuse-nvfp4-20260615T043037Z
decode env:    TRTLLM_OPTRT_MOE_MEGAKERNEL=1
decode env:    TRTLLM_FORCE_COMM_METHOD=NVLINK_TWO_SIDED
```

Comparison against original BEST_CONFIG:

- No experiment is currently proven better than BEST_CONFIG.
- q_b/wq_b variable-N is parity/noise at C32 OSL512:
  `50.61 tok/s/user, 1417.4 agg` and repeat `50.51, 1399.5`, versus
  BEST_CONFIG `51.07, 1424.9` in `sweep_confirmed.txt` and `50.08, 1386.6`
  in the restored repeat.
- KVA/WK/WP BF16 DSA reduces dense/proj work in the profile, but regresses
  C32 OSL128 aggregate versus BEST_CONFIG controls.
- Split-K atomic o_proj is a real isolated kernel win, but latest C32 serving
  rows are control-band/noise and not a standalone throughput lever.

q_b CuTeDSL tactic rerun:

```text
container: optrt-bench-claude
runtime:   installed package from /tmp with PYTHONPATH=
target:    q_b, M=32, N=24576, K=1536

cuBLASLt baseline:                         14.333 us
CuTe 128x64 cluster 4x1 swap_ab=true:      13.705 us
CuTe 128x64 cluster 2x1 swap_ab=true:      13.642 us
CuTe 128x128 cluster 4x1 swap_ab=true:     14.829 us
CuTe 128x128 cluster 1x4 swap_ab=false:    15.358 us
```

This resolves the apparent conflict in the prior notes:

- The faster q_b CuTe tactic is real, but it was not included in the later
  reduced tactic subset that made generic CuTe look like a dead end.
- The production blocker is not "no faster tactic exists"; it is that the
  Python/CuTe dispatch carrier caused slow startup and KV-transfer instability.

Code action:

- Made `_try_nvfp4_gemm_qb_cutedsl()` tactic-configurable via env.
- Default q_b tactic is now the best reproduced candidate:
  `tile=(128,64), cluster=(2,1), swap_ab=true, prefetch=false`.
- Updated `qb_cutedsl_mega_cf` to set the exact tactic env explicitly.

Next requirement before another serving deploy:

- Use this q_b route only in a carrier that avoids request-time CuTe compile
  behavior and does not fire on tiny CUDA graph warmup shapes.
- The acceptance proof is a C32/128-step profile where the
  `nvjet_sm100_ootst_256x128_256x4...` q_b slice shrinks without introducing
  KV-transfer stalls or replacing BEST_CONFIG with a slow startup profile.

Hook-level follow-up:

- Staged the env-configurable dispatcher into the `optrt-bench-claude`
  benchmark container only, timed `nvfp4_gemm()` directly, then restored the
  container package backup.
- The q_b route did fire; logged route key:
  `((128,64),(2,1),swap_ab=true,prefetch=false)`.

Side-by-side graph replay on identical M=32 tensors:

```text
cuBLASLt through nvfp4_gemm: 14.756 us
direct CuTe runner:          14.049 us  (1.050x)
q_b nvfp4_gemm hook:         14.554 us  (1.014x)
```

Hook sweep with the same env-configured tactic:

```text
M=16 base=14.783 us hook=14.713 us speedup=1.005
M=24 base=15.248 us hook=15.189 us speedup=1.004
M=32 base=15.336 us hook=15.432 us speedup=0.994
M=64 base=14.998 us hook=15.115 us speedup=0.992
```

Decision:

- The faster direct q_b CuTe tactic does not survive the current
  `nvfp4_gemm` pre-dispatch carrier strongly enough to justify another serving
  rollout.
- Keep the tactic configurability because it makes future probes reproducible,
  but do not treat `TRTLLM_NVFP4_GEMM_QB_CUTEDSL=1` as the production path.
- The next q_b implementation must remove the carrier overhead or become a
  first-class/precompiled op path. Otherwise this branch is too small to move
  the 193-call NVFP4 family in the C32 profile.

Module-level q_b Linear bypass follow-up:

- Staged only `tensorrt_llm/_torch/modules/linear.py` into the
  `optrt-bench-claude` installed package, tested through
  `NVFP4LinearMethod.apply()`, then restored the container backup.
- This bypass calls the same CuTe runner before the generic
  `torch.ops.trtllm.nvfp4_gemm` dispatch, avoiding the hook-level overhead.

```text
single-shape check, M=32:
  base=16.819 us module_cutedsl=16.064 us speedup=1.047
  cosine=0.99999595 max_abs=0.0078125

M sweep, same q_b shape N=24576,K=1536:
  M=16 base=16.124 us module_cutedsl=16.612 us speedup=0.971
  M=24 base=16.627 us module_cutedsl=16.769 us speedup=0.992
  M=32 base=16.664 us module_cutedsl=15.878 us speedup=1.050
  M=64 base=16.387 us module_cutedsl=15.923 us speedup=1.029
```

Decision:

- The module-level route is better than the generic hook, but only for M32+.
- Default and deployment gating should therefore be `32<=M<=64`; M16/M24 stay
  on the original BEST_CONFIG path.
- This is still not a proven BEST_CONFIG improvement until a clean carrier image
  shows C32/128 profile movement and no startup or KV-transfer regressions.

CUDA-event retest through `NVFP4LinearMethod.apply()`:

- Re-staged only `linear.py` into `optrt-bench-claude`, used CUDA events with
  80 warmup and 240 measured iterations, then restored the container package.
- Env gate was `TRTLLM_NVFP4_LINEAR_QB_CUTEDSL_MIN_M=32`, so M16/M24 are
  fallback controls; `max_abs=0` confirms they did not take the CuTe route.

```text
M=16 base_med=61.920 us cutedsl_med=69.392 us speedup=0.892 max_abs=0
M=24 base_med=74.176 us cutedsl_med=68.592 us speedup=1.081 max_abs=0
M=32 base_med=61.824 us cutedsl_med=50.720 us speedup=1.219 max_abs=0.0078125
M=64 base_med=62.560 us cutedsl_med=47.072 us speedup=1.329 max_abs=0.0078125
```

Carrier action:

- Added minimal image overlay
  `.bench_runs_claude/overlay_qb_linear_cutedsl/Dockerfile`.
- It copies only `tensorrt_llm/_torch/modules/linear.py` on top of the frozen
  BEST_CONFIG image, so the next serving probe isolates the q_b Linear bypass.
- Updated `QB_CUTEDSL_IMG` to
  `localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-qb-linear-cutedsl-20260619T021628Z`.

Serving probe:

```text
image: localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-qb-linear-cutedsl-20260619T021628Z
arm:   qb_cutedsl_mega_cf
env:   MEGAKERNEL=1, NVLINK_TWO_SIDED, combine=false,
       TRTLLM_NVFP4_LINEAR_QB_CUTEDSL=1, M gate 32..64
```

Observed behavior:

- Image build and local-registry push succeeded; image digest
  `sha256:c91ec3a0f7f7f1532d23054fefe5002baaad47b8e6f433de0800223f97448e4c`.
- Carrier contains the new `TRTLLM_NVFP4_LINEAR_QB_CUTEDSL_MIN_M` helper and
  `linear.py` compiles inside the image.
- Decode reached Ready after 459 s with zero restarts.
- After the standard 240 s kv-router settle, the C=1 / OSL32 handshake probe
  hung for multiple minutes and produced no valid result row:
  `.bench_runs_claude/results/277_qb_linear_cutedsl_handshake.txt`.
- During the hung probe the prefill worker logged repeated
  `num_fitting_reqs=0 and fitting_disagg_gen_init_requests is empty, may not
  have enough kvCache`, then both workers went quiet with GPUs idle.

Verdict:

- Initial probe rejected the q_b Linear CuTeDSL carrier as configured, because
  the clean carrier broke the disagg handshake/scheduling path before C32
  throughput could be measured.
- Post-mortem found `qb_cutedsl_mega_cf` was missing from `deploy_arm.py`'s
  "roll both workers to the same image" list. BEST_CONFIG explicitly requires
  matching decode/prefill images for the NIXL KV-transfer handshake, so the
  handshake hang is plausibly image mismatch rather than a q_b kernel fault.
- Fixed the deployment arm to roll both prefill and decode to the q_b carrier
  for the next probe.

Restored known-good deployment:

```text
arm:   vbfuse_nvfp4_mega_cf
image: localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-vbfuse-nvfp4-20260615T043037Z
state: decode Ready after 478 s, zero restarts
env:   MEGAKERNEL=1, NVLINK_TWO_SIDED, QB_LINEAR_CUTEDSL unset
```

Matched-image retest after adding `qb_cutedsl_mega_cf` to the both-worker
image list:

```text
decode:  localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-qb-linear-cutedsl-20260619T021628Z
prefill: localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-qb-linear-cutedsl-20260619T021628Z
state:   both Ready, zero restarts
decode env: MEGAKERNEL=1, NVLINK_TWO_SIDED, TRTLLM_NVFP4_LINEAR_QB_CUTEDSL=1
```

Handshake now passes:

```text
file: .bench_runs_claude/results/278_qb_linear_cutedsl_matched_handshake.txt
C=1 ok=8/8 ISL=4161 TTFT_p50=761 ms TTFT_p95=22216 ms
user_tok_s_p50=24.32 agg_out_tok_s=7.2
```

Serving rows:

```text
file: .bench_runs_claude/results/279_qb_linear_cutedsl_matched_c32_osl128.txt
C=32 OSL128 ok=64/64 TTFT_p50=1382 ms TTFT_p95=3111 ms
user_tok_s_p50=39.21 agg_out_tok_s=840.5

file: .bench_runs_claude/results/280_qb_linear_cutedsl_matched_c32_osl512.txt
C=32 OSL512 ok=64/64 TTFT_p50=1395 ms TTFT_p95=3141 ms
user_tok_s_p50=50.46 agg_out_tok_s=1409.7

file: .bench_runs_claude/results/281_qb_linear_cutedsl_matched_c32_osl512_r2.txt
C=32 OSL512 ok=64/64 TTFT_p50=1690 ms TTFT_p95=3263 ms
user_tok_s_p50=50.83 agg_out_tok_s=1379.6

file: .bench_runs_claude/results/282_qb_linear_cutedsl_matched_c16_24_32_64_osl512.txt
C=16 user_tok_s_p50=52.56 agg_out_tok_s=780.0
C=24 user_tok_s_p50=51.43 agg_out_tok_s=1105.2
C=32 user_tok_s_p50=50.40 agg_out_tok_s=1401.8
C=64 user_tok_s_p50=44.70 agg_out_tok_s=2328.8
```

Comparison to original BEST_CONFIG OSL512:

```text
BEST C16: 53.24 user,  783.2 agg
q_b  C16: 52.56 user,  780.0 agg

BEST C32: 51.07 user, 1424.9 agg
q_b  C32: 50.46/50.83/50.40 user, 1409.7/1379.6/1401.8 agg

BEST C64: 44.77 user, 2230.4 agg
q_b  C64: 44.70 user, 2328.8 agg
restored BEST C64 repeat was 44.85 user, 2329.7 agg, so this is control-band.
```

Verdict after image-match fix:

- q_b Linear CuTeDSL is now a valid serving path, but it is not a proven
  improvement over original BEST_CONFIG at the C32/OSL512 operating point.
- The C32/OSL128 row is slightly above prior OSL128 controls, but the real
  OSL512 rows remain below original BEST_CONFIG and within restored-control
  noise.
- Do not promote q_b Linear CuTeDSL as a standalone config. Keep the code path
  as a measured building block, but the next dense/proj attempt needs to remove
  a larger launch/time bucket or combine multiple projection reductions without
  hurting NIXL/TTFT.

Restored known-good deployment again after the matched test:

```text
arm:   vbfuse_nvfp4_mega_cf
image: localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-vbfuse-nvfp4-20260615T043037Z
state: decode Ready after 456 s, zero restarts
env:   MEGAKERNEL=1, NVLINK_TWO_SIDED, QB_LINEAR_CUTEDSL unset
```

Interpretation:

- The isolated o_proj split-K atomic win does not translate into a clear C32
  serving win. User tok/s is consistently below the clean BEST_CONFIG control,
  while aggregate output tok/s is within run-to-run noise.
- Startup cost is worse because tiny warmup M values go through the split-K
  path. This should be guarded before any future serving test.
- Do not treat o_proj split-K as a standalone throughput lever. If revisited,
  it should be part of a combined precompiled dense/proj path, with a min-M
  guard and no debug logging in the hot path.

## 2026-06-19 - KVA/WK/WP BF16 DSA production profile

Arm:

```text
kva_wkwp_fused_mega_cf
image: localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-kva-wkwp-bf16dsa-scaleguard-20260619T010519Z
```

Delta versus BEST_CONFIG:

- Keeps megakernel, `NVLINK_TWO_SIDED`, and
  `use_low_precision_moe_combine=false`.
- Enables same-input padded fusion of `kv_a_proj_with_mqa` plus DSA
  `wk/weights_proj`:

```text
TRTLLM_INDEXER_FUSE_KVA_WKWP=1
TRTLLM_INDEXER_FUSE_KVA_WKWP_STATIC=1
TRTLLM_INDEXER_FUSE_KVA_WKWP_BF16_DSA=1
TRTLLM_INDEXER_FUSE_KVA_WKWP_ALIAS_ORIGINALS=1
TRTLLM_INDEXER_FUSE_KVA_WKWP_DEBUG=0
```

Production activation proof:

- Initial debug deployment showed the helper firing in decode warmup for
  `M=8192`, `64`, and down to `M=3`.
- `wkwp_out_scale` initially became `inf`; the scaleguard image fixed this by
  falling back to `kv_a.alpha * kv_a.input_scale` when the raw scale ratio was
  non-finite.
- Post-fix logs showed finite scales and `cast_wkwp_to_fp32=True`; the
  measurement deployment disabled debug logging before benchmarking.

Serving result:

```text
handshake, C=1 OSL=32:
  ok=8/8 ISL=4161 TTFT_p50=690 ms TTFT_p95=739 ms
  user_tok_s_p50=24.54 agg_out_tok_s=23.9

C=32 OSL=128:
  ok=64/64 ISL=2055 TTFT_p50=1706 ms TTFT_p95=3562 ms
  user_tok_s_p50=39.15 agg_out_tok_s=766.5 wall=10.7
  file: .bench_runs_claude/results/kva_wkwp_bf16dsa_nodebug_c32_128.txt
```

Comparable BEST_CONFIG OSL128 controls:

```text
.bench_runs_claude/results/68_best_restored_control_osl128_c16_24_32.txt
  C=32 ok=64/64 user_tok_s_p50=38.45 agg_out_tok_s=818.2

.bench_runs_claude/results/147_restored_best_after_fsss_c32_rerun.txt
  C=32 ok=64/64 user_tok_s_p50=38.29 agg_out_tok_s=835.3
```

Kineto:

```text
trace: /var/lib/optrt-cache/nsys/kva_wkwp_bf16dsa_c32_128-rank-*.json
analysis: .bench_runs_claude/profiles/kineto_analysis_kva_wkwp_bf16dsa_c32_128_20260619T0133/kva_wkwp_bf16dsa_c32_128_20260619T0133_exhaustive.txt
window: TLLM_PROFILE_START_STOP=3000-3128
```

Aggregate profile delta versus the previous C32/128 BEST_CONFIG profile:

```text
Dense/proj GEMM:
  9102 us/iter, 640 launches/iter
  -> 8777 us/iter, 624 launches/iter

Elementwise/copy:
  1552 us/iter, 936.6 launches/iter
  -> 1358 us/iter, 693 launches/iter

Quant/dequant:
  1133 us/iter, 392 launches/iter
  -> 1054 us/iter, 376 launches/iter

Mean GPU idle:
  15.2% -> 3.9%
```

Interpretation:

- The KVA/WK/WP helper finally fired in production and did remove exactly the
  expected 16 dense/proj launches per decode iteration.
- Dense/proj time moved in the right direction, but only by about 325 us/iter
  (-3.6%). That is too small to carry E2E throughput.
- The profile classification shifts MoE expert work into the generic GEMM
  bucket, so do not read the "MoE expert GEMM" drop as a real MoE fix.
- E2E aggregate throughput regressed versus OSL128 BEST_CONFIG controls
  despite the small dense/proj profile win. This arm should not be promoted as
  a standalone change.
- Useful follow-up is a combined, precompiled dense/proj path that targets more
  than one projection family in the same serving run, plus a fresh profile to
  confirm the launch-count reductions add instead of just moving overhead.

Restored known-good deployment after the test:

```text
arm:   vbfuse_nvfp4_mega_cf
image: localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-vbfuse-nvfp4-20260615T043037Z
state: decode and prefill both 1/1 ready, zero restarts
```

## 2026-06-19 - Current status versus original BEST_CONFIG

Primary comparison point remains C32/OSL512:

```text
Original BEST_CONFIG:
  TTFT_p50=1396 ms, user_tok_s_p50=51.07, agg_out_tok_s=1424.9

Best current dense_family_sigmoid_mega_cf focused rows:
  TTFT_p50=1291/1432 ms, user_tok_s_p50=51.76/52.56,
  agg_out_tok_s=1420.6/1419.1

Latest o_proj Linear CuTeDSL A/B:
  TTFT_p50=1800/1405/1537 ms, user_tok_s_p50=51.10/50.92/51.46,
  agg_out_tok_s=1390.1/1396.2/1409.3
```

Bottom line:

- Current best candidate is still dense_family_sigmoid_mega_cf. It improves
  per-user decode versus original BEST_CONFIG but is still slightly behind on
  the C32 aggregate acceptance metric.
- Both o_proj CuTe variants failed as standalone serving changes. The
  first-class Linear route proved the local kernel can be faster, but the
  request-level aggregate still missed by about 1.1% versus original BEST.
- Cluster is restored to dense_family_sigmoid_mega_cf after the failed
  o_proj Linear test; decode and prefill are both 1/1 Ready with zero restarts.

## 2026-06-19 - DWDP route check on current best

Question:

- Is the current production decode path actually using the DWDP multi-B weight
  view, making a DWDP-aware MoE V2 megakernel the next implementation target?

Instrumentation:

```text
image: localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-dense-family-sigmoid-dwdp-debug-20260619T083344Z
arm:   dense_family_sigmoid_dwdp_debug_mega_cf
env:   dense_family_sigmoid_mega_cf + TRTLLM_OPTRT_MOE_DWDP_DEBUG=1

files:
  .bench_runs_claude/results/347_dense_family_sigmoid_dwdp_debug_handshake.txt
  .bench_runs_claude/results/348_decode_dwdp_debug_full.log
  .bench_runs_claude/results/352_dense_family_sigmoid_dwdp_debug_prints_handshake.txt
  .bench_runs_claude/results/353_decode_dwdp_debug_prints_full.log
  .bench_runs_claude/results/354_restore_dense_family_sigmoid_after_dwdp_debug.txt
```

Result:

```text
route markers: 232
impl markers: 0
is_dwdp=True: 0
is_dwdp=False: 232
multi_b op warnings: 4
```

Representative route marker:

```text
TRTLLM_DWDP_DEBUG_ROUTE
layer_idx=3
rank=1
is_dwdp=False
weight_buffers=1
weight_view_esp=32
weight_view_slot_start=32
top_k=8
x=shape=(4, 3584)
x_sf=shape=(4, 448)
token_selected_experts=shape=(4, 8)
token_final_scales=shape=(4, 8)
```

Interpretation:

- The current best decode generation path is not using DWDP multi-B weight
  views. It uses one local weight buffer per rank with local EP slot offsets
  (`0/32/64/96`) and `expert_size_per_partition=32`.
- The startup warning for
  `cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell_multi_b` was a
  false DWDP signal: the single-B custom op wrapper internally calls the
  multi-B op with a one-element list.
- Therefore a DWDP-specific MoE V2 implementation is not the immediate serving
  blocker for the current best config. The live MoE path is the non-DWDP
  WARPDECODE path with the existing phase-1 megakernel enabled.
- Next MoE work should target the non-DWDP phase-1/V2 path if we pursue MoE,
  but the measured profile still says dense/proj GEMM remains the larger
  bucket.

Restore:

```text
arm:   dense_family_sigmoid_mega_cf
image: localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-dense-family-sigmoid-20260619T045508Z
state: decode, prefill, and frontend are 1/1 Ready with zero restarts
```

## 2026-06-19 - Non-DWDP MoE V2 on current dense-family best

Question:

- Since the live decode route is non-DWDP, does the phase-3 persistent
  decode-MoE V2 path beat the original BEST_CONFIG C32/OSL512 target when
  layered on top of the current dense-family+sigmoid stack?

Arm:

```text
image: localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-dense-family-sigmoid-v2-20260619T085608Z
arm:   dense_family_sigmoid_v2_mega_cf
base:  dense_family_sigmoid_mega_cf
delta:
  TRTLLM_OPTRT_MOE_MEGAKERNEL_V2=1
  overlay fused_moe_cute_dsl.py
  overlay mega_persistent_moe.py
```

Files:

```text
.bench_runs_claude/overlay_dense_family_sigmoid_v2/Dockerfile
.bench_runs_claude/results/355_build_dense_family_sigmoid_v2.txt
.bench_runs_claude/results/356_push_dense_family_sigmoid_v2.txt
.bench_runs_claude/results/357_deploy_dense_family_sigmoid_v2.txt
.bench_runs_claude/results/358_wait_dense_family_sigmoid_v2.txt
.bench_runs_claude/results/359_dense_family_sigmoid_v2_handshake.txt
.bench_runs_claude/results/360_dense_family_sigmoid_v2_c32_512_r1.txt
.bench_runs_claude/results/361_dense_family_sigmoid_v2_c32_512_r2.txt
.bench_runs_claude/results/362_dense_family_sigmoid_v2_c32_512_r3.txt
```

Activation proof from decode logs:

```text
CuteDslFusedMoE V2 gate: env='1' use_fused_finalize=True tile_size=128 enabled=True
CuteDslFusedMoE: phase-3 persistent decode-MoE megakernel ENABLED (TRTLLM_OPTRT_MOE_MEGAKERNEL_V2).
```

Handshake:

```text
C=8, ok=8/8, ISL=209, OSL=32
TTFT_p50=44650 ms, user_tok_s_p50=31.10, agg_out_tok_s=5.7
```

The handshake paid first-use V2 Cutlass/Torch compilation and graph-capture
cost. It is a liveness/activation check, not a throughput datapoint.

C32/ISL2055/OSL512:

```text
Original BEST_CONFIG:
  TTFT_p50=1396 ms, user_tok_s_p50=51.07, agg_out_tok_s=1424.9

Previous current best dense_family_sigmoid_mega_cf:
  r1: TTFT_p50=1291 ms, user_tok_s_p50=51.76, agg_out_tok_s=1420.6
  r2: TTFT_p50=1432 ms, user_tok_s_p50=52.56, agg_out_tok_s=1419.1

V2 arm:
  r1: TTFT_p50=1998 ms, user_tok_s_p50=52.71, agg_out_tok_s=1376.0
  r2: TTFT_p50=1389 ms, user_tok_s_p50=51.98, agg_out_tok_s=1431.5
  r3: TTFT_p50=1286 ms, user_tok_s_p50=52.45, agg_out_tok_s=1436.2
```

Broader warmed sweep:

```text
file: .bench_runs_claude/results/363_dense_family_sigmoid_v2_c16_24_32_64_512.txt

C16: TTFT_p50=1316 ms, user_tok_s_p50=54.46, agg_out_tok_s=791.9
C24: TTFT_p50=1272 ms, user_tok_s_p50=54.23, agg_out_tok_s=1157.4
C32: TTFT_p50=1319 ms, user_tok_s_p50=52.67, agg_out_tok_s=1443.4
C64: TTFT_p50=1667 ms, user_tok_s_p50=46.37, agg_out_tok_s=2299.6
```

Full warmed sweep against the original BEST_CONFIG table:

```text
file: .bench_runs_claude/results/364_dense_family_sigmoid_v2_full_warm_512.txt

C1:  TTFT_p50=502  ms, user_tok_s_p50=44.53, agg_out_tok_s=46.4
C4:  TTFT_p50=800  ms, user_tok_s_p50=58.63, agg_out_tok_s=230.2
C8:  TTFT_p50=1128 ms, user_tok_s_p50=56.67, agg_out_tok_s=430.7
C16: TTFT_p50=1263 ms, user_tok_s_p50=54.66, agg_out_tok_s=793.7
C32: TTFT_p50=1319 ms, user_tok_s_p50=52.31, agg_out_tok_s=1452.9
C64: TTFT_p50=1458 ms, user_tok_s_p50=46.58, agg_out_tok_s=2341.7
```

Interpretation:

- V2 is the first campaign arm here that beats the original frozen BEST_CONFIG
  C32 aggregate target after warmup: `1431.5` and `1436.2` versus `1424.9`.
- The broader warmed sweep strengthens that: `1443.4` aggregate at C32 versus
  original BEST_CONFIG `1424.9`.
- The full warmed sweep is the cleanest current comparison: V2 beats the
  original frozen table at every measured concurrency. The C32 operating point
  is `1452.9` versus original BEST_CONFIG `1424.9` (`+2.0%` aggregate).
- The best V2 repeat is also above the previous current-best dense-family rows
  by about `+1.1%` aggregate (`1436.2` versus `1420.6`).
- V2 also beats the frozen original table at C16 (`791.9` vs `783.2`) and C64
  (`2299.6` vs `2230.4`) in the warmed sweep. Note that the separate
  2026-06-17 restore re-confirm had a higher C64 row (`2329.7`), so C64 should
  not be claimed as a universal win without another matched A/B.
- TTFT is not worse in the stable repeat: best V2 `1286 ms` is slightly better
  than original BEST_CONFIG `1396 ms` and comparable to the best dense-family
  row `1291 ms`.
- r1 still carried shape warmup/capture cost despite the short handshake, so do
  not use r1 as the steady-state comparison.
- Current cluster is intentionally left on `dense_family_sigmoid_v2_mega_cf`
  for follow-up sweeps unless a later test requires restoring
  `dense_family_sigmoid_mega_cf`.

Profile capture:

```text
manifest: .bench_runs_claude/results/365_build_kineto_dense_family_sigmoid_v2.txt
apply:    .bench_runs_claude/results/366_apply_kineto_dense_family_sigmoid_v2.txt
drive:    .bench_runs_claude/results/368_dense_family_sigmoid_v2_profile_c32_drive_r1.txt
          .bench_runs_claude/results/362_dense_family_sigmoid_v2_profile_c32_drive_r2.txt
          .bench_runs_claude/results/363_dense_family_sigmoid_v2_profile_c32_drive_r3.txt
analysis: .bench_runs_claude/profiles/kineto_analysis_dense_family_sigmoid_v2_c32_512_20260619T0926/analysis.txt
```

Profile verdict at C32/OSL512:

- GPU-bound path remains dominant: aggregate mean GPU busy is about `85.9%`;
  host/runtime overhead is secondary.
- Dense/proj GEMM is still the largest bucket at about `8926 us/iter`,
  `38.5%` of GPU kernel time, with `608 launches/iter`.
- Next buckets are MoE a2a/comm (`3364 us/iter`, `14.5%`) and MoE expert GEMM
  (`3067 us/iter`, `13.2%`).
- V2 did not erase the dense GEMM wall; the end-to-end win appears to be a
  serving/overlap/steady-state improvement rather than a massive single-bucket
  collapse.

Post-profile restore:

```text
arm:   dense_family_sigmoid_v2_mega_cf
image: localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-dense-family-sigmoid-v2-20260619T085608Z
decode pod: topo-c1-dp2tp4-disagg-r20-0-decode-5d867, 1/1 Running, 0 restarts
prefill pod: topo-c1-dp2tp4-disagg-r20-0-prefill-f67zc, 1/1 Running
frontend pod: topo-c1-dp2tp4-disagg-r20-0-frontend-mnhpv, 1/1 Running
env: MEGAKERNEL=1, MEGAKERNEL_V2=1, profiling env absent
```

Same-image V2 isolation:

```text
image: localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-dense-family-sigmoid-v2-20260619T085608Z
control arm: dense_family_sigmoid_v2_off_mega_cf
test arm:    dense_family_sigmoid_v2_mega_cf
shape:       C32, ISL2055, OSL512
```

Files:

```text
.bench_runs_claude/results/370_deploy_dense_family_sigmoid_v2_off_same_image.txt
.bench_runs_claude/results/371_wait_dense_family_sigmoid_v2_off_same_image.txt
.bench_runs_claude/results/373_dense_family_sigmoid_v2_off_same_image_c32_512_r1.txt
.bench_runs_claude/results/374_dense_family_sigmoid_v2_off_same_image_c32_512_r2.txt
.bench_runs_claude/results/375_dense_family_sigmoid_v2_off_same_image_c32_512_r3.txt
.bench_runs_claude/results/376_deploy_dense_family_sigmoid_v2_on_same_image.txt
.bench_runs_claude/results/377_wait_dense_family_sigmoid_v2_on_same_image.txt
.bench_runs_claude/results/379_dense_family_sigmoid_v2_on_same_image_c32_512_r1.txt
.bench_runs_claude/results/380_dense_family_sigmoid_v2_on_same_image_c32_512_r2.txt
.bench_runs_claude/results/381_dense_family_sigmoid_v2_on_same_image_c32_512_r3.txt
```

Results:

```text
V2 off, same image:
  r1: TTFT_p50=1733 ms, user_tok_s_p50=51.85, agg_out_tok_s=1384.3
  r2: TTFT_p50=1607 ms, user_tok_s_p50=51.41, agg_out_tok_s=1395.8
  r3: TTFT_p50=1339 ms, user_tok_s_p50=51.13, agg_out_tok_s=1406.7

V2 on, same image:
  r1: TTFT_p50=1808 ms, user_tok_s_p50=51.45, agg_out_tok_s=1365.7
  r2: TTFT_p50=1290 ms, user_tok_s_p50=51.82, agg_out_tok_s=1436.9
  r3: TTFT_p50=1524 ms, user_tok_s_p50=51.96, agg_out_tok_s=1433.4
```

Interpretation:

- r1 on both sides carries post-roll/capture/autotune cost and should not be
  used as steady-state evidence.
- Steady V2-on (`1436.9`, `1433.4`) beats steady V2-off (`1395.8`, `1406.7`)
  on the same image and same dense-family stack by about `+2.1%` to `+2.9%`.
- V2-on also remains above original BEST_CONFIG C32 (`1424.9`) in both steady
  same-image repeats. V2-off is below original BEST_CONFIG on this same-image
  control.
- Current cluster is back on `dense_family_sigmoid_v2_mega_cf`: decode pod
  `topo-c1-dp2tp4-disagg-r20-0-decode-fxp7z`, 1/1 Running, 0 restarts.

## 2026-06-20 - V2 full serving suite rerun

Live state:

```text
arm: dense_family_sigmoid_v2_mega_cf
decode pod: topo-c1-dp2tp4-disagg-r20-0-decode-fxp7z, 1/1 Running, 0 restarts
env: MEGAKERNEL=1, MEGAKERNEL_V2=1, SIGMOID_QUANT=1, QB_WQB=1, KVA_WKWP=1
shape: ISL2055, OSL512
```

Files:

```text
.bench_runs_claude/results/382_dense_family_sigmoid_v2_full_20260620_c1_64_512.txt
.bench_runs_claude/results/383_dense_family_sigmoid_v2_full_20260620_c1_64_512_r2.txt
```

Run 1:

```text
C1:  TTFT_p50=490  ms, user_tok_s_p50=44.68, agg_out_tok_s=46.6
C4:  TTFT_p50=782  ms, user_tok_s_p50=58.60, agg_out_tok_s=230.3
C8:  TTFT_p50=1122 ms, user_tok_s_p50=56.00, agg_out_tok_s=425.9
C16: TTFT_p50=1203 ms, user_tok_s_p50=53.92, agg_out_tok_s=792.9
C32: TTFT_p50=1245 ms, user_tok_s_p50=52.35, agg_out_tok_s=1423.6
C64: TTFT_p50=1432 ms, user_tok_s_p50=46.43, agg_out_tok_s=2325.1
```

Run 2:

```text
C1:  TTFT_p50=477  ms, user_tok_s_p50=44.74, agg_out_tok_s=46.7
C4:  TTFT_p50=790  ms, user_tok_s_p50=58.85, agg_out_tok_s=231.8
C8:  TTFT_p50=1069 ms, user_tok_s_p50=56.38, agg_out_tok_s=434.5
C16: TTFT_p50=1317 ms, user_tok_s_p50=54.29, agg_out_tok_s=792.9
C32: TTFT_p50=1493 ms, user_tok_s_p50=51.44, agg_out_tok_s=1419.5
C64: TTFT_p50=1553 ms, user_tok_s_p50=46.49, agg_out_tok_s=2330.9
```

Interpretation:

- Full-suite C32 on 2026-06-20 is lower than the prior C32-only same-image V2
  repeats (`1433.4` to `1436.9`) and the earlier full warmed row (`1452.9`),
  landing at `1419.5` to `1423.6` aggregate.
- Per-user tok/s remains above the original BEST_CONFIG C32 row in run 1
  (`52.35` vs `51.07`) and roughly tied/slightly above in run 2 (`51.44`).
- C64 remains strong at `2325.1` to `2330.9` aggregate, comparable to the
  2026-06-17 restored C64 row (`2329.7`) and above original frozen BEST_CONFIG
  C64 (`2230.4`).

## 2026-06-20 - Deep C32 profile against 60 tok/s/user target

Target:

```text
Goal: C32, ISL2055, OSL512, 60 tok/s/user
Current normal C32 band: 51.44 to 52.35 tok/s/user
Required uplift: about +7.7 to +8.6 tok/s/user, or roughly +15% to +17%
```

Profiled deployment:

```text
arm: dense_family_sigmoid_v2_mega_cf
image: localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-dense-family-sigmoid-v2-20260619T085608Z
decode pod: topo-c1-dp2tp4-disagg-r20-0-decode-94c8q
env: MEGAKERNEL=1, MEGAKERNEL_V2=1
profile: TLLM_PROFILE_START_STOP=3000-3128, TLLM_PROFILE_LOG_RANKS=0,1,2,3
trace path: /cache/optrt/nsys/dense_family_sigmoid_v2_c32_512_20260620_deep_trace.json
```

Files:

```text
.bench_runs_claude/results/384_build_kineto_dense_family_sigmoid_v2_20260620.txt
.bench_runs_claude/results/385_apply_kineto_dense_family_sigmoid_v2_20260620.txt
.bench_runs_claude/results/387_profile_decode_env_20260620.txt
.bench_runs_claude/results/388_profile_pre_nvidia_smi_20260620.csv
.bench_runs_claude/results/389_dmon_c32_profile_20260620.txt
.bench_runs_claude/results/390_gpm_c32_profile_20260620.txt
.bench_runs_claude/results/391_drive_c32_profile_20260620.txt
.bench_runs_claude/results/392_trace_files_c32_profile_20260620.txt
.bench_runs_claude/results/393_profile_post_nvidia_smi_20260620.csv
.bench_runs_claude/results/394_telemetry_summary_c32_profile_20260620.txt
.bench_runs_claude/results/395_restore_dense_family_sigmoid_v2_after_20260620_profile.txt
.bench_runs_claude/results/396_restore_pods_after_20260620_profile.txt
.bench_runs_claude/results/397_restore_env_after_20260620_profile.txt
.bench_runs_claude/profiles/kineto_analysis_dense_family_sigmoid_v2_c32_512_20260620T1934_deep/analysis.txt
```

Load during profile:

```text
round1: TTFT_p50=2040 ms, user_tok_s_p50=52.10, agg_out_tok_s=1385.3
round2: TTFT_p50=1193 ms, user_tok_s_p50=52.28, agg_out_tok_s=1444.2
round3: TTFT_p50=2045 ms, user_tok_s_p50=51.38, agg_out_tok_s=1129.5
```

Interpretation:

- Round 2 is the normal serving band under the profiling manifest.
- Round 3 is where the trace export/profiler work is visible in throughput.
- The trace produced four rank files, each about `348M` to `361M`, under
  `/var/lib/optrt-cache/nsys/dense_family_sigmoid_v2_c32_512_20260620_deep_trace-rank-*.json`.

Cross-rank summary:

```text
rank0: span=20903 us/iter, GPU_busy=82.8%, idle=3589 us/iter, GPU kernels=23273 us/iter
rank1: span=20442 us/iter, GPU_busy=84.5%, idle=3172 us/iter, GPU kernels=23161 us/iter
rank2: span=19383 us/iter, GPU_busy=88.9%, idle=2161 us/iter, GPU kernels=23137 us/iter
rank3: span=19838 us/iter, GPU_busy=87.4%, idle=2509 us/iter, GPU kernels=23254 us/iter
mean GPU idle across ranks: 14.1%
classification: GPU-bound, with host overhead secondary
```

Aggregate subsystem table:

```text
Dense/proj GEMM      8925 us/iter, 38.5%, 608 launches/iter
MoE a2a/comm         3413 us/iter, 14.7%, 290 launches/iter
MoE expert GEMM      3005 us/iter, 13.0%, 174 launches/iter
GEMM generic         1892 us/iter,  8.2%, 403 launches/iter
DSA indexer          1749 us/iter,  7.5%, 263 launches/iter
Elementwise/copy     1333 us/iter,  5.7%, 748 launches/iter
Quant/dequant        1224 us/iter,  5.3%, 360 launches/iter
Attention generic     760 us/iter,  3.3%, 122 launches/iter
Reduce/allreduce      510 us/iter,  2.2%, 117 launches/iter
RoPE/embedding        315 us/iter,  1.4%,  93 launches/iter
Norm                   29 us/iter,  0.1%,  16 launches/iter
```

Top aggregate kernels:

```text
2854.6 us/iter, 122 calls/iter, Dense/proj GEMM, nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT
2761.3 us/iter, 177 calls/iter, Dense/proj GEMM, nvjet_sm100_ootst_128x128_256x7_4x1_2cta
1758.7 us/iter,  58 calls/iter, MoE expert GEMM, CuteDSL Blackwell block kernel
1325.2 us/iter,  64 calls/iter, Dense/proj GEMM, cutlass3x_sm100_bstensorop_s256x128
1105.1 us/iter,  58 calls/iter, MoE a2a/comm, moeAllToAllKernel<1>
 983.1 us/iter,  58 calls/iter, MoE expert GEMM, CuteDSL Blackwell block kernel
 852.6 us/iter, 122 calls/iter, Dense/proj GEMM, CuteDSL Blackwell dense kernel
 810.3 us/iter, 299 calls/iter, Quant/dequant, quantize_with_block_size
 786.9 us/iter,  16 calls/iter, DSA indexer, topKPerRowDecode
 748.0 us/iter, 119 calls/iter, GEMM generic, lowrank_gate GEMM
 694.6 us/iter,  58 calls/iter, MoE a2a/comm, moeAllToAllKernel<4>
 586.8 us/iter,  58 calls/iter, MoE a2a/comm, computeCountAndIndiceDevice
 559.7 us/iter,  58 calls/iter, MoE a2a/comm, memsetExpertIdsDevice
 554.0 us/iter,  61 calls/iter, Attention generic, fmhaSm100fKernel
 466.5 us/iter,  58 calls/iter, MoE a2a/comm, moveIndiceDevice
```

GPU telemetry:

```text
decode GPUs: 4,5,6,7
dmon SM avg: 81.7%, 82.1%, 83.0%, 82.7%
dmon memory util avg: 30.1%, 30.1%, 30.8%, 30.5%
GPM DRAM avg: 24.6%, 24.5%, 24.8%, 24.8%
GPM NVLink rx avg: 920, 917, 921, 922 MiB/s
GPM NVLink tx avg: 922, 910, 950, 897 MiB/s
power avg: 553W, 579W, 562W, 594W
FB memory: about 145.6G to 146.4G per decode GPU
```

Host/runtime notes:

```text
cudaEventSynchronize: about 2687 to 2844 us/iter by rank
cudaStreamSynchronize: about 611 to 766 us/iter by rank
cudaLaunchKernel: about 235 to 257 us/iter by rank
cudaMemcpyAsync: about 239 to 253 us/iter by rank
host CPU aten/host_op total: 5818 us/iter
largest host ops: aten::copy_ 1238 us/iter, aten::to 1041 us/iter, aten::_to_copy 985 us/iter
```

Conclusions:

- The profile does not point at memory bandwidth, raw NVLink bandwidth, or
  power/thermal throttling as the primary blocker. Decode GPUs are active, but
  DRAM is only around `25%` in GPM and NVLink is around `0.9G` to `1.0G` MiB/s
  per GPU.
- The biggest recoverable bucket remains dense/proj GEMM: `8.9 ms/iter`,
  `38.5%` of GPU kernel time, and `608 launches/iter`. A `40%` reduction of
  that bucket alone would be roughly a `15%` total GPU-time reduction if it is
  on the critical path, which is the right order of magnitude for the C32
  60 tok/s/user goal.
- A more realistic path is probably dense/proj GEMM plus MoE prep/comm. The
  MoE a2a/comm bucket is `3.4 ms/iter`; important rows are
  `moeAllToAllKernel<1>` (`1.1 ms/iter`), `moeAllToAllKernel<4>`
  (`0.7 ms/iter`), `computeCountAndIndiceDevice` (`0.59 ms/iter`),
  `memsetExpertIdsDevice` (`0.56 ms/iter`), and `moveIndiceDevice`
  (`0.47 ms/iter`). That looks like launch/prep/small-kernel overhead as much
  as bandwidth pressure.
- Host overhead is visible but secondary for this target. Mean GPU idle is
  `14.1%`, while the profiler classification is still GPU-bound at `85.9%`
  busy. Host copies and syncs should be kept in view, but they are not the first
  place to spend the next kernel-implementation pass.

Next optimization targets:

1. Dense/proj GEMM tactic and launch reduction for the `nvjet_sm100_tst_64x8`,
   `nvjet_sm100_ootst_128x128`, `cutlass3x_sm100_bstensorop_s256x128`, and
   CuteDSL Blackwell dense families.
2. MoE a2a/prepare consolidation: reduce or fuse `computeCountAndIndiceDevice`,
   `memsetExpertIdsDevice`, and `moveIndiceDevice`, then re-check whether
   `moeAllToAllKernel<1>/<4>` remains dominant.
3. DSA/quant cleanup only after the first two buckets move. Combined
   DSA+quant+elementwise/copy is material, but the path to the full C32 target
   is clearer through dense/proj and MoE prep first.

Post-profile restore:

```text
arm: dense_family_sigmoid_v2_mega_cf
decode pod: topo-c1-dp2tp4-disagg-r20-0-decode-qhlv9, 1/1 Running, 0 restarts
prefill pod: topo-c1-dp2tp4-disagg-r20-0-prefill-f67zc, 1/1 Running
frontend pod: topo-c1-dp2tp4-disagg-r20-0-frontend-mnhpv, 1/1 Running
env: MEGAKERNEL=1, MEGAKERNEL_V2=1, profiling env absent
```

## 2026-06-21 - Dense/proj kernel-to-callsite map

Goal:

```text
Map the top dense/proj kernel families to exact model sites, shapes,
dtype/scale layout, rank behavior, and source path before another kernel pass.
```

Files:

```text
.bench_runs_claude/dense_callsite_mapper.py
.bench_runs_claude/results/399_dense_callsite_map_c32_steady_20260621.md
source traces:
  /var/lib/optrt-cache/nsys/dense_family_sigmoid_v2_c32_512_20260620_deep_trace-rank-*.json
```

Methodology:

- The raw Kineto trace has CUDA kernel names and launch grids, but CUDA graph
  replay collapses Python/module scope to `cudaGraphLaunch`; there are no
  projection names in the production replay trace itself.
- The mapper joins three sources:
  1. production graph kernel name + launch grid,
  2. DeepSeek V3.2 model code and fused helper code,
  3. existing NVFP4 shape/debug catalogue.
- The profiling window contains mixed graph signatures. The steady table uses
  `--require-steady-c32`, keeping only graph launches with the dominant full
  C32 dense signature instead of tail/lower-batch graph replays.

Cross-rank mean steady C32 map:

```text
self_attn.gate_proj, chunked output gate
  2856.1 us/step, 122 launches/step
  shape: M=32, N=8192 each chunk x2/layer, K=7168
  layout/source: bf16 x bf16, _gate_proj_chunked -> torch.mm -> cuBLASLt/NVJET
  rank behavior: attention-DP, every decode rank owns the full gate_proj

self_attn.fused_kv_a_wk_weights_proj
  1906.7 us/step, 61 launches/step
  shape: M=32, logical N=kv_a 2112 + pad64 + indexer wk/wp 192,
         compute grid pads to 2560, K=7168
  layout/source: NVFP4 swizzled scales, _FusedKvAWkWpNvfp4 -> nvfp4_gemm -> NVJET
  note: kv_a rows are padded to a 128-row scale boundary before concatenating wk/wp

self_attn.o_proj
  1257.7 us/step, 61 launches/step
  shape: M=32, N=7168, K=16384
  layout/source: NVFP4 input from fused sigmoid-mul-quant, swizzled scales,
                 Linear(o_proj) -> nvfp4_gemm -> CUTLASS

input/post-attention lowrank gated norms
  768.1 us/step, 122 launches/step
  shape: M=32, hidden=7168, low-rank gate
  layout/source: bf16 gated norm, quant variants also emit NVFP4 handoff,
                 CuteDSL lowrank gate kernels

MoE shared_experts.gate_up_proj, layers 3-60
  542.8 us/step, 58 launches/step
  shape: M=32, N=4096, K=7168
  layout/source: NVFP4 swizzled scales, nvfp4_gemm -> NVJET

MLA k_b absorption BMM
  465.7 us/step, 61 launches/step
  shape: batched heads, M=32, N=512, K=128-ish per head
  layout/source: bf16 BMM, MLA._bmm_bf16_out(k_b_proj_trans) -> CuteDSL BF16 BMM

MoE router gate
  462.5 us/step, 116 launches/step
  shape: M=32, N=256 experts, K=7168
  layout/source: bf16/f32 router logits, dsv3_router_gemm_op/cuBLASLt split-K

MLA v_b absorption BMM
  387.1 us/step, 61 launches/step
  shape: batched heads, M=32, N=128, K=512-ish per head
  layout/source: bf16 BMM, MLA._bmm_bf16_out(v_b_proj) -> CuteDSL BF16 BMM

self_attn.q_b_proj on S/reuse layers
  357.0 us/step, 45 launches/step
  shape: M=32, N=24576, K=1536
  layout/source: NVFP4 swizzled scales, q_b_proj -> nvfp4_gemm -> NVJET

MoE shared_experts.down_proj, layers 3-60
  313.0 us/step, 58 launches/step
  shape: M=32, N=7168, K=2048
  layout/source: NVFP4 swizzled scales, nvfp4_gemm -> NVJET

lm_head/logits
  265.2 us/step, 1 launch/step
  shape: M=32 decode rows, large local vocab/output shard
  layout/source: bf16 logits GEMM, cuBLASLt/NVJET

self_attn.fused_q_b_wq_b on F/indexer layers
  156.2 us/step, 16 launches/step
  shape: M=32, qb_out=24576, wq_out=8192, K=1536
  layout/source: NVFP4 swizzled scales, q_b scale reused for indexer wq_b,
                 _FusedQbWqBNvfp4 variable-N CuteDSL op

dense layers 0-2 mlp.gate_up_proj
  86.0 us/step, 3 launches/step
  shape: M=32, N=36864, K=7168
  layout/source: NVFP4 swizzled scales, nvfp4_gemm -> NVJET

dense layers 0-2 mlp.down_proj
  61.9 us/step, 3 launches/step
  shape: M=32, N=7168, K=18432
  layout/source: NVFP4 swizzled scales, nvfp4_gemm -> CUTLASS
```

Interpretation:

- The biggest dense/proj kernel family is not `q_b_proj` or `o_proj`; it is the
  bf16 attention output gate: `self_attn.gate_proj`, emitted as two
  `torch.mm` chunks per layer on a side stream.
- The second largest row is the fused attention/indexer projection
  `kv_a + wk/wp`, not a plain `kv_a_proj`.
- `q_b_proj` is split by DSA policy: `45` S/reuse layers run q_b alone, while
  `16` F/indexer layers run fused `q_b + wq_b`. Together they are about
  `513 us/step`, far below the gate and fused kv_a rows.
- `o_proj` is a real target at `1258 us/step`, but the earlier campaign's
  focus on generic NVFP4 projection tactics missed that the largest single
  callsite is bf16 gate overlap and that `kv_a` is already fused with DSA
  indexer work.
- This table is kernel-time ownership, not yet critical-path ownership. The
  gate projection is intentionally overlapped under the attention block, so the
  next proof must measure how much of `gate_proj` remains on the critical path
  before implementing a replacement kernel for it.

Recommended next step:

```text
Compute per-site critical-path/slack for the top four mapped sites:
  1. self_attn.gate_proj chunked bf16 gate,
  2. fused_kv_a_wk_weights_proj,
  3. o_proj,
  4. lowrank gated norm.

Only then pick the first implementation target.
```

### 2026-06-21 - Dense/proj critical-path exposure

Artifacts:

- Mapper: `.bench_runs_claude/dense_callsite_mapper.py`
- Critical-path proxy: `.bench_runs_claude/dense_critical_path_analyzer.py`
- Callsite map: `.bench_runs_claude/results/399_dense_callsite_map_c32_steady_20260621.md`
- Exposure report: `.bench_runs_claude/results/400_dense_critical_path_c32_steady_20260621.md`

Command:

```bash
python3 .bench_runs_claude/dense_critical_path_analyzer.py \
  /var/lib/optrt-cache/nsys/dense_family_sigmoid_v2_c32_512_20260620_deep_trace-rank-*.json \
  --skip-graphs 1
```

Run shape:

- `4` rank traces
- `99` steady C32 CUDA graph replays per rank after skipping the first replay
- Steady replay filter: full C32 dense signature with at least `64` launches of
  `cutlass3x_sm100_bstensorop_s256x128x64gemm_block_scaled`, grid `(2,28,1)`

Cross-rank mean steady C32 exposure:

| site | kernel us/step | launches/step | active us/step | exposed us/step | exposed active % | overlap active % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `self_attn.gate_proj_chunked` | 2856.1 | 122.0 | 2776.6 | 78.7 | 2.8 | 97.2 |
| `self_attn.fused_kv_a_wk_weights_proj` | 1906.7 | 61.0 | 1906.7 | 0.0 | 0.0 | 100.0 |
| `self_attn.o_proj` | 1257.7 | 61.0 | 1257.7 | 1251.7 | 99.5 | 0.5 |
| `lowrank_gated_norms` | 768.1 | 122.0 | 768.1 | 756.3 | 98.5 | 1.5 |

Gate-to-consumer slack:

| metric | value |
| --- | ---: |
| samples | 24156 |
| min us | 30.3 |
| p05 us | 32.8 |
| median us | 35.0 |
| p95 us | 184.3 |
| max us | 224.0 |
| negative slack count | 0 |

Interpretation:

- `gate_proj` owns the most dense/proj kernel time, but it is almost entirely
  overlapped on the side stream. Every sampled two-chunk gate pair finishes
  before its `extFusedSigmoidMulQuantizeKernel` consumer, with at least
  `30.3 us` slack.
- A standalone `gate_proj` kernel rewrite is therefore a weak first target for
  decode-step wall time. It can still reduce resource contention, but direct
  dependency shortening is mostly hidden until the existing slack is exhausted.
- `o_proj` and the low-rank gated norm kernels are exposed almost one-for-one
  on the timeline. Optimizing these should translate much more directly into
  wall-time improvement.
- `fused_kv_a_wk_weights_proj` has zero timeline-only exposed time because it
  overlaps with side-stream work, but it remains a plausible main-path target:
  the positive gate slack implies the consumer is waiting on the other attention
  dependency, not on gate completion. Reducing this path can move the consumer
  earlier until the `30-35 us` per-layer gate slack is consumed.

Target ordering from this proof:

1. `self_attn.o_proj`: exposed `1252 us/step`, NVFP4 CUTLASS path,
   `M=32,N=7168,K=16384`.
2. `lowrank_gated_norms`: exposed `756 us/step`, CuteDSL low-rank gate kernels.
3. `self_attn.fused_kv_a_wk_weights_proj`: large main-path candidate,
   `1907 us/step`, but needs a dependency-aware before/after test because the
   timeline proxy marks it as overlapped.
4. `self_attn.gate_proj_chunked`: largest kernel-time owner, but low direct
   wall-time priority unless we combine it with adjacent consumer work or prove
   resource contention is limiting the main path.

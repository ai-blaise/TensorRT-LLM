# Decode NVFP4 dense-GEMM backend: enable CuteDSL

## Change

`sdt_gen.yaml` (DeepSeek-V3.2-REAP-345B decode/generation phase) `nvfp4_gemm_config.allowed_backends`:

    - [cutlass, cublaslt, cuda_core]      # before (stock default; cutedsl excluded for build time)
    + [cutedsl, cublaslt, cutlass, cuda_core]   # after

The `torch.ops.trtllm.nvfp4_gemm` AutoTuner profiles every backend in `allowed_backends` once per
GEMM shape during warmup and caches the fastest (`autotuner.py` `_profile_runners`: starts at
`+inf`, a failing/unsupported tactic gets `+inf` and is never chosen, best is replayed with zero
steady-state overhead). Adding `cutedsl` is therefore **monotonically non-regressing**: it only
changes a shape's kernel when CuteDSL is *measured faster* than cutlass/cublaslt/cuda_core for that
exact shape; otherwise it is ignored. The only cost is one-time warmup JIT compilation of the
CuteDSL NVFP4 Blackwell kernels (observed ~3-4 min for the full shape set). The CuteDSL runtime is
already warm in this deploy — the indexer uses `use_cute_dsl_topk` + `use_cute_dsl_paged_mqa_logits`.

## Measurement

`nvfp4_gemm_backend_microbench.py` on one B200 (sm100), GPU lock A. For each dense decode GEMM that
routes through `nvfp4_gemm` (MLA q/kv/o projections — not TP-sharded under attention_dp — plus
dense-layer and shared-expert MLP), at decode M = cuda-graph batch sizes {1,2,4,8,16,32,64}: force
each backend, let the AutoTuner pick its best tactic, verify output vs cutlass (max rel-diff
<= 0.0036, pure FP4 quant noise), then time the cached selection under a CUDA graph (300 replays,
median, slow-tail dropped). Full table: `microbench_results_b200.log`.

Best-correct-backend histogram over 70 (shape, M) cells:

    cutedsl: 54    cublaslt: 14    cuda_core: 2    cutlass: 0

- `cutlass` (the stock default's primary kernel) **never wins** — `cublaslt`, already in the
  default list, beats it on every shape.
- `cuda_core` only wins at M<=2 on the tiny `kv_a_proj_mqa` (N=576); it is correctly capped at M<=8
  and is 10-40x slower for larger M (the AutoTuner avoids it automatically).
- `cutedsl` wins the majority, with the largest gains on the biggest-K / highest-cost GEMMs that
  dominate the decode dense-GEMM time:

    o_proj            (K=16384)            8.5 - 12.3 %  faster than best default
    dense_gate_up_tp4 (K=7168,  N=9216)    4.2 - 14.0 %
    dense_down_tp4    (K=4608)             2.0 -  9.3 %
    kv_b_proj         (K=512,   N=32768)   2.1 -  6.8 %
    q_b_proj          (K=1536,  N=24576)   1.5 -  7.4 %

`win%_vs_default` in the table = (best of cutlass/cublaslt/cuda_core - chosen) / best-default.

## Scope / caveats

- Microbench shapes are TP4 (MLA unsharded under attention_dp). The shipped gen config is TP2/EP2;
  o_proj/q_b_proj/kv_b_proj are TP-independent under attention_dp so their wins transfer directly;
  the dense-MLP shard dim changes (tp2: gate_up N=18432, down K=9216) but sits in the same win
  regime as the measured tp4/full points (cutedsl wins both `dense_gate_up_full` and
  `dense_gate_up_tp4`).
- This isolates the dense GEMM + MLA-projection latency. End-to-end decode tok/s/user also depends on
  attention, indexer, and MoE-expert GEMMs (separate ops). Net e2e delta to be confirmed with the
  `trtllm-bench` decode harness in `sched_r2/` once a GPU window frees from the Indexer fleet.

## Related, separately measurable lever (staged, not yet benched)

MLA absorb BMM (`q_nope x k_b_proj_trans`): on sm100 with `use_cute_dsl_blockscaling_bmm=False`
(default) and FP8 MLA weights, `attention.py` `fp8_block_scaling_bmm_out` runs `torch.bmm` on a
resident **dequantized BF16** weight (`k_b_proj_trans_dequant`). Setting
`use_cute_dsl_blockscaling_bmm=True` switches to the native FP8 `cute_dsl_fp8_bmm_blackwell` and
drops the resident BF16 dequant buffer. Worth a follow-up microbench at decode M.

# PDE broader-system decode probe — DeepSeek-V3.2-REAP-345B (2026-06-15)

Beyond the already-concluded dense-GEMM/topk work. Direct op/backend-level tests at prod decode shapes
under CUDA-graph capture; no full serve. Harnesses + progress.log in /home/spencer/wt/_hb/pde_system/.

## 1. Decode system map (per layer, M=1 tok/s/user regime, B200; measured here unless noted)
- dense GEMMs (q/kv/o proj + dense/shared MLP): ~6-16us each, ~5-6 => ~50us  [prior probe]
- attention (flash_mla_sparse_fwd, bf16/fp8 KV): ~18us, FLAT in batch (17.6->20.3us M1->64)  [MEASURED]
- indexer (topk over live KV): ~6us  [prior probe]
- MoE (WARPDECODE cute_dsl grouped-GEMM, EP2 64e): ~45us  [MEASURED]
- sum of measured COMPUTE ~120us/layer ; step budget ~340us/layer (20.8ms/61L)
- => ~65% of per-layer time is inter-kernel launch/overhead/sync + bmm/rope/quant glue, NOT compute kernels.

## 2. Ranked candidate list (impact x direct-testability)
1. MoE backend (CUTLASS vs WARPDECODE): HIGH impact, FULLY testable. RESULT below.
2. SMC spec-decode (gamma/n_particles/draft): HIGHEST potential, serve-gated (acceptance data-dependent).
3. Attention region: MEASURED small+flat (~18us); not a lever.
4. Cross-region control flow / metadata hoist: nvfp4-build-gated; served fp8 path uses standard MLA-gen (captured).
5. Topology: already DP2/TP4-tuned (prior); overlap scheduler already on.

## 3. VERIFIED measurement (cos=1.0)
MoE backend, real CutlassFusedMoE vs CuteDslFusedMoE(WARPDECODE), per-backend self-quantized input, captured:
  E=128: M1 CUTLASS 85.6 vs WARPDECODE 45.1 =1.90x ; M8 313 vs 236 =1.32x ; M32 597 vs 459 =1.30x
  E=64 (EP2/rank): M1 79.9 vs 45.3 =1.77x ; M8 240 vs 173 =1.38x ; M32 341 vs 262 =1.30x
  cos=1.000 vs each other across ALL batch/expert counts. (WARPDECODE/CUTEDSL->CuteDslFusedMoE confirmed.)
  trtllm_gen runner: AutoTuner==no-autotune (1.00-1.02x) => MoE tactic NOT misselected. cursor-mega ~= trtllm_gen (2%).
CONTEXT: ALL production decode configs (decode.yaml, prefill.yaml, topo-c1-dp2tp4-r20, smc_agg_tp4) ALREADY
  use WARPDECODE. Only sdt_gen_decode.yaml uses CUTLASS, and its README says it is a GEMM-MICROBENCH config
  (CUTLASS pinned as MoE baseline to isolate the dense-GEMM variable). So this CONFIRMS production MoE is optimal;
  it is NOT a new serving win. Corrected serving variant: sdt_gen_decode_SERVING_FIX.yaml (backend->WARPDECODE).

## 4. Serve-gated: SMC (measurable side quantified)
- Draft = max_draft_len=gamma=6 GLM-4-9B-FP8 forwards PER target verify step. forward0 = batch_size tokens;
  forwards 1..5 = batch_size x (n_particles*gamma+1)=batch_size x 25 tokens (the particle tree). 5/6 fwds at 25x batch.
- Draft model present locally. tok/s/user = accepted_len / (6x GLM-9B draft + 345B verify). Acceptance DATA-DEPENDENT.
- Cheap config levers: smc_vectorize_logprob_record (OFF; saves n_particles-1=3 launches/draft-layer ~15/step, 1-ULP);
  gamma/n_particles (pure acceptance<->cost tradeoff, serve-gated). Sync-free bookkeeping already optimized.

## 5. Biggest remaining tok/s/user lever (honest)
NOT a single compute kernel. Decode is ~65% inter-kernel overhead -> the PDE device-control-flow (G3) +
cross-step persistence (G9) primitives target exactly this but are NOT yet wired into the live serving runtime.
The other top lever is SMC acceptance x draft-cost (the 6x GLM-9B draft forwards), which is serve-gated.

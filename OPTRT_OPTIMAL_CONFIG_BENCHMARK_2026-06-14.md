# op-trt optimal-config in-depth benchmark — 2026-06-14

**Question (from the operator):** is the decode config actually optimal, or was perf
left on the table by not turning certain levers on? Pull in upstream, assemble
everything-good-on, benchmark in depth.

**Deploy:** single 8×B200 node `a4-us-002-rl9`, **no InfiniBand** (`NCCL_IB_DISABLE=1`).
Disagg r20: decode TP4+EP4+attention-DP (`moe_tp_size=1`), prefill TP2. Untrained
DeepSeek-V3.2-REAP-345B NVFP4 perf testbed — validation is throughput + no-error +
numerical, never output coherence.

## Method

- Pulled `origin/op-trt` commit `4c7fe2457` ("optimized SMC NIXL defaults") — clean local
  merge, **not pushed**. It documents the intended decode stack (DeepEP-LL + post-quant-a2a
  + SMC).
- Comprehensive lever audit (every `TRTLLM_OPTRT_*`/MoE/indexer/KVarN/comm flag vs live
  config): the live config was **already everything-good-on** (WARPDECODE+force+decode_1cta,
  `index_topk_freq:4`, fp4 indexer-K, NVFP4-HISA, KVarN-k2v2, cuda graphs, NIXL/PYTHON, PDL,
  fixed-tactic, all fusion flags default-on).
- Clean **same-image A/B**, full concurrency curve c=1..64, ISL≈2048, OSL=512. Arms differ
  only in MoE comm method; `combine=false` held constant where comparing comm methods.

## Results — per-user decode tok/s (p50)

| C | Baseline NVLINK 2-sided (combine=true, 0426) | NVLINK 2-sided (combine=false, deepep) | DeepEP-LL | M1 NVLinkOneSided (deep-fix) | M1 same-image control (2-sided) |
|---|---|---|---|---|---|
| 1 | 42.2 | 42.8 | 42.1 | 43.6 | 42.9 |
| 4 | 55.2 | 56.0 | 55.2 | 56.3 | — |
| 8 | 53.8 | 54.6 | 52.6 | 54.1 | — |
| 16 | 52.4 | 52.3 | 50.8 | 53.0 | 52.2 |
| 32 | 48.8 | 50.0 | 48.4 | 49.2 | — |
| 64 | 44.6 | 45.0 | 44.5 | 45.0 | 44.8 |

Run-to-run noise across NVLINK-equivalent arms is ≈ ±1.5%.

## Per-lever verdict

1. **DeepEP low-latency (M3)** — the one config-flip lever not previously adopted. Tested with
   the *complete* Spencer flag set (`DEEPEPLOWLATENCY` + `TRTLLM_DEEP_EP_TOKEN_LIMIT=64` +
   `TRTLLM_DEEP_EP_DISABLE_P2P_FOR_LOW_LATENCY_MODE=0` + `TRTLLM_MOE_POST_QUANT_ALLTOALLV=1`).
   **Loses at every concurrency (−0 to −3.2%).** Those last two flags are already-on code
   defaults; NVSHMEM still attempts IBGDA at buffer construction regardless of the P2P-pin flag
   and CPU-falls-back on this no-IB node (`ibgda_nic_mem_gpu_map failed`, `error=800`). The
   docs' "wins every point / −1.5 ms/step" is IB-hardware data. **NVLINK two-sided is optimal.**

2. **post-quant-a2a / P2P-pin flags** — already-on defaults (`=1`/`=0` in code). Not missed
   levers; setting them explicitly changes nothing.

3. **SMC speculative decoding** — correctly excluded from the throughput verdict: per
   `docs/blaise/smc_sd.md` it is validated by draft/verify kernel correctness, not e2e
   acceptance, because the target is untrained (`tokens_per_gen_step=25` → pure overhead at
   zero acceptance). A production lever, not benchmarkable here.

4. **M1 one-sided a2a** — was crash-on-enable (output-ownership mismatch: WARPDECODE writes its
   own buffer; NVLinkOneSided expects the output in the combine workspace). **Deep-fixed**
   (commit `179ad71f0`): thread the workspace payload tensor into the WARPDECODE overlay so the
   trtllm_gen runner writes MoE output directly into it (zero-copy), then combine reads it
   in-workspace. Pure-Python, validated via a thin overlay image. **Result: runs clean on pure
   NVLink (0 IBGDA, stable full sweep) but throughput-NEUTRAL** (+1.5/+1.5/+0.4% at c1/16/64 vs
   the same-image two-sided control — within ±1.5% noise). The projected +1.5–4.5%/step does
   not materialize: at decode the a2a payload is tiny (1–64 tok/rank), so one-sided-vs-two-sided
   handshake savings dilute below noise; the bulk NVLink transfer cost (identical both ways)
   dominates.

5. **MoE megakernel** — `TRTLLM_OPTRT_MOE_MEGAKERNEL=1` (V1, the WARPDECODE Phase-1 cursor op)
   crashes on this checkpoint: `assert global_sf.numel()==1` (REAP-128 has a multi-element
   global scale-factor). `TRTLLM_OPTRT_MOE_MEGAKERNEL_V2` only engages on the cute_dsl backend,
   not WARPDECODE — so prior "V2 neutral" tests were likely inert. Not pursued (checkpoint-SF
   kernel fix + rebuild; MoE GEMM is near-floor per prior profiling).

## Conclusion

The decode config was **already optimal for every flippable lever** on this no-IB node. The
levers that looked untapped were each blocked — DeepEP-LL by hardware (needs IB), M1 and the
megakernel by unfinished code (both crashed). M1 was carried to completion (deep fix, now
functional on pure NVLink) and measured **neutral** at decode scale. The validated optimum
remains **NVLINK two-sided + everything-else-on** (0426 image); the lab was restored to it.

Artifacts: sweep logs in `.bench_runs_claude/results/`; M1 fix in commit `179ad71f0`
(`warp_decode.py`, `moe_scheduler.py`); overlay image `optrt-179ad71f0-m1fix-*` in the local
registry. Nothing pushed to remote.

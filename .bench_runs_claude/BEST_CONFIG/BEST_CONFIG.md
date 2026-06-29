# op-trt — BEST KNOWN-GOOD SERVING CONFIG (frozen 2026-06-15, RE-CONFIRMED 2026-06-17)

The validated optimum for the DeepSeek-V3.2-REAP-345B-NVFP4 disagg deployment on
the single 8×B200 / no-IB node. This is the one config the whole optimization
campaign produced as a real, repeatable serving-side win.

**RE-CONFIRMED as the production config on 2026-06-17** after the config/parallelism
campaign (`CAMPAIGN_FINDINGS.md`) was explored and **ROLLED BACK** — the campaign config
(`max_batch 256` + `warp_decode.policy: auto`) did NOT beat this at the C=32 operating
point, so the lab was restored to the values below. Confirmed by two reruns:
`sweep_confirmed.txt` (revert_verify, 2026-06-15) and `results/26_best_restored.txt`
(best_restored, 2026-06-17 → C=32 = 50.08, within run-to-run noise of the prior 51.07).

## EXACT VALUES (reproducible)

**Image (BOTH workers — must match for the NIXL KV-transfer handshake):**
```
localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-vbfuse-nvfp4-20260615T043037Z
```
Lineage: hisparse-fixes (`7facf7585`, the megakernel fc2_input_scale fix + HISA merge)
→ v_b fusion (`1e101838e3`, e2e-neutral but correct) → nvfp4 small-M cuBLASLt fallback
(`529374445d`, e2e-neutral). The **`7facf7585` megakernel fix is what makes `MEGAKERNEL=1`
runnable** — do not use a pre-`7facf7585` image with the megakernel.

**Decode-worker env (the win is the FIRST two TOGETHER — it's an interaction):**
```
TRTLLM_OPTRT_MOE_MEGAKERNEL = 1            # the WARPDECODE Phase-1 megakernel
TRTLLM_FORCE_COMM_METHOD     = NVLINK_TWO_SIDED
# explicitly NOT set: TRTLLM_DEEP_EP_TOKEN_LIMIT, TRTLLM_DEEP_EP_DISABLE_P2P_FOR_LOW_LATENCY_MODE,
#                     TRTLLM_MOE_POST_QUANT_ALLTOALLV, TRTLLM_OPTRT_MOE_ONESIDED_A2A
```

**decode.yaml (the load-bearing key + the MoE/sparse config):**
```
moe_config.use_low_precision_moe_combine: false   # <-- REQUIRED with the megakernel; combine=true
                                                  #     re-quants the fused output and KILLS the +2%
moe_config.backend: WARPDECODE
moe_config.warp_decode: {enabled: true, policy: force, tile_mode: decode_1cta, max_batch_size: 64}
sparse_attention_config: dsa / indexer_mode=indexcache-hisa / index_topk=1024 / indexer_k_dtype=fp4
                         / mla_latent_kv_dtype=kvarn_k2v2 / enable_nvfp4_hisa=true / hisa_min_seq_len=65536
```
Full decode.yaml + prefill.yaml in `configmap.yaml`; full DGD in `dgd.yaml`.

**Topology:** r20 disaggregated. Decode = TP4 + EP4 + attention-DP (4 ranks, GPUs 4-7).
Prefill = TP2. allreduce=MNNVL. cache_transceiver = NIXL, transceiver_runtime=PYTHON.
CUDA graphs ON (batch_sizes 1..64). Overlap scheduler ON. NextN/MTP: head ABSENT from
checkpoint → spec-decode OFF (1 token/step).

## CONFIRMED PERFORMANCE (sweep_confirmed.txt; osl=512, ISL=2055, tok/s/user)
```
  C    TTFT_p50   tok/s/user   agg_out_tok/s
  1      465        43.30          45.3
  4      755        55.97         222.1
  8     1106        54.03         414.0
 16     1205        53.24         783.2
 32     1396        51.07        1424.9   <- the operating point; megakernel win lives here
 64     1797        44.77        2230.4
```
Megakernel+combine=false gives **~+2-3% at batch-32** (51 vs ~49 with MK off, either combine).
Neutral at other concurrencies. It is the ONLY validated serving-side throughput lever found.

**2026-06-17 restore re-confirm** (`results/26_best_restored.txt`, fresh reload of this exact config):
```
  C    TTFT_p50   tok/s/user   agg_out_tok/s
  1      478        43.22          45.2
  4      746        56.20         222.8
  8      977        54.32         415.5
 16     1130        52.32         774.9
 32     1642        50.08        1386.6   <- operating point, matches prior within noise
 64     1560        44.85        2329.7
```

## DEPLOY RECIPE
```
cd .bench_runs_claude
python3 deploy_arm.py vbfuse_nvfp4_mega_cf      # both workers -> vbfuse-nvfp4, MK=1, combine=false
# GOTCHA: a config-only combine flip does NOT auto-roll the decode pod (DGD unchanged) —
#   if only combine changed, `kubectl delete pod <decode>` to force a configmap re-read.
bash run_arm_sweep.sh vbfuse-nvfp4 verify results/verify.txt   # wait+settle+handshake+sweep
```

## WHAT IS *NOT* IN THIS CONFIG (campaign verdicts — see memory)
- **Campaign config (`max_batch 256` + `policy: auto` + `KV 0.6`)**: explored 2026-06-16 — real
  exploratory wins (concurrency cap +40–64% AGG @C=128–256; attn-DP=false +122% @C=1; the
  aggregated/2-pool "both" study) but it did NOT beat this at the C=32 operating point, so it was
  **ROLLED BACK 2026-06-17**. Full writeup in `CAMPAIGN_FINDINGS.md` + `REGIME_CONFIGS.md` (kept for
  reference; revisit deliberately only for a throughput-oriented or 2-node deployment).
- **C=1 async-dispatch reorder**: BROKE decode (engine event-id desequencing → hang), reverted. Landmine.
- **NextN/MTP spec-decode**: checkpoint has no MTP head → config-impossible. Needs offline re-graft.
- **Attn-gate NVFP4 (SM80→SM100 fix)**: checkpoint never quantized the gate. Needs offline re-quant.
- Neutral/at-floor (do not re-chase): FlashMLA v_b fusion, nvfp4 tactic, M1 a2a, moe_prepare, DeepEP-LL.

# op-trt — TWO REGIME CONFIGS (the 2026-06-16 config/parallelism campaign)

The big result: the prior "best config" (attn-DP=true, max_batch 64) was a **single mid
operating point** that left ~**2× single-stream latency** and ~**1.6× aggregate throughput**
unexploited. The decode was never compute/floor-bound — it was **config-capped and
batch-starved**. There is no one optimum: latency and throughput want **opposite parallelism**.
On the single 8-GPU node only one decode config runs at a time → pick per SLA.

All numbers: osl=512, ISL=2055, tok/s/user + agg_out_tok/s. Same image/base/topology as
`BEST_CONFIG.md` (image `optrt-529374445d-vbfuse-nvfp4`, env `TRTLLM_OPTRT_MOE_MEGAKERNEL=1`
+ `TRTLLM_FORCE_COMM_METHOD=NVLINK_TWO_SIDED`, `use_low_precision_moe_combine=false`, DSA/sparse
unchanged, disagg r20: prefill 4 GPUs TP2+EP4 / decode 4 GPUs).

---

## THE THREE LEVERS (measured)

| Lever | Result | Mechanism |
|-------|--------|-----------|
| **#1 concurrency cap** | **+40% agg @C=128 (free), +64% @C=256** | `max_batch_size` 64→256 + cuda-graphs to 256. GPU was still scaling at C=64; config capped it. |
| **#3 parallelism (the big one)** | **+122% per-user @C=1**, +28–49% @C≤16 | `enable_attention_dp=false` → TP attention shards the decode-GEMM weight → ¼ weight-load/rank at M=1. Regime knob: wins low-C, loses high-C. |
| **#2 prefill** | **neutral** | High-C TTFT is decode-saturation (decode full at max_batch when C=max_batch), NOT prefill-bound — 2× prefill `max_num_tokens` moved nothing. No lever here; it's the load curve. |

---

## LATENCY POOL  (single-stream / interactive, C≤16)

**decode.yaml deltas vs base:**
```
enable_attention_dp: false          # <-- THE lever: TP attention, ¼ weight-load/rank at M=1
max_batch_size: 32                  # cap LOW — attn-DP=false uses ~4x KV/req/rank; past its
                                    #   cap it hits num_fitting_reqs=0 and does NOT recover
free_gpu_memory_fraction: 0.5       # 0.8 OOMs at startup; 0.5 leaves ~28 GiB for executor
moe_config.warp_decode.policy: auto # 'force' is invalid with the fallback this layout needs
```
**Measured (zero errors, 8/8…64/64):**
```
  C   TTFT_p50   tok/s/user   vs attn-DP=true
  1     1174        96.3        +122%   (43.3)
  2     1236        89.1
  4     1532        83.4        +49%    (56.0)
  8     1864        78.6        +45%    (54.0)
 16     1979        68.2        +28%    (53.2)
 32     3469        41.9        crossover (loses; switch to throughput pool)
```
TTFT is higher (1174 vs 465ms @C=1) but the 2.2× decode rate → **~2× faster end-to-end** at
osl=512. This is also the real **C=1 fix** (the host-sync reorder was structurally blocked;
this beats it 6× and needs no code).

---

## THROUGHPUT POOL  (serving / batch, C≥32)

**decode.yaml deltas vs base:**
```
enable_attention_dp: true           # data-parallel batch split + no attn allreduce wins at scale
max_batch_size: 256                 # was 64 — the real cap-lift
free_gpu_memory_fraction: 0.6
cuda_graph_config.batch_sizes: [...,64,96,128,192,256]
moe_config.warp_decode.policy: auto
```
**Measured:**
```
  C   TTFT_p50   tok/s/user   agg_out_tok/s   vs C=64 cap
  1      460        43.3           45
 32     1407        49.9         1389
 64     1460        44.8         2297        (old ceiling)
128     1410        32.5         3210        +40%  (same TTFT — free)
192     2623        23.3         3513        +53%
256     4117        18.3         3757        +64%  (TTFT 4s = saturation/load)
```
**Sweet spot = C=128: +40% aggregate at the *same* 1.4s TTFT as C=64.** C=192–256 trade TTFT
for the last +24% (batch/offline only). prefill `max_num_tokens` stays 8192 (16384 was neutral).

---

## DEPLOY
Edit the live configmap's `decode.yaml` (see `/tmp/cm_noattndp_lat.json` = latency,
`/tmp/cm_highconc.json` = throughput), `kubectl apply`, then `kubectl delete pod <decode>` to
force a reload. The base env (MK=1 etc.) is unchanged from `BEST_CONFIG.md`.

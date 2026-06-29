# op-trt Perf Campaign + "Both Pools" Analysis — working notes (2026-06-16)

**Status: working notes. NOT committed.** Exact deploy values are in `REGIME_CONFIGS.md` +
`BEST_CONFIG.md` (same dir). Hardware: single 8×B200, no-IB node (`a4-us-002-rl9`).
Model: `BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft` (untrained perf
testbed → validate by tok/s + numerical cosine, never output coherence). Deployment: Dynamo
disagg r20 (`topo-c1-dp2tp4-disagg-r20`), image `optrt-529374445d-vbfuse-nvfp4`, env
`TRTLLM_OPTRT_MOE_MEGAKERNEL=1` + `TRTLLM_FORCE_COMM_METHOD=NVLINK_TWO_SIDED` +
`use_low_precision_moe_combine=false`. All sweeps: osl=512, ISL=2055, tok/s/user + agg tok/s.

---

## TL;DR

1. The decode was **NOT at the hardware floor** — it was **config-capped and batch-starved**.
   The prior "best" (disagg, attn-DP=true, `max_batch=64`) was simultaneously bad-for-latency
   and capped-for-throughput.
2. **Three levers measured.** Concurrency cap → +40–64% aggregate. Parallelism (attn-DP flip)
   → **+122% single-stream (C=1)** — the deepest lever and the real "fix the C=1". Prefill →
   neutral (the high-C TTFT is decode-saturation, not a prefill wall).
3. Latency and throughput want **opposite parallelism** → two regime configs, not one.
4. **"Both" on one 8-GPU node is tight.** Best achievable is either an all-disagg split (if a
   shared prefill can feed 2 decode pools — open question) or a hybrid (aggregated-throughput +
   disagg-latency). Aggregation is *free/favorable* for throughput but *dilutes* the latency win.

---

## 1. The reframe — config-capped, not at-floor

Earlier this campaign concluded "decode is at the memory floor" after 5 neutral per-kernel
experiments (megakernel, M1 a2a, moe_prepare, FlashMLA v_b, nvfp4 tactic) + 2 dead-end upstream
A/Bs (#14476 build defect, #14714 C=1 structurally blocked). **That conclusion was wrong.** The
per-USER rate is near-flat, but the AGGREGATE throughput was capped at C=64 by `max_batch_size=64`,
and the GPU was batch-starved (attn-DP splits per-rank M = concurrency/DP → dense GEMMs ran at
~2% of HBM peak — starved, not BW-bound). The real levers are CONFIG/PARALLELISM, not per-kernel.

---

## 2. The three levers

### #1 — Concurrency cap (+40–64% aggregate, attn-DP=true)
`max_batch_size` 64→256, `free_gpu_memory_fraction` 0.45→0.6, cuda-graph batch_sizes → 256,
`warp_decode.policy` force→auto (required for fallback). Aggregate kept climbing past the old C=64:

| C | agg tok/s | vs C=64 | per-user | TTFT_p50 |
|---|---|---|---|---|
| 64 (old cap) | 2297 | — | 44.8 | 1460ms |
| 128 | **3210** | **+40%** | 32.5 | 1410ms (flat — *free*) |
| 192 | 3513 | +53% | 23.3 | 2623ms |
| 256 | **3757** | **+64%** | 18.3 | 4117ms (saturation) |

**Sweet spot C=128: +40% at the *same* TTFT as C=64.** Past 128, TTFT climbs = decode at
`max_batch` is full (the natural throughput/latency curve, not a fixable bottleneck — see #2).

### #3 — Parallelism: attn-DP is a REGIME KNOB (the big finding)
`enable_attention_dp=false` (TP attention) shards the decode-GEMM weight → each rank loads ¼ the
weight at M=1 → arithmetic intensity ≈ batch (vs batch/DP). Latency config: attn-DP=false,
`max_batch=32`, `free_gpu_memory_fraction=0.5`, `warp_decode.policy=auto`. Zero errors.

| C | attn-DP=true (per-user) | attn-DP=false (per-user) | winner |
|---|---|---|---|
| **1** | 43.3 | **96.3** | **false +122%** |
| 2 | — | 89.1 | false |
| 4 | 56.0 | 83.4 | false +49% |
| 8 | 54.0 | 78.6 | false +45% |
| 16 | 53.2 | 68.2 | false +28% |
| 32 | 51.1 | 41.9 | **true (crossover)** |
| 128 | 32.5 / agg 3210 | 13.0 / agg 1105 | true (false collapses) |

**Latency regime = C≤16; crossover at C=32.** attn-DP=false TTFT is higher (1174 vs 465ms @C=1)
but the 2.2× decode rate wins end-to-end ~2× at osl=512. This is the real C=1 fix — 6× the
blocked host-sync reorder, zero code. NOTE: attn-DP=false uses ~4× KV/req/rank (full batch on
every rank), so it hits `num_fitting_reqs=0` and **doesn't recover** if pushed past its batch cap
— keep it capped low; it is a *latency* config only.

### #2 — Prefill: NEUTRAL (not a lever)
Raised prefill `max_num_tokens` 8192→16384 (~4→8 contexts/step at ISL 2055). A/B was identical
(C=128 3180 vs 3210, C=256 3678 vs 3757; TTFT noise). **The high-C TTFT is decode saturation**
(decode full at `max_batch` when C=`max_batch`), not prefill-throughput-bound — doubling the
prefill rate moved nothing. There is no prefill wall to push; high-C latency is just the load.

---

## 3. The two regime configs

Opposite parallelism → can't be one config. Exact deploy values + data in `REGIME_CONFIGS.md`.
- **Latency pool:** attn-DP=false, batch-32, KV-0.5 → C=1 **96 tok/s (+122%)**, strong ≤C16.
- **Throughput pool:** attn-DP=true, batch-256, KV-0.6 → +40% @C=128 (free), +64% @C=256.

---

## 4. "Both pools" on one 8-GPU node (mixed traffic)

Goal: serve interactive (latency) + batch (throughput) from one node, routed by model-name.
Constraint: the disagg already pairs prefill↔decode and the NIXL transfer converts KV layout,
so different attn configs on prefill vs decode are already normal. The hard limit is **GPU count**.

### 4.1 The aggregated experiment (measured)
Idea: fold prefill into decode (aggregated, `--disaggregation-mode prefill_and_decode`) to free
the prefill's 4 GPUs. Findings:

- **Aggregated THROUGHPUT works great.** A 4-GPU aggregated worker matches/beats the 8-GPU disagg
  (it skips the NIXL hop), and its high-C TTFT is *lower*:

  | C | disagg-8 agg | aggregated-4 agg |
  |---|---|---|
  | 64 | 2297 | 2117–2555 |
  | 128 | 3210 | 3384–3589 (+) |
  | 256 | 3757 | 3430–4128 (≈) |

  So aggregation **halves the throughput footprint at ~no cost** (prefill is ~1–2% of compute).

- **Aggregated LATENCY fails.** Same attn-DP=false config, but aggregated gives **44 not 96** at
  C=1 — the aggregated decode step is ~2.2× slower at M=1 (22.7 vs 10.4 ms/token). The pure-M=1
  weight-sharding win only materializes in a **disagg decode-only** worker. (Config + routing
  verified correct — `enable_attention_dp=False`, reads `lat.yaml`, distinct worker; it's the
  aggregated decode path itself that dilutes.)

- **Reproducible warmup glitch:** the first moderate-batch point (C=32) on a fresh aggregated
  worker stalls (TTFT 12–13s, agg ~700); C=64+ are clean. Looks like a one-time graph/autotune
  capture at that batch. Watch for it; not a steady-state issue.

### 4.2 The GPU budget
A disagg "both" = **3 workers** (prefill + throughput-decode + latency-decode) in 8 GPUs. A
worker's GPU count = its world size; the 345B MoE needs ≥2 GPUs (EP2 = 128 experts/rank fits
~tightly; EP1 OOMs). So two of the three must be 2-GPU EP2 workers. The split that keeps
throughput full: `prefill-2 + tput-decode-4 (TP4) + lat-decode-2 (TP2)` → **latency at TP2 ≈ ~70**
(a real disagg decode-only number, not the aggregated 44; TP4 latency = 96 but needs 4 GPUs +
its own prefill = no room).

### 4.3 THE OPEN QUESTION (decides the best architecture)
**Can one prefill pool feed two decode pools with different `--served-model-name`s?**
- **If YES** → all-disagg "both": `prefill-2 + tput-decode-4 + lat-decode-2`, full throughput
  (TP4) + TP2 latency (~70). No aggregation anywhere. *Preferred if it works.*
- **If NO** (each decode pool needs its own prefill) → 2 prefills + 2 decodes = both decodes
  forced to TP2 → **throughput ~halved**. In that case the **hybrid** wins:
  `aggregated-tput-4 (4128, full) + disagg-latency (prefill-2 + decode-2, ~70)`.

(KV-router is `--router-mode kv` = cache-affinity routing, not SLA/request-type routing, so the
two pools must be distinct model-names to regime-route. Whether a shared prefill serves both
model-names is the unresolved Dynamo-routing detail.)

### 4.4 Two-worker aggregated "both" — MEASURED (proof of concept)
Deployed `{Frontend, decode(tput, aggregated, model X), latency(lat, aggregated, model X-lat)}`,
4 GPUs each, routed by model. Routing WORKED (distinct workers, distinct TTFT). Throughput pool
4128 @C=256. Latency pool 44 @C=1 (the aggregation dilution — confirms latency needs disagg).
So the *topology + routing* are proven; only the latency worker must move to disagg.

---

## 5. Infra learnings (Dynamo / Grove / KAI on k3s)

- **Service topology is IMMUTABLE** — the `vdynamographdeployment.kb.io` webhook denies adding/
  removing services in-place. Topology changes (aggregated, multi-pool) require **DGD delete +
  recreate** (`kubectl delete dynamographdeployment … && kubectl apply -f new.json`).
- **Gang-scheduler gates on `replicas: 0`** — setting a service to 0 replicas leaves its pod-gang
  expecting it → other workers stuck `SchedulingGated`. Remove the service (recreate), don't zero it.
- The DGD live `spec` only exposes `backendFramework`; the real `spec.services` is in the
  `kubectl.kubernetes.io/last-applied-configuration` annotation (operator transforms the rest).
  Build new DGDs from that annotation (saved to `/tmp/dgd_src.json`).
- Config precedence: `--extra-engine-args <yaml>` **overrides** the DGD CLI args (e.g. yaml
  `max_batch_size: 256` beats CLI `--max-batch-size 64`).
- `e2e_sweep.py` now takes `--model` (added this session) to target a pool by served-model-name;
  the copy INSIDE the bench container (`docker cp … optrt-bench-claude:/tmp/e2e_sweep.py`) is what
  the probe runs.

---

## 6. Open decisions / next steps

1. **Answer 4.3** (shared prefill → 2 decode pools?). Either ask (operator knowledge) or build+
   test the all-disagg `prefill-2 + tput-4 + lat-2` and let the both-sweep verify routing.
2. If all-disagg shared-prefill works → ship it (full throughput + ~70 latency, all disagg).
3. If not → ship the hybrid (aggregated-tput-4 + disagg-lat-2).
4. Either way, latency on 8 GPUs is capped at **TP2 (~70)**, not the standalone TP4 (96) — a
   full-strength both needs a 2nd node (clean: latency node TP4 + throughput node).
5. Lab is currently reverted to plain disagg (throughput config). Restore points: disagg DGD
   `/tmp/dgd_src.json`; throughput cm `/tmp/cm_highconc.json`; latency cm `/tmp/cm_noattndp_lat.json`.

## Appendix — sweep files (`.bench_runs_claude/results/`)
`15_revert_verify` (baseline disagg b64) · `18_highconc` (tput b256) · `19_noattndp` +
`21_noattndp_lat` (latency attn-DP=false) · `22_prefill2` (#2 neutral) · `23_aggtput`
(aggregated 1-worker) · `24_both` (2-worker aggregated both).

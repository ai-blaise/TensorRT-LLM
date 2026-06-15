# PDE G8 — MTP / Speculative-Decode Device Loop (results)

**Gate:** keep the spec-decode/MTP draft -> verify -> **variable-length accept**
loop entirely device-side: the data-dependent accept-length `n` (how many of the
`K` drafts pass verification, varying per step) is computed AND consumed on the
device — it never crosses to the host as a control decision. Only accepted tokens
are emitted. Builds on the G9 cross-step device loop (request/completion rings +
device-resident state) + G3 device-side data-dependent control flow + the G0
substrate.

Node 001, GPU 0 (NVIDIA B200, 148 SMs, cc 10.0), branch `op-trt-pde`. Standalone
`nvcc 13.1 -arch=sm_100`. Model-free synthetic draft/verify; ABI-frozen
decode/model path untouched.

## Design

### (A) device MTP engine — `kDeviceMtpEngine`, ONE `cudaLaunchCooperativeKernel`
Per step, per active slot, all device-side (no host contact mid-run):
1. **draft** `K` candidate tokens for positions `pos..pos+K-1`,
2. **verify** them (representative GEMV stand-in over the candidate window),
3. **decide** the data-dependent accept-length `n` via `decide_step()` — the
   count of leading draft candidates that match the target's greedy token (the
   standard greedy spec-decode acceptance rule),
4. **advance** the per-slot position by the VARIABLE `n_emit = n+1` (accepted
   prefix + one corrected/bonus target token, clamped to remaining) and the
   target hash-chain state by `n_emit` tokens — ON-DEVICE (the data-dependent
   jump),
5. **emit** the `n_emit` accepted/bonus tokens to a device->host completion ring
   (MPSC reserve-via-atomicAdd, per-slot release stamp, exactly-once),
6. **retire** the slot device-side when `pos` reaches `gen_len`.

The accept-length is the device's branch input; it is **never d2h'd as a control
value**. It rides along on each emitted completion only as device->host telemetry
(so the host/CPU can verify it). Admission (host request ring) + termination
(progress counter + done flag) via the G9 pattern with `grid.sync()` region
barriers.

### (B) host-orchestrated baseline — `kHostMtpStep`, ONE launch per step
Today's spec-decode control pattern: the kernel computes the accept-length +
emitted tokens but **does NOT advance** `pos`/`hstate`. The host **d2h's the
accept-length**, **branches on it** (advance `pos` by `n_emit`, set `hstate`,
retire finished slots, rebuild the active list + next draft window), **h2d's** the
advanced state, and **relaunches**. The d2h-of-a-control-value + host branch + h2d
each step is exactly what is CUDA-graph-capture-illegal and is the overhead G8
eliminates.

### Determinism (the bit-exact gate)
- Target tokens = splitmix64 integer hash-chain keyed on `(request_id, pos)`,
  position-folded so a dropped/dup token diverges immediately. Identical
  recurrence to G9 -> a request's emitted sequence is **output-equivalent to a
  non-speculative greedy decode** (checked: `A == pure-target`).
- A deterministic per-`(request,pos)` agreement predicate with tunable
  `p_acc = num/den` gives a controllable, **variable** accept-length without any
  floats -> fully reproducible.
- CPU reference replays the identical MTP loop (same `decide_step`, same target
  chain). The gate asserts `A == B == CPU` bit-exact on BOTH the emitted-token
  stream AND the per-step accept-length sequence, with variable-n actually
  exercised.

## Gate results (REAL, GPU0)

Occupancy: `kDeviceMtpEngine` **6 blk/SM = 888 resident CTAs**, 40 regs, 0.5 KB
smem. `p_acc = 7/10` (representative EAGLE-class acceptance).

### Correctness — PASS (all scenarios, bit-exact vs independent CPU + B)

| scenario | reqs | tokens | MTP steps | accept-len hist (n=0..K) | mean n | A==B==CPU toks | A==B==CPU accept-len | exactly-once | greedy-equiv | variable-n |
|---|---|---|---|---|---|---|---|---|---|---|
| N16_S8_stag4_K4 | 16 | 432 | 154 | 37,42,23,18,34 | 1.805 | PASS | PASS | PASS | PASS | yes |
| N32_S16_stag8_K4 | 32 | 1424 | 524 | 161,120,69,54,120 | 1.718 | PASS | PASS | PASS | PASS | yes |
| N8_S8_nostag_K6 | 8 | 599 | 204 | 68,39,30,22,16,8,21 | 1.936 | PASS | PASS | PASS | PASS | yes |

The accept-length distribution spans the full `[0, K]` range every scenario (the
variable-n control path is genuinely exercised, not pinned to 0 or K). Zero
drops, zero dupes, zero mismatches on both rings. Internal consistency: the K=1
steady case gives mean accept = 0.70 == p_acc, confirming the accept logic.

### Perf — staggered admission (per-token, end-to-end incl. host arrival stagger)

| scenario | A device us/tok | B host-orch us/tok | speedup | round-trip cut us/tok |
|---|---|---|---|---|
| N16_S8_stag4_K4 | 2.3508 | 4.3190 | **1.84x** | 1.97 |
| N32_S16_stag8_K4 | 1.0991 | 2.0489 | **1.86x** | 0.95 |
| N8_S8_nostag_K6 | 1.8879 | 3.6302 | **1.92x** | 1.74 |

### Perf — steady-state per-step (clean isolation, fixed batch, gen_len=512, no admission)

| config | A us/step | B us/step | **per-step overhead cut** | speedup | A us/tok | B us/tok | mean accept | steps/req |
|---|---|---|---|---|---|---|---|---|
| B1_g512_K4 | 20.71 | 71.87 | **51.15 us/step** | **3.47x** | 8.011 | 27.792 | 1.59 | 198.0 |
| B8_g512_K4 | 32.17 | 70.42 | 38.25 us/step | 2.19x | 1.457 | 3.189 | 1.76 | 185.5 |
| B32_g512_K4 | 40.66 | 71.79 | 31.12 us/step | 1.77x | 0.456 | 0.805 | 1.79 | 183.7 |
| B128_g512_K4 | 52.93 | 79.47 | 26.54 us/step | 1.50x | 0.149 | 0.223 | 1.78 | 183.9 |
| B32_g512_K1 | 27.22 | 68.74 | 41.52 us/step | 2.53x | 0.499 | 1.260 | 0.70 | 300.3 |
| B32_g512_K8 | 51.38 | 76.68 | 25.30 us/step | 1.49x | 0.501 | 0.748 | 2.21 | 159.8 |

**The eliminated overhead** = per-step kernel relaunch + accept-length d2h + host
branch + advanced-state h2d. It is a **per-step constant ~25–51 us** that the
device loop removes outright (B's per-step time is ~flat at 68–79 us regardless of
batch; A's grows with batch as real compute but stays far below B).

### Scaling

- **vs batch B** (K=4): the per-step cut shrinks 51 -> 27 us as B grows 1 -> 128,
  because A's per-step compute rises with batch while B's host round-trip is
  ~batch-independent; the win is largest at small batch (latency-bound decode,
  the tok/s/user regime) — **3.47x at B=1**.
- **vs draft length K** (B=32): K=1 -> 41.5 us cut (2.53x), K=4 -> 31.1 (1.77x),
  K=8 -> 25.3 (1.49x). Larger K accepts more per step (mean accept 0.70 -> 2.21)
  so fewer steps/req (300 -> 160), amortizing both A and B's per-step cost over
  more tokens; the absolute per-step round-trip removed stays large. **Per-token**,
  the device loop is 2.5x faster at K=1 and 1.5x at K=8.

## Honest notes / negatives

- The verify "compute" is a synthetic GEMV stand-in, not a real target-model
  forward — G8 isolates the **control-flow** win (device-side variable-accept),
  not GEMM throughput. The eliminated overhead (relaunch + d2h-control + branch +
  h2d) is real and model-independent; the absolute A us/step would grow once a
  real forward replaces the stand-in, which would only *increase* the relative
  value of removing the per-step launch (B pays it every step).
- The staggered per-token numbers include host arrival pacing (the engine spins
  waiting for the host producer), so the steady-state table is the cleaner
  per-step-overhead isolation; both are reported.
- A's wall in the staggered runs is host-pacing-bound at the configured 20 us
  stagger; the win there (1.8–1.9x) is a conservative lower bound on the
  control-loop benefit.

## Files

- `cpp/tensorrt_llm/kernels/pde/pde_g8_mtp.cuh` — device MTP engine + host-orch
  step kernel + deterministic draft/verify/`decide_step`.
- `blaise_perf/pde/pde_g8_bench.cu` — CPU reference, A/B drivers, correctness
  (tokens + accept-len + exactly-once + greedy-equiv + variable-n), steady-state
  per-step + K/batch scaling, SUMMARY_JSON.
- `blaise_perf/pde/build_run_g8.sh` — flock + SIGKILL-bounded GPU0 build+run.

## SUMMARY_JSON (verbatim from the run)

```json
{"sm":148,"occ_engine":6,"resident_ctas":888,"regs":40,"pacc_num":7,"pacc_den":10,"scen":["N16_S8_stag4_K4","N32_S16_stag8_K4","N8_S8_nostag_K6"],"A_pertok_us":[2.3508,1.0991,1.8879],"B_pertok_us":[4.3190,2.0489,3.6302],"speedup":[1.837,1.864,1.923],"overhead_cut_us":[1.9682,0.9498,1.7423],"A_comps":[432,1424,599],"B_comps":[432,1424,599],"corr_per_scen":[1,1,1],"scen0_K":4,"scen0_accept_hist":[37,42,23,18,34,0],"scen0_steps":154,"ss_name":["B1_g512_K4","B8_g512_K4","B32_g512_K4","B128_g512_K4","B32_g512_K1","B32_g512_K8"],"ss_A_perstep_us":[20.7139,32.1744,40.6647,52.9336,27.2197,51.3761],"ss_B_perstep_us":[71.8662,70.4235,71.7863,79.4734,68.7423,76.6796],"ss_overhead_cut_us":[51.1523,38.2491,31.1216,26.5398,41.5226,25.3036],"ss_speedup":[3.469,2.189,1.765,1.501,2.525,1.493],"ss_mean_accept":[1.586,1.760,1.787,1.785,0.705,2.205],"ss_steps_per_req":[198.0,185.5,183.7,183.9,300.3,159.8],"corr_gate":"PASS"}
```

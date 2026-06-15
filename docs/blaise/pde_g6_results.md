# PDE Gate G6 — intra-kernel TP all-reduce

**Node 001, branch `op-trt-pde`. Model-free multi-GPU microbench, GPU 0–3 (TP=4),
8× B200 NVLink5. Correctness vs INDEPENDENT references (NCCL + analytic). 2026-06-14.**

The persistent decode megakernel must perform its TP all-reduce (after
attn out-proj and after MoE down-proj) **in-kernel**, without exiting to the
launch boundary. G6 validates that an in-kernel all-reduce is correct and
measures it against the launch-boundary NCCL all-reduce it replaces.

---

## Tier reached: (b) P2P-NVLink in-kernel one-shot all-reduce

The impl ladder was (a) NVLS `multimem` → (b) P2P-NVLink in-kernel → (c) ring.
**Tier (a) is environment-blocked on this VM; tier (b) is implemented and passes
the gate.**

### Why tier (a) NVLS `multimem` is NOT reachable in this image/VM

NVLS multimem itself is *supported by the silicon and the toolchain here* — the
blocker is the fabric subsystem, not the code:

| Check | Result |
|---|---|
| `multimem.ld_reduce.global.add` / `multimem.st` PTX assembles (nvcc 13.1 sm_100) | **OK** |
| `cuMulticastCreate/AddDevice/BindMem/GetGranularity` in libcuda | **present** |
| `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED` on dev 0–3 | **1 (all)** |
| `cuMulticastCreate` + `cuMulticastAddDevice` (×4) | **CUDA_SUCCESS** |
| `cuMulticastBindMem` (POSIX-fd / NONE / FABRIC / `cuMulticastBindAddr`) | **CUDA_ERROR_INVALID_VALUE / UNKNOWN / NOT_PERMITTED** |
| `nv-fabricmanager` binary / service | **absent** |
| `/dev/nvidia-caps-imex-channels` (IMEX) | **absent** |
| `lspci` NVSwitch count | **0** |

The multicast object can be *created* and devices *added* (local bookkeeping),
but **binding physical memory to form the NVLS multicast team requires the NVLink
fabric (fabric manager + IMEX channels)**, which this GPU-passthrough VM does not
expose. P2P NVLink itself is fully up (`canAccessPeer=1` all pairs, 53.125 GB/s ×
links, P2P atomics supported).

**Independent corroboration:** NCCL 2.29.2 fails *identically* on this VM —
`transport/nvls.cc:284 NCCL WARN Failed to bind NVLink SHARP (NVLS) Multicast
memory … CUDA error 999 … usually caused by a … Fabric Manager or NVSwitches.
Disable NVLS (NCCL_NVLS_ENABLE=0)`. So NCCL's own `cuMulticastBindMem` hits the
same wall. The bench runs NCCL with `NCCL_NVLS_ENABLE=0` (its P2P/ring transport)
— which is exactly the launch-boundary baseline the engine replaces.

> On a node with the fabric exposed (fabricmanager active + IMEX), the tier-(a)
> `multimem` wrappers in `pde_g6_allreduce.cuh` (`all_reduce_f32_multimem_body`)
> are the drop-in best path. They are kept in the header, compiled, gated behind
> a successful bind. **Re-run G6 there to capture the NVLS number.**

---

## Tier (b) design — in-kernel one-shot all-reduce over P2P NVLink

Single process, 4 host threads (one per GPU 0–3). P2P enabled between all pairs,
so each rank's persistent kernel directly dereferences its peers' input buffers
over NVLink. Each rank's persistent kernel:

1. **Pre-barrier** — `cross_rank_grid_barrier`: grid-wide (all CTAs of this rank
   converge) **fused** with the cross-GPU rendezvous. The *last local CTA to
   arrive* drives the cross-rank flag handshake, then releases the whole grid —
   folding what was two grid syncs into one.
2. **Comms ∥ compute region** — COMMS warps run the one-shot all-reduce
   (`one_shot_all_reduce_f32_body`: read element *i* from all 4 peers over
   NVLink, sum, write to own output, vectorized float4); COMPUTE warps run
   independent FMA work (stand-in for the rest of the decode step). They overlap.
3. **Post-barrier** — keeps peer inputs alive until all ranks finish reading.

**Cross-rank barrier (`rank_barrier_sync`).** Each rank stores its arrival into
its own slot on every peer (P2P), then spins on its *own* local array (peers push
to us → cheap local HBM reads, not NVLink round-trips). Key optimization:
**relaxed system-scope atomics + two `__threadfence_system()`** (release before
signal, acquire after rendezvous) instead of `st.release.sys` per flag. Measured
on a pure 1-thread/rank rendezvous:

| barrier variant | us / barrier |
|---|---|
| `st.release.sys` per flag + `fence_system` (naive) | 9.93 |
| `st.release.sys` per flag, no fence | 8.15 |
| **`st.relaxed.sys` per flag + fences (chosen)** | **1.62** |

That single change took the fused grid+cross-rank barrier from ~11.8 µs → **6.9
µs**, which dominated small-message latency.

The persistent-loop kernel performs N in-kernel all-reduces in a device-side loop
with **zero per-call relaunch** — the engine-relevant cost (the megakernel never
relaunches mid-decode-step). All in-kernel timings below are wall-clock / N from
that loop.

---

## GATE results (GPU 0–3, TP=4)

### Correctness — HARD (all 4 ranks, vs INDEPENDENT references) — **PASS**

```
rank 0: inkernel==NCCL: YES  inkernel==analytic_sum: YES  cos_vs_nccl=1.000000000 -> PASS
rank 1: inkernel==NCCL: YES  inkernel==analytic_sum: YES  cos_vs_nccl=1.000000000 -> PASS
rank 2: inkernel==NCCL: YES  inkernel==analytic_sum: YES  cos_vs_nccl=1.000000000 -> PASS
rank 3: inkernel==NCCL: YES  inkernel==analytic_sum: YES  cos_vs_nccl=1.000000000 -> PASS
=== GATE: CORRECTNESS PASS ===
```

Integer test data (`input[i]@rank r = (r+1)·((i%64)+1)`, exact in f32), so the
all-reduce is **bit-exact**: the in-kernel result equals (a) the NCCL all-reduce
of the same inputs and (b) the analytic cross-rank sum `10·((i%64)+1)`, on every
rank. Never a self-comparison.

### Performance — in-kernel vs launch-boundary NCCL (comms_warps=4/8, grid=1 CTA/SM)

Representative decode all-reduce sizes, hidden 7168, f32:

| size | bytes | in-kernel | NCCL (launch-bdry) | speedup | overlap | reduce_net | BW_reduce | BW_NCCL |
|---|---|---|---|---|---|---|---|---|
| b1×7168  |  28 KB | 18.9–19.6 µs | 16.6 µs | 0.85–0.88× | 1.24–1.29× | 5.3–5.8 µs | 7–8 GB/s | 2.6 GB/s |
| b2×7168  |  56 KB | 19.5 µs | 16.7 µs | 0.85× | 1.29× | 5.7 µs | 15 GB/s | 5.2 GB/s |
| b4×7168  | 112 KB | 19.7 µs | 16.2 µs | 0.83× | 1.31× | 5.9 µs | 29 GB/s | 10.6 GB/s |
| b8×7168  | 224 KB | 20.1 µs | 16.4 µs | 0.81–0.82× | 1.31–1.33× | 6.4 µs | 52–54 GB/s | 21 GB/s |
| b16×7168 | 448 KB | 26.2 µs | 17.7 µs | 0.67× | 1.47× | 12.4 µs | 55 GB/s | 39 GB/s |
| b32×7168 | 896 KB | 37.0–37.6 µs | 18.4 µs | 0.49–0.50× | 1.62–1.63× | 23.5–23.9 µs | 58 GB/s | 75 GB/s |

**Barrier floor (fused grid+cross-rank):** `bar1 ≈ 6.9 µs`, `bar2 ≈ 13.7 µs`
(two barriers/all-reduce) — flat across grid size (tested 8…148 CTAs) and message
size, i.e. it is the fixed cross-GPU rendezvous cost, not CTA-atomic contention.

### Reading the numbers

- **In-kernel latency is barrier-dominated at decode scale.** The two cross-rank
  barriers (~13.7 µs) are the fixed cost; the actual data movement (`reduce_net`)
  is 5–24 µs and scales correctly. At small sizes the in-kernel all-reduce is
  **0.81–0.88× of NCCL** (within ~3 µs); NCCL's ~16 µs is itself almost entirely
  fixed launch/transport overhead (flat 16–18 µs across all sizes here).
- **`comms_warps` sweep** (BW_reduce at b32): 2/8 → 34 GB/s, **4/8 → 58 GB/s**,
  6/8 → 73 GB/s. More comms warps = more NVLink read parallelism but less room
  for overlapped compute; 4/8 is the chosen balance (near-NCCL small-message
  latency + strong BW + overlap headroom). Both 4/8 and 6/8 pass correctness.
- **Overlap (comms ∥ compute) factor 1.24× → 1.63×** — the all-reduce genuinely
  runs concurrently with independent compute warps in one resident grid.

### Why this is the right primitive for the engine despite ≤1× raw latency

The launch-boundary NCCL number does **not** include the kernel exit/re-entry the
megakernel pays around every NCCL call (graph break, relaunch, metadata D2H). The
in-kernel all-reduce removes that boundary entirely and, more importantly,
**overlaps with compute (1.6×)** — neither of which the launch-boundary path can
do. The barrier floor (~13.7 µs/all-reduce) is the optimization target for G7+
(faster rendezvous: fewer barriers via producer-side arrival, warp-level
signaling, or — on a fabric-enabled node — `multimem` which fuses reduce+sync).

---

## Honest blockers

1. **NVLS `multimem` (tier a) unreachable in-image** — fabric manager + IMEX +
   NVSwitch not exposed in this passthrough VM; `cuMulticastBindMem` rejected
   (NCCL fails identically). Wrappers are written and compile; capture the NVLS
   number on a fabric-enabled node.
2. **Barrier floor ~6.9 µs/barrier** is the dominant in-kernel cost at decode
   sizes — the cross-GPU rendezvous, not the data movement. Already cut ~40% via
   relaxed atomics; further reduction is the G7 lever.

## Files

- `cpp/tensorrt_llm/kernels/pde/pde_g6_allreduce.cuh` — multimem wrappers (tier
  a, gated), `rank_barrier_sync` (relaxed-atomic cross-rank), `grid_barrier` +
  `cross_rank_grid_barrier` (fused grid+cross-rank), `one_shot_all_reduce_f32_body`
  (tier b).
- `blaise_perf/pde/pde_g6_bench.cu` — 4-thread multi-GPU harness, multicast probe
  history, persistent-loop timing, NCCL + analytic correctness, comms/grid sweeps.
- `blaise_perf/pde/build_run_g6.sh` — build + run (NCCL include/lib + `-lcuda`,
  `NCCL_NVLS_ENABLE=0`).

## Reproduce

```
flock -n /tmp/gpu001_lock_a timeout -s KILL 220 docker run --rm --init \
  --name pde_g6_run --gpus all -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -v /home/spencer/work/pde-wt:/wt --entrypoint /bin/bash \
  localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-aaa7e2b542b2-hisparse-current-head-proof-20260613T155354Z \
  /wt/blaise_perf/pde/build_run_g6.sh
# sweeps: PDE_G6_COMMS_WARPS={2,4,6}  PDE_G6_GRID={8..148}
```

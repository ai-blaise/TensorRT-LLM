# PDE Gate G7 — intra-kernel EP all-to-all (expert dispatch + combine)

**Node 001, branch `op-trt-pde`. Model-free multi-GPU microbench, GPU 0–3 (EP=4),
8× B200 NVLink5. Correctness vs INDEPENDENT references (NCCL grouped a2a + host
gather-route-scatter). 2026-06-14.**

(Results table filled by the bench run — see "GATE results" below.)

The MoE layer routes each token to the rank(s) owning its top-k expert(s) and
must perform the resulting expert **all-to-all in-kernel**, without exiting to
the launch boundary, so the persistent decode megakernel can span the EP
collective. G7 validates that an in-kernel **data-dependent** dispatch+combine is
correct and measures it against the launch-boundary NCCL all-to-all it replaces.

---

## Tier reached: (b) P2P-NVLink in-kernel push all-to-all

Impl ladder (a) NVLS `multimem` → (b) P2P-NVLink push a2a → (c) staged P2P.
**Tier (b) is implemented and passes the gate.**

- **(a) NVLS `multimem` is the wrong primitive AND environment-blocked.** `multimem`
  is a *reduction* primitive (fabric ld_reduce/st across a multicast team). An
  expert all-to-all moves *disjoint* per-(src,dst) payloads with no reduction, so
  multimem buys nothing even where it binds. And it does not bind here:
  `cuMulticastBindMem` is rejected on this GPU-passthrough VM (no
  fabric-manager / IMEX / NVSwitch — verified in G6, with NCCL failing
  identically). Not applicable on both counts.
- **(b) P2P-NVLink push a2a** is the implemented tier: each rank writes its tokens
  straight into the destination rank's recv buffer over peer-mapped NVLink
  pointers at device-computed offsets, then scatters expert outputs back the same
  way. Fully in-kernel, fully model-free.
- **(c) staged P2P** (explicit local stage before push) is subsumed: the
  destination recv buffer *is* the stage; the direct push needs no extra copy.

---

## DATA-DEPENDENT counts/offsets — handled entirely ON-DEVICE (the crux vs G6)

Unlike G6's fixed-size all-reduce, an EP a2a has variable per-(src,dst) counts
set by data-dependent routing. Everything below is computed on-device with no
host involvement and no `cudaMemcpy` between ranks:

1. **count** — each rank histograms its tokens by destination rank into its own
   row `matrix[my_rank][*]` (atomic, world entries). The router is a synthetic
   deterministic top-1 expert map; owning rank = `expert/(E/world)`; skewed so
   counts are genuinely variable.
2. **publish** — each rank pushes its send-count row into row `[my_rank]` of
   EVERY peer's P2P-shared `world×world` matrix (relaxed system-scope stores,
   ordered by the cross-rank barrier). After the barrier every rank holds the
   full matrix and derives `recv_count[dst][src] = send_count[src][dst]`.
3. **offsets** — exclusive prefix sums over the matrix give, on-device: our
   local pack base per dst (`send_base`), our payload base inside each
   destination's recv region (`recv_base_at_dst` = column-prefix up to our rank),
   and our per-source recv counts/total. The push lands tokens contiguously with
   no gaps under skew.
4. **pack + push** — one thread per token claims a packed slot via an atomic
   cursor seeded from `send_base` (race-free slot assignment), the grid copies
   payloads into packed order, then a grid-strided NVLink store writes each
   block into its destination's recv region at the device-computed base. A
   per-recv-slot meta word `(src_rank<<20 | src_tok)` rides along so **combine**
   can scatter expert outputs back to the exact origin token.

Pipeline (one in-kernel pass): `count → publish → derive → pack-assign →
pack-copy → DISPATCH-push → expert → COMBINE-push`. A stage boundary uses a
**cross-rank** barrier only at the 3 peer-read points (publish→derive,
dispatch→expert, combine→end — where a peer's P2P write must be visible) and a
cheaper **intra-grid** barrier at the 5 local-only transitions. Built on G6's
verified `rank_barrier_sync` / `cross_rank_grid_barrier` (relaxed-atomic
cross-rank rendezvous) + `grid_barrier` (intra-grid) + the G0 substrate,
unchanged.

---

## GATE results (GPU 0–3, EP=4)

### Correctness — HARD (all 4 ranks, vs INDEPENDENT references) — **PASS**

```
rank 0: inkernel==NCCL_a2a: YES  inkernel==host: YES  route_matrix_ok: YES  cos_vs_host=1.000000000 -> PASS
rank 1: inkernel==NCCL_a2a: YES  inkernel==host: YES  route_matrix_ok: YES  cos_vs_host=1.000000000 -> PASS
rank 2: inkernel==NCCL_a2a: YES  inkernel==host: YES  route_matrix_ok: YES  cos_vs_host=1.000000000 -> PASS
rank 3: inkernel==NCCL_a2a: YES  inkernel==host: YES  route_matrix_ok: YES  cos_vs_host=1.000000000 -> PASS
=== GATE: CORRECTNESS PASS ===
```

Integer payloads (`payload_val(rank,t,c)` distinct per token/element), expert
transform `out = in*2 + 1`. The in-kernel combined result is checked against:
(i) an **NCCL grouped `ncclSend`/`ncclRecv`** all-to-all with the same variable
counts (host-packed, host expert transform, returned to origin — fully
independent of the in-kernel push path) and (ii) a **host gather-route-scatter**.
Both give bit-exact agreement (`cos=1.000000000`) on every rank. Never a
self-comparison.

**Data-dependent routing — per-(src,dst) count matrix (T=8, device-published,
checked vs analytic router):**

```
--- per-(src,dst) count matrix [T8_h512] (rows=src, cols=dst) ---
  src0:    5    0    3    0   (src0 sends 8/8)
  src1:    2    2    3    1   (src1 sends 8/8)
  src2:    2    2    3    1   (src2 sends 8/8)
  src3:    2    1    2    3   (src3 sends 8/8)
  recv:   11    5   11    5   (dst totals)
  conservation: sum_send=32 sum_recv=32 expected=32 -> OK (no drop/dupe)
```

The counts are genuinely variable/skewed (src0 sends **0** tokens to ranks 1 & 3
but 5 to itself; dst totals 11/5/11/5 — non-uniform), the device-published matrix
**matches the analytic router exactly** on every entry, and global conservation
holds (`sum_send == sum_recv == world·T = 32`), so **no token is dropped,
duplicated, or misrouted**. The on-device offset derivation lands every block
contiguously under this skew.

### Performance — in-kernel vs launch-boundary NCCL all-to-all (grid = 1 CTA/SM)

Representative decode MoE a2a sizes (`hidden` = ints/token, `T` = tokens/rank).
`ik` = full in-kernel pipeline; `nccl` = launch-boundary NCCL grouped a2a
(dispatch + combine); `bars` = the pipeline barrier floor (**3 cross-rank + 5
intra-grid**, see lever below); `move_net` = `ik − nocomms` (net data movement,
floor subtracted); `BW_move` = NVLink BW over that net (dispatch+combine bytes);
`ov` = warp-specialized dispatch∥compute overlap factor.

| size | hidden | in-kernel | NCCL (launch-bdry) | speedup | overlap | bars (3+5) | move_net | BW_move | BW_NCCL |
|---|---|---|---|---|---|---|---|---|---|
| T8   | 512  | 29.2 µs | 20.3 µs | 0.70× | 1.05× | 25.3 µs | 1.4 µs | 27 GB/s | 1.8 GB/s |
| T16  | 1024 | 29.4 µs | 28.5 µs | 0.97× | 1.05× | 25.3 µs | 1.5 µs | 104 GB/s | 5.5 GB/s |
| T32  | 1024 | 29.7 µs | 30.7 µs | **1.03×** | 1.07× | 25.3 µs | 1.9 µs | 155 GB/s | 9.6 GB/s |
| T64  | 2048 | 34.2 µs | 35.1 µs | **1.03×** | 1.16× | 25.3 µs | 6.4 µs | 211 GB/s | 38 GB/s |
| T128 | 2048 | 39.3 µs | 36.4 µs | 0.93× | 1.24× | 25.2 µs | 11.4 µs | 242 GB/s | 76 GB/s |
| T256 | 2048 | 49.6 µs | 37.4 µs | 0.75× | 1.37× | 25.4 µs | 21.7 µs | **241 GB/s** | 140 GB/s |

**Barrier-count lever (the headline G7 optimization).** Naively the 8-stage
pipeline takes 8 cross-rank barriers (~50 µs floor, ≈6.25 µs each). But a stage
transition needs a *cross-rank* barrier ONLY when the next stage reads data a
PEER produced over P2P — exactly 3 points (publish→derive, dispatch→expert,
combine→end). The other 5 transitions are local and need just an *intra-grid*
barrier (no NVLink rendezvous). Splitting them **cut the floor ~50 µs → ~25 µs**
and the full in-kernel a2a from 55–76 µs → 29–50 µs, putting it at **NCCL parity
(0.97–1.03×) across the mid decode range** while still overlapping compute and
never exiting the kernel. Correctness unchanged (bit-exact, all ranks).

**Overlap-window decomposition** (mode-2 microbench: dispatch∥compute isolated
between 2 barriers, warp-specialized — half warps push over NVLink, half do
independent FMA):

| size | window floor (2 bars) | dispatch_net | compute_net | both | serial | overlap |
|---|---|---|---|---|---|---|
| T8   | 12.9 µs | 0.6 µs | 0.7 µs | 13.6 µs | 14.2 µs | 1.05× |
| T64  | 12.9 µs | 2.9 µs | 2.9 µs | 16.3 µs | 18.8 µs | 1.15× |
| T128 | 13.0 µs | 5.4 µs | 5.3 µs | 19.1 µs | 23.7 µs | 1.24× |
| T256 | 13.0 µs | 10.1 µs | 10.1 µs | 24.2 µs | 33.2 µs | **1.37×** |

### Reading the numbers

- **At/above NCCL parity across the mid decode range.** After the barrier-count
  lever the in-kernel a2a is **0.97–1.03× of launch-boundary NCCL** at T16–T64
  (the decode-relevant band), and within 0.70–0.93× at the extremes (T8 is
  barrier-floor-bound; T256 favors NCCL's ring BW). And NCCL's number does NOT
  include the kernel exit/re-entry per collective (graph break, relaunch,
  metadata D2H) that the megakernel cannot afford mid-decode-step — the same
  caveat as G6.
- **The push a2a itself is BW-efficient.** `move_net` is 1.4–22 µs and scales
  correctly with payload, reaching **241 GB/s** of NVLink bus BW at T128/T256.
  The residual latency is the ~25 µs barrier floor (3 cross-rank rendezvous),
  not the data movement.
- **Overlap (dispatch∥compute) 1.05× → 1.37×**, growing with payload exactly as
  expected (more comms to hide behind compute). Warp specialization is what makes
  it real: with all warps doing push-then-FMA sequentially the overlap was 1.00×
  (measured); splitting comms vs compute warps recovers the concurrency.

### Why this is the right primitive for the engine despite ≤1× raw latency

The launch-boundary NCCL number excludes the kernel exit/re-entry the megakernel
pays around every NCCL a2a; the in-kernel push removes that boundary entirely and
keeps the whole MoE dispatch+expert+combine device-resident, which is the entire
point of the persistent decode engine (it lets the megakernel span the EP
collective without the capture-illegal D2H wall). The barrier floor (8 × ~6.25 µs)
is the optimization target for G7+ — see below.

---

## Honest blockers

1. **NVLS `multimem` (tier a) is N/A for a2a + unreachable in-image** — it is a
   reduction primitive (no benefit for disjoint a2a payloads) and
   `cuMulticastBindMem` is fabric-blocked on this passthrough VM (NCCL fails
   identically; see G6). The reduction path's multimem wrappers stay in
   `pde_g6_allreduce.cuh` for the TP all-reduce on a fabric-enabled node.
2. **Remaining barrier floor ~25 µs (3 cross-rank rendezvous) is the dominant
   in-kernel cost at small decode sizes** — already cut from ~50 µs by the
   local-vs-cross-rank split (above). The 3 remaining cross-rank barriers
   (publish→derive, dispatch→expert, combine→end) are genuine peer-read points.
   Further G7+ levers: (i) overlap the dispatch push directly into the combine
   path (skip the dispatch→expert→combine round-trip by computing experts as
   tokens arrive); (ii) a producer-side / warp-level signal instead of the full
   grid+cross-rank rendezvous (the G6 barrier lever) to cut each cross-rank
   barrier's ~6.25 µs; (iii) on a fabric-enabled node, a one-shot NVLS-assisted
   rendezvous. These would push the in-kernel a2a clearly below NCCL across all
   sizes while keeping the overlap + no-kernel-exit wins.

## Files

- `cpp/tensorrt_llm/kernels/pde/pde_g7_alltoall.cuh` — in-kernel EP a2a: on-device
  count/publish/offset derivation (`count_by_dst`, `publish_send_counts`,
  `derive_offsets`), local pack, `dispatch_push_body`, synthetic
  `expert_compute_body`, `combine_push_body`. Reuses G6's cross-rank barrier.
- `blaise_perf/pde/pde_g7_bench.cu` — 4-thread multi-GPU harness, synthetic
  data-dependent router, in-kernel a2a kernel + persistent-loop timing, NCCL
  grouped-a2a + host references, count-matrix routing check.
- `blaise_perf/pde/build_run_g7.sh` — build + run (NCCL include/lib + `-lcuda`,
  `NCCL_NVLS_ENABLE=0`).

## Reproduce

```
flock -n /tmp/gpu001_lock_a timeout -s KILL 220 docker run --rm --init \
  --name pde_g7_run --gpus all -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -v /home/spencer/work/pde-wt:/wt --entrypoint /bin/bash \
  localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-aaa7e2b542b2-hisparse-current-head-proof-20260613T155354Z \
  /wt/blaise_perf/pde/build_run_g7.sh
# sweeps: PDE_G7_GRID={8..148}
```

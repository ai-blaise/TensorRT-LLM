# PDE Gate G9 — Cross-Step Persistence

**Node 001, branch `op-trt-pde`, GPU0 (NVIDIA B200, sm_100, 148 SMs), nvcc 13.1.**
Model-free single-GPU microbench. Correctness vs an independent reference (the
per-step-relaunch baseline + a CPU recompute), never a self-compare.

## What G9 proves

Every prior gate (G0..G6) still **relaunched the forward once per decode step**:
the host loop does, every step, a kernel launch + a decode-metadata rebuild
(active-request list, positions, seq-lens) + a host<->device sync to read/advance
state. That per-step host round-trip is the bulk of the decode "execution gap"
(decode ~20.8 ms/tok = 20-40x the HBM-BW floor; almost all of it is per-step
overhead, not compute).

G9 runs the decode loop **device-side across steps**: **ONE persistent
cooperative launch** that loops over steps internally — each step polls a
device-visible request ring, runs a synthetic-but-representative per-step decode
compute over the active slots, **advances a device-resident decode state on-GPU
(no host metadata rebuild)**, and emits the step's tokens into a device->host
completion ring. The host shrinks to **admission only**: enqueue requests over
time + drain completions. No per-step relaunch, no per-step metadata rebuild, no
per-step state sync.

## Design

### Device-visible request ring (host producer -> device consumer)
SPSC ring in **host-mapped pinned memory** (`cudaHostAllocMapped` +
`cudaHostGetDevicePointer`). The host writes a `RequestDesc{request_id, slot,
gen_len}` into `buf[tail % cap]` then publishes by bumping `tail` with a
**system-scope release** store; the engine's controller CTA reads `tail` with a
**system-scope acquire** load, consumes `[head, tail)`, and bumps `head`
(release) so the host can reclaim space. A host->device `done_flag` (system-scope)
is set once the host has enqueued its last request.

### Device-resident decode state (advanced on-GPU, never host-rebuilt)
Per slot: `occupied`, `req_id`, `pos`, `gen_len`, `hstate` (token hash-chain
state), and a `svec[kD]` float state vector for the GEMV stand-in. Every step the
engine advances `pos` and `hstate` **on the device**; the host is never consulted
for metadata.

### Device->host completion ring (device producer -> host consumer), MPSC
Every active compute CTA is a producer. It reserves a **unique** slot with an
atomic add on a shared `prod` counter (exactly-once, no inter-producer
coordination), writes the `Completion{request_id, token, slot, pos}` payload,
then stamps a **per-slot `ready` flag** (host-mapped) with a **system-scope
release** store. The host consumer holds a read cursor and drains `buf[cursor]`
while `ready[cursor]` is set (acquire), consuming the dense ready-prefix.

### Device-side step loop + termination
ONE cooperative launch. CTA 0 is the admission+termination controller; **all**
CTAs cooperate on the per-step compute (grid-stride over slots, one CTA per
active slot: integer token recurrence + GEMV `svec` advance). Cooperative-grid
invariant — every resident CTA executes the identical `grid.sync()` sequence per
step: `[sync after admission] -> [compute] -> [sync after compute] -> [sync after
termination decision]`. The engine loops until `done_flag` is set AND the request
ring is drained AND `tokens_emitted >= total_tokens`, then CTA 0 sets the stop
flag (broadcast via the next barrier) and every CTA breaks.

### Synthetic decode compute (deterministic, reproducible)
The emitted **token** is a pure integer hash-chain (`splitmix64`) keyed on
`(request_id, pos)` and chained `token(pos) = f(token(pos-1), pos)`, so device A,
device B and the CPU reference produce **bit-identical** token streams regardless
of float order / parallel reduction — this is what the HARD gate checks, and it
forces the engine to actually carry per-slot state across steps. A representative
float GEMV (`W[kH x kD] · svec`, kH=kD=128) runs per step as the compute
stand-in and evolves `svec` (carried across steps); it is deterministic but only
the integer token is the bit-exact gate.

### Per-step-relaunch baseline (B) — today's pattern
Same synthetic compute, host-driven: every step the host rebuilds the active-slot
list, copies per-slot state h2d, **relaunches** `kRelaunchStep`, then copies the
step's emissions + advanced state d2h and advances its host bookkeeping. This is
the per-step launch + metadata rebuild + state sync the engine eliminates.

## Gate results (GPU0, two independent runs; correctness PASS both)

Occupancy: `kPersistentEngine` **8 blocks/SM = 1184 resident CTAs**, 32 regs,
0.5 KB smem (cooperative launch fully populates the device).

### Correctness — HARD (exactly-once + bit-exact, never self-compared)
All 3 staggered-admission scenarios, both runs:

| Scenario | requests | tokens | A ring | B ring | bit-exact A==B==CPU |
|---|---|---|---|---|---|
| N16_S8_stag4  | 16 | 432  | exactly-once (0 drop/dup/mismatch/missing) | exactly-once | PASS |
| N32_S16_stag8 | 32 | 1424 | exactly-once | exactly-once | PASS |
| N8_S8_nostag  | 8  | 599  | exactly-once | exactly-once | PASS |

Both rings deliver **every request and every completion exactly once** (0 drops,
0 dupes) under time-staggered admission with a variable active count and slot
reuse; both the engine's and the baseline's token streams are **bit-identical to
the independent CPU recompute**.

### Perf — steady-state per-step (clean isolation of the eliminated overhead)
Fixed fully-active batch of B slots decoded for S steps, no admission mid-run, so
the number isolates the per-step kernel-launch + host metadata rebuild + state
sync that the engine removes (free of the host arrival-stagger). Per-step us
(lower = better):

| Batch / steps | A engine us/step | B relaunch us/step | overhead cut us/step | speedup |
|---|---|---|---|---|
| B1,  S64  | 17.8–18.0 | 66–68 | ~48–50 | **3.66–3.83x** |
| B8,  S64  | 20.1–20.3 | 67–70 | ~47–50 | **3.30–3.47x** |
| B32, S64  | 23.4–23.7 | 72–75 | ~49–51 | **3.08–3.16x** |
| B128,S64  | 30.2–30.6 | 73    | ~42    | **2.38–2.41x** |
| B32, S256 | 21.9–22.4 | 72–73 | ~50–51 | **3.28–3.29x** |

**Scaling:** the relaunch baseline is **flat ~66–75 us/step regardless of batch**
(its cost is the per-step launch + h2d/d2h metadata + sync floor, independent of
how much real work the step does). The persistent engine is **17.8 us/step at B1
and only 30.6 us/step at B128** — sub-linear in batch (4x the slots costs ~1.7x).
Longer horizon (S256 vs S64 at B32) is flat for both, confirming the win is a
**per-step** constant the engine removes, amortized over every step of the run.

### Perf — staggered continuous-batching admission (end-to-end wall)
Realistic: a host producer thread enqueues on a stagger, a host consumer drains;
A is one cooperative launch for the whole run, B relaunches per step. Per-token us:

| Scenario | A engine us/tok | B relaunch us/tok | overhead cut | speedup |
|---|---|---|---|---|
| N16_S8_stag4  | 3.60–3.70 | 12.1–12.5 | ~8.4–8.9 | **3.28–3.47x** |
| N32_S16_stag8 | 1.80–1.83 | 5.92–5.97 | ~4.1–4.2 | **3.23–3.31x** |
| N8_S8_nostag  | 3.18–3.24 | 11.2–11.3 | ~8.0–8.1 | **3.48–3.53x** |

## Findings / honest notes

- **The win is real and per-step-constant.** Cross-step persistence removes a
  flat ~42–51 us/step of host orchestration (launch + metadata rebuild + state
  d2h/h2d + stream sync), giving **2.4–3.8x** lower per-step time across B1–B128
  and **3.2–3.5x** on realistic staggered admission. Because it is a per-step
  constant, the benefit compounds over a full decode (hundreds–thousands of
  steps).
- **Two serialization bugs were found and fixed during bring-up** (both visible
  only via the B-scaling sweep, which is why the sweep matters):
  1. An initial completion-ring design advanced a single contiguous `pub` counter
     via a **producer CAS-chain** — this forced all B per-step publishes into a
     strict serial order (O(B) latency/step with backoff); at B128 the engine was
     **0.02x** (3579 us/step). Replaced with a **per-slot `ready` stamp** (MPSC,
     no inter-producer coordination) -> B128 to 0.47x.
  2. The `prod` reservation counter was a **system-scope** atomic on host-mapped
     memory; under B-way contention/step it dominated. Moved `prod` to **device
     memory + device-scope** atomic (the host drains via `ready`, never reads
     `prod`) -> B128 to **2.41x** and every point a win.
- **B128 (full batch) is the least favorable point (still 2.4x).** It is the only
  regime where the engine's per-token host-mapped completion writes (scattered,
  uncached PCIe) start to approach the relaunch baseline's single bulk d2h copy.
  This is a completion-*streaming* property, not a cross-step-persistence flaw —
  a batched-drain ring would flatten it. The PDE target (overhead-bound decode,
  moderate batch) is exactly where the engine wins most decisively.
- **Cooperative occupancy is full** (1184 CTAs), so the persistent grid is not
  occupancy-starved; the per-step floor it removes is pure host overhead.

## Files
- `cpp/tensorrt_llm/kernels/pde/pde_g9_persist.cuh` — request ring, completion
  ring (MPSC ready-stamp), device-resident decode state, `kPersistentEngine`
  (cross-step cooperative loop), `kRelaunchStep` (baseline). Builds on
  `pde_substrate.cuh` (G0 persistent grid + cooperative launch sizing).
- `blaise_perf/pde/pde_g9_bench.cu` — staggered-admission correctness + perf,
  steady-state per-step sweep, CPU reference, exactly-once ring check.
- `blaise_perf/pde/build_run_g9.sh` — standalone nvcc sm_100 build + GPU0 run.

ABI-frozen `SparseMlaDecodeKvarnHotOp.cpp` / `hisparseKvarnBdrRead.cuh`
untouched; no decode/model path touched.

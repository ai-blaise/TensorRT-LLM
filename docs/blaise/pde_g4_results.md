# PDE G4 — heterogeneous-worker copy ∥ compute overlap: gate results

Gate for **G4** of the Persistent Decode Engine: prove that a COPY warp/SM group
(the HiSparse cold→hot KV swap-in) and a COMPUTE group (attention over the
resident hot blocks) can run **concurrently** in one persistent cooperative grid
with the copy latency **hidden** under compute — TileRT's "Heterogeneous Workers".

Kernel `cpp/tensorrt_llm/kernels/pde/pde_g4_het_overlap.cuh`; bench
`blaise_perf/pde/pde_g4_bench.cu` (`build_run_g4.sh`). Standalone `nvcc
-arch=sm_100`; builds on the G0 substrate; ABI-frozen files untouched.

## Verification (orchestrator-run, GPU 0, true CPU reference)

```
device: NVIDIA B200, 148 SMs, cc 10.0
OCC: kOverlapped 8 blk/SM; cooperative grid sized to the MIN occupancy across the
     4 cooperative kernels (kOverlapped/kSerial/kCopyOnly/kComputeOnly) = 6 blk/SM
     -> 888 CTAs.

cfg c0 Nswap256_Mhot4096 (256 B/blk, D=128):
  copy-only=3.727us  compute-only=436.152us  serial(sum)=441.197us
  overlapped=436.915us  best-split(296)=426.422us  -> HIDDEN=1.0000
  corr: A(overlapped) cos=1.0  B(serial) cos=1.0  A(pinned) cos=1.0  A-vs-B cos=1.0  -> PASS
cfg c1 Nswap512_Mhot8192:
  copy-only=4.081us  compute-only=863.857us  serial=873.541us
  overlapped=865.781us  best-split(296)=845.619us  -> HIDDEN=1.0000
  corr: A cos=1.0  B cos=1.0  A(pinned) cos=1.0  A-vs-B cos=1.0  -> PASS

G4 GATE: PASS(correctness)
```

All correctness is vs an independent CPU reference (the copy lands all N_swap
blocks; the compute is a softmax-weighted reduction over the hot blocks). bit-exact
A==B==CPU.

## What G4 establishes

- **The copy is FULLY hidden** under compute (`hidden=1.0` both configs):
  overlapped latency ≈ compute-only, NOT copy+compute. The COPY SM group and the
  COMPUTE SM group genuinely run concurrently in one persistent cooperative grid
  — the heterogeneous-worker substrate the engine uses to overlap the HiSparse
  swap-in with the attention hot-read.
- Both device→hot (17.6–32.1 GB/s measured) and host-pinned→hot (10.6–15.9 GB/s)
  copy paths land correctly and stay hidden (`hidden_pin=1.0`).
- Occupancy: 6 blk/SM = 888 resident CTAs (the cooperative-launchable count after
  sizing to the min occupancy across all 4 kernels; regs=28, 544 B smem).

## Bug found + fixed (orchestrator finish after the impl agent stalled post-wip)

The G4 impl agent committed the code (`wip 0b2f5c66`) but **stalled before building
or gating it**. The orchestrator built + gated it and found a real bug the agent
never caught: the cooperative grid was sized from `kOverlapped`'s occupancy alone
(8 blk/SM = 1184 CTAs), but `kSerial`/`kCopyOnly`/`kComputeOnly` have lower
occupancy → `cudaLaunchCooperativeKernel` failed with **"too many blocks in
cooperative launch."** Fixed by sizing the grid to the **min** occupancy across
all 4 cooperative kernels (using the agent's own — previously unused —
`occ_blocks_per_sm` helper) → 6 blk/SM = 888 CTAs. (This is the same "agent wrote
it, never ran it" failure mode as G1; caught by the orchestrator building it.)

## Honest caveats

- Representative stand-in shapes (random data), not the real packed-kvarn KV /
  FlashMLA hot-read; G4 proves the heterogeneous-worker copy∥compute mechanism +
  the copy-is-hidden property, not the production KV numerics.
- The live overlap magnitude in the real decode loop (real per-step miss volume
  vs real hot-read duration) needs the model — Gate-5-7 territory.
- The PDE substrate (G0 grid+barrier, G1 two-region, G2 cp.async overlap, G3
  device control flow, G4 heterogeneous workers) is now all validated in
  isolation; composing them onto the real frozen kernels + the model forward is
  the remaining integration frontier (needs the model stack).

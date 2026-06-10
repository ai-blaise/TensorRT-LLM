# kvarn_inkernel result-log index — READ BEFORE CITING NUMBERS

These logs span **three kernel generations**. Only the per-sub-block
warp-cooperative kernel is what ships (fused into `mlaKernels.cu`,
commit `76204896f`). Citing read-vs-fp8 ratios from the wrong log gets
you answers that differ by ~9x in either direction (audit finding F-47).

| Log | Kernel generation | BDR read vs fp8 read |
|---|---|---|
| `inkernel_dequant_n32.log`, `inkernel_dequant_sweep_v1.log`, `inkernel_dequant_ab_v2.log` | gen-1 standalone (smem-heavy, 256-thread) | **5.75x SLOWER** |
| `bdr_inkernel_bench` outputs (`inkernel_fused_vs_readfloor_v3.log`, `inkernel_amortized_v1.log`) | gen-2 standalone | **1.3–1.5x slower** |
| `bdr_persub_inkernel_sweep.log` (+ `bdr_persub_inkernel_validate.cu`) | gen-3 PRODUCTION per-sub-block warp-coop | **0.65x (faster than fp8)** — matches the `76204896f` commit claim |
| `bdr_longctx_4k_128k.log` | gen-3 accuracy | per-sub-block holds ~0.992 cos @128K; per-token/naive collapse ~0.866 |
| `fmha_overlap_n1024.log` | overlap probe | attention hides ≤~27% of the fullFUSED dequant; amortization, not overlap, is the cover (see fixed probe conclusion) |
| `d4_e2e_projection.log`, `intree_build_evidence.log` | bookkeeping | — |

Reproduced 2026-06-10 on B200 (CUDA 13.1, `-arch=sm_100a`): all rows
above re-measured within noise; see `PERF_AUDIT_optrt_2026-06-10.md` §2.1.

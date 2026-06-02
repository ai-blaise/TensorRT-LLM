# WarpDecode Routing Distribution Analysis (corrected)

> This document previously claimed a 2.2-4.5× WarpDecode speedup at a
> "production-realistic" routing distribution, justified by EPLB hot-expert
> packing. **Those numbers were reward-hacked** — they compared a WarpDecode
> path that was (incorrectly) given zero all-to-all against a native path that
> paid the full all-to-all, and they leaned on an EPLB-packing story that does
> not change the measured decode physics. The honest analysis follows.

## TL;DR

WarpDecode on `BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft`
is **~1.0-1.13× local and ~1.05× system** over the native NVFP4 runner on B200,
**independent of context length** and **independent of route concentration** in
any way that crosses a "2.2×" bar. The all-to-all is common to both paths
because the model (196.6 GB) does not fit on one B200 (~179 GB) and the experts
must be expert-parallel sharded on both paths. See
`warpdecode_deployment_guide.md` for the measured tables and the model-physics
explanation.

## Why route concentration does not create a large WarpDecode win

The earlier analysis argued that EPLB packs hot experts onto local ranks so that
each decode step touches few unique experts ("slot1-8"), and that WarpDecode is
2-4× faster in that regime. Two things are wrong with that:

1. **The comparison was unfair.** The "WarpDecode" side was benchmarked with no
   inter-GPU communication while the native side paid the full all-to-all. On
   this model both paths are expert-parallel and both pay the all-to-all, so the
   correct comparison is matched-local (~1.0-1.13×) and system-with-real-a2a
   (~1.05×). When both paths see the same a2a, the route distribution does not
   move the ratio across a 2× threshold.

2. **Both paths benefit equally from fewer unique experts.** When fewer experts
   are routed, *both* the native runner and WarpDecode read fewer weight bytes
   from HBM. Concentration lowers absolute latency for both, but it does not give
   WarpDecode a structural advantage — the native NVFP4 runner already fuses the
   gather/scatter/finalize stages that WarpDecode targets.

At high route diversity the kernel becomes **HBM-bandwidth-bound on weight
reads**, and the native NVFP4 runner already runs at ~89-96% of B200 peak HBM
(see `warpdecode_hbm_floor_analysis.md`). No kernel can read the routed expert
weights faster than the memory bus, so there is no large speedup available there
either.

## What is actually true

- **Decode is context-independent.** The per-`(GPU count, users)` latency holds
  across `{1k .. 128k}` context, because decode reads the same routed expert
  weights regardless of KV length.
- **The honest win is the fused output-owned kernel structure**, worth
  ~1.0-1.13× local / ~1.05× system, selected automatically by the AutoTuner at
  1-CTA (`tile_size=128`).
- **EPLB still matters for the system**, but as a load-balancing / comm-shaping
  mechanism shared by both paths — not as a lever that makes WarpDecode multiple
  times faster than native.

## References

- `docs/source/features/warpdecode_deployment_guide.md` — measured tables + model physics.
- `docs/source/features/warpdecode_hbm_floor_analysis.md` — HBM bandwidth ceiling.
- `benchmarks/python/cute_warpdecode/WARPDECODE.md` — backend + tile_mode reference.
- Honest benches: `benchmarks/python/wd_matched.py`, `wd_honest_e2e.py`,
  `wd_2cta_compare.py`, `wd_2cta_full.py`, `wd_tactic_probe.py`.

# WarpDecode: output-owned NVFP4 MoE decode

WarpDecode is the Blaise output-owned NVFP4 MoE **decode** path for
`BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft` on NVIDIA
B200 (Blackwell SM100). This is the canonical reference for what it is, how to
configure it, how it is labeled in logs, and its honest measured performance.

## What it is

WarpDecode is realized by **`CuteDslFusedMoE`** (`tensorrt_llm/_torch/modules/
fused_moe/fused_moe_cute_dsl.py`). It runs the NVFP4 MoE as:

```
moe_sort
  -> cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell   (FC1 + SwiGLU, fused gather)
  -> cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell    (FC2 + combine, fused finalize)
```

The output-owned structure fuses the gather / activation / scatter / combine
stages that an expert-centric grouped baseline pays separately. `WARPDECODE` is
an explicit `moe_backend` alias that resolves to `CuteDslFusedMoE`, so the
selection is named in logs and the WarpDecode tile policy is applied.

## How to configure

```python
from tensorrt_llm.llmapi import MoeConfig, WarpDecodeConfig

# Default: AutoTuner selects the tile size per shape (1-CTA at decode).
MoeConfig(backend="WARPDECODE")

# Force decode-optimal 1-CTA (tile_size=128), skip profiling:
MoeConfig(backend="WARPDECODE",
          warp_decode=WarpDecodeConfig(enabled=True, tile_mode="decode_1cta"))

# Force 2-CTA (tile_size=256) for prefill / large batch (slower at pure decode):
MoeConfig(backend="WARPDECODE",
          warp_decode=WarpDecodeConfig(enabled=True, tile_mode="prefill_2cta"))
```

### `WarpDecodeConfig.tile_mode`

| value | behavior | when |
|-------|----------|------|
| `autotune` (default) | AutoTuner profiles `tile_size` in `{128, 256}` and caches the fastest per shape | always safe; selects 1-CTA at decode |
| `decode_1cta` | pin `tile_size=128` (1-CTA), no profiling | pure decode, lowest launch overhead |
| `prefill_2cta` | pin `tile_size=256` (2-CTA) | prefill / large batch only; **slower at pure decode** |

The 1-CTA vs 2-CTA lever is the grouped-GEMM `tile_size`: `cluster_shape =
(tile_size // 128, 1)`, so `128 -> 1-CTA` and `256 -> 2-CTA`. `tile_size` is
passed consistently to `moe_sort(tile_tokens_dim=)` and both cute_dsl ops. In
`autotune` mode the `CuteDslFusedMoENvfp4Runner` exposes both tiles via
`get_valid_tactics() -> [128, 256]` and `AutoTuner.choose_one` picks the best;
the forced modes bypass profiling and call the impl directly with the pinned
`tile_size`.

> `tile_mode` only affects the `WARPDECODE` backend. The plain `CUTEDSL` backend
> keeps `autotune` behavior unchanged.

## Explicit labeling

WarpDecode is named in logs at config time and at runtime:

| site | message |
|------|---------|
| `create_moe.get_moe_cls` | `Selecting CuteDslFusedMoE for WarpDecode (output-owned NVFP4 decode, tile_mode=...)` |
| `CuteDslFusedMoE.__init__` | `WarpDecode backend active: CuteDslFusedMoE output-owned NVFP4 decode, tile_mode=...` |
| `CuteDslFusedMoE.run_moe_nvfp4` (forced modes) | `WarpDecode run_moe_nvfp4 SELECTED forced tile_mode=... (tile_size=..., 1-CTA/2-CTA)` |
| `warp_decode.try_run_warp_decode` (legacy overlay) | `WarpDecode SELECTED (overlay): ...` / `WarpDecode FALLBACK to native MoE backend (reason=...)` |

## Measured performance (B200, graph + PDL, real NCCL all-to-all)

MoE decode is **context-independent**: each `(GPU count, users)` cell holds
across all context lengths `{1k .. 128k}`. WarpDecode below is 1-CTA
(autotune-selected) vs the native NVFP4 runner.

**Matched local** (all-to-all excluded as common): G8c16 1.00×, G8c32 1.08×,
G4c16 1.13×, G4c32 1.07×, G2c16 1.09×, G2c32 1.02× → mean ~1.06-1.10×.

**System end-to-end** (real a2a, common to both; ~50 µs in-graph ≈ 40% of e2e at
8 GPUs): G8c16 0.99×, G8c32 1.04×, G4c16 1.10×, G4c32 1.06×, G2c16 1.11×, G2c32
1.03× → mean ~1.05×.

**Forced 2-CTA at decode**: numerically correct (cosine vs 1-CTA = 0.99999) but
slower (0.69-0.84× of native) — per-expert M ~1-2 tokens, so 256-row tiles run
mostly empty. The AutoTuner prunes 2-CTA at decode and picks 1-CTA.

**Honest conclusion: ~1.0-1.13× local, ~1.05× system; 1-CTA is decode-correct;
autotune selects it.**

## Why the all-to-all is common (do not re-hack it)

The model is **196.6 GB** on disk; a single B200 holds **~179 GB**. Since
196.6 GB > 179 GB, the model **does not fit on one B200**, so the experts must be
**expert-parallel sharded** on both the native and WarpDecode paths, and **both
pay the same all-to-all**. WarpDecode is output-owned (it loops a token's 8
experts in-warp and needs them local), but on this model it still cannot run
all-to-all-free — it shards experts like native and pays the same comm.

The gather/scatter in the Cursor blog are **local within-GPU HBM ops** (a
single-GPU kernel with zero inter-GPU comm), already fused by op-trt's native
NVFP4 runner — which is why the realistic win is ~1.0-1.13×, not the blog's 1.84×
(measured vs an unfused BF16-activation baseline on one GPU with no a2a).

> Never benchmark WarpDecode with zero all-to-all against a native path that pays
> the full all-to-all. That is the reward-hacked framing this folder's corrected
> docs replace. The a2a is common; compare matched-local or full-system.

## Honest benchmarks

In `benchmarks/python/`:
- `wd_matched.py` — matched-local WD vs native (a2a excluded as common).
- `wd_honest_e2e.py` — full-system e2e with real NCCL a2a.
- `wd_2cta_compare.py`, `wd_2cta_full.py` — 1-CTA vs 2-CTA correctness + latency.
- `wd_tactic_probe.py` — tactic / tile enumeration.

The `cute_warpdecode/` numbered scripts (`00`-`19`) are the kernel-correctness
exploration (cosine 1.0; compute-bound discovery; tensor cores are the lever).
Their speedup numbers are superseded by the measured tables above.

## See also

- `docs/source/features/warpdecode_deployment_guide.md` — deployment guide.
- `docs/source/features/warpdecode_hbm_floor_analysis.md` — HBM bandwidth ceiling.
- `docs/source/features/warpdecode_production_routing_analysis.md` — corrected routing analysis.
- `tensorrt_llm/_torch/modules/fused_moe/MOE_DEVELOPER_GUIDE.md` — WARPDECODE section.

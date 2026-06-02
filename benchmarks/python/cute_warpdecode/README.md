# Output-Owned NVFP4 MoE Decode Kernel (CuTeDSL) — WarpDecode

CuTeDSL implementation of the Cursor "WarpDecode" output-owned MoE decode kernel for
`DeepSeek-V3.2-REAP-345B` (HIDDEN=7168, INTER=2048, 128 experts, top_k=8), pure NVFP4
(e2m1 data + e4m3 per-16 block scales).

Each CTA owns one routed (expert, token) pair; the 128 threads each own one output row.
This eliminates the grouped-GEMM layout stages (128-tile padding, scatter, combine,
per-expert activation/output buffers) that the grouped TRTLLMGen baseline pays at decode,
where the per-rank routed-row count is tiny (16–128) and tile-128 padding wastes 4–64×.
Experts stay EP-sharded — the win is the kernel structure, not DP-replication.

## Files (each is self-checking; run under the CuTeDSL container)

| file | what | correctness |
|---|---|---|
| `01_single_gemv.py` | single output-owned NVFP4 GEMV (per-thread-output) | cosine **1.000000** |
| `02_batched_gather_gemv.py` | batched GEMV, in-kernel expert+token gather (no weight materialization), real FC1 shape M=4096 K=7168 | cosine **1.000000** |
| `03_gateup_swiglu.py` | FC1 + SwiGLU fused (gate row j & up row j+INTER in one K-loop, `silu(g)*u` on-chip) | cosine **1.000000** |
| `04_down_scatter.py` | FC2 + route-weight + atomic scatter-add to `out[token]` (handles top_k collisions) | cosine **1.000000** |
| `05_full_moe.py` | full pipeline gate_up → NVFP4-quant(intermediate) → down | kernel-vs-quantref cosine **1.000000**; vs-fp32 0.9952 (NVFP4 intermediate floor) |

## Key implementation notes

- **In-kernel gather**: `e = meidx[bidz]; t = mtidx[bidz]; We = mW[e,None,None]` — `None` keeps
  the mode; a bare scalar index illegally dereferences a sub-byte FP4 tensor.
- **SF K-broadcast layout** (built inside `@cute.jit`): `make_layout((...,(SVS,ks),...),
  stride=(...,(0,1),...))` gives scale-broadcast-every-16 without the TMA atom-swizzle; the
  torch reference is then a plain `repeat_interleave(16)`. `tile_atom_to_shape_SF` / `cosize`
  are MLIR-context-only (callable only inside `@cute.jit`).
- **`cute.compile`**: `Constexpr` args are baked at compile and must be OMITTED from the runtime
  call (passing them shifts the stream into a tensor slot → "cannot be converted to pointer").
- **K-loop trip count** = `cute.size(gW.layout[3].shape)` (mode 3 = K/BK tile count); mode 2 is
  the M-tile count and silently truncates the contraction.
- **Scatter**: `cute.arch.atomic_add(mOut.iterator + (t*HIDDEN + h), contrib)`.

## Status and measured outcome

**Correctness complete** for the output-owned NVFP4 decode kernels (cosine 1.0 vs
the quant reference; the NVFP4 intermediate is the only numerical floor). The key
findings from this exploration are:

- The output-owned structure is **correct** (cosine 1.0).
- At decode, this path is **compute-/bandwidth-bound on the routed expert weight
  reads**, not on the padding/scatter overhead that the Cursor blog targets.
  Tensor-core utilization (not the gather structure) is the lever.

**Honest end-to-end result.** When wired into the production path as the
`WARPDECODE` backend (`CuteDslFusedMoE`) and compared against op-trt's *fused*
native NVFP4 runner on B200 with real NCCL all-to-all, the measured speedup is
**~1.0-1.13× local and ~1.05× system**, context-independent. It is **not** the
Cursor blog's 1.84× — that figure was measured on a single GPU against an unfused
BF16-activation baseline with no all-to-all, whereas op-trt native already fuses
the gather/scatter/finalize stages and (on this 196.6 GB model, which does not
fit on one 179 GB B200) both paths pay the same all-to-all. Earlier inflated
figures in this folder's history (e.g. multi-× speedups, 3.95 TB/s targets) are
superseded by the measured tables.

See `WARPDECODE.md` and `docs/source/features/warpdecode_deployment_guide.md` for
the measured performance, the model-physics reason the all-to-all is common, and
the configuration (`moe_backend="WARPDECODE"` + `WarpDecodeConfig.tile_mode`).

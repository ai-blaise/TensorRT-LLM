# MoE Developer Guide

## Architecture

### MoE Layer in Model

```text
Input Hidden States
       │
       ├──────────────────────┐
       │                      │
       ▼                      ▼
   fc_gate (Router)     Shared Expert (optional)
       │                      │
       ▼                      │
  Fused-MoE                   │
  ┌─────────────────────┐     │
  │ Routing (topK, etc) │     │
  │         │           │     │
  │         ▼           │     │
  │   MoE Backends      │     │
  │  (FC1→Act→FC2)      │     │
  │         │           │     │
  │   Apply Weights     │     │
  └─────────────────────┘     │
       │                      │
       ▼                      ▼
    Combine Outputs (sum) ◄───┘
       │
       ▼
  Final Hidden States
```

### ConfigurableMoE: The Orchestrator

`ConfigurableMoE` composes independent components via composition (not inheritance) and **owns module lifecycle** (backend construction, weight loading, comm-strategy creation, `repeat_idx` advancement, DWDP record). Forward-time execution is delegated to a **scheduler**:

```text
ConfigurableMoE
├── Backend           (pure computation: routing → quantize → FC1 → act → FC2)
├── Communication     (distributed, optional: dispatch tokens → compute → combine)
├── EPLB              (optional: dynamic expert migration across GPUs)
└── MoEScheduler      (forward-execution strategy: chunking, EPLB hook ordering,
                       comm orchestration; selected by backend.scheduler_kind)
```

`forward_impl` is thin — it resolves `output_dtype`, delegates to `self.scheduler.forward(...)`, then runs wrapper-level bookkeeping that both schedulers share:

```python
def forward_impl(self, x, router_logits, ...):
    outputs = self.scheduler.forward(x, router_logits, ...)
    if self.enable_dwdp:
        self.dwdp_manager.record_compute_and_prefetch_next(self.layer_idx)
    self.repeat_idx = (self.repeat_idx + 1) % self.repeat_count
    return outputs
```

### Scheduler Selection (`MoESchedulerKind`)

Each backend declares one of two scheduler kinds via the `scheduler_kind` class attribute (defined on `MoE` base, default `EXTERNAL_COMM`):

| Kind | Scheduler class | Used by | Cross-rank EP exchange |
|------|-----------------|---------|------------------------|
| `EXTERNAL_COMM` | `ExternalCommMoEScheduler` | Cutlass, DeepGemm, CuteDSL, DenseGEMM, TRTLLMGen | Host issues `Communication.dispatch` / `.combine` outside the MoE kernel; supports per-chunk EPLB hooks and multi-stream chunk overlap |
| `FUSED_COMM` | `FusedCommMoEScheduler` | MegaMoEDeepGemm | Comm is fused into the backend kernel via NVLink SymmBuffer; no host comm; lockstep chunk launches; EPLB stats AllReduced internally |

The two paths have *deliberately opposite* invariants (`use_dp_padding` honored vs ignored, ADP padding kept vs stripped, empty-chunk substituted vs zero-token kernel launch, multi-stream overlap allowed vs forbidden). See `moe_scheduler.py` class docstrings and `MOE_SCHEDULER_DESIGN.md` for the full contract.

### External-comm execution flow (most backends)

`ExternalCommMoEScheduler._forward_chunk_impl` runs per chunk:

```text
[EPLB start_wait_gpu] → routing → [EPLB done_wait_gpu + update_statistic + route]
  → [comm.prepare_dispatch (NVLink2-sided)] → quantize/dispatch (adaptive order)
  → backend.run_moe → [EPLB start_set_cpu] → comm.combine → [EPLB done_set_cpu]

Adaptive quantize/dispatch order (gated by comm.supports_post_quant_dispatch()):
  Post-quant flow: quantize_input() → comm.dispatch()   (send quantized data)
  Pre-quant flow:  comm.dispatch() → quantize_input()   (send raw, quantize locally)
```

EPLB hooks fire only at the first/last chunk of the first/last `repeat_idx`. Multi-stream chunk overlap is enabled when `not enable_alltoall and aux_stream is not None`.

### Fused-comm execution flow (MegaMoE-style)

`FusedCommMoEScheduler._forward_chunk` runs per chunk:

```text
[EPLB start_wait_gpu] → routing → [EPLB done_wait_gpu + update_statistic + route]
  → backend.quantize_input → backend.run_moe (fused dispatch+GEMM+act+GEMM+combine)
  → [EPLB start_set_cpu + done_set_cpu]
```

No external `Communication.dispatch` / `.combine`. Zero-token chunks still launch the kernel so peer EP ranks can cross the in-kernel NVLink barrier.

### Core Design Principles

1. **Composition over inheritance** — Backend, Communication, EPLB, and Scheduler are independent, composable components
2. **Any Backend × Any Communication × EPLB On/Off** — All valid combinations should work (subject to `can_implement` and `scheduler_kind`)
3. **Backend = pure computation** — No communication logic, no EPLB logic inside backends
4. **Communication is pluggable** — `EXTERNAL_COMM` backends pick a strategy via `CommunicationFactory` based on hardware/workload; `FUSED_COMM` backends bypass external comm entirely
5. **Backend declares capabilities** — `can_implement()` declares supported quant/dtype; ConfigurableMoE adapts flow accordingly
6. **Backend declares scheduler** — `scheduler_kind` class attribute selects the forward path; lifecycle code stays generic, forward path stays specialized

## Architecture Transition (IMPORTANT)

The codebase is transitioning between two architectures:

| | Old Path | New Path |
|---|---|---|
| Entry | `XXFusedMoE` (e.g., `CutlassFusedMoE`) | `ConfigurableMoE` + `XXBackend` + `MoEScheduler` |
| Communication | Embedded inside each backend | Separated into `communication/` (or fused into kernel for `FUSED_COMM`) |
| Forward execution | Inline in backend | `MoEScheduler` (`moe_scheduler.py`) |
| EPLB | Only in WideEPMoE | Available to all backends |
| Status | Being replaced | Active development |

ConfigurableMoE currently supports these backends (`create_moe.py`):
- `CutlassFusedMoE`, `TRTLLMGenFusedMoE`, `DeepGemmFusedMoE`, `CuteDslFusedMoE`, `DenseGEMMFusedMoE`, `MegaMoEDeepGemm`
  - `WARPDECODE` is an explicit `moe_backend` alias that resolves to `CuteDslFusedMoE` (output-owned NVFP4 decode); see the WARPDECODE section below.

Still on old path (standalone, with embedded communication):
- `TritonFusedMoE`, `WideEPMoE`, `VanillaMoE`

**Rule: All new features should target ConfigurableMoE + Backend + Scheduler architecture.**

## File Map

### Core (`fused_moe/`)

| File | Role |
|------|------|
| `configurable_moe.py` | Orchestrator — wires Backend + Communication + EPLB + Scheduler; owns lifecycle and `forward_impl` |
| `moe_scheduler.py` | Forward-execution strategies (`MoEScheduler` ABC, `ExternalCommMoEScheduler`, `FusedCommMoEScheduler`, `create_moe_scheduler` factory) |
| `create_moe.py` | Factory — selects MoE class based on `model_config.moe_backend` |
| `interface.py` | Base class `MoE` and enums (`MoEWeightLoadingMode`, `MoESchedulerKind`, `AlltoallMethodType`) |
| `quantization.py` | Quantization method implementations (`FusedMoEMethod` subclasses: weight creation, loading, quant/dequant ops per quant mode) |
| `routing.py` | Routing methods (`TopKRouting`, etc.) |
| `moe_load_balancer.py` | EPLB implementation |
| `moe_op_backend.py` | Op backend registry for TRTLLMGen (flashinfer/trtllm ops) |
| `warp_decode.py` | Optional **legacy** WarpDecode overlay (trtllm_gen fast path at scheduler dispatch). Secondary to the canonical `WARPDECODE` backend; see the WARPDECODE section. |

### Backends (`fused_moe/`)

| File | Backend | Hardware | Scenario | Scheduler |
|------|---------|----------|----------|-----------|
| `fused_moe_cutlass.py` | `CutlassFusedMoE` | SM80+ | High throughput, most comprehensive quant support | `EXTERNAL_COMM` |
| `fused_moe_trtllm_gen.py` | `TRTLLMGenFusedMoE` | SM100/SM103 | Min-latency and high-throughput on Blackwell | `EXTERNAL_COMM` |
| `fused_moe_deepgemm.py` | `DeepGemmFusedMoE` | SM100/SM103 | FP8 Block Scales on Blackwell | `EXTERNAL_COMM` |
| `fused_moe_densegemm.py` | `DenseGEMMFusedMoE` | SM100/SM103 | NVFP4 min-latency; CuTe DSL dense GEMM packs all experts into one matrix (vs Cutlass per-expert scatter), efficient for small token counts | `EXTERNAL_COMM` |
| `fused_moe_cute_dsl.py` | `CuteDslFusedMoE` | SM100/SM103 | High throughput NVFP4, generally faster than Cutlass. Also selectable via the `WARPDECODE` backend name — the output-owned NVFP4 MoE *decode* path (CuTe-DSL `gather_grouped_gemm_act_fusion` FC1+SwiGLU → `grouped_gemm_finalize_inplace` FC2+combine; the fused "warp compute" that drops the gather/pad/scatter/reduce bookkeeping stages of expert-centric MoE) | `EXTERNAL_COMM` |
| `fused_moe_cute_dsl_b12x.py` | `CuteDslB12xFusedMoE` | SM120/SM121 | NVFP4 hybrid CUTLASS-prefill / FlashInfer NVFP4 MoE decode — best perf on RTX PRO 6000 (SM120) and DGX Spark (SM121); select via the `CUTEDSL` backend path (auto-promoted when flashinfer is importable) | `EXTERNAL_COMM` |
| `mega_moe/mega_moe_deepgemm.py` | `MegaMoEDeepGemm` | SM100/SM103 | W4A8_MXFP4_MXFP8 via DeepGEMM `fp8_fp4_mega_moe` fused dispatch+GEMM+act+GEMM+combine kernel; requires `hidden_size % 512 == 0` | `FUSED_COMM` |
| `fused_moe_triton.py` | `TritonFusedMoE` | SM90 only | GPT-OSS on Hopper (requires `swiglu_gptoss_style=True`) | (legacy path) |
| `fused_moe_wide_ep.py` | `WideEPMoE` | All GPUs | Deprecating — use ConfigurableMoE instead | (legacy path) |
| `fused_moe_vanilla.py` | `VanillaMoE` | All devices | Reference / debugging only | (legacy path) |

### Communication (`fused_moe/communication/`)

Communication strategies are auto-selected at runtime by `CommunicationFactory` based on hardware and configuration. Skipped for `FUSED_COMM` backends. See `communication_factory.py` for selection logic and `base.py` for the `Communication` ABC.

### MegaMoE (`fused_moe/mega_moe/`)

| File | Role |
|------|------|
| `mega_moe_deepgemm.py` | `MegaMoEDeepGemm` backend (DeepGEMM `fp8_fp4_mega_moe` wrapper) |
| `CHUNKING_DESIGN.md` | Chunking design for MegaMoE (sequential multi-chunk, in-kernel barrier semantics) |
| `COMMUNICATION_COMPARISON.md` | Comparison of fused-comm SymmBuffer vs external comm strategies |
| `KERNEL_INTERNALS.html` | Reference for the underlying DeepGEMM kernel layout |

### Design Documents

| File | Topic |
|------|-------|
| `MOE_SCHEDULER_DESIGN.md` | Scheduler refactor design + `MoEScheduler` contract |
| `mega_moe/CHUNKING_DESIGN.md` | MegaMoE chunking invariants |

### Tests

| File | Tests | Status |
|------|-------|--------|
| `test_moe_backend.py` | Backend unit tests (`run_moe`, `can_implement`) | Active |
| `test_moe_module.py` | ConfigurableMoE integration tests (Backend × Comm × EPLB) | Active |
| `test_fused_moe.py` | Legacy MoE tests | Being replaced, do NOT add new tests here |
| `test_moe.py` | Legacy TRTLLM backend tests | Being replaced, do NOT add new tests here |

## WarpDecode (WARPDECODE backend)

**WarpDecode** is the Blaise output-owned NVFP4 MoE **decode** path for
DeepSeek-V3.2-REAP-345B on B200. It is **not a new backend class** — it is an
explicit `moe_backend="WARPDECODE"` alias that `create_moe.get_moe_cls` resolves
to **`CuteDslFusedMoE`** (the output-owned `cute_dsl` gather-grouped-GEMM +
SwiGLU / grouped-GEMM-finalize path), with explicit logging and a tile-mode
policy. Full reference:
`docs/source/features/warpdecode_deployment_guide.md`.

### Configuration

```python
from tensorrt_llm.llmapi import MoeConfig, WarpDecodeConfig

MoeConfig(backend="WARPDECODE")                                   # autotune (default)
MoeConfig(backend="WARPDECODE",
          warp_decode=WarpDecodeConfig(enabled=True, tile_mode="decode_1cta"))   # pin 1-CTA
MoeConfig(backend="WARPDECODE",
          warp_decode=WarpDecodeConfig(enabled=True, tile_mode="prefill_2cta"))  # pin 2-CTA
```

`WarpDecodeConfig.tile_mode` controls the grouped-GEMM `tile_size`
(`cluster_shape = (tile_size // 128, 1)`):

| value | tile_size | CTA | use |
|-------|-----------|-----|-----|
| `autotune` (default) | profiled in `{128, 256}` | picks 1-CTA at decode | always safe |
| `decode_1cta` | 128 | 1-CTA | pure decode, no profiling |
| `prefill_2cta` | 256 | 2-CTA | prefill / large batch; slower at pure decode |

`autotune` keeps the `CuteDslFusedMoENvfp4Runner` + `AutoTuner.choose_one` flow
(`get_valid_tactics() -> [128, 256]`). Forced modes bypass profiling and call
the impl with the pinned `tile_size`. `tile_mode` affects only the `WARPDECODE`
backend; the plain `CUTEDSL` backend is unchanged.

### Explicit labeling

`Selecting CuteDslFusedMoE for WarpDecode (... tile_mode=...)` at selection;
`WarpDecode backend active: CuteDslFusedMoE output-owned NVFP4 decode,
tile_mode=...` at init; `WarpDecode run_moe_nvfp4 SELECTED forced tile_mode=...`
on a forced tile mode.

### Measured performance (honest)

B200, graph + PDL, real NCCL all-to-all; decode is context-independent (each
`(GPU, users)` cell holds across `1k..128k` context). WarpDecode-1CTA vs native
NVFP4: **~1.0-1.13× local, ~1.05× system**. Forced 2-CTA is correct (cosine
0.99999) but slower at decode (0.69-0.84×); the AutoTuner prunes it and selects
1-CTA. **The all-to-all is common to both paths** — the 196.6 GB model does not
fit on one 179 GB B200, so experts are expert-parallel sharded on both and both
pay the same a2a. Do not benchmark WarpDecode with zero a2a vs a native path that
pays full a2a. See `docs/source/features/warpdecode_deployment_guide.md`.

### Legacy overlay (`warp_decode.py`)

`warp_decode.py` is a **separate, optional** runtime overlay that runs a
trtllm_gen `FP4BlockScaleMoERunner` at the scheduler dispatch point when
`MoeConfig.warp_decode.enabled` is set on top of another backend. It is
**secondary** to the `WARPDECODE` backend, defaults to letting the runner pick
its tactic automatically (`tactic=[-1, -1]`, matching the autotune-default
policy), and logs `WarpDecode SELECTED (overlay): ...` / `WarpDecode FALLBACK ...
(reason=...)`. Prefer the `WARPDECODE` backend; keep the overlay disabled unless
you specifically need the trtllm_gen fast path. When the overlay IS active its
hand-enumerated, retuned tactic tables are now used by default for the covered
decode buckets (1,2,4,8,16,32): validated cos=1.0 vs the AutoTuner pick and
graph-safe (no in-graph host AutoTuner call). Set `TRTLLM_WARP_DECODE_FIXED_TACTIC=0`
to force the pure `tactic=[-1,-1]` AutoTuner path. Uncovered shapes (padded 48/64)
always fall back to the AutoTuner automatically.

## Backend Capability Matrix

### Quantization Support

Each backend's `can_implement(quant_algo, dtype_activation, swiglu_gptoss_style, ...)` method declares supported quantizations. Source of truth: the `can_implement` classmethod in each backend file.

| Quantization | Cutlass | TRTLLMGen | DeepGemm | DenseGEMM | CuteDSL | MegaMoE-DG | Triton | WideEP | Vanilla |
|---|---|---|---|---|---|---|---|---|---|
| Unquantized (BF16/FP16) | Y (SM80+) | N | N | N | N | N | Y (SM90, BF16) | Y | Y |
| FP8 QDQ | Y (SM89+) | N | N | N | N | N | Y (SM90) | Y | Y |
| FP8 Block Scales | Y (SM90, SM120) | Y (SM100/103) | Y (SM100/103) | N | Y (SM100/103) | N | N | Y | Y |
| NVFP4 | Y (SM100/103/120/121) | Y (SM100/103) | N | Y (SM100/103) | Y (SM100/103/120/121) | N | N | Y | Y |
| W4A8 NVFP4 FP8 | N | Y (SM100/103) | N | N | N | N | N | N | N |
| W4A16 MXFP4 | Y (SM90) | Y (SM100/103) | N | N | N | N | Y (SM90) | N | N |
| W4A8 MXFP4 FP8 | Y (SM100/103) | Y (SM100/103) | N | N | N | N | Y (SM90) | N | N |
| W4A8 MXFP4 MXFP8 | Y (SM100/103) | Y (SM100/103) | N | N | N | Y (SM100/103, requires `hidden_size % 512 == 0`) | N | N | N |
| W4A8 AWQ | Y (SM89/90) | N | N | N | N | N | N | N | N |
| W8A16 | Y (SM80+) | N | N | N | N | N | N | N | N |
| INT4 WoQ (W4AFP8) | N | N | N | N | N | N | N | Y | N |

### Scheduler / EPLB Constraints

- `FUSED_COMM` backends (`MegaMoEDeepGemm`) **must not** layer host-side `Communication.dispatch` / `.combine` on top of the fused kernel — `ConfigurableMoE._create_comm_strategy_auto` returns `None` for them.
- Dynamic EPLB requires backend and quantization-method support. Backends gate
  wrapper-level constraints via `validate_configurable_moe`; `MegaMoEDeepGemm`
  supports dynamic EPLB by routing to slot IDs and migrating transformed DG
  weight tensors registered by its quantization method, with the constraint
  `num_slots % ep_size == 0`.
- `FUSED_COMM` backends use `ignore_allreduce=False` for EPLB statistic update because the fused kernel AllReduces routing stats internally.

## Canonical Examples

When adding new components, use these reference implementations:

| Task | Reference | Key methods to implement |
|------|-----------|--------------------------|
| New `EXTERNAL_COMM` Backend | `fused_moe_cutlass.py` (`CutlassFusedMoE`) | `can_implement`, `run_moe`, `create_weights`, `load_weights` |
| New `FUSED_COMM` Backend | `mega_moe/mega_moe_deepgemm.py` (`MegaMoEDeepGemm`) | Same as above + override `scheduler_kind = MoESchedulerKind.FUSED_COMM` and `validate_configurable_moe` for backend-specific constraints |
| New Quantization Method | `quantization.py` → `FP8QDQFusedMoEMethod` | Subclass `FusedMoEMethod`, implement quant/dequant ops |
| New Communication Strategy | `communication/nvlink_one_sided.py` (`NVLinkOneSided`) | Subclass `Communication`, implement `prepare_dispatch`, `dispatch`, `combine` |
| New Scheduler | `moe_scheduler.py` (`ExternalCommMoEScheduler` / `FusedCommMoEScheduler`) | Subclass `MoEScheduler`, implement `forward`; add new `MoESchedulerKind` value and wire into `create_moe_scheduler` factory |
| Backend Tests | `test_moe_backend.py` | Follow existing parametrize patterns |
| Integration Tests | `test_moe_module.py` | Test Backend × Communication × EPLB combinations |

**Note on backend inheritance:** New backends should inherit from `MoE` (in `interface.py`), NOT from `CutlassFusedMoE`. Current backends inherit from `CutlassFusedMoE` as a historical shortcut to reuse infrastructure (load balancer, weight management, TP/EP). This will be refactored — a dedicated `MoEBackend` interface will be extracted. `MegaMoEDeepGemm` and `DenseGEMMFusedMoE` already inherit directly from `MoE`.

## Anti-Patterns

- **Do NOT add communication logic inside backends** — Communication belongs in `communication/`, backends do pure computation (exception: `FUSED_COMM` backends own the SymmBuffer collective inside their fused kernel)
- **Do NOT add forward-execution policy inside backends** — chunking, EPLB hook ordering, dispatch/combine sequencing belong in `MoEScheduler`
- **Do NOT modify old `XXFusedMoE` files for new features** — Use ConfigurableMoE + Backend + Scheduler architecture
- **Do NOT add new tests to `test_fused_moe.py` or `test_moe.py`** — Use `test_moe_backend.py` and `test_moe_module.py`
- **Do NOT skip `can_implement()` checks** — Every backend must declare what it supports; unsupported combos must return `(False, reason)`
- **Do NOT pick `scheduler_kind` opportunistically** — Use `EXTERNAL_COMM` (default) unless your backend's fused kernel genuinely owns cross-rank exchange via SymmBuffer / equivalent in-kernel collective; `FUSED_COMM` brings hard invariants (no host comm, lockstep launches, no multi-stream overlap)
- **Schedulers MUST NOT write `moe.repeat_idx`** — `repeat_idx` is wrapper state advanced once per `forward_impl` regardless of chunk count

# Piecewise (PCG) & Breakable (BCG) CUDA Graph for the DSA prefill path — RESULTS

Branch `op-trt-hisparse-piecewise` (off `818065cb`), node 001, worktree
`/home/spencer/work/piecewise-wt`. This is the P3-DEFINITIVE / P4 / P5 result
record. P0–P2 design + findings are in `piecewise_p0_findings.md` and
`piecewise_cuda_graph_plan.md`; the partition-logic skeleton gate is in
`blaise_perf/piecewise/gate_partition.py`. The runnable definitive gate is
`blaise_perf/piecewise/gate_definitive.py` (+ `overhead_probe.py`), all run in
the proof image `local/dynamo-trtllm-optrt-custom:optrt-34fe7aaec-fixed-...`
on GPU 1 (`flock /tmp/gpu001_lock_b`, B200).

## TL;DR

op-trt already had the SGLang-#23351 fix-set for DSA (split into capturable
`trtllm::mla_dsa_proj` + eager `trtllm::mla_dsa_attn_inplace`, the latter a
registered piecewise split point). The work was therefore **enable + prove + max
coverage + characterize BCG**, not "un-exclude + strip constructs". All gated on
**random weights** through the **REAL** op-trt piecewise machinery
(`compilation/backend.py` `Backend` + `piecewise_optimizer`,
`enable_piecewise_cuda_graph=True`), which fully gates the three weight-
independent things (span count, capture bit-exactness, launch/latency delta).
Only end-to-end model ACCURACY needs the real 345B weights, and piecewise
changes no numerics, so that is honestly deferred.

| Result | Value | Source |
|---|---|---|
| Real captured-span count (N layers) | **N+1 captured + N eager**, no collapse | Tier 1 |
| Captured spans @ 61 layers | **62 captured + 61 eager** | Tier 1 sweep |
| Launch-count reduction @ 61 layers | **2013 → 123 = 93.9%** | Tier 1 sweep |
| Bit-exact eager-vs-piecewise (real Backend capture) | **TRUE, max_abs_diff=0.0, cos≈1.0** (all configs) | Tier 2/4/5 |
| Coverage (captured kernels / total) | **97.0%**, eager region == attention ops ONLY | Tier 5 |
| Latency win — compute-bound floor | 1.002–1.034× (dense GEMMs mask overhead) | Tier 2 |
| Latency win — overhead-bound ceiling | **4.72×** (1.811→0.384 ms, launch-bound regime) | overhead_probe |
| BCG (#25195) in op-trt | **NONE** (no break/resume); not needed for prefill | P4 / Tier 4 |

## P3-DEFINITIVE — the real benefit gate (gate_definitive.py)

A faithful `DeepseekV3DecoderLayer`-shaped module — input RMSNorm → real
`trtllm::mla_dsa_proj` (capturable) → real `trtllm::mla_dsa_attn_inplace`
(eager) → residual → post RMSNorm → MoE/down GEMM → residual, stacked N layers —
is driven through op-trt's REAL piecewise backend. The two custom ops are the
genuine production ops; `extract_extra_attrs` resolves through real
`MLA`/`TrtllmAttentionMetadata` subclasses (constructed weight-free). The op
bodies are real GPU kernels at the production tensor contract (`mla_dsa_proj`
returns the 9-tensor straight-line bundle; `mla_dsa_attn_inplace` writes the
attn output in place). DSA dims = the 345B graft config (hidden 7168, 128 heads,
qk_head 192, kv_lora 512, v_head 128, indexer 64×128).

### (a) Real captured-span count + launch count — Tier 1 (`piecewise_optimizer`)

Built the FX graph from the real custom-op targets via `make_fx`, then applied
the **production classifier verbatim** (`piecewise_optimizer.py:251-281`) and the
production `split_module` to count captured spans + per-span kernel launches.

```
layers=1   captured_spans=2   eager=1   launches 33  -> 3   (90.9%)
layers=8   captured_spans=9   eager=8   launches 264 -> 17  (93.6%)
layers=16  captured_spans=17  eager=16  launches 528 -> 33  (93.8%)
layers=61  captured_spans=62  eager=61  launches 2013-> 123 (93.9%)
```

- `aten.index.Tensor = 0`, `aten.cumsum = 0` on the captured path → **no
  `stop_partition` collapse** at any depth. The op-trt-specific risk the plan
  flagged (an early raw index/cumsum forcing everything eager) does NOT
  materialize for the DSA decoder-layer forward.
- **Span structure (a finding):** for N layers the partitioner yields exactly
  **N+1 captured spans**, not 2N. The post-attn MoE of layer L and the pre-attn
  proj of layer L+1 are contiguous capturable nodes with no split point between
  them, so they **merge into one captured span**. This is strictly better than 2
  spans/layer (fewer, larger captured graphs → fewer `cudaGraphLaunch`).
- **Launch-count win converges to ~94%** at depth: a 61-layer prefill step drops
  from **2013 → 123** kernel launches. This is the weight-independent structural
  benefit.

### (b) Bit-exact eager-vs-piecewise — Tier 2/4/5 (real `Backend` capture)

Drove the same module through `torch.compile(backend=Backend(
enable_piecewise_cuda_graph=True, capture_num_tokens=[...]))`, forced capture
(>3 warmups), replayed, and compared against the EAGER run of the same module
(TRUE reference, same random weights).

```
layers=4  tok=2048  BIT-EXACT True  max_abs_diff=0.000e+00  cos=0.99999994
layers=8  tok=1024  BIT-EXACT True  max_abs_diff=0.000e+00  cos=1.00000000
layers=8  tok=4096  BIT-EXACT True  max_abs_diff=0.000e+00  cos=1.00000024
layers=16 tok=2048  BIT-EXACT True  max_abs_diff=0.000e+00  cos=0.99999988
```

**Byte-for-byte identical**, every config. The one correctness pitfall found and
fixed: the attention output buffer must be allocated **inside** the captured
region per layer (mirroring `MLA.forward`'s `attn_output =
self.create_output(...)` at attention.py:3454), never crossed as an external
forward arg — passing it in aliases a stale captured address under replay
(initial attempt gave cos 0.128). Inputs must also use stable addresses (copy
new data into the same buffer, then replay). With both, capture is exact.

### (c) Launch-count + per-step latency win — the benefit envelope

The launch-count win (94% fewer, above) is exact and structural. Its conversion
to **latency** depends on the per-op compute/overhead ratio:

- **Compute-bound floor (Tier 2, dense-GEMM stand-in): 1.002–1.034×.** The
  synthetic MoE is one dense 7168×2048 GEMM/layer — heavy enough that matmul
  time dwarfs the fixed per-launch overhead removed, so the realized latency win
  is small (saved ~0.1–0.6 ms/step). This is the conservative lower bound.
- **Overhead-bound ceiling (`overhead_probe.py`): 4.72×** (1.811 → 0.384 ms,
  bit-exact). Same DSA-span op STRUCTURE but tiny per-op width so latency ==
  kernel-launch overhead. This is the regime where 94%-fewer-launches converts
  to a large win.
- **The real DSA prefill is overhead-bound** (project memory: prefill/decode
  runs 20–40× the bandwidth floor, launch/dispatch-overhead dominated), so the
  real per-step latency win sits between the floor and the ceiling — closer to
  the ceiling — and the exact number must be measured on the real 345B model
  under concurrency (see Deferred).

## P4 — Breakable CUDA Graph (BCG)

**op-trt has NO breakable/conditional CUDA-graph primitives.** Grep-confirmed:
no `breakable`, no `torch.cond`/`while_loop`, no `cudaGraphConditional`, no
break-and-resume, no `enable_bcg`/`enable_breakable` config. `TorchCompileConfig`
exposes only `enable_piecewise_cuda_graph` + `capture_num_tokens`.

What op-trt DOES have for dynamic conditions is **per-`num_tokens` bucketed
piecewise capture + graceful eager fallback**: `PiecewiseRunner.__call__` returns
`self.default_callable(*args)` (eager) for any runtime token count not in
`capture_num_tokens`. Tier 4 validates both legs end-to-end through the real
Backend:

```
[bucketed nt=2048] bit-exact=True max_abs_diff=0.000e+00  (captured graph replay)
[dynamic  nt=1536] bit-exact=True max_abs_diff=0.000e+00  (EAGER fallback, no recapture/crash)
```

**For the DSA-prefill scope, BCG is not needed.** Prefill dynamism is (1)
token-count — covered by bucketed capture, and (2) chunked prefill — handled in
eager metadata `prepare()` (`dsa.py:1158` "Chunked prefill metadata for indexer
(prefill-only, no CUDA graph needed)"); each chunk is a fixed-size forward the
buckets cover.

**Scope-gap (honest):** a true BCG (#25195, DeepSeek V4) keeps ONE graph and
breaks/resumes inside it for **decode-side variable speculative/MTP draft
length** (the `next_new_tokens`/variable-accepted-tokens dynamism, e.g.
`dsa.py:2054`, `speculative/interface.py`). That is **out of scope here**: decode
stays full-graph (full capture is strictly better than piecewise for fixed decode
shapes), and MTP is a decode concern. Implementing BCG would require adding
CUDA conditional-graph node support (`cudaGraphConditionalHandle`) or a
re-entrant splice-capture to the runtime — a substantial new primitive op-trt
lacks today, justified only by the decode/MTP path, not DSA prefill.

## P5 — maximize coverage + optimize (Tier 5)

**Coverage is already maximal — the eager region is the theoretical floor.**

```
COVERAGE: 256/264 kernels captured = 97.0%  | eager region = 8 kernels / 8 attn ops
eager region is EXACTLY the 8 attn ops (no proj/MoE/norm leakage): True
multi-bucket [1024,2048,4096]: all bit-exact
```

The eager region is **exactly** the N `mla_dsa_attn_inplace` ops; nothing else
(proj, MoE, norms) leaks to eager. It cannot shrink further: `forward_dsa_attn`
(attention.py:2014) opens with `q = q[:num_tokens]` — an `aten.slice` whose
length is a runtime int from batch metadata — and every downstream op depends on
those sliced tensors, plus the `num_contexts>0` / `num_generations>0`
data-dependent branches and the variable-seqlen sparse-attention kernel. None
can be hoisted into a fixed-shape captured graph. Op 1 (`mla_dsa_proj`) is
capturable precisely because it runs on the **full padded tensor**; the split
sits exactly at the padded→actual-`num_tokens` boundary. So **no #23351-P2-style
construct-stripping is needed** — op-trt's DSA split is already #23351-correct
and at the coverage floor. The remaining lever is **capture-bucket tuning**
(which prefill `num_tokens` to capture), validated multi-bucket above; unbucketed
shapes fall back to eager (Tier 4).

## Constraints / invariants (all satisfied)

- **No production runtime code changed except the P2 CP-under-PCG guard**
  (`attention.py` `_helix_cp_allgather_input`: `assert not is_torch_compiling()`
  inside the helix-CP branch). It adds one boolean check that always passes in
  normal eager execution (`is_torch_compiling()` is False) and never alters
  output bytes → decode + non-DSA + baseline remain **byte-identical when
  piecewise is off**. It fires only on the unsupported helix-CP-under-capture
  combo (fail-closed).
- **ABI-frozen `SparseMlaDecodeKvarnHotOp.cpp` / `hisparseKvarnBdrRead.cuh`:
  UNTOUCHED.** Baseline `deploy/disagg_pd_r20/prefill.yaml`: **UNTOUCHED**. The
  enable is an opt-in `prefill_piecewise.yaml` (P2, off by default, non-CP).
- **Correctness always vs the eager reference** (random weights), never
  self-compared between two piecewise variants. CP-under-PCG stays guarded.

## Deferred (the honest gating boundary)

- **End-to-end model ACCURACY** (gpqa-style, à la #23351's repeat-8) needs the
  real DeepSeek-V3.2-REAP-345B-NVFP4 weights, which are **metadata-only on 001**
  (18M: config + index.json; weight shards absent; the box can't fetch ~170 GB).
  Piecewise changes no numerics (Tier 2/4/5 prove bit-exactness on random
  weights), so accuracy is separable and is the ONLY thing that needs real
  weights — run it where the NVFP4 weights are resident, via the opt-in
  `prefill_piecewise.yaml` on a non-CP prefill worker.
- **Exact per-step latency / TTFT / throughput win under concurrency** (in1024/
  out1024 sweep) likewise needs the real model: the win lands between the
  compute-bound floor (1.0×) and the overhead-bound ceiling (4.72×), set by the
  real prefill regime. The structural launch-count win (94%) and bit-exactness
  are fully established here and are weight-independent.
- **MTP/NextN constraint:** the model is a NextN-Graft. Speculative/MTP draft is
  decode-side (variable accepted tokens), out of this prefill scope and unaffected
  (decode = full-graph). A future decode-side BCG (P4 scope-gap) would target it.
- **CP constraint:** the r20 prefill worker is CP=2 + LayerSplit; CP is
  unsupported under PCG (#23351). The opt-in enable is non-CP; the CP path is
  fail-closed guarded. Enabling the win on the production CP prefill worker
  requires either a non-CP prefill variant or CP-excluded-from-capture work.

## Files of record

- `blaise_perf/piecewise/gate_definitive.py` — the P3-DEFINITIVE harness
  (Tier 1 real partition + launch count, Tier 2 real-Backend bit-exact +
  latency, Tier 4 BCG/dynamic-condition, Tier 5 coverage + buckets).
- `blaise_perf/piecewise/overhead_probe.py` — the overhead-bound 4.72× probe.
- `blaise_perf/piecewise/gate_partition.py` — P3 partition-logic skeleton (clean
  + pathological early-cumsum collapse, risk-bounding).
- `deploy/disagg_pd_r20/prefill_piecewise.yaml` — opt-in enable (P2, off by
  default, non-CP).
- `tensorrt_llm/_torch/modules/attention.py` — P2 CP-under-PCG fail-closed guard.

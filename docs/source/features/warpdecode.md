# WarpDecode

WarpDecode is an opt-in decode-only MoE fast path for the Blaise DeepSeek V3.2 NVFP4 target shape. It is selected after routing has materialized `topk_ids` and `topk_weights`, and it falls back to the native MoE backend when its runtime guards do not match.

The current NVFP4 path has two explicitly named pieces: an explicit-tactic TRTLLMGen bridge and the future Cursor-style output-owned op. The bridge is now bucketed through c32 so the runtime has a fast, graph-shaped incumbent while the full Cursor kernel is being developed. It is not the final WarpDecode implementation: it still keeps the grouped MoE structure, so large-win optimization work remains focused on the route-owned Cursor contract. Unlike the BF16 compatibility path, the NVFP4 target path is allowed in CUDA-graph decode buckets because it runs after scheduler dispatch/remap has produced local slot ids.

Measured single-rank target-shape latencies on B200 (`hidden=7168`, `intermediate=2048`, `experts=128`, `topk=8`, finite FP8 scale tensors) were:

| Decode tokens | Native TRTLLM MoE min (ms) | Runtime bridge min (ms) | Runtime bridge median (ms) | Bridge speedup vs native | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 0.08732 | 0.06406 | 0.06429 | 1.36x | bridge incumbent |
| 2 | 0.08725 | 0.06746 | 0.06762 | 1.29x | bridge incumbent |
| 4 | 0.08764 | 0.06880 | 0.06889 | 1.27x | bridge incumbent |
| 8 | 0.08765 | 0.06870 | 0.06891 | 1.28x | bridge incumbent |
| 16 | 0.08715 | 0.07013 | 0.07024 | 1.24x | bridge incumbent |
| 32 | 0.08604 | 0.07183 | 0.07191 | 1.20x | bridge incumbent, not final |

The runtime bridge enables `TRTLLM_ENABLE_PDL=1` by default inside the NVFP4 custom op while preserving explicit environment overrides. Direct target-harness A/B showed c32 improving from 0.08419 ms min with PDL disabled to 0.07185 ms with PDL enabled. Runtime-op validation starts with the environment unset, confirms `pdl_env_after_runtime_op == "1"`, and matches the explicit runner exactly under finite scales (`max_abs=0`, cosine >= 0.99999988). The accepted tactic map is `{1: [8, 26], 2: [8, 75], 4: [8, 53], 8: [8, 53], 16: [8, 53], 32: [16, 52]}`. Finite-scale valid-only sweeps reconfirmed c16 and c32; best alternatives were below 1% and were not promoted.

The generic CuTeDSL grouped-GEMM wrapper was rejected as a production path because it measured 0.343147, 0.347054, 0.342320, 0.493565, 0.496066, and 0.498746 ms for 1, 4, 8, 16, 32, and 64 decode tokens respectively. Profiling showed the cost was dominated by the generic grouped GEMM and finalize kernels rather than by MoE sorting.

The selected NVFP4 crossover is graph-safe but not graph-dependent. A finite B200 capture/replay smoke with valid scale bytes matched eager exactly for 1, 4, and 8 tokens. Replay timing showed graph capture mainly helps the 1-token bucket: 1 token improved from 0.07495 ms eager to 0.04416 ms replay, while 2, 4, and 8 token buckets were approximately equal to eager.

The next optimization target is a true output-warp or hybrid CuTe/CZS NVFP4 kernel that consumes post-EPLB and post-dispatch top-k slot metadata directly, aligns with c1/c2/c4/c8/c16/c32 CUDA-graph buckets, and avoids expert padding, generic permutation/finalize, and full intermediate buffers. The NVFP4 guard intentionally allows external communication and EPLB slot ids because the scheduler hook runs after dispatch/remap; the BF16 compatibility path remains conservative. IKP on the direct 32-token path showed FC1/SwiGLU and FC2 BMM dominate runtime while routing/finalize are small, so the production path must change the compute and padding structure rather than only replacing the final scatter/combine stage. Route values are not cached across decode steps; only stable per-layer/per-bucket route and scratch buffer addresses are candidates for graph replay.

A small-tile grouped-MoE workaround was also rejected. The existing CuTe grouped kernels accept only 128- and 256-token tiles, so they cannot express exact-row decode work for c16/c32. A direct sweep showed tile sizes below 128 fail the kernel guard, and tile 128 measured roughly 0.32197 ms at 8 tokens, 0.48706 ms at 16 tokens, and 0.48933 ms at 32 tokens. That confirms the remaining path is not a grouped-kernel retune; it is a direct top-k/output-warp compute contract.

## Production Scheduler Contract

WarpDecode is optimized as a scheduler plus kernel path. The production target
keeps the existing hook after route selection, EPLB remap, external dispatch,
and NVFP4 activation quantization, then consumes final `token_selected_slots`
and `token_final_scales` directly. This is what keeps the feature compatible
with TP/EP, EPLB, and external communication while still removing the padded
expert-major grouped-MoE path.

The supported CUDA graph buckets for this campaign are c1, c2, c4, c8, c16,
and c32. Concurrency above 32 is out of scope until the c32 path is saturated.
The explicit TRTLLMGen bridge is an incumbent through c32, but it remains a
bridge rather than the final target. Once the Cursor NVFP4 op is available, c16
and c32 must prefer it when it beats the bridge, and force-mode tests must make
that selection visible rather than hiding it behind a generic grouped-MoE path.

Long-context performance must be validated in the full serving system. The MoE
kernel itself is decode-token shaped, but the selected implementation must
compose with attention/KV work, HISA/Indexer metadata, graph replay, and
request scheduling across 1k to 128k contexts. Systems optimizations such as
route metadata overlap, per-bucket scratch reuse, graph-stable buffers, and
launch fusion are part of the WarpDecode target, not optional follow-up work.

## Cursor-Style Target

The production target follows the Cursor WarpDecode shape rather than the traditional grouped-MoE shape. It flips the work axis from experts to outputs:

- Gate/up is a direct-top-k kernel. A CTA owns eight warps; each warp owns one intermediate neuron for one `(token, routed expert)` pair, streams the input activation once for gate and up, accumulates privately, applies `silu(gate) * up`, and writes one intermediate value.
- Down is output-owned. Each warp owns one `(token, output_dim)` scalar, loops over all top-k routed experts, folds the route weight into one FP32 accumulator, reduces with warp butterfly/shuffle, and writes the final scalar.
- The path must not form per-expert batches, pad expert token lists, scatter intermediate expert outputs, run a separate combine epilogue, or allocate an activation gather buffer. A candidate that still depends on those stages is a fallback/crossover path, not the final WarpDecode implementation.
- CUDA graph support is part of the production contract. Bucketed implementations should align with c1/c2/c4/c8/c16/c32 decode buckets and update graph-stable route/scratch buffers in place.
- Systems-level overlap remains in scope after the direct kernels are functional: route metadata preparation can overlap attention/KV metadata work, but route values must not be cached across decode steps.


## Rejected Cursor Bridges

The first direct output-owned down-projection prototype was a BF16 functional
bridge. It matched a Torch reference with minimum cosine similarity above
0.999998 for 1, 4, and 8 decode tokens, but it measured 0.106, 0.360, and
0.630 ms respectively. Increasing the CTA from four to eight output warps did
not materially change those timings. That bridge is therefore rejected as a
performance path: the production design must keep the Cursor output-owned
metadata contract while moving the heavy math onto NVFP4/CuTe tensor-core
primitives and eliminating the padded grouped-MoE stages.


The first direct-slot NVFP4 gate/up probe validated the c-bucket route/scratch
ABI and target-scale launch shape, but it is also rejected as a performance
path until the inner loop is replaced with tensor-core math. With sampled
`intermediate=256`, it measured 0.064, 0.182, 0.348, 0.658, and 1.283 ms for
c1, c4, c8, c16, and c32. With the full target `intermediate=2048`, c1 measured
0.354 ms for gate/up alone. The probe consumes `token_selected_slots` directly
and does not pad expert rows, so the remaining blocker is the scalar NVFP4
nibble-decode inner loop, not the route metadata contract.


A static top-k tensor-core floor using the existing CuTeDSL NVFP4 dense SwiGLU
primitive measured 0.105, 0.109, 0.112, 0.104, and 0.133 ms minimum latency for
c1, c4, c8, c16, and c32. This floor excludes dynamic route gather and uses the
same selected expert set for every token, so it is not deployable as
WarpDecode. It does show the right order of magnitude for the math core:
the custom kernel should keep the direct-slot route ABI while using a
CuTe/CZS tensor-core inner loop, not scalar per-warp dot products.

Scalar refinements after that floor did not recover the gap. Direct-slot
NVFP4 LUT, scale-grouped, and no-scale candidates all kept the Cursor ABI but
measured roughly 15.6-15.8 ms at c32 with sampled `intermediate=256`, far above
the grouped baselines and the tensor-core floor. The c32 IKP/nsys import showed
the candidate kernel itself consuming about 15.7 ms per launch. These candidates
are retained only as rejected evidence: the remaining production path is a
dynamic-route CuTe/CZS tensor-core implementation plus graph-stable scheduling,
not more scalar byte-decode tuning.

A compact selected-expert tensor-core hybrid was also measured and rejected.
It consumed dynamic `topk_ids`, selected the active expert rows, used the
existing CuTeDSL NVFP4 dense SwiGLU primitive for FC1, and used `nvfp4_gemm`
for FC2. The path was functionally runnable in the B200 buildtools container,
but it measured 0.912, 0.645, 1.114, 2.082, and 2.060 ms for c1/c4/c8/c16/c32,
respectively. A c32 stage split showed `unique(topk_ids)` at about 0.067 ms,
weight gather at about 1.012 ms, alpha-post scatter at about 0.055 ms, FC1 at
about 0.466 ms, and FC2 at about 0.483 ms. The synthetic c32 route touched all
128 experts, so compact-selected-expert materialization degenerated into
copying the whole expert slice. This path is retained only as negative systems
evidence: the production kernel must load routed expert rows directly from
`topk_ids` inside the selected kernel and must not build compact `w13`/`w2`
tensors or an `alpha_post` matrix before launch.

A route-grouped tensor-core FC1 bridge was then measured to remove compact
weight materialization while preserving the direct `token_selected_slots`
input. It groups exact routed rows by expert and invokes the existing NVFP4
CuTeDSL FC1 on expert weight slices. This proves tensor-core legality only by
grouping rows that share a B tile, but it is rejected for performance. In the
best-case `slot8` route pattern, where all buckets touch only eight experts,
c1/c4/c8/c16/c32 measured 1.958/1.922/1.936/1.903/1.974 ms for FC1 alone. In
the production-hard `worst128` pattern, c1/c4/c8/c16/c32 measured
1.957/7.087/13.893/27.410/27.837 ms. The conclusion is structural: arbitrary
per-row expert selection breaks shared-B tensor-core legality. Tensor-core
candidates must either group rows or materialize/gather B, and both reintroduce
the overheads WarpDecode is meant to remove unless they are fused into one
custom route-aware kernel. Repeated CuTeDSL calls by expert are a closed
branch.

The direct-slot PTX FP4 decoder candidate is the current best Cursor-style
gate/up subpath. Building for `sm_100a` allowed the hardware
`cvt.rn.f16x2.e2m1x2` instruction; `sm_100` rejected that instruction. With
the same direct `topk_ids` ABI and no grouped padding, sampled
`intermediate=256` measured 0.012, 0.031, 0.064, 0.111, and 0.212 ms for
c1/c4/c8/c16/c32. Full `intermediate=2048` measured 0.064, 0.211, 0.410,
0.807, and 1.601 ms. IKP/nsys import for c32 showed a single
`cursor_nvfp4_gate_up_ptx_kernel` launch with median duration about 1.598 ms.
This beats the hand-decoder direct-slot probe by roughly 5-6x, but gate/up
alone remains too slow for c32.

A branchy multi-neuron-per-warp variant was rejected. It tried to amortize
activation decode across 2, 4, or 8 adjacent intermediate neurons per warp, but
the added registers and control flow regressed c32 full-intermediate timings to
roughly 8.995, 7.933, and 7.401 ms; the generalized n=1 path also regressed to
about 10.471 ms. The next candidate should use compile-time specialization or
a different output-owned fusion, not a runtime-branching multi-neuron warp.
Compile-time specialization was tested next and also rejected: specialized
1/2/4/8 neurons per warp measured about 1.685/2.219/2.816/7.376 ms at c32
full-intermediate. Adjacent-neuron amortization therefore loses even without
runtime branching. The next candidate must change the fusion/output mapping
rather than asking one warp to carry more intermediate neurons.

The first direct output-owned down kernel was implemented as the second half of
the Cursor contract. It folds `token_final_scales` into the warp accumulator,
loads routed NVFP4 down rows directly from `topk_ids`, writes unique BF16 output
scalars, and uses no grouped-MoE padding or combine buffer. Sampled
`intermediate=256` measured 0.021, 0.062, 0.119, 0.232, and 0.459 ms for
c1/c4/c8/c16/c32. Full `intermediate=2048` measured 0.128, 0.448, 0.865,
1.703, and 3.392 ms. This validates the ABI but is not production-fast; direct
scalar down over the full intermediate dimension dominates the two-kernel
prototype. The next design must reduce the down pass cost or use a tensor-core
subtile/fusion that preserves direct routing without compact expert
materialization.

The next down-pass round corrected the prototype toward the Cursor paper shape
by reading BF16 intermediate activations instead of FP32 scratch. Full
`intermediate=2048` timings improved to 0.070, 0.242, 0.459, 0.893, and
1.773 ms for c1/c4/c8/c16/c32. A vector-output schedule then let one warp
compute two adjacent output dimensions while sharing each BF16 intermediate
load. That `outputs_per_warp=2` candidate measured 0.074, 0.212, 0.394,
0.713, and 1.376 ms for c1/c4/c8/c16/c32, and IKP/nsys showed the c32 kernel
itself at a stable median of about 1.365 ms with grid 14336, block 256, and
32 registers per thread. This is accepted as the current direct down prototype
incumbent, but it is not a production promotion because the native TRTLLMGen
c16/c32 path is still faster.

The current full Cursor ABI prototype combines the best direct gate/up kernel
with the `outputs_per_warp=2` output-owned down kernel. It consumes
`token_selected_slots` directly, avoids grouped-MoE padding, avoids compact
weight materialization, and writes the final BF16 output. B200 c1/c4/c8/c16/c32
latencies were 0.166/0.433/0.840/1.564/3.055 ms. Only c1 beats the native
TRTLLM reference; c4 and above are rejected for production. The c32 `nsys`
kernel summary recorded 43 instances of each direct kernel: gate/up median
1.679 ms (52.6% of GPU kernel time) and down median 1.370 ms (42.8%). This is
the current direct-route baseline for further IKP/CuTe iteration.

Folding `token_final_scales` into the gate/up scratch write was tested and
rejected. The intent was to remove the route-weight multiply from the down
inner loop. It improved c1 from 0.166 to 0.149 ms, but c4/c8/c16/c32 regressed
to 0.462/0.847/1.585/3.082 ms. The c32 regression shows the route-scale
multiply is not the limiting down cost; memory traffic and FP4 decode still
dominate.

A broader dense local-expert tensor-core lower bound was measured because
direct sparse scalar kernels were too slow. For 16 local experts, all-local
expert FC1 measured about 0.10 ms at every c-bucket, and FC2 measured about
0.096-0.10 ms. For 128 experts, FC1 measured about 0.48-0.53 ms and FC2 about
0.30 ms. This remains a lower bound only. The runtime local-dense wrapper was
retested under target EP8 assumptions and rejected: c8/c16/c32 measured about
0.266/0.267/0.266 ms minimum versus the explicit bridge at
0.084/0.085/0.082 ms, and it was not functionally equivalent under the current
alpha/scale/layout contract. The local-dense wrapper is not registered in
production; it is retained only as negative evidence in the benchmark artifacts.
DenseGEMM remains useful as a tensor-core shape reference, not as a production
WarpDecode crossover in the current wrapper.

Other down-pass variants were rejected. `outputs_per_warp=3/4/8` regressed
c32 to roughly 1.540/1.555/1.960 ms. Manual BF16 pair-load conversion and
explicit `__ldg` read-only cache hints also regressed c32 to about 1.457 and
1.444 ms. A split-K design inspired by NVFP4 GEMV K-mode parallelization was
tested with `outputs_per_warp=2`; `k_splits=2/4` regressed c32 to about
2.050/2.109 ms because the partial buffer and finalize kernel outweighed the
extra reduction parallelism for this shape.

Gate/up CTA-shared activation decode was also rejected. Although each CTA's
eight warps normally cover eight adjacent intermediate neurons for the same
token-route pair, sharing the decoded activation chunk through shared memory
introduced enough synchronization overhead to regress full `intermediate=2048`
timings to about 1.071 ms at c16 and 2.124 ms at c32. The current best gate/up
subpath remains the direct-slot PTX FP4 decoder; further progress likely needs
a route-compatible CuTe/CZS tensor-core design rather than CTA-level scalar
reuse.

Once a deployable Cursor NVFP4 op is registered, c16 and c32 must select it or
fail under force mode. They must not silently fall back to the current
`n <= 8` crossover path. Long-context performance must be validated in the
full serving matrix because WarpDecode is context-independent math that still
runs under long-context KV, IndexCache, and scheduler pressure.

## Direct Route Metadata Candidate

`benchmarks/python/warpdecode_direct_metadata_candidate.py` validates a
standalone generator for the Cursor route metadata tensors from final
`token_selected_slots`. The validator intentionally checks the grouped-kernel
contract rather than byte identity with `moe_sort`: every expanded local route
must map to a valid permuted row, that row must live in a tile for the selected
local expert, and `permuted_idx_to_expanded_idx` must invert the selected
`expanded_idx_to_permuted_idx` over valid rows. Exact `moe_sort` ordering is not
required.

On B200 the candidate is functionally valid under the target unique-route
contract, but it is slower as a standalone replacement for `moe_sort`:

| Decode tokens | `moe_sort` min ms | Direct metadata min ms | Decision |
| ---: | ---: | ---: | --- |
| 1 | 0.0161 | 0.0264 | keep only as fused design input |
| 4 | 0.0160 | 0.0262 | keep only as fused design input |
| 8 | 0.0166 | 0.0263 | keep only as fused design input |
| 16 | 0.0160 | 0.0263 | keep only as fused design input |
| 32 | 0.0163 | 0.0349 | keep only as fused design input |

Artifact: `artifacts/warpdecode/direct_metadata_candidate_20260601.json`.

This closes standalone direct metadata as a promotion path, but not as a system
idea. The useful next step is to fuse this metadata generation with route-alpha
or with the eventual output-owned Cursor kernel so it does not add an extra
launch. A promoted implementation must handle nonlocal slots and slot offsets
or explicitly guard them out.

## Fused Metadata And Route-Alpha Candidate

`benchmarks/python/warpdecode_fused_metadata_alpha_candidate.py` combines the
direct route-metadata generator with the local route-alpha matrix generation in
one target-shape Triton launch. It validates the same grouped-kernel metadata
contract as the standalone metadata candidate and checks exact route-alpha
values against the target unique-route scatter contract.

The fused launch is correct for the target route contract, but it is not a
standalone win versus `moe_sort` plus the route-alpha helper:

| Decode tokens | `moe_sort + alpha` min ms | Fused metadata+alpha min ms | Decision |
| ---: | ---: | ---: | --- |
| 1 | 0.0299 | 0.0298 | tie only |
| 4 | 0.0296 | 0.0299 | reject standalone |
| 8 | 0.0300 | 0.0297 | tie only |
| 16 | 0.0294 | 0.0300 | reject standalone |
| 32 | 0.0293 | 0.0431 | reject standalone |

Artifact:
`artifacts/warpdecode/warpdecode_fused_metadata_alpha_candidate_20260601.json`.

This closes the extra-launch metadata/alpha fusion branch. The result remains
useful only if the metadata and alpha stores are fused into the eventual
output-owned Cursor kernel itself, where they do not add a separate launch.

## Route-Locality Tactic Selection Probe

`benchmarks/python/warpdecode_route_locality_policy_probe.py` separates a
no-overhead route-locality upper bound from a hot-path dynamic selector. The
question was whether the explicit TRTLLMGen bridge could safely choose better
c16/c32 tactics for clustered routes without moving to a full Cursor kernel.

The upper bound is real for clustered synthetic routes. Under finite scales,
route-best tactics were output-equivalent to the incumbent bridge with
`max_abs=0.0` and cosine >= 0.99999988. Examples:

| Decode tokens | Route pattern | Incumbent tactic | Route-best tactic | Incumbent min ms | Route-best min ms |
| ---: | --- | --- | --- | ---: | ---: |
| 16 | `single_expert` | `[8, 53]` | `[32, 18]` | 0.042986 | 0.042277 |
| 16 | `slot8` | `[8, 53]` | `[16, 48]` | 0.054014 | 0.051843 |
| 32 | `slot8` | `[16, 52]` | `[32, 52]` | 0.050629 | 0.045303 |

The hot-path selector is rejected. Computing route locality from
`topk_ids` inside Python with `torch.unique(topk_ids).numel()` costs roughly
0.0366-0.0410 ms by itself and makes every dynamic-selector path slower than
fixed incumbent tactics. This cannot be promoted.

The C++ runner also does not currently expose a cheap equivalent. `tileN` and
config are selected before the TRTLLMGen routing kernel populates
`num_tokens_per_expert`, and `tileN` affects workspace sizing and runner
selection. Retrofitting route-sensitive tactics therefore requires a real
runner/scheduler metadata change, not a Python branch after routing.

Artifact: `artifacts/warpdecode/route_locality_policy_probe_20260602.json`.

Decision: keep the current fixed PDL+explicit-tactic bridge as the production
incumbent. Route-locality tactics remain a future system optimization only if
route metadata is produced no-sync by routing/EPLB/scheduler, or if the final
Cursor kernel consumes the route tensor directly and removes the need for a
Python tactic decision.

## Activation Predecode Candidate

`benchmarks/python/warpdecode_cursor_nvfp4_gate_up_predecode_x_prototype.py`
tested a direct-route systems candidate: predecode token NVFP4 activations once
into a graph-stable BF16 token buffer, then run output-owned gate/up over
`token_selected_slots`. This avoids grouped MoE, expert-major padding, and
compact expert gather. The intent was to remove repeated activation FP4 decode
work from every routed intermediate neuron.

The candidate is correct versus the direct PTX gate/up baseline under the same
scale-probe semantics (`max_abs=0.0`; cosine 1.0 except c16 at 0.99999994), but
it is slower in every measured bucket:

| Decode tokens | Direct PTX gate/up min ms | Predecode+gate/up min ms | Gate-only with predecoded x min ms | Decision |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.067567 | 0.078662 | 0.074999 | reject |
| 8 | 0.430409 | 0.495949 | 0.491989 | reject |
| 16 | 0.847901 | 0.974021 | 0.970385 | reject |
| 32 | 1.682645 | 1.928523 | 1.924527 | reject |

Artifact: `artifacts/warpdecode/cursor_gate_up_predecode_x_full2048_20260602.json`.

Decision: do not pursue activation predecode as the WarpDecode fix. It adds a
kernel launch and expands the activation stream from packed NVFP4 to BF16, so
it increases bandwidth pressure without reducing enough of the direct-kernel
cost. The final route-owned path still needs a deeper compute redesign or a
TRTLLMGen runner extension.

## Direct Explicit-Runner Dispatch

`artifacts/warpdecode/runtime_op/direct_vs_custom_dispatch_20260602.json`
checks the accepted explicit-tactic TRTLLMGen bridge against the registered
`torch.ops.trtllm.warp_decode_nvfp4_moe` custom-op wrapper on the same finite
scale tensors. Outputs are equivalent (`max_abs=0.0`, cosine >= 0.99999994).

| Decode tokens | Custom-op wrapper min ms | Direct explicit runner min ms | Direct speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.062297 | 0.041254 | 1.510x |
| 2 | 0.066968 | 0.066684 | 1.004x |
| 4 | 0.068533 | 0.068331 | 1.003x |
| 8 | 0.068534 | 0.068312 | 1.003x |
| 16 | 0.070129 | 0.069947 | 1.003x |
| 32 | 0.072446 | 0.072210 | 1.003x |

Decision: promote the direct explicit runner as the selected eager NVFP4 bridge
while keeping the registered custom op available for explicit callers and fake
shape support. This is not the final Cursor-style output-owned kernel; it is a
production-safe dispatch cleanup on the current incumbent. Synthetic
`try_run_warp_decode` timings include Python guard cost and are not the kernel
metric; production CUDA graph replay captures the selected kernel sequence after
guard execution.

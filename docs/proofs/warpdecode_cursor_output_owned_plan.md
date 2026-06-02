# WarpDecode Cursor Output-Owned Plan

This is the production direction after rejecting grouped-MoE retuning and the
CuTeDSL grouped wrapper. The current runtime keeps the guarded TRTLLMGen NVFP4
crossover for `n <= 8`; the next path should be a direct top-k/output-owned
implementation selected only after it beats the baselines.

## Runtime Contract

The direct path consumes post-EPLB/post-dispatch route tensors:

- `token_selected_slots`: `[bucket_tokens, top_k]`, `int32`
- `token_final_scales`: `[bucket_tokens, top_k]`, `float32` or `bfloat16`
- `x`: `[bucket_tokens, hidden/2]`, NVFP4 packed
- `x_sf`: `[bucket_tokens, hidden/scale_vector_size]`, FP8 scale bytes

The graph buckets are fixed at `c1/c2/c4/c8/c16/c32`. Route values are updated
each decode step, but the route tensor addresses, activation scale buffer, and
scratch buffers are bucket-stable so CUDA graph replay has a fixed ABI.

## Scheduler And Systems Contract

WarpDecode is a decode-scheduler feature, not only a kernel swap. The
production Cursor path must preserve these system-level guarantees while the
CuTe/CZS kernels evolve:

- The hook is after routing, EPLB slot remap, external-communication dispatch,
  and NVFP4 activation quantization. The kernel consumes final slot ids, not
  global expert ids that still need remapping.
- Concurrency is capped at 32 simultaneous decode tokens per graph bucket for
  this production campaign. The required buckets are exactly
  `c1/c2/c4/c8/c16/c32`; there is no `c64` fallback path in scope.
- The currently selected TRTLLMGen crossover for `n <= 8` is only an incumbent
  until the Cursor op is available. Once a deployable Cursor op exists, c16 and
  c32 must select that op or fail loudly in force mode; they must not silently
  return to padded grouped MoE because the old crossover limit was left in the
  guard.
- Route/scales buffers, intermediate scratch, and output scratch are
  per-layer/per-bucket owned buffers with stable addresses. Values are refreshed
  every decode step; graph replay captures shapes and addresses only.
- Route metadata preparation may overlap attention/KV metadata preparation. It
  must not introduce a synchronization point ahead of attention unless IKP/nsys
  proves the overlap is harmful.
- The kernel work is allowed to change internally from literal output-warp
  scalar ownership to a hybrid direct-slot tensor-core layout, but it must keep
  the direct route ABI and eliminate expert-major padding, sort, and
  scatter/combine from the selected path.

Proposed runtime addition, intentionally not implemented in this worker to
avoid conflict with the kernel lane: add a cursor-op availability predicate and
policy bit to the NVFP4 guard. When the predicate is true, the guard should
allow any bucket returned by `get_cursor_warp_decode_plan(num_tokens)` through
c32 and record a distinct selection reason such as `nvfp4_cursor_op`.

## Kernel Composition

`moe_gate_up_3d_batched`

- CTA has 8 independent warps.
- Each warp owns one intermediate neuron for one `(token, routed expert)` pair.
- The warp streams the token activation once, reads routed gate/up expert rows,
  accumulates gate and up in private FP32 registers, applies `silu(gate) * up`,
  and writes one exact intermediate value.

`moe_down_3d_batched`

- Each warp owns one `(token, output_dim)` scalar.
- The warp loops over all top-k experts, streams that expert's down row and the
  exact intermediate activations, folds the route weight into a single FP32
  accumulator, reduces with warp shuffle/butterfly, and writes the final scalar.

The path must not allocate or consume expert-major batches, padded grouped-MoE
rows, activation gather buffers, per-expert output buffers, or a scatter/combine
finalize stage.

## Scratch Plan

For target shape `hidden=7168`, `intermediate=2048`, `top_k=8`:

| Bucket | Exact rows | Gate/up warps | Down warps | Intermediate scratch |
| ---: | ---: | ---: | ---: | --- |
| 1 | 8 | 16,384 | 7,168 | `[1, 8, 2048]` |
| 2 | 16 | 32,768 | 14,336 | `[2, 8, 2048]` |
| 4 | 32 | 65,536 | 28,672 | `[4, 8, 2048]` |
| 8 | 64 | 131,072 | 57,344 | `[8, 8, 2048]` |
| 16 | 128 | 262,144 | 114,688 | `[16, 8, 2048]` |
| 32 | 256 | 524,288 | 229,376 | `[32, 8, 2048]` |

This removes the current n32 inflation from roughly 1152 padded rows to 256
exact expanded rows. The scratch buffer can be reused per layer and bucket; the
future CZS/CuTe kernel should keep the same public contract even if it packs the
intermediate scratch more tightly.

## Overlap And Capture

The route metadata producer should run after routing/EPLB has selected slots,
but its address-stable copies can overlap attention/KV metadata preparation.
The graph captures only bucket shapes and buffer addresses, never route values.

## Current Measurements To Beat

`benchmarks/python/warpdecode_target_harness.py` is the current inner-loop
measurement harness. It constructs the target EP8-local NVFP4 tensors directly
without the `bench_moe` MPI launcher overhead: `hidden=7168`,
`intermediate=2048`, `num_experts=128`, `local_num_experts=16`, `top_k=8`,
and graph buckets `c1/c2/c4/c8/c16/c32`.

Latest B200 round-robin route results, min latency in ms:

| Bucket | Native TRTLLMGen | Explicit tactic | `do_finalize=false` |
| ---: | ---: | ---: | ---: |
| 1 | 0.09422 | 0.04288 | 0.09208 |
| 2 | 0.09466 | 0.07014 | 0.09249 |
| 4 | 0.09437 | 0.07141 | 0.09224 |
| 8 | 0.09355 | 0.06892 | 0.09240 |
| 16 | 0.09407 | 0.07077 | 0.09245 |
| 32 | 0.09533 | 0.07945 | 0.09267 |

The explicit tactic wrapper is now the measured bridge incumbent through
`c32`. This is useful for production-shaped routing and graph buckets, but it
is still grouped-MoE-shaped and is not the large-win Cursor implementation.
`c16/c32` continue to require a route-owned Cursor/hybrid path that beats this
bridge and removes grouped sort/padding/materialization pressure.

`nsys` on the controlled `c32` harness shows FC1/SwiGLU at 44.4% of GPU
kernel time and FC2 at 24.4%.  Finalize is 7.9%, and routing is 4.5%.  The
next measured candidate must replace the compute layout and expert-major setup,
not only stage fusion around grouped-MoE kernels.

## Parallelism Composition

External communication and EPLB are compatible with the NVFP4 Cursor contract
because the WarpDecode hook is after scheduler dispatch and load-balancer remap.
The kernel consumes `token_selected_slots`, not pre-remap expert ids. BF16
compatibility remains guarded off for external communication and EPLB until it
has a separate proof.


## Direct-Slot Gate/Up Probe

`benchmarks/python/warpdecode_cursor_nvfp4_gate_up_prototype.py` is a
non-promoted launch proof for the gate/up half of the Cursor contract. It
consumes final `token_selected_slots`, writes exact `[bucket, top_k,
intermediate]` scratch, and records `uses_grouped_moe=false` and
`pads_expert_rows=false`. Its scalar NVFP4 probe is intentionally rejected for
performance: sampled `intermediate=256` measured 0.064/0.182/0.348/0.658/1.283
ms at c1/c4/c8/c16/c32, and full `intermediate=2048` measured 0.354 ms at c1.
The next candidate must keep this ABI but replace the scalar decode loop with
CuTe/CZS tensor-core block-scaled math.


## Static Top-K Tensor-Core Floor

`benchmarks/python/warpdecode_topk_dense_floor.py` reuses the existing
CuTeDSL NVFP4 dense SwiGLU primitive with `expert_count=8` as a tensor-core
lower-bound study. It is not a deployable WarpDecode kernel because it excludes
dynamic route gather and uses one static top-k expert set across the batch.
Minimum latencies for c1/c4/c8/c16/c32 were
0.105/0.109/0.112/0.104/0.133 ms. This makes the next target precise: preserve
the direct `token_selected_slots` ABI from the gate/up probe, but implement the
inner math with CuTe/CZS block-scaled tensor-core tiles.

## Rejected Scalar Refinements

`benchmarks/python/warpdecode_cursor_nvfp4_gate_up_lut_prototype.py` and the
scale-grouped/no-scale variants kept the same direct-slot ABI, exact c-buckets,
and no-padding route contract, but stayed scalar. They are rejected. At sampled
`intermediate=256`, the LUT candidate measured
0.700/2.280/4.155/7.981/15.666 ms for c1/c4/c8/c16/c32, the scale-grouped
candidate measured 0.678/2.276/4.227/8.053/15.808 ms, and the no-scale
candidate measured 0.723/2.230/4.093/7.924/15.561 ms. IKP/nsys import for the
LUT c32 run shows the candidate kernel itself taking about 15.7 ms per launch
with 8192 CTAs, 256 threads, 58 registers/thread, and no shared memory. The
lesson is structural: byte-level scalar decode and per-neuron warp ownership
are not enough for the NVFP4 target. The route ABI is retained; the math core
must move to the CZS-proved SM100 NVFP4 tensor-core tile contract.

## CZS Tensor-Core Contract

`docs/proofs/warpdecode_nvfp4_b200_target_inner_slice_czs_module.json` is the
current CuTe/CZS legality anchor for the next kernel. CZS proved the row and
scale vectorization obligations, `topk_ids` vectorization, output BF16 stores,
and the SM100 MXF4/NVF4 MMA operands for both FC1 and FC2:

| Obligation class | Status |
| --- | --- |
| Layout legality | 8/8 Proved |
| MMA operand legality | 4/4 Proved |
| Vectorization | 8/8 Proved |

The proof was rerun in the `local/dynamo-trtllm-optrt-custom:20260531`
container on 2026-06-02 and returned `20 Proved | 0 Disproved | 0 Unknown`.

The next implementation must instantiate this contract with dynamic
post-dispatch `token_selected_slots` and graph-stable c1/c2/c4/c8/c16/c32
workspace addresses.

## Rejected Compact Tensor-Core Hybrid

A compact selected-expert hybrid was tested as the first dynamic-route
tensor-core bridge. It used `topk_ids` to gather selected expert weights,
constructed an `alpha_post` matrix, called
`cute_dsl_nvfp4_dense_gemm_swiglu_moe_blackwell` for FC1, and called
`nvfp4_gemm` for FC2. It is rejected because it keeps the wrong systems shape:
at c32 the synthetic route spans all 128 experts, so the "compact" gather
copies effectively the whole expert slice before every decode step.

Measured B200 latencies for c1/c4/c8/c16/c32 were
0.912/0.645/1.114/2.082/2.060 ms. The c32 split was:

| Stage | c32 latency (ms) |
| --- | ---: |
| `unique(topk_ids)` | 0.067 |
| compact weight gather | 1.012 |
| alpha-post scatter | 0.055 |
| FC1/SwiGLU CuTeDSL | 0.466 |
| FC2 NVFP4 GEMM | 0.483 |

The next kernel must therefore avoid compact `w13`/`w2` tensors and avoid the
`alpha_post` matrix. It should load routed expert rows directly from
post-dispatch `token_selected_slots` inside the kernel. Tensor-core subtiles
remain required where route grouping makes them legal, but a tensor-core
wrapper that first materializes compact expert-major tensors is not a
production WarpDecode candidate.

## Rejected Route-Grouped Tensor-Core Bridge

`benchmarks/python/warpdecode_route_grouped_tensorcore_prototype.py` tested the
next legal tensor-core compromise after the compact bridge. It preserves the
direct `token_selected_slots` input and does not materialize compact expert
weight tensors. Instead, it groups exact routed rows by selected expert and
calls the existing NVFP4 CuTeDSL dense SwiGLU FC1 on the expert weight slice.

The best-case `slot8` pattern, where every bucket touches only eight experts,
still measured 1.958/1.922/1.936/1.903/1.974 ms for c1/c4/c8/c16/c32 FC1
alone. The production-hard `worst128` pattern measured
1.957/7.087/13.893/27.410/27.837 ms because c16/c32 touch all 128 experts and
the path degenerates into many tiny tensor-core launches.

This closes the repeated-CuTeDSL group-by-expert branch. It confirms the core
CuTe/CZS constraint: SM100 block-scaled tensor cores need a shared B tile
across the M rows. Arbitrary dynamic route rows do not satisfy that unless the
implementation groups rows or materializes B. Future tensor-core work must be
a single custom route-aware kernel that fuses grouping with compute and avoids
Python launch loops, or it should stay on the Cursor output-owned PTX path and
optimize memory bandwidth directly.

## Direct-Slot PTX FP4 Gate/Up

The direct-slot PTX FP4 decoder candidate is the current best measured
Cursor-style gate/up subpath. It keeps the direct `token_selected_slots` ABI,
does not build compact expert tensors, does not sort/pad expert-major rows,
and uses the Blackwell `cvt.rn.f16x2.e2m1x2` instruction for FP4 decode. This
requires compiling for `sm_100a`; `sm_100` rejects the instruction.

Measured sampled `intermediate=256` c1/c4/c8/c16/c32 latencies were
0.012/0.031/0.064/0.111/0.212 ms. Measured full `intermediate=2048` latencies
were 0.064/0.211/0.410/0.807/1.601 ms. IKP/nsys import for c32 found a single
`cursor_nvfp4_gate_up_ptx_kernel` launch with median duration about 1.598 ms.
This is a strong replacement for the hand-decoded scalar probe, but gate/up
alone is still too slow for c32 and still lacks the down/final accumulation.

A runtime-branching multi-neuron-per-warp variant was rejected. It attempted to
reuse one decoded activation vector across 2, 4, or 8 adjacent intermediate
neurons, but register pressure and branch control made c32 full-intermediate
timings much worse: about 8.995, 7.933, and 7.401 ms, with the generalized n=1
path at about 10.471 ms. Future attempts at amortizing activation decode should
be compile-time specialized or folded into a different fused output-owned
mapping rather than expressed as a runtime-branching warp loop.

Compile-time specialization was tested and rejected too. Specialized 1, 2, 4,
and 8 neurons per warp measured about 1.685, 2.219, 2.816, and 7.376 ms at c32
full-intermediate. That makes adjacent-neuron amortization a local dead end:
the next optimization must avoid materializing all gate/up intermediate values
or change the down/final accumulation ownership, not simply bundle more
intermediate neurons into the same warp.

## Direct Output-Owned Down

The first direct down kernel now matches the Cursor ABI: each warp owns one
`(token, output_dim)` scalar, loops over top-k routed experts, folds
`token_final_scales` into one FP32 accumulator, reads NVFP4 down rows directly,
and writes BF16 output without a combine buffer. It validates the route and
output ownership contract but is not production-fast.

Measured sampled `intermediate=256` c1/c4/c8/c16/c32 latencies were
0.021/0.062/0.119/0.232/0.459 ms. Measured full `intermediate=2048` latencies
were 0.128/0.448/0.865/1.703/3.392 ms. Combined with the best PTX gate/up
subpath, c32 is roughly five milliseconds, so the next design must reduce the
down pass cost. Candidate directions are: a tensor-core subtile that preserves
direct routed rows, route-pattern grouping inside the kernel without compact
expert tensor materialization, or a more aggressive fusion that avoids writing
and rereading all gate/up intermediate activations.

## Combined Direct Cursor ABI Baseline

`benchmarks/python/warpdecode_cursor_nvfp4_two_kernel_prototype.py` combines
the best direct-slot PTX gate/up kernel and the `outputs_per_warp=2`
output-owned down kernel behind one benchmarked Cursor ABI. It still launches
two kernels, but the dataflow is the production target shape: direct
`token_selected_slots`, no grouped-MoE padding, no compact expert weights, BF16
intermediate scratch, and final BF16 output.

Measured c1/c4/c8/c16/c32 B200 latencies were
0.166/0.433/0.840/1.564/3.055 ms. Only c1 beats the native TRTLLM reference
for the same target shape. The c32 `nsys` profile recorded 43 instances of each
direct kernel: `cursor_nvfp4_gate_up_ptx_kernel` median 1.679 ms and
`cursor_nvfp4_down_ptx_kernel` median 1.370 ms. Gate/up is 52.6% of GPU kernel
time, down is 42.8%, and random setup kernels account for the rest. Future
work must therefore reduce both compute halves; optimizing only down or only
gate/up cannot recover the native c16/c32 gap.

Route-weight prescaling was tested as the first combined-path refinement:
gate/up writes `silu(gate) * up * token_final_scales`, and down treats route
weights as already folded into the BF16 scratch. It is rejected. Timings were
0.149/0.462/0.847/1.585/3.082 ms for c1/c4/c8/c16/c32. Only c1 improved; c32
regressed from 3.055 to 3.082 ms. The repeated route-weight multiply is not the
dominant down bottleneck.

`docs/proofs/warpdecode_route_compatible_tensorcore_proof.md` and
`benchmarks/python/warpdecode_route_compatible_tensorcore_feasibility.py`
close the generic route-compatible tensor-core branch. CZS proved the existing
inner NVFP4 tensor-core slice module 20/20 inside the B200 buildtools
container, so the tensor-core math tile is legal when its shared-B contract is
satisfied. The blocker is route compatibility, not MMA legality.

For arbitrary `token_selected_slots`, an SM100 MMA CTA may only combine M rows
when every row shares the same selected expert and N tile, because the B operand
tile is shared. Preserving correctness without compacting weights therefore
requires grouping rows by expert or reintroducing expert-major padding. The
feasibility probe modeled a one-launch route-aware kernel that avoids Python
per-expert launches and compact B tensors but still assigns legal CTAs by
`(expert, N tile)`. At c32, `worst128` routing touched all 128 experts with two
rows per expert: M-tile utilization was 0.015625, with 2048 FC1 CTAs and 3584
FC2 CTAs. Random routing touched 114 experts with utilization 0.01754. Even the
idealized `slot8` case reached only 0.25 utilization at c32.

Recommendation: keep tensor-core paths only for route-specialized high-reuse
crossovers where CZS proves the shared-B contract, and keep the generic
WarpDecode path on the direct Cursor output-owned ABI.

`benchmarks/python/warpdecode_cursor_nvfp4_two_kernel_cta_sweep.py` tested the
current best direct two-kernel Cursor prototype with 4/8/16/32 warps per CTA.
The route ABI, no-padding/no-compact contract, gate/up kernel, and down kernel
were unchanged; only CTA granularity changed.

| Warps/CTA | c16 ms | c32 ms | Decision |
| ---: | ---: | ---: | --- |
| 4 | 1.637 | 3.156 | Reject |
| 8 | 1.563 | 3.048 | Incumbent |
| 16 | 1.749 | 3.407 | Reject |
| 32 | 2.319 | 4.536 | Reject |

The eight-warp CTA remains the best measured output-owned granularity. Larger
CTAs reduce block count but lose enough scheduling/occupancy quality to regress
the decode buckets that matter.

## Shape-Safety Gate

Every WarpDecode kernel candidate must be one of two things:

- exact target-shape code for DeepSeek V3.2 NVFP4 MoE
  (`hidden=7168`, `intermediate=2048`, `experts=128`, `top_k=8`,
  `scale_vec=16`, with the production EP/TP partitioning reflected where the
  kernel sees local experts), selected only behind explicit runtime guards and
  a safe fallback; or
- a truly arbitrary-model-safe implementation whose strides, dimensions,
  packing, and scale layout are derived from runtime tensors and covered by
  tests across more than the target shape.

Target-shape kernels must not look generic. Generic-looking prototypes that
silently assume the target model are rejected until they either add explicit
shape guards or become genuinely shape-derived.

## Dense Local-Expert Tensor-Core Lower Bound

The next systems-level check measured the existing CuTeDSL DenseGEMM shape as
a lower bound for local-expert decode. This is not a Cursor-sparse kernel, but
it answers whether tensor-core all-local-expert math can beat the scalar
direct route path when EP keeps the local expert count small.

For 16 local experts, dense FC1 measured roughly 0.10 ms across c1/c4/c8/c16
/c32, and FC2 over the corresponding expert scratch measured roughly
0.096-0.10 ms. For 128 experts, FC1 measured about 0.48-0.53 ms and FC2 about
0.30 ms. Therefore, if the target production EP8 path presents 16 local expert
slots to the backend, the tensor-core all-local-expert lower bound is about
0.20 ms at c32 before scheduler/routing overhead. That is below the current
native c32 reference and far below scalar Cursor c32. The next implementation
step should verify real `expert_size_per_partition` under the target config and
test/adapt `DenseGEMMFusedMoE` as the decode backend for small local expert
counts, while keeping the direct Cursor prototypes as fallback evidence rather
than the only optimization lane.

The target EP8 local-expert candidate was rerun in the runtime image with
installed TensorRT-LLM bindings rather than the source-only buildtools image.
`benchmarks/python/warpdecode_local_dense_tensorcore_candidate.py` with
`local_experts=16`, balanced local routes, and exact DeepSeek V3.2 target shape
measured c16/c32 minimum latencies of 0.308/0.313 ms and medians of
0.308/0.316 ms. This is still slightly behind the best native c32 reference
near 0.2826 ms, but it is within the same class and much faster than the
direct two-kernel Cursor prototype at 3.048 ms.

This branch is only valid as a target-shape production candidate when the
runtime backend really has `expert_size_per_partition=16`, target NVFP4
weights/scales, `top_k=8`, `scale_vec=16`, and graph buckets through c32. It
must be selected through explicit guards or through the existing
`DenseGEMMFusedMoE` backend configuration; it must not be exposed as an
arbitrary WarpDecode kernel.

### Route-Alpha Refinement

The split profile for the local-expert tensor-core candidate showed that FC1
and FC2 were already near 0.09-0.10 ms each at c16/c32, while route-alpha
generation and launch overhead were the remaining avoidable cost. A separate
prototype, `benchmarks/python/warpdecode_local_dense_tensorcore_store_alpha_candidate.py`,
replaced `gen_fc2_alpha_fused` with a tiny route-alpha store kernel that
matches scatter semantics for the target unique top-k route contract.

| Candidate | c1 ms | c4 ms | c8 ms | c16 ms | c32 ms | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| DenseGEMM helper alpha | - | - | - | 0.308 | 0.313 | Baseline target-shape bridge |
| Triton atomic alpha probe | 0.210 | 0.219 | 0.221 | 0.222 | 0.221 | Useful probe, duplicate-route semantics differ |
| Store-alpha probe, real expert alpha | 0.208 | 0.245 | 0.219 | 0.223 | 0.218 | Best measured c32, target unique-route only |

The store-alpha probe uses nontrivial per-expert alpha and is exactly equivalent
to `gen_fc2_alpha_fused` for balanced, unique-single-window, and
random-without-replacement route patterns across c1/c4/c8/c16/c32. It is not
generic-safe for duplicate selected experts; a production version must either
guard on the target unique-route top-k contract or fall back to the existing
helper. The preferred production implementation is not to ship this Triton probe
as final, but to fuse or rewrite the same route-alpha operation as a CuTe/CZS
target-shape primitive, then remeasure the full DenseGEMM decode path and
serving matrix.

An ordered-store Triton variant was also tested to see whether duplicate-route
scatter semantics could be preserved without returning to the helper. It failed
the duplicate-route equivalence check, so it is rejected. Do not treat same-CTA
same-address Triton stores as ordered scatter semantics for this path.

The current production scaffold was tightened accordingly:
`warp_decode_nvfp4_local_dense_moe` remains experimental and target-shaped,
checks `hidden=7168`, `intermediate=2048`, `local_experts=16`, `top_k=8`, and
`scale_vec=16`, and its route-alpha kernel uses plain stores rather than atomics
under the target unique-route contract.


## Long-Context Serving Requirements

WarpDecode is measured inside the full 1k through 128k context serving matrix,
not only as an isolated MoE microbenchmark. Longer context does not change the
per-token MoE arithmetic, but it changes scheduling pressure: attention/KV
work, HISA/Indexer metadata, CUDA graph replay, and decode microbatch packing
compete for the same B200 node. A promoted WarpDecode candidate therefore must
report both microbenchmarks and at least one c32 serving smoke at a long-context
cell before becoming the production default. The serving check must verify that
WarpDecode remains selected, that CUDA graph bucket reuse is active, and that
no padded grouped-MoE fallback appears in logs.

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

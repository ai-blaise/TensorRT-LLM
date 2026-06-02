# WarpDecode Route-Compatible Tensor-Core Proof

This proof covers the requested direct `token_selected_slots` WarpDecode
surface for the target `hidden=7168`, `intermediate=2048`, `experts=128`,
`top_k=8`, `scale_vec=16`, `c1/c2/c4/c8/c16/c32` decode buckets. It is
intentionally separate from production files.

Any kernel candidate that follows from this proof must be either exact
target-shape code for DeepSeek V3.2 NVFP4 MoE, selected behind explicit runtime
guards and fallback, or truly arbitrary-model safe with shape-derived strides,
dimensions, packing, and scale layout plus tests. Shape-specialized kernels
must not pretend to be generic.

## Inputs Reviewed

- Cursor WarpDecode blog: output-owned decode flips MoE parallelism away from
  experts, removes padding/scatter/combine, and uses independent warps that own
  one output value.
- NVIDIA CuTe DSL documentation and CuTe layout-algebra paper: CuTe/CuTeDSL
  makes the layout contract explicit; a tensor-core candidate is only legal when
  its operand layouts and shared tile semantics are preserved.
- Veitner CuTe materials: grouped block-scaled GEMM, scale tensor
  construction, warp specialization, and NVFP4 GEMV posts reinforce the same
  split: use tensor cores for shared-tile GEMM, use output-owned warp/GEMV
  structure when routes destroy shared-B reuse.
- CZS: `docs/proofs/warpdecode_nvfp4_b200_target_inner_slice_czs_module.json`
  was rerun in the B200 buildtools container and proved 20/20 obligations for
  the inner NVFP4 tensor-core slice.

## CZS Result

The CZS-proved inner slice is legal:

- 8/8 layout obligations proved.
- 4/4 SM100 MXF4/NVF4 MMA operand obligations proved.
- 8/8 vectorization obligations proved.

This proves that the math tile is legal when its operand contract is satisfied.
It does not prove that arbitrary dynamic `token_selected_slots` can feed that
tile without changing the route ABI.

## Shared-B Constraint

An SM100 NVFP4 MMA tile has one shared B operand tile for all M rows in the CTA.
For WarpDecode FC1, B is the selected expert's gate/up weight tile. For FC2, B
is the selected expert's down weight tile.

Therefore, rows can share a tensor-core CTA only when they have the same:

- selected expert slot,
- N tile,
- K tile and scale-vector layout.

Arbitrary post-dispatch `token_selected_slots[token, routed]` violates that
condition. Combining rows from different experts in one tensor-core CTA would
multiply different A rows by the wrong shared B tile. Preserving correctness
requires one of these transformations:

- group rows by expert,
- materialize compact selected-expert B tensors,
- pad/expert-major rows to tile shapes,
- or abandon shared-B tensor-core batching for the arbitrary-route part.

The first three are precisely the stages WarpDecode is meant to remove.

## Measured And Modeled Evidence

Already rejected:

- Compact selected-expert tensor-core bridge: c32 was 2.060 ms; c32 spent
  1.012 ms in weight gather alone.
- Route-grouped CuTeDSL bridge: best `slot8` pattern was about 1.974 ms for
  FC1 only at c32; `worst128` was about 27.837 ms for FC1 only.
- Direct scalar/two-kernel path: c32 was 3.055 ms total, with gate/up about
  1.679 ms and down about 1.370 ms. It is route-compatible but too slow.
- Native TRTLLM reference: c32 about 0.2826 ms.

The accompanying feasibility script models a one-launch route-aware tensor-core
kernel that preserves the direct route ABI but assigns CTAs by `(expert, N
tile)`. It avoids Python per-expert launches and avoids compact B materializing,
but it still pays the shared-B constraint.

For `c32`, there are only `32 * 8 = 256` routed rows. Under `worst128`, all 128
experts are touched, so each touched expert has about two rows. A legal
`M=128` tensor-core tile has about `2 / 128 = 1.6%` M utilization before
counting FC1/FC2 N tiles. That is not a viable production replacement for
native TRTLLMGen.

## Recommendation

Reject a generic direct shared-B tensor-core WarpDecode kernel for arbitrary
routes. It cannot preserve the Cursor direct route ABI, avoid compact/padded
expert-major staging, and keep tensor-core utilization high at c16/c32.

The production path should keep the direct Cursor ABI and move aggressively
toward the blog's output-owned implementation:

- retain the current c1-c32 direct route surface,
- enforce the target-shape guard/fallback contract for DeepSeek V3.2 kernels,
- keep tensor-core kernels only for high-reuse route-specialized crossover
  cases where CZS can prove the shared-B tile contract,
- continue optimizing the output-owned PTX/CuTe GEMV-like gate/up and down
  kernels with IKP,
- remove remaining global scratch traffic where feasible,
- overlap route metadata preparation with attention/KV metadata preparation,
- keep graph-stable route, scratch, and output buffers for c1/c2/c4/c8/c16/c32.

This is a rejection of the generic shared-B tensor-core branch, not a rejection
of CuTe/CZS. CuTe/CZS remains the promotion gate for route-specialized tensor
core subpaths and for the final vectorized output-owned kernels.

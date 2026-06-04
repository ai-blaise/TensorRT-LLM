# Lever-2: fused-affine global-index reuse on FSSS S-layers

## What

`sparse_attn_predict` converts each layer's local TopK indices to global paged
KV indices via `convert_req_index_to_global` (a per-row block_table gather +
affine). The floor fix (lever-1) skips the TopK *compute* on reuse ("S")
layers but the local->global *remap* still ran on all 43 S-layers.

On an S-layer the local TopK equals the owning ("F") layer's, and the global
formula is
    out = base * strideFactor + (tok % blockSize) + layerId * blockSize   (valid)
    out = -1                                                              (invalid)
so the only layer-dependent term is `layerId * blockSize`. Hence the S-layer
globals equal the cached F-layer globals plus a constant
    delta = (layerId_S - layerId_F) * blockSize
with -1 preserved. A single fused elementwise op replaces the gather.

## Wiring

- C++ op `trtllm::indexer_affine_reuse` (production path):
    cpp/tensorrt_llm/kernels/indexerAffineReuse.{h,cu}
    cpp/tensorrt_llm/thop/indexerAffineReuseOp.cpp
    + thop/CMakeLists.txt source-list entry
    + register_fake in tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py
- JIT fallback in dsa.py (`_ensure_indexer_affine_reuse_op`): registers an
  equivalent load_inline kernel under the same qualified name when the loaded
  .so predates the AOT op, so the path runs without a rebuild.
- `transform_local_topk_reuse_or_compute(topk, md, layer_idx, skip_topk,
  is_generation)`: F-layer (skip_topk False) runs the full remap and caches
  (global, layer_idx, is_generation) on metadata; S-layer (skip_topk True)
  affine-reuses with the per-pair delta. skip_topk is static per layer so the
  branch is constant under CUDA-graph capture; the cache is phase-guarded and
  shape-guarded, and the F-owner (layer 0 always F at prod) runs first each
  step and overwrites it before any S-layer reads.
- `sparse_attn_predict` calls the reuse-aware transform with
  `self.indexer.skip_topk`.

## Measurement (a4-us-001 B200/SM100, megamoe_dev)

Kernel A/B (idx_l2_affine_kernel.py, prod shape M=8 topk=1024 block=64,
against the real torch.ops.trtllm.convert_req_index_to_global, graphed):
    full remap        4.044 us
    fused affine      2.268 us   (1.78x; correct=True vs full remap)
    torch.where       6.149 us   (naive; slower than full -> needs a kernel)

Wired dispatcher A/B (bench_lever2_wiring.py, graphed):
    F full remap          4.106 us
    S affine reuse        2.450 us   (1.68x)
    save across 43 S      0.0712 ms/token
  Correctness: F_path matches full remap = True; S reuse matches full remap at
  the S layer_idx = True (cached_F_layer=4); phase fallback ok = True.

## Intel-correctness

  test_indexer_decode_custom_vs_fallback + test_indexer_topk_multi_request:
  14 passed, 6 skipped, 0 failed on the combined reconcile+lever-2 runnable
  dsa. No regressions vs the reconcile-only build.

This stacks on the reconcile (-48.1% per-token indexer cost): lever-2 removes a
further ~0.071 ms/token of dead local->global remap across the 43 reuse layers.

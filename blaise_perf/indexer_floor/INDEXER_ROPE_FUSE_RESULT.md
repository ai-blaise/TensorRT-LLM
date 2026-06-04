# Indexer proj: fused RoPE + cat + FP4-quant kernel (new decode lever)

New candidate from the vLLM GB200 "RoPE+Quant+Q-Write decode" fusion, applied to
the DSA indexer `pre_indexer_proj` Q/K projection path. Folds the standalone
flashinfer RoPE launch (and the BF16 q_pe/k_pe write-back + reload it forces)
into the existing `fused_cat_fp4` quantize kernel, halving the proj-quant cost of
every recompute-F indexer layer at decode.

## What it fuses

Production `pre_indexer_proj` (use_fp4) ran, per F-layer:
  1. `rotary_emb([q_pe, k_pe])`  — one flashinfer apply_rope_inplace kernel; reads
     q_pe (N*64 rows x rope_dim) + k_pe (N rows), writes them back rotated (BF16).
  2. `maybe_execute_in_parallel(_prep_q_or_k(q), _prep_q_or_k(k))` — two
     `fused_cat_fp4` kernels (cat pe+nope, per-block-32 FP4 E2M1 quant) on two
     streams, re-reading the rotated BF16 pe.

The fusion replaces this with two `fused_rope_cat_fp4` kernels (q ∥ k), each
doing neox-RoPE(pe) in-register + cat(pe,nope) + FP4 quant in one pass — no
standalone RoPE launch, no q_pe/k_pe round-trip.

## Graphed perf (B200, prod decode shape: n_heads=64, head_dim=128, rope_dim=64)

  N (tokens)   PROD rope+(cat_q||cat_k)   FUSED (fq||fk)   saved
     1               8.197 us                 4.113 us     +4.08 us
     8 (prod)        8.196 us                 4.11-4.58 us +3.2 .. +3.6 us
    32               7.343 us                 4.127 us     +3.22 us

FUSED min is ~4.11 us = the cat-alone launch floor; the RoPE math is register-
cheap and the two fused kernels overlap, so the whole RoPE cost (~4 us serial)
is removed. ~3.2-4.1 us saved per recompute-F indexer layer, graphed.

The first JIT prototype was net-NEGATIVE (6.15 us > ref) because of redundant
per-operand BF16 round-trips; the shipped kernel rounds the rotated value to
BF16 exactly once (mirroring prod) and is at the launch floor.

## Correctness — bit-identical FP4

The rotated value is rounded to BF16 once (prod materializes BF16 q_pe between
RoPE and fused_cat_fp4), then quantized with the byte-identical FP4/UE8M0 code
from fusedCatFp4.cu. Verified:

  - rope-only sub-kernel vs flashinfer apply_rope_inplace: max_abs = 0.0 (N=4..8).
  - full fused vs flashinfer-rope -> fused_cat_fp4: FP4 codes + UE8M0 scales
    BIT-IDENTICAL (0/512 .. 0/131072 byte diffs) at N=1, 8, 32, for both Q and K.
  - wiring glue (pre_indexer_proj fused branch): Q via per-head position
    broadcast (repeat_interleave), K standalone — bit-identical at N=1/8/32.
  - CUDA graph capture+replay: graph result == eager for Q and K; replay after a
    position-vector change still equals fresh eager (decode-replay safe).

### The load-bearing bug that was fixed
The neox pair shuffle ran inside the `if (from_pe)` branch (only pe lanes
[0,16)); a full-warp `__shfl_sync(0xFFFFFFFF, ...)` with the nope lanes absent
returned undefined values and corrupted the per-block amax (scales went to
garbage like 2^125). Fix: mask exactly the converged pe lanes
`(1u << (pe_dim/4)) - 1u` (= 0xFFFF). With the partial mask the kernel is
bit-exact.

## Build

  - cpp/tensorrt_llm/kernels/fusedRopeCatFp4.{cu,h}: nvcc -arch=sm_100a, EXIT 0,
    zero warnings (auto-globbed into the kernels lib).
  - cpp/tensorrt_llm/thop/fusedRopeCatFp4Op.cpp: compiles clean against torch +
    TRT headers; same TORCH_LIBRARY_FRAGMENT/IMPL + CHECK_TH_CUDA + empty_cuda
    pattern as the sibling fusedCatFp4Op.cpp; listed in thop/CMakeLists.txt.
  - register_fake("trtllm::fused_rope_cat_fp4") added in cpp_custom_ops.py.
  - JIT load_inline fallback (_ensure_fused_rope_cat_fp4_op) under the same
    qualified name (lever-2 pattern), so the wired path runs correctly on a .so
    that predates the AOT op. The fallback kernel is byte-for-byte the shipped
    .cu body and was used for all the correctness/perf/graph numbers above.

## Wiring (pre_indexer_proj)

Gated on `self._rope_cat_fuse_ok` (use_fp4 ∧ head_dim==128 ∧ rope_dim==64 ∧
rope_half %4==0 ∧ neox ∧ flashinfer cos/sin cache present) — a static per-layer
predicate, constant under graph capture. The non-fp4 / non-neox / non-128 paths
keep the original rotary_emb -> _prep_q_or_k path unchanged. cos/sin cache is the
same `rotary_emb.rotary_cos_sin.view(max_pos, -1)` flashinfer uses (cos first
half, sin second half), cached as FP32 on first use.

## Stacking

This lands on top of the R4 indexer-consolidation stack (op-trt-idx-consol-r4:
HISA-preamble skip + width topk gate + lever-2 affine reuse) and is orthogonal:
R4 cut logits+topk+remap; this cuts the proj-quant. Applies to all N_F=15
recompute-F layers at prod (reuse-S layers skip pre_indexer_proj entirely).

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

## Combined aggregate with the R4 stack

R4 profiled `sparse_attn_indexer` internals (logits + topk + kcache + remap) and
cut A_F 36.92 -> 16.43 us (-43.5% indexer-TPOT). The proj/RoPE/cat lives in
`pre_indexer_proj`, an UPSTREAM step R4 did not profile. This fusion cuts that
proj-quant ~8.2 -> ~4.1 us per F-layer, i.e. ~+4 us/F-layer that stacks on R4.

  per recompute-F layer (graphed, prod shape, B=8):
    R4 sparse_attn_indexer A_F                 16.43 us
    pre_indexer_proj rope+quant  (MAIN)         8.2  us   -> 4.1 us (this fusion)
  Across N_F=15 recompute-F layers: ~ -4 us * 15 = ~ -60 us/token added on top
  of R4's -256.7 us/token. Reuse-S layers skip pre_indexer_proj entirely
  (skip_topk early-return), so the proj saving applies only to the 15 F-layers,
  exactly where it is computed.

## Re-challenge of the R4 candidate dispositions (this session)

Per directive ("do not stop until each is at its proven floor"), re-examined:

  [1] cute_dsl logits retile/fuse: CONFIRMED FLOOR. The DSL paged-MQA-logits
      tile is fixed by the kernel warpgroup structure (SPLIT_KV=256 = compute
      tile 128 x 2 math warp groups -> block_kv=64 forced; see dsa.py:174-182).
      Not a free knob. Fusing logits+topk needs a streaming-topk rewrite inside
      the DeepGEMM CuTe kernel for ~1-2 us launch/BW (528 KB fp32 logits) -- high
      risk, low yield vs the proj win just landed. Logits stays at 6.17 us.
  [2] multi-stream overlap indexer<->MLA/MoE: CONFIRMED INFEASIBLE across ops
      (topk_indices feeds the same layer forward_dsa_attn; strict serial chain).
      But WITHIN the indexer, q||k multi-stream is real and is exactly what this
      fusion exploits (fused_q || fused_k on aux_stream).
  [3] in-graph metadata: unchanged (near-zero; hidden by overlap scheduler).
  [4][6] lever-2 affine reuse + IndexCache reuse: unchanged (maximal).
  [5] HISA activation: unchanged (structural net-negative at prod prefix 4608).

  NEW lever found this session: the proj-path RoPE+cat+quant fusion (above),
  which the R4 floor analysis did not cover. ~3.2-4.1 us/F-layer, bit-exact.

## Resume session (2026-06-04 PM): in-tree build fix + re-grounded correctness

### Build break caught + fixed (lever-2 affine kernel namespace)
The prior bundling pass verified only standalone-nvcc + the JIT load_inline
fallback, which do not include config.h. A real in-tree object compile against
the op-trt source (/repo in megamoe_dev: torch 2.11 / CUDA 13.1 / TRT headers,
sm_100a) exposed that indexerAffineReuse.{h,cu} hardcoded
"namespace tensorrt_llm { namespace kernels" while the build wraps kernel
symbols in the ABI inline namespace (TRTLLM_NAMESPACE_BEGIN/END ->
tensorrt_llm::_v1::kernels). The thop op (includes opUtils.h) therefore
referenced tensorrt_llm::_v1::kernels::invokeIndexerAffineReuse while the .cu
defined tensorrt_llm::kernels::invokeIndexerAffineReuse:
  - compile: "reference to 'kernels' is ambiguous" in indexerAffineReuseOp.cpp
  - link:    undefined reference to tensorrt_llm::_v1::kernels::invoke... in th_common
Fix: switch both affine files to the sibling convertReqIndexToGlobal.{h,cu}
pattern (config.h + cudaUtils.h includes, TRTLLM_NAMESPACE_BEGIN/END). The
fusedRopeCatFp4.{h,cu} files were already correct (used the macro).

Verified in-tree (EXIT 0) for all 7 C++ TUs of the rope-fuse stack:
  nvcc sm_100a: indexerAffineReuse.cu, fusedRopeCatFp4.cu
  g++ (torch+TRT): indexerAffineReuseOp.cpp, fusedRopeCatFp4Op.cpp,
                   + sibling convertReqIndexToGlobalOp.cpp (control, clean)
  nm: .cu exports T tensorrt_llm::_v1::kernels::invokeIndexerAffineReuse,
      matching the op TU's U reference (links).
Commit 3b704d6f (rope-fuse) / cherry-picked 72ee47b7 (consol-r4). Kernel logic
unchanged; namespace/linkage only.

### Re-grounded rope-fuse correctness (this session, GPU0, JIT fallback path)
Reference now built from the PRODUCTION RotaryEmbedding module (is_neox,
rope_dim=64) + a standalone kernel-exact cat_fp4 (byte-identical qFp4/UE8M0):

  pos=0 (RoPE == identity) -> fused == standalone cat_fp4([pe||nope]):
    N=1/8/32: packed 0/64, 0/512, 0/2048  scale 0/1, 0/8, 0/32  -> BIT-IDENTICAL.
    Proves the cat + FP4/UE8M0 quant path is exact.

  pos!=0 -> fused == (real rotary_emb rotation -> kernel-exact cat_fp4):
    N=1: 0/64 ; N=8: 1/512 ; N=32: 7/2048 packed nibble diffs; scales 0 diffs.
    The <=0.34% nibble diffs are FP4 bucket-boundary flips from a sub-ULP bf16
    rounding difference between the fused in-register RoPE (round-to-bf16 once)
    and rotary_emb's rotation; they vanish when both paths share one bf16
    intermediate (the prior-session bit-identical-vs-fused_cat_fp4 result).

  CUDA graph capture+replay: result == eager; replay after an in-place pos
    change == fresh eager. Decode-replay safe (re-confirmed this session).

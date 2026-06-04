# Indexer R4: graphed-floor analysis + two new decode-topk wins

Round-4 re-grounds the indexer optimization on the CUDA-graph-captured regime
that production decode actually runs in (cuda_graph_config buckets by batch only;
logits width = kv_cache max_seq_len = 132096), and finds the prior "indexer is
huge" figures were EAGER-dispatch-dominated. Two new graph-safe wins land on top
of the already-merged R2 stack (gate+syncfree+floor, origin/op-trt) and the
ported lever-2 affine reuse.

## The reframing: A_F is 77% eager dispatch overhead

idx_mb_r2 reports an EAGER per-token indexer cost (~3000us). But production runs
the indexer under graph capture. Graphing the F-layer sparse_attn_indexer at prod
decode shape (B=8 next_n=1 prefix=4608 index_topk=1024):

  A_F eager   = 158.97 us
  A_F graphed =  36.92 us   (-122 us, -76.8% pure CUDA launch/dispatch overhead)

flat across batch (B=1/8/32 all ~36-37us graphed). Graphed sub-kernels:
  logits 6.17us + topk 14.3us (C++) + kcache 4.12us + ~12us dead-preamble residue.
So the real prod indexer cost is set by these kernels, not the eager dispatch.

## Win A: skip the dead HISA-from-logits preamble (commit bc9a463c)

The decode topk path always built row_indices/next_n_offset/row_starts/row_ends
and called _hisa_topk_from_logits, which is statically gated by
_should_use_hisa_logits == False (HISA at decode runs from the NVFP4 cache, never
from dense logits). Pure dead work captured into the graph. Gated on the same
cheap capture-safe predicate (column count, no .item()).

  graphed A_F: 36.92 -> 22.58 us  (-14.34 us, -38.8%) at prod width 132096
  top-k SET bit-identical (sorted-per-row equal, 0/8 rows differ).

## Win B: gate decode Top-K on logits WIDTH, not kv_len (commit 71793579)

The gate chose the DSL topk only at kv_len >= 32768; else C++. But the cost driver
is the logits scan WIDTH (= max_seq_len = 132096 in prod, regardless of the short
~4.6K live prefix). The C++ kernel takes its fast insertion path only for
numColumns < 12288 and is ~2x slower above it; DSL is flat. Isolated topk sweep
(B=8 topk=1024 graphed):

  width    C++ us   DSL us   winner   set-equal
   4608     8.23     9.22     C++       True
   8192     8.23    10.27     C++       True
  12288    16.41    10.27     DSL       True   <- C++ leaves insertion path
  16384    16.42    10.27     DSL       True
 132096    16.43    12.32     DSL       True   <- production width

Add a width OR-term (_DSL_TOPK_MIN_COLS=12288). Prod (width 132096) now takes DSL.

  graphed A_F: 22.58 -> 16.43 us  (-6.15 us, -27.2%)
  top-k SET bit-identical at 16384 and 132096; max idx respects live kv bound.

(Tight-width logits -- pass live max_gen_kv_len instead of cache max_seq_len -->
topk width 4864 -> C++ insertion path 8.22us -- was probed and is correct, but is
GRAPH-INCOMPATIBLE: a captured decode graph is replayed as kv grows across the
full context, so the logits width must stay = max_seq_len. Rejected.)

## Aggregate (prod shape: 64 heads, B=8, width=132096, FSSS N_F=15/N_S=43, graphed)

                                A_F     A_S    AGG us/token
  MAIN (op-trt + lever2)       36.92   ~1.0      590.2
  + Win A + Win B              16.43   ~2.0      333.5
  = -256.7 us/token  (-43.5%) graphed indexer-TPOT, from the two new wins alone.

A_F per recompute-F layer: 36.92 -> 16.43 us (-55.5%). The two wins transfer
cleanly to 64-head prod (A_F 36.97 -> 17.07 at 64h) and stack on the merged R2
stack and ported lever-2.

## Candidate dispositions (this round)

  (1) cute_dsl_fp4_paged_mqa_logits: PROVEN FLOOR, graphed 6.17us, invariant to
      tiling/occupancy (re-confirmed). The adjacent topk, not the logits, was the
      real lever -> Win B.
  (2) multi-stream overlap: INFEASIBLE. topk output is consumed by the same
      layer's MLA attention (forward_dsa_attn); the indexer->attn->MoE->next-layer
      chain is strictly sequential. No independent work to overlap with at decode.
  (3) in-graph metadata: get_paged_mqa_logits_metadata is ~7.5us host, capturable,
      but runs in on_update_kv_lens (pre-graph) where the default overlap scheduler
      hides it behind the prior step's GPU exec. Near-zero net; not pursued.
  (4) affine remap C++ op + S-skip: DONE (lever-2 ported; F 4.107us -> S 2.457us,
      -0.071 ms/token across 43 S-layers; JIT-fallback verified correct).
  (5) HISA activation at prod prefix: NOT lowered. At prefix 4608 = 36 blocks the
      mean-pool-reps + block-score + 2-stage topk overhead dominates the direct
      topk; prod hisa_min_seq_len=65536 is deliberate. FP4-deepgemm-path
      unmeasurable in megamoe_dev (symbols absent); structural net-negative.
  (6) IndexCache reuse: MAXIMAL. 43/58 layers fully short-circuit (A_S ~2us);
      lever-2 removes the residual global remap. Nothing left.

## Correctness

  test_dsa_indexer.py full suite on the final consol dsa: 47 passed / 9 failed /
  6 skipped. The 9 failures are the pre-existing paged_kv_cache(_fp4) deepgemm/fp4
  set (stale container deep_gemm, no FP4 paged-MQA symbols + next_n>1 MTP),
  byte-identical to the origin/op-trt baseline -- zero new regressions.

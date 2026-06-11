# WarpDecode (MoE decode fast path) — campaign summary

> The **canonical, long-form WarpDecode reference** (`docs/source/features/warpdecode.md`
> + `docs/source/features/warpdecode_deployment_guide.md`) lands with the
> WarpDecode kernel branch, not with this docs branch, so the full kernel
> design / tactic tables / rejected-bridge log live there once that branch
> merges to `op-trt-ls`. This page is the self-contained campaign-context entry
> (figures + file map + composition) so the decode picture is complete in one
> place even before that merge.

WarpDecode is an **opt-in decode-only MoE fast path** for the Blaise
DeepSeek-V3.2 NVFP4 target shape (`hidden=7168`, `intermediate=2048`,
`experts=128`, `topk=8`). It is selected after routing has materialized
`topk_ids` / `topk_weights` (post-EPLB, post-dispatch) and falls back to the
native MoE backend when its runtime guards do not match.

## What landed this campaign

- **Retuned NVFP4 target tactics** (`_NVFP4_TARGET_TACTICS`): the explicit-tactic
  TRTLLMGen bridge, bucketed through c32, with the accepted tactic map
  `{1:[8,26], 2:[8,75], 4:[8,53], 8:[8,53], 16:[8,53], 32:[16,52]}`. Measured
  **1.20–1.36× vs native TRTLLM MoE** across c1–c32 (e.g. c1 0.0873 → 0.0641 ms
  1.36×; c32 0.0860 → 0.0718 ms 1.20×). `TRTLLM_ENABLE_PDL=1` by default inside
  the op (PDL-on graph replay beats PDL-off by 1.05–1.12× in every tested
  pattern/bucket).
- **NVFP4 FC1→SwiGLU→FC2 megakernel** build + bench (CuTeDSL tensor-core,
  2-CTA `cta_group::2` → FC1 1.27× / 24.3 µs); CZS index-contract proofs per
  config round. **Correctness fix `e105fd7a1`:** the megakernel JIT default
  `TRTLLM_OPTRT_MOE_MEGAKERNEL_FC2_N` was 160, which a TRUE-f32 reference
  shows is **numerically broken** (SFB weight-scale-factor miscompute —
  cosine 0.790 vs f32 at the REAP shape, vs 0.99984 at N=256); default is now
  **256**, an explicit 160 raises, and `validate_fused_moe_megakernel.py` was
  de-blinded to compare against a true f32 einsum (it previously compared the
  fused kernel against a sequential run of the SAME FC2 kernel —
  fusion-equivalence only, blind to the SFB bug). The fusion itself is
  timing-neutral under PDL; the megakernel's case is structural (persistent
  kernel), not the tile.
- **DeepEP low-latency unblocked under WarpDecode (`51918fba2`):** the LL
  adapter's padded top-1/sentinel recv layout made the trtllm_gen overlay
  RAISE under policy=force — now detected and skip-reasoned (the canonical
  CuteDslFusedMoE backend consumes it natively; moe_sort drops sentinel rows
  without inflating tiles — tiles_equal at EP2, xlayout cosine 0.99999). Plus
  the ConfigurableMoE oversize **park/restore** fix: an oversize forward (the
  max_num_tokens warmup) used to destroy the comm strategy and pin the
  AllGather fallback for the process lifetime, silently evicting LL. Sizing
  (superseding the earlier "inverts at limit=64" caveat, which was an
  emulated-FFN measurement artifact — REFUTED by the 2026-06-11 sweep): the
  padding curve is **flat** and LL wins at every measured point (2.4–2.5×
  at steady c16, 2.06× at a full 64-token batch); the decision is an
  **explicit `TRTLLM_DEEP_EP_TOKEN_LIMIT=64`** (unset would size the
  NVSHMEM heap to engine max_num_tokens). LL requires ADP attention +
  `moe_tp_size=1` — inert under plain TP (see optimization_candidates.md
  M3, **DECIDED GO**).
- **Honest rejected paths recorded** (do-not-redo): generic CuTeDSL grouped-GEMM
  (~0.34–0.50 ms, rejected), small-tile grouped-MoE (tiles < 128 fail the guard;
  tile-128 ~0.32–0.49 ms, rejected), CUTLASS FP4 GEMV floor (already slower than
  the bridge), BF16 output-owned down-proj bridge (0.106–0.630 ms, rejected),
  dynamic route-locality selector (selector overhead erased the win),
  **FC2 N-tile 160 in any form** (SFB miscompute cos 0.790 + prefill M=1024
  OOB since 7168 % 160 ≠ 0; killed `29f492b49`→`d01737397` — a correct 160
  needs a CUTLASS-internal SFB layout tiled at 160, not an epilogue tweak).

## Files

- `tensorrt_llm/_torch/modules/fused_moe/warp_decode.py` — the NVFP4 op +
  tactic map + PDL default.
- `tensorrt_llm/_torch/modules/fused_moe/moe_scheduler.py` — the post-dispatch
  scheduler hook that feeds `token_selected_slots` / `token_final_scales`.
- `tensorrt_llm/_torch/models/modeling_deepseekv3.py`,
  `tensorrt_llm/_torch/model_config.py`, `tensorrt_llm/llmapi/__init__.py` —
  wiring + config surface.
- Driver/validator: `tests/scripts/cute_dsl_kernels/warpdecode_mega_driver.py`,
  `warpdecode_mega_validate.py`; unit test
  `tests/unittest/_torch/modules/moe/test_warp_decode.py`.

## Enable / default

Opt-in via the configurable WarpDecode MoE backend (env/config selector). The
NVFP4 target path **is** allowed in CUDA-graph decode buckets (c1/c2/c4/c8/c16/
c32) because it runs after scheduler dispatch/remap has produced local slot ids;
the BF16 compatibility path stays conservative (graph-excluded). `TRTLLM_ENABLE_PDL=1`
is the default inside the op (explicit env overrides respected).

## Correctness

Matches the explicit TRTLLMGen runner **exactly** under finite scales
(`max_abs=0`, cosine ≥ 0.99999988) across a c1–c32 × route-pattern
(`round_robin`, `slot8`, `paired`, `single_expert`) CUDA-graph capture/replay
sweep. Validated by **numerical match vs the native MoE reference**, not e2e text.

## Composition

- **NVFP4 add+RMSNorm+quant fusion** (`nvfp4_fusions.md` #13): operates on the
  *pre-MoE* norm+quant; WarpDecode is the *expert GEMM*. Independent stages.
- **LayerSplit** (`../source/features/layersplit.md`): orthogonal — WarpDecode
  on the MoE dispatch/routing path, LayerSplit on the DSA KV cache; both compose
  with CP independently.
- **Indexer / Sparse-MLA / KVarN**: independent (attention/KV vs MoE).
- **Topology** (`topology_deploy.md`): the post-EPLB/post-dispatch hook is what
  keeps WarpDecode compatible with TP/EP and external communication.

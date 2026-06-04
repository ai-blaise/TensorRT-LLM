# WarpDecode (MoE decode fast path) — campaign summary

> The **canonical, long-form WarpDecode reference** is
> [`docs/source/features/warpdecode.md`](../source/features/warpdecode.md) and
> the deployment guide is
> [`docs/source/features/warpdecode_deployment_guide.md`](../source/features/warpdecode_deployment_guide.md).
> This page is the one-paragraph campaign-context entry so the decode picture is
> complete in one place; read the canonical doc for the full kernel design,
> tactic tables, and the rejected-bridge log.

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
  config round.
- **Honest rejected paths recorded** (do-not-redo): generic CuTeDSL grouped-GEMM
  (~0.34–0.50 ms, rejected), small-tile grouped-MoE (tiles < 128 fail the guard;
  tile-128 ~0.32–0.49 ms, rejected), CUTLASS FP4 GEMV floor (already slower than
  the bridge), BF16 output-owned down-proj bridge (0.106–0.630 ms, rejected),
  dynamic route-locality selector (selector overhead erased the win).

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

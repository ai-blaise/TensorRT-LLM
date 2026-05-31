# Blaise DeepSeek-V3.2

This path adapts TensorRT-LLM's native DeepSeek-V3.2 Blackwell DSA stack for
Blaise checkpoints that declare custom behavior in the Hugging Face
`config.json`. The implementation intentionally reuses TensorRT-LLM's existing
DSA Indexer, NVFP4 indexer K cache, CuTe DSL paged MQA logits, TopK,
memory-pool, and cache-transfer surfaces where they match the model contract.

## Model-card dispatch

When a DeepSeek config exposes `index_topk`, TensorRT-LLM treats it as a
DeepSeek-V3.2 DSA model even if `architectures` is still
`DeepseekV3ForCausalLM`. The loader promotes the runtime `model_type` to
`deepseek_v32` so the optimized DSA attention path is selected.

The following Blaise `quantization_config.indexer_quantization` fields are
recognized:

| Field | Runtime effect |
| --- | --- |
| `quant_method: "nvfp4_e2m1_ue8m0"` | Selects TensorRT-LLM's Blackwell NVFP4 E2M1/UE8M0 indexer K cache. |
| `hisa.enabled: true` | Selects the `indexcache-hisa` config contract. |
| `hisa.block_size` | Sets `hisa_block_size`. |
| `hisa.block_topk` | Sets `hisa_block_topk`. |
| `hisa.compression_ratio` | Sets `hisa_compression_ratio`. |
| `hisa.execution_mode` | Sets `hisa_execution_mode`. |
| `indexcache.freq` / `indexcache.pattern` | Sets the OP-compatible IndexCache TopK reuse policy fields. |
| `indexer_quantization.layersplit.enabled: true` | Enables the LayerSplit DSA KV/indexer config surface. |
| `indexer_quantization.layersplit.layout: "interleaved"` | Maps to the `round_robin` LayerSplit owner assignment. |
| `moe_runner_backend: "warp_decode"` | Enables the WarpDecode MoE overlay unless the operator supplied `moe_config.warp_decode`. |

`hisa.mode` must be `indexcache-hisa`. Standalone HISA is rejected because the
Blaise production path layers HISA on top of NVFP4 IndexCache.

`kv_cache_scheme.quant_method: "higgs_dense_2bit"` is deliberately ignored by
this TensorRT-LLM path. HIGGS is not part of the current integration.

## IndexCache and HISA

The adapter selects TensorRT-LLM's FP4-named indexer K cache only for NVFP4
indexer model cards. In this path `fp4` is the TensorRT-LLM internal name for
Blackwell NVFP4 E2M1/UE8M0, not a generic FP4 format. The adapter also enables
the CuTe DSL NVFP4 paged MQA logits and TopK surfaces by default when the model
card provides indexer overrides.

IndexCache follows the optimization-playground `F`/`S` policy:

| Pattern role | Runtime behavior |
| --- | --- |
| `F` | Compute the layer's DSA TopK normally and save it on the batch metadata. |
| `S` | Reuse the previous saved DSA TopK for the same batch metadata. |

When no explicit pattern is provided, `indexcache.freq` applies the same
frequency policy. The model loader validates explicit patterns against
`num_hidden_layers`. The current Blaise target config declares
`num_nextn_predict_layers=0`; deployments that add NextN layers should validate
their layer mapping before enabling IndexCache reuse.

The `indexcache-hisa` mode preserves the Blaise NVFP4 HISA contract and
validates that it uses the NVFP4 indexer K cache. Its current execution target is
TensorRT-LLM's native NVFP4 DSA path with IndexCache reuse and a HISA block
selector over the existing indexer logits. A future CuTe/CZS HISA selector can
replace this fallback without changing the model-card contract.

## LayerSplit

LayerSplit is represented as a DeepSeek DSA sparse-attention overlay:

```python
sparse_attention_config = {
    "algorithm": "dsa",
    "indexer_mode": "indexcache-hisa",
    "indexer_k_dtype": "fp4",
    "layersplit_enabled": True,
    "layersplit_owner_assignment": "round_robin",
    "layersplit_transfer_backend": "auto",
    "layersplit_all_cp_ranks_transfer": True,
}
```

The integration deliberately attaches LayerSplit to DSA metadata instead of to a
standalone cache-copy flag. That keeps the contract tied to the tensors that are
actually split: dense DSA KV and the indexer K cache. Runtime selection must
compose with the active TP, EP/MoE EP, attention DP, CP/DWDP, and
context/generation disaggregation mapping. Until partial-rank transfer is
implemented, `layersplit_all_cp_ranks_transfer` must remain true.

## WarpDecode

WarpDecode is exposed as an MoE overlay rather than a new MoE backend:

```python
moe_config = {
    "backend": "CUTEDSL",
    "warp_decode": {
        "enabled": True,
        "max_batch_size": 64,
        "policy": "auto",
        "allow_parallelism_fallback": True,
    },
}
```

`backend` remains TensorRT-LLM's optimized DeepSeek-V3.2 B200 MoE backend.
`warp_decode` only describes when a decode-only small-batch fast path may be
used. In `auto` mode, unsupported shapes, quantization modes, CUDA-graph
captures, TP/EP layouts, attention-DP groups, CP layouts, or disaggregated
generation mappings must fall back to the native backend. `policy: "force"`
is reserved for validation runs and turns fallback into an error.

## SMC-SD

SMC-SD is parsed as a PyTorch speculative decoding config:

```python
speculative_config = {
    "decoding_type": "SMC",
    "speculative_model": "BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP",
    "gamma": 6,
    "n_particles": 4,
    "resample_threshold": 0.5,
    "draft_attention_backend": "triton",
    "draft_kv_cache_dtype": "fp8_e4m3",
}
```

The config validates the Blaise SMC-SD contract and fails explicitly if runtime
metadata, resource-manager, drafter, sampler, or worker paths are reached before
their implementation is complete. The runtime implementation must preserve
separate draft/target KV ownership, CUDA-graph padding semantics, target verify
with `gamma + 1` tokens including the bonus token, particle resampling, and
decode-only use under disaggregated prefill/decode.

## Model semantics

`attention_output_gate: true` adds the Blaise G1 attention gate under each MLA
attention module as `self_attn.gate_proj`. The gate is applied as
`attention_output * sigmoid(gate_proj(hidden_states))` immediately before
`o_proj`, matching the optimization-playground and training placement.

`gated_norm: true` adds the two low-rank Blaise GatedNorm gates per decoder
layer:

| Module | Placement |
| --- | --- |
| `input_gated_norm_down/up` | After input RMSNorm and before self-attention. |
| `post_attention_gated_norm_down/up` | After post-attention RMSNorm and before MLP/MoE. |

Pre-MLP/MoE residual-norm fusion is disabled for gated-norm layers so the gate
is applied at the correct semantic point. Later fusion work can reintroduce a
fused gated RMSNorm path once parity is established.

## Current boundaries

Dense MLA KV cache support is separate from the NVFP4 IndexCache/HISA path.
TensorRT-LLM has general NVFP4 KV-cache plumbing, but the current DeepSeek MLA
generation path still rejects dense FP4/NVFP4 KV. Deployments may use FP8 dense
MLA KV as a bootstrapping fallback while NVFP4 IndexCache+HISA remains enabled,
but that is not the production target. If dense KV precision or absorbed MLA BMM
becomes the blocker, the expected fix is to add the missing op-trt NVFP4 MLA
BMM/KV kernels and dispatch support rather than permanently dequantizing helper
tensors or relying on FP8 dense KV.

The implementation adapts TensorRT-LLM's native NVFP4 DSA/indexer cache
machinery rather than wholesale-porting optimization-playground kernels. Gated
Attention and GatedNorm are unique Blaise model semantics and are ported
directly. LayerSplit, WarpDecode, and SMC-SD now have explicit config surfaces
and runtime guardrails, but their production fast paths still need the next
implementation phase and B200 validation before deployment.

# KVarN for SMC-SD GQA KV cache

This track is separate from dense MLA latent KVarN. Dense MLA uses
`sparse_attention_config.mla_latent_kv_dtype`; SMC-SD GQA uses the generic
`kv_cache_config.dtype` / Hugging Face `kv_cache_dtype` path.

## Source of truth

Huawei KVarN (arXiv:2606.03458, repo `huawei-csl/KVarN`) quantizes each
per-head KV cache tile after a channel-dimension Hadamard rotation and
Sinkhorn-style variance normalization across both token and channel axes. The
current public implementation fixes one KVarN tile to one paged KV block:
`group=128`, `head_dim=128`, with fp16 sink/tail storage for tokens that are not
yet in a full tile. The released GQA preset is `kvarn_k4v2_g128`; Blaise's
SMC-SD production target is the 2-bit variant `kvarn_k2v2_g128`.

For SMC-SD GQA shapes this maps as:

- tile: `[tokens_per_block=128, num_kv_heads_local, head_dim=128]` per layer;
- K orientation: normalize/pack per channel over the 128-token tile;
- V orientation: normalize/pack per token over the 128-channel head tile;
- packed store: 2-bit K and 2-bit V codes plus fp16 scale/zero-point vectors;
- sink/tail: first 128 tokens and the in-progress partial block stay fp16;
- decode: dequant full committed blocks, combine sink/tail in fp16, then score
  grouped-query heads with the usual GQA head grouping.

## Current op-trt status

The current op-trt tree has dense-MLA KVarN kernels and a `QuantAlgo.KVARN` /
`QuantMode.KVARN_KV_CACHE` flag, but the generic GQA KV path is not wired to any
KVarN storage/read backend yet. A GQA request such as
`kv_cache_config.dtype="kvarn_k2v2_g128"` is now accepted by config validation
only when `tokens_per_block=128`, and Hugging Face config artifacts can request
it via either:

```json
{
  "kv_cache_dtype": "kvarn_k2v2_g128"
}
```

or:

```json
{
  "quantization_config": {
    "kvarn": {
      "gqa": { "enabled": true, "dtype": "kvarn_k2v2_g128" }
    }
  }
}
```

Runtime validation intentionally raises `NotImplementedError` before allocation.
This prevents a hidden fallback to fp16/fp8/nvfp4 while the generic GQA KVarN
kernel/read path is absent.

## Missing implementation pieces

The minimal complete GQA patch must add all of the following before the runtime
fail-fast can be removed:

1. `tensorrt_llm/_torch/pyexecutor/_util.py`: map
   `QuantMode.has_kvarn_kv_cache()` to a KVarN allocation path instead of the
   current full-precision fallback.
2. `tensorrt_llm/_torch/pyexecutor/model_engine.py`: report KVarN packed bytes
   per token for scheduler/capacity accounting, including fp16 sink/tail pool
   overhead.
3. `tensorrt_llm/_torch/pyexecutor/resource_manager.py`: allocate packed KVarN
   records per `(block, layer, kv_head)` plus fp16 sink/tail side buffers and
   expose them to attention metadata and transfer metadata.
4. `tensorrt_llm/_torch/attention_backend/trtllm.py` and/or a dedicated KVarN
   backend: detect `has_kvarn_kv_cache`, drive full-block store/flush,
   maintain request block-to-tail mappings, and route decode through KVarN
   dequant/scoring instead of `torch.ops.trtllm.attention`'s fp8/nvfp4 path.
5. C++/CUDA or Triton kernels: implement KVarN GQA store, 2-bit pack/unpack,
   Hadamard rotation/unrotation, variance-normalization scale handling, and
   decode read/scoring for SMC-SD multi-token draft/verify shapes.
6. `tensorrt_llm/_torch/disaggregation/resource/kv_extractor.py` plus cache
   transceiver metadata: transfer packed records and fp16 sink/tail state without
   reinterpreting them as ordinary K/V tensors.
7. Tests/benches: compare BF16/FP8 logits or attention outputs against KVarN for
   GQA with odd SMC draft M, batch/concurrency 16, input lengths 1k through
   128k, LayerSplit/request-pinning/disagg transport enabled, and WarpDecode
   force/no-fallback active.

Until these pieces exist, `kvarn_k2v2_g128` for GQA is a recognized but blocked
configuration, not a runnable production path.

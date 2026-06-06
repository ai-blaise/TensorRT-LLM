# KVarN for SMC-SD GQA KV cache

This track is separate from dense MLA latent KVarN. Dense MLA uses
`sparse_attention_config.mla_latent_kv_dtype`; SMC-SD GQA uses the generic
`kv_cache_config.dtype` / Hugging Face `kv_cache_dtype` path.

## Source of truth

Huawei KVarN (arXiv:2606.03458, repo `huawei-csl/KVarN`) quantizes each
per-head KV cache tile after a channel-dimension Hadamard rotation and
Sinkhorn-style variance normalization across both token and channel axes. The
public GQA implementation fixes one KVarN tile to one paged KV block:
`group=128`, `head_dim=128`, with fp16 sink/tail storage for tokens that are not
yet in a full tile. The released GQA preset is `kvarn_k4v2_g128`; Blaise's
SMC-SD production target is the 2-bit variant `kvarn_k2v2_g128`.

For SMC-SD GQA shapes this maps as:

- tile: `[tokens_per_block=128, num_kv_heads_local, head_dim=128]` per layer;
- K orientation: rotate `K @ H`, then pack `[num_kv_heads, head_dim, 128]`;
- V orientation: rotate `V @ H`, then pack `[num_kv_heads, 128, head_dim]`;
- packed record: K codes + K fp16 absorbed scale/zp/token scale, followed by V
  codes + V fp16 channel scale/absorbed token scale/zp;
- sink/tail: first 128 tokens and the in-progress partial block remain fp16;
- decode: committed blocks are dequantized or read in-kernel, then scored with
  the usual GQA query-to-KV-head grouping.

For `kvarn_k2v2_g128`, one `(block, layer, kv_head)` record is 9,728 bytes:
4,096 K-code bytes + 768 K-scale bytes + 4,096 V-code bytes + 768 V-scale
bytes. Because `9,728 / 128 = 76`, op-trt can host the packed record as a
byte-backed self-only page with `head_dim=76` once the cache manager and
attention backend learn that this page is opaque KVarN data, not dense K/V.

## Current op-trt status

Implemented:

- `tensorrt_llm/_torch/attention_backend/kvarn_gqa.py` defines the GQA KVarN
  dtype parser, Huawei-compatible byte layout, 2/3/4-bit bitstream pack/unpack,
  Hadamard + variance-normalized store, dequant restore, a reference packed
  pool, and transfer-view metadata shape.
- `kv_cache_config.dtype="kvarn_k2v2_g128"` is accepted only with
  `tokens_per_block=128`.
- Hugging Face artifacts can request GQA KVarN explicitly through top-level
  `kv_cache_dtype`, or through `quantization_config.kvarn.gqa`.
- HF artifacts can declare production default support without a YAML override by
  setting `supports_kvarn_gqa=true`, `kvarn_gqa_supported=true`,
  `default_kvarn_gqa=true`, `smc_sd_gqa_kvarn=true`, or
  `quantization_config.kvarn.gqa.enabled=true` with no explicit dtype. The
  default dtype is `kvarn_k2v2_g128`.
- Runtime validation still fail-closes before allocation because the production
  generic paged K/V backend is not wired yet. This is intentional: there must be
  no hidden fallback to fp16/fp8/nvfp4 when KVarN was requested.

HF examples:

```json
{
  "kv_cache_dtype": "kvarn_k2v2_g128"
}
```

```json
{
  "quantization_config": {
    "kvarn": {
      "gqa": { "enabled": true, "dtype": "kvarn_k2v2_g128" }
    }
  }
}
```

```json
{
  "architectures": ["Glm4ForCausalLM"],
  "supports_kvarn_gqa": true
}
```

Set `kv_cache_dtype` to `"auto"`, `"none"`, or set
`quantization_config.kvarn.gqa.enabled=false` to disable the HF default.

## Missing production pieces

The runtime fail-close can be removed only after all of these exist:

1. `tensorrt_llm/_torch/pyexecutor/_util.py`: map
   `QuantMode.has_kvarn_kv_cache()` to a byte-backed KVarN allocation path.
   For k2v2/g128 the packed pool can use `CacheType.SELFKONLY`,
   `tokens_per_block=128`, and `head_dim=76` byte slots, plus fp16 sink/tail side
   pools.
2. `tensorrt_llm/_torch/pyexecutor/resource_manager.py`: expose packed KVarN
   records per `(block, layer, kv_head)`, request-owned fp16 sink/tail buffers,
   `valid`/`commit_gen` state, and correct byte/capacity accounting. This must
   work with request pinning and block reuse.
3. `tensorrt_llm/_torch/attention_backend/trtllm.py` or a dedicated KVarN GQA
   backend: store full 128-token blocks, keep sink/tail in fp16, and route decode
   through KVarN read/dequant instead of `torch.ops.trtllm.attention`'s fp8/nvfp4
   dense-KV path.
4. CUDA/Triton kernels: fused full-block store, packed 2-bit load, dequant/scaled
   dot-product/value accumulation for generation and SMC draft/verify query
   shapes, including odd M values such as 25. Python per-token restore loops are
   not acceptable for production throughput.
5. `tensorrt_llm/_torch/disaggregation/resource/kv_extractor.py` and
   `kv_cache_transceiver.py`: transfer packed records and fp16 sink/tail state as
   KVarN records, never as ordinary dense K/V tensors. UCX/NIXL/Mooncake choice
   stays transport-level; the payload contract is the same.
6. Scheduler/model-engine accounting: include packed bytes and fixed sink/tail
   pool bytes so max-token estimation, CUDA graph warmup, LayerSplit, SMC draft
   reservation, and request pinning see the real footprint.

## Tests and benchmark harness

Current focused coverage:

- `tests/unittest/llmapi/test_kvarn_gqa_config.py`: dtype validation, HF explicit
  request, HF default-on declaration, explicit disable, and startup fail-close.
- `tests/unittest/_torch/attention/test_kvarn_gqa.py`: k2v2 record layout,
  2/3/4-bit pack/unpack, store/restore shape/finiteness/cosine, packed-pool
  commit state, and transfer-view shape.
- `benchmarks/python/bench_kvarn_gqa_micro.py`: light pack/restore/reference
  scoring timing for `M in {1,5,25}`. Use it only on an idle GPU or CPU; it is a
  reference baseline, not the fused production-kernel benchmark.

Example:

```bash
python benchmarks/python/bench_kvarn_gqa_micro.py --device cuda --kv-heads 8 --iters 100 --queries 1 5 25
```

## Composition contract

- Dense MLA KVarN remains controlled by `mla_latent_kv_dtype`; GQA KVarN is the
  generic KV-cache dtype path.
- KVarN never replaces the Indexer K path. HISA/indexcache/fp4 indexer state
  remains separate.
- LayerSplit can split ownership of KVarN pages exactly like other dense KV
  pages, but non-owner scratch/broadcast paths must preserve opaque record bytes.
- SMC-SD draft/verify may produce odd query counts; KVarN pack granularity is
  the 128-token K/V tile, so odd M affects decode scoring only, not storage.
- WarpDecode is independent of KV storage; forcing WarpDecode must not enable a
  KV fallback.
- Disaggregated transfer must move packed records plus fp16 sink/tail state and
  must fail if the selected backend only knows dense K/V tensors.

Until the production backend pieces above are implemented, `kvarn_k2v2_g128` is
HF-deployable and defaultable as configuration, and has tested byte-layout
primitives, but remains intentionally not runnable end-to-end.

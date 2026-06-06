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

## Parent integration note

This handoff is intentionally split into two merge lanes:

- **A: safe scaffolding** can merge now. It adds the Huawei-compatible k2v2/g128
  byte layout, 2/3/4-bit pack/unpack tests, HF/config parsing, fail-closed
  validation, docs, and the microbench harness. It does not allocate generic GQA
  KVarN pages or route model execution through the reference backend.
- **B: reference backend** is not production-promoted. It adds byte-backed page
  allocation, Python-level store/restore, fp16 sink/tail side state, and SDPA
  scoring for isolated correctness experiments only. Startup still fails closed
  unless `TRTLLM_ENABLE_KVARN_GQA_REFERENCE=1` is set. Do not enable this in the
  op-trt deployment until the fused B200 store/decode kernel, packed-record
  disaggregated transfer with fp16 sink/tail side-state, sparse-indexed packed
  reads, CUDA graph request lifecycle, and E2E perf proof are complete.

Dense MLA KVarN remains separate and production-owned by
`sparse_attention_config.mla_latent_kv_dtype`; GQA KVarN never replaces the
Indexer/HISA sparse K path.

## Downtime audit checklist

| Requirement | Status | Notes |
|---|---|---|
| Packed 2-bit record format | **Implemented as primitives** | `kvarn_k2v2_g128` maps one 128-token block/head to a 9,728-byte record, hosted as 76 byte slots per token. Layout, bit pack/unpack, Hadamard/variance-normalized store, dequant restore, and transfer-view shapes are tested. |
| Dense/GQA separation from Indexer | **Implemented in config/docs; reference backend enforces separation** | Dense MLA KVarN uses `mla_latent_kv_dtype`; GQA uses `kv_cache_dtype`. Indexer/HISA sparse K remains separate and is not quantized. Sparse GQA KVarN read attempts fail closed. |
| HF deployability/default | **Implemented, fail-closed by default** | HF can request/default `kvarn_k2v2_g128` via top-level `kv_cache_dtype` or `quantization_config.kvarn.gqa`; startup rejects GQA KVarN unless the reference env opt-in is set or a future production backend removes the gate. |
| Disaggregated transfer compatibility | **Not production-ready** | Packed pages plus fp16 sink/tail side state need a connector payload contract. Current connector mode rejects rather than reinterpreting packed records as dense K/V. |
| CUDA graph lifecycle | **Not production-ready** | Side tensors are preallocated, but request-slot assignment, slot recycling, and reference restore/scoring still use Python/host control. |
| Sparse packed reads | **Missing** | HISA/Indexer sparse selection over packed KVarN records needs a dedicated read/dequant path. |
| Fused B200 store/decode kernels | **Missing** | Current backend is Python-level restore plus SDPA; no `torch.ops.trtllm.kvarn_gqa_decode`/store kernel is registered. |
| Correctness vs fp16/fp8 KV | **Partial only** | Pack/dequant round-trip, finite restore, cosine floor, side-state, and fail-close tests exist. Full attention/logit parity against fp16/fp8 GQA KV is not run/proven. |
| Performance proof | **Missing** | Microbench has a correctness floor and `--require-fused` promotion guard, but no fused B200 numbers or c16 tok/s/user proof exist. |
| Production enablement | **Blocked** | Requires fused kernels, disagg side-state transfer, sparse packed reads, graph-safe lifecycle, fp16/fp8 correctness proof, and c16 E2E performance proof. |

## Current op-trt status

Implemented:

- `tensorrt_llm/_torch/attention_backend/kvarn_gqa.py` defines the GQA KVarN
  dtype parser, Huawei-compatible byte layout, 2/3/4-bit bitstream pack/unpack,
  Hadamard + variance-normalized store, dequant restore, a reference packed
  pool, and transfer-view metadata shape.
- `tensorrt_llm/_torch/attention_backend/kvarn_gqa_attention.py` is the first
  runnable GQA KVarN backend. It forces split Q/K/V, commits full context blocks
  into packed KVarN records, keeps the first 128 sink tokens and in-progress
  speculative tail in fp16, restores packed records for GQA SDPA, and rejects
  sparse-indexed reads until a packed sparse read path exists. The side buffers
  are preallocated by layer/request slot with commit metadata sized to
  `max_blocks_per_seq`, so ordinary decode does not allocate new sink/tail
  tensors inside the step.
- `_util.py` allocates non-MLA KVarN GQA KV as an opaque UINT8 self-only page
  pool. For k2v2/g128 it uses `tokens_per_block=128`, `head_dim=76`, and
  `CacheType.SELFKONLY`, so each page is exactly one 9,728-byte KVarN record per
  local KV head.
- Resource/capacity accounting uses the packed slope
  `layers * local_kv_heads * 76 bytes/token` rather than dense K+V bytes.
- `kv_cache_config.dtype="kvarn_k2v2_g128"` is accepted only with
  `tokens_per_block=128`; validation remains fail-closed unless
  `TRTLLM_ENABLE_KVARN_GQA_REFERENCE=1` is set for isolated reference runs.
- Hugging Face artifacts can request GQA KVarN explicitly through top-level
  `kv_cache_dtype`, or through `quantization_config.kvarn.gqa`.
- HF artifacts can declare production default support without a YAML override by
  setting `supports_kvarn_gqa=true`, `kvarn_gqa_supported=true`,
  `default_kvarn_gqa=true`, `smc_sd_gqa_kvarn=true`, or
  `quantization_config.kvarn.gqa.enabled=true` with no explicit dtype. The
  default dtype is `kvarn_k2v2_g128`.

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

## Remaining production work

The production fail-close remains in place for GQA KVarN. The opt-in reference
backend is useful for isolated correctness experiments, but these pieces must be
finished before the deployment can be called complete:

1. CUDA/Triton kernels: replace Python restore + SDPA with fused full-block
   store, packed 2-bit load, dequant/scaled dot-product/value accumulation for
   generation and SMC draft/verify query shapes, including odd M values such as
   25. Python per-token/tile loops are functional but not acceptable for the
   c16 throughput target.
2. Disaggregated transfer: `kv_extractor.py` and the cache transceiver must
   transfer packed records plus fp16 sink/tail side-state as KVarN records. The
   side pool now exposes a tensor snapshot contract (`sink_k`, `sink_v`,
   `tail_k`, `tail_v`, `tail_filled`, `tail_block_start`, `committed`,
   `commit_gen`), but connector mode still rejects at startup until those
   tensors are registered/restored with the transfer backend.
3. Sparse-indexed GQA reads: HISA/Indexer state remains separate and is not
   quantized by KVarN, but sparse read selection over packed KVarN records needs
   its own read/dequant path before `sparse_attn_config` can compose with this
   backend.
4. CUDA graph state: sink/tail tensors and commit generations are preallocated,
   but request-slot assignment and reference restore/scoring are still
   Python-managed. Full graph capture needs native store/restore/decode kernels
   plus a request lifecycle hook to recycle side-pool slots without host-side
   mutation inside a captured region.
5. LayerSplit/CP proof: packed pages are byte pages and can be owner-split, but
   the current code still needs a full LayerSplit run to verify non-owner scratch
   routing for generic GQA KVarN, separate from dense MLA LayerSplit.

## Tests and benchmark harness

Current focused coverage:

- `tests/unittest/llmapi/test_kvarn_gqa_config.py`: dtype validation, HF explicit
  request, HF default-on declaration, explicit disable, default startup
  fail-close, and env-gated reference quant-mode selection.
- `tests/unittest/_torch/attention/test_kvarn_gqa.py`: k2v2 record layout,
  2/3/4-bit pack/unpack, store/restore shape/finiteness/cosine, packed-pool
  commit state, transfer-view shape, fixed-capacity side-pool behavior, sink/tail
  fail-close, and side-state snapshot shape.
- `benchmarks/python/bench_kvarn_gqa_micro.py`: light pack/restore/reference
  scoring timing for `M in {1,5,25}`. It enforces a restore-cosine floor and
  supports `--require-fused`, which fails until a real
  `torch.ops.trtllm.kvarn_gqa_decode` op is registered. Use it only on an idle
  GPU or CPU; it is a reference baseline, not the fused production-kernel
  benchmark.

Example:

```bash
python benchmarks/python/bench_kvarn_gqa_micro.py --device cuda --kv-heads 8 --iters 100 --queries 1 5 25
python benchmarks/python/bench_kvarn_gqa_micro.py --device cuda --require-fused
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

`kvarn_k2v2_g128` is now HF-deployable/defaultable and has an opt-in
reference GQA KV backend for non-MLA models. It remains fail-closed by default
and is not production-promoted; the remaining work is native fused decode/store,
disaggregated side-state transfer, sparse packed reads, CUDA graph lifecycle,
and E2E perf proof before it can satisfy the production throughput target.

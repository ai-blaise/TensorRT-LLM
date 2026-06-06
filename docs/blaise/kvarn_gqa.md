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
  scoring for isolated code review only. Startup still fails closed unless both
  `torch.ops.trtllm.kvarn_gqa_store` and
  `torch.ops.trtllm.kvarn_gqa_decode` are registered and
  `torch.ops.trtllm.kvarn_gqa_backend_ready()` returns true. Do not enable this in the
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
| HF deployability/default | **Implemented, fail-closed by default** | HF can request/default `kvarn_k2v2_g128` via top-level `kv_cache_dtype` or `quantization_config.kvarn.gqa`; startup rejects GQA KVarN unless the fused store/decode ops are registered and the production backend removes the gate. |
| Disaggregated transfer compatibility | **Not production-ready** | Packed pages plus fp16 sink/tail side state need a connector payload contract. Current connector mode rejects rather than reinterpreting packed records as dense K/V. |
| CUDA graph lifecycle | **Not production-ready** | Side tensors are preallocated, but request-slot assignment, slot recycling, and reference restore/scoring still use Python/host control. |
| Sparse packed reads | **Missing** | HISA/Indexer sparse selection over packed KVarN records needs a dedicated read/dequant path. |
| Fused B200 store/decode kernels | **Store/decode prototypes only; not production-ready** | `torch.ops.trtllm.kvarn_gqa_store` and `torch.ops.trtllm.kvarn_gqa_decode` now have experimental serial correctness kernels. Store performs Hadamard rotation, KVarN variance normalization, 2-bit packing, and fp16 scale/zp writes; decode reads fp16 sink tokens, compact records or byte-page KV-cache layout, and fp16 tail tokens directly, then performs Hadamard-rotated K/V dequant plus softmax attention across the combined sequence. Disaggregated side-state transfer, sparse packed reads, graph lifecycle, runtime parity, and performance proof are still missing, so `kvarn_gqa_backend_ready()` remains false. |
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
  `tokens_per_block=128`; validation remains fail-closed unless both
  `torch.ops.trtllm.kvarn_gqa_store` and
  `torch.ops.trtllm.kvarn_gqa_decode` are registered and
  `torch.ops.trtllm.kvarn_gqa_backend_ready()` returns true. The current
  store/decode ops are serial prototypes. Decode now covers fp16 sink + packed full
  blocks + fp16 tail, but the path is not runtime-parity or perf proven, so it
  must not be promoted by config defaults.
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

The production fail-close remains in place for GQA KVarN. The reference
backend is not a deployment path; these pieces must be finished before the
deployment can be called complete:

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

## Post-gen56 integration test plan

Run this matrix only after the active gen56 rollout is healthy and an explicit GPU
window is available. Until every item passes, keep `kvarn_gqa_backend_ready()`
false and keep SMC-SD/GQA KVarN out of production defaults.

1. **Config and HF defaults**
   - Dense MLA artifacts: verify supported dense-MLA configs resolve
     `mla_latent_kv_dtype="kvarn_k2v2"` and `mla_latent_kv_amortize=True`.
   - GQA artifacts: verify `kv_cache_dtype="kvarn_k2v2_g128"` and
     `quantization_config.kvarn.gqa` are accepted but startup fails closed while
     `kvarn_gqa_backend_ready()` is false.
   - Indexer configs: verify `indexer_k_dtype` / HISA/indexcache remain separate
     and no Indexer K tensor is routed through GQA KVarN.

2. **Single-node fused-kernel correctness**
   - Build with `torch.ops.trtllm.kvarn_gqa_store`,
     `torch.ops.trtllm.kvarn_gqa_decode`, and `kvarn_gqa_backend_ready()` still
     false. Run op-level shape/error tests for unsupported head/group sizes.
   - Enable readiness only in a test image and compare fused GQA logits/output
     with fp16/fp8 KV for `M in {1, 5, 25}`, batch 1 and 16, and sequence
     lengths `{1k, 8k, 32k, 64k, 128k}`.
   - Include odd-M SMC draft/verify shapes from the SGLang block-FP8 draft path;
     storage remains 128-token K/V tiles, odd M only affects decode scoring.

3. **LayerSplit and request pinning**
   - Prefill CP2 LayerSplit with decode CP1: verify packed pages follow the same
     physical block ids as dense KV and that non-owner ranks never read stale or
     missing GQA KVarN records.
   - Disaggregated prefill/decode: verify every request has matching
     `disagg request pin established` and `disagg request pin cleared` logs and
     no unpinned non-MORI request reaches decode.

4. **Moondream overlap, SMC, and WarpDecode**
   - Enable SMC overlap and verify overlap commit preserves KVarN metadata, SMC
     group identity, and request-pinning metadata.
   - Keep WarpDecode forced with TP/EP and verify GQA KVarN failure paths are
     explicit; no hidden fallback to fp16/fp8/nvfp4 KV and no HELIX routing.

5. **Transport variants**
   - UCX baseline: transfer packed records plus fp16 sink/tail side-state as
     typed KVarN payloads, not dense K/V reinterpretations.
   - NIXL and Mooncake: repeat the same transfer metadata checks once wrappers
     are available.
   - MORI: run only after request pinning and non-MORI transport gates pass;
     require a faster-than-UCX proof before promotion.

6. **Performance gates**
   - `benchmarks/python/bench_kvarn_gqa_micro.py --require-fused` must report the
     fused path present, ready, correct, and faster than the reference path.
   - `benchmarks/python/bench_kvarn_gqa_production_gate.py --dry-run` emits the
     required post-gen56 matrix for sequence length, odd-M, LayerSplit, request
     pinning, Moondream overlap, SMC, WarpDecode, and transport variants. Run it
     without `--dry-run` only in a fused-kernel test image; it fails closed unless
     `kvarn_gqa_backend_ready()` returns true.
   - Deployment benchmark must meet or beat the baseline and target c16
     tok/s/user after first token across `{1k, 8k, 32k, 64k, 128k}`.


## Fused B200 implementation plan

The next production patch should land a real fused path, not another reference
wrapper. File/function boundaries:

1. **CUDA/C++ op registration**
   - `cpp/tensorrt_llm/kernels/kvarnGqaKernels.{h,cu}` and
     `cpp/tensorrt_llm/thop/kvarnGqaOp.cpp` exist. The THOP layer accepts both
     compact record layout `[blocks, kv_heads, 9728]` and KV-cache page layout
     `[blocks, planes, 128, kv_heads, 76]` so the production kernel can avoid
     Python record copies.
   - `kvarnGqaStoreK2V2G128` currently contains only a serial store prototype
     for full 128-token blocks. It applies normalized Hadamard rotation, 16
     KVarN/SINQ variance-normalization iterations, asymmetric 2-bit RTN, and
     fp16 scale/zero-point writes into the packed record/page layout.
   - `kvarnGqaDecodeK2V2G128` currently contains only a serial decode prototype.
     It rotates Q, reads fp16 sink tokens, packed 2-bit K/V records, and fp16
     tail tokens, dequantizes packed blocks with stored fp16 scales/zero points,
     computes one softmax across the combined sequence, and inverse-rotates V.
     These are correctness stepping stones, not the B200 throughput kernels. The
     readiness op must remain false until CUDA graph lifecycle, transfer,
     sparse-read, runtime correctness, and performance gates all pass.
   - Build integration belongs in the existing CMake/Bazel custom-op lists next
     to the other TRT-LLM torch custom ops.

2. **Python dispatch and cache integration**
   - Replace the reference restore path in
     `tensorrt_llm/_torch/attention_backend/kvarn_gqa_attention.py` with calls to
     the fused ops when `model_loader._has_kvarn_gqa_fused_backend()` is true.
   - Keep `_util.py` allocation as UINT8 `CacheType.SELFKONLY`,
     `tokens_per_block=128`, `head_dim=76`, and add explicit fp16 sink/tail side
     buffer registration instead of Python dict lifetime.
   - Keep `utils.py` sparse-attention rejection until sparse packed reads are
     implemented; do not route Indexer/HISA through this path.

3. **Disaggregation and graph lifecycle**
   - Extend `tensorrt_llm/_torch/disaggregation/resource/kv_extractor.py` to
     expose KVarN record payloads plus `sink_k/sink_v/tail_k/tail_v/valid/commit_gen`
     side-state as typed regions.
   - Update connector/transceiver metadata so UCX/NIXL/Mooncake transport
     selection moves opaque KVarN records without treating them as dense K/V.
   - Add request lifecycle hooks in the KV manager/resource manager to recycle
     side-pool slots graph-safely on request finish, reject/replay, and block
     reuse.

4. **Correctness gates**
   - Add CPU/torch round-trip tests for record layout and fused op shape checks.
   - Add CUDA parity tests comparing fused GQA KVarN attention/logits against
     fp16/fp8 KV for M in `{1, 5, 25}`, batch/concurrency 1 and 16, and sequence
     lengths `{1k, 8k, 32k, 64k, 128k}`.
   - Include SMC draft/verify odd-M, LayerSplit CP2 ownership, request pinning,
     Moondream overlap, WarpDecode EP/TP, and connector transport selection in
     smoke/E2E gates.

5. **Benchmark gates**
   - Extend `benchmarks/python/bench_kvarn_gqa_micro.py --require-fused` to fail
     unless both fused ops are registered and faster than the reference restore
     path.
   - Add a deployment benchmark that records tok/s/user after first token at c16
     across 1k, 8k, 32k, 64k, and 128k. Do not remove the fail-close until this
     is faster than fp16/fp8 KV and meets the production target.

## Tests and benchmark harness

Current focused coverage:

- `tests/unittest/llmapi/test_kvarn_gqa_config.py`: dtype validation, HF explicit
  request, HF default-on declaration, explicit disable, default startup
  fail-close, and fused-op-gated quant-mode selection.
- `tests/unittest/_torch/attention/test_kvarn_gqa.py`: k2v2 record layout,
  2/3/4-bit pack/unpack, store/restore shape/finiteness/cosine, packed-pool
  commit state, transfer-view shape, fixed-capacity side-pool behavior, sink/tail
  fail-close, and side-state snapshot shape.
- `benchmarks/python/bench_kvarn_gqa_micro.py`: light pack/restore/reference
  scoring timing for `M in {1,5,25}`. It enforces a restore-cosine floor and
  supports `--require-fused`, which fails until a real
  `torch.ops.trtllm.kvarn_gqa_store` and
  `torch.ops.trtllm.kvarn_gqa_decode` are registered and
  `torch.ops.trtllm.kvarn_gqa_backend_ready()` returns true. `--try-store-op`,
  `--try-decode-op`, and `--try-side-op` are development-only parity checks for
  the experimental ops against the Python KVarN oracle, including fp16 sink +
  packed block + fp16 tail decode. They must be run only after a C++ build on an
  idle GPU and do not imply production readiness while `backend_ready()` is false. Use the
  default mode on CPU; it is a reference baseline, not the fused production-kernel
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

`kvarn_k2v2_g128` is now HF-deployable/defaultable and has reference GQA KV
code for review, but it remains fail-closed by default and is not
production-promoted. The remaining work is native fused decode/store,
disaggregated side-state transfer, sparse packed reads, CUDA graph lifecycle,
and E2E perf proof before it can satisfy the production throughput target.

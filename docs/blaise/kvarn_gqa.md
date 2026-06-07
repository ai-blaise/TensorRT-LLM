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
  validation, docs, and the microbench harness.
- **B: production-kernel candidate** is still not production-promoted. It adds
  byte-backed page allocation, block-parallel SM100/B200 packed store/decode
  ops, fused sparse top-k packed decode, fp16/bf16 sink/tail side state,
  BDR/in-kernel packed read hooks, and NIXL metadata/fragments for request-slot
  side-state. Startup still fails closed unless
  `torch.ops.trtllm.kvarn_gqa_backend_ready()` returns true. Do not enable this
  in the op-trt deployment until sparse-indexed packed reads have GPU parity,
  CUDA graph request lifecycle, full E2E transfer proof, and c16 B200 perf proof
  are complete.

Dense MLA KVarN remains separate and production-owned by
`sparse_attention_config.mla_latent_kv_dtype`; GQA KVarN never replaces the
Indexer/HISA sparse K path.

## Downtime audit checklist

| Requirement | Status | Notes |
|---|---|---|
| Packed 2-bit record format | **Implemented as primitives** | `kvarn_k2v2_g128` maps one 128-token block/head to a 9,728-byte record, hosted as 76 byte slots per token. Layout, bit pack/unpack, Hadamard/variance-normalized store, dequant restore, and transfer-view shapes are tested. |
| Dense/GQA separation from Indexer | **Implemented in config/docs; backend enforces separation** | Dense MLA KVarN uses `mla_latent_kv_dtype`; GQA uses `kv_cache_dtype`. Indexer/HISA sparse K remains separate and is not quantized. Unsupported GQA sparse offsets/shapes fail closed instead of routing through the Indexer or dense MLA path. |
| HF deployability/default | **Implemented, fail-closed by default** | HF can request/default `kvarn_k2v2_g128` via top-level `kv_cache_dtype`, `quantization_config.kvarn.gqa`, or explicit SMC `draft_kv_cache_dtype="kvarn_k2v2_g128"`; startup rejects GQA KVarN unless the fused store/decode/sparse-decode/dequant ops are registered and the readiness gate passes. |
| Disaggregated transfer compatibility | **Implemented as metadata/fragments; E2E proof pending** | Packed pages move through the existing byte-backed KV pool as opaque `UINT8` self-only blocks. KVarN GQA side-state now has explicit NIXL metadata (`kvarn_gqa_side_meta`) and final-slice transfer fragments for sink/tail/commit tensors keyed by request-pinned side-pool slots. The receiver mirrors its local block table into a device int64 tensor and reconstructs physical commit generation from transferred logical `committed`/`commit_gen`, so BDR restore can discover changed destination physical blocks after NIXL remapping without a host block-table loop. Unit coverage checks nonzero sender/receiver slot pointer offsets, `RecvReqInfo` serialization, and receiver-side BDR generation reconstruction. Full multi-rank NIXL proof is still pending, so backend readiness remains false. |
| CUDA graph lifecycle | **Partially guarded, not production-ready** | Side tensors are preallocated and the side pool now has an explicit `release_request()` cleanup path that clears sink/tail/commit state for abort/reuse. V1/V2 KV cache manager `free_resources()` calls release side-pool state before block removal. The BDR readable pool and `restored_gen`/`physical_commit_gen` tensors are preallocated/lazy-grown outside the hot restore. The sparse/store bench now has `--graph-replay` to capture/replay store, dense decode, and sparse decode, but B200 graph proof is still pending. |
| Sparse packed reads | **Fused candidate implemented, proof pending** | `sparse_kv_indices`/offsets compose by restoring through KVarN BDR/readable state and gathering selected tokens per KV head. `sparse_attn_indices` top-k scoring now has `torch.ops.trtllm.kvarn_gqa_decode_sparse`, a packed 2-bit in-kernel dequant/scoring path for top-k<=256 with -1 padding. Static SM100 compile and a runnable B200 harness exist, but GPU parity/perf, sparse offsets semantics, graph replay, and E2E HISA composition proof are still pending, so backend readiness remains false. |
| BDR fold / amortized dequant | **Prototype implemented; production fusion still gated** | GQA has a physical-block keyed readable pool plus device int64 commit/restored generation metadata. The request block table is mirrored into device int64 storage per layer/slot; restore maps logical full blocks to physical ids, filters valid committed records, reconstructs receiver physical generation after transfer when needed, applies torch.unique to stale physical ids, and batched-dequants only churn blocks into the persistent readable pool. The CUDA dense/sparse packed decode hooks also gather full-block physical ids from this mirrored device table, validate missing/uncommitted blocks with device async asserts instead of host `.item()` synchronizations, then bypass the readable-pool restore for safe generation/FULL-mask reads and call the GQA packed decode ops, which read packed 2-bit records and fold Hadamard dequant into scoring/value accumulation. The decode kernel is now block-parallel, but production readiness remains false until runtime parity, transfer, graph lifecycle, sparse-read, and B200 perf gates pass. |
| Fused B200 store/decode kernels | **Store/decode/sparse-decode candidates wired for safe cases; not production-ready** | The GQA store, dense packed decode, and sparse top-k packed decode ops now have block-parallel SM100/B200 kernels. CUDA aligned full-block commits are batched through the store op instead of walking each token through fp16 tail state; CUDA generation q_len=1 and FULL-mask reads call the packed decode op directly over byte pages, fp16 sink, and fp16 tail. The dense decode op launches one CTA per (query, head), 256 threads per CTA, shared-memory Q rotation, block reductions for softmax max/denom, and in-kernel packed 2-bit K/V dequant folded into scoring/value accumulation. For total sink+packed+tail length <=256 it dispatches a no-atomic small decoder. The sparse top-k op uses the same packed/sink/tail loaders and softmaxes only the selected logical token ids, avoiding readable-pool staging for `sparse_attn_indices`. Store packing now writes four 2-bit values per byte directly instead of zeroing and read/OR/writing each byte. Causal multi-token prefill, sliding-window, q_scaling, uncommitted full speculative blocks, and top-k>256 fail closed. Multi-rank NIXL E2E proof, CUDA graph replay, broader runtime parity, post-optimization store timing, and c16 B200 performance proof are still missing, so backend readiness remains false. |
| Correctness vs fp16/fp8 KV | **Partial only** | Pack/dequant round-trip, finite restore, cosine floor, side-state, and fail-close tests exist. Full attention/logit parity against fp16/fp8 GQA KV is not run/proven. |
| Performance proof | **Partial isolated decode only** | Earlier B200 microbench runs showed the packed decode candidate faster than the reference restore path for q_len/M in {1,5,25}, and the harness has a correctness floor plus `--require-fused` promotion guard. Store+read, sparse, graph replay, multi-rank transfer, and c16 tok/s/user proof are still missing, so this is not production-ready. |
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
  speculative tail in fp16, calls the CUDA packed store/decode ops when tensors
  are CUDA and the mask is a supported generation/FULL case, otherwise restores
  committed records through the BDR physical-block keyed readable pool for GQA
  SDPA on the CPU reference path, routes supported sparse attention top-k
  reads through the packed sparse decode hook, and fails closed for unsupported
  sparse offsets/shapes. The side buffers
  are preallocated by layer/request slot with commit metadata sized to
  `max_blocks_per_seq`, so ordinary decode does not allocate new sink/tail
  tensors inside the step. The current request block table is mirrored into
  device int64 storage before store/read; BDR restore and packed decode gather
  physical full-block ids from that device table, while `physical_commit_gen`
  and `restored_gen` plus `torch.unique` limit restore work to stale physical
  block ids, so reference dequant cost follows churn rather than the full
  working set. Receiver-side restore also rebuilds physical generation from
  transferred logical commit metadata after NIXL remaps packed pages to local
  physical block ids.
- `_util.py` allocates non-MLA KVarN GQA KV as an opaque UINT8 self-only page
  pool. For k2v2/g128 it uses `tokens_per_block=128`, `head_dim=76`, and
  `CacheType.SELFKONLY`, so each page is exactly one 9,728-byte KVarN record per
  local KV head.
- Resource/capacity accounting uses the packed slope
  `layers * local_kv_heads * 76 bytes/token` rather than dense K+V bytes.
- `kv_cache_config.dtype="kvarn_k2v2_g128"` is accepted only with
  `tokens_per_block=128`; validation remains fail-closed unless
  `torch.ops.trtllm.kvarn_gqa_store`, `torch.ops.trtllm.kvarn_gqa_decode`,
  `torch.ops.trtllm.kvarn_gqa_decode_sparse`,
  `torch.ops.trtllm.kvarn_gqa_dequant_amortized`, and
  `torch.ops.trtllm.kvarn_gqa_backend_ready()` are registered and the readiness
  op returns true. Store/decode/sparse-decode now launch block-parallel
  SM100/B200 kernels for supported full-block commit, q_len=1/FULL-mask reads,
  and top-k<=256 sparse packed reads, but the path is not graph/E2E/NIXL or c16
  perf proven, so it must not be promoted by config defaults.
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

## B200 decode proof snapshot

A low-memory GPU probe on `a4-us-002` built the GQA THOP extension from the
workspace sources and ran `torch.ops.trtllm.kvarn_gqa_decode` on one B200. The
probe used true GQA grouping (`num_heads=8`, `num_kv_heads=2`), k2v2/g128
packed records, fp16 and bf16 runtime dtypes, compact and paged layouts, and
odd SMC query counts `M in {1, 5, 25}`. The block-parallel no-atomic small
decode path is faster than the torch restore+score reference for packed full
blocks:

| dtype | layout | M | max_abs | packed decode us | restore+score ref us |
|---|---|---:|---:|---:|---:|
| fp16 | compact | 1 | 0.000088 | 111.12 | 403.97 |
| fp16 | compact | 5 | 0.000121 | 112.56 | 405.59 |
| fp16 | compact | 25 | 0.000120 | 118.24 | 399.05 |
| fp16 | paged | 1 | 0.000114 | 105.49 | 395.72 |
| fp16 | paged | 5 | 0.000120 | 107.51 | 355.46 |
| fp16 | paged | 25 | 0.000122 | 115.86 | 398.30 |
| bf16 | compact | 1 | 0.000876 | 111.53 | 379.19 |
| bf16 | compact | 5 | 0.000959 | 113.06 | 419.22 |
| bf16 | compact | 25 | 0.000970 | 116.12 | 423.98 |
| bf16 | paged | 1 | 0.000966 | 106.35 | 351.60 |
| bf16 | paged | 5 | 0.000958 | 107.38 | 358.64 |
| bf16 | paged | 25 | 0.000975 | 115.84 | 398.21 |

Side-state decode (`sink=16`, one packed block, `tail=7`) is correct but still
slower than the reference: about 579-680 us versus 358-459 us. The bottleneck is
fp16 sink/tail Hadamard rotation on read. The next optimization is to store or
transfer rotated sink/tail side buffers, or add a dedicated side-token tile path,
so side-state does not recompute 128-wide Hadamard rotations during decode.

The same GPU probe ran the new block-parallel store path and then decoded from
those stored records. Store restored max-abs versus the Python KVarN reference
was 0.028898 for fp16 and 0.041523 for bf16 across compact and paged layouts.
Store latency was about 2.57-2.59 ms per `(block, kv_head)` tile, so the path is
no longer serial but remains an optimization target. Store->decode retained the
packed decode timings above: about 105-116 us for paged/compact `M in {1,5,25}`.

## Remaining production work

The production fail-close remains in place for GQA KVarN. The reference
backend is not a deployment path; these pieces must be finished before the
deployment can be called complete:

1. CUDA/Triton kernels: optimize the first block-parallel packed store and
   decode kernels for generation and SMC draft/verify query shapes, including
   odd M values such as 25. Python per-token/tile loops are no longer on the aligned full-block CUDA
   store/decode path; full context/verify blocks batch into the packed store op,
   while odd SMC draft tails stay fp16 until accepted. Store latency still needs
   B200 tuning before production default. The packed decode kernel
   folds BDR dequant into the read/scoring path so packed records are not
   round-tripped through HBM beyond the persistent readable state required by
   the production design.
2. Disaggregated transfer: packed records now transfer as opaque byte pages and
   the native/NIXL worker registers request-slot side-state VRAM regions for
   `sink_k`, `sink_v`, `sink_len`, `tail_k`, `tail_v`, `tail_filled`,
   `tail_block_start`, `committed`, and `commit_gen`. The sender appends
   side-state fragments only on the final KV slice using the receiver-side
   request-pinned side slot. The receiver mirrors destination physical block ids
   into device storage and rebuilds physical commit generation from transferred
   logical side-state before BDR restore. Unit coverage now validates nonzero
   sender and receiver side-slot pointer offsets, `RecvReqInfo` serialization,
   and receiver-side BDR rerestore behavior. Remaining work is an actual
   multi-rank NIXL run
   that proves payload arrival, request abort/reuse behavior, and Moondream
   pinning interaction.
3. Sparse-indexed GQA reads: HISA/Indexer state remains separate and is not
   quantized by KVarN. Sparse KV token selection composes through the BDR
   readable pool and per-head gather. Sparse attention top-k scoring now has a
   fused packed read/dequant/scoring candidate op, but it still needs B200
   correctness/perf, sparse-offset/E2E HISA composition, and graph replay proof
   before this can be production default under HISA sparse decode.
4. CUDA graph state: sink/tail tensors and commit generations are preallocated,
   transfer uses fixed request-slot tensor regions, and request finish/abort
   releases side-pool slots through KV cache manager free hooks. The focused
   B200 harness supports `--graph-replay` for store, dense decode, and sparse
   decode. Full graph capture still needs a live replay run and confirmation that
   no host mutation occurs inside a captured region.
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
     `torch.ops.trtllm.kvarn_gqa_decode`,
     `torch.ops.trtllm.kvarn_gqa_dequant_amortized`, and `kvarn_gqa_backend_ready()` still
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
   - `benchmarks/python/bench_kvarn_gqa_micro.py --bdr-working-set-blocks 32 --bdr-churn-blocks 1` must show steady/churn restore cost tracking churn, not the full working set.
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
   - `kvarnGqaStoreK2V2G128` is now a block-parallel full-block pack path.
     It applies normalized Hadamard rotation, 16 KVarN/SINQ variance-normalization
     iterations, asymmetric 2-bit RTN, and fp16 scale/zero-point writes into the
     packed record/page layout. The latest measured B200 store latency is still
     too high, so this remains an optimization target before production default.
   - `kvarnGqaDecodeK2V2G128` and `kvarnGqaDecodeSparseK2V2G128` are
     block-parallel SM100/B200 candidate kernels. They rotate Q, read fp16/bf16
     sink tokens, packed 2-bit K/V records, and fp16/bf16 tail tokens, dequantize
     packed blocks with stored fp16 scales/zero points inside the attention read,
     score/softmax/accumulate, and rotate the output back. The sparse variant
     accepts `[num_kv_heads, q_len, topk]` logical token ids with -1 padding and
     top-k<=256. The readiness op must remain false until CUDA graph lifecycle,
     transfer, sparse-read runtime correctness, and performance gates all pass.
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
  supports `--require-fused`, which fails until real
  `torch.ops.trtllm.kvarn_gqa_store`, `torch.ops.trtllm.kvarn_gqa_decode`,
  `torch.ops.trtllm.kvarn_gqa_decode_sparse`, and
  `torch.ops.trtllm.kvarn_gqa_backend_ready()` are registered and ready.
  `blaise_perf/kvarn_gqa/bench_kvarn_gqa_sparse.py` is the focused low-memory
  B200 harness for the new packed sparse top-k op: it builds the THOP extension
  from the checkout, times store/dense decode/sparse-full/sparse-topk, optionally
  runs `--graph-replay`, accepts `--blocks` to compare fixed top-k against larger
  resident packed KV, and checks sparse-full equality against dense packed decode
  for odd M values when the resident set is <=256 tokens. The earlier
  `--try-store-op`, `--try-decode-op`, and `--try-side-op` development parity
  checks cover FP16/BF16 runtime tensors, compact records, paged KV-cache layout,
  odd SMC query counts, and fp16/bf16 sink + packed block + tail decode. They
  must be run only after a
  C++ build on an idle GPU and do not imply production readiness while
  `backend_ready()` is false. Use the default mode on CPU; it is a reference
  baseline, not the fused production-kernel benchmark.

Example:

```bash
python benchmarks/python/bench_kvarn_gqa_micro.py --device cuda --kv-heads 8 --iters 100 --queries 1 5 25
python benchmarks/python/bench_kvarn_gqa_micro.py --device cuda --runtime-dtype fp16 --sinkhorn-iters 16 --layouts compact paged --queries 1 5 25 --try-store-op --try-decode-op --try-side-op
python benchmarks/python/bench_kvarn_gqa_micro.py --device cuda --runtime-dtype bf16 --sinkhorn-iters 16 --layouts compact paged --queries 1 5 25 --try-store-op --try-decode-op --try-side-op
python benchmarks/python/bench_kvarn_gqa_micro.py --device cuda --require-fused
```



### Next GPU-window proof command

Run this from the VM worktree after building the TRT-LLM torch extension, on an
idle B200 only:

```bash
for dt in fp16 bf16; do
  python benchmarks/python/bench_kvarn_gqa_micro.py \
    --device cuda --runtime-dtype ${dt} --kv-heads 8 --iters 100 \
    --sinkhorn-iters 16 --layouts compact paged --queries 1 5 25 \
    --sink-side-tokens 16 --tail-side-tokens 7 \
    --try-store-op --try-decode-op --try-side-op
done
```

Then repeat with `--sink-side-tokens 128 --tail-side-tokens 1` and
`--sink-side-tokens 128 --tail-side-tokens 127` to cover nearly empty and nearly
full partial blocks. Promotion still requires this parity evidence plus the
production deployment benchmark and abort/reuse side-pool cleanup plus disagg lifecycle gates.

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

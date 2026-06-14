# HiSparse Capacity-Signal Gate -- 1a: Capacity Math (source-grounded)

All numbers below derive from the **exact** packed-record byte constant read
from production source, not an assumed value.

## The real byte constant (cite file:line)

`cpp/tensorrt_llm/thop/mlaBdrKvarnOp.cpp:32-41` `expectedBdrBytesPerBlock(...)`
(byte-identical formula at `cpp/tensorrt_llm/kernels/hisparseKvarnBdrRead.cuh:52-61`
and `cpp/tensorrt_llm/thop/SparseMlaDecodeKvarnHotOp.cpp:42-57`):

```
ckvBytes     = tokensPerBlock * kvLoraRank * ckvBits / 8
scaleZpBytes = tokensPerBlock * 2 * (kvLoraRank/128) * sizeof(half)   # sizeof(half)=2
peBytes      = tokensPerBlock * qkRopeHeadDim                         # E4M3 RoPE, 1 B/elem
bytes_per_block = ckvBytes + scaleZpBytes + peBytes
```

For the pinned production `kvarn_k2v2` block (tokens_per_block=64, kv_lora_rank=512, qk_rope_head_dim=64, kvarn_bits=2):

| term | bytes/block | note |
|------|-------------|------|
| ckvBytes (2-bit C-KV)        |   8192 | 64*512*2/8 |
| scaleZpBytes (fp16 scale+zp) |   1024 | 64*2*(512/128)*2, 2 scale + 2 zp per 128-subblock |
| peBytes (E4M3 RoPE)          |   4096 | 64*64, 1 byte/elem |
| **PACKED total**             | **13312** | = **208 B / token / layer** |

This matches the Python smoke constants exactly (`blaise_perf/hisparse/sparse_mla_kvarn_hot_smoke.py:25-32`: 128 + 16 + 64 = 208 B/token; *64 = 13312 B/block).

**Effective bit-width:** 208 B/token over 576 latent elems = 2.89 bits/elem (vs bf16 16 bits/elem). Density ratio vs bf16 latent = 5.54x.

## Per-token / per-layer footprints

| tier | bytes/token/layer | bytes/token (x61 layers) |
|------|-------------------|----------------------|
| (i) full-HBM bf16 latent (576*2) | 1152 | 70,272 B (68.6 KiB) |
| (ii) KVarN-only kvarn_k2v2 packed | 208 | 12,688 B (12.4 KiB) |

> Reconciliation with the plan's sketch (`hisparse_optrt_plan.md:2237-2242`): the
> plan's '~11 KB/tok (61 layers)' is the **packed** density
> (12.4 KiB/tok here), and its '~1.4 GB @128k' per-request figure is
> the **KVarN host-pool** size, NOT the full-HBM bf16 latent. The true full-HBM bf16
> per-request latent is ~8.6 GiB @128k (computed below). Both tiers are
> reported so the ceiling-lift is not understated.

## (iii) HiSparse hot buffer (device-resident working set)

HiSparse keeps only `hot_blocks_per_req` packed blocks per request DEVICE-resident
(across all layers); the rest of the packed KVarN pool lives on host and is
swapped in on miss. Device hot bytes/request:

    hot_blocks * 13312 B/block * 61 layers

| hot_blocks | hot tokens/layer | device hot bytes/request |
|-----------:|-----------------:|--------------------------|
| 32 | 2048 | 24.8 MiB (25,985,024 B) |
| 64 | 4096 | 49.6 MiB (51,970,048 B)  <- production default |
| 96 | 6144 | 74.3 MiB (77,955,072 B) |
| 128 | 8192 | 99.1 MiB (103,940,096 B) |

## Per-request device KV at context in {32k, 64k, 128k}

| context | (i) full-HBM bf16 | (ii) KVarN-only packed | (iii) HiSparse hot (hb=64) |
|--------:|-------------------|------------------------|----------------------------|
| 32k | 2.145 GiB | 0.387 GiB | 49.6 MiB |
| 64k | 4.289 GiB | 0.774 GiB | 49.6 MiB |
| 128k | 8.578 GiB | 1.549 GiB | 49.6 MiB |

Note (iii) is **context-independent** (the hot buffer is a fixed working set);
that is the entire point -- device KV per request stops growing with context once
>= hot_blocks_per_req*tokens_per_block tokens are resident, while the long tail
stays on host. HiSparse only engages at seq_len >= min_seq_len=65536.

## Headline: per-request HBM saving and ceiling lift vs KVarN-only

The relevant production comparison is HiSparse-hot (iii) vs the current
**KVarN-only** device residency (ii) -- both already use kvarn_k2v2, so this
isolates the HiSparse host/hot split from the KVarN quant win.

| context | KVarN-only/req (ii) | HiSparse hot/req (hb=64) | saving/req | **device-residency / ceiling lift** |
|--------:|---------------------|--------------------------|-----------:|------------------------------------:|
| 32k | 0.387 GiB | 49.6 MiB | 0.339 GiB | **8.0x** |
| 64k | 0.774 GiB | 49.6 MiB | 0.726 GiB | **16.0x** |
| 128k | 1.549 GiB | 49.6 MiB | 1.500 GiB | **32.0x** |

And vs the **full-HBM bf16** latent (the absolute dense baseline, e.g. a stack
with no KVarN and no HiSparse):

| context | full-HBM bf16/req (i) | HiSparse hot/req (hb=64) | **ceiling lift** |
|--------:|----------------------|--------------------------|-----------------:|
| 32k | 2.145 GiB | 49.6 MiB | **44.3x** |
| 64k | 4.289 GiB | 49.6 MiB | **88.6x** |
| 128k | 8.578 GiB | 49.6 MiB | **177.2x** |

## Concurrent-request ceiling on a fixed KV-HBM budget

Illustrative KV-cache HBM budget = 120 GiB/GPU (a conservative
slice of a 180 GB B200 after weights/activations/workspace). Max concurrent
requests = budget / per-request-device-KV:

| context | full-HBM bf16 | KVarN-only | HiSparse hot (hb=64) | HiSparse vs KVarN | HiSparse vs bf16 |
|--------:|--------------:|-----------:|---------------------:|------------------:|-----------------:|
| 32k | 56.0 | 309.9 | 2479 | 8.0x | 44.3x |
| 64k | 28.0 | 155.0 | 2479 | 16.0x | 88.6x |
| 128k | 14.0 | 77.5 | 2479 | 32.0x | 177.2x |

At 128k the HiSparse hot buffer is **context-independent 49.6 MiB/request**,
so the concurrent-128k ceiling is lifted **32.0x vs KVarN-only**
and **177.2x vs full-HBM bf16**, wherever decode is HBM-KV-bound.

### Max-context ceiling lift (single request, fixed device budget)

Equivalently, for a fixed per-request device-KV budget, the max context that
fits grows from O(budget/bytes_per_token) under KVarN-only to **unbounded by
the hot buffer** under HiSparse (the hot working set is fixed; context length
is limited only by host pool + page table). The device side stops scaling with
context at hot_blocks_per_req*tokens_per_block = 4096 tokens resident.

## Assumptions (explicit)

- num_hidden_layers = 61 (DeepSeek-V3.2-REAP config; all layers carry an MLA latent KV).
- bf16 full latent = 576 elems x 2 B (no separate K/V -- MLA stores a single latent; V is the first 512 dims, K is all 576 with RoPE in the last 64). This is the absorbed-MLA cache, matching the decode kernel's read (SparseMlaDecodeKvarnHotOp / sparse_mla_decode_kvarn_hot.cu).
- KVarN host-pool and HiSparse hot buffer both store the SAME 208 B/token/layer packed record.
- KV-HBM budget of 120 GiB/GPU is illustrative for the concurrency table only; the per-request
  ratios (saving, lift) are budget-independent.
- next_n / speculative draft tokens and the indexer-K cache are excluded (they are a small, context-independent additive term and do not change the ceiling-lift ratio).
- Excludes the resident sink/tail tokens kept in the normal bf16 decode pool (explicit_sink_tail_v1); these are O(sink_blocks + 1 tail block) per request, context-independent, and do not affect the lift.


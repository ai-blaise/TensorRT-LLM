#!/usr/bin/env python3
"""HiSparse capacity-signal gate -- Deliverable 1a.

Pure-arithmetic capacity model grounded in the REAL production BDR record byte
constant read from source (no assumed constants). Emits capacity_math.md.

Source-cited constants
----------------------
Packed kvarn_k2v2 BDR record bytes/block (cpp/tensorrt_llm/thop/mlaBdrKvarnOp.cpp
:32-41, expectedBdrBytesPerBlock; identical formula in
cpp/tensorrt_llm/kernels/hisparseKvarnBdrRead.cuh:52-61
hisparseKvarnK2v2BdrRecordBytes and SparseMlaDecodeKvarnHotOp.cpp:42-57):

    ckvBytes     = tokensPerBlock * kvLoraRank * ckvBits / 8
    scaleZpBytes = tokensPerBlock * 2 * (kvLoraRank/128) * sizeof(half)   # half=2B
    peBytes      = tokensPerBlock * qkRopeHeadDim                          # E4M3 = 1B/elem
    bytes_per_block = ckvBytes + scaleZpBytes + peBytes

Model dims (REAL, /models/BlaiseAI/DeepSeek-V3.2-REAP-345B-...-Graft/config.json):
    num_hidden_layers = 61, kv_lora_rank = 512, qk_rope_head_dim = 64,
    index_topk = 1024.  Production HiSparse params (docs/blaise/hisparse_optrt_plan.md
    :799,818,820,2229): hot_blocks_per_req = 64, host_to_device_ratio = 8,
    min_seq_len = 65536.
"""
from __future__ import annotations
import json

# ---- pinned production constants (cited above) ----
TOKENS_PER_BLOCK = 64
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
CKV_BITS = 2
SIZEOF_HALF = 2
LATENT_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM          # 576
NUM_LAYERS = 61                                        # DeepSeek-V3.2(-REAP) num_hidden_layers
BF16_BYTES = 2

def bdr_bytes_per_block(tpb=TOKENS_PER_BLOCK, klr=KV_LORA_RANK, qk=QK_ROPE_HEAD_DIM, bits=CKV_BITS):
    ckv = tpb * klr * bits // 8
    scalezp = tpb * 2 * (klr // 128) * SIZEOF_HALF
    pe = tpb * qk
    return ckv, scalezp, pe, ckv + scalezp + pe

CKV_B, SCALEZP_B, PE_B, PACKED_PER_BLOCK = bdr_bytes_per_block()
PACKED_PER_TOKEN = PACKED_PER_BLOCK // TOKENS_PER_BLOCK          # 208 B/token/layer
# full-HBM bf16 latent (dense MLA KV, no quant): 576 * 2 B
BF16_PER_TOKEN = LATENT_DIM * BF16_BYTES                          # 1152 B/token/layer

GiB = 1024**3
MiB = 1024**2

def per_request_bytes(ctx_tokens, bytes_per_token_layer, layers=NUM_LAYERS):
    return ctx_tokens * bytes_per_token_layer * layers

def fmt_gb(b): return f"{b/GiB:.3f} GiB"
def fmt_mb(b): return f"{b/MiB:.1f} MiB"

CONTEXTS = [32*1024, 64*1024, 128*1024]
HOT_BLOCKS = [32, 64, 96, 128]

lines = []
def P(s=""): lines.append(s)

P("# HiSparse Capacity-Signal Gate -- 1a: Capacity Math (source-grounded)")
P()
P("All numbers below derive from the **exact** packed-record byte constant read")
P("from production source, not an assumed value.")
P()
P("## The real byte constant (cite file:line)")
P()
P("`cpp/tensorrt_llm/thop/mlaBdrKvarnOp.cpp:32-41` `expectedBdrBytesPerBlock(...)`")
P("(byte-identical formula at `cpp/tensorrt_llm/kernels/hisparseKvarnBdrRead.cuh:52-61`")
P("and `cpp/tensorrt_llm/thop/SparseMlaDecodeKvarnHotOp.cpp:42-57`):")
P()
P("```")
P("ckvBytes     = tokensPerBlock * kvLoraRank * ckvBits / 8")
P("scaleZpBytes = tokensPerBlock * 2 * (kvLoraRank/128) * sizeof(half)   # sizeof(half)=2")
P("peBytes      = tokensPerBlock * qkRopeHeadDim                         # E4M3 RoPE, 1 B/elem")
P("bytes_per_block = ckvBytes + scaleZpBytes + peBytes")
P("```")
P()
P(f"For the pinned production `kvarn_k2v2` block "
  f"(tokens_per_block={TOKENS_PER_BLOCK}, kv_lora_rank={KV_LORA_RANK}, "
  f"qk_rope_head_dim={QK_ROPE_HEAD_DIM}, kvarn_bits={CKV_BITS}):")
P()
P(f"| term | bytes/block | note |")
P(f"|------|-------------|------|")
P(f"| ckvBytes (2-bit C-KV)        | {CKV_B:>6} | {TOKENS_PER_BLOCK}*{KV_LORA_RANK}*{CKV_BITS}/8 |")
P(f"| scaleZpBytes (fp16 scale+zp) | {SCALEZP_B:>6} | {TOKENS_PER_BLOCK}*2*({KV_LORA_RANK}/128)*2, 2 scale + 2 zp per 128-subblock |")
P(f"| peBytes (E4M3 RoPE)          | {PE_B:>6} | {TOKENS_PER_BLOCK}*{QK_ROPE_HEAD_DIM}, 1 byte/elem |")
P(f"| **PACKED total**             | **{PACKED_PER_BLOCK}** | = **{PACKED_PER_TOKEN} B / token / layer** |")
P()
P(f"This matches the Python smoke constants exactly "
  f"(`blaise_perf/hisparse/sparse_mla_kvarn_hot_smoke.py:25-32`: "
  f"128 + 16 + 64 = 208 B/token; *64 = {PACKED_PER_BLOCK} B/block).")
P()
P("**Effective bit-width:** "
  f"{PACKED_PER_TOKEN} B/token over {LATENT_DIM} latent elems = "
  f"{PACKED_PER_TOKEN*8/LATENT_DIM:.2f} bits/elem "
  f"(vs bf16 {BF16_PER_TOKEN*8/LATENT_DIM:.0f} bits/elem). "
  f"Density ratio vs bf16 latent = {BF16_PER_TOKEN/PACKED_PER_TOKEN:.2f}x.")
P()
P("## Per-token / per-layer footprints")
P()
P(f"| tier | bytes/token/layer | bytes/token (x{NUM_LAYERS} layers) |")
P(f"|------|-------------------|----------------------|")
P(f"| (i) full-HBM bf16 latent (576*2) | {BF16_PER_TOKEN} | {BF16_PER_TOKEN*NUM_LAYERS:,} B ({BF16_PER_TOKEN*NUM_LAYERS/1024:.1f} KiB) |")
P(f"| (ii) KVarN-only kvarn_k2v2 packed | {PACKED_PER_TOKEN} | {PACKED_PER_TOKEN*NUM_LAYERS:,} B ({PACKED_PER_TOKEN*NUM_LAYERS/1024:.1f} KiB) |")
P()
P(f"> Reconciliation with the plan's sketch (`hisparse_optrt_plan.md:2237-2242`): the")
P(f"> plan's '~11 KB/tok (61 layers)' is the **packed** density")
P(f"> ({PACKED_PER_TOKEN*NUM_LAYERS/1024:.1f} KiB/tok here), and its '~1.4 GB @128k' per-request figure is")
P(f"> the **KVarN host-pool** size, NOT the full-HBM bf16 latent. The true full-HBM bf16")
P(f"> per-request latent is ~{per_request_bytes(128*1024, BF16_PER_TOKEN)/GiB:.1f} GiB @128k (computed below). Both tiers are")
P(f"> reported so the ceiling-lift is not understated.")
P()
P("## (iii) HiSparse hot buffer (device-resident working set)")
P()
P("HiSparse keeps only `hot_blocks_per_req` packed blocks per request DEVICE-resident")
P("(across all layers); the rest of the packed KVarN pool lives on host and is")
P("swapped in on miss. Device hot bytes/request:")
P()
P(f"    hot_blocks * {PACKED_PER_BLOCK} B/block * {NUM_LAYERS} layers")
P()
P(f"| hot_blocks | hot tokens/layer | device hot bytes/request |")
P(f"|-----------:|-----------------:|--------------------------|")
for hb in HOT_BLOCKS:
    dev = hb * PACKED_PER_BLOCK * NUM_LAYERS
    star = "  <- production default" if hb == 64 else ""
    P(f"| {hb} | {hb*TOKENS_PER_BLOCK} | {fmt_mb(dev)} ({dev:,} B){star} |")
P()
P("## Per-request device KV at context in {32k, 64k, 128k}")
P()
P(f"| context | (i) full-HBM bf16 | (ii) KVarN-only packed | (iii) HiSparse hot (hb=64) |")
P(f"|--------:|-------------------|------------------------|----------------------------|")
for ctx in CONTEXTS:
    full = per_request_bytes(ctx, BF16_PER_TOKEN)
    kvarn = per_request_bytes(ctx, PACKED_PER_TOKEN)
    hot = 64 * PACKED_PER_BLOCK * NUM_LAYERS
    P(f"| {ctx//1024}k | {fmt_gb(full)} | {fmt_gb(kvarn)} | {fmt_mb(hot)} |")
P()
P("Note (iii) is **context-independent** (the hot buffer is a fixed working set);")
P("that is the entire point -- device KV per request stops growing with context once")
P(">= hot_blocks_per_req*tokens_per_block tokens are resident, while the long tail")
P("stays on host. HiSparse only engages at seq_len >= min_seq_len=65536.")
P()
P("## Headline: per-request HBM saving and ceiling lift vs KVarN-only")
P()
P("The relevant production comparison is HiSparse-hot (iii) vs the current")
P("**KVarN-only** device residency (ii) -- both already use kvarn_k2v2, so this")
P("isolates the HiSparse host/hot split from the KVarN quant win.")
P()
P(f"| context | KVarN-only/req (ii) | HiSparse hot/req (hb=64) | saving/req | **device-residency / ceiling lift** |")
P(f"|--------:|---------------------|--------------------------|-----------:|------------------------------------:|")
hot64 = 64 * PACKED_PER_BLOCK * NUM_LAYERS
for ctx in CONTEXTS:
    kvarn = per_request_bytes(ctx, PACKED_PER_TOKEN)
    save = kvarn - hot64
    lift = kvarn / hot64
    P(f"| {ctx//1024}k | {fmt_gb(kvarn)} | {fmt_mb(hot64)} | {fmt_gb(save)} | **{lift:.1f}x** |")
P()
P("And vs the **full-HBM bf16** latent (the absolute dense baseline, e.g. a stack")
P("with no KVarN and no HiSparse):")
P()
P(f"| context | full-HBM bf16/req (i) | HiSparse hot/req (hb=64) | **ceiling lift** |")
P(f"|--------:|----------------------|--------------------------|-----------------:|")
for ctx in CONTEXTS:
    full = per_request_bytes(ctx, BF16_PER_TOKEN)
    lift = full / hot64
    P(f"| {ctx//1024}k | {fmt_gb(full)} | {fmt_mb(hot64)} | **{lift:.1f}x** |")
P()
# concurrency ceiling on a fixed KV HBM budget
KV_BUDGET_GIB = 120.0  # illustrative per-GPU KV-cache HBM budget on a 180GB B200
budget = KV_BUDGET_GIB * GiB
P(f"## Concurrent-request ceiling on a fixed KV-HBM budget")
P()
P(f"Illustrative KV-cache HBM budget = {KV_BUDGET_GIB:.0f} GiB/GPU (a conservative")
P(f"slice of a 180 GB B200 after weights/activations/workspace). Max concurrent")
P(f"requests = budget / per-request-device-KV:")
P()
P(f"| context | full-HBM bf16 | KVarN-only | HiSparse hot (hb=64) | HiSparse vs KVarN | HiSparse vs bf16 |")
P(f"|--------:|--------------:|-----------:|---------------------:|------------------:|-----------------:|")
for ctx in CONTEXTS:
    full = per_request_bytes(ctx, BF16_PER_TOKEN)
    kvarn = per_request_bytes(ctx, PACKED_PER_TOKEN)
    n_full = budget / full
    n_kvarn = budget / kvarn
    n_hot = budget / hot64
    P(f"| {ctx//1024}k | {n_full:.1f} | {n_kvarn:.1f} | {n_hot:.0f} | {n_hot/n_kvarn:.1f}x | {n_hot/n_full:.1f}x |")
P()
P(f"At 128k the HiSparse hot buffer is **context-independent {fmt_mb(hot64)}/request**,")
P(f"so the concurrent-128k ceiling is lifted **{per_request_bytes(128*1024, PACKED_PER_TOKEN)/hot64:.1f}x vs KVarN-only**")
P(f"and **{per_request_bytes(128*1024, BF16_PER_TOKEN)/hot64:.1f}x vs full-HBM bf16**, wherever decode is HBM-KV-bound.")
P()
P("### Max-context ceiling lift (single request, fixed device budget)")
P()
P("Equivalently, for a fixed per-request device-KV budget, the max context that")
P("fits grows from O(budget/bytes_per_token) under KVarN-only to **unbounded by")
P("the hot buffer** under HiSparse (the hot working set is fixed; context length")
P("is limited only by host pool + page table). The device side stops scaling with")
P("context at hot_blocks_per_req*tokens_per_block = "
  f"{64*TOKENS_PER_BLOCK} tokens resident.")
P()
P("## Assumptions (explicit)")
P()
P(f"- num_hidden_layers = {NUM_LAYERS} (DeepSeek-V3.2-REAP config; all layers carry an MLA latent KV).")
P(f"- bf16 full latent = {LATENT_DIM} elems x 2 B (no separate K/V -- MLA stores a single latent; "
  "V is the first 512 dims, K is all 576 with RoPE in the last 64). This is the absorbed-MLA cache, "
  "matching the decode kernel's read (SparseMlaDecodeKvarnHotOp / sparse_mla_decode_kvarn_hot.cu).")
P(f"- KVarN host-pool and HiSparse hot buffer both store the SAME {PACKED_PER_TOKEN} B/token/layer packed record.")
P("- KV-HBM budget of 120 GiB/GPU is illustrative for the concurrency table only; the per-request")
P("  ratios (saving, lift) are budget-independent.")
P("- next_n / speculative draft tokens and the indexer-K cache are excluded (they are a small, "
  "context-independent additive term and do not change the ceiling-lift ratio).")
P("- Excludes the resident sink/tail tokens kept in the normal bf16 decode pool (explicit_sink_tail_v1); "
  "these are O(sink_blocks + 1 tail block) per request, context-independent, and do not affect the lift.")
P()

out = "\n".join(lines) + "\n"
with open("/work/capacity_math.md", "w") as f:
    f.write(out)
# also emit a small json of the headline numbers
summary = {
    "packed_bytes_per_block": PACKED_PER_BLOCK,
    "packed_bytes_per_token_layer": PACKED_PER_TOKEN,
    "bf16_bytes_per_token_layer": BF16_PER_TOKEN,
    "num_layers": NUM_LAYERS,
    "hot_buffer_bytes_per_req_hb64": hot64,
    "per_req_full_bf16_128k_bytes": per_request_bytes(128*1024, BF16_PER_TOKEN),
    "per_req_kvarn_only_128k_bytes": per_request_bytes(128*1024, PACKED_PER_TOKEN),
    "ceiling_lift_vs_kvarn_128k": per_request_bytes(128*1024, PACKED_PER_TOKEN)/hot64,
    "ceiling_lift_vs_bf16_128k": per_request_bytes(128*1024, BF16_PER_TOKEN)/hot64,
    "effective_bits_per_elem": PACKED_PER_TOKEN*8/LATENT_DIM,
    "density_ratio_vs_bf16": BF16_PER_TOKEN/PACKED_PER_TOKEN,
}
with open("/work/capacity_math.json", "w") as f:
    json.dump(summary, f, indent=2)
print(out)
print("WROTE /work/capacity_math.md and /work/capacity_math.json")

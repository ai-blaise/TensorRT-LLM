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
| `hisa.block_topk` | Sets the fixed candidate block count when `hisa.compression_ratio` is disabled. |
| `hisa.compression_ratio` | Sets the dynamic Figure 2(b)-style HISA candidate count: `ceil(num_blocks / compression_ratio)`, lower-bounded by the number of blocks needed to contain `index_topk` tokens. |
| `hisa.min_seq_len` | Sets the minimum sequence length for pre-Indexer HISA; the B200 default is 32768 so 8k/16k stay on the ordinary Indexer while 32k+ can use HISA. |
| `hisa.execution_mode` | Sets `hisa_execution_mode`; `auto` and `optimized` route to pre-Indexer HISA. |
| `indexcache.freq` / `indexcache.pattern` | Sets the OP-compatible IndexCache TopK reuse policy fields. |

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
validates that it uses the NVFP4 indexer K cache. HISA is implemented as a
pre-Indexer block-pruning stage: score pooled block representatives, keep the
candidate block set, and then run the token indexer only over tokens in those
blocks. The older post-logits HISA fallback is no longer a HISA execution mode;
if the pre-Indexer path is not eligible, runtime falls back to the ordinary
Indexer TopK rather than running a second HISA variant after full-prefix logits.

The first B200 HISA selector profiling pass used
`/tmp/optrt_dsa_ikp_bench.py` inside the Dynamo TensorRT-LLM runtime pod on
`a4-us-002-rl9` with the c32 production profile. Nsight Systems was used as the
available profiler on that node. For 32 rows, `next_n=1`, `index_topk=1024`,
and 131072 columns, the full HISA PyTorch selector took 0.3106 ms minimum; the
full-row fast path took 0.2299 ms minimum. For comparison, the CUDA C++ TopK
path took 0.0588 ms and the CuTe DSL TopK path took 0.0440 ms on the same
shape. The accepted change only removes redundant row masking and relative-index
repair when all rows cover the full logits width. The remaining large gap is the
next optimization target: replace the PyTorch HISA block selector with a fused
CuTe/CZS selector rather than iterating further on the fallback.

A second optimization gathers logits from the selected HISA blocks before the
final token TopK. This keeps the same HISA block selection but avoids masking and
scanning the full 65k/131k logits width for the token TopK, including CUDA graph
capture paths where host-side full-row checks are unavailable. On the same B200
runtime, selected TopK values matched the prior fallback exactly; returned
indices can differ only for tied values where PyTorch TopK has no unique
ordering. Minimum times were 0.1883 ms to 0.1373 ms for 32 rows and 65536
columns, 0.2376 ms to 0.1439 ms for 32 rows and 131072 columns, 0.2205 ms to
0.1769 ms for 64 rows and 65536 columns, and 0.2919 ms to 0.2064 ms for 128
rows and 65536 columns before widening the selected-block path beyond the
full-row fallback. A follow-up audit widened the selected-block path to ragged
rows and graph-capture-compatible execution. It matched the original HISA
selected-value semantics on full and ragged rows and passed a CUDA graph capture
smoke. Minimum times improved from 0.2813 ms to 0.1972 ms for 32 rows and
65536 columns, 0.3159 ms to 0.2356 ms for 32 rows and 131072 columns, 0.3144
ms to 0.2729 ms for ragged 64 by 65536, and 0.4311 ms to 0.3437 ms for 128
rows by 65536.

The TopK=1024, compression-ratio=4:1 fallback path also avoids zero-width
padding when the logits width is already block-aligned. This keeps selected
indices identical and removes a redundant allocation on the 65k, 131k, and
132096-width deployment shapes. B200 minimum times for the selected-block
fallback improved from 0.2875 ms to 0.2584 ms for 32 rows and 65536 columns,
0.3250 ms to 0.2830 ms for 32 rows and 131072 columns, 0.3286 ms to 0.2871 ms
for 32 rows and 132096 columns with a 131072-token valid prefix, 0.3147 ms to
0.2756 ms for ragged 64 by 65536, and 0.4289 ms to 0.3452 ms for 128 rows by
65536. CUDA graph capture smoke passed.

The same fallback caches the `arange` tensors used for validity masks and block
offset expansion. On the same TopK=1024, compression-ratio=4:1 B200 harness,
minimum times improved from 0.2806 ms to 0.2424 ms for 32 rows and 65536
columns, 0.3281 ms to 0.2781 ms for 32 rows and 131072 columns, 0.3299 ms to
0.2818 ms for 32 rows and 132096 columns with a 131072-token valid prefix,
0.3114 ms to 0.2670 ms for ragged 64 by 65536, and 0.4300 ms to 0.3388 ms for
128 rows by 65536. CUDA graph capture smoke passed.

Decode HISA calls pass a zero-start hint because their row starts are explicitly
constructed as zeros. That lets the selector build a one-sided tail mask instead
of a two-sided interval mask. On the same B200 harness, minimum times improved
from 0.2882 ms to 0.2167 ms for 32 rows and 65536 columns, 0.3271 ms to 0.2403
ms for 32 rows and 131072 columns, 0.3261 ms to 0.2433 ms for 32 rows and
132096 columns with a 131072-token valid prefix, and 0.4276 ms to 0.2829 ms for
128 rows by 65536. CUDA graph capture smoke passed.

The zero-start decode repair also skips the redundant lower-bound check in the
final index mask. Clean B200 comparison against the zero-start incumbent showed
0.2154 ms to 0.2011 ms for 32 rows and 65536 columns, 0.2417 ms to 0.2350 ms
for 32 rows and 131072 columns, 0.2450 ms to 0.2393 ms for 32 rows and 132096
columns with a 131072-token valid prefix, and 0.2889 ms to 0.2798 ms for 128
rows by 65536. CUDA graph capture smoke passed.

The final selected-candidate top-k in that fallback is unsorted. The downstream
sparse attention consumes the selected token set, while the stable block top-k
still determines the candidate blocks. A B200 comparison against the sorted
candidate incumbent preserved the selected token set for the tested deployment
shapes and improved minimum time from 0.2091 ms to 0.1952 ms for 32 rows and
65536 columns, 0.2528 ms to 0.2209 ms for 32 rows and 131072 columns, 0.2521
ms to 0.2232 ms for 32 rows and 132096 columns with a 131072-token valid
prefix, 0.3041 ms to 0.2801 ms for 128 rows by 65536, and 0.2387 ms to 0.2158
ms for ragged 64 by 65536.

On CUDA, the selected-candidate top-k uses the existing TensorRT-LLM Indexer
TopK op instead of PyTorch top-k. This keeps the stable HISA block selection and
accepts threshold-equivalent tied candidates inside the selected blocks. B200
comparison against the PyTorch unsorted candidate path showed threshold-correct
results and improved minimum time from 0.2045 ms to 0.1572 ms for 32 rows and
65536 columns, 0.2429 ms to 0.1861 ms for 32 rows and 131072 columns, 0.2433
ms to 0.1856 ms for 32 rows and 132096 columns with a 131072-token valid
prefix, 0.2954 ms to 0.2209 ms for 128 rows by 65536, and 0.2285 ms to 0.1648
ms for ragged 64 by 65536.

The HISA block top-k is also unsorted. HISA only requires the selected block set;
for BF16 tied block scores, either tied block is a valid representative. B200
comparison against the sorted-block, TRT-candidate-top-k incumbent preserved
threshold correctness and improved minimum time from 0.1192 ms to 0.1127 ms for
32 rows and 65536 columns, 0.1583 ms to 0.1424 ms for 32 rows and 131072
columns, 0.1592 ms to 0.1444 ms for 32 rows and 132096 columns with a
131072-token valid prefix, 0.1927 ms to 0.1848 ms for 128 rows by 65536, and
0.1382 ms to 0.1310 ms for ragged 64 by 65536.

The constant selected-candidate length vector passed to the TRT TopK op is
cached per `(device, rows, candidate_count)` just like the selector's range
tensors. This removes a decode-hot-path allocation. The B200 harness showed
minimum time improvements from 0.1077 ms to 0.1004 ms for 32 rows and 65536
columns, 0.1293 ms to 0.1255 ms for 32 rows and 131072 columns, 0.1331 ms to
0.1295 ms for 32 rows and 132096 columns with a 131072-token valid prefix,
0.1589 ms to 0.1567 ms for 128 rows by 65536, and 0.1186 ms to 0.1144 ms for
ragged 64 by 65536.

The decode path also includes `hisa.execution_mode: "reference"` for the
SGLang-style pre-Indexer flow against TensorRT-LLM's interleaved NVFP4 indexer
cache: NVFP4-dequantized mean block representatives, compression-ratio 4:1
block selection, candidate-token scoring only inside selected blocks, and the
existing TRT Indexer TopK for final token selection. `auto` and `optimized`
use the same pre-Indexer flow but route selected-candidate scoring through
TensorRT-LLM's FP4 paged MQA logits primitive instead of the Python dequantize
and `torch.matmul` reference loop. Block scoring is a single TF32-enabled
batched matmul over the full 128-dim indexer head rather than four 32-dim
matmuls. Mean-pool dequantizes each 128-dim token vector in one pass.

The NVFP4 mean-pool stage uses a fused persistent-grid CUDA kernel registered as
`trtllm::indexer_hisa_mean_pool_nvfp4`. It decodes TensorRT-LLM's interleaved
NVFP4 indexer cache and computes one 128-dim representative per HISA block
without materializing per-token dequantized vectors. In the B200 runtime pod,
the kernel matched the PyTorch dequantize-and-mean reference exactly on the
checked synthetic cache layouts. Minimum standalone mean-pool times improved
from 0.5138 ms to 0.0391 ms for batch 1 by 8192 tokens, 0.4547 ms to
0.0395 ms for batch 4 by 8192 tokens, 0.4426 ms to 0.0399 ms for batch 1 by
32768 tokens, 0.9686 ms to 0.0558 ms for batch 4 by 32768 tokens, 3.2539 ms to
0.1934 ms for batch 16 by 32768 tokens, and 6.2905 ms to 0.3668 ms for batch
32 by 32768 tokens.

With fused mean-pool and FP4 paged-MQA candidate scoring enabled, synthetic
64-head decode cells beat the pre-Indexer reference path across the checked
shapes: 1.3728 ms vs 2.5933 ms for batch 1 by 8192 tokens, 1.4085 ms vs
2.5513 ms for batch 4 by 8192 tokens, 1.3265 ms vs 2.5157 ms for batch 1 by
32768 tokens, 1.4243 ms vs 2.5851 ms for batch 4 by 32768 tokens, 3.7609 ms vs
4.8895 ms for batch 16 by 32768 tokens, and 6.8050 ms vs 8.8015 ms for batch
32 by 32768 tokens. This is the deployment path for HF configs that request
`hisa.execution_mode: "optimized"` or `auto`.

Pooled block representatives are quantized to NVFP4 and routed through the
DeepGEMM FP4 MQA primitive for block scoring. The promoted path passes
`max_seqlen_k=max_blocks`, which asks DeepGEMM for compressed per-row block
logits instead of a flattened batch-wide output followed by a gather. The
compressed output matched the gather-normalized output exactly on the B200
probe and reduced block-score time by about 4.26x to 4.42x across 8192 through
131072 tokens. With TopK=1024 and compression-ratio=4:1, the full score/top-k
probe showed HISA is still a loss at 8k/16k, but becomes useful at 32k and
improves with longer context: the 100k cell is about 1.86x faster than full
Indexer score+top-k and the 128k cell is about 2.09x faster before counting
the fused candidate-page, mask, and remap helper timings. The next
optimization target remains a CuTe/CZS or TensorRT fused block-score+top-k
selector, but the deployed block scorer is now the compressed DeepGEMM NVFP4
tensor path rather than the TF32 fallback.

## SMC-SD runtime configuration

SMC-SD is a deployment/runtime feature, not a target-model architecture field.
For the Blaise DeepSeek-V3.2 path, configure it through TensorRT-LLM/Dynamo
runtime arguments with the GLM draft model, `gamma=6`, `n_particles=4`, and draft
KV dtype `fp8_e4m3`. The SMC token-count contract is a hidden particle tree:
TensorRT-LLM stores `max_total_draft_tokens=gamma * n_particles` and verifies
`max_total_draft_tokens + 1` target positions, while only the selected parent
path is surfaced to the user.

The draft model receives an independent KV cache config when
`draft_kv_cache_dtype` is set. This keeps the target KV cache configuration
separate from the draft model's FP8 cache requirement. SMC accepts the deployment
spelling `fp8_e4m3` and maps it to TensorRT-LLM's unified `fp8` KV cache selector
for the draft model. The `trtllm_mha` draft attention selector maps to
TensorRT-LLM's `TRTLLM` attention backend.

The SMC drafter keeps the particle tree internal and returns selected draft-token
log probabilities, not full per-token draft logits. This preserves SMC log-weight
accounting while avoiding a `gamma * n_particles * batch * vocab` output tensor in
the static drafting loop. The sampler then scores all full particle paths with the
target probabilities from the tree-verify pass, selects the highest-weight
particle under the accumulated SMC weights, emits that particle without prefix
rejection, and uses the selected leaf's target bonus token for the next step.
CUDA graph capture pads only hidden dummy drafter rows; `SMCModelDrafter` copies
results back to the visible requests in the scheduled draft batch.

## LayerSplit

LayerSplit is a runtime/system feature: it configures how DSA KV and indexer
storage are partitioned and broadcast across context-parallel ranks. It is not
read from the HF model card and must be enabled explicitly through the
TensorRT-LLM runtime config, mirroring the WarpDecode and SMC-SD policy that
runtime topology features stay out of the checkpoint surface.

Enable via `SparseAttentionConfig` on `LlmArgs.sparse_attention_config` or its
`--trtllm.sparse_attention_config.*` CLI projection:

```python
sparse_attention_config = {
    "algorithm": "dsa",
    "indexer_mode": "indexcache-hisa",
    "indexer_k_dtype": "fp4",
    "layersplit_enabled": True,
    "layersplit_owner_assignment": "round_robin",        # or "contiguous"
    "layersplit_transfer_backend": "auto",                # auto | ucx | nixl
    "layersplit_all_cp_ranks_transfer": True,
}
```

The integration attaches LayerSplit to DSA metadata rather than to a standalone
cache-copy flag so the contract stays tied to the tensors that are actually
split: dense DSA KV and the indexer K cache. Runtime selection must compose with
the active TP, EP / MoE EP, attention DP, CP / DWDP, and context/generation
disaggregation mapping.

`layersplit_all_cp_ranks_transfer` must remain `true` until partial-rank
transfer is implemented; the `SparseAttentionConfig` validator rejects
`layersplit_enabled=True` with `layersplit_all_cp_ranks_transfer=False` to fail
fast at construction.

Legacy note: earlier preview versions of this adapter read
`quantization_config.indexer_quantization.layersplit.*` from the HF config.
That side-door has been removed; HF model cards no longer enable LayerSplit and
the keys are silently ignored. Use the runtime config above instead.

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
`warp_decode` is runtime configuration only; it is not read from the target
model's Hugging Face config because it describes a serving-time decode policy,
not an architecture or weight-format fact. The current op-trt hook is deliberately
narrow: it may call the packaged WarpDecode CuTe op only for BF16, top-k=8,
single-EP, no-communication decode shapes. The target NVFP4 production path must
use the native backend until the NVFP4 WarpDecode kernel is ported and validated.
In `auto` mode, unsupported shapes, quantization modes, CUDA-graph captures,
TP/EP layouts, attention-DP groups, CP layouts, or disaggregated generation
mappings fall back to the native backend. `policy: "force"` is reserved for
validation runs and turns fallback into an error.

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

The config validates the Blaise SMC-SD contract before constructing runtime
metadata, resource-manager, drafter, sampler, or worker state. SMC-SD keeps
separate draft/target KV ownership, CUDA-graph padding semantics, and decode-only
use under disaggregated prefill/decode. The target verification pass covers
`gamma * n_particles + 1` positions: every hidden particle token plus the selected
particle leaf bonus. Greedy user sampling still requests target probabilities for
SMC draft positions, and the SMC sampler disables the generic fast-greedy shortcut
so those probabilities are available for particle-weight updates.

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

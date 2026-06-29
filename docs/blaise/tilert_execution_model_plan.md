<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Blaise TileRT Execution-Model Plan

This branch treats TileRT as the target execution model direction for the
Blaise DeepSeek-V3.2 REAP checkpoint. It intentionally stops the prior local
fused-GEMM/MoE micro-campaign and focuses on a model-specific persistent decode
engine that owns the decode step pipeline.

## Current Compatibility Read

TileRT v0.1.4's public DeepSeek-V3.2 path is a model-specific runtime, not a
general TRT-LLM backend. The Python layer constructs a fixed eight-device DSA
graph and calls backend ops registered by `libtilert_dsv32.so`:

- `dsa_show_hands_prepare_money`
- `dsa_show_hands`
- `dsa_show_hands_reset`
- `dsa_show_hands_go_home`

The public converter expects official-style FP8 tensors named `.weight` and
`.weight_scale_inv`, then writes TileRT per-device keys suffixed
`_dev_{0..7}`. The Blaise target checkpoint is a compressed-tensors NVFP4
checkpoint with `.weight_packed`, `.weight_scale`, `.weight_global_scale`, and
`.input_global_scale` tensors. This is a real storage-format mismatch, not just
a path-name mismatch.

The major architectural mismatches are:

- **Routed experts:** TileRT's public default is 256; the Blaise REAP target is
  128. Backend temp buffers and expert kernels must accept 128 experts.
- **Index top-k:** TileRT's public default is 2048; the Blaise target is 1024.
  DSA sparse-selection buffers and kernels need a 1024-token contract.
- **MTP layer:** TileRT's converter assumes one extra layer after the base
  layers. The Blaise target declares `num_nextn_predict_layers=0` and has no
  `model.layers.61.*` keys. First target is non-MTP unless we add or graft a
  compatible MTP layer.
- **Weight format:** TileRT expects FP8 `.weight` and `.weight_scale_inv`
  tensors. The Blaise target uses NVFP4 packed weights, FP8 scales, global
  scales, and activation input scales. This needs a new converter and likely a
  new backend load/kernel layout.
- **Attention gate:** TileRT's public graph does not represent the Blaise
  `self_attn.gate_proj` stage after attention and before `o_proj`.
- **GatedNorm:** TileRT's public graph does not represent the input and
  post-attention low-rank gates. A persistent layer graph must include both
  gates at exact semantic points.
- **HISA/IndexCache:** TileRT has its own DSA sparse path. The Blaise target
  currently depends on TensorRT-LLM's NVFP4 IndexCache+HISA contract, so the
  persistent engine must either port Blaise HISA or match TileRT's sparse-index
  contract.

Routing is less concerning than it initially looked. TileRT's `ModelArgs`
default says `score_func="softmax"`, but the DeepSeek expert op path carries
sigmoid-plus-bias reference behavior and `route_scale`. The remaining risk is
whether the binary kernels bake the 256-expert and top-k-2048 shape.

## Working Hypothesis

The right path is not to wrap the public TileRT Python generator around the
Blaise checkpoint. The right path is to build a Blaise-specific TileRT-style
decode engine inside or adjacent to TRT-LLM:

1. Keep the TRT-LLM loader as the source of truth for Blaise semantics,
   compressed-tensors NVFP4 aliases, GatedNorm placement, attention output
   gate placement, and serving integration.
2. Add a persistent decode engine that owns several decode steps and keeps
   layer activations, routing state, DSA metadata, and scratch buffers resident.
3. Move whole layer-stage scheduling into the engine:
   input GatedNorm, DSA projection/index/attention, attention output gate,
   `o_proj`, post-attention GatedNorm, dense or MoE route/dispatch/expert/down,
   residual and next-layer norm.
4. Treat CUDA graphs as a launch wrapper only. The core win must come from a
   resident work scheduler and cross-op state residency, not more graph capture
   around existing launches.

## TRT-LLM Resident Path Status

The branch now has a production-shaped resident model-body handoff below
`ModelEngine.forward`. The handoff keeps the TRT-LLM scheduler, KV manager,
attention metadata, request slots, and loader as the source of truth, then calls
an optional DeepSeek native resident body before the normal per-token model path
is used.

The current native ABI carries:

- decode request shape: real batch, padded batch, token count, request ids,
  sequence lengths, and cached-token counts;
- persistent per-shape scratch for hidden states and logits;
- a stable resident tensor table built from the loaded model: embedding, final
  norm, LM head, per-layer norms, gated norms, attention tensors, dense MLP
  tensors, and MoE tensors;
- compact layer offsets, layer-kind ids, and per-layer semantic tensor-site
  indices so the C++ body can bind the tensor table without string parsing or
  fragile module-order assumptions in the hot path.

The registered C++ op still returns `deepseek_resident_decode_ready() == false`,
so serving falls back to the current production path. This is intentional until
the CUDA body owns at least the first real resident layer stage. The next
implementation target is a native layer scheduler that consumes this ABI and
starts replacing the Python DeepSeek layer loop, not another isolated fused-op
micro-optimization.

The executor-level resident decode backend now also has a strict native mode:
`TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_BACKEND=deepseek_native_resident`.
That backend refuses to own a multi-step decode window unless the previous
model-body step reports `deepseek_resident_native_v1`, sampling reports
`deepseek_resident_sampler_native_v1`, and the window backend reports
`deepseek_resident_window_native_v1`. This is a guardrail for C16/C32
benchmarking: if the native execution model is not active, the decode-window
path must report the specific missing native boundary instead of silently timing
the Python fallback.

The C++ resident path now also has a persistent
`torch.classes.trtllm.DeepseekResidentDecodeHandle`. The handle validates and
stores the resident layer offsets, layer kinds, and CUDA tensor table once, then
exposes a `decode(...)` method for the future resident CUDA body. The Python
native shim prefers this handle when it is available. The older
`deepseek_resident_decode_prepare(...)` entrypoint remains as a compatibility
validator for environments that have the op but not the class. This is the
native-side handshake for binding the model manifest before the layer scheduler
starts executing real stages.

The handle now owns the first executable resident stage:
`run_input_embedding(input_ids, hidden_states_scratch, input_tokens)`. It uses
resident tensor slot 0 (`model.embed_tokens.weight`) to populate the hidden-state
scratch for the active decode tokens. This is intentionally exposed as an
isolated stage for validation.

The Python native shim now also has a resident stage scheduler boundary:
`run_decode_step_scheduler(...)`. It sequences the owned handle stages, carries
the DeepSeek residual/next-layer-normalized hidden state across layers, and
returns an explicit first-missing-stage reason instead of pretending a partial
body is executable. This scheduler is available through
`TRTLLM_OPTRT_DEEPSEEK_RESIDENT_STAGE_SCHEDULER=1` for diagnostics, but serving
still falls back unless the full body produces logits. The key correctness rule
is that only the first layer runs `input_layernorm`; later layers consume the
previous layer's `next_layer_layernorm` output and the promoted residual state.

The scheduler can now bridge the current TRT-LLM DSA attention core through
`run_layer_attention_core_stage(...)`. When production `position_ids` and
`attn_metadata` are present, it now prefers the same split that TRT-LLM's DSA
MLA path exposes internally: `self_attn.forward_dsa_proj(...)` first produces
token-wise projection/indexer intermediates, the resident scheduler records
those tensors in per-layer scratch, then `self_attn.forward_dsa_attn(...)`
writes into resident `attention_core_output` scratch. If an older/fake layer
does not expose the split methods, the scheduler still falls back to
`self_attn.forward_impl_with_dsa(...)`. It deliberately does not call the full
attention module: the full module would also run the attention output gate and
`o_proj`, and those tail stages are resident-handle owned below. This is a
production-semantic bridge over the existing DSA/KV path, not the final
TileRT-style persistent CUDA attention scheduler. The scheduler exports
separate DSA projection and DSA attention-dispatch counters/reasons so the
remaining native gap is visible instead of one opaque attention-core failure.
The C++ resident handle now also exposes the compiled handoff for that second
half: `run_layer_dsa_attention_dispatch_ready()`,
`run_layer_dsa_attention_dispatch_not_ready_reason()`, and
`run_layer_dsa_attention_dispatch(...)`. The dispatch method validates the
projection tensors, indexer-intermediate list, position ids, sequence lengths,
KV lengths, DSA metadata descriptor, resident DSA dispatch scratch, output
scratch, and token count, then remains fail-closed until the CUDA body is
implemented. The descriptor currently requires the first production decode
read-set inputs the native owner will need: the verified TopK tensor from the
production indexer path, `block_table`, per-layer indexer K cache,
`indexer_k_cache_block_offsets`, and `scheduler_metadata_buffer`; optional
expanded/MTP and heuristic buffers are threaded through when present. The
resident scratch ABI now separately provides per-layer fused-q scratch,
latent-attention-output scratch, `cu_q_seqlens`, `cu_kv_seqlens`, and the FMHA
scheduler counter. The manifest also exposes the MLA dispatch assets
(`k_b_proj_trans`, `v_b_proj`, and their scale/dequant variants where present).
The runtime descriptor is named rather than positional and now includes the
production MLA rope/KV-write inputs (`rotary_cos_sin`,
`kv_cache_block_offsets`, host KV pool pointers/mapping, host KV lengths and
prompt lengths), dense NVFP4 sparse-read pools (`dense_kv_pool`,
`dense_kv_scale_pool`), optional block-id/Helix/sparse-MLA scheduler buffers,
and scalar MLA config (`tokens_per_block`, heads, LoRA/rq/rope/value dims,
quant mode, `q_scaling`, `softmax_scale`). The remaining native body work is to
consume those tensors and call the production MLA rope/KV write, HiSparse
KVarN-hot or dense NVFP4 sparse read, and v_b projection path without routing
through Python `forward_dsa_attn(...)`.

The first executable C++ body slice now exists behind
`TRTLLM_OPTRT_DEEPSEEK_RESIDENT_DSA_NATIVE_BODY=1`. It is deliberately narrow:
BF16 `k_b_proj_trans`, BF16 `v_b_proj`, 128 heads, dense NVFP4 KV pools,
`tokens_per_block=64`, and the v_b-fused sparse MLA read. The body builds
`fused_q` in resident scratch, calls the existing `mla_rope_generation(...)`
thop to apply RoPE and write KV, reshapes the dense NVFP4 data/scale pools,
uses the transformed pool/global TopK indices, calls
`sparse_mla_decode_nvfp4_vfuse(...)`, and writes projected attention output into
resident `attention_core_output` scratch. Default readiness remains false; the
gate is for live proof/smoke only until numerical and serving validation pass.
In diagnostics, the Python scheduler uses the native dispatch when it reports
ready; otherwise it records the native not-ready reason and falls back to the
existing Python `forward_dsa_attn(...)` bridge. If the native dispatch reports
ready but required metadata is missing, the stage fails closed instead of
falling back.

The handle also owns the next resident layer stage:
`run_layer_input_rmsnorm(layer_idx, hidden_states_scratch,
norm_hidden_states_scratch, input_tokens, eps, use_gemma)`. This reads the
embedding scratch, uses the layer's resident input RMSNorm weight, and writes a
separate normalized-hidden scratch so the embedding scratch remains available as
the first residual.

The handle now also owns the first Blaise-specific gated stage:
`run_layer_input_gated_norm(layer_idx, norm_hidden_states_scratch,
gated_hidden_states_scratch, input_tokens)`. This consumes the resident input
GatedNorm down/up weights, applies the same low-rank
`sigmoid(up(silu(down(x))))` gate as the Python DeepSeek layer, and writes a
separate gated-hidden scratch for the attention input.

The native manifest now also carries a compact semantic site table for each
layer. Early sites cover layer norms, input/post-attention GatedNorm weights,
attention projection sites, the DSA `k_b`/`v_b` projection tensors needed by a
native attention body, dense MLP sites, and common MoE/shared-expert sites. The
resident handle uses this site table for stage binding instead of assuming that
a specific module traversal order maps to a specific tensor meaning.

The handle also owns the first stages after attention output:
`run_layer_attention_output_tail(...)` consumes the attention-core output
scratch, applies the resident attention output gate from `gate_proj`, then runs
resident `o_proj` into hidden-state scratch. `run_layer_post_attention_rmsnorm(...)`
then performs the residual add plus post-attention RMSNorm into resident
residual and norm scratches, and `run_layer_post_attention_gated_norm(...)`
applies the post-attention low-rank gate into a separate scratch for dense MLP
or MoE input.

For dense layers, the handle now owns the FFN tail as well:
`run_layer_dense_mlp(...)` runs `gate_up_proj`, SwiGLU, and `down_proj` using
resident dense MLP weights and layer-specific intermediate scratch, and
`run_layer_post_ffn_rmsnorm(...)` performs the FFN residual add plus next-layer
RMSNorm into resident next-hidden and next-residual scratches. The dense method
checks the layer-kind id and refuses MoE layers, so the future MoE scheduler
cannot silently fall through this path.

For MoE layers, the handle now owns the first resident router stage:
`run_layer_moe_router(...)` consumes the post-attention gated hidden scratch,
runs the resident `mlp.gate.weight` projection into float32 router logits,
applies DeepSeek's sigmoid-plus-`e_score_correction_bias` no-aux grouped routing
contract, and writes normalized/scaled top-k weights plus int32 expert indices
into persistent scratch. This is still an ATen-backed staging method rather than
the final fused CUDA router/dispatch path, but it puts the routing semantics and
scratch ownership behind the native resident handle instead of the Python MoE
scheduler.

The scheduler can now bridge MoE expert execution through the existing
production backend as well. `run_layer_moe_experts_stage(...)` passes the
resident router-logits scratch directly to `mlp.experts(...)`, computes the
shared expert branch separately, combines routed plus shared outputs into
resident FFN scratch, and then lets the resident post-FFN RMSNorm stage carry
the layer state forward for the `do_finalize=True` path. It now also mirrors
the production deferred `POST_MOE_FUSION`/MoE-allreduce path: when the live
layer meets the same TRTLLM/NVFP4/P2P/max-token conditions as
`DeepseekV3DecoderLayer.forward_MoE(...)`, the bridge calls
`mlp.experts(..., do_finalize=False)`, computes shared experts, constructs
`MoEAllReduceParams`, and calls `layer.moe_allreduce(...)`. That allreduce
returns the next-layer normalized hidden/residual pair, so the resident
scheduler marks post-FFN as already finalized and skips the separate native
`run_layer_post_ffn_rmsnorm(...)` stage for that layer. The scheduler also now
promotes next-layer state only between layers, so the final layer's LM head
consumes the actual normalized final hidden state rather than stale scratch.

The native engine now exports per-execute state through `execution_state()`.
The executor-level strict backend (`deepseek_native_resident`) admits a
multi-step resident window only when the preceding model body reports one of
the explicit complete native-body reasons, including
`resident_stage_scheduler_completed`. Partial resident-stage runs such as
`layer_*_attention_core_missing` or `layer_*_moe_experts_missing` are rejected
before the window loop starts. This keeps future throughput numbers from
accidentally timing a Python fallback or an incomplete native body.

The multi-step window now has a production-shaped native handoff rather than a
PyExecutor loop placeholder. `PersistentDecodeWindowCallbacks` carries a
`resident_window_step(...)` callback, `ModelEngine` retains the last resident
model-forward request, and `deepseek_resident_native_v1` can delegate a
`PersistentDecodeModelWindowRequest` to the native engine with:

- the production forward request and already-prepared attention/KV metadata;
- the current `SampleState` from the previous token;
- the requested resident window length;
- the same invocation/body contracts used by the one-step resident body.

This is the C16/C32 insertion point. For `window_steps > 1`, strict native mode
now refuses to fall back to the Python per-token window unless explicitly
overridden for diagnostics. The real throughput path must make native
`execute_window(...)` return a `PersistentDecodeModelWindowResult` after owning
token feedback and multiple model/sampling steps. The current native engine
still reports `resident_window_native_not_implemented`, so this branch is not
yet benchmarkable as a true TileRT-style serving engine.

The native window handoff also now builds a stable serving-subset contract
before the engine can run. The contract records request ids, sequence slots,
remaining decode budget, cached-token/sequence-length snapshots, input and
position tensor specs, and the exact body invocation shape. It intentionally
rejects unstable features for the first C16/C32 target: streaming,
logprobs/generation logits, stop words, non-`-1` end ids, beam search, draft
tokens, dummy requests, duplicate slots, request-id mismatch, and too little
remaining token budget. That leaves a narrow but production-relevant fast path:
fixed decode-only cohorts with single-beam greedy sampling and deferred host
materialization.

The executor side now also has the multi-token sample-state protocol that this
window needs. `PersistentDecodeModelWindowRequest` carries a sample-state
factory owned by PyExecutor. A native window result can return a device tensor
of shape `[owned_steps, batch, 1]`, and PyExecutor will:

- placeholder-advance request state for the first `owned_steps - 1` tokens;
- keep the final returned `SampleState` as the normal executor boundary;
- on final update, replace the intermediate placeholders with the actual window
  tokens and append the final token;
- advance `py_decoding_iter` exactly once more at the final boundary.

The native shim now looks for a resident-handle method named
`run_decode_window(...)`, but method presence alone is not enough to admit the
production path. The handle must also return true from
`run_decode_window_ready()`. This keeps the compiled ABI fail-closed while the
real CUDA scheduler is still missing. When ready, the shim allocates persistent
window-token scratch, passes the current sampled token tensor, resident
hidden/logits scratch, position ids, device-side sequence/KV length tensors,
request ids, sequence lengths, and KV-length snapshots into the handle, and
converts the returned token tensor into the PyExecutor sample-state protocol.
The C++ handle now registers
`run_decode_window_ready()` and a validating `run_decode_window(...)` stub over
that ABI; the ready method remains false. The handle also exposes
`run_decode_window_not_ready_reason()`, currently returning
`resident_window_native_missing_dsa_attention_dispatch`, and the
Python/executor gates propagate that exact reason instead of the older generic
`resident_window_native_not_implemented`. With the Python bridge now split, the
next C++/CUDA task is concrete: replace the batch-dependent DSA attention
dispatch bridge with a native owner, then wire
`DeepseekResidentDecodeHandle::runDecodeWindow` logic that owns token feedback,
per-step position/KV metadata advancement, resident model execution, and native
greedy sampling for the stable C16/C32 cohort.

The handle also now exposes the first reusable window-loop primitive:
`run_decode_window_advance_state(...)`. It validates the resident token-window
contract, records the already-sampled token into output step 0, feeds that
token back into the mutable `input_ids` buffer for the next internal model
step, and advances device-side position and KV-length tensors by one. This is
still ATen-backed and is not a full scheduler, but it is the concrete state
transition the future `runDecodeWindow` loop must perform between internal
model/sampling substeps without returning to PyExecutor.

The second reusable window primitive is
`run_decode_window_sample_step(...)`. It samples the current logits with greedy
argmax directly into a caller-selected `[step, batch, beam]` slot in the
persistent window-token scratch. The native shim exposes this as
`run_decode_window_sample_step_stage(...)` so the eventual resident window loop
can keep internal sampled tokens on device. It deliberately does not advance
position or KV lengths; the loop should call `run_decode_window_advance_state`
only when that sampled token must be fed into another internal model step. The
final returned token should be sampled into the window tensor without an extra
metadata advance.

The third primitive is `run_decode_window_prepare_step(...)`. It combines the
token-feedback/metadata transition with the resident input-embedding stage for
the next internal decode step. The Python shim exposes this as
`run_decode_window_prepare_step_stage(...)` and verifies that it reuses the same
persistent window-token scratch later consumed by
`run_decode_window_sample_step_stage(...)`. This is the first native entry point
that spans two pieces of the loop contract: sampled-token feedback plus the next
step's model input preparation.

`run_decode_window(...)` now invokes that prepare step after the full window ABI
validates, then still throws because the resident model-body loop is not
implemented. Production serving cannot reach this partial body while
`run_decode_window_ready()` remains false, but direct native debug calls now
exercise the exact initial token-feedback, `input_ids`, position, KV-length, and
embedding boundary that the future loop must preserve.

The Python native-engine shim now also has a debug-only resident window stage
scheduler behind `TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_STAGE_SCHEDULER=1`.
That scheduler owns a full multi-step loop at the model-backend boundary:
`prepare_step` feeds back the previous token and embeds it, the resident
single-step stage scheduler starts from that prepared hidden-state scratch
without re-running embedding, and `sample_step` writes the next token into the
persistent window-token scratch. After each prepare step, the scheduler calls
`attn_metadata.on_update_kv_lens()` before re-entering attention so DSA/TRTLLM
derived scheduler/indexer buffers track the in-place KV-length advance. It
intentionally does not make `window_backend_state()` ready; strict executor
admission still requires the true native handle to return
`run_decode_window_ready() == true`. This gives us a safe way to debug the
window-owned execution model and the remaining attention core / MoE expert
ownership gaps without producing misleading throughput numbers.
The debug window scheduler now has explicit coverage for a MoE layer as well as
dense-only layers, so repeated in-window execution exercises router, expert,
shared-expert, combine, post-FFN, logits, and sampling stages before returning a
window sample state.

The local container test harness now needs the in-tree C++ build outputs ahead
of the preinstalled wheel libraries:

```bash
LD_LIBRARY_PATH=/repo/tensorrt_llm/libs:/repo/cpp/build/tensorrt_llm:$LD_LIBRARY_PATH
```

`th_common` also carries `$ORIGIN/..` in its build rpath so local loads can find
the matching in-tree `libtensorrt_llm.so` instead of an older installed copy.
With package-local generated artifacts staged from the current build, the three
resident executor unit files pass in `optrt-bench-claude`.

The DeepSeek-V3 final RMSNorm is not a separate model-tail call in this code
path: `post_load_weights()` points the final layer's `next_layer_layernorm` at
`model.norm`. That means the resident `run_layer_post_ffn_rmsnorm(...)` stage
owns the final norm when it executes the last layer. The handle now also owns
the LM-head projection stage through `run_lm_head_logits(...)`, which uses the
resident `lm_head.weight` and persistent logits scratch. Full native
`run_decode_window(...)` still refuses readiness until the model-body loop,
native sampling, token feedback, and multi-step scheduler are implemented
behind the same handle.

## Source-Grounded Execution Contract

TileRT's public DeepSeek runtime does not expose a normal Python-level decoder
forward. `ShowHandsDSALayer` loads per-device parameters and scratch buffers,
calls `dsa_show_hands_prepare_money(...)`, and each decode step calls a backend
op with only the token id. That means the persistent backend owns current
position, cache mutation, sampling state, layer scheduling, temp-buffer reuse,
and token feedback.

The source-compatible Blaise execution contract must preserve TRT-LLM semantics:

1. Convert compressed-tensors NVFP4 aliases from `.weight_packed`,
   `.weight_scale`, `.weight_global_scale`, and optional `.input_global_scale`.
2. Run input RMSNorm, then Blaise input low-rank GatedNorm before attention.
3. Run MLA/DSA projection, sparse index, cache update, and sparse attention.
4. Run Blaise attention output gate before `o_proj`.
5. Run residual add plus post-attention RMSNorm, then post-attention GatedNorm.
6. Run dense MLP for layers 0-2 or MoE route/dispatch/expert/down/combine for
   layers 3-60.
7. Carry residual plus next-layer norm state forward inside the persistent loop.
8. Run final norm, logits, sampling, and token feedback without bouncing through
   Python per layer or per small op.

The immediate implementation target is therefore a TRT-LLM-owned persistent
decode interface that can be fed by the existing Blaise loader and scheduling
metadata. The public TileRT binary can still be used as an ABI probe, but only if
we can provide dummy or converted tensors with Blaise's 128-expert and
index-topk-1024 shapes. If that binary bakes stock 256-expert/topk-2048
assumptions, it is reference material rather than the production path.

## First Milestones

1. **Compatibility gate, no GPU required.** Run
   `python scripts/blaise_tilert_checkpoint_audit.py` to reproduce the current
   model/config/key mismatch table from the checkpoint and TileRT source.

2. **Non-MTP conversion manifest, no GPU required.** Run
   `python scripts/blaise_tilert_non_mtp_manifest.py` to build the full dry-run
   key and shape contract for the 61-layer, non-MTP Blaise checkpoint. The
   optional `--write-json <path>` flag writes the complete source-key manifest.

3. **Stock TileRT sanity, separate from Blaise.** In the official TileRT Docker
   image, run a non-MTP DeepSeek-V3.2 sample with TileRT's expected checkpoint
   layout. This validates the environment and gives a reference profile for the
   execution model. Do not use this as a Blaise result.

4. **Blaise non-MTP loader prototype.** Fork TileRT's DeepSeek model args and
   converter logic for:
   routed experts 128, index top-k 1024, no layer 61, and NVFP4 packed weights.
   The first success criterion is converted key coverage, not generation.

5. **Backend ABI discovery.** With converted dummy or real tensors, test whether
   `libtilert_dsv32.so` accepts the 128-expert/top-k-1024 tensor shapes. If it
   rejects them or corrupts temp layout, stop trying to use the public binary as
   the production path.

6. **Persistent engine design inside TRT-LLM.** If the public binary is fixed to
   stock DeepSeek, define a new TRT-LLM-owned persistent engine interface around
   the existing Blaise model implementation and port TileRT-style scheduling
   principles into that engine.

## Current Branch Position

The branch is `op-trt-tilert`. The local TileRT source checkout used for this
audit is `/home/sjpat/TileRT`. The audit does not require the TileRT wheel,
`torch`, `safetensors`, CUDA, or model conversion.

Current local host limits:

- Host Python does not have `torch` or `safetensors` installed.
- The local TileRT source checkout does not include `libtilert_dsv32.so`; the
  PyPI `tilert==0.1.4` wheel does include both `libtilert_dsv32.so` and
  `libtilert_glm5.so`.
- The original Dynamo prefill/decode deployment was removed from Kubernetes to
  free the B200s for TileRT work. The eight visible B200 GPUs are currently free
  after the probes below.
- A full converted Blaise checkpoint copy is intentionally avoided until the ABI
  and storage target are known, because the source checkpoint is already about
  197 GiB.

## Runtime Probe Results

The repeatable probe entry point is:

```bash
python scripts/blaise_tilert_runtime_probe.py --mode backend-load
python scripts/blaise_tilert_runtime_probe.py --mode dsa-construct
```

On the `optrt-fusionproof-20260609134115` container with `tilert==0.1.4`
installed, backend loading succeeds against NVIDIA PyTorch
`2.11.0a0+eb65b36914.nv26.02` / CUDA `13.1`.

The all-eight-GPU DSA construction probe succeeds with Blaise overrides:

- `n_routed_experts=128`
- `index_topk=1024`
- `block_size=16`
- `max_seq_len=163840`
- `scores_shape=[1, 4, 128]`
- `idx_selects_shape=[1, 4, 1024]`
- `cache_var_count=183` per device
- first cache shapes per layer:
  `[1, 163848, 128]`, `[1, 163848, 512]`, `[1, 163848, 64]`

This proves the public Python-side TileRT DSA object and scratch/cache layout can
represent Blaise's expert count and sparse top-k across eight B200s.

The random-weight backend-step path does **not** work as a proxy for real
generation:

- With `block_size=16`, TileRT random init fails in `weight_dequant` because the
  public FP8 reference path expects block-128 scale geometry.
- With `block_size=128`, random init still fails on `kv_a_proj_with_mqa`
  dequantization because the projection output dimension is not divisible by the
  block-size reshape used by the reference initializer.

Conclusion: do not use `Dsa.init_random_weights()` or public random generation as
the ABI proof. The next meaningful runtime step is a streaming converter or
loader that produces the exact TileRT-format state dict expected by
`Dsa.init_tilert_weights(...)`, starting from the Blaise compressed-tensors NVFP4
checkpoint.

## Streaming Real-Weight Probe Results

`scripts/blaise_tilert_runtime_probe.py --mode blaise-layer-init` now loads real
Blaise checkpoint tensors, converts packed NVFP4 tensors to BF16, requantizes
the tensors TileRT expects as FP8 block-128 weights, and initializes individual
TileRT DSA blocks. The probe patches TileRT's Python reference `weight_dequant`
helper only inside the probe so partial block-128 rows such as the 64-row RoPE
slice and 576-row KV-A projection can be inspected.

Single-layer initialization succeeds for both dense and MoE layers:

- Layer 0 on device 7: 24 state keys, 19 TileRT weight tensors,
  94,636,300 weight bytes.
- Layer 3 on device 7: 26 state keys, 21 TileRT weight tensors,
  759,281,934 weight bytes.

The full streaming path also succeeds through all 61 transformer layers across
all eight B200 GPUs. It initializes layer 0-2 dense blocks and layer 3-60 MoE
blocks from real Blaise tensors without materializing a full converted
checkpoint on disk. Typical timing is about 2.7-3.0 seconds for dense layers and
25-26 seconds per MoE layer.

The initial unpadded full-stream failure boundary was exact:

- All 61 transformer layers initialize.
- Final norm, LM head, embedding, and RoPE frequency tensors initialize.
- Temp vars, continuous temp storage, cache vars, parameter lists, profile logs,
  and P2P pointer buffers initialize on devices 0-7.
- `dsa_show_hands_prepare_money(...)` succeeds and synchronizes on devices 0-7.
- `tilert_init()` and `dsa_show_hands_set_sampling_seed(42)` were also tested.
- The process crashes only when calling final `dsa_show_hands(token)`:
  `dsa_show_hands.cu(313): CUDA error: an illegal memory access was encountered`.

This rules out missing Python-side weight coverage, missing special tensors,
missing P2P pointer exchange, missing prepare step, missing packaged TileRT init,
and missing sampling seed as the immediate issue. The strongest current
hypothesis is that `libtilert_dsv32.so`'s fused `dsa_show_hands` path still has
public DeepSeek-V3.2 backend assumptions that are incompatible with Blaise's
runtime shape contract, especially 128 routed experts and index top-k 1024. The
public Python allocation path can express those shapes, but the final native
engine is the first point that actually executes the baked persistent decode
schedule.

A follow-up public-shape padding probe sharpened this. Padding `IDX_SELECTS` to
top-k 2048, `SCORES` to 256 routed experts, peer `ll_buf` to match the padded
top-k, and converted MoE expert tensors to the public 257 expert slots is enough
to run a real single-token forward through `dsa_show_hands(...)`:

- `dsa_show_hands_prepare_money(...)` succeeds on all eight devices.
- The native forward launches and synchronizes on all eight devices.
- Device 0 reports `token_out_device0: 223`.
- The measured single forward section is about 30 ms in this debug harness.

One tempting padding is explicitly wrong: `IDX_SEL_WS` must stay at
`200 * 1024 + 260 = 205060`; over-padding it to the top-k-2048 analogue causes
`prepare_money` to reject the workspace shape.

The successful debug command used:

```bash
python3 scripts/blaise_tilert_runtime_probe.py \
  --mode blaise-stream-step \
  --num-devices 8 \
  --block-size 128 \
  --max-seq-len 4096 \
  --conversion-device 0 \
  --native-index-topk-pad 2048 \
  --native-routed-experts-pad 256 \
  --native-expert-weight-pad 256
```

The process still exits with a teardown segfault after printing the JSON result,
so this is not yet a clean serving path. It does prove the public TileRT binary
can execute a full real-weight Blaise single-token forward if the public
DeepSeek-V3.2 native shape contract is satisfied at the boundary.

## Batch-Concurrency Contract

The public DeepSeek-V3.2 TileRT native path is batch-one even though the Python
DSA objects expose `max_batch_size` fields. Three probes establish the boundary:

- True native batch, e.g. `--max-batch-size 4 --sweep-concurrencies 4`, reaches
  `dsa_show_hands_prepare_money(...)` and is rejected by the native Q contract:
  `q must be bfloat16 with shape [1, seqlen, 1536]`.
- Packed logical concurrency with `batch=1, seq_len=C` runs only when all cache
  tensors also stay batch-one. Over-allocating cache capacity to `[C, L, ...]`
  first fails on device-0 `partial_buf`, then after forcing that buffer back to
  `[1, L, 7168]`, fails on device-1 `pe_cache must be [1, max_len, 64]`.
- Native `seq_len=1,2,4` works with batch-one caches and gives an upper-bound
  tile-lane measurement, but it is one sequence with multiple positions, not
  independent serving requests.

The installed `libtilert_dsv32.so` strings confirm this is not a single missing
pad. Hot projection, attention, MoE, sampling, top-k, and cache paths contain
batch-one contracts such as `[1, seq_len, ...]`, `[1, L, 512]`, `[1, L, 64]`,
`in_indices must be [1, seq_len]`, and `sampling_seed must be [1, seq_len]`.
The local public TileRT checkout has only Python source and does not include the
referenced native file `src/lib/models/deepseek_v3_2/dsa_show_hands.cu`, so this
batch contract cannot be patched from the public checkout alone.

`scripts/blaise_tilert_runtime_probe.py --mode
blaise-stream-serial-context-sweep` is now a debug-only bridge for c1/c2/c4. It
keeps the native engine batch-one, creates separate logical-request cache
snapshots, swaps each request's cache state into the native cache tensors,
sets that request's `cur_pos`, runs one native forward, and swaps the cache back
out. This makes independent logical contexts work for debugging, but it is
serial and copy-heavy. It is not evidence that the TileRT native engine supports
batched decode throughput.

The first implementation copied every cache tensor in and out for every logical
token, which measured cache-copy overhead rather than TileRT decode. The bridge
now skips cache movement for c1 and, for c2/c4, copies only the live cache prefix
into the native engine and only the newly written token slice back out.

Corrected c1 recovers native TileRT timing: p50 3.62-3.65 ms,
274-276 tok/s/user. The c2/c4 Python bridge remains non-viable as a performance
path: even with prefix copying it must issue thousands of tiny per-cache copy
operations around serial batch-one native forwards, producing only about
33 aggregate tok/s. Treat this only as a debugging scaffold for independent
context semantics, not as a throughput result.

The debug harness also stopped forcing `dsa_show_hands_set_cur_pos(...)` before
every c1 call. TileRT's non-MTP generator lets the native engine own position
advancement for the single live context; per-call `cur_pos` setting is retained
only for swapped multi-context debug runs unless explicitly requested.

The sweep JSON now records `measurement_semantics` and
`serving_concurrency_valid` for each point. The report also hard-rejects legacy
serial-context rows with nonzero `cache_bytes_per_logical_request`, including
the stale 44 tok/s c1 row from the first cache-copy harness. Current expected
meanings:

- `native_single_request`: real c1 public TileRT decode timing.
- `native_batch`: a true native batch attempt; currently rejected above c1 by
  the public batch-one ABI.
- `native_sequence_lanes` or `packed_sequence_lanes`: multiple positions in one
  native request, useful as an upper-bound lane probe but not serving
  concurrency.
- `serial_cache_swap_debug`: independent logical contexts serialized through
  one public native slot with Python cache copies; valid for debugging context
  ownership only, not throughput.

Nsight Systems currently perturbs `prepare_money` enough to trip TileRT's
120-second device-readiness timeout. The probe therefore has a bounded
`--dump-profile-logs` mode that clears TileRT profile-log tensors after warmup
and copies back only selected nonzero rows after measurement.

The first profile-log run did not expose native sections: after clearing the
device-0 profile tensor at measurement start, the c1 forward still measured
p50 3.62 ms / 276 tok/s/user, but the profile tensor had zero nonzero rows.
That means the public binary either does not emit these logs in the end-to-end
path or needs a separate enable switch not present in the public Python API.

A follow-up c1 top-k sweep reused one prepared native state and measured
`top_k=1,8,32,256` with CUDA events on all eight devices. Sampling policy is
not the missing 8 percent to 300 tok/s/user: all four settings measured about
3.65-3.67 ms device time, and all ranks were within about 0.01 ms of each
other. The slower host p50 in that run, about 3.77-3.78 ms, is attributable to
recording events on all devices and should not replace the cleaner device-0
event baseline.

A c1 max-context sweep reused one loaded native state, refreshed RoPE frequency
tensors, reset runtime cache tensors, and re-ran `prepare_money` for
`max_seq_len=1024,2048,4096`. This also is not the missing lever:

- 1024: p50 3.579 ms host, 3.558 ms device0 event, 279.4 tok/s/user.
- 2048: p50 3.573 ms host, 3.551 ms device0 event, 279.9 tok/s/user.
- 4096: p50 3.585 ms host, 3.563 ms device0 event, 278.9 tok/s/user.

So the scalar public TileRT c1 path is effectively context-independent in this
range and remains about 0.22 ms/token slower than the 300 tok/s/user target.

The next plausible path above 300 tok/s/user is multi-token-per-call execution,
not scalar tuning. TileRT's public DeepSeek generator uses an MTP e2e path with
`mtp_seq_len=4`, `dsa_mtp_e2e_show_hands(...)`, accepted-token accounting, and
`NEXT_DRAFT_TOKENS` feedback. The Blaise checkpoint config currently reports
`num_nextn_predict_layers=0`, so this is not a flag flip: TileRT expects an
extra layer at `layer_{n_layers}` containing MTP preprocess weights
(`enorm`, `hnorm`, `eh_proj`), a MoE block, and an RMSNorm/head projection.
A measurement-grade MTP capacity probe therefore needs an explicit dummy or
grafted MTP layer. That can estimate execution-model throughput, but it should
be labeled as untrained/speculative-capacity data unless a trained MTP
checkpoint is supplied.

A grafted-MTP capacity probe now runs end-to-end by reusing layer 60's MLA/MoE
weights as the `layer_61` MTP block and adding dummy preprocess projection
weights. That proves the public MTP ABI can be driven with the Blaise base model
loaded, but it is not a throughput fix:

- `mtp_seq_len=4`, feed `NEXT_DRAFT_TOKENS` back into the next call.
- Accepted tokens across measured calls: `[1, 1, 1, 1]`.
- Host p50: 5.632 ms, device-0 event p50: 5.609 ms.
- Effective accepted throughput: 177 tok/s/user.

The probe was then tightened to mirror TileRT's MTP generator state machine more
closely by running a repeated-token MTP prefill chunk before measured decode:

- `mtp_prefill_len=5`, one prefill call with four valid draft tokens.
- Prefill call time: 30.94 ms.
- Accepted tokens during measured decode still stayed `[1, 1, 1, 1]`.
- Host p50 was 5.653 ms, effective accepted throughput 176 tok/s/user.

So the MTP mechanics are alive, but an untrained/grafted head accepts only the
base token and adds overhead. The acceptance=1 result is not explained by
skipping the MTP prefill state machine. A useful speculative path requires a
real trained MTP/speculator checkpoint. The local Blaise safetensors index
currently has no `model.layers.61`, `mtp`, `draft`, `spec`, `eh_proj`,
`embedding_rmsnorm`, or `hidden_rmsnorm` keys.

The same conclusion holds for the accessible Hugging Face metadata. The current
Blaise repo and `BlaiseAI/corsaire-1-research-preview` both report
`num_nextn_predict_layers=0` and have no MTP/NextN keys in their safetensors
indices. The accessible GLM draft repos also have no TileRT-style MTP keys. So
there is no known trained MTP head available locally or in the currently visible
BlaiseAI model metadata.

## Concurrency Decision

Do not optimize the serial cache-swap bridge. It is useful only to debug
independent cache ownership, and its c2/c4 numbers are dominated by Python cache
copies around a batch-one native slot.

Do not present native `seq_len=2/4` as serving concurrency. It proves the public
engine can execute multiple adjacent positions in one call, but those lanes are
one sequence and share one cache contract.

The valid c2/c4 paths are therefore:

1. Obtain or produce a trained MTP/speculator artifact whose accepted-token mean
   is materially above one. TileRT's own target is around 3 accepted tokens per
   call; the grafted head measured exactly one.
2. Build a TRT-LLM-owned persistent decode engine for Blaise that has a real
   multi-request state contract instead of the public TileRT batch-one ABI.

Until one of those exists, the only valid public TileRT number is scalar c1,
currently about 276-280 tok/s/user.

The repeatable artifact-level target check is:

```bash
python3 scripts/blaise_tilert_result_report.py
```

Current report:

- Best valid scalar/native: 279.87 tok/s/user, p50 3.573 ms.
- Scalar target gap: p50 must be at most 3.333 ms, so the scalar path needs a
  0.240 ms cut, about 6.7 percent of current p50.
- Best grafted MTP capacity: 177.55 tok/s/user, p50 5.632 ms, accepted p50 1.
- MTP target gap: at 5.632 ms, the MTP path needs at least 1.69 accepted tokens
  per call, which means 2 integer accepted tokens. If all four draft tokens were
  accepted, the raw capacity would be about 710 tok/s/user.
- Invalid/debug records excluded from target proof: five serial cache-swap
  records.

## TRT-LLM Implementation Surface

The public TileRT ABI has served its purpose as an execution-model probe. The
TRT-LLM path that can actually satisfy independent concurrency has to attach
below the PyTorch scheduler but above or inside the DeepSeek layer loop:

- `tensorrt_llm/_torch/pyexecutor/model_engine.py`: `ModelEngine.forward(...)`
  is the per-step boundary with `ScheduledRequests`, attention metadata, CUDA
  graphs, speculative metadata, KV managers, and sampling state.
- `tensorrt_llm/_torch/models/modeling_deepseekv3.py`:
  `DeepseekV3ForCausalLM.forward(...)` delegates to `DeepseekV3Model`, whose
  `DeepseekV3DecoderLayer.forward(...)` already contains the Blaise semantic
  ordering: input RMSNorm, input GatedNorm, DSA/MLA, attention-output gate path,
  post-attention norm/GatedNorm, dense MLP or MoE, and residual handoff.
- `tensorrt_llm/_torch/attention_backend/sparse/dsa.py`: this owns the TRT-LLM
  DSA/IndexCache/HISA metadata and cache contracts that production serving
  already uses.

The next implementation should therefore be a TRT-LLM-owned persistent decode
adapter with a real multi-request state contract:

1. Start in decode-only mode, after prefill/KV admission has completed, with
   the same `ScheduledRequests` and `AttentionMetadata` inputs the current
   PyTorch backend receives.
2. Preserve TRT-LLM KV/cache ownership and request slots. Do not copy per-request
   cache state into a single public TileRT slot.
3. Keep layer-loop scratch, routing metadata, DSA metadata, and sampling state
   resident across a short decode-step window.
4. Reuse the current DeepSeek modules as the semantic source of truth first,
   then replace high-frequency stage boundaries with persistent kernels only
   after equivalence and metrics are in place.
5. Prove progress with `blaise_tilert_result_report.py` plus a serving-style
   c1/c2/c4/c8... concurrency sweep that marks every row as valid native
   serving concurrency.

The current branch has started that refactor by moving the resident-window
control loop behind a backend contract in
`tensorrt_llm/_torch/pyexecutor/persistent_decode_engine.py`:

- `PersistentDecodeBackend` is the execution-model boundary. The current
  `PythonResidentDecodeBackend` preserves the existing resident behavior, but it
  is intentionally a bridge, not the target.
- `PersistentDecodeWindowCallbacks` lists the executor services a backend is
  still borrowing: admission, resource prep, `_forward_step`, sampling, request
  state updates, timing, and logging.
- `PersistentDecodeTokenEgress` is the first explicit token-egress boundary.
  Today it flushes deferred samples synchronously at the end of a window. The
  TileRT-style backend needs to replace this with an async side channel so 16/32
  concurrent decode can keep issuing resident work while host-visible tokens are
  drained for streaming/final responses.

This is not yet the proper TileRT execution model. It is the handoff surface for
that work. The next backend must replace the `_forward_step`/`sample_async`
callbacks with a DeepSeek-specific resident layer loop that owns activation
scratch, DSA metadata, MoE routing/dispatch/combine state, sampling, and token
feedback across the decode window.

An initial disabled-by-default planner hook now lives in
`tensorrt_llm/_torch/pyexecutor/persistent_decode_planner.py` and is called from
`PyTorchModelEngine.forward(...)` after `_prepare_inputs(...)` has built
attention metadata. Set `TRTLLM_OPTRT_PERSISTENT_DECODE_PLAN_DEBUG=1` to log
`OPTRT_PERSISTENT_DECODE_PLAN` rows containing the real and padded request
counts, request ids, sequence lengths, cached-token lengths, CUDA graph padding
state, KV manager type, attention metadata type, and the first eligibility
reason that blocks a persistent decode handoff.

The same hook now has an aggregate timing companion in
`tensorrt_llm/_torch/pyexecutor/persistent_decode_profiler.py`. Set
`TRTLLM_OPTRT_PERSISTENT_DECODE_TIMING_DEBUG=1` to build eligibility plans
silently and log `OPTRT_PERSISTENT_DECODE_TIMING` summaries for eligible
decode-only batches. The local diagnostic deployment is:

```bash
python3 .bench_runs_claude/deploy_arm.py persistent_timing_mega_cf
```

First production-path timing measurements on the frozen best-config shape
(`ISL=2055`, `OSL=512`) show:

- `C=1`, two-request probe: serving steady rate 43.42 tok/s/user. Rank-0
  eligible batch size was one. CUDA event time around the graph/model execution
  region was 15.3-15.8 ms per step, or about 63-65 tok/s/user. The Python-side
  enqueue boundary was only about 2.5-2.8 ms; `_prepare_inputs(...)` accounted
  for about 2.1-2.3 ms of that.
- `C=32`, 64-request probe: serving steady rate 49.05 tok/s/user and
  1327 aggregate output tok/s. Rank-0 usually saw batch size eight. CUDA event
  time around graph/model execution was 17.6-18.2 ms per step, or about
  55-57 tok/s/user. `_prepare_inputs(...)` was about 6.4-7.1 ms at this
  per-rank batch shape.

So the current 300 tok/s/user gap is not explained by scheduler eligibility or
NIXL handoff. Eligible decode-only production batches are already reaching the
handoff point with `DSACacheManager`, `DSAtrtllmAttentionMetadata`, and CUDA
graph replay. The large gap is the current TRT-LLM graph/model execution path
itself versus native TileRT's about 3.57 ms scalar step. The first real
persistent-engine bypass therefore has to replace or subsume the DeepSeek layer
execution graph, not just trim logging, request padding, or postprocess code.

Follow-up substage profiling on the same production best-config shape used the
diagnostic image
`optrt-529374445d-persistent-substage-20260622`. This image is intentionally
heavier than the timing-only image because it captures CUDA events inside every
DeepSeek layer, MLA module, and MoE module. It should be used for attribution,
not headline throughput.

Serving rows from the diagnostic image:

- `C=1`, eight requests: 33.82 steady tok/s/user, 29.2 aggregate output tok/s,
  TTFT p50 612 ms, TTFT p95 21487 ms.
- `C=2`, eight requests: 32.19 steady tok/s/user, 65.6 aggregate output tok/s,
  TTFT p50 1239 ms, TTFT p95 1371 ms.
- `C=4`, eight requests: 38.91 steady tok/s/user, 153.7 aggregate output tok/s,
  TTFT p50 863 ms, TTFT p95 1440 ms.

Rank-0 steady decode rows were mostly local batch size one under DP, with CUDA
event time around 22.0 ms per replay step. The broad layer split averaged:

- attention: 11.217 ms
- FFN/MoE: 7.305 ms
- input norm/gate: 0.699 ms
- post-attention norm/gate: 0.973 ms
- post-FFN: 0.516 ms

The new component split shows where the broad buckets come from:

- MLA gate projection / side-stream wait: 7.563 ms
- MLA DSA projection: 4.146 ms
- MLA DSA attention: 2.690 ms
- MLA output projection: 1.785 ms
- MoE expert backend: 5.348 ms
- MoE shared experts: 1.536 ms
- MoE router gate: 0.664 ms
- MoE combine: 0.412 ms

These component timings include overlapped streams, so they should not be added
as a critical-path total. They do identify the first persistent-engine targets:
MLA gate/projection work, DSA projection/attention setup, output projection, and
MoE expert backend/dispatch. The current production path is still fundamentally
a per-layer graph replay path, not the native TileRT scalar execution model.

## Latest Serving Integration Status

The native TileRT probe remains the best proof of the execution-model upside,
not a measurement of the current serving path. The local result report over
`.bench_runs_claude/results/*tilert_blaise*.json` found:

- best valid scalar/native row: 279.87 tok/s/user, p50 3.573 ms, from
  `tilert_blaise_c1_maxseq_sweep_20260622.json`;
- other scalar/native probes were consistent at about 275-280 tok/s/user;
- best MTP capacity probe: 177.55 tok/s/user at p50 5.632 ms, with raw
  all-accepted capacity of 710.19 tok/s/user.

Production serving is still far below that because it is not yet executing the
native TileRT-style resident decode loop. The frozen best config documents
43.30 tok/s/user at `C=1`, 51.07 at `C=32`, and 44.77 at `C=64`. The diagnostic
persistent-stage image is slower because it carries CUDA-event instrumentation,
but it confirms the same qualitative gap.

On 2026-06-22, `persistent_window_mega_cf` using image
`optrt-529374445d-persistent-window2-20260622` was invalidated as a benchmark:
the first `C=1` request and one retry both failed to complete and repeatedly
logged `TxSession ... timed out after 1000ms`, followed by KV cache transfer
termination. The previous `persistent_stage_mega_cf` image
`optrt-529374445d-persistent-window-20260622` was redeployed and completed a
small `C=1` sanity pass: two of two requests completed at 33.87 tok/s/user, with
TTFT p50 21152 ms. That row is not a headline number; it only proves the rollback
restored functional NIXL/KV handoff.

The latest interpretation is therefore:

1. Native TileRT-style scalar execution has been proven locally at about
   3.57 ms/token.
2. Current TRT-LLM serving still runs the per-step/per-layer production graph
   path and is roughly an order of magnitude slower at `C=1`.
3. The next meaningful integration task is to create a serving-side resident
   decode engine that owns the multi-step token-feedback loop after KV admission,
   rather than adding more CUDA graph wrappers or fused-op micro-optimizations.
4. The `window2` KV failure should be treated as a separate instrumentation or
   deployment regression until reproduced on a minimal diff.

## Callback Takeover Proof

The first serving-side persistent-engine takeover branch is now functional, but
it is deliberately a control-boundary proof rather than a performance
implementation. `PersistentDecodeEngine.try_execute(...)` can accept an executor
callback and, when `TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE=1` plus the scalar
decode contract is ready, it returns the callback's `batch_outputs` and
`sample_state` instead of falling back to the split PyExecutor branch.

Diagnostic image:

```text
localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-persistent-engine-callback1-20260622
```

Runtime proof on 2026-06-22 used `persistent_engine_mega_cf`, then restored the
cluster to BEST_CONFIG (`optrt-529374445d-vbfuse-nvfp4-20260615T043037Z`) after
measurement. Decode logs showed repeated reports like:

```text
OPTRT_PERSISTENT_DECODE_ENGINE {... 'eligible': 64, 'contract_ready': 64,
'takeovers': 64, 'takeover_fallbacks': 0, 'batch_hist': {1: 64},
'takeover_enabled': True}
```

Serving rows, `ISL=2055`, `OSL=512`:

```text
  C   reqs  ok   TTFT_p50_ms  tok/s/user  agg_out_tok/s
  1      4   4          3924       42.36          29.3
  2      8   8           779       42.10          85.4
  4      8   8           767       56.16         222.4
```

The interpretation is important: the persistent engine boundary is now active in
serving, but the callback still runs the existing `_forward_step` and
`_sample_async` implementation. This correctly preserves KV/resource
finalization, and it proves the handoff is no longer just passive logging, but
it cannot close the 300 tok/s/user gap. The next performance-bearing step must
replace the callback body with resident execution over a stable decode window,
including generation KV-capacity growth and request/resource update ownership.

## Multi-Step Window Contract

The persistent engine now separates the one-step callback takeover contract from
the resident multi-step window contract. Set
`TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_STEPS=N` to request a future
resident window longer than one step; the report includes
`window_contract_ready`, `window_contract_reasons`, and
`requested_window_steps`.

This is intentionally conservative for the frozen BEST_CONFIG. That config has
attention-DP enabled, and low-concurrency ranks rely on one-step ADP dummy
requests so every rank enters the same collectives. Reusing a real-request
window while dummy ranks fall back would be unsafe. The new contract therefore
blocks multi-step ownership with `attention_dp_enabled` or
`attention_dp_dummy_request` until the resident engine owns dummy
creation/termination for each internal step.

Local validation on 2026-06-22:

```text
python3 -m py_compile \
  tensorrt_llm/_torch/pyexecutor/persistent_decode_engine.py \
  tensorrt_llm/_torch/pyexecutor/py_executor.py \
  tests/unittest/_torch/executor/test_persistent_decode_engine.py

persistent_decode_engine_window_contract_smoke=pass
```

Host `pytest` is still unavailable in this checkout, and direct package imports
hit the known `libpython3.9.so` bootstrap issue; the smoke loaded the module with
a minimal logger shim to exercise the new contract logic. Runtime state was
checked with `k3s kubectl`: the serving pods were still on BEST_CONFIG image
`optrt-529374445d-vbfuse-nvfp4-20260615T043037Z`, not a diagnostic image.

The next implementation target is an ADP-safe window loop:

1. All ranks must collectively agree to enter the window.
2. Ranks without real work must recreate and retire ADP dummy requests for each
   internal token step.
3. Each internal step must run the normal ordering of token feedback,
   `will_complete_next_iteration` marking, `_update_requests`, response flush,
   KV send/update, and V1 `prepare_resources` capacity growth.
4. Only after that is correct should `_forward_step` inside the window be
   replaced by the native TileRT body.

The first default-off executor hook for that loop now exists. It is activated
only when all of the following are set:

```text
TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE=1
TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_STEPS=<N greater than 1>
TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_ADP_WINDOW=1
TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_EXECUTE_WINDOW=1
```

The hook keeps the current overlap ordering: it prepares and samples the next
internal batch from device token feedback, then processes the previously pending
sample through `_update_requests`, first-token/stream response handling,
`_process_previous_batch`, and KV resource update. Internal admission is
collective under attention-DP: every TP/ADP rank allreduces local readiness, and
the group proceeds only when no rank blocks and at least one real request exists
globally. Dummy-only ADP ranks are admitted as `adp_dummy_lane_ready` and recreate
normal one-step dummy requests for each internal step.

This is still not the performance implementation: the internal body calls the
existing `_forward_step` and `_sample_async`, so it can validate executor
ownership but cannot deliver native TileRT token time. The next runtime proof
should be a short `C=1`/`OSL=128` diagnostic with
`TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_ADMISSION_DEBUG=1`, watching for
`OPTRT_PERSISTENT_DECODE_WINDOW_EXECUTED` and verifying no NIXL timeout or ADP
collective hang before attempting longer runs.

## Resident Python Loop Limit

The latest resident-cohort image added async token-egress capture plus substage
timing:

```text
localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-persistent-engine-resident7-egress-timing-20260622
```

Serving rows with `ISL=1041`, `OSL=128`, `WINDOW_STEPS=16`,
`RESIDENT_COHORT=1`, and `ASYNC_TOKEN_EGRESS=1`:

```text
  C   reqs  ok   TTFT_p50_ms  TTFT_p95_ms  tok/s/user  agg_out_tok/s
 16     64  64           832        24817       41.37         224.0
 32     64  64          1021         1923       37.10        1028.6
```

Rank-0 timing over 11 resident windows showed every window completed all 16
requested steps, but the per-window split was:

```text
resident_total_window:          258423 us
resident_forward_step total:     87959 us
resident_sample_async total:      9427 us
resident_materialize_deferred:  155119 us
resident_token_egress_wait:     154893 us
resident_token_egress_apply:       185 us
```

That rules out Python request mutation as the main issue. The bridge is blocked
on the host-visible sampler/token event at the end of each resident window, so
async capture does not create a real side channel. This confirms the current
Python-resident loop is not the TileRT execution model; it is only a diagnostic
boundary.

A follow-up response-boundary egress experiment moved deferred token
materialization out of the resident window when the request contract is simple
enough: single beam, no stop words, no logprobs, no generation logits, no draft
tokens, and no EOS stop handling. Runtime image:

```text
localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-persistent-engine-resident9-response-egress-20260622
```

Serving rows with the same `ISL=1041`, `OSL=128`, `WINDOW_STEPS=16`, and
resident cohort settings:

```text
  C   reqs  ok   TTFT_p50_ms  TTFT_p95_ms  tok/s/user  agg_out_tok/s
 16     64  64           861        24707       39.98         223.9
 32     64  64          1193         1934       34.79         986.7
```

This did engage the intended path: every logged resident window reported
`accumulate_token_egress=True`, `resident_token_egress_wait` disappeared, and
`resident_materialize_response_backlog` was only about 0.5-1.0 ms when present.
Average rank-0 resident-window time dropped from about 258 ms to about 101 ms.
However, the benchmark still regressed because the bridge was not executing a
true 16/32-wide resident serving path. The logged resident cohorts were
seven 4-request windows and four 8-request windows, each owning 16 internal
decode steps. The executor still schedules around those shard-local cohorts and
falls back through the normal serving machinery outside the resident window.

The conclusion is now stronger than the resident7 result: the token-egress sync
was a real local problem, but removing it does not turn the Python callback loop
into TileRT. Further work on this bridge should be limited to correctness
instrumentation. Performance work must move below `_forward_step` into a
DeepSeek resident body that owns the layer loop, sampling/token feedback, and
16/32 request state directly.

The next code boundary is now
`tensorrt_llm/_torch/pyexecutor/persistent_decode_model_backend.py`. It is
default-off and can be enabled with:

```text
TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND=deepseek_resident_v0
TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND_RANKS=0
```

`deepseek_resident_v0` currently declines execution and logs
`OPTRT_PERSISTENT_DECODE_MODEL_BACKEND` rejection reasons. The important change
is placement: the handoff receives the production `ScheduledRequests`, padded
requests, prepared model inputs, attention metadata, KV managers, graph key, and
DeepSeek model object after `_prepare_inputs(...)` but before CUDA graph replay.
It now also validates and caches a `DeepSeekResidentBodyContract` for the loaded
model before declining. That contract records layer count, dense-vs-MoE layer
kinds, attention output gate coverage, low-rank gated norm coverage, KV-A
projection presence, next-layer norm handoff, MoE expert count, and routing
top-k. That is the insertion point for the real resident DeepSeek body; it
should replace the model forward body first, then move sampling/token feedback
into the same resident engine rather than polishing the Python window bridge.

`deepseek_resident_python_v1` is the first executable backend at this lower
boundary:

```text
TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND=deepseek_resident_python_v1
TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND_RANKS=0
```

It is not expected to improve throughput. It still calls the existing DeepSeek
module graph for math, but it owns the post-`_prepare_inputs(...)` body that was
previously hidden inside `_forward_step`: input preprocessing, model-body call,
logit wrapping, and optional gather. The purpose is to make the replacement
boundary executable and testable before substituting a native resident
implementation for `model_forward`.

The next performance-bearing step is therefore specific: replace the
`deepseek_resident_python_v1` model-forward call with a resident DeepSeek decode
body for the production local batch shapes. Under the frozen config, global
`C=16` and `C=32` showed rank-0 resident cohorts of 4 and 8 requests,
respectively, because decode uses TP4/EP4 with attention-DP. The first native
body should target those local `B=4` and `B=8` scalar-decode shapes while
preserving the existing `DSACacheManager` and `DSAtrtllmAttentionMetadata`
contracts.

## Sampling / Feedback Gate

The resident executor window now treats sampling and token feedback as a
separate backend contract instead of an unnamed `_sample_async` callback. The
new callback result type is `PersistentDecodeSampleStepResult`, and the
executor passes a `sample_backend_state()` callback alongside
`model_body_backend_state()`.

This changes the meaning of
`TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_BACKEND=deepseek_native_resident`: the
strict native backend now requires both:

```text
model body backend  = deepseek_resident_native_v1
sample backend      = deepseek_resident_sampler_native_v1
```

The current PyExecutor sampler is exposed only as
`pyexecutor_sampling_bridge_v1`. That bridge still calls the normal
`_sample_async` / TRT-LLM sampler path, so it is not a TileRT execution model.
The strict native backend rejects it by default with a break reason like:

```text
resident_sampling_backend_not_native:pyexecutor_sampling_bridge_v1:resident_sampling_bridge_ready
```

There is an explicit diagnostic override:

```text
TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_ALLOW_SAMPLING_BRIDGE=1
```

Use that only for correctness probes. Throughput runs intended to prove the
TileRT execution model should leave the override unset; otherwise the benchmark
is still timing a resident model-body bridge plus normal PyExecutor sampling.

The first constrained resident sampler hook now exists. The native
`DeepseekResidentDecodeHandle` exposes `run_greedy_sample(logits_scratch,
new_tokens_scratch, input_tokens)`, and the Python resident engine allocates
persistent `[1, batch, 1]` `new_tokens` scratch for each stable decode shape.
When resident logits are produced, the engine writes greedy token ids into that
scratch and reports:

```text
sample backend = deepseek_resident_sampler_native_v1
reason         = resident_sampling_native_executed
```

PyExecutor now prefers `resident_sample_device_new_tokens` over the
`pyexecutor_sampling_bridge_v1` fallback when building the next step's
`SampleState`. Native sample states are request-ordered device snapshots, not
`py_seq_slot`-ordered TRT-LLM decoder buffers. That matters for both token
feedback and egress: internal resident-window samples stay on device and are
copied to host only when deferred token egress is materialized, while the final
resident sample is handled by a constrained resident request updater instead of
calling `TRTLLMSampler.update_requests`.

Strict native windows still require deferred host updates. The current updater
only covers the benchmark contract: non-streaming, single beam, no logprobs, no
generation logits, no stop words, no EOS stop handling, and no speculative
tokens. It appends the native greedy token in request order, increments the
decode iteration, and marks length completion. The resident-window log now also
includes `native_token_stats`, with counters for native sample-state creation,
device-token snapshots, final resident request updates, final-sample deferral,
and deferred device to host materialization. The final native sample is now also
placeholder-updated and enqueued into the same deferred egress path, so it does
not perform a host token read in `_update_requests`. Length completion is still
marked from the placeholder token count; the real sampled token is patched into
the generated-token list when the deferred backlog is materialized before the
non-streaming final response. These counters are the first things to inspect in
a C16/C32 diagnostic run: a real resident-token path should show native sample
states, device-token snapshots, final deferred samples, and resident request
updates, not only `pyexecutor_sampling_bridge_v1`.

This is the right next ownership boundary, but it is not the final TileRT loop
until finish handling, response semantics, and multi-step CUDA ownership are
moved below the executor as well.

## Native Window Backend Gate

The strict native backend now has a third gate for the execution model itself:

```text
window backend = deepseek_resident_window_native_v1
reason         = resident_window_native_ready
```

The DeepSeek native shim currently exports that backend name but reports
`ready=False` with reason `resident_window_native_not_implemented`. In strict
mode, a requested multi-step window therefore stops before any internal
forward/sample calls with:

```text
resident_window_backend_not_ready:deepseek_resident_window_native_v1:resident_window_native_not_implemented
```

If the native window backend reports ready, the strict executor backend still
does not fall through to the generic Python resident loop. It requires the
`resident_window_step` callback, which is the future native owner for the whole
requested window. Without that callback the window stops with:

```text
resident_missing_native_window_step
```

This is the current guardrail that prevents a benchmark from setting
`resident_window_native_ready` in metadata while still timing per-token
`forward_step` / `sample_step` callbacks.

The old PyExecutor per-step loop is exposed only as
`pyexecutor_window_loop_v1`. The strict backend rejects it by default:

```text
resident_window_backend_not_native:pyexecutor_window_loop_v1:resident_window_loop_bridge_ready
```

There is a diagnostic override:

```text
TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_ALLOW_PYTHON_WINDOW_LOOP=1
```

Use that only to debug token feedback and request-state handling. It is not a
valid TileRT C16/C32 benchmark configuration. The production target is a native
window backend that owns the decode-step loop under the resident handle: token
feedback, request-state progression, resource preparation, layer scheduling,
sampling, and deferred token egress for the whole requested window.

## 2026-06-24 Raw-Routing Native Window A/B

The compiled-plan native-window image
`optrt-b22117683eb1-native-window-compiled-plan-libonly-20260624T070620Z`
was re-run with the same strict resident-window settings and
`TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MOE_RAW_ROUTING=1`. This lets the WARPDECODE
MoE runner consume routing logits directly instead of paying the explicit ATen
sigmoid/group-topK/topK route path in `runLayerMoeRouter(...)`.

The first two post-rollout requests were not representative because the
non-window overhead was still settling. Warm requests three and four were
stable:

```text
C=1, ISL=4161, OSL=128, non-stream
raw routing off, warm2: 22.9 s, 5.59 tok/s/user
raw routing on, warm3:  16.5 s, 7.78 tok/s/user
raw routing on, warm4:  16.5 s, 7.76 tok/s/user
```

The C++ window timing also moved in the expected direction. The slow rank on
the raw-routing warm path was about 12.2-12.4 s for 125 decode steps, with
`moe_router_ms` around 280 ms and `moe_experts_ms` around 2.5 s. That is still
nowhere near the scalar public TileRT probe's 3.57 ms/token, but it is a real
execution-model step: routing ownership moved into the resident expert backend
instead of staying as a separate per-layer ATen subgraph. The
`persistent_native_window_mega_cf` deployment arm now defaults
`TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MOE_RAW_ROUTING` to `1`; set it to `0`
explicitly only for rollback A/Bs.

## 2026-06-24 Attention-Projection RMSNorm Native Path

`runLayerDsaAttentionProjection(...)` was updated to run the q-lora and
compressed-kv RMSNorm slices through the resident RMSNorm kernel when the view
contract matches, falling back to the prior ATen `rmsNorm2d(...)` path otherwise.
This removes two per-layer ATen norm calls from the attention projection stage
without changing the q_b projection, k_pe copy, or latent-cache layout.

Stable non-stream OSL=128 measurements after the image was warm:

```text
C=1, ISL=4161: 8.25 tok/s/user, 15.5 s p50 latency
C=2, ISL=4161: 7.72 tok/s/user, 28.5 s p50 latency
C=4, ISL=4161: 7.82 tok/s/user, 16.6 s p50 latency
```

The comparable raw-routing baseline was C1/C2/C4 = 7.74/7.24/7.38 tok/s/user.
The first post-rollout C1 row was not representative because it included peer
setup and first-use compilation. Resident-window timing confirms the local
model-body effect: `attention_projection_ms` fell from the prior roughly 2.8 s
bucket to about 1.5-1.8 s per 125-step window, and the slow rank total moved to
about 11.3-11.5 s. The remaining large resident buckets are MoE experts,
attention tail, indexer projection, attention dispatch, input/post-attention
norms, and the surrounding TRT-LLM serving path.

## 2026-06-25 Clean Native-Window Baseline

The `optrt-b22117683eb1-resident-clean-baseline-20260625111344` image removed
two failed scratch experiments from the prior dirty build: the resident MoE
direct-quant scratch path and the `fp4BlockScaleMoe.cpp` workspace-cache object
experiment. The clean image restored C32 warm parity with the prior resident
native-window baseline:

```text
C=32, ISL=4161, OSL=128: 5.90 tok/s/user, 177.2 aggregate tok/s, 46.2 s wall
```

The same deployment shows the low-concurrency native window remains in the
7-8 tok/s/user family:

```text
C=1, ISL=4161, OSL=128: 7.69 tok/s/user, 8.5 aggregate tok/s
C=2, ISL=4161, OSL=128: 7.64 tok/s/user, 16.5 aggregate tok/s
```

The current endpoint is not missing `native_window_body`,
`native_attention_metadata_refresh`, or `native_moe_experts`: logs report
`resident_window_native_plan_executed` and the C++ timing record is emitted.
The remaining gap is execution-model depth. The native window is a
serving-compatible C++ owner for the window, but inside that owner it still
walks decode steps and layers while launching the existing projection,
attention, MLP/MoE, norm, and sampling sub-ops. A representative C1 timing row
is about 0.8 s for a 15-step owned window, or about 53 ms per decoded token.
The largest C1 per-token buckets are:

```text
MoE experts:                         ~20.8 ms/token
attention projection + dispatch/tail: ~23.2 ms/token
post-FFN norm:                         ~2.9 ms/token
indexer projection:                    ~1.9 ms/token
MoE router:                            ~1.5 ms/token
```

This is why the public TileRT scalar probe's 276-280 tok/s/user number should
not be compared directly with the current TRT-LLM native-window serving number.
The TileRT number is a batch-one public native ABI probe. The serving number is
the current TRT-LLM integration scaffold with full request/KV/sampling
semantics active, but without TileRT's persistent layer schedule.

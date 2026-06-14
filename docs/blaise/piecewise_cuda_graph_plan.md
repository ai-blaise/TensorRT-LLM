# op-trt-hisparse — Piecewise (PCG) & Breakable (BCG) CUDA Graph for the DSA path

Build plan to bring SGLang's Piecewise/Breakable CUDA Graph capability (release
v0.5.13; PR #23351 "Support piecewise CUDA graph with NSA"; BCG ref #25195) to
op-trt-hisparse's DSA path, on branch `op-trt-hisparse-piecewise` (off
`op-trt-hisparse-flashmla` @ `818065cb` = the op-trt-hisparse 001 twin).

## Validated objective + benefit (why this is worth building)

op-trt ALREADY has piecewise CUDA graph (`tensorrt_llm/_torch/compilation/
piecewise_optimizer.py`, a torch.compile FX-split capture) AND full-graph CUDA
capture (`pyexecutor/cuda_graph_runner.py`). They split by batch type:

- **Pure decode** (no context requests) → **full-graph capture** (one graph per
  batch size). Fixed shapes → full capture is STRICTLY BETTER than piecewise.
  **Decode is OUT OF SCOPE — piecewise would regress it.**
- **Prefill / mixed** (`has_ctx_requests`, `model_engine.py:2465`) → **piecewise**.

**The gap (the benefit):** for the DSA model, prefill/context batches currently
run **FULLY EAGER** — `dsa.py:196`: *"in practice this path is always eager —
batches with context requests never run under CUDA graphs ... the unique() output
shape is data-dependent."* This is exactly SGLang's pre-#23351 state. Porting
#23351's approach makes the DSA prefill **piecewise-capturable** → captures the
graph-safe spans of the prefill forward, runs only the 2 incompatible indexer ops
eager → cuts the per-op kernel-launch overhead the eager prefill pays every layer
every chunk → **faster TTFT + higher prefill/mixed-batch throughput under
concurrency** (the metric #23351 benchmarked: concurrency 128, in 1024/out 1024).

This is a PREFILL/TTFT win, orthogonal to the decode tok/s/user campaign. User
greenlit it on that understanding.

## Reference design — SGLang PR #23351 (read directly from the diff)

The fix-set that made NSA piecewise-compatible (to mirror in op-trt):
1. **Un-exclude DSA from piecewise:** removed `DeepseekV32ForCausalLM` /
   `GlmMoeDsaForCausalLM` from `piecewise_cuda_graph_disabled_model_archs` and
   `is_deepseek_dsa(...)` from `is_piecewise_cuda_graph_disabled_model`
   (conditions: not context-parallel, on CUDA).
2. **The split op (the eager piece):** `k_cache_and_topk_result` =
   `@register_custom_op(mutates_args=["topk_result"]) @register_split_op()` —
   bundles `_store_index_k_cache` + `_get_topk_ragged`, runs EAGER between
   captured pieces. `topk_result` is **pre-allocated** by the caller and mutated
   in place (stable address for the graph); padding sliced (`[:extend_num_tokens]`);
   metadata fetched inside the op via `get_forward_context()`.
3. **Custom-op + fake-impl wrappers** so helpers are FX-traceable:
   `logits_head_gate_pcg` (deep_gemm logits gate), `hadamard_transform`,
   `layernorm` (flashinfer) — each `@register_custom_op(fake_impl=...)`.
4. **Strip Dynamo-hostile constructs:** `data_ptr()` compare guarded by
   `torch.compiler.is_compiling()`; avoid `seq_lens_cpu.max().item()` / `len(...)`
   in PCG (host syncs → Dynamo shape guards); fetch indexer metadata inside custom
   ops via `get_forward_context()` (not as forward args → no identity guard that
   changes each replay); force `use_mha=False` in PCG (can't branch on seq_lens);
   CP paths assert `not is_in_piecewise_cuda_graph()`.
5. **forward_context carries `dsa_indexers`** so the eager custom op resolves the
   right indexer by `layer_id`.
6. **radix_attention schema** extended with the MLA/NSA passthrough kwargs
   (`cos_sin_cache`, `is_neox`, `llama_4_scaling`, `topk_indices`) so they appear
   under `--enforce-piecewise-cuda-graph`.

## op-trt grounding — the real machinery to extend (@818065cb)

- **Piecewise engine:** `compilation/piecewise_optimizer.py` (per-`num_tokens`
  Entry, captures a `torch.cuda.CUDAGraph` per FX subgraph between split points;
  falls back to `default_callable` eager when the flag is off / shape unseen).
  `compilation/backend.py` (`enable_piecewise_cuda_graph`). Flags in
  `_torch/utils.py:338-366` (`get/set_piecewise_cuda_graph_flag`,
  `..._per_request_...`). **Find op-trt's analog of `register_split_op`** (how a
  callable is marked a split point) — this is the seam for the indexer ops.
- **Enable path:** `model_engine.py:2465` `can_run_piecewise_cuda_graph =
  has_ctx_requests and ...`; `_filter_piecewise_capture_num_tokens` (capture
  buckets); `TorchCompileConfig.enable_piecewise_cuda_graph`
  (`llmapi/llm_args.py:4292`). Piecewise needs `torch_compile_config` set — P0
  must confirm the hisparse deployment's compile config.
- **The DSA exclusion to remove:** the construct(s) that force "batches with
  context requests never run under CUDA graphs" for DSA (`dsa.py:196` + the
  `is_cuda_graph`/capture flags + the data-dependent `unique()` shape). Find the
  op-trt analog of SGLang's disabled-arch list / `is_deepseek_dsa` guard.
- **The capture-illegal ops (the split points):** the indexer top-k
  (`indexer_topk_decode` / `cute_dsl_indexer_topk_decode`, `dsa.py:4539/4547` per
  the decode map; find the prefill/extend equivalents), the index-K-cache store,
  the logits gate. The `.item()` host syncs (`dsa.py` 470,713,1039-1055,1773,
  1836-1877,1969,2342,2994,3207-3253,3480) — most are in metadata `prepare()`
  (eager, outside capture) but any on the captured forward path must move into the
  eager split op or be removed under `is_compiling()`.

## Build sequence (P0–P5, dependency-ordered, correctness-gated)

- **P0 — Characterize + feasibility (read-only + a baseline run).** Confirm DSA
  prefill runs eager today; locate op-trt's split-op mechanism + the DSA piecewise
  exclusion; enumerate the exact capture-illegal ops on the prefill forward path;
  confirm the compile/piecewise config the hisparse deployment uses. **Feasibility
  gate:** determine how to run a DSA prefill forward on 001 to test — preferably a
  **single DSA decoder-layer forward** at representative prefill shapes (load one
  layer's weights + the indexer; eager vs piecewise) so gating doesn't need the
  full serving stack. If only the full model can exercise it, say so (this is the
  heavier, model-stack-dependent regime — flag it, don't fake a gate).
- **P1 — Un-exclude DSA from piecewise.** Remove the op-trt construct that forces
  DSA context batches eager (the analog of #23351's disabled-list / `is_deepseek_dsa`
  removal), gated on (not-CP, CUDA), behind an off-switch so non-DSA + decode are
  byte-identical.
- **P2 — Make the DSA ops piecewise-compatible.** Register the indexer top-k +
  index-K-cache store as a SPLIT OP (eager), pre-allocate the topk buffer
  (stable address, mutate in place), slice padding; wrap the logits-gate /
  hadamard / layernorm helpers with custom-op fake-impls; strip the Dynamo-hostile
  constructs (`.item()`/`seq_lens_cpu`, `data_ptr()` compare under
  `is_compiling()`, metadata-identity → `get_forward_context()`); carry the
  indexers in the forward context.
- **P3 — Correctness + perf gate.** DSA prefill output **bit-exact / accuracy-
  equal vs the eager path** (TRUE reference = current eager DSA prefill; plus a
  numeric accuracy eval à la #23351's gpqa repeat-8 if the full model is
  runnable). Perf: per-op launch-count + per-step time (nsys) eager-vs-piecewise,
  and — if runnable — TTFT + prefill/mixed throughput under concurrency
  (in 1024/out 1024, concurrency sweep). This quantifies the benefit.
- **P4 — Breakable CUDA Graph (BCG).** Extend to breakable capture (ref #25195,
  DeepSeek V4) for the dynamic-shape conditions a fixed piecewise capture can't
  cover (graph break + resume). Gate: correctness across the dynamic conditions.
- **P5 — Maximal optimization.** Maximize piecewise COVERAGE (shrink the eager
  splits toward only the 2 truly-incompatible ops), tune the capture buckets,
  minimize recompiles, profile. Re-gate.

## Correctness & measurement methodology

- **True reference = the current EAGER DSA prefill output** (never self-compare two
  piecewise variants). Bit-exact / cos ≥ 0.999999 per layer; numeric accuracy eval
  if the full model runs.
- **Perf = the benefit:** eager-vs-piecewise per-op launch count + per-step latency
  (nsys), TTFT, prefill/mixed throughput under concurrency.
- Subagent works on **GPU 1 / `flock /tmp/gpu001_lock_b`**; orchestrator re-gates
  on **GPU 7**. Decode path + non-DSA models must stay **byte-identical** (the
  piecewise enable is DSA-prefill-scoped + off-switchable).

## Constraints / invariants

- **Decode path UNTOUCHED** (keeps full-graph capture). Piecewise change is
  prefill/has-ctx-scoped only.
- ABI-frozen `SparseMlaDecodeKvarnHotOp.cpp` + `hisparseKvarnBdrRead.cuh`
  untouched (this is a Python/runtime + compile-wiring change; expect no .cu edits).
- Off-switchable; non-DSA + decode byte-identical when off.
- Branch `op-trt-hisparse-piecewise` off `818065cb`; twin-history mirror to GitHub.
- Honest gating: if the DSA prefill can only be exercised by the full serving
  stack (not a layer-level harness), surface that as the gating boundary rather
  than claiming an unmeasured pass.

## Risk register

- **Highest risk — gating needs a model forward, not a standalone kernel.** Unlike
  the PDE microbenches, piecewise is a runtime/compile feature. Mitigate with a
  layer-level harness (one DSA decoder layer, real prefill shapes, eager vs
  piecewise); escalate to full-model only if available. If neither is runnable on
  001, the work is correctness-designed but perf-unverified — say so.
- **Dynamo/torch.compile fragility:** stripping the host-sync/`data_ptr`/identity
  constructs is delicate (silent graph breaks, recompiles). #23351 is the exact
  recipe; follow it construct-for-construct.
- **Don't regress decode / non-DSA.** The enable must be tightly scoped + tested
  off (byte-identical).
- **CP not supported under PCG** (per #23351) — assert-guard it; the hisparse decode
  uses CP in the r20 disagg deploy, but that's decode (out of scope); ensure the CP
  prefill path is excluded/guarded.

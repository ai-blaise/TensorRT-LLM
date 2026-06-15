# System-level optimization audit — DeepSeek-V3.2-REAP-345B decode (op-trt-pde-sysopt)

Audited served decode configs:
- `deploy/smcsd_fiport/smc_agg_tp4.yaml` (aggregated TP4/EP4, SMC-SD)
- `deploy/disagg_pd_r20/decode.yaml` (disagg DECODE worker, TP4/EP4, attn-DP, NIXL)

Scope: which SYSTEM-level opts are relevant to decode, and are the relevant ones
completely + correctly wired in the served config/engine. Validated against the
proof image `optrt-...-hisparse-current-head-proof-20260613T155354Z` on GPU 1.

## Completeness table

| Opt | Relevant to decode? | Enabled in prod | Correctly wired | Gap |
|---|---|---|---|---|
| Overlap scheduler | YES | NO — force-disabled at runtime | YES (by design) | doc/config honesty (FIXED) — note 1 |
| MLA gate-overlap (TRTLLM_OPTRT_MLA_GATE_OVERLAP / _gate_proj_chunked) | YES | YES (env default "1") | YES | None — note 2 |
| multi-stream | YES | YES (cudagraph capture) | YES | None — underpins gate-overlap |
| LayerSplit comm overlap | NO (prefill piece) | off (layersplit_enabled:false) | YES | None — note 3 |
| NIXL/UCX KV-transfer | YES (disagg only) | YES (NIXL + PYTHON V2) | YES | agg N/A — note 6 |
| Request-pinning (NIXL req->worker) | YES (disagg only) | YES | YES (3 audits PASS) | None — note 7 |
| nvfp4 GEMM prefetch / cp.async | partial (kernel-internal) | YES (in-kernel) | YES | None — note 8 |
| mla_latent_kv (dense KVarN k2v2 + amortize) | YES | YES | YES | agg via default — note 4 |
| DWDP | NO (prefill/context-phase, TP=1 only) | off (no dwdp_config) | YES | None — note 5 |
| disagg P/D coordination | YES (disagg) | YES | YES | None — note 7 |
| hisparse_enabled (host-tier capacity) | optional | off (default) | YES | separate from HISA indexer — note 9 |

## Empirical validation (proof image, GPU 1, model-free decision-logic probe)

```
SMC.support_overlap_scheduler = True          # SMC does NOT trip the line-402 disable
SMC.has_draft_model           = True
FlashInfer is TrtllmAttention  = False         # => draft decode engine is not overlap-eligible
=> overlap force-disabled for triton/flashinfer SMC draft decode = True   # CONFIRMED
GATE_OVERLAP default-on        = True          # MLA gate-overlap env default on
BLAISE_DEFAULT_MLA_KVARN_DTYPE = kvarn_k2v2    # agg "auto" resolves to same as disagg explicit
resolve_kvarn_config(k2v2)    -> KVarNConfig(non-None)
```

## Notes / mechanism

1. **Overlap scheduler** — both YAMLs set `disable_overlap_scheduler: false`, but it is
   SILENTLY FORCED to True at runtime in `py_executor_creator.py:653-657`:
   `has_draft_model_engine and (not use_chain_drafter or not issubclass(draft_model_engine.attn_backend, TrtllmAttention))`.
   SMC's `support_overlap_scheduler()` returns True (validated), so the line-402 path does
   NOT trip — the disable is purely from the draft engine. `draft_attention_backend: triton`
   → `py_executor_creator.py:605-606` sets draft `attn_backend="FLASHINFER"` + force_triton_prefill
   (triton forces ONLY draft PREFILL; draft DECODE runs FlashInfer because GLM dense GQA has no
   SM100 fused TRTLLM decode kernel — `trtllm_mha` would OOM the O(num_q*max_kv) MHA scratch at
   max_seq_len). FlashInfer ≠ TrtllmAttention → overlap disabled (validated True on GPU 1).
   By-design and unavoidable for this draft shape. The `false`-as-written value is dead.
   FIX (Phase B): annotated both decode YAMLs and corrected `request_pinning.md` line 57-61 so
   operators are not misled into believing scheduler-level overlap hides draft/target latency.
   No runtime value changed (it was already overridden). YAMLs still parse; static audits PASS.

2. **MLA gate-overlap** — `attention.py:3448-3490`. Active iff `gate_proj is not None`
   AND `_GATE_OVERLAP_ENABLED` (env `TRTLLM_OPTRT_MLA_GATE_OVERLAP` default "1"=on, validated)
   AND `do_multi_stream()` AND not compiling. `gate_proj` is created iff
   `config.pretrained_config.attention_output_gate` — and the TARGET model config.json HAS
   `"attention_output_gate": true`, so gate_proj EXISTS for DeepSeek-V3.2-REAP. `do_multi_stream()`
   is True only under CUDA-graph capture (`cuda_graph_runner.py:496` `with_multi_stream(True)`
   spans the warmup + `torch.cuda.graph(...)` capture, so the side-stream/events are baked into
   the graph and replayed). Both decode configs enable the full cuda_graph batch list.
   `_gate_proj_chunked` is bit-exact (two half-N nvjet mm). RELEVANT + ENABLED + WIRED. No gap.

3. **LayerSplit** — `layersplit_enabled: false` in both. It is the PREFILL comm-overlap piece
   (off on decode by design; disagg comment + request_pinning.md R20 prefill gate confirm).
   Inert at cp1+off. Not relevant to decode. No gap.

4. **mla_latent_kv (dense MLA latent KVarN k2v2)** — disagg sets `mla_latent_kv_dtype: kvarn_k2v2`
   + `mla_latent_kv_amortize: true` explicitly. Agg OMITS both. Field default is "auto"
   (llm_args.py:405) → resolved to Blaise default `kvarn_k2v2` in `model_config.py:811-817` for
   DeepSeek DSA, and amortize forced True at `model_config.py:823-828` whenever dtype is kvarn_*.
   So AGG gets the SAME end state via defaults (validated: BLAISE default = kvarn_k2v2). dsa.py:5613
   reads the RESOLVED sparse_attn_config, so both paths converge. Not a functional gap;
   agg-explicitness would be a readability improvement only (left as-is to keep the change minimal).

5. **DWDP** — context/prefill-phase MoE weight-prefetch for DISAGG CONTEXT workers, requires
   TP=1 (`py_executor_creator.py:472-476`; llm_args.py:3334 "accelerates the context (prefill)
   phase"). Both decode configs are TP=4 and set no `dwdp_config`. NOT relevant to decode.
   `_should_enable_dwdp` also requires CuteDslFusedMoE+NVFP4; decode uses WARPDECODE backend.
   Correctly absent.

6. **NIXL/UCX KV-transfer** — disagg `cache_transceiver_config`: `backend: NIXL`,
   `transceiver_runtime: PYTHON` (V2 write-mode/generation-first runtime; the legacy C++
   transceiver only supports completed-prefill metadata), `max_tokens_in_buffer: 131072`.
   `enable_block_reuse: false`, `tokens_per_block: 64`. UCX is an A/B candidate only.
   Correctly wired for disagg. Agg is single-instance (no transceiver). No gap.

7. **Request-pinning + disagg P/D coordination** — three repo audits PASS in this worktree:
   `audit_smc_decode_pinning_static.py`, `audit_cpp_context_endpoint_static.py`, and the
   no-traffic `offline_request_pinning_smoke.py` (all positive + ~11 negative fixtures PASS).
   Binds each generation request to the exact prefill producer (disagg_request_id / ctx_dp_rank /
   ctx_info_endpoint), fail-closed against unpinned ADP broadcast. RELEVANT + WIRED. No gap.

8. **nvfp4 GEMM prefetch / cp.async** — cp.async/TMA prefetch is KERNEL-INTERNAL (cutlass/cute
   pipelines, flashMLA nvfp4 sm100 decode, fp8_blockscale TMA) — ABI-frozen kernel internals,
   not a system-level config toggle. The system-level knob is `nvfp4_gemm_config.allowed_backends`
   = `[cutlass, cublaslt, cuda_core]` in both configs; the PDE work (commit ac71285c) already
   established the op-level backend selection is near-optimal (decode is ~65% inter-kernel
   overhead; adding cutedsl to allowed_backends realizes o_proj 1.14x but that is an op-level
   lever in the PDE lane, not a system-integration gap). No system-integration gap.

9. **hisparse_enabled** — host-tier HiSparse capacity feature; default False; NOT set in any
   served YAML. SEPARATE from the HISA indexer (which IS enabled in both configs via
   `indexer_mode: indexcache-hisa` + `enable_nvfp4_hisa: true` + hisa_* knobs). The
   `validate_hisparse_runtime_config` NIXL+PYTHON+block_reuse=false guard only fires when
   hisparse_enabled=True, so it does not constrain the current configs. No gap.

## Phase B — what was fixed

The only actionable gap was config/doc HONESTY around the overlap scheduler (no functional
mis-wiring found). FIXED:
- `deploy/smcsd_fiport/smc_agg_tp4.yaml`: added comment above `disable_overlap_scheduler` documenting the runtime force-disable by the FlashInfer SMC draft decode.
- `deploy/disagg_pd_r20/decode.yaml`: same annotation.
- `docs/blaise/request_pinning.md`: corrected the "SMC-SD decode is allowed to use overlap" line to state the runtime force-disable and that the `Disable overlap scheduler` log is an expected SMC signal.

No runtime value changed (the field was already overridden to True at runtime). Validation:
both YAMLs parse via PyYAML with values intact; both static pinning audits still PASS; the
GPU-1 decision-logic probe confirms the force-disable, gate-overlap default, and KVarN default.

## Flagged for full-serve / build-gated

- No code/.so changes were made, so no rebuild is required for these annotations.
- The one performance question that requires a full serve to settle (not in scope of this
  config audit): whether a TRTLLM-attention-capable draft (e.g. a GQA shape with an SM100
  fused decode kernel, or a reshaped draft) could re-enable the overlap scheduler on the SMC
  decode path. Today it cannot (GLM GQA OOMs trtllm_mha); this is a draft-model-selection
  question, FLAGGED for a future serve experiment, not a config fix.

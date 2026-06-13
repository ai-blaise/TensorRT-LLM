# op-trt perf work — Lever 1 (glue/quant fusion) + Lever 2 (MoE a2a)

Working notes for the two tractable levers chosen after the 2026-06-13 full-trace
profile (ranked enumeration in `PERF_AUDIT_optrt_2026-06-10.md` §10/§10.1).
Lab: `dynamo-system/topo-c1-dp2tp4-disagg-r20`, model
`DeepSeek-V3.2-REAP-345B-…-NVFP4-NextN-Graft`, serving image
`optrt-19d82b488-fullsource-msgpack-0426`. **No git push to remote.**

**Validation gate (per `docs/blaise/README.md` Validation philosophy +
`optimization_candidates.md` §"Correctness gating"):** kernel-level vs a TRUE
reference — cosine ≥ 0.98 for float activations/logits; **exact** equality for
routing-index/state; throughput parity; zero errors. **Never** output coherence
(untrained checkpoint). Pattern for a standalone numerical test:
`.bench_runs_claude/test_movefuse.cu` (compile in `hisa-buildtools` container,
`nvcc -arch=sm_100`).

Per-iter profile context (c16, GPU ~100% busy, ~19.8 ms/step): dense GEMM swarm
6.46 ms (**parked** — fixed ~14µs/GEMM floor, megakernel work); MoE a2a 2.08 ms
(Lever 2); fusable glue/quant/memset ~3.5 ms (Lever 1).

---

## Lever 1 — glue + quant + memset (~3.5 ms/iter)

### 1b. Quant→norm fusion — HIGHEST CONFIDENCE (the op already exists & is wired)
**Finding:** ~60–65 % of the 392 act-quants/iter sit immediately downstream of an
RMSNorm. A fused **add-RMSNorm→NVFP4** op is already in-tree AND wired into
`RMSNorm.forward` (`rms_norm.py:103-159`, `torch.ops.trtllm.fused_add_rms_norm_quant`),
gated by `is_nvfp4 (quantize_type="nvfp4") + has_residual + nvfp4_scale`. **It is
simply not enabled for V3.2** — every V3.2 `RMSNorm(...)` is built without
`quantize_type` (`modeling_deepseekv3.py:1485/1489`, `attention.py:1393/1441`), so
`is_nvfp4=False` and the fused branch is dead. Other models use it, so the op is proven.

**Plan (enablement, not new kernel):**
- Construct the residual-taking layer norms with `quantize_type="nvfp4"` and attach
  `nvfp4_scale = <downstream consumer>.input_scale` after weight load:
  - `input_layernorm` → `kv_a_proj_with_mqa` (the fused q_a+kv_a GEMM). **NEVER fused today — top target, ×61 layers.** Caveat: `input_layernorm` output also feeds the DSA indexer wk/wp (`dsa.py:2355`); only fuse into kv_a if the indexer shares the same `input_scale` (indexer already asserts a shared wk/wp scale, `dsa.py:2319`); else fuse kv_a only and leave the indexer quant.
  - `post_attention_layernorm` → MoE/shared gate_up: **already fused** on the gated-norm handoff (`_apply_post_attention_gated_norm_quant`, `modeling_deepseekv3.py:1639/1682`) or AR-fusion (`RESIDUAL_RMS_NORM_QUANT_NVFP4`). Only the non-TP/attention-DP decode path falls to the plain norm + separate quant — enable there.
- Consumer already accepts `Fp4QuantizedTensor` and short-circuits its own quant (`linear.py:1391/1399`).
- `q_a_layernorm`→`q_b_proj` (`attention.py:1967→1976`) is a NO-residual norm, so it does
  NOT hit the `has_residual` fused path — would need the non-residual fused-quant variant (check if one exists) or leave it.
- **Expected win:** ~235 quant launches/iter folded → ≈ 0.55–0.66 ms/iter + graph-node reduction.
- **Risk:** numerics (the in-kernel quant must use the same input_scale the Linear used) — gate with a standalone cosine test of fused(add+norm+quant) vs separate(add+norm)→fp4_quantize, ≥0.98 (expect ~1.0 since it's the same arithmetic).
- [ ] STATUS: planned, verified surface

### 1a. Elementwise/copy glue — 1.83 ms/iter, 1073 tiny ops (bit-exact wins)
Ranked targets (all read-confirmed, per-layer ×61 unless noted):
1. **Per-layer MLA scratch `torch.empty` ×4** (`attention.py:2768-2810`: cu_q_seqlens,
   cu_kv_seqlens, fmha_scheduler_counter, fused_q) → ~244 alloc/memset ops/iter.
   Shapes are CUDA-graph-frozen at decode → pre-allocate persistent buffers on the DSA
   metadata (the class already does this pattern, `dsa.py:1424/1431/1440`). **Biggest tiny-op source.**
2. **Fresh `topk_indices_buffer = torch.empty((num_tokens,2048),i32)`** per indexer layer
   (`dsa.py:4205`) + the `[:shape]=-1` memset (`dsa.py:4210`) → reuse the persistent
   `self.topk_indices_buffer` (`dsa.py:1440`), slice `[:num_tokens]`. 61 allocs + memsets.
3. **`latent_cache = torch.concat([compressed_kv, k_pe])`** (`attention.py:1974`, and
   `1842/1866/1897`) — matches the 272µs `CatArrayBatchedCopy`. The `kv_a_proj_with_mqa`
   output is already `[q|compressed_kv|k_pe]` contiguous and `.split()` at `dsa.py:1963`;
   the 576-wide slice `proj_out[..., q_lora_rank:]` IS `[compressed_kv|k_pe]` already → the
   cat re-materializes adjacent data. **Verify** the KV-cache append + attn kernel accept the
   view (no contiguity requirement) before removing. Likely fully removable.
4. **MoE routing casts** `token_selected_experts.to(int32)` + `token_final_scales.to(f32)`
   (`moe_scheduler.py:1038-1039`, also `332/342`) — the fused-comm kernel re-casts to int64
   internally (redundant round-trip); have `DeepSeekV3MoeRoutingMethod` emit the target dtypes. 116 ops/iter.
5. **`indexer_k.to(dtype)`** (`dsa.py:4861`) — only on the non-NVFP4-fused branch; the
   `_FusedWkWpNvfp4` path already avoids it. Make that path default.
- [ ] STATUS: planned; #1+#2 are clean bit-exact buffer-reuse wins (lowest risk)

### 1c. memsetExpertIds fold — 0.58 ms/iter (see Lever 2 #1; folds into the a2a payload)
- [ ] STATUS: see a2a #1 (interleaved-field / postquant path)

---

## Lever 2 — MoE a2a — 2.08 ms/iter (dispatch `<…,…>` 1.36 ms + combine 0.72 ms)

**Algorithm (`cpp/tensorrt_llm/kernels/fusedMoeCommKernels.cu:991`, `.h:429-531`):**
symmetric-memory peer-to-peer **FIFO ring over NVLINK** (MNNVL fabric-mapped per-rank
slabs, alloc `_mnnvl_utils.py:185-331`). One **warp per peer rank**; `grid.z` splits
sender (push tokens into peer's FIFO) vs receiver (pull). Per-token software pipeline:
TMA g2s → pack → (opt) NVFP4 quant → LL128 proto flag → store into peer FIFO → credit
round-trip on head/tail. Template `<FIELD_COUNT, LOW_PRECISION>`: #tensors fused per
transfer, and whether to NVFP4-quantize the payload in shared mem.

**Bound type: LATENCY/SYNC-bound, not bandwidth-bound.** Evidence: uses only **half the
SMs** (`computeMoeCommChannelCount`, `.h:~297`, `smCount/2`); decode payloads tiny (~1
token/rank); cost is FIFO handshake spin-waits (`waitEntryWritable` `.cu:771`, receiver
flag spin `ll128Proto.cuh:33-82`) + `tail` credit round-trips; `FIFO_DEPTH=4` (`.h:251`)
bounds in-flight entries.

**Overlap reality:** the routed path (prepare→dispatch→GEMM→combine) is **serial on one
stream**; multi-stream chunk overlap is **disabled when alltoall is on**
(`moe_scheduler.py:553`). **Shared experts already overlap the entire routed path** on
`aux_stream` (`modeling_deepseekv3.py:1315`) — NOT a free target.

**Ranked opportunities (verified file:line):**
1. **FIFO_DEPTH 4→8** (`fusedMoeCommKernels.h:251`) — more in-flight entries hides the
   credit round-trip latency (the bound). One constant; workspace auto-scales via
   `FIFO_TOTAL_BYTES` (`.h:254`) → check `getFusedMoeCommWorkspaceSize` reflects it +
   symmetric-mem footprint per rank. **LOW risk, cheap, directly targets the bound.** A/B on deploy.
2. **Decode-aware SM/channel count** (`computeMoeCommChannelCount` `.h:~290`, capped
   `smCount/2`) — at decode the GPU is idle during a2a, so more channels shorten the
   per-rank strided loop. Runtime-settable via `setMaxUsableSmCount` (`.h:262`, wired
   `moeCommOp.cpp:138`) — possibly a **config/env knob, no recompile** (verify the python caller).
3. **Reduce dispatch field count / enable postquant-alltoall** (`nvlink_two_sided.py:64`
   `enable_postquant_alltoall`, `:161` field list) — dispatch (1.36 ms = 65 %) moves up to
   4 fields; NVFP4-quantizing the payload + interleaving slots/scales (`isBasicInterleaved`,
   unused, `moeCommOp.cpp:80`) cuts per-token pack/flag/transfer. MED.
4. **Combine on a 3rd stream** overlapping the *next* layer (`moe_scheduler.py:496-504`) —
   MED-HIGH (stream/event + CUDA-graph capture correctness).
5. **Pipeline dispatch↔FC1 across token sub-tiles** (MegaMoE/`FusedCommMoEScheduler`
   `moe_scheduler.py:772`) — HIGH, biggest ceiling, backend-level project.
6. Verify `do_reduce`/`alltoall_result_do_sum` isn't adding a stray `torch.sum`
   (`_mnnvl_utils.py:715`) — for WideEP it's False (fused downstream); just confirm.

**Start: #1 (FIFO_DEPTH) + #2 (SM count)** — cheapest, attack the confirmed latency bound,
bit-exact data movement. Then #3.
- [ ] STATUS: planned; #1+#2 are the cheap first attacks

---

## Findings log
- 2026-06-13: 3 parallel investigations complete (glue / quant / a2a). Crux confirmed:
  fused norm+quant op exists & wired, just disabled for V3.2 (1b = enablement). a2a is
  latency-bound on half the SMs with FIFO_DEPTH=4 (2 cheap C++ knobs). Plans above.
- 2026-06-13 (verification pass — corrections after reading the code directly):
  - **1a #3 (latent_cache cat) is NOT removable.** `compressed_kv` is `kv_a_layernorm(compressed_kv)`
    (`attention.py:1965`) BEFORE the cat (`:1974`) — a fresh normed tensor, no longer adjacent to
    `k_pe`. The cat assembles normed_kv+k_pe; it's real work, not redundancy. (Agent missed the norm.)
  - **1a glue is mostly HOST-side and overlaps GPU at c16.** The profile showed GPU ~100% busy with
    host costs overlapped, so reducing host allocs/casts (1a #1/#2/#4) won't move c16 throughput —
    they help c1 latency + graph-node count, not the throughput A/B. Deprioritized for throughput.
  - **The only real GPU-time Lever-1 win is 1b (quant fusion).** But for V3.2-REAP the input norm is a
    REAP **gated** norm (`_maybe_apply_gated_norm`, `modeling_deepseekv3.py:1721`), and the
    post-attention gated-norm+quant is ALREADY fused (`apply_fused_lowrank_gate_quant_nvfp4`,
    `:1702`). So 1b = extend that fused gated-norm+quant to the INPUT gated norm → kv_a_proj
    (currently un-fused). Multi-step, numerics-gated (cosine ≥0.98). The real win, larger effort.
  - **Lever 2 #2 (SM count) has a hidden tradeoff:** raising the a2a SM budget steals SMs from the
    shared experts that already overlap the a2a on `aux_stream` (`modeling_deepseekv3.py:1315`).
    Net uncertain — NOT a safe bit-exact change. Skipped.

### Implemented
- **[x] Lever 2 #1 — `FIFO_DEPTH` 4→8** (`fusedMoeCommKernels.h:251`, commit `59e909469`).
  Bit-exact; workspace auto-scales (+~148MB/rank @ ep4). At c16 ~tens of sends/peer >> depth 4, so
  it binds. **A/B (throughput + correctness) pending the next rebuild.**

### Next (ranked, for the next rebuild cycles)
1. **Validate FIFO_DEPTH=8** e2e (rebuild → deploy → tight-c16 A/B + numerical/throughput parity).
2. **1b: fuse the INPUT gated-norm+quant → kv_a_proj** (the real Lever-1 GPU win, ~0.5ms). Standalone
   cosine test of the fused gated-norm+quant vs separate first, then wire, then e2e.
3. Lever 2 #3 (dispatch field reduction / postquant-alltoall) — bigger a2a win than the FIFO knob.

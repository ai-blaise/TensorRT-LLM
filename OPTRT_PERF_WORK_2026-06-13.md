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

## MEGAKERNEL THESIS-TEST — NEGATIVE (2026-06-13), and it CORRECTS the §10.1 floor framing
Built a persistent multi-problem NVFP4 GEMM (L-batch via `Sm100BlockScaledPersistentDenseGemmKernel`)
+ microbenched vs separate `nvfp4_gemm` at the real M=16 shapes (`.bench_runs_claude/megakernel/`,
commit `1d3c64529`). Cosine = 1.0 (clean measurement). **Verdict: do NOT pursue the dense-GEMM
megakernel.**
- **The ramp IS amortizable** (mega/cublas climbs with L: kv_a 0.83→1.62x at L=2→32) — premise mechanically correct.
- **BUT cuBLASLt already amortizes its own ramps in a CUDA graph:** 4 GEMMs back-to-back = **51.8µs, not
  4×14=56µs** (marginal per-kernel cost +5.7..+18.5µs, below each isolated floor). **So the "14µs floor
  stacks 315× = 6.5ms recoverable" framing in §10.1 OVERESTIMATED the real baseline** — the floor does NOT
  stack linearly in graph-replay decode; cuBLASLt overlaps ramps. The dense swarm is **much closer to its
  true floor than the isolated microbench suggested.**
- **The megakernel LOSES on the dominant GEMMs:** o_proj (the 22.9µs one) = **0.57x = 1.75x SLOWER**; q_b
  (N=24576) ≤0.71x. cuBLASLt's per-shape small-M tactics beat the generic persistent kernel at large N/K.
  It only wins on small-N (kv_a) at L≥4 — which the model doesn't have (heterogeneous N,K, L=1/layer).
- **Net:** the biggest profiled "lever" (dense GEMM swarm, 6.5ms) is **not recoverable** — it's near floor
  and the megakernel is net-negative. This is the third rigorously-tested lever to come back not-a-win
  (after cumsum fusion + FIFO_DEPTH). Reusable analysis: `.bench_runs_claude/megakernel/thesis_test.py`.

## DEFINITIVE CONCLUSION (2026-06-13) — cheap-win space is exhausted; headroom is structural
After the full profile + all-spots dig + two measured A/Bs, the picture is conclusive:
- **GPU 96.5% busy steady** (gap_analysis), at-floor kernels (GEMM microbench), cross-stream overlap.
- **Cheap levers measured NEUTRAL:** cumsum→moveIndice fusion (§9) and FIFO_DEPTH 4→8 (both ~52 tok/s).
- **a2a already data-optimized:** `TRTLLM_MOE_POST_QUANT_ALLTOALLV=1` by default → dispatch already
  sends NVFP4 (the `moeAllToAllKernel<1,true>` low-precision path). No config win left there.
- **1b (~0.5ms) likely neutral too** (cumsum was) — not worth the graph-surgery risk for THROUGHPUT
  (helps c1 latency / graph-node count only).
- **The ONLY c16-throughput headroom is large structural kernel work:** (A) **megakernel collapse of
  the dense GEMM/BMM swarm (~9ms, biggest)** — fuse per-layer projections into persistent kernels to
  amortize the ~14µs/GEMM fixed floor (the WarpDecode direction, extended to dense MLA); (B) **a2a
  dispatch↔FC1 pipelining / combine side-stream** to overlap the 2ms comm with expert compute
  (MegaMoE/`FusedCommMoEScheduler` is the in-tree vehicle). Both are multi-day kernel projects.

The op-trt stack is at its optimization frontier for cheap/config/fusion levers. Spencer's work has
already captured them; remaining wins need structural megakernel/comm-pipeline engineering.

## Profile revisit (2026-06-13) — GPU bubble analysis (CORRECTION)
Built `.bench_runs_claude/gap_analysis.py` (GPU busy-UNION across streams vs span, per-iter
segmented by cudaGraphLaunch). **Steady-state GPU is 96.5% busy — only ~620µs/iter idle** (3.5%).
My earlier "GPU 100% busy" was right for steady state; the global 87.6%/8-10ms gaps were
inter-round capture artifacts (excluded by median-iter segmentation). So **there is NO recoverable
bubble** — the `cudaEventSynchronize` (5.1ms) and `cudaGraphLaunch` (2.95ms) host costs overlap GPU
work and do NOT idle the GPU. Conclusion REINFORCED: throughput is gated by GPU kernel time;
wins must cut kernels (structural/megakernel) or fuse (1b). The 620µs idle is small
kernel-to-kernel scheduling gaps (launch-overhead, → fewer kernels = megakernel territory).

## All-spots dig (2026-06-13) — status of every profile spot

| spot | µs/iter | dug verdict | action |
|---|---|---|---|
| Dense GEMM swarm | 6460 | at-floor (~14µs/GEMM, all backends; microbench §10.1) | **megakernel** (parked, structural) |
| MLA absorb BMMs | 2390 | same small-M FP4 kernel class as dense GEMMs → same fixed floor | megakernel (parked) |
| MoE a2a | 2080 | latency-bound on FIFO handshake | **FIFO_DEPTH 4→8 DONE** (`59e909469`, building); structural pipelining parked |
| elementwise/copy glue | 1833 | host-overlapped at c16 (GPU 96.5% busy) → won't move throughput; latent_cache cat NOT removable (verified) | deprioritized (c1/graph only) |
| indexer top-k + routing | 1240 | `use_cute_dsl_topk` already on; Spencer-optimized | at-floor (parked) |
| quant storm | 1094 | ~60% norm-adjacent | **1b foundation DONE** (`9295cf895`, inert); wiring next |
| memsetExpertIds | 584 | per-layer recv-tail pad; launch-bound, grid already SM-wide | folds into a2a payload only (structural) |
| GPU idle / bubble | 620 | steady 96.5% busy — NO recoverable bubble (gap_analysis) | n/a — confirms kernel-bound |

**Honest conclusion of the dig:** the stack is genuinely well-optimized (Spencer's work). The big spots
(GEMM 6.5ms, BMMs 2.4ms, a2a-structural) are all **at their kernel-class floor → megakernel/comm-pipeline
projects**, not config wins. The implementable wins are **FIFO_DEPTH** (building, A/B pending) and **1b**
(foundation landed inert; the forward+graph wiring is the focused next step, ~0.5ms). The glue is
host-overlapped (no throughput impact at 96.5% GPU-busy). No quick breadth of wins exists — the headroom
is structural.

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
  Bit-exact; workspace auto-scales (+~148MB/rank @ ep4).
  **A/B RESULT (image `fifodepth8`, tight-c16): 51.97 / 52.05 tok/s/user, ~766 agg — NEUTRAL**
  (baseline 51.6, cumsumfuse 52.08; all within ±0.5 noise). The credit-wait knob did NOT help:
  the a2a's 2.08ms is real GPU transfer work, not credit-stall, at c16. Keep is optional (+148MB/rank
  for no c16 gain; might bind at higher concurrency — trivial revert otherwise).

### KEY EMPIRICAL FINDING (2026-06-13) — cheap levers are throughput-neutral; wins are structural
**Two cheap optimizations now measured NEUTRAL at c16:** cumsum→moveIndice fusion (−0.4ms launches,
§9) and FIFO_DEPTH 4→8 (−credit-latency). Combined with the bubble analysis (steady GPU **96.5%
busy**), this is conclusive: the decode step is **robustly GPU-work-bound with cross-stream overlap
that absorbs small per-kernel removals** (remove an overlapped kernel → the busy-union/critical-path
is unchanged → throughput flat). **Implication for 1b:** the cumsum fusion (0.4ms, overlapped) was
neutral, so the 1b quant fusion (~0.5ms) is **likely also throughput-neutral** unless the quant sits
on the busy-union critical path — its graph-surgery risk is probably NOT worth a throughput bet
(it still helps c1 latency + graph-node count). **The only changes that can move c16 throughput are
those that cut a large chunk of the busy-union GPU work:** megakernel collapse of the dense GEMM/BMM
swarm (~9ms, the biggest), or a structural a2a data/overlap redesign (postquant field reduction +
combine-side-stream / dispatch↔FC1 pipelining). Those are the real (large) projects.

### Next (ranked, for the next rebuild cycles)
1. **Validate FIFO_DEPTH=8** e2e (rebuild → deploy → tight-c16 A/B + numerical/throughput parity).
2. **1b: fuse the INPUT gated-norm+quant → kv_a_proj** (the real Lever-1 GPU win, ~0.5ms).
3. Lever 2 #3 (dispatch field reduction / postquant-alltoall) — bigger a2a win than the FIFO knob.

### 1b — FULL SPEC (scoped 2026-06-13; deliberate cosine-gated cycle, do NOT batch with FIFO_DEPTH)

**Goal:** the INPUT gated norm (`_maybe_apply_gated_norm`, `modeling_deepseekv3.py:1721`) emits a
swizzled `Fp4QuantizedTensor` for `kv_a_proj_with_mqa`, mirroring the ALREADY-SHIPPED dense
post-attention path `_apply_post_attention_gated_norm_quant_dense` (`:1682`, uses
`apply_fused_lowrank_gate_quant_nvfp4_swizzled`). Removes the kv_a_proj input-quant (`linear.py:1432`) ×61.

**Why it's surgery, not a flag:** the gate output feeds BOTH consumers in `forward_dsa_proj`
(`attention.py:1963` kv_a_proj — wants fp4; `:1992` `indexer.pre_indexer_proj(qr, hidden_states,…)` —
wants **bf16** for its own fp32/tf32 wk/wp GEMM). And the proj path is a **CUDA-graph custom op**
`mla_dsa_proj` (`:1035`, `.register_fake` `:1059`). So we must carry both forms (fp4 for kv_a, bf16 for
indexer) into the graph-captured op.

**Edits (verified file:line):**
1. `_resolve_prekv_quant_scale(self)` (new, mirror `_resolve_premlp_quant_scale` `:1665`): return
   `self.self_attn.<…>.kv_a_proj_with_mqa.input_scale` if it `has_nvfp4` + no pre_quant_scale + not
   force-dynamic; else None. (Confirm the attribute path to kv_a_proj from the decoder.)
2. `_apply_input_gated_norm_quant(self, hidden_states)` (new, copy of `_apply_post_attention_gated_norm_quant_dense`
   `:1682-1707` but with `input_gated_norm_down/up` + the prekv scale): returns
   `(y_bf16, Fp4QuantizedTensor swizzled | None)`.
3. `forward` (`:1721`): replace the input `_maybe_apply_gated_norm(...)` with
   `hs_bf16, hs_fp4 = self._apply_input_gated_norm_quant(hidden_states)`; pass BOTH to `self_attn`.
4. **Thread the fp4 into the graph op** — choose the lower-risk of:
   - (A) **metadata stash**: decoder stashes `hs_fp4` (data+sf) on the mla metadata before
     `self_attn`; `forward_dsa_proj` reads it for kv_a_proj, keeps `hidden_states` (bf16) for the
     indexer. Avoids a custom-op schema change; needs the stash to be a graph-stable buffer.
   - (B) **custom-op schema**: add optional `kv_fp4: Tensor?, kv_sf: Tensor?` to `mla_dsa_proj` (`:1035`)
     + its `register_fake` (`:1059`); `forward_dsa_proj` uses them for kv_a_proj. Cleaner data-flow,
     but a schema change on the hot graph op.
5. `forward_dsa_proj` (`:1963`): `kv_a_proj_with_mqa(kv_fp4 if kv_fp4 is not None else hidden_states)`;
   leave `:1992` indexer on bf16 `hidden_states`. (MLA `forward` already types `hidden_states` as
   `Union[torch.Tensor, Fp4QuantizedTensor]`, `:864` — the consumption side is half-ready.)

**Validation:** (1) standalone cosine — `apply_fused_lowrank_gate_quant_nvfp4_swizzled(gate)` →
dequant vs `_maybe_apply_gated_norm(gate)` → `fp4_quantize(input_scale)`, ≥0.98 (expect ~1.0, same
scale). (2) e2e throughput parity + zero errors. **Build/deploy as its OWN image (not batched with
FIFO_DEPTH) for clean attribution.**

**Caveat:** the input gate output also feeds the indexer with a (likely) different input_scale, so a
single fp4 cannot serve both — that's why we keep bf16 for the indexer (it re-quantizes as today). Net
removes only the kv_a_proj quant, not the indexer's.

# Blaise op-trt decode optimization campaign

This directory documents the custom kernels, fused ops, and system-level
optimizations added to `op-trt` (the `ai-blaise/TensorRT-LLM` fork, branch
`op-trt`) to drive **tok/s/user after first token** (decode throughput at
fixed concurrency) on the **DeepSeek-V3.2-Exp / REAP NVFP4** target on
**8×B200**.

The optimization target is the single-node decode step. Decode on this model
is **fixed-overhead-bound**, not bandwidth-bound: at the production topology a
decode token costs ~20.8 ms while the HBM bandwidth floor is ~20–40× lower.
The campaign therefore attacks *overhead* — launch-bound PyTorch op chains,
redundant d2h syncs, recomputation across decode steps, un-fused
elementwise+quant, and the DSA Indexer (which was ~50–74 % of TPOT at campaign
start; the indexer.md wins drove it to ≈ 4 % of the eager c16 profile) —
rather than chasing the bandwidth roofline.

Every piece here is **production code**. Each was validated by *kernel-level*
correctness (top-k set match / partial-O + LSE / numerical cosine vs a torch
reference) and *kernel microsecond / decode-tps* measurement, because the
target model is untrained and end-to-end text is gibberish (so e2e text is not
a correctness signal — see "Validation philosophy" below).

## Index

| # | Piece | Doc | One-line win | Default |
|---|-------|-----|--------------|---------|
| 1 | Indexer: width-correct decode logits buffer | [indexer.md](indexer.md#width-correct-decode-logits) | 2.08–2.30× top-k kernel | opt-in (`seq_len_threshold`) |
| 2 | Indexer: adaptive per-row final-sort | [indexer.md](indexer.md#adaptive-per-row-final-sort) | 1.58× top-k kernel | on (graph-safe) |
| 3 | Indexer: fused cross-step recency-patch op | [indexer.md](indexer.md#fused-cross-step-recency-patch) | 20× (~104 µs → ~5 µs) | on when reuse engaged |
| 4 | Indexer: cross-step IndexCache reuse | [indexer.md](indexer.md#cross-step-indexcache-reuse) | −39/60/68 % TPOT @ stride 2/4/8 | opt-in (`index_topk_freq`) |
| 5 | Indexer: in-graph metadata / sync-free decode | [indexer.md](indexer.md#in-graph-metadata--sync-free-decode) | −48 % decode TPOT (consolidated) | on |
| 6 | Indexer: native C++ / CuTe-DSL top-k dispatch | [indexer.md](indexer.md#native-c--cute-dsl-top-k-dispatch) | C++ ~1.7× @ prod live-kv; DSL ≥ 16k; floor proven | auto by live kv_len |
| 6b | Indexer: fp16 logits | [indexer.md](indexer.md#fp16-indexer-logits) | top-k −15…−22 % @ kv ≥ 33k; buffer halved | on (`auto` → fp16 on DSL path) |
| 7 | Sparse-MLA: MSA 2-stream head-split | [sparse_mla.md](sparse_mla.md#msa-2-stream-head-split) | −14.5…−19.5 % @ b1–4 | on @ small batch |
| 8 | Sparse-MLA: `get_decoding_sched_meta` parallelize | [sparse_mla.md](sparse_mla.md#get_decoding_sched_meta-parallelization) | 2.5–3.3× the meta kernel | on |
| 9 | Sparse-MLA: AB-swapped index-scoring (MSA #3) | [sparse_mla.md](sparse_mla.md#ab-swapped-index-scoring) | already-optimal on tcgen05 | on (tcgen05 path) |
| 10 | KVarN: k2v2/k4v4 dense latent KV quant | [kvarn.md](kvarn.md) | ~2.3 bits @ FP16 accuracy, 3–5× capacity | default k2v2 for production dense MLA |
| 11 | KVarN: BDR fold (in-kernel dequant-on-read) | [kvarn.md](kvarn.md#bdr-fold-in-kernel-dequant-on-read) | amortized restore under budget | default with KVarN |
| 11b | KVarN GQA: SMC-SD generic KV path | [kvarn_gqa.md](kvarn_gqa.md) | safe scaffolding + fused-op-gated reference code | fail-closed by default |
| 12 | WarpDecode: retuned NVFP4 tactics + bridge | [warpdecode.md](warpdecode.md) | 1.20–1.36× vs native MoE | opt-in (env/config) |
| 13 | NVFP4 fusion: add + RMSNorm + quant | [nvfp4_fusions.md](nvfp4_fusions.md#add--rmsnorm--quant-fusion) | −48…−54 % norm→quant sub-path | on (torch.compile) |
| 13b | NVFP4 fusion: shared-expert SwiGLU+FP4-out @ decode M | [nvfp4_fusions.md](nvfp4_fusions.md#shared-expert-swiglu--fp4-output-at-decode-m-guard-lift) | ~100 µs/step + 58 launches | on (guard lifted) |
| 14 | NVFP4 fusion: fused RoPE-cat-FP4 | [nvfp4_fusions.md](nvfp4_fusions.md#fused-rope-cat-fp4) | removes a cat + a quant launch | on when shape matches |
| 15 | NVFP4 fusion: KVarN-BDR fold into add+RMSNorm | [nvfp4_fusions.md](nvfp4_fusions.md#kvarn-bdr-fold) | see KVarN | opt-in |
| 16 | SMC-SD: static-particle speculative decode | [smc_sd.md](smc_sd.md) | draft validated; e2e in progress | opt-in (draft model) |
| 17 | LayerSplit: per-layer CP KV/indexer-K split | [../source/features/layersplit.md](../source/features/layersplit.md) | 21× broadcast latency @ scale | opt-in (`layersplit_enabled`) |
| 18 | Topology + deploy: DP2/TP4 disaggregated decode | [topology_deploy.md](topology_deploy.md) | DP2/TP4 best (64/49/41/40) | deployment choice |
| 19 | tok/s/user optimization candidates (open levers) | [optimization_candidates.md](optimization_candidates.md) | ranked plan: MoE/EP comm (M3), TP-regime KVarN (C2), glue (G2); + shipped cuBLASLt (B1) / lowrank-gate (G1) | living hill-climb plan |

> LayerSplit (17) has its canonical long-form doc at
> `docs/source/features/layersplit.md` (on `op-trt`); WarpDecode
> (12) has its canonical doc + deployment guide under `docs/source/features/`
> that land with the WarpDecode kernel branch. The `docs/blaise/` pages here
> are self-contained campaign-context entries (figures + file map +
> composition) and additionally cover the Indexer, Sparse-MLA, KVarN,
> NVFP4-fusion, SMC-SD, and topology work in full.

## Reading order

1. [topology_deploy.md](topology_deploy.md) — *where* decode runs (DP2/TP4) and
   why decode is overhead-bound. Sets the cost model the rest of the campaign
   optimizes against.
2. [indexer.md](indexer.md) — the campaign's **first big lever** (Indexer was
   50–74 % of TPOT at campaign start, now ≈ 4 % of the eager c16 profile).
   Seven composable wins on the DSA Indexer.
3. [sparse_mla.md](sparse_mla.md) — the sparse-MLA attention kernel that
   consumes the Indexer's top-k (MSA streams + scheduler-meta + scoring).
4. [nvfp4_fusions.md](nvfp4_fusions.md) — the elementwise+quant fusions that
   remove launches and HBM round-trips on the MoE and RoPE paths.
5. [kvarn.md](kvarn.md) — KV-cache capacity (variance-normalized dense MLA latent KV).
6. [kvarn_gqa.md](kvarn_gqa.md) — SMC-SD GQA KVarN KV-cache path and current backend blockers.
7. [request_pinning.md](request_pinning.md) — disaggregated request pinning,
   Moondream overlap gates, and rollout proof points.
8. [warpdecode.md](warpdecode.md) — the MoE decode fast path.
9. [smc_sd.md](smc_sd.md) — speculative decode (multiplies the others).
10. [foundry_iteration_speed.md](foundry_iteration_speed.md) — Foundry vs CRIU snapshot decision for iteration/runtime reuse.
11. [r20_snapshot_composition.md](r20_snapshot_composition.md) — R20 CRIU snapshot readiness and next hook patch.
12. [r20_snapshot_proof_criteria.md](r20_snapshot_proof_criteria.md) — proof gates for TRT-LLM hooks, NIXL, LayerSplit, KVarN/CUDA graph scratch, and checkpointctl.

## Validation philosophy

The DeepSeek-V3.2 target checkpoint used on this fleet is untrained, so
generated text is gibberish and cannot certify a kernel. Every piece is
validated instead by one or more of:

- **Top-k SET match** — the sparse-attention selection (Indexer / scoring)
  must select the same KV positions as the reference (Jaccard = 1.0, or an
  explicitly-bounded recall delta).
- **Partial-O + LSE numerical match** — the attention kernel's partial output
  and log-sum-exp must match the reference within a stated cosine / max-abs.
- **Cosine / max-abs vs torch reference** — for fused elementwise+quant ops,
  the fused result must match the unfused decomposition (often bit-identical on
  the residual; NVFP4 quant error is reported, e.g. 7.06–7.13 %).
- **Kernel µs and decode tps @ c16** — the production concurrency target is
  **c16**; all throughput figures are at c16 unless a sweep is stated.

## Composition

Every piece is designed to **compose** — enabling one must not silently disable
or corrupt another. The composition contract per piece is stated at the bottom
of each doc. The campaign-level composition matrix:

| Piece ↓ composes-with → | Indexer | Sparse-MLA | NVFP4-fusion | KVarN | WarpDecode | LayerSplit | SMC-SD |
|---|---|---|---|---|---|---|---|
| **Indexer (1–6b)** | — | feeds top-k | independent | reads dequantized dense MLA latent; Indexer K remains `fp8`/`fp4` | independent | shares KV pool | per-draft |
| **Sparse-MLA (7–9)** | consumes top-k | — | independent | reads latent | independent | CP-broadcast | per-draft |
| **NVFP4-fusion (13–15)** | independent | independent | — | BDR-fold path | MoE path | independent | per-draft |
| **KVarN (10–11)** | dense MLA latent only; not Indexer K | reads latent | BDR-fold path | — | independent | LayerSplit dense KV pool | per-draft |
| **KVarN GQA (11b)** | not Indexer K | generic GQA KV path; fused-op gated, not production-promoted | independent | separate from MLA KVarN | independent | packed-page + side-state proof pending | SMC-SD target path |
| **WarpDecode (12)** | independent | independent | MoE path | independent | — | orthogonal (MoE vs KV) | per-draft |
| **LayerSplit (17)** | shares KV pool | CP-broadcast | independent | shares pool | orthogonal | — | per-draft |
| **SMC-SD (16)** | per-draft | per-draft | per-draft | per-draft | per-draft | per-draft | — |

"per-draft" = the optimization applies independently to the draft and target
model forwards inside the speculative loop. "independent" = no shared state, no
ordering constraint. The detailed contracts are in each piece's doc.

## Source-of-record

- Campaign main: `op-trt`.
- Push policy: validated work lands on the `ai-blaise` fork only; never
  upstream to `sgl-project` / `NVIDIA`.
- Concurrency target: **c16**.

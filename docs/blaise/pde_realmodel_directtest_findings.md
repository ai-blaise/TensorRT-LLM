# PDE real-model direct-test findings (2026-06-15)

Investigation: do the validated Persistent Decode Engine (PDE) primitives (G0-G9)
add a real win to op-trt's **actual** decode path for the target model
`DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft`?

Method: fast JIT loop in the proof image at the model's **real prod decode shapes**
(index_topk=1024, index_n_heads=64, hisa_block_size=128, block_topk=64,
seq_len up to 132096, decode batch 1-64). No full serve A/B. Harnesses in
`blaise_perf/pde_directtest/` (runner `run_in_image.sh`).

## Results

| test | result |
|---|---|
| v1 naive fused two-level topk | exact (set-match 1.0), but 550 us @ S=132096 (re-reads all S x8 radix passes) |
| v2 SMEM-resident two-level fused | 1.4-3.0x vs cute_dsl op-sequence baseline; exact. **Does NOT transfer** (prod splits the two topks with the candidate-score GEMM) |
| v3 single-level (prod-faithful primitive) | device-resident radix-select **loses** to tuned `cute_dsl_indexer_topk_decode`: 0.55-0.67x @ final(8192->1024), ~0.95-1.26x @ block(1032->64) |
| capture_bench (decode-layer eager vs CUDA-graph) | full-layer capture helps 1.4-2.0x @ small batch (B=1-8), washes out by B=32-64; **cute_dsl IS capturable** |

## Key facts established (from source)

1. **The indexer/topk is only ~4% of TPOT** (06-11 hill-climb). Optimizing it caps at ~1-2% even at 2x.
2. **`cute_dsl_indexer_topk_decode` is CUDA-graph-capturable** (no capture-breaking d2h).
3. The decode runs **torch.compile OFF** (no `torch_compile_config` in decode configs) -> **no piecewise split** -> `CUDAGraphRunner` **full-captures the entire decode step including `mla_dsa_attn_inplace`** (num_tokens = captured batch size is static; the `kv_lens.max().item()` sync was hoisted to `prepare()` precisely to make this capturable).
4. The "excluded from CUDA graph capture" docstring on `mla_dsa_attn_inplace` (attention.py:1127) is its **piecewise split-point** role under torch.compile (prefill / variable-shape path), **not** a decode runtime exclusion.
5. Prod HISA block-scoring is a query-dependent GEMM (`bmm(q, block_mean_reps)` -> ReLU x head-weights), not a simple reduce; the two topks are separated by the candidate-score GEMM (paged_mqa_logits runs only on selected blocks).

## Conclusion

op-trt's real decode **already realizes the PDE's core ideas**: full CUDA-graph
capture, straight-line capturable control flow, hoisted host syncs, and tuned
capturable CuTe DSL kernels. The G0-G9 PDE microbench wins (e.g. G3 5.47x) were
measured against **naive host-orchestration, which op-trt does not use**, so they
are **not additive** on this model. The PDE primitives stand as a validated
toolbox; there is no transferable win at the indexer/topk/capture/sync levels
testable without running the model.

**Unexplored:** where the decode's 20-40x-over-BW-floor overhead actually goes
(MoE/WarpDecode expert dispatch, MLA/FlashMLA, proj GEMMs, TP/EP comms, KV
dequant). Locating that needs profiling the live decode.

## Probe round 2 — other decode angles (2026-06-15)

Broadened beyond the indexer/topk to the decode compute bulk. Harnesses:
`probe_ops.py` (op-surface enumeration, 98 trtllm ops), `gemm_backend_bench.py`.

| angle | finding |
|---|---|
| **NVFP4 dense GEMM backends** | `nvfp4_gemm` dispatcher (~30-42us eager) already beats raw `nvfp4_gemm_cutlass` (~36-47us). Latency FLAT across M=1-64 -> overhead-bound; the ~30us is launch overhead the decode's CUDA-graph capture removes, leaving ~BW-floor (~3us for o_proj). `cute_dsl_nvfp4_gemm_blackwell` is excluded by default "for faster build"; under capture the GEMM is BW-bound so backend choice is sub-us, immaterial to the ~20ms step. No win. |
| **MoE expert reading** | WarpDecode MoE (`warp_decode.py`) uses cute_dsl **gather-grouped-GEMM** (active-only experts, not dense). `DenseGEMMFusedMoE` (all-expert dense) is a separate, non-decode backend. So decode MoE already reads active-only -> no over-read waste. No win. |
| **FP4 format note** | model is `nvfp4_e2m1_ue8m0` -> scale block size 32 (not 16); quantize/GEMM must use scaling_vector_size=32. |

**Overall conclusion (both probe rounds):** op-trt's decode is comprehensively
well-optimized at every directly-testable level (capture, topk, syncs, GEMM
dispatch, active-only MoE). No op-level or structural win is available without the
running model. The only place a real win could still hide is the gap between these
per-op optima and the **realized** end-to-end decode efficiency (the 20-40x-over-
BW-floor figure), or in the data-dependent SMC spec-decode acceptance rate — both
require profiling the live model, not microbenchmarks.

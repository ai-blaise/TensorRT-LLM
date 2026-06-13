# KVarN variance-normalized latent KV-cache quantization

KVarN is a component-level port of **Huawei KVarN** (arXiv:2606.03458) into the
op-trt sparse / MLA KV-cache path. It quantizes the MLA latent KV cache to
**~2.3 bits at FP16-class accuracy**, giving **3–5× KV-cache capacity** (longer
context / larger batch per GPU) at ~0.18 % overhead. On the same MLA latent it
reaches the **highest capacity (fewest bytes/token) of any tested format** at a
fidelity on par with NVFP4.

KVarN is validated by **reconstruction cosine vs an FP16 ground-truth latent**
(not end-to-end text). For supported production dense-MLA lanes it is the
default KV format: `mla_latent_kv_dtype="kvarn_k2v2"` plus amortized restore.
That means **2-bit content latent + 2-bit RoPE key for dense MLA latent KV
only**. Indexer K stays on its own `indexer_k_dtype` path (`fp8`/`fp4`) and is
never routed through KVarN.

| # | Piece | File | Figure | Default |
|---|-------|------|--------|---------|
| 10 | Variance-normalized latent quant (k2v2/k4v4) | `kvarn_core.py`, `kvarn_mla.py`, `kvarn_backend.py` | k2v2 = 2.36 bits/elem @ group64; k4v4 = 314 B/tok, 3.67×, cos 0.99418 | default `kvarn_k2v2` for production dense MLA |
| 11 | BDR fold: amortized low-bit dequant-on-read | `mlaKernels.cu`, `kvarn_backend.py` | 71.7 us fill / 1.12 us steady (INT4 measured; k2 path same packed helper with qmax=3) | default on with KVarN |
| 11b | Delta-restore (C2): O(batch) host-integer pre-replay restore | `model_engine.py`, `kvarn_backend.py` | host 48.8 → 0.3 ms/fire; 12.1 → 0.08 ms/step @ TP bs=16 | default on (`5bc2b2cb8`; `TRTLLM_OPTRT_KVARN_DELTA_RESTORE=0` escape) |
| 11c | Stale-record-on-recycle invalidation | `dsa.py` (`DSACacheManager`), `kvarn_backend.py` (`invalidate_blocks`) | restore bit-equivalent to a never-recycled universe (cos 1.0, all 4 paths) | default on (`34fe7aaec`; `TRTLLM_KVARN_INVALIDATE_ON_FREE=0` escape) |
| 11d | Eager decode-restore host-path (C4) | `dsa.py` (`_kvarn_step_cand_host`) | 13.3 → 0.69 ms/61-layer step (19.4×) steady; 17.0 → 1.7 ms churn | default on (`0a1504755`; auto-fallback to the device scan) |

---

## The method

`tensorrt_llm/_torch/attention_backend/sparse/kvarn_core.py` (pure-torch,
import-light, unit-tests standalone) implements the paper's four-step quant,
verbatim from the paper + Huawei vLLM reference:

1. **Hadamard rotation along `head_dim`** — orthonormal `x @ H` on the
   **channel axis only** (never the token axis, which would cost `group²`
   ops/token). `H` is a normalized Sylvester-Hadamard matrix and is its own
   inverse, so dequant un-rotates with the same matmul. (`hadamard_matrix`,
   `head_dim` must be a power of two.)
2. **Iterative log-domain variance-normalization** (SINQ / Sinkhorn-style) —
   alternating column-std and row-std normalization in log space over the
   `[R, C]` tile, keeping the lowest-imbalance state seen (best-so-far).
   (`variance_normalize_batched`, `_imbalance`.)
3. **Asymmetric per-row RTN at `bits`** (zero-point = row min). (`kvarn_quant_rows`.)
4. **Absorb the per-row RTN scale + zero-point into the matching Sinkhorn axis**
   so dequant is **two multiplies and one add** (the paper's "second scale s2",
   fused so there is **no extra HBM round-trip**). (`kvarn_dequant_rows`.)

**Orientation (KIVI convention).** K tile is `[D, group]` (channels × tokens) →
per-channel RTN rows; V tile is `[group, D]` (tokens × channels) → per-token RTN
rows. For the **MLA latent path** the per-token `compressed_kv` latent (dim =
`kv_lora_rank`) uses the **V orientation** (per-token rows) — the latent is a
learned low-rank projection with no per-channel softmax-exponential sensitivity,
so token-magnitude is the error driver (paper Sec 3.1, Fig 1). The rope-pe key
sub-vector uses the **K orientation** (per-channel). The latent wiring is in
`kvarn_mla.py` (`quant_latent_block`, `dequant_latent_block`,
`packed_bytes_per_block`).

## Reconstruction fidelity vs the dense-KV baselines

Same MLA latent, cosine vs the FP16 ground truth (`bench_kvarn_intel_correctness.py`):

| Format | B/tok | Capacity vs fp16 | cos vs fp16 GT |
|--------|------:|-----------------:|---------------:|
| fp16 | 1152 | 1.00× | 1.00000 |
| fp8-e4m3 | 576 | 2.00× | 0.99965 |
| nvfp4 | 324 | 3.56× | 0.99581 |
| **kvarn_k4v4** | **314** | **3.67×** | **0.99418** (cos_ckv 0.99390, cos_kpe 0.99582) |

KVarN-k4v4 is the **fewest bytes/token** (highest capacity) at a fidelity **on
par with NVFP4** and **1.83× the fp8 capacity**. The amortized restore (below)
preserves this exactly (cos ≥ 1.0 vs the dequant reference,
`test_kvarn_amortize.py`).

- **Config / pool:** `kvarn_backend.py` — `KVarNConfig` (`name`, `latent_dim`,
  `packed_bytes`, `bits_per_elem`), `parse_kvarn_dtype` / `resolve_kvarn_config`
  (so `mla_latent_kv_dtype="kvarn_k4v4"` resolves), and `KVarNLatentPool`
  (`store_block`, `load_block`, `load_blocks`, the serialize/deserialize layout).
- **Enable:** production default — supported DeepSeek/SMC-SD dense-MLA model
  cards resolve to `mla_latent_kv_dtype="kvarn_k2v2"` and
  `mla_latent_kv_amortize=True`. HF configs may also set
  `quantization_config.kvarn.mla_latent_kv_dtype`,
  `quantization_config.kvarn.mla_latent_kv_amortize`, or top-level
  `mla_latent_kv_dtype` / `mla_latent_kv_amortize`.
- **Composes with:** the Indexer / sparse-MLA, which read the **dequantized
  dense MLA latent** — KVarN changes only dense MLA storage. It is not an
  Indexer K-cache dtype; Indexer storage remains `indexer_k_dtype="fp8"` or
  `"fp4"`.

## BDR fold: in-kernel dequant-on-read

**Problem.** A naive KVarN decode dequantizes the working set every step. At the
N = 1024 (batch = 32) operating point the FMHA-equivalent attention compute that
the dequant would overlap is ~131.6 µs/step, but a standalone full-working-set
fused dequant is ~481–489 µs — so at large batch the dequant adds net latency
(143 % of the 341 µs/layer/tok budget) and is **not** hidden by attention.
Worse, a Python double-loop over `B × 32` block-table entries in the restore
path was **2779 µs/step at b32 (815 % of budget)** even when dequant was ~0.

**Insight.** With DSA `index_topk = 2048`, a request's selected KV is **32
immutable blocks**, and a request adds **one new block only every 64 decode
steps**. So the per-step *churn* is tiny — once a committed block is
reconstructed into the FP16 main pool it never needs re-dequanting. The cost
should scale with **churn, not working-set**.

**Fix (two layers, both flag-gated, DEFAULT off):**

1. **Amortized block-id-keyed restore** (`kvarn(system)`): a fully **on-device
   set-diff** computes which committed blocks are new this step (gather
   committed block-ids off the device block table, mask by
   `valid & (restored_gen != commit_gen)`, `torch.unique`); `commit_gen` and the
   per-layer `restored_gen` are device `int64` tensors. Only the churn is
   dequantized. This removes the host double-loop entirely.
2. **In-kernel amortized dequant** (`kvarn(inkernel)`): `kvarn_dequant_amortized_kernel`
   scatters dequanted blocks into a **persistent FP16 pool at their physical
   block-id**, with `grid = churn` not working-set — the zero-round-trip
   end-state.

- **Files:** `tensorrt_llm/_torch/attention_backend/sparse/kvarn_backend.py`
  (`load_blocks`, the device set-diff `kvarn_restore_for_decode`),
  `kvarn_inkernel/kvarn_inkernel_bench.cu`, `kvarn_inkernel/bdr_inkernel_bench.cu`,
  `kvarn_inkernel/bdr_persub_inkernel_validate.cu`,
  `kvarn_inkernel/bdr_longctx_bench.cu`; benches `bench_kvarn_amort_e2e.py`,
  `bench_kvarn_accum.py`; tests `test_kvarn_amortize.py`, `test_kvarn_cycle.py`.
- **Win:**
  - **In-kernel:** un-amortized FUSED 481 µs = 141 % budget → **amortized
    fill-step 71.7 µs = 21 % budget (6.7×)**, **steady-state 1.12 µs = 0.33 %
    budget**. Fill-step stays under the 341 µs/layer/tok budget for churn ≤ ~600
    blocks/step (quantitative floor).
  - **Vectorized device restore (production python path,
    `bench_kvarn_amort_e2e.py`):** all-batch **under budget** —
    b1 642 µs (188 %) → 209 µs (61 %) **3.1×**; b8 780 µs (229 %) → 211 µs
    (62 %) **3.7×**; b32 1816 µs (533 %) → 240 µs (70 %) **7.6×**. ON cost is
    flat ~210–240 µs across batch (per-step loads drop 36 → 8).
  - **FMHA-overlap net-win verdict:** in-kernel removes the 242 % staging
    overhead (**10–53× faster** than python-staged at every batch). At **b ≤ 8**
    the fused dequant is ≤ 47 % budget AND ≤ the overlapping attention →
    effectively free → **NET WIN, +1.93× capacity vs fp8 @ cos 0.994**. At b32
    standalone it adds latency, which is exactly why the amortization is the
    lever (and drives it back under budget).
- **Enable:** default on whenever production dense-MLA KVarN is selected. The
  in-kernel CUDA path drives it to the 71.7 µs / 1.12 µs end-state.
- **Correctness:** `amortize == full-restore` within FP16 batch rounding,
  ground-truth **cos ≥ 1.0**, recycle re-commit re-restores correctly
  (`test_kvarn_amortize.py`, `test_kvarn_cycle.py`).
- **Composes with:** the add+RMSNorm+quant fusion (`nvfp4_fusions.md` #13) — the
  fused dequant can ride the kernel that produces the normed activation
  (zero extra round-trip, the paper's s2 fold); LayerSplit (the KVarN pool is a
  cache pool that LayerSplit can own/broadcast per CP rank).

## Decode-regime restore cost: host-gate + delta-restore + eager host-path (all shipped)

Four follow-ups on the amortized-restore path, tracked in
[optimization_candidates.md](optimization_candidates.md) as C1/C2/C2b/C4:

- **C1 — pre-replay host-gate (shipped `a1b13ea78`):** the pre-replay restore
  scan walked all 61 layer modules before every CUDA-graph replay, paying 3–4
  implicit device syncs per layer before the empty-set early-exit could fire.
  An O(B) host step key `(request_ids, kv_len//tokens_per_block)` proves the
  restore set empty when unchanged and skips the scan entirely.
- **C2 — delta-restore (shipped DEFAULT-ON `5bc2b2cb8`,
  `TRTLLM_OPTRT_KVARN_DELTA_RESTORE=0` escape):** when the step key *does*
  change, the 61-layer masked-select scan walk re-derived far more than the
  delta. The delta path restores only blocks newly full + newly onboarded,
  filtered by the pools' host mirrors; **any uncertainty returns False
  WITHOUT mutation** → full-scan fallback. Verification: 5-scenario lockstep
  equivalence vs the full scan (no-op key, boundary crossing, multi-request
  churn, onboard/free/rewind/recycle, fallback-decline) — **all PASS**
  (non-buffer state byte-exact; restored buffers reference-exact 0 ulp where
  the full scan itself wanders 3–9 ulp from batched-dequant
  nondeterminism). Win: host cost/fire 48.8 → 0.3 ms (162× on the common
  uncommitted fire); at TP bs=16 the amortized pre-replay cost — the single
  biggest TP-regime cost — drops **12.1 → 0.08 ms/step**.
- **C2b — stale-record-on-recycle invalidation (shipped DEFAULT-ON
  `34fe7aaec`, `TRTLLM_KVARN_INVALIDATE_ON_FREE=0` escape):** the C2
  adversarial probes exposed a pre-existing bug in the *full-scan* path:
  `KVarNLatentPool.valid` was never cleared when a paged KV block was freed
  and recycled, `kvarn_commit_full_blocks` skips valid blocks (so the
  recycled id was never re-committed), and the full restore
  (`keep = pool.valid[cand]`, no epoch check) dequantized the OLD owner's
  record over the NEW owner's fresh fp16 latent — stale-KV poisoning on every
  restore fire for that block (scenario d3: cos −0.02 vs the new owner's
  data). The delta path was immune via its host-mirror epoch filter. Fix:
  `KVarNLatentPool.invalidate_blocks()` clears `valid`/`valid_host` and
  resets `restored_gen_host`; `DSACacheManager.free_resources` (ids
  snapshotted before the C++ free) and `rewind_kv_cache` (freed blocks + the
  new tail block) fan out to all local layer pools with one shared device-id
  tensor — restoring the documented "re-committed when block-id is recycled"
  contract. Gates (B200): d3 fixed — gate-ON restore bit-equivalent to a
  never-recycled universe (cos 1.0) on all 4 restore paths; the full a–e
  scenario suite passes on BOTH delta and full-scan paths (worst cos 0.99964
  = the k2v2 quant floor); 7/7 unit tests
  (`test_kvarn_k2v2.py`: free→reuse→commit→restore + fan-out). Cost: decode
  walk byte-identical; 1.46 ms once per request-free at the full 61-pool
  fan-out.
- **C4 — eager decode-restore host-path (shipped DEFAULT-ON `0a1504755`):**
  C1/C2 fixed the *pre-replay* path; the *eager* per-layer
  `kvarn_restore_for_decode` itself still ran its candidate selection
  on-device — boolean-mask indexing that launched **671 ATen cub kernels +
  183 D2H syncs per step-rank** (61 layers × 3 `torch.nonzero`), ~1.24
  ms/step of GPU compaction selecting an **EMPTY set** in steady state.
  In the eager c16 decode profile this slice (~3 % of decode GPU) had been
  **misattributed as "indexer FSSS cub select"**. Fix: host-mirror
  selection (`_kvarn_step_cand_host`) — candidates from
  `kvarn_host_block_table` + host kv_lens, memoized on the step key, numpy
  filter via the same `valid`/`commit_gen` host mirrors the C2 delta walk
  trusts. Empty ⇒ zero launches/syncs; non-empty ⇒ the existing restore
  primitive; **any host-state inconsistency ⇒ unchanged device-scan
  fallback**. Verification: set-equality vs an exact HEAD-path replica,
  **300 randomized trials** (B 1/4/16, churn, duplicates, recycled epochs,
  padding) = 0 failures. Win: standalone 61-layer step **13.3 → 0.69 ms
  (19.4×) steady, 17.0 → 1.7 ms churn**.

## Enabling KVarN

```python
# Dense MLA latent KV dtype selects KVarN; resolve_kvarn_config parses "kvarn_k2v2".
mla_latent_kv_dtype = "kvarn_k2v2"   # production default for supported dense MLA
# Amortized restore is default-on with KVarN:
mla_latent_kv_amortize = True
```

The component-level quant (#10) gives the capacity; the BDR-fold amortized
restore (#11) makes decode affordable at batch by collapsing per-step cost to
churn. Dense-MLA deployment resolves through `ModelConfig.from_pretrained` from
the HF model card or runtime sparse-attention config. Generic/GQA KVarN uses a
separate KV-cache storage/read path and must not be silently mapped through
`mla_latent_kv_dtype`.


## SMC-SD k2v2 deployment contract

`kvarn_k2v2` is the dense-MLA storage mode used by the SMC-SD canary lane. The
runtime contract is:

- **Dense-only:** KVarN applies to the MLA latent cache (`compressed_kv + k_pe`)
  after RoPE/append. It does not change `indexer_k_dtype`, IndexCache, HISA, or
  sparse top-k scoring.
- **Odd draft batches:** packing is along the latent channel axis. SMC verify
  shapes such as `M=25` are valid because neither 2-bit pack/unpack nor the
  side-pool layout requires the row count to be 8-aligned.
- **LayerSplit / disagg:** the KVarN side-pool indexes the same physical dense
  block ids that LayerSplit and cache transfer use. Request pinning must keep a
  request's dense KV block ownership stable across prefill/decode transfer; no
  HELIX fallback is part of this path.
- **WarpDecode:** independent MoE path. Enabling KVarN must not disable or hide
  WarpDecode; failures should be explicit rather than silently falling back.

The low-level BDR CUDA helper in `cpp/tensorrt_llm/kernels/mlaKernels.cu` is
bit-width-aware for 2-bit and 4-bit packed values. The Python side-pool remains
the system integration point for `mla_latent_kv_dtype`; the C++ helper is the
fused-kernel building block for eliminating the staging copy when the host is
available for full CUDA validation.

HiSparse uses a stricter layout contract than the current Python side-pool.
`KVarNConfig.packed_bytes(group)` still describes the legacy
`legacy_sinkhorn_v1` side-pool record used by amortized restore. The production
HiSparse hot-read target is the BDR descriptor
`bdr_ckv_lowbit_fp8_pe_v1`, exposed by `KVarNConfig.hisparse_bdr_layout()`.
For `kvarn_k2v2` with 64-token blocks, that record is 13,312 bytes:
8,192 B low-bit C-KV, 1,024 B C-KV scale/zero-point metadata, and 4,096 B RoPE
byte payload. The descriptor records both the requested `v2` PE setting and the
current 8-bit PE storage used by this BDR hot-record contract. HiSparse must
not feed the legacy side-pool record into a BDR sparse-MLA hot-read kernel;
current code allocates a separate `KVarNBDRSourcePool` for HiSparse-enabled
runs and fails closed until a native BDR writer fills that pool and
`sparse_mla_decode_kvarn_hot` consumes it.

### Focused validation commands

CPU-only shape/config validation (safe on protected hosts):

```bash
python3 -m pytest -q tests/unittest/_torch/attention/sparse/test_kvarn_k2v2.py
```

If the bare VM lacks `torch`/`pytest`, the file can still be syntax-checked with
`python3 -m py_compile tests/unittest/_torch/attention/sparse/test_kvarn_k2v2.py`.

On an isolated/free B200 GPU, run the microbench without touching serving GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/python/bench_kvarn_k2v2_micro.py \
  --device cuda --blocks 128 --group 64 --iters 2 --repeat 20
```

## Correctness validation summary

| Piece | Method | Result |
|-------|--------|--------|
| Variance-normalized quant | cos vs fp16 GT, same latent | 0.99418 (cos_ckv 0.99390, cos_kpe 0.99582), 314 B/tok |
| Amortized / in-kernel restore | amortize vs full-restore | cos ≥ 1.0; per-step loads 36 → 8; recycle correct |
| Delta-restore (C2) | 5-scenario lockstep equivalence vs full scan | all PASS; non-buffer state byte-exact; buffers 0 ulp vs reference |
| Recycle invalidation (C2b) | restore vs never-recycled universe | bit-equivalent (cos 1.0) on all 4 restore paths; a–e suite worst cos 0.99964 (k2v2 floor) |
| Eager host-path (C4) | selected-set equality vs an exact HEAD-path replica | 300 randomized trials (B 1/4/16, churn, duplicates, recycled epochs, padding), 0 failures; inconsistency ⇒ device-scan fallback |

## Composition with the rest of the campaign

- **Indexer / Sparse-MLA**: read the dequantized dense MLA latent; KVarN is
  storage-only for dense MLA KV, transparent to top-k selection and attention,
  and never replaces the Indexer K-cache dtype.
- **NVFP4 fusions** (`nvfp4_fusions.md`): the s2-fold dequant rides the
  add+RMSNorm path; no extra HBM round-trip.
- **LayerSplit** (`../source/features/layersplit.md`): the KVarN latent pool is
  a KV pool LayerSplit can per-CP-rank own and broadcast.
- **WarpDecode**: independent (MoE path vs KV cache).

## Reference

- KVarN paper: arXiv:2606.03458 (Huawei) — Hadamard channel rotation +
  dual-axis Sinkhorn variance-normalization + 2-scale RTN; ~2.3 bits at FP16
  accuracy, 3–5× capacity, ~0.18 % overhead.

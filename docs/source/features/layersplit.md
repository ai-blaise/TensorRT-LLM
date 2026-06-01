# LayerSplit

LayerSplit partitions DSA KV cache + indexer-K cache across context-parallel
ranks so each CP rank owns and stores only a subset of transformer layers'
caches. The owner rank broadcasts its layer's cache to peer CP ranks just
before that layer's attention compute. The design is from z.ai's
[*Scaling Pain of Coding Agent Serving*](https://z.ai/blog/scaling-pain)
§4, motivated by long-context coding-agent workloads where prefill-side
per-rank KV memory pressure dominates GPU utilization.

The op-trt-ls implementation extends z.ai's SGLang reference design with
B200 / TRT-LLM specifics: NVFP4 indexer-K, CuTe kernels via the
[ai-blaise/CZS](https://github.com/ai-blaise/CZS) compiler, integration
with the existing DSA cache manager, and composition with the rest of the
TRT-LLM stack (TP, EP / MoE EP, attention DP, DWDP, cache transceivers).

## Quick start — runtime configuration

LayerSplit is a runtime / system feature. It is **not** enabled from the
HF model card; the only enable path is the runtime `SparseAttentionConfig`
under `--trtllm.sparse_attention_config.*`:

```python
sparse_attention_config = {
    "algorithm": "dsa",
    "indexer_mode": "indexcache-hisa",
    "indexer_k_dtype": "fp4",
    "layersplit_enabled": True,
    "layersplit_owner_assignment": "round_robin",   # or "contiguous"
    "layersplit_transfer_backend": "auto",          # auto | ucx | nixl
    "layersplit_all_cp_ranks_transfer": True,
}
```

Until partial-rank transfer is implemented, `layersplit_all_cp_ranks_transfer`
must remain `true`; the `SparseAttentionConfig` validator rejects
`layersplit_enabled=True` with `layersplit_all_cp_ranks_transfer=False`
to fail fast at construction.

## Owner-assignment policies

`compute_owner_assignment(num_layers, cp_size, policy)` in
`tensorrt_llm/_torch/attention_backend/sparse/layersplit.py` decides which
CP rank owns which layer:

- `round_robin`: `owner_map[L] = L mod cp_size`. Matches the SGLang op-ls
  reference implementation and the legacy `layout="interleaved"` alias.
  Spreads heterogeneous layer cost (dense MLA vs MoE) across CP ranks at
  the granularity of every layer.
- `contiguous`: each rank owns a contiguous block of layers; the
  remainder is distributed to the lowest-rank owners. Better for some
  comm patterns (windowed KV refresh, prefix-cache hand-off).

Edge cases:

- `cp_size == 1` collapses to "all layers on rank 0" regardless of policy.
- `cp_size > num_layers` leaves the high-rank owners idle (zero layers
  owned) instead of failing.
- `num_layers % cp_size != 0` is the common case (e.g. DeepSeek-V3.2's 61
  layers on `cp_size=8`).

`LayerSplitOwnership` is a frozen dataclass — hashable, CUDA-graph stable,
safe to share across decoding steps within the same model load.

## Implementation milestones (commit chain on `op-trt-ls`)

| Milestone | Commit       | Description                                                           |
|-----------|--------------|-----------------------------------------------------------------------|
| M1        | `0cb9b54b`   | Remove LayerSplit HF model-card side-door (runtime-only contract).    |
| M2        | `bb27f238`   | `LayerSplitOwnership` + `compute_owner_assignment` policy module.     |
| M3        | `eff03f64`   | `DSACacheManager` constructs with `layersplit_enabled=True`; LayerSplitRuntimeState + comm stream + transfer-backend selection. Replicated allocation. |
| M4        | `e3543a1f`   | Owner-local DSA cache allocation via `build_layersplit_layer_mask` + the existing `layer_mask` plumbing into `_create_kv_cache_manager`. ~87% per-rank memory reduction at `cp_size=8`. |
| M5        | `b06865a4`   | Per-layer broadcast scaffold: `maybe_broadcast_for_layer` + Indexer.forward hook (`payload=None`, sync-shape only). |
| M5b       | `f6c21223`   | `cp_group_pg` resolution from `mapping`; multi-process NCCL integration test on real GPUs (NVLS-disabled to coexist with sibling NCCL tenants). |
| M5c       | `2b3e12b3`   | Heartbeat payload (16 B per-layer broadcast) activates the real NCCL broadcast in the production hook. Surfaces CP-comm config errors at first decode step. |
| M7        | `675c3045`   | CZS-proved stage-for-broadcast kernel scaffold + reference torch implementation; 12/12 CZS obligations proved. |

## Queued work

| Milestone | Description                                                                                                  | Blocker                                                                                                        |
|-----------|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| M5d       | Real active-KV slice broadcast: owner publishes the layer-L active blocks; non-owner attention reads from broadcast recv buffer instead of (M4-truncated) cache pool. | Attention-backend KV-source override refactor; exact-token correctness validation at CP=2.                     |
| M6        | Cross-layer overlap: broadcast L+1 on `comm_stream` overlaps with indexer / sparse-attn compute on the default stream for layer L. (z.ai blog Fig 4(b).) | Depends on M5d real payload (heartbeat is too small for overlap to matter).                                    |
| M7b       | Lift the stage-for-broadcast scaffold into a real `@cute.kernel @cute.jit` body; bench vs the C++ `indexer_k_cache_scatter_op` reference. | None; scaffold + CZS proof already shipped, scope is the @cute.kernel body itself.                              |
| M8 vectors | Direct NVFP4-on-the-wire broadcast; indexer-cache-first 2-phase per-layer protocol; persistent cross-layer broadcast scheduler CTA; fused HISA-block-select + LayerSplit-owner-stage; owner-aware EPLB; owner-local indexer cache compaction; UCX/NIXL fast-path for disagg-PD. | M5d for most; M5d gives the baseline against which each vector is measured. |
| M9        | IKP-driven optimization loop per kernel.                                                                     | M5d / M6 baseline (need real bandwidth measurements before optimizing).                                         |
| M10       | Production gate (validation matrix per cells below) + push.                                                  | All previous milestones.                                                                                        |

## Validation matrix (M10 gate)

- Owner-map policies: `round_robin`, `contiguous`.
- CP sizes: 2, 4, 8.
- Parallelism composition: `TP=8`, `EP=8`, attention-DP, CP / DWDP, disagg-PD.
- CUDA graph capture / replay at the production decode buckets:
  `c1, c2, c4, c8, c16, c32` (per the op-trt c32 production profile).
- Exact-token end-to-end correctness across input lengths `1k, 8k, 16k, 32k,
  100k, 128k`.
- Long-context prefix-hit benchmark @ 90 % cache hit, inputs `40k, 60k, 80k,
  100k, 120k`. Compare against:
  - No-LayerSplit fallback (same hardware, same model).
  - SGLang op-ls reference numbers (CP=8 wall ratios `7.68×, 8.45×, 8.42×`
    at 40 / 60 / 80 k; TTFT speedups `6.83×, 7.34×, 7.79×`).

## Architecture overview

The data plane is built around an owner-broadcast pattern per the z.ai
blog §4:

```
                          (only owner has dense KV/indexer-K storage)
                          (M4 layer_mask cuts non-owner allocation)

   Decode step at layer L:                            comm_stream
   ┌──────────────────────────────┐         ┌────────────────────────┐
   │ owner CP rank: write new     │         │ owner: publish layer-L │
   │ tokens' KV into its layer-L  │  ──────▶│ active-KV slice via    │
   │ cache via indexer_k_cache_   │         │ nccl_broadcast(src=    │
   │ scatter_op                   │         │ owner, group=cp_grp)   │
   └──────────────────────────────┘         └────────────────────────┘
                                                       │
                                                       ▼
                                            non-owner ranks receive
                                            into transient buffer

   ┌──────────────────────────────┐
   │ all ranks: run indexer +     │  ◀── owner reads cache directly,
   │ sparse-attn over layer L's   │      non-owners read from buffer
   │ KV                           │      (M5d KV-source override)
   └──────────────────────────────┘
```

The indexer-K cache is approximately one-eighth the size of the dense KV
cache (z.ai blog §4). M8b will split the broadcast into a 2-phase
per-layer protocol: phase 1 broadcasts the small indexer-K (completes
well before attention); phase 2 broadcasts the dense KV. The receiver
can start indexer compute as soon as phase 1 lands, overlapping phase 2
with sparse-attention compute. This is strictly stronger than M6's
cross-layer overlap because it pipelines INSIDE a layer, not just
across layers.

## Code layout

| Path                                                                                                                | Purpose                                                                  |
|---------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| `tensorrt_llm/_torch/attention_backend/sparse/layersplit.py`                                                        | Policy + runtime state + broadcast method + layer-mask helper.           |
| `tensorrt_llm/_torch/attention_backend/sparse/dsa.py`                                                               | Indexer.forward hook + DSACacheManager LayerSplit state construction.    |
| `tensorrt_llm/_torch/pyexecutor/_util.py`                                                                           | `_create_kv_cache_manager` calls `build_layersplit_layer_mask`.          |
| `tensorrt_llm/llmapi/llm_args.py`                                                                                   | `DeepSeekSparseAttentionConfig.layersplit_*` fields + validator.         |
| `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/layersplit_stage_for_broadcast.py`                                  | CuTe DSL stage-for-broadcast kernel surface + reference torch impl.      |
| `docs/proofs/layersplit_stage_for_broadcast_czs_module.json`                                                        | CZS proof for the stage-for-broadcast kernel (12/12 Proved).             |
| `tests/unittest/_torch/test_layersplit_ownership.py`                                                                | 39 unit tests for ownership / runtime state / broadcast logic.           |
| `tests/unittest/_torch/test_layersplit_stage_for_broadcast.py`                                                      | 14 tests for stage-for-broadcast layout + reference correctness.         |
| `tests/unittest/_torch/test_layersplit_multiproc_nccl.py` + `run_layersplit_multiproc_nccl.py`                      | Real-NCCL CP=2 integration test for `maybe_broadcast_for_layer`.         |

## Running the tests

Unit tests (CPU-friendly via direct importlib runner — bypasses the
heavy `tensorrt_llm.__init__` chain):

```bash
python3 -c "
import sys, importlib.util, inspect
spec = importlib.util.spec_from_file_location(
    'tensorrt_llm._torch.attention_backend.sparse.layersplit',
    'tensorrt_llm/_torch/attention_backend/sparse/layersplit.py')
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

ts = importlib.util.spec_from_file_location(
    'ts', 'tests/unittest/_torch/test_layersplit_ownership.py')
tm = importlib.util.module_from_spec(ts)
ts.loader.exec_module(tm)
for name, fn in inspect.getmembers(tm, inspect.isfunction):
    if name.startswith('test_'):
        fn()
        print('OK', name)
"
```

Multi-process NCCL integration test (needs ≥2 CUDA devices; set
`NCCL_NVLS_ENABLE=0` if another NCCL tenant on the same node holds
NVSwitch resources):

```bash
NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=<gpu_a>,<gpu_b> \
    python3 tests/unittest/_torch/run_layersplit_multiproc_nccl.py
```

CZS proof:

```bash
~/work/CZS/build/src/czs prove --json \
    docs/proofs/layersplit_stage_for_broadcast_czs_module.json
```

## Composition with other op-trt features

- **TP / EP / MoE EP / attention DP**: LayerSplit composes through the
  existing `Mapping` object. `mapping.cp_size` and `mapping.cp_rank`
  determine the owner table; the parallelism modes that are not CP-aware
  (TP, EP, ADP) are unaffected and the runtime broadcast happens on the
  CP sub-group only.
- **CUDA graph capture / replay**: the comm stream is created at model
  load time so it stays graph-stable. The owner-map is computed once
  and stored on `LayerSplitRuntimeState`. The heartbeat payload is
  allocated lazily once and cached. M5d's transient broadcast buffer
  must be sized to the worst-case at metadata setup time so its address
  is graph-stable.
- **Cache transceiver (UCX / NIXL)**: M5b binds `cp_group_pg` for the
  same-node NCCL fast path. M8g will add UCX / NIXL fast-paths for
  disaggregated-PD deployments where prefill and decode live in
  different pods and the broadcast must cross the fabric.
- **WarpDecode (MoE decode overlay)**: orthogonal; WarpDecode operates
  on the MoE expert dispatch / routing path, LayerSplit on the DSA KV
  cache. Both compose with CP independently.

## References

- [z.ai *Scaling Pain* blog](https://z.ai/blog/scaling-pain) — §4 LayerSplit design + Fig 4 (data flow) + Fig 5 (throughput improvement).
- [ai-blaise/CZS](https://github.com/ai-blaise/CZS) — CuTe Z3 SMT solver used for the stage-for-broadcast proof.
- [yao-jz/intra-kernel-profiler](https://github.com/yao-jz/intra-kernel-profiler) — used for IKP-driven optimization (M9).
- [aturker1/cutest](https://github.com/aturker1/cutest) — used for elementwise fusion validation.
- NVIDIA CuTe DSL docs: <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html>
- CuTe paper: <https://arxiv.org/pdf/2603.02298>
- veitner CuTe blog: <https://veitner.bearblog.dev/blog/>

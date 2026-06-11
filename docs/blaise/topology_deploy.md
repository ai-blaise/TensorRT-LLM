# Topology + deployment: DP2/TP4 disaggregated decode

This page documents *where* the optimized decode step runs on 8×B200 and *why*
decode is overhead-bound — the cost model the rest of the campaign optimizes
against.

## The decode cost model: overhead-bound, not bandwidth-bound

A single decode token on the DeepSeek-V3.2-REAP NVFP4 target costs **~20.8 ms**
at the production topology, while the HBM **bandwidth floor** for the same
token's weight + KV reads is **20–40× lower**. Decode is therefore
**fixed-overhead-bound**: the time is dominated by launch latency, per-step
Python/host work, redundant d2h syncs, recomputation across steps, and the DSA
Indexer (50–74 % of TPOT at campaign start; since driven to ≈ 1 % — see
`indexer.md`) — *not* by the memory roofline.

This is the central fact that motivates the whole campaign. Every win attacks
overhead:

- Indexer (`indexer.md`): cross-step reuse (don't recompute top-k every step),
  fused recency-patch (20×: collapse ~20 launches → 1), in-graph metadata
  (remove ~29 d2h syncs), adaptive sort (stop paying static-width tax).
- Sparse-MLA (`sparse_mla.md`): 2-stream overlap, parallel scheduler-meta.
- NVFP4 fusions (`nvfp4_fusions.md`): remove standalone quant launches + HBM
  round-trips.
- WarpDecode (`warpdecode.md`): a graph-shaped MoE path that avoids the padded
  grouped-MoE launches.
- KVarN (`kvarn.md`): more KV capacity so larger batch amortizes the fixed
  overhead across more tokens.

Because decode is overhead-bound, **CUDA-graph capture/replay** is load-bearing:
it is what removes the per-step host launch cost. Every campaign piece is
designed to be graph-safe so it stays inside the captured decode graph.

## Topology sweep: tok/s/user after first token (8×B200)

The metric is **tok/s/user after first token** (decode throughput per user at
fixed concurrency); the four columns are a concurrency sweep. Measured on the
single-node 8×B200 op-trt/Dynamo decode deploy:

| Topology | Description | (c-sweep) tok/s/user | Verdict |
|----------|-------------|----------------------|---------|
| **DP2 / TP4** | 2 attention-DP groups × TP4 each | **64 / 49 / 41 / 40** | **winner** |
| DP4 / TP2 | 4 attention-DP groups × TP2 each | 57 / 45 / 37 / 36 | 2nd |
| TP8 | single TP8 group | 68 / 52 / 38 / 34 | best at lowest concurrency, worst tail |

**DP2/TP4 is the production choice.** TP8 wins at the very lowest concurrency
(68 vs 64) because a single 8-way tensor-parallel group has the least per-GPU
work, but it falls off fastest as concurrency rises (down to 34) because the
all-reduce / sync overhead per token doesn't amortize. DP2/TP4 holds **40** at
the high-concurrency tail — the best sustained per-user throughput — by splitting
into two independent attention-data-parallel groups (halving the collective
participants per group) while keeping TP4 for the MoE/expert math. DP4/TP2 is a
consistent second: more DP groups but TP2 is too narrow for the expert GEMMs.

At the **c16 production concurrency target**, DP2/TP4 is the configuration all
the kernel-level tok/s figures in the other docs are measured against.

## The ADP-vs-TP decision now owns the MoE-a2a trade (M3 interaction)

**Decision-relevant coupling (2026-06-11), recorded here because the
production-topology call cannot be made without it:**

- **EP a2a — any strategy, including the DeepEP low-latency flip decided GO
  as `optimization_candidates.md` M3 (2.4–2.5× the per-layer a2a roundtrip
  at steady c16, 2.06× at a full 64-token batch) — engages ONLY under
  attention-DP with `moe_tp_size=1`.** The MoE comm factory returns **no
  strategy at all** for plain-TP attention
  (`tensorrt_llm/_torch/modules/fused_moe/communication/`
  `communication_factory.py:126`: `(not enable_attention_dp) or dp_size==1
  ⇒ None`; and `moe_tp_size != 1 ⇒ AllGather/ReduceScatter`, never a2a).
- So the two production-topology candidates trade **different** wins:
  - **ADP attention (+ pure-EP MoE)** unlocks M3's measured 2.4–2.5× a2a
    win on EP comm — **48.4 % of the eager step per the 2026-06-11
    composite re-profile** (spin/skew-inflated under eager; re-measure
    under graphs+overlap post-flip) — but pays the ADP host collectives
    and 4× attention-weight reads.
  - **Pure-TP attention (the WarpDecode+TP plan)** gets faster attention at
    equal load and drops the ADP collectives (S2 moot) — but **has no a2a
    path at all and forfeits M3 entirely**.

Neither sub-win is reachable from the other topology; the ADP-vs-TP call
must weigh them jointly. Full M3 evidence (sizing sweep, the refuted
"inversion at 64", the staged env delta) is in
[optimization_candidates.md](optimization_candidates.md) § M3.

## Disaggregated decode

The deploy runs decode **disaggregated** from prefill (separate decode workers),
so the decode step is a pure generation loop with no prefill interference. This
is what makes the overhead-bound analysis clean: the decode worker's per-step
cost is the thing the campaign minimizes, independent of TTFT.

The custom pieces that are topology-aware:

- **Indexer / sparse-MLA / NVFP4 fusions / WarpDecode** run per-DP-group, per-TP
  rank. They are unaffected by the DP/TP split beyond the obvious per-rank work
  partition (WarpDecode's post-EPLB/post-dispatch hook is what keeps it
  TP/EP-compatible).
- **LayerSplit** (`../source/features/layersplit.md`) is **CP-aware**: it splits
  the DSA KV / indexer-K cache across **context-parallel** ranks and broadcasts
  per layer. It composes with DP/TP through the `Mapping` object — `cp_size` /
  `cp_rank` drive the owner table, and the non-CP modes (TP, EP, ADP) are
  unaffected. The validated initial topologies for LayerSplit are CP=2 and CP=4.
- **SMC-SD** (`smc_sd.md`) shares the TP group between draft and target; it is
  orthogonal to the DP/TP decode topology.

## Deploy surface

- Engine: op-trt + Dynamo single-node decode deploy on 8×B200.
- Production concurrency target: **c16**.
- Topology: **DP2 / TP4**, disaggregated decode.
- Recommended decode config: the Indexer stack (`indexer.md`) with
  `index_topk_freq` set, the sparse-MLA defaults on, the NVFP4 fusions on
  (automatic), WarpDecode opt-in per the deployment guide, LayerSplit opt-in for
  CP deployments, KVarN opt-in once the dense-MLA read path lands.

## Composition

The topology is the substrate every piece composes on. The campaign-level
composition matrix is in [`README.md`](README.md#composition); each piece's
per-topology behavior is in its own doc's "Composition" section. The key
invariant: **no piece changes the captured decode-graph shape per step** (so
graph replay stays valid at every concurrency bucket c1…c32), and **no piece
breaks the DP/TP/EP/CP partition** (the parallelism modes are orthogonal to the
kernel optimizations).

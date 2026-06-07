# Pre-A/B status and remaining gaps - 2026-06-07

This note records the current state of the `op-trt` custom-stack gate after the
r20 disaggregated prefill/decode rollout on the B200 canary. It is intentionally
explicit about incomplete work so later commits do not accidentally treat a
smoke-response or partial marker as production completion.

## Live deployment snapshot

- Repo/branch: `ai-blaise/TensorRT-LLM`, branch `op-trt`.
- Live DGD: `topo-c1-dp2tp4-disagg-r20` in namespace `dynamo-system`.
- Current generation observed: gen89.
- Current image:
  `localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-2750d5541c-nixl-ls-real61fix-cppsplitabi2-202606070338-routerpin-052c9a3-20260607T034750Z-routerpin-cppsplitabi2`.
- Topology: one prefill worker on four B200 GPUs and one decode worker on four
  B200 GPUs, plus the Dynamo KV frontend.
- Live readiness after rollout: frontend, prefill, and decode reached `1/1
  Running` with zero restarts.
- Active custom stack in live config:
  - NIXL cache transceiver on prefill and decode.
  - LayerSplit prefill with `TP2 x CP2`, `cp_type: LAYERSPLIT`, owner-local
    allocation, all-CP-rank transfer, and NIXL transfer backend.
  - Decode `TP4 x CP1` with WarpDecode forced on, `policy: force`, and no
    kernel-backend fallback.
  - Dense MLA latent KV uses 2-bit KVarN (`mla_latent_kv_dtype: kvarn_k2v2`)
    with amortized restore; Indexer K remains FP4/HISA, not KVarN.
  - Moondream-style overlap is enabled while SMC-SD remains deferred in the
    live manifest.

## Latest strict smoke result

Command:

```bash
REQUIRE_DYNAMO_PIN_MARKERS=1 \
REQUIRE_POSITIVE_TRANSFER_METRICS=1 \
REQUIRE_ABORT_CLEANUP_MARKER=1 \
SMC_GATE_MODE=deferred \
./deploy/disagg_pd_r20/smoke_request_pinning.sh
```

Result: failed. The service returned completions and worker metadata, and the
router emitted `dynamo request pin route selected`, `dynamo request pin cleanup
scheduled`, and `dynamo request pin cleared` markers. The smoke correctly failed
because there was still no `dynamo disagg request pin established` marker tying
the selected prefill producer, `ctx_dp_rank`, and outbound decode metadata into a
single request-lifecycle proof.

This is not a model crash: gen89 remained ready with zero restarts. It is a
request-pinning proof gap. The r20 gate must not be marked green until the
frontend emits and preserves the full pin lifecycle:

- route-selected prefill marker;
- pin-established marker with prefill worker, prefill DP rank, bootstrap or
  transfer endpoint metadata;
- outbound-to-decode marker with the same request id and non-null `ctx_dp_rank`;
- route-selected decode marker for the same request id;
- cleanup/clear markers for normal finish and early stream close;
- positive nonzero KV transfer proof from response timing, worker metrics, or
  explicit `OPTRT_NIXL_TRANSFER_PROOF` logs.

The smoke has also been hardened so `KV cache transfer timeout` is a fail-closed
bad pattern. A prior gen88 run completed responses but logged `Terminating
context request ... due to KV cache transfer timeout` on prefill ranks; that must
remain a hard NIXL gate failure until root-caused.

## Completed or partially integrated pieces

### LayerSplit

LayerSplit is live in the r20 prefill config and no HELIX fallback is present.
The owner-local LayerSplit implementation now publishes global transfer metadata
for the native transceiver while keeping local KV/indexer/KVarN pools trimmed to
the CP-owned layer shard. The C++ split path was corrected to use the real local
rank layer span for MLA contiguous CP shards while preserving the physical pool
stride. This keeps the implementation aligned with Z.ai LayerSplit semantics:
layer ownership is by layer shard, not HELIX-style token-block partitioning.

Remaining LayerSplit gap: the end-to-end NIXL transfer proof must pass under the
strict smoke, including the request-pinned handoff from TP2xCP2 prefill into
TP4xCP1 decode. The live readiness proves bootability, not full transfer proof.

Reference: https://z.ai/blog/scaling-pain
Reference doc: `docs/source/features/layersplit.md`

### NIXL

NIXL is the pre-A/B transfer gate and UCX is no longer accepted as the gate
backend. The live config uses NIXL for both `cache_transceiver_config.backend`
and `layersplit_transfer_backend`. The code path is fail-closed against implicit
legacy backend selector conflicts.

Remaining NIXL gaps:

- Prove positive nonzero NIXL KV transfer under strict smoke.
- Root-cause the prior prefill `KV cache transfer timeout` warning.
- Ensure request-pinning metadata is actually propagated into the NIXL handoff,
  not only logged by the scheduler route-selection path.
- Keep UCX, Mooncake, and MORI-IO only for later A/B comparisons until NIXL is
  green.

Reference: https://github.com/ai-blaise/dynamo-prod-k8s/tree/main/docs/api/nixl-connect

### Dense MLA KVarN

Dense MLA latent KV is configured as 2-bit KVarN (`kvarn_k2v2`) in the live r20
prefill/decode configs. The MLA load-path ABI now preserves all historical
`invokeMLALoadPagedKV` overloads so the KVarN/bits-aware path can coexist with
older native extension call sites.

Remaining dense MLA KVarN gap: strict transfer proof and post-first-token
throughput proof are still required before A/B.

Reference: https://github.com/huawei-csl/KVarN
Reference doc: `docs/blaise/kvarn.md`

### Moondream pipelining without SMC-SD

Moondream-style overlap is enabled in r20 prefill and decode via
`disable_overlap_scheduler: false`. The current non-SMC gate keeps the
pipelining path active while SMC-SD is deferred.

Remaining Moondream gap: when SMC-SD is re-enabled, decode must preserve SMC
payloads (`draft_token_log_probs`, sampler events, pinned host draft-token
buffers) through the same overlap scheduler. The source/test audit says this is
wired in op-trt, but live SMC-SD E2E is not complete.

Reference: https://moondream.ai/blog/popping-the-gpu-bubble
Reference doc: `docs/blaise/moondream_pipelining.md`

### Fast iteration/cache work

The r20 deployment now uses persistent cache/prewarm helpers for CUDA, Triton,
TorchInductor, DeepGEMM, TensorRT-LLM build cache, HF artifacts, and model
runtime cache paths. The prewarm script intentionally prewarms only the main
production model while SMC-SD is deferred.

Remaining infrastructure gaps:

- Move stable setup/prewarm scripts into the infrastructure repository once the
  exact deployment flow is final.
- Take and publish a fast-deployment snapshot/artifact bundle after the NIXL and
  request-pinning gates are green.
- Keep final snapshotting separate from speculative Foundry runtime integration;
  Foundry remains a potential complement to `ai-blaise/criu-snapshots`, not a
  proven production dependency.

## Major remaining gaps requested by the user

### Complete SMC-SD

SMC-SD remains deferred from the live r20 gate. The GLM/SGLang kernel work has
several June 6 commits in `op-trt`, and the service can boot without SMC-SD, but
full SMC-SD production integration is not done.

Required completion:

- Finish SMC-SD against https://github.com/abdelfattah-lab/smcsd and
  https://arxiv.org/pdf/2604.15672.
- Import/port any remaining SGLang kernels needed by
  `BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP`.
- Validate the GLM draft model path with the target `DeepSeekV32` main model,
  dense MLA KVarN, NIXL, request pinning, WarpDecode, and LayerSplit.
- Re-enable `SMC_GATE_MODE=required` only after live E2E is stable.

References:

- https://github.com/abdelfattah-lab/smcsd
- https://arxiv.org/pdf/2604.15672
- https://huggingface.co/BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP
- https://github.com/ai-blaise/TensorRT-LLM/commits/op-trt/

### Complete and optimize 2-bit GQA KVarN

The GQA KVarN subagent has advanced the implementation with packed decode,
NIXL/native side-state metadata, sparse KV token selection, and lifecycle
cleanup. It is not production-default yet.

Required completion:

- Prove multi-rank NIXL transfer for packed GQA KVarN pages plus side-state
  fragments.
- Add/finish BDR fold with in-kernel dequant-on-read for the GQA 2-bit path,
  matching the dense MLA optimization level described in `docs/blaise/kvarn.md`.
- Fuse sparse top-k packed read, dequant, and scoring instead of staging through
  a slow dense restore path.
- Prove CUDA graph capture/replay for the SMC-SD GQA path.
- Optimize the store kernel; the prior subagent report still called out a slow
  store path that is not acceptable for production default.
- Validate HF-config deployability and default selection for both dense MLA and
  SMC-SD GQA model paths.

References:

- https://github.com/huawei-csl/KVarN
- `docs/blaise/kvarn_gqa.md`
- `docs/blaise/kvarn.md#bdr-fold-in-kernel-dequant-on-read`

### Request pinning and MORI/Mooncake/NIXL transport composition

Request pinning is partially live: route selection and cleanup are logged, and
normal/early-close paths clear pins. It is not fully proven because the
pin-established/outbound decode markers are missing in strict smoke.

Required completion:

- Emit the full pin-established and outbound-to-decode lifecycle markers from
  the router/front-end path.
- Ensure the same request id and non-null `ctx_dp_rank` reach prefill and decode.
- Keep pinning transport-independent so NIXL, Mooncake, UCX, and MORI-IO can be
  compared later without changing correctness semantics.
- Keep MORI-IO out of the pre-A/B gate for now; use it in A/B only after NIXL is
  proven.

### Optimized disaggregated deployment

The current r20 topology is the intended initial deployment shape: one prefill
worker on four GPUs and one decode worker on four GPUs. It is live and ready,
but not yet accepted as the final optimized production config.

Required completion:

- Prove strict NIXL/request-pinning smoke.
- Verify every custom piece is active, composable, and no implicit fallback is
  being used.
- Warm/cache all recurring autotune shapes so cache-miss fallback tactics do not
  dominate perf runs.
- Keep TP vs EP, memory fraction, kernel backend, and transport variants as A/B
  axes after gates are green.

### A/B testing target

Do not start final A/B until the strict pre-A/B gates are green. The target is 16
simultaneous users and at least 150 tokens/second/user after first token across
1k to 128k input lengths. The custom stack should be expected to win; TP is
likely to beat EP for the target shape, but that must be measured.

Required A/B axes include at minimum:

- GPU topology and worker split.
- TP/EP/attention-DP settings.
- NIXL, Mooncake, UCX, and MORI-IO transports.
- CUDA graph batch-size sets.
- Memory fraction and KV block-size choices.
- WarpDecode and MoE tactic cache settings.
- Dense MLA KVarN and GQA KVarN variants.
- Sparse attention backends and CuteDSL/DeepGEMM kernel choices.

## Current gate checklist

- [x] r20 DGD boots on 1 prefill x 4 GPUs + 1 decode x 4 GPUs.
- [x] NIXL is selected in live config.
- [x] LayerSplit owner-local prefill config is selected.
- [x] Dense MLA KVarN 2-bit is selected.
- [x] WarpDecode is forced on decode with kernel-backend fallback disabled.
- [x] Moondream-style overlap is enabled while SMC-SD is deferred.
- [x] Routerpin image reaches route-selected and cleanup markers.
- [ ] Routerpin emits full pin-established/outbound decode lifecycle markers.
- [ ] Strict request-pinning smoke passes.
- [ ] Positive nonzero NIXL KV transfer proof passes.
- [ ] Prior KV transfer timeout warning is root-caused and eliminated.
- [ ] SMC-SD live E2E with GLM draft model passes.
- [ ] GQA KVarN 2-bit path is production-complete and optimized.
- [ ] Setup scripts are committed to infra repo and snapshot artifact is taken.
- [ ] 16-user A/B matrix is run and tokens/second/user target is met.

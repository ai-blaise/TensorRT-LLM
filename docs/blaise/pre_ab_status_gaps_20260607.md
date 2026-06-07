# Pre-A/B status and remaining gaps - 2026-06-07

This note records the current state of the `op-trt` custom-stack gate after the
r20 disaggregated prefill/decode rollout on the B200 canary. It is intentionally
explicit about incomplete work so later commits do not accidentally treat a
smoke-response, readiness result, or partial marker as production completion.

## Live deployment snapshot

- Repo/branch: `ai-blaise/TensorRT-LLM`, branch `op-trt`.
- Live DGD: `topo-c1-dp2tp4-disagg-r20` in namespace `dynamo-system`.
- Current generation observed: gen91.
- Current stable image:
  `localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-2750d5541c-nixl-ls-real61fix-cppsplitabi2-202606070338-routerpin-052c9a3-20260607T034750Z-routerpin-cppsplitabi2`.
- Current-head proof image under preparation:
  `localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-a3b1c19a481e-headproof-flat-20260607044600`.
- Topology: one prefill worker on four B200 GPUs and one decode worker on four
  B200 GPUs, plus the Dynamo KV frontend.
- Live readiness after rollback: frontend, prefill, and decode are `1/1
  Running` with zero restarts on the stable image.
- Rollout note: the unflattened current-head thin overlay
  (`optrt-a3b1c19a481e-headproof-20260607042547`) carried the expected source
  markers but failed at pod start with containerd rootfs `mount options is too
  long`. Do not deploy chained overlays for this gate. Flatten the current-head
  image, import it into k3s containerd, verify source markers in the flattened
  image, and only then roll it into the DGD.
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

## Latest strict smoke result and current gate state

NIXL is the pre-A/B KV-transfer gate. UCX, Mooncake, and MORI-IO are A/B-only
until NIXL proves end-to-end correctness under the custom r20 stack. The strict
smoke command remains:

```bash
REQUIRE_DYNAMO_PIN_MARKERS=1 \
REQUIRE_POSITIVE_TRANSFER_METRICS=1 \
REQUIRE_ABORT_CLEANUP_MARKER=1 \
SMC_GATE_MODE=deferred \
./deploy/disagg_pd_r20/smoke_request_pinning.sh
```

The prior gen88/gen89 failure is understood more narrowly now: requests reached
the service, emitted worker metadata, and selected NIXL, but prefill ranks logged
`Terminating context request ... due to KV cache transfer timeout`. The root
cause was in `cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp`: a canceled or
not-ready cache sender response erased the ready-response entry without
fulfilling its promise, so the sender future could remain pending until the
Python-side transfer timeout expired. Commit `342222145` (`fix(nixl): complete
cancelled cache sender futures`) completes the promise before erase and is now a
hard dependency for the NIXL gate.

This fix is not a green gate by itself. A rebuilt/full-source or ABI-compatible
image containing `342222145` must pass the strict smoke above. The smoke must
prove all of the following in one request lifecycle:

- route-selected prefill marker;
- pin-established marker with prefill worker, prefill DP rank, bootstrap or
  transfer endpoint metadata;
- outbound-to-decode marker with the same request id and non-null `ctx_dp_rank`;
- route-selected decode marker for the same request id;
- cleanup/clear markers for normal finish and early stream close;
- positive nonzero NIXL transfer proof from explicit
  `OPTRT_NIXL_TRANSFER_PROOF` logs or an equivalent worker-side metric source;
- no `KV cache transfer timeout`, MLA formatter rejection, illegal memory access,
  UCX fallback marker, HELIX fallback marker, or worker restart.

The `/perf_metrics` endpoint on the frontend returned 404 in earlier smokes, so
positive transfer proof must not depend on that frontend URL. Current proof
should come from worker logs and/or worker-side metrics surfaces. The smoke and
benchmark harness treat a missing positive NIXL transfer proof as failure rather
than silently accepting a decode response.

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

- Rebuild or select an image that contains commit `342222145` and the current
  r20 LayerSplit/KVarN/request-pinning sources.
- Prove positive nonzero NIXL KV transfer under strict smoke.
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

SMC-SD decode now has a fail-closed Moondream handoff guard in
`tensorrt_llm/_torch/speculative/smc.py`: generation-only decode requests must
carry request pin metadata (`disagg_request_id` or `ctx_request_id`,
`ctx_dp_rank`, and `ctx_info_endpoint`) before the delayed SMC draft-token
commit consumes `draft_token_log_probs`. The handoff also waits on
`sample_state.sampler_event` before reading pinned host draft-token buffers and
emits `SMC Moondream decode handoff preserved ... pinned_host_tokens=True ...
ctx_dp_rank=... ctx_info_endpoint=...` for live proof.

Remaining Moondream gap: live SMC-SD E2E is still required. Run the strict smoke
with `SMC_GATE_MODE=required` only after the NIXL/LayerSplit gate is green and
SMC-SD is explicitly enabled; that mode now fails if the decode handoff marker,
pinned host-token proof, ctx DP rank, or ctx endpoint is missing.

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
- Preserve the flat-image escape hatch in the iteration docs. It is required
  when an overlay-on-overlay rebuild reaches containerd rootfs mount-option
  limits; it should not replace full source builds for ABI-affecting C++/CUDA
  changes.

## NIXL c16 benchmark harness

After strict smoke is green and the GPU window is assigned, run the NIXL
throughput gate before any UCX/Mooncake/MORI A/B. The harness is intentionally
NIXL-first and fail-closed:

```bash
./deploy/disagg_pd_r20/run_c16_transport_bench.sh \
  --backend nixl \
  --lengths 1024,4096,8192,16384,32768,65536,131072 \
  --concurrency 16 \
  --max-tokens 128 \
  --min-tok-per-user 150
```

Artifacts are written under `/tmp/r20_transport_bench_<backend>_<timestamp>`
unless `BENCH_OUT` is set. Each run captures `results.jsonl`, `summary.json`,
frontend/prefill/decode logs, pod descriptions, `nvidia-smi dmon`, and
`ip -s link` before/after snapshots. The post-run verifier fails closed on:

- any request failure or worker restart;
- `KV cache transfer timeout`;
- `MLACacheFormatter::inquireSupport`, `CacheTransferLayer::validateSupport`,
  same-layer-count rejection, or CUDA illegal memory access;
- NIXL error/failure logs;
- UCX backend or `layersplit_transfer_backend: ucx` markers during a NIXL run;
- missing nonzero `OPTRT_NIXL_TRANSFER_PROOF` start+complete pair;
- missing `nvext.worker_id` on successful responses;
- any prompt length below the configured tokens/sec/user-after-first-token floor.

UCX, Mooncake, and MORI runs require explicit A/B opt-in and are not allowed to
replace the pre-A/B NIXL gate:

```bash
ALLOW_TRANSPORT_AB=1 ./deploy/disagg_pd_r20/run_c16_transport_bench.sh --backend ucx
ALLOW_TRANSPORT_AB=1 ./deploy/disagg_pd_r20/run_c16_transport_bench.sh --backend mooncake
ALLOW_TRANSPORT_AB=1 ./deploy/disagg_pd_r20/run_c16_transport_bench.sh --backend mori
```

Mooncake and native MORI remain blocked unless their TRT-LLM-compatible wrapper
libraries and runtime APIs are present in the selected image. Do not claim a
Mooncake or MORI win from config importability alone.

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
- Review the June 6 `op-trt` commits before further kernel work so the port does
  not duplicate or regress already-imported SGLang pieces.
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
- [x] Current-head thin overlay source markers are present.
- [ ] Current-head flattened image is imported into k3s containerd and deployed.
- [ ] Routerpin emits full pin-established/outbound decode lifecycle markers.
- [ ] Strict request-pinning smoke passes.
- [ ] Positive nonzero NIXL KV transfer proof passes.
- [ ] Prior KV transfer timeout warning is root-caused and eliminated.
- [ ] SMC-SD live E2E with GLM draft model passes.
- [ ] GQA KVarN 2-bit path is production-complete and optimized.
- [ ] Setup scripts are committed to infra repo and snapshot artifact is taken.
- [ ] 16-user A/B matrix is run and tokens/second/user target is met.

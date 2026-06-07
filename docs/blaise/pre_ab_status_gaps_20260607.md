# Pre-A/B status and remaining gaps - 2026-06-07

This note records the current state of the `op-trt` custom-stack gate after the
r20 disaggregated prefill/decode rollout on the B200 canary. It is intentionally
explicit about incomplete work so later commits do not accidentally treat a
smoke-response, readiness result, or partial marker as production completion.

## Live deployment snapshot

- Repo/branch: `ai-blaise/TensorRT-LLM`, branch `op-trt`.
- Current pushed head before this doc refresh: `45a05fe19`
  (`fix(nixl): preserve sender future timeout config`). This includes the
  earlier NIXL fail-closed, SMC/Moondream pinned-handoff, image-reuse,
  prewarm dry-run, snapshot-composition, gap-status, strict-smoke parsing,
  idle-transfer-poll, stream-drain, strict preflight, NIXL plugin probe, and
  sender future timeout preservation commits.
- Live DGD: `topo-c1-dp2tp4-disagg-r20` in namespace `dynamo-system`.
- Current DGD generation: `100`; DGD readiness is `True`.
- Current live image:
  `localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-45a05fe19409-fullsrc45a-nixlpin-20260607T114045Z`.
- Current live image digest:
  `sha256:14a2a2b26ad75533f868ab6f900b9545c9f09163efec07c9c4c0d9b9541737d4`.
- Image note: this is a full source TensorRT-LLM image built from `45a05fe19`,
  not a thin overlay. It includes the C++/nanobind sender future timeout
  preservation fix as well as the Python stream-drain and idle-transfer-poll
  fixes.
- Topology: one prefill worker on four B200 GPUs and one decode worker on four
  B200 GPUs, plus the Dynamo KV frontend.
- Live readiness: frontend, prefill, and decode are `1/1 Running` with zero
  restarts on the current image.
- Rollout note: chained thin overlays previously hit containerd rootfs
  `mount options is too long`. Do not deploy chained overlays for ABI-affecting
  C++/CUDA fixes. Use a full source build for the next endpoint/NIXL proof.
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

Latest live result: green for the NIXL/request-pinning pre-A/B correctness
gate on generation 100 using the full-source `45a05fe19` image.

- Live NIXL audit passed:
  `/tmp/nixl_gate_audit_live_20260607T121624Z_2312782`.
- Strict preflight passed before deploy:
  `/tmp/r20-strict-smoke-preflight-fullsrc45a-20260607T114045Z`.
- Strict smoke passed with:
  `prefill=('8531764574012418', '0')`,
  `decode=('8849396450087062', '0')`,
  `dynamo_required=True`,
  `established=2`,
  `outbound=2`,
  `cleared=4`,
  `cleanup_scheduled=1`,
  `positive_transfer_metrics=2`.
- The checked pod set was ready and zero-restart:
  `topo-c1-dp2tp4-disagg-r20-0-frontend-9ptgb`,
  `topo-c1-dp2tp4-disagg-r20-0-prefill-c6hdf`, and
  `topo-c1-dp2tp4-disagg-r20-0-decode-xrtt6`.
- Final log sanity after the smoke showed smoke request ids `6310396876148736`
  and `6310572160307201` had four CP-rank `context_send_start` entries and
  matching `context_send_complete` entries on prefill, `gen_recv_start` on
  decode, and no `KV cache transfer timeout` in the checked window.

The previous gen88/gen89 `KV cache transfer timeout` bug remains a historical
hard dependency. It is now covered by two fixes: cancelled/not-ready sender
promises are completed before ready-response entries are erased, and the
prefill executor keeps polling in-flight disaggregated transfers while idle so
context-only sends can complete even if an early-closed stream leaves no next
request to wake the rank-0 broadcast loop.

The strict smoke must prove all of the following in one request lifecycle:

- route-selected prefill marker;
- pin-established marker with prefill worker, prefill DP rank, and non-empty
  `ctx_info_endpoint`;
- outbound-to-decode marker with the same request id and non-null `ctx_dp_rank`;
- route-selected decode marker for the same request id;
- cleanup/clear markers for normal finish and early stream close;
- positive nonzero NIXL transfer proof from explicit
  `OPTRT_NIXL_TRANSFER_PROOF` logs or an equivalent worker-side metric source;
- no missing `ctx_info_endpoint`, `KV cache transfer timeout`, MLA formatter
  rejection, illegal memory access, UCX fallback marker, HELIX fallback marker,
  or worker restart.

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

Current LayerSplit gate state: strict smoke now proves the request-pinned
handoff from TP2xCP2 prefill into TP4xCP1 decode with positive NIXL transfer
proof. Remaining work is performance validation across the A/B matrix and the
SMC-SD decode path, not the non-SMC LayerSplit/NIXL correctness gate.

Reference: https://z.ai/blog/scaling-pain
Reference doc: `docs/source/features/layersplit.md`

### NIXL

NIXL is the pre-A/B transfer gate and UCX is no longer accepted as the gate
backend. The live config uses NIXL for both `cache_transceiver_config.backend`
and `layersplit_transfer_backend`. The code path is fail-closed against implicit
legacy backend selector conflicts.

Current NIXL gate state:

- Endpoint-fixed full-source base plus Python overlay is deployed and ready.
- Positive nonzero NIXL KV transfer is proven under strict smoke.
- Request-pinning metadata reaches the NIXL handoff, including non-empty
  completed-prefill endpoint metadata.
- The r20 gate config/source now pins
  `cache_transceiver_config.transceiver_runtime: PYTHON` and requires
  generation-first/write-mode markers. The previous live proof is still a
  correctness proof for NIXL transfer, but it is not the MORI-style write-mode
  target because the frontend logs show `handoff_mode="completed_prefill"`.
- The image now explicitly installs `msgpack`, and the smoke/audit check the
  native Python transfer import path so `transceiver_runtime: PYTHON` cannot be
  marked ready while its transfer worker dependency chain is broken.
- The r20 gate now defaults to the NIXL libfabric plugin
  (`TRTLLM_NIXL_KVCACHE_BACKEND=LIBFABRIC`), not direct UCX and not the NIXL UCX
  plugin. UCX, Mooncake, and MORI-IO remain A/B candidates until they prove
  equal or better correctness and throughput under the same custom stack.

Reference: https://github.com/ai-blaise/dynamo-prod-k8s/tree/main/docs/api/nixl-connect

### Dense MLA KVarN

Dense MLA latent KV is configured as 2-bit KVarN (`kvarn_k2v2`) in the live r20
prefill/decode configs. The MLA load-path ABI now preserves all historical
`invokeMLALoadPagedKV` overloads so the KVarN/bits-aware path can coexist with
older native extension call sites.

Remaining dense MLA KVarN gap: strict transfer proof is green in the non-SMC
r20 gate, but post-first-token throughput proof is still required before A/B is
accepted.

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
ctx_dp_rank=... ctx_info_endpoint=...` for live proof. If that evented pinned
`sample_state` is absent, SMC-SD now fails closed by default instead of falling
back to a blocking `.cpu()` draft-token copy; the unpinned path requires the
explicit diagnostic override `TRTLLM_SMC_ALLOW_UNPINNED_DRAFT_COMMIT=1`.

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
- Treat the SGLang GLM path as the practical source implementation where it
  already has optimized kernels. Port/copy the required kernels directly, then
  adapt the TensorRT-LLM runner/resource-manager APIs rather than re-deriving
  the kernel family from scratch.
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
batched aligned full-block store, NIXL/native side-state metadata, sparse KV
token selection, and lifecycle cleanup. It is not production-default yet.

Required completion:

- Keep HF/default config fail-closed and strict. Explicit invalid `kvarn_*` top-level dtypes and non-GQA-KVarN nested
  `quantization_config.kvarn.gqa.dtype` values must fail instead of silently
  rewriting to `kvarn_k2v2_g128`; omitted enabled `gqa.dtype` remains the
  production-default request.
- Prove multi-rank NIXL transfer for packed GQA KVarN pages plus side-state
  fragments. The integration branch has side-state metadata/fragments,
  receiver-side device block-table mirroring, physical generation reconstruction
  after destination block remap, and unit coverage for nonzero request-pinned
  slot offsets; it still needs the real multi-rank NIXL run.
- Finish and prove BDR fold with in-kernel dequant-on-read for the GQA 2-bit
  path, matching the dense MLA optimization level described in
  `docs/blaise/kvarn.md`. The branch has device-side block-table mirroring,
  physical-block generation tracking, receiver-side generation reconstruction,
  amortized dequant, packed decode full-block ids gathered from the mirrored
  device table with async CUDA validation, and packed decode/sparse-decode
  kernels that read/dequant in-kernel; live parity/perf proof is still pending.
- Keep the default production KV contract explicit: dense MLA defaults to
  `kvarn_k2v2`; the SMC-SD GQA draft/target model path must also be deployable
  from Hugging Face config with `kvarn_k2v2_g128` as its default once the backend
  is proven. KVarN must never be applied to the Indexer K path.
- Fuse sparse top-k packed read, dequant, and scoring instead of staging through
  a slow dense restore path. The branch adds
  `torch.ops.trtllm.kvarn_gqa_decode_sparse` for top-k<=256; B200 sparse parity
  and E2E HISA composition are still pending.
- Prove CUDA graph capture/replay for the SMC-SD GQA path. The branch adds a
  `--graph-replay` low-memory harness for store, dense decode, and sparse decode;
  it still needs a free B200 run.
- Optimize the store kernel; the branch writes packed 2-bit bytes directly rather
  than zeroing and read/OR/writing every byte, but post-optimization store timing
  still needs to be measured.
- Validate HF-config deployability and default selection for both dense MLA and
  SMC-SD GQA model paths. The integration branch accepts explicit SMC
  `draft_kv_cache_dtype="kvarn_k2v2_g128"`, forces 128-token draft KV blocks for
  that dtype, and keeps the runtime fail-closed until fused readiness is proven.

References:

- https://github.com/huawei-csl/KVarN
- `docs/blaise/kvarn_gqa.md`
- `docs/blaise/kvarn.md#bdr-fold-in-kernel-dequant-on-read`

### Request pinning and MORI/Mooncake/NIXL transport composition

Request pinning is live for the current non-SMC r20 gate. Route selection,
pin-establish, outbound-to-decode, cleanup, early stream close, positive NIXL
transfer, and completion markers are all proven in the generation-99 strict
smoke. Dynamo still fails closed if completed prefill lacks endpoint metadata.

Required completion:

- Keep the full pin-established and outbound decode lifecycle markers in every
  production manifest and benchmark run.
- Ensure the same request id and non-null `ctx_dp_rank` continue to reach
  prefill and decode across all A/B variants.
- Ensure completed-prefill metadata continues to include non-empty
  `ctx_info_endpoint`; the router must continue to reject unpinned decode if
  this field is missing.
- Keep pinning transport-independent so NIXL, Mooncake, UCX, and MORI-IO can be
  compared later without changing correctness semantics.
- Keep MORI-IO out of the pre-A/B gate for now; use it in A/B only after NIXL is
  proven.
- NIXL replaces UCX as the pre-A/B KV-pool gate. UCX, Mooncake, and MORI-IO are
  comparison candidates only after this now-green NIXL correctness gate.

### Optimized disaggregated deployment

The current r20 topology is the intended initial deployment shape: one prefill
worker on four GPUs and one decode worker on four GPUs. It is live and ready,
but not yet accepted as the final optimized production config.

Required completion:

- Keep strict NIXL/request-pinning smoke green after every production-path
  integration.
- Verify every custom piece is active, composable, and no implicit fallback is
  being used.
- Confirm the exact r20 custom stack in the accepted manifest: prefill TP2xCP2
  LayerSplit with owner-local allocation, all-CP transfer, and NIXL transfer;
  decode TP4/CP1 with WarpDecode forced on and no kernel-backend fallback;
  dense MLA `kvarn_k2v2` with BDR/amortized restore; HISA/Indexer K remaining
  fp4; Moondream overlap enabled; SMC-SD deferred until its GLM/GQA pieces are
  proven.
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
- [x] Current pushed `op-trt` head includes the C++ completed-prefill endpoint
  source fix, NIXL plugin fail-closed behavior, SMC/Moondream pinned-handoff
  guard, image reuse helper, prewarm dry-run path, and snapshot-composition
  hardening.
- [x] Static C++ endpoint audit passes from a clean VM checkout.
- [x] Offline request-pinning smoke, including negative endpoint and SMC
  handoff cases, passes.
- [x] Cached DGD render/server-dry-run validation passes with the active image.
- [x] Live NIXL gate readiness audit passes on the current generation.
- [x] Endpoint-fixed full-source base plus latest Python/shell overlay is built,
  pushed to the local registry, imported/resident, and deployed.
- [x] Routerpin emits full pin-established/outbound decode lifecycle markers
  with non-empty completed-prefill `ctx_info_endpoint`.
- [x] Strict request-pinning smoke passes.
- [x] Positive nonzero NIXL KV transfer proof passes.
- [x] Prior KV transfer timeout warning stays absent after the endpoint-fixed
  rollout and idle-transfer-poll overlay.
- [ ] SMC-SD live E2E with GLM draft model passes.
- [ ] Remaining SGLang GLM kernels are ported/adapted for
  `BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP`.
- [ ] GQA KVarN 2-bit path is production-complete and optimized.
- [ ] GQA KVarN BDR fold is implemented and benchmarked against the dense
  restore path.
- [ ] Moondream pipelining is proven with SMC-SD decode enabled, not only in the
  current SMC-deferred gate.
- [ ] Setup scripts are committed to infra repo and snapshot artifact is taken.
- [ ] 16-user A/B matrix is run and tokens/second/user target is met.

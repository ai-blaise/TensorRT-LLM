# Disagg P/D r20 deploy manifest -- cell1 DP2/TP4-disagg winner

Disaggregated prefill/decode Dynamo deploy for the production STARTING topology,
with the custom pieces toggled ON.

## Topology (single 8xB200 node, a4-us-002-rl9 / k3s)

| Worker   | GPUs  | Parallelism            | Custom piece ON                              |
|----------|-------|------------------------|----------------------------------------------|
| prefill  | 4 GPUs | TP2xCP2 LayerSplit / EP4, ADP=false | **LayerSplit** (`layersplit_enabled: true`) + **2-bit KVarN dense MLA latent KV** (`mla_latent_kv_dtype: kvarn_k2v2`) |
| decode   | 4 GPUs | TP4 / EP4, ADP=true, MNNVL | **WarpDecode** (`warp_decode.enabled: true`, `tile_mode: decode_1cta`) + **2-bit KVarN dense MLA latent KV** |
| Frontend | -     | KV router (`--router-mode kv`) | -                                  |

This is 1P x 4GPU + 1D x 4GPU disaggregated serving with real LayerSplit on
prefill (`TP2 x CP2`) and non-CP decode (`TP4 x CP1`).

## Why node 002 (k3s)

The disaggregated P/D artifact is the `DynamoGraphDeployment` CRD
(`nvidia.com/v1alpha1`), which is k3s-native and is the pre-A/B artifact for the NIXL + LayerSplit + request-pinning gate. Aggregated SMC-SD launch paths are not part of this disaggregated gate.

## Image

Both workers + frontend run the unified NIXL/LayerSplit gate image, parameterized as `${UNIFIED_IMAGE}`. Confirm the tag with the build agent before apply and keep draft-diagnostic images off this gate.

For TP2xCP2 prefill -> TP4xCP1 decode, use a full source-built runtime image,
not a Python-only overlay over an older base. The C++ MLA cache formatter must
include the CP-domain reassembly path (`mDomainCPSize > 1`) or decode KV receive
will reject the LayerSplit handoff.

## Deploy (orchestrator only -- gated)

```bash
export UNIFIED_IMAGE=localhost:5000/local/dynamo-trtllm-optrt-custom:<nixl-layer-split-gate-tag>   # from build agent
envsubst '$UNIFIED_IMAGE' < topo-c1-dp2tp4-disagg-r20.yaml | \
  KUBECONFIG=/etc/rancher/k3s/k3s.yaml k3s kubectl apply -f -
```

`hf-token-secret` (namespace `dynamo-system`) and a `/models` host directory must
exist on a4-us-002-rl9.

## Fast iteration path

Use `fast_iterate.sh` for Python/config/doc/test iterations. By default, it
derives the exact `tensorrt_llm/...` files copied by `Dockerfile.r20-overlay`,
syncs only those files plus `deploy/` and Docker metadata, then falls back to
tar-over-SSH when the VM does not have `rsync`. Before building, it installs
`Dockerfile.r20-overlay.dockerignore` as the remote build-root `.dockerignore`,
so `--full-sync` debugging does not accidentally send/hash the full repository
for a thin overlay rebuild. It builds the existing thin
overlay image on the B200 VM and uses `nerdctl -n k8s.io build` when available so
the image lands directly in k3s containerd. If `nerdctl` is unavailable, it falls
back to Docker BuildKit plus a single `ctr images import`; if the requested base
image is already in k3s containerd but not Docker, the script loads that base
into Docker once. By default, the script layers on the latest known-good
full-source runtime that carries the required NIXL transport wrapper. Use
`--base-image` only when intentionally selecting another proven full image. Do not chain thin overlays on top of earlier thin overlays: the
extra layer depth can exceed containerd rootfs mount option limits. New r20
overlay images are labeled and `fast_iterate.sh` refuses them as a base unless
`--allow-chained-overlay` is passed deliberately. Use `--full-sync` only when
remote debugging needs the full repository. The script also preflights required
transport wrappers before prewarm/deploy; the default requirement is `ucx,nixl`
and can be adjusted with `--required-transport-wrappers`.

For the fastest image handoff, add `--use-local-registry`. The script starts or
reuses a `registry:2` container on the VM, pushes the thin overlay to
`localhost:5000`, and renders the DGD with `imagePullPolicy: IfNotPresent`. That
lets k3s/containerd pull only missing thin layers instead of importing a full
`docker save` archive. Keep the default import path when you need the most
conservative `imagePullPolicy: Never` behavior.

If a deliberately chained thin overlay reaches containerd's rootfs mount option
limit (`failed to mount rootfs component: mount options is too long`), do not
continue rolling that tag. Roll back to the last ready image, flatten the
current-head image with `docker export | docker import`, import the flattened tag
into k3s containerd, and re-run the source-marker check before applying the DGD.
The flattened tag should carry `ai.blaise.flattened=true` and should only be
used as an iteration artifact; ABI-affecting C++/CUDA changes still require a
fresh full source build.

Build only:

```bash
deploy/disagg_pd_r20/fast_iterate.sh \
  --vm local \
  --target-node a4-us-001-rl9 \
  --tag-suffix swapab-host
```

Render the resolved plan without SSH, sync, build, prewarm, or deploy. This is
useful for checking local-registry/resident-mode tags before touching an active
DGD:

```bash
deploy/disagg_pd_r20/fast_iterate.sh \
  --vm 34.106.33.128 \
  --target-node a4-us-001-rl9 \
  --tag-suffix plan-check \
  --use-local-registry \
  --local-registry-mode resident \
  --prewarm \
  --deploy \
  --dry-run
```

Render and API-server-validate the DGD with the currently active image, without
building/importing a new image and without applying the DGD. This is the fastest
router/deploy-YAML check when only the DynamoGraphDeployment/config changed.
Keep alternate DGD names short enough for the `Frontend` service name; the
helpers fail fast when `len(DGD_NAME) + 8 > 45`:

```bash
deploy/disagg_pd_r20/render_dgd.sh \
  --image-from-dgd topo-c1-dp2tp4-disagg-r20 \
  --target-node a4-us-001-rl9 \
  --dgd-name r20-render-check \
  --image-pull-policy IfNotPresent \
  --out /tmp/r20-render-check.yaml \
  --server-dry-run
```

Build and apply the main DGD:

```bash
deploy/disagg_pd_r20/fast_iterate.sh \
  --vm 34.106.33.128 \
  --target-node a4-us-001-rl9 \
  --tag-suffix swapab-host \
  --use-local-registry \
  --prewarm \
  --deploy
```

Build and warm an isolated canary DGD on the second B200 VM:

```bash
deploy/disagg_pd_r20/fast_iterate.sh \
  --vm 34.106.191.132 \
  --base-image local/dynamo-trtllm-optrt-custom:<nixl-layer-split-gate-base> \
  --target-node a4-us-002-rl9 \
  --dgd-name topo-c1-dp2tp4-disagg-r20-canary \
  --tag-suffix canary \
  --deploy
```

The canary name rewrites both the `DynamoGraphDeployment` and ConfigMap names,
so it can coexist with the main DGD when there are enough free GPUs. Keep the
main DGD untouched while a production workload is active.

## Persistent caches

The DGD mounts `/var/lib/optrt-cache` from the host into both prefill and decode
as `/cache/optrt`. These paths persist across pod restarts:

- `/cache/optrt/hf_modules` for Hugging Face remote-code modules.
- `/cache/optrt/transformers` and `/cache/optrt/hf_datasets` for HF library
  metadata caches.
- `/cache/optrt/xdg` for generic Python/library caches.
- `/cache/optrt/pip` for Python package/download cache.
- `/cache/optrt/torch_extensions` for Torch extension builds.
- `/cache/optrt/torchinductor` for TorchInductor graph/codegen artifacts.
- `/cache/optrt/triton` for Triton kernel cache.
- `/cache/optrt/cuda` for CUDA JIT cache.
- `/cache/optrt/deep_gemm` for DeepGEMM JIT cubins generated via
  `DG_JIT_CACHE_DIR`.
- `/cache/optrt/tensorrt_llm/dg` for DeepGEMM/TRT-LLM generated artifacts.
- `/cache/optrt/tensorrt_llm/llmapi_build` for `TLLM_LLMAPI_BUILD_CACHE`.

Prepare and lightly prewarm those caches before a rollout:

```bash
IMAGE=docker.io/local/dynamo-trtllm-optrt-custom:optrt-<sha>-<suffix> \
deploy/disagg_pd_r20/prewarm_caches.sh \
  --vm 34.106.33.128 \
  --target-node a4-us-001-rl9 \
  --image "$IMAGE" \
  --image-pull-policy Never
```

For VM-local registry images from `fast_iterate.sh --use-local-registry`, use
`IfNotPresent` so k3s can pull the already-pushed thin image:

```bash
deploy/disagg_pd_r20/prewarm_caches.sh \
  --vm 34.106.33.128 \
  --target-node a4-us-001-rl9 \
  --image localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-<sha>-<suffix> \
  --image-pull-policy IfNotPresent
```

This prewarm job validates that the image is resident in k3s containerd, the
production model paths are visible under `/models`, and offline HF config/tokenizer
loading works before the full prefill/decode workers spend time loading weights
and compiling kernels.

When running directly on the B200 VM, use `--vm local` to bypass SSH setup. Use
`--dry-run` to render the prewarm Job YAML without applying it or touching the
persistent cache directories:

```bash
deploy/disagg_pd_r20/prewarm_caches.sh \
  --vm local \
  --dry-run \
  --image localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-<sha>-<suffix> \
  --image-pull-policy IfNotPresent
```

## Files

- `topo-c1-dp2tp4-disagg-r20.yaml` -- the deployable manifest: a `ConfigMap`
  (`topo-c1-dp2tp4-disagg-r20-config`, holds `prefill.yaml` + `decode.yaml`) plus
  the `DynamoGraphDeployment` (`topo-c1-dp2tp4-disagg-r20`). Apply this one file.
- `Dockerfile.r20-overlay` -- configurable-base thin overlay image for
  Python/config iterations.
- `prefill.yaml` / `decode.yaml` -- standalone copies of the two engine configs
  (identical to the ConfigMap data blocks) for review / diff / reuse.
- `fast_iterate.sh` -- fast rsync/build/import/apply helper for thin overlay
  iterations and alternate-name canary deployments.
- `render_dgd.sh` -- render and optionally server-dry-run the DGD with an
  existing image, so router/deploy-YAML checks do not require a new image loop.
- `prewarm_caches.sh` -- persistent-cache preparation and lightweight offline HF
  prewarm/validation job.
- `smoke_request_pinning.sh` -- ready-only live gate for non-MORI request
  pinning, Moondream overlap compatibility, normal close, and early stream
  close cleanup before A/B.
- `cache_report.sh` -- read-only VM report for persistent-cache growth, local
  registry availability, and k3s/containerd image residency.
- `snapshot_readiness.sh` -- read-only CRIU snapshot composition report for the
  R20 DGD; it checks host/Kubernetes substrate and reports the TensorRT-LLM hook
  blocker without creating snapshot resources or touching live pods.
- `Dockerfile.r20-overlay.dockerignore` -- overlay-specific build-context
  allowlist so thin-image rebuilds do not ship the full repo to Docker/BuildKit.

Inspect cache/image residency without touching pods. From the B200 VM itself,
`--vm local` avoids SSH and just runs the read-only report locally:

```bash
deploy/disagg_pd_r20/cache_report.sh \
  --vm local \
  --dgd-name topo-c1-dp2tp4-disagg-r20 \
  --image-filter dynamo-trtllm-optrt-custom
```

Run this after the first cold rollout and again after the next overlay rollout.
The useful signal is whether active prefill/decode images are already resident
in k3s/containerd with the intended pull policy, and whether `triton`, `cuda`,
`deep_gemm`, and `tensorrt_llm/*` grow and then stabilize. If the persistent
cache directories remain empty, the workers are not writing to the intended
cache paths. If an active image is not resident, a resident-mode deploy will not
be reproducible without a registry push or explicit image import.

Inspect CRIU snapshot composition readiness without touching pods:

```bash
deploy/disagg_pd_r20/snapshot_readiness.sh \
  --vm 34.106.33.128 \
  --dgd-name topo-c1-dp2tp4-disagg-r20
```

This is a preflight only. `safe_to_take_snapshot=0` is expected until R20 has a
gated TensorRT-LLM pre-snapshot/post-restore hook and the proof gates in
`docs/blaise/r20_snapshot_proof_criteria.md` pass: NIXL in-flight restore,
LayerSplit TP2xCP2 restore, dense KVarN/CUDA graph scratch restore, and
`checkpointctl` restore-image tooling.

## Knob provenance

- WarpDecode: `tensorrt_llm/llmapi/llm_args.py` `WarpDecodeConfig`
  (`enabled`/`policy`/`tile_mode`), `docs/blaise/warpdecode.md`. Fixed-tactic
  overlay is default-on via env `TRTLLM_WARP_DECODE_FIXED_TACTIC=1` (set on the
  decode worker) + `TRTLLM_ENABLE_PDL=1`.
- LayerSplit: `BaseSparseAttentionConfig.layersplit_*`,
  `docs/source/features/layersplit.md`.
- KVarN: `BaseSparseAttentionConfig.mla_latent_kv_dtype` and
  `mla_latent_kv_amortize`, `docs/blaise/kvarn.md`. This deployment uses
  `kvarn_k2v2` for **dense MLA latent KV only**; `indexer_k_dtype` remains
  `fp4` and is not replaced by KVarN.
- SMC-SD/GLM draft decoding is post-gate. Keep it out of this pre-A/B manifest, smoke, and image provenance until NIXL, LayerSplit, request pinning, WarpDecode, and dense KVarN are proven.
- Prefill NCCL: `NCCL_NVLS_ENABLE=0`. LayerSplit prefill uses native TP/CP
  subgroup allreduces; on the tested B200/K3s stack, NCCL NVLS multicast
  binding fails for those subgroups while NCCL CUMEM/P2P completes correctly.

## LayerSplit TP2xCP2 prefill

LayerSplit splits the DSA KV / indexer-K cache across **context-parallel** ranks.
This manifest runs the prefill worker as TP2 x CP2 on four GPUs
(`tensor_parallel_size: 2`, `context_parallel_size: 2`, `cp_config.cp_type:
LAYERSPLIT`) so LayerSplit is a real CP split and MoE EP remains supported. There is
**no** `--context-parallel-size` Dynamo CLI flag; CP is set only through the
engine YAML `context_parallel_size` field. Decode remains TP4/CP1 and consumes
the reassembled KV through the LayerSplit KV handoff path.

## NIXL gate audit and tuning knobs

Run the read-only audit before sending request traffic to a new NIXL gate image:

```bash
NIXL_AUDIT_MODE=live CHECK_RUNTIME_LIBS=1 \
  deploy/disagg_pd_r20/audit_nixl_gate_readiness.sh
```

For local manifest validation before deploy, use:

```bash
NIXL_AUDIT_MODE=local deploy/disagg_pd_r20/audit_nixl_gate_readiness.sh
```

The audit does not send completions, apply manifests, delete pods, or restart
workers. It validates the R20 composition and writes artifacts under
`/tmp/nixl_gate_audit_<timestamp>`. A passing audit proves only readiness for the
request smoke; it does not replace the request-pinning smoke or the c16
throughput gate.

The highest-impact NIXL knobs for the current B200/NVLink R20 shape are:

- `cache_transceiver_config.max_tokens_in_buffer: 131072` on both prefill and
  decode. TRT-LLM C++ warns that dynamic transfer buffers can fail with NIXL;
  the pre-registered buffer must cover the 128k ISL target.
- `TRTLLM_NIXL_KVCACHE_BACKEND=UCX` on both workers. This selects the UCX plugin
  inside NIXL explicitly; it is not the old direct UCX cache transceiver.
- `TRTLLM_NIXL_ENABLE_COALESCE=1` on both workers. NIXL coalesces contiguous
  VMM-split descriptors during registration, deregistration, and transfer request
  creation, reducing descriptor count and hot-path overhead.
- `UCX_CUDA_IPC_ENABLE_MNNVL=0`, `NVIDIA_GDRCOPY=1`, `NCCL_NET_PLUGIN=none`, and
  `TRTLLM_FORCE_COMM_METHOD=NVLINK_TWO_SIDED` keep the single-node B200/NVLink
  path explicit and avoid the direct UCX MNNVL warning path seen in earlier
  rollouts.
- Prefill keeps `NCCL_NVLS_ENABLE=0` because TP/CP subgroup allreduces hit NVLS
  binding failures on this stack; decode keeps `NCCL_NVLS_ENABLE=1` for the
  non-CP TP4 decode side.

Probe NIXL plugins before considering an alternative plugin backend:

```bash
PLUGINS=UCX,LIBFABRIC COMPONENT=prefill \
  deploy/disagg_pd_r20/probe_nixl_plugins.sh
```

The probe creates NIXL plugin backends in-process and verifies `VRAM_SEG` support
without sending model traffic. On the current B200 image, both UCX and LIBFABRIC
can be created, so LIBFABRIC is a valid one-at-a-time NIXL plugin A/B candidate.
The current probe may emit LIBFABRIC `fi_close fabric failed ... Device or
resource busy` cleanup warnings; they are captured in `plugin_probe.stderr` and
are not throughput proof. This is not a promotion claim: run the strict smoke and
c16 throughput gate before selecting it. Render an unapplied LIBFABRIC variant
with:

```bash
deploy/disagg_pd_r20/render_nixl_plugin_variant.sh \
  --plugin LIBFABRIC \
  --output /tmp/topo-c1-dp2tp4-disagg-r20-nixl-libfabric.yaml

EXPECTED_NIXL_PLUGIN_BACKEND=LIBFABRIC \
LOCAL_DGD_MANIFEST=/tmp/topo-c1-dp2tp4-disagg-r20-nixl-libfabric.yaml \
NIXL_AUDIT_MODE=local \
  deploy/disagg_pd_r20/audit_nixl_gate_readiness.sh
```

After the audit passes, run the strict smoke. After strict smoke passes, run the
NIXL c16 gate before any UCX/Mooncake/MORI A/B:

```bash
SMC_GATE_MODE=deferred REQUIRE_DYNAMO_PIN_MARKERS=1 \
REQUIRE_POSITIVE_TRANSFER_METRICS=1 REQUIRE_ABORT_CLEANUP_MARKER=1 \
  deploy/disagg_pd_r20/smoke_request_pinning.sh

deploy/disagg_pd_r20/run_c16_transport_bench.sh \
  --backend nixl \
  --lengths 1024,4096,8192,16384,32768,65536,131072 \
  --concurrency 16 \
  --max-tokens 128 \
  --min-tok-per-user 150
```

## KV handoff shape

The deployment uses the TRT-LLM disaggregated KV transceiver, not vLLM MORI-IO.
The MORI-IO write-mode shape remains an A/B-phase reference only: prefill is the
KV producer, decode owns pre-allocated KV blocks, and transfer metadata must
describe block and layer layout precisely. The current r20 image ships both the
C++ NIXL wrapper (`libtensorrt_llm_nixl_wrapper.so`) and Python `nixl`, so the
pre-A/B gate pins `cache_transceiver_config.backend: NIXL` and
`layersplit_transfer_backend: nixl`. UCX remains available only as an A/B
comparison candidate. The gate is fail-closed: explicit YAML backend selection
wins over legacy `TRTLLM_USE_*_KVCACHE` environment toggles, conflicting UCX /
Mooncake / MPI env selectors are rejected, and the smoke requires startup logs
showing `Initializing NIXL Connect`, `cache_transceiver_config.backend=NIXL`,
`OPTRT_LAYERSPLIT_XFER_DEBUG`, and `global_layers=61` before it sends traffic.
LayerSplit owns the prefill-side CP-sharded DSA KV/indexer-K layout with
owner-local allocation, and the NIXL handoff reassembles those shards into the
decode worker's TP4/CP1 KV layout before decode generation.

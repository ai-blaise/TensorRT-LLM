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

Build only:

```bash
deploy/disagg_pd_r20/fast_iterate.sh \
  --vm 34.106.33.128 \
  --target-node a4-us-001-rl9 \
  --tag-suffix swapab-host
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
- `prewarm_caches.sh` -- persistent-cache preparation and lightweight offline HF
  prewarm/validation job.
- `smoke_request_pinning.sh` -- ready-only live gate for non-MORI request
  pinning, Moondream overlap compatibility, normal close, and early stream
  close cleanup before A/B.
- `cache_report.sh` -- read-only VM report for persistent-cache growth, local
  registry availability, and k3s/containerd image residency.
- `Dockerfile.r20-overlay.dockerignore` -- overlay-specific build-context
  allowlist so thin-image rebuilds do not ship the full repo to Docker/BuildKit.

Inspect cache/image residency without touching pods:

```bash
deploy/disagg_pd_r20/cache_report.sh \
  --vm 34.106.33.128 \
  --image-filter dynamo-trtllm-optrt-custom
```

Run this after the first cold rollout and again after the next overlay rollout.
The useful signal is whether `triton`, `cuda`, `deep_gemm`, and
`tensorrt_llm/*` grow and then stabilize; if they remain empty, the workers are
not writing to the intended persistent cache paths.

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

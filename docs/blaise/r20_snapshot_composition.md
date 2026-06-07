# R20 Snapshot Composition Plan

## Decision

Use `ai-blaise/criu-snapshots` as the first runtime fast-start composition path
for R20. Do not integrate Foundry wholesale into `op-trt` now.

Foundry remains useful as a design reference and a future graph-capture residual
accelerator, but its current TensorRT-LLM support is placeholder-only. The
actionable path today is to make R20 measurable and ready for the existing CRIU
snapshot substrate without changing the live `topo-c1-dp2tp4-disagg-r20` pods.

## Smaller Pieces Worth Adopting

From Foundry:

- Treat CUDA graph capture time as a separate startup residual, measured after
  image-cache, persistent-cache, prewarm, and any CRIU fast-start work.
- Keep graph replay fail-closed behind an explicit flag and source/image
  provenance. No default `LD_PRELOAD` and no live DGD patching.
- Require SAVE/LOAD allocation parity tests before graph replay is considered
  runtime-safe for TensorRT-LLM.

From `criu-snapshots`:

- Reuse the controller/daemon/CRD lifecycle instead of creating an `op-trt`-only
  snapshot format.
- Keep thin snapshots as the target shape: runtime state plus model identity,
  with model files rehydrated from `/models` / node-local cache.
- Preserve the fingerprint contract: GPU SKU/count/order, driver, glibc, image
  digest, model digest, CRIU, and `cuda-checkpoint` must match before restore.
- Add TensorRT-LLM pre-snapshot/post-restore hooks equivalent in spirit to the
  SGLang hooks before any DGDS `take` operation is allowed.

## Current R20 State

R20 already has the cache and image pieces that should compose with snapshots:

- `fast_iterate.sh` for shallow overlay sync, local registry handoff, containerd
  residency, optional prewarm, and deploy-gated DGD apply.
- `prewarm_caches.sh` and the DGD hostPath mount at `/var/lib/optrt-cache` for
  HF, XDG, pip, Torch extension, TorchInductor, Triton, CUDA JIT, DeepGEMM, and
  TRT-LLM generated-artifact caches.
- `cache_report.sh` for read-only cache and image residency reporting.
- `render_dgd.sh` for API-server validation with the active image, avoiding a
  build/import loop for deploy-YAML and router checks.
- `foundry_prepare.sh` for read-only Foundry source/build readiness reporting.

What R20 does not yet have is the application hook that makes a live
TensorRT-LLM worker safe to checkpoint. The current DGD runs
`python3 -m dynamo.trtllm`; it does not configure `SGLANG_SNAPSHOT_HOOKS`,
`OPTRT_SNAPSHOT_HOOKS`, or a TensorRT-LLM equivalent, and there is no proven
drain -> destroy NCCL/process groups -> checkpoint -> rebuild sequence for
TP4 decode or TP2xCP2 LayerSplit prefill.

## New Safe Probe

The detailed pass/fail criteria live in
[`r20_snapshot_proof_criteria.md`](r20_snapshot_proof_criteria.md).

`deploy/disagg_pd_r20/snapshot_readiness.sh` is a read-only preflight for this
composition path. It reports:

- host capability floors: kernel, NVIDIA driver, CRIU, `cuda-checkpoint`,
  `checkpointctl`, `buildah`, `oras`, and `/opt/criu-snapshots` installation;
- Kubernetes substrate: `DynamoGraphDeploymentSnapshot` CRD, snapshot namespace,
  controller/daemon references, existing DGDS resources, and target DGD presence;
- R20 live config shape: TensorRT-LLM runtime marker, persistent cache mount,
  absence/presence of snapshot hook env vars, and absence/presence of Foundry or
  `LD_PRELOAD` markers;
- the hard blockers: `trtllm_snapshot_hook_proof=missing`,
  `nixl_inflight_restore_proof=missing`,
  `layersplit_tp2cp2_restore_proof=missing`, and
  `kvarn_cuda_graph_scratch_restore_proof=missing` until canary restore tests
  prove those paths; `safe_to_take_snapshot=0` remains expected for production.

It does not create a `DynamoGraphDeploymentSnapshot`, patch a ConfigMap, drain
traffic, restart pods, or run a restore probe.

Run from the VM checkout:

```bash
deploy/disagg_pd_r20/snapshot_readiness.sh --vm local
```

Use strict mode in CI or operator gates when absence of the snapshot substrate
should fail the command:

```bash
deploy/disagg_pd_r20/snapshot_readiness.sh --vm local --strict
```

## Commit-Ready Next Patch After This Probe

The next runtime patch should be a gated TensorRT-LLM hook surface, not a
Foundry runtime integration. Minimum shape:

1. Add `OPTRT_SNAPSHOT_HOOKS=1` as an opt-in only; leave the R20 DGD default off.
2. Register `SIGRTMIN+5` pre-snapshot and `SIGRTMIN+6` post-restore handlers in
   the TensorRT-LLM worker process before serving traffic.
3. Pre-snapshot: detach/drain routing, stop accepting new work, wait for
   in-flight requests to complete or abort, synchronize CUDA, quiesce NIXL
   send/receive tasks, destroy torch/TRT-LLM process groups, and write a ready
   file for the snapshot agent.
4. Post-restore: rebuild process groups, reconnect NIXL, reattach routing,
   prove one local health/first-token path, then write a restore-ready file.
5. Add offline unit tests for idempotent signal handlers and a canary-only DGDS
   dry run before any production `take` command is documented.

Only after that hook passes TP4 decode and TP2xCP2 LayerSplit prefill can the
infra wrapper safely run a real `DynamoGraphDeploymentSnapshot` against R20.

## Remaining No-Go Conditions

- Foundry TensorRT-LLM integration is still placeholder-only upstream.
- R20 has no TensorRT-LLM pre-snapshot/post-restore hook.
- NIXL in-flight transfer cleanup is not proven across snapshot/restore.
- LayerSplit owner-local CP-sharded prefill state has no restore parity test.
- Dense KVarN and CUDA graph capture buffers have no persistent-scratch restore
  proof.
- `checkpointctl` is still required before restore-image materialization / Tier 3
  restored-Pod launch can be considered complete.
- No R20 first-token restore benchmark exists to compare CRIU fast-start against
  cold start plus current image/cache/prewarm work.

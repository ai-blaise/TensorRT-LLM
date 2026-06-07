# Foundry / CRIU Runtime-Reuse Decision

## Decision

Do not integrate Foundry as a standalone `op-trt` runtime feature yet.

The useful near-term artifact for `op-trt` is a read-only Foundry readiness
probe plus this comparison document. Live `LD_PRELOAD`, Foundry TOML config,
SAVE/LOAD graph replay, and DGD manifest changes are explicitly out of scope
until TensorRT-LLM has a real Foundry hook surface and until the existing
`ai-blaise/criu-snapshots` fast-start path has live workload restore probes for
the relevant serving stack.

The likely long-term shape is composition, not replacement:

- `criu-snapshots` remains the process/container/runtime fast-start substrate.
- Foundry is a possible graph-capture accelerator inside that restored runtime,
  but only after a TensorRT-LLM integration can prove deterministic allocation,
  CUDA graph replay correctness, and NIXL/LayerSplit compatibility.

## Sources Read Directly

Foundry:

- Repo: `https://github.com/foundry-org/foundry`
- Audited commit on the VM: `eef12012aa0f85ae6079891144797b08c282152d`
- Paper: `https://arxiv.org/pdf/2604.06664`, downloaded on the VM to
  `/var/lib/optrt-cache/foundry-paper/2604.06664.pdf`

ai-blaise comparison inputs:

- `ai-blaise/criu-snapshots`, default branch `main`, private GitHub repo read
  via the GitHub connector because the VM has no GitHub credentials.
- `ai-blaise/infrastructure`:
  `scripts/dynamo-reap/deploy-a4-snapshots.sh`.
- VM worktrees:
  `/tmp/tensorrt-llm-op-trt-clean`, `/home/spencer/work/TensorRT-LLM-op-trt-ls`,
  `/home/spencer/optimization-playground`, and `/tmp/foundry-org-foundry-audit`.
- `ai-blaise/sglang` `main` was checked for `SGLANG_SNAPSHOT_HOOKS` / CRIU
  hook files; those hook files were not present there as of this audit. The
  hook source available on the VM is the `optimization-playground` runtime tree
  referenced by the `criu-snapshots` Tier-4 notes.

## What Foundry Accelerates

Foundry persists CUDA graph topology plus the execution context needed to replay
those graphs in a fresh process. The relevant implementation pieces are:

- deterministic CUDA VMM allocation at a fixed base address;
- CUDA module/library load interception so kernel binaries and entry points can
  be serialized and reloaded;
- CUDA graph serialization plus topology grouping, where one template graph can
  serve many batch-size variants via node-parameter updates;
- engine hooks that install before CUDA allocation and before child worker spawn,
  usually with `LD_PRELOAD=libcuda_hook.so`.

The paper and repo show the target benefit: skipping repeated CUDA graph warmup
and capture. The repo README demonstrates vLLM restoring 256 graphs in about 1
second versus about 30 seconds of baseline graph warmup/capture, and the paper
reports up to 99% cold-start reduction on vLLM workloads.

That is relevant to `op-trt` only if TensorRT-LLM exposes a matching graph
capture/load seam. It currently does not.

## Current Foundry TRT-LLM Status

The upstream Foundry repo marks TensorRT-LLM as not ready:

- `README.md` status table: TensorRT-LLM is in-progress for single GPU, DP, TP,
  and EP.
- `docs/trtllm/overview.md`: coming soon; design notes are not written.
- `recipe/trtllm/README.md`: coming soon; serve scripts and integration code
  have not landed.
- `python/foundry/integration/trtllm/__init__.py`: placeholder package only.

Foundry also has nontrivial build requirements. The audited repo requires
CMake 4.0+, Boost 1.83+, CUDA driver 12.0+, PyTorch 2.9+, and Python >=3.10.
The main VM currently reports CMake 3.26.5, so even a wheel build is not a
simple drop-in without dependency preparation.

## Existing ai-blaise Runtime-Reuse Mechanisms

### op-trt r20 iteration path

`deploy/disagg_pd_r20/` already has safe iteration-speed tools:

- `fast_iterate.sh`: syncs the shallow overlay subset, builds on the B200 VM,
  uses k3s containerd when available, can use a VM-local registry, and supports
  prewarm/deploy gates.
- `Dockerfile.r20-overlay.dockerignore`: installed by Carson path before remote
  build to avoid hashing/sending the full repo for thin overlay work.
- `prewarm_caches.sh`: prepares `/var/lib/optrt-cache` and runs a lightweight
  offline model/import prewarm job.
- `cache_report.sh`: read-only report for persistent cache, local registry, and
  containerd image residency.
- The DGD mounts `/var/lib/optrt-cache` as `/cache/optrt`, covering HF modules,
  Transformers/HF datasets metadata, XDG, pip, Torch extensions, TorchInductor,
  Triton, CUDA JIT, DeepGEMM, and TRT-LLM generated artifacts.

Those mechanisms target code/build/cache iteration. They do not snapshot a live
process and do not replace CUDA graph capture in a fully cold worker.

### criu-snapshots

`ai-blaise/criu-snapshots` is a real controller/daemon stack for host-kernel
CRIU plus NVIDIA `cuda-checkpoint` under containerd/runc. Its relevant shape:

- CRD `DynamoGraphDeploymentSnapshot` snapshots prefill or decode ranks of a
  DynamoGraphDeployment and pushes one OCI artifact per rank.
- Restore pulls snapshots to local NVMe, validates a fingerprint, materializes a
  checkpoint image, and lets Kubernetes start the restored Pod.
- Thin snapshot mode records model identity and rehydrates model files on
  restore instead of copying model weights into the CRIU payload.
- The production wrapper is
  `ai-blaise/infrastructure/scripts/dynamo-reap/deploy-a4-snapshots.sh`, with
  `take`, `promote`, `restore-verify`, and `rotate` operations.

Its hard constraints matter for Foundry:

- NCCL is not snapshottable; app hooks must destroy and rebuild process groups.
- GPU SKU/count/slot order, CPU flags, glibc/runtime image digest, driver, CRIU,
  and `cuda-checkpoint` version are fingerprinted and must match on restore.
- CUDA graphs captured against weight tensors do not survive the weight-offload
  pattern; the CRIU docs require graph capture against persistent scratch
  buffers only and weight H2D reload in `post_restore`.
- Tier 3 controller/agent/OCI/fingerprint restore mechanics are validated on
  B200/k3s. Tier 4 still needs full live first-token/training-step promotion for
  production workloads.

### optimization-playground / ai-blaise SGLang

The local `/home/spencer/optimization-playground` tree contains the SGLang-side
snapshot hooks that `criu-snapshots` expects:

- `python/sglang/srt/snapshot_hooks.py`: installs `SIGRTMIN+5` pre-snapshot and
  `SIGRTMIN+6` post-resume handlers when `SGLANG_SNAPSHOT_HOOKS=1`; drains the
  KV router, drains in-flight work, destroys NCCL, empties CUDA cache, then
  rebuilds NCCL and reattaches the router after restore.
- `python/sglang/srt/criu_multiprocessing.py`: preserves POSIX semaphore names
  for CRIU when enabled.
- `docker/sglang-criu-entrypoint`: restores from `/snap/img` when a CRIU image is
  present, then signals post-restore readiness.

It also carries other state-reuse mechanisms, such as HiCache and checkpoint
engine weight loading. Those are SGLang-specific and complementary to CRIU;
they do not provide a TensorRT-LLM Foundry hook surface.

## Comparison

| Mechanism | Saves | Helps | Current status | `op-trt` fit now |
|---|---|---|---|---|
| r20 overlay + local registry | Image layers and build context | Python/config/image iteration | Active, safe, Carson optimized context | Keep using; do not duplicate |
| `/var/lib/optrt-cache` + prewarm | Import/JIT/HF/Triton/Inductor caches | Pod restart and image rollout warmup | Active in r20 manifests/scripts | Keep using |
| `criu-snapshots` | Process/container/CUDA runtime state as OCI artifacts | Fast-start after validated snapshot | Validated substrate; live workload Tier 4 pending | Best existing fast-start direction, but not wired for op-trt TensorRT-LLM |
| Foundry | CUDA graphs + CUDA module execution context | CUDA graph warmup/capture skip | vLLM/SGLang real; TRT-LLM placeholder | Not standalone-ready |

The key distinction: CRIU snapshots are broad but fragile around NCCL, topology,
driver, and CUDA graph/weight-offload constraints. Foundry is narrower and more
portable in theory, but it needs engine-specific hooks at exactly the places
TensorRT-LLM has not yet been adapted.

## Recommended Path

1. Do not ship Foundry runtime integration in `op-trt` now.
2. Keep `foundry_prepare.sh`, if used, as a readiness/audit probe only:
   clone the audited Foundry commit, report prerequisites, report whether
   upstream TRT-LLM support is still placeholder, and never set `LD_PRELOAD` or
   mutate live pods.
3. Prioritize existing pre-A/B gates: NIXL KV transfer, LayerSplit owner-local
   correctness, request/Moondream pinning proof, and dense 2-bit KVarN.
4. For runtime fast-start, compare against `criu-snapshots` first once live
   first-token restore probes exist for the target stack. A Foundry patch should
   be justified only if measured startup breakdown shows CUDA graph capture is
   still a dominant residual after image/cache/prewarm and any CRIU fast-start
   path.
5. If Foundry becomes justified, integrate as a gated composition layer:
   fail-closed config flag, rank-local workspace, source/image provenance,
   no default `LD_PRELOAD`, no DGD mutation outside an explicit deploy script,
   and correctness tests for CUDA graph replay, NIXL, LayerSplit, request
   pinning, dense KVarN, and abort/rollback.

## No-Go Conditions For Immediate Implementation

Foundry should stay disabled while any of these remain true:

- upstream TensorRT-LLM Foundry integration is placeholder-only;
- the VM cannot build Foundry without dependency changes such as CMake 4.0+ and
  Boost 1.83+;
- no TensorRT-LLM hook is proven to install before CUDA allocation and worker
  spawn;
- no test proves SAVE/LOAD allocation parity for TP4 decode and TP2xCP2
  LayerSplit prefill;
- no test proves Foundry graph replay composes with NIXL KV transfer and the
  r20 request-pinning protocol;
- live workload snapshots in `criu-snapshots` are still pending for the relevant
  serving path.

## Integration-Ready Patch Status

This branch intentionally does not ship a runtime Foundry integration. It ships
an integration-ready decision artifact from the exact `op-trt` base below and
validates the already-present safe readiness probe:

- Repository: `https://github.com/ai-blaise/TensorRT-LLM.git`
- Base branch/commit: `op-trt` at `b5d317c4fdc259d20afbba749d51c74761a767db`
- Audit branch: `anscombe/foundry-snapshot-audit`
- Changed paths in this audit commit:
  - `docs/blaise/foundry_iteration_speed.md`
  - `docs/blaise/README.md`
- Existing gated probe validated by this audit:
  - `deploy/disagg_pd_r20/foundry_prepare.sh`

The readiness probe is acceptable to keep because it is gated, reports
`ld_preload_modified=0` and `live_workload_modified=0`, and never applies a DGD
or restarts pods. Treat it like `cache_report.sh`: an operator visibility tool,
not a deployment mechanism.

## Proof Commands

Safe commands used for this audit:

```bash
# op-trt branch/base proof
cd /tmp/tensorrt-llm-op-trt-clean
git rev-parse HEAD
git branch --show-current
git status --short

# Foundry direct-source proof
cd /var/lib/optrt-cache/foundry/src
git rev-parse HEAD
sed -n '1,120p' docs/trtllm/overview.md
sed -n '1,80p' recipe/trtllm/README.md
sed -n '1,40p' python/foundry/integration/trtllm/__init__.py

# Foundry paper direct-read proof
ls -lh /var/lib/optrt-cache/foundry-paper/2604.06664.pdf

# Readiness probe proof, no pod mutation
cd /tmp/tensorrt-llm-op-trt-clean
bash -n deploy/disagg_pd_r20/foundry_prepare.sh
deploy/disagg_pd_r20/foundry_prepare.sh --help
deploy/disagg_pd_r20/foundry_prepare.sh --vm local --mode check

# Live config guardrail, read-only
kubectl get cm topo-c1-dp2tp4-disagg-r20-config -n dynamo-system -o yaml \
  | grep -Ei 'foundry|LD_PRELOAD|graph_extension' -C 2
```

Expected readiness-probe signal on the current VM:

```text
trtllm_integration=placeholder
cmake_version=3.26.5
boost_status=not_found_in_ldconfig
ld_preload_modified=0
live_workload_modified=0
```

No output from the live config grep means the active r20 DGD config does not
contain Foundry or `LD_PRELOAD` markers.

## Residual Gaps

This audit is complete enough to block wholesale Foundry integration now. The
remaining work before any runtime integration is external to this patch:

- upstream Foundry TensorRT-LLM hooks or an equivalent local TRT-LLM hook design;
- CMake 4.0+ / Boost 1.83+ / clean Python+Torch Foundry build environment on the
  B200 VM or in a dedicated build image;
- a TensorRT-LLM SAVE/LOAD parity test for TP4 decode and TP2xCP2 LayerSplit
  prefill;
- proof that Foundry graph replay composes with NIXL KV transfer, request
  pinning, dense KVarN, and abort/cleanup paths;
- `criu-snapshots` live first-token restore proof for the target serving stack,
  so Foundry can be compared as a residual graph-capture accelerator rather than
  as a competing fast-start substrate;
- startup breakdowns showing CUDA graph capture remains a material residual
  after image-cache, persistent-cache, prewarm, and snapshot fast-start work.

## Safe Artifact Delivered Here

This document is the concrete no-go/composition artifact. The only safe script
surface is the read-only `deploy/disagg_pd_r20/foundry_prepare.sh` probe. It is
not a Foundry integration and must not be used to infer runtime readiness.

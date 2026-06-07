# R20 Snapshot Proof Criteria

## Scope

This is the proof checklist for composing `ai-blaise/criu-snapshots` with the
R20 TensorRT-LLM deployment. Foundry remains sidecar-only until these restore
proofs pass and a startup breakdown shows CUDA graph capture is still a dominant
residual after image/cache/prewarm/CRIU work.

No item here requires mutating the live `topo-c1-dp2tp4-disagg-r20` deployment
while it is serving traffic. The order is deliberate: prove hooks and local
restore mechanics before any production DGDS `take`.

## Gate -1: Cached Deploy Loop

This gate is not a snapshot proof. It is the fast path for deployment/config
iterations that should not pay build, image import, model download, or compile
costs.

Pass criteria:

- `render_dgd.sh --image-from-dgd topo-c1-dp2tp4-disagg-r20 --dgd-name <short-canary> --server-dry-run`
  succeeds without applying anything.
- The rendered image matches the active DGD image and `imagePullPolicy` is
  explicit for the intended path (`IfNotPresent` for local registry/resident
  images, `Never` for strictly preloaded images).
- `check_image_handoff.sh --vm local --image-from-dgd topo-c1-dp2tp4-disagg-r20 --mode auto --require`
  reports `handoff_ready=yes` and a recommended pull policy before prewarm or
  deploy spends time on the wrong image handoff path.
- `cache_report.sh --vm local` shows the active image is resident in k3s
  containerd and persistent cache directories exist.
- `prewarm_caches.sh --dry-run` renders the cache prewarm Job for the same image
  before an operator runs a real prewarm.

This gate is the highest-ROI non-disruptive path today: it turns router/YAML/DGD
checks into API-server validation against an already-resident image instead of a
new build/import/download/compile loop.

## Gate 0: Sidecar Substrate

Pass criteria:

- `snapshot_readiness.sh --vm local --run-host-preflight` reports
  `host_preflight_result=ok`.
- `dgds_crd=present`, `snapshot_controller_refs` is non-empty, and
  `snapshot_daemon_refs` is non-empty.
- `criu_status=present`, `cuda_checkpoint_status=present`, and
  `oras_status=present`.
- `checkpointctl_status=present` before restore-image materialization or Tier 3
  restored-Pod launch is considered complete. If it is absent, snapshot take may
  be inspectable, but restore-image proof is incomplete.

Current R20 signal: substrate is mostly present; `checkpointctl` is still the
known restore-tooling warning on `a4-us-001-rl9`.

## Gate 1: TensorRT-LLM Snapshot Hooks

Required patch shape:

- Opt-in only through `OPTRT_SNAPSHOT_HOOKS=1`; do not enable it in the live R20
  DGD by default.
- Register `SIGRTMIN+5` as pre-snapshot and `SIGRTMIN+6` as post-restore in the
  actual TensorRT-LLM worker process before serving traffic.
- Pre-snapshot handler drains routing, stops new work, waits for or aborts
  in-flight requests, synchronizes CUDA, quiesces NIXL sender/receiver work,
  destroys Torch/TRT-LLM process groups, and writes a per-pid ready file.
- Post-restore handler rebuilds process groups, reconnects NIXL, reattaches
  routing, proves a local health or first-token path, and writes a restore-ready
  file.

Current implementation status:

- `tensorrt_llm/_torch/pyexecutor/snapshot_hooks.py` provides the first
  opt-in signal substrate and is installed from `PyExecutor` only when
  `OPTRT_SNAPSHOT_HOOKS=1`.
- The current pre-snapshot hook blocks new queue admission, waits for no active
  requests plus no async transfer-manager work, checks generation transfer
  completion when the transceiver exposes it, synchronizes CUDA, and writes
  per-pid JSON proof files.
- The current post-restore hook synchronizes CUDA, reopens queue admission, and
  writes per-pid JSON proof files.
- This does not yet satisfy the full restore gate: process-group teardown and
  rebuild, explicit NIXL reconnect proof, LayerSplit/KVarN restore proof, and
  first-token restore proof remain required before `safe_to_take_snapshot=1`.

Proof commands:

```bash
# Offline/unit proof, no live pod mutation.
python3 -m pytest tests/unittest/disaggregated/test_snapshot_hooks.py

# Render-only canary validation. This does not apply or mutate a live DGD.
deploy/disagg_pd_r20/render_dgd.sh \
  --image-from-dgd topo-c1-dp2tp4-disagg-r20 \
  --dgd-name topo-c1-dp2tp4-hook-canary \
  --enable-snapshot-hooks \
  --snapshot-hook-proof-dir /tmp/optrt-snapshot-hooks-canary \
  --server-dry-run

# Guarded helper for the same default-safe render path. Add --apply only after
# it reports the canary is absent and the target node's GPUs are idle enough for
# an isolated canary.
deploy/disagg_pd_r20/snapshot_hook_canary.sh \
  --source-dgd topo-c1-dp2tp4-disagg-r20 \
  --canary-dgd topo-c1-dp2tp4-hook-canary \
  --target-node a4-us-001-rl9

# Canary readiness proof after the hook-enabled DGD has generated proof files.
deploy/disagg_pd_r20/snapshot_readiness.sh \
  --vm local \
  --dgd-name topo-c1-dp2tp4-hook-canary \
  --hook-proof-dir /tmp/optrt-snapshot-hooks-canary \
  --strict
```

Pass criteria:

- Signal handlers are idempotent and timeout-bounded.
- Ready/error files include pid, rank, component, and phase.
- `snapshot_readiness.sh` reports `OPTRT_SNAPSHOT_HOOKS_configured=1` only for a
  canary DGD, not the live production DGD.
- `snapshot_readiness.sh --hook-proof-dir <dir>` sees both
  `optrt_snapshot_*_pre_snapshot.ready.json` and
  `optrt_snapshot_*_post_restore.ready.json` without matching error files.

## Gate 2: NIXL In-Flight Restore

Required proof:

- Start a canary R20 DGD with NIXL selected and request pinning enabled.
- Drive prefill-to-decode traffic with at least one normal completion and one
  early client close while the hook dry-run path drains.
- Prove no pending NIXL futures, no orphan request ids, and no decode-side block
  ownership leak after post-restore.

Pass criteria:

- Existing NIXL/request-pinning smoke still passes before and after the hook
  dry-run.
- Hook logs show NIXL quiesce before checkpoint and NIXL reconnect after restore.
- No in-flight transfer is silently completed for a cancelled request.

## Gate 3: LayerSplit TP2xCP2 Restore

Required proof:

- Prefill canary uses `tensor_parallel_size: 2`, `context_parallel_size: 2`, and
  `cp_config.cp_type: LAYERSPLIT`.
- Decode canary remains TP4/CP1.
- Snapshot/restore preserves owner-local CP-sharded DSA KV/indexer-K metadata and
  reassembles into decode TP4/CP1 layout after restore.

Pass criteria:

- `layersplit_tp2cp2_configured=1` in `snapshot_readiness.sh` for the canary.
- Post-restore transfer proof reports the expected global layer count and no
  HELIX/UCX fallback.
- Sparse attention output proof matches the non-restored canary path within the
  existing LayerSplit tolerance.

## Gate 4: Dense KVarN And CUDA Graph Scratch Restore

Required proof:

- Dense MLA latent KV remains `kvarn_k2v2`; Indexer K remains separate.
- Any CUDA graph captured before snapshot is captured against persistent scratch
  buffers, not transient weight-offload tensors.
- Post-restore reloads or rebinds weight-dependent buffers before first token.

Pass criteria:

- KVarN decode/read proof passes before and after restore.
- CUDA graph replay either remains disabled across restore or proves scratch-only
  graph capture with explicit buffer identity checks.
- No Foundry `LD_PRELOAD` or graph replay is introduced until this CRIU restore
  proof says CUDA graph capture is the remaining startup bottleneck.

## Gate 5: First-Token Restore Benchmark

Required proof:

- Measure cold start with the existing image/cache/prewarm path.
- Measure CRIU restore time-to-Ready and time-to-first-token on an isolated
  canary DGD.
- Only then compare Foundry as a possible residual CUDA graph accelerator.

Pass criteria:

- Report p50/p95/p99 time-to-Ready and time-to-first-token.
- Include image tag, DGD name, snapshot artifact tag, model digest, driver,
  CRIU/cuda-checkpoint versions, and node fingerprint.
- Do not promote production snapshot commands until canary restore is repeatable.

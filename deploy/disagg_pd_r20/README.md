# Disagg P/D r20 deploy manifest -- cell1 DP2/TP4-disagg winner

Disaggregated prefill/decode Dynamo deploy for the production STARTING topology,
with the custom pieces toggled ON.

## Topology (single 8xB200 node, a4-us-002-rl9 / k3s)

| Worker   | GPUs  | Parallelism            | Custom piece ON                              |
|----------|-------|------------------------|----------------------------------------------|
| prefill  | 4 GPUs | TP2xCP2 / EP2, ADP=false | **LayerSplit** (`layersplit_enabled: true`) |
| decode   | 4 GPUs | TP4 / EP4, ADP=false | **WarpDecode** (`warp_decode.enabled: true`, `tile_mode: decode_1cta`) + **SMC-SD** (`speculative_config.decoding_type: SMC`) |
| Frontend | -     | KV router (`--router-mode kv`) | -                                  |

This is 1P x 4GPU + 1D x 4GPU disaggregated serving with real LayerSplit on
prefill (`TP2 x CP2`) and non-CP decode (`TP4 x CP1`).

## Why node 002 (k3s)

The disaggregated P/D artifact is the `DynamoGraphDeployment` CRD
(`nvidia.com/v1alpha1`), which is k3s-native and is the pattern the validated
`topo-c1-dp2tp4-disagg-smc` deploy already used on a4-us-002-rl9. The 001 docker
path (`deploy/smcsd_fiport/r12_001/smc_launch_001.sh`) is **aggregated** (a single
`trtllm-serve`, no prefill/decode split), so it is not the disagg P/D path.

## Image

Both workers + frontend run the **unified wins+SMC image**, parameterized as
`${UNIFIED_IMAGE}`. The build agent (branch `op-trt-canonical-smc-r20`, build
container `r20-unified-build`, `FROM
local/dynamo-trtllm-optrt-custom:canonical-r17-wins-20260605`) sets the final tag.

**Expected tag (confirm with the build agent before apply):**
`local/dynamo-trtllm-optrt-custom:canonical-smc-r20-cpfix-20260605`

## Deploy (orchestrator only -- gated)

```bash
export UNIFIED_IMAGE=local/dynamo-trtllm-optrt-custom:canonical-smc-r20-cpfix-20260605   # from build agent
envsubst '$UNIFIED_IMAGE' < topo-c1-dp2tp4-disagg-r20.yaml | \
  KUBECONFIG=/etc/rancher/k3s/k3s.yaml k3s kubectl apply -f -
```

`hf-token-secret` (namespace `dynamo-system`) and a `/models` host directory must
exist on a4-us-002-rl9.

## Files

- `topo-c1-dp2tp4-disagg-r20.yaml` -- the deployable manifest: a `ConfigMap`
  (`topo-c1-dp2tp4-disagg-r20-config`, holds `prefill.yaml` + `decode.yaml`) plus
  the `DynamoGraphDeployment` (`topo-c1-dp2tp4-disagg-r20`). Apply this one file.
- `prefill.yaml` / `decode.yaml` -- standalone copies of the two engine configs
  (identical to the ConfigMap data blocks) for review / diff / reuse.

## Knob provenance

- WarpDecode: `tensorrt_llm/llmapi/llm_args.py` `WarpDecodeConfig`
  (`enabled`/`policy`/`tile_mode`), `docs/blaise/warpdecode.md`. Fixed-tactic
  overlay is default-on via env `TRTLLM_WARP_DECODE_FIXED_TACTIC=1` (set on the
  decode worker) + `TRTLLM_ENABLE_PDL=1`.
- LayerSplit: `BaseSparseAttentionConfig.layersplit_*`,
  `docs/source/features/layersplit.md`.
- SMC-SD: `SMCDecodingConfig`, `docs/blaise/smc_sd.md`. Decode env needs
  `NCCL_NET_PLUGIN=none` (per `deploy/smcsd_fiport/r12_001/smc_launch_001.sh`).
- Base manifest pattern: `deploy/smcsd_fiport/dgd_smc_on.yaml` +
  `deploy/smcsd_fiport/smc_configmap.yaml` (the validated `topo-c1-dp2tp4-disagg-smc`).

## LayerSplit TP2xCP2 prefill

LayerSplit splits the DSA KV / indexer-K cache across **context-parallel** ranks.
This manifest runs the prefill worker as TP2 x CP2 on four GPUs
(`tensor_parallel_size: 2`, `context_parallel_size: 2`) so LayerSplit is a real
CP split. There is **no** `--context-parallel-size` Dynamo CLI flag; CP is set
only through the engine YAML `context_parallel_size` field. Decode remains
TP4/CP1 and consumes the reassembled KV through the cpfix image.

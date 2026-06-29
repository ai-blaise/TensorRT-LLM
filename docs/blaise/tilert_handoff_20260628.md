<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Blaise TileRT / Resident Decode Handoff

This is the handoff for the `op-trt-tilert` branch as of the 2026-06-28/29 UTC
benchmark pass. The short version is:

- The currently proven best serving configuration is still
  `.bench_runs_claude/BEST_CONFIG`, deployed by
  `.bench_runs_claude/deploy_arm.py vbfuse_nvfp4_mega_cf`.
- The resident / TileRT-style execution path is real infrastructure now, but it
  is not the best config. It is several times slower than the frozen production
  path at C16/C32 and still has startup/readiness fragility.
- The next developer should treat the resident path as an execution-model
  workstream, not as a pile of local fusion wins. CUDA graphs and isolated
  projection fusions are not enough.

## Current Live Best

The live cluster was restored to the frozen best after the ablation pass.

Topology:

- DGD: `topo-c1-dp2tp4-disagg-r20`
- Node: `a4-us-002-rl9`
- Frontend: `http://10.42.0.165:8000`
- Decode: TP4 + EP4 + attention-DP, GPUs 4-7
- Prefill: TP2 + EP4, context parallel layersplit path available in the config
- Model:
  `/models/BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft`

Best image, used by both decode and prefill:

```text
localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-vbfuse-nvfp4-20260615T043037Z
```

Load-bearing decode env:

```text
TRTLLM_OPTRT_MOE_MEGAKERNEL=1
TRTLLM_FORCE_COMM_METHOD=NVLINK_TWO_SIDED
ENABLE_CONFIGURABLE_MOE=1
TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER=1
TRTLLM_ENABLE_PDL=1
```

Decode config essentials:

```yaml
enable_attention_dp: true
allreduce_strategy: MNNVL
max_batch_size: 64
kv_cache_config:
  dtype: fp8
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.45
  tokens_per_block: 64
cache_transceiver_config:
  backend: NIXL
  transceiver_runtime: PYTHON
moe_config:
  backend: WARPDECODE
  use_low_precision_moe_combine: false
  warp_decode:
    enabled: true
    tile_mode: decode_1cta
    policy: force
    max_batch_size: 64
sparse_attention_config:
  algorithm: dsa
  indexer_mode: indexcache-hisa
  index_topk: 1024
  index_topk_pattern: FSSS
  indexer_k_dtype: fp4
  mla_latent_kv_dtype: kvarn_k2v2
  enable_nvfp4_hisa: true
  hisa_min_seq_len: 65536
```

Deploy recipe:

```bash
cd /home/sjpat/TensorRT-LLM
python3 .bench_runs_claude/deploy_arm.py vbfuse_nvfp4_mega_cf
```

The config source of truth is `.bench_runs_claude/BEST_CONFIG/`. The most
important file there is `BEST_CONFIG.md`; it includes the original OSL512
validated curve and the restore recipe.

## Fresh Ablation Results

Benchmark harness:

```bash
docker cp .bench_runs_claude/e2e_sweep.py optrt-bench-claude:/tmp/e2e_sweep.py
docker exec optrt-bench-claude /opt/dynamo/venv/bin/python3 -u /tmp/e2e_sweep.py ...
```

Streaming completions were used. TTFT is first non-empty streamed chunk. Per-user
tok/s is the harness steady-rate estimate from usage tokens. The first run after
a full image rollback can include CUDA graph/tactic warmup; use warm repeats for
steady comparisons.

### C16/C32 Fast Ablation

Shape: `OSL=128`, `ISL ~= 4161`, 32 requests per point.

| Variant | C | TTFT p50 ms | tok/user/s p50 | agg tok/s | Verdict |
|---|---:|---:|---:|---:|---|
| Resident native + fused KVA/WK/WP, cold | 16 | 93414 | 3.93 | 27.6 | Not competitive |
| Resident native + fused KVA/WK/WP, cold | 32 | 4433 | 6.56 | 226.0 | Not competitive |
| Resident native + fused KVA/WK/WP, warm | 16 | 2448 | 7.61 | 147.4 | Not competitive |
| Resident native + fused KVA/WK/WP, warm | 32 | 4379 | 6.98 | 241.1 | Not competitive |
| Resident native image with KVA/QB fusions disabled | - | - | - | - | Failed to become ready within the ablation window; startup probe stayed 503 |
| Frozen best, cold | 16 | 22465 | 41.24 | 122.2 | Warmup polluted C16 |
| Frozen best, cold | 32 | 4679 | 40.16 | 451.6 | Competitive |
| Frozen best, warm | 16 | 2093 | 39.88 | 359.8 | Current best |
| Frozen best, warm | 32 | 4489 | 37.99 | 439.4 | Current best |

Result files:

- `.bench_runs_claude/results/20260628_ablate_resident_kva_wkwp_c16_c32_128.txt`
- `.bench_runs_claude/results/20260628_ablate_resident_kva_wkwp_c16_c32_128_warm.txt`
- `.bench_runs_claude/results/20260628_deploy_resident_base_no_qb_no_kva.txt`
- `.bench_runs_claude/results/20260628_wait_resident_base_no_qb_no_kva.txt`
- `.bench_runs_claude/results/20260628_ablate_best_config_c16_c32_128.txt`
- `.bench_runs_claude/results/20260628_ablate_best_config_c16_c32_128_warm.txt`

### Fresh Best-Config Curve

Shape: `OSL=128`, `ISL ~= 4161`, warm production-best config.

| C | reqs | TTFT p50 ms | TTFT p95 ms | tok/user/s p50 | agg tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 645 | 684 | 35.48 | 38.1 |
| 2 | 8 | 699 | 1261 | 34.95 | 70.0 |
| 4 | 8 | 1132 | 1341 | 43.78 | 148.6 |
| 8 | 16 | 1168 | 2083 | 41.92 | 262.7 |
| 16 | 32 | 1981 | 3736 | 39.72 | 360.5 |
| 24 | 48 | 2095 | 4807 | 40.38 | 503.4 |
| 32 | 64 | 3163 | 5828 | 40.07 | 569.0 |
| 48 | 96 | 6392 | 8827 | 40.95 | 583.2 |
| 64 | 128 | 8476 | 11159 | 39.83 | 634.9 |

Result file:

- `.bench_runs_claude/results/20260628_best_config_full_c1_c64_osl128.txt`

### OSL512 Reconfirm

Shape: `C=32`, `OSL=512`, `ISL ~= 2055`, 64 requests.

| C | TTFT p50 ms | TTFT p95 ms | tok/user/s p50 | agg tok/s |
|---:|---:|---:|---:|---:|
| 32 | 1435 | 3564 | 50.42 | 1382.9 |

This matches the documented `BEST_CONFIG` class. The historical confirmed
numbers were about 50-51 tok/user/s and 1386-1425 aggregate tok/s at C32/OSL512.

Result file:

- `.bench_runs_claude/results/20260628_best_config_c32_osl512_reconfirm.txt`

## What Has Been Built

The branch has three broad workstreams.

### 1. Production best / Spencer documentation line

The proven serving stack is documented in:

- `.bench_runs_claude/BEST_CONFIG/BEST_CONFIG.md`
- `.bench_runs_claude/BEST_CONFIG/configmap.yaml`
- `.bench_runs_claude/BEST_CONFIG/dgd.yaml`
- `.bench_runs_claude/BEST_CONFIG/REGIME_CONFIGS.md`

Important conclusions from that line:

- `MEGAKERNEL=1` plus `use_low_precision_moe_combine=false` is the only
  validated serving-side throughput win in the frozen production path.
- `NVLINK_TWO_SIDED` is part of the best config.
- SMC speculative decoding is intentionally disabled. The documented draft path
  was a regression because the draft model path serialized behind expensive
  Python-dispatched work. Do not re-enable SMC by default.
- DeepEP-LL is not a viable lever on this deployment. Do not spend handoff time
  trying to make DeepEP the next step here.
- The older `REGIME_CONFIGS.md` latency/throughput pool findings are useful
  context but are not the current single production best. Revisit them only as a
  deliberate multi-pool / SLA-routing project.

### 2. TileRT-style resident execution infrastructure

The branch adds a resident decode execution path underneath the PyTorch backend.
Key files:

- `tensorrt_llm/_torch/pyexecutor/persistent_decode_engine.py`
- `tensorrt_llm/_torch/pyexecutor/persistent_decode_model_backend.py`
- `tensorrt_llm/_torch/pyexecutor/persistent_decode_planner.py`
- `tensorrt_llm/_torch/pyexecutor/persistent_decode_profiler.py`
- `tensorrt_llm/_torch/pyexecutor/persistent_decode_window.py`
- `tensorrt_llm/_torch/pyexecutor/deepseek_resident_native.py`
- `tensorrt_llm/_torch/pyexecutor/model_engine.py`
- `tensorrt_llm/_torch/pyexecutor/py_executor.py`
- `tensorrt_llm/_torch/models/modeling_deepseekv3.py`
- `cpp/tensorrt_llm/thop/deepseekResidentDecodeOp.cpp`

The intent is a TileRT-like decode owner:

- stable resident tensor table from the TRT-LLM-loaded model;
- per-layer semantic tensor-site binding instead of hot-path string lookup;
- persistent scratch for hidden states, logits, DSA intermediates, routing, and
  layer state;
- native resident model body handoff below `ModelEngine.forward`;
- strict backend modes that fail closed if model body, sampler, or window body
  are not native.

What is active in the resident branch:

- resident native model/body handoff;
- native window body boundary;
- native attention metadata refresh;
- native DSA/indexer pieces;
- native MoE expert path through the resident window;
- CUDA graph windowing experiments;
- scratch-cache experiments;
- device-token / token-egress experiments;
- focused projection fusion experiments.

The important status point is that these are architectural pieces, not a
validated performance win yet. The resident path still behaves like native
islands inside TRT-LLM production execution, not a fully persistent TileRT
decode engine that owns the complete 128-step loop.

### 3. Local resident fusion experiments

Recent experiments added or exercised:

- attention-tail FP4-output / BF16 gate path;
- fused post-attention gate;
- shared FP4-output SwiGLU path;
- `q_b + indexer wq_b` fused projection;
- `kv_a + indexer wk + indexer weights_proj` fused projection;
- window CUDA graph replay/caching controls.

The latest KVA/WK/WP image:

```text
localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-b22117683eb1-resident-fused-kva-wkwp-20260626T002733Z
```

The latest compiled resident `libth_common.so` used for that image:

```text
63d4075dc12d97a1a6f2c44bd514474a5e6967362a042d8dafacd8e939d83dc6
```

The KVA/WK/WP fusion compiled and the image came ready with
`TRTLLM_OPTRT_DEEPSEEK_RESIDENT_FUSED_KVA_WKWP=1`, but it was still far slower
than the frozen best. Disabling both KVA and QB fusions on that same image did
not become ready within the ablation window and should be treated as not
operationally clean until debugged.

## Current Gaps

The main gap is execution model, not one missing local fusion.

The branch still does not have a TileRT-level persistent decode engine that:

- owns an entire decode window without returning through the ordinary production
  path between small stages;
- keeps layer activations, router state, metadata, and scratch resident across
  steps with no repeated Python orchestration;
- fuses or pipelines dense projection, DSA projection/top-k/attention, MoE
  route/dispatch/expert/combine, residuals, norms, sampling, token feedback, and
  request updates as one model-specific schedule;
- produces competitive C16/C32 throughput before any micro-fusion is evaluated.

Concrete current gaps:

- Resident C16/C32 steady throughput is about 5-6x below frozen best on the
  fresh ablation shape.
- Resident cold TTFT has severe graph/startup stalls, including a 93s C16 p50 in
  the KVA-enabled run.
- Resident no-KVA/no-QB on the latest image stayed at startup probe 503 during
  the ablation window.
- Projection fusions are not enough: KVA/WK/WP and QB/WQB did not move the
  system toward best-config parity.
- The fresh ablation shape does not validate long-context HISA; use the existing
  long-context result files as context and re-run 64k+ before changing HISA.
- There is no current correctness/parity promotion gate for the full resident
  path beyond serving successful token generation.

## Recommended Next Steps

1. Keep the live deployment on `vbfuse_nvfp4_mega_cf` unless explicitly running a
   resident experiment.
2. Use a two-tier acceptance gate for resident work:
   - Fast gate: C16/C32, OSL128, ISL about 4161, same harness as the 20260628
     ablations.
   - Promotion gate: C32, OSL512, ISL about 2055; then C1-C64 if it is within
     striking distance of the frozen best.
3. Do not chase another isolated GEMM fusion until the resident path is within
   roughly 20% of frozen-best C16/C32 throughput.
4. Make the next resident milestone a true window owner:
   - one native loop over a short fixed window;
   - no per-stage Python callbacks inside the window;
   - explicit native ownership of sampling/token feedback for internal steps;
   - resident activation/metadata state that is updated in-place across steps.
5. Add a resident correctness/perf smoke gate before benchmarking:
   - confirms native window body is active;
   - confirms no Python-window fallback;
   - confirms expected native sampler path;
   - logs per-stage timing once per run;
   - fails closed if a required native boundary is missing.
6. Only after that, revisit grouped fusions:
   - KVA/WK/WP as part of a persistent attention/indexer stage;
   - qB/WQB only if it avoids a launch without adding copy/scale overhead;
   - MoE prepare/all-to-all/combine only when the resident loop owns routing and
     expert dispatch scheduling.

## Useful Commands

Restore current best:

```bash
cd /home/sjpat/TensorRT-LLM
python3 .bench_runs_claude/deploy_arm.py vbfuse_nvfp4_mega_cf
```

Quick health:

```bash
sudo /usr/local/bin/k3s kubectl -n dynamo-system get pods -o wide | rg 'NAME|topo-c1-dp2tp4-disagg-r20'
curl -sS -m 5 -w '\n%{http_code}\n' http://10.42.0.165:8000/health
```

Quick C16/C32 ablation shape:

```bash
docker cp .bench_runs_claude/e2e_sweep.py optrt-bench-claude:/tmp/e2e_sweep.py
docker exec optrt-bench-claude /opt/dynamo/venv/bin/python3 -u /tmp/e2e_sweep.py \
  --url http://10.42.0.165:8000/v1/completions \
  --concurrencies 16,32 \
  --osl 128 \
  --isl-text-reps 160 \
  --num-requests 32 \
  --label <label>
```

OSL512 C32 promotion point:

```bash
docker exec optrt-bench-claude /opt/dynamo/venv/bin/python3 -u /tmp/e2e_sweep.py \
  --url http://10.42.0.165:8000/v1/completions \
  --concurrencies 32 \
  --osl 512 \
  --isl-text-reps 79 \
  --num-requests 64 \
  --label <label>
```

Resident KVA image deploy example:

```bash
IMAGE=$(sed -n '1p' .bench_runs_claude/latest_resident_fused_kva_wkwp_image.txt)
TRTLLM_OPTRT_PERSISTENT_NATIVE_DSA_IMG="$IMAGE" \
TRTLLM_OPTRT_DEEPSEEK_RESIDENT_ATTENTION_TAIL_FP4OUT_GATE=1 \
TRTLLM_OPTRT_DEEPSEEK_RESIDENT_FUSED_QB_WQB=0 \
TRTLLM_OPTRT_DEEPSEEK_RESIDENT_FUSED_KVA_WKWP=1 \
TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_CUDA_GRAPH=1 \
TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_CUDA_GRAPH_REQUIRE_REPLAY=0 \
TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_SCRATCH_CACHE=1 \
TRTLLM_OPTRT_DEEPSEEK_RESIDENT_SHARED_FP4OUT_SWIGLU=1 \
TRTLLM_OPTRT_DEEPSEEK_RESIDENT_SHARED_FP4OUT_SWIGLU_MIN_BATCH=1 \
TRTLLM_OPTRT_DEEPSEEK_RESIDENT_FUSED_POST_ATTENTION_GATE=1 \
TRTLLM_OPTRT_NATIVE_WINDOW_STEPS=128 \
python3 .bench_runs_claude/deploy_arm.py persistent_native_window_mega_cf
```

## Landmines

- Do not enable SMC speculative decoding as part of the default path. The model
  side is not a trained/proven MTP checkpoint and the draft path has measured
  regressions.
- Do not rely on DeepEP here.
- Do not treat CUDA graph capture as the TileRT execution model. It only wraps
  the current launch structure.
- Do not use the resident path as the serving best until it beats the fast gate.
- Env-only DGD changes can roll pods, but stale configmap changes sometimes need
  explicit pod deletion. Always verify the decode pod env before benchmarking.
- First benchmark after image rollback can include warmup/capture noise. Run a
  warm repeat before deciding.
- The fresh OSL128 ablation has ISL about 4161 and does not exercise 64k+ HISA.


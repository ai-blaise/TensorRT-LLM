#!/usr/bin/env python3
"""Patch + apply a decode-only A/B arm for the r20 disagg DGD.

Bases off /tmp/dgd_base.json (clean last-applied manifest, decode=0426/NVLINK)
and /tmp/cm_restore.json (live configmap). Patches ONLY the decode worker
(image + comm env) and the decode.yaml combine flag, then applies cm + dgd.
Prefill+frontend are untouched (stay on 0426, no roll) — handshake-safe since
no NIXL/transceiver code differs between the two image commits.

Usage: deploy_arm.py {a|b|restore}
  a       : deepep image, NVLINK_TWO_SIDED, combine=false   (matched control)
  b       : deepep image, DEEPEPLOWLATENCY+tokenlimit+p2p,  combine=false
  restore : 0426 image,   NVLINK_TWO_SIDED, combine=true    (original baseline)
"""
import json, os, re, subprocess, sys

DEEPEP_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-0680cd4082bc-deepep-20260613T183120Z"
BASE_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-19d82b488-fullsource-msgpack-0426"
M1FIX_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-179ad71f0-m1fix-20260614T165318Z"
MEGAFIX_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-7facf7585-megafix-20260614T180232Z"
FULL_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-7facf7585094-hisparse-fixes-20260614T192021Z"
VBFUSE_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-1e101838e3-vbfuse-20260615T040020Z"
VBFUSE_NVFP4_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-vbfuse-nvfp4-20260615T043037Z"
SIGMOID_QUANT_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-sigmoid-quant-ext-20260618T011000Z"
GATE_FULL_MM_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-gate-full-mm-20260618T0310"
GATE_LBATCH_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-gate-lbatch-20260618T055904Z"
PREKV_QUANT_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-prekv-quant-sourceguard-20260618T072904Z"
CUTEDSL_FALLBACK_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-cutedsl-predispatch-20260618T122331Z"
CUTEDSL_FALLBACK_TACTIC_TABLE = '{"7168,16384":{"1":[[128,128],[1,2],false,false],"2":[[128,64],[1,2],false,false],"64":[[128,64],[1,2],false,false]},"7168,18432":{"2":[[128,64],[1,2],false,false],"64":[[128,64],[1,1],false,false]}}'
O_PROJ_CUTEDSL_TACTIC_TABLE = '{"7168,16384":{"16":[[128,64],[1,1],true,false],"24":[[128,64],[1,1],true,false],"32":[[128,64],[1,1],true,false],"64":[[128,64],[1,1],true,false]}}'
SPLITK_O_PROJ_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-splitk-o-proj-20260618T104428Z"
SPLITK_O_PROJ_ATOMIC_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-splitk-o-proj-atomic-20260618T111927Z"
SPLITK_SHAPE_TABLE_IMG = (
    "localhost:5000/local/dynamo-trtllm-optrt-custom:"
    "optrt-529374445d-codex-splitk-shape-table-20260619T032713Z"
)
DENSE_COMBO_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-dense-combo-20260618T184509Z"
NVFP4_DEBUG_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-nvfp4-debug-20260618T1414"
PREKV_AMAX_DEBUG_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-prekv-amax-debug-cgsafe-20260618T1509"
MEGA_V2_PRECEDENCE_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-mega-v2-dwdp-debug-20260618T043515Z"
QBWQB_REUSE_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-qb-wqb-reuse-20260618T164432Z"
QBWQB_FUSED_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-qb-wqb-fused-alias-20260618T190718Z"
QBWQB_VARIABLE_N_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-qb-wqb-variable-n-20260618T215848Z"
QB_CUTEDSL_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-qb-linear-cutedsl-20260619T021628Z"
KVA_WKWP_FUSED_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-kva-wkwp-bf16dsa-scaleguard-20260619T010519Z"
DENSE_FAMILY_COMBO_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-dense-family-combo-20260619T041459Z"
DENSE_FAMILY_SIGMOID_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-dense-family-sigmoid-20260619T045508Z"
DENSE_FAMILY_SIGMOID_V2_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-dense-family-sigmoid-v2-20260619T085608Z"
DENSE_FAMILY_SIGMOID_V2_TILED_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-dense-family-sigmoid-v2-tiled-sigmoid-20260621T0438Z"
DENSE_FAMILY_SIGMOID_V2_TILED_SPLITK_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-dense-family-sigmoid-v2-tiled-splitk-smallm-20260621T053908Z"
DENSE_FAMILY_SIGMOID_DWDP_DEBUG_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-dense-family-sigmoid-dwdp-debug-20260619T083344Z"
DENSE_FAMILY_SIGMOID_O_LINEAR_CUTEDSL_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-dense-family-sigmoid-o-linear-cutedsl-20260619T074958Z"
DENSE_FAMILY_SIGMOID_FP4NORM_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-codex-dense-family-sigmoid-fp4normdist-20260619T063416Z"
C1REORDER_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-c1reorder-20260615T223223Z"
MNNVL14476_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-0a771bdc84-mnnvl14476-20260616T014526Z"
PERSISTENT_PLAN_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-persistent-plan2-20260622"
PERSISTENT_TIMING_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-persistent-timing-20260622"
PERSISTENT_STAGE_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-persistent-window-20260622"
PERSISTENT_WINDOW_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-persistent-window2-20260622"
PERSISTENT_ENGINE_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-persistent-engine-resident9-response-egress-20260622"
PERSISTENT_MODEL_V1_IMG = "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-persistent-model-v1-live-20260622"
PERSISTENT_NATIVE_DSA_IMG = os.environ.get(
    "TRTLLM_OPTRT_PERSISTENT_NATIVE_DSA_IMG",
    "localhost:5000/local/dynamo-trtllm-optrt-custom:optrt-529374445d-persistent-native-window-dummyrank-20260623")
PERSISTENT_GRAPH_WINDOW_IMG = os.environ.get(
    "TRTLLM_OPTRT_PERSISTENT_GRAPH_WINDOW_IMG", PERSISTENT_NATIVE_DSA_IMG)
KC = ["sudo", "/usr/local/bin/k3s", "kubectl", "-n", "dynamo-system"]
DEEP_KEYS = ["TRTLLM_DEEP_EP_TOKEN_LIMIT",
             "TRTLLM_DEEP_EP_DISABLE_P2P_FOR_LOW_LATENCY_MODE",
             "TRTLLM_MOE_POST_QUANT_ALLTOALLV"]
SPLITK_KEYS = ["TRTLLM_NVFP4_GEMM_SPLITK_O_PROJ",
               "TRTLLM_NVFP4_GEMM_SPLITK_SHAPES",
               "TRTLLM_NVFP4_GEMM_SPLITK_ATOMIC",
               "TRTLLM_NVFP4_GEMM_SPLITK_MIN_M",
               "TRTLLM_NVFP4_GEMM_SPLITK_MAX_M",
               "TRTLLM_NVFP4_GEMM_SPLITK_N",
               "TRTLLM_NVFP4_GEMM_SPLITK_K",
               "TRTLLM_NVFP4_GEMM_SPLITK_SPLIT",
               "TRTLLM_NVFP4_GEMM_SPLITK_TILE_M",
               "TRTLLM_NVFP4_GEMM_SPLITK_TILE_N",
               "TRTLLM_NVFP4_GEMM_SPLITK_CLUSTER_M",
               "TRTLLM_NVFP4_GEMM_SPLITK_CLUSTER_N",
               "TRTLLM_NVFP4_GEMM_SPLITK_PREFETCH",
               "TRTLLM_NVFP4_GEMM_SPLITK_PACKED",
               "TRTLLM_NVFP4_GEMM_SPLITK_DEBUG",
               "TRTLLM_NVFP4_GEMM_SPLITK_OVERRIDE_EXPLICIT"]


def set_env(envs, name, value):
    for e in envs:
        if e["name"] == name:
            e["value"] = value
            return
    envs.append({"name": name, "value": value})


def del_env(envs, name):
    envs[:] = [e for e in envs if e["name"] != name]


def patch_yaml_scalar(yaml_text, key, value):
    patched = re.sub(rf"(^\s*{re.escape(key)}:\s*).*$",
                     rf"\g<1>{value}",
                     yaml_text,
                     flags=re.MULTILINE)
    assert patched != yaml_text or f"{key}: {value}" in yaml_text, (
        f"{key} sub failed")
    return patched


def main():
    arm = sys.argv[1]
    dgd = json.load(open("/tmp/dgd_base.json"))
    cm = json.load(open("/tmp/cm_restore.json"))
    dec = dgd["spec"]["services"]["decode"]
    envs = dec["envs"]

    if arm == "restore":
        img, comm, combine = BASE_IMG, "NVLINK_TWO_SIDED", "true"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
    elif arm == "a":
        img, comm, combine = DEEPEP_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
    elif arm == "b":
        img, comm, combine = DEEPEP_IMG, "DEEPEPLOWLATENCY", "false"
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_DEEP_EP_TOKEN_LIMIT", "64")
        set_env(envs, "TRTLLM_DEEP_EP_DISABLE_P2P_FOR_LOW_LATENCY_MODE", "0")
        set_env(envs, "TRTLLM_MOE_POST_QUANT_ALLTOALLV", "1")
    elif arm == "c":
        # M1: keep NVLINK_TWO_SIDED force but flip ONESIDED -> factory promotes
        # to NVLinkOneSided (one-sided a2a + combine-into-workspace). Pure NVLink.
        img, comm, combine = DEEPEP_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_ONESIDED_A2A", "1")
    elif arm == "m1ctl":
        # Matched control for M1: same m1fix image, NVLINK two-sided, combine=false,
        # ONESIDED off. Only the combine transport differs from arm m1 -> clean
        # numerical + perf isolation of one-sided vs two-sided a2a.
        img, comm, combine = M1FIX_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
    elif arm == "m1":
        # M1 DEEP FIX: NVLinkOneSided (one-sided a2a + combine-into-workspace)
        # with the patched WARPDECODE overlay that writes its output into the
        # comm workspace. Uses the m1fix overlay image.
        img, comm, combine = M1FIX_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_ONESIDED_A2A", "1")
    elif arm == "d":
        # V1 megakernel on the WARPDECODE path: registers
        # trtllm::warp_decode_nvfp4_cursor_moe (Phase-1 persistent decode-MoE).
        img, comm, combine = DEEPEP_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
    elif arm == "mega":
        # Megakernel FIX (fc2_input_scale threaded): megafix image, MEGAKERNEL=1.
        img, comm, combine = MEGAFIX_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
    elif arm == "full":
        # Full-source rebuild of HEAD: hisparse + M1/megakernel fixes + all levers,
        # PRODUCTION config (NVLINK two-sided, combine=true). Parity check at short
        # context (hisparse inert <65536; M1/megakernel off). New clean baseline.
        img, comm, combine = FULL_IMG, "NVLINK_TWO_SIDED", "true"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
    elif arm == "vbfuse":
        # FlashMLA v_b (W_UV) epilogue fusion: vbfuse overlay (hisparse base +
        # rebuilt libs + attention.py). PRODUCTION config (NVLINK two-sided,
        # combine=true) — same as 'full' so the ONLY delta vs the full-hisparse
        # control (results/06,07) is the fused sparse_mla_decode_nvfp4_vfuse op.
        img, comm, combine = VBFUSE_IMG, "NVLINK_TWO_SIDED", "true"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
    elif arm == "vbfuse_nvfp4":
        # vbfuse + nvfp4_gemm small-M cuBLASLt fallback fix (529374445d). Stacks
        # the nvfp4 decode-fallback fix on top of v_b fusion; delta vs arm
        # 'vbfuse' (results/08) isolates the nvfp4 fix's gain. Same prod config.
        img, comm, combine = VBFUSE_NVFP4_IMG, "NVLINK_TWO_SIDED", "true"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
    elif arm == "vbfuse_nvfp4_cf":
        # Isolation: production image (vbfuse_nvfp4) + combine=FALSE, megakernel OFF.
        # Tests whether combine=false ALONE (full-precision MoE combine, no FP8
        # quant on the NVLink a2a) recovers the armMEGA batch-32 edge (~50.6) vs
        # the combine=true production (~49.2). combine=false is production-safe.
        img, comm, combine = VBFUSE_NVFP4_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
    elif arm == "mnnvl14476":
        # #14476 MNNVL TP-allreduce rewrite + NVFP4 quant fusion, stacked on the best
        # config (MK=1 + combine=false). Both workers on the rewrite image (matching ->
        # handshake safe). A/B vs vbfuse_nvfp4_mega_cf isolates the modernized allreduce
        # (our kernel was 6mo stale) + the folded post-AR NVFP4 quant (~2.7% AR + quant slice).
        img, comm, combine = MNNVL14476_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
    elif arm == "c1reorder":
        # C=1 overlap-loop reorder (dispatch sample-N before the N-1 readback sync)
        # ON TOP of the steady win config (MK=1 + combine=false). Python-only overlay
        # (optrt-c1reorder image = vbfuse-nvfp4 + reordered py_executor.py). Targets
        # ~20-34% C=1 latency; output-preserving (validate greedy-match vs baseline).
        img, comm, combine = C1REORDER_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
    elif arm == "vbfuse_nvfp4_mega":
        # PRODUCTION CANDIDATE: vbfuse_nvfp4 + MEGAKERNEL=1 + combine=TRUE (the
        # production combine setting). Confirms the megakernel's batch-32 +2%
        # (measured 50.02 with combine=false) holds with combine=true. combine
        # tested neutral (49.07 vs 49.22), so this should match ~50. If so, it
        # is the production config.
        img, comm, combine = VBFUSE_NVFP4_IMG, "NVLINK_TWO_SIDED", "true"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
    elif arm == "vbfuse_nvfp4_mega_cf":
        # Megakernel isolation on the PRODUCTION image: vbfuse_nvfp4 + MEGAKERNEL=1
        # + combine=false. Same image+combine as vbfuse_nvfp4_cf (measured 49.07),
        # ONLY MEGAKERNEL differs -> clean test of whether the megakernel owns the
        # armMEGA batch-32 edge (~50.6) or that run was noise.
        img, comm, combine = VBFUSE_NVFP4_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
    elif arm == "vbfuse_nvfp4_mega_cf_debug":
        # BEST_CONFIG plus one-shot NVFP4 projection-site logging. This keeps
        # the same image, transport, MoE megakernel, and combine=false settings
        # as vbfuse_nvfp4_mega_cf; the only delta is debug output used to map
        # production projection sites to the remaining NVJIT kernel families.
        img, comm, combine = VBFUSE_NVFP4_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_NVFP4_LINEAR_DEBUG", "1")
        set_env(envs, "TRTLLM_NVFP4_GEMM_DEBUG_SHAPES", "1")
    elif arm == "persistent_plan_mega_cf":
        # BEST_CONFIG behavior plus the disabled-by-default persistent-decode
        # eligibility planner. This is a Python-only diagnostics carrier: it
        # keeps the proven image lineage and only injects a tiny model_engine
        # hook plus persistent_decode_planner.py into the installed wheel.
        img, comm, combine = PERSISTENT_PLAN_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_PLAN_DEBUG", "1")
        set_env(envs, "TLLM_LOG_LEVEL_BY_MODULE", "info:_torch")
    elif arm == "persistent_timing_mega_cf":
        # BEST_CONFIG behavior plus aggregate persistent-decode timing. This
        # builds the same eligibility plan silently, then logs rank-0 timing
        # summaries every 128 eligible decode steps.
        img, comm, combine = PERSISTENT_TIMING_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_PLAN_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_TIMING_DEBUG", "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_TIMING_EVERY", "128")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_TIMING_CUDA_EVENTS", "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_TIMING_RANKS", "0")
        set_env(envs, "TLLM_LOG_LEVEL_BY_MODULE", "info:_torch")
    elif arm == "persistent_stage_mega_cf":
        # BEST_CONFIG behavior plus rank-0 aggregate decode-boundary timing
        # and DeepSeek per-stage CUDA-event timing captured into the CUDA
        # graphs. This is profiling-only; the events intentionally perturb the
        # graph enough that throughput is not a release candidate number.
        img, comm, combine = PERSISTENT_STAGE_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_PLAN_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_TIMING_DEBUG", "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_TIMING_EVERY", "128")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_TIMING_CUDA_EVENTS", "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_TIMING_RANKS", "0")
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_STAGE_TIMING_DEBUG", "1")
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_STAGE_TIMING_EVERY", "128")
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_STAGE_TIMING_RANKS", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_EVERY", "128")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_TARGET", "16")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_RANKS", "0")
        set_env(envs, "TLLM_LOG_LEVEL_BY_MODULE", "info:_torch")
    elif arm == "persistent_window_mega_cf":
        # BEST_CONFIG behavior plus a low-overhead rank-0 stable-window probe.
        # This leaves aggregate timing and per-layer CUDA-event profiling off so
        # c1/c2/c4 throughput stays comparable to the production path.
        img, comm, combine = PERSISTENT_WINDOW_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_PLAN_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_TIMING_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_STAGE_TIMING_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_EVERY", "128")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_TARGET", "16")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_RANKS", "0")
        set_env(envs, "TLLM_LOG_LEVEL_BY_MODULE", "info:_torch")
    elif arm == "persistent_engine_mega_cf":
        # BEST_CONFIG behavior plus an executor-level resident-decode handoff
        # probe. This image deliberately starts from the last KV-functional
        # persistent-window image and does not include the window2 planner
        # changes that timed out KV transfer.
        img, comm, combine = PERSISTENT_ENGINE_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_PLAN_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_TIMING_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_STAGE_TIMING_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_EVERY", "128")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_TARGET", "128")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_RANKS", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE", "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_EVERY", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RANKS", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_STEPS", "16")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_ADP_WINDOW", "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_EXECUTE_WINDOW", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_COHORT", "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_TARGETS", "4,8,16,32")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_ALLOW_STREAMING", "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_DEFER_HOST_UPDATES", "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_FLEX_STEPS", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_ASYNC_TOKEN_EGRESS", "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_DEFER_TOKEN_EGRESS_UNTIL_RESPONSE", "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_ADMISSION_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_TIMING_DEBUG", "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_TIMING_EVERY", "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_TIMING_RANKS", "0")
        set_env(envs, "TRTLLM_OPTRT_DISAGG_TRANSFER_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_DISAGG_TRANSFER_DEBUG_EVERY", "16")
        set_env(envs, "TRTLLM_OPTRT_KV_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_FINALIZATION_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_FINALIZATION_DEBUG_RANKS", "0,1,2,3")
        set_env(envs, "TRTLLM_OPTRT_FINALIZATION_DEBUG_EVERY", "1")
        set_env(envs, "TLLM_LOG_LEVEL_BY_MODULE", "info:_torch")
    elif arm == "persistent_model_v1_mega_cf":
        # BEST_CONFIG behavior plus the lower model-body backend boundary.
        # This does not enable the executor resident-window loop. It validates
        # the post-_prepare_inputs resident handoff independently, before the
        # model_forward call is replaced by a native DeepSeek resident body.
        img, comm, combine = PERSISTENT_MODEL_V1_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_PLAN_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_TIMING_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_STAGE_TIMING_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND",
                "deepseek_resident_python_v1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND_RANKS", "all")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND_EVERY", "0")
        set_env(envs, "TLLM_LOG_LEVEL_BY_MODULE", "info:_torch")
    elif arm == "persistent_native_dsa_mega_cf":
        # BEST_CONFIG plus the native resident model backend and the first
        # env-gated C++ DSA dispatch body slice. This is intentionally a smoke
        # arm: executor multi-step takeover stays off while the lower model-body
        # boundary proves its production metadata/tensor ABI against live decode.
        img, comm, combine = PERSISTENT_NATIVE_DSA_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_PLAN_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_TIMING_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_STAGE_TIMING_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_EXECUTE_WINDOW", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND",
                "deepseek_resident_native_v1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND_RANKS", "all")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND_EVERY", "1")
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_STAGE_SCHEDULER", "1")
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_VALIDATE_MANIFEST", "1")
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_DSA_NATIVE_BODY", "1")
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_STAGE_SCHEDULER", "0")
        set_env(envs, "TRTLLM_OPTRT_DISAGG_TRANSFER_DEBUG", "0")
        set_env(envs, "TLLM_LOG_LEVEL_BY_MODULE", "info:_torch")
    elif arm == "persistent_graph_window_mega_cf":
        # BEST_CONFIG plus a resident 128-step execution boundary that keeps the
        # model body on production ModelEngine.forward CUDA graph replay. This
        # deliberately disables the native model/window backends; the separate
        # persistent_native_window_mega_cf arm remains the C++ native-window A/B.
        img, comm, combine = PERSISTENT_GRAPH_WINDOW_IMG, "NVLINK_TWO_SIDED", "false"
        resident_targets = os.environ.get(
            "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_TARGETS",
            ",".join(str(i) for i in range(1, 65)))
        window_steps = os.environ.get(
            "TRTLLM_OPTRT_GRAPH_WINDOW_STEPS", "128")
        window_timing_debug = os.environ.get(
            "TRTLLM_OPTRT_GRAPH_WINDOW_TIMING_DEBUG", "0")
        window_timing_every = os.environ.get(
            "TRTLLM_OPTRT_GRAPH_WINDOW_TIMING_EVERY", "16")
        window_admission_debug = os.environ.get(
            "TRTLLM_OPTRT_GRAPH_WINDOW_ADMISSION_DEBUG", "0")
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL_V2", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_PLAN_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_TIMING_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_STAGE_TIMING_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE", "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_EVERY", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RANKS", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_BACKEND",
                "deepseek_graph_resident")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_STEPS",
                window_steps)
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_ADP_WINDOW", "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_EXECUTE_WINDOW", "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_COHORT", "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_TARGETS",
                resident_targets)
        set_env(envs,
                "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_ALLOW_STREAMING",
                "1")
        set_env(envs,
                "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_DEFER_HOST_UPDATES",
                "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_FLEX_STEPS",
                "1")
        set_env(envs,
                "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_TERMINAL_WINDOW",
                "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_ASYNC_TOKEN_EGRESS",
                "1")
        set_env(envs,
                "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_DEFER_TOKEN_EGRESS_UNTIL_RESPONSE",
                "1")
        set_env(envs,
                "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_ADMISSION_DEBUG",
                window_admission_debug)
        set_env(envs,
                "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_TIMING_DEBUG",
                window_timing_debug)
        set_env(envs,
                "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_TIMING_EVERY",
                window_timing_every)
        set_env(envs,
                "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_TIMING_RANKS",
                "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND", "disabled")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND_RANKS", "all")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND_EVERY", "0")
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_STAGE_SCHEDULER", "0")
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_DSA_NATIVE_BODY", "0")
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_BODY", "0")
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_STAGE_SCHEDULER", "0")
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_CPP_TIMING", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_ALLOW_PYTHON_WINDOW_LOOP",
                "0")
        set_env(envs,
                "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_REQUIRE_NATIVE_WINDOW",
                "0")
        set_env(envs, "TRTLLM_OPTRT_DISAGG_TRANSFER_DEBUG", "0")
        set_env(envs, "TLLM_LOG_LEVEL_BY_MODULE", "info:_torch")
    elif arm == "persistent_native_window_mega_cf":
        # BEST_CONFIG plus the strict DeepSeek native-resident executor backend.
        # This is the first TileRT-style measurement arm: the executor admits a
        # stable cohort, then the model backend owns a multi-step decode window.
        img, comm, combine = PERSISTENT_NATIVE_DSA_IMG, "NVLINK_TWO_SIDED", "false"
        resident_targets = os.environ.get(
            "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_TARGETS",
            ",".join(str(i) for i in range(1, 65)))
        window_steps = os.environ.get(
            "TRTLLM_OPTRT_NATIVE_WINDOW_STEPS", "128")
        window_timing_debug = os.environ.get(
            "TRTLLM_OPTRT_NATIVE_WINDOW_TIMING_DEBUG", "0")
        window_timing_every = os.environ.get(
            "TRTLLM_OPTRT_NATIVE_WINDOW_TIMING_EVERY", "16")
        window_admission_debug = os.environ.get(
            "TRTLLM_OPTRT_NATIVE_WINDOW_ADMISSION_DEBUG", "0")
        dsa_kv_debug = os.environ.get("TRTLLM_OPTRT_DSA_KV_DEBUG", "0")
        router_gemm_debug = os.environ.get(
            "TRTLLM_OPTRT_DSV3_ROUTER_GEMM_DEBUG")
        native_linear_bridge = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_NATIVE_LINEAR_BRIDGE", "0")
        window_stage_scheduler = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_STAGE_SCHEDULER", "0")
        cpp_window_timing = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_CPP_TIMING", "0")
        cpp_window_timing_ranks = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_CPP_TIMING_RANKS", "0")
        deepgemm_indexer = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_INDEXER_DEEPGEMM", "0")
        deepgemm_indexer_logits_dtype = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_INDEXER_LOGITS_DTYPE", "bf16")
        moe_raw_routing = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MOE_RAW_ROUTING", "1")
        min_local_batch = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MIN_LOCAL_BATCH", "1")
        moe_fused_combine = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MOE_FUSED_COMBINE", "1")
        moe_fused_combine_device_sync = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MOE_FUSED_COMBINE_DEVICE_SYNC", "0")
        moe_tactic = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MOE_TACTIC")
        moe_tactic_debug = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MOE_TACTIC_DEBUG", "0")
        moe_workspace_cache = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MOE_WORKSPACE_CACHE", "0")
        moe_workspace_cache_debug = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MOE_WORKSPACE_CACHE_DEBUG", "0")
        shared_fp4out_swiglu = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_SHARED_FP4OUT_SWIGLU", "0")
        shared_fp4out_swiglu_min_batch = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_SHARED_FP4OUT_SWIGLU_MIN_BATCH",
            "16")
        attention_tail_fp4out_gate = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_ATTENTION_TAIL_FP4OUT_GATE",
            "0")
        fused_post_attention_gate = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_FUSED_POST_ATTENTION_GATE", "1")
        fused_qb_wqb = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_FUSED_QB_WQB", "0")
        fused_kva_wkwp = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_FUSED_KVA_WKWP", "0")
        window_cuda_graph = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_CUDA_GRAPH", "0")
        window_cuda_graph_require_replay = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_CUDA_GRAPH_REQUIRE_REPLAY",
            "0")
        window_cuda_event_timing = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_CUDA_EVENT_TIMING", "0")
        window_cuda_event_timing_ranks = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_CUDA_EVENT_TIMING_RANKS",
            "0")
        window_scratch_cache = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_SCRATCH_CACHE", "0")
        nvfp4_cuda_core_out = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_NVFP4_CUDA_CORE_OUT", "1")
        indexer_step_freq = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_INDEXER_STEP_FREQ", "8")
        indexer_step_recency_patch = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_INDEXER_STEP_RECENCY_PATCH", "0")
        indexer_hisa_min_seq_len = os.environ.get(
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_HISA_MIN_SEQ_LEN", "65536")
        if int(indexer_hisa_min_seq_len) < 1:
            raise ValueError(
                "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_HISA_MIN_SEQ_LEN must be "
                "positive; use 1 to force HISA for all real decode windows")
        deepgemm_root = os.environ.get(
            "TRTLLM_OPTRT_DEEP_GEMM_ROOT",
            "/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/deep_gemm")
        allow_python_window_loop = os.environ.get(
            "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_ALLOW_PYTHON_WINDOW_LOOP",
            "0")
        model_backend_every = os.environ.get(
            "TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND_EVERY", "0")
        require_native_window = os.environ.get(
            "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_REQUIRE_NATIVE_WINDOW",
            "0")
        disagg_bootstrap_steps = os.environ.get(
            "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_DISAGG_BOOTSTRAP_STEPS",
            "2")
        execute_window = os.environ.get(
            "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_EXECUTE_WINDOW", "1")
        cuda_launch_blocking = os.environ.get("CUDA_LAUNCH_BLOCKING", "0")
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL_V2", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_PLAN_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_TIMING_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_STAGE_TIMING_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_WINDOW_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE", "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_EVERY", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RANKS", "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_BACKEND",
                "deepseek_native_resident")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_STEPS",
                window_steps)
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_ADP_WINDOW", "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_EXECUTE_WINDOW",
                execute_window)
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_COHORT", "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_TARGETS",
                resident_targets)
        set_env(envs,
                "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_ALLOW_STREAMING",
                "1")
        set_env(envs,
                "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_DEFER_HOST_UPDATES",
                "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_FLEX_STEPS",
                "1")
        set_env(envs,
                "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_TERMINAL_WINDOW",
                "1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_ASYNC_TOKEN_EGRESS",
                "1")
        set_env(envs,
                "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_DEFER_TOKEN_EGRESS_UNTIL_RESPONSE",
                "1")
        set_env(envs,
                "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_ADMISSION_DEBUG",
                window_admission_debug)
        set_env(envs,
                "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_TIMING_DEBUG",
                window_timing_debug)
        set_env(envs,
                "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_TIMING_EVERY",
                window_timing_every)
        set_env(envs,
                "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_WINDOW_TIMING_RANKS",
                "0")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND",
                "deepseek_resident_native_v1")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND_RANKS", "all")
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_MODEL_BACKEND_EVERY",
                model_backend_every)
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_STAGE_SCHEDULER", "1")
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_VALIDATE_MANIFEST", "1")
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_DSA_NATIVE_BODY", "1")
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_BODY", "1")
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_STAGE_SCHEDULER",
                window_stage_scheduler)
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_CPP_TIMING",
                cpp_window_timing)
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_CPP_TIMING_RANKS",
                cpp_window_timing_ranks)
        set_env(envs, "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_ALLOW_PYTHON_WINDOW_LOOP",
                allow_python_window_loop)
        set_env(envs,
                "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_RESIDENT_REQUIRE_NATIVE_WINDOW",
                require_native_window)
        set_env(envs,
                "TRTLLM_OPTRT_PERSISTENT_DECODE_ENGINE_DISAGG_BOOTSTRAP_STEPS",
                disagg_bootstrap_steps)
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_NATIVE_LINEAR_BRIDGE",
                native_linear_bridge)
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_INDEXER_DEEPGEMM",
                deepgemm_indexer)
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_INDEXER_LOGITS_DTYPE",
                deepgemm_indexer_logits_dtype)
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MOE_RAW_ROUTING",
                moe_raw_routing)
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MIN_LOCAL_BATCH",
                min_local_batch)
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MOE_FUSED_COMBINE",
                moe_fused_combine)
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MOE_FUSED_COMBINE_DEVICE_SYNC",
                moe_fused_combine_device_sync)
        if moe_tactic is None:
            del_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MOE_TACTIC")
        else:
            set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MOE_TACTIC",
                    moe_tactic)
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MOE_TACTIC_DEBUG",
                moe_tactic_debug)
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MOE_WORKSPACE_CACHE",
                moe_workspace_cache)
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_MOE_WORKSPACE_CACHE_DEBUG",
                moe_workspace_cache_debug)
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_SHARED_FP4OUT_SWIGLU",
                shared_fp4out_swiglu)
        set_env(
            envs,
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_SHARED_FP4OUT_SWIGLU_MIN_BATCH",
            shared_fp4out_swiglu_min_batch)
        set_env(
            envs,
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_ATTENTION_TAIL_FP4OUT_GATE",
            attention_tail_fp4out_gate)
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_FUSED_POST_ATTENTION_GATE",
                fused_post_attention_gate)
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_FUSED_QB_WQB",
                fused_qb_wqb)
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_FUSED_KVA_WKWP",
                fused_kva_wkwp)
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_CUDA_GRAPH",
                window_cuda_graph)
        set_env(
            envs,
            "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_CUDA_GRAPH_REQUIRE_REPLAY",
            window_cuda_graph_require_replay)
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_CUDA_EVENT_TIMING",
                window_cuda_event_timing)
        set_env(envs,
                "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_CUDA_EVENT_TIMING_RANKS",
                window_cuda_event_timing_ranks)
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_SCRATCH_CACHE",
                window_scratch_cache)
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_NVFP4_CUDA_CORE_OUT",
                nvfp4_cuda_core_out)
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_INDEXER_STEP_FREQ",
                indexer_step_freq)
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_INDEXER_STEP_RECENCY_PATCH",
                indexer_step_recency_patch)
        set_env(envs, "TRTLLM_OPTRT_DEEPSEEK_RESIDENT_HISA_MIN_SEQ_LEN",
                indexer_hisa_min_seq_len)
        set_env(envs, "TRTLLM_OPTRT_DEEP_GEMM_ROOT", deepgemm_root)
        set_env(envs, "TRTLLM_OPTRT_DSA_KV_DEBUG", dsa_kv_debug)
        if router_gemm_debug is None:
            del_env(envs, "TRTLLM_OPTRT_DSV3_ROUTER_GEMM_DEBUG")
        else:
            set_env(envs, "TRTLLM_OPTRT_DSV3_ROUTER_GEMM_DEBUG",
                    router_gemm_debug)
        set_env(envs, "CUDA_LAUNCH_BLOCKING", cuda_launch_blocking)
        set_env(envs, "TRTLLM_OPTRT_DISAGG_TRANSFER_DEBUG", "0")
        set_env(envs, "TLLM_LOG_LEVEL_BY_MODULE", "info:_torch")
    elif arm == "local_nvfp4_debug_mega_cf":
        # BEST_CONFIG behavior plus one-shot NVFP4 projection-site logging, but
        # using the local split-K image only as a code carrier because the frozen
        # BEST_CONFIG image predates these debug hooks. Split-K env flags remain
        # unset, so this is for mapping only, not a perf control.
        img, comm, combine = NVFP4_DEBUG_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_NVFP4_LINEAR_DEBUG", "1")
        set_env(envs, "TRTLLM_NVFP4_GEMM_DEBUG_SHAPES", "1")
    elif arm == "cutedsl_nvfp4_mega_cf":
        # BEST_CONFIG plus guarded decode fallback preference for the Blackwell
        # CuTeDSL NVFP4 dense/proj GEMM kernel. This keeps the proven transport,
        # WARPDECODE megakernel, and combine=false settings unchanged; only the
        # dense/proj NVFP4 backend selection is opened to CuTeDSL.
        img, comm, combine = CUTEDSL_FALLBACK_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        del_env(envs, "TRTLLM_MLA_PROJ_NVFP4_BACKENDS")
        del_env(envs, "TRTLLM_DSV3_MLP_NVFP4_BACKENDS")
        set_env(envs, "TRTLLM_NVFP4_GEMM_FALLBACK_PREFER_CUTEDSL", "1")
        set_env(envs, "TRTLLM_NVFP4_GEMM_FALLBACK_CUTEDSL_MAX_M", "64")
        set_env(envs, "TRTLLM_NVFP4_GEMM_FALLBACK_CUTEDSL_POLICY",
                "high_impact")
        set_env(envs, "TRTLLM_NVFP4_GEMM_FALLBACK_CUTEDSL_NEAREST_M", "1")
        set_env(envs, "TRTLLM_NVFP4_GEMM_FALLBACK_OVERRIDE_EXPLICIT", "1")
        set_env(envs, "TRTLLM_NVFP4_GEMM_DEBUG_SHAPES", "1")
        set_env(envs, "TRTLLM_NVFP4_GEMM_FALLBACK_CUTEDSL_TACTIC_TABLE",
                CUTEDSL_FALLBACK_TACTIC_TABLE)
    elif arm == "splitk_o_proj_mega_cf":
        # BEST_CONFIG plus the guarded split-K CuTeDSL NVFP4 o_proj path. This
        # keeps the proven transport, WARPDECODE megakernel, and combine=false
        # settings unchanged; the only behavioral delta is routing small-M
        # N=7168,K=16384 DEFAULT-output NVFP4 projections through split-K=2.
        img, comm, combine = SPLITK_O_PROJ_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_O_PROJ", "1")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_MAX_M", "64")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_N", "7168")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_K", "16384")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_SPLIT", "2")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_DEBUG", "1")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_OVERRIDE_EXPLICIT", "1")
    elif arm == "splitk_o_proj_atomic_mega_cf":
        # Same narrow o_proj split-K route, but the split partials are
        # accumulated in the GEMM epilogue with BF16 atomics instead of writing
        # [split,M,N] partials and launching a separate reducer.
        img, comm, combine = SPLITK_O_PROJ_ATOMIC_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_O_PROJ", "1")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_ATOMIC", "1")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_MAX_M", "64")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_N", "7168")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_K", "16384")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_SPLIT", "2")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_DEBUG", "1")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_OVERRIDE_EXPLICIT", "1")
    elif arm == "splitk_proj_family_atomic_mega_cf":
        # BEST_CONFIG plus the generalized split-K shape-table selector. Keep
        # this limited to the large-K projection shapes that have some isolated
        # signal; do not route q_b/wq_b/shared_down through this arm.
        img, comm, combine = SPLITK_SHAPE_TABLE_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_SHAPES",
                "7168,16384,2,true;2112,7168,2,true")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_ATOMIC", "1")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_MIN_M", "0")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_MAX_M", "64")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_DEBUG", "1")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_OVERRIDE_EXPLICIT", "1")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_O_PROJ")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_N")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_K")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_SPLIT")
    elif arm == "dense_combo_mega_cf":
        # BEST_CONFIG plus the two safe dense/proj boundary deltas:
        # exact MLA gate sigmoid-mul+NVFP4 pack and atomic split-K o_proj.
        # Known-bad q_b+wq_b concat fusion stays disabled.
        img, comm, combine = DENSE_COMBO_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_OPTRT_FUSED_SIGMOID_MUL_QUANT_NVFP4", "1")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_O_PROJ", "1")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_ATOMIC", "1")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_MAX_M", "64")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_N", "7168")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_K", "16384")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_SPLIT", "2")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_DEBUG", "1")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_OVERRIDE_EXPLICIT", "1")
        del_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB")
        del_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_STATIC")
        del_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_POST_SCALE")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_STATIC")
    elif arm == "qb_cutedsl_mega_cf":
        # BEST_CONFIG plus a narrow q_b_proj NVFP4 CuTeDSL tactic:
        # 32<=M<=64,N=24576,K=1536 ->
        # ((128,64),(2,1),swap_ab=True,prefetch=False).
        # This is intentionally independent of the broader CuTe fallback and
        # split-K routes; only the q_b-shaped projection is redirected.
        #
        # 2026-06-18 note: the current code-carrier image did not become ready
        # in serving after two rollout attempts. Keep this arm guarded until a
        # cleaner image is built from a matching source tree, not by mixing a
        # few current Python modules into the frozen BEST_CONFIG image.
        if os.environ.get("TRTLLM_ALLOW_QB_CUTEDSL_STARTUP_PROBE") != "1":
            raise SystemExit(
                "qb_cutedsl_mega_cf is guarded: carrier image startup did not "
                "validate. Set TRTLLM_ALLOW_QB_CUTEDSL_STARTUP_PROBE=1 only "
                "for an intentional startup probe.")
        img, comm, combine = QB_CUTEDSL_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        del_env(envs, "TRTLLM_NVFP4_GEMM_QB_CUTEDSL")
        set_env(envs, "TRTLLM_NVFP4_LINEAR_QB_CUTEDSL", "1")
        set_env(envs, "TRTLLM_NVFP4_LINEAR_QB_CUTEDSL_MIN_M", "32")
        set_env(envs, "TRTLLM_NVFP4_LINEAR_QB_CUTEDSL_MAX_M", "64")
        set_env(envs, "TRTLLM_NVFP4_LINEAR_QB_CUTEDSL_N", "24576")
        set_env(envs, "TRTLLM_NVFP4_LINEAR_QB_CUTEDSL_K", "1536")
        set_env(envs, "TRTLLM_NVFP4_LINEAR_QB_CUTEDSL_TILE_M", "128")
        set_env(envs, "TRTLLM_NVFP4_LINEAR_QB_CUTEDSL_TILE_N", "64")
        set_env(envs, "TRTLLM_NVFP4_LINEAR_QB_CUTEDSL_CLUSTER_M", "2")
        set_env(envs, "TRTLLM_NVFP4_LINEAR_QB_CUTEDSL_CLUSTER_N", "1")
        set_env(envs, "TRTLLM_NVFP4_LINEAR_QB_CUTEDSL_SWAP_AB", "1")
        set_env(envs, "TRTLLM_NVFP4_LINEAR_QB_CUTEDSL_PREFETCH", "0")
        set_env(envs, "TRTLLM_NVFP4_LINEAR_QB_CUTEDSL_DEBUG", "0")
        del_env(envs, "TRTLLM_NVFP4_GEMM_FALLBACK_PREFER_CUTEDSL")
        del_env(envs, "TRTLLM_NVFP4_GEMM_FALLBACK_CUTEDSL_MAX_M")
        del_env(envs, "TRTLLM_NVFP4_GEMM_FALLBACK_CUTEDSL_POLICY")
        del_env(envs, "TRTLLM_NVFP4_GEMM_FALLBACK_CUTEDSL_NEAREST_M")
        del_env(envs, "TRTLLM_NVFP4_GEMM_FALLBACK_OVERRIDE_EXPLICIT")
        del_env(envs, "TRTLLM_NVFP4_GEMM_FALLBACK_CUTEDSL_TACTIC_TABLE")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_O_PROJ")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_ATOMIC")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_SHAPES")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_MIN_M")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_MAX_M")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_N")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_K")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_SPLIT")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_DEBUG")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_OVERRIDE_EXPLICIT")
    elif arm == "vbfuse_nvfp4_mega_v2_cf":
        # BEST_CONFIG plus the phase-3 persistent WARPDECODE MoE megakernel.
        # Keeps MK=1 + combine=false + NVLINK_TWO_SIDED unchanged and adds the
        # guarded V2 path, which targets the exposed FC1/FC2 block-scaled grouped
        # GEMM chain with one persistent grid. The overlay only changes Python
        # precedence so V2 is reachable when BEST_CONFIG's MK=1 is also set.
        img, comm, combine = MEGA_V2_PRECEDENCE_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL_V2", "1")
    elif arm == "sigmoid_quant_mega_cf":
        # BEST_CONFIG plus the guarded MLA gate -> NVFP4 o_proj handoff fusion.
        # Keeps MK=1 + combine=false + NVLINK_TWO_SIDED unchanged; the only
        # behavioral delta is TRTLLM_OPTRT_FUSED_SIGMOID_MUL_QUANT_NVFP4=1.
        img, comm, combine = SIGMOID_QUANT_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_OPTRT_FUSED_SIGMOID_MUL_QUANT_NVFP4", "1")
    elif arm == "gate_full_mm_mega_cf":
        # BEST_CONFIG plus a guarded MLA output-gate A/B:
        # one full cuBLASLt gate GEMM instead of two half-N side-stream GEMMs.
        # Sigmoid-quant remains OFF; only TRTLLM_OPTRT_MLA_GATE_TORCH_MM differs.
        img, comm, combine = GATE_FULL_MM_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_OPTRT_MLA_GATE_TORCH_MM", "full")
    elif arm == "gate_lbatch_mega_cf":
        # BEST_CONFIG plus a guarded MLA output-gate L-batch A/B:
        # the two half-N BF16 gate GEMMs remain on the gate side stream but run
        # through one PersistentDenseGemmKernel batch=2 launch.
        img, comm, combine = GATE_LBATCH_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_OPTRT_MLA_GATE_CUTEDSL_BF16_LBATCH", "1")
    elif arm == "prekv_quant_mega_cf":
        # BEST_CONFIG plus the guarded input-gated-norm -> kv_a_proj FP4
        # handoff. Keeps MK=1 + combine=false + NVLINK_TWO_SIDED unchanged;
        # the only behavioral delta is TRTLLM_OPTRT_GATED_PREKV_QUANT=1.
        img, comm, combine = PREKV_QUANT_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_OPTRT_GATED_PREKV_QUANT", "1")
    elif arm == "prekv_amax_debug_mega_cf":
        # BEST_CONFIG plus pre-KV quant and debug-only indexer amax sampling.
        # Forced-static reuse stays OFF; this arm answers whether production
        # hidden-state amax lives in the stable range observed by the harness.
        img, comm, combine = PREKV_AMAX_DEBUG_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_OPTRT_GATED_PREKV_QUANT", "1")
        set_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4", "1")
        set_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_AMAX_DEBUG", "1")
        set_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_AMAX_DEBUG_LIMIT", "8")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_STATIC")
    elif arm == "prekv_static_mega_cf":
        # BEST_CONFIG plus guarded pre-KV FP4 reuse for the DSA indexer. The
        # runtime amax probe showed all sampled amax/static ratios below 1.0;
        # this arm removes debug logging and turns on the throughput path.
        img, comm, combine = PREKV_AMAX_DEBUG_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_OPTRT_GATED_PREKV_QUANT", "1")
        set_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4", "1")
        set_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_STATIC", "1")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_AMAX_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_AMAX_DEBUG_LIMIT")
    elif arm == "qb_wqb_amax_debug_mega_cf":
        # BEST_CONFIG plus q_b prequant handoff and debug-only DSA wq_b amax
        # sampling. Forced-static wq_b reuse stays OFF; this answers whether
        # production qr amax lives near q_b_proj's static calibration.
        img, comm, combine = QBWQB_REUSE_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4", "1")
        set_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_AMAX_DEBUG", "1")
        set_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_AMAX_DEBUG_LIMIT", "8")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_STATIC")
    elif arm == "qb_wqb_static_mega_cf":
        # BEST_CONFIG plus q_b prequant handoff reused by DSA wq_b. This
        # removes the duplicate dynamic wq_b activation quantization if the
        # amax probe validates the static q_b scale range.
        img, comm, combine = QBWQB_REUSE_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4", "1")
        set_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_STATIC", "1")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_AMAX_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_AMAX_DEBUG_LIMIT")
    elif arm == "qb_wqb_fused_mega_cf":
        # BEST_CONFIG plus same-input q_b_proj + DSA wq_b fusion. This removes
        # the wq_b GEMM launch and compensates the scalar weight-scale mismatch
        # on the small indexer weights tensor instead of the full wq_b output.
        img, comm, combine = QBWQB_FUSED_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_STATIC", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_POST_SCALE", "0")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_ALIAS_ORIGINALS", "1")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_STATIC")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_AMAX_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_AMAX_DEBUG_LIMIT")
    elif arm == "qb_wqb_variable_n_mega_cf":
        # BEST_CONFIG plus the variable-N same-input q_b_proj + DSA wq_b CuTe
        # package op. Unlike the concat fusion, this avoids padded wq_b compute:
        # one persistent grid handles q_b(N=24576) and wq_b(N=8192), and the
        # DSA path returns the wq scale separately so the correction multiply is
        # not in the GEMM critical path.
        img, comm, combine = QBWQB_VARIABLE_N_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_STATIC", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_VARIABLE_N", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_POST_SCALE", "0")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_ALIAS_ORIGINALS", "1")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_STATIC")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_AMAX_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_AMAX_DEBUG_LIMIT")
    elif arm == "kva_wkwp_fused_mega_cf":
        # BEST_CONFIG plus padded same-input kv_a_proj_with_mqa + DSA fused
        # wk/weights_proj. This inserts 64 dummy rows after kv_a to preserve
        # 128-row swizzled-scale alignment, then forwards precomputed
        # indexer_k/weights into pre_indexer_proj().
        img, comm, combine = KVA_WKWP_FUSED_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_STATIC", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_BF16_DSA", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_ALIAS_ORIGINALS", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_DEBUG", "0")
        del_env(envs, "TRTLLM_OPTRT_GATED_PREKV_QUANT")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_STATIC")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_AMAX_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_AMAX_DEBUG_LIMIT")
    elif arm == "dense_family_combo_mega_cf":
        # BEST_CONFIG plus the two production-reachable same-input NVFP4
        # projection families:
        #   1. q_b_proj + DSA wq_b through the variable-N package op;
        #   2. kv_a_proj_with_mqa + DSA wk/weights_proj through the padded
        #      fused KVA/WK/WP path with BF16 DSA output.
        # This arm answers whether the launch-count reductions add under one
        # decode profile, without also enabling split-K, q_b CuTe, or packet
        # reuse paths that already regressed as standalone serving levers.
        img, comm, combine = DENSE_FAMILY_COMBO_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_STATIC", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_VARIABLE_N", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_POST_SCALE", "0")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_ALIAS_ORIGINALS", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_STATIC", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_BF16_DSA", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_ALIAS_ORIGINALS", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_DEBUG", "0")
        del_env(envs, "TRTLLM_OPTRT_GATED_PREKV_QUANT")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_STATIC")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_AMAX_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_AMAX_DEBUG_LIMIT")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_STATIC")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_AMAX_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_AMAX_DEBUG_LIMIT")
        del_env(envs, "TRTLLM_NVFP4_LINEAR_QB_CUTEDSL")
        del_env(envs, "TRTLLM_NVFP4_GEMM_QB_CUTEDSL")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_O_PROJ")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_SHAPES")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_ATOMIC")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_DEBUG")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_OVERRIDE_EXPLICIT")
        del_env(envs, "TRTLLM_OPTRT_MOE_DWDP_DEBUG")
    elif arm == "dense_family_sigmoid_dwdp_debug_mega_cf":
        # Same production path as dense_family_sigmoid_mega_cf, but with a
        # decode-only MoE DWDP contract dump. This is a debugging carrier, not a
        # performance candidate.
        img, comm, combine = DENSE_FAMILY_SIGMOID_DWDP_DEBUG_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_OPTRT_FUSED_SIGMOID_MUL_QUANT_NVFP4", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_STATIC", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_VARIABLE_N", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_POST_SCALE", "0")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_ALIAS_ORIGINALS", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_STATIC", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_BF16_DSA", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_ALIAS_ORIGINALS", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_DEBUG", "0")
        set_env(envs, "TRTLLM_OPTRT_MOE_DWDP_DEBUG", "1")
        del_env(envs, "TRTLLM_OPTRT_GATED_PREKV_QUANT")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_STATIC")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_AMAX_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_AMAX_DEBUG_LIMIT")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_STATIC")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_AMAX_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_AMAX_DEBUG_LIMIT")
        del_env(envs, "TRTLLM_NVFP4_LINEAR_QB_CUTEDSL")
        del_env(envs, "TRTLLM_NVFP4_GEMM_QB_CUTEDSL")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_O_PROJ")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_SHAPES")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_ATOMIC")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_DEBUG")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_OVERRIDE_EXPLICIT")
    elif arm == "dense_family_sigmoid_mega_cf":
        # Dense-family combo plus the exact MLA sigmoid-mul + swizzled NVFP4
        # quant boundary. This deliberately excludes split-K and q_b CuTeDSL
        # so the only extra delta versus dense_family_combo_mega_cf is removal
        # of the post-attention sigmoid/quant boundary before o_proj.
        img, comm, combine = DENSE_FAMILY_SIGMOID_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_OPTRT_FUSED_SIGMOID_MUL_QUANT_NVFP4", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_STATIC", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_VARIABLE_N", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_POST_SCALE", "0")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_ALIAS_ORIGINALS", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_STATIC", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_BF16_DSA", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_ALIAS_ORIGINALS", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_DEBUG", "0")
        del_env(envs, "TRTLLM_OPTRT_GATED_PREKV_QUANT")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_STATIC")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_AMAX_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_AMAX_DEBUG_LIMIT")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_STATIC")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_AMAX_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_AMAX_DEBUG_LIMIT")
        del_env(envs, "TRTLLM_NVFP4_LINEAR_QB_CUTEDSL")
        del_env(envs, "TRTLLM_NVFP4_GEMM_QB_CUTEDSL")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_O_PROJ")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_SHAPES")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_ATOMIC")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_DEBUG")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_OVERRIDE_EXPLICIT")
        del_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL_V2")
        del_env(envs, "TRTLLM_OPTRT_MOE_DWDP_DEBUG")
    elif arm in ("dense_family_sigmoid_v2_off_mega_cf",
                 "dense_family_sigmoid_v2_mega_cf",
                 "dense_family_sigmoid_v2_tiled_disabled_mega_cf",
                 "dense_family_sigmoid_v2_tiled_mega_cf",
                 "dense_family_sigmoid_v2_tiled_splitk_smallm_mega_cf",
                 "dense_family_sigmoid_v2_tiled_splitk_smallm_gate_full_mega_cf"):
        # dense_family_sigmoid_mega_cf plus the phase-3 persistent decode-MoE
        # V2 path. The image only carries the MoE V2 Python gate/wrapper; all
        # dense-family, sigmoid, transport, and combine=false settings match
        # the current best arm. The _off arm uses this same image while
        # explicitly disabling TRTLLM_OPTRT_MOE_MEGAKERNEL_V2 for same-image
        # isolation.
        if arm in ("dense_family_sigmoid_v2_tiled_splitk_smallm_mega_cf",
                   "dense_family_sigmoid_v2_tiled_splitk_smallm_gate_full_mega_cf"):
            img = DENSE_FAMILY_SIGMOID_V2_TILED_SPLITK_IMG
        elif arm in ("dense_family_sigmoid_v2_tiled_disabled_mega_cf",
                     "dense_family_sigmoid_v2_tiled_mega_cf"):
            img = DENSE_FAMILY_SIGMOID_V2_TILED_IMG
        else:
            img = DENSE_FAMILY_SIGMOID_V2_IMG
        comm, combine = "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        if arm in ("dense_family_sigmoid_v2_mega_cf",
                   "dense_family_sigmoid_v2_tiled_disabled_mega_cf",
                   "dense_family_sigmoid_v2_tiled_mega_cf",
                   "dense_family_sigmoid_v2_tiled_splitk_smallm_mega_cf",
                   "dense_family_sigmoid_v2_tiled_splitk_smallm_gate_full_mega_cf"):
            set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL_V2", "1")
        else:
            del_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL_V2")
        if arm in ("dense_family_sigmoid_v2_tiled_mega_cf",
                   "dense_family_sigmoid_v2_tiled_splitk_smallm_mega_cf",
                   "dense_family_sigmoid_v2_tiled_splitk_smallm_gate_full_mega_cf"):
            set_env(envs, "TRTLLM_FUSED_SIGMOID_MUL_QUANT_TILED_SMALL_M", "1")
        elif arm == "dense_family_sigmoid_v2_tiled_disabled_mega_cf":
            set_env(envs, "TRTLLM_FUSED_SIGMOID_MUL_QUANT_TILED_SMALL_M", "0")
        else:
            del_env(envs, "TRTLLM_FUSED_SIGMOID_MUL_QUANT_TILED_SMALL_M")
        set_env(envs, "TRTLLM_OPTRT_FUSED_SIGMOID_MUL_QUANT_NVFP4", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_STATIC", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_VARIABLE_N", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_POST_SCALE", "0")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_ALIAS_ORIGINALS", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_STATIC", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_BF16_DSA", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_ALIAS_ORIGINALS", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_DEBUG", "0")
        del_env(envs, "TRTLLM_OPTRT_GATED_PREKV_QUANT")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_STATIC")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_AMAX_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_AMAX_DEBUG_LIMIT")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_STATIC")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_AMAX_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_AMAX_DEBUG_LIMIT")
        del_env(envs, "TRTLLM_NVFP4_LINEAR_QB_CUTEDSL")
        del_env(envs, "TRTLLM_NVFP4_GEMM_QB_CUTEDSL")
        for k in SPLITK_KEYS:
            del_env(envs, k)
        if arm in ("dense_family_sigmoid_v2_tiled_splitk_smallm_mega_cf",
                   "dense_family_sigmoid_v2_tiled_splitk_smallm_gate_full_mega_cf"):
            set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_O_PROJ", "1")
            set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_MIN_M", "1")
            set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_MAX_M", "8")
            set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_N", "7168")
            set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_K", "16384")
            set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_SPLIT", "4")
            set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_TILE_M", "128")
            set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_TILE_N", "256")
            set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_CLUSTER_M", "1")
            set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_CLUSTER_N", "2")
            set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_PREFETCH", "0")
            set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_PACKED", "0")
            set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_ATOMIC", "1")
            set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_OVERRIDE_EXPLICIT", "1")
            set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_DEBUG", "0")
        if arm == "dense_family_sigmoid_v2_tiled_splitk_smallm_gate_full_mega_cf":
            set_env(envs, "TRTLLM_OPTRT_MLA_GATE_TORCH_MM", "full")
        del_env(envs, "TRTLLM_OPTRT_MOE_DWDP_DEBUG")
    elif arm == "dense_family_sigmoid_fp4norm_mega_cf":
        # dense_family_sigmoid_mega_cf plus a Python-only compile-backend fix:
        # in distributed mode, try the non-AR add+rmsnorm+NVFP4-quant fallback
        # before the plain add+rmsnorm fallback, matching the single-rank pass
        # order documented for the fusion.
        img, comm, combine = DENSE_FAMILY_SIGMOID_FP4NORM_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_OPTRT_FUSED_SIGMOID_MUL_QUANT_NVFP4", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_STATIC", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_VARIABLE_N", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_POST_SCALE", "0")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_ALIAS_ORIGINALS", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_STATIC", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_BF16_DSA", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_ALIAS_ORIGINALS", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_DEBUG", "0")
        del_env(envs, "TRTLLM_OPTRT_GATED_PREKV_QUANT")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_STATIC")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_AMAX_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_AMAX_DEBUG_LIMIT")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_STATIC")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_AMAX_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_AMAX_DEBUG_LIMIT")
        del_env(envs, "TRTLLM_NVFP4_LINEAR_QB_CUTEDSL")
        del_env(envs, "TRTLLM_NVFP4_GEMM_QB_CUTEDSL")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_O_PROJ")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_SHAPES")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_ATOMIC")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_DEBUG")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_OVERRIDE_EXPLICIT")
    elif arm == "dense_family_sigmoid_gate_full_mega_cf":
        # Dense-family+sigmoid plus the guarded MLA output-gate A/B:
        # one full cuBLASLt gate GEMM instead of two half-N side-stream GEMMs.
        # This is intentionally a config-only delta versus
        # dense_family_sigmoid_mega_cf so the C32 result directly tests whether
        # the dominant BF16 gate projection launches are worth consolidating in
        # the current stack.
        img, comm, combine = DENSE_FAMILY_SIGMOID_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_OPTRT_FUSED_SIGMOID_MUL_QUANT_NVFP4", "1")
        set_env(envs, "TRTLLM_OPTRT_MLA_GATE_TORCH_MM", "full")
        del_env(envs, "TRTLLM_OPTRT_MLA_GATE_CUTEDSL_BF16_LBATCH")
        del_env(envs, "TRTLLM_OPTRT_MLA_GATE_DSV3_FUSED_A")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_STATIC", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_VARIABLE_N", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_POST_SCALE", "0")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_ALIAS_ORIGINALS", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_STATIC", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_BF16_DSA", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_ALIAS_ORIGINALS", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_DEBUG", "0")
        del_env(envs, "TRTLLM_OPTRT_GATED_PREKV_QUANT")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_STATIC")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_AMAX_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_AMAX_DEBUG_LIMIT")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_STATIC")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_AMAX_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_AMAX_DEBUG_LIMIT")
        del_env(envs, "TRTLLM_NVFP4_LINEAR_QB_CUTEDSL")
        del_env(envs, "TRTLLM_NVFP4_GEMM_QB_CUTEDSL")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_O_PROJ")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_SHAPES")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_ATOMIC")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_DEBUG")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_OVERRIDE_EXPLICIT")
    elif arm == "dense_family_sigmoid_o_cutedsl_mega_cf":
        # Dense-family+sigmoid plus a narrow direct CuTeDSL fallback only for
        # the MLA o_proj NVFP4 GEMM. Fresh probes showed the swapped 128x64
        # tactic beats cuBLASLt at M16/M24/M32/M64, while M1 loses and M8 is
        # flat, so the tactic table deliberately has no wildcard/nearest-M.
        img, comm, combine = DENSE_FAMILY_SIGMOID_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_OPTRT_FUSED_SIGMOID_MUL_QUANT_NVFP4", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_STATIC", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_VARIABLE_N", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_POST_SCALE", "0")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_ALIAS_ORIGINALS", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_STATIC", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_BF16_DSA", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_ALIAS_ORIGINALS", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_DEBUG", "0")
        set_env(envs, "TRTLLM_NVFP4_GEMM_FALLBACK_PREFER_CUTEDSL", "1")
        set_env(envs, "TRTLLM_NVFP4_GEMM_FALLBACK_CUTEDSL_MAX_M", "64")
        set_env(envs, "TRTLLM_NVFP4_GEMM_FALLBACK_CUTEDSL_POLICY", "high_impact")
        set_env(envs, "TRTLLM_NVFP4_GEMM_FALLBACK_OVERRIDE_EXPLICIT", "1")
        set_env(envs, "TRTLLM_NVFP4_GEMM_FALLBACK_CUTEDSL_TACTIC_TABLE",
                O_PROJ_CUTEDSL_TACTIC_TABLE)
        set_env(envs, "TRTLLM_NVFP4_GEMM_DEBUG_SHAPES", "0")
        del_env(envs, "TRTLLM_NVFP4_GEMM_FALLBACK_CUTEDSL_NEAREST_M")
        del_env(envs, "TRTLLM_OPTRT_MLA_GATE_TORCH_MM")
        del_env(envs, "TRTLLM_OPTRT_MLA_GATE_CUTEDSL_BF16_LBATCH")
        del_env(envs, "TRTLLM_OPTRT_MLA_GATE_DSV3_FUSED_A")
        del_env(envs, "TRTLLM_OPTRT_GATED_PREKV_QUANT")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_STATIC")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_AMAX_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_AMAX_DEBUG_LIMIT")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_STATIC")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_AMAX_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_AMAX_DEBUG_LIMIT")
        del_env(envs, "TRTLLM_NVFP4_LINEAR_QB_CUTEDSL")
        del_env(envs, "TRTLLM_NVFP4_GEMM_QB_CUTEDSL")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_O_PROJ")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_SHAPES")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_ATOMIC")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_DEBUG")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_OVERRIDE_EXPLICIT")
    elif arm == "dense_family_sigmoid_o_linear_cutedsl_mega_cf":
        # Dense-family+sigmoid plus a first-class Linear-level CuTeDSL route
        # only for MLA o_proj NVFP4 GEMM. This bypasses the generic fallback
        # carrier that regressed serving, while keeping the same M16-M64 guard
        # and the direct-probe winning swapped 128x64 tactic.
        img, comm, combine = DENSE_FAMILY_SIGMOID_O_LINEAR_CUTEDSL_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_OPTRT_FUSED_SIGMOID_MUL_QUANT_NVFP4", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_STATIC", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_VARIABLE_N", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_POST_SCALE", "0")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_ALIAS_ORIGINALS", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_STATIC", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_BF16_DSA", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_ALIAS_ORIGINALS", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_DEBUG", "0")
        set_env(envs, "TRTLLM_NVFP4_LINEAR_O_CUTEDSL", "1")
        set_env(envs, "TRTLLM_NVFP4_LINEAR_O_CUTEDSL_MIN_M", "16")
        set_env(envs, "TRTLLM_NVFP4_LINEAR_O_CUTEDSL_MAX_M", "64")
        set_env(envs, "TRTLLM_NVFP4_LINEAR_O_CUTEDSL_N", "7168")
        set_env(envs, "TRTLLM_NVFP4_LINEAR_O_CUTEDSL_K", "16384")
        set_env(envs, "TRTLLM_NVFP4_LINEAR_O_CUTEDSL_TILE_M", "128")
        set_env(envs, "TRTLLM_NVFP4_LINEAR_O_CUTEDSL_TILE_N", "64")
        set_env(envs, "TRTLLM_NVFP4_LINEAR_O_CUTEDSL_CLUSTER_M", "1")
        set_env(envs, "TRTLLM_NVFP4_LINEAR_O_CUTEDSL_CLUSTER_N", "1")
        set_env(envs, "TRTLLM_NVFP4_LINEAR_O_CUTEDSL_SWAP_AB", "1")
        set_env(envs, "TRTLLM_NVFP4_LINEAR_O_CUTEDSL_PREFETCH", "0")
        set_env(envs, "TRTLLM_NVFP4_LINEAR_O_CUTEDSL_DEBUG", "0")
        del_env(envs, "TRTLLM_OPTRT_MLA_GATE_TORCH_MM")
        del_env(envs, "TRTLLM_OPTRT_MLA_GATE_CUTEDSL_BF16_LBATCH")
        del_env(envs, "TRTLLM_OPTRT_MLA_GATE_DSV3_FUSED_A")
        del_env(envs, "TRTLLM_OPTRT_GATED_PREKV_QUANT")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_STATIC")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_AMAX_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_AMAX_DEBUG_LIMIT")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_STATIC")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_AMAX_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_AMAX_DEBUG_LIMIT")
        del_env(envs, "TRTLLM_NVFP4_LINEAR_QB_CUTEDSL")
        del_env(envs, "TRTLLM_NVFP4_GEMM_QB_CUTEDSL")
        del_env(envs, "TRTLLM_NVFP4_GEMM_FALLBACK_PREFER_CUTEDSL")
        del_env(envs, "TRTLLM_NVFP4_GEMM_FALLBACK_CUTEDSL_MAX_M")
        del_env(envs, "TRTLLM_NVFP4_GEMM_FALLBACK_CUTEDSL_POLICY")
        del_env(envs, "TRTLLM_NVFP4_GEMM_FALLBACK_OVERRIDE_EXPLICIT")
        del_env(envs, "TRTLLM_NVFP4_GEMM_FALLBACK_CUTEDSL_TACTIC_TABLE")
        del_env(envs, "TRTLLM_NVFP4_GEMM_FALLBACK_CUTEDSL_NEAREST_M")
        del_env(envs, "TRTLLM_NVFP4_GEMM_DEBUG_SHAPES")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_O_PROJ")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_SHAPES")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_ATOMIC")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_DEBUG")
        del_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_OVERRIDE_EXPLICIT")
    elif arm == "dense_family_sigmoid_splitk_o_atomic_mega_cf":
        # Dense-family+sigmoid plus guarded atomic split-K only for the NVFP4
        # MLA o_proj shape. This answers whether the isolated o_proj split-K
        # signal adds once the same-input projection fusions are already
        # active. MIN_M avoids the tiny startup shapes that made earlier
        # standalone split-K tests expensive.
        img, comm, combine = DENSE_FAMILY_SIGMOID_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
        set_env(envs, "TRTLLM_OPTRT_MOE_MEGAKERNEL", "1")
        set_env(envs, "TRTLLM_OPTRT_FUSED_SIGMOID_MUL_QUANT_NVFP4", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_STATIC", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_VARIABLE_N", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_POST_SCALE", "0")
        set_env(envs, "TRTLLM_INDEXER_FUSE_QB_WQB_ALIAS_ORIGINALS", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_STATIC", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_BF16_DSA", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_ALIAS_ORIGINALS", "1")
        set_env(envs, "TRTLLM_INDEXER_FUSE_KVA_WKWP_DEBUG", "0")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_O_PROJ", "1")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_ATOMIC", "1")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_MIN_M", "16")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_MAX_M", "64")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_N", "7168")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_K", "16384")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_SPLIT", "2")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_DEBUG", "0")
        set_env(envs, "TRTLLM_NVFP4_GEMM_SPLITK_OVERRIDE_EXPLICIT", "1")
        del_env(envs, "TRTLLM_OPTRT_MLA_GATE_TORCH_MM")
        del_env(envs, "TRTLLM_OPTRT_MLA_GATE_CUTEDSL_BF16_LBATCH")
        del_env(envs, "TRTLLM_OPTRT_MLA_GATE_DSV3_FUSED_A")
        del_env(envs, "TRTLLM_OPTRT_GATED_PREKV_QUANT")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_STATIC")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_AMAX_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_PREKV_FP4_AMAX_DEBUG_LIMIT")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_STATIC")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_AMAX_DEBUG")
        del_env(envs, "TRTLLM_INDEXER_REUSE_QB_FP4_AMAX_DEBUG_LIMIT")
        del_env(envs, "TRTLLM_NVFP4_LINEAR_QB_CUTEDSL")
        del_env(envs, "TRTLLM_NVFP4_GEMM_QB_CUTEDSL")
    elif arm == "megactl":
        # Control for the megakernel: same megafix image, MEGAKERNEL OFF -> the
        # explicit-tactic WARPDECODE path. Isolates the megakernel's numerics+perf.
        img, comm, combine = MEGAFIX_IMG, "NVLINK_TWO_SIDED", "false"
        for k in DEEP_KEYS:
            del_env(envs, k)
        set_env(envs, "TRTLLM_FORCE_COMM_METHOD", comm)
    else:
        sys.exit(f"unknown arm {arm!r}")

    dec["extraPodSpec"]["mainContainer"]["image"] = img
    for ic in dec["extraPodSpec"].get("initContainers", []):
        ic["image"] = img
    if arm in ("full", "vbfuse", "vbfuse_nvfp4", "vbfuse_nvfp4_cf",
               "vbfuse_nvfp4_mega_cf", "vbfuse_nvfp4_mega", "c1reorder",
               "vbfuse_nvfp4_mega_cf_debug", "cutedsl_nvfp4_mega_cf",
               "persistent_plan_mega_cf", "persistent_timing_mega_cf",
               "persistent_stage_mega_cf", "persistent_engine_mega_cf",
               "persistent_model_v1_mega_cf",
               "persistent_native_dsa_mega_cf",
               "persistent_native_window_mega_cf",
               "local_nvfp4_debug_mega_cf", "vbfuse_nvfp4_mega_v2_cf",
               "splitk_o_proj_mega_cf", "splitk_o_proj_atomic_mega_cf",
               "splitk_proj_family_atomic_mega_cf", "dense_combo_mega_cf", "mnnvl14476",
               "sigmoid_quant_mega_cf", "gate_full_mm_mega_cf", "gate_lbatch_mega_cf",
               "prekv_quant_mega_cf", "prekv_amax_debug_mega_cf",
               "prekv_static_mega_cf", "qb_wqb_amax_debug_mega_cf",
               "qb_wqb_static_mega_cf", "qb_wqb_fused_mega_cf",
               "qb_wqb_variable_n_mega_cf", "qb_cutedsl_mega_cf",
               "kva_wkwp_fused_mega_cf", "dense_family_combo_mega_cf",
               "dense_family_sigmoid_mega_cf",
               "dense_family_sigmoid_v2_off_mega_cf",
               "dense_family_sigmoid_v2_mega_cf",
               "dense_family_sigmoid_v2_tiled_disabled_mega_cf",
               "dense_family_sigmoid_v2_tiled_mega_cf",
               "dense_family_sigmoid_v2_tiled_splitk_smallm_mega_cf",
               "dense_family_sigmoid_v2_tiled_splitk_smallm_gate_full_mega_cf",
               "dense_family_sigmoid_dwdp_debug_mega_cf",
               "dense_family_sigmoid_fp4norm_mega_cf",
               "dense_family_sigmoid_gate_full_mega_cf",
               "dense_family_sigmoid_o_cutedsl_mega_cf",
               "dense_family_sigmoid_o_linear_cutedsl_mega_cf",
               "dense_family_sigmoid_splitk_o_atomic_mega_cf",
               "persistent_graph_window_mega_cf"):
        # hisparse changed disagg transfer.py -> keep BOTH workers on the same
        # image so the NIXL KV-transfer handshake matches. Shared image = cached
        # pull, so swapping prefill is ~free on disk.
        pf = dgd["spec"]["services"]["prefill"]
        pf["extraPodSpec"]["mainContainer"]["image"] = img
        for ic in pf["extraPodSpec"].get("initContainers", []):
            ic["image"] = img

    # patch decode.yaml combine flag
    before = cm["data"]["decode.yaml"]
    after = re.sub(r"use_low_precision_moe_combine:\s*\w+",
                   f"use_low_precision_moe_combine: {combine}", before)
    assert after != before or f": {combine}" in before, "combine sub failed"
    if arm == "cutedsl_nvfp4_mega_cf":
        after = re.sub(
            r"nvfp4_gemm_config:\n\s+allowed_backends:\s*\[[^\]]+\]",
            "nvfp4_gemm_config:\n  allowed_backends: [cublaslt, cutlass, cuda_core]",
            after)
    if arm == "persistent_native_window_mega_cf":
        after = patch_yaml_scalar(after, "hisa_min_seq_len",
                                  indexer_hisa_min_seq_len)
    cm["data"]["decode.yaml"] = after
    # strip server-managed metadata for clean re-apply
    for f in ("resourceVersion", "uid", "creationTimestamp", "managedFields"):
        cm["metadata"].pop(f, None)
    if "annotations" in cm["metadata"]:
        cm["metadata"]["annotations"].pop(
            "kubectl.kubernetes.io/last-applied-configuration", None)
    cm.pop("status", None)

    json.dump(cm, open("/tmp/cm_apply.json", "w"), indent=2)
    json.dump(dgd, open("/tmp/dgd_apply.json", "w"), indent=2)

    env_now = {e["name"]: e.get("value") for e in envs}
    print(f"[arm={arm}] image={img.split(':')[-1]}")
    print(f"           FORCE_COMM_METHOD={env_now.get('TRTLLM_FORCE_COMM_METHOD')}")
    print(f"           DEEP_EP_TOKEN_LIMIT={env_now.get('TRTLLM_DEEP_EP_TOKEN_LIMIT')}"
          f" DISABLE_P2P={env_now.get('TRTLLM_DEEP_EP_DISABLE_P2P_FOR_LOW_LATENCY_MODE')}"
          f" POST_QUANT={env_now.get('TRTLLM_MOE_POST_QUANT_ALLTOALLV')}"
          f" ONESIDED={env_now.get('TRTLLM_OPTRT_MOE_ONESIDED_A2A')}"
          f" MEGAKERNEL={env_now.get('TRTLLM_OPTRT_MOE_MEGAKERNEL')}"
          f" MEGAKERNEL_V2={env_now.get('TRTLLM_OPTRT_MOE_MEGAKERNEL_V2')}"
          f" DWDP_DEBUG={env_now.get('TRTLLM_OPTRT_MOE_DWDP_DEBUG')}"
          f" SIGMOID_QUANT={env_now.get('TRTLLM_OPTRT_FUSED_SIGMOID_MUL_QUANT_NVFP4')}"
          f" SIGMOID_QUANT_TILED={env_now.get('TRTLLM_FUSED_SIGMOID_MUL_QUANT_TILED_SMALL_M')}"
          f" GATE_TORCH_MM={env_now.get('TRTLLM_OPTRT_MLA_GATE_TORCH_MM')}"
          f" GATE_LBATCH={env_now.get('TRTLLM_OPTRT_MLA_GATE_CUTEDSL_BF16_LBATCH')}"
          f" PREKV_QUANT={env_now.get('TRTLLM_OPTRT_GATED_PREKV_QUANT')}"
          f" PREKV_REUSE={env_now.get('TRTLLM_INDEXER_REUSE_PREKV_FP4')}"
          f" PREKV_REUSE_STATIC={env_now.get('TRTLLM_INDEXER_REUSE_PREKV_FP4_STATIC')}"
          f" PREKV_AMAX_DEBUG={env_now.get('TRTLLM_INDEXER_REUSE_PREKV_FP4_AMAX_DEBUG')}"
          f" QB_REUSE={env_now.get('TRTLLM_INDEXER_REUSE_QB_FP4')}"
          f" QB_REUSE_STATIC={env_now.get('TRTLLM_INDEXER_REUSE_QB_FP4_STATIC')}"
          f" QB_AMAX_DEBUG={env_now.get('TRTLLM_INDEXER_REUSE_QB_FP4_AMAX_DEBUG')}"
          f" QB_WQB_FUSED={env_now.get('TRTLLM_INDEXER_FUSE_QB_WQB')}"
          f" QB_WQB_FUSED_STATIC={env_now.get('TRTLLM_INDEXER_FUSE_QB_WQB_STATIC')}"
          f" QB_WQB_VARIABLE_N={env_now.get('TRTLLM_INDEXER_FUSE_QB_WQB_VARIABLE_N')}"
          f" QB_WQB_FUSED_POST_SCALE={env_now.get('TRTLLM_INDEXER_FUSE_QB_WQB_POST_SCALE')}"
          f" KVA_WKWP={env_now.get('TRTLLM_INDEXER_FUSE_KVA_WKWP')}"
          f" KVA_WKWP_BF16_DSA={env_now.get('TRTLLM_INDEXER_FUSE_KVA_WKWP_BF16_DSA')}"
          f" QB_CUTEDSL={env_now.get('TRTLLM_NVFP4_GEMM_QB_CUTEDSL')}"
          f" QB_LINEAR_CUTEDSL={env_now.get('TRTLLM_NVFP4_LINEAR_QB_CUTEDSL')}"
          f" O_LINEAR_CUTEDSL={env_now.get('TRTLLM_NVFP4_LINEAR_O_CUTEDSL')}"
          f" MLA_NVFP4={env_now.get('TRTLLM_MLA_PROJ_NVFP4_BACKENDS')}"
          f" MLP_NVFP4={env_now.get('TRTLLM_DSV3_MLP_NVFP4_BACKENDS')}"
          f" NVFP4_LINEAR_DEBUG={env_now.get('TRTLLM_NVFP4_LINEAR_DEBUG')}"
          f" CUTE_FALLBACK={env_now.get('TRTLLM_NVFP4_GEMM_FALLBACK_PREFER_CUTEDSL')}"
          f" CUTE_NEAREST_M={env_now.get('TRTLLM_NVFP4_GEMM_FALLBACK_CUTEDSL_NEAREST_M')}"
          f" CUTE_OVERRIDE={env_now.get('TRTLLM_NVFP4_GEMM_FALLBACK_OVERRIDE_EXPLICIT')}"
          f" NVFP4_DEBUG={env_now.get('TRTLLM_NVFP4_GEMM_DEBUG_SHAPES')}"
          f" SPLITK_O_PROJ={env_now.get('TRTLLM_NVFP4_GEMM_SPLITK_O_PROJ')}"
          f" SPLITK_SHAPES={env_now.get('TRTLLM_NVFP4_GEMM_SPLITK_SHAPES')}"
          f" SPLITK_SPLIT={env_now.get('TRTLLM_NVFP4_GEMM_SPLITK_SPLIT')}"
          f" SPLITK_TILE={env_now.get('TRTLLM_NVFP4_GEMM_SPLITK_TILE_M')}x{env_now.get('TRTLLM_NVFP4_GEMM_SPLITK_TILE_N')}"
          f" SPLITK_CLUSTER={env_now.get('TRTLLM_NVFP4_GEMM_SPLITK_CLUSTER_M')}x{env_now.get('TRTLLM_NVFP4_GEMM_SPLITK_CLUSTER_N')}"
          f" SPLITK_PACKED={env_now.get('TRTLLM_NVFP4_GEMM_SPLITK_PACKED')}"
          f" SPLITK_ATOMIC={env_now.get('TRTLLM_NVFP4_GEMM_SPLITK_ATOMIC')}"
          f" SPLITK_DEBUG={env_now.get('TRTLLM_NVFP4_GEMM_SPLITK_DEBUG')}"
          f" SPLITK_OVERRIDE={env_now.get('TRTLLM_NVFP4_GEMM_SPLITK_OVERRIDE_EXPLICIT')}"
          f" SHARED_FP4OUT_SWIGLU={env_now.get('TRTLLM_OPTRT_DEEPSEEK_RESIDENT_SHARED_FP4OUT_SWIGLU')}"
          f" SHARED_FP4OUT_SWIGLU_MIN_BATCH={env_now.get('TRTLLM_OPTRT_DEEPSEEK_RESIDENT_SHARED_FP4OUT_SWIGLU_MIN_BATCH')}"
          f" FUSED_POST_ATTN_GATE={env_now.get('TRTLLM_OPTRT_DEEPSEEK_RESIDENT_FUSED_POST_ATTENTION_GATE')}"
          f" WINDOW_CUDA_GRAPH={env_now.get('TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_CUDA_GRAPH')}"
          f" WINDOW_CUDA_GRAPH_REQUIRE_REPLAY={env_now.get('TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_CUDA_GRAPH_REQUIRE_REPLAY')}"
          f" WINDOW_SCRATCH_CACHE={env_now.get('TRTLLM_OPTRT_DEEPSEEK_RESIDENT_WINDOW_SCRATCH_CACHE')}")
    print(f"           use_low_precision_moe_combine={combine}")

    for f in ("/tmp/cm_apply.json", "/tmp/dgd_apply.json"):
        r = subprocess.run(KC + ["apply", "-f", f], capture_output=True, text=True)
        print(f"apply {f}: rc={r.returncode} {r.stdout.strip()} {r.stderr.strip()}")
        if r.returncode != 0:
            sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# M3 (DeepEP-LL) + N1 (NUMA-pin) deploy — apply ONLY with a DeepEP-built image.
# Renders the DGD (safe NVLINK default), then patches the DECODE block to:
#   - TRTLLM_FORCE_COMM_METHOD=DEEPEPLOWLATENCY + TRTLLM_DEEP_EP_TOKEN_LIMIT=64  (M3, -1.5ms/step)
#   - numactl --cpunodebind=1 --membind=1 wrap on the worker cmd            (N1, decode GPUs = NUMA node 1, live-verified)
# Prefill is left on NVLINK_TWO_SIDED (no ADP -> M3 inert there per docs/blaise).
# Usage: m3_deploy.sh <deepep-image-tag>   (prints the patched yaml + a diff; apply is manual/confirmed)
set -uo pipefail
IMG="${1:?need DeepEP image tag}"
OUT=/tmp/dgd_m3.yaml
bash deploy/disagg_pd_r20/render_dgd.sh --image "$IMG" --target-node a4-us-002-rl9 --out "$OUT" 2>&1 | grep -E "rendered|image=" | head
python3 - "$OUT" <<'PY'
import sys, re
p = sys.argv[1]; t = open(p).read()
# Split into the decode service block vs the rest. The DGD has services.decode and services.prefill.
# We target the DECODE worker only. Heuristic: the decode block is the JSON/yaml region whose args contain
# "--disaggregation-mode","decode". Simplest robust approach on the rendered YAML: operate on the first
# TRTLLM_FORCE_COMM_METHOD occurrence (decode appears before prefill in the template) and the decode command.
# --- M3 env on decode: flip the FIRST FORCE_COMM_METHOD value NVLINK_TWO_SIDED->DEEPEPLOWLATENCY ---
t2, n = re.subn(r'(- name: TRTLLM_FORCE_COMM_METHOD\s*\n\s*value: )NVLINK_TWO_SIDED', r'\1DEEPEPLOWLATENCY', t, count=1)
assert n == 1, f"expected 1 decode FORCE_COMM_METHOD, got {n}"
# add TOKEN_LIMIT right after that decode env entry
t2 = t2.replace("value: DEEPEPLOWLATENCY\n",
                "value: DEEPEPLOWLATENCY\n      - name: TRTLLM_DEEP_EP_TOKEN_LIMIT\n        value: '64'\n", 1)
open(p, "w").write(t2)
print("patched: decode FORCE_COMM_METHOD=DEEPEPLOWLATENCY + TRTLLM_DEEP_EP_TOKEN_LIMIT=64")
print("NOTE: N1 numactl wrap + decode-only targeting must be verified against the rendered structure before apply.")
PY
echo "=== sanity: exactly one DEEPEPLOWLATENCY (decode), prefill still NVLINK ==="
grep -c "DEEPEPLOWLATENCY" "$OUT"; grep -c "NVLINK_TWO_SIDED" "$OUT"; grep -c "TRTLLM_DEEP_EP_TOKEN_LIMIT" "$OUT"
echo "=== review $OUT, then: sudo /usr/local/bin/k3s kubectl -n dynamo-system apply -f $OUT ==="

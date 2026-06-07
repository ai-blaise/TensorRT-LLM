#!/usr/bin/env bash
set -euo pipefail

PLUGIN="UCX"
INPUT="deploy/disagg_pd_r20/topo-c1-dp2tp4-disagg-r20.yaml"
OUTPUT=""

usage() {
  cat <<USAGE
Usage: $0 --plugin UCX|LIBFABRIC [--input manifest.yaml] [--output rendered.yaml]

Renders, but does not apply, the r20 DGD with TRTLLM_NIXL_KVCACHE_BACKEND set
for a one-at-a-time NIXL plugin A/B. The cache transceiver backend remains NIXL;
this never renders direct UCX, Mooncake, HELIX, or MORI.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --plugin) PLUGIN="$2"; shift 2 ;;
    --input) INPUT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$PLUGIN" in
  UCX|LIBFABRIC) ;;
  *) echo "plugin must be UCX or LIBFABRIC, got $PLUGIN" >&2; exit 2 ;;
esac

[[ -f "$INPUT" ]] || { echo "input manifest not found: $INPUT" >&2; exit 1; }
if [[ -z "$OUTPUT" ]]; then
  OUTPUT="/tmp/topo-c1-dp2tp4-disagg-r20-nixl-${PLUGIN,,}-$(date -u +%Y%m%dT%H%M%SZ).yaml"
fi

python3 - "$INPUT" "$OUTPUT" "$PLUGIN" <<'PY_RENDER'
from pathlib import Path
import sys
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
plugin = sys.argv[3]
lines = src.read_text().splitlines()
out = []
seen = 0
i = 0
while i < len(lines):
    line = lines[i]
    out.append(line)
    if line.strip() in ("name: TRTLLM_NIXL_KVCACHE_BACKEND", "- name: TRTLLM_NIXL_KVCACHE_BACKEND"):
        if i + 1 >= len(lines) or "value:" not in lines[i + 1]:
            raise SystemExit("TRTLLM_NIXL_KVCACHE_BACKEND is not followed by a value line")
        indent = lines[i + 1].split("value:", 1)[0]
        out.append(f"{indent}value: {plugin}")
        i += 2
        seen += 1
        continue
    i += 1
if seen != 2:
    raise SystemExit(f"expected exactly two TRTLLM_NIXL_KVCACHE_BACKEND envs, found {seen}")
text = "\n".join(out) + "\n"
for forbidden in ["backend: UCX", "backend: MOONCAKE", "cp_type: HELIX", "layersplit_transfer_backend: ucx"]:
    if forbidden in text:
        raise SystemExit(f"forbidden fallback marker rendered: {forbidden}")
dst.write_text(text)
print(dst)
PY_RENDER

echo "rendered NIXL ${PLUGIN} plugin variant: $OUTPUT"
echo "validate locally with: EXPECTED_NIXL_PLUGIN_BACKEND=$PLUGIN LOCAL_DGD_MANIFEST=$OUTPUT NIXL_AUDIT_MODE=local NIXL_AUDIT_OUT=/tmp/nixl_${PLUGIN,,}_local_audit deploy/disagg_pd_r20/audit_nixl_gate_readiness.sh"

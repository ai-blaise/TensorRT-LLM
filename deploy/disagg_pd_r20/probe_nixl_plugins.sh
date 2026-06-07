#!/usr/bin/env bash
set -euo pipefail

DGD="${DGD:-topo-c1-dp2tp4-disagg-r20}"
KC="${KC:-sudo -E /usr/local/bin/k3s kubectl -n dynamo-system}"
COMPONENT="${COMPONENT:-prefill}"
PLUGINS="${PLUGINS:-UCX,LIBFABRIC}"
OUTPUT_DIR="${NIXL_PLUGIN_PROBE_OUT:-/tmp/nixl_plugin_probe_$(date -u +%Y%m%dT%H%M%SZ)_$$}"

mkdir -p "$OUTPUT_DIR"

pod="$($KC get pods -o name | grep "${DGD}-0-${COMPONENT}" | tail -1)"
[[ -n "$pod" ]] || { echo "could not resolve $COMPONENT pod for $DGD" >&2; exit 1; }

echo "$pod" >"$OUTPUT_DIR/pod.txt"
$KC get "$pod" -o jsonpath='{range .spec.containers[*].env[*]}{.name}{"="}{.value}{"\n"}{end}' >"$OUTPUT_DIR/env.txt" 2>/dev/null || true

$KC exec -i "$pod" -- /opt/dynamo/venv/bin/python - "$PLUGINS" <<'PY_PROBE' >"$OUTPUT_DIR/plugin_probe.json" 2>"$OUTPUT_DIR/plugin_probe.stderr"
import json
import os
import sys
import traceback

plugins = [p.strip().upper() for p in sys.argv[1].split(',') if p.strip()]
result = {
    "requested_plugins": plugins,
    "env": {
        "TRTLLM_NIXL_KVCACHE_BACKEND": os.environ.get("TRTLLM_NIXL_KVCACHE_BACKEND"),
        "TRTLLM_NIXL_ENABLE_COALESCE": os.environ.get("TRTLLM_NIXL_ENABLE_COALESCE"),
        "NIXL_PLUGIN_DIR": os.environ.get("NIXL_PLUGIN_DIR"),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
    },
    "plugins": {},
}
try:
    import nixl
    conf = nixl.nixlAgentConfig(True)
    agent = nixl.nixlAgent("r20_nixl_plugin_probe", conf)
    result["available_plugins"] = list(agent.getAvailPlugins())
    for plugin in plugins:
        entry = {}
        try:
            params, mems = agent.getPluginParams(plugin)
            entry["get_plugin_params"] = "ok"
            entry["params"] = dict(params)
            entry["memory_types"] = list(mems)
            try:
                handle = agent.createBackend(plugin, params)
                entry["create_backend"] = "ok"
                entry["backend_handle"] = int(handle)
            except Exception as exc:
                entry["create_backend"] = "fail"
                entry["create_backend_error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            entry["get_plugin_params"] = "fail"
            entry["get_plugin_params_error"] = f"{type(exc).__name__}: {exc}"
        result["plugins"][plugin] = entry
except Exception as exc:
    result["fatal"] = f"{type(exc).__name__}: {exc}"
    result["traceback"] = traceback.format_exc()
print(json.dumps(result, indent=2, sort_keys=True))
PY_PROBE

python3 - "$OUTPUT_DIR/plugin_probe.json" <<'PY_CHECK'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text())
if "fatal" in data:
    raise SystemExit(data["fatal"])
failed = {
    name: entry for name, entry in data["plugins"].items()
    if entry.get("get_plugin_params") != "ok" or entry.get("create_backend") != "ok" or "VRAM_SEG" not in entry.get("memory_types", [])
}
if failed:
    raise SystemExit(f"NIXL plugin probe failed: {failed}")
PY_CHECK

if [[ -s "$OUTPUT_DIR/plugin_probe.stderr" ]]; then
  echo "NIXL plugin probe stderr captured: $OUTPUT_DIR/plugin_probe.stderr"
fi
echo "NIXL plugin probe passed; artifacts: $OUTPUT_DIR"

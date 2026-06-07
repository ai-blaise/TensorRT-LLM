#!/usr/bin/env bash
set -euo pipefail

DGD="${DGD:-topo-c1-dp2tp4-disagg-r20}"
KC="${KC:-sudo -E /usr/local/bin/k3s kubectl -n dynamo-system}"
COMPONENT="${COMPONENT:-prefill}"
PLUGINS="${PLUGINS:-UCX,LIBFABRIC,GDS,GDS_MT}"
REQUIRE_PLUGINS="${REQUIRE_PLUGINS:-UCX,LIBFABRIC}"
OUTPUT_DIR="${NIXL_PLUGIN_MATRIX_OUT:-/tmp/nixl_plugin_matrix_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
PYTHON_BIN="${PYTHON_BIN:-/opt/dynamo/venv/bin/python}"

mkdir -p "$OUTPUT_DIR"

pod="$($KC get pods -o name | grep "${DGD}-0-${COMPONENT}" | tail -1)"
[[ -n "$pod" ]] || { echo "could not resolve $COMPONENT pod for $DGD" >&2; exit 1; }

echo "$pod" >"$OUTPUT_DIR/pod.txt"
$KC get "$pod" -o jsonpath='{range .spec.containers[*].env[*]}{.name}{"="}{.value}{"\n"}{end}' >"$OUTPUT_DIR/env.txt" 2>/dev/null || true
$KC get "$pod" -o yaml >"$OUTPUT_DIR/pod.yaml" 2>/dev/null || true

IFS=',' read -r -a plugin_array <<<"$PLUGINS"
for raw_plugin in "${plugin_array[@]}"; do
  plugin="$(echo "$raw_plugin" | tr '[:lower:]' '[:upper:]' | xargs)"
  [[ -n "$plugin" ]] || continue
  plugin_dir="$OUTPUT_DIR/$plugin"
  mkdir -p "$plugin_dir"
  set +e
  $KC exec -i "$pod" -- "$PYTHON_BIN" - "$plugin" <<'PY_PROBE' >"$plugin_dir/result.json" 2>"$plugin_dir/stderr"
import json
import os
import sys
import traceback

plugin = sys.argv[1].upper()
result = {
    "plugin": plugin,
    "env": {
        "TRTLLM_NIXL_KVCACHE_BACKEND": os.environ.get("TRTLLM_NIXL_KVCACHE_BACKEND"),
        "TRTLLM_NIXL_ENABLE_COALESCE": os.environ.get("TRTLLM_NIXL_ENABLE_COALESCE"),
        "NIXL_PLUGIN_DIR": os.environ.get("NIXL_PLUGIN_DIR"),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
    },
}
try:
    import nixl
    conf = nixl.nixlAgentConfig(True)
    agent = nixl.nixlAgent(f"r20_nixl_plugin_matrix_{plugin.lower()}", conf)
    result["available_plugins"] = list(agent.getAvailPlugins())
    params, mems = agent.getPluginParams(plugin)
    result["get_plugin_params"] = "ok"
    result["params"] = dict(params)
    result["memory_types"] = list(mems)
    handle = agent.createBackend(plugin, params)
    result["create_backend"] = "ok"
    result["backend_handle"] = int(handle)
except Exception as exc:
    if "get_plugin_params" not in result:
        result["get_plugin_params"] = "fail"
    if "create_backend" not in result:
        result["create_backend"] = "fail"
    result["error"] = f"{type(exc).__name__}: {exc}"
    result["traceback"] = traceback.format_exc()
print(json.dumps(result, indent=2, sort_keys=True))
PY_PROBE
  rc=$?
  set -e
  echo "$rc" >"$plugin_dir/exit_code.txt"
done

python3 - "$OUTPUT_DIR" "$PLUGINS" "$REQUIRE_PLUGINS" <<'PY_SUMMARY' >"$OUTPUT_DIR/summary.json"
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
plugins = [p.strip().upper() for p in sys.argv[2].split(',') if p.strip()]
required = {p.strip().upper() for p in sys.argv[3].split(',') if p.strip()}
cleanup_markers = (
    "fi_close",
    "Device or resource busy",
    "resource busy",
    "cleanup",
)
summary = {
    "artifact_dir": str(out),
    "plugins_requested": plugins,
    "required_plugins": sorted(required),
    "plugins": {},
    "status": "pass",
    "recommendations": [],
}
failures = {}
for plugin in plugins:
    plugin_dir = out / plugin
    result_path = plugin_dir / "result.json"
    stderr_path = plugin_dir / "stderr"
    exit_code_path = plugin_dir / "exit_code.txt"
    entry = {"plugin": plugin}
    if exit_code_path.exists():
        entry["exit_code"] = int(exit_code_path.read_text().strip() or "0")
    if result_path.exists() and result_path.read_text().strip():
        entry.update(json.loads(result_path.read_text()))
    else:
        entry["error"] = "missing result.json"
    stderr = stderr_path.read_text(errors="replace") if stderr_path.exists() else ""
    entry["stderr_nonempty"] = bool(stderr.strip())
    entry["cleanup_warning"] = any(marker in stderr for marker in cleanup_markers)
    entry["has_vram_seg"] = "VRAM_SEG" in entry.get("memory_types", [])
    entry["gate_candidate"] = (
        entry.get("get_plugin_params") == "ok"
        and entry.get("create_backend") == "ok"
        and entry["has_vram_seg"]
        and not entry["cleanup_warning"]
    )
    if plugin == "UCX" and entry["gate_candidate"]:
        entry["recommended_role"] = "current_nixl_gate_plugin"
    elif plugin == "LIBFABRIC" and entry.get("create_backend") == "ok" and entry["has_vram_seg"]:
        entry["recommended_role"] = "ab_candidate_cleanup_risk" if entry["cleanup_warning"] else "ab_candidate"
    elif plugin.startswith("GDS"):
        entry["recommended_role"] = "not_peer_kv_gate"
    else:
        entry["recommended_role"] = "blocked"
    summary["plugins"][plugin] = entry
    if plugin in required and not (entry.get("get_plugin_params") == "ok" and entry.get("create_backend") == "ok" and entry["has_vram_seg"]):
        failures[plugin] = entry

if failures:
    summary["status"] = "fail"
    summary["failures"] = failures

if summary["plugins"].get("UCX", {}).get("gate_candidate"):
    summary["recommendations"].append("Use NIXL with UCX plugin as the immediate gate candidate; it creates a VRAM-capable backend without cleanup warnings in this isolated probe.")
libfabric = summary["plugins"].get("LIBFABRIC", {})
if libfabric.get("create_backend") == "ok" and libfabric.get("has_vram_seg"):
    if libfabric.get("cleanup_warning"):
        summary["recommendations"].append("Keep LIBFABRIC as A/B-only until strict smoke proves lifecycle cleanup; isolated backend creation emitted cleanup stderr.")
    else:
        summary["recommendations"].append("LIBFABRIC is a valid A/B candidate after the NIXL UCX-plugin gate passes strict smoke.")
if any(name.startswith("GDS") for name in summary["plugins"]):
    summary["recommendations"].append("GDS/GDS_MT are recorded for dependency coverage only; do not promote them for peer KV transfer without an explicit supported storage/GDS design.")
print(json.dumps(summary, indent=2, sort_keys=True))
PY_SUMMARY

python3 - "$OUTPUT_DIR/summary.json" <<'PY_CHECK'
import json
import sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text())
if summary.get("status") != "pass":
    raise SystemExit(json.dumps(summary.get("failures", {}), indent=2, sort_keys=True))
PY_CHECK

echo "NIXL plugin matrix probe passed; artifacts: $OUTPUT_DIR"

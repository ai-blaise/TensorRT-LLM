#!/usr/bin/env bash
set -euo pipefail

VM_HOST="${VM_HOST:-34.106.33.128}"
VM_USER="${VM_USER:-spencergarnets}"
FOUNDRY_REPO="${FOUNDRY_REPO:-https://github.com/foundry-org/foundry.git}"
FOUNDRY_REF="${FOUNDRY_REF:-eef12012aa0f85ae6079891144797b08c282152d}"
FOUNDRY_CACHE_ROOT="${FOUNDRY_CACHE_ROOT:-/var/lib/optrt-cache/foundry}"
MODE="${MODE:-check}"
ALLOW_DOWNLOADS="${ALLOW_DOWNLOADS:-0}"
PYTHON_BIN="${PYTHON_BIN:-}"
SSH_OPTS=(
  -o BatchMode=yes
  -o IdentitiesOnly=yes
  -o UserKnownHostsFile=/Users/spencer/.ssh/google_compute_known_hosts
  -o CheckHostIP=no
)

usage() {
  cat <<'EOF_USAGE'
Usage: deploy/disagg_pd_r20/foundry_prepare.sh [options]

Prepare and audit a VM-local Foundry cache without enabling Foundry in live
TRT-LLM workers. This is a post-gate readiness harness for CUDA graph
save/load integration; it never applies a DGD, patches pods, or exports
LD_PRELOAD into production.

Options:
  --vm HOST             Target VM IP or hostname (default: $VM_HOST);
                        use local to run directly on the current VM
  --user USER           SSH user (default: $VM_USER)
  --repo URL            Foundry Git URL (default: https://github.com/foundry-org/foundry.git)
  --ref REF             Foundry ref to cache (default: audited eef12012)
  --cache-root PATH     VM cache root (default: /var/lib/optrt-cache/foundry)
  --mode MODE           check, clone, or wheel (default: check)
                        check: report prerequisites and cached state only
                        clone: clone/fetch the requested ref into cache
                        wheel: clone, then build a wheel in an isolated venv
  --allow-downloads     Allow wheel mode to fetch missing Python build deps;
                        default is cached/offline to avoid surprise downloads
  --python-bin PATH     Python >=3.10 for optional wheel mode (auto-detects
                        python3.12, python3.11, python3.10, then python3)
  -h, --help            Show this help

Environment overrides use the same names as the options.
EOF_USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vm) VM_HOST="$2"; shift 2 ;;
    --user) VM_USER="$2"; shift 2 ;;
    --repo) FOUNDRY_REPO="$2"; shift 2 ;;
    --ref) FOUNDRY_REF="$2"; shift 2 ;;
    --cache-root) FOUNDRY_CACHE_ROOT="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --allow-downloads) ALLOW_DOWNLOADS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$MODE" in
  check|clone|wheel) ;;
  *) echo "unknown --mode: $MODE" >&2; exit 2 ;;
esac

SSH_TARGET="${VM_USER}@${VM_HOST}"

read -r -d '' REMOTE_SCRIPT <<'EOS' || true
set -euo pipefail

select_python() {
  if [[ -n "$PYTHON_BIN" ]]; then
    command -v "$PYTHON_BIN"
    return
  fi
  local candidate version major minor
  for candidate in python3.12 python3.11 python3.10 python3; do
    if ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi
    version="$($candidate - <<PY
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
    major="${version%%.*}"
    minor="${version#*.}"
    if (( major > 3 || (major == 3 && minor >= 10) )); then
      command -v "$candidate"
      return
    fi
  done
  return 1
}

BUILD_PYTHON="$(select_python 2>/dev/null || true)"

sudo mkdir -p "$FOUNDRY_CACHE_ROOT"
sudo chown "$(id -u):$(id -g)" "$FOUNDRY_CACHE_ROOT"
SRC_DIR="$FOUNDRY_CACHE_ROOT/src"
WHEEL_DIR="$FOUNDRY_CACHE_ROOT/wheels"
VENV_DIR="$FOUNDRY_CACHE_ROOT/venv"
BUILD_LOG_DIR="$FOUNDRY_CACHE_ROOT/logs"
mkdir -p "$WHEEL_DIR" "$BUILD_LOG_DIR"

clone_or_update() {
  if [[ -d "$SRC_DIR/.git" ]]; then
    git -c safe.directory="$SRC_DIR" -C "$SRC_DIR" fetch --depth=1 origin "$FOUNDRY_REF"
    git -c safe.directory="$SRC_DIR" -C "$SRC_DIR" checkout --detach FETCH_HEAD
  else
    rm -rf "$SRC_DIR"
    git clone --depth=1 --branch "$FOUNDRY_REF" "$FOUNDRY_REPO" "$SRC_DIR" \
      || { rm -rf "$SRC_DIR"; git clone --depth=1 "$FOUNDRY_REPO" "$SRC_DIR"; git -c safe.directory="$SRC_DIR" -C "$SRC_DIR" fetch --depth=1 origin "$FOUNDRY_REF"; git -c safe.directory="$SRC_DIR" -C "$SRC_DIR" checkout --detach FETCH_HEAD; }
  fi
}

if [[ "$MODE" == "clone" || "$MODE" == "wheel" ]]; then
  clone_or_update
fi

foundry_sha="missing"
foundry_status="not_cloned"
if [[ -d "$SRC_DIR/.git" ]]; then
  foundry_sha="$(git -c safe.directory="$SRC_DIR" -C "$SRC_DIR" rev-parse HEAD)"
  foundry_status="cloned"
fi

cmake_version="missing"
if command -v cmake >/dev/null 2>&1; then
  cmake_version="$(cmake --version | head -1 | awk '{print $3}')"
fi
boost_status="unknown"
if command -v ldconfig >/dev/null 2>&1; then
  if ldconfig -p 2>/dev/null | grep -Eq 'libboost_(filesystem|json)'; then
    boost_status="present_in_ldconfig"
  else
    boost_status="not_found_in_ldconfig"
  fi
fi
python_bin="$BUILD_PYTHON"
torch_version="missing"
if [[ -n "$python_bin" ]]; then
  torch_version="$(env -u PYTHONPATH PYTHONNOUSERSITE=1 "$python_bin" - <<PY 2>/dev/null || true
try:
    import torch
    print(torch.__version__)
except Exception as exc:
    print(f"missing:{exc}")
PY
)"
fi

trtllm_integration="missing"
if [[ -f "$SRC_DIR/python/foundry/integration/trtllm/__init__.py" ]]; then
  if grep -qi 'placeholder\|roadmap\|coming soon' "$SRC_DIR/python/foundry/integration/trtllm/__init__.py" "$SRC_DIR/docs/trtllm/overview.md" 2>/dev/null; then
    trtllm_integration="placeholder"
  else
    trtllm_integration="present"
  fi
fi

wheel_status="not_requested"
if [[ "$MODE" == "wheel" ]]; then
  if [[ "$foundry_status" != "cloned" ]]; then
    echo "cannot build wheel: Foundry source missing" >&2
    exit 2
  fi
  build_log="$BUILD_LOG_DIR/wheel-$(date -u +%Y%m%dT%H%M%SZ).log"
  if [[ -z "$BUILD_PYTHON" ]]; then
    wheel_status="failed_no_python_ge_3_10"
  else
    env -u PYTHONPATH PYTHONNOUSERSITE=1 "$BUILD_PYTHON" -m venv "$VENV_DIR"
  fi
  # Build in isolation from production /opt/dynamo/venv; use a persistent pip cache.
  pip_env=(PIP_CACHE_DIR=/var/lib/optrt-cache/pip)
  if [[ "$ALLOW_DOWNLOADS" != 1 ]]; then
    pip_env+=(PIP_NO_INDEX=1)
  fi
  if [[ "$wheel_status" == failed_no_python_ge_3_10 ]]; then
    : >"$build_log"
  elif ! env -u PYTHONPATH PYTHONNOUSERSITE=1 "${pip_env[@]}" "$VENV_DIR/bin/python" -m pip install -U pip wheel build >"$build_log" 2>&1; then
    wheel_status="failed_missing_cached_build_deps:$build_log"
    cat <<REPORT
foundry_cache_root=$FOUNDRY_CACHE_ROOT
foundry_repo=$FOUNDRY_REPO
foundry_ref=$FOUNDRY_REF
foundry_status=$foundry_status
foundry_sha=$foundry_sha
trtllm_integration=$trtllm_integration
cmake_version=$cmake_version
boost_status=$boost_status
python=$python_bin
torch_version=$torch_version
wheel_status=$wheel_status
allow_downloads=$ALLOW_DOWNLOADS
ld_preload_modified=0
live_workload_modified=0
REPORT
    exit 0
  fi
  if [[ "$wheel_status" == failed_no_python_ge_3_10 ]]; then
    :
  elif env -u PYTHONPATH PYTHONNOUSERSITE=1 "${pip_env[@]}" "$VENV_DIR/bin/python" -m build --wheel --outdir "$WHEEL_DIR" "$SRC_DIR" >>"$build_log" 2>&1; then
    wheel_status="built:$(ls -1t "$WHEEL_DIR"/*.whl | head -1)"
  else
    wheel_status="failed:$build_log"
  fi
fi

cat <<REPORT
foundry_cache_root=$FOUNDRY_CACHE_ROOT
foundry_repo=$FOUNDRY_REPO
foundry_ref=$FOUNDRY_REF
foundry_status=$foundry_status
foundry_sha=$foundry_sha
trtllm_integration=$trtllm_integration
cmake_version=$cmake_version
boost_status=$boost_status
python=$python_bin
torch_version=$torch_version
wheel_status=$wheel_status
allow_downloads=$ALLOW_DOWNLOADS
ld_preload_modified=0
live_workload_modified=0
REPORT
EOS

if [[ "$VM_HOST" == "local" ]]; then
  FOUNDRY_REPO="$FOUNDRY_REPO" FOUNDRY_REF="$FOUNDRY_REF" \
    FOUNDRY_CACHE_ROOT="$FOUNDRY_CACHE_ROOT" MODE="$MODE" \
    ALLOW_DOWNLOADS="$ALLOW_DOWNLOADS" PYTHON_BIN="$PYTHON_BIN" bash -s <<<"$REMOTE_SCRIPT"
else
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
    "FOUNDRY_REPO='"$FOUNDRY_REPO"' FOUNDRY_REF='"$FOUNDRY_REF"' FOUNDRY_CACHE_ROOT='"$FOUNDRY_CACHE_ROOT"' MODE='"$MODE"' ALLOW_DOWNLOADS='"$ALLOW_DOWNLOADS"' PYTHON_BIN='"$PYTHON_BIN"' bash -s" \
    <<<"$REMOTE_SCRIPT"
fi

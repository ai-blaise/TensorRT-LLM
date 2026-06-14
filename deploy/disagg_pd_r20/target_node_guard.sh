#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

optrt_r20_reject_disallowed_target_node() {
  local target_node="${1:-}"
  case "$target_node" in
    a4-us-002-rl9|a4-us-002-rl9.*)
      cat >&2 <<'EOF'
r20 target-node guard failed: a4-us-002-rl9 is out of rotation for this run.
Use a4-us-001-rl9 for active preflight/build/cache/deploy work unless the
operator explicitly changes the node constraint and updates this guard.
EOF
      return 2
      ;;
  esac
  return 0
}

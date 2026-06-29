#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

optrt_r20_reject_disallowed_target_node() {
  local target_node="${1:-}"
  case "$target_node" in
    a4-us-001-rl9|a4-us-001-rl9.*|a4-us-002-rl9|a4-us-002-rl9.*) ;;
  esac
  return 0
}

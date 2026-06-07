#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light dense MLA KVarN production gate audit.

This audit intentionally avoids importing TRT-LLM/torch/pytest so it can run on
warm deployment hosts with partial Python environments. It checks the immediate
production gate only: dense MLA latent KV is 2-bit KVarN, Indexer K remains
FP4/HISA and not KVarN, LayerSplit owner-local prefill routes through NIXL, and
KVarN owner-local non-owner layers no-op instead of indexing missing
``layer_offsets`` entries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency availability varies by image
    raise SystemExit("PyYAML is required for deploy YAML audit") from exc


FORBIDDEN_MANIFEST_STRINGS = (
    "backend: UCX",
    "layersplit_transfer_backend: ucx",
    "mla_latent_kv_dtype: auto",
    "indexer_k_dtype: kvarn",
    "cp_type: HELIX",
)
REQUIRED_SMOKE_STRINGS = (
    "layersplit_transfer_backend: nixl",
    "layersplit_owner_local_alloc: true",
    "backend: NIXL",
    "mla_latent_kv_dtype: kvarn_k2v2",
    "UCX fallback is present",
    "Indexer K was incorrectly routed to KVarN",
)
REQUIRED_DSA_GUARDS = (
    "if layer_idx not in self.layer_offsets:\n            return None",
    "pool = mgr.get_kvarn_latent_pool(self.layer_idx)\n        if pool is None:\n            return",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def audit(repo_root: Path) -> dict[str, Any]:
    deploy = repo_root / "deploy" / "disagg_pd_r20"
    prefill = _load_yaml(deploy / "prefill.yaml")
    decode = _load_yaml(deploy / "decode.yaml")
    manifest = (deploy / "topo-c1-dp2tp4-disagg-r20.yaml").read_text()
    smoke = (deploy / "smoke_request_pinning.sh").read_text()
    dsa = (repo_root / "tensorrt_llm" / "_torch" / "attention_backend" /
           "sparse" / "dsa.py").read_text()
    docs = (repo_root / "docs" / "blaise" / "kvarn.md").read_text()

    failures: list[str] = []
    for name, cfg in (("prefill", prefill), ("decode", decode)):
        sparse = cfg["sparse_attention_config"]
        if cfg["cache_transceiver_config"]["backend"] != "NIXL":
            failures.append(f"{name}: cache_transceiver_config.backend is not NIXL")
        if sparse["mla_latent_kv_dtype"] != "kvarn_k2v2":
            failures.append(f"{name}: dense MLA KVarN dtype is not kvarn_k2v2")
        if sparse["mla_latent_kv_amortize"] is not True:
            failures.append(f"{name}: dense MLA KVarN amortize is not true")
        if sparse["indexer_k_dtype"] != "fp4":
            failures.append(f"{name}: Indexer K dtype is not fp4")
        if str(sparse["indexer_k_dtype"]).lower().startswith("kvarn"):
            failures.append(f"{name}: Indexer K was routed to KVarN")

    if prefill["tensor_parallel_size"] != 2 or prefill["context_parallel_size"] != 2:
        failures.append("prefill topology is not TP2xCP2")
    if prefill["cp_config"]["cp_type"] != "LAYERSPLIT":
        failures.append("prefill cp_config.cp_type is not LAYERSPLIT")
    if prefill["sparse_attention_config"].get("layersplit_transfer_backend") != "nixl":
        failures.append("LayerSplit transfer backend is not nixl")
    if prefill["sparse_attention_config"].get("layersplit_owner_local_alloc") is not True:
        failures.append("LayerSplit owner-local allocation is not enabled")
    if decode["tensor_parallel_size"] != 4 or decode["context_parallel_size"] != 1:
        failures.append("decode topology is not TP4xCP1")

    for forbidden in FORBIDDEN_MANIFEST_STRINGS:
        if forbidden in manifest:
            failures.append(f"manifest contains forbidden fallback string: {forbidden}")
    for required in REQUIRED_SMOKE_STRINGS:
        if required not in smoke:
            failures.append(f"smoke script missing fail-closed guard: {required}")
    for guard in REQUIRED_DSA_GUARDS:
        if guard not in dsa:
            failures.append(f"dsa.py missing dense KVarN owner-local guard: {guard}")
    if "KVarN dense MLA side-pool is owner-only" not in docs:
        failures.append("docs/blaise/kvarn.md missing owner-only side-pool note")

    return {
        "ok": not failures,
        "failures": failures,
        "dense_mla_dtype": prefill["sparse_attention_config"]["mla_latent_kv_dtype"],
        "dense_mla_amortize": prefill["sparse_attention_config"]["mla_latent_kv_amortize"],
        "indexer_k_dtype": prefill["sparse_attention_config"]["indexer_k_dtype"],
        "transport": prefill["cache_transceiver_config"]["backend"],
        "layersplit_transfer_backend": prefill["sparse_attention_config"].get("layersplit_transfer_backend"),
        "layersplit_owner_local_alloc": prefill["sparse_attention_config"].get("layersplit_owner_local_alloc"),
        "prefill_topology": "TP2xCP2",
        "decode_topology": "TP4xCP1",
        "gqa_kvarn_promoted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[2])
    parser.add_argument("--json-output")
    args = parser.parse_args()
    result = audit(Path(args.repo_root))
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
